//! Seq-major (BSNH) vs head-major (BNSH) decode-KV read microbenchmark.
//!
//! Models the single-token decode attention KV traffic: for each kv-head we
//! stream K[0..L] and V[0..L] once (fused online-softmax dot + weighted V
//! accumulate), which is exactly the decode-step KV read pattern. The two
//! kernels differ ONLY in how successive tokens are addressed:
//!
//!   head-major (BNSH): k[(head*L + t)*head_dim + d]   -- tokens contiguous
//!   seq-major  (BSNH): k[t*(heads*head_dim) + head*head_dim + d] -- strided
//!
//! Total bytes moved is identical; only spatial locality / cache-line
//! utilisation differs. Runs are interleaved (H,S,H,S,...) to defeat the
//! extreme run-to-run variance on this box, and we report min + median GB/s.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

const SRC: &str = r#"
#include <cuda_fp16.h>

// One warp (32 lanes) per (head). Each lane owns ceil(head_dim/32) dims (lanes
// with index >= head_dim are masked, so head_dim < 32 is supported to model the
// small contiguous runs of quantized KV). Fused online-softmax single pass over
// L tokens: read K row, warp-shuffle dot with query, softmax update, read V row,
// accumulate. Reads K and V exactly once each -> faithful decode KV traffic,
// warp-shuffle reduction (matches gqa_decode_fp16). Only the token address
// stride differs by layout. `head_dim` here is the number of contiguous fp16
// elements per token run; a run of B bytes is modelled with head_dim = B/2.

#define DEF_KERNEL(NAME, BASE)                                               \
extern "C" __global__ void NAME(                                            \
    const __half* __restrict__ k, const __half* __restrict__ v,             \
    const __half* __restrict__ q, float* __restrict__ out,                  \
    int heads, int L, int head_dim)                                         \
{                                                                            \
    const int group = blockIdx.x / heads;                                    \
    const int head = blockIdx.x % heads;                                     \
    const int lane = threadIdx.x;                                            \
    const int per = (head_dim + 31) / 32;                                     \
    const long gbase = (long)group * L * heads * head_dim;                    \
    float qd[8];                                                             \
    for (int i = 0; i < per; ++i) {                                          \
        int d = lane + i * 32;                                               \
        qd[i] = (d < head_dim) ? __half2float(q[head * head_dim + d]) : 0.f; \
    }                                                                        \
    float m = -1e30f, l = 0.f;                                               \
    float acc[8];                                                            \
    for (int i = 0; i < per; ++i) acc[i] = 0.f;                             \
    for (int t = 0; t < L; ++t) {                                            \
        long base = gbase + (BASE);                                          \
        float dot = 0.f;                                                     \
        for (int i = 0; i < per; ++i) {                                      \
            int d = lane + i * 32;                                           \
            if (d < head_dim) dot += qd[i] * __half2float(k[base + d]);      \
        }                                                                    \
        for (int s = 16; s > 0; s >>= 1)                                     \
            dot += __shfl_xor_sync(0xffffffffu, dot, s);                     \
        float score = dot * 0.125f;                                          \
        float nm = fmaxf(m, score);                                          \
        float corr = __expf(m - nm);                                         \
        float p = __expf(score - nm);                                        \
        l = l * corr + p;                                                    \
        for (int i = 0; i < per; ++i) {                                      \
            int d = lane + i * 32;                                           \
            if (d < head_dim)                                                \
                acc[i] = acc[i] * corr + p * __half2float(v[base + d]);      \
        }                                                                    \
        m = nm;                                                              \
    }                                                                        \
    for (int i = 0; i < per; ++i) {                                          \
        int d = lane + i * 32;                                               \
        if (d < head_dim)                                                    \
            out[(group * heads + head) * head_dim + d] = acc[i] / (l + 1e-9f); \
    }                                                                        \
}

DEF_KERNEL(read_head_major, ((long)(head * L + t) * head_dim))
DEF_KERNEL(read_seq_major,  ((long)t * heads * head_dim + head * head_dim))
"#;

fn compile(ctx: &Arc<CudaContext>) -> Arc<CudaModule> {
    use cudarc::driver::sys::CUdevice_attribute as A;
    let major = ctx
        .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap();
    let minor = ctx
        .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap();
    println!("# device compute capability sm_{major}{minor}");
    // Compile straight to cubin (SASS) for the exact SM. The installed CUDA 13
    // NVRTC emits a PTX ISA newer than the driver JITs, so we mirror the repo's
    // cubin fallback (runtime.rs load_nvrtc_cubin) via nvrtcGetCUBIN.
    let arch = format!("sm_{major}{minor}");
    let source = std::ffi::CString::new(SRC).unwrap();
    let name = std::ffi::CString::new("bench.cu").unwrap();
    let program =
        cudarc::nvrtc::result::create_program(source.as_c_str(), Some(name.as_c_str())).unwrap();
    let mut options: Vec<String> = nvrtc_includes()
        .iter()
        .map(|p| format!("--include-path={p}"))
        .collect();
    options.push(format!("--gpu-architecture={arch}"));
    options.push("--use_fast_math".into());
    let res = unsafe { cudarc::nvrtc::result::compile_program(program, &options) };
    if res.is_err() {
        let log = unsafe { cudarc::nvrtc::result::get_program_log(program) }
            .ok()
            .and_then(|l| String::from_utf8(l.into_iter().map(|c| c as u8).collect()).ok())
            .unwrap_or_default();
        panic!("nvrtc compile failed: {res:?}\n{log}");
    }
    let mut size = 0usize;
    unsafe { cudarc::nvrtc::sys::nvrtcGetCUBINSize(program, &mut size) }
        .result()
        .unwrap();
    let mut image = vec![0u8; size];
    unsafe { cudarc::nvrtc::sys::nvrtcGetCUBIN(program, image.as_mut_ptr().cast()) }
        .result()
        .unwrap();
    unsafe { cudarc::nvrtc::result::destroy_program(program) }.unwrap();
    ctx.load_module(Ptx::from_binary(image))
        .expect("load cubin")
}

fn nvrtc_includes() -> Vec<String> {
    // Point NVRTC at the cuda_fp16.h shipped in the anaconda site-packages wheels.
    let sp = r"C:\Users\justinchu\AppData\Local\anaconda3\Lib\site-packages";
    let mut v = Vec::new();
    for c in [
        "nvidia\\cu13\\include",
        "nvidia\\cuda_runtime\\include",
        "nvidia\\cuda_nvrtc\\include",
    ] {
        let p = std::path::Path::new(sp).join(c);
        if p.exists() {
            v.push(p.to_string_lossy().into_owned());
        }
    }
    v
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = compile(&ctx);
    let f_head = module.load_function("read_head_major").unwrap();
    let f_seq = module.load_function("read_seq_major").unwrap();

    // Realistic decode configs: (kv_heads_per_layer, head_dim, layers).
    // head_dim is the count of contiguous fp16 elements per token run; a run of
    // B bytes is modelled with head_dim = B/2, so smaller head_dim also stands in
    // for QUANTIZED KV (fp8/int4) whose contiguous run shrinks below 64 B:
    //   head_dim=16 -> 32 B run  (int4 hd64, or fp8 hd32)  -- worst quantized case
    //   head_dim=32 -> 64 B run  (fp16 hd32, fp8 hd64, int4 hd128)
    //   head_dim=64 -> 128 B run (fp16 hd64) ...
    let configs: &[(&str, i32, i32, i32)] = &[
        ("q-run32B (int4)", 8, 16, 40),
        ("dhd32-kv8", 8, 32, 40),
        ("qwen0.5b", 2, 64, 24),
        ("kv8-hd64", 8, 64, 40),
        ("qwen14b", 8, 128, 48),
    ];
    let reps = 40usize;
    let warmup = 8usize;

    println!("config, layout, head_dim, kv_heads, layers, L, blocks, bytes_MiB, min_us, med_us, min_GBs, med_GBs, contig_run_B, seq/head_med_ratio");
    for &(name, kv_heads, head_dim, layers) in configs {
        for &l in &[512i32, 2048, 8192, 32768] {
            let blocks = (layers * kv_heads) as u32;
            let elems =
                (layers as usize) * (kv_heads as usize) * (l as usize) * (head_dim as usize);
            let k = stream.alloc_zeros::<half::f16>(elems).unwrap();
            let v = stream.alloc_zeros::<half::f16>(elems).unwrap();
            let q = stream
                .alloc_zeros::<half::f16>((kv_heads as usize) * (head_dim as usize))
                .unwrap();
            let out = stream
                .alloc_zeros::<f32>((layers * kv_heads) as usize * (head_dim as usize))
                .unwrap();

            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            // bytes moved = K + V = 2 * layers * kv_heads * L * head_dim * sizeof(f16)
            let bytes = 2.0 * elems as f64 * 2.0;
            let bytes_mib = bytes / (1024.0 * 1024.0);
            let contig_run = head_dim as f64 * 2.0;

            let run = |func: &cudarc::driver::CudaFunction| -> f64 {
                let t0 = std::time::Instant::now();
                let mut b = stream.launch_builder(func);
                b.arg(&k)
                    .arg(&v)
                    .arg(&q)
                    .arg(&out)
                    .arg(&kv_heads)
                    .arg(&l)
                    .arg(&head_dim);
                unsafe { b.launch(cfg) }.unwrap();
                stream.synchronize().unwrap();
                t0.elapsed().as_secs_f64() * 1e6 // us
            };

            for _ in 0..warmup {
                run(&f_head);
                run(&f_seq);
            }
            let mut h_us = Vec::new();
            let mut s_us = Vec::new();
            for _ in 0..reps {
                h_us.push(run(&f_head));
                s_us.push(run(&f_seq));
            }
            let stat = |mut xs: Vec<f64>| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let min = xs[0];
                let med = xs[xs.len() / 2];
                (min, med)
            };
            let (hmin, hmed) = stat(h_us.clone());
            let (smin, smed) = stat(s_us.clone());
            let gbs = |us: f64| bytes / (us * 1e-6) / 1e9;
            let ratio = smed / hmed;
            println!(
                "{name}, head-major, {head_dim}, {kv_heads}, {layers}, {l}, {blocks}, {bytes_mib:.1}, {hmin:.1}, {hmed:.1}, {:.1}, {:.1}, {contig_run:.0}, -",
                gbs(hmin), gbs(hmed)
            );
            println!(
                "{name}, seq-major,  {head_dim}, {kv_heads}, {layers}, {l}, {blocks}, {bytes_mib:.1}, {smin:.1}, {smed:.1}, {:.1}, {:.1}, {contig_run:.0}, {ratio:.3}",
                gbs(smin), gbs(smed)
            );
        }
    }
}
