//! Token-major-across-all-layers KV read microbenchmark — the TLB-pressure probe.
//!
//! Background (measured, merged): the KV committed-bytes floor under a flat
//! contiguous VA is `objects x granule`, decided by how densely the live bytes
//! sit in virtual address space, i.e. by layout. Head-major floors at
//! `layers x 2 x kv_heads` granules, seq-major (BSNH, landed #782) at
//! `layers x 2`, and **token-major across all layers** — one reservation with
//! every layer's K and V interleaved by token — floors at **one granule per
//! sequence** (~2 MiB). See `docs/memory/TOKEN_MAJOR_KV_INVESTIGATION.md`.
//!
//! The seq-major investigation (#778) already measured that the *stride
//! magnitude itself* is free: reading one head strided by `kv_heads x head_dim`
//! (~2 KB) is within +/-2% of the contiguous head-major read, because each token
//! still contributes a contiguous `head_dim x dtype` run and L2 recovers the
//! co-read neighbouring head. The OPEN risk it left for token-major is **not
//! coalescing — it is TLB pressure**: token-major pushes the per-token read
//! stride to the *full per-token KV size* (~192 KB for qwen14b), so each head's
//! 256-byte run lands on a different page. This bench settles that.
//!
//! Method: one warp per (plane, head) streams its `L x head_dim` KV run once
//! (fused query-dot reduction so every load is live — faithful decode KV
//! traffic). The buffer holds the *identical* bytes for every layout; only the
//! token stride changes, controlled by `planes_per_group` (G):
//!
//!   G = 1        -> stride = kv_heads*head_dim         (~2 KB, BSNH per-layer)
//!   G = P/k      -> stride = (P/k)*kv_heads*head_dim   (intermediate)
//!   G = P (all)  -> stride = P*kv_heads*head_dim       (~192 KB, token-major)
//!
//! where P = layers*2 planes. Total bytes moved and the contiguous per-token run
//! (`head_dim x dtype`) are IDENTICAL across G; only the page spread differs, so
//! any BW delta is TLB reach, isolated. Runs are interleaved across G to defeat
//! this box's extreme throughput variance; min + median GB/s reported. If BW is
//! flat across G at the largest working sets, 2 MiB device pages cover the
//! 192 KB stride and token-major is essentially free on the read path.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

const SRC: &str = r#"
#include <cuda_fp16.h>

// One warp (32 lanes) per (plane, head). Streams this (plane, head)'s L tokens,
// each a contiguous head_dim run, striding successive tokens by
// G*kv_heads*head_dim elements. A group of G planes is interleaved by token:
//   buffer[group][t][plane_in_group][head][dim],  group = plane / G.
// The contiguous run per token is head_dim*sizeof(half); only the token stride
// changes with G, so this isolates TLB / page-locality from cache-line
// coalescing. Fused query-dot keeps every load live (no DCE), matching the
// decode read pattern and the warp-shuffle reduction in gqa_decode_fp16.
extern "C" __global__ void read_kv(
    const __half* __restrict__ kv, const __half* __restrict__ q,
    float* __restrict__ out,
    int kv_heads, int L, int head_dim, int G)
{
    const int plane = blockIdx.x / kv_heads;
    const int head  = blockIdx.x % kv_heads;
    const int lane  = threadIdx.x;
    const int per   = (head_dim + 31) / 32;
    const int group = plane / G;
    const int pin   = plane % G;
    // 64-bit indexing throughout: on Windows (LLP64) device `long` is 32-bit,
    // and the token-major buffer exceeds 2^31 elements at L=32768, so use
    // `long long` to avoid signed-int32 offset overflow -> illegal address.
    const long long stride = (long long)G * kv_heads * head_dim;
    const long long base = (long long)group * L * stride
                    + (long long)pin * kv_heads * head_dim
                    + (long long)head * head_dim;
    float qd[8];
    for (int i = 0; i < per; ++i) {
        int d = lane + i * 32;
        qd[i] = (d < head_dim) ? __half2float(q[head * head_dim + d]) : 0.f;
    }
    float acc = 0.f;
    for (int t = 0; t < L; ++t) {
        long long o = base + (long long)t * stride;
        for (int i = 0; i < per; ++i) {
            int d = lane + i * 32;
            if (d < head_dim) acc += qd[i] * __half2float(kv[o + d]);
        }
    }
    for (int s = 16; s > 0; s >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) out[blockIdx.x] = acc;
}
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
    let func = module.load_function("read_kv").unwrap();

    // (name, kv_heads, head_dim, layers). planes P = layers*2 (K plane + V plane
    // per layer). The G sweep below controls the token stride; the 192 KB
    // token-major stride is the qwen14b G=P case (the open-risk config).
    let configs: &[(&str, i32, i32, i32)] = &[("qwen0.5b", 2, 64, 24), ("qwen14b", 8, 128, 48)];
    let reps = 30usize;
    let warmup = 6usize;

    println!(
        "config, kv_heads, head_dim, layers, planes, L, G, stride_B, ws_MiB, blocks, \
         min_us, med_us, min_GBs, med_GBs, ratio_vs_G1"
    );
    for &(name, kv_heads, head_dim, layers) in configs {
        let planes = layers * 2;
        // G values that evenly divide the plane count, spanning ~2 KB .. ~192 KB.
        let g_all: Vec<i32> = [1i32, 4, planes / 4, planes]
            .into_iter()
            .filter(|&g| g >= 1 && planes % g == 0)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();

        for &l in &[512i32, 2048, 8192, 32768] {
            let elems =
                (planes as usize) * (kv_heads as usize) * (l as usize) * (head_dim as usize);
            let ws_bytes = elems * 2; // f16
            let ws_mib = ws_bytes as f64 / (1024.0 * 1024.0);

            // One shared buffer per (config, L), reinterpreted for each G. Skip
            // (don't abort) if it does not fit this 8 GiB card — larger L still
            // reported for configs that fit.
            let kv = match stream.alloc_zeros::<half::f16>(elems) {
                Ok(b) => b,
                Err(_) => {
                    println!(
                        "{name}, {kv_heads}, {head_dim}, {layers}, {planes}, {l}, -, -, \
                         {ws_mib:.0}, -, SKIP_OOM, -, -, -, -"
                    );
                    continue;
                }
            };
            let q = stream
                .alloc_zeros::<half::f16>((kv_heads as usize) * (head_dim as usize))
                .unwrap();
            let blocks = (planes * kv_heads) as u32;
            let out = stream.alloc_zeros::<f32>(blocks as usize).unwrap();
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let bytes = ws_bytes as f64; // whole buffer read exactly once

            let run = |g: i32| -> f64 {
                let t0 = std::time::Instant::now();
                let mut b = stream.launch_builder(&func);
                b.arg(&kv)
                    .arg(&q)
                    .arg(&out)
                    .arg(&kv_heads)
                    .arg(&l)
                    .arg(&head_dim)
                    .arg(&g);
                unsafe { b.launch(cfg) }.unwrap();
                stream.synchronize().unwrap();
                t0.elapsed().as_secs_f64() * 1e6 // us
            };

            for _ in 0..warmup {
                for &g in &g_all {
                    run(g);
                }
            }
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); g_all.len()];
            for _ in 0..reps {
                for (i, &g) in g_all.iter().enumerate() {
                    samples[i].push(run(g));
                }
            }
            let stat = |mut xs: Vec<f64>| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (xs[0], xs[xs.len() / 2])
            };
            // G=1 median as the ratio baseline.
            let base_med = {
                let (_, m) = stat(samples[0].clone());
                m
            };
            let gbs = |us: f64| bytes / (us * 1e-6) / 1e9;
            for (i, &g) in g_all.iter().enumerate() {
                let (mn, md) = stat(samples[i].clone());
                let stride_b = (g as usize) * (kv_heads as usize) * (head_dim as usize) * 2;
                let ratio = md / base_med; // >1 = this stride slower than 2 KB
                println!(
                    "{name}, {kv_heads}, {head_dim}, {layers}, {planes}, {l}, {g}, {stride_b}, \
                     {ws_mib:.0}, {blocks}, {mn:.1}, {md:.1}, {:.1}, {:.1}, {ratio:.3}",
                    gbs(mn),
                    gbs(md)
                );
            }
        }
    }
    println!(
        "\n# ratio_vs_G1 > 1.0 means the larger (token-major) stride is slower than the ~2 KB \
         BSNH per-layer stride — i.e. TLB pressure is real. ~1.0 across all G means 2 MiB \
         device pages cover the 192 KB stride and token-major is free on the read path."
    );
}
