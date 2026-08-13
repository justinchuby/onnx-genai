//! PROTOTYPE / MICRO-BENCHMARK — dense-decode megakernel headroom probe.
//!
//! This is a **throwaway measurement harness**, not a shipped code path. It is
//! deliberately isolated in the test binary: nothing here is wired into the
//! executor, the capture pipeline, or any kernel dispatch. It exists to answer
//! one decision-support question for `docs/research/dense-decode-megakernel-feasibility.md`:
//!
//!   At M=1 decode, how much of the ~21 ms/token wall time is *recoverable* by
//!   collapsing the ~2568-node serial launch chain into a fused/persistent
//!   megakernel — i.e. by removing per-node launch/schedule overhead (a),
//!   inter-op activation DRAM round-trips (b), and per-kernel occupancy
//!   fill/drain (c) — versus the essential work (weight/KV DRAM + MACs) a
//!   megakernel must still do?
//!
//! Muse-Glimmer-30B decode op mix (measured, eager, `ONNX_GENAI_PROFILE_OPS=1`):
//! MatMulNBits 417, GroupQueryAttention 52, Mul 210, Add 311,
//! SimplifiedLayerNormalization 312, Sigmoid 104, Reshape 208 — hidden H=6656.
//! The elementwise/norm "glue" (Mul/Add/Norm/Sigmoid/Reshape) is 70% of the node
//! COUNT and ~44% of eager decode time; each is a tiny launch that reads and
//! rewrites the 6656-element bf16 hidden vector (~13 KiB) through global memory.
//! That round-trip is exactly what a megakernel keeps in registers/shared.
//!
//! The harness measures three things with CUDA events (median of many iters):
//!   1. per-launch GPU floor — K sequential trivial launches on one stream;
//!   2. realistic per-glue-op cost — K sequential launches that read+rewrite the
//!      full H bf16 vector (models one Add/Mul/Norm/Sigmoid);
//!   3. fusion recovery — one fused kernel that loads H once, applies G ops in
//!      registers (fp32 accumulate), stores once, vs G sequential unfused ops.
//!
//! fp32 accumulation order inside the fused kernel matches the per-op chain
//! (each op is applied in sequence to the same running fp32 value), so the fused
//! result is byte-comparable to the unfused chain up to bf16 round-trip removal
//! — a real megakernel would keep the intermediate in fp32 too. The harness
//! verifies the fused/unfused outputs agree to a tight tolerance.
//!
//! Ignored by default (needs a CUDA device). Run explicitly:
//!   cargo test --release -p onnx-runtime-ep-cuda --features cuda \
//!     --test megakernel_headroom_gpu -- --ignored --nocapture
//! Knobs (env): NXRT_MK_HIDDEN (default 6656), NXRT_MK_GLUE_OPS (22),
//!   NXRT_MK_ITERS (200), NXRT_MK_LAYERS (52).

#![allow(clippy::uninlined_format_args)]

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_cuda::CudaExecutionProvider;

const MODULE: &str = "nxrt_megakernel_headroom_probe";

// Trivial kernel: one thread writes a single element. Isolates pure launch +
// grid schedule + fill/drain latency (component (a)) with negligible memory work.
// Glue kernel: read+rewrite the full H bf16 vector once (fp32 interior). Models a
// single Add/Mul/Norm/Sigmoid glue op — the per-op activation round-trip (b)+(c).
// Fused kernel: load H once into registers, apply the glue op `reps` times in
// fp32, store once. Models the megakernel collapsing `reps` glue ops into one
// launch with zero intermediate global traffic.
const SRC: &str = r#"
#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

extern "C" __global__ void trivial(int* scratch) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        scratch[0] = scratch[0] + 1;
    }
}

// One glue op: y = silu(x) + bias, computed in fp32, stored bf16. In-place so a
// chain of launches on one stream serializes into a real read-after-write DRAM
// round-trip per op, exactly like the captured per-op decode chain.
extern "C" __global__ void glue_op(__nv_bfloat16* buf, int n, float bias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __bfloat162float(buf[i]);
    float y = x / (1.0f + expf(-x)) + bias;
    buf[i] = __float2bfloat16_rn(y);
}

// Fused: load once, apply `reps` glue ops in registers (fp32), store once.
extern "C" __global__ void glue_fused(__nv_bfloat16* buf, int n, float bias, int reps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __bfloat162float(buf[i]);
    for (int r = 0; r < reps; ++r) {
        x = x / (1.0f + expf(-x)) + bias;
    }
    buf[i] = __float2bfloat16_rn(x);
}
"#;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "requires a CUDA device; run with --ignored --nocapture"]
fn megakernel_headroom_probe() {
    let ep = match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        _ => {
            eprintln!("[mk-probe] no CUDA runtime; skipping");
            return;
        }
    };
    let runtime = ep.runtime();
    runtime
        .require_nvrtc_half_headers("megakernel-headroom-probe")
        .expect("bf16 NVRTC headers");
    let stream = runtime.stream().clone();
    let ctx = stream.context().clone();

    let hidden = env_usize("NXRT_MK_HIDDEN", 6656);
    let glue_ops = env_usize("NXRT_MK_GLUE_OPS", 22);
    let iters = env_usize("NXRT_MK_ITERS", 200);
    let layers = env_usize("NXRT_MK_LAYERS", 52);

    let trivial = runtime.nvrtc_function(MODULE, SRC, "trivial").unwrap();
    let glue = runtime.nvrtc_function(MODULE, SRC, "glue_op").unwrap();
    let fused = runtime.nvrtc_function(MODULE, SRC, "glue_fused").unwrap();

    let block = 256u32;
    let grid = (hidden as u32).div_ceil(block);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg1 = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let scratch = runtime.alloc_raw(std::mem::size_of::<i32>()).unwrap();
    let buf_bytes = hidden * std::mem::size_of::<u16>();
    let buf = runtime.alloc_raw(buf_bytes).unwrap();
    // Seed the hidden buffer with small bf16 values (~0.1) so silu stays finite.
    let seed_bf16: u16 = 0x2E66; // ~0.1 in bf16
    let seed = vec![seed_bf16; hidden];
    let seed_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(seed.as_ptr().cast(), std::mem::size_of_val(seed.as_slice()))
    };

    let time_ms = |launch: &mut dyn FnMut()| -> f32 {
        let start = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        let end = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        start.record(&stream).unwrap();
        launch();
        end.record(&stream).unwrap();
        start.elapsed_ms(&end).unwrap()
    };

    // --- (1) per-launch GPU floor: K sequential trivial launches ---
    let launch_k = |k: usize| {
        let mut b = stream.launch_builder(&trivial);
        b.arg(&scratch);
        for _ in 0..k {
            unsafe { b.launch(cfg1) }.unwrap();
        }
    };
    launch_k(64); // warm
    runtime.synchronize().unwrap();
    let mut floor_per_launch_us = 0.0f32;
    for &k in &[256usize, 1024, 2568] {
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            samples.push(time_ms(&mut || launch_k(k)));
        }
        runtime.synchronize().unwrap();
        let ms = median(samples);
        let per = ms * 1000.0 / k as f32;
        if k == 2568 {
            floor_per_launch_us = per;
        }
        eprintln!(
            "[mk-probe] launch-floor K={:>4}: {:.3} ms total, {:.3} us/launch",
            k, ms, per
        );
    }

    // --- (2) realistic per-glue-op cost: K sequential H-vector round-trips ---
    let launch_glue = |k: usize, bias: f32| {
        let mut b = stream.launch_builder(&glue);
        let n = hidden as i32;
        b.arg(&buf).arg(&n).arg(&bias);
        for _ in 0..k {
            unsafe { b.launch(cfg) }.unwrap();
        }
    };
    unsafe { runtime.htod(seed_bytes, buf).unwrap() };
    launch_glue(8, 0.0); // warm (bias 0 keeps values stable)
    runtime.synchronize().unwrap();
    let mut glue_samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        unsafe { runtime.htod(seed_bytes, buf).unwrap() };
        glue_samples.push(time_ms(&mut || launch_glue(glue_ops, 0.0)));
    }
    runtime.synchronize().unwrap();
    let glue_ms = median(glue_samples);
    let per_glue_us = glue_ms * 1000.0 / glue_ops as f32;
    eprintln!(
        "[mk-probe] glue unfused G={} @H={}: {:.3} ms, {:.3} us/op",
        glue_ops, hidden, glue_ms, per_glue_us
    );

    // --- (3) fusion recovery: 1 fused launch of G ops vs G unfused launches ---
    unsafe { runtime.htod(seed_bytes, buf).unwrap() };
    let launch_fused = |reps: i32, bias: f32| {
        let mut b = stream.launch_builder(&fused);
        let n = hidden as i32;
        b.arg(&buf).arg(&n).arg(&bias).arg(&reps);
        unsafe { b.launch(cfg) }.unwrap();
    };
    launch_fused(glue_ops as i32, 0.0); // warm
    runtime.synchronize().unwrap();
    let mut fused_samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        unsafe { runtime.htod(seed_bytes, buf).unwrap() };
        fused_samples.push(time_ms(&mut || launch_fused(glue_ops as i32, 0.0)));
    }
    runtime.synchronize().unwrap();
    let fused_ms = median(fused_samples);
    eprintln!(
        "[mk-probe] glue fused   G={} @H={}: {:.3} ms (1 launch)",
        glue_ops, hidden, fused_ms
    );

    // Numerics check: fused (fp32 chained) vs unfused (bf16 round-trip each op).
    // Use bias 0 so both reduce to iterated silu; compare final buffers.
    unsafe { runtime.htod(seed_bytes, buf).unwrap() };
    launch_glue(glue_ops, 0.0);
    runtime.synchronize().unwrap();
    let mut unfused_out = vec![0u8; buf_bytes];
    unsafe { runtime.dtoh(&mut unfused_out, buf).unwrap() };
    unsafe { runtime.htod(seed_bytes, buf).unwrap() };
    launch_fused(glue_ops as i32, 0.0);
    runtime.synchronize().unwrap();
    let mut fused_out = vec![0u8; buf_bytes];
    unsafe { runtime.dtoh(&mut fused_out, buf).unwrap() };
    let read_bf16 = |bytes: &[u8], i: usize| -> f32 {
        let raw = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        half::bf16::from_bits(raw).to_f32()
    };
    let mut max_abs = 0.0f32;
    for i in 0..hidden {
        let d = (read_bf16(&unfused_out, i) - read_bf16(&fused_out, i)).abs();
        max_abs = max_abs.max(d);
    }
    eprintln!(
        "[mk-probe] numerics fused-vs-unfused max_abs={:.3e} (bf16 round-trip drift only)",
        max_abs
    );

    let recovered = (glue_ms - fused_ms) / glue_ms * 100.0;
    eprintln!(
        "[mk-probe] === glue fusion recovered {:.1}% of glue time ({:.3} -> {:.3} ms) ===",
        recovered, glue_ms, fused_ms
    );

    // --- projection to whole model ---
    // Eager op mix per token (measured): glue time ~44% of eager decode; captured
    // baseline ~21.4 ms/token. Report node-count collapse arithmetic for the doc.
    let glue_nodes_total = glue_ops * layers;
    eprintln!(
        "[mk-probe] projection: glue nodes/token ~= {} (G={} x {} layers); \
         per-launch floor ~{:.2} us; realistic glue op ~{:.2} us",
        glue_nodes_total, glue_ops, layers, floor_per_launch_us, per_glue_us
    );

    unsafe {
        runtime.free_raw(scratch).unwrap();
        runtime.free_raw(buf).unwrap();
    }
    assert!(
        max_abs < 5e-2,
        "fused vs unfused drifted more than bf16 round-trip explains"
    );
}

// ===========================================================================
// P1.5 — real fused-int4-GEMV one-layer megakernel probe + grid.sync-under-
// capture gate. Extends the Phase B glue-only recovery number with the part
// that actually gates the multi-week P2: (Exp 1) can a cooperative launch —
// the only launch path a persistent multi-CTA megakernel with grid.sync can
// use — even be *captured* into the decode CUDA graph? and (Exp 2) what does
// a single-CTA fused int4 MLP (gate/up/silu-mul/down with the 19968-wide
// intermediate held resident, zero activation DRAM round-trips) cost versus
// the current per-op int4 GEMV launch sequence on identical tensors?
//
// Both are throwaway measurement kernels — NOT wired into any dispatch path.
// ===========================================================================

// Faithful f32-activation int4 decode GEMV (block-32, symmetric zero-point 8,
// nibble unpack, fp32 accumulate + block_sum) — same math/order as the
// production `matmul_nbits_gemv_f32` reference. Used BOTH as the per-op
// baseline (grid = N columns, one block reduces one column over K, full-device
// parallel weight reads) AND, inlined, inside the fused single-CTA megakernel
// so the two are byte-comparable. `fused_mlp` keeps sx[H] and sact[I] resident
// in dynamic shared memory across the whole MLP: only the packed weights stream
// from DRAM, the 19968-wide intermediate never touches global memory.
const MLP_SRC: &str = r#"
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = threadIdx.x < ((blockDim.x + 31) >> 5) ? warp_sums[lane] : 0.0f;
    return warp == 0 ? warp_sum(value) : 0.0f;
}

// One int4 GEMV output column, symmetric zp=8, block_size=32, blob=16 bytes.
// `act` lives in `src` (global for the baseline, shared for the fused kernel).
__device__ __forceinline__ float gemv_col_f32(
    const float* act, const unsigned char* packed, const float* scales,
    int k, int col) {
    const int k_blocks = k >> 5;      // block_size 32
    const int blob = 16;              // 32 int4 = 16 bytes
    float value = 0.0f;
    for (int depth = threadIdx.x; depth < k; depth += blockDim.x) {
        int block = depth >> 5;
        int within = depth & 31;
        int base = (col * k_blocks + block) * blob;
        unsigned char byte = packed[base + (within >> 1)];
        int q = (within & 1) ? (byte >> 4) : (byte & 15);
        float s = scales[col * k_blocks + block];
        value += act[depth] * (float)(q - 8) * s;
    }
    return block_sum(value);
}

// Per-op baseline GEMV: grid = N, block reduces one column over K.
extern "C" __global__ void gemv_ref(
    const float* act, const unsigned char* packed, const float* scales,
    float* out, int k, int n) {
    int col = blockIdx.x;
    if (col >= n) return;
    float v = gemv_col_f32(act, packed, scales, k, col);
    if (threadIdx.x == 0) out[col] = v;
}

// Per-op baseline SiLU-mul: out[i] = silu(gate[i]) * up[i].
extern "C" __global__ void silu_mul(
    const float* gate, const float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

// Fused single-CTA MLP megakernel: one block executes the whole MLP with the
// hidden input and the 19968-wide intermediate resident in shared memory.
// Reduction order (thread stride + block_sum) is identical to gemv_ref, so the
// output is byte-exact-capable vs the per-op baseline (no reduction reorder).
extern "C" __global__ void fused_mlp(
    const float* xin,
    const unsigned char* gate_packed, const float* gate_scales,
    const unsigned char* up_packed, const float* up_scales,
    const unsigned char* down_packed, const float* down_scales,
    float* out, int h, int inter) {
    extern __shared__ float smem[];
    float* sx = smem;           // [h]
    float* sact = smem + h;     // [inter]
    for (int i = threadIdx.x; i < h; i += blockDim.x) sx[i] = xin[i];
    __syncthreads();
    // gate/up GEMV over K=h, fused SiLU-mul, intermediate stays in shared.
    for (int col = 0; col < inter; ++col) {
        float g = gemv_col_f32(sx, gate_packed, gate_scales, h, col);
        float u = gemv_col_f32(sx, up_packed, up_scales, h, col);
        if (threadIdx.x == 0) sact[col] = (g / (1.0f + expf(-g))) * u;
        __syncthreads();
    }
    // down GEMV over K=inter, reading the resident intermediate from shared.
    for (int col = 0; col < h; ++col) {
        float v = gemv_col_f32(sact, down_packed, down_scales, inter, col);
        if (threadIdx.x == 0) out[col] = v;
        __syncthreads();
    }
}
"#;

// Deterministic pseudo-random byte fill (no rand dep; stable across runs).
fn fill_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.push((s >> 24) as u8);
    }
    v
}

fn fill_f32(n: usize, seed: u64, scale: f32, bias: f32) -> Vec<u8> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n * 4);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = ((s >> 40) as f32) / ((1u64 << 24) as f32); // [0,1)
        out.extend_from_slice(&(u * scale + bias).to_le_bytes());
    }
    out
}

#[test]
#[ignore = "requires a CUDA device; run with --ignored --nocapture"]
fn megakernel_int4_mlp_probe() {
    use cudarc::driver::sys::CUfunction_attribute;

    let ep = match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        _ => {
            eprintln!("[mk15-mlp] no CUDA runtime; skipping");
            return;
        }
    };
    let runtime = ep.runtime();
    let stream = runtime.stream().clone();
    let ctx = stream.context().clone();

    // Real Muse-Glimmer-30B MLP shapes, block-32 int4.
    let h = env_usize("NXRT_MK_HIDDEN", 6656);
    let inter = env_usize("NXRT_MK_INTER", 19968);
    let iters = env_usize("NXRT_MK_MLP_ITERS", 50);
    let block = 256u32;
    assert_eq!(h % 32, 0);
    assert_eq!(inter % 32, 0);
    let kb_gate = h / 32; // k_blocks for gate/up (K = h)
    let kb_down = inter / 32; // k_blocks for down  (K = inter)
    let blob = 16usize;

    let gemv = runtime
        .nvrtc_function("nxrt_mk_mlp", MLP_SRC, "gemv_ref")
        .unwrap();
    let silu = runtime
        .nvrtc_function("nxrt_mk_mlp", MLP_SRC, "silu_mul")
        .unwrap();
    let fused = runtime
        .nvrtc_function("nxrt_mk_mlp", MLP_SRC, "fused_mlp")
        .unwrap();

    // Opt in to the 104 KiB dynamic shared the fused kernel needs (H200 SM = 227 KiB).
    let smem_bytes = ((h + inter) * std::mem::size_of::<f32>()) as u32;
    fused
        .set_attribute(
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_bytes as i32,
        )
        .expect("opt in to dynamic shared mem");

    // Device buffers: packed weights + scales for gate/up/down, activations, outputs.
    let alloc = |bytes: usize, data: &[u8]| {
        let p = runtime.alloc_raw(bytes).unwrap();
        unsafe { runtime.htod(data, p).unwrap() };
        p
    };
    let gate_packed = alloc(
        inter * kb_gate * blob,
        &fill_bytes(inter * kb_gate * blob, 11),
    );
    let up_packed = alloc(
        inter * kb_gate * blob,
        &fill_bytes(inter * kb_gate * blob, 22),
    );
    let down_packed = alloc(h * kb_down * blob, &fill_bytes(h * kb_down * blob, 33));
    let gate_scales = alloc(
        inter * kb_gate * 4,
        &fill_f32(inter * kb_gate, 44, 0.01, 0.0),
    );
    let up_scales = alloc(
        inter * kb_gate * 4,
        &fill_f32(inter * kb_gate, 55, 0.01, 0.0),
    );
    let down_scales = alloc(h * kb_down * 4, &fill_f32(h * kb_down, 66, 0.01, 0.0));
    let xin = alloc(h * 4, &fill_f32(h, 77, 0.2, -0.1));
    let gate_out = runtime.alloc_raw(inter * 4).unwrap();
    let up_out = runtime.alloc_raw(inter * 4).unwrap();
    let act_out = runtime.alloc_raw(inter * 4).unwrap();
    let base_out = runtime.alloc_raw(h * 4).unwrap();
    let fused_out = runtime.alloc_raw(h * 4).unwrap();

    let time_ms = |launch: &mut dyn FnMut()| -> f32 {
        let start = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        let end = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        start.record(&stream).unwrap();
        launch();
        end.record(&stream).unwrap();
        start.elapsed_ms(&end).unwrap()
    };

    // --- baseline: 4 separate launches (gate GEMV, up GEMV, silu-mul, down GEMV) ---
    let hi = h as i32;
    let interi = inter as i32;
    let run_baseline = || {
        let gcfg = LaunchConfig {
            grid_dim: (inter as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&gemv);
        b.arg(&xin)
            .arg(&gate_packed)
            .arg(&gate_scales)
            .arg(&gate_out)
            .arg(&hi)
            .arg(&interi);
        unsafe { b.launch(gcfg) }.unwrap();
        let mut b = stream.launch_builder(&gemv);
        b.arg(&xin)
            .arg(&up_packed)
            .arg(&up_scales)
            .arg(&up_out)
            .arg(&hi)
            .arg(&interi);
        unsafe { b.launch(gcfg) }.unwrap();
        let scfg = LaunchConfig {
            grid_dim: ((inter as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&silu);
        b.arg(&gate_out).arg(&up_out).arg(&act_out).arg(&interi);
        unsafe { b.launch(scfg) }.unwrap();
        let dcfg = LaunchConfig {
            grid_dim: (h as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&gemv);
        b.arg(&act_out)
            .arg(&down_packed)
            .arg(&down_scales)
            .arg(&base_out)
            .arg(&interi)
            .arg(&hi);
        unsafe { b.launch(dcfg) }.unwrap();
    };

    // --- fused single-CTA megakernel: 1 launch, intermediate resident in shared ---
    let run_fused = || {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        let mut b = stream.launch_builder(&fused);
        b.arg(&xin)
            .arg(&gate_packed)
            .arg(&gate_scales)
            .arg(&up_packed)
            .arg(&up_scales)
            .arg(&down_packed)
            .arg(&down_scales)
            .arg(&fused_out)
            .arg(&hi)
            .arg(&interi);
        unsafe { b.launch(cfg) }.unwrap();
    };

    run_baseline();
    run_fused();
    runtime.synchronize().unwrap();

    let mut base_samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        base_samples.push(time_ms(&mut || run_baseline()));
    }
    runtime.synchronize().unwrap();
    let mut fused_samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        fused_samples.push(time_ms(&mut || run_fused()));
    }
    runtime.synchronize().unwrap();

    let base_ms = median(base_samples);
    let fused_ms = median(fused_samples);
    eprintln!(
        "[mk15-mlp] per-op baseline MLP (4 launches, full-device parallel): {:.4} ms/layer",
        base_ms
    );
    eprintln!(
        "[mk15-mlp] fused single-CTA MLP (1 launch, intermediate resident):  {:.4} ms/layer",
        fused_ms
    );
    eprintln!(
        "[mk15-mlp] fused/baseline ratio = {:.2}x  (>1 => single-CTA is SLOWER: \
         loses full-device weight-read parallelism; a real megakernel MUST be multi-CTA)",
        fused_ms / base_ms
    );

    // Numerics: identical dequant + reduction order => byte-exact-capable.
    let mut base_host = vec![0u8; h * 4];
    let mut fused_host = vec![0u8; h * 4];
    unsafe {
        runtime.dtoh(&mut base_host, base_out).unwrap();
        runtime.dtoh(&mut fused_host, fused_out).unwrap();
    }
    let rd = |b: &[u8], i: usize| {
        f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
    };
    let mut max_abs = 0.0f32;
    let mut max_ulp = 0i64;
    for i in 0..h {
        let a = rd(&base_host, i);
        let b = rd(&fused_host, i);
        max_abs = max_abs.max((a - b).abs());
        max_ulp = max_ulp.max((a.to_bits() as i64 - b.to_bits() as i64).abs());
    }
    eprintln!(
        "[mk15-mlp] numerics fused-vs-baseline: max_abs={:.3e}, max_ulp={} \
         (0 => byte-exact; fp32 accumulate + identical block_sum order preserved)",
        max_abs, max_ulp
    );

    unsafe {
        for p in [
            gate_packed,
            up_packed,
            down_packed,
            gate_scales,
            up_scales,
            down_scales,
            xin,
            gate_out,
            up_out,
            act_out,
            base_out,
            fused_out,
        ] {
            runtime.free_raw(p).unwrap();
        }
    }

    // The MLP is ~2/3 of a decoder layer's GEMV FLOPs (gate+up+down vs the
    // 4 attention projections); reported per-layer numbers in the doc combine
    // this with the measured attention/glue costs.
    assert!(base_ms > 0.0 && fused_ms > 0.0);
}

#[test]
#[ignore = "requires a CUDA device; run with --ignored --nocapture"]
fn grid_sync_capture_gate_probe() {
    use cudarc::driver::result;
    use cudarc::driver::sys::{
        CUgraphInstantiate_flags, CUstreamCaptureMode, CUstreamCaptureStatus,
    };
    use std::ffi::{CString, c_void};

    let ep = match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        _ => {
            eprintln!("[mk15-coop] no CUDA runtime; skipping");
            return;
        }
    };
    let runtime = ep.runtime();
    let stream = runtime.stream().clone();

    // A persistent multi-CTA megakernel with grid-wide producer/consumer sync
    // (cg::this_grid().sync()) can ONLY be launched with cuLaunchCooperativeKernel.
    // The decisive P2 question is whether that launch path is capturable into the
    // decode CUDA graph. The kernel body is irrelevant to the launch-API gate, so
    // we use a trivial co-resident kernel (avoids NVRTC cooperative_groups.h dep).
    let coop_src = r#"
extern "C" __global__ void coop_noop(int* p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1;
}
"#;
    // Raw CUfunction (cudarc's CudaFunction hides cu_function) so we can call
    // cuLaunchCooperativeKernel directly. Compile to CUBIN (sm_90, this H200) and
    // load it via cuModuleLoadData — the same fallback the runtime uses to dodge
    // the "unsupported PTX toolchain" JIT check when NVRTC is newer than the driver.
    let source = CString::new(coop_src).unwrap();
    let name = CString::new("coop_mod").unwrap();
    let program =
        match cudarc::nvrtc::result::create_program(source.as_c_str(), Some(name.as_c_str())) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[mk15-coop] NVRTC create_program failed ({e:?}); skipping");
                return;
            }
        };
    let options = vec!["--gpu-architecture=sm_90".to_string()];
    if let Err(e) = unsafe { cudarc::nvrtc::result::compile_program(program, &options) } {
        eprintln!("[mk15-coop] NVRTC compile failed ({e:?}); skipping");
        let _ = unsafe { cudarc::nvrtc::result::destroy_program(program) };
        return;
    }
    let mut size = 0usize;
    unsafe { cudarc::nvrtc::sys::nvrtcGetCUBINSize(program, &mut size) }
        .result()
        .unwrap();
    let mut image = vec![0u8; size];
    unsafe { cudarc::nvrtc::sys::nvrtcGetCUBIN(program, image.as_mut_ptr().cast()) }
        .result()
        .unwrap();
    let _ = unsafe { cudarc::nvrtc::result::destroy_program(program) };
    let module = unsafe { result::module::load_data(image.as_ptr() as *const c_void) }.unwrap();
    let func = unsafe { result::module::get_function(module, CString::new("coop_noop").unwrap()) }
        .unwrap();

    let p = runtime.alloc_raw(std::mem::size_of::<i32>()).unwrap();
    let mut ptr = p;
    let mut params: Vec<*mut c_void> = vec![&mut ptr as *mut _ as *mut c_void];

    // (A) sanity: cooperative launch works OUTSIDE capture on this device.
    let coop_ok = unsafe {
        result::launch_cooperative_kernel(
            func,
            (1, 1, 1),
            (32, 1, 1),
            0,
            stream.cu_stream(),
            &mut params,
        )
    };
    runtime.synchronize().ok();
    match &coop_ok {
        Ok(()) => {
            eprintln!("[mk15-coop] (A) cooperative launch OUTSIDE capture: OK (device supports it)")
        }
        Err(e) => {
            eprintln!(
                "[mk15-coop] (A) cooperative launch unsupported on device: {e:?}; skipping gate"
            );
            unsafe { runtime.free_raw(p).ok() };
            return;
        }
    }

    // (B) THE GATE: attempt the same cooperative launch DURING stream capture.
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .expect("begin capture");
    let launch_during_capture = unsafe {
        result::launch_cooperative_kernel(
            func,
            (1, 1, 1),
            (32, 1, 1),
            0,
            stream.cu_stream(),
            &mut params,
        )
    };
    let status_after = stream.capture_status();
    // End capture regardless, to clean up the (possibly invalidated) stream.
    let graph = stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);

    let captured_ok = launch_during_capture.is_ok()
        && matches!(
            status_after,
            Ok(CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE)
        )
        && matches!(graph, Ok(Some(_)));

    eprintln!(
        "[mk15-coop] (B) cooperative launch DURING capture: launch={:?}, status_after={:?}, graph_built={}",
        launch_during_capture.map(|_| "Ok"),
        status_after,
        matches!(graph, Ok(Some(_)))
    );
    if captured_ok {
        eprintln!(
            "[mk15-coop] === VERDICT: grid.sync/cooperative launch IS capturable => \
             a multi-CTA persistent megakernel COULD live inside the decode graph ==="
        );
    } else {
        eprintln!(
            "[mk15-coop] === VERDICT: cooperative launch is NOT capturable (invalidates capture) => \
             a grid.sync megakernel CANNOT be captured; P2 must use a single giant CTA \
             (bandwidth-starved) OR an eager (uncaptured) seam for the megakernel ==="
        );
    }

    // Drain any capture error so the shared runtime stream stays usable.
    runtime.synchronize().ok();
    unsafe { runtime.free_raw(p).ok() };
}

// ===========================================================================
// P2-prototype — persistent MULTI-CTA cooperative one-layer megakernel.
//
// P1.5 pinned the architecture: single-CTA residency is 926x too slow (one SM
// ~= 1/132 of device weight-read bandwidth), and grid.sync IS capturable, so
// the megakernel must be a persistent MULTI-CTA cooperative kernel — every
// sub-GEMV reads its int4 weights across the FULL device, activations pass
// between sub-GEMVs through small L2-resident global scratch synchronized by
// grid.sync (NOT pinned to one CTA's shared memory).
//
// This probe builds that kernel for the MLP triple-GEMV block (gate/up ->
// SiLU-mul -> down, the largest self-contained GEMV chain in a decoder layer)
// and measures per-MLP GPU time vs the current per-op launch sequence on
// IDENTICAL tensors. The int4 GEMV math is a representative block-32 f32-accum
// dequant used on BOTH sides, so the recovered *fraction*, the grid.sync
// barrier cost, and the achieved occupancy are apples-to-apples even though the
// absolute ms is not the production f16 split-K dp4a kernel's absolute ms
// (same caveat as P1.5 §6.1). Also measures the standalone grid.sync barrier
// cost, the fixed per-seam overhead the projection must pay.
//
// Throwaway `#[ignore]`; not wired into any dispatch path.
// ===========================================================================

// Shared int4 dequant GEMV (block-32, symmetric zp=8, fp32 accumulate,
// identical block_sum order to matmul_nbits_gemv_f32). `gemv_grid` is the
// per-op baseline entry (grid-stride over columns so ONE launch with any grid
// size covers all N). Duplicated verbatim into MEGA_SRC below so the fused and
// per-op paths run byte-identical math.
const MC_GEMV_DEV: &str = r#"
__device__ __forceinline__ float mc_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float mc_block_sum(float value) {
    __shared__ float ws[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = mc_warp_sum(value);
    if (lane == 0) ws[warp] = value;
    __syncthreads();
    value = threadIdx.x < ((blockDim.x + 31) >> 5) ? ws[lane] : 0.0f;
    return warp == 0 ? mc_warp_sum(value) : 0.0f;
}
__device__ __forceinline__ float mc_gemv_col(
    const float* act, const unsigned char* packed, const float* scales,
    int k, int col) {
    const int k_blocks = k >> 5;
    const int blob = 16;
    float value = 0.0f;
    for (int depth = threadIdx.x; depth < k; depth += blockDim.x) {
        int block = depth >> 5;
        int within = depth & 31;
        int base = (col * k_blocks + block) * blob;
        unsigned char byte = packed[base + (within >> 1)];
        int q = (within & 1) ? (byte >> 4) : (byte & 15);
        float s = scales[col * k_blocks + block];
        value += act[depth] * (float)(q - 8) * s;
    }
    return mc_block_sum(value);
}
"#;

fn base_src() -> String {
    format!(
        "{MC_GEMV_DEV}\n{}",
        r#"
extern "C" __global__ void gemv_grid(
    const float* act, const unsigned char* packed, const float* scales,
    float* out, int k, int n) {
    for (int col = blockIdx.x; col < n; col += gridDim.x) {
        float v = mc_gemv_col(act, packed, scales, k, col);
        if (threadIdx.x == 0) out[col] = v;
        __syncthreads();
    }
}
extern "C" __global__ void silu_mul_grid(
    const float* g, const float* u, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float x = g[i];
        out[i] = (x / (1.0f + expf(-x))) * u[i];
    }
}
"#
    )
}

fn mega_src() -> String {
    format!(
        "#include <cooperative_groups.h>\nnamespace cg = cooperative_groups;\n{MC_GEMV_DEV}\n{}",
        r#"
// Persistent multi-CTA cooperative MLP: grid sized to occupancy so all CTAs are
// co-resident. Each sub-GEMV grid-strides its columns across the WHOLE grid
// (full-device weight reads). Activations pass through L2-resident global
// scratch, synchronized by grid.sync (2 barriers: after gate+up, after silu).
extern "C" __global__ void mlp_mega_coop(
    const float* xin,
    const unsigned char* gp, const float* gs,
    const unsigned char* up_packed, const float* us,
    const unsigned char* dp, const float* ds,
    float* gsc, float* usc, float* asc, float* out,
    int h, int inter) {
    cg::grid_group grid = cg::this_grid();
    for (int col = blockIdx.x; col < inter; col += gridDim.x) {
        float v = mc_gemv_col(xin, gp, gs, h, col);
        if (threadIdx.x == 0) gsc[col] = v;
        __syncthreads();
    }
    for (int col = blockIdx.x; col < inter; col += gridDim.x) {
        float v = mc_gemv_col(xin, up_packed, us, h, col);
        if (threadIdx.x == 0) usc[col] = v;
        __syncthreads();
    }
    grid.sync();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inter;
         i += gridDim.x * blockDim.x) {
        float x = gsc[i];
        asc[i] = (x / (1.0f + expf(-x))) * usc[i];
    }
    grid.sync();
    for (int col = blockIdx.x; col < h; col += gridDim.x) {
        float v = mc_gemv_col(asc, dp, ds, inter, col);
        if (threadIdx.x == 0) out[col] = v;
        __syncthreads();
    }
}
// Standalone grid.sync cost: `reps` back-to-back barriers, nothing else.
extern "C" __global__ void barrier_only(int reps, int* sink) {
    cg::grid_group grid = cg::this_grid();
    int acc = 0;
    for (int r = 0; r < reps; ++r) { grid.sync(); acc += r; }
    if (threadIdx.x == 0 && blockIdx.x == 0) sink[0] = acc;
}
"#
    )
}

// Discover NVRTC include dirs so cooperative_groups.h (and the libcudacxx
// `cuda/std/*` headers it pulls in) resolve. Prefer a self-consistent toolkit
// include set (CG header + its `cccl` libcudacxx subdir) over the wheel headers.
fn coop_include_paths() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push_if = |dir: std::path::PathBuf, marker: &str| {
        if dir.join(marker).exists() {
            let s = dir.to_string_lossy().into_owned();
            if !out.contains(&s) {
                out.push(s);
            }
        }
    };
    // Toolkit roots that ship both the CG header and the cccl libcudacxx tree.
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(var) {
            roots.push(std::path::PathBuf::from(root));
        }
    }
    for g in glob_cuda_roots() {
        roots.push(g);
    }
    for root in roots {
        for inc in [
            root.join("include"),
            root.join("targets/x86_64-linux/include"),
        ] {
            // CG header dir first, then its cccl libcudacxx subdir.
            push_if(inc.clone(), "cooperative_groups.h");
            push_if(inc.join("cccl"), "cuda/std/type_traits");
            push_if(inc.clone(), "cuda/std/type_traits");
        }
    }
    // Wheel fallback (may still miss cuda/std; only used if toolkit absent).
    if let Some(paths) = std::env::var_os("LD_LIBRARY_PATH") {
        for p in std::env::split_paths(&paths) {
            if let Some(nvidia) = p.parent().and_then(|x| x.parent()) {
                push_if(nvidia.join("cuda_runtime/include"), "cooperative_groups.h");
            }
        }
    }
    out
}

fn glob_cuda_roots() -> Vec<std::path::PathBuf> {
    let mut v = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        for e in entries.flatten() {
            let name = e.file_name();
            if name.to_string_lossy().starts_with("cuda") {
                v.push(e.path());
            }
        }
    }
    v.push("/usr/local/cuda".into());
    v
}

// Compile `src` to CUBIN (sm_90) with the given includes and return a raw
// CUfunction for `entry` — needed because cudarc hides cu_function and only the
// raw handle can be passed to cuLaunchCooperativeKernel / raw occupancy.
fn cubin_function(
    src: &str,
    entry: &str,
    includes: &[String],
) -> Option<cudarc::driver::sys::CUfunction> {
    use cudarc::driver::result;
    use std::ffi::{CString, c_void};
    let source = CString::new(src).ok()?;
    let name = CString::new("nxrt_mc_mod").ok()?;
    let program =
        cudarc::nvrtc::result::create_program(source.as_c_str(), Some(name.as_c_str())).ok()?;
    let mut opts: Vec<String> = vec!["--gpu-architecture=sm_90".into()];
    for inc in includes {
        opts.push(format!("--include-path={inc}"));
    }
    if let Err(e) = unsafe { cudarc::nvrtc::result::compile_program(program, &opts) } {
        let log = unsafe { cudarc::nvrtc::result::get_program_log(program) }
            .ok()
            .map(|b| {
                unsafe { std::ffi::CStr::from_ptr(b.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_default();
        eprintln!("[mk2] cubin compile failed for {entry}: {e:?}\n{log}");
        let _ = unsafe { cudarc::nvrtc::result::destroy_program(program) };
        return None;
    }
    let mut size = 0usize;
    unsafe { cudarc::nvrtc::sys::nvrtcGetCUBINSize(program, &mut size) }
        .result()
        .ok()?;
    let mut image = vec![0u8; size];
    unsafe { cudarc::nvrtc::sys::nvrtcGetCUBIN(program, image.as_mut_ptr().cast()) }
        .result()
        .ok()?;
    let _ = unsafe { cudarc::nvrtc::result::destroy_program(program) };
    let module = unsafe { result::module::load_data(image.as_ptr() as *const c_void) }.ok()?;
    unsafe { result::module::get_function(module, CString::new(entry).ok()?) }.ok()
}

#[test]
#[ignore = "requires a CUDA device; run with --ignored --nocapture"]
fn megakernel_multicta_mlp_probe() {
    use cudarc::driver::result;
    use cudarc::driver::sys::{CUdevice_attribute, CUevent_flags};
    use std::ffi::c_void;

    let ep = match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        _ => {
            eprintln!("[mk2] no CUDA runtime; skipping");
            return;
        }
    };
    let runtime = ep.runtime();
    let stream = runtime.stream().clone();
    let ctx = stream.context().clone();

    let h = env_usize("NXRT_MK_HIDDEN", 6656);
    let inter = env_usize("NXRT_MK_INTER", 19968);
    let iters = env_usize("NXRT_MK_MC_ITERS", 200);
    let block = 256u32;
    let kb_gate = h / 32;
    let kb_down = inter / 32;
    let blob = 16usize;

    // Baseline per-op kernels (CudaFunction via runtime — normal launches).
    let bsrc = base_src();
    let gemv = runtime
        .nvrtc_function(
            "nxrt_mc_base",
            Box::leak(bsrc.clone().into_boxed_str()),
            "gemv_grid",
        )
        .unwrap();
    let silu = runtime
        .nvrtc_function(
            "nxrt_mc_base",
            Box::leak(bsrc.into_boxed_str()),
            "silu_mul_grid",
        )
        .unwrap();

    // Megakernel + barrier probe: raw CUfunction (cooperative launch path).
    let includes = coop_include_paths();
    if includes.is_empty() {
        eprintln!("[mk2] no cooperative_groups.h include dir found; skipping");
        return;
    }
    let msrc = mega_src();
    let mega = match cubin_function(&msrc, "mlp_mega_coop", &includes) {
        Some(f) => f,
        None => {
            eprintln!("[mk2] mega kernel compile failed; skipping");
            return;
        }
    };
    let barrier = cubin_function(&msrc, "barrier_only", &includes).unwrap();

    // Device support + occupancy-based cooperative grid sizing.
    let coop_supported = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH)
        .unwrap_or(0);
    let sm_count = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .unwrap() as u32;
    let occ = unsafe {
        result::occupancy::max_active_block_per_multiprocessor_with_flags(mega, block as i32, 0, 0)
    }
    .unwrap_or(1) as u32;
    let coop_grid = (occ * sm_count).max(1);
    eprintln!(
        "[mk2] device: SMs={}, coop_launch_supported={}, mega occupancy={} blocks/SM => cooperative grid={} CTAs ({} resident/SM)",
        sm_count, coop_supported, occ, coop_grid, occ
    );
    if coop_supported == 0 {
        eprintln!("[mk2] cooperative launch unsupported; skipping");
        return;
    }

    // Buffers (deterministic fills reused from the P1.5 helpers).
    let mk = |bytes: usize, data: &[u8]| {
        let p = runtime.alloc_raw(bytes).unwrap();
        unsafe { runtime.htod(data, p).unwrap() };
        p
    };
    let gate_packed = mk(
        inter * kb_gate * blob,
        &fill_bytes(inter * kb_gate * blob, 11),
    );
    let up_packed = mk(
        inter * kb_gate * blob,
        &fill_bytes(inter * kb_gate * blob, 22),
    );
    let down_packed = mk(h * kb_down * blob, &fill_bytes(h * kb_down * blob, 33));
    let gate_scales = mk(
        inter * kb_gate * 4,
        &fill_f32(inter * kb_gate, 44, 0.01, 0.0),
    );
    let up_scales = mk(
        inter * kb_gate * 4,
        &fill_f32(inter * kb_gate, 55, 0.01, 0.0),
    );
    let down_scales = mk(h * kb_down * 4, &fill_f32(h * kb_down, 66, 0.01, 0.0));
    let xin = mk(h * 4, &fill_f32(h, 77, 0.2, -0.1));
    let gsc = runtime.alloc_raw(inter * 4).unwrap();
    let usc = runtime.alloc_raw(inter * 4).unwrap();
    let asc = runtime.alloc_raw(inter * 4).unwrap();
    let base_out = runtime.alloc_raw(h * 4).unwrap();
    let mega_out = runtime.alloc_raw(h * 4).unwrap();
    let sink = runtime.alloc_raw(4).unwrap();

    let time_ms = |launch: &mut dyn FnMut()| -> f32 {
        let s = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        let e = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        s.record(&stream).unwrap();
        launch();
        e.record(&stream).unwrap();
        s.elapsed_ms(&e).unwrap()
    };

    let hi = h as i32;
    let interi = inter as i32;

    // --- baseline: 4 per-op launches (gate GEMV, up GEMV, silu-mul, down GEMV) ---
    let run_baseline = || {
        let gcfg = LaunchConfig {
            grid_dim: (inter as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&gemv);
        b.arg(&xin)
            .arg(&gate_packed)
            .arg(&gate_scales)
            .arg(&gsc)
            .arg(&hi)
            .arg(&interi);
        unsafe { b.launch(gcfg) }.unwrap();
        let mut b = stream.launch_builder(&gemv);
        b.arg(&xin)
            .arg(&up_packed)
            .arg(&up_scales)
            .arg(&usc)
            .arg(&hi)
            .arg(&interi);
        unsafe { b.launch(gcfg) }.unwrap();
        let scfg = LaunchConfig {
            grid_dim: ((inter as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&silu);
        b.arg(&gsc).arg(&usc).arg(&asc).arg(&interi);
        unsafe { b.launch(scfg) }.unwrap();
        let dcfg = LaunchConfig {
            grid_dim: (h as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&gemv);
        b.arg(&asc)
            .arg(&down_packed)
            .arg(&down_scales)
            .arg(&base_out)
            .arg(&interi)
            .arg(&hi);
        unsafe { b.launch(dcfg) }.unwrap();
    };

    // --- mega: 1 cooperative launch, grid = occupancy x SMs, grid.sync seams ---
    let run_mega = || {
        let mut a_xin = xin;
        let mut a_gp = gate_packed;
        let mut a_gs = gate_scales;
        let mut a_up = up_packed;
        let mut a_us = up_scales;
        let mut a_dp = down_packed;
        let mut a_ds = down_scales;
        let mut a_gsc = gsc;
        let mut a_usc = usc;
        let mut a_asc = asc;
        let mut a_out = mega_out;
        let mut a_h = hi;
        let mut a_inter = interi;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_xin as *mut _ as *mut c_void,
            &mut a_gp as *mut _ as *mut c_void,
            &mut a_gs as *mut _ as *mut c_void,
            &mut a_up as *mut _ as *mut c_void,
            &mut a_us as *mut _ as *mut c_void,
            &mut a_dp as *mut _ as *mut c_void,
            &mut a_ds as *mut _ as *mut c_void,
            &mut a_gsc as *mut _ as *mut c_void,
            &mut a_usc as *mut _ as *mut c_void,
            &mut a_asc as *mut _ as *mut c_void,
            &mut a_out as *mut _ as *mut c_void,
            &mut a_h as *mut _ as *mut c_void,
            &mut a_inter as *mut _ as *mut c_void,
        ];
        unsafe {
            result::launch_cooperative_kernel(
                mega,
                (coop_grid, 1, 1),
                (block, 1, 1),
                0,
                stream.cu_stream(),
                &mut params,
            )
        }
        .unwrap();
    };

    run_baseline();
    run_mega();
    runtime.synchronize().unwrap();

    let base_ms = median(
        (0..iters)
            .map(|_| time_ms(&mut || run_baseline()))
            .collect(),
    );
    runtime.synchronize().unwrap();
    let mega_ms = median((0..iters).map(|_| time_ms(&mut || run_mega())).collect());
    runtime.synchronize().unwrap();

    // --- standalone grid.sync barrier cost ---
    let bar_reps = 1000i32;
    let mut a_reps = bar_reps;
    let mut a_sink = sink;
    let mut run_barriers = || {
        let mut params: Vec<*mut c_void> = vec![
            &mut a_reps as *mut _ as *mut c_void,
            &mut a_sink as *mut _ as *mut c_void,
        ];
        unsafe {
            result::launch_cooperative_kernel(
                barrier,
                (coop_grid, 1, 1),
                (block, 1, 1),
                0,
                stream.cu_stream(),
                &mut params,
            )
        }
        .unwrap();
    };
    run_barriers();
    runtime.synchronize().unwrap();
    let bar_ms = median(
        (0..iters.min(50))
            .map(|_| time_ms(&mut || run_barriers()))
            .collect(),
    );
    runtime.synchronize().unwrap();
    let per_barrier_us = bar_ms * 1000.0 / bar_reps as f32;

    eprintln!(
        "[mk2] per-op baseline MLP (4 launches, grid=N):        {:.4} ms/layer-MLP",
        base_ms
    );
    eprintln!(
        "[mk2] multi-CTA cooperative mega MLP (1 coop launch):  {:.4} ms/layer-MLP",
        mega_ms
    );
    let recovered = (base_ms - mega_ms) / base_ms * 100.0;
    eprintln!(
        "[mk2] recovered fraction = {:.1}%  (mega/baseline = {:.3}x)",
        recovered,
        mega_ms / base_ms
    );
    eprintln!(
        "[mk2] grid.sync barrier cost = {:.3} us/barrier (full {}-CTA grid); mega pays 2/MLP",
        per_barrier_us, coop_grid
    );

    // Numerics: identical dequant + reduction order => byte-exact.
    let mut bh = vec![0u8; h * 4];
    let mut mh = vec![0u8; h * 4];
    unsafe {
        runtime.dtoh(&mut bh, base_out).unwrap();
        runtime.dtoh(&mut mh, mega_out).unwrap();
    }
    let rd = |b: &[u8], i: usize| {
        f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
    };
    let mut max_abs = 0.0f32;
    let mut max_ulp = 0i64;
    for i in 0..h {
        let (a, b) = (rd(&bh, i), rd(&mh, i));
        max_abs = max_abs.max((a - b).abs());
        max_ulp = max_ulp.max((a.to_bits() as i64 - b.to_bits() as i64).abs());
    }
    eprintln!(
        "[mk2] numerics mega-vs-baseline: max_abs={:.3e}, max_ulp={} (0 => byte-exact)",
        max_abs, max_ulp
    );

    unsafe {
        for p in [
            gate_packed,
            up_packed,
            down_packed,
            gate_scales,
            up_scales,
            down_scales,
            xin,
            gsc,
            usc,
            asc,
            base_out,
            mega_out,
            sink,
        ] {
            runtime.free_raw(p).ok();
        }
    }
    assert!(base_ms > 0.0 && mega_ms > 0.0);
}
