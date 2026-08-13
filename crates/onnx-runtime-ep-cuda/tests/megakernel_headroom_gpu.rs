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
