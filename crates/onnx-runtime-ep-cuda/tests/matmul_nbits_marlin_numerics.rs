//! **GPU parity gate** for `com.microsoft::MatMulNBits` int4 — runs a real CUDA
//! kernel through the execution provider and measures it against the f64
//! dequant→GEMM oracle.
//!
//! The device-free machinery this gate depends on (the [`Int4Problem`] generator,
//! the `f64` oracle, and the justified [`Envelope`]/[`ParityReport`] tolerance
//! model) lives in the shared `marlin_numerics/mod.rs` module and is exercised
//! independently by the pure-CPU self-checks in `matmul_nbits_marlin_oracle.rs`.
//! That split keeps *this* target purely-CUDA: every test here is `ignore`d
//! without the `gpu-tests` feature, so the CUDA test-honesty checker
//! (`.github/scripts/verify_cuda_test_honesty.py`) sees it as a clean CUDA target
//! (all ignored on a CPU host, none passing). See #1177.
//!
//! ## Why a dedicated gate
//! Marlin's weight relayout reorders the per-K partial sums, so a Marlin output
//! is **not** byte-exact against the current tiled kernel and cannot be validated
//! by a bit-for-bit diff. The only defensible ground truth is a high-precision
//! reference: dequantize the packed int4 weights to `f64` and run the GEMM in
//! `f64`. The shared module provides that oracle plus a justified relative+absolute
//! tolerance so Chew (numerics reviewer) can sign off Marlin *apples-to-apples*
//! against the tiled baseline.
//!
//! ## Interface contract (Marlin-ready)
//! The gate talks to the kernel purely through **op semantics** — an ONNX
//! `MatMulNBits` node (`K`, `N`, `bits`, `block_size` attributes; packed-B,
//! scales, optional zero-points inputs) executed via the public
//! [`ExecutionProvider`] API. It never reaches into the kernel's internal weight
//! layout. Once Marlin is wired into the same `MatMulNBits` dispatch, the exact
//! same [`run_matmul_nbits_f16`] driver validates it with **zero changes**. If
//! Marlin is exercised out-of-band (feature flag / separate buffer), Deckard or
//! Chew can instead feed any candidate output slice straight into
//! [`Int4Problem::parity`].
//!
//! ## What is held identical between candidate and oracle
//! Both sides consume the **same fp16-rounded activations** and the **same
//! scale value rounded to its storage dtype**, so the measured residual isolates
//! only the kernel's accumulation precision + fp16 *output* rounding — never
//! input quantization, which both sides share. This mirrors the in-crate GEMV
//! parity harness (`kernels/matmul_nbits.rs` `run_parity_dims_block`) and the
//! asymmetric-zp / #928 fold-scale "validated against a dequant reference to
//! tolerance" convention documented at the top of that file.

mod marlin_numerics;

use half::f16;
use marlin_numerics::{GROUP_SIZES, Int4Problem, ParityReport};
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

// ---------------------------------------------------------------------------
// GPU driver — op-semantics interface (Marlin-ready)
// ---------------------------------------------------------------------------

/// Run one int4 `MatMulNBits` instance through the CUDA execution provider using
/// only op-level semantics, returning the fp16 output widened to f32. This is the
/// single entry point Deckard's Marlin kernel and Chew's sign-off share: it
/// builds the ONNX node, calls [`ExecutionProvider::get_kernel`], and executes —
/// whatever kernel the dispatch selects (tiled today, Marlin once merged).
fn run_matmul_nbits_f16(ep: &CudaExecutionProvider, p: &Int4Problem) -> Vec<f32> {
    let bits = 4usize;
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);

    let a = graph.create_named_value("A", DataType::Float16, static_shape([p.m, p.k]));
    let b = graph.create_named_value(
        "B",
        DataType::Uint8,
        static_shape([p.n, p.k_blocks, p.blob_size]),
    );
    let scales_dtype = if p.scales_fp16 {
        DataType::Float16
    } else {
        DataType::Float32
    };
    let scales_value =
        graph.create_named_value("scales", scales_dtype, static_shape([p.n, p.k_blocks]));
    for value in [a, b, scales_value] {
        graph.add_input(value);
    }
    let mut node_inputs = vec![Some(a), Some(b), Some(scales_value)];
    if p.zp_packed.is_some() {
        let zp = graph.create_named_value(
            "zero_points",
            DataType::Uint8,
            static_shape([p.n, p.zp_row_bytes]),
        );
        graph.add_input(zp);
        node_inputs.push(Some(zp));
    }
    let output = graph.create_named_value("Y", DataType::Float16, static_shape([p.m, p.n]));
    let mut node = Node::new(NodeId(0), "MatMulNBits", node_inputs, vec![output]);
    node.domain = "com.microsoft".into();
    node.attributes
        .insert("K".into(), Attribute::Int(p.k as i64));
    node.attributes
        .insert("N".into(), Attribute::Int(p.n as i64));
    node.attributes
        .insert("bits".into(), Attribute::Int(bits as i64));
    node.attributes
        .insert("block_size".into(), Attribute::Int(p.block_size as i64));
    let node = graph.insert_node(node);
    graph.add_output(output);

    let model = Model::new(&graph);
    let mut kernel = ep
        .get_kernel(model.graph.node(node), &[], 1)
        .expect("get_kernel for MatMulNBits int4");
    // Every operand in this op-semantics harness is a runtime graph input. In
    // particular B is mutable and must never enter Marlin's pointer-keyed
    // initializer repack cache.
    kernel.set_constant_inputs(&[false; 4]);

    // Host tensors (raw bytes) for each input in node order.
    let mut inputs: Vec<(DataType, Vec<usize>, Vec<u8>)> = vec![
        (
            DataType::Float16,
            vec![p.m, p.k],
            as_bytes(&p.activation_f16),
        ),
        (
            DataType::Uint8,
            vec![p.n, p.k_blocks, p.blob_size],
            p.packed.clone(),
        ),
        if p.scales_fp16 {
            (
                DataType::Float16,
                vec![p.n, p.k_blocks],
                as_bytes(&p.scale_f16),
            )
        } else {
            (
                DataType::Float32,
                vec![p.n, p.k_blocks],
                as_bytes(&p.scale_f32),
            )
        },
    ];
    if let Some(zp) = &p.zp_packed {
        inputs.push((DataType::Uint8, vec![p.n, p.zp_row_bytes], zp.clone()));
    }

    let runtime = ep.runtime();
    let device = ep.device_id();
    let mut buffers = Vec::<DeviceBuffer>::new();
    for (_, _, bytes) in &inputs {
        let buffer = ep.allocate(bytes.len(), 256).expect("allocate input");
        // SAFETY: allocation size equals the source byte length.
        unsafe {
            runtime
                .htod(bytes, cuptr(buffer.as_ptr()))
                .expect("htod input")
        };
        buffers.push(buffer);
    }
    let strides: Vec<Vec<i64>> = inputs
        .iter()
        .map(|(_, shape, _)| compute_contiguous_strides(shape))
        .collect();
    let views: Vec<TensorView> = inputs
        .iter()
        .zip(&buffers)
        .zip(&strides)
        .map(|(((dtype, shape, _), buffer), strides)| {
            TensorView::new(DevicePtr(buffer.as_ptr()), *dtype, shape, strides, device)
        })
        .collect();

    let output_len = p.m * p.n;
    let mut output_buffer = ep.allocate(output_len * 2, 256).expect("allocate output");
    let output_shape = [p.m, p.n];
    let output_strides = compute_contiguous_strides(&output_shape);
    let output_view = TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        DataType::Float16,
        &output_shape,
        &output_strides,
        device,
    );
    // Pre-zero the output so a kernel that fails to write some elements surfaces
    // as a clean parity failure rather than reading stale pool memory (which
    // would masquerade as a passing run whenever the allocator happens to hand
    // back zeroed pages). SAFETY: the buffer holds `output_len` fp16 values.
    unsafe {
        runtime
            .htod(&vec![0u8; output_len * 2], cuptr(output_buffer.as_ptr()))
            .expect("zero output");
    }
    kernel
        .execute(&views, &mut [output_view])
        .expect("execute MatMulNBits int4");
    runtime.synchronize().expect("synchronize after execute");

    let mut bytes = vec![0u8; output_len * 2];
    // SAFETY: output allocation holds `output_len` fp16 values.
    unsafe {
        runtime
            .dtoh(&mut bytes, cuptr(output_buffer.as_ptr()))
            .expect("dtoh output");
    }
    drop(views);
    for buffer in buffers {
        ep.deallocate(buffer).expect("deallocate input");
    }
    ep.deallocate(output_buffer).expect("deallocate output");

    bytes
        .chunks_exact(2)
        .map(|value| f16::from_bits(u16::from_ne_bytes(value.try_into().unwrap())).to_f32())
        .collect()
}

fn as_bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: reinterpreting a POD slice as raw bytes for a host->device copy.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
            .to_vec()
    }
}

fn require_cuda() -> CudaExecutionProvider {
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => panic!("gpu-tests requires a CUDA device: {error}"),
        Err(_) => panic!("gpu-tests requires a CUDA device; CUDA initialization panicked"),
    }
}

// ---------------------------------------------------------------------------
// Realistic projection shapes (glm-4-9b + Qwen2.5) driving the gate
// ---------------------------------------------------------------------------

/// `(label, K, N)` for the attention + MLP projections the decode/prefill path
/// actually hits. GLM-4-9B: hidden 4096, q_hidden 4096, kv_hidden 256, FFN
/// intermediate ~13696 (`docs/research/speculative-capture-feasibility.md`).
/// Qwen2.5-1.5B: gate/up K=1536,N=8960 and down K=8960,N=1536
/// (`kernels/matmul_nbits.rs` qwen dims guard). All K are multiples of 128, so
/// every `block_size ∈ {16,32,64,128}` divides them.
const PROJECTION_SHAPES: &[(&str, usize, usize)] = &[
    ("glm4-attn-qkv", 4096, 4096),
    ("glm4-attn-kv", 4096, 256),
    ("glm4-attn-o", 4096, 4096),
    ("glm4-mlp-gate-up", 4096, 13696),
    ("glm4-mlp-down", 13696, 4096),
    ("qwen2.5-1.5b-gate-up", 1536, 8960),
    ("qwen2.5-1.5b-down", 8960, 1536),
];

/// Prefill / speculative-verify batch heights the tiled (and future Marlin) M>1
/// path must serve. `M=1` is the decode GEMV; `M=8` is the settled speculative
/// verify width (`decisions.md` #949).
const BATCH_HEIGHTS: &[usize] = &[1, 2, 4, 8, 16, 32];

// ---------------------------------------------------------------------------
// GPU gate: baseline the CURRENT tiled/GEMV path (the Marlin yardstick)
// ---------------------------------------------------------------------------

/// Baseline the **current** int4 path (decode GEMV at M=1, `gemm_f16_tiled`
/// prefill at M>1 — the kernel Marlin replaces) against the f64 oracle across the
/// full matrix of `{group size} × {M} × {symmetric, asymmetric} × {fp16, fp32
/// scales}` at a representative projection shape, then report the observed
/// worst-case error so Marlin's tolerance can be compared apples-to-apples.
///
/// Requires a real CUDA device; without the `gpu-tests` feature it is reported
/// as ignored (CPU-only CI). Pins nothing — the caller sets
/// `CUDA_VISIBLE_DEVICES` to a verified-idle GPU.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn current_path_matches_f64_oracle_group_size_sweep() {
    let ep = require_cuda();
    // K=4096, N=896: a whole multiple of the 8-column CTA width and divisible by
    // every group size, deep enough for the K-reduction drift to show.
    let (k, n) = (4096usize, 896usize);
    let mut overall = ParityReport {
        all_finite: true,
        ..Default::default()
    };
    let mut seed = 0xC0DE_0000u64;
    for &block_size in GROUP_SIZES {
        for &m in BATCH_HEIGHTS {
            for &asymmetric in &[false, true] {
                for &scales_fp16 in &[false, true] {
                    seed = seed.wrapping_add(0x9E37_79B9);
                    let p = Int4Problem::new(m, k, n, block_size, scales_fp16, asymmetric, seed);
                    let candidate = run_matmul_nbits_f16(&ep, &p);
                    let report = p.parity(&candidate);
                    let label = format!(
                        "current M={m} K={k} N={n} bs={block_size} \
                         {}zp {} scales",
                        if asymmetric { "asym-" } else { "sym-" },
                        if scales_fp16 { "fp16" } else { "fp32" }
                    );
                    report.assert_within(&label);
                    overall.merge(&report);
                }
            }
        }
    }
    eprintln!(
        "[marlin-numerics] CURRENT-PATH BASELINE (group-size sweep, K={k} N={n}): \
         max_abs={:.3e} max_rel={:.3e} max_out={:.3e}",
        overall.max_abs, overall.max_rel, overall.max_out
    );
    assert!(overall.all_finite);
}

/// Baseline the current path across the **real projection shapes** (glm-4-9b +
/// Qwen2.5) at block-32 (the fleet's dominant group size) for both decode (M=1)
/// and prefill/verify (M=8) — the exact `(K, N, M)` combinations Marlin must
/// serve without regressing accuracy.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn current_path_matches_f64_oracle_projection_shapes() {
    let ep = require_cuda();
    let mut overall = ParityReport {
        all_finite: true,
        ..Default::default()
    };
    let mut seed = 0xB10C_0000u64;
    for &(name, k, n) in PROJECTION_SHAPES {
        for &m in &[1usize, 8] {
            for &asymmetric in &[false, true] {
                seed = seed.wrapping_add(0x9E37_79B9);
                let p = Int4Problem::new(m, k, n, 32, true, asymmetric, seed);
                let candidate = run_matmul_nbits_f16(&ep, &p);
                let report = p.parity(&candidate);
                let label = format!(
                    "current {name} M={m} K={k} N={n} bs=32 {}zp",
                    if asymmetric { "asym-" } else { "sym-" }
                );
                report.assert_within(&label);
                overall.merge(&report);
            }
        }
    }
    eprintln!(
        "[marlin-numerics] CURRENT-PATH BASELINE (projection shapes, bs=32): \
         max_abs={:.3e} max_rel={:.3e} max_out={:.3e}",
        overall.max_abs, overall.max_rel, overall.max_out
    );
    assert!(overall.all_finite);
}

// ---------------------------------------------------------------------------
// GPU gate: the opt-in fp16 (ORT-matching) wide GEMV (ONNX_GENAI_GEMV_FP16=1)
// ---------------------------------------------------------------------------

/// glm's real block-128 decode (M=1) projection shapes. block-128 is the group
/// size where the wide multicol GEMV (and its fp16 sibling) is active; the KV
/// projection (N=256) is excluded because it can route through the split-K
/// entry, not the wide/fp16 entry this gate targets.
const GLM_DECODE_SHAPES: &[(&str, usize, usize)] = &[
    ("glm4-attn-qkv", 4096, 4096),
    ("glm4-attn-o", 4096, 4096),
    ("glm4-mlp-gate-up", 4096, 13696),
    ("glm4-mlp-down", 13696, 4096),
];

/// **Accuracy gate for the opt-in fp16 (ORT-matching) wide GEMV.**
///
/// The fp16 kernel (`matmul_nbits_gemv_f16_general_bs_wide_multicol_fp16`,
/// selected by `ONNX_GENAI_GEMV_FP16=1`) runs the per-chunk MAC in fp16
/// `__hfma2` and keeps the *entire per-lane K reduction* in fp16 `__half2`
/// accumulators — exactly matching ORT's `MatMulFloat4BitsKernelM1` (the
/// equal-conditions fp16-vs-fp16 path); fp32 is used ONLY in the final
/// cross-lane warp-shuffle reduction. Because each lane strides K by 32 and
/// folds only a handful of chunks, the fp16 accumulation is a wide, shallow tree
/// (depth ~tens, not K=4096..13696), so almost no mantissa is lost. It is
/// **not** byte-identical to the fp32 path by construction, so it ships opt-in
/// gated on *this* accuracy bound rather than bit-identity.
///
/// The bar is deliberately the **same justified [`Envelope`]** the reviewed fp32
/// int4 path must satisfy (`current_path_matches_f64_oracle_*`): the fp16 path is
/// held to the identical numeric tolerance — no weakening — on glm's real
/// block-128 decode shapes with both symmetric and asymmetric zero points and
/// both scale dtypes. For transparency the test also runs the fp32 path on the
/// same problems and reports the fp16-vs-fp32 oracle-error ratio, and asserts the
/// fp16 output actually diverges from fp32 (proving the fp16 kernel — not the
/// fp32 fallback — was exercised).
///
/// `ONNX_GENAI_GENERAL_SPLITK=0` pins the wide/multicol entry so the selection is
/// deterministic regardless of the device's SM count.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn fp16_mixed_gemv_matches_f64_oracle_glm_decode() {
    let ep = require_cuda();
    // Pin the wide/multicol entry (not split-K) so both the fp32 reference and
    // the fp16 candidate route through the same wide GEMV family on every GPU.
    // SAFETY: single-threaded test body; restored before returning.
    unsafe { std::env::set_var("ONNX_GENAI_GENERAL_SPLITK", "0") };

    let mut fp16_overall = ParityReport {
        all_finite: true,
        ..Default::default()
    };
    let mut fp32_overall = ParityReport {
        all_finite: true,
        ..Default::default()
    };
    let mut worst_ratio = 0.0f64;
    let mut any_fp16_divergence = false;
    let mut seed = 0xF16A_0000u64;
    for &(name, k, n) in GLM_DECODE_SHAPES {
        for &asymmetric in &[false, true] {
            for &scales_fp16 in &[false, true] {
                seed = seed.wrapping_add(0x9E37_79B9);
                // glm decode: block-128, M=1.
                let p = Int4Problem::new(1, k, n, 128, scales_fp16, asymmetric, seed);

                // fp32 multicol reference (env off) — the precise int4 kernel.
                unsafe { std::env::remove_var("ONNX_GENAI_GEMV_FP16") };
                let fp32 = run_matmul_nbits_f16(&ep, &p);
                let fp32_report = p.parity(&fp32);

                // fp16 mixed candidate (env on).
                unsafe { std::env::set_var("ONNX_GENAI_GEMV_FP16", "1") };
                let fp16 = run_matmul_nbits_f16(&ep, &p);
                unsafe { std::env::remove_var("ONNX_GENAI_GEMV_FP16") };
                let fp16_report = p.parity(&fp16);

                // Sentinel: the fp16 kernel must actually have been selected —
                // its reordered/hfma2 arithmetic differs from fp32 in the low
                // bits. If the outputs were bit-identical the fp16 entry was not
                // exercised and the gate would be vacuous.
                if fp16
                    .iter()
                    .zip(&fp32)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    any_fp16_divergence = true;
                }

                let label = format!(
                    "fp16 {name} M=1 K={k} N={n} bs=128 {}zp {} scales",
                    if asymmetric { "asym-" } else { "sym-" },
                    if scales_fp16 { "fp16" } else { "fp32" }
                );
                // HARD GATE: fp16 must satisfy the SAME reviewed envelope as fp32.
                fp16_report.assert_within(&label);

                let ratio = fp16_report.max_rel / fp32_report.max_rel.max(1e-12);
                worst_ratio = worst_ratio.max(ratio);
                eprintln!(
                    "[marlin-numerics] {label}: fp16 max_rel={:.3e} vs fp32-path \
                     max_rel={:.3e} (ratio {:.2}x)",
                    fp16_report.max_rel, fp32_report.max_rel, ratio
                );
                fp16_overall.merge(&fp16_report);
                fp32_overall.merge(&fp32_report);
            }
        }
    }
    unsafe { std::env::remove_var("ONNX_GENAI_GENERAL_SPLITK") };

    eprintln!(
        "[marlin-numerics] FP16-MIXED GLM DECODE (block-128, M=1): \
         fp16 max_abs={:.3e} max_rel={:.3e} | fp32-path max_abs={:.3e} max_rel={:.3e} | \
         max_out={:.3e} | worst fp16/fp32 rel-error ratio={:.2}x",
        fp16_overall.max_abs,
        fp16_overall.max_rel,
        fp32_overall.max_abs,
        fp32_overall.max_rel,
        fp16_overall.max_out,
        worst_ratio
    );
    assert!(
        fp16_overall.all_finite,
        "fp16 candidate produced non-finite output"
    );
    assert!(
        any_fp16_divergence,
        "fp16 output was bit-identical to fp32 on every shape — the fp16 kernel \
         was not exercised (dispatch did not select the fp16 entry); the gate \
         would be vacuous"
    );
}
