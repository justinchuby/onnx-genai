//! Reusable **f64 dequant→GEMM numerics gate** for `com.microsoft::MatMulNBits`
//! int4, built so that *any* GEMM implementation — today's `gemm_f16_tiled`
//! prefill path and Deckard's forthcoming Marlin int4 GEMM (`squad/marlin-kernel`,
//! #957) — must pass the *same* oracle before it can ship.
//!
//! ## Why a dedicated gate
//! Marlin's weight relayout reorders the per-K partial sums, so a Marlin output
//! is **not** byte-exact against the current tiled kernel and cannot be validated
//! by a bit-for-bit diff. The only defensible ground truth is a high-precision
//! reference: dequantize the packed int4 weights to `f64` and run the GEMM in
//! `f64`. This module provides that oracle plus a justified relative+absolute
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

#![allow(clippy::too_many_arguments)]

use half::f16;
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
// Deterministic input generation (no external RNG crate)
// ---------------------------------------------------------------------------

/// Reproducible LCG identical in spirit to the in-crate parity harness so a
/// failure can be reproduced from `(seed, dims)` alone.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Uniform in `[-1, 1)`.
    fn signed(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        self.signed() * 0.5 + 0.5
    }
}

// ---------------------------------------------------------------------------
// Problem model + f64 oracle
// ---------------------------------------------------------------------------

/// A fully-materialized int4 `MatMulNBits` problem instance: fp16 activations
/// `[M, K]`, packed int4 weights `[N, k_blocks, block_size/2]`, per-`(col, block)`
/// scales rounded to the storage dtype, and an optional asymmetric per-block
/// zero-point tensor. Holds both the device-facing encodings (`packed`,
/// `scale_f16`/`scale_f32`, `zp_packed`) and the decoded twins (`quant`,
/// `scale_ref`, `zp_codes`) the f64 oracle consumes, guaranteeing both sides see
/// identical values.
struct Int4Problem {
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
    k_blocks: usize,
    blob_size: usize,
    zp_row_bytes: usize,
    scales_fp16: bool,
    /// fp16 activations (row-major `[M, K]`), the device input.
    activation_f16: Vec<f16>,
    /// `activation_f16` widened to f32 (bit-identical value) for the oracle.
    activation_ref: Vec<f32>,
    /// Packed int4 codes, two nibbles/byte, in the kernel's `[N, k_blocks, blob]` layout.
    packed: Vec<u8>,
    /// Per-weight int4 codes `0..=15` indexed `col * k + depth` (oracle-facing).
    quant: Vec<u8>,
    scale_f16: Vec<f16>,
    scale_f32: Vec<f32>,
    /// Scale value both paths use (fp16- or f32-rounded), indexed `col * k_blocks + block`.
    scale_ref: Vec<f32>,
    /// Per-`(col, block)` zero-point code (`8` when symmetric), oracle-facing.
    zp_codes: Vec<i32>,
    /// Packed asymmetric zero points; `None` for the symmetric (`zp = 8`) default.
    zp_packed: Option<Vec<u8>>,
}

impl Int4Problem {
    /// Build a deterministic instance. `block_size` must be a power of two `>= 16`
    /// and must divide `k`. When `asymmetric` is set, a non-uniform per-block int4
    /// zero point (packed two block-nibbles/byte exactly as the kernel unpacks) is
    /// generated; otherwise the symmetric `zp = 8` default is used.
    fn new(
        m: usize,
        k: usize,
        n: usize,
        block_size: usize,
        scales_fp16: bool,
        asymmetric: bool,
        seed: u64,
    ) -> Self {
        assert!(block_size >= 16 && block_size.is_power_of_two());
        assert_eq!(k % block_size, 0, "oracle requires block_size to divide K");
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);
        let mut rng = Lcg::new(seed);

        // fp16 activations + their exact f32 twin.
        let mut activation_f16 = vec![f16::ZERO; m * k];
        let mut activation_ref = vec![0.0f32; m * k];
        for (h, f) in activation_f16.iter_mut().zip(activation_ref.iter_mut()) {
            let v = f16::from_f32(rng.signed());
            *h = v;
            *f = v.to_f32();
        }

        // int4 quant codes 0..=15, packed two nibbles per byte.
        let mut quant = vec![0u8; n * k];
        for v in quant.iter_mut() {
            *v = (rng.unit() * 15.0).round().clamp(0.0, 15.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for col in 0..n {
            for block in 0..k_blocks {
                for pair in 0..blob_size {
                    let low = quant[col * k + block * block_size + pair * 2] & 0x0f;
                    let high = quant[col * k + block * block_size + pair * 2 + 1] & 0x0f;
                    packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                }
            }
        }

        // Zero points: symmetric zp=8 default, or explicit asymmetric per-block.
        let mut zp_codes = vec![8i32; n * k_blocks];
        let zp_packed = if asymmetric {
            let mut zp_packed = vec![0u8; n * zp_row_bytes];
            for code in zp_codes.iter_mut() {
                *code = (rng.unit() * 15.0).round().clamp(0.0, 15.0) as i32;
            }
            for col in 0..n {
                for block in 0..k_blocks {
                    let code = (zp_codes[col * k_blocks + block] & 0x0f) as u8;
                    let byte = &mut zp_packed[col * zp_row_bytes + block / 2];
                    if block & 1 == 0 {
                        *byte = (*byte & 0xf0) | code;
                    } else {
                        *byte = (*byte & 0x0f) | (code << 4);
                    }
                }
            }
            Some(zp_packed)
        } else {
            None
        };

        // Per-(col, block) scales, rounded to the storage dtype so both paths use
        // the same value. Range mirrors the in-crate harness (~0.015..0.025).
        let mut scale_f16 = vec![f16::ZERO; n * k_blocks];
        let mut scale_f32 = vec![0.0f32; n * k_blocks];
        let mut scale_ref = vec![0.0f32; n * k_blocks];
        for i in 0..n * k_blocks {
            let raw = 0.015 + 0.01 * rng.unit();
            if scales_fp16 {
                let h = f16::from_f32(raw);
                scale_f16[i] = h;
                scale_ref[i] = h.to_f32();
            } else {
                scale_f32[i] = raw;
                scale_ref[i] = raw;
            }
        }

        Self {
            m,
            k,
            n,
            block_size,
            k_blocks,
            blob_size,
            zp_row_bytes,
            scales_fp16,
            activation_f16,
            activation_ref,
            packed,
            quant,
            scale_f16,
            scale_f32,
            scale_ref,
            zp_codes,
            zp_packed,
        }
    }

    /// **The ground truth.** Dequantize each int4 code to `f64` as
    /// `(code - zero_point) * scale` (scale pre-rounded to its fp16/f32 storage
    /// value) and accumulate `sum_k activation_f64 * weight_f64` in `f64`.
    /// Returns a row-major `[M, N]` output. This is the reference every candidate
    /// GEMM — tiled today, Marlin tomorrow — is measured against.
    fn f64_oracle(&self) -> Vec<f64> {
        let mut out = vec![0.0f64; self.m * self.n];
        for row in 0..self.m {
            for col in 0..self.n {
                let mut acc = 0.0f64;
                for block in 0..self.k_blocks {
                    let scale = self.scale_ref[col * self.k_blocks + block] as f64;
                    let zp = self.zp_codes[col * self.k_blocks + block];
                    for within in 0..self.block_size {
                        let depth = block * self.block_size + within;
                        let code = self.quant[col * self.k + depth] as i32 - zp;
                        acc +=
                            self.activation_ref[row * self.k + depth] as f64 * code as f64 * scale;
                    }
                }
                out[row * self.n + col] = acc;
            }
        }
        out
    }

    /// Compare a candidate GEMM output (already widened to f32) against the f64
    /// oracle, returning the parity metrics. `candidate` is row-major `[M, N]`.
    fn parity(&self, candidate: &[f32]) -> ParityReport {
        let oracle = self.f64_oracle();
        ParityReport::compute(candidate, &oracle)
    }
}

// ---------------------------------------------------------------------------
// Parity metrics + justified tolerance envelope
// ---------------------------------------------------------------------------

/// Absolute floor on the relative-error denominator, so a degenerate tiny
/// problem (peak output ~0) does not divide by ~0.
const REL_FLOOR_ABS: f64 = 1e-1;

/// Conditioning-aware floor on the relative-error denominator, expressed as a
/// fraction of the problem's **peak** output magnitude. A dot product whose true
/// value is far below the operator's output scale is dominated by cancellation
/// (`|sum a_i w_i|` ≪ `sum |a_i w_i|`); its fp16 round-off is inherently large in
/// *relative* terms even though the *absolute* error is one fp16 ULP of the peak.
/// Such columns are governed by the absolute bound, not the relative one, so any
/// output below `3%` of the peak floors the denominator here. `3%` keeps ~3× of
/// margin on the worst measured cancellation column (glm-4 down-projection,
/// K=13696, asymmetric-zp: abs error 2.1e-2 on a 0.26-magnitude column against a
/// 46-magnitude peak).
const REL_FLOOR_FRAC: f64 = 3e-2;

/// Parity result of one candidate vs the f64 oracle.
#[derive(Clone, Copy, Debug, Default)]
struct ParityReport {
    /// Largest `|candidate - oracle|` over all outputs.
    max_abs: f64,
    /// Largest `|candidate - oracle| / max(|oracle|, conditioning floor)`.
    max_rel: f64,
    /// Largest `|oracle|` (output magnitude / conditioning proxy).
    max_out: f64,
    /// Every candidate output was finite.
    all_finite: bool,
}

impl ParityReport {
    fn compute(candidate: &[f32], oracle: &[f64]) -> Self {
        assert_eq!(candidate.len(), oracle.len());
        // Peak output magnitude sets the fp16 ULP scale and the conditioning
        // floor, so it must be known before the relative ratio is formed.
        let max_out = oracle.iter().fold(0.0f64, |m, &o| m.max(o.abs()));
        let rel_floor = REL_FLOOR_ABS.max(REL_FLOOR_FRAC * max_out);
        let mut report = ParityReport {
            max_out,
            all_finite: true,
            ..Default::default()
        };
        for (&c, &o) in candidate.iter().zip(oracle.iter()) {
            let c = c as f64;
            if !c.is_finite() {
                report.all_finite = false;
            }
            let abs = (c - o).abs();
            report.max_abs = report.max_abs.max(abs);
            report.max_rel = report.max_rel.max(abs / o.abs().max(rel_floor));
        }
        report
    }

    /// Fold another report in (worst-case across a sweep).
    fn merge(&mut self, other: &ParityReport) {
        self.max_abs = self.max_abs.max(other.max_abs);
        self.max_rel = self.max_rel.max(other.max_rel);
        self.max_out = self.max_out.max(other.max_out);
        self.all_finite &= other.all_finite;
    }

    /// Assert this report falls inside the justified [`Envelope`], with a
    /// descriptive label for the failing case.
    fn assert_within(&self, label: &str) {
        let env = Envelope::for_output(self.max_out);
        eprintln!(
            "[marlin-numerics] {label}: max_abs={:.3e} max_rel={:.3e} max_out={:.3e} \
             abs_bound={:.3e} rel_bound={:.3e}",
            self.max_abs, self.max_rel, self.max_out, env.abs_bound, env.rel_bound
        );
        assert!(
            self.all_finite,
            "{label}: candidate produced a non-finite output"
        );
        assert!(
            self.max_abs <= env.abs_bound,
            "{label}: abs error {:.3e} exceeds justified bound {:.3e} (max_out={:.3e})",
            self.max_abs,
            env.abs_bound,
            self.max_out
        );
        assert!(
            self.max_rel <= env.rel_bound,
            "{label}: rel error {:.3e} exceeds justified bound {:.3e}",
            self.max_rel,
            env.rel_bound
        );
    }
}

/// The **justified parity envelope** an int4 GEMM with fp16 output must satisfy.
///
/// *Absolute bound.* The output is stored fp16, whose ULP is `2^-11 ≈ 4.9e-4` of
/// a value's magnitude, so the absolute-error floor is set by the largest output
/// component. The weights are also dequantized through fp16 (another `2^-11`
/// relative per term) and the K-length reduction runs in fp32; a partial-sum
/// **relayout** (Marlin) re-associates that reduction, adding fp32 round-off
/// drift `~ K * eps_f32`. Over the deepest realistic `K` (~13696) that drift is
/// `< 2e-3` relative — an order of magnitude under the fp16 term — so
/// `max_out * 4e-3` (≈ 8 fp16 ULP of headroom) with a `4e-3` floor comfortably
/// covers both the tiled baseline *and* a re-associated Marlin reduction.
///
/// *Relative bound.* `5e-2` against the [`REL_FLOOR`] denominator isolates
/// per-element accuracy from output magnitude and matches the in-crate GEMV
/// parity guard's bound, so decode and prefill are held to the same standard.
#[derive(Clone, Copy, Debug)]
struct Envelope {
    abs_bound: f64,
    rel_bound: f64,
}

impl Envelope {
    fn for_output(max_out: f64) -> Self {
        Self {
            abs_bound: (max_out * 4e-3).max(4e-3),
            rel_bound: 5e-2,
        }
    }
}

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
    let kernel = ep
        .get_kernel(model.graph.node(node), &[], 1)
        .expect("get_kernel for MatMulNBits int4");

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

fn maybe_cuda() -> Option<CudaExecutionProvider> {
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => Some(ep),
        _ => None,
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

/// Group sizes the quantized checkpoints in the fleet actually use.
const GROUP_SIZES: &[usize] = &[16, 32, 64, 128];

// ---------------------------------------------------------------------------
// Pure-CPU unit tests (no GPU): oracle self-consistency + tolerance model
// ---------------------------------------------------------------------------

/// Independent f32 dequant→GEMM reference, deliberately written differently from
/// [`Int4Problem::f64_oracle`] (unpacks the *packed* bytes and *packed* zero
/// points rather than the decoded twins) so a bug in either encoding is caught.
fn f32_dequant_reference(p: &Int4Problem) -> Vec<f32> {
    let mut out = vec![0.0f32; p.m * p.n];
    for row in 0..p.m {
        for col in 0..p.n {
            let mut acc = 0.0f32;
            for depth in 0..p.k {
                let block = depth / p.block_size;
                let within = depth % p.block_size;
                let byte = p.packed[(col * p.k_blocks + block) * p.blob_size + within / 2];
                let code = if within.is_multiple_of(2) {
                    byte & 0x0f
                } else {
                    byte >> 4
                } as i32;
                let zp = p.zp_packed.as_ref().map_or(8, |zp| {
                    let byte = zp[col * p.zp_row_bytes + block / 2];
                    (if block.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    }) as i32
                });
                let weight = (code - zp) as f32 * p.scale_ref[col * p.k_blocks + block];
                acc += p.activation_ref[row * p.k + depth] * weight;
            }
            out[row * p.n + col] = acc;
        }
    }
    out
}

#[test]
fn oracle_matches_independent_reference_symmetric() {
    // Small dims keep this a fast CPU-only check while still exercising multiple
    // blocks, several columns, and M>1.
    for &block_size in GROUP_SIZES {
        let p = Int4Problem::new(4, 256, 12, block_size, true, false, 0xA53F_0001);
        let oracle = p.f64_oracle();
        let reference = f32_dequant_reference(&p);
        for (o, r) in oracle.iter().zip(reference.iter()) {
            // f64 oracle vs f32 reference: agreement to the f32 reduction floor
            // proves the packed encoding and the decoded twins describe the same
            // weights (a wiring bug would diverge by whole quanta, not ULPs).
            assert!(
                (o - *r as f64).abs() <= o.abs().max(1.0) * 1e-4,
                "block_size={block_size}: oracle={o:e} reference={r:e}"
            );
        }
    }
}

#[test]
fn oracle_matches_independent_reference_asymmetric() {
    for &block_size in GROUP_SIZES {
        let p = Int4Problem::new(3, 128, 9, block_size, false, true, 0xA53F_0002);
        let oracle = p.f64_oracle();
        let reference = f32_dequant_reference(&p);
        for (o, r) in oracle.iter().zip(reference.iter()) {
            assert!(
                (o - *r as f64).abs() <= o.abs().max(1.0) * 1e-4,
                "asymmetric block_size={block_size}: oracle={o:e} reference={r:e}"
            );
        }
    }
}

#[test]
fn oracle_is_exact_on_a_hand_checkable_case() {
    // K = block_size = 16, N = 1, M = 1, symmetric (zp = 8). Recompute the single
    // output from the decoded codes/scale and require bit-for-bit agreement with
    // the oracle's own f64 accumulation (same association) — this pins the oracle
    // arithmetic, not just its self-consistency.
    let p = Int4Problem::new(1, 16, 1, 16, false, false, 0xA53F_0003);
    let mut expected = 0.0f64;
    for depth in 0..16 {
        let code = p.quant[depth] as i32 - 8;
        expected += p.activation_ref[depth] as f64 * code as f64 * p.scale_ref[0] as f64;
    }
    assert_eq!(p.f64_oracle()[0], expected);
}

#[test]
fn envelope_scales_with_output_magnitude_and_has_a_floor() {
    let big = Envelope::for_output(1000.0);
    assert!(
        (big.abs_bound - 4.0).abs() < 1e-9,
        "abs bound tracks max_out * 4e-3"
    );
    assert_eq!(big.rel_bound, 5e-2);
    let tiny = Envelope::for_output(0.0);
    assert!(
        (tiny.abs_bound - 4e-3).abs() < 1e-12,
        "abs bound keeps a 4e-3 floor"
    );
}

#[test]
fn parity_flags_a_perturbed_candidate() {
    let p = Int4Problem::new(2, 64, 8, 32, true, false, 0xA53F_0004);
    let oracle = p.f64_oracle();
    let good: Vec<f32> = oracle.iter().map(|&o| o as f32).collect();
    let clean = p.parity(&good);
    assert!(clean.all_finite);
    let env = Envelope::for_output(clean.max_out);
    assert!(
        clean.max_abs <= env.abs_bound,
        "an f32-cast oracle must pass its own gate"
    );

    // Inject a gross error into one element; the gate must catch it.
    let mut bad = good.clone();
    bad[0] += (clean.max_out as f32).max(1.0);
    let dirty = p.parity(&bad);
    let env = Envelope::for_output(dirty.max_out);
    assert!(
        dirty.max_abs > env.abs_bound || dirty.max_rel > env.rel_bound,
        "a corrupted candidate must fail the gate"
    );
}

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
    let Some(ep) = maybe_cuda() else {
        eprintln!("[marlin-numerics] skipping: CUDA runtime unavailable");
        return;
    };
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
    let Some(ep) = maybe_cuda() else {
        eprintln!("[marlin-numerics] skipping: CUDA runtime unavailable");
        return;
    };
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
