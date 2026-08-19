//! Shared **CPU-only** numerics helpers for the `com.microsoft::MatMulNBits`
//! int4 parity gate.
//!
//! This module holds the device-free machinery both halves of the gate depend
//! on: the deterministic problem generator ([`Int4Problem`]), the high-precision
//! `f64` dequant→GEMM oracle, an independent `f32` dequant reference, and the
//! justified parity envelope ([`Envelope`] / [`ParityReport`]).
//!
//! It lives in a `tests/` **subdirectory** (`tests/marlin_numerics/mod.rs`) so it
//! is compiled as a plain shared module — never as its own integration-test
//! target — and is pulled into both sibling targets via `mod marlin_numerics;`:
//!
//! * `matmul_nbits_marlin_oracle.rs` — the pure-CPU oracle self-checks that
//!   validate this math on any host, and
//! * `matmul_nbits_marlin_numerics.rs` — the GPU gate that runs a real int4
//!   kernel through the CUDA EP and measures it against this oracle.
//!
//! Splitting the CPU self-checks out of the GPU target is what lets the CUDA
//! test-honesty checker (`.github/scripts/verify_cuda_test_honesty.py`) see the
//! numerics target as purely-CUDA (all tests `ignore`d without `gpu-tests`)
//! while these oracle probes run on the CPU lane — see #1177.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use half::f16;

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
pub struct Int4Problem {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub block_size: usize,
    pub k_blocks: usize,
    pub blob_size: usize,
    pub zp_row_bytes: usize,
    pub scales_fp16: bool,
    /// fp16 activations (row-major `[M, K]`), the device input.
    pub activation_f16: Vec<f16>,
    /// `activation_f16` widened to f32 (bit-identical value) for the oracle.
    pub activation_ref: Vec<f32>,
    /// Packed int4 codes, two nibbles/byte, in the kernel's `[N, k_blocks, blob]` layout.
    pub packed: Vec<u8>,
    /// Per-weight int4 codes `0..=15` indexed `col * k + depth` (oracle-facing).
    pub quant: Vec<u8>,
    pub scale_f16: Vec<f16>,
    pub scale_f32: Vec<f32>,
    /// Scale value both paths use (fp16- or f32-rounded), indexed `col * k_blocks + block`.
    pub scale_ref: Vec<f32>,
    /// Per-`(col, block)` zero-point code (`8` when symmetric), oracle-facing.
    pub zp_codes: Vec<i32>,
    /// Packed asymmetric zero points; `None` for the symmetric (`zp = 8`) default.
    pub zp_packed: Option<Vec<u8>>,
}

impl Int4Problem {
    /// Build a deterministic instance. `block_size` must be a power of two `>= 16`
    /// and must divide `k`. When `asymmetric` is set, a non-uniform per-block int4
    /// zero point (packed two block-nibbles/byte exactly as the kernel unpacks) is
    /// generated; otherwise the symmetric `zp = 8` default is used.
    pub fn new(
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
    pub fn f64_oracle(&self) -> Vec<f64> {
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
    pub fn parity(&self, candidate: &[f32]) -> ParityReport {
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
pub struct ParityReport {
    /// Largest `|candidate - oracle|` over all outputs.
    pub max_abs: f64,
    /// Largest `|candidate - oracle| / max(|oracle|, conditioning floor)`.
    pub max_rel: f64,
    /// Largest `|oracle|` (output magnitude / conditioning proxy).
    pub max_out: f64,
    /// Every candidate output was finite.
    pub all_finite: bool,
}

impl ParityReport {
    pub fn compute(candidate: &[f32], oracle: &[f64]) -> Self {
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
    pub fn merge(&mut self, other: &ParityReport) {
        self.max_abs = self.max_abs.max(other.max_abs);
        self.max_rel = self.max_rel.max(other.max_rel);
        self.max_out = self.max_out.max(other.max_out);
        self.all_finite &= other.all_finite;
    }

    /// Assert this report falls inside the justified [`Envelope`], with a
    /// descriptive label for the failing case.
    pub fn assert_within(&self, label: &str) {
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
/// *Relative bound.* `5e-2` against the `REL_FLOOR` denominator isolates
/// per-element accuracy from output magnitude and matches the in-crate GEMV
/// parity guard's bound, so decode and prefill are held to the same standard.
#[derive(Clone, Copy, Debug)]
pub struct Envelope {
    pub abs_bound: f64,
    pub rel_bound: f64,
}

impl Envelope {
    pub fn for_output(max_out: f64) -> Self {
        Self {
            abs_bound: (max_out * 4e-3).max(4e-3),
            rel_bound: 5e-2,
        }
    }
}

// ---------------------------------------------------------------------------
// Group sizes + independent f32 dequant reference
// ---------------------------------------------------------------------------

/// Group sizes the quantized checkpoints in the fleet actually use.
pub const GROUP_SIZES: &[usize] = &[16, 32, 64, 128];

/// Independent f32 dequant→GEMM reference, deliberately written differently from
/// [`Int4Problem::f64_oracle`] (unpacks the *packed* bytes and *packed* zero
/// points rather than the decoded twins) so a bug in either encoding is caught.
pub fn f32_dequant_reference(p: &Int4Problem) -> Vec<f32> {
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
