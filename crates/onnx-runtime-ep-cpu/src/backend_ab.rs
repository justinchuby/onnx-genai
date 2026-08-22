//! Same-binary A/B: run a kernel family's native route and the MLAS *reference*
//! route in one process, on one host, under one load.
//!
//! # This module never runs in a shipped build's hot path
//!
//! The production CPU EP is native and links no MLAS. Every `*_mlas` entry
//! point below is `#[cfg(feature = "mlas")]`, and that feature is not in
//! `default` — it exists so research and benchmarks can reach the reference
//! deliberately, never so a wheel or plugin can reach it by accident.
//!
//! # Why this is a module and not a test helper
//!
//! `docs/performance/ABSORBING_MLAS.md` records the method that works for
//! replacing a reference route with a native one: keep **both** implementations
//! in **one binary** behind an explicit toggle, then measure the gap. Without
//! that, the gap is asserted rather than measured, and a native "win" that is
//! actually a loss ships unnoticed.
//!
//! So the A/B entry points are part of the crate, not scaffolding inside a test
//! file: `tests/native_vs_mlas_differential.rs` uses them to prove the two
//! routes agree, and `benches/native_vs_mlas.rs` uses the same functions to
//! prove a native replacement is actually faster before it graduates.
//!
//! Every function here runs **inside this execution provider**. None of them
//! calls, links, or falls back to ORT's built-in CPU execution provider.

use onnx_runtime_ep_api::Result;

use crate::backend::CpuBackend;
use crate::dispatch_ledger::{Backend, KernelFamily};
use crate::kernels::{matmul, sdpa, simd_activations, softmax};

/// Whether this build linked the MLAS reference at all. False in every shipped
/// build. When false, the `*_mlas` entry points return `None` and the
/// differential tests degrade to native-vs-reference self-consistency rather
/// than silently passing on nothing.
pub const fn mlas_available() -> bool {
    cfg!(feature = "mlas")
}

// ─── f32 GEMM ───────────────────────────────────────────────────────────────

/// The f32 GEMM backends this build can actually run, native ones first.
///
/// The first entry is always a native route, so a caller can use
/// `gemm_backends()[0]` as the correctness baseline without a `cfg`.
pub fn gemm_backends() -> Vec<CpuBackend> {
    // aarch64 has neither route to add today -- the SIMD GEMM is x86-only and
    // MLAS's GEMM is reached through the x86-64 backend enum -- so `mut` is
    // genuinely unused there rather than mistakenly so.
    #[cfg_attr(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        allow(unused_mut)
    )]
    let mut backends = vec![CpuBackend::Generic];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::backend::has_simd_x86() {
        backends.push(CpuBackend::SimdX86);
    }
    #[cfg(all(feature = "mlas", target_arch = "x86_64"))]
    backends.push(CpuBackend::Mlas);
    backends
}

/// Run `c = a @ b` (row-major, `a` is `m x k`, `b` is `k x n`) on an explicit
/// backend, bypassing [`CpuBackend::auto_detect`].
///
/// This is the A/B toggle for [`KernelFamily::MatMulF32`]: pass
/// [`CpuBackend::Generic`] for the native baseline and [`CpuBackend::Mlas`] for
/// the MLAS route, in the same process.
pub fn gemm_f32(
    backend: CpuBackend,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    matmul::gemm_with_backend(backend, a, b, c, m, k, n)
}

/// How `backend` reads in the dispatch ledger.
pub fn gemm_ledger_backend(backend: CpuBackend) -> Backend {
    matmul::ledger_backend(backend)
}

// ─── Softmax ────────────────────────────────────────────────────────────────

/// Row-softmax `n` rows of `d` elements in place, native route.
pub fn softmax_rows_native(data: &mut [f32], n: usize, d: usize) {
    softmax::softmax_rows_serial_native(data, n, d);
}

/// Row-softmax `n` rows of `d` elements in place, MLAS route.
///
/// Returns `false` when this build has no MLAS, so the caller can report "not
/// compared" instead of reporting a pass it never ran.
pub fn softmax_rows_mlas(data: &mut [f32], n: usize, d: usize) -> bool {
    #[cfg(feature = "mlas")]
    {
        mlas_sys::compute_softmax_in_place(data, n, d);
        true
    }
    #[cfg(not(feature = "mlas"))]
    {
        let _ = (data, n, d);
        false
    }
}

// ─── Transcendental activations ─────────────────────────────────────────────

/// `Erf` over a contiguous f32 slice, native route.
pub fn erf_native(input: &[f32], output: &mut [f32]) {
    simd_activations::erf_f32_slice_native(input, output);
}

/// `Erf` over a contiguous f32 slice, MLAS route. See [`softmax_rows_mlas`] for
/// the `bool`.
pub fn erf_mlas(input: &[f32], output: &mut [f32]) -> bool {
    #[cfg(feature = "mlas")]
    {
        simd_activations::erf_f32_slice_mlas(input, output);
        true
    }
    #[cfg(not(feature = "mlas"))]
    {
        let _ = (input, output);
        false
    }
}

/// Exact (`approximate="none"`) `Gelu`, native route.
pub fn gelu_native(input: &[f32], output: &mut [f32]) {
    simd_activations::erf_gelu_f32_slice_native(input, output);
}

/// Exact `Gelu`, MLAS route with our special-value repair.
pub fn gelu_mlas(input: &[f32], output: &mut [f32]) -> bool {
    #[cfg(feature = "mlas")]
    {
        simd_activations::erf_gelu_f32_slice_mlas(input, output);
        true
    }
    #[cfg(not(feature = "mlas"))]
    {
        let _ = (input, output);
        false
    }
}

// ─── Scaled dot-product attention ───────────────────────────────────────────

/// One SDPA problem, in the shape the A/B entry points take.
///
/// This mirrors [`crate::kernels::sdpa::SdpaTensors`] plus the causal/scale
/// configuration, so a caller can describe a case without depending on the
/// kernel module's hook traits.
#[derive(Clone, Copy, Debug)]
pub struct SdpaCase {
    /// Batch size.
    pub batch: usize,
    /// Query head count.
    pub num_heads: usize,
    /// Key/value head count; equal to `num_heads` for plain MHA, smaller for GQA.
    pub num_kv_heads: usize,
    /// Query sequence length; `1` is decode.
    pub q_seq: usize,
    /// Total key/value sequence length.
    pub kv_seq: usize,
    /// Q/K head dimension.
    pub head_size: usize,
    /// V head dimension; may differ from `head_size`.
    pub v_head_size: usize,
    /// Whether the causal (`unidirectional`) mask applies.
    pub causal: bool,
}

impl SdpaCase {
    /// Element count of `q`.
    pub const fn q_len(&self) -> usize {
        self.batch * self.num_heads * self.q_seq * self.head_size
    }

    /// Element count of `k`.
    pub const fn k_len(&self) -> usize {
        self.batch * self.num_kv_heads * self.kv_seq * self.head_size
    }

    /// Element count of `v`.
    pub const fn v_len(&self) -> usize {
        self.batch * self.num_kv_heads * self.kv_seq * self.v_head_size
    }

    /// Element count of the context output `y`.
    pub const fn y_len(&self) -> usize {
        self.batch * self.num_heads * self.q_seq * self.v_head_size
    }

    fn config(&self) -> sdpa::SdpaConfig {
        sdpa::SdpaConfig {
            scale: sdpa::ScaleMode::PostDot(1.0 / (self.head_size as f32).sqrt()),
            softcap: None,
            causal: self.causal,
            past_seq: self.kv_seq.saturating_sub(self.q_seq),
            causal_fill: f32::MIN,
        }
    }
}

struct ZeroBias;
impl sdpa::AttnBias for ZeroBias {
    fn at(&self, _b: usize, _head: usize, _i: usize, _j: usize) -> f32 {
        0.0
    }
    fn is_identity(&self) -> bool {
        true
    }
}

struct ZeroMask;
impl sdpa::KeyMask for ZeroMask {
    fn at(&self, _b: usize, _i: usize, _j: usize) -> f32 {
        0.0
    }
    fn is_identity(&self) -> bool {
        true
    }
}

/// Scaled dot-product attention, native route — the one a shipped build runs.
///
/// `y` is written in full: every one of [`SdpaCase::y_len`] elements is
/// assigned, never accumulated into. Callers prefill it with a poison value and
/// check none survives; see `tests/native_vs_mlas_differential.rs`.
pub fn sdpa_native(case: &SdpaCase, q: &[f32], k: &[f32], v: &[f32], y: &mut [f32]) {
    let tensors = sdpa::SdpaTensors {
        q,
        k,
        v,
        batch: case.batch,
        num_heads: case.num_heads,
        num_kv_heads: case.num_kv_heads,
        q_seq: case.q_seq,
        kv_seq: case.kv_seq,
        head_size: case.head_size,
        v_head_size: case.v_head_size,
    };
    sdpa::sdpa_f32_native(&tensors, &case.config(), &ZeroBias, &ZeroMask, y);
}

/// Scaled dot-product attention, MLAS reference route. See
/// [`softmax_rows_mlas`] for the `bool`.
///
/// This is the route [`KernelFamily::AttentionTranspose`]'s plan entry means by
/// "MLAS SGEMM for the QK^T and PV products", and the route that carried the
/// dropped-output-partition defect in #1685. Its output contract is identical
/// to [`sdpa_native`]: the whole of `y` is written, because both inner GEMMs
/// run with `beta = 0`.
pub fn sdpa_mlas(case: &SdpaCase, q: &[f32], k: &[f32], v: &[f32], y: &mut [f32]) -> bool {
    #[cfg(feature = "mlas")]
    {
        let tensors = sdpa::SdpaTensors {
            q,
            k,
            v,
            batch: case.batch,
            num_heads: case.num_heads,
            num_kv_heads: case.num_kv_heads,
            q_seq: case.q_seq,
            kv_seq: case.kv_seq,
            head_size: case.head_size,
            v_head_size: case.v_head_size,
        };
        sdpa::sdpa_f32_mlas(&tensors, &case.config(), &ZeroBias, &ZeroMask, y);
        true
    }
    #[cfg(not(feature = "mlas"))]
    {
        let _ = (case, q, k, v, y);
        false
    }
}

/// The families this module can A/B today. The migration doc's graduation rule
/// requires a same-binary A/B, so a family absent here cannot graduate yet.
pub const AB_COVERED: &[KernelFamily] = &[
    KernelFamily::MatMulF32,
    KernelFamily::Softmax,
    KernelFamily::Activations,
    KernelFamily::AttentionTranspose,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic values in `[-1, 1)`, so an A/B failure reproduces.
    fn ab_values(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state ^= state >> 33;
                state = state.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn gemm_backends_lead_with_a_native_baseline() {
        let backends = gemm_backends();
        assert!(!backends.is_empty());
        assert_eq!(
            gemm_ledger_backend(backends[0]),
            Backend::Native,
            "the first A/B backend must be native, so callers get a baseline without a cfg"
        );
    }

    #[test]
    fn mlas_route_is_offered_exactly_when_it_is_linked() {
        let has_mlas_backend = gemm_backends()
            .iter()
            .any(|b| gemm_ledger_backend(*b) == Backend::Mlas);
        #[cfg(all(feature = "mlas", target_arch = "x86_64"))]
        assert!(has_mlas_backend, "mlas is linked but not offered for A/B");
        #[cfg(not(all(feature = "mlas", target_arch = "x86_64")))]
        assert!(
            !has_mlas_backend,
            "offered an MLAS backend that is not linked"
        );
    }

    #[test]
    fn mlas_entry_points_report_availability_honestly() {
        let mut data = vec![0.5f32; 64];
        let ran = softmax_rows_mlas(&mut data, 4, 16);
        assert_eq!(ran, mlas_available());
    }

    /// Every family this module claims to A/B must actually have both halves
    /// wired up, and the MLAS half must report itself present exactly when the
    /// reference is linked. Without this, `AB_COVERED` is a list a family could
    /// be added to without ever being comparable — and the graduation rule in
    /// `docs/performance/CPU_MLAS_MIGRATION.md` reads that list.
    #[test]
    fn every_ab_covered_family_has_both_halves() {
        assert!(!AB_COVERED.is_empty());
        for family in AB_COVERED {
            assert!(
                KernelFamily::ALL.contains(family),
                "{family} is A/B-covered but absent from the ledger's families"
            );
            let ran_mlas = match family {
                KernelFamily::MatMulF32 => {
                    let (m, k, n) = (2usize, 3usize, 4usize);
                    let a = vec![0.25f32; m * k];
                    let b = vec![0.5f32; k * n];
                    let mut c = vec![0.0f32; m * n];
                    let native = gemm_backends()[0];
                    gemm_f32(native, &a, &b, &mut c, m, k, n).expect("native gemm must run");
                    assert!(
                        c.iter().all(|v| (*v - 0.375).abs() < 1e-6),
                        "the native A/B gemm must compute a@b, got {c:?}"
                    );
                    gemm_backends()
                        .iter()
                        .any(|b| gemm_ledger_backend(*b) == Backend::Mlas)
                }
                KernelFamily::Softmax => {
                    let mut native = vec![1.0f32, 2.0, 3.0, 4.0];
                    softmax_rows_native(&mut native, 1, 4);
                    let sum: f32 = native.iter().sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-5,
                        "the native A/B softmax must normalize, got {sum}"
                    );
                    let mut probe = native.clone();
                    softmax_rows_mlas(&mut probe, 1, 4)
                }
                KernelFamily::Activations => {
                    let input = [-1.0f32, 0.0, 1.0, 2.0];
                    let mut out = [0.0f32; 4];
                    erf_native(&input, &mut out);
                    assert!(
                        out[1] == 0.0 && out[2] > 0.8 && out[0] == -out[2],
                        "the native A/B erf must be odd and finite, got {out:?}"
                    );
                    let mut gelu = [0.0f32; 4];
                    gelu_native(&input, &mut gelu);
                    assert!(
                        gelu[1] == 0.0 && gelu[3] > 1.9,
                        "the native A/B gelu must match 0.5x(1+erf(x/sqrt2)), got {gelu:?}"
                    );
                    let mut probe = [0.0f32; 4];
                    erf_mlas(&input, &mut probe) && gelu_mlas(&input, &mut probe)
                }
                KernelFamily::AttentionTranspose => {
                    let case = SdpaCase {
                        batch: 1,
                        num_heads: 2,
                        num_kv_heads: 1,
                        q_seq: 3,
                        kv_seq: 5,
                        head_size: 4,
                        v_head_size: 4,
                        causal: true,
                    };
                    let q = ab_values(case.q_len(), 0x5D_0001);
                    let k = ab_values(case.k_len(), 0x5D_0002);
                    let v = ab_values(case.v_len(), 0x5D_0003);
                    let mut native = vec![f32::NAN; case.y_len()];
                    sdpa_native(&case, &q, &k, &v, &mut native);
                    assert!(
                        native.iter().all(|value| value.is_finite()),
                        "the native A/B sdpa must write every output element, got {native:?}"
                    );
                    let mut probe = vec![f32::NAN; case.y_len()];
                    sdpa_mlas(&case, &q, &k, &v, &mut probe)
                }
                other => panic!(
                    "{other} was added to AB_COVERED without an A/B entry point in this test"
                ),
            };
            assert_eq!(
                ran_mlas,
                mlas_available(),
                "{family} must offer its MLAS half exactly when the reference is linked"
            );
        }
    }

    /// A duplicate backend would double-count in every benchmark row and make
    /// the A/B table claim a comparison it never ran.
    #[test]
    fn gemm_backends_are_distinct() {
        let backends = gemm_backends();
        let mut unique = backends.clone();
        unique.sort_by_key(|b| format!("{b:?}"));
        unique.dedup_by_key(|b| format!("{b:?}"));
        assert_eq!(
            unique.len(),
            backends.len(),
            "duplicate A/B backend offered"
        );
    }
}
