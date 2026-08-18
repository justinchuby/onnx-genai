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
use crate::kernels::{matmul, simd_activations, softmax};

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

/// The families this module can A/B today. The migration doc's graduation rule
/// requires a same-binary A/B, so a family absent here cannot graduate yet.
pub const AB_COVERED: &[KernelFamily] = &[
    KernelFamily::MatMulF32,
    KernelFamily::Softmax,
    KernelFamily::Activations,
];

#[cfg(test)]
mod tests {
    use super::*;

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
}
