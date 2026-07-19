//! CPU GEMM backend selection (`docs/ORT2.md` §25.2 "CPU Backend Strategy").
//!
//! The hot f32 GEMM behind [`crate::kernels::matmul`] can be serviced by more
//! than one implementation. [`CpuBackend`] names the family of backends from
//! the ORT2 design and [`CpuBackend::auto_detect`] picks one at runtime:
//!
//! * On x86-64 hosts with AVX2 + FMA (detected at runtime) we use the built-in
//!   **`SimdX86`** MLAS-style packed SIMD f32 GEMM — the default fast path with
//!   no extra dependency and no cargo feature required.
//! * With the `mlas` Cargo feature, `NXRT_CPU_GEMM_BACKEND=mlas` explicitly
//!   selects the vendored, single-threaded MLAS f32 GEMM on x86-64. It is never
//!   auto-selected while Rayon-to-MLAS threadpool bridging is pending.
//! * Everything else falls back to the **Generic** pure-Rust blocked GEMM,
//!   which compiles anywhere and is the correctness baseline.
//!
//! The `Xnnpack` (Android) and `Accelerate` (Apple) variants are present for
//! design fidelity with §25.2 but are not wired to kernels yet; they degrade to
//! the Generic path. Nothing above the [`onnx_runtime_ep_api::Kernel`] trait
//! observes which backend ran — the choice is an internal perf detail.

/// The CPU GEMM backend family, per `docs/ORT2.md` §25.2.
///
/// Selection is done by [`CpuBackend::auto_detect`]; callers should not hardcode
/// a variant so that the same binary adapts to the host it runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuBackend {
    /// Built-in MLAS-style packed SIMD f32 GEMM for x86-64 with AVX2 + FMA.
    /// Selected at runtime via `is_x86_feature_detected!` — no cargo feature and
    /// no external dependency. Falls back to [`CpuBackend::Generic`] arithmetic
    /// on hosts without AVX2/FMA.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SimdX86,
    /// Vendored MLAS f32 SGEMM for x86-64. Available only with the `mlas`
    /// Cargo feature and selected explicitly with `NXRT_CPU_GEMM_BACKEND=mlas`;
    /// it is single-threaded until MLAS threadpool bridging is implemented.
    #[cfg(feature = "mlas")]
    Mlas,
    /// XNNPACK (Android mobile). Design placeholder — currently routes to
    /// [`CpuBackend::Generic`] arithmetic.
    #[cfg(target_os = "android")]
    Xnnpack,
    /// Apple Accelerate (macOS / iOS). Design placeholder — currently routes to
    /// [`CpuBackend::Generic`] arithmetic.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Accelerate,
    /// Pure-Rust blocked, register-tiled, rayon-parallelized GEMM. Always
    /// available; the correctness baseline every other backend must match.
    Generic,
}

impl CpuBackend {
    /// Pick the best available backend for the current target and build, per
    /// `docs/ORT2.md` §25.2.
    ///
    /// * Android → `Xnnpack` (placeholder; Generic arithmetic today).
    /// * macOS / iOS → `Accelerate` (placeholder; Generic arithmetic today).
    /// * Otherwise → `SimdX86` when the host is x86-64 with AVX2 + FMA; else
    ///   `Generic`.
    pub fn auto_detect() -> Self {
        if let Some(backend) = Self::from_env_override(std::env::var("NXRT_CPU_GEMM_BACKEND").ok())
        {
            return backend;
        }

        #[cfg(target_os = "android")]
        {
            Self::Xnnpack
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            Self::Accelerate
        }
        #[cfg(all(
            not(target_os = "android"),
            not(target_os = "macos"),
            not(target_os = "ios")
        ))]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if has_simd_x86() {
                    return Self::SimdX86;
                }
            }
            Self::Generic
        }
    }

    /// Resolve the optional `NXRT_CPU_GEMM_BACKEND` value. Unsupported choices
    /// intentionally fall through to ordinary host auto-detection.
    fn from_env_override(value: Option<String>) -> Option<Self> {
        let value = value?;
        if value.eq_ignore_ascii_case("generic") {
            return Some(Self::Generic);
        }
        if value.eq_ignore_ascii_case("simd") {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            return Some(Self::simd_x86_or_generic(has_simd_x86()));
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            return Some(Self::Generic);
        }
        if value.eq_ignore_ascii_case("mlas") {
            #[cfg(all(feature = "mlas", target_arch = "x86_64"))]
            return Some(Self::Mlas);
        }
        None
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn simd_x86_or_generic(supported: bool) -> Self {
        if supported {
            Self::SimdX86
        } else {
            Self::Generic
        }
    }
}

/// Whether the host CPU supports the AVX2 + FMA instructions the built-in
/// [`CpuBackend::SimdX86`] microkernel requires. Runtime-detected so the same
/// binary stays correct on older x86 CPUs (falling back to `Generic`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub fn has_simd_x86() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_detect_is_stable() {
        // Deterministic for a given build/target: two calls agree.
        assert_eq!(CpuBackend::auto_detect(), CpuBackend::auto_detect());
    }

    #[cfg(all(
        not(target_os = "android"),
        not(target_os = "macos"),
        not(target_os = "ios")
    ))]
    #[test]
    fn auto_detect_tracks_simd_x86_support() {
        let expected = {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if has_simd_x86() {
                    CpuBackend::SimdX86
                } else {
                    CpuBackend::Generic
                }
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                CpuBackend::Generic
            }
        };
        assert_eq!(CpuBackend::auto_detect(), expected);
    }

    #[test]
    fn backend_env_override_is_case_insensitive() {
        assert_eq!(
            CpuBackend::from_env_override(Some("GeNeRiC".into())),
            Some(CpuBackend::Generic)
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert_eq!(
            CpuBackend::from_env_override(Some("SIMD".into())),
            Some(CpuBackend::simd_x86_or_generic(has_simd_x86()))
        );
        #[cfg(all(feature = "mlas", target_arch = "x86_64"))]
        assert_eq!(
            CpuBackend::from_env_override(Some("mLaS".into())),
            Some(CpuBackend::Mlas)
        );
        assert_eq!(CpuBackend::from_env_override(Some("unknown".into())), None);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn forced_simd_falls_back_to_generic_without_required_cpu_features() {
        assert_eq!(CpuBackend::simd_x86_or_generic(false), CpuBackend::Generic);
    }
}
