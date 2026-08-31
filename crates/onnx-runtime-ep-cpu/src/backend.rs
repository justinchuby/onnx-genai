//! CPU GEMM backend selection (`docs/architecture/ORT2.md` §25.2 "CPU Backend Strategy").
//!
//! The hot f32 GEMM behind [`crate::kernels::matmul`] can be serviced by more
//! than one implementation. [`CpuBackend`] names the family of backends from
//! the ORT2 design and [`CpuBackend::auto_detect`] picks one at runtime:
//!
//! * On x86-64 hosts with AVX2 + FMA (detected at runtime) we use the built-in
//!   **`SimdX86`** MLAS-style packed SIMD f32 GEMM — the default fast path with
//!   no extra dependency and no cargo feature required.
//! * With the `mlas` Cargo feature, `NXRT_CPU_GEMM_BACKEND=mlas` explicitly
//!   selects the vendored, **multi-threaded** MLAS f32 GEMM on x86-64. MLAS
//!   does its own cache-aware tile partitioning and dispatches the tiles across
//!   the process Rayon pool (see `mlas-sys`), so it honours the same thread
//!   budget as `SimdX86`/`Generic` without oversubscribing.
//! * Everything else falls back to the **Generic** pure-Rust blocked GEMM,
//!   which compiles anywhere and is the correctness baseline.
//!
//! Half-precision MatMul/Gemm uses a separate portable blocked `f16`/`bf16`
//! path with `f32` accumulation. It is backend-independent today; bf16 can
//! additionally select a runtime-gated AVX-512 BF16 microkernel.
//!
//! The `Xnnpack` (Android) and `Accelerate` (Apple) variants are present for
//! design fidelity with §25.2 but are not wired to kernels yet; they degrade to
//! the Generic path. Nothing above the [`onnx_runtime_ep_api::Kernel`] trait
//! observes which backend ran — the choice is an internal perf detail.

/// The CPU GEMM backend family, per `docs/architecture/ORT2.md` §25.2.
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
    /// Cargo feature. Multi-threaded: MLAS partitions the GEMM and runs the
    /// tiles on the process Rayon pool.
    ///
    /// This is the **default** on x86-64 whenever the feature is compiled in --
    /// [`CpuBackend::auto_detect`] returns it without needing
    /// `NXRT_CPU_GEMM_BACKEND=mlas`, which only forces it. Kernels that gate
    /// behaviour on `== Mlas` (for example the `f16` prefill widening in
    /// `matmul`) are therefore live by default, not dead code.
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
    /// `docs/architecture/ORT2.md` §25.2.
    ///
    /// * Android → `Xnnpack` (placeholder; Generic arithmetic today).
    /// * macOS / iOS → `Accelerate` (placeholder; Generic arithmetic today).
    /// * Otherwise → vendored **`Mlas`** when the `mlas` Cargo feature is
    ///   compiled on x86-64 (multi-threaded, and the fastest measured f32 GEMM
    ///   path — see `docs/BENCH_CPU_F32_GEMM.md`); else the built-in `SimdX86`
    ///   MLAS-style microkernel when the host is x86-64 with AVX2 + FMA; else
    ///   `Generic`.
    ///
    /// Flipping the f32 default to MLAS matters because dense-f32 decode is
    /// dominated (>95%) by MatMul, and MLAS's cache-aware tiling + threading
    /// beats the built-in `SimdX86` microkernel by a wide margin (measured 3-4x
    /// on Gemma-2-2B f32). The choice is host/build-derived only (no per-model
    /// or per-op branching), so it generalizes across f32 models.
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
            // Prefer the vendored, multi-threaded MLAS SGEMM when it is compiled
            // in on x86-64: it is the fastest f32 GEMM available here and MLAS
            // does its own runtime ISA dispatch (with a scalar fallback), so it
            // is safe on any x86-64 host regardless of AVX2/FMA availability.
            #[cfg(all(feature = "mlas", target_arch = "x86_64"))]
            {
                Self::Mlas
            }
            // Without the MLAS backend, fall back to the built-in AVX2/FMA
            // microkernel on capable x86 hosts, else the portable Generic path.
            #[cfg(not(all(feature = "mlas", target_arch = "x86_64")))]
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
    #[cfg(test)]
    if forced_no_simd_x86() {
        return false;
    }
    detected_simd_x86()
}

/// The raw CPUID answer, with the test override deliberately *not* applied.
///
/// Split out because a test that skips itself has to distinguish "this host has
/// no AVX2" from "something switched AVX2 off underneath me": the first is a
/// legitimate skip, the second is the whole of #1817's instance 2. Asking
/// [`has_simd_x86`] cannot tell them apart, because it answers `false` to both.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) fn detected_simd_x86() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

/// The test-only override that forces [`has_simd_x86`] to report `false` so the
/// `Generic` fallback can be exercised on a host that does have AVX2.
#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) const FORCE_NO_SIMD_X86_ENV: &str = "ONNX_RUNTIME_EP_CPU_FORCE_NO_SIMD_X86";

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn forced_no_simd_x86() -> bool {
    std::env::var(FORCE_NO_SIMD_X86_ENV).as_deref() == Ok("1")
}

/// The variable a CI lane sets to declare that this crate's AVX2 differential
/// falsifiers must actually execute there.
///
/// Modelled on `NXRT_REQUIRE_PLACEMENT_TESTS` in [`crate::core_topology`], for
/// the same reason and with the same asymmetry: no developer's laptop is
/// obliged to have AVX2, but the x86 lanes we point this crate at are, and
/// until something says so their absence is indistinguishable from a pass.
#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) const REQUIRE_SIMD_X86_ENV: &str = "NXRT_REQUIRE_SIMD_X86_TESTS";

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn simd_x86_tests_required() -> bool {
    std::env::var(REQUIRE_SIMD_X86_ENV).as_deref() == Ok("1")
}

/// The policy behind [`require_simd_x86`], split from the environment reads so
/// the mutation tests can drive every branch without mutating process-global
/// state that parallel tests share -- the same reason
/// [`crate::core_topology::topology_or_fail_closed`] takes its input as an
/// argument.
///
/// `detected` is the raw CPUID answer and `forced_off` the override, because
/// the two failure modes warrant opposite treatment: a host without AVX2 is
/// entitled to skip, whereas an override that silently converts eleven
/// differential tests into green no-ops is the defect being guarded against.
#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn simd_x86_or_fail_closed(
    detected: bool,
    forced_off: bool,
    required: bool,
    what: &str,
) -> Result<(), String> {
    if detected && !forced_off {
        return Ok(());
    }
    assert!(
        !required,
        "{}",
        if forced_off {
            format!(
                "{REQUIRE_SIMD_X86_ENV}=1 and {FORCE_NO_SIMD_X86_ENV}=1 are both set, so `{what}` \
                 would return without comparing the AVX2 kernel against anything and still \
                 report success. These two settings are a contradiction: one lane declares the \
                 AVX2 differential tests mandatory, the other switches the AVX2 path off. Set \
                 the override on a lane that does not require these tests."
            )
        } else {
            format!(
                "{REQUIRE_SIMD_X86_ENV}=1 but this host reports no AVX2+FMA, so `{what}` cannot \
                 exercise the SIMD kernel it exists to falsify and would pass without testing \
                 anything. Point this lane at an AVX2 runner or stop requiring the SIMD tests \
                 on it."
            )
        }
    );
    Err(if forced_off {
        format!("{FORCE_NO_SIMD_X86_ENV}=1 switched the AVX2 path off")
    } else {
        "this host reports no AVX2+FMA".to_string()
    })
}

/// AVX2 capability for the differential kernel tests: fails closed on a lane
/// that declared them mandatory, and skips with a stated reason elsewhere.
///
/// Returns `true` when the caller may proceed. The eleven sites in
/// `kernels::x86_sgemm` previously opened with a bare `if !has_simd_x86() {
/// return; }`, which is a silent pass -- and one environment variable turns all
/// eleven into green no-ops at once, including the sole remaining falsifier for
/// #1809's int4 packing change.
#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn require_simd_x86(what: &str) -> bool {
    match simd_x86_or_fail_closed(
        detected_simd_x86(),
        forced_no_simd_x86(),
        simd_x86_tests_required(),
        what,
    ) {
        Ok(()) => true,
        Err(reason) => {
            eprintln!("skipping {what}: {reason}");
            false
        }
    }
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
            // MLAS, when compiled in on x86-64, is the preferred f32 default and
            // wins regardless of AVX2/FMA (it does its own ISA dispatch).
            #[cfg(all(feature = "mlas", target_arch = "x86_64"))]
            {
                CpuBackend::Mlas
            }
            #[cfg(all(
                not(all(feature = "mlas", target_arch = "x86_64")),
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
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

    /// The positive arm, in the sense #1173 established: a guard that only ever
    /// declines proves nothing, because "it declined" and "it is broken" are the
    /// same observation. This is the direction that must *permit* work.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn an_avx2_host_may_run_the_differential_tests() {
        assert!(simd_x86_or_fail_closed(true, false, true, "w").is_ok());
        assert!(simd_x86_or_fail_closed(true, false, false, "w").is_ok());
    }

    /// The negative arm. Before this, the eleven sites in `kernels::x86_sgemm`
    /// opened with a bare `return`, so a lane that lost AVX2 reported eleven
    /// passes having compared nothing.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    #[should_panic(expected = "this host reports no AVX2+FMA")]
    fn a_lane_requiring_the_simd_tests_fails_closed_when_the_host_lacks_avx2() {
        let _ = simd_x86_or_fail_closed(false, false, true, "int4_dequant_panel");
    }

    /// The override is the sharper half of #1817's instance 2: it is reachable
    /// on a host that *does* have AVX2, so it converts the whole differential
    /// suite into no-ops without any hardware change to notice.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    #[should_panic(expected = "are both set")]
    fn a_lane_requiring_the_simd_tests_fails_closed_when_the_override_switched_it_off() {
        let _ = simd_x86_or_fail_closed(true, true, true, "int4_dequant_panel");
    }

    /// A host genuinely without AVX2 is entitled to skip -- but the skip has to
    /// be *stated*, and it has to say which of the two causes it was, since the
    /// remedies are opposite: buy a different runner, or stop setting a variable.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn an_unrequired_lane_skips_with_a_reason_that_names_the_cause() {
        let host = simd_x86_or_fail_closed(false, false, false, "w").unwrap_err();
        let over = simd_x86_or_fail_closed(true, true, false, "w").unwrap_err();
        assert!(host.contains("no AVX2+FMA"), "{host}");
        assert!(over.contains(FORCE_NO_SIMD_X86_ENV), "{over}");
        assert_ne!(
            host, over,
            "the two skip causes must be distinguishable; conflating them is what let the \
             override hide behind 'this host has no AVX2'"
        );
    }

    /// Totality: `Ok` must be reachable only with AVX2 present and the override
    /// clear. Written as an exhaustive sweep of the three-bit input space so a
    /// future edit cannot open a fourth way through without failing here.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn only_a_genuinely_available_avx2_permits_the_differential_tests() {
        for detected in [false, true] {
            for forced in [false, true] {
                let permitted = simd_x86_or_fail_closed(detected, forced, false, "w").is_ok();
                assert_eq!(
                    permitted,
                    detected && !forced,
                    "detected={detected} forced={forced}"
                );
            }
        }
    }

    /// The anti-vacuity guard for the env plumbing itself, mirroring
    /// `core_topology`'s `placement_capabilities_are_present_when_the_lane_requires_them`.
    ///
    /// The five tests above drive [`simd_x86_or_fail_closed`] with hard-coded
    /// booleans, which proves the *policy* but not that anything reads the
    /// variable CI sets. Make [`simd_x86_tests_required`] return `false`
    /// unconditionally and the whole guard reverts to fail-open -- and no test
    /// above notices, because they pass `required` explicitly, while the green
    /// lane never reaches the fail-closed branch on a host that has AVX2. That
    /// is this PR's own defect class re-entering through its own plumbing, so
    /// it gets a named test rather than transitive coverage through eleven
    /// differential tests in another module.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn simd_capabilities_are_present_when_the_lane_requires_them() {
        if !simd_x86_tests_required() {
            eprintln!(
                "{REQUIRE_SIMD_X86_ENV} is unset, so AVX2 is not required here; the differential \
                 tests in kernels::matmul::x86_sgemm may be skipping"
            );
            return;
        }
        assert!(
            detected_simd_x86(),
            "{REQUIRE_SIMD_X86_ENV}=1 but CPUID reports no AVX2+FMA on this runner, so every \
             AVX2 differential test in this crate would skip. Point this lane at an AVX2 runner \
             or stop requiring the SIMD tests on it."
        );
        assert!(
            !forced_no_simd_x86(),
            "{REQUIRE_SIMD_X86_ENV}=1 but {FORCE_NO_SIMD_X86_ENV}=1 is set in this lane's \
             environment, which switches the AVX2 path off and would skip every differential \
             test that requires it. These two settings are a contradiction."
        );
        assert!(require_simd_x86("this self-check"));
    }
}
