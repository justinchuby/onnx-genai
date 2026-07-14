//! CPU GEMM backend selection (`docs/ORT2.md` §25.2 "CPU Backend Strategy").
//!
//! The hot f32 GEMM behind [`crate::kernels::matmul`] can be serviced by more
//! than one implementation. [`CpuBackend`] names the family of backends from
//! the ORT2 design and [`CpuBackend::auto_detect`] picks one at runtime:
//!
//! * On x86 / ARM-server targets we prefer **oneDNN** when it is compiled in
//!   (the non-default `onednn` cargo feature, statically linked in this crate).
//! * Everything else — and any build without the `onednn` feature — falls back
//!   to the **Generic** pure-Rust blocked GEMM, which compiles anywhere and is
//!   the correctness baseline.
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
    /// oneDNN (x86 + ARM server). Requires the `onednn` cargo feature; when that
    /// feature is off this variant is never selected.
    OneDnn,
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
    /// * Otherwise → `OneDnn` when [`has_onednn`] is true, else `Generic`.
    pub fn auto_detect() -> Self {
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
            if has_onednn() {
                Self::OneDnn
            } else {
                Self::Generic
            }
        }
    }
}

/// Whether the statically-linked oneDNN backend is compiled into this build.
///
/// oneDNN is linked in exactly when the non-default `onednn` cargo feature is
/// enabled, so "compiled in" ⇒ "available" (`docs/ORT2.md` §25.2).
#[inline]
pub fn has_onednn() -> bool {
    cfg!(feature = "onednn")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_onednn_matches_feature() {
        assert_eq!(has_onednn(), cfg!(feature = "onednn"));
    }

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
    fn auto_detect_tracks_onednn_feature() {
        let expected = if cfg!(feature = "onednn") {
            CpuBackend::OneDnn
        } else {
            CpuBackend::Generic
        };
        assert_eq!(CpuBackend::auto_detect(), expected);
    }
}
