//! CPU-only guards for the content-preserving granule transition.
//!
//! These validate [`verify_safe_point`]'s rejection rules, which are pure
//! predicate logic over [`ResizeSafePoint`] and need no CUDA driver. They live
//! in a target *without* the `_gpu` suffix on purpose.
//!
//! `.github/scripts/verify_cuda_test_honesty.py` classifies every `*_gpu`
//! target as a device test and requires its tests to be **ignored, not passed**
//! on a CPU-only runner -- that is how the suite is stopped from reporting
//! green for a GPU it never touched. A genuinely CPU-only test placed in a
//! `_gpu` target therefore fails the lane, and the script's own comment records
//! the intended escape hatch: target naming, so that "a genuinely CPU-only
//! target is not policed as a device test merely because it lives in a CUDA
//! crate".
//!
//! These three arrived in `content_preserving_transition_gpu.rs` (#1836) and
//! ran on every CPU-only CI run, which is what made that lane red. Silencing
//! them with `#[ignore]` would have greened the lane by deleting the coverage.
//! Moving them keeps them running -- but only because the CUDA lane gained an
//! explicit `--test content_preserving_transition` step alongside the two
//! other CPU-only targets in this crate. Without that step this target would
//! be compiled and never executed: the honesty script skips non-`_gpu`
//! targets, and `workspace_test_packages.py` deny-lists this crate from every
//! offline lane. The device tests they were mixed in with stay behind, all
//! `#[ignore]`d.

use onnx_runtime_ep_api::ResizeSafePoint;
use onnx_runtime_ep_cuda::granule_transition::verify_safe_point;

/// A capture in progress means the graph may replay the mapping we are about to
/// change, so the transition must refuse to start.
#[test]
fn safe_point_rejects_capturing_state() {
    let unsafe_point = ResizeSafePoint {
        capturing: true,
        ..ResizeSafePoint::default()
    };
    let sp = verify_safe_point(unsafe_point);
    assert!(
        sp.is_err(),
        "verify_safe_point must reject a capturing safe point"
    );
}

/// A live routed residency guard is holding a device pointer into the range, so
/// remapping it underneath the guard would invalidate a pointer already handed
/// out.
#[test]
fn safe_point_rejects_routed_guards_active() {
    let unsafe_point = ResizeSafePoint {
        routed_guards_active: 1,
        ..ResizeSafePoint::default()
    };
    let sp = verify_safe_point(unsafe_point);
    assert!(sp.is_err(), "must reject routed_guards_active > 0");
    let reason = sp.err().unwrap();
    assert!(
        reason.contains("Routed") || reason.contains("routed") || reason.contains("Residency"),
        "error should mention routed guards or residency, got: {reason}"
    );
}

/// Deferred releases still queued may free backing the transition is about to
/// rely on, so the safe point is not yet safe.
#[test]
fn safe_point_rejects_pending_deferred_releases() {
    let unsafe_point = ResizeSafePoint {
        pending_deferred_releases: 3,
        ..ResizeSafePoint::default()
    };
    let sp = verify_safe_point(unsafe_point);
    assert!(sp.is_err(), "must reject pending_deferred_releases > 0");
}

/// The rejection rules must not fire on a clean safe point.
///
/// Without this, all three tests above would still pass if `verify_safe_point`
/// were changed to reject unconditionally -- they only ever assert `is_err()`,
/// so they cannot tell "rejects the unsafe field" from "rejects everything".
#[test]
fn safe_point_accepts_a_clean_state() {
    let sp = verify_safe_point(ResizeSafePoint::default());
    assert!(
        sp.is_ok(),
        "a default (clean) safe point must be accepted, got {:?}",
        sp.err()
    );
}
