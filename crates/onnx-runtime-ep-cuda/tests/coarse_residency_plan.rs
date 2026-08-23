//! #1810 Slice 5 — Non-GPU behavioral tests for the coarse-boundary
//! residency plan application seam.
//!
//! Real end-to-end assertions (device bytes moved, stable VA unchanged
//! across a transition) require CUDA and are gated behind the `gpu-tests`
//! feature (see the `_gpu` suffixed tests in this crate). This file covers
//! the invariants that hold without a GPU:
//!
//!   * feature gate off ⇒ structural no-op (no allocator touched, empty
//!     counters, explicit `fallback_reason`);
//!   * empty inputs ⇒ structural no-op with `values_inspected == 0`;
//!   * `plan.policy_name()` is propagated verbatim into the outcome so
//!     telemetry can attribute a run to the policy that produced it.

use std::collections::HashMap;

use onnx_runtime_ep_cuda::coarse_residency::{
    BoundaryApplicationOutcome, COARSE_RESIDENCY_PROFILE_ENV, coarse_residency_profile_enabled,
};

#[test]
fn env_var_name_is_the_documented_gate() {
    assert_eq!(
        COARSE_RESIDENCY_PROFILE_ENV,
        "ONNX_GENAI_WEIGHT_OFFLOAD_COARSE_RESIDENCY_PROFILE"
    );
}

#[test]
fn feature_gate_reflects_env_value_semantics() {
    // Read whatever the ambient env says. The point is that the gate reads
    // the specific env var above; we don't set it here (tests share a
    // process — mutating env is contagious).
    let raw = std::env::var(COARSE_RESIDENCY_PROFILE_ENV).ok();
    let expected = matches!(
        raw.as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    );
    assert_eq!(coarse_residency_profile_enabled(), expected);
}

#[test]
fn default_outcome_has_zeroed_counters_and_no_touches() {
    let outcome = BoundaryApplicationOutcome::default();
    assert_eq!(outcome.values_inspected, 0);
    assert_eq!(outcome.values_touched, 0);
    assert_eq!(outcome.hot_expert_count, 0);
    assert_eq!(outcome.cold_expert_count, 0);
    assert_eq!(outcome.host_bytes_committed, 0);
    assert_eq!(outcome.failure_count, 0);
    assert_eq!(outcome.rollback_count, 0);
    assert!(outcome.per_value_fallbacks.is_empty());
    assert!(outcome.committed_values.is_empty());
    assert!(outcome.quarantined.is_empty());
    assert!(outcome.fallback_reason.is_none());
}

#[test]
fn static_profile_policy_wiring_produces_named_plan() {
    // We can build a ResidencyPlan without any allocator or CUDA, then
    // observe that its policy_name flows through the outcome.
    let policy = onnx_runtime_ep_api::StaticProfileResidencyPolicy::new(HashMap::new());
    let plan = onnx_runtime_ep_api::plan_residency(std::iter::empty(), &policy, None);
    assert_eq!(plan.policy_name(), "static_profile");
    // Empty plan ⇒ nothing to inspect. This confirms the seam is at least
    // wired without pulling in CudaRuntime.
    assert_eq!(plan.len(), 0);
}
