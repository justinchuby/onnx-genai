//! Opset version resolution (`docs/EAGER.md` §5).
//!
//! Phase-1 implements the per-call and per-domain-default levels of the design's
//! three-level control (`docs/EAGER.md` §5.2):
//!
//! * **Level 3 — per-call**: `explicit_opset` passed to
//!   [`dispatch`](crate::EagerContext::dispatch) wins outright.
//! * **Level 1 — domain default**: otherwise the domain's registered default
//!   opset (see [`crate::domain::DomainRegistry`]) applies.
//!
//! The **Level 2 context-manager / thread-local override** (`nxrt.opset(...)`,
//! `docs/EAGER.md` §5.3) is DEFERRED — it is a binding-layer concern that hooks
//! in here between the explicit and domain-default steps.

/// The latest ONNX opset the runtime targets, used as the default for the
/// standard (`""`) domain when a domain has no registered default.
///
/// `docs/EAGER.md` §5.5 / §10.1 fix this at **26** ("latest at nxrt release").
/// The lower runtime layers key inference/kernel lookup on the *graph's*
/// `opset_imports` and have no single shared constant (the session executor
/// falls back to `u64::MAX`, `onnx-runtime-session/src/executor.rs`), so the
/// design doc is the authoritative source for the eager default.
pub const LATEST_ONNX_OPSET: u64 = 26;

/// Resolve the effective opset for a domain.
///
/// Priority (`docs/EAGER.md` §5.2, reconciled to Phase-1 scope):
/// explicit per-call value > `domain_default`.
pub fn resolve_opset(domain_default: u64, explicit: Option<u64>) -> u64 {
    explicit.unwrap_or(domain_default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_wins_over_default() {
        assert_eq!(resolve_opset(17, Some(11)), 11);
        assert_eq!(resolve_opset(17, None), 17);
        assert_eq!(resolve_opset(LATEST_ONNX_OPSET, None), LATEST_ONNX_OPSET);
    }
}
