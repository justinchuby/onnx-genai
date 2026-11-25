//! Scoped proof that an external host already validated ONNX schema legality.

use std::cell::Cell;

thread_local! {
    static AUTHORITATIVE_SCHEMA_DEPTH: Cell<u32> = const { Cell::new(0) };
}

struct AuthorityGuard;

impl Drop for AuthorityGuard {
    fn drop(&mut self) {
        AUTHORITATIVE_SCHEMA_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

/// Run capability discovery after the host has authoritatively validated the
/// graph against its ONNX schemas.
///
/// This scope is intentionally limited to the synchronous plugin
/// `GetCapability` callback. Native/standalone callers remain responsible for
/// schema-version admission.
///
/// # Safety
///
/// The caller must be an execution-provider bridge whose host has already
/// completed ONNX schema and type validation for every node queried inside
/// `f`. Using this scope for an unvalidated graph can admit schema-illegal
/// nodes to native kernels.
#[doc(hidden)]
pub unsafe fn with_authoritative_schema_validation<T>(f: impl FnOnce() -> T) -> T {
    AUTHORITATIVE_SCHEMA_DEPTH.with(|depth| depth.set(depth.get() + 1));
    let _guard = AuthorityGuard;
    f()
}

/// Whether capability discovery is currently covered by host schema validation.
#[doc(hidden)]
pub fn authoritative_schema_validation_active() -> bool {
    AUTHORITATIVE_SCHEMA_DEPTH.with(|depth| depth.get() != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_scope_is_nested_and_panic_safe() {
        assert!(!authoritative_schema_validation_active());
        unsafe {
            with_authoritative_schema_validation(|| {
                assert!(authoritative_schema_validation_active());
                with_authoritative_schema_validation(|| {
                    assert!(authoritative_schema_validation_active());
                });
                assert!(authoritative_schema_validation_active());
            });
        }
        assert!(!authoritative_schema_validation_active());

        let _ = std::panic::catch_unwind(|| unsafe {
            with_authoritative_schema_validation(|| panic!("scope test"));
        });
        assert!(!authoritative_schema_validation_active());
    }
}
