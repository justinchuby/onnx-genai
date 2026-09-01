//! Private collaboration boundary between the session executor and EP API.
//!
//! This crate is unpublished and is deliberately not re-exported by any
//! runtime crate. Cargo features cannot grant caller authority because the
//! session is the only production crate with a direct dependency on this
//! issuer surface.

use std::sync::Arc;

struct AuthoritySeal {
    allocation: Box<u8>,
}

/// Non-forgeable session lifecycle authority.
///
/// The owning `Arc` makes uninitialized and zeroed representations invalid.
/// The private allocation-bearing seal prevents the authority from being a
/// zero-sized or all-bit-pattern-valid token.
pub struct ExecutorArtifactSessionAuthority {
    seal: Arc<AuthoritySeal>,
}

impl ExecutorArtifactSessionAuthority {
    /// Issue the runtime session's private collaboration authority.
    ///
    /// This unpublished crate is a direct implementation dependency of
    /// `onnx-runtime-session`; public runtime crates do not re-export either
    /// this constructor or its type.
    #[doc(hidden)]
    pub fn issue_for_runtime_session() -> Self {
        Self {
            seal: Arc::new(AuthoritySeal {
                allocation: Box::new(0),
            }),
        }
    }

    /// Prove that the authority contains its live owning seal.
    #[doc(hidden)]
    pub fn witness(&self) {
        let _ = *self.seal.allocation;
    }
}

#[cfg(test)]
mod tests {
    use super::ExecutorArtifactSessionAuthority;

    #[test]
    fn authority_is_owning_and_non_zero_sized() {
        assert_ne!(std::mem::size_of::<ExecutorArtifactSessionAuthority>(), 0);
        assert!(std::mem::needs_drop::<ExecutorArtifactSessionAuthority>());
    }
}
