//! Translating nxmem statuses into the runtime's error vocabulary.

use onnx_runtime_memory_abi::{NxmemStatus, NxmemStatusCode};
use onnx_runtime_memory_api::{MemoryError, Tier};

/// Something went wrong loading or talking to a memory plugin.
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    /// The dynamic library could not be opened.
    #[error("cannot open memory plugin {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: libloading::Error,
    },

    /// A required entry point is missing.
    ///
    /// All three entry points are required. In particular a library without
    /// `NxmemQueryUnloadReadiness` is refused rather than loaded, because the
    /// host would then have no way to tell whether unloading it would strand
    /// live allocations.
    #[error(
        "memory plugin {path} does not export the required entry point `{symbol}`; every nxmem \
         plugin must export NxmemNegotiate, NxmemCreateAllocatorFactories and \
         NxmemQueryUnloadReadiness"
    )]
    MissingSymbol { path: String, symbol: String },

    /// The plugin and host could not agree on a version or capability set.
    #[error("memory plugin {path} failed version negotiation: {reason}")]
    Negotiation { path: String, reason: String },

    /// The plugin returned a failing status.
    ///
    /// The status is boxed because it carries a 256-byte inline message
    /// buffer, which is right on the wire — the buffer is what keeps heap
    /// ownership from crossing the boundary — but would otherwise make every
    /// `PluginError` that size. This type never crosses FFI, so boxing here
    /// costs nothing the ABI cares about.
    #[error("memory plugin call `{operation}` failed: {}", .status.describe())]
    Call {
        operation: &'static str,
        status: Box<NxmemStatus>,
    },

    /// The plugin violated the contract in a way that is not a mere failure.
    #[error("memory plugin {path} violated the nxmem contract: {reason}")]
    Contract { path: String, reason: String },

    /// The named mechanism does not exist in this plugin.
    #[error("memory plugin {path} exposes no mechanism named `{name}`; it offers: {available}")]
    UnknownMechanism {
        path: String,
        name: String,
        available: String,
    },
}

impl PluginError {
    /// Build a call failure from a status, for a named operation.
    pub fn call(operation: &'static str, status: NxmemStatus) -> Self {
        Self::Call {
            operation,
            status: Box::new(status),
        }
    }

    /// The concrete status code the plugin reported, when there was one.
    ///
    /// Exposed so a caller — a test above all — can assert on the code rather
    /// than on the wording of a human-readable message. A message is written
    /// for a person reading a log; matching on its text turns a rewording into
    /// a false pass or a false failure, and makes a test that never checked
    /// the real condition look like it did.
    pub fn status_code(&self) -> Option<NxmemStatusCode> {
        match self {
            Self::Call { status, .. } => status.status_code(),
            _ => None,
        }
    }
}

/// Turn a failing plugin status into a [`MemoryError`].
///
/// The mapping is deliberately narrow. Only genuine capacity refusals become
/// [`MemoryError::AllocationFailed`]; contract violations keep their own text
/// so a misbehaving plugin is not mistaken for an out-of-memory condition.
pub fn status_to_memory_error(
    operation: &str,
    tier: Tier,
    requested: u64,
    status: &NxmemStatus,
) -> MemoryError {
    // `describe` already renders the code name, including `UNKNOWN(n)` for a
    // code this host does not recognise, so a newer plugin's failure stays
    // traceable instead of being flattened into a generic message.
    let detail = status.describe();
    let reason = match status.status_code() {
        Some(NxmemStatusCode::OutOfMemory) => format!(
            "the memory plugin is out of {} memory for `{operation}`: {detail}",
            tier.name()
        ),
        Some(NxmemStatusCode::UnsupportedCapability) | Some(NxmemStatusCode::NotImplemented) => {
            format!(
                "the memory plugin does not support `{operation}`: {detail}; select a mechanism \
                 that advertises the capability this operation needs"
            )
        }
        Some(NxmemStatusCode::WrongDevice)
        | Some(NxmemStatusCode::WrongMechanism)
        | Some(NxmemStatusCode::UnknownAllocation)
        | Some(NxmemStatusCode::InvalidArgument) => format!(
            "the memory plugin rejected `{operation}` as misdirected or invalid: {detail}; this \
             is a host-side routing bug, not a capacity problem"
        ),
        _ => format!("the memory plugin failed `{operation}`: {detail}"),
    };
    MemoryError::AllocationFailed {
        tier: tier.name(),
        requested,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_out_of_memory_status_keeps_the_tier_and_size() {
        let status = NxmemStatus::with_message(NxmemStatusCode::OutOfMemory, "the pool is empty");
        let error = status_to_memory_error("allocate", Tier::Device, 4096, &status);
        let text = error.to_string();
        assert!(text.contains("4096"), "{text}");
        assert!(text.contains("device"), "{text}");
        assert!(text.contains("the pool is empty"), "{text}");
    }

    #[test]
    fn an_unsupported_status_says_how_to_fix_it() {
        let status =
            NxmemStatus::with_message(NxmemStatusCode::UnsupportedCapability, "no virtual backing");
        let error = status_to_memory_error("commit_range", Tier::Device, 0, &status);
        let text = error.to_string();
        assert!(text.contains("advertises the capability"), "{text}");
    }

    #[test]
    fn a_wrong_mechanism_status_is_not_reported_as_out_of_memory() {
        let status =
            NxmemStatus::with_message(NxmemStatusCode::WrongMechanism, "foreign allocation");
        let error = status_to_memory_error("deallocate", Tier::Device, 0, &status);
        let text = error.to_string();
        assert!(text.contains("misdirected"), "{text}");
        assert!(!text.contains("is out of"), "{text}");
    }

    #[test]
    fn an_unrecognised_wire_code_is_named_rather_than_guessed() {
        let mut status = NxmemStatus::with_message(NxmemStatusCode::InternalError, "boom");
        status.code = 9_999;
        let error = status_to_memory_error("allocate", Tier::Host, 8, &status);
        let text = error.to_string();
        assert!(text.contains("UNKNOWN(9999)"), "{text}");
    }
}
