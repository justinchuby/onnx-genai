//! Keeping a provider/context alive across an outstanding deferred release.
//!
//! A deferred release is a promise that someone will free device memory later.
//! Whoever performs that free needs the provider/context it was allocated
//! against to still exist: a stream-ordered free retiring after its CUDA
//! context has been torn down unmaps handles the teardown already released,
//! which is undefined behaviour at the driver level rather than a Rust error.
//!
//! The host side of a plugin boundary cannot depend on the governor, so the
//! pin is expressed here as two object-safe traits. The governor implements
//! them over its provider-context records; a mechanism holds
//! [`ProviderContextPin`] values and knows nothing about how the pin is
//! counted.
//!
//! The direction of the guarantee matters. A pin does not ask a context to
//! stay alive; it makes teardown observe the outstanding work. Acquiring a pin
//! against a context that is already retiring must fail rather than succeed,
//! because a release that cannot be pinned is a release that must not be
//! queued.

use core::fmt::Debug;

use crate::binding::ProviderContextIdentity;

/// Why a provider/context pin could not be acquired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderContextPinError {
    /// The context is retiring, lost, or terminated, so no new work may be
    /// attached to it. The caller must not queue the release.
    ContextUnavailable(ProviderContextIdentity),
    /// The implementation's outstanding-work counter cannot represent another
    /// pin. Treated as a refusal rather than a wrap.
    PinCountOverflow(ProviderContextIdentity),
}

impl core::fmt::Display for ProviderContextPinError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ContextUnavailable(identity) => write!(
                formatter,
                "provider context {identity:?} is no longer accepting work"
            ),
            Self::PinCountOverflow(identity) => write!(
                formatter,
                "provider context {identity:?} has too many outstanding pins"
            ),
        }
    }
}

impl std::error::Error for ProviderContextPinError {}

/// A live claim on a provider/context.
///
/// Dropping the value releases the claim, so implementations carry the
/// bookkeeping in `Drop` rather than in an explicit method. Nothing about the
/// pin is inspectable beyond the identity it belongs to: a holder must not be
/// able to extend, transfer, or interrogate the context through it.
pub trait ProviderContextPin: Send + Sync + Debug {
    /// Which context this pin holds.
    fn context(&self) -> ProviderContextIdentity;
}

/// Something that can hand out [`ProviderContextPin`]s for one context.
pub trait ProviderContextPinSource: Send + Sync + Debug {
    /// Which context this source pins.
    fn context(&self) -> ProviderContextIdentity;

    /// Claim the context until the returned pin is dropped.
    ///
    /// Fails when the context is no longer accepting work. Callers must treat
    /// a failure as a refusal to queue the deferred release, not as a reason
    /// to proceed unpinned.
    fn pin(&self) -> Result<Box<dyn ProviderContextPin>, ProviderContextPinError>;
}
