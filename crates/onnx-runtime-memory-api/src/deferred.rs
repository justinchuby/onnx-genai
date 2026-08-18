//! Owning release: lifecycle vocabulary, structured outcomes, and the
//! provider/context-owned deferred release queue.
//!
//! Phase 3 issued binding identity, allocation generations, and lifetime pins.
//! This module adds the piece that identity alone cannot express: *who owns the
//! final physical release, and what is true after it partially fails*.
//!
//! # The three-step contract
//!
//! 1. **Prepare.** [`crate::MemoryBinding::prepare_release`] matches the binding
//!    identity *and* the allocation generation, removes the live record exactly
//!    once under the per-mechanism lock, and returns an owned
//!    [`PreparedAllocationRelease`]. Because the record is removed under that
//!    lock, two racing final releases cannot both proceed, and a stale handle
//!    whose virtual address was reused cannot match a newer generation.
//! 2. **Queue (optional).** The prepared request is handed to a
//!    [`DeferredReleaseQueue`] owned by the provider context. Every queue call
//!    happens after all registry and mechanism locks are dropped.
//! 3. **Execute.** [`PreparedAllocationRelease::execute`] calls the pinned
//!    allocator with no lock held and returns an
//!    [`AllocationReleaseOutcome`].
//!
//! # Fail-safe rules
//!
//! * A prepared request that is *abandoned* (dropped without `execute`)
//!   quarantines its ownership. It never frees, never blocks, and never loses
//!   metadata.
//! * Enqueue failure returns the exact request to the caller inside
//!   [`DeferredEnqueueError`]; dropping that error quarantines the request
//!   rather than losing it.
//! * Device loss never calls the allocator. Queued requests finish as
//!   device-lost quarantine while keeping their allocator/authority/context
//!   pins.
//! * No `Err`/`Failed` shape may imply "nothing changed" after the device was
//!   mutated. Any partial mutation is [`AllocationReleaseOutcome::Quarantined`]
//!   and carries accounting plus residual facts.

use std::fmt::Debug;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::binding::{MechanismOperation, ReleaseGate};
use crate::{
    AllocationIdentity, AuthorityIdentity, BindingIdentity, DeviceAllocator, DeviceKey,
    MemoryBinding, ProviderContextIdentity,
};

/// Where one allocation's ownership currently sits.
///
/// This is the shared vocabulary used by owning handles, prepared requests,
/// mechanism snapshots, and release outcomes. It deliberately distinguishes
/// "the bytes are gone" from "we still own something we could not give back".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocationReleaseState {
    /// Live and mapped. The owner may still read, write, view, or release it.
    Live,
    /// Final ownership was handed to a deferred queue. The live record is
    /// already gone, so no further allocation-level operation can match it, but
    /// the physical bytes are not released yet.
    Queued,
    /// The allocator unmapped part of the allocation and stopped. Residual
    /// ownership is still held by the runtime and must not be reused.
    PartiallyUnmapped,
    /// Released back to the allocator (which may have pooled rather than freed
    /// the bytes). This is the only success terminal state.
    Released,
    /// The device or provider context was lost. No allocator call may be made,
    /// and reclamation happens only at confirmed context/process termination.
    DeviceLost,
    /// Ownership is retained deliberately because releasing it would be unsafe
    /// or dishonest. Quarantined ownership stays observable and blocks unsafe
    /// mechanism removal.
    Quarantined,
}

impl AllocationReleaseState {
    /// Whether the runtime still owns physical bytes in this state.
    pub const fn retains_ownership(self) -> bool {
        matches!(
            self,
            Self::Live
                | Self::Queued
                | Self::PartiallyUnmapped
                | Self::DeviceLost
                | Self::Quarantined
        )
    }

    /// Whether an allocator callback may still be made from this state.
    ///
    /// Device loss is `false` by construction: the allocator is never called
    /// after loss, in any phase.
    pub const fn permits_allocator_call(self) -> bool {
        matches!(self, Self::Live | Self::Queued)
    }

    /// Whether no further transition is expected without external teardown.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Released | Self::DeviceLost | Self::Quarantined | Self::PartiallyUnmapped
        )
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Queued => "queued",
            Self::PartiallyUnmapped => "partially unmapped",
            Self::Released => "released",
            Self::DeviceLost => "device lost",
            Self::Quarantined => "quarantined",
        }
    }
}

impl std::fmt::Display for AllocationReleaseState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Byte accounting for one release attempt.
///
/// `unmapped_bytes` is the mapped-attribution refund observed by the allocator.
/// **Zero is a valid complete result**: eager allocators have no mapped
/// attribution, and a virtual allocation with nothing committed unmaps nothing.
/// Zero is never an error proxy; failure is expressed by the outcome variant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ReleaseAccounting {
    /// The whole-allocation size the release was prepared for.
    pub allocation_bytes: u64,
    /// Bytes whose global mapping reference transitioned to unmapped.
    pub unmapped_bytes: u64,
}

impl ReleaseAccounting {
    pub const fn new(allocation_bytes: u64, unmapped_bytes: u64) -> Self {
        Self {
            allocation_bytes,
            unmapped_bytes,
        }
    }

    /// Accounting for an eager allocator with no mapped attribution.
    pub const fn eager(allocation_bytes: u64) -> Self {
        Self::new(allocation_bytes, 0)
    }
}

/// Why ownership was retained instead of released.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuarantineReason {
    /// A prepared request was dropped without being executed.
    AbandonedRequest,
    /// An owning handle was dropped without an explicit release.
    OwnerDropped,
    /// The deferred queue refused final ownership.
    EnqueueRejected(DeferredEnqueueRejection),
    /// The device or provider context was lost, so the allocator must not be
    /// called.
    DeviceLost,
    /// The mechanism was already terminated when release was attempted.
    MechanismTerminated,
    /// The allocator mutated part of the allocation and then stopped.
    PartialRelease,
    /// The allocator refused *after* preparation while promising it had not
    /// mutated anything. The live record is already gone, so live ownership
    /// cannot be restored without losing the owner; the conservative answer is
    /// quarantine.
    AllocatorRefused,
    /// A mechanism lock was poisoned, so ownership could not be settled safely.
    StatePoisoned,
}

impl QuarantineReason {
    pub const fn name(self) -> &'static str {
        match self {
            Self::AbandonedRequest => "a prepared release request was abandoned",
            Self::OwnerDropped => "an owning allocation was dropped without explicit release",
            Self::EnqueueRejected(_) => "the deferred release queue refused the request",
            Self::DeviceLost => "the device or provider context was lost",
            Self::MechanismTerminated => "the mechanism was already terminated",
            Self::PartialRelease => "the allocator released only part of the allocation",
            Self::AllocatorRefused => "the allocator refused after the record was retired",
            Self::StatePoisoned => "mechanism state was poisoned",
        }
    }
}

impl std::fmt::Display for QuarantineReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())?;
        if let Self::EnqueueRejected(rejection) = self {
            write!(formatter, " ({})", rejection.name())?;
        }
        Ok(())
    }
}

/// What the runtime still owns after a non-complete release.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResidualOwnership {
    /// The state the residual ownership is parked in.
    pub state: AllocationReleaseState,
    pub reason: QuarantineReason,
    /// Bytes still owned by the runtime. For a fully unreleased allocation this
    /// equals the allocation size.
    pub retained_bytes: u64,
    /// The address the residual ownership refers to. Kept so a manager can
    /// reconcile against provider-side records; it is never dereferenced here.
    pub address: usize,
    pub align: usize,
}

/// A release that failed *before* any device mutation.
///
/// Allocators return this only when nothing was mutated and the caller's state
/// is unchanged. It is the one shape that may imply "nothing happened"; every
/// post-mutation failure must be [`AllocationReleaseOutcome::Quarantined`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseFailure {
    reason: Arc<str>,
}

impl ReleaseFailure {
    pub fn new(reason: impl Into<Arc<str>>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl std::fmt::Display for ReleaseFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.reason)
    }
}

/// The structured result of one whole-allocation release.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocationReleaseOutcome {
    /// The allocation was fully released (freed or pooled). `accounting`
    /// reports the refund, and zero unmapped bytes is a valid complete result.
    Complete { accounting: ReleaseAccounting },
    /// Ownership was retained. `accounting` reports whatever was actually
    /// unmapped before stopping, and `residual` reports what is still owned.
    Quarantined {
        accounting: ReleaseAccounting,
        residual: ResidualOwnership,
    },
    /// Nothing was mutated. Only an allocator may produce this; the binding
    /// layer converts it to `Quarantined` when the live record is already gone.
    Failed { failure: ReleaseFailure },
}

impl AllocationReleaseOutcome {
    pub const fn complete(accounting: ReleaseAccounting) -> Self {
        Self::Complete { accounting }
    }

    pub const fn quarantined(accounting: ReleaseAccounting, residual: ResidualOwnership) -> Self {
        Self::Quarantined {
            accounting,
            residual,
        }
    }

    pub fn failed(reason: impl Into<Arc<str>>) -> Self {
        Self::Failed {
            failure: ReleaseFailure::new(reason),
        }
    }

    /// The lifecycle state this outcome leaves the allocation in.
    pub const fn state(&self) -> AllocationReleaseState {
        match self {
            Self::Complete { .. } => AllocationReleaseState::Released,
            Self::Quarantined { residual, .. } => residual.state,
            // Nothing was mutated, so the allocation is still exactly as live as
            // the caller left it.
            Self::Failed { .. } => AllocationReleaseState::Live,
        }
    }

    pub const fn accounting(&self) -> Option<ReleaseAccounting> {
        match self {
            Self::Complete { accounting } | Self::Quarantined { accounting, .. } => {
                Some(*accounting)
            }
            Self::Failed { .. } => None,
        }
    }

    pub const fn residual(&self) -> Option<ResidualOwnership> {
        match self {
            Self::Quarantined { residual, .. } => Some(*residual),
            _ => None,
        }
    }

    pub const fn failure(&self) -> Option<&ReleaseFailure> {
        match self {
            Self::Failed { failure } => Some(failure),
            _ => None,
        }
    }

    pub const fn is_complete(&self) -> bool {
        matches!(self, Self::Complete { .. })
    }

    pub const fn is_quarantined(&self) -> bool {
        matches!(self, Self::Quarantined { .. })
    }

    /// Unmapped bytes, or zero when nothing was mutated.
    pub const fn unmapped_bytes(&self) -> u64 {
        match self {
            Self::Complete { accounting } | Self::Quarantined { accounting, .. } => {
                accounting.unmapped_bytes
            }
            Self::Failed { .. } => 0,
        }
    }
}

/// Why a [`DeferredReleaseQueue`] refused final ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeferredEnqueueRejection {
    /// The queue is shutting down or already drained.
    Closed,
    /// The queue is bounded and full. Bounded queues are the reason this is a
    /// first-class rejection rather than an unbounded backlog.
    Full,
    /// The queue observed device loss and will not accept allocator work.
    DeviceLost,
    /// Implementation-specific refusal.
    Refused,
}

impl DeferredEnqueueRejection {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Closed => "closed",
            Self::Full => "full",
            Self::DeviceLost => "device lost",
            Self::Refused => "refused",
        }
    }
}

/// Enqueue failure that hands the **exact** prepared request back.
///
/// Nothing is cloned or reconstructed: the request that failed to enqueue is
/// the request returned. Dropping this error without calling
/// [`into_request`](Self::into_request) quarantines that request rather than
/// leaking or freeing it.
#[derive(Debug)]
pub struct DeferredEnqueueError {
    rejection: DeferredEnqueueRejection,
    /// Boxed so the `Ok` path of [`DeferredReleaseQueue::enqueue`] stays small.
    /// The boxed value is the same request the queue was handed; nothing is
    /// cloned or rebuilt.
    request: Box<PreparedAllocationRelease>,
}

impl DeferredEnqueueError {
    pub fn new(rejection: DeferredEnqueueRejection, request: PreparedAllocationRelease) -> Self {
        Self {
            rejection,
            request: Box::new(request),
        }
    }

    pub const fn rejection(&self) -> DeferredEnqueueRejection {
        self.rejection
    }

    pub const fn request(&self) -> &PreparedAllocationRelease {
        &self.request
    }

    /// Recover the exact prepared request.
    pub fn into_request(self) -> PreparedAllocationRelease {
        *self.request
    }

    /// Quarantine the request with the rejection recorded as its reason.
    pub fn quarantine(self) -> AllocationReleaseOutcome {
        let rejection = self.rejection;
        (*self.request).quarantine(QuarantineReason::EnqueueRejected(rejection))
    }
}

impl std::fmt::Display for DeferredEnqueueError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "the deferred release queue refused allocation {:?}: {}",
            self.request.identity(),
            self.rejection.name()
        )
    }
}

impl std::error::Error for DeferredEnqueueError {}

/// A provider/context-owned sink for final allocation ownership.
///
/// The queue is *not* owned by this crate. A CUDA provider owns one per context
/// or per stream, records a fence when a request arrives, and calls
/// [`PreparedAllocationRelease::execute`] once that fence is observed.
///
/// # Contract
///
/// * `enqueue` is always called with no registry lock and no mechanism lock
///   held, so an implementation may take its own locks freely.
/// * `enqueue` must not block on the device.
/// * On refusal, the implementation must return the exact request inside
///   [`DeferredEnqueueError`]. It must never drop the request silently to
///   signal failure; dropping it is defined as quarantine, not as free.
pub trait DeferredReleaseQueue: Send + Sync + Debug {
    /// Take final ownership of `request`.
    fn enqueue(&self, request: PreparedAllocationRelease) -> Result<(), DeferredEnqueueError>;

    /// How many requests are still waiting. Used for observability and for
    /// asserting that a queue does not grow without bound.
    fn pending(&self) -> usize {
        0
    }
}

/// What happened when an owning allocation was handed to a deferred queue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeferredReleaseDisposition {
    /// The queue accepted final ownership. The allocation is
    /// [`AllocationReleaseState::Queued`] until the queue executes it.
    Queued { identity: AllocationIdentity },
    /// The queue refused, so ownership was quarantined at its mechanism.
    Quarantined {
        identity: AllocationIdentity,
        rejection: DeferredEnqueueRejection,
        outcome: AllocationReleaseOutcome,
    },
}

impl DeferredReleaseDisposition {
    pub const fn identity(&self) -> AllocationIdentity {
        match self {
            Self::Queued { identity } | Self::Quarantined { identity, .. } => *identity,
        }
    }

    pub const fn state(&self) -> AllocationReleaseState {
        match self {
            Self::Queued { .. } => AllocationReleaseState::Queued,
            Self::Quarantined { .. } => AllocationReleaseState::Quarantined,
        }
    }

    pub const fn is_queued(&self) -> bool {
        matches!(self, Self::Queued { .. })
    }
}

/// Final ownership of one allocation, detached from its live record.
///
/// A prepared request is produced only after the binding identity and the
/// allocation generation matched and the live record was removed exactly once
/// under the per-mechanism lock. From that moment the allocation is
/// [`AllocationReleaseState::Queued`]: no view, commit, or second release can
/// match it, and address reuse cannot resurrect it.
///
/// The request pins the allocator, the accounting authority, and the provider
/// context, so a queue may hold it across mechanism retirement and across
/// threads.
///
/// # Abandonment is safe
///
/// Dropping a prepared request without calling [`execute`](Self::execute)
/// quarantines it: the ownership is recorded at the mechanism with residual
/// facts, and no allocator call and no blocking wait happen in `Drop`.
pub struct PreparedAllocationRelease {
    binding: MemoryBinding,
    identity: AllocationIdentity,
    ptr: NonNull<u8>,
    bytes: usize,
    align: usize,
    allocator: Arc<dyn DeviceAllocator>,
    authority: AuthorityIdentity,
    context: ProviderContextIdentity,
    /// Keeps the mechanism non-quiescent, so a queued request blocks mechanism
    /// removal and provider-context termination. Dropped only after the final
    /// state is recorded.
    operation: Option<MechanismOperation>,
    /// Cleared by any consuming path, so `Drop` quarantines only a genuinely
    /// abandoned request.
    armed: bool,
}

// SAFETY: the request carries allocation metadata over a provider-defined
// device address plus `Send + Sync` pins. It exposes no safe dereference, so
// moving it to a queue thread does not access the pointed-to bytes.
unsafe impl Send for PreparedAllocationRelease {}
// SAFETY: shared access exposes only copied metadata and a non-dereferenced
// address; every consuming operation takes the request by value.
unsafe impl Sync for PreparedAllocationRelease {}

impl Debug for PreparedAllocationRelease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedAllocationRelease")
            .field("identity", &self.identity)
            .field("address", &(self.ptr.as_ptr() as usize))
            .field("bytes", &self.bytes)
            .field("align", &self.align)
            .field("authority", &self.authority)
            .field("provider_context", &self.context)
            .field("armed", &self.armed)
            .finish()
    }
}

/// The resources a prepared request pins for its whole lifetime.
pub(crate) struct PreparedReleasePins {
    pub(crate) allocator: Arc<dyn DeviceAllocator>,
    pub(crate) authority: AuthorityIdentity,
    pub(crate) context: ProviderContextIdentity,
    pub(crate) operation: MechanismOperation,
}

impl PreparedAllocationRelease {
    pub(crate) fn new(
        binding: MemoryBinding,
        identity: AllocationIdentity,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
        pins: PreparedReleasePins,
    ) -> Self {
        Self {
            binding,
            identity,
            ptr,
            bytes,
            align,
            allocator: pins.allocator,
            authority: pins.authority,
            context: pins.context,
            operation: Some(pins.operation),
            armed: true,
        }
    }

    pub const fn identity(&self) -> AllocationIdentity {
        self.identity
    }

    pub const fn binding_identity(&self) -> BindingIdentity {
        self.identity.binding()
    }

    pub const fn device(&self) -> DeviceKey {
        self.identity.binding().device()
    }

    /// The pinned accounting authority. A manager refunds against this identity
    /// even if the mechanism has since been retired.
    pub const fn authority(&self) -> AuthorityIdentity {
        self.authority
    }

    /// The pinned provider context. The queue that owns this request belongs to
    /// this context.
    pub const fn provider_context(&self) -> ProviderContextIdentity {
        self.context
    }

    /// The pinned allocator that must perform the physical release.
    pub fn allocator(&self) -> &Arc<dyn DeviceAllocator> {
        &self.allocator
    }

    /// The address to release. Never dereferenced by this crate.
    pub const fn as_ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    pub const fn len(&self) -> usize {
        self.bytes
    }

    pub const fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub const fn alignment(&self) -> usize {
        self.align
    }

    /// Always [`AllocationReleaseState::Queued`]: the live record is gone and
    /// the bytes are not released yet.
    pub const fn state(&self) -> AllocationReleaseState {
        AllocationReleaseState::Queued
    }

    /// Perform the physical release through the pinned allocator.
    ///
    /// Returns [`AllocationReleaseOutcome::Complete`] or
    /// [`AllocationReleaseOutcome::Quarantined`], never
    /// [`AllocationReleaseOutcome::Failed`]: the live record was already
    /// retired at preparation, so "nothing changed" is no longer representable
    /// here and an allocator-level `Failed` is conservatively quarantined.
    ///
    /// # Lock order
    ///
    /// The mechanism lifecycle is read under the mechanism lock, that lock is
    /// dropped, the allocator runs with **no** lock held, and the final state is
    /// then recorded under the mechanism lock again. The registry lock is never
    /// taken here, so the two lock classes are still never nested.
    ///
    /// Device loss observed at this point never reaches the allocator: the
    /// request finishes as device-lost quarantine and keeps its pins.
    pub fn execute(mut self) -> AllocationReleaseOutcome {
        self.armed = false;
        match self.binding.mechanism().release_gate() {
            ReleaseGate::Allowed => {}
            ReleaseGate::DeviceLost => {
                return self.settle_quarantine(
                    ReleaseAccounting::new(self.bytes as u64, 0),
                    AllocationReleaseState::DeviceLost,
                    QuarantineReason::DeviceLost,
                    self.bytes as u64,
                );
            }
            ReleaseGate::Terminated => {
                return self.settle_quarantine(
                    ReleaseAccounting::new(self.bytes as u64, 0),
                    AllocationReleaseState::Quarantined,
                    QuarantineReason::MechanismTerminated,
                    self.bytes as u64,
                );
            }
            ReleaseGate::Poisoned => {
                return self.settle_quarantine(
                    ReleaseAccounting::new(self.bytes as u64, 0),
                    AllocationReleaseState::Quarantined,
                    QuarantineReason::StatePoisoned,
                    self.bytes as u64,
                );
            }
        }

        // SAFETY: preparation matched the binding identity and the allocation
        // generation and removed the exact live record under the mechanism
        // lock, so this is one live allocation of this mechanism with exactly
        // these bytes and alignment, and it cannot be released twice. No
        // registry or mechanism lock is held here.
        let outcome = unsafe { self.allocator.release(self.ptr, self.bytes, self.align) };

        match outcome {
            AllocationReleaseOutcome::Complete { accounting } => {
                self.settle_released();
                AllocationReleaseOutcome::Complete { accounting }
            }
            AllocationReleaseOutcome::Quarantined {
                accounting,
                residual,
            } => self.settle_quarantine(
                accounting,
                residual.state,
                residual.reason,
                residual.retained_bytes,
            ),
            // The allocator promises it mutated nothing, but preparation
            // already retired the live record and consumed the owner. Restoring
            // live ownership would require inventing a record the owner can no
            // longer reach, so the conservative, honest answer is quarantine.
            //
            // The allocator's own message is not carried into the residual
            // facts, which are deliberately `Copy`; an implementation that needs
            // the text should log it before returning `Failed`.
            AllocationReleaseOutcome::Failed { .. } => {
                let bytes = self.bytes as u64;
                self.settle_quarantine(
                    ReleaseAccounting::new(bytes, 0),
                    AllocationReleaseState::Quarantined,
                    QuarantineReason::AllocatorRefused,
                    bytes,
                )
            }
        }
    }

    /// Retain ownership deliberately without calling the allocator.
    ///
    /// This is the explicit form of what `Drop` does implicitly.
    pub fn quarantine(mut self, reason: QuarantineReason) -> AllocationReleaseOutcome {
        self.armed = false;
        let bytes = self.bytes as u64;
        self.settle_quarantine(
            ReleaseAccounting::new(bytes, 0),
            AllocationReleaseState::Quarantined,
            reason,
            bytes,
        )
    }

    fn settle_released(&mut self) {
        self.binding.mechanism().settle_release(self.identity);
        // The active-operation pin is released only after the mechanism has
        // observed the final state, so a snapshot never sees a quiescent
        // mechanism with unsettled ownership.
        self.operation = None;
    }

    fn settle_quarantine(
        &mut self,
        accounting: ReleaseAccounting,
        state: AllocationReleaseState,
        reason: QuarantineReason,
        retained_bytes: u64,
    ) -> AllocationReleaseOutcome {
        let residual = ResidualOwnership {
            state,
            reason,
            retained_bytes,
            address: self.ptr.as_ptr() as usize,
            align: self.align,
        };
        self.binding
            .mechanism()
            .settle_quarantine(QuarantinedAllocation {
                identity: self.identity,
                address: residual.address,
                bytes: self.bytes,
                align: self.align,
                state,
                reason,
                retained_bytes,
            });
        self.operation = None;
        AllocationReleaseOutcome::Quarantined {
            accounting,
            residual,
        }
    }
}

impl Drop for PreparedAllocationRelease {
    /// Quarantine an abandoned request.
    ///
    /// This never calls the allocator, never enqueues, and never waits. It takes
    /// only the per-mechanism lock, which is a leaf in the documented lock
    /// order, and records residual ownership so the bytes stay accounted for.
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.armed = false;
        let bytes = self.bytes as u64;
        let _ = self.settle_quarantine(
            ReleaseAccounting::new(bytes, 0),
            AllocationReleaseState::Quarantined,
            QuarantineReason::AbandonedRequest,
            bytes,
        );
    }
}

/// One piece of ownership the runtime kept instead of releasing.
///
/// Quarantined ownership stays observable through
/// [`crate::MechanismSnapshot`] and
/// [`crate::BindingRegistry::quarantined`], and blocks mechanism removal. It is
/// cleared only by confirmed provider-context termination, which is the point
/// where the device state it refers to provably no longer exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QuarantinedAllocation {
    pub identity: AllocationIdentity,
    /// The address that is still owned. Never dereferenced by this crate.
    pub address: usize,
    pub bytes: usize,
    pub align: usize,
    pub state: AllocationReleaseState,
    pub reason: QuarantineReason,
    pub retained_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_unmapped_bytes_is_a_valid_complete_outcome() {
        let outcome = AllocationReleaseOutcome::complete(ReleaseAccounting::eager(4096));
        assert!(outcome.is_complete());
        assert_eq!(outcome.unmapped_bytes(), 0);
        assert_eq!(outcome.state(), AllocationReleaseState::Released);
        assert!(outcome.residual().is_none());
    }

    #[test]
    fn failure_is_the_only_unchanged_shape() {
        let failed = AllocationReleaseOutcome::failed("driver busy");
        assert_eq!(failed.state(), AllocationReleaseState::Live);
        assert!(failed.accounting().is_none());
        assert_eq!(
            failed.failure().map(ReleaseFailure::reason),
            Some("driver busy")
        );
    }

    #[test]
    fn states_answer_ownership_and_callback_questions() {
        assert!(AllocationReleaseState::Live.retains_ownership());
        assert!(!AllocationReleaseState::Released.retains_ownership());
        assert!(!AllocationReleaseState::DeviceLost.permits_allocator_call());
        assert!(!AllocationReleaseState::Quarantined.permits_allocator_call());
        assert!(AllocationReleaseState::Queued.permits_allocator_call());
        assert!(AllocationReleaseState::PartiallyUnmapped.is_terminal());
    }
}
