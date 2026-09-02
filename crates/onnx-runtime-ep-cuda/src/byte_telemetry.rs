//! Bounded production-boundary byte telemetry.
//!
//! One provider-owned ledger belongs to one exact provider/device/executor/
//! generation/logical-session scope. The shipped default has no recorder and
//! performs one `Option` check before taking the original operation path.
//!
//! Enabled recording is preallocated. An authenticated executor requirement
//! installs a borrowed recorder pointer in a fixed-capacity thread-local stack;
//! CUDA operation boundaries reserve fixed event slots and byte totals with
//! atomics. The warmed path performs no mutex acquisition, hash lookup, vector
//! growth, reference-count clone, heap allocation, thread-id lookup, formatting,
//! or blocking synchronization. Snapshot/reset/close are cold lifecycle paths.
//!
//! Downstream code can read the returned ledger but cannot construct or clone
//! its mutation authority:
//!
//! ```compile_fail,E0624
//! use onnx_runtime_ep_cuda::byte_telemetry::{ObservedByteLedger, ObservedScope};
//!
//! fn mint() {
//!     let _ = ObservedByteLedger::new(
//!         ObservedScope {
//!             provider: 1,
//!             device: 0,
//!             executor: 1,
//!             generation: 1,
//!             logical_session: 1,
//!         },
//!         8,
//!     );
//! }
//! ```
//!
//! ```compile_fail,E0599
//! use onnx_runtime_ep_cuda::byte_telemetry::ObservedByteLedger;
//!
//! fn clone_authority(ledger: ObservedByteLedger) {
//!     let _: ObservedByteLedger = ledger.clone();
//! }
//! ```
//!
//! ```compile_fail,E0624
//! use onnx_runtime_ep_cuda::byte_telemetry::ObservedByteLedger;
//! use onnx_runtime_ep_cuda::runtime::CudaRuntime;
//!
//! fn bypass_provider(runtime: &CudaRuntime, ledger: &ObservedByteLedger) {
//!     runtime.install_observed_byte_ledger(ledger).unwrap();
//! }
//! ```
//!
//! ```compile_fail,E0599
//! use onnx_runtime_ep_api::{ExecutorArtifactGeneration, ExecutorInstanceId};
//! use onnx_runtime_ep_cuda::CudaExecutionProvider;
//!
//! fn forge_labels(provider: &CudaExecutionProvider) {
//!     let _ = provider.open_observed_byte_session(
//!         ExecutorInstanceId::from_raw(0xfeed),
//!         ExecutorArtifactGeneration::from_raw(0xbeef),
//!         0xcafe,
//!         16,
//!     );
//! }
//! ```

use std::cell::UnsafeCell;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};

use serde::Serialize;

pub const OBSERVED_BYTE_SCHEMA: &str = "onnx-genai.freetoken-observed-bytes.v2";

const MAX_BATCH_EVENTS: usize = 16;
const MAX_CONTEXT_DEPTH: usize = 8;
const PHASE_COUNT: usize = 9;
const CATEGORY_COUNT: usize = 17;
const STATUS_COUNT: usize = 10;
const TOTAL_COUNT: usize = CATEGORY_COUNT * STATUS_COUNT;
const PHASE_TOTAL_COUNT: usize = PHASE_COUNT * TOTAL_COUNT;

const LIFECYCLE_ACTIVE: u8 = 0;
const LIFECYCLE_SNAPSHOT_ACTIVE: u8 = 1;
const LIFECYCLE_RETIRING: u8 = 2;
const LIFECYCLE_SNAPSHOT_RETIRING: u8 = 3;
const LIFECYCLE_CLOSED: u8 = 4;

const SLOT_EMPTY: u8 = 0;
const SLOT_COMMITTED: u8 = 1;

const FAULT_NONE: u8 = 0;
const FAULT_SUBMISSION_ABANDONED: u8 = 1;
const FAULT_STALE_RECORDER: u8 = 2;
const FAULT_UNREPORTED_OPERATION: u8 = 3;
const FAULT_CONTEXT_ORDER: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ObservedPhase {
    Setup,
    Prefill,
    DirectWarmup,
    CaptureSetup,
    Replay,
    DecodeSteady,
    Verification,
    Failure,
    Teardown,
}

impl ObservedPhase {
    pub const ALL: [Self; PHASE_COUNT] = [
        Self::Setup,
        Self::Prefill,
        Self::DirectWarmup,
        Self::CaptureSetup,
        Self::Replay,
        Self::DecodeSteady,
        Self::Verification,
        Self::Failure,
        Self::Teardown,
    ];

    const fn index(self) -> usize {
        self as usize
    }

    fn from_index(index: u8) -> Self {
        Self::ALL[index as usize]
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ObservedCategory {
    SourceRead,
    MmapPageIn,
    HostAllocation,
    HostWrite,
    DeviceAllocation,
    DeviceRelease,
    H2d,
    D2h,
    D2d,
    CudaMemset,
    VmmReserve,
    VmmMap,
    VmmUnmap,
    PageIn,
    ExpertPublication,
    StatePublication,
    OutputPublication,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservedLayer {
    SourceIo,
    HostMemory,
    DeviceAllocator,
    Transport,
    DeviceMutation,
    MappingTopology,
    LogicalPublication,
}

impl ObservedCategory {
    pub const ALL: [Self; CATEGORY_COUNT] = [
        Self::SourceRead,
        Self::MmapPageIn,
        Self::HostAllocation,
        Self::HostWrite,
        Self::DeviceAllocation,
        Self::DeviceRelease,
        Self::H2d,
        Self::D2h,
        Self::D2d,
        Self::CudaMemset,
        Self::VmmReserve,
        Self::VmmMap,
        Self::VmmUnmap,
        Self::PageIn,
        Self::ExpertPublication,
        Self::StatePublication,
        Self::OutputPublication,
    ];

    const fn index(self) -> usize {
        self as usize
    }

    pub const fn layer(self) -> ObservedLayer {
        match self {
            Self::SourceRead | Self::MmapPageIn => ObservedLayer::SourceIo,
            Self::HostAllocation | Self::HostWrite => ObservedLayer::HostMemory,
            Self::DeviceAllocation | Self::DeviceRelease => ObservedLayer::DeviceAllocator,
            Self::H2d | Self::D2h | Self::D2d => ObservedLayer::Transport,
            Self::CudaMemset => ObservedLayer::DeviceMutation,
            Self::VmmReserve | Self::VmmMap | Self::VmmUnmap => ObservedLayer::MappingTopology,
            Self::PageIn
            | Self::ExpertPublication
            | Self::StatePublication
            | Self::OutputPublication => ObservedLayer::LogicalPublication,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ObservedStatus {
    Submitted,
    Completed,
    Committed,
    Published,
    Failed,
    RolledBack,
    Quarantined,
    Reclaimed,
    Unsupported,
    Elided,
}

impl ObservedStatus {
    pub const ALL: [Self; STATUS_COUNT] = [
        Self::Submitted,
        Self::Completed,
        Self::Committed,
        Self::Published,
        Self::Failed,
        Self::RolledBack,
        Self::Quarantined,
        Self::Reclaimed,
        Self::Unsupported,
        Self::Elided,
    ];

    const fn index(self) -> usize {
        self as usize
    }

    pub const fn is_useful(self) -> bool {
        matches!(self, Self::Completed | Self::Committed | Self::Published)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ObservedStream {
    Host,
    Compute,
    Transfer,
    LegacyDefault,
    Logical,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ObservedBoundary {
    RuntimeDeviceAllocate,
    RuntimeDevicePoolReuse,
    RuntimeDeviceRelease,
    RuntimeDevicePoolReclaim,
    RuntimeH2d,
    RuntimeD2h,
    RuntimeD2d,
    RuntimeCudaMemset,
    PinnedHostAllocate,
    PinnedHostReuse,
    MmapMaterialize,
    ResidentMaterialize,
    WeightVmmReserve,
    WeightVmmMap,
    WeightVmmUnmap,
    WeightPageIn,
    WeightExpertPublish,
    StatePublish,
    OutputPublish,
    AsyncCompletionUnsupported,
}

impl ObservedBoundary {
    const fn stream(self) -> ObservedStream {
        match self {
            Self::RuntimeH2d | Self::RuntimeD2h => ObservedStream::LegacyDefault,
            Self::RuntimeD2d | Self::RuntimeCudaMemset => ObservedStream::Compute,
            Self::StatePublish | Self::OutputPublish | Self::WeightExpertPublish => {
                ObservedStream::Logical
            }
            Self::AsyncCompletionUnsupported => ObservedStream::Transfer,
            _ => ObservedStream::Host,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ObservedScope {
    pub provider: u64,
    pub device: u32,
    pub executor: u64,
    pub generation: u64,
    pub logical_session: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ObservedEvent {
    pub scope: ObservedScope,
    pub epoch: u64,
    pub sequence: u64,
    pub submission: u64,
    pub phase: ObservedPhase,
    pub stream: ObservedStream,
    pub category: ObservedCategory,
    pub boundary: ObservedBoundary,
    pub status: ObservedStatus,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservedSnapshot {
    pub schema: &'static str,
    pub scope: ObservedScope,
    pub epoch: u64,
    pub phase: ObservedPhase,
    pub events: Vec<ObservedEvent>,
    totals: [u64; TOTAL_COUNT],
    phase_totals: [u64; PHASE_TOTAL_COUNT],
}

impl ObservedSnapshot {
    pub fn bytes(&self, category: ObservedCategory, status: ObservedStatus) -> u64 {
        self.totals[total_index(category, status)]
    }

    pub fn phase_bytes(
        &self,
        phase: ObservedPhase,
        category: ObservedCategory,
        status: ObservedStatus,
    ) -> u64 {
        self.phase_totals[phase_total_index(phase, category, status)]
    }

    pub fn useful_bytes(&self, category: ObservedCategory) -> Result<u64, LedgerError> {
        ObservedStatus::ALL
            .into_iter()
            .filter(|status| status.is_useful())
            .try_fold(0_u64, |total, status| {
                total
                    .checked_add(self.bytes(category, status))
                    .ok_or(LedgerError::SnapshotTotalOverflow { category })
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LedgerFault {
    SubmissionAbandoned,
    StaleRecorderUse,
    UnreportedOperation,
    ContextOrderViolation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LedgerError {
    ZeroCapacity,
    Retiring,
    Closed,
    StaleEpoch {
        expected: u64,
        actual: u64,
    },
    PendingSubmissions {
        count: usize,
    },
    Faulted {
        fault: LedgerFault,
    },
    SubmissionExhausted,
    SequenceExhausted,
    EventCapacityExceeded {
        capacity: usize,
        committed: usize,
        requested: usize,
    },
    BatchCapacityExceeded {
        capacity: usize,
    },
    ContextCapacityExceeded {
        capacity: usize,
    },
    ByteTotalOverflow {
        phase: ObservedPhase,
        category: ObservedCategory,
        status: ObservedStatus,
        current: u64,
        added: u64,
    },
    SnapshotTotalOverflow {
        category: ObservedCategory,
    },
    SubmissionTerminal,
    InvalidAbortStatus {
        status: ObservedStatus,
    },
}

impl fmt::Display for LedgerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => {
                formatter.write_str("observed-byte event capacity must be greater than zero")
            }
            Self::Retiring => formatter.write_str(
                "observed-byte ledger is retiring; no new production operation may attach",
            ),
            Self::Closed => formatter.write_str(
                "observed-byte ledger is closed; create a new ledger for further production work",
            ),
            Self::StaleEpoch { expected, actual } => write!(
                formatter,
                "observed-byte recorder epoch {expected} is stale; ledger is at epoch {actual}"
            ),
            Self::PendingSubmissions { count } => write!(
                formatter,
                "observed-byte ledger has {count} in-flight submission(s); complete or abort them \
                 before snapshot/reset/close"
            ),
            Self::Faulted { fault } => {
                write!(formatter, "observed-byte ledger is faulted: {fault:?}")
            }
            Self::SubmissionExhausted => formatter
                .write_str("observed-byte submission identity space exhausted; refusing ABA reuse"),
            Self::SequenceExhausted => {
                formatter.write_str("observed-byte event sequence identity space exhausted")
            }
            Self::EventCapacityExceeded {
                capacity,
                committed,
                requested,
            } => write!(
                formatter,
                "observed-byte event capacity {capacity} cannot hold {requested} additional \
                 event(s) after {committed}; no event was committed and no production operation \
                 was submitted"
            ),
            Self::BatchCapacityExceeded { capacity } => write!(
                formatter,
                "observed-byte submission exceeds its fixed {capacity}-event batch capacity"
            ),
            Self::ContextCapacityExceeded { capacity } => write!(
                formatter,
                "observed-byte authenticated context depth exceeds fixed capacity {capacity}; \
                 refusing an unmeasured nested operation"
            ),
            Self::ByteTotalOverflow {
                phase,
                category,
                status,
                current,
                added,
            } => write!(
                formatter,
                "observed-byte {phase:?}/{category:?}/{status:?} total overflows at {current} + \
                 {added}; no production operation was submitted"
            ),
            Self::SnapshotTotalOverflow { category } => write!(
                formatter,
                "observed-byte useful snapshot total overflowed for {category:?}"
            ),
            Self::SubmissionTerminal => {
                formatter.write_str("observed-byte submission already reached a terminal outcome")
            }
            Self::InvalidAbortStatus { status } => write!(
                formatter,
                "observed-byte abort requires failed, rolled_back, or quarantined status, got \
                 {status:?}"
            ),
        }
    }
}

impl std::error::Error for LedgerError {}

#[derive(Clone, Copy)]
pub(crate) struct EventSpec {
    pub category: ObservedCategory,
    pub boundary: ObservedBoundary,
    pub status: ObservedStatus,
    pub bytes: u64,
}

impl EventSpec {
    pub(crate) const fn new(
        category: ObservedCategory,
        boundary: ObservedBoundary,
        status: ObservedStatus,
        bytes: u64,
    ) -> Self {
        Self {
            category,
            boundary,
            status,
            bytes,
        }
    }
}

struct EventSlot {
    state: AtomicU8,
    event: UnsafeCell<MaybeUninit<ObservedEvent>>,
}

impl EventSlot {
    fn empty() -> Self {
        Self {
            state: AtomicU8::new(SLOT_EMPTY),
            event: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}

// A slot has one writer selected by `next_event`; readers access it only after
// lifecycle admission is frozen and `active_batches == 0`.
unsafe impl Send for EventSlot {}
unsafe impl Sync for EventSlot {}

struct LedgerInner {
    scope: ObservedScope,
    runtime_id: u64,
    lifecycle: AtomicU8,
    epoch: AtomicU64,
    phase: AtomicU8,
    next_submission: AtomicU64,
    next_event: AtomicUsize,
    active_batches: AtomicUsize,
    active_recorders: AtomicUsize,
    fault: AtomicU8,
    events: Box<[EventSlot]>,
    accepted_totals: [AtomicU64; TOTAL_COUNT],
    totals: [AtomicU64; TOTAL_COUNT],
    phase_totals: [AtomicU64; PHASE_TOTAL_COUNT],
    context_entries: AtomicU64,
    batch_reservations: AtomicU64,
    retained_recorder_clones: AtomicU64,
}

impl LedgerInner {
    fn fault(&self) -> Option<LedgerFault> {
        match self.fault.load(Ordering::Acquire) {
            FAULT_NONE => None,
            FAULT_SUBMISSION_ABANDONED => Some(LedgerFault::SubmissionAbandoned),
            FAULT_STALE_RECORDER => Some(LedgerFault::StaleRecorderUse),
            FAULT_UNREPORTED_OPERATION => Some(LedgerFault::UnreportedOperation),
            FAULT_CONTEXT_ORDER => Some(LedgerFault::ContextOrderViolation),
            _ => Some(LedgerFault::UnreportedOperation),
        }
    }

    fn set_fault(&self, fault: LedgerFault) {
        let code = match fault {
            LedgerFault::SubmissionAbandoned => FAULT_SUBMISSION_ABANDONED,
            LedgerFault::StaleRecorderUse => FAULT_STALE_RECORDER,
            LedgerFault::UnreportedOperation => FAULT_UNREPORTED_OPERATION,
            LedgerFault::ContextOrderViolation => FAULT_CONTEXT_ORDER,
        };
        let _ = self
            .fault
            .compare_exchange(FAULT_NONE, code, Ordering::AcqRel, Ordering::Acquire);
    }

    fn ensure_unfaulted(&self) -> Result<(), LedgerError> {
        self.fault()
            .map_or(Ok(()), |fault| Err(LedgerError::Faulted { fault }))
    }

    fn ensure_epoch(&self, expected: u64) -> Result<(), LedgerError> {
        let actual = self.epoch.load(Ordering::Acquire);
        if actual != expected {
            self.set_fault(LedgerFault::StaleRecorderUse);
            return Err(LedgerError::StaleEpoch { expected, actual });
        }
        Ok(())
    }

    fn begin_batch(&self, epoch: u64, allow_retiring: bool) -> Result<(), LedgerError> {
        self.ensure_unfaulted()?;
        self.ensure_epoch(epoch)?;
        loop {
            match self.lifecycle.load(Ordering::Acquire) {
                LIFECYCLE_ACTIVE => {}
                LIFECYCLE_RETIRING if allow_retiring => {}
                LIFECYCLE_RETIRING | LIFECYCLE_SNAPSHOT_RETIRING => {
                    return Err(LedgerError::Retiring);
                }
                LIFECYCLE_CLOSED => return Err(LedgerError::Closed),
                LIFECYCLE_SNAPSHOT_ACTIVE => {
                    std::hint::spin_loop();
                    continue;
                }
                _ => return Err(LedgerError::Closed),
            }
            self.active_batches.fetch_add(1, Ordering::AcqRel);
            let admitted = self.lifecycle.load(Ordering::Acquire);
            if admitted == LIFECYCLE_ACTIVE || (allow_retiring && admitted == LIFECYCLE_RETIRING) {
                if let Err(error) = self
                    .ensure_unfaulted()
                    .and_then(|()| self.ensure_epoch(epoch))
                {
                    self.active_batches.fetch_sub(1, Ordering::AcqRel);
                    return Err(error);
                }
                return Ok(());
            }
            self.active_batches.fetch_sub(1, Ordering::AcqRel);
        }
    }

    fn end_batch(&self) {
        self.active_batches.fetch_sub(1, Ordering::Release);
    }

    fn freeze(&self) -> Result<FrozenLifecycle<'_>, LedgerError> {
        loop {
            let current = self.lifecycle.load(Ordering::Acquire);
            let frozen = match current {
                LIFECYCLE_ACTIVE => LIFECYCLE_SNAPSHOT_ACTIVE,
                LIFECYCLE_RETIRING => LIFECYCLE_SNAPSHOT_RETIRING,
                LIFECYCLE_CLOSED => {
                    return Ok(FrozenLifecycle {
                        inner: self,
                        restore: LIFECYCLE_CLOSED,
                        owns_transition: false,
                    });
                }
                LIFECYCLE_SNAPSHOT_ACTIVE | LIFECYCLE_SNAPSHOT_RETIRING => {
                    std::hint::spin_loop();
                    continue;
                }
                _ => return Err(LedgerError::Closed),
            };
            if self
                .lifecycle
                .compare_exchange(current, frozen, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let pending = self.active_batches.load(Ordering::Acquire);
                if pending != 0 {
                    self.lifecycle.store(current, Ordering::Release);
                    return Err(LedgerError::PendingSubmissions { count: pending });
                }
                return Ok(FrozenLifecycle {
                    inner: self,
                    restore: current,
                    owns_transition: true,
                });
            }
        }
    }
}

struct FrozenLifecycle<'a> {
    inner: &'a LedgerInner,
    restore: u8,
    owns_transition: bool,
}

impl FrozenLifecycle<'_> {
    fn close(mut self) {
        self.inner
            .lifecycle
            .store(LIFECYCLE_CLOSED, Ordering::Release);
        self.owns_transition = false;
    }
}

impl Drop for FrozenLifecycle<'_> {
    fn drop(&mut self) {
        if self.owns_transition {
            self.inner.lifecycle.store(self.restore, Ordering::Release);
        }
    }
}

#[derive(Clone)]
pub struct ObservedByteReadHandle {
    inner: Arc<LedgerInner>,
}

impl ObservedByteReadHandle {
    pub fn snapshot(&self) -> Result<ObservedSnapshot, LedgerError> {
        snapshot_inner(&self.inner)
    }

    pub fn hot_path_stats(&self) -> ObservedHotPathStats {
        hot_path_stats(&self.inner)
    }
}

fn snapshot_inner(inner: &Arc<LedgerInner>) -> Result<ObservedSnapshot, LedgerError> {
    let _frozen = inner.freeze()?;
    inner.ensure_unfaulted()?;
    let event_len = inner.next_event.load(Ordering::Acquire);
    let mut events = Vec::with_capacity(event_len);
    for slot in &inner.events[..event_len] {
        if slot.state.load(Ordering::Acquire) != SLOT_COMMITTED {
            return Err(LedgerError::PendingSubmissions { count: 1 });
        }
        // SAFETY: committed slots were initialized before the release store;
        // lifecycle admission is frozen and no batch remains active.
        events.push(unsafe { (*slot.event.get()).assume_init() });
    }
    Ok(ObservedSnapshot {
        schema: OBSERVED_BYTE_SCHEMA,
        scope: inner.scope,
        epoch: inner.epoch.load(Ordering::Acquire),
        phase: ObservedPhase::from_index(inner.phase.load(Ordering::Acquire)),
        events,
        totals: std::array::from_fn(|index| inner.totals[index].load(Ordering::Acquire)),
        phase_totals: std::array::from_fn(|index| {
            inner.phase_totals[index].load(Ordering::Acquire)
        }),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservedHotPathStats {
    pub context_entries: u64,
    pub batch_reservations: u64,
    pub retained_recorder_clones: u64,
    pub mutex_acquisitions: u64,
    pub thread_id_lookups: u64,
    pub vector_growths: u64,
}

/// Unique public read/reset/close handle for one observed ledger.
pub struct ObservedByteLedger {
    inner: Arc<LedgerInner>,
}

impl fmt::Debug for ObservedByteLedger {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ObservedByteLedger")
            .field("scope", &self.inner.scope)
            .finish_non_exhaustive()
    }
}

impl ObservedByteLedger {
    #[cfg(test)]
    pub(crate) fn new(scope: ObservedScope, event_capacity: usize) -> Result<Self, LedgerError> {
        Self::new_for_runtime(scope, 0, event_capacity)
    }

    pub(crate) fn new_for_runtime(
        scope: ObservedScope,
        runtime_id: u64,
        event_capacity: usize,
    ) -> Result<Self, LedgerError> {
        if event_capacity == 0 {
            return Err(LedgerError::ZeroCapacity);
        }
        Ok(Self {
            inner: Arc::new(LedgerInner {
                scope,
                runtime_id,
                lifecycle: AtomicU8::new(LIFECYCLE_ACTIVE),
                epoch: AtomicU64::new(1),
                phase: AtomicU8::new(ObservedPhase::Setup as u8),
                next_submission: AtomicU64::new(1),
                next_event: AtomicUsize::new(0),
                active_batches: AtomicUsize::new(0),
                active_recorders: AtomicUsize::new(0),
                fault: AtomicU8::new(FAULT_NONE),
                events: (0..event_capacity)
                    .map(|_| EventSlot::empty())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                accepted_totals: std::array::from_fn(|_| AtomicU64::new(0)),
                totals: std::array::from_fn(|_| AtomicU64::new(0)),
                phase_totals: std::array::from_fn(|_| AtomicU64::new(0)),
                context_entries: AtomicU64::new(0),
                batch_reservations: AtomicU64::new(0),
                retained_recorder_clones: AtomicU64::new(0),
            }),
        })
    }

    pub fn scope(&self) -> ObservedScope {
        self.inner.scope
    }

    pub fn set_phase(&self, phase: ObservedPhase) -> Result<(), LedgerError> {
        let _frozen = self.inner.freeze()?;
        self.inner.ensure_unfaulted()?;
        self.inner.phase.store(phase as u8, Ordering::Release);
        Ok(())
    }

    pub fn snapshot(&self) -> Result<ObservedSnapshot, LedgerError> {
        snapshot_inner(&self.inner)
    }

    pub fn read_handle(&self) -> ObservedByteReadHandle {
        ObservedByteReadHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    pub fn hot_path_stats(&self) -> ObservedHotPathStats {
        hot_path_stats(&self.inner)
    }
}

fn hot_path_stats(inner: &LedgerInner) -> ObservedHotPathStats {
    ObservedHotPathStats {
        context_entries: inner.context_entries.load(Ordering::Acquire),
        batch_reservations: inner.batch_reservations.load(Ordering::Acquire),
        retained_recorder_clones: inner.retained_recorder_clones.load(Ordering::Acquire),
        mutex_acquisitions: 0,
        thread_id_lookups: 0,
        vector_growths: 0,
    }
}

impl ObservedByteLedger {
    pub fn reset(&self) -> Result<(), LedgerError> {
        let _frozen = self.inner.freeze()?;
        let next_epoch = self
            .inner
            .epoch
            .load(Ordering::Acquire)
            .checked_add(1)
            .ok_or(LedgerError::SequenceExhausted)?;
        let event_len = self.inner.next_event.swap(0, Ordering::AcqRel);
        for slot in &self.inner.events[..event_len] {
            slot.state.store(SLOT_EMPTY, Ordering::Release);
        }
        self.inner.next_submission.store(1, Ordering::Release);
        self.inner.epoch.store(next_epoch, Ordering::Release);
        self.inner
            .phase
            .store(ObservedPhase::Setup as u8, Ordering::Release);
        self.inner.fault.store(FAULT_NONE, Ordering::Release);
        for total in &self.inner.accepted_totals {
            total.store(0, Ordering::Release);
        }
        for total in &self.inner.totals {
            total.store(0, Ordering::Release);
        }
        for total in &self.inner.phase_totals {
            total.store(0, Ordering::Release);
        }
        Ok(())
    }

    pub fn close(&self) -> Result<(), LedgerError> {
        if self.inner.lifecycle.load(Ordering::Acquire) == LIFECYCLE_CLOSED {
            return Ok(());
        }
        self.inner.freeze()?.close();
        Ok(())
    }

    pub(crate) fn recorder(&self) -> Result<ProductionByteRecorder, LedgerError> {
        match self.inner.lifecycle.load(Ordering::Acquire) {
            LIFECYCLE_ACTIVE => {}
            LIFECYCLE_RETIRING | LIFECYCLE_SNAPSHOT_RETIRING => {
                return Err(LedgerError::Retiring);
            }
            LIFECYCLE_CLOSED => return Err(LedgerError::Closed),
            _ => {}
        }
        self.inner.ensure_unfaulted()?;
        self.inner.active_recorders.fetch_add(1, Ordering::AcqRel);
        Ok(ProductionByteRecorder {
            inner: Arc::clone(&self.inner),
            epoch: AtomicU64::new(self.inner.epoch.load(Ordering::Acquire)),
            refreshable: false,
            allow_retiring: false,
        })
    }

    pub(crate) fn persistent_recorder(&self) -> Result<ProductionByteRecorder, LedgerError> {
        let mut recorder = self.recorder()?;
        recorder.refreshable = true;
        recorder.allow_retiring = true;
        Ok(recorder)
    }

    #[cfg(test)]
    pub(crate) fn attach(&self) -> Result<ProductionByteRecorder, LedgerError> {
        self.recorder()
    }

    pub(crate) fn retire(&self) {
        loop {
            match self.inner.lifecycle.load(Ordering::Acquire) {
                LIFECYCLE_ACTIVE => {
                    if self
                        .inner
                        .lifecycle
                        .compare_exchange(
                            LIFECYCLE_ACTIVE,
                            LIFECYCLE_RETIRING,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        break;
                    }
                }
                LIFECYCLE_SNAPSHOT_ACTIVE => std::hint::spin_loop(),
                LIFECYCLE_RETIRING | LIFECYCLE_SNAPSHOT_RETIRING | LIFECYCLE_CLOSED => break,
                _ => break,
            }
        }
        if self.inner.active_recorders.load(Ordering::Acquire) == 0
            && self.inner.active_batches.load(Ordering::Acquire) == 0
        {
            let _ = self.inner.lifecycle.compare_exchange(
                LIFECYCLE_RETIRING,
                LIFECYCLE_CLOSED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

pub(crate) struct ProductionByteRecorder {
    inner: Arc<LedgerInner>,
    epoch: AtomicU64,
    refreshable: bool,
    allow_retiring: bool,
}

impl fmt::Debug for ProductionByteRecorder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProductionByteRecorder")
            .field("scope", &self.inner.scope)
            .field("epoch", &self.epoch.load(Ordering::Acquire))
            .finish()
    }
}

impl Drop for ProductionByteRecorder {
    fn drop(&mut self) {
        let prior = self.inner.active_recorders.fetch_sub(1, Ordering::AcqRel);
        if prior == 1
            && self.inner.active_batches.load(Ordering::Acquire) == 0
            && self.inner.lifecycle.load(Ordering::Acquire) == LIFECYCLE_RETIRING
        {
            let _ = self.inner.lifecycle.compare_exchange(
                LIFECYCLE_RETIRING,
                LIFECYCLE_CLOSED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

impl ProductionByteRecorder {
    fn active_epoch(&self) -> Result<u64, LedgerError> {
        let expected = self.epoch.load(Ordering::Acquire);
        let actual = self.inner.epoch.load(Ordering::Acquire);
        if expected == actual {
            return Ok(actual);
        }
        if self.refreshable {
            self.inner.ensure_unfaulted()?;
            self.epoch.store(actual, Ordering::Release);
            return Ok(actual);
        }
        self.inner.ensure_epoch(expected)?;
        unreachable!("ensure_epoch returns on a mismatch")
    }

    pub(crate) fn taint_unreported_operation(&self) {
        self.inner.set_fault(LedgerFault::UnreportedOperation);
    }

    /// Cold-path ownership retained by a deferred release receipt. Executor
    /// entry and every ordinary CUDA operation borrow the prebuilt recorder
    /// instead and never call this.
    pub(crate) fn retain_for_deferred(&self) -> Result<Self, LedgerError> {
        let epoch = self.active_epoch()?;
        self.inner.ensure_unfaulted()?;
        self.inner
            .retained_recorder_clones
            .fetch_add(1, Ordering::Relaxed);
        self.inner.active_recorders.fetch_add(1, Ordering::AcqRel);
        Ok(Self {
            inner: Arc::clone(&self.inner),
            epoch: AtomicU64::new(epoch),
            refreshable: true,
            allow_retiring: true,
        })
    }

    pub(crate) fn enter(&self) -> Result<RecorderContextGuard, LedgerError> {
        self.active_epoch()?;
        self.inner.ensure_unfaulted()?;
        RECORDER_CONTEXT.with(|context| {
            // SAFETY: thread-local storage is accessed only by its owning thread.
            let stack = unsafe { &mut *context.get() };
            if stack.len == MAX_CONTEXT_DEPTH {
                return Err(LedgerError::ContextCapacityExceeded {
                    capacity: MAX_CONTEXT_DEPTH,
                });
            }
            let generation = stack
                .next_generation
                .checked_add(1)
                .ok_or(LedgerError::SequenceExhausted)?;
            stack.next_generation = generation;
            let index = stack.len;
            stack.entries[index] = ContextEntry {
                runtime_id: self.inner.runtime_id,
                recorder: NonNull::from(self),
                generation,
            };
            stack.len += 1;
            self.inner.context_entries.fetch_add(1, Ordering::Relaxed);
            Ok(RecorderContextGuard {
                index,
                generation,
                recorder: NonNull::from(self),
            })
        })
    }

    pub(crate) fn prepare(&self, specs: &[EventSpec]) -> Result<PendingObservedBatch, LedgerError> {
        PendingObservedBatch::prepare(self, specs)
    }

    pub(crate) fn record(&self, spec: EventSpec) -> Result<(), LedgerError> {
        let mut batch = self.prepare(std::slice::from_ref(&spec))?;
        batch.commit()
    }

    pub(crate) fn record_pair(
        &self,
        first: EventSpec,
        second: EventSpec,
    ) -> Result<(), LedgerError> {
        let mut batch = self.prepare(&[first, second])?;
        batch.commit()
    }
}

#[derive(Clone, Copy)]
struct ContextEntry {
    runtime_id: u64,
    recorder: NonNull<ProductionByteRecorder>,
    generation: u64,
}

const EMPTY_CONTEXT_ENTRY: ContextEntry = ContextEntry {
    runtime_id: 0,
    recorder: NonNull::dangling(),
    generation: 0,
};

struct ContextStack {
    entries: [ContextEntry; MAX_CONTEXT_DEPTH],
    len: usize,
    next_generation: u64,
}

thread_local! {
    static RECORDER_CONTEXT: UnsafeCell<ContextStack> = const {
        UnsafeCell::new(ContextStack {
            entries: [EMPTY_CONTEXT_ENTRY; MAX_CONTEXT_DEPTH],
            len: 0,
            next_generation: 0,
        })
    };
}

pub(crate) struct RecorderContextGuard {
    index: usize,
    generation: u64,
    recorder: NonNull<ProductionByteRecorder>,
}

impl Drop for RecorderContextGuard {
    fn drop(&mut self) {
        RECORDER_CONTEXT.with(|context| {
            // SAFETY: thread-local storage is accessed only by its owning thread.
            let stack = unsafe { &mut *context.get() };
            let valid = stack.len == self.index + 1
                && stack.entries[self.index].generation == self.generation
                && stack.entries[self.index].recorder == self.recorder;
            if valid {
                stack.len -= 1;
                stack.entries[self.index] = EMPTY_CONTEXT_ENTRY;
            } else {
                // SAFETY: the guard cannot outlive the borrowed recorder whose
                // `enter` method created it.
                unsafe { self.recorder.as_ref() }
                    .inner
                    .set_fault(LedgerFault::ContextOrderViolation);
            }
        });
    }
}

pub(crate) fn current_recorder(runtime_id: u64) -> Option<&'static ProductionByteRecorder> {
    RECORDER_CONTEXT.with(|context| {
        // SAFETY: callers use the reference only synchronously while the
        // authenticated outer guard is live. The pointer is never retained by
        // an ordinary batch; deferred cold paths explicitly clone ownership.
        let stack = unsafe { &*context.get() };
        stack.entries[..stack.len]
            .iter()
            .rev()
            .find(|entry| entry.runtime_id == runtime_id)
            .map(|entry| unsafe { entry.recorder.as_ref() })
    })
}

pub(crate) struct PendingObservedBatch {
    inner: NonNull<LedgerInner>,
    epoch: u64,
    submission: u64,
    phase: ObservedPhase,
    slot_start: usize,
    specs: [EventSpec; MAX_BATCH_EVENTS],
    reserved_specs: [EventSpec; MAX_BATCH_EVENTS],
    len: usize,
    submitted: bool,
    terminal: bool,
}

// The ledger storage is stable behind an Arc retained by the exact provider
// state for at least as long as any CUDA/deferred operation can own a batch.
unsafe impl Send for PendingObservedBatch {}

impl PendingObservedBatch {
    fn empty_spec() -> EventSpec {
        EventSpec::new(
            ObservedCategory::SourceRead,
            ObservedBoundary::AsyncCompletionUnsupported,
            ObservedStatus::Unsupported,
            0,
        )
    }

    fn prepare(
        recorder: &ProductionByteRecorder,
        specs: &[EventSpec],
    ) -> Result<Self, LedgerError> {
        if specs.len() > MAX_BATCH_EVENTS {
            return Err(LedgerError::BatchCapacityExceeded {
                capacity: MAX_BATCH_EVENTS,
            });
        }
        let inner = recorder.inner.as_ref();
        let epoch = recorder.active_epoch()?;
        inner.begin_batch(epoch, recorder.allow_retiring)?;
        let phase = ObservedPhase::from_index(inner.phase.load(Ordering::Acquire));
        if let Err(error) = reserve_total_alternatives(inner, specs, phase) {
            inner.end_batch();
            return Err(error);
        }
        let slot_start =
            match inner
                .next_event
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |committed| {
                    committed
                        .checked_add(specs.len())
                        .filter(|next| *next <= inner.events.len())
                }) {
                Ok(start) => start,
                Err(committed) => {
                    release_total_alternatives(inner, specs);
                    inner.end_batch();
                    return Err(LedgerError::EventCapacityExceeded {
                        capacity: inner.events.len(),
                        committed,
                        requested: specs.len(),
                    });
                }
            };
        let submission = match inner.next_submission.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |current| current.checked_add(1),
        ) {
            Ok(current) => current,
            Err(_) => {
                release_total_alternatives(inner, specs);
                inner.set_fault(LedgerFault::SubmissionAbandoned);
                inner.end_batch();
                return Err(LedgerError::SubmissionExhausted);
            }
        };
        let mut batch = Self {
            inner: NonNull::from(inner),
            epoch,
            submission,
            phase,
            slot_start,
            specs: [Self::empty_spec(); MAX_BATCH_EVENTS],
            reserved_specs: [Self::empty_spec(); MAX_BATCH_EVENTS],
            len: specs.len(),
            submitted: false,
            terminal: false,
        };
        batch.specs[..specs.len()].copy_from_slice(specs);
        batch.reserved_specs[..specs.len()].copy_from_slice(specs);
        inner.batch_reservations.fetch_add(1, Ordering::Relaxed);
        Ok(batch)
    }

    fn inner(&self) -> &LedgerInner {
        // SAFETY: see the Send invariant above.
        unsafe { self.inner.as_ref() }
    }

    pub(crate) fn mark_submitted(&mut self) {
        self.submitted = true;
    }

    pub(crate) fn set_boundary(&mut self, index: usize, boundary: ObservedBoundary) {
        if !self.terminal && index < self.len {
            self.specs[index].boundary = boundary;
        }
    }

    pub(crate) fn set_bytes(&mut self, index: usize, bytes: u64) -> Result<(), LedgerError> {
        if self.terminal {
            return Err(LedgerError::SubmissionTerminal);
        }
        if index >= self.len || bytes > self.reserved_specs[index].bytes {
            return Err(LedgerError::BatchCapacityExceeded { capacity: self.len });
        }
        self.specs[index].bytes = bytes;
        Ok(())
    }

    pub(crate) fn commit(&mut self) -> Result<(), LedgerError> {
        self.finish(None)
    }

    pub(crate) fn abort(&mut self, status: ObservedStatus) -> Result<(), LedgerError> {
        if !matches!(
            status,
            ObservedStatus::Failed | ObservedStatus::RolledBack | ObservedStatus::Quarantined
        ) {
            return Err(LedgerError::InvalidAbortStatus { status });
        }
        self.finish(Some(status))
    }

    fn finish(&mut self, abort: Option<ObservedStatus>) -> Result<(), LedgerError> {
        if self.terminal {
            return Ok(());
        }
        let inner = self.inner();
        inner.ensure_epoch(self.epoch)?;
        inner.ensure_unfaulted()?;

        let mut actual = self.specs;
        if let Some(status) = abort {
            for spec in &mut actual[..self.len] {
                if !(self.submitted && spec.status == ObservedStatus::Submitted) {
                    spec.status = status;
                }
            }
        }

        for (offset, spec) in actual[..self.len].iter().enumerate() {
            let slot = &inner.events[self.slot_start + offset];
            // SAFETY: this batch exclusively owns its contiguous reserved slots.
            unsafe {
                (*slot.event.get()).write(ObservedEvent {
                    scope: inner.scope,
                    epoch: self.epoch,
                    sequence: (self.slot_start + offset + 1) as u64,
                    submission: self.submission,
                    phase: self.phase,
                    stream: spec.boundary.stream(),
                    category: spec.category,
                    boundary: spec.boundary,
                    status: spec.status,
                    bytes: spec.bytes,
                });
            }
            inner.totals[total_index(spec.category, spec.status)]
                .fetch_add(spec.bytes, Ordering::Relaxed);
            inner.phase_totals[phase_total_index(self.phase, spec.category, spec.status)]
                .fetch_add(spec.bytes, Ordering::Relaxed);
            slot.state.store(SLOT_COMMITTED, Ordering::Release);
        }
        release_unused_total_alternatives(
            inner,
            &self.reserved_specs[..self.len],
            &actual[..self.len],
        );
        inner.end_batch();
        self.terminal = true;
        Ok(())
    }

    fn abandon_inner(&mut self) {
        if self.terminal {
            return;
        }
        let inner = self.inner();
        release_total_alternatives(inner, &self.reserved_specs[..self.len]);
        inner.set_fault(LedgerFault::SubmissionAbandoned);
        inner.end_batch();
        self.terminal = true;
    }
}

impl Drop for PendingObservedBatch {
    fn drop(&mut self) {
        self.abandon_inner();
    }
}

fn alternative_statuses(success: ObservedStatus) -> [ObservedStatus; 4] {
    [
        success,
        ObservedStatus::Failed,
        ObservedStatus::RolledBack,
        ObservedStatus::Quarantined,
    ]
}

fn reserve_total_alternatives(
    inner: &LedgerInner,
    specs: &[EventSpec],
    phase: ObservedPhase,
) -> Result<(), LedgerError> {
    let mut reserved = [(0usize, 0u64); MAX_BATCH_EVENTS * 4];
    let mut reserved_len = 0usize;
    for spec in specs {
        let statuses = alternative_statuses(spec.status);
        for (status_index, status) in statuses.into_iter().enumerate() {
            if statuses[..status_index].contains(&status) {
                continue;
            }
            let index = total_index(spec.category, status);
            if let Err(current) = inner.accepted_totals[index].fetch_update(
                Ordering::AcqRel,
                Ordering::Acquire,
                |value| value.checked_add(spec.bytes),
            ) {
                for &(reserved_index, bytes) in reserved[..reserved_len].iter().rev() {
                    inner.accepted_totals[reserved_index].fetch_sub(bytes, Ordering::AcqRel);
                }
                return Err(LedgerError::ByteTotalOverflow {
                    phase,
                    category: spec.category,
                    status,
                    current,
                    added: spec.bytes,
                });
            }
            reserved[reserved_len] = (index, spec.bytes);
            reserved_len += 1;
        }
    }
    Ok(())
}

fn release_total_alternatives(inner: &LedgerInner, specs: &[EventSpec]) {
    for spec in specs {
        let statuses = alternative_statuses(spec.status);
        for (index, status) in statuses.into_iter().enumerate() {
            if statuses[..index].contains(&status) {
                continue;
            }
            inner.accepted_totals[total_index(spec.category, status)]
                .fetch_sub(spec.bytes, Ordering::AcqRel);
        }
    }
}

fn release_unused_total_alternatives(
    inner: &LedgerInner,
    reserved: &[EventSpec],
    actual: &[EventSpec],
) {
    for (reserved, actual) in reserved.iter().zip(actual) {
        let statuses = alternative_statuses(reserved.status);
        for (index, status) in statuses.into_iter().enumerate() {
            if statuses[..index].contains(&status) {
                continue;
            }
            let keep = if status == actual.status {
                actual.bytes
            } else {
                0
            };
            inner.accepted_totals[total_index(reserved.category, status)]
                .fetch_sub(reserved.bytes - keep, Ordering::AcqRel);
        }
    }
}

const fn total_index(category: ObservedCategory, status: ObservedStatus) -> usize {
    category.index() * STATUS_COUNT + status.index()
}

const fn phase_total_index(
    phase: ObservedPhase,
    category: ObservedCategory,
    status: ObservedStatus,
) -> usize {
    phase.index() * TOTAL_COUNT + total_index(category, status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::{Arc, Barrier};

    struct CountingAllocator;

    thread_local! {
        static COUNT_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
        static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    }

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            COUNT_ALLOCATIONS.with(|enabled| {
                if enabled.get() {
                    ALLOCATIONS.with(|count| count.set(count.get() + 1));
                }
            });
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            COUNT_ALLOCATIONS.with(|enabled| {
                if enabled.get() {
                    ALLOCATIONS.with(|count| count.set(count.get() + 1));
                }
            });
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            COUNT_ALLOCATIONS.with(|enabled| {
                if enabled.get() {
                    ALLOCATIONS.with(|count| count.set(count.get() + 1));
                }
            });
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static TEST_ALLOCATOR: CountingAllocator = CountingAllocator;

    fn count_allocations(operation: impl FnOnce()) -> u64 {
        ALLOCATIONS.with(|count| count.set(0));
        COUNT_ALLOCATIONS.with(|enabled| enabled.set(true));
        operation();
        COUNT_ALLOCATIONS.with(|enabled| enabled.set(false));
        ALLOCATIONS.with(Cell::get)
    }

    fn scope() -> ObservedScope {
        ObservedScope {
            provider: 7,
            device: 0,
            executor: 11,
            generation: 13,
            logical_session: 17,
        }
    }

    fn completed(bytes: u64) -> EventSpec {
        EventSpec::new(
            ObservedCategory::H2d,
            ObservedBoundary::RuntimeH2d,
            ObservedStatus::Completed,
            bytes,
        )
    }

    #[test]
    fn bounded_capacity_fails_before_a_new_batch_exists() {
        let ledger = ObservedByteLedger::new(scope(), 1).unwrap();
        let recorder = ledger.attach().unwrap();
        recorder.record(completed(3)).unwrap();
        assert!(matches!(
            recorder.prepare(&[completed(1)]),
            Err(LedgerError::EventCapacityExceeded { .. })
        ));
        assert_eq!(ledger.snapshot().unwrap().events.len(), 1);
    }

    #[test]
    fn total_overflow_is_preflighted_without_partial_publication() {
        let ledger = ObservedByteLedger::new(scope(), 3).unwrap();
        let recorder = ledger.attach().unwrap();
        recorder.record(completed(u64::MAX)).unwrap();
        assert!(matches!(
            recorder.prepare(&[completed(1)]),
            Err(LedgerError::ByteTotalOverflow { .. })
        ));
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.events.len(), 1);
        assert_eq!(
            snapshot.bytes(ObservedCategory::H2d, ObservedStatus::Completed),
            u64::MAX
        );
    }

    #[test]
    fn abort_after_submission_preserves_attempt_and_records_failure() {
        let ledger = ObservedByteLedger::new(scope(), 2).unwrap();
        let recorder = ledger.attach().unwrap();
        let mut batch = recorder
            .prepare(&[
                EventSpec::new(
                    ObservedCategory::D2h,
                    ObservedBoundary::RuntimeD2h,
                    ObservedStatus::Submitted,
                    9,
                ),
                EventSpec::new(
                    ObservedCategory::D2h,
                    ObservedBoundary::RuntimeD2h,
                    ObservedStatus::Completed,
                    9,
                ),
            ])
            .unwrap();
        batch.mark_submitted();
        batch.abort(ObservedStatus::Failed).unwrap();
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(
            snapshot.bytes(ObservedCategory::D2h, ObservedStatus::Submitted),
            9
        );
        assert_eq!(
            snapshot.bytes(ObservedCategory::D2h, ObservedStatus::Failed),
            9
        );
        assert_eq!(
            snapshot.bytes(ObservedCategory::D2h, ObservedStatus::Completed),
            0
        );
    }

    #[test]
    fn concurrent_producers_reserve_unique_slots_without_global_lock() {
        let ledger = Arc::new(ObservedByteLedger::new(scope(), 128).unwrap());
        let start = Arc::new(Barrier::new(9));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let ledger = Arc::clone(&ledger);
            let start = Arc::clone(&start);
            workers.push(std::thread::spawn(move || {
                let recorder = ledger.attach().unwrap();
                start.wait();
                for _ in 0..16 {
                    recorder.record(completed(1)).unwrap();
                }
            }));
        }
        start.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.events.len(), 128);
        let mut sequences = snapshot
            .events
            .iter()
            .map(|event| event.sequence)
            .collect::<Vec<_>>();
        sequences.sort_unstable();
        sequences.dedup();
        assert_eq!(sequences.len(), 128);
        assert_eq!(
            snapshot.bytes(ObservedCategory::H2d, ObservedStatus::Completed),
            128
        );
        let stats = ledger.hot_path_stats();
        assert_eq!(stats.mutex_acquisitions, 0);
        assert_eq!(stats.thread_id_lookups, 0);
        assert_eq!(stats.vector_growths, 0);
    }

    #[test]
    fn tls_context_is_bounded_lifo_and_rejects_stale_epoch() {
        let ledger = ObservedByteLedger::new(scope(), 8).unwrap();
        let recorder = ledger.attach().unwrap();
        let guard = recorder.enter().unwrap();
        assert!(std::ptr::eq(
            current_recorder(ledger.inner.runtime_id).unwrap(),
            &recorder
        ));
        drop(guard);
        assert!(current_recorder(ledger.inner.runtime_id).is_none());
        ledger.reset().unwrap();
        assert!(matches!(
            recorder.enter(),
            Err(LedgerError::StaleEpoch { .. })
        ));
    }

    #[test]
    fn reset_and_close_refuse_live_batches_and_stale_handles_cannot_publish() {
        let ledger = ObservedByteLedger::new(scope(), 8).unwrap();
        let recorder = ledger.attach().unwrap();
        let batch = recorder.prepare(&[completed(1)]).unwrap();
        assert!(matches!(
            ledger.reset(),
            Err(LedgerError::PendingSubmissions { count: 1 })
        ));
        assert!(matches!(
            ledger.close(),
            Err(LedgerError::PendingSubmissions { count: 1 })
        ));
        drop(batch);
        assert!(matches!(
            ledger.snapshot(),
            Err(LedgerError::Faulted {
                fault: LedgerFault::SubmissionAbandoned
            })
        ));
        ledger.reset().unwrap();
        ledger.close().unwrap();
        assert!(matches!(
            recorder.record(completed(1)),
            Err(LedgerError::Closed | LedgerError::StaleEpoch { .. })
        ));
    }

    #[test]
    fn nested_context_capacity_fails_before_unmeasured_work() {
        let ledger = ObservedByteLedger::new(scope(), 8).unwrap();
        let recorder = ledger.attach().unwrap();
        let mut guards = Vec::new();
        for _ in 0..MAX_CONTEXT_DEPTH {
            guards.push(recorder.enter().unwrap());
        }
        assert!(matches!(
            recorder.enter(),
            Err(LedgerError::ContextCapacityExceeded {
                capacity: MAX_CONTEXT_DEPTH
            })
        ));
        drop(guards);
    }

    #[test]
    fn warmed_recording_has_zero_allocations_locks_thread_lookups_or_arc_retains() {
        let ledger = ObservedByteLedger::new(scope(), 512).unwrap();
        let recorder = ledger.attach().unwrap();
        let allocations = count_allocations(|| {
            let _context = recorder.enter().unwrap();
            for _ in 0..128 {
                recorder.record(completed(1)).unwrap();
            }
        });
        assert_eq!(allocations, 0);
        let stats = ledger.hot_path_stats();
        assert_eq!(stats.context_entries, 1);
        assert_eq!(stats.batch_reservations, 128);
        assert_eq!(stats.retained_recorder_clones, 0);
        assert_eq!(stats.mutex_acquisitions, 0);
        assert_eq!(stats.thread_id_lookups, 0);
        assert_eq!(stats.vector_growths, 0);
    }

    #[test]
    fn first_middle_and_last_capacity_boundaries_are_atomic() {
        for used in 0..=3 {
            let ledger = ObservedByteLedger::new(scope(), 3).unwrap();
            let recorder = ledger.attach().unwrap();
            for _ in 0..used {
                recorder.record(completed(1)).unwrap();
            }
            let before = ledger.snapshot().unwrap();
            let result = recorder.prepare(&[completed(2)]);
            if used == 3 {
                assert!(matches!(
                    result,
                    Err(LedgerError::EventCapacityExceeded { .. })
                ));
                assert_eq!(ledger.snapshot().unwrap(), before);
            } else {
                result.unwrap().commit().unwrap();
                assert_eq!(ledger.snapshot().unwrap().events.len(), used + 1);
            }
        }
    }

    #[test]
    fn duplicate_terminal_calls_publish_once() {
        let ledger = ObservedByteLedger::new(scope(), 2).unwrap();
        let recorder = ledger.attach().unwrap();
        let mut batch = recorder.prepare(&[completed(9)]).unwrap();
        batch.commit().unwrap();
        batch.commit().unwrap();
        batch.abort(ObservedStatus::RolledBack).unwrap();
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.events.len(), 1);
        assert_eq!(
            snapshot.bytes(ObservedCategory::H2d, ObservedStatus::Completed),
            9
        );
        assert_eq!(
            snapshot.bytes(ObservedCategory::H2d, ObservedStatus::RolledBack),
            0
        );
    }

    #[test]
    fn snapshot_refuses_an_active_batch_then_observes_its_atomic_terminal_state() {
        let ledger = ObservedByteLedger::new(scope(), 2).unwrap();
        let recorder = ledger.attach().unwrap();
        let mut batch = recorder.prepare(&[completed(5)]).unwrap();
        assert!(matches!(
            ledger.snapshot(),
            Err(LedgerError::PendingSubmissions { count: 1 })
        ));
        batch.abort(ObservedStatus::RolledBack).unwrap();
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.events.len(), 1);
        assert_eq!(
            snapshot.bytes(ObservedCategory::H2d, ObservedStatus::RolledBack),
            5
        );
    }

    #[test]
    fn persistent_owner_refreshes_after_reset_but_stale_handle_does_not() {
        let ledger = ObservedByteLedger::new(scope(), 4).unwrap();
        let persistent = ledger.persistent_recorder().unwrap();
        let stale = ledger.attach().unwrap();
        persistent.record(completed(1)).unwrap();
        ledger.reset().unwrap();
        persistent.record(completed(2)).unwrap();
        assert!(matches!(
            stale.record(completed(3)),
            Err(LedgerError::StaleEpoch { .. })
        ));
        let snapshot = ledger.snapshot().unwrap_err();
        assert!(matches!(
            snapshot,
            LedgerError::Faulted {
                fault: LedgerFault::StaleRecorderUse
            }
        ));
    }

    #[test]
    fn retirement_blocks_fresh_authority_but_allows_preissued_teardown_receipts() {
        let ledger = ObservedByteLedger::new(scope(), 2).unwrap();
        let persistent = ledger.persistent_recorder().unwrap();
        ledger.retire();
        assert!(matches!(ledger.attach(), Err(LedgerError::Retiring)));
        persistent
            .record(EventSpec::new(
                ObservedCategory::DeviceRelease,
                ObservedBoundary::RuntimeDeviceRelease,
                ObservedStatus::Reclaimed,
                64,
            ))
            .unwrap();
        assert_eq!(ledger.snapshot().unwrap().events.len(), 1);
    }

    #[test]
    fn closed_snapshot_is_immutable_and_post_close_recording_is_rejected() {
        let ledger = ObservedByteLedger::new(scope(), 2).unwrap();
        let recorder = ledger.attach().unwrap();
        recorder.record(completed(7)).unwrap();
        ledger.close().unwrap();
        ledger.close().unwrap();
        assert_eq!(ledger.snapshot().unwrap().events.len(), 1);
        assert!(matches!(
            recorder.record(completed(1)),
            Err(LedgerError::Closed)
        ));
    }
}
