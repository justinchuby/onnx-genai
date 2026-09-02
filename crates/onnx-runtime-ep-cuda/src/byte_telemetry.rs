//! Bounded production-boundary byte telemetry.
//!
//! The ledger is opt-in and owned by one exact
//! provider/device/executor/generation/logical-session scope. Production
//! boundaries receive a private recorder only when the owning CUDA provider
//! opens an exact executor session. The
//! default runtime contains no recorder and performs no telemetry allocation,
//! lock, lookup, copy, synchronization, or environment read.
//!
//! Enabled recording uses a construction-time preallocated event ring. A
//! submission is prepared in fixed stack storage, validated in full, then
//! committed under one short atomic gate. Snapshot/reset use the same
//! linearization authority, so readers see the complete batch or none of it.
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
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;

pub const OBSERVED_BYTE_SCHEMA: &str = "onnx-genai.freetoken-observed-bytes.v1";

const MAX_BATCH_EVENTS: usize = 16;
const PHASE_COUNT: usize = 9;
const CATEGORY_COUNT: usize = 16;
const STATUS_COUNT: usize = 9;
const TOTAL_COUNT: usize = CATEGORY_COUNT * STATUS_COUNT;
const PHASE_TOTAL_COUNT: usize = PHASE_COUNT * TOTAL_COUNT;

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

#[derive(Clone)]
pub struct ObservedByteReadHandle {
    inner: Arc<LedgerInner>,
}

impl ObservedByteReadHandle {
    pub fn snapshot(&self) -> Result<ObservedSnapshot, LedgerError> {
        snapshot_inner(&self.inner)
    }
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
}

fn snapshot_inner(inner: &Arc<LedgerInner>) -> Result<ObservedSnapshot, LedgerError> {
    let guard = inner.lock();
    let state = guard.state();
    ensure_unfaulted(state)?;
    if state.pending != 0 {
        return Err(LedgerError::PendingSubmissions {
            count: state.pending,
        });
    }
    Ok(ObservedSnapshot {
        schema: OBSERVED_BYTE_SCHEMA,
        scope: inner.scope,
        epoch: state.epoch,
        phase: state.phase,
        events: state.events[..state.event_len].to_vec(),
        totals: state.totals,
        phase_totals: state.phase_totals,
    })
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
            Self::PageIn | Self::ExpertPublication | Self::StatePublication => {
                ObservedLayer::LogicalPublication
            }
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
    AsyncCompletionUnsupported,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
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
            Self::SequenceExhausted => formatter
                .write_str("observed-byte event sequence space exhausted; refusing wraparound"),
            Self::EventCapacityExceeded {
                capacity,
                committed,
                requested,
            } => write!(
                formatter,
                "observed-byte event capacity {capacity} cannot hold {requested} additional \
                 event(s) after {committed}; no event was committed"
            ),
            Self::BatchCapacityExceeded { capacity } => write!(
                formatter,
                "observed-byte submission exceeds its fixed {capacity}-event batch capacity"
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
                 {added}; no event was committed"
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

#[derive(Debug)]
struct LedgerState {
    epoch: u64,
    next_submission: u64,
    next_sequence: u64,
    phase: ObservedPhase,
    events: Box<[ObservedEvent]>,
    event_len: usize,
    totals: [u64; TOTAL_COUNT],
    phase_totals: [u64; PHASE_TOTAL_COUNT],
    reserved_events: usize,
    reserved_totals: [u64; TOTAL_COUNT],
    reserved_phase_totals: [u64; PHASE_TOTAL_COUNT],
    pending: usize,
    active_recorders: usize,
    retiring: bool,
    closed: bool,
    fault: Option<LedgerFault>,
}

struct LedgerInner {
    scope: ObservedScope,
    runtime_id: u64,
    gate: AtomicBool,
    state: UnsafeCell<LedgerState>,
}

// Every access to `state` is serialized by `gate`. The gate is never held
// across a CUDA call or allocation.
unsafe impl Send for LedgerInner {}
unsafe impl Sync for LedgerInner {}

struct GateGuard<'a> {
    inner: &'a LedgerInner,
}

impl Drop for GateGuard<'_> {
    fn drop(&mut self) {
        self.inner.gate.store(false, Ordering::Release);
    }
}

impl GateGuard<'_> {
    fn state(&self) -> &LedgerState {
        // SAFETY: this guard holds the sole gate for immutable access.
        unsafe { &*self.inner.state.get() }
    }

    fn state_mut(&mut self) -> &mut LedgerState {
        // SAFETY: this mutable guard holds the sole gate for mutable access.
        unsafe { &mut *self.inner.state.get() }
    }
}

impl LedgerInner {
    fn lock(&self) -> GateGuard<'_> {
        let mut spins = 0_u32;
        while self
            .gate
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            if spins < 64 {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        GateGuard { inner: self }
    }
}

/// The unique public read/reset/close handle for one observed ledger.
///
/// This type is intentionally not `Clone`. Production mutation is possible
/// only through the private recorder installed by `CudaRuntime::new_observed`.
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
        let empty = ObservedEvent {
            scope,
            epoch: 0,
            sequence: 0,
            submission: 0,
            phase: ObservedPhase::Setup,
            category: ObservedCategory::SourceRead,
            boundary: ObservedBoundary::AsyncCompletionUnsupported,
            status: ObservedStatus::Unsupported,
            bytes: 0,
        };
        Ok(Self {
            inner: Arc::new(LedgerInner {
                scope,
                runtime_id,
                gate: AtomicBool::new(false),
                state: UnsafeCell::new(LedgerState {
                    epoch: 1,
                    next_submission: 1,
                    next_sequence: 1,
                    phase: ObservedPhase::Setup,
                    events: vec![empty; event_capacity].into_boxed_slice(),
                    event_len: 0,
                    totals: [0; TOTAL_COUNT],
                    phase_totals: [0; PHASE_TOTAL_COUNT],
                    reserved_events: 0,
                    reserved_totals: [0; TOTAL_COUNT],
                    reserved_phase_totals: [0; PHASE_TOTAL_COUNT],
                    pending: 0,
                    active_recorders: 0,
                    retiring: false,
                    closed: false,
                    fault: None,
                }),
            }),
        })
    }

    pub fn scope(&self) -> ObservedScope {
        self.inner.scope
    }

    pub fn set_phase(&self, phase: ObservedPhase) -> Result<(), LedgerError> {
        let mut guard = self.inner.lock();
        let state = guard.state_mut();
        ensure_available(state)?;
        if state.pending != 0 {
            return Err(LedgerError::PendingSubmissions {
                count: state.pending,
            });
        }
        state.phase = phase;
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

    pub fn reset(&self) -> Result<(), LedgerError> {
        let mut guard = self.inner.lock();
        let state = guard.state_mut();
        if state.closed {
            return Err(LedgerError::Closed);
        }
        if state.pending != 0 {
            return Err(LedgerError::PendingSubmissions {
                count: state.pending,
            });
        }
        state.epoch = state
            .epoch
            .checked_add(1)
            .ok_or(LedgerError::SequenceExhausted)?;
        state.next_submission = 1;
        state.next_sequence = 1;
        state.event_len = 0;
        state.totals.fill(0);
        state.phase_totals.fill(0);
        state.reserved_events = 0;
        state.reserved_totals.fill(0);
        state.reserved_phase_totals.fill(0);
        state.fault = None;
        Ok(())
    }

    pub fn close(&self) -> Result<(), LedgerError> {
        let mut guard = self.inner.lock();
        let state = guard.state_mut();
        if state.closed {
            return Ok(());
        }
        if state.pending != 0 {
            return Err(LedgerError::PendingSubmissions {
                count: state.pending,
            });
        }
        state.closed = true;
        state.retiring = false;
        Ok(())
    }

    pub(crate) fn recorder(&self) -> Result<ProductionByteRecorder, LedgerError> {
        let mut guard = self.inner.lock();
        let state = guard.state_mut();
        ensure_available(state)?;
        state.active_recorders = state
            .active_recorders
            .checked_add(1)
            .ok_or(LedgerError::SubmissionExhausted)?;
        Ok(ProductionByteRecorder {
            inner: Arc::clone(&self.inner),
            epoch: state.epoch,
        })
    }

    #[cfg(test)]
    pub(crate) fn attach(&self) -> Result<ProductionByteRecorder, LedgerError> {
        self.recorder()
    }

    pub(crate) fn retire(&self) {
        let mut guard = self.inner.lock();
        let state = guard.state_mut();
        state.retiring = true;
        if state.active_recorders == 0 {
            state.closed = true;
        }
    }
}

pub(crate) struct ProductionByteRecorder {
    inner: Arc<LedgerInner>,
    epoch: u64,
}

impl Drop for ProductionByteRecorder {
    fn drop(&mut self) {
        let mut guard = self.inner.lock();
        let state = guard.state_mut();
        state.active_recorders = state.active_recorders.saturating_sub(1);
        if state.active_recorders == 0 && state.retiring {
            state.closed = true;
        }
    }
}

impl fmt::Debug for ProductionByteRecorder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProductionByteRecorder")
            .field("scope", &self.inner.scope)
            .field("epoch", &self.epoch)
            .finish()
    }
}

impl ProductionByteRecorder {
    pub(crate) fn taint_unreported_operation(&self) {
        let mut guard = self.inner.lock();
        guard.state_mut().fault = Some(LedgerFault::UnreportedOperation);
    }

    pub(crate) fn duplicate(&self) -> Result<Self, LedgerError> {
        let mut guard = self.inner.lock();
        let state = guard.state_mut();
        ensure_recorder(state, self.epoch)?;
        state.active_recorders = state
            .active_recorders
            .checked_add(1)
            .ok_or(LedgerError::SubmissionExhausted)?;
        Ok(Self {
            inner: Arc::clone(&self.inner),
            epoch: self.epoch,
        })
    }

    pub(crate) fn begin(&self) -> Result<PendingObservedBatch, LedgerError> {
        let mut guard = self.inner.lock();
        let state = guard.state_mut();
        ensure_recorder(state, self.epoch)?;
        let submission = state.next_submission;
        state.next_submission = submission
            .checked_add(1)
            .ok_or(LedgerError::SubmissionExhausted)?;
        state.pending = state
            .pending
            .checked_add(1)
            .ok_or(LedgerError::SubmissionExhausted)?;
        state.active_recorders = state
            .active_recorders
            .checked_add(1)
            .ok_or(LedgerError::SubmissionExhausted)?;
        Ok(PendingObservedBatch {
            recorder: ProductionByteRecorder {
                inner: Arc::clone(&self.inner),
                epoch: self.epoch,
            },
            epoch: self.epoch,
            submission,
            phase: state.phase,
            specs: [EventSpec::new(
                ObservedCategory::SourceRead,
                ObservedBoundary::AsyncCompletionUnsupported,
                ObservedStatus::Unsupported,
                0,
            ); MAX_BATCH_EVENTS],
            reserved_specs: [EventSpec::new(
                ObservedCategory::SourceRead,
                ObservedBoundary::AsyncCompletionUnsupported,
                ObservedStatus::Unsupported,
                0,
            ); MAX_BATCH_EVENTS],
            len: 0,
            reserved: false,
            terminal: false,
        })
    }

    pub(crate) fn prepare(&self, specs: &[EventSpec]) -> Result<PendingObservedBatch, LedgerError> {
        if specs.len() > MAX_BATCH_EVENTS {
            return Err(LedgerError::BatchCapacityExceeded {
                capacity: MAX_BATCH_EVENTS,
            });
        }
        let mut guard = self.inner.lock();
        let state = guard.state_mut();
        ensure_recorder(state, self.epoch)?;
        reserve_specs(state, specs, state.phase)?;
        let submission = state.next_submission;
        state.next_submission = submission
            .checked_add(1)
            .ok_or(LedgerError::SubmissionExhausted)?;
        state.pending = state
            .pending
            .checked_add(1)
            .ok_or(LedgerError::SubmissionExhausted)?;
        state.active_recorders = state
            .active_recorders
            .checked_add(1)
            .ok_or(LedgerError::SubmissionExhausted)?;
        let mut batch = PendingObservedBatch {
            recorder: ProductionByteRecorder {
                inner: Arc::clone(&self.inner),
                epoch: self.epoch,
            },
            epoch: self.epoch,
            submission,
            phase: state.phase,
            specs: [EventSpec::new(
                ObservedCategory::SourceRead,
                ObservedBoundary::AsyncCompletionUnsupported,
                ObservedStatus::Unsupported,
                0,
            ); MAX_BATCH_EVENTS],
            reserved_specs: [EventSpec::new(
                ObservedCategory::SourceRead,
                ObservedBoundary::AsyncCompletionUnsupported,
                ObservedStatus::Unsupported,
                0,
            ); MAX_BATCH_EVENTS],
            len: specs.len(),
            reserved: true,
            terminal: false,
        };
        batch.specs[..specs.len()].copy_from_slice(specs);
        batch.reserved_specs[..specs.len()].copy_from_slice(specs);
        Ok(batch)
    }
}

pub(crate) struct ExecutionRecorderRegistry {
    active: std::sync::Mutex<HashMap<std::thread::ThreadId, Vec<ProductionByteRecorder>>>,
}

impl Default for ExecutionRecorderRegistry {
    fn default() -> Self {
        Self {
            active: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl ExecutionRecorderRegistry {
    pub(crate) fn enter(
        self: &Arc<Self>,
        recorder: ProductionByteRecorder,
    ) -> RecorderContextGuard {
        let thread = std::thread::current().id();
        self.active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(thread)
            .or_default()
            .push(recorder);
        RecorderContextGuard {
            registry: Arc::clone(self),
            thread,
        }
    }

    pub(crate) fn current(
        &self,
        runtime_id: u64,
    ) -> Result<Option<ProductionByteRecorder>, LedgerError> {
        let active = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        active
            .get(&std::thread::current().id())
            .and_then(|stack| {
                stack
                    .iter()
                    .rev()
                    .find(|recorder| recorder.inner.runtime_id == runtime_id)
            })
            .map(ProductionByteRecorder::duplicate)
            .transpose()
    }
}

pub(crate) struct RecorderContextGuard {
    registry: Arc<ExecutionRecorderRegistry>,
    thread: std::thread::ThreadId,
}

impl Drop for RecorderContextGuard {
    fn drop(&mut self) {
        let mut active = self
            .registry
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(stack) = active.get_mut(&self.thread) {
            stack.pop();
            if stack.is_empty() {
                active.remove(&self.thread);
            }
        }
    }
}

impl ProductionByteRecorder {
    pub(crate) fn record(&self, spec: EventSpec) -> Result<(), LedgerError> {
        let mut batch = self.begin()?;
        batch.push(spec)?;
        batch.commit()
    }

    pub(crate) fn record_pair(
        &self,
        first: EventSpec,
        second: EventSpec,
    ) -> Result<(), LedgerError> {
        let mut batch = self.begin()?;
        batch.push(first)?;
        batch.push(second)?;
        batch.commit()
    }
}

pub(crate) struct PendingObservedBatch {
    recorder: ProductionByteRecorder,
    epoch: u64,
    submission: u64,
    phase: ObservedPhase,
    specs: [EventSpec; MAX_BATCH_EVENTS],
    reserved_specs: [EventSpec; MAX_BATCH_EVENTS],
    len: usize,
    reserved: bool,
    terminal: bool,
}

impl PendingObservedBatch {
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

    pub(crate) fn push(&mut self, spec: EventSpec) -> Result<(), LedgerError> {
        if self.terminal {
            return Err(LedgerError::SubmissionTerminal);
        }
        if self.len == MAX_BATCH_EVENTS {
            return Err(LedgerError::BatchCapacityExceeded {
                capacity: MAX_BATCH_EVENTS,
            });
        }
        self.specs[self.len] = spec;
        self.len += 1;
        Ok(())
    }

    pub(crate) fn commit(&mut self) -> Result<(), LedgerError> {
        if self.terminal {
            return Ok(());
        }
        let mut guard = self.recorder.inner.lock();
        let state = guard.state_mut();
        ensure_recorder(state, self.epoch)?;
        if self.reserved {
            release_reserved_specs(state, &self.reserved_specs[..self.len], self.phase);
            self.reserved = false;
        }
        let new_len =
            state
                .event_len
                .checked_add(self.len)
                .ok_or(LedgerError::EventCapacityExceeded {
                    capacity: state.events.len(),
                    committed: state.event_len,
                    requested: self.len,
                })?;
        if new_len > state.events.len() {
            return Err(LedgerError::EventCapacityExceeded {
                capacity: state.events.len(),
                committed: state.event_len,
                requested: self.len,
            });
        }
        let sequence_after = state
            .next_sequence
            .checked_add(self.len as u64)
            .ok_or(LedgerError::SequenceExhausted)?;

        let mut totals = state.totals;
        let mut phase_totals = state.phase_totals;
        for spec in &self.specs[..self.len] {
            let total = total_index(spec.category, spec.status);
            totals[total] =
                totals[total]
                    .checked_add(spec.bytes)
                    .ok_or(LedgerError::ByteTotalOverflow {
                        phase: self.phase,
                        category: spec.category,
                        status: spec.status,
                        current: totals[total],
                        added: spec.bytes,
                    })?;
            let phase_total = phase_total_index(self.phase, spec.category, spec.status);
            phase_totals[phase_total] = phase_totals[phase_total].checked_add(spec.bytes).ok_or(
                LedgerError::ByteTotalOverflow {
                    phase: self.phase,
                    category: spec.category,
                    status: spec.status,
                    current: phase_totals[phase_total],
                    added: spec.bytes,
                },
            )?;
        }

        for (offset, spec) in self.specs[..self.len].iter().enumerate() {
            state.events[state.event_len + offset] = ObservedEvent {
                scope: self.recorder.inner.scope,
                epoch: self.epoch,
                sequence: state.next_sequence + offset as u64,
                submission: self.submission,
                phase: self.phase,
                category: spec.category,
                boundary: spec.boundary,
                status: spec.status,
                bytes: spec.bytes,
            };
        }
        state.totals = totals;
        state.phase_totals = phase_totals;
        state.next_sequence = sequence_after;
        state.event_len = new_len;
        state.pending -= 1;
        self.terminal = true;
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn abort(&mut self, status: ObservedStatus) -> Result<(), LedgerError> {
        if self.terminal {
            return Ok(());
        }
        if !matches!(
            status,
            ObservedStatus::Failed | ObservedStatus::RolledBack | ObservedStatus::Quarantined
        ) {
            return Err(LedgerError::InvalidAbortStatus { status });
        }
        for spec in &mut self.specs[..self.len] {
            spec.status = status;
        }
        self.commit()
    }

    #[cfg(test)]
    fn abandon(&mut self) {
        self.abandon_inner();
    }

    fn abandon_inner(&mut self) {
        if self.terminal {
            return;
        }
        let mut guard = self.recorder.inner.lock();
        let state = guard.state_mut();
        if state.epoch == self.epoch && state.pending > 0 {
            if self.reserved {
                release_reserved_specs(state, &self.reserved_specs[..self.len], self.phase);
                self.reserved = false;
            }
            state.pending -= 1;
            state.fault = Some(LedgerFault::SubmissionAbandoned);
        }

        self.terminal = true;
    }
}

fn reserve_specs(
    state: &mut LedgerState,
    specs: &[EventSpec],
    phase: ObservedPhase,
) -> Result<(), LedgerError> {
    let requested = state.reserved_events.checked_add(specs.len()).ok_or(
        LedgerError::EventCapacityExceeded {
            capacity: state.events.len(),
            committed: state.event_len,
            requested: specs.len(),
        },
    )?;
    let occupied =
        state
            .event_len
            .checked_add(requested)
            .ok_or(LedgerError::EventCapacityExceeded {
                capacity: state.events.len(),
                committed: state.event_len,
                requested: specs.len(),
            })?;
    if occupied > state.events.len() {
        return Err(LedgerError::EventCapacityExceeded {
            capacity: state.events.len(),
            committed: state.event_len + state.reserved_events,
            requested: specs.len(),
        });
    }
    state
        .next_sequence
        .checked_add(requested as u64)
        .ok_or(LedgerError::SequenceExhausted)?;
    let mut reserved_totals = state.reserved_totals;
    let mut reserved_phase_totals = state.reserved_phase_totals;
    for spec in specs {
        let total = total_index(spec.category, spec.status);
        let current = state.totals[total]
            .checked_add(reserved_totals[total])
            .ok_or(LedgerError::ByteTotalOverflow {
                phase,
                category: spec.category,
                status: spec.status,
                current: state.totals[total],
                added: reserved_totals[total],
            })?;
        reserved_totals[total] = reserved_totals[total].checked_add(spec.bytes).ok_or(
            LedgerError::ByteTotalOverflow {
                phase,
                category: spec.category,
                status: spec.status,
                current,
                added: spec.bytes,
            },
        )?;
        state.totals[total]
            .checked_add(reserved_totals[total])
            .ok_or(LedgerError::ByteTotalOverflow {
                phase,
                category: spec.category,
                status: spec.status,
                current,
                added: spec.bytes,
            })?;
        let phase_total = phase_total_index(phase, spec.category, spec.status);
        let current_phase = state.phase_totals[phase_total]
            .checked_add(reserved_phase_totals[phase_total])
            .ok_or(LedgerError::ByteTotalOverflow {
                phase,
                category: spec.category,
                status: spec.status,
                current: state.phase_totals[phase_total],
                added: reserved_phase_totals[phase_total],
            })?;
        reserved_phase_totals[phase_total] = reserved_phase_totals[phase_total]
            .checked_add(spec.bytes)
            .ok_or(LedgerError::ByteTotalOverflow {
                phase,
                category: spec.category,
                status: spec.status,
                current: current_phase,
                added: spec.bytes,
            })?;
        state.phase_totals[phase_total]
            .checked_add(reserved_phase_totals[phase_total])
            .ok_or(LedgerError::ByteTotalOverflow {
                phase,
                category: spec.category,
                status: spec.status,
                current: current_phase,
                added: spec.bytes,
            })?;
    }
    state.reserved_events = requested;
    state.reserved_totals = reserved_totals;
    state.reserved_phase_totals = reserved_phase_totals;
    Ok(())
}

fn release_reserved_specs(state: &mut LedgerState, specs: &[EventSpec], phase: ObservedPhase) {
    state.reserved_events = state.reserved_events.saturating_sub(specs.len());
    for spec in specs {
        let total = total_index(spec.category, spec.status);
        state.reserved_totals[total] = state.reserved_totals[total].saturating_sub(spec.bytes);
        let phase_total = phase_total_index(phase, spec.category, spec.status);
        state.reserved_phase_totals[phase_total] =
            state.reserved_phase_totals[phase_total].saturating_sub(spec.bytes);
    }
}

impl Drop for PendingObservedBatch {
    fn drop(&mut self) {
        self.abandon_inner();
    }
}

fn ensure_available(state: &LedgerState) -> Result<(), LedgerError> {
    if state.closed {
        return Err(LedgerError::Closed);
    }
    if state.retiring {
        return Err(LedgerError::Retiring);
    }
    ensure_unfaulted(state)
}

fn ensure_unfaulted(state: &LedgerState) -> Result<(), LedgerError> {
    state
        .fault
        .map_or(Ok(()), |fault| Err(LedgerError::Faulted { fault }))
}

fn ensure_recorder(state: &mut LedgerState, epoch: u64) -> Result<(), LedgerError> {
    if state.closed {
        return Err(LedgerError::Closed);
    }
    ensure_unfaulted(state)?;
    if state.epoch != epoch {
        state.fault = Some(LedgerFault::StaleRecorderUse);
        return Err(LedgerError::StaleEpoch {
            expected: epoch,
            actual: state.epoch,
        });
    }
    Ok(())
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
    use std::sync::Barrier;

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
    fn exact_overflow_probe_commits_nothing_and_keeps_submission_pending() {
        let ledger = ObservedByteLedger::new(scope(), 4).unwrap();
        let recorder = ledger.attach().unwrap();
        let mut batch = recorder.begin().unwrap();
        batch.push(completed(u64::MAX)).unwrap();
        batch.push(completed(1)).unwrap();
        let error = batch.commit().unwrap_err();
        assert!(matches!(error, LedgerError::ByteTotalOverflow { .. }));
        assert!(matches!(
            ledger.snapshot(),
            Err(LedgerError::PendingSubmissions { count: 1 })
        ));
        {
            let guard = ledger.inner.lock();
            let state = guard.state();
            assert_eq!(state.event_len, 0);
            assert_eq!(
                state.totals[total_index(ObservedCategory::H2d, ObservedStatus::Completed)],
                0
            );
        }

        batch.abandon();
        assert!(matches!(
            ledger.snapshot(),
            Err(LedgerError::Faulted {
                fault: LedgerFault::SubmissionAbandoned
            })
        ));
    }

    #[test]
    fn prepared_operation_reserves_capacity_and_failure_releases_the_success_totals() {
        let ledger = ObservedByteLedger::new(scope(), 1).unwrap();
        let recorder = ledger.attach().unwrap();
        let mut first = recorder.prepare(&[completed(7)]).unwrap();
        assert!(matches!(
            recorder.prepare(&[completed(1)]),
            Err(LedgerError::EventCapacityExceeded {
                capacity: 1,
                committed: 1,
                requested: 1,
            })
        ));
        first.abort(ObservedStatus::Failed).unwrap();
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(
            snapshot.bytes(ObservedCategory::H2d, ObservedStatus::Completed),
            0
        );
        assert_eq!(
            snapshot.bytes(ObservedCategory::H2d, ObservedStatus::Failed),
            7
        );
    }

    #[test]
    fn prepared_operation_detects_total_overflow_before_submission() {
        let ledger = ObservedByteLedger::new(scope(), 2).unwrap();
        let recorder = ledger.attach().unwrap();
        recorder.record(completed(u64::MAX)).unwrap();
        assert!(matches!(
            recorder.prepare(&[completed(1)]),
            Err(LedgerError::ByteTotalOverflow {
                current: u64::MAX,
                added: 1,
                ..
            })
        ));
        assert_eq!(ledger.snapshot().unwrap().events.len(), 1);
    }

    #[test]
    fn capacity_exhaustion_at_first_middle_and_last_event_is_atomic() {
        for capacity in [0_usize, 1, 2] {
            let ledger = ObservedByteLedger::new(scope(), 2).unwrap();
            let recorder = ledger.attach().unwrap();
            if capacity > 0 {
                let mut initial = recorder.begin().unwrap();
                for _ in 0..capacity {
                    initial.push(completed(1)).unwrap();
                }
                initial.commit().unwrap();
            }
            let before = ledger.snapshot().unwrap();
            let mut batch = recorder.begin().unwrap();
            for _ in 0..3 {
                batch.push(completed(1)).unwrap();
            }
            assert!(matches!(
                batch.commit(),
                Err(LedgerError::EventCapacityExceeded { .. })
            ));
            {
                let guard = ledger.inner.lock();
                assert_eq!(guard.state().event_len, before.events.len());
            }
            batch.abandon();
        }
    }

    #[test]
    fn total_overflow_at_first_middle_and_last_event_never_publishes_a_prefix() {
        for overflow_index in 0..3 {
            let ledger = ObservedByteLedger::new(scope(), 8).unwrap();
            let recorder = ledger.attach().unwrap();
            {
                let mut guard = ledger.inner.lock();
                let state = guard.state_mut();
                state.totals[total_index(ObservedCategory::H2d, ObservedStatus::Completed)] =
                    u64::MAX;
                state.phase_totals[phase_total_index(
                    ObservedPhase::Setup,
                    ObservedCategory::H2d,
                    ObservedStatus::Completed,
                )] = u64::MAX;
            }
            let mut batch = recorder.begin().unwrap();
            for index in 0..3 {
                batch
                    .push(if index == overflow_index {
                        completed(1)
                    } else {
                        EventSpec::new(
                            ObservedCategory::D2d,
                            ObservedBoundary::RuntimeD2d,
                            ObservedStatus::Completed,
                            1,
                        )
                    })
                    .unwrap();
            }
            assert!(matches!(
                batch.commit(),
                Err(LedgerError::ByteTotalOverflow { .. })
            ));
            {
                let guard = ledger.inner.lock();
                let state = guard.state();
                assert_eq!(state.event_len, 0);
                assert_eq!(
                    state.totals[total_index(ObservedCategory::D2d, ObservedStatus::Completed)],
                    0
                );
            }
            batch.abandon();
        }
    }

    #[test]
    fn sequence_and_submission_exhaustion_fail_before_visible_mutation() {
        let ledger = ObservedByteLedger::new(scope(), 4).unwrap();
        let recorder = ledger.attach().unwrap();
        {
            let mut guard = ledger.inner.lock();
            guard.state_mut().next_sequence = u64::MAX;
        }
        let mut batch = recorder.begin().unwrap();
        batch.push(completed(1)).unwrap();
        assert!(matches!(
            batch.commit(),
            Err(LedgerError::SequenceExhausted)
        ));
        {
            let guard = ledger.inner.lock();
            assert_eq!(guard.state().event_len, 0);
        }
        batch.abandon();

        let second = ObservedByteLedger::new(scope(), 4).unwrap();
        let second_recorder = second.attach().unwrap();
        {
            let mut guard = second.inner.lock();
            guard.state_mut().next_submission = u64::MAX;
        }
        assert!(matches!(
            second_recorder.begin(),
            Err(LedgerError::SubmissionExhausted)
        ));
        assert!(second.snapshot().unwrap().events.is_empty());
    }

    #[test]
    fn abort_is_idempotent_and_keeps_non_useful_bytes_distinct() {
        let ledger = ObservedByteLedger::new(scope(), 4).unwrap();
        let recorder = ledger.attach().unwrap();
        let mut batch = recorder.begin().unwrap();
        batch
            .push(EventSpec::new(
                ObservedCategory::H2d,
                ObservedBoundary::RuntimeH2d,
                ObservedStatus::Submitted,
                64,
            ))
            .unwrap();
        batch.abort(ObservedStatus::RolledBack).unwrap();
        batch.abort(ObservedStatus::RolledBack).unwrap();
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(
            snapshot.bytes(ObservedCategory::H2d, ObservedStatus::RolledBack),
            64
        );
        assert_eq!(snapshot.useful_bytes(ObservedCategory::H2d).unwrap(), 0);
        assert_eq!(snapshot.events.len(), 1);
    }

    #[test]
    fn failed_completion_can_retry_without_promoting_failed_bytes() {
        let ledger = ObservedByteLedger::new(scope(), 8).unwrap();
        let recorder = ledger.attach().unwrap();
        let mut failed = recorder.begin().unwrap();
        failed
            .push(EventSpec::new(
                ObservedCategory::H2d,
                ObservedBoundary::RuntimeH2d,
                ObservedStatus::Submitted,
                32,
            ))
            .unwrap();
        failed.abort(ObservedStatus::Failed).unwrap();
        recorder
            .record_pair(
                EventSpec::new(
                    ObservedCategory::H2d,
                    ObservedBoundary::RuntimeH2d,
                    ObservedStatus::Submitted,
                    32,
                ),
                completed(32),
            )
            .unwrap();
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(
            snapshot.bytes(ObservedCategory::H2d, ObservedStatus::Failed),
            32
        );
        assert_eq!(snapshot.useful_bytes(ObservedCategory::H2d).unwrap(), 32);
    }

    #[test]
    fn sibling_with_same_public_scope_has_distinct_instance_state() {
        let first = ObservedByteLedger::new(scope(), 4).unwrap();
        let second = ObservedByteLedger::new(scope(), 4).unwrap();
        let first_recorder = first.attach().unwrap();
        let second_recorder = second.attach().unwrap();
        first_recorder.record(completed(3)).unwrap();
        second_recorder.record(completed(5)).unwrap();
        assert_eq!(
            first
                .snapshot()
                .unwrap()
                .bytes(ObservedCategory::H2d, ObservedStatus::Completed),
            3
        );
        assert_eq!(
            second
                .snapshot()
                .unwrap()
                .bytes(ObservedCategory::H2d, ObservedStatus::Completed),
            5
        );
    }

    #[test]
    fn moved_ledger_remains_bound_and_stale_recorder_fails_after_reset() {
        let ledger = ObservedByteLedger::new(scope(), 4).unwrap();
        let recorder = ledger.attach().unwrap();
        let moved = ledger;
        recorder.record(completed(2)).unwrap();
        assert_eq!(moved.snapshot().unwrap().events.len(), 1);
        moved.reset().unwrap();
        assert!(matches!(
            recorder.record(completed(2)),
            Err(LedgerError::StaleEpoch { .. })
        ));
        assert!(matches!(
            moved.snapshot(),
            Err(LedgerError::Faulted {
                fault: LedgerFault::StaleRecorderUse
            })
        ));
    }

    #[test]
    fn duplicate_finish_and_close_are_idempotent() {
        let ledger = ObservedByteLedger::new(scope(), 4).unwrap();
        let recorder = ledger.attach().unwrap();
        let mut batch = recorder.begin().unwrap();
        batch.push(completed(9)).unwrap();
        batch.commit().unwrap();
        batch.commit().unwrap();
        assert_eq!(ledger.snapshot().unwrap().events.len(), 1);
        ledger.close().unwrap();
        ledger.close().unwrap();
        assert_eq!(ledger.snapshot().unwrap().events.len(), 1);
    }

    #[test]
    fn snapshot_race_observes_only_whole_batches() {
        let ledger = Arc::new(ObservedByteLedger::new(scope(), 128).unwrap());
        let recorder = ledger.attach().unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let reader_ledger = Arc::clone(&ledger);
        let reader_barrier = Arc::clone(&barrier);
        let reader = std::thread::spawn(move || {
            reader_barrier.wait();
            for _ in 0..1000 {
                match reader_ledger.snapshot() {
                    Ok(snapshot) => assert_eq!(snapshot.events.len() % 2, 0),
                    Err(LedgerError::PendingSubmissions { .. }) => {}
                    Err(error) => panic!("unexpected snapshot error: {error}"),
                }
            }
        });
        barrier.wait();
        for _ in 0..32 {
            recorder.record_pair(completed(1), completed(1)).unwrap();
        }
        reader.join().unwrap();
        assert_eq!(ledger.snapshot().unwrap().events.len(), 64);
    }

    #[test]
    fn reset_and_close_refuse_in_flight_submission() {
        let ledger = ObservedByteLedger::new(scope(), 4).unwrap();
        let recorder = ledger.attach().unwrap();
        let mut batch = recorder.begin().unwrap();
        assert!(matches!(
            ledger.reset(),
            Err(LedgerError::PendingSubmissions { count: 1 })
        ));
        assert!(matches!(
            ledger.close(),
            Err(LedgerError::PendingSubmissions { count: 1 })
        ));
        batch.push(completed(1)).unwrap();
        batch.commit().unwrap();
        ledger.reset().unwrap();
    }

    #[test]
    fn post_close_recorder_is_stale_and_cannot_publish() {
        let ledger = ObservedByteLedger::new(scope(), 4).unwrap();
        let recorder = ledger.attach().unwrap();
        ledger.close().unwrap();
        assert!(matches!(
            recorder.record(completed(1)),
            Err(LedgerError::Closed)
        ));
    }
}
