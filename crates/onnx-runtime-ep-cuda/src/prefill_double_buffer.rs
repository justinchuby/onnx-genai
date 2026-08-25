//! Whole-layer prefill double-buffering primitive (FreeToken §3.1 derived).
//!
//! This is the smallest reviewable generalization of the CUDA EP's existing
//! *single-slot* ahead-of-need prefetch
//! ([`crate::weight_paging::CudaWeightResidency::prefetch_block_quantized_moe`])
//! into a **two explicitly owned stable staging slots** pipeline, so that layer
//! `N` computes while layer `N+1`'s transfer/preparation overlaps it. It is a
//! *staging* pipeline only: it never becomes a second residency/cache authority
//! and never remaps VMM/PMM memory — the existing paging lifecycle stays the
//! sole mapping/accounting authority (see the module-level contract below).
//!
//! # What this is (and is not)
//!
//! * It **is** a state machine + fencing choreographer over two slots, plus the
//!   capacity gate and typed refusal that decide *whether* the pipeline may run
//!   at all. All of that logic is device-free and lives here, tested with a fake
//!   transfer.
//! * It is **not** the CUDA implementation. The device-touching operations —
//!   reserving the two stable buffers, CPU-filling staging, enqueuing the H2D
//!   copy, and the four fence primitives — are behind the [`PrefillTransfer`]
//!   trait. `weight_paging.rs` supplies the real, `CudaRuntime`-backed impl;
//!   the tests here supply a deterministic fake so every failure mode (stale
//!   generation, transfer failure, OOM, cancellation, teardown-with-in-flight,
//!   capture rejection, leak/accounting) is exercised without a GPU.
//!
//! # Contract (matches `roy-freetoken-qstar-prefill-design.md` §5)
//!
//! * **Single authority.** The two slots draw their staging from the *same*
//!   pinned pool the decode residency owns; the capacity gate is the same
//!   [`crate::pinned_pool::PinnedStagingPool::can_retain_concurrent`] check the
//!   single-slot path uses. This primitive never maps or accounts device memory
//!   itself.
//! * **No host synchronize on the consume hot path.** `wait`/`release` are
//!   enqueue-only stream/event ops. The *only* host wait is at slot **reuse**
//!   (inside the ahead-of-need `prefetch`, never the consume path), and its
//!   duration is the falsifiable overlap metric ([`PrefillMetrics::reuse_wait_ns`],
//!   the two-slot analogue of the single-slot `prefetch_promote_wait_ns`):
//!   near-zero means the transfer was fully hidden.
//! * **Two-directional fencing.** ready→consume (`compute_wait` on the copy
//!   fence before a consumer reads a slot) and release→reuse (`copy_wait` on the
//!   previous consumer's release fence before a refill overwrites a slot's
//!   device buffer). Reusing a slot before its release fence is recorded is a
//!   hard state-machine refusal, not a silent race.
//! * **Fail closed.** Insufficient pool capacity, active capture, an unsupported
//!   layer, or an exhausted pipeline all return a typed [`PrefillReject`] so the
//!   caller falls back to today's synchronous single-buffer path; nothing is
//!   partially reserved and no success-shaped fallback is invented.
//! * **Default off / byte-identical.** Constructing a [`PrefillDoubleBuffer`] is
//!   the opt-in. When the caller never constructs one, no code path here runs
//!   and behavior is byte-identical to today.

use std::cell::RefCell;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{LazyWeight, MmapRegionSource};

use crate::pinned_pool::PinnedStagingPool;
use crate::runtime::{CudaRuntime, FailedHtodCompletion, PinnedStaging};
use crate::weight_paging::fill_staging_from_regions;

/// An opaque fence id in the abstract transfer engine, mirroring the `u64` ids
/// [`crate::runtime::CudaRuntime::record_copy_fence`] /
/// [`crate::runtime::CudaRuntime::record_compute_fence`] hand out. `0` is the
/// already-signalled/no-op fence, exactly as the runtime treats it.
pub type FenceId = u64;

/// Number of pipeline slots. Two is the whole point: one consumes while the
/// other fills. Kept as a named constant so the arithmetic below reads clearly.
const SLOTS: usize = 2;

/// Exact ownership state of one of the two prefill staging slots.
///
/// The legal transitions are:
/// `Free ─claim→ Filling ─fill ok→ Ready ─wait→ InUse ─release→ Draining`
/// then `Draining ─reuse claim→ Filling …` (wraparound). A failed transfer or
/// fence at any point moves the slot to `Poisoned`, which is terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillSlotStatus {
    /// No layer has ever occupied this slot; it may be claimed for a first fill.
    Free,
    /// A CPU fill + H2D enqueue is in progress (claimed, no copy fence yet).
    Filling,
    /// The layer's transfer is enqueued and a copy fence recorded. The device
    /// bytes are *not* valid to a consumer until it orders itself after that
    /// fence via `compute_wait`.
    Ready,
    /// A consumer has ordered its compute stream after the copy fence and is
    /// (or will be) reading the slot's device buffer.
    InUse,
    /// The consumer recorded a release fence. The slot's device buffer must not
    /// be overwritten by a refill until the copy stream waits on that fence.
    Draining,
    /// A transfer or fence failed while this slot held bytes. Its memory is
    /// quarantined and the slot must never be reused.
    Poisoned,
}

impl PrefillSlotStatus {
    /// Whether a new fill may claim this slot (a fresh slot, or one whose
    /// previous occupant has been released and is now safe to overwrite behind
    /// the release fence).
    fn is_claimable(self) -> bool {
        matches!(self, Self::Free | Self::Draining)
    }
}

/// Why the double buffer refused an operation. Every arm is a *property* of the
/// storage/location/lifecycle, never a model name.
#[derive(Debug)]
pub enum PrefillReject<E> {
    /// The backing pool cannot retain [`SLOTS`] concurrently-live buffers of the
    /// layer's byte size. The caller must use the synchronous single-buffer
    /// path. (OOM / insufficient-budget maps here.)
    PoolCapacity { layer_bytes: u64 },
    /// A CUDA graph capture/replay is active; reserving buffers or mutating the
    /// allocator is forbidden. The reservation must happen at a safe point.
    CaptureActive,
    /// The layer has zero transferable bytes — nothing to prefetch.
    EmptyLayer,
    /// Both slots are still occupied by not-yet-released layers; the pipeline is
    /// at its depth-2 limit. The caller must consume+release before prefetching
    /// further (this is also the reuse-before-release guard).
    SlotsBusy,
    /// `wait`/`release`/`cancel` referenced a ticket whose slot has since been
    /// reused for a different layer (generation advanced). The stale operation
    /// is refused rather than handed the wrong layer's bytes.
    StaleGeneration { layer_id: u64 },
    /// `wait`/`release`/`cancel` referenced a slot that is not in the state that
    /// operation requires.
    WrongState {
        layer_id: u64,
        status: PrefillSlotStatus,
    },
    /// The referenced slot is poisoned and can never be used again.
    Poisoned { layer_id: u64 },
    /// The whole-layer prefill double buffer is not enabled. It is opt-in and
    /// default-off (see the module doc); the caller must use the synchronous
    /// single-buffer path. Surfaced only by the gated construction entry, never
    /// by the primitive itself, so the shipped default path stays byte-identical.
    Disabled,
    /// The underlying transfer/fence failed. The slot has been poisoned and its
    /// memory quarantined; this is the error the transfer reported.
    Transfer(E),
}

impl<E: fmt::Display> fmt::Display for PrefillReject<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PoolCapacity { layer_bytes } => write!(
                f,
                "prefill double buffer declined: pinned pool cannot retain {SLOTS} concurrent \
                 buffers of {layer_bytes} bytes; fall back to synchronous single-buffer prefill"
            ),
            Self::CaptureActive => write!(
                f,
                "prefill double buffer declined: CUDA graph capture/replay active, buffer \
                 reservation must run at a scheduler safe point"
            ),
            Self::EmptyLayer => {
                write!(
                    f,
                    "prefill double buffer declined: layer has zero transferable bytes"
                )
            }
            Self::SlotsBusy => write!(
                f,
                "prefill double buffer declined: both slots occupied by not-yet-released layers \
                 (pipeline depth limit; consume and release before prefetching further)"
            ),
            Self::StaleGeneration { layer_id } => write!(
                f,
                "prefill double buffer refused a stale operation for layer {layer_id}: its slot \
                 was already reused for another layer"
            ),
            Self::WrongState { layer_id, status } => write!(
                f,
                "prefill double buffer refused an operation for layer {layer_id}: slot is {status:?}"
            ),
            Self::Poisoned { layer_id } => write!(
                f,
                "prefill double buffer refused an operation for layer {layer_id}: slot is poisoned"
            ),
            Self::Disabled => write!(
                f,
                "prefill double buffer is disabled (default-off): set \
                 ONNX_GENAI_PREFILL_DOUBLE_BUFFER=1 to enable, or use the synchronous \
                 single-buffer prefill path"
            ),
            Self::Transfer(error) => write!(f, "prefill double buffer transfer failed: {error}"),
        }
    }
}

/// The fencing context handed to [`PrefillTransfer::fill_slot`] when a slot is
/// (re)filled, so the transfer can enforce both WAR hazards a reused stable slot
/// carries.
#[derive(Clone, Copy, Debug, Default)]
pub struct SlotFillPlan {
    /// The slot's own previous copy fence, if it is being reused. The transfer
    /// must host-wait this before CPU-refilling the slot's stable staging so the
    /// refill cannot race the previous H2D read of that pinned buffer. `None`
    /// on a slot's first fill.
    pub prev_copy_fence: Option<FenceId>,
    /// The previous consumer's release fence, if any. The transfer must make the
    /// copy stream wait on it (`copy_wait`) before enqueuing the H2D copy, so a
    /// refill never overwrites device bytes the previous consumer is still
    /// reading. `None` on a slot's first fill or a cancelled-before-consume fill.
    pub prev_release_fence: Option<FenceId>,
}

/// What [`PrefillTransfer::fill_slot`] produced.
#[derive(Clone, Copy, Debug)]
pub struct FillOutcome {
    /// The copy-stream fence recorded after the H2D copy was enqueued. A
    /// consumer orders itself after this; a later reuse host-waits it.
    pub copy_fence: FenceId,
    /// Host time spent waiting on the slot's *previous* copy fence before the
    /// refill (0 on a first fill). This is the overlap metric.
    pub reuse_wait_ns: u64,
}

/// Per-slot teardown descriptor passed to [`PrefillTransfer::teardown`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SlotTeardown {
    /// The slot's last copy-stream fence, if it still holds an enqueued H2D
    /// transfer whose completion must be established before its buffer is freed.
    pub last_copy_fence: Option<FenceId>,
    /// The slot's last compute-stream (release) fence, if a consumer that was
    /// enqueued against this buffer must complete before it is freed.
    pub last_release_fence: Option<FenceId>,
    /// Whether the slot still owns a reserved buffer that teardown must release.
    /// Every reserved slot is live except a poisoned one, whose buffer stays
    /// quarantined and is reported `false`.
    pub live: bool,
}

/// The device-touching operations the double buffer orchestrates. Implemented
/// for real over [`crate::runtime::CudaRuntime`] +
/// [`crate::pinned_pool::PinnedStagingPool`] in `weight_paging.rs`, and by a
/// deterministic fake in this module's tests.
///
/// The primitive owns *sequencing, state, generations, and metrics*; the
/// transfer owns *physical buffers and CUDA calls*. The transfer is addressed by
/// slot index (`0..SLOTS`); it holds the two stable per-slot buffers internally.
pub trait PrefillTransfer {
    /// Cloneable consumer handle to a filled slot's device buffer (real: the
    /// layer's device pages; fake: a copy of the slot's filled bytes/id).
    type Payload: Clone;
    /// Describes one layer's source bytes (real: expert keys + lazy weights +
    /// mmap source; fake: an id + byte count).
    type LayerReq;
    /// The transfer's error type.
    type Error: fmt::Display;

    /// Byte size of one layer's buffer, from a request. Used by the capacity
    /// gate and the empty-layer check.
    fn layer_bytes(&self, req: &Self::LayerReq) -> u64;

    /// True while a device operation forbids allocator/VMM mutation (capture or
    /// replay in progress). Checked before every reservation and refill.
    fn capture_active(&self) -> bool;

    /// Whether the backing pool can retain [`SLOTS`] concurrent buffers of
    /// `layer_bytes` each without evicting one on release.
    fn can_retain_concurrent(&self, layer_bytes: u64) -> bool;

    /// Reserve the two stable slot buffers of `layer_bytes` each at a safe
    /// point. Called once by [`PrefillDoubleBuffer::new`]. Must be all-or-none.
    fn reserve(&mut self, layer_bytes: u64) -> Result<(), Self::Error>;

    /// (Re)fill `slot` from `req` under `plan`'s fencing, returning the new copy
    /// fence and the reuse host-wait observed. Enqueue-only on the device side
    /// apart from the single documented reuse host-wait.
    fn fill_slot(
        &self,
        slot: usize,
        req: &Self::LayerReq,
        plan: SlotFillPlan,
    ) -> Result<FillOutcome, Self::Error>;

    /// The consumer handle for a currently-filled `slot`.
    fn payload(&self, slot: usize) -> Self::Payload;

    /// Order the compute stream after `copy_fence` (enqueue-only, no host sync).
    fn compute_wait(&self, copy_fence: FenceId) -> Result<(), Self::Error>;

    /// Record and return a compute-stream fence marking a consumer's completion.
    fn record_release_fence(&self) -> Result<FenceId, Self::Error>;

    /// Quarantine `slot`'s buffer after a failed transition; it is never reused.
    fn quarantine_slot(&self, slot: usize);

    /// Release the reserved buffers. Called exactly once, from `Drop`. The
    /// transfer must establish each live slot's `last_copy_fence` completion
    /// before freeing its buffer, and must not touch quarantined slots.
    fn teardown(&mut self, slots: [SlotTeardown; SLOTS]);
}

/// A handle proving a prefetch was issued for a specific layer into a specific
/// slot generation. Required by `wait`/`release`/`cancel` so a stale reference
/// (slot already reused) is refused instead of silently reading another layer's
/// bytes. Not `Copy`: `release`/`cancel` consume it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerTicket {
    layer_id: u64,
    slot: usize,
    generation: u64,
}

impl LayerTicket {
    /// The layer this ticket was minted for.
    pub fn layer_id(&self) -> u64 {
        self.layer_id
    }
}

/// Falsifiable activity counters. `reuse_wait_ns` is the overlap proof (see the
/// module doc); the `declined_*` counters attribute every refusal to its
/// property-based cause.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PrefillMetrics {
    /// Prefetches that reached `Ready`.
    pub layers_prefetched: u64,
    /// `wait` calls that promoted a slot to `InUse`.
    pub layers_consumed: u64,
    /// `release` calls that recorded a release fence.
    pub layers_released: u64,
    /// Total host wait across every slot reuse — the two-slot overlap metric.
    pub reuse_wait_ns: u64,
    /// Prefetch attempts declined because the pipeline was at its depth limit.
    pub declined_slots_busy: u64,
    /// Prefetch attempts declined because a capture/replay was active.
    pub declined_capture: u64,
    /// Prefetch attempts declined because the layer had zero bytes.
    pub declined_empty: u64,
    /// Operations refused because their ticket's slot had been reused.
    pub stale_rejected: u64,
    /// Prefetches cancelled before consumption.
    pub cancelled: u64,
    /// Slots poisoned by a failed transfer/fence.
    pub poisoned: u64,
}

#[derive(Debug)]
struct Slot {
    status: PrefillSlotStatus,
    /// Monotonic per-slot fill counter; a ticket carries the generation it was
    /// minted under, so a reused slot invalidates older tickets.
    generation: u64,
    /// The layer currently associated with the slot (meaningful unless `Free`).
    layer_id: u64,
    /// Copy fence of the current/last fill, for `compute_wait` and reuse.
    copy_fence: Option<FenceId>,
    /// Release fence recorded when the consumer finished, for reuse `copy_wait`.
    release_fence: Option<FenceId>,
}

impl Default for Slot {
    fn default() -> Self {
        Self {
            status: PrefillSlotStatus::Free,
            generation: 0,
            layer_id: 0,
            copy_fence: None,
            release_fence: None,
        }
    }
}

/// The two-slot whole-layer prefill double buffer.
///
/// Construct one (opt-in, default-off) with [`PrefillDoubleBuffer::new`]; drive
/// it with [`prefetch`](Self::prefetch) → [`wait`](Self::wait) → compute →
/// [`release`](Self::release) per layer, prefetching one layer ahead. `Drop`
/// tears the reserved buffers down, establishing every in-flight transfer's
/// completion first.
/// The two-slot whole-layer prefill double buffer.
///
/// Construct one (opt-in, default-off) with [`PrefillDoubleBuffer::new`]; drive
/// it with [`prefetch`](Self::prefetch) → [`wait`](Self::wait) → compute →
/// [`release`](Self::release) per layer, prefetching one layer ahead. `Drop`
/// tears the reserved buffers down, establishing every in-flight transfer's
/// completion first.
pub struct PrefillDoubleBuffer<T: PrefillTransfer> {
    transfer: T,
    slots: [Slot; SLOTS],
    layer_bytes: u64,
    metrics: PrefillMetrics,
    torn_down: bool,
}

impl<T: PrefillTransfer> fmt::Debug for PrefillDoubleBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefillDoubleBuffer")
            .field("slots", &self.slots)
            .field("layer_bytes", &self.layer_bytes)
            .field("metrics", &self.metrics)
            .field("torn_down", &self.torn_down)
            .finish_non_exhaustive()
    }
}

impl<T: PrefillTransfer> PrefillDoubleBuffer<T> {
    /// Reserve the two stable buffers and build the pipeline, or fail closed.
    ///
    /// `layer_bytes` is the per-layer buffer size the whole pipeline is sized
    /// for (the largest layer's transfer bytes). Reservation is all-or-none:
    /// on any refusal the transfer is dropped and the caller uses the
    /// synchronous single-buffer path.
    ///
    /// # Errors
    /// * [`PrefillReject::EmptyLayer`] if `layer_bytes == 0`.
    /// * [`PrefillReject::CaptureActive`] if a capture/replay is active.
    /// * [`PrefillReject::PoolCapacity`] if the pool cannot retain two buffers,
    ///   or the transfer's own reservation fails ([`PrefillReject::Transfer`]).
    pub fn new(mut transfer: T, layer_bytes: u64) -> Result<Self, PrefillReject<T::Error>> {
        if layer_bytes == 0 {
            return Err(PrefillReject::EmptyLayer);
        }
        if transfer.capture_active() {
            return Err(PrefillReject::CaptureActive);
        }
        if !transfer.can_retain_concurrent(layer_bytes) {
            return Err(PrefillReject::PoolCapacity { layer_bytes });
        }
        transfer
            .reserve(layer_bytes)
            .map_err(PrefillReject::Transfer)?;
        Ok(Self {
            transfer,
            slots: Default::default(),
            layer_bytes,
            metrics: PrefillMetrics::default(),
            torn_down: false,
        })
    }

    /// The per-layer buffer size this pipeline is sized for.
    pub fn layer_bytes(&self) -> u64 {
        self.layer_bytes
    }

    /// Current activity counters.
    pub fn metrics(&self) -> PrefillMetrics {
        self.metrics
    }

    /// Status of slot `idx` (`idx < 2`), for tests/telemetry.
    pub fn slot_status(&self, idx: usize) -> PrefillSlotStatus {
        self.slots[idx].status
    }

    /// Read-only access to the transfer (for stats plumbed through it).
    pub fn transfer(&self) -> &T {
        &self.transfer
    }

    /// Issue an ahead-of-need prefetch of `layer_id` (described by `req`) into
    /// the next reusable slot, returning a ticket the consumer redeems with
    /// [`wait`](Self::wait).
    ///
    /// This is the *only* method that may host-wait (once, on a reused slot's
    /// previous copy fence — the overlap metric); it is deliberately off the
    /// consume path.
    ///
    /// # Errors
    /// * [`PrefillReject::CaptureActive`] / [`PrefillReject::EmptyLayer`] /
    ///   [`PrefillReject::SlotsBusy`] — property-based refusals; the caller
    ///   falls back without side effects.
    /// * [`PrefillReject::Transfer`] — the fill failed; the chosen slot is
    ///   poisoned and quarantined.
    pub fn prefetch(
        &mut self,
        layer_id: u64,
        req: &T::LayerReq,
    ) -> Result<LayerTicket, PrefillReject<T::Error>> {
        debug_assert!(!self.torn_down, "prefetch after teardown");
        if self.transfer.capture_active() {
            self.metrics.declined_capture += 1;
            return Err(PrefillReject::CaptureActive);
        }
        if self.transfer.layer_bytes(req) == 0 {
            self.metrics.declined_empty += 1;
            return Err(PrefillReject::EmptyLayer);
        }
        let Some(slot_idx) = self.pick_claimable_slot() else {
            self.metrics.declined_slots_busy += 1;
            return Err(PrefillReject::SlotsBusy);
        };

        // Reuse fencing plan: only a `Draining` slot carries a previous
        // occupant, and — the ported FreeToken invariant — a slot can only be
        // `Draining` after `release`/`cancel` recorded its release fence, so
        // reuse-before-release is structurally impossible here.
        let reused = self.slots[slot_idx].status == PrefillSlotStatus::Draining;
        let plan = if reused {
            debug_assert!(
                self.slots[slot_idx].release_fence.is_some(),
                "reuse-before-release: a Draining slot must carry a release fence"
            );
            SlotFillPlan {
                prev_copy_fence: self.slots[slot_idx].copy_fence,
                prev_release_fence: self.slots[slot_idx].release_fence,
            }
        } else {
            SlotFillPlan::default()
        };

        let generation = self.slots[slot_idx].generation + 1;
        self.slots[slot_idx].status = PrefillSlotStatus::Filling;
        self.slots[slot_idx].generation = generation;
        self.slots[slot_idx].layer_id = layer_id;

        match self.transfer.fill_slot(slot_idx, req, plan) {
            Ok(outcome) => {
                self.slots[slot_idx].copy_fence = Some(outcome.copy_fence);
                self.slots[slot_idx].release_fence = None;
                self.slots[slot_idx].status = PrefillSlotStatus::Ready;
                self.metrics.layers_prefetched += 1;
                self.metrics.reuse_wait_ns = self
                    .metrics
                    .reuse_wait_ns
                    .saturating_add(outcome.reuse_wait_ns);
                Ok(LayerTicket {
                    layer_id,
                    slot: slot_idx,
                    generation,
                })
            }
            Err(error) => {
                self.poison(slot_idx);
                Err(PrefillReject::Transfer(error))
            }
        }
    }

    /// Order the caller's compute stream after `ticket`'s transfer and take
    /// ownership of the slot for the duration of the compute (`InUse`),
    /// returning the consumer payload. Enqueue-only; no host sync.
    ///
    /// # Errors
    /// [`PrefillReject::StaleGeneration`] if the slot was reused,
    /// [`PrefillReject::Poisoned`], or [`PrefillReject::WrongState`] if the slot
    /// is not `Ready`. A fence failure poisons the slot and returns
    /// [`PrefillReject::Transfer`].
    pub fn wait(&mut self, ticket: &LayerTicket) -> Result<T::Payload, PrefillReject<T::Error>> {
        self.validate_ticket(ticket)?;
        let slot_idx = ticket.slot;
        match self.slots[slot_idx].status {
            PrefillSlotStatus::Ready => {}
            PrefillSlotStatus::Poisoned => {
                return Err(PrefillReject::Poisoned {
                    layer_id: ticket.layer_id,
                });
            }
            status => {
                return Err(PrefillReject::WrongState {
                    layer_id: ticket.layer_id,
                    status,
                });
            }
        }
        let copy_fence = self.slots[slot_idx].copy_fence.unwrap_or(0);
        if let Err(error) = self.transfer.compute_wait(copy_fence) {
            self.poison(slot_idx);
            return Err(PrefillReject::Transfer(error));
        }
        self.slots[slot_idx].status = PrefillSlotStatus::InUse;
        self.metrics.layers_consumed += 1;
        Ok(self.transfer.payload(slot_idx))
    }

    /// Record the consumer's completion fence and mark the slot `Draining` so a
    /// later prefetch may reuse it behind that fence. Consumes the ticket.
    /// Enqueue-only; no host sync.
    ///
    /// # Errors
    /// [`PrefillReject::StaleGeneration`], [`PrefillReject::Poisoned`], or
    /// [`PrefillReject::WrongState`] if the slot is not `InUse`. A fence failure
    /// poisons the slot.
    pub fn release(&mut self, ticket: LayerTicket) -> Result<(), PrefillReject<T::Error>> {
        self.validate_ticket(&ticket)?;
        let slot_idx = ticket.slot;
        match self.slots[slot_idx].status {
            PrefillSlotStatus::InUse => {}
            PrefillSlotStatus::Poisoned => {
                return Err(PrefillReject::Poisoned {
                    layer_id: ticket.layer_id,
                });
            }
            status => {
                return Err(PrefillReject::WrongState {
                    layer_id: ticket.layer_id,
                    status,
                });
            }
        }
        self.record_release(slot_idx)?;
        self.metrics.layers_released += 1;
        Ok(())
    }

    /// Cancel a prefetch that has not been consumed (or a consumed-but-cancelled
    /// one), releasing its slot for reuse. Safe mid-transfer: a release fence is
    /// still recorded so a later refill's `copy_wait` orders correctly against
    /// any enqueued consumer, and the reuse host-wait still drains the cancelled
    /// copy before the staging is refilled. Consumes the ticket.
    ///
    /// # Errors
    /// [`PrefillReject::StaleGeneration`] (already reused → nothing to cancel),
    /// [`PrefillReject::Poisoned`], or [`PrefillReject::WrongState`] if the slot
    /// is already `Draining`/`Free`.
    pub fn cancel(&mut self, ticket: LayerTicket) -> Result<(), PrefillReject<T::Error>> {
        self.validate_ticket(&ticket)?;
        let slot_idx = ticket.slot;
        match self.slots[slot_idx].status {
            PrefillSlotStatus::Ready | PrefillSlotStatus::InUse => {}
            PrefillSlotStatus::Poisoned => {
                return Err(PrefillReject::Poisoned {
                    layer_id: ticket.layer_id,
                });
            }
            status => {
                return Err(PrefillReject::WrongState {
                    layer_id: ticket.layer_id,
                    status,
                });
            }
        }
        self.record_release(slot_idx)?;
        self.metrics.cancelled += 1;
        Ok(())
    }

    /// Pick a `Free` or `Draining` slot, preferring a never-used `Free` slot so
    /// the first two layers spread across both slots before any reuse. Returns
    /// `None` when both slots are busy (Filling/Ready/InUse) or poisoned.
    fn pick_claimable_slot(&self) -> Option<usize> {
        (0..SLOTS)
            .find(|&i| self.slots[i].status == PrefillSlotStatus::Free)
            .or_else(|| (0..SLOTS).find(|&i| self.slots[i].status.is_claimable()))
    }

    fn record_release(&mut self, slot_idx: usize) -> Result<(), PrefillReject<T::Error>> {
        match self.transfer.record_release_fence() {
            Ok(fence) => {
                self.slots[slot_idx].release_fence = Some(fence);
                self.slots[slot_idx].status = PrefillSlotStatus::Draining;
                Ok(())
            }
            Err(error) => {
                self.poison(slot_idx);
                Err(PrefillReject::Transfer(error))
            }
        }
    }

    fn validate_ticket(&mut self, ticket: &LayerTicket) -> Result<(), PrefillReject<T::Error>> {
        if self.slots[ticket.slot].generation != ticket.generation {
            self.metrics.stale_rejected += 1;
            return Err(PrefillReject::StaleGeneration {
                layer_id: ticket.layer_id,
            });
        }
        Ok(())
    }

    fn poison(&mut self, slot_idx: usize) {
        self.transfer.quarantine_slot(slot_idx);
        self.slots[slot_idx].status = PrefillSlotStatus::Poisoned;
        self.metrics.poisoned += 1;
    }
}

impl<T: PrefillTransfer> Drop for PrefillDoubleBuffer<T> {
    fn drop(&mut self) {
        if self.torn_down {
            return;
        }
        self.torn_down = true;
        let descriptors = std::array::from_fn(|i| {
            let slot = &self.slots[i];
            // Every reserved buffer must be freed except a quarantined
            // (poisoned) one, which is retained out of caution. Free slots still
            // own their up-front reserved buffer, so they are live too.
            let live = !matches!(slot.status, PrefillSlotStatus::Poisoned);
            SlotTeardown {
                last_copy_fence: if live { slot.copy_fence } else { None },
                last_release_fence: if live { slot.release_fence } else { None },
                live,
            }
        });
        self.transfer.teardown(descriptors);
    }
}

/// A [`Duration`] rendered to whole nanoseconds saturated into `u64`, matching
/// how the single-slot path records `prefetch_promote_wait_ns`.
pub fn duration_ns(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

// ===========================================================================
// CUDA-backed realization of `PrefillTransfer`.
//
// This is the real, device-touching implementation the generic primitive above
// is driven over. It owns the two stable per-slot buffers (a device allocation
// plus a pinned host staging buffer each), reserved once and reused across
// layers, and it performs the H2D copy and the four fence primitives. It never
// inserts a page into the residency cache and never maps or accounts device
// memory through a second authority: the pinned pool's `can_retain_concurrent`
// gate and the runtime's own allocator remain the sole authorities, exactly as
// the single-slot `prefetch_block_quantized_moe` path uses them.
// ===========================================================================

/// Error from the CUDA-backed prefill transfer. Wraps the underlying driver /
/// runtime error text; the primitive turns it into [`PrefillReject::Transfer`]
/// after poisoning + quarantining the affected slot.
#[derive(Debug)]
pub struct CudaPrefillError(String);

impl CudaPrefillError {
    fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl fmt::Display for CudaPrefillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CudaPrefillError {}

/// A whole layer's transfer request: the layer's device-bound regions expressed
/// as a single [`LazyWeight`] whose `regions` are filled sequentially into the
/// slot's staging buffer, read through the transfer's shared
/// [`MmapRegionSource`]. Representing a "whole layer" as one region list keeps
/// this primitive a pure *staging* concern — it copies bytes; it does not learn
/// per-expert keys or admit anything into the residency cache.
pub type PrefillLayerRequest = LazyWeight;

/// Consumer handle to a filled slot's device buffer. The pointer addresses the
/// slot's *stable* device allocation (reused across layers); it is valid to a
/// consumer only after that consumer has ordered its compute stream after the
/// slot's copy fence via [`PrefillDoubleBuffer::wait`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillLayerView {
    /// Device pointer to the slot's stable buffer holding the layer's bytes.
    pub device_ptr: CUdeviceptr,
    /// Number of valid bytes at `device_ptr` (this layer's byte count, which may
    /// be smaller than the buffer the pipeline was sized for).
    pub len: usize,
}

/// One stable slot's physical buffers, reserved once and reused across layers.
struct CudaSlotBuffers {
    /// Stable device allocation (from `alloc_raw`); `0` once freed/quarantined.
    device_ptr: CUdeviceptr,
    /// Byte capacity of `device_ptr` (the pipeline's per-layer size).
    capacity: usize,
    /// Valid bytes written by the most recent fill (this layer's byte count).
    valid_len: usize,
    /// Stable pinned host staging buffer (the H2D copy source).
    staging: Option<PinnedStaging>,
    /// A copy-stream fence recorded *after* the slot's last H2D copy, kept
    /// un-consumed specifically so a later reuse can host-wait it (the staging
    /// write-after-read drain). Distinct from the fence a consumer's
    /// `compute_wait` consumes, so the consume path never blocks the host. `0`
    /// when the slot has no prior copy to drain.
    reuse_fence: FenceId,
}

impl CudaSlotBuffers {
    fn empty() -> Self {
        Self {
            device_ptr: 0,
            capacity: 0,
            valid_len: 0,
            staging: None,
            reuse_fence: 0,
        }
    }
}

/// CUDA realization of [`PrefillTransfer`] over a [`CudaRuntime`] and the
/// residency's shared [`PinnedStagingPool`].
///
/// Not `Sync` (its slot buffers use interior mutability): a pipeline instance is
/// owned by one request/device, matching the isolation the CPU tests assert.
pub struct CudaPrefillTransfer {
    runtime: Arc<CudaRuntime>,
    staging_pool: Arc<PinnedStagingPool>,
    source: Arc<dyn MmapRegionSource>,
    slots: [RefCell<CudaSlotBuffers>; SLOTS],
    /// Buffers moved out of a poisoned slot, retained for process lifetime (a
    /// DMA may still be reading them). Never freed on `Drop` — the same
    /// quarantine rule the single-slot path applies to an unprovable-completion
    /// transfer (`quarantine_in_flight_fill`).
    quarantined: RefCell<Vec<CudaSlotBuffers>>,
}

impl CudaPrefillTransfer {
    /// Build a transfer over `runtime`'s streams and the residency's shared
    /// `staging_pool`, reading layer bytes through `source`. Allocates nothing
    /// until [`PrefillTransfer::reserve`] runs.
    pub fn new(
        runtime: Arc<CudaRuntime>,
        staging_pool: Arc<PinnedStagingPool>,
        source: Arc<dyn MmapRegionSource>,
    ) -> Self {
        Self {
            runtime,
            staging_pool,
            source,
            slots: std::array::from_fn(|_| RefCell::new(CudaSlotBuffers::empty())),
            quarantined: RefCell::new(Vec::new()),
        }
    }

    /// Number of buffer sets currently quarantined (never freed). For tests.
    pub fn quarantined_len(&self) -> usize {
        self.quarantined.borrow().len()
    }
}

impl PrefillTransfer for CudaPrefillTransfer {
    type Payload = PrefillLayerView;
    type LayerReq = PrefillLayerRequest;
    type Error = CudaPrefillError;

    fn layer_bytes(&self, req: &Self::LayerReq) -> u64 {
        req.region_bytes_len() as u64
    }

    fn capture_active(&self) -> bool {
        // Fail closed: if capture state cannot be read, treat it as active so a
        // reservation (a VMM/allocator mutation) is declined rather than risked.
        self.runtime.is_capturing().unwrap_or(true)
    }

    fn can_retain_concurrent(&self, layer_bytes: u64) -> bool {
        self.staging_pool
            .can_retain_concurrent(layer_bytes as usize, SLOTS)
    }

    fn reserve(&mut self, layer_bytes: u64) -> Result<(), Self::Error> {
        let len = layer_bytes as usize;
        // All-or-none: acquire both pinned staging buffers and both device
        // allocations, unwinding anything acquired if a later step fails so a
        // partial reservation never leaks.
        let mut staging_bufs: Vec<PinnedStaging> = Vec::with_capacity(SLOTS);
        let mut device_ptrs: Vec<CUdeviceptr> = Vec::with_capacity(SLOTS);
        let mut acquire = || -> Result<(), Self::Error> {
            for _ in 0..SLOTS {
                let pooled = self.staging_pool.acquire(len).map_err(|error| {
                    CudaPrefillError::new(format!("pinned staging acquire: {error}"))
                })?;
                staging_bufs.push(pooled.into_inner());
                let ptr = self
                    .runtime
                    .alloc_raw(len)
                    .map_err(|error| CudaPrefillError::new(format!("device alloc: {error}")))?;
                device_ptrs.push(ptr);
            }
            Ok(())
        };
        if let Err(error) = acquire() {
            for ptr in device_ptrs {
                // SAFETY: each `ptr` is a fresh allocation owned here and freed
                // exactly once; nothing has been enqueued against it yet.
                let _ = unsafe { self.runtime.free_raw(ptr) };
            }
            // `staging_bufs` drop here, freeing their pinned host pages.
            return Err(error);
        }
        for (i, (ptr, staging)) in device_ptrs.into_iter().zip(staging_bufs).enumerate() {
            let mut slot = self.slots[i].borrow_mut();
            slot.device_ptr = ptr;
            slot.capacity = len;
            slot.valid_len = 0;
            slot.staging = Some(staging);
            slot.reuse_fence = 0;
        }
        Ok(())
    }

    fn fill_slot(
        &self,
        slot: usize,
        req: &Self::LayerReq,
        plan: SlotFillPlan,
    ) -> Result<FillOutcome, Self::Error> {
        let mut s = self.slots[slot].borrow_mut();
        let mut reuse_wait_ns = 0u64;
        // Staging write-after-read drain on reuse: host-wait the slot's previous
        // copy fence (kept un-consumed for exactly this) before overwriting the
        // pinned buffer. This is the only host wait; its duration is the overlap
        // metric. `plan.prev_copy_fence.is_some()` is the reuse signal; the
        // actual event awaited is the slot's private `reuse_fence`.
        if plan.prev_copy_fence.is_some() {
            let fence = std::mem::take(&mut s.reuse_fence);
            let start = Instant::now();
            if let Err(error) = self.runtime.resolve_prefetch_fence(fence) {
                let (detail, completion) = error.into_parts();
                if matches!(completion, FailedHtodCompletion::MayBeInFlight) {
                    return Err(CudaPrefillError::new(format!(
                        "reuse drain could not establish prior copy completion: {detail}"
                    )));
                }
                // `Completed`: a fallback copy-stream sync proved the read ended,
                // so it is safe to overwrite the staging buffer.
            }
            reuse_wait_ns = duration_ns(start.elapsed());
        }

        let src_len = req.region_bytes_len();
        if src_len == 0 {
            return Err(CudaPrefillError::new("fill_slot on a zero-byte layer"));
        }
        if src_len > s.capacity {
            return Err(CudaPrefillError::new(format!(
                "layer is {src_len} bytes but the pipeline reserved {}-byte buffers",
                s.capacity
            )));
        }

        // CPU-fill the stable pinned staging buffer from the layer's mmap
        // regions (concatenated in order).
        {
            let staging = s
                .staging
                .as_mut()
                .ok_or_else(|| CudaPrefillError::new("fill_slot on an unreserved slot"))?;
            fill_staging_from_regions(req, &*self.source, staging)
                .map_err(|error| CudaPrefillError::new(format!("staging fill: {error}")))?;
        }

        // Device write-after-read: order the copy stream after the previous
        // consumer's release fence before overwriting this slot's device buffer
        // (enqueue-only, no host sync).
        if let Some(rel) = plan.prev_release_fence {
            self.runtime
                .copy_wait_fence(rel)
                .map_err(|error| CudaPrefillError::new(format!("copy_wait release: {error}")))?;
        }

        let ptr = s.device_ptr;
        // SAFETY: `ptr` is this slot's stable device allocation of `capacity`
        // bytes (>= `src_len`); the staging source holds `src_len` valid bytes
        // and stays alive (owned by this slot) until a later reuse host-waits
        // this copy's `reuse_fence`.
        let enqueue = {
            let staging = s.staging.as_ref().expect("staging present after fill");
            unsafe { self.runtime.htod_async(&staging.as_slice()[..src_len], ptr) }
        };
        if let Err(error) = enqueue {
            return Err(CudaPrefillError::new(format!("H2D enqueue: {error}")));
        }
        s.valid_len = src_len;

        // The compute-ordering fence a consumer's `wait` consumes (enqueue-only).
        let copy_fence = self
            .runtime
            .record_copy_fence()
            .map_err(|error| CudaPrefillError::new(format!("record copy fence: {error}")))?;
        // A second copy-stream fence recorded after the same copy, kept for the
        // next reuse's host wait so the consume path never blocks the host on
        // this copy.
        let reuse_fence = self
            .runtime
            .record_copy_fence()
            .map_err(|error| CudaPrefillError::new(format!("record reuse fence: {error}")))?;
        s.reuse_fence = reuse_fence;

        Ok(FillOutcome {
            copy_fence,
            reuse_wait_ns,
        })
    }

    fn payload(&self, slot: usize) -> Self::Payload {
        let s = self.slots[slot].borrow();
        PrefillLayerView {
            device_ptr: s.device_ptr,
            len: s.valid_len,
        }
    }

    fn compute_wait(&self, copy_fence: FenceId) -> Result<(), Self::Error> {
        self.runtime
            .compute_wait_fence(copy_fence)
            .map_err(|error| CudaPrefillError::new(format!("compute_wait: {error}")))
    }

    fn record_release_fence(&self) -> Result<FenceId, Self::Error> {
        self.runtime
            .record_compute_fence()
            .map_err(|error| CudaPrefillError::new(format!("record release fence: {error}")))
    }

    fn quarantine_slot(&self, slot: usize) {
        // Retain the slot's buffers for process lifetime rather than free memory
        // a failed/aborted transfer may still be reading.
        let taken = std::mem::replace(
            &mut *self.slots[slot].borrow_mut(),
            CudaSlotBuffers::empty(),
        );
        self.quarantined.borrow_mut().push(taken);
    }

    fn teardown(&mut self, slots: [SlotTeardown; SLOTS]) {
        // Off the hot path: drain the compute stream so no consumer is still
        // reading a slot's device buffer when it is freed (covers a slot left
        // InUse at teardown).
        let _ = self.runtime.drain_for_unmap();
        for (i, descriptor) in slots.iter().enumerate() {
            if !descriptor.live {
                // Poisoned slot: its buffers were already moved to `quarantined`
                // at poison time; the slot is empty.
                continue;
            }
            let mut s = self.slots[i].borrow_mut();
            // Host-establish the slot's last H2D copy completion, minting the
            // witness the pool's release requires. Also drain any lingering
            // compute-ordering / release fences to release their events.
            let reuse_fence = std::mem::take(&mut s.reuse_fence);
            let witness = match self.runtime.resolve_prefetch_fence(reuse_fence) {
                Ok(completed) => Some(completed),
                Err(error) => {
                    let (_detail, completion) = error.into_parts();
                    if matches!(completion, FailedHtodCompletion::MayBeInFlight) {
                        // Cannot prove the copy read of staging ended: quarantine
                        // both instead of freeing memory a DMA may still touch.
                        let taken = std::mem::replace(&mut *s, CudaSlotBuffers::empty());
                        drop(s);
                        self.quarantined.borrow_mut().push(taken);
                        continue;
                    }
                    None
                }
            };
            let _ = self
                .runtime
                .resolve_prefetch_fence(descriptor.last_copy_fence.unwrap_or(0));
            let _ = self
                .runtime
                .resolve_prefetch_fence(descriptor.last_release_fence.unwrap_or(0));

            let ptr = std::mem::replace(&mut s.device_ptr, 0);
            if ptr != 0 {
                // SAFETY: `ptr` is this slot's stable allocation, owned
                // exclusively and freed exactly once; the copy and compute
                // streams were drained above so no in-flight op references it.
                let _ = unsafe { self.runtime.free_raw(ptr) };
            }
            // Return the staging buffer to the shared pool for reuse only when
            // `witness` proves its last copy read completed (issue #837 fix);
            // otherwise (fallback-sync path minted no witness) let the buffer
            // drop, freeing its pinned pages rather than reuse without proof.
            if let (Some(staging), Some(witness)) = (s.staging.take(), witness) {
                self.staging_pool.release(staging, witness);
            }
        }
    }
}

impl Drop for CudaPrefillTransfer {
    fn drop(&mut self) {
        // Quarantined buffers are retained for process lifetime (a DMA may still
        // be reading them); never run their destructors.
        for taken in self.quarantined.get_mut().drain(..) {
            std::mem::forget(taken);
        }
        // Defensive: after `teardown` every live slot is empty. Anything left
        // here means teardown never ran (e.g. a failed reservation that already
        // cleaned itself up, so this is normally empty). Free it; buffers are
        // only present if no in-flight op referenced them.
        for slot in &self.slots {
            let mut s = slot.borrow_mut();
            let ptr = std::mem::replace(&mut s.device_ptr, 0);
            if ptr != 0 {
                // SAFETY: leftover stable allocation owned here, freed once.
                let _ = unsafe { self.runtime.free_raw(ptr) };
            }
            // `s.staging` (if any) drops here, freeing its pinned pages.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// A recorded transfer operation, so tests can assert exact ordering and
    /// fence identities without a GPU.
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Op {
        Reserve {
            bytes: u64,
        },
        ReuseWait {
            slot: usize,
            fence: FenceId,
        },
        CopyWait {
            slot: usize,
            fence: FenceId,
        },
        Fill {
            slot: usize,
            layer: u64,
            copy_fence: FenceId,
        },
        ComputeWait {
            fence: FenceId,
        },
        ReleaseFence {
            fence: FenceId,
        },
        Quarantine {
            slot: usize,
        },
        Teardown {
            live: [bool; SLOTS],
            fences: [Option<FenceId>; SLOTS],
        },
    }

    #[derive(Clone, Debug)]
    struct FakeLayer {
        id: u64,
        bytes: u64,
    }

    /// Deterministic in-memory transfer. Mints monotonically increasing fence
    /// ids, records every op, models the two stable buffers as byte cells, and
    /// tracks buffer liveness for leak/accounting assertions.
    #[derive(Default)]
    struct FakeState {
        ops: Vec<Op>,
        next_fence: FenceId,
        /// Filled layer id per slot (what a consumer would read).
        slot_layer: [Option<u64>; SLOTS],
        /// Whether each stable buffer is currently reserved (live).
        buffer_live: [bool; SLOTS],
        reserved: bool,
        capture: bool,
        pool_ok: bool,
        /// If set, `reserve` fails.
        reserve_fails: bool,
        /// Layer ids for which `fill_slot` must fail (transfer failure).
        fail_fill_layers: Vec<u64>,
        /// If set, the next `compute_wait` fails.
        fail_compute_wait: bool,
        /// If set, the next `record_release_fence` fails.
        fail_release_fence: bool,
        /// Simulated remaining transfer time per slot's last copy fence, so the
        /// reuse host-wait can report a non-zero "unhidden" duration on demand.
        reuse_wait_ns: u64,
    }

    impl FakeState {
        fn fence(&mut self) -> FenceId {
            self.next_fence += 1;
            self.next_fence
        }
    }

    #[derive(Clone)]
    struct FakeTransfer {
        state: Rc<RefCell<FakeState>>,
    }

    #[derive(Debug)]
    struct FakeError(String);
    impl fmt::Display for FakeError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl FakeTransfer {
        fn new() -> Self {
            Self {
                state: Rc::new(RefCell::new(FakeState {
                    pool_ok: true,
                    ..Default::default()
                })),
            }
        }
        fn ops(&self) -> Vec<Op> {
            self.state.borrow().ops.clone()
        }
    }

    impl PrefillTransfer for FakeTransfer {
        type Payload = u64;
        type LayerReq = FakeLayer;
        type Error = FakeError;

        fn layer_bytes(&self, req: &Self::LayerReq) -> u64 {
            req.bytes
        }
        fn capture_active(&self) -> bool {
            self.state.borrow().capture
        }
        fn can_retain_concurrent(&self, _layer_bytes: u64) -> bool {
            self.state.borrow().pool_ok
        }
        fn reserve(&mut self, layer_bytes: u64) -> Result<(), Self::Error> {
            let mut s = self.state.borrow_mut();
            if s.reserve_fails {
                return Err(FakeError("reserve failed".into()));
            }
            s.ops.push(Op::Reserve { bytes: layer_bytes });
            s.reserved = true;
            s.buffer_live = [true, true];
            Ok(())
        }
        fn fill_slot(
            &self,
            slot: usize,
            req: &Self::LayerReq,
            plan: SlotFillPlan,
        ) -> Result<FillOutcome, Self::Error> {
            let mut s = self.state.borrow_mut();
            let mut reuse_wait_ns = 0;
            // Staging WAR: host-wait the slot's previous copy fence before
            // refilling. Recorded so tests can assert it precedes the fill.
            if let Some(prev) = plan.prev_copy_fence {
                reuse_wait_ns = s.reuse_wait_ns;
                s.ops.push(Op::ReuseWait { slot, fence: prev });
            }
            // Device WAR: order the copy stream after the previous consumer.
            if let Some(rel) = plan.prev_release_fence {
                s.ops.push(Op::CopyWait { slot, fence: rel });
            }
            if s.fail_fill_layers.contains(&req.id) {
                return Err(FakeError(format!("fill failed for layer {}", req.id)));
            }
            let copy_fence = s.fence();
            s.ops.push(Op::Fill {
                slot,
                layer: req.id,
                copy_fence,
            });
            s.slot_layer[slot] = Some(req.id);
            Ok(FillOutcome {
                copy_fence,
                reuse_wait_ns,
            })
        }
        fn payload(&self, slot: usize) -> Self::Payload {
            self.state.borrow().slot_layer[slot].expect("payload of a filled slot")
        }
        fn compute_wait(&self, copy_fence: FenceId) -> Result<(), Self::Error> {
            let mut s = self.state.borrow_mut();
            if s.fail_compute_wait {
                return Err(FakeError("compute_wait failed".into()));
            }
            s.ops.push(Op::ComputeWait { fence: copy_fence });
            Ok(())
        }
        fn record_release_fence(&self) -> Result<FenceId, Self::Error> {
            let mut s = self.state.borrow_mut();
            if s.fail_release_fence {
                return Err(FakeError("release fence failed".into()));
            }
            let fence = s.fence();
            s.ops.push(Op::ReleaseFence { fence });
            Ok(fence)
        }
        fn quarantine_slot(&self, slot: usize) {
            let mut s = self.state.borrow_mut();
            s.ops.push(Op::Quarantine { slot });
            // A quarantined buffer is retained (never freed), but it is no
            // longer a *reusable* live buffer for teardown accounting.
            s.buffer_live[slot] = false;
        }
        fn teardown(&mut self, slots: [SlotTeardown; SLOTS]) {
            let mut s = self.state.borrow_mut();
            let live = [slots[0].live, slots[1].live];
            let fences = [slots[0].last_copy_fence, slots[1].last_copy_fence];
            s.ops.push(Op::Teardown { live, fences });
            for (i, d) in slots.iter().enumerate() {
                if d.live {
                    // A real teardown would first establish `last_copy_fence` /
                    // `last_release_fence` completion; the fake just frees.
                    let _ = (d.last_copy_fence, d.last_release_fence);
                    s.buffer_live[i] = false;
                }
            }
        }
    }

    fn layer(id: u64) -> FakeLayer {
        FakeLayer { id, bytes: 1 << 20 }
    }

    /// Drive one full layer through the pipeline: prefetch → wait → release.
    fn cycle(db: &mut PrefillDoubleBuffer<FakeTransfer>, id: u64) {
        let ticket = db.prefetch(id, &layer(id)).expect("prefetch");
        let payload = db.wait(&ticket).expect("wait");
        assert_eq!(payload, id, "consumer read the layer it prefetched");
        db.release(ticket).expect("release");
    }

    fn assert_no_leak(fake: &FakeTransfer) {
        let s = fake.state.borrow();
        assert_eq!(
            s.buffer_live,
            [false, false],
            "every reserved buffer must be released or quarantined exactly once"
        );
    }

    #[test]
    fn reserves_two_buffers_then_runs_n_and_n_plus_one_in_order() {
        let fake = FakeTransfer::new();
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");

        // Canonical order: prefetch(0), then per layer prefetch(N+1); wait(N);
        // release(N). Prefetch ahead so N+1's transfer overlaps N's compute.
        let t0 = db.prefetch(0, &layer(0)).unwrap();
        let t1 = db.prefetch(1, &layer(1)).unwrap();
        // Two distinct slots hold the two layers concurrently.
        assert_ne!(t0.slot, t1.slot, "N and N+1 occupy different slots");
        assert_eq!(db.wait(&t0).unwrap(), 0);
        db.release(t0).unwrap();
        assert_eq!(db.wait(&t1).unwrap(), 1);
        db.release(t1).unwrap();

        let ops = fake.ops();
        // Fill(0) is enqueued and its copy fence is what wait(0) orders after.
        let fill0 = ops.iter().find_map(|op| match op {
            Op::Fill {
                layer: 0,
                copy_fence,
                ..
            } => Some(*copy_fence),
            _ => None,
        });
        assert!(
            ops.contains(&Op::ComputeWait {
                fence: fill0.unwrap()
            }),
            "wait(0) must order compute after fill(0)'s copy fence"
        );
        drop(db);
        assert_no_leak(&fake);
    }

    #[test]
    fn wraparound_reuses_both_slots_with_two_directional_fencing() {
        let fake = FakeTransfer::new();
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");
        // Six layers over two slots forces three reuses per slot.
        for id in 0..6 {
            cycle(&mut db, id);
        }
        let ops = fake.ops();
        // A reuse must host-wait the previous copy fence AND copy_wait the
        // previous release fence, both before the new fill on that slot.
        let reuse_waits = ops
            .iter()
            .filter(|op| matches!(op, Op::ReuseWait { .. }))
            .count();
        let copy_waits = ops
            .iter()
            .filter(|op| matches!(op, Op::CopyWait { .. }))
            .count();
        assert_eq!(
            reuse_waits, 4,
            "layers 2..6 each reuse a slot (staging WAR host-wait)"
        );
        assert_eq!(
            copy_waits, 4,
            "layers 2..6 each reuse a slot (device WAR copy_wait)"
        );

        // For the first reuse (layer 2 reuses slot 0), the ReuseWait/CopyWait
        // must appear immediately before its Fill, and reference slot 0's own
        // earlier fences.
        let idx_fill2 = ops
            .iter()
            .position(|op| matches!(op, Op::Fill { layer: 2, .. }))
            .unwrap();
        assert!(matches!(ops[idx_fill2 - 1], Op::CopyWait { slot: 0, .. }));
        assert!(matches!(ops[idx_fill2 - 2], Op::ReuseWait { slot: 0, .. }));

        assert_eq!(db.metrics().layers_prefetched, 6);
        assert_eq!(db.metrics().layers_released, 6);
        drop(db);
        assert_no_leak(&fake);
    }

    #[test]
    fn single_layer_prefill_never_reuses() {
        let fake = FakeTransfer::new();
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");
        cycle(&mut db, 0);
        let ops = fake.ops();
        assert!(
            !ops.iter()
                .any(|op| matches!(op, Op::ReuseWait { .. } | Op::CopyWait { .. })),
            "a single layer performs no reuse fencing"
        );
        assert_eq!(db.metrics().reuse_wait_ns, 0);
        drop(db);
        assert_no_leak(&fake);
    }

    #[test]
    fn final_layer_release_then_teardown_drains_in_flight() {
        let fake = FakeTransfer::new();
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");
        // Leave the final layer prefetched-and-consumed but with the other slot
        // still holding an unconsumed prefetch at teardown (in-flight).
        let t0 = db.prefetch(0, &layer(0)).unwrap();
        let t1 = db.prefetch(1, &layer(1)).unwrap();
        let _ = db.wait(&t0).unwrap();
        db.release(t0).unwrap();
        // t1 is Ready (in-flight) — never consumed.
        assert_eq!(db.slot_status(t1.slot), PrefillSlotStatus::Ready);
        drop(db);
        // Teardown must have established the in-flight slot's copy fence.
        let ops = fake.ops();
        let teardown = ops.iter().find_map(|op| match op {
            Op::Teardown { live, fences } => Some((*live, *fences)),
            _ => None,
        });
        let (live, fences) = teardown.expect("teardown ran");
        assert!(live[t1.slot], "the in-flight slot is live at teardown");
        assert!(
            fences[t1.slot].is_some(),
            "teardown drains the in-flight copy fence"
        );
        assert_no_leak(&fake);
    }

    #[test]
    fn cancellation_midflight_frees_slot_for_reuse() {
        let fake = FakeTransfer::new();
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");
        let t0 = db.prefetch(0, &layer(0)).unwrap();
        // Cancel before consuming (mid-transfer).
        db.cancel(t0.clone()).unwrap();
        assert_eq!(db.metrics().cancelled, 1);
        assert_eq!(db.slot_status(t0.slot), PrefillSlotStatus::Draining);
        // Occupy the other (still Free) slot so the cancelled slot becomes the
        // only reusable one, forcing its drain-before-refill path.
        let t_other = db.prefetch(1, &layer(1)).unwrap();
        assert_ne!(t_other.slot, t0.slot, "the fresh Free slot is taken first");
        // The next prefetch must reuse the cancelled slot, and it must drain the
        // cancelled copy first (ReuseWait) — not corrupt it.
        let t_new = db.prefetch(2, &layer(2)).unwrap();
        assert_eq!(t_new.slot, t0.slot, "cancelled slot is reused");
        assert_eq!(db.wait(&t_new).unwrap(), 2);
        db.release(t_new).unwrap();
        let _ = db.wait(&t_other).unwrap();
        db.release(t_other).unwrap();
        let ops = fake.ops();
        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::ReuseWait { slot, .. } if *slot == t0.slot)),
            "reusing a cancelled slot drains its in-flight copy before refill"
        );
        drop(db);
        assert_no_leak(&fake);
    }

    #[test]
    fn stale_ticket_after_reuse_is_refused_not_served() {
        let fake = FakeTransfer::new();
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");
        let stale = db.prefetch(0, &layer(0)).unwrap();
        let _ = db.wait(&stale).unwrap();
        db.release(stale.clone()).unwrap();
        // Fill the other slot, then reuse `stale`'s slot for a new layer.
        let other = db.prefetch(1, &layer(1)).unwrap();
        let reuse = db.prefetch(2, &layer(2)).unwrap();
        assert_eq!(reuse.slot, stale.slot, "layer 2 reused layer 0's slot");
        // The stale ticket's generation no longer matches — refuse, don't serve
        // layer 2's bytes as if they were layer 0's.
        match db.wait(&stale) {
            Err(PrefillReject::StaleGeneration { layer_id: 0 }) => {}
            other => panic!("expected StaleGeneration, got {other:?}"),
        }
        assert_eq!(db.metrics().stale_rejected, 1);
        // The live tickets still work.
        let _ = db.wait(&other).unwrap();
        db.release(other).unwrap();
        let _ = db.wait(&reuse).unwrap();
        db.release(reuse).unwrap();
        drop(db);
        assert_no_leak(&fake);
    }

    #[test]
    fn transfer_failure_poisons_slot_and_quarantines_without_leak() {
        let fake = FakeTransfer::new();
        fake.state.borrow_mut().fail_fill_layers = vec![7];
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");
        match db.prefetch(7, &layer(7)) {
            Err(PrefillReject::Transfer(_)) => {}
            other => panic!("expected Transfer error, got {other:?}"),
        }
        assert_eq!(db.metrics().poisoned, 1);
        // The failed slot is poisoned and quarantined; a wait on its ticket
        // cannot even be minted (prefetch returned Err). The other slot still
        // works, and the poisoned slot is never reused.
        let ok = db.prefetch(1, &layer(1)).unwrap();
        assert_ne!(ok.slot, 0, "poisoned slot 0 is not reused");
        let _ = db.wait(&ok).unwrap();
        db.release(ok).unwrap();
        let ops = fake.ops();
        assert!(ops.contains(&Op::Quarantine { slot: 0 }));
        drop(db);
        assert_no_leak(&fake);
    }

    #[test]
    fn compute_wait_failure_poisons_the_ready_slot() {
        let fake = FakeTransfer::new();
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");
        let t = db.prefetch(0, &layer(0)).unwrap();
        fake.state.borrow_mut().fail_compute_wait = true;
        match db.wait(&t) {
            Err(PrefillReject::Transfer(_)) => {}
            other => panic!("expected Transfer error, got {other:?}"),
        }
        assert_eq!(db.slot_status(t.slot), PrefillSlotStatus::Poisoned);
        assert_eq!(db.metrics().poisoned, 1);
        drop(db);
        assert_no_leak(&fake);
    }

    #[test]
    fn pool_capacity_decline_is_typed_and_makes_no_reservation() {
        let fake = FakeTransfer::new();
        fake.state.borrow_mut().pool_ok = false;
        match PrefillDoubleBuffer::new(fake.clone(), 1 << 20) {
            Err(PrefillReject::PoolCapacity { layer_bytes }) => assert_eq!(layer_bytes, 1 << 20),
            other => panic!("expected PoolCapacity, got {other:?}"),
        }
        // OOM/insufficient-capacity is all-or-none: nothing was reserved.
        assert!(!fake.state.borrow().reserved);
        assert_eq!(fake.ops(), Vec::new());
    }

    #[test]
    fn reserve_transfer_failure_is_all_or_none() {
        let fake = FakeTransfer::new();
        fake.state.borrow_mut().reserve_fails = true;
        match PrefillDoubleBuffer::new(fake.clone(), 1 << 20) {
            Err(PrefillReject::Transfer(_)) => {}
            other => panic!("expected Transfer error, got {other:?}"),
        }
        assert!(!fake.state.borrow().reserved);
    }

    #[test]
    fn capture_active_rejects_reservation_and_prefetch() {
        let fake = FakeTransfer::new();
        fake.state.borrow_mut().capture = true;
        // Reservation is a VMM/PMM commit — refused during capture.
        match PrefillDoubleBuffer::new(fake.clone(), 1 << 20) {
            Err(PrefillReject::CaptureActive) => {}
            other => panic!("expected CaptureActive at reservation, got {other:?}"),
        }
        // And once running, a capture that begins mid-stream refuses new
        // prefetch (no allocator mutation under capture).
        fake.state.borrow_mut().capture = false;
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");
        fake.state.borrow_mut().capture = true;
        match db.prefetch(0, &layer(0)) {
            Err(PrefillReject::CaptureActive) => {}
            other => panic!("expected CaptureActive at prefetch, got {other:?}"),
        }
        assert_eq!(db.metrics().declined_capture, 1);
        drop(db);
        assert_no_leak(&fake);
    }

    #[test]
    fn empty_layer_is_refused() {
        let fake = FakeTransfer::new();
        match PrefillDoubleBuffer::new(fake.clone(), 0) {
            Err(PrefillReject::EmptyLayer) => {}
            other => panic!("expected EmptyLayer at reservation, got {other:?}"),
        }
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");
        match db.prefetch(0, &FakeLayer { id: 0, bytes: 0 }) {
            Err(PrefillReject::EmptyLayer) => {}
            other => panic!("expected EmptyLayer at prefetch, got {other:?}"),
        }
        assert_eq!(db.metrics().declined_empty, 1);
    }

    #[test]
    fn depth_limit_refuses_a_third_unreleased_prefetch() {
        let fake = FakeTransfer::new();
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");
        let _t0 = db.prefetch(0, &layer(0)).unwrap();
        let _t1 = db.prefetch(1, &layer(1)).unwrap();
        // Both slots Ready and unreleased: a third prefetch has no reusable slot
        // and must refuse rather than reuse-before-release.
        match db.prefetch(2, &layer(2)) {
            Err(PrefillReject::SlotsBusy) => {}
            other => panic!("expected SlotsBusy, got {other:?}"),
        }
        assert_eq!(db.metrics().declined_slots_busy, 1);
    }

    #[test]
    fn instances_are_isolated() {
        // Two independent double buffers (e.g. two requests / two devices) do
        // not share slot state or fence identities.
        let fake_a = FakeTransfer::new();
        let fake_b = FakeTransfer::new();
        let mut a = PrefillDoubleBuffer::new(fake_a.clone(), 1 << 20).expect("reserve a");
        let mut b = PrefillDoubleBuffer::new(fake_b.clone(), 1 << 20).expect("reserve b");
        let ta = a.prefetch(10, &layer(10)).unwrap();
        let tb = b.prefetch(20, &layer(20)).unwrap();
        assert_eq!(a.wait(&ta).unwrap(), 10);
        assert_eq!(b.wait(&tb).unwrap(), 20);
        // A ticket from `a` has no meaning in `b`; b's slot generation differs
        // only coincidentally, but the payloads prove isolation.
        a.release(ta).unwrap();
        b.release(tb).unwrap();
        drop(a);
        drop(b);
        assert_no_leak(&fake_a);
        assert_no_leak(&fake_b);
    }

    #[test]
    fn release_fence_failure_poisons_slot() {
        let fake = FakeTransfer::new();
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");
        let t = db.prefetch(0, &layer(0)).unwrap();
        let _ = db.wait(&t).unwrap();
        fake.state.borrow_mut().fail_release_fence = true;
        match db.release(t) {
            Err(PrefillReject::Transfer(_)) => {}
            other => panic!("expected Transfer error, got {other:?}"),
        }
        assert_eq!(db.metrics().poisoned, 1);
        drop(db);
        assert_no_leak(&fake);
    }

    #[test]
    fn reuse_wait_ns_is_reported_when_transfer_not_hidden() {
        let fake = FakeTransfer::new();
        fake.state.borrow_mut().reuse_wait_ns = 4_242;
        let mut db = PrefillDoubleBuffer::new(fake.clone(), 1 << 20).expect("reserve");
        cycle(&mut db, 0);
        cycle(&mut db, 1);
        // Layers 0 and 1 take the two fresh Free slots; layer 2 reuses slot 0
        // (the first freed slot) — its unhidden reuse wait is surfaced as the
        // overlap metric.
        cycle(&mut db, 2);
        assert!(
            db.metrics().reuse_wait_ns >= 4_242,
            "unhidden reuse wait is surfaced"
        );
        drop(db);
        assert_no_leak(&fake);
    }
}
