//! Shared, inert expert-route telemetry contract for `QMoE` and
//! `BlockQuantizedMoE` (issue #1810, **Slice 7A — producer only**).
//!
//! This is the single, non-divergent contract fused into both route kernels.
//! It implements only the **PRODUCE** side of the approved Slice-6 design
//! (`docs/memory/EXPERT_ROUTE_TELEMETRY_SLICE6_DESIGN.md`): a fixed-capacity,
//! GPU-resident **route bitmap** (representation A, §2.1) plus the 6-word
//! identity/epoch **header** (§2.3). It is **default-disabled and inert**:
//! unless a kernel is explicitly armed, the route kernels receive null
//! telemetry pointers and do byte-for-byte identical work.
//!
//! ## Coarse-boundary window contract (Cycle-22 revision)
//!
//! The design (§2.3, §3) is a **coarse-safe-boundary** contract, not a per-call
//! one: the epoch is "bumped by the producer **at each safe boundary**" and the
//! record is consumed "at a boundary, **not per replay**". The **producer**
//! (this module) therefore never resets or advances the epoch on the
//! execute/replay path. Concretely:
//!
//! * **Arm** allocates the fixed-capacity, stable-VA record once through the
//!   existing runtime allocator, stamps request/device identity, opens the
//!   first window (`epoch = 1`), and zeroes the bitmap/counters. No launch.
//! * **Execute / replay** call **only** the fused `atomicOr`/`atomicAdd`/poison
//!   *mark* inside `qmoe_route`/`bqmoe_route`. Every eager call and every
//!   captured replay within a window **accumulates** the routed-expert *union*
//!   and the saturating in-range *count* into the same stable record, with the
//!   epoch held **fixed**. There is **no reset kernel and no host sync** on this
//!   path, and nothing telemetry-related is ever baked into a captured graph
//!   other than the marks themselves.
//! * **Snapshot** (test/observability) reads the record once, after an existing
//!   stream-completion authority (`dtoh` self-synchronizes).
//! * **`reset_boundary`** is the *only* place the window advances. It is an
//!   explicit host-ordered boundary operation: it is **rejected while the EP
//!   stream is capturing** (a drain/`htod` is illegal mid-capture), it drains
//!   prior stream/graph work through the existing `drain_for_unmap` authority,
//!   then bumps the epoch and re-zeroes the record so the next window starts
//!   empty with no stale carryover. It reuses existing authorities only — no new
//!   coordinator, allocator, cache, or PMM/VMM call.
//!
//! This is the cadence a residency **policy** needs — the route union/frequency
//! over the whole coarse interval since the last boundary (which may span many
//! decode steps under one captured graph) — not a per-step point sample thrown
//! away before the next replay begins.
//!
//! ## Scope (what is here / what is deliberately not)
//!
//! * **Here:** the header layout, the route-bitmap semantics, an
//!   `atomicOr`-only device *mark* helper fused into `qmoe_route`/`bqmoe_route`
//!   after the selected expert ids are finalized, an explicit host-ordered
//!   coarse-boundary reset, a persistent stable-VA device buffer allocated once
//!   at arm time through the existing runtime allocator, and a CPU oracle +
//!   boundary validator (§3) used only by tests.
//! * **Not here (future CONSUME work):** the bounded deduplicated route/miss
//!   *queue* (representation B, §2.2) is intentionally **omitted** — it is only
//!   needed by the boundary consumer, which is out of scope for a producer-only
//!   slice. No policy, hot-set, residency plan, mapping, boundary *policy* call
//!   site, or VMM action is implemented. The coarse-boundary consumer decision
//!   (§3) and the flat `layer*num_experts + expert` keying (§2) remain future
//!   work; `reset_boundary`/`consume_and_validate` ship here only as the tested,
//!   crate-internal producer-side seam a future consumer will drive.

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{EpError, Result};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::runtime::CudaRuntime;

// ---------------------------------------------------------------------------
// Header layout (design §2.3). Six u32 words, device-resident, one per armed
// kernel. The approved Slice-6 harness uses this exact layout; the production
// design widens `epoch`/`request` to u64 in the eventual 32-byte header — that
// widening is a forward-compatible CONSUME-side change and is out of scope
// here, so the producer stamps u32 identity matching the proven probe.
// ---------------------------------------------------------------------------

/// Word index of the epoch/generation stamp (design §2.3).
pub const H_EPOCH: usize = 0;
/// Word index of the owning request identity.
pub const H_REQUEST: usize = 1;
/// Word index of the owning CUDA device ordinal.
pub const H_DEVICE: usize = 2;
/// Word index of the sticky overflow flag (unused by the bitmap; reserved for
/// the future dedup queue and used here as an explicit counter-saturation bit).
pub const H_OVERFLOW: usize = 3;
/// Word index of the sticky poison flag (out-of-range expert id — fail closed).
pub const H_POISON: usize = 4;
/// Word index of the bounded total in-range route counter.
pub const H_COUNT: usize = 5;
/// Number of u32 words in the header.
pub const HEADER_LEN: usize = 6;
/// Header size in bytes.
pub const HEADER_BYTES: usize = HEADER_LEN * 4;

/// Number of 32-bit bitmap words needed for `num_experts` experts. The bitmap
/// capacity is `num_experts` bits by construction, so a valid route can never
/// overflow it (design §2.1).
pub fn words_for(num_experts: usize) -> usize {
    num_experts.div_ceil(32)
}

// ---------------------------------------------------------------------------
// Device source.
// ---------------------------------------------------------------------------

/// `__device__` helpers fused into the QMoE/BlockQuantizedMoE route modules.
/// Only integer atomics — no fp16 headers required. Prepended ahead of the route
/// kernels so `qmoe_route`/`bqmoe_route` can call `route_telemetry_mark_row`
/// after the selected expert ids are finalized.
///
/// Contract (design §2.1/§2.3):
/// * `atomicOr` sets bit `e` of the bitmap for each routed expert `e`. `atomicOr`
///   is commutative and associative, so the resulting *set* is independent of
///   thread/route order and needs no ordering fence — the record is a set, not a
///   sequence. Because nothing on the execute/replay path clears it, the bitmap
///   accumulates the routed-expert **union over the whole window** (until the
///   next explicit `reset_boundary`).
/// * an out-of-range id sets the sticky `poison` header bit and never touches the
///   bitmap (fail closed).
/// * a bounded per-row `atomicAdd` maintains `count` = total in-range routes
///   **accumulated over the window**; the only way it can overflow u32 is more
///   than 4 billion routes, which saturates and sets the sticky `overflow` bit
///   explicitly (fail closed, never wraps into a smaller "success" value).
pub(crate) const MARK_DEVICE_SRC: &str = r#"
// ==== expert-route telemetry (issue #1810 Slice 7A, inert observability) ====
// Header word indices: 0 epoch, 1 request, 2 device, 3 overflow, 4 poison,
// 5 count. `route_telemetry_bitmap`/`route_telemetry_header` are null when the
// owning kernel is disarmed, in which case every helper is a no-op and the
// route kernel's outputs are byte-for-byte unchanged.
__device__ __forceinline__ unsigned int route_telemetry_mark_one(
    unsigned int* route_telemetry_bitmap,
    unsigned int* route_telemetry_header,
    int expert,
    int experts)
{
    if (expert < 0 || expert >= experts) {
        atomicOr(&route_telemetry_header[4], 1u); // poison: fail closed
        return 0u;
    }
    atomicOr(&route_telemetry_bitmap[expert >> 5], 1u << (expert & 31));
    return 1u;
}

// Fuse point: called once per routed row after `indices[0..top_k]` is final.
__device__ __forceinline__ void route_telemetry_mark_row(
    unsigned int* route_telemetry_bitmap,
    unsigned int* route_telemetry_header,
    const int* indices,
    int top_k,
    int experts)
{
    if (route_telemetry_bitmap == 0) { return; } // disarmed: inert
    unsigned int valid = 0u;
    for (int slot = 0; slot < top_k; ++slot) {
        valid += route_telemetry_mark_one(
            route_telemetry_bitmap, route_telemetry_header, indices[slot], experts);
    }
    if (valid != 0u) {
        unsigned int* count = &route_telemetry_header[5];
        unsigned int observed = atomicAdd(count, 0u);
        for (;;) {
            if (observed == 0xffffffffu) {
                atomicOr(&route_telemetry_header[3], 1u);
                break;
            }
            bool overflow = valid > 0xffffffffu - observed;
            unsigned int desired = overflow ? 0xffffffffu : observed + valid;
            unsigned int prior = atomicCAS(count, observed, desired);
            if (prior == observed) {
                if (overflow) {
                    atomicOr(&route_telemetry_header[3], 1u);
                }
                break;
            }
            observed = prior;
        }
    }
}
// ==== end expert-route telemetry helpers ====
"#;

// ---------------------------------------------------------------------------
// Configuration and typed rejection (design §3 / requirement: fail-closed,
// never fail ordinary inference because optional telemetry cannot be armed).
// ---------------------------------------------------------------------------

/// Session/kernel-scoped arming request. Default is *disarmed*; a kernel is only
/// armed by an explicit call carrying this config. `device_id` must match the
/// owning runtime's ordinal (multi-device fails closed at arm), and
/// `num_experts` fixes the bitmap capacity for the armed lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteTelemetryConfig {
    /// Owning request/sequence identity, stamped into the header for isolation.
    pub request_id: u32,
    /// Owning CUDA device ordinal; must equal `runtime.ordinal()`.
    pub device_id: u32,
    /// Number of experts; fixes the bitmap capacity (bits) for this arming.
    pub num_experts: usize,
    /// Number of selected experts contributed by every clean routed row.
    pub routes_per_row: usize,
}

/// Typed reason telemetry could not be armed. Arming returns this instead of
/// failing; ordinary inference proceeds with telemetry left disabled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TelemetryUnsupported {
    /// The config's device ordinal does not match the runtime's device.
    DeviceMismatch { config: u32, runtime: u32 },
    /// `num_experts` was zero — no bitmap capacity is representable.
    ZeroExperts,
    /// The execution contract cannot select zero experts or more experts than
    /// the admitted expert domain for one row.
    InvalidRoutesPerRow {
        routes_per_row: usize,
        num_experts: usize,
    },
    /// The arming request disagrees with the kernel's prepared top-k contract.
    RouteWidthMismatch { config: usize, execution: usize },
    /// The device buffer could not be allocated (message carried for the log).
    Alloc(String),
    /// Reconfiguration would invalidate pointers embedded in an installed graph.
    GraphInstalled,
}

impl std::fmt::Display for TelemetryUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceMismatch { config, runtime } => write!(
                f,
                "route telemetry device mismatch: config device {config} != runtime device {runtime} (multi-device fails closed)"
            ),
            Self::ZeroExperts => write!(f, "route telemetry requires num_experts > 0"),
            Self::InvalidRoutesPerRow {
                routes_per_row,
                num_experts,
            } => write!(
                f,
                "route telemetry requires 0 < routes_per_row <= num_experts, got \
                 routes_per_row={routes_per_row} and num_experts={num_experts}"
            ),
            Self::RouteWidthMismatch { config, execution } => write!(
                f,
                "route telemetry routes_per_row={config} does not match the prepared execution \
                 contract {execution}; re-arm with the kernel's actual selected-expert width"
            ),
            Self::Alloc(message) => write!(f, "route telemetry buffer alloc failed: {message}"),
            Self::GraphInstalled => write!(
                f,
                "route telemetry must be configured before device-graph capture"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Armed device state. Persistent, stable-VA buffers allocated once at arm time
// through the existing runtime allocator (requirement: allocate/freeze during
// preparation; capture/replay never allocates, grows, resets on the host, or
// moves pointers). The epoch is a **host-side** window counter — the producer
// never advances it on the device from the execute/replay path; it moves only
// at an explicit host-ordered `reset_boundary`.
// ---------------------------------------------------------------------------

/// Live telemetry buffers + identity for an armed kernel. Owned by the kernel;
/// freed on disarm/drop. Pointers are fixed for the armed lifetime.
#[derive(Debug)]
pub(crate) struct ArmedTelemetry {
    request_id: u32,
    device_id: u32,
    num_experts: usize,
    routes_per_row: u32,
    words: usize,
    bitmap: CUdeviceptr,
    header: CUdeviceptr,
    /// Host-side window/epoch counter. Stamped into `header[H_EPOCH]` at arm
    /// (window 1) and at every `reset_boundary`; held fixed for the whole
    /// window. There is no device epoch counter and no reset kernel.
    epoch: AtomicU32,
    bitmap_bytes: usize,
}

impl ArmedTelemetry {
    /// Arm telemetry: validate identity, allocate the persistent stable-VA
    /// record through the existing runtime allocator, and open the first window
    /// (stamp identity, `epoch = 1`, zero the bitmap/counters). No kernel is
    /// compiled or launched — the only device work telemetry ever does is the
    /// fused `atomicOr`/`atomicAdd` mark inside the route kernel. Returns a typed
    /// rejection if the properties are unsupported; the caller then leaves
    /// telemetry disabled and ordinary inference is unaffected.
    pub(crate) fn arm(
        runtime: &CudaRuntime,
        config: RouteTelemetryConfig,
    ) -> std::result::Result<Self, TelemetryUnsupported> {
        let device = runtime.ordinal();
        if config.device_id != device {
            return Err(TelemetryUnsupported::DeviceMismatch {
                config: config.device_id,
                runtime: device,
            });
        }
        if config.num_experts == 0 {
            return Err(TelemetryUnsupported::ZeroExperts);
        }
        if config.routes_per_row == 0 || config.routes_per_row > config.num_experts {
            return Err(TelemetryUnsupported::InvalidRoutesPerRow {
                routes_per_row: config.routes_per_row,
                num_experts: config.num_experts,
            });
        }
        let routes_per_row = u32::try_from(config.routes_per_row).map_err(|_| {
            TelemetryUnsupported::InvalidRoutesPerRow {
                routes_per_row: config.routes_per_row,
                num_experts: config.num_experts,
            }
        })?;

        let words = words_for(config.num_experts);
        let bitmap_bytes = words * 4;
        let bitmap = runtime
            .alloc_raw(bitmap_bytes.max(1))
            .map_err(|error| TelemetryUnsupported::Alloc(error.to_string()))?;
        let header = match runtime.alloc_raw(HEADER_BYTES) {
            Ok(pointer) => pointer,
            Err(error) => {
                // SAFETY: `bitmap` came from this runtime and no launch reads it.
                unsafe {
                    let _ = runtime.free_raw(bitmap);
                }
                return Err(TelemetryUnsupported::Alloc(error.to_string()));
            }
        };

        let armed = Self {
            request_id: config.request_id,
            device_id: config.device_id,
            num_experts: config.num_experts,
            routes_per_row,
            words,
            bitmap,
            header,
            // Arm opens window 1. `reset_boundary` bumps this to 2, 3, ...
            epoch: AtomicU32::new(1),
            bitmap_bytes,
        };
        // Open the first window: stamp identity + epoch 1 and zero the record.
        if let Err(error) = armed.open_window(runtime) {
            armed.free(runtime);
            return Err(TelemetryUnsupported::Alloc(error.to_string()));
        }
        Ok(armed)
    }

    /// Write the header (`[epoch, request, device, 0, 0, 0]`) and zero the
    /// bitmap on the host, opening a fresh accumulation window. A synchronous
    /// `htod`; only ever called at arm or at an explicit boundary, never on the
    /// execute/replay path and never during capture.
    fn open_window(&self, runtime: &CudaRuntime) -> Result<()> {
        let header_words: [u32; HEADER_LEN] = [
            self.epoch.load(Ordering::Relaxed),
            self.request_id,
            self.device_id,
            0,
            0,
            0,
        ];
        let mut header_bytes = [0u8; HEADER_BYTES];
        for (word, chunk) in header_words.iter().zip(header_bytes.chunks_exact_mut(4)) {
            chunk.copy_from_slice(&word.to_ne_bytes());
        }
        // SAFETY: `header` is a live `HEADER_BYTES` allocation from this runtime;
        // `bitmap` is a live `bitmap_bytes` allocation. Both cover the sources.
        unsafe {
            runtime.htod(&header_bytes, self.header)?;
            if self.bitmap_bytes > 0 {
                let zeros = vec![0u8; self.bitmap_bytes];
                runtime.htod(&zeros, self.bitmap)?;
            }
        }
        Ok(())
    }

    /// Advance to the next accumulation window at an explicit coarse safe
    /// boundary (design §2.3/§3): bump the epoch, re-stamp identity, and zero the
    /// bitmap/counters so the next window starts empty with no stale carryover.
    ///
    /// This is the **only** place the window advances — nothing on the
    /// execute/replay path resets or bumps the epoch. It is a host-ordered
    /// boundary operation that reuses existing runtime authorities only:
    ///
    /// * **Rejected during capture/replay.** A `drain`/`htod` is illegal while
    ///   the EP stream is capturing, so a boundary reset in that state fails
    ///   closed with a typed error rather than corrupting a capture.
    /// * **Ordered after prior stream/graph completion** via the existing
    ///   `drain_for_unmap` drain authority, so the host zeroing can never race a
    ///   route kernel still accumulating into the record.
    ///
    /// It moves no pointer and touches no PMM/VMM/cache.
    pub(crate) fn reset_boundary(&self, runtime: &CudaRuntime) -> Result<()> {
        if runtime.is_capturing()? {
            return Err(EpError::KernelFailed(
                "cuda_ep: route telemetry boundary reset is illegal during graph capture/replay \
                 (the window advances only at a coarse safe boundary)"
                    .into(),
            ));
        }
        // Order the reset after all prior EP-stream work (the fused marks of the
        // window being closed) using the existing drain authority.
        runtime.drain_for_unmap()?;
        self.epoch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |epoch| {
                epoch.checked_add(1)
            })
            .map_err(|_| {
                EpError::KernelFailed(
                    "cuda_ep: route telemetry epoch overflow; re-arm the observer".into(),
                )
            })?;
        self.open_window(runtime)
    }

    /// True iff this arming's fixed capacity matches the executing shape. On a
    /// mismatch the caller leaves telemetry inert for that call (never fails
    /// inference).
    pub(crate) fn matches_experts(&self, experts: usize) -> bool {
        self.num_experts == experts
    }

    /// Device pointer to the route bitmap passed to the fused route kernel.
    pub(crate) fn bitmap_ptr(&self) -> CUdeviceptr {
        self.bitmap
    }

    /// Device pointer to the header passed to the fused route kernel.
    pub(crate) fn header_ptr(&self) -> CUdeviceptr {
        self.header
    }

    /// Total device bytes held by this record (for teardown/accounting tests).
    /// Bitmap + 6-word header; there is no device epoch buffer (the epoch is a
    /// host-side window counter).
    pub(crate) fn footprint_bytes(&self) -> usize {
        self.bitmap_bytes + HEADER_BYTES
    }

    /// Device virtual address of the bitmap (stable for the armed lifetime).
    /// Used by capture/replay tests to prove the pointer never moves.
    pub(crate) fn bitmap_addr(&self) -> CUdeviceptr {
        self.bitmap
    }

    /// Copy the header and bitmap back to the host (test/observability only —
    /// this is *not* the production CONSUME path). `dtoh` self-synchronizes, so
    /// this must never be called on the capture/decode critical path.
    pub(crate) fn snapshot(&self, runtime: &CudaRuntime) -> Result<TelemetrySnapshot> {
        let mut header = [0u32; HEADER_LEN];
        let mut bitmap = vec![0u32; self.words];
        // SAFETY: destinations exactly cover the device records.
        unsafe {
            let header_bytes =
                std::slice::from_raw_parts_mut(header.as_mut_ptr() as *mut u8, HEADER_BYTES);
            runtime.dtoh(header_bytes, self.header)?;
            if self.words > 0 {
                let bitmap_bytes = std::slice::from_raw_parts_mut(
                    bitmap.as_mut_ptr() as *mut u8,
                    self.bitmap_bytes,
                );
                runtime.dtoh(bitmap_bytes, self.bitmap)?;
            }
        }
        Ok(TelemetrySnapshot {
            header,
            bitmap,
            num_experts: self.num_experts,
            routes_per_row: self.routes_per_row,
        })
    }

    /// Read and validate the complete record against this immutable arming.
    pub(crate) fn validated_snapshot(
        &self,
        runtime: &CudaRuntime,
    ) -> Result<ValidatedTelemetrySnapshot> {
        let snapshot = self.snapshot(runtime)?;
        match consume_and_validate(
            &snapshot.header,
            &snapshot.bitmap,
            self.epoch.load(Ordering::Acquire),
            self.request_id,
            self.device_id,
            self.num_experts,
            usize::try_from(self.routes_per_row).expect("u32 routes-per-row fits usize"),
        ) {
            RouteDecision::HotSet(_) => {
                let unique_expert_count = snapshot
                    .bitmap
                    .iter()
                    .try_fold(0_u32, |total, word| total.checked_add(word.count_ones()))
                    .ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep: validated route telemetry unique-expert count overflow"
                                .into(),
                        )
                    })?;
                Ok(ValidatedTelemetrySnapshot {
                    selected_route_count: snapshot.count(),
                    unique_expert_count,
                })
            }
            RouteDecision::WholeBank(reason) => Err(EpError::KernelFailed(format!(
                "cuda_ep: invalid BlockQuantizedMoE traffic record: {reason}"
            ))),
        }
    }

    #[cfg(feature = "gpu-tests")]
    pub(crate) fn inject_header_word(
        &self,
        runtime: &CudaRuntime,
        index: usize,
        value: u32,
    ) -> Result<()> {
        if index >= HEADER_LEN {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: telemetry test header index {index} is out of range"
            )));
        }
        let byte_offset = index.checked_mul(4).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep: telemetry test header offset overflow".into())
        })?;
        let destination = self.header.checked_add(byte_offset as u64).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep: telemetry test header address overflow".into())
        })?;
        // SAFETY: `destination` names one u32 word within the live header.
        unsafe { runtime.htod(&value.to_ne_bytes(), destination) }
    }

    /// Best-effort free. Drains in-flight launches first (a captured/eager route
    /// kernel may still reference these pointers), then releases the record.
    pub(crate) fn free(&self, runtime: &CudaRuntime) {
        let _ = runtime.drain_for_unmap();
        // SAFETY: every pointer came from this runtime, prior launches have been
        // drained, and each is freed exactly once.
        unsafe {
            let _ = runtime.free_raw(self.header);
            let _ = runtime.free_raw(self.bitmap);
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side snapshot, CPU oracle, and boundary validator (design §3). These run
// on the host at a safe boundary against an already-copied record; the producer
// never runs them. Used by tests as the ground truth the device must match, and
// to demonstrate the fail-closed decision the future consumer will make.
// ---------------------------------------------------------------------------

/// A host copy of an armed record (test/observability only).
#[derive(Clone, Debug)]
pub struct TelemetrySnapshot {
    /// The 6-word header.
    pub header: [u32; HEADER_LEN],
    /// The route bitmap words.
    pub bitmap: Vec<u32>,
    /// Expert count the bitmap was sized for.
    pub num_experts: usize,
    /// Number of selected experts every clean routed row contributes.
    pub routes_per_row: u32,
}

impl TelemetrySnapshot {
    /// The set of experts whose bit is set, ascending.
    pub fn routed_experts(&self) -> Vec<usize> {
        (0..self.num_experts)
            .filter(|&e| self.bitmap[e >> 5] & (1u32 << (e & 31)) != 0)
            .collect()
    }

    /// Epoch stamp of this record.
    pub fn epoch(&self) -> u32 {
        self.header[H_EPOCH]
    }

    /// Total in-range routes recorded.
    pub fn count(&self) -> u32 {
        self.header[H_COUNT]
    }

    /// True iff the sticky poison bit is set.
    pub fn poison(&self) -> bool {
        self.header[H_POISON] != 0
    }

    /// True iff the sticky overflow (counter-saturation) bit is set.
    pub fn overflow(&self) -> bool {
        self.header[H_OVERFLOW] != 0
    }
}

pub(crate) struct ValidatedTelemetrySnapshot {
    selected_route_count: u32,
    unique_expert_count: u32,
}

impl ValidatedTelemetrySnapshot {
    pub(crate) fn selected_route_count(&self) -> u32 {
        self.selected_route_count
    }

    pub(crate) fn unique_expert_count(&self) -> u32 {
        self.unique_expert_count
    }
}

/// Reference route bitmap: bit `e` set iff expert `e` is routed at least once.
/// Returns `(bitmap_words, poison)` where poison is true iff any id is out of
/// range. Pure host code — the ground truth every device test diffs against.
pub fn cpu_bitmap(routes: &[i32], num_experts: usize) -> (Vec<u32>, bool) {
    let mut bits = vec![0u32; words_for(num_experts)];
    let mut poison = false;
    for &route in routes {
        if route < 0 || route as usize >= num_experts {
            poison = true;
            continue;
        }
        let e = route as usize;
        bits[e >> 5] |= 1u32 << (e & 31);
    }
    (bits, poison)
}

/// The §3 boundary decision. Fail-closed on any defect (poison/overflow/foreign
/// identity/stale epoch); carries the reason (design-discipline "carry the
/// reason"). This is the shape of the future consumer's decision; Slice 7A ships
/// it only as a tested host reference, wired to no production path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteDecision {
    /// Trustworthy record; the hot-set is the routed bitmap words.
    HotSet(Vec<u32>),
    /// Fail closed to the whole-bank proof, with the reason.
    WholeBank(String),
}

/// Validate an already-copied record against the expected identity/epoch. Any
/// mismatch fails closed to `WholeBank` (design §3).
pub fn consume_and_validate(
    header: &[u32],
    bitmap: &[u32],
    expected_epoch: u32,
    expected_request: u32,
    expected_device: u32,
    expected_num_experts: usize,
    expected_routes_per_row: usize,
) -> RouteDecision {
    if header.len() != HEADER_LEN {
        return RouteDecision::WholeBank(format!(
            "header length mismatch: record={} expected {HEADER_LEN}",
            header.len()
        ));
    }
    let expected_words = words_for(expected_num_experts);
    if bitmap.len() != expected_words {
        return RouteDecision::WholeBank(format!(
            "bitmap length mismatch: record={} expected {expected_words}",
            bitmap.len()
        ));
    }
    if expected_routes_per_row == 0 || expected_routes_per_row > expected_num_experts {
        return RouteDecision::WholeBank(format!(
            "invalid route-width contract: routes_per_row={expected_routes_per_row}, \
             num_experts={expected_num_experts}"
        ));
    }
    if header[H_POISON] != 0 {
        return RouteDecision::WholeBank("poison: out-of-range expert id observed".into());
    }
    if header[H_OVERFLOW] != 0 {
        return RouteDecision::WholeBank("overflow: bounded route counter saturated".into());
    }
    if header[H_DEVICE] != expected_device {
        return RouteDecision::WholeBank(format!(
            "device mismatch: record dev={} expected {expected_device}",
            header[H_DEVICE]
        ));
    }
    if header[H_REQUEST] != expected_request {
        return RouteDecision::WholeBank(format!(
            "request mismatch: record req={} expected {expected_request}",
            header[H_REQUEST]
        ));
    }
    if header[H_EPOCH] != expected_epoch {
        return RouteDecision::WholeBank(format!(
            "epoch mismatch: record epoch={} expected {expected_epoch}",
            header[H_EPOCH]
        ));
    }
    if let Some(last) = bitmap.last() {
        let valid_tail_bits = expected_num_experts % 32;
        if valid_tail_bits != 0 && (*last >> valid_tail_bits) != 0 {
            return RouteDecision::WholeBank(
                "bitmap contains experts outside the armed capacity".into(),
            );
        }
    }
    let Some(unique) = bitmap
        .iter()
        .try_fold(0u32, |total, word| total.checked_add(word.count_ones()))
    else {
        return RouteDecision::WholeBank("unique expert count overflow".into());
    };
    let count = header[H_COUNT];
    let Ok(routes_per_row) = u32::try_from(expected_routes_per_row) else {
        return RouteDecision::WholeBank(format!(
            "route-width contract {expected_routes_per_row} exceeds the telemetry counter domain"
        ));
    };
    if !count.is_multiple_of(routes_per_row) {
        return RouteDecision::WholeBank(format!(
            "route count {count} is impossible for routes_per_row={routes_per_row}; every clean \
             routed row contributes exactly {routes_per_row} selections"
        ));
    }
    if count < unique || (count == 0) != (unique == 0) {
        return RouteDecision::WholeBank(format!(
            "route count {count} is inconsistent with {unique} unique selected experts"
        ));
    }
    RouteDecision::HotSet(bitmap.to_vec())
}

#[allow(dead_code)]
fn _assert_ep6_contract() {
    // Compile-time-ish guard that the header stays 6 words (design §2.3).
    const _: () = assert!(HEADER_LEN == 6);
}

impl From<TelemetryUnsupported> for EpError {
    fn from(reason: TelemetryUnsupported) -> Self {
        EpError::KernelFailed(format!("cuda_ep: {reason}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_for_rounds_up() {
        assert_eq!(words_for(0), 0);
        assert_eq!(words_for(1), 1);
        assert_eq!(words_for(32), 1);
        assert_eq!(words_for(33), 2);
        assert_eq!(words_for(64), 2);
        assert_eq!(words_for(160), 5);
    }

    #[test]
    fn cpu_bitmap_sets_routed_bits_and_no_poison() {
        let (bits, poison) = cpu_bitmap(&[0, 5, 31, 32, 63], 64);
        assert!(!poison);
        assert_eq!(bits.len(), 2);
        assert_eq!(bits[0], (1 << 0) | (1 << 5) | (1 << 31));
        assert_eq!(bits[1], (1 << 0) | (1 << 31)); // experts 32 and 63
    }

    #[test]
    fn cpu_bitmap_flags_out_of_range_as_poison() {
        let (_bits, low) = cpu_bitmap(&[-1], 8);
        assert!(low, "negative id must poison");
        let (bits, high) = cpu_bitmap(&[8, 2], 8);
        assert!(high, "id == num_experts must poison");
        // The in-range id is still recorded even alongside a poisoning id.
        assert_eq!(bits[0], 1 << 2);
    }

    #[test]
    fn validate_accepts_matching_clean_record() {
        let mut header = [0u32; HEADER_LEN];
        header[H_EPOCH] = 4;
        header[H_REQUEST] = 9;
        header[H_DEVICE] = 1;
        header[H_COUNT] = 2;
        let bitmap = vec![0b1010u32];
        assert_eq!(
            consume_and_validate(&header, &bitmap, 4, 9, 1, 32, 2),
            RouteDecision::HotSet(vec![0b1010u32])
        );
        assert!(matches!(
            consume_and_validate(&header, &bitmap, 3, 9, 1, 32, 2),
            RouteDecision::WholeBank(_)
        ));
    }

    #[test]
    fn validate_fails_closed_on_each_defect() {
        let clean = |epoch: u32, request: u32, device: u32| {
            let mut header = [0u32; HEADER_LEN];
            header[H_EPOCH] = epoch;
            header[H_REQUEST] = request;
            header[H_DEVICE] = device;
            header[H_COUNT] = 1;
            header
        };
        let bitmap = vec![1u32];

        let mut poisoned = clean(4, 9, 1);
        poisoned[H_POISON] = 1;
        assert!(matches!(
            consume_and_validate(&poisoned, &bitmap, 4, 9, 1, 32, 1),
            RouteDecision::WholeBank(_)
        ));

        let mut overflowed = clean(4, 9, 1);
        overflowed[H_OVERFLOW] = 1;
        assert!(matches!(
            consume_and_validate(&overflowed, &bitmap, 4, 9, 1, 32, 1),
            RouteDecision::WholeBank(_)
        ));

        // Foreign device / request identity fails closed (isolation).
        assert!(matches!(
            consume_and_validate(&clean(4, 9, 2), &bitmap, 4, 9, 1, 32, 1),
            RouteDecision::WholeBank(_)
        ));
        assert!(matches!(
            consume_and_validate(&clean(4, 8, 1), &bitmap, 4, 9, 1, 32, 1),
            RouteDecision::WholeBank(_)
        ));

        // A stale epoch (record older than the boundary) fails closed.
        assert!(matches!(
            consume_and_validate(&clean(2, 9, 1), &bitmap, 3, 9, 1, 32, 1),
            RouteDecision::WholeBank(_)
        ));

        let mut inconsistent_count = clean(4, 9, 1);
        inconsistent_count[H_COUNT] = 0;
        assert!(matches!(
            consume_and_validate(&inconsistent_count, &bitmap, 4, 9, 1, 32, 1),
            RouteDecision::WholeBank(_)
        ));
        assert!(matches!(
            consume_and_validate(&clean(4, 9, 1), &[1, 0], 4, 9, 1, 32, 1),
            RouteDecision::WholeBank(_)
        ));
        assert!(matches!(
            consume_and_validate(&clean(4, 9, 1), &[1 << 31], 4, 9, 1, 17, 1),
            RouteDecision::WholeBank(_)
        ));

        let mut non_multiple = clean(4, 9, 1);
        non_multiple[H_COUNT] = 3;
        assert!(matches!(
            consume_and_validate(&non_multiple, &bitmap, 4, 9, 1, 32, 2),
            RouteDecision::WholeBank(reason) if reason.contains("impossible")
        ));
    }

    #[test]
    fn device_counter_uses_saturating_cas_not_wrapping_add() {
        assert!(MARK_DEVICE_SRC.contains("atomicCAS(count, observed, desired)"));
        assert!(MARK_DEVICE_SRC.contains("desired = overflow ? 0xffffffffu"));
        assert!(
            !MARK_DEVICE_SRC.contains("atomicAdd(&route_telemetry_header[5], valid)"),
            "the production counter must never use a wrapping increment"
        );
    }

    #[test]
    fn snapshot_reports_header_fields_and_routed_experts() {
        let mut header = [0u32; HEADER_LEN];
        header[H_EPOCH] = 7;
        header[H_COUNT] = 3;
        header[H_POISON] = 1;
        header[H_OVERFLOW] = 1;
        let snapshot = TelemetrySnapshot {
            header,
            bitmap: vec![(1 << 1) | (1 << 4)],
            num_experts: 8,
            routes_per_row: 1,
        };
        assert_eq!(snapshot.epoch(), 7);
        assert_eq!(snapshot.count(), 3);
        assert!(snapshot.poison());
        assert!(snapshot.overflow());
        assert_eq!(snapshot.routed_experts(), vec![1, 4]);
    }

    #[test]
    fn device_mismatch_display_is_descriptive() {
        let reason = TelemetryUnsupported::DeviceMismatch {
            config: 3,
            runtime: 0,
        };
        let text = reason.to_string();
        assert!(text.contains("device mismatch"));
        assert!(text.contains("fails closed"));
    }
}
