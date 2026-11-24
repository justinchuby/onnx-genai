//! B7 — stream-ordered CSA checkpoint/restore (D6, §4.6) + observability
//! metrics (§8).
//!
//! ## Checkpoint / restore
//!
//! Speculative decode drafts several tokens, then verifies and either accepts a
//! prefix or rejects the tail. The CSA backend owns the authoritative,
//! device-resident logical-length cursors and the bounded active carry state; the
//! engine owns the composite checkpoint orchestration (D6). A
//! [`CsaCheckpointJournal`] captures — with **no recompression** — the five
//! logical cursors plus a device snapshot of the bounded overwritten carry
//! buffers into pre-reserved, stable-address scratch. [`restore_prefix`] rolls
//! them back to the accepted prefix.
//!
//! Both operations are **stream-ordered**: the carry snapshot/restore are device
//! `cuMemcpyDtoD` copies and the cursor rollback is a bounded scalar write, so
//! the physical inactive record tail may stay stale while every reader is
//! length-masked. Checkpoint/restore run **between** captured decode steps (the
//! draft/verify/correct boundary), never inside a captured region — so they add
//! no host sync to the captured graph, and the stable snapshot addresses keep a
//! replayed graph reading the rolled-back state correctly.
//!
//! [`restore_prefix`]: CsaCheckpoint::restore_prefix
//!
//! ## Metrics (§8)
//!
//! [`CsaMetrics`] is the shared telemetry surface threaded from the EP into every
//! CSA kernel (instance state on the provider, not a process global). Kernels
//! record the per-layer attention mode, bytes avoided vs. host staging, the five
//! cursor lengths, coarse stage timings, sink mass, and host/device byte counts;
//! the journal accumulates rollback counts. Recording is gated off the captured
//! hot path (host-side struct updates only, skipped while a graph is capturing).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{
    DevicePtrMut, EpError, ExecutionProvider, Result, WorkspaceAllocation, WorkspaceLayout,
    WorkspaceView,
};

use crate::provider::CudaExecutionProvider;

/// The five logical CSA cursors (D6, §4.6). Every cursor is a bounded scalar
/// derived from the sequence cursor and the compression ratio, so capturing them
/// is a pure host-side length snapshot with no device sync.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CsaCursors {
    /// Selection / sequence cursor (`total_sequence_length`).
    pub seq_cursor: u64,
    /// Completed main compressed-KV records.
    pub compressed_len: u64,
    /// Positions held in the main compression carry (partial record fill).
    pub compression_carry_len: u64,
    /// Completed index-key records.
    pub index_len: u64,
    /// Positions held in the index compression carry.
    pub index_carry_len: u64,
}

impl CsaCursors {
    /// Derive all five cursors from the sequence cursor and compression ratio.
    /// Main and index streams both compress at `ratio`, so their record and
    /// carry lengths coincide here; they are modelled independently so a future
    /// asymmetric-ratio config can diverge without an API change.
    pub fn from_sequence(seq_cursor: u64, ratio: u64) -> Self {
        let ratio = ratio.max(1);
        let compressed_len = seq_cursor / ratio;
        let carry_len = seq_cursor % ratio;
        Self {
            seq_cursor,
            compressed_len,
            compression_carry_len: carry_len,
            index_len: compressed_len,
            index_carry_len: carry_len,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CheckpointStamp {
    cursors: CsaCursors,
    generation: u64,
}
impl CheckpointStamp {
    fn restore_scalar(
        &self,
        accepted: u64,
        generation: u64,
        write_scalar: bool,
    ) -> Result<Option<[u8; 8]>> {
        if generation != self.generation {
            return Err(EpError::KernelFailed(format!(
                "CSA restore generation {generation} does not match checkpoint {}; use the active checkpoint generation or start a new checkpoint",
                self.generation
            )));
        }
        if accepted > self.cursors.seq_cursor {
            return Err(EpError::KernelFailed(format!(
                "CSA restore accepted prefix {accepted} exceeds checkpoint {}; accept at most the checkpoint cursor and replay any later accepted tokens",
                self.cursors.seq_cursor
            )));
        }
        write_scalar.then(|| i64::try_from(accepted).map(i64::to_ne_bytes).map_err(|_| EpError::KernelFailed(format!("CSA restore accepted prefix {accepted} cannot fit the int64 device sequence cursor; reduce the accepted cursor before restoring")))).transpose()
    }
}

#[derive(Clone, Copy, Debug)]
struct CarrySpan {
    ptr: CUdeviceptr,
    bytes: usize,
}

fn snapshot_layout(main_bytes: usize, index_bytes: usize) -> Result<WorkspaceLayout<2>> {
    let mut regions = [std::alloc::Layout::new::<()>(); 2];
    for (index, bytes) in [main_bytes, index_bytes].into_iter().enumerate() {
        if bytes != 0 {
            regions[index] = std::alloc::Layout::from_size_align(bytes, 256).map_err(|source| EpError::Backend {
                operation: "plan CSA checkpoint snapshot region; reduce main or index carry capacity",
                source: Box::new(source),
            })?;
        }
    }
    WorkspaceLayout::new(regions)
}

fn validate_span(span: CarrySpan, storage: Option<WorkspaceView>, label: &str) -> Result<()> {
    if span.bytes == 0 {
        return Ok(());
    }
    let end = u64::try_from(span.bytes).ok().and_then(|bytes| span.ptr.checked_add(bytes)).filter(|_| span.ptr != 0).ok_or_else(|| EpError::KernelFailed(format!("CSA checkpoint {label} at {:#x} with {} bytes is null or wraps device address space; pass a live, nonoverlapping device span", span.ptr, span.bytes)))?;
    if let Some(storage) = storage {
        let start = storage.ptr().0 as usize as u64;
        let storage_end = start.checked_add(storage.bytes() as u64).ok_or_else(|| EpError::KernelFailed("CSA checkpoint storage wraps device address space; recreate the journal through the provider's native allocation path".into()))?;
        if span.ptr < storage_end && start < end {
            return Err(EpError::KernelFailed(format!(
                "CSA checkpoint {label} overlaps journal snapshot storage; keep live carry/cursor and reserved snapshots in distinct allocations"
            )));
        }
    }
    Ok(())
}

/// An exclusive borrow of one journal snapshot. Dropping the token unlocks the
/// journal for another checkpoint. The token cannot be cloned or restored via
/// another journal; its original carry addresses are retained internally.
///
/// Keep a snapshot for as long as it may be restored:
/// ```no_run
/// use onnx_runtime_ep_cuda::CsaCheckpointJournal;
/// fn round_trip(journal: &mut CsaCheckpointJournal<'_>) -> onnx_runtime_ep_api::Result<()> {
///     let mut checkpoint = unsafe { journal.checkpoint(0, 0, 0, 0, 0, 1)? };
///     unsafe { checkpoint.restore_prefix(0, 1, None)?; }
///     drop(checkpoint);
///     let _next = unsafe { journal.checkpoint(0, 0, 0, 0, 0, 1)? };
///     Ok(())
/// }
/// ```
/// Overwriting a still-used snapshot is rejected by the borrow checker:
/// ```compile_fail,E0499
/// use onnx_runtime_ep_cuda::CsaCheckpointJournal;
/// fn overwrite(journal: &mut CsaCheckpointJournal<'_>) -> onnx_runtime_ep_api::Result<()> {
///     let mut first = unsafe { journal.checkpoint(0, 0, 0, 0, 0, 1)? };
///     let _second = unsafe { journal.checkpoint(0, 0, 0, 0, 0, 1)? };
///     unsafe { first.restore_prefix(0, 1, None)?; }
///     Ok(())
/// }
/// ```
pub struct CsaCheckpoint<'journal, 'provider> {
    journal: &'journal mut CsaCheckpointJournal<'provider>,
    stamp: CheckpointStamp,
    carries: [CarrySpan; 2],
}
impl std::fmt::Debug for CsaCheckpoint<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsaCheckpoint")
            .field("stamp", &self.stamp)
            .field("carries", &self.carries)
            .finish()
    }
}
impl CsaCheckpoint<'_, '_> {
    pub fn cursors(&self) -> CsaCursors {
        self.stamp.cursors
    }
    pub fn generation(&self) -> u64 {
        self.stamp.generation
    }

    /// Restore the original carry spans and return the accepted logical cursors.
    /// A device failure can leave a partial restore: stop inference and retry
    /// this token explicitly, or retire the journal after an unrecoverable error.
    /// Only a complete successful restore increments rollback metrics. The
    /// generation must match and accepted must not exceed the checkpoint cursor;
    /// accepting later drafted tokens requires the engine's explicit replay.
    ///
    /// # Safety
    /// Original carry spans must still be live in this provider's context and
    /// exclusively writable. An optional sequence scalar must be a live writable
    /// int64 in that context. No span may overlap journal snapshot storage.
    pub unsafe fn restore_prefix(
        &mut self,
        accepted: u64,
        generation: u64,
        seq_scalar: Option<CUdeviceptr>,
    ) -> Result<CsaCursors> {
        let scalar_bytes = self
            .stamp
            .restore_scalar(accepted, generation, seq_scalar.is_some())?;
        let storage = self.journal.storage();
        for (index, span) in self.carries.into_iter().enumerate() {
            validate_span(
                span,
                storage,
                if index == 0 {
                    "main carry"
                } else {
                    "index carry"
                },
            )?;
        }
        if let Some(ptr) = seq_scalar {
            validate_span(CarrySpan { ptr, bytes: 8 }, storage, "sequence scalar")?;
        }
        self.journal.require_eager("restore")?;
        let snapshots = self.journal.layout.bind(storage)?;
        for (carry, snapshot) in self.carries.into_iter().zip(snapshots) {
            if carry.bytes != 0 {
                // SAFETY: exclusive journal borrow keeps the snapshot live;
                // caller promises the original destination remains valid.
                unsafe {
                    self.journal.provider.runtime().dtod(
                        snapshot.ptr().0 as usize as u64,
                        carry.ptr,
                        carry.bytes,
                    )?;
                }
            }
        }
        if let (Some(ptr), Some(bytes)) = (seq_scalar, scalar_bytes) {
            // SAFETY: scalar span is validated above and live per the contract.
            unsafe {
                self.journal.provider.runtime().htod(&bytes, ptr)?;
            }
        }
        self.journal.metrics.record_rollback();
        Ok(CsaCursors::from_sequence(accepted, self.journal.ratio))
    }
}

/// A dedicated native owner for a checkpoint window's main/index snapshots.
/// This storage is separate from the reusable compute peak. Provider borrowing
/// keeps its release queue alive; creation and retirement are the only quota
/// boundaries. Checkpoint/restore perform no allocation or quota operation.
pub struct CsaCheckpointJournal<'provider> {
    provider: &'provider CudaExecutionProvider,
    backing: Option<WorkspaceAllocation>,
    layout: WorkspaceLayout<2>,
    ratio: u64,
    metrics: Arc<CsaMetrics>,
}
impl std::fmt::Debug for CsaCheckpointJournal<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsaCheckpointJournal")
            .field("ratio", &self.ratio)
            .field("snapshot_bytes", &self.layout.bytes())
            .finish()
    }
}
impl<'provider> CsaCheckpointJournal<'provider> {
    /// Reserve both snapshots as one governed allocation. Zero capacity means
    /// no allocation, not an implicit one-byte carry. A zero ratio is invalid.
    pub fn new(
        provider: &'provider CudaExecutionProvider,
        ratio: u64,
        main_carry_bytes: usize,
        index_carry_bytes: usize,
        metrics: Arc<CsaMetrics>,
    ) -> Result<Self> {
        if ratio == 0 {
            return Err(EpError::KernelFailed("CSA checkpoint compression ratio must be positive; pass the declared compression ratio".into()));
        }
        let layout = snapshot_layout(main_carry_bytes, index_carry_bytes)?;
        let backing = if layout.bytes() == 0 {
            None
        } else {
            Some(provider.allocate_workspace(
                layout.bytes(),
                256,
                onnx_runtime_memory_api::MemoryRole::Workspace { step_scoped: false },
            )?)
        };
        Ok(Self {
            provider,
            backing,
            layout,
            ratio,
            metrics,
        })
    }

    fn storage(&self) -> Option<WorkspaceView> {
        self.backing
            .as_ref()
            .map(|owner| WorkspaceView::new(DevicePtrMut(owner.as_ptr().cast_mut()), owner.len()))
    }
    fn require_eager(&self, operation: &str) -> Result<()> {
        if self.provider.runtime().is_capturing()? {
            return Err(EpError::KernelFailed(format!(
                "CSA checkpoint {operation} is illegal during CUDA graph capture; finish capture before operating on the speculative checkpoint window"
            )));
        }
        Ok(())
    }

    /// Snapshot both bounded carries once, without recompression. Synchronous
    /// D2D preserves the existing completed-before-return behavior.
    ///
    /// # Safety
    /// Nonempty carry spans must be live in this provider's device context,
    /// readable through this call, and not overlap journal storage. Producing
    /// writes must be ordered before these copies, with no concurrent carry
    /// mutation while snapshotting. They must remain live/exclusively writable
    /// for every later token restore.
    pub unsafe fn checkpoint(
        &mut self,
        main_carry: CUdeviceptr,
        index_carry: CUdeviceptr,
        main_carry_bytes: usize,
        index_carry_bytes: usize,
        seq_cursor: u64,
        generation: u64,
    ) -> Result<CsaCheckpoint<'_, 'provider>> {
        let storage = self.storage();
        let snapshots = self.layout.bind(storage)?;
        let carries = [
            CarrySpan {
                ptr: main_carry,
                bytes: main_carry_bytes,
            },
            CarrySpan {
                ptr: index_carry,
                bytes: index_carry_bytes,
            },
        ];
        for (index, (carry, snapshot)) in carries.iter().zip(&snapshots).enumerate() {
            if carry.bytes > snapshot.bytes() {
                return Err(EpError::KernelFailed(format!(
                    "CSA checkpoint {} carry requires {} bytes but its snapshot capacity is {}; reserve enough journal capacity before checkpointing",
                    if index == 0 { "main" } else { "index" },
                    carry.bytes,
                    snapshot.bytes()
                )));
            }
            validate_span(
                *carry,
                storage,
                if index == 0 {
                    "main carry"
                } else {
                    "index carry"
                },
            )?;
        }
        self.require_eager("snapshot")?;
        for (carry, snapshot) in carries.into_iter().zip(snapshots) {
            if carry.bytes != 0 {
                // SAFETY: bound snapshot covers the source extent, which is
                // readable and nonoverlapping per the caller's contract.
                unsafe {
                    self.provider.runtime().dtod(
                        carry.ptr,
                        snapshot.ptr().0 as usize as u64,
                        carry.bytes,
                    )?;
                }
            }
        }
        let stamp = CheckpointStamp {
            cursors: CsaCursors::from_sequence(seq_cursor, self.ratio),
            generation,
        };
        Ok(CsaCheckpoint {
            journal: self,
            stamp,
            carries,
        })
    }

    pub fn metrics(&self) -> &Arc<CsaMetrics> {
        &self.metrics
    }

    /// Hand the dedicated owner to the provider's ordered release path. Success
    /// means release was accepted, not that quota has already been returned.
    /// Call after capture ends. On failure native ownership is quarantined and
    /// remains charged; the consumed journal is never retried by Drop.
    pub fn release(mut self) -> Result<()> {
        self.retire()
    }
    fn retire(&mut self) -> Result<()> {
        let Some(owner) = self.backing.take() else {
            return Ok(());
        };
        // Taking first makes both success and failure single-attempt. A rejected
        // preflight drops the native owner into its existing quarantine path.
        self.require_eager("retirement")?;
        self.provider.deallocate_workspace(owner)
    }
}
impl Drop for CsaCheckpointJournal<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.retire() {
            eprintln!(
                "cuda_ep: CSA checkpoint journal retirement failed; native ownership remains charged/quarantined: {error}"
            );
        }
    }
}

/// Per-layer attention execution mode (§8 "attention mode per layer").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CsaAttentionMode {
    /// Host-staged oracle path (diagnostic fallback, D7).
    #[default]
    Host,
    /// Device-resident, capture-clean path (default after B7 switchover).
    Device,
}

/// One decode's worth of §8 observability for a single CSA layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct CsaLayerMetrics {
    /// Attention mode taken this decode.
    pub mode: CsaAttentionMode,
    /// The five logical cursor lengths after the decode.
    pub cursors: CsaCursors,
    /// Host↔device staging bytes avoided by staying on device this decode.
    pub bytes_avoided: u64,
    /// Bytes staged through the host this decode (0 on the device path).
    pub host_bytes: u64,
    /// Device output bytes produced this decode.
    pub device_bytes: u64,
    /// Attention sink probability mass (0.0 when not sampled off the hot path).
    pub sink_mass: f32,
    /// Coarse per-pipeline-stage timings in microseconds (0 on the device path,
    /// where timing inside capture is illegal).
    pub stage_timings_us: [u32; 8],
    /// Number of decodes recorded for this layer.
    pub decode_count: u64,
}

/// Shared CSA telemetry surface (§8). Instance state threaded from the EP into
/// every CSA kernel — not a process-wide global. Cheap host-side struct/atomic
/// updates only, so recording never issues a device op on the captured stream.
#[derive(Debug, Default)]
pub struct CsaMetrics {
    rollback_count: AtomicU64,
    device_bytes_total: AtomicU64,
    host_bytes_total: AtomicU64,
    bytes_avoided_total: AtomicU64,
    layers: Mutex<BTreeMap<u64, CsaLayerMetrics>>,
}

impl CsaMetrics {
    /// Record one decode's observability for `layer_id`.
    pub fn record_layer(&self, layer_id: u64, sample: CsaLayerMetrics) {
        self.device_bytes_total
            .fetch_add(sample.device_bytes, Ordering::Relaxed);
        self.host_bytes_total
            .fetch_add(sample.host_bytes, Ordering::Relaxed);
        self.bytes_avoided_total
            .fetch_add(sample.bytes_avoided, Ordering::Relaxed);
        let mut layers = self.layers.lock().expect("CSA metrics mutex poisoned");
        let entry = layers.entry(layer_id).or_default();
        let decode_count = entry.decode_count + 1;
        *entry = sample;
        entry.decode_count = decode_count;
    }

    /// Increment the speculative rollback counter (§8 "rollback counts").
    pub fn record_rollback(&self) {
        self.rollback_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Total speculative rollbacks observed.
    pub fn rollback_count(&self) -> u64 {
        self.rollback_count.load(Ordering::Relaxed)
    }

    /// Cumulative device output bytes across all recorded decodes.
    pub fn device_bytes_total(&self) -> u64 {
        self.device_bytes_total.load(Ordering::Relaxed)
    }

    /// Cumulative host-staged bytes across all recorded decodes.
    pub fn host_bytes_total(&self) -> u64 {
        self.host_bytes_total.load(Ordering::Relaxed)
    }

    /// Cumulative bytes avoided by the device path across all recorded decodes.
    pub fn bytes_avoided_total(&self) -> u64 {
        self.bytes_avoided_total.load(Ordering::Relaxed)
    }

    /// Snapshot the most recent metrics for `layer_id`, if any decode ran.
    pub fn layer(&self, layer_id: u64) -> Option<CsaLayerMetrics> {
        self.layers
            .lock()
            .expect("CSA metrics mutex poisoned")
            .get(&layer_id)
            .copied()
    }

    /// Number of distinct CSA layers that have recorded a decode.
    pub fn layer_count(&self) -> usize {
        self.layers
            .lock()
            .expect("CSA metrics mutex poisoned")
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_keep_exact_capacities_and_zero_demand() {
        let layout = snapshot_layout(17, 9).unwrap();
        assert_eq!(layout.bytes(), 256 + 9);
        let ptr = DevicePtrMut(0x1000usize as *mut _);
        let regions = layout
            .bind(Some(WorkspaceView::new(ptr, layout.bytes())))
            .unwrap();
        assert_eq!(regions.map(|r| r.bytes()), [17, 9]);
        assert_eq!(regions[1].ptr().0 as usize, 0x1100);
        assert_eq!(snapshot_layout(0, 0).unwrap().bytes(), 0);
        assert_eq!(snapshot_layout(0, 9).unwrap().bytes(), 9);
        assert!(snapshot_layout(usize::MAX, 1).is_err());
        assert!(
            layout
                .bind(Some(WorkspaceView::new(ptr, layout.bytes() - 1)))
                .is_err()
        );
    }

    #[test]
    fn restore_preflight_rejects_bad_generation_prefix_and_int64_before_writes() {
        let stamp = CheckpointStamp {
            cursors: CsaCursors::from_sequence(17, 4),
            generation: 23,
        };
        assert!(
            stamp
                .restore_scalar(16, 24, true)
                .unwrap_err()
                .to_string()
                .contains("generation 24")
        );
        assert!(
            stamp
                .restore_scalar(18, 23, true)
                .unwrap_err()
                .to_string()
                .contains("prefix 18")
        );
        assert_eq!(
            stamp.restore_scalar(16, 23, true).unwrap(),
            Some(16i64.to_ne_bytes())
        );
        let large = CheckpointStamp {
            cursors: CsaCursors::from_sequence(u64::MAX, 4),
            generation: 23,
        };
        assert!(
            large
                .restore_scalar(u64::MAX, 23, true)
                .unwrap_err()
                .to_string()
                .contains("int64")
        );
        assert!(large.restore_scalar(u64::MAX, 23, false).unwrap().is_none());
    }

    #[test]
    fn live_carry_ranges_cannot_overlap_reserved_snapshots_or_wrap() {
        let storage = Some(WorkspaceView::new(DevicePtrMut(0x1000usize as *mut _), 512));
        validate_span(
            CarrySpan {
                ptr: 0x2000,
                bytes: 16,
            },
            storage,
            "main carry",
        )
        .unwrap();
        validate_span(CarrySpan { ptr: 0, bytes: 0 }, storage, "empty carry").unwrap();
        for span in [
            CarrySpan { ptr: 0, bytes: 8 },
            CarrySpan {
                ptr: u64::MAX,
                bytes: 8,
            },
            CarrySpan {
                ptr: 0x0fff,
                bytes: 2,
            },
            CarrySpan {
                ptr: 0x11ff,
                bytes: 1,
            },
        ] {
            assert!(validate_span(span, storage, "main carry").is_err());
        }
    }

    #[cfg(feature = "gpu-tests")]
    #[test]
    fn native_snapshot_owner_is_stable_and_refunds_only_after_provider_settlement() {
        use onnx_runtime_ep_api::DeviceBuffer;
        use onnx_runtime_memory_governor::AllocationSettlementStatus;
        use std::time::Duration;
        let ep = CudaExecutionProvider::new_default().expect("CUDA device required with gpu-tests");
        let manager = ep.process_memory_manager();
        let fingerprint = || {
            manager
                .snapshot()
                .unwrap()
                .allocations
                .into_iter()
                .map(|a| {
                    (
                        a.identity,
                        a.state,
                        a.charged_bytes,
                        a.process_reserved_bytes,
                    )
                })
                .collect::<Vec<_>>()
        };
        let before = fingerprint();
        let metrics = ep.csa_metrics().clone();
        assert!(CsaCheckpointJournal::new(&ep, 0, 8, 4, metrics.clone()).is_err());
        assert!(CsaCheckpointJournal::new(&ep, 4, usize::MAX, 4, metrics.clone()).is_err());
        assert_eq!(fingerprint(), before);
        let mut journal = CsaCheckpointJournal::new(&ep, 4, 12, 6, metrics.clone()).unwrap();
        assert_eq!(
            fingerprint().len(),
            before.len() + 1,
            "one complete native snapshot allocation"
        );
        let settlement = journal
            .backing
            .as_ref()
            .unwrap()
            .managed_settlement_wait()
            .unwrap();
        assert!(settlement.charged_bytes() > 0);
        let runtime = ep.runtime();
        let upload = |data: &[u8]| -> DeviceBuffer {
            let buffer = ep.allocate(data.len(), 256).unwrap();
            unsafe { runtime.htod(data, buffer.as_ptr() as usize as u64) }.unwrap();
            buffer
        };
        let initial_main = [1u8; 16];
        let initial_index = [2u8; 8];
        let main = upload(&initial_main);
        let index = upload(&initial_index);
        let scalar = upload(&99i64.to_ne_bytes());
        let main_ptr = main.as_ptr() as usize as u64;
        let index_ptr = index.as_ptr() as usize as u64;
        let scalar_ptr = scalar.as_ptr() as usize as u64;
        let state = fingerprint();
        let snapshot_ptr = journal.backing.as_ref().unwrap().as_ptr() as usize as u64;
        let snapshot_bytes = journal.layout.bytes();
        unsafe { runtime.htod(&vec![0xcc; snapshot_bytes], snapshot_ptr) }.unwrap();
        assert!(unsafe { journal.checkpoint(main_ptr, index_ptr, 13, 4, 17, 23) }.is_err());
        let mut snapshot_data = vec![0; snapshot_bytes];
        unsafe { runtime.dtoh(&mut snapshot_data, snapshot_ptr) }.unwrap();
        assert!(
            snapshot_data.iter().all(|&byte| byte == 0xcc),
            "capacity refusal cannot alter even part of the saved snapshot"
        );
        assert_eq!(fingerprint(), state);
        let raw_capture_exclusion =
            onnx_runtime_cuda_memory::capture_gate::CaptureExclusion::acquire();
        runtime
            .stream()
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .unwrap();
        let counts = runtime.transfer_counts();
        assert!(
            unsafe { journal.checkpoint(main_ptr, index_ptr, 8, 4, 17, 23) }
                .unwrap_err()
                .to_string()
                .contains("capture")
        );
        assert_eq!(runtime.transfer_counts(), counts);
        assert_eq!(
            runtime.stream_capture_status().unwrap(),
            cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
        );
        runtime.test_end_unregistered_graph_capture().unwrap();
        drop(raw_capture_exclusion);
        for _ in 0..3 {
            let mut checkpoint =
                unsafe { journal.checkpoint(main_ptr, index_ptr, 8, 4, u64::MAX, 23) }.unwrap();
            unsafe {
                runtime.htod(&[3u8; 16], main_ptr).unwrap();
                runtime.htod(&[4u8; 8], index_ptr).unwrap();
                runtime.htod(&99i64.to_ne_bytes(), scalar_ptr).unwrap();
            }
            let before_rollback = metrics.rollback_count();
            let counts = runtime.transfer_counts();
            assert!(unsafe { checkpoint.restore_prefix(u64::MAX, 23, Some(scalar_ptr)) }.is_err());
            assert!(unsafe { checkpoint.restore_prefix(8, 24, Some(scalar_ptr)) }.is_err());
            assert_eq!(
                runtime.transfer_counts(),
                counts,
                "preflight cannot copy part of a failed restore"
            );
            assert_eq!(metrics.rollback_count(), before_rollback);
            let mut untouched_main = [0u8; 16];
            let mut untouched_index = [0u8; 8];
            let mut untouched_scalar = [0u8; 8];
            unsafe {
                runtime.dtoh(&mut untouched_main, main_ptr).unwrap();
                runtime.dtoh(&mut untouched_index, index_ptr).unwrap();
                runtime.dtoh(&mut untouched_scalar, scalar_ptr).unwrap();
            }
            assert_eq!(
                untouched_main, [3; 16],
                "failed preflight must not partially restore main carry"
            );
            assert_eq!(
                untouched_index, [4; 8],
                "failed preflight must not partially restore index carry"
            );
            assert_eq!(i64::from_ne_bytes(untouched_scalar), 99);
            let counts = runtime.transfer_counts();
            let raw_capture_exclusion =
                onnx_runtime_cuda_memory::capture_gate::CaptureExclusion::acquire();
            runtime
                .stream()
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
                .unwrap();
            assert!(
                unsafe { checkpoint.restore_prefix(8, 23, Some(scalar_ptr)) }
                    .unwrap_err()
                    .to_string()
                    .contains("capture")
            );
            assert_eq!(runtime.transfer_counts(), counts);
            assert_eq!(
                runtime.stream_capture_status().unwrap(),
                cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
            );
            runtime.test_end_unregistered_graph_capture().unwrap();
            drop(raw_capture_exclusion);
            unsafe { checkpoint.restore_prefix(8, 23, Some(scalar_ptr)) }.unwrap();
            assert_eq!(metrics.rollback_count(), before_rollback + 1);
            let mut actual_main = [0u8; 16];
            let mut actual_index = [0u8; 8];
            unsafe {
                runtime.dtoh(&mut actual_main, main_ptr).unwrap();
                runtime.dtoh(&mut actual_index, index_ptr).unwrap();
            }
            assert_eq!(actual_main[..8], initial_main[..8]);
            assert_eq!(actual_main[8..], [3u8; 8]);
            assert_eq!(actual_index[..4], initial_index[..4]);
            assert_eq!(actual_index[4..], [4u8; 4]);
            let mut scalar_bytes = [0u8; 8];
            unsafe { runtime.dtoh(&mut scalar_bytes, scalar_ptr) }.unwrap();
            assert_eq!(i64::from_ne_bytes(scalar_bytes), 8);
            assert_eq!(
                fingerprint(),
                state,
                "native allocation identities and charges remain unchanged"
            );
        }
        let counts = runtime.transfer_counts();
        {
            let mut zero = CsaCheckpointJournal::new(&ep, 4, 0, 0, metrics.clone()).unwrap();
            assert!(zero.backing.is_none());
            let mut empty = unsafe { zero.checkpoint(0, 0, 0, 0, 0, 1) }.unwrap();
            unsafe { empty.restore_prefix(0, 1, None) }.unwrap();
        }
        assert_eq!(runtime.transfer_counts(), counts);
        assert_eq!(fingerprint(), state);
        ep.deallocate(main).unwrap();
        ep.deallocate(index).unwrap();
        ep.deallocate(scalar).unwrap();
        ep.wait_for_deferred_releases().unwrap();
        journal.release().unwrap();
        assert_eq!(
            settlement.wait(Duration::from_secs(30)),
            AllocationSettlementStatus::Released
        );
        assert_eq!(settlement.charged_bytes(), 0);
        let fallback = CsaCheckpointJournal::new(&ep, 4, 8, 4, metrics).unwrap();
        let settlement = fallback
            .backing
            .as_ref()
            .unwrap()
            .managed_settlement_wait()
            .unwrap();
        drop(fallback);
        assert_eq!(
            settlement.wait(Duration::from_secs(30)),
            AllocationSettlementStatus::Released
        );
    }
}
