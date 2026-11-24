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
//! [`restore_prefix`]: CsaCheckpointJournal::restore_prefix
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
use onnx_runtime_ep_api::{EpError, Result};

use crate::runtime::CudaRuntime;

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

/// Opaque, stream-ordered CSA checkpoint (D6). Holds the five logical cursors, a
/// generation stamp for identity validation, and the byte extents of the carry
/// snapshot stored in the owning [`CsaCheckpointJournal`]'s reserved scratch.
#[derive(Clone, Copy, Debug)]
pub struct CsaCheckpoint {
    cursors: CsaCursors,
    generation: u64,
    main_carry_bytes: usize,
    index_carry_bytes: usize,
}

impl CsaCheckpoint {
    /// The five logical cursors captured at checkpoint time.
    pub fn cursors(&self) -> CsaCursors {
        self.cursors
    }

    /// The sequence/generation identity stamp validated on restore.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Backend-owned journal of the bounded CSA carry state (D6). Reserves two
/// stable-address device snapshot buffers once (main + index carry) so
/// checkpoint/restore never allocate per speculative step; the physical
/// addresses stay pinned across capture/replay and rollback.
pub struct CsaCheckpointJournal {
    runtime: Arc<CudaRuntime>,
    ratio: u64,
    main_snapshot: CUdeviceptr,
    index_snapshot: CUdeviceptr,
    main_capacity: usize,
    index_capacity: usize,
    metrics: Arc<CsaMetrics>,
}

impl CsaCheckpointJournal {
    /// Reserve the fixed-capacity carry snapshot buffers. `main_carry_bytes` and
    /// `index_carry_bytes` bound the largest carry region a checkpoint may span.
    pub fn new(
        runtime: Arc<CudaRuntime>,
        ratio: u64,
        main_carry_bytes: usize,
        index_carry_bytes: usize,
        metrics: Arc<CsaMetrics>,
    ) -> Result<Self> {
        let main_snapshot = runtime.alloc_raw(main_carry_bytes.max(1))?;
        let index_snapshot = match runtime.alloc_raw(index_carry_bytes.max(1)) {
            Ok(ptr) => ptr,
            Err(error) => {
                // SAFETY: `main_snapshot` came from this runtime and has not escaped.
                let _ = unsafe { runtime.free_raw(main_snapshot) };
                return Err(error);
            }
        };
        Ok(Self {
            runtime,
            ratio: ratio.max(1),
            main_snapshot,
            index_snapshot,
            main_capacity: main_carry_bytes.max(1),
            index_capacity: index_carry_bytes.max(1),
            metrics,
        })
    }

    /// Snapshot the bounded active carry state at sequence position `seq_cursor`
    /// into the reserved scratch (no recompression), stamping it with
    /// `generation` for identity validation on restore. The carry copies are
    /// device→device and stream-ordered.
    ///
    /// # Safety
    /// `main_carry` / `index_carry` are live device allocations of at least
    /// `main_carry_bytes` / `index_carry_bytes` bytes.
    pub unsafe fn checkpoint(
        &self,
        main_carry: CUdeviceptr,
        index_carry: CUdeviceptr,
        main_carry_bytes: usize,
        index_carry_bytes: usize,
        seq_cursor: u64,
        generation: u64,
    ) -> Result<CsaCheckpoint> {
        if main_carry_bytes > self.main_capacity || index_carry_bytes > self.index_capacity {
            return Err(EpError::KernelFailed(format!(
                "CSA checkpoint: carry ({main_carry_bytes},{index_carry_bytes}) exceeds reserved \
                 snapshot capacity ({},{})",
                self.main_capacity, self.index_capacity
            )));
        }
        if main_carry_bytes > 0 {
            // SAFETY: both endpoints cover `main_carry_bytes` per the contract.
            unsafe {
                self.runtime
                    .dtod(main_carry, self.main_snapshot, main_carry_bytes)?;
            }
        }
        if index_carry_bytes > 0 {
            // SAFETY: both endpoints cover `index_carry_bytes` per the contract.
            unsafe {
                self.runtime
                    .dtod(index_carry, self.index_snapshot, index_carry_bytes)?;
            }
        }
        Ok(CsaCheckpoint {
            cursors: CsaCursors::from_sequence(seq_cursor, self.ratio),
            generation,
            main_carry_bytes,
            index_carry_bytes,
        })
    }

    /// Roll the carry buffers and cursors back to the accepted prefix (D6). The
    /// carry is restored from the checkpoint snapshot (stream-ordered device
    /// copy); when `seq_scalar` is supplied the device `total_sequence_length`
    /// scalar is reset to `accepted`. `accepted` must lie within the committed
    /// checkpoint (`accepted <= checkpoint.seq_cursor`) — accepting drafted
    /// tokens *beyond* the checkpoint is the engine's replay responsibility.
    ///
    /// # Safety
    /// `main_carry` / `index_carry` are the same live carry allocations passed to
    /// [`checkpoint`], and `seq_scalar` (if `Some`) is a live 8-byte device
    /// scalar.
    ///
    /// [`checkpoint`]: CsaCheckpointJournal::checkpoint
    pub unsafe fn restore_prefix(
        &self,
        checkpoint: &CsaCheckpoint,
        accepted: u64,
        generation: u64,
        main_carry: CUdeviceptr,
        index_carry: CUdeviceptr,
        seq_scalar: Option<CUdeviceptr>,
    ) -> Result<CsaCursors> {
        if generation != checkpoint.generation {
            return Err(EpError::KernelFailed(format!(
                "CSA restore: generation {generation} does not match checkpoint {}",
                checkpoint.generation
            )));
        }
        if accepted > checkpoint.cursors.seq_cursor {
            return Err(EpError::KernelFailed(format!(
                "CSA restore: accepted prefix {accepted} exceeds committed checkpoint {}",
                checkpoint.cursors.seq_cursor
            )));
        }
        if checkpoint.main_carry_bytes > 0 {
            // SAFETY: snapshot and carry both cover `main_carry_bytes`.
            unsafe {
                self.runtime
                    .dtod(self.main_snapshot, main_carry, checkpoint.main_carry_bytes)?;
            }
        }
        if checkpoint.index_carry_bytes > 0 {
            // SAFETY: snapshot and carry both cover `index_carry_bytes`.
            unsafe {
                self.runtime.dtod(
                    self.index_snapshot,
                    index_carry,
                    checkpoint.index_carry_bytes,
                )?;
            }
        }
        if let Some(scalar) = seq_scalar {
            let bytes = (accepted as i64).to_ne_bytes();
            // SAFETY: `scalar` is a live 8-byte device allocation per the contract.
            unsafe {
                self.runtime.htod(&bytes, scalar)?;
            }
        }
        self.metrics.record_rollback();
        Ok(CsaCursors::from_sequence(accepted, self.ratio))
    }

    /// The shared telemetry surface this journal accumulates rollbacks into.
    pub fn metrics(&self) -> &Arc<CsaMetrics> {
        &self.metrics
    }
}

impl Drop for CsaCheckpointJournal {
    fn drop(&mut self) {
        // SAFETY: this journal exclusively owns both reserved snapshot buffers.
        let _ = unsafe { self.runtime.free_raw(self.index_snapshot) };
        let _ = unsafe { self.runtime.free_raw(self.main_snapshot) };
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
