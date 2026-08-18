//! Low-level incremental model execution built on ORT IoBinding.
//!
//! This module owns one forward pass at a time: raw tensor I/O, IoBinding, and
//! runtime-owned KV buffer state including cursors and rewind. It deliberately
//! does not select tokens, apply sampling or constraints, enforce stop
//! conditions, or drive a multi-step generation loop. Those policies belong to
//! `onnx-genai-engine`, behind its `DecodeBackend` adapter seam.

#![allow(clippy::arc_with_non_send_sync)]
// ORT Values are session-owned handles. These Arcs provide shared ownership inside
// one decode session; they are not used to move Values across threads.

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
use crate::device_sampler::{CudaSampler, DeviceSampler};
use crate::{
    DataType, IoBinding, MemoryInfo, OrtError, Result, RunPhaseError, Session, TensorInfo, Value,
};

/// Parameters for one device sampling step.
///
/// Mirrors the device-portable subset of the engine's generation options.
/// History-dependent processing (penalties, constraints, stop sequences) is
/// applied host-side before/around this and is intentionally absent here.
///
/// `greedy` short-circuits to argmax and ignores every filter, since top-k /
/// top-p / min-p / temperature are all monotonic and never change the argmax.
///
/// Defined unconditionally (not behind `cuda`) so the engine can construct it
/// regardless of which compute backends are compiled in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceSampleParams {
    /// Softmax temperature. `<= 0.0` or `1.0` means "no scaling".
    pub temperature: f32,
    /// Keep only the `top_k` highest-probability tokens. `0` disables.
    pub top_k: usize,
    /// Nucleus threshold in `(0, 1]`. `>= 1.0` disables.
    pub top_p: f32,
    /// Minimum probability relative to the max (`min_p * p_max`). `<= 0.0` disables.
    pub min_p: f32,
    /// Select the argmax token (ignore every filter and the RNG).
    pub greedy: bool,
    /// Uniform draw in `[0, 1)` used for the categorical pick when `!greedy`.
    pub rng_value: f32,
}

impl DeviceSampleParams {
    /// Pure greedy selection: argmax, no filters, no RNG.
    pub fn greedy() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            greedy: true,
            rng_value: 0.0,
        }
    }
}

/// KV binding strategy selected for a decode session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeKvMode {
    /// ORT allocates `present.*` outputs; those OrtValues are rebound as next
    /// step's `past_key_values.*` inputs. No Rust-side KV copy is performed.
    ZeroCopyRebind,
    /// Caller/model-declared `past_present_share_buffer` mode. One max-length
    /// OrtValue per KV tensor is bound as both past input and present output.
    SharedBuffer,
}

/// Static-cache output binding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticCacheBindingMode {
    /// Bind `updated_key_cache.N` / `updated_value_cache.N` to the same
    /// runtime-owned OrtValue as the corresponding input cache.
    InPlaceAlias,
    /// Bind outputs to a second runtime-owned buffer and swap handles after a
    /// run. This is the fallback if an ORT build rejects input/output aliasing.
    HandleSwap,
}

/// Introspected static-cache model signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticCacheSignature {
    pub layers: usize,
    pub max_len: usize,
    pub kv_dim: usize,
    pub dtype: DataType,
    pub has_position_ids: bool,
}

/// Snapshot of a runtime-owned static-cache KV buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticCacheBufferInfo {
    pub input_name: String,
    pub output_name: String,
    pub shape: Vec<i64>,
    pub dtype: DataType,
    pub data_ptr: usize,
    pub numel: usize,
}

/// Options for [`StaticCacheDecodeSession`].
#[derive(Debug, Clone)]
pub struct StaticCacheDecodeOptions {
    pub batch_size: i64,
}

impl Default for StaticCacheDecodeOptions {
    fn default() -> Self {
        Self { batch_size: 1 }
    }
}

/// Options for [`DecodeSession`].
#[derive(Debug, Clone)]
pub struct DecodeSessionOptions {
    /// Batch size for empty/shared KV buffers. Generation currently uses 1.
    pub batch_size: i64,
    /// Maximum logical context length. Required for shared-buffer mode.
    pub max_length: Option<usize>,
    /// Override ORT custom metadata detection of `past_present_share_buffer`.
    pub past_present_share_buffer: Option<bool>,
}

impl Default for DecodeSessionOptions {
    fn default() -> Self {
        Self {
            batch_size: 1,
            max_length: None,
            past_present_share_buffer: None,
        }
    }
}

mod io;
use io::{KvPair, StaticCacheAbi, StaticCacheInputRole, detect_static_cache, infer_kv_pairs};

mod static_cache;
pub use static_cache::{BatchedStaticCacheDecodeSession, StaticCacheDecodeSession};
use static_cache::{StaticCacheBuffer, StaticCachePair};

mod dynamic;
pub use dynamic::DecodeSession;

mod kv_growth;
use kv_growth::{GrowDevice, grow_kv_value, kv_capacity_bucket};

/// Logits produced by one batched decode step, in a form every backend can
/// return without an extra per-step copy of the ~`[B, vocab]` logits.
///
/// The two batched backends produce logits in physically different places:
/// the ORT sessions hand back a host `[B, 1, vocab]` `Value`, while the native
/// CUDA seam (`decode_greedy_batch_ragged_logits`) reads the device logits into
/// per-row host `Vec<f32>` itself. Typing the trait around ORT `Value` alone
/// would force the native path to rebuild a `Value` (pulling ORT into the native
/// logits path) or the ORT path to eagerly demux; either way one backend pays a
/// copy purely to satisfy the signature. This enum lets each backend return what
/// it already has and defers the per-row extraction to [`Self::take_row`]:
///
/// - `Ort` wraps the host `Value` unchanged; [`Self::take_row`] copies one row
///   out exactly as the previous `row_logits(&Value, ..)` demux did, so the ORT
///   path keeps its behaviour and its cost.
/// - `HostRows` carries the per-row owned host logits the native seam already
///   allocated; [`Self::take_row`] *moves* a row out, adding no copy on top of
///   the single device→host read the throughput path already pays.
pub enum BatchStepLogits {
    /// Host `[B, 1, vocab]` ORT logits, indexed as the producing call documents
    /// (`step_select`: physical-row; `step_active`: active-row order).
    Ort(Value),
    /// Pre-demuxed per-row host logits in the same index order the producing
    /// call documents. `None` marks a row already taken by [`take_row`].
    ///
    /// [`take_row`]: BatchStepLogits::take_row
    HostRows(Vec<Option<Vec<f32>>>),
    /// Logits that are still resident in device memory for this step. Instead of
    /// eagerly copying the whole `[B, vocab]` batch to the host, the manager
    /// routes each row through [`DeviceRowLogits`]: device-portable rows are
    /// sampled entirely on the device (only a 4-byte token id crosses the bus)
    /// and only the rows that genuinely need the full vocabulary — history
    /// dependent processors, logprobs — are copied to the host on demand. This
    /// is the seam that lets a mixed batch pay `(host-required rows) x vocab x 4`
    /// bytes instead of `(all rows) x vocab x 4`.
    Device(Box<dyn DeviceRowLogits>),
}

/// One decode step's logits, still resident in device memory, exposing per-row
/// device selection.
///
/// A continuous-batch manager consults this to route each row independently:
/// [`Self::sample_row`] selects a token entirely on the device (copying back
/// only the 4-byte token id) for device-portable rows, while
/// [`Self::copy_row_to_host`] copies one row's full `[vocab]` logits to the host
/// for the rows that need host-side processors or logprobs. This mirrors the
/// single-row [`crate::device_sampler::DeviceSampler`] shape already used by the
/// captured-decode fast path, lifted to the batched seam.
pub trait DeviceRowLogits: Send {
    /// Number of rows carried by this step's buffer.
    fn rows(&self) -> usize;
    /// Vocabulary length of each row.
    fn vocab(&self) -> usize;
    /// Select one token id for `row` entirely on the device, applying `params`.
    /// Only the 4-byte token id is copied back to the host.
    fn sample_row(&mut self, row: usize, params: &DeviceSampleParams) -> Result<u32>;
    /// Copy `row`'s full `[vocab]` logits to the host. Paid only for rows that
    /// need the whole vocabulary host-side.
    fn copy_row_to_host(&mut self, row: usize) -> Result<Vec<f32>>;
}

impl BatchStepLogits {
    /// Take one row's `[vocab]` logits out of the step.
    ///
    /// For `Ort` this copies the row out of the shared host `Value` (the
    /// pre-existing demux cost paid by [`BatchedStaticCacheDecodeSession::row_logits`]);
    /// for `HostRows` it moves the already-owned row out with no copy. Each row
    /// may be taken at most once — a second take of the same `HostRows` row is an
    /// error rather than a silent second copy.
    pub fn take_row(&mut self, row: usize, seq_index: usize) -> Result<Vec<f32>> {
        match self {
            BatchStepLogits::Ort(value) => {
                BatchedStaticCacheDecodeSession::row_logits(value, row, seq_index)
            }
            BatchStepLogits::HostRows(rows) => {
                rows.get_mut(row).and_then(Option::take).ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "batch step logits row {row} already taken or out of range \
                         (have {} rows)",
                        rows.len()
                    ))
                })
            }
            BatchStepLogits::Device(device) => device.copy_row_to_host(row),
        }
    }
}

/// KV-representation-agnostic operations a continuous-batch manager needs from a
/// batched decode session.
///
/// Both [`BatchedStaticCacheDecodeSession`] (TensorScatter static cache) and
/// [`BatchedSharedBufferDecodeSession`] (past/present share-buffer GQA) implement
/// this so the same `ContinuousBatchManager` can drive either backend; the native
/// CUDA backend implements it in `onnx-genai-engine` over its host-logits seam.
///
/// Logits returned by `step_select`/`step_active` are per-row `Float32 [vocab]`
/// rows carried by [`BatchStepLogits`]; `step_select` indexes them by physical
/// batch slot (physical-row indexed), while `step_active` indexes them by active
/// row in [`Self::active_rows`] order. The manager extracts a row with
/// [`BatchStepLogits::take_row`], which is a copy on the ORT path and a move on
/// the native path — neither backend is charged an extra logits copy to satisfy
/// the shared signature.
pub trait BatchedDecodeSession<'a> {
    /// Fixed number of physical batch rows.
    fn batch_size(&self) -> usize;
    /// Maximum logical KV length (buffer capacity in tokens).
    fn max_len(&self) -> usize;
    /// Current logical token length of a row.
    fn row_len(&self, row: usize) -> Result<usize>;
    /// Active logical rows in the order `step_active` logits are returned.
    fn active_rows(&self) -> Vec<usize>;
    /// Mark a row inactive (its slot may be recycled by `assign_row`).
    fn deactivate_row(&mut self, row: usize) -> Result<()>;
    /// Reset a row's cursor to zero and mark it active for a new sequence.
    fn assign_row(&mut self, row: usize) -> Result<()>;
    /// Advance one token for rows where `advance_rows[row]` is true and the row
    /// is active, returning physical-row-indexed `[vocab]` logits per slot.
    fn step_select(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
        advance_rows: &[bool],
    ) -> Result<BatchStepLogits>;
    /// Advance one token for every active row, returning `[vocab]` logits per
    /// active row ordered by [`Self::active_rows`].
    fn step_active(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
    ) -> Result<BatchStepLogits>;
    /// Cumulative device→host logits transfer cost, for backends that read the
    /// logits back to the host each step (the native host-logits seam). The ORT
    /// backends keep logits host-side already and report `None`, so a caller can
    /// tell the manager's honest D2H cost from a backend that pays none.
    fn logits_d2h_stats(&self) -> Option<LogitsD2hStats> {
        None
    }
}

/// Cumulative device→host logits transfer cost reported by a batched decode
/// backend that round-trips logits to the host each step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogitsD2hStats {
    /// Total bytes transferred device→host for full per-row vocabularies. On the
    /// native host-logits seam this is every row every step; under the batched
    /// per-row router it is only the rows that needed the whole vocabulary
    /// (`host-required rows x vocab x 4`).
    pub bytes: u128,
    /// Total wall time of those transfers.
    pub time: std::time::Duration,
    /// Number of steps that performed a logits read.
    pub steps: u64,
    /// Rows whose full vocabulary was copied to the host (the expensive D2H).
    pub rows_host_copied: u64,
    /// Rows selected entirely on the device, copying back only the token id.
    pub rows_device_sampled: u64,
    /// Bytes transferred device→host for device-sampled token ids (4 per row).
    pub token_id_bytes: u128,
}

mod shared_batch;
pub use shared_batch::{BatchedSharedBufferDecodeSession, SharedBufferBatchOptions};

/// Convert a logits `Value` to a contiguous Float32 `[B, S, vocab]` tensor.
mod tensor;
#[cfg(feature = "cuda")]
use tensor::device_logits_to_host_value;
use tensor::{
    allocate_static_cache_buffers, clone_value_to_owned, empty_past_value, gather_logits_rows,
    to_f32_logits, zeroed_value,
};
