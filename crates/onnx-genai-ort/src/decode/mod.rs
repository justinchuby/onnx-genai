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
    /// Explicit caller-selected `past_present_share_buffer` mode. One max-length
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
    /// Explicitly enable or disable the low-level shared-buffer ABI.
    ///
    /// `None` defaults to functional past/present rebinding; artifact custom
    /// metadata never changes execution mode.
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

/// KV-representation-agnostic operations a continuous-batch manager needs from a
/// batched decode session.
///
/// Both [`BatchedStaticCacheDecodeSession`] (TensorScatter static cache) and
/// [`BatchedSharedBufferDecodeSession`] (past/present share-buffer GQA) implement
/// this so the same `ContinuousBatchManager` can drive either backend.
///
/// Logits returned by `step_select`/`step_active` are `Float32 [batch, 1, vocab]`;
/// `step_select` returns one row per physical batch slot (physical-row indexed),
/// while `step_active` returns one row per active row in [`Self::active_rows`]
/// order.
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
    /// is active, returning physical-row-indexed `[B, 1, vocab]` logits.
    fn step_select(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
        advance_rows: &[bool],
    ) -> Result<Value>;
    /// Advance one token for every active row, returning `[active, 1, vocab]`
    /// logits ordered by [`Self::active_rows`].
    fn step_active(&mut self, next_token_ids: &[i64], position_ids: &[i64]) -> Result<Value>;
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
