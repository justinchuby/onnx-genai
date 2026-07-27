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

use crate::decode_contract::{KvNamingConvention, kv_suffix, name_contains_present_key_value};
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
use io::{KvPair, detect_static_cache, infer_kv_pairs};

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
fn to_f32_logits(logits: &Value) -> Result<Value> {
    let shape = logits.shape().to_vec();
    if logits.dtype() == DataType::Float32 {
        return Value::from_vec_f32(logits.to_vec_f32()?, &shape);
    }
    Value::from_vec_f32(logits.to_vec_f32_lossy()?, &shape)
}

/// Gather selected batch rows of a `[B, S, vocab]` Float32 logits tensor into a
/// compact `[rows.len(), S, vocab]` tensor, preserving the given row order.
fn gather_logits_rows(logits: &Value, rows: &[usize]) -> Result<Value> {
    if logits.dtype() != DataType::Float32 || logits.shape().len() != 3 {
        return Err(OrtError::InvalidArgument(format!(
            "expected Float32 logits [B, S, V], got {:?} {:?}",
            logits.dtype(),
            logits.shape()
        )));
    }
    let shape = logits.shape();
    let batch = shape[0] as usize;
    let seq_len = shape[1] as usize;
    let vocab = shape[2] as usize;
    let data = logits.to_vec_f32()?;
    let row_stride = seq_len * vocab;
    let mut gathered = Vec::with_capacity(rows.len() * row_stride);
    for &row in rows {
        if row >= batch {
            return Err(OrtError::InvalidArgument(format!(
                "gather row {row} out of range for batch {batch}"
            )));
        }
        let start = row * row_stride;
        gathered.extend_from_slice(&data[start..start + row_stride]);
    }
    Value::from_vec_f32(gathered, &[rows.len() as i64, seq_len as i64, vocab as i64])
}

fn empty_past_value(info: &TensorInfo) -> Result<Value> {
    let seq_axis = info.shape.len() - 2;
    let mut shape = Vec::with_capacity(info.shape.len());
    for (axis, &dim) in info.shape.iter().enumerate() {
        let value = if axis == 0 {
            1
        } else if axis == seq_axis {
            0
        } else if dim > 0 {
            dim
        } else {
            return Err(OrtError::InvalidArgument(format!(
                "cannot infer static dimension {axis} for empty KV input '{}'",
                info.name
            )));
        };
        shape.push(value);
    }
    Value::empty(&shape, info.dtype)
}

fn is_logits_output(name: &str) -> bool {
    name.to_ascii_lowercase().contains("logits")
}

/// Copy a device-resident `logits [1, 1, vocab]` row back into a host CPU
/// [`Value`] of the same dtype, for the non-greedy path that still consumes the
/// full vocabulary. This mirrors ORT's implicit device->host logits copy that
/// the on-device path otherwise skips.
#[cfg(feature = "cuda")]
fn device_logits_to_host_value(
    device_sampler: &dyn DeviceSampler,
    dtype: DataType,
    dev_ptr: usize,
    vocab: usize,
) -> Result<Value> {
    let host = Value::empty(&[1, 1, vocab as i64], dtype)?;
    let nbytes = vocab
        .checked_mul(dtype.size_of())
        .ok_or_else(|| OrtError::InvalidArgument("logits byte size overflow".into()))?;
    let base = host.data_ptr_addr()? as *mut u8;
    // SAFETY: `host` is a freshly-allocated CPU tensor holding exactly `nbytes`
    // bytes; the slice aliases only that storage for the duration of the copy.
    let dst = unsafe { std::slice::from_raw_parts_mut(base, nbytes) };
    device_sampler.copy_row_to_host(dtype, dev_ptr, vocab, dst)?;
    Ok(host)
}

/// Copy an OrtValue's tensor data onto host-owned Rust buffers, producing a
/// new, session-independent CPU [`Value`]. Used to hand a KV cache between two
/// [`DecodeSession`]s (e.g. Metal-EP prefill → CPU-EP decode).
fn clone_value_to_owned(value: &Value) -> Result<Value> {
    let shape = value.shape().to_vec();
    match value.dtype() {
        DataType::Float32 => Value::from_vec_f32(value.to_vec_f32()?, &shape),
        DataType::Float16 => Value::from_vec_f16_bits(value.to_vec_f16_bits()?, &shape),
        DataType::BFloat16 => Value::from_vec_bf16_bits(value.to_vec_bf16_bits()?, &shape),
        dtype => Err(OrtError::InvalidArgument(format!(
            "cannot export/clone KV tensor with dtype {dtype:?}"
        ))),
    }
}

fn allocate_static_cache_buffers(
    batch_size: i64,
    pairs: &[StaticCachePair],
) -> Result<Vec<StaticCacheBuffer>> {
    if batch_size <= 0 {
        return Err(OrtError::InvalidArgument(format!(
            "batch_size must be positive, got {batch_size}"
        )));
    }
    let mut buffers = Vec::with_capacity(pairs.len() * 2);
    for pair in pairs {
        for (input, output) in [
            (&pair.key_input, &pair.key_output),
            (&pair.value_input, &pair.value_output),
        ] {
            let mut shape = input.shape.clone();
            shape[0] = batch_size;
            buffers.push(StaticCacheBuffer {
                input_name: input.name.clone(),
                output_name: output.clone(),
                current: Arc::new(zeroed_value(&shape, input.dtype)?),
                alternate: None,
            });
        }
    }
    Ok(buffers)
}

fn zeroed_value(shape: &[i64], dtype: DataType) -> Result<Value> {
    let numel = shape.iter().try_fold(1usize, |acc, &dim| {
        if dim < 0 {
            return Err(OrtError::InvalidArgument(format!(
                "cannot allocate tensor with dynamic shape {shape:?}"
            )));
        }
        acc.checked_mul(dim as usize)
            .ok_or_else(|| OrtError::InvalidArgument(format!("tensor shape too large: {shape:?}")))
    })?;
    match dtype {
        DataType::Float32 => Value::from_vec_f32(vec![0.0; numel], shape),
        DataType::Float16 => Value::from_vec_f16_bits(vec![0; numel], shape),
        DataType::BFloat16 => Value::from_vec_bf16_bits(vec![0; numel], shape),
        dtype => Err(OrtError::InvalidArgument(format!(
            "cannot allocate static-cache tensor with dtype {dtype:?}"
        ))),
    }
}
