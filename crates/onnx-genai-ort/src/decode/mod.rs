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

#[derive(Debug, Clone)]
struct KvPair {
    past: String,
    present: String,
    input: TensorInfo,
    seq_axis: usize,
}

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

/// Options for [`BatchedSharedBufferDecodeSession`].
#[derive(Debug, Clone)]
pub struct SharedBufferBatchOptions {
    /// Number of physical batch rows (concurrent sequences).
    pub batch_size: i64,
    /// Fixed KV buffer capacity in tokens.
    pub max_len: usize,
}

/// Batched stateful decode runner for shared-buffer (past/present) GQA models.
///
/// Unlike the static-cache path, share-buffer models carry no explicit
/// `write_indices`/`nonpad_kv_seqlen` inputs: the model derives each row's valid
/// KV length (`seqlens_k`) and the shared `total_sequence_length` from the
/// `attention_mask`, and `GroupQueryAttention` writes each row's new present KV
/// in place at that row's own offset. Rows of different lengths therefore share
/// one batched Run: a `[batch, W]` attention mask supplies each row its own
/// leading-ones prefix (`row_len + 1` ones), and the KV buffers are allocated
/// once as `[batch, kv_heads, max_len, head_dim]` and bound in place as both
/// `past_key_values.*` inputs and `present.*` outputs.
///
/// Inactive/non-advancing rows still run (their scratch write lands in the
/// not-yet-valid slot at their own offset and is later overwritten or ignored),
/// keeping the batch a fixed `batch_size` every step.
pub struct BatchedSharedBufferDecodeSession<'a> {
    session: &'a Session,
    binding: IoBinding,
    kv_pairs: Vec<KvPair>,
    kv_buffers: HashMap<String, Arc<Value>>,
    kv_allocator: Option<crate::Allocator>,
    batch_size: usize,
    max_len: usize,
    row_lens: Vec<usize>,
    active: Vec<bool>,
    has_position_ids: bool,
}

impl<'a> BatchedSharedBufferDecodeSession<'a> {
    /// Create a batched share-buffer decode session with all rows active at
    /// cursor 0. KV buffers are allocated once as `[batch, kv_heads, max_len,
    /// head_dim]` on the session's device allocator when available.
    pub fn new(session: &'a Session, options: SharedBufferBatchOptions) -> Result<Self> {
        let batch_size = usize::try_from(options.batch_size).map_err(|_| {
            OrtError::InvalidArgument(format!(
                "batch_size must be positive, got {}",
                options.batch_size
            ))
        })?;
        if batch_size == 0 {
            return Err(OrtError::InvalidArgument(
                "batch_size must be positive".into(),
            ));
        }
        if options.max_len == 0 {
            return Err(OrtError::InvalidArgument(
                "shared-buffer batch requires max_len > 0".into(),
            ));
        }
        let kv_pairs = infer_kv_pairs(session)?;
        if kv_pairs.is_empty() {
            return Err(OrtError::InvalidArgument(
                "model exposes no past/present KV pairs for shared-buffer batching".into(),
            ));
        }
        let has_position_ids = session.inputs().iter().any(|input| {
            let lower = input.name.to_ascii_lowercase();
            lower == "position_ids" || lower.ends_with(".position_ids")
        });
        let has_attention_mask = session.inputs().iter().any(|input| {
            let lower = input.name.to_ascii_lowercase();
            lower == "attention_mask" || lower.ends_with(".attention_mask")
        });
        if !has_attention_mask {
            return Err(OrtError::InvalidArgument(
                "shared-buffer batching requires an attention_mask input to signal per-row \
                 sequence lengths"
                    .into(),
            ));
        }
        let mut this = Self {
            session,
            binding: IoBinding::new(session)?,
            kv_pairs,
            kv_buffers: HashMap::new(),
            kv_allocator: None,
            batch_size,
            max_len: options.max_len,
            row_lens: vec![0; batch_size],
            active: vec![true; batch_size],
            has_position_ids,
        };
        this.allocate_shared_buffers()?;
        Ok(this)
    }

    /// Fixed number of physical batch rows.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// KV buffer capacity in tokens.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Current logical token length of a row.
    pub fn row_len(&self, row: usize) -> Result<usize> {
        self.check_row(row)?;
        Ok(self.row_lens[row])
    }

    /// Whether a row currently participates in decode steps.
    pub fn is_active(&self, row: usize) -> Result<bool> {
        self.check_row(row)?;
        Ok(self.active[row])
    }

    /// Active logical rows in ascending physical order.
    pub fn active_rows(&self) -> Vec<usize> {
        (0..self.batch_size)
            .filter(|&row| self.active[row])
            .collect()
    }

    /// Mark a row inactive; its slot may be recycled by [`Self::assign_row`].
    pub fn deactivate_row(&mut self, row: usize) -> Result<()> {
        self.check_row(row)?;
        self.active[row] = false;
        Ok(())
    }

    /// Reset a row's cursor and mark it active for a new sequence. Stale KV in
    /// the row's slice is left as-is; later attention masks exclude it and future
    /// writes overwrite it.
    pub fn assign_row(&mut self, row: usize) -> Result<()> {
        self.check_row(row)?;
        self.row_lens[row] = 0;
        self.active[row] = true;
        Ok(())
    }

    /// Alias for [`Self::assign_row`] to match the continuous-batch admit call.
    pub fn admit_row(&mut self, row: usize) -> Result<()> {
        self.assign_row(row)
    }

    /// Advance one token per row where `advance_rows[row]` is true and the row is
    /// active, returning physical-row-indexed `[batch, 1, vocab]` Float32 logits.
    pub fn step_select(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
        advance_rows: &[bool],
    ) -> Result<Value> {
        if advance_rows.len() != self.batch_size {
            return Err(OrtError::InvalidArgument(format!(
                "advance_rows length {} does not match batch {}",
                advance_rows.len(),
                self.batch_size
            )));
        }
        let advances = (0..self.batch_size)
            .map(|row| self.active[row] && advance_rows[row])
            .collect::<Vec<_>>();
        self.run_batch(next_token_ids, position_ids, &advances)
    }

    /// Advance one token per active row, returning `[active, 1, vocab]` Float32
    /// logits ordered by [`Self::active_rows`]. `next_token_ids`/`position_ids`
    /// are indexed in active-row order.
    pub fn step_active(&mut self, next_token_ids: &[i64], position_ids: &[i64]) -> Result<Value> {
        let rows = self.active_rows();
        if rows.is_empty() {
            return Err(OrtError::InvalidArgument(
                "active-only shared-buffer step requires at least one active row".into(),
            ));
        }
        if next_token_ids.len() != rows.len() {
            return Err(OrtError::InvalidArgument(format!(
                "next_token_ids length {} does not match active batch {}",
                next_token_ids.len(),
                rows.len()
            )));
        }
        let mut full_input = vec![0_i64; self.batch_size];
        let mut full_position = vec![0_i64; self.batch_size];
        let mut advances = vec![false; self.batch_size];
        for (active_index, &row) in rows.iter().enumerate() {
            full_input[row] = next_token_ids[active_index];
            if self.has_position_ids && active_index < position_ids.len() {
                full_position[row] = position_ids[active_index];
            }
            advances[row] = true;
        }
        let full_logits = self.run_batch(&full_input, &full_position, &advances)?;
        gather_logits_rows(&full_logits, &rows)
    }

    fn allocate_shared_buffers(&mut self) -> Result<()> {
        // NOTE: Unlike the single-stream `DecodeSession`, the batched
        // shared-buffer runner still allocates its KV buffers at the full
        // `max_len` up front rather than bucketing them (see
        // `kv_capacity_bucket` and `DecodeSession::ensure_kv_capacity`). This
        // session is not on the perf-critical single-stream captured-decode path
        // the CUDA capacity fix targets, and growing a *batched* buffer would
        // have to preserve every row's independent prefix and re-pack across
        // compaction — materially riskier than the single-row grow.
        // TODO: bucket the batched KV buffers too once the single-stream grow
        // path has been validated on CUDA.
        let batch_size = i64::try_from(self.batch_size)
            .map_err(|_| OrtError::InvalidArgument("batch_size exceeds i64".into()))?;
        let max_len = i64::try_from(self.max_len)
            .map_err(|_| OrtError::InvalidArgument("max_len exceeds i64".into()))?;
        let device_allocator = self.session.device_kv_allocator()?;
        let cpu_allocator;
        let allocator = match device_allocator.as_ref() {
            Some(allocator) => allocator,
            None => {
                cpu_allocator = crate::Allocator::default_cpu()?;
                &cpu_allocator
            }
        };
        let mut allocated = Vec::with_capacity(self.kv_pairs.len());
        for pair in &self.kv_pairs {
            let mut shape = pair.input.shape.clone();
            for (axis, dim) in shape.iter_mut().enumerate() {
                if axis == 0 {
                    *dim = batch_size;
                } else if axis == pair.seq_axis {
                    *dim = max_len;
                } else if *dim < 0 {
                    return Err(OrtError::InvalidArgument(format!(
                        "cannot infer shared-buffer static dimension {axis} for '{}'",
                        pair.past
                    )));
                }
            }
            allocated.push((
                pair.past.clone(),
                Arc::new(Value::empty_in(&shape, pair.input.dtype, allocator)?),
            ));
        }
        for (past, value) in allocated {
            self.kv_buffers.insert(past, value);
        }
        self.kv_allocator = device_allocator;
        Ok(())
    }

    /// Run one `[batch, 1]` decode step. Each row's attention mask carries
    /// `row_len + 1` leading ones (active rows) so the model derives that row's
    /// `seqlens_k` and writes its present KV at its own offset. Advancing rows
    /// have their logical cursor incremented afterwards.
    fn run_batch(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
        advances: &[bool],
    ) -> Result<Value> {
        if next_token_ids.len() != self.batch_size {
            return Err(OrtError::InvalidArgument(format!(
                "next_token_ids length {} does not match batch {}",
                next_token_ids.len(),
                self.batch_size
            )));
        }
        let batch = self.batch_size;
        // Per-row valid KV length for this step: active rows attend to their
        // prefix plus the new token (`row_len + 1`); inactive rows collapse to a
        // single position so their scratch write lands harmlessly at offset 0.
        let mut valid = vec![1usize; batch];
        for (row, valid_len) in valid.iter_mut().enumerate() {
            if self.active[row] {
                let next = self.row_lens[row] + 1;
                if next > self.max_len {
                    return Err(OrtError::InvalidArgument(format!(
                        "row {row} shared-buffer write at {} exceeds capacity {}",
                        self.row_lens[row], self.max_len
                    )));
                }
                *valid_len = next;
            }
        }
        let width = valid.iter().copied().max().unwrap_or(1).max(1);
        let width_i64 = i64::try_from(width)
            .map_err(|_| OrtError::InvalidArgument("mask width exceeds i64".into()))?;
        let batch_i64 = i64::try_from(batch)
            .map_err(|_| OrtError::InvalidArgument("batch exceeds i64".into()))?;

        let input_ids_value = Value::from_slice_i64(next_token_ids, &[batch_i64, 1])
            .map_err(|e| OrtError::InvalidArgument(format!("build input_ids value: {e}")))?;

        let mut mask = vec![0_i64; batch * width];
        for row in 0..batch {
            for col in 0..valid[row] {
                mask[row * width + col] = 1;
            }
        }
        let attention_mask_value = Value::from_slice_i64(&mask, &[batch_i64, width_i64])
            .map_err(|e| OrtError::InvalidArgument(format!("build attention_mask value: {e}")))?;

        let position_ids_value = if self.has_position_ids {
            let flat = if position_ids.len() == batch {
                position_ids.to_vec()
            } else {
                (0..batch).map(|row| self.row_lens[row] as i64).collect()
            };
            Some(Value::from_slice_i64(&flat, &[batch_i64, 1])?)
        } else {
            None
        };

        let bind_span = crate::prof_span!("ort.bind_inputs");
        self.binding.clear()?;
        for input in self.session.inputs() {
            let lower = input.name.to_ascii_lowercase();
            if lower == "input_ids" || lower.ends_with(".input_ids") {
                self.binding
                    .bind_input(&input.name, &input_ids_value)
                    .map_err(|e| {
                        OrtError::InvalidArgument(format!("bind input_ids '{}': {e}", input.name))
                    })?;
            } else if lower == "attention_mask" || lower.ends_with(".attention_mask") {
                self.binding
                    .bind_input(&input.name, &attention_mask_value)
                    .map_err(|e| {
                        OrtError::InvalidArgument(format!(
                            "bind attention_mask '{}': {e}",
                            input.name
                        ))
                    })?;
            } else if let Some(position_ids_value) = position_ids_value.as_ref()
                && (lower == "position_ids" || lower.ends_with(".position_ids"))
            {
                self.binding
                    .bind_input(&input.name, position_ids_value)
                    .map_err(|e| {
                        OrtError::InvalidArgument(format!(
                            "bind position_ids '{}': {e}",
                            input.name
                        ))
                    })?;
            }
        }
        for pair in &self.kv_pairs {
            let value = self.kv_buffers.get(&pair.past).ok_or_else(|| {
                OrtError::InvalidArgument(format!("missing shared KV buffer for '{}'", pair.past))
            })?;
            self.binding.bind_input(&pair.past, value).map_err(|e| {
                OrtError::InvalidArgument(format!(
                    "bind past '{}' shape {:?}: {e}",
                    pair.past,
                    value.shape()
                ))
            })?;
        }
        let mut borrowed_outputs = Vec::new();
        for output in self.session.output_names() {
            if let Some(pair) = self.kv_pairs.iter().find(|pair| pair.present == *output) {
                let value = self.kv_buffers.get(&pair.past).ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "missing shared KV buffer for '{}'",
                        pair.past
                    ))
                })?;
                borrowed_outputs.push(value.raw_ptr_addr());
                self.binding.bind_output(output, value).map_err(|e| {
                    OrtError::InvalidArgument(format!("bind present '{output}': {e}"))
                })?;
            } else {
                self.binding
                    .bind_output_to_device(output, &MemoryInfo::cpu()?)
                    .map_err(|e| {
                        OrtError::InvalidArgument(format!("bind output '{output}' to cpu: {e}"))
                    })?;
            }
        }
        drop(bind_span);

        {
            let _run_span = crate::prof_span!("ort.session_run");
            // Batched shared-buffer decode feeds a per-step-varying attention-mask
            // width (`total_sequence_length` grows as rows advance), so the graph
            // shape is not stable and cannot be CUDA-graph captured. When the
            // session was created with graph capture enabled we must therefore run
            // with annotation `-1` (execute normally, no capture/replay); a plain
            // `RunWithBinding` would attempt to capture the first shape and replay
            // it against later, differently-shaped steps, leaving outputs
            // unconstructed.
            let run_result = if self.session.graph_capture() {
                self.session.run_with_binding_graph(&self.binding, -1)
            } else {
                self.session.run_with_binding(&self.binding)
            };
            run_result.map_err(|e| {
                OrtError::InvalidArgument(format!(
                    "shared-buffer batched run (batch={batch}, width={width}): {e}"
                ))
            })?;
        }

        let _extract_span = crate::prof_span!("ort.extract_outputs");
        let outputs = self
            .binding
            .output_values_or_borrowed(&borrowed_outputs)
            .map_err(|e| OrtError::InvalidArgument(format!("extract batched outputs: {e}")))?;
        let mut logits = None;
        for (name, value) in self.session.output_names().iter().zip(outputs) {
            if is_logits_output(name) {
                logits = value;
                break;
            }
        }
        let logits = logits
            .ok_or_else(|| OrtError::InvalidArgument("model did not produce logits".into()))?;
        let logits = to_f32_logits(&logits).map_err(|e| {
            OrtError::InvalidArgument(format!("convert batched logits to f32: {e}"))
        })?;

        for (row, &advance) in advances[..batch].iter().enumerate() {
            if advance {
                self.row_lens[row] += 1;
            }
        }
        Ok(logits)
    }

    fn check_row(&self, row: usize) -> Result<()> {
        if row >= self.batch_size {
            return Err(OrtError::InvalidArgument(format!(
                "row {row} out of range for batch {}",
                self.batch_size
            )));
        }
        Ok(())
    }
}

impl<'a> BatchedDecodeSession<'a> for BatchedSharedBufferDecodeSession<'a> {
    fn batch_size(&self) -> usize {
        BatchedSharedBufferDecodeSession::batch_size(self)
    }
    fn max_len(&self) -> usize {
        BatchedSharedBufferDecodeSession::max_len(self)
    }
    fn row_len(&self, row: usize) -> Result<usize> {
        BatchedSharedBufferDecodeSession::row_len(self, row)
    }
    fn active_rows(&self) -> Vec<usize> {
        BatchedSharedBufferDecodeSession::active_rows(self)
    }
    fn deactivate_row(&mut self, row: usize) -> Result<()> {
        BatchedSharedBufferDecodeSession::deactivate_row(self, row)
    }
    fn assign_row(&mut self, row: usize) -> Result<()> {
        BatchedSharedBufferDecodeSession::assign_row(self, row)
    }
    fn step_select(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
        advance_rows: &[bool],
    ) -> Result<Value> {
        BatchedSharedBufferDecodeSession::step_select(
            self,
            next_token_ids,
            position_ids,
            advance_rows,
        )
    }
    fn step_active(&mut self, next_token_ids: &[i64], position_ids: &[i64]) -> Result<Value> {
        BatchedSharedBufferDecodeSession::step_active(self, next_token_ids, position_ids)
    }
}

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

fn infer_kv_pairs(session: &Session) -> Result<Vec<KvPair>> {
    let input_names = session.input_names();
    let mut pairs = Vec::new();
    for output in session.outputs() {
        if !name_contains_present_key_value(&output.name) {
            continue;
        }
        let Some(suffix) = kv_suffix(&output.name, KvNamingConvention::Dotted) else {
            continue;
        };
        let Some(past_name) = input_names.iter().find(|input| {
            kv_suffix(input, KvNamingConvention::Dotted).as_deref() == Some(suffix.as_str())
        }) else {
            continue;
        };
        let input = session
            .inputs()
            .iter()
            .find(|input| input.name == *past_name)
            .expect("past name came from session inputs")
            .clone();
        if !matches!(
            input.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(OrtError::InvalidArgument(format!(
                "KV input '{}' must be Float32, Float16, or BFloat16, got {:?}",
                input.name, input.dtype
            )));
        }
        if input.shape.len() < 3 {
            return Err(OrtError::InvalidArgument(format!(
                "KV input '{}' has unsupported shape {:?}",
                input.name, input.shape
            )));
        }
        let seq_axis = input.shape.len() - 2;
        pairs.push(KvPair {
            past: past_name.clone(),
            present: output.name.clone(),
            input,
            seq_axis,
        });
    }
    Ok(pairs)
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

fn detect_static_cache(
    session: &Session,
) -> Result<Option<(StaticCacheSignature, Vec<StaticCachePair>)>> {
    let has_write_indices = session
        .input_names()
        .iter()
        .any(|name| name == "write_indices");
    let has_nonpad = session
        .input_names()
        .iter()
        .any(|name| name == "nonpad_kv_seqlen");
    if !has_write_indices || !has_nonpad {
        return Ok(None);
    }

    let mut indices = session
        .inputs()
        .iter()
        .filter_map(|input| static_cache_suffix(&input.name, "key_cache."))
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    if indices.is_empty() {
        return Ok(None);
    }

    let mut pairs = Vec::with_capacity(indices.len());
    let mut max_len = None;
    let mut kv_dim = None;
    let mut dtype = None;
    for index in indices {
        let key_name = format!("key_cache.{index}");
        let value_name = format!("value_cache.{index}");
        let key_output = format!("updated_key_cache.{index}");
        let value_output = format!("updated_value_cache.{index}");
        let key_input = session
            .inputs()
            .iter()
            .find(|input| input.name == key_name)
            .cloned()
            .ok_or_else(|| OrtError::InvalidArgument(format!("missing input '{key_name}'")))?;
        let value_input = session
            .inputs()
            .iter()
            .find(|input| input.name == value_name)
            .cloned()
            .ok_or_else(|| OrtError::InvalidArgument(format!("missing input '{value_name}'")))?;
        if !session
            .output_names()
            .iter()
            .any(|name| name == &key_output)
        {
            return Err(OrtError::InvalidArgument(format!(
                "missing output '{key_output}'"
            )));
        }
        if !session
            .output_names()
            .iter()
            .any(|name| name == &value_output)
        {
            return Err(OrtError::InvalidArgument(format!(
                "missing output '{value_output}'"
            )));
        }
        validate_static_cache_tensor(&key_input)?;
        validate_static_cache_tensor(&value_input)?;
        if key_input.shape[1..] != value_input.shape[1..] {
            return Err(OrtError::InvalidArgument(format!(
                "key/value cache shape mismatch for layer {index}: {:?} vs {:?}",
                key_input.shape, value_input.shape
            )));
        }
        if key_input.dtype != value_input.dtype {
            return Err(OrtError::InvalidArgument(format!(
                "key/value cache dtype mismatch for layer {index}: {:?} vs {:?}",
                key_input.dtype, value_input.dtype
            )));
        }
        let layer_max_len = key_input.shape[1] as usize;
        let layer_kv_dim = key_input.shape[2] as usize;
        if max_len.get_or_insert(layer_max_len) != &layer_max_len {
            return Err(OrtError::InvalidArgument(
                "static-cache layers have inconsistent max lengths".into(),
            ));
        }
        if kv_dim.get_or_insert(layer_kv_dim) != &layer_kv_dim {
            return Err(OrtError::InvalidArgument(
                "static-cache layers have inconsistent KV dims".into(),
            ));
        }
        if dtype.get_or_insert(key_input.dtype) != &key_input.dtype {
            return Err(OrtError::InvalidArgument(
                "static-cache layers have inconsistent dtypes".into(),
            ));
        }
        pairs.push(StaticCachePair {
            index,
            key_input,
            value_input,
            key_output,
            value_output,
        });
    }
    pairs.sort_by_key(|pair| pair.index);
    let signature = StaticCacheSignature {
        layers: pairs.len(),
        max_len: max_len.expect("non-empty static cache pairs"),
        kv_dim: kv_dim.expect("non-empty static cache pairs"),
        dtype: dtype.expect("non-empty static cache pairs"),
        has_position_ids: session
            .input_names()
            .iter()
            .any(|name| name == "position_ids"),
    };
    Ok(Some((signature, pairs)))
}

fn static_cache_suffix(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?.parse().ok()
}

fn validate_static_cache_tensor(info: &TensorInfo) -> Result<()> {
    if !matches!(
        info.dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Err(OrtError::InvalidArgument(format!(
            "static-cache tensor '{}' must be Float32, Float16, or BFloat16, got {:?}",
            info.name, info.dtype
        )));
    }
    if info.shape.len() != 3 || info.shape[1] <= 0 || info.shape[2] <= 0 {
        return Err(OrtError::InvalidArgument(format!(
            "static-cache tensor '{}' must have shape [B, MAX_LEN, KV_DIM], got {:?}",
            info.name, info.shape
        )));
    }
    Ok(())
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
