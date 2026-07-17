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

use crate::{DataType, IoBinding, MemoryInfo, OrtError, Result, Session, TensorInfo, Value};

/// Prompt and prefill runs use CUDA-graph annotation id `-1` (no capture) so
/// only the fixed-shape decode step is captured and replayed. Each
/// [`DecodeSession`] that enables capture claims a process-unique positive id
/// (see [`next_capture_graph_id`]) so that reusing the underlying ORT session
/// for a new generation never re-captures a different graph under an id ORT
/// already holds — which corrupts ORT's per-id CUDA-graph bookkeeping.
static NEXT_CAPTURE_GRAPH_ID: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(1);

fn next_capture_graph_id() -> i32 {
    // Ids must be unique across concurrently-live sessions and strictly positive
    // so they never collide with the `-1` no-capture sentinel. Masking off the
    // sign bit keeps them positive and unique within each 2^31 cycle; the lone
    // zero per cycle is remapped. A wrap would only reuse an id after 2^31
    // generations, by which point the prior holder is long dropped.
    let raw = NEXT_CAPTURE_GRAPH_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    match raw & i32::MAX {
        0 => i32::MAX,
        id => id,
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

/// A stateful IoBinding decode runner that keeps KV OrtValues inside ORT.
pub struct DecodeSession<'a> {
    session: &'a Session,
    binding: IoBinding,
    kv_pairs: Vec<KvPair>,
    current_kv: HashMap<String, Arc<Value>>,
    current_len: usize,
    mode: DecodeKvMode,
    /// Owned device allocator that backs the shared-buffer KV `Value`s in
    /// `current_kv`. OrtValues created through an allocator free their memory
    /// via that allocator on release, so it MUST outlive the `Value`s. This
    /// field is declared after `current_kv` so Rust drops the KV `Value`s first
    /// and releases this allocator afterwards; releasing it earlier caused a
    /// use-after-free SIGSEGV at session close.
    kv_allocator: Option<crate::Allocator>,
    /// Static-shape captured-decode state, populated lazily on the first
    /// single-token step when the session has CUDA graph capture enabled.
    /// Holds the persistent, fixed-address I/O buffers a captured graph replays
    /// against: `input_ids [1,1]`, `position_ids [1,1]`, a max-length
    /// `attention_mask [1, max_len]`, and the `logits [1,1,vocab]` output.
    capture: Option<CaptureState>,
    /// Fixed KV capacity (shared-buffer `max_length`), needed to size the
    /// captured attention mask. `None` outside shared-buffer mode.
    max_length: Option<usize>,
    /// Whether the persistent captured-decode I/O is currently bound. Captured
    /// graphs require stable bindings across replays, so we bind the persistent
    /// buffers once and only rebind after a non-captured step clears them.
    capture_bound: bool,
    /// Process-unique CUDA-graph annotation id claimed lazily when this session
    /// first captures its decode graph. `None` until the first captured step.
    capture_graph_id: Option<i32>,
    /// Set when a captured decode step fails and we fall back to the standard
    /// decode path for the rest of this generation. Once set, the captured fast
    /// path is skipped even though the underlying session still reports
    /// `graph_capture() == true`, so graceful degradation persists per decode
    /// loop without mutating the shared session.
    graph_capture_disabled: bool,
}

/// Persistent I/O buffers for the static-shape captured decode graph.
struct CaptureState {
    input_ids: Value,
    position_ids: Value,
    attention_mask: Value,
    logits: Value,
    mask_len: usize,
    /// Number of leading `attention_mask` entries currently set to 1. The valid
    /// region only grows within a generation, so each step fills just the delta
    /// `[mask_valid_len, valid_len)` instead of rewriting the whole prefix,
    /// keeping the captured-decode step O(1) rather than O(context). Reset to 0
    /// by [`DecodeSession::reset_captured_mask`] on rewind/reset.
    mask_valid_len: usize,
}

impl Drop for DecodeSession<'_> {
    fn drop(&mut self) {
        // If this session captured a decode graph, release it now — while this
        // session's fixed-address I/O buffers are still alive (fields are
        // dropped after this method returns). The captured graph references
        // those buffers; leaving it registered on the shared ORT session would
        // let a later release clean up a graph whose buffers were already freed,
        // corrupting the heap.
        if let Some(graph_id) = self.capture_graph_id {
            let _ = self.session.release_captured_graph(graph_id);
        }
    }
}

struct StaticCachePair {
    index: usize,
    key_input: TensorInfo,
    value_input: TensorInfo,
    key_output: String,
    value_output: String,
}

struct StaticCacheBuffer {
    input_name: String,
    output_name: String,
    current: Arc<Value>,
    alternate: Option<Arc<Value>>,
}

/// Stateful decode runner for Mobius/STATIC-CACHE TensorScatter models.
///
/// The runtime owns fixed `[B, MAX_LEN, KV_DIM]` key/value buffers. The model's
/// `updated_*` outputs are bound back onto those buffers; the graph scatter is a
/// write hint, not the source of truth for cache ownership.
pub struct StaticCacheDecodeSession<'a> {
    session: &'a Session,
    binding: IoBinding,
    signature: StaticCacheSignature,
    batch_size: i64,
    current_len: usize,
    mode: StaticCacheBindingMode,
    buffers: Vec<StaticCacheBuffer>,
}

/// Batched stateful decode runner for static-cache TensorScatter models.
///
/// One agent/session is assigned to one logical row id. KV buffers are allocated
/// once as `[B, MAX_LEN, KV_DIM]` per layer and bound in-place like
/// [`StaticCacheDecodeSession`]. Logical rows can be compacted to a packed
/// physical prefix so active-only steps bind `[active, MAX_LEN, KV_DIM]` aliases
/// and avoid running model compute for inactive rows.
pub struct BatchedStaticCacheDecodeSession<'a> {
    session: &'a Session,
    binding: IoBinding,
    signature: StaticCacheSignature,
    batch_size: usize,
    row_lens: Vec<usize>,
    active: Vec<bool>,
    logical_to_physical: Vec<Option<usize>>,
    physical_to_logical: Vec<Option<usize>>,
    mode: StaticCacheBindingMode,
    buffers: Vec<StaticCacheBuffer>,
}

impl<'a> DecodeSession<'a> {
    /// Create a decode session and infer KV input/output pairs from graph names.
    pub fn new(session: &'a Session, options: DecodeSessionOptions) -> Result<Self> {
        let kv_pairs = infer_kv_pairs(session)?;
        let share_buffer = options
            .past_present_share_buffer
            .unwrap_or(session.past_present_share_buffer_supported());
        let mode = if share_buffer {
            DecodeKvMode::SharedBuffer
        } else {
            DecodeKvMode::ZeroCopyRebind
        };
        let mut this = Self {
            session,
            binding: IoBinding::new(session)?,
            kv_pairs,
            current_kv: HashMap::new(),
            current_len: 0,
            mode,
            kv_allocator: None,
            capture: None,
            max_length: None,
            capture_bound: false,
            capture_graph_id: None,
            graph_capture_disabled: false,
        };
        if mode == DecodeKvMode::SharedBuffer {
            let max_length = options.max_length.ok_or_else(|| {
                OrtError::InvalidArgument(
                    "DecodeSession shared-buffer mode requires max_length".into(),
                )
            })?;
            this.allocate_shared_buffers(options.batch_size, max_length)?;
            this.max_length = Some(max_length);
        }
        Ok(this)
    }

    /// The selected KV binding strategy.
    pub fn mode(&self) -> DecodeKvMode {
        self.mode
    }

    /// Current logical KV length in tokens.
    pub fn past_len(&self) -> usize {
        self.current_len
    }

    /// Run one incremental decode step and return the logits OrtValue.
    ///
    /// `attention_mask` is the full `[batch, past + new]` mask flattened row-major,
    /// while `position_ids` covers only `new_input_ids`.
    pub fn step(
        &mut self,
        new_input_ids: &[i64],
        attention_mask: &[i64],
        position_ids: &[i64],
    ) -> Result<Value> {
        if new_input_ids.is_empty() {
            return Err(OrtError::InvalidArgument(
                "decode step requires at least one input id".into(),
            ));
        }
        let seq_len = i64::try_from(new_input_ids.len())
            .map_err(|_| OrtError::InvalidArgument("input length exceeds i64".into()))?;
        let total_len = i64::try_from(attention_mask.len())
            .map_err(|_| OrtError::InvalidArgument("attention mask length exceeds i64".into()))?;
        if position_ids.len() != new_input_ids.len() {
            return Err(OrtError::InvalidArgument(
                "position_ids length must match input_ids length".into(),
            ));
        }

        // Static-shape captured decode fast path: once the prompt has been
        // consumed, every decode step feeds one token with fixed-shape inputs
        // and fixed-address KV buffers, so a single CUDA graph can be captured
        // and replayed to eliminate per-kernel launch overhead.
        if self.mode == DecodeKvMode::SharedBuffer
            && self.session.graph_capture()
            && !self.graph_capture_disabled
            && new_input_ids.len() == 1
            && self.current_len > 0
        {
            match self.step_captured(new_input_ids[0], attention_mask, position_ids[0]) {
                Ok(logits) => return Ok(logits),
                Err(err) => {
                    // The captured decode path failed (e.g. the EP could not
                    // capture/replay a graph for this session). Degrade
                    // gracefully: skip the captured path for the rest of this
                    // generation, drop any partial capture state, and fall
                    // through to the standard step below. `step_captured`
                    // advances `current_len` only on success, so no KV progress
                    // is lost by retrying here.
                    tracing::warn!(
                        error = %err,
                        "CUDA graph decode step failed; disabling graph capture and \
                         falling back to the standard decode path for the rest of this session"
                    );
                    self.graph_capture_disabled = true;
                    self.capture = None;
                }
            }
        }

        let input_ids = Value::from_slice_i64(new_input_ids, &[1, seq_len])?;
        let attention_mask = Value::from_slice_i64(attention_mask, &[1, total_len])?;
        let position_ids = Value::from_slice_i64(position_ids, &[1, seq_len])?;

        let bind_span = crate::prof_span!("ort.bind_inputs");
        self.binding.clear()?;
        // This step re-binds fresh per-step Values, so any persistent captured
        // binding is now stale and must be re-established before the next
        // captured step.
        self.capture_bound = false;
        self.bind_standard_inputs(&input_ids, &attention_mask, &position_ids)?;
        self.bind_kv_inputs()?;
        let mut borrowed_outputs = Vec::new();
        for output in self.session.output_names() {
            if self.mode == DecodeKvMode::SharedBuffer
                && let Some(pair) = self.kv_pairs.iter().find(|pair| pair.present == *output)
            {
                let value = self.current_kv.get(&pair.past).ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "missing shared KV buffer for '{}'",
                        pair.past
                    ))
                })?;
                borrowed_outputs.push(value.raw_ptr_addr());
                self.binding.bind_output(output, value)?;
            } else {
                self.binding
                    .bind_output_to_device(output, &MemoryInfo::cpu()?)?;
            }
        }
        drop(bind_span);

        {
            let _run_span = crate::prof_span!("ort.session_run");
            // When graph capture is enabled, prompt/prefill runs use annotation
            // -1 so ORT executes them normally instead of capturing them as the
            // (differently-shaped) decode graph.
            if self.session.graph_capture() && !self.graph_capture_disabled {
                self.session.run_with_binding_graph(&self.binding, -1)?;
            } else {
                self.session.run_with_binding(&self.binding)?;
            }
        }
        let _extract_span = crate::prof_span!("ort.extract_outputs");
        let mut logits = None;
        if self.mode == DecodeKvMode::SharedBuffer {
            let outputs = self.binding.output_values_or_borrowed(&borrowed_outputs)?;
            for (name, value) in self.session.output_names().iter().zip(outputs) {
                if is_logits_output(name) {
                    logits = value;
                    break;
                }
            }
        } else {
            let outputs = self.binding.output_values()?;
            self.rotate_outputs(outputs, &mut logits)?;
        }
        self.current_len = self
            .current_len
            .checked_add(new_input_ids.len())
            .ok_or_else(|| OrtError::InvalidArgument("decode length overflow".into()))?;
        logits.ok_or_else(|| OrtError::InvalidArgument("model did not produce logits".into()))
    }

    /// Single-token decode step replayed through a captured CUDA graph.
    ///
    /// All inputs are bound to persistent, fixed-address buffers whose shapes
    /// never change across steps: `input_ids [1,1]`, `position_ids [1,1]`, and a
    /// full-capacity `attention_mask [1, max_len]` whose leading `valid_len`
    /// entries are 1 (the model derives GQA sequence lengths from the mask, so
    /// the trailing zeros mask the unused KV-buffer tail). KV buffers are the
    /// same fixed shared buffers bound in place as both past inputs and present
    /// outputs. Logits are written into a persistent output buffer. The first
    /// such step captures the graph; subsequent steps replay it.
    fn step_captured(
        &mut self,
        token: i64,
        attention_mask: &[i64],
        position: i64,
    ) -> Result<Value> {
        self.ensure_capture_state()?;
        // Move the capture buffers out of `self` for the duration of the step so
        // the `&mut self` bind helpers don't alias the borrow; restore on the
        // success path (an error aborts generation and drops the state).
        let mut cap = self.capture.take().expect("capture state initialized");
        let valid_len = attention_mask.len();
        if valid_len > cap.mask_len {
            return Err(OrtError::InvalidArgument(format!(
                "attention length {valid_len} exceeds captured mask capacity {}",
                cap.mask_len
            )));
        }
        cap.input_ids.write_i64_prefix(&[token])?;
        cap.position_ids.write_i64_prefix(&[position])?;
        // The mask's valid region only grows within a generation (rewind/reset
        // clear it), and prior entries are already 1, so fill just the newly
        // valid tail — typically a single element — keeping this step O(1) in
        // context rather than rewriting (and heap-allocating) the whole prefix.
        if valid_len > cap.mask_valid_len {
            cap.attention_mask.fill_i64_range(
                cap.mask_valid_len,
                valid_len - cap.mask_valid_len,
                1,
            )?;
        } else if valid_len < cap.mask_valid_len {
            // Defensive: a shrink without an intervening reset — clear the tail
            // that is no longer valid so it does not leak into this step.
            cap.attention_mask
                .fill_i64_range(valid_len, cap.mask_valid_len - valid_len, 0)?;
        }
        cap.mask_valid_len = valid_len;

        // Re-bind the persistent buffers every step. ORT keys its stable
        // internal device input buffers off the binding and re-copies the bound
        // CPU inputs host->device on each Run; a captured graph replays against
        // those stable device buffers. Binding only once would freeze the device
        // inputs at their first-step values (the graph does not re-copy CPU
        // inputs on replay), so the model would repeat a single token.
        let bind_span = crate::prof_span!("ort.bind_inputs");
        self.binding.clear()?;
        self.bind_standard_inputs(&cap.input_ids, &cap.attention_mask, &cap.position_ids)?;
        self.bind_kv_inputs()?;
        for output in self.session.output_names() {
            if let Some(pair) = self.kv_pairs.iter().find(|pair| pair.present == *output) {
                let value = self.current_kv.get(&pair.past).ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "missing shared KV buffer for '{}'",
                        pair.past
                    ))
                })?;
                self.binding.bind_output(output, value)?;
            } else if is_logits_output(output) {
                self.binding.bind_output(output, &cap.logits)?;
            } else {
                self.binding
                    .bind_output_to_device(output, &MemoryInfo::cpu()?)?;
            }
        }
        drop(bind_span);

        {
            let _run_span = crate::prof_span!("ort.session_run");
            let graph_id = self
                .capture_graph_id
                .expect("capture graph id assigned in ensure_capture_state");
            self.session
                .run_with_binding_graph(&self.binding, graph_id)?;
        }
        // A graph is now captured under `capture_graph_id`; mark it so reset /
        // rewind / drop release it before this session's buffers are freed.
        self.capture_bound = true;
        self.current_len = self
            .current_len
            .checked_add(1)
            .ok_or_else(|| OrtError::InvalidArgument("decode length overflow".into()))?;

        // Copy the persistent logits buffer into an owned Value so the caller
        // can consume it while the captured buffer is reused next step.
        let _extract_span = crate::prof_span!("ort.extract_outputs");
        let logits = cap.logits.clone_owned();
        self.capture = Some(cap);
        logits
    }

    /// Lazily allocate the persistent captured-decode I/O buffers.
    fn ensure_capture_state(&mut self) -> Result<()> {
        if self.capture.is_some() {
            return Ok(());
        }
        let mask_len = self.max_length.ok_or_else(|| {
            OrtError::InvalidArgument("captured decode requires max_length".into())
        })?;
        let logits_info = self
            .session
            .outputs()
            .iter()
            .find(|info| is_logits_output(&info.name))
            .ok_or_else(|| OrtError::InvalidArgument("model exposes no logits output".into()))?;
        let vocab = logits_info
            .shape
            .last()
            .copied()
            .filter(|dim| *dim > 0)
            .ok_or_else(|| {
                OrtError::InvalidArgument("logits output has no static vocab dim".into())
            })?;

        let input_ids = Value::from_vec_i64(vec![0i64], &[1, 1])?;
        let position_ids = Value::from_vec_i64(vec![0i64], &[1, 1])?;
        let attention_mask = Value::from_vec_i64(vec![0i64; mask_len], &[1, mask_len as i64])?;
        let logits = Value::empty(&[1, 1, vocab], logits_info.dtype)?;

        // Claim a process-unique annotation id so this session captures its own
        // graph rather than re-capturing under an id ORT may still hold from a
        // prior generation on the same underlying ORT session.
        self.capture_graph_id = Some(next_capture_graph_id());

        self.capture = Some(CaptureState {
            input_ids,
            position_ids,
            attention_mask,
            logits,
            mask_len,
            mask_valid_len: 0,
        });
        Ok(())
    }

    /// Rewind to a smaller logical KV length.
    ///
    /// In zero-copy-rebind mode this rebinds a compact prefix tensor for each
    /// current present buffer. This is no-copy when the prefix is contiguous in
    /// memory; otherwise rewind performs a one-time compacting slice copy for
    /// correctness. In shared-buffer mode only the logical cursor changes; stale
    /// data beyond `target_len` remains in the buffers and must be masked out by
    /// subsequent attention masks/position ids.
    pub fn rewind(&mut self, target_len: usize) -> Result<()> {
        if target_len > self.current_len {
            return Err(OrtError::InvalidArgument(format!(
                "cannot rewind from {} to larger length {}",
                self.current_len, target_len
            )));
        }
        if target_len == self.current_len {
            return Ok(());
        }
        if target_len == 0 {
            if self.mode == DecodeKvMode::ZeroCopyRebind {
                self.current_kv.clear();
            }
            self.current_len = 0;
            self.invalidate_captured_graph();
            return Ok(());
        }
        if self.mode == DecodeKvMode::ZeroCopyRebind {
            let mut rewound = HashMap::with_capacity(self.current_kv.len());
            for pair in &self.kv_pairs {
                let value = self.current_kv.get(&pair.past).ok_or_else(|| {
                    OrtError::InvalidArgument(format!("missing KV tensor '{}'", pair.past))
                })?;
                let mut shape = value.shape().to_vec();
                shape[pair.seq_axis] = i64::try_from(target_len).map_err(|_| {
                    OrtError::InvalidArgument("target rewind length exceeds i64".into())
                })?;
                rewound.insert(
                    pair.past.clone(),
                    Arc::new(prefix_value(value, &shape, pair.seq_axis)?),
                );
            }

            fn prefix_value(value: &Arc<Value>, shape: &[i64], seq_axis: usize) -> Result<Value> {
                let owner_shape = value.shape();
                let prefix_is_contiguous = owner_shape.iter().take(seq_axis).all(|&dim| dim == 1);
                if prefix_is_contiguous {
                    return Value::alias_with_shape(Arc::clone(value), shape);
                }

                match value.dtype() {
                    DataType::Float32 => {
                        let data = value.to_vec_f32()?;
                        let prefix = copy_prefix(&data, owner_shape, shape);
                        Value::from_vec_f32(prefix, shape)
                    }
                    DataType::Float16 => {
                        let data = value.to_vec_f16_bits()?;
                        let prefix = copy_prefix(&data, owner_shape, shape);
                        Value::from_vec_f16_bits(prefix, shape)
                    }
                    DataType::BFloat16 => {
                        let data = value.to_vec_bf16_bits()?;
                        let prefix = copy_prefix(&data, owner_shape, shape);
                        Value::from_vec_bf16_bits(prefix, shape)
                    }
                    dtype => Err(OrtError::InvalidArgument(format!(
                        "cannot rewind KV tensor with dtype {dtype:?}"
                    ))),
                }
            }

            fn copy_prefix<T: Copy>(
                data: &[T],
                input_shape: &[i64],
                output_shape: &[i64],
            ) -> Vec<T> {
                let output_len = output_shape.iter().product::<i64>() as usize;
                let mut output = Vec::with_capacity(output_len);
                let input_strides = row_major_strides(input_shape);
                for mut linear in 0..output_len {
                    let mut input_offset = 0usize;
                    for (axis, &dim) in output_shape.iter().enumerate().rev() {
                        let index = linear % dim as usize;
                        linear /= dim as usize;
                        input_offset += index * input_strides[axis];
                    }
                    output.push(data[input_offset]);
                }
                output
            }

            fn row_major_strides(shape: &[i64]) -> Vec<usize> {
                let mut strides = vec![1; shape.len()];
                for axis in (0..shape.len().saturating_sub(1)).rev() {
                    strides[axis] = strides[axis + 1] * shape[axis + 1] as usize;
                }
                strides
            }
            self.current_kv = rewound;
        }
        self.current_len = target_len;
        // The captured attention mask relies on the valid region growing
        // monotonically, so a rewind must clear the now-invalid tail and drop
        // the captured graph so the next step re-captures at the new position.
        self.invalidate_captured_graph();
        self.reset_captured_mask()?;
        Ok(())
    }

    /// Zero the valid region of the persistent captured attention mask, if
    /// allocated, and reset the valid-length counter. Called on rewind/reset so
    /// a shorter or restarted sequence never sees stale ones in the trailing
    /// (masked-out) region. Only the previously-valid prefix is cleared — the
    /// rest is already zero — so this stays O(previous context), not O(max_len).
    fn reset_captured_mask(&mut self) -> Result<()> {
        if let Some(cap) = self.capture.as_mut() {
            cap.attention_mask
                .fill_i64_range(0, cap.mask_valid_len, 0)?;
            cap.mask_valid_len = 0;
        }
        Ok(())
    }

    /// Release any captured decode graph and force the next captured step to
    /// re-capture under a fresh annotation id. A captured CUDA graph replays
    /// against the exact buffers/positions seen at capture time; after a reset
    /// or rewind the sequence structure changes, so the stale graph must not be
    /// replayed. A fresh id avoids re-capturing under an id ORT may still hold.
    fn invalidate_captured_graph(&mut self) {
        if self.capture_bound {
            if let Some(graph_id) = self.capture_graph_id {
                let _ = self.session.release_captured_graph(graph_id);
            }
            // Re-capture under a new id if this session keeps decoding.
            self.capture_graph_id = Some(next_capture_graph_id());
            self.capture_bound = false;
        }
    }

    /// Reset the decode cursor and drop zero-copy-rebind KV state.
    pub fn reset(&mut self) {
        if self.mode == DecodeKvMode::ZeroCopyRebind {
            self.current_kv.clear();
        }
        self.current_len = 0;
        self.invalidate_captured_graph();
        let _ = self.reset_captured_mask();
    }

    /// Export the current KV cache as owned, session-independent CPU tensors.
    ///
    /// Each entry is `(past_key_values.* input name, materialized Value)` whose
    /// backing data is copied onto host-owned Rust buffers, so the returned
    /// values outlive the producing session and can be handed to a *different*
    /// [`DecodeSession`] loaded from the same model via [`Self::import_kv`].
    ///
    /// This is the KV-handoff primitive for hybrid execution (e.g. prefill on
    /// the Metal EP, decode on the CPU EP). On Apple-silicon unified memory the
    /// producing session's `present.*` outputs are already CPU-addressable, so
    /// the copy is a cheap host `memcpy`. Only supported in
    /// [`DecodeKvMode::ZeroCopyRebind`], where the runtime holds the present KV
    /// as materialized OrtValues; shared-buffer mode owns fixed max-length
    /// device buffers that are not portable across sessions.
    pub fn export_kv(&self) -> Result<Vec<(String, Value)>> {
        if self.mode != DecodeKvMode::ZeroCopyRebind {
            return Err(OrtError::InvalidArgument(
                "export_kv is only supported in ZeroCopyRebind mode".into(),
            ));
        }
        let mut exported = Vec::with_capacity(self.kv_pairs.len());
        for pair in &self.kv_pairs {
            let value = self.current_kv.get(&pair.past).ok_or_else(|| {
                OrtError::InvalidArgument(format!(
                    "cannot export KV: missing tensor '{}' (run a prefill/decode step first)",
                    pair.past
                ))
            })?;
            exported.push((pair.past.clone(), clone_value_to_owned(value)?));
        }
        Ok(exported)
    }

    /// Adopt a KV cache produced by another session (same model) and set the
    /// logical KV length to `len`.
    ///
    /// The counterpart to [`Self::export_kv`]: it replaces this session's KV
    /// state so the next [`Self::step`] continues generation from `len` tokens
    /// of context. Every `past_key_values.*` tensor this model expects must be
    /// present in `kv` and match the model's dtype; the sequence axis of each
    /// tensor must equal `len`. Only supported in
    /// [`DecodeKvMode::ZeroCopyRebind`].
    pub fn import_kv(&mut self, len: usize, kv: Vec<(String, Value)>) -> Result<()> {
        if self.mode != DecodeKvMode::ZeroCopyRebind {
            return Err(OrtError::InvalidArgument(
                "import_kv is only supported in ZeroCopyRebind mode".into(),
            ));
        }
        let mut incoming: HashMap<String, Value> = kv.into_iter().collect();
        let mut adopted = HashMap::with_capacity(self.kv_pairs.len());
        for pair in &self.kv_pairs {
            let value = incoming.remove(&pair.past).ok_or_else(|| {
                OrtError::InvalidArgument(format!("import_kv missing KV tensor '{}'", pair.past))
            })?;
            if value.dtype() != pair.input.dtype {
                return Err(OrtError::InvalidArgument(format!(
                    "import_kv dtype mismatch for '{}': got {:?}, expected {:?}",
                    pair.past,
                    value.dtype(),
                    pair.input.dtype
                )));
            }
            let seq_dim = value.shape().get(pair.seq_axis).copied().unwrap_or(-1);
            if seq_dim != i64::try_from(len).unwrap_or(-1) {
                return Err(OrtError::InvalidArgument(format!(
                    "import_kv length mismatch for '{}': seq axis {} = {}, expected {}",
                    pair.past, pair.seq_axis, seq_dim, len
                )));
            }
            adopted.insert(pair.past.clone(), Arc::new(value));
        }
        self.current_kv = adopted;
        self.current_len = len;
        Ok(())
    }

    fn bind_standard_inputs(
        &mut self,
        input_ids: &Value,
        attention_mask: &Value,
        position_ids: &Value,
    ) -> Result<()> {
        for input in self.session.inputs() {
            let lower = input.name.to_ascii_lowercase();
            if lower == "input_ids" || lower.ends_with(".input_ids") {
                self.binding.bind_input(&input.name, input_ids)?;
            } else if lower == "attention_mask" || lower.ends_with(".attention_mask") {
                self.binding.bind_input(&input.name, attention_mask)?;
            } else if lower == "position_ids" || lower.ends_with(".position_ids") {
                self.binding.bind_input(&input.name, position_ids)?;
            }
        }
        Ok(())
    }

    fn bind_kv_inputs(&mut self) -> Result<()> {
        for pair in &self.kv_pairs {
            let value = if let Some(value) = self.current_kv.get(&pair.past) {
                Arc::clone(value)
            } else {
                Arc::new(empty_past_value(&pair.input)?)
            };
            self.binding.bind_input(&pair.past, &value)?;
        }
        Ok(())
    }

    fn rotate_outputs(&mut self, outputs: Vec<Value>, logits: &mut Option<Value>) -> Result<()> {
        if self.mode == DecodeKvMode::SharedBuffer {
            for (name, value) in self.session.output_names().iter().zip(outputs) {
                if is_logits_output(name) {
                    *logits = Some(value);
                    break;
                }
            }
            return Ok(());
        }

        let present_to_past = self
            .kv_pairs
            .iter()
            .map(|pair| (pair.present.as_str(), pair.past.as_str()))
            .collect::<HashMap<_, _>>();
        let mut next_kv = HashMap::with_capacity(self.kv_pairs.len());
        for (name, value) in self.session.output_names().iter().zip(outputs) {
            if let Some(past_name) = present_to_past.get(name.as_str()) {
                next_kv.insert((*past_name).to_string(), Arc::new(value));
            } else if is_logits_output(name) || logits.is_none() {
                *logits = Some(value);
            }
        }
        self.current_kv = next_kv;
        Ok(())
    }

    fn allocate_shared_buffers(&mut self, batch_size: i64, max_length: usize) -> Result<()> {
        let max_length = i64::try_from(max_length)
            .map_err(|_| OrtError::InvalidArgument("max_length exceeds i64".into()))?;
        // Prefer a device-resident (e.g. CUDA/WebGPU) allocator so the runtime-owned
        // max-length KV buffers live on the EP device. Bound as both
        // `past_key_values.*` inputs and `present.*` outputs, the KV cache then
        // stays on-device across decode steps (present aliased in place onto
        // past), eliminating the per-step host<->device KV copies the default
        // CPU allocator would force under an accelerator EP. Falls back to the
        // CPU allocator for CPU / non-device EPs.
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
                    *dim = max_length;
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
        // The `allocator` borrow of `device_allocator` ends here; retain the
        // owned device allocator so it outlives the KV `Value`s it just backed
        // (see `DecodeSession::kv_allocator`). Moving the wrapper does not change
        // the underlying `OrtAllocator*` the `Value`s reference. The CPU fallback
        // allocator is the process-owned default and needs no retention.
        for (past, value) in allocated {
            self.current_kv.insert(past, value);
        }
        self.kv_allocator = device_allocator;
        Ok(())
    }
}

impl<'a> StaticCacheDecodeSession<'a> {
    /// Detect a STATIC-CACHE/TensorScatter signature from ONNX graph I/O.
    pub fn detect(session: &Session) -> Result<Option<StaticCacheSignature>> {
        Ok(detect_static_cache(session)?.map(|(signature, _)| signature))
    }

    /// Create a static-cache decode session if the graph exposes the signature.
    pub fn new(session: &'a Session, options: StaticCacheDecodeOptions) -> Result<Self> {
        let (signature, pairs) = detect_static_cache(session)?.ok_or_else(|| {
            OrtError::InvalidArgument(
                "model does not expose static-cache key_cache/write_indices inputs".into(),
            )
        })?;
        let buffers = allocate_static_cache_buffers(options.batch_size, &pairs)?;
        Ok(Self {
            session,
            binding: IoBinding::new(session)?,
            signature,
            batch_size: options.batch_size,
            current_len: 0,
            mode: StaticCacheBindingMode::InPlaceAlias,
            buffers,
        })
    }

    pub fn signature(&self) -> &StaticCacheSignature {
        &self.signature
    }

    pub fn binding_mode(&self) -> StaticCacheBindingMode {
        self.mode
    }

    pub fn max_len(&self) -> usize {
        self.signature.max_len
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Runtime-owned KV buffer identities and sizes.
    pub fn buffer_infos(&self) -> Result<Vec<StaticCacheBufferInfo>> {
        self.buffers
            .iter()
            .map(|buffer| {
                Ok(StaticCacheBufferInfo {
                    input_name: buffer.input_name.clone(),
                    output_name: buffer.output_name.clone(),
                    shape: buffer.current.shape().to_vec(),
                    dtype: buffer.current.dtype(),
                    data_ptr: buffer.current.data_ptr_addr()?,
                    numel: buffer.current.numel(),
                })
            })
            .collect()
    }

    /// Scatter a prompt chunk into slots `0..P` and return logits.
    pub fn prefill(&mut self, input_ids: &[i64], position_ids: &[i64]) -> Result<Value> {
        let seq_len = self.seq_len_from_flat_input(input_ids)?;
        self.run_static_chunk(input_ids, position_ids, seq_len, 0)?;
        self.current_len = seq_len;
        self.last_logits()
    }

    /// Scatter one token per batch row at the current write cursor.
    pub fn step(&mut self, next_token_ids: &[i64], position_ids: &[i64]) -> Result<Value> {
        if next_token_ids.len() != self.batch_size as usize {
            return Err(OrtError::InvalidArgument(format!(
                "static-cache step expects {} token ids, got {}",
                self.batch_size,
                next_token_ids.len()
            )));
        }
        self.run_static_chunk(next_token_ids, position_ids, 1, self.current_len)?;
        self.current_len += 1;
        self.last_logits()
    }

    /// Rewind the logical write cursor. Buffers are retained and stale suffix
    /// slots are overwritten by subsequent prefill/step calls.
    pub fn rewind(&mut self, target_len: usize) -> Result<()> {
        if target_len > self.current_len {
            return Err(OrtError::InvalidArgument(format!(
                "cannot rewind static cache from {} to larger length {}",
                self.current_len, target_len
            )));
        }
        self.current_len = target_len;
        Ok(())
    }

    fn seq_len_from_flat_input(&self, input_ids: &[i64]) -> Result<usize> {
        let batch = self.batch_size as usize;
        if batch == 0 || input_ids.is_empty() || !input_ids.len().is_multiple_of(batch) {
            return Err(OrtError::InvalidArgument(format!(
                "input_ids length {} is not a non-empty multiple of batch {}",
                input_ids.len(),
                batch
            )));
        }
        Ok(input_ids.len() / batch)
    }

    fn run_static_chunk(
        &mut self,
        input_ids: &[i64],
        position_ids: &[i64],
        seq_len: usize,
        write_index: usize,
    ) -> Result<()> {
        if write_index + seq_len > self.signature.max_len {
            return Err(OrtError::InvalidArgument(format!(
                "static-cache write {}..{} exceeds capacity {}",
                write_index,
                write_index + seq_len,
                self.signature.max_len
            )));
        }
        match self.try_run_static_chunk(input_ids, position_ids, seq_len, write_index) {
            Ok(()) => Ok(()),
            Err(first_err) if self.mode == StaticCacheBindingMode::InPlaceAlias => {
                self.enable_handle_swap()?;
                self.try_run_static_chunk(input_ids, position_ids, seq_len, write_index)
                    .map_err(|second_err| {
                        OrtError::InvalidArgument(format!(
                            "static-cache in-place alias run failed ({first_err}); handle-swap fallback also failed ({second_err})"
                        ))
                    })
            }
            Err(err) => Err(err),
        }
    }

    fn try_run_static_chunk(
        &mut self,
        input_ids: &[i64],
        position_ids: &[i64],
        seq_len: usize,
        write_index: usize,
    ) -> Result<()> {
        let batch = self.batch_size;
        let input_ids_value = Value::from_slice_i64(input_ids, &[batch, seq_len as i64])?;
        let position_ids_value = if self.signature.has_position_ids {
            if position_ids.len() != input_ids.len() {
                return Err(OrtError::InvalidArgument(
                    "position_ids length must match input_ids length".into(),
                ));
            }
            Some(Value::from_slice_i64(
                position_ids,
                &[batch, seq_len as i64],
            )?)
        } else {
            None
        };
        let write_indices =
            Value::from_slice_i64(&vec![write_index as i64; batch as usize], &[batch])?;
        let nonpad_kv_seqlen = Value::from_slice_i64(
            &vec![(write_index + seq_len) as i64; batch as usize],
            &[batch],
        )?;

        self.binding.clear()?;
        for input in self.session.inputs() {
            match input.name.as_str() {
                "input_ids" => self.binding.bind_input(&input.name, &input_ids_value)?,
                "position_ids" => {
                    let Some(position_ids_value) = position_ids_value.as_ref() else {
                        return Err(OrtError::InvalidArgument(
                            "model requires position_ids but none were prepared".into(),
                        ));
                    };
                    self.binding.bind_input(&input.name, position_ids_value)?;
                }
                "write_indices" => self.binding.bind_input(&input.name, &write_indices)?,
                "nonpad_kv_seqlen" => self.binding.bind_input(&input.name, &nonpad_kv_seqlen)?,
                name => {
                    let Some(buffer) = self.buffers.iter().find(|buffer| buffer.input_name == name)
                    else {
                        return Err(OrtError::InvalidArgument(format!(
                            "unsupported static-cache input '{}'",
                            input.name
                        )));
                    };
                    self.binding.bind_input(&input.name, &buffer.current)?;
                }
            }
        }

        let mut borrowed_outputs = Vec::new();
        for output in self.session.output_names() {
            if let Some(buffer) = self
                .buffers
                .iter()
                .find(|buffer| buffer.output_name == *output)
            {
                let output_value = match self.mode {
                    StaticCacheBindingMode::InPlaceAlias => &buffer.current,
                    StaticCacheBindingMode::HandleSwap => {
                        buffer.alternate.as_ref().ok_or_else(|| {
                            OrtError::InvalidArgument(format!(
                                "missing static-cache alternate buffer for '{}'",
                                buffer.output_name
                            ))
                        })?
                    }
                };
                borrowed_outputs.push(output_value.raw_ptr_addr());
                self.binding.bind_output(output, output_value)?;
            } else {
                self.binding
                    .bind_output_to_device(output, &MemoryInfo::cpu()?)?;
            }
        }

        self.session.run_with_binding(&self.binding)?;
        if self.mode == StaticCacheBindingMode::HandleSwap {
            for buffer in &mut self.buffers {
                let alternate = buffer.alternate.as_mut().ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "missing static-cache alternate buffer for '{}'",
                        buffer.output_name
                    ))
                })?;
                std::mem::swap(&mut buffer.current, alternate);
            }
        }
        Ok(())
    }

    fn last_logits(&self) -> Result<Value> {
        let borrowed_outputs = self
            .buffers
            .iter()
            .flat_map(|buffer| {
                std::iter::once(buffer.current.raw_ptr_addr())
                    .chain(buffer.alternate.as_ref().map(|value| value.raw_ptr_addr()))
            })
            .collect::<Vec<_>>();
        let outputs = self.binding.output_values_or_borrowed(&borrowed_outputs)?;
        for (name, value) in self.session.output_names().iter().zip(outputs) {
            if is_logits_output(name) {
                return value.ok_or_else(|| {
                    OrtError::InvalidArgument("logits unexpectedly aliased a KV buffer".into())
                });
            }
        }
        Err(OrtError::InvalidArgument(
            "model did not produce logits".into(),
        ))
    }

    fn enable_handle_swap(&mut self) -> Result<()> {
        for buffer in &mut self.buffers {
            if buffer.alternate.is_none() {
                buffer.alternate = Some(Arc::new(zeroed_value(
                    buffer.current.shape(),
                    buffer.current.dtype(),
                )?));
            }
        }
        self.mode = StaticCacheBindingMode::HandleSwap;
        Ok(())
    }
}

impl<'a> BatchedStaticCacheDecodeSession<'a> {
    /// Detect a STATIC-CACHE/TensorScatter signature from ONNX graph I/O.
    pub fn detect(session: &Session) -> Result<Option<StaticCacheSignature>> {
        StaticCacheDecodeSession::detect(session)
    }

    /// Create a batched static-cache decode session with all rows active at
    /// cursor 0.
    pub fn new(session: &'a Session, options: StaticCacheDecodeOptions) -> Result<Self> {
        let (signature, pairs) = detect_static_cache(session)?.ok_or_else(|| {
            OrtError::InvalidArgument(
                "model does not expose static-cache key_cache/write_indices inputs".into(),
            )
        })?;
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
        let buffers = allocate_static_cache_buffers(options.batch_size, &pairs)?;
        let logical_to_physical = (0..batch_size).map(Some).collect::<Vec<_>>();
        let physical_to_logical = (0..batch_size).map(Some).collect::<Vec<_>>();
        Ok(Self {
            session,
            binding: IoBinding::new(session)?,
            signature,
            batch_size,
            row_lens: vec![0; batch_size],
            active: vec![true; batch_size],
            logical_to_physical,
            physical_to_logical,
            mode: StaticCacheBindingMode::InPlaceAlias,
            buffers,
        })
    }

    pub fn signature(&self) -> &StaticCacheSignature {
        &self.signature
    }

    pub fn binding_mode(&self) -> StaticCacheBindingMode {
        self.mode
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn max_len(&self) -> usize {
        self.signature.max_len
    }

    pub fn row_len(&self, row: usize) -> Result<usize> {
        self.check_row(row)?;
        Ok(self.row_lens[row])
    }

    pub fn row_lens(&self) -> &[usize] {
        &self.row_lens
    }

    pub fn is_active(&self, row: usize) -> Result<bool> {
        self.check_row(row)?;
        Ok(self.active[row])
    }

    /// Physical slot currently holding a logical row, if that row is assigned.
    pub fn physical_slot(&self, row: usize) -> Result<Option<usize>> {
        self.check_row(row)?;
        Ok(self.logical_to_physical[row])
    }

    /// Logical row id currently held by a physical slot, if any.
    pub fn logical_row_for_physical_slot(&self, slot: usize) -> Result<Option<usize>> {
        self.check_row(slot)?;
        Ok(self.physical_to_logical[slot])
    }

    /// Number of rows that will participate in an active-only step.
    pub fn active_batch_size(&self) -> usize {
        self.active.iter().filter(|&&active| active).count()
    }

    /// Fraction of the fixed batch skipped by an active-only step after compaction.
    pub fn inactive_compute_fraction(&self) -> f32 {
        if self.batch_size == 0 {
            0.0
        } else {
            (self.batch_size - self.active_batch_size()) as f32 / self.batch_size as f32
        }
    }

    /// Active logical rows in the physical order used by active-only logits.
    pub fn active_rows(&self) -> Vec<usize> {
        self.physical_to_logical
            .iter()
            .filter_map(|row| row.and_then(|row| self.active[row].then_some(row)))
            .collect()
    }

    /// Mark a row inactive. It remains assigned until `compact` packs active
    /// rows to the prefix and frees inactive physical slots.
    pub fn deactivate_row(&mut self, row: usize) -> Result<()> {
        self.check_row(row)?;
        self.active[row] = false;
        Ok(())
    }

    /// Mark a retained row active without modifying its KV contents or cursor.
    pub fn activate_row(&mut self, row: usize) -> Result<()> {
        self.check_row(row)?;
        if self.logical_to_physical[row].is_none() {
            return Err(OrtError::InvalidArgument(format!(
                "row {row} is not assigned to a physical slot; call assign_row/admit_row first"
            )));
        }
        self.active[row] = true;
        Ok(())
    }

    /// Reset one row's KV region and cursor, then mark it active for a new
    /// agent/session.
    pub fn assign_row(&mut self, row: usize) -> Result<()> {
        self.check_row(row)?;
        let physical = match self.logical_to_physical[row] {
            Some(physical) => physical,
            None => self.free_physical_slot().ok_or_else(|| {
                OrtError::InvalidArgument(format!(
                    "no free physical slot available to assign row {row}; deactivate and compact first"
                ))
            })?,
        };
        self.logical_to_physical[row] = Some(physical);
        self.physical_to_logical[physical] = Some(row);
        self.binding.clear()?;
        for buffer in &mut self.buffers {
            Arc::get_mut(&mut buffer.current)
                .ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "static-cache buffer '{}' is still borrowed",
                        buffer.input_name
                    ))
                })?
                .zero_rank3_row(physical)?;
            if let Some(alternate) = buffer.alternate.as_mut() {
                Arc::get_mut(alternate)
                    .ok_or_else(|| {
                        OrtError::InvalidArgument(format!(
                            "static-cache alternate buffer '{}' is still borrowed",
                            buffer.output_name
                        ))
                    })?
                    .zero_rank3_row(physical)?;
            }
        }
        self.row_lens[row] = 0;
        self.active[row] = true;
        Ok(())
    }

    /// Alias for [`Self::assign_row`] that names the continuous-batching admit
    /// operation Sebastian's manager will call for a recycled logical row id.
    pub fn admit_row(&mut self, row: usize) -> Result<()> {
        self.assign_row(row)
    }

    /// Replace the active logical row set and compact it in the provided order.
    pub fn set_active_rows(&mut self, rows: &[usize]) -> Result<()> {
        let mut seen = vec![false; self.batch_size];
        for &row in rows {
            self.check_row(row)?;
            if self.logical_to_physical[row].is_none() {
                return Err(OrtError::InvalidArgument(format!(
                    "row {row} is not assigned to a physical slot"
                )));
            }
            if std::mem::replace(&mut seen[row], true) {
                return Err(OrtError::InvalidArgument(format!(
                    "row {row} appears more than once in active set"
                )));
            }
        }
        self.active.fill(false);
        for &row in rows {
            self.active[row] = true;
        }
        self.compact_active_rows_in_order(rows)
    }

    /// Pack active logical rows into physical slots `0..active_count`.
    ///
    /// ORT IoBinding binds whole OrtValues, not gathered batch-dimension views,
    /// so active-only execution uses compaction plus prefix aliases. The copy is
    /// `active_count * MAX_LEN * KV_DIM` per KV tensor when rows move, paid only
    /// when membership/order changes; subsequent decode steps avoid fixed-B
    /// model compute for inactive rows.
    pub fn compact(&mut self) -> Result<usize> {
        let rows = self.active_rows();
        self.compact_active_rows_in_order(&rows)?;
        Ok(rows.len())
    }

    /// Rewind one row's logical write cursor. Stale suffix slots are ignored by
    /// later `nonpad_kv_seqlen` values and overwritten by future writes.
    pub fn rewind_row(&mut self, row: usize, target_len: usize) -> Result<()> {
        self.check_row(row)?;
        if target_len > self.row_lens[row] {
            return Err(OrtError::InvalidArgument(format!(
                "cannot rewind row {row} from {} to larger length {target_len}",
                self.row_lens[row]
            )));
        }
        self.row_lens[row] = target_len;
        Ok(())
    }

    /// Runtime-owned KV buffer identities and sizes.
    pub fn buffer_infos(&self) -> Result<Vec<StaticCacheBufferInfo>> {
        self.buffers
            .iter()
            .map(|buffer| {
                Ok(StaticCacheBufferInfo {
                    input_name: buffer.input_name.clone(),
                    output_name: buffer.output_name.clone(),
                    shape: buffer.current.shape().to_vec(),
                    dtype: buffer.current.dtype(),
                    data_ptr: buffer.current.data_ptr_addr()?,
                    numel: buffer.current.numel(),
                })
            })
            .collect()
    }

    /// Scatter a same-length chunk for every active row and return `[B, S, V]`
    /// logits. Inactive rows receive the provided dummy ids but their row cursor
    /// and `nonpad_kv_seqlen` are left unchanged.
    pub fn prefill(&mut self, input_ids: &[i64], position_ids: &[i64]) -> Result<Value> {
        let seq_len = self.seq_len_from_flat_input(input_ids)?;
        self.run_batched_static_chunk(input_ids, position_ids, seq_len, None)?;
        self.last_logits()
    }

    /// Scatter one token per active row at each row's current cursor.
    pub fn step(&mut self, next_token_ids: &[i64], position_ids: &[i64]) -> Result<Value> {
        self.run_batched_static_chunk(next_token_ids, position_ids, 1, None)?;
        self.last_logits()
    }

    /// Scatter one token per row, advancing only rows where `advance_rows[row]`
    /// is true and the row is active. This is useful for ragged prompt prefill
    /// and continuous-batch join/leave tests.
    pub fn step_select(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
        advance_rows: &[bool],
    ) -> Result<Value> {
        self.run_batched_static_chunk(next_token_ids, position_ids, 1, Some(advance_rows))?;
        self.last_logits()
    }

    /// Scatter one token per active row after compacting active rows to the
    /// physical prefix. Inputs and returned logits are ordered by
    /// [`Self::active_rows`], and the returned tensor has shape
    /// `[active_count, 1, vocab]`.
    pub fn step_active(&mut self, next_token_ids: &[i64], position_ids: &[i64]) -> Result<Value> {
        self.run_active_static_chunk(next_token_ids, position_ids, 1, None)
    }

    /// Active-only variant of [`Self::step_select`]. `advance_active_rows` is
    /// indexed in active-row order, not fixed logical-row order.
    pub fn step_active_select(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
        advance_active_rows: &[bool],
    ) -> Result<Value> {
        self.run_active_static_chunk(next_token_ids, position_ids, 1, Some(advance_active_rows))
    }

    /// Extract logits for one row/sequence position from a `[B, S, vocab]`
    /// logits tensor.
    pub fn row_logits(logits: &Value, row: usize, seq_index: usize) -> Result<Vec<f32>> {
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
        if row >= batch || seq_index >= seq_len {
            return Err(OrtError::InvalidArgument(format!(
                "logits row/seq ({row}, {seq_index}) out of range for shape {:?}",
                logits.shape()
            )));
        }
        let data = logits.to_vec_f32()?;
        let start = (row * seq_len + seq_index) * vocab;
        Ok(data[start..start + vocab].to_vec())
    }

    fn seq_len_from_flat_input(&self, input_ids: &[i64]) -> Result<usize> {
        if input_ids.is_empty() || !input_ids.len().is_multiple_of(self.batch_size) {
            return Err(OrtError::InvalidArgument(format!(
                "input_ids length {} is not a non-empty multiple of batch {}",
                input_ids.len(),
                self.batch_size
            )));
        }
        Ok(input_ids.len() / self.batch_size)
    }

    fn run_batched_static_chunk(
        &mut self,
        input_ids: &[i64],
        position_ids: &[i64],
        seq_len: usize,
        advance_rows: Option<&[bool]>,
    ) -> Result<()> {
        if let Some(advance_rows) = advance_rows
            && advance_rows.len() != self.batch_size
        {
            return Err(OrtError::InvalidArgument(format!(
                "advance_rows length {} does not match batch {}",
                advance_rows.len(),
                self.batch_size
            )));
        }
        let advances = (0..self.batch_size)
            .map(|row| self.active[row] && advance_rows.is_none_or(|mask| mask[row]))
            .collect::<Vec<_>>();
        for (row, advance) in advances.iter().copied().enumerate() {
            if advance && self.row_lens[row] + seq_len > self.signature.max_len {
                return Err(OrtError::InvalidArgument(format!(
                    "row {row} static-cache write {}..{} exceeds capacity {}",
                    self.row_lens[row],
                    self.row_lens[row] + seq_len,
                    self.signature.max_len
                )));
            }
            if advance && self.logical_to_physical[row].is_none() {
                return Err(OrtError::InvalidArgument(format!(
                    "active row {row} is not assigned to a physical slot"
                )));
            }
        }
        match self.try_run_batched_static_chunk(input_ids, position_ids, seq_len, &advances) {
            Ok(()) => {
                for (row, advance) in advances.into_iter().enumerate() {
                    if advance {
                        self.row_lens[row] += seq_len;
                    }
                }
                Ok(())
            }
            Err(first_err) if self.mode == StaticCacheBindingMode::InPlaceAlias => {
                self.enable_handle_swap()?;
                self.try_run_batched_static_chunk(input_ids, position_ids, seq_len, &advances)
                    .map_err(|second_err| {
                        OrtError::InvalidArgument(format!(
                            "batched static-cache in-place alias run failed ({first_err}); handle-swap fallback also failed ({second_err})"
                        ))
                    })?;
                for (row, advance) in advances.into_iter().enumerate() {
                    if advance {
                        self.row_lens[row] += seq_len;
                    }
                }
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    fn run_active_static_chunk(
        &mut self,
        input_ids: &[i64],
        position_ids: &[i64],
        seq_len: usize,
        advance_active_rows: Option<&[bool]>,
    ) -> Result<Value> {
        self.compact()?;
        let rows = self.active_rows();
        if rows.is_empty() {
            return Err(OrtError::InvalidArgument(
                "active-only static-cache step requires at least one active row".into(),
            ));
        }
        if let Some(advance_active_rows) = advance_active_rows
            && advance_active_rows.len() != rows.len()
        {
            return Err(OrtError::InvalidArgument(format!(
                "advance_active_rows length {} does not match active batch {}",
                advance_active_rows.len(),
                rows.len()
            )));
        }
        if input_ids.len() != rows.len() * seq_len {
            return Err(OrtError::InvalidArgument(format!(
                "input_ids length {} does not match [active={}, S={}]",
                input_ids.len(),
                rows.len(),
                seq_len
            )));
        }
        let advances = rows
            .iter()
            .enumerate()
            .map(|(index, _)| advance_active_rows.is_none_or(|mask| mask[index]))
            .collect::<Vec<_>>();
        for (&row, &advance) in rows.iter().zip(&advances) {
            if advance && self.row_lens[row] + seq_len > self.signature.max_len {
                return Err(OrtError::InvalidArgument(format!(
                    "row {row} static-cache write {}..{} exceeds capacity {}",
                    self.row_lens[row],
                    self.row_lens[row] + seq_len,
                    self.signature.max_len
                )));
            }
        }

        match self.try_run_active_static_chunk(input_ids, position_ids, seq_len, &rows, &advances) {
            Ok(logits) => {
                for (&row, advance) in rows.iter().zip(advances) {
                    if advance {
                        self.row_lens[row] += seq_len;
                    }
                }
                Ok(logits)
            }
            Err(first_err) if self.mode == StaticCacheBindingMode::InPlaceAlias => {
                self.enable_handle_swap()?;
                let logits = self
                    .try_run_active_static_chunk(input_ids, position_ids, seq_len, &rows, &advances)
                    .map_err(|second_err| {
                        OrtError::InvalidArgument(format!(
                            "active static-cache in-place alias run failed ({first_err}); handle-swap fallback also failed ({second_err})"
                        ))
                    })?;
                for (&row, advance) in rows.iter().zip(advances) {
                    if advance {
                        self.row_lens[row] += seq_len;
                    }
                }
                Ok(logits)
            }
            Err(err) => Err(err),
        }
    }

    fn try_run_active_static_chunk(
        &mut self,
        input_ids: &[i64],
        position_ids: &[i64],
        seq_len: usize,
        rows: &[usize],
        advances: &[bool],
    ) -> Result<Value> {
        let batch = rows.len() as i64;
        let input_ids_value = Value::from_slice_i64(input_ids, &[batch, seq_len as i64])?;
        let position_ids_value = if self.signature.has_position_ids {
            if position_ids.len() != input_ids.len() {
                return Err(OrtError::InvalidArgument(
                    "position_ids length must match input_ids length".into(),
                ));
            }
            Some(Value::from_slice_i64(
                position_ids,
                &[batch, seq_len as i64],
            )?)
        } else {
            None
        };
        let write_indices = rows
            .iter()
            .map(|&row| self.row_lens[row] as i64)
            .collect::<Vec<_>>();
        let nonpad_kv_seqlen = rows
            .iter()
            .zip(advances)
            .map(|(&row, &advance)| {
                (if advance {
                    self.row_lens[row] + seq_len
                } else {
                    self.row_lens[row]
                }) as i64
            })
            .collect::<Vec<_>>();
        let write_indices = Value::from_slice_i64(&write_indices, &[batch])?;
        let nonpad_kv_seqlen = Value::from_slice_i64(&nonpad_kv_seqlen, &[batch])?;

        struct PrefixBinding {
            input_name: String,
            output_name: String,
            input: Value,
            output: Option<Value>,
        }

        let mut prefix_bindings = Vec::with_capacity(self.buffers.len());
        for buffer in &self.buffers {
            let shape = [batch, buffer.current.shape()[1], buffer.current.shape()[2]];
            let input = Value::alias_with_shape(Arc::clone(&buffer.current), &shape)?;
            let output = match self.mode {
                StaticCacheBindingMode::InPlaceAlias => None,
                StaticCacheBindingMode::HandleSwap => {
                    let alternate = buffer.alternate.as_ref().ok_or_else(|| {
                        OrtError::InvalidArgument(format!(
                            "missing static-cache alternate buffer for '{}'",
                            buffer.output_name
                        ))
                    })?;
                    Some(Value::alias_with_shape(Arc::clone(alternate), &shape)?)
                }
            };
            prefix_bindings.push(PrefixBinding {
                input_name: buffer.input_name.clone(),
                output_name: buffer.output_name.clone(),
                input,
                output,
            });
        }

        self.binding.clear()?;
        for input in self.session.inputs() {
            match input.name.as_str() {
                "input_ids" => self.binding.bind_input(&input.name, &input_ids_value)?,
                "position_ids" => {
                    let Some(position_ids_value) = position_ids_value.as_ref() else {
                        return Err(OrtError::InvalidArgument(
                            "model requires position_ids but none were prepared".into(),
                        ));
                    };
                    self.binding.bind_input(&input.name, position_ids_value)?;
                }
                "write_indices" => self.binding.bind_input(&input.name, &write_indices)?,
                "nonpad_kv_seqlen" => self.binding.bind_input(&input.name, &nonpad_kv_seqlen)?,
                name => {
                    let Some(binding) = prefix_bindings
                        .iter()
                        .find(|binding| binding.input_name == name)
                    else {
                        return Err(OrtError::InvalidArgument(format!(
                            "unsupported static-cache input '{}'",
                            input.name
                        )));
                    };
                    self.binding.bind_input(&input.name, &binding.input)?;
                }
            }
        }

        let mut borrowed_outputs = Vec::new();
        for output in self.session.output_names() {
            if let Some(binding) = prefix_bindings
                .iter()
                .find(|binding| binding.output_name == *output)
            {
                let output_value = binding.output.as_ref().unwrap_or(&binding.input);
                borrowed_outputs.push(output_value.raw_ptr_addr());
                self.binding.bind_output(output, output_value)?;
            } else {
                self.binding
                    .bind_output_to_device(output, &MemoryInfo::cpu()?)?;
            }
        }

        self.session.run_with_binding(&self.binding)?;
        let outputs = self.binding.output_values_or_borrowed(&borrowed_outputs)?;
        if self.mode == StaticCacheBindingMode::HandleSwap {
            for buffer in &mut self.buffers {
                let alternate = buffer.alternate.as_mut().ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "missing static-cache alternate buffer for '{}'",
                        buffer.output_name
                    ))
                })?;
                std::mem::swap(&mut buffer.current, alternate);
            }
        }
        for (name, value) in self.session.output_names().iter().zip(outputs) {
            if is_logits_output(name) {
                return value.ok_or_else(|| {
                    OrtError::InvalidArgument("logits unexpectedly aliased a KV buffer".into())
                });
            }
        }
        Err(OrtError::InvalidArgument(
            "model did not produce logits".into(),
        ))
    }

    fn try_run_batched_static_chunk(
        &mut self,
        input_ids: &[i64],
        position_ids: &[i64],
        seq_len: usize,
        advances: &[bool],
    ) -> Result<()> {
        let batch = self.batch_size as i64;
        if input_ids.len() != self.batch_size * seq_len {
            return Err(OrtError::InvalidArgument(format!(
                "input_ids length {} does not match [B={}, S={}]",
                input_ids.len(),
                self.batch_size,
                seq_len
            )));
        }
        let mut physical_input_ids = vec![0_i64; input_ids.len()];
        let mut physical_position_ids = if self.signature.has_position_ids {
            if position_ids.len() != input_ids.len() {
                return Err(OrtError::InvalidArgument(
                    "position_ids length must match input_ids length".into(),
                ));
            }
            vec![0_i64; position_ids.len()]
        } else {
            Vec::new()
        };
        for physical in 0..self.batch_size {
            let Some(logical) = self.physical_to_logical[physical] else {
                continue;
            };
            let src = logical * seq_len;
            let dst = physical * seq_len;
            physical_input_ids[dst..dst + seq_len].copy_from_slice(&input_ids[src..src + seq_len]);
            if self.signature.has_position_ids {
                physical_position_ids[dst..dst + seq_len]
                    .copy_from_slice(&position_ids[src..src + seq_len]);
            }
        }
        let input_ids_value = Value::from_slice_i64(&physical_input_ids, &[batch, seq_len as i64])?;
        let position_ids_value = if self.signature.has_position_ids {
            Some(Value::from_slice_i64(
                &physical_position_ids,
                &[batch, seq_len as i64],
            )?)
        } else {
            None
        };
        let write_indices = (0..self.batch_size)
            .map(|physical| {
                self.physical_to_logical[physical]
                    .map(|row| self.row_lens[row])
                    .unwrap_or(0) as i64
            })
            .collect::<Vec<_>>();
        let nonpad_kv_seqlen = (0..self.batch_size)
            .map(|physical| {
                let Some(row) = self.physical_to_logical[physical] else {
                    return 0_i64;
                };
                if advances[row] {
                    (self.row_lens[row] + seq_len) as i64
                } else {
                    self.row_lens[row] as i64
                }
            })
            .collect::<Vec<_>>();
        let write_indices = Value::from_slice_i64(&write_indices, &[batch])?;
        let nonpad_kv_seqlen = Value::from_slice_i64(&nonpad_kv_seqlen, &[batch])?;

        self.binding.clear()?;
        for input in self.session.inputs() {
            match input.name.as_str() {
                "input_ids" => self.binding.bind_input(&input.name, &input_ids_value)?,
                "position_ids" => {
                    let Some(position_ids_value) = position_ids_value.as_ref() else {
                        return Err(OrtError::InvalidArgument(
                            "model requires position_ids but none were prepared".into(),
                        ));
                    };
                    self.binding.bind_input(&input.name, position_ids_value)?;
                }
                "write_indices" => self.binding.bind_input(&input.name, &write_indices)?,
                "nonpad_kv_seqlen" => self.binding.bind_input(&input.name, &nonpad_kv_seqlen)?,
                name => {
                    let Some(buffer) = self.buffers.iter().find(|buffer| buffer.input_name == name)
                    else {
                        return Err(OrtError::InvalidArgument(format!(
                            "unsupported static-cache input '{}'",
                            input.name
                        )));
                    };
                    self.binding.bind_input(&input.name, &buffer.current)?;
                }
            }
        }

        let mut borrowed_outputs = Vec::new();
        for output in self.session.output_names() {
            if let Some(buffer) = self
                .buffers
                .iter()
                .find(|buffer| buffer.output_name == *output)
            {
                let output_value = match self.mode {
                    StaticCacheBindingMode::InPlaceAlias => &buffer.current,
                    StaticCacheBindingMode::HandleSwap => {
                        buffer.alternate.as_ref().ok_or_else(|| {
                            OrtError::InvalidArgument(format!(
                                "missing static-cache alternate buffer for '{}'",
                                buffer.output_name
                            ))
                        })?
                    }
                };
                borrowed_outputs.push(output_value.raw_ptr_addr());
                self.binding.bind_output(output, output_value)?;
            } else {
                self.binding
                    .bind_output_to_device(output, &MemoryInfo::cpu()?)?;
            }
        }

        self.session.run_with_binding(&self.binding)?;
        if self.mode == StaticCacheBindingMode::HandleSwap {
            for buffer in &mut self.buffers {
                let alternate = buffer.alternate.as_mut().ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "missing static-cache alternate buffer for '{}'",
                        buffer.output_name
                    ))
                })?;
                std::mem::swap(&mut buffer.current, alternate);
            }
        }
        Ok(())
    }

    fn last_logits(&self) -> Result<Value> {
        let borrowed_outputs = self
            .buffers
            .iter()
            .flat_map(|buffer| {
                std::iter::once(buffer.current.raw_ptr_addr())
                    .chain(buffer.alternate.as_ref().map(|value| value.raw_ptr_addr()))
            })
            .collect::<Vec<_>>();
        let outputs = self.binding.output_values_or_borrowed(&borrowed_outputs)?;
        for (name, value) in self.session.output_names().iter().zip(outputs) {
            if is_logits_output(name) {
                return value.ok_or_else(|| {
                    OrtError::InvalidArgument("logits unexpectedly aliased a KV buffer".into())
                });
            }
        }
        Err(OrtError::InvalidArgument(
            "model did not produce logits".into(),
        ))
    }

    fn enable_handle_swap(&mut self) -> Result<()> {
        for buffer in &mut self.buffers {
            if buffer.alternate.is_none() {
                buffer.alternate = Some(Arc::new(zeroed_value(
                    buffer.current.shape(),
                    buffer.current.dtype(),
                )?));
            }
        }
        self.mode = StaticCacheBindingMode::HandleSwap;
        Ok(())
    }

    fn compact_active_rows_in_order(&mut self, rows: &[usize]) -> Result<()> {
        let source_slots = rows
            .iter()
            .map(|&row| {
                self.check_row(row)?;
                if !self.active[row] {
                    return Err(OrtError::InvalidArgument(format!(
                        "row {row} is not active"
                    )));
                }
                self.logical_to_physical[row].ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "row {row} is not assigned to a physical slot"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if source_slots
            .iter()
            .copied()
            .enumerate()
            .all(|(target, source)| target == source)
            && self
                .physical_to_logical
                .iter()
                .enumerate()
                .all(|(physical, row)| physical < rows.len() || row.is_none())
        {
            return Ok(());
        }

        self.binding.clear()?;
        for buffer in &mut self.buffers {
            Arc::get_mut(&mut buffer.current)
                .ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "static-cache buffer '{}' is still borrowed",
                        buffer.input_name
                    ))
                })?
                .pack_rank3_rows_to_prefix(&source_slots)?;
            if let Some(alternate) = buffer.alternate.as_mut() {
                Arc::get_mut(alternate)
                    .ok_or_else(|| {
                        OrtError::InvalidArgument(format!(
                            "static-cache alternate buffer '{}' is still borrowed",
                            buffer.output_name
                        ))
                    })?
                    .pack_rank3_rows_to_prefix(&source_slots)?;
            }
        }

        let mut logical_to_physical = vec![None; self.batch_size];
        let mut physical_to_logical = vec![None; self.batch_size];
        for (physical, &row) in rows.iter().enumerate() {
            logical_to_physical[row] = Some(physical);
            physical_to_logical[physical] = Some(row);
        }
        self.logical_to_physical = logical_to_physical;
        self.physical_to_logical = physical_to_logical;
        Ok(())
    }

    fn free_physical_slot(&self) -> Option<usize> {
        self.physical_to_logical.iter().position(Option::is_none)
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

fn infer_kv_pairs(session: &Session) -> Result<Vec<KvPair>> {
    let input_names = session.input_names();
    let mut pairs = Vec::new();
    for output in session.outputs() {
        if !is_present_output(&output.name) {
            continue;
        }
        let Some(suffix) = kv_suffix(&output.name) else {
            continue;
        };
        let Some(past_name) = input_names
            .iter()
            .find(|input| kv_suffix(input).as_deref() == Some(suffix.as_str()))
        else {
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

fn is_present_output(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("present") && (lower.contains("key") || lower.contains("value"))
}

fn kv_suffix(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    for prefix in [
        "past_key_values.",
        "present_key_values.",
        "past.",
        "present.",
    ] {
        if let Some(suffix) = lower.strip_prefix(prefix) {
            return Some(suffix.to_string());
        }
    }
    None
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
