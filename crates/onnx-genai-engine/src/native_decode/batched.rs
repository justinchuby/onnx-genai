//! Native continuous-batch decode session (design §J.4/§J.5 P2e).
//!
//! [`NativeBatchedDecodeSession`] is the native, production `BatchedDecodeSession`
//! implementation the `ContinuousBatchManager` drives to carry **mixed adapters
//! in one batch**. Unlike the ORT `BatchedStaticCacheDecodeSession` — whose
//! `libonnxruntime.so` session has no `GroupedLoraDelta` op — this session runs
//! on our native [`InferenceSession`], which owns the grouped-LoRA custom op. So
//! its `set_lora_routes` really binds `lora.segments` and each physical row's
//! logits carry that row's own adapter delta.
//!
//! # KV scope (honest)
//!
//! The native CPU execution provider has no `TensorScatter` kernel, so a native
//! mirror of the ORT static-cache TensorScatter contract (independent per-row
//! write cursors into one `[batch, max_len, kv_dim]` buffer) is out of scope in
//! this pass. This session therefore supports:
//!
//! * **KV-free** grouped/projection models fully (the mixed-adapter routing
//!   surface — every decode step runs the active/physical batch in one
//!   `InferenceSession::run` with a per-row `lora.segments`). This is the
//!   production-reachable surface the continuous-batch manager exercises.
//! * **Batched decode-with-past (Concat past/present) in strict lockstep**, where
//!   every active row shares one sequence-length cursor. Ragged advance (a
//!   mid-flight prefill that grows one row while others hold) **fails loud**,
//!   because a shared-sequence-dimension Concat cache cannot hold independent
//!   per-row cursors and native has no scatter kernel. Ragged/KV continuous
//!   batching stays deferred (design §J.9).
//!
//! The base / single-adapter fast path is preserved byte-for-byte: routing is
//! engaged only when the graph declares a `lora.segments` input and a caller
//! actually feeds routes.

use super::*;
use onnx_genai_ort::decode::BatchedDecodeSession;
use onnx_genai_ort::{OrtError, Result as OrtResult, Value};

/// The `lora.segments` route id for a base-only row — the kernel's null route.
const BASE_LORA_ROUTE: i32 = -1;

/// A native, production `BatchedDecodeSession` backed by a grouped-capable
/// [`InferenceSession`]. Borrows the session (like the ORT sessions borrow their
/// `Session`) and owns the per-row batch/KV bookkeeping.
pub struct NativeBatchedDecodeSession<'session> {
    session: &'session mut InferenceSession,
    /// Declared token-ids graph input (`input_ids` / `decoder_input_ids`).
    token_input: String,
    /// Optional declared `attention_mask` graph input.
    attention_mask_input: Option<String>,
    /// Optional declared `position_ids` graph input.
    position_ids_input: Option<String>,
    /// Declared logits graph output.
    logits_output: String,
    /// Vocabulary width (last logits dimension), resolved on the first run.
    vocab: Option<usize>,
    /// Growable past KV input names paired to their present outputs. Empty for a
    /// KV-free projection model (the routing surface).
    kv_inputs: Vec<String>,
    present_to_past: HashMap<String, String>,
    /// Batched past KV tensors (`[batch, ..., length, head_dim]`), reused across
    /// steps. Present only in the batched decode-with-past lockstep mode.
    past: HashMap<String, Tensor>,
    /// The `lora.segments` graph input name (design §J), or `None` for a
    /// base-only / non-grouped session (routing dormant).
    lora_segments_input: Option<String>,
    batch_size: usize,
    max_len: usize,
    /// Logical token length per physical row.
    row_lens: Vec<usize>,
    /// Whether each physical row participates in an active-only step.
    active: Vec<bool>,
    /// Per-row grouped-LoRA route for the NEXT step, in the ordering the upcoming
    /// step consumes (physical-row indexed for `step_select`/prefill, active-row
    /// ordered for `step_active`). `None` when no routes were fed since the last
    /// step (the base fast path).
    pending_routes: Option<Vec<i32>>,
    /// Reused little-endian `Int32` `lora.segments` payload, refilled in place so
    /// the routing payload is not reallocated per step.
    segments_scratch: Vec<u8>,
}

impl<'session> NativeBatchedDecodeSession<'session> {
    /// Build a native batched decode session over a grouped-capable
    /// `InferenceSession`. `batch_size` is the fixed number of physical rows and
    /// `max_len` the logical KV capacity. Fails loud when the graph exposes no
    /// token input or no logits output.
    pub fn new(
        session: &'session mut InferenceSession,
        batch_size: usize,
        max_len: usize,
    ) -> anyhow::Result<Self> {
        if batch_size == 0 {
            bail!("native batched decode requires batch_size > 0");
        }
        let input_names = session
            .inputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let output_names = session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();

        let token_input = input_names
            .iter()
            .find(|name| matches!(name.as_str(), "input_ids" | "decoder_input_ids"))
            .cloned()
            .context(
                "native batched decode requires an `input_ids` / `decoder_input_ids` graph input",
            )?;
        let attention_mask_input = input_names
            .iter()
            .find(|name| name.as_str() == "attention_mask")
            .cloned();
        let position_ids_input = input_names
            .iter()
            .find(|name| name.as_str() == "position_ids")
            .cloned();
        let logits_output = output_names
            .iter()
            .find(|name| name.as_str() == "logits")
            .or_else(|| {
                // Single non-KV output → treat it as logits.
                let mut non_kv = output_names.iter().filter(|name| !is_present_name(name));
                let first = non_kv.next();
                if non_kv.next().is_none() { first } else { None }
            })
            .cloned()
            .context("native batched decode could not resolve a logits output")?;

        let kv_inputs = input_names
            .iter()
            .filter(|name| is_past_name(name))
            .cloned()
            .collect::<Vec<_>>();
        let mut present_to_past = HashMap::new();
        for output in output_names.iter().filter(|name| is_present_name(name)) {
            if let Some(past) = matching_past_name(output, &kv_inputs) {
                present_to_past.insert(output.clone(), past);
            }
        }
        if present_to_past.len() != kv_inputs.len() {
            bail!(
                "native batched decode has incomplete past/present KV pairs; past inputs: {kv_inputs:?}, present outputs matched: {}",
                present_to_past.len()
            );
        }

        let lora_segments_input = session.lora_segments_input().map(str::to_string);

        Ok(Self {
            session,
            token_input,
            attention_mask_input,
            position_ids_input,
            logits_output,
            vocab: None,
            kv_inputs,
            present_to_past,
            past: HashMap::new(),
            lora_segments_input,
            batch_size,
            max_len,
            row_lens: vec![0; batch_size],
            active: vec![true; batch_size],
            pending_routes: None,
            segments_scratch: Vec::new(),
        })
    }

    fn check_row(&self, row: usize) -> OrtResult<()> {
        if row >= self.batch_size {
            return Err(OrtError::InvalidArgument(format!(
                "row {row} out of range for native batch {}",
                self.batch_size
            )));
        }
        Ok(())
    }

    /// Build the `lora.segments` tensor for `rows` activation rows from the routes
    /// fed by [`Self::set_lora_routes`], reusing the little-endian scratch buffer.
    /// Returns `Ok(None)` when this session declares no `lora.segments` input, so
    /// the base fast path binds nothing extra. Fails loud when the graph expects
    /// routing but the fed route count does not match the rows this step runs.
    fn build_segments_tensor(&mut self, rows: usize) -> OrtResult<Option<Tensor>> {
        let Some(_) = self.lora_segments_input.as_ref() else {
            return Ok(None);
        };
        let routes = self.pending_routes.take();
        let routes = match routes {
            Some(routes) => routes,
            // No routes fed since the last step: every row is base-only. This
            // keeps a grouped-capable graph correct when the manager runs a
            // base-only batch (all rows route to the null adapter).
            None => vec![BASE_LORA_ROUTE; rows],
        };
        if routes.len() != rows {
            return Err(OrtError::InvalidArgument(format!(
                "native batched grouped-LoRA routes ({}) do not match the {rows} activation rows this step runs",
                routes.len()
            )));
        }
        self.segments_scratch.clear();
        self.segments_scratch.reserve(rows * 4);
        for route in &routes {
            self.segments_scratch.extend_from_slice(&route.to_le_bytes());
        }
        let tensor = Tensor::from_raw(DataType::Int32, vec![rows], &self.segments_scratch)
            .map_err(|e| {
                OrtError::InvalidArgument(format!("build native lora.segments tensor: {e}"))
            })?;
        Ok(Some(tensor))
    }

    /// Run the model over `physical_rows` (each an owning physical slot index),
    /// feeding one token per row plus optional mask/position/KV/segments inputs,
    /// and return `[rows, 1, vocab]` logits in the order of `physical_rows`.
    ///
    /// `carry_kv` selects the batched decode-with-past lockstep path (all rows
    /// share the sequence cursor); otherwise the run is KV-free.
    fn run_rows(
        &mut self,
        physical_rows: &[usize],
        token_ids: &[i64],
        position_ids: &[i64],
    ) -> OrtResult<Value> {
        let rows = physical_rows.len();
        if rows == 0 {
            return Err(OrtError::InvalidArgument(
                "native batched decode step requires at least one row".into(),
            ));
        }
        if token_ids.len() != rows || position_ids.len() != rows {
            return Err(OrtError::InvalidArgument(format!(
                "native batched decode step got {} tokens / {} positions for {rows} rows",
                token_ids.len(),
                position_ids.len()
            )));
        }
        let carry_kv = !self.kv_inputs.is_empty();
        // Lockstep KV requires every participating row at the same length.
        let past_len = if carry_kv {
            let first = self.row_lens[physical_rows[0]];
            if physical_rows
                .iter()
                .any(|&row| self.row_lens[row] != first)
            {
                return Err(OrtError::InvalidArgument(
                    "native batched decode-with-past requires all participating rows at one \
                     sequence length (ragged per-row KV cursors need the deferred native \
                     static-cache scatter path; design §J.9)"
                        .into(),
                ));
            }
            first
        } else {
            0
        };
        let total_len = past_len + 1;

        let segments = self.build_segments_tensor(rows)?;

        // Owned step-input tensors kept alive for the borrowed `run` call.
        let mut owned: Vec<(String, Tensor)> = Vec::with_capacity(4 + self.kv_inputs.len());
        let token_tensor = Tensor::from_i64(&[rows, 1], token_ids)
            .map_err(|e| OrtError::InvalidArgument(format!("native batched input_ids: {e}")))?;
        owned.push((self.token_input.clone(), token_tensor));
        if let Some(name) = self.attention_mask_input.clone() {
            let mask = vec![1_i64; rows * total_len];
            let mask_tensor = Tensor::from_i64(&[rows, total_len], &mask).map_err(|e| {
                OrtError::InvalidArgument(format!("native batched attention_mask: {e}"))
            })?;
            owned.push((name, mask_tensor));
        }
        if let Some(name) = self.position_ids_input.clone() {
            let position_tensor = Tensor::from_i64(&[rows, 1], position_ids).map_err(|e| {
                OrtError::InvalidArgument(format!("native batched position_ids: {e}"))
            })?;
            owned.push((name, position_tensor));
        }
        if carry_kv {
            for name in self.kv_inputs.clone() {
                let tensor = match self.past.remove(&name) {
                    Some(tensor) => tensor,
                    None => self.empty_batched_past(&name, rows)?,
                };
                owned.push((name, tensor));
            }
        }
        if let (Some(name), Some(tensor)) = (self.lora_segments_input.clone(), segments) {
            owned.push((name, tensor));
        }

        let bindings = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let outputs = self
            .session
            .run(&bindings)
            .map_err(|e| OrtError::InvalidArgument(format!("native batched forward pass: {e}")))?;
        if outputs.len() != self.session.outputs().len() {
            return Err(OrtError::InvalidArgument(format!(
                "native batched decoder returned {} outputs, graph declares {}",
                outputs.len(),
                self.session.outputs().len()
            )));
        }

        let mut logits_tensor = None;
        let mut next_past = HashMap::with_capacity(self.kv_inputs.len());
        for (metadata, tensor) in self.session.outputs().iter().zip(outputs) {
            if metadata.name == self.logits_output {
                logits_tensor = Some(tensor);
            } else if carry_kv {
                if let Some(past) = self.present_to_past.get(&metadata.name) {
                    next_past.insert(past.clone(), tensor);
                }
            }
        }
        let logits_tensor = logits_tensor.ok_or_else(|| {
            OrtError::InvalidArgument(format!(
                "native batched decoder omitted logits output '{}'",
                self.logits_output
            ))
        })?;
        if carry_kv {
            for past in self.present_to_past.values() {
                if !next_past.contains_key(past) {
                    return Err(OrtError::InvalidArgument(format!(
                        "native batched decoder omitted a present output for past '{past}'"
                    )));
                }
            }
            self.past = next_past;
        }

        self.reshape_logits(logits_tensor, rows)
    }

    /// Reshape a raw logits tensor into a `[rows, 1, vocab]` `Value`. The raw
    /// tensor may be `[rows, vocab]` or `[rows, 1, vocab]`; its element count must
    /// be a multiple of `rows`.
    fn reshape_logits(&mut self, tensor: Tensor, rows: usize) -> OrtResult<Value> {
        let data = tensor.to_vec_f32();
        if data.is_empty() || !data.len().is_multiple_of(rows) {
            return Err(OrtError::InvalidArgument(format!(
                "native batched logits length {} is not a positive multiple of rows {rows}",
                data.len()
            )));
        }
        let vocab = data.len() / rows;
        match self.vocab {
            Some(previous) if previous != vocab => {
                return Err(OrtError::InvalidArgument(format!(
                    "native batched logits vocab changed from {previous} to {vocab}"
                )));
            }
            _ => self.vocab = Some(vocab),
        }
        Value::from_slice_f32(&data, &[rows as i64, 1, vocab as i64])
    }

    /// Build a zero-length batched past KV tensor `[rows, ..., 0, head_dim]` for a
    /// fresh decode-with-past sequence set.
    fn empty_batched_past(&self, name: &str, rows: usize) -> OrtResult<Tensor> {
        let meta = self
            .session
            .inputs()
            .iter()
            .find(|meta| meta.name == name)
            .ok_or_else(|| {
                OrtError::InvalidArgument(format!("missing native KV metadata for '{name}'"))
            })?;
        if meta.shape.len() < 3 {
            return Err(OrtError::InvalidArgument(format!(
                "native KV input '{name}' has rank {} < 3",
                meta.shape.len()
            )));
        }
        let seq_axis = meta.shape.len() - 2;
        let mut shape = Vec::with_capacity(meta.shape.len());
        for (axis, dim) in meta.shape.iter().enumerate() {
            let value = if axis == 0 {
                rows
            } else if axis == seq_axis {
                0
            } else if let Dim::Static(value) = dim {
                *value
            } else {
                return Err(OrtError::InvalidArgument(format!(
                    "native KV input '{name}' axis {axis} is symbolic and is neither batch nor sequence"
                )));
            };
            shape.push(value);
        }
        let bytes = vec![0_u8; meta.dtype.storage_bytes(shape.iter().product::<usize>()).max(1)];
        Tensor::from_raw(meta.dtype, shape, &bytes)
            .map_err(|e| OrtError::InvalidArgument(format!("native empty batched KV '{name}': {e}")))
    }
}

impl<'session> BatchedDecodeSession<'session> for NativeBatchedDecodeSession<'session> {
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn max_len(&self) -> usize {
        self.max_len
    }

    fn row_len(&self, row: usize) -> OrtResult<usize> {
        self.check_row(row)?;
        Ok(self.row_lens[row])
    }

    fn active_rows(&self) -> Vec<usize> {
        (0..self.batch_size)
            .filter(|&row| self.active[row])
            .collect()
    }

    fn deactivate_row(&mut self, row: usize) -> OrtResult<()> {
        self.check_row(row)?;
        self.active[row] = false;
        Ok(())
    }

    fn assign_row(&mut self, row: usize) -> OrtResult<()> {
        self.check_row(row)?;
        self.active[row] = true;
        self.row_lens[row] = 0;
        // A freshly assigned row starts a new sequence; drop any carried batched
        // KV so a stale prefix cannot leak into the new sequence set. The next
        // decode-with-past step reseeds an empty batched cache.
        if !self.kv_inputs.is_empty() {
            self.past.clear();
        }
        Ok(())
    }

    fn step_select(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
        advance_rows: &[bool],
    ) -> OrtResult<Value> {
        if next_token_ids.len() != self.batch_size
            || position_ids.len() != self.batch_size
            || advance_rows.len() != self.batch_size
        {
            return Err(OrtError::InvalidArgument(format!(
                "native step_select expects batch_size {} entries (got tokens {}, positions {}, advance {})",
                self.batch_size,
                next_token_ids.len(),
                position_ids.len(),
                advance_rows.len()
            )));
        }
        let physical_rows = (0..self.batch_size).collect::<Vec<_>>();
        let logits = self.run_rows(&physical_rows, next_token_ids, position_ids)?;
        for row in 0..self.batch_size {
            if self.active[row] && advance_rows[row] {
                self.row_lens[row] += 1;
            }
        }
        Ok(logits)
    }

    fn step_active(&mut self, next_token_ids: &[i64], position_ids: &[i64]) -> OrtResult<Value> {
        let active_rows = self.active_rows();
        if active_rows.is_empty() {
            return Err(OrtError::InvalidArgument(
                "native step_active requires at least one active row".into(),
            ));
        }
        if next_token_ids.len() != active_rows.len() || position_ids.len() != active_rows.len() {
            return Err(OrtError::InvalidArgument(format!(
                "native step_active expects {} active-row entries (got tokens {}, positions {})",
                active_rows.len(),
                next_token_ids.len(),
                position_ids.len()
            )));
        }
        let logits = self.run_rows(&active_rows, next_token_ids, position_ids)?;
        for &row in &active_rows {
            self.row_lens[row] += 1;
        }
        Ok(logits)
    }

    fn set_lora_routes(&mut self, routes: &[i32]) -> OrtResult<()> {
        if self.lora_segments_input.is_none() {
            // No `lora.segments` input: routing is dormant, base fast path.
            return Ok(());
        }
        let mut buffer = self.pending_routes.take().unwrap_or_default();
        buffer.clear();
        buffer.extend_from_slice(routes);
        self.pending_routes = Some(buffer);
        Ok(())
    }
}
