use super::*;

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
        let kv_pairs = infer_kv_pairs(session, None)?;
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
