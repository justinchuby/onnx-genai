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
    binding: IoBinding<'a>,
    kv_pairs: Vec<KvPair>,
    kv_buffers: HashMap<String, Arc<Value>>,
    kv_allocator: Option<crate::Allocator<'a>>,
    batch_size: usize,
    max_len: usize,
    row_lens: Vec<usize>,
    active: Vec<bool>,
    token_input: String,
    attention_mask_input: String,
    position_ids_input: Option<String>,
    logits_output: String,
}

/// Resolved decode-step graph-port roles for a shared-buffer batched session.
///
/// Every role is selected from the package's declared port roles or an unambiguous
/// tensor-shape signal; graph port names are never interpreted.
#[derive(Debug)]
struct SharedBufferRoles {
    token_input: String,
    attention_mask_input: String,
    position_ids_input: Option<String>,
    logits_output: String,
}

impl SharedBufferRoles {
    fn resolve(
        session: &Session,
        io: Option<&onnx_genai_metadata::DecoderAbi>,
        kv_pairs: &[KvPair],
    ) -> Result<Self> {
        Self::resolve_from_ports(session.inputs(), session.outputs(), io, kv_pairs)
    }

    fn resolve_from_ports(
        inputs: &[TensorInfo],
        outputs: &[TensorInfo],
        io: Option<&onnx_genai_metadata::DecoderAbi>,
        kv_pairs: &[KvPair],
    ) -> Result<Self> {
        use crate::io_roles::{
            is_rank_one_or_two_sequence, is_rank_one_to_three_output, resolve_port,
        };
        let input_excluded = kv_pairs
            .iter()
            .map(|pair| pair.past.as_str())
            .chain(
                [
                    io.and_then(|io| io.attention_mask_input.as_deref()),
                    io.and_then(|io| io.position_ids_input.as_deref()),
                ]
                .into_iter()
                .flatten(),
            )
            .collect::<std::collections::HashSet<_>>();
        let resolve_input =
            |declared: Option<&str>, role: &str, structural: fn(&TensorInfo) -> bool| {
                resolve_port(inputs, declared, role, |tensor| {
                    !input_excluded.contains(tensor.name.as_str()) && structural(tensor)
                })
                .map_err(OrtError::InvalidArgument)
                .map(|port| port.map(|port| port.name))
            };
        let never = |_: &TensorInfo| false;
        let token_input = resolve_input(
            io.and_then(|io| io.token_input.as_deref()),
            "token_input",
            is_rank_one_or_two_sequence,
        )?
        .ok_or_else(|| {
            OrtError::InvalidArgument(
                "cannot resolve token_input from tensor shape; give the port the token_ids role \
                 in pipeline.workflow.components.<component>.ports.roles"
                    .into(),
            )
        })?;
        // The attention mask is shape-ambiguous against other integer sequence
        // inputs, so it is only ever taken from explicit metadata.
        let attention_mask_input = resolve_input(
            io.and_then(|io| io.attention_mask_input.as_deref()),
            "attention_mask_input",
            never,
        )?
        .ok_or_else(|| {
            OrtError::InvalidArgument(
                "shared-buffer batching derives each row's KV length from an attention mask, so \
                 attention_mask_input must resolve; give the port the attention_mask role in \
                 pipeline.workflow.components.<component>.ports.roles"
                    .into(),
            )
        })?;
        let position_ids_input = resolve_input(
            io.and_then(|io| io.position_ids_input.as_deref()),
            "position_ids_input",
            never,
        )?;
        let present_excluded = kv_pairs
            .iter()
            .map(|pair| pair.present.as_str())
            .collect::<std::collections::HashSet<_>>();
        let logits_output = resolve_port(
            outputs,
            io.and_then(|io| io.logits_output.as_deref()),
            "logits_output",
            |tensor| {
                !present_excluded.contains(tensor.name.as_str())
                    && is_rank_one_to_three_output(tensor)
            },
        )
        .map_err(OrtError::InvalidArgument)?
        .map(|port| port.name)
        .ok_or_else(|| {
            OrtError::InvalidArgument(
                "cannot resolve logits_output from tensor shape; give the port the logits role in \
                 pipeline.workflow.components.<component>.ports.roles"
                    .into(),
            )
        })?;
        Ok(Self {
            token_input,
            attention_mask_input,
            position_ids_input,
            logits_output,
        })
    }
}

impl<'a> BatchedSharedBufferDecodeSession<'a> {
    /// Create a batched share-buffer decode session with all rows active at
    /// cursor 0. KV buffers are allocated once as `[batch, kv_heads, max_len,
    /// head_dim]` on the session's device allocator when available.
    ///
    /// Graph-port roles (token, attention-mask, position, logits) and the
    /// past/present KV pairs are resolved from the declared state group or an
    /// unambiguous tensor-shape signal; graph port names are never interpreted.
    pub fn new(session: &'a Session, options: SharedBufferBatchOptions) -> Result<Self> {
        Self::new_with_io(session, options, None)
    }

    /// Create a batched share-buffer decode session using declarative
    /// declared graph-port roles when present.
    pub fn new_with_io(
        session: &'a Session,
        options: SharedBufferBatchOptions,
        io: Option<&onnx_genai_metadata::DecoderAbi>,
    ) -> Result<Self> {
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
        let kv_pairs = infer_kv_pairs(session, io)?;
        if kv_pairs.is_empty() {
            return Err(OrtError::InvalidArgument(
                "shared-buffer batching requires declared past/present KV pairs; declare \
                 kv_inputs and kv_outputs"
                    .into(),
            ));
        }
        let roles = SharedBufferRoles::resolve(session, io, &kv_pairs)?;
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
            token_input: roles.token_input,
            attention_mask_input: roles.attention_mask_input,
            position_ids_input: roles.position_ids_input,
            logits_output: roles.logits_output,
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

    /// Rewind one row's logical cursor and activity-visible cache residency.
    /// Stale KV beyond the restored cursor remains physically allocated but is
    /// excluded by the row's next attention mask and overwritten before use.
    pub fn rewind_row(&mut self, row: usize, target_len: usize) -> Result<()> {
        self.check_row(row)?;
        if target_len > self.row_lens[row] {
            return Err(OrtError::InvalidArgument(format!(
                "cannot rewind shared-buffer row {row} from {} to larger length {target_len}",
                self.row_lens[row]
            )));
        }
        self.row_lens[row] = target_len;
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
            if self.position_ids_input.is_some() && active_index < position_ids.len() {
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

        let position_ids_value = if self.position_ids_input.is_some() {
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
            if input.name == self.token_input {
                self.binding
                    .bind_input(&input.name, &input_ids_value)
                    .map_err(|e| {
                        OrtError::InvalidArgument(format!("bind input_ids '{}': {e}", input.name))
                    })?;
            } else if input.name == self.attention_mask_input {
                self.binding
                    .bind_input(&input.name, &attention_mask_value)
                    .map_err(|e| {
                        OrtError::InvalidArgument(format!(
                            "bind attention_mask '{}': {e}",
                            input.name
                        ))
                    })?;
            } else if let Some(position_ids_value) = position_ids_value.as_ref()
                && self.position_ids_input.as_deref() == Some(input.name.as_str())
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
            if name == &self.logits_output {
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
    fn snapshot_row(&mut self, row: usize) -> Result<crate::decode::BatchedRowSnapshot> {
        Ok(crate::decode::BatchedRowSnapshot::new(
            row,
            self.row_len(row)?,
            self.is_active(row)?,
        ))
    }
    fn restore_row(
        &mut self,
        row: usize,
        snapshot: &crate::decode::BatchedRowSnapshot,
    ) -> Result<()> {
        snapshot.validate_row(row)?;
        self.rewind_row(row, snapshot.logical_len())?;
        self.active[row] = snapshot.active();
        Ok(())
    }
    fn step_select(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
        advance_rows: &[bool],
    ) -> Result<crate::decode::BatchStepLogits> {
        BatchedSharedBufferDecodeSession::step_select(
            self,
            next_token_ids,
            position_ids,
            advance_rows,
        )
        .map(crate::decode::BatchStepLogits::Ort)
    }
    fn step_active(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
    ) -> Result<crate::decode::BatchStepLogits> {
        BatchedSharedBufferDecodeSession::step_active(self, next_token_ids, position_ids)
            .map(crate::decode::BatchStepLogits::Ort)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataType, TensorInfo};
    use onnx_genai_metadata::DecoderAbi;

    fn tensor(name: &str, dtype: DataType, shape: &[i64]) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    fn kv_pair(past: &str, present: &str) -> KvPair {
        KvPair {
            past: past.to_string(),
            present: present.to_string(),
            input: tensor(past, DataType::Float32, &[1, 2, 4, 8]),
            seq_axis: 2,
        }
    }

    fn io_with_roles() -> DecoderAbi {
        serde_json::from_str(
            r#"{
                "token_input": "opaque_tokens",
                "attention_mask_input": "opaque_mask",
                "position_ids_input": "opaque_positions",
                "logits_output": "opaque_scores"
            }"#,
        )
        .expect("io spec")
    }

    /// Non-standard port names must resolve purely from explicit metadata; graph
    /// port names are never interpreted.
    #[test]
    fn explicit_metadata_resolves_non_standard_names() {
        let inputs = vec![
            tensor("opaque_tokens", DataType::Int64, &[-1, -1]),
            tensor("opaque_mask", DataType::Int64, &[-1, -1]),
            tensor("opaque_positions", DataType::Int64, &[-1, -1]),
            tensor("layer0.past", DataType::Float32, &[1, 2, 4, 8]),
        ];
        let outputs = vec![
            tensor("opaque_scores", DataType::Float32, &[-1, -1, 32]),
            tensor("layer0.present", DataType::Float32, &[1, 2, 4, 8]),
        ];
        let pairs = vec![kv_pair("layer0.past", "layer0.present")];
        let io = io_with_roles();
        let roles =
            SharedBufferRoles::resolve_from_ports(&inputs, &outputs, Some(&io), &pairs).unwrap();
        assert_eq!(roles.token_input, "opaque_tokens");
        assert_eq!(roles.attention_mask_input, "opaque_mask");
        assert_eq!(
            roles.position_ids_input.as_deref(),
            Some("opaque_positions")
        );
        assert_eq!(roles.logits_output, "opaque_scores");
    }

    /// A single unambiguous rank-two integer sequence input and a single
    /// score-shaped output resolve by shape, with no attention mask declared.
    #[test]
    fn unique_shape_resolves_without_names_or_metadata() {
        let inputs = vec![
            tensor("weird_name", DataType::Int64, &[-1, -1]),
            tensor("kv.past", DataType::Float32, &[1, 2, 4, 8]),
        ];
        let outputs = vec![
            tensor("scores", DataType::Float32, &[-1, -1, 32]),
            tensor("kv.present", DataType::Float32, &[1, 2, 4, 8]),
        ];
        let pairs = vec![kv_pair("kv.past", "kv.present")];
        // The attention mask is required for shared-buffer batching and is never
        // shape-resolved, so an absent declaration is a clear error.
        let error =
            SharedBufferRoles::resolve_from_ports(&inputs, &outputs, None, &pairs).unwrap_err();
        assert!(
            format!("{error:?}").contains("attention_mask_input"),
            "{error:?}"
        );
    }

    /// Multiple shape-ambiguous integer sequence inputs must fail with an
    /// actionable error naming the metadata key to declare.
    #[test]
    fn ambiguous_token_input_requires_metadata() {
        let inputs = vec![
            tensor("input_ids", DataType::Int64, &[-1, -1]),
            tensor("attention_mask", DataType::Int64, &[-1, -1]),
            tensor("kv.past", DataType::Float32, &[1, 2, 4, 8]),
        ];
        let outputs = vec![
            tensor("logits", DataType::Float32, &[-1, -1, 32]),
            tensor("kv.present", DataType::Float32, &[1, 2, 4, 8]),
        ];
        let pairs = vec![kv_pair("kv.past", "kv.present")];
        let error =
            SharedBufferRoles::resolve_from_ports(&inputs, &outputs, None, &pairs).unwrap_err();
        assert!(format!("{error:?}").contains("token_input"), "{error:?}");
    }
}
