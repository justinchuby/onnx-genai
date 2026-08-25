use super::*;

pub(super) struct StaticCachePair {
    pub(super) key_input: TensorInfo,
    pub(super) value_input: TensorInfo,
    pub(super) key_output: String,
    pub(super) value_output: String,
}

pub(super) struct StaticCacheBuffer {
    pub(super) input_name: String,
    pub(super) output_name: String,
    pub(super) current: Arc<Value>,
    pub(super) alternate: Option<Arc<Value>>,
}

/// Resolve the static-cache logits output name-agnostically.
///
/// A static-cache graph exposes exactly one output that is not a runtime-owned
/// KV cache buffer (`updated_*`); that output is the logits. It is selected by
/// excluding the resolved cache output ports, never by interpreting the port
/// name.
fn resolve_static_cache_logits_output(
    session: &Session,
    buffers: &[StaticCacheBuffer],
) -> Result<String> {
    let cache_outputs = buffers
        .iter()
        .map(|buffer| buffer.output_name.as_str())
        .collect::<std::collections::HashSet<_>>();
    logits_output_by_exclusion(session.output_names(), &cache_outputs)
}

/// Select the sole output that is not a runtime-owned KV cache buffer.
fn logits_output_by_exclusion(
    output_names: &[String],
    cache_outputs: &std::collections::HashSet<&str>,
) -> Result<String> {
    let mut non_cache = output_names
        .iter()
        .filter(|name| !cache_outputs.contains(name.as_str()));
    let logits = non_cache.next().cloned().ok_or_else(|| {
        OrtError::InvalidArgument(
            "static-cache model exposes no non-cache output to read logits from".into(),
        )
    })?;
    if non_cache.next().is_some() {
        return Err(OrtError::InvalidArgument(
            "static-cache model exposes multiple non-cache outputs, so logits_output is \
             ambiguous; give the port the logits role in pipeline.workflow.components.<component>.ports.roles"
                .into(),
        ));
    }
    Ok(logits)
}

/// Stateful decode runner for Mobius/STATIC-CACHE TensorScatter models.
///
/// The runtime owns fixed `[B, MAX_LEN, KV_DIM]` key/value buffers. The model's
/// `updated_*` outputs are bound back onto those buffers; the graph scatter is a
/// write hint, not the source of truth for cache ownership.
pub struct StaticCacheDecodeSession<'a> {
    session: &'a Session,
    binding: IoBinding<'a>,
    signature: StaticCacheSignature,
    batch_size: i64,
    current_len: usize,
    mode: StaticCacheBindingMode,
    buffers: Vec<StaticCacheBuffer>,
    logits_output: String,
    abi: StaticCacheAbi,
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
    binding: IoBinding<'a>,
    signature: StaticCacheSignature,
    batch_size: usize,
    row_lens: Vec<usize>,
    active: Vec<bool>,
    logical_to_physical: Vec<Option<usize>>,
    physical_to_logical: Vec<Option<usize>>,
    mode: StaticCacheBindingMode,
    buffers: Vec<StaticCacheBuffer>,
    logits_output: String,
    abi: StaticCacheAbi,
}

impl<'a> StaticCacheDecodeSession<'a> {
    /// Detect a STATIC-CACHE/TensorScatter signature from ONNX graph I/O.
    pub fn detect(
        session: &Session,
        io: Option<&onnx_genai_metadata::DecoderAbi>,
    ) -> Result<Option<StaticCacheSignature>> {
        Ok(detect_static_cache(session, io)?.map(|(signature, ..)| signature))
    }

    /// Create a static-cache decode session if the graph exposes the signature.
    pub fn new(
        session: &'a Session,
        options: StaticCacheDecodeOptions,
        io: Option<&onnx_genai_metadata::DecoderAbi>,
    ) -> Result<Self> {
        let (signature, pairs, abi) = detect_static_cache(session, io)?.ok_or_else(|| {
            OrtError::InvalidArgument(
                "model does not expose static-cache key_cache/write_indices inputs".into(),
            )
        })?;
        let buffers = allocate_static_cache_buffers(options.batch_size, &pairs)?;
        let logits_output = resolve_static_cache_logits_output(session, &buffers)?;
        Ok(Self {
            session,
            binding: IoBinding::new(session)?,
            signature,
            batch_size: options.batch_size,
            current_len: 0,
            mode: StaticCacheBindingMode::InPlaceAlias,
            buffers,
            logits_output,
            abi,
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
            match self.abi.classify(&input.name) {
                Some(StaticCacheInputRole::Token) => {
                    self.binding.bind_input(&input.name, &input_ids_value)?
                }
                Some(StaticCacheInputRole::Position) => {
                    let Some(position_ids_value) = position_ids_value.as_ref() else {
                        return Err(OrtError::InvalidArgument(
                            "model requires position_ids but none were prepared".into(),
                        ));
                    };
                    self.binding.bind_input(&input.name, position_ids_value)?;
                }
                Some(StaticCacheInputRole::WriteIndices) => {
                    self.binding.bind_input(&input.name, &write_indices)?
                }
                Some(StaticCacheInputRole::KvSequenceLength) => {
                    self.binding.bind_input(&input.name, &nonpad_kv_seqlen)?
                }
                None => {
                    let Some(buffer) = self
                        .buffers
                        .iter()
                        .find(|buffer| buffer.input_name == input.name)
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
            if name == &self.logits_output {
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
    pub fn detect(
        session: &Session,
        io: Option<&onnx_genai_metadata::DecoderAbi>,
    ) -> Result<Option<StaticCacheSignature>> {
        StaticCacheDecodeSession::detect(session, io)
    }

    /// Create a batched static-cache decode session with all rows active at
    /// cursor 0.
    pub fn new(
        session: &'a Session,
        options: StaticCacheDecodeOptions,
        io: Option<&onnx_genai_metadata::DecoderAbi>,
    ) -> Result<Self> {
        let (signature, pairs, abi) = detect_static_cache(session, io)?.ok_or_else(|| {
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
        let logits_output = resolve_static_cache_logits_output(session, &buffers)?;
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
            logits_output,
            abi,
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
            match self.abi.classify(&input.name) {
                Some(StaticCacheInputRole::Token) => {
                    self.binding.bind_input(&input.name, &input_ids_value)?
                }
                Some(StaticCacheInputRole::Position) => {
                    let Some(position_ids_value) = position_ids_value.as_ref() else {
                        return Err(OrtError::InvalidArgument(
                            "model requires position_ids but none were prepared".into(),
                        ));
                    };
                    self.binding.bind_input(&input.name, position_ids_value)?;
                }
                Some(StaticCacheInputRole::WriteIndices) => {
                    self.binding.bind_input(&input.name, &write_indices)?
                }
                Some(StaticCacheInputRole::KvSequenceLength) => {
                    self.binding.bind_input(&input.name, &nonpad_kv_seqlen)?
                }
                None => {
                    let Some(binding) = prefix_bindings
                        .iter()
                        .find(|binding| binding.input_name == input.name)
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
            if name == &self.logits_output {
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
            match self.abi.classify(&input.name) {
                Some(StaticCacheInputRole::Token) => {
                    self.binding.bind_input(&input.name, &input_ids_value)?
                }
                Some(StaticCacheInputRole::Position) => {
                    let Some(position_ids_value) = position_ids_value.as_ref() else {
                        return Err(OrtError::InvalidArgument(
                            "model requires position_ids but none were prepared".into(),
                        ));
                    };
                    self.binding.bind_input(&input.name, position_ids_value)?;
                }
                Some(StaticCacheInputRole::WriteIndices) => {
                    self.binding.bind_input(&input.name, &write_indices)?
                }
                Some(StaticCacheInputRole::KvSequenceLength) => {
                    self.binding.bind_input(&input.name, &nonpad_kv_seqlen)?
                }
                None => {
                    let Some(buffer) = self
                        .buffers
                        .iter()
                        .find(|buffer| buffer.input_name == input.name)
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
            if name == &self.logits_output {
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

impl<'a> BatchedDecodeSession<'a> for BatchedStaticCacheDecodeSession<'a> {
    fn batch_size(&self) -> usize {
        BatchedStaticCacheDecodeSession::batch_size(self)
    }
    fn max_len(&self) -> usize {
        BatchedStaticCacheDecodeSession::max_len(self)
    }
    fn row_len(&self, row: usize) -> Result<usize> {
        BatchedStaticCacheDecodeSession::row_len(self, row)
    }
    fn active_rows(&self) -> Vec<usize> {
        BatchedStaticCacheDecodeSession::active_rows(self)
    }
    fn deactivate_row(&mut self, row: usize) -> Result<()> {
        BatchedStaticCacheDecodeSession::deactivate_row(self, row)
    }
    fn assign_row(&mut self, row: usize) -> Result<()> {
        BatchedStaticCacheDecodeSession::assign_row(self, row)
    }
    fn step_select(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
        advance_rows: &[bool],
    ) -> Result<crate::decode::BatchStepLogits> {
        BatchedStaticCacheDecodeSession::step_select(
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
        BatchedStaticCacheDecodeSession::step_active(self, next_token_ids, position_ids)
            .map(crate::decode::BatchStepLogits::Ort)
    }
}

#[cfg(test)]
mod tests {
    use super::logits_output_by_exclusion;
    use std::collections::HashSet;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|name| name.to_string()).collect()
    }

    /// The logits output is the sole output that is not a runtime-owned KV cache
    /// buffer, selected without interpreting any port name.
    #[test]
    fn logits_is_the_unique_non_cache_output() {
        let outputs = names(&[
            "opaque_scores",
            "updated_key_cache.0",
            "updated_value_cache.0",
        ]);
        let cache: HashSet<&str> = ["updated_key_cache.0", "updated_value_cache.0"]
            .into_iter()
            .collect();
        assert_eq!(
            logits_output_by_exclusion(&outputs, &cache).unwrap(),
            "opaque_scores"
        );
    }

    #[test]
    fn no_non_cache_output_is_an_error() {
        let outputs = names(&["updated_key_cache.0"]);
        let cache: HashSet<&str> = ["updated_key_cache.0"].into_iter().collect();
        let error = logits_output_by_exclusion(&outputs, &cache).unwrap_err();
        assert!(
            format!("{error:?}").contains("no non-cache output"),
            "{error:?}"
        );
    }

    #[test]
    fn multiple_non_cache_outputs_require_metadata() {
        let outputs = names(&["scores", "hidden", "updated_key_cache.0"]);
        let cache: HashSet<&str> = ["updated_key_cache.0"].into_iter().collect();
        let error = logits_output_by_exclusion(&outputs, &cache).unwrap_err();
        assert!(format!("{error:?}").contains("logits_output"), "{error:?}");
    }
}
