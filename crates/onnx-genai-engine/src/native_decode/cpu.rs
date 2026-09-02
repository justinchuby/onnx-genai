use super::*;

pub(crate) enum NativeCpuDecodeResult {
    Logits(Vec<Vec<f32>>),
    Token(TokenId),
}

/// Default persistent CPU KV cache capacity (sequence positions) when the
/// `ONNX_GENAI_CPU_KV_MAX_LEN` override is unset. Matches the CUDA default.
pub(crate) const DEFAULT_CPU_KV_MAX_LEN: usize = 4096;

/// Persistent in-place CPU KV cache — the CPU analogue of [`DecodeCudaState`]'s
/// present==past device bindings.
///
/// Each growable attention KV pair is bound to ONE persistent host buffer of
/// full physical capacity `[1, H, max_len, Dh]`, with the present output aliased
/// onto the past input. The CPU GroupQueryAttention kernel detects that aliasing
/// and appends each step's K/V rows in place (see `group_query_attention.rs`),
/// so the decode loop no longer re-feeds the growing past as freshly-copied host
/// inputs nor round-trips the full present cache to host every step. `input_ids`,
/// `attention_mask` and `position_ids` remain ordinary per-step host inputs, and
/// `logits`/`hidden` still materialize normally, so only the KV traffic changes.
pub(crate) struct DecodeCpuKvState {
    /// One present==past binding per growable KV pair, sorted by past name.
    bindings: Vec<DeviceIoBinding>,
    /// `(present, past)` port names, index-aligned with `bindings`, retained so
    /// a grown cache can be re-bound to the same ports.
    pairs: Vec<(String, String)>,
    /// Element type of every KV binding. `new` currently binds only `Float32`
    /// caches, but the growth path sizes its copies from this rather than
    /// assuming 4 bytes, so widening that gate does not silently corrupt it.
    dtype: DataType,
    pub(crate) max_len: usize,
    logical_len: usize,
}

impl DecodeCpuKvState {
    /// Build persistent CPU KV bindings, or `Ok(None)` when the model is not
    /// eligible for the in-place path (any KV input that is not rank-4 f32 with a
    /// static head geometry, e.g. an f16 cache). `present_to_past` must contain
    /// only growable attention KV pairs — recurrent state pairs are handled by
    /// the host copy path and their presence disables this fast path upstream.
    pub(crate) fn new(
        session: &mut InferenceSession,
        present_to_past: &HashMap<String, String>,
        max_len: usize,
    ) -> anyhow::Result<Option<Self>> {
        let mut pairs = present_to_past
            .iter()
            .map(|(present, past)| (present.clone(), past.clone()))
            .collect::<Vec<_>>();
        pairs.sort_unstable_by(|left, right| left.1.cmp(&right.1));
        // The append-only in-place path lives ONLY in the CPU GroupQueryAttention
        // kernel (see `group_query_attention.rs::detect_inplace_kv`). Binding
        // present==past for a cache consumed by any other op (e.g. a plain Concat)
        // would not append in place and could corrupt output, so require every
        // bound past KV input to feed a GroupQueryAttention node; otherwise leave
        // the model on the safe host copy path.
        let past_names = pairs
            .iter()
            .map(|(_, past)| past.clone())
            .collect::<Vec<_>>();
        if !all_pasts_consumed_by_gqa(session.graph(), &past_names) {
            return Ok(None);
        }
        let mut bindings = Vec::with_capacity(pairs.len());
        let mut bound_pairs = Vec::with_capacity(pairs.len());
        let mut kv_dtype = None;
        for (present, past) in pairs {
            let meta = session
                .inputs()
                .iter()
                .find(|meta| meta.name == past)
                .with_context(|| format!("missing native KV input metadata for '{past}'"))?;
            // Only contiguous rank-4 f32 caches take the in-place path; anything
            // else (f16, non-rank-4, dynamic non-seq dims) disables it so the
            // model keeps the correct host copy path with no regression.
            if meta.dtype != DataType::Float32 || meta.shape.len() != 4 {
                return Ok(None);
            }
            let mut physical_shape = Vec::with_capacity(4);
            for (axis, dim) in meta.shape.iter().copied().enumerate() {
                let value = if axis == 0 {
                    1
                } else if axis == 2 {
                    max_len
                } else if let Dim::Static(value) = dim {
                    value
                } else {
                    return Ok(None);
                };
                physical_shape.push(value);
            }
            let mut logical_shape = physical_shape.clone();
            logical_shape[2] = 0;
            kv_dtype = Some(meta.dtype);
            bindings.push(session.allocate_device_binding(
                past.clone(),
                Some(present.clone()),
                meta.dtype,
                physical_shape,
                logical_shape,
            )?);
            bound_pairs.push((present, past));
        }
        if bindings.is_empty() {
            return Ok(None);
        }
        let dtype = kv_dtype.context("bound KV pairs without an element type")?;
        Ok(Some(Self {
            bindings,
            pairs: bound_pairs,
            dtype,
            max_len,
            logical_len: 0,
        }))
    }

    /// Grow every KV binding so the cache can hold at least `needed` positions,
    /// preserving the first `live` positions already written.
    ///
    /// Capacity is the *third* axis of a `[B, H, capacity, Dh]` cache, not the
    /// outermost one, so a larger capacity relocates every head: head `i` starts
    /// at `i * capacity * Dh`. The live prefix is therefore carried over as one
    /// contiguous run per `(batch, head)` block rather than a single memcpy, and
    /// the trailing slots of each block are left untouched because only the
    /// first `logical_len` rows of a block are ever read back.
    ///
    /// Growth doubles (bounded below by `needed`), so re-binding is amortised
    /// O(1) per token and a decode that runs to length `n` copies O(n) rows in
    /// total rather than O(n) per step. Each old buffer is staged to host and
    /// released before its replacement is allocated, so peak RSS is
    /// `old + live_prefix` rather than `old + new`.
    ///
    /// `live` is the caller's history length, which is expected to equal
    /// `self.logical_len`; it is passed explicitly rather than read from the
    /// field so the rows actually preserved are the caller's decision, and
    /// clamped so a stale value cannot read past the buffer it is leaving.
    ///
    /// On failure every binding is returned to `self` and `max_len` is left at
    /// the old value, which stays true as the invariant "`max_len` is a lower
    /// bound on every binding's capacity" — bindings already grown simply have
    /// spare room. A retry re-reads each binding's own capacity and converges.
    pub(crate) fn grow_to(
        &mut self,
        session: &InferenceSession,
        needed: usize,
        live: usize,
    ) -> anyhow::Result<()> {
        if needed <= self.max_len {
            return Ok(());
        }
        let new_max = needed.max(self.max_len.saturating_mul(2));
        let live = live.min(self.max_len);
        let (dtype, pairs) = (self.dtype, std::mem::take(&mut self.pairs));
        let mut bindings = std::mem::take(&mut self.bindings);
        let mut failure = None;
        for (index, binding) in bindings.iter_mut().enumerate() {
            match Self::regrow(session, binding, &pairs[index], dtype, new_max, live) {
                Ok(replacement) => *binding = replacement,
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        // Restored unconditionally: an early return that left `self.bindings`
        // empty would make the next `grow_to` a silent no-op returning `Ok`
        // with no KV bound at all.
        self.bindings = bindings;
        self.pairs = pairs;
        if let Some(error) = failure {
            return Err(error);
        }
        self.max_len = new_max;
        Ok(())
    }

    /// Re-bind one KV pair at `capacity`, carrying its first `live` rows over.
    fn regrow(
        session: &InferenceSession,
        binding: &mut DeviceIoBinding,
        (present, past): &(String, String),
        dtype: DataType,
        capacity: usize,
        live: usize,
    ) -> anyhow::Result<DeviceIoBinding> {
        let physical = binding.physical_shape().to_vec();
        let (blocks, old_capacity, head_dim) =
            (physical[0] * physical[1], physical[2], physical[3]);
        let row_bytes = head_dim * dtype.byte_size();
        // Stage the live rows out of the old buffer, one run per head.
        let mut carried = Vec::with_capacity(blocks);
        for block in 0..blocks {
            carried.push(binding.read_bytes_range(
                block_offset(block, old_capacity, row_bytes),
                live * row_bytes,
            )?);
        }
        let mut grown_physical = physical;
        grown_physical[2] = capacity;
        let mut logical = grown_physical.clone();
        logical[2] = live;
        let mut grown = session.allocate_device_binding(
            past.clone(),
            Some(present.clone()),
            dtype,
            grown_physical,
            logical,
        )?;
        for (block, rows) in carried.into_iter().enumerate() {
            grown.write_bytes(block_offset(block, capacity, row_bytes), &rows)?;
        }
        Ok(grown)
    }

    /// Assemble a state directly from already-allocated bindings.
    ///
    /// Test-only: [`Self::new`] additionally requires a GQA-shaped graph, and
    /// the growth logic is independent of how the bindings were obtained.
    #[cfg(test)]
    pub(crate) fn from_parts_for_test(
        bindings: Vec<DeviceIoBinding>,
        pairs: Vec<(String, String)>,
        dtype: DataType,
        max_len: usize,
        logical_len: usize,
    ) -> Self {
        Self {
            bindings,
            pairs,
            dtype,
            max_len,
            logical_len,
        }
    }

    #[cfg(test)]
    pub(crate) fn binding_for_test(&mut self, index: usize) -> &mut DeviceIoBinding {
        &mut self.bindings[index]
    }

    /// Advance every KV binding's logical sequence length to `len`, exposing the
    /// freshly-appended rows to the next step's consumers.
    pub(crate) fn set_logical_len(&mut self, len: usize) -> anyhow::Result<()> {
        for binding in &mut self.bindings {
            let mut shape = binding.physical_shape().to_vec();
            shape[2] = len;
            binding.set_logical_shape(shape)?;
        }
        self.logical_len = len;
        Ok(())
    }
}

/// Byte offset of `block`'s first row in a `[B, H, capacity, Dh]` cache, where
/// a "block" is one `(batch, head)` pair and `row_bytes == Dh * element_size`.
///
/// Capacity sits on the third axis, so each block owns `capacity` rows and
/// changing the capacity moves every block but the first. This is the whole
/// reason a grow cannot be a single memcpy.
fn block_offset(block: usize, capacity: usize, row_bytes: usize) -> usize {
    block * capacity * row_bytes
}

/// Read the persistent CPU KV cache capacity override, falling back to
/// [`DEFAULT_CPU_KV_MAX_LEN`]. Returns `Ok(None)` when the in-place path is
/// disabled via `ONNX_GENAI_CPU_INPLACE_KV=0`.
pub(crate) fn cpu_inplace_kv_max_len_from_env() -> anyhow::Result<Option<usize>> {
    match std::env::var("ONNX_GENAI_CPU_INPLACE_KV") {
        Ok(value) if matches!(value.trim(), "0" | "off" | "false" | "no") => return Ok(None),
        _ => {}
    }
    match std::env::var("ONNX_GENAI_CPU_KV_MAX_LEN") {
        Ok(value) => {
            let parsed = value.trim().parse::<usize>().with_context(|| {
                format!("invalid ONNX_GENAI_CPU_KV_MAX_LEN={value:?}: expected a positive integer")
            })?;
            if parsed == 0 {
                bail!("ONNX_GENAI_CPU_KV_MAX_LEN must be greater than zero");
            }
            Ok(Some(parsed))
        }
        Err(std::env::VarError::NotPresent) => Ok(Some(DEFAULT_CPU_KV_MAX_LEN)),
        Err(error) => Err(error).context("read ONNX_GENAI_CPU_KV_MAX_LEN"),
    }
}

/// True when every `past` KV input name feeds at least one `GroupQueryAttention`
/// node — the only CPU kernel that implements the append-only in-place path. The
/// persistent present==past binding is safe to apply only under this condition;
/// any other producer (e.g. a plain `Concat`) requires the host copy path.
pub(crate) fn all_pasts_consumed_by_gqa(
    graph: &onnx_runtime_ir::Graph,
    past_names: &[String],
) -> bool {
    if past_names.is_empty() {
        return false;
    }
    let gqa_consumed: HashSet<&str> = graph
        .nodes
        .iter()
        .filter(|(_, node)| node.op_type == "GroupQueryAttention")
        .flat_map(|(_, node)| node.inputs.iter().copied().flatten())
        .filter_map(|value_id| graph.value(value_id).name.as_deref())
        .collect();
    past_names
        .iter()
        .all(|past| gqa_consumed.contains(past.as_str()))
}

impl NativeDecodeSession {
    /// Build the non-KV step input tensors (token ids, attention mask, position
    /// ids, and routed / inputs_embeds ports) shared by both CPU decode paths,
    /// validating that every supplied routed tensor maps to a declared graph
    /// port. KV inputs are appended by the caller: `decode_cpu` adds growable
    /// host past tensors, while `decode_cpu_inplace` binds them on-device.
    fn prepare_cpu_step_inputs(
        &self,
        token_ids: &[TokenId],
        past_len: usize,
        total_len: usize,
        supplied_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<Vec<(String, Tensor)>> {
        let ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let mut supplied = HashMap::with_capacity(supplied_inputs.len());
        for (name, tensor) in supplied_inputs {
            if supplied.insert(name.as_str(), tensor).is_some() {
                bail!("native decode received duplicate routed step input '{name}'");
            }
        }

        let mut owned = Vec::with_capacity(self.step_inputs.len());
        for binding in &self.step_inputs {
            let tensor = match binding.source {
                NativeStepInputSource::TokenIds => {
                    Tensor::from_i64(&[1, token_ids.len()], &ids)?
                }
                NativeStepInputSource::AttentionMask => {
                    Tensor::from_i64(&[1, total_len], &vec![1; total_len])?
                }
                NativeStepInputSource::PositionIds => {
                    self.build_step_positions(past_len, total_len)?
                }
                NativeStepInputSource::InputsEmbeds => supplied
                    .remove(binding.name.as_str())
                    .cloned()
                    .with_context(|| {
                        format!(
                            "declared inputs_embeds input '{}' was not supplied to the native decode step; route the current embedding component output to this exact decoder port",
                            binding.name
                        )
                    })?,
                NativeStepInputSource::Routed => supplied
                    .remove(binding.name.as_str())
                    .cloned()
                    .with_context(|| {
                        format!(
                            "native decode graph input '{}' has no generated role and no routed step tensor; declare a pipeline dataflow edge to this exact decoder port",
                            binding.name
                        )
                    })?,
            };
            owned.push((binding.name.clone(), tensor));
        }
        if !supplied.is_empty() {
            let mut unknown = supplied.keys().copied().collect::<Vec<_>>();
            unknown.sort_unstable();
            bail!(
                "native decode received routed step inputs that are not declared graph ports: {unknown:?}"
            );
        }
        Ok(owned)
    }

    /// Turn the raw logits / hidden outputs of a CPU forward pass into a decode
    /// result: greedy argmax or full-logits extraction plus recording the
    /// declared hidden state. Shared by both CPU decode paths; per-mode KV
    /// output handling stays in each caller.
    fn finalize_cpu_logits_and_hidden(
        &mut self,
        logits: Option<Tensor>,
        hidden: Option<Tensor>,
        greedy: bool,
    ) -> anyhow::Result<NativeCpuDecodeResult> {
        let logits = logits
            .with_context(|| format!("native decoder omitted logits output '{}'", self.logits))?;
        let result = if greedy {
            let _sampling_span = onnx_genai_ort::prof_span!("native.sampling");
            NativeCpuDecodeResult::Token(argmax_logits_tensor(&logits)?)
        } else {
            let logits = extract_logits(&logits)?;
            if logits.iter().flatten().any(|value| !value.is_finite()) {
                bail!("native decoder produced non-finite logits");
            }
            NativeCpuDecodeResult::Logits(logits)
        };
        self.last_hidden = match (&self.hidden_output, hidden) {
            (Some(name), Some(tensor)) => Some(
                extract_last_row(&tensor)
                    .with_context(|| format!("read native decoder hidden output '{name}'"))?,
            ),
            (Some(name), None) => {
                bail!("native decoder omitted declared hidden output '{name}'")
            }
            (None, _) => None,
        };
        Ok(result)
    }

    pub(crate) fn decode_cpu(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        greedy: bool,
        supplied_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<NativeCpuDecodeResult> {
        let total_len = past_len
            .checked_add(token_ids.len())
            .context("native decode context length overflow")?;
        let prepare_span = onnx_genai_ort::prof_span!("native.prepare_inputs");
        let mut owned =
            self.prepare_cpu_step_inputs(token_ids, past_len, total_len, supplied_inputs)?;
        owned.reserve(self.kv_inputs.len());
        for name in &self.kv_inputs {
            if !self.past.contains_key(name) {
                owned.push((name.clone(), self.make_empty_past(name)?));
            }
        }
        let mut bindings = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        bindings.reserve(self.past.len());
        for name in &self.kv_inputs {
            if let Some(tensor) = self.past.get(name) {
                bindings.push((name.as_str(), tensor));
            }
        }
        drop(prepare_span);

        let run_result: anyhow::Result<_> = {
            let _run_span = onnx_genai_ort::prof_span!("native.session_run");
            if token_ids.len() == 1 && !self.has_plugin_fused {
                let uses_decode_pool = self.uses_decode_pool;
                onnx_runtime_ep_cpu::with_decode_pool_scope(uses_decode_pool, || {
                    self.session.run(&bindings).map_err(anyhow::Error::from)
                })
            } else {
                self.session.run(&bindings).map_err(anyhow::Error::from)
            }
        };
        let outputs = match run_result {
            Ok(outputs) => outputs,
            Err(error) => {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                bail!("native decoder forward pass failed{diagnosis}: {error}");
            }
        };

        let fetch_span = onnx_genai_ort::prof_span!("native.logits_fetch");
        if outputs.len() != self.session.outputs().len() {
            bail!(
                "native decoder returned {} outputs, but the graph declares {}",
                outputs.len(),
                self.session.outputs().len()
            );
        }
        let mut logits = None;
        let mut hidden = None;
        let mut next_past = HashMap::with_capacity(self.kv_inputs.len());
        let compressed_state_enabled = !self.compressed_state.is_empty();
        for (metadata, tensor) in self.session.outputs().iter().zip(outputs) {
            if metadata.name == self.logits {
                logits = Some(tensor);
            } else if self.hidden_output.as_deref() == Some(metadata.name.as_str()) {
                hidden = Some(tensor);
            } else if let Some(past) = self.present_to_past.get(&metadata.name) {
                // Fixed-size recurrent states (conv_state / recurrent_state) are
                // replaced wholesale each step and carry a static feature dim on
                // the penultimate axis, so the growable-KV sequence-length check
                // does not apply to them.
                let recurrent = self
                    .session
                    .inputs()
                    .iter()
                    .find(|meta| &meta.name == past)
                    .is_some_and(|meta| is_recurrent_state_shape(&meta.shape));
                let compressed_transition = compressed_state_enabled
                    .then(|| self.compressed_state.transition_for_present(&metadata.name))
                    .flatten();
                match compressed_transition {
                    Some(csa::CompressedStateTransitionSpec::Record(spec)) => {
                        let prior = bindings
                            .iter()
                            .find(|(name, _)| *name == spec.input)
                            .map(|(_, tensor)| *tensor)
                            .with_context(|| {
                                format!(
                                    "compressed-attention transition '{}' => '{}' has no bound \
                                     past tensor",
                                    spec.input, spec.output
                                )
                            })?;
                        csa::validate_record_transition(
                            spec,
                            prior.into(),
                            (&tensor).into(),
                            past_len,
                            total_len,
                            1,
                        )?;
                        self.compressed_state_stats.transitions_validated = self
                            .compressed_state_stats
                            .transitions_validated
                            .saturating_add(1);
                        self.compressed_state_stats.host_output_allocations = self
                            .compressed_state_stats
                            .host_output_allocations
                            .saturating_add(1);
                        self.compressed_state_stats.host_output_bytes = self
                            .compressed_state_stats
                            .host_output_bytes
                            .saturating_add(
                                u64::try_from(tensor.as_bytes().len()).unwrap_or(u64::MAX),
                            );
                    }
                    Some(csa::CompressedStateTransitionSpec::Carry(spec)) => {
                        let prior = bindings
                            .iter()
                            .find(|(name, _)| *name == spec.input)
                            .map(|(_, tensor)| *tensor)
                            .with_context(|| {
                                format!(
                                    "compressed-attention transition '{}' => '{}' has no bound \
                                     past tensor",
                                    spec.input, spec.output
                                )
                            })?;
                        csa::validate_carry_transition(spec, prior.into(), (&tensor).into(), 1)?;
                        self.compressed_state_stats.transitions_validated = self
                            .compressed_state_stats
                            .transitions_validated
                            .saturating_add(1);
                        self.compressed_state_stats.host_output_allocations = self
                            .compressed_state_stats
                            .host_output_allocations
                            .saturating_add(1);
                        self.compressed_state_stats.host_output_bytes = self
                            .compressed_state_stats
                            .host_output_bytes
                            .saturating_add(
                                u64::try_from(tensor.as_bytes().len()).unwrap_or(u64::MAX),
                            );
                    }
                    None if !recurrent => {
                        let seq_axis = tensor.shape.len().checked_sub(2).with_context(|| {
                            format!("native present tensor '{}' rank is below 2", metadata.name)
                        })?;
                        if tensor.shape[seq_axis] != total_len {
                            bail!(
                                "native present tensor '{}' sequence length {} does not match \
                                 {total_len}",
                                metadata.name,
                                tensor.shape[seq_axis]
                            );
                        }
                    }
                    None => {}
                }
                next_past.insert(past.clone(), tensor);
            }
        }
        for (present, past) in &self.present_to_past {
            if !next_past.contains_key(past) {
                bail!("native decoder omitted present output '{present}'");
            }
        }
        let result = self.finalize_cpu_logits_and_hidden(logits, hidden, greedy)?;
        drop(fetch_span);

        let _kv_span = onnx_genai_ort::prof_span!("native.kv_update");
        self.past = next_past;
        self.current_len = total_len;
        Ok(result)
    }

    /// Decode one or more tokens using the persistent in-place CPU KV cache.
    ///
    /// Mirrors [`Self::decode_cpu`] but replaces the growing past-KV host inputs
    /// and the full present-KV host round-trip with present==past device
    /// bindings: `input_ids`, `attention_mask` and `position_ids` are still fed
    /// as fresh host inputs, but the KV caches persist in host buffers that the
    /// GroupQueryAttention kernel appends to in place. Only bound (present) KV
    /// outputs are suppressed; `logits`/`hidden` materialize as usual.
    pub(crate) fn decode_cpu_inplace(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        greedy: bool,
        supplied_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<NativeCpuDecodeResult> {
        let total_len = past_len
            .checked_add(token_ids.len())
            .context("native decode context length overflow")?;
        // Outgrowing the current capacity re-binds a larger cache rather than
        // failing the decode: capacity is a buffer-sizing decision, not a model
        // limit, and the old behaviour turned a 4097-token context into a hard
        // error.
        {
            let state = self
                .cpu_kv
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("decode_cpu_inplace requires CPU KV state"))?;
            if total_len > state.max_len {
                state
                    .grow_to(&self.session, total_len, past_len)
                    .with_context(|| {
                        format!("growing the CPU KV cache to hold {total_len} positions")
                    })?;
            }
        }

        let prepare_span = onnx_genai_ort::prof_span!("native.prepare_inputs");
        // The KV ports are device-bound (present==past), so only the generated
        // and routed non-KV step inputs are fed as fresh host tensors here.
        let owned =
            self.prepare_cpu_step_inputs(token_ids, past_len, total_len, supplied_inputs)?;
        let bindings = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        drop(prepare_span);

        let uses_decode_pool = self.uses_decode_pool;
        // Inc-1b PR-2: decide decode-inline routing before borrowing `cpu_kv`
        // mutably (the decision reads `self.session`/`self.decode_inline`).
        let route_inline = self.route_decode_inline(token_ids);
        let state = self
            .cpu_kv
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("decode_cpu_inplace requires CPU KV state"))?;
        let run_result: anyhow::Result<_> = {
            let _run_span = onnx_genai_ort::prof_span!("native.session_run");
            if route_inline {
                // Route this single-token decode step to the decode-specialized
                // inlined-body sibling exec (eager), binding the identical
                // persistent KV/state buffers so recurrent-state continuity is
                // preserved across the prefill→decode boundary (design §3).
                onnx_runtime_ep_cpu::with_decode_pool_scope(uses_decode_pool, || {
                    self.session
                        .run_decode_inline_with_device_bindings(&bindings, &mut state.bindings)
                        .map_err(anyhow::Error::from)
                })
            } else if token_ids.len() == 1 && !self.has_plugin_fused {
                onnx_runtime_ep_cpu::with_decode_pool_scope(uses_decode_pool, || {
                    self.session
                        .run_with_device_bindings(&bindings, &mut state.bindings)
                        .map_err(anyhow::Error::from)
                })
            } else {
                self.session
                    .run_with_device_bindings(&bindings, &mut state.bindings)
                    .map_err(anyhow::Error::from)
            }
        };
        let outputs = match run_result {
            Ok(outputs) => outputs,
            Err(error) => {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                bail!("native in-place decoder forward pass failed{diagnosis}: {error}");
            }
        };
        // The freshly-appended rows become visible to the next step's consumers.
        state.set_logical_len(total_len)?;

        let fetch_span = onnx_genai_ort::prof_span!("native.logits_fetch");
        if outputs.len() != self.session.outputs().len() {
            bail!(
                "native in-place decoder returned {} outputs, but the graph declares {}",
                outputs.len(),
                self.session.outputs().len()
            );
        }
        let mut logits = None;
        let mut hidden = None;
        for (metadata, tensor) in self.session.outputs().iter().zip(outputs) {
            let Some(tensor) = tensor else {
                // A `None` output is a suppressed present==past KV binding, which
                // is exactly what the in-place cache expects.
                if self.present_to_past.contains_key(&metadata.name) {
                    continue;
                }
                bail!(
                    "native in-place decoder suppressed non-KV output '{}'",
                    metadata.name
                );
            };
            if metadata.name == self.logits {
                logits = Some(tensor);
            } else if self.hidden_output.as_deref() == Some(metadata.name.as_str()) {
                hidden = Some(tensor);
            } else if self.present_to_past.contains_key(&metadata.name) {
                bail!(
                    "native in-place decoder unexpectedly materialized bound present output '{}'",
                    metadata.name
                );
            }
        }
        let result = self.finalize_cpu_logits_and_hidden(logits, hidden, greedy)?;
        drop(fetch_span);

        self.current_len = total_len;
        Ok(result)
    }
}
