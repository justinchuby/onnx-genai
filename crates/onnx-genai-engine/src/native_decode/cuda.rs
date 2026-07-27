use super::*;

/// Purely structural signals that gate whether whole-step CUDA graph capture is
/// *auto-attempted* on the native decode path. Never derived from a model or
/// architecture name (RULES.md §2/§2.1) — only from device placement and the
/// declared KV-ownership metadata. When these hold, per-step decode topology is
/// static and the KV cache is device-resident and owned, so a captured graph can
/// replay safely. The runtime decline machinery in `DecodeCudaState::new`
/// remains the final safety net: if a would-be capture still carries a dynamic
/// auxiliary seam it is transparently declined and decode continues eagerly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GraphCaptureStructuralSafety {
    /// Decode runs on a CUDA device (device-resident, replayable bindings).
    pub(crate) device_is_cuda: bool,
    /// KV cache is owned/device-resident (not a borrowed shared-KV proposer).
    pub(crate) kv_ownership: KvOwnership,
}

impl GraphCaptureStructuralSafety {
    /// True when structural conditions make whole-step capture safe to attempt.
    pub(crate) fn is_capture_safe(self) -> bool {
        self.device_is_cuda && self.kv_ownership == KvOwnership::Owned
    }
}

/// Resolve whether whole-step CUDA graph capture should be attempted for the
/// native decode path, honoring explicit overrides before the structural
/// auto-decision.
///
/// Precedence:
/// 1. Programmatic `NativeDecodeCudaOptions::graph_capture` (`Some`) always wins.
/// 2. An explicitly-set `ONNX_GENAI_CUDA_GRAPH` env var (`=0` forces OFF, `=1`
///    forces ON) is honored next.
/// 3. When neither is set, auto-decide from `structural` safety: attempt capture
///    only when the decode topology is structurally graph-safe.
pub(crate) fn resolve_graph_capture_enabled(
    programmatic: Option<bool>,
    env_explicit: bool,
    env_value: bool,
    structural: GraphCaptureStructuralSafety,
) -> bool {
    if let Some(explicit) = programmatic {
        return explicit;
    }
    if env_explicit {
        return env_value;
    }
    structural.is_capture_safe()
}

const DEFAULT_CUDA_KV_MAX_LEN: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaKvDebugStats {
    pub logical_len: usize,
    pub max_len: usize,
    pub device_ptrs: Vec<usize>,
    pub kv_transfers: DeviceBindingTransferStats,
    pub graph: CudaGraphDebugStats,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CudaGraphDebugStats {
    pub enabled: bool,
    pub captures: u64,
    pub replays: u64,
    pub fallbacks: u64,
    pub allocation_counts: DeviceAllocationCounts,
    /// Structured reasons from the most recent capture fallback.
    pub fallback_report: Option<CaptureDeclineReport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeCudaGraphPhase {
    NeedsWarmup,
    Armed,
    Ready,
    Unsupported,
}

pub(crate) struct DecodeCudaState {
    logical_len: usize,
    pub(crate) max_len: usize,
    bindings: Vec<DeviceIoBinding>,
    base_binding_count: usize,
    kv_binding_range: std::ops::Range<usize>,
    auxiliary_binding_range: std::ops::Range<usize>,
    input_ids_binding: usize,
    position_ids_binding: Option<usize>,
    logits_binding: usize,
    logits_shape: Vec<usize>,
    logits_dtype: DataType,
    greedy_result: DeviceIoBinding,
    graph_enabled: bool,
    /// `true` when the attention-mask binding exposes its *logical* valid length
    /// (not the padded physical capacity) to at least one consumer that is not a
    /// capacity-aware kernel — e.g. GLM-5.2's `indexer` arithmetic branch, which
    /// combines a logical-width score with the mask and would break if the mask
    /// leaked at physical `max_len`. When set, single-token decode must expose the
    /// mask at the growing logical length rather than freezing it to `max_len`,
    /// which also forfeits CUDA-graph capture (mirroring the eager prefill path).
    mask_exposes_logical: bool,
    graph_phase: DecodeCudaGraphPhase,
    graph_captures: u64,
    graph_replays: u64,
    graph_fallbacks: u64,
    pub(crate) graph_fallback_reason: Option<String>,
    pub(crate) graph_fallback_report: Option<CaptureDeclineReport>,
    /// Structural reasons, recorded at binding time, why one or more auxiliary
    /// graph outputs could not be persistently bound (an unresolved symbolic
    /// dimension that is not batch or query-seq). Non-empty here means CUDA
    /// graph capture was declined up front and the eager device path is in
    /// force for this generation. Empty when every auxiliary output was
    /// statically bindable.
    pub(crate) auxiliary_bind_declines: Vec<String>,
    /// When `false` (today's default), `NativeDecodeSession::rewind` invalidates
    /// the captured decode graph before rolling the device KV back — correct for
    /// the eager M=K verify path (option (b)), which captures nothing.
    ///
    /// When `true`, rewind performs a *contents-only* mutation (zero the mask
    /// tail + truncate the KV logical length) and **retains** the captured graph.
    /// This is the option (c) invariant: a single fixed-topology M=maxK graph
    /// whose device-binding pointers stay invariant while only buffer contents /
    /// logical shapes change across steps — exactly the data-driven mutation the
    /// captured graph already tolerates on the M=1 replay path. Kept dormant
    /// (default `false`) until WP4 graduates verify to the captured path.
    pub(crate) retain_graph_on_rewind: bool,
    /// Dormant option (c) scaffolding: the fixed query-row capacity (M=maxK) a
    /// padded single-capture verify graph would be captured at. `None` today —
    /// the eager verify path (option (b)) captures nothing. Set only by the
    /// dormant `configure_padded_verify_capture` switch (not on the hot path).
    #[cfg(test)]
    pub(crate) padded_query_capacity: Option<usize>,
}

pub(crate) struct DecodeCudaIo<'a> {
    pub(crate) input_ids: &'a str,
    pub(crate) attention_mask: &'a str,
    pub(crate) position_ids: Option<&'a str>,
    pub(crate) logits: &'a str,
}

pub(crate) fn trace_capture_declines(trace: &TraceContext, report: &CaptureDeclineReport) {
    for decline in &report.entries {
        if let Some(node_id) = decline.node_id {
            capture_rejected(
                trace,
                node_id,
                decline.op_type.as_str(),
                decline.domain.as_str(),
                decline.reason.as_str(),
            );
        }
    }
}

impl NativeDecodeSession {
    /// Run one eager (uncaptured) `[1, K]` device forward pass and return host
    /// `[K, vocab]` logits.
    ///
    /// This is the shared body of `decode_cuda`'s multi-token branch and the
    /// `decode_cuda_eager` verify path: invalidate any captured graph, rebuild
    /// the host token/position input tensors, run against the device KV/mask
    /// bindings, collect and validate the logits output, and advance the KV
    /// logical length. `error_context` (`"decoder"` or `"verify"`) only selects
    /// the wording of the two diagnostic messages so the extraction stays
    /// byte-identical to the two original inlined bodies.
    ///
    /// The caller resolves the token/position input names and computes
    /// `total_len`, and is responsible for the preceding `extend_mask` call
    /// (whose exposed length differs between the two paths).
    #[allow(clippy::too_many_arguments)]
    fn run_cuda_eager_rows(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        total_len: usize,
        token_input: &str,
        position_input: Option<&str>,
        error_context: &str,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        state.invalidate_graph(&mut self.session)?;
        let ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let input_ids = Tensor::from_i64(&[1, token_ids.len()], &ids)?;
        let mut owned = Vec::with_capacity(2);
        owned.push((token_input.to_owned(), input_ids));
        if let Some(position_ids_name) = position_input {
            let positions = (past_len..total_len)
                .map(|position| i64::try_from(position).context("position id exceeds i64 range"))
                .collect::<anyhow::Result<Vec<_>>>()?;
            owned.push((
                position_ids_name.to_owned(),
                Tensor::from_i64(&[1, token_ids.len()], &positions)?,
            ));
        }
        let bindings = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let outputs = match self
            .session
            .run_with_device_bindings(&bindings, &mut state.bindings[..state.base_binding_count])
        {
            Ok(outputs) => outputs,
            Err(error) => {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                bail!("native CUDA {error_context} forward pass failed{diagnosis}: {error}");
            }
        };
        let names = self
            .session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let mut named = names
            .into_iter()
            .zip(outputs)
            .filter_map(|(name, tensor)| tensor.map(|tensor| (name, tensor)))
            .collect::<HashMap<_, _>>();
        let logits = named
            .remove(&self.logits)
            .with_context(|| format!("native decoder omitted logits output '{}'", self.logits))?;
        if !named.is_empty() {
            bail!(
                "native CUDA {error_context} unexpectedly materialized bound outputs: {:?}",
                named.keys().collect::<Vec<_>>()
            );
        }
        let logits = extract_logits(&logits)?;
        if logits.iter().flatten().any(|value| !value.is_finite()) {
            bail!("native decoder produced non-finite logits");
        }
        state.set_logical_len(total_len)?;
        self.current_len = total_len;
        Ok(logits)
    }

    pub(crate) fn decode_cuda(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let total_len = past_len
            .checked_add(token_ids.len())
            .context("native decode context length overflow")?;
        let token_input = self
            .step_input_name(NativeStepInputSource::TokenIds)
            .context("native CUDA decoder has no token input binding")?
            .to_owned();
        let position_input = self
            .step_input_name(NativeStepInputSource::PositionIds)
            .map(str::to_owned);
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.max_len {
            bail!(
                "CUDA KV capacity exceeded: requested context length {total_len}, configured max_len {} (set ONNX_GENAI_CUDA_KV_MAX_LEN or use load_with_cuda_kv_max_len)",
                state.max_len
            );
        }
        // Single-token decode freezes the mask to physical capacity so the step
        // is CUDA-graph-capture eligible; multi-token prefill keeps the growing
        // logical length (prefix-sensitive causal island). A mask whose logical
        // valid length feeds a non-capacity-aware consumer (see
        // `decode_mask_expose_len`) cannot be frozen and uses `total_len`.
        let mask_expose = if token_ids.len() == 1 {
            state.decode_mask_expose_len(total_len)
        } else {
            total_len
        };
        state.extend_mask(past_len, total_len, mask_expose)?;

        if token_ids.len() == 1 {
            state.write_decode_inputs(token_ids[0], past_len)?;
            if let Err(error) = state.run_one_token(&mut self.session, &self.trace) {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                bail!("native CUDA decoder forward pass failed{diagnosis}: {error}");
            }
            let logits = state.read_logits()?;
            // Detection-before-consumption: the logits read above is the single
            // per-step device→host sync. Piggyback on it to poll the shared
            // capture-error word (no extra synchronize). If a captured replay
            // violates a device-side bound, kernels latch the flag and avoid the
            // unsafe access, so fail hard before consuming the produced token.
            let capture_error = self.session.check_device_capture_error()?;
            if capture_error != 0 {
                let _ = state.invalidate_graph(&mut self.session);
                bail!(
                    "native CUDA decoder aborted: device capture validation violation (flags=0x{capture_error:x}) detected during captured graph replay; the produced token was rejected before consumption and the decode graph was invalidated"
                );
            }
            if logits.iter().flatten().any(|value| !value.is_finite()) {
                bail!("native decoder produced non-finite logits");
            }
            state.set_logical_len(total_len)?;
            self.current_len = total_len;
            return Ok(logits);
        }

        self.run_cuda_eager_rows(
            token_ids,
            past_len,
            total_len,
            &token_input,
            position_input.as_deref(),
            "decoder",
        )
    }

    /// Speculative **verify** primitive (option (b): the safe eager M=K path).
    ///
    /// Runs the `draft` candidate tokens (K = `draft.len()`) through the target
    /// in a single eager forward and returns `[K, vocab]` host logits — one
    /// predicted-distribution row per draft position. This is the primitive
    /// WP2/WP3 build on: the driver compares each row's argmax against `draft`
    /// to find the accepted prefix (plus the free bonus token) and then rewinds
    /// the device KV to the committed length.
    ///
    /// It never enters the M=1 captured-graph greedy hot path — it always takes
    /// the eager multi-token forward (`decode_cuda_eager`) so the 762 tok/s plain
    /// path stays byte-identical. Greedy is the target regime, but returning raw
    /// logits also lets a driver fall back to host sampling for non-greedy
    /// requests. `past` must equal the committed length (`current_len`).
    pub fn decode_verify(
        &mut self,
        draft: &[TokenId],
        past: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if draft.is_empty() {
            bail!("native decode_verify requires at least one draft token");
        }
        if past != self.current_len {
            bail!(
                "native decode_verify past length mismatch: caller supplied {past}, adapter holds {}",
                self.current_len
            );
        }
        if self.cuda.is_some() {
            return self.decode_cuda_eager(draft, past);
        }
        // CPU sessions already run any M>1 forward eagerly through the shared
        // decode path, which returns the full [K, vocab] rows verify needs.
        <Self as DecodeBackend>::decode(self, draft, past)
    }

    /// Eager multi-token (M=K) CUDA forward used by the verify primitive.
    ///
    /// Self-contained on purpose: it mirrors `decode_cuda`'s eager branch but is
    /// a *separate* method so the M=1 captured-graph hot path in `decode_cuda`
    /// stays byte-identical and out of verify's blast radius. It invalidates any
    /// captured graph (option (b) captures nothing), rebuilds host `[1,K]`
    /// input/position tensors, runs against the device KV/mask bindings, and
    /// advances the KV logical length to `past_len + K`.
    ///
    /// The whole pass is wrapped in its own trace span so Deckard's per-op
    /// timings under it remain attributable to the verify forward.
    fn decode_cuda_eager(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let _verify_span = self
            .trace
            .span("native_decode_verify", "spec")
            .with_args(Args::new().with("rows", token_ids.len() as u64));
        let total_len = past_len
            .checked_add(token_ids.len())
            .context("native decode context length overflow")?;
        let token_input = self
            .step_input_name(NativeStepInputSource::TokenIds)
            .context("native CUDA decoder has no token input binding")?
            .to_owned();
        let position_input = self
            .step_input_name(NativeStepInputSource::PositionIds)
            .map(str::to_owned);
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.max_len {
            bail!(
                "CUDA KV capacity exceeded: requested context length {total_len}, configured max_len {} (set ONNX_GENAI_CUDA_KV_MAX_LEN or use load_with_cuda_kv_max_len)",
                state.max_len
            );
        }
        state.extend_mask(past_len, total_len, total_len)?;
        self.run_cuda_eager_rows(
            token_ids,
            past_len,
            total_len,
            &token_input,
            position_input.as_deref(),
            "verify",
        )
    }

    pub(crate) fn decode_cuda_greedy(
        &mut self,
        token_id: TokenId,
        past_len: usize,
    ) -> anyhow::Result<TokenId> {
        let total_len = past_len
            .checked_add(1)
            .context("native decode context length overflow")?;
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.max_len {
            bail!(
                "CUDA KV capacity exceeded: requested context length {total_len}, configured max_len {} (set ONNX_GENAI_CUDA_KV_MAX_LEN or use load_with_cuda_kv_max_len)",
                state.max_len
            );
        }
        state.extend_mask(past_len, total_len, state.decode_mask_expose_len(total_len))?;
        state.write_decode_inputs(token_id, past_len)?;
        if let Err(error) = state.run_one_token(&mut self.session, &self.trace) {
            let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
            bail!("native CUDA decoder forward pass failed{diagnosis}: {error}");
        }
        let (token_id, capture_error) = state.read_greedy_result()?;
        if capture_error != 0 {
            let _ = state.invalidate_graph(&mut self.session);
            bail!(
                "native CUDA decoder aborted: device capture validation violation (flags=0x{capture_error:x}) detected during captured graph replay; the produced token was rejected before consumption and the decode graph was invalidated"
            );
        }
        state.set_logical_len(total_len)?;
        self.current_len = total_len;
        Ok(token_id)
    }
}

impl DecodeCudaState {
    /// Collect the symbolic dimension ids that the native decoder structurally
    /// pins to `1` at decode time. Batch (axis 0 of every input) and query-seq
    /// (the remaining `input_ids` / `position_ids` axes, which are bound to a
    /// single token) are the only symbols that [`persistent_output_shape`] may
    /// safely collapse to `1`. `batch_only` restricts collection to axis 0 for
    /// inputs whose non-batch axes grow with the sequence (attention_mask and
    /// the past-KV tensors, whose total_seq axis is *not* a decode unit).
    pub(crate) fn collect_unit_symbols(
        shape: &[Dim],
        batch_only: bool,
        out: &mut HashSet<SymbolId>,
    ) {
        for (axis, dim) in shape.iter().enumerate() {
            if batch_only && axis != 0 {
                continue;
            }
            if let Dim::Symbolic(symbol) = dim {
                out.insert(*symbol);
            }
        }
    }

    /// First structurally-unresolved symbolic axis of an auxiliary output: a
    /// `Dim::Symbolic` that is *not* one of the decode-unit (batch / query-seq)
    /// symbols. Such a dimension is data-dependent (e.g. an accumulator indexed
    /// by total_seq / past+1), so collapsing it to `1` in a persistent device
    /// binding would under-allocate. Returns `(axis, symbol)` of the offender.
    pub(crate) fn unresolved_symbolic_axis(
        shape: &[Dim],
        unit_symbols: &HashSet<SymbolId>,
    ) -> Option<(usize, SymbolId)> {
        shape.iter().enumerate().find_map(|(axis, dim)| match dim {
            Dim::Symbolic(symbol) if !unit_symbols.contains(symbol) => Some((axis, *symbol)),
            _ => None,
        })
    }

    pub(crate) fn persistent_output_shape(
        name: &str,
        dtype: DataType,
        shape: &[Dim],
    ) -> anyhow::Result<Vec<usize>> {
        if matches!(dtype, DataType::Undefined | DataType::String) {
            bail!(
                "cannot bind auxiliary CUDA graph output '{name}' persistently: dtype {dtype:?} does not have fixed-size device tensor storage, but CUDA graph capture requires every declared graph output to use stable device storage; export this output as a numeric tensor or remove the unused graph output"
            );
        }
        let shape = shape
            .iter()
            .map(|dim| match dim {
                Dim::Static(value) => *value,
                Dim::Symbolic(_) => 1,
            })
            .collect::<Vec<_>>();
        let elements = shape.iter().try_fold(1usize, |product, &dim| {
            product.checked_mul(dim).with_context(|| {
                format!(
                    "cannot bind auxiliary CUDA graph output '{name}' persistently: shape {shape:?} overflows the device allocation size; export a bounded output shape or remove the unused graph output"
                )
            })
        })?;
        dtype.checked_storage_bytes(elements).with_context(|| {
            format!(
                "cannot bind auxiliary CUDA graph output '{name}' persistently: dtype {dtype:?} shape {shape:?} has no representable device allocation size; export a fixed-size numeric tensor or remove the unused graph output"
            )
        })?;
        Ok(shape)
    }

    pub(crate) fn new(
        session: &mut InferenceSession,
        io: DecodeCudaIo<'_>,
        present_to_past: &HashMap<String, String>,
        max_len: usize,
        graph_enabled: bool,
    ) -> anyhow::Result<Self> {
        let mut mask = session.allocate_device_binding(
            io.attention_mask,
            None::<String>,
            DataType::Int64,
            vec![1, max_len],
            vec![1, max_len],
        )?;
        mask.write_bytes(0, &vec![0; max_len * std::mem::size_of::<i64>()])?;

        let mut pairs = present_to_past
            .iter()
            .map(|(present, past)| (present.clone(), past.clone()))
            .collect::<Vec<_>>();
        pairs.sort_unstable_by(|left, right| left.1.cmp(&right.1));
        let mut bindings = Vec::with_capacity(4 + pairs.len());
        bindings.push(mask);
        let kv_start = bindings.len();
        for (present, past) in pairs {
            let meta = session
                .inputs()
                .iter()
                .find(|meta| meta.name == past)
                .with_context(|| format!("missing CUDA KV input metadata for '{past}'"))?;
            if !matches!(meta.dtype, DataType::Float32 | DataType::Float16) || meta.shape.len() != 4
            {
                bail!(
                    "CUDA KV input '{past}' must be rank-4 f32 or f16, got {:?} {:?}",
                    meta.dtype,
                    meta.shape
                );
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
                    bail!(
                        "cannot infer CUDA KV dimension {axis} for '{past}' shape {:?}",
                        meta.shape
                    );
                };
                physical_shape.push(value);
            }
            let mut logical_shape = physical_shape.clone();
            logical_shape[2] = 0;
            bindings.push(session.allocate_device_binding(
                past,
                Some(present),
                meta.dtype,
                physical_shape,
                logical_shape,
            )?);
        }
        let kv_end = bindings.len();

        let logits_meta = session
            .outputs()
            .iter()
            .find(|meta| meta.name == io.logits)
            .with_context(|| format!("missing CUDA logits output metadata for '{}'", io.logits))?;
        if !matches!(logits_meta.dtype, DataType::Float32 | DataType::Float16)
            || logits_meta.shape.is_empty()
        {
            bail!(
                "CUDA logits output '{}' must be non-scalar f32 or f16, got {:?} {:?}",
                io.logits,
                logits_meta.dtype,
                logits_meta.shape
            );
        }
        let logits_dtype = logits_meta.dtype;
        let logits_shape =
            Self::persistent_output_shape(io.logits, logits_dtype, &logits_meta.shape)?;
        let logits_device_binding = session.allocate_device_output_binding(
            io.logits,
            logits_dtype,
            logits_shape.clone(),
            logits_shape.clone(),
        )?;

        let present_outputs = present_to_past.keys().cloned().collect::<HashSet<_>>();
        let auxiliary_meta = session
            .outputs()
            .iter()
            .filter(|meta| meta.name != io.logits && !present_outputs.contains(&meta.name))
            .cloned()
            .collect::<Vec<_>>();

        // Structural safe-collapse analysis for auxiliary outputs. The native
        // decoder pins batch and query-seq to `1` at decode, so a symbolic aux
        // dimension is only safe to collapse to `1` when it is one of those
        // structurally-unit axes. Gather every symbol the decoder binds to `1`:
        // input_ids / position_ids (bound to `[1, 1]`) on all axes, plus the
        // batch axis (axis 0) of attention_mask and each past-KV input. Any
        // other symbolic aux dim (e.g. one indexed by total_seq / past+1) is
        // data-dependent and must not be collapsed. See RULES.md §2 — this is a
        // purely structural signal, never a model-name gate.
        let mut unit_symbols: HashSet<SymbolId> = HashSet::new();
        if let Some(meta) = session
            .inputs()
            .iter()
            .find(|meta| meta.name == io.input_ids)
        {
            Self::collect_unit_symbols(&meta.shape, false, &mut unit_symbols);
        }
        if let Some(position_ids) = io.position_ids
            && let Some(meta) = session
                .inputs()
                .iter()
                .find(|meta| meta.name == position_ids)
        {
            Self::collect_unit_symbols(&meta.shape, false, &mut unit_symbols);
        }
        if let Some(meta) = session
            .inputs()
            .iter()
            .find(|meta| meta.name == io.attention_mask)
        {
            Self::collect_unit_symbols(&meta.shape, true, &mut unit_symbols);
        }
        for past in present_to_past.values() {
            if let Some(meta) = session.inputs().iter().find(|meta| &meta.name == past) {
                Self::collect_unit_symbols(&meta.shape, true, &mut unit_symbols);
            }
        }

        let auxiliary_start = bindings.len();
        let mut declined_auxiliary: Vec<String> = Vec::new();
        for meta in auxiliary_meta {
            if let Some((axis, symbol)) = Self::unresolved_symbolic_axis(&meta.shape, &unit_symbols)
            {
                // The output's extent on this axis is data-dependent and not
                // structurally identifiable as batch or query-seq. Collapsing
                // it to `1` (as a persistent device binding requires) would
                // under-allocate, so we deliberately do NOT bind it. The eager
                // executor JIT-sizes and materializes this output every step,
                // so decode still works; only CUDA graph capture is forfeited
                // (capture demands a stable device address for every output).
                let symbol_label = session
                    .graph()
                    .symbol_constraints
                    .get(&symbol)
                    .and_then(|constraints| constraints.name.clone())
                    .unwrap_or_else(|| format!("symbol#{}", symbol.0));
                declined_auxiliary.push(format!(
                    "'{}' (axis {axis} is symbolic dim '{symbol_label}', not structurally batch or query-seq)",
                    meta.name
                ));
                continue;
            }
            let shape = Self::persistent_output_shape(&meta.name, meta.dtype, &meta.shape)?;
            bindings.push(
                session
                    .allocate_device_output_binding(
                        &meta.name,
                        meta.dtype,
                        shape.clone(),
                        shape,
                    )
                    .with_context(|| {
                        format!(
                            "failed to allocate persistent CUDA device binding for auxiliary graph output '{}'; CUDA graph capture requires every declared output to keep a stable device address",
                            meta.name
                        )
                    })?,
            );
        }
        let auxiliary_end = bindings.len();
        let base_binding_count = bindings.len();

        // If any auxiliary output could not be persistently bound, CUDA graph
        // capture is impossible (an unbound output would materialize on the
        // host mid-capture). Decline capture up front, with a clear structural
        // reason, and fall back to the eager device path — which still decodes
        // correctly by dynamically allocating the unbindable output each step.
        let graph_enabled = if !declined_auxiliary.is_empty() {
            if graph_enabled {
                tracing::warn!(
                    "native CUDA decode graph capture disabled: auxiliary output(s) {} carry unresolved symbolic dimensions that cannot be collapsed to a fixed persistent device binding; decode continues eagerly with dynamic allocation for those outputs",
                    declined_auxiliary.join(", ")
                );
            } else {
                tracing::debug!(
                    "native CUDA decode leaving auxiliary output(s) {} unbound (unresolved symbolic dimensions); eager path allocates them dynamically",
                    declined_auxiliary.join(", ")
                );
            }
            false
        } else {
            graph_enabled
        };

        let input_ids_binding = bindings.len();
        bindings.push(session.allocate_device_binding(
            io.input_ids,
            None::<String>,
            DataType::Int64,
            vec![1, 1],
            vec![1, 1],
        )?);
        let position_ids_binding = if let Some(position_ids) = io.position_ids {
            let index = bindings.len();
            bindings.push(session.allocate_device_binding(
                position_ids,
                None::<String>,
                DataType::Int64,
                vec![1, 1],
                vec![1, 1],
            )?);
            Some(index)
        } else {
            None
        };
        let logits_binding = bindings.len();
        bindings.push(logits_device_binding);

        #[cfg(feature = "cuda")]
        let argmax_words = {
            let vocab = *logits_shape
                .last()
                .context("CUDA logits shape has no vocabulary dimension")?;
            2 + onnx_runtime_ep_cuda::device_argmax_scratch_words(vocab)
        };
        #[cfg(not(feature = "cuda"))]
        let argmax_words = 2;
        let greedy_result = session.allocate_device_output_binding(
            "__native_greedy_argmax",
            DataType::Uint32,
            vec![argmax_words],
            vec![2],
        )?;

        // A graph records launch geometry, so replay is unsafe when a persistent
        // binding exposes a growing logical prefix instead of fixed capacity.
        // Surfacing *which* bindings force the eager fallback (previously a
        // silent `graph_enabled = false`) is essential for capture bring-up of
        // new architectures — enable with `RUST_LOG=onnx_genai_engine=debug`.
        let dynamic_logical: Vec<String> = bindings
            .iter()
            .filter(|binding| binding.has_dynamic_logical_input_shape())
            .map(|binding| {
                format!(
                    "{} (physical {:?} vs logical {:?})",
                    binding.input_name(),
                    binding.physical_shape(),
                    binding.logical_shape()
                )
            })
            .collect();
        if graph_enabled && !dynamic_logical.is_empty() {
            tracing::debug!(
                "native CUDA decode graph capture disabled: input binding(s) {} expose a growing logical prefix instead of fixed capacity (their consumers are not capacity-aware kernels); decode continues eagerly",
                dynamic_logical.join(", ")
            );
        }
        // The attention-mask binding (bindings[0]) is allocated with the
        // consumer-scoped capacity policy: it exposes its logical valid length
        // whenever any consumer is not a padded-capacity-safe kernel (Shape /
        // ReduceSum). Such a mask cannot be frozen to physical `max_len` during
        // single-token decode — doing so leaks the padded width into that
        // consumer (e.g. GLM-5.2's indexer `Add`, which broadcasts the mask
        // against a logical-width score). At construction the mask's logical and
        // physical shapes are still equal (`max_len`), so it is not yet caught by
        // the `has_dynamic_logical_input_shape` scan above; recognise it here from
        // the static policy so decode drives it at the growing logical length and,
        // like any growing logical input, forfeits CUDA-graph capture.
        let mask_exposes_logical = bindings
            .first()
            .is_some_and(DeviceIoBinding::exposes_logical_input_shape);
        if graph_enabled && mask_exposes_logical {
            tracing::debug!(
                "native CUDA decode graph capture disabled: attention-mask binding '{}' exposes its logical valid length to a non-capacity-aware consumer (e.g. an indexer arithmetic branch); single-token decode uses the growing logical mask width and continues eagerly",
                bindings[0].input_name()
            );
        }
        let graph_enabled = graph_enabled && dynamic_logical.is_empty() && !mask_exposes_logical;

        Ok(Self {
            logical_len: 0,
            max_len,
            bindings,
            base_binding_count,
            kv_binding_range: kv_start..kv_end,
            auxiliary_binding_range: auxiliary_start..auxiliary_end,
            input_ids_binding,
            position_ids_binding,
            logits_binding,
            logits_shape,
            logits_dtype,
            greedy_result,
            graph_enabled,
            mask_exposes_logical,
            graph_phase: DecodeCudaGraphPhase::NeedsWarmup,
            graph_captures: 0,
            graph_replays: 0,
            graph_fallbacks: 0,
            graph_fallback_reason: None,
            graph_fallback_report: None,
            auxiliary_bind_declines: declined_auxiliary,
            retain_graph_on_rewind: false,
            #[cfg(test)]
            padded_query_capacity: None,
        })
    }

    /// Write the valid `1`s for keys `[start, end)` and set the mask's exposed
    /// logical length. `expose_len` is the last-dim extent the graph's mask
    /// island (and hence the Attention kernel) sees: for a single-token decode
    /// step it is the fixed physical capacity (`max_len`), which freezes the
    /// island to a shape-static `[1,1,1,max_len]` additive bias (correct for a
    /// single query row — verified: the padding suffix maps to `-inf`, the valid
    /// prefix to `0`), so the decode step carries no growing logical input and
    /// stays CUDA-graph-capture eligible. Multi-token prefill passes `end`
    /// (the growing valid length) because the causal island is prefix-sensitive
    /// for `q_seq > 1` and must see the exact logical length.
    /// The last-dim extent to expose for the attention mask on a single-token
    /// decode step. Frozen to the physical capacity (`max_len`) so the step stays
    /// CUDA-graph-capture eligible — *unless* the mask binding exposes its logical
    /// valid length to a non-capacity-aware consumer (`mask_exposes_logical`), in
    /// which case the true valid length (`total_len`) must be used or the padded
    /// width leaks into that consumer's arithmetic (see [`Self::mask_exposes_logical`]).
    fn decode_mask_expose_len(&self, total_len: usize) -> usize {
        if self.mask_exposes_logical {
            total_len
        } else {
            self.max_len
        }
    }

    fn extend_mask(&mut self, start: usize, end: usize, expose_len: usize) -> anyhow::Result<()> {
        if end > self.max_len || start > end || expose_len > self.max_len || end > expose_len {
            bail!(
                "invalid CUDA mask update {start}..{end} (expose {expose_len}) for capacity {}",
                self.max_len
            );
        }
        let ones = (start..end)
            .flat_map(|_| 1i64.to_le_bytes())
            .collect::<Vec<_>>();
        self.bindings[0].write_bytes(start * std::mem::size_of::<i64>(), &ones)?;
        self.bindings[0].set_logical_shape(vec![1, expose_len])?;
        Ok(())
    }

    pub(crate) fn set_logical_len(&mut self, len: usize) -> anyhow::Result<()> {
        for binding in &mut self.bindings[self.kv_binding_range.clone()] {
            let mut shape = binding.physical_shape().to_vec();
            shape[2] = len;
            binding.set_logical_shape(shape)?;
        }
        self.logical_len = len;
        Ok(())
    }

    pub(crate) fn rewind(&mut self, target_len: usize) -> anyhow::Result<()> {
        if target_len < self.logical_len {
            let zeros = vec![0u8; (self.logical_len - target_len) * std::mem::size_of::<i64>()];
            self.bindings[0].write_bytes(target_len * std::mem::size_of::<i64>(), &zeros)?;
        }
        self.bindings[0].set_logical_shape(vec![1, target_len])?;
        self.set_logical_len(target_len)
    }

    fn write_decode_inputs(&mut self, token_id: TokenId, position: usize) -> anyhow::Result<()> {
        self.bindings[self.input_ids_binding].write_bytes(0, &i64::from(token_id).to_le_bytes())?;
        if let Some(index) = self.position_ids_binding {
            let position = i64::try_from(position).context("position id exceeds i64 range")?;
            self.bindings[index].write_bytes(0, &position.to_le_bytes())?;
        }
        Ok(())
    }

    fn run_one_token(
        &mut self,
        session: &mut InferenceSession,
        trace: &TraceContext,
    ) -> anyhow::Result<()> {
        debug_assert!(self.auxiliary_binding_range.end <= self.base_binding_count);
        if !self.graph_enabled {
            session.run_with_device_bindings(&[], &mut self.bindings)?;
            return Ok(());
        }

        match self.graph_phase {
            DecodeCudaGraphPhase::NeedsWarmup => {
                session.run_with_device_bindings(&[], &mut self.bindings)?;
                self.graph_phase = DecodeCudaGraphPhase::Armed;
            }
            DecodeCudaGraphPhase::Armed => {
                match session.try_capture_with_device_bindings(&[], &mut self.bindings)? {
                    DeviceGraphCaptureResult::Captured(outputs) => {
                        if outputs.iter().any(Option::is_some) {
                            bail!("captured CUDA decode unexpectedly materialized a host output");
                        }
                        self.graph_captures += 1;
                        self.graph_phase = DecodeCudaGraphPhase::Ready;
                    }
                    DeviceGraphCaptureResult::NotCapturable(report) => {
                        self.graph_fallbacks += 1;
                        self.graph_phase = DecodeCudaGraphPhase::Unsupported;
                        trace_capture_declines(trace, &report);
                        let reason = report.to_string();
                        self.graph_fallback_reason = Some(reason.clone());
                        self.graph_fallback_report = Some(report);
                        tracing::warn!(
                            "native CUDA decode graph capture disabled for this generation: {reason}"
                        );
                        session.run_with_device_bindings(&[], &mut self.bindings)?;
                    }
                }
            }
            DecodeCudaGraphPhase::Ready => {
                let still_valid = session.replay_device_graph(&mut self.bindings)?;
                self.graph_replays += 1;
                if !still_valid {
                    // A control-flow branch flip (e.g. LongRoPE short↔long at the
                    // context threshold) changed a seeded output shape and retired
                    // the captured graph after producing this token eagerly.
                    // Re-warm and re-capture for the new branch.
                    self.graph_phase = DecodeCudaGraphPhase::NeedsWarmup;
                }
            }
            DecodeCudaGraphPhase::Unsupported => {
                session.run_with_device_bindings(&[], &mut self.bindings)?;
            }
        }
        Ok(())
    }

    fn read_logits(&mut self) -> anyhow::Result<Vec<Vec<f32>>> {
        let bytes = self.bindings[self.logits_binding].read_bytes()?;
        let logits = Tensor::from_raw(self.logits_dtype, self.logits_shape.clone(), &bytes)?;
        extract_logits(&logits)
    }

    pub(crate) fn greedy_fastpath_supported(&self) -> bool {
        self.bindings[self.logits_binding].device_argmax_supported()
    }

    fn read_greedy_result(&mut self) -> anyhow::Result<(TokenId, u32)> {
        let vocab = *self
            .logits_shape
            .last()
            .context("CUDA logits shape has no vocabulary dimension")?;
        self.bindings[self.logits_binding].device_argmax(vocab, &mut self.greedy_result)?;
        let mut bytes = [0_u8; 2 * std::mem::size_of::<u32>()];
        self.greedy_result.read_bytes_into(&mut bytes)?;
        Ok((
            u32::from_ne_bytes(
                bytes[..4]
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("four token-id bytes"))?,
            ),
            u32::from_ne_bytes(
                bytes[4..]
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("four capture-error bytes"))?,
            ),
        ))
    }

    pub(crate) fn invalidate_graph(
        &mut self,
        session: &mut InferenceSession,
    ) -> anyhow::Result<()> {
        session.reset_device_graph()?;
        self.graph_phase = DecodeCudaGraphPhase::NeedsWarmup;
        Ok(())
    }

    /// Dormant option (c) switch (kept off until WP4). Arm the padded single
    /// M=maxK captured verify graph: fix the query-row capacity at `max_query_rows`
    /// and retain the captured graph across `rewind` (contents-only mutation)
    /// instead of invalidating it. Not reachable from the plain M=1 hot path nor
    /// the eager (option (b)) verify path; only a future WP4 driver flips it on.
    #[cfg(test)]
    pub(crate) fn configure_padded_verify_capture(&mut self, max_query_rows: usize) {
        self.padded_query_capacity = Some(max_query_rows);
        self.retain_graph_on_rewind = true;
    }

    /// Toggle whether `rewind` retains the captured decode graph (option (c),
    /// contents-only mutation) or invalidates it (option (b), the eager default).
    /// Dormant: only exercised by option-(c) correctness tests until WP4.
    #[cfg(test)]
    pub(crate) fn set_retain_graph_on_rewind(&mut self, retain: bool) {
        self.retain_graph_on_rewind = retain;
    }

    /// Fixed query-row capacity (M=maxK) of the dormant padded verify capture, or
    /// `None` while the eager (option (b)) verify path is in force.
    #[cfg(test)]
    pub(crate) fn padded_query_capacity(&self) -> Option<usize> {
        self.padded_query_capacity
    }

    pub(crate) fn debug_stats(&self, session: &InferenceSession) -> CudaKvDebugStats {
        let mut transfers = DeviceBindingTransferStats::default();
        let device_ptrs = self.bindings[self.kv_binding_range.clone()]
            .iter()
            .map(|binding| {
                let stats = binding.transfer_stats();
                transfers.host_upload_calls += stats.host_upload_calls;
                transfers.host_upload_bytes += stats.host_upload_bytes;
                transfers.host_download_calls += stats.host_download_calls;
                transfers.host_download_bytes += stats.host_download_bytes;
                binding.device_ptr() as usize
            })
            .collect();
        CudaKvDebugStats {
            logical_len: self.logical_len,
            max_len: self.max_len,
            device_ptrs,
            kv_transfers: transfers,
            graph: CudaGraphDebugStats {
                enabled: self.graph_enabled,
                captures: self.graph_captures,
                replays: self.graph_replays,
                fallbacks: self.graph_fallbacks,
                allocation_counts: session.device_allocation_counts().unwrap_or_default(),
                fallback_report: self.graph_fallback_report.clone(),
            },
        }
    }
}

pub(crate) fn cuda_kv_max_len_from_env() -> anyhow::Result<usize> {
    match std::env::var("ONNX_GENAI_CUDA_KV_MAX_LEN") {
        Ok(value) => {
            let parsed = value.trim().parse::<usize>().with_context(|| {
                format!("invalid ONNX_GENAI_CUDA_KV_MAX_LEN={value:?}: expected a positive integer")
            })?;
            if parsed == 0 {
                bail!("ONNX_GENAI_CUDA_KV_MAX_LEN must be greater than zero");
            }
            Ok(parsed)
        }
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_CUDA_KV_MAX_LEN),
        Err(error) => Err(error).context("read ONNX_GENAI_CUDA_KV_MAX_LEN"),
    }
}
