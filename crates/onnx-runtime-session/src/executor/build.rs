use super::*;

impl Executor {
    /// Compile a graph + weights into a runnable executor on the CPU EP.
    pub(crate) fn build(
        graph: Graph,
        weights: Arc<WeightStore>,
        ep: Arc<dyn ExecutionProvider>,
    ) -> Result<Self> {
        Self::build_with_cuda_requirement(
            graph,
            weights,
            ep,
            onnx_genai_runtime_config::runtime_config().require_cuda,
        )
    }

    pub(super) fn build_with_cuda_requirement(
        mut graph: Graph,
        weights: Arc<WeightStore>,
        mut ep: Arc<dyn ExecutionProvider>,
        require_cuda: bool,
    ) -> Result<Self> {
        let mut placement_span = trace_span("session.node_placement", "session");
        let requested_provider = placement_span.as_ref().map(|_| ep.name().to_string());
        let requested_device = placement_span
            .as_ref()
            .map(|_| ep.device_type().trace_name().into_owned());
        let nodes_before_placement = graph.num_nodes();
        // Reject incompatible control-flow signatures before EP optimizers run:
        // optimizer postconditions recursively validate subgraphs and can
        // otherwise obscure the actionable If diagnostic with a structural
        // error from a malformed branch.
        validate_control_flow_signatures(&graph)?;
        // Reject structurally invalid graphs (a non-DAG) and operators no EP can
        // run *before* EP optimizers run. An optimizer pass's postcondition
        // validation would otherwise obscure the actionable load-time diagnostic
        // (a wrapped `CycleDetected`, or an opset-import invariant instead of the
        // unsupported-operator error) with a structural error. Mirrors the
        // control-flow signature check above.
        graph.topological_order()?;
        reject_unsupported_operators(&graph, ep.as_ref())?;
        let silu_fused = fuse_silu_patterns(&mut graph);
        let graph_before_ep_passes = graph.clone();
        let ep_pass_nodes_before = graph.num_nodes();
        run_ep_scoped_passes(&mut graph, &weights, ep.as_ref())?;
        let ep_pass_nodes_after = graph.num_nodes();
        let mut execution_provider_fallback_report = cuda_fallback_report(&graph, ep.as_ref());
        let fallback_declines = execution_provider_fallback_report
            .as_ref()
            .map_or(0, |report| report.declines.len());
        if let Some(report) = &mut execution_provider_fallback_report {
            if require_cuda {
                return Err(SessionError::HeterogeneousPlacementRequired {
                    unsupported_nodes: report.to_string(),
                });
            }
            graph = graph_before_ep_passes;
            ep = auto_detect_cpu_ep()?;
            run_ep_scoped_passes(&mut graph, &weights, ep.as_ref())?;
            let mut assigned_ops = BTreeSet::new();
            report.assigned_node_count = collect_executable_ops(&graph, &mut assigned_ops);
            report.assigned_ops = assigned_ops.into_iter().collect();
            eprintln!(
                "[onnx-genai-warning] {report}. Set ONNX_GENAI_REQUIRE_CUDA=1 to reject this fallback"
            );
        }
        if let Some(span) = placement_span.as_mut() {
            let mut assigned_ops = BTreeSet::new();
            let assigned_nodes = collect_executable_ops(&graph, &mut assigned_ops);
            span.set_args(
                Args::new()
                    .with("requested_provider", requested_provider.unwrap_or_default())
                    .with("requested_device", requested_device.unwrap_or_default())
                    .with("selected_provider", ep.name().to_string())
                    .with(
                        "selected_device",
                        ep.device_type().trace_name().into_owned(),
                    )
                    .with("nodes_before", nodes_before_placement as u64)
                    .with("nodes_after", graph.num_nodes() as u64)
                    .with("ep_pass_nodes_before", ep_pass_nodes_before as u64)
                    .with("ep_pass_nodes_after", ep_pass_nodes_after as u64)
                    .with("silu_fused", silu_fused as u64)
                    .with("assigned_nodes", assigned_nodes as u64)
                    .with("assigned_op_classes", assigned_ops.len() as u64)
                    .with("fallback_declines", fallback_declines as u64),
            );
        }
        drop(placement_span);
        // Topological order up front: also validates the selected graph is a DAG.
        let order = graph.topological_order()?;
        let weight_handles = {
            let mut span = trace_span("session.lazy_weight_handles", "session");
            let handles = build_lazy_weight_handles(&graph, &weights, ep.as_ref())?;
            if let Some(span) = span.as_mut() {
                span.set_args(
                    Args::new()
                        .with("handles", handles.len() as u64)
                        .with("initializers", graph.initializers.len() as u64),
                );
            }
            handles
        };

        let mut value_shapes: HashMap<ValueId, Shape> = HashMap::new();
        let mut value_dtypes: HashMap<ValueId, DataType> = HashMap::new();
        let mut buffers: HashMap<ValueId, DeviceBuffer> = HashMap::new();
        let mut buffer_shapes: HashMap<ValueId, Vec<usize>> = HashMap::new();

        // 1) Initializers: record metadata and back resident consumers with a
        //    device buffer. A non-host nxrt initializer used exclusively at the
        //    lazy fused-MoE boundary deliberately has no eager buffer; the EP
        //    materializes it through its WeightHandle on demand. If any resident
        //    consumer (or graph output) coexists, no handle is built and the one
        //    eager buffer is shared by every consumer. Host mmap bytes retain the
        //    existing zero-copy borrow path.
        let init_align = TensorLayout::contiguous().alignment;
        let mut initializer_span = trace_span("session.initializer_buffers", "session");
        let mut initializer_count = 0_u64;
        let mut initializer_bytes = 0_u64;
        let mut borrowed_initializers = 0_u64;
        let mut copied_initializers = 0_u64;
        let mut lazy_initializers = 0_u64;
        for (&vid, weight) in &graph.initializers {
            let dtype = weight.dtype();
            let dims = weight.dims().to_vec();
            value_dtypes.insert(vid, dtype);
            value_shapes.insert(vid, dims.iter().map(|&d| Dim::Static(d)).collect());
            if !ep.device_id().is_host_accessible() && weight_handles.contains_key(&vid) {
                if initializer_span.is_some() {
                    lazy_initializers += 1;
                }
                continue;
            }
            let bytes = weights.bytes(weight).ok_or_else(|| {
                SessionError::Internal(format!("weight bytes unavailable for value#{}", vid.0))
            })?;
            if initializer_span.is_some() {
                initializer_count += 1;
                initializer_bytes += bytes.len() as u64;
            }
            // Only borrow when the value has NO producer. The borrowed
            // `DeviceBuffer` aliases read-only mmap/inline storage, so it must
            // never be written. A legitimate initializer always has
            // `producer == None`; a malformed graph can reuse an initializer's
            // `ValueId` as a node output (see loader `validate_no_initializer_producer`),
            // giving it a producer — a kernel would then write through
            // `as_mut_ptr()` into read-only mmap (SIGSEGV / aliasing UB). In
            // that case fall back to the owned writable copy below.
            let producer_less = graph.value(vid).producer.is_none();
            let borrow_align = if matches!(weight, WeightRef::External { .. }) {
                host_dtype_alignment(dtype)
            } else {
                init_align
            };
            let buf = if ep.device_id().is_host_accessible()
                && producer_less
                && !bytes.is_empty()
                && (bytes.as_ptr() as usize).is_multiple_of(borrow_align)
            {
                if initializer_span.is_some() {
                    borrowed_initializers += 1;
                }
                // Zero-copy: alias the suitably aligned initializer bytes. For
                // external data this is only the dtype alignment; inline data
                // retains the EP allocation alignment requirement.
                // SAFETY: `bytes` borrows live mmap storage in `weights` or
                // inline storage in `graph`; both executor fields outlive every
                // buffer use. The range is `bytes.len()` long,
                // `borrow_align`-aligned, and treated as read-only.
                unsafe {
                    DeviceBuffer::from_borrowed_parts(
                        bytes.as_ptr() as *mut std::ffi::c_void,
                        ep.device_id(),
                        bytes.len(),
                        borrow_align,
                    )
                }
            } else {
                if initializer_span.is_some() {
                    copied_initializers += 1;
                }
                let mut owned = ep.allocate(bytes.len().max(1), init_align)?;
                ep.copy_from_host(bytes, &mut owned)?;
                owned
            };
            buffer_shapes.insert(vid, dims);
            buffers.insert(vid, buf);
        }
        if let Some(span) = initializer_span.as_mut() {
            span.set_args(
                Args::new()
                    .with("initializers", initializer_count)
                    .with("bytes", initializer_bytes)
                    .with("borrowed_initializers", borrowed_initializers)
                    .with("copied_initializers", copied_initializers)
                    .with("lazy_initializers", lazy_initializers)
                    .with("buffers", buffers.len() as u64),
            );
        }

        // 2) Record the loader shape + dtype of every remaining value (graph
        //    inputs and node outputs). No allocation yet — shapes may be
        //    symbolic and are only sized once resolved.
        for &vid in &graph.inputs {
            value_shapes
                .entry(vid)
                .or_insert_with(|| graph.value(vid).shape.clone());
            value_dtypes.entry(vid).or_insert(graph.value(vid).dtype);
        }
        for &nid in &order {
            for &out in &graph.node(nid).outputs {
                value_shapes
                    .entry(out)
                    .or_insert_with(|| graph.value(out).shape.clone());
                value_dtypes.entry(out).or_insert(graph.value(out).dtype);
            }
        }

        let has_symbols = value_shapes.values().any(|s| as_static_shape(s).is_none());

        // Sequence-typed values own no tensor buffer: a Sequence op stores its
        // list in `sequences` at run time. Mark every value produced by a
        // sequence-producing op so buffer sizing skips it (and so a Sequence
        // graph output is diagnosed cleanly rather than read as tensor bytes).
        let mut sequence_values: HashSet<ValueId> = HashSet::new();
        for &nid in &order {
            let node = graph.node(nid);
            if produces_sequence_output(&node.op_type, &node.domain) {
                for &out in &node.outputs {
                    sequence_values.insert(out);
                }
            }
        }

        // Output value ids of every control-flow node, used to seed their
        // concrete (branch-selected) shapes into the capture plan so downstream
        // capturable consumers do not each form an eager seam.
        let mut control_flow_output_values: HashSet<ValueId> = HashSet::new();
        for &nid in &order {
            let node = graph.node(nid);
            if is_control_flow_op(&node.op_type, &node.domain) {
                for &out in &node.outputs {
                    control_flow_output_values.insert(out);
                }
            }
        }

        // 3) Build the structural per-node plan.
        let mut plan_span = trace_span("session.execution_plan", "session");
        let mut plan = Vec::with_capacity(order.len());
        let mut skipped_epcontext = 0_u64;
        for &nid in &order {
            let node = graph.node(nid);
            // EPContext nodes are pre-compiled: they bypass placement and were
            // already restored through their owning EP by the session's
            // consume path (§55.3). They must never be resolved as ordinary
            // kernels — the CPU EP has no `EPContext` kernel — so skip them
            // here.
            if onnx_runtime_loader::is_ep_context_op(&node.op_type, &node.domain) {
                if plan_span.is_some() {
                    skipped_epcontext += 1;
                }
                continue;
            }
            // Preserve positional input arity: keep interior `None` (omitted
            // optional) slots so a later present input is not misread as the
            // omitted one, but trim trailing `None`s (a trailing omitted
            // optional just lowers the arity, matching ONNX semantics).
            let mut slots: Vec<Option<ValueId>> = node.inputs.clone();
            while matches!(slots.last(), Some(None)) {
                slots.pop();
            }
            let inputs = slots;
            let outputs: Vec<ValueId> = node.outputs.clone();
            let input_dtypes: Vec<DataType> = inputs
                .iter()
                .map(|v| {
                    v.map(|vid| value_dtypes[&vid])
                        .unwrap_or(DataType::Undefined)
                })
                .collect();
            let output_dtypes: Vec<DataType> = outputs.iter().map(|v| value_dtypes[v]).collect();
            plan.push(NodePlan {
                node_id: nid,
                inputs,
                outputs,
                input_dtypes,
                output_dtypes,
            });
        }
        if let Some(span) = plan_span.as_mut() {
            span.set_args(
                Args::new()
                    .with("topological_nodes", order.len() as u64)
                    .with("plan_len", plan.len() as u64)
                    .with("skipped_epcontext_nodes", skipped_epcontext)
                    .with("values", graph.values.len() as u64)
                    .with("inputs", graph.inputs.len() as u64)
                    .with("outputs", graph.outputs.len() as u64)
                    .with("has_symbols", has_symbols),
            );
        }

        // 4) name → value id and the set of caller-required inputs.
        let mut input_index = HashMap::new();
        let mut required_inputs = Vec::new();
        for &vid in &graph.inputs {
            if graph.initializers.contains_key(&vid) {
                continue; // pre-filled; not a caller input
            }
            required_inputs.push(vid);
            if let Some(name) = &graph.value(vid).name {
                input_index.insert(name.clone(), vid);
            }
        }

        // Full name → value id map (every named value in the graph), used to
        // resolve a nested subgraph's outer-scope captures by name.
        let mut name_index = HashMap::new();
        for (vid, value) in graph.values.iter() {
            if let Some(name) = &value.name {
                name_index.insert(name.clone(), vid);
            }
        }

        let mut exec = Self {
            graph,
            weights,
            ep,
            weight_handles,
            buffers,
            buffer_shapes,
            value_shapes,
            value_dtypes,
            plan,
            input_index,
            required_inputs,
            has_symbols,
            cache: KernelCache::default(),
            name_index,
            subgraph_execs: HashMap::new(),
            control_flow_stats: ControlFlowStats::default(),
            if_last_predicate: HashMap::new(),
            device_graph_signature: None,
            capture_schedule: None,
            capture_segmentation: Vec::new(),
            control_flow_output_values,
            capture_cf_shapes: HashMap::new(),
            capture_warm_signature: None,
            capture_warm_shapes: HashMap::new(),
            capture_warm_seeded: HashMap::new(),
            capture_quarantine_ops: HashSet::new(),
            last_capture_failed_node: None,
            views: HashMap::new(),
            pinned: HashSet::new(),
            sequence_values,
            shared_buffers: HashMap::new(),
            sequences: HashMap::new(),
            seq_elem_values: HashMap::new(),
            execution_provider_fallback_report,
            trace: TraceContext::noop(),
            scratch_input_shapes: Vec::new(),
            decode_memo_enabled: decode_memo_env_enabled(),
            decode_memo_verify: cfg!(debug_assertions) || decode_memo_verify_env_enabled(),
            decode_memo: None,
            decode_memo_prev_bindings: None,
            decode_memo_last_action: DecodeMemoAction::Disabled,
            decode_memo_resolved: HashMap::new(),
            decode_memo_primed_count: 0,
            decode_memo_rebuilt_count: 0,
            decode_memo_replayed_count: 0,
            decode_memo_ineligible_count: 0,
            decode_view_plan: None,
            decode_views_reused_count: 0,
            decode_dispatch_elided_count: 0,
            decode_view_plan_sig_mismatch_streak: 0,
            decode_view_plan_disabled: false,
        };

        // 5) Fully-static graphs are materialized eagerly (buffers + the whole
        //    "compiled plan" of kernels), so the first `run` sees only cache
        //    hits. Symbolic graphs cannot be sized until a `run` fixes their
        //    shapes, so their buffers/kernels are created on first use.
        if !exec.has_symbols {
            let mut span = trace_span("session.static_materialize", "session");
            let empty = HashMap::new();
            let resolved = exec.resolve_all(&empty)?;
            exec.size_buffers(&resolved)?;
            exec.compile_all(&resolved)?;
            if let Some(span) = span.as_mut() {
                span.set_args(
                    Args::new()
                        .with("resolved_values", resolved.len() as u64)
                        .with("buffers", exec.buffers.len() as u64)
                        .with("plan_len", exec.plan.len() as u64)
                        .with("cache_entries", exec.cache.stats().entries as u64),
                );
            }
        }
        Ok(exec)
    }

    /// Allocate `vid`'s buffer for `dims`, or reuse the existing allocation when
    /// it is already sized for `dims` (the run-scoped reuse path).
    pub(super) fn ensure_buffer(&mut self, vid: ValueId, dtype: DataType, dims: &[usize]) -> Result<()> {
        if self.buffer_shapes.get(&vid).map(|s| s.as_slice()) == Some(dims) {
            return Ok(()); // identical shape → reuse allocation
        }
        if let Some(old) = self.buffers.remove(&vid) {
            self.ep.deallocate(old)?;
        }
        self.shared_buffers.remove(&vid);
        let numel = checked_numel(dims, || format!("value#{}", vid.0))?;
        let size = checked_storage_bytes(dtype, numel, || format!("value#{}", vid.0), dims)?;
        let buf = self
            .ep
            .allocate(size.max(1), TensorLayout::contiguous().alignment)?;
        self.buffers.insert(vid, buf);
        self.buffer_shapes.insert(vid, dims.to_vec());
        Ok(())
    }

    /// Resolve every value's concrete shape by substituting `bindings` into its
    /// loader shape. A value whose shape stays symbolic (unbound) cannot be
    /// sized: report it as an uninferred shape, naming its producing op.
    pub(super) fn resolve_all(
        &self,
        bindings: &HashMap<SymbolId, usize>,
    ) -> Result<HashMap<ValueId, Vec<usize>>> {
        let mut resolved = HashMap::with_capacity(self.value_shapes.len());
        for (&vid, shape) in &self.value_shapes {
            // Sequence-typed values have no meaningful tensor shape and are
            // never buffer-sized; skip them so a static graph does not trip the
            // unresolved-shape check on a sequence value.
            if self.sequence_values.contains(&vid) {
                continue;
            }
            match substitute(shape, bindings) {
                Some(dims) => {
                    resolved.insert(vid, dims);
                }
                None => {
                    let value = self.graph.value(vid);
                    let name = value
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("value#{}", vid.0));
                    let op = value
                        .producer
                        .map(|nid| self.graph.node(nid).op_type.clone())
                        .unwrap_or_else(|| "<graph input>".to_string());
                    return Err(SessionError::UnresolvedShape { value: name, op });
                }
            }
        }
        Ok(resolved)
    }

    /// Like [`Self::resolve_all`] but never errors: values whose shape stays
    /// symbolic (a data-dependent extent the loader could not pin down) are
    /// simply omitted, to be resolved just-in-time during execution once their
    /// producing node's inputs are concrete.
    pub(super) fn resolve_soft(&self, bindings: &HashMap<SymbolId, usize>) -> HashMap<ValueId, Vec<usize>> {
        let mut resolved = HashMap::with_capacity(self.value_shapes.len());
        for (&vid, shape) in &self.value_shapes {
            if let Some(dims) = substitute(shape, bindings) {
                resolved.insert(vid, dims);
            }
        }
        resolved
    }

    /// F5 Stage 1: resolve every value's concrete shape for a memo-eligible
    /// eager step, replaying the length-invariant partition through the
    /// [`DecodePlanMemo`] when the step is plan-identical to the memoized one,
    /// and re-substituting only the length-varying tail. On any signature change
    /// (prefill→decode, batch change, non-length dim change, …) it falls back to
    /// a full [`Self::resolve_soft`] and (re)builds the memo by diffing this
    /// step's bindings with the previous eligible step's (R1 two-real-step
    /// derivation). The output is provably byte-identical to `resolve_soft`
    /// (asserted every replay when [`Self::decode_memo_verify`] is set).
    pub(super) fn resolve_soft_decode_memo(
        &mut self,
        bindings: &HashMap<SymbolId, usize>,
        external: &ExternalBindings,
    ) -> HashMap<ValueId, Vec<usize>> {
        // L-abstracted fingerprint of the persistent binding set (KV cache). Pure
        // length growth leaves it unchanged; a structural change forces a rebuild.
        let external_sig = self.decode_external_signature(external);
        // --- Fast path: an active memo whose non-varying bindings and binding
        //     signature are unchanged. Replays the invariant partition with ZERO
        //     allocation: the persistent working map is taken in place, the
        //     previous step's just-in-time entries are stripped, invariant entries
        //     are left untouched (byte-identical by construction), and only the
        //     variant tail is re-substituted into its existing `Vec`s.
        if self
            .decode_memo
            .as_ref()
            .is_some_and(|memo| memo.matches(bindings, &external_sig))
        {
            // Own the memo for the duration so `self.value_shapes` /
            // `decode_memo_resolved` can be borrowed disjointly; restored below.
            let memo = self.decode_memo.take().unwrap();
            let mut resolved = std::mem::take(&mut self.decode_memo_resolved);
            // Drop the previous step's data-dependent (JIT) entries so the run
            // loop recomputes them; the canonical partition is retained in place.
            resolved.retain(|vid, _| memo.canonical.contains(vid));
            // Restore any length-invariant entry missing from the persistent map.
            // By construction (the run loop only adds/overwrites, never drops
            // canonical keys, and the rebuild step persisted the full map) this
            // never fires in steady state, so replay stays allocation-free; it is
            // a defensive re-seed from the memo's authoritative invariant plan.
            for (&vid, dims) in &memo.invariant_shapes {
                resolved.entry(vid).or_insert_with(|| dims.clone());
            }
            // Re-substitute only the variant tail, reusing each `Vec`'s capacity.
            for &vid in &memo.variant_values {
                let shape = &self.value_shapes[&vid];
                match resolved.get_mut(&vid) {
                    Some(slot) => {
                        if !substitute_into(shape, bindings, slot) {
                            resolved.remove(&vid);
                        }
                    }
                    None => {
                        if let Some(dims) = substitute(shape, bindings) {
                            resolved.insert(vid, dims);
                        }
                    }
                }
            }
            if self.decode_memo_verify {
                // R1 verifiable safety net: the replay must equal a fresh resolve.
                let fresh = self.resolve_soft(bindings);
                assert_eq!(
                    resolved, fresh,
                    "decode-plan memo replay diverged from resolve_soft (unsound invariant \
                     classification)"
                );
            }
            self.decode_memo = Some(memo);
            self.decode_memo_last_action = DecodeMemoAction::Replayed;
            self.decode_memo_replayed_count += 1;
            self.decode_memo_prev_bindings = Some(bindings.clone());
            return resolved;
        }

        // --- Slow path: full resolve, then try to (re)build the memo by diffing
        //     this step with the previous eligible step (two real steps, R1) —
        //     but only for a steady single-token-decode growth transition (M==1
        //     gate), so the memo never activates on prefill.
        //
        // Defense-in-depth (Chew): drop the persistent working map on every
        // non-replay step so a stale invariant `Vec` from a retired plan can
        // never leak into a future replay (e.g. if a run errored before the
        // end-of-step persist-back). It is repopulated by this step's persist-back
        // (or, if that step errors, left empty and defensively re-seeded next
        // replay), so the clear costs nothing on the steady path.
        self.decode_memo_resolved.clear();
        // F5 Stage 2 defense-in-depth: retire the cached view plan on every
        // non-replay (rebuild/prime) step. A Rebuilt step rebuilds it fresh only
        // at its successful end (below, in `run_scoped_mode`); a step that errors
        // before that leaves it `None`, so a stale invariant view alias from a
        // retired plan can never be reinstated into a later replay.
        self.decode_view_plan = None;
        let resolved = self.resolve_soft(bindings);
        match self.decode_memo_prev_bindings.take() {
            Some(prev) if is_decode_growth_transition(&prev, bindings) => {
                let decode_varying: HashSet<SymbolId> = bindings
                    .iter()
                    .filter(|(sym, val)| prev.get(*sym) != Some(*val))
                    .map(|(&sym, _)| sym)
                    .collect();
                let mut invariant_shapes = HashMap::with_capacity(resolved.len());
                let mut variant_values = Vec::new();
                let mut canonical = HashSet::with_capacity(resolved.len());
                for (&vid, dims) in &resolved {
                    canonical.insert(vid);
                    if shape_references_any(&self.value_shapes[&vid], &decode_varying) {
                        variant_values.push(vid);
                    } else {
                        invariant_shapes.insert(vid, dims.clone());
                    }
                }
                self.decode_memo = Some(DecodePlanMemo {
                    reference_bindings: bindings.clone(),
                    decode_varying,
                    invariant_shapes,
                    variant_values,
                    canonical,
                    reference_external_sig: external_sig,
                });
                self.decode_memo_last_action = DecodeMemoAction::Rebuilt;
                self.decode_memo_rebuilt_count += 1;
            }
            _ => {
                // First observation of a regime, a bound-symbol-set change, or a
                // non-decode transition (e.g. prefill): drop any stale memo and
                // wait for the next steady-decode step to diff against.
                self.decode_memo = None;
                self.decode_memo_last_action = DecodeMemoAction::Primed;
                self.decode_memo_primed_count += 1;
            }
        }
        self.decode_memo_prev_bindings = Some(bindings.clone());
        resolved
    }

    /// L-abstracted structural fingerprint of the persistent device-I/O binding
    /// set (see [`DecodeBindingSig`]). Order-independent; the declared symbolic
    /// shape (graph-static) stands in for the concrete one, so pure-L KV growth
    /// yields an unchanged signature while a binding added/removed, a role flip,
    /// or a dtype change yields a different one.
    pub(super) fn decode_external_signature(&self, external: &ExternalBindings) -> Vec<DecodeBindingSig> {
        let mut sig: Vec<DecodeBindingSig> = external
            .inputs
            .keys()
            .map(|&vid| (vid, true))
            .chain(external.outputs.keys().map(|&vid| (vid, false)))
            .map(|(vid, is_input)| DecodeBindingSig {
                vid,
                is_input,
                dtype: self.value_dtypes[&vid],
                decl_shape: self.value_shapes[&vid].clone(),
            })
            .collect();
        sig.sort_by_key(|s| (s.vid.0, s.is_input));
        sig
    }

    #[cfg(test)]
    pub(super) fn set_decode_memo_enabled(&mut self, enabled: bool) {
        self.decode_memo_enabled = enabled;
        self.decode_memo_verify = true;
        self.decode_memo = None;
        self.decode_memo_prev_bindings = None;
        self.decode_memo_resolved.clear();
        self.decode_memo_last_action = DecodeMemoAction::Disabled;
        self.decode_memo_primed_count = 0;
        self.decode_memo_rebuilt_count = 0;
        self.decode_memo_replayed_count = 0;
        self.decode_memo_ineligible_count = 0;
        self.decode_view_plan = None;
        self.decode_views_reused_count = 0;
        self.decode_dispatch_elided_count = 0;
        self.decode_view_plan_sig_mismatch_streak = 0;
        self.decode_view_plan_disabled = false;
    }

    #[cfg(test)]
    pub(super) fn decode_memo_action(&self) -> DecodeMemoAction {
        self.decode_memo_last_action
    }

    /// F5 Stage 1 memo activity counters `(primed, rebuilt, replayed, ineligible)`
    /// accumulated over this executor's lifetime. `replayed > 0` on a real decode
    /// run is the proof the memo actually fires (not silently gated out); the
    /// coordinator's on-model A/B reads these to reject a vacuous pass.
    pub(crate) fn decode_memo_counts(&self) -> (u64, u64, u64, u64) {
        (
            self.decode_memo_primed_count,
            self.decode_memo_rebuilt_count,
            self.decode_memo_replayed_count,
            self.decode_memo_ineligible_count,
        )
    }

    /// F5 Stage 2 activity counters `(views_reused, dispatch_elided)` accumulated
    /// over this executor's lifetime. Both `> 0` on a real decode run prove the
    /// invariant view-reuse / dispatch-elision path actually fired (not a vacuous
    /// pass); an on-model A/B reads these alongside the Stage-1 counters.
    pub(crate) fn decode_view_plan_counts(&self) -> (u64, u64) {
        (
            self.decode_views_reused_count,
            self.decode_dispatch_elided_count,
        )
    }

    /// F5 Stage 2 replay guard: every retained view's source buffer must still be
    /// the identical allocation (same base pointer *and* capacity) it was under
    /// when the plan was built. A realloc or move — even one that preserves the
    /// logical shape — invalidates the cached byte offsets/strides, so this must
    /// return `false` and force a full rebuild. This is the pointer/capacity
    /// obligation Stage 1 deferred (it cached shapes only); Stage 2 pays it here.
    pub(super) fn stage2_buffer_sig_matches(&self, plan: &DecodeViewPlan) -> bool {
        plan.source_buffer_sig.iter().all(|(vid, ptr, cap)| {
            self.buffers
                .get(vid)
                .is_some_and(|buf| buf.as_ptr() as usize == *ptr && buf.len() == *cap)
        })
    }

    /// F5 Stage 2: build the *candidate* view plan from the state left by a
    /// successful memo Rebuilt step. A node is a candidate iff every one of its
    /// outputs is a zero-copy view (`self.views`) whose **shape is in the memo's
    /// proven-invariant partition** — so Stage 1 guarantees the replayed `resolved`
    /// map carries that exact shape every step. The candidate's source buffers can
    /// still be classified variant (e.g. a fixed-capacity KV buffer whose logical
    /// length grows): its concrete stability is confirmed separately by
    /// [`Self::validate_decode_view_plan`] (byte-identical view across a second real
    /// step) and guarded each replay by the buffer-identity signature. Returns
    /// `None` if nothing is a candidate.
    pub(super) fn build_decode_view_plan(&self) -> Option<DecodeViewPlan> {
        let memo = self.decode_memo.as_ref()?;
        let invariant = |vid: &ValueId| memo.invariant_shapes.contains_key(vid);
        let mut elided_nodes = HashSet::new();
        let mut retained_views = Vec::new();
        let mut sources: HashSet<ValueId> = HashSet::new();
        for pi in 0..self.plan.len() {
            let outputs = &self.plan[pi].outputs;
            if outputs.is_empty() {
                continue;
            }
            // Every output must be a zero-copy view whose shape Stage 1 already
            // proves invariant (so `resolved[output]` is stable and correct when
            // the node is elided).
            let all_view_invariant = outputs
                .iter()
                .all(|ovid| invariant(ovid) && self.views.contains_key(ovid));
            if !all_view_invariant {
                continue;
            }
            elided_nodes.insert(pi);
            for ovid in outputs {
                let view = self.views[ovid].clone();
                sources.insert(view.source);
                retained_views.push((*ovid, view));
            }
        }
        if elided_nodes.is_empty() {
            return None;
        }
        // Record the buffer identity of every aliased source (the Stage-2 guard).
        let mut source_buffer_sig = Vec::with_capacity(sources.len());
        for &src in &sources {
            let buf = self.buffers.get(&src)?;
            source_buffer_sig.push((src, buf.as_ptr() as usize, buf.len()));
        }
        Some(DecodeViewPlan {
            elided_nodes,
            retained_views,
            pinned_sources: sources.into_iter().collect(),
            source_buffer_sig,
            validated: false,
        })
    }

    /// F5 Stage 2: confirm a candidate plan on a second real decode step. The step
    /// ran every node normally (no elision), so `self.views` now holds freshly
    /// built aliases; keep only the candidate nodes whose every output view is
    /// **byte-identical** (source, shape, strides, byte offset) to the one captured
    /// when the plan was built. This two-real-step confirmation (mirroring Stage 1's
    /// varying-set derivation) rejects any view whose geometry actually drifts — e.g.
    /// a position-indexed slice into a fixed-capacity buffer — before it is ever
    /// elided. Sources and the buffer-identity signature are recomputed from the
    /// surviving views. The plan is marked validated iff anything survives.
    pub(super) fn validate_decode_view_plan(&self, mut plan: DecodeViewPlan) -> Option<DecodeViewPlan> {
        let view_matches = |a: &ValueView, b: &ValueView| {
            a.source == b.source
                && a.shape == b.shape
                && a.strides == b.strides
                && a.byte_offset == b.byte_offset
        };
        // A node survives iff every one of its retained outputs still matches the
        // freshly rebuilt view this step.
        let mut surviving_nodes: HashSet<usize> = HashSet::new();
        let node_outputs = |pi: usize| self.plan[pi].outputs.clone();
        for &pi in &plan.elided_nodes {
            let ok = node_outputs(pi).iter().all(|ovid| {
                match (
                    plan.retained_views.iter().find(|(v, _)| v == ovid),
                    self.views.get(ovid),
                ) {
                    (Some((_, cached)), Some(fresh)) => view_matches(cached, fresh),
                    _ => false,
                }
            });
            if ok {
                surviving_nodes.insert(pi);
            }
        }
        if surviving_nodes.is_empty() {
            return None;
        }
        // Rebuild retained views / sources / signature from the survivors only,
        // using the freshly built (identical) views.
        let surviving_outputs: HashSet<ValueId> = surviving_nodes
            .iter()
            .flat_map(|&pi| self.plan[pi].outputs.clone())
            .collect();
        let mut retained_views = Vec::new();
        let mut sources: HashSet<ValueId> = HashSet::new();
        for ovid in surviving_outputs {
            let view = self.views.get(&ovid)?.clone();
            sources.insert(view.source);
            retained_views.push((ovid, view));
        }
        let mut source_buffer_sig = Vec::with_capacity(sources.len());
        for &src in &sources {
            let buf = self.buffers.get(&src)?;
            source_buffer_sig.push((src, buf.as_ptr() as usize, buf.len()));
        }
        plan.elided_nodes = surviving_nodes;
        plan.retained_views = retained_views;
        plan.pinned_sources = sources.into_iter().collect();
        plan.source_buffer_sig = source_buffer_sig;
        plan.validated = true;
        Some(plan)
    }

    /// Size (allocate or reuse) a backing buffer for every value from its
    /// resolved concrete shape. Initializers already hold their weights and are
    /// left untouched. Values whose shape is not (yet) in `resolved` — the
    /// data-dependent ones filled in during execution — are skipped here and
    /// sized just-in-time in the run loop.
    pub(super) fn size_buffers(&mut self, resolved: &HashMap<ValueId, Vec<usize>>) -> Result<()> {
        self.size_buffers_excluding(resolved, &HashSet::new())
    }

    pub(super) fn size_buffers_excluding(
        &mut self,
        resolved: &HashMap<ValueId, Vec<usize>>,
        excluded: &HashSet<ValueId>,
    ) -> Result<()> {
        let vids: Vec<ValueId> = self.value_shapes.keys().copied().collect();
        for vid in vids {
            if self.graph.initializers.contains_key(&vid) || excluded.contains(&vid) {
                continue;
            }
            // Sequence-typed values own no tensor buffer (their list lives in
            // `sequences` at run time), so never size one for them.
            if self.sequence_values.contains(&vid) {
                continue;
            }
            let dtype = self.value_dtypes[&vid];
            let Some(dims) = resolved.get(&vid).cloned() else {
                continue;
            };
            self.ensure_buffer(vid, dtype, &dims)?;
        }
        Ok(())
    }

    /// Resolved input shapes of a plan node, in positional order. An omitted
    /// optional input (`None` slot) has no shape; it takes an empty shape,
    /// which the run loop only ever pairs with an absent placeholder view.
    pub(super) fn node_input_shapes(
        plan: &NodePlan,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Vec<Vec<usize>> {
        plan.inputs
            .iter()
            .map(|v| v.map(|vid| resolved[&vid].clone()).unwrap_or_default())
            .collect()
    }

    /// Populate the kernel cache for the compiled plan against `resolved` shapes.
    pub(super) fn compile_all(&mut self, resolved: &HashMap<ValueId, Vec<usize>>) -> Result<()> {
        let mut span = trace_span("session.kernel_compile_plan", "session");
        let cache_entries_before = self.cache.stats().entries;
        let mut compiled_nodes = 0_u64;
        let mut skipped_control_flow = 0_u64;
        let mut skipped_sequence = 0_u64;
        for i in 0..self.plan.len() {
            let node_id = self.plan[i].node_id;
            let node = self.graph.node(node_id);
            // Control-flow ops (If/Loop/Scan) are not leaf kernels — they execute
            // nested subgraphs through the executor's own path, so they have no
            // entry in the EP kernel registry and must not be compiled here.
            if is_control_flow_op(&node.op_type, &node.domain) {
                if span.is_some() {
                    skipped_control_flow += 1;
                }
                continue;
            }
            // Sequence ops are executor-handled (they operate on sequence-of-
            // tensor values, not tensor views) — they have no EP kernel and must
            // not be compiled here, exactly like control-flow ops.
            if is_sequence_op(&node.op_type, &node.domain) {
                if span.is_some() {
                    skipped_sequence += 1;
                }
                continue;
            }
            if span.is_some() {
                compiled_nodes += 1;
            }
            let input_shapes = Self::node_input_shapes(&self.plan[i], resolved);
            let input_dtypes = self.plan[i].input_dtypes.clone();
            let constant_inputs: Vec<bool> = self.plan[i]
                .inputs
                .iter()
                .map(|input| input.is_some_and(|vid| self.graph.initializers.contains_key(&vid)))
                .collect();
            let node = self.graph.node(node_id);
            let opset = effective_opset(&self.graph, node);
            self.cache.get_or_create(
                node_id,
                node,
                &input_shapes,
                &input_dtypes,
                &constant_inputs,
                opset,
                self.ep.as_ref(),
            )?;
        }
        if let Some(span) = span.as_mut() {
            span.set_args(
                Args::new()
                    .with("plan_len", self.plan.len() as u64)
                    .with("compiled_nodes", compiled_nodes)
                    .with("skipped_control_flow", skipped_control_flow)
                    .with("skipped_sequence", skipped_sequence)
                    .with("cache_entries_before", cache_entries_before as u64)
                    .with("cache_entries_after", self.cache.stats().entries as u64),
            );
        }
        Ok(())
    }

    pub(crate) fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    pub(crate) fn control_flow_stats(&self) -> ControlFlowStats {
        self.control_flow_stats
    }

    pub(crate) fn device_id(&self) -> onnx_runtime_ir::DeviceId {
        self.ep.device_id()
    }

    pub(crate) fn allocate_device_binding(
        &self,
        input_name: String,
        output_name: Option<String>,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
    ) -> Result<DeviceIoBinding> {
        let expose_logical_input_shape = self.input_index.get(&input_name).is_none_or(|&vid| {
            if output_name.is_some() {
                !self.binding_consumers_use_physical_capacity(vid)
            } else {
                !self.binding_consumers_use_padded_capacity(vid)
            }
        });
        DeviceIoBinding::allocate(
            self.ep.clone(),
            input_name,
            true,
            output_name,
            dtype,
            physical_shape,
            logical_shape,
            expose_logical_input_shape,
        )
    }

    pub(crate) fn allocate_device_output_binding(
        &self,
        output_name: String,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
    ) -> Result<DeviceIoBinding> {
        DeviceIoBinding::allocate(
            self.ep.clone(),
            String::new(),
            false,
            Some(output_name),
            dtype,
            physical_shape,
            logical_shape,
            false,
        )
    }

    pub(super) fn binding_consumers_use_physical_capacity(&self, input: ValueId) -> bool {
        let mut found = false;
        for plan in &self.plan {
            for (slot, value) in plan.inputs.iter().enumerate() {
                if *value != Some(input) {
                    continue;
                }
                found = true;
                if !kernel_input_uses_physical_capacity(self.graph.node(plan.node_id), slot) {
                    return false;
                }
            }
        }
        found
    }

    pub(super) fn binding_consumers_use_padded_capacity(&self, input: ValueId) -> bool {
        let mut found = false;
        for plan in &self.plan {
            for (slot, value) in plan.inputs.iter().enumerate() {
                if *value != Some(input) {
                    continue;
                }
                found = true;
                if !kernel_input_uses_padded_capacity(self.graph.node(plan.node_id), slot) {
                    return false;
                }
            }
        }
        found
    }

    /// The compiled graph, retained for the §55.4 EPContext dump path: the
    /// exporter needs the (post-optimize) graph to serialise a `*_ctx.onnx`
    /// context-cache model with compiled partitions spliced out.
    pub(crate) fn graph(&self) -> &Graph {
        &self.graph
    }

    pub(crate) fn execution_provider_fallback_report(
        &self,
    ) -> Option<&ExecutionProviderFallbackReport> {
        self.execution_provider_fallback_report.as_ref()
    }

    /// Attach the shared runtime trace context. When enabled, the executor opens
    /// one span per executed op so kernels can attach kernel-variant and
    /// capture-rejection reasons. Propagated to any already-built child
    /// (control-flow subgraph) executors so nested ops are traced too.
    pub(crate) fn set_trace_context(&mut self, trace: TraceContext) {
        for child in self.subgraph_execs.values_mut() {
            child.set_trace_context(trace.clone());
        }
        self.trace = trace;
    }

    /// Live weight bytes backing the graph, needed alongside [`Self::graph`] so
    /// the EPContext dump can encode initializers into the context model.
    pub(crate) fn weights(&self) -> &Arc<WeightStore> {
        &self.weights
    }

    /// Warmup: re-touch the shape-keyed cache for the compiled plan so the first
    /// real `run` sees only cache hits (§11.3). Only meaningful for fully-static
    /// graphs, whose plan shapes are known at build; symbolic graphs cannot be
    /// pre-compiled without a concrete shape and warm up on their first `run`.
    pub(crate) fn warmup(&mut self) -> Result<()> {
        if self.has_symbols {
            return Ok(());
        }
        let empty = HashMap::new();
        let resolved = self.resolve_all(&empty)?;
        self.compile_all(&resolved)
    }
}
