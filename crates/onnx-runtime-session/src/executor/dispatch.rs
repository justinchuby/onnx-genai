use super::*;
use onnx_runtime_ep_api::{
    ExecutionProviderCapabilities, ExternalMmapRegion, MmapRegionSource, WeightHandleError,
};

/// Per-input-slot result of the strided-input materialization gate: `Some` with
/// the gathered contiguous bytes and their strides when a private temp was
/// needed for that slot, `None` when the input was used in place.
type MaterializedInputs = Vec<Option<(Vec<u8>, Vec<i64>)>>;

/// Whether GroupQueryAttention with capacity-bound (persistently bound, aliased
/// in-place) present-KV outputs sizes its present shape from the host-known
/// past-KV physical capacity instead of reading `seqlens_k`/`total_sequence_length`
/// back to host per layer. On by default — this removes the dominant blocking
/// scalar-D2H cost of eager decode (the present extent equals the fixed capacity,
/// so the read-back is redundant). Disable via `ONNX_GENAI_GQA_SHAPE_ONDEVICE=0`
/// to restore the always-read-back path (escape hatch for debugging).
fn gqa_shape_capacity_bound_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ONNX_GENAI_GQA_SHAPE_ONDEVICE")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .is_none_or(|v| !matches!(v.as_str(), "0" | "false" | "off" | "no"))
    })
}

struct WeightStoreRegionSource<'a>(&'a onnx_runtime_loader::weights::WeightStore);

impl MmapRegionSource for WeightStoreRegionSource<'_> {
    fn region_bytes(
        &self,
        region: &ExternalMmapRegion,
    ) -> std::result::Result<&[u8], WeightHandleError> {
        self.0
            .mmap_region_bytes(region.mapping_id, region.offset, region.len)
            .ok_or_else(|| {
                WeightHandleError::InvalidResident(format!(
                    "external mmap region id={} offset={} len={} is no longer available",
                    region.mapping_id, region.offset, region.len
                ))
            })
    }

    fn full_mapping_bytes(&self, mapping_id: usize) -> Option<&[u8]> {
        self.0.mmap_full_bytes(mapping_id)
    }
}

impl Executor {
    /// Dispatch one plan node to its execution path (control-flow, sequence, or
    /// leaf kernel). Shared by the eager loop and the segmented runner.
    ///
    /// When tracing is enabled, opens one span per op so the dispatched kernel
    /// can attach kernel-variant and capture-rejection reasons via
    /// [`annotate_current_span_with`]; `capture` records the node's device-graph
    /// disposition onto that span. When tracing is disabled this costs a single
    /// relaxed atomic load and never allocates.
    pub(super) fn exec_plan_node(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
        capture: OpCaptureTrace<'_>,
    ) -> Result<()> {
        // Dispatch by op-type/domain borrowed straight from the node, so a
        // steady-state decode step compares `&str`s and never clones the
        // op-type/domain `String`s per node. The immutable borrow of
        // `self.graph` is confined to this block and dropped before the
        // `&mut self` dispatch below; the span guard it yields owns its name
        // (and a cheap `Arc`-clone of the trace context), so it borrows nothing
        // from `self` and can stay live across the dispatch.
        let (is_control_flow, is_sequence, _span) = {
            let node = self.graph.node(self.plan[pi].node_id);
            let is_control_flow = is_control_flow_op(&node.op_type, &node.domain);
            let is_sequence = is_sequence_op(&node.op_type, &node.domain);
            // Open the span only when tracing is live so an untraced decode step
            // never allocates a span name or touches the thread-local span stack.
            // Everything that clones a node field lives inside this closure for
            // the same reason: an untraced step must not pay for identity it is
            // never going to record.
            let span = self.trace.is_enabled().then(|| {
                // Every op span comes from this one line, so its source location
                // would be the same string on all of them; the node args below
                // identify each span far better. Keeping it cost 22% of a trace.
                let span = self.trace.span(node.op_type.clone(), "op").without_source();
                // Identify *which* node this is. The span name stays the bare op
                // type so Perfetto still aggregates all `MatMul`s together; the
                // identity rides along as args. A model has hundreds of
                // same-typed nodes, and without this a slow one cannot be told
                // from a fast one.
                //
                // Device is stamped here rather than by each kernel. Kernels have
                // to opt in to annotating themselves and in practice most never
                // do -- the CPU provider annotates 11 of its 122 kernels, the
                // CUDA provider annotated none -- so a per-kernel convention
                // leaves most of a trace unlabelled. The node's placement is
                // known here for every node on every provider, which makes the
                // coverage structural instead of something each kernel must
                // remember.
                annotate_current_span_with(|| {
                    let mut args = Args::new().with("node_id", node.id.0 as u64);
                    if !node.name.is_empty() {
                        args = args.with("node", node.name.clone());
                    }
                    // Only non-default domains are worth the bytes: `Attention`
                    // and `MatMulNBits` exist in both the default and
                    // `com.microsoft` domains, so the op type alone is ambiguous
                    // for custom ops.
                    if !node.domain.is_empty() {
                        args = args.with("domain", node.domain.clone());
                    }
                    if let Some(device) = node.device {
                        args = args.with(
                            onnx_runtime_ep_api::ARG_DEVICE,
                            device.device_type.trace_name().into_owned(),
                        );
                    }
                    args
                });
                // Span is now active on this thread; stamp the capture disposition
                // (and let the kernel below stamp its selected variant).
                capture.annotate();
                span
            });
            (is_control_flow, is_sequence, span)
        };
        if is_control_flow {
            self.exec_control_flow(pi, resolved, outer_scope, external)
        } else if is_sequence {
            self.exec_sequence_node(pi, resolved, external)
        } else {
            self.exec_kernel_node(pi, resolved, external, capture)
        }
    }

    /// Execute every plan node eagerly on the stream (no capture).
    ///
    /// F5 Stage 2: when `elided` is `Some`, the plan-node indices it contains are
    /// pure invariant view nodes whose zero-copy output aliases have already been
    /// reinstated into `self.views` for this step, so their re-dispatch is skipped.
    /// The set is empty (or `None`) on every non-Stage-2 run, so ordinary steps
    /// pay only one `HashSet::is_empty`/`contains` check per node.
    pub(super) fn run_plan_eager(
        &mut self,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
        elided: Option<&HashSet<usize>>,
    ) -> Result<()> {
        let elided = elided.filter(|set| !set.is_empty());
        if profile_ops_enabled() {
            let run_start = Instant::now();
            let mut timings: HashMap<String, (Duration, usize)> = HashMap::new();
            for pi in 0..self.plan.len() {
                if elided.is_some_and(|set| set.contains(&pi)) {
                    continue;
                }
                self.prefetch_lazy_weights_after(pi)?;
                let op_type = self.graph.node(self.plan[pi].node_id).op_type.clone();
                let start = Instant::now();
                let result =
                    self.exec_plan_node(pi, resolved, outer_scope, external, OpCaptureTrace::Eager);
                let elapsed = start.elapsed();
                let entry = timings.entry(op_type).or_insert((Duration::ZERO, 0));
                entry.0 += elapsed;
                entry.1 += 1;
                result?;
                self.release_dead_values(pi, external);
            }
            print_op_profile(run_start.elapsed(), timings);
        } else {
            for pi in 0..self.plan.len() {
                if elided.is_some_and(|set| set.contains(&pi)) {
                    continue;
                }
                self.prefetch_lazy_weights_after(pi)?;
                self.exec_plan_node(pi, resolved, outer_scope, external, OpCaptureTrace::Eager)?;
                self.release_dead_values(pi, external);
            }
        }
        Ok(())
    }

    /// Release the buffers of values whose last consumer was plan node `pi`.
    ///
    /// The guards mirror the in-place-overwrite path: anything the caller owns,
    /// anything aliased by a view, and anything a sequence or shared buffer still
    /// refers to is left alone. A value that fails a guard simply stays resident,
    /// which is the pre-existing behaviour for every value.
    /// Free the buffers of values whose last consumer was node `pi`.
    ///
    /// The guard list is deliberately the same one `inplace_input` applies,
    /// because both are asking the same question: may this buffer be taken away
    /// from the value that currently holds it? Any guard added there belongs
    /// here too.
    ///
    /// A value that is aliased -- by a view, by a sequence's shared storage --
    /// is never released, even after every alias has died. `pinned` records
    /// that a value was ever an alias source and is only cleared between runs,
    /// so it is a deliberately conservative over-approximation. Releasing on
    /// the exact liveness test (`views` no longer names it as a source) would
    /// be wrong for the decode memo, which restores a step's view table from
    /// `retained_views` on the *next* run and expects the source buffer to
    /// still be resident. Measured cost of the conservative choice on the
    /// vision encoder is 36 MiB out of ~12 GB reclaimed, which does not buy an
    /// aliasing analysis that has to be right across runs.
    fn release_dead_values(&mut self, pi: usize, external: &ExternalBindings) {
        if !self.release_dead_values_enabled {
            return;
        }
        let dead = std::mem::take(&mut self.plan[pi].dead_after);
        for &vid in &dead {
            if external.inputs.contains_key(&vid)
                || external.outputs.contains_key(&vid)
                || self.pinned.contains(&vid)
                || self.shared_buffers.contains_key(&vid)
                || self.seq_elem_values.contains_key(&vid)
                || self.views.contains_key(&vid)
            {
                continue;
            }
            // A memoized loop-invariant `If` skips its branch on later runs and
            // serves the outputs straight from the buffers the first run left
            // resident (see `exec_if`), and the memo outlives an eager run.
            // Freeing one of those outputs turns the next skip into a missing
            // buffer. `try_move_host_output` declines for the same reason.
            if let Some(producer) = self.graph.try_value(vid).and_then(|value| value.producer)
                && self.if_last_predicate.contains_key(&producer)
            {
                continue;
            }
            // A borrowed buffer aliases memory this session does not own.
            if self.buffers.get(&vid).is_some_and(|b| b.is_borrowed()) {
                continue;
            }
            // `DeviceBuffer` is a bare handle with no `Drop`: dropping it
            // strands the allocation, so the owning EP has to free it.
            if let Some(buffer) = self.buffers.remove(&vid) {
                // The reuse fast path treats a surviving shape entry as proof
                // the allocation is still there, so it has to go with it.
                self.buffer_shapes.remove(&vid);
                let _ = self.ep.deallocate(buffer);
            }
        }
        self.plan[pi].dead_after = dead;
    }

    fn prefetch_lazy_weights_after(&self, pi: usize) -> Result<()> {
        if self.prefetch_lookahead_nodes == 0 {
            return Ok(());
        }
        let Some(lookahead) = pi.checked_add(self.prefetch_lookahead_nodes) else {
            return Ok(());
        };
        let Some(next) = self.plan.get(lookahead) else {
            return Ok(());
        };
        for vid in &next.lazy_weight_inputs {
            if self.plan[pi].lazy_weight_inputs.contains(vid) {
                continue;
            }
            if let Some(lazy) = self
                .weight_handles
                .get(vid)
                .and_then(|handle| handle.as_lazy())
                && self.ep.prefetch_lazy_weight_for_executor(
                    self.instance_id,
                    vid.0 as u64,
                    lazy,
                    &WeightStoreRegionSource(self.weights.as_ref()),
                )?
            {
                self.prefetch_issue_nodes.lock().unwrap().insert(*vid, pi);
            }
        }
        Ok(())
    }

    /// Run the plan against a [`CaptureSchedule`], interleaving captured device
    /// graphs with eager seam nodes.
    ///
    /// * [`RunMode::Capture`] records each captured segment into its own device
    ///   graph, then immediately replays it so the following eager seam node
    ///   reads real bytes from the stable seam buffers. Eager seam nodes execute
    ///   normally on the stream (not recorded).
    /// * [`RunMode::Replay`] launches each captured segment's installed graph in
    ///   order and re-runs only the eager seam nodes.
    ///
    /// Seam correctness relies on the executor's per-value buffer reuse: for a
    /// fixed decode shape, intermediate buffers keep the same device address
    /// every step, so a captured segment and the eager node on either side of a
    /// seam always read and write the same stable buffers.
    pub(super) fn run_plan_segmented(
        &mut self,
        schedule: &CaptureSchedule,
        mode: RunMode,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
    ) -> Result<bool> {
        let ep = Arc::clone(&self.ep);
        // Set once a control-flow branch flip retires the installed graph mid
        // replay: every remaining node then runs eagerly (its captured segment's
        // baked device pointers are stale) so the step still produces a correct
        // token. Only ever set in `RunMode::Replay`.
        let mut invalidated = false;
        for seg in &schedule.segments {
            if invalidated {
                // Graph retired earlier this step: run this segment's nodes
                // eagerly instead of replaying a stale installed graph.
                for pi in seg.start..seg.end {
                    self.exec_plan_node(
                        pi,
                        resolved,
                        outer_scope,
                        external,
                        OpCaptureTrace::Eager,
                    )?;
                }
                continue;
            }
            if seg.captured {
                match mode {
                    RunMode::Capture => {
                        {
                            let kernels = self.collect_segment_kernels(seg, resolved)?;
                            ep.begin_device_graph_capture_for_executor(
                                self.instance_id,
                                self.graph_slot,
                                &kernels,
                            )?;
                        }
                        // Any early return (`?`) while recording this segment
                        // must end the stream capture before it propagates —
                        // otherwise the stream stays wedged in capture mode and
                        // the caller's `reset_device_graph()` is a no-op (reset
                        // is rejected while capturing). The guard aborts the
                        // capture on drop; `disarm()` hands off to the normal
                        // `end_device_graph_capture()` on the success path.
                        let mut capture_guard = SegmentCaptureGuard::arm(
                            ep.as_ref(),
                            self.instance_id,
                            self.graph_slot,
                        );
                        for pi in seg.start..seg.end {
                            let node_id = self.plan[pi].node_id;
                            if let Err(error) = self.exec_plan_node(
                                pi,
                                resolved,
                                outer_scope,
                                external,
                                OpCaptureTrace::Captured,
                            ) {
                                // Record which node aborted recording so the
                                // capture retry loop can quarantine its op-type.
                                // `capture_guard` drops here, ending the wedged
                                // stream capture before the error propagates.
                                self.last_capture_failed_node = Some(node_id);
                                return Err(error);
                            }
                        }
                        capture_guard.disarm();
                        ep.end_device_graph_capture_for_executor(
                            self.instance_id,
                            self.graph_slot,
                        )?;
                        // Capture is closed: syncs are legal again. Free any
                        // view-owner buffers that `install_view_outputs` parked
                        // while recording (freeing mid-capture is rejected).
                        for old in self.capture_deferred_frees.drain(..) {
                            ep.deallocate(old)?;
                        }
                        ep.replay_device_graph_segment_for_executor(
                            self.instance_id,
                            self.graph_slot,
                            seg.graph_index,
                        )?;
                    }
                    RunMode::Replay => {
                        ep.replay_device_graph_segment_for_executor(
                            self.instance_id,
                            self.graph_slot,
                            seg.graph_index,
                        )?;
                    }
                    RunMode::Eager => {
                        unreachable!("eager runs never build a segment schedule")
                    }
                }
            } else {
                for pi in seg.start..seg.end {
                    // Seam node: eager because some kernel/predicate declined
                    // capture. Surface that reason on the node's span.
                    let node_id = self.plan[pi].node_id.0;
                    let reason = schedule
                        .boundaries
                        .iter()
                        .find(|decline| decline.node_id == Some(node_id))
                        .map(|decline| decline.reason.as_str())
                        .unwrap_or("non-capturable seam node (no recorded reason)");
                    self.exec_plan_node(
                        pi,
                        resolved,
                        outer_scope,
                        external,
                        OpCaptureTrace::Rejected(reason),
                    )?;
                    // A control-flow seam (e.g. LongRoPE's `If`) that now selects
                    // a different-shaped branch than capture assumed reallocated
                    // an output a later captured segment reads: retire the graph
                    // and finish this step eagerly.
                    if mode == RunMode::Replay && self.control_flow_seam_invalidated(pi, resolved) {
                        invalidated = true;
                    }
                }
            }
        }
        Ok(!invalidated)
    }

    /// Refill [`Self::scratch_input_shapes`] with the resolved shapes of plan
    /// node `pi`'s inputs, so the dispatch path reads shapes from a reused buffer
    /// instead of allocating a fresh `Vec<Vec<usize>>` per node per token.
    ///
    /// The scratch is truncated to the node's arity and each inner `Vec` is
    /// cleared and refilled in place (retaining its heap capacity), so a
    /// steady-state decode step — a fixed sequence of fixed-arity nodes — does
    /// zero shape-vector allocation after warmup. An omitted optional input
    /// (`None` slot) yields an empty inner shape, exactly as the previous
    /// `.unwrap_or_default()` collect did. `self.plan` and
    /// `self.scratch_input_shapes` are disjoint fields, so the shared read of the
    /// former coexists with the `&mut` refill of the latter.
    pub(super) fn refill_input_shapes(
        &mut self,
        pi: usize,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) {
        let inputs = &self.plan[pi].inputs;
        let scratch = &mut self.scratch_input_shapes;
        scratch.truncate(inputs.len());
        for (i, slot) in inputs.iter().enumerate() {
            if i < scratch.len() {
                scratch[i].clear();
            } else {
                scratch.push(Vec::new());
            }
            if let Some(vid) = slot {
                scratch[i].extend_from_slice(&resolved[vid]);
            }
        }
    }

    /// Execute one ordinary (leaf-kernel) plan node: resolve any data-dependent
    /// output shapes, size buffers, build the input/output views (with Holden's
    /// bounds gate), resolve the shape-keyed kernel, and dispatch it.
    pub(super) fn exec_kernel_node(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
        capture: OpCaptureTrace<'_>,
    ) -> Result<()> {
        // Whole-node dispatch span: its lifetime minus `exec_kernel.compute` is
        // the serial per-node dispatch glue (shape resolve, input/output view
        // build, kernel-cache lookup) the F5 Stage 3 record would elide.
        let _node_span = phase_span!("exec_kernel.node");
        // Borrow the plan facts in place rather than cloning them per node per
        // token: `self.plan` is a distinct field from the buffer/view/cache
        // fields mutated below, so these shared borrows coexist with the
        // disjoint `&mut self.<field>` borrows the compute path takes (the
        // dispatch never goes through a `&mut self` method while they are held).
        let node_id = self.plan[pi].node_id;
        // Refill the reusable per-executor input-shape scratch first (before the
        // shared borrows below), so a steady-state decode step allocates no
        // fresh `Vec<Vec<usize>>` for shape lookup -- see `refill_input_shapes`.
        self.refill_input_shapes(pi, resolved);
        let inputs = &self.plan[pi].inputs;
        let outputs = &self.plan[pi].outputs;
        let inplace_dead_inputs = &self.plan[pi].inplace_dead_inputs;
        let input_dtypes = &self.plan[pi].input_dtypes;
        let output_dtypes = &self.plan[pi].output_dtypes;
        let input_shapes = &self.scratch_input_shapes;

        let node = self.graph.node(node_id);

        // Resolve every output's concrete shape: static elementwise broadcast,
        // then data-dependent just-in-time sizing, then the Attention present-KV
        // capacity widening. Mutates `resolved`; returns the final shapes.
        let output_shapes = self.resolve_node_outputs(
            node_id,
            inputs,
            outputs,
            input_dtypes,
            input_shapes,
            resolved,
            external,
        )?;

        let capabilities = self.ep.capabilities();
        let accepts_lazy_weights = LazyWeightBoundary::matches_any(&node.domain, &node.op_type);

        // Resolve each input's real geometry (root buffer + strides/offset) and
        // bounds-check it while only shared borrows of `self` are live. Lazy
        // weights the EP can page are uploaded here and bound as normal device
        // views; ones it declines stay absent and are flagged `lazy_unresolved`.
        let in_infos = self.build_input_bindings(
            pi,
            inputs,
            input_dtypes,
            input_shapes,
            external,
            accepts_lazy_weights,
            &capabilities,
        )?;

        // Only inputs the EP could NOT page remain "lazy" for kernel dispatch;
        // paged weights now have concrete views and take the normal compute path.
        let has_lazy_inputs = in_infos.iter().any(|info| info.lazy_unresolved);

        let ep = self.ep.clone();
        let instance_id = self.instance_id;

        // Bind the mutated fields as disjoint borrows so `self` is never borrowed
        // whole while the kernel (from `cache`) and the buffers/views are held.
        // `cache` and `kernel_bindings` are kept as separate locals because the
        // resolved kernel reference borrows `cache` for the rest of the dispatch.
        let cache = &mut self.cache;
        let kernel_bindings = &mut self.kernel_bindings;
        let provider_artifact_readiness = &mut self.provider_artifact_readiness;
        let finalized_expert_banks = &self.finalized_expert_banks;
        let capture_growing = &self.capture_growing_symbols;
        let mut ctx = KernelDispatchContext {
            executor: self.instance_id,
            ep: &ep,
            graph: &self.graph,
            weight_handles: &self.weight_handles,
            expert_region_candidates: &self.expert_region_candidates,
            buffers: &mut self.buffers,
            buffer_shapes: &mut self.buffer_shapes,
            shared_buffers: &mut self.shared_buffers,
            views_meta: &mut self.views,
            pinned: &mut self.pinned,
            capture_deferred_frees: &mut self.capture_deferred_frees,
            capturing: matches!(capture, OpCaptureTrace::Captured),
            persistent_workspace: &mut self.persistent_workspace,
            step_workspace: &mut self.step_workspace,
            inherited_workspace: self
                .inherited_workspace
                .map(|(ptr, bytes)| WorkspaceView::new(DevicePtrMut(ptr as *mut _), bytes)),
            workspace_preparation_required: self.workspace_preparation_required,
            eager_workspace_growth: matches!(capture, OpCaptureTrace::Eager),
        };

        // Build the (possibly strided) input views once; they feed both the
        // view-output probe and, on the compute path, the kernel itself.
        let mut views: Vec<TensorView> = Vec::with_capacity(in_infos.len());
        for info in &in_infos {
            if !info.present {
                views.push(TensorView::absent(info.dtype));
                continue;
            }
            views.push(
                TensorView::new(
                    DevicePtr(info.base_ptr),
                    info.dtype,
                    &info.shape,
                    &info.strides,
                    info.device,
                )
                .with_byte_offset(info.byte_offset)
                .with_backing(info.backing),
            );
        }

        let opset = effective_opset(ctx.graph, node);
        let kernel = {
            let _s = phase_span!("exec_kernel.get_kernel");
            // Pre-bound fast path: if the binding exists and shapes match, skip
            // the HashMap-key allocation entirely. This is the hot path during
            // steady-state decode — shapes are fixed, so every call after warmup
            // hits here. The binding is populated on miss (below) or at build.
            if let Some(binding) = &kernel_bindings[pi] {
                if let Some(k) = cache.get_prebound(binding, input_shapes) {
                    k
                } else {
                    // Shape changed (prefill→decode or batch change). Fall through
                    // to get_or_create which allocates the key and compiles/fetches.
                    let constant_inputs: Vec<bool> = inputs
                        .iter()
                        .map(|input| {
                            input.is_some_and(|vid| {
                                ctx.graph.initializers.contains_key(&vid)
                                    || ctx.views_meta.get(&vid).is_some_and(|view| {
                                        ctx.graph.initializers.contains_key(&view.source)
                                    })
                            })
                        })
                        .collect();
                    let (k, key) = cache.get_or_create(
                        node_id,
                        node,
                        input_shapes,
                        input_dtypes,
                        &constant_inputs,
                        opset,
                        node_capture_seq_independent(ctx.graph, node, capture_growing),
                        instance_id,
                        provider_artifact_readiness,
                        ep.as_ref(),
                    )?;
                    kernel_bindings[pi] = Some(key);
                    k
                }
            } else {
                // No binding yet (first dispatch of this node for a symbolic
                // graph, or a control-flow/sequence node that was skipped at
                // build). Compute constant_inputs and go through get_or_create.
                let constant_inputs: Vec<bool> = inputs
                    .iter()
                    .map(|input| {
                        input.is_some_and(|vid| {
                            ctx.graph.initializers.contains_key(&vid)
                                || ctx.views_meta.get(&vid).is_some_and(|view| {
                                    ctx.graph.initializers.contains_key(&view.source)
                                })
                        })
                    })
                    .collect();
                let (k, key) = cache.get_or_create(
                    node_id,
                    node,
                    input_shapes,
                    input_dtypes,
                    &constant_inputs,
                    opset,
                    node_capture_seq_independent(ctx.graph, node, capture_growing),
                    instance_id,
                    provider_artifact_readiness,
                    ep.as_ref(),
                )?;
                kernel_bindings[pi] = Some(key);
                k
            }
        };
        // Preflight cannot compile a node whose inputs become concrete only
        // during runtime shape resolution. If lookup above created that
        // specialization, the cache publication chokepoint invalidated the
        // authority. Finalize it now, while no kernel work has been enqueued.
        provider_artifact_readiness.finalize_if_needed(
            ep.as_ref(),
            instance_id,
            ctx.graph,
            finalized_expert_banks,
        )?;
        // --- Zero-copy view fast path ---------------------------------------
        // Ask the kernel whether its outputs are strided views over its inputs
        // (a layout/movement op such as Slice). If so, record view metadata
        // aliasing the source buffer and skip compute + allocation entirely.
        if !has_lazy_inputs
            && let Some(specs) = kernel.view_outputs(&views, &output_shapes, outputs.len())
        {
            if outputs
                .iter()
                .any(|output| external.outputs.contains_key(output))
            {
                return Err(SessionError::Internal(format!(
                    "op '{}' cannot bind a zero-copy view output to external storage",
                    node.op_type
                )));
            }
            drop(views);
            ctx.install_view_outputs(node, inputs, outputs, output_dtypes, resolved, specs)?;
            return Ok(());
        }

        // --- Compute path ----------------------------------------------------
        // Size (allocate or reuse) each output's contiguous buffer, JIT-sizing
        // data-dependent ones.
        let inplace_input =
            if self.compute_in_place_enabled && outputs.len() == 1 && !has_lazy_inputs {
                inplace_dead_inputs
                    .iter()
                    .enumerate()
                    .find_map(|(index, dead)| {
                        let vid = inputs[index]?;
                        let info = &in_infos[index];
                        (*dead
                            && kernel.can_run_in_place(index)
                            && !external.inputs.contains_key(&vid)
                            && !external.outputs.contains_key(&vid)
                            && !ctx.graph.outputs.contains(&vid)
                            && !ctx.views_meta.contains_key(&vid)
                            && !ctx.pinned.contains(&vid)
                            && !ctx.shared_buffers.contains_key(&vid)
                            // A borrowed buffer aliases the caller's own input
                            // tensor for this run (`prepare_run_buffers`). Running
                            // in place would write through it, mutating memory
                            // this session does not own.
                            && ctx.buffers.get(&vid).is_some_and(|b| !b.is_borrowed())
                            && !self.seq_elem_values.contains_key(&vid)
                            && info.dtype == output_dtypes[0]
                            && info.shape == output_shapes[0]
                            && onnx_runtime_ir::is_contiguous(&info.shape, &info.strides)
                            && info.byte_offset == 0)
                            .then_some(vid)
                    })
            } else {
                None
            };
        if let Some(input) = inplace_input {
            ctx.alias_input_as_output(input, outputs[0], &output_shapes[0])?;
            self.compute_in_place_alias_count += 1;
        } else {
            ctx.ensure_output_backings(outputs, &output_shapes, output_dtypes, external)?;
        }

        // Auto-materialization gate: strided (view) inputs feeding a kernel that
        // does not accept them on that slot are gathered into private contiguous
        // temps. Temps must outlive the views that borrow them.
        let mat = materialize_strided_inputs(kernel, &in_infos, node)?;

        // Rebuild input views, swapping any materialized slot to its contiguous
        // temp (offset 0, contiguous strides over the fresh buffer).
        drop(views);
        let mut views: Vec<TensorView> = Vec::with_capacity(in_infos.len());
        for (i, info) in in_infos.iter().enumerate() {
            if !info.present {
                views.push(TensorView::absent(info.dtype));
                continue;
            }
            match &mat[i] {
                Some((buf, strides)) => views.push(TensorView::new(
                    DevicePtr(buf.as_ptr() as *const std::ffi::c_void),
                    info.dtype,
                    &info.shape,
                    strides,
                    onnx_runtime_ir::DeviceId::cpu(),
                )),
                None => views.push(
                    TensorView::new(
                        DevicePtr(info.base_ptr),
                        info.dtype,
                        &info.shape,
                        &info.strides,
                        info.device,
                    )
                    .with_byte_offset(info.byte_offset)
                    .with_backing(info.backing),
                ),
            }
        }

        // Take output buffers out so they can be borrowed `&mut` disjointly from
        // the input reads (SSA guarantees outputs are disjoint from inputs).
        let out_strides: Vec<Vec<i64>> = output_shapes
            .iter()
            .map(|s| compute_contiguous_strides(s))
            .collect();
        let (out_bufs, mut outs) = ctx.build_output_bindings(
            outputs,
            &output_shapes,
            &out_strides,
            output_dtypes,
            external,
        )?;

        ctx.execute_kernel(
            kernel,
            inputs,
            outputs,
            node,
            &capabilities,
            has_lazy_inputs,
            &views,
            &mut outs,
            out_bufs,
        )?;
        Ok(())
    }

    /// Resolve every output value's concrete shape for this node.
    ///
    /// Runs, in order: the static elementwise broadcast rule, then (only if any
    /// output is still unresolved) data-dependent just-in-time sizing from the
    /// runtime integer/float values of shape inputs, then the default-domain
    /// Attention present-KV physical-capacity widening. Inserts each resolved
    /// shape into `resolved` and returns the per-output shapes in output order.
    #[allow(clippy::too_many_arguments)]
    fn resolve_node_outputs(
        &self,
        node_id: NodeId,
        inputs: &[Option<ValueId>],
        outputs: &[ValueId],
        input_dtypes: &[DataType],
        input_shapes: &[Vec<usize>],
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<Vec<Vec<usize>>> {
        let node = self.graph.node(node_id);
        if let Some(output_shape) = runtime_elementwise_output_shape(node, input_shapes) {
            let output_shape = output_shape.map_err(|_| {
                let node_name = if node.name.is_empty() {
                    format!("<unnamed node #{}>", node_id.0)
                } else {
                    format!("{:?}", node.name)
                };
                SessionError::RuntimeBroadcastIncompatible {
                    node: node_name,
                    domain: canonical_domain(node),
                    op_type: node.op_type.clone(),
                    input_shapes: input_shapes.to_vec(),
                }
            })?;
            if outputs.len() != 1 {
                return Err(SessionError::OutputShapeCountMismatch {
                    op: node.op_type.clone(),
                    expected: outputs.len(),
                    got: 1,
                });
            }
            resolved.insert(outputs[0], output_shape);
        }

        // Data-dependent shapes: if any output's shape is still unresolved,
        // compute it now from the concrete input shapes + the runtime *values*
        // of this node's integer inputs. Buffers are NOT sized here — a view
        // output needs none, and the compute path sizes them just below.
        if outputs.iter().any(|v| !resolved.contains_key(v)) {
            let opset = effective_opset(&self.graph, node);
            // GroupQueryAttention's present-KV output shape is the physical cache
            // capacity (see `kernel_input_uses_physical_capacity`): the native EP
            // binds present aliased in-place to the fixed-capacity past-KV, so the
            // present sequence extent equals `past_key[2]` — known on host from the
            // input *shape*. Reading its integer shape inputs (`seqlens_k`,
            // `total_sequence_length`) back to host per layer is therefore
            // unnecessary and, in eager decode, the dominant blocking-D2H launch
            // cost. Skip materializing them; `dynamic_output_shapes` sizes present
            // from the past-KV capacity when the total value is absent.
            let is_gqa = node.domain == "com.microsoft" && node.op_type == "GroupQueryAttention";
            // Only skip the shape-input read-backs when the present-KV outputs are
            // externally (persistently) bound: their bound buffer shape is the
            // authoritative fixed physical capacity, aliased in-place to past-KV,
            // so `present_sequence == past_key[2]` (host-known) and reading
            // `seqlens_k`/`total_sequence_length` back to host is redundant. A
            // GQA node whose present outputs are NOT capacity-bound (e.g. a
            // growing `past ⧺ current` cache) still reads `total_sequence_length`.
            let gqa_present_capacity_bound = is_gqa
                && gqa_shape_capacity_bound_enabled()
                && outputs
                    .get(1)
                    .is_some_and(|vid| external.outputs.contains_key(vid));
            let input_values: Vec<Option<Vec<i64>>> = inputs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if gqa_present_capacity_bound {
                        return None;
                    }
                    v.and_then(|vid| {
                        if node.is_default_domain() && node.op_type == "Compress" && i == 1 {
                            self.compress_condition_i64(vid, &input_shapes[i], input_dtypes[i])
                        } else {
                            self.shape_input_i64(vid, &input_shapes[i], input_dtypes[i])
                        }
                    })
                })
                .collect();
            // Only materialize a *float* input value for the specific inputs an
            // op actually reads as float shape data (today: `Resize` scales).
            // Downloading any other float input here would both waste a host copy
            // and break the "reject an invalid shape input before any host
            // materialization" contract — e.g. a data tensor feeding an
            // `Unsqueeze` whose integer axes is invalid must never be copied to
            // host just to reach the unresolved-shape rejection.
            let input_float_values: Vec<Option<Vec<f64>>> = inputs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if !reads_float_shape_input(node, i, opset) {
                        return None;
                    }
                    v.and_then(|vid| {
                        if node.is_default_domain() && node.op_type == "NonMaxSuppression" {
                            self.nms_input_f64(vid, &input_shapes[i], input_dtypes[i])
                        } else {
                            self.shape_input_f64(vid, &input_shapes[i], input_dtypes[i])
                        }
                    })
                })
                .collect();
            let out_shapes = dynamic_output_shapes(
                node,
                input_shapes,
                input_dtypes,
                &input_values,
                &input_float_values,
                opset,
            )
            .ok_or_else(|| {
                let vid = outputs
                    .iter()
                    .find(|v| !resolved.contains_key(v))
                    .copied()
                    .unwrap_or(outputs[0]);
                let value = self.graph.value(vid);
                SessionError::UnresolvedShape {
                    value: value
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("value#{}", vid.0)),
                    op: node.op_type.clone(),
                }
            })?;
            if out_shapes.len() != outputs.len() {
                return Err(SessionError::OutputShapeCountMismatch {
                    op: self.graph.node(node_id).op_type.clone(),
                    expected: outputs.len(),
                    got: out_shapes.len(),
                });
            }
            for (oi, &ovid) in outputs.iter().enumerate() {
                resolved.insert(ovid, out_shapes[oi].clone());
            }
        }
        let mut output_shapes: Vec<Vec<usize>> =
            outputs.iter().map(|v| resolved[v].clone()).collect();
        // Fixed-capacity KV for the default-domain Attention op, and for a
        // decomposed attention's plain-`Concat` KV-cache append (see
        // `geometry::is_kv_cache_growth_concat`). Both present outputs are
        // consumer-less graph outputs bound to a growing device cache; the two
        // branches below correct for that binding, but via different targets
        // (`output_shapes` for `Attention`, `resolved` for the `Concat` case —
        // see the `Concat` branch's comment for why) since only `Attention`'s
        // own kernel needs the widened shape to place the new token correctly.
        // Either way, the valid attended length is still derived from the
        // logical past+current extent, so this only widens the *storage*
        // stride/tracked shape and never changes what is attended over. Only
        // present values bound sub-shape (logical != physical) capacity buffers
        // are corrected; a dense/unbound present keeps its inferred shape.
        {
            let node = self.graph.node(node_id);
            if node.is_default_domain() && node.op_type == "Attention" {
                // When the past K/V inputs are themselves bound at physical
                // capacity (fixed-capacity decode, capture path), the standard
                // `present = past + current` shape rule sees the *physical* past
                // extent and over-counts the present seq axis beyond the bound
                // buffer. In that case the present buffer's true shape is simply
                // its physical capacity (mirroring GroupQueryAttention, whose
                // present rule takes `past_capacity.max(total)`); the valid
                // length lives on-device and context-overflow is caught earlier
                // in the decoder (`total_len > max_len`). Otherwise keep the
                // conservative `physical >= logical` guard.
                let kv_capacity_bound = kernel_input_uses_physical_capacity(node, 4)
                    && kernel_input_uses_physical_capacity(node, 5);
                for (oi, &ovid) in outputs.iter().enumerate() {
                    if oi == 0 {
                        continue;
                    }
                    if let Some(value) = external.outputs.get(&ovid)
                        && value.accepts_subshape
                        && value.shape.len() == output_shapes[oi].len()
                        && value
                            .shape
                            .iter()
                            .zip(&output_shapes[oi])
                            .enumerate()
                            .all(|(axis, (&physical, &logical))| axis == 2 || physical == logical)
                        && (kv_capacity_bound
                            || value
                                .shape
                                .get(2)
                                .zip(output_shapes[oi].get(2))
                                .is_some_and(|(&physical, &logical)| physical >= logical))
                    {
                        output_shapes[oi] = value.shape.clone();
                    }
                }
            } else if is_kv_cache_growth_concat(&self.graph, node)
                && let Some(axis) = node.attr("axis").and_then(Attribute::as_int)
            {
                // Decomposed attention grows its KV cache with a plain `Concat`
                // (`present.* = Concat(past.*, current, axis)`) instead of an
                // in-op cache. `is_kv_cache_growth_concat` recognizes this by
                // input/output role alone (see its doc comment) — the same
                // predicate `geometry::classify_mask_consumer` uses to decide
                // whether a mask combined with a value derived from this append
                // can also be frozen to physical width; the two are two sides
                // of one invariant.
                //
                // Unlike `Attention`, whose kernel is *designed* to receive a
                // physical-capacity output shape (it derives the true attended
                // length from the mask/cache extent instead, so widening tells
                // it where to place the new token without repacking), a plain
                // `Concat` kernel independently validates that its declared
                // output shape equals `past.shape[axis] + current.shape[axis]`
                // and correctly writes only that (small) delta — relying on the
                // present==past device aliasing for the append to land in
                // place. Widening `output_shapes` here would make the node's
                // *own* dispatch shape lie about that arithmetic and the kernel
                // would (rightly) reject it. So only `resolved` is corrected —
                // it is what a same-step downstream consumer (a decomposed
                // attention's `Shape(present.*)` / `Unsqueeze` chain) reads via
                // `refill_input_shapes`, and it is what was stale before this
                // fix (the original crash). This node's own `output_shapes`
                // stays the naive, arithmetically-correct value below.
                let rank = output_shapes[0].len();
                let axis = if axis < 0 { axis + rank as i64 } else { axis };
                if let Ok(axis) = usize::try_from(axis)
                    && axis < rank
                {
                    let ovid = outputs[0];
                    if let Some(value) = external.outputs.get(&ovid)
                        && value.accepts_subshape
                        && value.shape.len() == output_shapes[0].len()
                        && value
                            .shape
                            .iter()
                            .zip(&output_shapes[0])
                            .enumerate()
                            .all(|(a, (&physical, &logical))| a == axis || physical == logical)
                        && value
                            .shape
                            .get(axis)
                            .zip(output_shapes[0].get(axis))
                            .is_some_and(|(&physical, &logical)| physical >= logical)
                    {
                        resolved.insert(ovid, value.shape.clone());
                    }
                }
            }
        }
        Ok(output_shapes)
    }

    /// Resolve each input's real geometry (root buffer + strides/offset) and
    /// bounds-check it, producing one [`InInfo`] per positional input slot.
    /// View inputs read through their recorded strides; omitted optional inputs
    /// become absent placeholders. Runs while only shared borrows of `self` are
    /// live, so the returned owned vector can outlive the disjoint `&mut`
    /// borrows the compute path takes afterwards.
    #[allow(clippy::too_many_arguments)]
    fn build_input_bindings(
        &self,
        pi: usize,
        inputs: &[Option<ValueId>],
        input_dtypes: &[DataType],
        input_shapes: &[Vec<usize>],
        external: &ExternalBindings,
        accepts_lazy_weights: bool,
        capabilities: &ExecutionProviderCapabilities,
    ) -> Result<Vec<InInfo>> {
        let mut in_infos: Vec<InInfo> = Vec::with_capacity(inputs.len());
        let _build_inputs_span = phase_span!("exec_kernel.build_inputs");
        for (i, slot) in inputs.iter().enumerate() {
            let Some(vid) = *slot else {
                in_infos.push(InInfo {
                    present: false,
                    dtype: input_dtypes[i],
                    shape: Vec::new(),
                    strides: Vec::new(),
                    byte_offset: 0,
                    base_ptr: std::ptr::null(),
                    device: self.ep.device_id(),
                    backing: TensorBacking::Opaque,
                    root_len: 0,
                    lazy_unresolved: false,
                    paged: None,
                });
                continue;
            };
            if let Some(value) = external
                .inputs
                .get(&vid)
                .or_else(|| external.outputs.get(&vid))
            {
                let strides = compute_contiguous_strides(&value.shape);
                view_bounds(&value.shape, &strides, 0, value.dtype, value.len)?;
                in_infos.push(InInfo {
                    present: true,
                    dtype: value.dtype,
                    shape: value.shape.clone(),
                    strides,
                    byte_offset: 0,
                    base_ptr: value.ptr.cast_const(),
                    device: value.device,
                    backing: TensorBacking::Opaque,
                    root_len: value.len,
                    lazy_unresolved: false,
                    paged: None,
                });
                continue;
            }
            // A tensor input backed by a shared sequence element (SequenceAt
            // output) owns no DeviceBuffer: read its possibly-strided view
            // directly over the immutable shared allocation.
            if let Some(elem) = self.seq_elem_values.get(&vid) {
                let shape = input_shapes[i].clone();
                let strides = elem.layout.resolved_strides(&shape);
                let root_len = elem.root_len();
                let base_ptr = elem.as_ptr();
                view_bounds(
                    &shape,
                    &strides,
                    elem.byte_offset(),
                    input_dtypes[i],
                    root_len,
                )?;
                in_infos.push(InInfo {
                    present: true,
                    dtype: input_dtypes[i],
                    shape,
                    strides,
                    byte_offset: elem.byte_offset(),
                    base_ptr,
                    device: elem.device(),
                    backing: TensorBacking::Opaque,
                    root_len,
                    lazy_unresolved: false,
                    paged: None,
                });
                continue;
            }
            if accepts_lazy_weights
                && let Some(lazy) = self
                    .weight_handles
                    .get(&vid)
                    .filter(|handle| handle.is_lazy_for(capabilities))
                    .and_then(|handle| handle.as_lazy())
            {
                // Ask the EP to page this lazy weight into device memory (or reuse
                // a resident page). On success we bind a normal device view over
                // the paged bytes and keep the page pinned for the kernel's
                // lifetime via `paged`. On `None` (EP can't page) the input stays
                // absent and is routed to the kernel as a lazy `KernelInput::Weight`.
                let issued_at = self.prefetch_issue_nodes.lock().unwrap().remove(&vid);
                let paged = self.ep.page_lazy_weight_for_executor(
                    self.instance_id,
                    vid.0 as u64,
                    lazy,
                    &WeightStoreRegionSource(self.weights.as_ref()),
                )?;
                match paged {
                    Some(paged) => {
                        if let Some(issued_at) = issued_at {
                            let nodes_between = pi.saturating_sub(issued_at);
                            record_dense_prefetch_gap(nodes_between as u64);
                        }
                        let shape = input_shapes[i].clone();
                        let strides = compute_contiguous_strides(&shape);
                        in_infos.push(InInfo {
                            present: true,
                            dtype: input_dtypes[i],
                            shape,
                            strides,
                            byte_offset: 0,
                            base_ptr: paged.device_ptr(),
                            device: paged.device(),
                            backing: TensorBacking::Opaque,
                            root_len: paged.len(),
                            lazy_unresolved: false,
                            paged: Some(paged),
                        });
                    }
                    None => {
                        in_infos.push(InInfo {
                            present: false,
                            dtype: input_dtypes[i],
                            shape: input_shapes[i].clone(),
                            strides: compute_contiguous_strides(&input_shapes[i]),
                            byte_offset: 0,
                            base_ptr: std::ptr::null(),
                            device: self.ep.device_id(),
                            backing: TensorBacking::Opaque,
                            root_len: 0,
                            lazy_unresolved: true,
                            paged: None,
                        });
                    }
                }
                continue;
            }
            let root = self.root_of(vid);
            let buf = self.buffers.get(&root).ok_or_else(|| {
                SessionError::Internal(format!("missing buffer for input value#{}", vid.0))
            })?;
            let root_len = buf.len();
            let base_ptr = buf.as_ptr();
            let (shape, strides, byte_offset) = match self.views.get(&vid) {
                Some(view) => (view.shape.clone(), view.strides.clone(), view.byte_offset),
                None => {
                    let shape = input_shapes[i].clone();
                    let strides = compute_contiguous_strides(&shape);
                    (shape, strides, 0)
                }
            };
            view_bounds(&shape, &strides, byte_offset, input_dtypes[i], root_len)?;
            let backing = self
                .graph
                .initializers
                .get(&root)
                .filter(|_| buf.is_borrowed())
                .and_then(|weight| self.weights.external_mmap_provenance(weight))
                .map(|(mapping_id, offset, len)| {
                    TensorBacking::ExternalMmap(ExternalMmapRegion {
                        mapping_id,
                        offset,
                        len,
                    })
                })
                .unwrap_or(TensorBacking::Opaque);
            in_infos.push(InInfo {
                present: true,
                dtype: input_dtypes[i],
                shape,
                strides,
                byte_offset,
                base_ptr,
                device: buf.device(),
                backing,
                root_len,
                lazy_unresolved: false,
                paged: None,
            });
        }
        drop(_build_inputs_span);
        Ok(in_infos)
    }

    /// Read the integer *values* of input `vid` as `i64`, materializing a view
    /// first if needed. Used to resolve data-dependent output shapes (e.g. a
    /// `Slice` whose `ends` is produced at runtime). Returns `None` if the value
    /// has no readable buffer/view or its dtype is not an integer.
    pub(super) fn input_i64(
        &self,
        vid: ValueId,
        shape: &[usize],
        dtype: DataType,
    ) -> Option<Vec<i64>> {
        let bytes = self.contiguous_bytes(vid, shape, dtype).ok()?;
        bytes_as_i64(&bytes, dtype)
    }

    /// Bounded integer reader for dynamic shape propagation. Views and sequence
    /// elements can have a tiny logical shape backed by a much larger root
    /// allocation, so cap that allocation before `contiguous_bytes` can copy it.
    pub(super) fn shape_input_i64(
        &self,
        vid: ValueId,
        shape: &[usize],
        dtype: DataType,
    ) -> Option<Vec<i64>> {
        if !bounded_shape_input(dtype, shape) {
            return None;
        }
        let max_bytes = MAX_SHAPE_DATA_ELEMS.checked_mul(dtype.byte_size())?;
        if let Some(view) = self.views.get(&vid) {
            let source = self.buffers.get(&view.source)?;
            if source.len() > max_bytes {
                return None;
            }
        }
        if self
            .seq_elem_values
            .get(&vid)
            .is_some_and(|elem| elem.root_len() > max_bytes)
        {
            return None;
        }
        self.input_i64(vid, shape, dtype)
    }

    fn compress_condition_i64(
        &self,
        vid: ValueId,
        shape: &[usize],
        dtype: DataType,
    ) -> Option<Vec<i64>> {
        if !bounded_compress_condition(dtype, shape) {
            return None;
        }
        let max_bytes = shape[0].checked_mul(dtype.byte_size())?;
        if let Some(view) = self.views.get(&vid) {
            let source = self.buffers.get(&view.source)?;
            if source.len() > max_bytes {
                return None;
            }
        }
        if self
            .seq_elem_values
            .get(&vid)
            .is_some_and(|elem| elem.root_len() > max_bytes)
        {
            return None;
        }
        self.input_i64(vid, shape, dtype)
    }

    pub(super) fn shape_input_f64(
        &self,
        vid: ValueId,
        shape: &[usize],
        dtype: DataType,
    ) -> Option<Vec<f64>> {
        if !matches!(dtype, DataType::Float32 | DataType::Float64)
            || shape.len() > 1
            || shape
                .iter()
                .try_fold(1usize, |count, &dim| count.checked_mul(dim))
                .is_none_or(|count| count > MAX_SHAPE_DATA_ELEMS)
        {
            return None;
        }
        let max_bytes = MAX_SHAPE_DATA_ELEMS.checked_mul(dtype.byte_size())?;
        if let Some(view) = self.views.get(&vid) {
            let source = self.buffers.get(&view.source)?;
            if source.len() > max_bytes {
                return None;
            }
        }
        if self
            .seq_elem_values
            .get(&vid)
            .is_some_and(|elem| elem.root_len() > max_bytes)
        {
            return None;
        }
        let bytes = self.contiguous_bytes(vid, shape, dtype).ok()?;
        bytes_as_f64(&bytes, dtype)
    }

    /// `NonMaxSuppression` needs its boxes and scores to determine the exact
    /// data-dependent output extent. Unlike ordinary shape tensors these inputs
    /// are rank 3 and may exceed `MAX_SHAPE_DATA_ELEMS`; materialize them only
    /// for this operator, immediately before its output allocation.
    pub(super) fn nms_input_f64(
        &self,
        vid: ValueId,
        shape: &[usize],
        dtype: DataType,
    ) -> Option<Vec<f64>> {
        if dtype != DataType::Float32 {
            return None;
        }
        let bytes = self.contiguous_bytes(vid, shape, dtype).ok()?;
        bytes_as_f64(&bytes, dtype)
    }
}

/// Owned backing for one kernel output: either an internal [`DeviceBuffer`]
/// taken out of the executor's buffer map (to be reinserted after compute) or
/// an external binding written in place. Holds the raw pointer and length used
/// to build the mutable [`TensorMut`] the kernel writes through.
struct OutBacking {
    vid: ValueId,
    internal: Option<DeviceBuffer>,
    ptr: *mut std::ffi::c_void,
    len: usize,
    device: onnx_runtime_ir::DeviceId,
}

/// Auto-materialization gate: a strided (view) input feeding a kernel that does
/// not accept strided input on that slot is gathered into a private contiguous
/// temp so contiguous-assuming kernels stay correct. Returns, per input slot,
/// the gathered bytes and their contiguous strides when a temp was needed.
/// Temps must outlive the views that borrow them.
fn materialize_strided_inputs(
    kernel: &dyn Kernel,
    in_infos: &[InInfo],
    node: &Node,
) -> Result<MaterializedInputs> {
    let mut mat: Vec<Option<(Vec<u8>, Vec<i64>)>> = Vec::with_capacity(in_infos.len());
    for (i, info) in in_infos.iter().enumerate() {
        if !info.present {
            mat.push(None);
            continue;
        }
        let contiguous = onnx_runtime_ir::is_contiguous(&info.shape, &info.strides);
        if contiguous || kernel.supports_strided_input(i) {
            mat.push(None);
            continue;
        }
        if !info.device.is_host_accessible() {
            return Err(SessionError::Internal(format!(
                "op '{}' requires host-only strided materialization for CUDA input {i}",
                node.op_type
            )));
        }
        let esize = info.dtype.byte_size();
        if esize == 0 {
            return Err(SessionError::from(
                onnx_runtime_ep_api::EpError::InvalidTensorView {
                    reason: format!(
                        "cannot materialize sub-byte strided input {i} of op '{}'",
                        node.op_type
                    ),
                },
            ));
        }
        let src = unsafe { std::slice::from_raw_parts(info.base_ptr as *const u8, info.root_len) };
        let gathered = gather_view(src, &info.shape, &info.strides, info.byte_offset, esize);
        let strides = compute_contiguous_strides(&info.shape);
        mat.push(Some((gathered, strides)));
    }
    Ok(mat)
}

/// Disjoint borrows of the executor fields the per-node compute path mutates,
/// carried as a struct-of-borrows so the dispatch helpers can be called in
/// order without ever borrowing `self` whole (which would collide with the
/// resolved kernel's borrow of `self.cache`). No field is cloned.
struct KernelDispatchContext<'a> {
    executor: ExecutorInstanceId,
    ep: &'a Arc<dyn ExecutionProvider>,
    graph: &'a Graph,
    weight_handles: &'a HashMap<ValueId, WeightHandle>,
    expert_region_candidates: &'a HashMap<ValueId, onnx_runtime_loader::WeightRegionCatalog>,
    buffers: &'a mut HashMap<ValueId, DeviceBuffer>,
    buffer_shapes: &'a mut HashMap<ValueId, Vec<usize>>,
    shared_buffers: &'a mut HashMap<ValueId, Arc<SharedTensorBuffer>>,
    views_meta: &'a mut HashMap<ValueId, ValueView>,
    pinned: &'a mut HashSet<ValueId>,
    /// Stale view-owner buffers parked here instead of freed while a captured
    /// segment is being recorded (freeing synchronizes, which stream capture
    /// forbids). Flushed by the caller once capture closes. See
    /// [`super::state`]`::capture_deferred_frees`.
    capture_deferred_frees: &'a mut Vec<DeviceBuffer>,
    /// True while this node is recorded into a device graph
    /// ([`OpCaptureTrace::Captured`]): buffer frees must be deferred, not issued.
    capturing: bool,
    persistent_workspace: &'a mut Option<PreparedWorkspace>,
    step_workspace: &'a mut Option<PreparedWorkspace>,
    inherited_workspace: Option<WorkspaceView>,
    workspace_preparation_required: bool,
    /// True when this node is dispatched on a plain eager run (no device-graph
    /// capture in progress), so growing a prepared workspace slot in place is
    /// safe: nothing has baked its device pointer into a captured graph. This is
    /// what lets a prepared session re-prepare (grow) its governed workspace when
    /// execution rebuckets to a larger shape bucket than preparation reserved
    /// for, instead of failing the prepared-workspace invariant (#1223). It is
    /// false while recording a captured segment ([`OpCaptureTrace::Captured`]),
    /// where the workspace pointer must stay fixed for replay.
    eager_workspace_growth: bool,
}

impl KernelDispatchContext<'_> {
    /// Transfer a dead, exclusively-owned input allocation to the output value.
    /// The input views were constructed before this move and retain its raw
    /// address; the kernel capability contract is what makes simultaneous read
    /// and write through that identical range safe.
    fn alias_input_as_output(
        &mut self,
        input: ValueId,
        output: ValueId,
        output_shape: &[usize],
    ) -> Result<()> {
        let buffer = self.buffers.remove(&input).ok_or_else(|| {
            SessionError::Internal(format!(
                "missing buffer for in-place input value#{}",
                input.0
            ))
        })?;
        self.buffer_shapes.remove(&input);
        if let Some(old) = self.buffers.remove(&output) {
            self.ep.deallocate(old)?;
        }
        self.shared_buffers.remove(&output);
        self.buffers.insert(output, buffer);
        self.buffer_shapes.insert(output, output_shape.to_vec());
        Ok(())
    }

    /// Install the kernel's zero-copy view outputs: bounds-gate each composed
    /// view against its source allocation, drop any buffer the output used to
    /// own, record the alias, and pin the source for the rest of the run.
    /// Called only on the fast path after the external-output guard.
    #[allow(clippy::too_many_arguments)]
    fn install_view_outputs(
        &mut self,
        node: &Node,
        inputs: &[Option<ValueId>],
        outputs: &[ValueId],
        output_dtypes: &[DataType],
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        specs: Vec<onnx_runtime_ep_api::ViewOutput>,
    ) -> Result<()> {
        if specs.len() != outputs.len() {
            return Err(SessionError::Internal(format!(
                "op '{}' returned {} view outputs for {} outputs",
                node.op_type,
                specs.len(),
                outputs.len()
            )));
        }
        for (oi, spec) in specs.into_iter().enumerate() {
            let ovid = outputs[oi];
            let Some(in_vid) = inputs.get(spec.input_index).copied().flatten() else {
                return Err(SessionError::Internal(format!(
                    "op '{}' view output {} references invalid input index {}",
                    node.op_type, oi, spec.input_index
                )));
            };
            let root = match self.views_meta.get(&in_vid) {
                Some(v) => v.source,
                None => in_vid,
            };
            let root_len = self.buffers.get(&root).map(|b| b.len()).ok_or_else(|| {
                SessionError::Internal(format!("view source value#{} has no buffer", root.0))
            })?;
            // Bounds-gate the composed view against the source allocation.
            view_bounds(
                &spec.shape,
                &spec.strides,
                spec.byte_offset,
                output_dtypes[oi],
                root_len,
            )?;
            // The output becomes a view: drop any buffer it used to own so a
            // later run re-sizes cleanly, then record the alias and pin the
            // source (conservative liveness — a source with any live view is
            // never reused/freed for the rest of the run; no use-after-free).
            // A freshly-produced output can never already be pinned (its
            // viewers run strictly after it under SSA topo order).
            debug_assert!(
                !self.pinned.contains(&ovid),
                "value#{} is pinned as a live view source yet is being reproduced",
                ovid.0
            );
            if let Some(old) = self.buffers.remove(&ovid) {
                // Freeing synchronizes the copy stream (pooled unmap), which is
                // illegal while a device graph is recording. During capture we
                // park the now-orphaned buffer and let the caller free it once
                // capture closes; outside capture we free immediately.
                if self.capturing {
                    self.capture_deferred_frees.push(old);
                } else {
                    self.ep.deallocate(old)?;
                }
            }
            self.shared_buffers.remove(&ovid);
            self.buffer_shapes.remove(&ovid);
            self.views_meta.insert(
                ovid,
                ValueView {
                    source: root,
                    shape: spec.shape.clone(),
                    strides: spec.strides,
                    byte_offset: spec.byte_offset,
                },
            );
            self.pinned.insert(root);
            resolved.insert(ovid, spec.shape);
        }
        Ok(())
    }

    /// Size (allocate or reuse) each output's contiguous buffer, JIT-sizing
    /// data-dependent ones. A value that was a view on a prior run has no buffer
    /// here and is freshly allocated. External outputs are validated in place.
    fn ensure_output_backings(
        &mut self,
        outputs: &[ValueId],
        output_shapes: &[Vec<usize>],
        output_dtypes: &[DataType],
        external: &ExternalBindings,
    ) -> Result<()> {
        for (oi, &ovid) in outputs.iter().enumerate() {
            let dims = &output_shapes[oi];
            let numel = checked_numel(dims, || format!("value#{}", ovid.0))?;
            let need = checked_storage_bytes(
                output_dtypes[oi],
                numel,
                || format!("value#{}", ovid.0),
                dims,
            )?
            .max(1);
            if let Some(value) = external.outputs.get(&ovid) {
                if !value.accepts_output(output_dtypes[oi], dims, need) {
                    let name = self
                        .graph
                        .value(ovid)
                        .name
                        .as_deref()
                        .unwrap_or("<unnamed>");
                    return Err(SessionError::Internal(format!(
                        "external output '{name}' has {:?} {:?} ({} bytes), kernel requires {:?} {:?} ({need} bytes)",
                        value.dtype, value.shape, value.len, output_dtypes[oi], dims
                    )));
                }
                continue;
            }
            let fits = self
                .buffers
                .get(&ovid)
                .map(|b| b.len() == need)
                .unwrap_or(false);
            if !fits {
                // Never free a buffer that has a live view alias (would dangle
                // the viewer). Unreachable under SSA topo order, but enforced.
                debug_assert!(
                    !self.pinned.contains(&ovid),
                    "value#{} is pinned as a live view source yet is being resized",
                    ovid.0
                );
                if let Some(old) = self.buffers.remove(&ovid) {
                    self.ep.deallocate(old)?;
                }
                self.shared_buffers.remove(&ovid);
                let buf = self
                    .ep
                    .allocate(need, TensorLayout::contiguous().alignment)?;
                self.buffers.insert(ovid, buf);
            }
        }
        Ok(())
    }

    /// Take each output buffer out (internal buffers removed from the map,
    /// external bindings aliased in place), bounds-check the contiguous output
    /// window, and build the [`TensorMut`] the kernel writes through. The
    /// returned views borrow `output_shapes`/`out_strides`; the returned
    /// backings must outlive them.
    fn build_output_bindings<'o>(
        &mut self,
        outputs: &[ValueId],
        output_shapes: &'o [Vec<usize>],
        out_strides: &'o [Vec<i64>],
        output_dtypes: &[DataType],
        external: &ExternalBindings,
    ) -> Result<(Vec<OutBacking>, Vec<TensorMut<'o>>)> {
        let mut out_bufs: Vec<OutBacking> = Vec::with_capacity(outputs.len());
        for &vid in outputs {
            if let Some(value) = external.outputs.get(&vid) {
                out_bufs.push(OutBacking {
                    vid,
                    internal: None,
                    ptr: value.ptr,
                    len: value.len,
                    device: value.device,
                });
            } else {
                let mut buf = self.buffers.remove(&vid).ok_or_else(|| {
                    SessionError::Internal(format!("missing buffer for output value#{}", vid.0))
                })?;
                let ptr = buf.as_mut_ptr();
                out_bufs.push(OutBacking {
                    vid,
                    ptr,
                    len: buf.len(),
                    device: buf.device(),
                    internal: Some(buf),
                });
            }
        }
        let mut outs: Vec<TensorMut> = Vec::with_capacity(out_bufs.len());
        for (i, backing) in out_bufs.iter_mut().enumerate() {
            view_bounds(
                &output_shapes[i],
                &out_strides[i],
                0,
                output_dtypes[i],
                backing.len,
            )?;
            outs.push(TensorMut::new(
                DevicePtrMut(backing.ptr),
                output_dtypes[i],
                &output_shapes[i],
                &out_strides[i],
                backing.device,
            ));
        }
        Ok((out_bufs, outs))
    }

    /// Run the kernel over the bound input/output views (routing lazy-weight
    /// inputs through [`KernelInput`] when present), map any failure to a
    /// diagnostic naming the node and its input/output types and shapes, then
    /// reinsert the internal output buffers back into the buffer map.
    #[allow(clippy::too_many_arguments)]
    fn execute_kernel(
        &mut self,
        kernel: &dyn Kernel,
        inputs: &[Option<ValueId>],
        outputs: &[ValueId],
        node: &Node,
        capabilities: &ExecutionProviderCapabilities,
        has_lazy_inputs: bool,
        views: &[TensorView],
        outs: &mut Vec<TensorMut>,
        out_bufs: Vec<OutBacking>,
    ) -> Result<()> {
        let kernel_inputs = has_lazy_inputs.then(|| {
            inputs
                .iter()
                .zip(views.iter().copied())
                .map(|(value, view)| {
                    // A lazy weight the EP paged into device memory already has a
                    // concrete device view, so route it as a normal tensor. Only
                    // an unresolved lazy weight (absent view) is handed to the
                    // kernel as `KernelInput::Weight` for host-side materialization.
                    let lazy_handle = value
                        .and_then(|value| self.weight_handles.get(&value))
                        .filter(|handle| handle.is_lazy_for(capabilities));
                    match lazy_handle {
                        Some(handle) if view.is_absent() => KernelInput::Weight(handle),
                        _ => KernelInput::Tensor(view),
                    }
                })
                .collect::<Vec<_>>()
        });

        // Acquire a routed-residency guard for QMoE-family boundary nodes
        // (issue #82 slice 5). This is the one dispatch-time authority: no
        // parallel ad-hoc residency check exists elsewhere. Fused-routing
        // kernels cannot name their routed set host-side before launch, so
        // `FusedRoutingUnknown` is the only requirement this call site can
        // construct today; `acquire_routed_residency`'s CUDA impl always
        // proves `WholeBank` in response (see its doc comment). The guard is
        // held exactly as long as `kernel_inputs`/the paged weight pins are
        // held below, and dropped at the same point, so the resize seam
        // cannot observe a safe point mid-dispatch. Capture: this call
        // itself does not gate on capture state; the guard's presence is
        // what makes `resize_safe_point` fail closed, and captured nodes
        // never call this residency's resize path while replaying, so a
        // guard acquired during capture recording remains valid for the
        // node's normal (non-captured) re-dispatch semantics without needing
        // separate capture-lifetime bookkeeping here.
        let _routed_residency_guard = if LazyWeightBoundary::QMoe
            .matches(&node.domain, &node.op_type)
            || LazyWeightBoundary::BlockQuantizedMoe.matches(&node.domain, &node.op_type)
        {
            inputs.iter().find_map(|value| {
                let value = (*value)?;
                let catalog = self.expert_region_candidates.get(&value)?;
                self.ep
                    .acquire_routed_residency_for_executor(
                        self.executor,
                        value.0 as u64,
                        onnx_runtime_ep_api::RoutedResidencyRequirement::FusedRoutingUnknown,
                        catalog,
                    )
                    .ok()
                    .flatten()
            })
        } else {
            None
        };

        let execution = {
            let _s = phase_span!("exec_kernel.compute");
            let metadata = views
                .iter()
                .map(|view| TensorMetadata::new(view.dtype, view.shape, !view.is_absent()))
                .collect::<Vec<_>>();
            let requirement = kernel.workspace_requirement_for_execution(views, &metadata)?;
            let prepared = match requirement.lifetime {
                WorkspaceLifetime::SessionPersistent => &mut *self.persistent_workspace,
                WorkspaceLifetime::StepScoped => &mut *self.step_workspace,
            };
            let workspace = if requirement.bytes == 0 {
                None
            } else if let Some(inherited) = self.inherited_workspace {
                let required = usize::try_from(requirement.bytes).map_err(|_| {
                    EpError::KernelFailed(format!(
                        "kernel workspace requirement {} does not fit usize",
                        requirement.bytes
                    ))
                })?;
                if required > inherited.bytes() {
                    Err(EpError::KernelFailed(format!(
                        "node {} (op '{}::{}') workspace invariant mismatch: execute requires {} bytes, enclosing preparation supplied {} bytes",
                        node.id.0,
                        node.domain,
                        node.op_type,
                        required,
                        inherited.bytes()
                    )))?;
                }
                Some(inherited)
            } else {
                let required = usize::try_from(requirement.bytes).map_err(|_| {
                    EpError::KernelFailed(format!(
                        "kernel workspace requirement {} does not fit usize",
                        requirement.bytes
                    ))
                })?;
                let needs_replacement = prepared.as_ref().is_none_or(|workspace| {
                    required > workspace.bytes || requirement.alignment > workspace.alignment
                });
                // A prepared session normally refuses to allocate at execution
                // time — its governed workspace must be reserved up front so a
                // captured device graph can bake a stable pointer. That invariant
                // only holds *within* one shape bucket, though: when execution
                // rebuckets to a larger bucket than preparation reserved for, the
                // reserved slot may be absent or undersized for the new geometry
                // (#1223). Re-run preparation in place on the eager (non-capture)
                // dispatch that a rebucket forces before any re-capture — growing
                // the slot there is safe because no captured graph references it
                // yet. This generalizes the in-bucket reservation fix from #1221
                // to the cross-bucket case, for every governed-workspace operator
                // rather than any one op-type.
                if needs_replacement
                    && (!self.workspace_preparation_required || self.eager_workspace_growth)
                {
                    // Sequential dispatch is the scratch hand-off boundary. An
                    // EP may enqueue asynchronous device work, so synchronize
                    // before retiring the old disposable workspace. Release it
                    // before acquiring the replacement to avoid charging both.
                    self.ep.sync()?;
                    let old = prepared.take().map(|workspace| workspace.buffer);
                    let buffer = self.ep.replace_workspace(
                        old,
                        required,
                        requirement.alignment,
                        requirement.role,
                    )?;
                    *prepared = Some(PreparedWorkspace {
                        buffer,
                        bytes: required,
                        alignment: requirement.alignment,
                    });
                }
                let prepared = prepared.as_mut().ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "node {} (op '{}::{}') reached execution without prepared {:?} workspace",
                        node.id.0, node.domain, node.op_type, requirement.lifetime
                    ))
                })?;
                if required > prepared.bytes || requirement.alignment > prepared.alignment {
                    Err(EpError::KernelFailed(format!(
                        "node {} (op '{}::{}') workspace invariant mismatch: execute requires {} bytes aligned to {}, prepared {} bytes aligned to {}",
                        node.id.0,
                        node.domain,
                        node.op_type,
                        required,
                        requirement.alignment,
                        prepared.bytes,
                        prepared.alignment
                    )))?;
                }
                Some(WorkspaceView::new(
                    DevicePtrMut(prepared.buffer.as_mut_ptr()),
                    prepared.bytes,
                ))
            };
            match &kernel_inputs {
                Some(inputs) if requirement.bytes == 0 => kernel.execute_with_inputs(inputs, outs),
                Some(_) => Err(EpError::KernelFailed(
                    "workspace-bearing lazy-input kernels are not supported".into(),
                )),
                None => kernel.execute_with_workspace(views, outs, workspace),
            }
        };
        execution.map_err(|error| {
                let input_types = views.iter().map(|view| view.dtype).collect::<Vec<_>>();
                let output_types = outs.iter().map(|output| output.dtype).collect::<Vec<_>>();
                let input_shapes = views
                    .iter()
                    .map(|view| view.shape.to_vec())
                    .collect::<Vec<_>>();
                let output_shapes = outs
                    .iter()
                    .map(|output| output.shape.to_vec())
                    .collect::<Vec<_>>();
                let input_names = inputs
                    .iter()
                    .map(|input| {
                        input
                            .map(|value| {
                                self.graph.value(value).name.as_deref().unwrap_or("<unnamed>")
                            })
                            .unwrap_or("<absent>")
                    })
                    .collect::<Vec<_>>();
                let output_names = outputs
                    .iter()
                    .map(|&value| {
                        self.graph.value(value).name.as_deref().unwrap_or("<unnamed>")
                    })
                    .collect::<Vec<_>>();
                SessionError::Internal(format!(
                    "node {} ({:?}, op '{}::{}', inputs {input_names:?} {input_types:?} {input_shapes:?}, outputs {output_names:?} {output_types:?} {output_shapes:?}) failed: {error}",
                    node.id.0, node.name, node.domain, node.op_type,
                ))
            })?;

        drop(kernel_inputs);
        drop(_routed_residency_guard);
        for backing in out_bufs {
            if let Some(buf) = backing.internal {
                self.buffers.insert(backing.vid, buf);
            }
        }
        Ok(())
    }
}
