use super::*;

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
            self.exec_control_flow(pi, resolved, outer_scope)
        } else if is_sequence {
            self.exec_sequence_node(pi, resolved, external)
        } else {
            self.exec_kernel_node(pi, resolved, external)
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
                let op_type = self.graph.node(self.plan[pi].node_id).op_type.clone();
                let start = Instant::now();
                let result =
                    self.exec_plan_node(pi, resolved, outer_scope, external, OpCaptureTrace::Eager);
                let elapsed = start.elapsed();
                let entry = timings.entry(op_type).or_insert((Duration::ZERO, 0));
                entry.0 += elapsed;
                entry.1 += 1;
                result?;
            }
            print_op_profile(run_start.elapsed(), timings);
        } else {
            for pi in 0..self.plan.len() {
                if elided.is_some_and(|set| set.contains(&pi)) {
                    continue;
                }
                self.exec_plan_node(pi, resolved, outer_scope, external, OpCaptureTrace::Eager)?;
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
                            ep.begin_device_graph_capture(&kernels)?;
                        }
                        // Any early return (`?`) while recording this segment
                        // must end the stream capture before it propagates —
                        // otherwise the stream stays wedged in capture mode and
                        // the caller's `reset_device_graph()` is a no-op (reset
                        // is rejected while capturing). The guard aborts the
                        // capture on drop; `disarm()` hands off to the normal
                        // `end_device_graph_capture()` on the success path.
                        let mut capture_guard = SegmentCaptureGuard::arm(ep.as_ref());
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
                        ep.end_device_graph_capture()?;
                        ep.replay_device_graph_segment(seg.graph_index)?;
                    }
                    RunMode::Replay => {
                        ep.replay_device_graph_segment(seg.graph_index)?;
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
    pub(super) fn refill_input_shapes(&mut self, pi: usize, resolved: &HashMap<ValueId, Vec<usize>>) {
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
        // fresh `Vec<Vec<usize>>` for shape lookup — see `refill_input_shapes`.
        self.refill_input_shapes(pi, resolved);
        let inputs = &self.plan[pi].inputs;
        let outputs = &self.plan[pi].outputs;
        let input_dtypes = &self.plan[pi].input_dtypes;
        let output_dtypes = &self.plan[pi].output_dtypes;
        let input_shapes = &self.scratch_input_shapes;

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
            let input_values: Vec<Option<Vec<i64>>> = inputs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v.and_then(|vid| self.shape_input_i64(vid, &input_shapes[i], input_dtypes[i]))
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
        // Fixed-capacity KV for the default-domain Attention op. Its present
        // K/V outputs (slots 1..) are consumer-less graph outputs bound to a
        // growing device cache. Expose them to the kernel at the binding's
        // physical capacity so the kernel can append the new token into a fixed
        // per-head slot (constant stride, no per-step restride) instead of
        // repacking the whole cache densely. The valid attended length is still
        // derived from the logical past+current extent, so this only widens the
        // *storage* stride and never changes what the kernel attends over. Only
        // present slots that are bound sub-shape (logical != physical) capacity
        // buffers are widened; a dense/unbound present keeps its inferred shape.
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
            }
        }
        let capabilities = self.ep.capabilities();
        let accepts_lazy_weights =
            LazyWeightBoundary::BlockQuantizedMoe.matches(&node.domain, &node.op_type);
        let has_lazy_inputs = accepts_lazy_weights
            && inputs.iter().any(|input| {
                input
                    .and_then(|value| self.weight_handles.get(&value))
                    .is_some_and(|handle| handle.is_lazy_for(&capabilities))
            });

        // Resolve each input's real geometry (root buffer + strides/offset) and
        // bounds-check it. View inputs read through their recorded strides.
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
                });
                continue;
            }
            if accepts_lazy_weights
                && self
                    .weight_handles
                    .get(&vid)
                    .is_some_and(|handle| handle.is_lazy_for(&capabilities))
            {
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
                });
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
            });
        }
        drop(_build_inputs_span);

        let ep = self.ep.clone();

        // Bind the mutated fields as disjoint locals so `self` is never borrowed
        // whole while the kernel (from `cache`) and the buffers/views are held.
        let graph = &self.graph;
        let cache = &mut self.cache;
        let weight_handles = &self.weight_handles;
        let buffers = &mut self.buffers;
        let buffer_shapes = &mut self.buffer_shapes;
        let shared_buffers = &mut self.shared_buffers;
        let views_meta = &mut self.views;
        let pinned = &mut self.pinned;

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

        let opset = effective_opset(graph, node);
        let constant_inputs: Vec<bool> = inputs
            .iter()
            .map(|input| {
                input.is_some_and(|vid| {
                    graph.initializers.contains_key(&vid)
                        || views_meta
                            .get(&vid)
                            .is_some_and(|view| graph.initializers.contains_key(&view.source))
                })
            })
            .collect();
        let kernel = {
            let _s = phase_span!("exec_kernel.get_kernel");
            cache.get_or_create(
                node_id,
                node,
                input_shapes,
                input_dtypes,
                &constant_inputs,
                opset,
                ep.as_ref(),
            )?
        };
        // --- Zero-copy view fast path ---------------------------------------
        // Ask the kernel whether its outputs are strided views over its inputs
        // (a layout/movement op such as Slice). If so, record view metadata
        // aliasing the source buffer and skip compute + allocation entirely.
        if !has_lazy_inputs && let Some(specs) = kernel.view_outputs(&views, outputs.len()) {
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
                let root = match views_meta.get(&in_vid) {
                    Some(v) => v.source,
                    None => in_vid,
                };
                let root_len = buffers.get(&root).map(|b| b.len()).ok_or_else(|| {
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
                    !pinned.contains(&ovid),
                    "value#{} is pinned as a live view source yet is being reproduced",
                    ovid.0
                );
                if let Some(old) = buffers.remove(&ovid) {
                    ep.deallocate(old)?;
                }
                shared_buffers.remove(&ovid);
                buffer_shapes.remove(&ovid);
                views_meta.insert(
                    ovid,
                    ValueView {
                        source: root,
                        shape: spec.shape.clone(),
                        strides: spec.strides,
                        byte_offset: spec.byte_offset,
                    },
                );
                pinned.insert(root);
                resolved.insert(ovid, spec.shape);
            }
            return Ok(());
        }

        // --- Compute path ----------------------------------------------------
        // Size (allocate or reuse) each output's contiguous buffer, JIT-sizing
        // data-dependent ones. A value that was a view on a prior run has no
        // buffer here and is freshly allocated.
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
                    let name = graph.value(ovid).name.as_deref().unwrap_or("<unnamed>");
                    return Err(SessionError::Internal(format!(
                        "external output '{name}' has {:?} {:?} ({} bytes), kernel requires {:?} {:?} ({need} bytes)",
                        value.dtype, value.shape, value.len, output_dtypes[oi], dims
                    )));
                }
                continue;
            }
            let fits = buffers.get(&ovid).map(|b| b.len() == need).unwrap_or(false);
            if !fits {
                // Never free a buffer that has a live view alias (would dangle
                // the viewer). Unreachable under SSA topo order, but enforced.
                debug_assert!(
                    !pinned.contains(&ovid),
                    "value#{} is pinned as a live view source yet is being resized",
                    ovid.0
                );
                if let Some(old) = buffers.remove(&ovid) {
                    ep.deallocate(old)?;
                }
                shared_buffers.remove(&ovid);
                let buf = ep.allocate(need, TensorLayout::contiguous().alignment)?;
                buffers.insert(ovid, buf);
            }
        }

        // Auto-materialization gate: a strided (view) input feeding a kernel
        // that does not accept strided input on that slot is gathered into a
        // private contiguous temp so contiguous-assuming kernels stay correct.
        // Temps must outlive the views that borrow them.
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
            let src =
                unsafe { std::slice::from_raw_parts(info.base_ptr as *const u8, info.root_len) };
            let gathered = gather_view(src, &info.shape, &info.strides, info.byte_offset, esize);
            let strides = compute_contiguous_strides(&info.shape);
            mat.push(Some((gathered, strides)));
        }

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
        struct OutBacking {
            vid: ValueId,
            internal: Option<DeviceBuffer>,
            ptr: *mut std::ffi::c_void,
            len: usize,
            device: onnx_runtime_ir::DeviceId,
        }
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
                let mut buf = buffers.remove(&vid).ok_or_else(|| {
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

        let kernel_inputs = has_lazy_inputs.then(|| {
            inputs
                .iter()
                .zip(views.iter().copied())
                .map(|(value, view)| {
                    value
                        .and_then(|value| weight_handles.get(&value))
                        .filter(|handle| handle.is_lazy_for(&capabilities))
                        .map(KernelInput::Weight)
                        .unwrap_or(KernelInput::Tensor(view))
                })
                .collect::<Vec<_>>()
        });
        let execution = {
            let _s = phase_span!("exec_kernel.compute");
            match &kernel_inputs {
                Some(inputs) => kernel.execute_with_inputs(inputs, &mut outs),
                None => kernel.execute(&views, &mut outs),
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
        drop(views);
        drop(outs);
        for backing in out_bufs {
            if let Some(buf) = backing.internal {
                buffers.insert(backing.vid, buf);
            }
        }
        Ok(())
    }

    /// Read the integer *values* of input `vid` as `i64`, materializing a view
    /// first if needed. Used to resolve data-dependent output shapes (e.g. a
    /// `Slice` whose `ends` is produced at runtime). Returns `None` if the value
    /// has no readable buffer/view or its dtype is not an integer.
    pub(super) fn input_i64(&self, vid: ValueId, shape: &[usize], dtype: DataType) -> Option<Vec<i64>> {
        let bytes = self.contiguous_bytes(vid, shape, dtype).ok()?;
        bytes_as_i64(&bytes, dtype)
    }

    /// Bounded integer reader for dynamic shape propagation. Views and sequence
    /// elements can have a tiny logical shape backed by a much larger root
    /// allocation, so cap that allocation before `contiguous_bytes` can copy it.
    pub(super) fn shape_input_i64(&self, vid: ValueId, shape: &[usize], dtype: DataType) -> Option<Vec<i64>> {
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

    pub(super) fn shape_input_f64(&self, vid: ValueId, shape: &[usize], dtype: DataType) -> Option<Vec<f64>> {
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
    pub(super) fn nms_input_f64(&self, vid: ValueId, shape: &[usize], dtype: DataType) -> Option<Vec<f64>> {
        if dtype != DataType::Float32 {
            return None;
        }
        let bytes = self.contiguous_bytes(vid, shape, dtype).ok()?;
        bytes_as_f64(&bytes, dtype)
    }
}
