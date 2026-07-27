use super::*;

impl Executor {

    /// Execute the graph with `inputs` bound by name, plus an `outer_scope` of
    /// enclosing named values a nested control-flow subgraph body may capture.
    /// The top-level session `run` passes an empty scope; a control-flow body's
    /// child executor is invoked with its enclosing graph's live values so a
    /// deeply-nested body can still reach an outer capture (ONNX lexical scope).
    pub(super) fn run_scoped(
        &mut self,
        inputs: &[(&str, &Tensor)],
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
    ) -> Result<Vec<Option<SessionOutput>>> {
        match self.run_scoped_mode(inputs, outer_scope, external, RunMode::Eager)? {
            ScopedRunResult::Executed(outputs) => Ok(outputs),
            ScopedRunResult::NotCapturable(_) => unreachable!("eager runs are always executed"),
        }
    }

    pub(super) fn run_scoped_mode(
        &mut self,
        inputs: &[(&str, &Tensor)],
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
        mode: RunMode,
    ) -> Result<ScopedRunResult> {
        // Distinguish the outermost (top-level graph) run from nested
        // control-flow subgraph runs so the phase profiler can attribute
        // overhead to the right layer.
        thread_local! {
            static RUN_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        }
        let depth = RUN_DEPTH.with(|d| {
            let cur = d.get();
            d.set(cur + 1);
            cur
        });
        struct DepthGuard;
        impl Drop for DepthGuard {
            fn drop(&mut self) {
                RUN_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
            }
        }
        let _depth_guard = DepthGuard;
        let nested = depth > 0;
        // Zero-copy view metadata is run-scoped: a value that aliased another's
        // buffer last run must not leak into this one (buffers may be resized).
        self.views.clear();
        self.pinned.clear();
        // Sequence values and their zero-copy element-backed tensors are equally
        // run-scoped (element Arcs from a prior run must not leak in).
        self.sequences.clear();
        self.seq_elem_values.clear();
        self.restore_shared_buffers()?;

        // --- Resolve shapes from the actual bound inputs --------------------
        let _phase_setup = phase_span!(if nested {
            "run_scoped.setup_total.child"
        } else {
            "run_scoped.setup_total.top"
        });
        let bindings = self.bind_symbols(inputs, external)?;

        for (name, _) in inputs {
            let vid = self.input_index[*name];
            if external.inputs.contains_key(&vid) {
                return Err(SessionError::Internal(format!(
                    "input '{name}' is bound both as a host tensor and a persistent device buffer"
                )));
            }
        }

        // Every required input must be supplied.
        let mut provided: HashSet<ValueId> = inputs
            .iter()
            .filter_map(|(name, _)| self.input_index.get(*name).copied())
            .collect();
        provided.extend(external.inputs.keys().copied());
        for &vid in &self.required_inputs {
            if !provided.contains(&vid) {
                let name = self
                    .graph
                    .value(vid)
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("value#{}", vid.0));
                return Err(SessionError::InputNotFound { name });
            }
        }

        // Substitute the bindings into every value → concrete shapes, then size
        // the run-scoped buffers from them (reused when unchanged). Values with a
        // data-dependent shape stay unresolved here and are filled in during the
        // execution loop, once their producing node's inputs are concrete.
        //
        // F5 Stage 1: on the top-level CPU eager decode path the steady-state
        // decode-plan memo replays the length-invariant partition of this map
        // instead of rebuilding it every token. It is a pure optimization of
        // `resolve_soft` (a function of `bindings` only, since on the eager path
        // no external/control-flow/warm seeding runs — that is Capture/Replay
        // only), gated OFF by default and asserted byte-identical under
        // `decode_memo_verify`.
        //
        // Persistent device-I/O bindings (the KV cache) are the NORMAL decode
        // case, not an exclusion: the real native decode path always carries them
        // (ext_in/ext_out non-empty), and `bind_symbols` already folds every
        // external *input* binding's shape into `bindings`, so the growing KV
        // length symbol L is captured by the replay guard exactly like any other
        // varying symbol. The memo additionally fingerprints the external binding
        // set (`decode_external_signature`) with L abstracted to its symbolic
        // identity, so pure-L growth replays while any structural change (binding
        // added/removed, role flip, dtype change) forces a rebuild.
        let decode_memo_eligible = self.decode_memo_enabled
            && mode == RunMode::Eager
            && !nested
            && self.ep.device_type() != DeviceType::Cuda;
        let mut resolved = {
            let _s = phase_span!("run_scoped.resolve_soft");
            if decode_memo_eligible {
                self.resolve_soft_decode_memo(&bindings, external)
            } else {
                // Observability: if the master switch is on but this step is
                // structurally ineligible (CUDA, nested, non-eager), count it so
                // an over-restrictive gate silently excluding the real decode path
                // is never shipped again (the F5 regression Ripley caught).
                if self.decode_memo_enabled && !nested {
                    self.decode_memo_ineligible_count += 1;
                }
                let mut resolved = self.resolve_soft(&bindings);
                if mode != RunMode::Eager {
                    // Persistent bindings seed the kernel-visible geometry selected by
                    // their input/output contracts. Seed only unresolved values:
                    // statically/symbolically resolved shapes remain authoritative.
                    external.seed_capture_shapes(&mut resolved);
                    // Control-flow outputs (e.g. LongRoPE cos/sin caches) are symbolic to
                    // shape inference but stable within a generation: seed their concrete
                    // prior-run shape so downstream capturable consumers fold into
                    // captured segments instead of forming per-consumer eager seams.
                    self.seed_control_flow_capture_shapes(&mut resolved);
                    // Steady-state decode ops (Cast/Mul/QMoE/ScatterElements …) whose
                    // output shape is data-dependent stay unresolved in `resolve_soft`
                    // and would each form an eager seam even though their kernels are
                    // already capture-safe. Seed their exact just-in-time shapes from
                    // the eager warmup — but only for the identical persistent-binding
                    // signature the warmup ran under, so a changed pointer/capacity
                    // withholds the seed instead of baking a stale shape.
                    self.seed_warm_decode_capture_shapes(&mut resolved, external);
                }
                resolved
            }
        };
        // --- F5 Stage 2: reinstate the cached invariant view/buffer plan --------
        // On a memo Replayed step whose per-source buffer identity still matches,
        // reinstate the zero-copy view aliases (lever 1) instead of clearing and
        // rebuilding them, mark the pure-view nodes for dispatch elision (lever 3),
        // and exclude the invariant partition from buffer sizing (lever 2). Taken
        // out of `self` for the duration so an errored step drops it (a stale alias
        // can never be reinstated into a later replay); restored on success.
        let mut stage2_plan: Option<DecodeViewPlan> = None;
        let mut stage2_candidate: Option<DecodeViewPlan> = None;
        let mut stage2_excluded: Option<HashSet<ValueId>> = None;
        if decode_memo_eligible
            && !self.decode_view_plan_disabled
            && self.decode_memo_last_action == DecodeMemoAction::Replayed
            && let Some(plan) = self.decode_view_plan.take()
        {
            if !plan.validated {
                // Candidate plan built on the preceding Rebuilt step: run this step
                // in full (no reinstate/elide) so every invariant view is freshly
                // built, then confirm two-real-step byte-identity below before it is
                // ever used to elide. This is the second-real-step confirmation.
                stage2_candidate = Some(plan);
            } else if self.stage2_buffer_sig_matches(&plan) {
                self.decode_view_plan_sig_mismatch_streak = 0;
                // Lever 1: reinstate the invariant zero-copy view aliases and
                // re-pin their source buffers (conservative liveness). Also
                // restore each elided output's resolved shape to the view's own
                // shape — the value the elided view node would have written into
                // `resolved` (which can differ from the pre-loop `resolve_soft`
                // shape Stage 1 restored, e.g. a Reshape with an inferred dim), so
                // downstream consumers read the identical geometry as a full step.
                for (vid, view) in &plan.retained_views {
                    self.views.insert(*vid, view.clone());
                    resolved.insert(*vid, view.shape.clone());
                }
                for &src in &plan.pinned_sources {
                    self.pinned.insert(src);
                }
                self.decode_views_reused_count += plan.retained_views.len() as u64;
                self.decode_dispatch_elided_count += plan.elided_nodes.len() as u64;
                // Lever 2: exclude the memo's proven-invariant partition from
                // per-step buffer sizing — those buffers were sized under the
                // rebuild and are byte-identical (guarded by the buffer-identity
                // signature above); the compute path still self-heals any output
                // whose length unexpectedly differs.
                if let Some(memo) = self.decode_memo.as_ref() {
                    stage2_excluded = Some(memo.invariant_shapes.keys().copied().collect());
                }
                stage2_plan = Some(plan);
            } else {
                // A source buffer moved/resized under a plan that classified it
                // invariant: retire the plan (dropped here) and run the full step.
                // After repeated mismatches the assumption is untrustworthy on this
                // model, so latch Stage 2 off for the session (defense-in-depth).
                self.decode_view_plan_sig_mismatch_streak += 1;
                if self.decode_view_plan_sig_mismatch_streak >= STAGE2_SIG_MISMATCH_LIMIT {
                    self.decode_view_plan_disabled = true;
                }
            }
        }
        let external_values = external
            .inputs
            .keys()
            .chain(external.outputs.keys())
            .copied()
            .collect::<HashSet<_>>();
        for &vid in &external_values {
            if let Some(old) = self.buffers.remove(&vid) {
                self.ep.deallocate(old)?;
            }
            self.shared_buffers.remove(&vid);
            self.buffer_shapes.remove(&vid);
        }
        {
            let _s = phase_span!("run_scoped.size_buffers");
            match &stage2_excluded {
                // Stage 2 (lever 2): size only the values outside the memo's
                // invariant partition (variant/JIT/external) — the invariant
                // buffers are reused untouched from the rebuild step.
                Some(invariant) => {
                    let mut excluded = external_values.clone();
                    excluded.extend(invariant.iter().copied());
                    self.size_buffers_excluding(&resolved, &excluded)?;
                }
                None => {
                    self.size_buffers_excluding(&resolved, &external_values)?;
                }
            }
        }

        // --- Bind input bytes into their (now correctly sized) buffers ------
        for (name, tensor) in inputs {
            let vid = self.input_index[*name];
            let buf = self
                .buffers
                .get_mut(&vid)
                .expect("input value has a buffer");
            self.ep.copy_from_host(tensor.as_bytes(), buf)?;
        }
        drop(_phase_setup);

        // --- Execute nodes ---------------------------------------------------
        // Iterate by index so a control-flow node can take `&mut self` (it must
        // build/reuse child executors) while an ordinary kernel node uses the
        // disjoint-field borrow split inside `exec_kernel_node`.
        match mode {
            RunMode::Eager => {
                let _s = phase_span!(if nested {
                    "run_scoped.plan_eager.child"
                } else {
                    "run_scoped.plan_eager.top"
                });
                // F5 Stage 2: elide the plan's pure-view nodes only in production.
                // Under `decode_memo_verify` (the R1 safety net) run every node so
                // the invariant views are freshly rebuilt, then assert each equals
                // the reinstated alias (bytes/shape/ptr) — proving reuse is exact.
                let verify_stage2 = self.decode_memo_verify && stage2_plan.is_some();
                let verify_snapshot: Option<Vec<(ValueId, ValueView)>> = if verify_stage2 {
                    stage2_plan.as_ref().map(|p| p.retained_views.clone())
                } else {
                    None
                };
                let elided = if verify_stage2 {
                    None
                } else {
                    stage2_plan.as_ref().map(|p| &p.elided_nodes)
                };
                self.run_plan_eager(&mut resolved, outer_scope, external, elided)?;
                if let (Some(snapshot), Some(plan)) = (&verify_snapshot, &stage2_plan) {
                    for (vid, cached) in snapshot {
                        let fresh = self.views.get(vid).unwrap_or_else(|| {
                            panic!(
                                "F5 Stage 2 verify: elided view value#{} was not rebuilt by a \
                                 full dispatch",
                                vid.0
                            )
                        });
                        assert!(
                            fresh.source == cached.source
                                && fresh.shape == cached.shape
                                && fresh.strides == cached.strides
                                && fresh.byte_offset == cached.byte_offset,
                            "F5 Stage 2 verify: cached view for value#{} ({cached:?}) diverged \
                             from a freshly built one ({fresh:?}) — invariant view reuse is unsound",
                            vid.0
                        );
                    }
                    assert!(
                        self.stage2_buffer_sig_matches(plan),
                        "F5 Stage 2 verify: a cached view source buffer moved during the step"
                    );
                }
                // F5 Stage 2 plan lifecycle: rebuild the cached view plan at the
                // successful end of a memo Rebuilt step (the plan was invalidated
                // at the top of the rebuild path, so a mid-step error leaves it
                // `None`); restore the in-flight plan after a successful replay.
                if decode_memo_eligible {
                    match self.decode_memo_last_action {
                        DecodeMemoAction::Rebuilt => {
                            self.decode_view_plan = self.build_decode_view_plan();
                        }
                        DecodeMemoAction::Replayed => {
                            if let Some(cand) = stage2_candidate.take() {
                                // This replay ran full dispatch as the candidate's
                                // second-real-step confirmation: keep only the nodes
                                // whose view is byte-identical to the built one, and
                                // promote to validated (or drop if none survive).
                                self.decode_view_plan = self.validate_decode_view_plan(cand);
                            } else if let Some(plan) = stage2_plan.take() {
                                self.decode_view_plan = Some(plan);
                            }
                        }
                        _ => {}
                    }
                }
                // Snapshot the exact just-in-time shapes this warm run resolved,
                // together with the persistent-binding signature they were
                // derived under. Capture-mode seeding replays these shapes only
                // when a later step presents this exact signature (pointer- and
                // capacity-stable), so a changed binding forces recapture, never
                // a stale-shape replay. Skipped on the memo-eligible CPU decode
                // path: that path never captures (CPU EP), so cloning the whole
                // ~600-entry resolved map every token would be pure waste and
                // would defeat the memo's allocation amortization.
                if !decode_memo_eligible {
                    self.capture_warm_shapes = resolved.clone();
                    self.capture_warm_signature = Some(external.capture_signature());
                }
            }
            RunMode::Capture => {
                // A fresh capture may have resized/reallocated the `If` output
                // buffers, so force every `If` to actually execute its branch
                // this run (repopulating those buffers) rather than trusting the
                // steady-decode memo. Cleared before segmentation so the branch
                // runs as a normal eager seam during the capture pass.
                self.if_last_predicate.clear();
                // Partition the claimed subgraph into maximal capturable segments
                // separated by non-capturable seam nodes. Only a graph-level hard
                // decline (e.g. no persistent output binding, or nothing
                // capturable at all) falls back to a fully eager run.
                //
                // Warm-decode shape seeding can admit a node whose kernel wrongly
                // advertises capture support but aborts device-graph recording
                // (e.g. a stream synchronize, which CUDA rejects mid-capture).
                // A single such kernel aborts the whole segmented capture. Rather
                // than regress to a fully eager step, quarantine the offending
                // op-type to a forced eager seam and re-plan/re-record: the
                // genuinely-capturable ops still fold while the mislabeled kernel
                // stays eager. Re-recording a fixed-capacity decode step is
                // idempotent (same position/token → same values into the same
                // slots), so retrying is safe. Bounded by the node count.
                let max_capture_attempts = self.plan.len() + 1;
                let schedule = 'capture: loop {
                    let schedule = match self.plan_capture_segments(&resolved, external) {
                        Ok(schedule) => schedule,
                        Err(report) => return Ok(ScopedRunResult::NotCapturable(report)),
                    };
                    self.last_capture_failed_node = None;
                    match self.run_plan_segmented(
                        &schedule,
                        RunMode::Capture,
                        &mut resolved,
                        outer_scope,
                        external,
                    ) {
                        Ok(_) => break 'capture schedule,
                        Err(error) => {
                            let _ = self.ep.reset_device_graph();
                            // Quarantine the op-type that aborted recording and
                            // retry, unless we already quarantined it (no
                            // progress), hit the attempt bound, or cannot
                            // attribute the failure to a node.
                            let quarantined =
                                self.last_capture_failed_node.take().and_then(|node_id| {
                                    let node = self.graph.node(node_id);
                                    let key = (canonical_domain(node), node.op_type.clone());
                                    self.capture_quarantine_ops.insert(key).then_some(())
                                });
                            if quarantined.is_some()
                                && self.capture_quarantine_ops.len() < max_capture_attempts
                            {
                                // Re-plan with the offending op-type forced eager.
                                self.if_last_predicate.clear();
                                continue 'capture;
                            }
                            self.capture_schedule = None;
                            self.capture_segmentation.clear();
                            self.capture_cf_shapes.clear();
                            self.capture_warm_seeded.clear();
                            return Ok(ScopedRunResult::NotCapturable(CaptureDeclineReport::one(
                                CaptureDecline::graph(format!(
                                    "segmented CUDA graph capture failed: {error}"
                                )),
                            )));
                        }
                    }
                };
                // A warm-seeded shape that the capture pass re-resolved to a
                // different value means the seed was stale for this step (a
                // genuinely per-step-varying interior extent). The recorded
                // graph would replay that shape unconditionally, so retire it
                // and decline: the caller re-warms and either re-captures (if
                // the shape restabilizes) or keeps this op eager. This upholds
                // "recapture when any shape changes; never replay a stale graph."
                if let Some((vid, seeded)) = self
                    .capture_warm_seeded
                    .iter()
                    .find(|(vid, seeded)| resolved.get(vid) != Some(*seeded))
                    .map(|(vid, seeded)| (*vid, seeded.clone()))
                {
                    let current = resolved.get(&vid).cloned();
                    let _ = self.ep.reset_device_graph();
                    self.capture_schedule = None;
                    self.capture_segmentation.clear();
                    self.capture_cf_shapes.clear();
                    self.capture_warm_seeded.clear();
                    return Ok(ScopedRunResult::NotCapturable(CaptureDeclineReport::one(
                        CaptureDecline::graph(format!(
                            "warm decode shape seed for value#{} ({seeded:?}) diverged from the \
                             captured shape ({current:?}); recapturing",
                            vid.0
                        )),
                    )));
                }
                // Snapshot the concrete control-flow output shapes this capture
                // assumed so a later replay can detect a branch flip that changes
                // them and retire the now-stale installed graph.
                self.capture_cf_shapes = self
                    .control_flow_output_values
                    .iter()
                    .filter_map(|vid| resolved.get(vid).map(|shape| (*vid, shape.clone())))
                    .collect();
                self.capture_segmentation = schedule.boundaries.clone();
                if capture_segmentation_logging_enabled() {
                    log_capture_segmentation(&schedule);
                }
                self.capture_schedule = Some(schedule);
            }
            RunMode::Replay => {
                // Move the schedule out so the segmented runner can take `&mut
                // self`; restore it afterwards for the next step's replay.
                let Some(schedule) = self.capture_schedule.take() else {
                    return Ok(ScopedRunResult::NotCapturable(CaptureDeclineReport::one(
                        CaptureDecline::graph(
                            "segmented device graph replay requested without a capture schedule",
                        ),
                    )));
                };
                let still_valid = self.run_plan_segmented(
                    &schedule,
                    RunMode::Replay,
                    &mut resolved,
                    outer_scope,
                    external,
                )?;
                if still_valid {
                    self.capture_schedule = Some(schedule);
                } else {
                    // A control-flow branch flip changed a seeded output shape:
                    // the remaining plan already ran eagerly this step (correct
                    // token), but the installed segments are stale. Retire the
                    // device graph so the caller re-warms and re-captures for the
                    // new branch. `capture_schedule` stays `None`.
                    self.capture_segmentation.clear();
                    self.capture_cf_shapes.clear();
                    self.device_graph_signature = None;
                    self.ep.reset_device_graph()?;
                }
            }
        }

        // --- Collect graph outputs into owned tensors -----------------------
        // A view output (a layout op whose result aliases an input buffer) is
        // materialized to contiguous owned bytes here — external consumers and
        // the Python/DLPack boundary expect contiguous tensors.
        let _phase_collect = phase_span!(if nested {
            "run_scoped.collect_outputs.child"
        } else {
            "run_scoped.collect_outputs.top"
        });
        let mut results = Vec::with_capacity(self.graph.outputs.len());
        let mut host_output_bytes = 0usize;
        let output_vids: Vec<ValueId> = self.graph.outputs.clone();
        for vid in output_vids {
            if external.outputs.contains_key(&vid) {
                results.push(None);
                continue;
            }
            if self.sequence_values.contains(&vid) {
                let sequence = self.sequences.get(&vid).cloned().ok_or_else(|| {
                    SessionError::Internal(format!(
                        "sequence graph output value#{} has no live runtime value",
                        vid.0
                    ))
                })?;
                results.push(Some(SessionOutput::Sequence(sequence)));
                continue;
            }

            let dtype = self.value_dtypes[&vid];
            let shape = resolved[&vid].clone();
            // Top-level outputs: hand the produced host buffer to the caller
            // zero-copy when safe (the KV-cache round-trip the decode hot path
            // otherwise pays every step). Child (subgraph) outputs are copied
            // back into the parent scope, so keep them on the copy path.
            if !nested && let Some(tensor) = self.try_move_host_output(vid, &shape, dtype)? {
                results.push(Some(SessionOutput::Tensor(tensor)));
                continue;
            }
            let bytes = self.contiguous_bytes(vid, &shape, dtype)?;
            host_output_bytes += bytes.len();
            results.push(Some(SessionOutput::Tensor(Tensor::from_raw(
                dtype, shape, &bytes,
            )?)));
        }
        // Attribution aid: at the top level, the number of graph-output bytes
        // materialized to host each run is the per-step cost of *not* keeping
        // outputs (e.g. a growing KV cache) in persistent device/host bindings.
        // Recorded as a counter (bytes as the "nanos" field) so the phase table
        // exposes total and per-call host-output traffic without extra logging.
        if !nested {
            phase_profile::record("collect_outputs.top_host_bytes", host_output_bytes as u128);
        }
        // F5 Stage 1: hand the just-used shape map (now including this step's
        // data-dependent JIT tail) back to the persistent working buffer so the
        // next replay step can take it in place — retaining every invariant
        // `Vec`'s allocation — rather than allocating a fresh map/`Vec`s per
        // token. Only on the memo-eligible CPU decode path; otherwise the buffer
        // stays untouched (and empty).
        if decode_memo_eligible {
            self.decode_memo_resolved = std::mem::take(&mut resolved);
        }
        Ok(ScopedRunResult::Executed(results))
    }
}
