use super::*;

impl Executor {

    /// Classify why one plan node cannot be recorded into a device graph, or
    /// `None` when it is capturable. Mirrors the per-node predicates the
    /// all-or-nothing audit used, but returns the reason instead of aborting so
    /// the caller can form segments around each non-capturable seam node.
    /// Seed the concrete shapes of control-flow (`If`/`Loop`/`Scan`) outputs from
    /// the previous run's buffer allocation so downstream capturable kernels that
    /// read them (e.g. GroupQueryAttention reading LongRoPE's `If`-selected
    /// cos/sin caches) resolve their input shapes and fold into captured segments
    /// instead of each forming an eager seam.
    ///
    /// ONNX shape inference cannot statically resolve a control-flow output whose
    /// branches declare different shapes, so it stays symbolic. Within a decode
    /// generation the selected branch — and thus the concrete output shape — is
    /// stable across steps, so the prior run's shape is authoritative for capture
    /// planning. A branch flip changes the shape and is detected on replay
    /// ([`Self::control_flow_seam_invalidated`]), which retires the captured graph
    /// for re-capture, so seeding never risks replaying against a stale shape.
    ///
    /// Only genuinely-unresolved outputs are seeded: a statically/symbolically
    /// resolved shape stays authoritative, matching [`ExternalBindings::seed_capture_shapes`].
    pub(super) fn seed_control_flow_capture_shapes(&self, resolved: &mut HashMap<ValueId, Vec<usize>>) {
        for &vid in &self.control_flow_output_values {
            if resolved.contains_key(&vid) {
                continue;
            }
            if let Some(shape) = self.buffer_shapes.get(&vid) {
                resolved.insert(vid, shape.clone());
            }
        }
    }

    /// Seed every still-unresolved value's shape from the most recent eager
    /// warmup's fully-resolved shape map ([`Self::capture_warm_shapes`]) so the
    /// decode ops whose output shape is data-dependent (omitted by
    /// [`Self::resolve_soft`]) — Cast/Mul/QMoE/ScatterElements downstream of a
    /// data-dependent extent — resolve their input/output shapes and fold into
    /// captured segments instead of each forming an eager seam. This generalizes
    /// the control-flow seeding above from `If`/`Loop`/`Scan` outputs to any
    /// warmed data-dependent value.
    ///
    /// Correctness rests entirely on the *decode binding signature*: the warm
    /// shapes are trusted only when the current persistent-binding signature is
    /// byte-for-byte identical to the one the warmup ran under
    /// ([`ExternalBindings::capture_signature`]). A changed pointer or capacity
    /// withholds every seed (those values stay unresolved → eager seams), and the
    /// top-level replay guard ([`Self::replay_device_graph`]) independently
    /// retires the installed graph on any binding change. Values resolvable from
    /// the current symbol bindings are never overridden — only genuinely
    /// unresolved (value-dependent) extents are seeded — and the capture pass
    /// re-resolves each seeded shape, retiring the graph if any diverged, so a
    /// per-step-varying extent can never be replayed against a stale shape.
    /// Persistent bindings and initializers are excluded (seeded/owned elsewhere).
    pub(super) fn seed_warm_decode_capture_shapes(
        &mut self,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) {
        self.capture_warm_seeded.clear();
        // Trust the warm just-in-time shapes only for the exact signature they
        // were derived under; otherwise leave values unresolved (eager seams).
        if self.capture_warm_signature.as_ref() != Some(&external.capture_signature()) {
            return;
        }
        let external_values: HashSet<ValueId> = external
            .inputs
            .keys()
            .chain(external.outputs.keys())
            .copied()
            .collect();
        let warm: Vec<(ValueId, Vec<usize>)> = self
            .capture_warm_shapes
            .iter()
            .map(|(&vid, shape)| (vid, shape.clone()))
            .collect();
        for (vid, shape) in warm {
            if resolved.contains_key(&vid)
                || external_values.contains(&vid)
                || self.graph.initializers.contains_key(&vid)
                || self.sequence_values.contains(&vid)
            {
                continue;
            }
            self.capture_warm_seeded.insert(vid, shape.clone());
            resolved.insert(vid, shape);
        }
    }

    /// Whether the control-flow seam node at plan index `pi` produced a different
    /// output shape than the most recent capture assumed. A change means a branch
    /// flip (e.g. LongRoPE short↔long at the context threshold) reallocated an
    /// output buffer a later captured segment reads, so that segment's baked
    /// device pointer is now stale and the installed graph must be retired.
    pub(super) fn control_flow_seam_invalidated(
        &self,
        pi: usize,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> bool {
        let node = self.graph.node(self.plan[pi].node_id);
        if !is_control_flow_op(&node.op_type, &node.domain) {
            return false;
        }
        self.plan[pi].outputs.iter().any(|out| {
            match (self.capture_cf_shapes.get(out), resolved.get(out)) {
                (Some(captured), Some(current)) => captured != current,
                (Some(_), None) => true,
                _ => false,
            }
        })
    }

    pub(super) fn node_capture_reason(
        &self,
        plan: &NodePlan,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Option<CaptureDecline> {
        let node = self.graph.node(plan.node_id);
        // A kernel that aborted device-graph recording on a prior capture pass is
        // quarantined by op-type: force it (and every sibling of the same op-type)
        // to an eager seam so warm-decode shape seeding can still fold the rest of
        // the graph instead of one mislabeled kernel aborting the whole capture.
        if self
            .capture_quarantine_ops
            .contains(&(canonical_domain(node), node.op_type.clone()))
        {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::CaptureRecordingFailed,
                "kernel aborted device-graph recording on a prior capture pass; \
                 quarantined to an eager seam",
            ));
        }
        let outputs_resolved = plan
            .outputs
            .iter()
            .all(|output| resolved.contains_key(output));
        let inputs_resolved = plan.inputs.iter().all(|input| match input {
            Some(value) => resolved.contains_key(value),
            None => true,
        });
        if let Some(decline) = self.ep.plan_capture_region(
            node,
            CaptureRegionShapeStatus {
                inputs_resolved,
                outputs_resolved,
            },
        ) {
            return Some(structural_capture_decline(plan.node_id, node, decline));
        }
        assert!(
            inputs_resolved && outputs_resolved,
            "EP capture-region policy admitted a node with unresolved shapes"
        );
        let input_shapes = plan
            .inputs
            .iter()
            .map(|input| {
                input.map_or_else(Vec::new, |value| {
                    resolved
                        .get(&value)
                        .cloned()
                        .expect("resolved input shape checked above")
                })
            })
            .collect();
        let key = KernelKey {
            node: plan.node_id.0,
            shapes: input_shapes,
        };
        let Some(kernel) = self.cache.entries.get(&key) else {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::KernelNotWarmed,
                "kernel has not been warmed for the requested capture shape",
            ));
        };
        kernel_capture_decline(plan.node_id, node, kernel.as_ref())
    }

    /// Partition the plan into maximal contiguous captured segments separated by
    /// eager (non-capturable) seam nodes.
    ///
    /// The CUDA EP keeps ownership of the whole claimed subgraph: this never
    /// declines a run because *some* node is non-capturable. It only returns a
    /// hard [`CaptureDeclineReport`] for a graph-level precondition (outputs must
    /// land in persistent device bindings) or when *nothing* is capturable — in
    /// which case a device graph adds no value and the caller runs fully eager
    /// (still on the CUDA EP, so placement is unchanged).
    pub(super) fn plan_capture_segments(
        &self,
        resolved: &HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> std::result::Result<CaptureSchedule, CaptureDeclineReport> {
        if self
            .graph
            .outputs
            .iter()
            .any(|output| !external.outputs.contains_key(output))
        {
            return Err(CaptureDeclineReport::one(CaptureDecline::graph(
                "every graph output must use a persistent device binding during capture",
            )));
        }

        let declines: Vec<Option<CaptureDecline>> = self
            .plan
            .iter()
            .map(|plan| self.node_capture_reason(plan, resolved))
            .collect();

        let mut segments: Vec<ScheduledSegment> = Vec::new();
        let mut boundaries: Vec<CaptureDecline> = Vec::new();
        let mut next_graph_index = 0usize;
        let mut pi = 0usize;
        while pi < declines.len() {
            let captured = declines[pi].is_none();
            let start = pi;
            while pi < declines.len() && declines[pi].is_none() == captured {
                if let Some(decline) = &declines[pi] {
                    boundaries.push(decline.clone());
                }
                pi += 1;
            }
            let graph_index = if captured {
                let index = next_graph_index;
                next_graph_index += 1;
                index
            } else {
                0
            };
            segments.push(ScheduledSegment {
                start,
                end: pi,
                captured,
                graph_index,
            });
        }

        if next_graph_index == 0 {
            return Err(CaptureDeclineReport {
                entries: boundaries,
            });
        }

        Ok(CaptureSchedule {
            segments,
            boundaries,
        })
    }

    /// Gather the warmed, capturable kernels backing one captured segment, in
    /// plan order, ready to hand to the EP's `begin_device_graph_capture` audit.
    pub(super) fn collect_segment_kernels(
        &self,
        seg: &ScheduledSegment,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Result<Vec<&dyn onnx_runtime_ep_api::Kernel>> {
        let mut kernels = Vec::with_capacity(seg.end - seg.start);
        for pi in seg.start..seg.end {
            let plan = &self.plan[pi];
            let input_shapes = plan
                .inputs
                .iter()
                .map(|input| {
                    input
                        .map(|value| resolved.get(&value).cloned())
                        .unwrap_or(Some(Vec::new()))
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    SessionError::Internal(format!(
                        "segment kernel node {} lost its resolved input shape before capture",
                        plan.node_id.0
                    ))
                })?;
            let key = KernelKey {
                node: plan.node_id.0,
                shapes: input_shapes,
            };
            let kernel = self.cache.entries.get(&key).ok_or_else(|| {
                SessionError::Internal(format!(
                    "segment kernel node {} was not warmed before capture",
                    plan.node_id.0
                ))
            })?;
            kernels.push(kernel.as_ref());
        }
        Ok(kernels)
    }
}
