use super::*;

/// Print, to stderr, how the capture pass split a claimed subgraph into captured
/// device-graph segments and eager seam nodes, and why each seam exists. Gated
/// by `ONNX_GENAI_LOG_CAPTURE_SEGMENTS` for transparency into segmentation.
pub(super) fn log_capture_segmentation(schedule: &CaptureSchedule) {
    let captured = schedule.captured_segments();
    let seams = schedule.segments.len() - captured;
    eprintln!(
        "[onnx-genai-capture] segmented CUDA graph: {captured} captured segment(s), \
         {seams} eager seam(s)"
    );
    for boundary in &schedule.boundaries {
        match boundary.node_id {
            Some(id) => {
                let seam_label = boundary
                    .seam_reason
                    .map(SeamReason::label)
                    .unwrap_or("unclassified-seam");
                eprintln!(
                    "[onnx-genai-capture]   seam node {id} ({}::{}) [{seam_label}] ran eagerly: {}",
                    boundary.domain, boundary.op_type, boundary.reason
                );
            }
            None => eprintln!(
                "[onnx-genai-capture]   seam ({}): {}",
                boundary.op_type, boundary.reason
            ),
        }
    }
}

/// Observable control-flow executor statistics. These counters make subgraph
/// reuse deterministic to test without relying on timing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ControlFlowStats {
    /// Child executors built, including shape-signature rebuilds.
    pub subgraph_builds: u64,
    /// Child subgraph invocations served by those executors.
    pub subgraph_runs: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceAllocationCounts {
    pub allocations: u64,
    pub frees: u64,
}

/// Structural execution path used by a node during a captured run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapturePathKind {
    /// Recorded into a device graph and replayed.
    CaptureRegion,
    /// Dispatched eagerly while remaining on the device.
    EagerDeviceSeam,
    /// Host-driven work or a host round-trip between captured regions.
    HostSeam,
}

impl CapturePathKind {
    /// Stable short label used by capture diagnostics.
    pub const fn label(self) -> &'static str {
        match self {
            Self::CaptureRegion => "capture-region",
            Self::EagerDeviceSeam => "eager-device-seam",
            Self::HostSeam => "host-seam",
        }
    }
}

/// Structural reason a node forms an eager seam during device-graph capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeamReason {
    /// Host-driven control-flow or sequence semantics.
    HostControlFlowOrSequence,
    /// A data-dependent output shape was unresolved before capture.
    UnresolvedOutputShape,
    /// A data-dependent input shape was unresolved before capture.
    UnresolvedInputShape,
    /// The requested concrete kernel shape has not completed warmup.
    KernelNotWarmed,
    /// The selected device kernel explicitly opts out of capture.
    KernelCaptureUnsupported,
    /// The build-time capture classifier disqualified this node: one of its
    /// input or output shapes references a GROWING symbol (KV/total-sequence
    /// length that increases each decode step). Capturing such a node bakes a
    /// stale launch grid/count from the warmed extent into the replayed device
    /// graph → silent decode corruption. This is an authoritative HARD VETO
    /// applied centrally, independent of the kernel's own `capture_support()`
    /// opinion, so no kernel returning `CaptureSupport::Supported` can re-admit
    /// a disqualified node.
    ClassifierDisqualified,
    /// The kernel aborted device-graph *recording* (e.g. it advertised capture
    /// support but issued a stream synchronize, which CUDA rejects mid-capture)
    /// and was quarantined to a forced eager seam so the rest of the graph can
    /// still be captured.
    CaptureRecordingFailed,
}

impl SeamReason {
    /// Execution path implied by this structural seam cause.
    pub const fn path_kind(self) -> CapturePathKind {
        match self {
            Self::HostControlFlowOrSequence => CapturePathKind::HostSeam,
            Self::UnresolvedOutputShape
            | Self::UnresolvedInputShape
            | Self::KernelNotWarmed
            | Self::CaptureRecordingFailed
            | Self::ClassifierDisqualified
            | Self::KernelCaptureUnsupported => CapturePathKind::EagerDeviceSeam,
        }
    }

    /// Stable short path-kind label used by capture diagnostics.
    pub const fn label(self) -> &'static str {
        self.path_kind().label()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// One actionable reason a device-graph capture attempt was rejected.
pub struct CaptureDecline {
    /// Graph node id, or `None` for graph/capture-lifecycle requirements.
    pub node_id: Option<u32>,
    /// ONNX operator type, or `"<graph>"` for graph-level requirements.
    pub op_type: String,
    /// Canonical ONNX domain (`"ai.onnx"` by default), or `"nxrt"` graph-level.
    pub domain: String,
    /// Failed precondition and, where applicable, how to reach the capture path.
    pub reason: String,
    /// Structural seam classification, or `None` for graph-level hard preconditions.
    pub seam_reason: Option<SeamReason>,
}

impl CaptureDecline {
    pub(super) fn node(
        node_id: NodeId,
        node: &Node,
        seam_reason: SeamReason,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            node_id: Some(node_id.0),
            op_type: node.op_type.clone(),
            domain: canonical_domain(node),
            reason: reason.into(),
            seam_reason: Some(seam_reason),
        }
    }

    pub(super) fn graph(reason: impl Into<String>) -> Self {
        Self {
            node_id: None,
            op_type: "<graph>".to_string(),
            domain: "nxrt".to_string(),
            reason: reason.into(),
            seam_reason: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// Structured reasons a device graph could not be captured.
pub struct CaptureDeclineReport {
    /// All graph- and node-level declines found by the pre-capture audit.
    pub entries: Vec<CaptureDecline>,
}

impl CaptureDeclineReport {
    pub(super) fn one(decline: CaptureDecline) -> Self {
        Self {
            entries: vec![decline],
        }
    }

    /// Whether the capture audit found no declines.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// One node-level reason the requested execution provider declined placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionProviderDecline {
    /// Stable graph/subgraph node identity used in diagnostics.
    pub node: String,
    /// Canonical ONNX domain (`"ai.onnx"` for the default domain).
    pub domain: String,
    /// ONNX operator type.
    pub op_type: String,
    /// Actionable reason returned by [`ExecutionProvider::supports_op`].
    pub reason: String,
}

/// Structured report for an accelerator request that executes on CPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionProviderFallbackReport {
    /// Requested provider name, such as `"cuda_ep"`.
    pub requested_provider: String,
    /// Provider that will execute the graph.
    pub fallback_provider: String,
    /// Number of executable graph/subgraph nodes assigned to the fallback EP.
    pub assigned_node_count: usize,
    /// Sorted distinct `domain::op` classes assigned to the fallback EP.
    pub assigned_ops: Vec<String>,
    /// Nodes the requested provider did not claim, with colocated reasons.
    pub declines: Vec<ExecutionProviderDecline>,
}

impl std::fmt::Display for ExecutionProviderFallbackReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} nodes assigned to CPU (ops: {}) — GPU EP {} did not claim {} node(s): {}. \
             Heterogeneous CUDA+CPU placement is unavailable, so the whole session uses {}",
            self.assigned_node_count,
            self.assigned_ops.join(", "),
            self.requested_provider,
            self.declines.len(),
            format_cuda_coverage_issues(&self.declines),
            self.fallback_provider,
        )
    }
}

impl std::fmt::Display for CaptureDeclineReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDA graph capture rejected")?;
        for (index, decline) in self.entries.iter().enumerate() {
            if index == 0 {
                write!(f, ": ")?;
            } else {
                write!(f, "; ")?;
            }
            match decline.node_id {
                Some(node_id) => write!(
                    f,
                    "node {node_id} ({}::{}) — {}",
                    decline.domain, decline.op_type, decline.reason
                )?,
                None => write!(f, "{} — {}", decline.op_type, decline.reason)?,
            }
        }
        Ok(())
    }
}

pub enum DeviceGraphCaptureResult {
    Captured(Vec<Option<Tensor>>),
    NotCapturable(CaptureDeclineReport),
}

// Boxing the executed variant would allocate on every warmed device-bound run.
#[allow(clippy::large_enum_variant)]
pub(super) enum ScopedRunResult {
    Executed(ScopedOutputs),
    NotCapturable(CaptureDeclineReport),
}

pub(super) fn kernel_capture_decline(
    node_id: NodeId,
    node: &Node,
    kernel: &dyn Kernel,
) -> Option<CaptureDecline> {
    kernel.capture_support().reason().map(|reason| {
        CaptureDecline::node(node_id, node, SeamReason::KernelCaptureUnsupported, reason)
    })
}

pub(super) fn structural_capture_decline(
    node_id: NodeId,
    node: &Node,
    decline: StructuralCaptureDecline,
) -> CaptureDecline {
    let seam_reason = match decline {
        StructuralCaptureDecline::HostControlFlowOrSequence => {
            SeamReason::HostControlFlowOrSequence
        }
        StructuralCaptureDecline::UnresolvedOutputShape => SeamReason::UnresolvedOutputShape,
        StructuralCaptureDecline::UnresolvedInputShape => SeamReason::UnresolvedInputShape,
    };
    CaptureDecline::node(node_id, node, seam_reason, decline.reason())
}

/// Whether verbose segmented-capture diagnostics are printed to stderr.
///
/// Gated identically to op profiling so a run can surface exactly where the
/// CUDA EP split a claimed subgraph into captured segments and eager seam nodes.
pub(super) fn capture_segmentation_logging_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ONNX_GENAI_LOG_CAPTURE_SEGMENTS")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

/// How a scoped run drives the device-graph lifecycle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RunMode {
    /// No capture: execute every node eagerly on the stream.
    Eager,
    /// First capture pass: partition the plan into segments, record each
    /// capturable segment into its own device graph, and run the non-capturable
    /// seam nodes eagerly in between.
    Capture,
    /// Subsequent steps: replay each captured segment graph in order, re-running
    /// only the eager seam nodes.
    Replay,
}

/// The device-graph capture disposition of a single op, used to annotate its
/// trace span with **why** it was or was not captured. Carries a borrowed
/// reason string rather than an owned one so an untraced run never allocates.
#[derive(Clone, Copy)]
pub(super) enum OpCaptureTrace<'a> {
    /// Plain eager run — no capture attempt is in progress for this op.
    Eager,
    /// The op was recorded into a captured device-graph segment.
    Captured,
    /// The op runs eagerly as a capture seam; `reason` explains why it could
    /// not be recorded into a device graph (which kernel/predicate declined).
    Rejected(&'a str),
}

/// Trace-arg key: whether an op was captured into a device graph.
pub(super) const ARG_CAPTURE_STATUS: &str = "capture_status";
/// Trace-arg key: why an op was not captured into a device graph.
pub(super) const ARG_CAPTURE_REASON: &str = "capture_reason";

impl OpCaptureTrace<'_> {
    /// Annotate the active op-span with this capture disposition. A no-op for
    /// [`OpCaptureTrace::Eager`] (nothing was being captured) and when no span
    /// is active.
    pub(super) fn annotate(self) {
        match self {
            OpCaptureTrace::Eager => {}
            OpCaptureTrace::Captured => {
                annotate_current_span_with(|| {
                    onnx_runtime_tracer::Args::new().with(ARG_CAPTURE_STATUS, "captured")
                });
            }
            OpCaptureTrace::Rejected(reason) => {
                annotate_current_span_with(|| {
                    onnx_runtime_tracer::Args::new()
                        .with(ARG_CAPTURE_STATUS, "rejected")
                        .with(ARG_CAPTURE_REASON, reason)
                });
            }
        }
    }
}

/// Scope guard that guarantees an in-progress segment capture is always ended
/// before its enclosing function returns.
///
/// During [`RunMode::Capture`], nodes are recorded between
/// `begin_device_graph_capture` and `end_device_graph_capture`. If a node fails
/// mid-record, the `?` early return would otherwise skip the end call and leave
/// the CUDA stream wedged in capture mode — the caller's
/// `reset_device_graph()` is then a no-op (reset is rejected while capturing),
/// so every later eager/replay launch fails with `STREAM_CAPTURE_INVALIDATED`.
///
/// While armed, [`Drop`] aborts the capture (ending stream capture and
/// discarding the half-recorded graph). The success path calls [`disarm`] and
/// then ends the capture normally via `end_device_graph_capture`.
///
/// [`disarm`]: SegmentCaptureGuard::disarm
pub(super) struct SegmentCaptureGuard<'a> {
    pub(super) ep: &'a dyn ExecutionProvider,
    pub(super) slot: DeviceGraphSlot,
    pub(super) armed: bool,
}

impl<'a> SegmentCaptureGuard<'a> {
    pub(super) fn arm(ep: &'a dyn ExecutionProvider, slot: DeviceGraphSlot) -> Self {
        Self {
            ep,
            slot,
            armed: true,
        }
    }

    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for SegmentCaptureGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            // Best-effort: the abort itself may fail, but the caller is already
            // unwinding a capture failure and will reset the lifecycle next.
            let _ = self.ep.abort_device_graph_capture_in(self.slot);
        }
    }
}

/// One contiguous run of plan nodes that either share a captured device graph or
/// all execute eagerly (a non-capturable seam).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ScheduledSegment {
    /// First plan index (inclusive).
    pub(super) start: usize,
    /// One past the last plan index (exclusive).
    pub(super) end: usize,
    /// `true` when `[start, end)` is captured into a device graph; `false` for an
    /// eager seam of non-capturable (but still device-placed or CPU) nodes.
    pub(super) captured: bool,
    /// Capture-order index of this segment's graph in the EP, set only when
    /// `captured`.
    pub(super) graph_index: usize,
}

/// The plan's partition into captured segments and eager seams, plus the
/// structured reason each segment boundary exists (which node forced the split).
///
/// Recorded once during the capture pass and reused for every subsequent replay
/// so the interleaving of graph replays and eager seam execution is stable.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct CaptureSchedule {
    pub(super) segments: Vec<ScheduledSegment>,
    /// One entry per non-capturable seam node, explaining why it forced a
    /// boundary (its `CaptureSupport::Unsupported` reason or structural cause).
    pub(super) boundaries: Vec<CaptureDecline>,
}

impl CaptureSchedule {
    /// Number of captured device-graph segments (1 for a whole-subgraph capture).
    pub(super) fn captured_segments(&self) -> usize {
        self.segments.iter().filter(|seg| seg.captured).count()
    }

    /// Whether the whole plan captured as a single graph (no eager seams).
    pub(super) fn is_single_graph(&self) -> bool {
        self.segments.len() == 1 && self.segments[0].captured
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DeviceBindingSignature {
    pub(super) input_name: String,
    pub(super) binds_input: bool,
    pub(super) output_name: Option<String>,
    pub(super) dtype: DataType,
    pub(super) physical_shape: Vec<usize>,
    pub(super) device_ptr: usize,
}

impl Executor {
    /// Pin the fixed-capacity KV sequence-axis symbols CONSTANT so the capture
    /// classifier ADMITS the attention nodes (`GroupQueryAttention`, capacity
    /// `Attention`, `IndexShare`) into device-graph capture instead of vetoing
    /// each as a growing-seq seam. Returns the number of symbols newly pinned.
    ///
    /// # Why this is correct
    /// The build-time classifier ([`compute_capture_disqualifying_symbols`]) seeds
    /// the KV present/past seq axis as GROWING from the STATIC graph shape, where
    /// it is left symbolic. But when the engine binds fixed-capacity,
    /// device-resident KV (physical `[.., max_len, ..]`, valid length read
    /// on-device — GQA's `seqlens_k`), the attention kernel's launch grid is
    /// capacity-sized (bounded by the physically-allocated `max_len`), NOT
    /// growing-seq-sized. The present shape is `past_capacity.max(total)` =
    /// `max_len` (a constant within the capture), and overflow is caught by
    /// `total_len > max_len`. So every replayed step has an identical launch grid
    /// — a captured replay is shape-static and correct. This mirrors the runtime
    /// fixed-capacity present-shape widening the default-domain `Attention` path
    /// already performs ([`super::dispatch`]), extended to the same effect for
    /// `com.microsoft::GroupQueryAttention`.
    ///
    /// # Safety / guard preservation
    /// Only [`collect_capacity_pinned_kv_symbols`] symbols are pinned: nodes whose
    /// EVERY past-KV input is read as physical capacity
    /// ([`super::geometry::kernel_input_uses_physical_capacity`]). A growing-concat
    /// or paged KV form does not qualify and stays vetoed. This is invoked by the
    /// engine ONLY when the runtime actually binds fixed-capacity KV and CUDA
    /// graphs are enabled ([`DecodeCudaState::new`]); a growing/paged decoder
    /// clears `graph_enabled` and never calls it. KV growth / capacity-bucket
    /// rebucket invalidates and re-captures the graph (`invalidate_graph`), and a
    /// binding-signature mismatch on replay is independently caught, so a pinned
    /// symbol is never replayed against a stale grid.
    pub(crate) fn pin_fixed_capacity_kv_capture_symbols(&mut self) -> usize {
        let mut pinned = collect_capacity_pinned_kv_symbols(&self.graph);
        // Also pin the decode-freeze-safe attention-mask length symbol(s): the
        // mask/causal-bias axis is a fixed-capacity constant on the single-token
        // decode path (the frozen-width mask saturates to the true valid length),
        // exactly like a pinned KV seq axis. Without this the mask-builder cone
        // AND every capacity-form `Attention` consuming the bias stay eager seams
        // (an MLA / HF-causal model captures with dozens of interleaved seams that
        // replay incoherently); pinning admits them into capture. See
        // [`collect_freeze_safe_mask_symbols`].
        pinned.extend(collect_freeze_safe_mask_symbols(&self.graph));
        if pinned.is_empty() {
            return 0;
        }
        // Recompute the disqualifying set with the pinned KV seq axes treated as
        // constant capacity, so their attention consumers are admitted.
        self.capture_growing_symbols =
            compute_capture_disqualifying_symbols_excluding(&self.graph, &pinned);
        let count = pinned.len();
        self.capacity_pinned_kv_symbols = pinned;
        if std::env::var("ONNX_GENAI_LOG_GROWING_SYMBOLS").is_ok() {
            eprintln!(
                "[onnx-genai-capture] pinned {} fixed-capacity KV seq / freeze-safe mask symbol(s): {:?}; \
                 disqualifying set now {} symbol(s)",
                count,
                self.capacity_pinned_kv_symbols,
                self.capture_growing_symbols.len(),
            );
        }
        count
    }

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
    pub(super) fn seed_control_flow_capture_shapes(
        &self,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
    ) {
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
        self.cap_mut().capture_warm_seeded.clear();
        // Trust the warm just-in-time shapes only for the exact signature they
        // were derived under; otherwise leave values unresolved (eager seams).
        if self.cap().capture_warm_signature.as_ref() != Some(&external.capture_signature()) {
            return;
        }
        let external_values: HashSet<ValueId> = external
            .inputs
            .keys()
            .chain(external.outputs.keys())
            .copied()
            .collect();
        let warm: Vec<(ValueId, Vec<usize>)> = self
            .cap()
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
            self.cap_mut()
                .capture_warm_seeded
                .insert(vid, shape.clone());
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
            match (self.cap().capture_cf_shapes.get(out), resolved.get(out)) {
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
            .cap()
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
        // CENTRAL HARD VETO — the build-time capture classifier is authoritative.
        // If any input/output shape of this node references a GROWING symbol
        // (KV/total-sequence length that increases each decode step), the node is
        // capture-DISQUALIFIED: replaying it would bake a stale launch grid/count
        // from the warmed extent into the device graph → silent decode
        // corruption. This veto runs AFTER structural classification
        // (`plan_capture_region` → host control-flow/sequence and unresolved-shape
        // declines keep precedence, so a disqualified host node reports the HOST
        // seam, not a device seam) but BEFORE the kernel admission check below, so
        // a kernel that returns `CaptureSupport::Supported` unconditionally (and
        // never stores the classifier flag) can NEVER re-admit a disqualified
        // node. A node is capture-eligible only if (classifier says
        // seq-independent) AND (kernel says Supported for the shape);
        // over-declining is correctness-safe (extra eager nodes), under-declining
        // is corruption, so we bias to veto.
        if !node_capture_seq_independent(&self.graph, node, &self.capture_growing_symbols) {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::ClassifierDisqualified,
                "capture classifier disqualified this node: an input or output \
                 shape depends on a growing (KV/total-sequence-length) symbol, so \
                 capturing it would replay a stale launch grid — forced eager seam",
            ));
        }
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
