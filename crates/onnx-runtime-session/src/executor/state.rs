use super::*;

/// Host-side captured-graph bookkeeping for ONE [`DeviceGraphSlot`]. The
/// executor holds one of these per slot (see [`Executor::slot_capture`]) so the
/// M=1 `Primary` decode graph and the M=k+1 `Verify` speculative-verify graph
/// can be captured, seeded, and replayed independently on the same EP/stream
/// without either step overwriting the other's signature/schedule/warm-shape
/// state — the fix for the `replays=0` MTP verify blocker (a single shared set
/// of these fields let the M=1 decode clobber the Verify slot between its
/// capture and replay). `Primary` maps to index 0, so a greedy decode (which
/// only ever drives `Primary`) sees byte-identical bookkeeping to the historical
/// single-field layout.
#[derive(Default)]
pub(crate) struct SlotCaptureState {
    /// Exact provider installation token for this executor/slot.
    pub(super) device_graph_token: Option<DeviceGraphToken>,
    /// Binding signature (I/O name + dtype + physical shape + device ptr) the
    /// installed graph for this slot was captured under; a replay whose bindings
    /// differ retires the graph. `None` when no device graph is installed.
    pub(super) device_graph_signature: Option<Vec<DeviceBindingSignature>>,
    /// The captured-segment schedule from the most recent successful capture,
    /// reused to interleave segment replays with eager seam nodes on each
    /// subsequent step. `None` when no device graph is installed.
    pub(super) capture_schedule: Option<CaptureSchedule>,
    /// Structured segment-boundary reasons from the most recent capture, retained
    /// for diagnostics after `capture_schedule` is taken for replay.
    pub(super) capture_segmentation: Vec<CaptureDecline>,
    /// Concrete control-flow output shapes the most recent capture assumed (a
    /// snapshot of the seeded shapes from [`Executor::control_flow_output_values`]).
    /// On replay the control-flow seam re-executes eagerly; if it now produces a
    /// different shape (a branch flip, e.g. LongRoPE short↔long at the context
    /// threshold) the installed graph's baked device pointers are stale, so the
    /// step falls back to eager and the graph is retired for re-capture.
    pub(super) capture_cf_shapes: HashMap<ValueId, Vec<usize>>,
    /// Persistent-binding signature the most recent eager warmup ran under (see
    /// [`ExternalBindings::capture_signature`]). Capture-mode shape seeding only
    /// trusts the warm just-in-time shapes recorded in [`Executor::buffer_shapes`]
    /// when a later step presents this exact signature, so any changed pointer
    /// or capacity withholds the seed instead of baking a stale shape.
    pub(super) capture_warm_signature: Option<Vec<ExternalCaptureSig>>,
    /// Every value's concrete just-in-time shape as resolved by the most recent
    /// eager warmup. The data-dependent decode shapes we seed for capture are
    /// JIT-sized on the compute path (which populates `buffers` but not
    /// [`Executor::buffer_shapes`]), so the authoritative warm geometry is
    /// snapshotted from the eager run's fully-resolved shape map, not the buffer
    /// bookkeeping.
    pub(super) capture_warm_shapes: HashMap<ValueId, Vec<usize>>,
    /// The warm decode shapes actually seeded into the most recent capture. After
    /// the capture pass re-resolves each node's true shape, a divergence here
    /// means the warm seed was stale for this step, so the graph is retired and
    /// the caller re-warms/re-captures rather than replaying a stale shape.
    pub(super) capture_warm_seeded: HashMap<ValueId, Vec<usize>>,
    /// `(domain, op_type)` pairs whose kernel aborted device-graph *recording*
    /// during a capture pass (e.g. it declared `CaptureSupport::Supported` but
    /// issued a stream synchronize, which CUDA rejects mid-capture). Warm-decode
    /// shape seeding can admit such a node once its output shape is known; if the
    /// resulting capture fails, the offending op-type is quarantined here and
    /// [`Executor::node_capture_reason`] then forces every node of that op-type
    /// to a forced eager seam, so the capture is re-planned and the remaining
    /// genuinely-capturable ops still fold. Grows monotonically within an
    /// executor: a kernel that breaks recording once breaks it every time.
    pub(super) capture_quarantine_ops: HashSet<(String, String)>,
}

/// The compiled, runnable graph: buffers + plan + kernel cache. Owned by the
/// public [`InferenceSession`](crate::InferenceSession).
pub(crate) struct Executor {
    pub(super) graph: Graph,
    /// Kept alive so external-weight memory maps outlive buffer population —
    /// **and**, since the weight-streaming change, so borrowed initializer
    /// buffers that alias this store's mmap bytes stay valid for the executor's
    /// whole lifetime. `weights` MUST outlive every live use of `buffers`: a
    /// borrowed `DeviceBuffer` in `buffers` points into `weights`' mmap/inline
    /// storage. Teardown is safe because `Executor::drop` **drains and
    /// deallocates `buffers` first** (a borrowed deallocate is a no-op free), so
    /// no buffer still aliases `weights` when the `Arc<WeightStore>` field is
    /// dropped afterwards — no use-after-free regardless of field drop order.
    pub(super) weights: Arc<WeightStore>,
    pub(super) ep: Arc<dyn ExecutionProvider>,
    /// Opt-in mixed-provider coordinator. When present, ordinary tensor runs
    /// dispatch through its per-provider child executors; the legacy fields
    /// remain an inert empty scaffold so default single-EP behavior is unchanged.
    pub(super) heterogeneous: Option<Box<crate::hetero::HeterogeneousExecutor>>,
    /// Which of the EP's captured-graph slots this executor drives. Every
    /// executor gets its own independent CUDA-graph slot on the shared EP: the
    /// main/inline decode executors default to [`DeviceGraphSlot::Primary`]
    /// (byte-identical to the historical single-slot behaviour), while a
    /// verify-dedicated sibling drives [`DeviceGraphSlot::Verify`] so a fixed
    /// M=k+1 speculative verify graph can be captured and replayed without
    /// invalidating the M=1 decode graph every step (option-c). All EP graph
    /// calls this executor makes route through this slot; defaulting to Primary
    /// keeps the field inert unless a caller explicitly retargets it.
    pub(super) graph_slot: DeviceGraphSlot,
    /// Immutable namespace identity for every captured graph owned by this
    /// executor. Sibling sessions sharing one provider receive distinct owners.
    pub(super) graph_owner: DeviceGraphOwner,
    /// Setup-time owner slot. Warming borrows this proof directly instead of
    /// looking up an owner or cloning a shared handle.
    pub(super) validation_registration: Option<DeviceValidationRegistration>,
    /// Most recent submitted generation in this executor's registered
    /// owner-scoped validation slot.
    pub(super) pending_device_validation: Option<DeviceValidationToken>,
    /// Lazy external initializers available only at the nxrt fused-MoE boundary.
    /// Stock EPs ignore this map and keep receiving the resident buffers below.
    pub(super) weight_handles: HashMap<ValueId, WeightHandle>,
    /// Per-expert region candidates for `com.microsoft::QMoE` expert-bank
    /// initializers, keyed by the same [`ValueId`] as [`Self::weight_handles`].
    ///
    /// This is purely additive bookkeeping for a future pluggable residency
    /// policy (see issue #82): it records *where each expert's bytes would
    /// live* inside an already-fully-resident lazy weight, but nothing reads it
    /// to change binding, allocation, or paging behavior today. A value present
    /// here is classified [`onnx_runtime_loader::Pageability::Pageable`]; a
    /// value classified `NonPageable` records its rejection reason for
    /// diagnostics instead, so no entry silently vanishes — either the
    /// candidate is a validated partition of the expert bank, or the exact
    /// reason it isn't is recorded.
    ///
    /// Read only under `#[cfg(test)]` via [`Executor::expert_region_candidates`]
    /// today; production consumers land in a follow-up slice per issue #82.
    #[allow(dead_code)]
    pub(super) expert_region_candidates: HashMap<ValueId, onnx_runtime_loader::WeightRegionCatalog>,
    /// The validated [`onnx_runtime_ep_api::ResidencyPlan`] derived from
    /// `expert_region_candidates` using [`onnx_runtime_ep_api::WholeBankResidentPolicy`]
    /// (the only shipped policy today). Pure bookkeeping: it does not own
    /// allocation, copying, or synchronization, and no production dispatch
    /// path reads it yet. Read only under `#[cfg(test)]` via
    /// [`Executor::residency_plan`].
    #[allow(dead_code)]
    pub(super) residency_plan: onnx_runtime_ep_api::ResidencyPlan,
    pub(super) prefetch_issue_nodes: std::sync::Mutex<HashMap<ValueId, usize>>,
    pub(super) prefetch_lookahead_nodes: usize,
    /// One device buffer per backed value. Static values are allocated once at
    /// build; dynamic (symbol-shaped) values are allocated per run and cached
    /// here so a run whose resolved shape is unchanged reuses the allocation.
    pub(super) buffers: HashMap<ValueId, DeviceBuffer>,
    /// The concrete shape each live buffer in [`Self::buffers`] is currently
    /// sized for — the reuse key for run-scoped buffers.
    pub(super) buffer_shapes: HashMap<ValueId, Vec<usize>>,
    /// Owned input buffers parked for the duration of one run while
    /// [`Self::buffers`] instead holds a read-only *borrowed* handle over the
    /// caller's host tensor (see `prepare_run_buffers`). Always drained back into
    /// `buffers` before `run_scoped_mode` returns, so it is empty between runs
    /// and no borrowed handle ever outlives the `&Tensor` it aliases.
    pub(super) parked_input_buffers: Vec<(ValueId, DeviceBuffer)>,
    /// Stale output buffers whose owner became a zero-copy view while a device
    /// graph was being recorded. Freeing a buffer mid-capture is illegal (the
    /// pooled-memory unmap synchronizes the copy stream, which CUDA rejects
    /// during stream capture), so `install_view_outputs` parks the orphaned
    /// allocation here and [`Self::run_plan_segmented`] flushes it (a normal
    /// `deallocate`) once `end_device_graph_capture` has closed the capture.
    /// Empty outside an in-progress capture pass.
    pub(super) capture_deferred_frees: Vec<DeviceBuffer>,
    /// Loader-produced (possibly symbolic) shape of every value.
    pub(super) value_shapes: HashMap<ValueId, Shape>,
    /// Element type of every value.
    pub(super) value_dtypes: HashMap<ValueId, DataType>,
    /// Topologically ordered execution plan (structure only; shapes per run).
    pub(super) plan: Vec<NodePlan>,
    /// name → value id for the graph inputs the caller must supply.
    pub(super) input_index: HashMap<String, ValueId>,
    /// Value ids the caller must supply at `run` (graph inputs minus initializers).
    pub(super) required_inputs: Vec<ValueId>,
    /// Whether any value in the graph carries a symbolic dim. A fully-static
    /// graph is materialized eagerly at build; a symbolic graph defers buffer
    /// allocation and kernel compilation to the first `run` that fixes shapes.
    pub(super) has_symbols: bool,
    pub(super) cache: KernelCache,
    /// name → value id for every named value in this graph (inputs, outputs,
    /// initializers and interior SSA values). Used to resolve outer-scope
    /// captures referenced by name from a nested control-flow subgraph body.
    pub(super) name_index: HashMap<String, ValueId>,
    /// Reusable child executors for this graph's control-flow subgraph bodies,
    /// keyed by `(control-flow node, subgraph attr key)`. Built lazily on first
    /// execution (once concrete input shapes are known) and **reused across
    /// Loop/Scan iterations** — the whole point of the efficiency directive: a
    /// body's topo-sort, buffer sizing and kernel compilation happen once, then
    /// every iteration is just a re-bind + dispatch. Rebuilt only if a later
    /// invocation's external input shapes differ from the ones it was compiled
    /// for (a shape-varying loop body — rare).
    pub(super) subgraph_execs: HashMap<(NodeId, String), ChildExecutor>,
    pub(super) control_flow_stats: ControlFlowStats,
    /// Per-`If` memo of the last observed branch predicate. During steady decode
    /// a loop-invariant `If` (e.g. the LongRoPE cos/sin cache selector) keeps the
    /// same predicate every step, so its branch outputs are already resident in
    /// their persistent buffers. The predicate is still read each step (the
    /// correctness guard); only the redundant branch re-execution — here two
    /// `Constant` materializations plus their host→device cache copies — is
    /// skipped. A predicate flip re-runs the branch (and, on an output-shape
    /// change, retires the captured graph via the existing seam invalidation).
    pub(super) if_last_predicate: HashMap<NodeId, bool>,
    /// Per-slot host-side captured-graph bookkeeping, indexed by
    /// [`Self::graph_slot`]'s [`DeviceGraphSlot::index`]. Splitting these fields
    /// off per slot lets the M=1 `Primary` decode graph and the M=k+1 `Verify`
    /// speculative-verify graph coexist on one executor without clobbering each
    /// other's captured signature/schedule/warm state every step (the `replays=0`
    /// MTP blocker). Greedy only ever drives `Primary` (index 0), so its
    /// bookkeeping stays byte-identical to the historical single-field layout.
    pub(super) slot_capture: [SlotCaptureState; DeviceGraphSlot::COUNT],
    /// Output value ids of every control-flow (`If`/`Loop`/`Scan`) node. ONNX
    /// shape inference cannot statically resolve a control-flow output whose
    /// branches declare different shapes (e.g. LongRoPE's `If` selecting a short
    /// vs. long RoPE cos/sin cache), so such outputs stay symbolic and any
    /// downstream capturable kernel that reads one would form a per-consumer
    /// eager seam. Within a decode generation the selected branch is stable, so
    /// [`Self::seed_control_flow_capture_shapes`] seeds each output's concrete
    /// shape from the prior run for capture planning, folding those consumers
    /// back into captured segments. Computed once at build.
    pub(super) control_flow_output_values: HashSet<ValueId>,
    /// Build-time set of GROWING symbols: the KV/past/total-sequence-length
    /// symbols on the sequence axis of attention `present`/`past` KV caches (GQA,
    /// default `Attention`, `IndexShare`, `CompressedSparseAttention` — incl. the
    /// ratio-4 `selections` axis), unioned with a generic scan of the model's
    /// declared `past…`/`present…` rank-4 KV I/O, and then CLOSED under shape
    /// inference's broadcast symbol unification. Computed once at build (see
    /// [`compute_capture_growing_symbols`](super::kernel_cache::compute_capture_growing_symbols)).
    /// The shared pointwise/elementwise/bitwise/prelu capture gate rejects an op
    /// whose INPUT **or** OUTPUT references any of these symbols — a denylist on
    /// both edges. The OUTPUT check keeps eager any op sized by a growing length;
    /// the INPUT check plus the class closure keep eager both a first-hop alias
    /// and any DOWNSTREAM consumer that only ever sees the pinned-looking
    /// representative (finding 1). A benign FRESH symbol (warm-decode seeded,
    /// non-growing, not unified with a growing one) is absent from this set, so
    /// ops carrying it stay capturable — preserving the 154→34 collapse.
    pub(super) capture_growing_symbols: HashSet<SymbolId>,
    /// KV sequence-axis symbols pinned CONSTANT to their bound physical capacity
    /// by [`Self::pin_fixed_capacity_kv_capture_symbols`] once the engine has
    /// bound fixed-capacity, device-valid-length KV (see
    /// [`super::kernel_cache::collect_capacity_pinned_kv_symbols`]). Empty until
    /// the engine calls the pin; recorded for diagnostics/tests and to document
    /// which symbols were removed from `capture_growing_symbols`.
    pub(super) capacity_pinned_kv_symbols: HashSet<SymbolId>,
    /// Node whose kernel returned an error while recording a captured segment,
    /// set transiently by [`Self::run_plan_segmented`] so the capture retry loop
    /// can quarantine its op-type. `None` outside a failed capture pass.
    pub(super) last_capture_failed_node: Option<NodeId>,
    /// Run-scoped zero-copy **view** metadata (§5.4). A value id present here is
    /// a strided view aliasing another value's buffer (a layout/movement-op
    /// output such as `Slice`) rather than an owner in [`Self::buffers`]. Built
    /// during the run loop and cleared at the start of every run.
    pub(super) views: HashMap<ValueId, ValueView>,
    /// Run-scoped set of buffer-owning value ids that have ≥1 live view alias.
    /// A pinned buffer must not be reused or deallocated for the remainder of
    /// the run (conservative liveness: a source buffer outlives every view that
    /// aliases it, guaranteeing no use-after-free). Cleared each run.
    pub(super) pinned: HashSet<ValueId>,
    /// Value ids whose runtime value is a **sequence of tensors** rather than a
    /// single tensor (produced by `SequenceEmpty`/`SequenceConstruct`/
    /// `SequenceInsert`/`SequenceErase`/`SplitToSequence`). Computed once at
    /// build; these values own no [`DeviceBuffer`] and are skipped by buffer
    /// sizing — their storage lives in [`Self::sequences`] at run time.
    pub(super) sequence_values: HashSet<ValueId>,
    /// Most recent activation-memory planner result from a measured top-level
    /// eager run once shapes and view aliases are concrete. Stage-2 replay and
    /// nested runs skip re-planning and leave this as the last measured top-level
    /// result. Observational only: the executor still owns one buffer per value
    /// until issue #514's allocator surgery lands.
    pub(super) activation_memory_plan: Option<ActivationMemoryPlanStats>,
    /// Allocation owners promoted into ref-counted storage when a tensor enters
    /// an ONNX Sequence. `buffers` retains a non-owning dispatch alias, while
    /// sequence elements clone the owner Arc. At the next run boundary, after
    /// all sequence handles are cleared, the unique owner is restored to
    /// `buffers` before any input/output can be mutated.
    pub(super) shared_buffers: HashMap<ValueId, Arc<SharedTensorBuffer>>,
    /// Run-scoped storage for sequence values: `value id → SequenceValue`. A
    /// [`SequenceValue`] holds its elements as `Arc`-shared immutable tensors,
    /// so a sequence op that inserts/erases/etc. shares element `Arc`s with the
    /// source rather than deep-copying bytes (see [`crate::sequence`] for the
    /// no-copy + no-race invariants). Cleared each run.
    pub(super) sequences: HashMap<ValueId, SequenceValue>,
    /// Run-scoped **zero-copy** backing for a *tensor* value whose bytes are a
    /// shared sequence element (the output of `SequenceAt`): the tensor aliases
    /// the element's `Arc` instead of owning a `DeviceBuffer`, so no bytes are
    /// copied out of the sequence. A downstream kernel reads it through a
    /// [`TensorView`] over the `Arc`'s bytes; it is materialized to owned bytes
    /// only at the graph-output/control-flow boundary. Cleared each run.
    pub(super) seq_elem_values: HashMap<ValueId, SeqTensor>,
    pub(super) execution_provider_fallback_report: Option<ExecutionProviderFallbackReport>,
    /// Shared runtime trace context. Defaults to a disabled [`TraceContext::noop`]
    /// so an untraced run pays only a single relaxed atomic load per op when
    /// deciding whether to open a span. When enabled, the executor opens one
    /// span per executed op so kernels can attach kernel-variant and
    /// capture-rejection reasons via [`annotate_current_span_with`].
    pub(super) trace: TraceContext,
    /// Reusable scratch for the resolved input shapes of the node currently
    /// being dispatched by [`Self::exec_kernel_node`]. Refilled (truncate +
    /// refill, retaining inner `Vec` capacity) once per node via
    /// [`Self::refill_input_shapes`], so a steady-state decode step performs no
    /// per-node `Vec<Vec<usize>>` allocation for shape lookup. Reuse invariant:
    /// it is fully rewritten at the top of each `exec_kernel_node` call and only
    /// read within that same call — never aliased or carried across nodes.
    pub(super) scratch_input_shapes: Vec<Vec<usize>>,
    /// Bounded per-executor dispatch metadata reused by fixed-shape runs.
    pub(super) scratch_input_infos: Vec<InInfo>,
    pub(super) scratch_output_shapes: Vec<Vec<usize>>,
    pub(super) scratch_output_strides: Vec<Vec<i64>>,
    pub(super) scratch_materialized_inputs: Vec<Option<(Vec<u8>, Vec<i64>)>>,
    /// Persistent run metadata reused by device-bound eager execution.
    pub(super) scratch_external_bindings: ExternalBindings,
    pub(super) scratch_resolved_shapes: HashMap<ValueId, Vec<usize>>,
    /// Stable value traversal order prepared once with the executor.
    pub(super) all_value_ids: Vec<ValueId>,
    /// F5 Stage 1 — master switch for the steady-state decode-plan memo. Default
    /// ON; disabled by `ONNX_GENAI_DECODE_MEMO=0`. Consulted on the top-level CPU
    /// eager decode path — including the normal persistent-KV-binding case.
    pub(super) decode_memo_enabled: bool,
    /// When set (`ONNX_GENAI_DECODE_MEMO_VERIFY=1`, or always under
    /// `debug_assertions`), every memo replay is asserted equal to a fresh
    /// `resolve_soft` — the R1 verifiable safety net. Off in release by default.
    pub(super) decode_memo_verify: bool,
    /// The active decode-plan memo, primed after two consecutive plan-matching
    /// eager steps and rebuilt on any signature change.
    pub(super) decode_memo: Option<DecodePlanMemo>,
    /// Bindings of the previous memo-eligible eager step, diffed against the
    /// current step to derive the varying-symbol set (R1 two-real-step rule).
    pub(super) decode_memo_prev_bindings: Option<HashMap<SymbolId, usize>>,
    /// Diagnostic: what the memo did on the most recent memo-eligible eager
    /// step. Exposed to the guard tests.
    pub(super) decode_memo_last_action: DecodeMemoAction,
    /// F5 Stage 1 — persistent working shape map reused across decode steps.
    /// On a replay step it is taken in place (no allocation): its previous
    /// just-in-time entries are stripped, the length-invariant partition is left
    /// untouched (byte-identical by construction), and only the variant tail is
    /// re-substituted into its existing `Vec`s. The run loop then refills the
    /// small data-dependent tail. This is what makes replay genuinely
    /// allocation-amortized (Stage 1's whole purpose) rather than a per-token
    /// `HashMap`/`Vec` rebuild.
    pub(super) decode_memo_resolved: HashMap<ValueId, Vec<usize>>,
    /// Diagnostic counters (proof the memo actually fires on the real path, so a
    /// gate that silently excludes it is never shipped again). Incremented per
    /// memo-eligible eager step; a summary is emitted on drop when
    /// `ONNX_GENAI_DECODE_MEMO_STATS=1`.
    pub(super) decode_memo_primed_count: u64,
    pub(super) decode_memo_rebuilt_count: u64,
    pub(super) decode_memo_replayed_count: u64,
    /// Steps that routed through the memo path but were structurally ineligible
    /// (memo OFF, CUDA, nested, or non-eager) — counted only when the master
    /// switch is on, to make an over-restrictive gate observable.
    pub(super) decode_memo_ineligible_count: u64,
    /// F5 Stage 2 — cached invariant buffer/view plan. Present only after a
    /// successful memo rebuild that found ≥1 fully-invariant pure-view node; it
    /// records the zero-copy view aliases to reinstate and the pure-view plan
    /// nodes to elide on a matching replay, guarded by a per-source buffer
    /// identity signature. Invalidated on every non-replay step (mirrors the
    /// Stage-1 Chew defense-in-depth) so a stale plan from a retired/errored
    /// step can never serve a future replay. Default ON (shares the Stage-1
    /// `ONNX_GENAI_DECODE_MEMO` gate; set =0 to disable).
    pub(super) decode_view_plan: Option<DecodeViewPlan>,
    /// F5 Stage 2 counters. `views_reused` = zero-copy view aliases reinstated
    /// without rebuild; `dispatch_elided` = pure-view plan nodes whose re-dispatch
    /// was skipped. Both prove non-vacuous firing on the real decode path.
    pub(super) decode_views_reused_count: u64,
    pub(super) decode_dispatch_elided_count: u64,
    /// F5 Stage 2 defense-in-depth: consecutive replay steps whose buffer-identity
    /// signature failed to match (a source buffer moved/resized under a plan that
    /// classified it invariant). After [`STAGE2_SIG_MISMATCH_LIMIT`] such steps the
    /// view plan is disabled for the rest of the session — an invariant-buffer
    /// assumption that keeps breaking must never keep serving cached views.
    pub(super) decode_view_plan_sig_mismatch_streak: u32,
    /// Latched off after repeated signature mismatches (see above).
    pub(super) decode_view_plan_disabled: bool,
    /// Master switch for graph-level compute-in-place aliasing.
    pub(super) compute_in_place_enabled: bool,
    /// Release an intermediate value's buffer once its last consumer has run.
    ///
    /// Opt-in per session, and default OFF, because freeing lets a later run
    /// place the same value at a different address. A session that records a
    /// device graph needs its buffer addresses to stay put between the recording
    /// run and every replay, so capture-eligible sessions (the decoder) leave
    /// this alone. Prompt-phase component graphs never capture and are where the
    /// resident set actually hurts, so they turn it on.
    pub(super) release_dead_values_enabled: bool,
    /// Successful dead-input buffer aliases, retained for parity/safety tests.
    pub(super) compute_in_place_alias_count: u64,
    /// Opt-in (default OFF) master switch for the single-trip `Scan` inline
    /// dual-path (`ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP`). When ON, a `Scan` whose
    /// runtime scan-axis length is exactly 1 (a single decode step) runs its body
    /// once straight-line instead of the generic `exec_scan` loop; any other
    /// trip count — including prefill at `prompt_len > 1` — keeps the unchanged
    /// loop. The selection is at RUNTIME, keyed on the observed trip count, never
    /// baked into the graph: prefill and decode share one executor/plan, so a
    /// static single-trip rewrite would corrupt prefill. Flag OFF ⇒ every trip
    /// count uses the loop (zero behavior change). Slice 1a is host-execution
    /// only; it does not interact with device-graph capture.
    pub(super) scan_inline_single_trip_enabled: bool,
    /// Diagnostic: how many times the single-trip `Scan` inline path actually
    /// engaged over this executor's lifetime. `> 0` after a decode run proves the
    /// dual-path is non-vacuously firing (an on-model A/B and the CUDA-gated
    /// regression test read it to reject a silently-gated-out pass); it stays 0
    /// whenever the flag is OFF or every `Scan` runs at `trip_count != 1`.
    pub(super) scan_inline_single_trip_count: u64,
    /// Per-plan-node kernel pre-binding (Stage 3). Each slot stores the
    /// [`KernelKey`] from the most recent successful kernel lookup for that plan
    /// node. On subsequent dispatch, if the current input shapes match the stored
    /// key's shapes, the kernel is retrieved via `get_prebound` — a single
    /// `HashMap::get` with no allocation (the key is already owned). This
    /// eliminates the 2.15 µs/op dispatch tax (shape-vec allocation + hash) in
    /// steady-state decode.
    ///
    /// Populated lazily: `None` until the first successful dispatch of that node.
    /// Invalidated (replaced) when shapes change (prefill→decode transition).
    /// Control-flow and sequence nodes always have `None` (they don't use the
    /// kernel cache).
    pub(super) kernel_bindings: Vec<Option<KernelKey>>,
    pub(super) persistent_workspace: Option<PreparedWorkspace>,
    pub(super) step_workspace: Option<PreparedWorkspace>,
    /// When set, [`Executor::release_step_workspace`] is a no-op: the StepScoped
    /// `step_workspace` buffer is kept alive between runs instead of being freed
    /// after each one. A captured device graph bakes the physical address of the
    /// StepScoped scratch it reads; if that buffer is freed and re-reserved
    /// between the capture and a later replay (as happens once a larger M=K
    /// speculative verify forward reserves a bigger scratch than the M=1 decode),
    /// the replay reads a stale pointer and produces non-finite logits (#1647).
    /// Pinning the workspace at the M=K peak keeps the pointer stable across
    /// replays. Inert by default (`false`) so every non-verify executor frees its
    /// StepScoped scratch exactly as before.
    pub(super) pin_step_workspace: bool,
    /// Non-owning view of an enclosing executor's prepared workspace. Nested
    /// control-flow executors run sequentially, so they may reuse the parent's
    /// peak allocation without reserving or allocating a second buffer.
    pub(super) inherited_workspace: Option<(usize, usize)>,
    pub(super) workspace_preparation_required: bool,
}

pub(super) struct PreparedWorkspace {
    pub(super) buffer: WorkspaceAllocation,
    pub(super) bytes: usize,
    pub(super) alignment: usize,
}

/// After this many consecutive buffer-identity signature mismatches, F5 Stage 2
/// view reuse is latched off for the session (Chew defense-in-depth).
pub(super) const STAGE2_SIG_MISMATCH_LIMIT: u32 = 2;

/// Run-scoped metadata for a zero-copy view value: it owns no buffer but
/// borrows `source`'s buffer with the given (real, possibly non-contiguous or
/// negative-strided) geometry. `strides`/`byte_offset` are expressed relative
/// to `source`'s allocation base, so a view-of-a-view is flattened to a single
/// hop whose `source` is always a real buffer owner (never itself a view).
#[derive(Clone, Debug)]
pub(super) struct ValueView {
    pub(super) source: ValueId,
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<i64>,
    pub(super) byte_offset: usize,
}

/// F5 Stage 1 — steady-state decode-plan memo.
///
/// [`Executor::resolve_soft`] is a **pure function of the current symbol
/// `bindings`** (see [`substitute`]): a value's resolved shape depends only on
/// its interned [`Shape`] and the bindings, and [`Executor::bind_symbols`]
/// derives bindings purely from the input *shapes*. During steady-state
/// single-token (M=1) decode only a small set of length symbols changes each
/// step, so every value whose shape references no such symbol resolves to a
/// byte-identical shape every step. This memo caches that length-invariant
/// partition and, on a plan-matching step, re-substitutes only the small
/// length-varying tail — avoiding a full ~600-entry map rebuild per token.
///
/// **Soundness (why a wrong shape can never be replayed).** A step may replay
/// the invariant partition iff every symbol the memo did *not* classify as
/// varying has the same binding it was built under and the bound-symbol set is
/// identical ([`DecodePlanMemo::matches`]). Because each invariant shape
/// references only static dims and non-varying symbols, an unchanged
/// non-varying binding set guarantees it re-substitutes to the identical value —
/// so the replayed map is byte-identical to a fresh `resolve_soft`. Crucially,
/// if a symbol that *actually* varies were mis-classified invariant, its next
/// change is a change to a **non-varying** binding ⇒ `matches` fails ⇒ the memo
/// rebuilds; a stale shape is therefore never emitted, regardless of how
/// `decode_varying` was derived. The variant tail is always re-substituted from
/// the fresh bindings, never replayed. A debug/opt-in full re-resolve
/// ([`Executor::decode_memo_verify`]) asserts equality every replay (R1 net).
pub(super) struct DecodePlanMemo {
    /// Bindings the invariant partition was built under — the replay guard.
    pub(super) reference_bindings: HashMap<SymbolId, usize>,
    /// Symbols observed to change value between two consecutive real eager
    /// steps (R1: derived by diffing, never guessed).
    pub(super) decode_varying: HashSet<SymbolId>,
    /// Resolved shape of every value whose [`Shape`] references no varying
    /// symbol — replayed verbatim.
    pub(super) invariant_shapes: HashMap<ValueId, Vec<usize>>,
    /// Values whose [`Shape`] references ≥1 varying symbol — re-substituted from
    /// the fresh bindings on every replay step.
    pub(super) variant_values: Vec<ValueId>,
    /// All value ids the memo owns (`invariant_shapes` keys ∪ `variant_values`) —
    /// i.e. exactly the keys `resolve_soft` produces for this regime. Used to
    /// strip the previous step's just-in-time (data-dependent) entries from the
    /// persistent working map before replay, so the run loop recomputes them.
    pub(super) canonical: HashSet<ValueId>,
    /// L-abstracted structural fingerprint of the persistent device-I/O binding
    /// set the memo was built under. Pure-L KV growth leaves this unchanged (so
    /// the step replays); a binding appearing/disappearing, a role flip, or a
    /// dtype change alters it and forces a rebuild. See [`DecodeBindingSig`].
    pub(super) reference_external_sig: Vec<DecodeBindingSig>,
}

/// L-abstracted structural fingerprint of one persistent (device-bound) I/O
/// binding, for the decode-plan memo replay guard.
///
/// Unlike [`ExternalCaptureSig`] — which is pointer/capacity- and concrete-shape-
/// exact for CUDA capture seeding — this abstracts the growing length symbol `L`
/// to its symbolic identity by fingerprinting the binding's *declared* (symbolic)
/// shape (`value_shapes[vid]`, which is graph-static). Two decode steps that
/// differ only by KV length therefore compare **equal** and replay, while a
/// structural change (a binding added/removed, an input/output role flip, or a
/// dtype change) compares unequal and forces a rebuild. Pointer and byte capacity
/// are deliberately **excluded**: Stage 1 memoizes shape resolution only (buffers
/// are re-sized every step outside the memo), so a KV-cache realloc must not
/// invalidate the plan — including `ptr`/`len` here would force a rebuild on every
/// growth-driven reallocation and leave the memo perpetually dead on the real
/// decode path.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct DecodeBindingSig {
    pub(super) vid: ValueId,
    pub(super) is_input: bool,
    pub(super) dtype: DataType,
    pub(super) decl_shape: Shape,
}

impl DecodePlanMemo {
    /// A step is a plan-match (may replay) iff it binds exactly the same symbol
    /// set as the reference and agrees with it on every non-varying symbol (only
    /// the varying length / past-length symbols may differ) **and** presents the
    /// same L-abstracted persistent-binding signature.
    pub(super) fn matches(
        &self,
        bindings: &HashMap<SymbolId, usize>,
        external_sig: &[DecodeBindingSig],
    ) -> bool {
        if external_sig != self.reference_external_sig {
            return false;
        }
        if bindings.len() != self.reference_bindings.len() {
            return false;
        }
        bindings.iter().all(|(sym, &val)| {
            match self.reference_bindings.get(sym) {
                Some(&ref_val) => val == ref_val || self.decode_varying.contains(sym),
                // A symbol the reference did not bind: the plan shape differs.
                None => false,
            }
        })
    }
}

/// F5 Stage 2 — cached invariant buffer/view plan.
///
/// Stage 1 proved that during steady single-token decode a large partition of
/// values resolve to a byte-identical shape every step (the memo's
/// `invariant_shapes`). Empirically, on real decoders the ~113 pure layout ops
/// (`Reshape`/`Squeeze`/`Unsqueeze`/no-op views) produce a **byte-identical
/// zero-copy [`ValueView`] every step** — yet Stage 1 still re-cleared and
/// re-dispatched every one per token. This plan caches those view aliases and the
/// nodes that produce them so a matching replay step can:
///   1. reinstate the invariant view aliases instead of clearing+rebuilding them,
///   2. exclude the invariant partition from per-step buffer sizing, and
///   3. elide re-dispatch of the pure-view nodes entirely.
///
/// **Membership (why an elided view is never geometrically stale).** A node is a
/// *candidate* iff every output's shape is in the memo's proven-invariant partition
/// (`invariant_shapes`) — so Stage 1 already guarantees the output shape is
/// byte-identical every replay step and the replayed `resolved` map always carries
/// it. A candidate is *promoted* to the active elision set only after its produced
/// view is observed **byte-identical across a second real decode step**
/// ([`Executor::validate_decode_view_plan`]) — the same two-real-step confirmation
/// Stage 1 uses to derive its varying set. Contiguous-view strides are a pure
/// function of the (invariant) output shape, and any per-step `byte_offset` drift
/// (e.g. a position-indexed slice into a fixed-capacity KV buffer) would differ
/// across the two observed steps and so is rejected before it can ever be elided.
///
/// **Soundness — the buffer-identity obligation.** Stage 1 could exclude
/// pointer/capacity from its replay key because it cached *shapes only* and every
/// kernel re-read fresh bytes each step. Stage 2 caches actual **buffers/views**, so
/// a realloc or pointer move of a cached view's source would leave the reinstated
/// alias pointing at a stale/dangling region. Therefore this plan records
/// `source_buffer_sig` = `(source, base_ptr, capacity)` for every buffer a retained
/// view aliases, and a replay step reinstates the plan **iff** every signature still
/// matches ([`Executor::stage2_buffer_sig_matches`]); any mismatch forces a full
/// rebuild. (A retained [`ValueView`] references its source by [`ValueId`], so a
/// consumer already re-reads the *current* base pointer — but the byte offset and
/// capacity assumptions are exactly what the pointer+capacity guard protects.)
///
/// The plan is only ever *built* at the successful end of a memo Rebuilt step,
/// *validated* on the following replay, and *used* on a memo Replayed step whose
/// bindings, external signature (Stage 1) and buffer identity (Stage 2) all match;
/// it is invalidated on every non-replay step so an errored/retired step can never
/// serve a stale alias. Under `decode_memo_verify` every reinstated view is also
/// asserted equal to a freshly built one in-flight (the R1 safety net).
pub(super) struct DecodeViewPlan {
    /// Plan-node indices (into [`Executor::plan`]) whose every output shape is in
    /// the memo's invariant partition — candidates until validated, then the active
    /// elision set (re-dispatch skipped on a matching replay).
    pub(super) elided_nodes: HashSet<usize>,
    /// The zero-copy view aliases to reinstate each replay step (`vid` → its
    /// invariant [`ValueView`]), verbatim from the reference step.
    pub(super) retained_views: Vec<(ValueId, ValueView)>,
    /// Distinct buffer-owning source value ids to re-pin (conservative liveness:
    /// a source with a live view is never reused/freed within the run).
    pub(super) pinned_sources: Vec<ValueId>,
    /// Buffer-identity signature `(source, base_ptr as usize, capacity)` for every
    /// retained view's source buffer. The Stage-2 replay guard.
    pub(super) source_buffer_sig: Vec<(ValueId, usize, usize)>,
    /// `false` for a freshly built candidate; set `true` once every retained view
    /// has been confirmed byte-identical on a second real decode step. Only a
    /// validated plan is ever used to elide dispatch.
    pub(super) validated: bool,
}

/// Outcome of the most recent memo-eligible eager resolve, exposed for the F5
/// guard tests to distinguish a rebuild from a replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DecodeMemoAction {
    /// The memo was disabled, or the step was not memo-eligible.
    Disabled,
    /// First observation of a regime: bindings recorded, no memo built yet
    /// (the two-real-step derivation needs a second matching step).
    Primed,
    /// A full resolve whose result (re)built the memo by diffing this step with
    /// the previous eligible step.
    Rebuilt,
    /// The invariant partition was replayed and only the variant tail
    /// re-substituted.
    Replayed,
}

/// True iff two binding maps bind exactly the same symbol set.
pub(super) fn same_symbol_keys(a: &HashMap<SymbolId, usize>, b: &HashMap<SymbolId, usize>) -> bool {
    a.len() == b.len() && a.keys().all(|k| b.contains_key(k))
}

/// M==1 single-token-decode gate (residual #3): admit a memo (re)build only for
/// a steady autoregressive-decode transition, where sequence/KV length symbols
/// only ever *grow*. `prev`→`cur` qualifies iff both bind the same symbol set,
/// at least one symbol increased, and **no** symbol decreased. This excludes the
/// prefill→decode transition (the query-length symbol drops from the prompt
/// length P to 1) and any non-decode reshape, so the memo activates only on
/// single-token decode — not prefill — tightening the blast radius. Soundness
/// does not rely on this gate (the `matches` guard is the correctness invariant);
/// it only decides *when* the memo is worth building.
pub(super) fn is_decode_growth_transition(
    prev: &HashMap<SymbolId, usize>,
    cur: &HashMap<SymbolId, usize>,
) -> bool {
    if !same_symbol_keys(prev, cur) {
        return false;
    }
    let mut any_grew = false;
    for (sym, &c) in cur {
        let p = prev[sym];
        if c > p {
            any_grew = true;
        } else if c < p {
            return false; // a shrinking extent is not steady decode (e.g. prefill→decode)
        }
    }
    any_grew
}

/// True iff `shape` references any symbol in `symbols`.
pub(super) fn shape_references_any(shape: &Shape, symbols: &HashSet<SymbolId>) -> bool {
    shape
        .iter()
        .any(|d| matches!(d, Dim::Symbolic(s) if symbols.contains(s)))
}

/// Whether the decode-plan memo master switch (`ONNX_GENAI_DECODE_MEMO`) is on.
/// Default ON; set `ONNX_GENAI_DECODE_MEMO=0` to disable.
///
/// The explicit OFF values are `0`, `false`, and `off` (case-insensitive,
/// surrounding whitespace trimmed). Every other state — unset, empty, or an
/// unrecognized value — enables the memo, so parsing fails safe toward the
/// validated fast path (worst case: rebuild every step, no speedup, never
/// wrong). Ripley's authoritative A/B recorded 0 token flips and a non-negative
/// speedup at every tested core count, so default-ON is token-exact by
/// construction.
pub(super) fn decode_memo_env_enabled() -> bool {
    match std::env::var("ONNX_GENAI_DECODE_MEMO") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

/// Whether graph-level compute-in-place aliasing is enabled. Default ON; setting
/// `ONNX_GENAI_COMPUTE_IN_PLACE=0` retains the fully out-of-place reference path.
pub(super) fn compute_in_place_env_enabled() -> bool {
    match std::env::var("ONNX_GENAI_COMPUTE_IN_PLACE") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

/// Whether the opt-in per-step replay verification (`ONNX_GENAI_DECODE_MEMO_VERIFY`)
/// is set. Always on under `debug_assertions`.
pub(super) fn decode_memo_verify_env_enabled() -> bool {
    matches!(
        std::env::var("ONNX_GENAI_DECODE_MEMO_VERIFY")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// Whether the single-trip `Scan` inline dual-path is enabled
/// (`ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP`). Default OFF: this is an opt-in
/// correctness-foundation path, so it engages only on an explicit ON value
/// (`1`/`true`/`on`, case-insensitive, whitespace-trimmed). Every other state —
/// unset, empty, `0`, or unrecognized — leaves it OFF, so the executor keeps
/// running the unchanged `exec_scan` loop for every trip count.
pub(super) fn scan_inline_single_trip_env_enabled() -> bool {
    matches!(
        std::env::var("ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// Per-input geometry the run loop resolves once per node: the raw base pointer
/// of the backing (root) buffer plus the real view (shape, element strides —
/// possibly non-contiguous or negative — and byte offset) to read it through.
/// A plain owned value yields contiguous strides at offset 0; a view value
/// yields its recorded strides/offset over its source buffer. `present` is false
/// for an omitted optional input (an absent placeholder).
pub(super) struct InInfo {
    pub(super) present: bool,
    pub(super) dtype: DataType,
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<i64>,
    pub(super) byte_offset: usize,
    pub(super) base_ptr: usize,
    pub(super) device: onnx_runtime_ir::DeviceId,
    pub(super) backing: TensorBacking,
    /// Length in bytes of the backing (root) allocation, for the bounds gate.
    pub(super) root_len: usize,
    /// True for a lazy weight the EP declined to page into device memory (offload
    /// disabled or no residency); such inputs stay absent and are routed to the
    /// kernel as a lazy `KernelInput::Weight` instead of a bound view.
    pub(super) lazy_unresolved: bool,
    /// True when the ordinary initializer buffer was deliberately omitted
    /// because the provider promised an immutable prepared override.
    pub(super) prepared_unresolved: bool,
}

#[derive(Clone)]
pub(super) struct ExternalValue {
    pub(super) dtype: DataType,
    pub(super) shape: Vec<usize>,
    pub(super) accepts_subshape: bool,
    pub(super) ptr: usize,
    pub(super) len: usize,
    pub(super) alignment: usize,
    pub(super) device: onnx_runtime_ir::DeviceId,
}

impl ExternalValue {
    pub(super) fn accepts_output(&self, dtype: DataType, shape: &[usize], bytes: usize) -> bool {
        self.dtype == dtype
            && self.len >= bytes
            && if self.accepts_subshape {
                shape.len() == self.shape.len()
                    && shape
                        .iter()
                        .zip(&self.shape)
                        .all(|(&required, &capacity)| required <= capacity)
            } else {
                self.shape == shape
            }
    }

    pub(super) fn writable_buffer(&self) -> Result<DeviceBuffer> {
        // SAFETY: `prepare_external_bindings` obtains this pointer from a live
        // `DeviceIoBinding` exclusively borrowed for the run. The binding owns
        // the allocation, outlives this alias, and is not otherwise accessed
        // until execution returns.
        unsafe {
            DeviceBuffer::from_borrowed_mut_parts(
                self.ptr as *mut std::ffi::c_void,
                self.device,
                self.len,
                self.alignment,
            )
        }
        .ok_or_else(|| SessionError::Internal("external output binding has a null pointer".into()))
    }

    pub(super) fn readable_buffer(&self) -> Result<DeviceBuffer> {
        if self.ptr == 0 {
            return Err(SessionError::Internal(
                "external input binding has a null pointer".into(),
            ));
        }
        // SAFETY: `prepare_external_bindings` obtains this pointer from a live
        // `DeviceIoBinding` borrowed for the complete run. This read-only alias
        // neither owns nor mutates the binding's allocation.
        Ok(unsafe {
            DeviceBuffer::from_borrowed_parts(
                self.ptr as *mut std::ffi::c_void,
                self.device,
                self.len,
                self.alignment,
            )
        })
    }
}

#[derive(Default)]
pub(super) struct ExternalBindings {
    pub(super) inputs: HashMap<ValueId, ExternalValue>,
    pub(super) outputs: HashMap<ValueId, ExternalValue>,
}

/// One persistent (device-bound) I/O binding's identity for capture: its value,
/// role, dtype, kernel-visible shape, backing device pointer and byte capacity.
/// The full set forms the *decode binding signature* under which a warm eager
/// resolution's just-in-time shapes are trusted for capture-mode seeding: a
/// change to any pointer or shape means the warm geometry may be stale, so the
/// seed is withheld (nodes stay eager) rather than baked into a captured graph.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct ExternalCaptureSig {
    pub(super) vid: ValueId,
    pub(super) is_input: bool,
    pub(super) dtype: DataType,
    pub(super) shape: Vec<usize>,
    pub(super) ptr: usize,
    pub(super) len: usize,
}

impl ExternalBindings {
    pub(super) fn seed_capture_shapes(&self, resolved: &mut HashMap<ValueId, Vec<usize>>) {
        for (&vid, value) in self.inputs.iter().chain(&self.outputs) {
            resolved.entry(vid).or_insert_with(|| value.shape.clone());
        }
    }

    /// Order-independent signature of every persistent binding (pointer, byte
    /// capacity and kernel-visible shape). Two runs whose signatures compare
    /// equal present pointer- and capacity-stable buffers, which is the exact
    /// precondition for trusting a prior eager run's just-in-time shapes.
    pub(super) fn capture_signature(&self) -> Vec<ExternalCaptureSig> {
        let mut sig: Vec<ExternalCaptureSig> = self
            .inputs
            .iter()
            .map(|(&vid, v)| (vid, true, v))
            .chain(self.outputs.iter().map(|(&vid, v)| (vid, false, v)))
            .map(|(vid, is_input, v)| ExternalCaptureSig {
                vid,
                is_input,
                dtype: v.dtype,
                shape: v.shape.clone(),
                ptr: v.ptr,
                len: v.len,
            })
            .collect();
        sig.sort_by_key(|a| (a.vid.0, a.is_input));
        sig
    }

    pub(super) fn refill_capture_signature(&self, sig: &mut Vec<ExternalCaptureSig>) {
        sig.retain(|entry| {
            if entry.is_input {
                self.inputs.contains_key(&entry.vid)
            } else {
                self.outputs.contains_key(&entry.vid)
            }
        });
        for (vid, is_input, value) in self
            .inputs
            .iter()
            .map(|(&vid, value)| (vid, true, value))
            .chain(self.outputs.iter().map(|(&vid, value)| (vid, false, value)))
        {
            if let Some(entry) = sig
                .iter_mut()
                .find(|entry| entry.vid == vid && entry.is_input == is_input)
            {
                entry.dtype = value.dtype;
                entry.shape.clear();
                entry.shape.extend_from_slice(&value.shape);
                entry.ptr = value.ptr;
                entry.len = value.len;
            } else {
                sig.push(ExternalCaptureSig {
                    vid,
                    is_input,
                    dtype: value.dtype,
                    shape: value.shape.clone(),
                    ptr: value.ptr,
                    len: value.len,
                });
            }
        }
        sig.sort_by_key(|entry| (entry.vid.0, entry.is_input));
    }
}

/// Concrete child plan cached for one external-input dtype/shape signature.
pub(super) struct CompiledChildPlan {
    pub(super) exec: Executor,
    pub(super) signature: Vec<ChildInputSignature>,
}

/// Control-flow bodies commonly alternate among a handful of stable shapes.
/// Four entries cover those cases without retaining an unbounded set of plans.
pub(super) const CHILD_EXECUTOR_CACHE_CAPACITY: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ChildInputSignature {
    pub(super) dtype: DataType,
    pub(super) shape: Vec<usize>,
}

/// A reusable executor for one nested graph body.
///
/// The body signature and lexical-capture set are resolved once at construction.
/// Concrete [`Executor`]s are compiled lazily and retained in a small,
/// deterministic LRU keyed by external-input dtype/shapes, so alternating
/// Loop/Scan/If signatures reuse prior plans instead of recompiling each switch.
pub(crate) struct ChildExecutor {
    pub(super) name: String,
    pub(super) template: Graph,
    pub(super) inherited_opsets: HashMap<String, u64>,
    pub(super) weights: Arc<WeightStore>,
    pub(super) ep: Arc<dyn ExecutionProvider>,
    pub(super) formal_names: Vec<String>,
    pub(super) capture_names: Vec<String>,
    pub(super) input_names: Vec<String>,
    pub(super) compiled: Vec<CompiledChildPlan>,
    pub(super) builds: u64,
    pub(super) runs: u64,
    /// Shared trace context, propagated to every compiled child plan's executor.
    pub(super) trace: TraceContext,
    /// Dead-value release setting, propagated to every compiled child plan's
    /// executor so a Scan/Loop body inherits the parent session's choice.
    pub(super) release_dead_values: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ChildExecutorStats {
    pub builds: u64,
    pub runs: u64,
}

/// Invocation-invariant binding metadata for one selected subgraph. Loop/Scan
/// prepare this once outside the iteration loop, including one-time capture
/// materialization, then only rebind the changing formal tensors each step.
pub(super) struct PreparedSubgraph {
    pub(super) key: (NodeId, String),
    /// Direct captures plus transitive captures needed only by nested bodies.
    pub(super) captures: HashMap<String, Tensor>,
}
