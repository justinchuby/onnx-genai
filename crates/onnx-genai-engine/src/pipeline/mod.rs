//! Universal metadata-declared workflow runtime.

use crate::decode::clone_value;
use crate::engine::{
    Engine, EngineConfig, EngineResourceGovernor, Holder, MemoryStrategyPlanInput,
    analyze_model_memory, build_memory_strategy_plan, combine_graph_memory, component_governor,
    log_memory_strategy_plan, requested_decode_backend, resolve_memory_strategy_hot_tier_bytes,
    resolve_vram_limit_bytes,
};
use crate::memory_authority::{MemoryAuthorityProvider, SharedMemoryAuthorityProvider};
use crate::{
    EngineDecodeBackend, FinishReason, GeneratePrompt, GenerateRequest, GenerateResult,
    GenerateToken, GenerateTokenCallback, MemoryStrategyPlan, TokenId,
};
use anyhow::Context;
use onnx_genai_metadata::{
    CompiledWorkflow, ComponentImplementation, DeviceKind, LiteralValue, PreprocessingSpec,
    RuntimeInputRole, ScalarValue, TensorContract, TensorDimension, WorkflowEmitMode,
    WorkflowInputSource, WorkflowNode, WorkflowSpec,
};
use onnx_genai_ort::{
    DataType, PipelineModelDirectory, PipelineModels, SessionOptions, Tokenizer, Value,
};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

mod adapters;
mod arg_reduce;
mod islands;
#[cfg(feature = "native-backend")]
mod native_component;
mod row_state;
mod workflow;

pub use adapters::{AdapterActivation, AdapterLifecycleDiagnostic, AdapterSelection};
pub use arg_reduce::{ArgReduceRewrites, WideArgReduceLowering, lower_degenerate_arg_reductions};
pub use islands::ExecutionIslandDiagnostic;
pub use onnx_genai_metadata::WorkflowOutputRole;
pub use row_state::{RowScopedState, RowTable, check_selection, gather_rows};
pub use workflow::{
    MISSING_REQUIRED_INPUT, WorkflowExecutionPlan, WorkflowPerformanceDiagnostic,
    is_missing_required_input,
};

pub type PipelineTensors = HashMap<String, Value>;

/// Structured workflow outputs with request-aligned rows.
///
/// Rows are positional: row `i` of a request-aligned output belongs to batch
/// row `i`. The runtime associates a batch row with a request through its own
/// private request table, so no scheduler identity is serialized here.
pub struct PipelineOutputs {
    tensors: PipelineTensors,
    rows: BTreeMap<String, Vec<String>>,
}

impl PipelineOutputs {
    pub fn tensors(&self) -> &PipelineTensors {
        &self.tensors
    }

    pub fn into_tensors(self) -> PipelineTensors {
        self.tensors
    }

    pub fn aggregate(&self, output: &str) -> Option<&Value> {
        self.tensors.get(output)
    }

    /// Request-aligned rows of one output, in batch-row order.
    pub fn rows(&self, output: &str) -> Vec<(usize, &Value)> {
        self.rows
            .get(output)
            .into_iter()
            .flatten()
            .enumerate()
            .filter_map(|(row, name)| self.tensors.get(name).map(|value| (row, value)))
            .collect()
    }
}

impl std::ops::Deref for PipelineOutputs {
    type Target = PipelineTensors;

    fn deref(&self) -> &Self::Target {
        &self.tensors
    }
}

/// Replayable snapshot of semantic session-scoped workflow state.
pub struct WorkflowSessionCheckpoint {
    semantic_state: HashMap<String, Value>,
    contracts: HashMap<String, TensorContract>,
}

/// A request for the universal workflow interpreter.
pub struct PipelineGenerateRequest {
    pub request: GenerateRequest,
    /// Application tensors keyed by a declared package input or application source name.
    pub inputs: PipelineTensors,
    /// Identity used by session-scoped workflow state cells.
    pub session_id: Option<String>,
    /// Application-selected package components that replace overridable components.
    pub component_overrides: HashMap<String, String>,
}

impl PipelineGenerateRequest {
    pub fn new(request: GenerateRequest) -> Self {
        Self {
            request,
            inputs: HashMap::new(),
            session_id: None,
            component_overrides: HashMap::new(),
        }
    }

    pub fn with_input(mut self, name: impl Into<String>, value: Value) -> Self {
        self.inputs.insert(name.into(), value);
        self
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_component_override(
        mut self,
        component: impl Into<String>,
        replacement: impl Into<String>,
    ) -> Self {
        self.component_overrides
            .insert(component.into(), replacement.into());
        self
    }
}

impl From<GenerateRequest> for PipelineGenerateRequest {
    fn from(request: GenerateRequest) -> Self {
        Self::new(request)
    }
}

/// Engine for packages expressed exclusively with `pipeline.workflow`.
pub struct PipelineEngine {
    package_root: std::path::PathBuf,
    models: PipelineModels,
    resource_governor: EngineResourceGovernor,
    memory_strategy_plan: MemoryStrategyPlan,
    decode_backend: EngineDecodeBackend,
    workflow: WorkflowSpec,
    compiled_workflow: CompiledWorkflow,
    /// Outputs this workflow fills one request row at a time, derived once from
    /// the compiled graph so every emit into one output agrees.
    row_wise_outputs: HashSet<String>,
    movable_emit_values: HashSet<String>,
    execution_islands: Vec<islands::ExecutionIsland>,
    device_bridge_components: HashSet<String>,
    component_bindings:
        RefCell<HashMap<workflow::ComponentBindingKey, workflow::StableComponentBinding>>,
    component_allocators: RefCell<HashMap<String, Arc<onnx_genai_ort::Allocator>>>,
    component_outputs: RefCell<HashMap<workflow::ComponentOutputKey, Arc<onnx_genai_ort::Value>>>,
    workflow_performance: RefCell<workflow::WorkflowPerformanceCounters>,
    workflow_execution_generation: Cell<u64>,
    workflow_session_state: RefCell<HashMap<(String, String), Value>>,
    adapter_service: Option<onnx_genai_metadata::AdapterServiceContract>,
    adapter_cache: RefCell<adapters::AdapterCache>,
    active_adapter_context: RefCell<Option<adapters::AdapterRunContext>>,
    preprocessing: Option<PreprocessingSpec>,
    /// Native (pure-Rust) component sessions, present only when the engine was
    /// built for `EngineDecodeBackend::Native`. The universal interpreter drives
    /// these through the same seam it uses for ORT sessions; see
    /// `docs/architecture/NATIVE_WORKFLOW_BACKEND.md`.
    #[cfg(feature = "native-backend")]
    native_components: Option<RefCell<native_component::NativeComponentSet>>,
}

impl Drop for PipelineEngine {
    fn drop(&mut self) {
        for island in &mut self.execution_islands {
            island.clear_bindings();
        }
        self.component_bindings.get_mut().clear();
        self.component_outputs.get_mut().clear();
        self.component_allocators.get_mut().clear();
    }
}

/// Validate an explicit pipeline backend request without touching model files.
///
/// The universal workflow interpreter is backend-neutral: it drives every
/// declared component through a seam that either an ORT `Session` or a native
/// `InferenceSession` fulfills (see `docs/architecture/NATIVE_WORKFLOW_BACKEND.md`).
/// `EngineDecodeBackend::Native` is therefore accepted whenever the native
/// backend is compiled in. A build without the `native-backend` feature has no
/// native sessions to run, so a native request fails closed here (Rule 4: no
/// silent ORT fallback) rather than pretending to honor it.
pub fn validate_pipeline_backend_request(
    requested: EngineDecodeBackend,
) -> anyhow::Result<EngineDecodeBackend> {
    let backend = requested_decode_backend(requested)?;
    #[cfg(not(feature = "native-backend"))]
    if backend == EngineDecodeBackend::Native {
        anyhow::bail!(
            "pipeline.workflow native execution requires the `native-backend` feature, but this \
             build was compiled without it. Rebuild with --features native-backend (or \
             --features native-cuda for the native CUDA EP), or select the ORT backend \
             (decode_backend = EngineDecodeBackend::Ort / ONNX_GENAI_BACKEND=ort)."
        );
    }
    Ok(backend)
}

fn workflow_initializer_reservation_bytes(
    source_session_bytes: u64,
    linked_session_initializer_bytes: u64,
    runtime_managed: bool,
) -> anyhow::Result<u64> {
    let maximum_residency = source_session_bytes
        .checked_add(linked_session_initializer_bytes)
        .context("workflow initializer reservation size overflow")?;
    Ok(if runtime_managed {
        0
    } else {
        maximum_residency
    })
}

impl Engine {
    pub fn from_pipeline_dir(
        pipeline_dir: &Path,
        config: EngineConfig,
    ) -> anyhow::Result<PipelineEngine> {
        PipelineEngine::from_dir_with_config(pipeline_dir, config)
    }

    pub fn from_pipeline_dir_with_memory_authority_provider(
        pipeline_dir: &Path,
        config: EngineConfig,
        provider: Arc<dyn MemoryAuthorityProvider>,
    ) -> anyhow::Result<PipelineEngine> {
        PipelineEngine::from_dir_with_memory_authority_provider(pipeline_dir, config, provider)
    }
}

impl PipelineEngine {
    pub fn from_dir(pipeline_dir: &Path) -> anyhow::Result<Self> {
        Self::from_dir_with_config(pipeline_dir, EngineConfig::default())
    }

    pub fn from_dir_with_config(pipeline_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        Self::build(pipeline_dir, config, SessionOptions::default(), None)
    }

    pub fn from_dir_with_session_options(
        pipeline_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
    ) -> anyhow::Result<Self> {
        Self::build(pipeline_dir, config, session_options, None)
    }

    pub fn from_dir_with_memory_authority_provider(
        pipeline_dir: &Path,
        config: EngineConfig,
        provider: Arc<dyn MemoryAuthorityProvider>,
    ) -> anyhow::Result<Self> {
        Self::build(
            pipeline_dir,
            config,
            SessionOptions::default(),
            Some(provider),
        )
    }

    fn build(
        pipeline_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
        authority_provider: Option<SharedMemoryAuthorityProvider>,
    ) -> anyhow::Result<Self> {
        let decode_backend = validate_pipeline_backend_request(config.decode_backend)?;
        let authority_domain = crate::engine::session_device_domain(&session_options)?;
        crate::engine::validate_shared_authority_limit(
            authority_provider.as_ref(),
            &authority_domain,
            config.limits.vram_limit,
        )?;
        let directory = PipelineModelDirectory::load(pipeline_dir)
            .map_err(|error| anyhow::anyhow!("Failed to resolve workflow package: {error}"))?;
        for (component, declaration) in &directory.spec.workflow.components {
            if let ComponentImplementation::Adapter { abi, version, .. } =
                &declaration.implementation
                && !workflow::supports_workflow_adapter(abi, version)
            {
                anyhow::bail!(
                    "workflow adapter '{component}' requires unsupported ABI {abi}@{version}"
                );
            }
        }

        let model_weights_bytes =
            directory
                .model_paths
                .values()
                .try_fold(0_u64, |total, path| {
                    total
                        .checked_add(onnx_genai_ort::model_weight_bytes(path))
                        .context("workflow component weight size overflow")
                })?;
        let graph_memory = combine_graph_memory(
            directory
                .model_paths
                .values()
                .map(|path| analyze_model_memory(path)),
            false,
        );
        let minimum_useful_weight_budget_bytes = graph_memory
            .per_layer_weight_bytes
            .iter()
            .map(|layer| layer.bytes)
            .max()
            .unwrap_or(0);
        // The workflow runtime executes every component through ORT sessions
        // (the native decoder backend is rejected above), so there is no native
        // CUDA ordinal to resolve the VRAM fraction against. The device (VRAM)
        // capacity stays honestly `None` when it cannot be measured (#947): it
        // is reported verbatim as `resolved_device_budget` and never borrows the
        // host tier. The residency verdict is a separate fact, sized against the
        // measured host-RAM ceiling, so a fitting model reads `FullResident`
        // instead of `Unknown` without fabricating a device number.
        let resolved_vram_bytes = resolve_vram_limit_bytes(&config.limits, None)?;
        let residency_ceiling_bytes = resolve_memory_strategy_hot_tier_bytes(&config.limits, None)?;
        #[cfg(feature = "native-cuda")]
        let memory_strategy_overrides = crate::engine::memory_strategy_overrides_from_cuda_env(
            onnx_runtime_ep_cuda::DeviceOffloadPolicy::from_env(),
        );
        #[cfg(not(feature = "native-cuda"))]
        let memory_strategy_overrides = crate::engine::MemoryStrategyOverrides::default();
        let managed_vmm = matches!(config.limits.vram_limit, crate::ResourceLimit::Bytes(_));
        let memory_strategy_plan = build_memory_strategy_plan(MemoryStrategyPlanInput {
            config: &config,
            resolved_vram_bytes,
            residency_ceiling_bytes,
            model_weight_bytes: model_weights_bytes,
            // #971: the resident dequantised f32 MatMulNBits cache is a native
            // CPU EP kernel behaviour. Workflow components run on ORT sessions,
            // which use their own kernels, so there is no expansion here.
            resident_f32_cache_bytes: 0,
            kv_config: crate::engine::governor_no_paged_kv_config(&config)?,
            graph: graph_memory,
            required_device_non_weight_bytes: 0,
            minimum_useful_weight_budget_bytes,
            #[cfg(feature = "native-cuda")]
            default_dynamic_device_budget_bytes: Some(
                onnx_runtime_ep_cuda::DEFAULT_DEVICE_OFFLOAD_BUDGET_BYTES,
            ),
            #[cfg(not(feature = "native-cuda"))]
            default_dynamic_device_budget_bytes: None,
            inferred_policy_enabled: managed_vmm,
            managed_vmm,
            overrides: memory_strategy_overrides,
            advisory_only: true,
            // #864: WDDM shared-memory fallback is a Windows platform property.
            shared_memory_weight_fallback: cfg!(windows),
            force_managed_weight_streaming: crate::engine::force_managed_weight_streaming_enabled(),
        });
        log_memory_strategy_plan(&memory_strategy_plan, "workflow");
        let runtime_manages_initializer_residency = !memory_strategy_plan.advisory_only
            && memory_strategy_plan.runtime_application().managed_no_spill;
        let source_initializer_reservation = workflow_initializer_reservation_bytes(
            model_weights_bytes,
            0,
            runtime_manages_initializer_residency,
        )?;
        // Reserve every component's package bytes before constructing the first
        // session. CPU and ordinary ORT-CUDA sessions both own their initializer
        // residency, so both need the fixed claim. A non-advisory managed VMM
        // charges committed initializer pages itself and must not be charged a
        // second time here.
        let resource_governor = component_governor(
            &config,
            None,
            model_weights_bytes,
            source_initializer_reservation,
            None,
            authority_provider.as_ref(),
            &authority_domain,
        )?;
        // CUDA Graph capture applies to stable linked execution islands. Enabling
        // it on every source component rejects valid setup/control-flow graphs
        // before the workflow planner can determine capture eligibility.
        let mut component_session_options = session_options.clone();
        component_session_options.graph_capture = false;
        // Under Native, every component executes on a native `InferenceSession`
        // (see `native_component`), so building an ORT `Session` for each one is
        // redundant, pulls in the ORT runtime unnecessarily, double-loads
        // weights, and misreports the execution provider — and a native-only
        // operator would make ORT reject the graph at load. Build ORT sessions
        // only for the ORT backend; under Native, load backend-neutral graph I/O
        // only (`PipelineModels::graph_io_metadata`) so the package's I/O
        // contract stays available without an ORT session.
        #[cfg(feature = "native-backend")]
        let native_device = if decode_backend == EngineDecodeBackend::Native {
            Some(crate::engine::resolve_native_decode_device(
                config.native_device.clone(),
                &session_options,
            )?)
        } else {
            None
        };
        let models = if decode_backend == EngineDecodeBackend::Native {
            PipelineModels::load_with_ort_session_filter(
                pipeline_dir,
                component_session_options,
                |_| false,
            )
        } else {
            PipelineModels::load_with_component_options(
                pipeline_dir,
                component_session_options,
                session_options,
            )
        }
        .map_err(|error| anyhow::anyhow!("Failed to load workflow components: {error}"))?;

        let workflow = directory.spec.workflow;
        let mut compiled_workflow = onnx_genai_metadata::compile_workflow(&workflow)
            .map_err(|error| anyhow::anyhow!("Failed to lower workflow metadata: {error}"))?;
        let row_wise_outputs = workflow::workflow_row_wise_outputs(&compiled_workflow.graph);
        let movable_emit_values =
            workflow::compile_movable_emit_values(&compiled_workflow.graph, &workflow);
        let aliasable_output_values =
            workflow::compile_aliasable_output_values(&compiled_workflow.graph);
        let bridge_graph = compiled_workflow.graph.clone();
        // Source sessions stay resident when fusion adds linked sessions. Dry-run
        // the same island linker before constructing any linked session and
        // reserve the exact initializer payload of every island that can reach
        // session creation. This is a topology-derived maximum: a component
        // invoked in two different islands contributes twice, while a candidate
        // the linker rejects contributes nothing.
        let maximum_island_initializer_bytes = islands::maximum_execution_island_initializer_bytes(
            &compiled_workflow.graph,
            &workflow,
            &models,
        )?;
        let maximum_initializer_reservation = workflow_initializer_reservation_bytes(
            model_weights_bytes,
            maximum_island_initializer_bytes,
            runtime_manages_initializer_residency,
        )?;
        let additional_initializer_reservation = maximum_initializer_reservation
            .checked_sub(source_initializer_reservation)
            .context("workflow initializer reservation accounting underflow")?;
        let island_reservation_admitted = match resource_governor.plan().reserve(
            Holder::FixedDeviceReservation,
            additional_initializer_reservation,
        ) {
            Ok(_) => true,
            Err(error) => {
                tracing::warn!(
                    additional_initializer_reservation,
                    %error,
                    "workflow execution-island fusion declined because duplicate initializer \
                     residency was not admitted"
                );
                false
            }
        };
        let execution_islands =
            if island_reservation_admitted && decode_backend != EngineDecodeBackend::Native {
                // Execution islands are the ORT `IoBinding` / CUDA-graph optimization.
                // The native backend has no equivalent yet (follow-up boundary A), so
                // under Native we keep the compiled graph's individual component nodes
                // and drive each through the native seam — correct, just unfused.
                islands::plan_execution_islands(
                    &mut compiled_workflow.graph,
                    &workflow,
                    &models,
                    &aliasable_output_values,
                )
                .map_err(|error| {
                    anyhow::anyhow!("Failed to plan workflow execution islands: {error}")
                })?
            } else {
                Vec::new()
            };
        let live_island_initializer_bytes =
            islands::execution_island_initializer_bytes(&execution_islands)?;
        let live_initializer_reservation = workflow_initializer_reservation_bytes(
            model_weights_bytes,
            live_island_initializer_bytes,
            runtime_manages_initializer_residency,
        )?;
        let unused_initializer_reservation = if island_reservation_admitted {
            maximum_initializer_reservation
                .checked_sub(live_initializer_reservation)
                .context("live workflow initializer residency exceeded its admitted maximum")?
        } else {
            0
        };
        let released = resource_governor.plan().release(
            Holder::FixedDeviceReservation,
            unused_initializer_reservation,
        );
        anyhow::ensure!(
            released == unused_initializer_reservation,
            "workflow initializer reservation release mismatch: requested \
             {unused_initializer_reservation} bytes, released {released}"
        );
        let island_components = execution_islands
            .iter()
            .flat_map(|island| island.components().iter().cloned())
            .collect::<HashSet<_>>();
        let device_bridge_components =
            workflow::compile_device_bridge_components(&bridge_graph, &island_components);
        // Load a native session per component only when the native backend was
        // requested, bound to the resolved native device/EP. The ORT
        // `PipelineModels` above holds only backend-neutral graph I/O in that
        // case (no ORT sessions were built).
        #[cfg(feature = "native-backend")]
        let native_components = if decode_backend == EngineDecodeBackend::Native {
            let device = native_device
                .as_ref()
                .expect("native device is resolved when the decode backend is Native");
            Some(RefCell::new(native_component::NativeComponentSet::load(
                &directory.model_paths,
                device,
            )?))
        } else {
            None
        };
        Ok(Self {
            package_root: directory.root.clone(),
            models,
            resource_governor,
            memory_strategy_plan,
            decode_backend,
            workflow,
            compiled_workflow,
            row_wise_outputs,
            movable_emit_values,
            execution_islands,
            device_bridge_components,
            component_bindings: RefCell::new(HashMap::new()),
            component_allocators: RefCell::new(HashMap::new()),
            component_outputs: RefCell::new(HashMap::new()),
            workflow_performance: RefCell::new(workflow::WorkflowPerformanceCounters::default()),
            workflow_execution_generation: Cell::new(0),
            workflow_session_state: RefCell::new(HashMap::new()),
            adapter_service: directory.adapters,
            adapter_cache: RefCell::new(adapters::AdapterCache::default()),
            active_adapter_context: RefCell::new(None),
            preprocessing: directory.preprocessing,
            #[cfg(feature = "native-backend")]
            native_components,
        })
    }

    pub fn decode_backend(&self) -> EngineDecodeBackend {
        self.decode_backend
    }

    /// Number of native component invocations performed by this engine so far,
    /// or `None` when it is not running the native backend. Lets tests prove the
    /// native sessions — not an ORT fallback — executed a workflow.
    #[cfg(feature = "native-backend")]
    pub fn native_component_run_count(&self) -> Option<u64> {
        self.native_components
            .as_ref()
            .map(|set| set.borrow().run_count())
    }

    /// `(device_input_bindings, device_outputs)` accumulated by the native
    /// backend, or `None` when not running the native backend. Non-zero
    /// `device_input_bindings` proves an intermediate or recurring/state tensor
    /// entered a component still device-resident (bound zero-copy, no host
    /// round-trip); both are always zero on the CPU native device. Lets a CUDA
    /// test prove end-to-end device residency rather than a host round-trip.
    #[cfg(feature = "native-backend")]
    pub fn native_device_residency_counts(&self) -> Option<(u64, u64)> {
        self.native_components
            .as_ref()
            .map(|set| set.borrow().device_residency_counts())
    }

    pub fn resource_snapshot(&self) -> onnx_genai_scheduler::GovernorSnapshot {
        self.resource_governor.snapshot()
    }

    pub fn memory_strategy_plan(&self) -> &MemoryStrategyPlan {
        &self.memory_strategy_plan
    }

    pub fn models(&self) -> &PipelineModels {
        &self.models
    }

    /// Effective context limit for a request, combining the package metadata
    /// with an explicit per-request override.
    pub fn effective_max_context(&self, options: &crate::GenerateOptions) -> Option<usize> {
        options.max_context.or_else(|| {
            self.models
                .directory
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.model.as_ref())
                .and_then(|model| model.max_sequence_length)
        })
    }

    /// Execution-provider placement reported by the loaded component sessions.
    pub fn execution_provider_status(&self) -> String {
        // Under the native backend no ORT sessions exist; report the native
        // device the components actually run on instead of an empty/"native"
        // placeholder derived from an absent ORT session set.
        #[cfg(feature = "native-backend")]
        if let Some(native) = self.native_components.as_ref() {
            return native.borrow().device_label().to_string();
        }
        let mut summaries = self
            .models
            .sessions
            .values()
            .map(|session| session.execution_provider_status().summary())
            .collect::<Vec<_>>();
        summaries.sort();
        summaries.dedup();
        if summaries.is_empty() {
            "native".to_string()
        } else {
            summaries.join("; ")
        }
    }

    pub fn adapter_lifecycle_diagnostic(&self) -> AdapterLifecycleDiagnostic {
        self.adapter_cache.borrow().diagnostic()
    }

    pub fn device_authority(&self) -> crate::memory_authority::DeviceMemoryAuthority {
        self.resource_governor.device_authority()
    }

    pub fn set_vram_limit(
        &self,
        limit: onnx_genai_scheduler::ResourceLimit,
    ) -> Result<onnx_genai_scheduler::GovernorReconfigureOutcome, crate::engine::EngineGovernorError>
    {
        self.resource_governor.set_vram_limit(limit)
    }

    pub fn run_pipeline(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        self.run_workflow(request)
    }

    /// Drop every reusable execution result this pipeline is holding, so the
    /// next generation recomputes its workflow from scratch. Returns how many
    /// memoized entries were dropped.
    ///
    /// The workflow engine reuses work through two caches: memoized component
    /// outputs (deterministic components replayed across requests) and
    /// session-scoped workflow state. Benchmarks need to drop both. A harness
    /// that replays one prompt — which is what warmup runs do — otherwise
    /// answers each measured generation almost entirely out of retained state,
    /// and reports a "prefill" that does not vary with prompt length (#1529).
    pub fn clear_prefix_cache(&mut self) -> usize {
        let dropped =
            self.component_outputs.get_mut().len() + self.workflow_session_state.get_mut().len();
        self.component_outputs.get_mut().clear();
        self.workflow_session_state.get_mut().clear();
        dropped
    }

    /// Encode text with the same tokenizer this pipeline uses for prompts.
    ///
    /// The public seam benchmarks need to report how many prompt tokens a
    /// generation actually processed, and to build prompts of an exact token
    /// length. Without it a harness has to re-open `tokenizer.json` itself and
    /// hope it picked the same component this pipeline routes prompts through.
    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<TokenId>> {
        self.tokenizer()?.encode(text).map_err(|e| {
            anyhow::anyhow!(
                "failed to tokenize input text with the pipeline's tokenizer: {e}; \
                 verify the model directory contains a valid tokenizer.json"
            )
        })
    }

    /// Decode token ids back to text with this pipeline's tokenizer, the
    /// inverse seam of [`PipelineEngine::tokenize`].
    pub fn detokenize(&self, tokens: &[TokenId]) -> anyhow::Result<String> {
        self.tokenizer()?
            .decode(tokens)
            .map_err(|e| anyhow::anyhow!("failed to detokenize token ids: {e}"))
    }

    fn tokenizer(&self) -> anyhow::Result<&Tokenizer> {
        self.models
            .tokenizer_for("")
            .context("no tokenizer available for this workflow package")
    }

    pub fn run_pipeline_outputs(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineOutputs> {
        self.run_workflow_outputs(request)
    }

    /// Return an aggregate output for a semantic role.
    pub fn output_for_role<'a>(
        &self,
        outputs: &'a PipelineTensors,
        role: WorkflowOutputRole,
    ) -> Option<&'a Value> {
        let name = self
            .workflow
            .outputs
            .iter()
            .find(|(_, output)| output.role == role)
            .map(|(name, _)| name)?;
        outputs.get(name)
    }

    /// Return the aggregate output or the first row-wise output for a semantic role.
    pub fn structured_output_for_role<'a>(
        &self,
        outputs: &'a PipelineOutputs,
        role: WorkflowOutputRole,
    ) -> Option<&'a Value> {
        let name = self
            .workflow
            .outputs
            .iter()
            .find(|(_, output)| output.role == role)
            .map(|(name, _)| name)?;
        outputs.aggregate(name).or_else(|| {
            self.output_rows_for_role(outputs, role)
                .into_iter()
                .next()
                .map(|(_, value)| value)
        })
    }

    /// Return request-aligned rows for a semantic role, in batch-row order.
    ///
    /// Row indices are positional. The caller maps a row onto its request
    /// through the runtime's own request table, not through any identity the
    /// package serialized.
    pub fn output_rows_for_role<'a>(
        &self,
        outputs: &'a PipelineOutputs,
        role: WorkflowOutputRole,
    ) -> Vec<(usize, &'a Value)> {
        let Some(name) = self
            .workflow
            .outputs
            .iter()
            .find(|(_, output)| output.role == role)
            .map(|(name, _)| name)
        else {
            return Vec::new();
        };
        outputs.rows(name)
    }

    /// Compile request bindings and reusable interpreter state for repeated execution.
    pub fn prepare_workflow_execution(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<WorkflowExecutionPlan<'_>> {
        WorkflowExecutionPlan::new(self, request)
    }

    pub fn checkpoint_session(
        &self,
        session_id: &str,
    ) -> anyhow::Result<WorkflowSessionCheckpoint> {
        let session_state = self.workflow_session_state.borrow();
        let mut semantic_state = HashMap::new();
        let mut contracts = HashMap::new();
        for (cell, declaration) in &self.workflow.state {
            if declaration.scope != onnx_genai_metadata::WorkflowStateScope::Session
                || declaration.class != onnx_genai_metadata::WorkflowStateClass::Semantic
            {
                continue;
            }
            if let Some(value) = session_state.get(&(session_id.to_string(), cell.clone())) {
                semantic_state.insert(cell.clone(), clone_value(value)?);
                contracts.insert(cell.clone(), declaration.contract.clone());
            }
        }
        Ok(WorkflowSessionCheckpoint {
            semantic_state,
            contracts,
        })
    }

    pub fn restore_session_checkpoint(
        &mut self,
        session_id: &str,
        checkpoint: &WorkflowSessionCheckpoint,
    ) -> anyhow::Result<()> {
        for cell in checkpoint.semantic_state.keys() {
            let Some(declaration) = self.workflow.state.get(cell) else {
                anyhow::bail!("workflow checkpoint references unknown state cell '{cell}'");
            };
            if declaration.scope != onnx_genai_metadata::WorkflowStateScope::Session
                || declaration.class != onnx_genai_metadata::WorkflowStateClass::Semantic
            {
                anyhow::bail!("workflow checkpoint state '{cell}' is not semantic session state");
            }
            if checkpoint.contracts.get(cell) != Some(&declaration.contract) {
                anyhow::bail!(
                    "workflow checkpoint state '{cell}' has an incompatible tensor contract"
                );
            }
        }
        let staged = checkpoint
            .semantic_state
            .iter()
            .map(|(cell, value)| Ok((cell.clone(), clone_value(value)?)))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut session_state = self.workflow_session_state.borrow_mut();
        for (cell, declaration) in &self.workflow.state {
            if declaration.scope == onnx_genai_metadata::WorkflowStateScope::Session
                && declaration.class == onnx_genai_metadata::WorkflowStateClass::Semantic
            {
                session_state.remove(&(session_id.to_string(), cell.clone()));
            }
        }
        for (cell, value) in staged {
            session_state.insert((session_id.to_string(), cell), value);
        }
        Ok(())
    }

    /// Convenience text API lowered through the generic tokens package output.
    pub fn generate(&mut self, request: GenerateRequest) -> anyhow::Result<GenerateResult> {
        self.generate_with_callbacks(request.into(), None, None)
    }

    pub fn generate_with_pipeline_request(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_with_callbacks(request, None, None)
    }

    pub fn generate_with_callback(
        &mut self,
        request: PipelineGenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_with_callbacks(request, None, callback)
    }

    pub fn generate_with_callbacks(
        &mut self,
        request: PipelineGenerateRequest,
        mut on_admitted: Option<&mut dyn FnMut()>,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        if let Some(on_admitted) = on_admitted.as_mut() {
            on_admitted();
        }
        let values = self.run_workflow_outputs(request)?;
        let output = self
            .workflow
            .outputs
            .iter()
            .find(|(_, output)| output.role == WorkflowOutputRole::Tokens)
            .map(|(name, _)| name)
            .context("workflow generate() requires one package output with role: tokens")?;
        let rows = values.rows(output);
        if rows.len() > 1 {
            anyhow::bail!(
                "workflow generate() cannot flatten multi-row ragged output '{output}'; use \
                 run_pipeline_outputs() to consume semantic row streams"
            );
        }
        let tokens = values
            .aggregate(output)
            .or_else(|| rows.first().map(|(_, value)| *value))
            .with_context(|| format!("workflow did not emit tokens output '{output}'"))?
            .to_vec_i64()?
            .into_iter()
            .map(|token| u32::try_from(token).context("workflow emitted token outside uint32"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let tokenizer = self.models.tokenizer_for("");
        let text = tokenizer
            .map(|tokenizer| tokenizer.decode(&tokens))
            .transpose()?
            .unwrap_or_default();
        if let Some(callback) = callback.as_mut() {
            for (index, token_id) in tokens.iter().copied().enumerate() {
                let token_text = tokenizer
                    .map(|tokenizer| tokenizer.decode(&[token_id]))
                    .transpose()?
                    .unwrap_or_default();
                callback(GenerateToken {
                    token_id,
                    text: token_text,
                    finish_reason: (index + 1 == tokens.len()).then_some(FinishReason::MaxTokens),
                })?;
            }
        }
        Ok(GenerateResult {
            text,
            token_ids: tokens,
            finish_reason: FinishReason::MaxTokens,
            prefix_cache_hit_len: 0,
            logprobs: None,
            budget_cap: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::workflow_initializer_reservation_bytes;

    #[test]
    fn fused_workflow_reserves_source_and_every_linked_initializer_copy() {
        assert_eq!(
            workflow_initializer_reservation_bytes(100, 75, false).unwrap(),
            175
        );
    }

    #[test]
    fn managed_vmm_does_not_double_charge_initializer_residency() {
        assert_eq!(
            workflow_initializer_reservation_bytes(100, 75, true).unwrap(),
            0
        );
    }

    #[test]
    fn workflow_initializer_reservation_rejects_overflow() {
        assert!(workflow_initializer_reservation_bytes(u64::MAX, 1, false).is_err());
        assert!(
            workflow_initializer_reservation_bytes(u64::MAX, 1, true).is_err(),
            "managed accounting still validates the topology-derived maximum"
        );
    }
}
