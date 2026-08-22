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
    /// Kept last so the ORT environment, its registered allocator bridge, and
    /// their plugin/provider teardown outlive every component and execution-island
    /// session that may still call back into them.
    _ort_environment: Option<Arc<onnx_genai_ort::Environment>>,
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
pub fn validate_pipeline_backend_request(
    requested: EngineDecodeBackend,
) -> anyhow::Result<EngineDecodeBackend> {
    let backend = requested_decode_backend(requested)?;
    if backend == EngineDecodeBackend::Native {
        anyhow::bail!(
            "pipeline.workflow executes through generic ONNX component invocations; select the \
             ORT backend and configure its execution provider instead of the legacy native \
             decoder backend"
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

#[cfg(feature = "ort-cuda")]
fn shared_cuda_allocator_config(
    session_options: &SessionOptions,
    authority_provider: Option<&SharedMemoryAuthorityProvider>,
    authority_domain: &crate::memory_authority::DeviceCompatibilityDomain,
    resource_governor: &EngineResourceGovernor,
) -> anyhow::Result<Option<onnx_genai_ort::ManagedCudaAllocatorConfig>> {
    if authority_provider.is_none() || !session_options.selects_cuda() {
        return Ok(None);
    }
    let Some(session_device_id) = session_options.cuda_device_id() else {
        anyhow::bail!(
            "CUDA session options selected a non-host execution provider without a concrete device id"
        );
    };
    let crate::memory_authority::DeviceCompatibilityDomain::Cuda(shared_device_index) =
        authority_domain
    else {
        anyhow::bail!(
            "a shared CUDA allocator bridge was requested for workflow sessions, but the shared \
             authority domain is {authority_domain} rather than cuda:{session_device_id}"
        );
    };
    let shared_device_id = i32::try_from(*shared_device_index)
        .context("shared CUDA authority device id exceeded i32")?;
    anyhow::ensure!(
        shared_device_id == session_device_id,
        "shared CUDA allocator bridge targets device {shared_device_id}, but workflow session \
         options target CUDA device {session_device_id}"
    );
    Ok(Some(onnx_genai_ort::ManagedCudaAllocatorConfig::new(
        session_device_id,
        resource_governor.process_memory_manager(),
        Arc::new(resource_governor.device_authority()),
    )?))
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

    pub fn from_pipeline_dir_with_session_options_and_memory_authority_provider(
        pipeline_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
        provider: Arc<dyn MemoryAuthorityProvider>,
    ) -> anyhow::Result<PipelineEngine> {
        PipelineEngine::from_dir_with_session_options_and_memory_authority_provider(
            pipeline_dir,
            config,
            session_options,
            provider,
        )
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

    pub fn from_dir_with_session_options_and_memory_authority_provider(
        pipeline_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
        provider: Arc<dyn MemoryAuthorityProvider>,
    ) -> anyhow::Result<Self> {
        Self::build(pipeline_dir, config, session_options, Some(provider))
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
        #[cfg(feature = "ort-cuda")]
        let mut session_options = session_options;
        #[cfg(feature = "ort-cuda")]
        if let Some(allocator_config) = shared_cuda_allocator_config(
            &session_options,
            authority_provider.as_ref(),
            &authority_domain,
            &resource_governor,
        )? {
            component_session_options.use_managed_cuda_allocator(allocator_config.clone());
            session_options.use_managed_cuda_allocator(allocator_config);
        }
        component_session_options.graph_capture = false;
        let models = PipelineModels::load_with_component_options(
            pipeline_dir,
            component_session_options,
            session_options,
        )
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
        let execution_islands = if island_reservation_admitted {
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
        let ort_environment = models.environment_handle();
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
            _ort_environment: ort_environment,
        })
    }

    pub fn decode_backend(&self) -> EngineDecodeBackend {
        self.decode_backend
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

    #[cfg(feature = "ort-cuda")]
    mod cuda_managed_bridge {
        use super::super::PipelineEngine;
        use crate::{
            DeviceCompatibilityDomain, DeviceMemoryAuthority, EngineConfig, GeneratePrompt,
            GenerateRequest, MemoryAuthorityProvider, ProcessMemoryManager, ResourceLimit,
        };
        use onnx_genai_ort::{
            DataType, SessionOptions, Value, available_execution_providers, ep_selection,
        };
        use std::fs;
        use std::path::{Path, PathBuf};
        use std::sync::{Arc, Mutex, OnceLock};
        use std::time::{Duration, Instant};

        const BATCH: usize = 4;
        const VOCAB: usize = 128;

        const DECODER: &str = r#"
ir_version: 8
graph {
  node {
    input: "scores" output: "logits" op_type: "Softmax"
    attribute { name: "axis" i: -1 type: INT }
  }
  name: "decoder"
  input { name: "scores" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  output { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

        const SAMPLER: &str = r#"
ir_version: 8
graph {
  node {
    input: "logits" output: "token" op_type: "ArgMax"
    attribute { name: "axis" i: -1 type: INT }
    attribute { name: "keepdims" i: 0 type: INT }
  }
  name: "sampler"
  input { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  output { name: "token" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

        const TERMINATION: &str = r#"
ir_version: 8
graph {
  node { input: "token" input: "eos" output: "done" op_type: "Equal" }
  name: "termination"
  input { name: "token" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "eos" type { tensor_type { elem_type: 7 shape {
    dim { dim_value: 1 }
  }}}}
  output { name: "done" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

        #[derive(Debug, Clone)]
        struct FixedAuthorityProvider {
            manager: ProcessMemoryManager,
            authority: DeviceMemoryAuthority,
        }

        impl FixedAuthorityProvider {
            fn new(device_id: u32) -> Self {
                let manager = ProcessMemoryManager::new().expect("process memory manager");
                let capacity = onnx_genai_ort::cuda_rt::device_memory_info(device_id as i32)
                    .map(|memory| memory.total_bytes)
                    .unwrap_or(1 << 33);
                Self {
                    manager,
                    authority: DeviceMemoryAuthority::new(
                        DeviceCompatibilityDomain::Cuda(device_id),
                        capacity as u64,
                    ),
                }
            }
        }

        impl MemoryAuthorityProvider for FixedAuthorityProvider {
            fn process_memory_manager(&self) -> ProcessMemoryManager {
                self.manager.clone()
            }

            fn validate_limit(
                &self,
                domain: &DeviceCompatibilityDomain,
                _requested: ResourceLimit,
            ) -> anyhow::Result<()> {
                anyhow::ensure!(
                    domain == self.authority.domain(),
                    "unexpected compatibility domain {domain}"
                );
                Ok(())
            }

            fn authority(
                &self,
                domain: &DeviceCompatibilityDomain,
                _resolved_limit_bytes: u64,
            ) -> anyhow::Result<DeviceMemoryAuthority> {
                anyhow::ensure!(
                    domain == self.authority.domain(),
                    "unexpected compatibility domain {domain}"
                );
                Ok(self.authority.clone())
            }
        }

        fn ort_test_lock() -> &'static Mutex<()> {
            static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
            LOCK.get_or_init(|| Mutex::new(()))
        }

        fn cuda_ready() -> bool {
            available_execution_providers()
                .ok()
                .is_some_and(|providers| {
                    providers
                        .iter()
                        .any(|provider| provider.eq_ignore_ascii_case("CUDAExecutionProvider"))
                })
                && onnx_genai_ort::cuda_rt::device_memory_info(0).is_ok()
        }

        fn package_root(name: &str) -> anyhow::Result<PathBuf> {
            let root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/test-fixtures/pipeline-managed-cuda-bridge")
                .join(name);
            fs::create_dir_all(&root)?;
            fs::write(root.join("inference_metadata.yaml"), workflow_metadata())?;
            fs::write(root.join("decoder.onnx.textproto"), DECODER)?;
            fs::write(root.join("sampler.onnx.textproto"), SAMPLER)?;
            fs::write(root.join("termination.onnx.textproto"), TERMINATION)?;
            Ok(root)
        }

        fn workflow_metadata() -> String {
            format!(
                r#"
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      logits:
        contract: {{ dtype: float32, rank: 2, shape: [{BATCH}, {VOCAB}] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: logits }}
        required: true
      eos:
        contract: {{ dtype: int64, rank: 1, shape: [1] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: eos }}
        required: true
    outputs:
      token:
        contract: {{ dtype: int64, rank: 1, shape: [{BATCH}] }}
        role: tokens
        stage: pre_adapter
      done:
        contract: {{ dtype: bool, rank: 1, shape: [{BATCH}] }}
        role: tensor
        stage: pre_adapter
    components:
      decoder:
        implementation: {{ kind: onnx, artifact: decoder.onnx.textproto }}
      sampler:
        implementation: {{ kind: onnx, artifact: sampler.onnx.textproto }}
      termination:
        implementation: {{ kind: onnx, artifact: termination.onnx.textproto }}
    steps:
      - kind: invoke
        component: decoder
        inputs: {{ scores: logits }}
        outputs: {{ logits: policy_logits }}
      - kind: invoke
        component: sampler
        inputs: {{ logits: policy_logits }}
        outputs: {{ token: sampled }}
      - kind: invoke
        component: termination
        inputs: {{ token: sampled, eos: eos }}
        outputs: {{ done: is_done }}
      - kind: emit
        value: sampled
        output: token
        mode: replace
      - kind: emit
        value: is_done
        output: done
        mode: replace
"#
            )
        }

        fn logits_bytes() -> Vec<u8> {
            (0..BATCH * VOCAB)
                .flat_map(|index| {
                    let value = ((index % VOCAB) as f32 * 0.0001).sin();
                    value.to_le_bytes()
                })
                .collect()
        }

        fn workflow_request() -> anyhow::Result<super::super::PipelineGenerateRequest> {
            let request = GenerateRequest::new(GeneratePrompt::TokenIds(Vec::new()));
            Ok(super::super::PipelineGenerateRequest::new(request)
                .with_input(
                    "logits",
                    Value::from_raw_bytes(
                        logits_bytes(),
                        &[BATCH as i64, VOCAB as i64],
                        DataType::Float32,
                    )?,
                )
                .with_input("eos", Value::from_slice_i64(&[0], &[1])?))
        }

        fn wait_for_allocator_idle(
            engine: &PipelineEngine,
            expected_live_allocations: usize,
        ) -> anyhow::Result<()> {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let snapshot = engine
                    .resource_governor
                    .process_memory_manager()
                    .snapshot()?;
                let stats = engine
                    .models
                    .environment()?
                    .managed_cuda_allocator_stats(0)
                    .expect("managed CUDA allocator stats");
                if snapshot.allocations.len() == expected_live_allocations
                    && stats.deferred_release_pending == 0
                {
                    return Ok(());
                }
                anyhow::ensure!(
                    Instant::now() < deadline,
                    "managed CUDA allocator never returned to {expected_live_allocations} live \
                     allocations; snapshot={} stats={stats:?}",
                    snapshot.allocations.len()
                );
                std::thread::sleep(Duration::from_millis(20));
            }
        }

        #[test]
        fn shared_cuda_allocator_bridge_is_visible_to_component_and_island_sessions() {
            let _guard = ort_test_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !cuda_ready() {
                return;
            }

            let root = package_root("component-and-island").expect("workflow fixture");
            let provider = Arc::new(FixedAuthorityProvider::new(0));
            let options = SessionOptions::with_execution_provider(ep_selection("cuda"))
                .with_intra_op_threads(1);
            let mut engine =
                PipelineEngine::from_dir_with_session_options_and_memory_authority_provider(
                    &root,
                    EngineConfig::default(),
                    options,
                    provider,
                )
                .expect("pipeline engine");
            assert!(
                !engine.execution_islands.is_empty(),
                "workflow should lower into an execution island"
            );
            let build_stats = engine
                .models
                .environment()
                .expect("pipeline environment")
                .managed_cuda_allocator_stats(0)
                .expect("managed CUDA allocator stats");
            assert!(
                build_stats.total_allocations > 0 && build_stats.reserve_allocations > 0,
                "session construction did not allocate through the managed CUDA bridge: \
                 {build_stats:?}"
            );

            engine
                .run_pipeline(workflow_request().expect("workflow request"))
                .expect("workflow run");
            let island = engine
                .execution_island_diagnostics()
                .into_iter()
                .next()
                .expect("execution island diagnostics");
            assert!(island.runs > 0, "execution island never ran: {island:?}");
            let after_run = engine
                .models
                .environment()
                .expect("pipeline environment")
                .managed_cuda_allocator_stats(0)
                .expect("managed CUDA allocator stats");
            assert!(
                after_run.deferred_release_accepted > 0,
                "runtime frees never flowed through the managed CUDA bridge queue: {after_run:?}"
            );
            wait_for_allocator_idle(
                &engine,
                engine
                    .resource_governor
                    .process_memory_manager()
                    .snapshot()
                    .expect("baseline snapshot")
                    .allocations
                    .len(),
            )
            .expect("idle after run");

            let baseline_live_allocations = engine
                .resource_governor
                .process_memory_manager()
                .snapshot()
                .expect("baseline snapshot")
                .allocations
                .len();
            let component_allocator = engine
                .models
                .session("decoder")
                .expect("component session")
                .device_allocator()
                .expect("component device allocator")
                .expect("CUDA device allocator");
            let component_value = Value::empty_in(
                &[BATCH as i64, VOCAB as i64],
                DataType::Float32,
                &component_allocator,
            )
            .expect("component-managed device allocation");
            let island_allocator = engine.execution_islands[0]
                .session()
                .device_allocator()
                .expect("island device allocator")
                .expect("CUDA island allocator");
            let island_value = Value::empty_in(&[BATCH as i64], DataType::Int64, &island_allocator)
                .expect("island-managed device allocation");
            let live_snapshot = engine
                .resource_governor
                .process_memory_manager()
                .snapshot()
                .expect("live snapshot");
            assert!(
                live_snapshot.allocations.len() >= baseline_live_allocations + 2,
                "component and island allocations were not both published through the shared \
                 process memory manager: baseline={baseline_live_allocations} now={} ",
                live_snapshot.allocations.len()
            );

            drop(component_value);
            drop(island_value);
            wait_for_allocator_idle(&engine, baseline_live_allocations)
                .expect("component and island frees must settle");
        }

        #[test]
        fn shared_cuda_allocator_bridge_survives_graph_capture() {
            let _guard = ort_test_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !cuda_ready() {
                return;
            }

            let root = package_root("graph-capture").expect("workflow fixture");
            let provider = Arc::new(FixedAuthorityProvider::new(0));
            let mut options = SessionOptions::with_execution_provider(ep_selection("cuda"))
                .with_intra_op_threads(1);
            options.graph_capture = true;
            let mut engine =
                PipelineEngine::from_dir_with_session_options_and_memory_authority_provider(
                    &root,
                    EngineConfig::default(),
                    options,
                    provider,
                )
                .expect("pipeline engine");

            engine
                .run_pipeline(workflow_request().expect("workflow request"))
                .expect("capture warmup run");
            engine
                .run_pipeline(workflow_request().expect("workflow request"))
                .expect("capture replay run");
            let island = engine
                .execution_island_diagnostics()
                .into_iter()
                .next()
                .expect("execution island diagnostics");
            assert!(
                island.capture_eligible,
                "expected a capture-eligible island: {island:?}"
            );
            assert!(island.captures >= 1, "island never captured: {island:?}");
            assert!(island.replays >= 1, "island never replayed: {island:?}");
            let stats = engine
                .models
                .environment()
                .expect("pipeline environment")
                .managed_cuda_allocator_stats(0)
                .expect("managed CUDA allocator stats");
            assert!(
                stats.deferred_release_pending == 0 || stats.deferred_release_accepted > 0,
                "graph-captured workflow did not preserve managed CUDA allocator accounting: {stats:?}"
            );
        }
    }
}
