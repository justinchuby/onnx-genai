//! Universal metadata-declared workflow runtime.

use crate::decode::clone_value;
use crate::engine::{
    Engine, EngineConfig, EngineResourceGovernor, MemoryStrategyPlanInput, analyze_model_memory,
    build_memory_strategy_plan, combine_graph_memory, component_governor, log_memory_strategy_plan,
    requested_decode_backend, resolve_vram_limit_bytes,
};
use crate::memory_authority::{MemoryAuthorityProvider, SharedMemoryAuthorityProvider};
use crate::{
    EngineDecodeBackend, FinishReason, GeneratePrompt, GenerateRequest, GenerateResult,
    GenerateToken, GenerateTokenCallback, MemoryStrategyPlan,
};
use anyhow::Context;
use onnx_genai_metadata::{
    CompiledWorkflow, ComponentImplementation, DeviceKind, PreprocessingSpec, RuntimeInputRole,
    ScalarValue, TensorContract, TensorDimension, WorkflowEmitMode, WorkflowInputSource,
    WorkflowNode, WorkflowOutputRole, WorkflowSpec,
};
use onnx_genai_ort::{DataType, PipelineModelDirectory, PipelineModels, SessionOptions, Value};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

mod islands;
mod workflow;

pub use islands::ExecutionIslandDiagnostic;
pub use workflow::{WorkflowExecutionPlan, WorkflowPerformanceDiagnostic};

pub type PipelineTensors = HashMap<String, Value>;

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
    models: PipelineModels,
    resource_governor: EngineResourceGovernor,
    memory_strategy_plan: MemoryStrategyPlan,
    decode_backend: EngineDecodeBackend,
    workflow: WorkflowSpec,
    compiled_workflow: CompiledWorkflow,
    movable_emit_values: HashSet<String>,
    execution_islands: Vec<islands::ExecutionIsland>,
    workflow_performance: RefCell<workflow::WorkflowPerformanceCounters>,
    workflow_execution_generation: Cell<u64>,
    workflow_session_state: RefCell<HashMap<(String, String), Value>>,
    preprocessing: Option<PreprocessingSpec>,
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
        let resolved_vram_bytes = resolve_vram_limit_bytes(&config.limits, None)?;
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
        let resource_governor = component_governor(
            &config,
            None,
            model_weights_bytes,
            None,
            authority_provider.as_ref(),
            &authority_domain,
        )?;
        // CUDA Graph capture applies to stable linked execution islands. Enabling
        // it on every source component rejects valid setup/control-flow graphs
        // before the workflow planner can determine capture eligibility.
        let mut component_session_options = session_options.clone();
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
        let movable_emit_values =
            workflow::compile_movable_emit_values(&compiled_workflow.graph, &workflow);
        let aliasable_output_values =
            workflow::compile_aliasable_output_values(&compiled_workflow.graph);
        let execution_islands = islands::plan_execution_islands(
            &mut compiled_workflow.graph,
            &workflow,
            &models,
            &aliasable_output_values,
        )
        .map_err(|error| anyhow::anyhow!("Failed to plan workflow execution islands: {error}"))?;
        Ok(Self {
            models,
            resource_governor,
            memory_strategy_plan,
            decode_backend,
            workflow,
            compiled_workflow,
            movable_emit_values,
            execution_islands,
            workflow_performance: RefCell::new(workflow::WorkflowPerformanceCounters::default()),
            workflow_execution_generation: Cell::new(0),
            workflow_session_state: RefCell::new(HashMap::new()),
            preprocessing: directory.preprocessing,
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
        let values = self.run_workflow(request)?;
        let output = self
            .workflow
            .outputs
            .iter()
            .find(|(_, output)| output.role == WorkflowOutputRole::Tokens)
            .map(|(name, _)| name)
            .context("workflow generate() requires one package output with role: tokens")?;
        let token_value = values
            .get(output)
            .or_else(|| values.get(&format!("{output}.row.0")));
        if values.contains_key(&format!("{output}.row.1")) {
            anyhow::bail!(
                "workflow generate() cannot flatten multi-row ragged output '{output}'; use \
                 run_pipeline() to consume row streams"
            );
        }
        let tokens = token_value
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
