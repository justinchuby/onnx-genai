//! Universal metadata-declared workflow runtime.

use crate::decode::clone_value;
use crate::engine::{
    EngineConfig, EngineResourceGovernor, Holder, MemoryStrategyPlanInput, analyze_model_memory,
    build_memory_strategy_plan, combine_graph_memory, component_governor, log_memory_strategy_plan,
    requested_decode_backend, resolve_memory_strategy_hot_tier_bytes, resolve_vram_limit_bytes,
};
use crate::memory_authority::{MemoryAuthorityProvider, SharedMemoryAuthorityProvider};
use crate::{EngineDecodeBackend, GeneratePrompt, GenerateRequest, MemoryStrategyPlan, TokenId};
use anyhow::Context;
use onnx_genai_metadata::decoder_workflow::IterationPolicy;
use onnx_genai_metadata::{
    ComponentImplementation, DeviceKind, LiteralValue, RuntimeInputRole, ScalarValue,
    TensorContract, TensorDimension, WorkflowEmitMode, WorkflowInputSource, WorkflowNode,
    WorkflowSpec,
};
use onnx_genai_ort::{
    DataType, PipelineModelDirectory, PipelineModels, SessionOptions, Tokenizer, Value,
};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

mod adapters;
mod arg_reduce;
mod audio;
mod batching;
mod device_ops;
pub(crate) mod generation;
mod islands;
#[cfg(feature = "native-backend")]
mod native_component;
mod row_state;
mod runtime_state;
pub mod speculative;
mod workflow;

pub use adapters::{AdapterActivation, AdapterLifecycleDiagnostic, AdapterSelection};
pub use arg_reduce::{ArgReduceRewrites, WideArgReduceLowering, lower_degenerate_arg_reductions};
pub use audio::{EncodedAudio, encode_pcm16_wav, resample_planar};
pub use batching::{
    BatchContractError, ComposedOwnership, OwnershipLevelValues, PackedOwnership,
    PackedRequestView, PackedTensor, PackedValueView, RebasedI64Slice, batch_contract_error,
};
pub(crate) use generation::validate_generation_workflow;
pub use islands::ExecutionIslandDiagnostic;
pub use onnx_genai_metadata::WorkflowOutputRole;
pub use row_state::{RowScopedState, RowTable, check_selection, gather_rows};
pub use workflow::{
    MISSING_REQUIRED_INPUT, WorkflowExecutionPlan, WorkflowPerformanceDiagnostic,
    is_missing_required_input, workflow_carries_session_state,
};
pub(crate) use workflow::{
    PerTokenCursorIneligible, WorkflowGenerationCursor, WorkflowNodeHost, WorkflowNodeRequest,
};

pub fn has_buffered_pcm16_wav_output(workflow: &WorkflowSpec) -> bool {
    audio::has_buffered_pcm16_wav_output(workflow)
}

pub use audio::buffered_pcm16_wav_output_names;

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
///
/// The three fields below are the state split of
/// `docs/architecture/SESSION_CONCURRENCY.md` §3, one field per category: the
/// frozen plan that could be shared by every worker, the backend handles that
/// belong to exactly one, and the mutable state that one worker accumulates
/// while executing them. Per-pass state (§3.3) is not a field at all — it is
/// [`runtime_state::WorkflowPass`], created when a pass starts and dropped when
/// it ends.
///
/// This is a `W = 1` runtime, and stays one: the runtime is constructed, used
/// and dropped on a single thread exactly as before. What the split changes is
/// that the thread affinity is now a property of the types rather than of the
/// comments.
pub(crate) struct WorkflowRuntime {
    /// §3.1 — immutable after load, and shareable because of it.
    plan: Arc<runtime_state::WorkflowPlan>,
    /// §3.2 storage under §3.3 access: everything this worker mutates.
    ///
    /// Declared before [`Self::backend`] so the values, bindings and allocators
    /// derived from a session are released before the session itself even if
    /// the explicit `Drop` below were ever removed.
    worker: runtime_state::WorkerRuntimeState,
    /// §3.2 — the ORT and native handles this worker owns.
    backend: runtime_state::WorkerBackend,
    /// Kept last so the ORT environment, its registered allocator bridge, and
    /// their plugin/provider teardown outlive every component and execution-island
    /// session that may still call back into them.
    _ort_environment: Option<Arc<onnx_genai_ort::Environment>>,
}

/// Immutable construction plan for another hosted single-decoder worker.
///
/// It contains only the frozen workflow plan and the session-free hosted model
/// directory. Calling [`Self::build`] creates fresh worker state and backend
/// holders on the calling thread.
pub(crate) struct HostedWorkflowWorkerFactory {
    plan: Arc<runtime_state::WorkflowPlan>,
    directory: PipelineModelDirectory,
}

impl HostedWorkflowWorkerFactory {
    pub(crate) fn build(&self) -> WorkflowRuntime {
        WorkflowRuntime {
            plan: Arc::clone(&self.plan),
            worker: runtime_state::WorkerRuntimeState::default(),
            backend: runtime_state::WorkerBackend::new(
                PipelineModels::hosted(self.directory.clone(), SessionOptions::default(), None),
                Vec::new(),
                #[cfg(feature = "native-backend")]
                None,
            ),
            _ort_environment: None,
        }
    }
}

/// Release ORT state that outranks its owner's field order.
///
/// Bindings and session-derived allocators now co-own their `Arc<Session>`, so
/// they can no longer outlive the session whatever order the fields drop in.
/// A `Value` cannot: it is a bare `OrtValue` handle with no back-reference to
/// the allocator whose memory it holds, so ORT still requires every value to be
/// released before that allocator. Nothing in the type system says so, which is
/// exactly why it is done here explicitly instead of left to the field list.
///
/// The state split moved where these live, not whether they are cleared:
/// §13 Phase 1 retains this `Drop` verbatim until `Value` gains an ownership
/// edge to its `Allocator`.
impl Drop for WorkflowRuntime {
    fn drop(&mut self) {
        for island in &mut self.backend.execution_islands {
            island.clear_bindings();
        }
        self.worker.release_ort_state();
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

#[cfg(feature = "ort-cuda")]
fn managed_cuda_bridge_device_id(
    session_options: &SessionOptions,
    authority_provider: Option<&SharedMemoryAuthorityProvider>,
    authority_domain: &crate::memory_authority::DeviceCompatibilityDomain,
) -> anyhow::Result<Option<i32>> {
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
    Ok(Some(session_device_id))
}

#[cfg(feature = "ort-cuda")]
fn shared_cuda_allocator_config(
    session_options: &SessionOptions,
    authority_provider: Option<&SharedMemoryAuthorityProvider>,
    authority_domain: &crate::memory_authority::DeviceCompatibilityDomain,
    resource_governor: &EngineResourceGovernor,
) -> anyhow::Result<Option<onnx_genai_ort::ManagedCudaAllocatorConfig>> {
    let Some(session_device_id) =
        managed_cuda_bridge_device_id(session_options, authority_provider, authority_domain)?
    else {
        return Ok(None);
    };
    Ok(Some(onnx_genai_ort::ManagedCudaAllocatorConfig::new(
        session_device_id,
        resource_governor.process_memory_manager(),
        Arc::new(resource_governor.device_authority()),
    )?))
}

impl WorkflowRuntime {
    /// Load a workflow package.
    ///
    /// Returns the interpreter and the one resource governor built for it. The
    /// governor is handed back rather than kept because exactly one exists per
    /// engine: a runtime holding its own beside the engine's would let each
    /// believe it had the whole tier to itself, and every reservation would be
    /// counted twice.
    pub fn from_dir_with_session_options(
        pipeline_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
    ) -> anyhow::Result<(Self, EngineResourceGovernor)> {
        Self::build(pipeline_dir, config, session_options, None)
    }

    pub fn from_dir_with_session_options_and_memory_authority_provider(
        pipeline_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
        provider: Arc<dyn MemoryAuthorityProvider>,
    ) -> anyhow::Result<(Self, EngineResourceGovernor)> {
        Self::build(pipeline_dir, config, session_options, Some(provider))
    }

    /// An interpreter over a workflow whose graph components the caller runs.
    ///
    /// A single-decoder package's decoder is executed by the engine's fused
    /// decode session — the executor registered for
    /// `onnx-genai.autoregressive-decode`, which owns paged KV, the device fast
    /// paths and the CUDA graph — so this runtime builds no ORT session for it,
    /// plans no execution islands, and holds no resource governor: the decode
    /// core it belongs to already owns all three, and a second of any of them
    /// would double-count the same bytes.
    ///
    /// What it does own is the interpreter, which is the point. The loop bound,
    /// the liveness predicate, the carries, the emits and the token stream come
    /// from the workflow the package declares, executed by the same
    /// `run_workflow_node` that executes a composite pipeline. There is one
    /// execution machinery; what differs between packages is which executor
    /// implements a declared step, and that is answered by the step's contract.
    pub(crate) fn hosted(
        package_root: std::path::PathBuf,
        workflow: WorkflowSpec,
        decode_backend: EngineDecodeBackend,
        memory_strategy_plan: MemoryStrategyPlan,
        models: PipelineModels,
        speculative: Option<onnx_genai_metadata::SpeculativeContract>,
    ) -> anyhow::Result<Self> {
        let compiled_workflow = onnx_genai_metadata::compile_workflow(&workflow)
            .map_err(|error| anyhow::anyhow!("Failed to lower workflow metadata: {error}"))?;
        let row_wise_outputs = workflow::workflow_row_wise_outputs(&compiled_workflow.graph);
        let movable_emit_values =
            workflow::compile_movable_emit_values(&compiled_workflow.graph, &workflow);
        let device_bridge_components =
            workflow::compile_device_bridge_components(&compiled_workflow.graph, &HashSet::new());
        Ok(Self {
            plan: Arc::new(runtime_state::WorkflowPlan {
                package_root,
                workflow,
                compiled_workflow,
                row_wise_outputs,
                movable_emit_values,
                device_bridge_components,
                memory_strategy_plan,
                decode_backend,
                adapter_service: None,
                preprocessing: None,
                speculative,
            }),
            worker: runtime_state::WorkerRuntimeState::default(),
            backend: runtime_state::WorkerBackend::new(
                models,
                Vec::new(),
                #[cfg(feature = "native-backend")]
                None,
            ),
            _ort_environment: None,
        })
    }

    /// Capture the immutable pieces needed to construct another hosted worker.
    pub(crate) fn hosted_worker_factory(
        &self,
        directory: PipelineModelDirectory,
    ) -> HostedWorkflowWorkerFactory {
        HostedWorkflowWorkerFactory {
            plan: Arc::clone(&self.plan),
            directory,
        }
    }

    fn build(
        pipeline_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
        authority_provider: Option<SharedMemoryAuthorityProvider>,
    ) -> anyhow::Result<(Self, EngineResourceGovernor)> {
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
        // This early graph-memory accounting precedes native-device resolution,
        // so it must not fabricate a CUDA ordinal or VRAM capacity. Native
        // component sessions are resolved later when Native is selected; until
        // then the device (VRAM) capacity stays honestly `None` (#947), is
        // reported verbatim as `resolved_device_budget`, and never borrows the
        // host tier. The residency verdict is a separate fact, sized against
        // the measured host-RAM ceiling, so a fitting model reads
        // `FullResident` instead of `Unknown` without inventing a device number.
        let resolved_vram_bytes = resolve_vram_limit_bytes(&config.limits, None)?;
        let residency_ceiling_bytes = resolve_memory_strategy_hot_tier_bytes(&config.limits, None)?;
        #[cfg(feature = "native-cuda")]
        let memory_strategy_overrides = crate::engine::memory_strategy_overrides_from_cuda_env(
            onnx_runtime_ep_cuda::DeviceOffloadPolicy::from_env(),
        );
        #[cfg(not(feature = "native-cuda"))]
        let memory_strategy_overrides = crate::engine::MemoryStrategyOverrides::default();
        let managed_vmm = matches!(config.limits.vram_limit, crate::ResourceLimit::Bytes(_));
        #[cfg(feature = "ort-cuda")]
        let managed_cuda_device_id = managed_cuda_bridge_device_id(
            &session_options,
            authority_provider.as_ref(),
            &authority_domain,
        )?;
        #[cfg(not(feature = "ort-cuda"))]
        let managed_cuda_device_id: Option<i32> = None;
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
        let runtime_manages_initializer_residency = managed_cuda_device_id.is_some();
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
            managed_cuda_device_id.and_then(|device| u32::try_from(device).ok()),
            authority_provider.as_ref(),
            &authority_domain,
        )?;
        if runtime_manages_initializer_residency {
            let available = onnx_runtime_memory_governor::MemoryGovernor::available(
                &resource_governor.device_authority(),
                onnx_runtime_memory_governor::Tier::Device,
            );
            anyhow::ensure!(
                model_weights_bytes <= available,
                "workflow source initializers require {model_weights_bytes} bytes before session \
                 construction, but the shared managed CUDA authority has only {available} bytes \
                 available"
            );
        }
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

        let speculative = directory
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.speculative.clone());
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
        // Values a speculative proposal driver consumes after a pass are real
        // consumers island fusion cannot see; keep them live in both the
        // reservation dry-run and the plan so the two agree.
        let speculative_live_values =
            speculative::externally_used_values(speculative.as_ref(), &workflow);
        let maximum_island_initializer_bytes = islands::maximum_execution_island_initializer_bytes(
            &compiled_workflow.graph,
            &workflow,
            &models,
            &speculative_live_values,
        )?;
        let maximum_initializer_reservation = workflow_initializer_reservation_bytes(
            model_weights_bytes,
            maximum_island_initializer_bytes,
            runtime_manages_initializer_residency,
        )?;
        let additional_initializer_reservation = maximum_initializer_reservation
            .checked_sub(source_initializer_reservation)
            .context("workflow initializer reservation accounting underflow")?;
        let island_reservation_admitted = if runtime_manages_initializer_residency {
            let available = onnx_runtime_memory_governor::MemoryGovernor::available(
                &resource_governor.device_authority(),
                onnx_runtime_memory_governor::Tier::Device,
            );
            if maximum_island_initializer_bytes <= available {
                true
            } else {
                tracing::warn!(
                    additional_initializer_reservation = maximum_island_initializer_bytes,
                    available,
                    "workflow execution-island fusion declined because duplicate initializer \
                     residency was not admitted"
                );
                false
            }
        } else {
            match resource_governor.plan().reserve(
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
            }
        };
        let execution_islands =
            if island_reservation_admitted && decode_backend != EngineDecodeBackend::Native {
                // Execution islands are an ORT `IoBinding` / CUDA-graph
                // performance optimization. Native deliberately does not plan
                // them: it retains the compiled graph's individual component
                // nodes and drives each through the native seam. This is a
                // functional path without ORT fallback, not an island or
                // performance-parity claim.
                islands::plan_execution_islands(
                    &mut compiled_workflow.graph,
                    &workflow,
                    &models,
                    &aliasable_output_values,
                    &speculative_live_values,
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
        let ort_environment = models.environment_handle();
        Ok((
            Self {
                plan: Arc::new(runtime_state::WorkflowPlan {
                    package_root: directory.root.clone(),
                    workflow,
                    compiled_workflow,
                    row_wise_outputs,
                    movable_emit_values,
                    device_bridge_components,
                    memory_strategy_plan,
                    decode_backend,
                    adapter_service: directory.adapters,
                    preprocessing: directory.preprocessing,
                    speculative,
                }),
                worker: runtime_state::WorkerRuntimeState::default(),
                backend: runtime_state::WorkerBackend::new(
                    models,
                    execution_islands,
                    #[cfg(feature = "native-backend")]
                    native_components,
                ),
                _ort_environment: ort_environment,
            },
            resource_governor,
        ))
    }

    pub fn decode_backend(&self) -> EngineDecodeBackend {
        self.plan.decode_backend
    }

    /// The immutable plan this runtime executes (§3.1).
    ///
    /// Handed out behind an `Arc` because that is what it is: frozen at load and
    /// therefore shareable. Nothing yet shares it — there is one worker — but a
    /// caller that could only borrow it would have to be rewritten before one
    /// could, and the point of the split is that it does not.
    #[cfg(test)]
    pub(crate) fn plan(&self) -> &Arc<runtime_state::WorkflowPlan> {
        &self.plan
    }

    /// Open one execution pass over this worker's state (§3.3).
    ///
    /// Minting the pass id here is what invalidates the previous pass's cached
    /// service state: an island or a stable binding compares the id it was
    /// written under against the id of the pass now running.
    pub(crate) fn begin_pass(&self, max_iterations_only: bool) -> runtime_state::WorkflowPass {
        self.worker.begin_pass(max_iterations_only)
    }

    /// Number of native component invocations performed by this engine so far,
    /// or `None` when it is not running the native backend. Lets tests prove the
    /// native sessions — not an ORT fallback — executed a workflow.
    #[cfg(feature = "native-backend")]
    pub fn native_component_run_count(&self) -> Option<u64> {
        self.backend
            .native_components
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
        self.backend
            .native_components
            .as_ref()
            .map(|set| set.borrow().device_residency_counts())
    }

    /// Where a declared component will execute its tensors.
    ///
    /// Placement is configured, not discovered: the native backend is built for
    /// a resolved device, and every component it holds a session for runs
    /// there. An interpreter construct that has to *build* a tensor for a
    /// component — a speculative chain's fused proposer input — can therefore
    /// build it in the right memory the first time, instead of assembling it on
    /// the host and letting the binding upload it once per draft token.
    ///
    /// Anything else — the ORT backend, or a native CPU device — is host, which
    /// is what keeps a CPU or ORT run byte-identical to what it was. That is a
    /// statement about where a construct *builds* its own tensors, not a claim
    /// about every value it will see: an ORT session on a CUDA device can still
    /// publish device-resident outputs, and a construct meeting one adopts it
    /// explicitly rather than assuming.
    pub(crate) fn component_execution_residency(&self) -> anyhow::Result<device_ops::Residency> {
        #[cfg(feature = "native-backend")]
        if let Some(components) = self.backend.native_components.as_ref()
            && let Some(ordinal) = components.borrow().cuda_ordinal()
        {
            let device = i32::try_from(ordinal).map_err(|_| {
                anyhow::anyhow!(
                    "the native backend reports CUDA device ordinal {ordinal}, which does not fit \
                     in the i32 a CUDA device id is; select a device in 0..={} with \
                     ONNX_GENAI_CUDA_DEVICE",
                    i32::MAX
                )
            })?;
            return Ok(device_ops::Residency::Cuda(device));
        }
        Ok(device_ops::Residency::Host)
    }

    /// The workflow this runtime executes.
    pub(crate) fn workflow_spec(&self) -> &onnx_genai_metadata::WorkflowSpec {
        &self.plan.workflow
    }

    /// How many device→host materializations this runtime has performed.
    ///
    /// Summed over the iteration variants this runtime authored, for the same
    /// reason the counter lives on the runtime rather than at its call sites: a
    /// copy is counted without the copier having to remember to count itself.
    /// A variant is a second interpreter over the same package, so a copy it
    /// performs is a copy this package performed — and a residency ratchet that
    /// read only the parent would pass while a variant staged a tensor down.
    pub fn host_staging_count(&self) -> u64 {
        self.fold_counter(WorkflowRuntime::own_host_staging_count)
    }

    /// How many bytes this runtime has deliberately read back off a device.
    pub fn device_readback_bytes(&self) -> u64 {
        self.fold_counter(WorkflowRuntime::own_device_readback_bytes)
    }

    fn own_host_staging_count(&self) -> u64 {
        self.worker.counters.host_staging_count.get()
    }

    fn own_device_readback_bytes(&self) -> u64 {
        self.worker.counters.device_readback_bytes.get()
    }

    /// This runtime's own count plus every iteration variant's.
    ///
    /// One level deep is exhaustive: a variant's body declares one iteration
    /// step and `iteration_variant` refuses to re-author a body that has none,
    /// so a variant can never hold variants of its own.
    fn fold_counter(&self, counter: fn(&WorkflowRuntime) -> u64) -> u64 {
        counter(self)
            + self
                .worker
                .iteration_runtimes
                .borrow()
                .values()
                .map(|variant| counter(variant))
                .sum::<u64>()
    }

    /// How many nodes this runtime executed per declared contract.
    ///
    /// Aggregated across the iteration variants this runtime authored, because
    /// they are the same package's loop expressed with a different body: a
    /// caller asking what this package executed wants one answer, not one per
    /// algorithm it happened to run.
    pub fn contract_executions(&self) -> std::collections::BTreeMap<String, u64> {
        let mut executed = self.worker.counters.contract_executions.borrow().clone();
        for variant in self.worker.iteration_runtimes.borrow().values() {
            for (contract, count) in variant.contract_executions() {
                *executed.entry(contract).or_default() += count;
            }
        }
        executed
    }

    /// This package's generation loop, re-authored to declare `policy`.
    ///
    /// The selection of an iteration algorithm happens *here and only here*:
    /// the caller names the policy it is serving — a row-major batch, a
    /// speculative block — and gets an interpreter over a body whose node
    /// declares that policy's contract. Nothing downstream inspects the package
    /// to decide which executor advances the loop; the interpreter dispatches on
    /// the contract the authored node names, exactly as it does for the
    /// single-token body.
    pub(crate) fn iteration_runtime(
        &self,
        policy: IterationPolicy,
    ) -> anyhow::Result<Rc<WorkflowRuntime>> {
        if let Some(existing) = self.worker.iteration_runtimes.borrow().get(&policy) {
            return Ok(Rc::clone(existing));
        }
        let mut variant =
            onnx_genai_metadata::decoder_workflow::iteration_variant(&self.plan.workflow, policy)
                .map_err(|error| {
                anyhow::anyhow!(
                    "this package's declared generation loop cannot be re-authored as \
                         {policy:?}: {error}"
                )
            })?;
        if policy == IterationPolicy::ContinuousBatch {
            // This variant is created only behind the continuous scheduler,
            // which has already admitted and assigned every live row. Record
            // that proven batching boundary on the runtime-local component so
            // the universal pre-dispatch check does not mistake scheduler
            // capacity for an undeclared package capability.
            let component = variant
                .components
                .get_mut(onnx_genai_metadata::decoder_workflow::BATCH_STEP_COMPONENT)
                .context("continuous-batch variant has no batch_step component")?;
            component.batch_capacity = Some(onnx_genai_metadata::ComponentBatchCapacity::default());
        }
        generation::validate_generation_workflow(&variant)?;
        // A variant body invokes bindings only, so the directory it is built
        // over names no artifacts. Declaring that explicitly — rather than
        // handing it the package's model paths — is what makes "the executor
        // owns the session" a checked property: an ONNX node reaching this
        // interpreter would fail to resolve, instead of quietly loading a
        // second copy of the decoder.
        let directory = PipelineModelDirectory {
            root: self.plan.package_root.clone(),
            metadata_path: None,
            spec: onnx_genai_metadata::PipelineSpec {
                workflow: variant.clone(),
            },
            adapters: None,
            metadata: None,
            preprocessing: None,
            model_paths: BTreeMap::new(),
            tokenizer_paths: onnx_genai_ort::PipelineTokenizerPaths {
                shared: None,
                per_component: BTreeMap::new(),
            },
        };
        let runtime = Rc::new(WorkflowRuntime::hosted(
            self.plan.package_root.clone(),
            variant,
            self.plan.decode_backend,
            self.plan.memory_strategy_plan.clone(),
            PipelineModels::hosted(directory, SessionOptions::default(), None),
            self.plan.speculative.clone(),
        )?);
        self.worker
            .iteration_runtimes
            .borrow_mut()
            .insert(policy, Rc::clone(&runtime));
        Ok(runtime)
    }

    /// Record one node executed through a declared contract.
    pub(crate) fn record_contract_execution(&self, contract: &str) {
        *self
            .worker
            .counters
            .contract_executions
            .borrow_mut()
            .entry(contract.to_string())
            .or_default() += 1;
    }

    /// Drop every session-scoped cell a conversation accumulated.
    ///
    /// Closing a session must actually release what it held; leaving the cells
    /// behind would grow the map for the process's life and let a re-used id
    /// resume a conversation the caller ended.
    pub(crate) fn forget_session(&self, session_id: &str) {
        self.worker
            .session_state
            .borrow_mut()
            .retain(|(session, _), _| session != session_id);
    }

    /// Length of the conversation this session has accumulated, when the
    /// package declares one.
    ///
    /// `None` means the package declares no prompt continuation, so there is no
    /// conversation length to report and the caller keeps its own count.
    pub(crate) fn session_conversation_len(&self, session_id: &str) -> Option<usize> {
        self.session_conversation(session_id)
            .map(|conversation| conversation.len())
    }

    /// Whether this package continues a conversation by putting it in front of
    /// the next turn's prompt.
    ///
    /// True only for `SessionStateCarrier::PromptContinuation`, read from the
    /// shared classifier. Every other carrier hands its lease back inside the
    /// graph or inside a decode core's own state, so nothing is prepended and a
    /// request must not be charged for it.
    pub(crate) fn prepends_session_conversation(&self) -> bool {
        onnx_genai_metadata::classify_session_state(self.workflow_spec())
            .prompt_continuation()
            .is_some()
    }

    /// Tokens this runtime will put in front of the next turn's prompt.
    ///
    /// Only a `prompt_prefix` continuation prepends anything: it is the one
    /// carrier whose mechanism *is* the prompt binding, so the tokens it holds
    /// are prefilled again on every turn and are part of what the next request
    /// costs. A loop carry or a state service group hands its lease back inside
    /// the graph — the tokens it represents live in a cache the package bounds
    /// itself, not in front of a prompt — so nothing is prepended and this is
    /// zero.
    ///
    /// Answered from the typed classification rather than from "does this
    /// session hold a conversation", because those are different questions and
    /// only this one is a length a caller's request will be charged for.
    /// Put a conversation into a session's continuation cell.
    ///
    /// Production fills this cell by completing a turn. The interpreted test
    /// fixtures cannot complete one -- their decoder wants a KV value no
    /// component produces -- so a test about what a *second* turn costs has no
    /// way to reach that state by running a first turn.
    ///
    /// It seeds through the cell the workflow declares rather than a name
    /// spelled in the test, so a fixture that declares no continuation fails
    /// loudly instead of writing an entry nothing will ever read -- which would
    /// leave a test asserting a carry against a silently absent one.
    #[cfg(all(test, feature = "native-backend"))]
    pub(crate) fn seed_session_conversation(
        &self,
        session_id: &str,
        tokens: &[i64],
    ) -> anyhow::Result<()> {
        let facts = onnx_genai_metadata::classify_session_state(self.workflow_spec());
        let cell = facts.prompt_continuation().context(
            "this workflow declares no prompt continuation, so it has no conversation to seed",
        )?;
        let rows = i64::try_from(tokens.len())?;
        self.worker.session_state.borrow_mut().insert(
            (session_id.to_string(), cell.to_string()),
            Value::from_slice_i64(tokens, &[1, rows])?,
        );
        Ok(())
    }

    pub(crate) fn session_prepended_prompt_len(&self, session_id: &str) -> usize {
        let facts = onnx_genai_metadata::classify_session_state(self.workflow_spec());
        let Some(cell) = facts.prompt_continuation() else {
            return 0;
        };
        self.worker
            .session_state
            .borrow()
            .get(&(session_id.to_string(), cell.to_string()))
            .and_then(|value| value.to_vec_i64().ok())
            .map(|tokens| tokens.len())
            .unwrap_or(0)
    }

    /// The tokens a session's declared conversation holds, oldest first.
    ///
    /// This is the value the next turn's prompt input is built from, so it is
    /// also the answer to "what has this session heard" — reported from the
    /// lease itself rather than from a count kept beside it.
    pub(crate) fn session_conversation(&self, session_id: &str) -> Option<Vec<i64>> {
        let (cell, _, _, _) = workflow::workflow_prompt_continuation(self.workflow_spec())?;
        Some(
            self.worker
                .session_state
                .borrow()
                .get(&(session_id.to_string(), cell.to_string()))
                .map(|value| value.to_vec_i64().unwrap_or_default())
                .unwrap_or_default(),
        )
    }

    pub fn memory_strategy_plan(&self) -> &MemoryStrategyPlan {
        &self.plan.memory_strategy_plan
    }

    pub fn models(&self) -> &PipelineModels {
        &self.backend.models
    }

    /// Execution-provider placement reported by the loaded component sessions.
    pub fn execution_provider_status(&self) -> String {
        // Under the native backend no ORT sessions exist; report the native
        // device the components actually run on instead of an empty/"native"
        // placeholder derived from an absent ORT session set.
        #[cfg(feature = "native-backend")]
        if let Some(native) = self.backend.native_components.as_ref() {
            return native.borrow().device_label().to_string();
        }
        let mut summaries = self
            .backend
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
        self.worker.adapter_cache.borrow().diagnostic()
    }

    pub fn run_pipeline(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        self.run_workflow(request)
    }

    /// Run the workflow and keep every value the pass bound, not only the
    /// package's declared outputs.
    ///
    /// This is the seam interpreter-level constructs consume: a chained
    /// speculative proposal reads its `folded_carry_seed` from the target output
    /// the pass produced and binds the proposer's borrowed read-only KV from the
    /// same values the workflow itself bound, so the proposal chain sees exactly
    /// the tensors the package declared rather than a caller's reconstruction of
    /// them.
    pub fn run_pipeline_retained(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        self.run_workflow_retained(request)
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

    /// The tokenizer this package ships, when it ships one.
    ///
    /// A workflow package need not: an image-generation pipeline has no text to
    /// encode, and inventing an empty tokenizer for it would turn a load-time
    /// fact into a runtime surprise.
    pub(crate) fn package_tokenizer(&self) -> Option<&Tokenizer> {
        self.backend.models.tokenizer_for("")
    }

    fn tokenizer(&self) -> anyhow::Result<&Tokenizer> {
        self.backend
            .models
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
            .plan
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
            .plan
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

    /// Return the aggregate output or the first row-wise output for an exact
    /// output name. Unlike [`Self::structured_output_for_role`], this binds to
    /// one specific declared output so a caller that resolved a particular audio
    /// output serves exactly that output rather than the first of its role.
    pub fn structured_output_for_name<'a>(
        &self,
        outputs: &'a PipelineOutputs,
        name: &str,
    ) -> Option<&'a Value> {
        outputs.aggregate(name).or_else(|| {
            outputs
                .rows(name)
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
            .plan
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
        let session_state = self.worker.session_state.borrow();
        let mut semantic_state = HashMap::new();
        let mut contracts = HashMap::new();
        for (cell, declaration) in &self.plan.workflow.state {
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
            let Some(declaration) = self.plan.workflow.state.get(cell) else {
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
        let mut session_state = self.worker.session_state.borrow_mut();
        for (cell, declaration) in &self.plan.workflow.state {
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
        use super::super::WorkflowRuntime;
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
                let capacity = onnx_genai_ort::cuda_rt::device_memory_info(device_id as i32)
                    .map(|memory| memory.total_bytes)
                    .unwrap_or(1 << 33);
                Self::with_capacity(device_id, capacity as u64)
            }

            fn with_capacity(device_id: u32, capacity: u64) -> Self {
                let manager = ProcessMemoryManager::new().expect("process memory manager");
                Self {
                    manager,
                    authority: DeviceMemoryAuthority::new(
                        DeviceCompatibilityDomain::Cuda(device_id),
                        capacity,
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
        contract: {{ dtype: float32, shape: [{BATCH}, {VOCAB}] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: logits }}
        required: true
      eos:
        contract: {{ dtype: int64, shape: [1] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: eos }}
        required: true
    outputs:
      token:
        contract: {{ dtype: int64, shape: [{BATCH}] }}
        role: tokens
        stage: pre_adapter
      done:
        contract: {{ dtype: bool, shape: [{BATCH}] }}
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
            engine: &WorkflowRuntime,
            governor: &crate::engine::EngineResourceGovernor,
            expected_live_allocations: usize,
        ) -> anyhow::Result<()> {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let snapshot = governor.process_memory_manager().snapshot()?;
                let stats = engine
                    .backend
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
            let (mut engine, engine_governor) =
                WorkflowRuntime::from_dir_with_session_options_and_memory_authority_provider(
                    &root,
                    EngineConfig::default(),
                    options,
                    provider,
                )
                .expect("pipeline engine");
            assert!(
                !engine.backend.execution_islands.is_empty(),
                "workflow should lower into an execution island"
            );
            let build_stats = engine
                .backend
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
                .backend
                .models
                .environment()
                .expect("pipeline environment")
                .managed_cuda_allocator_stats(0)
                .expect("managed CUDA allocator stats");
            assert!(
                after_run.deferred_release_quarantined == 0
                    && after_run.deferred_release_enqueue_failures == 0,
                "runtime frees must reclaim after device quiescence without quarantine: \
                 {after_run:?}"
            );
            wait_for_allocator_idle(
                &engine,
                &engine_governor,
                engine_governor
                    .process_memory_manager()
                    .snapshot()
                    .expect("baseline snapshot")
                    .allocations
                    .len(),
            )
            .expect("idle after run");

            let baseline_live_allocations = engine_governor
                .process_memory_manager()
                .snapshot()
                .expect("baseline snapshot")
                .allocations
                .len();
            let component_allocator = engine
                .backend
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
            let island_allocator = engine.backend.execution_islands[0]
                .session()
                .device_allocator()
                .expect("island device allocator")
                .expect("CUDA island allocator");
            let island_value = Value::empty_in(&[BATCH as i64], DataType::Int64, &island_allocator)
                .expect("island-managed device allocation");
            let live_snapshot = engine_governor
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
            wait_for_allocator_idle(&engine, &engine_governor, baseline_live_allocations)
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
            let (mut engine, _engine_governor) =
                WorkflowRuntime::from_dir_with_session_options_and_memory_authority_provider(
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
                .backend
                .models
                .environment()
                .expect("pipeline environment")
                .managed_cuda_allocator_stats(0)
                .expect("managed CUDA allocator stats");
            assert!(
                stats.deferred_release_pending == 0
                    && stats.deferred_release_quarantined == 0
                    && stats.deferred_release_enqueue_failures == 0,
                "graph-captured workflow did not settle managed CUDA allocator accounting: \
                 {stats:?}"
            );
        }

        #[test]
        fn managed_initializer_residency_is_charged_once_near_the_device_limit() {
            let _guard = ort_test_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !cuda_ready() {
                return;
            }

            let root = package_root("near-budget-single-charge").expect("workflow fixture");
            let options = SessionOptions::with_execution_provider(ep_selection("cuda"))
                .with_intra_op_threads(1);
            let calibration_provider = Arc::new(FixedAuthorityProvider::new(0));
            let (calibration, calibration_governor) =
                WorkflowRuntime::from_dir_with_session_options_and_memory_authority_provider(
                    &root,
                    EngineConfig::default(),
                    options.clone(),
                    calibration_provider,
                )
                .expect("calibration pipeline engine");
            let calibration_snapshot = calibration_governor
                .process_memory_manager()
                .snapshot()
                .expect("calibration PMM snapshot");
            let actual_initializer_charge = calibration_snapshot.charged_bytes;
            assert!(actual_initializer_charge > 0);
            let package_bytes = [
                "decoder.onnx.textproto",
                "sampler.onnx.textproto",
                "termination.onnx.textproto",
            ]
            .into_iter()
            .map(|name| {
                fs::metadata(root.join(name))
                    .expect("component metadata")
                    .len()
            })
            .sum::<u64>();
            drop(calibration);

            let limit = actual_initializer_charge
                .checked_add(package_bytes / 2)
                .expect("near-budget limit");
            let provider = Arc::new(FixedAuthorityProvider::with_capacity(0, limit));
            let mut config = EngineConfig::default();
            config.limits.vram_limit = ResourceLimit::Bytes(limit);
            let (_engine, engine_governor) =
                WorkflowRuntime::from_dir_with_session_options_and_memory_authority_provider(
                    &root, config, options, provider,
                )
                .expect("a model that fits once must not fail from a duplicate fixed reservation");
            let snapshot = engine_governor
                .process_memory_manager()
                .snapshot()
                .expect("near-budget PMM snapshot");
            assert!(
                snapshot.charged_bytes <= limit,
                "managed initializer charges exceeded the shared authority limit: {snapshot:?}"
            );
            assert!(
                engine_governor
                    .plan()
                    .breakdown()
                    .iter()
                    .all(|(name, _, bytes)| *name != "fixed device reservation" || *bytes == 0),
                "managed ORT initializer residency must not keep a second fixed reservation"
            );
            assert_eq!(
                snapshot.authority_snapshots[0].used[0],
                engine_governor.snapshot().vram.used,
                "PMM and engine snapshots must report the same canonical device charge"
            );
        }
    }
}

#[cfg(test)]
mod iteration_runtime_tests {
    use super::*;

    /// A variant interpreter's device→host copies are the package's copies.
    ///
    /// The residency ratchets read `Engine::host_staging_count()` and assert
    /// zero. Once a package can hold more than one interpreter, a counter that
    /// reported only the parent would let a copy performed by a batch or block
    /// body pass the ratchet unseen — the counter would still be honest about
    /// the runtime it names, and wrong about the question the test is asking.
    #[test]
    fn a_variant_runtimes_transfers_are_counted_as_the_packages() -> anyhow::Result<()> {
        let runtime = generation::test_decoder_runtime()?;
        let variant = runtime.iteration_runtime(IterationPolicy::ContinuousBatch)?;
        assert_eq!(runtime.host_staging_count(), 0);
        assert_eq!(runtime.device_readback_bytes(), 0);

        variant.worker.counters.host_staging_count.set(3);
        variant.worker.counters.device_readback_bytes.set(12);
        assert_eq!(
            runtime.host_staging_count(),
            3,
            "a copy a variant performed is a copy this package performed"
        );
        assert_eq!(runtime.device_readback_bytes(), 12);

        runtime.worker.counters.host_staging_count.set(1);
        assert_eq!(
            runtime.host_staging_count(),
            4,
            "the parent's own transfers are still counted beside the variants'"
        );

        // A variant holds no variants, so the fold is exhaustive one level deep.
        assert!(variant.worker.iteration_runtimes.borrow().is_empty());
        Ok(())
    }

    /// Two policies are two interpreters, and both are counted.
    #[test]
    fn every_authored_variant_is_folded_in() -> anyhow::Result<()> {
        let runtime = generation::test_decoder_runtime()?;
        for policy in [
            IterationPolicy::ContinuousBatch,
            IterationPolicy::SpeculativeBlock,
        ] {
            runtime
                .iteration_runtime(policy)?
                .worker
                .counters
                .host_staging_count
                .set(2);
        }
        assert_eq!(runtime.host_staging_count(), 4);
        Ok(())
    }
}

/// The state split of `docs/architecture/SESSION_CONCURRENCY.md` §3, held to by
/// the compiler where it can be and by a test where it cannot.
///
/// These are `W = 1` contracts. None of them claims the runtime is concurrent;
/// they claim that the state a second worker would have to own is separated
/// from the state it could share, and that the compiler now knows which is
/// which.
#[cfg(test)]
mod state_split_contracts {
    use super::runtime_state::{PassId, WorkflowPlan, assert_impl_all, assert_not_impl_all};
    use super::*;

    // §3.1: the plan is shareable because it is frozen. If a mutable cell is
    // ever added to `WorkflowPlan`, this stops compiling — which is the point.
    assert_impl_all!(Arc<WorkflowPlan>: Send, Sync);

    // §3.2/§3.3: the runtime that owns backend handles and worker state is not
    // shareable and not sendable. `Engine`'s hand-written `unsafe impl Send`
    // (`engine/model.rs`) still moves the whole engine onto the driver thread
    // exactly once, as it did before this split; what these assert is that the
    // interpreter does not offer that on its own.
    assert_not_impl_all!(WorkflowRuntime: Send);
    assert_not_impl_all!(WorkflowRuntime: Sync);

    /// The plan can be read from another thread; the runtime holding it cannot
    /// go with it.
    ///
    /// The static assertions above are the real contract. This is what makes
    /// the first half of it *useful* rather than merely true: an `Arc` of the
    /// plan really does cross a thread boundary and answer the same question on
    /// the other side, which is what §3.1's "shared by every worker behind
    /// `Arc`" will mean when there is more than one worker.
    #[test]
    fn the_immutable_plan_can_be_read_from_another_thread() -> anyhow::Result<()> {
        let runtime = generation::test_decoder_runtime()?;
        let expected = runtime.plan().workflow.outputs.len();
        let shared = Arc::clone(runtime.plan());
        let observed = std::thread::spawn(move || shared.workflow.outputs.len())
            .join()
            .map_err(|_| anyhow::anyhow!("plan reader thread panicked"))?;
        assert_eq!(
            observed, expected,
            "a frozen plan must read the same on any thread"
        );
        assert_eq!(
            Arc::strong_count(runtime.plan()),
            1,
            "the borrowed clone is released with the thread that read it"
        );
        Ok(())
    }

    /// Pass ids come from the worker that runs the pass, and start at "none".
    ///
    /// The id is what invalidates a previous pass's cached service state, so a
    /// runtime that has run nothing must not compare equal to one that has run
    /// a pass — which is what `PassId::NONE` is for.
    #[test]
    fn each_pass_gets_a_fresh_id_from_the_worker_that_runs_it() -> anyhow::Result<()> {
        let runtime = generation::test_decoder_runtime()?;
        assert_eq!(
            runtime.worker.current_pass(),
            PassId::NONE,
            "a runtime that has run nothing is in no pass"
        );
        let first = runtime.begin_pass(false);
        let second = runtime.begin_pass(false);
        assert_ne!(first.id, second.id, "two passes must not share an id");
        assert_eq!(runtime.worker.current_pass(), second.id);

        // A re-authored iteration variant is a second interpreter with its own
        // worker state, so it counts its own passes. Its ids are only ever
        // compared against its own cached state, which is what makes a
        // worker-local namespace sound.
        let variant = runtime.iteration_runtime(IterationPolicy::ContinuousBatch)?;
        assert_eq!(variant.worker.current_pass(), PassId::NONE);
        assert_eq!(variant.begin_pass(false).id, first.id);
        assert_eq!(
            runtime.worker.current_pass(),
            second.id,
            "a variant's pass must not advance its parent's"
        );
        Ok(())
    }

    /// A conversation still continues, and the cells it continues from are the
    /// worker's.
    ///
    /// The session-scoped cells moved from a field of the runtime to a field of
    /// its worker state. This reads one back through the public accessors a
    /// continuing turn uses, so the move is a change of owner and not of
    /// behaviour.
    #[cfg(feature = "native-backend")]
    #[test]
    fn a_session_continues_from_the_workers_own_cells() -> anyhow::Result<()> {
        let runtime = generation::test_conversation_decoder_runtime()?;
        let (cell, _, _, _) = workflow::workflow_prompt_continuation(runtime.workflow_spec())
            .context("the conversation fixture must declare a prompt continuation")?;
        assert!(runtime.prepends_session_conversation());

        runtime.worker.session_state.borrow_mut().insert(
            ("session-a".to_string(), cell.to_string()),
            Value::from_slice_i64(&[7, 8, 9], &[1, 3])?,
        );
        runtime.worker.session_state.borrow_mut().insert(
            ("session-b".to_string(), cell.to_string()),
            Value::from_slice_i64(&[4], &[1, 1])?,
        );

        assert_eq!(
            runtime.session_conversation("session-a"),
            Some(vec![7, 8, 9])
        );
        assert_eq!(runtime.session_conversation_len("session-a"), Some(3));
        assert_eq!(runtime.session_prepended_prompt_len("session-a"), 3);

        runtime.forget_session("session-a");
        assert_eq!(
            runtime.session_conversation("session-a"),
            Some(Vec::new()),
            "closing a session must release the cells it accumulated"
        );
        assert_eq!(
            runtime.session_conversation("session-b"),
            Some(vec![4]),
            "and must release no other session's"
        );
        Ok(())
    }

    /// Dropping the runtime still releases the values it cached.
    ///
    /// A `Value` is a bare `OrtValue` handle with no back-reference to the
    /// allocator it came from (§3.4), so nothing in the type system orders their
    /// release and the clears in `Drop` are load-bearing. Moving the maps into
    /// worker state moved those clears; it did not remove them. Holding a second
    /// `Arc` to a cached output is how that is observed: if the drop path stops
    /// clearing, the count stays at two.
    // The output cache is keyed to `Arc<Value>` because that is the owner type
    // the stable-binding path uses; the refcount is never contended because the
    // whole worker is single-threaded, which is exactly what this asserts.
    #[allow(clippy::arc_with_non_send_sync)]
    #[test]
    fn dropping_the_runtime_still_releases_the_values_it_cached() -> anyhow::Result<()> {
        let runtime = generation::test_decoder_runtime()?;
        let cached = Arc::new(Value::from_slice_i64(&[11, 12], &[1, 2])?);
        runtime.worker.component_outputs.borrow_mut().insert(
            (
                "decoder".to_string(),
                "logits".to_string(),
                vec![1, 2],
                "f32".to_string(),
            ),
            Arc::clone(&cached),
        );
        assert_eq!(Arc::strong_count(&cached), 2);
        drop(runtime);
        assert_eq!(
            Arc::strong_count(&cached),
            1,
            "the drop path must release cached values, and before the allocators \
             that own their memory"
        );
        Ok(())
    }

    /// And releases them *before* the allocators, which drop order alone would
    /// not guarantee.
    ///
    /// Release order is not observable from a test: a released `Value` reports
    /// nothing, and an allocator freed too early is undefined behaviour rather
    /// than a failed assertion. So this reads the two places the ordering lives
    /// and fails if either the call or the sequence inside it is lost — which is
    /// what #2019's contract and §13 Phase 1 both say must be retained.
    #[test]
    fn the_drop_path_clears_values_before_allocators() -> anyhow::Result<()> {
        let source_of = |file: &str| -> anyhow::Result<String> {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src/pipeline")
                .join(file);
            std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))
        };

        let owner = source_of("mod.rs")?;
        let drop_impl = owner
            .split("impl Drop for WorkflowRuntime")
            .nth(1)
            .context("WorkflowRuntime must keep its explicit Drop")?;
        let drop_body = &drop_impl[..drop_impl.find("\n}").unwrap_or(drop_impl.len())];
        assert!(
            drop_body.contains("release_ort_state()"),
            "WorkflowRuntime::drop must still release ORT state through its worker"
        );

        let state = source_of("runtime_state.rs")?;
        let release = state
            .split("fn release_ort_state")
            .nth(1)
            .context("worker state must keep release_ort_state")?;
        let body = &release[..release.find("\n    }").unwrap_or(release.len())];
        let position = |field: &str| {
            body.find(&format!("{field}.get_mut().clear()"))
                .unwrap_or_else(|| panic!("release_ort_state must clear {field}"))
        };
        assert!(
            position("component_bindings") < position("component_outputs"),
            "bindings hold bound values, so they are released first"
        );
        assert!(
            position("component_outputs") < position("component_allocators"),
            "a Value has no back-reference to its allocator, so values go first"
        );
        Ok(())
    }

    /// One pass at a time per session, still refused by name.
    ///
    /// The lease set moved into worker state. It is the inner-layer invariant
    /// of §4.2 and stays exactly as reachable as it was — this asserts the
    /// refusal still comes back typed after the move.
    #[test]
    fn the_workers_session_lease_still_refuses_a_second_holder() -> anyhow::Result<()> {
        let runtime = generation::test_decoder_runtime()?;
        let leases = &runtime.worker.session_leases;
        assert!(leases.borrow().is_empty());

        let held = workflow::SessionLeaseGuard::acquire(leases, "session-a")?;
        let Err(refused) = workflow::SessionLeaseGuard::acquire(leases, "session-a") else {
            anyhow::bail!("a second holder of an exclusive lease must be refused");
        };
        assert!(matches!(
            refused.downcast_ref::<crate::engine::PackageCapabilityError>(),
            Some(crate::engine::PackageCapabilityError::ExclusiveLeaseConflict { session })
                if session == "session-a"
        ));
        // A different conversation is not blocked by this one.
        let other = workflow::SessionLeaseGuard::acquire(leases, "session-b")?;
        drop(other);
        drop(held);
        assert!(
            leases.borrow().is_empty(),
            "a lease is released when its pass ends"
        );
        Ok(())
    }
}
