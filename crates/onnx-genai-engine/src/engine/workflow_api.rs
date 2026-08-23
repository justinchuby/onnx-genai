//! The one runtime's workflow surface.
//!
//! [`Engine`] is the single generation runtime. When the loaded package declares
//! `pipeline.workflow`, its interpreter state lives in `Engine::workflow` and
//! every operation below is served from there; when the package declares a bare
//! decoder, the same operations are served by the decode core.
//!
//! This is what removes the caller-side split. A server, CLI, C ABI, or
//! benchmark used to have to know *which* of two runtime types it held and
//! branch on it — `handle.pipeline`, `EngineBackend::{Single,Pipeline}`,
//! `EngineDriver::start` vs `start_pipeline`. Now there is one type, one
//! constructor family, and one method per operation that resolves the package's
//! own declaration internally.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use onnx_genai_ort::{PipelineModels, SessionOptions};

use crate::config::{EngineConfig, GenerateRequest, GenerateResult, GenerateTokenCallback};
use crate::engine::Engine;
/// What kind of workflow a runtime executes.
///
/// Every package declares a workflow; this reports its *shape*, which is the
/// thing an operator can act on. It never implies the runtime built the
/// workflow — nothing does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowShapeReport {
    /// One ONNX component consuming the sequence and producing logits.
    SingleDecoder,
    /// Several components, driven step by step by the generic interpreter.
    Composite,
    /// No workflow at all, which no loaded package can be.
    None,
}

impl WorkflowShapeReport {
    /// Stable operator-facing label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SingleDecoder => "single_decoder",
            Self::Composite => "composite",
            Self::None => "none",
        }
    }
}

use crate::pipeline::{
    PipelineGenerateRequest, PipelineOutputs, PipelineTensors, WorkflowPerformanceDiagnostic,
    WorkflowRuntime,
};

impl Engine {
    /// The shape of the workflow this runtime executes.
    pub fn workflow_shape(&self) -> WorkflowShapeReport {
        match self.package_workflow() {
            Some(workflow) => {
                if onnx_genai_metadata::sole_decoder_component(workflow).is_some() {
                    WorkflowShapeReport::SingleDecoder
                } else {
                    WorkflowShapeReport::Composite
                }
            }
            None => WorkflowShapeReport::None,
        }
    }

    /// The workflow this package declares, exactly as authored.
    ///
    /// There is one workflow and it comes from the package. Nothing is
    /// synthesized, so a diagnostic that prints this is printing what the
    /// package on disk says, not a runtime reconstruction of it.
    pub fn package_workflow(&self) -> Option<&onnx_genai_metadata::WorkflowSpec> {
        crate::engine::canonical_workflow(self.workflow.as_deref(), self.lowered_workflow.as_ref())
            .ok()
    }

    /// The package's workflow rendered as the document it was read from.
    pub fn package_workflow_document(&self) -> anyhow::Result<String> {
        let workflow = self
            .package_workflow()
            .context("this package declares no pipeline.workflow")?;
        serde_yaml::to_string(workflow).map_err(Into::into)
    }

    /// Drop the canonical workflow, so the next request has no path to take.
    ///
    /// Test-only. Nothing in the public API can produce this state — the loader
    /// installs a canonical workflow for every package it accepts — so this
    /// exists purely to let a test prove the refusal is real rather than
    /// unreachable.
    #[doc(hidden)]
    pub fn forget_canonical_workflow_for_test(&mut self) {
        self.workflow = None;
        self.lowered_workflow = None;
    }

    /// Whether this package is executed by the workflow interpreter.
    ///
    /// Callers should rarely need this: every operation below already resolves
    /// the right execution path. It exists for diagnostics and for the few
    /// capability reports (batching, KV telemetry) whose *answer* genuinely
    /// differs between a composed workflow and a single decoder.
    pub fn is_workflow(&self) -> bool {
        self.workflow.is_some()
    }

    pub(crate) fn workflow_runtime(&self) -> anyhow::Result<&WorkflowRuntime> {
        self.workflow.as_deref().context(
            "this package declares no pipeline.workflow, so it has no workflow runtime state",
        )
    }

    pub(crate) fn workflow_runtime_mut(&mut self) -> anyhow::Result<&mut WorkflowRuntime> {
        self.workflow.as_deref_mut().context(
            "this package declares no pipeline.workflow, so it has no workflow runtime state",
        )
    }

    /// Load a package that declares `pipeline.workflow`.
    pub fn from_pipeline_dir(pipeline_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        WorkflowRuntime::from_dir_with_config(pipeline_dir, config).and_then(Self::from_workflow)
    }

    pub fn from_pipeline_dir_with_memory_authority_provider(
        pipeline_dir: &Path,
        config: EngineConfig,
        provider: Arc<dyn crate::MemoryAuthorityProvider>,
    ) -> anyhow::Result<Self> {
        WorkflowRuntime::from_dir_with_memory_authority_provider(pipeline_dir, config, provider)
            .and_then(Self::from_workflow)
    }

    pub fn from_pipeline_dir_with_session_options_and_memory_authority_provider(
        pipeline_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
        provider: Arc<dyn crate::MemoryAuthorityProvider>,
    ) -> anyhow::Result<Self> {
        WorkflowRuntime::from_dir_with_session_options_and_memory_authority_provider(
            pipeline_dir,
            config,
            session_options,
            provider,
        )
        .and_then(Self::from_workflow)
    }

    /// Run this package's workflow and return its declared outputs.
    pub fn run_pipeline(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        self.workflow_runtime_mut()?.run_pipeline(request)
    }

    pub fn run_pipeline_outputs(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineOutputs> {
        self.workflow_runtime_mut()?.run_pipeline_outputs(request)
    }

    pub fn run_pipeline_retained(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        self.workflow_runtime_mut()?.run_pipeline_retained(request)
    }

    /// Generate from an explicit workflow request (application-supplied tensors).
    pub fn generate_with_pipeline_request(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<GenerateResult> {
        self.workflow_runtime_mut()?
            .generate_with_pipeline_request(request)
    }

    pub fn generate_pipeline_with_callback(
        &mut self,
        request: PipelineGenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.workflow_runtime_mut()?
            .generate_with_callback(request, callback)
    }

    pub fn generate_pipeline_with_callbacks(
        &mut self,
        request: PipelineGenerateRequest,
        on_admitted: Option<&mut dyn FnMut()>,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.workflow_runtime_mut()?
            .generate_with_callbacks(request, on_admitted, callback)
    }

    /// Encode a workflow's declared buffered-PCM16 audio output for serving.
    pub fn encode_audio_output(
        &self,
        outputs: &PipelineOutputs,
        output_name: &str,
    ) -> anyhow::Result<crate::pipeline::EncodedAudio> {
        self.workflow_runtime()?
            .encode_audio_output(outputs, output_name)
    }

    pub fn models(&self) -> anyhow::Result<&PipelineModels> {
        Ok(self.workflow_runtime()?.models())
    }

    pub fn output_for_role<'a>(
        &self,
        outputs: &'a PipelineTensors,
        role: onnx_genai_metadata::WorkflowOutputRole,
    ) -> Option<&'a onnx_genai_ort::Value> {
        self.workflow
            .as_deref()
            .and_then(|workflow| workflow.output_for_role(outputs, role))
    }

    pub fn structured_output_for_role<'a>(
        &self,
        outputs: &'a PipelineOutputs,
        role: onnx_genai_metadata::WorkflowOutputRole,
    ) -> Option<&'a onnx_genai_ort::Value> {
        self.workflow
            .as_deref()
            .and_then(|workflow| workflow.structured_output_for_role(outputs, role))
    }

    pub fn output_rows_for_role<'a>(
        &self,
        outputs: &'a PipelineOutputs,
        role: onnx_genai_metadata::WorkflowOutputRole,
    ) -> Vec<(usize, &'a onnx_genai_ort::Value)> {
        self.workflow
            .as_deref()
            .map(|workflow| workflow.output_rows_for_role(outputs, role))
            .unwrap_or_default()
    }

    pub fn prepare_workflow_execution(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<crate::pipeline::WorkflowExecutionPlan<'_>> {
        self.workflow_runtime()?.prepare_workflow_execution(request)
    }

    pub fn workflow_performance_diagnostic(&self) -> WorkflowPerformanceDiagnostic {
        self.workflow
            .as_deref()
            .map(WorkflowRuntime::workflow_performance_diagnostic)
            .unwrap_or_default()
    }

    pub fn adapter_lifecycle_diagnostic(&self) -> crate::pipeline::AdapterLifecycleDiagnostic {
        self.workflow
            .as_deref()
            .map(WorkflowRuntime::adapter_lifecycle_diagnostic)
            .unwrap_or_default()
    }

    /// Execution-island diagnostics for a workflow package, empty otherwise.
    pub fn execution_island_diagnostics(&self) -> Vec<crate::pipeline::ExecutionIslandDiagnostic> {
        self.workflow
            .as_deref()
            .map(WorkflowRuntime::execution_island_diagnostics)
            .unwrap_or_default()
    }

    /// Restore a workflow session checkpoint.
    pub fn restore_workflow_session_checkpoint(
        &mut self,
        session_id: &str,
        checkpoint: &crate::pipeline::WorkflowSessionCheckpoint,
    ) -> anyhow::Result<()> {
        self.workflow_runtime_mut()?
            .restore_session_checkpoint(session_id, checkpoint)
    }

    /// Capture a workflow session checkpoint.
    pub fn checkpoint_workflow_session(
        &self,
        session_id: &str,
    ) -> anyhow::Result<crate::pipeline::WorkflowSessionCheckpoint> {
        self.workflow_runtime()?.checkpoint_session(session_id)
    }

    /// The package's speculative compatibility contract, when it declares one.
    pub fn speculative_contract(&self) -> Option<&onnx_genai_metadata::SpeculativeContract> {
        self.workflow
            .as_deref()
            .and_then(WorkflowRuntime::speculative_contract)
    }

    pub fn propose_chained(
        &self,
        run: &PipelineTensors,
        options: crate::pipeline::speculative::ChainedProposalOptions,
    ) -> anyhow::Result<crate::pipeline::speculative::ChainedProposal> {
        self.workflow_runtime()?.propose_chained(run, options)
    }

    pub fn accept_chained_proposal(
        &self,
        proposal: &crate::pipeline::speculative::ChainedProposal,
        target_tokens: &[i64],
    ) -> anyhow::Result<crate::pipeline::speculative::ProposalAcceptance> {
        self.workflow_runtime()?
            .accept_chained_proposal(proposal, target_tokens)
    }

    pub fn speculative_rollback_state(
        &self,
        run: &PipelineTensors,
    ) -> anyhow::Result<PipelineTensors> {
        self.workflow_runtime()?.speculative_rollback_state(run)
    }

    pub fn rollback_speculative_state(
        &self,
        state: &mut PipelineTensors,
        length: usize,
    ) -> anyhow::Result<()> {
        self.workflow_runtime()?
            .rollback_speculative_state(state, length)
    }

    pub fn embedding_table(
        &self,
        component: &str,
        table: &str,
    ) -> anyhow::Result<crate::pipeline::speculative::EmbeddingTable> {
        self.workflow_runtime()?.embedding_table(component, table)
    }

    /// Native component invocations performed by this package's workflow, or
    /// `None` when it is not running the native backend.
    #[cfg(feature = "native-backend")]
    pub fn native_component_run_count(&self) -> Option<u64> {
        self.workflow
            .as_deref()
            .and_then(WorkflowRuntime::native_component_run_count)
    }

    #[cfg(feature = "native-backend")]
    pub fn native_device_residency_counts(&self) -> Option<(u64, u64)> {
        self.workflow
            .as_deref()
            .and_then(WorkflowRuntime::native_device_residency_counts)
    }

    /// Text generation, resolved from the package's own declaration.
    ///
    /// This is the single entry point: a workflow package is driven by the
    /// interpreter over its declared `tokens` output, a decoder package by the
    /// decode core. A caller never chooses.
    pub(crate) fn workflow_generate(
        &mut self,
        request: GenerateRequest,
        on_admitted: Option<&mut dyn FnMut()>,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.workflow_runtime_mut()?
            .generate_with_callbacks(request.into(), on_admitted, callback)
    }
}
