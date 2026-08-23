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

use anyhow::Context as _;
use onnx_genai_ort::PipelineModels;

use crate::config::{GenerateResult, GenerateTokenCallback};
use crate::engine::Engine;
use crate::pipeline::{
    PipelineGenerateRequest, PipelineOutputs, PipelineTensors, WorkflowPerformanceDiagnostic,
    WorkflowRuntime,
};

impl Engine {
    /// How many components the package's declared workflow names.
    ///
    /// A fact about the serialized document, for diagnostics. Nothing branches
    /// on it: a caller has the same operations available whatever it says.
    pub fn workflow_component_count(&self) -> usize {
        self.workflow.workflow_spec().components.len()
    }

    /// How many of those components name an ONNX graph.
    pub fn workflow_graph_component_count(&self) -> usize {
        self.workflow
            .workflow_spec()
            .components
            .values()
            .filter(|component| {
                !matches!(
                    component.implementation,
                    onnx_genai_metadata::ComponentImplementation::Binding
                )
            })
            .count()
    }

    /// Whether the declared workflow contains a generation loop.
    pub fn workflow_declares_generation_loop(&self) -> bool {
        self.workflow
            .workflow_spec()
            .steps
            .iter()
            .any(|step| matches!(step, onnx_genai_metadata::WorkflowStep::Loop { .. }))
    }

    /// The workflow this package declares, exactly as authored.
    ///
    /// There is one workflow and it comes from the package. Nothing is
    /// synthesized, so a diagnostic that prints this is printing what the
    /// package on disk says, not a runtime reconstruction of it.
    pub fn package_workflow(&self) -> Option<&onnx_genai_metadata::WorkflowSpec> {
        Some(self.workflow.workflow_spec())
    }

    /// The package's workflow rendered as the document it was read from.
    pub fn package_workflow_document(&self) -> anyhow::Result<String> {
        let workflow = self
            .package_workflow()
            .context("this package declares no pipeline.workflow")?;
        serde_yaml::to_string(workflow).map_err(Into::into)
    }

    pub(crate) fn workflow_runtime(&self) -> &WorkflowRuntime {
        &self.workflow
    }

    pub(crate) fn workflow_runtime_mut(&mut self) -> &mut WorkflowRuntime {
        &mut self.workflow
    }

    /// Run this package's workflow and return its declared outputs.
    pub fn run_pipeline(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        self.workflow_runtime_mut().run_pipeline(request)
    }

    pub fn run_pipeline_outputs(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineOutputs> {
        self.workflow_runtime_mut().run_pipeline_outputs(request)
    }

    pub fn run_pipeline_retained(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        self.workflow_runtime_mut().run_pipeline_retained(request)
    }

    /// Generate from an explicit workflow request (application-supplied tensors).
    ///
    /// The same drive `generate` uses. A caller reaches for this when it has
    /// tensors to bind that no prompt can express — an image, an audio window,
    /// a caller-built mask — not because the package is a different kind.
    pub fn generate_with_pipeline_request(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_with_pipeline_callbacks(request, None, None)
    }

    pub fn generate_with_pipeline_callbacks(
        &mut self,
        request: PipelineGenerateRequest,
        mut on_admitted: Option<&mut dyn FnMut()>,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        // A request that binds no tensors is a prompt, and a prompt is served
        // by the ordinary entry point — which admits through the scheduler,
        // reuses a cached prefix, and routes the declared decode step to the
        // fused executor when this runtime has one. Sending it down the
        // tensor-binding path instead would quietly give up all three for a
        // request that asked for none of it.
        if request.inputs.is_empty()
            && request.component_overrides.is_empty()
            && self.holds_decode_core()
        {
            return match request.session_id.as_deref().and_then(|id| id.parse().ok()) {
                Some(session) if self.sessions.contains_key(&session) => self
                    .generate_in_session_with_callbacks(
                        session,
                        request.request,
                        on_admitted,
                        callback,
                    ),
                _ => self.generate_with_callbacks(request.request, on_admitted, callback),
            };
        }
        if let Some(on_admitted) = on_admitted.as_mut() {
            on_admitted();
        }
        let runtime = &*self.workflow;
        let options = request.request.options.clone();
        let tokenizer = runtime.models().tokenizer_for("");
        crate::pipeline::generation::run_declared_generation(
            runtime, &options, tokenizer, request, None, callback,
        )
    }

    /// Encode a workflow's declared buffered-PCM16 audio output for serving.
    pub fn encode_audio_output(
        &self,
        outputs: &PipelineOutputs,
        output_name: &str,
    ) -> anyhow::Result<crate::pipeline::EncodedAudio> {
        self.workflow_runtime()
            .encode_audio_output(outputs, output_name)
    }

    pub fn models(&self) -> anyhow::Result<&PipelineModels> {
        Ok(self.workflow_runtime().models())
    }

    pub fn output_for_role<'a>(
        &self,
        outputs: &'a PipelineTensors,
        role: onnx_genai_metadata::WorkflowOutputRole,
    ) -> Option<&'a onnx_genai_ort::Value> {
        self.workflow.output_for_role(outputs, role)
    }

    pub fn structured_output_for_role<'a>(
        &self,
        outputs: &'a PipelineOutputs,
        role: onnx_genai_metadata::WorkflowOutputRole,
    ) -> Option<&'a onnx_genai_ort::Value> {
        self.workflow.structured_output_for_role(outputs, role)
    }

    pub fn output_rows_for_role<'a>(
        &self,
        outputs: &'a PipelineOutputs,
        role: onnx_genai_metadata::WorkflowOutputRole,
    ) -> Vec<(usize, &'a onnx_genai_ort::Value)> {
        self.workflow.output_rows_for_role(outputs, role)
    }

    pub fn prepare_workflow_execution(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<crate::pipeline::WorkflowExecutionPlan<'_>> {
        self.workflow_runtime().prepare_workflow_execution(request)
    }

    pub fn workflow_performance_diagnostic(&self) -> WorkflowPerformanceDiagnostic {
        WorkflowRuntime::workflow_performance_diagnostic(&self.workflow)
    }

    pub fn adapter_lifecycle_diagnostic(&self) -> crate::pipeline::AdapterLifecycleDiagnostic {
        WorkflowRuntime::adapter_lifecycle_diagnostic(&self.workflow)
    }

    /// Execution-island diagnostics for a workflow package, empty otherwise.
    pub fn execution_island_diagnostics(&self) -> Vec<crate::pipeline::ExecutionIslandDiagnostic> {
        WorkflowRuntime::execution_island_diagnostics(&self.workflow)
    }

    /// Restore a workflow session checkpoint.
    pub fn restore_workflow_session_checkpoint(
        &mut self,
        session_id: &str,
        checkpoint: &crate::pipeline::WorkflowSessionCheckpoint,
    ) -> anyhow::Result<()> {
        self.workflow_runtime_mut()
            .restore_session_checkpoint(session_id, checkpoint)
    }

    /// Capture a workflow session checkpoint.
    pub fn checkpoint_workflow_session(
        &self,
        session_id: &str,
    ) -> anyhow::Result<crate::pipeline::WorkflowSessionCheckpoint> {
        self.workflow_runtime().checkpoint_session(session_id)
    }

    /// The package's speculative compatibility contract, when it declares one.
    pub fn speculative_contract(&self) -> Option<&onnx_genai_metadata::SpeculativeContract> {
        Some(&*self.workflow).and_then(WorkflowRuntime::speculative_contract)
    }

    pub fn propose_chained(
        &self,
        run: &PipelineTensors,
        options: crate::pipeline::speculative::ChainedProposalOptions,
    ) -> anyhow::Result<crate::pipeline::speculative::ChainedProposal> {
        self.workflow_runtime().propose_chained(run, options)
    }

    pub fn accept_chained_proposal(
        &self,
        proposal: &crate::pipeline::speculative::ChainedProposal,
        target_tokens: &[i64],
    ) -> anyhow::Result<crate::pipeline::speculative::ProposalAcceptance> {
        self.workflow_runtime()
            .accept_chained_proposal(proposal, target_tokens)
    }

    pub fn speculative_rollback_state(
        &self,
        run: &PipelineTensors,
    ) -> anyhow::Result<PipelineTensors> {
        self.workflow_runtime().speculative_rollback_state(run)
    }

    pub fn rollback_speculative_state(
        &self,
        state: &mut PipelineTensors,
        length: usize,
    ) -> anyhow::Result<()> {
        self.workflow_runtime()
            .rollback_speculative_state(state, length)
    }

    pub fn embedding_table(
        &self,
        component: &str,
        table: &str,
    ) -> anyhow::Result<crate::pipeline::speculative::EmbeddingTable> {
        self.workflow_runtime().embedding_table(component, table)
    }

    /// Native component invocations performed by this package's workflow, or
    /// `None` when it is not running the native backend.
    #[cfg(feature = "native-backend")]
    pub fn native_component_run_count(&self) -> Option<u64> {
        Some(&*self.workflow).and_then(WorkflowRuntime::native_component_run_count)
    }

    #[cfg(feature = "native-backend")]
    pub fn native_device_residency_counts(&self) -> Option<(u64, u64)> {
        Some(&*self.workflow).and_then(WorkflowRuntime::native_device_residency_counts)
    }
}
