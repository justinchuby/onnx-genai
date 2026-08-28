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
    PipelineGenerateRequest, PipelineOutputs, PipelineTensors, WorkflowExecutionPlan,
    WorkflowPerformanceDiagnostic, WorkflowRuntime,
};

impl Engine {
    /// Defense-in-depth for public entries on an already-built runtime.
    ///
    /// Runtime construction is the canonical admission boundary. This reads
    /// the same stored typed decision; it never reclassifies metadata.
    pub(crate) fn reject_undispatched_dflash_generation(&self) -> anyhow::Result<()> {
        self.workflow.require_execution_admitted()
    }

    fn reject_dflash_raw_workflow_api(&self, operation: &str) -> anyhow::Result<()> {
        self.workflow.reject_dflash_raw_execution(operation)
    }

    /// How many device→host materializations this runtime has performed.
    ///
    /// A proposal chain's per-token work is supposed to stay on the device that
    /// produced it. That is not something a throughput number diagnoses — a
    /// reintroduced copy surfaces months later as "it got slower", attributed
    /// to anything but the line responsible — so it is counted and a test holds
    /// it to zero.
    pub fn host_staging_count(&self) -> u64 {
        self.workflow.host_staging_count()
    }

    /// How many bytes this runtime has deliberately read back off a device.
    ///
    /// The companion to [`Engine::host_staging_count`]: that counts whole
    /// tensors brought down, this counts the bytes of the one path that is
    /// allowed to bring anything down — the token id a device argmax produces.
    /// Together they account for every device→host byte the interpreter can
    /// produce, so a test asserting "only token ids came back" can say so as a
    /// number rather than as a hope.
    pub fn device_readback_bytes(&self) -> u64 {
        self.workflow.device_readback_bytes()
    }

    /// How many nodes this runtime executed through each declared contract.
    ///
    /// Which algorithmic executor ran is decided by the contract a component
    /// declares, and this is the count that proves it: a package whose loop
    /// body names the autoregressive-decode step routes its nodes there, and a
    /// package whose body names none routes nothing there and runs its
    /// components from their own artifacts instead.
    pub fn contract_executions(&self) -> std::collections::BTreeMap<String, u64> {
        self.workflow.contract_executions()
    }

    /// How many times each declared component was invoked.
    ///
    /// The other half of the same question: a body that authors its sampler as
    /// an ONNX component shows that component's invocations here and no
    /// contract executions at all.
    pub fn component_invocations(&self) -> std::collections::BTreeMap<String, u64> {
        self.workflow
            .workflow_performance_diagnostic()
            .last_stage_runs
            .into_iter()
            .filter_map(|(stage, runs)| {
                stage
                    .strip_prefix("component:")
                    .map(|component| (component.to_string(), runs))
            })
            .collect()
    }

    /// How many components the package's declared workflow names.
    ///
    /// A fact about the serialized document, for diagnostics. Nothing branches
    /// on it: a caller has the same operations available whatever it says.
    pub fn workflow_component_count(&self) -> usize {
        self.workflow.workflow_spec().components.len()
    }

    /// How many of those components name an ONNX graph.
    pub fn workflow_graph_component_count(&self) -> usize {
        onnx_genai_metadata::classify_workflow(self.workflow.workflow_spec())
            .graph_component_count()
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
        self.reject_dflash_raw_workflow_api("Engine::run_pipeline")?;
        let request = self.apply_pipeline_request_defaults(request)?;
        self.workflow_runtime_mut().run_pipeline(request)
    }

    /// Bind a workflow request once and execute its immutable canonical inputs
    /// repeatedly. Per-execution row selection and intermediate values remain
    /// private to [`WorkflowExecutionPlan::execute`].
    pub fn prepare_pipeline(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<WorkflowExecutionPlan<'_>> {
        self.reject_dflash_raw_workflow_api("Engine::prepare_pipeline")?;
        let request = self.apply_pipeline_request_defaults(request)?;
        WorkflowExecutionPlan::new(self.workflow_runtime(), request)
    }

    pub fn run_pipeline_outputs(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineOutputs> {
        self.reject_dflash_raw_workflow_api("Engine::run_pipeline_outputs")?;
        let request = self.apply_pipeline_request_defaults(request)?;
        self.workflow_runtime_mut().run_pipeline_outputs(request)
    }

    /// Take the ordered workflow publications produced by the most recently
    /// committed generation on this engine worker.
    ///
    /// Publications are installed only after semantic commit. Transport
    /// delivery can therefore fail without rolling back the committed turn.
    pub fn take_committed_workflow_publications(
        &mut self,
    ) -> Vec<crate::pipeline::WorkflowOutputPublication> {
        self.workflow_runtime_mut()
            .take_committed_output_publications()
    }

    pub fn run_pipeline_retained(
        &mut self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        self.reject_dflash_raw_workflow_api("Engine::run_pipeline_retained")?;
        let request = self.apply_pipeline_request_defaults(request)?;
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
        self.reject_undispatched_dflash_generation()?;
        if request.generation_control.is_some()
            && self.workflow_runtime().dflash_diagnostic().is_none()
        {
            return Err(crate::pipeline::GenerationControlUnsupported {
                operation: "Engine::generate_with_pipeline_callbacks",
                runtime: "a non-DFlash workflow",
            }
            .into());
        }
        // A request that binds no tensors is a prompt, and a prompt is served
        // by the ordinary entry point — which admits through the scheduler,
        // reuses a cached prefix, and routes the declared decode step to the
        // fused executor when this runtime has one. Sending it down the
        // tensor-binding path instead would quietly give up all three for a
        // request that asked for none of it.
        let prompt_only = request.inputs.is_empty() && request.component_overrides.is_empty();
        if prompt_only && self.holds_decode_core() {
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
        // Everything that reaches here runs the declared workflow: a
        // no-decode-core prompt, and any request carrying tensors or component
        // overrides. All of it still shares this runtime's scheduler, and —
        // when the process configured one — its KV byte budget: components own
        // their own caches, but the byte accounting a shared budget protects is
        // shared regardless of which executor a step names. Admitting here,
        // under whichever id already names this request's conversation (a
        // continuing workflow session) or a fresh one minted for a cold call,
        // gives this branch the same "reject at the door" guarantee the
        // decode-core branch above already has instead of letting the request
        // fail deep inside node execution once a value the loop needed never
        // arrives.
        //
        // Binding a tensor used to skip all of it: the `!prompt_only` case
        // returned above this block, so the budget was enforced for
        // `generate("hello")` and not for the same prompt plus an image, and
        // `on_admitted()` fired on a path that had made no admission decision
        // to report. That callback is a promise to whoever holds the other end
        // — in the server driver it is a oneshot telling a waiting client it is
        // in — so it must be sent by a request that was admitted, not by one
        // that was never asked about.
        let scheduler_session_id = match request
            .session_id
            .as_deref()
            .and_then(|id| id.parse::<crate::config::SessionId>().ok())
        {
            Some(session_id) if self.workflow_sessions.contains_key(&session_id) => session_id,
            _ => self.workflow_session_ids.mint(),
        };
        let (budget_cap, max_new_tokens) = self.admit_interpreted_generate_request(
            scheduler_session_id,
            &request.request.prompt,
            request.request.options.max_new_tokens,
        )?;
        let mut request = request;
        request.request.options.max_new_tokens = max_new_tokens;
        self.apply_eos_defaults(&mut request.request.options)?;
        if let Some(on_admitted) = on_admitted.as_mut() {
            on_admitted();
        }
        let result = self.run_declared_workflow_generation(request, callback);
        self.scheduler.complete(scheduler_session_id);
        let mut result = result?;
        result.budget_cap = budget_cap;
        Ok(result)
    }

    /// Tokenize a text prompt with the package's own tokenizer and run its
    /// declared workflow.
    ///
    /// A prompt is text; a workflow input declaring the `prompt_tokens` role
    /// wants ids. Encoding it here is what lets `generate("hello")` work on a
    /// package whose components the interpreter invokes — the alternative was
    /// refusing the request and telling the caller to tokenize, which is the
    /// runtime declining to do the one thing it has the tokenizer for.
    fn run_declared_workflow_generation(
        &mut self,
        request: PipelineGenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        let runtime = &*self.workflow;
        let mut request = request;
        if let crate::config::GeneratePrompt::Text(text) = &request.request.prompt {
            let tokenizer = runtime.package_tokenizer().context(
                "this package declares a prompt_tokens input but ships no tokenizer, so a text                  prompt cannot be encoded for it; supply token ids instead",
            )?;
            let encoded = tokenizer
                .encode(text)
                .map_err(|error| anyhow::anyhow!("failed to encode the prompt: {error}"))?;
            request.request.prompt = crate::config::GeneratePrompt::TokenIds(encoded);
        }
        let options = request.request.options.clone();
        let tokenizer = runtime.package_tokenizer();
        if runtime.dflash_diagnostic().is_some() {
            return runtime.run_dflash_generation(&options, request, tokenizer, callback);
        }
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
        self.reject_dflash_raw_workflow_api("Engine::models")?;
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
        self.reject_dflash_raw_workflow_api("Engine::prepare_workflow_execution")?;
        let request = self.apply_pipeline_request_defaults(request)?;
        self.workflow_runtime().prepare_workflow_execution(request)
    }

    fn apply_pipeline_request_defaults(
        &self,
        mut request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineGenerateRequest> {
        self.apply_eos_defaults(&mut request.request.options)?;
        Ok(request)
    }

    /// Group request-local component inputs under the package's symbol-keyed
    /// capacity contract.
    ///
    /// Input maps are keyed by component port name. This performs ownership,
    /// uniformity, padding-aware footprint, and budget admission only; a backend
    /// packer still owns materializing the grouped payload allocation.
    pub fn group_workflow_component_inputs(
        &self,
        component: &str,
        requests: &[(crate::config::SessionId, &PipelineTensors)],
    ) -> anyhow::Result<Vec<onnx_genai_scheduler::AdmittedBatch>> {
        self.workflow_runtime()
            .group_component_batch_inputs(component, requests)
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

    pub fn dflash_diagnostic(&self) -> Option<crate::pipeline::speculative::DFlashDiagnostic> {
        self.workflow_runtime().dflash_diagnostic()
    }

    /// Take execution evidence from the last committed DFlash turn.
    ///
    /// Aborted turns publish no traces, matching state, output, and contract
    /// execution visibility.
    pub fn take_dflash_block_traces(
        &mut self,
    ) -> Vec<crate::pipeline::speculative::DFlashBlockTrace> {
        self.workflow_runtime_mut().take_dflash_block_traces()
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

    /// The declared token-embedding table, as the proposer consumes it.
    ///
    /// Takes the whole [`onnx_genai_metadata::TokenEmbeddingSource`] rather
    /// than a name pair because the declared normalizer is part of what the
    /// table *is*: a caller holding only `(component, table)` cannot say
    /// whether the rows it gets are the ones a proposer must be fed.
    pub fn embedding_table(
        &self,
        source: &onnx_genai_metadata::TokenEmbeddingSource,
    ) -> anyhow::Result<crate::pipeline::speculative::EmbeddingTable> {
        self.workflow_runtime().embedding_table(source)
    }

    /// How many times this runtime read an embedding table out of an artifact.
    ///
    /// A declared `[vocab, hidden]` table is loaded — and, on a device,
    /// uploaded — once for the runtime's life. Re-reading it per proposal is
    /// correct and, at a real vocabulary, costs more than the proposal it
    /// feeds, so the cache is a contract rather than an optimization and this
    /// is what holds it.
    pub fn embedding_table_loads(&self) -> u64 {
        self.workflow.embedding_table_loads()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::WorkflowExecutionAdmission;
    use onnx_genai_scheduler::{ModelKvConfig, ResourceLimits};

    fn refused_engine() -> anyhow::Result<Engine> {
        let mut runtime = crate::pipeline::generation::test_decoder_runtime()?;
        runtime.set_execution_admission_for_test(WorkflowExecutionAdmission::DFlashUnavailable {
            version: "1".to_string(),
            capability: onnx_genai_metadata::capabilities::DFLASH_FLAT_BLOCK,
        });
        let governor = crate::engine::EngineResourceGovernor::new(
            ResourceLimits::default(),
            false,
            ModelKvConfig::known(1, 1),
            0,
        )?;
        Engine::from_workflow(runtime, governor)
    }

    fn request() -> PipelineGenerateRequest {
        PipelineGenerateRequest::new(crate::GenerateRequest::new(
            crate::GeneratePrompt::TokenIds(vec![1]),
        ))
    }

    fn assert_dflash_refusal(error: anyhow::Error) {
        let capability =
            crate::engine::package_capability_error(&error).expect("refusal stays typed");
        assert!(matches!(
            capability,
            crate::engine::PackageCapabilityError::DFlashExecutionUnavailable {
                ref version,
                ref capability,
            } if version == "1"
                && capability == onnx_genai_metadata::capabilities::DFLASH_FLAT_BLOCK
        ));
    }

    #[test]
    fn every_engine_execution_family_consumes_the_canonical_admission() -> anyhow::Result<()> {
        let mut engine = refused_engine()?;
        let before_runs = engine.workflow_performance_diagnostic().runs;
        let before_sessions = engine.sessions.len();
        let before_outputs = engine.workflow.output_publication_state_for_test();
        let mut admitted = false;
        let mut published = false;
        let mut on_admitted = || admitted = true;
        let mut on_token = |_| {
            published = true;
            Ok(())
        };

        assert_dflash_refusal(
            engine
                .generate_with_callbacks(
                    crate::GenerateRequest::new(crate::GeneratePrompt::TokenIds(vec![1])),
                    Some(&mut on_admitted),
                    Some(&mut on_token),
                )
                .expect_err("plain generation must refuse"),
        );
        assert_dflash_refusal(
            engine
                .generate_in_session_with_callback(
                    1,
                    crate::GenerateRequest::new(crate::GeneratePrompt::TokenIds(vec![1])),
                    Some(&mut on_token),
                )
                .expect_err("session generation must refuse"),
        );
        assert_dflash_refusal(
            engine
                .generate_with_pipeline_callbacks(
                    request(),
                    Some(&mut on_admitted),
                    Some(&mut on_token),
                )
                .expect_err("pipeline generation must refuse"),
        );
        assert_dflash_refusal(
            engine
                .run_pipeline(request())
                .err()
                .expect("run_pipeline must refuse"),
        );
        assert_dflash_refusal(
            engine
                .run_pipeline_outputs(request())
                .err()
                .expect("run_pipeline_outputs must refuse"),
        );
        assert_dflash_refusal(
            engine
                .run_pipeline_retained(request())
                .err()
                .expect("retained execution must refuse"),
        );
        let prepared = engine.prepare_pipeline(request());
        assert_dflash_refusal(prepared.err().expect("prepare_pipeline must refuse"));
        let prepared = engine.prepare_workflow_execution(request());
        assert_dflash_refusal(
            prepared
                .err()
                .expect("prepare_workflow_execution must refuse"),
        );
        assert!(
            !admitted,
            "refusal must precede scheduler admission callback"
        );
        assert!(!published, "refusal must precede output callback");
        assert_eq!(engine.sessions.len(), before_sessions);
        assert_eq!(engine.workflow_performance_diagnostic().runs, before_runs);
        assert_eq!(
            engine.workflow.output_publication_state_for_test(),
            before_outputs,
            "DFlash refusal must precede S4 output stream/transaction creation"
        );
        Ok(())
    }
}
