//! Canonical lowering: a serialized `model.io` decoder to an in-memory workflow.
//!
//! `pipeline.workflow` is the canonical *execution* representation, but a bare
//! decoder package declares its ABI as `model.io` and must keep doing so:
//! `validate_model_io_against_workflow` rejects a package that serializes both,
//! and that rule exists to keep exactly one *writable* answer to "what is this
//! package's graph ABI".
//!
//! This module resolves the tension without touching either side. It *compiles*
//! the authored `model.io` into a [`WorkflowSpec`] held only in memory, so:
//!
//! * the package on disk is unchanged and still valid — `model.io` stays the
//!   sole serialized source;
//! * nothing writes the result back into `InferenceMetadata::pipeline`, so
//!   validation never sees a pair and no second writable answer exists;
//! * the runtime still gets one execution representation, because the lowered
//!   workflow is what the interpreter runs.
//!
//! The lowering is a pure function of the declared ABI: the same [`ModelIoSpec`]
//! in yields a byte-identical document out. That determinism is what makes the
//! lowered form *derived* rather than a second authored answer — it cannot drift
//! from `model.io`, because it is recomputed from it on every load.
//!
//! # Why it emits a document and then parses it
//!
//! [`lower_decoder_abi`] builds canonical YAML and deserializes it through the
//! ordinary schema path. That is deliberate: the lowered form is exactly what an
//! author would have had to write, it is diffable in a test, and it cannot
//! silently drift from the schema — a field this module got wrong is a parse
//! error at the point of lowering rather than a subtly malformed spec.
//!
//! # The two canonical component contracts
//!
//! The lowered workflow names two `binding` components, dispatched by contract
//! id the way workflow adapters already are, so **no new
//! `ComponentImplementation` variant and no schema change is required**:
//!
//! * [`AUTOREGRESSIVE_DECODE_CONTRACT`] — one decoder forward pass. The runtime
//!   backs it with whichever executor the package resolved (ORT session or
//!   native session). Its KV is declared `management: runtime`, which is the
//!   schema's existing way of saying "the runtime owns these buffers" — so the
//!   paged / share-buffer / CUDA-graph executors keep owning their KV and no
//!   per-step host round-trip is introduced.
//! * [`TOKEN_POLICY_CONTRACT`] — next-token selection and stop detection,
//!   backed by the one Rust sampling/stopping policy (`processors.rs`), which is
//!   the only next-token implementation any *Rust-side* loop uses. An authored
//!   workflow that expresses selection as an in-graph ONNX policy is a different
//!   thing: its sampling runs in the graph, not through this contract. The
//!   contract exists so a lowered decoder keeps the rich Rust sampler rather
//!   than needing an ONNX one.
//!
//! Everything else — the loop, its induction value, the stopping predicate, and
//! the tokens emit — is ordinary workflow structure the interpreter owns.

use crate::schema::{InferenceMetadata, ModelIoSpec, SequenceInputKind, WorkflowSpec};

/// Contract id of the canonical one-step decoder component.
pub const AUTOREGRESSIVE_DECODE_CONTRACT: &str = "onnx-genai.autoregressive-decode";
/// Contract id of the canonical next-token policy component.
pub const TOKEN_POLICY_CONTRACT: &str = "onnx-genai.token-policy";
/// Version both canonical contracts are lowered at.
pub const CANONICAL_CONTRACT_VERSION: &str = "1";

/// Component name the lowered decoder forward pass is bound to.
pub const DECODER_COMPONENT: &str = "decoder";
/// Component name the lowered token policy is bound to.
pub const POLICY_COMPONENT: &str = "token_policy";
/// Package output the lowered workflow emits generated tokens into.
pub const TOKENS_OUTPUT: &str = "tokens";

/// Why a package could not be lowered to a canonical workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringError {
    /// The package already declares a workflow; lowering would manufacture the
    /// second answer the canonical rule exists to prevent.
    AlreadyDeclaresWorkflow,
    /// The package declares no decoder ABI at all.
    NoDecoderAbi,
    /// The declared ABI is missing a port the canonical loop requires.
    MissingPort(&'static str),
    /// `kv_inputs` and `kv_outputs` disagree in length, so the pairs are
    /// ambiguous.
    UnpairedKvPorts { inputs: usize, outputs: usize },
    /// The declared sequence source names no matching input port.
    SequenceSourceWithoutPort(&'static str),
    /// The emitted document did not round-trip through the workflow schema.
    ///
    /// This is a bug in this module, never a package problem, so it says so.
    Malformed(String),
}

impl std::fmt::Display for LoweringError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyDeclaresWorkflow => formatter.write_str(
                "this package already declares pipeline.workflow, which is already the canonical \
                 execution representation; lowering model.io beside it would create the second \
                 answer that rule exists to prevent",
            ),
            Self::NoDecoderAbi => formatter.write_str(
                "this package declares no decoder ABI: neither pipeline.workflow nor model.io \
                 names the executable graph's ports",
            ),
            Self::MissingPort(port) => write!(
                formatter,
                "the declared decoder ABI names no '{port}', which the canonical decode loop \
                 requires; declare it at model.io"
            ),
            Self::UnpairedKvPorts { inputs, outputs } => write!(
                formatter,
                "model.io declares {inputs} kv_inputs and {outputs} kv_outputs; they pair \
                 positionally, so the counts must match"
            ),
            Self::SequenceSourceWithoutPort(kind) => write!(
                formatter,
                "model.io declares sequence_source: {kind} but names no matching input port"
            ),
            Self::Malformed(error) => write!(
                formatter,
                "canonical lowering produced a document the workflow schema rejected ({error}); \
                 this is a runtime bug, not a package problem"
            ),
        }
    }
}

impl std::error::Error for LoweringError {}

/// Compile a package's authored decoder ABI into the canonical workflow.
pub fn lower_decoder_metadata(metadata: &InferenceMetadata) -> Result<WorkflowSpec, LoweringError> {
    if metadata.pipeline.is_some() {
        return Err(LoweringError::AlreadyDeclaresWorkflow);
    }
    let io = metadata
        .decoder_io()
        .ok_or(LoweringError::NoDecoderAbi)?
        .clone();
    lower_decoder_abi(&io)
}

/// The canonical workflow document for a decoder ABI, before parsing.
///
/// Exposed so a test can diff the text and a diagnostic can print exactly what
/// the runtime is about to execute — a lowered package is otherwise invisible,
/// which is the usual way a derived representation drifts unnoticed.
pub fn canonical_workflow_document(io: &ModelIoSpec) -> Result<String, LoweringError> {
    let sequence_port = sequence_port(io)?;
    let sequence_is_embeds = matches!(io.sequence_source, Some(SequenceInputKind::InputsEmbeds));
    let logits = io
        .logits_output
        .as_deref()
        .ok_or(LoweringError::MissingPort("logits_output"))?;
    let pairs = kv_pairs(io)?;

    let mut document = String::new();
    document.push_str(
        "# Canonical workflow lowered from model.io. In-memory only: this document is\n\
         # never written back into the package, so model.io stays the sole serialized\n\
         # answer to the graph ABI and the package on disk remains valid.\n",
    );
    document.push_str("manifest:\n  capabilities: [workflow_ssa, typed_emit, streaming_emit]\n");

    // ── inputs ───────────────────────────────────────────────────────────────
    document.push_str("inputs:\n");
    document.push_str(
        "  request.input_ids:\n    \
         contract: {dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}\n    \
         role: {kind: runtime, version: '1.0', role: prompt_tokens}\n    \
         source: {kind: request}\n    required: true\n",
    );
    if io.attention_mask_input.is_some() {
        document.push_str(
            "  request.attention_mask:\n    \
             contract: {dtype: int64, rank: 2, shape: [batch, kv_sequence], batch_layout: {kind: request_aligned, axis: 0}}\n    \
             role: {kind: opaque}\n    source: {kind: application, name: attention_mask}\n    required: false\n",
        );
    }
    if !pairs.is_empty() {
        document.push_str(
            "  package.max_context:\n    \
             contract: {dtype: int64, rank: 1, shape: [1]}\n    \
             role: {kind: opaque}\n    source: {kind: literal}\n    required: false\n    default: 0\n",
        );
    }
    if io.position_ids_input.is_some() {
        document.push_str(
            "  request.position_ids:\n    \
             contract: {dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}\n    \
             role: {kind: opaque}\n    source: {kind: application, name: position_ids}\n    required: false\n",
        );
    }

    // ── outputs ──────────────────────────────────────────────────────────────
    document.push_str(&format!(
        "outputs:\n  {TOKENS_OUTPUT}:\n    \
         contract: {{dtype: int64, rank: 2, shape: [batch, generated], batch_layout: {{kind: request_aligned, axis: 0}}}}\n    \
         role: tokens\n    stage: pre_adapter\n"
    ));

    // ── components ───────────────────────────────────────────────────────────
    document.push_str("components:\n");
    document.push_str(&format!("  {DECODER_COMPONENT}:\n"));
    document.push_str("    implementation: {kind: binding}\n");
    document.push_str(&format!(
        "    contract: {{id: {AUTOREGRESSIVE_DECODE_CONTRACT}, version: '{CANONICAL_CONTRACT_VERSION}'}}\n"
    ));
    document.push_str("    ports:\n      inputs:\n");
    if sequence_is_embeds {
        document.push_str(&format!(
            "        {sequence_port}: {{dtype: float32, rank: 3, shape: [batch, sequence, hidden], batch_layout: {{kind: request_aligned, axis: 0}}}}\n"
        ));
    } else {
        document.push_str(&format!(
            "        {sequence_port}: {{dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: {{kind: request_aligned, axis: 0}}}}\n"
        ));
    }
    if let Some(mask) = io.attention_mask_input.as_deref() {
        document.push_str(&format!(
            "        {mask}: {{dtype: int64, rank: 2, shape: [batch, kv_sequence], batch_layout: {{kind: request_aligned, axis: 0}}}}\n"
        ));
    }
    if let Some(positions) = io.position_ids_input.as_deref() {
        document.push_str(&format!(
            "        {positions}: {{dtype: int64, rank: 2, shape: [batch, sequence], batch_layout: {{kind: request_aligned, axis: 0}}}}\n"
        ));
    }
    for (past, _) in &pairs {
        document.push_str(&format!(
            "        {past}: {{dtype: float32, rank: 4, shape: [batch, kv_heads, past_sequence, head_dim], batch_layout: {{kind: request_aligned, axis: 0}}}}\n"
        ));
    }
    document.push_str("      outputs:\n");
    document.push_str(&format!(
        "        {logits}: {{dtype: float32, rank: 3, shape: [batch, sequence, vocabulary], batch_layout: {{kind: request_aligned, axis: 0}}}}\n"
    ));
    if let Some(hidden) = io.hidden_output.as_deref() {
        document.push_str(&format!(
            "        {hidden}: {{dtype: float32, rank: 3, shape: [batch, sequence, hidden], batch_layout: {{kind: request_aligned, axis: 0}}}}\n"
        ));
    }
    for (_, present) in &pairs {
        document.push_str(&format!(
            "        {present}: {{dtype: float32, rank: 4, shape: [batch, kv_heads, kv_sequence, head_dim], batch_layout: {{kind: request_aligned, axis: 0}}}}\n"
        ));
    }

    document.push_str(&format!("  {POLICY_COMPONENT}:\n"));
    document.push_str("    implementation: {kind: binding}\n");
    document.push_str(&format!(
        "    contract: {{id: {TOKEN_POLICY_CONTRACT}, version: '{CANONICAL_CONTRACT_VERSION}'}}\n"
    ));
    document.push_str(
        "    ports:\n      inputs:\n        \
         logits: {dtype: float32, rank: 3, shape: [batch, sequence, vocabulary], batch_layout: {kind: request_aligned, axis: 0}}\n      \
         outputs:\n        \
         token_ids: {dtype: int64, rank: 2, shape: [batch, step], batch_layout: {kind: request_aligned, axis: 0}}\n        \
         finished: {dtype: bool, rank: 1, shape: [batch], batch_layout: {kind: request_aligned, axis: 0}}\n",
    );

    // ── state ────────────────────────────────────────────────────────────────
    //
    // Declared `management: runtime`: the decode executor owns these buffers.
    // That is what keeps the paged / share-buffer / CUDA-graph fast paths and
    // their device residency intact — the interpreter owns the *loop*, not the
    // KV bytes, and the schema already has a word for that distinction.
    if !pairs.is_empty() {
        document.push_str("state:\n");
        for (past, _) in &pairs {
            document.push_str(&format!(
                "  {past}:\n    \
                 contract: {{dtype: float32, rank: 4, shape: [batch, kv_heads, past_sequence, head_dim], batch_layout: {{kind: request_aligned, axis: 0}}}}\n    \
                 class: semantic\n    scope: invocation\n    initializer: request.input_ids\n    \
                 recurrence: {{kind: bounded, axis: 2, max: package.max_context}}\n    \
                 service_group: self_attention\n    \
                 management: runtime\n    release_boundary: invocation\n"
            ));
        }
    }

    // ── steps ────────────────────────────────────────────────────────────────
    let mut decoder_inputs = vec![format!("{sequence_port}: step.sequence")];
    if let Some(mask) = io.attention_mask_input.as_deref() {
        decoder_inputs.push(format!("{mask}: request.attention_mask"));
    }
    if let Some(positions) = io.position_ids_input.as_deref() {
        decoder_inputs.push(format!("{positions}: request.position_ids"));
    }
    for (past, _) in &pairs {
        decoder_inputs.push(format!("{past}: {past}"));
    }
    let mut decoder_outputs = vec![format!("{logits}: step.logits")];
    if let Some(hidden) = io.hidden_output.as_deref() {
        decoder_outputs.push(format!("{hidden}: step.hidden_states"));
    }
    for (past, present) in &pairs {
        decoder_outputs.push(format!("{present}: {past}"));
    }

    document.push_str("steps:\n");
    document.push_str("- kind: loop\n  steps:\n");
    document.push_str(&format!(
        "  - {{kind: invoke, component: {DECODER_COMPONENT}, inputs: {{{}}}, outputs: {{{}}}}}\n",
        decoder_inputs.join(", "),
        decoder_outputs.join(", ")
    ));
    document.push_str(&format!(
        "  - {{kind: invoke, component: {POLICY_COMPONENT}, inputs: {{logits: step.logits}}, \
         outputs: {{token_ids: step.token_ids, finished: step.finished}}}}\n"
    ));
    document.push_str(&format!(
        "  - {{kind: emit, value: step.token_ids, output: {TOKENS_OUTPUT}, mode: append}}\n"
    ));
    document.push_str(
        "  continue_when: step.continue\n  max_iterations: request.max_new_tokens\n  \
         termination: generation_eos\n  iteration: {value: step.index, contract: {dtype: int64, rank: 0, shape: []}}\n  \
         carried:\n  - {cell: step.sequence, initial: request.input_ids, next: step.token_ids}\n",
    );

    if !pairs.is_empty() {
        document.push_str("serving:\n  active: request.active\n  done: step.finished\n");
        document.push_str("  state_service:\n    groups:\n      self_attention:\n");
        document.push_str(
            "        kind: full_attention\n        sequence_axis: 2\n        layout: bnsh\n        \
             update: {kind: append}\n        ports:\n          decoder:\n",
        );
        for (past, present) in &pairs {
            document.push_str(&format!(
                "            {past}: {{input: {past}, output: {present}}}\n"
            ));
        }
    }
    Ok(document)
}

/// Compile a decoder ABI into the canonical workflow.
pub fn lower_decoder_abi(io: &ModelIoSpec) -> Result<WorkflowSpec, LoweringError> {
    let document = canonical_workflow_document(io)?;
    serde_yaml::from_str(&document).map_err(|error| LoweringError::Malformed(error.to_string()))
}

fn sequence_port(io: &ModelIoSpec) -> Result<&str, LoweringError> {
    match io.sequence_source.unwrap_or(SequenceInputKind::TokenIds) {
        SequenceInputKind::TokenIds => io
            .token_input
            .as_deref()
            .ok_or(LoweringError::SequenceSourceWithoutPort("token_ids")),
        SequenceInputKind::InputsEmbeds => io
            .inputs_embeds_input
            .as_deref()
            .ok_or(LoweringError::SequenceSourceWithoutPort("inputs_embeds")),
    }
}

fn kv_pairs(io: &ModelIoSpec) -> Result<Vec<(String, String)>, LoweringError> {
    let inputs = io.kv_inputs.clone().unwrap_or_default();
    let outputs = io.kv_outputs.clone().unwrap_or_default();
    if inputs.len() != outputs.len() {
        return Err(LoweringError::UnpairedKvPorts {
            inputs: inputs.len(),
            outputs: outputs.len(),
        });
    }
    Ok(inputs.into_iter().zip(outputs).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoder_io() -> ModelIoSpec {
        serde_yaml::from_str(
            "token_input: input_ids
attention_mask_input: attention_mask
position_ids_input: position_ids
logits_output: logits
kv_inputs: [past_key_values.0.key, past_key_values.0.value]
kv_outputs: [present.0.key, present.0.value]
",
        )
        .expect("decoder io")
    }

    /// The lowering is a pure function of the declared ABI.
    ///
    /// If it were not, the derived workflow could differ between two loads of
    /// the same package, and "recomputed from model.io" would stop being a
    /// guarantee that the two cannot disagree.
    #[test]
    fn lowering_is_deterministic() {
        let io = decoder_io();
        let first = canonical_workflow_document(&io).expect("first lowering");
        for _ in 0..8 {
            assert_eq!(
                canonical_workflow_document(&io).expect("repeat lowering"),
                first,
                "lowering the same ABI twice produced different documents"
            );
        }
        assert_eq!(
            lower_decoder_abi(&io).expect("spec"),
            lower_decoder_abi(&io).expect("spec"),
        );
    }

    /// The lowered document parses through the ordinary workflow schema, so a
    /// field this module gets wrong is a loud error at lowering time rather
    /// than a subtly malformed spec the interpreter trips over later.
    #[test]
    fn lowered_document_round_trips_through_the_schema() {
        let workflow = lower_decoder_abi(&decoder_io()).expect("lowered workflow");
        assert!(workflow.components.contains_key(DECODER_COMPONENT));
        assert!(workflow.components.contains_key(POLICY_COMPONENT));
        assert_eq!(
            workflow.components[DECODER_COMPONENT]
                .contract
                .as_ref()
                .map(|contract| contract.id.as_str()),
            Some(AUTOREGRESSIVE_DECODE_CONTRACT)
        );
        assert_eq!(
            workflow.components[POLICY_COMPONENT]
                .contract
                .as_ref()
                .map(|contract| contract.id.as_str()),
            Some(TOKEN_POLICY_CONTRACT)
        );
        assert!(workflow.outputs.contains_key(TOKENS_OUTPUT));
        // Both canonical components are `binding`: the lowering adds no schema
        // variant, which is what keeps the published JSON schema unchanged.
        for component in workflow.components.values() {
            assert!(matches!(
                component.implementation,
                crate::schema::ComponentImplementation::Binding
            ));
        }
    }

    /// Every declared KV pair becomes exactly one runtime-managed state cell.
    ///
    /// `management: runtime` is the schema's existing way of saying the runtime
    /// owns these buffers, which is what lets the paged / share-buffer /
    /// CUDA-graph executors keep their KV in place instead of round-tripping it
    /// through the interpreter as SSA values every step.
    #[test]
    fn kv_pairs_become_runtime_managed_state() {
        let workflow = lower_decoder_abi(&decoder_io()).expect("lowered workflow");
        assert_eq!(workflow.state.len(), 2);
        for cell in workflow.state.values() {
            assert_eq!(cell.management, crate::schema::StateManagement::Runtime);
            assert_eq!(cell.service_group.as_deref(), Some("self_attention"));
        }
        let groups = &workflow
            .serving
            .as_ref()
            .expect("serving contract")
            .state_service
            .groups;
        assert_eq!(groups["self_attention"].ports["decoder"].len(), 2);
    }

    /// A package that already declares a workflow is refused, because that
    /// workflow *is* the canonical form — lowering beside it would be the second
    /// answer the design exists to prevent.
    #[test]
    fn an_authored_workflow_is_never_lowered_beside() {
        let metadata: InferenceMetadata = serde_yaml::from_str(
            "pipeline:
  workflow:
    manifest: {capabilities: [workflow_ssa]}
    components: {}
    steps: []
",
        )
        .expect("metadata");
        assert_eq!(
            lower_decoder_metadata(&metadata),
            Err(LoweringError::AlreadyDeclaresWorkflow)
        );
    }

    /// Lowering never mutates the package: `model.io` remains the sole
    /// serialized answer and `pipeline` stays absent, so `validate_metadata`
    /// sees exactly what the author wrote.
    #[test]
    fn lowering_leaves_the_serialized_metadata_untouched() {
        let source = "model:\n  io:\n    token_input: input_ids\n    logits_output: logits\n";
        let metadata: InferenceMetadata = serde_yaml::from_str(source).expect("metadata");
        assert!(
            crate::validate_metadata(&metadata).is_ok(),
            "the authored package must be valid before lowering"
        );
        let _ = lower_decoder_metadata(&metadata).expect("lowered");
        // The lowered workflow is returned to the caller and never stored, so
        // the package still declares `model.io` alone. That is what keeps
        // `validate_model_io_against_workflow` satisfied: it never sees a pair.
        assert!(
            metadata.pipeline.is_none(),
            "the lowered workflow must never be written back into pipeline"
        );
        assert!(
            crate::validate_metadata(&metadata).is_ok(),
            "the authored package must stay valid after lowering"
        );
    }

    /// An ABI with unpaired KV ports is refused rather than silently truncated.
    #[test]
    fn unpaired_kv_ports_are_refused() {
        let mut io = decoder_io();
        io.kv_outputs = Some(vec!["present.0.key".to_string()]);
        assert_eq!(
            lower_decoder_abi(&io),
            Err(LoweringError::UnpairedKvPorts {
                inputs: 2,
                outputs: 1
            })
        );
    }

    /// A missing logits port names the field to fix.
    #[test]
    fn a_missing_logits_port_is_named() {
        let mut io = decoder_io();
        io.logits_output = None;
        let error = lower_decoder_abi(&io).expect_err("no logits");
        assert!(format!("{error}").contains("logits_output"), "{error}");
    }
}
