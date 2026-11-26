//! Transactional observation of staged generated tool-call output.
//!
//! This is deliberately below serving: it decodes the canonical generated
//! token stream and invokes the shared exact parser, but never publishes a
//! transport event or executes a tool.

use onnx_genai_metadata::{ToolCall, ToolCallStream, ToolParseOutcome, ToolProtocol};
use onnx_genai_ort::Tokenizer;

use crate::{FinishReason, TokenId, ToolCallPolicy};

/// A normal host stop that prevents a later workflow-loop body entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationStopReason {
    IndentationLimit,
    StopSequence { index: usize },
    PredictionTimeLimit,
    BudgetExhausted,
    EosCommitted,
    ContextExhausted,
    ToolCallsReady(Vec<ToolCall>),
}

impl GenerationStopReason {
    /// The transport-facing completion label for this committed normal stop.
    pub fn finish_reason(&self) -> FinishReason {
        match self {
            Self::IndentationLimit => FinishReason::IndentationLimit,
            Self::StopSequence { index } => FinishReason::StopSequence { index: *index },
            Self::PredictionTimeLimit => FinishReason::PredictionTimeLimit,
            Self::BudgetExhausted => FinishReason::MaxTokens,
            Self::EosCommitted => FinishReason::EosToken,
            Self::ContextExhausted => FinishReason::Length,
            Self::ToolCallsReady(_) => FinishReason::ToolCalls,
        }
    }
}

/// Result of observing provisional generated output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StagedOutputObservation {
    Continue,
    CompleteSoFar(Vec<ToolCall>),
    TerminalComplete(Vec<ToolCall>),
}

/// An output protocol failure observed before semantic turn commit.
#[derive(Debug, thiserror::Error)]
pub enum StagedOutputObservationError {
    #[error(
        "failed to decode staged generated tokens for tool protocol {identity}@{version}: {source}"
    )]
    Decode {
        identity: &'static str,
        version: &'static str,
        #[source]
        source: anyhow::Error,
    },
    #[error("project declared response content for tool protocol {identity}@{version}: {source:#}")]
    Projection {
        identity: &'static str,
        version: &'static str,
        #[source]
        source: anyhow::Error,
    },
    #[error(
        "declared tool protocol {identity}@{version} produced malformed staged output: {reason}"
    )]
    Malformed {
        identity: &'static str,
        version: &'static str,
        reason: String,
    },
    #[error(
        "declared tool protocol {identity}@{version} produced incomplete staged output at {boundary}"
    )]
    Incomplete {
        identity: &'static str,
        version: &'static str,
        boundary: &'static str,
    },
    #[error(
        "declared tool protocol {identity}@{version} violated the request tool policy at \
         {boundary}: {reason}"
    )]
    Policy {
        identity: &'static str,
        version: &'static str,
        boundary: &'static str,
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, Default)]
enum ProjectedInput {
    #[default]
    Unset,
    Text,
    Tokens,
}
#[derive(Clone)]
enum ProjectionTokenCodec {
    Bytes(std::sync::Arc<crate::byte_stream::ByteStreamContract>),
    Canonical(std::sync::Arc<Tokenizer>),
    Deferred {
        tokenizer: std::sync::Arc<Tokenizer>,
        reason: std::sync::Arc<anyhow::Error>,
    },
}
impl std::fmt::Debug for ProjectionTokenCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes(_) => f.write_str("ByteTokens"),
            Self::Canonical(_) => f.write_str("CanonicalTokens"),
            Self::Deferred { reason, .. } => f
                .debug_struct("DeferredCanonicalTokens")
                .field("reason", reason)
                .finish(),
        }
    }
}

/// Replayable decoder and parser state captured at a transaction boundary.
///
/// It contains canonical token decoding state as well as the parser's
/// incremental input. Restoring it before retry ensures token/chunk grouping
/// cannot leak an incomplete envelope into a later turn.
#[derive(Debug, Clone)]
pub struct StagedOutputCheckpoint {
    tokens: Vec<TokenId>,
    text: String,
    raw: Vec<u8>,
    input: ProjectedInput,
    stream: ToolCallStream,
    terminal: Option<Vec<ToolCall>>,
}

/// Shared-protocol observer owned by a generation host while its turn is
/// provisional.
#[derive(Debug, Clone)]
pub struct ToolCallStagedOutputObserver {
    protocol: ToolProtocol,
    policy: ToolCallPolicy,
    tokens: Vec<TokenId>,
    text: String,
    raw: Vec<u8>,
    input: ProjectedInput,
    stream: ToolCallStream,
    terminal: Option<Vec<ToolCall>>,
    baseline: Option<StagedOutputCheckpoint>,
    projection: Option<crate::response_template::ResponseProjection>,
    codec: Option<ProjectionTokenCodec>,
    preserved_tokens: std::collections::BTreeSet<u32>,
}

impl ToolCallStagedOutputObserver {
    pub fn new(
        protocol: ToolProtocol,
        policy: ToolCallPolicy,
        preserved_tokens: std::collections::BTreeSet<u32>,
    ) -> Self {
        debug_assert!(policy.observes_output());
        Self {
            protocol,
            policy,
            tokens: Vec::new(),
            text: String::new(),
            raw: Vec::new(),
            input: ProjectedInput::Unset,
            stream: ToolCallStream::default(),
            terminal: None,
            baseline: None,
            projection: None,
            codec: None,
            preserved_tokens,
        }
    }

    /// Apply an already validated response channel definition before observation
    /// starts. Tools are parsed only from its content, including assistant prefill.
    /// Bind the model tokenizer for token observation; None is for text hosts.
    /// Unsupported contextual byte decoders defer tool parsing until terminal
    /// cumulative decode, retaining the reason in this observer's Debug state.
    pub fn with_projection(
        mut self,
        projection: crate::response_template::ResponseProjection,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<Self, StagedOutputObservationError> {
        if !self.raw.is_empty()
            || !self.tokens.is_empty()
            || self.baseline.is_some()
            || self.projection.is_some()
        {
            return Err(self.projection_error(anyhow::anyhow!("response projection must be configured before tool observation starts; create a fresh observer")));
        }
        if let Some(ids) = projection.preserved_tokens() {
            self.preserved_tokens.extend(ids);
        }
        if let Some(tokenizer) = tokenizer {
            projection
                .validate_tokenizer(tokenizer)
                .map_err(|source| self.projection_error(source))?;
        }
        let content = projection
            .project(b"", false)
            .map_err(|source| self.projection_error(source))?;
        self.codec =
            tokenizer.map(
                |tokenizer| match crate::byte_stream::ByteStreamContract::resolve_output(
                    tokenizer,
                    &self.preserved_tokens,
                ) {
                    Ok(Some(contract)) => {
                        ProjectionTokenCodec::Bytes(std::sync::Arc::new(contract))
                    }
                    Ok(None) => {
                        ProjectionTokenCodec::Canonical(std::sync::Arc::new(tokenizer.clone()))
                    }
                    Err(reason) => ProjectionTokenCodec::Deferred {
                        tokenizer: std::sync::Arc::new(tokenizer.clone()),
                        reason: std::sync::Arc::new(reason),
                    },
                },
            );
        self.projection = Some(projection);
        self.replace_content(content)?;
        Ok(self)
    }

    fn projected_token_bytes(
        &self,
        terminal: bool,
    ) -> Result<Option<Vec<u8>>, StagedOutputObservationError> {
        let decoded = match &self.codec {
            Some(ProjectionTokenCodec::Bytes(contract)) => {
                let mut decoder = contract.new_state();
                let mut bytes = Vec::new();
                let limit = self.projection.as_ref().expect("projected token codec owns a projection").max_bytes();
                for &id in &self.tokens {
                    let piece = contract.pieces.get(id as usize).ok_or_else(|| self.projection_error(anyhow::anyhow!("tool observation token {id} is absent from the bound tokenizer vocabulary; use IDs from that model")))?;
                    let delta = decoder.step(piece,contract.kinds[id as usize],limit.saturating_sub(bytes.len())).map_err(|error|self.projection_error(error))?;
                    if bytes.len().checked_add(delta.len()).and_then(|size|size.checked_add(decoder.retained_bytes())).is_none_or(|size| size > limit) {
                        return Err(self.projection_error(anyhow::anyhow!("projected tool output exceeds its {limit}-byte response buffer; reduce max_tokens or increase the response buffer")));
                    }
                    bytes.extend_from_slice(&delta);
                }
                if terminal {
                    let tail = decoder.finish(limit.saturating_sub(bytes.len())).map_err(|error|self.projection_error(error))?;
                    if bytes.len().checked_add(tail.len()).is_none_or(|size| size > limit) {
                        return Err(self.projection_error(anyhow::anyhow!("terminal projected tool output exceeds its {limit}-byte response buffer; reduce max_tokens or increase the response buffer")));
                    }
                    bytes.extend(tail);
                }
                bytes
            }
            Some(ProjectionTokenCodec::Canonical(tokenizer)) => self.decode_tokens(tokenizer)?.into_bytes(),
            Some(ProjectionTokenCodec::Deferred { tokenizer, .. }) if terminal => self.decode_tokens(tokenizer)?.into_bytes(),
            Some(ProjectionTokenCodec::Deferred { .. }) => return Ok(None),
            None => return Err(self.projection_error(anyhow::anyhow!("token observation requires a tokenizer bound by with_projection; provide the model tokenizer before observing token IDs"))),
        };
        Ok(Some(decoded))
    }

    fn projection_error(&self, source: anyhow::Error) -> StagedOutputObservationError {
        let (identity, version) = self.protocol.declaration();
        StagedOutputObservationError::Projection {
            identity,
            version,
            source,
        }
    }

    fn replace_content(
        &mut self,
        content: String,
    ) -> Result<StagedOutputObservation, StagedOutputObservationError> {
        if let Some(chunk) = content.strip_prefix(&self.text) {
            let chunk = chunk.to_owned();
            self.text = content;
            self.observe_stream(&chunk)
        } else {
            self.text = content;
            self.stream = ToolCallStream::default();
            self.observe_stream(&self.text.clone())
        }
    }

    pub fn protocol(&self) -> ToolProtocol {
        self.protocol
    }

    pub fn policy(&self) -> &ToolCallPolicy {
        &self.policy
    }

    pub fn committed_calls(&self) -> Vec<ToolCall> {
        self.terminal.clone().unwrap_or_default()
    }

    /// The normal terminal condition, retained for the host after observation.
    pub fn stop_reason(&self) -> Option<GenerationStopReason> {
        self.terminal
            .as_ref()
            .map(|calls| GenerationStopReason::ToolCallsReady(calls.clone()))
    }

    /// Observe a text chunk already canonicalized by the generation runtime.
    ///
    /// This is useful to a host whose generation unit is text rather than one
    /// token. A complete envelope remains nonterminal because both supported v1
    /// protocols allow another adjacent call and declare no sequence marker.
    pub fn observe_text(
        &mut self,
        chunk: &str,
    ) -> Result<StagedOutputObservation, StagedOutputObservationError> {
        if self.projection.is_some() && matches!(self.input, ProjectedInput::Tokens) {
            return Err(self.projection_error(anyhow::anyhow!("cannot mix text and token observation in one projected turn; use one input representation")));
        }
        self.input = ProjectedInput::Text;
        self.raw.extend_from_slice(chunk.as_bytes());
        if let Some(projection) = &self.projection {
            let content = projection
                .project(&self.raw, false)
                .map_err(|source| self.projection_error(source))?;
            self.replace_content(content)
        } else {
            self.text.push_str(chunk);
            self.observe_stream(chunk)
        }
    }

    /// Decode and observe generated tokens before output publication.
    ///
    /// Tokenizers can finalize text differently as a later token completes a
    /// byte sequence. The observer therefore reparses the canonical cumulative
    /// decode whenever it is not an append-only extension, rather than treating
    /// independently decoded token fragments as protocol authority.
    pub fn observe_tokens(
        &mut self,
        tokenizer: &Tokenizer,
        tokens: &[TokenId],
    ) -> Result<StagedOutputObservation, StagedOutputObservationError> {
        if self.projection.is_some() && matches!(self.input, ProjectedInput::Text) {
            return Err(self.projection_error(anyhow::anyhow!("cannot mix text and token observation in one projected turn; use one input representation")));
        }
        self.input = ProjectedInput::Tokens;
        self.tokens.extend_from_slice(tokens);
        if let Some(projection) = &self.projection {
            let Some(bytes) = self.projected_token_bytes(false)? else {
                // Contextual byte decoders are authoritative at terminal
                // cumulative decode; provisional replacement glyphs stay out
                // of tool syntax and response framing.
                return Ok(StagedOutputObservation::Continue);
            };
            let content = projection
                .project(&bytes, false)
                .map_err(|source| self.projection_error(source))?;
            self.raw = bytes;
            self.replace_content(content)
        } else {
            let decoded = self.decode_tokens(tokenizer)?;
            self.raw = decoded.as_bytes().to_vec();
            self.replace_content(decoded)
        }
    }

    /// Validate the terminal boundary. Only an unfinished envelope fails here;
    /// ordinary text remains a valid no-call completion.
    pub fn finish(
        &mut self,
        boundary: &'static str,
    ) -> Result<StagedOutputObservation, StagedOutputObservationError> {
        if matches!(self.input, ProjectedInput::Tokens) && self.projection.is_some() {
            self.raw = self
                .projected_token_bytes(true)?
                .expect("terminal token decoding returns bytes");
        }
        if let Some(projection) = &self.projection {
            let content = projection
                .project(&self.raw, true)
                .map_err(|source| self.projection_error(source))?;
            self.replace_content(content)?;
        }
        let outcome = self.stream.clone().finish(self.protocol);
        match &outcome {
            ToolParseOutcome::NoCall => {
                self.validate_policy(&outcome, boundary)?;
                Ok(StagedOutputObservation::Continue)
            }
            ToolParseOutcome::TerminalComplete(calls) => {
                self.validate_policy(&outcome, boundary)?;
                self.terminal = Some(calls.clone());
                Ok(StagedOutputObservation::TerminalComplete(calls.clone()))
            }
            ToolParseOutcome::CompleteSoFar(_) => {
                unreachable!("finish marks complete calls terminal")
            }
            ToolParseOutcome::Incomplete => {
                let (identity, version) = self.protocol.declaration();
                Err(StagedOutputObservationError::Incomplete {
                    identity,
                    version,
                    boundary,
                })
            }
            ToolParseOutcome::Malformed(reason) => {
                let (identity, version) = self.protocol.declaration();
                Err(StagedOutputObservationError::Malformed {
                    identity,
                    version,
                    reason: reason.clone(),
                })
            }
        }
    }

    /// Snapshot state at admission so a failed provisional turn can retry from
    /// exactly its committed decoder/parser baseline.
    pub fn begin_turn(&mut self) {
        self.baseline = Some(self.checkpoint());
    }

    pub fn checkpoint(&self) -> StagedOutputCheckpoint {
        StagedOutputCheckpoint {
            tokens: self.tokens.clone(),
            text: self.text.clone(),
            raw: self.raw.clone(),
            input: self.input,
            stream: self.stream.clone(),
            terminal: self.terminal.clone(),
        }
    }

    pub fn restore(&mut self, checkpoint: StagedOutputCheckpoint) {
        self.tokens = checkpoint.tokens;
        self.text = checkpoint.text;
        self.raw = checkpoint.raw;
        self.input = checkpoint.input;
        self.stream = checkpoint.stream;
        self.terminal = checkpoint.terminal;
    }

    pub fn abort_turn(&mut self) {
        if let Some(baseline) = self.baseline.take() {
            self.restore(baseline);
        }
    }

    pub fn commit_turn(&mut self) {
        self.baseline = None;
    }

    fn observe_stream(
        &mut self,
        chunk: &str,
    ) -> Result<StagedOutputObservation, StagedOutputObservationError> {
        match self.stream.push(self.protocol, chunk) {
            ToolParseOutcome::NoCall | ToolParseOutcome::Incomplete => {
                Ok(StagedOutputObservation::Continue)
            }
            ToolParseOutcome::CompleteSoFar(calls) => {
                Ok(StagedOutputObservation::CompleteSoFar(calls))
            }
            ToolParseOutcome::TerminalComplete(_) => {
                unreachable!("incremental parsing cannot invent a terminal boundary")
            }
            ToolParseOutcome::Malformed(reason) => {
                let (identity, version) = self.protocol.declaration();
                Err(StagedOutputObservationError::Malformed {
                    identity,
                    version,
                    reason,
                })
            }
        }
    }

    fn decode_tokens(&self, tokenizer: &Tokenizer) -> Result<String, StagedOutputObservationError> {
        crate::tokenization::decode_output(tokenizer, &self.tokens, &self.preserved_tokens)
        .map_err(|source| self.decode_error(source.context(format!(
            "decode staged tool token {:?} through tokenizer.decoder ({} generated tokens); repair the decoder or vocabulary",
            self.tokens.last(), self.tokens.len()
        ))))
    }

    fn decode_error(&self, source: anyhow::Error) -> StagedOutputObservationError {
        let (identity, version) = self.protocol.declaration();
        StagedOutputObservationError::Decode {
            identity,
            version,
            source,
        }
    }

    fn validate_policy(
        &self,
        outcome: &ToolParseOutcome,
        boundary: &'static str,
    ) -> Result<(), StagedOutputObservationError> {
        let reason = match (&self.policy, outcome) {
            (ToolCallPolicy::Required, ToolParseOutcome::NoCall) => {
                Some("the model produced no tool call, but at least one was required".to_string())
            }
            (ToolCallPolicy::Specific { function }, ToolParseOutcome::NoCall) => Some(format!(
                "the model produced no tool call, but function {function:?} was required"
            )),
            (
                ToolCallPolicy::Specific { function },
                ToolParseOutcome::CompleteSoFar(calls) | ToolParseOutcome::TerminalComplete(calls),
            ) => {
                let observed = calls
                    .iter()
                    .map(|call| call.name.as_str())
                    .collect::<std::collections::BTreeSet<_>>();
                (!(observed.len() == 1 && observed.contains(function.as_str()))).then(|| {
                    format!(
                        "function {function:?} was required, but parsed function name(s) were \
                         {observed:?}"
                    )
                })
            }
            _ => None,
        };
        let Some(reason) = reason else {
            return Ok(());
        };
        let (identity, version) = self.protocol.declaration();
        Err(StagedOutputObservationError::Policy {
            identity,
            version,
            boundary,
            reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn protocol(identity: &str) -> ToolProtocol {
        ToolProtocol::from_declaration(&onnx_genai_metadata::ToolProtocolDeclaration {
            identity: identity.to_string(),
            version: "v1".to_string(),
        })
        .unwrap()
    }

    fn projected_observer(prefix: &str) -> ToolCallStagedOutputObserver {
        let template = crate::response_template::ResponseTemplate::compile(&serde_json::json!({
            "start_anchor":"A", "fields":{
                "reasoning_content":{"open":"[r]","close":"[/r]"},
                "content":{"optional":false}
            }
        }))
        .unwrap();
        let projection = crate::response_template::ResponseProjection::resolve(
            template,
            prefix.to_owned(),
            4096,
            None,
        )
        .unwrap();
        ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        )
        .with_projection(projection, None)
        .unwrap()
    }

    #[test]
    fn projected_tool_observer_ignores_reasoning_and_restores_both_text_histories() {
        let mut observer = projected_observer("A[r]prefill");
        observer.begin_turn();
        let fake = r#"<tool_call>{"name":"example","arguments":{}}</tool_call>"#;
        assert_eq!(
            observer.observe_text(fake).unwrap(),
            StagedOutputObservation::Continue
        );
        let checkpoint = observer.checkpoint();
        observer
            .observe_text(r#"[/r]<tool_call>{"name":"read","arguments":{}}</tool_call>"#)
            .unwrap();
        assert!(
            matches!(observer.finish("EOS").unwrap(), StagedOutputObservation::TerminalComplete(ref calls) if calls.len()==1 && calls[0].name=="read")
        );
        observer.restore(checkpoint);
        observer
            .observe_text(r#"[/r]<tool_call>{"name":"write","arguments":{}}</tool_call>"#)
            .unwrap();
        assert!(
            matches!(observer.finish("EOS").unwrap(), StagedOutputObservation::TerminalComplete(ref calls) if calls.len()==1 && calls[0].name=="write")
        );
        observer.abort_turn();
        assert!(observer.committed_calls().is_empty());
        observer
            .observe_text(
                r#"different thought[/r]<tool_call>{"name":"retry","arguments":{}}</tool_call>"#,
            )
            .unwrap();
        assert!(
            matches!(observer.finish("retry EOS").unwrap(), StagedOutputObservation::TerminalComplete(ref calls) if calls.len()==1 && calls[0].name=="retry")
        );
    }

    #[test]
    fn projected_tool_observer_consumes_assistant_tool_prefill_and_special_closers() {
        let mut observer =
            projected_observer(r#"A[r]thought[/r]<tool_call>{"name":"read","arguments":{"#);
        observer.observe_text("}}</tool_call>").unwrap();
        assert!(
            matches!(observer.finish("EOS").unwrap(), StagedOutputObservation::TerminalComplete(ref calls) if calls.len()==1 && calls[0].name=="read")
        );
        let tokenizer = Tokenizer::from_file(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-tool-call/tokenizer.json"),
        )
        .unwrap();
        let template = crate::response_template::ResponseTemplate::compile(&serde_json::json!({
            "start_anchor":"A","fields":{"reasoning_content":{"open":"[r]","close":"<eos>"},"content":{}}
        })).unwrap();
        let projection = crate::response_template::ResponseProjection::resolve(
            template,
            "A[r]thought".to_owned(),
            4096,
            Some(&tokenizer),
        )
        .unwrap();
        let mut observer = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        )
        .with_projection(projection, Some(&tokenizer))
        .unwrap();
        assert_eq!(
            observer.observe_tokens(&tokenizer, &[3]).unwrap(),
            StagedOutputObservation::Continue
        );
        observer.observe_tokens(&tokenizer, &[22]).unwrap();
        assert!(
            matches!(observer.finish("EOS").unwrap(), StagedOutputObservation::TerminalComplete(ref calls) if calls.len()==1 && calls[0].name=="weather")
        );
    }

    #[test]
    fn projected_tool_observer_preserves_required_field_and_buffer_error_causes() {
        let mut observer = projected_observer("A[r]thought");
        observer.observe_text("x").unwrap();
        let error = observer.finish("EOS").unwrap_err();
        assert!(format!("{error:#}").contains("required field"), "{error:#}");
        let mut observer = projected_observer("A[r]thought");
        observer.begin_turn();
        let error = observer.observe_text(&"x".repeat(4096)).unwrap_err();
        assert!(format!("{error:#}").contains("4096"), "{error:#}");
        assert!(std::error::Error::source(&error).is_some());
        observer.abort_turn();
        observer
            .observe_text(r#"retry[/r]<tool_call>{"name":"retry","arguments":{}}</tool_call>"#)
            .unwrap();
        assert!(
            matches!(observer.finish("retry EOS").unwrap(), StagedOutputObservation::TerminalComplete(ref calls) if calls[0].name=="retry")
        );
    }

    #[test]
    fn projected_tool_observer_retains_split_unicode_boundaries_and_deferred_decoders() {
        use serde_json::json;
        for byte_level in [false, true] {
            for deferred in [false, true] {
                let mut tokenizer: serde_json::Value = serde_json::from_str(include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../tests/fixtures/tiny-llm/tokenizer.json"
                )))
                .unwrap();
                let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
                for (id, text) in [
                    (4, if byte_level { "Ã" } else { "<0xC3>" }),
                    (5, if byte_level { "©" } else { "<0xA9>" }),
                    (
                        7,
                        if deferred {
                            "<tool_call>{\"name\":\"weather\",\"arguments\":{}}</tool_call>x"
                        } else {
                            "<tool_call>{\"name\":\"weather\",\"arguments\":{}}</tool_call>"
                        },
                    ),
                ] {
                    let old = vocab
                        .iter()
                        .find(|(_, value)| **value == json!(id))
                        .unwrap()
                        .0
                        .clone();
                    vocab.remove(&old);
                    vocab.insert(text.to_owned(), json!(id));
                }
                tokenizer["model"] = json!({"type":"BPE","unk_token":"<unk>","vocab":tokenizer["model"]["vocab"],"merges":[],"byte_fallback":!byte_level});
                let decoder = if byte_level {
                    json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false})
                } else {
                    json!({"type":"ByteFallback"})
                };
                tokenizer["decoder"] = if deferred {
                    json!({"type":"Sequence","decoders":[decoder,{"type":"Replace","pattern":{"Regex":"x+"},"content":""}]})
                } else {
                    decoder
                };
                let scratch = tempfile::tempdir().unwrap();
                let path = scratch.path().join("tokenizer.json");
                std::fs::write(&path, tokenizer.to_string()).unwrap();
                let tokenizer = Tokenizer::from_file(path).unwrap();
                let template = crate::response_template::ResponseTemplate::compile(
                    &json!({"start_anchor":"A","fields":{"content":{"close":"é"}}}),
                )
                .unwrap();
                let projection = crate::response_template::ResponseProjection::resolve(
                    template,
                    "A".to_owned(),
                    4096,
                    Some(&tokenizer),
                )
                .unwrap();
                let mut observer = ToolCallStagedOutputObserver::new(
                    protocol("tagged-json"),
                    ToolCallPolicy::Auto,
                    Default::default(),
                )
                .with_projection(projection, Some(&tokenizer))
                .unwrap();
                let observed = observer.observe_tokens(&tokenizer, &[7]).unwrap();
                if deferred {
                    assert_eq!(observed, StagedOutputObservation::Continue);
                    assert!(format!("{observer:?}").contains("DeferredCanonicalTokens"));
                } else {
                    assert!(
                        matches!(observed, StagedOutputObservation::CompleteSoFar(ref calls) if calls[0].name=="weather")
                    );
                    observer.observe_tokens(&tokenizer, &[4]).unwrap();
                    let checkpoint = observer.checkpoint();
                    observer.observe_tokens(&tokenizer, &[5]).unwrap();
                    assert!(
                        matches!(observer.finish("EOS").unwrap(), StagedOutputObservation::TerminalComplete(ref calls) if calls[0].name=="weather")
                    );
                    observer.restore(checkpoint);
                    observer.observe_tokens(&tokenizer, &[5]).unwrap();
                }
                assert!(
                    matches!(observer.finish("EOS").unwrap(), StagedOutputObservation::TerminalComplete(ref calls) if calls.len()==1 && calls[0].name=="weather")
                );
            }
        }
    }

    #[test]
    fn projected_tool_observer_rejects_mixed_inputs_and_fences_decoder_panics() {
        let tokenizer = Tokenizer::from_file(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-llm/tokenizer.json"),
        )
        .unwrap();
        let mut value: serde_json::Value =
            serde_json::from_str(&tokenizer.inner().to_string(false).unwrap()).unwrap();
        value["model"]["vocab"]
            .as_object_mut()
            .unwrap()
            .remove("hello");
        value["model"]["vocab"]["x"] = serde_json::json!(4);
        value["decoder"] = serde_json::json!({"type":"Strip","content":"x","start":0,"stop":100});
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, value.to_string()).unwrap();
        let tokenizer = Tokenizer::from_file(path).unwrap();
        let broken_frame = crate::response_template::ResponseTemplate::compile(
            &serde_json::json!({"start_anchor":"A","fields":{"content":{"open":"x"}}}),
        )
        .unwrap();
        let error = crate::response_template::ResponseProjection::resolve(
            broken_frame,
            "A".to_owned(),
            4096,
            Some(&tokenizer),
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("tokenizer.decoder"),
            "{error:#}"
        );
        let template = crate::response_template::ResponseTemplate::compile(
            &serde_json::json!({"start_anchor":"A","fields":{"content":{}}}),
        )
        .unwrap();
        assert!(
            crate::response_template::ResponseProjection::resolve(
                template.clone(),
                "A".to_owned(),
                0,
                Some(&tokenizer)
            )
            .unwrap_err()
            .to_string()
            .contains("positive")
        );
        let projection = crate::response_template::ResponseProjection::resolve(
            template,
            "A".to_owned(),
            4096,
            Some(&tokenizer),
        )
        .unwrap();
        let mut observer = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        )
        .with_projection(projection.clone(), Some(&tokenizer))
        .unwrap();
        let error = observer.observe_tokens(&tokenizer, &[4]).unwrap_err();
        assert!(matches!(error, StagedOutputObservationError::Decode { .. }));
        assert!(
            format!("{error:#}").contains("tokenizer.decoder"),
            "{error:#}"
        );
        assert!(format!("{error:#}").contains("Some(4)"), "{error:#}");
        assert!(
            observer
                .observe_text("text")
                .unwrap_err()
                .to_string()
                .contains("cannot mix")
        );
        let observer = projected_observer("A[r]thought");
        assert!(
            observer
                .with_projection(projection, None)
                .unwrap_err()
                .to_string()
                .contains("fresh observer")
        );
    }

    #[test]
    fn incomplete_staged_text_keeps_the_loop_running() {
        let mut observer = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        );
        assert_eq!(
            observer
                .observe_text(r#"<tool_call>{"name":"read","arguments":{"path":"src/"#)
                .unwrap(),
            StagedOutputObservation::Continue
        );
        assert!(matches!(
            observer.finish("semantic commit"),
            Err(StagedOutputObservationError::Incomplete { .. })
        ));
    }

    #[test]
    fn complete_calls_are_pending_until_an_independent_terminal_boundary() {
        let mut observer = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        );
        let StagedOutputObservation::CompleteSoFar(calls) = observer
            .observe_text(r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#)
            .unwrap()
        else {
            panic!("complete envelope must remain pending");
        };
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert!(observer.stop_reason().is_none());
        assert!(matches!(
            observer.finish("EOS boundary").unwrap(),
            StagedOutputObservation::TerminalComplete(ref calls) if calls.len() == 1
        ));
        assert_eq!(
            observer.stop_reason().unwrap().finish_reason(),
            FinishReason::ToolCalls
        );

        let mut observer = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        );
        let StagedOutputObservation::CompleteSoFar(calls) = observer
            .observe_text(concat!(
                r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#,
                "\n",
                r#"<tool_call>{"name":"write","arguments":{"path":"x"}}</tool_call>"#
            ))
            .unwrap()
        else {
            panic!("complete envelope sequence must remain pending");
        };
        assert_eq!(calls.len(), 2);
        assert!(matches!(
            observer.finish("semantic commit").unwrap(),
            StagedOutputObservation::TerminalComplete(ref calls) if calls.len() == 2
        ));
    }

    #[test]
    fn malformed_text_is_a_typed_precommit_failure() {
        let mut observer = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        );
        assert!(matches!(
            observer.observe_text("<tool_call>{bad}</tool_call>"),
            Err(StagedOutputObservationError::Malformed { .. })
        ));
    }

    #[test]
    fn text_after_a_complete_envelope_remains_malformed() {
        let mut observer = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        );
        assert!(matches!(
            observer.observe_text(r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#),
            Ok(StagedOutputObservation::CompleteSoFar(_))
        ));
        assert!(matches!(
            observer.observe_text("not a tool envelope"),
            Err(StagedOutputObservationError::Malformed { .. })
        ));
    }

    #[test]
    fn abort_restores_incremental_decoder_and_parser_state_for_retry() {
        let mut observer = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        );
        observer.begin_turn();
        assert_eq!(
            observer
                .observe_text(r#"<tool_call>{"name":"read","arguments":{"path":"src/"#)
                .unwrap(),
            StagedOutputObservation::Continue
        );
        observer.abort_turn();
        let StagedOutputObservation::CompleteSoFar(calls) = observer
            .observe_text(r#"<tool_call>{"name":"write","arguments":{}}</tool_call>"#)
            .unwrap()
        else {
            panic!("retry must not inherit the aborted partial envelope");
        };
        assert_eq!(calls[0].name, "write");
    }

    #[test]
    fn abort_after_complete_call_does_not_duplicate_the_retry() {
        let mut observer = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        );
        observer.begin_turn();
        assert!(matches!(
            observer.observe_text(r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#),
            Ok(StagedOutputObservation::CompleteSoFar(_))
        ));
        observer.abort_turn();
        assert!(observer.stop_reason().is_none());
        let StagedOutputObservation::CompleteSoFar(calls) = observer
            .observe_text(r#"<tool_call>{"name":"write","arguments":{}}</tool_call>"#)
            .unwrap()
        else {
            panic!("retry must not retain an aborted complete call");
        };
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write");
    }

    #[test]
    fn request_policy_is_enforced_only_at_the_terminal_boundary() {
        let mut required = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Required,
            Default::default(),
        );
        assert_eq!(
            required.observe_text("ordinary assistant text").unwrap(),
            StagedOutputObservation::Continue
        );
        assert!(matches!(
            required.finish("semantic commit"),
            Err(StagedOutputObservationError::Policy { ref reason, .. })
                if reason.contains("at least one was required")
        ));

        let mut specific = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Specific {
                function: "weather".to_string(),
            },
            Default::default(),
        );
        specific
            .observe_text(r#"<tool_call>{"name":"calendar","arguments":{}}</tool_call>"#)
            .unwrap();
        assert!(matches!(
            specific.finish("semantic commit"),
            Err(StagedOutputObservationError::Policy { ref reason, .. })
                if reason.contains("weather") && reason.contains("calendar")
        ));

        let mut auto = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        );
        auto.observe_text("ordinary assistant text").unwrap();
        assert!(auto.finish("semantic commit").is_ok());
    }

    #[test]
    fn oversized_staged_output_fails_before_commit() {
        let mut observer = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        );
        let oversized = "x".repeat(onnx_genai_metadata::MAX_TOOL_PAYLOAD_BYTES + 1);
        assert!(matches!(
            observer.observe_text(&oversized),
            Err(StagedOutputObservationError::Malformed { ref reason, .. })
                if reason.contains("byte protocol limit")
        ));
    }

    #[test]
    fn token_boundaries_match_single_decode_result() -> anyhow::Result<()> {
        let tokenizer = Tokenizer::from_file(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-llm/tokenizer.json"),
        )?;
        let text = r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#;
        let tokens = tokenizer.encode(text)?;
        let expected = {
            let mut observer = ToolCallStagedOutputObserver::new(
                protocol("tagged-json"),
                ToolCallPolicy::Auto,
                Default::default(),
            );
            observer.observe_tokens(&tokenizer, &tokens)?
        };
        for split in 0..=tokens.len() {
            let mut observer = ToolCallStagedOutputObserver::new(
                protocol("tagged-json"),
                ToolCallPolicy::Auto,
                Default::default(),
            );
            observer.observe_tokens(&tokenizer, &tokens[..split])?;
            assert_eq!(
                observer.observe_tokens(&tokenizer, &tokens[split..])?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn complete_so_far_never_requests_host_stop_before_finish() -> anyhow::Result<()> {
        let mut observer = ToolCallStagedOutputObserver::new(
            protocol("tagged-json"),
            ToolCallPolicy::Auto,
            Default::default(),
        );
        for chunk in [
            r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#,
            "\n<tool_call>",
            r#"{"name":"write","arguments":{}}</tool_call>"#,
        ] {
            observer.observe_text(chunk)?;
            assert!(
                observer.stop_reason().is_none(),
                "a chunk boundary cannot terminate a v1 call sequence"
            );
        }
        assert!(matches!(
            observer.finish("committed EOS").unwrap(),
            StagedOutputObservation::TerminalComplete(ref calls)
                if calls.iter().map(|call| call.name.as_str()).eq(["read", "write"])
        ));
        assert!(matches!(
            observer.stop_reason(),
            Some(GenerationStopReason::ToolCallsReady(ref calls)) if calls.len() == 2
        ));
        Ok(())
    }
}
