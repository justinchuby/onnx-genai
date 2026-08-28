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
    BudgetExhausted,
    EosCommitted,
    ContextExhausted,
    ToolCallsReady(Vec<ToolCall>),
}

impl GenerationStopReason {
    /// The transport-facing completion label for this committed normal stop.
    pub fn finish_reason(&self) -> FinishReason {
        match self {
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
    ToolCallsReady(Vec<ToolCall>),
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
        source: onnx_genai_ort::OrtError,
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

/// Replayable decoder and parser state captured at a transaction boundary.
///
/// It contains canonical token decoding state as well as the parser's
/// incremental input. Restoring it before retry ensures token/chunk grouping
/// cannot leak an incomplete envelope into a later turn.
#[derive(Debug, Clone)]
pub struct StagedOutputCheckpoint {
    tokens: Vec<TokenId>,
    text: String,
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
    stream: ToolCallStream,
    terminal: Option<Vec<ToolCall>>,
    baseline: Option<StagedOutputCheckpoint>,
}

impl ToolCallStagedOutputObserver {
    pub fn new(protocol: ToolProtocol, policy: ToolCallPolicy) -> Self {
        debug_assert!(policy.observes_output());
        Self {
            protocol,
            policy,
            tokens: Vec::new(),
            text: String::new(),
            stream: ToolCallStream::default(),
            terminal: None,
            baseline: None,
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
    /// token. Complete is terminal for the current turn; incomplete deliberately
    /// remains nonterminal until more staged output or [`Self::finish`].
    pub fn observe_text(
        &mut self,
        chunk: &str,
    ) -> Result<StagedOutputObservation, StagedOutputObservationError> {
        self.text.push_str(chunk);
        self.observe_stream(chunk)
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
        self.tokens.extend_from_slice(tokens);
        let decoded = tokenizer
            .decode(&self.tokens)
            .map_err(|source| self.decode_error(source))?;
        if let Some(chunk) = decoded.strip_prefix(&self.text) {
            let chunk = chunk.to_string();
            self.text = decoded;
            self.observe_stream(&chunk)
        } else {
            self.text = decoded;
            self.stream = ToolCallStream::default();
            self.observe_stream(&self.text.clone())
        }
    }

    /// Validate the terminal boundary. Only an unfinished envelope fails here;
    /// ordinary text remains a valid no-call completion.
    pub fn finish(&self, boundary: &'static str) -> Result<(), StagedOutputObservationError> {
        let outcome = self.stream.clone().finish(self.protocol);
        match &outcome {
            ToolParseOutcome::NoCall | ToolParseOutcome::Complete(_) => {
                self.validate_policy(&outcome, boundary)
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
            stream: self.stream.clone(),
            terminal: self.terminal.clone(),
        }
    }

    pub fn restore(&mut self, checkpoint: StagedOutputCheckpoint) {
        self.tokens = checkpoint.tokens;
        self.text = checkpoint.text;
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
            ToolParseOutcome::Complete(calls) => {
                self.terminal = Some(calls.clone());
                Ok(StagedOutputObservation::ToolCallsReady(calls))
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

    fn decode_error(&self, source: onnx_genai_ort::OrtError) -> StagedOutputObservationError {
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
            (ToolCallPolicy::Specific { function }, ToolParseOutcome::Complete(calls)) => {
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

    #[test]
    fn incomplete_staged_text_keeps_the_loop_running() {
        let mut observer =
            ToolCallStagedOutputObserver::new(protocol("tagged-json"), ToolCallPolicy::Auto);
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
    fn completed_single_and_multiple_calls_are_normal_stops_without_publication() {
        let mut observer =
            ToolCallStagedOutputObserver::new(protocol("tagged-json"), ToolCallPolicy::Auto);
        let StagedOutputObservation::ToolCallsReady(calls) = observer
            .observe_text(r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#)
            .unwrap()
        else {
            panic!("complete envelope must stop normally");
        };
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert!(matches!(
            observer.stop_reason(),
            Some(GenerationStopReason::ToolCallsReady(ref calls))
                if calls.len() == 1
        ));
        assert_eq!(
            observer.stop_reason().unwrap().finish_reason(),
            FinishReason::ToolCalls
        );

        let mut observer =
            ToolCallStagedOutputObserver::new(protocol("tagged-json"), ToolCallPolicy::Auto);
        let StagedOutputObservation::ToolCallsReady(calls) = observer
            .observe_text(concat!(
                r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#,
                "\n",
                r#"<tool_call>{"name":"write","arguments":{"path":"x"}}</tool_call>"#
            ))
            .unwrap()
        else {
            panic!("complete envelope sequence must stop normally");
        };
        assert_eq!(calls.len(), 2);
        assert!(observer.finish("semantic commit").is_ok());
    }

    #[test]
    fn malformed_text_is_a_typed_precommit_failure() {
        let mut observer =
            ToolCallStagedOutputObserver::new(protocol("tagged-json"), ToolCallPolicy::Auto);
        assert!(matches!(
            observer.observe_text("<tool_call>{bad}</tool_call>"),
            Err(StagedOutputObservationError::Malformed { .. })
        ));
    }

    #[test]
    fn text_after_a_complete_envelope_remains_malformed() {
        let mut observer =
            ToolCallStagedOutputObserver::new(protocol("tagged-json"), ToolCallPolicy::Auto);
        assert!(matches!(
            observer.observe_text(r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#),
            Ok(StagedOutputObservation::ToolCallsReady(_))
        ));
        assert!(matches!(
            observer.observe_text("not a tool envelope"),
            Err(StagedOutputObservationError::Malformed { .. })
        ));
    }

    #[test]
    fn abort_restores_incremental_decoder_and_parser_state_for_retry() {
        let mut observer =
            ToolCallStagedOutputObserver::new(protocol("tagged-json"), ToolCallPolicy::Auto);
        observer.begin_turn();
        assert_eq!(
            observer
                .observe_text(r#"<tool_call>{"name":"read","arguments":{"path":"src/"#)
                .unwrap(),
            StagedOutputObservation::Continue
        );
        observer.abort_turn();
        let StagedOutputObservation::ToolCallsReady(calls) = observer
            .observe_text(r#"<tool_call>{"name":"write","arguments":{}}</tool_call>"#)
            .unwrap()
        else {
            panic!("retry must not inherit the aborted partial envelope");
        };
        assert_eq!(calls[0].name, "write");
    }

    #[test]
    fn abort_after_complete_call_does_not_duplicate_the_retry() {
        let mut observer =
            ToolCallStagedOutputObserver::new(protocol("tagged-json"), ToolCallPolicy::Auto);
        observer.begin_turn();
        assert!(matches!(
            observer.observe_text(r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#),
            Ok(StagedOutputObservation::ToolCallsReady(_))
        ));
        observer.abort_turn();
        assert!(observer.stop_reason().is_none());
        let StagedOutputObservation::ToolCallsReady(calls) = observer
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
        let mut required =
            ToolCallStagedOutputObserver::new(protocol("tagged-json"), ToolCallPolicy::Required);
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
        );
        specific
            .observe_text(r#"<tool_call>{"name":"calendar","arguments":{}}</tool_call>"#)
            .unwrap();
        assert!(matches!(
            specific.finish("semantic commit"),
            Err(StagedOutputObservationError::Policy { ref reason, .. })
                if reason.contains("weather") && reason.contains("calendar")
        ));

        let mut auto =
            ToolCallStagedOutputObserver::new(protocol("tagged-json"), ToolCallPolicy::Auto);
        auto.observe_text("ordinary assistant text").unwrap();
        assert!(auto.finish("semantic commit").is_ok());
    }

    #[test]
    fn oversized_staged_output_fails_before_commit() {
        let mut observer =
            ToolCallStagedOutputObserver::new(protocol("tagged-json"), ToolCallPolicy::Auto);
        let oversized = "x".repeat(onnx_genai_metadata::MAX_TOOL_PAYLOAD_BYTES + 1);
        assert!(matches!(
            observer.observe_text(&oversized),
            Err(StagedOutputObservationError::Malformed { ref reason, .. })
                if reason.contains("byte protocol limit")
        ));
    }

    #[test]
    fn token_boundaries_match_one_canonical_decode() -> anyhow::Result<()> {
        let tokenizer = Tokenizer::from_file(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-llm/tokenizer.json"),
        )?;
        let text = r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#;
        let tokens = tokenizer.encode(text)?;
        let expected = {
            let mut observer =
                ToolCallStagedOutputObserver::new(protocol("tagged-json"), ToolCallPolicy::Auto);
            observer.observe_tokens(&tokenizer, &tokens)?
        };
        for split in 0..=tokens.len() {
            let mut observer =
                ToolCallStagedOutputObserver::new(protocol("tagged-json"), ToolCallPolicy::Auto);
            observer.observe_tokens(&tokenizer, &tokens[..split])?;
            assert_eq!(
                observer.observe_tokens(&tokenizer, &tokens[split..])?,
                expected
            );
        }
        Ok(())
    }
}
