//! Native, session-owned realtime execution. Transport framing lives in server.
use crate::{
    Engine, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest, SessionId,
};
use anyhow::{Context, ensure};
use onnx_genai_metadata::{
    RealtimeMode,
    realtime::{RealtimeContract, RealtimeOperation, realtime_contract},
};
use onnx_genai_ort::Value;

/// One admitted native quantum. Audio is mono normalized PCM at the contract's
/// input rate; token inputs have already been assembled with package prompting.
pub enum RealtimeInput<'a> {
    Audio(&'a [f32]),
    Text(&'a [u32]),
    Commit,
    Generate(&'a [u32]),
    Poll,
    Cancel,
}

/// Remaining response allowance supplied with each quantum. Unlimited duplex execution never
/// inherits an ordinary GenerateOptions default.
#[derive(Debug, Clone, Copy)]
pub enum RealtimeTokenBudget {
    Unlimited,
    Remaining(u64),
}
impl RealtimeTokenBudget {
    fn wire(self) -> anyhow::Result<i64> {
        match self { Self::Unlimited=>Ok(-1),Self::Remaining(tokens)=>i64::try_from(tokens).context("realtime output token budget exceeds int64; use Unlimited or a representable remaining count") }
    }
}

#[derive(Debug)]
pub struct RealtimeOutput {
    pub tokens: Vec<u32>,
    pub audio: Vec<f32>,
    pub done: bool,
    pub needs_input: bool,
    /// Counts for this quantum, not a fabricated duration-based estimate.
    pub prompt_tokens: usize,
    pub output_tokens: usize,
}

/// Exclusive engine borrow and native session lifetime are the same lifetime.
/// Neither transport IDs nor clients can forge this session authority.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SessionHealth {
    Ready,
    RequiresReset,
}
pub struct RealtimeSession<'a> {
    engine: &'a mut Engine,
    id: SessionId,
    contract: RealtimeContract,
    health: SessionHealth,
    grammar: Option<RealtimeGrammar>,
}

struct RealtimeGrammar {
    constraint: RealtimeConstraint,
    context: crate::logits::ProcessorContext,
    vocabulary: usize,
    endings: Vec<u32>,
    text_ended: bool,
}
enum RealtimeConstraint {
    Logical(Box<crate::response_constraint::ContentConstraint>),
    Raw(crate::ProcessorChain),
}
impl RealtimeGrammar {
    fn mask(&self) -> anyhow::Result<Vec<bool>> {
        if self.text_ended {
            return Ok(vec![false; self.vocabulary]);
        }
        match &self.constraint {
            RealtimeConstraint::Logical(constraint) => {
                use crate::logits::Constraint as _;
                let candidates = (0..self.vocabulary)
                    .map(|id| crate::logits::TokenCandidate {
                        token_id: id as u32,
                        text: String::new(),
                        is_eos: self.endings.contains(&(id as u32)),
                    })
                    .collect::<Vec<_>>();
                let mask = constraint.allowed_next_tokens(&self.context, &candidates)?;
                ensure!(
                    mask.iter().any(|allowed| *allowed) || constraint.can_finish(&self.context)?,
                    "realtime grammar admits no token before logical completion; repair the grammar or tokenizer vocabulary"
                );
                Ok(mask)
            }
            RealtimeConstraint::Raw(chain) => {
                let mut logits = vec![0.0; self.vocabulary];
                chain.process(&mut logits, &self.context)?;
                Ok(logits.into_iter().map(|value| value.is_finite()).collect())
            }
        }
    }
    fn can_finish(&self) -> anyhow::Result<bool> {
        if self.text_ended {
            return Ok(true);
        }
        match &self.constraint {
            RealtimeConstraint::Logical(constraint) => constraint.can_finish(&self.context),
            RealtimeConstraint::Raw(_) => {
                let next = self.mask()?;
                Ok(self
                    .endings
                    .iter()
                    .any(|id| next.get(*id as usize) == Some(&true)))
            }
        }
    }
}

impl Engine {
    pub fn open_realtime_session(
        &mut self,
        mode: RealtimeMode,
        initial_tokens: &[u32],
        control: crate::GenerationControl,
    ) -> anyhow::Result<(RealtimeSession<'_>, RealtimeOutput)> {
        let contract = realtime_contract(self.metadata()).map_err(anyhow::Error::msg)?
            .context("model has no realtime quantum profile; select a package declaring native realtime session inputs and outputs")?;
        ensure!(
            contract.supports(mode),
            "realtime profile '{}' does not support mode {mode:?}; select a declared mode or compatible model",
            contract.profile()
        );
        let id = self
            .create_session()
            .context("opening native realtime workflow session")?;
        let mut session = RealtimeSession {
            engine: self,
            id,
            contract,
            health: SessionHealth::Ready,
            grammar: None,
        };
        let operation = match mode {
            RealtimeMode::Realtime => RealtimeOperation::StartRealtime,
            RealtimeMode::Transcription => RealtimeOperation::StartTranscription,
            RealtimeMode::Duplex => RealtimeOperation::StartDuplex,
        };
        let output = session.invoke(
            operation,
            &[],
            initial_tokens,
            RealtimeTokenBudget::Unlimited,
            control,
        )?;
        Ok((session, output))
    }
}

impl RealtimeSession<'_> {
    /// Install a fresh response-local grammar before Generate. Native graphs
    /// must explicitly consume the declared token mask before selecting tokens.
    pub fn set_constraint(
        &mut self,
        constraint: Option<crate::GenerateConstraint>,
    ) -> anyhow::Result<()> {
        let grammar = if let Some(constraint) = constraint {
            ensure!(
                self.contract.spec().inputs.token_mask.is_some(),
                "realtime constrained sampling requires profiles.{}.realtime.inputs.token_mask; export a workflow that masks native logits before selecting one text token per quantum",
                self.contract.profile()
            );
            let tokenizer = self.engine.workflow_runtime().package_tokenizer().context("realtime constrained sampling requires the workflow package tokenizer; declare and load its tokenizer artifact")?;
            let endings = self.contract.ending_tokens().to_vec();
            let mut options = GenerateOptions {
                greedy: true,
                constraint: Some(constraint),
                eos_token_id: endings.first().copied(),
                eos_token_ids: endings.clone(),
                ..Default::default()
            };
            if let Some(declaration) = self
                .engine
                .metadata()
                .package
                .as_ref()
                .and_then(|p| p.tool_protocol.as_ref())
            {
                let protocol = onnx_genai_metadata::ToolProtocol::from_declaration(declaration)?;
                for marker in protocol.syntax_markers() {
                    options
                        .preserved_tokens
                        .extend(tokenizer.encode_with_options(marker, Default::default())?);
                }
            }
            // Realtime publishes canonical streaming decoder text. Token-local
            // byte approximations cannot model contextual whitespace or prove
            // completion, so text grammars use the shared logical cursor.
            let text = |grammar: &crate::config::TextGrammar, prefix: &str| {
                crate::response_constraint::ContentConstraint::text(
                    tokenizer,
                    grammar,
                    crate::text_projection::ProjectionKind::Plain,
                    prefix,
                    options.max_constraint_bytes,
                    endings.clone(),
                    &options.effective_preserved_tokens(),
                )
            };
            let json = |schema: &serde_json::Value, prefix: &str| {
                crate::response_constraint::ContentConstraint::json_schema(
                    tokenizer,
                    schema,
                    crate::text_projection::ProjectionKind::Plain,
                    prefix,
                    options.max_constraint_bytes,
                    endings.clone(),
                    &options.effective_preserved_tokens(),
                )
            };
            let logical = match options.constraint.as_ref() {
                Some(crate::GenerateConstraint::XmlToolCall { grammar, prefix }) => Some(
                    crate::response_constraint::ContentConstraint::xml_tool_call(
                        tokenizer,
                        grammar,
                        crate::text_projection::ProjectionKind::Plain,
                        prefix,
                        options.max_constraint_bytes,
                        endings.clone(),
                        &options.effective_preserved_tokens(),
                    )?,
                ),
                Some(crate::GenerateConstraint::Regex { pattern, prefix }) => Some(text(
                    &crate::config::TextGrammar::Regex(pattern.clone()),
                    prefix,
                )?),
                Some(crate::GenerateConstraint::Lark { grammar, prefix }) => Some(text(
                    &crate::config::TextGrammar::Lark(grammar.clone()),
                    prefix,
                )?),
                Some(crate::GenerateConstraint::Json { prefix }) => {
                    Some(json(&serde_json::Value::Bool(true), prefix)?)
                }
                Some(crate::GenerateConstraint::JsonSchema { schema, prefix }) => Some(json(
                    &serde_json::from_str(schema)
                        .context("parse realtime JSON Schema constraint")?,
                    prefix,
                )?),
                _ => None,
            }
            .map(Box::new);
            let constraint = if let Some(constraint) = logical {
                RealtimeConstraint::Logical(constraint)
            } else {
                ensure!(
                    !endings.is_empty(),
                    "raw realtime grammar requires package tokenizer EOS IDs to prove token grammar completion"
                );
                RealtimeConstraint::Raw(crate::processors::build_processor_chain(
                    &options,
                    Some(tokenizer),
                    false,
                )?)
            };
            let grammar = RealtimeGrammar {
                constraint,
                context: Default::default(),
                vocabulary: usize::try_from(
                    tokenizer
                        .inner()
                        .get_vocab(true)
                        .values()
                        .copied()
                        .max()
                        .context("realtime constraint tokenizer vocabulary is empty")?,
                )?
                .checked_add(1)
                .context("realtime tokenizer ID span exceeds address space")?,
                endings,
                text_ended: false,
            };
            ensure!(
                grammar.mask()?.iter().any(|allowed| *allowed),
                "realtime constraint admits no tokenizer tokens; repair the grammar or vocabulary"
            );
            Some(grammar)
        } else {
            None
        };
        self.grammar = grammar;
        Ok(())
    }

    pub fn contract(&self) -> &RealtimeContract {
        &self.contract
    }

    pub fn step(
        &mut self,
        input: RealtimeInput<'_>,
        budget: RealtimeTokenBudget,
        control: crate::GenerationControl,
    ) -> anyhow::Result<RealtimeOutput> {
        let (operation, audio, tokens) = match input {
            RealtimeInput::Audio(audio) => {
                ensure!(
                    !audio.is_empty() && audio.len() <= self.contract.spec().input_frame_samples,
                    "realtime audio quantum has {} samples; provide 1..={} mono samples at {} Hz",
                    audio.len(),
                    self.contract.spec().input_frame_samples,
                    self.contract.spec().input_sample_rate_hz
                );
                ensure!(
                    audio
                        .iter()
                        .all(|value| value.is_finite() && (-1.0..=1.0).contains(value)),
                    "realtime audio quantum contains nonfinite or out-of-range PCM; normalize samples to [-1, 1]"
                );
                (RealtimeOperation::Audio, audio, &[][..])
            }
            RealtimeInput::Text(tokens) => (RealtimeOperation::Text, &[][..], tokens),
            RealtimeInput::Commit => (RealtimeOperation::Commit, &[][..], &[][..]),
            RealtimeInput::Generate(tokens) => (RealtimeOperation::Generate, &[][..], tokens),
            RealtimeInput::Poll => (RealtimeOperation::Poll, &[][..], &[][..]),
            RealtimeInput::Cancel => (RealtimeOperation::Cancel, &[][..], &[][..]),
        };
        self.invoke(operation, audio, tokens, budget, control)
    }

    /// Rebuild model-authored conditioning before transport freezes a session,
    /// or between independent transcription turns. Reset is native state reset,
    /// not a fabricated transport acknowledgement.
    pub fn configure(
        &mut self,
        mode: RealtimeMode,
        tokens: &[u32],
        control: crate::GenerationControl,
    ) -> anyhow::Result<RealtimeOutput> {
        ensure!(
            self.contract.supports(mode),
            "realtime profile '{}' does not support mode {mode:?}",
            self.contract.profile()
        );
        self.health = SessionHealth::RequiresReset;
        self.grammar = None;
        self.engine
            .reset_session(self.id)
            .context("resetting native realtime conditioning state")?;
        self.health = SessionHealth::Ready;
        let operation = match mode {
            RealtimeMode::Realtime => RealtimeOperation::StartRealtime,
            RealtimeMode::Transcription => RealtimeOperation::StartTranscription,
            RealtimeMode::Duplex => RealtimeOperation::StartDuplex,
        };
        self.invoke(
            operation,
            &[],
            tokens,
            RealtimeTokenBudget::Unlimited,
            control,
        )
    }

    fn invoke(
        &mut self,
        operation: RealtimeOperation,
        audio: &[f32],
        tokens: &[u32],
        budget: RealtimeTokenBudget,
        control: crate::GenerationControl,
    ) -> anyhow::Result<RealtimeOutput> {
        ensure!(
            self.health == SessionHealth::Ready,
            "realtime session requires native reset after a failed quantum; configure a fresh conditioning context or close the session before submitting more input"
        );
        let mask = self
            .grammar
            .as_ref()
            .map(RealtimeGrammar::mask)
            .transpose()?;
        let inputs = &self.contract.spec().inputs;
        let budget_wire = budget.wire()?;
        let mut request = PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(Vec::new()),
            // This is a bounded workflow quantum, never an ordinary generation
            // loop. Its authored control flow decides what one invocation does.
            options: GenerateOptions {
                max_new_tokens: 1,
                ..Default::default()
            },
        })
        .with_session_id(self.id.to_string())
        .with_generation_control(control)
        .with_input(
            &inputs.output_token_budget,
            Value::from_vec_i64(vec![budget_wire], &[])?,
        )
        .with_input(
            &inputs.operation,
            Value::from_vec_i64(vec![operation as i64], &[])?,
        )
        .with_input(
            &inputs.audio,
            Value::from_vec_f32(audio.to_vec(), &[i64::try_from(audio.len())?])?,
        )
        .with_input(
            &inputs.tokens,
            Value::from_vec_i64(
                tokens.iter().map(|token| i64::from(*token)).collect(),
                &[i64::try_from(tokens.len())?],
            )?,
        );
        if let Some(endpoint) = &inputs.token_mask {
            let mask = mask.as_deref().unwrap_or(&[]);
            request = request.with_input(
                endpoint,
                Value::from_raw_bytes(
                    mask.iter().map(|allowed| u8::from(*allowed)).collect(),
                    &[i64::try_from(mask.len())?],
                    onnx_genai_ort::DataType::Bool,
                )?,
            );
        }
        let execution = self.engine.run_pipeline_outputs(request).with_context(|| {
            format!(
                "realtime profile '{}' failed {operation:?} quantum",
                self.contract.profile()
            )
        });
        let outputs = match execution {
            Ok(outputs) => outputs,
            Err(error) => {
                self.health = SessionHealth::RequiresReset;
                return Err(error);
            }
        };
        let result = (|| {
            let value = |role: &str| -> anyhow::Result<&Value> {
                let name = self
                    .contract
                    .output(role)
                    .with_context(|| format!("realtime profile has no '{role}' output binding"))?;
                outputs.aggregate(name).with_context(|| format!("realtime {operation:?} quantum omitted declared output '{name}' ({role}); emit every result, including empty PCM/token vectors"))
            };
            let count = |role: &str| -> anyhow::Result<usize> {
                let value = value(role)?;
                ensure!(
                    value.shape().is_empty(),
                    "realtime {role} output must be scalar; got {:?}",
                    value.shape()
                );
                let raw = value.to_vec_i64()?[0];
                usize::try_from(raw).with_context(|| format!("realtime {role} returned invalid count {raw}; emit a nonnegative count for this quantum"))
            };
            let token_value = value("tokens")?;
            ensure!(
                token_value.shape().len() == 1,
                "realtime tokens must be a vector; got {:?}",
                token_value.shape()
            );
            let tokens: Vec<u32> = token_value.to_vec_i64()?.into_iter().map(|token| u32::try_from(token).with_context(|| format!("realtime workflow emitted invalid token ID {token}; emit representable tokenizer IDs"))).collect::<anyhow::Result<_>>()?;
            if let Some(mask) = &mask {
                if let RealtimeTokenBudget::Remaining(remaining) = budget {
                    ensure!(
                        u64::try_from(tokens.len())? <= remaining,
                        "realtime constrained quantum emitted {} text tokens with only {remaining} remaining; obey output_token_budget even when reporting output usage",
                        tokens.len()
                    );
                }
                ensure!(
                    tokens.len() <= 1,
                    "realtime constrained quantum emitted {} tokens; emit at most one text token after consuming token_mask",
                    tokens.len()
                );
                for token in &tokens {
                    ensure!(
                        mask.get(*token as usize) == Some(&true),
                        "realtime workflow emitted token {token} forbidden by token_mask; apply the mask before native sampling"
                    );
                }
            }
            let audio = if self.contract.output("audio").is_some() {
                let audio = value("audio")?;
                ensure!(
                    audio.shape().len() == 1,
                    "realtime audio must be a mono vector; got {:?}",
                    audio.shape()
                );
                let samples = audio.to_vec_f32()?;
                ensure!(
                    samples.iter().all(|sample| sample.is_finite()),
                    "realtime workflow emitted nonfinite PCM; emit finite mono samples (the PCM16 transport saturates finite peaks)"
                );
                samples
            } else {
                Vec::new()
            };
            let done = value("done")?;
            ensure!(
                done.shape().is_empty(),
                "realtime done must be a bool scalar; got {:?}",
                done.shape()
            );
            let needs_input = value("needs_input")?;
            ensure!(
                needs_input.shape().is_empty(),
                "realtime needs_input must be a bool scalar; got {:?}",
                needs_input.shape()
            );
            let needs_input = needs_input.to_vec_bool()?[0];
            let output_tokens = count("output_tokens")?;
            ensure!(
                output_tokens >= tokens.len(),
                "realtime output_tokens={output_tokens} underreports {} emitted text tokens; report actual generated-token usage so response budgets remain enforceable",
                tokens.len()
            );
            if let RealtimeTokenBudget::Remaining(remaining) = budget {
                ensure!(
                    u64::try_from(output_tokens)? <= remaining,
                    "realtime {operation:?} quantum emitted {output_tokens} output tokens with only {remaining} tokens remaining; repair the workflow to obey its explicit output_token_budget input"
                );
            }
            let done = done.to_vec_bool()?[0];
            if let Some(grammar) = self.grammar.as_mut() {
                grammar.context.generated_tokens.extend_from_slice(&tokens);
                grammar.context.step = grammar.context.generated_tokens.len();
                grammar.text_ended |= tokens
                    .last()
                    .is_some_and(|token| grammar.endings.contains(token));
                let at_response_limit = matches!(budget, RealtimeTokenBudget::Remaining(limit) if u64::try_from(output_tokens)? == limit);
                if done
                    && operation != RealtimeOperation::Cancel
                    && !grammar.text_ended
                    && !at_response_limit
                {
                    let complete = grammar.can_finish()?;
                    ensure!(
                        complete,
                        "realtime workflow completed before its constrained grammar accepted an ending; continue sampling or cancel explicitly"
                    );
                }
            }
            Ok(RealtimeOutput {
                tokens,
                audio,
                done,
                needs_input,
                prompt_tokens: count("prompt_tokens")?,
                output_tokens,
            })
        })();
        if result.as_ref().is_ok_and(|output| output.done) {
            self.grammar = None;
        }
        if result.is_err() {
            self.health = SessionHealth::RequiresReset;
        }
        result
    }
}

impl Drop for RealtimeSession<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.engine.close_session(self.id) {
            tracing::warn!(session = %self.id, error = %error, "closing native realtime session failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_realtime_rejects_ignored_masks_and_premature_completion_until_reset()
    -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime-constrained");
        for (ignored, pattern, diagnostic) in [
            (true, "hello", "forbidden by token_mask"),
            (
                false,
                "hello world",
                "completed before its constrained grammar",
            ),
        ] {
            let package = tempfile::tempdir()?;
            for file in ["tokenizer.json", "inference_metadata.yaml"] {
                std::fs::copy(root.join(file), package.path().join(file))?;
            }
            let mut graph = std::fs::read_to_string(root.join("model.onnx.textproto"))?;
            if ignored {
                graph = graph.replace(
                    "input: \"effective_mask\" input: \"ranked_logits\"",
                    "input: \"all_allowed\" input: \"ranked_logits\"",
                );
            }
            std::fs::write(package.path().join("model.onnx.textproto"), graph)?;
            let mut engine = Engine::from_dir(package.path(), crate::EngineConfig::default())?;
            let (mut session, _) = engine.open_realtime_session(
                RealtimeMode::Realtime,
                &[],
                crate::GenerationControl::new(),
            )?;
            session.set_constraint(Some(crate::GenerateConstraint::Regex {
                pattern: pattern.into(),
                prefix: String::new(),
            }))?;
            let failure = session
                .step(
                    RealtimeInput::Generate(&[7]),
                    RealtimeTokenBudget::Unlimited,
                    crate::GenerationControl::new(),
                )
                .unwrap_err();
            assert!(format!("{failure:#}").contains(diagnostic), "{failure:#}");
            let poisoned = session
                .step(
                    RealtimeInput::Poll,
                    RealtimeTokenBudget::Unlimited,
                    crate::GenerationControl::new(),
                )
                .unwrap_err();
            assert!(poisoned.to_string().contains("requires native reset"));
            session.configure(RealtimeMode::Realtime, &[], crate::GenerationControl::new())?;
            let recovered = session.step(
                RealtimeInput::Generate(&[7]),
                RealtimeTokenBudget::Unlimited,
                crate::GenerationControl::new(),
            )?;
            assert_eq!(recovered.tokens, [31]);
        }
        Ok(())
    }

    #[test]
    fn native_realtime_advances_grammar_across_poll_quanta_and_admits_eos_only_at_completion()
    -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime-constrained");
        let package = tempfile::tempdir()?;
        for file in ["tokenizer.json", "inference_metadata.yaml"] {
            std::fs::copy(root.join(file), package.path().join(file))?;
        }
        let graph = std::fs::read_to_string(root.join("model.onnx.textproto"))?
            .replace("node { input: \"generating\" input: \"asr\"", "initializer { name: \"polling_op\" data_type: 7 int64_data: 7 }\nnode { input: \"operation\" input: \"polling_op\" output: \"polling\" op_type: \"Equal\" }\nnode { input: \"generating\" input: \"polling\" output: \"decoding\" op_type: \"Or\" }\nnode { input: \"decoding\" input: \"asr\"")
            .replace("node { input: \"generating\" output: \"candidate_tokens\"", "node { input: \"decoding\" output: \"candidate_tokens\"")
            .replace("node { input: \"responding\" input: \"cancelling\" output: \"done\" op_type: \"Or\" }", "node { input: \"tokens_out\" output: \"emitted_id\" op_type: \"ReduceSum\" attribute { name: \"keepdims\" i: 0 type: INT } }\nnode { input: \"emitted_id\" input: \"three\" output: \"ending\" op_type: \"Equal\" }\nnode { input: \"ending\" input: \"cancelling\" output: \"done\" op_type: \"Or\" }");
        std::fs::write(package.path().join("model.onnx.textproto"), graph)?;
        let mut engine = Engine::from_dir(package.path(), crate::EngineConfig::default())?;
        let (mut session, _) = engine.open_realtime_session(
            RealtimeMode::Realtime,
            &[],
            crate::GenerationControl::new(),
        )?;
        session.set_constraint(Some(crate::GenerateConstraint::Regex {
            pattern: "hello world".into(),
            prefix: String::new(),
        }))?;
        for (input, expected, done) in [
            (RealtimeInput::Generate(&[]), 4, false),
            (RealtimeInput::Poll, 5, false),
            (RealtimeInput::Poll, 3, true),
        ] {
            let output = session.step(
                input,
                RealtimeTokenBudget::Unlimited,
                crate::GenerationControl::new(),
            )?;
            assert_eq!(output.tokens, [expected]);
            assert_eq!(output.done, done);
            assert!(!output.needs_input);
        }
        assert!(session.grammar.is_none());
        Ok(())
    }

    #[test]
    fn native_realtime_budget_cannot_be_bypassed_by_underreported_usage() -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime-constrained");
        let package = tempfile::tempdir()?;
        for file in ["tokenizer.json", "inference_metadata.yaml"] {
            std::fs::copy(root.join(file), package.path().join(file))?;
        }
        let graph = std::fs::read_to_string(root.join("model.onnx.textproto"))?
            .replace(
                "input: \"responding\" input: \"output_token_budget\"",
                "input: \"responding\" input: \"unlimited\"",
            )
            .replace(
                "node { input: \"tokens_out\" output: \"count\" op_type: \"Size\" }",
                "node { input: \"zero\" output: \"count\" op_type: \"Identity\" }",
            );
        std::fs::write(package.path().join("model.onnx.textproto"), graph)?;
        let mut engine = Engine::from_dir(package.path(), crate::EngineConfig::default())?;
        let (mut session, _) = engine.open_realtime_session(
            RealtimeMode::Realtime,
            &[],
            crate::GenerationControl::new(),
        )?;
        for (remaining, diagnostic) in [(0, "only 0 remaining"), (2, "underreports")] {
            session.configure(RealtimeMode::Realtime, &[], crate::GenerationControl::new())?;
            session.set_constraint(Some(crate::GenerateConstraint::Regex {
                pattern: "hello".into(),
                prefix: String::new(),
            }))?;
            let failure = session
                .step(
                    RealtimeInput::Generate(&[]),
                    RealtimeTokenBudget::Remaining(remaining),
                    crate::GenerationControl::new(),
                )
                .unwrap_err();
            assert!(failure.to_string().contains(diagnostic), "{failure:#}");
        }
        Ok(())
    }

    #[test]
    fn native_realtime_keeps_codec_tail_after_text_eos() -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime-constrained");
        let package = tempfile::tempdir()?;
        for file in ["tokenizer.json", "inference_metadata.yaml"] {
            std::fs::copy(root.join(file), package.path().join(file))?;
        }
        let graph = std::fs::read_to_string(root.join("model.onnx.textproto"))?
            .replace("node { input: \"generating\" input: \"asr\"", "initializer { name: \"polling_op\" data_type: 7 int64_data: 7 }\nnode { input: \"operation\" input: \"polling_op\" output: \"polling\" op_type: \"Equal\" }\nnode { input: \"generating\" input: \"polling\" output: \"decoding\" op_type: \"Or\" }\nnode { input: \"decoding\" input: \"asr\"")
            .replace("node { input: \"generating\" output: \"candidate_tokens\"", "node { input: \"sampling\" output: \"candidate_tokens\"")
            .replace("node { input: \"responding\" input: \"cancelling\" output: \"done\" op_type: \"Or\" }", r#"node { input: "token_mask" output: "allowed_bits" op_type: "Cast" attribute { name: "to" i: 7 type: INT } }
node { input: "allowed_bits" output: "allowed_count" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
node { input: "allowed_count" input: "zero" output: "no_allowed" op_type: "Equal" }
node { input: "has_mask" input: "no_allowed" output: "text_finished" op_type: "And" }
node { input: "text_finished" output: "text_active" op_type: "Not" }
node { input: "decoding" input: "text_active" output: "sampling" op_type: "And" }
node { input: "text_finished" input: "cancelling" output: "done" op_type: "Or" }
initializer { name: "tail" data_type: 1 dims: 1 float_data: 0.25 }
node { input: "text_finished" output: "audio_out" op_type: "If"
attribute { name: "then_branch" type: GRAPH g { name: "codec_tail" node { input: "tail" output: "result" op_type: "Identity" } output { name: "result" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } } } } } } }
attribute { name: "else_branch" type: GRAPH g { name: "live_audio" node { input: "raw_audio" output: "result" op_type: "Identity" } output { name: "result" type { tensor_type { elem_type: 1 shape { dim { dim_param: "samples" } } } } } } }
}"#)
            .replace("output: \"audio_out\" op_type: \"Add\"", "output: \"raw_audio\" op_type: \"Add\"");
        std::fs::write(package.path().join("model.onnx.textproto"), graph)?;
        let mut engine = Engine::from_dir(package.path(), crate::EngineConfig::default())?;
        let (mut session, _) = engine.open_realtime_session(
            RealtimeMode::Realtime,
            &[],
            crate::GenerationControl::new(),
        )?;
        session.set_constraint(Some(crate::GenerateConstraint::Regex {
            pattern: "hello".into(),
            prefix: String::new(),
        }))?;
        for (input, expected) in [(RealtimeInput::Generate(&[]), 4), (RealtimeInput::Poll, 3)] {
            let output = session.step(
                input,
                RealtimeTokenBudget::Unlimited,
                crate::GenerationControl::new(),
            )?;
            assert_eq!(output.tokens, [expected]);
            assert!(!output.done);
            assert!(!output.needs_input);
        }
        let tail = session.step(
            RealtimeInput::Poll,
            RealtimeTokenBudget::Unlimited,
            crate::GenerationControl::new(),
        )?;
        assert!(tail.tokens.is_empty());
        assert_eq!(tail.audio, [0.25]);
        assert!(tail.done);
        assert!(session.grammar.is_none());
        Ok(())
    }

    #[test]
    fn native_realtime_logical_completion_does_not_require_an_eos_role() -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime-constrained");
        let package = tempfile::tempdir()?;
        for file in ["tokenizer.json", "model.onnx.textproto"] {
            std::fs::copy(root.join(file), package.path().join(file))?;
        }
        let metadata = std::fs::read_to_string(root.join("inference_metadata.yaml"))?
            .replace("eos_token_id: [3]", "eos_token_id: []");
        std::fs::write(package.path().join("inference_metadata.yaml"), metadata)?;
        let mut engine = Engine::from_dir(package.path(), crate::EngineConfig::default())?;
        let (mut session, _) = engine.open_realtime_session(
            RealtimeMode::Realtime,
            &[],
            crate::GenerationControl::new(),
        )?;
        assert!(session.contract.ending_tokens().is_empty());
        session.set_constraint(Some(crate::GenerateConstraint::Regex {
            pattern: "hello".into(),
            prefix: String::new(),
        }))?;
        let output = session.step(
            RealtimeInput::Generate(&[]),
            RealtimeTokenBudget::Unlimited,
            crate::GenerationControl::new(),
        )?;
        assert_eq!(output.tokens, [4]);
        assert!(output.done);
        Ok(())
    }

    #[test]
    fn native_realtime_masks_follow_actual_token_id_span_with_vocabulary_holes()
    -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime-constrained");
        let package = tempfile::tempdir()?;
        let mut tokenizer: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("tokenizer.json"))?)?;
        tokenizer["model"]["vocab"]["tok31"] = serde_json::json!(40);
        std::fs::write(
            package.path().join("tokenizer.json"),
            serde_json::to_vec(&tokenizer)?,
        )?;
        std::fs::write(
            package.path().join("inference_metadata.yaml"),
            std::fs::read_to_string(root.join("inference_metadata.yaml"))?
                .replace("vocab_size: 32", "vocab_size: 41"),
        )?;
        let graph = std::fs::read_to_string(root.join("model.onnx.textproto"))?
            .lines()
            .map(|line| {
                if line.starts_with("initializer { name: \"ranked_logits\"") {
                    format!(
                        "initializer {{ name: \"ranked_logits\" data_type: 1 dims: 41 {} }}",
                        (0..41)
                            .map(|i| format!("float_data: {i}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                } else if line.starts_with("initializer { name: \"all_allowed\"") {
                    format!(
                        "initializer {{ name: \"all_allowed\" data_type: 9 dims: 41 {} }}",
                        std::iter::repeat_n("int32_data: 1", 41)
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                } else {
                    line.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(package.path().join("model.onnx.textproto"), graph)?;
        let mut engine = Engine::from_dir(package.path(), crate::EngineConfig::default())?;
        let (mut session, _) = engine.open_realtime_session(
            RealtimeMode::Realtime,
            &[],
            crate::GenerationControl::new(),
        )?;
        session.set_constraint(Some(crate::GenerateConstraint::Regex {
            pattern: "tok31".into(),
            prefix: String::new(),
        }))?;
        let output = session.step(
            RealtimeInput::Generate(&[]),
            RealtimeTokenBudget::Unlimited,
            crate::GenerationControl::new(),
        )?;
        assert_eq!(output.tokens, [40]);
        assert!(output.done);
        Ok(())
    }

    #[test]
    fn native_realtime_transmits_remaining_response_budget_without_quantum_clamping()
    -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime-constrained");
        let package = tempfile::tempdir()?;
        for file in ["tokenizer.json", "inference_metadata.yaml"] {
            std::fs::copy(root.join(file), package.path().join(file))?;
        }
        let graph = std::fs::read_to_string(root.join("model.onnx.textproto"))?.replace("node { input: \"responding\" input: \"cancelling\" output: \"done\" op_type: \"Or\" }", "node { input: \"output_token_budget\" input: \"count\" output: \"limit_equal\" op_type: \"Equal\" }\nnode { input: \"limit_equal\" input: \"responding\" output: \"limit_done\" op_type: \"And\" }\nnode { input: \"limit_done\" input: \"cancelling\" output: \"done\" op_type: \"Or\" }");
        std::fs::write(package.path().join("model.onnx.textproto"), graph)?;
        let mut engine = Engine::from_dir(package.path(), crate::EngineConfig::default())?;
        let (mut session, _) = engine.open_realtime_session(
            RealtimeMode::Realtime,
            &[],
            crate::GenerationControl::new(),
        )?;
        session.set_constraint(Some(crate::GenerateConstraint::Regex {
            pattern: "hello world".into(),
            prefix: String::new(),
        }))?;
        let output = session.step(
            RealtimeInput::Generate(&[]),
            RealtimeTokenBudget::Remaining(4),
            crate::GenerationControl::new(),
        )?;
        assert_eq!(output.tokens, [4]);
        assert!(
            !output.done,
            "one quantum must not masquerade as the remaining response allowance"
        );
        assert!(
            session
                .step(
                    RealtimeInput::Cancel,
                    RealtimeTokenBudget::Unlimited,
                    crate::GenerationControl::new()
                )?
                .done
        );
        Ok(())
    }

    #[test]
    fn native_realtime_allows_budget_truncation_but_rejects_premature_model_completion()
    -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime-constrained");
        let mut engine = Engine::from_dir(&root, crate::EngineConfig::default())?;
        let (mut session, _) = engine.open_realtime_session(
            RealtimeMode::Realtime,
            &[],
            crate::GenerationControl::new(),
        )?;
        session.set_constraint(Some(crate::GenerateConstraint::Regex {
            pattern: "hello world".into(),
            prefix: String::new(),
        }))?;
        let output = session.step(
            RealtimeInput::Generate(&[]),
            RealtimeTokenBudget::Remaining(1),
            crate::GenerationControl::new(),
        )?;
        assert_eq!(output.tokens, [4]);
        assert!(output.done);
        session.configure(RealtimeMode::Realtime, &[], crate::GenerationControl::new())?;
        session.set_constraint(Some(crate::GenerateConstraint::Regex {
            pattern: "hello world".into(),
            prefix: String::new(),
        }))?;
        let failure = session
            .step(
                RealtimeInput::Generate(&[]),
                RealtimeTokenBudget::Remaining(4),
                crate::GenerationControl::new(),
            )
            .unwrap_err();
        assert!(
            failure
                .to_string()
                .contains("completed before its constrained grammar"),
            "{failure:#}"
        );
        Ok(())
    }

    #[test]
    fn native_realtime_applies_response_grammar_before_selecting_tokens() -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime-constrained");
        let mut engine = Engine::from_dir(&root, crate::EngineConfig::default())?;
        let (mut session, _) = engine.open_realtime_session(
            RealtimeMode::Realtime,
            &[],
            crate::GenerationControl::new(),
        )?;
        let baseline = session.step(
            RealtimeInput::Generate(&[7]),
            RealtimeTokenBudget::Unlimited,
            crate::GenerationControl::new(),
        )?;
        assert_eq!(baseline.tokens, [31]);
        for (pattern, expected) in [("hello", 4), ("world", 5)] {
            session.set_constraint(Some(crate::GenerateConstraint::Regex {
                pattern: pattern.into(),
                prefix: String::new(),
            }))?;
            let output = session.step(
                RealtimeInput::Generate(&[7]),
                RealtimeTokenBudget::Unlimited,
                crate::GenerationControl::new(),
            )?;
            assert_eq!(output.tokens, [expected]);
            assert!(output.done);
            assert_eq!(output.output_tokens, 1);
            assert!(
                session.grammar.is_none(),
                "completed response releases its grammar"
            );
        }
        Ok(())
    }

    #[test]
    fn native_realtime_quanta_keep_state_and_drop_closes_the_session() -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime");
        let mut engine = Engine::from_dir(&root, crate::EngineConfig::default())?;
        let closed_id;
        {
            let (mut session, initial) = engine.open_realtime_session(
                RealtimeMode::Duplex,
                &[4, 5],
                crate::GenerationControl::new(),
            )?;
            closed_id = session.id;
            assert_eq!(initial.tokens, [4, 5]);
            assert_eq!((initial.prompt_tokens, initial.output_tokens), (2, 2));
            assert!(!initial.done);
            assert!(initial.audio.is_empty());
            assert_eq!(
                session
                    .step(
                        RealtimeInput::Audio(&[0.1, 0.2]),
                        RealtimeTokenBudget::Unlimited,
                        crate::GenerationControl::new()
                    )?
                    .audio,
                [0.1, 0.2]
            );
            let next = session.step(
                RealtimeInput::Audio(&[0.2]),
                RealtimeTokenBudget::Unlimited,
                crate::GenerationControl::new(),
            )?;
            assert!((next.audio[0] - 0.5).abs() < 1e-6);
            assert!(
                !session
                    .step(
                        RealtimeInput::Poll,
                        RealtimeTokenBudget::Unlimited,
                        crate::GenerationControl::new()
                    )?
                    .done
            );
            assert!(
                session
                    .step(
                        RealtimeInput::Cancel,
                        RealtimeTokenBudget::Unlimited,
                        crate::GenerationControl::new()
                    )?
                    .done
            );
            let reset =
                session.configure(RealtimeMode::Duplex, &[6], crate::GenerationControl::new())?;
            assert_eq!(reset.tokens, [6]);
            assert_eq!(
                session
                    .step(
                        RealtimeInput::Audio(&[0.1]),
                        RealtimeTokenBudget::Unlimited,
                        crate::GenerationControl::new()
                    )?
                    .audio,
                [0.1]
            );
            let error = session
                .step(
                    RealtimeInput::Text(&[4, 5]),
                    RealtimeTokenBudget::Remaining(1),
                    crate::GenerationControl::new(),
                )
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("output_token_budget") && error.contains("only 1"),
                "{error}"
            );
            assert!(
                session
                    .step(
                        RealtimeInput::Poll,
                        RealtimeTokenBudget::Unlimited,
                        crate::GenerationControl::new()
                    )
                    .unwrap_err()
                    .to_string()
                    .contains("requires native reset")
            );
            session.configure(RealtimeMode::Duplex, &[], crate::GenerationControl::new())?;
            assert_eq!(
                session
                    .step(
                        RealtimeInput::Audio(&[0.1]),
                        RealtimeTokenBudget::Unlimited,
                        crate::GenerationControl::new()
                    )?
                    .audio,
                [0.1]
            );

            session.step(
                RealtimeInput::Audio(&[0.6]),
                RealtimeTokenBudget::Unlimited,
                crate::GenerationControl::new(),
            )?;
            assert!(
                session
                    .step(
                        RealtimeInput::Audio(&[0.6]),
                        RealtimeTokenBudget::Unlimited,
                        crate::GenerationControl::new()
                    )?
                    .audio[0]
                    > 1.0,
                "finite model peaks are preserved until PCM16 transport saturation"
            );
            assert!(
                session
                    .step(
                        RealtimeInput::Audio(&[f32::NAN]),
                        RealtimeTokenBudget::Unlimited,
                        crate::GenerationControl::new()
                    )
                    .unwrap_err()
                    .to_string()
                    .contains("PCM")
            );
            assert!(
                session
                    .step(
                        RealtimeInput::Audio(&[0.0; 321]),
                        RealtimeTokenBudget::Unlimited,
                        crate::GenerationControl::new()
                    )
                    .unwrap_err()
                    .to_string()
                    .contains("320")
            );
        }
        assert!(
            engine.session_token_count(closed_id).is_err(),
            "dropped realtime session must be removed from the canonical engine registry"
        );
        // A separately owned session starts from the model's initializer rather
        // than retaining the previous connection's codec/conversation history.
        let (mut next, _) = engine.open_realtime_session(
            RealtimeMode::Transcription,
            &[],
            crate::GenerationControl::new(),
        )?;
        assert_eq!(
            next.step(
                RealtimeInput::Audio(&[0.1]),
                RealtimeTokenBudget::Unlimited,
                crate::GenerationControl::new()
            )?
            .audio,
            [0.1]
        );
        Ok(())
    }
}
