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

/// Explicit per-quantum generation allowance. Unlimited duplex execution never
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
        let inputs = &self.contract.spec().inputs;
        let budget_wire = budget.wire()?;
        let request = PipelineGenerateRequest::new(GenerateRequest {
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
            let tokens = token_value.to_vec_i64()?.into_iter().map(|token| u32::try_from(token).with_context(|| format!("realtime workflow emitted invalid token ID {token}; emit representable tokenizer IDs"))).collect::<anyhow::Result<_>>()?;
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
            if let RealtimeTokenBudget::Remaining(remaining) = budget {
                ensure!(
                    u64::try_from(output_tokens)? <= remaining,
                    "realtime {operation:?} quantum emitted {output_tokens} output tokens with only {remaining} tokens remaining; repair the workflow to obey its explicit output_token_budget input"
                );
            }
            Ok(RealtimeOutput {
                tokens,
                audio,
                done: done.to_vec_bool()?[0],
                needs_input,
                prompt_tokens: count("prompt_tokens")?,
                output_tokens,
            })
        })();
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
