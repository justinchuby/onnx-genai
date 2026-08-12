// Copyright (c) Microsoft Corporation.
//
//! Text-to-audio (speech) synthesis for declarative TTS pipelines.
//!
//! This module turns text into a waveform by driving a composite pipeline whose
//! final stage emits audio (see `DESIGN.md` §20):
//!
//! ```text
//! decoder (autoregressive, emits audio codes) -> vocoder (final_only) -> waveform
//! ```
//!
//! It also drives the multi-decoder (`nested_autoregressive`) talker shape,
//! because [`PipelineEngine::synthesize`] handles both and publishes the
//! assembled codes into the shared pool either way.
//!
//! Everything the synthesizer needs is read from the package's declared
//! metadata: the waveform endpoint and its sample rate come from
//! `pipeline.audio`, and the final-phase component comes from `pipeline.phases`.
//! No model family, vendor, or architecture name is hardcoded (`RULES.md` §2).
//! A package that does not declare its sample rate fails with an actionable
//! error rather than being played back at a guessed pitch.

use std::path::Path;

use anyhow::{Context, Result, bail};
use onnx_genai_engine::{
    GenerateOptions, GeneratePrompt, GenerateRequest, PipelineEngine, PipelineGenerateRequest,
};
use onnx_genai_metadata::{PhaseRunOn, PipelineSpec, PipelineStrategy, PipelineStrategyKind};
use onnx_genai_ort::Tokenizer;
use onnx_genai_preprocess::audio::encode_wav_pcm16;

/// Parameters for one synthesis.
#[derive(Debug, Clone, Default)]
pub struct TextToAudioRequest {
    /// Text to speak.
    pub text: String,
    /// Maximum audio tokens (codes or frames) to generate. `None` keeps the
    /// package's declared `max_tokens`.
    pub max_new_tokens: Option<usize>,
    /// Sampling temperature. Defaults to greedy (0.0) so a synthesis is
    /// reproducible unless the caller asks otherwise.
    pub temperature: Option<f32>,
    /// Sampling seed, used only when `temperature` is above zero.
    pub seed: Option<u64>,
    /// Override for the package's declared output sample rate. Only needed for
    /// a package whose metadata omits `pipeline.audio.sample_rate`.
    pub sample_rate: Option<u32>,
}

/// A synthesized waveform.
#[derive(Debug, Clone)]
pub struct SynthesizedAudio {
    /// Interleaved samples in `[-1, 1]`.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl SynthesizedAudio {
    /// Largest absolute sample value, for range diagnostics.
    pub fn peak(&self) -> f32 {
        self.samples
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
    }

    /// Duration in seconds, for reporting.
    pub fn duration_secs(&self) -> f32 {
        let frames = self.samples.len() / usize::from(self.channels).max(1);
        frames as f32 / self.sample_rate as f32
    }

    /// Encode as 16-bit PCM WAV bytes.
    pub fn to_wav(&self) -> Result<Vec<u8>> {
        encode_wav_pcm16(&self.samples, self.sample_rate, self.channels).map_err(|error| {
            anyhow::anyhow!(
                "What: the synthesized waveform could not be encoded as WAV. \
                 Why: {error}. \
                 How: report this as a synthesis bug."
            )
        })
    }

    /// Encode as raw little-endian 16-bit PCM, without a container.
    pub fn to_pcm16(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.samples.len() * 2);
        for &sample in &self.samples {
            let scaled = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            bytes.extend_from_slice(&scaled.to_le_bytes());
        }
        bytes
    }
}

/// Save a waveform as a 16-bit PCM WAV file.
pub fn save_wav(audio: &SynthesizedAudio, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, audio.to_wav()?).with_context(|| {
        format!(
            "What: the synthesized audio could not be written to {}. \
             Why: the file could not be created. \
             How: choose an output path in a writable directory.",
            path.display()
        )
    })
}

/// Peak above which a waveform is reported as out of range. Chosen well above
/// 1.0 so ordinary transient overshoot in a correctly scaled signal stays quiet.
const OUT_OF_RANGE_PEAK: f32 = 1.5;

/// The waveform endpoint and format a package declares.
#[derive(Debug, Clone)]
struct AudioOutput {
    endpoint: Option<String>,
    final_component: Option<String>,
    sample_rate: Option<u32>,
    channels: u16,
}

fn resolve_audio_output(spec: &PipelineSpec) -> AudioOutput {
    let declared = spec.audio.as_ref();
    AudioOutput {
        endpoint: declared.and_then(|audio| audio.output.clone()),
        final_component: final_strategy_component(&spec.strategy).or_else(|| {
            spec.phases
                .iter()
                .find(|(_, phase)| phase.run_on == PhaseRunOn::FinalOnly)
                .map(|(name, _)| name.clone())
        }),
        sample_rate: declared.and_then(|audio| audio.sample_rate),
        channels: declared.and_then(|audio| audio.channels).unwrap_or(1),
    }
}

fn final_strategy_component(strategy: &PipelineStrategy) -> Option<String> {
    let loop_index = strategy
        .stages
        .iter()
        .position(|stage| strategy_contains_decode_loop(&stage.strategy))?;
    strategy.stages[loop_index + 1..]
        .last()
        .and_then(|stage| terminal_strategy_component(&stage.strategy))
}

fn strategy_contains_decode_loop(strategy: &PipelineStrategy) -> bool {
    matches!(
        strategy.kind,
        PipelineStrategyKind::Autoregressive | PipelineStrategyKind::NestedAutoregressive
    ) || strategy
        .stages
        .iter()
        .any(|stage| strategy_contains_decode_loop(&stage.strategy))
}

fn terminal_strategy_component(strategy: &PipelineStrategy) -> Option<String> {
    strategy
        .stages
        .last()
        .and_then(|stage| terminal_strategy_component(&stage.strategy))
        .or_else(|| strategy.model.clone())
}

/// Returns true when `spec` describes a pipeline that emits audio.
///
/// A package qualifies when it declares `pipeline.audio`, or when it has a
/// final strategy stage fed by an autoregressive decoder — the TTS shape.
pub fn is_text_to_audio(spec: &PipelineSpec) -> bool {
    if spec.audio.is_some() {
        return true;
    }
    let has_final_stage = final_strategy_component(&spec.strategy).is_some()
        || spec
            .phases
            .values()
            .any(|phase| phase.run_on == PhaseRunOn::FinalOnly);
    let has_decoder = strategy_contains_decode_loop(&spec.strategy);
    has_final_stage && has_decoder
}

/// Synthesize `request.text` through the TTS pipeline already loaded in `engine`.
///
/// `tokenizer` is the package's prompt tokenizer; the text is tokenized here
/// because the decoder consumes token ids, not text.
pub fn synthesize(
    engine: &mut PipelineEngine,
    tokenizer: &Tokenizer,
    request: &TextToAudioRequest,
) -> Result<SynthesizedAudio> {
    let output = resolve_audio_output(engine.spec());
    if !is_text_to_audio(engine.spec()) {
        bail!(
            "What: this package cannot be synthesized to audio. \
             Why: its pipeline declares neither `pipeline.audio` nor a `run_on: final_only` stage fed by an autoregressive decoder, so nothing produces a waveform. \
             How: point the command at a text-to-speech package."
        );
    }

    let token_ids = tokenizer.encode(&request.text).map_err(|error| {
        anyhow::anyhow!(
            "What: the text could not be tokenized. \
             Why: {error}. \
             How: verify the package's tokenizer.json matches its decoder."
        )
    })?;

    let mut options = GenerateOptions {
        temperature: request.temperature.unwrap_or(0.0),
        // Audio codes are not text: an EOS-like id in the code vocabulary would
        // truncate the waveform, so decoding runs for the declared budget.
        stop_on_eos: false,
        seed: request.seed,
        ..GenerateOptions::default()
    };
    if let Some(max_new_tokens) = request
        .max_new_tokens
        .or(engine.spec().strategy.max_tokens)
        .or_else(|| {
            engine
                .spec()
                .strategy
                .stages
                .iter()
                .find_map(|stage| stage.strategy.max_tokens)
        })
    {
        options.max_new_tokens = max_new_tokens;
    }

    let synthesis = engine
        .synthesize(PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(token_ids),
            options,
        }))
        .context(
            "What: speech synthesis failed. \
             Why: the pipeline rejected the tokenized prompt or its decode budget. \
             How: check the package's declared decoder contract and max_tokens.",
        )?;

    let (endpoint, waveform) = match &output.endpoint {
        Some(endpoint) => (
            endpoint.clone(),
            synthesis.tensors.get(endpoint).with_context(|| {
                let mut produced: Vec<&str> =
                    synthesis.tensors.keys().map(String::as_str).collect();
                produced.sort_unstable();
                format!(
                    "What: the declared waveform endpoint '{endpoint}' was not produced. \
                     Why: the pipeline emitted [{}] instead. \
                     How: correct `pipeline.audio.output` to name an endpoint the final stage actually writes.",
                    produced.join(", ")
                )
            })?,
        ),
        None => {
            let component = output.final_component.as_ref().context(
                "What: the waveform could not be located. \
                 Why: the package declares neither `pipeline.audio.output` nor a final strategy stage. \
                 How: declare `pipeline.audio.output` naming the endpoint that carries the waveform.",
            )?;
            let prefix = format!("{component}.");
            let mut candidates: Vec<(&String, _)> = synthesis
                .tensors
                .iter()
                .filter(|(endpoint, _)| endpoint.starts_with(&prefix))
                .collect();
            candidates.sort_by_key(|(endpoint, _)| (*endpoint).clone());
            match candidates.as_slice() {
                [(endpoint, value)] => ((*endpoint).clone(), *value),
                [] => bail!(
                    "What: the waveform could not be located. \
                     Why: the final-phase component '{component}' produced no output. \
                     How: declare `pipeline.audio.output` naming the endpoint that carries the waveform."
                ),
                many => bail!(
                    "What: the waveform endpoint is ambiguous. \
                     Why: the final-phase component '{component}' produced {} outputs ({}). \
                     How: declare `pipeline.audio.output` naming the one that carries the waveform.",
                    many.len(),
                    many.iter()
                        .map(|(endpoint, _)| endpoint.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }
        }
    };

    let sample_rate = request.sample_rate.or(output.sample_rate).with_context(|| {
        format!(
            "What: the synthesized waveform has no sample rate. \
             Why: the package does not declare `pipeline.audio.sample_rate`, and a runtime cannot infer it from the '{endpoint}' tensor — guessing would change the pitch and duration. \
             How: add `pipeline.audio.sample_rate` to the package's metadata, or pass the rate explicitly."
        )
    })?;

    let samples = waveform.to_vec_f32_lossy().with_context(|| {
        format!(
            "What: the waveform at '{endpoint}' could not be read as samples. \
             Why: its dtype is not a float this runtime can convert. \
             How: export the final stage with a float waveform output."
        )
    })?;
    if samples.is_empty() {
        bail!(
            "What: synthesis produced no audio. \
             Why: the waveform at '{endpoint}' is empty. \
             How: raise the decode budget (`--max-new-tokens` or the package's `max_tokens`)."
        );
    }

    let audio = SynthesizedAudio {
        samples,
        sample_rate,
        channels: output.channels,
    };
    // PCM encoding clamps to [-1, 1]. A waveform far outside that range is
    // almost always a scaling mismatch (for example a vocoder emitting
    // int16-ranged values), which would otherwise be silently flattened into
    // full-scale noise. Say so rather than shipping garbage audio quietly.
    let peak = audio.peak();
    if peak > OUT_OF_RANGE_PEAK {
        tracing::warn!(
            peak,
            endpoint = %endpoint,
            "waveform samples exceed the [-1, 1] range PCM encoding expects and will be clamped; \
             the final stage may emit integer-ranged samples that need scaling before output"
        );
    }
    Ok(audio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_metadata::{
        PipelineAudioConfig, PipelineComponentSpec, PipelineStrategy, PipelineStrategyStage,
    };

    fn component(filename: &str, role: &str) -> PipelineComponentSpec {
        PipelineComponentSpec {
            filename: filename.to_string(),
            role: role.to_string(),
            device_preference: None,
            tokenizer: None,
            io: None,
        }
    }

    fn tts_spec() -> PipelineSpec {
        let mut spec = PipelineSpec {
            strategy: PipelineStrategy {
                stages: vec![
                    PipelineStrategyStage {
                        name: "decode_codes".to_string(),
                        strategy: Box::new(PipelineStrategy {
                            decoder: Some("decoder".to_string()),
                            max_tokens: Some(4),
                            ..Default::default()
                        }),
                    },
                    PipelineStrategyStage {
                        name: "synthesize_audio".to_string(),
                        strategy: Box::new(PipelineStrategy {
                            kind: PipelineStrategyKind::SinglePass,
                            model: Some("vocoder".to_string()),
                            ..Default::default()
                        }),
                    },
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        spec.models
            .insert("decoder".to_string(), component("decoder.onnx", "decoder"));
        spec.models
            .insert("vocoder".to_string(), component("vocoder.onnx", "vocoder"));
        spec
    }

    #[test]
    fn a_decoder_plus_final_stage_is_recognized_as_text_to_audio() {
        assert!(is_text_to_audio(&tts_spec()));
    }

    #[test]
    fn a_package_without_a_final_stage_is_not_text_to_audio() {
        let mut spec = tts_spec();
        spec.strategy.stages.pop();

        assert!(!is_text_to_audio(&spec));
    }

    #[test]
    fn declared_audio_metadata_is_authoritative() {
        let mut spec = tts_spec();
        spec.audio = Some(PipelineAudioConfig {
            sample_rate: Some(24_000),
            output: Some("vocoder.audio".to_string()),
            channels: Some(2),
        });

        let output = resolve_audio_output(&spec);

        assert_eq!(output.endpoint.as_deref(), Some("vocoder.audio"));
        assert_eq!(output.sample_rate, Some(24_000));
        assert_eq!(output.channels, 2);
    }

    #[test]
    fn a_package_without_declared_audio_falls_back_to_the_final_component() {
        let output = resolve_audio_output(&tts_spec());

        assert!(output.endpoint.is_none());
        assert_eq!(output.final_component.as_deref(), Some("vocoder"));
        assert_eq!(output.channels, 1, "mono is the default");
        assert!(output.sample_rate.is_none());
    }

    #[test]
    fn peak_reports_the_largest_absolute_sample() {
        let audio = SynthesizedAudio {
            samples: vec![0.1, -0.9, 0.4],
            sample_rate: 16_000,
            channels: 1,
        };

        assert!((audio.peak() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn waveforms_round_trip_through_wav_and_pcm() {
        let audio = SynthesizedAudio {
            samples: vec![0.0, 0.5, -0.5, 1.0],
            sample_rate: 16_000,
            channels: 1,
        };

        assert_eq!(audio.to_pcm16().len(), 8);
        let wav = audio.to_wav().unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert!((audio.duration_secs() - 4.0 / 16_000.0).abs() < 1e-9);

        let decoded = onnx_genai_preprocess::audio::decode_wav_pcm16(&wav).unwrap();
        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.samples.len(), 4);
        assert!((decoded.samples[1] - 0.5).abs() < 1e-3);
    }
}
