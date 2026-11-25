use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use onnx_genai_preprocess::audio::{
    LogMelExtractor, WHISPER_SAMPLE_RATE, decode_audio, encode_wav_f32,
};

use crate::types::InputAudio;

#[derive(Clone, Debug)]
pub struct AudioInputSpec {
    pub endpoint: String,
    binding: AudioInputBinding,
}
#[derive(Clone, Debug)]
enum AudioInputBinding {
    Encoded(std::sync::Arc<onnx_genai_preprocess::audio::AudioProgram>),
    Features {
        extractor: std::sync::Arc<LogMelExtractor>,
        n_frames: std::num::NonZeroUsize,
    },
}
impl AudioInputSpec {
    pub fn from_input(endpoint: String, shape: &[i64]) -> anyhow::Result<Self> {
        anyhow::ensure!(
            shape.len() == 3 && matches!(shape[0], -1 | 1),
            "audio input '{endpoint}' must declare [1 or dynamic, mel_bins, frames], got {shape:?}; publish explicit positive feature dimensions"
        );
        let positive = |axis: usize| {
            usize::try_from(shape[axis]).ok().and_then(std::num::NonZeroUsize::new).with_context(||format!("audio input '{endpoint}' axis {axis} must declare a positive concrete feature dimension, got {}; author the mel/frame count instead of guessing it",shape[axis]))
        };
        let n_mels = positive(1)?;
        let n_frames = positive(2)?;
        n_mels.get().checked_mul(n_frames.get()).context("audio feature tensor size exceeds this host's address range; reduce the declared mel/frame geometry")?;
        Ok(Self {
            endpoint,
            binding: AudioInputBinding::Features {
                extractor: std::sync::Arc::new(LogMelExtractor::new(
                    n_mels.get(),
                    WHISPER_SAMPLE_RATE,
                )?),
                n_frames,
            },
        })
    }
    pub fn from_encoded_input(
        endpoint: String,
        program: &onnx_genai_metadata::AudioPreprocessingProgram,
    ) -> anyhow::Result<Self> {
        let program = onnx_genai_preprocess::audio::AudioProgram::from_program(program)
            .context("compile declared encoded-audio preprocessing for serving")?;
        Ok(Self {
            endpoint,
            binding: AudioInputBinding::Encoded(std::sync::Arc::new(program)),
        })
    }
    pub fn is_encoded(&self) -> bool {
        matches!(self.binding, AudioInputBinding::Encoded(_))
    }
    pub fn feature_shape(&self) -> Option<[usize; 3]> {
        match &self.binding {
            AudioInputBinding::Features {
                extractor,
                n_frames,
            } => Some([1, extractor.n_mels(), n_frames.get()]),
            AudioInputBinding::Encoded(_) => None,
        }
    }
    pub fn maximum_input_samples(
        &self,
        source_rate: u32,
    ) -> anyhow::Result<Option<std::num::NonZeroUsize>> {
        anyhow::ensure!(
            source_rate > 0,
            "audio source sample rate must be positive; use the recording's actual rate"
        );
        match &self.binding {
            AudioInputBinding::Encoded(program) => Ok(program.maximum_input_samples(source_rate)?),
            AudioInputBinding::Features {
                extractor,
                n_frames,
            } => Ok(Some(
                extractor.maximum_input_samples_for_frames(source_rate, *n_frames)?,
            )),
        }
    }
}

#[derive(Debug)]
pub struct AudioTensor {
    pub endpoint: String,
    pub data: Vec<f32>,
    pub shape: Vec<i64>,
}

pub(crate) fn decode_chat_audio(input: &InputAudio) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        !input.format.trim().is_empty(),
        "input_audio.format must be a nonempty audio format label (e.g. wav, flac, mp3); encoded content determines the decoder"
    );
    STANDARD
        .decode(&input.data)
        .context("input_audio.data must contain valid base64 encoded audio")
}

pub fn preprocess_audio(bytes: &[u8], spec: &AudioInputSpec) -> anyhow::Result<AudioTensor> {
    let audio = decode_audio(bytes)?;
    preprocess_samples(&audio.samples, audio.sample_rate, spec)
}

/// Extract the model's declared audio features from raw `[-1, 1]` mono samples.
///
/// The counterpart to [`preprocess_audio`] for callers that already hold samples
/// — a live stream segment, for instance — so a waveform never has to be
/// re-encoded into a WAV container just to be transcribed.
pub fn preprocess_samples(
    samples: &[f32],
    sample_rate: u32,
    spec: &AudioInputSpec,
) -> anyhow::Result<AudioTensor> {
    let AudioInputBinding::Features {
        extractor,
        n_frames,
    } = &spec.binding
    else {
        anyhow::bail!(
            "encoded audio preprocessing belongs to the workflow; bind encoded bytes or samples with MultimodalInput instead of extracting convenience features"
        );
    };
    let features = extractor.extract_fixed_frames(samples, sample_rate, *n_frames)?;
    let shape = features
        .shape()
        .into_iter()
        .map(|n| {
            i64::try_from(n)
                .context("audio feature dimension exceeds int64; reduce the declared geometry")
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(AudioTensor {
        endpoint: spec.endpoint.clone(),
        data: features.data,
        shape,
    })
}

/// Re-encodes finite mono samples into a float32 WAV container.
///
/// A package whose preprocessing program starts at `decode` accepts a container,
/// not bare samples, so a stream segment has to be wrapped before it is bound.
pub fn encode_samples_wav(samples: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    Ok(encode_wav_f32(samples, sample_rate, 1)?)
}
