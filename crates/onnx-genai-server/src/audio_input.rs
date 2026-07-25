use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use onnx_genai_preprocess::audio::{
    LogMelExtractor, WHISPER_N_FRAMES, WHISPER_SAMPLE_RATE, decode_wav_pcm16,
};

use crate::types::InputAudio;

#[derive(Clone, Debug)]
pub struct AudioInputSpec {
    pub endpoint: String,
    pub n_mels: usize,
    pub n_frames: usize,
    pub max_tokens: Option<usize>,
}

impl AudioInputSpec {
    pub fn from_input(
        endpoint: String,
        shape: &[i64],
        max_tokens: Option<usize>,
    ) -> anyhow::Result<Self> {
        if shape.len() != 3 {
            anyhow::bail!(
                "audio input '{endpoint}' must have rank 3 [batch, mel, frames], got {shape:?}"
            );
        }
        if !matches!(shape[0], -1 | 1) {
            anyhow::bail!("audio input '{endpoint}' must have batch dimension 1 or dynamic");
        }
        let n_mels = usize::try_from(shape[1])
            .ok()
            .filter(|value| matches!(value, 80 | 128))
            .with_context(|| {
                format!(
                    "audio input '{endpoint}' must declare 80 or 128 mel bins, got {}",
                    shape[1]
                )
            })?;
        let n_frames = usize::try_from(shape[2])
            .ok()
            .filter(|value| *value > 0)
            .unwrap_or(WHISPER_N_FRAMES);
        Ok(Self {
            endpoint,
            n_mels,
            n_frames,
            max_tokens,
        })
    }
}

#[derive(Debug)]
pub struct AudioTensor {
    pub endpoint: String,
    pub data: Vec<f32>,
    pub shape: Vec<i64>,
}

pub(crate) fn decode_chat_audio(input: &InputAudio) -> anyhow::Result<Vec<u8>> {
    match input.format.as_str() {
        "wav" => STANDARD
            .decode(&input.data)
            .context("input_audio.data must be valid base64"),
        "mp3" => anyhow::bail!("MP3 audio is not supported yet; provide a PCM16 WAV file"),
        format => anyhow::bail!("unsupported input_audio format '{format}'"),
    }
}

pub fn preprocess_wav(bytes: &[u8], spec: &AudioInputSpec) -> anyhow::Result<AudioTensor> {
    let audio = decode_wav_pcm16(bytes)?;
    preprocess_samples(&audio.samples, audio.sample_rate, spec)
}

/// Extract the model's declared audio features from raw `[-1, 1]` mono samples.
///
/// The counterpart to [`preprocess_wav`] for callers that already hold samples
/// — a live stream segment, for instance — so a waveform never has to be
/// re-encoded into a WAV container just to be transcribed.
pub fn preprocess_samples(
    samples: &[f32],
    sample_rate: u32,
    spec: &AudioInputSpec,
) -> anyhow::Result<AudioTensor> {
    let extractor = LogMelExtractor::new(spec.n_mels, WHISPER_SAMPLE_RATE)?;
    let features = if spec.n_frames == WHISPER_N_FRAMES {
        extractor.extract_padded(samples, sample_rate)?
    } else {
        extractor.extract(samples, sample_rate)?
    };
    let data = resize_feature_frames(
        &features.data,
        spec.n_mels,
        features.n_frames,
        spec.n_frames,
    );
    Ok(AudioTensor {
        endpoint: spec.endpoint.clone(),
        data,
        shape: vec![1, spec.n_mels as i64, spec.n_frames as i64],
    })
}

fn resize_feature_frames(
    source: &[f32],
    n_mels: usize,
    source_frames: usize,
    target_frames: usize,
) -> Vec<f32> {
    if source_frames == target_frames {
        return source.to_vec();
    }
    let copy_frames = source_frames.min(target_frames);
    let mut output = vec![-1.5; n_mels * target_frames];
    for mel in 0..n_mels {
        let source_start = mel * source_frames;
        let target_start = mel * target_frames;
        output[target_start..target_start + copy_frames]
            .copy_from_slice(&source[source_start..source_start + copy_frames]);
    }
    output
}
