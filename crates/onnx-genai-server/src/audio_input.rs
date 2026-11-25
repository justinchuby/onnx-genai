use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use onnx_genai_preprocess::audio::{AudioProgram, AudioTensorData, encode_wav_f32};

use crate::types::InputAudio;

#[derive(Clone, Debug)]
pub struct AudioInputSpec {
    pub endpoint: String,
    binding: AudioInputBinding,
    pub(crate) transcription: Option<
        std::sync::Arc<onnx_genai_engine::transcription::generative::GenerativeTranscription>,
    >,
}
#[derive(Clone, Debug)]
enum AudioInputBinding {
    Encoded(std::sync::Arc<onnx_genai_preprocess::audio::AudioProgram>),
    Features {
        program: std::sync::Arc<AudioProgram>,
        shape: [usize; 3],
    },
}
impl AudioInputSpec {
    pub fn transcription(
        &self,
    ) -> Option<&onnx_genai_engine::transcription::generative::GenerativeTranscription> {
        self.transcription.as_deref()
    }
    pub fn from_input(
        endpoint: String,
        shape: &[i64],
        declaration: &onnx_genai_metadata::AudioPreprocessingProgram,
    ) -> anyhow::Result<Self> {
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
        let mut selected = declaration.clone();
        let output = declaration.outputs.iter().find(|output| output.name == endpoint)
            .with_context(||format!("audio feature input '{endpoint}' needs preprocessing.audio.outputs with the same name; declare its exact waveform/analysis program and binding"))?;
        anyhow::ensure!(
            output.content == "audio_features" && output.dtype == "float32",
            "audio feature input '{endpoint}' requires its preprocessing output to declare content audio_features and dtype float32; correct the named output"
        );
        selected.outputs = vec![output.clone()];
        let program = AudioProgram::from_program(&selected).with_context(|| {
            format!("compile preprocessing for audio feature input '{endpoint}'")
        })?;
        Ok(Self {
            endpoint,
            transcription: None,
            binding: AudioInputBinding::Features {
                program: std::sync::Arc::new(program),
                shape: [1, n_mels.get(), n_frames.get()],
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
            transcription: None,
            binding: AudioInputBinding::Encoded(std::sync::Arc::new(program)),
        })
    }
    pub fn is_encoded(&self) -> bool {
        matches!(self.binding, AudioInputBinding::Encoded(_))
    }
    pub fn feature_shape(&self) -> Option<[usize; 3]> {
        match &self.binding {
            AudioInputBinding::Features { shape, .. } => Some(*shape),
            AudioInputBinding::Encoded(_) => None,
        }
    }
    pub fn maximum_input_samples(
        &self,
        source_rate: u32,
    ) -> anyhow::Result<Option<std::num::NonZeroUsize>> {
        let program = match &self.binding {
            AudioInputBinding::Encoded(program) | AudioInputBinding::Features { program, .. } => {
                program
            }
        };
        Ok(program.maximum_input_samples(source_rate)?)
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

/// Execute the package's selected feature output, retaining authored analysis semantics.
pub fn preprocess_audio(bytes: &[u8], spec: &AudioInputSpec) -> anyhow::Result<AudioTensor> {
    let AudioInputBinding::Features { program, shape } = &spec.binding else {
        anyhow::bail!(
            "encoded audio preprocessing belongs to the workflow; bind encoded bytes with MultimodalInput instead of extracting features"
        );
    };
    let tensor = program.run(&[bytes])?.into_iter().next().context(
        "selected audio preprocessing output is absent; repair the package's output binding",
    )?;
    anyhow::ensure!(
        tensor.shape == shape.map(|value| value as i64),
        "audio preprocessing output '{}' has shape {:?}, expected {:?}; correct the authored padding/analysis geometry",
        spec.endpoint,
        tensor.shape,
        shape
    );
    let AudioTensorData::Fp32(data) = tensor.data else {
        anyhow::bail!(
            "audio preprocessing output '{}' is not float32; correct its declared transform/output dtype",
            spec.endpoint
        );
    };
    Ok(AudioTensor {
        endpoint: spec.endpoint.clone(),
        data,
        shape: tensor.shape,
    })
}

/// Wrap finite mono samples without quantization, then run the same declared program.
pub fn preprocess_samples(
    samples: &[f32],
    sample_rate: u32,
    spec: &AudioInputSpec,
) -> anyhow::Result<AudioTensor> {
    preprocess_audio(&encode_wav_f32(samples, sample_rate, 1)?, spec)
}

/// Re-encodes finite mono samples into a float32 WAV container.
///
/// A package whose preprocessing program starts at `decode` accepts a container,
/// not bare samples, so a stream segment has to be wrapped before it is bound.
pub fn encode_samples_wav(samples: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    Ok(encode_wav_f32(samples, sample_rate, 1)?)
}
