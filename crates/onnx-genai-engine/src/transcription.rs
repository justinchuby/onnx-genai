//! One-pass frame decoding through a package-declared acoustic workflow.
pub mod generative;
use crate::{Engine, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest};
use anyhow::{Context, ensure};
use onnx_genai_metadata::transcription::{
    TranscriptionContract, TranscriptionLengthLayout, transcription_contract,
};
use onnx_genai_ort::Tokenizer;
use std::{collections::HashSet, num::NonZeroUsize, sync::Arc};
#[derive(Debug, Clone, Copy)]
pub struct TranscriptionLimits {
    pub max_frames: NonZeroUsize,
    pub max_tokens: NonZeroUsize,
    pub max_text_bytes: NonZeroUsize,
}
impl Default for TranscriptionLimits {
    fn default() -> Self {
        Self {
            max_frames: NonZeroUsize::new(1_000_000).unwrap(),
            max_tokens: NonZeroUsize::new(16_384).unwrap(),
            max_text_bytes: NonZeroUsize::new(4 * 1024 * 1024).unwrap(),
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error(
    "transcription {resource} uses {used}, exceeding its limit {limit}; shorten the audio or raise the configured transcription limit"
)]
pub struct TranscriptionLimitError {
    pub resource: &'static str,
    pub used: usize,
    pub limit: usize,
}
fn bounded(resource: &'static str, used: usize, limit: NonZeroUsize) -> anyhow::Result<()> {
    if used > limit.get() {
        return Err(TranscriptionLimitError {
            resource,
            used,
            limit: limit.get(),
        }
        .into());
    }
    Ok(())
}
#[derive(Debug)]
pub struct PreparedTranscription {
    contract: Arc<TranscriptionContract>,
    bytes: Arc<[u8]>,
    limits: TranscriptionLimits,
    output_limit: Option<NonZeroUsize>,
}
impl PreparedTranscription {
    pub fn prepare(
        contract: Arc<TranscriptionContract>,
        bytes: Arc<[u8]>,
        limits: TranscriptionLimits,
    ) -> anyhow::Result<Self> {
        contract.audio().admit_bytes(bytes.len())?;
        Ok(Self {
            contract,
            bytes,
            limits,
            output_limit: None,
        })
    }
    pub fn with_output_limit(mut self, maximum: NonZeroUsize) -> anyhow::Result<Self> {
        ensure!(
            maximum.get() <= self.limits.max_tokens.get(),
            "requested transcription max_tokens {} exceeds the configured {}-token limit; lower max_tokens or raise the server output allowance",
            maximum,
            self.limits.max_tokens
        );
        self.output_limit = Some(maximum);
        Ok(self)
    }
}
#[derive(Debug)]
struct ResolvedVocabulary {
    tokens: Vec<String>,
    ignored: Vec<bool>,
}
fn vocabulary(
    contract: &TranscriptionContract,
    tokenizer: Option<&Tokenizer>,
) -> anyhow::Result<ResolvedVocabulary> {
    let spec = contract.decoding().vocabulary.as_ref().context(
        "admitted transcription profile lost its vocabulary; report the invalid runtime contract",
    )?;
    let vocabulary = if spec.source == "inline" {
        spec.tokens.clone()
    } else {
        let tokenizer = tokenizer.context("transcription vocabulary source is tokenizer, but the selected package ships none; supply tokenizer.json or an inline class table")?;
        (0..contract.classes()).map(|id| {
            tokenizer.inner().id_to_token(id as u32).with_context(|| format!("transcription vocabulary class {id} is absent from the selected package's tokenizer.json; repair the acoustic vocabulary table or size"))
        }).collect::<anyhow::Result<Vec<_>>>()?
    };
    let available = vocabulary
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    for token in spec.ignored_tokens.iter().chain(spec.word_delimiter.iter()) {
        ensure!(
            available.contains(token.as_str()),
            "transcription vocabulary selection '{token}' is absent from the selected package's resolved table; correct its exact token spelling"
        );
    }
    let ignored = spec.ignored_tokens.iter().collect::<HashSet<_>>();
    let ignored = vocabulary
        .iter()
        .map(|token| ignored.contains(token))
        .collect();
    Ok(ResolvedVocabulary {
        tokens: vocabulary,
        ignored,
    })
}
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionResult {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub frames: usize,
}
impl Engine {
    pub fn transcribe(
        &mut self,
        request: PreparedTranscription,
    ) -> anyhow::Result<TranscriptionResult> {
        let contract=transcription_contract(self.metadata()).map_err(anyhow::Error::msg)?.context("model declares no non-generative transcription profile; select a package with frame logits and explicit decoding")?;
        ensure!(
            contract == *request.contract,
            "prepared transcription belongs to a different model contract; prepare audio using the selected model's transcription profile"
        );
        let vocabulary = vocabulary(&contract, self.workflow.package_tokenizer())?;
        let mut execution = PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(Vec::new()),
            options: GenerateOptions {
                max_new_tokens: 0,
                ..Default::default()
            },
        });
        execution = execution.with_input(
            contract.audio().endpoint(),
            self.encoded_media_value(contract.audio().endpoint(), &request.bytes)?,
        );
        let outputs = self.run_pipeline_outputs(execution).with_context(|| {
            format!(
                "execute transcription profile '{}' for encoded audio input '{}'",
                contract.profile(),
                contract.audio().endpoint()
            )
        })?;
        let logits = outputs.aggregate(contract.logits()).with_context(|| {
            format!(
                "transcription workflow did not emit logits '{}'",
                contract.logits()
            )
        })?;
        let shape = logits.shape();
        ensure!(
            shape.len() == contract.rank(),
            "transcription logits '{}' returned rank {}, expected {}; correct the workflow output contract",
            contract.logits(),
            shape.len(),
            contract.rank()
        );
        let dimensions=shape.iter().map(|n|usize::try_from(*n).context("transcription logits contain a negative/unrepresentable dimension; emit a concrete shape")).collect::<anyhow::Result<Vec<_>>>()?;
        let spec = contract.decoding();
        ensure!(
            dimensions[spec.class_axis] == contract.classes(),
            "transcription logits '{}' returned {} classes, but vocabulary declares {}; align the acoustic head and vocabulary",
            contract.logits(),
            dimensions[spec.class_axis],
            contract.classes()
        );
        for (axis, size) in dimensions.iter().enumerate() {
            ensure!(
                axis == spec.time_axis || axis == spec.class_axis || *size == 1,
                "transcription logits batch axis {axis} has {size} rows; this file request binds exactly one audio row"
            );
        }
        let frames = dimensions[spec.time_axis];
        let valid = if let Some(name) = contract.lengths() {
            let value = outputs.aggregate(name).with_context(|| {
                format!("transcription workflow omitted valid-frame output '{name}'")
            })?;
            let expected_shape = match contract
                .length_layout()
                .context("admitted transcription length layout is missing")?
            {
                TranscriptionLengthLayout::Singleton { rank } => vec![1; rank],
                TranscriptionLengthLayout::TimePrefix => shape[..spec.time_axis].to_vec(),
            };
            ensure!(
                value.shape() == expected_shape,
                "transcription lengths '{name}' returned shape {:?}, expected {:?}; emit the declared valid-frame companion layout",
                value.shape(),
                expected_shape
            );
            let data = value
                .to_vec_i64()
                .with_context(|| format!("read transcription lengths '{name}' as host int64"))?;
            let first=*data.first().with_context(||format!("transcription lengths '{name}' contain no audio-row count; emit one shared valid frame prefix"))?;
            ensure!(
                data.iter().all(|count| *count == first),
                "transcription lengths '{name}' disagree across class positions for one audio row; emit one consistent valid-frame prefix for all classes"
            );
            let count=usize::try_from(first).with_context(||format!("transcription lengths '{name}' returned negative/unrepresentable count {first}; emit a nonnegative count"))?;
            ensure!(
                count <= frames,
                "transcription lengths '{name}' returned {count} for {frames} frames; correct the model's valid-prefix accounting"
            );
            count
        } else {
            frames
        };
        bounded("logit frames", frames, request.limits.max_frames)?;
        // Inspect the full storage extent before materializing host logits.
        let elements = dimensions.iter().try_fold(1usize, |product, size| {
            product
                .checked_mul(*size)
                .context("transcription logits size overflows; reduce the acoustic output shape")
        })?;
        ensure!(
            elements == logits.numel(),
            "transcription logits storage does not match its declared shape; repair the acoustic output"
        );
        let data = logits.to_vec_f32_lossy().context(
            "transcription logits must be host-readable float16/bfloat16/float32 values",
        )?;
        decode(
            &request.contract,
            &vocabulary,
            &dimensions,
            &data,
            valid,
            request.limits,
            request.output_limit,
        )
    }
}
fn decode(
    contract: &TranscriptionContract,
    vocabulary: &ResolvedVocabulary,
    shape: &[usize],
    data: &[f32],
    valid: usize,
    limits: TranscriptionLimits,
    output_limit: Option<NonZeroUsize>,
) -> anyhow::Result<TranscriptionResult> {
    let spec = contract.decoding();
    let strides = (0..shape.len())
        .map(|axis| shape[axis + 1..].iter().product::<usize>())
        .collect::<Vec<_>>();
    let vocab = spec
        .vocabulary
        .as_ref()
        .context("admitted transcription vocabulary is missing")?;
    let mut previous = None;
    let mut token_ids = Vec::new();
    let mut text = String::new();
    for frame in 0..valid {
        let mut best = 0;
        let mut maximum = f32::NEG_INFINITY;
        for class in 0..vocabulary.tokens.len() {
            let value = data[frame * strides[spec.time_axis] + class * strides[spec.class_axis]];
            ensure!(
                value.is_finite(),
                "transcription logits '{}' frame {frame}, class {class} is nonfinite; correct the acoustic head output",
                contract.logits()
            );
            if value > maximum {
                maximum = value;
                best = class;
            }
        }
        let repeated = previous == Some(best);
        previous = Some(best);
        if (spec.collapse_repeats && repeated)
            || spec.blank_id == Some(best as u32)
            || vocabulary.ignored[best]
        {
            continue;
        }
        bounded("decoded tokens", token_ids.len() + 1, limits.max_tokens)?;
        let bytes = text
            .len()
            .checked_add(vocabulary.tokens[best].len())
            .context("transcription text length overflows; shorten the audio/vocabulary strings")?;
        bounded("text bytes", bytes, limits.max_text_bytes)?;
        token_ids.push(best as u32);
        text.push_str(&vocabulary.tokens[best]);
        if output_limit.is_some_and(|limit| token_ids.len() == limit.get()) {
            break;
        }
    }
    if let Some(delimiter) = &vocab.word_delimiter {
        text = text
            .split(delimiter)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
    }
    Ok(TranscriptionResult {
        text,
        token_ids,
        frames: valid,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_metadata::{InferenceMetadata, TensorDimension};
    fn metadata() -> InferenceMetadata {
        onnx_genai_metadata::parse_metadata(include_str!("../../../tests/fixtures/onnx_genai_workflows/transcription_ctc/inference_metadata.yaml"),None).unwrap()
    }
    fn contract(time: usize, class: usize, rank: usize) -> TranscriptionContract {
        let mut metadata = metadata();
        let spec = metadata
            .profiles
            .get_mut("transcription")
            .unwrap()
            .decoding
            .as_mut()
            .unwrap();
        spec.time_axis = time;
        spec.class_axis = class;
        let workflow = &mut metadata.pipeline.as_mut().unwrap().workflow;
        let logits = workflow.outputs.get_mut("logits").unwrap();
        logits.contract.shape = (0..rank)
            .map(|axis| {
                if axis == time {
                    TensorDimension::Symbol("frames".into())
                } else if axis == class {
                    TensorDimension::Fixed(4)
                } else {
                    TensorDimension::Fixed(1)
                }
            })
            .collect();
        let batch = (0..rank).find(|axis| *axis != time && *axis != class);
        logits.contract.batch_layout = batch
            .map_or(onnx_genai_metadata::BatchLayout::Shared, |axis| {
                onnx_genai_metadata::BatchLayout::RequestAligned { axis }
            });
        let logits_contract = logits.contract.clone();
        let prefix = logits.contract.shape[..time].to_vec();
        let lengths = workflow.outputs.get_mut("frame_lengths").unwrap();
        lengths.contract.shape = prefix;
        lengths.contract.batch_layout = batch
            .filter(|axis| *axis < time)
            .map_or(onnx_genai_metadata::BatchLayout::Shared, |axis| {
                onnx_genai_metadata::BatchLayout::RequestAligned { axis }
            });
        let length_contract = lengths.contract.clone();
        let wave_contract = workflow.components["audio"].ports.outputs["wave"].clone();
        let head = workflow.components.get_mut("head").unwrap();
        head.ports.inputs.insert("wave".into(), wave_contract);
        head.ports.outputs.insert("logits".into(), logits_contract);
        head.ports.outputs.insert("counts".into(), length_contract);
        for step in &mut workflow.steps {
            match step {
                onnx_genai_metadata::WorkflowStep::Invoke {
                    component, outputs, ..
                } if component == "head" => {
                    outputs.insert("counts".into(), "axis.counts".into());
                }
                onnx_genai_metadata::WorkflowStep::Emit { output, value, .. }
                    if output == "frame_lengths" =>
                {
                    *value = "axis.counts".into()
                }
                _ => {}
            }
        }
        onnx_genai_metadata::validate_metadata(&metadata).unwrap();
        transcription_contract(&metadata).unwrap().unwrap()
    }
    fn scores(time: usize, class: usize, shape: &[usize], winners: &[usize]) -> Vec<f32> {
        // Enumerate coordinates independently of the decoder's stride gather.
        fn visit(
            axis: usize,
            shape: &[usize],
            coords: &mut Vec<usize>,
            time: usize,
            class: usize,
            winners: &[usize],
            values: &mut Vec<f32>,
        ) {
            if axis == shape.len() {
                values.push(if coords[class] == winners[coords[time]] {
                    1.0
                } else {
                    0.0
                });
                return;
            }
            for coordinate in 0..shape[axis] {
                coords.push(coordinate);
                visit(axis + 1, shape, coords, time, class, winners, values);
                coords.pop();
            }
        }
        let mut values = Vec::new();
        visit(0, shape, &mut Vec::new(), time, class, winners, &mut values);
        values
    }
    #[test]
    fn frame_decoding_obeys_all_declared_time_class_axis_layouts() {
        let winners = [3, 3, 0, 3, 2, 1, 1, 0, 1, 3, 3];
        for rank in [2, 3, 4, 5] {
            for time in 0..rank {
                for class in 0..rank {
                    if time == class {
                        continue;
                    }
                    let contract = contract(time, class, rank);
                    let vocab = vocabulary(&contract, None).unwrap();
                    let shape = (0..rank)
                        .map(|axis| {
                            if axis == time {
                                winners.len()
                            } else if axis == class {
                                4
                            } else {
                                1
                            }
                        })
                        .collect::<Vec<_>>();
                    let data = scores(time, class, &shape, &winners);
                    let result = decode(
                        &contract,
                        &vocab,
                        &shape,
                        &data,
                        9,
                        TranscriptionLimits::default(),
                        None,
                    )
                    .unwrap();
                    assert_eq!(result.text, "bb aa", "axes {time}/{class} rank {rank}");
                    assert_eq!(result.token_ids, [3, 3, 2, 1, 1]);
                }
            }
        }
    }
    #[test]
    fn blank_collapse_ignored_tokens_and_delimiters_have_declared_order() {
        let mut metadata = metadata();
        let spec = metadata
            .profiles
            .get_mut("transcription")
            .unwrap()
            .decoding
            .as_mut()
            .unwrap();
        spec.vocabulary.as_mut().unwrap().ignored_tokens = vec!["a".into()];
        let contract = transcription_contract(&metadata).unwrap().unwrap();
        let vocab = vocabulary(&contract, None).unwrap();
        let winners = [2, 2, 3, 1, 3, 2, 2, 3, 2];
        let data = scores(1, 2, &[1, 9, 4], &winners);
        let result = decode(
            &contract,
            &vocab,
            &[1, 9, 4],
            &data,
            9,
            TranscriptionLimits::default(),
            None,
        )
        .unwrap();
        assert_eq!(result.text, "bb b");
        assert_eq!(result.token_ids, [2, 3, 3, 2, 3, 2]);
        let mut metadata = metadata.clone();
        metadata
            .profiles
            .get_mut("transcription")
            .unwrap()
            .decoding
            .as_mut()
            .unwrap()
            .collapse_repeats = false;
        let contract = transcription_contract(&metadata).unwrap().unwrap();
        let result = decode(
            &contract,
            &vocab,
            &[1, 9, 4],
            &data,
            9,
            TranscriptionLimits::default(),
            None,
        )
        .unwrap();
        assert_eq!(result.text, "bb b");
        assert_eq!(result.token_ids, [2, 2, 3, 3, 2, 2, 3, 2]);
    }
    #[test]
    fn nonfinite_valid_logits_fail_but_padding_is_not_decoded_and_ties_are_stable() {
        let contract = contract(1, 2, 3);
        let vocab = vocabulary(&contract, None).unwrap();
        let mut data = vec![0.0; 8];
        data[4..].fill(f32::NAN);
        let result = decode(
            &contract,
            &vocab,
            &[1, 2, 4],
            &data,
            1,
            TranscriptionLimits::default(),
            None,
        )
        .unwrap();
        assert_eq!(result.text, "");
        assert!(result.token_ids.is_empty());
        data[0] = f32::NAN;
        let error = decode(
            &contract,
            &vocab,
            &[1, 2, 4],
            &data,
            1,
            TranscriptionLimits::default(),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("frame 0, class 0"), "{error}");
    }
    #[test]
    fn tokenizer_vocabulary_uses_the_selected_package_table_and_checks_selections() {
        let mut metadata = metadata();
        let vocab = metadata
            .profiles
            .get_mut("transcription")
            .unwrap()
            .decoding
            .as_mut()
            .unwrap()
            .vocabulary
            .as_mut()
            .unwrap();
        vocab.source = "tokenizer".into();
        vocab.tokens.clear();
        vocab.word_delimiter = None;
        let contract = transcription_contract(&metadata).unwrap().unwrap();
        assert!(
            vocabulary(&contract, None)
                .unwrap_err()
                .to_string()
                .contains("selected package ships none")
        );
        let tokenizer = Tokenizer::from_file(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/onnx_genai_workflows/transcription_ctc/tokenizer.json"),
        )
        .unwrap();
        assert_eq!(
            vocabulary(&contract, Some(&tokenizer)).unwrap().tokens,
            ["<pad>", "<unk>", "<bos>", "<eos>"]
        );
        metadata
            .profiles
            .get_mut("transcription")
            .unwrap()
            .decoding
            .as_mut()
            .unwrap()
            .vocabulary
            .as_mut()
            .unwrap()
            .ignored_tokens = vec!["absent".into()];
        let contract = transcription_contract(&metadata).unwrap().unwrap();
        assert!(
            vocabulary(&contract, Some(&tokenizer))
                .unwrap_err()
                .to_string()
                .contains("selected package's resolved table")
        );
    }
}
