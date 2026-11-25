//! Admission for frame-synchronous, non-generative speech workflows.
use crate::task_profiles::EncodedMediaInput;
use crate::{
    InferenceMetadata, RuntimeInputRole, SemanticInputRole, SequenceDecodingSpec, TensorDimension,
    WorkflowInputSource, WorkflowOutputRole,
};
use std::collections::BTreeSet;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptionLengthLayout {
    Singleton { rank: usize },
    TimePrefix,
}
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionContract {
    profile: String,
    audio: EncodedMediaInput,
    logits: String,
    lengths: Option<String>,
    length_layout: Option<TranscriptionLengthLayout>,
    rank: usize,
    decoding: SequenceDecodingSpec,
    classes: usize,
}
impl TranscriptionContract {
    pub fn profile(&self) -> &str {
        &self.profile
    }
    pub fn audio(&self) -> &EncodedMediaInput {
        &self.audio
    }
    pub fn logits(&self) -> &str {
        &self.logits
    }
    pub fn lengths(&self) -> Option<&str> {
        self.lengths.as_deref()
    }
    pub fn length_layout(&self) -> Option<TranscriptionLengthLayout> {
        self.length_layout
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn decoding(&self) -> &SequenceDecodingSpec {
        &self.decoding
    }
    pub fn classes(&self) -> usize {
        self.classes
    }
}
fn invalid(message: impl Into<String>) -> String {
    message.into()
}
/// Admit one encoded request audio input and one frame-by-class output. More
/// complex batch/conditioning contracts remain explicit, not guessed bindings.
pub fn transcription_contract(
    metadata: &InferenceMetadata,
) -> Result<Option<TranscriptionContract>, String> {
    let mut result = None;
    for (name, profile) in &metadata.profiles {
        if profile.kind != "transcription"
            || crate::task_profiles::skip_unsupported_version(profile)
        {
            continue;
        }
        let path = format!("profiles.{name}");
        if profile.version != "1.0" {
            return Err(invalid(format!(
                "{path}.version '{}' is unsupported for transcription; use version '1.0' or mark a future profile ignorable",
                profile.version
            )));
        }
        if result.is_some() {
            return Err(invalid(
                "multiple transcription profiles are ambiguous; declare one executable transcription contract per package",
            ));
        }
        let workflow = &metadata
            .pipeline
            .as_ref()
            .ok_or_else(|| {
                invalid(format!(
                    "{path} needs pipeline.workflow; declare an acoustic workflow"
                ))
            })?
            .workflow;
        let decoding=profile.decoding.as_ref().ok_or_else(||invalid(format!("{path}.decoding is required; declare ctc or greedy_argmax with time/class axes and a vocabulary")))?;
        if !matches!(decoding.kind.as_str(), "ctc" | "greedy_argmax")
            || profile.pooling.is_some()
            || profile.decision.is_some()
            || profile.reranking.is_some()
            || profile.realtime.is_some()
        {
            return Err(invalid(format!(
                "{path} needs frame-synchronous ctc/greedy_argmax decoding without pooling or another task contract"
            )));
        }
        if decoding.kind == "ctc" && decoding.blank_id.is_none() {
            return Err(invalid(format!(
                "{path}.decoding.blank_id is required for ctc; declare its blank class"
            )));
        }
        let logits = profile.outputs.get("logits").ok_or_else(|| {
            invalid(format!(
                "{path}.outputs.logits must name the acoustic frame logits workflow output"
            ))
        })?;
        let output = workflow.outputs.get(logits).ok_or_else(|| {
            invalid(format!(
                "{path}.outputs.logits references absent workflow output '{logits}'"
            ))
        })?;
        let shape = &output.contract.shape;
        let rank = shape.len();
        if rank < 2
            || decoding.time_axis >= rank
            || decoding.class_axis >= rank
            || decoding.time_axis == decoding.class_axis
            || output.role != WorkflowOutputRole::Tensor
            || !matches!(
                output.contract.dtype.as_str(),
                "float16" | "bfloat16" | "float32"
            )
        {
            return Err(invalid(format!(
                "{path} logits '{logits}' need float16/bfloat16/float32 tensor rank at least 2 with distinct in-range time_axis/class_axis; correct the output contract"
            )));
        }
        let vocab=decoding.vocabulary.as_ref().ok_or_else(||invalid(format!("{path}.decoding.vocabulary is required; declare inline tokens or tokenizer source with a declared class count")))?;
        let classes = match vocab.source.as_str() {
            "inline" => {
                if vocab.tokens.is_empty()
                    || vocab.size.is_some_and(|size| size != vocab.tokens.len())
                {
                    return Err(invalid(format!(
                        "{path}.decoding.vocabulary needs a nonempty inline table whose size matches tokens"
                    )));
                }
                for value in vocab
                    .ignored_tokens
                    .iter()
                    .chain(vocab.word_delimiter.iter())
                {
                    if !vocab.tokens.contains(value) {
                        return Err(invalid(format!(
                            "{path}.decoding.vocabulary selection '{value}' is absent from tokens; correct its exact spelling"
                        )));
                    }
                }
                vocab.tokens.len()
            }
            "tokenizer" => {
                if !vocab.tokens.is_empty() {
                    return Err(invalid(format!(
                        "{path}.decoding.vocabulary tokenizer source must omit inline tokens; choose one authoritative table"
                    )));
                }
                match vocab.size {
                    Some(0) => return Err(invalid(format!("{path}.decoding.vocabulary.size must be positive; declare the acoustic class count"))),
                    Some(size) => size,
                    None => match &shape[decoding.class_axis] {
                        TensorDimension::Fixed(size) if *size > 0 => usize::try_from(*size).map_err(|_|invalid(format!("{path} class dimension {size} exceeds this host's address range")))?,
                        _ => return Err(invalid(format!("{path} tokenizer vocabulary needs a positive size or fixed class dimension; declare the acoustic class count"))),
                    }
                }
            }
            source => {
                return Err(invalid(format!(
                    "{path}.decoding.vocabulary.source '{source}' is unsupported; use inline or tokenizer"
                )));
            }
        };
        if classes > u32::MAX as usize
            || decoding.blank_id.is_some_and(|id| id as usize >= classes)
            || vocab.word_delimiter.as_ref().is_some_and(String::is_empty)
        {
            return Err(invalid(format!(
                "{path}.decoding has an invalid blank/vocabulary range or empty word delimiter; use an in-range class id and a nonempty delimiter"
            )));
        }
        let media = workflow
            .inputs
            .iter()
            .filter(|(_, input)| {
                matches!(
                    &input.role,
                    SemanticInputRole::Runtime {
                        role: RuntimeInputRole::Media,
                        ..
                    }
                )
            })
            .collect::<Vec<_>>();
        let [(input_name, input)] = media.as_slice() else {
            return Err(invalid(format!(
                "{path} needs exactly one runtime media workflow input for encoded audio; declare its request authority"
            )));
        };
        if !matches!(input.source, WorkflowInputSource::Request) {
            return Err(invalid(format!(
                "{path} input '{input_name}' must be request-sourced runtime media; do not infer an application binding"
            )));
        }
        let audio = crate::task_profiles::encoded_media_input(input_name, input, &path)
            .map_err(|e| e.to_string())?;
        for (other, binding) in &workflow.inputs {
            if other != *input_name && binding.required && binding.default.is_none() {
                return Err(invalid(format!(
                    "{path} requires additional workflow input '{other}' with no default; publish a self-contained acoustic workflow or declare explicit conditioning before exposing file transcription"
                )));
            }
        }
        if workflow.outputs.values().any(|output| {
            matches!(
                output.role,
                WorkflowOutputRole::Tokens | WorkflowOutputRole::Text
            )
        }) {
            return Err(invalid(format!(
                "{path} must expose non-generative frame logits, not token/text workflow outputs; use the generative speech path for autoregressive decoding"
            )));
        }
        for (axis, dimension) in shape.iter().enumerate() {
            if let TensorDimension::Fixed(size) = dimension
                && (*size < 0
                    || (axis == decoding.class_axis
                        && usize::try_from(*size).ok() != Some(classes))
                    || (axis != decoding.class_axis && axis != decoding.time_axis && *size != 1))
            {
                return Err(invalid(format!(
                    "{path} logits '{logits}' axis {axis} dimension {size} conflicts with the declared {classes}-class/single-audio-row contract"
                )));
            }
        }
        let padding=output.contract.padding.iter().find(|padding|matches!(&shape[decoding.time_axis],TensorDimension::Symbol(symbol) if symbol==&padding.dimension));
        let lengths=decoding.lengths.as_ref().map(|role|->Result<(String,TranscriptionLengthLayout),String>{
            let endpoint=profile.outputs.get(role).ok_or_else(||invalid(format!("{path}.decoding.lengths role '{role}' has no output binding")))?;
            let count=workflow.outputs.get(endpoint).ok_or_else(||invalid(format!("{path} lengths output '{endpoint}' is absent from workflow outputs")))?;
            if count.role!=WorkflowOutputRole::Tensor || count.contract.dtype!="int64" {return Err(invalid(format!("{path} lengths '{endpoint}' need an int64 tensor; emit the valid frame prefix with its declared shape")));}
            let layout=if let Some(padding)=padding {
                let mut errors=Vec::new();
                crate::validation::validate_valid_lengths_shape(&path,&output.contract,&padding.dimension,decoding.time_axis,endpoint,&count.contract,&mut errors);
                if !errors.is_empty(){return Err(errors.join("; "));}
                TranscriptionLengthLayout::TimePrefix
            }else{
                if count.contract.shape.iter().any(|dim| matches!(dim,TensorDimension::Fixed(size) if *size != 1)) {
                    return Err(invalid(format!("{path} lengths '{endpoint}' need singleton dimensions for one audio row, or the logits padding contract's exact outer-axis shape")));
                }
                TranscriptionLengthLayout::Singleton { rank: count.contract.shape.len() }
            };
            Ok((endpoint.clone(),layout))
        }).transpose()?;
        if let Some(padding) = padding
            && lengths.as_ref().map(|(endpoint, _)| endpoint.as_str())
                != Some(padding.valid_lengths.as_str())
        {
            return Err(invalid(format!(
                "{path} logits '{logits}' padding and decoding.lengths must bind the same '{}'; decode only that valid frame prefix",
                padding.valid_lengths
            )));
        }
        let (lengths, length_layout) = match lengths {
            Some((endpoint, layout)) => (Some(endpoint), Some(layout)),
            None => (None, None),
        };
        let expected =
            BTreeSet::from_iter(std::iter::once("logits").chain(decoding.lengths.as_deref()));
        if profile
            .outputs
            .keys()
            .any(|role| !expected.contains(role.as_str()))
        {
            return Err(invalid(format!(
                "{path}.outputs contains unsupported roles; expose logits and its declared valid-frame count only"
            )));
        }
        result = Some(TranscriptionContract {
            profile: name.clone(),
            audio,
            logits: logits.clone(),
            lengths,
            length_layout,
            rank,
            decoding: decoding.clone(),
            classes,
        });
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn doc() -> InferenceMetadata {
        crate::parse_metadata(include_str!("../../../tests/fixtures/onnx_genai_workflows/transcription_ctc/inference_metadata.yaml"),None).unwrap()
    }
    #[test]
    fn ctc_admission_resolves_audio_axes_lengths_and_vocabulary() {
        let contract = transcription_contract(&doc()).unwrap().unwrap();
        assert_eq!(contract.audio().endpoint(), "request.audio");
        assert_eq!(contract.logits(), "logits");
        assert_eq!(contract.lengths(), Some("frame_lengths"));
        assert_eq!(contract.classes(), 4);
        assert_eq!(contract.decoding().time_axis, 1);
        assert_eq!(contract.decoding().class_axis, 2);
        contract.audio().admit_bytes(44).unwrap();
        assert!(contract.audio().admit_bytes(0).is_err());
    }
    #[test]
    fn ctc_admission_rejects_ambiguous_or_nonexecutable_profiles() {
        for case in 0..14 {
            let mut metadata = doc();
            let profile = metadata.profiles.get_mut("transcription").unwrap();
            match case {
                0 => profile.version = "99".into(),
                1 => profile.decoding.as_mut().unwrap().blank_id = None,
                2 => profile.decoding.as_mut().unwrap().time_axis = 2,
                3 => profile.decoding.as_mut().unwrap().vocabulary = None,
                4 => profile
                    .decoding
                    .as_mut()
                    .unwrap()
                    .vocabulary
                    .as_mut()
                    .unwrap()
                    .ignored_tokens
                    .push("missing".into()),
                5 => {
                    profile
                        .decoding
                        .as_mut()
                        .unwrap()
                        .vocabulary
                        .as_mut()
                        .unwrap()
                        .source = "tokenizer".into()
                }
                6 => {
                    profile.outputs.insert("unknown".into(), "logits".into());
                }
                7 => {
                    let other = profile.clone();
                    metadata.profiles.insert("other".into(), other);
                }
                8 => {
                    metadata
                        .pipeline
                        .as_mut()
                        .unwrap()
                        .workflow
                        .outputs
                        .get_mut("logits")
                        .unwrap()
                        .contract
                        .dtype = "int64".into()
                }
                9 => {
                    metadata
                        .pipeline
                        .as_mut()
                        .unwrap()
                        .workflow
                        .outputs
                        .get_mut("logits")
                        .unwrap()
                        .contract
                        .shape[2] = TensorDimension::Fixed(5)
                }
                10 => {
                    metadata
                        .pipeline
                        .as_mut()
                        .unwrap()
                        .workflow
                        .outputs
                        .get_mut("frame_lengths")
                        .unwrap()
                        .contract
                        .shape[0] = TensorDimension::Fixed(2)
                }
                11 => {
                    metadata
                        .pipeline
                        .as_mut()
                        .unwrap()
                        .workflow
                        .inputs
                        .get_mut("request.audio")
                        .unwrap()
                        .contract
                        .dtype = "float32".into()
                }
                12 => {
                    metadata
                        .pipeline
                        .as_mut()
                        .unwrap()
                        .workflow
                        .inputs
                        .get_mut("request.audio")
                        .unwrap()
                        .role = SemanticInputRole::Opaque
                }
                13 => {
                    let mut extra =
                        metadata.pipeline.as_ref().unwrap().workflow.inputs["request.audio"]
                            .clone();
                    extra.role = SemanticInputRole::Opaque;
                    extra.required = true;
                    metadata
                        .pipeline
                        .as_mut()
                        .unwrap()
                        .workflow
                        .inputs
                        .insert("conditioning".into(), extra);
                }
                _ => unreachable!(),
            }
            let error = transcription_contract(&metadata).unwrap_err();
            assert!(error.len() > 55, "case {case}: {error}");
        }
        let mut metadata = doc();
        let profile = metadata.profiles.get_mut("transcription").unwrap();
        profile.version = "99".into();
        profile.requirement = crate::ProfileRequirement::Ignorable;
        assert!(transcription_contract(&metadata).unwrap().is_none());
    }
}
