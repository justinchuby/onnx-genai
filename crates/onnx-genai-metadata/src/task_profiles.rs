//! Resolved execution contracts for non-generative task profiles.
use crate::{
    InferenceMetadata, ProfileRequirement, RuntimeInputRole, SemanticInputRole, TensorDimension,
    WorkflowInputSource, WorkflowOutputRole,
};

/// Unsupported versions of an ignorable task contract have no semantic effect.
pub(crate) fn skip_unsupported_version(profile: &crate::TaskProfile) -> bool {
    profile.requirement == ProfileRequirement::Ignorable
        && matches!(profile.kind.as_str(), "embedding" | "reranking")
        && profile.version != "1"
}

/// A tokenizer-produced feature bound by its declared application role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairFeature {
    AttentionMask,
    TokenTypeIds,
}

/// Proof that a package declares one executable token-pair reranking profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankingContract {
    profile: String,
    score_output: String,
    bindings: Vec<(String, PairFeature)>,
    fixed_sequence_length: Option<usize>,
}
impl RerankingContract {
    pub fn profile(&self) -> &str {
        &self.profile
    }
    pub fn score_output(&self) -> &str {
        &self.score_output
    }
    pub fn bindings(&self) -> &[(String, PairFeature)] {
        &self.bindings
    }
    pub fn fixed_sequence_length(&self) -> Option<usize> {
        self.fixed_sequence_length
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct TaskProfileContractError(String);
fn invalid(message: impl Into<String>) -> TaskProfileContractError {
    TaskProfileContractError(message.into())
}

/// Resolve the v1 token-pair ABI from task kind/version, semantic input sources
/// and the explicit score output. No graph/port/model name is guessed.
pub fn reranking_contract(
    metadata: &InferenceMetadata,
) -> Result<Option<RerankingContract>, TaskProfileContractError> {
    let mut resolved = None;
    for (name, profile) in &metadata.profiles {
        if skip_unsupported_version(profile) {
            continue;
        }
        if profile.kind != "reranking" {
            continue;
        }
        if profile.version != "1" {
            return Err(invalid(format!(
                "profiles.{name}.version '{}' is unsupported for reranking; declare version '1' for the token-pair ABI, or make an unsupported profile ignorable",
                profile.version
            )));
        }
        if resolved.is_some() {
            return Err(invalid(
                "more than one reranking v1 profile is declared; keep one scoring contract per served package",
            ));
        }
        if profile.pooling.is_some() || profile.decoding.is_some() {
            return Err(invalid(format!(
                "profiles.{name} must expose a final scalar score; move pooling/decoding into its workflow before declaring reranking v1"
            )));
        }
        let workflow = &metadata
            .pipeline
            .as_ref()
            .ok_or_else(|| invalid(format!("profiles.{name} requires pipeline.workflow")))?
            .workflow;
        let score=profile.outputs.get("score").ok_or_else(||invalid(format!("profiles.{name}.outputs.score must name the final relevance-probability workflow output")))?;
        let output = workflow.outputs.get(score).ok_or_else(|| {
            invalid(format!(
                "profiles.{name}.outputs.score references undeclared workflow output '{score}'"
            ))
        })?;
        if output.role != WorkflowOutputRole::Tensor
            || !matches!(
                output.contract.dtype.as_str(),
                "float16" | "bfloat16" | "float32"
            )
            || output.contract.shape.len() > 2
            || output
                .contract
                .shape
                .iter()
                .any(|dim| matches!(dim,TensorDimension::Fixed(size) if *size!=1))
        {
            return Err(invalid(format!(
                "profiles.{name}.outputs.score '{score}' must be a floating scalar or one score per single batch row; found {} {:?}",
                output.contract.dtype, output.contract.shape
            )));
        }
        let mut prompt_count = 0;
        let mut bindings = Vec::new();
        let mut fixed_sequence_length = None;
        for (input_name, input) in &workflow.inputs {
            let feature = match (&input.role, &input.source) {
                (
                    SemanticInputRole::Runtime {
                        role: RuntimeInputRole::PromptTokens,
                        ..
                    },
                    WorkflowInputSource::Request,
                ) => {
                    prompt_count += 1;
                    None
                }
                (_, WorkflowInputSource::Application { name: role })
                    if role == "attention_mask" =>
                {
                    Some(PairFeature::AttentionMask)
                }
                (_, WorkflowInputSource::Application { name: role })
                    if role == "token_type_ids" =>
                {
                    Some(PairFeature::TokenTypeIds)
                }
                (_, WorkflowInputSource::Application { name: role })
                    if input.required && input.default.is_none() =>
                {
                    return Err(invalid(format!(
                        "profiles.{name} requires unsupported application input '{input_name}' ({role}); reranking v1 provides attention_mask and token_type_ids from the package tokenizer"
                    )));
                }
                (
                    SemanticInputRole::Runtime {
                        role:
                            RuntimeInputRole::Media
                            | RuntimeInputRole::PromptText
                            | RuntimeInputRole::NegativePromptText
                            | RuntimeInputRole::NegativePromptTokens,
                        ..
                    },
                    _,
                ) if input.required && input.default.is_none() => {
                    return Err(invalid(format!(
                        "profiles.{name} requires input '{input_name}' outside the text token-pair ABI; use a workflow declaring token-pair inputs"
                    )));
                }
                _ => continue,
            };
            if input.contract.dtype != "int64"
                || input.contract.shape.len() != 2
                || matches!(input.contract.shape[0],TensorDimension::Fixed(size) if size!=1)
            {
                return Err(invalid(format!(
                    "profiles.{name} input '{input_name}' must declare int64 [batch,sequence] accepting one batch row; found {} {:?}",
                    input.contract.dtype, input.contract.shape
                )));
            }
            if let TensorDimension::Fixed(size) = input.contract.shape[1] {
                let size=usize::try_from(size).ok().filter(|size|*size>0).ok_or_else(||invalid(format!("profiles.{name} input '{input_name}' requires a positive sequence length")))?;
                if fixed_sequence_length.is_some_and(|previous| previous != size) {
                    return Err(invalid(format!(
                        "profiles.{name} token-pair inputs disagree on fixed sequence length; align their contracts"
                    )));
                }
                fixed_sequence_length = Some(size);
            }
            if let Some(feature) = feature {
                bindings.push((input_name.clone(), feature));
            }
        }
        if prompt_count == 0 {
            return Err(invalid(format!(
                "profiles.{name} must bind a request-sourced prompt_tokens runtime input for the tokenizer's query/document pair"
            )));
        }
        resolved = Some(RerankingContract {
            profile: name.clone(),
            score_output: score.clone(),
            bindings,
            fixed_sequence_length,
        });
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn metadata() -> InferenceMetadata {
        crate::parser::parse_metadata(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/fixtures/onnx_genai_workflows/reranking/inference_metadata.yaml"
            )),
            Some("yaml"),
        )
        .unwrap()
    }
    #[test]
    fn reranking_profile_resolves_semantic_roles_and_rejects_ambiguous_or_incomplete_contracts() {
        let mut document = metadata();
        let contract = reranking_contract(&document).unwrap().unwrap();
        assert_eq!(contract.score_output(), "probability");
        assert_eq!(contract.bindings().len(), 2);
        let profile = document.profiles["relevance"].clone();
        document.profiles.insert("another".into(), profile);
        assert!(
            reranking_contract(&document)
                .unwrap_err()
                .to_string()
                .contains("more than one")
        );
        document.profiles.remove("another");
        document
            .profiles
            .get_mut("relevance")
            .unwrap()
            .outputs
            .remove("score");
        assert!(
            reranking_contract(&document)
                .unwrap_err()
                .to_string()
                .contains("outputs.score")
        );
    }
    #[test]
    fn reranking_profile_versions_and_required_features_are_explicit() {
        let mut document = metadata();
        document.profiles.get_mut("relevance").unwrap().version = "future".into();
        assert!(
            reranking_contract(&document)
                .unwrap_err()
                .to_string()
                .contains("unsupported")
        );
        document.profiles.get_mut("relevance").unwrap().requirement = ProfileRequirement::Ignorable;
        assert!(reranking_contract(&document).unwrap().is_none());
        let mut document = metadata();
        document
            .pipeline
            .as_mut()
            .unwrap()
            .workflow
            .inputs
            .get_mut("pair.mask")
            .unwrap()
            .source = WorkflowInputSource::Application {
            name: "unknown_feature".into(),
        };
        assert!(
            reranking_contract(&document)
                .unwrap_err()
                .to_string()
                .contains("unsupported application input")
        );
    }
    #[test]
    fn reranking_contract_rejects_vector_scores_and_inconsistent_pair_shapes() {
        let mut document = metadata();
        document
            .pipeline
            .as_mut()
            .unwrap()
            .workflow
            .outputs
            .get_mut("probability")
            .unwrap()
            .contract
            .shape = vec![TensorDimension::Fixed(2)];
        let error = reranking_contract(&document).unwrap_err().to_string();
        assert!(error.contains("scalar"), "{error}");
        let mut document = metadata();
        let inputs = &mut document.pipeline.as_mut().unwrap().workflow.inputs;
        inputs.get_mut("pair.mask").unwrap().contract.shape[1] = TensorDimension::Fixed(8);
        inputs.get_mut("pair.segments").unwrap().contract.shape[1] = TensorDimension::Fixed(16);
        let error = reranking_contract(&document).unwrap_err().to_string();
        assert!(error.contains("disagree"), "{error}");
    }
}

/// Resolved workflow embedding output and optional sequence reduction.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingContract {
    pub output: String,
    pub pooling: Option<crate::PoolingSpec>,
}

/// An embedding profile names an executable floating workflow output. A sequence
/// reduction is explicit; without one the output is already one final vector.
pub fn embedding_contract(
    metadata: &InferenceMetadata,
) -> Result<Option<EmbeddingContract>, TaskProfileContractError> {
    let mut resolved = None;
    for (name, profile) in &metadata.profiles {
        if skip_unsupported_version(profile) {
            continue;
        }
        if profile.kind != "embedding" {
            continue;
        }
        if profile.version != "1" {
            return Err(invalid(format!(
                "profiles.{name}.version '{}' is unsupported for embedding; declare version '1' or mark an unsupported profile ignorable",
                profile.version
            )));
        }
        if resolved.is_some() {
            return Err(invalid(
                "more than one embedding v1 profile is declared; retain one embedding output contract per served package",
            ));
        }
        if profile.decoding.is_some() {
            return Err(invalid(format!(
                "profiles.{name}.decoding is incompatible with embedding vectors; declare pooling or emit a final vector"
            )));
        }
        let workflow = &metadata
            .pipeline
            .as_ref()
            .ok_or_else(|| invalid(format!("profiles.{name} requires pipeline.workflow")))?
            .workflow;
        let output_name = profile.outputs.get("embedding").ok_or_else(|| {
            invalid(format!(
                "profiles.{name}.outputs.embedding must name the workflow's embedding tensor"
            ))
        })?;
        let output=workflow.outputs.get(output_name).ok_or_else(||invalid(format!("profiles.{name}.outputs.embedding references undeclared workflow output '{output_name}'")))?;
        let shape = &output.contract.shape;
        if output.role != WorkflowOutputRole::Tensor
            || !matches!(
                output.contract.dtype.as_str(),
                "float16" | "bfloat16" | "float32"
            )
        {
            return Err(invalid(format!(
                "profiles.{name}.outputs.embedding '{output_name}' must be a float16/bfloat16/float32 tensor; declare a floating embedding output"
            )));
        }
        match &profile.pooling {
            Some(pooling) if !(2..=3).contains(&shape.len()) || pooling.axis != shape.len() - 2 => {
                return Err(invalid(format!(
                    "profiles.{name}.pooling.axis must reduce the sequence axis of [sequence,features] or [batch,sequence,features]; found axis {} and shape {shape:?}",
                    pooling.axis
                )));
            }
            None if !(1..=2).contains(&shape.len()) => {
                return Err(invalid(format!(
                    "profiles.{name}.outputs.embedding must be a final [features] or [batch,features] vector without pooling; declare pooling for a sequence tensor"
                )));
            }
            _ => {}
        }
        if shape
            .iter()
            .any(|dim| matches!(dim,TensorDimension::Fixed(n) if *n<=0))
        {
            return Err(invalid(format!(
                "profiles.{name} embedding output '{output_name}' must have positive fixed dimensions; found {shape:?}"
            )));
        }
        let has_batch = shape.len() == if profile.pooling.is_some() { 3 } else { 2 };
        if has_batch && matches!(shape[0],TensorDimension::Fixed(n) if n!=1) {
            return Err(invalid(format!(
                "profiles.{name} output '{output_name}' must accept one request batch row; found {shape:?}"
            )));
        }
        resolved = Some(EmbeddingContract {
            output: output_name.clone(),
            pooling: profile.pooling.clone(),
        });
    }
    Ok(resolved)
}

#[cfg(test)]
mod embedding_tests {
    use super::*;
    fn metadata() -> InferenceMetadata {
        let mut doc = crate::parser::parse_metadata(
            include_str!(
                "../../../tests/fixtures/onnx_genai_workflows/adapter/inference_metadata.yaml"
            ),
            Some("yaml"),
        )
        .unwrap();
        doc.profiles.insert(
            "vectors".into(),
            serde_yaml::from_str("kind: embedding\nversion: '1'\noutputs: {embedding: result}\n")
                .unwrap(),
        );
        doc
    }
    #[test]
    fn embedding_profile_resolves_vectors_and_explicit_pooling() {
        let mut doc = metadata();
        let contract = embedding_contract(&doc).unwrap().unwrap();
        assert_eq!(contract.output, "result");
        assert!(contract.pooling.is_none());
        doc.pipeline
            .as_mut()
            .unwrap()
            .workflow
            .outputs
            .get_mut("result")
            .unwrap()
            .contract
            .shape = vec![
            TensorDimension::Fixed(1),
            TensorDimension::Symbol("positions".into()),
            TensorDimension::Symbol("features".into()),
        ];
        assert!(
            embedding_contract(&doc)
                .unwrap_err()
                .to_string()
                .contains("declare pooling")
        );
        doc.profiles.get_mut("vectors").unwrap().pooling = Some(crate::PoolingSpec {
            kind: crate::PoolingKind::Mean,
            axis: 1,
            normalize: true,
        });
        assert!(
            embedding_contract(&doc)
                .unwrap()
                .unwrap()
                .pooling
                .unwrap()
                .normalize
        );
        doc.profiles
            .get_mut("vectors")
            .unwrap()
            .pooling
            .as_mut()
            .unwrap()
            .axis = 2;
        assert!(
            embedding_contract(&doc)
                .unwrap_err()
                .to_string()
                .contains("sequence axis")
        );
    }
    #[test]
    fn embedding_profile_refuses_ambiguity_versions_types_and_missing_outputs() {
        let mut doc = metadata();
        doc.profiles
            .insert("duplicate".into(), doc.profiles["vectors"].clone());
        assert!(
            embedding_contract(&doc)
                .unwrap_err()
                .to_string()
                .contains("more than one")
        );
        doc.profiles.remove("duplicate");
        doc.profiles.get_mut("vectors").unwrap().version = "2".into();
        assert!(
            embedding_contract(&doc)
                .unwrap_err()
                .to_string()
                .contains("version")
        );
        doc.profiles.get_mut("vectors").unwrap().requirement = ProfileRequirement::Ignorable;
        assert!(embedding_contract(&doc).unwrap().is_none());
        doc.profiles.get_mut("vectors").unwrap().version = "1".into();
        doc.profiles.get_mut("vectors").unwrap().outputs.clear();
        assert!(
            embedding_contract(&doc)
                .unwrap_err()
                .to_string()
                .contains("outputs.embedding")
        );
        doc.profiles
            .get_mut("vectors")
            .unwrap()
            .outputs
            .insert("embedding".into(), "result".into());
        doc.pipeline
            .as_mut()
            .unwrap()
            .workflow
            .outputs
            .get_mut("result")
            .unwrap()
            .contract
            .dtype = "int64".into();
        assert!(
            embedding_contract(&doc)
                .unwrap_err()
                .to_string()
                .contains("float16/bfloat16/float32")
        );
    }
}

#[cfg(test)]
mod ignored_profile_tests {
    #[test]
    fn unsupported_ignorable_profile_skips_its_output_and_decoding_contract() {
        let mut doc = crate::parser::parse_metadata(
            include_str!(
                "../../../tests/fixtures/onnx_genai_workflows/adapter/inference_metadata.yaml"
            ),
            Some("yaml"),
        )
        .unwrap();
        for kind in ["embedding", "reranking"] {
            doc.profiles.insert("future".into(),serde_yaml::from_str(&format!("kind: {kind}\nversion: '99'\nrequirement: ignorable\noutputs: {{embedding: absent}}\npooling: {{kind: mean, axis: 999}}\ndecoding: {{kind: ctc, time_axis: 0, class_axis: 0}}\n")).unwrap());
            crate::validation::validate_metadata(&doc).unwrap();
        }
    }
}
