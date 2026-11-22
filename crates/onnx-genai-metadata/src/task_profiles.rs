//! Resolved execution contracts for non-generative task profiles.
use crate::{
    InferenceMetadata, ProfileRequirement, RuntimeInputRole, SemanticInputRole, TensorDimension,
    WorkflowInputSource, WorkflowOutputRole,
};

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
pub struct RerankingContractError(String);
fn invalid(message: impl Into<String>) -> RerankingContractError {
    RerankingContractError(message.into())
}

/// Resolve the v1 token-pair ABI from task kind/version, semantic input sources
/// and the explicit score output. No graph/port/model name is guessed.
pub fn reranking_contract(
    metadata: &InferenceMetadata,
) -> Result<Option<RerankingContract>, RerankingContractError> {
    let mut resolved = None;
    for (name, profile) in &metadata.profiles {
        if profile.kind != "reranking" {
            continue;
        }
        if profile.version != "1" {
            if profile.requirement == ProfileRequirement::Ignorable {
                continue;
            }
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
