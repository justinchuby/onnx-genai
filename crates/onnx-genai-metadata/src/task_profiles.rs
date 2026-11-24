//! Resolved execution contracts for non-generative task profiles.
use crate::{
    InferenceMetadata, ProfileRequirement, RuntimeInputRole, SemanticInputRole, TensorDimension,
    WorkflowInputSource, WorkflowOutputRole,
};

/// Unsupported versions of an ignorable task contract have no semantic effect.
pub(crate) fn skip_unsupported_version(profile: &crate::TaskProfile) -> bool {
    let supported = match profile.kind.as_str() {
        "embedding" | "decision" => profile.version == "1",
        "reranking" => matches!(profile.version.as_str(), "1" | "2"),
        _ => return false,
    };
    profile.requirement == ProfileRequirement::Ignorable && !supported
}

/// A tokenizer-produced feature bound by its declared application role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairFeature {
    AttentionMask,
    TokenTypeIds,
}

/// One encoded-image endpoint owned by a reranking profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankingImageInput {
    endpoint: String,
    required: bool,
    encoded_bytes: Option<usize>,
}
impl RerankingImageInput {
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
    pub fn admit_bytes(&self, bytes: usize) -> Result<(), TaskProfileContractError> {
        if bytes == 0 {
            return Err(invalid(format!(
                "reranking image '{}' is empty; supply nonempty encoded image bytes",
                self.endpoint
            )));
        }
        if let Some(expected) = self.encoded_bytes
            && expected != bytes
        {
            return Err(invalid(format!(
                "reranking image '{}' contains {bytes} bytes, but its declared input requires {expected}; supply an image matching the package's byte contract",
                self.endpoint
            )));
        }
        Ok(())
    }
}

/// Proof that a package declares one executable token-pair reranking profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankingContract {
    profile: String,
    score_output: String,
    prompt_tokens_output: Option<String>,
    bindings: Vec<(String, PairFeature)>,
    fixed_sequence_length: Option<usize>,
    query_images: Vec<RerankingImageInput>,
    document_images: Vec<RerankingImageInput>,
}
impl RerankingContract {
    pub fn profile(&self) -> &str {
        &self.profile
    }
    pub fn score_output(&self) -> &str {
        &self.score_output
    }
    pub fn prompt_tokens_output(&self) -> Option<&str> {
        self.prompt_tokens_output.as_deref()
    }
    pub fn bindings(&self) -> &[(String, PairFeature)] {
        &self.bindings
    }
    pub fn fixed_sequence_length(&self) -> Option<usize> {
        self.fixed_sequence_length
    }
    pub fn query_images(&self) -> &[RerankingImageInput] {
        &self.query_images
    }
    pub fn document_images(&self) -> &[RerankingImageInput] {
        &self.document_images
    }
    pub fn admit_images(
        &self,
        query: usize,
        document: usize,
    ) -> Result<(), TaskProfileContractError> {
        for (side, count, bindings) in [
            ("query", query, &self.query_images),
            ("document", document, &self.document_images),
        ] {
            if count > bindings.len() {
                return Err(invalid(format!(
                    "{side}.image contains {count} images but reranking profile '{}' declares {} ordered image inputs; choose a compatible package or reduce images",
                    self.profile,
                    bindings.len()
                )));
            }
            if let Some(binding) = bindings.iter().skip(count).find(|binding| binding.required) {
                return Err(invalid(format!(
                    "{side}.image omits required image input '{}' in reranking profile '{}'; supply the declared ordered images",
                    binding.endpoint, self.profile
                )));
            }
        }
        Ok(())
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct TaskProfileContractError(String);
fn invalid(message: impl Into<String>) -> TaskProfileContractError {
    TaskProfileContractError(message.into())
}

/// Resolve text/image pair ABIs from task kind/version, semantic input sources
/// and the explicit score output. No graph/port/model name is guessed.
pub fn reranking_contract(
    metadata: &InferenceMetadata,
) -> Result<Option<RerankingContract>, TaskProfileContractError> {
    let mut resolved = None;
    for (name, profile) in &metadata.profiles {
        if profile.reranking.is_some() || (profile.kind == "reranking" && profile.version == "2") {
            let version =
                crate::version::normalize(metadata.schema_version.as_deref()).map_err(invalid)?;
            crate::version::gate_feature_use(
                version,
                crate::version::SchemaFeature::MultimodalReranking,
                &format!("profiles.{name}.reranking"),
            )
            .map_err(invalid)?;
        }
        if skip_unsupported_version(profile) {
            continue;
        }
        if let Some(spec) = &profile.reranking {
            if profile.kind != "reranking" || profile.version != "2" {
                return Err(invalid(format!(
                    "profiles.{name}.reranking requires kind 'reranking' and version '2'"
                )));
            }
            if spec.query_images.is_empty() && spec.document_images.is_empty() {
                return Err(invalid(format!(
                    "profiles.{name}.reranking must declare at least one query_images/document_images endpoint; use reranking v1 for text-only pairs"
                )));
            }
        }
        if profile.kind != "reranking" {
            continue;
        }
        if !matches!(profile.version.as_str(), "1" | "2") {
            return Err(invalid(format!(
                "profiles.{name}.version '{}' is unsupported for reranking; declare version '1' for text pairs or '2' with explicit image endpoints, or make an unsupported profile ignorable",
                profile.version
            )));
        }
        if resolved.is_some() {
            return Err(invalid(
                "more than one reranking profile is declared; keep one scoring contract per served package",
            ));
        }
        if profile.pooling.is_some() || profile.decoding.is_some() {
            return Err(invalid(format!(
                "profiles.{name} must expose a final scalar score; move pooling/decoding into its workflow before declaring reranking"
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
        let prompt_tokens_output = if profile.version == "2" {
            let name = profile.outputs.get("prompt_tokens").ok_or_else(|| invalid(format!("profiles.{name}.outputs.prompt_tokens is required for reranking v2; the workflow must report its complete token usage, including model-authored media expansion")))?;
            let output = workflow.outputs.get(name).ok_or_else(|| invalid(format!("reranking token-usage output '{name}' is undeclared; expose it in pipeline.workflow.outputs")))?;
            if output.role != WorkflowOutputRole::Tensor
                || output.contract.dtype != "int64"
                || output.contract.shape.len() > 2
                || output
                    .contract
                    .shape
                    .iter()
                    .any(|dim| matches!(dim, TensorDimension::Fixed(size) if *size != 1))
            {
                return Err(invalid(format!(
                    "reranking token-usage output '{name}' must be an int64 scalar or one count per single batch row"
                )));
            }
            Some(name.clone())
        } else {
            None
        };
        let mut media_names = std::collections::BTreeSet::new();
        let mut resolve_images =
            |names: &[String]| -> Result<Vec<RerankingImageInput>, TaskProfileContractError> {
                names.iter().map(|input_name| {
                if !media_names.insert(input_name.clone()) {
                    return Err(invalid(format!("profiles.{name}.reranking binds input '{input_name}' more than once; use a distinct workflow endpoint for each image position")));
                }
                let input = workflow.inputs.get(input_name).ok_or_else(|| invalid(format!("profiles.{name}.reranking references undeclared workflow input '{input_name}'")))?;
                if !matches!(input.source, WorkflowInputSource::Application { .. })
                    || input.role != SemanticInputRole::Opaque
                    || input.contract.dtype != "uint8" || input.contract.shape.len() != 1 {
                    return Err(invalid(format!("profiles.{name}.reranking input '{input_name}' must be an opaque application-sourced uint8 [encoded_bytes] vector; the profile owns binding and its workflow owns image decoding")));
                }
                let encoded_bytes = if let TensorDimension::Fixed(size) = input.contract.shape[0] {
                    Some(usize::try_from(size).ok().filter(|size| *size > 0).ok_or_else(|| invalid(format!("profiles.{name}.reranking input '{input_name}' requires a positive encoded byte length")))?)
                } else { None };
                Ok(RerankingImageInput { endpoint: input_name.clone(), required: input.required && input.default.is_none(), encoded_bytes })
            }).collect()
            };
        let (query_images, document_images) = if profile.version == "2" {
            let spec = profile.reranking.as_ref().ok_or_else(|| invalid(format!("profiles.{name} reranking v2 requires an explicit reranking image-input contract")))?;
            (
                resolve_images(&spec.query_images)?,
                resolve_images(&spec.document_images)?,
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let mut prompt_count = 0;
        let mut bindings = Vec::new();
        let mut fixed_sequence_length = None;
        for (input_name, input) in &workflow.inputs {
            if media_names.contains(input_name) {
                continue;
            }
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
                        "profiles.{name} requires unsupported application input '{input_name}' ({role}); reranking provides attention_mask/token_type_ids from the tokenizer and v2 image inputs through its explicit contract"
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
                        "profiles.{name} requires input '{input_name}' outside the declared pair-scoring ABI; bind tokenizer features or explicit application image inputs"
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
            prompt_tokens_output,
            bindings,
            fixed_sequence_length,
            query_images,
            document_images,
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
    fn image_preprocessing_reuse_is_versioned_and_every_port_stays_bound() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/onnx_genai_workflows/reranking_multimodal/inference_metadata.yaml"
        ));
        let document = crate::parse_metadata(source, Some("yaml")).unwrap();
        crate::validate_metadata(&document).unwrap();
        let legacy = source
            .replace(
                "onnx-genai.image-preprocess: '2'",
                "onnx-genai.image-preprocess: '1'",
            )
            .replace(
                "abi: onnx-genai.image-preprocess, version: '2'",
                "abi: onnx-genai.image-preprocess, version: '1'",
            );
        let document = crate::parse_metadata(&legacy, Some("yaml")).unwrap();
        let errors = crate::validate_metadata(&document).unwrap_err().join("; ");
        assert!(errors.contains("exactly one invocation"), "{errors}");
        let document = crate::parse_metadata(
            &source.replace("name: pixel_values", "name: unbound_pixels"),
            Some("yaml"),
        )
        .unwrap();
        let errors = crate::validate_metadata(&document).unwrap_err().join("; ");
        assert!(errors.contains("unbound_pixels"), "{errors}");
    }

    fn multimodal_metadata() -> InferenceMetadata {
        let mut document = metadata();
        document.schema_version = Some("v1.12".into());
        let profile = document.profiles.get_mut("relevance").unwrap();
        profile.version = "2".into();
        profile.reranking = Some(crate::RerankingSpec {
            query_images: vec!["request.query_picture".into()],
            document_images: vec!["request.doc_first".into(), "request.doc_second".into()],
        });
        document
            .profiles
            .get_mut("relevance")
            .unwrap()
            .outputs
            .insert("prompt_tokens".into(), "usage".into());
        let mut usage = document.pipeline.as_ref().unwrap().workflow.outputs["probability"].clone();
        usage.contract.dtype = "int64".into();
        document
            .pipeline
            .as_mut()
            .unwrap()
            .workflow
            .outputs
            .insert("usage".into(), usage);
        let inputs = &mut document.pipeline.as_mut().unwrap().workflow.inputs;
        for endpoint in [
            "request.query_picture",
            "request.doc_first",
            "request.doc_second",
        ] {
            let input: crate::WorkflowInput = serde_yaml::from_str(&format!(
                "contract: {{ dtype: uint8, shape: [encoded_bytes] }}\nrole: {{ kind: opaque }}\nsource: {{ kind: application, name: '{endpoint}' }}\nrequired: true"
            )).unwrap();
            inputs.insert(endpoint.into(), input);
        }
        document
    }
    #[test]
    fn reranking_images_have_explicit_ordered_capacity_and_required_positions() {
        let mut document = multimodal_metadata();
        let contract = reranking_contract(&document).unwrap().unwrap();
        assert_eq!(
            contract.document_images()[1].endpoint(),
            "request.doc_second"
        );
        contract.admit_images(1, 2).unwrap();
        assert!(
            contract
                .admit_images(0, 2)
                .unwrap_err()
                .to_string()
                .contains("query.image")
        );
        assert!(
            contract
                .admit_images(1, 3)
                .unwrap_err()
                .to_string()
                .contains("declares 2")
        );
        let inputs = &mut document.pipeline.as_mut().unwrap().workflow.inputs;
        inputs.get_mut("request.doc_second").unwrap().required = false;
        reranking_contract(&document)
            .unwrap()
            .unwrap()
            .admit_images(1, 1)
            .unwrap();
    }
    #[test]
    fn reranking_images_reject_aliases_missing_inputs_and_untyped_bindings() {
        let mut document = multimodal_metadata();
        document
            .profiles
            .get_mut("relevance")
            .unwrap()
            .reranking
            .as_mut()
            .unwrap()
            .document_images[0] = "request.query_picture".into();
        assert!(
            reranking_contract(&document)
                .unwrap_err()
                .to_string()
                .contains("more than once")
        );
        let mut document = multimodal_metadata();
        document
            .pipeline
            .as_mut()
            .unwrap()
            .workflow
            .inputs
            .remove("request.doc_first");
        assert!(
            reranking_contract(&document)
                .unwrap_err()
                .to_string()
                .contains("undeclared workflow input")
        );
        let mut document = multimodal_metadata();
        document
            .pipeline
            .as_mut()
            .unwrap()
            .workflow
            .inputs
            .get_mut("request.doc_first")
            .unwrap()
            .contract
            .dtype = "float32".into();
        assert!(
            reranking_contract(&document)
                .unwrap_err()
                .to_string()
                .contains("uint8")
        );
    }
    #[test]
    fn multimodal_reranking_requires_authoritative_integer_token_usage() {
        let mut document = multimodal_metadata();
        document
            .profiles
            .get_mut("relevance")
            .unwrap()
            .outputs
            .remove("prompt_tokens");
        assert!(
            reranking_contract(&document)
                .unwrap_err()
                .to_string()
                .contains("outputs.prompt_tokens is required")
        );
        let mut document = multimodal_metadata();
        document
            .pipeline
            .as_mut()
            .unwrap()
            .workflow
            .outputs
            .get_mut("usage")
            .unwrap()
            .contract
            .dtype = "float32".into();
        assert!(
            reranking_contract(&document)
                .unwrap_err()
                .to_string()
                .contains("int64 scalar")
        );
    }

    #[test]
    fn reranking_image_contract_has_a_schema_floor_and_profile_version() {
        let mut document = multimodal_metadata();
        document.schema_version = Some("v1.11".into());
        assert!(
            reranking_contract(&document)
                .unwrap_err()
                .to_string()
                .contains("v1.12")
        );
        let mut document = multimodal_metadata();
        document.profiles.get_mut("relevance").unwrap().version = "1".into();
        assert!(
            reranking_contract(&document)
                .unwrap_err()
                .to_string()
                .contains("version '2'")
        );
        let mut document = multimodal_metadata();
        let profile = document.profiles.get_mut("relevance").unwrap();
        profile.version = "future".into();
        profile.requirement = ProfileRequirement::Ignorable;
        assert!(reranking_contract(&document).unwrap().is_none());
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
