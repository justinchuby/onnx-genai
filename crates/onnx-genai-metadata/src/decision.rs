//! Model-independent, non-generative decision workflow admission.
use crate::{
    DecisionSpec, InferenceMetadata, RuntimeInputRole, SemanticInputRole, TensorDimension,
    WorkflowInputSource, WorkflowOutputRole,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionFeature {
    AttentionMask,
    TokenTypeIds,
    OptionCount { vector: bool },
}
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionContract {
    profile: String,
    output: String,
    spec: DecisionSpec,
    bindings: Vec<(String, DecisionFeature)>,
    sequence_length: Option<usize>,
    option_count: Option<usize>,
}
impl DecisionContract {
    pub fn profile(&self) -> &str {
        &self.profile
    }
    pub fn output(&self) -> &str {
        &self.output
    }
    pub fn spec(&self) -> &DecisionSpec {
        &self.spec
    }
    pub fn bindings(&self) -> &[(String, DecisionFeature)] {
        &self.bindings
    }
    pub fn sequence_length(&self) -> Option<usize> {
        self.sequence_length
    }
    pub fn option_count(&self) -> Option<usize> {
        self.option_count
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct DecisionContractError(String);
fn invalid(message: impl Into<String>) -> DecisionContractError {
    DecisionContractError(message.into())
}

pub fn decision_contract(
    metadata: &InferenceMetadata,
) -> Result<Option<DecisionContract>, DecisionContractError> {
    let mut result = None;
    for (name, profile) in &metadata.profiles {
        if profile.decision.is_some() {
            let version =
                crate::version::normalize(metadata.schema_version.as_deref()).map_err(invalid)?;
            crate::version::gate_feature_use(
                version,
                crate::version::SchemaFeature::DecisionProfiles,
                &format!("profiles.{name}.decision"),
            )
            .map_err(invalid)?;
        }
        if crate::task_profiles::skip_unsupported_version(profile) {
            continue;
        }
        if profile.kind != "decision" {
            if profile.decision.is_some() {
                return Err(invalid(format!(
                    "profiles.{name}.decision requires kind 'decision'; move decision prompting to its own task profile"
                )));
            }
            continue;
        }
        if profile.version != "1" {
            return Err(invalid(format!(
                "profiles.{name}.version '{}' is unsupported for decision; use version '1' or mark an unsupported profile ignorable",
                profile.version
            )));
        }
        if result.is_some() {
            return Err(invalid(
                "multiple decision profiles are ambiguous; declare one decision v1 workflow contract per package",
            ));
        }
        let spec=profile.decision.as_ref().ok_or_else(||invalid(format!("profiles.{name}.decision must declare prompt_template, max_options, variants, confidence and score policies")))?;
        if spec.prompt_template.trim().is_empty()
            || profile.pooling.is_some()
            || profile.decoding.is_some()
        {
            return Err(invalid(format!(
                "profiles.{name} needs a nonempty decision.prompt_template and final probabilities without pooling or decoding"
            )));
        }
        let workflow = &metadata
            .pipeline
            .as_ref()
            .ok_or_else(|| invalid(format!("profiles.{name} requires pipeline.workflow")))?
            .workflow;
        if workflow.outputs.values().any(|output| {
            matches!(
                output.role,
                WorkflowOutputRole::Tokens | WorkflowOutputRole::Text
            )
        }) {
            return Err(invalid(format!(
                "profiles.{name} decision v1 must execute a non-generative workflow without text/token outputs; expose a separate decision-head workflow instead of reporting generated tokens as decisions"
            )));
        }
        if profile.outputs.len() != 1 || !profile.outputs.contains_key("probabilities") {
            return Err(invalid(format!(
                "profiles.{name}.outputs must name exactly probabilities; compute calibrated option probabilities in the workflow"
            )));
        }
        let output = &profile.outputs["probabilities"];
        let declaration=workflow.outputs.get(output).ok_or_else(||invalid(format!("profiles.{name}.outputs.probabilities references missing workflow output '{output}'")))?;
        let shape = &declaration.contract.shape;
        if declaration.role != WorkflowOutputRole::Tensor
            || !matches!(
                declaration.contract.dtype.as_str(),
                "float16" | "bfloat16" | "float32"
            )
            || !(1..=2).contains(&shape.len())
            || (shape.len() == 2 && matches!(shape[0],TensorDimension::Fixed(n) if n!=1))
        {
            return Err(invalid(format!(
                "profiles.{name}.outputs.probabilities '{output}' must be a floating [options] or [1,options] tensor; found {} {shape:?}",
                declaration.contract.dtype
            )));
        }
        let option_count=match shape.last().unwrap() {
            TensorDimension::Fixed(n)=>Some(usize::try_from(*n).ok().filter(|n|*n>0 && *n<=spec.max_options.get()).ok_or_else(||invalid(format!("profiles.{name} probability dimension must be positive and within decision.max_options")))?),
            _=>None,
        };
        let mut prompt_count = 0;
        let mut bindings = Vec::new();
        let mut sequence_length = None;
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
                (_, WorkflowInputSource::Application { name }) if name == "attention_mask" => {
                    Some(DecisionFeature::AttentionMask)
                }
                (_, WorkflowInputSource::Application { name }) if name == "token_type_ids" => {
                    Some(DecisionFeature::TokenTypeIds)
                }
                (_, WorkflowInputSource::Application { name }) if name == "option_count" => {
                    Some(DecisionFeature::OptionCount {
                        vector: input.contract.shape.len() == 1,
                    })
                }
                _ if input.required && input.default.is_none() => {
                    return Err(invalid(format!(
                        "profiles.{name} required input '{input_name}' is outside decision v1; use request prompt_tokens or application attention_mask, token_type_ids, option_count"
                    )));
                }
                _ => continue,
            };
            let shape = &input.contract.shape;
            let scalar = matches!(feature, Some(DecisionFeature::OptionCount { .. }));
            if input.contract.dtype != "int64"
                || (scalar && !(shape.is_empty() || shape == &[TensorDimension::Fixed(1)]))
                || (!scalar
                    && (shape.len() != 2 || matches!(shape[0],TensorDimension::Fixed(n) if n!=1)))
            {
                return Err(invalid(format!(
                    "profiles.{name} input '{input_name}' must be int64 {} ; found {} {shape:?}",
                    if scalar {
                        "scalar or [1] option_count"
                    } else {
                        "[1,sequence] tokenizer feature"
                    },
                    input.contract.dtype
                )));
            }
            if !scalar && let TensorDimension::Fixed(n) = shape[1] {
                let length = usize::try_from(n).ok().filter(|n| *n > 0).ok_or_else(|| {
                    invalid(format!(
                        "profiles.{name} input '{input_name}' needs a positive sequence dimension"
                    ))
                })?;
                if sequence_length.is_some_and(|old| old != length) {
                    return Err(invalid(format!(
                        "profiles.{name} tokenizer inputs disagree on fixed sequence length; align their contracts"
                    )));
                }
                sequence_length = Some(length);
            }
            if let Some(feature) = feature {
                bindings.push((input_name.clone(), feature));
            }
        }
        if prompt_count == 0 {
            return Err(invalid(format!(
                "profiles.{name} requires request-sourced runtime prompt_tokens; declare a tokenizer input"
            )));
        }
        result = Some(DecisionContract {
            profile: name.clone(),
            output: output.clone(),
            spec: spec.clone(),
            bindings,
            sequence_length,
            option_count,
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn document() -> InferenceMetadata {
        crate::parser::parse_metadata(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/fixtures/onnx_genai_workflows/decision/inference_metadata.yaml"
            )),
            Some("yaml"),
        )
        .unwrap()
    }
    #[test]
    fn decision_contract_resolves_declared_cardinality_and_skips_ignorable_future_versions() {
        let mut doc = document();
        let contract = decision_contract(&doc).unwrap().unwrap();
        assert_eq!(contract.option_count(), Some(2));
        assert_eq!(contract.output(), "decision.probabilities");
        let profile = doc.profiles.get_mut("decision_head").unwrap();
        profile.version = "future".into();
        assert!(
            decision_contract(&doc)
                .unwrap_err()
                .to_string()
                .contains("version")
        );
        doc.profiles.get_mut("decision_head").unwrap().requirement =
            crate::ProfileRequirement::Ignorable;
        assert!(decision_contract(&doc).unwrap().is_none());
    }
    #[test]
    fn decision_declarations_require_their_authored_schema_version_in_parser_and_typed_admission() {
        let mut doc = document();
        doc.schema_version = Some("v1.9".into());
        assert!(
            decision_contract(&doc)
                .unwrap_err()
                .to_string()
                .contains("minimum schema version v1.10")
        );
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/onnx_genai_workflows/decision/inference_metadata.yaml"
        ))
        .replace("schema_version: v1.10", "schema_version: v1.9");
        let error = crate::parser::parse_metadata(&source, Some("yaml"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("minimum schema version v1.10"), "{error}");
    }

    #[test]
    fn decision_profile_rejects_unexecutable_declarations_at_admission() {
        for case in 0..5 {
            let mut doc = document();
            match case {
                0 => doc.profiles.get_mut("decision_head").unwrap().decision = None,
                1 => doc.profiles.get_mut("decision_head").unwrap().kind = "embedding".into(),
                2 => {
                    doc.pipeline
                        .as_mut()
                        .unwrap()
                        .workflow
                        .outputs
                        .get_mut("decision.probabilities")
                        .unwrap()
                        .contract
                        .dtype = "int64".into()
                }
                3 => {
                    doc.pipeline
                        .as_mut()
                        .unwrap()
                        .workflow
                        .inputs
                        .get_mut("question.ids")
                        .unwrap()
                        .source = WorkflowInputSource::Application {
                        name: "unknown".into(),
                    }
                }
                _ => {
                    let mut profile = doc.profiles["decision_head"].clone();
                    profile.decision.as_mut().unwrap().prompt_template.clear();
                    doc.profiles.insert("decision_head".into(), profile);
                }
            }
            assert!(decision_contract(&doc).is_err(), "case {case}");
        }
    }
}
