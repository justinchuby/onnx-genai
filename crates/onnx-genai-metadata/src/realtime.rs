//! Admission of the session-scoped native realtime quantum ABI.
use crate::{
    InferenceMetadata, ProfileRequirement, RealtimeMode, RealtimeSpec, SemanticInputRole,
    TensorDimension, WorkflowInputSource, WorkflowOutputFamily, WorkflowOutputRole,
    WorkflowPublicationMode, WorkflowStateScope,
};
use std::collections::BTreeSet;

/// Operations are transport-independent. Each invocation is one transaction;
/// polling a duplex model never implies end of response or end of session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum RealtimeOperation {
    StartRealtime = 0,
    StartTranscription = 1,
    StartDuplex = 2,
    Audio = 3,
    Commit = 4,
    Text = 5,
    Generate = 6,
    Poll = 7,
    Cancel = 8,
}

/// Proof that named endpoints implement the versioned realtime quantum ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimeContract {
    profile: String,
    spec: RealtimeSpec,
    outputs: std::collections::BTreeMap<String, String>,
    ending_tokens: Vec<u32>,
}
impl RealtimeContract {
    pub fn profile(&self) -> &str {
        &self.profile
    }
    pub fn spec(&self) -> &RealtimeSpec {
        &self.spec
    }
    pub fn output(&self, role: &str) -> Option<&str> {
        self.outputs.get(role).map(String::as_str)
    }
    pub fn ending_tokens(&self) -> &[u32] {
        &self.ending_tokens
    }
    pub fn supports(&self, mode: RealtimeMode) -> bool {
        self.spec.modes.contains(&mode)
    }
}

pub fn realtime_contract(metadata: &InferenceMetadata) -> Result<Option<RealtimeContract>, String> {
    let mut resolved = None;
    for (name, profile) in &metadata.profiles {
        if profile.kind != "realtime" && profile.realtime.is_none() {
            continue;
        }
        crate::version::gate_feature_use(
            crate::version::normalize(metadata.schema_version.as_deref())?,
            crate::version::SchemaFeature::RealtimeSessions,
            &format!("profiles.{name}.realtime"),
        )?;
        if profile.kind == "realtime"
            && profile.version != "1"
            && profile.requirement == ProfileRequirement::Ignorable
        {
            continue;
        }
        let fail = |reason: &str| format!("profiles.{name}.realtime {reason}");
        if profile.kind != "realtime" || profile.version != "1" {
            return Err(fail(
                "requires kind 'realtime' and version '1'; choose a supported quantum ABI",
            ));
        }
        if resolved.is_some() {
            return Err(fail(
                "is ambiguous; declare one realtime profile per served package",
            ));
        }
        let spec = profile.realtime.as_ref().ok_or_else(|| {
            fail("must declare input endpoints, modes, frame size and sample rates")
        })?;
        if spec.modes.is_empty()
            || spec.modes.iter().copied().collect::<BTreeSet<_>>().len() != spec.modes.len()
        {
            return Err(fail(
                "modes must contain distinct supported modes and may not be empty",
            ));
        }
        if !spec.modes.contains(&spec.default_mode) || spec.prompt_template.trim().is_empty() {
            return Err(fail(
                "requires a supported default_mode and nonempty model-authored prompt_template; declare how session settings and text reach the workflow",
            ));
        }
        for (field, selection) in [("voices", &spec.voices), ("languages", &spec.languages)] {
            if let crate::RealtimeSelections::Choices { values } = selection
                && (values.is_empty()
                    || values.iter().any(|value| value.is_empty())
                    || values.iter().collect::<BTreeSet<_>>().len() != values.len())
            {
                return Err(fail(&format!(
                    "{field}.values must contain distinct nonempty selections; use kind unsupported or any for those capabilities"
                )));
            }
        }

        if spec.input_sample_rate_hz == 0
            || spec.input_frame_samples == 0
            || spec.output_sample_rate_hz == Some(0)
        {
            return Err(fail(
                "sample rates and input_frame_samples must be positive; use the codec's declared native rate/frame size",
            ));
        }
        if profile.pooling.is_some()
            || profile.decoding.is_some()
            || profile.reranking.is_some()
            || profile.decision.is_some()
        {
            return Err(fail(
                "cannot combine pooling, decoding or another task ABI; publish final PCM/token quanta in the workflow",
            ));
        }
        let workflow = &metadata
            .pipeline
            .as_ref()
            .ok_or_else(|| fail("requires pipeline.workflow"))?
            .workflow;
        if workflow.publication_mode != WorkflowPublicationMode::CommitOnly {
            return Err(fail(
                "requires commit_only publication; commit each bounded quantum before sending its output",
            ));
        }
        if !workflow
            .state
            .values()
            .any(|cell| cell.scope == WorkflowStateScope::Session)
        {
            return Err(fail(
                "requires session-scoped workflow state for codec/conversation continuity; declare state with scope: session",
            ));
        }
        if let Some((input, _)) = workflow.inputs.iter().find(|(_, input)| {
            matches!(input.role, SemanticInputRole::Runtime { .. })
                || matches!(input.source, WorkflowInputSource::Request)
        }) {
            return Err(fail(&format!(
                "input '{input}' uses ordinary generation request binding; realtime quanta must bind explicit opaque application endpoints, including output_token_budget, rather than inherit max_new_tokens or generation-loop defaults"
            )));
        }
        let mut bound = BTreeSet::new();
        for (endpoint, dtype, scalar) in [
            (&spec.inputs.operation, "int64", true),
            (&spec.inputs.output_token_budget, "int64", true),
            (&spec.inputs.audio, "float32", false),
            (&spec.inputs.tokens, "int64", false),
        ] {
            if !bound.insert(endpoint) {
                return Err(fail("input endpoints must be distinct"));
            }
            let input = workflow
                .inputs
                .get(endpoint)
                .ok_or_else(|| fail(&format!("references undeclared input '{endpoint}'")))?;
            let shape_ok = if scalar {
                input.contract.shape.is_empty()
            } else {
                matches!(
                    input.contract.shape.as_slice(),
                    [TensorDimension::Symbol(_)]
                )
            };
            if !matches!(input.source, WorkflowInputSource::Application { .. })
                || input.role != SemanticInputRole::Opaque
                || input.contract.dtype != dtype
                || !shape_ok
                || !input.required
            {
                return Err(fail(&format!(
                    "input '{endpoint}' must be a required opaque application {dtype} {} endpoint; use a symbolic vector length for empty/partial quanta",
                    if scalar { "scalar" } else { "vector" }
                )));
            }
        }
        if let Some(endpoint) = &spec.inputs.token_mask {
            if !bound.insert(endpoint) {
                return Err(fail("input endpoints must be distinct"));
            }
            let input = workflow.inputs.get(endpoint).ok_or_else(|| {
                fail(&format!(
                    "references undeclared token_mask input '{endpoint}'"
                ))
            })?;
            if !matches!(input.source, WorkflowInputSource::Application { .. })
                || input.role != SemanticInputRole::Opaque
                || input.contract.dtype != "bool"
                || !matches!(
                    input.contract.shape.as_slice(),
                    [TensorDimension::Symbol(_)]
                )
                || !input.required
            {
                return Err(fail(
                    "token_mask must be a required opaque application bool symbolic vector; consume each vocabulary bit before selecting the quantum's single text token",
                ));
            }
        }
        let mut output_names = BTreeSet::new();
        for (role, dtype, scalar, required) in [
            ("tokens", "int64", false, true),
            (
                "audio",
                "float32",
                false,
                spec.output_sample_rate_hz.is_some(),
            ),
            ("done", "bool", true, true),
            ("needs_input", "bool", true, true),
            ("prompt_tokens", "int64", true, true),
            ("output_tokens", "int64", true, true),
        ] {
            let Some(endpoint) = profile.outputs.get(role) else {
                if required {
                    return Err(fail(&format!(
                        "outputs.{role} is required; expose the model-authored quantum result"
                    )));
                }
                continue;
            };
            if role == "audio" && spec.output_sample_rate_hz.is_none() {
                return Err(fail("audio output requires output_sample_rate_hz"));
            }
            if !output_names.insert(endpoint) {
                return Err(fail("output endpoints must be distinct"));
            }
            let output = workflow
                .outputs
                .get(endpoint)
                .ok_or_else(|| fail(&format!("references undeclared output '{endpoint}'")))?;
            let shape_ok = if scalar {
                output.contract.shape.is_empty()
            } else {
                matches!(
                    output.contract.shape.as_slice(),
                    [TensorDimension::Symbol(_)]
                )
            };
            if output.contract.dtype != dtype
                || !shape_ok
                || output.role != WorkflowOutputRole::Tensor
                || output.family != WorkflowOutputFamily::Materialized
            {
                return Err(fail(&format!(
                    "output '{endpoint}' must be a materialized tensor {dtype} {}; publish mono PCM samples or discrete token IDs, not an encoded container",
                    if scalar { "scalar" } else { "symbolic vector" }
                )));
            }
        }
        if profile.outputs.keys().any(|role| {
            !matches!(
                role.as_str(),
                "tokens" | "audio" | "done" | "needs_input" | "prompt_tokens" | "output_tokens"
            )
        }) {
            return Err(fail(
                "contains an unknown output role; use tokens, audio, done, needs_input, prompt_tokens or output_tokens",
            ));
        }
        resolved = Some(RealtimeContract {
            profile: name.clone(),
            spec: spec.clone(),
            outputs: profile.outputs.clone(),
            ending_tokens: metadata
                .package
                .as_ref()
                .and_then(|p| p.tokenizer.as_ref())
                .and_then(|t| t.special_tokens.as_ref())
                .map(|t| t.eos_token_id.clone())
                .unwrap_or_default(),
        });
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> InferenceMetadata {
        serde_yaml::from_str(include_str!(
            "../../../tests/fixtures/onnx_genai_workflows/realtime/inference_metadata.yaml"
        ))
        .unwrap()
    }
    #[test]
    fn realtime_token_mask_requires_a_distinct_boolean_application_endpoint() {
        let mut doc = fixture();
        doc.profiles
            .get_mut("live")
            .unwrap()
            .realtime
            .as_mut()
            .unwrap()
            .inputs
            .token_mask = Some("request.mask".into());
        assert!(
            realtime_contract(&doc)
                .unwrap_err()
                .contains("undeclared token_mask")
        );
        let mut input = doc.pipeline.as_ref().unwrap().workflow.inputs["request.tokens"].clone();
        input.contract.dtype = "bool".into();
        doc.pipeline
            .as_mut()
            .unwrap()
            .workflow
            .inputs
            .insert("request.mask".into(), input);
        assert_eq!(
            realtime_contract(&doc)
                .unwrap()
                .unwrap()
                .spec()
                .inputs
                .token_mask
                .as_deref(),
            Some("request.mask")
        );
        doc.pipeline
            .as_mut()
            .unwrap()
            .workflow
            .inputs
            .get_mut("request.mask")
            .unwrap()
            .contract
            .dtype = "int64".into();
        assert!(
            realtime_contract(&doc)
                .unwrap_err()
                .contains("bool symbolic vector")
        );
    }
    #[test]
    fn realtime_admission_resolves_model_authored_rates_and_modes() {
        let contract = realtime_contract(&fixture()).unwrap().unwrap();
        assert_eq!(contract.spec().input_sample_rate_hz, 16000);
        assert_eq!(contract.spec().output_sample_rate_hz, Some(24000));
        assert!(contract.supports(RealtimeMode::Duplex));
        assert_eq!(contract.output("done"), Some("done"));
    }
    #[test]
    fn realtime_admission_rejects_missing_state_and_inconsistent_endpoints() {
        let mut doc = fixture();
        doc.pipeline.as_mut().unwrap().workflow.state.clear();
        assert!(
            realtime_contract(&doc)
                .unwrap_err()
                .contains("session-scoped")
        );
        let mut doc = fixture();
        doc.profiles
            .get_mut("live")
            .unwrap()
            .realtime
            .as_mut()
            .unwrap()
            .inputs
            .audio = "request.operation".into();
        assert!(realtime_contract(&doc).unwrap_err().contains("distinct"));
        let mut doc = fixture();
        doc.profiles.get_mut("live").unwrap().outputs.remove("done");
        assert!(
            realtime_contract(&doc)
                .unwrap_err()
                .contains("outputs.done")
        );
        let mut doc = fixture();
        doc.profiles
            .get_mut("live")
            .unwrap()
            .realtime
            .as_mut()
            .unwrap()
            .input_sample_rate_hz = 0;
        assert!(realtime_contract(&doc).unwrap_err().contains("positive"));
    }
    #[test]
    fn realtime_profile_requires_version_floor_and_skips_unknown_ignorable_abi() {
        let mut doc = fixture();
        doc.schema_version = Some("v1.12".into());
        assert!(realtime_contract(&doc).unwrap_err().contains("v1.13"));
        let mut doc = fixture();
        let profile = doc.profiles.get_mut("live").unwrap();
        profile.version = "99".into();
        profile.requirement = ProfileRequirement::Ignorable;
        assert!(realtime_contract(&doc).unwrap().is_none());
    }
}
