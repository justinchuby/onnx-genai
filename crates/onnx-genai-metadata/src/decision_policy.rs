//! Decision request rules are metadata, not model-family heuristics.
use crate::*;
use serde_json::Value;
use std::{collections::BTreeSet, num::NonZeroUsize};

impl DecisionPayload {
    pub fn validate(self, value: &Value, field: &str) -> Result<(), String> {
        let payload = value.is_string() || value.is_object() || value.is_array();
        let valid = match self {
            Self::Any => true,
            Self::AnyNonNull => !value.is_null(),
            Self::Payload => payload,
            Self::PayloadOrNull => payload || value.is_null(),
            Self::NonemptyPayload => payload && value.as_str() != Some(""),
            Self::Text => value.is_string(),
        };
        if valid {
            return Ok(());
        }
        let expected = match self {
            Self::Any => "a JSON value",
            Self::AnyNonNull => "a non-null JSON value",
            Self::Payload => "text, an object or an array",
            Self::PayloadOrNull => "text, an object, an array or null",
            Self::NonemptyPayload => "nonempty text, an object or an array",
            Self::Text => "text",
        };
        Err(format!(
            "{field} must be {expected} under the model's decision input policy; supply an accepted value"
        ))
    }
}
impl DecisionInputPolicy {
    /// The published schema v1.10 request contract, used only by that version.
    pub(crate) fn schema_v1_10() -> Self {
        Self {
            state: DecisionPayload::Any,
            instructions: DecisionInstructions::Required {
                payload: DecisionPayload::Any,
            },
            default_type: None,
            max_questions: None,
            choice: Some(DecisionChoices::Dynamic {
                minimum: NonZeroUsize::new(1).unwrap(),
                order: DecisionOrder::Lexical,
                descriptions: DecisionPayload::Any,
                nonempty_keys: true,
            }),
            score: Some(DecisionLevels {
                minimum: NonZeroUsize::new(2).unwrap(),
                maximum: NonZeroUsize::new(10).unwrap(),
                descriptions: DecisionPayload::Any,
                legend: DecisionLegend::Structured,
            }),
            noul: Some(DecisionBoolean {
                order: DecisionBooleanOrder::FalseFirst,
                criteria: DecisionBooleanCriteria::Strict {
                    descriptions: DecisionPayload::Any,
                },
            }),
        }
    }
    pub fn require_type(&self, kind: DecisionKind) -> Result<(), String> {
        let allowed = match kind {
            DecisionKind::Choice => self.choice.is_some(),
            DecisionKind::Score => self.score.is_some(),
            DecisionKind::Noul => self.noul.is_some(),
        };
        if allowed {
            Ok(())
        } else {
            Err(format!(
                "decision type {kind:?} is disabled by the model's input policy; select one of its declared task types"
            ))
        }
    }
    pub(crate) fn validate_definition(&self, max_options: usize) -> Result<(), String> {
        if self.choice.is_none() && self.score.is_none() && self.noul.is_none() {
            return Err("input_policy must enable at least one of choice, score or noul".into());
        }
        if let Some(kind) = self.default_type {
            self.require_type(kind)
                .map_err(|e| format!("input_policy.default_type: {e}"))?;
        }
        if let DecisionInstructions::Optional { default, payload } = &self.instructions {
            payload.validate(default, "input_policy.instructions.default")?;
        }
        if let Some(choice) = &self.choice {
            match choice {
                DecisionChoices::Dynamic { minimum, .. } => {
                    if minimum.get() > max_options {
                        return Err("input_policy.choice.minimum exceeds decision.max_options; provide a usable option range".into());
                    }
                }
                DecisionChoices::Fixed { options } => {
                    if options.is_empty() || options.len() > max_options {
                        return Err("input_policy.choice.options must contain 1..=decision.max_options fixed options".into());
                    }
                    let mut keys = BTreeSet::new();
                    if !options.iter().all(|option| keys.insert(&option.key)) {
                        return Err("input_policy.choice.options contains duplicate keys; declare unique fixed labels".into());
                    }
                }
            }
        }
        if let Some(score) = &self.score
            && (score.minimum > score.maximum || score.minimum.get() > max_options)
        {
            return Err("input_policy.score minimum/maximum do not provide a usable range within decision.max_options; repair the level bounds".into());
        }
        if self.noul.is_some() && max_options < 2 {
            return Err("input_policy.noul requires capacity for false and true; raise decision.max_options to at least 2 or omit noul".into());
        }
        Ok(())
    }
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
    fn explicit_policy_has_a_bidirectional_core_version_gate() {
        let mut doc = document();
        assert!(crate::decision::decision_contract(&doc).unwrap().is_some());
        doc.schema_version = Some("v1.11".into());
        let error = crate::decision::decision_contract(&doc)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("input_policy") && error.contains("required"),
            "{error}"
        );
        doc.profiles
            .get_mut("decision_head")
            .unwrap()
            .decision
            .as_mut()
            .unwrap()
            .input_policy = Some(DecisionInputPolicy::schema_v1_10());
        assert!(crate::decision::decision_contract(&doc).unwrap().is_some());
        doc.schema_version = Some("v1.10".into());
        assert!(
            crate::decision::decision_contract(&doc)
                .unwrap_err()
                .to_string()
                .contains("minimum schema version v1.11")
        );
        let mut raw: serde_yaml::Value = serde_yaml::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/onnx_genai_workflows/decision/inference_metadata.yaml"
        )))
        .unwrap();
        raw["profiles"]["decision_head"]["decision"]["input_policy"] =
            serde_yaml::to_value(DecisionInputPolicy::schema_v1_10()).unwrap();
        let yaml = serde_yaml::to_string(&raw).unwrap();
        assert!(
            crate::parser::parse_metadata(&yaml, Some("yaml"))
                .unwrap_err()
                .to_string()
                .contains("minimum schema version v1.11")
        );
    }
    #[test]
    fn impossible_or_ambiguous_model_policies_fail_definition_admission() {
        for case in 0..5 {
            let mut policy = DecisionInputPolicy::schema_v1_10();
            match case {
                0 => {
                    policy.choice = None;
                    policy.score = None;
                    policy.noul = None;
                }
                1 => {
                    policy.default_type = Some(DecisionKind::Choice);
                    policy.choice = None;
                }
                2 => {
                    policy.instructions = DecisionInstructions::Optional {
                        default: Value::Null,
                        payload: DecisionPayload::Text,
                    }
                }
                3 => {
                    policy.choice = Some(DecisionChoices::Fixed {
                        options: vec![
                            DecisionOption {
                                key: "x".into(),
                                description: Value::Null
                            };
                            2
                        ],
                    })
                }
                _ => policy.score.as_mut().unwrap().minimum = NonZeroUsize::new(20).unwrap(),
            }
            assert!(policy.validate_definition(16).is_err(), "case {case}");
        }
        assert!(
            DecisionPayload::NonemptyPayload
                .validate(&Value::String("".into()), "instructions")
                .is_err()
        );
        assert!(
            DecisionPayload::NonemptyPayload
                .validate(&serde_json::json!({}), "instructions")
                .is_ok()
        );
        assert!(
            DecisionPayload::AnyNonNull
                .validate(&Value::Null, "state")
                .is_err()
        );
        assert!(
            DecisionPayload::PayloadOrNull
                .validate(&Value::Null, "state")
                .is_ok()
        );
    }
}
