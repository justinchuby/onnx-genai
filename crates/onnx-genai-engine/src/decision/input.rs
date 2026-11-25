//! One request-policy interpreter shared by all decision transports.
use anyhow::{Context, bail, ensure};
use onnx_genai_metadata::decision::DecisionContract;
use onnx_genai_metadata::*;
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug)]
enum Instructions {
    Provided(Value),
    ModelDefault,
}
#[derive(Debug)]
enum Options {
    Provided(Vec<DecisionOption>),
    ModelFixed,
}
#[derive(Debug)]
pub struct DecisionQuestion {
    kind: DecisionKind,
    instructions: Instructions,
    options: QuestionOptions,
}
#[derive(Debug)]
pub(super) struct QuestionOptions {
    contract: Arc<DecisionContract>,
    values: Options,
}
impl QuestionOptions {
    pub(super) fn policy(&self) -> &DecisionInputPolicy {
        self.contract.input_policy()
    }
    pub(super) fn values(&self) -> &[DecisionOption] {
        match &self.values {
            Options::Provided(options) => options,
            Options::ModelFixed => match &self.contract.input_policy().choice {
                Some(DecisionChoices::Fixed { options }) => options,
                _ => {
                    unreachable!("private fixed-option constructor requires a fixed choice policy")
                }
            },
        }
    }
}
impl DecisionQuestion {
    /// Resolve defaults, constraints and ordering before any tokenization or model work.
    pub fn from_value(
        contract: &Arc<DecisionContract>,
        value: Value,
        max_options: std::num::NonZeroUsize,
    ) -> anyhow::Result<Self> {
        let Value::Object(mut object) = value else {
            bail!("question must be an object with type, instructions and criteria");
        };
        let policy = contract.input_policy();
        let maximum = contract.spec().max_options.get().min(max_options.get());
        let kind = match object
            .remove("type")
            .and_then(|value| value.as_str().map(str::to_owned))
        {
            Some(name) => serde_json::from_value::<DecisionKind>(Value::String(name))
                .context("question.type must be choice, score or noul")?,
            None => policy.default_type.context(
                "question.type must be provided; this model declares no default question type",
            )?,
        };
        policy.require_type(kind).map_err(anyhow::Error::msg)?;
        let provided = object.remove("instructions");
        let instructions = match &policy.instructions {
            DecisionInstructions::Required { payload } => {
                let value = provided.context(
                    "question.instructions must be provided under this model's input policy",
                )?;
                payload
                    .validate(&value, "question.instructions")
                    .map_err(anyhow::Error::msg)?;
                Instructions::Provided(value)
            }
            DecisionInstructions::Optional { payload, .. } => match provided {
                Some(value) => {
                    payload
                        .validate(&value, "question.instructions")
                        .map_err(anyhow::Error::msg)?;
                    Instructions::Provided(value)
                }
                None => Instructions::ModelDefault,
            },
            DecisionInstructions::Forbidden => {
                ensure!(
                    provided.as_ref().is_none_or(Value::is_null),
                    "question.instructions is forbidden: this model's prompt supplies its fixed task; omit instructions"
                );
                Instructions::Provided(Value::Null)
            }
        };
        let mut criteria = object.remove("criteria").unwrap_or(Value::Null);
        let values = match kind {
            DecisionKind::Choice => match policy
                .choice
                .as_ref()
                .expect("require_type admitted choice")
            {
                DecisionChoices::Dynamic {
                    minimum,
                    order,
                    descriptions,
                    nonempty_keys,
                } => {
                    let Value::Object(criteria) = criteria else {
                        bail!(
                            "choice criteria must be a nonempty object mapping option keys to descriptions"
                        );
                    };
                    ensure!(
                        criteria.len() >= minimum.get(),
                        "choice criteria needs at least {minimum} options; supply the model's minimum choice count"
                    );
                    ensure!(
                        criteria.len() <= maximum,
                        "choice criteria exceeds effective model/serving option limit {maximum}; reduce criteria or increase the configured limit"
                    );
                    let mut options = Vec::new();
                    for (key, description) in criteria {
                        ensure!(
                            !*nonempty_keys || !key.is_empty(),
                            "choice criteria keys must be nonempty under this model's input policy"
                        );
                        descriptions
                            .validate(&description, &format!("choice criteria[{key:?}]"))
                            .map_err(anyhow::Error::msg)?;
                        options.push(DecisionOption { key, description });
                    }
                    if *order == DecisionOrder::Lexical {
                        options.sort_by(|a, b| a.key.cmp(&b.key));
                    }
                    Options::Provided(options)
                }
                DecisionChoices::Fixed { options } => {
                    ensure!(
                        options.len() <= maximum,
                        "fixed choice count exceeds effective model/serving option limit {maximum}; raise the serving limit or select another task"
                    );
                    let object=criteria.as_object().context("choice criteria must be an object matching the model's fixed option keys and descriptions")?;
                    ensure!(
                        object.len() == options.len()
                            && options
                                .iter()
                                .all(|option| object.get(&option.key) == Some(&option.description)),
                        "choice criteria must exactly match the model's fixed options and descriptions; inspect decision.input_policy.choice.options"
                    );
                    Options::ModelFixed
                }
            },
            DecisionKind::Score => {
                let levels = policy.score.as_ref().expect("require_type admitted score");
                let Value::Array(criteria) = criteria else {
                    bail!("score criteria must be an array of level descriptions");
                };
                ensure!(
                    (levels.minimum.get()..=levels.maximum.get()).contains(&criteria.len()),
                    "score criteria must contain {}..={} levels under this model's input policy",
                    levels.minimum,
                    levels.maximum
                );
                ensure!(
                    criteria.len() <= maximum,
                    "score criteria exceeds effective model/serving option limit {maximum}; reduce levels or raise the serving limit"
                );
                let mut options = Vec::new();
                for (index, description) in criteria.into_iter().enumerate() {
                    levels
                        .descriptions
                        .validate(&description, &format!("score criteria[{index}]"))
                        .map_err(anyhow::Error::msg)?;
                    options.push(DecisionOption {
                        key: index.to_string(),
                        description,
                    });
                }
                Options::Provided(options)
            }
            DecisionKind::Noul => {
                let rule = policy.noul.as_ref().expect("require_type admitted noul");
                ensure!(
                    maximum >= 2,
                    "noul needs false/true options but the effective model/serving option limit is {maximum}; raise the serving limit or select another task"
                );
                ensure!(
                    criteria.is_null() || criteria.is_object(),
                    "noul criteria must be an object or null"
                );
                let criteria_first = criteria.as_object().is_some_and(|map| {
                    map.len() == 2 && map.keys().next().is_some_and(|key| key == "true")
                });
                let aliases = match &rule.criteria {
                    DecisionBooleanCriteria::Forbidden => {
                        ensure!(
                            criteria.is_null(),
                            "noul criteria is forbidden by this model's fixed-task policy; omit criteria"
                        );
                        false
                    }
                    DecisionBooleanCriteria::Strict { descriptions } => {
                        if let Some(map) = criteria.as_object() {
                            for (key, value) in map {
                                ensure!(
                                    matches!(key.as_str(), "false" | "true"),
                                    "noul criteria may only describe false and true under this model's input policy"
                                );
                                descriptions
                                    .validate(value, &format!("noul criteria[{key:?}]"))
                                    .map_err(anyhow::Error::msg)?;
                            }
                        }
                        false
                    }
                    DecisionBooleanCriteria::Permissive { no_yes_aliases } => *no_yes_aliases,
                };
                let mut options = Vec::new();
                for (key, alias) in [("false", "no"), ("true", "yes")] {
                    let description = criteria
                        .as_object_mut()
                        .and_then(|map| {
                            map.remove(key)
                                .or_else(|| if aliases { map.remove(alias) } else { None })
                        })
                        .unwrap_or(Value::Null);
                    options.push(DecisionOption {
                        key: key.into(),
                        description,
                    });
                }
                if rule.order == DecisionBooleanOrder::TrueFirst
                    || (rule.order == DecisionBooleanOrder::CriteriaTrueFirst && criteria_first)
                {
                    options.reverse();
                }
                Options::Provided(options)
            }
        };
        let question = Self {
            kind,
            instructions,
            options: QuestionOptions {
                contract: Arc::clone(contract),
                values,
            },
        };
        let count = question.option_values().len();
        ensure!(
            contract
                .option_count()
                .is_none_or(|required| required == count),
            "decision has {count} options, but profile '{}' declares a fixed probability head with {:?} options; use the declared count",
            contract.profile(),
            contract.option_count()
        );
        ensure!(
            question.instructions().get("type").and_then(Value::as_str) != Some("input_audio"),
            "spoken instructions require a multimodal decision contract; supply text instructions"
        );
        Ok(question)
    }
    pub(super) fn kind(&self) -> DecisionKind {
        self.kind
    }
    pub(super) fn contract(&self) -> &Arc<DecisionContract> {
        &self.options.contract
    }
    pub(super) fn into_options(self) -> QuestionOptions {
        self.options
    }
    pub fn option_count(&self) -> usize {
        self.option_values().len()
    }
    pub(super) fn instructions(&self) -> &Value {
        match &self.instructions {
            Instructions::Provided(value) => value,
            Instructions::ModelDefault => {
                match &self.options.contract.input_policy().instructions {
                    DecisionInstructions::Optional { default, .. } => default,
                    _ => unreachable!(
                        "private default-instruction constructor requires an optional policy"
                    ),
                }
            }
        }
    }
    pub(super) fn option_values(&self) -> &[DecisionOption] {
        self.options.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::{DecisionAnswer, DecisionLimits, PreparedDecision};
    use onnx_genai_ort::Tokenizer;
    use serde_json::json;
    fn policy() -> Value {
        json!({
            "state":"payload", "instructions":{"kind":"optional","default":"the","payload":"text"},
            "default_type":"choice", "max_questions":2,
            "choice":{"kind":"dynamic","minimum":2,"order":"request","descriptions":"payload_or_null","nonempty_keys":true},
            "score":{"minimum":2,"maximum":10,"descriptions":"payload","legend":"canonical_json_text"},
            "noul":{"order":"criteria_true_first","criteria":{"kind":"permissive","no_yes_aliases":true}}
        })
    }
    fn make_package(policy: Value, template: Option<&str>) -> tempfile::TempDir {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/decision");
        let package = tempfile::tempdir().unwrap();
        for name in ["tokenizer.json", "model.onnx.textproto"] {
            std::fs::copy(root.join(name), package.path().join(name)).unwrap();
        }
        let mut metadata: serde_yaml::Value = serde_yaml::from_str(
            &std::fs::read_to_string(root.join("inference_metadata.yaml")).unwrap(),
        )
        .unwrap();
        metadata["schema_version"] = serde_yaml::Value::String("v1.11".into());
        metadata["profiles"]["decision_head"]["decision"]["input_policy"] =
            serde_yaml::to_value(policy).unwrap();
        if let Some(template) = template {
            metadata["profiles"]["decision_head"]["decision"]["prompt_template"] =
                serde_yaml::Value::String(template.into());
        }
        std::fs::write(
            package.path().join("inference_metadata.yaml"),
            serde_yaml::to_string(&metadata).unwrap(),
        )
        .unwrap();
        package
    }
    fn run(package: &std::path::Path, body: Value) -> crate::decision::DecisionResult {
        let mut engine = crate::Engine::from_dir(package, crate::EngineConfig::default()).unwrap();
        let contract = Arc::new(
            onnx_genai_metadata::decision::decision_contract(engine.metadata())
                .unwrap()
                .unwrap(),
        );
        let tokenizer = Tokenizer::from_file(package.join("tokenizer.json")).unwrap();
        let q = DecisionQuestion::from_value(
            &contract,
            body,
            std::num::NonZeroUsize::new(128).unwrap(),
        )
        .unwrap();
        let prepared = PreparedDecision::prepare(
            &tokenizer,
            crate::decision::DecisionInput::text(&json!("hello")),
            q,
            DecisionLimits::default(),
        )
        .unwrap();
        engine.decide(prepared).unwrap()
    }
    #[test]
    fn authored_defaults_request_order_and_permissive_fields_reach_actual_inference() {
        let package = make_package(policy(), None);
        for kind in [None, Some(json!(null)), Some(json!(9))] {
            let mut request =
                json!({"criteria":{"b":"hello","a":"hello"},"ignored_extension":true});
            if let Some(kind) = kind {
                request["type"] = kind;
            }
            let result = run(package.path(), request);
            assert_eq!(
                result
                    .options()
                    .iter()
                    .map(|o| o.key.as_str())
                    .collect::<Vec<_>>(),
                ["b", "a"]
            );
            match result.answer() {
                DecisionAnswer::Choice { key, .. } => assert_eq!(key, "b"),
                _ => panic!("expected choice"),
            }
            assert_eq!(result.prompt_tokens(), 6);
        }
    }
    #[test]
    fn boolean_aliases_and_true_first_criteria_preserve_true_probability() {
        let package = make_package(policy(), None);
        for criteria in [
            json!({"yes":"hello","no":"world","unread":123}),
            json!({"true":"hello","false":"world"}),
        ] {
            let result = run(package.path(), json!({"type":"noul","criteria":criteria}));
            match result.answer() {
                DecisionAnswer::Noul { probability } => {
                    assert!((*probability - 1.0 / (1.0 + (-1.0_f64).exp())).abs() < 1e-6)
                }
                _ => panic!("expected noul"),
            }
        }
        let metadata =
            onnx_genai_metadata::load_metadata(&package.path().join("inference_metadata.yaml"))
                .unwrap();
        let contract = Arc::new(
            onnx_genai_metadata::decision::decision_contract(&metadata)
                .unwrap()
                .unwrap(),
        );
        let q = DecisionQuestion::from_value(
            &contract,
            json!({"type":"noul","criteria":{"true":null,"yes":"hello","false":"world"}}),
            contract.spec().max_options,
        )
        .unwrap();
        assert!(
            q.option_values()
                .iter()
                .find(|o| o.key == "true")
                .unwrap()
                .description
                .is_null()
        );
    }
    #[test]
    fn fixed_tasks_forbid_overrides_and_can_serve_true_first_boolean_heads() {
        let mut p = policy();
        p["instructions"] = json!({"kind":"forbidden"});
        p["state"] = json!("text");
        p["choice"] = json!({"kind":"fixed","options":[{"key":"0","description":null},{"key":"1","description":null}]});
        p["score"] = Value::Null;
        p["noul"] = json!({"order":"true_first","criteria":{"kind":"forbidden"}});
        let package = make_package(p, Some("{{ state }} the hello world"));
        let result = run(package.path(), json!({"criteria":{"1":null,"0":null}}));
        match result.answer() {
            DecisionAnswer::Choice { key, .. } => assert_eq!(key, "0"),
            _ => panic!("expected choice"),
        }
        let result = run(package.path(), json!({"type":"noul"}));
        match result.answer() {
            DecisionAnswer::Noul { probability } => {
                assert!((*probability - 1.0 / (1.0 + (-1.0_f64).exp())).abs() < 1e-6)
            }
            _ => panic!("expected noul"),
        }
        let metadata =
            onnx_genai_metadata::load_metadata(&package.path().join("inference_metadata.yaml"))
                .unwrap();
        let contract = Arc::new(
            onnx_genai_metadata::decision::decision_contract(&metadata)
                .unwrap()
                .unwrap(),
        );
        for (q, needle) in [
            (
                json!({"instructions":"override","criteria":{"0":null,"1":null}}),
                "forbidden",
            ),
            (
                json!({"criteria":{"0":"changed","1":null}}),
                "fixed options",
            ),
            (json!({"type":"score","criteria":["a","b"]}), "disabled"),
            (json!({"type":"noul","criteria":{}}), "forbidden"),
        ] {
            let error = DecisionQuestion::from_value(&contract, q, contract.spec().max_options)
                .unwrap_err();
            assert!(error.to_string().contains(needle), "{error}");
        }
    }
    #[test]
    fn canonical_legends_sort_nested_keys_and_leave_string_descriptions_unchanged() {
        let package = make_package(
            policy(),
            Some(
                "{{ state }} {{ instructions }} {{ options[0].description.label }} {{ options[1].description.label }}",
            ),
        );
        let result = run(
            package.path(),
            json!({"type":"score","criteria":[{"z":2,"label":"hello","a":{"z":false,"a":1}},{"label":"world","a":[{"z":2,"a":1}]}]}),
        );
        let legend = result.legend().unwrap();
        assert_eq!(
            legend["0"],
            json!(r#"{"a":{"a":1,"z":false},"label":"hello","z":2}"#)
        );
        assert_eq!(
            legend["1"],
            json!(r#"{"a":[{"a":1,"z":2}],"label":"world"}"#)
        );
        let package = make_package(policy(), None);
        let result = run(
            package.path(),
            json!({"type":"score","criteria":["hello","world"]}),
        );
        assert_eq!(result.legend().unwrap()["0"], json!("hello"));
        let metadata =
            onnx_genai_metadata::load_metadata(&package.path().join("inference_metadata.yaml"))
                .unwrap();
        let contract = Arc::new(
            onnx_genai_metadata::decision::decision_contract(&metadata)
                .unwrap()
                .unwrap(),
        );
        assert!(
            DecisionQuestion::from_value(
                &contract,
                json!({"type":"score","criteria":[null,"world"]}),
                contract.spec().max_options
            )
            .unwrap_err()
            .to_string()
            .contains("criteria[0]")
        );
    }
}
