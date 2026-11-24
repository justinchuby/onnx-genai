//! Non-generative decisions over a model-authored workflow and prompt contract.
use crate::{Engine, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest};
use anyhow::{Context, ensure};
use onnx_genai_metadata::{
    DecisionConfidence, DecisionScore, DecisionVariants,
    decision::{DecisionContract, DecisionFeature, decision_contract},
};
use onnx_genai_ort::{ChatTemplate, TemplateLimits, Tokenizer, Value};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, num::NonZeroUsize};

/// Independent token, rendered-byte and template-execution admission limits.
#[derive(Debug, Clone, Copy)]
pub struct DecisionLimits {
    pub max_input_tokens: NonZeroUsize,
    pub template: TemplateLimits,
}
impl Default for DecisionLimits {
    fn default() -> Self {
        Self {
            max_input_tokens: NonZeroUsize::new(8192).unwrap(),
            template: TemplateLimits {
                max_output_bytes: NonZeroUsize::new(4 * 1024 * 1024).unwrap(),
                max_operations: std::num::NonZeroU64::new(1_000_000).unwrap(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionKind {
    Choice,
    Score,
    Noul,
}
#[derive(Debug, Clone, Serialize)]
pub struct DecisionOption {
    pub key: String,
    pub description: serde_json::Value,
}
#[derive(Debug)]
pub struct DecisionQuestion {
    pub kind: DecisionKind,
    pub instructions: serde_json::Value,
    pub options: Vec<DecisionOption>,
}
#[derive(Debug)]
struct Variant {
    ids: Vec<u32>,
    mask: Vec<i64>,
    segments: Vec<i64>,
}
/// Tokenizer features and output cardinality admitted before any inference.
#[derive(Debug)]
pub struct PreparedDecision {
    contract: DecisionContract,
    question: DecisionQuestion,
    variants: Vec<Variant>,
}
impl PreparedDecision {
    pub fn prepare(
        tokenizer: &Tokenizer,
        contract: &DecisionContract,
        state: &serde_json::Value,
        question: DecisionQuestion,
        limits: DecisionLimits,
    ) -> anyhow::Result<Self> {
        let max_input_tokens = limits.max_input_tokens;
        let count = question.options.len();
        ensure!(
            count > 0 && count <= contract.spec().max_options.get(),
            "decision has {count} options; profile '{}' accepts 1..={} options",
            contract.profile(),
            contract.spec().max_options
        );
        ensure!(
            contract
                .option_count()
                .is_none_or(|required| required == count),
            "decision has {count} options, but profile '{}' declares a fixed probability head with {:?} options; use the declared option count",
            contract.profile(),
            contract.option_count()
        );
        let mut keys = std::collections::BTreeSet::new();
        ensure!(
            question
                .options
                .iter()
                .all(|option| !option.key.is_empty() && keys.insert(&option.key)),
            "decision option keys must be nonempty and unique; repair the criteria"
        );
        match question.kind {
            DecisionKind::Score => ensure!(
                (2..=10).contains(&count)
                    && question
                        .options
                        .iter()
                        .enumerate()
                        .all(|(i, o)| o.key == i.to_string()),
                "score decisions need 2..=10 levels keyed by their zero-based ordinal"
            ),
            DecisionKind::Noul => ensure!(
                question
                    .options
                    .iter()
                    .map(|o| o.key.as_str())
                    .eq(["false", "true"]),
                "noul decisions must declare false then true options"
            ),
            DecisionKind::Choice => {}
        }
        let n = match contract.spec().variants {
            DecisionVariants::Direct => 1,
            DecisionVariants::ReverseAverage => 2,
        };
        let template = ChatTemplate::from_source(&contract.spec().prompt_template);
        let mut variants = Vec::new();
        for variant in 0..n {
            let options = if variant == 0 {
                question.options.clone()
            } else {
                question.options.iter().rev().cloned().collect()
            };
            let kwargs = BTreeMap::from([
                ("state".into(), state.clone()),
                ("instructions".into(), question.instructions.clone()),
                ("type".into(), serde_json::to_value(question.kind)?),
                ("options".into(), serde_json::to_value(options)?),
                ("variant".into(), serde_json::json!(variant)),
            ]);
            let prompt=template.render_with_limited_kwargs(&[],None,false,None,&kwargs,limits.template).with_context(||format!("render decision profile '{}' prompt variant {variant}; repair decision.prompt_template",contract.profile()))?;
            let encoded = tokenizer.inner().encode(prompt, true).map_err(|error| {
                anyhow::anyhow!(
                    "tokenize decision variant {variant}: {error}; repair the package tokenizer"
                )
            })?;
            let count = encoded.get_ids().len();
            ensure!(
                count > 0 && count <= max_input_tokens.get(),
                "decision variant {variant} has {count} tokens; the input budget is {max_input_tokens}; shorten state/instructions/criteria or configure tokenizer truncation"
            );
            ensure!(
                contract.sequence_length().is_none_or(|n| n == count),
                "decision variant {variant} has {count} tokens, but profile '{}' requires {:?}; configure tokenizer padding/truncation",
                contract.profile(),
                contract.sequence_length()
            );
            ensure!(
                encoded.get_attention_mask().len() == count
                    && encoded.get_type_ids().len() == count,
                "decision tokenizer feature lengths disagree; repair tokenizer.json"
            );
            variants.push(Variant {
                ids: encoded.get_ids().to_vec(),
                mask: encoded
                    .get_attention_mask()
                    .iter()
                    .map(|x| i64::from(*x))
                    .collect(),
                segments: encoded
                    .get_type_ids()
                    .iter()
                    .map(|x| i64::from(*x))
                    .collect(),
            });
        }
        Ok(Self {
            contract: contract.clone(),
            question,
            variants,
        })
    }
}
#[derive(Debug)]
pub enum DecisionAnswer {
    Choice { key: String, confidence: f64 },
    Score { score: f64, confidence: f64 },
    Noul { probability: f64 },
}
impl DecisionAnswer {
    pub fn kind(&self) -> DecisionKind {
        match self {
            Self::Choice { .. } => DecisionKind::Choice,
            Self::Score { .. } => DecisionKind::Score,
            Self::Noul { .. } => DecisionKind::Noul,
        }
    }
}
#[derive(Debug)]
pub struct DecisionResult {
    pub answer: DecisionAnswer,
    pub options: Vec<DecisionOption>,
    pub probabilities: Vec<f64>,
    pub prompt_tokens: usize,
    _credit: onnx_runtime_memory_governor::MemoryLease,
}
impl Engine {
    pub fn decide(&mut self, request: PreparedDecision) -> anyhow::Result<DecisionResult> {
        let contract=decision_contract(self.metadata())?.context("model has no decision v1 workflow profile; select a package declaring a non-generative probability head")?;
        ensure!(
            contract == request.contract,
            "prepared decision does not match this model's admitted profile; prepare it against the executing model's decision contract"
        );
        let PreparedDecision {
            question, variants, ..
        } = request;
        let count = question.options.len();
        let holder = crate::engine::memory_plan::Holder::Activations;
        let credit = onnx_runtime_memory_governor::MemoryGovernor::reserve(
            self.governor.memory(), onnx_runtime_memory_api::Tier::Host,
            u64::try_from(count)?.checked_mul(24).context("decision staging size overflow")?,
            holder.role(),holder.id(),
        ).context("reserve decision probability conversion/aggregation ownership; reduce options or increase host budget")?;
        let mut probabilities = Vec::new();
        probabilities
            .try_reserve_exact(count)
            .context("allocate decision probability buffer; reduce options or free host memory")?;
        probabilities.resize(count, 0.0);
        let mut prompt_tokens = 0usize;
        let n = variants.len();
        for (index, variant) in variants.into_iter().enumerate() {
            if let Some(limit) = self.effective_max_context(&GenerateOptions::default()) {
                ensure!(
                    variant.ids.len() <= limit,
                    "decision input exceeds model context limit {limit}; shorten the request"
                );
            }
            prompt_tokens = prompt_tokens
                .checked_add(variant.mask.iter().filter(|x| **x != 0).count())
                .context("decision token usage overflow; reduce input length")?;
            let shape = [1, i64::try_from(variant.ids.len())?];
            let mut execution = PipelineGenerateRequest::new(GenerateRequest {
                prompt: GeneratePrompt::TokenIds(variant.ids),
                options: GenerateOptions {
                    max_new_tokens: 0,
                    ..Default::default()
                },
            });
            for (name, feature) in contract.bindings() {
                let value = match feature {
                    DecisionFeature::AttentionMask => {
                        Value::from_vec_i64(variant.mask.clone(), &shape)?
                    }
                    DecisionFeature::TokenTypeIds => {
                        Value::from_vec_i64(variant.segments.clone(), &shape)?
                    }
                    DecisionFeature::OptionCount { vector } => Value::from_vec_i64(
                        vec![i64::try_from(count)?],
                        if *vector { &[1] } else { &[] },
                    )?,
                };
                execution = execution.with_input(name, value);
            }
            let outputs = self.run_pipeline_outputs(execution).with_context(|| {
                format!(
                    "execute decision profile '{}' variant {index}",
                    contract.profile()
                )
            })?;
            let value = outputs.aggregate(contract.output()).with_context(|| {
                format!(
                    "decision workflow did not emit probability output '{}'",
                    contract.output()
                )
            })?;
            ensure!(
                value.numel() == count,
                "decision output '{}' has {} values for {count} options; repair the workflow's option selection",
                contract.output(),
                value.numel()
            );
            let shape = value.shape();
            ensure!(
                shape.len() == 1 || (shape.len() == 2 && shape[0] == 1),
                "decision output '{}' must be [options] or [1,options]; found {shape:?}",
                contract.output()
            );
            let data=value.to_vec_f32_lossy().with_context(||format!("decision output '{}' must be host-readable floating probabilities; declare an explicit host boundary",contract.output()))?;
            let sum = data.iter().map(|x| f64::from(*x)).sum::<f64>();
            let tolerance = match value.dtype() {
                onnx_genai_ort::DataType::Float16 => f64::from(half::f16::EPSILON.to_f32()),
                onnx_genai_ort::DataType::BFloat16 => f64::from(half::bf16::EPSILON.to_f32()),
                _ => f64::from(f32::EPSILON),
            }
            .max(1e-5);
            ensure!(
                data.iter()
                    .all(|x| x.is_finite() && (0.0..=1.0).contains(x))
                    && (sum - 1.0).abs() <= tolerance,
                "decision output '{}' is not a calibrated probability distribution (sum {sum}); apply calibration and softmax in the model workflow",
                contract.output()
            );
            for (j, p) in data.iter().enumerate() {
                let target = if index == 0 { j } else { count - 1 - j };
                probabilities[target] += f64::from(*p) / sum / (n as f64);
            }
        }
        let answer = match question.kind {
            DecisionKind::Choice => {
                let best = probabilities.iter().enumerate().fold(0, |best, (i, p)| {
                    if *p > probabilities[best] { i } else { best }
                });
                DecisionAnswer::Choice {
                    key: question.options[best].key.clone(),
                    confidence: confidence(contract.spec().choice_confidence, &probabilities),
                }
            }
            DecisionKind::Score => {
                let score = match contract.spec().score {
                    DecisionScore::ExpectedLevel => probabilities
                        .iter()
                        .enumerate()
                        .map(|(i, p)| (i as f64) * p)
                        .sum(),
                    DecisionScore::MedianLevel => {
                        let mut mass = 0.0;
                        probabilities
                            .iter()
                            .position(|p| {
                                mass += p;
                                mass >= 0.5
                            })
                            .unwrap_or(count - 1) as f64
                    }
                };
                DecisionAnswer::Score {
                    score,
                    confidence: confidence(contract.spec().score_confidence, &probabilities),
                }
            }
            DecisionKind::Noul => DecisionAnswer::Noul {
                probability: probabilities[1],
            },
        };
        Ok(DecisionResult {
            answer,
            options: question.options,
            probabilities,
            prompt_tokens,
            _credit: credit,
        })
    }
}
fn confidence(policy: DecisionConfidence, p: &[f64]) -> f64 {
    if p.len() < 2 {
        return 1.0;
    }
    let n = p.len() as f64;
    let (mode, maximum) =
        p.iter().enumerate().fold(
            (0, 0.0),
            |(index, best), (i, p)| if *p > best { (i, *p) } else { (index, best) },
        );
    (match policy {
        DecisionConfidence::UniformMargin => (maximum - 1.0 / n) / (1.0 - 1.0 / n),
        DecisionConfidence::MaximumProbability => maximum,
        DecisionConfidence::NormalizedEntropy => {
            1.0 + p
                .iter()
                .filter(|p| **p > 0.0)
                .map(|p| p * p.ln())
                .sum::<f64>()
                / n.ln()
        }
        DecisionConfidence::ModeDistance => {
            let distance = p
                .iter()
                .enumerate()
                .map(|(i, p)| p * ((i as f64) - (mode as f64)).abs())
                .sum::<f64>();
            let uniform = (0..p.len())
                .map(|i| ((i as f64) - (n - 1.0) / 2.0).abs() / n)
                .sum::<f64>();
            1.0 - distance / uniform
        }
    })
    .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn root() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/decision")
    }
    fn question(kind: DecisionKind) -> DecisionQuestion {
        let keys = match kind {
            DecisionKind::Choice => ["a", "b"],
            DecisionKind::Score => ["0", "1"],
            DecisionKind::Noul => ["false", "true"],
        };
        DecisionQuestion {
            kind,
            instructions: serde_json::json!("the"),
            options: keys
                .into_iter()
                .zip(["hello", "world"])
                .map(|(key, text)| DecisionOption {
                    key: key.into(),
                    description: serde_json::json!(text),
                })
                .collect(),
        }
    }
    #[test]
    fn decision_runs_the_declared_head_and_changes_with_state_for_every_question_kind() {
        let root = root();
        let tokenizer = Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
        let mut engine = Engine::from_dir(&root, crate::EngineConfig::default()).unwrap();
        let contract = decision_contract(engine.metadata()).unwrap().unwrap();
        for kind in [
            DecisionKind::Choice,
            DecisionKind::Score,
            DecisionKind::Noul,
        ] {
            for (state, best) in [("hello", 0), ("world", 1)] {
                let prepared = PreparedDecision::prepare(
                    &tokenizer,
                    &contract,
                    &serde_json::json!(state),
                    question(kind),
                    DecisionLimits {
                        max_input_tokens: NonZeroUsize::new(64).unwrap(),
                        ..Default::default()
                    },
                )
                .unwrap();
                let result = engine.decide(prepared).unwrap();
                assert_eq!(result.prompt_tokens, 6);
                assert!(result.probabilities[best] > result.probabilities[1 - best]);
                assert!((result.probabilities[best] - 1.0 / (1.0 + (-1.0_f64).exp())).abs() < 1e-6);
                assert_eq!(result.answer.kind(), kind);
                if let DecisionAnswer::Score { score, .. } = result.answer {
                    assert!((score - result.probabilities[1]).abs() < 1e-8);
                }
            }
        }
        assert!(
            engine
                .component_invocations()
                .get("classifier")
                .is_some_and(|n| *n > 0)
        );
        assert!(!engine.workflow_declares_generation_loop());
    }
    fn modified_package(change: impl FnOnce(String) -> String) -> tempfile::TempDir {
        let package = tempfile::tempdir().unwrap();
        for file in ["tokenizer.json", "model.onnx.textproto"] {
            std::fs::copy(root().join(file), package.path().join(file)).unwrap();
        }
        let metadata = std::fs::read_to_string(root().join("inference_metadata.yaml")).unwrap();
        std::fs::write(
            package.path().join("inference_metadata.yaml"),
            change(metadata),
        )
        .unwrap();
        package
    }
    #[test]
    fn reverse_variants_align_options_and_median_score_uses_the_declared_policy() {
        let package = modified_package(|s| {
            s.replace("variants: direct", "variants: reverse_average")
                .replace("score: expected_level", "score: median_level")
                .replace(
                    "score_confidence: mode_distance",
                    "score_confidence: maximum_probability",
                )
        });
        let mut engine = Engine::from_dir(package.path(), crate::EngineConfig::default()).unwrap();
        let tokenizer = Tokenizer::from_file(package.path().join("tokenizer.json")).unwrap();
        let contract = decision_contract(engine.metadata()).unwrap().unwrap();
        for (state, level) in [("hello", 0.0), ("world", 1.0)] {
            let prepared = PreparedDecision::prepare(
                &tokenizer,
                &contract,
                &serde_json::json!(state),
                question(DecisionKind::Score),
                DecisionLimits {
                    max_input_tokens: NonZeroUsize::new(64).unwrap(),
                    ..Default::default()
                },
            )
            .unwrap();
            let result = engine.decide(prepared).unwrap();
            assert_eq!(result.prompt_tokens, 12);
            match result.answer {
                DecisionAnswer::Score { score, confidence } => {
                    assert_eq!(score, level);
                    assert!((confidence - 1.0 / (1.0 + (-1.0_f64).exp())).abs() < 1e-6);
                }
                _ => panic!("score request returned the wrong answer kind"),
            }
        }
    }
    #[test]
    fn dynamic_option_heads_execute_scalar_and_vector_count_bindings() {
        for scalar in [false, true] {
            let package = modified_package(|source| {
                let mut doc: serde_yaml::Value = serde_yaml::from_str(&source).unwrap();
                doc["profiles"]["decision_head"]["decision"]["max_options"] =
                    serde_yaml::to_value(8).unwrap();
                doc["profiles"]["decision_head"]["decision"]["prompt_template"]=serde_yaml::Value::String("{{ state }} {{ instructions }} {% for option in options %}{{ option.description }} {% endfor %}".into());
                doc["pipeline"]["workflow"]["outputs"]["decision.probabilities"]["contract"]["shape"] =
                    serde_yaml::from_str("[1, options]").unwrap();
                let shape = if scalar { "[]" } else { "[1]" };
                doc["pipeline"]["workflow"]["inputs"]["option.count"]=serde_yaml::from_str(&format!("contract: {{dtype: int64, shape: {shape}}}\nrole: {{kind: opaque}}\nsource: {{kind: application, name: option_count}}\nrequired: true")).unwrap();
                doc["pipeline"]["workflow"]["steps"][0]["inputs"]["option_count"] =
                    serde_yaml::Value::String("option.count".into());
                serde_yaml::to_string(&doc).unwrap()
            });
            let path = package.path().join("model.onnx.textproto");
            let source = std::fs::read_to_string(&path).unwrap();
            let old = r#"node { input: "ids.float" input: "option_indices" output: "options" op_type: "Gather" attribute { name: "axis" i: 1 type: INT } }"#;
            let replacement = r#"node { input: "option_count" input: "option_start" output: "option_end" op_type: "Add" }
node { input: "ids.float" input: "option_start" input: "option_end" input: "option_axis" output: "options" op_type: "Slice" }"#;
            assert!(source.contains(old));
            let mut source = source
                .replace(old, replacement)
                .replace("dim { dim_value: 2 }", "dim { dim_param: \"options\" }");
            let shape = if scalar { "" } else { "dim { dim_value: 1 }" };
            let extra = format!(
                r#"input {{ name: "option_count" type {{ tensor_type {{ elem_type: 7 shape {{ {shape} }} }} }} }}
initializer {{ name: "option_start" data_type: 7 dims: 1 int64_data: 3 }}
initializer {{ name: "option_axis" data_type: 7 dims: 1 int64_data: 1 }}
"#
            );
            source.insert_str(source.rfind('}').unwrap(), &extra);
            std::fs::write(path, source).unwrap();
            let mut engine =
                Engine::from_dir(package.path(), crate::EngineConfig::default()).unwrap();
            let tokenizer = Tokenizer::from_file(package.path().join("tokenizer.json")).unwrap();
            let contract = decision_contract(engine.metadata()).unwrap().unwrap();
            for count in [3, 4] {
                let options = ["hello", "world", "quick", "brown"]
                    .into_iter()
                    .take(count)
                    .enumerate()
                    .map(|(i, text)| DecisionOption {
                        key: format!("key-{i}"),
                        description: serde_json::json!(text),
                    })
                    .collect();
                let question = DecisionQuestion {
                    kind: DecisionKind::Choice,
                    instructions: serde_json::json!("the"),
                    options,
                };
                let prepared = PreparedDecision::prepare(
                    &tokenizer,
                    &contract,
                    &serde_json::json!("quick"),
                    question,
                    DecisionLimits::default(),
                )
                .unwrap();
                let result = engine.decide(prepared).unwrap();
                assert_eq!(result.probabilities.len(), count);
                assert_eq!(result.prompt_tokens, count + 4);
                match result.answer {
                    DecisionAnswer::Choice { key, .. } => assert_eq!(key, "key-2"),
                    _ => panic!("choice request returned the wrong answer kind"),
                }
            }
        }
    }

    #[test]
    fn low_precision_heads_allow_rounding_but_refuse_mass_outside_dtype_precision() {
        for (dtype, element, epsilon) in [
            ("float16", 10, f64::from(half::f16::EPSILON.to_f32())),
            ("bfloat16", 16, f64::from(half::bf16::EPSILON.to_f32())),
        ] {
            for invalid in [false, true] {
                let package = modified_package(|source| {
                    let mut doc: serde_yaml::Value = serde_yaml::from_str(&source).unwrap();
                    doc["pipeline"]["workflow"]["outputs"]["decision.probabilities"]["contract"]
                        ["dtype"] = serde_yaml::Value::String(dtype.into());
                    serde_yaml::to_string(&doc).unwrap()
                });
                let path = package.path().join("model.onnx.textproto");
                let source = std::fs::read_to_string(&path).unwrap();
                let old = r#"node { input: "logits" output: "probabilities" op_type: "Softmax" attribute { name: "axis" i: 1 type: INT } }"#;
                let replacement = format!(
                    r#"node {{ input: "logits" output: "raw.probabilities" op_type: "Softmax" attribute {{ name: "axis" i: 1 type: INT }} }}
node {{ input: "raw.probabilities" input: "mass_scale" output: "scaled.probabilities" op_type: "Mul" }}
node {{ input: "scaled.probabilities" output: "probabilities" op_type: "Cast" attribute {{ name: "to" i: {element} type: INT }} }}"#
                );
                assert!(source.contains(old));
                let mut source=source.replace(old,&replacement).replace("output { name: \"probabilities\" type { tensor_type { elem_type: 1",&format!("output {{ name: \"probabilities\" type {{ tensor_type {{ elem_type: {element}"));
                let factor = if invalid { 1.0 + 2.0 * epsilon } else { 1.0 };
                let extra = format!(
                    "initializer {{ name: \"mass_scale\" data_type: 1 dims: 1 float_data: {factor} }}\n"
                );
                source.insert_str(source.rfind('}').unwrap(), &extra);
                std::fs::write(path, source).unwrap();
                let mut engine =
                    Engine::from_dir(package.path(), crate::EngineConfig::default()).unwrap();
                let tokenizer =
                    Tokenizer::from_file(package.path().join("tokenizer.json")).unwrap();
                let contract = decision_contract(engine.metadata()).unwrap().unwrap();
                let prepared = PreparedDecision::prepare(
                    &tokenizer,
                    &contract,
                    &serde_json::json!("hello"),
                    question(DecisionKind::Choice),
                    DecisionLimits::default(),
                )
                .unwrap();
                if invalid {
                    let error = engine.decide(prepared).unwrap_err();
                    assert!(
                        format!("{error:#}").contains("calibrated probability distribution"),
                        "{dtype}: {error:#}"
                    );
                } else {
                    let result = engine.decide(prepared).unwrap();
                    assert!((result.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                    assert!(result.probabilities[0] > result.probabilities[1]);
                }
            }
        }
    }

    #[test]
    fn decision_rejects_uncalibrated_actual_model_outputs() {
        let package = modified_package(|s| s);
        let path = package.path().join("model.onnx.textproto");
        let graph = std::fs::read_to_string(&path).unwrap().replace(
            "op_type: \"Softmax\" attribute { name: \"axis\" i: 1 type: INT }",
            "op_type: \"Identity\"",
        );
        assert!(!graph.contains("Softmax"));
        std::fs::write(path, graph).unwrap();
        let mut engine = Engine::from_dir(package.path(), crate::EngineConfig::default()).unwrap();
        let tokenizer = Tokenizer::from_file(package.path().join("tokenizer.json")).unwrap();
        let contract = decision_contract(engine.metadata()).unwrap().unwrap();
        let prepared = PreparedDecision::prepare(
            &tokenizer,
            &contract,
            &serde_json::json!("hello"),
            question(DecisionKind::Choice),
            DecisionLimits {
                max_input_tokens: NonZeroUsize::new(64).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
        let error = engine.decide(prepared).unwrap_err();
        assert!(
            format!("{error:#}").contains("calibrated probability distribution"),
            "{error:#}"
        );
    }

    #[test]
    fn decision_confidence_policies_and_score_contracts_are_explicit() {
        for policy in [
            DecisionConfidence::UniformMargin,
            DecisionConfidence::ModeDistance,
            DecisionConfidence::NormalizedEntropy,
        ] {
            assert!((confidence(policy, &[0.5, 0.5])).abs() < 1e-12);
            assert!((confidence(policy, &[0.0, 1.0]) - 1.0).abs() < 1e-12);
        }
        assert_eq!(
            confidence(DecisionConfidence::MaximumProbability, &[0.25, 0.75]),
            0.75
        );
        assert_eq!(confidence(DecisionConfidence::UniformMargin, &[1.0]), 1.0);
        let root = root();
        let engine = Engine::from_dir(&root, crate::EngineConfig::default()).unwrap();
        let tokenizer = Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
        let contract = decision_contract(engine.metadata()).unwrap().unwrap();
        assert!(
            PreparedDecision::prepare(
                &tokenizer,
                &contract,
                &serde_json::json!("hello"),
                question(DecisionKind::Choice),
                DecisionLimits {
                    max_input_tokens: NonZeroUsize::new(2).unwrap(),
                    ..Default::default()
                }
            )
            .unwrap_err()
            .to_string()
            .contains("budget")
        );
        let mut invalid = question(DecisionKind::Noul);
        invalid.options.reverse();
        assert!(
            PreparedDecision::prepare(
                &tokenizer,
                &contract,
                &serde_json::json!("hello"),
                invalid,
                DecisionLimits {
                    max_input_tokens: NonZeroUsize::new(64).unwrap(),
                    ..Default::default()
                }
            )
            .unwrap_err()
            .to_string()
            .contains("false then true")
        );
    }
}
