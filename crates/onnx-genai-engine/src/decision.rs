//! Non-generative decisions over a model-authored workflow and prompt contract.
use crate::{Engine, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest};
use anyhow::{Context, ensure};
use onnx_genai_metadata::{
    DecisionConfidence, DecisionKind, DecisionLegend, DecisionOption, DecisionScore,
    DecisionVariants,
    decision::{DecisionContract, DecisionFeature, decision_contract},
};
use onnx_genai_ort::{ChatTemplate, TemplateLimits, Tokenizer, Value};
use serde::Serialize;
mod input;
pub use input::DecisionQuestion;
use std::{collections::BTreeMap, num::NonZeroUsize};

/// Actual workflow media expansion exceeded the caller/model input allowance.
#[derive(Debug, thiserror::Error)]
#[error(
    "decision variant {variant} uses {used} tokens including media, exceeding its {limit}-token input budget; reduce media/state or raise the configured input limit within model context"
)]
pub struct DecisionInputLimitError {
    pub variant: usize,
    pub used: usize,
    pub limit: usize,
}

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

struct OptionView<'a> {
    values: &'a [DecisionOption],
    reverse: bool,
}
impl Serialize for OptionView<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.values.len()))?;
        for index in 0..self.values.len() {
            let i = if self.reverse {
                self.values.len() - 1 - index
            } else {
                index
            };
            seq.serialize_element(&self.values[i])?;
        }
        seq.end()
    }
}

#[derive(Debug)]
struct Variant {
    ids: Vec<u32>,
    mask: Vec<i64>,
    segments: Vec<i64>,
}
/// One typed media position; its bytes are shared across questions/variants.
#[derive(Debug, Clone)]
pub struct DecisionMedia {
    pub kind: onnx_genai_metadata::EncodedMediaKind,
    pub bytes: std::sync::Arc<[u8]>,
}
impl DecisionMedia {
    pub fn image(bytes: std::sync::Arc<[u8]>) -> Self {
        Self {
            kind: onnx_genai_metadata::EncodedMediaKind::Image,
            bytes,
        }
    }
    pub fn audio(bytes: std::sync::Arc<[u8]>) -> Self {
        Self {
            kind: onnx_genai_metadata::EncodedMediaKind::Audio,
            bytes,
        }
    }
}
/// Structured state, typed ordered media and independently bound spoken audio.
pub struct DecisionInput<'a> {
    pub state: &'a serde_json::Value,
    pub media: &'a [DecisionMedia],
    pub spoken: Option<&'a std::sync::Arc<[u8]>>,
}
impl<'a> DecisionInput<'a> {
    pub fn text(state: &'a serde_json::Value) -> Self {
        Self {
            state,
            media: &[],
            spoken: None,
        }
    }
}

/// Tokenizer features and output cardinality admitted before any inference.
#[derive(Debug)]
pub struct PreparedDecision {
    contract: std::sync::Arc<DecisionContract>,
    question: DecisionQuestion,
    variants: Vec<Variant>,
    inputs: Vec<(String, std::sync::Arc<[u8]>)>,
    max_input_tokens: NonZeroUsize,
}
impl PreparedDecision {
    pub fn prepare(
        tokenizer: &Tokenizer,
        input: DecisionInput<'_>,
        question: DecisionQuestion,
        limits: DecisionLimits,
    ) -> anyhow::Result<Self> {
        let state = input.state;
        let max_input_tokens = limits.max_input_tokens;
        let contract = std::sync::Arc::clone(question.contract());
        contract
            .input_policy()
            .state
            .validate(state, "state")
            .map_err(anyhow::Error::msg)?;
        contract.admit_media(
            &input
                .media
                .iter()
                .map(|media| media.kind)
                .collect::<Vec<_>>(),
        )?;
        ensure!(
            question.is_spoken() == input.spoken.is_some(),
            "decision spoken-question bytes must be supplied exactly when question.instructions is input_audio; prepare matching encoded audio instead of dropping or inventing instructions"
        );
        let mut inputs = contract
            .media()
            .iter()
            .zip(input.media)
            .map(|(binding, media)| {
                binding.input.admit_bytes(media.bytes.len())?;
                Ok((
                    binding.input.endpoint().to_owned(),
                    std::sync::Arc::clone(&media.bytes),
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if let Some(bytes) = input.spoken {
            let binding = contract
                .spoken_input()
                .context("decision has no spoken input binding; select an audio-capable package")?;
            binding.admit_bytes(bytes.len())?;
            inputs.push((binding.endpoint().to_owned(), std::sync::Arc::clone(bytes)));
        }
        let n = match contract.spec().variants {
            DecisionVariants::Direct => 1,
            DecisionVariants::ReverseAverage => 2,
        };
        let template = ChatTemplate::from_source(&contract.spec().prompt_template);
        let mut variants = Vec::new();
        for variant in 0..n {
            let options = OptionView {
                values: question.option_values(),
                reverse: variant != 0,
            };
            let kwargs = BTreeMap::from([
                ("state".into(), state.clone()),
                (
                    "image_count".into(),
                    serde_json::json!(
                        input
                            .media
                            .iter()
                            .filter(|m| m.kind == onnx_genai_metadata::EncodedMediaKind::Image)
                            .count()
                    ),
                ),
                (
                    "audio_count".into(),
                    serde_json::json!(
                        input
                            .media
                            .iter()
                            .filter(|m| m.kind == onnx_genai_metadata::EncodedMediaKind::Audio)
                            .count()
                    ),
                ),
                (
                    "media_types".into(),
                    serde_json::to_value(input.media.iter().map(|m| m.kind).collect::<Vec<_>>())?,
                ),
                (
                    "spoken_instructions".into(),
                    serde_json::json!(question.is_spoken()),
                ),
                ("instructions".into(), question.instructions().clone()),
                ("type".into(), serde_json::to_value(question.kind())?),
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
            inputs,
            max_input_tokens,
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
    answer: DecisionAnswer,
    options: input::QuestionOptions,
    probabilities: Vec<f64>,
    prompt_tokens: usize,
    _credit: onnx_runtime_memory_governor::MemoryLease,
}
impl DecisionResult {
    pub fn answer(&self) -> &DecisionAnswer {
        &self.answer
    }
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }
    pub fn options(&self) -> &[DecisionOption] {
        self.options.values()
    }
    pub fn legend(&self) -> anyhow::Result<BTreeMap<String, serde_json::Value>> {
        ensure!(
            matches!(&self.answer, DecisionAnswer::Score { .. }),
            "only score decision results have a legend"
        );
        let policy = self
            .options
            .policy()
            .score
            .as_ref()
            .context("this result is not a score decision; only score answers have a legend")?
            .legend;
        self.options()
            .iter()
            .map(|o| {
                let value =
                    if policy == DecisionLegend::CanonicalJsonText && !o.description.is_string() {
                        serde_json::Value::String(
                            serde_json::to_string(&SortedJson(&o.description))
                                .context("serialize canonical decision score legend")?,
                        )
                    } else {
                        o.description.clone()
                    };
                Ok((o.key.clone(), value))
            })
            .collect()
    }
}
struct SortedJson<'a>(&'a serde_json::Value);
impl Serialize for SortedJson<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self.0 {
            serde_json::Value::Object(object) => {
                let mut entries = object.iter().collect::<Vec<_>>();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, &SortedJson(value))?;
                }
                map.end()
            }
            serde_json::Value::Array(array) => {
                let mut seq = serializer.serialize_seq(Some(array.len()))?;
                for value in array {
                    seq.serialize_element(&SortedJson(value))?;
                }
                seq.end()
            }
            value => value.serialize(serializer),
        }
    }
}

impl Engine {
    pub fn decide(&mut self, request: PreparedDecision) -> anyhow::Result<DecisionResult> {
        let contract=decision_contract(self.metadata())?.context("model has no decision workflow profile; select a package declaring a non-generative probability head")?;
        ensure!(
            &contract == request.contract.as_ref(),
            "prepared decision does not match this model's admitted profile; prepare it against the executing model's decision contract"
        );
        let PreparedDecision {
            question,
            variants,
            inputs,
            max_input_tokens,
            ..
        } = request;
        let count = question.option_values().len();
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
            let text_tokens = variant.mask.iter().filter(|x| **x != 0).count();
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
            for (name, bytes) in &inputs {
                execution = execution.with_input(name, self.encoded_media_value(name, bytes)?);
            }
            let outputs = self.run_pipeline_outputs(execution).with_context(|| {
                format!(
                    "execute decision profile '{}' variant {index}",
                    contract.profile()
                )
            })?;
            let used = if let Some(name) = contract.prompt_tokens_output() {
                let usage = outputs.aggregate(name).with_context(|| {
                    format!("decision workflow did not emit token-usage output '{name}'")
                })?;
                ensure!(
                    usage.numel() == 1,
                    "decision token-usage output '{name}' must contain one count per variant"
                );
                let count = usage.to_vec_i64().with_context(|| {
                    format!("decision token-usage output '{name}' must be host-readable int64")
                })?[0];
                let count = usize::try_from(count).with_context(|| format!("decision token-usage output '{name}' returned invalid count {count}; emit a nonnegative representable count"))?;
                ensure!(
                    count >= text_tokens,
                    "decision token-usage output '{name}' underreports {count} tokens for {text_tokens} admitted text tokens; include text and media expansion in usage"
                );
                let limit = self
                    .effective_max_context(&GenerateOptions::default())
                    .unwrap_or(usize::MAX)
                    .min(max_input_tokens.get());
                if count > limit {
                    return Err(DecisionInputLimitError {
                        variant: index,
                        used: count,
                        limit,
                    }
                    .into());
                }
                count
            } else {
                text_tokens
            };
            prompt_tokens = prompt_tokens
                .checked_add(used)
                .context("decision token usage overflow; reduce input length")?;
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
        let answer = match question.kind() {
            DecisionKind::Choice => {
                let best = probabilities.iter().enumerate().fold(0, |best, (i, p)| {
                    if *p > probabilities[best] { i } else { best }
                });
                DecisionAnswer::Choice {
                    key: question.option_values()[best].key.clone(),
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
            DecisionKind::Noul => {
                let index=question.option_values().iter().position(|o|o.key=="true").context("resolved boolean policy has no true option; repair the input policy interpreter")?;
                DecisionAnswer::Noul {
                    probability: probabilities[index],
                }
            }
        };
        Ok(DecisionResult {
            answer,
            options: question.into_options(),
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
    fn question(
        contract: &std::sync::Arc<DecisionContract>,
        kind: DecisionKind,
    ) -> DecisionQuestion {
        let criteria = match kind {
            DecisionKind::Choice => serde_json::json!({"a":"hello","b":"world"}),
            DecisionKind::Score => serde_json::json!(["hello", "world"]),
            DecisionKind::Noul => serde_json::json!({"false":"hello","true":"world"}),
        };
        DecisionQuestion::from_value(
            contract,
            serde_json::json!({"type":kind,"instructions":"the","criteria":criteria}),
            contract.spec().max_options,
        )
        .unwrap()
    }
    #[test]
    fn decision_runs_the_declared_head_and_changes_with_state_for_every_question_kind() {
        let root = root();
        let tokenizer = Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
        let mut engine = Engine::from_dir(&root, crate::fixture_engine_config()).unwrap();
        let contract = std::sync::Arc::new(decision_contract(engine.metadata()).unwrap().unwrap());
        for kind in [
            DecisionKind::Choice,
            DecisionKind::Score,
            DecisionKind::Noul,
        ] {
            for (state, best) in [("hello", 0), ("world", 1)] {
                let prepared = PreparedDecision::prepare(
                    &tokenizer,
                    DecisionInput::text(&serde_json::json!(state)),
                    question(&contract, kind),
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
        let mut engine = Engine::from_dir(package.path(), crate::fixture_engine_config()).unwrap();
        let tokenizer = Tokenizer::from_file(package.path().join("tokenizer.json")).unwrap();
        let contract = std::sync::Arc::new(decision_contract(engine.metadata()).unwrap().unwrap());
        for (state, level) in [("hello", 0.0), ("world", 1.0)] {
            let prepared = PreparedDecision::prepare(
                &tokenizer,
                DecisionInput::text(&serde_json::json!(state)),
                question(&contract, DecisionKind::Score),
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
    fn dynamic_option_package(
        scalar: bool,
        edit: impl FnOnce(&mut serde_yaml::Value),
    ) -> tempfile::TempDir {
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
            edit(&mut doc);
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
        package
    }
    #[test]
    fn dynamic_option_heads_execute_scalar_and_vector_count_bindings() {
        for scalar in [false, true] {
            let package = dynamic_option_package(scalar, |_| {});
            let mut engine =
                Engine::from_dir(package.path(), crate::fixture_engine_config()).unwrap();
            let tokenizer = Tokenizer::from_file(package.path().join("tokenizer.json")).unwrap();
            let contract =
                std::sync::Arc::new(decision_contract(engine.metadata()).unwrap().unwrap());
            for count in [3, 4] {
                let criteria = ["hello", "world", "quick", "brown"]
                    .into_iter()
                    .take(count)
                    .enumerate()
                    .map(|(i, text)| (format!("key-{i}"), serde_json::json!(text)))
                    .collect::<serde_json::Map<_, _>>();
                let question = DecisionQuestion::from_value(
                    &contract,
                    serde_json::json!({"type":"choice","instructions":"the","criteria":criteria}),
                    contract.spec().max_options,
                )
                .unwrap();
                let prepared = PreparedDecision::prepare(
                    &tokenizer,
                    DecisionInput::text(&serde_json::json!("quick")),
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
    fn authored_score_ranges_execute_single_and_eleven_level_heads() {
        let package = dynamic_option_package(false, |doc| {
            doc["schema_version"] = serde_yaml::Value::String("v1.11".into());
            doc["profiles"]["decision_head"]["decision"]["max_options"] =
                serde_yaml::to_value(16).unwrap();
            doc["profiles"]["decision_head"]["decision"]["input_policy"]=serde_yaml::to_value(serde_json::json!({
                "state":"payload_or_null","instructions":{"kind":"optional","default":"the","payload":"text"},
                "default_type":"score","max_questions":null,"choice":null,"noul":null,
                "score":{"minimum":1,"maximum":16,"descriptions":"text","legend":"structured"}
            })).unwrap();
        });
        let mut engine = Engine::from_dir(package.path(), crate::fixture_engine_config()).unwrap();
        let tokenizer = Tokenizer::from_file(package.path().join("tokenizer.json")).unwrap();
        let contract = std::sync::Arc::new(decision_contract(engine.metadata()).unwrap().unwrap());
        for count in [1, 11] {
            let q = DecisionQuestion::from_value(
                &contract,
                serde_json::json!({"criteria":vec!["hello";count]}),
                contract.spec().max_options,
            )
            .unwrap();
            let prepared = PreparedDecision::prepare(
                &tokenizer,
                DecisionInput::text(&serde_json::Value::Null),
                q,
                DecisionLimits::default(),
            )
            .unwrap();
            let result = engine.decide(prepared).unwrap();
            assert_eq!(result.probabilities().len(), count);
            match result.answer() {
                DecisionAnswer::Score { score, .. } => {
                    assert!((*score - ((count - 1) as f64) / 2.0).abs() < 1e-6)
                }
                _ => panic!("expected score"),
            }
            assert_eq!(result.prompt_tokens(), count + 4);
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
                    Engine::from_dir(package.path(), crate::fixture_engine_config()).unwrap();
                let tokenizer =
                    Tokenizer::from_file(package.path().join("tokenizer.json")).unwrap();
                let contract =
                    std::sync::Arc::new(decision_contract(engine.metadata()).unwrap().unwrap());
                let prepared = PreparedDecision::prepare(
                    &tokenizer,
                    DecisionInput::text(&serde_json::json!("hello")),
                    question(&contract, DecisionKind::Choice),
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
        let mut engine = Engine::from_dir(package.path(), crate::fixture_engine_config()).unwrap();
        let tokenizer = Tokenizer::from_file(package.path().join("tokenizer.json")).unwrap();
        let contract = std::sync::Arc::new(decision_contract(engine.metadata()).unwrap().unwrap());
        let prepared = PreparedDecision::prepare(
            &tokenizer,
            DecisionInput::text(&serde_json::json!("hello")),
            question(&contract, DecisionKind::Choice),
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
        let engine = Engine::from_dir(&root, crate::fixture_engine_config()).unwrap();
        let tokenizer = Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
        let contract = std::sync::Arc::new(decision_contract(engine.metadata()).unwrap().unwrap());
        assert!(
            PreparedDecision::prepare(
                &tokenizer,
                DecisionInput::text(&serde_json::json!("hello")),
                question(&contract, DecisionKind::Choice),
                DecisionLimits {
                    max_input_tokens: NonZeroUsize::new(2).unwrap(),
                    ..Default::default()
                }
            )
            .unwrap_err()
            .to_string()
            .contains("budget")
        );
        let error = DecisionQuestion::from_value(
            &contract,
            serde_json::json!({"type":"noul","instructions":"the","criteria":{"invalid":1}}),
            contract.spec().max_options,
        )
        .unwrap_err();
        assert!(error.to_string().contains("false and true"), "{error}");
    }
}
