//! Text/image pair scoring through a package-declared cross-encoder workflow.
use crate::{Engine, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest};
use anyhow::{Context, ensure};
use onnx_genai_metadata::task_profiles::{PairFeature, RerankingContract, reranking_contract};
use onnx_genai_ort::{Tokenizer, Value};
use std::{num::NonZeroUsize, sync::Arc};

/// Text and ordered encoded images for one side of a relevance pair.
pub struct RerankingInput<'a> {
    pub text: &'a str,
    pub images: &'a [Arc<[u8]>],
}
impl<'a> RerankingInput<'a> {
    pub fn text(text: &'a str) -> Self {
        Self { text, images: &[] }
    }
}

/// One admitted query/document pair with tokenizer features and named media.
#[derive(Debug)]
pub struct PreparedRerankingPair {
    ids: Vec<u32>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
    images: Vec<(String, Arc<[u8]>)>,
    contract: RerankingContract,
}
impl PreparedRerankingPair {
    pub fn prepare(
        tokenizer: &Tokenizer,
        contract: &RerankingContract,
        query: RerankingInput<'_>,
        document: RerankingInput<'_>,
        max_input_tokens: NonZeroUsize,
    ) -> anyhow::Result<Self> {
        contract.admit_images(query.images.len(), document.images.len())?;
        let mut images = Vec::with_capacity(query.images.len() + document.images.len());
        for (bindings, data) in [
            (contract.query_images(), query.images),
            (contract.document_images(), document.images),
        ] {
            for (binding, bytes) in bindings.iter().zip(data) {
                binding.admit_bytes(bytes.len())?;
                images.push((binding.endpoint().to_owned(), Arc::clone(bytes)));
            }
        }
        let encoding=tokenizer.inner().encode((query.text,document.text),true).map_err(|err|anyhow::anyhow!("reranking pair tokenization failed: {err}; check the package tokenizer's pair post-processor"))?;
        let ids = encoding.get_ids().to_vec();
        ensure!(
            !ids.is_empty(),
            "reranking query/document pair encodes to no tokens; provide nonempty text or declare tokenizer special tokens"
        );
        ensure!(
            ids.len() <= max_input_tokens.get(),
            "reranking pair has {} tokens, exceeding the configured input limit {}; shorten query/document or configure explicit tokenizer truncation",
            ids.len(),
            max_input_tokens
        );
        if let Some(length) = contract.fixed_sequence_length() {
            ensure!(
                ids.len() == length,
                "reranking tokenizer produced {} tokens, but profile '{}' requires {length}; configure tokenizer padding/truncation for that fixed input contract",
                ids.len(),
                contract.profile()
            );
        }
        ensure!(
            encoding.get_attention_mask().len() == ids.len()
                && encoding.get_type_ids().len() == ids.len(),
            "reranking tokenizer returned inconsistent input, attention-mask or token-type row lengths; repair tokenizer.json"
        );
        Ok(Self {
            images,
            contract: contract.clone(),
            ids,
            attention_mask: encoding
                .get_attention_mask()
                .iter()
                .map(|value| i64::from(*value))
                .collect(),
            token_type_ids: encoding
                .get_type_ids()
                .iter()
                .map(|value| i64::from(*value))
                .collect(),
        })
    }
    pub fn prompt_tokens(&self) -> usize {
        self.attention_mask
            .iter()
            .filter(|value| **value != 0)
            .count()
    }
}
/// One final score and the actual non-padding pair token count.
#[derive(Debug)]
pub struct RerankingScore {
    pub probability: f64,
    pub prompt_tokens: usize,
}

impl Engine {
    /// Score one tokenizer-admitted pair using this model's declared scoring head.
    pub fn score_reranking_pair(
        &mut self,
        pair: PreparedRerankingPair,
    ) -> anyhow::Result<RerankingScore> {
        let contract = reranking_contract(self.metadata())?.context(
            "model declares no reranking profile; select a package exposing a final score output",
        )?;
        ensure!(
            pair.contract == contract,
            "prepared reranking pair does not match this model's profile; prepare it with the target model's declared contract"
        );
        if let Some(limit) = self.effective_max_context(&GenerateOptions::default()) {
            ensure!(
                pair.ids.len() <= limit,
                "reranking pair has {} tokens but the model context limit is {limit}; shorten input or configure declared tokenizer truncation",
                pair.ids.len()
            );
        }
        if let Some(length) = contract.fixed_sequence_length() {
            ensure!(
                pair.ids.len() == length,
                "reranking pair no longer matches the admitted profile's fixed sequence length {length}"
            );
        }
        let shape = [1, i64::try_from(pair.ids.len())?];
        let prompt_tokens = pair.prompt_tokens();
        let mut request = PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(pair.ids),
            options: GenerateOptions {
                max_new_tokens: 1,
                ..Default::default()
            },
        });
        for (name, feature) in contract.bindings() {
            let row = match feature {
                PairFeature::AttentionMask => &pair.attention_mask,
                PairFeature::TokenTypeIds => &pair.token_type_ids,
            };
            request = request.with_input(name, Value::from_vec_i64(row.clone(), &shape)?);
        }
        for (name, bytes) in pair.images {
            let value = self.encoded_image_value(&name, &bytes)?;
            request = request.with_input(name, value);
        }
        let outputs = self
            .run_pipeline_outputs(request)
            .context("reranking workflow execution failed")?;
        let value = outputs
            .aggregate(contract.score_output())
            .with_context(|| {
                format!(
                    "reranking workflow did not emit declared score output '{}'",
                    contract.score_output()
                )
            })?;
        ensure!(
            value.numel() == 1,
            "reranking output '{}' contains {} values; declare a final scalar match probability in the workflow",
            contract.score_output(),
            value.numel()
        );
        let probability = f64::from(
            value.to_vec_f32_lossy().context(
                "reranking score must be a host-readable float16/bfloat16/float32 scalar",
            )?[0],
        );
        ensure!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "reranking output '{}' produced {probability}; the profile must expose a final match probability in [0,1] (apply sigmoid/softmax in the model workflow)",
            contract.score_output()
        );
        let prompt_tokens = if let Some(name) = contract.prompt_tokens_output() {
            let value = outputs.aggregate(name).with_context(|| {
                format!("reranking workflow did not emit declared token-usage output '{name}'")
            })?;
            ensure!(
                value.numel() == 1,
                "reranking token-usage output '{name}' must contain one count per pair"
            );
            let count = value.to_vec_i64().with_context(|| {
                format!("reranking token-usage output '{name}' must be a host-readable int64 count")
            })?[0];
            usize::try_from(count).with_context(|| format!("reranking token-usage output '{name}' returned invalid count {count}; emit a nonnegative representable token count"))?
        } else {
            prompt_tokens
        };
        Ok(RerankingScore {
            probability,
            prompt_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reranking_rejects_a_workflow_exposing_raw_logits_as_probabilities() {
        let source = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/reranking");
        let package = tempfile::tempdir().unwrap();
        for name in ["inference_metadata.yaml", "tokenizer.json"] {
            std::fs::copy(source.join(name), package.path().join(name)).unwrap();
        }
        let graph = std::fs::read_to_string(source.join("model.onnx.textproto"))
            .unwrap()
            .replace("op_type: \"Sigmoid\"", "op_type: \"Identity\"");
        std::fs::write(package.path().join("model.onnx.textproto"), graph).unwrap();
        let mut engine = Engine::from_dir(package.path(), crate::EngineConfig::default()).unwrap();
        let contract = reranking_contract(engine.metadata()).unwrap().unwrap();
        let tokenizer = Tokenizer::from_file(package.path().join("tokenizer.json")).unwrap();
        let pair = PreparedRerankingPair::prepare(
            &tokenizer,
            &contract,
            RerankingInput::text("hello"),
            RerankingInput::text("world"),
            NonZeroUsize::new(64).unwrap(),
        )
        .unwrap();
        let error = engine.score_reranking_pair(pair).unwrap_err().to_string();
        assert!(error.contains("[0,1]"), "{error}");
        assert!(error.contains("sigmoid/softmax"), "{error}");
    }

    #[test]
    fn declared_cross_encoder_scores_query_document_pairs_and_reports_usage() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/reranking");
        let mut engine = Engine::from_dir(&root, crate::EngineConfig::default()).unwrap();
        let contract = reranking_contract(engine.metadata()).unwrap().unwrap();
        let tokenizer = Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
        for (query, matching, other) in [("hello", "hello", "world"), ("world", "world", "hello")] {
            let pair = PreparedRerankingPair::prepare(
                &tokenizer,
                &contract,
                RerankingInput::text(query),
                RerankingInput::text(matching),
                NonZeroUsize::new(64).unwrap(),
            )
            .unwrap();
            let matching = engine.score_reranking_pair(pair).unwrap();
            assert_eq!(matching.prompt_tokens, 5);
            assert!((matching.probability - 0.5).abs() < 1e-6);
            let pair = PreparedRerankingPair::prepare(
                &tokenizer,
                &contract,
                RerankingInput::text(query),
                RerankingInput::text(other),
                NonZeroUsize::new(64).unwrap(),
            )
            .unwrap();
            assert!(engine.score_reranking_pair(pair).unwrap().probability < matching.probability);
        }
        assert!(
            PreparedRerankingPair::prepare(
                &tokenizer,
                &contract,
                RerankingInput::text("hello"),
                RerankingInput::text("world"),
                NonZeroUsize::new(4).unwrap()
            )
            .unwrap_err()
            .to_string()
            .contains("input limit")
        );
    }
    #[test]
    fn multimodal_pairs_decode_images_and_preserve_query_document_order() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/reranking_multimodal");
        let mut engine = Engine::from_dir(&root, crate::EngineConfig::default()).unwrap();
        let contract = reranking_contract(engine.metadata()).unwrap().unwrap();
        let tokenizer = Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
        let white: Arc<[u8]> = std::fs::read(root.join("white.png")).unwrap().into();
        let black: Arc<[u8]> = std::fs::read(root.join("black.png")).unwrap().into();
        assert_ne!(
            white.len(),
            black.len(),
            "encoded inputs exercise independent byte dimensions"
        );
        let query_images = [Arc::clone(&white)];
        for (images, distance) in [
            ([Arc::clone(&white), Arc::clone(&white)], 0.0_f64),
            ([Arc::clone(&black), Arc::clone(&white)], 1.0),
            ([Arc::clone(&white), Arc::clone(&black)], 2.0),
        ] {
            let pair = PreparedRerankingPair::prepare(
                &tokenizer,
                &contract,
                RerankingInput {
                    text: "hello",
                    images: &query_images,
                },
                RerankingInput {
                    text: "hello",
                    images: &images,
                },
                NonZeroUsize::new(64).unwrap(),
            )
            .unwrap();
            let result = engine.score_reranking_pair(pair).unwrap();
            assert!(
                (result.probability - 1.0 / (1.0 + distance.exp())).abs() < 1e-5,
                "distance={distance}, score={}",
                result.probability
            );
            assert_eq!(result.prompt_tokens, 5);
        }
        let error = PreparedRerankingPair::prepare(
            &tokenizer,
            &contract,
            RerankingInput::text("hello"),
            RerankingInput::text("hello"),
            NonZeroUsize::new(64).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("query.image"), "{error}");
    }
    #[test]
    fn optional_reranking_images_follow_workflow_presence_branches() {
        let source = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/reranking_multimodal");
        let root = tempfile::tempdir().unwrap();
        for name in ["tokenizer.json", "model.onnx.textproto"] {
            std::fs::copy(source.join(name), root.path().join(name)).unwrap();
        }
        let mut metadata: serde_yaml::Value = serde_yaml::from_str(
            &std::fs::read_to_string(source.join("inference_metadata.yaml")).unwrap(),
        )
        .unwrap();
        let workflow = &mut metadata["pipeline"]["workflow"];
        workflow["inputs"]["request.document_secondary"]["required"] =
            serde_yaml::Value::Bool(false);
        workflow["inputs"]["request.document_secondary"]["present_as"] =
            serde_yaml::Value::String("media.has_secondary".into());
        workflow["steps"][2] = serde_yaml::from_str(
            r#"
kind: branch
predicate: media.has_secondary
cases:
  'true':
    kind: invoke
    component: image_preprocess
    inputs: { encoded: request.document_secondary }
    outputs: { pixel_values: pixels.secondary.provided }
  'false':
    kind: invoke
    component: identity_pixels
    inputs: { pixels: pixels.primary }
    outputs: { copy: pixels.secondary.default }
outputs:
  pixels.secondary:
    cases: { 'true': pixels.secondary.provided, 'false': pixels.secondary.default }
"#,
        )
        .unwrap();
        workflow["components"]["identity_pixels"] = serde_yaml::from_str(
            r#"
implementation: { kind: onnx, artifact: identity.onnx.textproto }
ports:
  inputs: { pixels: { dtype: float32, shape: [image_batch, 3, 2, 2] } }
  outputs: { copy: { dtype: float32, shape: [image_batch, 3, 2, 2] } }
"#,
        )
        .unwrap();
        std::fs::write(root.path().join("identity.onnx.textproto"),r#"
ir_version: 11
opset_import { domain: "" version: 24 }
graph {
 name: "declared_optional_image_fallback"
 node { input: "pixels" output: "copy" op_type: "Identity" }
 input { name: "pixels" type { tensor_type { elem_type: 1 shape { dim { dim_param: "image_batch" } dim { dim_value: 3 } dim { dim_value: 2 } dim { dim_value: 2 } } } } }
 output { name: "copy" type { tensor_type { elem_type: 1 shape { dim { dim_param: "image_batch" } dim { dim_value: 3 } dim { dim_value: 2 } dim { dim_value: 2 } } } } }
}
"#).unwrap();
        std::fs::write(
            root.path().join("inference_metadata.yaml"),
            serde_yaml::to_string(&metadata).unwrap(),
        )
        .unwrap();
        let mut engine = Engine::from_dir(root.path(), crate::EngineConfig::default()).unwrap();
        let contract = reranking_contract(engine.metadata()).unwrap().unwrap();
        let tokenizer = Tokenizer::from_file(root.path().join("tokenizer.json")).unwrap();
        let white: Arc<[u8]> = std::fs::read(source.join("white.png")).unwrap().into();
        let black: Arc<[u8]> = std::fs::read(source.join("black.png")).unwrap().into();
        let query = [Arc::clone(&white)];
        for (documents, distance) in [
            (vec![Arc::clone(&white)], 0.0_f64),
            (vec![white, black], 2.0),
        ] {
            let pair = PreparedRerankingPair::prepare(
                &tokenizer,
                &contract,
                RerankingInput {
                    text: "hello",
                    images: &query,
                },
                RerankingInput {
                    text: "hello",
                    images: &documents,
                },
                NonZeroUsize::new(64).unwrap(),
            )
            .unwrap();
            let score = engine.score_reranking_pair(pair).unwrap();
            assert!(
                (score.probability - 1.0 / (1.0 + distance.exp())).abs() < 1e-5,
                "{score:?}"
            );
        }
    }
    #[test]
    fn multimodal_reranking_rejects_negative_workflow_usage_counts() {
        let source = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/reranking_multimodal");
        let root = tempfile::tempdir().unwrap();
        for name in ["inference_metadata.yaml", "tokenizer.json"] {
            std::fs::copy(source.join(name), root.path().join(name)).unwrap();
        }
        let graph=std::fs::read_to_string(source.join("model.onnx.textproto")).unwrap().replace(
            "node { input: \"attention_mask\" input: \"sequence_axis\" output: \"prompt_tokens\" op_type: \"ReduceSum\" attribute { name: \"keepdims\" i: 0 type: INT } }",
            "node { output: \"prompt_tokens\" op_type: \"Constant\" attribute { name: \"value\" t { dims: 1 data_type: 7 int64_data: -1 } type: TENSOR } }"
        );
        assert!(graph.contains("int64_data: -1"));
        std::fs::write(root.path().join("model.onnx.textproto"), graph).unwrap();
        let mut engine = Engine::from_dir(root.path(), crate::EngineConfig::default()).unwrap();
        let contract = reranking_contract(engine.metadata()).unwrap().unwrap();
        let tokenizer = Tokenizer::from_file(root.path().join("tokenizer.json")).unwrap();
        let white: Arc<[u8]> = std::fs::read(source.join("white.png")).unwrap().into();
        let query = [Arc::clone(&white)];
        let document = [Arc::clone(&white), white];
        let pair = PreparedRerankingPair::prepare(
            &tokenizer,
            &contract,
            RerankingInput {
                text: "hello",
                images: &query,
            },
            RerankingInput {
                text: "hello",
                images: &document,
            },
            NonZeroUsize::new(64).unwrap(),
        )
        .unwrap();
        let error = engine.score_reranking_pair(pair).unwrap_err().to_string();
        assert!(
            error.contains("invalid count -1") && error.contains("nonnegative"),
            "{error}"
        );
    }
}
