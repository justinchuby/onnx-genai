//! Token-pair cross-encoder scoring through a package-declared workflow.
use crate::{Engine, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest};
use anyhow::{Context, ensure};
use onnx_genai_metadata::task_profiles::{PairFeature, RerankingContract, reranking_contract};
use onnx_genai_ort::{Tokenizer, Value};
use std::num::NonZeroUsize;

/// One admitted query/document pair with the tokenizer's complete feature rows.
#[derive(Debug)]
pub struct PreparedTokenPair {
    ids: Vec<u32>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
}
impl PreparedTokenPair {
    pub fn prepare(
        tokenizer: &Tokenizer,
        contract: &RerankingContract,
        query: &str,
        document: &str,
        max_input_tokens: NonZeroUsize,
    ) -> anyhow::Result<Self> {
        let encoding=tokenizer.inner().encode((query,document),true).map_err(|err|anyhow::anyhow!("reranking pair tokenization failed: {err}; check the package tokenizer's pair post-processor"))?;
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
    pub fn score_token_pair(&mut self, pair: PreparedTokenPair) -> anyhow::Result<RerankingScore> {
        let contract=reranking_contract(self.metadata())?.context("model declares no reranking v1 profile; select a package exposing a final score output")?;
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
        let pair = PreparedTokenPair::prepare(
            &tokenizer,
            &contract,
            "hello",
            "world",
            NonZeroUsize::new(64).unwrap(),
        )
        .unwrap();
        let error = engine.score_token_pair(pair).unwrap_err().to_string();
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
            let pair = PreparedTokenPair::prepare(
                &tokenizer,
                &contract,
                query,
                matching,
                NonZeroUsize::new(64).unwrap(),
            )
            .unwrap();
            let matching = engine.score_token_pair(pair).unwrap();
            assert_eq!(matching.prompt_tokens, 5);
            assert!((matching.probability - 0.5).abs() < 1e-6);
            let pair = PreparedTokenPair::prepare(
                &tokenizer,
                &contract,
                query,
                other,
                NonZeroUsize::new(64).unwrap(),
            )
            .unwrap();
            assert!(engine.score_token_pair(pair).unwrap().probability < matching.probability);
        }
        assert!(
            PreparedTokenPair::prepare(
                &tokenizer,
                &contract,
                "hello",
                "world",
                NonZeroUsize::new(4).unwrap()
            )
            .unwrap_err()
            .to_string()
            .contains("input limit")
        );
    }
}
