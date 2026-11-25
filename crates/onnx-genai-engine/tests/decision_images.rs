//! Actual Native CPU vision preprocessing and decision-v2 input/usage contracts.
use onnx_genai_engine::{
    Engine, EngineConfig,
    decision::{DecisionInput, DecisionLimits, DecisionQuestion, PreparedDecision},
};
use onnx_genai_metadata::decision::decision_contract;
use onnx_genai_ort::Tokenizer;
use serde_json::json;
use std::{num::NonZeroUsize, path::PathBuf, sync::Arc};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/decision_images")
}
fn question(engine: &Engine) -> DecisionQuestion {
    DecisionQuestion::from_value(
        &Arc::new(decision_contract(engine.metadata()).unwrap().unwrap()),
        json!({"criteria":{"a":"first", "b":"second"}}),
        NonZeroUsize::new(2).unwrap(),
    )
    .unwrap()
}
fn images() -> (Arc<[u8]>, Arc<[u8]>) {
    (
        std::fs::read(root().join("white.png")).unwrap().into(),
        std::fs::read(root().join("black.png")).unwrap().into(),
    )
}
fn text_usage(tokenizer: &Tokenizer) -> usize {
    tokenizer
        .inner()
        .encode("\"hello\" the", true)
        .unwrap()
        .get_attention_mask()
        .iter()
        .filter(|x| **x != 0)
        .count()
}
#[test]
fn native_decision_images_decode_in_order_and_report_media_usage() {
    let mut engine = Engine::from_dir(&root(), EngineConfig::default()).unwrap();
    let tokenizer = Tokenizer::from_file(root().join("tokenizer.json")).unwrap();
    let (white, black) = images();
    let mut logits = Vec::new();
    for ordered in [
        [white.clone(), white.clone(), white.clone()],
        [white.clone(), black.clone(), white.clone()],
        [white.clone(), white.clone(), black.clone()],
    ] {
        let prepared = PreparedDecision::prepare(
            &tokenizer,
            DecisionInput {
                state: &json!("hello"),
                images: &ordered,
            },
            question(&engine),
            DecisionLimits::default(),
        )
        .unwrap();
        let result = engine.decide(prepared).unwrap();
        assert_eq!(result.prompt_tokens(), text_usage(&tokenizer) + 6);
        let probabilities = result.probabilities();
        assert_eq!(probabilities.len(), 2);
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        let p = probabilities[0];
        assert!(p > 0.0 && p < 1.0);
        logits.push((p / (1.0 - p)).ln());
    }
    assert!((logits[0] - logits[1] - 1.0).abs() < 1e-5, "{logits:?}");
    assert!((logits[0] - logits[2] - 2.0).abs() < 1e-5, "{logits:?}");
}
#[test]
fn native_decision_images_admit_positions_and_bytes_before_execution() {
    let engine = Engine::from_dir(&root(), EngineConfig::default()).unwrap();
    let tokenizer = Tokenizer::from_file(root().join("tokenizer.json")).unwrap();
    let (white, _) = images();
    for (ordered, diagnostic) in [
        (vec![white.clone(), white.clone()], "omits required"),
        (vec![white.clone(); 4], "supplies 4"),
        (
            vec![white.clone(), Arc::<[u8]>::from([]), white.clone()],
            "empty",
        ),
    ] {
        let failure = PreparedDecision::prepare(
            &tokenizer,
            DecisionInput {
                state: &json!("hello"),
                images: &ordered,
            },
            question(&engine),
            DecisionLimits::default(),
        )
        .unwrap_err();
        assert!(failure.to_string().contains(diagnostic), "{failure:#}");
    }
}
#[test]
fn native_decision_image_expansion_obeys_total_input_budget() {
    let mut engine = Engine::from_dir(&root(), EngineConfig::default()).unwrap();
    let tokenizer = Tokenizer::from_file(root().join("tokenizer.json")).unwrap();
    let (white, _) = images();
    let ordered = [white.clone(), white.clone(), white];
    let prepared = PreparedDecision::prepare(
        &tokenizer,
        DecisionInput {
            state: &json!("hello"),
            images: &ordered,
        },
        question(&engine),
        DecisionLimits {
            max_input_tokens: NonZeroUsize::new(text_usage(&tokenizer)).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    let failure = engine.decide(prepared).unwrap_err();
    assert!(
        failure.to_string().contains("including images, exceeding"),
        "{failure:#}"
    );
}

fn package_with(model_replace: Option<(&str, &str)>, reverse: bool) -> tempfile::TempDir {
    let package = tempfile::tempdir().unwrap();
    std::fs::copy(
        root().join("tokenizer.json"),
        package.path().join("tokenizer.json"),
    )
    .unwrap();
    let mut metadata = std::fs::read_to_string(root().join("inference_metadata.yaml")).unwrap();
    if reverse {
        metadata = metadata.replace(
            "\"variants\": \"direct\"",
            "\"variants\": \"reverse_average\"",
        );
    }
    std::fs::write(package.path().join("inference_metadata.yaml"), metadata).unwrap();
    let mut model = std::fs::read_to_string(root().join("model.onnx.textproto")).unwrap();
    if let Some((from, to)) = model_replace {
        assert!(model.contains(from));
        model = model.replace(from, to);
    }
    std::fs::write(package.path().join("model.onnx.textproto"), model).unwrap();
    package
}
#[test]
fn native_decision_image_usage_rejects_negative_and_underreported_counts() {
    for (count, diagnostic) in [(-999, "invalid count"), (-1, "underreports")] {
        let replacement =
            format!("name: \"media_tokens\" data_type: 7 dims: 1 int64_data: {count}");
        let package = package_with(
            Some((
                "name: \"media_tokens\" data_type: 7 dims: 1 int64_data: 6",
                &replacement,
            )),
            false,
        );
        let mut engine = Engine::from_dir(package.path(), EngineConfig::default()).unwrap();
        let tokenizer = Tokenizer::from_file(root().join("tokenizer.json")).unwrap();
        let (white, _) = images();
        let ordered = [white.clone(), white.clone(), white];
        let prepared = PreparedDecision::prepare(
            &tokenizer,
            DecisionInput {
                state: &json!("hello"),
                images: &ordered,
            },
            question(&engine),
            DecisionLimits::default(),
        )
        .unwrap();
        let failure = engine.decide(prepared).unwrap_err();
        assert!(failure.to_string().contains(diagnostic), "{failure:#}");
    }
}
#[test]
fn native_decision_image_variants_retain_media_and_sum_actual_usage() {
    let package = package_with(None, true);
    let mut engine = Engine::from_dir(package.path(), EngineConfig::default()).unwrap();
    let tokenizer = Tokenizer::from_file(root().join("tokenizer.json")).unwrap();
    let (white, _) = images();
    let ordered = [white.clone(), white.clone(), white];
    let prepared = PreparedDecision::prepare(
        &tokenizer,
        DecisionInput {
            state: &json!("hello"),
            images: &ordered,
        },
        question(&engine),
        DecisionLimits::default(),
    )
    .unwrap();
    let result = engine.decide(prepared).unwrap();
    assert_eq!(result.prompt_tokens(), 2 * (text_usage(&tokenizer) + 6));
    for p in result.probabilities() {
        assert!((p - 0.5).abs() < 1e-6);
    }
}
