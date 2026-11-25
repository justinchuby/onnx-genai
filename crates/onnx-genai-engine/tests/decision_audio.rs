//! Real mixed vision/audio preparation and spoken-question Native inference.
use onnx_genai_engine::{
    Engine, EngineConfig,
    decision::{
        DecisionAnswer, DecisionInput, DecisionLimits, DecisionMedia, DecisionQuestion,
        PreparedDecision,
    },
};
use onnx_genai_metadata::decision::decision_contract;
use onnx_genai_ort::Tokenizer;
use serde_json::json;
use std::{num::NonZeroUsize, path::PathBuf, sync::Arc};
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/decision_audio")
}
fn wav(samples: usize, tone: bool) -> Arc<[u8]> {
    let values = (0..samples)
        .map(|i| {
            if tone {
                (i as f32 * 0.19).sin() * 0.5
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    onnx_genai_preprocess::audio::encode_wav_pcm16(&values, 16000, 1)
        .unwrap()
        .into()
}
fn picture(name: &str) -> Arc<[u8]> {
    std::fs::read(root().join(name)).unwrap().into()
}
fn question(engine: &Engine) -> DecisionQuestion {
    DecisionQuestion::from_value(&Arc::new(decision_contract(engine.metadata()).unwrap().unwrap()),json!({"instructions":{"type":"input_audio","input_audio":{"data":"wire bytes are supplied separately, never rendered"}},"criteria":{"a":"first","b":"second"}}),NonZeroUsize::new(2).unwrap()).unwrap()
}
fn text_tokens(tokenizer: &Tokenizer) -> usize {
    tokenizer
        .inner()
        .encode(
            format!("{} spoken [\"image\", \"audio\"]", json!("hello")),
            true,
        )
        .unwrap()
        .get_attention_mask()
        .iter()
        .filter(|x| **x != 0)
        .count()
}
#[test]
fn native_mixed_decision_media_and_spoken_audio_affect_probabilities_and_actual_usage() {
    let mut engine = Engine::from_dir(&root(), EngineConfig::default()).unwrap();
    let tokenizer = Tokenizer::from_file(root().join("tokenizer.json")).unwrap();
    let mut probabilities = Vec::new();
    for (name, state_tone, question_tone) in [
        ("white.png", false, false),
        ("black.png", false, false),
        ("white.png", true, false),
        ("white.png", false, true),
    ] {
        let media = [
            DecisionMedia::image(picture(name)),
            DecisionMedia::audio(wav(1600, state_tone)),
        ];
        let spoken = wav(3200, question_tone);
        let prepared = PreparedDecision::prepare(
            &tokenizer,
            DecisionInput {
                state: &json!("hello"),
                media: &media,
                spoken: Some(&spoken),
            },
            question(&engine),
            DecisionLimits::default(),
        )
        .unwrap();
        let result = engine.decide(prepared).unwrap();
        assert_eq!(
            result.prompt_tokens(),
            text_tokens(&tokenizer) + 10 + 20 + 1
        );
        let DecisionAnswer::Choice { key, .. } = result.answer() else {
            panic!("choice head must return choice")
        };
        if !state_tone && !question_tone {
            assert_eq!(key, if name == "white.png" { "a" } else { "b" });
        }
        probabilities.push(result.probabilities()[0]);
    }
    assert!(probabilities[0] > probabilities[1]);
    assert!(
        probabilities[2] > probabilities[0],
        "state audio must reach its own head input: {probabilities:?}"
    );
    assert!(
        probabilities[3] < probabilities[0],
        "question audio must reach its own head input: {probabilities:?}"
    );
}
#[test]
fn native_decision_audio_frame_expansion_is_counted_and_bounded() {
    let mut engine = Engine::from_dir(&root(), EngineConfig::default()).unwrap();
    let tokenizer = Tokenizer::from_file(root().join("tokenizer.json")).unwrap();
    let media = [
        DecisionMedia::image(picture("white.png")),
        DecisionMedia::audio(wav(1600, false)),
    ];
    let mut usage = Vec::new();
    for samples in [1600, 3200] {
        let spoken = wav(samples, false);
        let prepared = PreparedDecision::prepare(
            &tokenizer,
            DecisionInput {
                state: &json!("hello"),
                media: &media,
                spoken: Some(&spoken),
            },
            question(&engine),
            DecisionLimits::default(),
        )
        .unwrap();
        usage.push(engine.decide(prepared).unwrap().prompt_tokens());
    }
    assert_eq!(usage[1] - usage[0], 10);
    let spoken = wav(3200, false);
    let prepared = PreparedDecision::prepare(
        &tokenizer,
        DecisionInput {
            state: &json!("hello"),
            media: &media,
            spoken: Some(&spoken),
        },
        question(&engine),
        DecisionLimits {
            max_input_tokens: NonZeroUsize::new(text_tokens(&tokenizer)).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    let failure = engine.decide(prepared).unwrap_err();
    assert!(
        failure
            .downcast_ref::<onnx_genai_engine::decision::DecisionInputLimitError>()
            .is_some(),
        "{failure:#}"
    );
}
#[test]
fn native_decision_spoken_and_mixed_media_are_admitted_before_execution() {
    let engine = Engine::from_dir(&root(), EngineConfig::default()).unwrap();
    let tokenizer = Tokenizer::from_file(root().join("tokenizer.json")).unwrap();
    let spoken = wav(3200, false);
    for (media, provided, diagnostic) in [
        (
            vec![
                DecisionMedia::audio(wav(1600, false)),
                DecisionMedia::image(picture("white.png")),
            ],
            Some(&spoken),
            "mixed-media order",
        ),
        (
            vec![DecisionMedia::image(picture("white.png"))],
            Some(&spoken),
            "omits required",
        ),
        (
            vec![
                DecisionMedia::image(picture("white.png")),
                DecisionMedia::audio(wav(1600, false)),
            ],
            None,
            "spoken-question bytes",
        ),
    ] {
        let failure = PreparedDecision::prepare(
            &tokenizer,
            DecisionInput {
                state: &json!("hello"),
                media: &media,
                spoken: provided,
            },
            question(&engine),
            DecisionLimits::default(),
        )
        .unwrap_err();
        assert!(failure.to_string().contains(diagnostic), "{failure:#}");
    }
    let contract = Arc::new(decision_contract(engine.metadata()).unwrap().unwrap());
    let failure = DecisionQuestion::from_value(
        &contract,
        json!({"instructions":"text", "criteria":{"a":"first","b":"second"}}),
        NonZeroUsize::new(2).unwrap(),
    )
    .unwrap_err();
    assert!(
        failure.to_string().contains("requires spoken instructions"),
        "{failure:#}"
    );
}
