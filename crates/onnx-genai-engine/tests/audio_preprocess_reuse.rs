//! Repeated real WAV preprocessing into independent Native component inputs.
use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};
use std::path::PathBuf;
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/audio_reuse")
}
fn wav(samples: usize, tone: bool) -> Vec<u8> {
    let values = (0..samples)
        .map(|i| {
            if tone {
                (i as f32 * 0.19).sin() * 0.5
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    onnx_genai_preprocess::audio::encode_wav_pcm16(&values, 16000, 1).unwrap()
}
fn request(first: Vec<u8>, second: Vec<u8>) -> PipelineGenerateRequest {
    let mut request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: GenerateOptions {
            max_new_tokens: 0,
            ..Default::default()
        },
    });
    for (name, bytes) in [("encoded.first", first), ("encoded.second", second)] {
        let size = bytes.len() as i64;
        request = request.with_input(
            name,
            Value::from_raw_bytes(bytes, &[size], DataType::Uint8).unwrap(),
        );
    }
    request
}
#[test]
fn native_reused_audio_component_keeps_clip_lengths_features_and_heads_independent() {
    let mut engine = Engine::from_dir(&root(), EngineConfig::default()).unwrap();
    for (a, b) in [(1600, 3200), (3200, 1600)] {
        let outputs = engine
            .run_pipeline_outputs(request(wav(a, false), wav(b, true)))
            .unwrap();
        let first = outputs.aggregate("first.features").unwrap();
        let second = outputs.aggregate("second.features").unwrap();
        assert_eq!(first.shape(), [1, 80, (a / 160) as i64]);
        assert_eq!(second.shape(), [1, 80, (b / 160) as i64]);
        assert_eq!(
            outputs
                .aggregate("first.length")
                .unwrap()
                .to_vec_i64()
                .unwrap(),
            [a as i64]
        );
        assert_eq!(
            outputs
                .aggregate("second.length")
                .unwrap()
                .to_vec_i64()
                .unwrap(),
            [b as i64]
        );
        let f = first.to_vec_f32().unwrap();
        let s = second.to_vec_f32().unwrap();
        assert_ne!(&f[..40], &s[..40]);
        for (name, data) in [("first.score", f), ("second.score", s)] {
            let expected = data.iter().sum::<f32>() / data.len() as f32;
            let actual = outputs.aggregate(name).unwrap().to_vec_f32().unwrap()[0];
            assert!(
                (actual - expected).abs() < 1e-5,
                "{name}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn native_reused_audio_recovers_after_rejected_encoded_clip() {
    let mut engine = Engine::from_dir(&root(), EngineConfig::default()).unwrap();
    let failure = engine
        .run_pipeline_outputs(request(vec![0, 1, 2], wav(3200, true)))
        .err()
        .expect("invalid WAV must fail before publication");
    let message = format!("{failure:#}");
    assert!(
        message.contains("audio") && (message.contains("WAV") || message.contains("RIFF")),
        "{message}"
    );
    let outputs = engine
        .run_pipeline_outputs(request(wav(1600, false), wav(3200, true)))
        .unwrap();
    assert_eq!(
        outputs
            .aggregate("first.length")
            .unwrap()
            .to_vec_i64()
            .unwrap(),
        [1600]
    );
    assert_eq!(
        outputs
            .aggregate("second.length")
            .unwrap()
            .to_vec_i64()
            .unwrap(),
        [3200]
    );
}
#[test]
fn native_audio_v1_keeps_its_workflow_scoped_single_invocation_contract() {
    let package = tempfile::tempdir().unwrap();
    for name in ["model.onnx.textproto", "tokenizer.json"] {
        std::fs::copy(root().join(name), package.path().join(name)).unwrap();
    }
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root().join("inference_metadata.yaml")).unwrap())
            .unwrap();
    metadata["preprocessing"]["audio"]["outputs"][0]["name"] = serde_json::json!("first.features");
    metadata["preprocessing"]["audio"]["outputs"][1]["name"] = serde_json::json!("first.length");
    let workflow = &mut metadata["pipeline"]["workflow"];
    workflow["manifest"]["adapter_abis"]["onnx-genai.audio-preprocess"] = serde_json::json!("1");
    workflow["components"]["audio"]["implementation"]["version"] = serde_json::json!("1");
    workflow["steps"].as_array_mut().unwrap().truncate(5);
    workflow["inputs"]
        .as_object_mut()
        .unwrap()
        .remove("encoded.second");
    workflow["outputs"]
        .as_object_mut()
        .unwrap()
        .retain(|name, _| name.starts_with("first."));
    std::fs::write(
        package.path().join("inference_metadata.yaml"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    let mut engine = Engine::from_dir(package.path(), EngineConfig::default()).unwrap();
    let mut input = request(wav(1600, false), wav(3200, true));
    input.inputs.remove("encoded.second");
    let outputs = engine.run_pipeline_outputs(input).unwrap();
    assert_eq!(
        outputs.aggregate("first.features").unwrap().shape(),
        [1, 80, 10]
    );
    assert_eq!(
        outputs
            .aggregate("first.length")
            .unwrap()
            .to_vec_i64()
            .unwrap(),
        [1600]
    );
}
