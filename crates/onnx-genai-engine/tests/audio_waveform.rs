//! Actual Native pure-waveform preprocessing and named branch publication.
use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};
use std::path::PathBuf;
#[test]
fn native_waveform_branches_normalization_padding_and_masks_reach_the_head() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/audio_waveform");
    let mut engine = Engine::from_dir(&root, EngineConfig::default()).unwrap();
    for samples in [vec![-0.5, -0.25, 0.0, 0.25, 0.5], vec![0.0, 0.25, 0.5]] {
        let bytes = onnx_genai_preprocess::audio::encode_wav_pcm16(&samples, 16000, 1).unwrap();
        let len = bytes.len() as i64;
        let request = PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![]),
            options: GenerateOptions {
                max_new_tokens: 0,
                ..Default::default()
            },
        })
        .with_input(
            "encoded",
            Value::from_raw_bytes(bytes, &[len], DataType::Uint8).unwrap(),
        );
        let outputs = engine.run_pipeline_outputs(request).unwrap();
        let raw = outputs
            .aggregate("audio.raw")
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(raw, samples);
        assert_eq!(
            outputs
                .aggregate("audio.scaled")
                .unwrap()
                .to_vec_f32()
                .unwrap(),
            samples.iter().map(|v| 2.0 * v).collect::<Vec<_>>()
        );
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance =
            samples.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / samples.len() as f32;
        let normalized = outputs
            .aggregate("audio.normalized")
            .unwrap()
            .to_vec_f32()
            .unwrap();
        for (actual, value) in normalized.iter().zip(&samples) {
            assert!((actual - (value - mean) / (variance + 1e-7).sqrt()).abs() < 1e-6);
        }
        let features = outputs.aggregate("audio.features").unwrap();
        assert_eq!(features.shape(), [1, 8]);
        let values = features.to_vec_f32().unwrap();
        assert_eq!(&values[..samples.len()], normalized);
        assert!(values[samples.len()..].iter().all(|v| *v == 0.75));
        assert_eq!(
            outputs
                .aggregate("audio.mask")
                .unwrap()
                .to_vec_i64()
                .unwrap(),
            (0..8)
                .map(|i| i64::from(i < samples.len()))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            outputs
                .aggregate("audio.valid_samples")
                .unwrap()
                .to_vec_i64()
                .unwrap(),
            [samples.len() as i64]
        );
        let score = outputs.aggregate("score").unwrap().to_vec_f32().unwrap()[0];
        assert!((score - values.iter().sum::<f32>() / 8.0).abs() < 1e-6);
    }
}

#[test]
fn native_opus_presentation_survives_authored_decode_resample_and_head() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let source = root.join("onnx_genai_workflows/audio_waveform");
    let package = tempfile::tempdir().unwrap();
    for name in ["tokenizer.json", "model.onnx.textproto"] {
        std::fs::copy(source.join(name), package.path().join(name)).unwrap();
    }
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("inference_metadata.yaml")).unwrap())
            .unwrap();
    // The existing tiny head consumes eight values. This authored window trims
    // only the feature branch; the published raw waveform retains the full clip.
    for transform in metadata["preprocessing"]["audio"]["transforms"]
        .as_array_mut()
        .unwrap()
    {
        if transform["op"] == "pad" {
            transform["mode"] = serde_json::json!("fixed_window");
        }
    }
    for output in metadata["preprocessing"]["audio"]["outputs"]
        .as_array_mut()
        .unwrap()
    {
        if output["name"] == "normalized" {
            output["contract"]["shape"][1] = serde_json::json!("resampled_samples");
        }
    }
    metadata["pipeline"]["workflow"]["components"]["audio"]["ports"]["outputs"]["normalized"]["shape"]
        [1] = serde_json::json!("resampled_samples");
    metadata["pipeline"]["workflow"]["outputs"]["audio.normalized"]["contract"]["shape"][1] =
        serde_json::json!("resampled_samples");
    std::fs::write(
        package.path().join("inference_metadata.yaml"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    let mut engine = Engine::from_dir(package.path(), EngineConfig::default()).unwrap();
    for (name, channels) in [
        ("silk-1", 1),
        ("hybrid-2", 2),
        ("celt-2", 2),
        ("mapping1-6", 6),
    ] {
        for suffix in ["ogg", "webm"] {
            let bytes = std::fs::read(root.join(format!("audio/opus/{name}.{suffix}"))).unwrap();
            let reference = std::fs::read(root.join(format!("audio/opus/{name}.reference.f32")))
                .unwrap()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            let request = PipelineGenerateRequest::new(GenerateRequest {
                prompt: GeneratePrompt::TokenIds(vec![]),
                options: GenerateOptions {
                    max_new_tokens: 0,
                    ..Default::default()
                },
            })
            .with_input(
                "encoded",
                Value::from_raw_bytes(bytes.clone(), &[bytes.len() as i64], DataType::Uint8)
                    .unwrap(),
            );
            let outputs = engine
                .run_pipeline_outputs(request)
                .unwrap_or_else(|error| panic!("{name}: {error:#}"));
            let raw = outputs.aggregate("audio.raw").unwrap();
            assert_eq!(
                raw.shape(),
                [1, 11520],
                "{name}: presented source frame count"
            );
            let raw = raw.to_vec_f32().unwrap();
            for (actual, frame) in raw.iter().zip(reference.chunks_exact(channels)) {
                let expected = (frame.iter().map(|sample| f64::from(*sample)).sum::<f64>()
                    / channels as f64) as f32;
                assert!(
                    (actual - expected).abs() <= 1.0 / 32768.0,
                    "{name}: independent source PCM reference"
                );
            }
            let scaled = outputs
                .aggregate("audio.scaled")
                .unwrap()
                .to_vec_f32()
                .unwrap();
            assert_eq!(
                scaled,
                raw.iter().map(|sample| 2.0 * sample).collect::<Vec<_>>()
            );
            let features = outputs.aggregate("audio.features").unwrap();
            assert_eq!(features.shape(), [1, 8]);
            let features = features.to_vec_f32().unwrap();
            assert!(features.iter().all(|value| value.is_finite()));
            assert_eq!(
                outputs
                    .aggregate("audio.valid_samples")
                    .unwrap()
                    .to_vec_i64()
                    .unwrap(),
                [8]
            );
            let score = outputs.aggregate("score").unwrap().to_vec_f32().unwrap()[0];
            assert!(
                (score - features.iter().sum::<f32>() / 8.0).abs() < 1e-6,
                "{name}: native head value"
            );
        }
    }
    let mut bounded = Engine::from_dir(
        package.path(),
        EngineConfig {
            audio_decode_limits: onnx_genai_preprocess::audio::AudioDecodeLimits {
                max_decoded_bytes: std::num::NonZeroUsize::new(1024).unwrap(),
            },
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let bytes = std::fs::read(root.join("audio/opus/celt-2.ogg")).unwrap();
    let len = bytes.len() as i64;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: GenerateOptions {
            max_new_tokens: 0,
            ..Default::default()
        },
    })
    .with_input(
        "encoded",
        Value::from_raw_bytes(bytes, &[len], DataType::Uint8).unwrap(),
    );
    let error = bounded
        .run_pipeline_outputs(request)
        .err()
        .expect("native workflow must enforce explicit decoder budget");
    let text = format!("{error:#}");
    assert!(
        text.contains("max_decoded_bytes=1024") && text.contains("component 'audio'"),
        "{text}"
    );
}
