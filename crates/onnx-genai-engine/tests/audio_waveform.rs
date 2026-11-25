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
