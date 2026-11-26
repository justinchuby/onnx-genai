//! Authored linear-power spectra execute through the Native workflow and head.
#![cfg(feature = "native-backend")]
use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GeneratePrompt, GenerateRequest,
    PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};
use serde_json::json;
use std::path::PathBuf;

#[test]
fn native_spectrogram_features_validity_and_waveform_energy_reach_the_head() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/audio_waveform");
    let package = tempfile::tempdir().unwrap();
    std::fs::copy(
        source.join("tokenizer.json"),
        package.path().join("tokenizer.json"),
    )
    .unwrap();
    let head = std::fs::read_to_string(source.join("model.onnx.textproto"))
        .unwrap()
        .replace(
            "dim { dim_value: 8 }",
            "dim { dim_value: 3 } dim { dim_param: \"power_frames\" }",
        );
    std::fs::write(package.path().join("model.onnx.textproto"), head).unwrap();
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("inference_metadata.yaml")).unwrap())
            .unwrap();
    let program = &mut metadata["preprocessing"]["audio"];
    program["transforms"].as_array_mut().unwrap().extend([
        json!({"op":"spectrogram","inputs":["raw"],"outputs":["power"],"mode":"power","window":"hann","sample_rate":16000,"n_fft":4,"hop_length":1,"win_length":4}),
        json!({"op":"emit_valid_frames","inputs":["power"],"outputs":["power_length"]}),
        json!({"op":"emit_validity_mask","inputs":["power"],"outputs":["power_mask"]}),
    ]);
    let contracts = [
        (
            "spectrum",
            "power",
            "audio_features",
            json!({"dtype":"float32","shape":["audio_batch",3,"power_frames"]}),
        ),
        (
            "power_frames",
            "power_length",
            "valid_frames",
            json!({"dtype":"int64","shape":["audio_batch"]}),
        ),
        (
            "power_mask",
            "power_mask",
            "validity_mask",
            json!({"dtype":"int64","shape":["audio_batch","power_frames"]}),
        ),
    ];
    for (name, source, content, contract) in &contracts {
        program["outputs"].as_array_mut().unwrap().push(json!({"name":name,"source":source,"content":content,"dtype":contract["dtype"],"contract":contract}));
    }
    let workflow = &mut metadata["pipeline"]["workflow"];
    for (name, _, _, contract) in &contracts {
        let value = format!("audio.{name}");
        workflow["components"]["audio"]["ports"]["outputs"][name] = contract.clone();
        workflow["steps"][0]["outputs"][name] = json!(value);
        workflow["outputs"][&value] = json!({"contract":contract,"family":{"kind":"materialized"},"role":"tensor","stage":"pre_adapter"});
        workflow["steps"]
            .as_array_mut()
            .unwrap()
            .push(json!({"kind":"emit","value":value,"output":value,"mode":"replace"}));
    }
    workflow["steps"][1]["inputs"]["features"] = json!("audio.spectrum");
    std::fs::write(
        package.path().join("inference_metadata.yaml"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    let mut engine = Engine::from_dir(
        package.path(),
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    for (amplitude, samples) in [(0.25, 5), (0.5, 3)] {
        let bytes =
            onnx_genai_preprocess::audio::encode_wav_pcm16(&vec![amplitude; samples], 16000, 1)
                .unwrap();
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
        let spectrum = outputs.aggregate("audio.spectrum").unwrap();
        assert_eq!(spectrum.shape(), [1, 3, samples as i64]);
        // A constant waveform times periodic Hann(4) has DFT magnitudes
        // [2A, A, 0] in its one-sided bins, independent of reflection/frame.
        for (bin, row) in spectrum
            .to_vec_f32()
            .unwrap()
            .chunks_exact(samples)
            .enumerate()
        {
            let expected = [4.0, 1.0, 0.0][bin] * amplitude * amplitude;
            assert!(
                row.iter().all(|value| (value - expected).abs() < 1e-6),
                "bin {bin}: {row:?} != {expected}"
            );
        }
        assert_eq!(
            outputs
                .aggregate("audio.power_frames")
                .unwrap()
                .to_vec_i64()
                .unwrap(),
            [samples as i64]
        );
        assert_eq!(
            outputs
                .aggregate("audio.power_mask")
                .unwrap()
                .to_vec_i64()
                .unwrap(),
            vec![1; samples]
        );
        let score = outputs.aggregate("score").unwrap().to_vec_f32().unwrap()[0];
        assert!(
            (score - (5.0 / 3.0) * amplitude * amplitude).abs() < 1e-6,
            "{score}"
        );
    }
}
