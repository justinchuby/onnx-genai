//! Actual Native waveform inference and non-generative CTC decoding.
use onnx_genai_engine::{
    Engine, EngineConfig,
    transcription::{PreparedTranscription, TranscriptionLimitError, TranscriptionLimits},
};
use onnx_genai_metadata::transcription::transcription_contract;
use std::{num::NonZeroUsize, path::PathBuf, sync::Arc};
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/transcription_ctc")
}
fn prepared(
    engine: &Engine,
    samples: &[f32],
    limits: TranscriptionLimits,
) -> PreparedTranscription {
    let contract = Arc::new(transcription_contract(engine.metadata()).unwrap().unwrap());
    let bytes = onnx_genai_preprocess::audio::encode_wav_pcm16(samples, 8000, 1).unwrap();
    PreparedTranscription::prepare(contract, bytes.into(), limits).unwrap()
}
#[test]
fn native_ctc_transcribes_waveforms_and_excludes_nonblank_padding() {
    let mut engine = Engine::from_dir(&root(), EngineConfig::default()).unwrap();
    let samples = [-0.5, -0.5, 0.125, -0.5, 0.0, 0.5, 0.5, 0.125, 0.5];
    for (input, text, ids) in [
        (samples.to_vec(), "bb aa", vec![3, 3, 2, 1, 1]),
        (
            samples.iter().map(|v| -v).collect(),
            "aa bb",
            vec![1, 1, 2, 3, 3],
        ),
        (vec![0.125; 5], "", vec![]),
        (vec![], "", vec![]),
    ] {
        let request = prepared(&engine, &input, TranscriptionLimits::default());
        let result = engine.transcribe(request).unwrap();
        assert_eq!(result.text, text);
        assert_eq!(result.token_ids, ids);
        assert_eq!(result.frames, input.len());
    }
}
#[test]
fn native_ctc_limits_fail_with_typed_actionable_errors() {
    let mut engine = Engine::from_dir(&root(), EngineConfig::default()).unwrap();
    let samples = [-0.5, -0.5, 0.125, -0.5, 0.0, 0.5, 0.5, 0.125, 0.5];
    for limits in [
        TranscriptionLimits {
            max_frames: NonZeroUsize::new(8).unwrap(),
            ..Default::default()
        },
        TranscriptionLimits {
            max_tokens: NonZeroUsize::new(1).unwrap(),
            ..Default::default()
        },
        TranscriptionLimits {
            max_text_bytes: NonZeroUsize::new(1).unwrap(),
            ..Default::default()
        },
    ] {
        let request = prepared(&engine, &samples, limits);
        let error = engine.transcribe(request).unwrap_err();
        assert!(
            error.downcast_ref::<TranscriptionLimitError>().is_some(),
            "{error:#}"
        );
        assert!(error.to_string().contains("shorten the audio"));
    }
    let request = prepared(&engine, &samples, TranscriptionLimits::default());
    assert_eq!(engine.transcribe(request).unwrap().text, "bb aa");
}
fn package(change: impl FnOnce(&mut serde_json::Value)) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for name in ["tokenizer.json", "model.onnx.textproto"] {
        std::fs::copy(root().join(name), dir.path().join(name)).unwrap();
    }
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root().join("inference_metadata.yaml")).unwrap())
            .unwrap();
    change(&mut metadata);
    std::fs::write(
        dir.path().join("inference_metadata.yaml"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    dir
}
#[test]
fn native_transcription_resolves_tokenizer_classes_and_greedy_decoding_from_its_own_package() {
    let samples = [-0.5, -0.5, 0.125, -0.5, 0.0, 0.5, 0.5, 0.125, 0.5];
    for (mode, expected) in [
        ("tokenizer", "<eos><eos><unk><unk>"),
        ("tokenizer_inferred", "<eos><eos><unk><unk>"),
        ("greedy", "bbb aaa"),
    ] {
        let dir = package(|metadata| {
            let spec = &mut metadata["profiles"]["transcription"]["decoding"];
            if mode.starts_with("tokenizer") {
                spec["vocabulary"] =
                    serde_json::json!({"source":"tokenizer","size":4,"ignored_tokens":["<bos>"]});
                if mode == "tokenizer_inferred" {
                    spec["vocabulary"].as_object_mut().unwrap().remove("size");
                }
            } else {
                spec["kind"] = serde_json::json!("greedy_argmax");
                spec["collapse_repeats"] = serde_json::json!(false);
                spec.as_object_mut().unwrap().remove("blank_id");
                spec["vocabulary"]["ignored_tokens"] = serde_json::json!(["<blank>"]);
            }
        });
        let mut engine = Engine::from_dir(dir.path(), EngineConfig::default()).unwrap();
        let request = prepared(&engine, &samples, TranscriptionLimits::default());
        assert_eq!(engine.transcribe(request).unwrap().text, expected, "{mode}");
    }
}
#[test]
fn native_transcription_rejects_negative_or_overreported_valid_lengths() {
    for bad in [-1, 13] {
        let dir = package(|metadata| {
            metadata["pipeline"]["workflow"]["steps"][1]["outputs"]["bad_length"] =
                serde_json::json!("bad_length");
            metadata["pipeline"]["workflow"]["steps"][3]["value"] = serde_json::json!("bad_length");
        });
        let graph_path = dir.path().join("model.onnx.textproto");
        let graph = std::fs::read_to_string(&graph_path).unwrap();
        let end = graph.rfind('}').unwrap();
        let extra = format!(
            "\nnode {{ input: \"invalid_length\" output: \"bad_length\" op_type: \"Identity\" }}\ninitializer {{ name: \"invalid_length\" data_type: 7 dims: 1 int64_data: {bad} }}\noutput {{ name: \"bad_length\" type {{ tensor_type {{ elem_type: 7 shape {{ dim {{ dim_value: 1 }} }} }} }} }}\n"
        );
        std::fs::write(
            graph_path,
            format!("{}{}{}", &graph[..end], extra, &graph[end..]),
        )
        .unwrap();
        let mut engine = Engine::from_dir(dir.path(), EngineConfig::default()).unwrap();
        let request = prepared(&engine, &[0.5; 5], TranscriptionLimits::default());
        let error = engine.transcribe(request).unwrap_err();
        assert!(
            error.to_string().contains("transcription lengths")
                && error.to_string().contains(&bad.to_string()),
            "{error:#}"
        );
    }
}

#[test]
fn native_transposed_logits_obey_outer_axis_validity_and_reject_class_disagreement() {
    for disagree in [false, true] {
        let dir = package(|metadata| {
            let workflow = &mut metadata["pipeline"]["workflow"];
            workflow["outputs"]["logits"]["contract"]["shape"] =
                serde_json::json!([1, 4, "frames"]);
            workflow["outputs"]["frame_lengths"]["contract"]["shape"] = serde_json::json!([1, 4]);
            workflow["steps"][1]["inputs"]["length"] = serde_json::json!("length");
            workflow["steps"][1]["outputs"]["class_lengths"] = serde_json::json!("class_lengths");
            workflow["steps"][3]["value"] = serde_json::json!("class_lengths");
            let decoding = &mut metadata["profiles"]["transcription"]["decoding"];
            decoding["time_axis"] = serde_json::json!(2);
            decoding["class_axis"] = serde_json::json!(1);
        });
        let path = dir.path().join("model.onnx.textproto");
        let graph = std::fs::read_to_string(&path).unwrap();
        let original = "output { name: \"logits\" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 12 } dim { dim_value: 4 } } } } }";
        assert!(graph.contains(original));
        let graph=graph.replace("output: \"logits\" op_type: \"Concat\"","output: \"native_logits\" op_type: \"Concat\"")
            .replace(original,"output { name: \"logits\" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 4 } dim { dim_value: 12 } } } } }");
        let end = graph.rfind('}').unwrap();
        let mut extra = String::from(
            r#"
node { input: "native_logits" output: "logits" op_type: "Transpose" attribute { name: "perm" ints: 0 ints: 2 ints: 1 type: INTS } }
input { name: "length" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } } } } }
output { name: "class_lengths" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 4 } } } } }
"#,
        );
        extra.push_str(if disagree {r#"
initializer { name: "invalid_lengths" data_type: 7 dims: 1 dims: 4 int64_data: 9 int64_data: 8 int64_data: 9 int64_data: 9 }
node { input: "invalid_lengths" output: "class_lengths" op_type: "Identity" }
"#}else{r#"
initializer { name: "length_axis" data_type: 7 dims: 1 int64_data: 1 }
initializer { name: "length_shape" data_type: 7 dims: 2 int64_data: 1 int64_data: 4 }
node { input: "length" input: "length_axis" output: "length.row" op_type: "Unsqueeze" }
node { input: "length.row" input: "length_shape" output: "class_lengths" op_type: "Expand" }
"#});
        std::fs::write(path, format!("{}{}{}", &graph[..end], extra, &graph[end..])).unwrap();
        let mut engine = Engine::from_dir(dir.path(), EngineConfig::default()).unwrap();
        let request = prepared(
            &engine,
            &[-0.5, -0.5, 0.125, -0.5, 0.0, 0.5, 0.5, 0.125, 0.5],
            TranscriptionLimits::default(),
        );
        let result = engine.transcribe(request);
        if disagree {
            let error = result.unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("disagree across class positions"),
                "{error:#}"
            );
        } else {
            let result = result.unwrap();
            assert_eq!(result.text, "bb aa");
            assert_eq!(result.frames, 9);
        }
    }
}
