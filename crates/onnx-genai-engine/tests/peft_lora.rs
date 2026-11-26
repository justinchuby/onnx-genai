//! Real saved F32 SafeTensors PEFT factors through Native ONNX and the existing
//! parameter-overlay target manifest. No Python/PEFT dependency or mock executor.
#![cfg(feature = "native-backend")]
use onnx_genai_engine::{
    AdapterActivation, AdapterSelection, Engine, EngineConfig, EngineDecodeBackend,
    GenerateOptions, GeneratePrompt, GenerateRequest, NativeDecodeDevice, PipelineGenerateRequest,
};
use onnx_genai_ort::Value;
use serde_json::json;
use std::path::{Path, PathBuf};

const MODULE: &str = "base_model.model.explicit.projection";
fn config() -> serde_json::Value {
    json!({"peft_type":"LORA","r":1,"lora_alpha":2,"target_modules":["projection"],"bias":"none","fan_in_fan_out":false,"use_dora":false,"use_rslora":false,"lora_dropout":0.1,"inference_mode":false,"init_lora_weights":true,"rank_pattern":{},"alpha_pattern":{},"modules_to_save":null,"qalora_group_size":16,"use_qalora":false,"ensure_weight_tying":false,"alora_invocation_tokens":null,"arrow_config":null})
}
fn saved_tensors(a: &[f32], b: &[f32], dtype: &str, extra: bool) -> Vec<u8> {
    let mut header = json!({
        format!("{MODULE}.lora_A.weight"):{"dtype":dtype,"shape":[1,2],"data_offsets":[0,a.len()*4]},
        format!("{MODULE}.lora_B.weight"):{"dtype":dtype,"shape":[2,1],"data_offsets":[a.len()*4,(a.len()+b.len())*4]},
    });
    if extra {
        header["unmapped"] = json!({"dtype":"F32","shape":[0],"data_offsets":[(a.len()+b.len())*4,(a.len()+b.len())*4]});
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    for n in a.iter().chain(b) {
        bytes.extend(n.to_le_bytes());
    }
    bytes
}
fn package() -> tempfile::TempDir {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/adapter");
    let temp = tempfile::tempdir().unwrap();
    std::fs::copy(
        root.join("model.onnx.textproto"),
        temp.path().join("model.onnx.textproto"),
    )
    .unwrap();
    let graph_path = temp.path().join("model.onnx.textproto");
    let graph = std::fs::read_to_string(&graph_path).unwrap();
    let raw = graph
        .lines()
        .find(|line| line.trim_start().starts_with("raw_data:"))
        .unwrap();
    std::fs::write(
        graph_path,
        graph.replace(
            raw,
            "    float_data: 2.0 float_data: 0.0 float_data: 0.0 float_data: 3.0",
        ),
    )
    .unwrap();
    let mut metadata: serde_yaml::Value = serde_yaml::from_str(
        &std::fs::read_to_string(root.join("inference_metadata.yaml")).unwrap(),
    )
    .unwrap();
    metadata["pipeline"]["workflow"]["steps"] = serde_yaml::to_value(json!([
        {"kind":"invoke","component":"decoder","inputs":{"activations":"activations"},"outputs":{"projection.output":"projected"}},
        {"kind":"invoke","component":"overlay","inputs":{"input":"activations","base":"projected"},"outputs":{"output":"adapted"}},
        {"kind":"emit","value":"adapted","output":"result","mode":"replace"}
    ])).unwrap();
    let mut artifact = metadata["adapters"]["artifacts"]["peft"].clone();
    artifact["index"] = serde_yaml::to_value(0).unwrap();
    artifact["alpha"] = serde_yaml::to_value(2.0).unwrap();
    artifact["weights"]=serde_yaml::to_value(json!([{"location":"adapters/peft/adapter_model.safetensors","loader_capability":"onnx-genai.adapters.hf-peft@1","config_location":"adapters/peft/adapter_config.json","scale_encoding":"alpha_over_rank","format":"hf_peft"}])).unwrap();
    artifact["bindings"][0]["weight_key"] = serde_yaml::Value::String(MODULE.into());
    metadata["adapters"]["artifacts"] = serde_yaml::Value::Mapping(serde_yaml::Mapping::new());
    metadata["adapters"]["artifacts"]["peft"] = artifact;
    metadata["adapters"]["target_manifest"]["targets"][0]["alpha"] =
        serde_yaml::to_value(2.0).unwrap();
    std::fs::create_dir_all(temp.path().join("adapters/peft")).unwrap();
    std::fs::write(
        temp.path().join("inference_metadata.yaml"),
        serde_yaml::to_string(&metadata).unwrap(),
    )
    .unwrap();
    std::fs::write(
        config_path(temp.path()),
        serde_json::to_vec(&config()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        tensor_path(temp.path()),
        saved_tensors(&[1.0, 2.0], &[3.0, 4.0], "F32", false),
    )
    .unwrap();
    temp
}
fn config_path(root: &Path) -> PathBuf {
    root.join("adapters/peft/adapter_config.json")
}
fn tensor_path(root: &Path) -> PathBuf {
    root.join("adapters/peft/adapter_model.safetensors")
}
fn engine(root: &Path) -> Engine {
    Engine::from_dir(
        root,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cpu),
            ..Default::default()
        },
    )
    .unwrap()
}
fn run(engine: &mut Engine, scale: Option<f32>) -> anyhow::Result<Vec<f32>> {
    let selection = AdapterSelection::default().with_row(
        scale
            .into_iter()
            .map(|scale| AdapterActivation::new("peft", scale)),
    );
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: GenerateOptions {
            adapter_selection: Some(selection),
            ..Default::default()
        },
    })
    .with_input("activations", Value::from_slice_f32(&[2.0, 5.0], &[1, 2])?);
    Ok(engine.run_pipeline(request)?["result"].to_vec_f32()?)
}
#[test]
fn native_peft_safetensors_use_explicit_module_targets_and_alpha_over_rank_scale() {
    let root = package();
    let mut engine = engine(root.path());
    assert_eq!(run(&mut engine, None).unwrap(), [4.0, 15.0]);
    // x A^T = 12; B*[12]=[36,48]; alpha/r=2, then caller scale.
    for (scale, expected) in [
        (1.0, [76.0, 111.0]),
        (0.5, [40.0, 63.0]),
        (-0.25, [-14.0, -9.0]),
        (0.0, [4.0, 15.0]),
    ] {
        assert_eq!(
            run(&mut engine, Some(scale)).unwrap(),
            expected,
            "scale={scale}"
        );
    }
    assert_eq!(run(&mut engine, None).unwrap(), [4.0, 15.0]);
}
#[test]
fn native_peft_config_changes_cannot_hide_behind_a_loaded_weights_cache() {
    let root = package();
    let mut engine = engine(root.path());
    assert_eq!(run(&mut engine, Some(0.5)).unwrap(), [40.0, 63.0]);
    let mut changed = config();
    changed["lora_alpha"] = json!(4);
    std::fs::write(
        config_path(root.path()),
        serde_json::to_vec(&changed).unwrap(),
    )
    .unwrap();
    let error = run(&mut engine, Some(1.0)).unwrap_err();
    assert!(
        format!("{error:#}").contains("configuration changed"),
        "{error:#}"
    );
    assert_eq!(
        run(&mut engine, None).unwrap(),
        [4.0, 15.0],
        "failed selection cannot apply an overlay to base"
    );
    std::fs::write(
        config_path(root.path()),
        serde_json::to_vec(&config()).unwrap(),
    )
    .unwrap();
    assert_eq!(
        run(&mut engine, Some(0.5)).unwrap(),
        [40.0, 63.0],
        "restoring immutable config retains the prior valid factors"
    );
}
#[test]
fn native_peft_rejects_variants_rank_mismatch_and_unmapped_tensors_without_partial_overlay() {
    for (field, value) in [
        ("use_dora", json!(true)),
        ("use_rslora", json!(true)),
        ("rank_pattern", json!({"projection":2})),
        ("r", json!(2)),
        ("lora_alpha", json!(3)),
        ("modules_to_save", json!(["head"])),
        ("init_lora_weights", json!("pissa")),
        ("unknown_numeric_semantics", json!(7)),
    ] {
        let root = package();
        let mut changed = config();
        changed[field] = value;
        std::fs::write(
            config_path(root.path()),
            serde_json::to_vec(&changed).unwrap(),
        )
        .unwrap();
        let mut engine = engine(root.path());
        let error = run(&mut engine, Some(1.0)).unwrap_err();
        let text = format!("{error:#}");
        assert!(text.contains("PEFT"), "{field}: {text}");
        assert_eq!(
            run(&mut engine, None).unwrap(),
            [4.0, 15.0],
            "{field}: failed admission cannot modify base"
        );
        std::fs::write(
            config_path(root.path()),
            serde_json::to_vec(&config()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            run(&mut engine, Some(0.5)).unwrap(),
            [40.0, 63.0],
            "{field}: subsequent valid selection recovers"
        );
    }
    let root = package();
    let metadata_path = root.path().join("inference_metadata.yaml");
    let mut metadata: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata_path).unwrap()).unwrap();
    metadata["adapters"]["artifacts"]["peft"]["weights"][0]["scale_encoding"] =
        serde_yaml::Value::String("baked".into());
    std::fs::write(metadata_path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
    let mut native = engine(root.path());
    let error = run(&mut native, Some(1.0)).unwrap_err();
    assert!(
        format!("{error:#}").contains("require scale_encoding alpha_over_rank"),
        "{error:#}"
    );
    assert_eq!(run(&mut native, None).unwrap(), [4.0, 15.0]);
    let f16_header = serde_json::to_vec(&json!({
        format!("{MODULE}.lora_A.weight"):{"dtype":"F16","shape":[1,2],"data_offsets":[0,4]},
        format!("{MODULE}.lora_B.weight"):{"dtype":"F16","shape":[2,1],"data_offsets":[4,8]}
    }))
    .unwrap();
    let mut f16 = (f16_header.len() as u64).to_le_bytes().to_vec();
    f16.extend(f16_header);
    f16.extend([0_u8; 8]);
    let root = package();
    std::fs::write(tensor_path(root.path()), f16).unwrap();
    let mut native = engine(root.path());
    let error = run(&mut native, Some(1.0)).unwrap_err();
    assert!(format!("{error:#}").contains("export F32"), "{error:#}");
    assert_eq!(run(&mut native, None).unwrap(), [4.0, 15.0]);
    for bytes in [
        saved_tensors(&[1.0, 2.0], &[3.0, 4.0], "F32", true),
        saved_tensors(&[f32::NAN, 2.0], &[3.0, 4.0], "F32", false),
        vec![0; 8],
    ] {
        let root = package();
        std::fs::write(tensor_path(root.path()), bytes).unwrap();
        let mut engine = engine(root.path());
        assert!(run(&mut engine, Some(1.0)).is_err());
        assert_eq!(run(&mut engine, None).unwrap(), [4.0, 15.0]);
    }
}
