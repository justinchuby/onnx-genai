//! Real saved floating-point SafeTensors PEFT factors through Native ONNX and the existing
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
    let error = Engine::from_dir(
        root.path(),
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cpu),
            ..Default::default()
        },
    )
    .err()
    .expect("baked PEFT must be rejected during metadata admission");
    let text = format!("{error:#}");
    assert!(
        text.contains("adapters.artifacts.peft.weights[0].scale_encoding")
            && text.contains("must be alpha_over_rank for hf_peft"),
        "{text}"
    );
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
    assert!(
        format!("{error:#}").contains("storage dtype Float16")
            && format!("{error:#}").contains("declared dtype Float32"),
        "{error:#}"
    );
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

#[test]
fn native_peft_malformed_custom_config_names_the_source_and_adapter() {
    let root = package();
    let metadata_path = root.path().join("inference_metadata.yaml");
    let mut metadata: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata_path).unwrap()).unwrap();
    let custom = "adapters/peft/custom-linear-config.json";
    metadata["adapters"]["artifacts"]["peft"]["weights"][0]["config_location"] =
        serde_yaml::Value::String(custom.into());
    std::fs::write(metadata_path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
    for contents in ["{malformed", "[]"] {
        std::fs::write(root.path().join(custom), contents).unwrap();
        let mut native = engine(root.path());
        let error = run(&mut native, Some(1.0)).unwrap_err();
        let text = format!("{error:#}");
        assert!(
            text.contains(custom) && text.contains("peft@1") && text.contains("JSON"),
            "{text}"
        );
        assert_eq!(run(&mut native, None).unwrap(), [4.0, 15.0]);
    }
}

#[test]
fn native_peft_rank_two_preserves_independent_row_compositions() {
    let root = package();
    let metadata_path = root.path().join("inference_metadata.yaml");
    let mut metadata: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata_path).unwrap()).unwrap();
    let capacity = json!({"budgets":[{"dimensions":["batch"],"max_total":2}]});
    metadata["pipeline"]["workflow"]["components"]["decoder"]["batch_capacity"] =
        serde_yaml::to_value(&capacity).unwrap();
    metadata["pipeline"]["workflow"]["components"]["overlay"]["batch_capacity"] =
        serde_yaml::to_value(&capacity).unwrap();
    let port = json!({"dtype":"float32","shape":["batch",2],"batch_layout":{"kind":"request_aligned","axis":0}});
    metadata["pipeline"]["workflow"]["components"]["decoder"]["ports"] = serde_yaml::to_value(
        json!({"inputs":{"activations":port.clone()},"outputs":{"projection.output":port}}),
    )
    .unwrap();
    metadata["adapters"]["artifacts"]["peft"]["rank"] = serde_yaml::to_value(2).unwrap();
    metadata["adapters"]["artifacts"]["peft"]["alpha"] = serde_yaml::to_value(4.0).unwrap();
    metadata["adapters"]["target_manifest"]["targets"][0]["rank"] =
        serde_yaml::to_value(2).unwrap();
    metadata["adapters"]["target_manifest"]["targets"][0]["alpha"] =
        serde_yaml::to_value(4.0).unwrap();
    std::fs::write(metadata_path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
    let mut cfg = config();
    cfg["r"] = json!(2);
    cfg["lora_alpha"] = json!(4);
    std::fs::write(config_path(root.path()), serde_json::to_vec(&cfg).unwrap()).unwrap();
    let header = serde_json::to_vec(&json!({
        format!("{MODULE}.lora_A.weight"):{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
        format!("{MODULE}.lora_B.weight"):{"dtype":"F32","shape":[2,2],"data_offsets":[16,32]}
    }))
    .unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    for value in [1_f32, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 4.0] {
        bytes.extend(value.to_le_bytes());
    }
    std::fs::write(tensor_path(root.path()), bytes).unwrap();
    let selection = AdapterSelection::default()
        .with_row([AdapterActivation::new("peft", 0.5)])
        .with_row([AdapterActivation::new("peft", -0.25)]);
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: GenerateOptions {
            adapter_selection: Some(selection),
            ..Default::default()
        },
    })
    .with_input(
        "activations",
        Value::from_slice_f32(&[2.0, 5.0, 7.0, 11.0], &[2, 2]).unwrap(),
    );
    let mut native = engine(root.path());
    assert_eq!(
        native.run_pipeline(request).unwrap()["result"]
            .to_vec_f32()
            .unwrap(),
        [10.0, 35.0, 3.5, 11.0]
    );
    assert_eq!(run(&mut native, None).unwrap(), [4.0, 15.0]);
}
#[test]
fn native_peft_governor_denies_staging_before_safetensors_decode_and_recovers() {
    let root = package();
    let original = std::fs::read(tensor_path(root.path())).unwrap();
    let old_header = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    // Whitespace is valid SafeTensors header padding: retain the same actual
    // factors while making the required governed staging exceed the host cap.
    let padded_header = 8 * 1024 * 1024;
    let file = std::fs::File::create(tensor_path(root.path())).unwrap();
    use std::io::Write;
    let mut writer = std::io::BufWriter::new(file);
    writer
        .write_all(&(padded_header as u64).to_le_bytes())
        .unwrap();
    writer.write_all(&original[8..8 + old_header]).unwrap();
    let padding = vec![b' '; padded_header - old_header];
    writer.write_all(&padding).unwrap();
    writer.write_all(&original[8 + old_header..]).unwrap();
    writer.flush().unwrap();
    drop(writer);
    let mut native = Engine::from_dir(
        root.path(),
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cpu),
            limits: onnx_genai_engine::ResourceLimits {
                host_ram_limit: onnx_genai_engine::ResourceLimit::Bytes(64 * 1024 * 1024),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(run(&mut native, None).unwrap(), [4.0, 15.0]);
    let error = run(&mut native, Some(1.0)).unwrap_err();
    let text = format!("{error:#}");
    assert!(
        text.contains("reserve adapter container parsing memory"),
        "{text}"
    );
    assert!(
        !text.contains("parse HF PEFT safetensors"),
        "container must be rejected before decode: {text}"
    );
    assert_eq!(run(&mut native, None).unwrap(), [4.0, 15.0]);
    std::fs::write(tensor_path(root.path()), original).unwrap();
    assert_eq!(run(&mut native, Some(0.5)).unwrap(), [40.0, 63.0]);
}

fn raw_factors(dtype_a: &str, dtype_b: &str, rank: usize, a: &[u8], b: &[u8]) -> Vec<u8> {
    let header=serde_json::to_vec(&json!({
        format!("{MODULE}.lora_A.weight"):{"dtype":dtype_a,"shape":[rank,2],"data_offsets":[0,a.len()]},
        format!("{MODULE}.lora_B.weight"):{"dtype":dtype_b,"shape":[2,rank],"data_offsets":[a.len(),a.len()+b.len()]}
    })).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(a);
    bytes.extend(b);
    bytes
}
fn half_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|bits| bits.to_le_bytes()).collect()
}
fn half_package(dtype: &str, a: &[u16], b: &[u16]) -> tempfile::TempDir {
    let root = package();
    let path = root.path().join("inference_metadata.yaml");
    let mut metadata: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    metadata["adapters"]["artifacts"]["peft"]["dtype"] = serde_yaml::Value::String(
        if dtype == "F16" {
            "float16"
        } else {
            "bfloat16"
        }
        .into(),
    );
    metadata["adapters"]["artifacts"]["peft"]["rank"] = serde_yaml::to_value(2).unwrap();
    metadata["adapters"]["artifacts"]["peft"]["alpha"] = serde_yaml::to_value(4.0).unwrap();
    metadata["adapters"]["target_manifest"]["targets"][0]["rank"] =
        serde_yaml::to_value(2).unwrap();
    metadata["adapters"]["target_manifest"]["targets"][0]["alpha"] =
        serde_yaml::to_value(4.0).unwrap();
    let capacity = json!({"budgets":[{"dimensions":["batch"],"max_total":2}]});
    for component in ["decoder", "overlay"] {
        metadata["pipeline"]["workflow"]["components"][component]["batch_capacity"] =
            serde_yaml::to_value(&capacity).unwrap();
    }
    let port = json!({"dtype":"float32","shape":["batch",2],"batch_layout":{"kind":"request_aligned","axis":0}});
    metadata["pipeline"]["workflow"]["components"]["decoder"]["ports"] = serde_yaml::to_value(
        json!({"inputs":{"activations":port.clone()},"outputs":{"projection.output":port}}),
    )
    .unwrap();
    std::fs::write(path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
    let mut cfg = config();
    cfg["r"] = json!(2);
    cfg["lora_alpha"] = json!(4);
    std::fs::write(config_path(root.path()), serde_json::to_vec(&cfg).unwrap()).unwrap();
    std::fs::write(
        tensor_path(root.path()),
        raw_factors(dtype, dtype, 2, &half_bytes(a), &half_bytes(b)),
    )
    .unwrap();
    root
}
fn run_rows(
    engine: &mut Engine,
    rows: Vec<Vec<AdapterActivation>>,
    values: &[f32],
) -> anyhow::Result<Vec<f32>> {
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: GenerateOptions {
            adapter_selection: Some(AdapterSelection { rows }),
            ..Default::default()
        },
    })
    .with_input(
        "activations",
        Value::from_slice_f32(values, &[(values.len() / 2) as i64, 2])?,
    );
    Ok(engine.run_pipeline(request)?["result"].to_vec_f32()?)
}
#[test]
fn native_peft_f16_bf16_rank_two_storage_preserves_f32_computation_and_row_scales() {
    // Hand-authored IEEE storage bits; no half conversion library in the oracle.
    // F16 A[0,0]=1365/4096; BF16 A[0,0]=171/512. Other A values=-1,1/2,2;
    // B=[[1,1/2],[-1/2,1]]. alpha/r=2. Exact dyadic results below include base diag[2,3].
    for (dtype, a, b, positive, negative) in [
        (
            "F16",
            [0x3555, 0xbc00, 0x3800, 0x4000],
            [0x3c00, 0x3800, 0xb800, 0x3c00],
            [10581.0 / 2048.0, 115371.0 / 4096.0],
            [97965.0 / 8192.0, 296275.0 / 16384.0],
        ),
        (
            "BF16",
            [0x3eab, 0xbf80, 0x3f00, 0x4000],
            [0x3f80, 0x3f00, 0xbf00, 0x3f80],
            [1323.0 / 256.0, 14421.0 / 512.0],
            [12243.0 / 1024.0, 37037.0 / 2048.0],
        ),
    ] {
        let root = half_package(dtype, &a, &b);
        let mut native = engine(root.path());
        assert_eq!(
            run_rows(
                &mut native,
                vec![
                    vec![AdapterActivation::new("peft", 0.5)],
                    vec![AdapterActivation::new("peft", -0.25)]
                ],
                &[2.0, 5.0, 7.0, 11.0]
            )
            .unwrap(),
            [positive[0], positive[1], negative[0], negative[1]],
            "{dtype}"
        );
        assert_eq!(
            run_rows(
                &mut native,
                vec![vec![AdapterActivation::new("peft", 0.0)], vec![]],
                &[2.0, 5.0, 7.0, 11.0]
            )
            .unwrap(),
            [4.0, 15.0, 14.0, 33.0],
            "{dtype}: zero and absent selection retain F32 base"
        );
        assert_eq!(run(&mut native, None).unwrap(), [4.0, 15.0]);
    }
}
#[test]
fn native_peft_half_dtype_and_nonfinite_errors_leave_cached_valid_adapters_usable() {
    for (dtype, a, b, invalid, expected) in [
        (
            "F16",
            [0x3555, 0xbc00, 0x3800, 0x4000],
            [0x3c00, 0x3800, 0xb800, 0x3c00],
            [0x7c00, 0xfc00, 0x7c01],
            [10581.0 / 2048.0, 115371.0 / 4096.0],
        ),
        (
            "BF16",
            [0x3eab, 0xbf80, 0x3f00, 0x4000],
            [0x3f80, 0x3f00, 0xbf00, 0x3f80],
            [0x7f80, 0xff80, 0x7f81],
            [1323.0 / 256.0, 14421.0 / 512.0],
        ),
    ] {
        let root = half_package(dtype, &a, &b);
        let path = root.path().join("inference_metadata.yaml");
        let mut metadata: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let mut bad = metadata["adapters"]["artifacts"]["peft"].clone();
        bad["index"] = serde_yaml::to_value(1).unwrap();
        bad["identity"] = serde_yaml::Value::String("bad".into());
        bad["weights"][0]["location"] =
            serde_yaml::Value::String("adapters/bad/adapter_model.safetensors".into());
        bad["weights"][0]["config_location"] =
            serde_yaml::Value::String("adapters/bad/adapter_config.json".into());
        metadata["adapters"]["artifacts"]["bad"] = bad;
        std::fs::write(path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
        std::fs::create_dir(root.path().join("adapters/bad")).unwrap();
        std::fs::copy(
            config_path(root.path()),
            root.path().join("adapters/bad/adapter_config.json"),
        )
        .unwrap();
        let bad_path = root.path().join("adapters/bad/adapter_model.safetensors");
        let fp32 = [1_f32, 0.5, -0.5, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let fp64 = [1_f64, 0.5, -0.5, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let other = if dtype == "F16" { "BF16" } else { "F16" };
        let mut failures = vec![
            (
                raw_factors(other, other, 2, &half_bytes(&a), &half_bytes(&b)),
                "storage dtype",
            ),
            (
                raw_factors(dtype, "F32", 2, &half_bytes(&a), &fp32),
                "storage dtype",
            ),
            (raw_factors("F64", "F64", 2, &fp64, &fp64), "storage dtype"),
        ];
        for bits in invalid {
            let mut broken = a;
            broken[0] = bits;
            failures.push((
                raw_factors(dtype, dtype, 2, &half_bytes(&broken), &half_bytes(&b)),
                "nonfinite",
            ));
        }
        let mut native = engine(root.path());
        assert_eq!(
            run(&mut native, Some(0.5)).unwrap(),
            expected,
            "{dtype}: preload valid cache"
        );
        for (bytes, message) in failures {
            std::fs::write(&bad_path, bytes).unwrap();
            let error = run_rows(
                &mut native,
                vec![vec![
                    AdapterActivation::new("peft", 0.5),
                    AdapterActivation::new("bad", 1.0),
                ]],
                &[2.0, 5.0],
            )
            .unwrap_err();
            let text = format!("{error:#}");
            assert!(text.contains(message), "{dtype}: {text}");
            assert_eq!(
                run(&mut native, Some(0.5)).unwrap(),
                expected,
                "{dtype}: invalid mixed selection cannot contaminate cached valid overlay"
            );
            assert_eq!(run(&mut native, None).unwrap(), [4.0, 15.0]);
        }
        std::fs::copy(tensor_path(root.path()), bad_path).unwrap();
        assert_eq!(
            run_rows(
                &mut native,
                vec![vec![AdapterActivation::new("bad", 0.5)]],
                &[2.0, 5.0]
            )
            .unwrap(),
            expected,
            "{dtype}: failed load is retryable after a valid file is supplied"
        );
    }
}
#[test]
fn native_peft_half_staging_denial_precedes_conversion_and_recovers() {
    for (dtype, a, b, expected) in [
        (
            "F16",
            [0x3555, 0xbc00, 0x3800, 0x4000],
            [0x3c00, 0x3800, 0xb800, 0x3c00],
            [10581.0 / 2048.0, 115371.0 / 4096.0],
        ),
        (
            "BF16",
            [0x3eab, 0xbf80, 0x3f00, 0x4000],
            [0x3f80, 0x3f00, 0xbf00, 0x3f80],
            [1323.0 / 256.0, 14421.0 / 512.0],
        ),
    ] {
        let root = half_package(dtype, &a, &b);
        let original = std::fs::read(tensor_path(root.path())).unwrap();
        let length = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
        let mut padded = (8 * 1024 * 1024_u64).to_le_bytes().to_vec();
        padded.extend(&original[8..8 + length]);
        padded.resize(8 + 8 * 1024 * 1024, b' ');
        padded.extend(&original[8 + length..]);
        std::fs::write(tensor_path(root.path()), padded).unwrap();
        let mut native = Engine::from_dir(
            root.path(),
            EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(NativeDecodeDevice::Cpu),
                limits: onnx_genai_engine::ResourceLimits {
                    host_ram_limit: onnx_genai_engine::ResourceLimit::Bytes(64 * 1024 * 1024),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(run(&mut native, None).unwrap(), [4.0, 15.0]);
        let error = run(&mut native, Some(0.5)).unwrap_err();
        assert!(
            format!("{error:#}").contains("reserve adapter container parsing memory"),
            "{dtype}: {error:#}"
        );
        assert_eq!(run(&mut native, None).unwrap(), [4.0, 15.0]);
        std::fs::write(tensor_path(root.path()), original).unwrap();
        assert_eq!(
            run(&mut native, Some(0.5)).unwrap(),
            expected,
            "{dtype}: valid inference recovers after pre-conversion budget refusal"
        );
    }
}

#[test]
fn native_portable_json_requires_f32_storage_instead_of_guessing_half_from_numbers() {
    for dtype in ["float16", "bfloat16"] {
        let root = package();
        let path = root.path().join("inference_metadata.yaml");
        let mut metadata: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        metadata["adapters"]["artifacts"]["peft"]["dtype"] =
            serde_yaml::Value::String(dtype.into());
        metadata["adapters"]["artifacts"]["peft"]["weights"]=serde_yaml::to_value(json!([{"format":"json","location":"adapters/peft/adapter.json","loader_capability":"onnx-genai.adapters.json@1","scale_encoding":"alpha_over_rank"}])).unwrap();
        std::fs::write(&path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
        std::fs::write(
            root.path().join("adapters/peft/adapter.json"),
            serde_json::to_vec(&json!({"targets":{(MODULE):{"a":[1.0,2.0],"b":[3.0,4.0]}}}))
                .unwrap(),
        )
        .unwrap();
        let mut native = engine(root.path());
        let error = run(&mut native, Some(0.5)).unwrap_err();
        assert!(
            format!("{error:#}").contains("JSON adapter normalization requires float32"),
            "{dtype}: {error:#}"
        );
        assert_eq!(run(&mut native, None).unwrap(), [4.0, 15.0]);
        metadata["adapters"]["artifacts"]["peft"]["dtype"] =
            serde_yaml::Value::String("float32".into());
        std::fs::write(path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
        let mut repaired = engine(root.path());
        assert_eq!(run(&mut repaired, Some(0.5)).unwrap(), [40.0, 63.0]);
    }
}
