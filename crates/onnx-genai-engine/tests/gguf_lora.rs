//! Real F32 GGUF LoRA through a non-square Native ONNX projection.
#![cfg(feature = "native-backend")]
use onnx_genai_engine::{
    AdapterActivation, AdapterSelection, Engine, EngineConfig, EngineDecodeBackend,
    GenerateOptions, GeneratePrompt, GenerateRequest, NativeDecodeDevice, PipelineGenerateRequest,
};
use onnx_genai_ort::Value;
use serde_json::json;
use std::path::{Path, PathBuf};
mod wire {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/build_gguf_lora.rs"
    ));
}
const PREFIX: &str = "arbitrary.saved.parameter";
fn facts() -> Vec<wire::Fact> {
    vec![
        ("general.type".into(), 8, wire::string("adapter")),
        ("adapter.type".into(), 8, wire::string("lora")),
        ("adapter.lora.alpha".into(), 6, 4f32.to_le_bytes().to_vec()),
        (
            "general.author".into(),
            8,
            wire::string("fixture provenance"),
        ),
        ("general.architecture".into(), 8, wire::string("llama")),
    ]
}
fn tensors() -> Vec<wire::Tensor> {
    vec![
        (
            format!("{PREFIX}.lora_a"),
            0,
            vec![2, 3],
            [1f32, -1., 2., 0.5, 2., -1.]
                .iter()
                .flat_map(|n| n.to_le_bytes())
                .collect(),
        ),
        (
            format!("{PREFIX}.lora_b"),
            0,
            vec![2, 2],
            [1f32, 0.5, -0.5, 1.]
                .iter()
                .flat_map(|n| n.to_le_bytes())
                .collect(),
        ),
    ]
}
fn path(root: &Path) -> PathBuf {
    root.join("adapters/gguf/adapter.gguf")
}
fn package() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/adapter/inference_metadata.yaml");
    let mut doc: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(source).unwrap()).unwrap();
    let mut artifact = doc["adapters"]["artifacts"]["peft"].clone();
    artifact["index"] = serde_yaml::to_value(0).unwrap();
    artifact["identity"] = serde_yaml::Value::String("gguf".into());
    artifact["rank"] = serde_yaml::to_value(2).unwrap();
    artifact["alpha"] = serde_yaml::to_value(4.0).unwrap();
    artifact["weights"]=serde_yaml::to_value(json!([{"location":"adapters/gguf/adapter.gguf","loader_capability":"onnx-genai.adapters.gguf@1","format":"gguf","scale_encoding":"alpha_over_rank"}])).unwrap();
    artifact["bindings"][0]["weight_key"] = serde_yaml::Value::String(PREFIX.into());
    doc["adapters"]["artifacts"] = serde_yaml::to_value(json!({})).unwrap();
    doc["adapters"]["artifacts"]["gguf"] = artifact;
    doc["adapters"]["target_manifest"]["targets"][0]["input_features"] =
        serde_yaml::to_value(3).unwrap();
    doc["adapters"]["target_manifest"]["targets"][0]["rank"] = serde_yaml::to_value(2).unwrap();
    doc["adapters"]["target_manifest"]["targets"][0]["alpha"] = serde_yaml::to_value(4.0).unwrap();
    doc["schema_version"] = serde_yaml::Value::String("1.19".into());
    doc["pipeline"]["workflow"]["publication_mode"] =
        serde_yaml::Value::String("commit_only".into());
    doc["pipeline"]["workflow"]["outputs"]["result"]["family"] =
        serde_yaml::to_value(json!({"kind":"materialized"})).unwrap();
    doc["pipeline"]["workflow"]["inputs"]["activations"]["contract"]["shape"][1] =
        serde_yaml::to_value(3).unwrap();
    doc["pipeline"]["workflow"]["components"]["overlay"]["ports"]["inputs"]["input"]["shape"][1] =
        serde_yaml::to_value(3).unwrap();
    for component in ["decoder", "overlay"] {
        doc["pipeline"]["workflow"]["components"][component]["batch_capacity"] =
            serde_yaml::to_value(json!({"budgets":[{"dimensions":["batch"],"max_total":2}]}))
                .unwrap();
    }
    let input = json!({"dtype":"float32","shape":["batch",3],"batch_layout":{"kind":"request_aligned","axis":0}});
    let output = json!({"dtype":"float32","shape":["batch",2],"batch_layout":{"kind":"request_aligned","axis":0}});
    doc["pipeline"]["workflow"]["components"]["decoder"]["ports"] = serde_yaml::to_value(
        json!({"inputs":{"activations":input},"outputs":{"projection.output":output}}),
    )
    .unwrap();
    doc["pipeline"]["workflow"]["steps"]=serde_yaml::to_value(json!([
        {"kind":"invoke","component":"decoder","inputs":{"activations":"activations"},"outputs":{"projection.output":"projected"}},
        {"kind":"invoke","component":"overlay","inputs":{"input":"activations","base":"projected"},"outputs":{"output":"adapted"}},
        {"kind":"emit","value":"adapted","output":"result","mode":"replace"}
    ])).unwrap();
    std::fs::write(
        root.path().join("inference_metadata.yaml"),
        serde_yaml::to_string(&doc).unwrap(),
    )
    .unwrap();
    std::fs::write(root.path().join("model.onnx.textproto"),r#"ir_version:11
opset_import { domain:"" version:24 }
graph {
 name:"explicit_projection"
 initializer { name:"projection" data_type:1 dims:3 dims:2 float_data:2 float_data:0 float_data:0 float_data:3 float_data:0 float_data:0 }
 node { name:"projection" op_type:"MatMul" input:"activations" input:"projection" output:"projection.output" }
 input { name:"activations" type { tensor_type { elem_type:1 shape { dim { dim_param:"batch" } dim { dim_value:3 } } } } }
 output { name:"projection.output" type { tensor_type { elem_type:1 shape { dim { dim_param:"batch" } dim { dim_value:2 } } } } }
}"#).unwrap();
    std::fs::create_dir_all(root.path().join("adapters/gguf")).unwrap();
    std::fs::write(path(root.path()), wire::container(&facts(), &tensors())).unwrap();
    root
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
fn run(engine: &mut Engine, scales: &[Option<f32>]) -> anyhow::Result<Vec<f32>> {
    let mut selection = AdapterSelection::default();
    for scale in scales {
        selection = selection.with_row(scale.map(|s| AdapterActivation::new("gguf", s)));
    }
    let input = if scales.len() == 1 {
        vec![2f32, 5., 7.]
    } else {
        vec![2f32, 5., 7., 7., 11., 13.]
    };
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: GenerateOptions {
            adapter_selection: Some(selection),
            ..Default::default()
        },
    })
    .with_input(
        "activations",
        Value::from_slice_f32(&input, &[scales.len() as i64, 3])?,
    );
    Ok(engine.run_pipeline(request)?["result"].to_vec_f32()?)
}
#[test]
fn native_gguf_rank_two_rectangular_factors_preserve_alpha_and_independent_row_scales() {
    let root = package();
    let mut native = engine(root.path());
    assert_eq!(
        run(&mut native, &[None, None]).unwrap(),
        [4., 15., 14., 33.]
    );
    // A*x=[11,4]/[22,12.5]; B*(A*x)=[13,-1.5]/[28.25,1.5]. alpha/r=2.
    assert_eq!(
        run(&mut native, &[Some(0.5), Some(-0.25)]).unwrap(),
        [17., 13.5, -0.125, 32.25]
    );
    assert_eq!(
        run(&mut native, &[Some(0.0), None]).unwrap(),
        [4., 15., 14., 33.]
    );
}
#[test]
fn native_gguf_embedded_metadata_changes_cannot_hide_behind_cached_factors() {
    let root = package();
    let original = std::fs::read(path(root.path())).unwrap();
    let mut native = engine(root.path());
    assert_eq!(run(&mut native, &[Some(0.5)]).unwrap(), [17., 13.5]);
    let mut changed = facts();
    changed[2].2 = 8f32.to_le_bytes().to_vec();
    let mut changed_factor = tensors();
    changed_factor[0].3[..4].copy_from_slice(&9f32.to_le_bytes());
    for bytes in [
        wire::container(&changed, &tensors()),
        wire::container(&facts(), &changed_factor),
    ] {
        assert_eq!(
            bytes.len(),
            original.len(),
            "same-length metadata/factor mutation"
        );
        std::fs::write(path(root.path()), bytes).unwrap();
        let error = run(&mut native, &[Some(0.5)]).unwrap_err();
        assert!(
            format!("{error:#}").contains("changed after loading"),
            "{error:#}"
        );
        assert_eq!(run(&mut native, &[None]).unwrap(), [4., 15.]);
        std::fs::write(path(root.path()), &original).unwrap();
        assert_eq!(run(&mut native, &[Some(0.5)]).unwrap(), [17., 13.5]);
    }
}
#[test]
fn native_gguf_rejects_bad_metadata_storage_and_truncation_then_recovers() {
    let mut cases = Vec::new();
    for (index, kind, bytes, message) in [
        (0, 8, wire::string("model"), "general.type"),
        (1, 8, wire::string("dora"), "adapter.type"),
        (0, 4, 0u32.to_le_bytes().to_vec(), "STRING"),
        (1, 6, 1f32.to_le_bytes().to_vec(), "STRING"),
        (2, 4, 4u32.to_le_bytes().to_vec(), "FLOAT32"),
        (2, 6, 0f32.to_le_bytes().to_vec(), "finite and positive"),
        (2, 6, f32::NAN.to_le_bytes().to_vec(), "finite and positive"),
        (2, 6, 8f32.to_le_bytes().to_vec(), "disagrees"),
    ] {
        let mut f = facts();
        f[index].1 = kind;
        f[index].2 = bytes;
        cases.push((wire::container(&f, &tensors()), message));
    }
    for index in 0..3 {
        let mut missing = facts();
        missing.remove(index);
        cases.push((wire::container(&missing, &tensors()), "is missing"));
    }
    let mut wrong_shape = tensors();
    wrong_shape[0].2 = vec![3, 2];
    cases.push((
        wire::container(&facts(), &wrong_shape),
        "canonical row-major shape",
    ));
    let mut quantized = tensors();
    quantized[0].1 = 2;
    quantized[0].2 = vec![2, 32];
    quantized[0].3 = vec![0; 36];
    let mut quant_facts = facts();
    quant_facts.push((
        "general.quantization_version".into(),
        4,
        2u32.to_le_bytes().to_vec(),
    ));
    cases.push((wire::container(&quant_facts, &quantized), "unquantized F32"));
    let mut duplicate = facts();
    duplicate.push(duplicate[2].clone());
    cases.push((
        wire::container(&duplicate, &tensors()),
        "repeated GGUF metadata",
    ));
    let mut variant = facts();
    variant.push((
        "adapter.alora.invocation_tokens".into(),
        8,
        wire::string("unsupported"),
    ));
    cases.push((
        wire::container(&variant, &tensors()),
        "unsupported semantics",
    ));
    for dtype in [1u32, 30] {
        let mut t = tensors();
        for tensor in &mut t {
            tensor.1 = dtype;
            tensor.3 = vec![0; tensor.3.len() / 2];
        }
        cases.push((wire::container(&facts(), &t), "unquantized F32"));
    }
    let mut nonfinite = tensors();
    nonfinite[0].3[..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
    cases.push((wire::container(&facts(), &nonfinite), "nonfinite"));
    let mut truncated = wire::container(&facts(), &tensors());
    truncated.pop();
    cases.push((truncated, "beyond the file"));
    let mut extra = tensors();
    extra.push((
        "unmapped.lora_a".into(),
        0,
        vec![1, 1],
        0f32.to_le_bytes().to_vec(),
    ));
    cases.push((
        wire::container(&facts(), &extra),
        "without exact manifest bindings",
    ));
    for (bytes, message) in cases {
        let root = package();
        let original = std::fs::read(path(root.path())).unwrap();
        std::fs::write(path(root.path()), bytes).unwrap();
        let mut native = engine(root.path());
        let error = run(&mut native, &[Some(0.5)]).unwrap_err();
        assert!(
            format!("{error:#}").contains(message),
            "{message}: {error:#}"
        );
        assert_eq!(run(&mut native, &[None]).unwrap(), [4., 15.]);
        std::fs::write(path(root.path()), original).unwrap();
        assert_eq!(run(&mut native, &[Some(0.5)]).unwrap(), [17., 13.5]);
    }
}
#[test]
fn native_gguf_governor_rejects_before_factor_decode_and_keeps_base_usable() {
    let root = package();
    let original = std::fs::read(path(root.path())).unwrap();
    let mut provenance = facts();
    provenance.push((
        "general.description".into(),
        8,
        wire::string(&"x".repeat(8 * 1024 * 1024)),
    ));
    std::fs::write(path(root.path()), wire::container(&provenance, &tensors())).unwrap();
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
    let error = run(&mut native, &[Some(0.5)]).unwrap_err();
    assert!(
        format!("{error:#}").contains("reserve adapter container parsing memory"),
        "{error:#}"
    );
    assert_eq!(run(&mut native, &[None]).unwrap(), [4., 15.]);
    std::fs::write(path(root.path()), original).unwrap();
    assert_eq!(run(&mut native, &[Some(0.5)]).unwrap(), [17., 13.5]);
}
