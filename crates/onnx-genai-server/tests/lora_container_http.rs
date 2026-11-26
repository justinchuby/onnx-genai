//! HF PEFT-only and GGUF-only catalogues must select the same actual portable source.
#![cfg(feature = "native-backend")]
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use onnx_genai_engine::{EngineConfig, EngineDecodeBackend, NativeDecodeDevice};
use onnx_genai_server::{AppState, ServerConfig, app};
use serde_json::{Value, json};
use std::path::PathBuf;
use tower::ServiceExt;
mod decoder {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/build_lora_decoder.rs"
    ));
}
mod wire {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/build_gguf_lora.rs"
    ));
}
fn package(format: &str, embedding: bool) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    if embedding {
        decoder::embedding(root.path(), &fixtures).unwrap();
    } else {
        decoder::build(root.path(), &fixtures).unwrap();
    }
    let path = root.path().join("inference_metadata.yaml");
    let mut doc: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let original: Value = serde_json::from_slice(
        &std::fs::read(root.path().join("adapters/red/adapter.json")).unwrap(),
    )
    .unwrap();
    let a = original["targets"]["projection"]["a"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    let width = original["targets"]["projection"]["b"]
        .as_array()
        .unwrap()
        .len();
    let mut b = vec![0f32; width];
    b[4] = 10000.;
    b[6] = -10000.;
    // Deterministic actual Native base projection: only appended constant feature
    // contributes, so no adapter always selects token7. +/- scales select4/6.
    let mut projection = vec![0f32; a.len() * width];
    projection[(a.len() - 1) * width + 7] = 3.;
    std::fs::write(
        root.path().join("projection.data"),
        projection
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut artifact = doc["adapters"]["artifacts"]["red"].clone();
    artifact["index"] = serde_yaml::to_value(0).unwrap();
    let prefix = if format == "gguf" {
        "exported.projection"
    } else {
        "base_model.model.explicit.projection"
    };
    artifact["bindings"][0]["weight_key"] = serde_yaml::Value::String(prefix.into());
    let filename = if format == "gguf" {
        "adapters/red/adapter.gguf"
    } else {
        "adapters/red/adapter_model.safetensors"
    };
    let mut weight = json!({"format":format,"location":filename,"scale_encoding":"alpha_over_rank","loader_capability":if format=="gguf" {"onnx-genai.adapters.gguf@1"} else {"onnx-genai.adapters.hf-peft@1"}});
    if format == "hf_peft" {
        weight["config_location"] = json!("adapters/red/adapter_config.json");
        std::fs::write(root.path().join("adapters/red/adapter_config.json"),serde_json::to_vec(&json!({"peft_type":"LORA","r":1,"lora_alpha":1,"target_modules":["projection"],"bias":"none","fan_in_fan_out":false})).unwrap()).unwrap();
        let header=serde_json::to_vec(&json!({format!("{prefix}.lora_A.weight"):{"dtype":"F32","shape":[1,a.len()],"data_offsets":[0,a.len()*4]},format!("{prefix}.lora_B.weight"):{"dtype":"F32","shape":[width,1],"data_offsets":[a.len()*4,(a.len()+b.len())*4]}})).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(a.iter().chain(&b).flat_map(|v| v.to_le_bytes()));
        std::fs::write(root.path().join(filename), bytes).unwrap();
    } else {
        let facts = vec![
            ("general.type".into(), 8, wire::string("adapter")),
            ("adapter.type".into(), 8, wire::string("lora")),
            ("adapter.lora.alpha".into(), 6, 1f32.to_le_bytes().to_vec()),
            ("general.architecture".into(), 8, wire::string("llama")),
        ];
        let tensors = vec![
            (
                format!("{prefix}.lora_a"),
                0,
                vec![1, a.len() as u64],
                a.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            (
                format!("{prefix}.lora_b"),
                0,
                vec![width as u64, 1],
                b.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        ];
        std::fs::write(
            root.path().join(filename),
            wire::container(&facts, &tensors),
        )
        .unwrap();
    }
    artifact["weights"] = serde_yaml::to_value(vec![weight]).unwrap();
    doc["adapters"]["artifacts"] = serde_yaml::to_value(json!({})).unwrap();
    doc["adapters"]["artifacts"]["red"] = artifact;
    doc["schema_version"] = serde_yaml::Value::String("1.19".into());
    doc["pipeline"]["workflow"]["publication_mode"] =
        serde_yaml::Value::String("commit_only".into());
    for output in doc["pipeline"]["workflow"]["outputs"]
        .as_mapping_mut()
        .unwrap()
        .values_mut()
    {
        output["family"] = serde_yaml::to_value(json!({"kind":"materialized"})).unwrap();
    }
    std::fs::write(path, serde_yaml::to_string(&doc).unwrap()).unwrap();
    std::fs::remove_file(root.path().join("adapters/red/adapter.json")).unwrap();
    root
}
async fn call(router: axum::Router, method: &str, path: &str, body: Value) -> (StatusCode, Value) {
    let response = router
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}
fn generation() -> Value {
    json!({"prompt":[2,4],"n_predict":2,"temperature":0,"ignore_eos":true,"return_tokens":true,"cache_prompt":false})
}
async fn exercise(format: &str) {
    let root = package(format, false);
    let router = app(AppState::load_with_config(
        root.path(),
        Some("containers".into()),
        ServerConfig {
            engine_config: EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(NativeDecodeDevice::Cpu),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap());
    let (status, catalog) = call(router.clone(), "GET", "/lora-adapters", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{catalog}");
    assert_eq!(catalog.as_array().unwrap().len(), 1);
    assert_eq!(catalog[0]["id"], 0);
    assert_eq!(catalog[0]["scale"], 1.0);
    assert!(
        catalog[0]["path"]
            .as_str()
            .unwrap()
            .ends_with(if format == "gguf" {
                "adapter.gguf"
            } else {
                "adapter_model.safetensors"
            })
    );
    let (status, positive) = call(router.clone(), "POST", "/completion", generation()).await;
    assert_eq!(status, StatusCode::OK, "{positive}");
    assert_eq!(positive["tokens"], json!([4, 4]));
    for (scale, expected) in [
        (0.5, json!([4, 4])),
        (-0.5, json!([6, 6])),
        (0., json!([7, 7])),
    ] {
        let mut request = generation();
        request["lora"] = json!([{"id":0,"scale":scale}]);
        let (status, result) = call(router.clone(), "POST", "/completion", request).await;
        assert_eq!(status, StatusCode::OK, "{result}");
        assert_eq!(result["tokens"], expected, "{format}, scale={scale}");
        let (_, catalog) = call(router.clone(), "GET", "/lora-adapters", Value::Null).await;
        assert_eq!(
            catalog[0]["scale"], 1.0,
            "request override must not mutate default"
        );
    }
    for (scale, expected) in [
        (-0.5, json!([6, 6])),
        (0., json!([7, 7])),
        (0.5, json!([4, 4])),
    ] {
        let (status, changed) = call(
            router.clone(),
            "POST",
            "/lora-adapters",
            json!([{"id":0,"scale":scale}]),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{changed}");
        let (status, result) = call(router.clone(), "POST", "/completion", generation()).await;
        assert_eq!(status, StatusCode::OK, "{result}");
        assert_eq!(result["tokens"], expected);
    }
    let (status, error) = call(
        router.clone(),
        "POST",
        "/lora-adapters",
        json!([{"id":99,"scale":1}]),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    let mut invalid = generation();
    invalid["lora"] = json!([{"id":99,"scale":1}]);
    assert_eq!(
        call(router.clone(), "POST", "/completion", invalid).await.0,
        StatusCode::BAD_REQUEST
    );
    let (_, catalog) = call(router.clone(), "GET", "/lora-adapters", Value::Null).await;
    assert_eq!(catalog[0]["scale"], 0.5);
    let (status, result) = call(router, "POST", "/completion", generation()).await;
    assert_eq!(status, StatusCode::OK, "{result}");
    assert_eq!(
        result["tokens"],
        json!([4, 4]),
        "invalid selection cannot corrupt active default"
    );
    let embeddings = package(format, true);
    let router = app(AppState::load_with_config(
        embeddings.path(),
        Some("containers".into()),
        ServerConfig {
            engine_config: EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(NativeDecodeDevice::Cpu),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap());
    for scale in [0.5, -0.5, 0.0] {
        let (status, result) = call(
            router.clone(),
            "POST",
            "/embedding",
            json!({"content":[2,4],"embd_normalize":-1,"lora":[{"id":0,"scale":scale}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{result}");
        assert_eq!(
            result[0]["embedding"][0][4],
            json!(scale * 10000.0),
            "{format}: raw Native embedding positive factor"
        );
        assert_eq!(
            result[0]["embedding"][0][6],
            json!(-scale * 10000.0),
            "{format}: raw Native embedding negative factor"
        );
        assert_eq!(
            result[0]["embedding"][0][7],
            json!(3.0),
            "actual Native base projection retained"
        );
    }
}
#[tokio::test]
async fn native_http_hf_peft_only_catalogue_defaults_and_request_scales_use_actual_factors() {
    exercise("hf_peft").await;
}
#[tokio::test]
async fn native_http_gguf_only_catalogue_defaults_and_request_scales_use_actual_factors() {
    exercise("gguf").await;
}
