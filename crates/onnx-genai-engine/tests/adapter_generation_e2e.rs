#![cfg(feature = "native-backend")]
// Adapter selection must change executed token logits, not just catalog state.
use onnx_genai_engine::{
    AdapterActivation, AdapterSelection, Engine, EngineConfig, GenerateOptions, GeneratePrompt,
    GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::Value;
use std::path::{Path, PathBuf};
#[path = "../../../tests/fixtures/build_lora_decoder.rs"]
mod fixture;
fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}
fn request(alias: Option<&str>) -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![2, 4]),
        options: GenerateOptions {
            max_new_tokens: 3,
            stop_on_eos: false,
            adapter_selection: alias.map(|name| {
                AdapterSelection::default().with_row([AdapterActivation::new(name, 1.0)])
            }),
            ..Default::default()
        },
    }
}
#[test]
fn native_adapter_generation_executes_selected_weights_and_keeps_base_immutable()
-> anyhow::Result<()> {
    const CHILD: &str = "NXRT_NATIVE_ADAPTER_NO_ORT_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "native_adapter_generation_executes_selected_weights_and_keeps_base_immutable",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env(
                "ONNX_GENAI_ORT_LIB",
                "/missing/nxrt-native-adapter-test/onnxruntime",
            )
            .env("ONNX_GENAI_EP", "cpu")
            .env_remove("ONNX_GENAI_BACKEND")
            .output()?;
        anyhow::ensure!(
            output.status.success(),
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        anyhow::ensure!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "adapter subprocess selected no test"
        );
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    fixture::build(dir.path(), &fixtures())?;
    let mut engine = Engine::from_dir(dir.path(), EngineConfig::default())?;
    assert!(engine.adapter_service().is_some());
    let baseline = engine.generate(request(None))?.token_ids;
    assert_eq!(engine.generate(request(Some("red")))?.token_ids, vec![4; 3]);
    assert_eq!(
        engine.generate(request(Some("blue")))?.token_ids,
        vec![5; 3]
    );
    assert_eq!(engine.generate(request(None))?.token_ids, baseline);
    let mut disabled = request(None);
    disabled.options.adapter_selection = Some(AdapterSelection::default().with_row([]));
    assert_eq!(engine.generate(disabled)?.token_ids, baseline);
    Ok(())
}
#[test]
fn adapter_rows_lower_through_declared_ids_and_reject_two_sources() -> anyhow::Result<()> {
    let root = fixtures().join("onnx_genai_workflows/adapter");
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let mut options = GenerateOptions {
        adapter_selection: Some(
            AdapterSelection::default()
                .with_row([AdapterActivation::new("red", 0.5)])
                .with_row([]),
        ),
        ..Default::default()
    };
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenRows(vec![vec![], vec![]]),
        options: options.clone(),
    })
    .with_input(
        "activations",
        Value::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?,
    );
    assert_eq!(
        engine.run_pipeline(request)?["result"].to_vec_f32()?,
        vec![1.5, 3.0, 3.0, 4.0]
    );
    options.adapter_selection = Some(AdapterSelection::default().with_row([]));
    let ambiguous = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options,
    })
    .with_input("request.adapter_counts", Value::from_slice_i64(&[0], &[1])?);
    let error = engine
        .run_pipeline(ambiguous)
        .err()
        .expect("two adapter sources must be refused");
    assert!(format!("{error:#}").contains("two sources"), "{error:#}");
    Ok(())
}
#[test]
fn unsupported_executor_and_unqualified_session_fail_before_admission() -> anyhow::Result<()> {
    let mut bare = Engine::from_dir(
        &fixtures().join("tiny-native-engine"),
        EngineConfig::default(),
    )?;
    let mut admitted = false;
    let error = bare
        .generate_with_callbacks(request(Some("red")), Some(&mut || admitted = true), None)
        .unwrap_err();
    assert!(format!("{error:#}").contains("no admitted adapter service"));
    assert!(!admitted);
    let dir = tempfile::tempdir()?;
    fixture::build(dir.path(), &fixtures())?;
    let metadata = dir.path().join("inference_metadata.yaml");
    let mut document: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata)?)?;
    document["pipeline"]["workflow"]["state"]["logits"]["scope"] =
        serde_yaml::Value::String("session".into());
    std::fs::write(metadata, serde_yaml::to_string(&document)?)?;
    let mut engine = Engine::from_dir(dir.path(), EngineConfig::default())?;
    let session = engine.create_session()?;
    let error = engine
        .generate_in_session(session, request(Some("red")))
        .unwrap_err();
    assert!(format!("{error:#}").contains("stateless request"));
    assert_eq!(engine.session_token_count(session)?, 0);
    Ok(())
}

#[test]
fn replayed_conversation_can_switch_adapters_without_stale_state() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    fixture::build(dir.path(), &fixtures())?;
    let mut engine = Engine::from_dir(dir.path(), EngineConfig::default())?;
    assert!(engine.supports_adapter_sessions());
    let session = engine.create_session()?;
    assert_eq!(
        engine
            .generate_in_session(session, request(Some("red")))?
            .token_ids,
        vec![4; 3]
    );
    let before = engine.session_token_count(session)?;
    let mut next = request(Some("blue"));
    next.prompt = GeneratePrompt::TokenIds(vec![5]);
    let result = engine.generate_in_session(session, next.clone())?;
    assert_eq!(result.token_ids, vec![5; 3]);
    assert_eq!(engine.session_token_count(session)?, before + 4);
    next.prompt = GeneratePrompt::TokenIds(vec![2, 4, 4, 4, 4, 5]);
    assert_eq!(engine.generate(next)?.token_ids, result.token_ids);
    Ok(())
}

#[test]
fn native_embedding_executes_adapter_workflow_without_ort() -> anyhow::Result<()> {
    const CHILD: &str = "NXRT_ADAPTER_EMBEDDING_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "native_embedding_executes_adapter_workflow_without_ort",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env(
                "ONNX_GENAI_ORT_LIB",
                "/missing/nxrt-native-embedding/onnxruntime",
            )
            .env("ONNX_GENAI_EP", "cpu")
            .env_remove("ONNX_GENAI_BACKEND")
            .output()?;
        anyhow::ensure!(
            output.status.success(),
            "{} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        anyhow::ensure!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "child did not execute embedding test"
        );
        return Ok(());
    }
    let package = tempfile::tempdir()?;
    fixture::embedding(package.path(), &fixtures())?;
    let mut engine = Engine::from_dir(package.path(), EngineConfig::default())?;
    let base = engine.embed(&[2, 4])?;
    for (alias, index) in [("red", 4), ("blue", 5)] {
        let actual = engine.embed_with_options(
            &[2, 4],
            onnx_genai_engine::EmbeddingOptions {
                adapter_selection: Some(
                    AdapterSelection::default().with_row([AdapterActivation::new(alias, 0.5)]),
                ),
                ..Default::default()
            },
        )?;
        assert_eq!(base.len(), actual.len());
        for (i, (a, b)) in actual.iter().zip(&base).enumerate() {
            assert!(
                (a - b - if i == index { 5000.0 } else { 0.0 }).abs() < 0.01,
                "{alias} column {i}: {a} vs {b}"
            );
        }
    }
    assert_eq!(engine.embed(&[2, 4])?, base);
    let mut bad = Engine::from_dir(&fixtures().join("tiny-mtp-full"), EngineConfig::default())?;
    let error = bad
        .embed_with_options(
            &[2, 4],
            onnx_genai_engine::EmbeddingOptions {
                adapter_selection: Some(
                    AdapterSelection::default().with_row([AdapterActivation::new("red", 1.0)]),
                ),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(format!("{error:#}").contains("no admitted adapter service"));
    Ok(())
}

#[test]
fn embedding_profiles_execute_declared_sequence_pooling() -> anyhow::Result<()> {
    let package = tempfile::tempdir()?;
    std::fs::copy(
        fixtures().join("tiny-llm/tokenizer.json"),
        package.path().join("tokenizer.json"),
    )?;
    std::fs::write(
        package.path().join("encoder.onnx.textproto"),
        r#"
ir_version: 11
opset_import { domain: "" version: 24 }
graph { name: "sequence_features"
  node { input: "ids" output: "numbers" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { input: "numbers" input: "axis" output: "features" op_type: "Unsqueeze" }
  initializer { name: "axis" dims: 1 data_type: 7 int64_data: 2 }
  input { name: "ids" type { tensor_type { elem_type: 7 shape { dim {dim_param:"batch"} dim {dim_param:"sequence"} } } } }
  output { name: "features" type { tensor_type { elem_type: 1 shape { dim {dim_param:"batch"} dim {dim_param:"sequence"} dim {dim_value:1} } } } }
}
"#,
    )?;
    for (kind, expected) in [
        ("mean", 3.0),
        ("max", 4.0),
        ("cls", 2.0),
        ("last_token", 4.0),
    ] {
        let doc = serde_json::json!({
            "schema_version":"v1.2",
            "profiles":{"vectors":{"kind":"embedding","version":"1","outputs":{"embedding":"vector"},"pooling":{"kind":kind,"axis":1}}},
            "pipeline":{"workflow":{
                "manifest":{},
                "inputs":{"ids":{"contract":{"dtype":"int64","shape":["batch","sequence"],"batch_layout":{"kind":"request_aligned","axis":0}},"role":{"kind":"runtime","version":"1.0","role":"prompt_tokens"},"source":{"kind":"request"}}},
                "outputs":{"vector":{"contract":{"dtype":"float32","shape":["batch","sequence",1],"batch_layout":{"kind":"request_aligned","axis":0}},"role":"tensor","stage":"pre_adapter"}},
                "components":{"encoder":{"implementation":{"kind":"onnx","artifact":"encoder.onnx.textproto"}}},
                "steps":[{"kind":"invoke","component":"encoder","inputs":{"ids":"ids"},"outputs":{"features":"hidden"}},{"kind":"emit","value":"hidden","output":"vector","mode":"replace"}]
            }}
        });
        std::fs::write(
            package.path().join("inference_metadata.yaml"),
            serde_yaml::to_string(&doc)?,
        )?;
        let mut engine = Engine::from_dir(package.path(), EngineConfig::default())?;
        assert_eq!(engine.embed(&[2, 4])?, vec![expected], "{kind}");
        assert_eq!(
            engine.embed_with_options(
                &[2, 4],
                onnx_genai_engine::EmbeddingOptions {
                    normalize: true,
                    ..Default::default()
                }
            )?,
            vec![1.0],
            "{kind}"
        );
    }
    Ok(())
}
