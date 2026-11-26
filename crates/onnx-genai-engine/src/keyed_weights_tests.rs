use crate::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest,
    NativeStateFileLimits, NativeStateFilePolicy,
};
use anyhow::{Context, Result, ensure};
use onnx_runtime_loader::proto::onnx;
use prost::Message;
use serde_json::json;
use std::{
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine")
}
fn copy_package(source: &Path, target: &Path) -> Result<()> {
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        ensure!(
            entry.file_type()?.is_file(),
            "test package must contain only regular files"
        );
        std::fs::copy(entry.path(), target.join(entry.file_name()))?;
    }
    Ok(())
}
fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}
pub(crate) fn container(format: &str, dims: &[i64], payload: &[u8]) -> (Vec<u8>, usize) {
    let mut bytes = Vec::new();
    if format == "gguf" {
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        string(&mut bytes, "container.parameter");
        bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &dim in dims.iter().rev() {
            bytes.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // GGML F32
        bytes.extend_from_slice(&0_u64.to_le_bytes()); // tensor offset from data start
        bytes.resize(bytes.len().next_multiple_of(32), 0);
    } else {
        let mut header = json!({"container.parameter":{"dtype":"F32","shape":dims,"data_offsets":[0,payload.len()]}}).to_string().into_bytes();
        header.resize(header.len().next_multiple_of(8), b' ');
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend(header);
    }
    let offset = bytes.len();
    bytes.extend_from_slice(payload);
    (bytes, offset)
}
struct Package {
    root: tempfile::TempDir,
    source: PathBuf,
    payload_offset: usize,
    dims: Vec<i64>,
    payload: Vec<u8>,
}
fn package(format: &str) -> Result<Package> {
    let root = tempfile::tempdir()?;
    copy_package(&fixture(), root.path())?;
    let original = onnx_genai_ort::ModelDirectory::load(root.path())?;
    let original_name = original
        .model_path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let mut model = onnx::ModelProto::decode(
        onnx_runtime_loader::read_model_binary(&original.model_path)?.as_slice(),
    )?;
    let graph = model.graph.as_mut().context("fixture graph")?;
    ensure!(graph.initializer.iter().all(|tensor| tensor.data_location != onnx::tensor_proto::DataLocation::External as i32), "fixture weights must be inline");
    // Move the fixture's Constant logits into an ordinary initializer before
    // externalizing it. This preserves its ONNX semantics and real KV graph.
    let node = graph
        .node
        .iter()
        .position(|node| {
            node.op_type == "Constant"
                && node
                    .attribute
                    .iter()
                    .any(|attr| attr.t.as_ref().is_some_and(|tensor| tensor.data_type == 1))
        })
        .context("fixture float Constant")?;
    let node = graph.node.remove(node);
    let mut tensor = node
        .attribute
        .into_iter()
        .find_map(|attr| attr.t)
        .context("constant tensor")?;
    tensor.name = node.output.first().context("constant output")?.clone();
    let payload = if tensor.raw_data.is_empty() {
        tensor
            .float_data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    } else {
        tensor.raw_data.clone()
    };
    ensure!(
        !payload.is_empty(),
        "fixture payload must exercise real container bytes"
    );
    let dims = tensor.dims.clone();
    tensor.raw_data.clear();
    tensor.float_data.clear();
    tensor.data_location = onnx::tensor_proto::DataLocation::External as i32;
    // Keep the optional length consistent with the exact container tensor;
    // admission must retain the loader's descriptor consistency checks.
    tensor.external_data = vec![onnx::StringStringEntryProto {
        key: "length".into(),
        value: payload.len().to_string(),
    }];
    let name = tensor.name.clone();
    graph.initializer.push(tensor);
    std::fs::write(root.path().join("model.onnx"), model.encode_to_vec())?;
    std::fs::remove_file(&original.model_path)?;
    let metadata_path = original.metadata_path.context("fixture metadata")?;
    let mut metadata: serde_json::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata_path)?)?;
    for component in metadata["pipeline"]["workflow"]["components"]
        .as_object_mut()
        .context("fixture components")?
        .values_mut()
    {
        if component["implementation"]["artifact"] == original_name {
            component["implementation"]["artifact"] = json!("model.onnx");
        }
    }
    let file = format!("weights.{format}");
    metadata["schema_version"] = json!("v1.17");
    metadata["package"]["tokenizer"]["special_tokens"]["eos_token_id"] = json!([1]);
    metadata["pipeline"]["workflow"]["publication_mode"] = json!("commit_only");
    for output in metadata["pipeline"]["workflow"]["outputs"]
        .as_object_mut()
        .context("fixture outputs")?
        .values_mut()
    {
        output["family"] = json!({"kind":"materialized"});
    }
    metadata["graph_weights"] = json!({"model.onnx":[{"path":file,"format":format,"renames":{"container.parameter":name}}]});
    std::fs::write(&metadata_path, serde_yaml::to_string(&metadata)?)?;
    let (bytes, payload_offset) = container(format, &dims, &payload);
    let source = root.path().join(&file);
    std::fs::write(&source, bytes)?;
    assert_eq!(
        onnx_genai_ort::model_weight_bytes(&root.path().join("model.onnx")),
        std::fs::metadata(root.path().join("model.onnx"))?.len() + payload.len() as u64
    );
    Ok(Package {
        root,
        source,
        payload_offset,
        dims,
        payload,
    })
}
fn load(root: &Path, backend: EngineDecodeBackend) -> Result<Engine> {
    Engine::from_dir(
        root,
        EngineConfig {
            decode_backend: backend,
            native_device: Some(crate::NativeDecodeDevice::Cpu),
            native_state_files: if backend == EngineDecodeBackend::Ort {
                NativeStateFilePolicy::Disabled
            } else {
                NativeStateFilePolicy::Required
            },
            ..Default::default()
        },
    )
}
fn request(tokens: Vec<u32>, count: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
    request.options.max_new_tokens = count;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request
}
#[test]
fn keyed_containers_execute_native_and_bind_portable_state_to_their_content() -> Result<()> {
    let mut reference = load(&fixture(), EngineDecodeBackend::Ort)?;
    let expected = reference.generate(request(vec![0], 3))?.token_ids;
    for format in ["gguf", "safetensors"] {
        let package = package(format)?;
        let mut original = load(package.root.path(), EngineDecodeBackend::Auto)?;
        let session = original.create_session()?;
        let generated = original.generate_in_session(session, request(vec![0], 3))?;
        assert_eq!(generated.token_ids, expected, "{format}");
        let mut prefix = vec![0];
        prefix.extend(generated.token_ids);
        let mut saved = Vec::new();
        original.write_native_session_state(
            session,
            &mut saved,
            NativeStateFileLimits::default(),
        )?;
        let continuation = original
            .generate_in_session(session, request(prefix.clone(), 2))?
            .token_ids;
        drop(original);
        let moved = tempfile::tempdir()?;
        copy_package(package.root.path(), moved.path())?;
        let mut restarted = load(moved.path(), EngineDecodeBackend::Native)?;
        let target = restarted.create_session()?;
        restarted.restore_native_session_state(
            target,
            &mut saved.as_slice(),
            NativeStateFileLimits::default(),
        )?;
        assert_eq!(
            restarted
                .generate_in_session(target, request(prefix.clone(), 2))?
                .token_ids,
            continuation,
            "{format}: moved package continuation"
        );
        drop(restarted);
        let mut bytes = std::fs::read(&package.source)?;
        bytes[package.payload_offset] ^= 1;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&package.source)?;
        file.seek(SeekFrom::Start(package.payload_offset as u64))?;
        file.write_all(&bytes[package.payload_offset..package.payload_offset + 1])?;
        drop(file);
        let mut changed = load(package.root.path(), EngineDecodeBackend::Native)?;
        let target = changed.create_session()?;
        let error = changed
            .restore_native_session_state(
                target,
                &mut saved.as_slice(),
                NativeStateFileLimits::default(),
            )
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("identity"),
            "{format}: {error:#}"
        );
        assert_eq!(changed.session_token_count(target)?, 0);
    }
    Ok(())
}
#[test]
fn keyed_containers_refuse_ort_and_shape_coercion() -> Result<()> {
    for format in ["gguf", "safetensors"] {
        let package = package(format)?;
        let error = load(package.root.path(), EngineDecodeBackend::Ort)
            .err()
            .context("ORT must not ignore keyed bindings")?;
        assert!(
            format!("{error:#}").contains("graph_weights")
                && format!("{error:#}").contains("Native"),
            "{error:#}"
        );
        let flattened = vec![package.dims.iter().product()];
        let (bytes, _) = container(format, &flattened, &package.payload);
        std::fs::write(&package.source, bytes)?;
        let error = load(package.root.path(), EngineDecodeBackend::Native)
            .err()
            .context("shape coercion must be refused")?;
        assert!(
            format!("{error:#}").contains("exactly matching dtype and shape"),
            "{format}: {error:#}"
        );
    }
    Ok(())
}

#[test]
fn keyed_containers_execute_through_the_workflow_component_loader() -> Result<()> {
    use crate::pipeline::{PipelineGenerateRequest, WorkflowRuntime};
    use onnx_genai_ort::SessionOptions;
    for format in ["gguf", "safetensors"] {
        let package = package(format)?;
        let path = package.root.path().join("model.onnx");
        let mut model = onnx::ModelProto::decode(std::fs::read(&path)?.as_slice())?;
        let graph = model.graph.as_mut().context("workflow graph")?;
        let name = graph
            .initializer
            .iter()
            .find(|tensor| {
                tensor.data_location == onnx::tensor_proto::DataLocation::External as i32
            })
            .context("keyed initializer")?
            .name
            .clone();
        graph.node.clear();
        graph.input.clear();
        graph.initializer.retain(|tensor| tensor.name == name);
        graph.output.retain(|output| output.name == name);
        graph.value_info.clear();
        ensure!(
            graph.output.len() == 1,
            "fixture keyed value must be a graph output"
        );
        std::fs::write(&path, model.encode_to_vec())?;
        let metadata_path = package.root.path().join("inference_metadata.yaml");
        let mut metadata: serde_json::Value =
            serde_yaml::from_str(&std::fs::read_to_string(&metadata_path)?)?;
        let contract = json!({"dtype":"float32", "shape":package.dims});
        metadata["pipeline"]["workflow"] = json!({
            "manifest":{}, "publication_mode":"commit_only", "inputs":{},
            "outputs":{"scores":{"contract":contract,"role":"tensor","stage":"pre_adapter","family":{"kind":"materialized"}}},
            "components":{"weights":{"implementation":{"kind":"onnx","artifact":"model.onnx"},"ports":{"inputs":{},"outputs":{name.clone():contract}}}},
            "steps":[{"kind":"invoke","component":"weights","inputs":{},"outputs":{name:"result"}},
                     {"kind":"emit","value":"result","output":"scores","mode":"replace"}]
        });
        std::fs::write(metadata_path, serde_yaml::to_string(&metadata)?)?;
        let (mut workflow, _governor) = WorkflowRuntime::from_dir_with_session_options(
            package.root.path(),
            EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(crate::NativeDecodeDevice::Cpu),
                ..Default::default()
            },
            SessionOptions::default(),
        )?;
        let outputs = workflow.run_pipeline(PipelineGenerateRequest::new(request(vec![0], 1)))?;
        let scores = outputs.get("scores").context("workflow scores output")?;
        let expected = package
            .payload
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(scores.to_vec_f32()?, expected, "{format}");
    }
    Ok(())
}
