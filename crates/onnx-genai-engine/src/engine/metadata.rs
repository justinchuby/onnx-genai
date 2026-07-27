//! Compatibility metadata derivation and ONNX graph inspection.

use super::*;

pub(crate) fn default_inference_metadata() -> InferenceMetadata {
    InferenceMetadata::default()
}

/// Optional cap (in tokens) on the runtime-owned fixed-capacity KV buffer,
/// read from `ONNX_GENAI_KV_MAX_LEN`. Returns `None` when the variable is
/// unset, empty, or unparseable (in which case the model's full advertised
/// context length is used, preserving prior behavior).
pub(crate) fn shared_buffer_cap_from_env() -> Option<usize> {
    std::env::var("ONNX_GENAI_KV_MAX_LEN")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&cap| cap > 0)
}

/// Apply an optional KV-buffer capacity cap: the effective length is the
/// smaller of the model's advertised max length and the cap, if any.
pub(crate) fn cap_kv_len(model_max_len: usize, cap: Option<usize>) -> usize {
    cap.map_or(model_max_len, |cap| model_max_len.min(cap))
}

/// Like [`genai_config_compat_metadata`], but derives the decoder graph
/// inventory by inspecting the ONNX model file directly (used by the native
/// decoder constructor, which builds metadata before a `Session` exists).
///
/// The native decode loop binds exactly the ports the metadata names, so — like
/// the ORT path — a hybrid SSM/attention decoder must have its KV/state topology
/// read from the graph, not expanded from a uniform layer count. If the graph
/// cannot be inspected, this falls back to pattern-expanded metadata so no
/// currently loading model regresses.
#[cfg(feature = "native-backend")]
pub(crate) fn genai_config_compat_metadata_from_model_path(
    model_dir: &Path,
    model_path: &Path,
) -> anyhow::Result<Option<InferenceMetadata>> {
    let decoder_graph = decoder_graph_info_from_model_path(model_path);
    let result = match &decoder_graph {
        Some(graph) => {
            let kv_native_dtype = graph
                .inputs
                .iter()
                .find(|info| crate::decode::is_kv_input(&info.name))
                .map(|info| info.dtype.as_str());
            onnx_genai_genai_config::inference_metadata_from_dir_with_graph(
                model_dir,
                kv_native_dtype,
                graph,
            )
        }
        None => onnx_genai_genai_config::inference_metadata_from_dir(model_dir, None),
    };
    result.map_err(|e| anyhow::anyhow!("Failed to convert genai_config.json: {e}"))
}

/// Best-effort decoder graph inventory read straight from an ONNX model file,
/// mirroring the ORT loader's graph inspection. Returns `None` on any failure so
/// callers fall back to pattern-expanded metadata. Only the graph interface
/// (port names, dtypes, shapes) is needed — external weight data is never read.
#[cfg(feature = "native-backend")]
pub(crate) fn decoder_graph_info_from_model_path(
    model_path: &Path,
) -> Option<onnx_genai_genai_config::ModelGraphInfo> {
    use onnx_runtime_ir::Dim;
    let graph = onnx_runtime_loader::load_model(model_path).ok()?;
    let tensor_info =
        |id: &onnx_runtime_ir::ValueId| -> Option<onnx_genai_genai_config::GraphTensorInfo> {
            let value = graph.value(*id);
            let name = value.name.clone()?;
            Some(onnx_genai_genai_config::GraphTensorInfo {
                name,
                dtype: ir_dtype_name(value.dtype).to_owned(),
                dimensions: value
                    .shape
                    .iter()
                    .map(|dim| match dim {
                        Dim::Static(value) => Some(*value),
                        Dim::Symbolic(_) => None,
                    })
                    .collect(),
            })
        };
    let inputs = graph
        .inputs
        .iter()
        .map(tensor_info)
        .collect::<Option<Vec<_>>>()?;
    let outputs = graph
        .outputs
        .iter()
        .map(tensor_info)
        .collect::<Option<Vec<_>>>()?;
    Some(onnx_genai_genai_config::ModelGraphInfo { inputs, outputs })
}

/// Canonical lowercase dtype spelling for an `onnx_runtime_ir` graph dtype.
#[cfg(feature = "native-backend")]
pub(crate) fn ir_dtype_name(dtype: onnx_runtime_ir::DataType) -> &'static str {
    use onnx_runtime_ir::DataType;
    match dtype {
        DataType::Float32 => "float32",
        DataType::Float16 => "float16",
        DataType::BFloat16 => "bfloat16",
        DataType::Float64 => "float64",
        DataType::Uint8 => "uint8",
        DataType::Int8 => "int8",
        DataType::Uint16 => "uint16",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::Uint32 => "uint32",
        DataType::Uint64 => "uint64",
        DataType::Bool => "bool",
        DataType::String => "string",
        DataType::Complex64 => "complex64",
        DataType::Complex128 => "complex128",
        DataType::Float8E4M3FN => "float8_e4m3fn",
        DataType::Float8E4M3FNUZ => "float8_e4m3fnuz",
        DataType::Float8E5M2 => "float8_e5m2",
        DataType::Float8E5M2FNUZ => "float8_e5m2fnuz",
        DataType::Uint4 => "uint4",
        DataType::Int4 => "int4",
        _ => "undefined",
    }
}

/// Best-effort native metadata derived from an onnxruntime-genai
/// `genai_config.json` in `model_dir`, used only when no
/// `inference_metadata.yaml` is present. Returns `Ok(None)` when there is no
/// `genai_config.json`. The KV cache native dtype is read from the loaded
/// session's KV inputs, since it is not present in `genai_config.json`.
pub(crate) fn genai_config_compat_metadata(
    model_dir: &Path,
    session: &Session,
) -> anyhow::Result<Option<InferenceMetadata>> {
    let kv_native_dtype = session
        .inputs()
        .iter()
        .find(|info| crate::decode::is_kv_input(&info.name))
        .and_then(|info| match info.dtype {
            DataType::Float16 => Some("float16"),
            DataType::BFloat16 => Some("bfloat16"),
            DataType::Float32 => Some("float32"),
            _ => None,
        });
    // Hand the decoder's actual ONNX graph inventory to the compatibility
    // converter so it declares exactly the KV/state ports the graph exposes.
    // onnxruntime-genai `genai_config.json` only carries a uniform per-layer KV
    // name pattern and a total layer count; for hybrid SSM/attention decoders
    // (qwen3.5: most layers are linear-attention with `conv_state`/
    // `recurrent_state`, only the periodic full-attention layers expose dense
    // `key`/`value`) that pattern names ports the graph never exposes and warmup
    // aborts. Deriving the topology from the graph yields sparse `kv_inputs`/
    // `kv_outputs` plus recurrent `state_pairs`; uniform dense-KV decoders are
    // unchanged.
    let decoder_graph = session_model_graph_info(session);
    onnx_genai_genai_config::inference_metadata_from_dir_with_graph(
        model_dir,
        kv_native_dtype,
        &decoder_graph,
    )
    .map_err(|e| anyhow::anyhow!("Failed to convert genai_config.json: {e}"))
}

/// Build a [`ModelGraphInfo`] inventory from a loaded session's input/output
/// port metadata, mirroring the ONNX graph interface the strict compatibility
/// converter consumes (names, dtype spelling, and per-axis static/symbolic
/// dimensions). ORT reports dynamic axes as negative dimensions, which map to
/// symbolic (`None`) entries.
pub(crate) fn session_model_graph_info(
    session: &Session,
) -> onnx_genai_genai_config::ModelGraphInfo {
    fn tensor_info(meta: &onnx_genai_ort::TensorInfo) -> onnx_genai_genai_config::GraphTensorInfo {
        onnx_genai_genai_config::GraphTensorInfo {
            name: meta.name.clone(),
            dtype: graph_dtype_name(meta.dtype).to_owned(),
            dimensions: meta
                .shape
                .iter()
                .map(|&dim| usize::try_from(dim).ok())
                .collect(),
        }
    }
    onnx_genai_genai_config::ModelGraphInfo {
        inputs: session.inputs().iter().map(tensor_info).collect(),
        outputs: session.outputs().iter().map(tensor_info).collect(),
    }
}

/// Canonical lowercase dtype spelling used by the compatibility metadata
/// converter's graph inventory (`float32`, `float16`, `bfloat16`, ...).
pub(crate) fn graph_dtype_name(dtype: DataType) -> &'static str {
    match dtype {
        DataType::Float32 => "float32",
        DataType::Float16 => "float16",
        DataType::BFloat16 => "bfloat16",
        DataType::Float8E4M3 => "float8_e4m3fn",
        DataType::Float8E5M2 => "float8_e5m2",
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::Uint8 => "uint8",
        DataType::Uint16 => "uint16",
        DataType::Uint32 => "uint32",
        DataType::Uint64 => "uint64",
        DataType::Bool => "bool",
    }
}

/// Preserve explicit ORT graph settings and warn about known-unsafe opt-ins.
pub(crate) fn configure_ort_cuda_graph(options: &mut SessionOptions, model_path: &Path) {
    if options.selects_cuda() && options.graph_capture && model_has_control_flow_nodes(model_path) {
        tracing::warn!(
            "ORT CUDA graph capture was explicitly enabled for model '{}', but it contains control-flow nodes (If/Loop/Scan); capture may fail or run substantially slower",
            model_path.display()
        );
    }
}

/// Whether the ONNX model at `model_path` contains top-level control-flow nodes
/// (`If`/`Loop`/`Scan`). ORT cannot capture a CUDA graph for such models, and
/// requesting capture anyway (via the `enable_cuda_graph` provider option)
/// forces a pathological ~6× slower per-Run path, so the caller must leave graph
/// capture disabled when this returns `true`.
///
/// Returns `true` whenever the model cannot be inspected. CUDA graph capture is
/// an optional optimization, so uncertain models conservatively skip it rather
/// than risking ORT's pathological uncaptured per-Run path.
pub(crate) fn model_has_control_flow_nodes(model_path: &Path) -> bool {
    scan_top_level_control_flow(model_path).unwrap_or(true)
}

/// Control-flow op names that block CUDA graph capture, in the default ONNX
/// domain (`""`/`ai.onnx`).
const CONTROL_FLOW_OPS: [&str; 3] = ["If", "Loop", "Scan"];

/// A deliberately minimal view of an ONNX `ModelProto` carrying only the fields
/// needed to reach each top-level node's `op_type`/`domain`.
///
/// Every other field is *absent* from these structs — crucially
/// `GraphProto.initializer` and its `TensorProto.raw_data`, which hold the
/// multi-gigabyte inline weights of models like the qwen3 exports (whose
/// `model.onnx` is over 1 GB). prost's decoder skips any field not declared here
/// with `Buf::advance` (pointer arithmetic), so those weight bytes are never
/// copied — and, when decoding from a memory map, never even faulted in. This
/// keeps the scan cheap regardless of weight size while reusing prost's
/// well-tested wire parser instead of a bespoke byte walker.
#[derive(Clone, PartialEq, ::prost::Message)]
struct ScanModelProto {
    /// `ModelProto.graph`. Repeated occurrences merge per protobuf semantics, so
    /// nodes from every graph field accumulate here.
    #[prost(message, optional, tag = "7")]
    graph: Option<ScanGraphProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct ScanGraphProto {
    /// `GraphProto.node`. `initializer` (tag 5) and every other field is skipped.
    #[prost(message, repeated, tag = "1")]
    node: Vec<ScanNodeProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct ScanNodeProto {
    /// `NodeProto.op_type`.
    #[prost(string, tag = "4")]
    op_type: String,
    /// `NodeProto.domain`. `attribute` (tag 5), which carries subgraph bodies, is
    /// skipped, so only top-level nodes are inspected (matching ORT's capture
    /// eligibility, which only cares about the top-level graph).
    #[prost(string, tag = "7")]
    domain: String,
}

/// Scan for top-level control-flow ops in the ONNX model at `model_path`.
///
/// Returns `Some(true)`/`Some(false)` when the model's graph could be parsed, or
/// `None` when the file cannot be opened, memory-mapped, or decoded, or when it
/// carries no graph at all. The caller treats every `None` conservatively as
/// "has control flow".
///
/// Binary models are memory-mapped and decoded into [`ScanModelProto`], whose
/// minimal field set makes prost skip the inline weight tensors without reading
/// them. Textproto fixtures are first converted through the loader's canonical
/// path. An earlier revision gave up on any file over 512 MB and conservatively
/// reported "has control flow", which wrongly disabled CUDA graph capture for
/// large inline-weight models (a ~20% decode-throughput loss).
pub(crate) fn scan_top_level_control_flow(model_path: &Path) -> Option<bool> {
    use prost::Message;

    let model = if onnx_runtime_loader::is_textproto_path(model_path) {
        let bytes = onnx_runtime_loader::read_model_binary(model_path).ok()?;
        ScanModelProto::decode(bytes.as_slice()).ok()?
    } else {
        let file = std::fs::File::open(model_path).ok()?;
        // SAFETY: the model file is treated as immutable for the brief lifetime of
        // this scan. Model files are not rewritten in place while their directory is
        // in use, so no concurrent truncation (which could raise SIGBUS) is expected.
        let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
        ScanModelProto::decode(&mmap[..]).ok()?
    };
    let graph = model.graph?;
    Some(graph.node.iter().any(|node| {
        matches!(node.domain.as_str(), "" | "ai.onnx")
            && CONTROL_FLOW_OPS.contains(&node.op_type.as_str())
    }))
}
