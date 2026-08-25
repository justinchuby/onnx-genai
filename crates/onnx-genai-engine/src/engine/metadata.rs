//! Compatibility metadata derivation and ONNX graph inspection.

use super::*;

const MEBIBYTE: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum MetadataDevice {
    Cpu,
    Cuda(u32),
}

/// Load and validate embedded `onnx_runtime.*` hints without materializing the
/// model's initializer payloads.
pub(crate) fn load_model_metadata_hints(model_path: &Path) -> anyhow::Result<MetadataHints> {
    use prost::Message;

    let model = if onnx_runtime_loader::is_textproto_path(model_path) {
        let bytes = onnx_runtime_loader::read_model_binary(model_path)
            .map_err(|error| anyhow::anyhow!("Failed to read ONNX metadata hints: {error}"))?;
        ScanModelProto::decode(bytes.as_slice())
            .map_err(|error| anyhow::anyhow!("Failed to decode ONNX metadata hints: {error}"))?
    } else {
        let file = std::fs::File::open(model_path).with_context(|| {
            format!(
                "Failed to open ONNX model '{}' for metadata-hint scanning",
                model_path.display()
            )
        })?;
        // SAFETY: model packages are immutable while loaded. The minimal protobuf
        // projection skips initializer bytes, so even large inline models are not
        // copied merely to inspect metadata.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.with_context(|| {
            format!(
                "Failed to memory-map ONNX model '{}' for metadata-hint scanning",
                model_path.display()
            )
        })?;
        ScanModelProto::decode(&mmap[..])
            .map_err(|error| anyhow::anyhow!("Failed to decode ONNX metadata hints: {error}"))?
    };

    let mut entries = model
        .metadata_props
        .into_iter()
        .map(|entry| onnx_std::HintEntry {
            scope: onnx_std::HintScope::Model,
            source: onnx_std::HintSource::OnnxMetadata,
            key: entry.key,
            value: entry.value,
        })
        .collect::<Vec<_>>();
    if let Some(graph) = model.graph {
        collect_scan_graph_hint_entries(graph, &mut entries);
    }
    Ok(MetadataHints::scan(entries))
}

fn collect_scan_graph_hint_entries(root: ScanGraphProto, entries: &mut Vec<onnx_std::HintEntry>) {
    enum Work {
        Graph {
            graph: ScanGraphProto,
            display_path: String,
            parent_path: Option<onnx_std::NodePath>,
        },
        Node {
            node: ScanNodeProto,
            structural_path: onnx_std::NodePath,
        },
    }

    let mut work = vec![Work::Graph {
        display_path: root.name.clone(),
        parent_path: None,
        graph: root,
    }];
    while let Some(item) = work.pop() {
        match item {
            Work::Graph {
                graph,
                display_path,
                parent_path,
            } => {
                entries.extend(
                    graph
                        .metadata_props
                        .into_iter()
                        .map(|entry| onnx_std::HintEntry {
                            scope: onnx_std::HintScope::Graph {
                                graph_name: display_path.clone(),
                            },
                            source: onnx_std::HintSource::OnnxMetadata,
                            key: entry.key,
                            value: entry.value,
                        }),
                );
                for (index, node) in graph.node.into_iter().enumerate().rev() {
                    let structural_path = match &parent_path {
                        Some(parent_path) => {
                            parent_path.clone().with_node(node.name.clone(), index)
                        }
                        None => onnx_std::NodePath::root_node(node.name.clone(), index),
                    };
                    work.push(Work::Node {
                        node,
                        structural_path,
                    });
                }
            }
            Work::Node {
                node,
                structural_path,
            } => {
                entries.extend(
                    node.metadata_props
                        .into_iter()
                        .map(|entry| onnx_std::HintEntry {
                            scope: onnx_std::HintScope::Node {
                                path: structural_path.clone(),
                            },
                            source: onnx_std::HintSource::OnnxMetadata,
                            key: entry.key,
                            value: entry.value,
                        }),
                );

                let mut subgraphs = Vec::new();
                for (attribute_index, attribute) in node.attribute.into_iter().enumerate() {
                    if let Some(graph) = attribute.g {
                        let path = structural_path.clone().with_attribute(
                            attribute.name.clone(),
                            attribute_index,
                            None,
                        );
                        subgraphs.push((*graph, path));
                    }
                    for (graph_index, graph) in attribute.graphs.into_iter().enumerate() {
                        let path = structural_path.clone().with_attribute(
                            attribute.name.clone(),
                            attribute_index,
                            Some(graph_index),
                        );
                        subgraphs.push((graph, path));
                    }
                }
                for (graph, parent_path) in subgraphs.into_iter().rev() {
                    work.push(Work::Graph {
                        graph,
                        display_path: parent_path.display_name(),
                        parent_path: Some(parent_path),
                    });
                }
            }
        }
    }
}

pub(crate) fn report_metadata_hint_warnings(hints: &MetadataHints) {
    for warning in &hints.warnings {
        match warning {
            MetadataWarning::UnknownKey { node, key } => tracing::warn!(
                location = node,
                key,
                "Ignoring unrecognized ONNX runtime metadata hint"
            ),
            MetadataWarning::InvalidValue {
                node,
                key,
                value,
                expected,
            } => tracing::warn!(
                location = node,
                key,
                value,
                expected,
                "Ignoring malformed ONNX runtime metadata hint"
            ),
            MetadataWarning::ConflictingForce {
                node,
                source_a,
                source_b,
            } => tracing::error!(
                node,
                source_a = ?source_a,
                source_b = ?source_b,
                "Conflicting forced ONNX runtime placement hints"
            ),
        }
    }
}

/// Apply model arena suggestions only while the corresponding programmatic
/// resource limit remains at its default. This preserves the documented source
/// priority: explicit engine configuration wins over embedded model metadata.
pub(crate) fn apply_model_memory_hints(
    config: &mut EngineConfig,
    hints: &MetadataHints,
) -> anyhow::Result<()> {
    let defaults = ResourceLimits::default();
    if config.limits.vram_limit == defaults.vram_limit
        && let Some(mebibytes) = hints.model.arena_gpu_mb
    {
        config.limits.vram_limit =
            ResourceLimit::Bytes(metadata_mebibytes_to_bytes("arena_gpu_mb", mebibytes)?);
    }
    if config.limits.host_ram_limit == defaults.host_ram_limit
        && let Some(mebibytes) = hints.model.arena_cpu_mb
    {
        config.limits.host_ram_limit =
            ResourceLimit::Bytes(metadata_mebibytes_to_bytes("arena_cpu_mb", mebibytes)?);
    }
    Ok(())
}

fn metadata_mebibytes_to_bytes(key: &str, value: i64) -> anyhow::Result<u64> {
    let value = u64::try_from(value).map_err(|_| {
        anyhow::anyhow!("onnx_runtime.memory.{key} must be a non-negative integer, got {value}")
    })?;
    value.checked_mul(MEBIBYTE).ok_or_else(|| {
        anyhow::anyhow!("onnx_runtime.memory.{key} is too large to represent in bytes: {value}")
    })
}

/// Resolve homogeneous model placement into the session's execution-provider
/// selection. The current engine has one provider per decoder session, so mixed
/// forced devices fail actionably instead of silently violating a hard hint.
///
/// Deferred: per-node heterogeneous placement, colocation groups, kernel
/// selection, pin/priority, overlap, and prefetch require planner-level consumer
/// APIs. The validated hints remain available through `Engine::metadata_hints`.
pub(crate) fn apply_model_placement_hints(
    options: &mut SessionOptions,
    hints: &MetadataHints,
    options_are_programmatic: bool,
) -> anyhow::Result<()> {
    let forced = hinted_devices(hints, PlacementStrength::Force)?;
    if forced.len() > 1 {
        anyhow::bail!(
            "ONNX model metadata forces nodes onto multiple devices ({forced:?}), but the decoder session currently supports one execution provider; use a homogeneous placement or remove force"
        );
    }
    let preferred = hinted_devices(hints, PlacementStrength::Prefer)?;
    let (target, forced_target) = if let Some(target) = forced.first().copied() {
        (Some(target), true)
    } else if preferred.len() == 1 {
        (preferred.first().copied(), false)
    } else {
        (None, false)
    };
    let Some(target) = target else {
        return Ok(());
    };

    if options_are_programmatic {
        if session_options_match_device(options, target) {
            return Ok(());
        }
        if forced_target {
            anyhow::bail!(
                "ONNX model metadata forces placement on {}, but programmatic session options select {}; force hints cannot be overridden",
                metadata_device_name(target),
                selected_session_device_name(options)
            );
        }
        return Ok(());
    }

    let selection = match target {
        MetadataDevice::Cpu => onnx_genai_ort::ep_selection("cpu"),
        MetadataDevice::Cuda(index) => {
            let mut selection = onnx_genai_ort::ep_selection("cuda");
            selection
                .options
                .insert("device_id".to_string(), index.to_string());
            selection
        }
    };
    let selected = SessionOptions::with_execution_provider(selection);
    options.execution_providers = selected.execution_providers;
    options.auto_selected = false;
    Ok(())
}

fn hinted_devices(
    hints: &MetadataHints,
    strength: PlacementStrength,
) -> anyhow::Result<std::collections::BTreeSet<MetadataDevice>> {
    let mut devices = std::collections::BTreeSet::new();
    for value in hints
        .nodes
        .values()
        .filter(|node| node.device_strength == Some(strength))
        .filter_map(|node| node.device.as_deref())
    {
        match parse_metadata_device(value) {
            Ok(device) => {
                devices.insert(device);
            }
            Err(error) if strength == PlacementStrength::Prefer => {
                tracing::warn!(
                    device = value,
                    error = %error,
                    "Ignoring unsupported preferred ONNX runtime device"
                );
            }
            Err(error) => return Err(error),
        }
    }
    Ok(devices)
}

fn parse_metadata_device(value: &str) -> anyhow::Result<MetadataDevice> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized == "cpu" || normalized == "cpu:0" {
        return Ok(MetadataDevice::Cpu);
    }
    if normalized == "gpu" || normalized == "cuda" {
        return Ok(MetadataDevice::Cuda(0));
    }
    if let Some(index) = normalized
        .strip_prefix("gpu:")
        .or_else(|| normalized.strip_prefix("cuda:"))
    {
        return index
            .parse::<u32>()
            .map(MetadataDevice::Cuda)
            .map_err(|_| {
                anyhow::anyhow!(
                    "unsupported onnx_runtime.device value {value:?}; expected cpu, gpu, gpu:<index>, cuda, or cuda:<index>"
                )
            });
    }
    anyhow::bail!(
        "unsupported onnx_runtime.device value {value:?}; this engine currently supports cpu and cuda placement"
    )
}

fn session_options_match_device(options: &SessionOptions, target: MetadataDevice) -> bool {
    options
        .execution_providers
        .iter()
        .any(|provider| match target {
            MetadataDevice::Cpu => provider.caps.is_host(),
            MetadataDevice::Cuda(index) => {
                provider.caps.is_gpu()
                    && provider.caps.is_nvidia()
                    && provider
                        .caps
                        .device_id()
                        .and_then(|value| u32::try_from(value).ok())
                        == Some(index)
            }
        })
}

fn selected_session_device_name(options: &SessionOptions) -> String {
    options
        .execution_providers
        .first()
        .map(|provider| provider.caps.name.clone())
        .unwrap_or_else(|| "no execution provider".to_string())
}

fn metadata_device_name(device: MetadataDevice) -> String {
    match device {
        MetadataDevice::Cpu => "cpu".to_string(),
        MetadataDevice::Cuda(index) => format!("cuda:{index}"),
    }
}

pub(crate) fn default_inference_metadata() -> InferenceMetadata {
    InferenceMetadata::default()
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
/// Import a legacy `genai_config.json` through the fail-closed importer.
///
/// The runtime accepts a lossy import — refusing to load a package the stock
/// runtime can run would help nobody — but it does not accept a *silent* one.
/// Every key the new contract cannot carry is named in the log, with its reason
/// when one is recorded, so an operator sees exactly which package semantics
/// stopped at the boundary instead of discovering them as behaviour drift.
pub(crate) fn genai_config_compat_metadata_from_model_path(
    genai_config_path: Option<&Path>,
    model_path: &Path,
) -> anyhow::Result<Option<InferenceMetadata>> {
    let decoder_graph = decoder_graph_info_from_model_path(model_path);
    let kv_native_dtype = decoder_graph.as_ref().and_then(|graph| {
        graph
            .inputs
            .iter()
            .find(|info| crate::decode::is_kv_input(&info.name))
            .map(|info| info.dtype.as_str())
    });
    // Lossy is permitted, silence is not: the report is the whole point.
    let options = onnx_genai_genai_config::ImportOptions { allow_lossy: true };
    let result = match genai_config_path {
        Some(path) => onnx_genai_genai_config::import_from_path(
            path,
            kv_native_dtype,
            decoder_graph.as_ref(),
            options,
        )
        .map(Some),
        None => onnx_genai_genai_config::import_from_dir(
            model_path.parent().unwrap_or_else(|| Path::new(".")),
            kv_native_dtype,
            decoder_graph.as_ref(),
            options,
        ),
    };
    let imported =
        result.map_err(|e| anyhow::anyhow!("Failed to convert genai_config.json: {e}"))?;
    Ok(imported.map(|(metadata, report)| {
        if report.is_lossy() {
            tracing::warn!(
                dropped_keys = report.dropped_keys.len(),
                "genai_config.json import dropped keys the inference metadata contract does not \
                 carry: {}",
                report
                    .dropped_keys
                    .iter()
                    .map(|key| match onnx_genai_genai_config::drop_reason(key) {
                        Some(reason) => format!("{key} ({reason})"),
                        None => key.clone(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        metadata
    }))
}

/// Best-effort decoder graph inventory read straight from an ONNX model file,
/// mirroring the ORT loader's graph inspection. Returns `None` on any failure so
/// callers fall back to pattern-expanded metadata. Only the graph interface
/// (port names, dtypes, shapes) is needed — external weight data is never read.
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
    /// `ModelProto.metadata_props`.
    #[prost(message, repeated, tag = "14")]
    metadata_props: Vec<ScanStringEntryProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct ScanGraphProto {
    /// `GraphProto.node`. `initializer` (tag 5) and every other field is skipped.
    #[prost(message, repeated, tag = "1")]
    node: Vec<ScanNodeProto>,
    /// `GraphProto.name`.
    #[prost(string, tag = "2")]
    name: String,
    /// `GraphProto.metadata_props`.
    #[prost(message, repeated, tag = "16")]
    metadata_props: Vec<ScanStringEntryProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct ScanNodeProto {
    /// `NodeProto.op_type`.
    #[prost(string, tag = "4")]
    op_type: String,
    /// `NodeProto.domain`.
    #[prost(string, tag = "7")]
    domain: String,
    /// `NodeProto.name`.
    #[prost(string, tag = "3")]
    name: String,
    /// `NodeProto.metadata_props`.
    #[prost(message, repeated, tag = "9")]
    metadata_props: Vec<ScanStringEntryProto>,
    /// `NodeProto.attribute`; only graph-valued payload fields are retained.
    #[prost(message, repeated, tag = "5")]
    attribute: Vec<ScanAttributeProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct ScanAttributeProto {
    /// `AttributeProto.name`.
    #[prost(string, tag = "1")]
    name: String,
    /// Singular `GRAPH` payload.
    #[prost(message, optional, boxed, tag = "6")]
    g: Option<Box<ScanGraphProto>>,
    /// Repeated `GRAPHS` payload.
    #[prost(message, repeated, tag = "11")]
    graphs: Vec<ScanGraphProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct ScanStringEntryProto {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
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

/// The dtype and rank of every port an ONNX graph exposes.
///
/// Used by the offline package conversion so a generated workflow states the
/// graph's real port contracts instead of a plausible guess. A guess is a lie
/// that fails at load for whichever package happens to disagree, which is
/// strictly worse than not converting.
pub fn graph_port_contracts(
    model_path: &Path,
) -> Option<std::collections::BTreeMap<String, onnx_genai_metadata::TensorContract>> {
    let graph = onnx_runtime_loader::load_model(model_path).ok()?;
    let mut contracts = std::collections::BTreeMap::new();
    for id in graph.inputs.iter().chain(graph.outputs.iter()) {
        let value = graph.value(*id);
        let Some(name) = value.name.clone() else {
            continue;
        };
        let rank = value.shape.len();
        contracts.insert(
            name,
            onnx_genai_metadata::TensorContract {
                dtype: ir_dtype_name(value.dtype).to_owned(),
                rank,
                // The graph's own symbols are its shape; restating them here
                // would be a second place they could drift.
                shape: None,
                optional: false,
                batch_layout: onnx_genai_metadata::BatchLayout::RequestAligned { axis: 0 },
                padding: Vec::new(),
            },
        );
    }
    Some(contracts)
}

#[cfg(test)]
mod metadata_hint_tests {
    use super::*;
    use onnx_std::{HintEntry, HintScope, HintSource};
    use prost::Message;

    fn node_hint(node: &str, key: &str, value: &str) -> HintEntry {
        HintEntry {
            scope: HintScope::NamedNode {
                name: node.to_string(),
            },
            source: HintSource::OnnxMetadata,
            key: key.to_string(),
            value: value.to_string(),
        }
    }

    fn model_hint(key: &str, value: &str) -> HintEntry {
        HintEntry {
            scope: HintScope::Model,
            source: HintSource::OnnxMetadata,
            key: key.to_string(),
            value: value.to_string(),
        }
    }

    #[test]
    fn lightweight_model_scan_reads_all_metadata_scopes() -> anyhow::Result<()> {
        let model = ScanModelProto {
            metadata_props: vec![ScanStringEntryProto {
                key: "onnx_runtime.memory.arena_cpu_mb".to_string(),
                value: "512".to_string(),
            }],
            graph: Some(ScanGraphProto {
                name: "decoder".to_string(),
                metadata_props: vec![ScanStringEntryProto {
                    key: "onnx_runtime.model.num_layers".to_string(),
                    value: "2".to_string(),
                }],
                node: vec![ScanNodeProto {
                    name: "attention".to_string(),
                    metadata_props: vec![
                        ScanStringEntryProto {
                            key: "onnx_runtime.device".to_string(),
                            value: "gpu:1".to_string(),
                        },
                        ScanStringEntryProto {
                            key: "onnx_runtime.layer".to_string(),
                            value: "0".to_string(),
                        },
                    ],
                    ..Default::default()
                }],
            }),
        };
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        std::fs::create_dir_all(&directory)?;
        let path = directory.join("metadata_hints_scan.onnx");
        std::fs::write(&path, model.encode_to_vec())?;

        let hints = load_model_metadata_hints(&path)?;
        std::fs::remove_file(path)?;

        assert!(hints.warnings.is_empty());
        assert_eq!(hints.model.arena_cpu_mb, Some(512));
        assert_eq!(hints.model.num_layers, Some(2));
        let node = hints.nodes.get("attention").expect("node hints");
        assert_eq!(node.device.as_deref(), Some("gpu:1"));
        assert_eq!(node.layer, Some(0));
        Ok(())
    }

    #[test]
    fn lightweight_model_scan_recurses_graph_attributes() -> anyhow::Result<()> {
        let model = ScanModelProto {
            graph: Some(ScanGraphProto {
                node: vec![ScanNodeProto {
                    name: "loop".to_string(),
                    op_type: "Loop".to_string(),
                    attribute: vec![ScanAttributeProto {
                        name: "body".to_string(),
                        g: Some(Box::new(ScanGraphProto {
                            metadata_props: vec![ScanStringEntryProto {
                                key: "onnx_runtime.model.architecture".to_string(),
                                value: "nested".to_string(),
                            }],
                            node: vec![ScanNodeProto {
                                name: "inner".to_string(),
                                metadata_props: vec![ScanStringEntryProto {
                                    key: "onnx_runtime.kernel".to_string(),
                                    value: "nested_kernel".to_string(),
                                }],
                                ..Default::default()
                            }],
                            ..Default::default()
                        })),
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        std::fs::create_dir_all(&directory)?;
        let path = directory.join("metadata_hints_subgraph_scan.onnx");
        std::fs::write(&path, model.encode_to_vec())?;

        let hints = load_model_metadata_hints(&path)?;
        std::fs::remove_file(path)?;

        assert!(hints.warnings.is_empty());
        assert_eq!(hints.model.architecture.as_deref(), Some("nested"));
        assert_eq!(
            hints
                .nodes
                .get("loop/body/inner")
                .and_then(|node| node.kernel.as_deref()),
            Some("nested_kernel")
        );
        Ok(())
    }

    #[test]
    fn lightweight_scan_keeps_slash_named_top_level_and_nested_nodes_distinct() -> anyhow::Result<()>
    {
        let nested_path = onnx_std::NodePath::root_node("owner".to_string(), 1)
            .with_attribute("body".to_string(), 0, None)
            .with_node("inner".to_string(), 0);
        let model = ScanModelProto {
            graph: Some(ScanGraphProto {
                node: vec![
                    ScanNodeProto {
                        name: "owner/body/inner".to_string(),
                        metadata_props: vec![ScanStringEntryProto {
                            key: "onnx_runtime.layer".to_string(),
                            value: "1".to_string(),
                        }],
                        ..Default::default()
                    },
                    ScanNodeProto {
                        name: "owner".to_string(),
                        attribute: vec![ScanAttributeProto {
                            name: "body".to_string(),
                            g: Some(Box::new(ScanGraphProto {
                                node: vec![ScanNodeProto {
                                    name: "inner".to_string(),
                                    metadata_props: vec![ScanStringEntryProto {
                                        key: "onnx_runtime.layer".to_string(),
                                        value: "2".to_string(),
                                    }],
                                    ..Default::default()
                                }],
                                ..Default::default()
                            })),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        };
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        std::fs::create_dir_all(&directory)?;
        let path = directory.join("metadata_hints_collision.onnx");
        std::fs::write(&path, model.encode_to_vec())?;

        let hints = load_model_metadata_hints(&path)?;
        std::fs::remove_file(path)?;

        assert_eq!(hints.nodes.len(), 2);
        assert_eq!(
            hints
                .nodes
                .get("owner/body/inner")
                .and_then(|node| node.layer),
            Some(1)
        );
        assert_eq!(
            hints
                .nodes
                .get_path(&nested_path)
                .and_then(|node| node.layer),
            Some(2)
        );
        Ok(())
    }

    #[test]
    fn lightweight_model_scan_reports_unknown_and_malformed_values() -> anyhow::Result<()> {
        let model = ScanModelProto {
            graph: Some(ScanGraphProto {
                node: vec![ScanNodeProto {
                    name: "attention".to_string(),
                    metadata_props: vec![
                        ScanStringEntryProto {
                            key: "onnx_runtime.devcie".to_string(),
                            value: "gpu".to_string(),
                        },
                        ScanStringEntryProto {
                            key: "onnx_runtime.layer".to_string(),
                            value: "first".to_string(),
                        },
                    ],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        std::fs::create_dir_all(&directory)?;
        let path = directory.join("metadata_hints_warnings.onnx");
        std::fs::write(&path, model.encode_to_vec())?;

        let hints = load_model_metadata_hints(&path)?;
        std::fs::remove_file(path)?;

        assert!(hints.warnings.iter().any(|warning| matches!(
            warning,
            MetadataWarning::UnknownKey { key, .. } if key == "onnx_runtime.devcie"
        )));
        assert!(hints.warnings.iter().any(|warning| matches!(
            warning,
            MetadataWarning::InvalidValue { key, expected, .. }
                if key == "onnx_runtime.layer" && *expected == "an integer"
        )));
        Ok(())
    }

    #[test]
    fn lightweight_scan_keeps_anonymous_top_level_nodes_distinct() -> anyhow::Result<()> {
        // Round-3 reviewer probe through the real load path: two unnamed
        // top-level nodes must each keep their hints instead of collapsing under
        // an empty-string key.
        let model = ScanModelProto {
            graph: Some(ScanGraphProto {
                node: vec![
                    ScanNodeProto {
                        name: String::new(),
                        metadata_props: vec![ScanStringEntryProto {
                            key: "onnx_runtime.layer".to_string(),
                            value: "1".to_string(),
                        }],
                        ..Default::default()
                    },
                    ScanNodeProto {
                        name: String::new(),
                        metadata_props: vec![ScanStringEntryProto {
                            key: "onnx_runtime.layer".to_string(),
                            value: "2".to_string(),
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        };
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        std::fs::create_dir_all(&directory)?;
        let path = directory.join("metadata_hints_anonymous.onnx");
        std::fs::write(&path, model.encode_to_vec())?;

        let hints = load_model_metadata_hints(&path)?;
        std::fs::remove_file(path)?;

        assert_eq!(hints.nodes.len(), 2);
        let mut layers: Vec<i64> = hints.nodes.values().filter_map(|node| node.layer).collect();
        layers.sort_unstable();
        assert_eq!(layers, vec![1, 2]);
        assert_eq!(
            hints
                .nodes
                .get_path(&onnx_std::NodePath::root_node(String::new(), 0))
                .and_then(|node| node.layer),
            Some(1)
        );
        assert_eq!(
            hints
                .nodes
                .get_path(&onnx_std::NodePath::root_node(String::new(), 1))
                .and_then(|node| node.layer),
            Some(2)
        );
        Ok(())
    }

    #[test]
    fn model_arena_hints_configure_default_governor_limits() -> anyhow::Result<()> {
        let hints = MetadataHints::scan([
            model_hint("onnx_runtime.memory.arena_gpu_mb", "4096"),
            model_hint("onnx_runtime.memory.arena_cpu_mb", "8192"),
        ]);
        let mut config = EngineConfig::default();

        apply_model_memory_hints(&mut config, &hints)?;

        assert_eq!(
            config.limits.vram_limit,
            ResourceLimit::Bytes(4096 * MEBIBYTE)
        );
        assert_eq!(
            config.limits.host_ram_limit,
            ResourceLimit::Bytes(8192 * MEBIBYTE)
        );
        Ok(())
    }

    #[test]
    fn programmatic_memory_limit_has_priority_over_model_hint() -> anyhow::Result<()> {
        let hints = MetadataHints::scan([model_hint("onnx_runtime.memory.arena_gpu_mb", "4096")]);
        let mut config = EngineConfig::default();
        config.limits.vram_limit = ResourceLimit::Bytes(1234);

        apply_model_memory_hints(&mut config, &hints)?;

        assert_eq!(config.limits.vram_limit, ResourceLimit::Bytes(1234));
        Ok(())
    }

    #[test]
    fn forced_model_device_reaches_default_session_options() -> anyhow::Result<()> {
        let hints = MetadataHints::scan([
            node_hint("attention", "onnx_runtime.device", "gpu:3"),
            node_hint("attention", "onnx_runtime.device.strength", "force"),
        ]);
        let mut options = SessionOptions::default();

        apply_model_placement_hints(&mut options, &hints, false)?;

        let provider = options.execution_providers.first().expect("provider");
        assert!(provider.caps.is_gpu());
        assert!(provider.caps.is_nvidia());
        assert_eq!(provider.caps.device_id(), Some(3));
        Ok(())
    }

    #[test]
    fn forced_model_device_cannot_override_programmatic_device() {
        let hints = MetadataHints::scan([
            node_hint("attention", "onnx_runtime.device", "gpu"),
            node_hint("attention", "onnx_runtime.device.strength", "force"),
        ]);
        let mut options =
            SessionOptions::with_execution_provider(onnx_genai_ort::ep_selection("cpu"));

        let error = apply_model_placement_hints(&mut options, &hints, true)
            .expect_err("forced mismatch must fail")
            .to_string();

        assert!(
            error.contains("force hints cannot be overridden"),
            "{error}"
        );
    }

    #[test]
    fn programmatic_device_overrides_model_preference() -> anyhow::Result<()> {
        let hints = MetadataHints::scan([node_hint("attention", "onnx_runtime.device", "gpu")]);
        let mut options =
            SessionOptions::with_execution_provider(onnx_genai_ort::ep_selection("cpu"));

        apply_model_placement_hints(&mut options, &hints, true)?;

        assert!(options.execution_providers[0].caps.is_host());
        Ok(())
    }

    #[test]
    fn unsupported_preferred_device_is_advisory() -> anyhow::Result<()> {
        let hints = MetadataHints::scan([node_hint("attention", "onnx_runtime.device", "npu")]);
        let mut options = SessionOptions::default();

        apply_model_placement_hints(&mut options, &hints, false)?;

        assert!(!options.execution_providers.is_empty());
        Ok(())
    }

    #[test]
    fn heterogeneous_forced_model_devices_fail_actionably() {
        let hints = MetadataHints::scan([
            node_hint("cpu_node", "onnx_runtime.device", "cpu"),
            node_hint("cpu_node", "onnx_runtime.device.strength", "force"),
            node_hint("gpu_node", "onnx_runtime.device", "gpu"),
            node_hint("gpu_node", "onnx_runtime.device.strength", "force"),
        ]);

        let error = apply_model_placement_hints(&mut SessionOptions::default(), &hints, false)
            .expect_err("heterogeneous force must fail")
            .to_string();

        assert!(
            error.contains("currently supports one execution provider"),
            "{error}"
        );
    }
}
