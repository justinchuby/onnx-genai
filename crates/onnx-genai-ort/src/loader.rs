//! Model directory resolution for Phase 1 runtime loading.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use onnx_genai_metadata::{
    InferenceMetadata, PipelineSpec, PreprocessingSpec, SpeculatorDescriptor, detect_speculator,
    load_metadata, load_pipeline_spec,
};
use onnx_model_package::{ModelPackage, SelectionRequest, is_model_package_directory};

use crate::{
    DataType, Environment, GraphIo, GraphIoMetadata, OrtError, Result, Session, SessionOptions,
    TensorInfo, Tokenizer,
};

/// The canonical error for a model-load entry point invoked on a path that is
/// not an existing directory. Hoisted to a single definition so the message is
/// structurally one source of truth rather than three copies that merely happen
/// to match today — callers that match on this text match one string.
fn model_dir_missing_err(root: &Path) -> OrtError {
    OrtError::InvalidArgument(format!(
        "model directory does not exist: {}",
        root.display()
    ))
}

/// Resolved files needed to load a single ONNX text-generation model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDirectory {
    pub root: PathBuf,
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    /// Optional Phase 1 metadata path. Missing metadata is tolerated.
    pub metadata_path: Option<PathBuf>,
    /// Resolved compatibility configuration path, including package references.
    pub genai_config_path: Option<PathBuf>,
    /// Detected standalone speculator declaration, if present.
    pub speculator: Option<SpeculatorDescriptor>,
}

impl ModelDirectory {
    /// Resolve `decoder.onnx` (or a single `.onnx` fallback), `tokenizer.json`,
    /// and optional `inference_metadata.{yaml,yml,json}` under `root`.
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        if !root.is_dir() {
            return Err(model_dir_missing_err(root));
        }

        if is_model_package_directory(root) {
            return Self::load_package(root, &SelectionRequest::default());
        }

        Self::load_flat(root)
    }

    /// Resolve a flat model directory or select a variant from an ORT package.
    pub fn load_with_package_selection(
        root: impl AsRef<Path>,
        selection: &SelectionRequest,
    ) -> Result<Self> {
        let root = root.as_ref();
        if is_model_package_directory(root) {
            Self::load_package(root, selection)
        } else {
            Self::load_flat(root)
        }
    }

    fn load_flat(root: &Path) -> Result<Self> {
        let tokenizer_path = root.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            return Err(OrtError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("tokenizer.json not found in {}", root.display()),
            )));
        }

        let model_path = resolve_model_path(root)?;
        let metadata_path = find_metadata_path(root);
        let speculator = detect_speculator(root);
        let genai_config_path = onnx_genai_genai_config::find_in_dir(root);

        Ok(Self {
            root: root.to_path_buf(),
            model_path,
            tokenizer_path,
            metadata_path,
            genai_config_path,
            speculator,
        })
    }

    fn load_package(root: &Path, selection: &SelectionRequest) -> Result<Self> {
        let package = ModelPackage::open(root)
            .map_err(|error| OrtError::InvalidArgument(error.to_string()))?;
        package
            .validate()
            .map_err(|error| OrtError::InvalidArgument(error.to_string()))?;
        let component_name = if package.manifest().components.contains_key("model") {
            "model"
        } else if package.manifest().components.len() == 1 {
            package
                .manifest()
                .components
                .first()
                .map(|(name, _)| name.as_str())
                .expect("non-empty components validated by ModelPackage::open")
        } else {
            return Err(OrtError::InvalidArgument(
                "model package must contain a 'model' component when multiple components exist"
                    .to_string(),
            ));
        };
        let selected = package
            .select(component_name, selection)
            .map_err(|error| OrtError::InvalidArgument(error.to_string()))?;
        let tokenizer_directory = selected
            .tokenizer_directory
            .clone()
            .unwrap_or_else(|| selected.variant_directory.clone());
        let tokenizer_path = tokenizer_directory.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            return Err(OrtError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "tokenizer.json not found in resolved tokenizer directory {}",
                    tokenizer_directory.display()
                ),
            )));
        }
        let metadata_path = selected
            .inference_metadata_path
            .or_else(|| find_metadata_path(&selected.variant_directory));
        let genai_config_path = selected
            .genai_config_path
            .or_else(|| onnx_genai_genai_config::find_in_dir(&selected.variant_directory));
        let speculator = detect_speculator(&selected.variant_directory);
        Ok(Self {
            root: selected.variant_directory,
            model_path: selected.model_path,
            tokenizer_path,
            metadata_path,
            genai_config_path,
            speculator,
        })
    }
}

/// Bytes the model's weights actually occupy.
///
/// ONNX keeps large initializers in a sibling *external data* file rather than
/// inside the `.onnx` protobuf, so the graph file alone understates a model's
/// weights by orders of magnitude — a 2 GB model can have a 300 KB `.onnx`.
/// The sibling is found by the ecosystem's file-naming convention
/// (`<model>.onnx.data`, `<model>.onnx_data`, …), which is a property of the
/// ONNX format, not of any model family.
///
/// **File size is only an upper bound.** An external-data file can contain
/// regions no initializer references — most commonly when a re-export appends
/// fresh tensors without truncating the original, orphaning a prefix. One local
/// export was exactly 50% dead space, so reporting file size told the runtime it
/// had 2.00x more weights than it did, and every budget, fits/doesn't-fit and
/// residency decision derived from that number was skewed (#853). So this parses
/// the graph and sums what the initializers actually reference, falling back to
/// file size only when the graph cannot be read.
///
/// Reads the graph protobuf (a few MB even for tens of GB of weights) plus
/// directory metadata; it never opens the weight blobs themselves.
pub fn model_weight_bytes(model_path: &Path) -> u64 {
    let file_total = model_weight_file_bytes(model_path);
    let Ok(bytes) = onnx_runtime_loader::read_model_binary(model_path) else {
        return file_total;
    };
    let Ok(model) = onnx_runtime_loader::proto::decode_model(&bytes) else {
        return file_total;
    };
    let referenced = onnx_runtime_loader::weights::referenced_weight_bytes(&model);
    let graph_bytes = std::fs::metadata(model_path)
        .map(|meta| meta.len())
        .unwrap_or(0);
    // The graph file is charged whole (it carries inline payloads plus the
    // protobuf itself); only the external blobs are narrowed to what is
    // referenced.
    let total = graph_bytes.saturating_add(referenced.external);
    // A materially oversized external blob is a defective export. Paying for it
    // silently in budget decisions, disk, and page cache is worse than saying so.
    if file_total > total && total > 0 {
        let waste = file_total - total;
        if waste.saturating_mul(10) > file_total {
            tracing::warn!(
                model = %model_path.display(),
                file_bytes = file_total,
                referenced_bytes = total,
                unreferenced_bytes = waste,
                "external data contains unreferenced regions; using referenced bytes as the \
                 weight total. Repacking the export would reclaim this on disk"
            );
        }
    }
    total
}

/// On-disk size of the graph file plus its external-data siblings.
///
/// The upper bound [`model_weight_bytes`] narrows; used as its fallback when the
/// graph cannot be parsed.
fn model_weight_file_bytes(model_path: &Path) -> u64 {
    let Some(file_name) = model_path.file_name().and_then(|name| name.to_str()) else {
        return 0;
    };
    let graph_bytes = std::fs::metadata(model_path)
        .map(|meta| meta.len())
        .unwrap_or(0);
    let Some(directory) = model_path.parent() else {
        return graph_bytes;
    };
    let Ok(entries) = std::fs::read_dir(directory) else {
        return graph_bytes;
    };
    // External data is named after the graph file it belongs to, so a directory
    // holding several models still attributes each blob to the right one.
    let external: u64 = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_str()?.to_string();
            let is_external =
                name != file_name && name.starts_with(file_name) && !name.ends_with(".onnx");
            is_external.then(|| entry.metadata().ok()).flatten()
        })
        .filter(|meta| meta.is_file())
        .map(|meta| meta.len())
        .sum();
    graph_bytes.saturating_add(external)
}

/// Resolved tokenizer files for a pipeline model directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineTokenizerPaths {
    pub shared: Option<PathBuf>,
    pub per_component: BTreeMap<String, PathBuf>,
}

impl PipelineTokenizerPaths {
    /// Return a component-specific tokenizer path, falling back to the shared tokenizer.
    pub fn for_component(&self, component: &str) -> Option<&Path> {
        self.per_component
            .get(component)
            .or(self.shared.as_ref())
            .map(PathBuf::as_path)
    }
}

/// Resolved files for a generalized multi-model pipeline directory.
#[derive(Debug, Clone)]
pub struct PipelineModelDirectory {
    pub root: PathBuf,
    /// Native inference metadata path, when the package provides one.
    ///
    /// Compatibility packages synthesize typed metadata in memory and therefore
    /// leave this unset rather than mislabeling `genai_config.json` as native metadata.
    pub metadata_path: Option<PathBuf>,
    pub spec: PipelineSpec,
    /// The package's parsed inference metadata, when it ships one.
    ///
    /// Resolving the directory already reads and validates this file, so every
    /// setting it declares -- context length, chunked prefill, EOS ids,
    /// sampling defaults -- is served from here rather than re-read. A reader
    /// that re-opened the file could disagree with the spec built beside it,
    /// and a reader that swallowed the parse error would silently see a model
    /// that declares nothing.
    pub metadata: Option<InferenceMetadata>,
    /// Typed preprocessing synthesized from compatibility config or loaded natively.
    pub preprocessing: Option<PreprocessingSpec>,
    pub model_paths: BTreeMap<String, PathBuf>,
    pub tokenizer_paths: PipelineTokenizerPaths,
}

impl PipelineModelDirectory {
    /// Resolve a pipeline only when the package structurally declares one.
    ///
    /// Native metadata is authoritative. Without native metadata, a compatibility
    /// package is considered a pipeline only when it explicitly declares a
    /// recognized multi-component shape: a vision-language model (both vision and
    /// embedding components) or an encoder-decoder model (an `model.encoder`
    /// section feeding a cross-attention decoder, e.g. Whisper).
    pub fn load_if_declared(root: impl AsRef<Path>) -> Result<Option<Self>> {
        let root = root.as_ref();
        if !root.is_dir() {
            return Err(model_dir_missing_err(root));
        }
        if let Some(metadata_path) = find_metadata_path(root) {
            let metadata = load_metadata(&metadata_path)
                .map_err(|error| OrtError::InvalidArgument(error.to_string()))?;
            return if metadata.pipeline.is_some() {
                Self::load(root).map(Some)
            } else {
                Ok(None)
            };
        }
        let Some(genai_path) = onnx_genai_genai_config::find_in_dir(root) else {
            return Ok(None);
        };
        let config = onnx_genai_genai_config::load(&genai_path)
            .map_err(|error| OrtError::InvalidArgument(error.to_string()))?;
        let is_vision_language = config.model.vision.is_some() && config.model.embedding.is_some();
        // A transducer (RNN-T) also declares `model.encoder`, but it is a
        // distinct, not-yet-executable family — exclude it so it is never
        // recognized as a loadable encoder-decoder pipeline and silently
        // mis-bound with Whisper-style cross-attention bindings.
        let is_encoder_decoder = config.model.encoder.is_some() && !config.is_transducer();
        if !is_vision_language && !is_encoder_decoder {
            return Ok(None);
        }
        Self::load(root).map(Some)
    }

    /// Resolve the validated pipeline spec and all referenced model/tokenizer files.
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        if !root.is_dir() {
            return Err(model_dir_missing_err(root));
        }

        let metadata_path = find_metadata_path(root).ok_or_else(|| {
            OrtError::InvalidArgument(format!(
                "pipeline package '{}' must contain inference_metadata.yaml or \
                 inference_metadata.json with pipeline.workflow",
                root.display()
            ))
        })?;
        let spec = load_pipeline_spec(&metadata_path)
            .map_err(|err| OrtError::InvalidArgument(err.to_string()))?;
        let preprocessing = load_metadata(&metadata_path)
            .map_err(|err| OrtError::InvalidArgument(err.to_string()))?
            .preprocessing;

        let mut model_paths = BTreeMap::new();
        for (name, component) in &spec.workflow.components {
            match &component.implementation {
                onnx_genai_metadata::ComponentImplementation::Onnx { artifact } => {
                    model_paths.insert(
                        name.clone(),
                        resolve_relative_file(root, artifact, "workflow ONNX component")?,
                    );
                }
                onnx_genai_metadata::ComponentImplementation::Adapter {
                    artifact: Some(artifact),
                    ..
                } => {
                    let _ = resolve_relative_file(root, artifact, "workflow adapter artifact")?;
                }
                onnx_genai_metadata::ComponentImplementation::Adapter {
                    artifact: None, ..
                }
                | onnx_genai_metadata::ComponentImplementation::Binding => {}
            }
        }

        let shared_tokenizer = root.join("tokenizer.json");
        let tokenizer_paths = PipelineTokenizerPaths {
            shared: shared_tokenizer.is_file().then_some(shared_tokenizer),
            per_component: BTreeMap::new(),
        };

        crate::pipeline_admission::validate_pipeline_admission(&spec, &model_paths)?;

        Ok(Self {
            root: root.to_path_buf(),
            metadata_path: Some(metadata_path),
            spec,
            metadata,
            preprocessing,
            model_paths,
            tokenizer_paths,
        })
    }
}

/// Loaded ORT sessions and tokenizer assets for a pipeline model directory.
pub struct PipelineModels {
    pub sessions: BTreeMap<String, Session>,
    /// Declared graph I/O for components whose ORT [`Session`] was intentionally
    /// not built because the pipeline executes them on the native backend (an
    /// ORT session for such a component would be redundant, and a native-only
    /// operator would make ORT reject the graph at load).
    pub graph_io_metadata: BTreeMap<String, GraphIoMetadata>,
    pub tokenizers: BTreeMap<String, Tokenizer>,
    pub shared_tokenizer: Option<Tokenizer>,
    pub directory: PipelineModelDirectory,
    session_options: SessionOptions,
    _environment: Environment,
}

impl PipelineModels {
    /// Resolve and load all pipeline ONNX models using default CPU session options.
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_options(root, SessionOptions::default())
    }

    /// Resolve and load all pipeline ONNX models using caller-provided session options.
    pub fn load_with_options(root: impl AsRef<Path>, options: SessionOptions) -> Result<Self> {
        Self::load_with_ort_session_filter(root, options, |_| true)
    }

    /// Resolve and load pipeline assets, building an ORT [`Session`] only for the
    /// components for which `build_ort_session(name)` returns `true`.
    ///
    /// A component the predicate rejects is loaded as session-free
    /// [`GraphIoMetadata`] (its declared graph I/O only, read from the ONNX graph
    /// without instantiating ORT or materializing weights). This is how a
    /// pipeline whose component runs on the native backend avoids building — and
    /// having ORT reject at load — an ORT session it would never execute, while
    /// still exposing that component's I/O contract for decode resolution through
    /// the backend-neutral [`GraphIo`] seam ([`PipelineModels::graph_io`]).
    pub fn load_with_ort_session_filter(
        root: impl AsRef<Path>,
        options: SessionOptions,
        build_ort_session: impl Fn(&str) -> bool,
    ) -> Result<Self> {
        let directory = PipelineModelDirectory::load(root)?;

        let mut sessions = BTreeMap::new();
        let mut graph_io_metadata = BTreeMap::new();
        let mut environment = None;
        for (name, path) in &directory.model_paths {
            if build_ort_session(name) {
                let environment = match environment.as_ref() {
                    Some(environment) => environment,
                    None => environment.insert(Environment::new("onnx-genai-pipeline")?),
                };
                sessions.insert(
                    name.clone(),
                    Session::new(environment, path, options.clone())?,
                );
            } else {
                graph_io_metadata.insert(name.clone(), graph_io_from_model_path(path)?);
            }
        }

        let shared_tokenizer = directory
            .tokenizer_paths
            .shared
            .as_ref()
            .map(Tokenizer::from_file)
            .transpose()?;
        let tokenizers = directory
            .tokenizer_paths
            .per_component
            .iter()
            .map(|(name, path)| Ok((name.clone(), Tokenizer::from_file(path)?)))
            .collect::<Result<_>>()?;

        Ok(Self {
            sessions,
            graph_io_metadata,
            tokenizers,
            shared_tokenizer,
            directory,
            session_options: options,
            _environment: environment,
        })
    }

    /// Return a component-specific tokenizer, falling back to the shared tokenizer.
    pub fn tokenizer_for(&self, component: &str) -> Option<&Tokenizer> {
        self.tokenizers
            .get(component)
            .or(self.shared_tokenizer.as_ref())
    }

    /// Return a loaded session by component name.
    pub fn session(&self, component: &str) -> Option<&Session> {
        self.sessions.get(component)
    }

    /// Return a component's declared graph I/O through the backend-neutral
    /// [`GraphIo`] seam: the ORT session's own I/O when it was loaded on ORT,
    /// otherwise the session-free metadata captured for a native-executed
    /// component. Decode I/O, KV-layout, and state-budget resolution read the
    /// contract from here so they work identically regardless of which backend
    /// runs the component.
    pub fn graph_io(&self, component: &str) -> Option<&dyn GraphIo> {
        if let Some(session) = self.sessions.get(component) {
            return Some(session as &dyn GraphIo);
        }
        self.graph_io_metadata
            .get(component)
            .map(|graph| graph as &dyn GraphIo)
    }

    /// Environment shared by package sessions and generated execution islands.
    pub fn environment(&self) -> &Environment {
        &self._environment
    }

    /// Session options used to load package components.
    pub fn session_options(&self) -> SessionOptions {
        self.session_options.clone()
    }
}

/// Read only a component's declared graph I/O — input/output tensor names,
/// dtypes, and shapes — from an ONNX model file, without building an ORT session
/// or materializing external weights. Dynamic axes are represented as `-1`, the
/// same convention [`TensorInfo`] uses for an ORT session's declared I/O.
pub fn graph_io_from_model_path(path: &Path) -> Result<GraphIoMetadata> {
    graph_io_from_model_path_filtered(path, None, None)
}

/// Read only the named graph ports needed by a caller.
///
/// Unlike [`graph_io_from_model_path`], unrelated non-dense ports are never
/// parsed. A selected port is still validated strictly and reports its own
/// unsupported type.
pub fn graph_io_from_model_path_for_names(
    path: &Path,
    input_names: &[String],
    output_names: &[String],
) -> Result<GraphIoMetadata> {
    let inputs = input_names
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let outputs = output_names
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    graph_io_from_model_path_filtered(path, Some((&inputs, &outputs)), None)
}

/// Read declared KV pairs without inspecting unrelated graph ports.
///
/// Some exported models omit type metadata on graph outputs even though their
/// paired past inputs are fully typed. An untyped present output inherits its
/// paired past input's tensor metadata; an explicitly non-tensor present output
/// remains an unsupported KV error.
pub fn graph_io_from_model_path_for_kv_pairs(
    path: &Path,
    input_names: &[String],
    output_names: &[String],
) -> Result<GraphIoMetadata> {
    if input_names.len() != output_names.len() {
        return Err(OrtError::InvalidArgument(format!(
            "KV input/output mapping length mismatch: {} inputs, {} outputs",
            input_names.len(),
            output_names.len()
        )));
    }
    let inputs = input_names
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let outputs = output_names
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let output_fallbacks = output_names
        .iter()
        .zip(input_names)
        .map(|(output, input)| (output.as_str(), input.as_str()))
        .collect::<HashMap<_, _>>();
    graph_io_from_model_path_filtered(path, Some((&inputs, &outputs)), Some(&output_fallbacks))
}

fn graph_io_from_model_path_filtered(
    path: &Path,
    selected_names: Option<(&HashSet<&str>, &HashSet<&str>)>,
    untyped_output_fallbacks: Option<&HashMap<&str, &str>>,
) -> Result<GraphIoMetadata> {
    let bytes = model_proto_bytes(path)?;
    let model = onnx_runtime_loader::proto::decode_model(&bytes).map_err(|error| {
        OrtError::InvalidArgument(format!(
            "failed to parse ONNX model {} for graph I/O: {error}",
            path.display()
        ))
    })?;
    let graph = model.graph.as_ref().ok_or_else(|| {
        OrtError::InvalidArgument(format!("ONNX model {} has no graph", path.display()))
    })?;
    // Names that are also initializers are constants (weights), not real graph
    // inputs (invariant §3.5.3). ONNX allows an initializer to be listed in
    // `graph.input` — mandatory pre-IR-4, still legal in IR>=4 — but ORT's
    // `Session` (GetInputCount) and this repo's own native loader
    // (`onnx-runtime-loader::graph_builder`, §2) both exclude them. Mirror that
    // exclusion here so `GraphIoMetadata` never leaks weight tensors as ports:
    // a leaked fp16/fp32 MoE/attention weight would otherwise falsely trip the
    // decode float-rank>=3 native-load guard and route as a spurious port.
    let initializer_names: HashSet<&str> = graph
        .initializer
        .iter()
        .map(|initializer| initializer.name.as_str())
        .collect();
    let inputs = graph
        .input
        .iter()
        .filter(|value_info| !initializer_names.contains(value_info.name.as_str()))
        .filter(|value_info| {
            selected_names.is_none_or(|(inputs, _)| inputs.contains(value_info.name.as_str()))
        })
        .map(value_info_to_tensor_info)
        .collect::<Result<Vec<_>>>()?;
    let outputs = graph
        .output
        .iter()
        .filter(|value_info| {
            selected_names.is_none_or(|(_, outputs)| outputs.contains(value_info.name.as_str()))
        })
        .map(|value_info| {
            if value_info
                .r#type
                .as_ref()
                .and_then(|ty| ty.value.as_ref())
                .is_none()
                && let Some(input_name) = untyped_output_fallbacks
                    .and_then(|fallbacks| fallbacks.get(value_info.name.as_str()))
            {
                let mut info = inputs
                    .iter()
                    .find(|info| info.name == *input_name)
                    .cloned()
                    .ok_or_else(|| {
                        OrtError::InvalidArgument(format!(
                            "declared KV output '{}' has no type metadata and its paired input \
                             '{}' is unavailable",
                            value_info.name, input_name
                        ))
                    })?;
                info.name = value_info.name.clone();
                return Ok(info);
            }
            value_info_to_tensor_info(value_info)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(GraphIoMetadata::new(inputs, outputs))
}

/// Serialized `ModelProto` bytes for `path`, converting a git-friendly
/// `*.textproto` fixture to binary first (mirroring [`Session::new`]).
fn model_proto_bytes(path: &Path) -> Result<Vec<u8>> {
    let is_textproto = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("textproto"));
    if is_textproto {
        let text = std::fs::read_to_string(path)?;
        onnx_std::textproto::to_binary(&text).map_err(|error| {
            OrtError::InvalidArgument(format!(
                "failed to convert textproto model {}: {error}",
                path.display()
            ))
        })
    } else {
        onnx_runtime_loader::read_model_binary(path)
            .map_err(|error| OrtError::InvalidArgument(error.to_string()))
    }
}

fn value_info_to_tensor_info(
    value_info: &onnx_runtime_loader::proto::onnx::ValueInfoProto,
) -> Result<TensorInfo> {
    use onnx_runtime_loader::proto::onnx::{tensor_shape_proto, type_proto};

    let tensor_type = match value_info.r#type.as_ref().and_then(|ty| ty.value.as_ref()) {
        Some(type_proto::Value::TensorType(tensor_type)) => tensor_type,
        _ => {
            return Err(OrtError::InvalidArgument(format!(
                "graph I/O '{}' is not a dense tensor type",
                value_info.name
            )));
        }
    };
    let dtype = onnx_elem_type_to_data_type(tensor_type.elem_type, &value_info.name)?;
    let shape = tensor_type
        .shape
        .as_ref()
        .map(|shape| {
            shape
                .dim
                .iter()
                .map(|dim| match dim.value.as_ref() {
                    Some(tensor_shape_proto::dimension::Value::DimValue(value)) if *value >= 0 => {
                        *value
                    }
                    // Symbolic, unset, or negative axes are dynamic: -1.
                    _ => -1,
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(TensorInfo {
        name: value_info.name.clone(),
        dtype,
        shape,
    })
}

/// Map an ONNX `TensorProto.DataType` code to a [`DataType`], erroring on codes
/// the runtime does not model as a dense tensor element type.
fn onnx_elem_type_to_data_type(elem_type: i32, name: &str) -> Result<DataType> {
    Ok(match elem_type {
        1 => DataType::Float32,
        2 => DataType::Uint8,
        3 => DataType::Int8,
        4 => DataType::Uint16,
        5 => DataType::Int16,
        6 => DataType::Int32,
        7 => DataType::Int64,
        9 => DataType::Bool,
        10 => DataType::Float16,
        12 => DataType::Uint32,
        13 => DataType::Uint64,
        16 => DataType::BFloat16,
        17 => DataType::Float8E4M3,
        19 => DataType::Float8E5M2,
        other => {
            return Err(OrtError::InvalidArgument(format!(
                "graph I/O '{name}' uses unsupported ONNX tensor element type {other}"
            )));
        }
    })
}

fn resolve_model_path(root: &Path) -> Result<PathBuf> {
    // Prefer a conventionally named decoder, in either binary or textproto form.
    for candidate in ["decoder.onnx", "decoder.onnx.textproto"] {
        let path = root.join(candidate);
        if path.is_file() {
            return Ok(path);
        }
    }

    let mut onnx_files = std::fs::read_dir(root)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && is_onnx_model_file(path))
        .collect::<Vec<_>>();
    onnx_files.sort();
    prefer_binary_onnx_twins(&mut onnx_files);

    match onnx_files.as_slice() {
        [only] => Ok(only.clone()),
        [] => Err(OrtError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no .onnx model found in {}", root.display()),
        ))),
        many => Err(OrtError::InvalidArgument(format!(
            "multiple .onnx files found in {}; expected decoder.onnx or exactly one .onnx file: {:?}",
            root.display(),
            many
        ))),
    }
}

fn prefer_binary_onnx_twins(paths: &mut Vec<PathBuf>) {
    let binary_paths = paths
        .iter()
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("onnx"))
        })
        .cloned()
        .collect::<BTreeSet<_>>();

    paths.retain(|path| {
        path.extension()
            .is_none_or(|extension| !extension.eq_ignore_ascii_case("textproto"))
            || !binary_paths.contains(&path.with_extension(""))
    });
}

/// Whether `path` names an ONNX model file: a binary `*.onnx` or a git-friendly
/// ONNX protobuf TextFormat `*.onnx.textproto`.
fn is_onnx_model_file(path: &Path) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("onnx") => true,
        Some(ext) if ext.eq_ignore_ascii_case("textproto") => path
            .file_stem()
            .and_then(|stem| Path::new(stem).extension())
            .is_some_and(|inner| inner.eq_ignore_ascii_case("onnx")),
        _ => false,
    }
}

#[cfg(test)]
mod model_package_tests {
    use super::*;

    #[test]
    fn package_directory_resolves_selected_model_and_shared_tokenizer() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../onnx-model-package/tests/fixtures/valid-package")
            .canonicalize()
            .unwrap();
        let selection = SelectionRequest {
            execution_provider: Some("CPUExecutionProvider".to_string()),
            precision: Some("float32".to_string()),
            ..Default::default()
        };
        let directory = ModelDirectory::load_with_package_selection(&root, &selection).unwrap();
        assert_eq!(directory.model_path, root.join("cpu-fp32/model.onnx"));
        assert_eq!(
            directory.tokenizer_path,
            root.join(format!(
                "shared_assets/sha256-{}/tokenizer.json",
                "a".repeat(64)
            ))
        );
    }

    #[test]
    fn flat_directory_with_unrelated_manifest_remains_backward_compatible() {
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm-scatter");
        let directory = ModelDirectory::load(&root).unwrap();
        // The fixture carries both `model.onnx` and `model.onnx.textproto`.
        // `prefer_binary_onnx_twins` resolves that pair to the binary, so this
        // expects the binary. It expected the textproto until now: the loader
        // changed deliberately and this assertion was never updated, because
        // this crate's tests do not run in CI. That is fixed in the same change
        // as this line.
        assert_eq!(directory.model_path, root.join("model.onnx"));
        assert_eq!(directory.tokenizer_path, root.join("tokenizer.json"));
    }
}

/// The package's inference-metadata sidecar, resolved by the format's own rule
/// so every loader in the workspace agrees on what counts as one.
fn find_metadata_path(root: &Path) -> Option<PathBuf> {
    onnx_genai_metadata::find_metadata_path(root)
}

fn resolve_relative_file(root: &Path, relative: &str, description: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(OrtError::InvalidArgument(format!(
            "{description} path must be relative to the model directory without '..': {relative}"
        )));
    }

    let path = root.join(relative_path);
    if path.is_file() {
        Ok(path)
    } else {
        Err(OrtError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{description} file not found: {}", path.display()),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_stem_binary_and_textproto_are_one_logical_model() {
        let binary = PathBuf::from("/models/model.onnx");
        let textproto = PathBuf::from("/models/model.onnx.textproto");
        let mut paths = vec![binary.clone(), textproto];

        prefer_binary_onnx_twins(&mut paths);

        assert_eq!(paths, vec![binary]);
    }

    fn non_dense_logits_fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-glm52-qmoe-indexshare/model.onnx")
    }

    #[test]
    fn selected_dense_kv_ignores_unrelated_non_dense_logits() {
        let inputs = vec![
            "past_key_values.0.key".to_string(),
            "past_key_values.0.value".to_string(),
        ];
        let outputs = vec!["present.0.key".to_string(), "present.0.value".to_string()];

        let graph =
            graph_io_from_model_path_for_kv_pairs(&non_dense_logits_fixture(), &inputs, &outputs)
                .expect("unrelated non-dense logits must not block selected KV geometry");

        assert_eq!(graph.inputs().len(), 2);
        assert_eq!(graph.outputs().len(), 2);
    }

    #[test]
    fn selected_non_dense_candidate_fails_explicitly() {
        let error = graph_io_from_model_path_for_names(
            &non_dense_logits_fixture(),
            &[],
            &["logits".to_string()],
        )
        .expect_err("a selected non-dense candidate must fail")
        .to_string();

        assert!(error.contains("graph I/O 'logits' is not a dense tensor type"));
    }

    #[test]
    fn explicitly_non_dense_kv_candidate_fails_instead_of_using_pair_fallback() {
        use onnx_runtime_loader::proto::onnx::{TypeProto, ValueInfoProto, type_proto};

        let candidate = ValueInfoProto {
            name: "present.0.key".to_string(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::SequenceType(Box::default())),
                ..TypeProto::default()
            }),
            ..ValueInfoProto::default()
        };

        let error = value_info_to_tensor_info(&candidate)
            .expect_err("an explicitly non-dense KV candidate must fail")
            .to_string();
        assert!(error.contains("graph I/O 'present.0.key' is not a dense tensor type"));
    }
}

#[cfg(test)]
mod weight_bytes_tests {
    use super::*;
    use std::fs;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("onnx-genai-weights-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn external_data_counts_toward_the_weight_total() {
        let dir = scratch("external");
        fs::write(dir.join("model.onnx"), vec![0_u8; 100]).unwrap();
        // The ONNX external-data convention: initializers live beside the graph.
        fs::write(dir.join("model.onnx.data"), vec![0_u8; 4_000]).unwrap();
        // Unrelated package files must not be counted as weights.
        fs::write(dir.join("tokenizer.json"), vec![0_u8; 900]).unwrap();

        assert_eq!(model_weight_bytes(&dir.join("model.onnx")), 4_100);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_second_model_in_the_directory_keeps_its_own_weights() {
        let dir = scratch("two-models");
        fs::write(dir.join("decoder.onnx"), vec![0_u8; 10]).unwrap();
        fs::write(dir.join("decoder.onnx.data"), vec![0_u8; 1_000]).unwrap();
        fs::write(dir.join("encoder.onnx"), vec![0_u8; 20]).unwrap();
        fs::write(dir.join("encoder.onnx.data"), vec![0_u8; 2_000]).unwrap();

        assert_eq!(model_weight_bytes(&dir.join("decoder.onnx")), 1_010);
        assert_eq!(model_weight_bytes(&dir.join("encoder.onnx")), 2_020);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unreferenced_external_regions_are_excluded_from_the_weight_total() {
        let dir = scratch("dead-external-prefix");
        // A minimal ONNX graph with one external initializer of 1,024 bytes at
        // offset 4,096, hand-encoded so the test owns the exact layout: the
        // first 4 KiB of the blob is orphaned, which is the shape of a re-export
        // that appended fresh tensors without truncating the original (#853).
        //
        // Wire format, all length-delimited (tag = field << 3 | 2):
        //   ModelProto.graph = 7, GraphProto.initializer = 5,
        //   TensorProto: dims = 1 (varint), data_type = 2 (varint), name = 8,
        //                external_data = 13, data_location = 14 (varint),
        //   StringStringEntryProto: key = 1, value = 2.
        fn len_delim(field: u32, payload: &[u8]) -> Vec<u8> {
            let mut out = vec![u8::try_from((field << 3) | 2).unwrap()];
            let mut len = payload.len();
            loop {
                let mut byte = u8::try_from(len & 0x7f).unwrap();
                len >>= 7;
                if len > 0 {
                    byte |= 0x80;
                }
                out.push(byte);
                if len == 0 {
                    break;
                }
            }
            out.extend_from_slice(payload);
            out
        }
        fn entry(key: &str, value: &str) -> Vec<u8> {
            let mut out = len_delim(1, key.as_bytes());
            out.extend(len_delim(2, value.as_bytes()));
            out
        }

        let mut tensor = vec![0x08, 0x80, 0x04]; // dims: 512
        tensor.extend([0x10, 0x0a]); // data_type: 10 (FLOAT16) -> 512 * 2 = 1,024
        tensor.extend(len_delim(8, b"w")); // name
        tensor.extend(len_delim(13, &entry("location", "model.onnx.data")));
        tensor.extend(len_delim(13, &entry("offset", "4096")));
        tensor.extend(len_delim(13, &entry("length", "1024")));
        tensor.extend([0x70, 0x01]); // data_location: EXTERNAL
        let graph = len_delim(5, &tensor);
        let graph_bytes = len_delim(7, &graph);
        let graph_len = graph_bytes.len() as u64;

        fs::write(dir.join("model.onnx"), &graph_bytes).unwrap();
        // 5,120 bytes on disk, of which only the last 1,024 are referenced.
        fs::write(dir.join("model.onnx.data"), vec![0_u8; 5_120]).unwrap();

        assert_eq!(
            model_weight_bytes(&dir.join("model.onnx")),
            graph_len + 1_024,
            "the 4 KiB orphaned prefix must not be charged as weights"
        );
        assert_eq!(
            model_weight_file_bytes(&dir.join("model.onnx")),
            graph_len + 5_120,
            "the file-size upper bound still sees the whole blob"
        );

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_self_contained_model_reports_its_own_size() {
        let dir = scratch("self-contained");
        fs::write(dir.join("model.onnx"), vec![0_u8; 512]).unwrap();

        assert_eq!(model_weight_bytes(&dir.join("model.onnx")), 512);
        // A missing file is zero rather than an error: the reservation is an
        // input to budgeting, not a correctness gate.
        assert_eq!(model_weight_bytes(&dir.join("absent.onnx")), 0);

        fs::remove_dir_all(dir).unwrap();
    }
}
