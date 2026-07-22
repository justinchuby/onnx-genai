//! Model directory resolution for Phase 1 runtime loading.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use onnx_genai_genai_config::{
    GraphTensorInfo, ModelGraphInfo, PipelineGraphInfo, pipeline_inference_metadata_from_dir,
};
use onnx_genai_metadata::{
    PipelineSpec, PreprocessingSpec, SpeculatorDescriptor, detect_speculator, load_metadata,
    load_pipeline_spec,
};

use crate::{Environment, OrtError, Result, Session, SessionOptions, Tokenizer};

/// Resolved files needed to load a single ONNX text-generation model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDirectory {
    pub root: PathBuf,
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    /// Optional Phase 1 metadata path. Missing metadata is tolerated.
    pub metadata_path: Option<PathBuf>,
    /// Detected standalone speculator declaration, if present.
    pub speculator: Option<SpeculatorDescriptor>,
}

impl ModelDirectory {
    /// Resolve `decoder.onnx` (or a single `.onnx` fallback), `tokenizer.json`,
    /// and optional `inference_metadata.{yaml,yml,json}` under `root`.
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        if !root.is_dir() {
            return Err(OrtError::InvalidArgument(format!(
                "model directory does not exist: {}",
                root.display()
            )));
        }

        let tokenizer_path = root.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            return Err(OrtError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("tokenizer.json not found in {}", root.display()),
            )));
        }

        let model_path = resolve_model_path(root)?;
        let metadata_path = [
            "inference_metadata.yaml",
            "inference_metadata.yml",
            "inference_metadata.json",
        ]
        .iter()
        .map(|name| root.join(name))
        .find(|path| path.is_file());
        let speculator = detect_speculator(root);

        Ok(Self {
            root: root.to_path_buf(),
            model_path,
            tokenizer_path,
            metadata_path,
            speculator,
        })
    }
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
    /// Typed preprocessing synthesized from compatibility config or loaded natively.
    pub preprocessing: Option<PreprocessingSpec>,
    pub model_paths: BTreeMap<String, PathBuf>,
    pub tokenizer_paths: PipelineTokenizerPaths,
}

impl PipelineModelDirectory {
    /// Resolve a pipeline only when the package structurally declares one.
    ///
    /// Native metadata is authoritative. Without native metadata, a compatibility
    /// package is considered a pipeline only when it explicitly declares both
    /// vision and embedding components.
    pub fn load_if_declared(root: impl AsRef<Path>) -> Result<Option<Self>> {
        let root = root.as_ref();
        if !root.is_dir() {
            return Err(OrtError::InvalidArgument(format!(
                "model directory does not exist: {}",
                root.display()
            )));
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
        if config.model.vision.is_none() || config.model.embedding.is_none() {
            return Ok(None);
        }
        Self::load(root).map(Some)
    }

    /// Resolve the validated pipeline spec and all referenced model/tokenizer files.
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        if !root.is_dir() {
            return Err(OrtError::InvalidArgument(format!(
                "model directory does not exist: {}",
                root.display()
            )));
        }

        let native_metadata_path = find_metadata_path(root);
        let (metadata_path, spec, preprocessing) = if let Some(metadata_path) = native_metadata_path
        {
            let spec = load_pipeline_spec(&metadata_path)
                .map_err(|err| OrtError::InvalidArgument(err.to_string()))?;
            let preprocessing = load_metadata(&metadata_path)
                .map_err(|err| OrtError::InvalidArgument(err.to_string()))?
                .preprocessing;
            (Some(metadata_path), spec, preprocessing)
        } else {
            load_compatibility_pipeline(root)?
        };

        let mut model_paths = BTreeMap::new();
        let mut per_component_tokenizers = BTreeMap::new();
        for (name, component) in &spec.models {
            model_paths.insert(
                name.clone(),
                resolve_relative_file(root, &component.filename, "pipeline model")?,
            );
            if let Some(tokenizer) = &component.tokenizer {
                per_component_tokenizers.insert(
                    name.clone(),
                    resolve_relative_file(root, tokenizer, "component tokenizer")?,
                );
            }
        }

        let shared_tokenizer = root.join("tokenizer.json");
        let tokenizer_paths = PipelineTokenizerPaths {
            shared: shared_tokenizer.is_file().then_some(shared_tokenizer),
            per_component: per_component_tokenizers,
        };

        crate::pipeline_admission::validate_pipeline_admission(
            &spec,
            preprocessing.as_ref(),
            &model_paths,
        )?;

        Ok(Self {
            root: root.to_path_buf(),
            metadata_path,
            spec,
            preprocessing,
            model_paths,
            tokenizer_paths,
        })
    }
}

/// Loaded ORT sessions and tokenizer assets for a pipeline model directory.
pub struct PipelineModels {
    pub sessions: BTreeMap<String, Session>,
    pub tokenizers: BTreeMap<String, Tokenizer>,
    pub shared_tokenizer: Option<Tokenizer>,
    pub directory: PipelineModelDirectory,
    _environment: Environment,
}

impl PipelineModels {
    /// Resolve and load all pipeline ONNX models using default CPU session options.
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_options(root, SessionOptions::default())
    }

    /// Resolve and load all pipeline ONNX models using caller-provided session options.
    pub fn load_with_options(root: impl AsRef<Path>, options: SessionOptions) -> Result<Self> {
        let directory = PipelineModelDirectory::load(root)?;
        let environment = Environment::new("onnx-genai-pipeline")?;

        let mut sessions = BTreeMap::new();
        for (name, path) in &directory.model_paths {
            sessions.insert(
                name.clone(),
                Session::new(&environment, path, options.clone())?,
            );
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
            tokenizers,
            shared_tokenizer,
            directory,
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

fn find_metadata_path(root: &Path) -> Option<PathBuf> {
    [
        "inference_metadata.yaml",
        "inference_metadata.yml",
        "inference_metadata.json",
    ]
    .iter()
    .map(|name| root.join(name))
    .find(|path| path.is_file())
}

fn load_compatibility_pipeline(
    root: &Path,
) -> Result<(Option<PathBuf>, PipelineSpec, Option<PreprocessingSpec>)> {
    let genai_path = onnx_genai_genai_config::find_in_dir(root).ok_or_else(|| {
        OrtError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "pipeline metadata not found in {}; expected inference_metadata.{{yaml,yml,json}} \
                 or a complete genai_config.json compatibility package",
                root.display()
            ),
        ))
    })?;
    let config = onnx_genai_genai_config::load(&genai_path)
        .map_err(|error| OrtError::InvalidArgument(error.to_string()))?;
    let vision =
        config.model.vision.as_ref().ok_or_else(|| {
            incomplete_compatibility_error(root, "model.vision in genai_config.json")
        })?;
    let embedding = config.model.embedding.as_ref().ok_or_else(|| {
        incomplete_compatibility_error(root, "model.embedding in genai_config.json")
    })?;
    let vision_filename =
        compatibility_filename(root, vision.filename.as_deref(), "model.vision.filename")?;
    let embedding_filename = compatibility_filename(
        root,
        embedding.filename.as_deref(),
        "model.embedding.filename",
    )?;
    let decoder_filename = compatibility_filename(
        root,
        config.model.decoder.filename.as_deref(),
        "model.decoder.filename",
    )?;
    let graphs = PipelineGraphInfo {
        vision: inspect_model_graph(&vision_filename, "vision")?,
        embedding: inspect_model_graph(&embedding_filename, "embedding")?,
        decoder: inspect_model_graph(&decoder_filename, "decoder")?,
    };
    let metadata = pipeline_inference_metadata_from_dir(root, &graphs)
        .map_err(|error| OrtError::InvalidArgument(error.to_string()))?
        .ok_or_else(|| {
            incomplete_compatibility_error(
                root,
                "a multimodal genai_config.json with vision, embedding, and decoder components",
            )
        })?;
    let preprocessing = metadata.preprocessing;
    let spec = metadata
        .pipeline
        .ok_or_else(|| incomplete_compatibility_error(root, "the synthesized metadata pipeline"))?;
    onnx_genai_metadata::validate_pipeline_spec(&spec)
        .map_err(|error| OrtError::InvalidArgument(error.to_string()))?;
    Ok((None, spec, preprocessing))
}

fn compatibility_filename(root: &Path, value: Option<&str>, field: &str) -> Result<PathBuf> {
    let filename = value
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| incomplete_compatibility_error(root, field))?;
    resolve_relative_file(root, filename, field)
}

fn incomplete_compatibility_error(root: &Path, missing: &str) -> OrtError {
    OrtError::InvalidArgument(format!(
        "cannot synthesize compatibility pipeline metadata for {}: missing required semantics: \
         {missing}. Why: compatibility loading uses only explicit genai_config.json, config.json, \
         processor-config, and ONNX graph facts; it never guesses from model.type or a model name. \
         How to fix: regenerate the package with native inference_metadata.json (preferred), or \
         export a complete compatibility package that declares the missing facts",
        root.display()
    ))
}

fn inspect_model_graph(path: &Path, component: &str) -> Result<ModelGraphInfo> {
    let model = onnx_std::load_model(path).map_err(|error| {
        OrtError::InvalidArgument(format!(
            "failed to inspect {component} ONNX graph at {} while synthesizing compatibility \
             pipeline metadata: {error}",
            path.display()
        ))
    })?;
    let graph = &model.graph;
    let inputs = graph
        .inputs
        .iter()
        .map(|id| graph_tensor_info(graph.value(*id), component, "input"))
        .collect::<Result<Vec<_>>>()?;
    let outputs = graph
        .outputs
        .iter()
        .map(|id| graph_tensor_info(graph.value(*id), component, "output"))
        .collect::<Result<Vec<_>>>()?;
    Ok(ModelGraphInfo { inputs, outputs })
}

fn graph_tensor_info(
    value: &onnx_std::ir::Value,
    component: &str,
    direction: &str,
) -> Result<GraphTensorInfo> {
    let name = value.name.clone().ok_or_else(|| {
        OrtError::InvalidArgument(format!(
            "{component} ONNX graph has an unnamed {direction}; compatibility loading requires \
             explicit graph-port names"
        ))
    })?;
    let dimensions = value
        .shape
        .iter()
        .map(|dimension| match dimension {
            onnx_std::ir::Dim::Static(value) => Some(*value),
            onnx_std::ir::Dim::Symbolic(_) => None,
        })
        .collect();
    Ok(GraphTensorInfo {
        name,
        dtype: graph_dtype_name(value.dtype).to_owned(),
        dimensions,
    })
}

fn graph_dtype_name(dtype: onnx_std::ir::DataType) -> &'static str {
    use onnx_std::ir::DataType;
    match dtype {
        DataType::Undefined => "undefined",
        DataType::Float32 => "float32",
        DataType::Uint8 => "uint8",
        DataType::Int8 => "int8",
        DataType::Uint16 => "uint16",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::String => "string",
        DataType::Bool => "bool",
        DataType::Float16 => "float16",
        DataType::Float64 => "float64",
        DataType::Uint32 => "uint32",
        DataType::Uint64 => "uint64",
        DataType::Complex64 => "complex64",
        DataType::Complex128 => "complex128",
        DataType::BFloat16 => "bfloat16",
        DataType::Float8E4M3FN => "float8_e4m3fn",
        DataType::Float8E4M3FNUZ => "float8_e4m3fnuz",
        DataType::Float8E5M2 => "float8_e5m2",
        DataType::Float8E5M2FNUZ => "float8_e5m2fnuz",
        DataType::Uint4 => "uint4",
        DataType::Int4 => "int4",
        DataType::Float4E2M1 => "float4_e2m1",
        DataType::Float8E8M0 => "float8_e8m0",
        DataType::Uint2 => "uint2",
        DataType::Int2 => "int2",
    }
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
}
