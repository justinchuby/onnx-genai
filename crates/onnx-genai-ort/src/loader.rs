//! Model directory resolution for Phase 1 runtime loading.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use onnx_genai_metadata::{
    PipelineSpec, SpeculatorDescriptor, detect_speculator, load_pipeline_spec,
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
    pub metadata_path: PathBuf,
    pub spec: PipelineSpec,
    pub model_paths: BTreeMap<String, PathBuf>,
    pub tokenizer_paths: PipelineTokenizerPaths,
}

impl PipelineModelDirectory {
    /// Resolve the validated pipeline spec and all referenced model/tokenizer files.
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        if !root.is_dir() {
            return Err(OrtError::InvalidArgument(format!(
                "model directory does not exist: {}",
                root.display()
            )));
        }

        let metadata_path = resolve_metadata_path(root)?;
        let spec = load_pipeline_spec(&metadata_path)
            .map_err(|err| OrtError::InvalidArgument(err.to_string()))?;

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

        Ok(Self {
            root: root.to_path_buf(),
            metadata_path,
            spec,
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

fn resolve_metadata_path(root: &Path) -> Result<PathBuf> {
    [
        "inference_metadata.yaml",
        "inference_metadata.yml",
        "inference_metadata.json",
    ]
    .iter()
    .map(|name| root.join(name))
    .find(|path| path.is_file())
    .ok_or_else(|| {
        OrtError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("pipeline metadata not found in {}", root.display()),
        ))
    })
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
