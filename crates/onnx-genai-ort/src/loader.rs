//! Model directory resolution for Phase 1 runtime loading.

use std::path::{Path, PathBuf};

use crate::{OrtError, Result};

/// Resolved files needed to load a single ONNX text-generation model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDirectory {
    pub root: PathBuf,
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    /// Optional Phase 1 metadata path. Missing metadata is tolerated.
    pub metadata_path: Option<PathBuf>,
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

        Ok(Self {
            root: root.to_path_buf(),
            model_path,
            tokenizer_path,
            metadata_path,
        })
    }
}

fn resolve_model_path(root: &Path) -> Result<PathBuf> {
    let decoder = root.join("decoder.onnx");
    if decoder.is_file() {
        return Ok(decoder);
    }

    let mut onnx_files = std::fs::read_dir(root)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("onnx"))
        })
        .collect::<Vec<_>>();
    onnx_files.sort();

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
