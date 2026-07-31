//! Multi-model configuration: file-based config and directory scanning.
//!
//! # Config file (TOML example)
//! ```toml
//! [[models]]
//! id    = "my-llm"
//! path  = "/models/my-llm"
//! eager = true
//! warmup = true
//!
//! [[models]]
//! id    = "my-embedder"
//! path  = "/models/my-embedder"
//! # eager defaults to true; set false to declare the model without loading it at
//! # startup — it is lazily loaded on first request.
//! eager = false
//! ```
//!
//! # Config file (JSON equivalent)
//! ```json
//! { "models": [
//!     { "id": "my-llm", "path": "/models/my-llm" },
//!     { "id": "my-embedder", "path": "/models/my-embedder", "eager": false }
//! ]}
//! ```
//!
//! # `--models-dir` scanning
//! Every immediate subdirectory of the given directory that looks like a model
//! directory (contains `tokenizer.json`, `model.onnx`, or `genai_config.json`)
//! is registered as one spec with `id = <directory name>` and `eager = true`.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}

/// A single model declaration: where to find it and whether to load it eagerly.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSpec {
    /// User-defined identifier; used as the routing key in API requests (`"model"` field).
    pub id: String,
    /// Filesystem path to the model directory.
    pub path: PathBuf,
    /// If `true` (the default), the model is loaded immediately at server startup.
    /// If `false`, the model is recorded as available but not loaded; it is lazily
    /// loaded on the first request that routes to it (or via the admin load endpoint).
    #[serde(default = "default_true")]
    pub eager: bool,
    /// If `true`, run a small generation after each load to initialize lazy
    /// runtime allocations before the first user request. Defaults to `false`.
    #[serde(default)]
    pub warmup: bool,
}

/// Top-level structure for a TOML or JSON multi-model config file.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelsConfig {
    pub models: Vec<ModelSpec>,
}

impl ModelsConfig {
    /// Load and validate a config file.  The format is detected by file extension:
    /// `.toml` → TOML, `.json` → JSON.  Any other extension is an error.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            anyhow::anyhow!("failed to read config file '{}': {}", path.display(), e)
        })?;
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let config: ModelsConfig = match ext.as_str() {
            "toml" => toml::from_str(&content)
                .map_err(|e| anyhow::anyhow!("invalid TOML in '{}': {}", path.display(), e))?,
            "json" => serde_json::from_str(&content)
                .map_err(|e| anyhow::anyhow!("invalid JSON in '{}': {}", path.display(), e))?,
            other => anyhow::bail!(
                "unsupported config file format '.{other}': expected '.toml' or '.json'"
            ),
        };
        config.validate()
    }

    fn validate(self) -> anyhow::Result<Self> {
        if self.models.is_empty() {
            anyhow::bail!("models config must contain at least one entry");
        }
        for (i, spec) in self.models.iter().enumerate() {
            if spec.id.trim().is_empty() {
                anyhow::bail!("models[{i}]: id must not be empty or whitespace-only");
            }
            if spec.path.as_os_str().is_empty() {
                anyhow::bail!("models[{}] (id='{}'): path must not be empty", i, spec.id);
            }
        }
        let mut seen = HashSet::new();
        for spec in &self.models {
            if !seen.insert(spec.id.as_str()) {
                anyhow::bail!("duplicate model id: '{}'", spec.id);
            }
        }
        Ok(self)
    }
}

/// Build a `Vec<ModelSpec>` from a `--models-dir` path.
///
/// Every immediate subdirectory of `dir` that `ModelDirectory::load` accepts
/// yields one spec.  The spec `id` is the directory name, and `eager = true`.
/// Results are sorted by id for determinism.
///
/// Admission is delegated to the loader ON PURPOSE, and this is the whole point
/// of the function.  A scanner that decides "this is a model directory" by its
/// own criterion will eventually disagree with the component that actually
/// opens the directory, and when it does, the user is told their model is fine
/// and then told it cannot be loaded -- two contradictory messages about one
/// directory, neither naming the real cause.  The only validator that cannot
/// drift from the loader is the loader.
///
/// Returns an error if `dir` is not readable or no model directories are found.
pub fn from_models_dir(dir: &Path) -> anyhow::Result<Vec<ModelSpec>> {
    let mut specs = Vec::new();
    let mut rejected: Vec<String> = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| {
        anyhow::anyhow!("failed to read models directory '{}': {}", dir.display(), e)
    })?;
    for entry in entries {
        let entry = entry
            .map_err(|e| anyhow::anyhow!("directory entry error in '{}': {}", dir.display(), e))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        // Ask the loader. Not a cheap lookalike of it.
        if let Err(why) = onnx_genai_ort::ModelDirectory::load(&path) {
            // A skipped directory used to vanish without a word, so a model
            // that was ALMOST right -- one missing file, one misnamed weight --
            // was indistinguishable from a directory nobody intended to serve.
            // Keep the loader's reason and show it if the scan finds nothing:
            // the near-misses are the only entries a user can act on.
            rejected.push(format!("  {}: {}", path.display(), why));
            continue;
        }
        let id = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        if id.is_empty() {
            continue;
        }
        specs.push(ModelSpec {
            id,
            path,
            eager: true,
            warmup: false,
        });
    }
    specs.sort_by(|a, b| a.id.cmp(&b.id));
    if specs.is_empty() {
        rejected.sort();
        if rejected.is_empty() {
            anyhow::bail!(
                "no model directories found in '{}': no subdirectories were present",
                dir.display(),
            );
        } else {
            anyhow::bail!(
                "no model directories found in '{}'. Each candidate was rejected by the model loader:\n{}",
                dir.display(),
                rejected.join("\n")
            );
        }
    }
    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(dir: &Path, name: &str, content: &str) {
        let path = dir.join(name);
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    fn make_model_subdir(parent: &Path, name: &str) {
        let d = parent.join(name);
        std::fs::create_dir_all(&d).unwrap();
        // Both files, because the LOADER requires both. The fixture previously
        // wrote only tokenizer.json and passed, which is precisely the drift
        // this module now forbids: a test fixture is itself a claim about what
        // counts as a model directory, and it was the fourth such claim.
        write_file(&d, "tokenizer.json", r#"{"version":"1.0"}"#);
        write_file(&d, "model.onnx", "not a real graph, but a real file");
    }

    #[test]
    fn parse_toml_minimal() {
        let toml = r#"
[[models]]
id   = "llm-a"
path = "/models/a"
"#;
        let config: ModelsConfig = toml::from_str(toml).unwrap();
        let config = config.validate().unwrap();
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.models[0].id, "llm-a");
        assert_eq!(config.models[0].path, PathBuf::from("/models/a"));
        assert!(config.models[0].eager, "eager defaults to true");
        assert!(!config.models[0].warmup, "warmup defaults to false");
    }

    #[test]
    fn parse_toml_with_eager_false() {
        let toml = r#"
[[models]]
id    = "llm-a"
path  = "/models/a"
eager = false
"#;
        let config: ModelsConfig = toml::from_str(toml).unwrap();
        let config = config.validate().unwrap();
        assert!(!config.models[0].eager);
    }

    #[test]
    fn parse_toml_with_warmup() {
        let toml = r#"
[[models]]
id     = "llm-a"
path   = "/models/a"
warmup = true
"#;
        let config: ModelsConfig = toml::from_str(toml).unwrap();
        let config = config.validate().unwrap();
        assert!(config.models[0].warmup);
    }

    #[test]
    fn parse_json_minimal() {
        let json = r#"{"models":[{"id":"my-model","path":"/models/x"}]}"#;
        let config: ModelsConfig = serde_json::from_str(json).unwrap();
        let config = config.validate().unwrap();
        assert_eq!(config.models[0].id, "my-model");
        assert!(config.models[0].eager);
        assert!(!config.models[0].warmup);
    }

    #[test]
    fn parse_json_eager_false() {
        let json = r#"{"models":[{"id":"m","path":"/p","eager":false}]}"#;
        let config: ModelsConfig = serde_json::from_str(json).unwrap();
        let config = config.validate().unwrap();
        assert!(!config.models[0].eager);
    }

    #[test]
    fn parse_json_warmup() {
        let json = r#"{"models":[{"id":"m","path":"/p","warmup":true}]}"#;
        let config: ModelsConfig = serde_json::from_str(json).unwrap();
        let config = config.validate().unwrap();
        assert!(config.models[0].warmup);
    }

    #[test]
    fn duplicate_id_is_rejected() {
        let toml = r#"
[[models]]
id   = "dup"
path = "/a"

[[models]]
id   = "dup"
path = "/b"
"#;
        let config: ModelsConfig = toml::from_str(toml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(
            err.to_string().contains("duplicate model id"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn empty_models_list_is_rejected() {
        let toml = r#"models = []"#;
        let config: ModelsConfig = toml::from_str(toml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("at least one entry"));
    }

    #[test]
    fn empty_id_is_rejected() {
        let toml = r#"
[[models]]
id   = ""
path = "/a"
"#;
        let config: ModelsConfig = toml::from_str(toml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("id must not be empty"));
    }

    #[test]
    fn whitespace_id_is_rejected() {
        let toml = r#"
[[models]]
id   = "   "
path = "/a"
"#;
        let config: ModelsConfig = toml::from_str(toml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("id must not be empty"));
    }

    #[test]
    fn from_file_detects_toml_by_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        write_file(
            dir.path(),
            "config.toml",
            r#"[[models]]
id   = "x"
path = "/x"
"#,
        );
        let config = ModelsConfig::from_file(&path).unwrap();
        assert_eq!(config.models[0].id, "x");
    }

    #[test]
    fn from_file_detects_json_by_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        write_file(
            dir.path(),
            "config.json",
            r#"{"models":[{"id":"y","path":"/y"}]}"#,
        );
        let config = ModelsConfig::from_file(&path).unwrap();
        assert_eq!(config.models[0].id, "y");
    }

    #[test]
    fn from_file_rejects_unknown_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        write_file(dir.path(), "config.yaml", "models: []");
        let err = ModelsConfig::from_file(&path).unwrap_err();
        assert!(err.to_string().contains("unsupported config file format"));
    }

    #[test]
    fn models_dir_scan_returns_sorted_specs() {
        let dir = tempfile::tempdir().unwrap();
        make_model_subdir(dir.path(), "model-b");
        make_model_subdir(dir.path(), "model-a");
        // A non-model dir should be ignored
        std::fs::create_dir(dir.path().join("not-a-model")).unwrap();
        // A file should be ignored
        write_file(dir.path(), "readme.txt", "hello");

        let specs = from_models_dir(dir.path()).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].id, "model-a");
        assert_eq!(specs[1].id, "model-b");
        assert!(specs[0].eager);
        assert_eq!(specs[0].path, dir.path().join("model-a"));
    }

    #[test]
    fn models_dir_scan_empty_dir_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = from_models_dir(dir.path()).unwrap_err();
        assert!(err.to_string().contains("no model directories found"));
    }

    #[test]
    fn models_dir_scan_requires_model_marker_file() {
        let dir = tempfile::tempdir().unwrap();
        // A dir without any of the marker files is not a model dir
        std::fs::create_dir(dir.path().join("empty-subdir")).unwrap();
        let err = from_models_dir(dir.path()).unwrap_err();
        assert!(err.to_string().contains("no model directories found"));
    }

    /// Weights alone are not a model directory.
    ///
    /// This is not hypothetical: a `mobilenetv2/` directory containing exactly
    /// one `model.onnx` and no tokenizer sits in the shared models directory
    /// this project is developed against. The previous OR-of-markers heuristic
    /// ADMITTED it, so a vision model was registered as a text-generation model
    /// and failed later, at load, with an error naming the wrong cause.
    #[test]
    fn models_dir_scan_rejects_weights_without_a_tokenizer() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path().join("mobilenetv2");
        std::fs::create_dir_all(&d).unwrap();
        write_file(&d, "model.onnx", "weights, but nothing to tokenise with");

        let err = from_models_dir(dir.path()).unwrap_err();
        assert!(err.to_string().contains("no model directories found"));
        // and the rejection must say WHY, naming the directory it rejected --
        // a silent skip is what made this class of mistake unreportable.
        assert!(
            err.to_string().contains("mobilenetv2"),
            "rejection must name the directory it skipped, got: {err}"
        );
        assert!(
            err.to_string().contains("tokenizer.json"),
            "rejection must carry the loader's reason, got: {err}"
        );
    }
}
