//! Resolve package-authored keyed containers once, before loading any session.
//! Consumers receive canonical graph/file bindings, never unvalidated paths.
use anyhow::{Context, Result, ensure};
use onnx_genai_metadata::{InferenceMetadata, KeyedWeightFormat};
use onnx_runtime_session::KeyedWeightFile;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

#[derive(Default)]
pub(crate) struct GraphWeightFiles {
    graphs: BTreeMap<PathBuf, GraphFiles>,
}

pub(crate) struct GraphFiles {
    pub(crate) containers: Vec<KeyedWeightFile>,
    /// Actual immutable source files used for Native state identity. Repeated
    /// bindings to the same file retain their mappings but hash its bytes once.
    pub(crate) sources: BTreeSet<PathBuf>,
}

impl GraphWeightFiles {
    pub(crate) fn resolve(root: &Path, metadata: &InferenceMetadata) -> Result<Self> {
        if metadata.graph_weights.is_empty() {
            return Ok(Self::default());
        }
        if let Err(errors) = onnx_genai_metadata::validate_metadata(metadata) {
            anyhow::bail!(
                "validate graph_weights metadata in '{}': {}",
                root.display(),
                errors.join("; ")
            );
        }
        let root = root
            .canonicalize()
            .with_context(|| format!("resolve graph_weights package root '{}'", root.display()))?;
        let mut graphs = BTreeMap::new();
        for (graph, declarations) in &metadata.graph_weights {
            let graph_path = artifact(&root, graph, &format!("graph_weights[{graph:?}]"))?;
            let mut containers = Vec::with_capacity(declarations.len());
            let mut sources = BTreeSet::new();
            for (index, declaration) in declarations.iter().enumerate() {
                let path = artifact(
                    &root,
                    &declaration.path,
                    &format!("graph_weights[{graph:?}][{index}].path"),
                )?;
                sources.insert(path.clone());
                let mut file = match declaration.format {
                    KeyedWeightFormat::Gguf => KeyedWeightFile::gguf(path),
                    KeyedWeightFormat::SafeTensors => KeyedWeightFile::safetensors(path),
                };
                for (key, initializer) in &declaration.renames {
                    file = file.rename(key, initializer);
                }
                containers.push(file);
            }
            ensure!(
                graphs
                    .insert(
                        graph_path,
                        GraphFiles {
                            containers,
                            sources
                        }
                    )
                    .is_none(),
                "graph_weights[{graph:?}] aliases another declared graph artifact; bind each canonical graph once instead of choosing an ambiguous container set"
            );
        }
        Ok(Self { graphs })
    }

    pub(crate) fn sources(&self) -> BTreeSet<&Path> {
        self.graphs
            .values()
            .flat_map(|graph| graph.sources.iter().map(PathBuf::as_path))
            .collect()
    }

    pub(crate) fn for_graph(&self, graph: &Path) -> Result<Option<&GraphFiles>> {
        if self.graphs.is_empty() {
            return Ok(None);
        }
        let canonical = graph.canonicalize().with_context(|| {
            format!(
                "resolve graph '{}' for its graph_weights binding",
                graph.display()
            )
        })?;
        Ok(self.graphs.get(&canonical))
    }
}

fn artifact(root: &Path, path: &str, field: &str) -> Result<PathBuf> {
    let resolved = root.join(path).canonicalize().with_context(|| {
        format!(
            "resolve {field} artifact {path:?} beneath '{}'; restore the declared package file",
            root.display()
        )
    })?;
    ensure!(
        resolved.starts_with(root),
        "{field} artifact {path:?} resolves outside package root '{}'; use an immutable file beneath the admitted package",
        root.display()
    );
    ensure!(
        resolved.is_file(),
        "{field} artifact {path:?} is not a regular file; supply the declared immutable graph or weight container"
    );
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn resolver_checks_files_and_canonical_graph_aliases_before_session_loading() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("model.onnx"), b"graph").unwrap();
        std::fs::write(root.path().join("weights.gguf"), b"container").unwrap();
        let metadata = onnx_genai_metadata::parse_metadata_json(&json!({"schema_version":"v1.17","graph_weights":{"model.onnx":[{"path":"weights.gguf","format":"gguf"}]}})).unwrap();
        let resolved = GraphWeightFiles::resolve(root.path(), &metadata).unwrap();
        let files = resolved
            .for_graph(&root.path().join("model.onnx"))
            .unwrap()
            .unwrap();
        assert_eq!(files.containers.len(), 1);
        assert_eq!(files.sources.len(), 1);
        std::fs::remove_file(root.path().join("weights.gguf")).unwrap();
        let error = GraphWeightFiles::resolve(root.path(), &metadata)
            .err()
            .unwrap();
        assert!(format!("{error:#}").contains("weights.gguf"), "{error:#}");
    }
    #[cfg(unix)]
    #[test]
    fn resolver_refuses_symlink_escapes_and_duplicate_canonical_graph_bindings() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("model.onnx"), b"graph").unwrap();
        std::fs::write(outside.path().join("weights.gguf"), b"container").unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("weights.gguf"),
            root.path().join("weights.gguf"),
        )
        .unwrap();
        let mut value = json!({"schema_version":"v1.17","graph_weights":{"model.onnx":[{"path":"weights.gguf","format":"gguf"}]}});
        let metadata = onnx_genai_metadata::parse_metadata_json(&value).unwrap();
        let error = GraphWeightFiles::resolve(root.path(), &metadata)
            .err()
            .unwrap();
        assert!(
            format!("{error:#}").contains("outside package root"),
            "{error:#}"
        );
        std::fs::remove_file(root.path().join("weights.gguf")).unwrap();
        std::fs::write(root.path().join("weights.gguf"), b"container").unwrap();
        std::os::unix::fs::symlink(
            root.path().join("model.onnx"),
            root.path().join("alias.onnx"),
        )
        .unwrap();
        value["graph_weights"]["alias.onnx"] = value["graph_weights"]["model.onnx"].clone();
        let metadata = onnx_genai_metadata::parse_metadata_json(&value).unwrap();
        let error = GraphWeightFiles::resolve(root.path(), &metadata)
            .err()
            .unwrap();
        assert!(
            format!("{error:#}").contains("aliases another declared graph"),
            "{error:#}"
        );
    }
}
