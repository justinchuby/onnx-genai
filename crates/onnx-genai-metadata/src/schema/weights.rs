use super::*;
use std::collections::BTreeSet;

/// One immutable keyed container supplying external initializers of an ONNX
/// graph. Tensor geometry and element/packed representation come from the
/// graph and container; the runtime never casts, reshapes or guesses a mapping.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KeyedWeightFileSpec {
    /// Package-relative file path, resolved beneath the admitted package root.
    pub path: String,
    pub format: KeyedWeightFormat,
    /// Container tensor key -> ONNX initializer name. Unmapped keys retain
    /// their names; the loader rejects ambiguous matches and unmatched renames.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub renames: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
pub enum KeyedWeightFormat {
    #[serde(rename = "gguf")]
    Gguf,
    #[serde(rename = "safetensors")]
    SafeTensors,
}

/// Portable package paths have one spelling: slash-separated relative segments.
/// Lexical admission is followed by canonical containment checks when loading.
pub(crate) fn weight_artifact_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 4096
        && !path.contains(['\\', ':'])
        && !path.chars().any(char::is_control)
        && path
            .split('/')
            .all(|part| !part.is_empty() && !matches!(part, "." | ".."))
}

pub(crate) fn validate_graph_weights(
    graphs: &BTreeMap<String, Vec<KeyedWeightFileSpec>>,
    errors: &mut Vec<String>,
) {
    for (graph, files) in graphs {
        let field = format!("graph_weights[{graph:?}]");
        if !weight_artifact_path(graph) {
            errors.push(format!("{field} must name a package-relative graph artifact with normal slash-separated segments; absolute paths, traversal and control characters are not allowed"));
        }
        if files.is_empty() {
            errors.push(format!(
                "{field} requires at least one keyed weight file; remove an unused graph binding"
            ));
        }
        let mut destinations = BTreeSet::new();
        for (index, file) in files.iter().enumerate() {
            if !weight_artifact_path(&file.path) {
                errors.push(format!("{field}[{index}].path {:?} must name a package-relative weight artifact with normal slash-separated segments", file.path));
            }
            for (source, destination) in &file.renames {
                if source.is_empty() || destination.is_empty() {
                    errors.push(format!("{field}[{index}].renames requires nonempty container keys and initializer names"));
                }
                if !destinations.insert(destination) {
                    errors.push(format!("{field}[{index}].renames maps initializer {destination:?} more than once; supply exactly one explicit source for each initializer"));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn graph_weights_require_their_version_and_validate_declared_paths() {
        let declaration = serde_json::json!({"decoder.onnx":[{"path":"weights.gguf","format":"gguf","renames":{"tensor":"projection"}}]});
        let metadata = crate::parse_metadata_json(
            &serde_json::json!({"schema_version":"v1.17","graph_weights":declaration}),
        )
        .unwrap();
        crate::validate_metadata(&metadata).unwrap();
        let error = crate::parse_metadata_json(
            &serde_json::json!({"schema_version":"v1.16","graph_weights":declaration}),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("graph_weights") && error.to_string().contains("v1.17"),
            "{error}"
        );
        let bad = crate::parse_metadata_json(&serde_json::json!({"schema_version":"v1.17","graph_weights":{"decoder.onnx":[{"path":"../weights.gguf","format":"gguf"}]}})).unwrap();
        let errors = crate::validate_metadata(&bad).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("package-relative weight")),
            "{errors:?}"
        );
    }

    #[test]
    fn weight_paths_are_portable_and_cannot_escape_the_package() {
        for path in [
            "weights.gguf",
            "weights/shard-1.safetensors",
            "模型/权重.gguf",
        ] {
            assert!(weight_artifact_path(path), "{path}");
        }
        for path in [
            "",
            "/weights.gguf",
            "../weights",
            "a/../b",
            "a//b",
            "a/./b",
            "C:/weights",
            "a\\b",
            "a\0b",
        ] {
            assert!(!weight_artifact_path(path), "{path:?}");
        }
    }
    #[test]
    fn graph_weight_bindings_reject_ambiguous_explicit_destinations() {
        let file = |key: &str| KeyedWeightFileSpec {
            path: "weights.gguf".into(),
            format: KeyedWeightFormat::Gguf,
            renames: BTreeMap::from([(key.into(), "projection".into())]),
        };
        let graphs = BTreeMap::from([("decoder.onnx".into(), vec![file("left"), file("right")])]);
        let mut errors = Vec::new();
        validate_graph_weights(&graphs, &mut errors);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].contains("projection") && errors[0].contains("more than once"),
            "{errors:?}"
        );
    }
}
