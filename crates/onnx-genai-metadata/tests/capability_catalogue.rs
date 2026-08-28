use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use onnx_genai_metadata::capabilities;

#[test]
fn runtime_catalogue_lists_every_builtin_capability() {
    let document = include_str!("../../../docs/genai/RUNTIME_CAPABILITY_CATALOGUE.md");
    let catalogue = document
        .split_once("<!-- capability-catalogue:start -->")
        .expect("capability catalogue start marker")
        .1
        .split_once("<!-- capability-catalogue:end -->")
        .expect("capability catalogue end marker")
        .0;

    let rows = catalogue
        .lines()
        .filter_map(|line| {
            let first_cell = line.split('|').nth(1)?.trim();
            first_cell
                .strip_prefix('`')
                .and_then(|value| value.strip_suffix('`'))
        })
        .collect::<Vec<_>>();
    let documented = rows.iter().copied().collect::<BTreeSet<_>>();
    let builtin = capabilities::BUILTIN
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();

    assert_eq!(
        rows.len(),
        documented.len(),
        "the capability catalogue contains duplicate identifiers"
    );
    assert_eq!(
        capabilities::BUILTIN.len(),
        builtin.len(),
        "the built-in capability source contains duplicate identifiers"
    );
    assert_eq!(
        documented, builtin,
        "update the runtime capability catalogue when the built-in vocabulary changes"
    );
}

#[test]
fn validation_cannot_bypass_the_builtin_capability_source() {
    let validation = include_str!("../src/validation.rs");
    for bypass in [
        "capabilities.insert(\"",
        "used.insert(\"",
        "supported: vec![\"",
    ] {
        assert!(
            !validation.contains(bypass),
            "validation introduced a capability literal outside capabilities::BUILTIN: {bypass}"
        );
    }
}

#[test]
fn canonical_metadata_uses_the_builtin_or_an_explicit_extension_vocabulary() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_metadata_files(
        &root.join("../../examples/inference_metadata/catalogue"),
        &mut files,
    );
    collect_metadata_files(&root.join("../../tests/fixtures"), &mut files);
    collect_metadata_files(&root.join("tests/fixtures"), &mut files);

    let builtin = capabilities::BUILTIN
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    for path in files {
        let Ok(document) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(&document) else {
            continue;
        };
        for identifier in serialized_capabilities(&value) {
            assert!(
                builtin.contains(identifier.as_str()) || identifier.contains('.'),
                "{} uses undocumented built-in-looking capability '{identifier}'; add it to the \
                 central built-in vocabulary and normative catalogue, or use a namespaced \
                 extension identifier",
                path.display()
            );
        }
    }
}

fn collect_metadata_files(directory: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_metadata_files(&path, files);
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("yaml" | "yml" | "json")
        ) {
            files.push(path);
        }
    }
}

fn serialized_capabilities(document: &serde_yaml::Value) -> Vec<String> {
    let Some(root) = document.as_mapping() else {
        return Vec::new();
    };
    let mut capabilities = sequence_strings(root.get("required_capabilities"));
    let manifest = root
        .get("pipeline")
        .and_then(serde_yaml::Value::as_mapping)
        .and_then(|pipeline| pipeline.get("workflow"))
        .and_then(serde_yaml::Value::as_mapping)
        .and_then(|workflow| workflow.get("manifest"))
        .and_then(serde_yaml::Value::as_mapping);
    if let Some(manifest) = manifest {
        capabilities.extend(sequence_strings(manifest.get("capabilities")));
    }
    capabilities
}

fn sequence_strings(value: Option<&serde_yaml::Value>) -> Vec<String> {
    value
        .and_then(serde_yaml::Value::as_sequence)
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .map(str::to_owned)
        .collect()
}
