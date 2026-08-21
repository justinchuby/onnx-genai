use std::{fs, path::PathBuf};

use onnx_genai_metadata::{ComponentImplementation, InferenceMetadata, validate_metadata};

#[test]
fn every_catalogue_example_parses_and_validates() {
    let catalogue = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("examples/inference_metadata/catalogue");
    let mut examples = fs::read_dir(&catalogue)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", catalogue.display()))
        .map(|entry| entry.expect("catalogue entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "yaml")
        })
        .collect::<Vec<_>>();
    examples.sort();

    assert_eq!(
        examples.len(),
        20,
        "catalogue must cover all requested cases"
    );
    for path in examples {
        let document = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let metadata = serde_yaml::from_str::<InferenceMetadata>(&document)
            .unwrap_or_else(|error| panic!("{} did not parse: {error}", path.display()));
        validate_metadata(&metadata).unwrap_or_else(|errors| {
            panic!(
                "{} did not validate:\n{}",
                path.display(),
                errors.join("\n")
            )
        });

        let workflow = &metadata
            .pipeline
            .expect("catalogue example has a workflow")
            .workflow;
        if path
            .file_name()
            .is_some_and(|name| name == std::ffi::OsStr::new("01-gemma4-text-decoder.yaml"))
        {
            let full_key = workflow.state["full_key"]
                .contract
                .shape
                .as_ref()
                .expect("full key shape");
            let full_value = workflow.state["full_value"]
                .contract
                .shape
                .as_ref()
                .expect("full value shape");
            let sliding_key = workflow.state["sliding_key"]
                .contract
                .shape
                .as_ref()
                .expect("sliding key shape");
            assert_ne!(
                full_key[1], full_value[1],
                "K and V head geometry must remain independently representable"
            );
            assert_ne!(
                full_key[1], sliding_key[1],
                "different layers must not inherit one package-wide KV head count"
            );
        }
        for (name, component) in &workflow.components {
            if let ComponentImplementation::Onnx { artifact } = &component.implementation {
                assert!(
                    artifact.ends_with(".onnx.textproto"),
                    "{} component '{name}' must reference a textproto graph, got '{artifact}'",
                    path.display()
                );
            }
        }
    }
}
