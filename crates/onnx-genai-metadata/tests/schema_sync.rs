use std::{fs, path::PathBuf};

use onnx_genai_metadata::InferenceMetadata;
use schemars::generate::SchemaSettings;

#[test]
fn committed_inference_metadata_schema_is_current() {
    let schema = SchemaSettings::draft2020_12()
        .into_generator()
        .into_root_schema_for::<InferenceMetadata>();
    let generated = format!(
        "{}\n",
        serde_json::to_string_pretty(&schema).expect("schema serializes")
    );
    let path = schema_path();
    let committed = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "failed to read {}: {error}; regenerate it with \
             `cargo run -p onnx-genai-metadata --bin gen_schema`",
            path.display()
        )
    });

    assert_eq!(
        committed,
        generated,
        "{} is out of date; regenerate it with \
         `cargo run -p onnx-genai-metadata --bin gen_schema`",
        path.display()
    );
}

fn schema_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("schema/inference_metadata.schema.json")
}
