use std::{fs, path::PathBuf};

use onnx_genai_metadata::inference_metadata_schema_json;

#[test]
fn committed_inference_metadata_schema_is_current() {
    let generated = format!(
        "{}\n",
        inference_metadata_schema_json().expect("schema serializes")
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
