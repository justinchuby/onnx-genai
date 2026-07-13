use std::{error::Error, fs, path::PathBuf};

use onnx_genai_metadata::InferenceMetadata;
use schemars::generate::SchemaSettings;

fn main() -> Result<(), Box<dyn Error>> {
    let schema = SchemaSettings::draft2020_12()
        .into_generator()
        .into_root_schema_for::<InferenceMetadata>();
    let contents = format!("{}\n", serde_json::to_string_pretty(&schema)?);
    let path = schema_path();

    fs::create_dir_all(path.parent().expect("schema path has a parent"))?;
    fs::write(&path, contents)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn schema_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("schema/inference_metadata.schema.json")
}
