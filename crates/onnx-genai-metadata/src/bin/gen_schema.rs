use std::{error::Error, fs, path::PathBuf};

use onnx_genai_metadata::inference_metadata_schema_json;

fn main() -> Result<(), Box<dyn Error>> {
    let contents = format!("{}\n", inference_metadata_schema_json()?);
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
