use std::{error::Error, fs, path::PathBuf};

use onnx_genai_metadata::extensions::extension_registry_markdown;

fn main() -> Result<(), Box<dyn Error>> {
    let path = registry_path();
    fs::write(&path, extension_registry_markdown())?;
    println!("wrote {}", path.display());
    Ok(())
}

fn registry_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("docs/genai/METADATA_EXTENSION_REGISTRY.md")
}
