use std::path::{Path, PathBuf};

fn metadata_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        for name in [
            "inference_metadata.yaml",
            "inference_metadata.yml",
            "inference_metadata.json",
        ] {
            let candidate = path.join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    path.to_path_buf()
}

fn main() {
    let paths = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if paths.is_empty() {
        eprintln!("usage: validate_metadata <metadata-file-or-package-dir> [...]");
        std::process::exit(2);
    }

    let mut failed = false;
    for input in paths {
        let path = metadata_path(&input);
        match onnx_genai_metadata::load_pipeline_spec(&path) {
            Ok(_) => println!("valid: {}", path.display()),
            Err(error) => {
                failed = true;
                eprintln!("invalid: {}: {error}", path.display());
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
