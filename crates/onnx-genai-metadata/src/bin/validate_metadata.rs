use std::path::PathBuf;

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
        match onnx_genai_metadata::load_metadata_package(&input) {
            Ok(_) => println!("valid: {}", input.display()),
            Err(error) => {
                failed = true;
                eprintln!("invalid: {}: {error}", input.display());
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
