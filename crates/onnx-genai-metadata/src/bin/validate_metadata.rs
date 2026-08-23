use std::path::{Path, PathBuf};

/// Validate inference metadata, as a package or as a bare document.
///
/// Two modes, because they answer different questions:
///
/// * **package** (default) — the directory is a complete, loadable package: the
///   document is valid *and* every artifact it names is present. This is what a
///   runtime needs before it can run the model.
/// * **`--metadata-only`** — the document alone is valid. This is what a
///   *publisher* needs: a package on a model hub can be hundreds of gigabytes,
///   and requiring its weights to be present before its metadata could be
///   checked would mean nobody checks metadata before uploading it.
///
/// Both modes refuse the retired `model.io` block with the same error, so a
/// package that would fail to load fails here first.
fn main() {
    let mut metadata_only = false;
    let mut show_shape = false;
    let mut paths: Vec<PathBuf> = Vec::new();
    for argument in std::env::args_os().skip(1) {
        match argument.to_string_lossy().as_ref() {
            "--metadata-only" => metadata_only = true,
            "--shape" => show_shape = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            _ => paths.push(PathBuf::from(&argument)),
        }
    }
    if paths.is_empty() {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }

    let mut failed = false;
    for input in paths {
        let loaded = if metadata_only {
            document_of(&input)
        } else {
            onnx_genai_metadata::load_metadata_package(&input).map_err(|error| error.to_string())
        };
        match loaded {
            Ok(metadata) => {
                if show_shape {
                    println!("valid: {} [{}]", input.display(), shape_of(&metadata));
                } else {
                    println!("valid: {}", input.display());
                }
            }
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

const USAGE: &str = "\
validate_metadata — check inference metadata

    validate_metadata <metadata-file-or-package-dir> [...]
        Full package validation: the document is valid and every artifact it
        names is present.

    validate_metadata --metadata-only <metadata-file-or-package-dir> [...]
        Document validation only. Use this to check a package's metadata
        without its weights -- for example before uploading to a model hub.

    validate_metadata --shape [...]
        Also report whether each package is a single decoder or a composite
        pipeline, which is the triage question when migrating a fleet.
";

/// Load and validate the document, without requiring the artifacts beside it.
fn document_of(input: &Path) -> Result<onnx_genai_metadata::InferenceMetadata, String> {
    let path = if input.is_dir() {
        onnx_genai_metadata::find_metadata_path(input).ok_or_else(|| {
            format!(
                "no inference_metadata.yaml in {}; a package must declare its metadata",
                input.display()
            )
        })?
    } else {
        input.to_path_buf()
    };
    let metadata = onnx_genai_metadata::load_metadata(&path).map_err(|error| error.to_string())?;
    onnx_genai_metadata::validate_metadata(&metadata).map_err(|errors| errors.join("; "))?;
    Ok(metadata)
}

/// How this package will be executed, which is the migration triage question.
fn shape_of(metadata: &onnx_genai_metadata::InferenceMetadata) -> &'static str {
    match metadata.pipeline.as_ref() {
        Some(pipeline) => {
            if onnx_genai_metadata::is_single_decoder_workflow(&pipeline.workflow) {
                "single-decoder workflow"
            } else {
                "composite workflow"
            }
        }
        None => "no workflow",
    }
}
