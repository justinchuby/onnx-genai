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
        Also report how each package classifies: a single decoder (with or
        without the decode contract), or a composite pipeline and how many graph
        components it names. This is the triage question when migrating a fleet.
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
///
/// Both layers of the shared classification are reported, because they answer
/// different halves of "will this load and how". A package can be a
/// structurally recognizable single decoder and still not name the decode step,
/// which is what a fleet migration needs to see rather than have folded into a
/// yes/no.
fn shape_of(metadata: &onnx_genai_metadata::InferenceMetadata) -> String {
    use onnx_genai_metadata::GraphCardinality::{Composite, NoGraph, SingleGraph};

    let Some(pipeline) = metadata.pipeline.as_ref() else {
        return "no workflow".to_string();
    };
    let classification = onnx_genai_metadata::classify_workflow(&pipeline.workflow);
    // "graph component", not "ONNX component": an `adapter` is an artifact the
    // package ships and something has to execute, and it counts here exactly as
    // the classification counts it. Saying "ONNX" would report three for a
    // package with one graph and two adapters, and the republishing step tells
    // a publisher to check this number against the one they expect.
    match classification.cardinality() {
        NoGraph => "workflow with no graph component".to_string(),
        SingleGraph if classification.contracted_single_decoder().is_some() => {
            "single-decoder workflow".to_string()
        }
        SingleGraph if classification.is_single_decoder() => {
            "single-decoder workflow, no decode contract".to_string()
        }
        SingleGraph if classification.decoder_evidence().contradictory() => {
            "one graph component declaring the decode contract but no port roles".to_string()
        }
        SingleGraph => "one graph component, not a decoder".to_string(),
        Composite => format!(
            "composite workflow ({} graph components)",
            classification.graph_component_count()
        ),
    }
}
