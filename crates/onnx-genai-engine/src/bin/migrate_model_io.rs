//! Offline migration: rewrite a package's retired `model.io` block as the
//! canonical `pipeline.workflow`.
//!
//! This is the tool the loader's rejection error names. It is deliberately a
//! separate binary and not a load-time step: converting a package is a decision
//! its owner makes once and commits, not something a runtime does silently on
//! every start. A runtime that repaired packages in memory would be the second
//! authoritative answer the canonical rule exists to prevent — the package on
//! disk would say one thing and the runtime would execute another.
//!
//! The retired block is read as plain YAML rather than through the schema,
//! because the schema no longer has a field for it. That is the point: this
//! tool understands the old shape so nothing else has to.
//!
//! ```text
//! migrate_model_io <package-dir>...        # rewrite in place
//! migrate_model_io --check <package-dir>   # report, change nothing
//! ```

use std::path::{Path, PathBuf};

type Fallible<T> = Result<T, Box<dyn std::error::Error>>;

use onnx_genai_metadata::decoder_workflow::{DecoderFacts, decoder_workflow};
use onnx_genai_metadata::schema::DecoderAbi;

fn main() -> std::process::ExitCode {
    let mut check_only = false;
    let mut abi_path: Option<PathBuf> = None;
    let mut packages: Vec<PathBuf> = Vec::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--abi" => {
                abi_path = arguments.next().map(PathBuf::from);
                if abi_path.is_none() {
                    eprintln!("--abi needs a path\n{USAGE}");
                    return std::process::ExitCode::FAILURE;
                }
            }
            "--check" => check_only = true,
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return std::process::ExitCode::SUCCESS;
            }
            _ => packages.push(PathBuf::from(argument)),
        }
    }
    if packages.is_empty() {
        eprintln!("{USAGE}");
        return std::process::ExitCode::FAILURE;
    }

    let mut failures = 0;
    for package in &packages {
        match migrate(package, check_only, abi_path.as_deref()) {
            Ok(Outcome::Converted) => println!("converted {}", package.display()),
            Ok(Outcome::NeedsConversion) => {
                println!("needs conversion: {}", package.display());
                failures += 1;
            }
            Ok(Outcome::AlreadyCanonical) => {
                println!("already canonical: {}", package.display())
            }
            Err(error) => {
                eprintln!("{}: {error}", package.display());
                failures += 1;
            }
        }
    }
    if failures == 0 {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    }
}

const USAGE: &str = "\
migrate_model_io — rewrite a retired `model.io` block as `pipeline.workflow`

    migrate_model_io <package-dir>...            rewrite each package in place
    migrate_model_io --check <package-dir>       report what would change, write nothing
    migrate_model_io --abi <ports.yaml> <dir>    convert a package that never declared ports
";

enum Outcome {
    Converted,
    NeedsConversion,
    AlreadyCanonical,
}

fn migrate(package: &Path, check_only: bool, abi_path: Option<&Path>) -> Fallible<Outcome> {
    let path = metadata_path(package)?;
    let text = std::fs::read_to_string(&path)?;
    let mut document: serde_yaml::Value = serde_yaml::from_str(&text)?;

    if document
        .get("pipeline")
        .is_some_and(|value| !value.is_null())
    {
        return Ok(Outcome::AlreadyCanonical);
    }
    // A package that never carried the retired block still needs a workflow —
    // it was relying on the runtime guessing its ports from the graph, which is
    // the same silent second answer by another route. `--abi` is how its author
    // states the ports explicitly, once.
    let io = document
        .get("model")
        .and_then(|model| model.get("io"))
        .cloned();
    let abi: DecoderAbi = match (io, abi_path) {
        (Some(io), _) => serde_yaml::from_value(io)?,
        (None, Some(path)) => serde_yaml::from_str(&std::fs::read_to_string(path)?)?,
        (None, None) => {
            return Err(
                "this package declares neither pipeline.workflow nor a retired \
                        model.io block. If its ports were previously guessed from the ONNX \
                        graph, state them explicitly with --abi <ports.yaml>."
                    .into(),
            );
        }
    };
    if check_only {
        return Ok(Outcome::NeedsConversion);
    }

    let artifact = decoder_artifact(package)?;
    let facts = DecoderFacts {
        max_sequence_length: document
            .get("model")
            .and_then(|model| model.get("max_sequence_length"))
            .and_then(serde_yaml::Value::as_u64)
            .map(|limit| limit as usize),
        // Read from the graph, never guessed. A state tensor's rank differs by
        // cache discipline, and a contract that disagrees with the graph is
        // rejected at load — so the conversion asks the artifact rather than
        // assuming.
        // Carried across from the package's own generation defaults, so a
        // model with several end tokens keeps all of them. Losing every id but
        // the first is silent: generation simply runs past its end.
        eos_token_ids: document
            .get("generation")
            .and_then(|generation| generation.get("defaults"))
            .and_then(|defaults| defaults.get("eos_token_ids"))
            .and_then(serde_yaml::Value::as_sequence)
            .map(|ids| ids.iter().filter_map(serde_yaml::Value::as_i64).collect())
            .unwrap_or_default(),
        port_contracts: onnx_genai_engine::graph_port_contracts(
            &package_directory(package).join(&artifact),
        )
        .unwrap_or_default(),
    };
    let workflow = decoder_workflow(&abi, &artifact, &facts)?;

    // Drop the retired block and add the canonical one. Everything else in the
    // document is preserved verbatim: conversion changes how the graph ABI is
    // stated, not what else the package declares.
    if let Some(model) = document
        .get_mut("model")
        .and_then(serde_yaml::Value::as_mapping_mut)
    {
        model.remove(serde_yaml::Value::String("io".to_string()));
        if model.is_empty() {
            document
                .as_mapping_mut()
                .expect("a metadata document is a mapping")
                .remove(serde_yaml::Value::String("model".to_string()));
        }
    }
    let pipeline = serde_yaml::to_value(serde_yaml::Mapping::from_iter([(
        serde_yaml::Value::String("workflow".to_string()),
        serde_yaml::to_value(&workflow)?,
    )]))?;
    document
        .as_mapping_mut()
        .expect("a metadata document is a mapping")
        .insert(serde_yaml::Value::String("pipeline".to_string()), pipeline);

    let rendered = format!("{HEADER}{}", serde_yaml::to_string(&document)?);
    // Parse the result back through the real schema before writing. A tool that
    // emits a package the runtime rejects is worse than no tool.
    let parsed: onnx_genai_metadata::InferenceMetadata = serde_yaml::from_str(&rendered)?;
    onnx_genai_metadata::validation::validate_metadata(&parsed)
        .map_err(|errors| format!("converted package does not validate: {errors:#?}"))?;
    std::fs::write(&path, rendered)?;
    Ok(Outcome::Converted)
}

const HEADER: &str = "\
# Canonical single-decoder package.
#
# The graph ABI is declared where every package declares it: the workflow's
# component ports and roles, and the state_service group that owns the KV
# cache. A single decoder is not a special package shape — it is a workflow
# with one ONNX component and one runtime-bound token policy.
";

fn metadata_path(package: &Path) -> Fallible<PathBuf> {
    for name in ["inference_metadata.yaml", "inference_metadata.yml"] {
        let candidate = package.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    if package.is_file() {
        return Ok(package.to_path_buf());
    }
    Err(format!("no inference_metadata.yaml in {}", package.display()).into())
}

/// The ONNX artifact the converted workflow binds its decoder component to.
/// The directory a package's artifacts live in.
fn package_directory(package: &Path) -> PathBuf {
    if package.is_dir() {
        package.to_path_buf()
    } else {
        package
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

fn decoder_artifact(package: &Path) -> Fallible<String> {
    let directory = package_directory(package);
    let mut candidates: Vec<String> = std::fs::read_dir(&directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".onnx") || name.ends_with(".onnx.textproto"))
        .collect();
    candidates.sort();
    candidates.into_iter().next().ok_or_else(|| {
        format!(
            "no .onnx artifact in {}, so the converted workflow would name no graph",
            directory.display()
        )
        .into()
    })
}
