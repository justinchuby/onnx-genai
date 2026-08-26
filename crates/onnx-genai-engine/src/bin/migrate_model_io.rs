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
    let mut reemit = false;
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
            "--reemit" => reemit = true,
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
        match migrate(package, check_only, reemit, abi_path.as_deref()) {
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
    migrate_model_io --reemit <package-dir>      rebuild an already-converted package's workflow
";

enum Outcome {
    Converted,
    NeedsConversion,
    AlreadyCanonical,
}

fn migrate(
    package: &Path,
    check_only: bool,
    reemit: bool,
    abi_path: Option<&Path>,
) -> Fallible<Outcome> {
    let path = metadata_path(package)?;
    let text = std::fs::read_to_string(&path)?;
    let mut document: serde_yaml::Value = serde_yaml::from_str(&text)?;

    let already_converted = document
        .get("pipeline")
        .is_some_and(|value| !value.is_null());
    if already_converted && !reemit {
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
    // Re-emitting reads the ABI back out of the workflow the package already
    // declares. That direction is the same one the runtime uses to resolve a
    // package's ports, and `decoder_workflow_roundtrip` pins the pair as exact,
    // so a rebuild states the ports the package always stated — only in
    // whatever form the current emitter produces.
    let reemit_abi = already_converted
        .then(|| -> Fallible<DecoderAbi> {
            let metadata = onnx_genai_metadata::load_metadata(&path)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            metadata
                .decoder_io()
                .cloned()
                .ok_or_else(|| "this package's workflow resolves no decoder ABI".into())
        })
        .transpose()?;
    let abi: DecoderAbi = match (reemit_abi, io, abi_path) {
        (Some(abi), _, _) => abi,
        (None, Some(io), _) => serde_yaml::from_value(io)?,
        (None, None, Some(path)) => serde_yaml::from_str(&std::fs::read_to_string(path)?)?,
        (None, None, None) => {
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
    let eos_token_ids = package_eos_token_ids(&document)
        .or_else(|| {
            document
                .get("tokens")
                .and_then(|tokens| tokens.get("eos_token_id"))
                .and_then(sequence_i64)
        })
        .or_else(|| {
            document
                .get("generation")
                .and_then(|generation| generation.get("defaults"))
                .and_then(|defaults| defaults.get("eos_token_ids"))
                .and_then(sequence_i64)
        })
        .or_else(|| declared_eos_token_ids(&document))
        .unwrap_or_default();
    if eos_token_ids.is_empty() {
        return Err(
            "this autoregressive package declares no EOS token ids. Add its authoritative model \
             defaults before conversion; the migrator will not guess numeric ids from tokenizer \
             assets."
                .into(),
        );
    }
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
    document.as_mapping_mut().expect("metadata mapping").insert(
        serde_yaml::Value::String("schema_version".to_string()),
        serde_yaml::Value::String(
            onnx_genai_metadata::version::TOKEN_AUTHORITY_SCHEMA_VERSION.to_string(),
        ),
    );
    let mut package = document
        .get("package")
        .and_then(serde_yaml::Value::as_mapping)
        .cloned()
        .unwrap_or_default();
    let mut tokenizer = package
        .get(serde_yaml::Value::String("tokenizer".to_string()))
        .and_then(serde_yaml::Value::as_mapping)
        .cloned()
        .unwrap_or_default();
    let mut special_tokens = tokenizer
        .get(serde_yaml::Value::String("special_tokens".to_string()))
        .and_then(serde_yaml::Value::as_mapping)
        .cloned()
        .or_else(|| {
            document
                .get("tokens")
                .and_then(serde_yaml::Value::as_mapping)
                .cloned()
        })
        .unwrap_or_default();
    special_tokens.insert(
        serde_yaml::Value::String("eos_token_id".to_string()),
        serde_yaml::to_value(
            eos_token_ids
                .iter()
                .map(|id| u32::try_from(*id))
                .collect::<Result<Vec<_>, _>>()?,
        )?,
    );
    tokenizer.insert(
        serde_yaml::Value::String("special_tokens".to_string()),
        serde_yaml::Value::Mapping(special_tokens),
    );
    package.insert(
        serde_yaml::Value::String("tokenizer".to_string()),
        serde_yaml::Value::Mapping(tokenizer),
    );
    let document_mapping = document.as_mapping_mut().expect("metadata mapping");
    document_mapping.remove(serde_yaml::Value::String("tokens".to_string()));
    document_mapping.insert(
        serde_yaml::Value::String("package".to_string()),
        serde_yaml::Value::Mapping(package),
    );
    if let Some(defaults) = document
        .get_mut("generation")
        .and_then(|generation| generation.get_mut("defaults"))
        .and_then(serde_yaml::Value::as_mapping_mut)
    {
        defaults.remove(serde_yaml::Value::String("eos_token_ids".to_string()));
    }

    let rendered = format!("{HEADER}{}", serde_yaml::to_string(&document)?);
    // Parse the result back through the real schema before writing. A tool that
    // emits a package the runtime rejects is worse than no tool.
    let parsed = onnx_genai_metadata::parse_metadata(&rendered, Some("yaml"))?;
    onnx_genai_metadata::validation::validate_metadata(&parsed)
        .map_err(|errors| format!("converted package does not validate: {errors:#?}"))?;
    std::fs::write(&path, rendered)?;
    Ok(Outcome::Converted)
}

/// End tokens an old already-converted package states in its workflow.
///
/// A re-emit must not lose them: the ids live on the workflow's own
/// `eos_token_ids` literal input. Reading it is a one-time migration fallback
/// that keeps a rebuild a rebuild rather than a quiet reset to
/// "this model has no end token".
fn declared_eos_token_ids(document: &serde_yaml::Value) -> Option<Vec<i64>> {
    let inputs = document.get("pipeline")?.get("workflow")?.get("inputs")?;
    ["package.eos_token_ids", "package.eos_ids"]
        .into_iter()
        .find_map(|name| inputs.get(name)?.get("default").and_then(sequence_i64))
}

/// Canonical package-default EOS ids.
///
/// Presence outranks every retired location even when the list is empty: an
/// empty authoritative declaration is an error, not permission to resurrect a
/// stale workflow literal during re-emission.
fn package_eos_token_ids(document: &serde_yaml::Value) -> Option<Vec<i64>> {
    document
        .get("package")?
        .get("tokenizer")?
        .get("special_tokens")?
        .get("eos_token_id")
        .and_then(sequence_i64)
}

fn sequence_i64(value: &serde_yaml::Value) -> Option<Vec<i64>> {
    Some(
        value
            .as_sequence()?
            .iter()
            .filter_map(serde_yaml::Value::as_i64)
            .collect(),
    )
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

#[cfg(test)]
mod tests {
    use super::{declared_eos_token_ids, package_eos_token_ids};

    #[test]
    fn canonical_eos_ids_preserve_order_and_multi_eos() {
        let document: serde_yaml::Value = serde_yaml::from_str(
            r#"
package:
  tokenizer:
    special_tokens:
      eos_token_id: [151643, 151645]
pipeline:
  workflow:
    inputs:
      package.eos_ids:
        default: [2]
"#,
        )
        .expect("metadata");

        assert_eq!(package_eos_token_ids(&document), Some(vec![151643, 151645]));
    }

    #[test]
    fn legacy_workflow_eos_spellings_remain_migration_fallbacks() {
        for name in ["package.eos_token_ids", "package.eos_ids"] {
            let document: serde_yaml::Value = serde_yaml::from_str(&format!(
                r#"
pipeline:
  workflow:
    inputs:
      {name}:
        default: [2, 3]
"#
            ))
            .expect("metadata");

            assert_eq!(declared_eos_token_ids(&document), Some(vec![2, 3]));
        }
    }
}
