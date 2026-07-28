use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use onnx_runtime_operator_selection::{find_cpu_operator, normalize_domain};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(about = "Repository build and maintenance tasks")]
struct Cli {
    #[command(subcommand)]
    command: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Generate a deterministic operator-selection manifest from an operator list.
    OperatorManifest(ManifestArguments),
    /// Generate or consume a manifest and build a minimal CPU execution provider.
    MinimalBuild(MinimalBuildArguments),
}

#[derive(Args)]
struct OperatorInputs {
    /// Operator requirement (`OpType@opset` or `domain::OpType@opset`).
    #[arg(long = "operator")]
    operators: Vec<String>,
    /// Newline-delimited operator requirements; blank lines and `#` comments are ignored.
    #[arg(long)]
    operators_file: Option<PathBuf>,
}

#[derive(Args)]
struct ManifestArguments {
    #[command(flatten)]
    inputs: OperatorInputs,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "release-default")]
    optimizer_profile: String,
}

#[derive(Args)]
struct MinimalBuildArguments {
    /// Reuse a checked-in manifest instead of generating one from operators.
    #[arg(long, conflicts_with_all = ["operators", "operators_file"])]
    manifest: Option<PathBuf>,
    #[command(flatten)]
    inputs: OperatorInputs,
    /// Directory for the manifest and isolated Cargo target directory.
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "release-default")]
    optimizer_profile: String,
    /// Build in debug mode instead of release mode.
    #[arg(long)]
    debug: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct OperatorSelectionManifest {
    format_version: u32,
    target_ep: String,
    optimizer_profile: String,
    operators: Vec<OperatorRequirement>,
    cargo_features: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct OperatorRequirement {
    domain: String,
    op_type: String,
    opset: u64,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Task::OperatorManifest(arguments) => {
            let manifest = generate_manifest(&arguments.inputs, arguments.optimizer_profile)?;
            write_manifest(&arguments.output, &manifest)?;
            println!(
                "wrote {} operators using features: {}",
                manifest.operators.len(),
                manifest.cargo_features.join(",")
            );
        }
        Task::MinimalBuild(arguments) => minimal_build(arguments)?,
    }
    Ok(())
}

fn generate_manifest(
    inputs: &OperatorInputs,
    optimizer_profile: String,
) -> Result<OperatorSelectionManifest> {
    let mut specifications = inputs.operators.clone();
    if let Some(path) = &inputs.operators_file {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read operator list {}", path.display()))?;
        specifications.extend(
            contents
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(ToOwned::to_owned),
        );
    }
    if specifications.is_empty() {
        bail!("supply at least one --operator or --operators-file");
    }

    let mut operators = BTreeSet::new();
    let mut cargo_features = BTreeSet::new();
    for specification in specifications {
        let requirement = parse_operator_requirement(&specification)?;
        let entry = find_cpu_operator(&requirement.domain, &requirement.op_type, requirement.opset)
            .with_context(|| {
                format!(
                    "CPU operator catalog has no compatible {}::{} at opset {}",
                    requirement.domain, requirement.op_type, requirement.opset
                )
            })?;
        cargo_features.insert(entry.group.feature.to_owned());
        operators.insert(requirement);
    }

    Ok(OperatorSelectionManifest {
        format_version: 1,
        target_ep: "cpu".to_owned(),
        optimizer_profile,
        operators: operators.into_iter().collect(),
        cargo_features: cargo_features.into_iter().collect(),
    })
}

fn parse_operator_requirement(specification: &str) -> Result<OperatorRequirement> {
    let (identity, opset) = specification
        .rsplit_once('@')
        .with_context(|| format!("operator `{specification}` must end in `@opset`"))?;
    let opset = opset
        .parse::<u64>()
        .with_context(|| format!("invalid opset in operator `{specification}`"))?;
    let (domain, op_type) = identity
        .rsplit_once("::")
        .map_or(("ai.onnx", identity), |(domain, op_type)| (domain, op_type));
    if op_type.is_empty() {
        bail!("operator type is empty in `{specification}`");
    }
    Ok(OperatorRequirement {
        domain: normalize_domain(domain).to_owned(),
        op_type: op_type.to_owned(),
        opset,
    })
}

fn write_manifest(path: &Path, manifest: &OperatorSelectionManifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let serialized = toml::to_string_pretty(manifest).context("failed to serialize manifest")?;
    fs::write(path, serialized)
        .with_context(|| format!("failed to write manifest {}", path.display()))
}

fn read_manifest(path: &Path) -> Result<OperatorSelectionManifest> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read manifest {}", path.display()))?;
    let mut manifest: OperatorSelectionManifest = toml::from_str(&contents)
        .with_context(|| format!("invalid manifest {}", path.display()))?;
    if manifest.format_version != 1 || manifest.target_ep != "cpu" {
        bail!(
            "unsupported manifest format/target: version {}, target {}",
            manifest.format_version,
            manifest.target_ep
        );
    }
    manifest.operators.sort();
    manifest.operators.dedup();
    let expected_features = manifest
        .operators
        .iter()
        .map(|requirement| {
            find_cpu_operator(&requirement.domain, &requirement.op_type, requirement.opset)
                .with_context(|| {
                    format!(
                        "CPU operator catalog has no compatible {}::{} at opset {}",
                        requirement.domain, requirement.op_type, requirement.opset
                    )
                })
                .map(|entry| entry.group.feature.to_owned())
        })
        .collect::<Result<BTreeSet<_>>>()?
        .into_iter()
        .collect::<Vec<_>>();
    if manifest.cargo_features != expected_features {
        bail!(
            "manifest Cargo features {:?} do not match catalog-derived features {:?}",
            manifest.cargo_features,
            expected_features
        );
    }
    Ok(manifest)
}

fn minimal_build(arguments: MinimalBuildArguments) -> Result<()> {
    fs::create_dir_all(&arguments.output)
        .with_context(|| format!("failed to create {}", arguments.output.display()))?;
    let destination = arguments.output.join("operator-selection.toml");
    let manifest = if let Some(path) = &arguments.manifest {
        let manifest = read_manifest(path)?;
        write_manifest(&destination, &manifest)?;
        manifest
    } else {
        let manifest = generate_manifest(&arguments.inputs, arguments.optimizer_profile)?;
        write_manifest(&destination, &manifest)?;
        manifest
    };
    if manifest.cargo_features.is_empty() {
        bail!("manifest selects no Cargo operator features");
    }

    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("xtask must live directly under the workspace root")?
        .to_owned();
    let mut command = Command::new("cargo");
    command
        .current_dir(workspace)
        .arg("build")
        .arg("--package")
        .arg("onnx-runtime-ep-cpu")
        .arg("--no-default-features")
        .arg("--features")
        .arg(manifest.cargo_features.join(","))
        .arg("--target-dir")
        .arg(arguments.output.join("cargo-target"));
    if !arguments.debug {
        command.arg("--release");
    }
    let status = command.status().context("failed to invoke Cargo")?;
    if !status.success() {
        bail!("minimal Cargo build failed with {status}");
    }
    println!(
        "built {} operators using features: {}; manifest: {}",
        manifest.operators.len(),
        manifest.cargo_features.join(","),
        destination.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(operators: &[&str]) -> OperatorInputs {
        OperatorInputs {
            operators: operators.iter().map(ToString::to_string).collect(),
            operators_file: None,
        }
    }

    #[test]
    fn sample_operator_list_generates_sorted_exact_manifest() {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample-operators.txt");
        let manifest = generate_manifest(
            &OperatorInputs {
                operators: Vec::new(),
                operators_file: Some(fixture),
            },
            "release-default".to_owned(),
        )
        .unwrap();

        assert_eq!(
            manifest.cargo_features,
            vec!["ops-cnn", "ops-core", "ops-reduction"]
        );
        assert_eq!(
            manifest.operators,
            vec![
                OperatorRequirement {
                    domain: "ai.onnx".to_owned(),
                    op_type: "Conv".to_owned(),
                    opset: 21,
                },
                OperatorRequirement {
                    domain: "ai.onnx".to_owned(),
                    op_type: "MatMul".to_owned(),
                    opset: 21,
                },
                OperatorRequirement {
                    domain: "ai.onnx".to_owned(),
                    op_type: "Softmax".to_owned(),
                    opset: 13,
                },
            ]
        );
    }

    #[test]
    fn unknown_operator_is_actionable() {
        let error = generate_manifest(
            &inputs(&["ai.onnx::DefinitelyMissing@21"]),
            "release-default".to_owned(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("CPU operator catalog"));
    }
}
