//! Checked-in graph fixtures must remain reviewable ONNX TextFormat.

use std::path::Path;
use std::process::Command;

const MIN_TEXTPROTO_FIXTURES: usize = 200;
const SENTINEL_FIXTURES: &[&str] = &[
    "crates/onnx-genai-engine/tests/fixtures/model-package-cpu/cpu/model.onnx.textproto",
    "crates/onnx-genai-metadata/tests/fixtures/validator_package/identity.onnx.textproto",
    "crates/onnx-runtime-ep-cuda-plugin/tests/fixtures/unique_all_outputs/model.onnx.textproto",
    "crates/onnx-runtime-session/tests/fixtures/bert_toy/model.onnx.textproto",
    "tests/fixtures/onnx_genai_workflows/diffusion/denoiser/model.onnx.textproto",
    "tests/fixtures/tiny-llm/model.onnx.textproto",
];

fn fixture_inventory_errors(paths: &[String]) -> Vec<String> {
    let textproto_count = paths
        .iter()
        .filter(|path| path.ends_with(".textproto"))
        .count();
    let binaries = paths
        .iter()
        .filter(|path| path.ends_with(".onnx"))
        .cloned()
        .collect::<Vec<_>>();
    let missing_sentinels = SENTINEL_FIXTURES
        .iter()
        .filter(|sentinel| !paths.iter().any(|path| path == **sentinel))
        .copied()
        .collect::<Vec<_>>();

    let mut errors = Vec::new();
    if textproto_count < MIN_TEXTPROTO_FIXTURES {
        errors.push(format!(
            "tracked textproto fixture inventory collapsed to {textproto_count}; expected at least \
             {MIN_TEXTPROTO_FIXTURES}. Check the repository root and git pathspecs"
        ));
    }
    if !missing_sentinels.is_empty() {
        errors.push(format!(
            "tracked textproto fixture census is missing sentinel(s): {}. A fixture path or the \
             census root/pathspec drifted",
            missing_sentinels.join(", ")
        ));
    }
    if !binaries.is_empty() {
        errors.push(format!(
            "checked-in ONNX graphs must use *.onnx.textproto, found binary fixture(s): {}",
            binaries.join(", ")
        ));
    }
    errors
}

fn tracked_graph_fixtures(root: &Path) -> Vec<String> {
    let output = Command::new("git")
        .args(["ls-files", "*.textproto", "*.onnx"])
        .current_dir(root)
        .output()
        .expect("git must be available to audit checked-in fixtures");
    assert!(
        output.status.success(),
        "git fixture audit failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout)
        .expect("git paths are UTF-8")
        .lines()
        .filter(|path| root.join(path).is_file())
        .map(str::to_owned)
        .collect()
}

#[test]
fn checked_in_graph_fixtures_are_textproto_and_census_is_intact() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixtures = tracked_graph_fixtures(&root);
    let errors = fixture_inventory_errors(&fixtures);
    assert!(
        errors.is_empty(),
        "checked-in ONNX fixture census failed:\n  - {}",
        errors.join("\n  - ")
    );
}

fn healthy_fixture_inventory() -> Vec<String> {
    let mut paths = SENTINEL_FIXTURES
        .iter()
        .map(|path| (*path).to_owned())
        .collect::<Vec<_>>();
    paths.extend(
        (paths.len()..MIN_TEXTPROTO_FIXTURES)
            .map(|index| format!("tests/fixtures/synthetic-{index}/model.onnx.textproto")),
    );
    paths
}

#[test]
fn fixture_census_rejects_empty_enumeration() {
    assert!(
        !fixture_inventory_errors(&[]).is_empty(),
        "an empty fixture enumeration must fail closed"
    );
}

#[test]
fn fixture_census_rejects_binary_onnx() {
    let mut fixtures = healthy_fixture_inventory();
    fixtures.push("tests/fixtures/accidental/model.onnx".to_owned());
    assert!(
        fixture_inventory_errors(&fixtures)
            .iter()
            .any(|error| error.contains("binary fixture")),
        "a tracked binary .onnx fixture must be rejected"
    );
}

#[test]
fn fixture_census_rejects_sentinel_path_drift() {
    let mut fixtures = healthy_fixture_inventory();
    fixtures.retain(|path| path != SENTINEL_FIXTURES[0]);
    fixtures.push("tests/fixtures/renamed/model.onnx.textproto".to_owned());
    assert_eq!(
        fixtures
            .iter()
            .filter(|path| path.ends_with(".textproto"))
            .count(),
        MIN_TEXTPROTO_FIXTURES,
        "the mutation must preserve the count so only the sentinel detects it"
    );
    assert!(
        fixture_inventory_errors(&fixtures)
            .iter()
            .any(|error| error.contains(SENTINEL_FIXTURES[0])),
        "renaming or removing a sentinel path must fail even above the count floor"
    );
}
