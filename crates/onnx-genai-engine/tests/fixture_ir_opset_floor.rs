//! IR / opset floor for the fixtures this branch owns.
//!
//! Repository requirement: a new or modified test fixture ONNX model must
//! declare **IR version >= 11** and a **default-domain opset >= 24**. This guard
//! enforces it for the artifacts PR #1723 adds, so the branch cannot reintroduce
//! an IR 8 / opset 13 model after the fact — a fixture is added once and read for
//! years, and nothing else in the build would notice the drift.
//!
//! # Why the scope is deliberately narrow
//!
//! It covers only what this PR added:
//!
//! * the committed workflow packages `gemma4_chained` and `gemma4_chained_mixed`;
//! * the models authored inline in the test sources this PR added.
//!
//! Pre-existing fixtures elsewhere in the tree are the mainline fixture-upgrade
//! PR's scope. Widening this guard to them now would make *this* branch red for
//! work it does not own, which is the fastest way to get a guard disabled. When
//! that PR lands, replace [`OWNED_PACKAGES`] / [`OWNED_SOURCES`] with a
//! repository-wide walk — the checks themselves already generalize.

use std::path::{Path, PathBuf};

/// Minimum ONNX IR version for a fixture this branch owns.
const MIN_IR_VERSION: u32 = 11;
/// Minimum default-domain opset for a fixture this branch owns.
const MIN_DEFAULT_OPSET: u32 = 24;

/// Committed fixture packages added by this PR.
const OWNED_PACKAGES: &[&str] = &[
    "onnx_genai_workflows/gemma4_chained",
    "onnx_genai_workflows/gemma4_chained_mixed",
];

/// Test sources added by this PR that author ONNX models inline.
const OWNED_SOURCES: &[&str] = &[
    "tests/native_workflow_parity.rs",
    "tests/native_workflow_smoke.rs",
    "tests/one_runtime_e2e.rs",
    "tests/gemma4_chained_workflow.rs",
    "tests/real_model_workflow_corpus.rs",
    "tests/common/chained.rs",
];

fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Collect every `*.onnx.textproto` under `root`, recursively.
fn textprotos(root: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            textprotos(&path, found);
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".onnx.textproto"))
        {
            found.push(path);
        }
    }
}

/// The declared `ir_version`, if the document states one.
///
/// ONNX treats an absent `ir_version` as 0, which is below any floor, so an
/// omission is reported rather than skipped.
fn ir_version(document: &str) -> Option<u32> {
    document.lines().find_map(|line| {
        line.trim()
            .strip_prefix("ir_version:")
            .and_then(|value| value.trim().parse().ok())
    })
}

/// Every default-domain (`""`) opset version the document imports.
///
/// Handles both the single-line `opset_import { domain: "" version: N }` form
/// used by inline test models and the multi-line block form emitted by
/// generators, because a fixture written either way must clear the same floor.
fn default_domain_opsets(document: &str) -> Vec<u32> {
    let mut versions = Vec::new();
    let mut rest = document;
    while let Some(start) = rest.find("opset_import") {
        let after = &rest[start + "opset_import".len()..];
        let Some(open) = after.find('{') else { break };
        let Some(close) = after[open..].find('}') else {
            break;
        };
        let block = &after[open + 1..open + close];
        // A default-domain import either says `domain: ""` or omits `domain`.
        let domain_is_default = match block.find("domain:") {
            Some(index) => {
                let value = block[index + "domain:".len()..].trim_start();
                value.starts_with("\"\"")
            }
            None => true,
        };
        if domain_is_default
            && let Some(index) = block.find("version:")
            && let Ok(version) = block[index + "version:".len()..]
                .split(|c: char| !c.is_ascii_digit())
                .find(|piece| !piece.is_empty())
                .unwrap_or("")
                .parse::<u32>()
        {
            versions.push(version);
        }
        rest = &after[open + close..];
    }
    versions
}

fn check_document(label: &str, document: &str, failures: &mut Vec<String>) {
    match ir_version(document) {
        Some(version) if version >= MIN_IR_VERSION => {}
        Some(version) => failures.push(format!(
            "{label}: ir_version {version} is below the required {MIN_IR_VERSION}"
        )),
        None => failures.push(format!(
            "{label}: declares no ir_version; ONNX reads that as 0, below the required \
             {MIN_IR_VERSION}"
        )),
    }
    let opsets = default_domain_opsets(document);
    if opsets.is_empty() {
        failures.push(format!(
            "{label}: imports no default-domain opset; declare one at >= {MIN_DEFAULT_OPSET}"
        ));
    }
    for version in opsets {
        if version < MIN_DEFAULT_OPSET {
            failures.push(format!(
                "{label}: default opset {version} is below the required {MIN_DEFAULT_OPSET}"
            ));
        }
    }
}

/// Committed fixture packages this PR adds clear the IR/opset floor.
#[test]
fn owned_fixture_packages_meet_the_ir_and_opset_floor() {
    let root = fixtures_root();
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for package in OWNED_PACKAGES {
        let directory = root.join(package);
        assert!(
            directory.is_dir(),
            "fixture package '{package}' is missing at {}; this guard names the packages this \
             PR owns, so a rename must update it rather than silently stop checking",
            directory.display()
        );
        let mut models = Vec::new();
        textprotos(&directory, &mut models);
        assert!(
            !models.is_empty(),
            "fixture package '{package}' contains no *.onnx.textproto to check"
        );
        for model in models {
            let document = std::fs::read_to_string(&model)
                .unwrap_or_else(|error| panic!("cannot read {}: {error}", model.display()));
            check_document(&model.display().to_string(), &document, &mut failures);
            checked += 1;
        }
    }
    assert!(
        failures.is_empty(),
        "fixture models below the IR/opset floor:\n  {}",
        failures.join("\n  ")
    );
    eprintln!("FIXTURE_FLOOR packages checked={checked}");
}

/// Models authored inline in the test sources this PR adds clear the same floor.
///
/// Inline models are the easy place for an old version to creep back: they are
/// string literals, so no ONNX tool ever sees them until a test runs.
#[test]
fn owned_inline_models_meet_the_ir_and_opset_floor() {
    let root = crate_root();
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for source in OWNED_SOURCES {
        let path = root.join(source);
        assert!(
            path.is_file(),
            "test source '{source}' is missing at {}; this guard names the sources this PR \
             owns, so a rename must update it rather than silently stop checking",
            path.display()
        );
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        // Each inline model starts at its own `ir_version:` line; check the
        // slice from there to the next one so a file with several models
        // reports the offending one rather than the file.
        let starts = text
            .match_indices("ir_version:")
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        for (ordinal, start) in starts.iter().copied().enumerate() {
            let end = starts.get(ordinal + 1).copied().unwrap_or(text.len());
            let label = format!("{source} (inline model #{})", ordinal + 1);
            check_document(&label, &text[start..end], &mut failures);
            checked += 1;
        }
    }
    assert!(
        failures.is_empty(),
        "inline models below the IR/opset floor:\n  {}",
        failures.join("\n  ")
    );
    eprintln!("FIXTURE_FLOOR inline models checked={checked}");
}

/// The parser this guard relies on actually rejects what it claims to.
///
/// A version guard that silently matches nothing passes forever; these cases
/// keep it honest about both the single-line and block `opset_import` forms, and
/// about a non-default domain never satisfying the default-domain requirement.
#[test]
fn the_guard_detects_what_it_claims_to() {
    assert_eq!(ir_version("ir_version: 11\n"), Some(11));
    assert_eq!(ir_version("graph { }\n"), None);

    assert_eq!(
        default_domain_opsets("opset_import { domain: \"\" version: 24 }"),
        vec![24]
    );
    assert_eq!(
        default_domain_opsets("opset_import {\n  domain: \"\"\n  version: 24\n}\n"),
        vec![24]
    );
    // An import with no `domain` is the default domain.
    assert_eq!(
        default_domain_opsets("opset_import { version: 24 }"),
        vec![24]
    );
    // A custom domain does not satisfy the default-domain floor.
    assert_eq!(
        default_domain_opsets("opset_import { domain: \"com.microsoft\" version: 1 }"),
        Vec::<u32>::new()
    );
    // Several imports are all reported, so one stale block cannot hide behind a
    // current one.
    assert_eq!(
        default_domain_opsets(
            "opset_import { domain: \"\" version: 24 }\n\
             opset_import { domain: \"com.microsoft\" version: 1 }\n\
             opset_import { domain: \"\" version: 13 }"
        ),
        vec![24, 13]
    );

    let mut failures = Vec::new();
    check_document(
        "stale",
        "ir_version: 8\nopset_import { domain: \"\" version: 13 }",
        &mut failures,
    );
    assert_eq!(failures.len(), 2, "{failures:?}");
    assert!(failures[0].contains("ir_version 8"), "{failures:?}");
    assert!(failures[1].contains("opset 13"), "{failures:?}");

    let mut clean = Vec::new();
    check_document(
        "current",
        "ir_version: 11\nopset_import { domain: \"\" version: 24 }",
        &mut clean,
    );
    assert!(clean.is_empty(), "{clean:?}");
}
