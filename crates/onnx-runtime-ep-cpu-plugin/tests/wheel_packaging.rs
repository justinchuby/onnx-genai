//! The `nxrt-ep-cpu` wheel is built from this crate, and its packaging is only
//! exercised by a tag-triggered release workflow. These tests pin the parts of
//! it that would otherwise fail for the first time at release time.

use std::path::{Path, PathBuf};

fn wheel_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/nxrt-ep-cpu")
        .canonicalize()
        .expect("the nxrt-ep-cpu packaging directory exists next to this crate")
}

fn read(name: &str) -> String {
    let path = wheel_dir().join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The wheel's smoke command must name a file that exists, via `{package}`.
///
/// cibuildwheel is invoked as `cibuildwheel python/nxrt-ep-cpu` **from the
/// repository root**, so `{project}` expands to the repository root and
/// `{package}` to this directory. Writing `{project}/check_wheel.py` therefore
/// produces `python: can't open file '/project/check_wheel.py'` — and because
/// the wheel job only runs on an `nxrt-ep-v*` tag, that failure would appear
/// for the first time during a release, in all four wheel lanes at once.
#[test]
fn wheel_test_command_names_a_file_that_exists() {
    let pyproject = read("pyproject.toml");
    let line = pyproject
        .lines()
        .find(|l| l.trim_start().starts_with("test-command"))
        .expect("pyproject.toml declares a cibuildwheel test-command");

    assert!(
        !line.contains("{project}"),
        "test-command uses {{project}} (the repository root): {line}"
    );

    let script = line
        .split("{package}/")
        .nth(1)
        .and_then(|rest| {
            rest.split(|c: char| c.is_whitespace() || c == '\'' || c == '"')
                .next()
        })
        .unwrap_or_else(|| panic!("test-command does not reference a {{package}} path: {line}"));
    let path = wheel_dir().join(script);
    assert!(
        path.exists(),
        "the wheel's test-command runs {script}, which does not exist at {}",
        path.display()
    );
}

/// Every operating system the wheel enables MLAS for is compiled by a CI lane.
///
/// `MLAS_TARGETS` is a promise that the vendored C++/asm builds there. Nothing
/// else in the repository checks that promise, and the cost of breaking it is a
/// release-time wheel failure on that platform. Counting *operating systems*
/// rather than targets is what the CI shape supports: one step in the coverage
/// matrix covers both `windows/amd64` and `darwin/arm64`, so a per-target count
/// would be wrong in the other direction.
#[test]
fn every_mlas_wheel_target_is_built_by_a_ci_lane() {
    let setup = read("setup.py");
    // The block ends at the first `)` in column zero; a `)` inside it always
    // follows a tuple element, never a line start.
    let targets = setup
        .split("MLAS_TARGETS = frozenset(")
        .nth(1)
        .expect("setup.py declares MLAS_TARGETS")
        .split("\n)")
        .next()
        .expect("MLAS_TARGETS is a closed expression")
        .to_owned();
    let systems: std::collections::BTreeSet<&str> = targets
        .split("(\"")
        .skip(1)
        .filter_map(|rest| rest.split('"').next())
        .collect();
    assert!(!systems.is_empty(), "MLAS_TARGETS is empty");

    let ci = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/ci.yml"),
    )
    .expect("read ci.yml");
    let mlas_builds = ci
        .lines()
        .filter(|l| l.contains("-p onnx-runtime-ep-cpu-plugin --features mlas"))
        .count();
    assert!(
        mlas_builds >= systems.len(),
        "setup.py enables MLAS for the operating systems {systems:?} but ci.yml \
         builds the MLAS cdylib on only {mlas_builds} lanes; an unproven target \
         fails at release time"
    );
}
