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

/// Every operating system the wheel's *research* MLAS opt-in permits is
/// compiled by a CI lane.
///
/// No shipped wheel enables MLAS — see `default_artifacts_are_mlas_free.rs`.
/// `MLAS_TARGETS` is the allow-list for `NXRT_EP_CPU_RESEARCH_MLAS=1`, and a
/// research build that cannot link is not useful either, so the promise that
/// the vendored C++/asm builds on those systems still has to hold. Nothing else
/// in the repository checks it, and the cost of breaking it is discovering, at
/// the moment someone wants a differential number, that the reference does not
/// compile on their machine.
///
/// The check is structural rather than textual: it reads `ci.yml`, resolves
/// each job's runner operating systems (including `runs-on: ${{ matrix.os }}`
/// expansions) and each MLAS build step's `runner.os` condition, then asserts
/// the union covers every operating system in `MLAS_TARGETS`. Counting
/// occurrences instead would stay green when a matrix row is deleted -- the
/// darwin build exists only as a `runner.os != 'Linux'` step on the coverage
/// matrix, so removing the `macos-latest` row silently drops it.
#[test]
fn every_mlas_wheel_target_is_built_by_a_ci_lane() {
    let systems = mlas_target_systems();
    assert!(!systems.is_empty(), "MLAS_TARGETS is empty");

    let built = systems_building_mlas_in_ci();
    let missing: Vec<&&str> = systems.iter().filter(|os| !built.contains(**os)).collect();
    assert!(
        missing.is_empty(),
        "setup.py permits the MLAS research opt-in on {systems:?} but no ci.yml \
         lane compiles the MLAS cdylib on {missing:?}; an unproven target only \
         fails once someone tries to take a differential measurement there \
         (lanes cover {built:?})"
    );
}

/// The operating systems named by `MLAS_TARGETS` in `setup.py`.
fn mlas_target_systems() -> std::collections::BTreeSet<&'static str> {
    // Leaked so the returned names borrow from a `'static` string: this is a
    // test binary that reads the file once.
    let setup: &'static str = Box::leak(read("setup.py").into_boxed_str());
    // The block ends at the first `)` in column zero; a `)` inside it always
    // follows a tuple element, never a line start.
    let targets = setup
        .split("MLAS_TARGETS = frozenset(")
        .nth(1)
        .expect("setup.py declares MLAS_TARGETS")
        .split("\n)")
        .next()
        .expect("MLAS_TARGETS is a closed expression");
    targets
        .split("(\"")
        .skip(1)
        .filter_map(|rest| rest.split('"').next())
        .collect()
}

/// `sys.platform`-style names of every operating system on which some `ci.yml`
/// step compiles the plugin cdylib with `--features mlas`.
fn systems_building_mlas_in_ci() -> std::collections::BTreeSet<&'static str> {
    let ci: &'static str = Box::leak(
        std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/ci.yml"),
        )
        .expect("read ci.yml")
        .into_boxed_str(),
    );
    let workflow: serde_yaml::Value = serde_yaml::from_str(ci).expect("ci.yml parses as YAML");
    let jobs = workflow
        .get("jobs")
        .and_then(serde_yaml::Value::as_mapping)
        .expect("ci.yml declares jobs");

    let mut built = std::collections::BTreeSet::new();
    for (_, job) in jobs {
        let job_systems = runner_systems(job);
        let Some(steps) = job.get("steps").and_then(serde_yaml::Value::as_sequence) else {
            continue;
        };
        for step in steps {
            let run = step
                .get("run")
                .and_then(serde_yaml::Value::as_str)
                .unwrap_or_default();
            if !run.contains("-p onnx-runtime-ep-cpu-plugin") || !run.contains("--features mlas") {
                continue;
            }
            let condition = step
                .get("if")
                .and_then(serde_yaml::Value::as_str)
                .unwrap_or_default();
            built.extend(
                job_systems
                    .iter()
                    .filter(|os| runs_on_system(condition, os)),
            );
        }
    }
    built
}

/// The operating systems a job's runners provide, expanding `matrix.os`.
///
/// Only `runs-on` shapes this repository uses are understood; an unrecognised
/// one contributes nothing, which can only make the coverage assertion
/// stricter.
fn runner_systems(job: &serde_yaml::Value) -> std::collections::BTreeSet<&'static str> {
    let runs_on = job
        .get("runs-on")
        .and_then(serde_yaml::Value::as_str)
        .unwrap_or_default();
    if !runs_on.contains("${{") {
        return runner_label_system(runs_on).into_iter().collect();
    }
    let Some(matrix) = job.get("strategy").and_then(|s| s.get("matrix")) else {
        return std::collections::BTreeSet::new();
    };
    // `runs-on: ${{ matrix.os }}` over either `matrix.include[].os` or a plain
    // `matrix.os` list.
    let include = matrix
        .get("include")
        .and_then(serde_yaml::Value::as_sequence)
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row.get("os"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let plain = matrix
        .get("os")
        .and_then(serde_yaml::Value::as_sequence)
        .map(|values| values.iter().collect::<Vec<_>>())
        .unwrap_or_default();
    include
        .into_iter()
        .chain(plain)
        .filter_map(serde_yaml::Value::as_str)
        .filter_map(runner_label_system)
        .collect()
}

/// Map a GitHub runner label (`ubuntu-latest`, `windows-11-arm`, ...) to the
/// `sys.platform`-style name `setup.py` uses.
fn runner_label_system(label: &str) -> Option<&'static str> {
    match label {
        l if l.contains("ubuntu") => Some("linux"),
        l if l.contains("windows") => Some("windows"),
        l if l.contains("macos") => Some("darwin"),
        _ => None,
    }
}

/// Whether a step-level `if:` expression still runs on `system`.
///
/// Only `runner.os` comparisons restrict the operating system; other
/// conditions (`always()`, event filters) gate *when* a step runs, not where,
/// so they are ignored.
fn runs_on_system(condition: &str, system: &str) -> bool {
    let runner_os = match system {
        "linux" => "Linux",
        "windows" => "Windows",
        "darwin" => "macOS",
        _ => return false,
    };
    let excluded = condition.contains(&format!("runner.os != '{runner_os}'"));
    let equality = condition.split("runner.os == '").skip(1).count() > 0;
    let included = condition.contains(&format!("runner.os == '{runner_os}'"));
    !excluded && (!equality || included)
}
