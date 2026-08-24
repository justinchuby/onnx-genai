//! The vendored-submodule guard in `build.rs`, tested as a normal library.
//!
//! `build.rs` cannot be unit-tested, so the predicate and the message it emits
//! live in `build_support.rs` and are `include!`d by both. What is under test
//! here is the text a reader sees when their checkout is broken, which is the
//! entire point of the guard: cmake already fails on an unpopulated submodule,
//! just with a message that names a third-party `CMakeLists.txt` and never says
//! "submodule" (#1816).
//!
//! Deliberately not `cfg`-gated on architecture or OS. The predicate is pure
//! path arithmetic and the failure it guards hits every platform, so gating it
//! would produce the failure mode Roy hit in #1809 — a green check from a leg
//! where the test did not exist.

use std::fs;
use std::path::{Path, PathBuf};

include!("../build_support.rs");

/// Scratch trees go under `target/`, never a shared temp directory: two
/// concurrent `cargo test` runs on this box must not collide, and the fixture
/// names are fixed rather than randomised so a crashed run leaves something
/// recognisable rather than litter.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn populate(vendor: &Path, relatives: &[&str]) {
    for relative in relatives {
        let path = vendor.join(relative);
        fs::create_dir_all(path.parent().expect("relative path has a parent"))
            .expect("create fixture parent");
        fs::write(&path, b"fixture").expect("write fixture file");
    }
}

#[test]
fn an_unpopulated_vendor_directory_is_reported_as_a_submodule_fault() {
    let vendor = scratch("vendor-empty");

    let problem = vendor_tree_problem(&vendor).expect("an empty vendor tree is a problem");

    assert!(
        problem.contains("is not populated"),
        "an empty tree is not merely partial: {problem}"
    );
    // The command is the deliverable. Assert the whole thing, not that the
    // word "submodule" appears somewhere, and assert it stands on its own
    // indented line so it survives being copied out of a build log.
    assert!(
        problem.contains(
            "\n    git submodule update --init crates/onnx-runtime-cpuinfo/vendor/cpuinfo"
        ),
        "the message must carry the exact fix command on its own line: {problem}"
    );
    assert!(
        problem.contains("git worktree add"),
        "the message must name the cause, not only the cure: {problem}"
    );
    assert!(
        problem.contains(&vendor.display().to_string()),
        "the message must name the directory it looked in: {problem}"
    );
}

#[test]
fn a_populated_vendor_tree_is_not_a_problem() {
    let vendor = scratch("vendor-populated");
    populate(&vendor, &VENDOR_REQUIRED_FILES);

    assert_eq!(
        vendor_tree_problem(&vendor),
        None,
        "a tree with every required file must build without complaint"
    );
}

#[test]
fn a_half_populated_tree_is_reported_and_names_only_what_is_missing() {
    let vendor = scratch("vendor-half");
    populate(&vendor, &["CMakeLists.txt"]);

    let problem = vendor_tree_problem(&vendor).expect("a missing header is a problem");

    assert!(
        problem.contains("is only partially populated"),
        "a partial tree must not be described as empty: {problem}"
    );
    assert!(
        problem.contains("include/cpuinfo.h"),
        "the message must name the missing file: {problem}"
    );
    assert!(
        !problem.contains("CMakeLists.txt"),
        "a file that is present must not be listed as missing: {problem}"
    );
}

#[test]
fn a_directory_standing_in_for_a_required_file_is_still_a_fault() {
    let vendor = scratch("vendor-dir-not-file");
    fs::create_dir_all(vendor.join("CMakeLists.txt")).expect("create decoy directory");
    populate(&vendor, &["include/cpuinfo.h"]);

    let problem =
        vendor_tree_problem(&vendor).expect("a directory is not a CMakeLists cmake can read");
    assert!(
        problem.contains("CMakeLists.txt"),
        "the message must name the unusable entry: {problem}"
    );
}

/// Anti-vacuity, and drift protection for the predicate itself.
///
/// Every test above runs against a fixture, so all four would still pass if the
/// required-file list stopped describing the real vendored tree — and the guard
/// would then reject every correct checkout on this repo. This is the only cell
/// that reads the tree the crate actually builds from.
#[test]
fn the_vendor_tree_this_crate_builds_from_satisfies_the_guard() {
    let vendor = Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/cpuinfo");

    assert_eq!(
        vendor_tree_problem(&vendor),
        None,
        "this crate compiled, so its vendored tree is usable; if the guard \
         disagrees, the guard is wrong and would now reject every checkout"
    );
}

/// The fix instruction names a path. Pin it to the only authority for that
/// path, so moving the submodule breaks this test rather than silently
/// shipping a command that does nothing.
#[test]
fn the_fix_command_names_the_path_gitmodules_declares() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the crate lives two levels below the repository root")
        .to_path_buf();
    let gitmodules = root.join(".gitmodules");

    let declared = fs::read_to_string(&gitmodules)
        .unwrap_or_else(|e| panic!("read {}: {e}", gitmodules.display()));

    let paths: Vec<&str> = declared
        .lines()
        .filter_map(|line| line.split_once('='))
        .filter(|(key, _)| key.trim() == "path")
        .map(|(_, value)| value.trim())
        .collect();

    assert!(
        paths.contains(&VENDOR_SUBMODULE),
        "the fix command says `{VENDOR_SUBMODULE}` but .gitmodules declares {paths:?}"
    );
}
