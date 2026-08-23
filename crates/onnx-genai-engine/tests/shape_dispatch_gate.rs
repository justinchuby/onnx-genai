//! The shape-dispatch ban, as a test.
//!
//! The invariant this PR exists to establish is that *one* runtime executes
//! every package: a single decoder, a composite pipeline, a speculative block.
//! Which loop algorithm runs is selected by what the workflow **authors**, never
//! by Rust inspecting the package's shape.
//!
//! Shape dispatch is easy to reintroduce and hard to see in review — one
//! `if engine.is_workflow()` restores the split without touching any test's
//! assertions. This gate names the symbols that constitute the split and fails
//! when one appears outside the places allowed to mention it.
//!
//! # Why a grep and not a type
//!
//! The honest answer is that the type-level version of this rule is the *goal*:
//! when `lowered_workflow`, `canonical_decode` and `WorkflowShape` are gone, the
//! symbols will not exist and this test becomes trivially true. Until then it is
//! the only thing standing between "the split is being removed" and "the split
//! grew a new caller while nobody was looking". The allowed set below is the
//! remaining work, written down: it must only ever shrink.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Symbols that constitute package-shape dispatch.
const BANNED: &[&str] = &[
    "lowered_workflow",
    "canonical_decode",
    "WorkflowShape",
    "is_workflow",
];

/// Files still permitted to mention each symbol.
///
/// This was the convergence work's remaining-work ledger, and it is now empty
/// but for the gate itself: none of these symbols exists any more. It stays
/// because the failure mode it guards against does not need the old code to
/// come back — one `if engine.is_workflow()` on a fresh accessor would restore
/// the split without touching a single assertion elsewhere.
///
/// A file dropping to zero is deleted from this table, not left at 0.
fn allowance() -> BTreeMap<&'static str, usize> {
    BTreeMap::from([
        // This gate names them by construction.
        (
            "crates/onnx-genai-engine/tests/shape_dispatch_gate.rs",
            usize::MAX,
        ),
    ])
}

fn repo_root() -> PathBuf {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .expect("git must be available to locate the repository root");
    PathBuf::from(
        String::from_utf8(output.stdout)
            .expect("git path is UTF-8")
            .trim(),
    )
}

fn tracked_rust_sources(root: &Path) -> Vec<String> {
    let output = Command::new("git")
        .args(["ls-files", "*.rs"])
        .current_dir(root)
        .output()
        .expect("git must be available to enumerate tracked sources");
    String::from_utf8(output.stdout)
        .expect("git paths are UTF-8")
        .lines()
        .map(str::to_owned)
        .collect()
}

/// No new file starts dispatching on package shape.
#[test]
fn shape_dispatch_does_not_spread() {
    let root = repo_root();
    let allowance = allowance();
    let mut offenders = Vec::new();
    let mut scanned = 0usize;

    for relative in tracked_rust_sources(&root) {
        let Ok(source) = std::fs::read_to_string(root.join(&relative)) else {
            continue;
        };
        scanned += 1;
        let uses: usize = BANNED
            .iter()
            .map(|symbol| source.matches(symbol).count())
            .sum();
        if uses == 0 {
            continue;
        }
        let permitted = allowance.get(relative.as_str()).copied().unwrap_or(0);
        if uses > permitted {
            offenders.push(format!(
                "{relative}: {uses} use(s) of package-shape dispatch, allowance {permitted}"
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "package-shape dispatch appeared in a file that must not have it. One runtime executes \
         every package; which loop algorithm runs is selected by what the workflow authors, not \
         by Rust inspecting the package.\n{}",
        offenders.join("\n")
    );
    assert!(scanned > 100, "the scan is not reaching the repository");
}

/// The ledger describes files that exist.
///
/// An allowance for a deleted file is a rule nobody is subject to, and it would
/// silently outlive the work it was tracking.
#[test]
fn the_allowance_ledger_has_no_stale_entries() {
    let root = repo_root();
    let ledger = allowance();
    let stale: Vec<_> = ledger
        .keys()
        .filter(|relative| !root.join(relative).is_file())
        .collect();
    assert!(
        stale.is_empty(),
        "the shape-dispatch allowance names files that no longer exist; remove them so the \
         ledger keeps describing real remaining work: {stale:?}"
    );
}
