//! Stamps the running binary with the commit it was built from.
//!
//! A process cannot otherwise be asked which source produced it. `ps` reports
//! the binary's *path*, which is set by `CARGO_TARGET_DIR` and is shared across
//! worktrees, so it names a directory rather than a history. Comparing the
//! executable's inode against the file on disk proves only that the file has
//! not been replaced since exec -- a binary correctly built from the wrong
//! branch passes that check exactly as well as a correct one.
//!
//! The launcher has the same blind spot: `if [[ ! -x $SERVER_BIN ]]` decides
//! whether to rebuild from a mode bit, and a mode bit cannot test which tree a
//! binary came from. So a stale binary and a fresh one are indistinguishable to
//! every instrument we had, which is why "which code is actually running" cost
//! this project hours.
//!
//! This build script answers it directly: the commit is resolved at compile
//! time and served on `/v1/status`, so the question becomes one `curl`.
//!
//! **This script never fails the build.** Every git invocation that errors,
//! times out, or returns non-UTF-8 degrades to `unknown`. A provenance stamp
//! that can break compilation would be traded away the first time it was
//! inconvenient, and an honest `unknown` is worth more than a value nobody
//! trusts.

use std::process::Command;

/// Runs `git` with `args` and returns trimmed stdout, or `None` on any failure.
fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    Some(text.trim().to_owned())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Re-stamp when HEAD moves. `.git` is a *file* in a linked worktree, so the
    // usual `../../.git/HEAD` guess resolves to nothing and the stamp silently
    // freezes at whatever commit was checked out the first time the crate was
    // compiled. Ask git for the directory instead of constructing the path.
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        // Changes on branch SWITCH. It does not change on commit: on a branch
        // this file holds the constant text `ref: refs/heads/<name>`, so
        // watching it alone leaves the stamp frozen across every commit -- as
        // it did, which is how this was found. Necessary, not sufficient.
        println!("cargo:rerun-if-changed={git_dir}/HEAD");

        if let Some(common) = git(&["rev-parse", "--path-format=absolute", "--git-common-dir"]) {
            // The file that actually moves on every commit: the loose ref that
            // HEAD points at, in the SHARED store rather than this worktree's
            // git dir. Resolve it symbolically instead of parsing the HEAD
            // file, so a detached HEAD (no symbolic ref) correctly yields
            // nothing here and falls back to the two watches above.
            if let Some(head_ref) = git(&["symbolic-ref", "--quiet", "HEAD"]) {
                println!("cargo:rerun-if-changed={common}/{head_ref}");
            }
            // Covers the case the line above cannot: once a ref is packed, the
            // loose file stops existing and the tip moves inside packed-refs.
            println!("cargo:rerun-if-changed={common}/packed-refs");
        }
    }

    let sha = git(&["rev-parse", "--short=8", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());

    // Whether the tree had uncommitted changes at build time. This is
    // deliberately a third state rather than a bool: "we could not tell" and
    // "there was nothing uncommitted" are different claims, and collapsing them
    // is the same defect as reporting an unmeasurable field as zero.
    //
    // Honest limitation, stated because it cannot be fixed here: cargo decides
    // whether to re-run this script, and editing a file without moving HEAD may
    // not trigger it. `dirty` can therefore report a stale `false`. The SHA is
    // the load-bearing half -- a binary built from another tree carries that
    // tree's HEAD -- and `dirty` is a hint, not a guarantee.
    let dirty = match git(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(out) if out.is_empty() => "false",
        Some(_) => "true",
        None => "unknown",
    };

    println!("cargo:rustc-env=ONNX_GENAI_BUILD_SHA={sha}");
    println!("cargo:rustc-env=ONNX_GENAI_BUILD_DIRTY={dirty}");
}
