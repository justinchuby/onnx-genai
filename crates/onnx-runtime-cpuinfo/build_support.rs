// Shared by `build.rs` and `tests/vendor_submodule_guard.rs` through `include!`.
//
// A build script cannot be unit-tested: it is its own crate, it has no test
// harness, and the only way to observe it is to run a build that fails. That
// is exactly the wrong instrument for a diagnostic whose entire value is the
// wording of its message, so the predicate and the message live here and both
// the build script and a normal integration test read the same copy. The test
// can then assert what a reader will actually see, on every CI leg, without
// building cmake or emptying the real vendor tree.

// An `include!`d file must not bring names into its host's namespace: both
// `build.rs` and the test import from `std::path` themselves, and a `use` here
// would collide. Paths are therefore fully qualified below.

/// The submodule path exactly as `.gitmodules` records it.
///
/// This string is what a reader has to type, so `vendor_submodule_guard.rs`
/// pins it against `.gitmodules` rather than trusting it: a fix instruction
/// that names the wrong path is worse than no instruction, because it costs a
/// second round of confusion on top of the first.
const VENDOR_SUBMODULE: &str = "crates/onnx-runtime-cpuinfo/vendor/cpuinfo";

/// The files this crate's build genuinely needs out of the vendored tree.
///
/// `CMakeLists.txt` is what `cmake::Config::build()` opens and `include/cpuinfo.h`
/// is what bindgen parses. Both are checked because a half-populated tree fails
/// in the second step with an equally opaque message ("Unable to generate
/// cpuinfo bindings") several minutes later than the first.
const VENDOR_REQUIRED_FILES: [&str; 2] = ["CMakeLists.txt", "include/cpuinfo.h"];

/// Describes what is wrong with the vendored cpuinfo tree, or `None` if it is
/// usable.
///
/// The message is the deliverable. `git worktree add` does not populate
/// submodules and neither does a `git clone` without `--recurse-submodules`,
/// so the directory exists but is empty; cmake then reports a missing
/// `CMakeLists.txt` in a third-party vendor path, which names a file the
/// reader has no reason to connect to submodules at all. Classifying that as
/// environmental rather than a code or toolchain break has cost at least one
/// full benchmark validation-matrix re-run (#1816).
fn vendor_tree_problem(vendor: &std::path::Path) -> Option<String> {
    let missing: Vec<&str> = VENDOR_REQUIRED_FILES
        .iter()
        .copied()
        .filter(|relative| !vendor.join(relative).is_file())
        .collect();

    if missing.is_empty() {
        return None;
    }

    // "empty" and "partially populated" are different faults with the same fix
    // but very different likely causes, so say which one this is instead of
    // making the reader infer it.
    let state = if missing.len() == VENDOR_REQUIRED_FILES.len() {
        "is not populated"
    } else {
        "is only partially populated"
    };

    Some(format!(
        "the vendored cpuinfo submodule {state}\n\
         \n\
         directory: {vendor}\n\
         missing:   {missing}\n\
         \n\
         `git worktree add` does not populate submodules, and neither does a\n\
         clone without `--recurse-submodules`. This is an environment fault,\n\
         not a code or toolchain failure.\n\
         \n\
         Fix, from the repository root:\n\
         \n\
         \x20   git submodule update --init {VENDOR_SUBMODULE}\n",
        vendor = vendor.display(),
        missing = missing.join(", "),
    ))
}
