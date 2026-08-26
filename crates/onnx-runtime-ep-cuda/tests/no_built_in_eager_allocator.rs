//! Criterion 2 and 8, asserted where they can actually be checked without a
//! GPU: the built-in eager `cuMemAlloc` allocator is gone from the memory
//! crate, and the only eager call sites that remain elsewhere are the raw
//! metadata upload seam in `runtime.rs`.
//!
//! The GPU tests next door prove the default path *reaches* the arena. They
//! cannot run here, and they also cannot prove a negative: a second eager site
//! added tomorrow on a path those tests do not exercise would not turn them
//! red. This file is the complement — it is a structural assertion, it runs on
//! every host including ones with no CUDA at all, and it is the reason
//! "no built-in eager `cuMemAlloc` calls" is a claim rather than a hope.
//!
//! It is deliberately an *exact-count allowlist* rather than a "no new sites"
//! check. Two eager sites survive Phase 7 on purpose, both outside the
//! `DeviceAllocator` seam:
//!
//! - `runtime.rs` — `CudaRuntime::alloc_raw`/`free_raw`, small synchronous
//!   metadata uploads for kernel launches.
//!
//! Naming them here means they are disclosed rather than hidden, and it means
//! removing one is as visible as adding one.
//!
//! The companion `alloc_zeros::<u8>(` scan below is the cuDNN-specific guard:
//! the allowlist still counts only `malloc_sync`/`free_sync`, but a fresh eager
//! stream allocation for a cuDNN workspace would be just as much a regression
//! and must therefore stay textually pinned too.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Call sites, not mentions: the trailing `(` keeps prose, `use` lines and
/// SAFETY comments that name the function from being counted as calls.
const EAGER_ALLOC: &str = "malloc_sync(";
const EAGER_FREE: &str = "free_sync(";
const DELETED_VMM_FLAG: &str = "ONNX_GENAI_CUDA_VMM";

fn crate_src(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join(name)
        .join("src")
}

fn rust_sources(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|error| {
            panic!("cannot read {}: {error}", dir.display());
        }) {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }
    assert!(
        !found.is_empty(),
        "no sources found under {} -- the scan is looking in the wrong place, so every \
         assertion built on it would be vacuous",
        root.display()
    );
    found.sort();
    found
}

/// `file name -> occurrences`, counting every occurrence on every line.
fn count(root: &Path, needle: &str) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for path in rust_sources(root) {
        let text = std::fs::read_to_string(&path).expect("a readable source file");
        let hits = text.matches(needle).count();
        if hits > 0 {
            let key = path
                .strip_prefix(root)
                .expect("under the scanned root")
                .to_string_lossy()
                .into_owned();
            counts.insert(key, hits);
        }
    }
    counts
}

/// As `count`, but ignoring lines that are entirely a comment.
fn count_code(root: &Path, needle: &str) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for path in rust_sources(root) {
        let text = std::fs::read_to_string(&path).expect("a readable source file");
        let hits: usize = text
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .map(|line| line.matches(needle).count())
            .sum();
        if hits > 0 {
            let key = path
                .strip_prefix(root)
                .expect("under the scanned root")
                .to_string_lossy()
                .into_owned();
            counts.insert(key, hits);
        }
    }
    counts
}

fn is_user_facing_path(path: &str) -> bool {
    path.ends_with(".md")
        || path.ends_with(".txt")
        || path.ends_with(".toml")
        || path.ends_with(".py")
        || path.ends_with(".ps1")
        || path.ends_with(".sh")
        || path.ends_with(".yml")
        || path.ends_with(".yaml")
        || (path.ends_with(".rs") && path.contains("/src/") && !path.ends_with("/tests.rs"))
}

fn tracked_user_facing_files(root: &Path) -> Vec<(String, String)> {
    let output = Command::new("git")
        .args([
            "grep",
            "-l",
            "-I",
            DELETED_VMM_FLAG,
            "--",
            "*.md",
            "*.txt",
            "*.toml",
            "*.py",
            "*.ps1",
            "*.sh",
            "*.yml",
            "*.yaml",
            "*.rs",
        ])
        .current_dir(root)
        .output()
        .expect("git must be available to search user-facing repository text");
    assert!(
        output.status.success(),
        "git grep found no deleted-flag history or failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mut files = String::from_utf8(output.stdout)
        .expect("git paths are UTF-8")
        .lines()
        .filter(|path| is_user_facing_path(path))
        .filter(|path| root.join(path).is_file())
        .map(|path| {
            (
                path.to_owned(),
                std::fs::read_to_string(root.join(path))
                    .unwrap_or_else(|error| panic!("cannot read {path}: {error}")),
            )
        })
        .collect::<Vec<_>>();
    files.sort_by(|left, right| left.0.cmp(&right.0));
    files
}

fn deleted_vmm_mention_is_history(path: &str, lines: &[&str], index: usize) -> bool {
    // Dated benchmark records preserve the command that produced their
    // historical measurements; they are evidence, not current instructions.
    if path.starts_with("docs/benchmarks/") {
        return true;
    }
    let start = index.saturating_sub(2);
    let end = (index + 3).min(lines.len());
    let context = lines[start..end].join(" ").to_lowercase();
    [
        "deleted",
        "removed",
        "no longer",
        "used to",
        "formerly",
        "historical",
        "before phase",
        "there used to be",
        "now-deleted",
        "删除",
        "原先",
    ]
    .iter()
    .any(|marker| context.contains(marker))
}

fn deleted_vmm_guidance_errors(files: &[(String, String)]) -> Vec<String> {
    let mut errors = Vec::new();
    for (path, text) in files {
        let lines = text.lines().collect::<Vec<_>>();
        for (index, line) in lines.iter().enumerate() {
            if line.contains(DELETED_VMM_FLAG)
                && !deleted_vmm_mention_is_history(path, &lines, index)
            {
                errors.push(format!(
                    "{path}:{} presents deleted flag {DELETED_VMM_FLAG} outside an explicit \
                     historical/deletion statement",
                    index + 1
                ));
            }
        }
    }
    errors
}

/// Both scans can see what they claim to see.
///
/// Every assertion below is of the form "this count is zero" or "this count is
/// exactly N". Both are satisfied by a scanner that reads nothing at all, so
/// each scanner is checked against a site that is known to exist before any of
/// them are believed.
///
/// `count` and `count_code` are pinned *separately* and on purpose. They are
/// two helpers, so anchoring one says nothing about the other: replacing
/// `count_code`'s comment filter with `.filter(|_line| false)` leaves a
/// scanner that reads no lines at all, and every `count_code` assertion in
/// this file is an `is_empty()` — all of which a blind scanner satisfies
/// trivially. The `count` anchor below does not touch `count_code`, and
/// `the_removal_stays_explained_in_prose_and_the_code_scan_can_tell_the_difference`
/// does not anchor it either: it asserts `count` sees `ONNX_GENAI_CUDA_VMM`
/// and `count_code` does not, and *both* of those stay true when `count_code`
/// is blinded. So `count_code` gets its own positive assertion, against a
/// needle that lives on a real code line right now.
#[test]
fn the_scan_can_observe_an_eager_call_site_that_is_known_to_exist() {
    let ep = crate_src("onnx-runtime-ep-cuda");
    let allocs = count(&ep, EAGER_ALLOC);
    assert!(
        allocs.contains_key("runtime.rs"),
        "the scanner did not find the known eager site in runtime.rs, so nothing it reports \
         about the absence of other sites means anything: {allocs:?}"
    );
    let memory = crate_src("onnx-runtime-cuda-memory");
    assert!(
        count(&memory, "impl DeviceAllocator for")
            .values()
            .sum::<usize>()
            > 0,
        "the scanner found no allocator implementation in the memory crate at all"
    );

    // `CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV` is declared on a code line in
    // `vmm_allocator.rs` and is not going away: it is the surviving pool-bound
    // environment variable, and the shipped-constraints table documents it. A
    // `count_code` that cannot find it is a `count_code` that cannot find
    // anything, which would make every emptiness assertion in this file a
    // statement about the helper rather than about the code.
    let observable = count_code(&memory, "CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV");
    assert!(
        !observable.is_empty(),
        "the code scan found no occurrence of a constant that is declared on a code line right \
         now, so it is reading nothing and every `count_code(..).is_empty()` assertion below is \
         vacuous: {observable:?}"
    );
}

/// Criterion 2: the CUDA memory crate contains no eager `cuMemAlloc`
/// allocation or free at all.
///
/// This is the crate that owns EP-managed device memory. Before Phase 7 it
/// held `CudaDeviceAllocator`, and a VMM failure fell back to it. Zero here is
/// what "no production CUDA EP-managed allocation silently falls back to the
/// built-in `cuMemAlloc` allocator" reduces to structurally: there is nothing
/// to fall back *to*.
#[test]
fn the_cuda_memory_crate_has_no_eager_allocation_sites() {
    let memory = crate_src("onnx-runtime-cuda-memory");
    let allocs = count(&memory, EAGER_ALLOC);
    let frees = count(&memory, EAGER_FREE);
    assert!(
        allocs.is_empty() && frees.is_empty(),
        "the built-in eager allocator is supposed to be gone, but the memory crate still calls \
         it: allocations {allocs:?}, frees {frees:?}"
    );
}

/// Criterion 6: the memory crate offers exactly one built-in mechanism.
///
/// Two implementations is the state criterion 6 names — an eager allocator
/// beside a separately preferred VMM one — and it is the state that makes a
/// silent fallback expressible in the first place.
#[test]
fn the_memory_crate_provides_exactly_one_built_in_mechanism() {
    let memory = crate_src("onnx-runtime-cuda-memory");
    let impls = count(&memory, "impl DeviceAllocator for");
    let total: usize = impls.values().sum();
    assert_eq!(
        total, 1,
        "exactly one built-in CUDA memory mechanism should implement the allocator contract, \
         found {total}: {impls:?}"
    );
    assert_eq!(
        impls.keys().next().map(String::as_str),
        Some("vmm_allocator.rs"),
        "the one built-in mechanism must be the VMM arena: {impls:?}"
    );
}

/// The removed type and the removed opt-in flag are not referenced by any
/// production *code* in either crate.
///
/// A leftover reference would mean either a resurrected implementation or a
/// path still branching on a flag that no longer does anything.
///
/// Comments are excluded, and that is a deliberate weakening with a reason:
/// criterion 12 requires the removal to stay explained where the reader is,
/// so `vmm_allocator.rs` still says in prose why there is no arena on/off
/// switch any more. A scan that could not tell prose from code would force
/// that explanation to be deleted to stay green, which is the opposite of what
/// is wanted. The companion test below pins that the explanation is still
/// there, so the exclusion cannot be used to smuggle anything back in.
#[test]
fn the_removed_type_and_flag_are_absent_from_production_code() {
    for crate_name in ["onnx-runtime-cuda-memory", "onnx-runtime-ep-cuda"] {
        let root = crate_src(crate_name);
        for removed in ["CudaDeviceAllocator", "ONNX_GENAI_CUDA_VMM", "CUDA_VMM_ENV"] {
            let hits = count_code(&root, removed);
            assert!(
                hits.is_empty(),
                "{crate_name} still references the removed `{removed}` in code: {hits:?}"
            );
        }
    }
}

/// The deleted arena-selection flag has no parser left, and native decode must
/// not offer it as an actionable recovery step.
#[test]
fn the_deleted_vmm_flag_has_no_parser_or_native_decode_guidance() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/");
    let parser_tokens = [
        "env::var(\"ONNX_GENAI_CUDA_VMM\")",
        "env::var_os(\"ONNX_GENAI_CUDA_VMM\")",
        "var(\"ONNX_GENAI_CUDA_VMM\")",
        "var_os(\"ONNX_GENAI_CUDA_VMM\")",
    ];
    for crate_name in [
        "onnx-runtime-cuda-memory",
        "onnx-runtime-ep-cuda",
        "onnx-genai-engine",
        "onnx-genai-cli",
    ] {
        let source = crates.join(crate_name).join("src");
        for parser in parser_tokens {
            let hits = count_code(&source, parser);
            assert!(
                hits.is_empty(),
                "{crate_name} still parses the deleted ONNX_GENAI_CUDA_VMM flag via \
                 {parser:?}: {hits:?}"
            );
        }
    }

    let native_decode = crates
        .join("onnx-genai-engine")
        .join("src")
        .join("native_decode")
        .join("cuda.rs");
    let text = std::fs::read_to_string(&native_decode).expect("native CUDA decode source");
    assert!(
        !text.contains("ONNX_GENAI_CUDA_VMM"),
        "{} still presents the deleted allocator-selection flag to users",
        native_decode.display()
    );
}

#[test]
fn user_facing_repository_text_has_no_actionable_deleted_vmm_guidance() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repository root");
    let files = tracked_user_facing_files(root);
    assert!(
        is_user_facing_path("crates/onnx-genai-cli/src/profile.rs")
            && root.join("crates/onnx-genai-cli/src/profile.rs").is_file(),
        "the repository-wide guidance scan omitted the CLI profile output"
    );
    assert!(
        files
            .iter()
            .any(|(path, _)| path == "docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md"),
        "the repository-wide guidance scan omitted maintained documentation"
    );

    let errors = deleted_vmm_guidance_errors(&files);
    assert!(
        errors.is_empty(),
        "deleted VMM flag appears as current/actionable guidance:\n{}",
        errors.join("\n")
    );
}

#[test]
fn deleted_vmm_guidance_scan_rejects_raw_cli_advice() {
    let files = vec![(
        "crates/onnx-genai-cli/src/profile.rs".to_owned(),
        "\"not installed (set ONNX_GENAI_CUDA_VMM=1)\"".to_owned(),
    )];
    let errors = deleted_vmm_guidance_errors(&files);
    assert_eq!(
        errors.len(),
        1,
        "reinserting raw CLI advice must fail the production guidance guard"
    );
    assert!(errors[0].contains("crates/onnx-genai-cli/src/profile.rs:1"));
}

#[test]
fn deleted_vmm_guidance_scan_allows_explicit_deletion_history() {
    let files = vec![(
        "docs/memory/history.md".to_owned(),
        "The ONNX_GENAI_CUDA_VMM flag was deleted, not deprecated.".to_owned(),
    )];
    assert!(
        deleted_vmm_guidance_errors(&files).is_empty(),
        "an explicit deletion statement is historical context, not actionable advice"
    );
}

#[test]
fn cudnn_package_rationale_matches_claim_then_execute_sources() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repository root");
    let read = |relative: &str| {
        std::fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("cannot read {relative}: {error}"))
    };

    let registry = read("crates/onnx-runtime-ep-cuda/src/kernels/mod.rs");
    assert!(registry.contains("OpKey::new(\"Conv\", \"\", 1)"));
    assert!(registry.contains("(\"MaxPool\", pooling::PoolKind::Max)"));
    assert!(registry.contains("(\"AveragePool\", pooling::PoolKind::Average)"));

    let compact = |text: &str| text.split_whitespace().collect::<String>();
    let conv = read("crates/onnx-runtime-ep-cuda/src/kernels/conv.rs");
    let pooling = read("crates/onnx-runtime-ep-cuda/src/kernels/pooling.rs");
    assert!(compact(&conv).contains("self.runtime.cudnn().with_handle"));
    assert!(compact(&pooling).contains("self.runtime.cudnn().with_handle"));

    let requirements = read("requirements-cuda-dev.txt");
    let pyproject = read("crates/onnx-runtime-python/pyproject.toml");
    let strategy = read("docs/execution/CUDA_STRATEGY.md");
    for (path, text) in [
        ("requirements-cuda-dev.txt", requirements.as_str()),
        (
            "crates/onnx-runtime-python/pyproject.toml",
            pyproject.as_str(),
        ),
        ("docs/execution/CUDA_STRATEGY.md", strategy.as_str()),
    ] {
        assert!(
            !text.contains("decline placement"),
            "{path} repeats the false claim that missing cuDNN declines placement"
        );
    }
    assert!(requirements.contains("claimed before runtime-library discovery"));
    assert!(pyproject.contains("registers and claims its"));
    assert!(
        pyproject.contains("supported Conv and pooling nodes before runtime-library discovery")
    );
    assert!(strategy.contains("claimed without probing cuDNN availability"));
    assert!(strategy.contains("#2198"));
}

/// Criterion 12, and the non-vacuity anchor for the test above: the removed
/// flag is still *explained* in prose, and the code scan is what makes the
/// difference.
///
/// If comment-stripping silently stopped working, the test above would become
/// unable to distinguish a documented removal from a live reference. Asserting
/// that the raw scan sees the mention and the code scan does not proves the
/// two scans actually differ, on a case that exists right now.
#[test]
fn the_removal_stays_explained_in_prose_and_the_code_scan_can_tell_the_difference() {
    let memory = crate_src("onnx-runtime-cuda-memory");
    let documented = count(&memory, "ONNX_GENAI_CUDA_VMM");
    assert!(
        documented.contains_key("vmm_allocator.rs"),
        "the reason the arena has no on/off switch any more must stay written down where the \
         surviving mechanism is defined: {documented:?}"
    );
    assert!(
        count_code(&memory, "ONNX_GENAI_CUDA_VMM").is_empty(),
        "prose stripping is not working, so the code-reference test proves nothing"
    );
}

/// The eager call sites outside the allocator seam are exactly the two known
/// ones.
///
/// These are not EP-managed allocations through `DeviceAllocator`; they are the
/// raw metadata-upload seam in `runtime.rs`, and Phase 7 does not claim to have
/// removed them. Pinning the exact set is what stops the claim from quietly
/// widening: a third site, or a fourth, goes red here.
///
/// The counts are textual occurrences, not seams, and `runtime.rs` is two of
/// each for one seam. `alloc_raw` calls `malloc_sync`, and on failure drains
/// the raw pool and calls it again rather than reporting out-of-memory while
/// still holding device memory back from everyone else; the frees are that
/// drain and `free_raw`. That retry arrived on `main` while this file was
/// being written on the memory-refactor stack, which expected one of each, so
/// the merge of the two produced a red that is a genuine reconciliation and
/// not a widened claim -- the seam count is still two.
#[test]
fn the_eager_sites_outside_the_allocator_seam_are_exactly_the_two_disclosed_ones() {
    let ep = crate_src("onnx-runtime-ep-cuda");
    let expected_allocs: BTreeMap<String, usize> =
        [(String::from("runtime.rs"), 2usize)].into_iter().collect();
    let expected_frees = expected_allocs.clone();

    assert_eq!(
        count(&ep, EAGER_ALLOC),
        expected_allocs,
        "the set of eager cuMemAlloc call sites in the CUDA EP changed; if this is intentional \
         the allowlist and the PR's disclosure must both be updated"
    );
    assert_eq!(
        count(&ep, EAGER_FREE),
        expected_frees,
        "the set of eager cuMemFree call sites in the CUDA EP changed"
    );
}

#[test]
fn cudnn_has_no_direct_stream_workspace_allocations() {
    let ep = crate_src("onnx-runtime-ep-cuda");
    let eager_stream_allocs = count_code(&ep, "alloc_zeros::<u8>(");
    assert!(
        eager_stream_allocs.is_empty(),
        "cuDNN workspaces must come from the prepared provider workspace, not a direct stream allocation: {eager_stream_allocs:?}"
    );
}
