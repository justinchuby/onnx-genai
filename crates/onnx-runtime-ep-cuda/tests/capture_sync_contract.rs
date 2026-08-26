use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

struct FunctionBlock<'a> {
    name: &'a str,
    signature: &'a str,
    body: &'a str,
}

fn function_blocks(source: &str) -> Vec<FunctionBlock<'_>> {
    let mut blocks = Vec::new();
    let mut offset = 0;
    while let Some(relative) = source[offset..].find("fn ") {
        let start = offset + relative;
        let Some(open_relative) = source[start..].find('{') else {
            break;
        };
        let open = start + open_relative;
        let signature = &source[start..open];
        let name_start = start + 3;
        let name_end = source[name_start..]
            .find(|character: char| character == '(' || character.is_whitespace())
            .map(|relative| name_start + relative)
            .unwrap_or(open);

        let bytes = source.as_bytes();
        let mut index = open;
        let mut depth = 0usize;
        let mut string = false;
        let mut escaped = false;
        let mut line_comment = false;
        let mut block_comment_depth = 0usize;
        let mut close = None;
        while index < bytes.len() {
            let byte = bytes[index];
            let next = bytes.get(index + 1).copied();
            if line_comment {
                if byte == b'\n' {
                    line_comment = false;
                }
            } else if block_comment_depth > 0 {
                if byte == b'/' && next == Some(b'*') {
                    block_comment_depth += 1;
                    index += 1;
                } else if byte == b'*' && next == Some(b'/') {
                    block_comment_depth -= 1;
                    index += 1;
                }
            } else if string {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    string = false;
                }
            } else if byte == b'/' && next == Some(b'/') {
                line_comment = true;
                index += 1;
            } else if byte == b'/' && next == Some(b'*') {
                block_comment_depth = 1;
                index += 1;
            } else if byte == b'"' {
                string = true;
            } else if byte == b'{' {
                depth += 1;
            } else if byte == b'}' {
                depth -= 1;
                if depth == 0 {
                    close = Some(index + 1);
                    break;
                }
            }
            index += 1;
        }
        let Some(close) = close else {
            break;
        };
        blocks.push(FunctionBlock {
            name: &source[name_start..name_end],
            signature,
            body: &source[start..close],
        });
        offset = close;
    }
    blocks
}

fn has_capture_guard(function: &FunctionBlock<'_>) -> bool {
    function.body.contains("is_capturing()")
        || (function.signature.contains("sync: bool") && function.body.contains("if sync"))
        || (function.signature.contains("capturing")
            && (function.body.contains("if capturing") || function.body.contains("if !capturing")))
}

fn kernel_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernels")
}

/// Paths that may hold an unguarded `.synchronize()`.
///
/// Every entry is a path whose `capture_support` is explicitly `unsupported`,
/// or a dynamically-admitted kernel's fallback path that `capture_support`
/// rejects. That sentence used to live in a comment, where nothing checked it;
/// `every_allowlisted_file_can_decline_capture` now does.
fn allowlisted_unconditional_syncs() -> BTreeSet<String> {
    BTreeSet::from([
        "attention.rs::run_attention_phase2a".to_string(),
        "block_quantized_matmul.rs::execute".to_string(),
        "block_quantized_moe.rs::execute_with_workspace".to_string(),
        // Drains the non-blocking compute stream before the synchronous
        // default-stream metadata upload, so a prior DFT still reading the
        // step-scoped metadata prefix cannot have it overwritten. The kernel
        // declares CaptureSupport::unsupported and names this host barrier as
        // one of its reasons.
        "dft.rs::run".to_string(),
        "fused_gemm.rs::run".to_string(),
        "gemm.rs::run".to_string(),
        "matmul_nbits.rs::run".to_string(),
        "mod_op.rs::run".to_string(),
        // Declares CaptureSupport::unsupported: the Phase-2a workspace path
        // allocates per-call scratch and drains the trailing transpose before
        // returning it to the pool. Same shape as packed_varlen_attention below.
        "multi_head_attention.rs::execute".to_string(),
        "nary.rs::run".to_string(),
        "non_max_suppression.rs::materialize".to_string(),
        "nonzero.rs::execute".to_string(),
        "packed_varlen_attention.rs::execute".to_string(),
        "pooling.rs::execute".to_string(),
        "pooling.rs::run".to_string(),
        "sparse_kv_gather.rs::execute".to_string(),
        "unique.rs::materialize".to_string(),
        "varlen_attention.rs::execute".to_string(),
    ])
}

#[test]
fn unconditional_syncs_are_limited_to_capture_unsupported_paths() {
    let root = kernel_root();
    let mut unconditional = BTreeSet::new();
    for entry in fs::read_dir(&root).expect("read CUDA kernel sources") {
        let path = entry.expect("kernel source entry").path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }
        let source = fs::read_to_string(&path).expect("read CUDA kernel source");
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source.as_str(), |(before_tests, _)| before_tests);
        for function in function_blocks(production) {
            if function.name == "drop"
                || !function.body.contains(".synchronize()")
                || has_capture_guard(&function)
            {
                continue;
            }
            let file = path.file_name().unwrap().to_string_lossy();
            unconditional.insert(format!("{file}::{}", function.name));
        }
    }

    // Every entry is reviewed, and `every_allowlisted_file_can_decline_capture`
    // checks the review's premise rather than trusting this comment.
    let expected = allowlisted_unconditional_syncs();
    assert_eq!(
        unconditional, expected,
        "a CUDA kernel added or removed an unconditional stream synchronization; \
         capture-supported paths must guard it with CudaRuntime::is_capturing, while \
         capture-unsupported paths must be reviewed and listed explicitly"
    );
}

/// The allowlist is a review decision, and this is the part of that decision a
/// test can check: a kernel may only hold an unguarded `.synchronize()` if it
/// is able to decline capture in the first place.
///
/// Without this, the allowlist is an unconditional escape hatch — one line
/// silences the contract for a kernel that advertises `CaptureSupport::
/// Supported`, and graph capture breaks with the suite green. That is a worse
/// failure than the one the contract exists to catch, because it is silent.
///
/// **What this cannot see.** It resolves `capture_support` per *file*, not per
/// kernel, because the source scan above is flat and has no `impl` awareness.
/// A file holding two kernels — one that declines capture and one that does
/// not — satisfies this check even if the listed function belongs to the
/// second. Two files already contain both a `Supported` and an `unsupported`
/// arm (`attention.rs`, `matmul_nbits.rs`), though in both cases they are the
/// two arms of one dynamically-admitted kernel, which is the documented reason
/// those entries are listed. Narrowing this to per-kernel means teaching
/// `function_blocks` about `impl` blocks; until then the check is a lower
/// bound, and stated as one.
#[test]
fn every_allowlisted_file_can_decline_capture() {
    let root = kernel_root();
    for entry in allowlisted_unconditional_syncs() {
        let (file, function) = entry
            .split_once("::")
            .unwrap_or_else(|| panic!("allowlist entry {entry} is not `file.rs::function`"));
        let path = root.join(file);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read allowlisted kernel source {file}: {error}"));
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source.as_str(), |(before_tests, _)| before_tests);
        let declines = function_blocks(production).into_iter().any(|block| {
            block.name == "capture_support" && block.body.contains("CaptureSupport::unsupported")
        });
        assert!(
            declines,
            "{file} is on the unconditional-sync allowlist for {function}, but no \
             capture_support in it returns CaptureSupport::unsupported; an unguarded \
             .synchronize() is only reviewable in a kernel that can decline capture"
        );
    }
}
