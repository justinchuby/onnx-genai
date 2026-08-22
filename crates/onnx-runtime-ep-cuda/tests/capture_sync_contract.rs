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

#[test]
fn unconditional_syncs_are_limited_to_capture_unsupported_paths() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernels");
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

    // Every entry is a path whose capture_support is explicitly Unsupported, or
    // a dynamically-admitted kernel's fallback path that capture_support rejects.
    let expected = BTreeSet::from([
        "attention.rs::run_attention_phase2a".to_string(),
        "block_quantized_matmul.rs::execute".to_string(),
        "block_quantized_moe.rs::execute_with_workspace".to_string(),
        "fused_gemm.rs::run".to_string(),
        "gemm.rs::run".to_string(),
        "matmul_nbits.rs::run".to_string(),
        "mod_op.rs::run".to_string(),
        "nary.rs::run".to_string(),
        "nonzero.rs::execute".to_string(),
        "packed_varlen_attention.rs::execute".to_string(),
        "pooling.rs::execute".to_string(),
        "pooling.rs::run".to_string(),
        "sparse_kv_gather.rs::execute".to_string(),
        "varlen_attention.rs::execute".to_string(),
    ]);
    assert_eq!(
        unconditional, expected,
        "a CUDA kernel added or removed an unconditional stream synchronization; \
         capture-supported paths must guard it with CudaRuntime::is_capturing, while \
         capture-unsupported paths must be reviewed and listed explicitly"
    );
}
