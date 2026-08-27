use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::Path;

struct FunctionBlock<'a> {
    name: &'a str,
    signature: &'a str,
    body: &'a str,
}

fn braced_block(source: &str, start: usize) -> Option<&str> {
    let open = start + source[start..].find('{')?;
    let bytes = source.as_bytes();
    let mut index = open;
    let mut depth = 0usize;
    let mut string = false;
    let mut escaped = false;
    let mut line_comment = false;
    let mut block_comment_depth = 0usize;
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
                return Some(&source[start..=index]);
            }
        }
        index += 1;
    }
    None
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
            .find(|character: char| {
                character == '(' || character == '<' || character.is_whitespace()
            })
            .map(|relative| name_start + relative)
            .unwrap_or(open);

        let Some(body) = braced_block(source, start) else {
            break;
        };
        let close = start + body.len();
        blocks.push(FunctionBlock {
            name: &source[name_start..name_end],
            signature,
            body,
        });
        offset = close;
    }
    blocks
}

fn code_only(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut code = String::with_capacity(source.len());
    let mut index = 0;
    let mut string = false;
    let mut escaped = false;
    let mut line_comment = false;
    let mut block_comment_depth = 0usize;
    while index < bytes.len() {
        let byte = bytes[index];
        let next = bytes.get(index + 1).copied();
        if line_comment {
            if byte == b'\n' {
                line_comment = false;
                code.push('\n');
            } else {
                code.push(' ');
            }
        } else if block_comment_depth > 0 {
            if byte == b'/' && next == Some(b'*') {
                block_comment_depth += 1;
                code.push_str("  ");
                index += 1;
            } else if byte == b'*' && next == Some(b'/') {
                block_comment_depth -= 1;
                code.push_str("  ");
                index += 1;
            } else if byte == b'\n' {
                code.push('\n');
            } else {
                code.push(' ');
            }
        } else if string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                string = false;
            }
            code.push(if byte == b'\n' { '\n' } else { ' ' });
        } else if byte == b'/' && next == Some(b'/') {
            line_comment = true;
            code.push_str("  ");
            index += 1;
        } else if byte == b'/' && next == Some(b'*') {
            block_comment_depth = 1;
            code.push_str("  ");
            index += 1;
        } else if byte == b'"' {
            string = true;
            code.push(' ');
        } else {
            code.push(byte as char);
        }
        index += 1;
    }
    code
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

fn runtime_host_sync_methods() -> BTreeSet<String> {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("runtime.rs"),
    )
    .expect("read CUDA runtime source");
    let production = source
        .split_once("#[cfg(test)]\nmod tests")
        .map_or(source.as_str(), |(before_tests, _)| before_tests);
    let functions = function_blocks(production);
    let mut synchronizing = functions
        .iter()
        .filter(|function| {
            function.name != "drop" && code_only(function.body).contains(".synchronize()")
        })
        .map(|function| function.name.to_string())
        .collect::<BTreeSet<_>>();

    loop {
        let mut discovered = BTreeSet::new();
        for function in &functions {
            if synchronizing.contains(function.name) {
                continue;
            }
            let body = code_only(function.body);
            if synchronizing.iter().any(|callee| {
                body.contains(&format!("self.{callee}("))
                    || body.contains(&format!("self.{callee} ("))
            }) {
                discovered.insert(function.name.to_string());
            }
        }
        if discovered.is_subset(&synchronizing) {
            break;
        }
        synchronizing.extend(discovered);
    }

    synchronizing
}

fn single_named_function<'a>(source: &'a str, name: &str, source_name: &str) -> FunctionBlock<'a> {
    let mut matches = function_blocks(source)
        .into_iter()
        .filter(|function| function.name == name);
    let function = matches
        .next()
        .unwrap_or_else(|| panic!("{source_name} has no function named {name}"));
    assert!(
        matches.next().is_none(),
        "{source_name} has more than one function named {name}; the production call-graph \
         contract needs an unambiguous owner"
    );
    function
}

fn impl_type_name(signature: &str, kernel_impl: bool) -> Option<&str> {
    let prefix = if kernel_impl {
        "impl Kernel for "
    } else {
        "impl "
    };
    let rest = signature.strip_prefix(prefix)?.trim_start();
    let end = rest
        .find(|character: char| character == '<' || character.is_whitespace())
        .unwrap_or(rest.len());
    (end != 0).then_some(&rest[..end])
}

fn impl_blocks<'a>(source: &'a str, prefix: &str) -> Vec<(&'a str, &'a str)> {
    let mut blocks = Vec::new();
    let mut offset = 0;
    while let Some(relative) = source[offset..].find(prefix) {
        let start = offset + relative;
        let Some(open_relative) = source[start..].find('{') else {
            break;
        };
        let open = start + open_relative;
        let signature = source[start..open].trim();
        let Some(block) = braced_block(source, start) else {
            break;
        };
        let close = start + block.len();
        blocks.push((signature, block));
        offset = close;
    }
    blocks
}

fn calls_method(body: &str, method: &str) -> bool {
    body.contains(&format!(".{method}(")) || body.contains(&format!(".{method} ("))
}

fn calls_function(body: &str, function: &str) -> bool {
    let needle = format!("{function}(");
    body.match_indices(&needle).any(|(index, _)| {
        index == 0
            || !matches!(
                body.as_bytes()[index - 1],
                b'.' | b':' | b'_' | b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9'
            )
    })
}

fn reachable_host_syncs(
    source: &str,
    kernel_impl: &str,
    kernel_type: &str,
    runtime_syncs: &BTreeSet<String>,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut functions = BTreeMap::new();
    for function in function_blocks(source) {
        functions
            .entry(function.name.to_string())
            .or_insert(function);
    }
    for (signature, block) in impl_blocks(source, "impl ") {
        if impl_type_name(signature, false) != Some(kernel_type) {
            continue;
        }
        for function in function_blocks(block) {
            functions.insert(function.name.to_string(), function);
        }
    }
    for function in function_blocks(kernel_impl) {
        functions.insert(function.name.to_string(), function);
    }

    let mut queue = VecDeque::from([
        "execute".to_string(),
        "execute_with_workspace".to_string(),
        "prepare_kernel_sized_device".to_string(),
        "materialize_kernel_sized_device".to_string(),
    ]);
    let mut visited = BTreeSet::new();
    let mut hazards = BTreeMap::new();
    while let Some(name) = queue.pop_front() {
        if !visited.insert(name.clone()) {
            continue;
        }
        let Some(function) = functions.get(&name) else {
            continue;
        };
        let body = code_only(function.body);
        // A guard at the caller dominates every helper reached from it. This is
        // the same accepted shape as the direct-sync census above: the runtime
        // refuses the path before any host-synchronizing operation is reached.
        let guarded = body.contains("is_capturing()")
            || (function.signature.contains("sync: bool") && body.contains("if sync"))
            || (function.signature.contains("capturing")
                && (body.contains("if capturing") || body.contains("if !capturing")));
        if guarded {
            continue;
        }

        let called_syncs = runtime_syncs
            .iter()
            .filter(|method| calls_method(&body, method))
            .cloned()
            .collect::<BTreeSet<_>>();
        if !called_syncs.is_empty() {
            hazards.insert(name.clone(), called_syncs);
        }

        for candidate in functions.keys() {
            if !visited.contains(candidate)
                && (calls_method(&body, candidate) || calls_function(&body, candidate))
            {
                queue.push_back(candidate.clone());
            }
        }
    }
    hazards
}

struct ReviewedDynamicOutputSync {
    file: &'static str,
    function: &'static str,
    kernel: &'static str,
    operator: &'static str,
}

fn reviewed_dynamic_output_syncs() -> [ReviewedDynamicOutputSync; 2] {
    [
        ReviewedDynamicOutputSync {
            file: "non_max_suppression.rs",
            function: "materialize",
            kernel: "NonMaxSuppressionKernel",
            operator: "NonMaxSuppression",
        },
        ReviewedDynamicOutputSync {
            file: "unique.rs",
            function: "materialize",
            kernel: "UniqueKernel",
            operator: "Unique",
        },
    ]
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
        "fused_gemm.rs::run".to_string(),
        "gemm.rs::run".to_string(),
        "matmul_nbits.rs::run".to_string(),
        "mod_op.rs::run".to_string(),
        // Declares CaptureSupport::unsupported: the Phase-2a workspace path
        // allocates per-call scratch and drains the trailing transpose before
        // returning it to the pool. Same shape as packed_varlen_attention below.
        "multi_head_attention.rs::synchronize_runtime".to_string(),
        "nary.rs::run".to_string(),
        // DeviceWorkspace prepare must copy the scalar count D2H so ORT can
        // allocate the dynamic output before materialize. The per-kernel audit
        // below binds this entry to that explicit capture refusal.
        "non_max_suppression.rs::materialize".to_string(),
        "nonzero.rs::execute".to_string(),
        "packed_varlen_attention.rs::execute".to_string(),
        "pooling.rs::execute".to_string(),
        "pooling.rs::run".to_string(),
        "sparse_kv_gather.rs::execute".to_string(),
        // Same two-phase DeviceWorkspace contract as NonMaxSuppression: the
        // output extent is known only after a scalar count D2H and ORT allocates.
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

/// Production reachability contract for asynchronous H2D page-in fencing.
///
/// This is intentionally separate from the CUDA outcome tests. It proves that
/// both production upload entry points record a copy-stream fence, the EP wait
/// dispatch reaches the runtime fence registry, and the uniquely removed event
/// is passed to `CudaStream::wait`. It does not claim that deleting the wait
/// forces a particular result under an otherwise identical CUDA schedule.
#[test]
fn production_async_pagein_reaches_cuda_stream_wait() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let read_production = |file: &str| {
        let mut source = fs::read_to_string(root.join(file))
            .unwrap_or_else(|error| panic!("read CUDA production source {file}: {error}"));
        if let Some(tests) = source.find("#[cfg(test)]\nmod tests") {
            source.truncate(tests);
        }
        source
    };

    let provider = read_production("provider.rs");
    let weight_paging = read_production("weight_paging.rs");
    let runtime = read_production("runtime.rs");

    let provider_copy = single_named_function(&provider, "copy_async", "provider.rs");
    let provider_copy = code_only(provider_copy.body);
    assert!(
        calls_method(&provider_copy, "htod_async")
            && calls_method(&provider_copy, "record_copy_fence"),
        "CudaExecutionProvider::copy_async must enqueue H2D and record its copy-stream fence"
    );

    let page_upload =
        single_named_function(&weight_paging, "upload_async_inner", "weight_paging.rs");
    let page_upload = code_only(page_upload.body);
    assert!(
        calls_method(&page_upload, "htod_async") && calls_method(&page_upload, "record_copy_fence"),
        "CudaWeightPage::upload_async_inner must enqueue H2D and record its copy-stream fence"
    );

    let provider_wait = single_named_function(&provider, "wait_fence", "provider.rs");
    assert!(
        calls_method(&code_only(provider_wait.body), "compute_wait_fence"),
        "CudaExecutionProvider::wait_fence must dispatch to CudaRuntime::compute_wait_fence"
    );

    let record = single_named_function(&runtime, "record_copy_fence", "runtime.rs");
    let record = code_only(record.body);
    assert!(
        calls_method(&record, "record_fence_on") && record.contains("&self.copy_stream"),
        "record_copy_fence must register an event recorded on the copy stream"
    );

    let compute_wait = single_named_function(&runtime, "compute_wait_fence", "runtime.rs");
    assert!(
        calls_method(&code_only(compute_wait.body), "wait_fence_on"),
        "compute_wait_fence must reach the production fence-registry dispatcher"
    );

    let wait_fence_on = single_named_function(&runtime, "wait_fence_on", "runtime.rs");
    let wait_fence_code = code_only(wait_fence_on.body);
    assert!(
        wait_fence_on.signature.contains("waiter: &CudaStream")
            && calls_function(&wait_fence_code, "dispatch_registered_fence_wait"),
        "wait_fence_on must bind a CudaStream waiter to the shared registry-dispatch core"
    );
    assert_eq!(
        wait_fence_code.matches(".wait(").count(),
        1,
        "the production registry dispatch must invoke exactly one CudaStream::wait(event)"
    );
    assert!(
        wait_fence_code.contains(".wait(event)"),
        "the event removed by the registry core must be the event passed to CudaStream::wait"
    );

    let dispatch = single_named_function(&runtime, "dispatch_registered_fence_wait", "runtime.rs");
    let dispatch = code_only(dispatch.body);
    assert!(
        calls_method(&dispatch, "remove") && calls_function(&dispatch, "wait"),
        "the shared core must remove event ownership before invoking its backend wait"
    );

    for forbidden in ["StreamWaitOperation", "PRODUCTION_STREAM_WAIT"] {
        assert!(
            !runtime.contains(forbidden),
            "production wait dispatch must not expose the bypass selector {forbidden}"
        );
    }
}

/// Follow every CUDA kernel entry point through its local helper calls and then
/// through `CudaRuntime` methods until a concrete host synchronization is
/// reached. Capture support is safe only when that reachable path is guarded
/// before the synchronization or the same kernel can explicitly decline
/// capture.
///
/// This complements the direct `.synchronize()` census above. In particular,
/// moving a barrier behind a runtime helper must not make it disappear from the
/// contract merely because the kernel file no longer spells the low-level call.
#[test]
fn capture_supported_kernels_do_not_reach_host_synchronizing_runtime_operations() {
    let root = kernel_root();
    let runtime_syncs = runtime_host_sync_methods();
    let mut violations = Vec::new();

    for entry in fs::read_dir(&root).expect("read CUDA kernel sources") {
        let path = entry.expect("kernel source entry").path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }
        let source = fs::read_to_string(&path).expect("read CUDA kernel source");
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source.as_str(), |(before_tests, _)| before_tests);

        for (signature, kernel_impl) in impl_blocks(production, "impl Kernel for ") {
            let Some(kernel_type) = impl_type_name(signature, true) else {
                continue;
            };
            let hazards =
                reachable_host_syncs(production, kernel_impl, kernel_type, &runtime_syncs);
            if hazards.is_empty() {
                continue;
            }
            let capture_support = function_blocks(kernel_impl)
                .into_iter()
                .find(|function| function.name == "capture_support");
            let can_decline = capture_support.is_some_and(|function| {
                code_only(function.body).contains("CaptureSupport::unsupported")
            });
            if !can_decline {
                let file = path.file_name().unwrap().to_string_lossy();
                let reasons = hazards
                    .into_iter()
                    .map(|(function, methods)| {
                        format!(
                            "{function} -> CudaRuntime::{}",
                            methods.into_iter().collect::<Vec<_>>().join("/")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                violations.push(format!("{file}::{kernel_type} reaches {reasons}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "capture-supported CUDA kernels may not reach host-synchronizing runtime operations; \
         guard the path before synchronization or return CaptureSupport::unsupported with the \
         failed precondition:\n{}",
        violations.join("\n")
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
/// those entries are listed. The reviewed dynamic-output entries have the
/// stronger per-kernel check below; the remaining entries keep this lower bound.
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

#[test]
fn dynamic_output_syncs_are_bound_to_their_capture_refusal() {
    let root = kernel_root();
    for reviewed in reviewed_dynamic_output_syncs() {
        let path = root.join(reviewed.file);
        let source = fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!("read reviewed kernel source {}: {error}", reviewed.file)
        });
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source.as_str(), |(before_tests, _)| before_tests);

        let inherent_header = format!("impl {} ", reviewed.kernel);
        let inherent_start = production.find(&inherent_header).unwrap_or_else(|| {
            panic!(
                "{} has no inherent impl for {}",
                reviewed.file, reviewed.kernel
            )
        });
        let inherent = braced_block(production, inherent_start)
            .unwrap_or_else(|| panic!("parse inherent impl for {}", reviewed.kernel));
        let materialize = function_blocks(inherent)
            .into_iter()
            .find(|block| block.name == reviewed.function)
            .unwrap_or_else(|| {
                panic!(
                    "{} has no {}::{} method",
                    reviewed.file, reviewed.kernel, reviewed.function
                )
            });
        assert!(
            materialize.body.contains(".synchronize()") && !has_capture_guard(&materialize),
            "{}::{} is reviewed only while it retains its unconditional synchronization",
            reviewed.file,
            reviewed.function
        );

        let kernel_header = format!("impl Kernel for {} ", reviewed.kernel);
        let kernel_start = production.find(&kernel_header).unwrap_or_else(|| {
            panic!(
                "{} has no Kernel impl for {}",
                reviewed.file, reviewed.kernel
            )
        });
        let kernel_impl = braced_block(production, kernel_start)
            .unwrap_or_else(|| panic!("parse Kernel impl for {}", reviewed.kernel));
        let methods = function_blocks(kernel_impl);
        let policy = methods
            .iter()
            .find(|method| method.name == "kernel_sized_output_policy")
            .unwrap_or_else(|| {
                panic!(
                    "{} must explicitly declare its dynamic output policy",
                    reviewed.operator
                )
            });
        assert!(
            policy
                .body
                .contains("KernelSizedOutputPolicy::DeviceWorkspace"),
            "{} is allowlisted only for the DeviceWorkspace two-phase path",
            reviewed.operator
        );
        assert!(
            methods
                .iter()
                .any(|method| method.name == "prepare_kernel_sized_device")
                && methods
                    .iter()
                    .any(|method| method.name == "materialize_kernel_sized_device"),
            "{} must retain both DeviceWorkspace phases",
            reviewed.operator
        );

        let capture = methods
            .iter()
            .find(|method| method.name == "capture_support")
            .unwrap_or_else(|| {
                panic!(
                    "{} must explicitly declare capture support",
                    reviewed.operator
                )
            });
        for required in [
            "CaptureSupport::unsupported",
            "DeviceWorkspace two-phase path",
            "8-byte count D2H synchronization",
            "dynamic ORT output allocation",
        ] {
            assert!(
                capture.body.contains(required),
                "{} capture refusal must retain the load-bearing reason fragment {required:?}",
                reviewed.operator
            );
        }
    }
}
