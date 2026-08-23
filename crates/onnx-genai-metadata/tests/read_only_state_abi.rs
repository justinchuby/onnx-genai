//! Read-only state aliases are frozen observations, not decode transitions.
//!
//! A component may bind a state group only to read a value another component
//! advanced — alternating CFG branches sharing a frozen prefix, or an
//! any-to-any stage consuming state an earlier stage produced. Such a binding
//! still names the graph's `present` output for kernel ABI reasons, but that
//! output is discarded, not a successor. The decode-step ABI lowering
//! (`decoder_io()` / `decoder_abi`) must therefore drop read-only KV and
//! static-cache aliases so a frozen reader never manufactures an
//! input/output transition pair or a fixed-capacity scatter ABI.

use std::path::{Path, PathBuf};

use onnx_genai_metadata::{InferenceMetadata, KvOwnership, StatePortAccess, load_metadata};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows")
        .join(name)
        .join("inference_metadata.yaml")
}

fn package(name: &str) -> InferenceMetadata {
    load_metadata(&fixture(name)).expect("package parses")
}

/// Mark every state alias in every group as a frozen read-only observation.
///
/// `decoder_io()` memoizes its result, so callers load a fresh package and
/// freeze it before the first resolution rather than mutating a resolved one.
fn freeze_all_state_aliases(metadata: &mut InferenceMetadata) {
    let serving = metadata
        .pipeline
        .as_mut()
        .expect("package has a pipeline")
        .workflow
        .serving
        .as_mut()
        .expect("package has a serving contract");
    for group in serving.state_service.groups.values_mut() {
        for bindings in group.ports.values_mut() {
            for alias in bindings.values_mut() {
                alias.access = StatePortAccess::ReadOnly;
            }
        }
    }
}

#[test]
fn read_only_kv_aliases_create_no_decode_transitions() {
    // Baseline: the writable full-attention cache is a real transition that the
    // decoder owns and advances.
    let writable = package("decoder");
    let baseline = writable.decoder_io().expect("writable ABI resolves");
    assert_eq!(baseline.kv_ownership, Some(KvOwnership::Owned));
    assert_eq!(
        baseline.kv_inputs.as_deref(),
        Some(["past_key_values.0.key".to_string()].as_slice())
    );
    assert_eq!(
        baseline.kv_outputs.as_deref(),
        Some(["present.0.key".to_string()].as_slice())
    );

    // Freezing the same aliases erases every KV transition pair. The component
    // still names `present.0.key` for the kernel, but the ABI must not alias it
    // back onto `past_key_values.0.key`, so no `past`/`present` pair survives
    // and the decoder no longer owns a cache it advances.
    let mut frozen = package("decoder");
    freeze_all_state_aliases(&mut frozen);
    let io = frozen.decoder_io().expect("frozen ABI still resolves");
    assert_eq!(io.kv_inputs, None);
    assert_eq!(io.kv_outputs, None);
    assert_eq!(io.kv_ownership, Some(KvOwnership::Shared));
    assert_eq!(io.static_cache, None);
}

#[test]
fn read_only_static_cache_aliases_are_dropped() {
    // Baseline: the fixed-capacity cache derives an indexed-scatter static-cache
    // ABI with graph-visible key/value halves.
    let writable = package("static_cache");
    let baseline = writable.decoder_io().expect("writable ABI resolves");
    assert!(
        baseline.static_cache.is_some(),
        "a writable fixed-capacity cache derives a static-cache ABI"
    );
    // A fixed-capacity cache is described by the static-cache ABI and by
    // nothing else. Reporting the same buffers *also* as growing past/present
    // pairs would be two answers for one cache, and the paged KV bridge reads
    // exactly this absence as "no growing cache to page" — so a decoder that
    // reported both would have a paged bridge built over a buffer that never
    // grows and addressed with the wrong discipline.
    assert_eq!(baseline.kv_inputs, None);
    assert_eq!(baseline.kv_outputs, None);
    assert_eq!(
        baseline.kv_ownership,
        Some(KvOwnership::Owned),
        "a fixed-capacity decoder still owns the cache it scatters into"
    );

    // Frozen: no key or value half survives the read-only filter, so neither the
    // scatter ABI nor any KV transition is derived.
    let mut frozen = package("static_cache");
    freeze_all_state_aliases(&mut frozen);
    let io = frozen.decoder_io().expect("frozen ABI still resolves");
    assert_eq!(io.static_cache, None);
    assert_eq!(io.kv_inputs, None);
    assert_eq!(io.kv_outputs, None);
    assert_eq!(io.kv_ownership, Some(KvOwnership::Shared));
}
