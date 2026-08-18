//! Regression lock for #1179: native CUDA decode of a default-domain
//! (multi-head) `::Attention` model must not fault at the first single-token
//! decode step with:
//!
//! ```text
//! node N (op '::Attention') reached execution without prepared
//! SessionPersistent workspace
//! ```
//!
//! Root cause (two coupled defects):
//!
//! 1. The default-domain `Attention` kernel classifies its composite workspace
//!    lifetime by *route*: multi-row query (prefill/verify, `q_seq > 1`) is
//!    `StepScoped`, single-row query (decode, `q_seq == 1`) is
//!    `SessionPersistent`. The prepare-only planner ran once against the
//!    multi-token prefill shape, so it only ever reserved the `StepScoped`
//!    slot; the `SessionPersistent` slot the decode route needs was never
//!    reserved. (GQA/MoE sidestep this because they charge `SessionPersistent`
//!    unconditionally.)
//! 2. On the eager growing-logical-KV decode path (graph capture declined
//!    because the KV cache exposes a growing logical prefix rather than fixed
//!    capacity), the governed `Attention` workspace scales with the *logical*
//!    attended length, which grows every step — but a prepared workspace slot
//!    may only ever grow, never per-call, so a once-prepared reservation sized
//!    to the logical length is undersized on the next step.
//!
//! The fix reserves the decode route's `SessionPersistent` slot up front by
//! driving one extra single-query-row prepare pass, and sizes growing/
//! logical-exposing KV inputs to their *physical capacity* during prepare-only
//! planning so the reservation is a valid upper bound for every in-bucket decode
//! step.
//!
//! This test drives the exact reproduction from the issue (the in-repo
//! `tiny-llm` fixture, greedy, several decode steps) on a real CUDA device. It
//! skips cleanly when no CUDA device / runtime is present, so it is safe in
//! CPU-only CI, but it *runs* on a GPU box. Reverting either half of the fix
//! turns it red (the `generate` call returns the workspace error above).
#![cfg(all(feature = "cuda", feature = "native-backend"))]

use std::path::PathBuf;

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, NativeDecodeDevice,
};

fn tiny_llm_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm")
}

/// Probe whether a usable CUDA device + runtime is present. Returns `false`
/// (skip) rather than failing when the box has no GPU or the CUDA runtime DLLs
/// are not discoverable.
fn cuda_available() -> bool {
    match onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        Ok(_) => true,
        Err(error) => {
            eprintln!(
                "skipping tiny-llm native CUDA decode workspace regression: \
                 CUDA unavailable: {error}"
            );
            false
        }
    }
}

#[test]
fn tiny_llm_native_cuda_decode_prepares_attention_workspace() -> anyhow::Result<()> {
    if !cuda_available() {
        return Ok(());
    }

    let dir = tiny_llm_dir()
        .canonicalize()
        .expect("tiny-llm fixture directory must exist");

    let mut engine = Engine::from_dir(
        &dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            ..EngineConfig::default()
        },
    )?;

    // Match the issue reproduction: a multi-token prompt (forces a `q_seq > 1`
    // prefill that classifies its Attention workspace as `StepScoped`) followed
    // by several `q_seq == 1` decode steps (which need the `SessionPersistent`
    // slot). `stop_on_eos = false` guarantees we actually take multiple decode
    // steps regardless of what the tiny fixture emits.
    let mut request = GenerateRequest::new("hello world");
    request.options.max_new_tokens = 8;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;

    // Before the fix this returns:
    //   native CUDA decoder forward pass failed
    //   Caused by: kernel execution failed: node 14 (op '::Attention') reached
    //   execution without prepared SessionPersistent workspace
    let result = engine.generate(request).map_err(|error| {
        anyhow::anyhow!(
            "native CUDA decode of the tiny-llm fixture must not fault preparing \
             the Attention workspace (#1179): {error:#}"
        )
    })?;

    assert_eq!(
        result.token_ids.len(),
        8,
        "native CUDA decode must produce every requested token; got {:?}",
        result.token_ids
    );
    eprintln!(
        "tiny-llm native CUDA decode OK (#1179): text={:?} tokens={:?}",
        result.text, result.token_ids
    );
    Ok(())
}
