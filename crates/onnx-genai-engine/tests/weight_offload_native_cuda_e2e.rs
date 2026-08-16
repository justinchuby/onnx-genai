//! Live GPU weight-offload correctness lock (issue #63, GAP A/B/C).
//!
//! Proves that paging quantized weights host↔device DURING native CUDA decode is
//! transparent: with `ONNX_GENAI_WEIGHT_OFFLOAD=1` and a VRAM residency budget
//! small enough to force page-ins AND evictions, the greedy token stream is
//! byte-identical to the non-offloaded resident path. The model is Qwen3-0.6B
//! int4 (MatMulNBits), whose native CUDA output is already locked by
//! `qwen3_0_6b_native_cuda_e2e`, so any drift here is the offload path's fault.
//!
//! The test asserts the process-global offload counters (`page_ins`,
//! `evictions`) are both > 0, so it cannot silently pass with paging disabled.
//!
//! ```bash
//! QWEN3_0_6B_CUDA_E2E_DIR=/path/to/qwen3-0.6b-int4-cuda-postfix \
//! CUDA_VISIBLE_DEVICES=0 taskset -c 0 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend --test weight_offload_native_cuda_e2e \
//!   -- --ignored --nocapture
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, GenerateResult, NativeDecodeDevice,
    ResourceLimit, ResourceLimits,
};

const DEFAULT_MODEL_DIR: &str = "/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda-postfix";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 32;
/// VRAM residency budget for the offloaded run: deliberately tiny (2 MiB) so the
/// int4 MatMulNBits weights cannot all stay resident, forcing real page-ins and
/// LRU evictions each decode step.
const TINY_DEVICE_BUDGET_BYTES: u64 = 2 * 1024 * 1024;

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("QWEN3_0_6B_CUDA_E2E_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
    let required = [
        "model.onnx",
        "model.onnx.data",
        "inference_metadata.yaml",
        "tokenizer.json",
    ];
    let missing: Vec<_> = required
        .iter()
        .filter(|name| !dir.join(name).is_file())
        .collect();
    if missing.is_empty() {
        Some(dir)
    } else {
        eprintln!(
            "skipping weight-offload native CUDA e2e: model directory {} is missing {:?}",
            dir.display(),
            missing
        );
        None
    }
}

fn generate(dir: &Path) -> anyhow::Result<GenerateResult> {
    let mut engine = Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            ..EngineConfig::default()
        },
    )?;
    let mut request = GenerateRequest::new(PROMPT.to_string());
    request.options.max_new_tokens = MAX_NEW_TOKENS;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    engine.generate(request)
}

/// True when a native load failure is the #384 load-chain metadata gap (the
/// export does not declare its `model.io` ports) rather than an offload bug.
fn is_load_chain_metadata_gap(error: &anyhow::Error) -> bool {
    let message = format!("{error:#}");
    message.contains("model.io")
        || message.contains("token_input")
        || message.contains("kv_inputs")
        || message.contains("explicit decoder state")
}

#[test]
#[ignore = "requires the real Qwen3-0.6B int4 postfix export and a CUDA device"]
fn offloaded_native_cuda_decode_is_token_identical_and_pages() -> anyhow::Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping weight-offload native CUDA e2e: CUDA unavailable: {error}");
        return Ok(());
    }

    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
    }

    // --- Baseline: offload OFF (resident fast path, the trusted output) --------
    unsafe {
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_ENV);
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_DEVICE_BYTES_ENV);
    }
    onnx_runtime_ep_cuda::reset_global_offload_stats();
    let baseline_start = Instant::now();
    let baseline = match generate(&dir) {
        Ok(result) => result,
        // The native loader needs the export's `model.io` ports (token_input,
        // kv_inputs/kv_outputs) declared in inference_metadata.yaml. Auto-wiring
        // that from the graph is the separate #384 load-chain work; until it
        // lands, an export without a declared `io:` block cannot load natively.
        // Skip rather than fail so this offload lock isn't a landmine.
        Err(error) if is_load_chain_metadata_gap(&error) => {
            eprintln!(
                "skipping weight-offload native CUDA e2e: export {} lacks declared \
                 model.io ports (blocked on #384 load-chain): {error:#}",
                dir.display()
            );
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let baseline_elapsed = baseline_start.elapsed();
    let baseline_stats = onnx_runtime_ep_cuda::global_offload_stats();
    assert_eq!(
        baseline_stats.page_ins, 0,
        "offload was disabled yet the residency cache paged weights in: {baseline_stats:?}"
    );

    // --- Offloaded: ONNX_GENAI_WEIGHT_OFFLOAD=1 with a tiny VRAM budget --------
    unsafe {
        std::env::set_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_ENV, "1");
        std::env::set_var(
            onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_DEVICE_BYTES_ENV,
            TINY_DEVICE_BUDGET_BYTES.to_string(),
        );
    }
    onnx_runtime_ep_cuda::reset_global_offload_stats();
    // Lock the new default: with only WEIGHT_OFFLOAD=1 set (async var unset), the
    // resolved policy uses async page-in so dense lookahead can overlap the known
    // next layer. The old synchronous path remains available via
    // ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=0 for A/B.
    assert!(
        onnx_runtime_ep_cuda::DeviceOffloadPolicy::from_env().async_pagein,
        "async page-in must be the default: otherwise dense prefetch cannot engage"
    );
    let offloaded_start = Instant::now();
    let offloaded = generate(&dir)?;
    let offloaded_elapsed = offloaded_start.elapsed();
    let offloaded_stats = onnx_runtime_ep_cuda::global_offload_stats();

    unsafe {
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_ENV);
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_DEVICE_BYTES_ENV);
    }

    // --- Proof 1: paging actually happened (page-ins AND evictions) -----------
    assert!(
        offloaded_stats.page_ins > 0,
        "expected weight page-ins under a {TINY_DEVICE_BUDGET_BYTES}-byte budget, got {offloaded_stats:?}"
    );
    assert!(
        offloaded_stats.evictions > 0,
        "expected LRU evictions under a {TINY_DEVICE_BUDGET_BYTES}-byte budget, got {offloaded_stats:?}"
    );

    // --- Proof 2: offload is transparent (token-exact) ------------------------
    assert_eq!(
        offloaded.token_ids, baseline.token_ids,
        "weight offload changed the greedy token stream — offload must be an \
         optimization, never an output change.\nbaseline={:?}\noffloaded={:?}",
        baseline.token_ids, offloaded.token_ids
    );

    let tokens = MAX_NEW_TOKENS as f64;
    let baseline_tok_s = tokens / baseline_elapsed.as_secs_f64();
    let offloaded_tok_s = tokens / offloaded_elapsed.as_secs_f64();
    eprintln!(
        "weight-offload native CUDA e2e OK: tokens={:?}\n  \
         page_ins={}, evictions={}, hits={}\n  \
         baseline {:.2} tok/s ({:?}); offloaded {:.2} tok/s ({:?}); \
         slowdown {:.2}x",
        offloaded.token_ids,
        offloaded_stats.page_ins,
        offloaded_stats.evictions,
        offloaded_stats.hits,
        baseline_tok_s,
        baseline_elapsed,
        offloaded_tok_s,
        offloaded_elapsed,
        baseline_elapsed.as_secs_f64() / offloaded_elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
    );
    Ok(())
}

#[test]
#[ignore = "requires the real Qwen3-0.6B int4 postfix export and a CUDA device"]
fn vram_limit_auto_enables_offload_or_fails_at_load() -> anyhow::Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping VRAM-limit native CUDA e2e: CUDA unavailable: {error}");
        return Ok(());
    }

    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_ENV);
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_DEVICE_BYTES_ENV);
    }
    onnx_runtime_ep_cuda::reset_global_offload_stats();

    let load = Engine::from_dir(
        &dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            limits: ResourceLimits {
                vram_limit: ResourceLimit::Bytes(TINY_DEVICE_BUDGET_BYTES),
                ..ResourceLimits::default()
            },
            ..EngineConfig::default()
        },
    );

    let mut engine = match load {
        Ok(engine) => engine,
        Err(error) => {
            let message = format!("{error:#}");
            assert!(
                message.contains("cannot grant")
                    || message.contains("requires")
                    || message.contains("allows"),
                "load failed, but not with admission arithmetic: {message}"
            );
            return Ok(());
        }
    };

    let used = engine
        .governor()
        .leased_bytes_on(onnx_runtime_memory_governor::Tier::Device);
    let limit = engine
        .resource_snapshot()
        .resolved_limits
        .vram_bytes
        .expect("native CUDA load resolves a measured device VRAM budget");
    assert!(
        used <= limit,
        "load admitted {used} committed device bytes under a {limit} byte VRAM limit"
    );

    let mut request = GenerateRequest::new(PROMPT.to_string());
    request.options.max_new_tokens = 1;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    match engine.generate(request) {
        Ok(_) => {
            let stats = onnx_runtime_ep_cuda::global_offload_stats();
            assert!(
                stats.page_ins > 0,
                "generation succeeded under a too-small explicit VRAM limit without automatic \
                 weight offload: {stats:?}"
            );
        }
        Err(error) => {
            let message = format!("{error:#}");
            assert!(
                message.contains("weight-residency cache")
                    || message.contains("cannot grant")
                    || message.contains("bytes beyond"),
                "generation failed, but not with residency arithmetic: {message}"
            );
        }
    }

    Ok(())
}
