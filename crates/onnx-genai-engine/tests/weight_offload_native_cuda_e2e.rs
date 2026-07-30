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
};

const DEFAULT_MODEL_DIR: &str = "/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda-postfix";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 32;
/// VRAM residency budget for the offloaded run: deliberately tiny (2 MiB) so the
/// int4 MatMulNBits weights cannot all stay resident, forcing real page-ins and
/// LRU evictions each decode step.
const TINY_DEVICE_BUDGET_BYTES: u64 = 2 * 1024 * 1024;
/// A *realistic* residency budget for the prefetch A/B: large enough that many
/// weights stay resident (so decode isn't purely H2D-bound), yet small enough
/// that a meaningful fraction still pages + evicts each step. This is the regime
/// where async page-in overlap can actually hide latency. Overridable so the
/// budget can be swept without recompiling.
const REALISTIC_DEVICE_BUDGET_BYTES: u64 = 64 * 1024 * 1024;
/// Timed runs per configuration for the perf A/B (median reported); one warmup
/// run precedes these and is discarded.
const AB_RUNS: usize = 3;

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

/// Median of a small sample (sorted, middle element).
fn median_secs(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// The residency budget for the prefetch A/B, overridable via
/// `WEIGHT_OFFLOAD_AB_BUDGET_BYTES` so the paging fraction can be swept.
fn ab_budget_bytes() -> u64 {
    std::env::var("WEIGHT_OFFLOAD_AB_BUDGET_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(REALISTIC_DEVICE_BUDGET_BYTES)
}

/// Run one offloaded decode at `budget`, `prefetch` on/off. Returns the token
/// stream, the wall-clock elapsed, and the global offload counters after the run
/// (reset immediately before). Env is restored to a clean offload-off state.
fn offloaded_run(
    dir: &Path,
    budget: u64,
    prefetch: bool,
) -> anyhow::Result<(
    GenerateResult,
    std::time::Duration,
    onnx_runtime_ep_cuda::GlobalOffloadStats,
)> {
    unsafe {
        std::env::set_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_ENV, "1");
        std::env::set_var(
            onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_DEVICE_BYTES_ENV,
            budget.to_string(),
        );
        if prefetch {
            std::env::set_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_PREFETCH_ENV, "1");
        } else {
            std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_PREFETCH_ENV);
        }
    }
    onnx_runtime_ep_cuda::reset_global_offload_stats();
    let start = Instant::now();
    let result = generate(dir)?;
    let elapsed = start.elapsed();
    let stats = onnx_runtime_ep_cuda::global_offload_stats();
    unsafe {
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_ENV);
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_DEVICE_BYTES_ENV);
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_PREFETCH_ENV);
    }
    Ok((result, elapsed, stats))
}

/// #87 Increment 1 proof: async copy-stream page-in (prefetch ON) is token-exact
/// with the synchronous page-in (prefetch OFF), and the async path actually ran
/// (`async_page_ins > 0`). Also reports a perf A/B (median tok/s both ways) at a
/// realistic budget so the overlap benefit — or its absence — is recorded with
/// real numbers, not asserted.
#[test]
#[ignore = "requires the real Qwen3-0.6B int4 postfix export and a CUDA device"]
fn async_prefetch_native_cuda_decode_matches_sync_offload_and_pages() -> anyhow::Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping async-prefetch native CUDA e2e: CUDA unavailable: {error}");
        return Ok(());
    }
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
    }
    let budget = ab_budget_bytes();

    // --- Baseline: offload OFF (the trusted resident output) ------------------
    unsafe {
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_ENV);
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_DEVICE_BYTES_ENV);
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_PREFETCH_ENV);
    }
    onnx_runtime_ep_cuda::reset_global_offload_stats();
    let baseline = match generate(&dir) {
        Ok(result) => result,
        Err(error) if is_load_chain_metadata_gap(&error) => {
            eprintln!(
                "skipping async-prefetch native CUDA e2e: export {} lacks declared \
                 model.io ports (blocked on #384 load-chain): {error:#}",
                dir.display()
            );
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    // --- Offload ON, prefetch OFF (synchronous page-in) -----------------------
    let (sync_off, _sync_elapsed, sync_stats) = offloaded_run(&dir, budget, false)?;
    assert!(
        sync_stats.page_ins > 0 && sync_stats.evictions > 0,
        "realistic budget {budget} did not force paging+eviction (prefetch OFF): {sync_stats:?} \
         — lower WEIGHT_OFFLOAD_AB_BUDGET_BYTES"
    );
    assert_eq!(
        sync_stats.async_page_ins, 0,
        "prefetch was OFF yet async page-ins were recorded: {sync_stats:?}"
    );

    // --- Offload ON, prefetch ON (async copy-stream page-in) ------------------
    let (async_on, _async_elapsed, async_stats) = offloaded_run(&dir, budget, true)?;
    assert!(
        async_stats.page_ins > 0 && async_stats.evictions > 0,
        "realistic budget {budget} did not force paging+eviction (prefetch ON): {async_stats:?}"
    );
    assert!(
        async_stats.async_page_ins > 0,
        "prefetch ON but no async page-ins ran — the async path silently fell back: {async_stats:?}"
    );

    // --- Proof: async page-in is transparent (token-exact, both vs baseline) --
    assert_eq!(
        sync_off.token_ids, baseline.token_ids,
        "synchronous offload changed the token stream vs baseline"
    );
    assert_eq!(
        async_on.token_ids, sync_off.token_ids,
        "async prefetch page-in changed the greedy token stream vs synchronous \
         page-in — the copy fence is not ordering the H2D before the kernel.\n\
         sync={:?}\nasync={:?}",
        sync_off.token_ids, async_on.token_ids
    );

    // --- Perf A/B: median tok/s, prefetch OFF vs ON, at the realistic budget --
    let tokens = MAX_NEW_TOKENS as f64;
    // Warmup (discarded) then AB_RUNS timed runs per configuration.
    let _ = offloaded_run(&dir, budget, false)?;
    let mut off_secs = Vec::with_capacity(AB_RUNS);
    for _ in 0..AB_RUNS {
        off_secs.push(offloaded_run(&dir, budget, false)?.1.as_secs_f64());
    }
    let _ = offloaded_run(&dir, budget, true)?;
    let mut on_secs = Vec::with_capacity(AB_RUNS);
    for _ in 0..AB_RUNS {
        on_secs.push(offloaded_run(&dir, budget, true)?.1.as_secs_f64());
    }
    let off_median = median_secs(off_secs);
    let on_median = median_secs(on_secs);
    let off_tok_s = tokens / off_median;
    let on_tok_s = tokens / on_median;
    eprintln!(
        "async-prefetch native CUDA e2e OK (budget={} bytes): tokens={:?}\n  \
         prefetch OFF: page_ins={}, evictions={}, async_page_ins={}\n  \
         prefetch ON : page_ins={}, evictions={}, async_page_ins={}\n  \
         median tok/s  OFF={:.2} ({:.3}s)  ON={:.2} ({:.3}s)  speedup={:.3}x",
        budget,
        async_on.token_ids,
        sync_stats.page_ins,
        sync_stats.evictions,
        sync_stats.async_page_ins,
        async_stats.page_ins,
        async_stats.evictions,
        async_stats.async_page_ins,
        off_tok_s,
        off_median,
        on_tok_s,
        on_median,
        on_tok_s / off_tok_s,
    );
    Ok(())
}
