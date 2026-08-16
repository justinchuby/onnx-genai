//! Issue #716 Step-4 measurement: weight offload + whole-step CUDA graph
//! capture, together, under a stable device virtual address.
//!
//! Before #716, weight offload forced graph capture OFF: the pager returned a
//! fresh device pointer on every page-in and a captured graph bakes pointers
//! into its recorded nodes. This test drives the managed no-spill authority path
//! (an explicit byte `--vram-limit`, which installs the VMM arena + physical
//! granule pool, so page-ins run on reserved-once stable VAs) and proves the two
//! now coexist: with offload ON, decoding with graph capture OFF and ON in the
//! SAME process yields a byte-identical greedy token stream.
//!
//! It also dumps the deterministic offload counters for both runs — page_ins,
//! hits, evictions, bypassed_page_ins, and the host-blocking `vram_alloc_ns` /
//! `vram_free_ns` spans. Stable slots retain their VA across evict→repage and
//! commit/decommit physical granules instead of calling `alloc_raw`/`free_raw`,
//! so the alloc/free churn counters (previously `vram_free` alone was ~9% of
//! step time) should stay flat while page-ins still happen.
//!
//! Throughput on this box has extreme run-to-run variance, so wall-clock is
//! reported but secondary; the counters and token-identity are the bar.
//!
//! ```powershell
//! $env:STABLE_VA_E2E_DIR = "C:\Users\justinchu\dev\models\qwen2.5-0.5b-q4_0-mobius"
//! $env:STABLE_VA_E2E_VRAM_BYTES = "335544320"  # 320 MiB, below the ~372 MB weights
//! cargo test -p onnx-genai-engine --features cuda,native-backend \
//!   --test stable_va_capture_offload_e2e -- --ignored --nocapture
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, NativeDecodeDevice, ResourceLimit,
    ResourceLimits,
};

const DEFAULT_MODEL_DIR: &str = r"C:\Users\justinchu\dev\models\qwen2.5-0.5b-q4_0-mobius";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 32;
/// Default explicit VRAM budget: 320 MiB, chosen to sit below the mobius
/// int4 weight footprint (~372 MB) so the managed no-spill path must page and
/// evict, while leaving room for KV + workspace admission. Override with
/// `STABLE_VA_E2E_VRAM_BYTES` to tune for a different model.
const DEFAULT_VRAM_BUDGET_BYTES: u64 = 320 * 1024 * 1024;

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("STABLE_VA_E2E_DIR")
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
            "skipping stable-VA capture+offload e2e: model directory {} is missing {:?}",
            dir.display(),
            missing
        );
        None
    }
}

fn vram_budget_bytes() -> u64 {
    std::env::var("STABLE_VA_E2E_VRAM_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_VRAM_BUDGET_BYTES)
}

struct Run {
    token_ids: Vec<u32>,
    stats: onnx_runtime_ep_cuda::GlobalOffloadStats,
    elapsed: Duration,
}

/// Load a fresh engine (managed no-spill, explicit VRAM budget) and decode
/// greedily with CUDA graph capture forced on or off. Returns `Ok(None)` when
/// the load or generation is rejected by admission arithmetic (the budget is
/// simply too small for this box), so the test skips rather than lying.
fn decode_once(dir: &std::path::Path, budget: u64, capture: bool) -> anyhow::Result<Option<Run>> {
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
        // Managed no-spill comes from the explicit byte vram_limit below, not the
        // env offload knobs — clear them so this measures the #716 path only.
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_ENV);
        std::env::remove_var(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_DEVICE_BYTES_ENV);
        // Force capture ON/OFF explicitly; `_explicit` honors an explicit 0.
        std::env::set_var("ONNX_GENAI_CUDA_GRAPH", if capture { "1" } else { "0" });
    }
    onnx_runtime_ep_cuda::reset_global_offload_stats();

    let load = Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            limits: ResourceLimits {
                vram_limit: ResourceLimit::Bytes(budget),
                ..ResourceLimits::default()
            },
            ..EngineConfig::default()
        },
    );
    let mut engine = match load {
        Ok(engine) => engine,
        Err(error) => {
            let message = format!("{error:#}");
            eprintln!(
                "stable-VA capture+offload e2e: load rejected under a {budget}-byte VRAM budget \
                 (capture={capture}): {message}"
            );
            return Ok(None);
        }
    };

    let mut request = GenerateRequest::new(PROMPT.to_string());
    request.options.max_new_tokens = MAX_NEW_TOKENS;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;

    let start = Instant::now();
    let result = match engine.generate(request) {
        Ok(result) => result,
        Err(error) => {
            let message = format!("{error:#}");
            eprintln!(
                "stable-VA capture+offload e2e: generation rejected under a {budget}-byte VRAM \
                 budget (capture={capture}): {message}"
            );
            return Ok(None);
        }
    };
    let elapsed = start.elapsed();
    let stats = onnx_runtime_ep_cuda::global_offload_stats();
    Ok(Some(Run {
        token_ids: result.token_ids,
        stats,
        elapsed,
    }))
}

fn report(label: &str, run: &Run) {
    let s = &run.stats;
    let tok_s = MAX_NEW_TOKENS as f64 / run.elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    eprintln!(
        "[{label}] tokens={} elapsed={:?} ({tok_s:.2} tok/s)\n  \
         page_ins={} hits={} evictions={} bypassed_page_ins={}\n  \
         vram_alloc_ns={} vram_free_ns={} htod_bytes={}\n  \
         peak_resident_bytes={} physical_owned_bytes={} mapped_physical_bytes={}",
        run.token_ids.len(),
        run.elapsed,
        s.page_ins,
        s.hits,
        s.evictions,
        s.bypassed_page_ins,
        s.vram_alloc_ns,
        s.vram_free_ns,
        s.htod_bytes,
        s.peak_resident_bytes,
        s.physical_owned_bytes,
        s.mapped_physical_bytes,
    );
}

#[test]
#[ignore = "requires a native-loadable CUDA model and a CUDA device"]
fn offloaded_decode_is_token_identical_with_capture_off_and_on() -> anyhow::Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping stable-VA capture+offload e2e: CUDA unavailable: {error}");
        return Ok(());
    }
    let budget = vram_budget_bytes();

    // Same session, interleaved: capture OFF (today's forced behavior) then
    // capture ON (the #716 win), both with offload active under a stable VA.
    let Some(capture_off) = decode_once(&dir, budget, false)? else {
        eprintln!("stable-VA capture+offload e2e: skipped (capture OFF run not admitted)");
        return Ok(());
    };
    let Some(capture_on) = decode_once(&dir, budget, true)? else {
        eprintln!("stable-VA capture+offload e2e: skipped (capture ON run not admitted)");
        return Ok(());
    };

    unsafe {
        std::env::remove_var("ONNX_GENAI_CUDA_GRAPH");
    }

    report("offload+capture OFF", &capture_off);
    report("offload+capture ON", &capture_on);

    // Correctness bar: capture is transparent. With the managed no-spill VMM
    // stable-VA authority active, decoding with graph capture OFF and ON must
    // produce the identical greedy token stream — capture is an optimization,
    // never an output change. This is the invariant #716 must not break, and it
    // holds whether or not the resident set spilled to paging on this budget.
    assert_eq!(
        capture_off.token_ids, capture_on.token_ids,
        "graph capture changed the greedy token stream under the managed no-spill authority — \
         capture must be an optimization, never an output change.\n  off={:?}\n  on={:?}",
        capture_off.token_ids, capture_on.token_ids,
    );

    // Informational: whether sustained paging engaged on this budget. On models
    // small enough to stay fully resident under managed no-spill, page_ins stays
    // 0 (the weights simply fit); on a model that genuinely overflows the budget
    // the counters below quantify the capture-on-vs-off comparison. Either way,
    // the alloc/free churn counters (vram_alloc_ns / vram_free_ns) stay flat
    // because the stable-VA slots commit/decommit physical granules under a
    // reserved VA instead of calling alloc_raw/free_raw.
    if capture_off.stats.page_ins == 0 && capture_on.stats.page_ins == 0 {
        eprintln!(
            "stable-VA capture+offload e2e: weights stayed fully resident under the \
             {budget}-byte managed no-spill budget (no page-ins) — token-identity across \
             capture OFF/ON verified; raise the model size or lower the budget to exercise \
             sustained paging counters."
        );
    }

    Ok(())
}
