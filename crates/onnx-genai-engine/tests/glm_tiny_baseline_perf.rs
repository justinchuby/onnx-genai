//! Manual baseline performance probe for the tiny GLM-5.2 DSA/IndexShare and
//! `--glm-full-attention` fixtures (and, where present, the tiny DeepSeek-V4
//! QMoE fixture), run through the native CPU and CUDA decode backends.
//!
//! This is **not** a correctness gate — the fixtures are deliberately tiny
//! (random weights, ~1-2 layers) so their absolute tok/s numbers say nothing
//! about a real ~744GB checkpoint's performance. It exists to give the
//! `benchmark-moe-end-to-end` effort a *reproducible, structurally faithful*
//! starting point (real op mix: MLA/DSA IndexShare, fused QMoE, GQA/Attention)
//! while the real weights are unavailable, and to surface native CUDA
//! graph-capture/replay/fallback counters and a coarse VRAM delta. Ignored by
//! default (hardware/timing-sensitive, like the existing
//! `workflow_islands_are_competitive_with_native_composites` convention); run
//! explicitly:
//!
//! ```bash
//! CUDA_VISIBLE_DEVICES=1 cargo test -p onnx-genai-engine --features native-cuda \
//!   --test glm_tiny_baseline_perf -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "native-cuda")]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
};

const RUNS: usize = 3;
const DECODE_TOKENS: usize = 24;

fn fixture(name: &str) -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../tests/fixtures/{name}"));
    let has_model = dir.join("model.onnx").is_file() || dir.join("model.onnx.textproto").is_file();
    (has_model && dir.join("inference_metadata.yaml").is_file()).then_some(dir)
}

fn gpu_mem_used_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn range_report(label: &str, mut values: Vec<f64>, unit: &str) {
    let lo = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "  {label}: median={:.3}{unit} range=[{:.3}, {:.3}]{unit} (n={})",
        median(&mut values),
        lo,
        hi,
        values.len()
    );
}

/// One load + prefill(1 token) + decode(N tokens) run, timed in separate
/// phases so load/prefill/decode are never conflated into one number.
fn run_once(
    dir: &Path,
    backend: EngineDecodeBackend,
    device: Option<NativeDecodeDevice>,
) -> anyhow::Result<(Duration, Duration, Duration, usize)> {
    let load_start = Instant::now();
    let mut engine = Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: backend,
            native_device: device,
            ..EngineConfig::default()
        },
    )?;
    let load = load_start.elapsed();

    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![123]));
    request.options.max_new_tokens = 1;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    let prefill_start = Instant::now();
    engine.generate(request)?;
    let prefill = prefill_start.elapsed();

    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![123]));
    request.options.max_new_tokens = DECODE_TOKENS;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    let decode_start = Instant::now();
    let result = engine.generate(request)?;
    let decode = decode_start.elapsed();

    if let Some(stats) = engine.native_cuda_debug_stats() {
        eprintln!(
            "    cuda graph: captures={} replays={} fallbacks={} invalidations={} \
             kv_committed_bytes={} kv_transfers={:?}",
            stats.graph.captures,
            stats.graph.replays,
            stats.graph.fallbacks,
            stats.graph.invalidations,
            stats.kv_committed_bytes,
            stats.kv_transfers,
        );
    }
    Ok((load, prefill, decode, result.token_ids.len()))
}

fn bench(
    label: &str,
    dir: &Path,
    backend: EngineDecodeBackend,
    device: Option<NativeDecodeDevice>,
) {
    eprintln!("== {label} ({}) ==", dir.display());
    let mem_before = gpu_mem_used_mib();
    let mut loads = Vec::with_capacity(RUNS);
    let mut prefills = Vec::with_capacity(RUNS);
    let mut decode_tps = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        match run_once(dir, backend, device.clone()) {
            Ok((load, prefill, decode, tokens)) => {
                loads.push(load.as_secs_f64() * 1000.0);
                prefills.push(prefill.as_secs_f64() * 1000.0);
                let decode_steps = tokens.saturating_sub(1).max(1) as f64;
                decode_tps.push(decode_steps / decode.as_secs_f64());
                eprintln!(
                    "  run {run}: load={:.2}ms prefill(TTFT)={:.2}ms decode={:.2}ms for {tokens} tokens",
                    load.as_secs_f64() * 1000.0,
                    prefill.as_secs_f64() * 1000.0,
                    decode.as_secs_f64() * 1000.0,
                );
            }
            Err(error) => {
                eprintln!("  run {run}: FAILED: {error:#}");
                return;
            }
        }
    }
    range_report("load", loads, "ms");
    range_report("prefill (TTFT)", prefills, "ms");
    range_report("decode throughput", decode_tps, " tok/s");
    if let (Some(before), Some(after)) = (mem_before, gpu_mem_used_mib()) {
        eprintln!(
            "  GPU 0 memory.used: before={before}MiB after={after}MiB delta={}MiB",
            after as i64 - before as i64
        );
    }
}

#[test]
#[ignore = "hardware/timing-sensitive manual baseline probe, not a correctness gate"]
fn glm_tiny_baseline_perf() {
    let cuda_available = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0).is_ok();
    if !cuda_available {
        eprintln!("CUDA unavailable: CPU-only baseline");
    }

    for (name, dir) in [
        (
            "glm52-dsa-indexshare",
            fixture("tiny-glm52-qmoe-indexshare"),
        ),
        ("glm52-full-attention", fixture("tiny-glm52-full-attention")),
        ("deepseek-v4-qmoe", fixture("tiny-deepseek-v4-qmoe")),
    ] {
        let Some(dir) = dir else {
            eprintln!("skipping {name}: fixture not present");
            continue;
        };
        bench(
            &format!("{name} native CPU"),
            &dir,
            EngineDecodeBackend::Native,
            Some(NativeDecodeDevice::Cpu),
        );
        if cuda_available {
            bench(
                &format!("{name} native CUDA"),
                &dir,
                EngineDecodeBackend::Native,
                Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            );
        }
    }
}
