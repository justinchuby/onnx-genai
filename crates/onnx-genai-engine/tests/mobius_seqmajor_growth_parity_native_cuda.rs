//! Real-model seq-major KV parity **with forced growth**, native CUDA decode.
//!
//! This is the end-to-end validation of the fixed-stride BSNH KV state build
//! (the documented next step after PR #797). #794 patched `qwen2.5-0.5b-q4_0-
//! mobius` with `kv_layout=1` and got identical 32 token IDs — but with the
//! default 256-token initial KV bucket a 32-token run **never grows**, so it
//! never exercised seq-major *growth* geometry, which is exactly what the engine
//! shape/stride build corrects. This harness forces growth with a tiny
//! `ONNX_GENAI_KV_MIN_BUCKET` and asserts the head-major and seq-major token
//! streams are byte-identical across several growth events, capture ON and OFF,
//! under both KV growth mechanisms. Every individual layout/configuration runs
//! in its own process because `ONNX_GENAI_CUDA_GRAPH` is a process-wide
//! `RuntimeConfig` value parsed once; mutating the environment after the first
//! engine load does not change policy.
//!
//! ## The two decoupled seq-major levers
//!
//! The engine growth geometry (`ONNX_GENAI_CUDA_KV_LAYOUT`) and the CUDA GQA
//! kernel stride arithmetic (the ONNX `kv_layout` node attribute) are two
//! *separate* levers — a correct seq-major run needs both. So this test uses two
//! model directories:
//!
//! * head-major: the stock export (GQA nodes have no `kv_layout` attribute →
//!   BNSH), run with the default engine layout;
//! * seq-major: a copy whose 24 GQA nodes carry `kv_layout=1` (→ BSNH), run with
//!   `ONNX_GENAI_CUDA_KV_LAYOUT=seq_major`.
//!
//! Patch the seq-major copy once with:
//!
//! ```python
//! import onnx
//! from onnx import helper
//! m = onnx.load("model.onnx", load_external_data=False)
//! for n in m.graph.node:
//!     if n.op_type == "GroupQueryAttention":
//!         n.attribute.append(helper.make_attribute("kv_layout", 1))
//! onnx.save(m, "model.onnx", save_as_external_data=False)  # keep model.onnx.data
//! ```
//!
//! ## The two KV growth mechanisms
//!
//! * **In-place VMM** (the default since #798's managed no-spill VMM;
//!   `commits_on_demand`): growth maps fresh granules onto the *same* base VA. A
//!   seq-major buffer's per-token stride `kv_heads*head_dim` is
//!   capacity-independent, so the live prefix keeps its byte offsets and **no KV
//!   data moves** (`d2d_copy_bytes == 0`), while head-major re-strides every head
//!   stripe. This is the fixed-stride property #797 measured at the driver level,
//!   here on the engine growth path.
//! * **Reallocation** (`ONNX_GENAI_LEGACY_ALLOCATOR=1`, the #755 opt-out): a
//!   bucket-sized growth allocates a *fresh* buffer and copies the live prefix
//!   into it. A new buffer must be filled either way, so seq-major copies the
//!   same byte count as head-major — its only win here is one contiguous copy
//!   instead of `kv_heads` stripes.
//!
//! Run:
//!
//! ```bash
//! MOBIUS_SEQMAJOR_HEAD_DIR=/models/qwen2.5-0.5b-q4_0-mobius \
//! MOBIUS_SEQMAJOR_SEQ_DIR=/models/qwen2.5-0.5b-q4_0-mobius-seqmajor \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend \
//!   --test mobius_seqmajor_growth_parity_native_cuda \
//!   -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Context;
use onnx_genai_engine::{
    CudaKvDebugStats, Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, GenerateResult,
    NativeDecodeDevice,
};
use serde::{Deserialize, Serialize};

const PROMPT: &str = "The capital of France is";
/// Long enough that a `min_bucket = 8` run crosses several power-of-two bucket
/// boundaries (8 → 16 → 32 → 64), so growth actually happens.
const MAX_NEW_TOKENS: usize = 48;
/// Force growth: initial and subsequent KV buckets round up from here.
const FORCE_GROWTH_MIN_BUCKET: &str = "8";
/// Hold the same 48-token generation in one bucket, providing a measured
/// no-growth control for attributing invalidations to capacity growth.
const NO_GROWTH_MIN_BUCKET: &str = "64";

const DEFAULT_HEAD_DIR: &str = r"C:\Users\justinchu\dev\models\qwen2.5-0.5b-q4_0-mobius";
const DEFAULT_SEQ_DIR: &str = r"C:\Users\justinchu\dev\models\qwen2.5-0.5b-q4_0-mobius-seqmajor";

fn resolve_dir(var: &str, default: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default));
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
            "skipping mobius seq-major growth parity: {} ({}) missing {}",
            var,
            dir.display(),
            missing
                .iter()
                .map(|name| name.as_ref())
                .collect::<Vec<&str>>()
                .join(", ")
        );
        None
    }
}

struct Run {
    result: GenerateResult,
    stats: Option<CudaKvDebugStats>,
}

#[derive(Clone, Copy)]
struct Mode {
    seq_major: bool,
    capture: bool,
    vmm_arena: bool,
    min_bucket: &'static str,
}

#[derive(Debug, Deserialize, Serialize)]
struct Measurement {
    label: String,
    token_ids: Vec<u32>,
    text: String,
    seq_major: bool,
    physical_committed_bytes: usize,
    growth_events: u64,
    kv_bytes_moved: u64,
    captures: u64,
    replays: u64,
    invalidations: u64,
    growth_keeps: u64,
    decline_reason: Option<String>,
    growth_decision: Option<String>,
}

const CHILD_LAYOUT_ENV: &str = "MOBIUS_SEQMAJOR_CHILD_LAYOUT";
const CHILD_CAPTURE_ENV: &str = "MOBIUS_SEQMAJOR_CHILD_CAPTURE";
const CHILD_VMM_ENV: &str = "MOBIUS_SEQMAJOR_CHILD_VMM";
const CHILD_MIN_BUCKET_ENV: &str = "MOBIUS_SEQMAJOR_CHILD_MIN_BUCKET";
const MEASUREMENT_PREFIX: &str = "MOBIUS_MEASUREMENT=";

/// Generate greedily on `dir` under `mode`. `ONNX_GENAI_KV_MIN_BUCKET` is set on
/// every run so both layouts exercise the same growth schedule.
fn generate(dir: &Path, mode: Mode) -> anyhow::Result<Run> {
    // SAFETY: single-threaded (`--test-threads=1`); we set process env before
    // constructing the engine, which reads these at load time.
    unsafe {
        std::env::set_var("ONNX_GENAI_KV_MIN_BUCKET", mode.min_bucket);
        std::env::set_var(
            "ONNX_GENAI_CUDA_GRAPH",
            if mode.capture { "1" } else { "0" },
        );
        if mode.vmm_arena {
            // In-place VMM arena (the default since #798's managed no-spill VMM).
            std::env::set_var("ONNX_GENAI_CUDA_VMM", "1");
            std::env::remove_var("ONNX_GENAI_LEGACY_ALLOCATOR");
        } else {
            // Force the legacy reallocation allocator. Since #798 the managed VMM
            // arena is auto-enabled (`auto_dynamic_lending`) even without
            // `ONNX_GENAI_CUDA_VMM`, so the realloc path must be opted into via
            // the #755 legacy knob or this phase would silently run VMM-backed.
            std::env::remove_var("ONNX_GENAI_CUDA_VMM");
            std::env::set_var("ONNX_GENAI_LEGACY_ALLOCATOR", "1");
        }
        if mode.seq_major {
            std::env::set_var("ONNX_GENAI_CUDA_KV_LAYOUT", "seq_major");
        } else {
            std::env::remove_var("ONNX_GENAI_CUDA_KV_LAYOUT");
        }
    }

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

    let result = engine.generate(request)?;
    let stats = engine.native_cuda_debug_stats();
    Ok(Run { result, stats })
}

fn report(label: &str, run: &Run) {
    let tok = run.result.token_ids.len();
    eprint!("[{label}] tokens={tok}");
    if let Some(s) = &run.stats {
        eprint!(
            " seq_major={} physical_committed_bytes={} growth_events={} kv_bytes_moved={} \
             captures={} invalidations={} growth_keeps={} replays={} max_len={} decline_reason={:?} \
             growth_decision={:?}",
            s.kv_layout_seq_major,
            s.kv_committed_bytes,
            s.kv_growth_events,
            s.kv_growth_d2d_copy_bytes,
            s.graph.captures,
            s.graph.invalidations,
            s.graph.growth_keeps,
            s.graph.replays,
            s.max_len,
            s.graph.decline_reason.as_deref(),
            s.graph.growth_decision.as_deref(),
        );
    } else {
        eprint!(" (no native CUDA stats)");
    }
    eprintln!();
}

fn measurement(label: String, run: Run) -> anyhow::Result<Measurement> {
    let stats = run
        .stats
        .context("native CUDA measurement did not expose debug stats")?;
    Ok(Measurement {
        label,
        token_ids: run.result.token_ids,
        text: run.result.text,
        seq_major: stats.kv_layout_seq_major,
        physical_committed_bytes: stats.kv_committed_bytes,
        growth_events: stats.kv_growth_events,
        kv_bytes_moved: stats.kv_growth_d2d_copy_bytes,
        captures: stats.graph.captures,
        replays: stats.graph.replays,
        invalidations: stats.graph.invalidations,
        growth_keeps: stats.graph.growth_keeps,
        decline_reason: stats.graph.decline_reason,
        growth_decision: stats.graph.growth_decision,
    })
}

fn run_child_config(
    seq_major: bool,
    vmm_arena: bool,
    capture: bool,
    min_bucket: &'static str,
) -> anyhow::Result<Measurement> {
    let exe = std::env::current_exe()?;
    let output = Command::new(exe)
        .arg("--exact")
        .arg("mobius_seq_major_growth_is_bit_identical_to_head_major")
        .arg("--ignored")
        .arg("--nocapture")
        .env(
            CHILD_LAYOUT_ENV,
            if seq_major { "seq-major" } else { "head-major" },
        )
        .env(CHILD_CAPTURE_ENV, if capture { "1" } else { "0" })
        .env(CHILD_VMM_ENV, if vmm_arena { "1" } else { "0" })
        .env(CHILD_MIN_BUCKET_ENV, min_bucket)
        .output()?;
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    anyhow::ensure!(
        output.status.success(),
        "mobius child configuration failed with {}:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout)
    );
    let stdout = String::from_utf8(output.stdout)?;
    let payload = stdout
        .lines()
        .find_map(|line| line.strip_prefix(MEASUREMENT_PREFIX))
        .context("mobius child emitted no structured measurement")?;
    Ok(serde_json::from_str(payload)?)
}

/// Interleave head-major and seq-major as separate processes, then assert their
/// generated bytes are identical.
fn parity_phase(
    vmm_arena: bool,
    capture: bool,
    min_bucket: &'static str,
) -> anyhow::Result<(Measurement, Measurement)> {
    let growth = if vmm_arena { "vmm-inplace" } else { "realloc" };
    let cap = if capture { "capture=ON" } else { "capture=OFF" };
    let bucket = if min_bucket == FORCE_GROWTH_MIN_BUCKET {
        "forced-growth"
    } else {
        "no-growth-control"
    };
    eprintln!("=== mobius seq-major parity, {growth}, {cap}, {bucket} ===");

    let head = run_child_config(false, vmm_arena, capture, min_bucket)?;
    let seq = run_child_config(true, vmm_arena, capture, min_bucket)?;
    eprintln!("{head:?}");
    eprintln!("{seq:?}");

    assert!(!head.token_ids.is_empty(), "head-major generated no tokens");
    assert_eq!(
        head.token_ids, seq.token_ids,
        "seq-major token stream diverged from head-major ({growth}, {cap}, {bucket})"
    );
    assert_eq!(
        head.text, seq.text,
        "seq-major text diverged from head-major ({growth}, {cap}, {bucket})"
    );
    assert!(
        seq.seq_major && !head.seq_major,
        "layout resolution wrong ({growth}, {cap}, {bucket}): head.seq_major={}, seq.seq_major={}",
        head.seq_major,
        seq.seq_major
    );
    Ok((head, seq))
}

#[test]
#[ignore = "requires the stock + kv_layout=1 mobius exports and a CUDA device"]
fn mobius_seq_major_growth_is_bit_identical_to_head_major() -> anyhow::Result<()> {
    if let Ok(layout) = std::env::var(CHILD_LAYOUT_ENV) {
        let seq_major = match layout.as_str() {
            "head-major" => false,
            "seq-major" => true,
            _ => anyhow::bail!("unknown {CHILD_LAYOUT_ENV} value {layout:?}"),
        };
        let dir = if seq_major {
            resolve_dir("MOBIUS_SEQMAJOR_SEQ_DIR", DEFAULT_SEQ_DIR)
        } else {
            resolve_dir("MOBIUS_SEQMAJOR_HEAD_DIR", DEFAULT_HEAD_DIR)
        };
        let Some(dir) = dir else {
            return Ok(());
        };
        if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
            eprintln!("skipping mobius seq-major growth parity: CUDA unavailable: {error}");
            return Ok(());
        }
        let capture = std::env::var(CHILD_CAPTURE_ENV)?.as_str() == "1";
        let vmm_arena = std::env::var(CHILD_VMM_ENV)?.as_str() == "1";
        let min_bucket = match std::env::var(CHILD_MIN_BUCKET_ENV)?.as_str() {
            FORCE_GROWTH_MIN_BUCKET => FORCE_GROWTH_MIN_BUCKET,
            NO_GROWTH_MIN_BUCKET => NO_GROWTH_MIN_BUCKET,
            value => anyhow::bail!("unknown {CHILD_MIN_BUCKET_ENV} value {value:?}"),
        };
        let mode = Mode {
            seq_major,
            capture,
            vmm_arena,
            min_bucket,
        };
        let label = format!(
            "{} {} capture={} min_bucket={min_bucket}",
            if seq_major { "seq-major" } else { "head-major" },
            if vmm_arena { "vmm-inplace" } else { "realloc" },
            if capture { "ON" } else { "OFF" }
        );
        let run = generate(&dir, mode)?;
        report(&label, &run);
        let measured = measurement(label, run)?;
        println!("{MEASUREMENT_PREFIX}{}", serde_json::to_string(&measured)?);
        return Ok(());
    }

    let (head_off, seq_off) = parity_phase(true, false, FORCE_GROWTH_MIN_BUCKET)?;
    let (head_on, seq_on) = parity_phase(true, true, FORCE_GROWTH_MIN_BUCKET)?;
    let (head_control, seq_control) = parity_phase(true, true, NO_GROWTH_MIN_BUCKET)?;
    let (head_realloc, seq_realloc) = parity_phase(false, false, FORCE_GROWTH_MIN_BUCKET)?;

    for measured in [&head_on, &seq_on] {
        assert!(
            measured.captures > 0,
            "capture=ON child must actually capture: {measured:?}"
        );
        assert!(measured.decline_reason.is_none());
        assert_eq!(measured.growth_events, 3);
    }
    for measured in [&head_off, &seq_off, &head_realloc, &seq_realloc] {
        assert_eq!(measured.captures, 0);
        let reason = measured
            .decline_reason
            .as_deref()
            .expect("capture=OFF must report the named declining predicate");
        assert!(
            reason.contains("predicate `ONNX_GENAI_CUDA_GRAPH`"),
            "{reason}"
        );
    }

    assert_eq!(seq_on.kv_bytes_moved, 0);
    assert_eq!(head_on.kv_bytes_moved, 688_576);
    assert_eq!(
        seq_on.physical_committed_bytes,
        head_on.physical_committed_bytes
    );
    assert_eq!(seq_realloc.kv_bytes_moved, head_realloc.kv_bytes_moved);

    // Head-major genuinely re-strides every head stripe on growth (it moves
    // 688,576 bytes above), so it MUST keep invalidating and re-capturing: one
    // extra invalidation and one extra capture per growth, and it never keeps a
    // captured graph across a growth.
    assert_eq!(head_control.growth_events, 0);
    assert_eq!(head_control.kv_bytes_moved, 0);
    assert_eq!(
        head_on.growth_keeps, 0,
        "head-major must never keep a captured graph across growth: growth={head_on:?}"
    );
    assert_eq!(
        head_on.invalidations - head_control.invalidations,
        head_on.growth_events,
        "each head-major bucket growth must invalidate the captured graph: \
         growth={head_on:?}, control={head_control:?}"
    );
    assert_eq!(
        head_on.captures - head_control.captures,
        head_on.growth_events,
        "each head-major growth invalidation must force one measured re-capture: \
         growth={head_on:?}, control={head_control:?}"
    );

    // Seq-major uses a fixed full-context stride: growth commits token stripes on
    // demand without moving KV bytes or changing any dependency the captured
    // graph baked in (device pointers, physical shapes, capacity-independent
    // batch-0 addressing), so the captured decode graph is KEPT. That means zero
    // growth-attributable invalidations, zero growth-attributable re-captures,
    // and exactly one recorded keep per growth — the headline of this change.
    assert_eq!(seq_control.growth_events, 0);
    assert_eq!(seq_control.kv_bytes_moved, 0);
    assert_eq!(
        seq_on.invalidations - seq_control.invalidations,
        0,
        "seq-major growth must NOT invalidate the captured graph: \
         growth={seq_on:?}, control={seq_control:?}"
    );
    assert_eq!(
        seq_on.captures - seq_control.captures,
        0,
        "seq-major growth must NOT force any re-capture: \
         growth={seq_on:?}, control={seq_control:?}"
    );
    assert_eq!(
        seq_on.growth_keeps - seq_control.growth_keeps,
        seq_on.growth_events,
        "each seq-major growth must record exactly one kept captured graph: \
         growth={seq_on:?}, control={seq_control:?}"
    );
    assert!(
        seq_on
            .growth_decision
            .as_deref()
            .is_some_and(|decision| decision.starts_with("kept")),
        "seq-major must report the named keep decision through #804 reporting: {seq_on:?}"
    );

    // Before/after headline (measured, process-isolated, on this box): head-major
    // holds at 4 invalidations / 4 captures because it moves bytes; seq-major
    // keeps its captured graph across every growth, so its forced-growth graph
    // accounting collapses onto the no-growth control — one initial capture,
    // replayed thereafter, and the single baseline (non-growth) invalidation.
    // Raw invalidations drop 4 -> 1 and captures drop 4 -> 1; the 3 eliminated
    // invalidations are exactly the growth-attributable ones now recorded as keeps.
    assert_eq!(
        (head_on.captures, head_on.replays, head_on.invalidations),
        (4, 39, 4),
        "head-major graph accounting regressed: {head_on:?}"
    );
    assert_eq!(
        (seq_on.captures, seq_on.replays, seq_on.invalidations),
        (
            seq_control.captures,
            seq_control.replays,
            seq_control.invalidations
        ),
        "seq-major forced-growth graph accounting must match its no-growth control \
         exactly (the captured graph is kept across every growth): \
         growth={seq_on:?}, control={seq_control:?}"
    );
    assert_eq!(
        (seq_on.captures, seq_on.replays, seq_on.invalidations),
        (1, 45, 1),
        "seq-major must keep its captured graph across all 3 growths: {seq_on:?}"
    );
    assert_eq!(
        seq_on.growth_keeps, 3,
        "seq-major must record one kept graph per growth (3 growths): {seq_on:?}"
    );
    Ok(())
}

/// Negative oracle for the conditional-keep decision — **the test that matters
/// most**: it fails if a captured graph is kept across a growth that genuinely
/// changed a dependency.
///
/// Head-major KV re-strides every head stripe on growth (the per-head stride is
/// `cache_capacity`), so a bucket growth moves KV bytes and relocates the exact
/// device addresses the captured decode graph baked into its recorded nodes.
/// Keeping the graph in that case would replay into stale addresses and corrupt
/// output. This test forces such a growth and asserts the graph is invalidated
/// and re-captured — never kept (`growth_keeps == 0`). If a future change made
/// the keep decision unsound for head-major, this test turns red instead of
/// silently corrupting tokens.
#[test]
#[ignore = "requires the stock mobius export and a CUDA device"]
fn head_major_growth_that_moves_addresses_must_invalidate_not_keep() -> anyhow::Result<()> {
    // Guard: this is a parent-only orchestration test. When invoked as a spawned
    // child worker (via the shared entrypoint), defer to that worker instead.
    if std::env::var(CHILD_LAYOUT_ENV).is_ok() {
        return mobius_seq_major_growth_is_bit_identical_to_head_major();
    }
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping head-major growth negative oracle: CUDA unavailable: {error}");
        return Ok(());
    }
    if resolve_dir("MOBIUS_SEQMAJOR_HEAD_DIR", DEFAULT_HEAD_DIR).is_none() {
        return Ok(());
    }

    // Head-major, in-place VMM, capture ON: force growth vs. a no-growth control.
    let grew = run_child_config(false, true, true, FORCE_GROWTH_MIN_BUCKET)?;
    let held = run_child_config(false, true, true, NO_GROWTH_MIN_BUCKET)?;
    eprintln!("grew:  {grew:?}");
    eprintln!("held:  {held:?}");

    assert!(
        grew.growth_events > 0,
        "head-major forced-growth did not actually grow: {grew:?}"
    );
    assert!(
        grew.kv_bytes_moved > 0,
        "head-major growth must move KV bytes (device addresses changed): {grew:?}"
    );
    assert_eq!(
        held.growth_events, 0,
        "no-growth control unexpectedly grew: {held:?}"
    );

    // The dependency changed, so the graph must NOT be kept.
    assert_eq!(
        grew.growth_keeps, 0,
        "a captured graph was KEPT across a growth that moved device addresses — \
         this replays into stale pointers and corrupts output: {grew:?}"
    );
    assert_eq!(
        grew.invalidations - held.invalidations,
        grew.growth_events,
        "each address-moving growth must invalidate the captured graph: \
         grew={grew:?}, held={held:?}"
    );
    assert_eq!(
        grew.captures - held.captures,
        grew.growth_events,
        "each address-moving growth must force one re-capture: grew={grew:?}, held={held:?}"
    );
    Ok(())
}
