//! Real-model KV **committed-physical-bytes floor** sweep, native CUDA decode,
//! head-major (BNSH) vs seq-major (BSNH), on qwen2.5-14b (the #827 follow-up).
//!
//! ## Why a new harness, and why this model
//!
//! `mobius_seqmajor_growth_parity_native_cuda` proved seq-major growth is
//! bit-identical and keeps the captured graph, but it is hard-wired to
//! qwen2.5-0.5b-q4_0-mobius, and #827 established (with arithmetic) that the
//! `layers x 2` floor **cannot separate from `layers x 2 x kv_heads` on that
//! model**: at head_dim 64 / fp16 with a 512-token cap, a single binding's whole
//! reservation is 128 KiB — sub-granule against the measured 2 MiB granule
//! (#776) — so both layouts pin one granule per binding (= the `layers x 2`
//! floor already). There is nothing to recover there.
//!
//! The `kv_heads x` separation only appears once a single head's **stripe**
//! crosses a granule. In head-major BNSH `[kv_heads, capacity, head_dim]` the
//! stripe for one head is `capacity * head_dim * sizeof(dtype)` contiguous
//! bytes, and the `kv_heads` stripes start `capacity * head_dim * sizeof(dtype)`
//! apart. When that stride reaches the 2 MiB commit granule, writing the live
//! prefix touches a **separate** granule in every head stripe, so head-major
//! commits `kv_heads` granules per binding; seq-major BSNH
//! `[capacity, kv_heads, head_dim]` writes the same prefix contiguously and
//! commits `ceil(prefix_bytes / granule)` — one granule while the prefix is
//! small. That is the `kv_heads x` = 8x floor this line has chased since #797.
//!
//! qwen14b geometry (from `genai_config.json`, confirmed): 48 layers,
//! `kv_heads = 8`, `head_dim = 128`, context 8192, fp16.
//!
//! ```text
//! head stripe bytes = capacity * head_dim * sizeof(dtype) = capacity * 256
//!   capacity 1024 -> 256 KiB  (sub-granule; 8 stripes share 1 granule)
//!   capacity 2048 -> 512 KiB  (4 stripes per granule)
//!   capacity 4096 ->   1 MiB  (2 stripes per granule)
//!   capacity 8192 ->   2 MiB  (exactly one granule per stripe)
//! ```
//!
//! So sweeping the reserved KV capacity across the 2 MiB threshold should walk
//! the head-major committed floor from `layers x 2` (96 granules, ~192 MiB) up
//! to `layers x 2 x kv_heads` (768 granules, ~1.5 GiB), while seq-major stays at
//! `layers x 2` throughout — an unambiguous ramp rather than the single marginal
//! point at full context. Reserved capacity is set with
//! `ONNX_GENAI_KV_MIN_BUCKET`; a short generation stays inside the bucket so the
//! measured number is the committed **floor** at that capacity, not a
//! growth artifact.
//!
//! ## Honest-reporting contract (#794 / #812 / #827)
//!
//! This test **reports** the swept committed-bytes comparison and asserts only
//! what is deterministic and layout-defining: token IDs byte-identical between
//! layouts at every capacity, and the layout actually resolved (seq vs head).
//! It does **not** bake in a predicted granule count, because whether the floor
//! separates is exactly the open question — if bucket sizing keeps stripes
//! sub-granule, or weight offload changes the KV allocator, the numbers say so.
//! qwen14b is 16.6 GB against a ~7.7 GB budget, so weight streaming is active
//! (#796 made offload capture-compatible); `decline_reason` is reported so an
//! operator can see whether capture ran.
//!
//! ## Measured result (this box, in-place VMM, capture OFF)
//!
//! Head-major and seq-major commit **byte-identical** physical KV at every swept
//! capacity — the floor does **not** separate on the engine's initial-reservation
//! path:
//!
//! ```text
//! capacity | head committed        | seq committed         | ratio | tokens
//!     1024 | 402,653,184 (192 gr)  | 402,653,184 (192 gr)  | 1.00x | identical
//!     2048 | 603,979,776 (288 gr)  | 603,979,776 (288 gr)  | 1.00x | identical
//!     4096 | 1,006,632,960 (480 gr)| 1,006,632,960 (480 gr)| 1.00x | identical
//!     8192 | 1,811,939,328 (864 gr)| 1,811,939,328 (864 gr)| 1.00x | identical
//! ```
//!
//! Why (from the counters, asserted below): `growth_events == 0` and
//! `committed_len == capacity` for **both** layouts.
//!
//! **Do not read that as an engine property.** This harness sets
//! `ONNX_GENAI_KV_MIN_BUCKET = capacity` at every swept point (see
//! [`run_child_config`]), and `onnx_genai_kv::kv_capacity_bucket` is
//! `len.next_power_of_two().max(min_bucket).min(hard_max)`. Pinning
//! `min_bucket == capacity` therefore *forces* `initial_bucket_len == capacity`,
//! which *forces* `committed_len == capacity`. The equality above is this knob's
//! own output, not a discovery about the engine — and it disables the on-demand
//! commit that is the only mechanism able to separate the layouts.
//!
//! Measured refutation: re-running this same child at the engine's **default**
//! `ONNX_GENAI_KV_MIN_BUCKET=256` yields, for seq-major, `committed_len=256`
//! with `max_len=8192` — a live-prefix commit far short of the full bucket. The
//! engine does *not* commit the whole bucket eagerly.
//!
//! What this sweep therefore establishes is narrower but still real: at equal
//! committed length the two layouts commit byte-identical physical bytes, and
//! seq-major's fixed full-context stride is confirmed active (`max_len == 8192`
//! at every capacity vs head-major's `max_len == capacity`) with byte-identical
//! token streams, so the seq-major kernel and layout resolution are correct.
//!
//! Separation was never given a chance here, because it needs *both* of:
//!   1. head-major capacity large enough that its per-head stripe reaches a
//!      granule (capacity ~8192 => stripe = 2 MiB), **and**
//!   2. the seq-major committed dense prefix left free to stay small — i.e.
//!      `ONNX_GENAI_KV_MIN_BUCKET` *not* pinned to the capacity.
//!
//! At the swept points below, condition 2 is violated by construction. At the
//! default bucket 256 both layouts commit 192 granules because condition 1 is
//! then violated instead (a 256-token head stripe is 64 KiB, sub-granule). The
//! regime where the 8x can appear — small bucket *and* long live prefix — is
//! measured separately; a fixed stride makes *growth* free (#797's 0 bytes moved)
//! and keeps the captured graph across growth (#811/#812).
//!
//! ## Layout levers (two, both required for a real seq-major run) — as #801/#805
//!
//! * engine growth geometry: `ONNX_GENAI_CUDA_KV_LAYOUT=seq_major`;
//! * CUDA GQA kernel stride: the ONNX `kv_layout=1` node attribute, patched into
//!   a **copy** of the export (never the original) with
//!   `scripts/gen_inference_metadata.py`-adjacent tooling — see the module doc of
//!   `mobius_seqmajor_growth_parity_native_cuda` for the one-liner.
//!
//! Run:
//!
//! ```bash
//! QWEN14B_HEAD_DIR=/models/qwen14b-zp \
//! QWEN14B_SEQ_DIR=/models/qwen14b-zp-seqmajor \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend \
//!   --test qwen14b_kv_floor_sweep_native_cuda \
//!   -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "native-cuda")]

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Context;
use onnx_genai_engine::{
    CudaKvDebugStats, Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, GenerateResult,
    NativeDecodeDevice,
};
use serde::{Deserialize, Serialize};

const PROMPT: &str = "The capital of France is";
/// Short: stay well inside every swept bucket so no growth fires and the
/// committed number is the reserved-capacity floor, not a growth transient.
/// Also keeps the 16.6 GB model's slow decode bounded.
const MAX_NEW_TOKENS: usize = 16;

/// Reserved KV capacities (tokens) that bracket the 2 MiB-per-stripe granule
/// threshold for qwen14b (`head_dim = 128`, fp16): the head-major stripe is
/// `capacity * 256` bytes, crossing 2 MiB at capacity 8192.
const CAPACITIES: [usize; 4] = [1024, 2048, 4096, 8192];

/// The measured commit granule on this platform (#776); used only to render a
/// human-readable granule count in the report, never asserted.
const GRANULE_BYTES: usize = 2 * 1024 * 1024;

const DEFAULT_HEAD_DIR: &str = r"C:\Users\justinchu\dev\models\qwen14b-zp";
const DEFAULT_SEQ_DIR: &str = r"C:\Users\justinchu\dev\models\qwen14b-zp-seqmajor";

const CHILD_LAYOUT_ENV: &str = "QWEN14B_CHILD_LAYOUT";
const CHILD_MIN_BUCKET_ENV: &str = "QWEN14B_CHILD_MIN_BUCKET";
const MEASUREMENT_PREFIX: &str = "QWEN14B_MEASUREMENT=";

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
            "skipping qwen14b kv floor sweep: {} ({}) missing {}",
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

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Measurement {
    label: String,
    capacity: usize,
    seq_major: bool,
    token_ids: Vec<u32>,
    text: String,
    physical_committed_bytes: usize,
    kv_committed_len: usize,
    max_len: usize,
    per_binding_bytes: Vec<usize>,
    growth_events: u64,
    kv_bytes_moved: u64,
    captures: u64,
    replays: u64,
    invalidations: u64,
    growth_keeps: u64,
    decline_reason: Option<String>,
    growth_decision: Option<String>,
}

/// Generate greedily on `dir` with reserved capacity `min_bucket`, in-place VMM
/// arena, capture OFF. In-place VMM is required so `kv_committed_bytes` reports
/// **granule-committed physical** bytes (the VMM allocator rounds to committed
/// granules); the legacy realloc allocator would report nominal content bytes,
/// which is exactly what the physical-floor measurement must not do.
fn generate(dir: &Path, seq_major: bool, min_bucket: usize) -> anyhow::Result<Run> {
    // SAFETY: single-threaded (`--test-threads=1`); env is set before the engine
    // is constructed, and the process-frozen runtime config (#804/#807) is read
    // exactly once at load — which is why each configuration runs in its own
    // child process.
    unsafe {
        std::env::set_var("ONNX_GENAI_KV_MIN_BUCKET", min_bucket.to_string());
        // Capture OFF: this measurement is about committed KV granules, and a
        // 16.6 GB weight-streamed model would decline capture anyway; keeping it
        // off avoids a pointless capture attempt and is faster.
        std::env::set_var("ONNX_GENAI_CUDA_GRAPH", "0");
        // In-place managed VMM arena (the #798 default) — granule-accurate
        // committed accounting.
        std::env::set_var("ONNX_GENAI_CUDA_VMM", "1");
        std::env::remove_var("ONNX_GENAI_LEGACY_ALLOCATOR");
        if seq_major {
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

fn measurement(label: String, capacity: usize, run: Run) -> anyhow::Result<Measurement> {
    let stats = run
        .stats
        .context("native CUDA measurement did not expose debug stats")?;
    Ok(Measurement {
        label,
        capacity,
        seq_major: stats.kv_layout_seq_major,
        token_ids: run.result.token_ids,
        text: run.result.text,
        physical_committed_bytes: stats.kv_committed_bytes,
        kv_committed_len: stats.kv_committed_len,
        max_len: stats.max_len,
        per_binding_bytes: stats.kv_physical_bytes_by_binding,
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

fn granules(bytes: usize) -> f64 {
    bytes as f64 / GRANULE_BYTES as f64
}

fn report(m: &Measurement) {
    eprintln!(
        "  [{}] cap={} committed={} B (~{:.0} granules, {:.3} GiB) committed_len={} max_len={} \
         growth_events={} kv_bytes_moved={} captures={} replays={} invalidations={} \
         growth_keeps={} decline_reason={:?}",
        m.label,
        m.capacity,
        m.physical_committed_bytes,
        granules(m.physical_committed_bytes),
        m.physical_committed_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        m.kv_committed_len,
        m.max_len,
        m.growth_events,
        m.kv_bytes_moved,
        m.captures,
        m.replays,
        m.invalidations,
        m.growth_keeps,
        m.decline_reason,
    );
}

/// Run one (layout, capacity) configuration in its own process (process-frozen
/// config #804/#807), returning its structured measurement.
fn run_child_config(seq_major: bool, capacity: usize) -> anyhow::Result<Measurement> {
    let exe = std::env::current_exe()?;
    let output = Command::new(exe)
        .arg("--exact")
        .arg("qwen14b_kv_committed_floor_separates_head_from_seq_major")
        .arg("--ignored")
        .arg("--nocapture")
        .env(
            CHILD_LAYOUT_ENV,
            if seq_major { "seq-major" } else { "head-major" },
        )
        .env(CHILD_MIN_BUCKET_ENV, capacity.to_string())
        .output()?;
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    anyhow::ensure!(
        output.status.success(),
        "qwen14b child (seq_major={seq_major}, capacity={capacity}) failed with {}:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout)
    );
    let stdout = String::from_utf8(output.stdout)?;
    let payload = stdout
        .lines()
        .find_map(|line| line.strip_prefix(MEASUREMENT_PREFIX))
        .context("qwen14b child emitted no structured measurement")?;
    Ok(serde_json::from_str(payload)?)
}

/// Sweep the reserved KV capacity across the 2 MiB-per-stripe granule threshold
/// and report head-major vs seq-major committed physical bytes at each point,
/// interleaved in the same test, with byte-identical token streams asserted per
/// capacity. Reporting is the headline; the only hard assertions are the
/// deterministic, layout-defining ones (token parity, resolved layout).
#[test]
#[ignore = "requires the qwen14b head + kv_layout=1 exports and a CUDA device"]
fn qwen14b_kv_committed_floor_separates_head_from_seq_major() -> anyhow::Result<()> {
    // Child worker: run exactly one (layout, capacity) and emit its measurement.
    if let Ok(layout) = std::env::var(CHILD_LAYOUT_ENV) {
        let seq_major = match layout.as_str() {
            "head-major" => false,
            "seq-major" => true,
            other => anyhow::bail!("unknown {CHILD_LAYOUT_ENV} value {other:?}"),
        };
        let dir = if seq_major {
            resolve_dir("QWEN14B_SEQ_DIR", DEFAULT_SEQ_DIR)
        } else {
            resolve_dir("QWEN14B_HEAD_DIR", DEFAULT_HEAD_DIR)
        };
        let Some(dir) = dir else {
            return Ok(());
        };
        if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
            eprintln!("skipping qwen14b kv floor sweep: CUDA unavailable: {error}");
            return Ok(());
        }
        let capacity: usize = std::env::var(CHILD_MIN_BUCKET_ENV)?.parse()?;
        let label = format!(
            "{} cap={capacity}",
            if seq_major { "seq-major" } else { "head-major" }
        );
        let run = generate(&dir, seq_major, capacity)?;
        let measured = measurement(label, capacity, run)?;
        report(&measured);
        println!("{MEASUREMENT_PREFIX}{}", serde_json::to_string(&measured)?);
        return Ok(());
    }

    // Parent orchestrator: gate on model + CUDA availability, then sweep.
    if resolve_dir("QWEN14B_HEAD_DIR", DEFAULT_HEAD_DIR).is_none()
        || resolve_dir("QWEN14B_SEQ_DIR", DEFAULT_SEQ_DIR).is_none()
    {
        return Ok(());
    }
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping qwen14b kv floor sweep: CUDA unavailable: {error}");
        return Ok(());
    }

    eprintln!(
        "=== qwen14b KV committed-physical-bytes floor sweep (head-major vs seq-major) ===\n\
         geometry: 48 layers, kv_heads=8, head_dim=128, fp16, context 8192; \
         96 KV bindings; granule {GRANULE_BYTES} B\n\
         prompt={PROMPT:?} max_new_tokens={MAX_NEW_TOKENS}"
    );

    let mut rows: Vec<(usize, Measurement, Measurement)> = Vec::new();
    for capacity in CAPACITIES {
        eprintln!(
            "--- capacity {capacity} (head stripe {} B) ---",
            capacity * 256
        );
        let head = run_child_config(false, capacity)?;
        let seq = run_child_config(true, capacity)?;

        assert!(
            !head.token_ids.is_empty(),
            "head-major generated no tokens at capacity {capacity}: {head:?}"
        );
        assert_eq!(
            head.token_ids, seq.token_ids,
            "seq-major token stream diverged from head-major at capacity {capacity}"
        );
        assert_eq!(
            head.text, seq.text,
            "seq-major text diverged from head-major at capacity {capacity}"
        );
        assert!(
            seq.seq_major && !head.seq_major,
            "layout resolution wrong at capacity {capacity}: head.seq_major={}, seq.seq_major={}",
            head.seq_major,
            seq.seq_major
        );

        // Mechanism guardrails (deterministic given `min_bucket == capacity` and
        // a short generation). These document *why* the committed bytes are what
        // they are, and turn the measured finding into a regression guard.
        //
        // 1. No growth fired: the whole bucket is reserved up front, so what we
        //    measure is the pure reservation floor at `capacity`, not a growth
        //    transient.
        for m in [&head, &seq] {
            assert_eq!(
                m.growth_events, 0,
                "expected no KV growth at min_bucket == capacity {capacity}: {m:?}"
            );
        }
        // 2. Both layouts report `committed_len == capacity` — because this
        //    harness pinned `ONNX_GENAI_KV_MIN_BUCKET = capacity`, which forces
        //    `initial_bucket_len == capacity` through
        //    `kv_capacity_bucket(len, hard_max) =
        //     len.next_power_of_two().max(min_bucket).min(hard_max)`.
        //    These two asserts therefore pin the HARNESS's own configuration, not
        //    an engine property: they exist so that a future change to bucketing
        //    that silently breaks this precondition turns the sweep red instead of
        //    quietly comparing two different committed lengths. At the default
        //    bucket (256) seq-major reports `committed_len=256` with
        //    `max_len=8192`, i.e. the engine does NOT commit the bucket eagerly.
        //    Seq-major resolves the fixed full-context stride (`max_len ==
        //    hard-max 8192`) while head-major's stride is its bucket (`max_len ==
        //    capacity`) — the stride differs, the committed total does not.
        assert_eq!(
            head.kv_committed_len, capacity,
            "harness precondition: min_bucket == capacity must pin head-major's \
             committed length to the capacity: {head:?}"
        );
        assert_eq!(
            seq.kv_committed_len, capacity,
            "harness precondition: min_bucket == capacity must pin seq-major's \
             committed length to the capacity: {seq:?}"
        );
        assert_eq!(
            head.max_len, capacity,
            "head-major stride is its bucket: {head:?}"
        );
        assert!(
            seq.max_len >= capacity,
            "seq-major uses the fixed full-context stride (>= capacity): {seq:?}"
        );

        // 3. The headline, asserted so a future change cannot silently move it:
        //    committed physical KV bytes are IDENTICAL head-major vs seq-major at
        //    this capacity. The kv_heads x floor separation does NOT appear on the
        //    engine's eager full-bucket reservation path — it is a driver-level /
        //    growth-path property (see the module doc and MEMORY_ARCHITECTURE.md).
        //    If this assertion ever fails because seq-major commits LESS, that is
        //    the long-sought win: update the floor table and this guard rather
        //    than treating it as a regression.
        assert_eq!(
            head.physical_committed_bytes, seq.physical_committed_bytes,
            "committed physical KV bytes unexpectedly diverged at capacity {capacity} \
             (head={} seq={}). If seq-major now commits less, this is the kv_heads x \
             floor win landing end-to-end — update MEMORY_ARCHITECTURE.md's floor \
             progression table and this guard.",
            head.physical_committed_bytes, seq.physical_committed_bytes,
        );

        rows.push((capacity, head, seq));
    }

    eprintln!("\n=== SWEEP SUMMARY: committed physical KV bytes (in-place VMM) ===");
    eprintln!(
        "{:>8} | {:>10} {:>8} {:>9} | {:>10} {:>8} {:>9} | {:>6} | tokens match",
        "capacity", "head B", "head~gr", "head GiB", "seq B", "seq~gr", "seq GiB", "ratio",
    );
    for (capacity, head, seq) in &rows {
        let ratio = if seq.physical_committed_bytes == 0 {
            f64::NAN
        } else {
            head.physical_committed_bytes as f64 / seq.physical_committed_bytes as f64
        };
        eprintln!(
            "{:>8} | {:>10} {:>8.0} {:>9.3} | {:>10} {:>8.0} {:>9.3} | {:>5.2}x | {}",
            capacity,
            head.physical_committed_bytes,
            granules(head.physical_committed_bytes),
            head.physical_committed_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            seq.physical_committed_bytes,
            granules(seq.physical_committed_bytes),
            seq.physical_committed_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            ratio,
            head.token_ids == seq.token_ids,
        );
    }
    eprintln!(
        "\nMeasured finding (this box, in-place VMM, capture OFF): head-major and seq-major \
         commit BYTE-IDENTICAL physical KV at every capacity (ratio 1.00x), ramping together \
         192 -> 288 -> 480 -> 864 granules as the reserved bucket grows, with byte-identical \
         token streams and seq-major's fixed full-context stride confirmed active (max_len == \
         8192 at every capacity vs head-major's max_len == capacity). \
         CAVEAT ON SCOPE: this sweep pins ONNX_GENAI_KV_MIN_BUCKET = capacity, which FORCES \
         committed_len == capacity via kv_capacity_bucket(len, hard_max) = \
         len.next_power_of_two().max(min_bucket).min(hard_max). So the equal committed lengths \
         are this harness's own configuration, NOT evidence that the engine commits eagerly -- \
         at the default bucket 256 seq-major reports committed_len=256 with max_len=8192, i.e. a \
         live-prefix commit well short of the bucket. What is established here is that AT EQUAL \
         COMMITTED LENGTH the layouts commit identical physical bytes. Separation needs both (a) \
         a head-major capacity whose per-head stripe reaches a 2 MiB granule (~8192) and (b) a \
         small committed dense prefix -- i.e. min_bucket NOT pinned to capacity. Condition (b) is \
         violated by construction here; at bucket 256 condition (a) is violated instead. The \
         768-vs-96-granule (8x) regime is measured separately."
    );

    // Deterministic, layout-defining evidence (token parity, resolved layout) and
    // the harness preconditions (growth_events, committed_len == the pinned
    // min_bucket, head == seq committed) are asserted above per capacity. Per the
    // #794/#812/#827 honest-reporting contract, the equality is the reported
    // result -- scoped to equal committed length, which is the condition this
    // harness pins rather than discovers.
    Ok(())
}
