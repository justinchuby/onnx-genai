//! Real-model KV **committed-physical-bytes** floor experiment, native CUDA
//! decode, head-major (BNSH) vs seq-major (BSNH), on qwen2.5-14b. This is the
//! decisive follow-up to #834 / the #797 line: does the predicted `kv_heads×`
//! (8×) floor separation between head-major and seq-major appear once the
//! measurement stops disabling the very on-demand-commit mechanism that could
//! separate the layouts?
//!
//! ## Why #834's harness could not answer the question
//!
//! `qwen14b_kv_floor_sweep_native_cuda` set `ONNX_GENAI_KV_MIN_BUCKET =
//! capacity` at every swept point. `kv_capacity_bucket(len, hard_max)` is
//! `len.next_power_of_two().max(min_bucket).min(hard_max)`, so pinning
//! `min_bucket == capacity` **forces** the initial bucket to the full capacity,
//! which forces `committed_len == capacity` and `growth_events == 0`. That
//! harness then "discovered" `committed_len == capacity` and attributed the
//! flat commit to the *engine* — but it was the knob, not the engine. It also
//! disabled the on-demand growth path that is the only place the two layouts
//! commit differently. So #834's "byte-identical, engine commits the full
//! bucket eagerly" is a measurement artifact.
//!
//! ## What this harness does instead
//!
//! Keep `ONNX_GENAI_KV_MIN_BUCKET` at its default (256) so **on-demand commit
//! is active** (`growth_events >= 1`), and compare the two layouts at the same
//! live prefix. Two phases:
//!
//! * **Phase A — head-vs-seq equality (both layouts, same prefix).** Drive the
//!   prefix past `min_bucket` (256) so growth fires, then assert byte-identical
//!   token streams *and* byte-identical committed physical KV. This is the
//!   direct refutation of #834: the engine does NOT commit the full bucket
//!   eagerly (growth fired), yet the layouts still commit identically.
//! * **Phase B — head-major dense-floor vs the `kv_heads×` fixed-stride floor.**
//!   At each reachable bucket, assert head-major's committed KV is the *dense
//!   packed bucket*, strictly BELOW the fixed-stride 8× scatter floor
//!   (`96 × kv_heads × ceil(bucket × head_dim × elem / granule)` granules). The
//!   2 MiB-per-stripe crossover (`capacity × head_dim × sizeof(fp16) =
//!   capacity × 256` reaches 2 MiB at capacity 8192) is the point where the dense
//!   bucket and the scatter floor numerically coincide — but that bucket is
//!   **physically unreachable on this box** (constraint 3 below), so the honest
//!   deliverable is: at every bucket the hardware permits, head-major stays on
//!   its dense floor, well below the 8× scatter the engine never instantiates.
//!
//! ### The three hard constraints this design works around (all reported, not hidden)
//!
//! 1. **Seq-major multi-token prefill above 512 is unconverted.**
//!    `selected_backend_for_shape` only auto-selects the fused/flash prefill path
//!    when `valid_sequence_length <= 512`; a longer *prompt* falls to the
//!    Phase2a unfused-decode-prep path, which `require_converted_path_support`
//!    rejects for BSNH (hard `KernelFailed`). So seq-major cannot be driven to a
//!    long prefix by a long **prompt**. Single-token **decode** always takes the
//!    converted `FusedDecodePrep` + `Fp16DecodeRead` BSNH path, so the seq-major
//!    prefix is nudged just past 512 by a few generated tokens (Phase A only).
//! 2. **Decode on this 16.6 GB model is weight-offload bound (~0.25 tok/s).**
//!    The model is 16.6 GB against a ~7.7 GB budget, so every decode step streams
//!    weights; generating the thousands of tokens needed to reach an 8k prefix by
//!    decode is infeasible here. So neither layout can reach a long prefix by
//!    decode.
//! 3. **Head-major long prefill OOMs the attention workspace.** This box is an
//!    RTX 4060 Laptop with only 8 GiB of VRAM, so the ~7.7 GB budget is
//!    essentially the whole card and there is no headroom to raise it. The
//!    16.6 GB weights lease ~7.5–7.7 GB, leaving almost nothing for the
//!    prefill attention workspace, which grows with prompt length. Measured
//!    here: head-major prefill succeeds up to ~600 tokens (bucket 1024) but a
//!    ~2000-token prompt (bucket 2048) already fails with
//!    `cuMemMap: growing physical handle pool lease failed ... 0 bytes became
//!    available` (Workspace), and ~3900/7800-token prompts fail the same way.
//!    So head-major cannot reach bucket ≥ 2048 by prefill either.
//!
//! **Consequence (itself part of the answer):** the long-prefix / bucket-8192
//! regime the 8× separation would require is **physically unreachable on this
//! box by any path** — seq-major prefill is unconverted, head-major prefill
//! OOMs, and decode is offload-bound. The maximum bucket reachable by *both*
//! layouts is ~1024. Phase A therefore compares the layouts across every
//! reachable shared bucket (512, 1024); Phase B shows head-major's dense floor
//! stays below the 8× scatter at those buckets; and the crossover is left as a
//! symbolic result (proved by `kv_commit.rs`'s unit tests), not a hardware run.
//!
//! Each configuration runs in its OWN child process (runtime config is
//! process-frozen on first read, #804/#807), sets `ONNX_GENAI_CUDA_VMM=1` so the
//! reported `kv_committed_bytes` is the VMM allocator's **granule-rounded
//! physical** number (never nominal content bytes).
//!
//! ## The mechanism this measures (from the commit path, not a knob)
//!
//! The 8× floor of `kv_commit::live_prefix_committed_bytes` is a property of a
//! **fixed full-context stride** head-major layout, where each head's stripe is
//! `max_len × head_dim × elem` apart and only a short prefix is live, so the
//! prefix scatters into one granule per head. The engine's head-major path does
//! **not** use a fixed stride: `vmm_growth_requests` commits a single flat range
//! `0..(bucket × kv_heads × head_dim × elem)` whose per-head stride is the
//! *current bucket* (`new_shape[2] = new_capacity`), so the live bytes stay
//! densely packed from offset 0 and head-major re-strides + re-captures on
//! growth (`apply_vmm_growth`, `invalidate_graph`) to keep that packing.
//! Seq-major (`seq_major_kv_commit_requests`) commits the same dense prefix
//! `0..(committed_len × kv_heads × head_dim × elem)` on a fixed stride but keeps
//! its captured graph (`growth_keeps`). Because `kv_capacity_bucket(required)`
//! is identical for both, the two dense-from-zero runs commit the **same**
//! granules — so the layouts stay byte-identical in committed KV. This harness
//! asserts that through counters, so a regression that silently changed
//! head-major to a fixed stride (which *would* separate the floor) is caught, and
//! — if seq-major ever commits strictly less — the byte-equality guard fails
//! loudly with instructions to record the win.
//!
//! Run (release strongly recommended):
//!
//! ```bash
//! QWEN14B_HEAD_DIR=/models/qwen14b-zp \
//! QWEN14B_SEQ_DIR=/models/qwen14b-zp-seqmajor \
//! CUDA_VISIBLE_DEVICES=0 cargo test --release -p onnx-genai-engine \
//!   --features cuda,native-backend \
//!   --test qwen14b_kv_longprefix_native_cuda \
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

/// A distinctive base sentence repeated to synthesize prompts. Repetition keeps
/// prompts deterministic and layout-independent (the only thing under test is
/// *where* identical KV bytes land, not what they are).
const PROMPT_UNIT: &str = "The quick brown fox jumps over the lazy dog beside the \
     quiet river bank while thirteen weary travellers count the distant stars. ";

/// The engine-default reserved KV bucket floor. Held FIXED at the default — the
/// single most important difference from #834, which pinned it to the capacity
/// and thereby forced `committed_len == capacity` and `growth_events == 0`. At
/// 256 the bucket starts tiny and grows on demand as the prefix lengthens.
const MIN_BUCKET: usize = 256;

/// Phase A: (prompt_chars, new_tokens) points where BOTH layouts can reach the
/// same live prefix. The prompt stays under ~512 tokens (~5.6 chars/token on
/// this tokenizer) so seq-major's prefill is on the converted fused path; a few
/// generated tokens then nudge the prefix past `min_bucket` (and, for the second
/// point, just past 512 into the next bucket). Both layouts run identical
/// prompt+greedy generation, so the live prefix and token stream match.
const EQ_POINTS: [(usize, usize); 2] = [
    (2_200, 8),  // ~392-token prompt + 8 ⇒ prefix ~400 ⇒ bucket 512
    (2_600, 64), // ~464-token prompt + 64 ⇒ prefix ~528 ⇒ bucket 1024
];

/// Phase B: head-major-only buckets at which we assert the dense-floor-vs-8×
/// property. These are the buckets *reachable on this box* (see constraint 3):
/// bucket 2048+ OOMs the prefill workspace, so the ramp cannot extend to the
/// 8192 crossover here. Phase B reuses Phase A's head-major rows rather than
/// re-running — the reachable head-major buckets are exactly Phase A's (512,
/// 1024), and re-driving them by prefill would only re-hit the same OOM ceiling.
///
/// The empirically-measured head-major prefill OOM ceiling on this box, recorded
/// for the report (not asserted — shared-GPU residency varies run to run):
/// ~600-token prompt (bucket 1024) succeeds; ~2000-token prompt (bucket 2048)
/// fails with a Workspace `cuMemMap` lease error.
const HEAD_PREFILL_OOM_NOTE: &str = "head-major prefill OOM ceiling (measured): bucket 1024 (~600-token prompt) OK; \
     bucket 2048 (~2000-token prompt) and above fail with Workspace cuMemMap lease \
     exhaustion — 8 GiB card, 16.6 GB weights leave no workspace headroom";

/// The measured commit granule on this platform (#776); used only to render a
/// human-readable granule count in the report, never asserted.
const GRANULE_BYTES: usize = 2 * 1024 * 1024;

const DEFAULT_HEAD_DIR: &str = r"C:\Users\justinchu\dev\models\qwen14b-zp";
const DEFAULT_SEQ_DIR: &str = r"C:\Users\justinchu\dev\models\qwen14b-zp-seqmajor";

const CHILD_LAYOUT_ENV: &str = "QWEN14B_CHILD_LAYOUT";
const CHILD_PROMPT_CHARS_ENV: &str = "QWEN14B_CHILD_PROMPT_CHARS";
const CHILD_NEW_TOKENS_ENV: &str = "QWEN14B_CHILD_NEW_TOKENS";
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
            "skipping qwen14b kv long-prefix experiment: {} ({}) missing {}",
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

fn build_prompt(prompt_chars: usize) -> String {
    let mut prompt = String::with_capacity(prompt_chars + PROMPT_UNIT.len());
    while prompt.len() < prompt_chars {
        prompt.push_str(PROMPT_UNIT);
    }
    prompt
}

struct Run {
    result: GenerateResult,
    stats: Option<CudaKvDebugStats>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Measurement {
    label: String,
    prompt_chars: usize,
    new_tokens: usize,
    seq_major: bool,
    token_ids: Vec<u32>,
    generated_tokens: usize,
    /// Granule-rounded committed physical KV bytes (VMM allocator), summed over
    /// the KV bindings only (the mask island is excluded, see `debug_stats`).
    physical_committed_bytes: usize,
    /// Live prefix length in tokens (prompt + generated) — the residency driver.
    logical_len: usize,
    /// Committed KV bucket high-water mark in tokens, maintained on the seq-major
    /// path (`next_pow2` of the live prefix). Not maintained on the head-major
    /// path — read `max_len` there.
    kv_committed_len: usize,
    /// Reported physical stride axis-2. Seq-major fixed stride pins this at the
    /// hard max (8192); head-major tracks its growing bucket.
    max_len: usize,
    per_binding_bytes_first: usize,
    kv_binding_count: usize,
    growth_events: u64,
    kv_bytes_moved: u64,
    captures: u64,
    replays: u64,
    invalidations: u64,
    growth_keeps: u64,
    decline_reason: Option<String>,
    growth_decision: Option<String>,
}

/// Head-major's committed bucket lives in `max_len`; seq-major's in
/// `kv_committed_len`. Returns the layout-correct committed bucket.
fn committed_bucket(m: &Measurement) -> usize {
    if m.seq_major {
        m.kv_committed_len
    } else {
        m.max_len
    }
}

fn generate(
    dir: &Path,
    seq_major: bool,
    prompt_chars: usize,
    new_tokens: usize,
) -> anyhow::Result<Run> {
    // SAFETY: single-threaded (`--test-threads=1`); env is set before the engine
    // is constructed, and the process-frozen runtime config (#804/#807) is read
    // exactly once at load — which is why each configuration runs in its own
    // child process.
    unsafe {
        // The FIX vs #834: default bucket, NOT the capacity, so on-demand growth
        // from 256 up toward the prefix length is exercised.
        std::env::set_var("ONNX_GENAI_KV_MIN_BUCKET", MIN_BUCKET.to_string());
        std::env::set_var("ONNX_GENAI_CUDA_GRAPH", "0");
        // In-place managed VMM arena (#798 default) — granule-accurate committed
        // accounting; the legacy allocator reports nominal bytes and must not be
        // used for a physical-floor measurement.
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
    let mut request = GenerateRequest::new(build_prompt(prompt_chars));
    request.options.max_new_tokens = new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;

    let result = engine.generate(request)?;
    let stats = engine.native_cuda_debug_stats();
    Ok(Run { result, stats })
}

fn measurement(
    label: String,
    prompt_chars: usize,
    new_tokens: usize,
    run: Run,
) -> anyhow::Result<Measurement> {
    let stats = run
        .stats
        .context("native CUDA measurement did not expose debug stats")?;
    let per_binding_bytes_first = stats
        .kv_physical_bytes_by_binding
        .first()
        .copied()
        .unwrap_or(0);
    Ok(Measurement {
        label,
        prompt_chars,
        new_tokens,
        seq_major: stats.kv_layout_seq_major,
        generated_tokens: run.result.token_ids.len(),
        token_ids: run.result.token_ids,
        physical_committed_bytes: stats.kv_committed_bytes,
        logical_len: stats.logical_len,
        kv_committed_len: stats.kv_committed_len,
        max_len: stats.max_len,
        per_binding_bytes_first,
        kv_binding_count: stats.kv_physical_bytes_by_binding.len(),
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
        "  [{}] prompt_chars={} new_tokens={} generated={} committed={} B (~{:.0} gr, {:.3} GiB) \
         logical_len={} committed_bucket={} committed_len={} max_len={} growth_events={} \
         kv_bytes_moved={} invalidations={} growth_keeps={} decline_reason={:?}",
        m.label,
        m.prompt_chars,
        m.new_tokens,
        m.generated_tokens,
        m.physical_committed_bytes,
        granules(m.physical_committed_bytes),
        m.physical_committed_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        m.logical_len,
        committed_bucket(m),
        m.kv_committed_len,
        m.max_len,
        m.growth_events,
        m.kv_bytes_moved,
        m.invalidations,
        m.growth_keeps,
        m.decline_reason,
    );
}

/// Run one (layout, prompt_chars, new_tokens) configuration in its own process
/// (process-frozen config #804/#807), returning its structured measurement.
fn run_child_config(
    seq_major: bool,
    prompt_chars: usize,
    new_tokens: usize,
) -> anyhow::Result<Measurement> {
    let exe = std::env::current_exe()?;
    let output = Command::new(exe)
        .arg("--exact")
        .arg("qwen14b_kv_longprefix_head_vs_seq_major")
        .arg("--ignored")
        .arg("--nocapture")
        .env(
            CHILD_LAYOUT_ENV,
            if seq_major { "seq-major" } else { "head-major" },
        )
        .env(CHILD_PROMPT_CHARS_ENV, prompt_chars.to_string())
        .env(CHILD_NEW_TOKENS_ENV, new_tokens.to_string())
        .output()?;
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    anyhow::ensure!(
        output.status.success(),
        "qwen14b child (seq_major={seq_major}, prompt_chars={prompt_chars}, \
         new_tokens={new_tokens}) failed with {}:\n{}",
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

#[test]
#[ignore = "requires the qwen14b head + kv_layout=1 exports and a CUDA device"]
fn qwen14b_kv_longprefix_head_vs_seq_major() -> anyhow::Result<()> {
    // Child worker: run exactly one configuration and emit its measurement.
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
            eprintln!("skipping qwen14b kv long-prefix experiment: CUDA unavailable: {error}");
            return Ok(());
        }
        let prompt_chars: usize = std::env::var(CHILD_PROMPT_CHARS_ENV)?.parse()?;
        let new_tokens: usize = std::env::var(CHILD_NEW_TOKENS_ENV)?.parse()?;
        let label = format!(
            "{} chars={prompt_chars} new={new_tokens}",
            if seq_major { "seq-major" } else { "head-major" }
        );
        let run = generate(&dir, seq_major, prompt_chars, new_tokens)?;
        let measured = measurement(label, prompt_chars, new_tokens, run)?;
        report(&measured);
        println!("{MEASUREMENT_PREFIX}{}", serde_json::to_string(&measured)?);
        return Ok(());
    }

    // Parent orchestrator: gate on model + CUDA availability, then run.
    if resolve_dir("QWEN14B_HEAD_DIR", DEFAULT_HEAD_DIR).is_none()
        || resolve_dir("QWEN14B_SEQ_DIR", DEFAULT_SEQ_DIR).is_none()
    {
        return Ok(());
    }
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping qwen14b kv long-prefix experiment: CUDA unavailable: {error}");
        return Ok(());
    }

    eprintln!(
        "=== qwen14b KV committed-physical-bytes experiment (head-major vs seq-major) ===\n\
         geometry: 48 layers, kv_heads=8, head_dim=128, fp16, context 8192; 96 KV bindings; \
         granule {GRANULE_BYTES} B\n\
         min_bucket={MIN_BUCKET} (DEFAULT — on-demand commit ACTIVE, unlike #834)"
    );

    // ------------------------------------------------------------------ //
    // Phase A — head-vs-seq equality at the same live prefix.
    // ------------------------------------------------------------------ //
    eprintln!("\n--- Phase A: head-vs-seq committed KV at the same prefix (on-demand active) ---");
    let mut eq_rows: Vec<(Measurement, Measurement)> = Vec::new();
    for (prompt_chars, new_tokens) in EQ_POINTS {
        eprintln!("- point prompt_chars={prompt_chars} new_tokens={new_tokens}");
        let head = run_child_config(false, prompt_chars, new_tokens)?;
        let seq = run_child_config(true, prompt_chars, new_tokens)?;

        // Deterministic, layout-defining assertions.
        assert!(
            !head.token_ids.is_empty(),
            "head-major generated no tokens: {head:?}"
        );
        assert_eq!(
            head.token_ids, seq.token_ids,
            "seq-major token stream diverged from head-major (VOIDS the comparison, critical bug): \
             head={:?} seq={:?}",
            head.token_ids, seq.token_ids
        );
        assert_eq!(
            head.logical_len, seq.logical_len,
            "live prefix length diverged: head={} seq={}",
            head.logical_len, seq.logical_len
        );
        assert!(
            seq.seq_major && !head.seq_major,
            "layout resolution wrong: head.seq_major={} seq.seq_major={}",
            head.seq_major,
            seq.seq_major
        );

        // Instrument-integrity: on-demand commit is ACTIVE (the #834 fix). The
        // bucket grew past the initial floor and growth fired, so what we measure
        // is the on-demand path, not a forced eager full-bucket reservation.
        for m in [&head, &seq] {
            assert!(
                m.logical_len > MIN_BUCKET,
                "prefix too short to exercise growth (logical_len {} <= min_bucket {}): {m:?}",
                m.logical_len,
                MIN_BUCKET
            );
            assert!(
                m.growth_events >= 1,
                "expected on-demand KV growth to fire (mechanism #834 disabled): {m:?}"
            );
            assert!(
                committed_bucket(m) > MIN_BUCKET,
                "committed bucket must have grown past min_bucket on demand: {m:?}"
            );
        }

        // Mechanism guardrails: seq-major fixed stride keeps its graph and moves
        // 0 bytes; head-major grows a packed bucket stride. If a regression
        // flipped head-major to a fixed stride (which WOULD separate the floor),
        // these break.
        assert!(
            seq.max_len >= 8192,
            "seq-major must report the fixed full-context stride (>= 8192): {seq:?}"
        );
        assert_eq!(
            seq.kv_bytes_moved, 0,
            "seq-major fixed stride moves no KV bytes on growth: {seq:?}"
        );
        assert!(
            seq.growth_keeps >= 1,
            "seq-major must keep its captured graph across on-demand growth: {seq:?}"
        );
        assert_eq!(
            head.max_len,
            committed_bucket(&head),
            "head-major stride is its growing (packed) bucket: {head:?}"
        );

        // The headline: committed physical KV bytes are IDENTICAL. The kv_heads×
        // floor separation does NOT appear on the engine's real on-demand-commit
        // path. If seq-major ever commits LESS, that is the win — update
        // MEMORY_ARCHITECTURE.md's floor table and this guard.
        assert_eq!(
            head.physical_committed_bytes, seq.physical_committed_bytes,
            "committed physical KV bytes diverged (head={} seq={}). If seq-major now commits less, \
             the kv_heads× floor win landed — update MEMORY_ARCHITECTURE.md and this guard.",
            head.physical_committed_bytes, seq.physical_committed_bytes
        );

        // Phase B property, checked here on the head row: head-major's committed
        // floor is the DENSE PACKED bucket, STRICTLY BELOW the fixed-stride 8×
        // scatter floor (96 × kv_heads × ceil(bucket × head_dim × elem /
        // granule) granules). At bucket 8192 the per-head stripe is exactly one
        // granule and the two coincide, but bucket 8192 is unreachable on this
        // box (constraint 3), so every reachable bucket is strictly below.
        let bucket = committed_bucket(&head);
        let stripe_granules = (bucket * 128 * 2).div_ceil(GRANULE_BYTES); // head_dim * fp16
        let fixed_stride_floor_bytes = 96 * 8 * stripe_granules * GRANULE_BYTES;
        assert!(
            head.physical_committed_bytes < fixed_stride_floor_bytes,
            "head-major committed {} B reached the 8× fixed-stride scatter floor {} B at bucket \
             {bucket} — head-major is supposed to stay dense-packed below the 8192 crossover: \
             {head:?}",
            head.physical_committed_bytes,
            fixed_stride_floor_bytes
        );

        eq_rows.push((head, seq));
    }
    // Phase A must ramp across at least two distinct buckets, both showing
    // equality — one point could be a coincidence.
    let eq_buckets: std::collections::BTreeSet<usize> =
        eq_rows.iter().map(|(h, _)| committed_bucket(h)).collect();
    assert!(
        eq_buckets.len() >= 2,
        "Phase A did not cover >=2 distinct buckets (saw {eq_buckets:?})"
    );

    // ------------------------------------------------------------------ //
    // Phase B — head-major dense floor vs the kv_heads× fixed-stride floor.
    // ------------------------------------------------------------------ //
    // Reuse Phase A's head-major rows: the reachable head-major buckets are
    // exactly Phase A's (512, 1024). Re-driving them by prefill would only
    // re-hit the OOM ceiling (constraint 3); the dense-floor assertion has
    // already run above per point. Here we only report and re-confirm.
    eprintln!(
        "\n--- Phase B: head-major dense floor vs kv_heads× fixed-stride floor (reachable buckets) \
         ---"
    );
    eprintln!("  reachability: {HEAD_PREFILL_OOM_NOTE}");
    let ramp_rows: Vec<Measurement> = eq_rows.iter().map(|(h, _)| h.clone()).collect();
    let ramp_buckets: std::collections::BTreeSet<usize> =
        ramp_rows.iter().map(committed_bucket).collect();
    assert!(
        ramp_buckets.len() >= 2,
        "Phase B did not cover >=2 distinct head-major buckets (saw {ramp_buckets:?})"
    );

    // ------------------------------------------------------------------ //
    // Report.
    // ------------------------------------------------------------------ //
    eprintln!("\n=== PHASE A SUMMARY: head-vs-seq committed physical KV (in-place VMM) ===");
    eprintln!(
        "{:>8} | {:>8} | {:>14} | {:>14} | {:>6} | tokens | head growth_events/inval | seq \
         growth_events/keeps/moved",
        "prefix", "bucket", "head B", "seq B", "ratio"
    );
    for (head, seq) in &eq_rows {
        let ratio = head.physical_committed_bytes as f64 / seq.physical_committed_bytes as f64;
        eprintln!(
            "{:>8} | {:>8} | {:>14} | {:>14} | {:>5.2}x | {:>6} | {}/{} | {}/{}/{}",
            head.logical_len,
            committed_bucket(head),
            head.physical_committed_bytes,
            seq.physical_committed_bytes,
            ratio,
            head.token_ids == seq.token_ids,
            head.growth_events,
            head.invalidations,
            seq.growth_events,
            seq.growth_keeps,
            seq.kv_bytes_moved,
        );
    }

    eprintln!("\n=== PHASE B SUMMARY: head-major dense floor vs kv_heads× fixed-stride floor ===");
    eprintln!("  {HEAD_PREFILL_OOM_NOTE}");
    eprintln!(
        "{:>8} | {:>8} | {:>14} {:>8} {:>9} | {:>14} | dense/floor",
        "prefix", "bucket", "committed B", "~gr", "GiB", "8x-floor B"
    );
    for head in &ramp_rows {
        let bucket = committed_bucket(head);
        let stripe_granules = (bucket * 128 * 2).div_ceil(GRANULE_BYTES);
        let fixed_stride_floor_bytes = 96 * 8 * stripe_granules * GRANULE_BYTES;
        let ratio = head.physical_committed_bytes as f64 / fixed_stride_floor_bytes as f64;
        eprintln!(
            "{:>8} | {:>8} | {:>14} {:>8.0} {:>9.3} | {:>14} | {:.2}x",
            head.logical_len,
            bucket,
            head.physical_committed_bytes,
            granules(head.physical_committed_bytes),
            head.physical_committed_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            fixed_stride_floor_bytes,
            ratio,
        );
    }

    eprintln!(
        "\nMechanism / conclusion: with on-demand commit ACTIVE (growth fired, unlike #834's \
         min_bucket==capacity artifact), head-major and seq-major commit BYTE-IDENTICAL physical \
         KV at every shared prefix (Phase A). Head-major grows a PACKED bucket (per-head stride = \
         current bucket) and re-strides+re-captures on growth; seq-major commits the same dense \
         prefix on a FIXED 8192 stride, moving 0 bytes and keeping its graph. Same dense-from-zero \
         granules => identical committed KV. Head-major's committed floor stays on the dense bucket \
         at every reachable bucket (Phase B), well below the kv_heads× (8×) fixed-stride scatter \
         floor (last column) that the engine deliberately never instantiates. The 8× crossover \
         (bucket 8192) is PHYSICALLY UNREACHABLE on this 8 GiB box by any path: seq-major prefill \
         >512 is unconverted (hard error), head-major prefill >~600 tokens OOMs the attention \
         workspace, and decode is weight-offload bound (~0.25 tok/s). So the 8× separation cannot \
         be exercised at long context here at all. NEGATIVE RESULT: the 8× floor separation does \
         not appear where it CAN be measured, and the mechanism (dense packed growing bucket, not \
         fixed-stride scatter) shows it never will on the engine's real commit path."
    );

    Ok(())
}
