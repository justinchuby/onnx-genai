//! Marlin int4 GEMM BEFORE/AFTER perf harness (`marlin_bench`).
//!
//! A reproducible, self-contained measurement harness that gates the Marlin
//! int4 GEMM build (Lever A) and the capture-stable M=K verify build (Lever B).
//! It establishes the BEFORE baseline on today's code so the moment Marlin
//! lands the same command produces the decisive delta.
//!
//! It measures two of the four Marlin gate axes directly (the other two are
//! covered by companion tools — see the module docs and the decision drop):
//!
//!   Axis 2 — **M-scaling wall (the cliff).** For M ∈ {1,2,4,8,16} it times a
//!     single real multi-row forward (`decode_verify`) on the model's own
//!     MatMulNBits stack at a realistic `past_len`, and decomposes the curve
//!     into the M=1→M=2 *cliff* (the fixed penalty for leaving the captured
//!     single-token fast path) and the M≥2 *tail* slope (marginal compute per
//!     extra verify row). This is the eager, capture-breaking wall Marlin +
//!     Lever B must collapse. Baseline reference (Deckard Increment-0, glm-4-9b,
//!     H200): M=1 ≈ 10 ms, M=8 ≈ 87 ms, cliff ≈ 67 ms, tail ≈ 1.85 ms/row.
//!
//!   Axis 3 — **End-to-end tok/s** (native CUDA EP). With `--e2e` it times
//!     prefill wall and steady-state decode tok/s through the captured decode
//!     path. Run the identical command against `profile_native --backend ort`
//!     for the head-to-head ORT comparison (see the decision drop).
//!
//! Companion tools for the other two axes:
//!   Axis 1 — **M=1 DRAM% microbench** — Nsight `ncu` on the int4 GEMV kernel
//!     via `profile_native` (see `.agents/skills/profiling/SKILL.md` and the
//!     decision drop for the exact `ncu` command).
//!   Axis 4 — **Capture-safety re-probe** — the `leverb_phase0` test in
//!     `onnx-genai-engine` (`segments`, captured-M=8 vs M=1 ratio); wired to be
//!     re-run against Marlin.
//!
//! Portability (Rule 11): Marlin is SM80+. On today's (pre-Marlin) code every
//! number here IS the fallback / current split-K layout, so re-running this
//! harness unchanged on a <SM80 device (or post-Marlin with the fallback
//! selected) proves the fallback path does not regress. The harness prints the
//! device compute capability so the arch the numbers belong to is never
//! ambiguous.
//!
//! MEASUREMENT DISCIPLINE: warms up, reports median + full spread (min / p10 /
//! p90 / max), names the GPU + head SHA, and re-checks the device is idle before
//! the run. Never a single-shot or unpinned number.
//!
//! Usage (pin a verified-idle HIGH-index GPU on the shared 8× H200 box):
//!
//! ```bash
//! source /home/justinchu/onnx-genai/.cudaenv.sh
//! nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # verify idle
//! CUDA_VISIBLE_DEVICES=7 \
//!   cargo run --release -p onnx-genai-bench --features bench-native,cuda \
//!   --bin marlin_bench -- \
//!   --model /home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda \
//!   --label glm-4-9b-int4 --past 512 --iters 30 --e2e --e2e-tokens 128
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use onnx_genai_engine::{NativeDecodeDevice, NativeDecodeSession};
use tokenizers::Tokenizer;

type TokenId = u32;

#[derive(Debug, Parser)]
#[command(
    about = "Marlin int4 GEMM BEFORE/AFTER perf harness: M-scaling wall (the cliff) + e2e tok/s"
)]
struct Args {
    /// Model directory (must contain model.onnx + inference_metadata.yaml or
    /// genai_config.json + tokenizer.json).
    #[arg(long)]
    model: PathBuf,

    /// Human label for the model in the report (e.g. glm-4-9b-int4).
    #[arg(long, default_value = "model")]
    label: String,

    /// Comma-separated M values for the scaling wall.
    #[arg(long, default_value = "1,2,4,8,16")]
    m_list: String,

    /// Realistic `past_len` (KV context) at which the M-scaling wall is timed.
    #[arg(long, default_value_t = 512)]
    past: usize,

    /// Timed iterations per M (median + spread reported over these).
    #[arg(long, default_value_t = 30)]
    iters: usize,

    /// Warmup iterations per M before timing.
    #[arg(long, default_value_t = 3)]
    warmups: usize,

    /// Prompt used to prime a realistic KV cache before measuring.
    #[arg(
        long,
        default_value = "The quick brown fox jumps over the lazy dog and then"
    )]
    prompt: String,

    /// Also measure end-to-end native prefill wall + steady decode tok/s.
    #[arg(long, default_value_t = false)]
    e2e: bool,

    /// Number of decode tokens for the `--e2e` steady-state throughput window.
    #[arg(long, default_value_t = 128)]
    e2e_tokens: usize,

    /// Steady-window warmup tokens skipped before timing e2e decode.
    #[arg(long, default_value_t = 8)]
    e2e_skip: usize,

    /// Comma-separated prompt lengths (tokens) for the long-prompt prefill
    /// sweep, e.g. "512,1024,2048". Empty (default) disables the sweep. Prefill
    /// is an M>1 workload, so this is the clean BEFORE prefill number that
    /// Marlin's cliff-collapse must improve end-to-end.
    #[arg(long, default_value = "")]
    prefill_lens: String,

    /// Timed iterations per prefill length (contention-invariant MIN reported).
    #[arg(long, default_value_t = 5)]
    prefill_iters: usize,

    /// In-process CUDA device index (after CUDA_VISIBLE_DEVICES remap this is 0).
    #[arg(long, default_value_t = 0)]
    device: u32,
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn parse_m_list(value: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in value.split(',') {
        let m = part
            .trim()
            .parse::<usize>()
            .with_context(|| format!("parse M value {part:?}"))?;
        if m == 0 {
            bail!("M values must be positive");
        }
        out.push(m);
    }
    if out.is_empty() {
        bail!("--m-list must contain at least one M");
    }
    Ok(out)
}

/// Like `parse_m_list` but an empty/whitespace string yields an empty vec
/// (used for the optional prefill sweep length list).
fn parse_m_list_opt(value: &str) -> Result<Vec<usize>> {
    if value.trim().is_empty() {
        return Ok(Vec::new());
    }
    parse_m_list(value)
}

/// Percentile of an unsorted slice (nearest-rank on a copy).
fn percentile(values: &[u64], pct: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (pct / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn median(values: &[u64]) -> u64 {
    percentile(values, 50.0)
}

/// Report head SHA from the environment (CI) or `git rev-parse` (best effort).
fn head_sha() -> String {
    if let Ok(sha) = std::env::var("MARLIN_BENCH_SHA") {
        return sha;
    }
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Query `nvidia-smi` for the visible device name and re-check no foreign
/// compute app is contending (measurement discipline: verify idle before a run).
fn gpu_report() {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<all>".to_string());
    println!("  CUDA_VISIBLE_DEVICES = {visible}");
    if let Some(out) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,compute_cap,memory.used",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        // With CUDA_VISIBLE_DEVICES set nvidia-smi still lists all GPUs; the
        // visible index is the process's device 0. Print the whole map so the
        // reader can see which physical GPU the pinned index resolves to.
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            println!("  gpu: {}", line.trim());
        }
    }
    if let Some(out) = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let apps = String::from_utf8_lossy(&out.stdout);
        let apps = apps.trim();
        if apps.is_empty() {
            println!("  compute-apps: <none> (idle)");
        } else {
            println!("  compute-apps (verify none contend with the pinned GPU):");
            for line in apps.lines() {
                println!("    {}", line.trim());
            }
        }
    }
}

fn argmax(row: &[f32]) -> TokenId {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as TokenId)
        .expect("logits row must not be empty")
}

fn load(model: &std::path::Path, device: u32) -> Result<NativeDecodeSession> {
    NativeDecodeSession::load_with_resolved_io(
        model.join("model.onnx"),
        NativeDecodeDevice::Cuda {
            index: Some(device),
        },
    )
    .context("load native CUDA decoder (resolved IO)")
}

/// Prime a realistic KV cache: prefill the prompt, then greedily advance real
/// tokens until `current_len == past`. Returns the final decode logits row.
fn prime_to_past(
    sess: &mut NativeDecodeSession,
    prompt: &[TokenId],
    past: usize,
) -> Result<Vec<f32>> {
    let mut logits = sess
        .decode(prompt, 0)?
        .pop()
        .context("prefill produced no logits")?;
    while sess.current_len() < past {
        let token = argmax(&logits);
        let at = sess.current_len();
        logits = sess
            .decode(&[token], at)?
            .pop()
            .context("advance decode produced no logits")?;
    }
    Ok(logits)
}

struct WallStats {
    m: usize,
    median_ns: u64,
    min_ns: u64,
    p10_ns: u64,
    p90_ns: u64,
    max_ns: u64,
}

fn measure_m_wall(
    sess: &mut NativeDecodeSession,
    m: usize,
    past: usize,
    warmups: usize,
    iters: usize,
) -> Result<WallStats> {
    let draft = vec![1 as TokenId; m];
    // Warm.
    for _ in 0..warmups {
        let rows = sess.decode_verify(&draft, past)?;
        debug_assert_eq!(rows.len(), m);
        sess.rewind(past)?;
    }
    let mut walls = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let rows = sess.decode_verify(&draft, past)?;
        walls.push(start.elapsed().as_nanos() as u64);
        debug_assert_eq!(rows.len(), m);
        sess.rewind(past)?;
    }
    Ok(WallStats {
        m,
        median_ns: median(&walls),
        min_ns: percentile(&walls, 0.0),
        p10_ns: percentile(&walls, 10.0),
        p90_ns: percentile(&walls, 90.0),
        max_ns: percentile(&walls, 100.0),
    })
}

fn print_graph_stats(sess: &NativeDecodeSession, tag: &str) {
    if let Some(stats) = sess.cuda_kv_debug_stats() {
        let g = &stats.graph;
        println!(
            "  [{tag}] capture: enabled={} captures={} replays={} invalidations={} growth_keeps={} fallbacks={} decline={:?}",
            g.enabled,
            g.captures,
            g.replays,
            g.invalidations,
            g.growth_keeps,
            g.fallbacks,
            g.decline_reason
        );
    }
}

/// Build a prompt of exactly `target` tokens by cycling the base prompt ids.
fn build_prompt_of_len(base: &[TokenId], target: usize) -> Vec<TokenId> {
    let mut out = Vec::with_capacity(target);
    while out.len() < target {
        for &t in base {
            if out.len() == target {
                break;
            }
            out.push(t);
        }
    }
    out
}

/// Axis 3b: long-prompt prefill sweep. For each length L we time a fresh
/// `decode(prompt, 0)` prefill wall (KV reset via `rewind(0)` between iters) and
/// report prefill tok/s = L / wall. Prefill is an M>1 GEMM workload, so it is a
/// direct Marlin gate; the contention-invariant MIN is the reported statistic.
fn run_prefill_sweep(
    sess: &mut NativeDecodeSession,
    base: &[TokenId],
    lens: &[usize],
    iters: usize,
) -> Result<()> {
    println!();
    println!("== Axis 3b: long-prompt prefill sweep (prefill tok/s vs prompt length) ==");
    println!(
        "  contention-invariant MIN across {iters} iters (+1 warmup); prefill is a Marlin M>1 gate"
    );
    for &len in lens {
        let prompt = build_prompt_of_len(base, len);
        // Warmup (excludes first-touch alloc/autotune from the timed window).
        sess.rewind(0)?;
        let _ = sess.decode(&prompt, 0)?;
        let mut walls = Vec::with_capacity(iters);
        for _ in 0..iters {
            sess.rewind(0)?;
            let start = Instant::now();
            let _ = sess.decode(&prompt, 0)?;
            walls.push(start.elapsed().as_nanos() as u64);
        }
        let min_ns = percentile(&walls, 0.0);
        let med_ns = median(&walls);
        println!(
            "  L={:<5} min {:>9.3} ms ({:>9.1} tok/s) | median {:>9.3} ms ({:>9.1} tok/s)",
            len,
            ms(min_ns),
            len as f64 / (min_ns as f64 / 1e9),
            ms(med_ns),
            len as f64 / (med_ns as f64 / 1e9),
        );
    }
    print_graph_stats(sess, "prefill-sweep");
    Ok(())
}

fn run_e2e(sess: &mut NativeDecodeSession, prompt: &[TokenId], args: &Args) -> Result<()> {
    println!();
    println!("== Axis 3: end-to-end native CUDA tok/s ==");
    // Prefill.
    let prefill_start = Instant::now();
    let mut logits = sess
        .decode(prompt, 0)?
        .pop()
        .context("prefill produced no logits")?;
    let prefill_ns = prefill_start.elapsed().as_nanos() as u64;
    let prefill_tok_s = prompt.len() as f64 / (prefill_ns as f64 / 1e9);
    println!(
        "  prefill: {} prompt tokens in {:.3} ms ({:.1} tok/s)",
        prompt.len(),
        ms(prefill_ns),
        prefill_tok_s
    );

    // Steady decode window.
    let total = args.e2e_tokens;
    let mut step_walls = Vec::with_capacity(total);
    let mut greedy_tokens: Vec<TokenId> = Vec::with_capacity(total);
    for _ in 0..total {
        let token = argmax(&logits);
        greedy_tokens.push(token);
        let at = sess.current_len();
        let start = Instant::now();
        logits = sess
            .decode(&[token], at)?
            .pop()
            .context("decode produced no logits")?;
        step_walls.push(start.elapsed().as_nanos() as u64);
    }
    let dump = greedy_tokens.len().min(20);
    println!(
        "  greedy first {dump} tokens = {:?}",
        &greedy_tokens[..dump]
    );
    let skip = args.e2e_skip.min(step_walls.len().saturating_sub(1));
    let window = &step_walls[skip..];
    let med = median(window);
    let tok_s = 1e9 / med as f64;
    println!(
        "  decode steady window: {} tokens (skipped {} warmup) | median step = {:.3} ms | p10 = {:.3} | p90 = {:.3} | {:.2} tok/s",
        window.len(),
        skip,
        ms(med),
        ms(percentile(window, 10.0)),
        ms(percentile(window, 90.0)),
        tok_s
    );
    print_graph_stats(sess, "e2e");
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let m_list = parse_m_list(&args.m_list)?;

    println!("=== marlin_bench — Marlin int4 GEMM BEFORE/AFTER harness ===");
    println!("  head SHA   = {}", head_sha());
    println!("  model      = {} ({})", args.label, args.model.display());
    println!(
        "  past_len   = {}  iters = {}  warmups = {}",
        args.past, args.iters, args.warmups
    );
    gpu_report();

    let tokenizer = Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(args.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?;
    let prompt: Vec<TokenId> = encoding.get_ids().to_vec();
    if prompt.is_empty() {
        bail!("prompt encoded to zero tokens");
    }
    println!("  prompt     = {} tokens", prompt.len());

    // ---- Axis 2: M-scaling wall (the cliff) ----
    println!();
    println!("== Axis 2: M-scaling wall (single eager forward per M; capture-breaking) ==");
    let mut sess = load(&args.model, args.device)?;
    let _ = prime_to_past(&mut sess, &prompt, args.past)?;
    let past0 = sess.current_len();
    println!("  primed KV to past_len = {past0}");
    print_graph_stats(&sess, "primed");

    let mut stats_by_m: std::collections::BTreeMap<usize, WallStats> =
        std::collections::BTreeMap::new();
    for &m in &m_list {
        let s = measure_m_wall(&mut sess, m, past0, args.warmups, args.iters)?;
        println!(
            "  M={:<3} median = {:>8.3} ms | min {:>8.3} | p10 {:>8.3} | p90 {:>8.3} | max {:>8.3}",
            s.m,
            ms(s.median_ns),
            ms(s.min_ns),
            ms(s.p10_ns),
            ms(s.p90_ns),
            ms(s.max_ns)
        );
        stats_by_m.insert(m, s);
    }

    // Curve decomposition (needs M=1 and M=2 to name the cliff).
    if let Some(m1) = stats_by_m.get(&1) {
        let base = m1.median_ns;
        let max_m = *stats_by_m.keys().max().unwrap();
        let top = stats_by_m[&max_m].median_ns;
        let per_row = (top.saturating_sub(base)) as f64 / (max_m.max(1) - 1).max(1) as f64;
        println!("  --- curve decomposition ---");
        println!("  M=1 base wall            = {:.3} ms", ms(base));
        if let Some(m2) = stats_by_m.get(&2) {
            let cliff = m2.median_ns.saturating_sub(base);
            let tail =
                (top.saturating_sub(m2.median_ns)) as f64 / (max_m.saturating_sub(2)).max(1) as f64;
            println!(
                "  M=1->M=2 CLIFF           = {:.3} ms (fixed penalty leaving captured M=1 fast path)",
                ms(cliff)
            );
            println!(
                "  M=2..M={} TAIL slope      = {:.3} ms/row (marginal compute per extra verify row)",
                max_m,
                tail / 1e6
            );
        }
        println!(
            "  M={}/M=1 ratio            = {:.2}x | mean per-row = {:.3} ms",
            max_m,
            top as f64 / base.max(1) as f64,
            per_row / 1e6
        );
        println!("  NOTE: this is the eager, capture-breaking wall. Marlin (kernel) + Lever B");
        println!("        (capture-stable M=K verify) must collapse the cliff toward ~1x.");
    }
    drop(sess);

    // ---- Axis 3: end-to-end tok/s (optional) ----
    if args.e2e {
        let mut sess = load(&args.model, args.device)?;
        run_e2e(&mut sess, &prompt, &args)?;
        drop(sess);
    }

    // ---- Axis 3b: long-prompt prefill sweep (optional) ----
    let prefill_lens = parse_m_list_opt(&args.prefill_lens)?;
    if !prefill_lens.is_empty() {
        let mut sess = load(&args.model, args.device)?;
        run_prefill_sweep(&mut sess, &prompt, &prefill_lens, args.prefill_iters)?;
        drop(sess);
    }

    println!();
    println!("=== marlin_bench complete ===");
    Ok(())
}
