//! Root-cause probe for the native prompt-lookup divergence (M=K verify argmax
//! != M=1 greedy argmax) reported in PR #932.
//!
//! It does NOT change the engine. It replays a greedy decode on the native CUDA
//! path (the captured M=1 hot path used by real generation), then re-runs the
//! SAME positions through the eager M=K `decode_verify` primitive using the
//! greedy continuation itself as the draft. Because the draft is the true greedy
//! continuation, every verify row has mathematically identical causal inputs to
//! the corresponding M=1 forward, so any logit difference is pure kernel numerics
//! (M=K batched eager forward vs M=1 captured forward). We then classify the
//! divergence:
//!   1. near-tie fp noise  (argmax flips only where top1..top2 gap ~ the |delta|)
//!   2. systematic offset  (a consistent scale/bias across the whole vocab row)
//!   3. wildly different   (row looks like a different position -> real bug)
//!
//! Usage:
//!   verify_logits_probe --model DIR --prompt TEXT [--tokens N] [--k K] [--ep cuda]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use onnx_genai_engine::{NativeDecodeDevice, NativeDecodeSession};
use onnx_genai_ort::Tokenizer;

#[derive(Debug, Parser)]
#[command(about = "Compare M=1 greedy logits vs M=K eager-verify logits per position")]
struct Args {
    /// Native decoder directory (expects model.onnx + tokenizer.json) or an
    /// explicit model.onnx path.
    #[arg(long)]
    model: PathBuf,
    /// Prompt text.
    #[arg(long, default_value = "Explain how a hash map works in simple terms.")]
    prompt: String,
    /// Number of greedy tokens to generate / scan.
    #[arg(long, default_value_t = 64)]
    tokens: usize,
    /// Verify draft width K (M=K block size).
    #[arg(long, default_value_t = 4)]
    k: usize,
    /// Execution provider (cuda or cpu).
    #[arg(long, default_value = "cuda")]
    ep: String,
    /// Top-N entries to print for each logit row at the divergence.
    #[arg(long, default_value_t = 5)]
    topk: usize,
}

fn model_file(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("model.onnx")
    } else {
        path.to_path_buf()
    }
}

fn tokenizer_file(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("tokenizer.json")
    } else {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join("tokenizer.json")
    }
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Indices sorted by descending logit, truncated to `n`.
fn topk(logits: &[f32], n: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.into_iter().take(n).map(|i| (i, logits[i])).collect()
}

/// top1 - top2 gap (how close the argmax race was).
fn top1_top2_gap(logits: &[f32]) -> f32 {
    let mut first = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &v in logits {
        if v > first {
            second = first;
            first = v;
        } else if v > second {
            second = v;
        }
    }
    first - second
}

/// Difference statistics between two equal-length logit rows.
struct DiffStats {
    max_abs: f32,
    max_abs_at: usize,
    mean: f32,
    mean_abs: f32,
    max_rel: f32,
}

fn diff_stats(a: &[f32], b: &[f32]) -> DiffStats {
    let mut max_abs = 0.0f32;
    let mut max_abs_at = 0usize;
    let mut sum = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut max_rel = 0.0f32;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let d = x - y;
        let ad = d.abs();
        if ad > max_abs {
            max_abs = ad;
            max_abs_at = i;
        }
        let denom = x.abs().max(y.abs()).max(1e-6);
        max_rel = max_rel.max(ad / denom);
        sum += d as f64;
        sum_abs += ad as f64;
    }
    DiffStats {
        max_abs,
        max_abs_at,
        mean: (sum / a.len() as f64) as f32,
        mean_abs: (sum_abs / a.len() as f64) as f32,
        max_rel,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = match args.ep.as_str() {
        "cpu" => NativeDecodeDevice::Cpu,
        "cuda" => NativeDecodeDevice::Cuda { index: None },
        other => bail!("unknown --ep {other} (expected cuda or cpu)"),
    };

    let model = model_file(&args.model);
    let tokenizer = Tokenizer::from_file(tokenizer_file(&args.model))
        .context("load tokenizer.json beside native decoder")?;
    let prompt_tokens = tokenizer.encode(&args.prompt).context("tokenize prompt")?;
    if prompt_tokens.is_empty() {
        bail!("prompt tokenized to an empty sequence");
    }
    let prompt_len = prompt_tokens.len();
    let mut session = NativeDecodeSession::load_with_resolved_io(&model, device)
        .with_context(|| format!("load native decoder {}", model.display()))?;

    println!(
        "verify_probe: model={} ep={} prompt_len={} tokens={} k={}",
        model.display(),
        args.ep,
        prompt_len,
        args.tokens,
        args.k
    );

    // --- Phase 1: greedy M=1 decode, recording full logits per output index.
    //
    // A hybrid GDN + attention decoder carries destructive recurrent/conv state
    // that the attention-KV `rewind` cannot restore (it has no per-step history to
    // prefix-slice). Phase 2 rewinds the KV to `base` before each eager verify, so
    // to make the comparison a VALID oracle we must ALSO restore the recurrent
    // state to `S_base` — exactly what the speculative driver does via
    // `snapshot_recurrent_state` before a draft window. We therefore snapshot the
    // recurrent state at every `base` length here (keyed by `current_len`) and
    // restore the matching snapshot before each verify below. Non-recurrent
    // (pure-attention) decoders skip this entirely.
    let track_recurrent = session.has_recurrent_state_public();
    let mut recurrent_snapshots: std::collections::HashMap<
        usize,
        onnx_genai_engine::NativeRecurrentSnapshot,
    > = std::collections::HashMap::new();
    let last_p = args.tokens.saturating_sub(args.k);

    let prefill = session.decode(&prompt_tokens, 0)?;
    let mut logits = prefill
        .last()
        .context("prefill produced no logits")?
        .clone();
    let mut greedy_tokens: Vec<u32> = Vec::with_capacity(args.tokens);
    let mut greedy_logits: Vec<Vec<f32>> = Vec::with_capacity(args.tokens);
    for _ in 0..args.tokens {
        let tok = argmax(&logits) as u32;
        greedy_tokens.push(tok);
        greedy_logits.push(logits.clone());
        let past = session.current_len();
        // Snapshot S_base for every base a verify will start from (base in
        // [prompt_len, prompt_len + last_p - 1]); this is the state BEFORE the
        // m=1 forward that produces greedy_logits[base - prompt_len + 1], i.e. the
        // exact state the corresponding verify row 0 must run from.
        if track_recurrent && past >= prompt_len && past < prompt_len + last_p {
            recurrent_snapshots.insert(past, session.snapshot_recurrent_state_public()?);
        }
        logits = session
            .decode(std::slice::from_ref(&tok), past)?
            .last()
            .context("greedy decode produced no logits")?
            .clone();
    }
    println!("verify_probe: greedy_token_ids={greedy_tokens:?}");
    if track_recurrent {
        println!(
            "verify_probe: recurrent decoder detected — restoring recurrent state to S_base \
             before each verify ({} snapshots)",
            recurrent_snapshots.len()
        );
    }

    // --- Phase 2: for each position P, re-run the SAME greedy continuation as an
    // M=K draft through eager decode_verify, compare each row's argmax + logits to
    // the M=1 reference for the same output index.
    let mut first_div: Option<(usize, usize)> = None; // (P, row)
    let mut flip_count = 0usize;
    let mut min_flip_gap = f32::INFINITY;

    for p in 1..=last_p {
        // KV must hold prompt + greedy[0..P-2]; base = prompt_len + P - 1 so row 0
        // predicts output index P from context identical to greedy_logits[P].
        let base = prompt_len + p - 1;
        session.rewind(base)?;
        // Restore the destructive recurrent/conv state to S_base so verify starts
        // from the SAME state that produced greedy_logits[P] (the KV rewind above
        // only handles the prefix-sliceable attention cache).
        if track_recurrent {
            let snapshot = recurrent_snapshots
                .get(&base)
                .with_context(|| format!("missing recurrent snapshot for base {base}"))?;
            session.restore_recurrent_state_public(snapshot)?;
        }
        let draft: Vec<u32> = greedy_tokens[(p - 1)..(p - 1 + args.k)].to_vec();
        let rows = session.decode_verify(&draft, base)?;
        for (i, row) in rows.iter().enumerate() {
            let out_idx = p + i;
            let m1 = &greedy_logits[out_idx];
            let mk_arg = argmax(row);
            let m1_arg = argmax(m1);
            if mk_arg != m1_arg {
                flip_count += 1;
                let gap = top1_top2_gap(m1).abs().min(top1_top2_gap(row).abs());
                min_flip_gap = min_flip_gap.min(gap);
                if first_div.is_none() {
                    first_div = Some((p, i));
                    dump_divergence(out_idx, p, i, m1, row, &args, &greedy_tokens);
                }
            }
        }
    }

    println!(
        "verify_probe: SUMMARY flips={flip_count} first_divergence={:?} min_top1_top2_gap_at_flip={}",
        first_div,
        if min_flip_gap.is_finite() {
            format!("{min_flip_gap:.4}")
        } else {
            "n/a".into()
        }
    );
    if first_div.is_none() {
        println!(
            "verify_probe: NO argmax divergence across {} positions x k={} — M=K verify is \
             argmax-identical to M=1 on this run.",
            last_p, args.k
        );
    }
    Ok(())
}

fn dump_divergence(
    out_idx: usize,
    p: usize,
    row: usize,
    m1: &[f32],
    mk: &[f32],
    args: &Args,
    greedy_tokens: &[u32],
) {
    let stats = diff_stats(m1, mk);
    println!("========== FIRST DIVERGENCE ==========");
    println!(
        "output_index={out_idx} (verify base position P={p}, row_in_block={row}) greedy_token={}",
        greedy_tokens[out_idx]
    );
    println!(
        "M=1 argmax={} (logit {:.5})  |  M=K argmax={} (logit {:.5})",
        argmax(m1),
        m1[argmax(m1)],
        argmax(mk),
        mk[argmax(mk)]
    );
    println!(
        "M=1 top1-top2 gap={:.6}  M=K top1-top2 gap={:.6}",
        top1_top2_gap(m1),
        top1_top2_gap(mk)
    );
    println!(
        "logit diff: max|Δ|={:.6} at token {}  mean Δ={:.6}  mean|Δ|={:.6}  max_rel={:.4e}",
        stats.max_abs, stats.max_abs_at, stats.mean, stats.mean_abs, stats.max_rel
    );
    // Cross-evaluate: what does each kernel assign to the OTHER kernel's argmax?
    let a1 = argmax(m1);
    let ak = argmax(mk);
    println!(
        "at M=1 argmax token {a1}: M=1={:.5} M=K={:.5} (Δ={:.6})",
        m1[a1],
        mk[a1],
        m1[a1] - mk[a1]
    );
    println!(
        "at M=K argmax token {ak}: M=1={:.5} M=K={:.5} (Δ={:.6})",
        m1[ak],
        mk[ak],
        m1[ak] - mk[ak]
    );
    println!("M=1 top{}:", args.topk);
    for (t, v) in topk(m1, args.topk) {
        println!("    token {t:>6}  logit {v:.5}");
    }
    println!("M=K top{}:", args.topk);
    for (t, v) in topk(mk, args.topk) {
        println!("    token {t:>6}  logit {v:.5}");
    }
    println!("======================================");
}
