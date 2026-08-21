//! THROWAWAY Lever-B Phase-0 capture-stability probe (leverb-phase0).
//!
//! This is an `#[ignore]`d, UN-WIRED measurement probe. It does NOT change the
//! decode pipeline. It answers the single load-bearing question that gates the
//! multi-week Lever B build (capture-stable padded M=K verify):
//!
//!   Can a fixed-shape, padded M=K forward graph be captured, replay stably
//!   across ~1000 steps (including bucket-growth boundaries) at ~1 dispatch per
//!   verify, and cost ≈ ONE M=1 replay (not ~K×)?
//!
//! Pass = all three of:
//!   (a) instantiates capture-safe (no alloc/free/sync in the captured region),
//!   (b) replays ~1 dispatch/verify across bucket growth (no eager thrash),
//!   (c) per-verify replay wall ≈ M=1 replay wall.
//!
//! Run it deliberately (all 8 H200s must be checked idle first; pin one):
//!
//! ```bash
//! source .cudaenv.sh
//! CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_RUN_CUDA_SMOKE=1 \
//!   ONNX_GENAI_LEVERB_MODEL=/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda \
//!   cargo test -p onnx-genai-engine --features cuda,native-backend \
//!   --release leverb_phase0 -- --ignored --nocapture
//! ```
//!
//! Acceptance logic, draft models, KV-commit correctness, and the exact-greedy
//! near-tie guard are ALL out of Phase-0 scope. The probe deliberately dirties
//! device KV (correctness is out of scope) and discards each session.

use super::*;

fn leverb_model_dir() -> Option<std::path::PathBuf> {
    let model_dir = std::env::var_os("ONNX_GENAI_LEVERB_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from("/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda")
        });
    if !model_dir.join("model.onnx").is_file() {
        eprintln!(
            "skipping Lever-B phase0 probe; model is not installed (set ONNX_GENAI_LEVERB_MODEL to its directory)"
        );
        return None;
    }
    Some(model_dir)
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn median(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn argmax(row: &[f32]) -> TokenId {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as TokenId)
        .expect("logits row must not be empty")
}

fn load(model_dir: &std::path::Path, graph_capture: bool, kv_max: usize) -> NativeDecodeSession {
    // glm-4-9b declares its I/O in inference_metadata.yaml but leaves
    // `token_input` unset, and its two rank-2 int64 inputs (input_ids,
    // attention_mask) are ambiguous under shape-only autoderive. Load the real
    // metadata I/O and disambiguate explicitly, exactly as the pipeline would.
    let meta = onnx_genai_metadata::load_metadata(&model_dir.join("inference_metadata.yaml"))
        .expect("load inference_metadata.yaml");
    let mut io = meta
        .model
        .and_then(|m| m.io)
        .expect("inference_metadata.yaml declares model.io");
    io.sequence_source = Some(SequenceInputKind::TokenIds);
    if io.token_input.is_none() {
        io.token_input = Some("input_ids".into());
    }
    if io.attention_mask_input.is_none() {
        io.attention_mask_input = Some("attention_mask".into());
    }
    NativeDecodeSession::load_with_cuda_options_and_io_spec(
        model_dir.join("model.onnx"),
        NativeDecodeDevice::Cuda { index: Some(0) },
        NativeDecodeCudaOptions {
            kv_max_len: Some(kv_max),
            graph_capture: Some(graph_capture),
            ..NativeDecodeCudaOptions::default()
        },
        Some(&io),
    )
    .expect("load decoder")
}

#[cfg(feature = "native-cuda")]
#[test]
#[ignore = "Lever-B phase0 GPU probe; run deliberately with --ignored on a verified-idle H200"]
fn leverb_phase0_capture_probe() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping Lever-B phase0 probe; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    let Some(model_dir) = leverb_model_dir() else {
        return Ok(());
    };

    const K_MAX: usize = 8; // fixed padded capture width (real draft K=4 pads into this)
    const KV_MAX: usize = 2048; // hard cap; ~1000 steps cross 256/512/1024 buckets
    const DECODE_STEPS: usize = 1000;
    const REPLAYS: usize = 200;
    const VERIFY_ITERS: usize = 30;

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;
    let prompt = tokenizer.encode("The quick brown fox jumps over the lazy dog and then")?;
    eprintln!("[leverb-phase0] prompt tokens = {}", prompt.len());

    // ------------------------------------------------------------------
    // PART B — capture stability of the REAL M=1 machine across ~1000 steps
    // including bucket-growth boundaries. This is the exact "freeze mask to
    // bucket, re-capture only on growth" state machine and the exact batched
    // GQA/GEMM kernels the padded M=K graph inherits.
    // ------------------------------------------------------------------
    let mut sess = load(&model_dir, true, KV_MAX);
    let mut logits = sess.decode(&prompt, 0)?.pop().context("prefill logits")?;
    let vocab = logits.len();
    let mut step_walls_ns: Vec<u64> = Vec::with_capacity(DECODE_STEPS);
    for _ in 0..DECODE_STEPS {
        let token = argmax(&logits);
        let past = sess.current_len();
        let start = std::time::Instant::now();
        logits = sess
            .decode(&[token], past)?
            .pop()
            .context("decode logits")?;
        step_walls_ns.push(start.elapsed().as_nanos() as u64);
    }
    let stats = sess.cuda_kv_debug_stats().context("cuda stats")?;
    let g = &stats.graph;
    let median_step_ns = median(&step_walls_ns[10..]);
    let tok_per_s = 1e9 / median_step_ns as f64;
    eprintln!(
        "[leverb-phase0][B] steps={DECODE_STEPS} captures={} replays={} invalidations={} growth_keeps={} kv_growth_events={} enabled={} decline={:?}",
        g.captures,
        g.replays,
        g.invalidations,
        g.growth_keeps,
        stats.kv_growth_events,
        g.enabled,
        g.decline_reason
    );
    eprintln!(
        "[leverb-phase0][B] median M=1 captured step wall = {:.3} ms ({:.1} tok/s); logical_len={}",
        ms(median_step_ns),
        tok_per_s,
        stats.logical_len
    );
    let growths = stats.kv_growth_events;
    drop(sess);

    // ------------------------------------------------------------------
    // PART C — eager M=K forward-wall SCALING sweep on the REAL kernels.
    // Each is a SINGLE forward (same op/dispatch count); the wall-vs-M curve
    // isolates how compute+activation+logit-readback+workspace-alloc scale with
    // M. A flat curve (≈1×) would validate "weights read once per token, K×
    // arithmetic is free on an idle GPU". We ALSO record per-forward device
    // (alloc,free) counts, because the un-built eager M=K path allocates fresh
    // workspaces per op — a confound that a captured/pre-allocated path removes,
    // so the eager wall is an UPPER BOUND on the captured M=K cost, not the
    // captured cost itself.
    // ------------------------------------------------------------------
    let mut sess = load(&model_dir, true, KV_MAX);
    sess.decode(&prompt, 0)?;
    // Advance a little so past_len is realistic and inside a stable bucket.
    for _ in 0..16 {
        let past = sess.current_len();
        sess.decode(&[1], past)?; // token value irrelevant to timing
    }
    let past0 = sess.current_len();

    let alloc_now = |sess: &NativeDecodeSession| -> (u64, u64) {
        sess.cuda_kv_debug_stats()
            .map(|s| {
                (
                    s.graph.allocation_counts.allocations,
                    s.graph.allocation_counts.frees,
                )
            })
            .unwrap_or((0, 0))
    };
    let time_verify =
        |sess: &mut NativeDecodeSession, m: usize| -> anyhow::Result<(u64, (u64, u64))> {
            let draft = vec![1 as TokenId; m];
            let before = alloc_now(sess);
            let start = std::time::Instant::now();
            let rows = sess.decode_verify(&draft, past0)?;
            let ns = start.elapsed().as_nanos() as u64;
            debug_assert_eq!(rows.len(), m);
            let after = alloc_now(sess);
            sess.rewind(past0)?;
            Ok((
                ns,
                (
                    after.0.saturating_sub(before.0),
                    after.1.saturating_sub(before.1),
                ),
            ))
        };

    let ms_of = [1usize, 2, 4, K_MAX];
    let mut med_by_m: std::collections::BTreeMap<usize, u64> = std::collections::BTreeMap::new();
    for &m in &ms_of {
        // Warm.
        let _ = time_verify(&mut sess, m)?;
        let mut walls = Vec::with_capacity(VERIFY_ITERS);
        let mut last_alloc = (0u64, 0u64);
        for _ in 0..VERIFY_ITERS {
            let (ns, alloc) = time_verify(&mut sess, m)?;
            walls.push(ns);
            last_alloc = alloc;
        }
        let med = median(&walls);
        med_by_m.insert(m, med);
        eprintln!(
            "[leverb-phase0][C] eager M={m}: wall = {:.3} ms | device alloc/free per forward = {}/{}",
            ms(med),
            last_alloc.0,
            last_alloc.1
        );
    }
    let med_1 = med_by_m[&1];
    let med_k = med_by_m[&K_MAX];
    let eager_ratio = med_k as f64 / med_1.max(1) as f64;
    let per_row_ns = (med_k.saturating_sub(med_1)) as f64 / (K_MAX - 1) as f64;
    // Decompose the curve: the M=1->M=2 "cliff" is the fixed penalty for leaving
    // the single-token fast path (per-op host dispatch + generic multi-row GEMM
    // + scratch alloc — exactly the overhead CUDA-graph capture removes); the
    // M=2..K_MAX "tail" slope is the true marginal compute per extra verify row.
    let cliff_ns = med_by_m[&2].saturating_sub(med_1);
    let tail_slope_ns = med_by_m[&K_MAX].saturating_sub(med_by_m[&2]) as f64 / (K_MAX - 2) as f64;
    let logit_readback_bytes = K_MAX * vocab * std::mem::size_of::<f32>();
    eprintln!(
        "[leverb-phase0][C] eager M={K_MAX}/M=1 ratio = {:.2}x | incremental per-row wall ≈ {:.3} ms (vs M=1 base {:.3} ms; vocab={vocab})",
        eager_ratio,
        per_row_ns / 1e6,
        ms(med_1)
    );
    eprintln!(
        "[leverb-phase0][C] curve decomposition: M=1->M=2 CLIFF = {:.3} ms (un-captured multi-row overhead) | M=2..{K_MAX} TAIL slope = {:.3} ms/row (marginal compute per extra verify row)",
        cliff_ns as f64 / 1e6,
        tail_slope_ns / 1e6
    );
    eprintln!(
        "[leverb-phase0][C] M={K_MAX} host logit readback = {:.2} MB",
        logit_readback_bytes as f64 / (1024.0 * 1024.0)
    );
    drop(sess);

    // ------------------------------------------------------------------
    // PART A — REAL padded M=K_MAX capture ATTEMPT + timed replay, and an M=1
    // capture attempt for the apples-to-apples replay-wall comparison. Fresh
    // sessions: the probe dirties device KV, so it must not pollute B/C.
    // ------------------------------------------------------------------
    let mut sess = load(&model_dir, true, KV_MAX);
    sess.decode(&prompt, 0)?;
    for _ in 0..16 {
        let past = sess.current_len();
        sess.decode(&[1], past)?;
    }
    let mk = sess.leverb_phase0_capture_attempt(K_MAX, REPLAYS)?;
    drop(sess);

    let mut sess = load(&model_dir, true, KV_MAX);
    sess.decode(&prompt, 0)?;
    for _ in 0..16 {
        let past = sess.current_len();
        sess.decode(&[1], past)?;
    }
    let m1 = sess.leverb_phase0_capture_attempt(1, REPLAYS)?;
    drop(sess);

    let mk_med = median(&mk.replay_walls_ns);
    let m1_med = median(&m1.replay_walls_ns);
    eprintln!(
        "[leverb-phase0][A] M={K_MAX} capture: captured={} segments={} rows={} past_len={} bucket={} alloc_delta={:?} decline={:?}",
        mk.captured, mk.segments, mk.rows, mk.past_len, mk.bucket, mk.alloc_delta, mk.decline
    );
    eprintln!(
        "[leverb-phase0][A] M=1 capture:  captured={} segments={} rows={} past_len={} bucket={} alloc_delta={:?} decline={:?}",
        m1.captured, m1.segments, m1.rows, m1.past_len, m1.bucket, m1.alloc_delta, m1.decline
    );
    if mk.captured && m1.captured {
        eprintln!(
            "[leverb-phase0][A] captured replay wall: M={K_MAX} = {:.3} ms | M=1 = {:.3} ms | ratio = {:.2}x",
            ms(mk_med),
            ms(m1_med),
            mk_med as f64 / m1_med.max(1) as f64
        );
    }

    // ------------------------------------------------------------------
    // PART D — INCREMENT-0: the same REAL M=K forward, now with the three
    // capture-enablement fixes the Phase-0 (a)-FAIL identified applied as a
    // test-only overlay (persistent padded [1,K,vocab] logits binding +
    // pre-capture warm forward to grow the alloc-free scratch arena + inherited
    // KV-symbol pin). This is THE decisive measurement: does a CAPTURED M=K
    // replay cost ≈ a CAPTURED M=1 replay (cliff was dispatch → Lever B GO), or
    // does the ~80ms eager floor persist under capture (cliff is generic
    // GEMM/arithmetic → Lever B NO-GO, promote Lever A)?
    // ------------------------------------------------------------------
    let mut sess = load(&model_dir, true, KV_MAX);
    sess.decode(&prompt, 0)?;
    for _ in 0..16 {
        let past = sess.current_len();
        sess.decode(&[1], past)?;
    }
    let d_mk = sess.leverb_increment0_capture_attempt(K_MAX, REPLAYS)?;
    drop(sess);

    let mut sess = load(&model_dir, true, KV_MAX);
    sess.decode(&prompt, 0)?;
    for _ in 0..16 {
        let past = sess.current_len();
        sess.decode(&[1], past)?;
    }
    let d_m1 = sess.leverb_increment0_capture_attempt(1, REPLAYS)?;
    drop(sess);

    let d_mk_med = median(&d_mk.replay_walls_ns);
    let d_m1_med = median(&d_m1.replay_walls_ns);
    eprintln!(
        "[leverb-phase0][D] INC0 M={K_MAX} capture: captured={} segments={} rows={} bucket={} warm_alloc={:?} capture_alloc={:?} decline={:?}",
        d_mk.captured,
        d_mk.segments,
        d_mk.rows,
        d_mk.bucket,
        d_mk.warm_alloc_delta,
        d_mk.alloc_delta,
        d_mk.decline
    );
    eprintln!(
        "[leverb-phase0][D] INC0 M={K_MAX} seam nodes (root cause of segmented capture): {}",
        d_mk.seam_summary
            .as_deref()
            .unwrap_or("<none: whole-graph>")
    );
    eprintln!(
        "[leverb-phase0][D] INC0 M=1  capture: captured={} segments={} rows={} bucket={} warm_alloc={:?} capture_alloc={:?} decline={:?}",
        d_m1.captured,
        d_m1.segments,
        d_m1.rows,
        d_m1.bucket,
        d_m1.warm_alloc_delta,
        d_m1.alloc_delta,
        d_m1.decline
    );
    let d_captured_ratio = if d_mk.captured && d_m1.captured {
        let ratio = d_mk_med as f64 / d_m1_med.max(1) as f64;
        eprintln!(
            "[leverb-phase0][D] DECISIVE captured replay wall: M={K_MAX} = {:.3} ms | M=1 = {:.3} ms | ratio = {:.2}x",
            ms(d_mk_med),
            ms(d_m1_med),
            ratio
        );
        Some(ratio)
    } else {
        eprintln!(
            "[leverb-phase0][D] DECISIVE number UNAVAILABLE: M={K_MAX} captured={} M=1 captured={} (see decline above)",
            d_mk.captured, d_m1.captured
        );
        None
    };

    // ------------------------------------------------------------------
    // PART E — CAPTURED-vs-EAGER TOKEN PARITY at M=K (and M=1) on REAL tokens.
    // Fills the "captured M=K == eager M=K, same Marlin config, no tiled oracle"
    // cell: run the identical M=K forward eagerly (pre-capture warm) and via the
    // CAPTURED graph replay over the identical device bindings, then compare the
    // per-row greedy argmax AND the raw logits bytes. Deterministic (same kernel,
    // same inputs) → byte-identical expected; argmax equality is the token-level
    // gate even if low-order bits ever differ.
    // ------------------------------------------------------------------
    let prompt_i64: Vec<i64> = prompt.iter().map(|&t| t as i64).collect();
    let mut parity_pass = true;
    for &m in &[K_MAX, 1usize] {
        let mut sess = load(&model_dir, true, KV_MAX);
        sess.decode(&prompt, 0)?;
        for _ in 0..16 {
            let past = sess.current_len();
            sess.decode(&[1], past)?;
        }
        let p = sess.leverb_increment0_token_parity_attempt(m, &prompt_i64)?;
        drop(sess);

        if !p.captured {
            eprintln!(
                "[leverb-phase0][E] M={m} token-parity UNAVAILABLE: capture declined ({:?})",
                p.decline
            );
            parity_pass = false;
            continue;
        }
        let argmax_match = p.warm_argmax == p.replay_argmax;
        parity_pass &= argmax_match;
        eprintln!(
            "[leverb-phase0][E] M={m} captured-vs-eager: argmax_match={argmax_match} logits_byte_identical={:?} segments={} rows={}",
            p.logits_byte_identical, p.segments, p.rows
        );
        eprintln!(
            "[leverb-phase0][E] M={m} eager  argmax = {:?}",
            p.warm_argmax
        );
        eprintln!(
            "[leverb-phase0][E] M={m} capture argmax = {:?}",
            p.replay_argmax
        );
    }
    eprintln!(
        "[leverb-phase0][E] CAPTURED-vs-EAGER TOKEN PARITY = {}",
        if parity_pass {
            "PASS (captured == eager)"
        } else {
            "FAIL"
        }
    );

    // ------------------------------------------------------------------
    // VERDICT
    // ------------------------------------------------------------------
    // (a) capture-safe: real M=K capture instantiated, OR (fallback evidence)
    //     the eager M=K forward performed no device alloc/free in-region and the
    //     identical kernels already capture at M=1.
    // (a) capture-safe: the INCREMENT-0 M=K forward instantiated a device graph
    //     (padded logits binding + alloc-free warm arena); Phase-0's raw M=K
    //     attempt is retained above only as the pre-fix contrast.
    let a_direct = d_mk.captured;
    let a_alloc_clean = d_mk.alloc_delta == Some((0, 0));
    // (b) stable replay across growth: real M=1 machine replayed ~1/step and
    //     invalidated only around bucket growths (NOT the eager 6->280 thrash).
    let b_replays_per_step = g.replays as f64 / DECODE_STEPS as f64;
    let b_no_thrash = g.invalidations <= growths.max(1) * 2 + 2;
    let b_pass = g.enabled && b_replays_per_step >= 0.95 && b_no_thrash;
    // (c) per-verify wall ≈ M=1: THE DECISIVE test is the INCREMENT-0 CAPTURED
    //     M=K vs CAPTURED M=1 replay-wall ratio. The eager ratio is retained
    //     only as the pre-capture upper bound.
    let c_captured_pass = d_captured_ratio.is_some_and(|ratio| ratio <= 1.5);
    let c_eager_pass = eager_ratio <= 1.5;

    eprintln!(
        "[leverb-phase0][VERDICT] (a) capture-safe: inc0_capture={a_direct} capture_alloc_clean={a_alloc_clean} (phase0_raw_capture={})",
        mk.captured
    );
    eprintln!(
        "[leverb-phase0][VERDICT] (b) stable replay: replays/step={b_replays_per_step:.3} invalidations={} growths={growths} pass={b_pass}",
        g.invalidations
    );
    eprintln!(
        "[leverb-phase0][VERDICT] (c) wall≈M=1: DECISIVE captured_ratio={d_captured_ratio:?} captured_pass={c_captured_pass} | eager_ratio(upper bound)={eager_ratio:.2} eager_pass={c_eager_pass}"
    );

    // GO requires the DECISIVE captured evidence: (a) the M=K graph instantiated
    // capture-safe, (b) the machine replays ~1/step across growth, and (c) the
    // CAPTURED M=K replay wall ≈ the CAPTURED M=1 replay wall. A green (a)/(b)
    // with a captured M=K wall still on the ~80ms eager floor is a NO-GO for B.
    let go = a_direct && b_pass && c_captured_pass;
    eprintln!(
        "[leverb-phase0][VERDICT] GO/NO-GO = {}",
        if go { "GO" } else { "NO-GO" }
    );

    // Sanity assertions (the probe must have exercised the real machine); the
    // perf verdict itself is reported, not hard-asserted, so the raw numbers are
    // always visible under --nocapture.
    assert!(
        g.enabled,
        "graph capture was not even enabled on this model"
    );
    assert!(
        g.replays as usize >= DECODE_STEPS / 2,
        "expected the M=1 machine to replay across the run"
    );
    Ok(())
}
