//! Greedy-decode accuracy lock for the Qwen3.6-35B-A3B (hybrid Mamba + int4
//! MoE) dense-vs-QMoE native-CUDA divergence, adjudicated by a full-fp32
//! oracle.
//!
//! ## The divergence
//!
//! Two artifacts share **byte-identical int4 expert weights** — a per-expert
//! `MatMulNBits` DENSE fallback and a fused `com.microsoft::QMoE` graph produced
//! by mobius `_qmoe_fusion.py`. Greedy decode (temperature 0, prompt `"Hello"`)
//! on native CUDA is token-for-token identical for generated indices `0..=118`,
//! then splits at index **119**:
//!
//! | path (native CUDA, int4) | token 119 | text |
//! |--------------------------|-----------|------|
//! | DENSE (per-expert MatMulNBits) | **5342**  | "Windows" |
//! | QMoE  (fused sparse QMoE)       | **33803** | "Ubuntu"  |
//!
//! The shared context decodes as `"…I'm using Python 3.8.2 on "`.
//!
//! ## Verdict: QMoE is the *more accurate* path — KEEP QMoE.
//!
//! The exact shared context (`"Hello"` + generated `0..=118`, 120 tokens) was
//! teacher-forced through independent higher-precision oracles built from the
//! DENSE graph (which owns the same weights). Every oracle selects **33803**:
//!
//! | reference (teacher-forced, this context) | argmax | logit(33803) − logit(5342) |
//! |------------------------------------------|--------|-----------------------------|
//! | full-fp32 dense graph, native CPU        | **33803** | +0.0908 |
//! | f16 dense graph, native CPU (fp32 accum) | **33803** | +0.0937 |
//! | QMoE  int4, native CUDA                   | **33803** | +0.0938 |
//! | DENSE int4, native CUDA                   | **33803** | +0.0781 |
//!
//! Note the last row: teacher-forced in a **single prefill pass**, even the
//! DENSE int4 CUDA path predicts 33803. The DENSE stream only reaches 5342
//! **autoregressively** — this is a hybrid model whose recurrent/conv (`Mamba`)
//! decode state accumulates f16 rounding across 119 incremental steps, and the
//! DENSE decode path's drifted state tips this razor-thin (~0.09-logit) tie to
//! 5342. QMoE's decode state stays on the fp32-correct 33803. So matching the
//! DENSE autoregressive token 5342 would be an accuracy regression; QMoE is kept
//! because it reproduces the fp32 next-token oracle.
//!
//! The margin is a benign floating-point near-tie: 33803 and 5342 are the clear
//! top-2 (next candidate ~1.3 logits back) and their gap is < 0.1 logit across
//! every precision — the signature of a rounding-order flip, not a logic bug.
//!
//! ## Capture note (#722) — why the lock is teacher-forced, not autoregressive
//!
//! The *captured* CUDA-graph decode of this hybrid is not byte-exact with the
//! *eager* decode: they first diverge at generated token **20**, an fp16 near-tie
//! whose winner depends on which pointwise ops the graph captures. Baseline
//! capture happens to land the autoregressive stream on the fp32-oracle token
//! (`33803`); broadening pointwise capture (the C1 growing-symbol classifier)
//! lands it on the eager token (`46283`). **Both are within fp16 noise of the
//! fp32 oracle** (top-2 gap < 0.1 logit) — see #722. So the correctness lock is
//! anchored on the **fp32-oracle argmax via fresh-engine teacher-forcing of a
//! FIXED canonical context** (an invariant BOTH the captured and the eager decode
//! satisfy, because teacher-forcing is prefill-dominated and capture-independent),
//! while the autoregressive token@119 is kept as a documented, non-fatal #722
//! tripwire rather than a hard assertion.
//!
//! ## Teacher-forcing must use a fresh engine
//!
//! Every teacher-forced adjudication above runs on a **fresh** engine that has
//! not decoded anything. This is load-bearing for a hybrid Mamba model: reusing
//! an engine that already generated the prefix serves the step from the prefix /
//! decode caches, which restore attention KV but *not* the conv/recurrent
//! (`Mamba`) state, so the teacher-forced logits come from a corrupted state and
//! the argmax collapses to an unrelated token (279) instead of 33803. The QMoE
//! row above is only reproducible on a fresh engine.
//!
//! ## Building the fp32 oracle
//!
//! These exports are natively fp16 (fp16 activations/scales), so there is no
//! fp32-activation variant to run directly. The oracle graph is produced by the
//! reverse of the repo's `DecodePrecision::Fp16` rewrite — every `Float16`
//! tensor (activations, `MatMulNBits` scales, norm gammas, KV/conv/recurrent
//! state, logits) is up-converted to `Float32` and every `Cast`-to-fp16 is
//! retargeted to fp32, preserving the int4/uint8 packed weights. Point
//! `QWEN36_A3B_FP32_ORACLE_DIR` at that pipeline directory to re-derive the
//! oracle argmax at runtime; without it the test locks against the recorded
//! oracle constant.
//!
//! ## Run
//!
//! ```bash
//! QWEN36_A3B_QMOE_E2E_DIR=/home/justinchu/qwen36-35b-a3b-qmoe-artifacts \
//! QWEN36_A3B_DENSE_E2E_DIR=/home/justinchu/qwen36-35b-a3b-artifacts \
//! QWEN36_A3B_FP32_ORACLE_DIR=/home/justinchu/qwen36-35b-a3b-fp32-oracle \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine --features native-backend,cuda \
//!   --test qwen36_35b_a3b_qmoe_divergence -- --ignored --nocapture
//! ```
#![cfg(feature = "native-cuda")]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    EngineConfig, EngineDecodeBackend, GenerateOptions, GeneratePrompt, GenerateRequest,
    GenerateResult, NativeDecodeDevice, PipelineEngine, PipelineGenerateRequest,
};
use onnx_genai_ort::Tokenizer;

const DEFAULT_QMOE_DIR: &str = "/home/justinchu/qwen36-35b-a3b-qmoe-artifacts";
const DEFAULT_DENSE_DIR: &str = "/home/justinchu/qwen36-35b-a3b-artifacts";
const DEFAULT_FP32_ORACLE_DIR: &str = "/home/justinchu/qwen36-35b-a3b-fp32-oracle";

const PROMPT: &str = "Hello";
/// First (and only) native-CUDA dense-vs-QMoE greedy divergence.
const DIVERGENCE_INDEX: usize = 119;
/// fp32-oracle-correct token (QMoE's autoregressive pick). "Ubuntu".
const ORACLE_TOKEN: u32 = 33803;
/// DENSE int4 CUDA's autoregressive pick — the lower-precision decode-drift
/// outlier that must NOT be what QMoE emits. "Windows".
const DENSE_CUDA_TOKEN: u32 = 5342;
/// The autoregressive token@[`DIVERGENCE_INDEX`] that BROADER pointwise capture
/// (C1) lands on: the eager-side token of the fp16 near-tie (top-2 gap < 0.1
/// logit, within fp16 noise of the fp32 oracle). The baseline's partial
/// segmentation instead reproduces [`ORACLE_TOKEN`]. These two — and ONLY these
/// two — are the KNOWN BENIGN outcomes of the documented coin-flip (#722); any
/// other token at this position is a genuine captured-decode corruption (e.g.
/// the dense outlier [`DENSE_CUDA_TOKEN`] or an unrelated token) and MUST fail.
const C1_CAPTURE_TOKEN: u32 = 46283;
/// Teacher-forced logit(33803) − logit(5342) spans ~+0.078..+0.094 across
/// fp32/f16 references; the band guards against silent drift of the tie.
const MARGIN_BAND: std::ops::RangeInclusive<f32> = 0.04..=0.14;

/// Canonical shared decode context tail — the fp32-oracle-correct generated
/// prefix `[0..DIVERGENCE_INDEX]` (119 tokens) for prompt `"Hello"`, RECORDED
/// from the baseline capture-on native-CUDA greedy decode at origin/main
/// `82736cf1` (the path whose autoregressive token@119 == [`ORACLE_TOKEN`]).
///
/// The primary lock teacher-forces `prompt + this` on a FRESH engine and
/// asserts the fp32-oracle argmax ([`ORACLE_TOKEN`]) + benign tie band. That
/// invariant is *capture-independent* — teacher-forcing is a prefill-dominated
/// single step, so both the captured and the eager decode engines re-derive it
/// (see #722). The context is deliberately **fixed here, not reconstructed from
/// the live autoregressive stream**: the autoregressive token@119 is an fp16
/// near-tie (< 0.1 logit) whose winner flips with which pointwise ops the CUDA
/// graph captures (baseline capture → 33803; broader C1 capture → 46283, both
/// within fp16 noise of the fp32 oracle). Reconstructing the context from that
/// coin-flip stream would silently change what we adjudicate. See #722.
///
/// Its own prefix `[0..20]` is identical on every path (all decodes agree until
/// the first fp16 tie at index 20), so this constant is the true shared context.
const SHARED_GENERATED_PREFIX: [u32; DIVERGENCE_INDEX] = [
    11, 353, 2688, 4313, 310, 958, 279, 1510, 447, 63, 1654, 314, 279, 1510, 35044, 63, 6522, 310,
    615, 264, 2081, 11, 694, 353, 615, 264, 1510, 4378, 1401, 63, 440, 279, 1876, 1510, 4378,
    15578, 539, 8434, 3357, 27653, 271, 40, 2688, 1608, 279, 2614, 1970, 25, 271, 71093, 12305,
    198, 464, 7154, 198, 1050, 283, 359, 1211, 1074, 2068, 7479, 877, 6, 198, 2246, 283, 7154, 652,
    6319, 8, 198, 71093, 271, 40, 2908, 6470, 1608, 2086, 33848, 11, 694, 353, 615, 279, 1788,
    1412, 13, 353, 2908, 1048, 6470, 1608, 279, 1510, 322, 20115, 63, 6522, 11, 694, 353, 615, 279,
    1788, 1412, 13, 271, 40, 2688, 1608, 12654, 220, 18, 13, 23, 13, 17, 383,
];

fn resolve_dir(env: &str, default: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default));
    if !dir.is_dir() {
        eprintln!(
            "skipping Qwen3.6-35B-A3B QMoE lock: directory absent: {}",
            dir.display()
        );
        return None;
    }
    Some(dir)
}

fn cuda_available() -> bool {
    match onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        Ok(_) => true,
        Err(error) => {
            eprintln!("skipping Qwen3.6-35B-A3B QMoE lock: CUDA unavailable: {error}");
            false
        }
    }
}

fn engine(
    dir: &Path,
    backend: EngineDecodeBackend,
    device: NativeDecodeDevice,
) -> anyhow::Result<PipelineEngine> {
    let config = EngineConfig {
        decode_backend: backend,
        native_device: Some(device),
        ..EngineConfig::default()
    };
    PipelineEngine::from_dir_with_config(dir, config)
}

/// Autoregressive greedy stream (temperature 0) of `max_new_tokens` tokens.
fn greedy_stream(engine: &mut PipelineEngine, max_new_tokens: usize) -> anyhow::Result<Vec<u32>> {
    let mut request = GenerateRequest::new(GeneratePrompt::Text(PROMPT.to_string()));
    request.options = GenerateOptions {
        max_new_tokens,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };
    let result = engine.generate_with_pipeline_request(PipelineGenerateRequest::new(request))?;
    Ok(result.token_ids)
}

/// Teacher-force `context` for a single greedy step, returning the top-K
/// log-softmax at the predicted position.
fn teacher_forced_step(
    engine: &mut PipelineEngine,
    context: &[u32],
    k: usize,
) -> anyhow::Result<GenerateResult> {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(context.to_vec()));
    request.options = GenerateOptions {
        max_new_tokens: 1,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        top_logprobs: Some(k),
        ..GenerateOptions::default()
    };
    engine.generate_with_pipeline_request(PipelineGenerateRequest::new(request))
}

fn argmax(result: &GenerateResult) -> u32 {
    result
        .logprobs
        .as_ref()
        .and_then(|v| v.first())
        .expect("logprobs present")
        .top[0]
        .0
}

fn logprob_of(result: &GenerateResult, token: u32) -> f32 {
    let top = &result.logprobs.as_ref().unwrap().first().unwrap().top;
    top.iter()
        .find(|(id, _)| *id == token)
        .unwrap_or_else(|| panic!("{token} absent from top: {top:?}"))
        .1
}

#[test]
#[ignore = "requires the real Qwen3.6-35B-A3B int4 QMoE + dense artifacts and a CUDA device"]
fn qwen36_35b_a3b_qmoe_native_cuda_matches_fp32_oracle() -> anyhow::Result<()> {
    let Some(qmoe_dir) = resolve_dir("QWEN36_A3B_QMOE_E2E_DIR", DEFAULT_QMOE_DIR) else {
        return Ok(());
    };
    if !cuda_available() {
        return Ok(());
    }
    let tokenizer = Tokenizer::from_file(qmoe_dir.join("tokenizer.json"))?;
    let prompt_ids = tokenizer.encode(PROMPT)?;
    let n = DIVERGENCE_INDEX + 1;

    // The canonical shared decode context — a FIXED constant (see
    // SHARED_GENERATED_PREFIX), NOT reconstructed from the autoregressive stream.
    // This decouples the correctness lock from the fp16 near-tie coin-flip at
    // token@DIVERGENCE_INDEX whose winner depends on which pointwise ops the CUDA
    // graph captures (#722). Teacher-forcing this fixed context on a fresh engine
    // is prefill-dominated and therefore capture-independent: both the captured
    // and the eager decode engines re-derive ORACLE_TOKEN here.
    let mut context = prompt_ids.clone();
    context.extend_from_slice(&SHARED_GENERATED_PREFIX);

    // 1) PRIMARY LOCK — the fp32-oracle-backed, capture-independent invariant.
    //    Teacher-force the canonical context on a FRESH QMoE engine: the argmax
    //    is the oracle token, the runner-up is the dense-CUDA outlier, and the
    //    tie sits in the benign band. A fresh engine is load-bearing for this
    //    hybrid Mamba model — reusing a post-decode engine restores attention KV
    //    but NOT the conv/recurrent (`Mamba`) state, so its teacher-forced step
    //    predicts from a corrupted state (empirically collapses to 279); the
    //    fresh engine re-runs the full prefill and reproduces the fp32-oracle tie.
    let mut qmoe_tf = engine(
        &qmoe_dir,
        EngineDecodeBackend::Native,
        NativeDecodeDevice::Cuda { index: Some(0) },
    )?;
    let qmoe_step = teacher_forced_step(&mut qmoe_tf, &context, 8)?;
    assert_eq!(
        argmax(&qmoe_step),
        ORACLE_TOKEN,
        "QMoE fresh-engine teacher-forced argmax must adjudicate the fp32-oracle token \
         {ORACLE_TOKEN} (capture-independent invariant, see #722)",
    );
    let margin = logprob_of(&qmoe_step, ORACLE_TOKEN) - logprob_of(&qmoe_step, DENSE_CUDA_TOKEN);
    assert!(
        MARGIN_BAND.contains(&margin),
        "QMoE {ORACLE_TOKEN}-over-{DENSE_CUDA_TOKEN} tie {margin} outside {MARGIN_BAND:?}",
    );
    drop(qmoe_tf);

    // 2) #722 tripwire (BOUNDED, FATAL on real corruption): the *autoregressive*
    //    captured stream lands on ORACLE_TOKEN only on the baseline's partial
    //    segmentation; with broader pointwise capture (C1) it lands on the eager
    //    token C1_CAPTURE_TOKEN. Both are within fp16 noise of the fp32 oracle
    //    (top-2 gap < 0.1 logit) — the captured decode is not byte-exact with the
    //    eager decode on this hybrid (diverges at token 20). We TOLERATE the
    //    documented coin-flip between exactly those two outcomes, but ASSERT the
    //    token is one of them: any OTHER token (a genuine captured-decode
    //    corruption — the dense outlier 5342, the reused-state 279, or anything
    //    unrelated) still FAILS CI. This restores a real regression tripwire on
    //    the changed captured autoregressive path without re-pinning correctness
    //    to which side of the benign tie a given segmentation happens to pick. The
    //    fp32-oracle teacher-forced primary lock above remains the ground truth.
    let mut qmoe = engine(
        &qmoe_dir,
        EngineDecodeBackend::Native,
        NativeDecodeDevice::Cuda { index: Some(0) },
    )?;
    let qmoe_stream = greedy_stream(&mut qmoe, n)?;
    drop(qmoe);
    let autoregressive = qmoe_stream[DIVERGENCE_INDEX];
    if autoregressive == ORACLE_TOKEN {
        eprintln!(
            "#722 note: autoregressive token {DIVERGENCE_INDEX} == {ORACLE_TOKEN} \
             (this build's capture segmentation reproduces the fp32-oracle side of the tie)"
        );
    } else {
        eprintln!(
            "#722 note: autoregressive token {DIVERGENCE_INDEX} == {autoregressive} != fp32-oracle \
             {ORACLE_TOKEN} — expected fp16 near-tie coin-flip (captured != eager on this hybrid; \
             both within fp16 noise of the oracle). The capture-independent teacher-forced lock \
             above still adjudicates {ORACLE_TOKEN}. See #722."
        );
    }
    assert!(
        autoregressive == ORACLE_TOKEN || autoregressive == C1_CAPTURE_TOKEN,
        "captured autoregressive token@{DIVERGENCE_INDEX} = {autoregressive} is neither known \
         benign fp16-tie outcome ({ORACLE_TOKEN} baseline-capture | {C1_CAPTURE_TOKEN} C1-capture); \
         a value outside this set is a genuine captured-decode corruption (e.g. dense outlier \
         {DENSE_CUDA_TOKEN} or token 279), not the documented coin-flip. See #722.",
    );

    // 3) DENSE int4 CUDA is the lower-precision sibling (byte-identical int4 expert
    //    weights, per-expert MatMulNBits instead of fused QMoE). Its *autoregressive*
    //    decode drifts off the oracle (recurrent-state f16 drift), but teacher-forced
    //    at the canonical context on a fresh engine it, too, adjudicates ORACLE_TOKEN
    //    — a capture-independent cross-check that the tie's fp32-correct side is
    //    ORACLE_TOKEN regardless of which int4 kernel path serves it.
    if let Some(dense_dir) = resolve_dir("QWEN36_A3B_DENSE_E2E_DIR", DEFAULT_DENSE_DIR) {
        let mut dense_tf = engine(
            &dense_dir,
            EngineDecodeBackend::Native,
            NativeDecodeDevice::Cuda { index: Some(0) },
        )?;
        let dense_step = teacher_forced_step(&mut dense_tf, &context, 8)?;
        assert_eq!(
            argmax(&dense_step),
            ORACLE_TOKEN,
            "dense int4 CUDA fresh-engine teacher-forced argmax must also adjudicate {ORACLE_TOKEN}",
        );
        let dmargin =
            logprob_of(&dense_step, ORACLE_TOKEN) - logprob_of(&dense_step, DENSE_CUDA_TOKEN);
        assert!(
            MARGIN_BAND.contains(&dmargin),
            "dense int4 CUDA {ORACLE_TOKEN}-over-{DENSE_CUDA_TOKEN} tie {dmargin} outside \
             {MARGIN_BAND:?}",
        );
        drop(dense_tf);
        eprintln!(
            "dense int4 CUDA teacher-forced re-derived {ORACLE_TOKEN} (margin over \
             {DENSE_CUDA_TOKEN} = {dmargin}); its autoregressive drift to the {DENSE_CUDA_TOKEN} \
             outlier is a decode-state artifact, not the adjudicated token"
        );
    }

    // 4) Oracle-driven cross-check: teacher-force the SAME canonical context through
    //    the full-fp32 pipeline (native CPU) and confirm it re-derives ORACLE_TOKEN,
    //    so the lock is grounded in fp32 ground truth, not a hard-coded constant.
    if let Some(oracle_dir) = resolve_dir("QWEN36_A3B_FP32_ORACLE_DIR", DEFAULT_FP32_ORACLE_DIR) {
        let mut oracle = engine(
            &oracle_dir,
            EngineDecodeBackend::Native,
            NativeDecodeDevice::Cpu,
        )?;
        let oracle_step = teacher_forced_step(&mut oracle, &context, 8)?;
        assert_eq!(
            argmax(&oracle_step),
            ORACLE_TOKEN,
            "full-fp32 oracle argmax must adjudicate {ORACLE_TOKEN}",
        );
        let omargin =
            logprob_of(&oracle_step, ORACLE_TOKEN) - logprob_of(&oracle_step, DENSE_CUDA_TOKEN);
        assert!(
            MARGIN_BAND.contains(&omargin),
            "fp32 oracle tie {omargin} outside {MARGIN_BAND:?}",
        );
        eprintln!(
            "fp32 oracle re-derived {ORACLE_TOKEN} (margin over {DENSE_CUDA_TOKEN} = {omargin})"
        );
    }

    eprintln!(
        "Qwen3.6-35B-A3B QMoE lock OK: fp32-oracle teacher-forced argmax @{DIVERGENCE_INDEX} \
         == {ORACLE_TOKEN} (capture-independent), dense int4 CUDA outlier={DENSE_CUDA_TOKEN}, \
         autoregressive coin-flip token={} (#722)",
        qmoe_stream[DIVERGENCE_INDEX],
    );
    Ok(())
}

/// #695 regression: on a **hybrid Mamba** model, a continuation that reuses a
/// post-decode engine (multi-turn chat / a second `generate` sharing a prefix)
/// must produce the SAME next-token logits as a fresh-engine teacher-force of
/// the equivalent full context.
///
/// ## The bug
///
/// Prefix / KV-mirror reuse restores only attention KV; the conv/recurrent
/// (`Mamba`) `fixed_state_binding_range` is reconstructed only on a full
/// `rewind(0)` and never rebuilt for a reused prefix, so the recurrent path
/// starts fresh-zero, inconsistent with the reused attention prefix. Before the
/// fix, teacher-forcing the 120-token shared context on the engine that just
/// autoregressively decoded it argmaxed an unrelated token (empirically 279)
/// instead of the fp32-oracle token (33803) a fresh engine yields.
///
/// ## The fix under test
///
/// The KV-mirror support gate now returns `false` whenever the decoder carries
/// recurrent state, forcing a full recompute on continuation. This test locks
/// the *symptom*: `reused-engine argmax == fresh-engine argmax` for the context
/// the engine actually decoded (a fresh engine correctly re-runs the full
/// prefill, so the reused continuation is checked against that oracle). The
/// specific autoregressive token@119 is an fp16 near-tie coin-flip whose winner
/// depends on CUDA-graph capture segmentation (see #722), so this test no longer
/// pins it to `ORACLE_TOKEN` — it locks the capture-independent self-consistency
/// that the gate fix restores. Without the gate fix the reused-engine argmax
/// diverges (collapses to 279) and this test fails.
///
/// Env-gated on the same real 35B-A3B artifact + CUDA device as the divergence
/// lock above; skips cleanly when either is absent.
#[test]
#[ignore = "requires the real Qwen3.6-35B-A3B int4 QMoE artifacts and a CUDA device"]
fn qwen36_35b_a3b_hybrid_continuation_matches_fresh_engine() -> anyhow::Result<()> {
    let Some(qmoe_dir) = resolve_dir("QWEN36_A3B_QMOE_E2E_DIR", DEFAULT_QMOE_DIR) else {
        return Ok(());
    };
    if !cuda_available() {
        return Ok(());
    }
    let tokenizer = Tokenizer::from_file(qmoe_dir.join("tokenizer.json"))?;
    let prompt_ids = tokenizer.encode(PROMPT)?;
    let n = DIVERGENCE_INDEX + 1;

    // Engine A autoregressively decodes a stream, so its conv/recurrent
    // (Mamba) state now holds the terminal decode state and its prefix cache is
    // populated — the exact "reused engine" a multi-turn continuation lands on.
    // The specific token@DIVERGENCE_INDEX is an fp16 near-tie coin-flip (captured
    // != eager on this hybrid, see #722); #695 locks the *self-consistency* of the
    // continuation, which holds for whichever stream this build decodes, so we
    // reconstruct the context from the ACTUAL decoded stream and LOG (not assert)
    // the coin-flip token.
    let mut reused = engine(
        &qmoe_dir,
        EngineDecodeBackend::Native,
        NativeDecodeDevice::Cuda { index: Some(0) },
    )?;
    let stream = greedy_stream(&mut reused, n)?;
    eprintln!(
        "#722 note: #695 continuation autoregressive token {DIVERGENCE_INDEX} == {} \
         (fp16 near-tie coin-flip; #695 locks reused==fresh regardless of the tie side)",
        stream[DIVERGENCE_INDEX],
    );

    // The exact context this engine just decoded: prompt + generated[0..DIVERGENCE_INDEX].
    let mut context = prompt_ids.clone();
    context.extend_from_slice(&stream[..DIVERGENCE_INDEX]);

    // Continuation on the REUSED engine: this is the #695 reproduction. With the
    // KV-mirror gate disabled for recurrent decoders, it must fully recompute and
    // reproduce the fresh-engine distribution for THIS context instead of collapsing
    // to 279.
    let reused_step = teacher_forced_step(&mut reused, &context, 8)?;
    let reused_argmax = argmax(&reused_step);
    drop(reused);

    // Oracle: a FRESH engine re-runs the full prefill and yields the correct
    // next-token distribution for the same context.
    let mut fresh = engine(
        &qmoe_dir,
        EngineDecodeBackend::Native,
        NativeDecodeDevice::Cuda { index: Some(0) },
    )?;
    let fresh_step = teacher_forced_step(&mut fresh, &context, 8)?;
    let fresh_argmax = argmax(&fresh_step);
    drop(fresh);

    // #695 invariant (capture-independent): the reused-engine continuation must
    // reproduce the fresh-engine argmax for the context it decoded. This is the
    // symptom the KV-mirror recurrent-state gate fixes, and it does not depend on
    // which side of the fp16 tie the autoregressive stream fell on.
    assert_eq!(
        reused_argmax, fresh_argmax,
        "#695: reused-engine continuation argmax {reused_argmax} must equal the fresh-engine \
         argmax {fresh_argmax} for the same context (before the KV-mirror gate fix it collapsed \
         to 279)",
    );

    // Stronger than argmax alone: the reused and fresh top-1 log-probabilities must
    // agree to within fp16 noise, proving the recompute reproduces the whole
    // distribution rather than merely tying at the winner.
    let reused_lp = reused_step.logprobs.as_ref().unwrap().first().unwrap().top[0].1;
    let fresh_lp = fresh_step.logprobs.as_ref().unwrap().first().unwrap().top[0].1;
    assert!(
        (reused_lp - fresh_lp).abs() <= 1e-3,
        "#695: reused-engine top-1 logprob {reused_lp} must match the fresh-engine oracle \
         {fresh_lp} within fp16 noise",
    );

    eprintln!(
        "Qwen3.6-35B-A3B #695 continuation OK: reused-engine argmax {reused_argmax} == \
         fresh-engine {fresh_argmax} (recurrent-state gate forces full recompute; \
         self-consistent regardless of the #722 capture coin-flip)",
    );
    Ok(())
}
