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
#![cfg(all(feature = "native-backend", feature = "cuda"))]

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
/// Teacher-forced logit(33803) − logit(5342) spans ~+0.078..+0.094 across
/// fp32/f16 references; the band guards against silent drift of the tie.
const MARGIN_BAND: std::ops::RangeInclusive<f32> = 0.04..=0.14;

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

    // 1) QMoE autoregressive greedy: our decode must land on the oracle token.
    let mut qmoe = engine(
        &qmoe_dir,
        EngineDecodeBackend::Native,
        NativeDecodeDevice::Cuda { index: Some(0) },
    )?;
    let qmoe_stream = greedy_stream(&mut qmoe, n)?;
    assert_eq!(
        qmoe_stream[DIVERGENCE_INDEX], ORACLE_TOKEN,
        "QMoE autoregressive token {DIVERGENCE_INDEX} regressed from the fp32-oracle token \
         {ORACLE_TOKEN} to {}",
        qmoe_stream[DIVERGENCE_INDEX],
    );

    // Reconstruct the exact shared context: prompt + generated[0..DIVERGENCE_INDEX].
    let mut context = prompt_ids.clone();
    context.extend_from_slice(&qmoe_stream[..DIVERGENCE_INDEX]);

    // 2) QMoE teacher-forced at that context: argmax is the oracle token, the
    //    runner-up is the dense-CUDA outlier, and the tie is the benign band.
    let qmoe_step = teacher_forced_step(&mut qmoe, &context, 8)?;
    assert_eq!(
        argmax(&qmoe_step),
        ORACLE_TOKEN,
        "QMoE teacher-forced argmax"
    );
    let margin = logprob_of(&qmoe_step, ORACLE_TOKEN) - logprob_of(&qmoe_step, DENSE_CUDA_TOKEN);
    assert!(
        MARGIN_BAND.contains(&margin),
        "QMoE {ORACLE_TOKEN}-over-{DENSE_CUDA_TOKEN} tie {margin} outside {MARGIN_BAND:?}",
    );
    drop(qmoe);

    // 3) DENSE int4 CUDA autoregressive decode is the lower-precision outlier:
    //    its recurrent-state drift tips this tie to DENSE_CUDA_TOKEN. Confirm the
    //    divergence still reproduces (guards the reconstruction), when available.
    if let Some(dense_dir) = resolve_dir("QWEN36_A3B_DENSE_E2E_DIR", DEFAULT_DENSE_DIR) {
        let mut dense = engine(
            &dense_dir,
            EngineDecodeBackend::Native,
            NativeDecodeDevice::Cuda { index: Some(0) },
        )?;
        let dense_stream = greedy_stream(&mut dense, n)?;
        assert_eq!(
            dense_stream[..DIVERGENCE_INDEX],
            qmoe_stream[..DIVERGENCE_INDEX],
            "dense and QMoE must agree before the divergence index",
        );
        assert_eq!(
            dense_stream[DIVERGENCE_INDEX], DENSE_CUDA_TOKEN,
            "dense autoregressive divergence token changed",
        );
        assert_ne!(
            dense_stream[DIVERGENCE_INDEX], ORACLE_TOKEN,
            "dense int4 CUDA unexpectedly matched the fp32 oracle",
        );
    }

    // 4) Oracle-driven cross-check: teacher-force the SAME context through the
    //    full-fp32 pipeline (native CPU) and confirm it re-derives ORACLE_TOKEN,
    //    so the lock is not a hard-coded constant.
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
        "Qwen3.6-35B-A3B QMoE lock OK: index {DIVERGENCE_INDEX} QMoE={ORACLE_TOKEN} \
         (fp32-oracle-correct), dense int4 CUDA outlier={DENSE_CUDA_TOKEN}",
    );
    Ok(())
}
