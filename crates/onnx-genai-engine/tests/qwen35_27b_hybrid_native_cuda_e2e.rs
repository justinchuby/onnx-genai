//! Native-CUDA end-to-end decode + fp32-oracle parity lock for the **Qwen3.6/3.5
//! -27B hybrid** (gated linear-attention `LinearAttention` + causal short-conv
//! `CausalConvWithState` + periodic full-attention `GroupQueryAttention` + int4
//! `MatMulNBits`) single-file export — the 27B counterpart of the passing
//! `qwen35_0_8b_hybrid_native_cuda_e2e` capstone and the last model the
//! ORT-vs-native benchmark flagged as an unsupported native load.
//!
//! ## What this locks
//!
//! The 27B artifact is a *correct* hybrid graph (64 decoder layers = 48
//! linear-attention/GDN layers carrying `conv_state`/`recurrent_state` + 16
//! periodic full-attention layers carrying dense `key`/`value`), but ships a
//! **thin** `inference_metadata.yaml` that declares only
//! `grouped_query_attention` and NO `io` port contract. Without that contract
//! the Resource Governor cannot derive the per-layer KV page byte geometry (only
//! the 16 full-attention layers hold dense KV) and native load fails with
//! `per-layer KV page geometry is unknown`. The loader now auto-derives the
//! decoder `io` port contract from the ONNX graph's own port inventory for
//! recurrent-hybrid decoders (attribute/shape-driven, never model-name-gated;
//! see `maybe_fill_hybrid_io_from_graph` in `engine/load.rs`), so the package
//! loads and decodes natively.
//!
//! ## Correctness gate — teacher-forced fp32 oracle
//!
//! Autoregressive fp16 greedy is a near-tie coin-flip at some positions (benign,
//! per the 35B QMoE lock), so the correctness gate is the **teacher-forced
//! next-token argmax** adjudicated by an independent **full-fp32 oracle** — the
//! exact fp16->fp32 up-conversion recipe used for the 35B QMoE lock
//! (`f16_to_f32.py`: every fp16 activation/scale/norm -> fp32, int4
//! `MatMulNBits` packed weights preserved). The fp16 native-CUDA decoder must
//! select the SAME next token as the fp32 oracle on the fixed prompt.
//!
//! ## Run
//!
//! ```bash
//! QWEN35_27B_DIR=/home/justinchu/mary-models/qwen3.6-27b-int4-cuda \
//! QWEN35_27B_FP32_ORACLE_DIR=/home/justinchu/qwen35-27b-fp32-oracle \
//! ONNX_GENAI_CUDA_GRAPH=1 ONNX_GENAI_CUDA_KV_MAX_LEN=4096 \
//! LD_LIBRARY_PATH=/home/tlwu/cudnn9.19_cuda13/lib CUDA_VISIBLE_DEVICES=2 \
//! cargo test -q -p onnx-genai-engine --features "cuda native-backend" \
//!   --test qwen35_27b_hybrid_native_cuda_e2e -- --ignored --nocapture
//! ```
#![cfg(feature = "native-cuda")]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GeneratePrompt, GenerateRequest,
    GenerateResult, NativeDecodeDevice,
};

const DEFAULT_MODEL_DIR: &str = "/home/justinchu/mary-models/qwen3.6-27b-int4-cuda";
const DEFAULT_ORACLE_DIR: &str = "/home/justinchu/qwen35-27b-fp32-oracle";

const PROMPT: &str = "The capital of France is";
/// The fp32-oracle-correct, semantically-correct next token for `PROMPT`:
/// " Paris" (id 11751). Recorded from the native-CUDA fp16 greedy decode and
/// re-derived by the fp32 oracle at runtime when the oracle artifact is present.
const EXPECTED_FIRST_TOKEN: u32 = 11751;
const NATIVE_TOKENS: usize = 8;

fn resolve_dir(env: &str, default: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default));
    if !dir.join("model.onnx").is_file() {
        eprintln!(
            "skipping Qwen3.5-27B hybrid native-CUDA lock: {} has no model.onnx",
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
            eprintln!("skipping Qwen3.5-27B hybrid native-CUDA lock: CUDA unavailable: {error}");
            false
        }
    }
}

fn cuda_engine(dir: &Path) -> anyhow::Result<Engine> {
    Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            ..EngineConfig::default()
        },
    )
}

fn cpu_engine(dir: &Path) -> anyhow::Result<Engine> {
    Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cpu),
            ..EngineConfig::default()
        },
    )
}

fn request(max_new_tokens: usize, top_logprobs: Option<usize>) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::Text(PROMPT.to_string()));
    request.options = GenerateOptions {
        max_new_tokens,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        top_logprobs,
        ..GenerateOptions::default()
    };
    request
}

fn top(result: &GenerateResult) -> Vec<(u32, f32)> {
    result
        .logprobs
        .as_ref()
        .and_then(|logprobs| logprobs.first())
        .expect("teacher-forced step must return logprobs")
        .top
        .clone()
}

#[test]
#[ignore = "requires the real Qwen3.5/3.6-27B hybrid int4 artifact (QWEN35_27B_DIR) and a CUDA device"]
fn qwen35_27b_hybrid_native_cuda_runs_and_matches_fp32_oracle() -> anyhow::Result<()> {
    let Some(dir) = resolve_dir("QWEN35_27B_DIR", DEFAULT_MODEL_DIR) else {
        return Ok(());
    };
    if !cuda_available() {
        return Ok(());
    }

    // (1) Load-unblock proof: the thin-metadata hybrid graph places on the
    // native backend (no ORT fallback) and autoregressively decodes coherent
    // tokens. The graph-derived `io` contract is what makes this load at all.
    let native_tokens = cuda_engine(&dir)?
        .generate(request(NATIVE_TOKENS, None))?
        .token_ids;
    eprintln!("native-CUDA fp16 greedy: {native_tokens:?}");
    assert_eq!(
        native_tokens.len(),
        NATIVE_TOKENS,
        "native-CUDA hybrid decode did not emit the requested token count"
    );
    assert_eq!(
        native_tokens.first().copied(),
        Some(EXPECTED_FIRST_TOKEN),
        "native-CUDA hybrid decode of {PROMPT:?} must begin with the coherent next token \
         (id {EXPECTED_FIRST_TOKEN}, \" Paris\")"
    );

    // (2) Teacher-forced next-token argmax on a FRESH native-CUDA fp16 engine. A
    // fresh engine is load-bearing for a hybrid recurrent model: reusing a
    // decoded engine serves the step from caches that restore attention KV but
    // not conv/recurrent state, corrupting the logits (see the 35B QMoE lock).
    let native_result = cuda_engine(&dir)?.generate(request(1, Some(8)))?;
    let native_top = top(&native_result);
    eprintln!("native-CUDA fp16 teacher-forced top: {native_top:?}");
    assert_eq!(
        native_top[0].0, EXPECTED_FIRST_TOKEN,
        "native-CUDA fp16 teacher-forced argmax must be the coherent next token"
    );

    // (3) fp32 oracle adjudication. The oracle is the independent fp16->fp32
    // up-conversion (int4 weights preserved) run on the native CPU backend at
    // full fp32 precision; the fp16 native-CUDA decoder MUST agree byte-for-byte.
    let Some(oracle_dir) = resolve_dir("QWEN35_27B_FP32_ORACLE_DIR", DEFAULT_ORACLE_DIR) else {
        eprintln!(
            "fp32 oracle artifact absent; native-CUDA coherence locked, oracle parity auto-\
             activates once QWEN35_27B_FP32_ORACLE_DIR is present"
        );
        return Ok(());
    };
    let oracle_result = cpu_engine(&oracle_dir)?.generate(request(1, Some(8)))?;
    let oracle_top = top(&oracle_result);
    let oracle_argmax = oracle_top[0].0;
    let runner_up = oracle_top
        .get(1)
        .map(|entry| entry.1)
        .unwrap_or(f32::NEG_INFINITY);
    let oracle_margin = oracle_top[0].1 - runner_up;
    eprintln!("fp32 oracle teacher-forced top: {oracle_top:?} (top-1 margin {oracle_margin:.4})");

    assert_eq!(
        oracle_argmax, EXPECTED_FIRST_TOKEN,
        "fp32 oracle argmax for {PROMPT:?} must be the coherent next token (id \
         {EXPECTED_FIRST_TOKEN}, \" Paris\")"
    );
    // Byte-exact teacher-forced parity: the fp16 native-CUDA decoder selects the
    // SAME next token as the fp32 oracle.
    assert_eq!(
        native_top[0].0, oracle_argmax,
        "native-CUDA fp16 next-token must match the fp32 oracle argmax"
    );
    Ok(())
}
