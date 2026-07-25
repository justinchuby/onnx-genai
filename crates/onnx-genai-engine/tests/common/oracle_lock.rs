//! Shared fp32-oracle greedy-decode adjudication harness.
//!
//! A razor-thin `MatMulNBits` logit tie can make native CUDA and ORT CUDA pick
//! different argmax tokens at a single decode step. To decide which backend is
//! correct we consult an independent fp32-activation oracle: ORT CPU run on the
//! same int4/block-32 graph. For these DeepSeek-R1 exports every `MatMulNBits`
//! node ships with **no** `accuracy_level` attribute, which selects the fp32
//! (level-1) activation path on the ORT CPU kernel, so ORT CPU is the fp32
//! oracle without any graph rewrite. (Rewriting all nodes to an explicit
//! `accuracy_level=1` was verified to leave the oracle argmax unchanged.)
//!
//! Both the deepseek `"capital of France"` lock and the benchmark-prompt lock
//! reuse these helpers so the adjudication logic lives in exactly one place.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, GenerateResult,
    NativeDecodeDevice,
};

/// Resolve a model directory from `env`, falling back to `default`. Returns
/// `None` (with a skip message) when the directory is absent, so the ignored
/// real-model locks degrade to a no-op on machines without the artifact.
#[allow(dead_code)]
pub fn resolve_model_dir(env: &str, default: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default));
    if !dir.is_dir() {
        eprintln!(
            "skipping DeepSeek-R1 divergence lock: model directory absent: {}",
            dir.display()
        );
        return None;
    }
    Some(dir)
}

/// Whether a CUDA device is usable; prints a skip message and returns `false`
/// otherwise.
#[allow(dead_code)]
pub fn cuda_available() -> bool {
    match onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        Ok(_) => true,
        Err(error) => {
            eprintln!("skipping DeepSeek-R1 divergence lock: CUDA unavailable: {error}");
            false
        }
    }
}

/// Greedy-decode `max_new_tokens` from `prompt` on `backend`, requesting the
/// top-8 log-probabilities at every step so the caller can inspect the margin
/// at the divergent decision.
#[allow(dead_code)]
pub fn generate(
    dir: &Path,
    prompt: &str,
    max_new_tokens: usize,
    backend: EngineDecodeBackend,
    device: Option<NativeDecodeDevice>,
) -> anyhow::Result<GenerateResult> {
    let config = EngineConfig {
        decode_backend: backend,
        native_device: device,
        ..EngineConfig::default()
    };
    let mut engine = Engine::from_dir(dir, config)?;
    let mut request = GenerateRequest::new(GeneratePrompt::Text(prompt.to_string()));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request.options.top_logprobs = Some(8);
    engine.generate(request)
}

/// Assert `result` reproduced the fp32-oracle-correct greedy stream:
///
/// * the full token sequence equals `expected_tokens`,
/// * the argmax at `index` is `oracle_token`, and
/// * `oracle_token`'s log-prob margin over the ORT-CUDA `competitor` at `index`
///   lands inside `margin_band` (guards against silent numeric drift).
///
/// Returns the observed margin.
#[allow(dead_code)]
pub fn assert_oracle_argmax(
    label: &str,
    result: &GenerateResult,
    expected_tokens: &[u32],
    index: usize,
    oracle_token: u32,
    competitor: u32,
    margin_band: std::ops::RangeInclusive<f32>,
) -> f32 {
    assert_eq!(
        result.token_ids, expected_tokens,
        "{label} greedy stream drifted from the fp32-oracle-correct sequence"
    );
    let top = &result
        .logprobs
        .as_ref()
        .expect("top_logprobs requested but absent")[index]
        .top;
    assert_eq!(
        top.first().map(|(token, _)| *token),
        Some(oracle_token),
        "{label} argmax at index {index} is not the oracle token: {top:?}"
    );
    let logprob = |token: u32| {
        top.iter()
            .find(|(id, _)| *id == token)
            .map(|(_, value)| *value)
            .unwrap_or_else(|| {
                panic!("token {token} missing from {label} top-8 at index {index}: {top:?}")
            })
    };
    let margin = logprob(oracle_token) - logprob(competitor);
    assert!(
        margin_band.contains(&margin),
        "{label} must preserve the oracle's {oracle_token}-over-{competitor} margin; \
         got {margin} (expected within {margin_band:?})"
    );
    eprintln!(
        "{label}: index={index}, token={oracle_token}, logit margin over {competitor}={margin}"
    );
    margin
}
