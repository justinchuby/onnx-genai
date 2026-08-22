//! End-to-end text-decode regression lock for the **Qwen3.5-0.8B hybrid**
//! (Mamba/linear-attention + sliding-window attention) split package
//! (issues #67, #384).
//!
//! This is the integration proof that the per-op CUDA coverage work
//! (CausalConvWithState #480, LinearAttention #484, RoPE-contrib + Bool NonZero
//! #525) composes into a **working model that actually decodes**, not just
//! isolated green kernels.
//!
//! The Foundry export is a three-ONNX split package (`vision.onnx`,
//! `embedding.onnx`, `text.onnx`) whose declared image preprocessing uses
//! Qwen-style `smart_resize`, which has no lossless runtime encoding. Before the
//! text-only pipeline synthesis landed, the whole package was refused at
//! metadata-synthesis time (the unrepresentable image preprocessing aborted the
//! entire pipeline spec), so text decode could never run. The loader now falls
//! back to a **text-only decode pipeline** (embedding → decoder, no vision, no
//! image preprocessing, rank-3 `linear_increment` mrope positions) for a split
//! VLM package whose image path is unusable — driven purely by the package's
//! declared modality shape, never a model name.
//!
//! The reference here is an ORT-driven decode of the same synthesized pipeline
//! (ORT places the standard-attention layers on its EP and falls back to CPU for
//! the `com.microsoft` hybrid ops it does not implement). Greedy decode is
//! deterministic, so the completion is locked exactly; the assertion that the
//! output contains the correct fact (`Paris`) is an independent coherence oracle
//! that a positions/state/dataflow regression would break.
//!
//! Native-CUDA decoder parity for this model is a documented follow-up: the
//! native step driver (`native_decode/{cuda,cpu}.rs`) currently emits rank-2
//! `position_ids` and must construct the rank-3 mrope coordinates this hybrid
//! decoder declares before native CUDA can drive it. See
//! `.squad/decisions/inbox/cohaagen-hybrid-loader.md`.
#![cfg(feature = "native-cuda")]

use std::path::PathBuf;

use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};

const DEFAULT_MODEL_DIR: &str =
    "/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-0.8b-generic-cpu-2/v2";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 16;

/// The validated greedy completion of the Qwen3.5-0.8B hybrid text pipeline.
///
/// ` Paris, and the capital of Germany is Berlin.\nThe capital of France is`
const EXPECTED_TOKENS: [u32; MAX_NEW_TOKENS] = [
    11751, 11, 321, 279, 6511, 314, 9564, 369, 19241, 13, 198, 760, 6511, 314, 9338, 369,
];

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("QWEN35_0_8B_HYBRID_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
    let required = [
        "genai_config.json",
        "embedding.onnx",
        "text.onnx",
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
            "skipping Qwen3.5-0.8B hybrid text-decode regression: {} is missing {missing:?}",
            dir.display()
        );
        None
    }
}

#[test]
fn qwen35_0_8b_hybrid_text_decode_is_coherent_and_locked() -> anyhow::Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };

    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let mut request = GenerateRequest::new(GeneratePrompt::Text(PROMPT.to_string()));
    request.options = GenerateOptions {
        max_new_tokens: MAX_NEW_TOKENS,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };
    let result = engine.generate_with_pipeline_request(
        onnx_genai_engine::pipeline::PipelineGenerateRequest::new(request),
    )?;

    // Coherence oracle: the hybrid decode graph must still produce the correct
    // fact. A positions / loop-state / dataflow regression would corrupt this.
    assert!(
        result.text.contains("Paris"),
        "Qwen3.5-0.8B hybrid text decode lost its coherent completion: {:?}",
        result.text
    );
    // Exact greedy lock: guards the whole loader-unblock + text-only synthesis +
    // symbolic-batch loop-state init path against silent regression.
    assert_eq!(
        result.token_ids, EXPECTED_TOKENS,
        "Qwen3.5-0.8B hybrid greedy stream drifted from the validated anchor: {:?}",
        result.text
    );
    eprintln!(
        "Qwen3.5-0.8B hybrid text-decode lock OK: text={:?}",
        result.text
    );
    Ok(())
}
