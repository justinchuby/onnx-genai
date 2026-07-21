//! End-to-end test for the generic **every_step component executor** (VLM WP3)
//! over a multi-binding VLM whose embedding component emits MORE than the single
//! `inputs_embeds` tensor the `tiny-gemma4-vlm` fixture covers. It locks in two
//! properties a one-output `inputs_embeds` special case cannot express:
//!
//!   1. **Every declared component output is refreshed each step.** The
//!      `embedding` component (declared `every_step`, running-token port
//!      `io.token_input: input_ids`) emits TWO sequence-dependent tensors —
//!      `inputs_embeds` AND `aux` — both routed into the decoder via `dataflow`.
//!      The generic executor re-runs the whole component every step (over the
//!      full prompt at prefill, the single running token at decode), so both
//!      tensors track the running token. Had `aux` been left stale (frozen at its
//!      prompt value, as a single-output special case would), the generated ids
//!      would be `stale_aux` below instead of `fresh_both` — a different stream.
//!
//!   2. **No token / embeds exclusivity in pipeline execution.** The `decoder`
//!      consumes the RAW `input_ids` token stream *and* the routed
//!      `inputs_embeds` (+ `aux`) in the same forward pass. Nothing forbids a
//!      decoder from taking both; if `input_ids` were ignored the ids would be
//!      `ids_ignored` below.
//!
//! Everything is architecture-neutral DATA: the running-token port is declared
//! via `io.token_input` and the two outputs route through `dataflow` edges. The
//! engine never inspects tensor names to decide any of this.
//!
//! Closed form (H = 8, V = 8, E = identity; built + validated against ORT by
//! `scripts/build_tiny_vlm_multibinding.py`): with `A[t] = 0.6*e_{(t+3)%8}` and
//! the decoder's raw-token gate `G[t] = 0.6*e_{(t+3)%8}` sharing slot `(t+3)`,
//! `combined = inputs_embeds + aux + G[input_ids]` peaks at `(t+3)` (1.2) over
//! the `inputs_embeds` self-slot `t` (1.0), so the next token is `(t+4) mod 8`.
//! Freezing `aux` OR dropping the gate collapses that slot to 0.6 < 1.0 and the
//! argmax falls back to `(t+1) mod 8` — hence the three distinct id streams.

use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};

fn tiny_vlm_multibinding_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-vlm-multibinding")
}

// Prompt and closed-form references, mirrored from the fixture's expected.json.
const PROMPT: [u32; 2] = [1, 4];
/// BOTH tensors refreshed every step (the correct generic executor).
const FRESH_BOTH: [u32; 4] = [0, 4, 0, 4];
/// `aux` frozen at the prompt's last position while `inputs_embeds` refreshes —
/// what a stale single-output special case would produce.
const STALE_AUX: [u32; 4] = [0, 1, 2, 3];
/// The decoder's raw-token gate dropped (input_ids not consumed alongside embeds).
const IDS_IGNORED: [u32; 4] = [5, 6, 7, 0];

fn generate(dir: &Path) -> anyhow::Result<Vec<u32>> {
    let mut engine = Engine::from_pipeline_dir(dir, EngineConfig::default())?;

    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(PROMPT.to_vec()));
    request.options = GenerateOptions {
        max_new_tokens: 4,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };

    // No external inputs: the every_step `embedding` takes only `input_ids`,
    // which the executor seeds from the running token each step.
    let result = engine.generate_with_pipeline_request(PipelineGenerateRequest::new(request))?;
    Ok(result.token_ids)
}

/// Acceptance #2: the every_step component's SECOND output (`aux`) is refreshed
/// each step. The exact id stream only matches `fresh_both` when both
/// `inputs_embeds` and `aux` track the running token; a stale `aux` would yield
/// the distinct `stale_aux` stream.
#[test]
fn multibinding_refreshes_every_component_output_each_step() -> anyhow::Result<()> {
    let tokens = generate(&tiny_vlm_multibinding_dir())?;

    assert_eq!(
        tokens, FRESH_BOTH,
        "generic executor must refresh BOTH inputs_embeds and aux each step"
    );
    // Sanity: the references are genuinely distinct, so matching fresh_both is a
    // real signal that the second tensor was exercised (not a stale coincidence).
    assert_ne!(
        FRESH_BOTH, STALE_AUX,
        "a stale second tensor would produce a different id stream"
    );
    Ok(())
}

/// Acceptance #3: the decoder consumes the RAW `input_ids` token stream in the
/// SAME forward pass as the routed `inputs_embeds`/`aux`, proving there is no
/// token/embeds exclusivity in pipeline execution. Dropping the raw-token
/// contribution would yield the distinct `ids_ignored` stream.
#[test]
fn multibinding_decoder_consumes_raw_input_ids_and_embeds_together() -> anyhow::Result<()> {
    let tokens = generate(&tiny_vlm_multibinding_dir())?;

    assert_eq!(
        tokens, FRESH_BOTH,
        "decoder must consume raw input_ids alongside routed inputs_embeds/aux"
    );
    assert_ne!(
        FRESH_BOTH, IDS_IGNORED,
        "ignoring the raw input_ids would produce a different id stream"
    );
    Ok(())
}
