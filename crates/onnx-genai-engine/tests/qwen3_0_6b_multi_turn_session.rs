//! The regression, on the package it was found in.
//!
//! `qwen3-0.6b-onnx-genai` is a real eleven-component decoder package:
//! `model.onnx` plus ten ONNX policy graphs, no `binding` token policy, and
//! therefore no decode core. It is the shape whose multi-turn continuation was
//! lost when the interpreter started executing what packages declare, because
//! what it declared was that nothing survives an invocation.
//!
//! The hermetic cases in `workflow_session_continuation.rs` pin the same
//! property against a fixture whose synthetic weights decode a constant token.
//! This one pins it where token values mean something: a third turn must decode
//! the same tokens as one request carrying the whole conversation, and must not
//! decode the same tokens as its own prompt sent cold.
//!
//! # Published revision
//!
//! ```text
//! justinchuby/qwen3-0.6b-onnx-genai @ 38714511f57e01df01808b930168459a8e7aa9a3
//! ```
//!
//! This was the repository's default revision when the session-continuation
//! contract and development-runtime requirement were published.
//!
//! ```text
//! huggingface-cli download justinchuby/qwen3-0.6b-onnx-genai \
//!   --revision 38714511f57e01df01808b930168459a8e7aa9a3 --local-dir /path/to/pkg
//!
//! ONNX_GENAI_QWEN3_WORKFLOW_DIR=/path/to/pkg ONNX_GENAI_KV_MAX_LEN=1024 \
//!   cargo test -p onnx-genai-engine --test qwen3_0_6b_multi_turn_session \
//!   -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use sha2::{Digest, Sha256};

/// The revision whose metadata declares the conversation.
const PINNED_REVISION: &str = "38714511f57e01df01808b930168459a8e7aa9a3";
const PINNED_METADATA_SHA256: &str =
    "277582c682e3136854ef87be949467bd8308d22ffc3dc0f2aef7f13b7fe8f015";

/// Resolve the package, failing with what to do rather than skipping.
///
/// A green skip on a machine with no package and a green skip on a machine whose
/// package is the wrong revision look identical, and the second is a regression
/// reported as success.
fn package() -> PathBuf {
    let dir = std::env::var_os("ONNX_GENAI_QWEN3_WORKFLOW_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!(
                "set ONNX_GENAI_QWEN3_WORKFLOW_DIR to a checkout of \
                 justinchuby/qwen3-0.6b-onnx-genai@{PINNED_REVISION} (the published \
                 default revision)."
            )
        });
    let metadata = dir.join("inference_metadata.yaml");
    let document = std::fs::read_to_string(&metadata)
        .unwrap_or_else(|error| panic!("{}: {error}", metadata.display()));
    let digest = format!("{:x}", Sha256::digest(document.as_bytes()));
    assert_eq!(
        digest,
        PINNED_METADATA_SHA256,
        "{} is not the metadata published at {PINNED_REVISION}",
        metadata.display()
    );
    assert!(
        document.contains("continuation:"),
        "{} declares no `session.continuation`; fetch the published default revision \
         {PINNED_REVISION}.",
        metadata.display()
    );
    dir
}

fn tokens(ids: &[u32], max_new_tokens: usize) -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::TokenIds(ids.to_vec()),
        options: GenerateOptions {
            max_new_tokens,
            greedy: true,
            temperature: 0.0,
            stop_on_eos: false,
            ..GenerateOptions::default()
        },
    }
}

#[test]
#[ignore = "requires justinchuby/qwen3-0.6b-onnx-genai@38714511 \
            in ONNX_GENAI_QWEN3_WORKFLOW_DIR"]
fn a_third_turn_decodes_the_conversation_and_not_its_own_prompt() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&package(), EngineConfig::default())?;
    let opening = engine.tokenize("My name is Ada. Remember it.")?;
    let follow_up = engine.tokenize(" What is my name?")?;
    let third = engine.tokenize(" Say it once more.")?;

    let session = engine.create_session()?;
    let turn_one = engine.generate_in_session(session, tokens(&opening, 8))?;
    assert_eq!(
        engine.session_token_count(session)?,
        opening.len() + turn_one.token_ids.len()
    );
    let turn_two = engine.generate_in_session(session, tokens(&follow_up, 8))?;
    let turn_three = engine.generate_in_session(session, tokens(&third, 8))?;

    let mut conversation = opening.clone();
    conversation.extend(turn_one.token_ids.iter().copied());
    conversation.extend(follow_up.iter().copied());
    conversation.extend(turn_two.token_ids.iter().copied());
    conversation.extend(third.iter().copied());
    let single_shot = engine.generate(tokens(&conversation, 8))?;
    let cold = engine.generate(tokens(&third, 8))?;

    println!("turn 3      {:?}", turn_three.token_ids);
    println!("single shot {:?}", single_shot.token_ids);
    println!("cold        {:?}", cold.token_ids);

    assert_eq!(
        turn_three.token_ids, single_shot.token_ids,
        "the third turn must decode the conversation"
    );
    assert_ne!(
        turn_three.token_ids, cold.token_ids,
        "the third turn must not decode its own prompt alone, which is the regression"
    );
    assert_eq!(
        engine.session_token_count(session)?,
        conversation.len() + turn_three.token_ids.len()
    );
    assert_eq!(
        engine.session_conversation(session)?.map(|held| held.len()),
        Some(conversation.len() + turn_three.token_ids.len())
    );

    // Independent sessions, and a reset that releases what one held.
    let other = engine.create_session()?;
    let isolated = engine.generate_in_session(other, tokens(&third, 8))?;
    assert_eq!(
        isolated.token_ids, cold.token_ids,
        "a session that has heard nothing decodes what a stateless request decodes"
    );
    engine.reset_session(session)?;
    assert_eq!(engine.session_token_count(session)?, 0);
    let after_reset = engine.generate_in_session(session, tokens(&third, 8))?;
    assert_eq!(
        after_reset.token_ids, cold.token_ids,
        "reset releases the conversation"
    );
    Ok(())
}
