//! The regression, on the package it was found in.
//!
//! `qwen3-0.6b-onnx-genai` is a real eleven-component decoder package: `model.onnx`
//! plus ten ONNX policy graphs, no `binding` token policy, and therefore no
//! decode core. It is the shape whose multi-turn continuation was lost when the
//! interpreter started executing what packages declare, because what it declared
//! was that nothing survives an invocation.
//!
//! The hermetic case in `workflow_session_continuation.rs` pins the same
//! property against a fixture whose synthetic weights decode a constant token.
//! This one pins it where token values mean something: a third turn must decode
//! the same tokens as one request carrying the whole conversation, and must not
//! decode the same tokens as its own prompt sent cold.
//!
//! ```text
//! ONNX_GENAI_QWEN3_WORKFLOW_DIR=/path/to/qwen3-0.6b-onnx-genai \
//!   ONNX_GENAI_KV_MAX_LEN=1024 \
//!   cargo test -p onnx-genai-engine --test qwen3_0_6b_multi_turn_session -- --nocapture
//! ```

use std::path::PathBuf;

use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};

fn package() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("ONNX_GENAI_QWEN3_WORKFLOW_DIR")?);
    dir.join("inference_metadata.yaml").is_file().then_some(dir)
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
fn a_third_turn_decodes_the_conversation_and_not_its_own_prompt() -> anyhow::Result<()> {
    let Some(package) = package() else {
        eprintln!("skipped: set ONNX_GENAI_QWEN3_WORKFLOW_DIR to the package directory");
        return Ok(());
    };
    let mut engine = Engine::from_dir(&package, EngineConfig::default())?;
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
