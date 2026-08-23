//! A conversation a package declares, carried across its invocations.
//!
//! The property under test is the one a caller opening a session is asking for:
//! **turn N continues turns 1..N-1**. It is pinned the only way that cannot pass
//! by accident — against the same package answering the whole conversation as a
//! single request. If the third turn of a session and one stateless generation
//! over the concatenated context produce different tokens, the session did not
//! carry the conversation, whatever else it reported.
//!
//! Everything here runs against
//! `tests/fixtures/onnx_genai_workflows/decoder`, an interpreted package: eleven
//! ONNX components, no `binding` token policy and therefore no decode core. That
//! is the shape whose conversation lives in the workflow's `scope: session`
//! state rather than in a paged KV sequence, which is exactly what regressed.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};

fn decoder_package() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/onnx_genai_workflows/decoder")
}

/// A package with no session-scoped state at all: the same fixture with its
/// declared conversation removed, which is what every migrated package looked
/// like before this.
fn package_without_conversation(root: &Path) -> anyhow::Result<()> {
    let metadata = root.join("inference_metadata.yaml");
    let mut document: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata)?)?;
    document["pipeline"]["workflow"]["state"]
        .as_mapping_mut()
        .expect("workflow declares state")
        .remove(serde_yaml::Value::String("conversation".into()))
        .expect("the fixture declares a conversation");
    let capabilities = document["pipeline"]["workflow"]["manifest"]["capabilities"]
        .as_sequence_mut()
        .expect("the manifest declares capabilities");
    capabilities.retain(|capability| capability.as_str() != Some("session_state_lease"));
    std::fs::write(&metadata, serde_yaml::to_string(&document)?)?;
    Ok(())
}

fn copy_package(destination: &Path) -> anyhow::Result<()> {
    fn copy_tree(from: &Path, to: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(to)?;
        for entry in std::fs::read_dir(from)? {
            let entry = entry?;
            let target = to.join(entry.file_name());
            if entry.file_type()?.is_dir() {
                copy_tree(&entry.path(), &target)?;
            } else {
                std::fs::copy(entry.path(), target)?;
            }
        }
        Ok(())
    }
    copy_tree(&decoder_package(), destination)
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

/// The third turn of a session decodes what one request carrying the whole
/// conversation decodes.
///
/// The comparison is on token ids rather than text: re-tokenizing decoded text
/// can merge differently at a turn boundary, which would make a passing run
/// evidence about the tokenizer rather than about the session.
#[test]
fn third_turn_matches_the_single_shot_conversation() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&decoder_package(), EngineConfig::default())?;
    let turn_one = [2u32, 4, 6, 3];
    let turn_two = [5u32, 7];
    let turn_three = [9u32, 8, 11];

    let session = engine.create_session()?;
    assert_eq!(engine.session_token_count(session)?, 0);

    let first = engine.generate_in_session(session, tokens(&turn_one, 3))?;
    assert_eq!(
        engine.session_token_count(session)?,
        turn_one.len() + first.token_ids.len(),
        "a session's token count is the conversation, prompt and generation alike"
    );
    let second = engine.generate_in_session(session, tokens(&turn_two, 3))?;
    let third = engine.generate_in_session(session, tokens(&turn_three, 3))?;

    // The conversation as one request: every turn's prompt and every turn's
    // published tokens, in order.
    let mut conversation = turn_one.to_vec();
    conversation.extend(first.token_ids.iter().copied());
    conversation.extend(turn_two.iter().copied());
    conversation.extend(second.token_ids.iter().copied());
    conversation.extend(turn_three.iter().copied());
    let single_shot = engine.generate(tokens(&conversation, 3))?;

    assert_eq!(
        third.token_ids, single_shot.token_ids,
        "turn 3 must decode the conversation, not its own prompt"
    );
    assert_eq!(
        engine.session_token_count(session)?,
        conversation.len() + third.token_ids.len()
    );
    // The prompt turn 3 actually ran, read back from the lease: this is what
    // makes the equality above evidence about the session rather than about a
    // fixture whose synthetic weights decode a constant.
    let mut expected = conversation.clone();
    expected.extend(third.token_ids.iter().copied());
    assert_eq!(
        engine.session_conversation(session)?.as_deref(),
        Some(expected.as_slice()),
        "the session's conversation is every turn's prompt and generation, in order"
    );
    Ok(())
}

/// A second turn is decoded against a cache the first turn filled, and the
/// prompt it was decoded from says so.
///
/// This fixture's synthetic weights decode a constant token, so comparing token
/// values here would prove nothing about it. What it can prove — and what the
/// tokens of a real model follow from — is *what the decoder was asked to
/// decode*: the conversation the pass ran, read back from the lease the package
/// declared rather than from a count kept beside it.
#[test]
fn a_continued_turn_decodes_against_the_earlier_turns_context() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&decoder_package(), EngineConfig::default())?;
    let session = engine.create_session()?;
    let opening = [2u32, 4, 6, 3];
    let follow_up = [5u32, 7];

    let first = engine.generate_in_session(session, tokens(&opening, 4))?;
    let mut expected = opening.to_vec();
    expected.extend(first.token_ids.iter().copied());
    assert_eq!(
        engine.session_conversation(session)?.as_deref(),
        Some(expected.as_slice()),
        "the first turn leaves its prompt and its generation behind"
    );

    let second = engine.generate_in_session(session, tokens(&follow_up, 4))?;
    expected.extend(follow_up.iter().copied());
    expected.extend(second.token_ids.iter().copied());
    assert_eq!(
        engine.session_conversation(session)?.as_deref(),
        Some(expected.as_slice()),
        "the second turn extends the first turn's context instead of replacing it"
    );
    assert_eq!(engine.session_token_count(session)?, expected.len());

    // The same follow-up with no conversation behind it is decoded from its own
    // tokens alone — which is what the second turn must not have done.
    let cold_session = engine.create_session()?;
    engine.generate_in_session(cold_session, tokens(&follow_up, 4))?;
    let cold = engine
        .session_conversation(cold_session)?
        .expect("the package declares a conversation");
    assert_eq!(
        &cold[..follow_up.len()],
        follow_up.as_slice(),
        "a cold turn's context is its own prompt"
    );
    assert!(
        cold.len() < expected.len(),
        "a continued turn decoded more context than a cold one"
    );
    Ok(())
}

/// Two sessions are two conversations.
#[test]
fn independent_sessions_do_not_see_each_other() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&decoder_package(), EngineConfig::default())?;
    let first_session = engine.create_session()?;
    let second_session = engine.create_session()?;
    let probe = [9u32, 8, 11];

    engine.generate_in_session(first_session, tokens(&[2, 4, 6, 3], 4))?;
    let isolated = engine.generate_in_session(second_session, tokens(&probe, 3))?;
    let stateless = engine.generate(tokens(&probe, 3))?;

    assert_eq!(
        isolated.token_ids, stateless.token_ids,
        "a session that has heard nothing decodes what a stateless request decodes"
    );
    assert_eq!(
        engine.session_token_count(second_session)?,
        probe.len() + isolated.token_ids.len()
    );
    Ok(())
}

/// Reset frees the conversation and keeps the id; close frees it and does not.
#[test]
fn reset_and_close_release_the_conversation() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&decoder_package(), EngineConfig::default())?;
    let session = engine.create_session()?;
    let probe = [9u32, 8, 11];

    engine.generate_in_session(session, tokens(&[2, 4, 6, 3], 4))?;
    assert!(engine.session_token_count(session)? > 0);

    engine.reset_session(session)?;
    assert_eq!(
        engine.session_token_count(session)?,
        0,
        "reset releases what the conversation held"
    );
    assert_eq!(
        engine.session_conversation(session)?,
        Some(Vec::new()),
        "a reset session has heard nothing"
    );

    let after_reset = engine.generate_in_session(session, tokens(&probe, 3))?;
    let stateless = engine.generate(tokens(&probe, 3))?;
    assert_eq!(
        after_reset.token_ids, stateless.token_ids,
        "a reset session starts the conversation again"
    );

    engine.close_session(session)?;
    assert!(
        engine.session_token_count(session).is_err(),
        "a closed session is gone, not emptied"
    );
    assert!(
        engine.session_conversation(session).is_err(),
        "a closed session has no conversation to report"
    );
    assert!(
        engine
            .generate_in_session(session, tokens(&probe, 3))
            .is_err(),
        "a closed session cannot be generated in"
    );
    Ok(())
}

/// Declaring a conversation costs the package nothing when nobody asks for one.
#[test]
fn stateless_generation_is_unchanged_by_a_declared_conversation() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&decoder_package(), EngineConfig::default())?;
    let probe = [2u32, 4, 6, 3];
    let first = engine.generate(tokens(&probe, 4))?;
    let repeat = engine.generate(tokens(&probe, 4))?;
    assert_eq!(
        first.token_ids, repeat.token_ids,
        "a stateless request leaves nothing behind for the next one"
    );

    // And a session's first turn is that same stateless generation.
    let session = engine.create_session()?;
    let opening = engine.generate_in_session(session, tokens(&probe, 4))?;
    assert_eq!(first.token_ids, opening.token_ids);
    Ok(())
}

/// A package that cannot continue a conversation refuses to open a session.
///
/// The alternative is what regressed: `create_session` succeeds, every turn
/// silently restarts, and the failure reaches the caller as a model that forgot
/// what it was told.
#[test]
fn a_package_with_no_declared_conversation_refuses_a_session() -> anyhow::Result<()> {
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("no_conversation_package");
    let _ = std::fs::remove_dir_all(&scratch);
    copy_package(&scratch)?;
    package_without_conversation(&scratch)?;

    let mut engine = Engine::from_dir(&scratch, EngineConfig::default())?;
    // It still generates: a package with no conversation is a package that
    // answers one question at a time, which is a fact about it rather than a
    // fault.
    let stateless = engine.generate(tokens(&[2, 4, 6, 3], 3))?;
    assert_eq!(stateless.token_ids.len(), 3);

    let refused = engine
        .create_session()
        .expect_err("a package that cannot continue a conversation must not hand out a session");
    let message = format!("{refused:#}");
    assert!(
        message.contains("scope: session"),
        "the refusal names what the package has to declare: {message}"
    );
    Ok(())
}

/// The bound the conversation declares is the bound it keeps.
///
/// A continuation is deliberately not loop-carried, so it never reaches the
/// carry path's recurrence check — this is the only place its declared `max` is
/// honoured. Without it the conversation would grow past the package's context
/// limit and the turn that crossed it would fail somewhere inside the decoder
/// graph, with a shape error nobody could attribute to a session.
#[test]
fn a_conversation_is_refused_past_the_bound_it_declares() -> anyhow::Result<()> {
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("short_context_package");
    let _ = std::fs::remove_dir_all(&scratch);
    copy_package(&scratch)?;
    let metadata = scratch.join("inference_metadata.yaml");
    let mut document: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata)?)?;
    // Only the conversation's bound is narrowed. Narrowing `package.max_context`
    // would narrow the cache cells with it and prove nothing about this check.
    let limit: serde_yaml::Value = serde_yaml::from_str(
        "contract: { dtype: int64, rank: 1, shape: [1] }\n\
         role: { kind: opaque }\n\
         source: { kind: literal }\n\
         required: false\n\
         default: 6\n",
    )?;
    document["pipeline"]["workflow"]["inputs"]
        .as_mapping_mut()
        .expect("workflow declares inputs")
        .insert("package.conversation_limit".into(), limit);
    document["pipeline"]["workflow"]["state"]["conversation"]["recurrence"]["max"] =
        serde_yaml::Value::String("package.conversation_limit".into());
    std::fs::write(&metadata, serde_yaml::to_string(&document)?)?;

    let mut engine = Engine::from_dir(&scratch, EngineConfig::default())?;
    let session = engine.create_session()?;
    // Four prompt tokens and two generated leaves the conversation exactly at
    // the bound, which is legal.
    engine.generate_in_session(session, tokens(&[2, 4, 6, 3], 2))?;
    assert_eq!(engine.session_token_count(session)?, 6);

    let refused = engine
        .generate_in_session(session, tokens(&[5, 7], 2))
        .expect_err("a turn past the declared bound must be refused");
    let message = format!("{refused:#}");
    assert!(
        message.contains("declares a bound of 6"),
        "the refusal names the bound and the conversation: {message}"
    );
    // The refusal left the conversation as it was, so the session is still
    // usable once it is reset.
    assert_eq!(engine.session_token_count(session)?, 6);
    engine.reset_session(session)?;
    engine.generate_in_session(session, tokens(&[5, 7], 2))?;
    assert_eq!(engine.session_token_count(session)?, 4);
    Ok(())
}

/// A lease nothing carries is refused before a caller can hold the package.
///
/// The runtime hands a lease back through a loop carry, a state service group
/// whose alias names the cell, or a declared continuation — and
/// `classify_session_state` is the one place that says which. A cell with none
/// of them is written back on every pass and read by nothing, so the document is
/// refused at load rather than at the third turn of a conversation.
#[test]
fn a_session_cell_nothing_carries_is_refused_at_load() -> anyhow::Result<()> {
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("unread_lease_package");
    let _ = std::fs::remove_dir_all(&scratch);
    copy_package(&scratch)?;
    let metadata = scratch.join("inference_metadata.yaml");
    let mut document: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata)?)?;
    // Keep the cell session-scoped and keep its initializer a value the steps
    // read — drop only the contract that says how the lease rejoins a turn.
    document["pipeline"]["workflow"]["state"]["conversation"]["session"]
        .as_mapping_mut()
        .expect("the conversation declares a lease")
        .remove(serde_yaml::Value::String("continuation".into()))
        .expect("the conversation declares a continuation");
    std::fs::write(&metadata, serde_yaml::to_string(&document)?)?;

    let refused = match Engine::from_dir(&scratch, EngineConfig::default()) {
        Ok(_) => panic!("a lease nothing carries cannot continue a conversation"),
        Err(error) => error,
    };
    let message = format!("{refused:#}");
    assert!(
        message.contains("nothing in this document says how the next invocation reaches"),
        "the refusal names what is missing: {message}"
    );
    Ok(())
}

/// The runtime and the validator agree about what carries a lease.
///
/// They used to disagree: the validator blessed a session-scoped cell held by a
/// state service group, and the runtime refused to open a session for the same
/// package because its own predicate only knew about loop carries. A caller
/// then held a package that validated and could not have a conversation. Both
/// now read `classify_session_state`, so this asserts the classification the
/// package declares and that a session opens on the strength of it.
#[test]
fn a_state_service_group_carries_a_lease_the_validator_blessed() -> anyhow::Result<()> {
    let metadata = onnx_genai_metadata::load_metadata_from_dir(&decoder_package())?
        .expect("the fixture ships metadata");
    let workflow = &metadata
        .pipeline
        .as_ref()
        .expect("the fixture declares a pipeline")
        .workflow;
    let facts = onnx_genai_metadata::classify_session_state(workflow);
    assert_eq!(
        facts.carrier("conversation"),
        Some(onnx_genai_metadata::SessionStateCarrier::PromptContinuation)
    );
    assert_eq!(facts.prompt_continuation(), Some("conversation"));
    assert_eq!(facts.uncarried().count(), 0);

    // The same fixture with its cache group declared session-scoped: the group
    // holds the storage, the alias names the ports, and that is a carrier.
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("group_carried_package");
    let _ = std::fs::remove_dir_all(&scratch);
    copy_package(&scratch)?;
    let path = scratch.join("inference_metadata.yaml");
    let mut document: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(&path)?)?;
    let state = document["pipeline"]["workflow"]["state"]
        .as_mapping_mut()
        .expect("workflow declares state");
    state
        .remove(serde_yaml::Value::String("conversation".into()))
        .expect("the fixture declares a conversation");
    let cache = state
        .get_mut(serde_yaml::Value::String("cache_0".into()))
        .expect("the fixture declares a cache cell");
    cache["scope"] = serde_yaml::Value::String("session".into());
    cache["release_boundary"] = serde_yaml::Value::String("session".into());
    std::fs::write(&path, serde_yaml::to_string(&document)?)?;

    let metadata =
        onnx_genai_metadata::load_metadata_from_dir(&scratch)?.expect("the fixture ships metadata");
    onnx_genai_metadata::validate_metadata(&metadata)
        .map_err(|errors| anyhow::anyhow!("{errors:?}"))?;
    let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;
    let facts = onnx_genai_metadata::classify_session_state(workflow);
    // The cache is both group-backed and loop-carried, and the loop carry is
    // the mechanism a pass actually reaches first.
    assert_eq!(
        facts.carrier("cache_0"),
        Some(onnx_genai_metadata::SessionStateCarrier::LoopCarry)
    );
    assert_eq!(facts.uncarried().count(), 0);

    // And a session opens, because something carries the lease.
    let mut engine = Engine::from_dir(&scratch, EngineConfig::default())?;
    let session = engine.create_session()?;
    assert_eq!(engine.session_token_count(session)?, 0);
    // This package declares no prompt continuation, so it holds no conversation
    // of tokens to report — and says so rather than inventing a number.
    assert_eq!(engine.session_conversation(session)?, None);
    engine.close_session(session)?;
    Ok(())
}

/// A group named but never aliased for the cell carries nothing, and the
/// document is refused before a caller can open a session against it.
#[test]
fn a_session_cell_whose_group_does_not_alias_it_is_refused_at_load() -> anyhow::Result<()> {
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("unaliased_group_package");
    let _ = std::fs::remove_dir_all(&scratch);
    copy_package(&scratch)?;
    let path = scratch.join("inference_metadata.yaml");
    let mut document: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(&path)?)?;
    let conversation = document["pipeline"]["workflow"]["state"]["conversation"]
        .as_mapping_mut()
        .expect("the fixture declares a conversation");
    conversation.remove(serde_yaml::Value::String("session".into()));
    conversation.insert(
        serde_yaml::Value::String("service_group".into()),
        serde_yaml::Value::String("decoder_cache".into()),
    );
    std::fs::write(&path, serde_yaml::to_string(&document)?)?;

    let refused = match Engine::from_dir(&scratch, EngineConfig::default()) {
        Ok(_) => panic!("a lease bound to a group that never names it must not load"),
        Err(error) => error,
    };
    let message = format!("{refused:#}");
    assert!(
        message.contains("no component alias in that group names it"),
        "the refusal names the group that holds nothing: {message}"
    );
    Ok(())
}
