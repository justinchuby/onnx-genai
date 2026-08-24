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

/// Narrow only the conversation's own bound.
///
/// Narrowing `package.max_context` would narrow the cache cells with it and
/// prove nothing about the conversation's bound.
fn narrow_the_conversation_bound(root: &Path, limit: i64) -> anyhow::Result<()> {
    let metadata = root.join("inference_metadata.yaml");
    let mut document: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata)?)?;
    let declared: serde_yaml::Value = serde_yaml::from_str(&format!(
        "contract: {{ dtype: int64, rank: 1, shape: [1] }}\nrole: {{ kind: opaque }}\nsource:          {{ kind: literal }}\nrequired: false\ndefault: {limit}\n"
    ))?;
    document["pipeline"]["workflow"]["inputs"]
        .as_mapping_mut()
        .expect("workflow declares inputs")
        .insert("package.conversation_limit".into(), declared);
    document["pipeline"]["workflow"]["state"]["conversation"]["recurrence"]["max"] =
        serde_yaml::Value::String("package.conversation_limit".into());
    std::fs::write(&metadata, serde_yaml::to_string(&document)?)?;
    Ok(())
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
    let _ = &mut document;
    drop(document);
    narrow_the_conversation_bound(&scratch, 6)?;

    let mut engine = Engine::from_dir(&scratch, EngineConfig::default())?;
    let session = engine.create_session()?;
    // Four prompt tokens and two generated leaves the conversation exactly at
    // the bound, which is legal.
    engine.generate_in_session(session, tokens(&[2, 4, 6, 3], 2))?;
    assert_eq!(engine.session_token_count(session)?, 6);

    let refused = engine
        .generate_in_session(session, tokens(&[5, 7], 2))
        .expect_err("a turn past the declared bound must be refused");
    let capability =
        onnx_genai_engine::package_capability_error(&refused).expect("the refusal is typed");
    match capability {
        onnx_genai_engine::PackageCapabilityError::ConversationOverBound {
            cell,
            requested,
            bound,
        } => {
            assert_eq!(cell, "conversation");
            assert_eq!(bound, 6);
            assert!(requested > bound);
        }
        other => panic!("unexpected capability refusal: {other:?}"),
    }
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

/// A group-backed lease enters at the port its alias declares, not by
/// overwriting the value the cell's initializer names.
///
/// The distinction is the difference between working and silently restarting.
/// The canonical group cell's initializer is a value a *setup step produces*
/// (`decoder.setup.present.0.key` in this fixture), so writing the lease there
/// before the pass would be overwritten by that step and dropped with no error —
/// exactly the regression this change exists to fix. The lease is bound at the
/// alias's `input` port instead, and the validator refuses an alias no step
/// reaches so the binding always has a reader.
#[test]
fn a_group_lease_is_bound_at_its_port_and_must_have_a_reader() -> anyhow::Result<()> {
    let metadata = onnx_genai_metadata::load_metadata_from_dir(&decoder_package())?
        .expect("the fixture ships metadata");
    let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;

    // The alias the lease would enter through, and the step that reads it.
    let aliases = onnx_genai_metadata::session_group_aliases(workflow, "cache_0");
    let (component, alias) = aliases.first().expect("the cache cell is aliased");
    assert_eq!(*component, "model");
    assert_eq!(alias.input, "past_key_values.0.key");
    assert_eq!(alias.output.as_deref(), Some("present.0.key"));

    // The cell's initializer is produced by a setup step, which is precisely why
    // the lease cannot be written there.
    let initializer = &workflow.state["cache_0"].initializer;
    assert_eq!(initializer, "decoder.setup.present.0.key");

    // A group-only lease is written to the value the step binds to the alias's
    // `input` port, before the pass. That is what every way of invoking a
    // component reads — generically, fused into an execution island, through a
    // host contract, or redirected by an override — so a document where a step
    // could then overwrite that value is refused rather than left to restart
    // the session quietly. This fixture is that document: `cache_0`'s port is
    // bound to a value the setup step produces.
    {
        let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("overwritten_lease_package");
        let _ = std::fs::remove_dir_all(&scratch);
        copy_package(&scratch)?;
        let path = scratch.join("inference_metadata.yaml");
        let mut document: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(&path)?)?;
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
        let carried = document["pipeline"]["workflow"]["steps"][0]["carried"]
            .as_sequence_mut()
            .expect("the loop carries state");
        carried.retain(|carry| carry["cell"].as_str() != Some("cache_0"));
        std::fs::write(&path, serde_yaml::to_string(&document)?)?;

        let metadata =
            onnx_genai_metadata::load_metadata_from_dir(&scratch)?.expect("metadata was written");
        let reported = onnx_genai_metadata::validate_metadata(&metadata)
            .expect_err("a lease a step overwrites is not a carrier");
        assert!(
            reported
                .iter()
                .any(|error| error.contains("would overwrite the lease")),
            "expected the overwrite refusal, got {reported:?}"
        );
    }

    // A group whose alias names a port no step binds is refused: the lease would
    // have no reader.
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("unread_group_port_package");
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
    // Session-scope the cache and take it out of the loop's carries, so the
    // group is the only thing that could hold it.
    let cache = state
        .get_mut(serde_yaml::Value::String("cache_0".into()))
        .expect("the fixture declares a cache cell");
    cache["scope"] = serde_yaml::Value::String("session".into());
    cache["release_boundary"] = serde_yaml::Value::String("session".into());
    let carried = document["pipeline"]["workflow"]["steps"][0]["carried"]
        .as_sequence_mut()
        .expect("the loop carries state");
    carried.retain(|carry| carry["cell"].as_str() != Some("cache_0"));
    // And rename the port the alias reads to one no step binds.
    document["pipeline"]["workflow"]["serving"]["state_service"]["groups"]["decoder_cache"]["ports"]
        ["model"]["cache_0"]["input"] = serde_yaml::Value::String("past_key_values.absent".into());
    std::fs::write(&path, serde_yaml::to_string(&document)?)?;

    let metadata =
        onnx_genai_metadata::load_metadata_from_dir(&scratch)?.expect("metadata was written");
    let reported =
        onnx_genai_metadata::validate_metadata(&metadata).expect_err("the lease has no reader");
    assert!(
        reported
            .iter()
            .any(|error| error.contains("the lease would have no reader")),
        "expected the missing-reader refusal, got {reported:?}"
    );
    Ok(())
}

/// A turn that fails hands back everything it took, so the next one is not
/// refused for it.
///
/// Two things are released on the error path and neither has an observable of
/// its own: the session's exclusive lease, which a later turn would be refused
/// by name for, and the scheduler reservation #1900 added for the interpreted
/// path, which a later turn would be refused admission for. A refusal followed
/// by a success is falsified by leaking either, which is what makes this the
/// test for both.
#[test]
fn a_failed_turn_releases_its_lease_and_its_reservation() -> anyhow::Result<()> {
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("released_after_failure_package");
    let _ = std::fs::remove_dir_all(&scratch);
    copy_package(&scratch)?;
    narrow_the_conversation_bound(&scratch, 6)?;

    let mut engine = Engine::from_dir(&scratch, EngineConfig::default())?;
    let session = engine.create_session()?;
    // Four prompt tokens and two generated leaves the conversation at the bound.
    engine.generate_in_session(session, tokens(&[2, 4, 6, 3], 2))?;
    assert_eq!(engine.session_token_count(session)?, 6);

    // This turn is refused, after taking a lease and a reservation.
    let refused = engine
        .generate_in_session(session, tokens(&[5, 7], 2))
        .expect_err("a turn past the declared bound must be refused");
    let capability = onnx_genai_engine::package_capability_error(&refused)
        .expect("the refusal is typed, so a front end never reads its wording");
    assert!(
        matches!(
            capability,
            onnx_genai_engine::PackageCapabilityError::ConversationOverBound { bound: 6, .. }
        ),
        "{capability:?}"
    );
    assert!(
        !capability.is_retryable(),
        "the same request against the same conversation will not start succeeding"
    );

    // The conversation is untouched, and the next turn runs — which it could not
    // if the failed turn had kept either the lease or the reservation.
    assert_eq!(engine.session_token_count(session)?, 6);
    engine.reset_session(session)?;
    let after = engine.generate_in_session(session, tokens(&[5, 7], 2))?;
    assert_eq!(after.token_ids.len(), 2);
    assert_eq!(engine.session_token_count(session)?, 4);
    Ok(())
}

/// A busy session is the other capability refusal, and it is retryable.
///
/// The lease is what makes `policy: exclusive` true rather than assumed. Its
/// variant is separated from the others because the answer a caller should get
/// is different: the same request succeeds once the turn in flight finishes.
#[test]
fn a_busy_session_is_a_retryable_capability_refusal() {
    let busy = onnx_genai_engine::PackageCapabilityError::ExclusiveLeaseConflict {
        session: "shared".to_string(),
    };
    assert!(busy.is_retryable());
    assert!(busy.to_string().contains("already has a turn in flight"));

    let error: anyhow::Error = busy.into();
    assert!(matches!(
        onnx_genai_engine::package_capability_error(&error),
        Some(onnx_genai_engine::PackageCapabilityError::ExclusiveLeaseConflict { .. })
    ));
}

#[test]
fn copy_on_write_session_mutation_is_refused_at_load() -> anyhow::Result<()> {
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("copy_on_write_lease_package");
    let _ = std::fs::remove_dir_all(&scratch);
    copy_package(&scratch)?;
    let path = scratch.join("inference_metadata.yaml");
    let mut document: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(&path)?)?;
    document["pipeline"]["workflow"]["state"]["conversation"]["session"]["policy"] =
        serde_yaml::Value::String("copy_on_write".into());
    std::fs::write(&path, serde_yaml::to_string(&document)?)?;

    let refused = match Engine::from_dir(&scratch, EngineConfig::default()) {
        Ok(_) => panic!("an unsupported mutation policy must not load"),
        Err(error) => format!("{error:#}"),
    };
    assert!(refused.contains("copy-on-write"), "{refused}");
    Ok(())
}

/// Carriers decide independently what is attended and what is recomputed.
#[test]
fn a_carrier_decides_what_is_attended_and_what_is_recomputed() -> anyhow::Result<()> {
    let mut interpreted = Engine::from_dir(&decoder_package(), EngineConfig::default())?;
    assert!(interpreted.prepends_session_conversation());
    let session = interpreted.create_session()?;
    assert_eq!(
        interpreted.session_prefill_carry(session)?,
        onnx_genai_engine::SessionPrefillCarry::default()
    );
    let opening = [2u32, 4, 6, 3];
    let first = interpreted.generate_in_session(session, tokens(&opening, 3))?;
    let conversation = opening.len() + first.token_ids.len();
    assert_eq!(
        interpreted.session_prefill_carry(session)?,
        onnx_genai_engine::SessionPrefillCarry {
            attended: conversation,
            reprefilled: conversation,
        }
    );
    assert_eq!(interpreted.session_token_count(session)?, conversation);

    // An ORT decode core attends its retained sequence from KV.
    let tiny = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let mut decode_core = Engine::from_dir(&tiny, EngineConfig::default())?;
    assert!(!decode_core.prepends_session_conversation());
    let session = decode_core.create_session()?;
    let first = decode_core.generate_in_session(session, tokens(&[2, 4, 3], 2))?;
    assert_eq!(
        decode_core.session_token_count(session)?,
        3 + first.token_ids.len()
    );
    assert_eq!(
        decode_core.session_prefill_carry(session)?,
        onnx_genai_engine::SessionPrefillCarry {
            attended: 3 + first.token_ids.len(),
            reprefilled: 0,
        },
        "the retained sequence is attended without being re-prefilled"
    );
    assert_eq!(decode_core.session_conversation(session)?, None);

    // An unknown session is reported as one rather than answered with a zero.
    decode_core.close_session(session)?;
    assert!(decode_core.session_prefill_carry(session).is_err());
    Ok(())
}

/// A lease the graph carries prepends nothing either.
#[test]
fn a_graph_carried_lease_does_not_prepend_a_conversation() -> anyhow::Result<()> {
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("graph_carried_gate_package");
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

    let mut engine = Engine::from_dir(&scratch, EngineConfig::default())?;
    assert!(
        !engine.prepends_session_conversation(),
        "a loop-carried lease lives in a cache the package bounds, not in front of a prompt"
    );
    let session = engine.create_session()?;
    assert_eq!(
        engine.session_prefill_carry(session)?,
        onnx_genai_engine::SessionPrefillCarry::default()
    );
    Ok(())
}
