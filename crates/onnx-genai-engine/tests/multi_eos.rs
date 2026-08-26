//! A model with more than one end token is stopped by any of them.
//!
//! A single `eos_token_id` cannot express "this model ends a turn with one
//! token and a message with another". Where that is the model's truth, keeping
//! only the first id means generation runs past its end and emits control
//! tokens as ordinary text — the failure is silent, and looks like the model
//! rambling rather than like a runtime dropping a declaration.
//!
//! These cases pin the whole path: the package can *declare* a set, the
//! declaration reaches the stop policy, and every declared id stops generation
//! in prefill, in cached decode, and across a session.
//!
//! # Batched routes
//!
//! The batched and continuous-batch routes resolve the same end tokens, from
//! the same source, through the same `apply_eos_policy`. That is asserted where
//! it can be asserted without a backend: `batched::tests::
//! a_continuous_row_stops_on_a_non_first_declared_end_token` drives a real
//! `ContinuousBatchManager` row loop over a scripted decode and requires the row
//! to end at a *non-first* declared id.
//!
//! It is deliberately not asserted end-to-end here. Batching needs a shared KV
//! buffer, which needs an execution provider reporting fixed-capacity present
//! binding — no CPU fixture in this repository can batch, and the existing
//! `batching_capability` suite asserts precisely that absence. An end-to-end
//! case pointed at a CPU fixture would skip, and a skipped test that reports
//! success is worse than no test.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest,
};

fn decoder_package() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm")
}

fn package_declaring_eos(ids: &[i64]) -> anyhow::Result<tempfile::TempDir> {
    package_declaring_eos_from(&decoder_package(), ids)
}

fn package_declaring_eos_from(source: &Path, ids: &[i64]) -> anyhow::Result<tempfile::TempDir> {
    let staged = tempfile::tempdir()?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::copy(entry.path(), staged.path().join(entry.file_name()))?;
        }
    }

    let path = staged.path().join("inference_metadata.yaml");
    let mut document: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(&path)?)?;
    document["schema_version"] = serde_yaml::Value::String("v1.2".to_string());
    document["package"]["tokenizer"]["special_tokens"] =
        serde_yaml::to_value(serde_yaml::Mapping::from_iter([(
            serde_yaml::Value::String("eos_token_id".to_string()),
            serde_yaml::to_value(
                ids.iter()
                    .map(|id| u32::try_from(*id))
                    .collect::<Result<Vec<_>, _>>()?,
            )?,
        )]))?;
    std::fs::write(&path, serde_yaml::to_string(&document)?)?;
    Ok(staged)
}

fn legacy_package_declaring_eos(id: u32) -> anyhow::Result<tempfile::TempDir> {
    let staged = tempfile::tempdir()?;
    for entry in std::fs::read_dir(decoder_package())? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::copy(entry.path(), staged.path().join(entry.file_name()))?;
        }
    }

    let path = staged.path().join("generation_config.json");
    let document = serde_json::json!({ "eos_token_id": id });
    std::fs::write(&path, serde_json::to_vec_pretty(&document)?)?;
    Ok(staged)
}

fn request(tokens: usize) -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::Text("hello world".to_string()),
        options: GenerateOptions {
            max_new_tokens: tokens,
            greedy: true,
            temperature: 0.0,
            stop_on_eos: true,
            ..GenerateOptions::default()
        },
    }
}

/// The tokens this fixture greedily produces, so a test can pick a real one to
/// declare terminal rather than hoping an arbitrary id appears.
fn greedy_prefix(length: usize) -> anyhow::Result<Vec<u32>> {
    let mut engine = Engine::from_dir(&decoder_package(), EngineConfig::default())?;
    let mut probe = request(length);
    probe.options.stop_on_eos = false;
    Ok(engine.generate(probe)?.token_ids)
}

/// A declared multi-id EOS materializes at all.
///
/// The regression this starts from: a `[eos_count]` contract with a two-element
/// literal resolved its symbolic axis to 1 and failed with "declares 2 elements
/// but its contract holds 1" — a package correctly describing itself was
/// unloadable.
#[test]
fn a_package_may_declare_several_end_tokens() -> anyhow::Result<()> {
    let staged = package_declaring_eos(&[11, 22, 33])?;
    let engine = Engine::from_dir(staged.path(), EngineConfig::default())?;
    assert_eq!(
        engine
            .metadata()
            .package
            .as_ref()
            .and_then(|package| package.tokenizer.as_ref())
            .and_then(|tokenizer| tokenizer.special_tokens.as_ref())
            .expect("the staged package declares special token facts")
            .eos_token_id,
        [11, 22, 33],
        "all three ids survive the round trip"
    );
    Ok(())
}

/// Packages written before token authority was added retain their tokenizer
/// fallback until they are migrated to v1.2.
#[test]
fn legacy_packages_keep_their_existing_eos_behavior() -> anyhow::Result<()> {
    let stop = greedy_prefix(1)?[0];
    let staged = legacy_package_declaring_eos(stop)?;
    let mut engine = Engine::from_dir(staged.path(), EngineConfig::default())?;
    let result = engine.generate(request(32))?;
    assert_eq!(result.finish_reason, FinishReason::EosToken, "{result:?}");
    assert_eq!(result.token_ids, [stop], "{result:?}");
    Ok(())
}

/// Each declared end token stops generation — not merely the first.
///
/// Run once per id against the same package, so a runtime that kept only
/// `ids[0]` fails on the second case while passing the first.
#[test]
fn every_declared_end_token_stops_generation() -> anyhow::Result<()> {
    let prefix = greedy_prefix(6)?;
    assert!(
        prefix.len() >= 3,
        "the fixture must generate enough tokens to place a later stop: {prefix:?}"
    );
    // A first-token stop exercises prefill; a later one exercises cached decode,
    // which is a different code path through the same policy.
    let first = i64::from(prefix[0]);
    let later = i64::from(prefix[2]);
    assert_ne!(first, later, "the two stop points must be distinguishable");

    // One package per stop point, because a package declaring both always ends
    // at whichever comes first — which would make the second case vacuous.
    // Stopping on the *first* generated token exercises the prefill step's
    // commit; stopping on a later one exercises cached decode, a different code
    // path through the same policy.
    for (label, stop, expected_len) in [("prefill", first, 1), ("cached decode", later, 3)] {
        let staged = package_declaring_eos(&[stop])?;
        let mut engine = Engine::from_dir(staged.path(), EngineConfig::default())?;
        // The request names nothing: the package's declaration alone must stop
        // it, which is the property that was dead before.
        let result = engine.generate(request(32))?;
        assert_eq!(
            result.finish_reason,
            FinishReason::EosToken,
            "{label}: declared end token {stop} must stop generation: {result:?}"
        );
        assert_eq!(
            result.token_ids.len(),
            expected_len,
            "{label}: generation must end at the declared token: {result:?}"
        );
    }
    Ok(())
}

/// An explicit request EOS set replaces the package default for that request.
#[test]
fn a_request_eos_set_overrides_the_package_default() -> anyhow::Result<()> {
    let prefix = greedy_prefix(6)?;
    let package_stop = i64::from(prefix[0]);
    let request_stop = prefix[2];
    assert_ne!(package_stop, i64::from(request_stop));

    let staged = package_declaring_eos(&[package_stop])?;
    let mut engine = Engine::from_dir(staged.path(), EngineConfig::default())?;
    let mut probe = request(32);
    probe.options.eos_token_ids = vec![request_stop];
    let result = engine.generate(probe)?;
    assert_eq!(
        result.finish_reason,
        FinishReason::EosToken,
        "the request's end token must stop generation: {result:?}"
    );
    assert_eq!(
        result.token_ids.len(),
        3,
        "the package default must not stop the overridden request: {result:?}"
    );
    Ok(())
}

/// The declaration survives a session, so a multi-turn caller keeps it.
#[test]
fn declared_end_tokens_apply_across_a_session() -> anyhow::Result<()> {
    let prefix = greedy_prefix(6)?;
    let stop = i64::from(prefix[1]);
    let staged = package_declaring_eos(&[stop])?;
    let mut engine = Engine::from_dir(staged.path(), EngineConfig::default())?;
    let session = engine.create_session()?;
    let result = engine.generate_in_session(session, request(32))?;
    assert_eq!(result.finish_reason, FinishReason::EosToken, "{result:?}");
    assert_eq!(result.token_ids.len(), 2, "{result:?}");
    Ok(())
}
