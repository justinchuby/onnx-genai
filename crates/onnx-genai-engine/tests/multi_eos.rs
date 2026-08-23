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
use onnx_genai_metadata::decoder_workflow::PACKAGE_EOS_TOKEN_IDS;

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
    let declaration: serde_yaml::Value = serde_yaml::from_str(&format!(
        "contract: {{dtype: int64, rank: 1, shape: [eos_count]}}\n\
         role: {{kind: runtime, version: '1.0', role: eos_token_ids}}\n\
         source: {{kind: literal}}\n\
         required: false\n\
         externally_suppliable: true\n\
         default: [{}]\n",
        ids.iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    ))?;
    document["pipeline"]["workflow"]["inputs"]
        .as_mapping_mut()
        .expect("workflow inputs")
        .insert(
            serde_yaml::Value::String(PACKAGE_EOS_TOKEN_IDS.to_string()),
            declaration,
        );
    std::fs::write(&path, serde_yaml::to_string(&document)?)?;
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
    let workflow = engine
        .package_workflow()
        .expect("a loaded package declares a workflow");
    let declared = workflow
        .inputs
        .get(PACKAGE_EOS_TOKEN_IDS)
        .expect("the staged package declares its end tokens");
    match declared.default.as_ref().expect("a literal default") {
        onnx_genai_metadata::LiteralValue::Elements(elements) => {
            assert_eq!(elements.len(), 3, "all three ids survive the round trip");
        }
        other => panic!("expected an element list, got {other:?}"),
    }
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

/// A second declared end token stops even when the request names the first.
///
/// This is the Muse-Glimmer shape: `<|eot|>` and `<|eom|>` are both terminal,
/// and a caller naming one must not resurrect the other as ordinary text.
#[test]
fn a_request_naming_one_end_token_does_not_disarm_the_others() -> anyhow::Result<()> {
    let prefix = greedy_prefix(6)?;
    let unreachable = i64::from(u16::MAX);
    let real_stop = i64::from(prefix[1]);
    assert_ne!(real_stop, unreachable);

    let staged = package_declaring_eos(&[unreachable, real_stop])?;
    let mut engine = Engine::from_dir(staged.path(), EngineConfig::default())?;
    let mut probe = request(32);
    // The request names the id the model never emits. The *other* declared id
    // must still stop it.
    probe.options.eos_token_id = Some(unreachable as u32);
    let result = engine.generate(probe)?;
    assert_eq!(
        result.finish_reason,
        FinishReason::EosToken,
        "the model's other end token must still stop generation: {result:?}"
    );
    assert_eq!(result.token_ids.len(), 2, "{result:?}");
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
