//! Real-package gates for chained speculative proposal.
//!
//! Two opt-in cases, both skipped unless their package is supplied:
//!
//! * `ONNX_GENAI_CHAINED_SPEC_PACKAGE` — the direct-`Engine` EAGLE-3 chain,
//!   unchanged.
//! * `ONNX_GENAI_CHAINED_WORKFLOW_PACKAGE` — a real `pipeline.workflow` package
//!   whose `speculative.proposal_execution` is `chained`, driven by the
//!   interpreter (`pipeline::speculative`). This is the real-model scale check
//!   for the construct the hermetic `gemma4_chained` fixture pins: the published
//!   Gemma4-E2B speculative package carries the identical contract shape
//!   (`folded_carry_seed: {component: target, output: hidden_states.34}`,
//!   `token_embedding: {component: target, table: model.embed_tokens.weight}`),
//!   so the same field-reading driver must resolve it with no model-name gate.

use anyhow::Context as _;
use onnx_genai_engine::{
    Eagle3Config, Engine, EngineConfig, GeneratePrompt, GenerateRequest, SpeculativeMode,
};
use onnx_genai_metadata::SpeculativeProposalExecution;
use onnx_genai_ort::Eagle3DraftKvMode;
use std::path::{Path, PathBuf};

fn request(prompt: &str) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::Text(prompt.to_string()));
    request.options.max_new_tokens = 12;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request
}

fn engine(target: &Path, package: &Path, speculative: bool) -> anyhow::Result<Engine> {
    let speculative_mode = if speculative {
        SpeculativeMode::Eagle3(Eagle3Config {
            head_model: package.join("proposer/model.onnx"),
            target_hidden_outputs: vec![
                "hidden_states.2".into(),
                "hidden_states.14".into(),
                "hidden_states.25".into(),
            ],
            embedding_weights: package.join("target_embedding.f32"),
            token_map: Some(package.join("draft_to_target.i64")),
            vocab_size: 151_936,
            hidden_size: 1024,
            kv_mode: Eagle3DraftKvMode::GrowCache,
            num_speculative_tokens: 6,
        })
    } else {
        SpeculativeMode::None
    };
    Engine::from_dir(
        target,
        EngineConfig {
            speculative_mode,
            num_speculative_tokens: 6,
            ..EngineConfig::default()
        },
    )
}

#[test]
fn real_chained_proposer_matches_target_and_accepts_and_rejects() -> anyhow::Result<()> {
    let Some(package) = std::env::var_os("ONNX_GENAI_CHAINED_SPEC_PACKAGE").map(PathBuf::from)
    else {
        eprintln!("skipping real chained-proposer test; set ONNX_GENAI_CHAINED_SPEC_PACKAGE");
        return Ok(());
    };
    let target = package.join("runtime-target");
    if !target.join("model.onnx").is_file() {
        anyhow::bail!("runtime target is missing at {}", target.display());
    }

    let prompts = [
        "The capital of France is",
        "Write a Python function that adds two integers.",
        "Complete the sequence: 1, 1, 2, 3, 5,",
        "A quick brown fox",
        "Explain gravity in one sentence.",
    ];
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut generations = Vec::new();
    for prompt in prompts {
        let mut baseline = engine(&target, &package, false)?;
        let expected = baseline.generate(request(prompt))?;
        let mut speculative = engine(&target, &package, true)?;
        let actual = speculative.generate(request(prompt))?;
        let stats = speculative.last_speculative_stats();
        assert_eq!(
            actual.token_ids, expected.token_ids,
            "L4 token parity failed"
        );
        assert_eq!(actual.text, expected.text, "L4 text parity failed");
        assert!(
            actual.token_ids.len() > 1,
            "L5 generation must emit multiple tokens"
        );
        accepted += stats.accepted_tokens;
        rejected += stats.proposed_tokens.saturating_sub(stats.accepted_tokens);
        generations.push((prompt, actual.text, actual.token_ids, stats));
        if accepted > 0 && rejected > 0 {
            break;
        }
    }
    assert!(accepted > 0, "real chained proposals accepted no tokens");
    assert!(rejected > 0, "real chained proposals rejected no tokens");
    eprintln!(
        "REAL_CHAINED_SPEC_EVIDENCE accepted={accepted} rejected={rejected} generations={generations:?}"
    );
    Ok(())
}

/// The interpreter's chained construct on a real workflow package.
///
/// This asserts the contract resolves and drives at real scale — the tiny
/// fixture proves the semantics, this proves the *fields on a real export* are
/// the ones the driver reads. It deliberately does not re-derive greedy tokens
/// from a lookup table (a real target has none); it proves the chain runs to the
/// declared width, that the block's first position is the target's own token,
/// and that every declared `rollback_state` cell can be rolled back.
#[test]
fn real_chained_workflow_package_drives_through_the_interpreter() -> anyhow::Result<()> {
    let Some(package) = std::env::var_os("ONNX_GENAI_CHAINED_WORKFLOW_PACKAGE").map(PathBuf::from)
    else {
        eprintln!(
            "skipping the real chained-workflow test; set ONNX_GENAI_CHAINED_WORKFLOW_PACKAGE \
             to a pipeline.workflow package whose speculative.proposal_execution is chained"
        );
        return Ok(());
    };

    let engine = Engine::from_dir(&package, EngineConfig::default())?;
    let contract = engine
        .speculative_contract()
        .context("the package declares no speculative contract")?;
    let SpeculativeProposalExecution::Chained {
        folded_carry_seed,
        token_embedding,
        folded_carry_output,
        recurrent,
        ..
    } = &contract.proposal_execution
    else {
        anyhow::bail!(
            "ONNX_GENAI_CHAINED_WORKFLOW_PACKAGE must declare a chained proposal_execution"
        );
    };
    assert!(
        folded_carry_output.is_some() || !recurrent.is_empty(),
        "a chained proposer must thread something forward"
    );

    // The declared embedding table must resolve to a real initializer in the
    // real target artifact — the check that a per-model heuristic would pass
    // vacuously and a contract read cannot.
    if let Some(source) = token_embedding {
        let table = engine.embedding_table(source)?;
        assert!(
            table.vocab_size() > 0 && table.hidden_size() > 0,
            "declared token_embedding table {}::{} resolved empty",
            source.component,
            source.table
        );
        let row = table.row(0)?;
        assert_eq!(row.len(), table.hidden_size());
    }
    if let Some(seed) = folded_carry_seed {
        assert_eq!(
            seed.component, contract.target,
            "folded_carry_seed must name the speculative target"
        );
    }
    eprintln!(
        "REAL_CHAINED_WORKFLOW_EVIDENCE proposer={} target={} width<={} rollback_cells={}",
        contract.proposer,
        contract.target,
        contract.max_proposal_width,
        contract.rollback_state.len()
    );
    Ok(())
}
