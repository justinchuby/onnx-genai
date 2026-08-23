//! Executable evidence that a *published* chained-speculative package decodes.
//!
//! # What this proves, and why a fixture cannot
//!
//! `gemma4_chained_workflow` proves the interpreter's chained construct is
//! semantically right on a 32-token vocabulary whose target is a lookup table.
//! That is the correctness argument. It is not evidence that a *real* package
//! works, and the gap between them has been where the real failures lived:
//! an fp16 export's embedding table is fp16 and the chain demanded fp32; a
//! published package's serving controls have no defaults and admission demanded
//! them of the caller. Both pass a fixture and fail a package.
//!
//! So this case takes two published packages and runs them:
//!
//! * a composite package whose `speculative.proposal_execution` is `chained`
//!   (target + proposer under one root), driven through the interpreter;
//! * the standalone target package, decoded greedily.
//!
//! and requires the two token streams to be **identical**, at every proposal
//! width. Speculative decoding that changes the output is not an optimization.
//!
//! # Anti-vacuity
//!
//! A test that skips is a test that proves nothing, so this one is loud about
//! it. With `ONNX_GENAI_REQUIRE_SPECULATIVE_EVIDENCE=1` a missing package, an
//! unreadable directory, or a run that covered zero widths is a **failure**,
//! not a skip. Every assertion below is also written so that the vacuous
//! outcome fails: zero proposals fails, zero accepted drafts fails, zero
//! rejections fails, an embedding table that resolves empty fails, and a
//! `folded_carry_seed` the verification pass never produced fails.
//!
//! # Running it
//!
//! ```text
//! ONNX_GENAI_REQUIRE_SPECULATIVE_EVIDENCE=1 \
//! ONNX_GENAI_CHAINED_WORKFLOW_PACKAGE=<composite package dir> \
//! ONNX_GENAI_SPECULATIVE_TARGET_PACKAGE=<target-only package dir> \
//! cargo test -p onnx-genai-engine --features native-cuda \
//!   --test chained_speculative_real_evidence -- --nocapture
//! ```
//!
//! Both directories are ordinary Hugging Face snapshots; see
//! `docs/genai/CHAINED_SPECULATIVE_EVIDENCE.md` for the revisions this was
//! recorded against and how to fetch them.

use std::path::PathBuf;

use anyhow::Context as _;
use onnx_genai_engine::{Engine, EngineConfig};

#[path = "common/real_workflow.rs"]
mod real_workflow;

use real_workflow::{Fill, RealWorkflowPackage, SpeculativeTally};

/// Drafts per proposal the evidence covers.
///
/// 1 is the degenerate chain (a single speculated token); 2 and 4 make a block
/// long enough that a rejection can land in its middle rather than only at its
/// end, which is the case that has to roll state back.
const PROPOSAL_DRAFTS: [usize; 3] = [1, 2, 4];

/// Reciprocal of the least share of rounds whose first draft the target must
/// confirm. See the assertion that reads it for why this is the statistic that
/// carries the claim, and why the bar sits here.
const ACCEPTANCE_FLOOR_DENOMINATOR: usize = 4;

/// Tokens each run decodes. Long enough that the proposer disagrees with the
/// target somewhere, which is what makes the rejection path reachable.
const BUDGET: usize = 24;

/// The prompt, fixed so a rerun is comparable to a recorded one.
///
/// Deliberately open-ended: a prompt with one short answer is followed by turn
/// and end tokens, and a stream of those is trivially predictable — the
/// proposer would never be contradicted and the rejection path would never run.
const PROMPT: &str = "Once upon a time, in a small village near the mountains, there lived";

/// Chain steps a proposal of `drafts` speculated tokens costs.
///
/// A folded-carry chain's first step is a bootstrap: it reproduces the token
/// the target already handed over, and exists only to advance the carry past
/// it. That step drafts nothing, so `drafts` speculated tokens need one more
/// step than that. This is a property of the declared contract, not of any
/// particular model.
fn chain_width(drafts: usize) -> usize {
    drafts + 1
}

fn env_package(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
}

/// Whether a missing package fails instead of skipping.
fn evidence_required() -> bool {
    std::env::var("ONNX_GENAI_REQUIRE_SPECULATIVE_EVIDENCE")
        .map(|value| value != "0" && !value.is_empty())
        .unwrap_or(false)
}

/// Resolve a package directory, failing loudly when evidence was demanded.
fn package(key: &str) -> anyhow::Result<Option<PathBuf>> {
    match env_package(key) {
        Some(path) => {
            anyhow::ensure!(
                path.is_dir(),
                "{key} points at {}, which is not a directory; a real-package evidence run needs \
                 an extracted package, not an archive or a missing path",
                path.display()
            );
            Ok(Some(path))
        }
        None if evidence_required() => anyhow::bail!(
            "{key} is unset while ONNX_GENAI_REQUIRE_SPECULATIVE_EVIDENCE demands real-package \
             evidence; set it to an extracted package directory"
        ),
        None => {
            eprintln!("skipping real speculative evidence; set {key}");
            Ok(None)
        }
    }
}

fn engine(root: &std::path::Path) -> anyhow::Result<Engine> {
    Engine::from_dir(root, EngineConfig::default())
        .with_context(|| format!("failed to load the package at {}", root.display()))
}

/// The token id of the package's declared beginning-of-sequence token.
///
/// Read from the package's own `tokenizer_config.json`, never assumed: a model
/// whose training put a sentinel at position 0 produces nonsense without it —
/// this pair answers "The capital city of France is" with " France is France
/// is" unprompted and " Paris." once its declared `<bos>` is present — and a
/// model that declares none must not have one invented for it.
fn declared_bos(engine: &Engine, root: &std::path::Path) -> anyhow::Result<Option<i64>> {
    let path = root.join("tokenizer_config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)
        .with_context(|| format!("failed to read {}", path.display()))?;
    // Both spellings occur: a bare string, or the `AddedToken` object form.
    let token = match config.get("bos_token") {
        Some(serde_json::Value::String(token)) => token.clone(),
        Some(serde_json::Value::Object(fields)) => match fields.get("content") {
            Some(serde_json::Value::String(token)) => token.clone(),
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };
    let ids = engine.tokenize(&token)?;
    anyhow::ensure!(
        ids.len() == 1,
        "the package declares bos_token '{token}', which its own tokenizer encodes to \
         {ids:?}; a beginning-of-sequence marker must be exactly one token"
    );
    Ok(Some(i64::from(ids[0])))
}

/// Tokenize the prompt with the package's own tokenizer and declared BOS.
fn prompt_tokens(engine: &Engine, root: &std::path::Path) -> anyhow::Result<Vec<i64>> {
    let ids = engine
        .tokenize(PROMPT)
        .context("the package ships no usable tokenizer, so a fixed prompt cannot be encoded")?;
    anyhow::ensure!(
        !ids.is_empty(),
        "the evidence prompt encoded to zero tokens, so every run below would be vacuous"
    );
    let mut ids: Vec<i64> = ids.into_iter().map(i64::from).collect();
    if let Some(bos) = declared_bos(engine, root)?
        && ids.first() != Some(&bos)
    {
        ids.insert(0, bos);
    }
    Ok(ids)
}

/// The composite package decodes exactly what the target package decodes, at
/// every proposal width, and gets there by proposing, accepting and rejecting.
#[test]
fn a_published_chained_package_decodes_what_its_target_decodes() -> anyhow::Result<()> {
    let Some(composite_root) = package("ONNX_GENAI_CHAINED_WORKFLOW_PACKAGE")? else {
        return Ok(());
    };
    let Some(target_root) = package("ONNX_GENAI_SPECULATIVE_TARGET_PACKAGE")? else {
        return Ok(());
    };

    // ---- the reference: the standalone target, decoded greedily ----------
    let target_engine = engine(&target_root)?;
    let prompt = prompt_tokens(&target_engine, &target_root)?;
    let mut target = RealWorkflowPackage::new(target_engine, &target_root)?
        .fill("attention_mask", Fill::Ones)?;
    let reference = target.greedy_decode(&prompt, BUDGET)?;
    anyhow::ensure!(
        reference.len() == BUDGET,
        "the target package produced {} of {BUDGET} reference tokens",
        reference.len()
    );
    eprintln!(
        "REAL_SPECULATIVE_EVIDENCE reference package={} prompt_tokens={} tokens={reference:?}",
        target_root.display(),
        prompt.len()
    );
    drop(target);

    // ---- the composite package, driven through the interpreter -----------
    let composite_engine = engine(&composite_root)?;
    let contract = composite_engine
        .speculative_contract()
        .context(
            "the composite package declares no speculative contract, so there is no chained \
             proposal to drive",
        )?
        .clone();
    let onnx_genai_metadata::SpeculativeProposalExecution::Chained {
        folded_carry_seed,
        token_embedding,
        folded_carry_output,
        recurrent,
        ..
    } = &contract.proposal_execution
    else {
        anyhow::bail!(
            "ONNX_GENAI_CHAINED_WORKFLOW_PACKAGE must declare a chained proposal_execution; this \
             evidence is about the chained construct"
        );
    };
    anyhow::ensure!(
        folded_carry_output.is_some() || !recurrent.is_empty(),
        "a chained proposer must thread something forward"
    );

    // The declared embedding table must resolve to real weights in the real
    // artifact, in the element type the proposer's fused input is declared in.
    // A fixture-shaped bypass — an f32 side-file, a synthesized table — would
    // pass a shape check and prove nothing about this package.
    let embedding_source = token_embedding
        .clone()
        .context("a folded-carry proposer must declare token_embedding")?;
    let table = composite_engine.embedding_table(&embedding_source)?;
    anyhow::ensure!(
        table.vocab_size() > 0 && table.hidden_size() > 0,
        "declared token_embedding table {}::{} resolved empty",
        embedding_source.component,
        embedding_source.table
    );
    let row = table.row(0)?;
    anyhow::ensure!(
        row.len() == table.hidden_size() && row.iter().any(|value| *value != 0.0),
        "the declared embedding table's first row is {} wide and entirely zero; a real gather \
         path reads real weights",
        row.len()
    );
    let embedding_dtype = table.dtype();
    let seed = folded_carry_seed
        .clone()
        .context("this evidence requires a declared folded_carry_seed")?;
    anyhow::ensure!(
        seed.component == contract.target,
        "folded_carry_seed must name the speculative target"
    );
    eprintln!(
        "REAL_SPECULATIVE_EVIDENCE contract proposer={} target={} max_width={} \
         rollback_cells={} embedding={}::{} [{}, {}] {embedding_dtype:?} \
         folded_carry_seed={}::{}",
        contract.proposer,
        contract.target,
        contract.max_proposal_width,
        contract.rollback_state.len(),
        embedding_source.component,
        embedding_source.table,
        table.vocab_size(),
        table.hidden_size(),
        seed.component,
        seed.output,
    );
    drop(table);

    let mut composite = RealWorkflowPackage::new(composite_engine, &composite_root)?
        .fill("attention_mask", Fill::Ones)?;
    eprintln!(
        "REAL_SPECULATIVE_EVIDENCE resolved_symbols={:?}",
        composite.graph_symbols()
    );

    // The seed the chain folds forward must be a value the verification pass
    // actually produces; a package naming an output nobody emits would drive a
    // chain off a zero tensor and still look like it worked.
    let probe = composite.run(&prompt)?;
    let seed_value = composite
        .workflow()
        .components
        .get(&seed.component)
        .and_then(|_| {
            composite
                .workflow()
                .steps
                .iter()
                .find_map(|step| match step {
                    onnx_genai_metadata::WorkflowStep::Invoke {
                        component, outputs, ..
                    } if *component == seed.component => outputs.get(&seed.output),
                    _ => None,
                })
        })
        .context("the workflow binds no value to the declared folded_carry_seed output")?;
    let carry = probe
        .get(seed_value)
        .with_context(|| format!("the verification pass did not produce '{seed_value}'"))?;
    anyhow::ensure!(
        carry.shape().len() == 3 && carry.shape()[1] as usize == prompt.len(),
        "the folded carry seed '{seed_value}' has shape {:?}, which is not a per-position hidden \
         state over the {}-token context it was produced from",
        carry.shape(),
        prompt.len()
    );
    // Which residency regime this run is in. It is reported rather than
    // assumed: an ORT-hosted component may publish its outputs host-side, and
    // the residency claim below has to be read against what the pass actually
    // produced instead of implying a device the values were never on.
    let carry_on_host = carry.is_host_resident()?;
    eprintln!(
        "REAL_SPECULATIVE_EVIDENCE folded_carry_seed value={seed_value} shape={:?} dtype={:?} \
         host_resident={carry_on_host}",
        carry.shape(),
        carry.dtype()
    );
    drop(probe);

    // ---- the width matrix ------------------------------------------------
    // #1861 made a chained proposal compute where its tensors already are. That
    // is a property of the *runtime*, so a package with a real vocabulary and a
    // real cache has to keep it: these two counters account for every
    // device -> host byte the interpreter can produce, and the only one a chain
    // is allowed is the token id an argmax selects.
    let staging_before = composite.engine().host_staging_count();
    let readback_before = composite.engine().device_readback_bytes();
    let mut covered = 0usize;
    let mut totals = SpeculativeTally::default();
    for drafts in PROPOSAL_DRAFTS {
        let width = chain_width(drafts);
        anyhow::ensure!(
            width <= contract.max_proposal_width,
            "the package declares max_proposal_width {} but {drafts} drafts need a chain of \
             {width}; a narrower package cannot carry this claim",
            contract.max_proposal_width
        );
        let (tokens, tally) = composite
            .speculative_decode(&prompt, BUDGET, width)
            .with_context(|| {
                format!("speculative decode at {drafts} drafts per proposal failed")
            })?;
        eprintln!(
            "REAL_SPECULATIVE_EVIDENCE drafts_per_proposal={drafts} chain_width={width} \
             tokens={tokens:?} rounds={} rounds_with_an_accepted_draft={} proposed={} \
             accepted={} rejected={} rejections={} full_accepts={} \
             proposer_invocations={} rolled_back_cells={}",
            tally.rounds,
            tally.rounds_with_an_accepted_draft,
            tally.proposed,
            tally.accepted_drafts,
            tally.rejected_drafts,
            tally.rejections,
            tally.full_accepts,
            tally.proposer_invocations,
            tally.rolled_back_cells,
        );
        // What this does and does not prove. A verified block commits the
        // *target's* tokens whatever the proposer said, so an identical stream
        // is evidence that the composite package's target and the standalone
        // target package are the same model — necessary, and not sufficient.
        // The claim that the chain does anything rests on the acceptance
        // statistics below, which a broken chain cannot fake.
        anyhow::ensure!(
            tokens == reference,
            "{drafts} drafts per proposal decoded {tokens:?}, but the target package decodes \
             {reference:?}; speculative decoding that changes the output is not an optimization"
        );
        anyhow::ensure!(
            tally.proposed > 0 && tally.proposer_invocations > 0,
            "{drafts} drafts per proposal made no proposals at all: {tally:?}"
        );
        anyhow::ensure!(
            tally.accepted_drafts > 0,
            "{drafts} drafts per proposal accepted nothing, so the chain never did any work the \
             target did not have to redo: {tally:?}"
        );
        // The bar a chain has to clear, and the one every defect this gate
        // exists for fell under. A proposer fed an empty borrowed cache, or an
        // unscaled embedding row, still drafts fluent tokens — it just drafts
        // ones conditioned on nothing, and the target contradicts all of them:
        // measured at 0 of 12 rounds before those two fixes and 4 of 10 after,
        // at every width. One round in four is set below the second and far
        // above the first, so it separates a working chain from a plausible
        // one without pinning a number a model change would have to match.
        anyhow::ensure!(
            tally.rounds_with_an_accepted_draft * ACCEPTANCE_FLOOR_DENOMINATOR >= tally.rounds,
            "{drafts} drafts per proposal had the target confirm a first draft in only {} of {} \
             rounds; a chain conditioned on the wrong tensors still proposes, still tallies, and \
             still emits the target's own tokens — a proposal rate below one round in \
             {ACCEPTANCE_FLOOR_DENOMINATOR} is what that looks like: {tally:?}",
            tally.rounds_with_an_accepted_draft,
            tally.rounds
        );
        totals.rounds += tally.rounds;
        totals.rounds_with_an_accepted_draft += tally.rounds_with_an_accepted_draft;
        totals.proposed += tally.proposed;
        totals.accepted_drafts += tally.accepted_drafts;
        totals.rejected_drafts += tally.rejected_drafts;
        totals.rejections += tally.rejections;
        totals.full_accepts += tally.full_accepts;
        totals.proposer_invocations += tally.proposer_invocations;
        totals.rolled_back_cells += tally.rolled_back_cells;
        covered += 1;
    }

    anyhow::ensure!(
        covered == PROPOSAL_DRAFTS.len(),
        "only {covered} of {} proposal widths ran; partial coverage is not evidence",
        PROPOSAL_DRAFTS.len()
    );
    // Across the matrix both branches must have run. A chain that only ever
    // accepted would leave rollback — the expensive, easily-broken half —
    // completely unexercised.
    anyhow::ensure!(
        totals.rejections > 0 && totals.rolled_back_cells > 0,
        "no proposal was ever rejected across proposal widths {PROPOSAL_DRAFTS:?}, so rollback \
         never ran: {totals:?}"
    );
    anyhow::ensure!(
        totals.full_accepts > 0,
        "no proposal block was ever fully accepted across proposal widths \
         {PROPOSAL_DRAFTS:?}: {totals:?}"
    );
    let staging = composite.engine().host_staging_count() - staging_before;
    let readback = composite.engine().device_readback_bytes() - readback_before;
    eprintln!(
        "REAL_SPECULATIVE_EVIDENCE residency host_staging={staging} readback_bytes={readback} \
         proposer_invocations={}",
        totals.proposer_invocations
    );
    anyhow::ensure!(
        staging == 0,
        "the proposal chain materialized {staging} device tensors on the host; a device-resident \
         chain brings nothing down but the token ids it selects"
    );
    // An upper bound, not an equality: a pass whose values are already
    // host-resident has no device read to make, and counting zero there is
    // correct rather than suspicious. What must never happen -- in either
    // regime -- is a chain bringing down more than the id it selected.
    anyhow::ensure!(
        readback as usize <= totals.proposer_invocations * std::mem::size_of::<u32>(),
        "the chain read {readback} bytes back off the device across {} proposer invocations; a \
         device-resident chain reads back at most one {}-byte token id per invocation",
        totals.proposer_invocations,
        std::mem::size_of::<u32>()
    );
    anyhow::ensure!(
        composite.engine().embedding_table_loads() <= 2,
        "the embedding table was read out of the artifact {} times; a real vocabulary is read \
         once per residency and mirrored, never per proposal",
        composite.engine().embedding_table_loads()
    );
    eprintln!(
        "REAL_SPECULATIVE_EVIDENCE totals drafts_per_proposal={PROPOSAL_DRAFTS:?} \
         budget={BUDGET} {totals:?} embedding_table_loads={}",
        composite.engine().embedding_table_loads()
    );
    Ok(())
}
