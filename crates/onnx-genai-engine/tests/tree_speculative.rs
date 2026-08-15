//! End-to-end greedy-equivalence proof for tree-structured speculative decoding.
//!
//! The tree-speculation core (`onnx_genai_engine::speculative::tree`) is verified
//! here against a *real* ONNX fixture model. We drive a full tree-speculative decode
//! loop whose per-node verification logits come from the real target model and assert
//! the committed token sequence is byte-for-byte identical to a plain greedy decode.
//!
//! This exercises tree construction, the ancestor attention semantics (each node is
//! scored with exactly its root-to-node path context), the acceptance walk, the bonus
//! token, and the KV-retention plan (retained length == accepted path length) — the
//! single most important invariant: speculation changes throughput, never output.

use onnx_genai_engine::speculative::{
    AcceptanceRule, SpecTree, SpecTreeBuilder, TreeScorer, verify_tree,
};
use onnx_genai_engine::{
    Engine, EngineConfig, GeneratePrompt, GenerateRequest, SpeculativeMode, TokenId,
};
use onnx_genai_ort::SessionOptions;
use std::path::{Path, PathBuf};

fn tiny_llm() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm")
        .canonicalize()?)
}

fn deterministic_engine(fixture: &Path) -> anyhow::Result<Engine> {
    Engine::from_dir_with_session_options(
        fixture,
        EngineConfig {
            speculative_mode: SpeculativeMode::None,
            ..Default::default()
        },
        SessionOptions::default().with_intra_op_threads(1),
    )
}

fn greedy_request(tokens: Vec<TokenId>, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request
}

/// The target model's greedy next token for a given committed context.
fn greedy_next(engine: &mut Engine, context: &[TokenId]) -> anyhow::Result<TokenId> {
    let result = engine.generate(greedy_request(context.to_vec(), 1))?;
    result
        .token_ids
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("model produced no token"))
}

/// One-hot logits whose deterministic argmax is `token`, matching the real model's
/// greedy decision. Sizing to `token + 1` keeps the argmax unambiguous.
fn onehot(token: TokenId) -> Vec<f32> {
    let mut logits = vec![0.0_f32; token as usize + 1];
    logits[token as usize] = 1.0;
    logits
}

/// Scores tree nodes by asking the real target model for its greedy next token given
/// the committed context concatenated with the node's ancestor path. This is exactly
/// the per-node context a correct tree attention mask would supply.
struct RealModelScorer<'a> {
    engine: &'a mut Engine,
    committed: Vec<TokenId>,
}

impl TreeScorer for RealModelScorer<'_> {
    fn score(&mut self, path: &[TokenId]) -> anyhow::Result<Vec<f32>> {
        let mut context = self.committed.clone();
        context.extend_from_slice(path);
        Ok(onehot(greedy_next(self.engine, &context)?))
    }
}

/// Build a branching draft tree: the model's true greedy chain of `depth` tokens plus
/// two decoy siblings per slot that the target must reject. The truthful chain lets
/// the acceptance walk accept multiple tokens; the decoys prove sibling branches are
/// scored independently and discarded.
fn build_draft_tree(
    engine: &mut Engine,
    committed: &[TokenId],
    depth: usize,
) -> anyhow::Result<SpecTree> {
    let mut builder = SpecTreeBuilder::new(Some(64));
    let mut context = committed.to_vec();
    let mut parent: Option<usize> = None;
    for _ in 0..depth {
        let truth = greedy_next(engine, &context)?;
        let node = match parent {
            None => builder.add_root(truth)?,
            Some(p) => builder.add_child(p, truth)?,
        };
        for flip in [1_u32, 2_u32] {
            // XOR keeps the decoy within the model's 32-token vocab and distinct
            // from the true greedy token, so it is always rejected.
            let decoy = truth ^ flip;
            if decoy != truth {
                match parent {
                    None => {
                        builder.add_root(decoy)?;
                    }
                    Some(p) => {
                        builder.add_child(p, decoy)?;
                    }
                }
            }
        }
        context.push(truth);
        parent = Some(node);
    }
    Ok(builder.build())
}

fn run_equivalence(prompt: Vec<TokenId>, target_steps: usize, depth: usize) -> anyhow::Result<()> {
    let fixture = tiny_llm()?;

    let mut reference_engine = deterministic_engine(&fixture)?;
    let expected = reference_engine
        .generate(greedy_request(prompt.clone(), target_steps))?
        .token_ids;

    let mut engine = deterministic_engine(&fixture)?;
    let mut committed = prompt.clone();
    let mut produced: Vec<TokenId> = Vec::new();
    let mut saw_branching = false;
    let mut saw_multi_accept = false;

    while produced.len() < target_steps {
        let base_len = committed.len();
        let tree = build_draft_tree(&mut engine, &committed, depth)?;
        // A real branching tree, not a linear chain.
        saw_branching |= tree.len() > tree.nodes().iter().filter(|n| n.parent.is_none()).count();

        let mut scorer = RealModelScorer {
            engine: &mut engine,
            committed: committed.clone(),
        };
        let verification = verify_tree(&tree, base_len, AcceptanceRule::Greedy, &mut scorer)?;

        // KV compaction retains exactly the accepted path.
        assert_eq!(
            verification.plan.final_len,
            base_len + verification.outcome.nodes.len(),
            "retained KV length must equal accepted path length",
        );
        assert_eq!(
            verification.plan.retained_nodes, verification.outcome.nodes,
            "retained nodes must be the accepted path",
        );
        if verification.outcome.nodes.len() >= 2 {
            saw_multi_accept = true;
        }

        assert!(
            !verification.outcome.tokens.is_empty(),
            "each step must commit at least the bonus token",
        );
        for token in verification.outcome.tokens {
            if produced.len() == target_steps {
                break;
            }
            committed.push(token);
            produced.push(token);
        }
    }

    assert_eq!(
        produced, expected,
        "tree-speculative greedy decode must match plain greedy decode exactly",
    );
    assert!(saw_branching, "the draft tree should actually branch");
    assert!(
        saw_multi_accept,
        "the truthful chain should yield at least one multi-token accept",
    );
    Ok(())
}

#[test]
fn tree_speculative_greedy_matches_plain_greedy_exactly() -> anyhow::Result<()> {
    run_equivalence(vec![3, 26, 11], 6, 3)
}

#[test]
fn tree_speculative_greedy_matches_plain_greedy_repetitive_prompt() -> anyhow::Result<()> {
    run_equivalence(vec![3, 26, 11, 9, 29, 3], 4, 3)
}
