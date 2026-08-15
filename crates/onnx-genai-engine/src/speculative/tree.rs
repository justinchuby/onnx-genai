//! Tree-structured speculative decoding core (DESIGN §3.5, `Topology::Tree`).
//!
//! The linear speculative path drafts a single chain of `K` tokens and verifies
//! it in one target forward pass. Tree speculation generalises the *shape* of the
//! draft: instead of one chain, the producer emits a branching tree of candidate
//! continuations, all verified against the target with the correct causal context
//! and the single best root-to-leaf path accepted.
//!
//! This module is deliberately self-contained and free of any ORT/session state.
//! It provides the reusable primitives shared by both topologies:
//!
//! * [`SpecTree`] / [`TreeNode`] — an explicit parent-pointer tree representation.
//! * [`ancestor_attention_mask`] — the tree attention mask: every node attends to
//!   its ancestors (its root-to-node path) and itself, so one forward pass scores
//!   each node with exactly the context of its own path.
//! * [`relative_position_ids`] — position ids equal to node depth (so RoPE sees the
//!   path length, not the flattened index).
//! * [`SpecTree::accept`] — the acceptance walk over the tree using an
//!   [`AcceptanceRule`], following the longest accepted root-to-leaf path and
//!   extending it by the target's bonus token.
//! * [`kv_retention_plan`] / [`KvRetentionPlan`] — the KV compaction plan that keeps
//!   only the accepted path's nodes and discards every rejected branch.
//! * [`TreeScorer`] / [`verify_tree`] — a scorer abstraction that turns per-node
//!   context into logits, plus the end-to-end verify+accept driver. In production a
//!   scorer is one batched masked forward; in tests it can wrap a reference model to
//!   prove greedy equivalence.
//!
//! The single most important invariant, proven by [`verify_tree`] under
//! [`AcceptanceRule::Greedy`], is that tree speculation returns *exactly* the token
//! sequence a plain greedy decode would produce — speculation is a throughput
//! optimisation, never an output change.

use crate::TokenId;
use crate::speculative::{AcceptanceRule, argmax};

/// Shape of a speculative draft: a single linear chain, or a branching tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Topology {
    /// A single chain of `K` draft tokens (the classic speculative path).
    #[default]
    Linear,
    /// A branching tree of candidate continuations verified in one pass.
    Tree,
}

/// A single node in a speculative draft tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeNode {
    /// Candidate token this node proposes.
    pub token: TokenId,
    /// Index of the parent node in [`SpecTree::nodes`], or `None` for a root.
    ///
    /// Invariant: `parent < self_index`, i.e. every parent precedes its children
    /// in the flattened node vector (a valid topological order).
    pub parent: Option<usize>,
    /// Distance from the (conceptual) committed anchor. Roots have depth `0`.
    ///
    /// Sibling candidates for the same slot share a depth, and therefore a
    /// position id — they are alternative tokens for the same position.
    pub depth: usize,
}

/// An explicit parent-pointer speculative draft tree.
///
/// Nodes are stored in a flattened vector in topological order (parents before
/// children). The committed context is a conceptual anchor that is *not* stored as
/// a node; root nodes ([`TreeNode::parent`] is `None`) attach directly to it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecTree {
    nodes: Vec<TreeNode>,
}

impl SpecTree {
    /// All nodes in flattened topological order.
    pub fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    /// Number of candidate nodes (excludes the conceptual anchor).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree has no candidate nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Indices of the root nodes (those attached to the committed anchor).
    pub fn roots(&self) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|&i| self.nodes[i].parent.is_none())
            .collect()
    }

    /// Indices of the direct children of `node`.
    pub fn children(&self, node: usize) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|&i| self.nodes[i].parent == Some(node))
            .collect()
    }

    /// Proper ancestors of `node`, ordered from the root down to `node`'s parent.
    pub fn ancestors(&self, node: usize) -> Vec<usize> {
        let mut chain = Vec::new();
        let mut cursor = self.nodes[node].parent;
        while let Some(parent) = cursor {
            chain.push(parent);
            cursor = self.nodes[parent].parent;
        }
        chain.reverse();
        chain
    }

    /// Token path from the root down to and including `node`.
    pub fn path_tokens(&self, node: usize) -> Vec<TokenId> {
        let mut path: Vec<TokenId> = self
            .ancestors(node)
            .into_iter()
            .map(|i| self.nodes[i].token)
            .collect();
        path.push(self.nodes[node].token);
        path
    }

    /// Longest accepted root-to-leaf path plus the target's bonus token.
    ///
    /// `base_logits` is the target's prediction at the committed anchor (it verifies
    /// the root candidates). `node_logits[i]` is the target's prediction produced at
    /// node `i`'s slot — it verifies node `i`'s children, or supplies the bonus token
    /// when `i` is the last accepted node. See [`verify_tree`] for how these are
    /// produced from a [`TreeScorer`].
    ///
    /// Returns the accepted `(nodes, tokens)`: `nodes` is the accepted path (may be
    /// empty when the very first target token matches no candidate), and `tokens` is
    /// the committed continuation — the accepted path's tokens followed by exactly
    /// one bonus/correction token. Its length is therefore `1..=path_len + 1`.
    pub fn accept(
        &self,
        rule: AcceptanceRule,
        base_logits: &[f32],
        node_logits: &[Vec<f32>],
    ) -> anyhow::Result<AcceptOutcome> {
        assert_eq!(
            node_logits.len(),
            self.nodes.len(),
            "node_logits must have one row per tree node",
        );
        let mut accepted_nodes = Vec::new();
        let mut current = base_logits;
        let mut frontier = self.roots();
        loop {
            let decision = target_decision(current, rule)?;
            let matched = frontier
                .iter()
                .copied()
                .find(|&n| self.nodes[n].token == decision.token);
            match matched {
                Some(node) if decision.accept => {
                    accepted_nodes.push(node);
                    current = &node_logits[node];
                    frontier = self.children(node);
                }
                _ => {
                    return Ok(AcceptOutcome {
                        tokens: accepted_nodes
                            .iter()
                            .map(|&i| self.nodes[i].token)
                            .chain(std::iter::once(decision.token))
                            .collect(),
                        nodes: accepted_nodes,
                    });
                }
            }
        }
    }
}

/// Result of an acceptance walk over a [`SpecTree`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptOutcome {
    /// Accepted node indices, ordered root-to-leaf along the accepted path.
    pub nodes: Vec<usize>,
    /// Committed continuation: accepted path tokens plus one bonus/correction token.
    pub tokens: Vec<TokenId>,
}

/// A target token together with whether a matching candidate may be accepted.
struct TargetDecision {
    token: TokenId,
    accept: bool,
}

fn target_decision(logits: &[f32], rule: AcceptanceRule) -> anyhow::Result<TargetDecision> {
    let index = argmax(logits).ok_or_else(|| anyhow::anyhow!("target logits were empty"))?;
    let token = TokenId::try_from(index).map_err(|_| anyhow::anyhow!("token id exceeds u32"))?;
    // Speculative decoding only runs at temperature 0, so the target is
    // deterministic and `argmax` is the target's sampled token. Greedy and
    // rejection sampling therefore accept any candidate that equals it; typical
    // acceptance additionally gates on the target probability mass.
    let accept = match rule {
        AcceptanceRule::Greedy | AcceptanceRule::RejectionSampling => true,
        AcceptanceRule::Typical { threshold } => softmax_prob(logits, index) >= threshold,
    };
    Ok(TargetDecision { token, accept })
}

fn softmax_prob(logits: &[f32], index: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    if sum == 0.0 {
        0.0
    } else {
        (logits[index] - max).exp() / sum
    }
}

/// The tree attention mask: `mask[q][k] == true` iff key node `k` is on query node
/// `q`'s root-to-node path (i.e. `k` is an ancestor of `q`, or `k == q`).
///
/// The mask governs attention *among draft nodes*; every node additionally attends
/// to the full committed past KV, which the target forward supplies separately. With
/// this mask a single forward pass scores every node with exactly the causal context
/// of its own path — no sibling branch leaks across.
pub fn ancestor_attention_mask(tree: &SpecTree) -> Vec<Vec<bool>> {
    let n = tree.len();
    let mut mask = vec![vec![false; n]; n];
    for (q, row) in mask.iter_mut().enumerate() {
        row[q] = true;
        for ancestor in tree.ancestors(q) {
            row[ancestor] = true;
        }
    }
    mask
}

/// Position ids for the flattened tree: each node's position equals its depth, so
/// RoPE/positions reflect the path length rather than the flat index. Sibling
/// candidates for one slot share a position id.
pub fn relative_position_ids(tree: &SpecTree) -> Vec<usize> {
    tree.nodes().iter().map(|node| node.depth).collect()
}

/// Absolute position ids, offsetting [`relative_position_ids`] by `base` (the number
/// of committed tokens already in the KV cache).
pub fn absolute_position_ids(tree: &SpecTree, base: usize) -> Vec<usize> {
    tree.nodes().iter().map(|node| base + node.depth).collect()
}

/// Plan for compacting the KV cache after a tree verification pass.
///
/// The verify pass materialises KV for *every* flattened node. After the acceptance
/// walk chooses a path, only that path's nodes must be retained (in order); all other
/// branches' KV is discarded. This mirrors the linear rewind primitive, generalised
/// to keep an arbitrary subset in committed order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvRetentionPlan {
    /// Flattened node indices to keep, in committed (root-to-leaf) order.
    pub retained_nodes: Vec<usize>,
    /// KV sequence length after compaction: `base_len + retained_nodes.len()`.
    pub final_len: usize,
}

/// Build a [`KvRetentionPlan`] that retains exactly the accepted path's nodes.
pub fn kv_retention_plan(base_len: usize, accepted_nodes: &[usize]) -> KvRetentionPlan {
    KvRetentionPlan {
        retained_nodes: accepted_nodes.to_vec(),
        final_len: base_len + accepted_nodes.len(),
    }
}

/// Produces target logits for tree nodes given their ancestor path context.
///
/// In production a scorer wraps one batched masked target forward. In tests it can
/// wrap a reference model (scoring each node by its ancestor path) to prove that the
/// tree machinery reproduces plain decoding exactly.
pub trait TreeScorer {
    /// Target logits predicting the slot that follows `path`.
    ///
    /// `path` is the committed continuation from the anchor to (and including) the
    /// node being scored; an empty `path` requests the anchor's base logits, which
    /// verify the root candidates.
    fn score(&mut self, path: &[TokenId]) -> anyhow::Result<Vec<f32>>;
}

/// End-to-end tree verification: score every node, run the acceptance walk, and
/// compute the KV compaction plan.
///
/// `base_len` is the committed KV length before this step. The returned
/// [`TreeVerification::plan`] retains only the accepted path.
pub fn verify_tree<S: TreeScorer>(
    tree: &SpecTree,
    base_len: usize,
    rule: AcceptanceRule,
    scorer: &mut S,
) -> anyhow::Result<TreeVerification> {
    let base_logits = scorer.score(&[])?;
    let mut node_logits = Vec::with_capacity(tree.len());
    for i in 0..tree.len() {
        node_logits.push(scorer.score(&tree.path_tokens(i))?);
    }
    let outcome = tree.accept(rule, &base_logits, &node_logits)?;
    let plan = kv_retention_plan(base_len, &outcome.nodes);
    Ok(TreeVerification { outcome, plan })
}

/// Result of [`verify_tree`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeVerification {
    /// Accepted path and committed tokens.
    pub outcome: AcceptOutcome,
    /// KV compaction plan retaining only the accepted path.
    pub plan: KvRetentionPlan,
}

/// Incremental builder that maintains the topological (parent-before-child) node
/// order and enforces an optional node budget.
#[derive(Debug, Clone)]
pub struct SpecTreeBuilder {
    nodes: Vec<TreeNode>,
    budget: Option<usize>,
}

impl SpecTreeBuilder {
    /// A builder with an optional maximum node count (`None` = unbounded).
    pub fn new(budget: Option<usize>) -> Self {
        Self {
            nodes: Vec::new(),
            budget,
        }
    }

    fn check_budget(&self) -> anyhow::Result<()> {
        if let Some(budget) = self.budget
            && self.nodes.len() >= budget
        {
            anyhow::bail!("speculative tree exceeds node budget of {budget}");
        }
        Ok(())
    }

    /// Add a root candidate (attached to the committed anchor). Returns its index.
    pub fn add_root(&mut self, token: TokenId) -> anyhow::Result<usize> {
        self.check_budget()?;
        let index = self.nodes.len();
        self.nodes.push(TreeNode {
            token,
            parent: None,
            depth: 0,
        });
        Ok(index)
    }

    /// Add a child of `parent`. Returns the new node index.
    ///
    /// `parent` must already exist, guaranteeing parents precede children.
    pub fn add_child(&mut self, parent: usize, token: TokenId) -> anyhow::Result<usize> {
        let depth = self
            .nodes
            .get(parent)
            .map(|node| node.depth + 1)
            .ok_or_else(|| anyhow::anyhow!("parent node {parent} does not exist yet"))?;
        self.check_budget()?;
        let index = self.nodes.len();
        self.nodes.push(TreeNode {
            token,
            parent: Some(parent),
            depth,
        });
        Ok(index)
    }

    /// Append a linear chain of `tokens` as a single branch, returning the node
    /// indices in order. This reproduces the linear topology as a one-branch tree.
    pub fn add_chain(&mut self, tokens: &[TokenId]) -> anyhow::Result<Vec<usize>> {
        let mut indices = Vec::with_capacity(tokens.len());
        let mut parent: Option<usize> = None;
        for &token in tokens {
            let index = match parent {
                None => self.add_root(token)?,
                Some(p) => self.add_child(p, token)?,
            };
            indices.push(index);
            parent = Some(index);
        }
        Ok(indices)
    }

    /// Finalise the tree.
    pub fn build(self) -> SpecTree {
        SpecTree { nodes: self.nodes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// root -> {a, b}; a -> {c, d}; b -> {e, f}.
    fn sample_tree() -> (SpecTree, [usize; 7]) {
        let mut builder = SpecTreeBuilder::new(Some(16));
        let root = builder.add_root(10).unwrap();
        let a = builder.add_child(root, 11).unwrap();
        let b = builder.add_child(root, 12).unwrap();
        let c = builder.add_child(a, 13).unwrap();
        let d = builder.add_child(a, 14).unwrap();
        let e = builder.add_child(b, 15).unwrap();
        let f = builder.add_child(b, 16).unwrap();
        (builder.build(), [root, a, b, c, d, e, f])
    }

    #[test]
    fn attention_mask_sets_exactly_ancestor_and_self_edges() {
        let (tree, [root, a, b, c, d, e, f]) = sample_tree();
        let mask = ancestor_attention_mask(&tree);

        let mut expected = vec![vec![false; 7]; 7];
        // Self edges.
        for (i, row) in expected.iter_mut().enumerate() {
            row[i] = true;
        }
        // Ancestor edges (query attends to each ancestor).
        expected[a][root] = true;
        expected[b][root] = true;
        expected[c][a] = true;
        expected[c][root] = true;
        expected[d][a] = true;
        expected[d][root] = true;
        expected[e][b] = true;
        expected[e][root] = true;
        expected[f][b] = true;
        expected[f][root] = true;

        assert_eq!(mask, expected);
        // No sibling / cross-branch leakage.
        assert!(!mask[a][b]);
        assert!(!mask[c][d]);
        assert!(!mask[c][b]);
        assert!(!mask[e][a]);
    }

    #[test]
    fn position_ids_equal_node_depths() {
        let (tree, _) = sample_tree();
        assert_eq!(relative_position_ids(&tree), vec![0, 1, 1, 2, 2, 2, 2]);
        assert_eq!(absolute_position_ids(&tree, 5), vec![5, 6, 6, 7, 7, 7, 7]);
    }

    /// One-hot logits selecting `token` as the deterministic argmax.
    fn onehot(token: TokenId) -> Vec<f32> {
        let mut logits = vec![0.0_f32; 32];
        logits[token as usize] = 10.0;
        logits
    }

    #[test]
    fn greedy_accept_full_path_plus_bonus() {
        let (tree, [_root, a, _b, c, _d, _e, _f]) = sample_tree();
        // Target greedy chain: 10 -> 11 -> 13 -> 20 (bonus).
        let base = onehot(10);
        let mut node = vec![vec![0.0]; 7];
        node[_root] = onehot(11);
        node[a] = onehot(13);
        node[c] = onehot(20); // leaf reached -> 20 is the bonus token
        node[_b] = onehot(0);
        node[_d] = onehot(0);
        node[_e] = onehot(0);
        node[_f] = onehot(0);

        let outcome = tree.accept(AcceptanceRule::Greedy, &base, &node).unwrap();
        assert_eq!(outcome.nodes, vec![_root, a, c]);
        assert_eq!(outcome.tokens, vec![10, 11, 13, 20]);
    }

    #[test]
    fn greedy_accept_partial_path_then_correction() {
        let (tree, [_root, a, _b, _c, _d, _e, _f]) = sample_tree();
        // 10 -> 11 accepted, then target wants 99 which no child offers.
        let base = onehot(10);
        let mut node = vec![vec![0.0]; 7];
        node[_root] = onehot(11);
        node[a] = onehot(25); // correction token, not among children {13,14}
        for i in [_b, _c, _d, _e, _f] {
            node[i] = onehot(0);
        }
        let outcome = tree.accept(AcceptanceRule::Greedy, &base, &node).unwrap();
        assert_eq!(outcome.nodes, vec![_root, a]);
        assert_eq!(outcome.tokens, vec![10, 11, 25]);
    }

    #[test]
    fn greedy_root_reject_returns_only_bonus() {
        let (tree, _) = sample_tree();
        // Target's first token (7) matches no root candidate {10}.
        let base = onehot(7);
        let node = vec![onehot(0); 7];
        let outcome = tree.accept(AcceptanceRule::Greedy, &base, &node).unwrap();
        assert!(outcome.nodes.is_empty());
        assert_eq!(outcome.tokens, vec![7]);
    }

    #[test]
    fn typical_rule_rejects_low_probability_match() {
        let mut builder = SpecTreeBuilder::new(None);
        let root = builder.add_root(3).unwrap();
        builder.add_child(root, 4).unwrap();
        let tree = builder.build();
        // Base logits make token 3 the argmax but with modest probability.
        let base = vec![1.0, 1.0, 1.0, 1.2, 1.0];
        let node = vec![onehot(9); 2];

        // Greedy accepts the root, then 9 is the bonus.
        let greedy = tree.accept(AcceptanceRule::Greedy, &base, &node).unwrap();
        assert_eq!(greedy.tokens, vec![3, 9]);

        // Typical with a high threshold rejects the low-confidence root: only the
        // target token 3 is committed as the correction.
        let typical = tree
            .accept(AcceptanceRule::Typical { threshold: 0.9 }, &base, &node)
            .unwrap();
        assert!(typical.nodes.is_empty());
        assert_eq!(typical.tokens, vec![3]);
    }

    #[test]
    fn kv_retention_keeps_only_accepted_path() {
        let plan = kv_retention_plan(5, &[0, 1, 3]);
        assert_eq!(plan.retained_nodes, vec![0, 1, 3]);
        assert_eq!(plan.final_len, 5 + 3);

        let empty = kv_retention_plan(5, &[]);
        assert!(empty.retained_nodes.is_empty());
        assert_eq!(empty.final_len, 5);
    }

    /// A deterministic reference "model": next token is a fixed function of the last
    /// context token, so plain greedy decoding is fully predictable.
    struct OracleModel {
        prompt: Vec<TokenId>,
    }

    impl OracleModel {
        fn next_token(&self, ctx: &[TokenId]) -> TokenId {
            // Simple deterministic transition with a modulo wrap.
            let last = ctx.last().copied().unwrap_or(0);
            (last.wrapping_mul(2).wrapping_add(1)) % 17
        }

        fn plain_greedy(&self, steps: usize) -> Vec<TokenId> {
            let mut ctx = self.prompt.clone();
            let mut out = Vec::new();
            for _ in 0..steps {
                let token = self.next_token(&ctx);
                out.push(token);
                ctx.push(token);
            }
            out
        }
    }

    /// Scorer that returns one-hot logits at the oracle's greedy next token for the
    /// committed prompt concatenated with the queried path.
    struct OracleScorer<'a> {
        model: &'a OracleModel,
        committed: Vec<TokenId>,
    }

    impl TreeScorer for OracleScorer<'_> {
        fn score(&mut self, path: &[TokenId]) -> anyhow::Result<Vec<f32>> {
            let mut ctx = self.committed.clone();
            ctx.extend_from_slice(path);
            Ok(onehot(self.model.next_token(&ctx)))
        }
    }

    /// Build a branching draft tree that includes the oracle's true greedy chain for
    /// `depth` steps plus per-slot decoy siblings that must be rejected.
    fn draft_tree(model: &OracleModel, committed: &[TokenId], depth: usize) -> SpecTree {
        let mut builder = SpecTreeBuilder::new(Some(64));
        let mut ctx = committed.to_vec();
        let mut parent: Option<usize> = None;
        for _ in 0..depth {
            let truth = model.next_token(&ctx);
            let node = match parent {
                None => builder.add_root(truth).unwrap(),
                Some(p) => builder.add_child(p, truth).unwrap(),
            };
            // Two decoy siblings that never match the deterministic target.
            let decoy_a = (truth + 100) % 17;
            let decoy_b = (truth + 101) % 17;
            for decoy in [decoy_a, decoy_b] {
                if decoy != truth {
                    match parent {
                        None => {
                            builder.add_root(decoy).unwrap();
                        }
                        Some(p) => {
                            builder.add_child(p, decoy).unwrap();
                        }
                    }
                }
            }
            ctx.push(truth);
            parent = Some(node);
        }
        builder.build()
    }

    #[test]
    fn tree_greedy_equivalence_matches_plain_greedy_oracle() {
        let model = OracleModel {
            prompt: vec![2, 5, 1],
        };
        let target_steps = 20;
        let expected = model.plain_greedy(target_steps);

        // Drive a full tree-speculative loop against the oracle and assert the
        // committed sequence is byte-for-byte identical to plain greedy decoding.
        let mut committed = model.prompt.clone();
        let mut produced = Vec::new();
        while produced.len() < target_steps {
            let base_len = committed.len();
            let tree = draft_tree(&model, &committed, 4);
            let mut scorer = OracleScorer {
                model: &model,
                committed: committed.clone(),
            };
            let verification =
                verify_tree(&tree, base_len, AcceptanceRule::Greedy, &mut scorer).unwrap();

            // The retained KV length must equal the accepted path length.
            assert_eq!(
                verification.plan.final_len,
                base_len + verification.outcome.nodes.len()
            );

            for &token in &verification.outcome.tokens {
                if produced.len() == target_steps {
                    break;
                }
                committed.push(token);
                produced.push(token);
            }
            assert!(
                !verification.outcome.tokens.is_empty(),
                "must make progress"
            );
        }

        assert_eq!(produced, expected);
    }
}
