//! Token-aware GBNF replay, including an initial piece truncated by a trigger.
//!
//! Reuse the bounded GBNF parser. A token terminal consumes one original model
//! token; a scalar terminal consumes its supplied bytes, not a guessed spelling.
use super::{Atom, Expression, MAX_ITEMS, parse};
use anyhow::{Context, Result, ensure};
use std::{collections::HashSet, sync::Arc};

type NodeId = usize;

#[derive(Clone)]
enum Node {
    Scalar(Vec<(u32, u32)>),
    Token {
        id: u32,
        negate: bool,
    },
    Sequence(Vec<NodeId>),
    Choice(Vec<NodeId>),
    Repeat {
        body: NodeId,
        min: usize,
        max: Option<usize>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Frame {
    Node(NodeId),
    Repeat {
        body: NodeId,
        min: usize,
        max: Option<usize>,
    },
}

type Stack = Vec<Frame>;

/// A reversible recognizer for raw token events. Rejected events and resource
/// failures leave the cursor unchanged. Clone makes an independent checkpoint.
#[derive(Clone)]
pub struct GbnfReplay {
    nodes: Arc<Vec<Node>>,
    stacks: Vec<Stack>,
    pending_utf8: Vec<u8>,
    bytes: usize,
    max_bytes: usize,
}

struct Compiler {
    nodes: Vec<Node>,
}
impl Compiler {
    fn push(&mut self, node: Node) -> Result<NodeId> {
        ensure!(
            self.nodes.len() < MAX_ITEMS,
            "GBNF replay program exceeds {MAX_ITEMS} nodes; simplify the grammar"
        );
        let id = self.nodes.len();
        self.nodes.push(node);
        Ok(id)
    }
    fn expression(
        &mut self,
        expression: &Expression,
        resolve: &mut impl FnMut(&str) -> Result<u32>,
    ) -> Result<NodeId> {
        let mut choices = Vec::new();
        for sequence in expression {
            let mut items = Vec::new();
            for item in sequence {
                let mut node = match &item.atom {
                    Atom::Reference(id) => *id,
                    Atom::Literal(text) => {
                        let scalars = text
                            .chars()
                            .map(|c| self.push(Node::Scalar(vec![(c as u32, c as u32)])))
                            .collect::<Result<Vec<_>>>()?;
                        self.push(Node::Sequence(scalars))?
                    }
                    Atom::Class(class) => self.push(Node::Scalar(class_ranges(class)?))?,
                    Atom::Any => self.push(Node::Scalar(vec![(0, 0xd7ff), (0xe000, 0x10ffff)]))?,
                    Atom::Token { spelling, negate } => self.push(Node::Token {
                        id: resolve(spelling)
                            .with_context(|| format!("resolve GBNF replay token {spelling}"))?,
                        negate: *negate,
                    })?,
                    Atom::Group(group) => self.expression(group, resolve)?,
                };
                for &(min, max) in &item.repeats {
                    node = self.push(Node::Repeat {
                        body: node,
                        min,
                        max,
                    })?;
                }
                items.push(node);
            }
            choices.push(self.push(Node::Sequence(items))?);
        }
        self.push(Node::Choice(choices))
    }
}

fn class_ranges(class: &str) -> Result<Vec<(u32, u32)>> {
    use regex_syntax::hir::{Class, HirKind};
    let hir = regex_syntax::Parser::new()
        .parse(class)
        .context("compile GBNF replay character class")?;
    match hir.kind() {
        HirKind::Class(Class::Unicode(class)) => Ok(class
            .ranges()
            .iter()
            .map(|r| (r.start() as u32, r.end() as u32))
            .collect()),
        // regex-syntax simplifies singleton classes to literals.
        HirKind::Literal(literal) => {
            let text = std::str::from_utf8(&literal.0).context("decode GBNF singleton class")?;
            let mut chars = text.chars();
            let scalar = chars.next().context("GBNF singleton class is empty")? as u32;
            ensure!(
                chars.next().is_none(),
                "GBNF class compiled to multiple scalars; use a character class"
            );
            Ok(vec![(scalar, scalar)])
        }
        _ => anyhow::bail!(
            "GBNF replay class did not compile to Unicode ranges; use scalar character classes"
        ),
    }
}

fn enqueue(queue: &mut Vec<Stack>, queued: &mut usize, visited: usize, stack: Stack) -> Result<()> {
    let total = queued
        .checked_add(stack.len().max(1))
        .context("GBNF replay queued state size overflow")?;
    ensure!(
        total.saturating_add(visited) <= MAX_ITEMS,
        "GBNF replay expansion exceeds {MAX_ITEMS} stack cells; simplify ambiguous or deeply recursive grammar rules"
    );
    *queued = total;
    queue.push(stack);
    Ok(())
}

impl GbnfReplay {
    /// Resolve token spellings through the model's actual vocabulary. The byte
    /// budget covers the authored grammar and, separately, accepted replay data.
    pub fn compile(
        source: &str,
        max_bytes: usize,
        mut resolve: impl FnMut(&str) -> Result<u32>,
    ) -> Result<Self> {
        let parsed = parse(source, max_bytes)?;
        let mut compiler = Compiler {
            nodes: vec![Node::Sequence(Vec::new()); parsed.rules.len()],
        };
        for (id, rule) in parsed.rules.iter().enumerate() {
            let body = compiler.expression(
                rule.expression
                    .as_ref()
                    .context("GBNF replay rule is undefined")?,
                &mut resolve,
            )?;
            compiler.nodes[id] = Node::Sequence(vec![body]);
        }
        let root = *parsed
            .names
            .get("root")
            .context("GBNF replay root is missing")?;
        let mut cursor = Self {
            nodes: Arc::new(compiler.nodes),
            stacks: Vec::new(),
            pending_utf8: Vec::new(),
            bytes: 0,
            max_bytes,
        };
        cursor.stacks = cursor.close(vec![vec![Frame::Node(root)]])?;
        Ok(cursor)
    }

    fn close(&self, mut queue: Vec<Stack>) -> Result<Vec<Stack>> {
        let mut seen = HashSet::new();
        let mut ready = Vec::new();
        let mut cells = 0usize;
        let mut queued = queue.iter().map(|stack| stack.len().max(1)).sum::<usize>();
        ensure!(
            queued <= MAX_ITEMS,
            "GBNF replay expansion exceeds {MAX_ITEMS} stack cells; simplify the grammar"
        );
        while let Some(mut stack) = queue.pop() {
            queued -= stack.len().max(1);
            if !seen.insert(stack.clone()) {
                continue;
            }
            cells = cells
                .checked_add(stack.len().max(1))
                .context("GBNF replay state size overflow")?;
            ensure!(
                cells <= MAX_ITEMS,
                "GBNF replay expansion exceeds {MAX_ITEMS} stack cells; simplify ambiguous or deeply recursive grammar rules"
            );
            match stack.pop() {
                None => ready.push(stack),
                Some(Frame::Node(id)) => match &self.nodes[id] {
                    Node::Scalar(_) | Node::Token { .. } => {
                        stack.push(Frame::Node(id));
                        ready.push(stack);
                    }
                    Node::Sequence(items) => {
                        stack.extend(items.iter().rev().copied().map(Frame::Node));
                        enqueue(&mut queue, &mut queued, cells, stack)?;
                    }
                    Node::Choice(choices) => {
                        for &id in choices {
                            let mut branch = stack.clone();
                            branch.push(Frame::Node(id));
                            enqueue(&mut queue, &mut queued, cells, branch)?;
                        }
                    }
                    Node::Repeat { body, min, max } => {
                        stack.push(Frame::Repeat {
                            body: *body,
                            min: *min,
                            max: *max,
                        });
                        enqueue(&mut queue, &mut queued, cells, stack)?;
                    }
                },
                Some(Frame::Repeat { body, min, max }) => {
                    if min == 0 {
                        enqueue(&mut queue, &mut queued, cells, stack.clone())?;
                    }
                    if max != Some(0) {
                        stack.push(Frame::Repeat {
                            body,
                            min: min.saturating_sub(1),
                            max: max.map(|n| n - 1),
                        });
                        stack.push(Frame::Node(body));
                        enqueue(&mut queue, &mut queued, cells, stack)?;
                    }
                }
            }
        }
        Ok(ready)
    }

    pub(crate) fn fits_piece(&self, bytes: usize) -> bool {
        self.bytes
            .checked_add(bytes)
            .is_some_and(|count| count <= self.max_bytes)
    }

    /// Whether an ending is legal. As in llama.cpp, completed token-reference
    /// branches may end independently of partial UTF-8 in the token's spelling.
    pub fn is_accepting(&self) -> bool {
        self.stacks.iter().any(Vec::is_empty)
    }

    /// Accept exactly one original model token and its actual replay piece.
    /// A trigger may truncate this piece without changing `token_id`.
    pub fn consume(&mut self, token_id: u32, piece: &[u8]) -> Result<bool> {
        let next_bytes = self
            .bytes
            .checked_add(piece.len())
            .context("GBNF replay byte count overflow")?;
        ensure!(
            next_bytes <= self.max_bytes,
            "GBNF replay would exceed max_constraint_bytes={}; shorten generated output or increase the explicit byte budget",
            self.max_bytes
        );
        let mut combined = self.pending_utf8.clone();
        combined.extend_from_slice(piece);
        // Invalid bytes can still satisfy an explicit token reference. The
        // upstream decoder starts fresh on the next piece after that invalid
        // event; only a valid incomplete UTF-8 scalar carries between pieces.
        let (text, pending, invalid_utf8) = match std::str::from_utf8(&combined) {
            Ok(text) => (text, &[][..], false),
            Err(error) if error.error_len().is_none() => (
                std::str::from_utf8(&combined[..error.valid_up_to()])?,
                &combined[error.valid_up_to()..],
                false,
            ),
            Err(_) => ("", &[][..], true),
        };
        let mut accepted = Vec::new();
        let mut text_stacks = Vec::new();
        for stack in &self.stacks {
            match stack.last().and_then(|frame| match frame {
                Frame::Node(id) => Some(&self.nodes[*id]),
                _ => None,
            }) {
                Some(Node::Token { id, negate }) => {
                    if (token_id == *id) != *negate {
                        let mut stack = stack.clone();
                        stack.pop();
                        accepted.push(stack);
                    }
                }
                Some(Node::Scalar(_)) if !invalid_utf8 => text_stacks.push(stack.clone()),
                _ => {}
            }
        }
        for scalar in text.chars() {
            let mut next = Vec::new();
            for mut stack in text_stacks {
                if let Some(Frame::Node(id)) = stack.last()
                    && let Node::Scalar(ranges) = &self.nodes[*id]
                    && ranges
                        .iter()
                        .any(|&(lo, hi)| (lo..=hi).contains(&(scalar as u32)))
                {
                    stack.pop();
                    next.push(stack);
                }
            }
            text_stacks = self.close(next)?;
        }
        if !pending.is_empty() {
            let Some((lo, hi)) = utf8_extent(pending) else {
                return Ok(false);
            };
            text_stacks.retain(|stack| matches!(stack.last(), Some(Frame::Node(id)) if matches!(&self.nodes[*id], Node::Scalar(ranges) if ranges.iter().any(|&(a,b)| a <= hi && lo <= b))));
        }
        accepted.extend(text_stacks);
        let stacks = self.close(accepted)?;
        if stacks.is_empty() {
            return Ok(false);
        }
        self.stacks = stacks;
        self.pending_utf8 = pending.to_vec();
        self.bytes = next_bytes;
        Ok(true)
    }
}

fn utf8_extent(prefix: &[u8]) -> Option<(u32, u32)> {
    let (&first, rest) = prefix.split_first()?;
    let (length, mut value, minimum) = match first {
        0xc2..=0xdf => (2, u32::from(first & 0x1f), 0x80),
        0xe0..=0xef => (3, u32::from(first & 0x0f), 0x800),
        0xf0..=0xf4 => (4, u32::from(first & 7), 0x10000),
        _ => return None,
    };
    if prefix.len() >= length {
        return None;
    }
    for &byte in rest {
        if byte & 0xc0 != 0x80 {
            return None;
        }
        value = (value << 6) | u32::from(byte & 0x3f);
    }
    let bits = 6 * (length - prefix.len());
    let lo = (value << bits).max(minimum);
    let hi = ((value << bits) | ((1 << bits) - 1)).min(0x10ffff);
    (lo <= hi).then_some((lo, hi))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn grammar(source: &str) -> GbnfReplay {
        GbnfReplay::compile(source, 4096, |spelling| {
            Ok(if spelling == "<[7]>" { 7 } else { 8 })
        })
        .unwrap()
    }
    #[test]
    fn replay_preserves_identity_of_truncated_and_negated_token_pieces() {
        for source in [r#"root ::= <[7]> "x""#, r#"root ::= !<[8]> "x""#] {
            let mut cursor = grammar(source);
            assert!(cursor.consume(7, b"tail").unwrap());
            assert!(!cursor.is_accepting());
            assert!(cursor.consume(9, b"x").unwrap());
            assert!(cursor.is_accepting());
        }
    }
    #[test]
    fn replay_keeps_text_and_token_branches_and_rejection_is_transactional() {
        let mut cursor = grammar(r#"root ::= (<[7]> "A" | "tailB")"#);
        assert!(cursor.consume(7, b"tail").unwrap());
        let mut other = cursor.clone();
        assert!(!cursor.consume(1, b"X").unwrap());
        assert!(cursor.consume(1, b"A").unwrap());
        assert!(other.consume(1, b"B").unwrap());
        assert!(cursor.is_accepting() && other.is_accepting());
    }
    #[test]
    fn replay_handles_split_unicode_classes_repetition_and_recursion() {
        let mut cursor = grammar(r#"root ::= [é]+ ("(" root ")")?"#);
        assert!(cursor.consume(1, &[0xc3]).unwrap());
        assert!(!cursor.consume(1, b"x").unwrap());
        assert!(cursor.consume(1, &[0xa9]).unwrap());
        assert!(cursor.is_accepting());
        assert!(cursor.consume(1, "(é)".as_bytes()).unwrap());
        assert!(cursor.is_accepting());
    }
    #[test]
    fn replay_rejects_wrong_token_identity_and_illegal_unicode_prefix() {
        assert!(!grammar("root ::= <[7]>").consume(8, b"tail").unwrap());
        assert!(!grammar("root ::= [a-z]").consume(1, &[0xc3]).unwrap());
        assert!(!grammar("root ::= .").consume(1, &[0xed, 0xa0]).unwrap());
    }
    #[test]
    fn replay_token_identity_survives_a_truncated_utf8_scalar() {
        let mut cursor = grammar("root ::= <[7]>");
        assert!(cursor.consume(7, &[0xa9]).unwrap());
        assert!(cursor.is_accepting());
        for piece in [&[0xa9][..], &[0xff][..], &[0xc3, 0x28][..]] {
            let mut cursor = grammar(r#"root ::= <[7]> "x""#);
            assert!(cursor.consume(7, piece).unwrap());
            let mut checkpoint = cursor.clone();
            assert!(!cursor.consume(9, b"wrong").unwrap());
            assert!(cursor.consume(9, b"x").unwrap());
            assert!(checkpoint.consume(9, b"x").unwrap());
            assert!(cursor.is_accepting() && checkpoint.is_accepting());
        }
    }
    #[test]
    fn replay_singleton_class_quantifiers_and_finite_bounds() {
        let mut cursor = grammar("root ::= [x]{2,3}");
        assert!(cursor.consume(1, b"xx").unwrap());
        assert!(cursor.is_accepting());
        assert!(cursor.consume(1, b"x").unwrap());
        assert!(!cursor.consume(1, b"x").unwrap());
        assert!(cursor.is_accepting());
    }
    #[test]
    fn replay_byte_budget_failure_does_not_change_state() {
        let mut cursor = GbnfReplay::compile("root ::= .+", 16, |_| unreachable!()).unwrap();
        assert!(cursor.consume(1, b"a").unwrap());
        assert!(
            cursor
                .consume(1, &[b'a'; 16])
                .unwrap_err()
                .to_string()
                .contains("max_constraint_bytes=16")
        );
        assert!(cursor.consume(1, b"b").unwrap());
        assert!(cursor.is_accepting());
    }

    #[test]
    fn replay_ambiguous_expansion_has_a_transactional_state_budget() {
        let mut cursor = grammar(r#"root ::= "x" root root | """#);
        for _ in 0..200 {
            let before = cursor.clone();
            match cursor.consume(1, b"x") {
                Ok(true) => assert!(cursor.is_accepting()),
                Ok(false) => panic!("recursive grammar accepts every x prefix"),
                Err(error) => {
                    assert!(error.to_string().contains("stack cells"), "{error:#}");
                    assert_eq!(cursor.stacks, before.stacks);
                    assert_eq!(cursor.bytes, before.bytes);
                    return;
                }
            }
        }
        panic!("ambiguous replay did not reach its declared expansion budget");
    }

    #[test]
    fn replay_full_token_masks_match_existing_grammar_backend() -> Result<()> {
        use crate::logits::{
            Constraint, GrammarConstraintKind, LlguidanceConstraint, ProcessorContext,
            TokenCandidate,
        };
        let pieces = ["a", "b", "ab", "é", ""];
        for source in [
            r#"root ::= [a-b]{1,3}"#,
            r#"root ::= ("a" root "b") | "é""#,
            r#"root ::= ("ab" | "é")+"#,
            r#"root ::= <[2]> "a""#,
            r#"root ::= !<[2]> "b""#,
        ] {
            let resolve = |_: &str| Ok(2);
            let initial = GbnfReplay::compile(source, 4096, resolve)?;
            let lark = super::super::render_rules(&parse(source, 4096)?, &mut |_: &str| Ok(2))?;
            let reference = LlguidanceConstraint::from_token_bytes(
                GrammarConstraintKind::Lark,
                &lark,
                pieces.iter().map(|s| s.as_bytes().to_vec()).collect(),
                &[4],
                "",
            )?
            .with_completion_only_eos(vec![4]);
            let candidates = pieces
                .iter()
                .enumerate()
                .map(|(id, text)| TokenCandidate {
                    token_id: id as u32,
                    text: (*text).into(),
                    is_eos: id == 4,
                })
                .collect::<Vec<_>>();
            let mut pending = vec![(initial, Vec::new())];
            while let Some((cursor, history)) = pending.pop() {
                let mask = reference.allowed_next_tokens(
                    &ProcessorContext {
                        generated_tokens: history.clone(),
                        ..Default::default()
                    },
                    &candidates,
                )?;
                for (id, &piece) in pieces.iter().enumerate() {
                    let mut next = cursor.clone();
                    let allowed = if id == 4 {
                        cursor.is_accepting()
                    } else {
                        next.consume(id as u32, piece.as_bytes())?
                    };
                    assert_eq!(
                        allowed, mask[id],
                        "{source}: history={history:?}, candidate={id}"
                    );
                    if allowed && id != 4 && history.len() < 3 {
                        let mut history = history.clone();
                        history.push(id as u32);
                        pending.push((next, history));
                    }
                }
            }
        }
        Ok(())
    }
}
