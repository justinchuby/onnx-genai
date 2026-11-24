//! Byte-native ECMAScript execution with explicit capture and work ownership.
use anyhow::{Context, Result, ensure};
use std::ops::Range;

#[derive(Clone, Copy)]
pub(super) struct Id(pub(super) usize);
pub(super) struct Node {
    pub(super) kind: Kind,
    pub(super) captures: Range<usize>,
}
pub(super) enum Kind {
    Byte(u8),
    Set([u64; 4]),
    Seq(Vec<Id>),
    Alt(Vec<Id>),
    Capture {
        group: usize,
        body: Id,
    },
    Look {
        positive: bool,
        body: Id,
    },
    Repeat {
        body: Id,
        min: usize,
        max: Option<usize>,
        greedy: bool,
    },
    Start,
    End,
    Boundary(bool),
    Backref(usize),
    Fail,
}
pub(super) struct Program {
    pub(super) nodes: Vec<Node>,
    pub(super) root: Id,
    pub(super) captures: usize,
    pub(super) full: bool,
}
#[derive(Clone)]
struct Repetition {
    body: Id,
    min: usize,
    max: Option<usize>,
    greedy: bool,
    captures: Range<usize>,
    count: usize,
}
#[derive(Clone)]
enum Op {
    Node(Id),
    Close {
        group: usize,
        start: usize,
    },
    Repeat(Repetition),
    Repeated {
        repetition: Repetition,
        start: usize,
    },
}
#[derive(Clone)]
struct State {
    position: usize,
    captures: Vec<Option<Range<usize>>>,
    ops: Vec<Op>,
}
impl State {
    fn units(&self) -> usize {
        1 + self.captures.len() + self.ops.len()
    }
}
struct Budget {
    fuel: usize,
    live: usize,
}
impl Budget {
    fn spend(&mut self, work: usize) -> Result<()> {
        ensure!(
            work <= self.fuel,
            "grammar trigger execution budget exhausted; simplify the pattern or use a token/literal trigger"
        );
        self.fuel -= work;
        Ok(())
    }
    fn reserve(&mut self, units: usize) -> Result<()> {
        let live = self
            .live
            .checked_add(units)
            .context("grammar trigger state size overflow; simplify the pattern")?;
        ensure!(
            live <= 1_000_000,
            "grammar trigger backtracking state budget exhausted; simplify ambiguous captures or use a token/literal trigger"
        );
        self.live = live;
        Ok(())
    }
}
struct Matched {
    end: usize,
    captures: Vec<Option<Range<usize>>>,
}
fn push(state: &mut State, op: Op, budget: &mut Budget) -> Result<()> {
    budget.reserve(1)?;
    state.ops.push(op);
    Ok(())
}
fn branch(state: &State, queued: &mut Vec<State>, budget: &mut Budget) -> Result<()> {
    budget.reserve(state.units())?;
    queued.push(state.clone());
    Ok(())
}
fn take(state: &mut State, mut repetition: Repetition, budget: &mut Budget) -> Result<()> {
    budget.spend(repetition.captures.len())?;
    for group in repetition.captures.clone() {
        state.captures[group] = None;
    }
    repetition.count += 1;
    let body = repetition.body;
    push(
        state,
        Op::Repeated {
            repetition,
            start: state.position,
        },
        budget,
    )?;
    push(state, Op::Node(body), budget)
}
fn word(byte: Option<&u8>) -> bool {
    byte.is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
}

fn run(
    program: &Program,
    root: Id,
    input: &[u8],
    start: usize,
    captures: Vec<Option<Range<usize>>>,
    full: bool,
    budget: &mut Budget,
) -> Result<Option<Matched>> {
    // Nested lookahead shares fuel and heap admission, but owns its choices.
    // Restore only heap accounting on every exit, including errors.
    let entered = budget.live;
    let result = (|| -> Result<Option<Matched>> {
        let mut state = State {
            position: start,
            captures,
            ops: vec![Op::Node(root)],
        };
        budget.reserve(state.units())?;
        let mut queued = Vec::new();
        loop {
            budget.spend(1)?;
            let mut valid = true;
            if let Some(op) = state.ops.pop() {
                budget.live -= 1;
                match op {
                    Op::Node(id) => {
                        let node = &program.nodes[id.0];
                        match &node.kind {
                            Kind::Byte(byte) => {
                                valid = input.get(state.position) == Some(byte);
                                if valid {
                                    state.position += 1;
                                }
                            }
                            Kind::Set(set) => {
                                valid = input.get(state.position).is_some_and(|&b| {
                                    set[usize::from(b) / 64] & (1u64 << (b % 64)) != 0
                                });
                                if valid {
                                    state.position += 1;
                                }
                            }
                            Kind::Seq(items) => {
                                budget.reserve(items.len())?;
                                state.ops.extend(items.iter().rev().copied().map(Op::Node));
                            }
                            Kind::Alt(items) => {
                                if let Some((&first, rest)) = items.split_first() {
                                    for &id in rest.iter().rev() {
                                        budget.reserve(state.units() + 1)?;
                                        let mut alternative = state.clone();
                                        alternative.ops.push(Op::Node(id));
                                        queued.push(alternative);
                                    }
                                    push(&mut state, Op::Node(first), budget)?;
                                } else {
                                    valid = false;
                                }
                            }
                            Kind::Capture { group, body } => {
                                let start = state.position;
                                push(
                                    &mut state,
                                    Op::Close {
                                        group: *group,
                                        start,
                                    },
                                    budget,
                                )?;
                                push(&mut state, Op::Node(*body), budget)?;
                            }
                            Kind::Look { positive, body } => {
                                budget.spend(state.captures.len())?;
                                let matched = run(
                                    program,
                                    *body,
                                    input,
                                    state.position,
                                    state.captures.clone(),
                                    false,
                                    budget,
                                )?;
                                match (positive, matched) {
                                    (true, Some(found)) => state.captures = found.captures,
                                    (false, None) => {}
                                    _ => valid = false,
                                }
                            }
                            Kind::Repeat {
                                body,
                                min,
                                max,
                                greedy,
                            } => push(
                                &mut state,
                                Op::Repeat(Repetition {
                                    body: *body,
                                    min: *min,
                                    max: *max,
                                    greedy: *greedy,
                                    captures: node.captures.clone(),
                                    count: 0,
                                }),
                                budget,
                            )?,
                            Kind::Start => valid = state.position == 0,
                            Kind::End => valid = state.position == input.len(),
                            Kind::Boundary(boundary) => {
                                valid =
                                    (word(state.position.checked_sub(1).and_then(|n| input.get(n)))
                                        != word(input.get(state.position)))
                                        == *boundary
                            }
                            Kind::Backref(group) => {
                                if let Some(range) = &state.captures[*group] {
                                    let end = state.position.checked_add(range.len()).context("grammar trigger backreference offset overflow; shorten the input")?;
                                    if end <= input.len() {
                                        budget.spend(range.len())?;
                                        valid = input[state.position..end] == input[range.clone()];
                                        if valid {
                                            state.position = end;
                                        }
                                    } else {
                                        valid = false;
                                    }
                                } else {
                                    valid = false;
                                }
                            }
                            Kind::Fail => valid = false,
                        }
                    }
                    Op::Close { group, start } => {
                        state.captures[group] = Some(start..state.position)
                    }
                    Op::Repeat(repetition) => {
                        let exit = repetition.count >= repetition.min;
                        let enter = repetition.max.is_none_or(|max| repetition.count < max);
                        if enter && exit {
                            if repetition.greedy {
                                branch(&state, &mut queued, budget)?;
                                take(&mut state, repetition, budget)?;
                            } else {
                                budget.reserve(state.units() + 2)?;
                                let mut alternative = state.clone();
                                // Those two queued operations were already admitted above.
                                budget.spend(repetition.captures.len())?;
                                for group in repetition.captures.clone() {
                                    alternative.captures[group] = None;
                                }
                                let mut next = repetition;
                                next.count += 1;
                                let body = next.body;
                                alternative.ops.push(Op::Repeated {
                                    repetition: next,
                                    start: alternative.position,
                                });
                                alternative.ops.push(Op::Node(body));
                                queued.push(alternative);
                            }
                        } else if enter {
                            take(&mut state, repetition, budget)?;
                        } else if !exit {
                            valid = false;
                        }
                    }
                    Op::Repeated { repetition, start } => {
                        // An empty iteration beyond the mandatory minimum must
                        // backtrack inside the body, not overwrite captures.
                        if state.position == start && repetition.count > repetition.min {
                            valid = false;
                        } else {
                            push(&mut state, Op::Repeat(repetition), budget)?;
                        }
                    }
                }
            } else if !full || state.position == input.len() {
                return Ok(Some(Matched {
                    end: state.position,
                    captures: state.captures,
                }));
            } else {
                valid = false;
            }
            if !valid {
                budget.live -= state.units();
                if let Some(alternative) = queued.pop() {
                    state = alternative;
                } else {
                    return Ok(None);
                }
            }
        }
    })();
    budget.live = entered;
    result
}
impl Program {
    pub(super) fn find(&self, input: &[u8]) -> Result<Option<usize>> {
        let fuel = input
            .len()
            .checked_mul(16)
            .and_then(|n| n.checked_add(1_000_000))
            .context("grammar trigger execution budget size overflow; shorten the input")?;
        let mut budget = Budget { fuel, live: 0 };
        let starts = if self.full { 0..=0 } else { 0..=input.len() };
        for start in starts {
            budget.spend(self.captures + 1)?;
            if let Some(found) = run(
                self,
                self.root,
                input,
                start,
                vec![None; self.captures + 1],
                self.full,
                &mut budget,
            )? {
                let offset = found
                    .captures
                    .iter()
                    .skip(1)
                    .flatten()
                    .find(|range| !range.is_empty())
                    .map_or(start, |range| range.start);
                ensure!(
                    found.end <= input.len() && offset <= input.len(),
                    "grammar trigger returned an invalid byte offset; restart with a freshly compiled pattern"
                );
                return Ok(Some(offset));
            }
        }
        Ok(None)
    }
}
