//! What carries a session-scoped cell from one invocation to the next.
//!
//! `scope: session` says a value outlives its invocation. It does not say how
//! the next invocation reaches it, and a package that leaves that unanswered
//! advertises a continuity it does not have. There are exactly three mechanisms
//! a document can name, and this module is the single place that reads a
//! workflow and says which one each session cell uses.
//!
//! It exists because the answer was being computed twice — once by the
//! validator, deciding whether a declaration was well formed, and once by the
//! runtime, deciding whether a session could be opened — and the two disagreed
//! about state service groups. A caller then held a package the validator
//! blessed and the runtime refused. One classification, both readers.

use std::collections::{BTreeMap, BTreeSet};

use crate::schema::{WorkflowSpec, WorkflowStateScope, WorkflowStep};

/// How one session-scoped cell's lease reaches the next invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SessionStateCarrier {
    /// The lease states the request binding it rejoins.
    ///
    /// Bound before the pass starts, so a component that never sees the cell
    /// still consumes the conversation through the input it already reads.
    PromptContinuation,
    /// A loop carries the cell; its lease seeds the carry in place of the
    /// initializer the document names.
    LoopCarry,
    /// A state service group holds the storage for the session.
    ///
    /// The group's alias names the ports the graph reads and writes, so the
    /// lease replaces the value the cell's initializer names and the alias's
    /// `output` port is what the next lease holds.
    StateServiceGroup,
}

/// Which mechanism carries each session-scoped cell of one workflow.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionStateFacts {
    carriers: BTreeMap<String, SessionStateCarrier>,
    uncarried: BTreeSet<String>,
}

impl SessionStateFacts {
    /// The carrier for one cell, or `None` when the cell is not session-scoped
    /// or nothing carries it.
    pub fn carrier(&self, cell: &str) -> Option<SessionStateCarrier> {
        self.carriers.get(cell).copied()
    }

    /// Every session-scoped cell that something carries, with its mechanism.
    pub fn carried(&self) -> impl Iterator<Item = (&str, SessionStateCarrier)> {
        self.carriers
            .iter()
            .map(|(cell, carrier)| (cell.as_str(), *carrier))
    }

    /// Session-scoped cells nothing carries.
    ///
    /// Such a cell is written back on every pass and read by nothing, so a
    /// session over it restarts every turn. Reporting them is what lets a
    /// runtime refuse a session rather than hand out a conversation it cannot
    /// continue.
    pub fn uncarried(&self) -> impl Iterator<Item = &str> {
        self.uncarried.iter().map(String::as_str)
    }

    /// Whether any session-scoped cell is carried.
    pub fn carries_any(&self) -> bool {
        !self.carriers.is_empty()
    }

    /// The sole cell whose lease rejoins the request binding, if one exists.
    ///
    /// Validation admits at most one, so this returns it rather than choosing
    /// between two: a package with two conversations is refused at load.
    pub fn prompt_continuation(&self) -> Option<&str> {
        self.carriers
            .iter()
            .find(|(_, carrier)| **carrier == SessionStateCarrier::PromptContinuation)
            .map(|(cell, _)| cell.as_str())
    }
}

/// Read a workflow once and say what carries each of its session-scoped cells.
///
/// A cell may satisfy more than one mechanism — a KV group is routinely both
/// group-backed and loop-carried — and the reported carrier is the one the
/// runtime actually uses, in the order a pass reaches them: the request binding
/// before the pass, the loop carry on entry, the group's alias otherwise.
pub fn classify_session_state(workflow: &WorkflowSpec) -> SessionStateFacts {
    let carried_by_loop = loop_carried_cells(&workflow.steps);
    let groups = workflow
        .serving
        .as_ref()
        .map(|serving| &serving.state_service.groups);

    let mut facts = SessionStateFacts::default();
    for (cell, state) in &workflow.state {
        if state.scope != WorkflowStateScope::Session {
            continue;
        }
        let carrier = if state
            .session
            .as_ref()
            .is_some_and(|lease| lease.continuation.is_some())
        {
            Some(SessionStateCarrier::PromptContinuation)
        } else if carried_by_loop.contains(cell) {
            Some(SessionStateCarrier::LoopCarry)
        } else if state
            .service_group
            .as_deref()
            .and_then(|group| groups.and_then(|groups| groups.get(group)))
            .is_some_and(|group| {
                group
                    .ports
                    .values()
                    .any(|component| component.contains_key(cell))
            })
        {
            // A group that does not exist, or one whose aliases never name this
            // cell, holds nothing — so it is not a carrier, and the validator
            // reports the same declaration as an error.
            Some(SessionStateCarrier::StateServiceGroup)
        } else {
            None
        };
        match carrier {
            Some(carrier) => {
                facts.carriers.insert(cell.clone(), carrier);
            }
            None => {
                facts.uncarried.insert(cell.clone());
            }
        }
    }
    facts
}

/// Cells a loop carries, at any nesting depth.
pub(crate) fn loop_carried_cells(steps: &[WorkflowStep]) -> BTreeSet<String> {
    fn walk(step: &WorkflowStep, cells: &mut BTreeSet<String>) {
        match step {
            WorkflowStep::Sequence { steps } => steps.iter().for_each(|step| walk(step, cells)),
            WorkflowStep::Loop {
                setup,
                steps,
                carried,
                ..
            } => {
                cells.extend(carried.iter().map(|carry| carry.cell.clone()));
                setup.iter().for_each(|step| walk(step, cells));
                steps.iter().for_each(|step| walk(step, cells));
            }
            WorkflowStep::Branch { cases, default, .. } => {
                cases.values().for_each(|step| walk(step, cells));
                if let Some(default) = default {
                    walk(default, cells);
                }
            }
            WorkflowStep::Invoke { .. } | WorkflowStep::Emit { .. } => {}
        }
    }
    let mut cells = BTreeSet::new();
    steps.iter().for_each(|step| walk(step, &mut cells));
    cells
}

/// The graph ports a state service group aliases for one session-scoped cell.
///
/// The `input` port is where the lease is read and the `output` port is what
/// the next lease holds, so a runtime carrying group-backed state needs both
/// and needs them per component.
pub fn session_group_aliases<'a>(
    workflow: &'a WorkflowSpec,
    cell: &str,
) -> Vec<(&'a str, &'a crate::schema::StatePortAlias)> {
    let Some(group) = workflow
        .state
        .get(cell)
        .and_then(|state| state.service_group.as_deref())
    else {
        return Vec::new();
    };
    workflow
        .serving
        .as_ref()
        .and_then(|serving| serving.state_service.groups.get(group))
        .map(|group| {
            group
                .ports
                .iter()
                .filter_map(|(component, aliases)| {
                    aliases.get(cell).map(|alias| (component.as_str(), alias))
                })
                .collect()
        })
        .unwrap_or_default()
}
