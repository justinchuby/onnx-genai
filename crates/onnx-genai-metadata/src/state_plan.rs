//! The resolved cross-invocation state contract.
//!
//! Serialized state declarations describe graph bindings.  This module reads
//! those bindings once and produces the plan consumed by validation and the
//! runtime.  In particular, storage-management declarations never participate
//! in selecting an initializer or final writer.

use std::collections::{BTreeMap, BTreeSet};

use crate::schema::{
    StateReleaseBoundary, StateUpdate, WorkflowCarry, WorkflowSpec, WorkflowStateClass,
    WorkflowStateScope, WorkflowStep,
};

/// Stable semantic identity of one mutable workflow location.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StateIdentity(pub String);

/// A graph value that seeds a state cell before its first write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSource {
    pub binding: String,
}

/// Why a loop carry reads a particular binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateCarrySourceKind {
    Initializer,
    Explicit,
    PriorState,
}

/// The resolved input to one loop-carried state edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCarrySource {
    pub source: StateSource,
    pub kind: StateCarrySourceKind,
}

/// A component port or loop edge that observes a state cell.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum StateReader {
    LoopCarry {
        binding: String,
    },
    ComponentPort {
        component: String,
        port: String,
        binding: String,
    },
}

/// A component port or loop edge that advances a state cell.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StateWriter {
    pub id: String,
    pub component: Option<String>,
    pub port: Option<String>,
    pub binding: String,
}

/// The one dataflow path whose value is committed for the next invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateFinalWriter {
    Writer(StateWriter),
    Continuation {
        prompt_input: String,
        output: String,
    },
}

/// The graph relation used when a writer advances state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateUpdateRelation {
    Append,
    Replace,
    Indexed,
}

/// Lifecycle facts that define semantic retention, independently of storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateLifecycle {
    pub scope: WorkflowStateScope,
    pub release: Option<StateReleaseBoundary>,
}

/// Snapshot and fork eligibility of a cell's complete semantic value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StateSnapshotParticipation {
    pub snapshot: bool,
    pub fork: bool,
}

/// Whether a state cell must join the invocation transaction.
///
/// Semantic cells participate; advisory cells are deliberately outside the
/// semantic write set and may be discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateTransactionParticipation {
    pub required: bool,
}

/// One fully resolved state/dataflow contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedStateCell {
    pub identity: StateIdentity,
    pub lifecycle: StateLifecycle,
    pub source: StateSource,
    pub readers: Vec<StateReader>,
    pub writers: Vec<StateWriter>,
    pub update: StateUpdateRelation,
    pub final_writer: Option<StateFinalWriter>,
    pub snapshot: StateSnapshotParticipation,
    pub transaction: StateTransactionParticipation,
}

/// Canonical state contract for a workflow.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedStatePlan {
    cells: BTreeMap<String, ResolvedStateCell>,
    carry_sources: BTreeMap<(String, String), StateCarrySource>,
    terminal_candidates: BTreeMap<String, BTreeSet<TerminalCandidate>>,
    flow_errors: Vec<String>,
}

impl ResolvedStatePlan {
    pub fn cell(&self, identity: &str) -> Option<&ResolvedStateCell> {
        self.cells.get(identity)
    }

    pub fn cells(&self) -> impl Iterator<Item = (&str, &ResolvedStateCell)> {
        self.cells
            .iter()
            .map(|(identity, cell)| (identity.as_str(), cell))
    }

    pub fn session_cells(&self) -> impl Iterator<Item = (&str, &ResolvedStateCell)> {
        self.cells()
            .filter(|(_, cell)| cell.lifecycle.scope == WorkflowStateScope::Session)
    }

    pub fn carry_source(&self, carry: &WorkflowCarry) -> Option<&StateCarrySource> {
        self.carry_sources
            .get(&(carry.cell.clone(), carry.next.clone()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TerminalCandidate {
    binding: String,
    writer: Option<StateWriter>,
    /// Component writers whose output representation reaches this terminal
    /// binding through control-flow joins and loop carries.
    origins: BTreeSet<StateWriter>,
    provenance: ControlFlowProvenance,
}

/// The structural scope that owns a terminal value.
///
/// A component/port/binding triple identifies a value producer, but not the
/// branch edge that makes that producer's value visible. Branch cases must keep
/// their identity until a declared phi exports the selected value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ControlFlowProvenance {
    Enclosing,
    BranchEdge { branch: usize, edge: usize },
}

impl ControlFlowProvenance {
    fn is_local_to(&self, branch: usize) -> bool {
        matches!(self, Self::BranchEdge { branch: owner, .. } if *owner == branch)
    }
}

#[derive(Debug, Default)]
struct StateFlowAnalysis {
    carries: Vec<(String, String, StateCarrySource)>,
    carry_sources: BTreeMap<(String, String), StateCarrySource>,
    terminal_candidates: BTreeMap<String, BTreeSet<TerminalCandidate>>,
    errors: Vec<String>,
    next_branch: usize,
}

/// Resolve state source, readers, writers, and final writers from explicit
/// workflow bindings.  This function deliberately does not read
/// `StateManagement`: allocation policy cannot change dataflow.
pub fn resolve_state_plan(workflow: &WorkflowSpec) -> ResolvedStatePlan {
    let flow = analyze_state_flow(workflow);
    let groups = workflow
        .serving
        .as_ref()
        .map(|serving| &serving.state_service.groups);
    let mut cells = BTreeMap::new();

    for (name, state) in &workflow.state {
        let mut readers = Vec::new();
        let mut writers = Vec::new();
        for (cell, next, source) in flow.carries.iter().filter(|(cell, _, _)| cell == name) {
            readers.push(StateReader::LoopCarry {
                binding: source.source.binding.clone(),
            });
            writers.push(StateWriter {
                id: format!("loop:{next}"),
                component: None,
                port: None,
                binding: cell.clone(),
            });
        }

        let mut update = StateUpdateRelation::Replace;
        let mut snapshot = StateSnapshotParticipation::default();
        if let Some(group) = state
            .service_group
            .as_deref()
            .and_then(|name| groups.and_then(|groups| groups.get(name)))
        {
            update = match group.update.as_ref() {
                Some(StateUpdate::Append) | None => StateUpdateRelation::Append,
                Some(StateUpdate::Replace) => StateUpdateRelation::Replace,
                Some(StateUpdate::IndexedScatter { .. }) => StateUpdateRelation::Indexed,
            };
            snapshot = StateSnapshotParticipation {
                snapshot: group.capabilities.snapshot,
                fork: group.capabilities.fork,
            };
            for (component, aliases) in &group.ports {
                let Some(alias) = aliases.get(name) else {
                    continue;
                };
                let read_bindings =
                    component_port_bindings(&workflow.steps, component, &alias.input);
                for binding in read_bindings {
                    readers.push(StateReader::ComponentPort {
                        component: component.clone(),
                        port: alias.input.clone(),
                        binding,
                    });
                }
                if let Some(port) = &alias.output {
                    for binding in component_port_bindings(&workflow.steps, component, port) {
                        writers.push(StateWriter {
                            id: format!("{component}:{port}:{binding}"),
                            component: Some(component.clone()),
                            port: Some(port.clone()),
                            binding,
                        });
                    }
                }
            }
        }

        let terminal_writer = flow
            .terminal_candidates
            .get(name)
            .filter(|candidates| candidates.len() == 1)
            .and_then(|candidates| candidates.iter().next())
            .and_then(|candidate| candidate.writer.clone());
        if let Some(writer) = &terminal_writer
            && !writers.iter().any(|candidate| candidate.id == writer.id)
        {
            writers.push(writer.clone());
        }
        let final_writer = state
            .session
            .as_ref()
            .and_then(|lease| lease.continuation.as_ref())
            .map(|continuation| match continuation {
                crate::schema::SessionContinuation::PromptPrefix {
                    prompt_input,
                    tokens_output,
                } => StateFinalWriter::Continuation {
                    prompt_input: prompt_input.clone(),
                    output: tokens_output.clone(),
                },
            })
            .or_else(|| terminal_writer.map(StateFinalWriter::Writer));

        cells.insert(
            name.clone(),
            ResolvedStateCell {
                identity: StateIdentity(name.clone()),
                lifecycle: StateLifecycle {
                    scope: state.scope.clone(),
                    release: state.release_boundary,
                },
                source: StateSource {
                    binding: state.initializer.clone(),
                },
                readers,
                writers,
                update,
                final_writer,
                snapshot,
                transaction: StateTransactionParticipation {
                    required: state.class == WorkflowStateClass::Semantic,
                },
            },
        );
    }
    ResolvedStatePlan {
        cells,
        carry_sources: flow.carry_sources,
        terminal_candidates: flow.terminal_candidates,
        flow_errors: flow.errors,
    }
}

/// Validate the canonical plan.  Diagnostics name the affected state and
/// binding, and describe the missing declaration needed to repair it.
pub fn validate_state_plan(workflow: &WorkflowSpec, plan: &ResolvedStatePlan) -> Vec<String> {
    let mut errors = plan.flow_errors.clone();
    let mut source_edges = BTreeMap::new();

    for (name, cell) in plan.cells() {
        if workflow.state.contains_key(&cell.source.binding) {
            source_edges.insert(name.to_string(), cell.source.binding.clone());
        }

        let mut readers = BTreeSet::new();
        for reader in &cell.readers {
            if !readers.insert(reader.clone()) {
                errors.push(format!(
                    "pipeline.workflow.state.{name} declares reader {reader:?} more than once; \
                     keep one binding for each read edge"
                ));
            }
        }
        let mut writers = BTreeSet::new();
        for writer in &cell.writers {
            if !writers.insert(writer.id.clone()) {
                errors.push(format!(
                    "pipeline.workflow.state.{name} declares writer '{}' more than once; keep \
                     one binding for each write edge",
                    writer.id
                ));
            }
        }

        if cell.lifecycle.scope == WorkflowStateScope::Session
            && cell.transaction.required
            && cell.final_writer.is_none()
        {
            errors.push(format!(
                "pipeline.workflow.state.{name} is semantic session state but has no \
                 unambiguous final writer; bind exactly one writer, carry it through a loop, \
                 or declare a continuation output"
            ));
        }
        if cell.lifecycle.scope == WorkflowStateScope::Session
            && cell.transaction.required
            && cell.final_writer.is_none()
            && let Some(candidates) = plan
                .terminal_candidates
                .get(name)
                .filter(|candidates| candidates.len() > 1)
        {
            let candidates = candidates
                .iter()
                .map(describe_terminal_candidate)
                .collect::<Vec<_>>()
                .join(", ");
            errors.push(format!(
                "pipeline.workflow.state.{name} has ambiguous terminal writers [{candidates}]; \
                 route every path through one explicit branch output or loop carry before commit"
            ));
        }
        if cell.lifecycle.scope == WorkflowStateScope::Session && cell.transaction.required {
            let persisted = &workflow
                .state
                .get(name)
                .expect("resolved state cell is declared")
                .contract;
            let incompatible_terminal = match &cell.final_writer {
                Some(StateFinalWriter::Continuation { output, .. }) => {
                    workflow.outputs.get(output).is_some_and(|output| {
                        !output.contract.representation_compatible_with(persisted)
                    })
                }
                Some(StateFinalWriter::Writer(_)) => plan
                    .terminal_candidates
                    .get(name)
                    .filter(|candidates| candidates.len() == 1)
                    .into_iter()
                    .flatten()
                    .flat_map(|candidate| &candidate.origins)
                    .any(|origin| {
                        origin
                            .component
                            .as_deref()
                            .zip(origin.port.as_deref())
                            .and_then(|(component, port)| {
                                workflow
                                    .components
                                    .get(component)
                                    .and_then(|component| component.ports.outputs.get(port))
                            })
                            .is_some_and(|contract| {
                                !contract.representation_compatible_with(persisted)
                            })
                    }),
                None => false,
            };
            if incompatible_terminal {
                let writer = match cell
                    .final_writer
                    .as_ref()
                    .expect("matched only when a final writer exists")
                {
                    StateFinalWriter::Writer(writer) => writer
                        .component
                        .as_deref()
                        .zip(writer.port.as_deref())
                        .map(|(component, port)| format!("{component}:{port}"))
                        .unwrap_or_else(|| writer.id.clone()),
                    StateFinalWriter::Continuation { output, .. } => {
                        format!("continuation output '{output}'")
                    }
                };
                errors.push(format!(
                    "pipeline.workflow.state.{name} terminal writer '{writer}' persists a \
                     representation incompatible with its persisted state contract; add an \
                     explicit typed, versioned conversion back to the state representation and \
                     make that conversion the final writer"
                ));
            }
        }

        if cell.lifecycle.scope != WorkflowStateScope::Session || !cell.transaction.required {
            continue;
        }
        let Some(group) = workflow
            .state
            .get(name)
            .and_then(|state| state.service_group.as_deref())
            .and_then(|group| {
                workflow
                    .serving
                    .as_ref()
                    .and_then(|serving| serving.state_service.groups.get(group))
            })
        else {
            continue;
        };
        for (component, aliases) in &group.ports {
            let Some(alias) = aliases.get(name) else {
                continue;
            };
            if !cell.readers.iter().any(|reader| {
                matches!(
                    reader,
                    StateReader::ComponentPort {
                        component: reader_component,
                        port,
                        ..
                    } if reader_component == component && port == &alias.input
                )
            }) {
                errors.push(format!(
                    "pipeline.workflow.state.{name} reader '{component}:{}' has no invocation \
                     binding; bind that input port in a workflow step before execution",
                    alias.input
                ));
            }
            if let Some(output) = &alias.output
                && !cell.writers.iter().any(|writer| {
                    writer.component.as_deref() == Some(component)
                        && writer.port.as_deref() == Some(output)
                })
            {
                errors.push(format!(
                    "pipeline.workflow.state.{name} writer '{component}:{output}' has no \
                     invocation binding; bind that output port or remove the writer declaration"
                ));
            }
        }
    }

    for start in source_edges.keys() {
        let mut seen = BTreeSet::new();
        let mut cursor = start.as_str();
        while let Some(next) = source_edges.get(cursor) {
            // A self reference is an existing cell's retained value, not a
            // source cycle: it is how an invocation-scoped advisory cell can
            // state "retain the value I was given" without inventing an
            // otherwise-unused input.  A cycle across distinct identities has
            // no external seed and remains invalid.
            if next == cursor {
                break;
            }
            if !seen.insert(cursor.to_string()) || next == start {
                errors.push(format!(
                    "pipeline.workflow.state.{start} has a cyclic initialization path through \
                     state '{next}'; seed one state from an input or produced binding"
                ));
                break;
            }
            cursor = next;
        }
    }
    errors
}

fn analyze_state_flow(workflow: &WorkflowSpec) -> StateFlowAnalysis {
    type Flow = BTreeMap<String, BTreeSet<TerminalCandidate>>;

    fn singleton(
        binding: String,
        writer: Option<StateWriter>,
        provenance: ControlFlowProvenance,
    ) -> BTreeSet<TerminalCandidate> {
        let origins = writer.iter().cloned().collect();
        BTreeSet::from([TerminalCandidate {
            binding,
            writer,
            origins,
            provenance,
        }])
    }

    fn derived_singleton(
        binding: String,
        writer: StateWriter,
        origins: BTreeSet<StateWriter>,
        provenance: ControlFlowProvenance,
    ) -> BTreeSet<TerminalCandidate> {
        BTreeSet::from([TerminalCandidate {
            binding,
            writer: Some(writer),
            origins,
            provenance,
        }])
    }

    fn invoke_writers(
        workflow: &WorkflowSpec,
        component: &str,
        outputs: &BTreeMap<String, String>,
    ) -> Vec<(String, StateWriter)> {
        let groups = workflow
            .serving
            .as_ref()
            .map(|serving| &serving.state_service.groups);
        let mut writers = Vec::new();
        for (cell, state) in &workflow.state {
            let Some(aliases) = state
                .service_group
                .as_deref()
                .and_then(|group| groups.and_then(|groups| groups.get(group)))
                .and_then(|group| group.ports.get(component))
                .and_then(|aliases| aliases.get(cell))
            else {
                continue;
            };
            let Some(port) = aliases.output.as_deref() else {
                continue;
            };
            let Some(binding) = outputs.get(port) else {
                continue;
            };
            writers.push((
                cell.clone(),
                StateWriter {
                    id: format!("{component}:{port}:{binding}"),
                    component: Some(component.to_string()),
                    port: Some(port.to_string()),
                    binding: binding.clone(),
                },
            ));
        }
        writers
    }

    fn state_port_alias<'a>(
        workflow: &'a WorkflowSpec,
        cell: &str,
        component: &str,
    ) -> Option<&'a crate::schema::StatePortAlias> {
        let state = workflow.state.get(cell)?;
        workflow
            .serving
            .as_ref()?
            .state_service
            .groups
            .get(state.service_group.as_deref()?)?
            .ports
            .get(component)?
            .get(cell)
    }

    fn candidate_contract<'a>(
        workflow: &'a WorkflowSpec,
        candidate: &TerminalCandidate,
    ) -> Option<&'a crate::schema::TensorContract> {
        let writer = candidate.writer.as_ref()?;
        let component = workflow.components.get(writer.component.as_deref()?)?;
        component.ports.outputs.get(writer.port.as_deref()?)
    }

    fn validate_state_representation_transition(
        workflow: &WorkflowSpec,
        component: &str,
        inputs: &BTreeMap<String, String>,
        outputs: &BTreeMap<String, String>,
        flow: &Flow,
        analysis: &mut StateFlowAnalysis,
    ) {
        let Some(declaration) = workflow.components.get(component) else {
            return;
        };
        for (cell, state) in &workflow.state {
            let Some(alias) = state_port_alias(workflow, cell, component) else {
                continue;
            };
            let Some(input_binding) = inputs.get(&alias.input) else {
                continue;
            };
            let Some(input_contract) = declaration.ports.inputs.get(&alias.input) else {
                continue;
            };
            let Some(candidates) = flow.get(cell) else {
                continue;
            };
            if candidates.len() == 1 {
                let candidate = candidates.iter().next().expect("one candidate");
                let source_contract =
                    candidate_contract(workflow, candidate).unwrap_or(&state.contract);
                if !source_contract.representation_compatible_with(input_contract) {
                    analysis.errors.push(format!(
                        "pipeline.workflow state '{cell}' flows from '{}' into \
                         '{component}:{}' as binding '{input_binding}', but their tensor \
                         representations are incompatible; insert an explicit typed, versioned \
                         conversion component before this reader",
                        candidate.binding, alias.input
                    ));
                }
            }

            let Some(output_port) = alias.output.as_deref() else {
                continue;
            };
            if !outputs.contains_key(output_port) {
                continue;
            }
            let Some(output_contract) = declaration.ports.outputs.get(output_port) else {
                continue;
            };
            if input_contract.representation_compatible_with(output_contract) {
                continue;
            }
            let versioned = declaration.contract.as_ref().is_some_and(|contract| {
                !contract.id.trim().is_empty() && !contract.version.trim().is_empty()
            });
            if !versioned {
                analysis.errors.push(format!(
                    "pipeline.workflow component '{component}' changes state '{cell}' \
                     representation from input '{}' to output '{output_port}' without a \
                     versioned component contract; declare the conversion protocol id and \
                     version",
                    alias.input
                ));
            }
        }
    }

    fn walk_sequence(
        workflow: &WorkflowSpec,
        steps: &[WorkflowStep],
        mut flow: Flow,
        analysis: &mut StateFlowAnalysis,
        provenance: ControlFlowProvenance,
    ) -> Flow {
        for step in steps {
            flow = walk_step(workflow, step, flow, analysis, provenance.clone());
        }
        flow
    }

    fn walk_step(
        workflow: &WorkflowSpec,
        step: &WorkflowStep,
        mut flow: Flow,
        analysis: &mut StateFlowAnalysis,
        provenance: ControlFlowProvenance,
    ) -> Flow {
        match step {
            WorkflowStep::Sequence { steps } => {
                walk_sequence(workflow, steps, flow, analysis, provenance)
            }
            WorkflowStep::Invoke {
                component,
                inputs,
                outputs,
                ..
            } => {
                validate_state_representation_transition(
                    workflow, component, inputs, outputs, &flow, analysis,
                );
                for (cell, writer) in invoke_writers(workflow, component, outputs) {
                    flow.insert(
                        cell,
                        singleton(writer.binding.clone(), Some(writer), provenance.clone()),
                    );
                }
                flow
            }
            WorkflowStep::Loop {
                setup,
                steps,
                carried,
                ..
            } => {
                let setup_flow = walk_sequence(workflow, setup, flow, analysis, provenance.clone());
                let mut body_flow = setup_flow.clone();
                let mut carried_cells = BTreeSet::new();
                for carry in carried {
                    if !carried_cells.insert(carry.cell.clone()) {
                        analysis.errors.push(format!(
                            "pipeline.workflow.state.{} is carried more than once by one loop; \
                             keep one carry edge for the state cell",
                            carry.cell
                        ));
                    }
                    let Some(state) = workflow.state.get(&carry.cell) else {
                        analysis.errors.push(format!(
                            "pipeline.workflow loop carries unknown state cell '{}'",
                            carry.cell
                        ));
                        continue;
                    };
                    let source = if let Some(initial) = &carry.initial {
                        StateCarrySource {
                            source: StateSource {
                                binding: initial.clone(),
                            },
                            kind: StateCarrySourceKind::Explicit,
                        }
                    } else {
                        let candidates = setup_flow.get(&carry.cell);
                        let candidate = candidates
                            .filter(|candidates| candidates.len() == 1)
                            .and_then(|candidates| candidates.iter().next());
                        match candidate {
                            Some(candidate) => StateCarrySource {
                                source: StateSource {
                                    binding: candidate.binding.clone(),
                                },
                                kind: if candidate.writer.is_none()
                                    && candidate.binding == state.initializer
                                {
                                    StateCarrySourceKind::Initializer
                                } else {
                                    StateCarrySourceKind::PriorState
                                },
                            },
                            None => {
                                let candidates = candidates
                                    .map(|candidates| {
                                        candidates
                                            .iter()
                                            .map(describe_terminal_candidate)
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    })
                                    .unwrap_or_else(|| "none".to_string());
                                analysis.errors.push(format!(
                                    "pipeline.workflow.state.{} has ambiguous carry-in sources \
                                     [{candidates}] before loop writer 'loop:{}'; set 'initial' \
                                     explicitly or join the paths into one binding",
                                    carry.cell, carry.next
                                ));
                                StateCarrySource {
                                    source: StateSource {
                                        binding: state.initializer.clone(),
                                    },
                                    kind: StateCarrySourceKind::Initializer,
                                }
                            }
                        }
                    };
                    let key = (carry.cell.clone(), carry.next.clone());
                    if analysis.carry_sources.insert(key, source.clone()).is_some() {
                        analysis.errors.push(format!(
                            "pipeline.workflow.state.{} reuses loop writer 'loop:{}'; each carry \
                             edge must have a unique next binding",
                            carry.cell, carry.next
                        ));
                    }
                    analysis
                        .carries
                        .push((carry.cell.clone(), carry.next.clone(), source));
                    body_flow.insert(
                        carry.cell.clone(),
                        singleton(carry.cell.clone(), None, provenance.clone()),
                    );
                }
                let body_flow =
                    walk_sequence(workflow, steps, body_flow, analysis, provenance.clone());
                let mut exit_flow = setup_flow;
                for (cell, candidates) in &body_flow {
                    if !carried_cells.contains(cell) {
                        exit_flow
                            .entry(cell.clone())
                            .or_default()
                            .extend(candidates.iter().cloned());
                    }
                }
                for carry in carried {
                    let writer = StateWriter {
                        id: format!("loop:{}", carry.next),
                        component: None,
                        port: None,
                        binding: carry.cell.clone(),
                    };
                    let origins = body_flow
                        .get(&carry.cell)
                        .into_iter()
                        .flatten()
                        .flat_map(|candidate| candidate.origins.iter().cloned())
                        .collect();
                    exit_flow.insert(
                        carry.cell.clone(),
                        derived_singleton(carry.cell.clone(), writer, origins, provenance.clone()),
                    );
                }
                exit_flow
            }
            WorkflowStep::Branch {
                cases,
                default,
                outputs,
                ..
            } => {
                let branch = analysis.next_branch;
                analysis.next_branch += 1;
                let case_flows = cases
                    .iter()
                    .enumerate()
                    .map(|(edge, (case, step))| {
                        (
                            case.as_str(),
                            walk_step(
                                workflow,
                                step,
                                flow.clone(),
                                analysis,
                                ControlFlowProvenance::BranchEdge { branch, edge },
                            ),
                        )
                    })
                    .collect::<Vec<_>>();
                let default_flow = default.as_deref().map(|step| {
                    walk_step(
                        workflow,
                        step,
                        flow.clone(),
                        analysis,
                        ControlFlowProvenance::BranchEdge {
                            branch,
                            edge: cases.len(),
                        },
                    )
                });
                let mut joined = flow.clone();
                for cell in workflow.state.keys() {
                    let mut candidates = BTreeSet::new();
                    for (_, case_flow) in &case_flows {
                        candidates.extend(case_flow.get(cell).cloned().unwrap_or_default());
                    }
                    if let Some(default_flow) = &default_flow {
                        candidates.extend(default_flow.get(cell).cloned().unwrap_or_default());
                    }
                    if candidates.len() <= 1
                        && candidates
                            .iter()
                            .all(|candidate| !candidate.provenance.is_local_to(branch))
                    {
                        joined.insert(cell.clone(), candidates);
                        continue;
                    }
                    let joins = outputs
                        .iter()
                        .filter(|(_, phi)| {
                            case_flows.iter().all(|(case, case_flow)| {
                                let Some(source) = phi.cases.get(*case) else {
                                    return false;
                                };
                                case_flow.get(cell).is_some_and(|candidates| {
                                    candidates.len() == 1
                                        && candidates
                                            .iter()
                                            .next()
                                            .is_some_and(|candidate| candidate.binding == *source)
                                })
                            }) && match (&default_flow, &phi.default) {
                                (Some(default_flow), Some(source)) => {
                                    default_flow.get(cell).is_some_and(|candidates| {
                                        candidates.len() == 1
                                            && candidates.iter().next().is_some_and(|candidate| {
                                                candidate.binding == *source
                                            })
                                    })
                                }
                                (None, _) => true,
                                _ => false,
                            }
                        })
                        .map(|(output, _)| output)
                        .collect::<Vec<_>>();
                    if joins.len() == 1 {
                        let output = joins[0];
                        let writer = StateWriter {
                            id: format!("branch:{output}"),
                            component: None,
                            port: None,
                            binding: output.clone(),
                        };
                        let origins = candidates
                            .iter()
                            .flat_map(|candidate| candidate.origins.iter().cloned())
                            .collect();
                        joined.insert(
                            cell.clone(),
                            derived_singleton(output.clone(), writer, origins, provenance.clone()),
                        );
                    } else {
                        joined.insert(cell.clone(), candidates);
                    }
                }
                joined
            }
            WorkflowStep::Emit { .. } => flow,
        }
    }

    let initial = workflow
        .state
        .iter()
        .map(|(cell, state)| {
            (
                cell.clone(),
                singleton(
                    state.initializer.clone(),
                    None,
                    ControlFlowProvenance::Enclosing,
                ),
            )
        })
        .collect();
    let mut analysis = StateFlowAnalysis::default();
    analysis.terminal_candidates = walk_sequence(
        workflow,
        &workflow.steps,
        initial,
        &mut analysis,
        ControlFlowProvenance::Enclosing,
    );
    analysis
}

fn describe_terminal_candidate(candidate: &TerminalCandidate) -> String {
    candidate.writer.as_ref().map_or_else(
        || format!("initializer '{}'", candidate.binding),
        |writer| format!("'{}' at '{}'", writer.id, writer.binding),
    )
}

fn component_port_bindings(steps: &[WorkflowStep], component: &str, port: &str) -> Vec<String> {
    let mut bindings = Vec::new();
    fn walk(steps: &[WorkflowStep], component: &str, port: &str, bindings: &mut Vec<String>) {
        for step in steps {
            match step {
                WorkflowStep::Sequence { steps } => {
                    walk(steps, component, port, bindings);
                }
                WorkflowStep::Invoke {
                    component: invoked,
                    inputs,
                    outputs,
                } if invoked == component => {
                    if let Some(binding) = inputs.get(port).or_else(|| outputs.get(port)) {
                        bindings.push(binding.clone());
                    }
                }
                WorkflowStep::Loop { setup, steps, .. } => {
                    walk(setup, component, port, bindings);
                    walk(steps, component, port, bindings);
                }
                WorkflowStep::Branch { cases, default, .. } => {
                    for case in cases.values() {
                        walk(std::slice::from_ref(case), component, port, bindings);
                    }
                    if let Some(default) = default {
                        walk(
                            std::slice::from_ref(default.as_ref()),
                            component,
                            port,
                            bindings,
                        );
                    }
                }
                WorkflowStep::Invoke { .. } | WorkflowStep::Emit { .. } => {}
            }
        }
    }
    walk(steps, component, port, &mut bindings);
    bindings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InferenceMetadata;

    fn workflow(document: &str) -> WorkflowSpec {
        serde_yaml::from_str::<InferenceMetadata>(document)
            .expect("fixture parses")
            .pipeline
            .expect("fixture has a pipeline")
            .workflow
    }

    fn split_representation_workflow() -> WorkflowSpec {
        serde_yaml::from_str(
            r#"
manifest: {}
inputs:
  seed:
    contract: { dtype: float32, shape: [batch, width] }
    role: { kind: opaque }
    source: { kind: application, name: seed }
outputs: {}
components:
  first:
    implementation: { kind: binding }
    ports:
      inputs:
        state: { dtype: float32, shape: [batch, width] }
      outputs:
        state: { dtype: float32, shape: [batch, width] }
  convert:
    implementation: { kind: binding }
    contract: { id: test.state-conversion, version: "1" }
    ports:
      inputs:
        state: { dtype: float32, shape: [batch, width] }
      outputs:
        state: { dtype: float16, shape: [batch, width] }
  last:
    implementation: { kind: binding }
    ports:
      inputs:
        state: { dtype: float16, shape: [batch, width] }
      outputs:
        state: { dtype: float16, shape: [batch, width] }
  restore:
    implementation: { kind: binding }
    contract: { id: test.state-conversion, version: "1" }
    ports:
      inputs:
        state: { dtype: float16, shape: [batch, width] }
      outputs:
        state: { dtype: float32, shape: [batch, width] }
state:
  memory:
    contract: { dtype: float32, shape: [batch, width] }
    scope: session
    initializer: seed
    recurrence: { kind: invariant }
    management: runtime
    release_boundary: session
    service_group: memory
steps:
  - kind: invoke
    component: first
    inputs: { state: memory }
    outputs: { state: first.next }
  - kind: invoke
    component: convert
    inputs: { state: first.next }
    outputs: { state: converted.next }
  - kind: invoke
    component: last
    inputs: { state: converted.next }
    outputs: { state: last.next }
  - kind: invoke
    component: restore
    inputs: { state: last.next }
    outputs: { state: restored.next }
serving:
  active: active
  done: done
  accepted_len: accepted_len
  state_service:
    groups:
      memory:
        kind: recurrent
        layout: bw
        update: { kind: replace }
        ports:
          first:
            memory: { input: state, output: state }
          convert:
            memory: { input: state, output: state }
          last:
            memory: { input: state, output: state }
          restore:
            memory: { input: state, output: state }
"#,
        )
        .expect("split representation workflow")
    }

    #[test]
    fn one_plan_covers_external_and_loop_carried_state() {
        let external = workflow(include_str!(
            "../../../examples/inference_metadata/catalogue/01-gemma4-text-decoder.yaml"
        ));
        let external_plan = resolve_state_plan(&external);
        let key = external_plan
            .cell("full_key")
            .expect("attention state is planned");
        assert_eq!(key.source.binding, "request.full_key");
        assert!(
            key.readers
                .iter()
                .any(|reader| matches!(reader, StateReader::ComponentPort { .. }))
        );
        assert_eq!(key.update, StateUpdateRelation::Append);

        let carried = workflow(include_str!(
            "../../../examples/inference_metadata/catalogue/14-weathernext-rollout.yaml"
        ));
        let carried_plan = resolve_state_plan(&carried);
        let accumulator = carried_plan
            .cell("atmosphere")
            .expect("generic carried state is planned");
        assert_eq!(accumulator.source.binding, "request.atmosphere");
        assert!(
            accumulator
                .readers
                .iter()
                .any(|reader| matches!(reader, StateReader::LoopCarry { .. }))
        );
        assert_eq!(accumulator.update, StateUpdateRelation::Replace);
    }

    #[test]
    fn conformance_fixtures_cover_each_generic_update_relation() {
        let window = workflow(include_str!(
            "../../../examples/inference_metadata/catalogue/15-windowed-attention.yaml"
        ));
        assert!(
            resolve_state_plan(&window)
                .cells()
                .any(|(_, cell)| cell.update == StateUpdateRelation::Append)
        );

        let scatter = workflow(include_str!(
            "../../../examples/inference_metadata/catalogue/18-static-cache-indexed-scatter.yaml"
        ));
        assert!(
            resolve_state_plan(&scatter)
                .cells()
                .any(|(_, cell)| cell.update == StateUpdateRelation::Indexed)
        );

        let feature = workflow(include_str!(
            "../../../examples/inference_metadata/catalogue/14-weathernext-rollout.yaml"
        ));
        assert!(
            resolve_state_plan(&feature)
                .cells()
                .any(|(_, cell)| cell.source.binding == "request.atmosphere")
        );
    }

    #[test]
    fn missing_and_cyclic_sources_are_rejected_with_repair_guidance() {
        let mut missing = workflow(include_str!(
            "../../../examples/inference_metadata/catalogue/16-linear-attention-recurrent.yaml"
        ));
        missing
            .state
            .get_mut("linear_accumulator")
            .expect("fixture declares accumulator")
            .scope = WorkflowStateScope::Session;
        missing
            .serving
            .as_mut()
            .expect("fixture declares state service")
            .state_service
            .groups
            .get_mut("linear_accumulator")
            .expect("fixture declares state group")
            .ports
            .get_mut("model")
            .expect("fixture declares state ports")
            .get_mut("linear_accumulator")
            .expect("fixture declares accumulator port")
            .input = "not.a.binding".to_string();
        let errors = validate_state_plan(&missing, &resolve_state_plan(&missing));
        assert!(
            errors.iter().any(|error| {
                error.contains("linear_accumulator")
                    && error.contains("not.a.binding")
                    && error.contains("invocation binding")
            }),
            "{errors:?}"
        );

        let mut cyclic = workflow(include_str!(
            "../../../examples/inference_metadata/catalogue/16-linear-attention-recurrent.yaml"
        ));
        let second = cyclic
            .state
            .get("linear_accumulator")
            .expect("fixture declares accumulator")
            .clone();
        cyclic.state.insert("second_state".to_string(), second);
        cyclic
            .state
            .get_mut("linear_accumulator")
            .expect("fixture declares accumulator")
            .initializer = "second_state".to_string();
        cyclic
            .state
            .get_mut("second_state")
            .expect("test added second state")
            .initializer = "linear_accumulator".to_string();
        let errors = validate_state_plan(&cyclic, &resolve_state_plan(&cyclic));
        assert!(
            errors
                .iter()
                .any(|error| error.contains("cyclic initialization path")),
            "{errors:?}"
        );

        let mut missing_writer = workflow(include_str!(
            "../../../examples/inference_metadata/catalogue/16-linear-attention-recurrent.yaml"
        ));
        missing_writer
            .state
            .get_mut("linear_accumulator")
            .expect("fixture declares accumulator")
            .scope = WorkflowStateScope::Session;
        missing_writer
            .serving
            .as_mut()
            .expect("fixture declares state service")
            .state_service
            .groups
            .get_mut("linear_accumulator")
            .expect("fixture declares state group")
            .ports
            .get_mut("model")
            .expect("fixture declares state ports")
            .get_mut("linear_accumulator")
            .expect("fixture declares accumulator port")
            .output = Some("missing_output".to_string());
        let errors = validate_state_plan(&missing_writer, &resolve_state_plan(&missing_writer));
        assert!(
            errors.iter().any(|error| {
                error.contains("linear_accumulator")
                    && error.contains("missing_output")
                    && error.contains("writer")
                    && error.contains("invocation binding")
            }),
            "{errors:?}"
        );
    }

    #[test]
    fn split_state_representation_requires_explicit_versioned_conversion() {
        let workflow = split_representation_workflow();
        let errors = validate_state_plan(&workflow, &resolve_state_plan(&workflow));
        assert!(
            errors.iter().all(|error| !error.contains("representation")),
            "{errors:?}"
        );

        let mut unversioned = workflow.clone();
        unversioned
            .components
            .get_mut("convert")
            .expect("fixture declares converter")
            .contract = None;
        let errors = validate_state_plan(&unversioned, &resolve_state_plan(&unversioned));
        assert!(
            errors.iter().any(|error| {
                error.contains("convert")
                    && error.contains("changes state 'memory' representation")
                    && error.contains("versioned component contract")
            }),
            "{errors:?}"
        );

        let mut missing = workflow;
        missing.steps.remove(1);
        let WorkflowStep::Invoke { inputs, .. } =
            missing.steps.get_mut(1).expect("last reader remains")
        else {
            panic!("last step is an invocation");
        };
        inputs.insert("state".to_string(), "first.next".to_string());
        let errors = validate_state_plan(&missing, &resolve_state_plan(&missing));
        assert!(
            errors.iter().any(|error| {
                error.contains("state 'memory'")
                    && error.contains("first.next")
                    && error.contains("last:state")
                    && error.contains("explicit typed, versioned conversion")
            }),
            "{errors:?}"
        );

        let mut wrong_terminal = split_representation_workflow();
        wrong_terminal
            .components
            .get_mut("restore")
            .expect("fixture declares restore conversion")
            .ports
            .outputs
            .get_mut("state")
            .expect("restore declares state output")
            .dtype = "float16".to_string();
        let errors = validate_state_plan(&wrong_terminal, &resolve_state_plan(&wrong_terminal));
        assert!(
            errors.iter().any(|error| {
                error.contains("terminal writer 'restore:state'")
                    && error.contains("persisted state contract")
                    && error.contains("typed, versioned conversion")
            }),
            "{errors:?}"
        );

        let mut carried_terminal = split_representation_workflow();
        let mut steps = std::mem::take(&mut carried_terminal.steps);
        let convert = steps.remove(0);
        let last = steps.remove(0);
        carried_terminal.steps = vec![WorkflowStep::Loop {
            setup: vec![convert],
            steps: vec![last],
            continue_when: "active".to_string(),
            max_iterations: "request.max_iterations".to_string(),
            termination: crate::schema::WorkflowLoopTermination::Predicate,
            iteration: None,
            carried: vec![crate::schema::WorkflowCarry {
                cell: "memory".to_string(),
                initial: None,
                next: "loop.memory".to_string(),
            }],
        }];
        let errors = validate_state_plan(&carried_terminal, &resolve_state_plan(&carried_terminal));
        assert!(
            errors.iter().any(|error| {
                error.contains("terminal writer 'loop:loop.memory'")
                    && error.contains("persisted state contract")
            }),
            "{errors:?}"
        );

        let mut branch_terminal = split_representation_workflow();
        let mut steps = std::mem::take(&mut branch_terminal.steps);
        let first = steps.remove(0);
        let convert = steps.remove(0);
        let last = steps.remove(0);
        branch_terminal.steps = vec![
            first,
            convert,
            WorkflowStep::Branch {
                predicate: "active".to_string(),
                cases: BTreeMap::from([
                    ("false".to_string(), last.clone()),
                    ("true".to_string(), last),
                ]),
                default: None,
                outputs: BTreeMap::from([(
                    "joined.memory".to_string(),
                    crate::schema::WorkflowBranchOutput {
                        cases: BTreeMap::from([
                            ("false".to_string(), "last.next".to_string()),
                            ("true".to_string(), "last.next".to_string()),
                        ]),
                        default: None,
                    },
                )]),
            },
        ];
        let errors = validate_state_plan(&branch_terminal, &resolve_state_plan(&branch_terminal));
        assert!(
            errors.iter().any(|error| {
                error.contains("terminal writer 'branch:joined.memory'")
                    && error.contains("persisted state contract")
            }),
            "{errors:?}"
        );
    }

    #[test]
    fn resolved_state_plan_is_invariant_to_component_map_reordering() {
        let workflow = split_representation_workflow();
        let mut reordered = workflow.clone();
        reordered.components = std::mem::take(&mut reordered.components)
            .into_iter()
            .rev()
            .collect();
        let group = reordered
            .serving
            .as_mut()
            .expect("fixture declares serving")
            .state_service
            .groups
            .get_mut("memory")
            .expect("fixture declares memory group");
        group.ports = std::mem::take(&mut group.ports).into_iter().rev().collect();

        assert_eq!(
            resolve_state_plan(&workflow),
            resolve_state_plan(&reordered)
        );
    }

    #[test]
    fn sequential_loops_resolve_the_later_carry_as_the_final_writer() {
        let mut workflow = workflow(include_str!(
            "../../../examples/inference_metadata/catalogue/14-weathernext-rollout.yaml"
        ));
        let mut decode = workflow.steps[0].clone();
        let WorkflowStep::Loop { carried, .. } = &mut decode else {
            panic!("fixture starts with its rollout loop");
        };
        let atmosphere = carried
            .iter_mut()
            .find(|carry| carry.cell == "atmosphere")
            .expect("fixture carries atmosphere");
        atmosphere.next = "decode.atmosphere".to_string();
        workflow.steps.push(decode);
        let decode_index = workflow.steps.len() - 1;

        let plan = resolve_state_plan(&workflow);
        let state = plan.cell("atmosphere").expect("state is planned");
        assert_eq!(
            state.final_writer,
            Some(StateFinalWriter::Writer(StateWriter {
                id: "loop:decode.atmosphere".to_string(),
                component: None,
                port: None,
                binding: "atmosphere".to_string(),
            }))
        );
        let first = match &workflow.steps[0] {
            WorkflowStep::Loop { carried, .. } => carried
                .iter()
                .find(|carry| carry.cell == "atmosphere")
                .expect("first loop carries atmosphere"),
            _ => unreachable!(),
        };
        assert_eq!(
            plan.carry_source(first),
            Some(&StateCarrySource {
                source: StateSource {
                    binding: "request.atmosphere".to_string(),
                },
                kind: StateCarrySourceKind::Initializer,
            })
        );
        let second = match &workflow.steps[decode_index] {
            WorkflowStep::Loop { carried, .. } => carried
                .iter()
                .find(|carry| carry.cell == "atmosphere")
                .expect("second loop carries atmosphere"),
            _ => unreachable!(),
        };
        assert_eq!(
            plan.carry_source(second),
            Some(&StateCarrySource {
                source: StateSource {
                    binding: "atmosphere".to_string(),
                },
                kind: StateCarrySourceKind::PriorState,
            })
        );
    }

    #[test]
    fn competing_final_writers_are_rejected_before_execution() {
        let mut workflow = workflow(include_str!(
            "../../../examples/inference_metadata/catalogue/18-static-cache-indexed-scatter.yaml"
        ));
        let state = workflow
            .state
            .get_mut("cache")
            .expect("fixture declares cache state");
        state.scope = WorkflowStateScope::Session;
        workflow.steps.push(WorkflowStep::Branch {
            predicate: "request.position".to_string(),
            cases: BTreeMap::from([
                (
                    "0".to_string(),
                    WorkflowStep::Invoke {
                        component: "decoder".to_string(),
                        inputs: BTreeMap::new(),
                        outputs: BTreeMap::from([(
                            "present_cache".to_string(),
                            "prefill.cache".to_string(),
                        )]),
                    },
                ),
                (
                    "1".to_string(),
                    WorkflowStep::Invoke {
                        component: "decoder".to_string(),
                        inputs: BTreeMap::new(),
                        outputs: BTreeMap::from([(
                            "present_cache".to_string(),
                            "decode.cache".to_string(),
                        )]),
                    },
                ),
            ]),
            default: None,
            outputs: BTreeMap::new(),
        });

        let errors = validate_state_plan(&workflow, &resolve_state_plan(&workflow));
        assert!(
            errors.iter().any(|error| {
                error.contains("cache")
                    && error.contains("decoder:present_cache:prefill.cache")
                    && error.contains("decoder:present_cache:decode.cache")
                    && error.contains("explicit branch output")
            }),
            "{errors:?}"
        );

        let WorkflowStep::Branch { outputs, .. } =
            workflow.steps.last_mut().expect("test appended a branch")
        else {
            unreachable!()
        };
        outputs.insert(
            "joined.cache".to_string(),
            crate::schema::WorkflowBranchOutput {
                cases: BTreeMap::from([
                    ("0".to_string(), "prefill.cache".to_string()),
                    ("1".to_string(), "decode.cache".to_string()),
                ]),
                default: None,
            },
        );
        let plan = resolve_state_plan(&workflow);
        assert_eq!(
            plan.cell("cache")
                .and_then(|state| state.final_writer.clone()),
            Some(StateFinalWriter::Writer(StateWriter {
                id: "branch:joined.cache".to_string(),
                component: None,
                port: None,
                binding: "joined.cache".to_string(),
            }))
        );
        assert!(
            validate_state_plan(&workflow, &plan)
                .iter()
                .all(|error| !error.contains("ambiguous terminal writers"))
        );
    }

    #[test]
    fn branch_local_writers_with_equal_bindings_still_require_an_output_phi() {
        let mut workflow = workflow(include_str!(
            "../../../examples/inference_metadata/catalogue/18-static-cache-indexed-scatter.yaml"
        ));
        let state = workflow
            .state
            .get_mut("cache")
            .expect("fixture declares cache state");
        state.scope = WorkflowStateScope::Session;
        workflow.steps.push(WorkflowStep::Branch {
            predicate: "request.position".to_string(),
            cases: BTreeMap::from([
                (
                    "0".to_string(),
                    WorkflowStep::Invoke {
                        component: "decoder".to_string(),
                        inputs: BTreeMap::new(),
                        outputs: BTreeMap::from([(
                            "present_cache".to_string(),
                            "branch.cache".to_string(),
                        )]),
                    },
                ),
                (
                    "1".to_string(),
                    WorkflowStep::Invoke {
                        component: "decoder".to_string(),
                        inputs: BTreeMap::new(),
                        outputs: BTreeMap::from([(
                            "present_cache".to_string(),
                            "branch.cache".to_string(),
                        )]),
                    },
                ),
            ]),
            default: None,
            outputs: BTreeMap::new(),
        });

        let errors = validate_state_plan(&workflow, &resolve_state_plan(&workflow));
        assert!(
            errors.iter().any(|error| {
                error.contains("cache")
                    && error.contains("branch.cache")
                    && error.contains("explicit branch output")
            }),
            "{errors:?}"
        );

        let WorkflowStep::Branch { outputs, .. } =
            workflow.steps.last_mut().expect("test appended a branch")
        else {
            unreachable!()
        };
        outputs.insert(
            "joined.cache".to_string(),
            crate::schema::WorkflowBranchOutput {
                cases: BTreeMap::from([
                    ("0".to_string(), "branch.cache".to_string()),
                    ("1".to_string(), "branch.cache".to_string()),
                ]),
                default: None,
            },
        );
        let plan = resolve_state_plan(&workflow);
        assert_eq!(
            plan.cell("cache")
                .and_then(|state| state.final_writer.clone()),
            Some(StateFinalWriter::Writer(StateWriter {
                id: "branch:joined.cache".to_string(),
                component: None,
                port: None,
                binding: "joined.cache".to_string(),
            }))
        );
        assert!(
            validate_state_plan(&workflow, &plan)
                .iter()
                .all(|error| !error.contains("ambiguous terminal writers"))
        );
    }
}
