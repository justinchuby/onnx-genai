//! The resolved cross-invocation state contract.
//!
//! Serialized state declarations describe graph bindings.  This module reads
//! those bindings once and produces the plan consumed by validation and the
//! runtime.  In particular, storage-management declarations never participate
//! in selecting an initializer or final writer.

use std::collections::{BTreeMap, BTreeSet};

use crate::schema::{
    StateReleaseBoundary, StateUpdate, WorkflowSpec, WorkflowStateClass, WorkflowStateScope,
    WorkflowStep,
};

/// Stable semantic identity of one mutable workflow location.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StateIdentity(pub String);

/// A graph value that seeds a state cell before its first write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSource {
    pub binding: String,
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
    Continuation { output: String },
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
}

/// Resolve state source, readers, writers, and final writers from explicit
/// workflow bindings.  This function deliberately does not read
/// `StateManagement`: allocation policy cannot change dataflow.
pub fn resolve_state_plan(workflow: &WorkflowSpec) -> ResolvedStatePlan {
    let loop_carries = loop_carries(&workflow.steps);
    let groups = workflow
        .serving
        .as_ref()
        .map(|serving| &serving.state_service.groups);
    let mut cells = BTreeMap::new();

    for (name, state) in &workflow.state {
        let mut readers = Vec::new();
        let mut writers = Vec::new();
        for carry in loop_carries.get(name).into_iter().flatten() {
            readers.push(StateReader::LoopCarry {
                binding: carry.initial.clone(),
            });
            writers.push(StateWriter {
                id: format!("loop:{}", carry.next),
                component: None,
                port: None,
                binding: carry.next.clone(),
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

        let final_writer = state
            .session
            .as_ref()
            .and_then(|lease| lease.continuation.as_ref())
            .map(|continuation| match continuation {
                crate::schema::SessionContinuation::PromptPrefix { tokens_output, .. } => {
                    StateFinalWriter::Continuation {
                        output: tokens_output.clone(),
                    }
                }
            })
            .or_else(|| {
                writers
                    .iter()
                    .find(|writer| writer.component.is_none())
                    .cloned()
                    .map(StateFinalWriter::Writer)
            })
            .or_else(|| {
                let component_writers = writers
                    .iter()
                    .filter(|writer| writer.component.is_some())
                    .collect::<Vec<_>>();
                (component_writers.len() == 1)
                    .then(|| StateFinalWriter::Writer(component_writers[0].clone()))
            });

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
    ResolvedStatePlan { cells }
}

/// Validate the canonical plan.  Diagnostics name the affected state and
/// binding, and describe the missing declaration needed to repair it.
pub fn validate_state_plan(workflow: &WorkflowSpec, plan: &ResolvedStatePlan) -> Vec<String> {
    let mut errors = Vec::new();
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
        let component_writers = cell
            .writers
            .iter()
            .filter(|writer| writer.component.is_some())
            .count();
        if cell.final_writer.is_none() && component_writers > 1 {
            errors.push(format!(
                "pipeline.workflow.state.{name} has {component_writers} component writers and \
                 no final-writer join; route the selected successor through one loop carry or \
                 one explicit join binding before commit"
            ));
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

#[derive(Debug, Clone)]
struct LoopCarryBinding {
    initial: String,
    next: String,
}

fn loop_carries(steps: &[WorkflowStep]) -> BTreeMap<String, Vec<LoopCarryBinding>> {
    fn walk(steps: &[WorkflowStep], carries: &mut BTreeMap<String, Vec<LoopCarryBinding>>) {
        for step in steps {
            match step {
                WorkflowStep::Sequence { steps } => walk(steps, carries),
                WorkflowStep::Loop {
                    setup,
                    steps,
                    carried,
                    ..
                } => {
                    for carry in carried {
                        carries
                            .entry(carry.cell.clone())
                            .or_default()
                            .push(LoopCarryBinding {
                                initial: carry
                                    .initial
                                    .clone()
                                    .unwrap_or_else(|| carry.cell.clone()),
                                next: carry.next.clone(),
                            });
                    }
                    walk(setup, carries);
                    walk(steps, carries);
                }
                WorkflowStep::Branch { cases, default, .. } => {
                    for case in cases.values() {
                        walk(std::slice::from_ref(case), carries);
                    }
                    if let Some(default) = default {
                        walk(std::slice::from_ref(default.as_ref()), carries);
                    }
                }
                WorkflowStep::Invoke { .. } | WorkflowStep::Emit { .. } => {}
            }
        }
    }
    let mut carries = BTreeMap::new();
    walk(steps, &mut carries);
    carries
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
        workflow.steps.push(WorkflowStep::Invoke {
            component: "decoder".to_string(),
            inputs: BTreeMap::new(),
            outputs: BTreeMap::from([("present_cache".to_string(), "other.cache".to_string())]),
        });

        let errors = validate_state_plan(&workflow, &resolve_state_plan(&workflow));
        assert!(
            errors.iter().any(|error| {
                error.contains("cache")
                    && error.contains("component writers")
                    && error.contains("final-writer join")
            }),
            "{errors:?}"
        );
    }
}
