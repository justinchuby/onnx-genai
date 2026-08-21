//! Cache correctness dependencies derived from the metadata document.
//!
//! Cache identity — the key, the hash function, the salt, and the tenant
//! namespace — is runtime-owned. What is *not* runtime-owned is the set of
//! facts a cache entry depends on: omitting one silently returns another
//! request's tokens. This module derives that set from the workflow SSA graph,
//! component dataflow, and package facts so a producer cannot forget one.

use std::collections::{BTreeMap, BTreeSet};

use crate::schema::{InferenceMetadata, WorkflowNode, WorkflowSpec};

/// Facts a cached result depends on, grouped by origin.
///
/// Every entry is a stable, portable string. The runtime freely constructs a
/// key, hash, salt, or tenant namespace over them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheDependencies {
    /// Implementation identity of every component that can affect the result.
    pub components: BTreeSet<String>,
    /// Adapter artifacts whose activation changes the result.
    pub adapters: BTreeSet<String>,
    /// Preprocessing and encoder results spliced into the cached computation.
    pub inputs: BTreeSet<String>,
    /// Non-dataflow state that native or external components read.
    pub external_state: BTreeSet<String>,
    /// Task profiles that change generated output.
    pub profiles: BTreeSet<String>,
}

impl CacheDependencies {
    /// Every dependency fact in one deterministically ordered list.
    pub fn facts(&self) -> Vec<String> {
        let mut facts = Vec::new();
        facts.extend(
            self.components
                .iter()
                .map(|value| format!("component:{value}")),
        );
        facts.extend(self.adapters.iter().map(|value| format!("adapter:{value}")));
        facts.extend(self.inputs.iter().map(|value| format!("input:{value}")));
        facts.extend(
            self.external_state
                .iter()
                .map(|value| format!("external_state:{value}")),
        );
        facts.extend(self.profiles.iter().map(|value| format!("profile:{value}")));
        facts
    }

    /// Whether this dependency set is empty.
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
            && self.adapters.is_empty()
            && self.inputs.is_empty()
            && self.external_state.is_empty()
            && self.profiles.is_empty()
    }
}

/// Derive the cache correctness dependencies of a metadata document.
///
/// Component dependencies are transitive: a component's identity is included
/// whenever any value it produces reaches an emitted output or a state write,
/// directly or through another component.
pub fn cache_dependencies(metadata: &InferenceMetadata) -> CacheDependencies {
    let mut dependencies = CacheDependencies::default();

    if let Some(pipeline) = &metadata.pipeline {
        let workflow = &pipeline.workflow;
        let graph = match crate::compile_workflow(workflow) {
            Ok(compiled) => compiled.graph,
            // A workflow that does not lower has no derivable dependencies; the
            // semantic validator reports the lowering failure separately.
            Err(_) => return dependencies,
        };
        let mut dataflow = Dataflow::default();
        dataflow.collect(&graph);

        // Seed with the values that leave the workflow: emitted outputs and
        // every state write. Anything that can reach one of them matters.
        let mut live = dataflow.emitted.clone();
        live.extend(dataflow.state_writes.iter().cloned());
        let mut pending = live.iter().cloned().collect::<Vec<_>>();
        while let Some(value) = pending.pop() {
            // A value merged by a branch phi or moved by a transfer keeps every
            // source alive, so the components behind each source stay in scope.
            for source in dataflow.aliases.get(&value).into_iter().flatten() {
                if live.insert(source.clone()) {
                    pending.push(source.clone());
                }
            }
            for component in dataflow.producers.get(&value).into_iter().flatten() {
                if !dependencies.components.insert(component.clone()) {
                    continue;
                }
                for input in dataflow
                    .component_inputs
                    .get(component)
                    .into_iter()
                    .flatten()
                {
                    if live.insert(input.clone()) {
                        pending.push(input.clone());
                    }
                }
            }
        }

        for component in dependencies.components.clone() {
            let Some(declaration) = workflow.components.get(&component) else {
                continue;
            };
            dependencies
                .components
                .insert(component_identity(&component, declaration));
            dependencies.components.remove(&component);
            dependencies
                .external_state
                .extend(declaration.cache_affects_state.iter().cloned());
        }

        // Externally suppliable values — cached encoder results, preprocessed
        // media — change the computation without appearing as a component.
        for (name, input) in &workflow.inputs {
            if input.externally_suppliable && live.contains(name) {
                dependencies.inputs.insert(name.clone());
            }
        }
        for binding in preprocessing_bindings(metadata) {
            if live.contains(&binding) {
                dependencies.inputs.insert(binding);
            }
        }
        collect_live_state_inputs(workflow, &live, &mut dependencies);
    }

    if let Some(adapters) = &metadata.adapters {
        for (alias, artifact) in &adapters.artifacts {
            dependencies.adapters.insert(format!(
                "{alias}@{}:{}",
                artifact.identity, artifact.version
            ));
        }
    }

    for (name, profile) in &metadata.profiles {
        if profile.generation_affecting {
            dependencies
                .profiles
                .insert(format!("{name}@{}", profile.version));
        }
    }

    dependencies
}

/// Stable implementation identity of one component.
fn component_identity(name: &str, component: &crate::schema::WorkflowComponent) -> String {
    match &component.implementation {
        crate::schema::ComponentImplementation::Onnx { artifact } => {
            format!("{name}=onnx:{artifact}")
        }
        crate::schema::ComponentImplementation::Adapter {
            abi,
            version,
            artifact,
        } => match artifact {
            Some(artifact) => format!("{name}=adapter:{abi}@{version}:{artifact}"),
            None => format!("{name}=adapter:{abi}@{version}"),
        },
        crate::schema::ComponentImplementation::Binding => format!("{name}=binding"),
    }
}

/// Preprocessing outputs bound into the workflow's SSA namespace.
fn preprocessing_bindings(metadata: &InferenceMetadata) -> BTreeSet<String> {
    metadata
        .preprocessing
        .iter()
        .filter_map(|preprocessing| preprocessing.image.as_ref())
        .flat_map(|image| image.outputs.iter())
        .map(|binding| binding.name.clone())
        .collect()
}

/// Workflow inputs that initialize a live state cell also change cached results.
fn collect_live_state_inputs(
    workflow: &WorkflowSpec,
    live: &BTreeSet<String>,
    dependencies: &mut CacheDependencies,
) {
    for (name, state) in &workflow.state {
        if !live.contains(name) {
            continue;
        }
        if let Some(input) = workflow.inputs.get(&state.initializer)
            && input.externally_suppliable
        {
            dependencies.inputs.insert(state.initializer.clone());
        }
    }
}

#[derive(Default)]
struct Dataflow {
    /// SSA value name to the components that can produce it.
    producers: BTreeMap<String, BTreeSet<String>>,
    /// SSA value name to the values it forwards unchanged (phi merges, transfers).
    aliases: BTreeMap<String, BTreeSet<String>>,
    /// Component name to every SSA value it consumes.
    component_inputs: BTreeMap<String, BTreeSet<String>>,
    /// SSA values that leave the workflow through an emit.
    emitted: BTreeSet<String>,
    /// SSA values written back into a state cell.
    state_writes: BTreeSet<String>,
}

impl Dataflow {
    fn collect(&mut self, node: &WorkflowNode) {
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    self.collect(node);
                }
            }
            WorkflowNode::Invoke {
                component,
                inputs,
                outputs,
                ..
            } => {
                let consumed = self.component_inputs.entry(component.clone()).or_default();
                consumed.extend(inputs.values().cloned());
                for value in outputs.values() {
                    self.producers
                        .entry(value.clone())
                        .or_default()
                        .insert(component.clone());
                }
            }
            WorkflowNode::Loop {
                setup,
                body,
                carried,
                ..
            } => {
                self.collect(setup);
                self.collect(body);
                for carry in carried {
                    self.state_writes.insert(carry.body_output.clone());
                    self.state_writes.insert(carry.current.clone());
                }
            }
            WorkflowNode::Branch {
                cases,
                default,
                outputs,
                ..
            } => {
                for case in cases.values() {
                    self.collect(case);
                }
                if let Some(default) = default {
                    self.collect(default);
                }
                // A branch phi forwards its case values unchanged, so the
                // merged output aliases every source value.
                for (output, phi) in outputs {
                    self.aliases.entry(output.clone()).or_default().extend(
                        phi.cases
                            .values()
                            .cloned()
                            .chain(phi.default.iter().cloned()),
                    );
                }
            }
            WorkflowNode::Emit { value, .. } => {
                self.emitted.insert(value.clone());
            }
            WorkflowNode::Transfer { input, output, .. } => {
                // A transfer moves a value between placements without changing
                // it, so the destination aliases the source.
                self.aliases
                    .entry(output.clone())
                    .or_default()
                    .insert(input.clone());
            }
            WorkflowNode::ExecutionIsland { .. } => {}
        }
    }
}
