use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, bail};
use onnx_genai_metadata::{ComponentImplementation, WorkflowComponent, WorkflowNode, WorkflowSpec};
use onnx_genai_ort::{Allocator, IoBinding, Session, Value};
use onnx_runtime_loader::proto::onnx::{
    GraphProto, ModelProto, TensorProto, ValueInfoProto, tensor_proto,
};
use prost::Message;

use super::{PipelineEngine, PipelineTensors};
use crate::decode::clone_value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionIslandDiagnostic {
    pub id: usize,
    pub components: Vec<String>,
    pub device: String,
    pub capture_eligible: bool,
    pub linked_node_count: usize,
    pub component_boundaries_elided: usize,
    pub runs: u64,
    pub session_runs: u64,
    pub eager_runs: u64,
    pub stable_binding_runs: u64,
    pub captures: u64,
    pub replays: u64,
    pub device_synchronizations: u64,
    pub host_to_host_copies: u64,
    pub host_to_device_copies: u64,
    pub device_to_host_copies: u64,
    pub device_to_device_copies: u64,
    pub host_to_host_bytes: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub device_to_device_bytes: u64,
    pub stable_binding_bytes: u64,
    pub external_initializer_bytes: u64,
    pub device_memory_total_bytes: Option<u64>,
    pub device_memory_baseline_free_bytes: Option<u64>,
    pub device_memory_min_free_bytes: Option<u64>,
    pub observed_device_memory_high_watermark_bytes: Option<u64>,
    pub total_run_ns: u128,
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct IslandInvocation {
    pub component: String,
    pub inputs: BTreeMap<String, String>,
    pub outputs: BTreeMap<String, String>,
}

pub(crate) struct ExecutionIsland {
    pub id: usize,
    pub components: Vec<String>,
    pub inputs: BTreeMap<String, String>,
    pub outputs: BTreeMap<String, String>,
    session: Session,
    capture_eligible: bool,
    capture_disabled: Cell<bool>,
    device: String,
    bindings: RefCell<HashMap<Vec<(String, Vec<i64>)>, StableIslandBinding>>,
    device_allocator: Option<Allocator>,
    linked_node_count: usize,
    external_initializer_bytes: u64,
    next_graph_id: Cell<i32>,
    runs: Cell<u64>,
    session_runs: Cell<u64>,
    eager_runs: Cell<u64>,
    stable_binding_runs: Cell<u64>,
    captures: Cell<u64>,
    replays: Cell<u64>,
    device_synchronizations: Cell<u64>,
    host_to_host_copies: Cell<u64>,
    host_to_device_copies: Cell<u64>,
    device_to_host_copies: Cell<u64>,
    device_to_device_copies: Cell<u64>,
    host_to_host_bytes: Cell<u64>,
    host_to_device_bytes: Cell<u64>,
    device_to_host_bytes: Cell<u64>,
    device_to_device_bytes: Cell<u64>,
    stable_binding_bytes: Cell<u64>,
    device_memory_total_bytes: Cell<Option<u64>>,
    device_memory_baseline_free_bytes: Cell<Option<u64>>,
    device_memory_min_free_bytes: Cell<Option<u64>>,
    total_run_ns: Cell<u128>,
    fallback_reason: RefCell<Option<String>>,
}

struct StableIslandBinding {
    binding: IoBinding,
    inputs: Vec<(String, Value)>,
    outputs: Vec<(String, Value)>,
    captured: bool,
    graph_id: i32,
}

impl ExecutionIsland {
    pub(crate) fn component_count(&self) -> usize {
        self.components.len()
    }

    fn diagnostic(&self) -> ExecutionIslandDiagnostic {
        ExecutionIslandDiagnostic {
            id: self.id,
            components: self.components.clone(),
            device: self.device.clone(),
            capture_eligible: self.capture_eligible,
            linked_node_count: self.linked_node_count,
            component_boundaries_elided: self.components.len().saturating_sub(1),
            runs: self.runs.get(),
            session_runs: self.session_runs.get(),
            eager_runs: self.eager_runs.get(),
            stable_binding_runs: self.stable_binding_runs.get(),
            captures: self.captures.get(),
            replays: self.replays.get(),
            device_synchronizations: self.device_synchronizations.get(),
            host_to_host_copies: self.host_to_host_copies.get(),
            host_to_device_copies: self.host_to_device_copies.get(),
            device_to_host_copies: self.device_to_host_copies.get(),
            device_to_device_copies: self.device_to_device_copies.get(),
            host_to_host_bytes: self.host_to_host_bytes.get(),
            host_to_device_bytes: self.host_to_device_bytes.get(),
            device_to_host_bytes: self.device_to_host_bytes.get(),
            device_to_device_bytes: self.device_to_device_bytes.get(),
            stable_binding_bytes: self.stable_binding_bytes.get(),
            external_initializer_bytes: self.external_initializer_bytes,
            device_memory_total_bytes: self.device_memory_total_bytes.get(),
            device_memory_baseline_free_bytes: self.device_memory_baseline_free_bytes.get(),
            device_memory_min_free_bytes: self.device_memory_min_free_bytes.get(),
            observed_device_memory_high_watermark_bytes: self
                .device_memory_baseline_free_bytes
                .get()
                .zip(self.device_memory_min_free_bytes.get())
                .map(|(baseline, minimum)| baseline.saturating_sub(minimum)),
            total_run_ns: self.total_run_ns.get(),
            fallback_reason: self.fallback_reason.borrow().clone(),
        }
    }

    pub(crate) fn run(
        &self,
        values: &mut PipelineTensors,
        component_overrides: &HashMap<String, String>,
    ) -> anyhow::Result<()> {
        let started = Instant::now();
        self.record_device_memory();
        if self
            .components
            .iter()
            .any(|component| component_overrides.contains_key(component))
        {
            bail!(
                "execution island {} contains an application-overridden component; \
                 the caller must execute its unfused fallback",
                self.id
            );
        }
        let resolved = self
            .inputs
            .iter()
            .map(|(port, value)| {
                values
                    .get(value)
                    .with_context(|| {
                        format!(
                            "execution island {} input '{port}' references unavailable \
                             workflow value '{value}'",
                            self.id
                        )
                    })
                    .map(|tensor| (port.as_str(), tensor))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let signature = resolved
            .iter()
            .map(|(_, value)| (format!("{:?}", value.dtype()), value.shape().to_vec()))
            .collect::<Vec<_>>();
        self.runs.set(self.runs.get() + 1);

        let mut bindings = self.bindings.borrow_mut();
        if let Some(stable) = bindings.get_mut(&signature) {
            self.stable_binding_runs
                .set(self.stable_binding_runs.get() + 1);
            for ((_, source), (_, destination)) in resolved.iter().zip(&stable.inputs) {
                self.record_copy(source, destination)?;
                if let Some(device_id) = self.session.cuda_device_id() {
                    destination.copy_from_cuda(source, device_id)?;
                } else {
                    destination.copy_from_host(source)?;
                }
            }
            let replay = stable.captured;
            let capture_enabled = self.capture_eligible
                && self.session.graph_capture()
                && !self.capture_disabled.get();
            let result = if capture_enabled {
                self.record_device_synchronization();
                self.session.synchronize_device()?;
                self.session_runs.set(self.session_runs.get() + 1);
                self.session
                    .run_with_binding_graph(&stable.binding, stable.graph_id)
            } else {
                self.session_runs.set(self.session_runs.get() + 1);
                self.session.run_with_binding(&stable.binding)
            };
            if let Err(error) = result {
                *self.fallback_reason.borrow_mut() = Some(format!(
                    "stable island binding/capture failed; executing eagerly: {error}"
                ));
                self.capture_disabled.set(true);
                self.eager_runs.set(self.eager_runs.get() + 1);
                self.session_runs.set(self.session_runs.get() + 1);
                let produced = self.session.run(&resolved)?;
                self.store_outputs(values, produced)?;
                self.record_elapsed(started);
                return Ok(());
            }
            if capture_enabled {
                if replay {
                    self.replays.set(self.replays.get() + 1);
                } else {
                    self.captures.set(self.captures.get() + 1);
                    stable.captured = true;
                }
            }
            for (port, output) in &stable.outputs {
                let value_ref = &self.outputs[port];
                let output = if let Some(device_id) = self.session.cuda_device_id() {
                    let host = Value::empty(output.shape(), output.dtype())?;
                    self.record_copy(output, &host)?;
                    host.copy_from_cuda(output, device_id)?;
                    host
                } else {
                    self.record_host_copy(output);
                    clone_value(output)?
                };
                values.insert(value_ref.clone(), output);
            }
            self.record_elapsed(started);
            return Ok(());
        }

        // Warmup resolves any artifact-inferred dynamic output extents. The next
        // equal-shape run uses stable buffers and becomes capture/replay eligible.
        if self.capture_eligible {
            let allocator = self.device_allocator.as_ref().with_context(|| {
                format!(
                    "execution island {} is capture eligible but has no device allocator",
                    self.id
                )
            })?;
            let device_id = self.session.cuda_device_id().with_context(|| {
                format!(
                    "execution island {} is capture eligible but has no CUDA device",
                    self.id
                )
            })?;
            let mut binding = IoBinding::new(&self.session)?;
            let mut stable_inputs = Vec::new();
            for (name, source) in &resolved {
                let stable = Value::empty_in(source.shape(), source.dtype(), allocator)?;
                self.record_copy(source, &stable)?;
                stable.copy_from_cuda(source, device_id)?;
                binding.bind_input(name, &stable)?;
                stable_inputs.push(((*name).to_string(), stable));
            }
            for name in self.session.output_names() {
                if !self
                    .session
                    .bind_output_to_execution_device(&mut binding, name)?
                {
                    bail!(
                        "execution island {} cannot bind output '{name}' to its CUDA device",
                        self.id
                    );
                }
            }
            self.record_device_synchronization();
            self.session.synchronize_device()?;
            self.eager_runs.set(self.eager_runs.get() + 1);
            self.session_runs.set(self.session_runs.get() + 1);
            self.session.run_with_binding(&binding)?;
            let produced = binding.output_values()?;
            binding.clear()?;
            for (name, input) in &stable_inputs {
                binding.bind_input(name, input)?;
            }
            let mut stable_outputs = Vec::new();
            for (name, output) in self.session.output_names().iter().zip(produced) {
                binding.bind_output(name, &output)?;
                stable_outputs.push((name.clone(), output));
            }
            let graph_id = self.next_graph_id.get();
            self.next_graph_id.set(graph_id + 1);
            for (port, output) in &stable_outputs {
                let value_ref = &self.outputs[port];
                let host = Value::empty(output.shape(), output.dtype())?;
                self.record_copy(output, &host)?;
                host.copy_from_cuda(output, device_id)?;
                values.insert(value_ref.clone(), host);
            }
            self.record_stable_binding_bytes(&stable_inputs, &stable_outputs);
            bindings.insert(
                signature,
                StableIslandBinding {
                    binding,
                    inputs: stable_inputs,
                    outputs: stable_outputs,
                    captured: false,
                    graph_id,
                },
            );
            self.record_elapsed(started);
            return Ok(());
        }

        self.eager_runs.set(self.eager_runs.get() + 1);
        self.session_runs.set(self.session_runs.get() + 1);
        let produced = self.session.run(&resolved)?;
        let mut binding = IoBinding::new(&self.session)?;
        let mut stable_inputs = Vec::new();
        for (name, source) in &resolved {
            let stable =
                Value::from_raw_bytes(source.to_raw_bytes()?, source.shape(), source.dtype())?;
            self.record_host_copy(source);
            binding.bind_input(name, &stable)?;
            stable_inputs.push(((*name).to_string(), stable));
        }
        let mut stable_outputs = Vec::new();
        for (name, output) in self.session.output_names().iter().zip(&produced) {
            let stable = Value::empty(output.shape(), output.dtype())?;
            self.record_host_copy(output);
            binding.bind_output(name, &stable)?;
            stable_outputs.push((name.clone(), stable));
        }
        let graph_id = self.next_graph_id.get();
        self.next_graph_id.set(graph_id + 1);
        self.record_stable_binding_bytes(&stable_inputs, &stable_outputs);
        bindings.insert(
            signature,
            StableIslandBinding {
                binding,
                inputs: stable_inputs,
                outputs: stable_outputs,
                captured: false,
                graph_id,
            },
        );
        self.store_outputs(values, produced)?;
        self.record_elapsed(started);
        Ok(())
    }

    fn record_elapsed(&self, started: Instant) {
        self.record_device_memory();
        self.total_run_ns
            .set(self.total_run_ns.get() + started.elapsed().as_nanos());
    }

    fn record_device_memory(&self) {
        #[cfg(not(any(feature = "cuda", feature = "ort-cuda")))]
        return;

        #[cfg(any(feature = "cuda", feature = "ort-cuda"))]
        {
            let Some(device_id) = self.session.cuda_device_id() else {
                return;
            };
            let Ok(memory) = onnx_genai_ort::cuda_rt::device_memory_info(device_id) else {
                return;
            };
            let free = memory.free_bytes as u64;
            self.device_memory_total_bytes
                .set(Some(memory.total_bytes as u64));
            if self.device_memory_baseline_free_bytes.get().is_none() {
                self.device_memory_baseline_free_bytes.set(Some(free));
            }
            self.device_memory_min_free_bytes.set(Some(
                self.device_memory_min_free_bytes
                    .get()
                    .map_or(free, |minimum| minimum.min(free)),
            ));
        }
    }

    fn record_device_synchronization(&self) {
        self.device_synchronizations
            .set(self.device_synchronizations.get() + 1);
    }

    fn record_host_copy(&self, value: &Value) {
        self.host_to_host_copies
            .set(self.host_to_host_copies.get() + 1);
        self.host_to_host_bytes
            .set(self.host_to_host_bytes.get() + (value.numel() * value.dtype().size_of()) as u64);
    }

    fn record_copy(&self, source: &Value, destination: &Value) -> anyhow::Result<()> {
        let bytes = (source.numel() * source.dtype().size_of()) as u64;
        match (source.is_host_resident()?, destination.is_host_resident()?) {
            (true, true) => {
                self.host_to_host_copies
                    .set(self.host_to_host_copies.get() + 1);
                self.host_to_host_bytes
                    .set(self.host_to_host_bytes.get() + bytes);
            }
            (true, false) => {
                self.host_to_device_copies
                    .set(self.host_to_device_copies.get() + 1);
                self.host_to_device_bytes
                    .set(self.host_to_device_bytes.get() + bytes);
            }
            (false, true) => {
                self.device_to_host_copies
                    .set(self.device_to_host_copies.get() + 1);
                self.device_to_host_bytes
                    .set(self.device_to_host_bytes.get() + bytes);
            }
            (false, false) => {
                self.device_to_device_copies
                    .set(self.device_to_device_copies.get() + 1);
                self.device_to_device_bytes
                    .set(self.device_to_device_bytes.get() + bytes);
            }
        }
        Ok(())
    }

    fn record_stable_binding_bytes(&self, inputs: &[(String, Value)], outputs: &[(String, Value)]) {
        let bytes = inputs
            .iter()
            .chain(outputs)
            .map(|(_, value)| value.numel() * value.dtype().size_of())
            .sum::<usize>() as u64;
        self.stable_binding_bytes
            .set(self.stable_binding_bytes.get() + bytes);
    }

    fn store_outputs(
        &self,
        values: &mut PipelineTensors,
        produced: Vec<Value>,
    ) -> anyhow::Result<()> {
        for (port, tensor) in self.session.output_names().iter().zip(produced) {
            let value_ref = self.outputs.get(port).with_context(|| {
                format!(
                    "execution island {} produced undeclared boundary output '{port}'",
                    self.id
                )
            })?;
            values.insert(value_ref.clone(), tensor);
        }
        Ok(())
    }
}

impl PipelineEngine {
    pub fn execution_island_diagnostics(&self) -> Vec<ExecutionIslandDiagnostic> {
        self.execution_islands
            .iter()
            .map(ExecutionIsland::diagnostic)
            .collect()
    }
}

pub(crate) fn plan_execution_islands(
    graph: &mut WorkflowNode,
    workflow: &WorkflowSpec,
    models: &onnx_genai_ort::PipelineModels,
) -> anyhow::Result<Vec<ExecutionIsland>> {
    let mut uses = HashMap::<String, usize>::new();
    collect_value_uses(graph, &mut uses);
    let mut islands = Vec::new();
    lower_node(graph, workflow, models, &uses, &mut islands)?;
    Ok(islands)
}

fn lower_node(
    node: &mut WorkflowNode,
    workflow: &WorkflowSpec,
    models: &onnx_genai_ort::PipelineModels,
    uses: &HashMap<String, usize>,
    islands: &mut Vec<ExecutionIsland>,
) -> anyhow::Result<()> {
    match node {
        WorkflowNode::Sequence { nodes } => {
            for child in nodes.iter_mut() {
                lower_node(child, workflow, models, uses, islands)?;
            }
            let mut lowered = Vec::new();
            let mut index = 0;
            while index < nodes.len() {
                let Some(device) = pure_onnx_device(&nodes[index], workflow, models) else {
                    lowered.push(nodes[index].clone());
                    index += 1;
                    continue;
                };
                let start = index;
                index += 1;
                while index < nodes.len()
                    && pure_onnx_device(&nodes[index], workflow, models).as_deref()
                        == Some(device.as_str())
                {
                    index += 1;
                }
                if index - start < 2 {
                    lowered.push(nodes[start].clone());
                    continue;
                }
                let invocations = nodes[start..index]
                    .iter()
                    .map(island_invocation)
                    .collect::<anyhow::Result<Vec<_>>>()?;
                match build_execution_island(
                    islands.len(),
                    &device,
                    invocations,
                    workflow,
                    models,
                    uses,
                ) {
                    Ok(island) => {
                        let id = island.id;
                        islands.push(island);
                        lowered.push(WorkflowNode::ExecutionIsland { id });
                    }
                    Err(error) => {
                        tracing::warn!(
                            components = ?nodes[start..index]
                                .iter()
                                .filter_map(|node| match node {
                                    WorkflowNode::Invoke { component, .. } => Some(component),
                                    _ => None,
                                })
                                .collect::<Vec<_>>(),
                            "workflow execution-island fusion declined: {error}"
                        );
                        lowered.extend(nodes[start..index].iter().cloned());
                    }
                }
            }
            *nodes = lowered;
        }
        WorkflowNode::Loop { setup, body, .. } => {
            lower_node(setup, workflow, models, uses, islands)?;
            lower_node(body, workflow, models, uses, islands)?;
        }
        WorkflowNode::Branch { cases, default, .. } => {
            for case in cases.values_mut() {
                lower_node(case, workflow, models, uses, islands)?;
            }
            if let Some(default) = default {
                lower_node(default, workflow, models, uses, islands)?;
            }
        }
        WorkflowNode::Invoke { .. }
        | WorkflowNode::Emit { .. }
        | WorkflowNode::Transfer { .. }
        | WorkflowNode::ExecutionIsland { .. } => {}
    }
    Ok(())
}

fn pure_onnx_device(
    node: &WorkflowNode,
    workflow: &WorkflowSpec,
    models: &onnx_genai_ort::PipelineModels,
) -> Option<String> {
    let WorkflowNode::Invoke {
        component, effects, ..
    } = node
    else {
        return None;
    };
    if !effects.is_empty() {
        return None;
    }
    let declaration = workflow.components.get(component)?;
    if !is_fusible_component(declaration) {
        return None;
    }
    let session = models.session(component)?;
    Some(match session.cuda_device_id() {
        Some(device) => format!("cuda:{device}"),
        None => "cpu".to_string(),
    })
}

fn is_fusible_component(component: &WorkflowComponent) -> bool {
    component.effects.is_empty()
        && !component.application_overridable
        && matches!(
            component.implementation,
            ComponentImplementation::Onnx { .. }
        )
}

fn island_invocation(node: &WorkflowNode) -> anyhow::Result<IslandInvocation> {
    match node {
        WorkflowNode::Invoke {
            component,
            inputs,
            outputs,
            ..
        } => Ok(IslandInvocation {
            component: component.clone(),
            inputs: inputs.clone(),
            outputs: outputs.clone(),
        }),
        _ => bail!("execution island contains a non-invoke node"),
    }
}

fn collect_value_uses(node: &WorkflowNode, uses: &mut HashMap<String, usize>) {
    let mut use_value = |value: &str| *uses.entry(value.to_string()).or_default() += 1;
    match node {
        WorkflowNode::Sequence { nodes } => {
            for node in nodes {
                collect_value_uses(node, uses);
            }
        }
        WorkflowNode::Invoke { inputs, .. } => {
            for value in inputs.values() {
                use_value(value);
            }
        }
        WorkflowNode::Loop {
            setup,
            body,
            continue_when,
            max_iterations,
            carried,
            ..
        } => {
            use_value(continue_when);
            use_value(max_iterations);
            for carry in carried {
                use_value(&carry.current);
                use_value(&carry.body_output);
            }
            collect_value_uses(setup, uses);
            collect_value_uses(body, uses);
        }
        WorkflowNode::Branch {
            predicate,
            cases,
            default,
            outputs,
            ..
        } => {
            use_value(predicate);
            for output in outputs.values() {
                for value in output.cases.values() {
                    use_value(value);
                }
                if let Some(value) = &output.default {
                    use_value(value);
                }
            }
            for case in cases.values() {
                collect_value_uses(case, uses);
            }
            if let Some(default) = default {
                collect_value_uses(default, uses);
            }
        }
        WorkflowNode::Emit {
            value,
            when,
            valid_length,
            ..
        } => {
            use_value(value);
            if let Some(when) = when {
                use_value(when);
            }
            if let Some(valid_length) = valid_length {
                use_value(valid_length);
            }
        }
        WorkflowNode::Transfer { input, .. } => use_value(input),
        WorkflowNode::ExecutionIsland { .. } => {}
    }
}

fn build_execution_island(
    id: usize,
    device: &str,
    invocations: Vec<IslandInvocation>,
    workflow: &WorkflowSpec,
    models: &onnx_genai_ort::PipelineModels,
    uses: &HashMap<String, usize>,
) -> anyhow::Result<ExecutionIsland> {
    let internal_uses = invocations
        .iter()
        .flat_map(|invoke| invoke.inputs.values())
        .fold(HashMap::<String, usize>::new(), |mut uses, value| {
            *uses.entry(value.clone()).or_default() += 1;
            uses
        });
    let produced = invocations
        .iter()
        .flat_map(|invoke| invoke.outputs.values().cloned())
        .collect::<HashSet<_>>();
    let boundary_outputs = produced
        .iter()
        .filter(|value| {
            uses.get(*value).copied().unwrap_or_default()
                > internal_uses.get(*value).copied().unwrap_or_default()
        })
        .cloned()
        .collect::<HashSet<_>>();
    let linked = link_models(id, &invocations, models, &boundary_outputs)?;
    let options = models.session_options();
    let capture_requested = options.graph_capture;
    let structurally_capture_eligible =
        device.starts_with("cuda:") && capture_requested && linked.capture_declines.is_empty();
    let session = Session::from_model_bytes_with_external_files(
        models.environment(),
        format!("workflow-island-{id}"),
        &linked.bytes,
        &linked.external_files,
        options,
    )?;
    let device_allocator = if structurally_capture_eligible {
        session.device_allocator()?
    } else {
        None
    };
    let capture_eligible = structurally_capture_eligible && device_allocator.is_some();
    let external_initializer_bytes = linked
        .external_files
        .iter()
        .map(|(_, bytes)| bytes.len() as u64)
        .sum();
    let components = invocations
        .iter()
        .map(|invoke| invoke.component.clone())
        .collect();
    let _ = workflow;
    Ok(ExecutionIsland {
        id,
        components,
        inputs: linked.inputs,
        outputs: linked.outputs,
        session,
        capture_eligible,
        capture_disabled: Cell::new(false),
        device: device.to_string(),
        bindings: RefCell::new(HashMap::new()),
        device_allocator,
        linked_node_count: linked.node_count,
        external_initializer_bytes,
        next_graph_id: Cell::new((id as i32).saturating_mul(1000)),
        runs: Cell::new(0),
        session_runs: Cell::new(0),
        eager_runs: Cell::new(0),
        stable_binding_runs: Cell::new(0),
        captures: Cell::new(0),
        replays: Cell::new(0),
        device_synchronizations: Cell::new(0),
        host_to_host_copies: Cell::new(0),
        host_to_device_copies: Cell::new(0),
        device_to_host_copies: Cell::new(0),
        device_to_device_copies: Cell::new(0),
        host_to_host_bytes: Cell::new(0),
        host_to_device_bytes: Cell::new(0),
        device_to_host_bytes: Cell::new(0),
        device_to_device_bytes: Cell::new(0),
        stable_binding_bytes: Cell::new(0),
        device_memory_total_bytes: Cell::new(None),
        device_memory_baseline_free_bytes: Cell::new(None),
        device_memory_min_free_bytes: Cell::new(None),
        total_run_ns: Cell::new(0),
        fallback_reason: RefCell::new(if !device.starts_with("cuda:") {
            Some("island is not placed on CUDA".to_string())
        } else if !linked.capture_declines.is_empty() {
            Some(linked.capture_declines.join("; "))
        } else if !capture_requested {
            Some("CUDA graph capture is disabled by session options".to_string())
        } else if !capture_eligible {
            Some("CUDA execution-island device allocator is unavailable".to_string())
        } else {
            None
        }),
    })
}

struct LinkedModel {
    bytes: Vec<u8>,
    inputs: BTreeMap<String, String>,
    outputs: BTreeMap<String, String>,
    capture_declines: Vec<String>,
    external_files: Vec<(String, Vec<u8>)>,
    node_count: usize,
}

fn link_models(
    id: usize,
    invocations: &[IslandInvocation],
    models: &onnx_genai_ort::PipelineModels,
    boundary_outputs: &HashSet<String>,
) -> anyhow::Result<LinkedModel> {
    let mut model = ModelProto {
        ir_version: 8,
        producer_name: "onnx-genai-execution-island".to_string(),
        graph: Some(GraphProto {
            name: "workflow_execution_island".to_string(),
            ..Default::default()
        }),
        ..Default::default()
    };
    let graph = model.graph.as_mut().context("linked graph missing")?;
    let mut opsets = BTreeMap::<String, i64>::new();
    let mut ssa_values = HashMap::<String, String>::new();
    let mut fused_inputs = BTreeMap::new();
    let mut fused_outputs = BTreeMap::new();
    let mut graph_inputs = BTreeMap::<String, ValueInfoProto>::new();
    let mut boundary_input_names = HashMap::<String, String>::new();
    let mut capture_declines = Vec::new();
    let mut external_files = BTreeMap::<String, Vec<u8>>::new();
    let mut external_names = BTreeMap::<std::path::PathBuf, String>::new();

    for (index, invocation) in invocations.iter().enumerate() {
        let path = models
            .directory
            .model_paths
            .get(&invocation.component)
            .with_context(|| format!("component '{}' has no model path", invocation.component))?;
        let bytes = onnx_runtime_loader::read_model_binary(path)?;
        let mut source = ModelProto::decode(bytes.as_slice())?;
        if !source.functions.is_empty() {
            bail!(
                "component '{}' contains model-local functions",
                invocation.component
            );
        }
        model.ir_version = model.ir_version.max(source.ir_version);
        for import in source.opset_import.drain(..) {
            // A node serialized against an older schema is not automatically valid under a newer
            // import (for example ReduceSum moved `axes` from an attribute to an input at opset
            // 13). Linking therefore requires conversion, not merely taking the maximum version.
            if let Some(version) = opsets.get(&import.domain)
                && *version != import.version
            {
                bail!(
                    "component '{}' imports opset domain '{}' at version {}, but the island \
                     already uses version {}; convert artifacts to one opset before fusion",
                    invocation.component,
                    import.domain,
                    import.version,
                    version
                );
            }
            opsets.insert(import.domain, import.version);
        }
        let mut source_graph = source
            .graph
            .take()
            .with_context(|| format!("component '{}' has no graph", invocation.component))?;
        if !source_graph.sparse_initializer.is_empty()
            || source_graph
                .node
                .iter()
                .flat_map(|node| &node.attribute)
                .any(|attribute| attribute.g.is_some() || !attribute.graphs.is_empty())
        {
            bail!(
                "component '{}' uses sparse initializers or nested graph attributes",
                invocation.component
            );
        }
        let prefix = format!("island{id}_c{index}_");
        collect_external_initializers(
            &mut source_graph.initializer,
            path,
            &prefix,
            &mut external_files,
            &mut external_names,
        )?;
        let initializer_names = source_graph
            .initializer
            .iter()
            .map(|initializer| initializer.name.clone())
            .collect::<HashSet<_>>();
        let mut names = HashMap::<String, String>::new();
        for input in &source_graph.input {
            if let Some(value_ref) = invocation.inputs.get(&input.name) {
                let fused_name = if let Some(produced) = ssa_values.get(value_ref) {
                    produced.clone()
                } else {
                    let next_index = boundary_input_names.len();
                    let name = boundary_input_names
                        .entry(value_ref.clone())
                        .or_insert_with(|| format!("in__{next_index}_{}", safe_name(value_ref)))
                        .clone();
                    let mut info = input.clone();
                    info.name = name.clone();
                    graph_inputs.entry(name.clone()).or_insert(info);
                    fused_inputs.insert(name.clone(), value_ref.clone());
                    name
                };
                names.insert(input.name.clone(), fused_name);
            } else if initializer_names.contains(&input.name) {
                names.insert(input.name.clone(), format!("{prefix}{}", input.name));
            } else {
                bail!(
                    "component '{}' required input '{}' is not bound by the workflow invoke",
                    invocation.component,
                    input.name
                );
            }
        }
        for initializer in &source_graph.initializer {
            names
                .entry(initializer.name.clone())
                .or_insert_with(|| format!("{prefix}{}", initializer.name));
        }
        for node in &source_graph.node {
            for output in &node.output {
                if !output.is_empty() {
                    names
                        .entry(output.clone())
                        .or_insert_with(|| format!("{prefix}{output}"));
                }
            }
        }
        for output in &source_graph.output {
            names
                .entry(output.name.clone())
                .or_insert_with(|| format!("{prefix}{}", output.name));
        }
        for initializer in &mut source_graph.initializer {
            initializer.name = names[&initializer.name].clone();
        }
        for value_info in source_graph.value_info.iter_mut() {
            if let Some(name) = names.get(&value_info.name) {
                value_info.name = name.clone();
            }
        }
        for (node_index, node) in source_graph.node.iter_mut().enumerate() {
            if matches!(
                node.op_type.as_str(),
                "NonZero"
                    | "Unique"
                    | "Compress"
                    | "Loop"
                    | "If"
                    | "Scan"
                    | "SequenceConstruct"
                    | "SequenceInsert"
                    | "SequenceErase"
            ) {
                capture_declines.push(format!(
                    "{} contains host/data-dependent allocation or control op {}",
                    invocation.component, node.op_type
                ));
            }
            if matches!(
                node.op_type.as_str(),
                "RandomNormal"
                    | "RandomNormalLike"
                    | "RandomUniform"
                    | "RandomUniformLike"
                    | "Multinomial"
            ) {
                capture_declines.push(format!(
                    "{} uses implicit ONNX RNG op {}; use explicit counter RNG tensor state",
                    invocation.component, node.op_type
                ));
            }
            node.name = format!("{prefix}n{node_index}_{}", node.name);
            for input in &mut node.input {
                if !input.is_empty() {
                    *input = names
                        .get(input)
                        .cloned()
                        .unwrap_or_else(|| format!("{prefix}{input}"));
                }
            }
            for output in &mut node.output {
                if !output.is_empty() {
                    *output = names[output].clone();
                }
            }
        }
        for (port, value_ref) in &invocation.outputs {
            let internal = names.get(port).with_context(|| {
                format!(
                    "component '{}' workflow output port '{port}' is not a graph output",
                    invocation.component
                )
            })?;
            ssa_values.insert(value_ref.clone(), internal.clone());
            if boundary_outputs.contains(value_ref) {
                let mut info = source_graph
                    .output
                    .iter()
                    .find(|output| output.name == *port)
                    .cloned()
                    .with_context(|| {
                        format!(
                            "component '{}' has no output metadata for '{port}'",
                            invocation.component
                        )
                    })?;
                // Component-local values are already uniquely prefixed. Export the producer
                // value directly so a logical component boundary adds no executable ONNX node.
                info.name = internal.clone();
                graph.output.push(info);
                fused_outputs.insert(internal.clone(), value_ref.clone());
            }
        }
        graph.node.extend(source_graph.node);
        graph.initializer.extend(source_graph.initializer);
        graph.value_info.extend(source_graph.value_info);
    }
    graph.input.extend(graph_inputs.into_values());
    model.opset_import = opsets
        .into_iter()
        .map(
            |(domain, version)| onnx_runtime_loader::proto::onnx::OperatorSetIdProto {
                domain,
                version,
            },
        )
        .collect();
    if fused_outputs.is_empty() {
        bail!("execution island has no externally used outputs");
    }
    let node_count = graph.node.len();
    Ok(LinkedModel {
        bytes: model.encode_to_vec(),
        inputs: fused_inputs,
        outputs: fused_outputs,
        capture_declines,
        external_files: external_files.into_iter().collect(),
        node_count,
    })
}

fn collect_external_initializers(
    initializers: &mut [TensorProto],
    model_path: &Path,
    prefix: &str,
    external_files: &mut BTreeMap<String, Vec<u8>>,
    external_names: &mut BTreeMap<std::path::PathBuf, String>,
) -> anyhow::Result<()> {
    for (initializer_index, initializer) in initializers.iter_mut().enumerate() {
        if initializer.data_location != tensor_proto::DataLocation::External as i32 {
            continue;
        }
        let entries = initializer
            .external_data
            .iter()
            .map(|entry| (entry.key.as_str(), entry.value.as_str()))
            .collect::<HashMap<_, _>>();
        let location = entries.get("location").with_context(|| {
            format!(
                "external initializer '{}' has no location",
                initializer.name
            )
        })?;
        let path = model_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(location);
        let path = path
            .canonicalize()
            .with_context(|| format!("failed to resolve external weights {}", path.display()))?;
        let virtual_name = if let Some(name) = external_names.get(&path) {
            name.clone()
        } else {
            let name = format!("{prefix}external_{initializer_index}");
            external_files.insert(
                name.clone(),
                std::fs::read(&path).with_context(|| {
                    format!("failed to read external weights {}", path.display())
                })?,
            );
            external_names.insert(path, name.clone());
            name
        };
        initializer
            .external_data
            .iter_mut()
            .find(|entry| entry.key == "location")
            .expect("external initializer location was checked above")
            .value = virtual_name;
    }
    Ok(())
}

fn safe_name(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_metadata::ComponentPorts;

    fn component(implementation: ComponentImplementation) -> WorkflowComponent {
        WorkflowComponent {
            implementation,
            ports: ComponentPorts::default(),
            contract: None,
            application_overridable: false,
            effects: Vec::new(),
        }
    }

    #[test]
    fn only_pure_fixed_onnx_components_are_fusible() {
        let mut onnx = component(ComponentImplementation::Onnx {
            artifact: "policy.onnx".into(),
        });
        assert!(is_fusible_component(&onnx));

        onnx.effects.push("stream".into());
        assert!(!is_fusible_component(&onnx));
        onnx.effects.clear();
        onnx.application_overridable = true;
        assert!(!is_fusible_component(&onnx));

        let adapter = component(ComponentImplementation::Adapter {
            abi: "onnx-genai.grammar-guidance".into(),
            version: "1".into(),
            artifact: None,
            custom_ops: BTreeMap::new(),
        });
        assert!(!is_fusible_component(&adapter));
    }
}
