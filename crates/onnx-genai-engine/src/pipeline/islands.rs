use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, bail};
use onnx_genai_metadata::{
    ComponentImplementation, KvStorageMode, WorkflowComponent, WorkflowNode, WorkflowSpec,
};
use onnx_genai_ort::{Allocator, IoBinding, Session, Value};
use onnx_runtime_loader::proto::onnx::{
    GraphProto, ModelProto, TensorProto, TensorShapeProto, TypeProto, ValueInfoProto, tensor_proto,
    tensor_shape_proto, type_proto,
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

type IslandBindingKey = Vec<(String, Vec<i64>)>;

pub(crate) struct ExecutionIsland {
    pub id: usize,
    pub components: Vec<String>,
    pub inputs: BTreeMap<String, String>,
    pub outputs: BTreeMap<String, String>,
    aliasable_output_values: HashSet<String>,
    shared_buffer_inputs: HashMap<String, String>,
    immutable_inputs: HashSet<String>,
    fallback: WorkflowNode,
    session: Session,
    capture_eligible: bool,
    capture_disabled: Cell<bool>,
    device: String,
    bindings: RefCell<HashMap<IslandBindingKey, StableIslandBinding>>,
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
    execution_generation: Cell<u64>,
}

struct StableIslandBinding {
    binding: IoBinding,
    inputs: Vec<(String, Value)>,
    outputs: Vec<(String, Value)>,
    captured: bool,
    graph_id: i32,
    service_generation: u64,
    source_ptrs: Vec<usize>,
}

impl ExecutionIsland {
    pub(crate) fn begin_execution(&self, generation: u64) {
        self.execution_generation.set(generation);
    }

    pub(crate) fn clear_bindings(&mut self) {
        self.bindings.get_mut().clear();
    }

    pub(crate) fn component_count(&self) -> usize {
        self.components.len()
    }

    pub(crate) fn components(&self) -> &[String] {
        &self.components
    }

    pub(crate) fn cuda_device_id(&self) -> Option<i32> {
        self.session.cuda_device_id()
    }

    pub(crate) fn materialize_host(&self, value: &Value) -> anyhow::Result<Value> {
        if value.is_host_resident()? {
            return clone_value(value);
        }

        let device_id = self.session.cuda_device_id().with_context(|| {
            format!(
                "execution island {} cannot materialize a non-host value without a CUDA device",
                self.id
            )
        })?;
        let host = Value::empty(value.shape(), value.dtype())?;
        self.record_copy(value, &host)?;
        host.copy_from_cuda(value, device_id)?;
        Ok(host)
    }

    pub(crate) fn uses_override(&self, overrides: &HashMap<String, String>) -> bool {
        self.components
            .iter()
            .any(|component| overrides.contains_key(component))
    }

    pub(crate) fn fallback(&self) -> &WorkflowNode {
        &self.fallback
    }

    fn clone_output_for_store(&self, value_ref: &str, output: &Value) -> anyhow::Result<Value> {
        // Stable binding outputs own their allocation for the lifetime of the
        // island. Workflow SSA values only need a view until the next island
        // invocation, so retain that allocation instead of allocating and
        // copying every boundary tensor.
        if self.session.cuda_device_id().is_some()
            && self.aliasable_output_values.contains(value_ref)
            && let Some(aliased) = output.try_alias_clone()
        {
            return aliased.map_err(|error| {
                anyhow::anyhow!(
                    "execution island {} could not alias a stable output: {error}",
                    self.id
                )
            });
        }
        let Some(device_id) = self.session.cuda_device_id() else {
            self.record_host_copy(output);
            return Value::from_raw_bytes(output.to_raw_bytes()?, output.shape(), output.dtype())
                .map_err(Into::into);
        };
        let allocator = self.device_allocator.as_ref().with_context(|| {
            format!(
                "execution island {} has a CUDA output but no device allocator",
                self.id
            )
        })?;
        let stored = Value::empty_in(output.shape(), output.dtype(), allocator)?;
        self.record_copy(output, &stored)?;
        stored.copy_from_cuda(output, device_id)?;
        Ok(stored)
    }

    fn clone_output_for_store_async(
        &self,
        value_ref: &str,
        output: &Value,
    ) -> anyhow::Result<(Value, bool)> {
        if self.session.cuda_device_id().is_none()
            || self.aliasable_output_values.contains(value_ref)
        {
            return Ok((self.clone_output_for_store(value_ref, output)?, false));
        }
        let device_id = self
            .session
            .cuda_device_id()
            .context("CUDA island output has no CUDA device")?;
        let allocator = self.device_allocator.as_ref().with_context(|| {
            format!(
                "execution island {} has a CUDA output but no device allocator",
                self.id
            )
        })?;
        let stored = Value::empty_in(output.shape(), output.dtype(), allocator)?;
        self.record_copy(output, &stored)?;
        stored.copy_from_cuda_async(output, device_id)?;
        Ok((stored, true))
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
        self.record_device_memory_if_due();
        if self.uses_override(component_overrides) {
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
        let source_ptrs = resolved
            .iter()
            .map(|(name, value)| {
                if value.numel() == 0 {
                    return Ok(0);
                }
                value.data_ptr_addr().with_context(|| {
                    format!(
                        "execution island {} input '{name}' has no tensor data",
                        self.id
                    )
                })
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
            let reset_services = stable.service_generation != self.execution_generation.get();
            for (index, ((name, source), (_, destination))) in
                resolved.iter().zip(&stable.inputs).enumerate()
            {
                let source_ptr = source_ptrs[index];
                if source.numel() == 0 {
                    continue;
                }
                let shared_service = self.shared_buffer_inputs.contains_key(*name);
                let aliases_destination = source_ptr
                    == destination.data_ptr_addr().with_context(|| {
                        format!(
                            "execution island {} stable input '{name}' lost its tensor data",
                            self.id
                        )
                    })?;
                let unchanged_immutable = !reset_services
                    && self.immutable_inputs.contains(*name)
                    && source_ptr == stable.source_ptrs[index];
                if (shared_service && !reset_services)
                    || (!shared_service && (aliases_destination || unchanged_immutable))
                {
                    continue;
                }
                self.record_copy(source, destination).with_context(|| {
                    format!(
                        "execution island {} could not classify input '{name}'",
                        self.id
                    )
                })?;
                if let Some(device_id) = self.session.cuda_device_id() {
                    destination
                        .copy_from_cuda(source, device_id)
                        .with_context(|| {
                            format!(
                                "execution island {} could not refresh input '{name}'",
                                self.id
                            )
                        })?;
                } else {
                    destination.copy_from_host(source)?;
                }
                stable.source_ptrs[index] = source_ptr;
            }
            stable.service_generation = self.execution_generation.get();
            let replay = stable.captured;
            let capture_enabled = self.capture_eligible
                && self.session.graph_capture()
                && !self.capture_disabled.get();
            let mut result = if capture_enabled {
                if !replay {
                    self.record_device_synchronization();
                    self.session.synchronize_device()?;
                }
                self.session_runs.set(self.session_runs.get() + 1);
                self.session
                    .run_with_binding_graph(&stable.binding, stable.graph_id)
            } else {
                self.session_runs.set(self.session_runs.get() + 1);
                self.session.run_with_binding(&stable.binding)
            };
            if capture_enabled && let Err(capture_error) = result {
                *self.fallback_reason.borrow_mut() = Some(format!(
                    "island graph capture/replay failed; continuing with stable binding: \
                     {capture_error}"
                ));
                self.capture_disabled.set(true);
                self.session_runs.set(self.session_runs.get() + 1);
                result = self.session.run_with_binding(&stable.binding);
            }
            if let Err(error) = result {
                *self.fallback_reason.borrow_mut() = Some(format!(
                    "stable island binding failed; executing eagerly: {error}"
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
            let mut pending_copies = false;
            for (port, output) in &stable.outputs {
                let value_ref = &self.outputs[port];
                let (stored, pending) = self
                    .clone_output_for_store_async(value_ref, output)
                    .with_context(|| {
                        format!(
                            "execution island {} could not retain output '{port}'",
                            self.id
                        )
                    })?;
                pending_copies |= pending;
                values.insert(value_ref.clone(), stored);
            }
            if pending_copies {
                self.record_device_synchronization();
                self.session.synchronize_device()?;
            }
            self.record_elapsed(started);
            return Ok(());
        }

        // Warmup resolves any artifact-inferred dynamic output extents. The next
        // equal-shape run uses stable buffers and becomes capture/replay eligible.
        if self.device_allocator.is_some() {
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
                let stable = if source.numel() == 0 {
                    Value::empty_in(source.shape(), source.dtype(), allocator)?
                } else if !source.is_host_resident()?
                    && source.device_id()? == device_id
                    && let Some(alias) = source.try_alias_clone()
                {
                    alias?
                } else {
                    let stable = Value::empty_in(source.shape(), source.dtype(), allocator)?;
                    self.record_copy(source, &stable)?;
                    if source.numel() != 0 {
                        stable.copy_from_cuda(source, device_id)?;
                    }
                    stable
                };
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
            if self.session.graph_capture() {
                // Populate ORT's non-arena constants and discover output extents
                // without consuming a graph id. The stable binding is captured
                // only on its next run, after every OrtValue address is fixed.
                self.session.run_with_binding_graph(&binding, -1)?;
            } else {
                self.session.run_with_binding(&binding)?;
            }
            let produced = binding.output_values().with_context(|| {
                format!(
                    "execution island {} could not extract warmup outputs",
                    self.id
                )
            })?;
            binding.clear()?;
            for (name, input) in &stable_inputs {
                binding.bind_input(name, input)?;
            }
            let mut stable_outputs = Vec::new();
            for (name, output) in self.session.output_names().iter().zip(produced) {
                if self.session.supports_fixed_capacity_present_binding()
                    && let Some(input_name) = self
                        .shared_buffer_inputs
                        .iter()
                        .find_map(|(input, shared)| (shared == name).then_some(input))
                {
                    let input = stable_inputs
                        .iter()
                        .find_map(|(stable_name, value)| {
                            (stable_name == input_name).then_some(value)
                        })
                        .with_context(|| {
                            format!(
                                "execution island {} shared output '{name}' has no stable input \
                                 '{input_name}'",
                                self.id
                            )
                        })?;
                    // Alias only after warmup proves that present preserves the
                    // full physical KV capacity. Dynamic Concat outputs remain
                    // distinct even when metadata requests shared storage.
                    if input.shape() == output.shape() && input.dtype() == output.dtype() {
                        binding.bind_output(name, input)?;
                        continue;
                    }
                }
                let shape = output.shape().to_vec();
                let output = Value::into_alias_with_shape(output, &shape).with_context(|| {
                    format!(
                        "execution island {} output '{name}' could not retain its warmup buffer",
                        self.id
                    )
                })?;
                binding.bind_output(name, &output)?;
                stable_outputs.push((name.clone(), output));
            }
            // The discovery run used distinct present buffers so it could prove
            // their concrete extents without risking a too-small in-place bind.
            // Re-run the original inputs with the now-proven aliases: shared-KV
            // kernels may select a different path when past and present share an
            // address, so discovery output is not a semantic substitute.
            self.session_runs.set(self.session_runs.get() + 1);
            if self.session.graph_capture() {
                self.session.run_with_binding_graph(&binding, -1)?;
            } else {
                self.session.run_with_binding(&binding)?;
            }
            let graph_id = self.next_graph_id.get();
            self.next_graph_id.set(graph_id + 1);
            let mut pending_copies = false;
            for (port, output) in &stable_outputs {
                let value_ref = &self.outputs[port];
                let (stored, pending) = self.clone_output_for_store_async(value_ref, output)?;
                pending_copies |= pending;
                values.insert(value_ref.clone(), stored);
            }
            if pending_copies {
                self.record_device_synchronization();
                self.session.synchronize_device()?;
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
                    service_generation: self.execution_generation.get(),
                    source_ptrs,
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
            let stable = Value::into_alias_with_shape(
                Value::empty(output.shape(), output.dtype())?,
                output.shape(),
            )?;
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
                service_generation: self.execution_generation.get(),
                source_ptrs,
            },
        );
        self.store_outputs(values, produced)?;
        self.record_elapsed(started);
        Ok(())
    }

    fn record_elapsed(&self, started: Instant) {
        self.record_device_memory_if_due();
        self.total_run_ns
            .set(self.total_run_ns.get() + started.elapsed().as_nanos());
    }

    fn record_device_memory_if_due(&self) {
        let runs = self.runs.get();
        if runs == 0 || runs.is_multiple_of(128) {
            self.record_device_memory();
        }
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
                self.record_device_synchronization();
                self.host_to_device_copies
                    .set(self.host_to_device_copies.get() + 1);
                self.host_to_device_bytes
                    .set(self.host_to_device_bytes.get() + bytes);
            }
            (false, true) => {
                self.record_device_synchronization();
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

impl Drop for ExecutionIsland {
    fn drop(&mut self) {
        self.bindings.get_mut().clear();
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
    aliasable_output_values: &HashSet<String>,
) -> anyhow::Result<Vec<ExecutionIsland>> {
    let mut uses = HashMap::<String, usize>::new();
    collect_value_uses(graph, &mut uses);
    let mut islands = Vec::new();
    lower_node(
        graph,
        workflow,
        models,
        &uses,
        aliasable_output_values,
        &mut islands,
    )?;
    Ok(islands)
}

fn lower_node(
    node: &mut WorkflowNode,
    workflow: &WorkflowSpec,
    models: &onnx_genai_ort::PipelineModels,
    uses: &HashMap<String, usize>,
    aliasable_output_values: &HashSet<String>,
    islands: &mut Vec<ExecutionIsland>,
) -> anyhow::Result<()> {
    match node {
        WorkflowNode::Sequence { nodes } => {
            for child in nodes.iter_mut() {
                lower_node(
                    child,
                    workflow,
                    models,
                    uses,
                    aliasable_output_values,
                    islands,
                )?;
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
                    aliasable_output_values,
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
            lower_node(
                setup,
                workflow,
                models,
                uses,
                aliasable_output_values,
                islands,
            )?;
            lower_node(
                body,
                workflow,
                models,
                uses,
                aliasable_output_values,
                islands,
            )?;
        }
        WorkflowNode::Branch { cases, default, .. } => {
            for case in cases.values_mut() {
                lower_node(
                    case,
                    workflow,
                    models,
                    uses,
                    aliasable_output_values,
                    islands,
                )?;
            }
            if let Some(default) = default {
                lower_node(
                    default,
                    workflow,
                    models,
                    uses,
                    aliasable_output_values,
                    islands,
                )?;
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
    let session = models.session(component)?;
    let mut resolved = declaration.clone();
    if resolved.ports.inputs.is_empty() && resolved.ports.outputs.is_empty() {
        resolved.ports.inputs = session
            .inputs()
            .iter()
            .map(|tensor| (tensor.name.clone(), session_tensor_contract(tensor)))
            .collect();
        resolved.ports.outputs = session
            .outputs()
            .iter()
            .map(|tensor| (tensor.name.clone(), session_tensor_contract(tensor)))
            .collect();
    }
    if !is_fusible_component(&resolved) {
        return None;
    }
    Some(match session.cuda_device_id() {
        Some(device) => format!("cuda:{device}"),
        None => "cpu".to_string(),
    })
}

fn session_tensor_contract(
    tensor: &onnx_genai_ort::TensorInfo,
) -> onnx_genai_metadata::TensorContract {
    use onnx_genai_ort::DataType;
    let dtype = match tensor.dtype {
        DataType::Float32 => "float32",
        DataType::Float16 => "float16",
        DataType::BFloat16 => "bfloat16",
        DataType::Float8E4M3 => "float8e4m3",
        DataType::Float8E5M2 => "float8e5m2",
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::Uint8 => "uint8",
        DataType::Uint16 => "uint16",
        DataType::Uint32 => "uint32",
        DataType::Uint64 => "uint64",
        DataType::Bool => "bool",
    };
    let shape = tensor
        .shape
        .iter()
        .enumerate()
        .map(|(axis, dimension)| {
            if *dimension < 0 {
                onnx_genai_metadata::TensorDimension::Symbol(if axis == 0 {
                    "batch".into()
                } else {
                    format!("axis_{axis}")
                })
            } else {
                onnx_genai_metadata::TensorDimension::Fixed(*dimension)
            }
        })
        .collect();
    onnx_genai_metadata::TensorContract {
        dtype: dtype.into(),
        rank: tensor.shape.len(),
        shape: Some(shape),
        optional: false,
    }
}

fn is_fusible_component(component: &WorkflowComponent) -> bool {
    component.effects.is_empty()
        && matches!(
            component.implementation,
            ComponentImplementation::Onnx { .. }
        )
        && batching_safe_policy_contract(component)
}

fn batching_safe_policy_contract(component: &WorkflowComponent) -> bool {
    let Some(contract) = &component.contract else {
        return true;
    };
    let (input_roles, output_roles): (&[&str], &[&str]) = match contract.id.as_str() {
        "onnx-genai.token-sampler" => (
            &[
                "logits",
                "temperature",
                "top_k",
                "top_p",
                "min_p",
                "active",
                "done",
                "seed",
                "counter",
            ],
            &["token", "next_counter"],
        ),
        "onnx-genai.termination-predicate" => (
            &[
                "tokens",
                "active",
                "eos_ids",
                "eos_lengths",
                "iteration",
                "max_iterations",
            ],
            &["done", "next_active", "continue"],
        ),
        "onnx-genai.state-update" => (&["current", "update", "active", "done"], &["next"]),
        _ => return true,
    };
    let mut bound_ports = HashSet::new();
    contract.version == "2"
        && contract.bindings.len() == input_roles.len() + output_roles.len()
        && input_roles
            .iter()
            .all(|role| batching_role_port(component, contract, role, true, &mut bound_ports))
        && output_roles
            .iter()
            .all(|role| batching_role_port(component, contract, role, false, &mut bound_ports))
        && matches!(
            contract.parameters.get("batching"),
            Some(onnx_genai_metadata::ScalarValue::String(value)) if value == "per_row"
        )
        && matches!(
            contract.parameters.get("inactive_rows"),
            Some(onnx_genai_metadata::ScalarValue::String(value)) if value == "preserve"
        )
        && state_update_contracts_match(component, contract)
}

fn state_update_contracts_match(
    component: &WorkflowComponent,
    contract: &onnx_genai_metadata::ComponentContract,
) -> bool {
    if contract.id != "onnx-genai.state-update" {
        return true;
    }
    let tensor = |role: &str, input: bool| {
        contract.bindings.get(role).and_then(|port| {
            if input {
                component.ports.inputs.get(port)
            } else {
                component.ports.outputs.get(port)
            }
        })
    };
    let (Some(current), Some(update), Some(next)) = (
        tensor("current", true),
        tensor("update", true),
        tensor("next", false),
    ) else {
        return false;
    };
    current.dtype == "int64"
        && update.dtype == "int64"
        && next.dtype == "int64"
        && current.rank == 2
        && update.rank == 2
        && next.rank == 2
        && current.shape == update.shape
        && current.shape == next.shape
}

fn batching_role_port(
    component: &WorkflowComponent,
    contract: &onnx_genai_metadata::ComponentContract,
    role: &str,
    input: bool,
    bound_ports: &mut HashSet<String>,
) -> bool {
    let Some(port) = contract.bindings.get(role) else {
        return false;
    };
    if !bound_ports.insert(port.clone()) {
        return false;
    }
    let tensor = if input {
        component.ports.inputs.get(port)
    } else {
        component.ports.outputs.get(port)
    };
    let Some(tensor) = tensor else {
        return false;
    };
    let integer = tensor.dtype.starts_with("int") || tensor.dtype.starts_with("uint");
    let first = tensor.shape.as_deref().and_then(|shape| shape.first());
    if role == "continue" {
        return tensor.dtype == "bool"
            && tensor.rank == 1
            && matches!(first, Some(onnx_genai_metadata::TensorDimension::Fixed(1)));
    }
    if role == "iteration" {
        return integer
            && tensor.rank == 1
            && matches!(first, Some(onnx_genai_metadata::TensorDimension::Fixed(1)));
    }
    if role == "token" && first.is_none() {
        return integer && tensor.rank == 1;
    }
    if tensor.rank == 0
        || !matches!(
            first,
            Some(onnx_genai_metadata::TensorDimension::Symbol(symbol))
                if symbol == "batch" || symbol.ends_with(".batch")
        )
    {
        return false;
    }
    match role {
        "active" | "done" | "next_active" => tensor.dtype == "bool" && tensor.rank == 1,
        "logits" => {
            matches!(tensor.dtype.as_str(), "float32" | "float16" | "bfloat16")
                && matches!(tensor.rank, 2 | 3)
        }
        "temperature" | "top_p" | "min_p" => {
            matches!(tensor.dtype.as_str(), "float32" | "float16" | "bfloat16") && tensor.rank == 1
        }
        "eos_ids" => {
            (tensor.dtype.starts_with("int") || tensor.dtype.starts_with("uint"))
                && tensor.rank == 2
        }
        "top_k" | "seed" | "counter" | "next_counter" | "tokens" | "token" | "eos_lengths"
        | "max_iterations" => integer && tensor.rank == 1,
        "current" | "update" | "next" => true,
        _ => false,
    }
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
            row_ids,
            ..
        } => {
            use_value(value);
            if let Some(when) = when {
                use_value(when);
            }
            if let Some(valid_length) = valid_length {
                use_value(valid_length);
            }
            if let Some(row_ids) = row_ids {
                use_value(row_ids);
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
    aliasable_output_values: &HashSet<String>,
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
    let mut options = models.session_options();
    let capture_requested = options.graph_capture;
    if !linked.external_files.is_empty() && !(device.starts_with("cuda:") && capture_requested) {
        bail!("file-backed external-data fusion requires ORT-managed CUDA graph capture");
    }
    let mut shared_buffer_inputs = HashMap::new();
    if let Some(serving) = &workflow.serving {
        for group in serving
            .kv_service
            .groups
            .values()
            .filter(|group| group.storage == KvStorageMode::SharedBuffer)
        {
            for invocation in &invocations {
                let Some(aliases) = group.ports.get(&invocation.component) else {
                    continue;
                };
                for alias in aliases.values() {
                    let Some(input_value) = invocation.inputs.get(&alias.input) else {
                        continue;
                    };
                    let Some(output_value) = invocation.outputs.get(&alias.output) else {
                        continue;
                    };
                    let input_name = linked
                        .inputs
                        .iter()
                        .find_map(|(name, value)| (value == input_value).then_some(name.clone()))
                        .with_context(|| {
                            format!(
                                "execution island {id} cannot bind shared KV input '{}.{}'",
                                invocation.component, alias.input
                            )
                        })?;
                    let output_name = linked
                        .outputs
                        .iter()
                        .find_map(|(name, value)| (value == output_value).then_some(name.clone()))
                        .with_context(|| {
                            format!(
                                "execution island {id} cannot bind shared KV output '{}.{}'",
                                invocation.component, alias.output
                            )
                        })?;
                    shared_buffer_inputs.insert(input_name, output_name);
                }
            }
        }
    }
    let structurally_capture_eligible =
        device.starts_with("cuda:") && capture_requested && linked.capture_declines.is_empty();
    let linked_path = (!linked.external_files.is_empty())
        .then(|| materialize_linked_model(&linked))
        .transpose()?;
    let create_session = |options| {
        if let Some(path) = &linked_path {
            Session::new(models.environment(), path, options)
        } else {
            Session::from_model_bytes(
                models.environment(),
                format!("workflow-island-{id}"),
                &linked.bytes,
                options,
            )
        }
    };
    let (session, capture_session_failure) = match create_session(options.clone()) {
        Ok(session) => (session, None),
        Err(error) if capture_requested => {
            options.graph_capture = false;
            let reason = format!(
                "ORT rejected CUDA graph capture for this island; using stable binding: {error}"
            );
            let session = create_session(options)?;
            (session, Some(reason))
        }
        Err(error) => return Err(error.into()),
    };
    let device_allocator = if device.starts_with("cuda:") {
        session
            .device_allocator()
            .with_context(|| format!("execution island {id} could not acquire CUDA allocator"))?
    } else {
        None
    };
    let capture_eligible = structurally_capture_eligible
        && capture_session_failure.is_none()
        && device_allocator.is_some();
    let external_initializer_bytes = linked.external_files.iter().map(|file| file.len).sum();
    let components = invocations
        .iter()
        .map(|invoke| invoke.component.clone())
        .collect();
    let immutable_inputs = linked
        .inputs
        .iter()
        .filter_map(|(name, value)| {
            (value.starts_with("package.")
                || value.starts_with("request.")
                || value.starts_with("vision."))
            .then_some(name.clone())
        })
        .collect();
    let _ = workflow;
    Ok(ExecutionIsland {
        id,
        components,
        inputs: linked.inputs,
        outputs: linked.outputs,
        aliasable_output_values: aliasable_output_values.clone(),
        shared_buffer_inputs,
        immutable_inputs,
        fallback: WorkflowNode::Sequence {
            nodes: invocations
                .iter()
                .map(|invocation| WorkflowNode::Invoke {
                    component: invocation.component.clone(),
                    inputs: invocation.inputs.clone(),
                    outputs: invocation.outputs.clone(),
                    effects: BTreeMap::new(),
                })
                .collect(),
        },
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
        fallback_reason: RefCell::new(if let Some(reason) = capture_session_failure {
            Some(reason)
        } else if !device.starts_with("cuda:") {
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
        execution_generation: Cell::new(0),
    })
}

struct LinkedModel {
    bytes: Vec<u8>,
    inputs: BTreeMap<String, String>,
    outputs: BTreeMap<String, String>,
    capture_declines: Vec<String>,
    external_files: Vec<LinkedExternalFile>,
    node_count: usize,
}

struct LinkedExternalFile {
    virtual_name: String,
    source_path: std::path::PathBuf,
    len: u64,
}

static LINKED_MODEL_STAGING_ID: AtomicU64 = AtomicU64::new(0);

fn materialize_linked_model(linked: &LinkedModel) -> anyhow::Result<std::path::PathBuf> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    linked.bytes.hash(&mut hasher);
    for file in &linked.external_files {
        file.virtual_name.hash(&mut hasher);
        file.source_path.hash(&mut hasher);
        file.len.hash(&mut hasher);
        hash_file_identity(&file.source_path, &mut hasher)?;
    }
    let cache_root = std::env::var_os("ONNX_GENAI_LINKED_MODEL_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or(std::env::current_dir()?.join("target/onnx-genai-linked"));
    let directory = cache_root.join(format!("{:016x}", hasher.finish()));
    if directory.exists() {
        validate_linked_model_directory(linked, &directory)?;
        return Ok(directory.join("model.onnx"));
    }
    std::fs::create_dir_all(&cache_root)?;
    let staging = create_linked_model_staging_directory(&cache_root)?;
    let materialize_result = materialize_linked_model_directory(linked, &staging);
    if let Err(error) = materialize_result {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error);
    }
    match std::fs::rename(&staging, &directory) {
        Ok(()) => {}
        Err(_) if directory.exists() => {
            std::fs::remove_dir_all(&staging)?;
            validate_linked_model_directory(linked, &directory)?;
        }
        Err(error) => {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(error.into());
        }
    }
    Ok(directory.join("model.onnx"))
}

fn create_linked_model_staging_directory(cache_root: &Path) -> anyhow::Result<std::path::PathBuf> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    for _ in 0..1024 {
        let staging = cache_root.join(format!(
            ".staging-{}-{timestamp}-{}",
            std::process::id(),
            LINKED_MODEL_STAGING_ID.fetch_add(1, Ordering::Relaxed)
        ));
        match std::fs::create_dir(&staging) {
            Ok(()) => return Ok(staging),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    bail!(
        "could not allocate a linked-model staging directory under {}",
        cache_root.display()
    )
}

fn materialize_linked_model_directory(
    linked: &LinkedModel,
    directory: &Path,
) -> anyhow::Result<()> {
    for file in &linked.external_files {
        let destination = directory.join(&file.virtual_name);
        link_external_file(&file.source_path, &destination)?;
    }
    std::fs::write(directory.join("model.onnx"), &linked.bytes)?;
    Ok(())
}

fn validate_linked_model_directory(linked: &LinkedModel, directory: &Path) -> anyhow::Result<()> {
    for file in &linked.external_files {
        let destination = directory.join(&file.virtual_name);
        anyhow::ensure!(
            destination.metadata()?.len() == file.len,
            "linked external initializer '{}' changed size",
            file.source_path.display()
        );
    }
    let model_path = directory.join("model.onnx");
    anyhow::ensure!(
        std::fs::read(&model_path)? == linked.bytes,
        "linked model cache collision at {}",
        model_path.display()
    );
    Ok(())
}

fn link_external_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::hard_link(source, destination).or_else(|_| {
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(source, destination)
        }
        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_file(source, destination)
        }
    })
}

fn hash_file_identity(path: &Path, hasher: &mut impl Hasher) -> anyhow::Result<()> {
    let metadata = path.metadata()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        metadata.dev().hash(hasher);
        metadata.ino().hash(hasher);
        metadata.mtime().hash(hasher);
        metadata.mtime_nsec().hash(hasher);
    }
    #[cfg(not(unix))]
    {
        let modified = metadata.modified()?.duration_since(std::time::UNIX_EPOCH)?;
        modified.as_secs().hash(hasher);
        modified.subsec_nanos().hash(hasher);
    }
    Ok(())
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
    let mut external_files = BTreeMap::<String, LinkedExternalFile>::new();
    let mut external_names = BTreeMap::<std::path::PathBuf, String>::new();

    for (index, invocation) in invocations.iter().enumerate() {
        let path = models
            .directory
            .model_paths
            .get(&invocation.component)
            .with_context(|| format!("component '{}' has no model path", invocation.component))?;
        let bytes = onnx_runtime_loader::read_model_binary(path)?;
        let mut source = ModelProto::decode(bytes.as_slice())?;
        merge_local_functions(&mut model.functions, &mut source, id, index);
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
        alpha_rename_graph_dim_params(&mut source_graph, &prefix);
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
        let graph_output_names = source_graph
            .output
            .iter()
            .map(|output| output.name.as_str())
            .collect::<HashSet<_>>();
        for port in invocation.outputs.keys() {
            anyhow::ensure!(
                graph_output_names.contains(port.as_str()),
                "component '{}' workflow output port '{port}' is not a graph output",
                invocation.component
            );
        }
        for output in &source_graph.output {
            let port = &output.name;
            let Some(value_ref) = invocation.outputs.get(port) else {
                continue;
            };
            let internal = &names[port];
            ssa_values.insert(value_ref.clone(), internal.clone());
            if boundary_outputs.contains(value_ref) {
                let mut info = output.clone();
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
        external_files: external_files.into_values().collect(),
        node_count,
    })
}

/// Namespace artifact-local symbolic dimensions before models share one graph.
///
/// ONNX `dim_param` equality is graph-scoped. Component artifacts commonly use
/// generic names such as `batch` and `sequence`, but those names are independent
/// until the linker combines their `ValueInfoProto`s. Prefixing each artifact's
/// symbols preserves equality within that artifact without asserting equality
/// between unrelated component axes.
fn alpha_rename_graph_dim_params(graph: &mut GraphProto, prefix: &str) {
    let mut symbols = BTreeMap::<String, String>::new();
    for value_info in graph
        .input
        .iter_mut()
        .chain(&mut graph.output)
        .chain(&mut graph.value_info)
    {
        if let Some(value_type) = &mut value_info.r#type {
            alpha_rename_type_dim_params(value_type, prefix, &mut symbols);
        }
    }
}

fn merge_local_functions(
    target: &mut Vec<onnx_runtime_loader::proto::onnx::FunctionProto>,
    source: &mut ModelProto,
    island_id: usize,
    component_index: usize,
) {
    let mut functions_to_add = Vec::new();
    let mut function_names = HashMap::new();
    for mut function in source.functions.drain(..) {
        let key = (
            function.domain.clone(),
            function.name.clone(),
            function.overload.clone(),
        );
        let existing = target.iter().find(|existing| {
            existing.domain == function.domain
                && existing.name == function.name
                && existing.overload == function.overload
        });
        let selected_name = match existing {
            Some(existing) if existing.encode_to_vec() == function.encode_to_vec() => {
                function.name.clone()
            }
            Some(_) => format!(
                "island{island_id}_component{component_index}__{}",
                function.name
            ),
            None => function.name.clone(),
        };
        if existing.is_none() || selected_name != function.name {
            function.name = selected_name.clone();
            functions_to_add.push(function);
        }
        function_names.insert(key, selected_name);
    }
    if function_names.is_empty() {
        return;
    }
    if let Some(graph) = source.graph.as_mut() {
        rename_function_calls(&mut graph.node, &function_names);
    }
    for function in &mut functions_to_add {
        rename_function_calls(&mut function.node, &function_names);
    }
    target.extend(functions_to_add);
}

fn rename_function_calls(
    nodes: &mut [onnx_runtime_loader::proto::onnx::NodeProto],
    names: &HashMap<(String, String, String), String>,
) {
    for node in nodes {
        if let Some(name) = names.get(&(
            node.domain.clone(),
            node.op_type.clone(),
            node.overload.clone(),
        )) {
            node.op_type = name.clone();
        }
        for attribute in &mut node.attribute {
            if let Some(graph) = attribute.g.as_mut() {
                rename_function_calls(&mut graph.node, names);
            }
            for graph in &mut attribute.graphs {
                rename_function_calls(&mut graph.node, names);
            }
        }
    }
}

fn alpha_rename_type_dim_params(
    value_type: &mut TypeProto,
    prefix: &str,
    symbols: &mut BTreeMap<String, String>,
) {
    match value_type.value.as_mut() {
        Some(type_proto::Value::TensorType(tensor)) => {
            if let Some(shape) = &mut tensor.shape {
                alpha_rename_shape_dim_params(shape, prefix, symbols);
            }
        }
        Some(type_proto::Value::SparseTensorType(tensor)) => {
            if let Some(shape) = &mut tensor.shape {
                alpha_rename_shape_dim_params(shape, prefix, symbols);
            }
        }
        Some(type_proto::Value::SequenceType(sequence)) => {
            if let Some(element_type) = &mut sequence.elem_type {
                alpha_rename_type_dim_params(element_type, prefix, symbols);
            }
        }
        Some(type_proto::Value::MapType(map)) => {
            if let Some(mapped_type) = &mut map.value_type {
                alpha_rename_type_dim_params(mapped_type, prefix, symbols);
            }
        }
        Some(type_proto::Value::OptionalType(optional)) => {
            if let Some(element_type) = &mut optional.elem_type {
                alpha_rename_type_dim_params(element_type, prefix, symbols);
            }
        }
        Some(type_proto::Value::OpaqueType(_)) | None => {}
    }
}

fn alpha_rename_shape_dim_params(
    shape: &mut TensorShapeProto,
    prefix: &str,
    symbols: &mut BTreeMap<String, String>,
) {
    for dimension in &mut shape.dim {
        let Some(tensor_shape_proto::dimension::Value::DimParam(symbol)) = dimension.value.as_mut()
        else {
            continue;
        };
        let renamed = symbols
            .entry(symbol.clone())
            .or_insert_with(|| format!("{prefix}dim__{symbol}"));
        symbol.clone_from(renamed);
    }
}

fn collect_external_initializers(
    initializers: &mut [TensorProto],
    model_path: &Path,
    prefix: &str,
    external_files: &mut BTreeMap<String, LinkedExternalFile>,
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
            let len = path.metadata()?.len();
            external_files.insert(
                name.clone(),
                LinkedExternalFile {
                    virtual_name: name.clone(),
                    source_path: path.clone(),
                    len,
                },
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

    fn symbolic_tensor(name: &str, symbol: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: 1,
                    shape: Some(TensorShapeProto {
                        dim: vec![
                            tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
                                ..Default::default()
                            },
                            tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimParam(
                                    symbol.to_string(),
                                )),
                                ..Default::default()
                            },
                        ],
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn symbol(value_info: &ValueInfoProto) -> &str {
        let Some(TypeProto {
            value: Some(type_proto::Value::TensorType(tensor)),
            ..
        }) = &value_info.r#type
        else {
            panic!("expected tensor type");
        };
        let Some(tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(symbol)),
            ..
        }) = tensor.shape.as_ref().and_then(|shape| shape.dim.get(1))
        else {
            panic!("expected symbolic second dimension");
        };
        symbol
    }

    fn component(implementation: ComponentImplementation) -> WorkflowComponent {
        WorkflowComponent {
            implementation,
            ports: ComponentPorts::default(),
            contract: None,
            application_overridable: false,
            effects: Vec::new(),
        }
    }

    fn batch_tensor(dtype: &str, rank: usize) -> onnx_genai_metadata::TensorContract {
        onnx_genai_metadata::TensorContract {
            dtype: dtype.into(),
            rank,
            shape: Some(
                std::iter::once(onnx_genai_metadata::TensorDimension::Symbol("batch".into()))
                    .chain(
                        (1..rank)
                            .map(|_| onnx_genai_metadata::TensorDimension::Symbol("axis".into())),
                    )
                    .collect(),
            ),
            optional: false,
        }
    }

    fn singleton_tensor(dtype: &str) -> onnx_genai_metadata::TensorContract {
        onnx_genai_metadata::TensorContract {
            dtype: dtype.into(),
            rank: 1,
            shape: Some(vec![onnx_genai_metadata::TensorDimension::Fixed(1)]),
            optional: false,
        }
    }

    #[test]
    fn external_initializers_are_linked_by_path_without_copying_bytes() {
        let test_root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("island-external-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&test_root);
        std::fs::create_dir_all(&test_root).unwrap();
        let model_path = test_root.join("source.onnx");
        std::fs::write(&model_path, []).unwrap();
        let weights_path = test_root.join("weights.bin");
        std::fs::write(&weights_path, b"0123456789abcdef").unwrap();
        let mut initializer = TensorProto {
            name: "weight".to_string(),
            data_location: tensor_proto::DataLocation::External as i32,
            external_data: vec![
                onnx_runtime_loader::proto::onnx::StringStringEntryProto {
                    key: "location".to_string(),
                    value: "weights.bin".to_string(),
                },
                onnx_runtime_loader::proto::onnx::StringStringEntryProto {
                    key: "offset".to_string(),
                    value: "4".to_string(),
                },
                onnx_runtime_loader::proto::onnx::StringStringEntryProto {
                    key: "length".to_string(),
                    value: "8".to_string(),
                },
                onnx_runtime_loader::proto::onnx::StringStringEntryProto {
                    key: "checksum".to_string(),
                    value: "unchanged".to_string(),
                },
            ],
            ..Default::default()
        };
        let mut files = BTreeMap::new();
        let mut names = BTreeMap::new();
        collect_external_initializers(
            std::slice::from_mut(&mut initializer),
            &model_path,
            "island0_c0_",
            &mut files,
            &mut names,
        )
        .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files.values().next().unwrap().len, 16);
        assert_eq!(
            initializer
                .external_data
                .iter()
                .find(|entry| entry.key == "offset")
                .unwrap()
                .value,
            "4"
        );
        assert_eq!(
            initializer
                .external_data
                .iter()
                .find(|entry| entry.key == "length")
                .unwrap()
                .value,
            "8"
        );
        assert_eq!(
            initializer
                .external_data
                .iter()
                .find(|entry| entry.key == "checksum")
                .unwrap()
                .value,
            "unchanged"
        );

        let linked = std::sync::Arc::new(LinkedModel {
            bytes: b"linked-model".to_vec(),
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            capture_declines: Vec::new(),
            external_files: files.into_values().collect(),
            node_count: 0,
        });
        let linked_paths = (0..4)
            .map(|_| {
                let linked = std::sync::Arc::clone(&linked);
                std::thread::spawn(move || materialize_linked_model(&linked).unwrap())
            })
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert!(linked_paths.windows(2).all(|paths| paths[0] == paths[1]));
        let linked_path = linked_paths[0].clone();
        let linked_weights = linked_path.parent().unwrap().join("island0_c0_external_0");
        assert_eq!(std::fs::read(&linked_weights).unwrap(), b"0123456789abcdef");
        assert_eq!(std::fs::read(&linked_path).unwrap(), b"linked-model");
        let linked_directory = linked_path.parent().unwrap().to_path_buf();

        let replacement = test_root.join("replacement.bin");
        std::fs::write(&replacement, b"fedcba9876543210").unwrap();
        std::fs::rename(replacement, &weights_path).unwrap();
        let replacement_linked_path = materialize_linked_model(&linked).unwrap();
        assert_ne!(replacement_linked_path, linked_path);
        assert_eq!(
            std::fs::read(
                replacement_linked_path
                    .parent()
                    .unwrap()
                    .join("island0_c0_external_0")
            )
            .unwrap(),
            b"fedcba9876543210"
        );
        let replacement_linked_directory = replacement_linked_path.parent().unwrap().to_path_buf();
        std::fs::remove_dir_all(test_root).unwrap();
        std::fs::remove_dir_all(linked_directory).unwrap();
        std::fs::remove_dir_all(replacement_linked_directory).unwrap();
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
        assert!(is_fusible_component(&onnx));

        let adapter = component(ComponentImplementation::Adapter {
            abi: "onnx-genai.grammar-guidance".into(),
            version: "1".into(),
            artifact: None,
            custom_ops: BTreeMap::new(),
        });
        assert!(!is_fusible_component(&adapter));
    }

    #[test]
    fn sampler_fusion_requires_the_per_row_batching_abi() {
        let mut sampler = component(ComponentImplementation::Onnx {
            artifact: "sampler.onnx".into(),
        });
        sampler.contract = Some(onnx_genai_metadata::ComponentContract {
            id: "onnx-genai.token-sampler".into(),
            version: "1".into(),
            bindings: BTreeMap::from([
                ("logits".into(), "logits".into()),
                ("token".into(), "token".into()),
            ]),
            parameters: BTreeMap::new(),
        });
        assert!(!is_fusible_component(&sampler));

        for (role, dtype, rank) in [
            ("logits", "float32", 2),
            ("active", "bool", 1),
            ("done", "bool", 1),
            ("temperature", "float32", 1),
            ("top_k", "int64", 1),
            ("top_p", "float32", 1),
            ("min_p", "float32", 1),
            ("seed", "int64", 1),
            ("counter", "int64", 1),
        ] {
            sampler
                .ports
                .inputs
                .insert(role.into(), batch_tensor(dtype, rank));
        }
        for (role, dtype) in [("token", "int64"), ("next_counter", "int64")] {
            sampler
                .ports
                .outputs
                .insert(role.into(), batch_tensor(dtype, 1));
        }
        {
            let contract = sampler.contract.as_mut().unwrap();
            contract.version = "2".into();
            for role in [
                "logits",
                "active",
                "done",
                "temperature",
                "top_k",
                "top_p",
                "min_p",
                "seed",
                "counter",
            ] {
                contract.bindings.insert(role.into(), role.into());
            }
            for role in ["token", "next_counter"] {
                contract.bindings.insert(role.into(), role.into());
            }
            contract.parameters.insert(
                "batching".into(),
                onnx_genai_metadata::ScalarValue::String("per_row".into()),
            );
            contract.parameters.insert(
                "inactive_rows".into(),
                onnx_genai_metadata::ScalarValue::String("preserve".into()),
            );
        }
        assert!(is_fusible_component(&sampler));

        sampler
            .contract
            .as_mut()
            .unwrap()
            .bindings
            .insert("active".into(), "logits".into());
        assert!(!is_fusible_component(&sampler));
        sampler
            .contract
            .as_mut()
            .unwrap()
            .bindings
            .insert("active".into(), "active".into());
        sampler
            .contract
            .as_mut()
            .unwrap()
            .bindings
            .remove("counter");
        assert!(!is_fusible_component(&sampler));

        sampler
            .contract
            .as_mut()
            .unwrap()
            .bindings
            .insert("counter".into(), "counter".into());
        sampler
            .ports
            .inputs
            .insert("temperature".into(), batch_tensor("float32", 2));
        assert!(!is_fusible_component(&sampler));
    }

    #[test]
    fn termination_and_state_fusion_require_heterogeneous_batch_contracts() {
        let parameters = BTreeMap::from([
            (
                "batching".into(),
                onnx_genai_metadata::ScalarValue::String("per_row".into()),
            ),
            (
                "inactive_rows".into(),
                onnx_genai_metadata::ScalarValue::String("preserve".into()),
            ),
        ]);
        let mut termination = component(ComponentImplementation::Onnx {
            artifact: "termination.onnx".into(),
        });
        for (role, dtype, rank) in [
            ("tokens", "int64", 1),
            ("active", "bool", 1),
            ("eos_ids", "int64", 2),
            ("eos_lengths", "int64", 1),
            ("max_iterations", "int64", 1),
        ] {
            termination
                .ports
                .inputs
                .insert(role.into(), batch_tensor(dtype, rank));
        }
        for role in ["done", "next_active"] {
            termination
                .ports
                .outputs
                .insert(role.into(), batch_tensor("bool", 1));
        }
        termination
            .ports
            .outputs
            .insert("continue".into(), singleton_tensor("bool"));
        termination.contract = Some(onnx_genai_metadata::ComponentContract {
            id: "onnx-genai.termination-predicate".into(),
            version: "2".into(),
            bindings: [
                "tokens",
                "active",
                "eos_ids",
                "eos_lengths",
                "iteration",
                "max_iterations",
                "done",
                "next_active",
                "continue",
            ]
            .into_iter()
            .map(|role| (role.into(), role.into()))
            .collect(),
            parameters: parameters.clone(),
        });
        termination
            .ports
            .inputs
            .insert("iteration".into(), singleton_tensor("int64"));
        assert!(is_fusible_component(&termination));
        termination
            .contract
            .as_mut()
            .unwrap()
            .bindings
            .insert("request_scalar".into(), "request_scalar".into());
        assert!(!is_fusible_component(&termination));
        termination
            .contract
            .as_mut()
            .unwrap()
            .bindings
            .remove("request_scalar");
        let next_active = termination
            .contract
            .as_mut()
            .unwrap()
            .bindings
            .remove("next_active")
            .unwrap();
        assert!(!is_fusible_component(&termination));
        termination
            .contract
            .as_mut()
            .unwrap()
            .bindings
            .insert("next_active".into(), next_active);
        termination
            .ports
            .outputs
            .insert("continue".into(), batch_tensor("bool", 1));
        assert!(!is_fusible_component(&termination));
        termination
            .ports
            .outputs
            .insert("continue".into(), singleton_tensor("bool"));
        termination
            .ports
            .inputs
            .insert("max_iterations".into(), batch_tensor("int64", 1));
        assert!(is_fusible_component(&termination));
        termination
            .ports
            .inputs
            .insert("eos_ids".into(), batch_tensor("int64", 1));
        assert!(!is_fusible_component(&termination));

        let mut state = component(ComponentImplementation::Onnx {
            artifact: "state.onnx".into(),
        });
        state
            .ports
            .inputs
            .insert("current".into(), batch_tensor("int64", 2));
        state
            .ports
            .inputs
            .insert("update".into(), batch_tensor("int64", 2));
        state
            .ports
            .inputs
            .insert("active".into(), batch_tensor("bool", 1));
        state
            .ports
            .inputs
            .insert("done".into(), batch_tensor("bool", 1));
        state
            .ports
            .outputs
            .insert("next".into(), batch_tensor("int64", 2));
        state.contract = Some(onnx_genai_metadata::ComponentContract {
            id: "onnx-genai.state-update".into(),
            version: "2".into(),
            bindings: ["current", "update", "active", "done", "next"]
                .into_iter()
                .map(|role| (role.into(), role.into()))
                .collect(),
            parameters,
        });
        assert!(is_fusible_component(&state));
        state
            .ports
            .outputs
            .insert("next".into(), batch_tensor("float32", 2));
        assert!(!is_fusible_component(&state));
    }

    #[test]
    fn linker_alpha_renames_artifact_local_symbols_with_different_runtime_extents() {
        // Both source artifacts call their independent dynamic axis `sequence`.
        // The first invocation binds it to 2 while the second binds it to 5.
        // Once linked, preserving the original spelling would falsely make
        // those axes equal in the combined ONNX graph.
        let mut first = GraphProto {
            input: vec![symbolic_tensor("first_input", "sequence")],
            output: vec![symbolic_tensor("first_output", "sequence")],
            value_info: vec![symbolic_tensor("first_internal", "sequence")],
            ..Default::default()
        };
        let mut second = GraphProto {
            input: vec![symbolic_tensor("second_input", "sequence")],
            output: vec![symbolic_tensor("second_output", "sequence")],
            value_info: vec![symbolic_tensor("second_internal", "sequence")],
            ..Default::default()
        };

        alpha_rename_graph_dim_params(&mut first, "island0_c0_");
        alpha_rename_graph_dim_params(&mut second, "island0_c1_");

        assert_eq!(symbol(&first.input[0]), symbol(&first.output[0]));
        assert_eq!(symbol(&first.input[0]), symbol(&first.value_info[0]));
        assert_eq!(symbol(&second.input[0]), symbol(&second.output[0]));
        assert_eq!(symbol(&second.input[0]), symbol(&second.value_info[0]));
        assert_ne!(symbol(&first.input[0]), symbol(&second.input[0]));

        let runtime_extents = BTreeMap::from([
            (symbol(&first.input[0]).to_string(), 2_i64),
            (symbol(&second.input[0]).to_string(), 5_i64),
        ]);
        assert_eq!(runtime_extents.len(), 2);
    }

    #[test]
    fn linker_preserves_unique_local_function_identity_and_renames_collisions() {
        fn source(op_type: &str) -> ModelProto {
            ModelProto {
                graph: Some(GraphProto {
                    node: vec![onnx_runtime_loader::proto::onnx::NodeProto {
                        domain: "com.microsoft".into(),
                        op_type: "SkipNorm".into(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                functions: vec![onnx_runtime_loader::proto::onnx::FunctionProto {
                    domain: "com.microsoft".into(),
                    name: "SkipNorm".into(),
                    node: vec![onnx_runtime_loader::proto::onnx::NodeProto {
                        op_type: op_type.into(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }
        }

        let mut linked = ModelProto::default();
        let mut first = source("Add");
        merge_local_functions(&mut linked.functions, &mut first, 0, 0);
        assert_eq!(linked.functions[0].name, "SkipNorm");
        assert_eq!(first.graph.unwrap().node[0].op_type, "SkipNorm");

        let mut identical = source("Add");
        merge_local_functions(&mut linked.functions, &mut identical, 0, 1);
        assert_eq!(linked.functions.len(), 1);
        assert_eq!(identical.graph.unwrap().node[0].op_type, "SkipNorm");

        let mut collision = source("Mul");
        merge_local_functions(&mut linked.functions, &mut collision, 0, 2);
        assert_eq!(linked.functions.len(), 2);
        assert_eq!(linked.functions[1].name, "island0_component2__SkipNorm");
        assert_eq!(
            collision.graph.unwrap().node[0].op_type,
            "island0_component2__SkipNorm"
        );
    }
}
