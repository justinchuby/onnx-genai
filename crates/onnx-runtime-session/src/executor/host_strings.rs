//! Native host-only execution for kernels consuming managed string tensors.

use super::*;
use onnx_runtime_ep_api::{HostKernelData, HostKernelInput, HostStringInput};
use onnx_runtime_ir::{checked_expected_bytes, compute_contiguous_strides};

use super::geometry::{gather_view, view_bounds};

pub(crate) const MANAGED_STRING_DEVICE_DECLINE: &str =
    "managed tensor(string) values have no device representation; use a string-aware host provider";

pub(crate) fn uses_managed_strings(graph: &Graph, node: &Node) -> bool {
    node.inputs
        .iter()
        .chain(&node.outputs)
        .flatten()
        .any(|&value| graph.value(value).tensor_dtype() == Some(DataType::String))
}

pub(crate) fn producerless_string_outputs(graph: &Graph) -> impl Iterator<Item = ValueId> + '_ {
    graph.outputs.iter().copied().filter(|&value_id| {
        let value = graph.value(value_id);
        value.producer.is_none() && value.tensor_dtype() == Some(DataType::String)
    })
}

fn missing_host_string(graph: &Graph, value: ValueId) -> SessionError {
    let name = graph.value(value).name.as_deref().unwrap_or("<unnamed>");
    SessionError::Internal(format!(
        "tensor(string) '{name}' (value#{}) has no managed host storage; supply a host string tensor or a string-aware producer",
        value.0
    ))
}

pub(crate) fn decode_string_initializer(value: ValueId, weight: &WeightRef) -> Result<Tensor> {
    let WeightRef::Inline(data) = weight else {
        return Err(SessionError::Internal(format!(
            "tensor(string) initializer value#{} must use inline ONNX string_data; external raw bytes cannot represent strings",
            value.0
        )));
    };
    let strings = data
        .strings
        .iter()
        .enumerate()
        .map(|(index, bytes)| {
            std::str::from_utf8(bytes).map(str::to_owned).map_err(|error| {
                SessionError::Internal(format!(
                    "tensor(string) initializer value#{} element {index} is not valid UTF-8: {error}",
                    value.0
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::from_strings(data.dims.clone(), strings)
}

impl Executor {
    pub(super) fn load_string_initializers(
        graph: &Graph,
        ep: &dyn ExecutionProvider,
    ) -> Result<HashMap<ValueId, Tensor>> {
        let mut initializers = HashMap::new();
        for (&value, weight) in &graph.initializers {
            if weight.dtype() != DataType::String {
                continue;
            }
            if !ep.device_id().is_host_accessible() {
                return Err(SessionError::Internal(format!(
                    "initializer value#{} is tensor(string), which has no device representation; use a string-aware host provider instead of '{}'",
                    value.0,
                    ep.name()
                )));
            }
            initializers.insert(value, decode_string_initializer(value, weight)?);
        }
        Ok(initializers)
    }

    pub(super) fn missing_host_string(&self, value: ValueId) -> SessionError {
        missing_host_string(&self.graph, value)
    }

    pub(super) fn exec_host_string_node(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        let plan = &self.plan[pi];
        let node = self.graph.node(plan.node_id);
        let op = node.op_type.clone();
        let fail = |detail| EpError::KernelFailed(format!("{op}: {detail}"));
        if !self.ep.device_id().is_host_accessible() || plan.graph_kernel {
            return Err(fail(
                "managed strings require a native host kernel, not a device or plugin graph kernel",
            )
            .into());
        }
        self.refill_input_shapes(pi, resolved);
        let plan = &self.plan[pi];
        let node = self.graph.node(plan.node_id);
        let input_shapes = &self.scratch_input_shapes;
        let string_values = &self.string_values;
        let string_initializers = &self.string_initializers;
        let graph = &self.graph;
        let bool_inputs = plan
            .inputs
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                let value = slot.ok_or_else(|| {
                    EpError::KernelFailed(format!("{op}: required host input {index} is missing"))
                })?;
                let shape = &input_shapes[index];
                match plan.input_dtypes[index] {
                    DataType::String => Ok(None),
                    DataType::Bool => self
                        .host_bool_bytes(value, shape, external)
                        .map(Some),
                    dtype => Err(EpError::KernelFailed(format!(
                        "{op}: managed host input {index} has unsupported dtype {dtype:?}; expected tensor(string) or tensor(bool)"
                    ))
                    .into()),
                }
            })
            .collect::<Result<smallvec::SmallVec<[_; 2]>>>()?;
        let inputs = plan
            .inputs
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                if let Some(bytes) = &bool_inputs[index] {
                    return Ok(HostKernelInput::Bool {
                        shape: &input_shapes[index],
                        bytes,
                    });
                }
                let value = slot.ok_or_else(|| fail("a required host input is missing"))?;
                let tensor = string_values
                    .get(&value)
                    .or_else(|| string_initializers.get(&value))
                    .ok_or_else(|| missing_host_string(graph, value))?;
                Ok(HostKernelInput::Strings(HostStringInput {
                    shape: &tensor.shape,
                    values: tensor
                        .as_strings()
                        .ok_or_else(|| missing_host_string(graph, value))?,
                }))
            })
            .collect::<Result<smallvec::SmallVec<[_; 2]>>>()?;
        let shape_facts = self.kernel_shape_facts[pi];
        let kernel = if let Some(kernel) = self.kernel_bindings[pi]
            .as_ref()
            .and_then(|binding| self.cache.get_prebound(binding, input_shapes, shape_facts))
        {
            kernel
        } else {
            let count = inputs.len();
            let constant_inputs: smallvec::SmallVec<[_; 2]> = smallvec::smallvec![false; count];
            let constant_values: smallvec::SmallVec<[_; 2]> = smallvec::smallvec![None; count];
            let (kernel, key) = self.cache.get_or_create(
                plan.node_id,
                node,
                input_shapes,
                &plan.input_dtypes,
                &constant_inputs,
                &constant_values,
                effective_opset(&self.graph, node),
                shape_facts,
                self.artifact_config,
                &mut self.provider_artifact_readiness,
                self.ep.as_ref(),
                [None; DeviceGraphSlot::COUNT],
                None,
            )?;
            self.kernel_bindings[pi] = Some(key);
            kernel
        };
        let outputs = kernel.execute_managed_host(&inputs)?;
        drop(inputs);
        if outputs.len() != plan.outputs.len() {
            return Err(fail(
                "native host kernel produced a different output count than the graph declares",
            )
            .into());
        }

        let slots = plan
            .outputs
            .iter()
            .copied()
            .collect::<smallvec::SmallVec<[_; 2]>>();
        for (value, output) in slots.into_iter().zip(outputs) {
            let dtype = match &output.data {
                HostKernelData::Strings(_) => DataType::String,
                HostKernelData::Raw { dtype, .. } if *dtype != DataType::String => *dtype,
                HostKernelData::Raw { .. } => {
                    return Err(fail("kernel returned raw bytes for tensor(string)").into());
                }
            };
            let declared = self.value_dtypes[&value];
            if declared != dtype {
                return Err(EpError::KernelFailed(format!(
                    "{op}: output value#{} is {dtype:?}, but the graph declares {declared:?}",
                    value.0
                ))
                .into());
            }
            if let Some(shape) = self.value_shapes.get(&value).and_then(Option::as_ref)
                && (shape.len() != output.shape.len()
                    || shape.iter().zip(&output.shape).any(
                        |(dim, actual)| matches!(dim, Dim::Static(declared) if declared != actual),
                    ))
            {
                return Err(EpError::KernelFailed(format!(
                    "{op}: output value#{} has shape {:?}, but the graph declares {shape:?}",
                    value.0, output.shape
                ))
                .into());
            }
            match output.data {
                HostKernelData::Strings(strings) => {
                    if external.outputs.contains_key(&value) {
                        return Err(fail("tensor(string) cannot bind a raw external output; use the returned host tensor").into());
                    }
                    let tensor = Tensor::from_strings(output.shape, strings)?;
                    resolved.insert(value, tensor.shape.clone());
                    self.string_values.insert(value, tensor);
                }
                HostKernelData::Raw { bytes, .. } => {
                    if checked_expected_bytes(dtype, &output.shape) != Some(bytes.len()) {
                        return Err(fail("native host kernel returned a byte count incompatible with its dtype and shape").into());
                    }
                    self.store_output_bytes(
                        value,
                        dtype,
                        output.shape,
                        &bytes,
                        resolved,
                        external,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn host_bool_bytes(
        &self,
        value: ValueId,
        shape: &[usize],
        external: &ExternalBindings,
    ) -> Result<Vec<u8>> {
        let Some(binding) = external.inputs.get(&value) else {
            return self.contiguous_bytes(value, shape, DataType::Bool);
        };
        let strides = binding
            .strides
            .as_deref()
            .map_or_else(|| compute_contiguous_strides(shape), <[i64]>::to_vec);
        view_bounds(shape, &strides, 0, DataType::Bool, binding.len).map_err(|cause| {
            SessionError::Internal(format!(
                "managed host BOOL input value#{} cannot read shape {shape:?} from its external binding: {cause}; bind storage that covers the input layout",
                value.0
            ))
        })?;
        let buffer = binding.readable_buffer()?;
        let mut bytes = vec![0; buffer.len()];
        self.ep.copy_to_host(&buffer, &mut bytes)?;
        Ok(gather_view(&bytes, shape, &strides, 0, 1))
    }
}
