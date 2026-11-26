use super::*;
use onnx_runtime_ir::onnx::{TypeProto, type_proto};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PayloadKind {
    Tensor,
    Sequence,
}

pub(super) enum OptionalPayload {
    Tensor(SeqTensor),
    StringTensor(Tensor),
    Sequence(SequenceValue),
}

pub(super) fn is_optional_op(op: &str, domain: &str) -> bool {
    domain.is_empty() && matches!(op, "Optional" | "OptionalHasElement" | "OptionalGetElement")
}

pub(super) fn graph_uses_optional_ops(graph: &Graph) -> bool {
    graph
        .nodes
        .iter()
        .any(|(_, node)| is_optional_op(&node.op_type, &node.domain))
        || graph.subgraphs.values().any(graph_uses_optional_ops)
}

fn error(op: &str, reason: impl Into<String>) -> SessionError {
    SessionError::OptionalOp {
        op: op.into(),
        reason: reason.into(),
    }
}

fn element_kind(tp: &TypeProto) -> Option<PayloadKind> {
    let tensor = |tp: &TypeProto| match &tp.value {
        Some(type_proto::Value::TensorType(t)) => {
            DataType::from_onnx(t.elem_type).is_some_and(|dtype| dtype != DataType::Undefined)
        }
        _ => false,
    };
    match &tp.value {
        _ if tensor(tp) => Some(PayloadKind::Tensor),
        Some(type_proto::Value::SequenceType(s)) if s.elem_type.as_deref().is_some_and(tensor) => {
            Some(PayloadKind::Sequence)
        }
        _ => None,
    }
}

fn declared_optional(graph: &Graph, vid: ValueId) -> bool {
    matches!(
        graph
            .value(vid)
            .declared_type
            .as_ref()
            .and_then(|tp| tp.value.as_ref()),
        Some(type_proto::Value::OptionalType(_))
    )
}

pub(super) fn classify_optional_values(
    graph: &Graph,
    order: &[NodeId],
    sequences: &mut HashSet<ValueId>,
) -> Result<HashMap<ValueId, PayloadKind>> {
    let mut optionals = HashMap::new();
    for &input in &graph.inputs {
        if declared_optional(graph, input) {
            return Err(error(
                "<graph input>",
                format!(
                    "value '{}' is Optional; the standalone input API accepts tensors only. Supply a tensor input and construct Optional inside the graph",
                    graph.value(input).name.as_deref().unwrap_or("<unnamed>"),
                ),
            ));
        }
    }
    for &nid in order {
        let node = graph.node(nid);
        let op = node.op_type.as_str();
        let is_optional = is_optional_op(op, &node.domain);
        let input = node.inputs.first().copied().flatten();
        for &vid in node.inputs.iter().flatten() {
            if (optionals.contains_key(&vid) || declared_optional(graph, vid))
                && !(is_optional
                    && matches!(op, "OptionalHasElement" | "OptionalGetElement")
                    && optionals.contains_key(&vid))
            {
                return Err(error(
                    op,
                    format!(
                        "node #{} cannot consume Optional value #{} from this producer; construct it in the same graph and use OptionalHasElement or OptionalGetElement",
                        nid.0, vid.0,
                    ),
                ));
            }
        }
        if !is_optional {
            continue;
        }
        let invalid = |reason: &str| error(op, format!("node #{}: {reason}", nid.0));
        let opset = graph.effective_opset(node).unwrap_or_default();
        if opset < 15 {
            return Err(invalid("requires ONNX opset 15 or newer"));
        }
        let [Some(output)] = node.outputs.as_slice() else {
            return Err(invalid("expected exactly one named output"));
        };
        if node.inputs.len() > 1 {
            return Err(invalid("expected at most one input slot"));
        }
        let operand_kind = |vid| {
            if sequences.contains(&vid) {
                return Some(PayloadKind::Sequence);
            }
            match graph.value(vid).declared_type.as_ref() {
                Some(tp) if tp.value.is_some() => element_kind(tp),
                _ => Some(PayloadKind::Tensor),
            }
        };
        match op {
            "Optional" => {
                let kind = match input {
                    Some(vid) => operand_kind(vid),
                    None => match node.attr("type") {
                        Some(Attribute::TypeProto(tp)) => element_kind(tp),
                        _ => None,
                    },
                }.ok_or_else(|| invalid("supply a tensor or tensor-sequence input, or omit it and set the 'type' attribute to a tensor or tensor-sequence TypeProto"))?;
                optionals.insert(*output, kind);
            }
            "OptionalHasElement" | "OptionalGetElement" => {
                let kind = match input {
                    Some(vid) if optionals.contains_key(&vid) => optionals.get(&vid).copied(),
                    Some(vid) if opset >= 18 => operand_kind(vid),
                    None if opset >= 18 && op == "OptionalHasElement" => continue,
                    _ => None,
                }
                .ok_or_else(|| {
                    invalid(if opset < 18 {
                        "requires an Optional input before opset 18"
                    } else {
                        "supply an Optional, tensor, or tensor-sequence input"
                    })
                })?;
                if op == "OptionalGetElement" && kind == PayloadKind::Sequence {
                    sequences.insert(*output);
                }
            }
            _ => unreachable!("checked by is_optional_op"),
        }
    }
    for &output in &graph.outputs {
        if optionals.contains_key(&output) || declared_optional(graph, output) {
            return Err(error(
                "<graph output>",
                format!(
                    "value '{}' is Optional; the standalone output API cannot return Optional values. Export its presence or extract its payload inside the graph",
                    graph.value(output).name.as_deref().unwrap_or("<unnamed>"),
                ),
            ));
        }
    }
    if !optionals.is_empty() && !graph.subgraphs.is_empty() {
        for (nid, captures) in graph.captured_inputs_by_node() {
            if let Some(vid) = captures.iter().find(|vid| optionals.contains_key(vid)) {
                return Err(error(
                    &graph.node(nid).op_type,
                    format!(
                        "node #{} captures Optional value #{} across a subgraph boundary; extract the payload in its defining graph before capturing it",
                        nid.0, vid.0,
                    ),
                ));
            }
        }
    }
    Ok(optionals)
}

impl Executor {
    pub(super) fn is_container_value(&self, vid: ValueId) -> bool {
        self.sequence_values.contains(&vid) || self.optional_values.contains_key(&vid)
    }

    fn optional_payload(
        &mut self,
        vid: ValueId,
        op: &str,
        resolved: &HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<OptionalPayload> {
        if self.sequence_values.contains(&vid) {
            return self
                .get_sequence(Some(vid), op)
                .map(OptionalPayload::Sequence);
        }
        if self.value_dtypes.get(&vid) == Some(&DataType::String) {
            let tensor = self
                .string_values
                .get(&vid)
                .or_else(|| self.string_initializers.get(&vid))
                .ok_or_else(|| self.missing_host_string(vid));
            return tensor?
                .try_clone()
                .map(OptionalPayload::StringTensor)
                .map_err(seq_err);
        }
        if let Some(value) = external
            .inputs
            .get(&vid)
            .or_else(|| external.outputs.get(&vid))
        {
            let source = value.readable_buffer()?;
            let shape = resolved.get(&vid).unwrap_or(&value.shape);
            if value
                .strides
                .as_ref()
                .is_some_and(|strides| *strides != compute_contiguous_strides(shape))
            {
                return Err(error(
                    op,
                    "bound device input requires contiguous payload storage; materialize the tensor before binding",
                ));
            }
            let size = checked_storage_bytes(
                value.dtype,
                checked_numel(shape, || format!("Optional input #{}", vid.0))?,
                || format!("Optional input #{}", vid.0),
                shape,
            )?;
            if size > source.len() {
                return Err(error(
                    op,
                    format!(
                        "bound input #{} requires {size} bytes but has capacity {}",
                        vid.0,
                        source.len()
                    ),
                ));
            }
            let mut buffer = self.ep.allocate(size.max(1), source.alignment())?;
            if size != 0
                && let Err(cause) = self.ep.copy(&source, &mut buffer, size)
            {
                self.ep.deallocate(buffer)?;
                return Err(cause.into());
            }
            return SeqTensor::from_shared(
                SharedTensorBuffer::new(Arc::clone(&self.ep), buffer),
                value.dtype,
                shape.clone(),
                TensorLayout::contiguous(),
                0,
            )
            .map(OptionalPayload::Tensor)
            .map_err(seq_err);
        }
        self.read_seq_element(vid, resolved)
            .map(OptionalPayload::Tensor)
            .map_err(|source| {
                error(
                    op,
                    format!("cannot read payload value #{}: {source}", vid.0),
                )
            })
    }

    pub(super) fn exec_optional_node(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        let plan = &self.plan[pi];
        let input = plan.inputs.first().copied().flatten();
        let output = plan.outputs[0];
        let op = self.graph.node(plan.node_id).op_type.clone();
        match op.as_str() {
            "Optional" => {
                let payload = input
                    .map(|vid| self.optional_payload(vid, &op, resolved, external))
                    .transpose()?;
                self.optionals.insert(output, payload);
            }
            "OptionalHasElement" => {
                let present = match input {
                    None => false,
                    Some(vid) if self.optional_values.contains_key(&vid) => {
                        self.live_optional(vid, &op)?.is_some()
                    }
                    Some(_) => true,
                };
                self.store_raw_tensor_output(
                    output,
                    DataType::Bool,
                    vec![],
                    &[u8::from(present)],
                    resolved,
                    external,
                )?;
            }
            "OptionalGetElement" => {
                let vid = input.ok_or_else(|| {
                    error(
                        &op,
                        "supply a non-empty Optional or a permitted direct operand",
                    )
                })?;
                let payload = if self.optional_values.contains_key(&vid) {
                    let payload = self.live_optional(vid, &op)?.as_ref().ok_or_else(|| error(&op, format!(
                        "input value #{} is absent; check OptionalHasElement before extracting its payload",
                        vid.0,
                    )))?;
                    match payload {
                        OptionalPayload::Tensor(tensor) => OptionalPayload::Tensor(tensor.clone()),
                        OptionalPayload::StringTensor(tensor) => {
                            OptionalPayload::StringTensor(tensor.try_clone().map_err(seq_err)?)
                        }
                        OptionalPayload::Sequence(seq) => OptionalPayload::Sequence(seq.clone()),
                    }
                } else {
                    self.optional_payload(vid, &op, resolved, external)?
                };
                match payload {
                    OptionalPayload::Tensor(elem) => {
                        if let Some(binding) = external.outputs.get(&output) {
                            let size = checked_storage_bytes(
                                elem.dtype,
                                elem.numel(),
                                || format!("Optional output #{}", output.0),
                                &elem.shape,
                            )?;
                            if elem.byte_offset() != 0
                                || elem.layout.resolved_strides(&elem.shape)
                                    != compute_contiguous_strides(&elem.shape)
                                || binding.strides.as_ref().is_some_and(|strides| {
                                    *strides != compute_contiguous_strides(&elem.shape)
                                })
                            {
                                return Err(error(
                                    &op,
                                    "bound device output requires contiguous, zero-offset payload and output storage; materialize the tensor before binding",
                                ));
                            }
                            self.prepare_tensor_output(
                                output,
                                elem.dtype,
                                elem.shape.clone(),
                                size,
                                resolved,
                                external,
                            )?;
                            if size != 0 {
                                self.ep.copy(
                                    elem.storage().buffer(),
                                    &mut binding.writable_buffer()?,
                                    size,
                                )?;
                            }
                        } else {
                            self.store_seq_element_output(output, elem, resolved, external)?;
                        }
                    }
                    OptionalPayload::StringTensor(tensor) => {
                        resolved.insert(output, tensor.shape.clone());
                        self.value_dtypes.insert(output, DataType::String);
                        self.string_values.insert(output, tensor);
                    }
                    OptionalPayload::Sequence(sequence) => {
                        self.sequences.insert(output, sequence);
                    }
                }
            }
            _ => unreachable!("checked by is_optional_op"),
        }
        Ok(())
    }

    fn live_optional(&self, vid: ValueId, op: &str) -> Result<&Option<OptionalPayload>> {
        self.optionals.get(&vid).ok_or_else(|| error(op, format!(
            "input value #{} has no live Optional; ensure its Optional producer executes in the same graph",
            vid.0,
        )))
    }
}
