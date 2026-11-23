//! A row-major packed RHS can be consumed without the decoded transpose.

use std::collections::HashSet;

use crate::{Attribute, DataType, Dim, Graph, Node, NodeId, ValueId, as_static_shape};

pub struct PackedMatMul {
    pub activation: ValueId,
    pub weight: ValueId,
    pub output: ValueId,
    pub removed: Vec<NodeId>,
    pub k: usize,
    pub n: usize,
    pub single_row: bool,
    pub format: String,
}

impl PackedMatMul {
    pub fn recognize(
        graph: &Graph,
        matmul: &Node,
        captured: &HashSet<ValueId>,
        block_layout: impl Fn(&str) -> Option<(usize, usize)>,
    ) -> Result<Self, &'static str> {
        let [Some(activation), Some(rhs)] = matmul.inputs.as_slice() else {
            return Err("MatMul requires two inputs");
        };
        let [Some(output)] = matmul.outputs.as_slice() else {
            return Err("MatMul requires one output");
        };
        if !matmul.is_default_domain() || matmul.op_type != "MatMul" || !matmul.attributes.is_empty() {
            return Err("expected standard MatMul without attributes");
        }
        let private = |value: ValueId, consumer: NodeId| {
            let value_info = graph.value(value);
            if value_info.is_graph_output
                || captured.contains(&value)
                || graph.uses(value) != [(consumer, 0)] && graph.uses(value) != [(consumer, 1)]
            {
                return Err("decoded weight escapes the MatMul chain");
            }
            if !graph.value_type_is_known(value) || value_info.dtype != DataType::Float32 {
                return Err("GGUF MatMul requires a Float32 weight path");
            }
            value_info.producer.map(|id| graph.node(id)).ok_or("missing producer")
        };
        let transpose = private(*rhs, matmul.id)?;
        if !transpose.is_default_domain() || transpose.op_type != "Transpose"
            || !transpose.attributes.iter().all(|(key, attr)| {
                key == "perm" && matches!(attr, Attribute::Ints(p) if p == &[1, 0])
            })
        {
            return Err("packed GGUF RHS must swap N and K before MatMul");
        }
        let [Some(decoded)] = transpose.inputs.as_slice() else {
            return Err("Transpose requires one input");
        };
        let producer = private(*decoded, transpose.id)?;
        let mut removed = vec![transpose.id];
        let dq = if producer.is_default_domain() && producer.op_type == "Cast" {
            let [Some(input)] = producer.inputs.as_slice() else {
                return Err("Cast requires one input");
            };
            if producer.attributes.len() != 1 || producer.attr("to").and_then(Attribute::as_int) != Some(1) {
                return Err("packed GGUF fusion only removes identity Float32 casts");
            }
            removed.push(producer.id);
            private(*input, producer.id)?
        } else {
            producer
        };
        if !dq.is_default_domain() || dq.op_type != "DequantizePacked"
            || graph.effective_opset(dq).unwrap_or(0) < 29
            || !dq.attributes.keys().all(|s| matches!(s.as_str(), "format" | "shape" | "output_dtype"))
            || !matches!(dq.attr("output_dtype"), None | Some(Attribute::Int(1)))
        {
            return Err("expected standard DequantizePacked with Float32 output");
        }
        let [Some(weight)] = dq.inputs.as_slice() else {
            return Err("DequantizePacked requires one packed input");
        };
        let [n, k] = dq.attr("shape").and_then(Attribute::as_ints).ok_or("missing decoded shape")? else {
            return Err("packed GGUF weights must declare [N,K]");
        };
        let n = usize::try_from(*n).ok().filter(|n| *n > 0).ok_or("invalid N")?;
        let k = usize::try_from(*k).ok().filter(|k| *k > 0).ok_or("invalid K")?;
        if as_static_shape(&graph.value(*decoded).shape).as_deref() != Some(&[n, k])
            || as_static_shape(&graph.value(*rhs).shape).as_deref() != Some(&[k, n])
        {
            return Err("decoded RHS must be a plain [N,K] to [K,N] transpose");
        }
        let format = dq.attr("format").and_then(Attribute::as_str)
            .and_then(|s| s.strip_prefix("gguf:")).ok_or("missing standard GGUF profile")?;
        let (width, bytes) = block_layout(format).ok_or("device cannot execute this packed GGUF layout")?;
        let packed_bytes = n.checked_mul(k / width).and_then(|count| count.checked_mul(bytes))
            .ok_or("packed GGUF byte count overflows")?;
        if !k.is_multiple_of(width)
            || graph.value(*weight).dtype != DataType::Uint8
            || as_static_shape(&graph.value(*weight).shape).as_deref() != Some(&[packed_bytes])
            || !graph.is_constant_initializer(*weight)
        {
            return Err("packed GGUF weights must be immutable UINT8, with complete blocks along K");
        }
        let a = graph.value(*activation);
        let y = graph.value(*output);
        if !graph.value_type_is_known(*activation) || !graph.value_type_is_known(*output)
            || a.dtype != DataType::Float32 || y.dtype != DataType::Float32
        {
            return Err("packed GGUF MatMul requires Float32 activation and result");
        }
        let Some((Dim::Static(depth), leading)) = a.shape.split_last() else {
            return Err("activation must have known nonzero rank and K");
        };
        if *depth != k {
            return Err("activation K does not match packed RHS");
        }
        removed.push(dq.id);
        Ok(Self {
            activation: *activation, weight: *weight, output: *output,
            removed, k, n, single_row: leading.iter().all(|d| *d == Dim::Static(1)),
            format: format.to_owned(),
        })
    }
}
