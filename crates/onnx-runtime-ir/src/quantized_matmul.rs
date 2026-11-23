//! Recognizes the affine quantization contract needed by a packed RHS dot.

use std::collections::HashSet;

use crate::{Attribute, DataType, Dim, Graph, Node, NodeId, ValueId, as_static_shape};

pub struct AffineMatMul {
    pub activation: ValueId,
    pub weight: ValueId,
    pub scale: ValueId,
    pub zero_point: Option<ValueId>,
    pub output: ValueId,
    pub removed: Vec<NodeId>,
    pub k: usize,
    pub n: usize,
    pub block: usize,
    pub transposed: bool,
    pub single_row: bool,
    pub codes: Codes,
}

pub enum Codes {
    Native { signed: bool },
    Unpacked { count: ValueId },
}

impl AffineMatMul {
    pub fn recognize(
        graph: &Graph,
        matmul: &Node,
        captured: &HashSet<ValueId>,
    ) -> Result<Self, &'static str> {
        let [Some(activation), Some(rhs)] = matmul.inputs.as_slice() else {
            return Err("MatMul requires two inputs");
        };
        let [Some(output)] = matmul.outputs.as_slice() else {
            return Err("MatMul requires one output");
        };
        if !matmul.is_default_domain()
            || matmul.op_type != "MatMul"
            || !matmul.attributes.is_empty()
        {
            return Err("expected standard MatMul without attributes");
        }
        let private_producer = |value: ValueId, consumer: NodeId| {
            let metadata = graph.value(value);
            if metadata.is_graph_output
                || graph.num_uses(value) != 1
                || graph.consumers(value) != [consumer]
                || captured.contains(&value)
            {
                return Err("intermediate escapes the MatMul chain");
            }
            metadata
                .producer
                .map(|id| graph.node(id))
                .ok_or("missing producer")
        };
        let mut removed = vec![];
        let post = private_producer(*rhs, matmul.id)?;
        let (dq, transposed) = if post.is_default_domain() && post.op_type == "Transpose" {
            if !post.attributes.iter().all(|(key, attr)| {
                key == "perm" && matches!(attr, Attribute::Ints(p) if p == &[1, 0])
            }) {
                return Err("Transpose must swap the two RHS axes");
            }
            let [Some(input)] = post.inputs.as_slice() else {
                return Err("Transpose requires one input");
            };
            removed.push(post.id);
            (private_producer(*input, post.id)?, true)
        } else {
            (post, false)
        };
        if !dq.is_default_domain()
            || dq.op_type != "DequantizeLinear"
            || graph.effective_opset(dq).unwrap_or(0) < 21
        {
            return Err("expected standard DequantizeLinear, opset 21 or later");
        }
        let (x, scale, zero_point) = match dq.inputs.as_slice() {
            [Some(x), Some(s)] => (*x, *s, None),
            [Some(x), Some(s), z] => (*x, *s, *z),
            _ => return Err("DequantizeLinear requires weights and scales"),
        };
        let [Some(dq_output)] = dq.outputs.as_slice() else {
            return Err("DequantizeLinear requires one output");
        };
        if !dq
            .attributes
            .iter()
            .all(|(name, attr)| match (name.as_str(), attr) {
                ("axis" | "block_size", Attribute::Int(_)) => true,
                ("output_dtype", Attribute::Int(0 | 1)) => {
                    graph.effective_opset(dq).unwrap_or(0) >= 23
                }
                _ => false,
            })
        {
            return Err("DequantizeLinear attributes must preserve FP32 weight rounding");
        }
        let shape = as_static_shape(&graph.value(x).shape).ok_or("weight shape is not static")?;
        let [d0, d1] = shape.as_slice() else {
            return Err("weight must have rank two");
        };
        let (k, n, k_axis) = if transposed {
            (*d1, *d0, 1)
        } else {
            (*d0, *d1, 0)
        };
        let axis = dq.attr("axis").and_then(Attribute::as_int).unwrap_or(1);
        if !(-2..=1).contains(&axis) || axis.rem_euclid(2) != k_axis {
            return Err("quantization axis must be the MatMul contraction axis K");
        }
        let block = dq
            .attr("block_size")
            .and_then(Attribute::as_int)
            .and_then(|block| usize::try_from(block).ok())
            .filter(|&block| block >= 16 && block.is_power_of_two())
            .ok_or("packed dot requires power-of-two block_size >= 16")?;
        if k == 0 || n == 0 {
            return Err("packed dot requires nonempty weights");
        }
        let blocks = k.div_ceil(block);
        let scale_shape = if transposed { [n, blocks] } else { [blocks, n] };
        if as_static_shape(&graph.value(scale).shape).as_deref() != Some(&scale_shape) {
            return Err("scale shape must preserve N and replace K with ceil(K/block_size)");
        }
        for id in [*activation, scale, *dq_output, *rhs, *output] {
            if !graph.value_type_is_known(id) || graph.value(id).dtype != DataType::Float32 {
                return Err("packed dot requires FP32 activation, scales, and results");
            }
        }
        let dtype = graph.value(x).dtype;
        if let Some(zp) = zero_point
            && (graph.value(zp).dtype != dtype
                || as_static_shape(&graph.value(zp).shape).as_deref() != Some(&scale_shape))
        {
            return Err("zero point must match the quantized weight dtype and scale shape");
        }
        let Some((last, leading)) = graph.value(*activation).shape.split_last() else {
            return Err("activation must have known nonzero rank");
        };
        if *last != Dim::Static(k) {
            return Err("activation must have known matching K");
        }
        removed.push(dq.id);
        let (weight, codes) = match dtype {
            DataType::Int4 | DataType::Uint4 => (
                x,
                Codes::Native {
                    signed: dtype == DataType::Int4,
                },
            ),
            DataType::Uint8 => {
                let unpack = private_producer(x, dq.id)?;
                if !unpack.is_default_domain()
                    || unpack.op_type != "Unpack"
                    || graph.effective_opset(unpack).unwrap_or(0) < 29
                    || !unpack
                        .attributes
                        .keys()
                        .all(|s| matches!(s.as_str(), "axis" | "bits" | "radix" | "group_size"))
                {
                    return Err("UINT8 weights require standard 4-bit Unpack");
                }
                let Ok(crate::packed_routed::Op::Unpack(packing)) =
                    crate::packed_routed::Op::from_node(unpack)
                else {
                    return Err("invalid Unpack attributes");
                };
                let [Some(weight), Some(count)] = unpack.inputs.as_slice() else {
                    return Err("Unpack requires packed words and scalar count");
                };
                let packed_shape = if transposed {
                    [n, k.div_ceil(2)]
                } else {
                    [k.div_ceil(2), n]
                };
                if packing.radix != 16
                    || packing.group != 1
                    || packing.axis(2).ok() != Some(k_axis as usize)
                    || graph.value(*weight).dtype != DataType::Uint8
                    || as_static_shape(&graph.value(*weight).shape).as_deref()
                        != Some(&packed_shape)
                    || graph.value(*count).dtype != DataType::Int64
                    || as_static_shape(&graph.value(*count).shape).as_deref() != Some(&[])
                {
                    return Err("Unpack must expose unsigned nibbles along K from UINT8 words");
                }
                removed.push(unpack.id);
                (*weight, Codes::Unpacked { count: *count })
            }
            _ => return Err("expected INT4/UINT4 weights or UINT8 codes from Unpack"),
        };
        Ok(Self {
            activation: *activation,
            weight,
            scale,
            zero_point,
            output: *output,
            removed,
            k,
            n,
            block,
            transposed,
            codes,
            single_row: leading.iter().all(|d| *d == Dim::Static(1)),
        })
    }
}
