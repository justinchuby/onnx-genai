//! Provider-selected lowering of immutable standard affine RHS operands.

use onnx_runtime_ir::quantized_matmul::{AffineMatMul, Codes};
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef, static_shape,
};

use crate::{OptimizationPass, PassContext, Result};

pub const ROUND_AFFINE_WEIGHTS: &str = "_nxrt_round_affine_weights";

pub struct AffineMatMulFusion {
    /// CUDA only uses the packed GEMV when the graph proves one row.
    pub single_row_only: bool,
    pub max_columns: usize,
}

impl OptimizationPass for AffineMatMulFusion {
    fn name(&self) -> &str {
        "AffineMatMulFusion"
    }

    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> Result<()> {
        let candidates: Vec<_> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.is_default_domain() && node.op_type == "MatMul").then_some(id)
            })
            .collect();
        let captured: std::collections::HashSet<_> = graph
            .captured_inputs_by_node()
            .into_values()
            .flatten()
            .collect();
        for id in candidates {
            let Ok(pattern) = AffineMatMul::recognize(graph, graph.node(id), &captured) else {
                continue;
            };
            if pattern.n > self.max_columns
                || self.single_row_only
                    && (!pattern.single_row
                        || [pattern.k, pattern.n, pattern.block]
                            .iter()
                            .any(|&v| v > i32::MAX as usize))
            {
                continue;
            }
            let Ok(repacked) = Repacked::new(graph, ctx, &pattern) else {
                continue;
            };
            let [b, s, z] = repacked.install(graph, &pattern);
            let mut fused = Node::new(
                NodeId(0),
                "MatMulNBits",
                vec![Some(pattern.activation), b, s, z],
                vec![pattern.output],
            );
            fused.domain = "com.microsoft".into();
            for (name, value) in [
                ("K", pattern.k as i64),
                ("N", pattern.n as i64),
                ("bits", 4),
                ("block_size", pattern.block as i64),
                ("accuracy_level", 0),
                (ROUND_AFFINE_WEIGHTS, 1),
            ] {
                fused.attributes.insert(name.into(), Attribute::Int(value));
            }
            graph.opset_imports.entry(fused.domain.clone()).or_insert(1);
            graph.replace_node(id, fused);
            graph.remove_nodes(&pattern.removed);
            let count = match pattern.codes {
                Codes::Native { .. } => None,
                Codes::Unpacked { count } => Some(count),
            };
            for source in [
                Some(pattern.weight),
                Some(pattern.scale),
                pattern.zero_point,
                count,
            ]
            .into_iter()
            .flatten()
            {
                if !graph.has_uses(source)
                    && !graph.value(source).is_graph_output
                    && !captured.contains(&source)
                {
                    graph.initializers.remove(&source);
                    graph.gc_value_if_orphan(source);
                }
            }
        }
        Ok(())
    }
}

struct Repacked {
    weight: Vec<u8>,
    scales: Vec<u8>,
    zero_points: Option<Vec<u8>>,
}

impl Repacked {
    fn new(
        graph: &Graph,
        ctx: &PassContext,
        p: &AffineMatMul,
    ) -> std::result::Result<Self, &'static str> {
        let bytes = |id| {
            if !graph.is_constant_initializer(id) {
                return Err("packed operands must be immutable dense initializers");
            }
            graph
                .initializers
                .get(&id)
                .and_then(|w| ctx.initializer_bytes(w))
                .ok_or("could not resolve packed initializer bytes")
        };
        let weight_src = bytes(p.weight)?;
        let scale_src = bytes(p.scale)?;
        let zp_src = p.zero_point.map(bytes).transpose()?;
        let blocks = p.k.div_ceil(p.block);
        let scale_count = p.n.checked_mul(blocks).ok_or("scale shape overflow")?;
        let weight_count = p.n.checked_mul(p.k).ok_or("weight shape overflow")?;
        let (src_len, sign_bias, unpacked) = match p.codes {
            Codes::Native { signed } => {
                (weight_count.div_ceil(2), if signed { 8 } else { 0 }, false)
            }
            Codes::Unpacked { count } => {
                let count =
                    <[u8; 8]>::try_from(bytes(count)?).map_err(|_| "invalid Unpack count")?;
                if usize::try_from(i64::from_le_bytes(count)).ok() != Some(p.k) {
                    return Err("Unpack count must equal logical K");
                }
                (
                    p.n.checked_mul(p.k.div_ceil(2))
                        .ok_or("packed shape overflow")?,
                    0,
                    true,
                )
            }
        };
        let scale_len = scale_count
            .checked_mul(4)
            .ok_or("scale byte size overflow")?;
        if weight_src.len() != src_len
            || scale_src.len() != scale_len
            || zp_src.is_some_and(|z| {
                z.len()
                    != if unpacked {
                        scale_count
                    } else {
                        scale_count.div_ceil(2)
                    }
            })
        {
            return Err("initializer byte length disagrees with weight/parameter shape");
        }
        // All conversions are a permutation and/or a bias of integer codes;
        // no real-valued weight is ever constructed. Check scale values before
        // bypassing standalone CPU DequantizeLinear's range validation.
        if scale_src.as_chunks::<4>().0.iter().any(|s| {
            let s = f32::from_le_bytes(*s);
            !s.is_finite() || s <= 0.0
        }) || zp_src.is_some_and(|z| unpacked && z.iter().any(|&x| x > 15))
        {
            return Err(
                "scales must be finite and positive; Unpack zero points must fit four bits",
            );
        }
        let row_bytes = blocks
            .checked_mul(p.block / 2)
            .ok_or("packed row overflow")?;
        let weight_len =
            p.n.checked_mul(row_bytes)
                .ok_or("packed RHS byte size overflow")?;
        if weight_len > isize::MAX as usize {
            return Err("packed RHS exceeds addressable size");
        }
        let mut weight = vec![0; weight_len];
        let mut scales = vec![0; scale_len];
        // MatMulNBits defaults to 8, whereas ONNX DequantizeLinear defaults to 0.
        let zp_row = blocks.div_ceil(2);
        let mut zero_points = (zp_src.is_some() || sign_bias == 0).then(|| vec![0; p.n * zp_row]);
        let get_nibble = |bytes: &[u8], i| (bytes[i / 2] >> ((i % 2) * 4)) & 15;
        for n in 0..p.n {
            for k in 0..p.k {
                let q = if unpacked {
                    let byte = if p.transposed {
                        n * p.k.div_ceil(2) + k / 2
                    } else {
                        (k / 2) * p.n + n
                    };
                    (weight_src[byte] >> ((k % 2) * 4)) & 15
                } else {
                    get_nibble(
                        weight_src,
                        if p.transposed {
                            n * p.k + k
                        } else {
                            k * p.n + n
                        },
                    )
                } ^ sign_bias;
                weight[n * row_bytes + k / 2] |= q << ((k % 2) * 4);
            }
            for b in 0..blocks {
                let src = if p.transposed {
                    n * blocks + b
                } else {
                    b * p.n + n
                };
                let dst = n * blocks + b;
                scales[dst * 4..dst * 4 + 4].copy_from_slice(&scale_src[src * 4..src * 4 + 4]);
                if let (Some(z), Some(out)) = (zp_src, &mut zero_points) {
                    let code = if unpacked {
                        z[src]
                    } else {
                        get_nibble(z, src) ^ sign_bias
                    };
                    out[n * zp_row + b / 2] |= code << ((b % 2) * 4);
                }
            }
        }
        Ok(Self {
            weight,
            scales,
            zero_points,
        })
    }

    fn install(self, graph: &mut Graph, p: &AffineMatMul) -> [Option<ValueId>; 3] {
        let mut make = |dtype, shape: Vec<usize>, raw| {
            let value = graph.create_value(dtype, static_shape(shape.iter().copied()));
            graph.set_initializer(
                value,
                WeightRef::Inline(TensorData::from_raw(dtype, shape, raw)),
            );
            Some(value)
        };
        let blocks = p.k.div_ceil(p.block);
        [
            make(DataType::Uint8, vec![p.n, blocks, p.block / 2], self.weight),
            make(DataType::Float32, vec![p.n, blocks], self.scales),
            self.zero_points
                .and_then(|z| make(DataType::Uint8, vec![p.n, blocks.div_ceil(2)], z)),
        ]
    }
}
