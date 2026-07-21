use onnx_runtime_ir::{Attribute, DataType, Graph, NodeId, ValueId};
use onnx_runtime_optimizer::{
    OptimizationPass, OptimizerError, PassContext, Result as OptimizerResult,
};

pub(crate) const SILU_MUL_FUSION_ATTR: &str = "_cuda_silu_mul";

/// Private marker set on a `MatMulNBits` node whose trailing bias input came
/// from folding a *separate* elementwise `Add(MatMulNBits(x), bias)`.
///
/// The distinction matters for fp16 numerics: the two-op path rounds the GEMV
/// accumulator to fp16 first and then adds the fp16 bias (a second fp16 round),
/// whereas a *native* MatMulNBits bias is an epilogue add rounded only once. The
/// CUDA GEMV consumes this marker to reproduce the fp16-after-round form so the
/// fused decode keeps byte-identical greedy tokens.
pub(crate) const MATMUL_NBITS_FOLDED_BIAS_ATTR: &str = "_cuda_matmul_nbits_folded_bias";

/// Private marker set on a synthetic `MatMulNBits` node that fuses the paired
/// gate/up projections *and* the trailing `Silu(gate) * up` (SwiGLU) into a
/// single kernel. Its five inputs are `[x, W_gate, scales_gate, W_up,
/// scales_up]` (not the standard `[x, B, scales, zero_points, g_idx]`); the CUDA
/// factory recognizes the marker and dispatches the paired kernel instead of the
/// ordinary GEMV. Standard `MatMulNBits` shape inference derives the output from
/// input 0 and the `N` attribute only, so the extra weight inputs are ignored and
/// the inferred `[.., N]` shape stays correct. The session restores the pre-pass
/// graph before any non-CUDA fallback, so no other EP ever sees this node.
pub(crate) const GATE_UP_SWIGLU_FUSION_ATTR: &str = "_cuda_gate_up_swiglu";

const MICROSOFT_DOMAIN: &str = "com.microsoft";

/// Exact validated decode shape for the paired gate/up SwiGLU fusion. Gating on
/// these keeps the pass from misfiring on any other `Mul`/activation graph and
/// guarantees the paired kernel only ever runs on dimensions it was verified
/// against (block-32, fp16, M=1 decode).
const GATE_UP_SWIGLU_K: usize = 896;
const GATE_UP_SWIGLU_N: usize = 4864;
const GATE_UP_SWIGLU_BLOCK_SIZE: usize = 32;

/// Fuse `Mul(Silu(gate), up)` into CUDA's tagged two-input `Mul` variant.
///
/// Keeping the node as standard `Mul` preserves ordinary binary shape
/// inference. The private marker is consumed only by the CUDA kernel factory;
/// the session restores the pre-pass graph before falling back to another EP.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaSwiGluFusion;

pub(crate) fn cuda_optimization_passes() -> Vec<Box<dyn OptimizationPass>> {
    vec![
        Box::new(CudaMatMulNBitsBiasFusion),
        Box::new(CudaSwiGluFusion),
        Box::new(CudaGateUpSwiGluFusion),
    ]
}

/// Fold a standalone `Add(MatMulNBits(x), bias)` into the `MatMulNBits` bias
/// input, removing the separate elementwise launch.
///
/// Only the exact QKV-style decode pattern is fused: a `MatMulNBits` with no
/// zero-points / group-index / existing bias, whose sole consumer is a plain
/// two-input `Add` against a 1-D initializer bias of shape `[N]` and matching
/// element type. The fused node keeps its standard `MatMulNBits` op type (so
/// ordinary shape inference and non-CUDA fallback are unaffected) and gains the
/// private [`MATMUL_NBITS_FOLDED_BIAS_ATTR`] marker so the CUDA GEMV reproduces
/// the two-op fp16 rounding exactly.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaMatMulNBitsBiasFusion;

impl OptimizationPass for CudaMatMulNBitsBiasFusion {
    fn name(&self) -> &str {
        "CudaMatMulNBitsBiasFusion"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> OptimizerResult<()> {
        let add_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.op_type == "Add"
                    && matches!(node.domain.as_str(), "" | "ai.onnx")
                    && node.inputs.len() == 2
                    && node.outputs.len() == 1
                    && node.attributes.is_empty())
                .then_some(id)
            })
            .collect();

        let mut plans: Vec<BiasFoldPlan> = Vec::new();
        for add_id in add_nodes {
            if let Some(plan) = self.plan_fold(graph, add_id) {
                plans.push(plan);
            }
        }

        let changed = !plans.is_empty();
        for plan in plans {
            // Rewire `add_out`'s consumers (and any graph-output slot) onto the
            // MatMulNBits output, drop the now-dead Add, then rebuild the
            // MatMulNBits node with the bias slot populated. Ordering keeps the
            // arena free of dangling values at every step. The surviving value
            // inherits the Add output's name so downstream/output binding by
            // name is unaffected.
            let downstream_name = graph.value(plan.add_out).name.clone();
            graph.replace_all_uses(plan.add_out, plan.matmul_out);
            graph.remove_node(plan.add_id);
            if downstream_name.is_some() {
                graph.value_mut(plan.matmul_out).name = downstream_name;
            }

            let mut fused = graph.node(plan.matmul_id).clone();
            fused.inputs = vec![
                plan.matmul_inputs[0],
                plan.matmul_inputs[1],
                plan.matmul_inputs[2],
                None,
                None,
                Some(plan.bias),
            ];
            fused
                .attributes
                .insert(MATMUL_NBITS_FOLDED_BIAS_ATTR.into(), Attribute::Int(1));
            graph.replace_node(plan.matmul_id, fused);
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

struct BiasFoldPlan {
    add_id: NodeId,
    matmul_id: NodeId,
    matmul_inputs: [Option<ValueId>; 3],
    matmul_out: ValueId,
    add_out: ValueId,
    bias: ValueId,
}

impl CudaMatMulNBitsBiasFusion {
    fn plan_fold(&self, graph: &Graph, add_id: NodeId) -> Option<BiasFoldPlan> {
        let add = graph.try_node(add_id)?;
        let lhs = add.inputs[0]?;
        let rhs = add.inputs[1]?;
        let add_out = add.outputs[0];

        // Exactly one Add operand must be a foldable MatMulNBits output; the
        // other is the bias.
        let (matmul_out, bias) = match (
            self.matmul_producer(graph, lhs),
            self.matmul_producer(graph, rhs),
        ) {
            (Some(_), Some(_)) => return None,
            (Some(_), None) => (lhs, rhs),
            (None, Some(_)) => (rhs, lhs),
            (None, None) => return None,
        };
        let matmul_id = self.matmul_producer(graph, matmul_out)?;

        // The GEMV output must feed only this Add and must not escape as a
        // graph output (folding would otherwise drop an observable value).
        if graph.consumers(matmul_out).len() != 1 || graph.value(matmul_out).is_graph_output {
            return None;
        }

        let matmul = graph.node(matmul_id);
        // Only the plain A/B/scales form is eligible: no zero-points, group
        // index, or pre-existing bias.
        let present: Vec<ValueId> = matmul.input_values().collect();
        if present.len() != 3 || matmul.inputs.iter().skip(3).any(Option::is_some) {
            return None;
        }
        let n = matmul.attr("N").and_then(Attribute::as_int)? as usize;

        // Bias must be a persistent 1-D `[N]` initializer whose element type
        // matches the GEMV output, so the fused node is capture-safe and the
        // epilogue add is well-typed.
        if !graph.initializers.contains_key(&bias) {
            return None;
        }
        let bias_value = graph.value(bias);
        let out_value = graph.value(matmul_out);
        if bias_value.dtype != out_value.dtype
            || !matches!(
                bias_value.dtype,
                DataType::Float16 | DataType::Float32 | DataType::BFloat16
            )
        {
            return None;
        }
        let bias_dims = onnx_runtime_ir::as_static_shape(&bias_value.shape)?;
        if bias_dims != [n] {
            return None;
        }

        Some(BiasFoldPlan {
            add_id,
            matmul_id,
            matmul_inputs: [matmul.inputs[0], matmul.inputs[1], matmul.inputs[2]],
            matmul_out,
            add_out,
            bias,
        })
    }

    fn matmul_producer(&self, graph: &Graph, value: ValueId) -> Option<NodeId> {
        let producer = graph.try_value(value)?.producer?;
        let node = graph.try_node(producer)?;
        (node.op_type == "MatMulNBits" && node.domain == MICROSOFT_DOMAIN).then_some(producer)
    }
}

impl OptimizationPass for CudaSwiGluFusion {
    fn name(&self) -> &str {
        "CudaSwiGluFusion"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> OptimizerResult<()> {
        let silu_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.op_type == "Silu"
                    && node.domain == MICROSOFT_DOMAIN
                    && node.inputs.len() == 1
                    && node.outputs.len() == 1)
                    .then_some(id)
            })
            .collect();

        let mut changed = false;
        for silu_id in silu_nodes {
            let Some(silu) = graph.try_node(silu_id) else {
                continue;
            };
            let Some(gate) = silu.inputs[0] else {
                continue;
            };
            let silu_output = silu.outputs[0];
            if graph.outputs.contains(&silu_output) {
                continue;
            }
            let consumers = graph.consumers(silu_output);
            if consumers.len() != 1 {
                continue;
            }

            let mul_id = consumers[0];
            let mul = graph.node(mul_id);
            if mul.op_type != "Mul"
                || !matches!(mul.domain.as_str(), "" | "ai.onnx")
                || mul.inputs.len() != 2
                || mul.outputs.len() != 1
                || !mul.attributes.is_empty()
            {
                continue;
            }
            let up = if mul.inputs[0] == Some(silu_output) {
                mul.inputs[1]
            } else if mul.inputs[1] == Some(silu_output) {
                mul.inputs[0]
            } else {
                None
            };
            let Some(up) = up else {
                continue;
            };

            let gate_value = graph.value(gate);
            let up_value = graph.value(up);
            if gate_value.dtype != up_value.dtype
                || gate_value.shape != up_value.shape
                || !matches!(
                    gate_value.dtype,
                    DataType::Float16 | DataType::Float32 | DataType::BFloat16
                )
            {
                continue;
            }

            let mut fused = mul.clone();
            fused.inputs = vec![Some(gate), Some(up)];
            fused
                .attributes
                .insert(SILU_MUL_FUSION_ATTR.into(), Attribute::Int(1));
            graph.replace_node(mul_id, fused);
            graph.remove_node(silu_id);
            changed = true;
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

/// Fuse the paired gate/up projections plus their `Silu(gate) * up` (SwiGLU)
/// into one synthetic `MatMulNBits` node consumed by a dedicated CUDA kernel.
///
/// Runs *after* [`CudaSwiGluFusion`], so the trailing multiply is already the
/// tagged two-input `Mul[_cuda_silu_mul](gate, up)`. When `gate` and `up` are
/// each produced by a plain `MatMulNBits` sharing the *same* activation and both
/// match the exact validated decode shape (K=896, N=4864, block-32, fp16 scales,
/// fp16 output), the three ops collapse into a single node marked
/// [`GATE_UP_SWIGLU_FUSION_ATTR`] whose inputs are
/// `[x, W_gate, scales_gate, W_up, scales_up]`. The paired kernel reads the
/// activation once, runs both GEMVs, and writes `silu(gate)*up` directly —
/// reproducing the two-op fp16 rounding so greedy tokens stay byte-identical.
///
/// Any shape/dtype/structure mismatch leaves the separate GEMVs + tagged
/// `silu_mul` in place (the existing fallback path), so the pass never misfires.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaGateUpSwiGluFusion;

struct GateUpSwiGluPlan {
    mul_id: NodeId,
    gate_matmul_id: NodeId,
    up_matmul_id: NodeId,
    activation: ValueId,
    gate_weight: ValueId,
    gate_scales: ValueId,
    up_weight: ValueId,
    up_scales: ValueId,
    gate_out: ValueId,
    up_out: ValueId,
}

impl OptimizationPass for CudaGateUpSwiGluFusion {
    fn name(&self) -> &str {
        "CudaGateUpSwiGluFusion"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> OptimizerResult<()> {
        let mul_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.op_type == "Mul"
                    && matches!(node.domain.as_str(), "" | "ai.onnx")
                    && node.attr(SILU_MUL_FUSION_ATTR).and_then(Attribute::as_int) == Some(1)
                    && node.inputs.len() == 2
                    && node.outputs.len() == 1)
                    .then_some(id)
            })
            .collect();

        let mut plans: Vec<GateUpSwiGluPlan> = Vec::new();
        for mul_id in mul_nodes {
            if let Some(plan) = self.plan_fuse(graph, mul_id) {
                plans.push(plan);
            }
        }

        let changed = !plans.is_empty();
        for plan in plans {
            // Reuse the `Mul` node's slot (and its already-inferred `[.., N]`
            // output value) as the fused node so downstream/output binding is
            // untouched, then drop the two now-dead projection GEMVs. Their
            // outputs lose their only consumer and are GC'd by `remove_node`,
            // leaving no dangling values.
            let mut fused = graph.node(plan.gate_matmul_id).clone();
            fused.id = plan.mul_id;
            fused.inputs = vec![
                Some(plan.activation),
                Some(plan.gate_weight),
                Some(plan.gate_scales),
                Some(plan.up_weight),
                Some(plan.up_scales),
            ];
            fused.outputs = graph.node(plan.mul_id).outputs.clone();
            fused
                .attributes
                .insert(GATE_UP_SWIGLU_FUSION_ATTR.into(), Attribute::Int(1));
            graph.replace_node(plan.mul_id, fused);
            debug_assert_eq!(graph.consumers(plan.gate_out).len(), 0);
            debug_assert_eq!(graph.consumers(plan.up_out).len(), 0);
            graph.remove_node(plan.gate_matmul_id);
            graph.remove_node(plan.up_matmul_id);
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

impl CudaGateUpSwiGluFusion {
    fn plan_fuse(&self, graph: &Graph, mul_id: NodeId) -> Option<GateUpSwiGluPlan> {
        let mul = graph.try_node(mul_id)?;
        // `CudaSwiGluFusion` always emits `[gate, up]` in this order.
        let gate_out = mul.inputs[0]?;
        let up_out = mul.inputs[1]?;
        if gate_out == up_out {
            return None;
        }

        let gate_matmul_id = self.matmul_producer(graph, gate_out)?;
        let up_matmul_id = self.matmul_producer(graph, up_out)?;
        if gate_matmul_id == up_matmul_id {
            return None;
        }

        // Each projection output must feed only this multiply and must not
        // escape as a graph output.
        for out in [gate_out, up_out] {
            if graph.consumers(out).len() != 1 || graph.value(out).is_graph_output {
                return None;
            }
        }

        let gate = self.eligible_projection(graph, gate_matmul_id)?;
        let up = self.eligible_projection(graph, up_matmul_id)?;

        // Both projections must consume the *same* activation and share the
        // exact validated decode shape.
        if gate.activation != up.activation {
            return None;
        }

        Some(GateUpSwiGluPlan {
            mul_id,
            gate_matmul_id,
            up_matmul_id,
            activation: gate.activation,
            gate_weight: gate.weight,
            gate_scales: gate.scales,
            up_weight: up.weight,
            up_scales: up.scales,
            gate_out,
            up_out,
        })
    }

    /// Validate one projection `MatMulNBits` and return its `[x, W, scales]`
    /// value ids if it matches the exact fused-kernel contract.
    fn eligible_projection(&self, graph: &Graph, matmul_id: NodeId) -> Option<Projection> {
        let matmul = graph.try_node(matmul_id)?;
        // Plain A/B/scales form only: no zero-points, group index, or bias.
        let present: Vec<ValueId> = matmul.input_values().collect();
        if present.len() != 3 || matmul.inputs.iter().skip(3).any(Option::is_some) {
            return None;
        }

        let n = matmul.attr("N").and_then(Attribute::as_int)? as usize;
        let k = matmul.attr("K").and_then(Attribute::as_int)? as usize;
        let block_size = matmul.attr("block_size").and_then(Attribute::as_int)? as usize;
        let bits = matmul.attr("bits").and_then(Attribute::as_int).unwrap_or(4);
        if n != GATE_UP_SWIGLU_N
            || k != GATE_UP_SWIGLU_K
            || block_size != GATE_UP_SWIGLU_BLOCK_SIZE
            || bits != 4
        {
            return None;
        }

        let activation = matmul.inputs[0]?;
        let weight = matmul.inputs[1]?;
        let scales = matmul.inputs[2]?;

        // fp16 activation + output, fp16 scales, persistent (initializer)
        // weights/scales: the exact form the paired kernel reproduces bit-for-bit
        // and the only form that is capture-safe with a fixed device signature.
        if graph.value(activation).dtype != DataType::Float16
            || graph.value(matmul.outputs[0]).dtype != DataType::Float16
            || graph.value(scales).dtype != DataType::Float16
        {
            return None;
        }
        if !graph.initializers.contains_key(&weight) || !graph.initializers.contains_key(&scales) {
            return None;
        }

        Some(Projection {
            activation,
            weight,
            scales,
        })
    }

    fn matmul_producer(&self, graph: &Graph, value: ValueId) -> Option<NodeId> {
        let producer = graph.try_value(value)?.producer?;
        let node = graph.try_node(producer)?;
        (node.op_type == "MatMulNBits" && node.domain == MICROSOFT_DOMAIN).then_some(producer)
    }
}

struct Projection {
    activation: ValueId,
    weight: ValueId,
    scales: ValueId,
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Dim, Node, NodeId, ValueId};

    fn value(graph: &mut Graph, name: &str, dtype: DataType, width: usize) -> ValueId {
        graph.create_named_value(name, dtype, vec![Dim::Static(1), Dim::Static(width)])
    }

    fn swiglu_graph(dtype: DataType, gate_width: usize, up_width: usize) -> Graph {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);
        let gate = value(&mut graph, "gate", dtype, gate_width);
        let up = value(&mut graph, "up", dtype, up_width);
        let silu_output = value(&mut graph, "silu", dtype, gate_width);
        let output = value(&mut graph, "output", dtype, gate_width);
        graph.add_input(gate);
        graph.add_input(up);
        let mut silu = Node::new(NodeId(0), "Silu", vec![Some(gate)], vec![silu_output]);
        silu.domain = MICROSOFT_DOMAIN.into();
        graph.insert_node(silu);
        graph.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(silu_output), Some(up)],
            vec![output],
        ));
        graph.add_output(output);
        graph
    }

    #[test]
    fn fuses_equal_shape_silu_mul() {
        let mut graph = swiglu_graph(DataType::Float16, 7, 7);
        CudaSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert_eq!(graph.num_nodes(), 1);
        let fused = graph.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "Mul");
        assert_eq!(
            fused.attr(SILU_MUL_FUSION_ATTR).and_then(Attribute::as_int),
            Some(1)
        );
        assert_eq!(fused.inputs.len(), 2);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn leaves_broadcast_silu_mul_separate() {
        let mut graph = swiglu_graph(DataType::Float16, 7, 1);
        CudaSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert_eq!(graph.num_nodes(), 2);
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.attr(SILU_MUL_FUSION_ATTR).is_none())
        );
    }

    // === QKV bias fold ===

    use onnx_runtime_ir::{TensorData, WeightRef};

    fn vec1d(graph: &mut Graph, name: &str, dtype: DataType, width: usize) -> ValueId {
        graph.create_named_value(name, dtype, vec![Dim::Static(width)])
    }

    fn matmul_nbits(inputs: Vec<Option<ValueId>>, output: ValueId, k: usize, n: usize) -> Node {
        let mut node = Node::new(NodeId(0), "MatMulNBits", inputs, vec![output]);
        node.domain = MICROSOFT_DOMAIN.into();
        node.attributes.insert("K".into(), Attribute::Int(k as i64));
        node.attributes.insert("N".into(), Attribute::Int(n as i64));
        node.attributes
            .insert("block_size".into(), Attribute::Int(32));
        node.attributes.insert("bits".into(), Attribute::Int(4));
        node
    }

    /// `Add(MatMulNBits(x), bias)` with a 1-D `[N]` initializer bias. When
    /// `bias_is_initializer` is false the bias is a graph input instead.
    fn qkv_bias_graph(dtype: DataType, n: usize, bias_is_initializer: bool) -> Graph {
        let k = 896usize;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);
        let x = value(&mut graph, "x", dtype, k);
        let packed = vec1d(&mut graph, "packed", DataType::Uint8, n * (k / 32) * 16);
        let scales = vec1d(&mut graph, "scales", dtype, n * (k / 32));
        let mm_out = value(&mut graph, "mm_out", dtype, n);
        let bias = vec1d(&mut graph, "bias", dtype, n);
        let out = value(&mut graph, "out", dtype, n);
        graph.add_input(x);
        graph.set_initializer(
            packed,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Uint8,
                vec![n * (k / 32) * 16],
                vec![0u8; n * (k / 32) * 16],
            )),
        );
        graph.set_initializer(
            scales,
            WeightRef::Inline(TensorData::from_raw(
                dtype,
                vec![n * (k / 32)],
                vec![0u8; n * (k / 32) * 2],
            )),
        );
        if bias_is_initializer {
            graph.set_initializer(
                bias,
                WeightRef::Inline(TensorData::from_raw(dtype, vec![n], vec![0u8; n * 2])),
            );
        } else {
            graph.add_input(bias);
        }
        graph.insert_node(matmul_nbits(
            vec![Some(x), Some(packed), Some(scales)],
            mm_out,
            k,
            n,
        ));
        graph.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(mm_out), Some(bias)],
            vec![out],
        ));
        graph.add_output(out);
        graph
    }

    #[test]
    fn folds_qkv_bias_into_matmul_nbits() {
        let mut graph = qkv_bias_graph(DataType::Float16, 1152, true);
        CudaMatMulNBitsBiasFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert_eq!(graph.num_nodes(), 1, "the Add must be folded away");
        let fused = graph.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "MatMulNBits");
        assert_eq!(
            fused
                .attr(MATMUL_NBITS_FOLDED_BIAS_ATTR)
                .and_then(Attribute::as_int),
            Some(1)
        );
        assert_eq!(fused.inputs.len(), 6, "bias occupies input slot 5");
        assert!(fused.inputs[3].is_none() && fused.inputs[4].is_none());
        assert!(fused.inputs[5].is_some(), "bias must be wired at index 5");
        // The fused node's single output is still the (sole) graph output and
        // inherits the folded Add output's name so output binding is stable.
        let out = fused.outputs[0];
        assert_eq!(graph.outputs, vec![out]);
        assert_eq!(graph.value(out).name.as_deref(), Some("out"));
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn does_not_fold_non_initializer_bias() {
        let mut graph = qkv_bias_graph(DataType::Float16, 1152, false);
        CudaMatMulNBitsBiasFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert_eq!(graph.num_nodes(), 2, "a runtime bias must not be folded");
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.attr(MATMUL_NBITS_FOLDED_BIAS_ATTR).is_none())
        );
    }

    #[test]
    fn does_not_fold_wrong_shape_bias() {
        let mut graph = qkv_bias_graph(DataType::Float16, 1152, true);
        // Retype the bias initializer value to a 2-D shape so it no longer
        // matches the `[N]` epilogue contract.
        let bias = graph
            .values
            .iter()
            .find_map(|(id, v)| (v.name.as_deref() == Some("bias")).then_some(id))
            .unwrap();
        graph.value_mut(bias).shape = vec![Dim::Static(2), Dim::Static(1152)];
        CudaMatMulNBitsBiasFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert_eq!(graph.num_nodes(), 2, "a non-[N] bias must not be folded");
    }

    #[test]
    fn does_not_fold_when_matmul_output_is_shared() {
        let mut graph = qkv_bias_graph(DataType::Float16, 1152, true);
        // Add a second consumer of the MatMulNBits output so the GEMV result
        // escapes beyond the Add; folding would drop that observable value.
        let mm_out = graph
            .values
            .iter()
            .find_map(|(id, v)| (v.name.as_deref() == Some("mm_out")).then_some(id))
            .unwrap();
        let sink = value(&mut graph, "sink", DataType::Float16, 1152);
        graph.insert_node(Node::new(NodeId(0), "Neg", vec![Some(mm_out)], vec![sink]));
        graph.add_output(sink);
        CudaMatMulNBitsBiasFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.attr(MATMUL_NBITS_FOLDED_BIAS_ATTR).is_none()),
            "a shared GEMV output must not be folded"
        );
    }

    // === Paired gate/up + SwiGLU fusion ===

    fn projection(graph: &mut Graph, tag: &str, x: ValueId, k: usize, n: usize) -> ValueId {
        let packed = vec1d(
            graph,
            &format!("{tag}_packed"),
            DataType::Uint8,
            n * (k / 32) * 16,
        );
        let scales = vec1d(
            graph,
            &format!("{tag}_scales"),
            DataType::Float16,
            n * (k / 32),
        );
        let out = value(graph, &format!("{tag}_out"), DataType::Float16, n);
        graph.set_initializer(
            packed,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Uint8,
                vec![n * (k / 32) * 16],
                vec![0u8; n * (k / 32) * 16],
            )),
        );
        graph.set_initializer(
            scales,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float16,
                vec![n * (k / 32)],
                vec![0u8; n * (k / 32) * 2],
            )),
        );
        graph.insert_node(matmul_nbits(
            vec![Some(x), Some(packed), Some(scales)],
            out,
            k,
            n,
        ));
        out
    }

    /// A post-`CudaSwiGluFusion` graph: two `MatMulNBits` projections (gate, up)
    /// feeding the tagged `Mul[_cuda_silu_mul](gate, up)`. When `shared` is false
    /// the up projection consumes a *different* activation.
    fn gate_up_graph(k: usize, n: usize, shared: bool) -> Graph {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);
        let x = value(&mut graph, "x", DataType::Float16, k);
        graph.add_input(x);
        let up_x = if shared {
            x
        } else {
            let x2 = value(&mut graph, "x2", DataType::Float16, k);
            graph.add_input(x2);
            x2
        };
        let gate_out = projection(&mut graph, "gate", x, k, n);
        let up_out = projection(&mut graph, "up", up_x, k, n);
        let out = value(&mut graph, "output", DataType::Float16, n);
        let mut mul = Node::new(
            NodeId(0),
            "Mul",
            vec![Some(gate_out), Some(up_out)],
            vec![out],
        );
        mul.attributes
            .insert(SILU_MUL_FUSION_ATTR.into(), Attribute::Int(1));
        graph.insert_node(mul);
        graph.add_output(out);
        graph
    }

    #[test]
    fn fuses_paired_gate_up_swiglu() {
        let mut graph = gate_up_graph(GATE_UP_SWIGLU_K, GATE_UP_SWIGLU_N, true);
        CudaGateUpSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert_eq!(
            graph.num_nodes(),
            1,
            "both projections and the Mul collapse into one node"
        );
        let fused = graph.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "MatMulNBits");
        assert_eq!(fused.domain, MICROSOFT_DOMAIN);
        assert_eq!(
            fused
                .attr(GATE_UP_SWIGLU_FUSION_ATTR)
                .and_then(Attribute::as_int),
            Some(1)
        );
        assert!(
            fused.attr(SILU_MUL_FUSION_ATTR).is_none(),
            "the silu_mul marker must not leak onto the fused MatMulNBits"
        );
        assert_eq!(
            fused.inputs.len(),
            5,
            "inputs are [x, W_gate, scales_gate, W_up, scales_up]"
        );
        assert!(fused.inputs.iter().all(Option::is_some));
        assert_eq!(
            fused.attr("N").and_then(Attribute::as_int),
            Some(GATE_UP_SWIGLU_N as i64)
        );
        // The fused node keeps the Mul's output value, so downstream binding is
        // stable.
        let out = fused.outputs[0];
        assert_eq!(graph.outputs, vec![out]);
        assert_eq!(graph.value(out).name.as_deref(), Some("output"));
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn does_not_fuse_when_activation_differs() {
        let mut graph = gate_up_graph(GATE_UP_SWIGLU_K, GATE_UP_SWIGLU_N, false);
        CudaGateUpSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert_eq!(
            graph.num_nodes(),
            3,
            "projections on different activations must not be paired"
        );
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.attr(GATE_UP_SWIGLU_FUSION_ATTR).is_none())
        );
    }

    #[test]
    fn does_not_fuse_unvalidated_shape() {
        // Correct structure but the wrong K/N, so the paired kernel (validated
        // only for K=896,N=4864) must not claim it.
        let mut graph = gate_up_graph(512, 2048, true);
        CudaGateUpSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert_eq!(graph.num_nodes(), 3, "off-shape projections stay separate");
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.attr(GATE_UP_SWIGLU_FUSION_ATTR).is_none())
        );
    }

    #[test]
    fn does_not_fuse_untagged_mul() {
        let mut graph = gate_up_graph(GATE_UP_SWIGLU_K, GATE_UP_SWIGLU_N, true);
        // Strip the silu_mul marker: without it the multiply is an ordinary
        // elementwise op, not a SwiGLU, and must be left alone.
        let mul_id = graph
            .nodes
            .iter()
            .find_map(|(id, node)| (node.op_type == "Mul").then_some(id))
            .unwrap();
        graph
            .node_mut(mul_id)
            .attributes
            .remove(SILU_MUL_FUSION_ATTR);
        CudaGateUpSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert_eq!(graph.num_nodes(), 3, "an untagged Mul must not be fused");
    }

    #[test]
    fn gate_up_pass_chains_after_swiglu_fusion() {
        // End-to-end through the real CUDA pass list: Sigmoid+Mul... is already
        // Silu here, so CudaSwiGluFusion tags the Mul and CudaGateUpSwiGluFusion
        // collapses the pair.
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);
        let x = value(&mut graph, "x", DataType::Float16, GATE_UP_SWIGLU_K);
        graph.add_input(x);
        let gate_out = projection(&mut graph, "gate", x, GATE_UP_SWIGLU_K, GATE_UP_SWIGLU_N);
        let up_out = projection(&mut graph, "up", x, GATE_UP_SWIGLU_K, GATE_UP_SWIGLU_N);
        let silu_out = value(&mut graph, "silu", DataType::Float16, GATE_UP_SWIGLU_N);
        let out = value(&mut graph, "output", DataType::Float16, GATE_UP_SWIGLU_N);
        let mut silu = Node::new(NodeId(0), "Silu", vec![Some(gate_out)], vec![silu_out]);
        silu.domain = MICROSOFT_DOMAIN.into();
        graph.insert_node(silu);
        graph.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(silu_out), Some(up_out)],
            vec![out],
        ));
        graph.add_output(out);

        for pass in cuda_optimization_passes() {
            pass.run(&mut graph, &PassContext::new()).unwrap();
        }

        assert_eq!(graph.num_nodes(), 1);
        let fused = graph.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "MatMulNBits");
        assert_eq!(
            fused
                .attr(GATE_UP_SWIGLU_FUSION_ATTR)
                .and_then(Attribute::as_int),
            Some(1)
        );
        assert!(graph.validate().is_ok());
    }
}
