use onnx_runtime_ir::{
    static_shape, Attribute, DataType, Dim, Graph, Node, NodeId, TensorData, ValueId, WeightRef,
};
use onnx_runtime_optimizer::{
    OptimizationPass, OptimizerError, PassContext, Result as OptimizerResult,
};

const PROJECTION_FUSION_ENV: &str = "ONNX_GENAI_PROJECTION_FUSION";
const MICROSOFT_DOMAIN: &str = "com.microsoft";

/// CPU-only gate/up `MatMulNBits` fusion.
///
/// The environment gate is captured once when the pass is constructed, so a
/// session's optimization policy cannot change midway through graph rewriting.
pub struct ProjectionFusion {
    enabled: bool,
}

impl ProjectionFusion {
    pub fn new() -> Self {
        Self {
            enabled: std::env::var_os(PROJECTION_FUSION_ENV).is_some_and(|value| value == "1"),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for ProjectionFusion {
    fn default() -> Self {
        Self::new()
    }
}

/// The ordered CPU EP optimization registry.
pub fn cpu_optimization_passes() -> Vec<Box<dyn OptimizationPass>> {
    let mut passes: Vec<Box<dyn OptimizationPass>> = Vec::new();
    let projection_fusion = ProjectionFusion::new();
    if projection_fusion.enabled() {
        passes.push(Box::new(projection_fusion));
    }
    // Always-on, byte-identical bias fold: `Add(MatMulNBits, [N]-bias)` becomes
    // the `MatMulNBits` optional bias input, which the kernel adds inside the
    // MLAS GEMV epilogue instead of paying for a standalone element-wise `Add`.
    passes.push(Box::new(MatMulNBitsBiasFusion::new()));
    // Always-on Conv+BatchNormalization(+Relu) fold: pushes the inference-time BN
    // affine transform back into the Conv weight/bias and folds a trailing Relu
    // into the Conv activation epilogue, eliminating standalone BN/Relu kernels.
    passes.push(Box::new(ConvBatchNormActivationFusion::new()));
    passes
}

impl OptimizationPass for ProjectionFusion {
    fn name(&self) -> &str {
        "CpuProjectionFusion"
    }

    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> OptimizerResult<()> {
        if !self.enabled {
            return Ok(());
        }

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
            let Some(candidate) = GateUpCandidate::match_from_silu(graph, silu_id, ctx) else {
                continue;
            };
            candidate.apply(graph);
            changed = true;
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

/// Folds a standalone bias `Add` into the preceding `MatMulNBits`.
///
/// Recognizes the generic pattern `Add(MatMulNBits(A, ...), bias)` where `bias`
/// is a rank-1 `[N]` tensor broadcast over the `MatMulNBits` output's last (`N`)
/// dimension, and rewrites it to `MatMulNBits(A, ..., bias)` using the op's
/// optional bias input (ONNX contrib input index 5). The kernel adds the bias
/// inside the MLAS GEMV epilogue, so the result is byte-identical to running the
/// `MatMul` and `Add` separately while eliminating a full-tensor read/write and a
/// separate kernel launch per decode step (`docs/ORT2.md` §15.1, RULE 2.1).
///
/// The match is purely structural — it never inspects model identity, and it
/// falls back cleanly (no rewrite) whenever the shapes, dtypes, or fan-out do
/// not fit the pattern.
pub struct MatMulNBitsBiasFusion;

impl MatMulNBitsBiasFusion {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MatMulNBitsBiasFusion {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for MatMulNBitsBiasFusion {
    fn name(&self) -> &str {
        "CpuMatMulNBitsBiasFusion"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> OptimizerResult<()> {
        let add_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.op_type == "Add"
                    && matches!(node.domain.as_str(), "" | "ai.onnx")
                    && node.inputs.len() == 2
                    && node.outputs.len() == 1)
                    .then_some(id)
            })
            .collect();

        let mut changed = false;
        for add_id in add_nodes {
            if let Some(fusion) = MatMulNBitsBias::match_from_add(graph, add_id) {
                fusion.apply(graph);
                changed = true;
            }
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

struct MatMulNBitsBias {
    matmul_id: NodeId,
    add_id: NodeId,
    matmul_output: ValueId,
    add_output: ValueId,
    bias: ValueId,
}

impl MatMulNBitsBias {
    fn match_from_add(graph: &Graph, add_id: NodeId) -> Option<Self> {
        let add = graph.try_node(add_id)?;
        let lhs = add.inputs[0]?;
        let rhs = add.inputs[1]?;
        let add_output = add.outputs[0];

        // Try each operand as the MatMulNBits side; the other is the bias.
        for (matmul_output, bias) in [(lhs, rhs), (rhs, lhs)] {
            if matmul_output == bias {
                continue;
            }
            let Some(matmul_id) = graph.value(matmul_output).producer else {
                continue;
            };
            if !Self::matmul_is_biasable(graph, matmul_id, matmul_output, add_id) {
                continue;
            }
            let n = match graph.value(matmul_output).shape.last() {
                Some(&Dim::Static(n)) => n,
                _ => continue,
            };
            if !Self::bias_is_row_vector(graph, bias, n) {
                continue;
            }
            return Some(Self {
                matmul_id,
                add_id,
                matmul_output,
                add_output,
                bias,
            });
        }
        None
    }

    /// The producer is a bias-free `MatMulNBits` whose sole consumer is this
    /// `Add` and whose output is not exposed as a graph output.
    fn matmul_is_biasable(
        graph: &Graph,
        matmul_id: NodeId,
        matmul_output: ValueId,
        add_id: NodeId,
    ) -> bool {
        let Some(matmul) = graph.try_node(matmul_id) else {
            return false;
        };
        if matmul.op_type != "MatMulNBits"
            || matmul.domain != MICROSOFT_DOMAIN
            || matmul.outputs.len() != 1
            || matmul.outputs[0] != matmul_output
        {
            return false;
        }
        // A bias input (contrib index 5) already present means nothing to fold.
        if matmul.inputs.get(5).copied().flatten().is_some() {
            return false;
        }
        // The intermediate MatMulNBits result must be private to this Add so it
        // can be dropped once the bias is folded in.
        !graph.outputs.contains(&matmul_output) && graph.consumers(matmul_output) == [add_id]
    }

    /// The bias must be a rank-1 `[N]` float tensor so it broadcasts over the
    /// `MatMulNBits` output's last dimension exactly as the kernel's bias add.
    fn bias_is_row_vector(graph: &Graph, bias: ValueId, n: usize) -> bool {
        let value = graph.value(bias);
        matches!(
            value.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) && value.shape.as_slice() == [Dim::Static(n)]
    }

    fn apply(self, graph: &mut Graph) {
        let mut fused = graph.node(self.matmul_id).clone();
        // Grow the input list to the contrib bias slot, leaving the optional
        // zero_points / g_idx slots disconnected when the source op omitted them.
        if fused.inputs.len() < 6 {
            fused.inputs.resize(6, None);
        }
        fused.inputs[5] = Some(self.bias);
        graph.replace_node(self.matmul_id, fused);

        // Route the Add's consumers (and any graph-output slot) onto the fused
        // MatMulNBits output, then drop the now-dead Add and its output value.
        graph.replace_all_uses(self.add_output, self.matmul_output);
        graph.remove_node(self.add_id);
    }
}

struct Projection {
    node_id: NodeId,
    activation: ValueId,
    output: ValueId,
    weight: ValueId,
    scales: ValueId,
    k: usize,
    n: usize,
    bits: usize,
    block_size: usize,
    accuracy_level: i64,
    weight_bytes: Vec<u8>,
    scale_bytes: Vec<u8>,
}

impl Projection {
    fn parse(graph: &Graph, node_id: NodeId, ctx: &PassContext) -> Option<Self> {
        let node = graph.try_node(node_id)?;
        if node.op_type != "MatMulNBits"
            || node.domain != MICROSOFT_DOMAIN
            || node.outputs.len() != 1
            || !(3..=6).contains(&node.inputs.len())
            || node.inputs.iter().skip(3).any(Option::is_some)
        {
            return None;
        }
        const ALLOWED_ATTRIBUTES: &[&str] = &[
            "K",
            "N",
            "bits",
            "block_size",
            "accuracy_level",
            "weight_prepacked",
        ];
        if node
            .attributes
            .keys()
            .any(|attribute| !ALLOWED_ATTRIBUTES.contains(&attribute.as_str()))
        {
            return None;
        }

        let activation = node.inputs[0]?;
        let weight = node.inputs[1]?;
        let scales = node.inputs[2]?;
        let output = node.outputs[0];
        let k = positive_attr(node, "K")?;
        let n = positive_attr(node, "N")?;
        let bits = optional_nonnegative_attr(node, "bits", 4)?;
        let block_size = positive_attr(node, "block_size")?;
        let accuracy_level = node
            .attr("accuracy_level")
            .and_then(Attribute::as_int)
            .unwrap_or(0);
        let weight_prepacked = optional_nonnegative_attr(node, "weight_prepacked", 0)?;
        if bits != 4
            || weight_prepacked != 0
            || block_size < 16
            || !block_size.is_power_of_two()
            || graph.value(activation).dtype != DataType::Float32
            || graph.value(activation).shape.last() != Some(&Dim::Static(k))
            || graph.value(output).dtype != DataType::Float32
        {
            return None;
        }

        let output_shape = &graph.value(output).shape;
        if output_shape.last() != Some(&Dim::Static(n)) {
            return None;
        }

        let k_blocks = k.div_ceil(block_size);
        let packed_block_bytes = block_size.checked_mul(bits)?.checked_div(8)?;
        let expected_weight_bytes = n.checked_mul(k_blocks)?.checked_mul(packed_block_bytes)?;
        let expected_scale_values = n.checked_mul(k_blocks)?;
        let expected_scale_bytes =
            expected_scale_values.checked_mul(DataType::Float32.byte_size())?;

        let weight_ref = graph.initializers.get(&weight)?;
        if weight_ref.dtype() != DataType::Uint8
            || weight_ref.dims() != [n, k_blocks, packed_block_bytes]
        {
            return None;
        }
        let scale_ref = graph.initializers.get(&scales)?;
        if scale_ref.dtype() != DataType::Float32
            || !(scale_ref.dims() == [n, k_blocks] || scale_ref.dims() == [expected_scale_values])
        {
            return None;
        }

        let weight_bytes = ctx.initializer_bytes(weight_ref)?;
        let scale_bytes = ctx.initializer_bytes(scale_ref)?;
        if weight_bytes.len() != expected_weight_bytes || scale_bytes.len() != expected_scale_bytes
        {
            return None;
        }

        Some(Self {
            node_id,
            activation,
            output,
            weight,
            scales,
            k,
            n,
            bits,
            block_size,
            accuracy_level,
            weight_bytes: weight_bytes.to_vec(),
            scale_bytes: scale_bytes.to_vec(),
        })
    }

    fn compatible_with(&self, other: &Self, graph: &Graph) -> bool {
        self.activation == other.activation
            && self.k == other.k
            && self.bits == other.bits
            && self.block_size == other.block_size
            && self.accuracy_level == other.accuracy_level
            && graph.value(self.output).shape[..graph.value(self.output).shape.len() - 1]
                == graph.value(other.output).shape[..graph.value(other.output).shape.len() - 1]
    }
}

struct GateUpCandidate {
    gate: Projection,
    up: Projection,
    total_n: usize,
    fused_weight: Vec<u8>,
    fused_scales: Vec<u8>,
}

impl GateUpCandidate {
    fn match_from_silu(graph: &Graph, silu_id: NodeId, ctx: &PassContext) -> Option<Self> {
        let silu = graph.try_node(silu_id)?;
        let gate_output = silu.inputs[0]?;
        let silu_output = silu.outputs[0];
        if graph.outputs.contains(&gate_output)
            || graph.outputs.contains(&silu_output)
            || graph.consumers(gate_output) != [silu_id]
        {
            return None;
        }

        let silu_consumers = graph.consumers(silu_output);
        if silu_consumers.len() != 1 {
            return None;
        }
        let mul_id = silu_consumers[0];
        let mul = graph.try_node(mul_id)?;
        if mul.op_type != "Mul"
            || !matches!(mul.domain.as_str(), "" | "ai.onnx")
            || mul.inputs.len() != 2
            || mul.outputs.len() != 1
        {
            return None;
        }
        let up_output = if mul.inputs[0] == Some(silu_output) {
            mul.inputs[1]?
        } else if mul.inputs[1] == Some(silu_output) {
            mul.inputs[0]?
        } else {
            return None;
        };
        if up_output == gate_output
            || graph.outputs.contains(&up_output)
            || graph.consumers(up_output) != [mul_id]
        {
            return None;
        }

        let gate_id = graph.value(gate_output).producer?;
        let up_id = graph.value(up_output).producer?;
        if gate_id == up_id {
            return None;
        }
        let mut gate = Projection::parse(graph, gate_id, ctx)?;
        let mut up = Projection::parse(graph, up_id, ctx)?;
        if gate.output != gate_output || up.output != up_output || !gate.compatible_with(&up, graph)
        {
            return None;
        }
        let total_n = gate.n.checked_add(up.n)?;
        i64::try_from(total_n).ok()?;
        let fused_weight_len =
            checked_combined_initializer_len(gate.weight_bytes.len(), up.weight_bytes.len())?;
        let fused_scale_len =
            checked_combined_initializer_len(gate.scale_bytes.len(), up.scale_bytes.len())?;
        let fused_weight = concatenate_initializer_bytes(
            std::mem::take(&mut gate.weight_bytes),
            std::mem::take(&mut up.weight_bytes),
            fused_weight_len,
        )?;
        let fused_scales = concatenate_initializer_bytes(
            std::mem::take(&mut gate.scale_bytes),
            std::mem::take(&mut up.scale_bytes),
            fused_scale_len,
        )?;

        Some(Self {
            gate,
            up,
            total_n,
            fused_weight,
            fused_scales,
        })
    }

    fn apply(self, graph: &mut Graph) {
        let first_id = self.gate.node_id.0;
        let k_blocks = self.gate.k.div_ceil(self.gate.block_size);
        let packed_block_bytes = self.gate.block_size * self.gate.bits / 8;
        let total_n = self.total_n;

        let fused_weight = self.fused_weight;
        let fused_scales = self.fused_scales;

        let weight_name = format!("__nxrt_fused_projection_{first_id}_weight");
        let fused_weight_value = graph.create_named_value(
            weight_name,
            DataType::Uint8,
            static_shape([total_n, k_blocks, packed_block_bytes]),
        );
        graph.set_initializer(
            fused_weight_value,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Uint8,
                vec![total_n, k_blocks, packed_block_bytes],
                fused_weight,
            )),
        );

        let scales_name = format!("__nxrt_fused_projection_{first_id}_scales");
        let fused_scale_value = graph.create_named_value(
            scales_name,
            DataType::Float32,
            static_shape([total_n, k_blocks]),
        );
        graph.set_initializer(
            fused_scale_value,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![total_n, k_blocks],
                fused_scales,
            )),
        );

        let split_name = format!("__nxrt_fused_projection_{first_id}_split");
        let split_value = graph.create_named_value(split_name, DataType::Int64, static_shape([2]));
        let mut split_bytes = Vec::with_capacity(2 * std::mem::size_of::<i64>());
        split_bytes.extend_from_slice(&(self.gate.n as i64).to_le_bytes());
        split_bytes.extend_from_slice(&(self.up.n as i64).to_le_bytes());
        graph.set_initializer(
            split_value,
            WeightRef::Inline(TensorData::from_raw(DataType::Int64, vec![2], split_bytes)),
        );

        let mut fused_shape = graph.value(self.gate.output).shape.clone();
        *fused_shape
            .last_mut()
            .expect("MatMulNBits output has a last dimension") = Dim::Static(total_n);
        let fused_output = graph.create_named_value(
            format!("__nxrt_fused_projection_{first_id}_output"),
            DataType::Float32,
            fused_shape,
        );

        let mut fused_node = graph.node(self.gate.node_id).clone();
        fused_node.name = format!("fused_projection_gate_up_{first_id}");
        fused_node.inputs = vec![
            Some(self.gate.activation),
            Some(fused_weight_value),
            Some(fused_scale_value),
        ];
        fused_node.outputs = vec![fused_output];
        fused_node
            .attributes
            .insert("N".to_string(), Attribute::Int(total_n as i64));

        graph.remove_node(self.gate.node_id);
        graph.remove_node(self.up.node_id);
        graph.insert_node(fused_node);

        let mut split = Node::new(
            NodeId(0),
            "Split",
            vec![Some(fused_output), Some(split_value)],
            vec![self.gate.output, self.up.output],
        );
        split.name = format!("fused_projection_split_{first_id}");
        split
            .attributes
            .insert("axis".to_string(), Attribute::Int(-1));
        graph.insert_node(split);

        remove_orphan_initializer(graph, self.gate.weight);
        remove_orphan_initializer(graph, self.gate.scales);
        remove_orphan_initializer(graph, self.up.weight);
        remove_orphan_initializer(graph, self.up.scales);
    }
}

/// Folds `Conv -> BatchNormalization` (and an optional trailing `Relu`) into a
/// single `Conv` with adjusted weight/bias initializers and a fused activation.
///
/// Batch normalization at inference is an affine per-output-channel transform
/// `y = a[c]*x + b[c]` with `a[c] = scale[c]/sqrt(var[c]+eps)` and
/// `b[c] = beta[c] - a[c]*mean[c]`. Because a convolution is linear in its
/// weights and bias, that transform can be pushed back into the Conv:
/// `W'[c] = a[c]*W[c]` and `B'[c] = a[c]*(B[c]-mean[c]) + beta[c]`. ORT folds BN
/// this way in its `nchwc_transformer`, eliminating a full-tensor BN kernel per
/// convolution (which profiling showed dominates ResNet/MobileNet inference on
/// the native EP). A trailing `Relu` is folded into the Conv's `activation`
/// attribute, which the MLAS NCHWc epilogue (and the im2col fallback) applies for
/// free.
///
/// The match is purely structural and only fires when BN's scale/B/mean/var (and
/// the Conv weight/bias) are constant `Float32` initializers, so it is a general
/// graph rewrite with no model-specific knowledge.
pub struct ConvBatchNormActivationFusion;

impl ConvBatchNormActivationFusion {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ConvBatchNormActivationFusion {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for ConvBatchNormActivationFusion {
    fn name(&self) -> &str {
        "CpuConvBatchNormActivationFusion"
    }

    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> OptimizerResult<()> {
        let bn_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.op_type == "BatchNormalization"
                    && matches!(node.domain.as_str(), "" | "ai.onnx"))
                .then_some(id)
            })
            .collect();

        let mut changed = false;
        for bn_id in bn_nodes {
            if let Some(fusion) = ConvBnFusion::match_from_bn(graph, bn_id, ctx) {
                fusion.apply(graph);
                changed = true;
            }
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

struct ConvBnFusion {
    conv_id: NodeId,
    bn_id: NodeId,
    /// The shared value produced by Conv and consumed by BN.
    conv_output: ValueId,
    bn_output: ValueId,
    old_weight: ValueId,
    old_bias: Option<ValueId>,
    bn_params: Vec<ValueId>,
    weight_dims: Vec<usize>,
    new_weight: Vec<f32>,
    new_bias: Vec<f32>,
    /// `(relu_node, relu_output)` when a trailing `Relu` is folded in.
    relu: Option<(NodeId, ValueId)>,
}

impl ConvBnFusion {
    fn match_from_bn(graph: &Graph, bn_id: NodeId, ctx: &PassContext) -> Option<Self> {
        let bn = graph.try_node(bn_id)?;
        // Inference BatchNormalization: X, scale, B, mean, var -> Y only. Skip the
        // training form (which also emits running-stat outputs).
        if bn.inputs.len() != 5 || bn.outputs.len() != 1 {
            return None;
        }
        let conv_output = bn.inputs[0]?;
        let bn_output = bn.outputs[0];

        let conv_id = graph.value(conv_output).producer?;
        let conv = graph.try_node(conv_id)?;
        if conv.op_type != "Conv"
            || !matches!(conv.domain.as_str(), "" | "ai.onnx")
            || conv.outputs.len() != 1
            || conv.outputs[0] != conv_output
            || conv.attr("activation").is_some()
        {
            return None;
        }
        // The Conv result must be private to this BN so it can be rewritten away.
        if graph.outputs.contains(&conv_output) || graph.consumers(conv_output) != [bn_id] {
            return None;
        }

        let weight = conv.inputs.get(1).copied().flatten()?;
        let weight_ref = graph.initializers.get(&weight)?;
        if weight_ref.dtype() != DataType::Float32 || weight_ref.dims().len() != 4 {
            return None;
        }
        let weight_dims = weight_ref.dims().to_vec();
        let out_ch = weight_dims[0];
        if out_ch == 0 {
            return None;
        }
        let weight_vals = bytes_to_f32(ctx.initializer_bytes(weight_ref)?)?;
        if !weight_vals.len().is_multiple_of(out_ch) {
            return None;
        }

        let old_bias = conv.inputs.get(2).copied().flatten();
        let bias_vals = match old_bias {
            Some(bias) => {
                let bias_ref = graph.initializers.get(&bias)?;
                if bias_ref.dtype() != DataType::Float32 || bias_ref.dims() != [out_ch] {
                    return None;
                }
                bytes_to_f32(ctx.initializer_bytes(bias_ref)?)?
            }
            None => vec![0.0f32; out_ch],
        };

        // scale, B (beta), mean, var — all per-output-channel constants.
        let mut bn_params: Vec<ValueId> = Vec::with_capacity(4);
        let mut params: Vec<Vec<f32>> = Vec::with_capacity(4);
        for slot in 0..4 {
            let value = bn.inputs[slot + 1]?;
            let param_ref = graph.initializers.get(&value)?;
            if param_ref.dtype() != DataType::Float32 || param_ref.dims() != [out_ch] {
                return None;
            }
            bn_params.push(value);
            params.push(bytes_to_f32(ctx.initializer_bytes(param_ref)?)?);
        }
        let (scale, beta, mean, var) = (&params[0], &params[1], &params[2], &params[3]);
        let epsilon = bn
            .attr("epsilon")
            .and_then(Attribute::as_float)
            .unwrap_or(1e-5);

        let mut a = vec![0.0f32; out_ch];
        for c in 0..out_ch {
            let denom = (var[c] + epsilon).sqrt();
            if !denom.is_finite() || denom <= 0.0 {
                return None;
            }
            a[c] = scale[c] / denom;
        }

        let per_filter = weight_vals.len() / out_ch;
        let mut new_weight = weight_vals;
        for c in 0..out_ch {
            let factor = a[c];
            for value in &mut new_weight[c * per_filter..(c + 1) * per_filter] {
                *value *= factor;
            }
        }
        let mut new_bias = vec![0.0f32; out_ch];
        for c in 0..out_ch {
            new_bias[c] = (bias_vals[c] - mean[c]) * a[c] + beta[c];
        }

        // Optionally fold a trailing Relu when BN feeds exactly one Relu.
        let relu = (!graph.outputs.contains(&bn_output))
            .then(|| {
                let consumers = graph.consumers(bn_output);
                let relu_id = *consumers.first()?;
                if consumers.len() != 1 {
                    return None;
                }
                let relu = graph.try_node(relu_id)?;
                (relu.op_type == "Relu"
                    && matches!(relu.domain.as_str(), "" | "ai.onnx")
                    && relu.inputs.len() == 1
                    && relu.outputs.len() == 1)
                    .then_some((relu_id, relu.outputs[0]))
            })
            .flatten();

        Some(Self {
            conv_id,
            bn_id,
            conv_output,
            bn_output,
            old_weight: weight,
            old_bias,
            bn_params,
            weight_dims,
            new_weight,
            new_bias,
            relu,
        })
    }

    fn apply(self, graph: &mut Graph) {
        let conv_index = self.conv_id.0;
        let out_ch = self.weight_dims[0];

        let weight_value = graph.create_named_value(
            format!("__nxrt_convbn_{conv_index}_weight"),
            DataType::Float32,
            static_shape(self.weight_dims.iter().copied()),
        );
        graph.set_initializer(
            weight_value,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                self.weight_dims.clone(),
                f32_to_bytes(&self.new_weight),
            )),
        );

        let bias_value = graph.create_named_value(
            format!("__nxrt_convbn_{conv_index}_bias"),
            DataType::Float32,
            static_shape([out_ch]),
        );
        graph.set_initializer(
            bias_value,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![out_ch],
                f32_to_bytes(&self.new_bias),
            )),
        );

        let mut conv = graph.node(self.conv_id).clone();
        if conv.inputs.len() < 3 {
            conv.inputs.resize(3, None);
        }
        conv.inputs[1] = Some(weight_value);
        conv.inputs[2] = Some(bias_value);
        if self.relu.is_some() {
            conv.attributes
                .insert("activation".into(), Attribute::String(b"Relu".to_vec()));
        }
        graph.replace_node(self.conv_id, conv);

        // Route BN's (and any trailing Relu's) consumers back onto the Conv
        // output, then drop the now-dead BN / Relu nodes.
        graph.replace_all_uses(self.bn_output, self.conv_output);
        graph.remove_node(self.bn_id);
        if let Some((relu_id, relu_output)) = self.relu {
            graph.replace_all_uses(relu_output, self.conv_output);
            graph.remove_node(relu_id);
        }

        // Drop initializers left orphaned by the rewrite.
        remove_orphan_initializer(graph, self.old_weight);
        if let Some(bias) = self.old_bias {
            remove_orphan_initializer(graph, bias);
        }
        for param in self.bn_params {
            remove_orphan_initializer(graph, param);
        }
    }
}

/// Reinterpret a little-endian `Float32` initializer blob as `f32` values.
fn bytes_to_f32(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}

/// Serialize `f32` values into a little-endian byte blob for an initializer.
fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn positive_attr(node: &Node, name: &str) -> Option<usize> {
    usize::try_from(node.attr(name)?.as_int()?)
        .ok()
        .filter(|&v| v > 0)
}

fn optional_nonnegative_attr(node: &Node, name: &str, default: usize) -> Option<usize> {
    match node.attr(name) {
        Some(value) => usize::try_from(value.as_int()?).ok(),
        None => Some(default),
    }
}

fn checked_combined_initializer_len(first: usize, second: usize) -> Option<usize> {
    first
        .checked_add(second)
        .filter(|&combined| combined <= isize::MAX as usize)
}

fn concatenate_initializer_bytes(
    mut first: Vec<u8>,
    second: Vec<u8>,
    combined_len: usize,
) -> Option<Vec<u8>> {
    first.try_reserve_exact(second.len()).ok()?;
    first.extend_from_slice(&second);
    debug_assert_eq!(first.len(), combined_len);
    Some(first)
}

fn remove_orphan_initializer(graph: &mut Graph, value: ValueId) {
    if !graph.has_uses(value) && !graph.inputs.contains(&value) && !graph.outputs.contains(&value) {
        graph.initializers.remove(&value);
    }
}

#[cfg(test)]
mod tests {
    use super::checked_combined_initializer_len;
    use super::{MatMulNBitsBiasFusion, MICROSOFT_DOMAIN};
    use onnx_runtime_ir::{static_shape, Attribute, DataType, Dim, Graph, Node, NodeId, ValueId};
    use onnx_runtime_optimizer::{OptimizationPass, PassContext};

    #[test]
    fn fused_initializer_capacity_rejects_overflow_and_isize_excess() {
        assert_eq!(checked_combined_initializer_len(usize::MAX, 1), None);
        assert_eq!(
            checked_combined_initializer_len(isize::MAX as usize, 1),
            None
        );
        assert_eq!(
            checked_combined_initializer_len(0, isize::MAX as usize),
            Some(isize::MAX as usize)
        );
    }

    const K: usize = 8;
    const N: usize = 4;

    /// Builds `add = Add(MatMulNBits(a, w, s), bias)` and returns the graph plus
    /// the MatMulNBits node id and the bias value id. `bias_shape` lets tests
    /// exercise the row-vector requirement; `swap_add_inputs` puts the bias on
    /// the first Add operand to check operand-order independence.
    fn matmul_bias_graph(bias_shape: &[Dim], swap_add_inputs: bool) -> (Graph, NodeId, ValueId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.to_string(), 1);

        let a = graph.create_named_value("a", DataType::Float32, static_shape([1, K]));
        let weight = graph.create_named_value("w", DataType::Uint8, static_shape([N, 1, 4]));
        let scales = graph.create_named_value("s", DataType::Float32, static_shape([N, 1]));
        let bias = graph.create_named_value("bias", DataType::Float32, bias_shape.to_vec());
        for value in [a, weight, scales, bias] {
            graph.add_input(value);
        }

        let mm_out = graph.create_named_value("mm", DataType::Float32, static_shape([1, N]));
        let mut mm = Node::new(
            NodeId(0),
            "MatMulNBits",
            vec![Some(a), Some(weight), Some(scales)],
            vec![mm_out],
        );
        mm.domain = MICROSOFT_DOMAIN.to_string();
        mm.attributes.insert("K".into(), Attribute::Int(K as i64));
        mm.attributes.insert("N".into(), Attribute::Int(N as i64));
        mm.attributes.insert("bits".into(), Attribute::Int(4));
        mm.attributes
            .insert("block_size".into(), Attribute::Int(K as i64));
        let mm_id = graph.insert_node(mm);

        let add_out = graph.create_named_value("y", DataType::Float32, static_shape([1, N]));
        let add_inputs = if swap_add_inputs {
            vec![Some(bias), Some(mm_out)]
        } else {
            vec![Some(mm_out), Some(bias)]
        };
        let add = Node::new(NodeId(0), "Add", add_inputs, vec![add_out]);
        graph.insert_node(add);
        graph.add_output(add_out);

        (graph, mm_id, bias)
    }

    fn count(graph: &Graph, op_type: &str) -> usize {
        graph
            .nodes
            .values()
            .filter(|node| node.op_type == op_type)
            .count()
    }

    fn run_pass(graph: &mut Graph) {
        MatMulNBitsBiasFusion::new()
            .run(graph, &PassContext::new())
            .expect("bias fusion pass");
    }

    #[test]
    fn folds_bias_add_into_matmul_nbits() {
        let (mut graph, mm_id, bias) = matmul_bias_graph(&static_shape([N]), false);
        run_pass(&mut graph);

        assert_eq!(count(&graph, "Add"), 0, "standalone Add should be folded");
        assert_eq!(count(&graph, "MatMulNBits"), 1);
        let mm = graph.node(mm_id);
        assert_eq!(mm.inputs.len(), 6, "bias occupies contrib input slot 5");
        assert_eq!(mm.inputs[3], None);
        assert_eq!(mm.inputs[4], None);
        assert_eq!(mm.inputs[5], Some(bias));
        // The fused MatMulNBits output now feeds the graph output directly.
        assert_eq!(graph.outputs, vec![mm.outputs[0]]);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn folds_bias_add_regardless_of_operand_order() {
        let (mut graph, mm_id, bias) = matmul_bias_graph(&static_shape([N]), true);
        run_pass(&mut graph);

        assert_eq!(count(&graph, "Add"), 0);
        assert_eq!(graph.node(mm_id).inputs[5], Some(bias));
    }

    #[test]
    fn keeps_add_when_matmul_nbits_already_has_bias() {
        let (mut graph, mm_id, add_bias) = matmul_bias_graph(&static_shape([N]), false);
        let existing_bias =
            graph.create_named_value("existing_bias", DataType::Float32, static_shape([N]));
        graph.add_input(existing_bias);
        graph.node_mut(mm_id).inputs.resize(6, None);
        graph.replace_input(mm_id, 5, Some(existing_bias));

        run_pass(&mut graph);

        assert_eq!(
            count(&graph, "Add"),
            1,
            "already-biased MatMulNBits must not fold"
        );
        assert_eq!(graph.node(mm_id).inputs[5], Some(existing_bias));
        assert_ne!(graph.node(mm_id).inputs[5], Some(add_bias));
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn keeps_add_when_bias_is_not_a_row_vector() {
        // A rank-2 [1, N] bias is not the `[N]` shape the kernel accepts.
        let (mut graph, mm_id, _) = matmul_bias_graph(&static_shape([1, N]), false);
        run_pass(&mut graph);

        assert_eq!(count(&graph, "Add"), 1, "non-row bias must not be folded");
        assert_eq!(graph.node(mm_id).inputs.len(), 3);
    }

    #[test]
    fn keeps_add_when_matmul_output_has_another_consumer() {
        let (mut graph, mm_id, _) = matmul_bias_graph(&static_shape([N]), false);
        // Add a second consumer of the MatMulNBits output so it is no longer
        // private to the Add and cannot be dropped.
        let mm_out = graph.node(mm_id).outputs[0];
        let other = graph.create_named_value("other", DataType::Float32, static_shape([1, N]));
        graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(mm_out)],
            vec![other],
        ));
        graph.add_output(other);

        run_pass(&mut graph);

        assert_eq!(count(&graph, "Add"), 1, "shared output blocks the fold");
        assert_eq!(graph.node(mm_id).inputs.len(), 3);
    }

    #[test]
    fn keeps_add_when_matmul_output_is_a_graph_output() {
        let (mut graph, mm_id, _) = matmul_bias_graph(&static_shape([N]), false);
        // Expose the intermediate MatMulNBits result as a graph output; folding
        // would remove an observable value.
        let mm_out = graph.node(mm_id).outputs[0];
        let mut outputs = graph.outputs.clone();
        outputs.push(mm_out);
        graph.set_outputs(outputs);

        run_pass(&mut graph);

        assert_eq!(count(&graph, "Add"), 1);
        assert_eq!(graph.node(mm_id).inputs.len(), 3);
    }

    use super::{f32_to_bytes, ConvBatchNormActivationFusion};
    use onnx_runtime_ir::{TensorData, WeightRef};

    fn set_f32_initializer(graph: &mut Graph, name: &str, dims: &[usize], data: &[f32]) -> ValueId {
        let value = graph.create_named_value(name, DataType::Float32, static_shape(dims.to_vec()));
        graph.set_initializer(
            value,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                dims.to_vec(),
                f32_to_bytes(data),
            )),
        );
        value
    }

    fn read_f32_initializer(graph: &Graph, value: ValueId) -> Vec<f32> {
        let weight = graph.initializers.get(&value).expect("initializer present");
        let WeightRef::Inline(tensor) = weight else {
            panic!("expected inline initializer");
        };
        tensor
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Builds `Conv(x, w[, b]) -> BatchNormalization -> [Relu]` with constant BN
    /// parameters and returns the graph plus the Conv node id.
    #[allow(clippy::too_many_arguments)]
    fn conv_bn_graph(
        weight: &[f32],
        weight_dims: &[usize],
        bias: Option<&[f32]>,
        scale: &[f32],
        beta: &[f32],
        mean: &[f32],
        var: &[f32],
        epsilon: f32,
        with_relu: bool,
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let out_ch = weight_dims[0];

        let x = graph.create_named_value("x", DataType::Float32, static_shape([1, 3, 8, 8]));
        graph.add_input(x);
        let w = set_f32_initializer(&mut graph, "w", weight_dims, weight);
        let mut conv_inputs = vec![Some(x), Some(w)];
        if let Some(bias) = bias {
            conv_inputs.push(Some(set_f32_initializer(&mut graph, "b", &[out_ch], bias)));
        }

        let conv_out = graph.create_named_value(
            "conv_out",
            DataType::Float32,
            static_shape([1, out_ch, 6, 6]),
        );
        let conv = Node::new(NodeId(0), "Conv", conv_inputs, vec![conv_out]);
        let conv_id = graph.insert_node(conv);

        let scale_v = set_f32_initializer(&mut graph, "scale", &[out_ch], scale);
        let beta_v = set_f32_initializer(&mut graph, "beta", &[out_ch], beta);
        let mean_v = set_f32_initializer(&mut graph, "mean", &[out_ch], mean);
        let var_v = set_f32_initializer(&mut graph, "var", &[out_ch], var);
        let bn_out =
            graph.create_named_value("bn_out", DataType::Float32, static_shape([1, out_ch, 6, 6]));
        let mut bn = Node::new(
            NodeId(0),
            "BatchNormalization",
            vec![
                Some(conv_out),
                Some(scale_v),
                Some(beta_v),
                Some(mean_v),
                Some(var_v),
            ],
            vec![bn_out],
        );
        bn.attributes
            .insert("epsilon".into(), Attribute::Float(epsilon));
        graph.insert_node(bn);

        let final_out = if with_relu {
            let relu_out = graph.create_named_value(
                "relu_out",
                DataType::Float32,
                static_shape([1, out_ch, 6, 6]),
            );
            graph.insert_node(Node::new(
                NodeId(0),
                "Relu",
                vec![Some(bn_out)],
                vec![relu_out],
            ));
            relu_out
        } else {
            bn_out
        };
        graph.add_output(final_out);
        (graph, conv_id)
    }

    fn run_conv_bn_pass(graph: &mut Graph) {
        ConvBatchNormActivationFusion::new()
            .run(graph, &PassContext::new())
            .expect("conv-bn fusion pass");
    }

    #[test]
    fn folds_conv_bn_relu_into_conv() {
        // Two output channels, 3 input channels, 3x3 kernel.
        let out_ch = 2usize;
        let per_filter = 3 * 3 * 3;
        let weight: Vec<f32> = (0..out_ch * per_filter)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let bias = [0.5f32, -0.25];
        let scale = [1.5f32, 0.75];
        let beta = [0.1f32, -0.2];
        let mean = [0.3f32, 0.6];
        let var = [4.0f32, 9.0];
        let epsilon = 1e-5f32;

        let (mut graph, conv_id) = conv_bn_graph(
            &weight,
            &[out_ch, 3, 3, 3],
            Some(&bias),
            &scale,
            &beta,
            &mean,
            &var,
            epsilon,
            true,
        );
        run_conv_bn_pass(&mut graph);

        assert_eq!(count(&graph, "BatchNormalization"), 0, "BN folded away");
        assert_eq!(count(&graph, "Relu"), 0, "Relu folded into activation");
        assert_eq!(count(&graph, "Conv"), 1);

        let conv = graph.node(conv_id);
        assert_eq!(
            conv.attr("activation").and_then(Attribute::as_str),
            Some("Relu")
        );
        let folded_w = read_f32_initializer(&graph, conv.inputs[1].unwrap());
        let folded_b = read_f32_initializer(&graph, conv.inputs[2].unwrap());

        for c in 0..out_ch {
            let a = scale[c] / (var[c] + epsilon).sqrt();
            for j in 0..per_filter {
                let expected = weight[c * per_filter + j] * a;
                assert!((folded_w[c * per_filter + j] - expected).abs() < 1e-6);
            }
            let expected_b = (bias[c] - mean[c]) * a + beta[c];
            assert!((folded_b[c] - expected_b).abs() < 1e-6);
        }
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn folds_conv_bn_without_bias_or_relu() {
        let out_ch = 2usize;
        let per_filter = 3 * 3 * 3;
        let weight: Vec<f32> = (0..out_ch * per_filter)
            .map(|i| (i as f32) * 0.02)
            .collect();
        let scale = [2.0f32, 0.5];
        let beta = [0.0f32, 1.0];
        let mean = [1.0f32, -1.0];
        let var = [1.0f32, 4.0];
        let epsilon = 1e-5f32;

        let (mut graph, conv_id) = conv_bn_graph(
            &weight,
            &[out_ch, 3, 3, 3],
            None,
            &scale,
            &beta,
            &mean,
            &var,
            epsilon,
            false,
        );
        run_conv_bn_pass(&mut graph);

        assert_eq!(count(&graph, "BatchNormalization"), 0);
        let conv = graph.node(conv_id);
        assert!(conv.attr("activation").is_none(), "no Relu to fold");
        let folded_b = read_f32_initializer(&graph, conv.inputs[2].unwrap());
        for c in 0..out_ch {
            let a = scale[c] / (var[c] + epsilon).sqrt();
            let expected_b = (0.0 - mean[c]) * a + beta[c];
            assert!((folded_b[c] - expected_b).abs() < 1e-6);
        }
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn skips_fusion_when_conv_output_is_shared() {
        let out_ch = 2usize;
        let per_filter = 3 * 3 * 3;
        let weight = vec![0.01f32; out_ch * per_filter];
        let (mut graph, conv_id) = conv_bn_graph(
            &weight,
            &[out_ch, 3, 3, 3],
            None,
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            1e-5,
            false,
        );
        // Expose the Conv output as a graph output so folding would drop an
        // observable value.
        let conv_out = graph.node(conv_id).outputs[0];
        let mut outputs = graph.outputs.clone();
        outputs.push(conv_out);
        graph.set_outputs(outputs);

        run_conv_bn_pass(&mut graph);
        assert_eq!(
            count(&graph, "BatchNormalization"),
            1,
            "shared Conv output blocks the fold"
        );
    }
}
