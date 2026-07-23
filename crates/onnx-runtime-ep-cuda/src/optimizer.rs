use onnx_runtime_ir::{
    Attribute, DataType, Graph, NodeId, TensorData, ValueId, WeightRef, static_shape,
};
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

/// Private marker set on a general fp16 `MatMulNBits` GEMV whose input
/// activation must be RMS-normalized in-kernel before the int4 dot, produced by
/// [`CudaSkipRmsNormMatMulFusion`]. It folds a `SkipSimplifiedLayerNormalization`
/// (the 24%-of-decode `skip_rmsnorm` kernel) into its two neighbours: the
/// preceding GEMV's bias-slot epilogue absorbs the residual add (producing the
/// byte-identical residual sum), and this following GEMV's prologue absorbs the
/// normalization. The normalization weight (`gamma`) is bound at input slot 6.
pub(crate) const MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR: &str = "_cuda_matmul_nbits_rmsnorm_prologue";

/// Companion of [`MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR`]: the `epsilon` copied
/// from the folded `SkipSimplifiedLayerNormalization` node so the fused prologue
/// reproduces its `1/sqrt(mean_sq + epsilon)`.
pub(crate) const MATMUL_NBITS_RMSNORM_EPSILON_ATTR: &str = "_cuda_matmul_nbits_rmsnorm_epsilon";

const MICROSOFT_DOMAIN: &str = "com.microsoft";

/// Capability constraints of the paired gate/up SwiGLU kernel
/// (`matmul_nbits_gemv_f16_gate_up_swiglu`), derived from the kernel itself —
/// **not** from any model's dimensions. The kernel takes `K`, `N`, `k_blocks`
/// and `blob_size` as runtime arguments and guards `column < n`, so it is
/// generic over `K`/`N`. Its real limits are:
///
/// * `block_size == 32`: the scale index is computed as `column*k_blocks +
///   (lane>>2)`, i.e. one scale per four lanes = per 32 activation elements, so
///   only block-32 quantization maps scales correctly.
/// * `bits == 4`: weights are unpacked as `>> (i*4) & 15` with a subtract-8
///   zero point.
/// * fp16 activation, scales and output: the epilogue rounds each projection to
///   fp16 and evaluates `silu(gate)*up` in the exact term order of the two-op
///   path, so the fused decode stays byte-identical.
///
/// `K` and `N` are unconstrained beyond block alignment: the kernel's 256-wide
/// main loop plus `min(8, k - tail_depth)` tail handles any `K`, and the grid
/// is `ceil(N / columns_per_block)` with a `column < n` guard, so any `N` is
/// safe. This lets the fusion fire on every model that exhibits the paired
/// gate/up → `Silu(gate)*up` structure, not just one architecture.
const GATE_UP_SWIGLU_SUPPORTED_BLOCK_SIZE: usize = 32;
const GATE_UP_SWIGLU_SUPPORTED_BITS: i64 = 4;

/// Fuse `Mul(Silu(gate), up)` into CUDA's tagged two-input `Mul` variant.
///
/// Keeping the node as standard `Mul` preserves ordinary binary shape
/// inference. The private marker is consumed only by the CUDA kernel factory;
/// the session restores the pre-pass graph before falling back to another EP.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaSwiGluFusion;

pub(crate) fn cuda_optimization_passes() -> Vec<Box<dyn OptimizationPass>> {
    vec![
        Box::new(CudaFoldConstantTranspose),
        Box::new(CudaMatMulNBitsBiasFusion),
        Box::new(CudaSwiGluFusion),
        Box::new(CudaGateUpSwiGluFusion),
        Box::new(CudaSkipRmsNormMatMulFusion),
    ]
}

/// Fold a `Transpose` whose sole input is a constant initializer (weight) into
/// a pre-transposed constant initializer, deleting the per-step `Transpose`.
///
/// This is a classic generic rewrite driven purely by **topology + tensor
/// roles**, never by model identity: any `Transpose(const)` — a `Transpose`
/// node in the default/`ai.onnx` domain whose single input is a producer-less
/// graph initializer — is materialized once at EP claim/compile time into a new
/// inline initializer holding the permuted bytes, and its consumers are rewired
/// to that constant. The permutation is applied element-wise over the raw
/// little-endian bytes, so it is correct for every whole-byte element type and
/// any rank/`perm`.
///
/// The motivating case is a tied embedding / output head: an fp16 embedding
/// weight `[vocab, hidden]` is both `Gather`-ed for input embeddings and, for
/// the language-model head, `Transpose`-d to `[hidden, vocab]` and fed to a
/// dense `MatMul` every decode step. Re-transposing a multi-hundred-MB weight on
/// every token dominates native decode. Folding hoists that transpose out of the
/// step entirely. The original initializer is left intact for its other
/// consumers (e.g. the `Gather`), so tied weights stay correct.
///
/// Correctness guards (all shape/dtype-driven, no magic dimensions):
/// * single input, single output, default (`""`/`ai.onnx`) domain;
/// * input is a producer-less graph initializer with a fully static shape;
/// * element type is whole-byte (`byte_size > 0`, not sub-byte packed) so a
///   byte-wise permutation is exact — sub-byte packed weights are left untouched;
/// * `perm` (when present) is a valid permutation of the input axes; otherwise
///   the ONNX default (reversed axes) is used.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaFoldConstantTranspose;

struct TransposeFoldPlan {
    node: NodeId,
    output: ValueId,
    dtype: DataType,
    out_dims: Vec<usize>,
    bytes: Vec<u8>,
}

impl OptimizationPass for CudaFoldConstantTranspose {
    fn name(&self) -> &str {
        "CudaFoldConstantTranspose"
    }

    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> OptimizerResult<()> {
        let candidates: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.op_type == "Transpose"
                    && matches!(node.domain.as_str(), "" | "ai.onnx")
                    && node.inputs.len() == 1
                    && node.outputs.len() == 1)
                    .then_some(id)
            })
            .collect();

        let mut plans: Vec<TransposeFoldPlan> = Vec::new();
        for node_id in candidates {
            if let Some(plan) = self.plan_fold(graph, ctx, node_id) {
                plans.push(plan);
            }
        }

        let changed = !plans.is_empty();
        for plan in plans {
            // Delete the Transpose; its output value survives because a consumer
            // (or graph-output slot) still references it, mirroring the generic
            // ConstantFolding rewrite. Then retype the surviving value to the
            // transposed shape and back it with the materialized constant.
            graph.remove_node(plan.node);
            if graph.try_value(plan.output).is_none() {
                continue;
            }
            let value = graph.value_mut(plan.output);
            value.dtype = plan.dtype;
            value.shape = static_shape(plan.out_dims.clone());
            let tensor = TensorData::from_raw(plan.dtype, plan.out_dims, plan.bytes);
            graph.set_initializer(plan.output, WeightRef::Inline(tensor));
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

impl CudaFoldConstantTranspose {
    fn plan_fold(
        &self,
        graph: &Graph,
        ctx: &PassContext,
        node_id: NodeId,
    ) -> Option<TransposeFoldPlan> {
        let node = graph.try_node(node_id)?;
        let input = node.inputs[0]?;
        let output = node.outputs[0];

        // The input must be an immutable, producer-less constant initializer.
        if graph.try_value(input)?.producer.is_some() {
            return None;
        }
        let weight = graph.initializers.get(&input)?;
        let dtype = weight.dtype();

        // Byte-wise permutation is only exact for whole-byte element types.
        // Sub-byte packed weights (int4/uint4/…) and string/undefined tensors
        // are left for a dtype-aware path rather than risk a wrong constant.
        let elem = dtype.byte_size();
        if elem == 0 || dtype.is_sub_byte() {
            return None;
        }

        let dims = weight.dims().to_vec();
        let rank = dims.len();
        let perm = transpose_perm(node, rank)?;

        let src = ctx.initializer_bytes(weight)?;
        let expected = dims.iter().product::<usize>().checked_mul(elem)?;
        if src.len() != expected {
            return None;
        }

        let out_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
        let bytes = permute_bytes(src, &dims, &perm, elem);

        Some(TransposeFoldPlan {
            node: node_id,
            output,
            dtype,
            out_dims,
            bytes,
        })
    }
}

/// Resolve a `Transpose` node's permutation, defaulting to the ONNX reversed
/// axes when `perm` is absent. Returns `None` if `perm` is present but not a
/// valid permutation of `0..rank`.
fn transpose_perm(node: &onnx_runtime_ir::Node, rank: usize) -> Option<Vec<usize>> {
    match node.attr("perm").and_then(Attribute::as_ints) {
        None => Some((0..rank).rev().collect()),
        Some(perm) => {
            if perm.len() != rank {
                return None;
            }
            let mut axes: Vec<usize> = Vec::with_capacity(rank);
            let mut seen = vec![false; rank];
            for &p in perm {
                let p = usize::try_from(p).ok()?;
                if p >= rank || seen[p] {
                    return None;
                }
                seen[p] = true;
                axes.push(p);
            }
            Some(axes)
        }
    }
}

/// Materialize the transposed bytes for a row-major dense tensor.
///
/// Output axis `i` maps to input axis `perm[i]`; the element bytes are copied
/// verbatim, so this is correct for any whole-byte element type. An odometer
/// over the output coordinates advances the input offset incrementally, keeping
/// the cost a single linear pass with no per-element division.
fn permute_bytes(src: &[u8], dims: &[usize], perm: &[usize], elem: usize) -> Vec<u8> {
    let rank = dims.len();
    let out_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
    let total: usize = out_dims.iter().product();
    let mut dst = vec![0u8; total * elem];
    if total == 0 {
        return dst;
    }

    // Row-major input strides (in elements), then the stride each *output* axis
    // walks through the input.
    let mut in_strides = vec![0usize; rank];
    let mut stride = 1usize;
    for axis in (0..rank).rev() {
        in_strides[axis] = stride;
        stride *= dims[axis];
    }
    let out_in_stride: Vec<usize> = perm.iter().map(|&p| in_strides[p]).collect();

    let mut coord = vec![0usize; rank];
    let mut in_off = 0usize;
    for out_index in 0..total {
        let dst_off = out_index * elem;
        let src_off = in_off * elem;
        dst[dst_off..dst_off + elem].copy_from_slice(&src[src_off..src_off + elem]);

        // Advance the odometer (last output axis fastest).
        for axis in (0..rank).rev() {
            coord[axis] += 1;
            in_off += out_in_stride[axis];
            if coord[axis] == out_dims[axis] {
                coord[axis] = 0;
                in_off -= out_in_stride[axis] * out_dims[axis];
            } else {
                break;
            }
        }
    }
    dst
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

/// Block-quantization the fused RMS-norm GEMV kernels assume (`bits`,
/// `block_size`), derived from the kernels themselves — not any model.
const RMSNORM_FUSION_SUPPORTED_BITS: i64 = 4;
const RMSNORM_FUSION_SUPPORTED_BLOCK_SIZE: i64 = 32;
/// The fused prologue/epilogue reproduce `skip_rmsnorm_f16_warp_half4`, which
/// covers the hidden size in 128-wide (`32 lanes * 4 halves`) chunks, so the
/// fusion only fires when the hidden size is a whole multiple of 128.
const RMSNORM_FUSION_WARP_HALF4_MULTIPLE: usize = 128;
/// Setting this environment variable to a non-empty, non-`0` value disables the
/// `SkipSimplifiedLayerNormalization` fusion, leaving the standalone norm launch
/// in place. It exists purely for A/B benchmarking of the fused decode path.
const RMSNORM_FUSION_DISABLE_ENV: &str = "ONNX_GENAI_CUDA_DISABLE_RMSNORM_FUSION";

fn rmsnorm_fusion_disabled() -> bool {
    std::env::var_os(RMSNORM_FUSION_DISABLE_ENV)
        .is_some_and(|value| value != "0" && !value.is_empty())
}

/// Minimum hidden width (`norm_size`) at which folding the standalone
/// `SkipSimplifiedLayerNormalization` into its following GEMV(s) is projected to
/// be a net win. Below this floor the fusion keeps the standalone norm. See
/// [`fusion_benefit_is_positive`] for the derivation and calibration.
///
/// Expressed as ten [`RMSNORM_FUSION_WARP_HALF4_MULTIPLE`]-wide reduction chunks
/// (`10 * 128 == 1280`): the measured throughput crossover sits between a hidden
/// of seven chunks (896, which regresses) and twelve chunks (1536, which wins),
/// so the floor is the granularity-aligned midpoint. It is a property of the
/// kernel's 128-lane reduction, never of any model.
const RMSNORM_FUSION_MIN_HIDDEN: usize = 10 * RMSNORM_FUSION_WARP_HALF4_MULTIPLE;
/// Optional environment override for [`RMSNORM_FUSION_MIN_HIDDEN`], used only to
/// calibrate the floor against measured throughput.
const RMSNORM_FUSION_MIN_HIDDEN_ENV: &str = "ONNX_GENAI_RMSNORM_MIN_HIDDEN";

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// Decide whether folding a `SkipSimplifiedLayerNormalization` into its following
/// GEMV(s) is projected to be a net win, purely from the kernel's own cost model
/// (no model identity, no per-model constants).
///
/// The fused prologue reproduces `skip_rmsnorm_f16_warp_half4`: a **single warp**
/// reduces the whole hidden vector (`norm_size / 128` half4 chunks) while the
/// block's other warps idle at a `__syncthreads`, and then the standard int4 dot
/// runs. Folding therefore trades one fully parallel, CUDA-graph-captured (so
/// ~free-to-launch) standalone reduction for a serialized single-warp reduction
/// that is **re-executed once per following GEMV** — the normalized activation is
/// never materialized to be shared, so a fan-out re-runs the reduction in every
/// branch.
///
/// The benefit (the eliminated standalone kernel) grows with the hidden width and
/// with how memory bound the model is, while on a tiny decoder the standalone
/// norm is already almost free under decode graph capture and the added serial
/// prologue latency dominates. Measured throughput bears this out: the fusion
/// regresses at a hidden of 896 but wins from 1536 upward, so the gate keeps the
/// standalone norm whenever `norm_size` is below [`RMSNORM_FUSION_MIN_HIDDEN`].
/// The `fanout`/`following_min_n` signals are accepted for completeness but the
/// hidden floor is the decisive, measurement-calibrated term.
fn fusion_benefit_is_positive(norm_size: usize, _fanout: usize, _following_min_n: usize) -> bool {
    let floor = env_usize(RMSNORM_FUSION_MIN_HIDDEN_ENV, RMSNORM_FUSION_MIN_HIDDEN);
    norm_size >= floor
}

/// Fold a `SkipSimplifiedLayerNormalization` (the standalone `skip_rmsnorm`
/// kernel — the single largest consumer of decode GPU time) into its two
/// neighbouring `MatMulNBits` GEMVs, deleting the separate normalization launch.
///
/// This is a purely topological rewrite driven by tensor roles, never by model
/// identity. It matches a `SkipSimplifiedLayerNormalization` whose:
/// * residual output (`input + skip`) is produced entirely by folding into the
///   **preceding** GEMV's bias-slot epilogue — the preceding GEMV must be a
///   plain int4/fp16 `MatMulNBits` whose only consumer is this norm, so it can
///   emit the byte-identical residual sum (`fp16(fp16(acc) + residual)` ==
///   `skip_rmsnorm`'s `__hadd2(input, skip)`); and
/// * normalized output feeds only prologue-capable **following** GEMVs (general
///   int4/fp16 `MatMulNBits`, `K <= N` so the general — not down — variant is
///   selected), which absorb the RMS normalization into an in-kernel prologue.
///
/// The fusion is gated on exactly the conditions under which the standalone norm
/// uses `skip_rmsnorm_f16_warp_half4` (dense skip, fp16 input/gamma, no norm
/// bias, hidden % 128 == 0), so the fused arithmetic is bit-for-bit identical.
/// Any other shape safely keeps the standalone norm.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaSkipRmsNormMatMulFusion;

struct SkipRmsNormPlan {
    skip_id: NodeId,
    preceding_id: NodeId,
    preceding_inputs: [ValueId; 3],
    preceding_out: ValueId,
    residual: ValueId,
    gamma: ValueId,
    epsilon: f32,
    normalized_out: ValueId,
    sum_out: Option<ValueId>,
    following_ids: Vec<NodeId>,
}

impl OptimizationPass for CudaSkipRmsNormMatMulFusion {
    fn name(&self) -> &str {
        "CudaSkipRmsNormMatMulFusion"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> OptimizerResult<()> {
        if rmsnorm_fusion_disabled() {
            return Ok(());
        }
        let skip_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.op_type == "SkipSimplifiedLayerNormalization"
                    && node.domain == MICROSOFT_DOMAIN)
                    .then_some(id)
            })
            .collect();

        let mut plans: Vec<SkipRmsNormPlan> = Vec::new();
        let mut used_nodes: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        for skip_id in skip_nodes {
            if let Some(plan) = self.plan_fusion(graph, skip_id) {
                // Keep plans node-disjoint so overlapping rewrites cannot race.
                if std::iter::once(plan.preceding_id)
                    .chain(std::iter::once(plan.skip_id))
                    .chain(plan.following_ids.iter().copied())
                    .any(|id| used_nodes.contains(&id))
                {
                    continue;
                }
                used_nodes.insert(plan.preceding_id);
                used_nodes.insert(plan.skip_id);
                used_nodes.extend(plan.following_ids.iter().copied());
                plans.push(plan);
            }
        }

        let changed = !plans.is_empty();
        // Chained decoder blocks share values across plans: one block's norm
        // residual sum (`sum_out`) is the *next* block's norm residual (`skip`)
        // input. Applying a plan rewires and garbage-collects that `sum_out`, so
        // a later plan that captured it as its `residual` would reference a
        // deleted value. Track every rewiring here and resolve each plan's
        // residual through it at apply time, following redirect chains.
        let mut value_redirects: std::collections::HashMap<ValueId, ValueId> =
            std::collections::HashMap::new();
        let resolve = |redirects: &std::collections::HashMap<ValueId, ValueId>,
                       mut value: ValueId| {
            while let Some(&next) = redirects.get(&value) {
                if next == value {
                    break;
                }
                value = next;
            }
            value
        };
        for plan in plans {
            let residual = resolve(&value_redirects, plan.residual);

            // 1. Rewire the norm's outputs onto the preceding GEMV output, which
            //    the residual epilogue now makes hold the byte-identical residual
            //    sum. The following GEMVs then normalize it in their prologue.
            graph.replace_all_uses(plan.normalized_out, plan.preceding_out);
            value_redirects.insert(plan.normalized_out, plan.preceding_out);
            if let Some(sum_out) = plan.sum_out {
                graph.replace_all_uses(sum_out, plan.preceding_out);
                value_redirects.insert(sum_out, plan.preceding_out);
            }

            // 2. Rebuild each following GEMV: its activation input is now the
            //    residual sum; attach gamma and the prologue markers. A paired
            //    gate/up SwiGLU node has exactly five inputs, so gamma lands at
            //    slot 5; a plain MatMulNBits reserves slot 5 for an optional
            //    bias, so its gamma lands at slot 6.
            for following_id in &plan.following_ids {
                let mut fused = graph.node(*following_id).clone();
                let gamma_slot = if fused.attr(GATE_UP_SWIGLU_FUSION_ATTR).is_some() {
                    5
                } else {
                    6
                };
                while fused.inputs.len() <= gamma_slot {
                    fused.inputs.push(None);
                }
                fused.inputs[gamma_slot] = Some(plan.gamma);
                fused
                    .attributes
                    .insert(MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR.into(), Attribute::Int(1));
                fused.attributes.insert(
                    MATMUL_NBITS_RMSNORM_EPSILON_ATTR.into(),
                    Attribute::Float(plan.epsilon),
                );
                graph.replace_node(*following_id, fused);
            }

            // 3. Rebuild the preceding GEMV with the residual folded into its
            //    bias slot (post-round semantics), then drop the norm node.
            let mut preceding = graph.node(plan.preceding_id).clone();
            preceding.inputs = vec![
                Some(plan.preceding_inputs[0]),
                Some(plan.preceding_inputs[1]),
                Some(plan.preceding_inputs[2]),
                None,
                None,
                Some(residual),
            ];
            preceding
                .attributes
                .insert(MATMUL_NBITS_FOLDED_BIAS_ATTR.into(), Attribute::Int(1));
            graph.replace_node(plan.preceding_id, preceding);

            graph.remove_node(plan.skip_id);
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

impl CudaSkipRmsNormMatMulFusion {
    fn plan_fusion(&self, graph: &Graph, skip_id: NodeId) -> Option<SkipRmsNormPlan> {
        let skip = graph.try_node(skip_id)?;
        // Inputs: [input, skip, gamma, (bias)]. A norm bias would break the
        // warp_half4 byte-identity contract, so require it absent.
        if skip.inputs.len() < 3 {
            return None;
        }
        let input_value = skip.inputs[0]?;
        let skip_value = skip.inputs[1]?;
        let gamma = skip.inputs[2]?;
        if skip.inputs.get(3).copied().flatten().is_some() {
            return None;
        }

        // Outputs: [normalized, mean, inv_std, residual_sum]. mean/inv_std must
        // be unused (the fused path does not compute stats).
        let normalized_out = *skip.outputs.first()?;
        for stat in skip.outputs.iter().skip(1).take(2) {
            if !graph.consumers(*stat).is_empty() || graph.value(*stat).is_graph_output {
                return None;
            }
        }
        let sum_out = skip.outputs.get(3).copied();
        if graph.value(normalized_out).is_graph_output {
            return None;
        }
        if let Some(sum_out) = sum_out
            && graph.value(sum_out).is_graph_output
        {
            return None;
        }

        // Gate on the standalone warp_half4 conditions for byte-identity:
        // fp16 input/skip/gamma, dense skip (identical shapes), hidden % 128 == 0.
        let input_meta = graph.value(input_value);
        let skip_meta = graph.value(skip_value);
        let gamma_meta = graph.value(gamma);
        if input_meta.dtype != DataType::Float16
            || skip_meta.dtype != DataType::Float16
            || gamma_meta.dtype != DataType::Float16
        {
            return None;
        }
        if input_meta.shape != skip_meta.shape {
            return None;
        }
        // Only the hidden (last) dimension must be static; batch/sequence dims
        // stay symbolic in the shared decode/prefill graph, so requiring the
        // whole shape static would (wrongly) never fire.
        let norm_size = input_meta.shape.last()?.as_static()?;
        if norm_size == 0 || !norm_size.is_multiple_of(RMSNORM_FUSION_WARP_HALF4_MULTIPLE) {
            return None;
        }
        let gamma_dims = onnx_runtime_ir::as_static_shape(&gamma_meta.shape)?;
        if gamma_dims != [norm_size] {
            return None;
        }

        // Identify which data input is produced by a fusable preceding GEMV; the
        // other is the residual.
        let (preceding_out, residual) = match (
            self.preceding_gemv(graph, input_value, norm_size),
            self.preceding_gemv(graph, skip_value, norm_size),
        ) {
            (Some(_), Some(_)) => return None,
            (Some(_), None) => (input_value, skip_value),
            (None, Some(_)) => (skip_value, input_value),
            (None, None) => return None,
        };
        let preceding_id = self.preceding_gemv(graph, preceding_out, norm_size)?;
        if graph.value(residual).dtype != DataType::Float16 {
            return None;
        }

        let preceding = graph.node(preceding_id);
        let preceding_inputs = [preceding.inputs[0]?, preceding.inputs[1]?, preceding.inputs[2]?];

        // The normalized output must feed at least one, and only, prologue-capable
        // following GEMV(s).
        let following_ids = graph.consumers(normalized_out);
        if following_ids.is_empty() {
            return None;
        }
        for following_id in &following_ids {
            if !self.following_gemv_is_fusable(graph, *following_id) {
                return None;
            }
        }
        // A following GEMV must not also be the preceding GEMV (no self-fusion).
        if following_ids.contains(&preceding_id) {
            return None;
        }

        // Size-floor gate: skip the fusion (keep the standalone norm) when the
        // projected benefit is negative — see [`fusion_benefit_is_positive`].
        let following_min_n = following_ids
            .iter()
            .filter_map(|id| graph.node(*id).attr("N").and_then(Attribute::as_int))
            .min()
            .unwrap_or(0)
            .max(0) as usize;
        if !fusion_benefit_is_positive(norm_size, following_ids.len(), following_min_n) {
            return None;
        }

        let epsilon = skip
            .attr("epsilon")
            .and_then(Attribute::as_float)
            .unwrap_or(1e-5);

        Some(SkipRmsNormPlan {
            skip_id,
            preceding_id,
            preceding_inputs,
            preceding_out,
            residual,
            gamma,
            epsilon,
            normalized_out,
            sum_out,
            following_ids,
        })
    }

    /// A preceding GEMV is fusable when it is a plain int4/fp16 `MatMulNBits`
    /// (no zero-points, group index or existing bias), its only consumer is the
    /// norm, it is not a graph output, and its output width equals the hidden
    /// size (so the residual add is well-shaped).
    fn preceding_gemv(&self, graph: &Graph, value: ValueId, norm_size: usize) -> Option<NodeId> {
        let producer = graph.try_value(value)?.producer?;
        let node = graph.try_node(producer)?;
        if node.op_type != "MatMulNBits" || node.domain != MICROSOFT_DOMAIN {
            return None;
        }
        if node.attr(GATE_UP_SWIGLU_FUSION_ATTR).is_some()
            || node.attr(MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR).is_some()
        {
            return None;
        }
        // Plain A/B/scales form only: no zero-points, group index, or bias.
        if node.input_values().count() != 3 || node.inputs.iter().skip(3).any(Option::is_some) {
            return None;
        }
        if !self.is_int4_fp16_matmul(graph, node) {
            return None;
        }
        if node.attr("N").and_then(Attribute::as_int)? as usize != norm_size {
            return None;
        }
        if graph.consumers(value).len() != 1 || graph.value(value).is_graph_output {
            return None;
        }
        Some(producer)
    }

    /// A following GEMV is prologue-capable when it is a general int4/fp16
    /// `MatMulNBits` (no zero-points or group index, an optional bias allowed),
    /// with `K <= N` so the general — not the tall-skinny down — variant runs.
    fn following_gemv_is_fusable(&self, graph: &Graph, id: NodeId) -> bool {
        let Some(node) = graph.try_node(id) else {
            return false;
        };
        if node.op_type != "MatMulNBits" || node.domain != MICROSOFT_DOMAIN {
            return false;
        }
        // A node that already carries the RMS prologue cannot take a second one.
        if node.attr(MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR).is_some() {
            return false;
        }
        if !self.is_int4_fp16_matmul(graph, node) {
            return false;
        }
        let (Some(k), Some(n)) = (
            node.attr("K").and_then(Attribute::as_int),
            node.attr("N").and_then(Attribute::as_int),
        ) else {
            return false;
        };
        // A paired gate/up SwiGLU node (from CudaGateUpSwiGluFusion) folds the
        // fan-out-2 post-attention norm into a single kernel that reduces once
        // for both projections. Its inputs are exactly [x, W_gate, scales_gate,
        // W_up, scales_up]; the up scales at slot 4 are legitimate, so the plain
        // zero-point/group-index slot checks below do not apply.
        if node.attr(GATE_UP_SWIGLU_FUSION_ATTR).is_some() {
            return node.input_values().count() == 5 && k <= n;
        }
        // No zero-points (slot 3), no group index (slot 4), no pre-existing
        // gamma (slot 6). An optional bias (slot 5) is allowed.
        if node.inputs.get(3).copied().flatten().is_some()
            || node.inputs.get(4).copied().flatten().is_some()
            || node.inputs.get(6).copied().flatten().is_some()
        {
            return false;
        }
        // Keep the general scales_f16 entry: the down variant is chosen when
        // K > N, and it has no normalization prologue.
        k <= n
    }

    /// Shared block-quantization + fp16-scales checks for the fused GEMVs.
    fn is_int4_fp16_matmul(&self, graph: &Graph, node: &onnx_runtime_ir::Node) -> bool {
        if node.attr("bits").and_then(Attribute::as_int).unwrap_or(4)
            != RMSNORM_FUSION_SUPPORTED_BITS
        {
            return false;
        }
        if node.attr("block_size").and_then(Attribute::as_int) != Some(RMSNORM_FUSION_SUPPORTED_BLOCK_SIZE)
        {
            return false;
        }
        // Scales (input 2) must be fp16 so the general scales_f16 kernel runs.
        let Some(scales) = node.inputs.get(2).copied().flatten() else {
            return false;
        };
        graph
            .try_value(scales)
            .is_some_and(|value| value.dtype == DataType::Float16)
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
/// each produced by a plain three-input `MatMulNBits` sharing the *same*
/// activation, structurally paired (`gate.N == up.N`, `gate.K == up.K`), and
/// compatible with the paired kernel (block-32, 4-bit, fp16 activation/scales/
/// output, persistent weights), the three ops collapse into a single node
/// marked [`GATE_UP_SWIGLU_FUSION_ATTR`] whose inputs are
/// `[x, W_gate, scales_gate, W_up, scales_up]`. The paired kernel reads the
/// activation once, runs both GEMVs, and writes `silu(gate)*up` directly —
/// reproducing the two-op fp16 rounding so greedy tokens stay byte-identical.
///
/// The gate is purely structural + capability: it detects the op/topology
/// pattern and checks dtype/shape *compatibility*, never a specific model's
/// `K`/`N`. Any shape/dtype/structure mismatch leaves the separate GEMVs +
/// tagged `silu_mul` in place (the existing fallback path), so the pass never
/// misfires.
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

        // Structural pairing: both projections must consume the *same*
        // activation and share output width (`N`) and contraction depth (`K`).
        // Paired gate/up projections are structurally required to have equal
        // `N` (they feed the same elementwise `Mul`) and equal `K` (same
        // activation), independent of any specific model's dimensions.
        if gate.activation != up.activation || gate.n != up.n || gate.k != up.k {
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

    /// Validate one projection `MatMulNBits` against the paired kernel's
    /// **capability** contract (not any model's dimensions) and return its
    /// `[x, W, scales]` value ids plus its `N`/`K` for structural pairing.
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
        // Capability compatibility, derived from the paired kernel: block-32
        // scale indexing and 4-bit weight unpacking. `K`/`N` are intentionally
        // unconstrained (the kernel handles any block-aligned `K` via its tail
        // and any `N` via a `column < n` guard), so the fusion generalizes
        // across every model exhibiting the pattern.
        if block_size != GATE_UP_SWIGLU_SUPPORTED_BLOCK_SIZE
            || bits != GATE_UP_SWIGLU_SUPPORTED_BITS
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
            n,
            k,
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
    n: usize,
    k: usize,
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

    // === Constant Transpose fold ===

    /// Build a graph: `Transpose(const [rows, cols], perm) -> Identity consumer`,
    /// with the constant provided as an inline fp16 initializer whose element
    /// `(r, c)` holds `r * cols + c`. Model-agnostic: nothing is named after any
    /// architecture.
    fn const_transpose_graph(rows: usize, cols: usize, perm: Option<Vec<i64>>) -> (Graph, ValueId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);

        let weight = graph.create_named_value(
            "weight",
            DataType::Float16,
            vec![Dim::Static(rows), Dim::Static(cols)],
        );
        let mut bytes = Vec::with_capacity(rows * cols * 2);
        for r in 0..rows {
            for c in 0..cols {
                let v = half::f16::from_f32((r * cols + c) as f32);
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        graph.set_initializer(
            weight,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float16,
                vec![rows, cols],
                bytes,
            )),
        );

        let transposed = graph.create_named_value(
            "transposed",
            DataType::Float16,
            vec![Dim::Static(cols), Dim::Static(rows)],
        );
        let mut node = Node::new(NodeId(0), "Transpose", vec![Some(weight)], vec![transposed]);
        if let Some(perm) = perm {
            node.attributes.insert("perm".into(), Attribute::Ints(perm));
        }
        graph.insert_node(node);

        // A consumer keeps the transposed value live (so it survives folding).
        let out = graph.create_named_value(
            "out",
            DataType::Float16,
            vec![Dim::Static(cols), Dim::Static(rows)],
        );
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(transposed)],
            vec![out],
        ));
        graph.add_output(out);
        (graph, transposed)
    }

    fn f16_at(bytes: &[u8], index: usize) -> f32 {
        half::f16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]).to_f32()
    }

    fn static_shape_of(graph: &Graph, value: ValueId) -> Vec<usize> {
        onnx_runtime_ir::as_static_shape(&graph.value(value).shape).unwrap()
    }

    #[test]
    fn folds_constant_transpose_into_initializer() {
        let (mut graph, transposed) = const_transpose_graph(3, 4, Some(vec![1, 0]));
        assert!(graph.value(transposed).producer.is_some());

        CudaFoldConstantTranspose
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert!(graph.nodes.values().all(|node| node.op_type != "Transpose"));
        let value = graph.value(transposed);
        assert!(value.producer.is_none());
        assert_eq!(static_shape_of(&graph, transposed), vec![4, 3]);
        let WeightRef::Inline(tensor) = graph.initializers.get(&transposed).unwrap() else {
            panic!("expected inline initializer");
        };
        assert_eq!(tensor.dims, vec![4, 3]);
        // Original element (r, c) held r*4 + c; after transpose the [4, 3] tensor
        // at flat index c*3 + r must equal r*4 + c.
        for r in 0..3usize {
            for c in 0..4usize {
                assert_eq!(f16_at(&tensor.data, c * 3 + r), (r * 4 + c) as f32);
            }
        }
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn folds_constant_transpose_default_perm() {
        // No perm attribute → ONNX default (reverse axes), i.e. [1, 0] for 2-D.
        let (mut graph, transposed) = const_transpose_graph(2, 5, None);
        CudaFoldConstantTranspose
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(graph.nodes.values().all(|node| node.op_type != "Transpose"));
        assert_eq!(static_shape_of(&graph, transposed), vec![5, 2]);
    }

    #[test]
    fn leaves_transpose_of_non_constant() {
        // A Transpose whose input is a graph input (not an initializer) must not
        // be folded — its bytes are only known at run time.
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let input =
            graph.create_named_value("x", DataType::Float16, vec![Dim::Static(3), Dim::Static(4)]);
        graph.add_input(input);
        let out =
            graph.create_named_value("y", DataType::Float16, vec![Dim::Static(4), Dim::Static(3)]);
        let mut node = Node::new(NodeId(0), "Transpose", vec![Some(input)], vec![out]);
        node.attributes
            .insert("perm".into(), Attribute::Ints(vec![1, 0]));
        graph.insert_node(node);
        graph.add_output(out);

        CudaFoldConstantTranspose
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert_eq!(
            graph
                .nodes
                .values()
                .filter(|n| n.op_type == "Transpose")
                .count(),
            1
        );
        assert!(!graph.initializers.contains_key(&out));
    }

    #[test]
    fn leaves_sub_byte_constant_transpose() {
        // Sub-byte packed weights cannot be byte-permuted; the pass must skip
        // them rather than emit a wrong constant.
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let weight =
            graph.create_named_value("w", DataType::Int4, vec![Dim::Static(4), Dim::Static(4)]);
        graph.set_initializer(
            weight,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Int4,
                vec![4, 4],
                vec![0u8; 8],
            )),
        );
        let out =
            graph.create_named_value("wt", DataType::Int4, vec![Dim::Static(4), Dim::Static(4)]);
        let mut node = Node::new(NodeId(0), "Transpose", vec![Some(weight)], vec![out]);
        node.attributes
            .insert("perm".into(), Attribute::Ints(vec![1, 0]));
        graph.insert_node(node);
        let consumer_out =
            graph.create_named_value("o", DataType::Int4, vec![Dim::Static(4), Dim::Static(4)]);
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(out)],
            vec![consumer_out],
        ));
        graph.add_output(consumer_out);

        CudaFoldConstantTranspose
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert_eq!(
            graph
                .nodes
                .values()
                .filter(|n| n.op_type == "Transpose")
                .count(),
            1,
            "sub-byte Transpose must be left intact"
        );
    }

    #[test]
    fn folds_rank3_constant_transpose() {
        // Generic over rank: permute a [2, 3, 4] fp16 constant with perm [2, 0, 1].
        let (rows, mid, cols) = (2usize, 3usize, 4usize);
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let weight = graph.create_named_value(
            "w",
            DataType::Float16,
            vec![Dim::Static(rows), Dim::Static(mid), Dim::Static(cols)],
        );
        let mut bytes = Vec::new();
        for i in 0..rows * mid * cols {
            bytes.extend_from_slice(&half::f16::from_f32(i as f32).to_le_bytes());
        }
        graph.set_initializer(
            weight,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float16,
                vec![rows, mid, cols],
                bytes,
            )),
        );
        let out = graph.create_named_value(
            "wt",
            DataType::Float16,
            vec![Dim::Static(cols), Dim::Static(rows), Dim::Static(mid)],
        );
        let mut node = Node::new(NodeId(0), "Transpose", vec![Some(weight)], vec![out]);
        node.attributes
            .insert("perm".into(), Attribute::Ints(vec![2, 0, 1]));
        graph.insert_node(node);
        let consumer_out = graph.create_named_value(
            "o",
            DataType::Float16,
            vec![Dim::Static(cols), Dim::Static(rows), Dim::Static(mid)],
        );
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(out)],
            vec![consumer_out],
        ));

        CudaFoldConstantTranspose
            .run(&mut graph, &PassContext::new())
            .unwrap();
        let WeightRef::Inline(tensor) = graph.initializers.get(&out).unwrap() else {
            panic!("expected inline initializer");
        };
        assert_eq!(tensor.dims, vec![cols, rows, mid]);
        // out[c, r, m] == in[r, m, c] == r*(mid*cols) + m*cols + c
        for c in 0..cols {
            for r in 0..rows {
                for m in 0..mid {
                    let out_flat = (c * rows + r) * mid + m;
                    let expected = (r * mid + m) * cols + c;
                    assert_eq!(f16_at(&tensor.data, out_flat), expected as f32);
                }
            }
        }
    }

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

    // Representative gate/up (K=hidden, N=intermediate) shapes. These are test
    // fixtures only — the pass itself gates on structure + capability, never on
    // these dimensions. `QWEN_*` is the 762 tok/s non-regression shape; the
    // others exercise unrelated architectures to prove genericity.
    const QWEN_GATE_UP_K: usize = 896;
    const QWEN_GATE_UP_N: usize = 4864;
    // (K, N) pairs from non-Qwen architectures, all block-32/4-bit/fp16.
    const NON_QWEN_GATE_UP_SHAPES: [(usize, usize); 2] = [
        (2048, 5632),  // Llama-ish: hidden 2048, intermediate 5632
        (2048, 16384), // Gemma-ish: hidden 2048, intermediate 16384
    ];

    fn projection(graph: &mut Graph, tag: &str, x: ValueId, k: usize, n: usize) -> ValueId {
        projection_dtype(graph, tag, x, k, n, DataType::Float16)
    }

    fn projection_dtype(
        graph: &mut Graph,
        tag: &str,
        x: ValueId,
        k: usize,
        n: usize,
        dtype: DataType,
    ) -> ValueId {
        let scale_bytes = if dtype == DataType::Float16 { 2 } else { 4 };
        let packed = vec1d(
            graph,
            &format!("{tag}_packed"),
            DataType::Uint8,
            n * (k / 32) * 16,
        );
        let scales = vec1d(graph, &format!("{tag}_scales"), dtype, n * (k / 32));
        let out = value(graph, &format!("{tag}_out"), dtype, n);
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
                vec![0u8; n * (k / 32) * scale_bytes],
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
        gate_up_graph_dtype_impl(k, n, shared, DataType::Float16)
    }

    fn gate_up_graph_dtype(k: usize, n: usize, dtype: DataType) -> Graph {
        gate_up_graph_dtype_impl(k, n, true, dtype)
    }

    fn gate_up_graph_dtype_impl(k: usize, n: usize, shared: bool, dtype: DataType) -> Graph {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);
        let x = value(&mut graph, "x", dtype, k);
        graph.add_input(x);
        let up_x = if shared {
            x
        } else {
            let x2 = value(&mut graph, "x2", dtype, k);
            graph.add_input(x2);
            x2
        };
        let gate_out = projection_dtype(&mut graph, "gate", x, k, n, dtype);
        let up_out = projection_dtype(&mut graph, "up", up_x, k, n, dtype);
        let out = value(&mut graph, "output", dtype, n);
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

    /// A gate/up graph where the two projections have *different* output widths
    /// (`gate.N = n_gate`, `up.N = n_up`). Such a pair cannot feed one
    /// elementwise `Mul`, so it must never fuse.
    fn gate_up_graph_asymmetric(k: usize, n_gate: usize, n_up: usize) -> Graph {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);
        let x = value(&mut graph, "x", DataType::Float16, k);
        graph.add_input(x);
        let gate_out = projection(&mut graph, "gate", x, k, n_gate);
        let up_out = projection(&mut graph, "up", x, k, n_up);
        let out = value(&mut graph, "output", DataType::Float16, n_gate);
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
        let mut graph = gate_up_graph(QWEN_GATE_UP_K, QWEN_GATE_UP_N, true);
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
            Some(QWEN_GATE_UP_N as i64)
        );
        // The fused node keeps the Mul's output value, so downstream binding is
        // stable.
        let out = fused.outputs[0];
        assert_eq!(graph.outputs, vec![out]);
        assert_eq!(graph.value(out).name.as_deref(), Some("output"));
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn fuses_paired_gate_up_swiglu_for_non_qwen_shapes() {
        // Genericity proof: the identical structural + capability gate must fire
        // on architectures with dimensions unrelated to Qwen, because it never
        // looks at K/N magnitudes — only block-32/4-bit/fp16 compatibility and
        // the paired op/topology.
        for (k, n) in NON_QWEN_GATE_UP_SHAPES {
            let mut graph = gate_up_graph(k, n, true);
            CudaGateUpSwiGluFusion
                .run(&mut graph, &PassContext::new())
                .unwrap();
            assert_eq!(
                graph.num_nodes(),
                1,
                "gate/up SwiGLU must fuse for non-Qwen shape K={k}, N={n}"
            );
            let fused = graph.nodes.values().next().unwrap();
            assert_eq!(
                fused
                    .attr(GATE_UP_SWIGLU_FUSION_ATTR)
                    .and_then(Attribute::as_int),
                Some(1),
                "fused marker missing for K={k}, N={n}"
            );
            assert_eq!(fused.attr("N").and_then(Attribute::as_int), Some(n as i64));
            assert!(graph.validate().is_ok());
        }
    }

    #[test]
    fn does_not_fuse_when_activation_differs() {
        let mut graph = gate_up_graph(QWEN_GATE_UP_K, QWEN_GATE_UP_N, false);
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

    /// Set an integer attribute on every `MatMulNBits` in the graph.
    fn set_matmul_attr(graph: &mut Graph, name: &str, value: i64) {
        let ids: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| (node.op_type == "MatMulNBits").then_some(id))
            .collect();
        for id in ids {
            graph
                .node_mut(id)
                .attributes
                .insert(name.into(), Attribute::Int(value));
        }
    }

    fn asserts_not_fused(graph: &Graph) {
        assert_eq!(
            graph.num_nodes(),
            3,
            "incompatible projections must stay separate"
        );
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.attr(GATE_UP_SWIGLU_FUSION_ATTR).is_none())
        );
    }

    #[test]
    fn does_not_fuse_incompatible_block_size() {
        // Correct structure but block_size != 32: the paired kernel's `lane>>2`
        // scale indexing only maps for block-32 quantization.
        let mut graph = gate_up_graph(QWEN_GATE_UP_K, QWEN_GATE_UP_N, true);
        set_matmul_attr(&mut graph, "block_size", 64);
        CudaGateUpSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        asserts_not_fused(&graph);
    }

    #[test]
    fn does_not_fuse_incompatible_bits() {
        // Correct structure but bits != 4: the paired kernel unpacks 4-bit
        // nibbles only.
        let mut graph = gate_up_graph(QWEN_GATE_UP_K, QWEN_GATE_UP_N, true);
        set_matmul_attr(&mut graph, "bits", 8);
        CudaGateUpSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        asserts_not_fused(&graph);
    }

    #[test]
    fn does_not_fuse_non_fp16_projection() {
        // fp32 activation/scales/output: the paired kernel is fp16-only.
        let mut graph = gate_up_graph_dtype(QWEN_GATE_UP_K, QWEN_GATE_UP_N, DataType::Float32);
        CudaGateUpSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        asserts_not_fused(&graph);
    }

    #[test]
    fn does_not_fuse_mismatched_output_width() {
        // Structurally impossible pairing: gate.N != up.N. Paired projections
        // feeding one elementwise Mul must share output width.
        let mut graph =
            gate_up_graph_asymmetric(QWEN_GATE_UP_K, QWEN_GATE_UP_N, QWEN_GATE_UP_N / 2);
        CudaGateUpSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        // The mismatched-width Mul is not even a valid SwiGLU pairing, so the
        // three nodes are untouched.
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.attr(GATE_UP_SWIGLU_FUSION_ATTR).is_none())
        );
    }

    #[test]
    fn does_not_fuse_untagged_mul() {
        let mut graph = gate_up_graph(QWEN_GATE_UP_K, QWEN_GATE_UP_N, true);
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
        let x = value(&mut graph, "x", DataType::Float16, QWEN_GATE_UP_K);
        graph.add_input(x);
        let gate_out = projection(&mut graph, "gate", x, QWEN_GATE_UP_K, QWEN_GATE_UP_N);
        let up_out = projection(&mut graph, "up", x, QWEN_GATE_UP_K, QWEN_GATE_UP_N);
        let silu_out = value(&mut graph, "silu", DataType::Float16, QWEN_GATE_UP_N);
        let out = value(&mut graph, "output", DataType::Float16, QWEN_GATE_UP_N);
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

    // === SkipSimplifiedLayerNormalization <-> MatMulNBits fold ===

    /// Build the decode hot chain the fusion targets:
    ///
    /// ```text
    ///   preceding MatMulNBits (down, N == norm_size) ─┐
    ///                                                 ├─► SkipSimplifiedLayerNormalization
    ///   residual (fp16 activation) ────────────────────┘        │ normalized │ sum
    ///                                                            ▼            ▼
    ///                                       following MatMulNBits (K==norm_size)   Identity → out
    /// ```
    ///
    /// `norm_size` is the hidden width, and the following GEMV runs `K==norm_size
    /// → N==following_n`. Nothing is named after any architecture. The middle
    /// `mean`/`inv_std` outputs are anonymous (unused), matching how the loader
    /// materializes omitted optional outputs.
    fn skip_rms_graph(norm_size: usize, following_n: usize) -> Graph {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);

        // Preceding GEMV: any activation width → the hidden width.
        let pre_x = value(&mut graph, "pre_x", DataType::Float16, norm_size + 128);
        graph.add_input(pre_x);
        let pre_out = projection(&mut graph, "pre", pre_x, norm_size + 128, norm_size);

        // The residual (skip) term is a plain fp16 activation of the hidden width.
        let residual = value(&mut graph, "residual", DataType::Float16, norm_size);
        graph.add_input(residual);

        // Gamma scale initializer `[norm_size]`.
        let gamma = vec1d(&mut graph, "gamma", DataType::Float16, norm_size);
        graph.set_initializer(
            gamma,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float16,
                vec![norm_size],
                vec![0u8; norm_size * 2],
            )),
        );

        let normalized = value(&mut graph, "normalized", DataType::Float16, norm_size);
        let stat_mean = graph.create_value(DataType::Float32, Vec::new());
        let stat_inv_std = graph.create_value(DataType::Float32, Vec::new());
        let sum = value(&mut graph, "sum", DataType::Float16, norm_size);
        let mut skip = Node::new(
            NodeId(0),
            "SkipSimplifiedLayerNormalization",
            vec![Some(pre_out), Some(residual), Some(gamma)],
            vec![normalized, stat_mean, stat_inv_std, sum],
        );
        skip.domain = MICROSOFT_DOMAIN.into();
        skip.attributes
            .insert("epsilon".into(), Attribute::Float(9.999_999e-7));
        graph.insert_node(skip);

        // Following GEMV normalizes and projects; K == norm_size, N == following_n.
        let post_out = projection(&mut graph, "post", normalized, norm_size, following_n);
        graph.add_output(post_out);

        // Keep the residual sum live so its rewiring onto the preceding GEMV
        // output is observable.
        let sum_sink = value(&mut graph, "sum_sink", DataType::Float16, norm_size);
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(sum)],
            vec![sum_sink],
        ));
        graph.add_output(sum_sink);

        graph
    }

    fn value_id_by_name(graph: &Graph, name: &str) -> ValueId {
        graph
            .values
            .iter()
            .find_map(|(id, v)| (v.name.as_deref() == Some(name)).then_some(id))
            .unwrap()
    }

    fn node_producing(graph: &Graph, output: ValueId) -> &Node {
        let producer = graph.value(output).producer.unwrap();
        graph.node(producer)
    }

    #[test]
    fn folds_skip_rmsnorm_into_neighbouring_gemvs() {
        // Hidden at the size floor, following K(1280) <= N(1536): general variant.
        let mut graph = skip_rms_graph(RMSNORM_FUSION_MIN_HIDDEN, RMSNORM_FUSION_MIN_HIDDEN + 256);
        let pre_out = value_id_by_name(&graph, "pre_out");
        let residual = value_id_by_name(&graph, "residual");
        let gamma = value_id_by_name(&graph, "gamma");

        CudaSkipRmsNormMatMulFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();

        // The standalone norm launch is gone.
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.op_type != "SkipSimplifiedLayerNormalization"),
            "the SkipSimplifiedLayerNormalization must be deleted"
        );

        // Preceding GEMV now folds the residual into its bias slot (post-round).
        let preceding = node_producing(&graph, pre_out);
        assert_eq!(preceding.op_type, "MatMulNBits");
        assert_eq!(
            preceding
                .attr(MATMUL_NBITS_FOLDED_BIAS_ATTR)
                .and_then(Attribute::as_int),
            Some(1)
        );
        assert_eq!(preceding.inputs.len(), 6);
        assert_eq!(preceding.inputs[5], Some(residual), "residual at slot 5");
        assert!(preceding.inputs[3].is_none() && preceding.inputs[4].is_none());

        // Following GEMV carries the norm prologue: gamma at slot 6, markers set,
        // and its activation input is now the preceding residual sum.
        let post_out = value_id_by_name(&graph, "post_out");
        let following = node_producing(&graph, post_out);
        assert_eq!(following.inputs[0], Some(pre_out), "activation is residual sum");
        assert_eq!(following.inputs.get(6).copied().flatten(), Some(gamma));
        assert_eq!(
            following
                .attr(MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR)
                .and_then(Attribute::as_int),
            Some(1)
        );
        assert_eq!(
            following
                .attr(MATMUL_NBITS_RMSNORM_EPSILON_ATTR)
                .and_then(Attribute::as_float),
            Some(9.999_999e-7)
        );

        // The residual-sum consumer (Identity) is rewired onto the preceding GEMV.
        let identity = graph
            .nodes
            .values()
            .find(|node| node.op_type == "Identity")
            .unwrap();
        assert_eq!(identity.inputs[0], Some(pre_out));

        assert!(graph.validate().is_ok());
    }

    #[test]
    fn leaves_skip_rmsnorm_when_hidden_not_multiple_of_128() {
        // 1288 % 128 != 0 → warp_half4 byte-identity does not hold, so no fusion.
        // Kept above the size floor so the `% 128` check is the sole reason.
        let mut graph = skip_rms_graph(1288, RMSNORM_FUSION_MIN_HIDDEN + 256);
        CudaSkipRmsNormMatMulFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(
            graph
                .nodes
                .values()
                .any(|node| node.op_type == "SkipSimplifiedLayerNormalization"),
            "an unaligned hidden size must keep the standalone norm"
        );
    }

    #[test]
    fn leaves_skip_rmsnorm_when_following_is_down_variant() {
        // Following K(1280) > N(1152): the tall-skinny down variant has no
        // prologue. Hidden is at the floor so the down variant is the sole reason.
        let mut graph = skip_rms_graph(RMSNORM_FUSION_MIN_HIDDEN, RMSNORM_FUSION_MIN_HIDDEN - 128);
        CudaSkipRmsNormMatMulFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(
            graph
                .nodes
                .values()
                .any(|node| node.op_type == "SkipSimplifiedLayerNormalization"),
            "a down-variant following GEMV must block the fusion"
        );
    }

    #[test]
    fn leaves_skip_rmsnorm_with_norm_bias() {
        // A norm bias (4th input) breaks the no-bias warp_half4 contract. Hidden
        // is at the floor so the bias is the sole reason the fusion declines.
        let hidden = RMSNORM_FUSION_MIN_HIDDEN;
        let mut graph = skip_rms_graph(hidden, hidden + 256);
        let bias = vec1d(&mut graph, "norm_bias", DataType::Float16, hidden);
        graph.set_initializer(
            bias,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float16,
                vec![hidden],
                vec![0u8; hidden * 2],
            )),
        );
        let skip_id = graph
            .nodes
            .iter()
            .find_map(|(id, n)| {
                (n.op_type == "SkipSimplifiedLayerNormalization").then_some(id)
            })
            .unwrap();
        let mut skip = graph.node(skip_id).clone();
        skip.inputs.push(Some(bias));
        graph.replace_node(skip_id, skip);

        CudaSkipRmsNormMatMulFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(
            graph
                .nodes
                .values()
                .any(|node| node.op_type == "SkipSimplifiedLayerNormalization"),
            "a norm bias must block the fusion"
        );
    }

    #[test]
    fn leaves_skip_rmsnorm_when_preceding_gemv_shared() {
        // A second consumer of the preceding GEMV output means the residual
        // epilogue would drop an observable value; the fusion must decline.
        let hidden = RMSNORM_FUSION_MIN_HIDDEN;
        let mut graph = skip_rms_graph(hidden, hidden + 256);
        let pre_out = value_id_by_name(&graph, "pre_out");
        let sink = value(&mut graph, "pre_sink", DataType::Float16, hidden);
        graph.insert_node(Node::new(NodeId(0), "Neg", vec![Some(pre_out)], vec![sink]));
        graph.add_output(sink);

        CudaSkipRmsNormMatMulFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(
            graph
                .nodes
                .values()
                .any(|node| node.op_type == "SkipSimplifiedLayerNormalization"),
            "a shared preceding GEMV must block the fusion"
        );
    }

    #[test]
    fn leaves_skip_rmsnorm_when_broadcast_skip() {
        // A broadcast (non-dense) skip term is not covered by warp_half4.
        let hidden = RMSNORM_FUSION_MIN_HIDDEN;
        let mut graph = skip_rms_graph(hidden, hidden + 256);
        let residual = value_id_by_name(&graph, "residual");
        graph.value_mut(residual).shape = vec![Dim::Static(1), Dim::Static(1)];

        CudaSkipRmsNormMatMulFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(
            graph
                .nodes
                .values()
                .any(|node| node.op_type == "SkipSimplifiedLayerNormalization"),
            "a broadcast skip must block the fusion"
        );
    }

    #[test]
    fn folds_skip_rmsnorm_with_symbolic_batch_and_sequence_dims() {
        // Regression: the real decode graph carries symbolic batch/sequence dims
        // ([batch, sequence, hidden]); only the hidden dim is static. The fusion
        // must still fire — an earlier version required the whole shape static
        // and silently never matched in production.
        let mut graph = skip_rms_graph(RMSNORM_FUSION_MIN_HIDDEN, RMSNORM_FUSION_MIN_HIDDEN + 256);
        let batch = graph.create_symbol(Some("batch".into()));
        let sequence = graph.create_symbol(Some("sequence".into()));
        let symbolic = vec![
            Dim::Symbolic(batch),
            Dim::Symbolic(sequence),
            Dim::Static(RMSNORM_FUSION_MIN_HIDDEN),
        ];
        // Retype every hidden-width activation edge to the symbolic 3-D shape.
        for name in ["pre_out", "residual", "normalized", "sum"] {
            let id = value_id_by_name(&graph, name);
            graph.value_mut(id).shape = symbolic.clone();
        }

        CudaSkipRmsNormMatMulFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.op_type != "SkipSimplifiedLayerNormalization"),
            "symbolic batch/sequence dims must not block the fusion"
        );
        let post_out = value_id_by_name(&graph, "post_out");
        let following = node_producing(&graph, post_out);
        assert_eq!(
            following
                .attr(MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR)
                .and_then(Attribute::as_int),
            Some(1)
        );
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn skip_rmsnorm_fires_through_full_cuda_pass_list() {
        // End-to-end through the registered CUDA passes: the fusion is the last
        // pass and must fire on the eligible chain.
        let mut graph = skip_rms_graph(RMSNORM_FUSION_MIN_HIDDEN, RMSNORM_FUSION_MIN_HIDDEN + 256);
        for pass in cuda_optimization_passes() {
            pass.run(&mut graph, &PassContext::new()).unwrap();
        }
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.op_type != "SkipSimplifiedLayerNormalization"),
            "the fusion must fire through the full pass list"
        );
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn gate_leaves_skip_rmsnorm_below_hidden_floor() {
        // One reduction chunk (128) below the size floor: an otherwise fully
        // fusable chain, blocked solely because the projected benefit is negative
        // for such a small hidden (the serial prologue recompute dominates the
        // ~free graph-captured standalone launch it would remove).
        let hidden = RMSNORM_FUSION_MIN_HIDDEN - RMSNORM_FUSION_WARP_HALF4_MULTIPLE;
        let mut graph = skip_rms_graph(hidden, hidden + 256);
        CudaSkipRmsNormMatMulFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(
            graph
                .nodes
                .values()
                .any(|node| node.op_type == "SkipSimplifiedLayerNormalization"),
            "a hidden below the size floor must keep the standalone norm"
        );
    }

    #[test]
    fn gate_folds_skip_rmsnorm_at_hidden_floor() {
        // Exactly at the floor the same chain fuses, proving the boundary is the
        // only thing the smaller case tripped (not any structural mismatch).
        let hidden = RMSNORM_FUSION_MIN_HIDDEN;
        let mut graph = skip_rms_graph(hidden, hidden + 256);
        CudaSkipRmsNormMatMulFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.op_type != "SkipSimplifiedLayerNormalization"),
            "a hidden at the size floor must fuse"
        );
        assert!(graph.validate().is_ok());
    }

    /// Build the post-attention decode chain the fan-out-2 fold targets:
    /// `o_proj MatMulNBits (N == hidden)` → `SkipSimplifiedLayerNormalization`
    /// → gate + up `MatMulNBits` → `Silu(gate)` → `Mul(silu, up)`. Running the
    /// full CUDA pass list first collapses gate+up into one SwiGLU node, then the
    /// skip-rmsnorm fold must route its RMS prologue through that single node.
    fn post_attention_swiglu_graph(hidden: usize, intermediate: usize) -> Graph {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);

        let pre_x = value(&mut graph, "pre_x", DataType::Float16, hidden + 128);
        graph.add_input(pre_x);
        let pre_out = projection(&mut graph, "pre", pre_x, hidden + 128, hidden);

        let residual = value(&mut graph, "residual", DataType::Float16, hidden);
        graph.add_input(residual);

        let gamma = vec1d(&mut graph, "gamma", DataType::Float16, hidden);
        graph.set_initializer(
            gamma,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float16,
                vec![hidden],
                vec![0u8; hidden * 2],
            )),
        );

        let normalized = value(&mut graph, "normalized", DataType::Float16, hidden);
        let stat_mean = graph.create_value(DataType::Float32, Vec::new());
        let stat_inv_std = graph.create_value(DataType::Float32, Vec::new());
        let sum = value(&mut graph, "sum", DataType::Float16, hidden);
        let mut skip = Node::new(
            NodeId(0),
            "SkipSimplifiedLayerNormalization",
            vec![Some(pre_out), Some(residual), Some(gamma)],
            vec![normalized, stat_mean, stat_inv_std, sum],
        );
        skip.domain = MICROSOFT_DOMAIN.into();
        skip.attributes
            .insert("epsilon".into(), Attribute::Float(9.999_999e-7));
        graph.insert_node(skip);

        // Fan-out 2: the normalized output feeds both the gate and up projection.
        let gate_out = projection(&mut graph, "gate", normalized, hidden, intermediate);
        let up_out = projection(&mut graph, "up", normalized, hidden, intermediate);
        let silu_out = value(&mut graph, "silu", DataType::Float16, intermediate);
        let out = value(&mut graph, "output", DataType::Float16, intermediate);
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

        let sum_sink = value(&mut graph, "sum_sink", DataType::Float16, hidden);
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(sum)],
            vec![sum_sink],
        ));
        graph.add_output(sum_sink);

        graph
    }

    #[test]
    fn folds_skip_rmsnorm_into_gate_up_swiglu_node() {
        // Hidden at the size floor, intermediate > hidden so the paired gate/up
        // node keeps the general scales_f16 entry (K <= N).
        let hidden = RMSNORM_FUSION_MIN_HIDDEN;
        let intermediate = hidden * 2;
        let mut graph = post_attention_swiglu_graph(hidden, intermediate);

        for pass in cuda_optimization_passes() {
            pass.run(&mut graph, &PassContext::new()).unwrap();
        }

        // Norm is gone and gate/up are collapsed into one SwiGLU node.
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.op_type != "SkipSimplifiedLayerNormalization"),
            "the standalone norm must be folded away"
        );
        let swiglu = graph
            .nodes
            .values()
            .find(|node| node.attr(GATE_UP_SWIGLU_FUSION_ATTR).is_some())
            .expect("a fused gate/up SwiGLU node must exist");

        let pre_out = value_id_by_name(&graph, "pre_out");
        let gamma = value_id_by_name(&graph, "gamma");
        // The SwiGLU node now carries the RMS prologue: activation is the
        // residual sum, gamma lands at slot 5 (after [x, W_gate, Sg, W_up, Su]),
        // and the prologue markers are set. A single reduction now serves both
        // projections.
        assert_eq!(
            swiglu.inputs[0],
            Some(pre_out),
            "activation is the preceding residual sum"
        );
        assert_eq!(swiglu.inputs.len(), 6, "gamma appended at slot 5");
        assert_eq!(swiglu.inputs.get(5).copied().flatten(), Some(gamma));
        assert_eq!(
            swiglu
                .attr(MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR)
                .and_then(Attribute::as_int),
            Some(1)
        );
        assert_eq!(
            swiglu
                .attr(MATMUL_NBITS_RMSNORM_EPSILON_ATTR)
                .and_then(Attribute::as_float),
            Some(9.999_999e-7)
        );
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn gate_up_swiglu_node_stays_unfused_below_hidden_floor() {
        // The same fan-out-2 chain, one reduction chunk below the floor: the
        // SwiGLU node must keep the standalone norm (benefit is negative for a
        // tiny hidden even though a single fused reduction would serve both).
        let hidden = RMSNORM_FUSION_MIN_HIDDEN - RMSNORM_FUSION_WARP_HALF4_MULTIPLE;
        let intermediate = hidden * 2;
        let mut graph = post_attention_swiglu_graph(hidden, intermediate);

        for pass in cuda_optimization_passes() {
            pass.run(&mut graph, &PassContext::new()).unwrap();
        }

        assert!(
            graph
                .nodes
                .values()
                .any(|node| node.op_type == "SkipSimplifiedLayerNormalization"),
            "below the floor the standalone norm must survive"
        );
        let swiglu = graph
            .nodes
            .values()
            .find(|node| node.attr(GATE_UP_SWIGLU_FUSION_ATTR).is_some())
            .expect("the gate/up pair still fuses on its own");
        assert!(
            swiglu.attr(MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR).is_none(),
            "the SwiGLU node must not carry an RMS prologue below the floor"
        );
    }

    #[test]
    fn folds_chained_blocks_sharing_residual_sum() {
        // Two chained decoder blocks: block 0's norm residual sum feeds block 1's
        // norm as its skip/residual. Both norms are independently fusable, and
        // applying block 0's fold rewires+GCs the shared sum value — so block 1's
        // fold must resolve its (now-redirected) residual instead of referencing
        // the deleted value. This is the graph shape that dangled before the
        // redirect map. Every value stays live and both norms fold.
        let hidden = RMSNORM_FUSION_MIN_HIDDEN;
        let following_n = hidden + 256;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);

        // The residual stream entering block 0.
        let res_in = value(&mut graph, "res_in", DataType::Float16, hidden);
        graph.add_input(res_in);

        let gamma0 = vec1d(&mut graph, "gamma0", DataType::Float16, hidden);
        let gamma1 = vec1d(&mut graph, "gamma1", DataType::Float16, hidden);
        for g in [gamma0, gamma1] {
            graph.set_initializer(
                g,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Float16,
                    vec![hidden],
                    vec![0u8; hidden * 2],
                )),
            );
        }

        // Block 0: preceding GEMV → SkipSLN(input, skip=res_in) → following GEMV.
        let x0 = value(&mut graph, "x0", DataType::Float16, hidden + 128);
        graph.add_input(x0);
        let pre0 = projection(&mut graph, "pre0", x0, hidden + 128, hidden);
        let norm0 = value(&mut graph, "norm0", DataType::Float16, hidden);
        let mean0 = graph.create_value(DataType::Float32, Vec::new());
        let inv0 = graph.create_value(DataType::Float32, Vec::new());
        let sum0 = value(&mut graph, "sum0", DataType::Float16, hidden);
        let mut skip0 = Node::new(
            NodeId(0),
            "SkipSimplifiedLayerNormalization",
            vec![Some(pre0), Some(res_in), Some(gamma0)],
            vec![norm0, mean0, inv0, sum0],
        );
        skip0.domain = MICROSOFT_DOMAIN.into();
        skip0
            .attributes
            .insert("epsilon".into(), Attribute::Float(1e-6));
        graph.insert_node(skip0);
        let post0 = projection(&mut graph, "post0", norm0, hidden, following_n);
        graph.add_output(post0);

        // Block 1: preceding GEMV → SkipSLN(input, skip=sum0) → following GEMV.
        // Block 1's preceding consumes its own activation so block 0's normalized
        // output keeps fan-out 1; only the residual sum (sum0) is shared.
        let x1 = value(&mut graph, "x1", DataType::Float16, hidden + 128);
        graph.add_input(x1);
        let pre1 = projection(&mut graph, "pre1", x1, hidden + 128, hidden);
        let norm1 = value(&mut graph, "norm1", DataType::Float16, hidden);
        let mean1 = graph.create_value(DataType::Float32, Vec::new());
        let inv1 = graph.create_value(DataType::Float32, Vec::new());
        let sum1 = value(&mut graph, "sum1", DataType::Float16, hidden);
        let mut skip1 = Node::new(
            NodeId(0),
            "SkipSimplifiedLayerNormalization",
            vec![Some(pre1), Some(sum0), Some(gamma1)],
            vec![norm1, mean1, inv1, sum1],
        );
        skip1.domain = MICROSOFT_DOMAIN.into();
        skip1
            .attributes
            .insert("epsilon".into(), Attribute::Float(1e-6));
        graph.insert_node(skip1);
        let post1 = projection(&mut graph, "post1", norm1, hidden, following_n);
        graph.add_output(post1);
        // Keep block 1's residual sum live via an Identity sink (not a graph
        // output, which the fusion intentionally refuses to fold).
        let sum1_sink = value(&mut graph, "sum1_sink", DataType::Float16, hidden);
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(sum1)],
            vec![sum1_sink],
        ));
        graph.add_output(sum1_sink);

        CudaSkipRmsNormMatMulFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();

        // Both norms fold and the graph stays structurally valid.
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.op_type != "SkipSimplifiedLayerNormalization"),
            "both chained norms must fold"
        );
        assert!(graph.validate().is_ok(), "no dangling shared residual value");

        // Block 1's preceding GEMV folds the *redirected* residual: sum0 became
        // block 0's preceding output (pre0_out), so pre1 must fold pre0_out.
        let pre0_out = value_id_by_name(&graph, "pre0_out");
        let pre1_out = value_id_by_name(&graph, "pre1_out");
        let pre1_node = node_producing(&graph, pre1_out);
        assert_eq!(
            pre1_node.inputs.get(5).copied().flatten(),
            Some(pre0_out),
            "block 1 folds the redirected residual (block 0's preceding output)"
        );
    }
}