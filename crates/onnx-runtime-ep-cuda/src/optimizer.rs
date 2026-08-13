use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, NodeId, TensorData, ValueId, WeightRef, static_shape,
};
use onnx_runtime_optimizer::{
    OptimizationPass, OptimizerError, PassContext, Result as OptimizerResult,
};

pub(crate) const SILU_MUL_FUSION_ATTR: &str = "_cuda_silu_mul";
pub(crate) const DECOMPOSED_SILU_ATTR: &str = "_cuda_decomposed_silu";

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

/// Lower an exact `x * Sigmoid(x)` pair to the CUDA EP's fused SiLU kernel.
///
/// Fires on fp16 **and** bf16 activations: the fused SwiGLU epilogue has a
/// byte-exact decomposed kernel for each (`decomposed_silu_mul_{f16,bf16}`),
/// so collapsing the standalone `Sigmoid` + `Mul(x, sigmoid)` glue keeps greedy
/// tokens byte-identical. bf16 decoders (e.g. Muse-Glimmer-30B) export SwiGLU as
/// that standalone pair, so bf16 support is what lets the per-layer glue node
/// collapse into the already-landed fused launch.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaSiluFusion;

pub(crate) fn cuda_optimization_passes() -> Vec<Box<dyn OptimizationPass>> {
    vec![
        Box::new(CudaSiluFusion),
        Box::new(CudaFoldConstantTranspose),
        Box::new(CudaFoldConstantCast),
        // Fuse the per-layer Q/K/V projections into one wider MatMulNBits while
        // the projections are still pristine three/four-input GEMVs (before the
        // norm-into-matmul fusion could add a gamma slot). Reuses the existing
        // MatMulNBits + Split kernels; no new kernel required.
        Box::new(CudaQkvProjectionFusion),
        // Runs before the fusions so they see the fp16-native normalization
        // form (no `Cast` wrappers) that the rest of the pipeline expects.
        Box::new(CudaDropNormalizationCasts),
        Box::new(CudaMatMulNBitsBiasFusion),
        Box::new(CudaSwiGluFusion),
        Box::new(CudaGateUpSwiGluFusion),
        Box::new(CudaSkipRmsNormMatMulFusion),
        // Lower a capture-unsafe LongRoPE-style `If(Greater, const, const)` cos/sin
        // cache selector into an on-device `Where`, collapsing the per-token host
        // cond readback / graph split so decode captures as a single graph.
        Box::new(CudaOnDeviceConstantSelect),
    ]
}

impl OptimizationPass for CudaSiluFusion {
    fn name(&self) -> &str {
        "CudaSiluFusion"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> OptimizerResult<()> {
        let sigmoid_ids: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.op_type == "Sigmoid"
                    && node.is_default_domain()
                    && node.inputs.len() == 1
                    && node.outputs.len() == 1)
                    .then_some(id)
            })
            .collect();
        let mut changed = false;

        for sigmoid_id in sigmoid_ids {
            let Some(sigmoid) = graph.try_node(sigmoid_id) else {
                continue;
            };
            let Some(x) = sigmoid.inputs[0] else {
                continue;
            };
            // The fused SwiGLU epilogue has a byte-exact decomposed kernel for
            // both fp16 (`decomposed_silu_mul_f16`) and bf16
            // (`decomposed_silu_mul_bf16`): each rounds the intermediate
            // `sigmoid` and the `x * sigmoid` product back to the narrow dtype
            // exactly as the standalone `Sigmoid`/`Mul` ops do, so greedy tokens
            // stay byte-identical. bf16 decoders such as Muse-Glimmer-30B export
            // the SwiGLU as a standalone `Sigmoid` + `Mul(x, sigmoid)` pair in
            // their native bf16 stream; without bf16 here the ~2 glue nodes per
            // layer never collapse into the fused launch.
            if !matches!(graph.value(x).dtype, DataType::Float16 | DataType::BFloat16) {
                continue;
            }
            let sigmoid_output = sigmoid.outputs[0];
            if graph.outputs.contains(&sigmoid_output) {
                continue;
            }
            let consumers = graph.consumers(sigmoid_output);
            if consumers.len() != 1 {
                continue;
            }
            let mul_id = consumers[0];
            let mul = graph.node(mul_id);
            if mul.op_type != "Mul"
                || !mul.is_default_domain()
                || mul.inputs.len() != 2
                || mul.outputs.len() != 1
                || !((mul.inputs[0] == Some(x) && mul.inputs[1] == Some(sigmoid_output))
                    || (mul.inputs[1] == Some(x) && mul.inputs[0] == Some(sigmoid_output)))
            {
                continue;
            }

            let mut silu = mul.clone();
            silu.op_type = "Silu".to_string();
            silu.domain = MICROSOFT_DOMAIN.to_string();
            silu.version = None;
            silu.inputs = vec![Some(x)];
            silu.attributes.clear();
            silu.attributes
                .insert(DECOMPOSED_SILU_ATTR.into(), Attribute::Int(1));
            graph.replace_node(mul_id, silu);
            graph.remove_node(sigmoid_id);
            graph
                .opset_imports
                .entry(MICROSOFT_DOMAIN.to_string())
                .or_insert(1);
            changed = true;
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

/// Drop the redundant `Cast` pairs that some exporters (e.g. Phi-4-mini,
/// Muse-Glimmer) wrap around every simplified-layer-norm / RMS-norm, running the
/// norm directly on its narrow (fp16 or bf16) activations instead.
///
/// ## The pattern
///
/// Certain decoders export each `SimplifiedLayerNormalization` /
/// `SkipSimplifiedLayerNormalization` / `RMSNormalization` in fp32: each narrow
/// (fp16 or bf16) activation input is preceded by a `Cast(narrow → fp32)`, and
/// each fp32 result is followed by a `Cast(fp32 → narrow)` back to the residual
/// stream. A 32-layer fp16 decoder emits ~256 of these tiny `Cast` kernels per
/// token; a 52-layer bf16 decoder such as Muse-Glimmer emits ~624 (a `Cast`
/// pair around every one of its 312 RMS-norms) — a per-token launch/round-trip
/// cost that is pure overhead. Models like Qwen2.5 instead run the same norm in
/// their native storage dtype with **no** surrounding casts.
///
/// ## Where the win lands
///
/// Removing the casts shrinks the graph and drops hundreds of kernels per
/// token. On the **eager** decode path each removed kernel is a saved launch
/// (measured on Phi-4-mini/H200: ~25 %). On a **CUDA-graph captured** decode
/// step, replay amortizes per-kernel launch overhead, so for a norm count that
/// is a small slice of the graph (Phi's fp16 path) captured throughput is flat
/// within noise. But when the cast kernels themselves dominate captured GPU
/// time — Muse-Glimmer's 626 casts/token are ~40 % of decode, each reading and
/// writing the full hidden state — removing them removes real memory traffic
/// from every replay, which is a direct captured-path speedup.
///
/// ## The rewrite
///
/// For a matching norm this pass rewires each activation input to the narrow
/// value feeding its `Cast`, retypes the consumed norm outputs to that same
/// narrow dtype, and bypasses/deletes the surrounding `Cast` nodes. The CUDA
/// norm kernels already accept fp16/bf16 activations with fp32 accumulation (and
/// either a narrow or fp32 `gamma`/`scale`), so the fp32 scale weight is left
/// untouched — the arithmetic is the same fp32-accumulate / narrow-rounded
/// scheme those kernels use for natively fp16/bf16 models. (Muse-Glimmer's RMS
/// scale is a constant `Cast(weight_bf16 → fp32) + 1` expression on input 1,
/// which is not an activation index and so is left exactly as exported.)
///
/// ## Generality and safety
///
/// The rewrite is driven purely by topology + tensor dtypes, never by model
/// identity:
/// * the node is a simplified-layer-norm / RMS-norm in its expected domain;
/// * every activation data input is produced by a `Cast(→ fp32)` whose source
///   is the **same** narrow float dtype (fp16 or bf16), and weights are
///   producer-less initializers or non-activation inputs, so they never match
///   and are left alone;
/// * every consumed norm output feeds only `Cast(→ narrow)` nodes (matching the
///   activation dtype) and is not a graph output;
/// * an optional norm bias is required absent (kept conservative).
///
/// When any condition fails the norm is left exactly as exported. Input `Cast`
/// nodes are deleted only once they are provably orphaned, so a `Cast` shared
/// with another consumer stays intact.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaDropNormalizationCasts;

/// Environment opt-out for [`CudaDropNormalizationCasts`], mirroring the other
/// CUDA-fusion switches. Any value other than unset/empty/`0` restores the
/// exact exported cast-wrapped normalization form (for A/B measurement or
/// rollback).
const NORM_CAST_FOLD_DISABLE_ENV: &str = "ONNX_GENAI_CUDA_DISABLE_NORM_CAST_FOLD";

fn norm_cast_fold_disabled() -> bool {
    std::env::var_os(NORM_CAST_FOLD_DISABLE_ENV)
        .is_some_and(|value| value != "0" && !value.is_empty())
}

/// A planned rewrite of one cast-wrapped normalization node.
struct NormCastFoldPlan {
    node_id: NodeId,
    /// The narrow activation storage dtype (fp16 or bf16) the norm is rewired to
    /// run on directly. Every un-cast activation input and every retyped output
    /// share this dtype.
    narrow_dtype: DataType,
    /// Full input vector with each cast activation input rewired to the pre-cast
    /// narrow (fp16/bf16) source value.
    new_inputs: Vec<Option<ValueId>>,
    /// Norm outputs to retype from fp32 to the narrow activation dtype (the
    /// consumed float results).
    retyped_outputs: Vec<ValueId>,
    /// `(cast_output, norm_output, cast_node)` triples: downstream uses of
    /// `cast_output` are moved onto `norm_output` and the `Cast` is deleted.
    output_cast_bypass: Vec<(ValueId, ValueId, NodeId)>,
    /// Input `Cast` nodes to delete once they are left with no consumers.
    dead_input_casts: Vec<NodeId>,
}

impl OptimizationPass for CudaDropNormalizationCasts {
    fn name(&self) -> &str {
        "CudaDropNormalizationCasts"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> OptimizerResult<()> {
        if norm_cast_fold_disabled() {
            return Ok(());
        }
        // Fold one norm at a time, re-planning against the live graph. Adjacent
        // layers share a residual `Cast`, so folding one layer moves the value
        // its neighbour's plan referenced; re-planning after every rewrite keeps
        // every source value current. Each fold deletes at least one `Cast`, so
        // the loop strictly shrinks the graph and always terminates.
        let mut changed = false;
        loop {
            let candidates: Vec<NodeId> = graph
                .nodes
                .iter()
                .filter_map(|(id, node)| Self::activation_input_indices(node).map(|_| id))
                .collect();
            let Some(plan) = candidates
                .into_iter()
                .find_map(|id| self.plan_fold(graph, id))
            else {
                break;
            };
            self.apply_fold(graph, plan);
            changed = true;
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

impl CudaDropNormalizationCasts {
    fn apply_fold(&self, graph: &mut Graph, plan: NormCastFoldPlan) {
        // 1. Rewire the norm onto its pre-cast fp16 activation inputs.
        let mut node = graph.node(plan.node_id).clone();
        node.inputs = plan.new_inputs;
        // `RMSNormalization` (ai.onnx opset 23) is the one supported norm whose
        // output element type follows the *scale* (`V`), not the activation `X`
        // (`T`) — see the ONNX schema and `rms_norm` in the shape-inference
        // crate. Muse-Glimmer keeps an f32 scale (`Add(Cast(weight_bf16→f32), 1)`)
        // even for its bf16 activations, so re-running shape inference after this
        // pass would clobber the narrow output we just retyped back to f32 (the
        // scale dtype), leaving a bf16-in / f32-out norm the kernel rejects.
        // `SimplifiedLayerNormalization` is the identical operation (RMS scaling,
        // no mean subtraction) whose output follows `X`; both ops map to the same
        // fused `RmsNormKernel` on the CUDA and CPU EPs (see
        // `kernels/mod.rs`). Converting the folded node preserves the exact
        // arithmetic (fp32 accumulation, f32 scale, single round-to-nearest-even
        // on the narrow output — byte-identical to the removed `Cast`) while
        // letting inference keep the narrow output dtype.
        if node.op_type == "RMSNormalization" && matches!(node.domain.as_str(), "" | "ai.onnx") {
            node.op_type = "SimplifiedLayerNormalization".into();
        }
        graph.replace_node(plan.node_id, node);

        // 2. Retype the consumed float outputs to the narrow activation dtype to
        //    match the inputs.
        for output in plan.retyped_outputs {
            graph.value_mut(output).dtype = plan.narrow_dtype;
        }

        // 3. Bypass and delete the output `Cast` nodes.
        for (cast_output, norm_output, cast_id) in plan.output_cast_bypass {
            graph.replace_all_uses(cast_output, norm_output);
            graph.remove_node(cast_id);
        }

        // 4. Delete the input `Cast` nodes that are now orphaned.
        for cast_id in plan.dead_input_casts {
            if let Some(cast) = graph.try_node(cast_id) {
                let cast_output = *cast.outputs.first().expect("Cast has one output");
                if graph.consumers(cast_output).is_empty()
                    && !graph.value(cast_output).is_graph_output
                {
                    graph.remove_node(cast_id);
                }
            }
        }
    }
}

impl CudaDropNormalizationCasts {
    /// The activation (non-weight) input indices for a supported simplified
    /// layer-norm, or `None` if the node is not one. `SkipSimplified*` takes
    /// `(input, skip, gamma[, bias])`; the plain `Simplified*` / `RMSNormalization`
    /// take `(X, scale)`. Only the data inputs are candidates for un-casting; the
    /// weight (`gamma` / `scale`) and any bias are left untouched (they may be
    /// producer-less initializers or a constant `weight + 1` scale expression).
    fn activation_input_indices(node: &onnx_runtime_ir::Node) -> Option<&'static [usize]> {
        match (node.op_type.as_str(), node.domain.as_str()) {
            ("SkipSimplifiedLayerNormalization", MICROSOFT_DOMAIN) => Some(&[0, 1]),
            ("SimplifiedLayerNormalization", "" | "ai.onnx") => Some(&[0]),
            ("RMSNormalization", "" | "ai.onnx") => Some(&[0]),
            _ => None,
        }
    }

    /// If `value` is produced by a `Cast(narrow → fp32)` where `narrow` is a
    /// narrow float storage dtype (fp16 or bf16), return the `Cast` node, its
    /// narrow source value, and that narrow dtype.
    fn fp32_cast_from_narrow(
        &self,
        graph: &Graph,
        value: ValueId,
    ) -> Option<(NodeId, ValueId, DataType)> {
        let producer = graph.try_value(value)?.producer?;
        let node = graph.try_node(producer)?;
        if node.op_type != "Cast" || !matches!(node.domain.as_str(), "" | "ai.onnx") {
            return None;
        }
        if cast_target(node)? != DataType::Float32 {
            return None;
        }
        let source = node.inputs.first().copied().flatten()?;
        let source_dtype = graph.try_value(source)?.dtype;
        if source_dtype != DataType::Float16 && source_dtype != DataType::BFloat16 {
            return None;
        }
        Some((producer, source, source_dtype))
    }

    fn plan_fold(&self, graph: &Graph, node_id: NodeId) -> Option<NormCastFoldPlan> {
        let node = graph.try_node(node_id)?;
        let activation_indices = Self::activation_input_indices(node)?;

        // A norm bias (index 3 for the skip form) is left conservatively alone.
        if node.op_type == "SkipSimplifiedLayerNormalization"
            && node.inputs.get(3).copied().flatten().is_some()
        {
            return None;
        }

        // A narrowed `RMSNormalization` is rewritten to `SimplifiedLayerNormalization`
        // (identical arithmetic, output follows `X`) so shape inference keeps the
        // narrow output dtype. That synonym only carries the single normalized
        // output, so decline any `RMSNormalization` that also emits the optional
        // `InvStdDev` stat (whose f32 dtype the synonym's inference would not
        // preserve). Muse-Glimmer's decode norms emit only `Y`.
        if node.op_type == "RMSNormalization" && node.outputs.len() != 1 {
            return None;
        }

        // Every activation input must be a narrow (fp16/bf16) → fp32 `Cast`;
        // rewire it to the narrow source. All activation inputs must share the
        // same narrow dtype.
        let mut new_inputs = node.inputs.clone();
        let mut dead_input_casts = Vec::new();
        let mut narrow_dtype: Option<DataType> = None;
        for &index in activation_indices {
            let value = node.inputs.get(index).copied().flatten()?;
            let (cast_id, source, source_dtype) = self.fp32_cast_from_narrow(graph, value)?;
            if *narrow_dtype.get_or_insert(source_dtype) != source_dtype {
                return None;
            }
            new_inputs[index] = Some(source);
            dead_input_casts.push(cast_id);
        }
        let narrow_dtype = narrow_dtype?;

        // Every consumed float output must feed only `Cast(→ narrow)` nodes,
        // where `narrow` matches the activation dtype.
        let mut retyped_outputs = Vec::new();
        let mut output_cast_bypass = Vec::new();
        for &output in &node.outputs {
            if graph.value(output).is_graph_output {
                return None;
            }
            let consumers = graph.consumers(output);
            if consumers.is_empty() {
                continue;
            }
            for consumer in &consumers {
                let cast = graph.try_node(*consumer)?;
                if cast.op_type != "Cast" || !matches!(cast.domain.as_str(), "" | "ai.onnx") {
                    return None;
                }
                if cast_target(cast)? != narrow_dtype {
                    return None;
                }
                let cast_output = *cast.outputs.first()?;
                output_cast_bypass.push((cast_output, output, *consumer));
            }
            retyped_outputs.push(output);
        }

        // The primary (normalized) output must actually be one we retyped; the
        // kernel requires the output dtype to match the narrow input dtype.
        let primary = *node.outputs.first()?;
        if !retyped_outputs.contains(&primary) {
            return None;
        }

        Some(NormCastFoldPlan {
            node_id,
            narrow_dtype,
            new_inputs,
            retyped_outputs,
            output_cast_bypass,
            dead_input_casts,
        })
    }
}

/// Read a `Cast` node's `to` attribute as a [`DataType`].
fn cast_target(node: &onnx_runtime_ir::Node) -> Option<DataType> {
    let raw = node.attr("to").and_then(Attribute::as_int)?;
    DataType::from_onnx(raw as i32)
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

/// Fold a `Cast` whose input is a producer-less constant initializer into a
/// pre-converted constant, removing the per-decode-step cast launch.
///
/// WHY: bf16 decoders that compute their RMS/layer norms in fp32 export, around
/// every norm, a `Cast(bf16→f32)` of the *constant* norm weight (`gamma`). That
/// cast recomputes the identical fp32 weight on every single token — pure launch
/// overhead in the decode step, which at M=1 is dominated by per-op dispatch
/// (kernel launch), not GPU math. Muse-Glimmer-30B has 208 such constant-weight
/// casts per step (four norms × 52 layers); hoisting them out of the step
/// removes 208 launches per token. The generic [`ConstantFolding`] pass caps at
/// 1024 elements (shape math only) and has no `Cast` evaluator, so weight-sized
/// casts reach it unfolded; this CUDA-scoped pass materializes them exactly.
///
/// Correctness: the fold is byte-identical to the runtime `Cast` kernel.
/// Widening (e.g. bf16→f32) is exact; narrowing to f16/bf16 uses `half`'s
/// round-to-nearest-even, matching the kernel's `__float2half_rn` /
/// `__float2bfloat16`. Only producer-less, statically-shaped, whole-byte float
/// initializers among {f16, bf16, f32, f64} are folded; every other case is
/// skipped (a safe no-op), never risking a wrong constant. The original
/// initializer is left intact for any other consumers.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaFoldConstantCast;

/// Environment opt-out for [`CudaFoldConstantCast`], mirroring the other
/// CUDA-pass switches. Any value other than unset/empty/`0` restores the
/// exported per-step constant `Cast` launches (for A/B measurement or rollback).
const CONST_CAST_FOLD_DISABLE_ENV: &str = "ONNX_GENAI_CUDA_DISABLE_CONST_CAST_FOLD";

fn const_cast_fold_disabled() -> bool {
    std::env::var_os(CONST_CAST_FOLD_DISABLE_ENV)
        .is_some_and(|value| value != "0" && !value.is_empty())
}

struct CastFoldPlan {
    node: NodeId,
    output: ValueId,
    dtype: DataType,
    dims: Vec<usize>,
    bytes: Vec<u8>,
}

impl OptimizationPass for CudaFoldConstantCast {
    fn name(&self) -> &str {
        "CudaFoldConstantCast"
    }

    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> OptimizerResult<()> {
        if const_cast_fold_disabled() {
            return Ok(());
        }
        let candidates: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.op_type == "Cast"
                    && matches!(node.domain.as_str(), "" | "ai.onnx")
                    && node.inputs.len() == 1
                    && node.outputs.len() == 1)
                    .then_some(id)
            })
            .collect();

        let mut plans: Vec<CastFoldPlan> = Vec::new();
        for node_id in candidates {
            if let Some(plan) = self.plan_fold(graph, ctx, node_id) {
                plans.push(plan);
            }
        }

        let changed = !plans.is_empty();
        for plan in plans {
            // Delete the Cast; its output value survives because a consumer (or
            // graph-output slot) still references it, mirroring the transpose
            // fold. Then retype the surviving value and back it with the
            // materialized constant.
            graph.remove_node(plan.node);
            if graph.try_value(plan.output).is_none() {
                continue;
            }
            let value = graph.value_mut(plan.output);
            value.dtype = plan.dtype;
            value.shape = static_shape(plan.dims.clone());
            let tensor = TensorData::from_raw(plan.dtype, plan.dims, plan.bytes);
            graph.set_initializer(plan.output, WeightRef::Inline(tensor));
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

impl CudaFoldConstantCast {
    fn plan_fold(&self, graph: &Graph, ctx: &PassContext, node_id: NodeId) -> Option<CastFoldPlan> {
        let node = graph.try_node(node_id)?;
        let input = node.inputs[0]?;
        let output = node.outputs[0];

        // The input must be an immutable, producer-less constant initializer.
        if graph.try_value(input)?.producer.is_some() {
            return None;
        }
        let weight = graph.initializers.get(&input)?;
        let src_dtype = weight.dtype();
        let dst_dtype = DataType::from_onnx(node.attr("to").and_then(Attribute::as_int)? as i32)?;

        // Only whole-byte float element types are converted here; sub-byte and
        // non-float casts are left untouched rather than risk a wrong constant.
        if src_dtype.byte_size() == 0 || src_dtype.is_sub_byte() {
            return None;
        }
        let dims = weight.dims().to_vec();
        let src = ctx.initializer_bytes(weight)?;
        let expected = dims
            .iter()
            .product::<usize>()
            .checked_mul(src_dtype.byte_size())?;
        if src.len() != expected {
            return None;
        }
        // No-op cast (same dtype): let dead-node elimination handle it.
        if src_dtype == dst_dtype {
            return None;
        }
        let bytes = convert_float_bytes(src, src_dtype, dst_dtype)?;

        Some(CastFoldPlan {
            node: node_id,
            output,
            dtype: dst_dtype,
            dims,
            bytes,
        })
    }
}

/// Convert raw tensor bytes between float element types, matching the runtime
/// `Cast` kernel's rounding (round-to-nearest-even for narrowing, exact for
/// widening). Returns `None` for any element type outside {f16, bf16, f32, f64},
/// so an unsupported cast is skipped rather than mis-folded.
fn convert_float_bytes(src: &[u8], from: DataType, to: DataType) -> Option<Vec<u8>> {
    // Decode source into an f32 working value per element. f64 is decoded via
    // f32 to keep the kernel's `double` intermediate exactness for the float
    // set we support (f16/bf16/f32 all fit exactly in f32 and f64).
    let values: Vec<f64> = match from {
        DataType::Float32 => src
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        DataType::Float64 => src
            .chunks_exact(8)
            .map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect(),
        DataType::Float16 => src
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_ne_bytes([c[0], c[1]])).to_f32() as f64)
            .collect(),
        DataType::BFloat16 => src
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_ne_bytes([c[0], c[1]])).to_f32() as f64)
            .collect(),
        _ => return None,
    };

    let mut out = Vec::with_capacity(values.len() * to.byte_size().max(1));
    match to {
        DataType::Float32 => {
            for v in values {
                out.extend_from_slice(&(v as f32).to_ne_bytes());
            }
        }
        DataType::Float64 => {
            for v in values {
                out.extend_from_slice(&v.to_ne_bytes());
            }
        }
        DataType::Float16 => {
            for v in values {
                out.extend_from_slice(&half::f16::from_f32(v as f32).to_bits().to_ne_bytes());
            }
        }
        DataType::BFloat16 => {
            for v in values {
                out.extend_from_slice(&half::bf16::from_f32(v as f32).to_bits().to_ne_bytes());
            }
        }
        _ => return None,
    }
    Some(out)
}

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

/// Block-quantization the fused RMS-norm GEMV kernels assume. The fused decode
/// GEMVs (and their prefill GEMM) implement both the packed int4 layout and the
/// one-byte-per-weight int8 layout, so either bit width fuses; the block size is
/// fixed at 32 by the tuned four-lane/eight-block warp walk.
const RMSNORM_FUSION_SUPPORTED_BITS: [i64; 2] = [4, 8];
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
    preceding_zero_points: Option<ValueId>,
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
            //    bias slot (post-round semantics), preserving any asymmetric
            //    zero point at slot 3, then drop the norm node.
            let mut preceding = graph.node(plan.preceding_id).clone();
            preceding.inputs = vec![
                Some(plan.preceding_inputs[0]),
                Some(plan.preceding_inputs[1]),
                Some(plan.preceding_inputs[2]),
                plan.preceding_zero_points,
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
        // fp16 input/skip, dense skip (identical shapes), hidden % 128 == 0.
        // Gamma may be fp16 OR fp32 — it is only a final multiplicand (never in
        // the fp32 variance accumulation), so an fp32 gamma (e.g. Phi-4-mini)
        // is numerically safe and the fused kernel reads it at full precision.
        let input_meta = graph.value(input_value);
        let skip_meta = graph.value(skip_value);
        let gamma_meta = graph.value(gamma);
        if input_meta.dtype != DataType::Float16
            || skip_meta.dtype != DataType::Float16
            || (gamma_meta.dtype != DataType::Float16 && gamma_meta.dtype != DataType::Float32)
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
        let preceding_inputs = [
            preceding.inputs[0]?,
            preceding.inputs[1]?,
            preceding.inputs[2]?,
        ];
        // Preserve an optional asymmetric zero point (slot 3) so the residual
        // fold below does not silently drop it (Phi-4-mini's o_proj is int4 with
        // an explicit zero point).
        let preceding_zero_points = preceding.inputs.get(3).copied().flatten();

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
            preceding_zero_points,
            preceding_out,
            residual,
            gamma,
            epsilon,
            normalized_out,
            sum_out,
            following_ids,
        })
    }

    /// A preceding GEMV is fusable when it is an int4/fp16 `MatMulNBits` with an
    /// optional asymmetric zero point (no group index or existing bias), its
    /// only consumer is the norm, it is not a graph output, and its output width
    /// equals the hidden size (so the residual add is well-shaped).
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
        // A/B/scales with an optional asymmetric zero point (slot 3); no group
        // index (slot 4) or pre-existing bias (slot 5), which the residual fold
        // reuses.
        let value_count = node.input_values().count();
        if !(value_count == 3 || value_count == 4)
            || node.inputs.iter().skip(4).any(Option::is_some)
        {
            return None;
        }
        // A present zero point (slot 3) must be uint8.
        if let Some(zero_points) = node.inputs.get(3).copied().flatten()
            && graph.try_value(zero_points).map(|value| value.dtype) != Some(DataType::Uint8)
        {
            return None;
        }
        if !self.is_fusable_bits_fp16_matmul(graph, node) {
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

    /// A following GEMV is prologue-capable when it is a general int4/int8 fp16
    /// `MatMulNBits` (an optional asymmetric zero point and/or bias allowed, no
    /// group index), with `K <= N` so the general — not the tall-skinny down —
    /// variant runs.
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
        if !self.is_fusable_bits_fp16_matmul(graph, node) {
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
        // for both projections. Its inputs are [x, W_gate, scales_gate, W_up,
        // scales_up] for symmetric weights, or those five plus a reserved gamma
        // slot and both projections' zero points for asymmetric weights; the up
        // scales at slot 4 are legitimate, so the plain zero-point/group-index
        // slot checks below do not apply.
        if node.attr(GATE_UP_SWIGLU_FUSION_ATTR).is_some() {
            let value_count = node.input_values().count();
            return (value_count == 5 || value_count == 7) && k <= n;
        }
        // An optional asymmetric zero point (slot 3, uint8) is allowed — the
        // fused GEMVs dequant it exactly as the non-fused path. No group index
        // (slot 4), no pre-existing gamma (slot 6); an optional bias (slot 5) is
        // allowed.
        if let Some(zero_points) = node.inputs.get(3).copied().flatten()
            && graph.try_value(zero_points).map(|value| value.dtype) != Some(DataType::Uint8)
        {
            return false;
        }
        if node.inputs.get(4).copied().flatten().is_some()
            || node.inputs.get(6).copied().flatten().is_some()
        {
            return false;
        }
        // Keep the general scales_f16 entry: the down variant is chosen when
        // K > N, and it has no normalization prologue.
        k <= n
    }

    /// Shared block-quantization + fp16-scales checks for the fused GEMVs. Admits
    /// both int4 (packed nibble) and int8 (one byte per weight) MatMulNBits: the
    /// fused decode GEMVs and prefill GEMM implement both layouts, keyed off the
    /// `bits` attribute — never a model name.
    fn is_fusable_bits_fp16_matmul(&self, graph: &Graph, node: &onnx_runtime_ir::Node) -> bool {
        if !RMSNORM_FUSION_SUPPORTED_BITS
            .contains(&node.attr("bits").and_then(Attribute::as_int).unwrap_or(4))
        {
            return false;
        }
        if node.attr("block_size").and_then(Attribute::as_int)
            != Some(RMSNORM_FUSION_SUPPORTED_BLOCK_SIZE)
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
            if silu.attr(DECOMPOSED_SILU_ATTR).and_then(Attribute::as_int) == Some(1) {
                fused
                    .attributes
                    .insert(DECOMPOSED_SILU_ATTR.into(), Attribute::Int(1));
            }
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
/// each produced by a three- or four-input `MatMulNBits` sharing the *same*
/// activation, structurally paired (`gate.N == up.N`, `gate.K == up.K`), and
/// compatible with the paired kernel (block-32, 4-bit, fp16 activation/scales/
/// output, persistent weights), the three ops collapse into a single node
/// marked [`GATE_UP_SWIGLU_FUSION_ATTR`] whose inputs are
/// `[x, W_gate, scales_gate, W_up, scales_up]` for symmetric weights, or those
/// five plus a reserved gamma slot and both projections' asymmetric zero points
/// (`[.., None, zp_gate, zp_up]`) for asymmetric weights. The paired kernel
/// reads the activation once, runs both GEMVs (dequantizing `(code - zp) *
/// scale`), and writes `silu(gate)*up` directly — reproducing the two-op fp16
/// rounding so greedy tokens stay byte-identical.
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
    gate_zero_points: Option<ValueId>,
    up_weight: ValueId,
    up_scales: ValueId,
    up_zero_points: Option<ValueId>,
    gate_out: ValueId,
    up_out: ValueId,
    decomposed_silu: bool,
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
            // Asymmetric weights carry per-projection zero points at slots 6/7.
            // Slot 5 is reserved for the RMS-norm gamma that
            // `CudaSkipRmsNormMatMulFusion` folds in later, so leave it empty
            // (None) here; symmetric weights add no trailing slots at all and
            // stay byte-identical to the pre-zero-point contract.
            if let (Some(gate_zp), Some(up_zp)) = (plan.gate_zero_points, plan.up_zero_points) {
                fused.inputs.push(None);
                fused.inputs.push(Some(gate_zp));
                fused.inputs.push(Some(up_zp));
            }
            fused.outputs = graph.node(plan.mul_id).outputs.clone();
            fused
                .attributes
                .insert(GATE_UP_SWIGLU_FUSION_ATTR.into(), Attribute::Int(1));
            if plan.decomposed_silu {
                fused
                    .attributes
                    .insert(DECOMPOSED_SILU_ATTR.into(), Attribute::Int(1));
            }
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
        // Either both projections carry an asymmetric zero point or neither
        // does: the fused kernel takes both zero-point tensors together.
        if gate.zero_points.is_some() != up.zero_points.is_some() {
            return None;
        }

        Some(GateUpSwiGluPlan {
            mul_id,
            gate_matmul_id,
            up_matmul_id,
            activation: gate.activation,
            gate_weight: gate.weight,
            gate_scales: gate.scales,
            gate_zero_points: gate.zero_points,
            up_weight: up.weight,
            up_scales: up.scales,
            up_zero_points: up.zero_points,
            gate_out,
            up_out,
            decomposed_silu: mul.attr(DECOMPOSED_SILU_ATTR).and_then(Attribute::as_int) == Some(1),
        })
    }

    /// Validate one projection `MatMulNBits` against the paired kernel's
    /// **capability** contract (not any model's dimensions) and return its
    /// `[x, W, scales, (zero_points?)]` value ids plus its `N`/`K` for
    /// structural pairing.
    fn eligible_projection(&self, graph: &Graph, matmul_id: NodeId) -> Option<Projection> {
        let matmul = graph.try_node(matmul_id)?;
        // A/B/scales with an optional asymmetric zero-point (slot 3). The paired
        // kernel dequantizes `(code - zp) * scale` per block, so a 4-input
        // MatMulNBits (Phi-4-mini) fuses just like the 3-input symmetric form
        // (Qwen); a symmetric weight simply omits the zero point. Reject group
        // index (slot 4) and bias (slot 5), which the kernel does not model.
        let present: Vec<ValueId> = matmul.input_values().collect();
        if !(present.len() == 3 || present.len() == 4)
            || matmul.inputs.iter().skip(4).any(Option::is_some)
        {
            return None;
        }
        let zero_points = matmul.inputs.get(3).copied().flatten();

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
        // A persistent zero point (uint8 initializer) is required when present so
        // the fused kernel's fixed device signature stays capture-safe.
        if let Some(zero_points) = zero_points
            && (graph.value(zero_points).dtype != DataType::Uint8
                || !graph.initializers.contains_key(&zero_points))
        {
            return None;
        }

        Some(Projection {
            activation,
            weight,
            scales,
            zero_points,
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
    zero_points: Option<ValueId>,
    n: usize,
    k: usize,
}

/// Environment opt-in: set to `1` to enable [`CudaQkvProjectionFusion`].
///
/// The pass is **disabled by default**. An end-to-end A/B on Muse-Glimmer-30B
/// int4 (#872 follow-up) showed the fusion is byte-exact and cuts the captured
/// `MatMulNBits` launch count 417 → 313 (−104 GEMV launches/token), yet decode
/// throughput is flat-to-marginally-worse (47.33 → 47.26 tok/s median). Decode
/// is weight-bandwidth/compute-floor bound, not expensive-dispatch bound, so
/// enabling the fusion by default would ship a (small) regression for no gain.
/// The correct, tested implementation is retained behind this flag for future
/// architectures where the projections are dispatch-bound (e.g. fp16 activations
/// or shapes with a higher launch-latency-to-bandwidth ratio).
const QKV_FUSION_ENABLE_ENV: &str = "ONNX_GENAI_CUDA_ENABLE_QKV_FUSION";

fn qkv_fusion_enabled() -> bool {
    std::env::var_os(QKV_FUSION_ENABLE_ENV).is_some_and(|value| value != "0" && !value.is_empty())
}

/// Fuse the three per-decoder-layer attention projections (`q_proj`, `k_proj`,
/// `v_proj`) — separate `MatMulNBits` GEMVs that all read the *same*
/// input-normalization activation — into a single wider `MatMulNBits` whose
/// output is demultiplexed by one `Split` back to the original Q/K/V consumers.
///
/// ## Why (the lever this tests)
///
/// Under CUDA-graph capture, decode is dominated by serially issued grid
/// launches. Prior work (#870, #872) showed reducing *cheap* elementwise node
/// count does not help. This pass tests the remaining hypothesis by cutting the
/// count of *expensive* GEMV launches: 3 projection launches/layer become 1
/// (+ one `Split` copy), i.e. −2 `MatMulNBits` launches × 52 layers = −104
/// GEMV launches/token. If tok/s improves, decode is dispatch-bound on the
/// expensive path; if flat/worse, it is weight-bandwidth/compute-floor bound
/// (the three projections read disjoint weights, so fusing cannot reduce bytes).
///
/// ## Result (measured, Muse-Glimmer-30B int4, #872 follow-up)
///
/// The fusion is byte-exact and cuts the captured `MatMulNBits` launch count
/// 417 → 313 (−104 GEMV launches/token, capture stays 1 segment / 0 seams), yet
/// decode throughput is **flat-to-marginally-worse** (47.33 → 47.26 tok/s median
/// over 3 interleaved trials). The three projections read disjoint weights, so
/// fusing cannot cut bytes moved — this is the definitive evidence that decode
/// is weight-bandwidth/compute-floor bound, not expensive-dispatch bound. The
/// pass is therefore **disabled by default** (opt-in via
/// [`QKV_FUSION_ENABLE_ENV`]) and retained for future dispatch-bound shapes.
///
/// ## Byte-exactness
///
/// `MatMulNBits` computes each output column independently from its own weight
/// row / scale / zero-point block, reducing over the shared contraction `K`.
/// Column-concatenating the Q, K, V weight/scale/zero-point initializers along
/// their output (`N`) axis therefore leaves every column's dequant + reduction
/// bit-for-bit identical; the trailing `Split` is a pure copy. Greedy tokens
/// stay byte-identical (verified end-to-end).
///
/// ## Gate (structural + capability, never a specific model's dimensions)
///
/// Fires on a `GroupQueryAttention` whose query/key/value trace back to three
/// distinct `MatMulNBits` GEMVs that (a) share one activation input, (b) share
/// `K`, `block_size` (32) and `bits` (4), (c) hold persistent initializer
/// weights/scales (and either all or none carry an asymmetric zero point), and
/// (d) each feed exactly one consumer and no graph output. Any mismatch leaves
/// the three separate GEMVs untouched.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaQkvProjectionFusion;

struct QkvProj {
    matmul: NodeId,
    out: ValueId,
    weight: ValueId,
    scales: ValueId,
    zero_points: Option<ValueId>,
    activation: ValueId,
    n: usize,
    k: usize,
    block_size: i64,
    bits: i64,
}

struct QkvFusionPlan {
    activation: ValueId,
    q: QkvProj,
    k: QkvProj,
    v: QkvProj,
}

impl OptimizationPass for CudaQkvProjectionFusion {
    fn name(&self) -> &str {
        "CudaQkvProjectionFusion"
    }

    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> OptimizerResult<()> {
        if !qkv_fusion_enabled() {
            return Ok(());
        }
        self.fuse_all(graph, ctx)
    }
}

impl CudaQkvProjectionFusion {
    /// Perform the fusion regardless of the [`QKV_FUSION_ENABLE_ENV`] gate.
    /// `run` applies the opt-in gate and then delegates here; tests exercise the
    /// fusion logic directly without mutating process-global environment state.
    fn fuse_all(&self, graph: &mut Graph, ctx: &PassContext) -> OptimizerResult<()> {
        let gqa_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.op_type == "GroupQueryAttention" && node.domain == MICROSOFT_DOMAIN)
                    .then_some(id)
            })
            .collect();

        let mut plans: Vec<QkvFusionPlan> = Vec::new();
        for gqa_id in gqa_nodes {
            if let Some(plan) = self.plan_fuse(graph, gqa_id) {
                plans.push(plan);
            }
        }

        let changed = !plans.is_empty();
        for plan in plans {
            self.apply_fuse(graph, ctx, plan)?;
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }

    fn plan_fuse(&self, graph: &Graph, gqa_id: NodeId) -> Option<QkvFusionPlan> {
        let gqa = graph.try_node(gqa_id)?;
        if gqa.inputs.len() < 3 {
            return None;
        }
        let q_mm = self.trace_back_to_matmul(graph, gqa.inputs[0]?)?;
        let k_mm = self.trace_back_to_matmul(graph, gqa.inputs[1]?)?;
        let v_mm = self.trace_back_to_matmul(graph, gqa.inputs[2]?)?;
        if q_mm == k_mm || q_mm == v_mm || k_mm == v_mm {
            return None;
        }

        let q = self.eligible_projection(graph, q_mm)?;
        let k = self.eligible_projection(graph, k_mm)?;
        let v = self.eligible_projection(graph, v_mm)?;

        // All three projections must read the same activation and be mutually
        // compatible (shared contraction depth, quantization geometry, and
        // zero-point presence) so their weights concatenate byte-exactly.
        if q.activation != k.activation || q.activation != v.activation {
            return None;
        }
        if q.k != k.k || q.k != v.k {
            return None;
        }
        if q.block_size != k.block_size || q.block_size != v.block_size {
            return None;
        }
        if q.bits != k.bits || q.bits != v.bits {
            return None;
        }
        if q.zero_points.is_some() != k.zero_points.is_some()
            || q.zero_points.is_some() != v.zero_points.is_some()
        {
            return None;
        }
        // Weight/scale (and zero-point) element types must match across the
        // three so the concatenated initializer stays homogeneous.
        if graph.value(q.out).dtype != graph.value(k.out).dtype
            || graph.value(q.out).dtype != graph.value(v.out).dtype
        {
            return None;
        }
        if graph.initializers.get(&q.scales)?.dtype() != graph.initializers.get(&k.scales)?.dtype()
            || graph.initializers.get(&q.scales)?.dtype()
                != graph.initializers.get(&v.scales)?.dtype()
        {
            return None;
        }

        Some(QkvFusionPlan {
            activation: q.activation,
            q,
            k,
            v,
        })
    }

    /// Walk back from `value` along the primary (`input[0]`) data edge until a
    /// `MatMulNBits` producer is found, skipping the Q/K post-projection
    /// norm/scale/reshape glue. Returns `None` on a dead end or if no
    /// `MatMulNBits` is reached within a small depth bound.
    fn trace_back_to_matmul(&self, graph: &Graph, mut value: ValueId) -> Option<NodeId> {
        for _ in 0..12 {
            let producer = graph.try_value(value)?.producer?;
            let node = graph.try_node(producer)?;
            if node.op_type == "MatMulNBits" && node.domain == MICROSOFT_DOMAIN {
                return Some(producer);
            }
            value = node.inputs.first().copied().flatten()?;
        }
        None
    }

    /// Validate one projection `MatMulNBits` against the capability contract and
    /// return its value ids + geometry, or `None` if it cannot be fused.
    fn eligible_projection(&self, graph: &Graph, matmul_id: NodeId) -> Option<QkvProj> {
        let node = graph.try_node(matmul_id)?;
        // `[activation, weight, scales]` plus an optional asymmetric zero point
        // (slot 3). Reject a group index (slot 4) or bias (slot 5).
        let present: Vec<ValueId> = node.input_values().collect();
        if !(present.len() == 3 || present.len() == 4)
            || node.inputs.iter().skip(4).any(Option::is_some)
        {
            return None;
        }
        let zero_points = node.inputs.get(3).copied().flatten();

        let n = node.attr("N").and_then(Attribute::as_int)? as usize;
        let k = node.attr("K").and_then(Attribute::as_int)? as usize;
        let block_size = node.attr("block_size").and_then(Attribute::as_int)?;
        let bits = node.attr("bits").and_then(Attribute::as_int).unwrap_or(4);
        // Match the block-32 / 4-bit form the concatenation reasons about; the
        // byte layout of any other geometry is not assumed here.
        if block_size != GATE_UP_SWIGLU_SUPPORTED_BLOCK_SIZE as i64
            || bits != GATE_UP_SWIGLU_SUPPORTED_BITS
        {
            return None;
        }

        let activation = node.inputs[0]?;
        let weight = node.inputs[1]?;
        let scales = node.inputs[2]?;
        if !graph.initializers.contains_key(&weight) || !graph.initializers.contains_key(&scales) {
            return None;
        }
        if let Some(zp) = zero_points
            && !graph.initializers.contains_key(&zp)
        {
            return None;
        }

        // The projection output must feed exactly one consumer and not escape as
        // a graph output, so reusing it as a `Split` output is safe.
        let out = node.outputs[0];
        if graph.consumers(out).len() != 1 || graph.value(out).is_graph_output {
            return None;
        }

        Some(QkvProj {
            matmul: matmul_id,
            out,
            weight,
            scales,
            zero_points,
            activation,
            n,
            k,
            block_size,
            bits,
        })
    }

    fn apply_fuse(
        &self,
        graph: &mut Graph,
        ctx: &PassContext,
        plan: QkvFusionPlan,
    ) -> OptimizerResult<()> {
        let QkvFusionPlan {
            activation,
            q,
            k,
            v,
        } = plan;
        let n_total = q.n + k.n + v.n;

        // Concatenate the int4 weight / scale / (zero-point) initializers along
        // their output (`N`) axis. Axis 0 is the outermost, row-major dimension,
        // so an N-axis concatenation is a byte concatenation; each sub-tensor is
        // byte-aligned (`N * n_blocks` even for the zero points), keeping it exact.
        let weight_bytes = self.concat_initializers(graph, ctx, &[q.weight, k.weight, v.weight])?;
        let scale_bytes = self.concat_initializers(graph, ctx, &[q.scales, k.scales, v.scales])?;
        let zero_bytes = match (q.zero_points, k.zero_points, v.zero_points) {
            (Some(qz), Some(kz), Some(vz)) => {
                Some(self.concat_initializers(graph, ctx, &[qz, kz, vz])?)
            }
            _ => None,
        };

        // Fused weight dims: [N_total, n_blocks, blob] cloned from Q (the three
        // share n_blocks/blob because they share K, block_size and bits).
        let q_weight = graph.initializers.get(&q.weight).ok_or_else(|| {
            OptimizerError::Fusion("qkv fusion: missing q weight initializer".into())
        })?;
        let mut weight_dims = q_weight.dims().to_vec();
        let weight_dtype = q_weight.dtype();
        if weight_dims.is_empty() {
            return Err(OptimizerError::Fusion(
                "qkv fusion: scalar weight initializer".into(),
            ));
        }
        weight_dims[0] = n_total;

        let scale_dtype = graph
            .initializers
            .get(&q.scales)
            .ok_or_else(|| OptimizerError::Fusion("qkv fusion: missing q scales".into()))?
            .dtype();

        let fused_weight = graph.create_value(weight_dtype, static_shape(weight_dims.clone()));
        graph.set_initializer(
            fused_weight,
            WeightRef::Inline(TensorData::from_raw(
                weight_dtype,
                weight_dims,
                weight_bytes,
            )),
        );

        let scale_len = scale_bytes.len() / scale_dtype.byte_size().max(1);
        let fused_scales = graph.create_value(scale_dtype, static_shape(vec![scale_len]));
        graph.set_initializer(
            fused_scales,
            WeightRef::Inline(TensorData::from_raw(
                scale_dtype,
                vec![scale_len],
                scale_bytes,
            )),
        );

        let fused_zero = zero_bytes.map(|bytes| {
            let value = graph.create_value(DataType::Uint8, static_shape(vec![bytes.len()]));
            graph.set_initializer(
                value,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Uint8,
                    vec![bytes.len()],
                    bytes,
                )),
            );
            value
        });

        // Fused MatMulNBits output: Q's output shape with the trailing (`N`) axis
        // widened to N_total.
        let out_dtype = graph.value(q.out).dtype;
        let mut fused_shape = graph.value(q.out).shape.clone();
        match fused_shape.last_mut() {
            Some(dim) => *dim = Dim::Static(n_total),
            None => {
                return Err(OptimizerError::Fusion(
                    "qkv fusion: scalar projection output".into(),
                ));
            }
        }
        let fused_out = graph.create_value(out_dtype, fused_shape);

        // Clone Q's attributes (preserving block_size/bits/accuracy_level/etc.)
        // and widen N.
        let mut attributes = graph.node(q.matmul).attributes.clone();
        attributes.insert("N".into(), Attribute::Int(n_total as i64));
        let version = graph.node(q.matmul).version;

        let mut fused_inputs = vec![Some(activation), Some(fused_weight), Some(fused_scales)];
        if let Some(zero) = fused_zero {
            fused_inputs.push(Some(zero));
        }
        let mut fused_matmul = Node::new(NodeId(0), "MatMulNBits", fused_inputs, vec![fused_out]);
        fused_matmul.domain = MICROSOFT_DOMAIN.into();
        fused_matmul.version = version;
        fused_matmul.attributes = attributes;
        graph.insert_node(fused_matmul);

        // Drop the three now-redundant projection GEMVs. Their outputs survive
        // (each still has its downstream consumer) with a cleared producer, so
        // they can be re-attached as the Split outputs below.
        let axis = graph.value(q.out).shape.len() as i64 - 1;
        for proj in [&q, &k, &v] {
            graph.remove_node(proj.matmul);
        }
        // Retire the original per-projection weight/scale/zero-point initializers
        // so they are neither uploaded nor left dangling.
        for proj in [&q, &k, &v] {
            for value in [Some(proj.weight), Some(proj.scales), proj.zero_points]
                .into_iter()
                .flatten()
            {
                graph.initializers.remove(&value);
                graph.gc_value_if_orphan(value);
            }
        }

        // One Split demuxes [Q|K|V] back to the original consumer edges.
        let mut split = Node::new(
            NodeId(0),
            "Split",
            vec![Some(fused_out)],
            vec![q.out, k.out, v.out],
        );
        split.attributes.insert("axis".into(), Attribute::Int(axis));
        split.attributes.insert(
            "split".into(),
            Attribute::Ints(vec![q.n as i64, k.n as i64, v.n as i64]),
        );
        graph.insert_node(split);

        Ok(())
    }

    /// Read and byte-concatenate the raw little-endian bytes of several
    /// initializers in order.
    fn concat_initializers(
        &self,
        graph: &Graph,
        ctx: &PassContext,
        values: &[ValueId],
    ) -> OptimizerResult<Vec<u8>> {
        let mut out = Vec::new();
        for &value in values {
            let weight = graph.initializers.get(&value).ok_or_else(|| {
                OptimizerError::Fusion("qkv fusion: missing initializer for concat".into())
            })?;
            let bytes = ctx.initializer_bytes(weight).ok_or_else(|| {
                OptimizerError::Fusion("qkv fusion: could not resolve initializer bytes".into())
            })?;
            out.extend_from_slice(bytes);
        }
        Ok(out)
    }
}

/// Lower a capture-unsafe `If` whose branches are pure, side-effect-free
/// constant selections into an on-device `Where`, so its loop-invariant scalar
/// predicate is evaluated on the device every step and the decode collapses from
/// two captured CUDA graphs into one — with **no** per-step host `cond` readback
/// / graph split.
///
/// ## The seam this removes
///
/// A decoder's rotary-embedding cache selector exports as
/// `Greater(seq_len, T) → If → (cos_cache, sin_cache)`, where each `If` branch is
/// just `Constant`s emitting a full cos/sin table (a short-context table sized to
/// `T`, and a long-context table sized to the model's max positions). Executing
/// the `If` reads the predicate to the **host** every decode step to pick a
/// branch, which splits the captured graph in two around the read. During steady
/// decode the predicate is loop-invariant, yet the host round-trip and the
/// mid-step graph boundary cost real wall-clock time.
///
/// ## The rewrite (topology + shape driven, never model identity)
///
/// The pass fires on any `If` whose two branches contain **only** `Constant`
/// nodes (so each output depends on nothing but its own constant — pure, no outer
/// captures, guaranteed loop-invariant) and whose per-output branch tensors are
/// selectable on-device. For output `i` it emits
/// `Where(cond, then_const_i, else_const_i)`, keeping both branch tables resident
/// as constants and evaluating the *device* predicate (`Greater`'s bool output)
/// every step. The `If` and its subgraphs are deleted; `cond` is rewired to the
/// `Where`s. `Where`'s capture-safe path (see `kernels::where_op`) then folds
/// into the single captured graph.
///
/// ## Correctness (why the selection stays exact, including at the boundary)
///
/// `Where` recomputes the selection from the live predicate every step, so a
/// genuine branch flip is never missed — unlike a host memo, there is no stale
/// branch. Two shape cases:
///
/// * **Equal-shaped branches** — `then` and `else` tensors have identical shape.
///   The `Where` is a byte-exact select for either predicate value,
///   unconditionally correct for any consumer.
/// * **Differing leading dimension** (the LongRoPE case: `else` short table
///   `[T, ..]`, `then` long table `[maxpos, ..]`, `maxpos > T`, trailing dims
///   equal). The predicate must be `Greater`/`GreaterOrEqual(_, T)` and the
///   short table's leading dim must equal `T`. The short (`else`) table is padded
///   with zeros up to the long leading dim so both operands share the long
///   shape. When the predicate is false the short branch is selected: its
///   original `[0, T)` rows are preserved byte-for-byte, and the appended rows at
///   indices `>= T` are ones the *original* short table never had — so no
///   execution that was in-bounds against the original model can ever read them.
///   When the predicate is true the long branch is selected unchanged. The
///   selection therefore matches the original `If` for every predicate value and
///   crosses the `T` boundary exactly.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaOnDeviceConstantSelect;

/// One rewritten `If` output: the surviving `If`-output value (reused as the
/// `Where` output) plus the materialized constant operands for `Where`.
struct SelectOutput {
    value: ValueId,
    dtype: DataType,
    out_dims: Vec<usize>,
    /// `then` (predicate-true) constant, already sized to `out_dims`.
    x_bytes: Vec<u8>,
    /// `else` (predicate-false) constant, padded to `out_dims`.
    y_bytes: Vec<u8>,
}

struct SelectPlan {
    if_node: NodeId,
    then_key: (NodeId, String),
    else_key: (NodeId, String),
    cond: ValueId,
    name: String,
    outputs: Vec<SelectOutput>,
}

impl OptimizationPass for CudaOnDeviceConstantSelect {
    fn name(&self) -> &str {
        "CudaOnDeviceConstantSelect"
    }

    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> OptimizerResult<()> {
        let candidates: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.op_type == "If" && matches!(node.domain.as_str(), "" | "ai.onnx"))
                    .then_some(id)
            })
            .collect();

        let mut plans: Vec<SelectPlan> = Vec::new();
        for if_node in candidates {
            if let Some(plan) = self.plan_select(graph, ctx, if_node) {
                plans.push(plan);
            }
        }

        let changed = !plans.is_empty();
        for plan in plans {
            // Delete the host `If` first: its output values survive because
            // downstream consumers still reference them, so they can be reused
            // as the `Where` outputs (SSA: one producer, now the `Where`).
            graph.remove_node(plan.if_node);
            graph.subgraphs.remove(&plan.then_key);
            graph.subgraphs.remove(&plan.else_key);
            for (index, out) in plan.outputs.into_iter().enumerate() {
                if graph.try_value(out.value).is_none() {
                    continue;
                }
                let shape = static_shape(out.out_dims.clone());
                let x = graph.create_value(out.dtype, shape.clone());
                graph.set_initializer(
                    x,
                    WeightRef::Inline(TensorData::from_raw(
                        out.dtype,
                        out.out_dims.clone(),
                        out.x_bytes,
                    )),
                );
                let y = graph.create_value(out.dtype, shape.clone());
                graph.set_initializer(
                    y,
                    WeightRef::Inline(TensorData::from_raw(
                        out.dtype,
                        out.out_dims.clone(),
                        out.y_bytes,
                    )),
                );

                let value = graph.value_mut(out.value);
                value.dtype = out.dtype;
                value.shape = shape;

                let mut node = onnx_runtime_ir::Node::new(
                    NodeId(0),
                    "Where",
                    vec![Some(plan.cond), Some(x), Some(y)],
                    vec![out.value],
                );
                node.name = format!("{}/on_device_select_{index}", plan.name);
                graph.insert_node(node);
            }
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

impl CudaOnDeviceConstantSelect {
    fn plan_select(&self, graph: &Graph, ctx: &PassContext, if_node: NodeId) -> Option<SelectPlan> {
        let node = graph.try_node(if_node)?;
        let cond = node.inputs.first().copied().flatten()?;
        let then_key = (if_node, "then_branch".to_string());
        let else_key = (if_node, "else_branch".to_string());
        let then_branch = graph.subgraphs.get(&then_key)?;
        let else_branch = graph.subgraphs.get(&else_key)?;

        // Both branches must be pure constant selections: every node a
        // `Constant`, zero formal inputs (so no outer captures), and one output
        // per node — the guarantee that each output is loop-invariant.
        if !branch_is_pure_constants(then_branch) || !branch_is_pure_constants(else_branch) {
            return None;
        }
        let output_count = node.outputs.len();
        if output_count == 0
            || then_branch.outputs.len() != output_count
            || else_branch.outputs.len() != output_count
        {
            return None;
        }

        // The predicate-false / -true index tie (only needed when a branch is
        // padded): `cond = Greater/GreaterOrEqual(_, T)` with a scalar-int `T`.
        let threshold = greater_threshold(graph, ctx, cond);

        let mut outputs = Vec::with_capacity(output_count);
        for i in 0..output_count {
            let then_tensor = branch_constant(then_branch, then_branch.outputs[i])?;
            let else_tensor = branch_constant(else_branch, else_branch.outputs[i])?;
            if then_tensor.dtype != else_tensor.dtype {
                return None;
            }
            let dtype = then_tensor.dtype;
            let elem = dtype.byte_size();
            if elem == 0 || dtype.is_sub_byte() {
                return None;
            }
            let then_dims = then_tensor.dims.clone();
            let else_dims = else_tensor.dims.clone();
            let then_bytes: &[u8] = &then_tensor.data;
            let else_bytes: &[u8] = &else_tensor.data;
            // Guard against malformed constants before trusting the byte lengths.
            if then_bytes.len() != dims_bytes(&then_dims, elem)?
                || else_bytes.len() != dims_bytes(&else_dims, elem)?
            {
                return None;
            }

            let plan = if then_dims == else_dims {
                // Equal-shaped select: byte-exact for either predicate value.
                SelectOutput {
                    value: node.outputs[i],
                    dtype,
                    out_dims: then_dims,
                    x_bytes: then_bytes.to_vec(),
                    y_bytes: else_bytes.to_vec(),
                }
            } else {
                // Differing leading dim: the predicate-true (`then`) branch must
                // be the larger table; the predicate-false (`else`) branch is
                // padded, and its original leading dim must equal the `Greater`
                // threshold so the appended rows are provably never indexed.
                let (then_lead, then_trail) = split_leading(&then_dims)?;
                let (else_lead, else_trail) = split_leading(&else_dims)?;
                if then_trail != else_trail || then_lead <= else_lead {
                    return None;
                }
                let threshold = threshold?;
                // This tie is a RoPE-cache-pattern topological proxy, not a
                // general proof for arbitrary Ifs: the consumer only indexes
                // rows below `threshold` while the predicate is false.
                if i64::try_from(else_lead).ok()? != threshold {
                    return None;
                }
                let row_bytes = else_trail.iter().product::<usize>().checked_mul(elem)?;
                let mut y_bytes = else_bytes.to_vec();
                let pad_rows = then_lead.checked_sub(else_lead)?;
                y_bytes.resize(else_bytes.len() + pad_rows.checked_mul(row_bytes)?, 0);
                SelectOutput {
                    value: node.outputs[i],
                    dtype,
                    out_dims: then_dims,
                    x_bytes: then_bytes.to_vec(),
                    y_bytes,
                }
            };
            outputs.push(plan);
        }

        Some(SelectPlan {
            if_node,
            then_key,
            else_key,
            cond,
            name: node.name.clone(),
            outputs,
        })
    }
}

/// Whether every node in `branch` is a producer-less `Constant` with no formal
/// inputs — a pure, side-effect-free constant selection with no outer captures.
fn branch_is_pure_constants(branch: &Graph) -> bool {
    if !branch.inputs.is_empty() {
        return false;
    }
    branch.nodes.iter().all(|(_, node)| {
        node.op_type == "Constant"
            && matches!(node.domain.as_str(), "" | "ai.onnx")
            && node.inputs.is_empty()
            && node.outputs.len() == 1
            && matches!(node.attr("value"), Some(Attribute::Tensor(_)))
    })
}

/// Read the `Constant` tensor backing branch output `out`, or `None` if `out` is
/// not produced by a default-domain `Constant` with a `value` tensor attribute.
fn branch_constant(branch: &Graph, out: ValueId) -> Option<&TensorData> {
    let producer = branch.try_value(out)?.producer?;
    let node = branch.try_node(producer)?;
    if node.op_type != "Constant" || !matches!(node.domain.as_str(), "" | "ai.onnx") {
        return None;
    }
    match node.attr("value") {
        Some(Attribute::Tensor(tensor)) => Some(tensor),
        _ => None,
    }
}

/// The scalar-int threshold `T` of a `cond = Greater/GreaterOrEqual(_, T)`, when
/// `T` is a producer-less constant int initializer (or a `Constant` node).
fn greater_threshold(graph: &Graph, ctx: &PassContext, cond: ValueId) -> Option<i64> {
    let producer = graph.try_value(cond)?.producer?;
    let node = graph.try_node(producer)?;
    if !matches!(node.op_type.as_str(), "Greater" | "GreaterOrEqual")
        || !matches!(node.domain.as_str(), "" | "ai.onnx")
    {
        return None;
    }
    let threshold_value = node.inputs.get(1).copied().flatten()?;
    scalar_int(graph, ctx, threshold_value)
}

/// Read a single-element integer value, whether backed by an initializer or a
/// `Constant` node. Returns `None` for non-integer or multi-element tensors.
fn scalar_int(graph: &Graph, ctx: &PassContext, value: ValueId) -> Option<i64> {
    let tensor_owned;
    let tensor: &TensorData = if let Some(weight) = graph.initializers.get(&value) {
        match weight {
            WeightRef::Inline(tensor) => tensor,
            WeightRef::External { .. } => {
                tensor_owned = TensorData::from_raw(
                    weight.dtype(),
                    weight.dims().to_vec(),
                    ctx.initializer_bytes(weight)?.to_vec(),
                );
                &tensor_owned
            }
        }
    } else {
        let producer = graph.try_value(value)?.producer?;
        match graph.try_node(producer)?.attr("value") {
            Some(Attribute::Tensor(tensor)) => tensor,
            _ => return None,
        }
    };
    if tensor.dims.iter().product::<usize>() != 1 {
        return None;
    }
    match tensor.dtype {
        DataType::Int64 => Some(i64::from_le_bytes(tensor.data.get(..8)?.try_into().ok()?)),
        DataType::Int32 => Some(i64::from(i32::from_le_bytes(
            tensor.data.get(..4)?.try_into().ok()?,
        ))),
        _ => None,
    }
}

/// Split a shape into `(leading, trailing)` where `trailing` is `dims[1..]`.
fn split_leading(dims: &[usize]) -> Option<(usize, Vec<usize>)> {
    let (&lead, trail) = dims.split_first()?;
    Some((lead, trail.to_vec()))
}

/// Byte length of a dense tensor of `dims` with `elem`-byte elements.
fn dims_bytes(dims: &[usize], elem: usize) -> Option<usize> {
    dims.iter().product::<usize>().checked_mul(elem)
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

    /// `Sigmoid(x)` + `Mul(x, sigmoid)` + `Mul(silu, up)` in a narrow stream
    /// collapses through `CudaSiluFusion` -> `CudaSwiGluFusion` into a single
    /// tagged decomposed `Mul[_cuda_silu_mul]`, deleting the standalone
    /// `Sigmoid` and the intermediate `Mul`. The plain (non-projection) `up`
    /// value means `CudaGateUpSwiGluFusion` cannot fold the GEMVs, so the glue
    /// collapse is observed in isolation.
    fn decomposed_swiglu_glue_graph(dtype: DataType) -> Graph {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let x = value(&mut graph, "x", dtype, 7);
        let up = value(&mut graph, "up", dtype, 7);
        let sigmoid_out = value(&mut graph, "sigmoid", dtype, 7);
        let silu_out = value(&mut graph, "silu", dtype, 7);
        let output = value(&mut graph, "output", dtype, 7);
        graph.add_input(x);
        graph.add_input(up);
        graph.insert_node(Node::new(
            NodeId(0),
            "Sigmoid",
            vec![Some(x)],
            vec![sigmoid_out],
        ));
        graph.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(x), Some(sigmoid_out)],
            vec![silu_out],
        ));
        graph.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(silu_out), Some(up)],
            vec![output],
        ));
        graph.add_output(output);
        graph
    }

    #[test]
    fn collapses_bf16_decomposed_swiglu_glue() {
        let mut graph = decomposed_swiglu_glue_graph(DataType::BFloat16);
        CudaSiluFusion.run(&mut graph, &PassContext::new()).unwrap();
        CudaSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert_eq!(graph.num_nodes(), 1, "Sigmoid + inner Mul must be removed");
        let fused = graph.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "Mul");
        assert_eq!(
            fused.attr(SILU_MUL_FUSION_ATTR).and_then(Attribute::as_int),
            Some(1)
        );
        assert_eq!(
            fused.attr(DECOMPOSED_SILU_ATTR).and_then(Attribute::as_int),
            Some(1),
            "the bf16 fused Mul must carry the decomposed marker so the runtime \
             selects the byte-exact decomposed_silu_mul_bf16 kernel"
        );
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn leaves_fp32_decomposed_swiglu_glue_separate() {
        let mut graph = decomposed_swiglu_glue_graph(DataType::Float32);
        CudaSiluFusion.run(&mut graph, &PassContext::new()).unwrap();

        // No fp32 decomposed-SiLU kernel: the Sigmoid must not lower to Silu.
        assert_eq!(graph.num_nodes(), 3);
        assert!(
            graph.nodes.values().all(|node| node.op_type != "Silu"),
            "fp32 has no byte-exact decomposed SiLU kernel; must stay unfused"
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

    // === Constant Cast fold ===

    /// Build a graph `Cast(const [n], to=dst) -> Identity consumer`, with the
    /// constant provided as an inline initializer of `src_dtype` holding the
    /// bytes `src_bytes`.
    fn const_cast_graph(
        n: usize,
        src_dtype: DataType,
        src_bytes: Vec<u8>,
        dst_dtype: DataType,
    ) -> (Graph, ValueId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);

        let weight = graph.create_named_value("weight", src_dtype, vec![Dim::Static(n)]);
        graph.set_initializer(
            weight,
            WeightRef::Inline(TensorData::from_raw(src_dtype, vec![n], src_bytes)),
        );

        let cast_out = graph.create_named_value("cast_out", dst_dtype, vec![Dim::Static(n)]);
        let mut node = Node::new(NodeId(0), "Cast", vec![Some(weight)], vec![cast_out]);
        node.attributes
            .insert("to".into(), Attribute::Int(dst_dtype.to_onnx() as i64));
        graph.insert_node(node);

        let out = graph.create_named_value("out", dst_dtype, vec![Dim::Static(n)]);
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(cast_out)],
            vec![out],
        ));
        graph.add_output(out);
        (graph, cast_out)
    }

    #[test]
    fn folds_constant_cast_bf16_to_f32_byte_identical() {
        // bf16 → f32 widening is exact. Element i holds bf16(i * 1.5).
        let n = 6usize;
        let mut bytes = Vec::with_capacity(n * 2);
        let mut expected = Vec::with_capacity(n);
        for i in 0..n {
            let v = half::bf16::from_f32(i as f32 * 1.5);
            bytes.extend_from_slice(&v.to_le_bytes());
            expected.push(v.to_f32());
        }
        let (mut graph, cast_out) =
            const_cast_graph(n, DataType::BFloat16, bytes, DataType::Float32);
        assert!(graph.value(cast_out).producer.is_some());

        CudaFoldConstantCast
            .run(&mut graph, &PassContext::new())
            .unwrap();

        // Cast node gone; the value is now a producer-less f32 initializer.
        assert!(graph.nodes.values().all(|node| node.op_type != "Cast"));
        assert!(graph.value(cast_out).producer.is_none());
        assert_eq!(graph.value(cast_out).dtype, DataType::Float32);
        let WeightRef::Inline(tensor) = graph.initializers.get(&cast_out).unwrap() else {
            panic!("expected inline initializer");
        };
        assert_eq!(tensor.dims, vec![n]);
        for (i, want) in expected.iter().enumerate() {
            let got = f32::from_le_bytes([
                tensor.data[i * 4],
                tensor.data[i * 4 + 1],
                tensor.data[i * 4 + 2],
                tensor.data[i * 4 + 3],
            ]);
            assert_eq!(got, *want, "element {i}");
        }
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn folds_constant_cast_f32_to_bf16_round_to_nearest_even() {
        // f32 → bf16 narrowing must match the kernel's round-to-nearest-even
        // (`half::bf16::from_f32`).
        let values = [0.0f32, 1.0, 1.5, -2.75, 3.141_592_7, 65_504.0];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let (mut graph, cast_out) =
            const_cast_graph(values.len(), DataType::Float32, bytes, DataType::BFloat16);

        CudaFoldConstantCast
            .run(&mut graph, &PassContext::new())
            .unwrap();

        let WeightRef::Inline(tensor) = graph.initializers.get(&cast_out).unwrap() else {
            panic!("expected inline initializer");
        };
        for (i, v) in values.iter().enumerate() {
            let got = half::bf16::from_le_bytes([tensor.data[i * 2], tensor.data[i * 2 + 1]]);
            assert_eq!(got, half::bf16::from_f32(*v), "element {i}");
        }
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn leaves_cast_of_non_constant() {
        // A Cast whose input is a graph input (not an initializer) is left alone.
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let x = graph.create_named_value("x", DataType::BFloat16, vec![Dim::Static(4)]);
        graph.add_input(x);
        let cast_out =
            graph.create_named_value("cast_out", DataType::Float32, vec![Dim::Static(4)]);
        let mut node = Node::new(NodeId(0), "Cast", vec![Some(x)], vec![cast_out]);
        node.attributes.insert(
            "to".into(),
            Attribute::Int(DataType::Float32.to_onnx() as i64),
        );
        graph.insert_node(node);
        graph.add_output(cast_out);

        CudaFoldConstantCast
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert_eq!(
            graph
                .nodes
                .values()
                .filter(|node| node.op_type == "Cast")
                .count(),
            1
        );
    }

    #[test]
    fn leaves_sub_byte_constant_cast() {
        // An int4-packed constant must not be folded by the byte-wise path.
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let weight = graph.create_named_value("weight", DataType::Int4, vec![Dim::Static(4)]);
        graph.set_initializer(
            weight,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Int4,
                vec![4],
                vec![0x21, 0x43],
            )),
        );
        let cast_out =
            graph.create_named_value("cast_out", DataType::Float32, vec![Dim::Static(4)]);
        let mut node = Node::new(NodeId(0), "Cast", vec![Some(weight)], vec![cast_out]);
        node.attributes.insert(
            "to".into(),
            Attribute::Int(DataType::Float32.to_onnx() as i64),
        );
        graph.insert_node(node);
        graph.add_output(cast_out);

        CudaFoldConstantCast
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert_eq!(
            graph
                .nodes
                .values()
                .filter(|node| node.op_type == "Cast")
                .count(),
            1
        );
    }

    #[test]
    fn const_cast_fold_respects_disable_env() {
        // SAFETY: single-threaded test; env is restored before returning.
        let n = 4usize;
        let mut bytes = Vec::new();
        for i in 0..n {
            bytes.extend_from_slice(&half::bf16::from_f32(i as f32).to_le_bytes());
        }
        let (mut graph, _cast_out) =
            const_cast_graph(n, DataType::BFloat16, bytes, DataType::Float32);
        unsafe { std::env::set_var(CONST_CAST_FOLD_DISABLE_ENV, "1") };
        let result = CudaFoldConstantCast.run(&mut graph, &PassContext::new());
        unsafe { std::env::remove_var(CONST_CAST_FOLD_DISABLE_ENV) };
        result.unwrap();
        assert_eq!(
            graph
                .nodes
                .values()
                .filter(|node| node.op_type == "Cast")
                .count(),
            1
        );
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
        // End-to-end through the real CUDA pass list: the exported
        // x*Sigmoid(x) decomposition becomes Silu, then the SwiGLU passes
        // collapse both projections and the trailing activation/multiply.
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let x = value(&mut graph, "x", DataType::Float16, QWEN_GATE_UP_K);
        graph.add_input(x);
        let gate_out = projection(&mut graph, "gate", x, QWEN_GATE_UP_K, QWEN_GATE_UP_N);
        let up_out = projection(&mut graph, "up", x, QWEN_GATE_UP_K, QWEN_GATE_UP_N);
        let sigmoid_out = value(&mut graph, "sigmoid", DataType::Float16, QWEN_GATE_UP_N);
        let silu_out = value(&mut graph, "silu", DataType::Float16, QWEN_GATE_UP_N);
        let out = value(&mut graph, "output", DataType::Float16, QWEN_GATE_UP_N);
        graph.insert_node(Node::new(
            NodeId(0),
            "Sigmoid",
            vec![Some(gate_out)],
            vec![sigmoid_out],
        ));
        graph.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(gate_out), Some(sigmoid_out)],
            vec![silu_out],
        ));
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
        assert_eq!(
            fused.attr(DECOMPOSED_SILU_ATTR).and_then(Attribute::as_int),
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
        assert_eq!(
            following.inputs[0],
            Some(pre_out),
            "activation is residual sum"
        );
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
    fn folds_skip_rmsnorm_into_int8_neighbouring_gemvs() {
        // Phi's qkv/down are int8 (bits == 8), block-32, fp16 scales. The fused
        // GEMV family implements the one-byte-per-weight int8 dequant, so the
        // skip-rmsnorm fold must fire on int8 exactly as on int4 — keyed off the
        // `bits` attribute, never a model name.
        let mut graph = skip_rms_graph(RMSNORM_FUSION_MIN_HIDDEN, RMSNORM_FUSION_MIN_HIDDEN + 256);
        set_matmul_attr(&mut graph, "bits", 8);
        let pre_out = value_id_by_name(&graph, "pre_out");
        let gamma = value_id_by_name(&graph, "gamma");

        CudaSkipRmsNormMatMulFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.op_type != "SkipSimplifiedLayerNormalization"),
            "int8 skip-rmsnorm must fuse just like int4"
        );
        let preceding = node_producing(&graph, pre_out);
        assert_eq!(
            preceding
                .attr(MATMUL_NBITS_FOLDED_BIAS_ATTR)
                .and_then(Attribute::as_int),
            Some(1)
        );
        let post_out = value_id_by_name(&graph, "post_out");
        let following = node_producing(&graph, post_out);
        assert_eq!(following.inputs.get(6).copied().flatten(), Some(gamma));
        assert_eq!(
            following
                .attr(MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR)
                .and_then(Attribute::as_int),
            Some(1)
        );
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
            .find_map(|(id, n)| (n.op_type == "SkipSimplifiedLayerNormalization").then_some(id))
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
        assert!(
            graph.validate().is_ok(),
            "no dangling shared residual value"
        );

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

    /// Build a `Cast(input) -> output` node and return the fresh output value.
    fn cast_node(
        graph: &mut Graph,
        name: &str,
        input: ValueId,
        to: DataType,
        width: usize,
    ) -> ValueId {
        let out = value(graph, name, to, width);
        let mut node = Node::new(NodeId(0), "Cast", vec![Some(input)], vec![out]);
        node.attributes
            .insert("to".into(), Attribute::Int(to as i64));
        graph.insert_node(node);
        out
    }

    /// A `SkipSimplifiedLayerNormalization` exported in fp32 with a `Cast` on
    /// every fp16 activation input and every fp32 result (the Phi-4-mini shape).
    /// The `gamma` weight stays fp32. Returns the graph plus the pre-cast fp16
    /// activation values and the fp32 gamma so tests can assert the rewiring.
    fn cast_wrapped_skip_norm_graph(hidden: usize) -> (Graph, ValueId, ValueId, ValueId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);

        let residual = value(&mut graph, "residual", DataType::Float16, hidden);
        let mm = value(&mut graph, "mm", DataType::Float16, hidden);
        graph.add_input(residual);
        graph.add_input(mm);
        let gamma = vec1d(&mut graph, "gamma", DataType::Float32, hidden);
        graph.set_initializer(
            gamma,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![hidden],
                vec![0u8; hidden * 4],
            )),
        );

        let in0 = cast_node(&mut graph, "in0", residual, DataType::Float32, hidden);
        let in1 = cast_node(&mut graph, "in1", mm, DataType::Float32, hidden);
        let norm_out = value(&mut graph, "norm_out", DataType::Float32, hidden);
        let sum_out = value(&mut graph, "sum_out", DataType::Float32, hidden);
        let mut skip = Node::new(
            NodeId(0),
            "SkipSimplifiedLayerNormalization",
            vec![Some(in0), Some(in1), Some(gamma)],
            vec![norm_out, sum_out],
        );
        skip.domain = MICROSOFT_DOMAIN.into();
        skip.attributes
            .insert("epsilon".into(), Attribute::Float(1e-5));
        graph.insert_node(skip);

        let normalized = cast_node(
            &mut graph,
            "normalized",
            norm_out,
            DataType::Float16,
            hidden,
        );
        let residual_out = cast_node(
            &mut graph,
            "residual_out",
            sum_out,
            DataType::Float16,
            hidden,
        );
        graph.add_output(normalized);
        graph.add_output(residual_out);
        (graph, residual, mm, gamma)
    }

    #[test]
    fn drops_casts_around_fp32_wrapped_skip_norm() {
        let hidden = 128;
        let (mut graph, residual, mm, gamma) = cast_wrapped_skip_norm_graph(hidden);

        CudaDropNormalizationCasts
            .run(&mut graph, &PassContext::new())
            .unwrap();

        // Every wrapping Cast is gone.
        assert_eq!(
            graph.nodes.values().filter(|n| n.op_type == "Cast").count(),
            0,
            "all cast wrappers must be removed"
        );

        // The norm now reads its fp16 activations directly and keeps fp32 gamma.
        let skip = graph
            .nodes
            .values()
            .find(|n| n.op_type == "SkipSimplifiedLayerNormalization")
            .expect("norm node retained");
        assert_eq!(
            skip.inputs,
            vec![Some(residual), Some(mm), Some(gamma)],
            "activation inputs rewired to fp16 sources; gamma untouched"
        );
        // Its consumed float outputs are retyped to fp16, matching the inputs.
        for &out in &skip.outputs {
            assert_eq!(graph.value(out).dtype, DataType::Float16);
        }
        // The fp32 gamma weight is preserved (kernel upcasts it internally).
        assert_eq!(graph.value(gamma).dtype, DataType::Float32);
        // Graph outputs are now the norm's fp16 outputs.
        assert_eq!(graph.outputs.len(), 2);
        for &out in &graph.outputs {
            assert_eq!(graph.value(out).dtype, DataType::Float16);
        }
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn leaves_native_fp16_skip_norm_untouched() {
        // A norm already exported in fp16 (no cast wrappers, like Qwen2.5) must
        // be left byte-for-byte unchanged.
        let hidden = 128;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);
        let residual = value(&mut graph, "residual", DataType::Float16, hidden);
        let mm = value(&mut graph, "mm", DataType::Float16, hidden);
        graph.add_input(residual);
        graph.add_input(mm);
        let gamma = vec1d(&mut graph, "gamma", DataType::Float16, hidden);
        graph.set_initializer(
            gamma,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float16,
                vec![hidden],
                vec![0u8; hidden * 2],
            )),
        );
        let norm_out = value(&mut graph, "norm_out", DataType::Float16, hidden);
        let mut skip = Node::new(
            NodeId(0),
            "SkipSimplifiedLayerNormalization",
            vec![Some(residual), Some(mm), Some(gamma)],
            vec![norm_out],
        );
        skip.domain = MICROSOFT_DOMAIN.into();
        graph.insert_node(skip);
        graph.add_output(norm_out);

        let before = graph.nodes.len();
        CudaDropNormalizationCasts
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert_eq!(graph.nodes.len(), before, "no nodes added or removed");
        let skip = graph
            .nodes
            .values()
            .find(|n| n.op_type == "SkipSimplifiedLayerNormalization")
            .expect("norm retained");
        assert_eq!(skip.inputs, vec![Some(residual), Some(mm), Some(gamma)]);
        assert_eq!(graph.value(norm_out).dtype, DataType::Float16);
    }

    #[test]
    fn norm_cast_fold_fires_through_full_cuda_pass_list() {
        // End-to-end through the registered CUDA passes: the cast-drop pass runs
        // and removes every wrapper, leaving the norm in fp16-native form.
        let (mut graph, ..) = cast_wrapped_skip_norm_graph(128);
        for pass in cuda_optimization_passes() {
            pass.run(&mut graph, &PassContext::new()).unwrap();
        }
        assert_eq!(
            graph.nodes.values().filter(|n| n.op_type == "Cast").count(),
            0,
            "cast wrappers must be gone after the full pass list"
        );
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn leaves_cast_wrapped_norm_with_fp32_consumer_intact() {
        // A cast-wrapped norm whose fp32 result also feeds a genuine fp32
        // consumer (not a `Cast(-> fp16)`) is a real fp32 boundary: retyping the
        // output to fp16 would silently change that consumer's input dtype, so
        // the pass must leave the whole norm — casts and all — as exported.
        let hidden = 128;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);

        let residual = value(&mut graph, "residual", DataType::Float16, hidden);
        let mm = value(&mut graph, "mm", DataType::Float16, hidden);
        graph.add_input(residual);
        graph.add_input(mm);
        let gamma = vec1d(&mut graph, "gamma", DataType::Float32, hidden);
        graph.set_initializer(
            gamma,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![hidden],
                vec![0u8; hidden * 4],
            )),
        );

        let in0 = cast_node(&mut graph, "in0", residual, DataType::Float32, hidden);
        let in1 = cast_node(&mut graph, "in1", mm, DataType::Float32, hidden);
        let norm_out = value(&mut graph, "norm_out", DataType::Float32, hidden);
        let mut skip = Node::new(
            NodeId(0),
            "SkipSimplifiedLayerNormalization",
            vec![Some(in0), Some(in1), Some(gamma)],
            vec![norm_out],
        );
        skip.domain = MICROSOFT_DOMAIN.into();
        graph.insert_node(skip);

        // Consumer A: the usual `Cast(-> fp16)` back to the residual stream.
        let normalized = cast_node(
            &mut graph,
            "normalized",
            norm_out,
            DataType::Float16,
            hidden,
        );
        graph.add_output(normalized);
        // Consumer B: a genuine fp32 sink — the boundary that blocks folding.
        let fp32_kept = value(&mut graph, "fp32_kept", DataType::Float32, hidden);
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(norm_out)],
            vec![fp32_kept],
        ));
        graph.add_output(fp32_kept);

        let casts_before = graph.nodes.values().filter(|n| n.op_type == "Cast").count();
        CudaDropNormalizationCasts
            .run(&mut graph, &PassContext::new())
            .unwrap();

        // Nothing folded: both input casts survive, the norm still reads them,
        // and its output stays fp32 for the fp32 consumer.
        assert_eq!(
            graph.nodes.values().filter(|n| n.op_type == "Cast").count(),
            casts_before,
            "a fp32 boundary consumer must block the fold"
        );
        let skip = graph
            .nodes
            .values()
            .find(|n| n.op_type == "SkipSimplifiedLayerNormalization")
            .expect("norm retained");
        assert_eq!(skip.inputs, vec![Some(in0), Some(in1), Some(gamma)]);
        assert_eq!(graph.value(norm_out).dtype, DataType::Float32);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn folds_norms_sharing_an_input_cast() {
        // Two norms consume the *same* fp16 -> fp32 input `Cast`. Folding the
        // first rewires it off the shared cast but must not delete that cast
        // while the second norm still consumes it; folding the second then
        // orphans and removes it. Both norms end up reading the pre-cast fp16
        // source and no `Cast` survives, with the graph structurally valid.
        let hidden = 128;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);

        let x = value(&mut graph, "x", DataType::Float16, hidden);
        let res_a = value(&mut graph, "res_a", DataType::Float16, hidden);
        let res_b = value(&mut graph, "res_b", DataType::Float16, hidden);
        graph.add_input(x);
        graph.add_input(res_a);
        graph.add_input(res_b);

        let gamma_a = vec1d(&mut graph, "gamma_a", DataType::Float32, hidden);
        let gamma_b = vec1d(&mut graph, "gamma_b", DataType::Float32, hidden);
        for g in [gamma_a, gamma_b] {
            graph.set_initializer(
                g,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Float32,
                    vec![hidden],
                    vec![0u8; hidden * 4],
                )),
            );
        }

        // The single shared input cast, consumed by both norms' input slot 0.
        let shared = cast_node(&mut graph, "shared", x, DataType::Float32, hidden);
        let skip_a_in = cast_node(&mut graph, "skip_a_in", res_a, DataType::Float32, hidden);
        let skip_b_in = cast_node(&mut graph, "skip_b_in", res_b, DataType::Float32, hidden);

        let norm_a = value(&mut graph, "norm_a", DataType::Float32, hidden);
        let mut skip_a = Node::new(
            NodeId(0),
            "SkipSimplifiedLayerNormalization",
            vec![Some(shared), Some(skip_a_in), Some(gamma_a)],
            vec![norm_a],
        );
        skip_a.domain = MICROSOFT_DOMAIN.into();
        graph.insert_node(skip_a);

        let norm_b = value(&mut graph, "norm_b", DataType::Float32, hidden);
        let mut skip_b = Node::new(
            NodeId(0),
            "SkipSimplifiedLayerNormalization",
            vec![Some(shared), Some(skip_b_in), Some(gamma_b)],
            vec![norm_b],
        );
        skip_b.domain = MICROSOFT_DOMAIN.into();
        graph.insert_node(skip_b);

        let out_a = cast_node(&mut graph, "out_a", norm_a, DataType::Float16, hidden);
        let out_b = cast_node(&mut graph, "out_b", norm_b, DataType::Float16, hidden);
        graph.add_output(out_a);
        graph.add_output(out_b);

        CudaDropNormalizationCasts
            .run(&mut graph, &PassContext::new())
            .unwrap();

        // Both norms fold: every Cast (shared input, per-norm inputs, outputs)
        // is gone and each norm now reads the pre-cast fp16 sources directly.
        assert_eq!(
            graph.nodes.values().filter(|n| n.op_type == "Cast").count(),
            0,
            "the shared input cast and all wrappers must be removed once unused"
        );
        let norms: Vec<_> = graph
            .nodes
            .values()
            .filter(|n| n.op_type == "SkipSimplifiedLayerNormalization")
            .collect();
        assert_eq!(norms.len(), 2, "both norms retained");
        for norm in norms {
            assert_eq!(
                norm.inputs[0],
                Some(x),
                "both norms read the shared pre-cast fp16 source"
            );
        }
        assert_eq!(graph.value(norm_a).dtype, DataType::Float16);
        assert_eq!(graph.value(norm_b).dtype, DataType::Float16);
        assert!(graph.validate().is_ok());
    }

    /// A bf16 `RMSNormalization` exported in fp32 with a `Cast(bf16 -> f32)` on
    /// its activation input and a `Cast(f32 -> bf16)` on its result — the
    /// Muse-Glimmer shape. The `scale` weight stays fp32 (in the real model it is
    /// a constant `Cast(weight_bf16 -> f32) + 1` expression; here we model it as a
    /// plain fp32 initializer, since the pass only touches activation index 0).
    fn cast_wrapped_rms_norm_graph_bf16(hidden: usize) -> (Graph, ValueId, ValueId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);

        let x_bf16 = value(&mut graph, "x_bf16", DataType::BFloat16, hidden);
        graph.add_input(x_bf16);
        let scale = vec1d(&mut graph, "scale", DataType::Float32, hidden);
        graph.set_initializer(
            scale,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![hidden],
                vec![0u8; hidden * 4],
            )),
        );

        let x_f32 = cast_node(&mut graph, "x_f32", x_bf16, DataType::Float32, hidden);
        let norm_out = value(&mut graph, "norm_out", DataType::Float32, hidden);
        let mut rms = Node::new(
            NodeId(0),
            "RMSNormalization",
            vec![Some(x_f32), Some(scale)],
            vec![norm_out],
        );
        rms.attributes
            .insert("epsilon".into(), Attribute::Float(1e-6));
        graph.insert_node(rms);

        let normalized = cast_node(
            &mut graph,
            "normalized",
            norm_out,
            DataType::BFloat16,
            hidden,
        );
        graph.add_output(normalized);
        (graph, x_bf16, scale)
    }

    #[test]
    fn drops_casts_around_fp32_wrapped_bf16_rms_norm() {
        let hidden = 128;
        let (mut graph, x_bf16, scale) = cast_wrapped_rms_norm_graph_bf16(hidden);

        CudaDropNormalizationCasts
            .run(&mut graph, &PassContext::new())
            .unwrap();

        // Every wrapping Cast is gone.
        assert_eq!(
            graph.nodes.values().filter(|n| n.op_type == "Cast").count(),
            0,
            "all bf16 cast wrappers must be removed"
        );

        // The narrowed `RMSNormalization` is rewritten to the identical-arithmetic
        // `SimplifiedLayerNormalization` (whose output follows `X`, not the scale),
        // so re-running shape inference keeps the bf16 output instead of clobbering
        // it back to the f32 scale dtype.
        assert!(
            graph
                .nodes
                .values()
                .all(|n| n.op_type != "RMSNormalization"),
            "narrowed RMSNormalization must be converted to SimplifiedLayerNormalization"
        );
        let rms = graph
            .nodes
            .values()
            .find(|n| n.op_type == "SimplifiedLayerNormalization")
            .expect("converted norm node retained");
        assert_eq!(rms.domain, "", "converted norm stays in the ai.onnx domain");
        assert_eq!(
            rms.inputs,
            vec![Some(x_bf16), Some(scale)],
            "activation input rewired to bf16 source; scale untouched"
        );
        // Its consumed float output is retyped to bf16, matching the input.
        assert_eq!(graph.value(rms.outputs[0]).dtype, DataType::BFloat16);
        // The fp32 scale weight is preserved (kernel upcasts activation, applies
        // fp32 scale).
        assert_eq!(graph.value(scale).dtype, DataType::Float32);
        // Graph output is now the norm's bf16 output.
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.value(graph.outputs[0]).dtype, DataType::BFloat16);
        assert!(graph.validate().is_ok());

        // Re-running shape inference (as the session does after optimization) must
        // preserve the bf16 output — the whole point of the op conversion.
        let registry = onnx_runtime_shape_inference::InferenceRegistry::default_registry();
        let opsets = graph.opset_imports.clone();
        registry
            .infer_graph(
                &mut graph,
                &opsets,
                onnx_runtime_shape_inference::MergePolicy::Permissive,
            )
            .unwrap();
        let rms_out = graph
            .nodes
            .values()
            .find(|n| n.op_type == "SimplifiedLayerNormalization")
            .expect("converted norm retained after inference")
            .outputs[0];
        assert_eq!(
            graph.value(rms_out).dtype,
            DataType::BFloat16,
            "re-inference must keep the narrow (bf16) output, not the f32 scale dtype"
        );
    }

    #[test]
    fn leaves_native_bf16_rms_norm_untouched() {
        // A bf16 `RMSNormalization` already exported without cast wrappers must
        // be left byte-for-byte unchanged.
        let hidden = 128;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let x = value(&mut graph, "x", DataType::BFloat16, hidden);
        graph.add_input(x);
        let scale = vec1d(&mut graph, "scale", DataType::BFloat16, hidden);
        graph.set_initializer(
            scale,
            WeightRef::Inline(TensorData::from_raw(
                DataType::BFloat16,
                vec![hidden],
                vec![0u8; hidden * 2],
            )),
        );
        let norm_out = value(&mut graph, "norm_out", DataType::BFloat16, hidden);
        let mut rms = Node::new(
            NodeId(0),
            "RMSNormalization",
            vec![Some(x), Some(scale)],
            vec![norm_out],
        );
        rms.attributes
            .insert("epsilon".into(), Attribute::Float(1e-6));
        graph.insert_node(rms);
        graph.add_output(norm_out);

        let before = graph.nodes.len();
        CudaDropNormalizationCasts
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert_eq!(graph.nodes.len(), before, "no nodes added or removed");
        let rms = graph
            .nodes
            .values()
            .find(|n| n.op_type == "RMSNormalization")
            .expect("norm retained");
        assert_eq!(rms.inputs, vec![Some(x), Some(scale)]);
        assert_eq!(graph.value(norm_out).dtype, DataType::BFloat16);
    }

    #[test]
    fn leaves_mixed_narrow_dtype_norm_untouched() {
        // A norm whose two activation inputs are cast from *different* narrow
        // dtypes (one fp16, one bf16) is malformed for a single-dtype fold and
        // must be left exactly as exported.
        let hidden = 128;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);

        let a = value(&mut graph, "a", DataType::Float16, hidden);
        let b = value(&mut graph, "b", DataType::BFloat16, hidden);
        graph.add_input(a);
        graph.add_input(b);
        let gamma = vec1d(&mut graph, "gamma", DataType::Float32, hidden);
        graph.set_initializer(
            gamma,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![hidden],
                vec![0u8; hidden * 4],
            )),
        );

        let in0 = cast_node(&mut graph, "in0", a, DataType::Float32, hidden);
        let in1 = cast_node(&mut graph, "in1", b, DataType::Float32, hidden);
        let norm_out = value(&mut graph, "norm_out", DataType::Float32, hidden);
        let mut skip = Node::new(
            NodeId(0),
            "SkipSimplifiedLayerNormalization",
            vec![Some(in0), Some(in1), Some(gamma)],
            vec![norm_out],
        );
        skip.domain = MICROSOFT_DOMAIN.into();
        graph.insert_node(skip);
        let normalized = cast_node(
            &mut graph,
            "normalized",
            norm_out,
            DataType::Float16,
            hidden,
        );
        graph.add_output(normalized);

        let casts_before = graph.nodes.values().filter(|n| n.op_type == "Cast").count();
        CudaDropNormalizationCasts
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert_eq!(
            graph.nodes.values().filter(|n| n.op_type == "Cast").count(),
            casts_before,
            "a mixed fp16/bf16 activation norm must not be folded"
        );
        assert_eq!(graph.value(norm_out).dtype, DataType::Float32);
        assert!(graph.validate().is_ok());
    }

    // ---- CudaOnDeviceConstantSelect ----------------------------------------

    fn fp16_bytes(rows: usize, cols: usize, fill: u16) -> Vec<u8> {
        (0..rows * cols).flat_map(|_| fill.to_le_bytes()).collect()
    }

    const THEN_COS_SENTINEL: u16 = 0x3C00; // 1.0
    const THEN_SIN_SENTINEL: u16 = 0x4000; // 2.0
    const ELSE_COS_SENTINEL: u16 = 0x4200; // 3.0
    const ELSE_SIN_SENTINEL: u16 = 0x4400; // 4.0

    fn constant_branch(outputs: &[(&str, DataType, Vec<usize>, Vec<u8>)]) -> Graph {
        let mut branch = Graph::new();
        for (name, dtype, dims, bytes) in outputs {
            let out = branch.create_named_value(
                *name,
                *dtype,
                dims.iter().map(|&d| Dim::Static(d)).collect::<Vec<_>>(),
            );
            branch.add_output(out);
            let mut node = Node::new(NodeId(0), "Constant", vec![], vec![out]);
            node.attributes.insert(
                "value".into(),
                Attribute::Tensor(TensorData::from_raw(*dtype, dims.clone(), bytes.clone())),
            );
            branch.insert_node(node);
        }
        branch
    }

    /// Build `Greater(seq_len, threshold) → If → (cos, sin)` with pure-constant
    /// branches. `cond_from_greater` toggles whether the predicate is produced by
    /// a `Greater` (with a scalar-int threshold initializer) or is a plain bool
    /// graph input (exercising the equal-shape path that needs no threshold tie).
    fn longrope_if_graph(
        then_dims: Vec<usize>,
        else_dims: Vec<usize>,
        threshold: i64,
        cond_from_greater: bool,
    ) -> (Graph, ValueId, ValueId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);

        let cond = if cond_from_greater {
            let seq = graph.create_named_value("seq_len", DataType::Int64, Vec::new());
            graph.add_input(seq);
            let thr = graph.create_named_value("threshold", DataType::Int64, Vec::new());
            graph.set_initializer(
                thr,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Int64,
                    Vec::new(),
                    threshold.to_le_bytes().to_vec(),
                )),
            );
            let cond = graph.create_named_value("cond", DataType::Bool, Vec::new());
            graph.insert_node(Node::new(
                NodeId(0),
                "Greater",
                vec![Some(seq), Some(thr)],
                vec![cond],
            ));
            cond
        } else {
            let cond = graph.create_named_value("cond", DataType::Bool, Vec::new());
            graph.add_input(cond);
            cond
        };

        let cos = graph.create_named_value(
            "cos_cache",
            DataType::Float16,
            then_dims
                .iter()
                .map(|&d| Dim::Static(d))
                .collect::<Vec<_>>(),
        );
        let sin = graph.create_named_value(
            "sin_cache",
            DataType::Float16,
            then_dims
                .iter()
                .map(|&d| Dim::Static(d))
                .collect::<Vec<_>>(),
        );
        let if_node =
            graph.insert_node(Node::new(NodeId(0), "If", vec![Some(cond)], vec![cos, sin]));
        graph.add_output(cos);
        graph.add_output(sin);

        let then_cos = fp16_bytes(then_dims[0], then_dims[1], THEN_COS_SENTINEL);
        let then_sin = fp16_bytes(then_dims[0], then_dims[1], THEN_SIN_SENTINEL);
        let else_cos = fp16_bytes(else_dims[0], else_dims[1], ELSE_COS_SENTINEL);
        let else_sin = fp16_bytes(else_dims[0], else_dims[1], ELSE_SIN_SENTINEL);
        graph.subgraphs.insert(
            (if_node, "then_branch".into()),
            constant_branch(&[
                ("cos_large", DataType::Float16, then_dims.clone(), then_cos),
                ("sin_large", DataType::Float16, then_dims.clone(), then_sin),
            ]),
        );
        graph.subgraphs.insert(
            (if_node, "else_branch".into()),
            constant_branch(&[
                ("cos_small", DataType::Float16, else_dims.clone(), else_cos),
                ("sin_small", DataType::Float16, else_dims.clone(), else_sin),
            ]),
        );
        (graph, cos, sin)
    }

    fn where_nodes(graph: &Graph) -> Vec<NodeId> {
        graph
            .nodes
            .iter()
            .filter_map(|(id, n)| (n.op_type == "Where").then_some(id))
            .collect()
    }

    #[test]
    fn lowers_differing_shape_longrope_if_to_padded_where() {
        // then (predicate-true / long) = [4,2], else (false / short) = [2,2],
        // threshold = 2 == short leading dim.
        let (mut graph, cos, sin) = longrope_if_graph(vec![4, 2], vec![2, 2], 2, true);
        CudaOnDeviceConstantSelect
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert!(
            graph.nodes.values().all(|n| n.op_type != "If"),
            "the host If must be gone"
        );
        assert!(graph.subgraphs.is_empty(), "If subgraphs must be removed");
        let wheres = where_nodes(&graph);
        assert_eq!(wheres.len(), 2, "one Where per If output");

        // Both cache outputs now have the long shape and are Where-produced.
        assert_eq!(graph.value(cos).shape, static_shape([4, 2]));
        assert_eq!(graph.value(sin).shape, static_shape([4, 2]));

        for &w in &wheres {
            let node = graph.node(w);
            // cond wired to input 0; x/y are constant initializers.
            let x = node.inputs[1].unwrap();
            let y = node.inputs[2].unwrap();
            let x_bytes = match graph.initializers.get(&x).unwrap() {
                WeightRef::Inline(t) => &t.data,
                _ => panic!("x must be inline"),
            };
            let y_bytes = match graph.initializers.get(&y).unwrap() {
                WeightRef::Inline(t) => &t.data,
                _ => panic!("y must be inline"),
            };
            let (then_sentinel, else_sentinel) = match node.outputs[0] {
                output if output == cos => (THEN_COS_SENTINEL, ELSE_COS_SENTINEL),
                output if output == sin => (THEN_SIN_SENTINEL, ELSE_SIN_SENTINEL),
                output => panic!("unexpected Where output {output:?}"),
            };
            assert_eq!(
                x_bytes.len(),
                4 * 2 * 2,
                "then const is the full long table"
            );
            assert_eq!(
                y_bytes.len(),
                4 * 2 * 2,
                "else const padded to the long shape"
            );
            assert_eq!(
                x_bytes,
                &fp16_bytes(4, 2, then_sentinel),
                "x must carry the predicate-true branch values"
            );
            // First 2 rows preserved from the short table; the appended 2 rows are
            // zeros (provably never indexed when the predicate is false).
            assert_eq!(
                &y_bytes[..2 * 2 * 2],
                fp16_bytes(2, 2, else_sentinel),
                "y must carry the predicate-false branch values"
            );
            assert!(
                y_bytes[2 * 2 * 2..].iter().all(|&b| b == 0),
                "appended rows are zero padding"
            );
        }
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn lowers_equal_shape_if_without_threshold() {
        // Equal-shaped branches: no Greater / threshold tie required.
        let (mut graph, _cos, _sin) = longrope_if_graph(vec![3, 2], vec![3, 2], 0, false);
        CudaOnDeviceConstantSelect
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(graph.nodes.values().all(|n| n.op_type != "If"));
        let wheres = where_nodes(&graph);
        assert_eq!(wheres.len(), 2);
        for &w in &wheres {
            let node = graph.node(w);
            let y = node.inputs[2].unwrap();
            let y_bytes = match graph.initializers.get(&y).unwrap() {
                WeightRef::Inline(t) => &t.data,
                _ => panic!(),
            };
            assert_eq!(y_bytes.len(), 3 * 2 * 2, "no padding for equal shapes");
        }
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn skips_if_with_non_constant_branch() {
        let (mut graph, _cos, _sin) = longrope_if_graph(vec![4, 2], vec![2, 2], 2, true);
        // Corrupt the then branch: replace one Constant with an Add so the branch
        // is no longer a pure constant selection.
        let key = (
            graph
                .nodes
                .iter()
                .find_map(|(id, n)| (n.op_type == "If").then_some(id))
                .unwrap(),
            "then_branch".to_string(),
        );
        let branch = graph.subgraphs.get_mut(&key).unwrap();
        let victim = branch
            .nodes
            .iter()
            .find_map(|(id, n)| (n.op_type == "Constant").then_some(id))
            .unwrap();
        branch.node_mut(victim).op_type = "Add".into();

        CudaOnDeviceConstantSelect
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(
            graph.nodes.values().any(|n| n.op_type == "If"),
            "a non-constant branch must not be rewritten"
        );
        assert!(where_nodes(&graph).is_empty());
    }

    #[test]
    fn skips_padded_if_when_threshold_mismatches_short_table() {
        // else leading dim (2) != threshold (3): padded rows are not provably
        // unread, so the rewrite must not fire.
        let (mut graph, _cos, _sin) = longrope_if_graph(vec![4, 2], vec![2, 2], 3, true);
        CudaOnDeviceConstantSelect
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(graph.nodes.values().any(|n| n.op_type == "If"));
        assert!(where_nodes(&graph).is_empty());
    }

    #[test]
    fn skips_padded_if_without_greater_predicate() {
        // Differing shapes but the predicate is a plain bool input (no threshold
        // tie available): must not rewrite.
        let (mut graph, _cos, _sin) = longrope_if_graph(vec![4, 2], vec![2, 2], 2, false);
        CudaOnDeviceConstantSelect
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(graph.nodes.values().any(|n| n.op_type == "If"));
        assert!(where_nodes(&graph).is_empty());
    }

    #[test]
    fn skips_if_when_true_branch_is_smaller() {
        // then (predicate-true) smaller than else: selecting the padded true
        // branch on a long sequence could read appended rows — must not rewrite.
        let (mut graph, _cos, _sin) = longrope_if_graph(vec![2, 2], vec![4, 2], 2, true);
        CudaOnDeviceConstantSelect
            .run(&mut graph, &PassContext::new())
            .unwrap();
        assert!(graph.nodes.values().any(|n| n.op_type == "If"));
        assert!(where_nodes(&graph).is_empty());
    }

    // === QKV projection fusion ===

    struct QkvGraph {
        graph: Graph,
        q_weight: ValueId,
        k_weight: ValueId,
        v_weight: ValueId,
        q_out: ValueId,
        k_out: ValueId,
        v_out: ValueId,
    }

    /// Build one `q_proj`/`k_proj`/`v_proj` → (Reshape, Reshape, direct) → GQA
    /// block over a shared activation, with block-32 4-bit weights. Each
    /// projection's weight/scale/zero-point bytes are filled with a distinct
    /// constant so concatenation order is checkable.
    fn qkv_graph(k: usize, nq: usize, nk: usize, nv: usize, with_zp: bool) -> QkvGraph {
        assert!(k.is_multiple_of(32));
        let n_blocks = k / 32;
        let blob = 16; // 32 * 4 / 8
        let dt = DataType::BFloat16;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);

        let x = graph.create_named_value("x", dt, vec![Dim::Static(1), Dim::Static(k)]);
        graph.add_input(x);

        let make_proj = |graph: &mut Graph, tag: &str, n: usize, fill: u8| -> (ValueId, ValueId) {
            let w = graph.create_named_value(
                format!("{tag}.weight"),
                DataType::Uint8,
                vec![Dim::Static(n), Dim::Static(n_blocks), Dim::Static(blob)],
            );
            graph.set_initializer(
                w,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Uint8,
                    vec![n, n_blocks, blob],
                    vec![fill; n * n_blocks * blob],
                )),
            );
            let s = graph.create_named_value(
                format!("{tag}.scales"),
                dt,
                vec![Dim::Static(n * n_blocks)],
            );
            graph.set_initializer(
                s,
                WeightRef::Inline(TensorData::from_raw(
                    dt,
                    vec![n * n_blocks],
                    vec![fill.wrapping_add(1); n * n_blocks * 2],
                )),
            );
            let mut inputs = vec![Some(x), Some(w), Some(s)];
            if with_zp {
                let zp_bytes = n * n_blocks / 2;
                let z = graph.create_named_value(
                    format!("{tag}.zp"),
                    DataType::Uint8,
                    vec![Dim::Static(zp_bytes)],
                );
                graph.set_initializer(
                    z,
                    WeightRef::Inline(TensorData::from_raw(
                        DataType::Uint8,
                        vec![zp_bytes],
                        vec![fill.wrapping_add(2); zp_bytes],
                    )),
                );
                inputs.push(Some(z));
            }
            let out = graph.create_named_value(
                format!("{tag}.out"),
                dt,
                vec![Dim::Static(1), Dim::Static(n)],
            );
            let mut mm = Node::new(NodeId(0), "MatMulNBits", inputs, vec![out]);
            mm.domain = MICROSOFT_DOMAIN.into();
            mm.attributes.insert("K".into(), Attribute::Int(k as i64));
            mm.attributes.insert("N".into(), Attribute::Int(n as i64));
            mm.attributes
                .insert("block_size".into(), Attribute::Int(32));
            mm.attributes.insert("bits".into(), Attribute::Int(4));
            graph.insert_node(mm);
            (w, out)
        };

        let (q_weight, q_out) = make_proj(&mut graph, "q", nq, 0x11);
        let (k_weight, k_out) = make_proj(&mut graph, "k", nk, 0x22);
        let (v_weight, v_out) = make_proj(&mut graph, "v", nv, 0x33);

        // Q and K flow through a Reshape before attention; V feeds it directly.
        let q_res = graph.create_named_value("q_res", dt, vec![Dim::Static(1), Dim::Static(nq)]);
        graph.insert_node(Node::new(
            NodeId(0),
            "Reshape",
            vec![Some(q_out)],
            vec![q_res],
        ));
        let k_res = graph.create_named_value("k_res", dt, vec![Dim::Static(1), Dim::Static(nk)]);
        graph.insert_node(Node::new(
            NodeId(0),
            "Reshape",
            vec![Some(k_out)],
            vec![k_res],
        ));

        let attn = graph.create_named_value("attn", dt, vec![Dim::Static(1), Dim::Static(nq)]);
        graph.add_output(attn);
        let mut gqa = Node::new(
            NodeId(0),
            "GroupQueryAttention",
            vec![Some(q_res), Some(k_res), Some(v_out)],
            vec![attn],
        );
        gqa.domain = MICROSOFT_DOMAIN.into();
        graph.insert_node(gqa);

        QkvGraph {
            graph,
            q_weight,
            k_weight,
            v_weight,
            q_out,
            k_out,
            v_out,
        }
    }

    fn inline_bytes(graph: &Graph, value: ValueId) -> &[u8] {
        match graph.initializers.get(&value).unwrap() {
            WeightRef::Inline(t) => &t.data,
            WeightRef::External { .. } => panic!("expected inline"),
        }
    }

    #[test]
    fn fuses_qkv_projections_into_one_matmul_and_split() {
        let mut g = qkv_graph(64, 8, 4, 4, true);
        let (q_out, k_out, v_out) = (g.q_out, g.k_out, g.v_out);
        CudaQkvProjectionFusion
            .fuse_all(&mut g.graph, &PassContext::new())
            .unwrap();

        let matmuls: Vec<_> = g
            .graph
            .nodes
            .values()
            .filter(|n| n.op_type == "MatMulNBits")
            .collect();
        assert_eq!(matmuls.len(), 1, "three projections collapse to one GEMV");
        let fused = matmuls[0];
        assert_eq!(fused.attr("N").and_then(Attribute::as_int), Some(16));
        assert_eq!(fused.attr("K").and_then(Attribute::as_int), Some(64));
        assert_eq!(
            fused.input_values().count(),
            4,
            "activation+weight+scales+zp"
        );

        let splits: Vec<_> = g
            .graph
            .nodes
            .values()
            .filter(|n| n.op_type == "Split")
            .collect();
        assert_eq!(splits.len(), 1);
        let split = splits[0];
        assert_eq!(
            split.attr("split").and_then(Attribute::as_ints),
            Some([8i64, 4, 4].as_slice())
        );
        assert_eq!(split.attr("axis").and_then(Attribute::as_int), Some(1));
        // Split re-attaches the original consumer edges.
        assert_eq!(split.outputs, vec![q_out, k_out, v_out]);

        // Fused weight bytes are the byte concatenation of Q||K||V weights.
        let fused_weight = fused.inputs[1].unwrap();
        let bytes = inline_bytes(&g.graph, fused_weight);
        let n_blocks = 2usize;
        let blob = 16usize;
        assert_eq!(bytes.len(), 16 * n_blocks * blob);
        assert!(bytes[..8 * n_blocks * blob].iter().all(|&b| b == 0x11));
        assert!(
            bytes[8 * n_blocks * blob..12 * n_blocks * blob]
                .iter()
                .all(|&b| b == 0x22)
        );
        assert!(bytes[12 * n_blocks * blob..].iter().all(|&b| b == 0x33));

        // The original per-projection weight initializers are retired.
        for old in [g.q_weight, g.k_weight, g.v_weight] {
            assert!(!g.graph.initializers.contains_key(&old));
        }

        g.graph.validate().unwrap();
    }

    #[test]
    fn fuses_symmetric_qkv_without_zero_points() {
        let mut g = qkv_graph(32, 6, 2, 2, false);
        CudaQkvProjectionFusion
            .fuse_all(&mut g.graph, &PassContext::new())
            .unwrap();
        let fused = g
            .graph
            .nodes
            .values()
            .find(|n| n.op_type == "MatMulNBits")
            .unwrap();
        assert_eq!(fused.attr("N").and_then(Attribute::as_int), Some(10));
        assert_eq!(fused.input_values().count(), 3, "no zero-point slot");
        g.graph.validate().unwrap();
    }

    #[test]
    fn does_not_fuse_when_activations_differ() {
        let mut g = qkv_graph(64, 8, 4, 4, true);
        // Repoint K's projection at a different activation.
        let other = g.graph.create_named_value(
            "other",
            DataType::BFloat16,
            vec![Dim::Static(1), Dim::Static(64)],
        );
        g.graph.add_input(other);
        let k_mm = g.graph.value(g.k_out).producer.unwrap();
        g.graph.replace_input(k_mm, 0, Some(other));

        CudaQkvProjectionFusion
            .fuse_all(&mut g.graph, &PassContext::new())
            .unwrap();
        let matmuls = g
            .graph
            .nodes
            .values()
            .filter(|n| n.op_type == "MatMulNBits")
            .count();
        assert_eq!(matmuls, 3, "mismatched activation leaves projections split");
    }

    #[test]
    fn qkv_fusion_is_opt_in_and_disabled_by_default() {
        // The pass must not fire unless the opt-in env flag is set, so the
        // default release binary keeps the three separate GEMVs (no regression).
        // SAFETY: single-threaded test; env is restored before returning.
        let mut g = qkv_graph(64, 8, 4, 4, true);
        unsafe { std::env::remove_var(QKV_FUSION_ENABLE_ENV) };
        CudaQkvProjectionFusion
            .run(&mut g.graph, &PassContext::new())
            .unwrap();
        let unfused = g
            .graph
            .nodes
            .values()
            .filter(|n| n.op_type == "MatMulNBits")
            .count();
        assert_eq!(
            unfused, 3,
            "default (flag unset) leaves projections unfused"
        );

        // With the flag set, `run` performs the fusion.
        unsafe { std::env::set_var(QKV_FUSION_ENABLE_ENV, "1") };
        let result = CudaQkvProjectionFusion.run(&mut g.graph, &PassContext::new());
        unsafe { std::env::remove_var(QKV_FUSION_ENABLE_ENV) };
        result.unwrap();
        let fused = g
            .graph
            .nodes
            .values()
            .filter(|n| n.op_type == "MatMulNBits")
            .count();
        assert_eq!(fused, 1, "opt-in flag enables the fusion");
    }
}
