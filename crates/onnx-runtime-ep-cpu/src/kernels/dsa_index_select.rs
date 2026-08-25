//! CPU reference kernel for frozen `pkg.nxrt::DsaIndexSelect` v1.
//!
//! `DsaIndexSelect` is the *index-selection* half of GLM-5.2 / DeepSeek Sparse
//! Attention (DSA). Its output, `selected_indices`, is consumed by the
//! *sparse-attention* half, `pkg.nxrt::IndexShare`. Together they express the
//! query-dependent sparse attention that PagedAttention cannot (PagedAttention
//! has no query-dependent sparse-index input).
//!
//! # Why this op exists (capture-safety value-add)
//!
//! The Mobius decomposition emits the indexer as generic ops ending in
//! `k = Min(key_length, index_topk)` — a *data-dependent* TopK width that
//! changes every decode step. A CUDA graph cannot capture a kernel whose output
//! width is read from a device tensor at run time, so the decomposed path is
//! capture-hostile. `DsaIndexSelect` fuses the indexer scoring + causal mask +
//! top-k + ascending sort into one kernel with a **stable output width**
//! (always `top_k`), padding short rows with the `-1` sentinel that `IndexShare`
//! already understands. Stable width ⇒ CUDA-graph capture/replay safe.
//!
//! # Semantics (faithful to Mobius `glm_moe_dsa.py::select`)
//!
//! For each batch `b` and query row `s`:
//!   * `dot(h, t)      = Σ_d query[b,s,h,d] · key[b,t,d]`
//!   * `score(h, t)    = relu(scale · dot(h, t))`
//!   * `weighted(t)    = Σ_h score(h, t) · (weights[b,s,h] · weights_scale)`
//!   * `masked(t)      = weighted(t) + attention_bias[b,0,s,t]`
//!   * a position `t` is *allowed* iff `attention_bias[b,0,s,t] > -1e30`
//!     (i.e. not the `-inf` / `finfo.min` causal fill),
//!   * select the top `min(#allowed, top_k)` allowed positions by
//!     `(masked(t) descending, t ascending)`, sort them **ascending by `t`**,
//!     and right-pad with `-1` to width `top_k`.
//!
//! This produces the same *valid* (finite-score) selections as the decomposed
//! `Min(key_length, index_topk)` path: `IndexShare` re-applies the causal bias,
//! so a masked-future slot (decomposed) and a `-1` slot (fused) both contribute
//! zero to the attention output. The tie rule (equal score ⇒ lower index) mirrors
//! ONNX Runtime `TopK`, keeping CPU and CUDA bit-identical.

use std::borrow::Cow;
use std::cmp::Ordering;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, Shape};

use super::{check_arity, write_dense_bytes};
use crate::dtype::to_dense_f32_widen;

const OP: &str = "DsaIndexSelect";

const INPUT_NAMES: [&str; 4] = ["query", "key", "weights", "attention_bias"];

/// Bias values at or below this magnitude are treated as `-inf` causal fill
/// (`-inf` and `f32::MIN`/`torch.finfo.min` ≈ `-3.4e38` both qualify).
const MASK_THRESHOLD: f32 = -1e30;

pub struct DsaIndexSelectFactory;

pub struct DsaIndexSelectKernel {
    top_k: usize,
    scale: f32,
    weights_scale: f32,
}

#[derive(Clone, Copy)]
struct Attributes {
    top_k: usize,
    scale: f32,
    weights_scale: f32,
}

impl KernelFactory for DsaIndexSelectFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let attrs = validate_metadata(node, None)?;
        Ok(Box::new(DsaIndexSelectKernel {
            top_k: attrs.top_k,
            scale: attrs.scale,
            weights_scale: attrs.weights_scale,
        }))
    }
}

/// Claim-time gate shared with the CUDA execution provider so the device kernel
/// rejects exactly the attr/dtype/rank/shape combinations the CPU oracle does,
/// keeping the two backends' `supports_op` contracts in lockstep.
pub fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    validate_metadata(node, Some((shapes, input_dtypes)))
        .err()
        .map(|error| Cow::Owned(error.to_string()))
}

#[derive(Clone, Copy)]
struct Dims {
    batch: usize,
    q_seq: usize,
    heads: usize,
    head_dim: usize,
    key_seq: usize,
}

impl DsaIndexSelectKernel {
    fn derive_dims(&self, inputs: &[TensorView]) -> Result<Dims> {
        let query = inputs[0].shape;
        let key = inputs[1].shape;
        let weights = inputs[2].shape;
        let bias = inputs[3].shape;
        if query.len() != 4 {
            return Err(error(format!(
                "input 0 ('query') rank {} unsupported; expected 4 (B, S, H, D)",
                query.len()
            )));
        }
        if key.len() != 3 {
            return Err(error(format!(
                "input 1 ('key') rank {} unsupported; expected 3 (B, T, D)",
                key.len()
            )));
        }
        if weights.len() != 3 {
            return Err(error(format!(
                "input 2 ('weights') rank {} unsupported; expected 3 (B, S, H)",
                weights.len()
            )));
        }
        if bias.len() != 4 {
            return Err(error(format!(
                "input 3 ('attention_bias') rank {} unsupported; expected 4 (B, 1, S, T)",
                bias.len()
            )));
        }
        let dims = Dims {
            batch: query[0],
            q_seq: query[1],
            heads: query[2],
            head_dim: query[3],
            key_seq: key[1],
        };
        require_eq("query/key batch", query[0], key[0])?;
        require_eq("query/weights batch", query[0], weights[0])?;
        require_eq("query/bias batch", query[0], bias[0])?;
        require_eq("query/weights seq", dims.q_seq, weights[1])?;
        require_eq("query/weights heads", dims.heads, weights[2])?;
        require_eq("query/key head_dim", dims.head_dim, key[2])?;
        if bias[1] != 1 {
            return Err(error(format!(
                "input 3 ('attention_bias') dim 1 must be 1 (head-broadcast), got {}",
                bias[1]
            )));
        }
        require_eq("bias/query seq", dims.q_seq, bias[2])?;
        require_eq("bias/key seq", dims.key_seq, bias[3])?;
        Ok(dims)
    }
}

impl Kernel for DsaIndexSelectKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 4, 4, 1)?;
        for (index, name) in INPUT_NAMES.iter().enumerate() {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{name}') is absent"
                )));
            }
            if !is_supported_float(inputs[index].dtype) {
                return Err(error(format!(
                    "input {index} ('{name}') dtype {:?} unsupported; expected Float32, Float16, or BFloat16",
                    inputs[index].dtype
                )));
            }
        }
        // query/key/weights share one element type; the bias is an additive f32
        // mask that may arrive at a different (also-float) precision.
        for index in [1, 2] {
            if inputs[index].dtype != inputs[0].dtype {
                return Err(error(format!(
                    "input {index} ('{}') dtype {:?} must match query dtype {:?}",
                    INPUT_NAMES[index], inputs[index].dtype, inputs[0].dtype
                )));
            }
        }
        if outputs[0].dtype != DataType::Int64 {
            return Err(error(format!(
                "output 0 ('selected_indices') dtype {:?} unsupported; expected Int64",
                outputs[0].dtype
            )));
        }

        let dims = self.derive_dims(inputs)?;
        let expected_out = [dims.batch, 1, dims.q_seq, self.top_k];
        if outputs[0].shape != expected_out {
            return Err(error(format!(
                "output 0 ('selected_indices') shape {:?} unsupported; expected {expected_out:?} (B, 1, S, top_k)",
                outputs[0].shape
            )));
        }

        let query = to_dense_f32_widen(OP, &inputs[0])?;
        let key = to_dense_f32_widen(OP, &inputs[1])?;
        let weights = to_dense_f32_widen(OP, &inputs[2])?;
        let bias = to_dense_f32_widen(OP, &inputs[3])?;

        let Dims {
            batch,
            q_seq,
            heads,
            head_dim,
            key_seq,
        } = dims;

        let mut out = vec![-1i64; batch * q_seq * self.top_k];
        // Reused per-row scratch so a large prefill does not thrash the allocator.
        let mut candidates: Vec<(f32, usize)> = Vec::with_capacity(key_seq);

        for b in 0..batch {
            for s in 0..q_seq {
                candidates.clear();
                let weights_base = (b * q_seq + s) * heads;
                let bias_base = (b * q_seq + s) * key_seq; // bias[b,0,s,·], dim1 == 1
                for t in 0..key_seq {
                    let bias_bt = bias[bias_base + t];
                    // NaN and the `-inf`/finfo.min causal fill are both "not
                    // allowed"; binding the comparison keeps the negation on a
                    // bool (partial-order safe) rather than on the float compare.
                    let allowed = bias_bt > MASK_THRESHOLD;
                    if !allowed {
                        continue; // masked / -inf causal fill ⇒ never selected
                    }
                    let mut weighted = 0.0f32;
                    for h in 0..heads {
                        let q_base = ((b * q_seq + s) * heads + h) * head_dim;
                        let k_base = (b * key_seq + t) * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += query[q_base + d] * key[k_base + d];
                        }
                        let scored = (self.scale * dot).max(0.0); // Relu
                        weighted += scored * (weights[weights_base + h] * self.weights_scale);
                    }
                    candidates.push((weighted + bias_bt, t));
                }

                let keep = candidates.len().min(self.top_k);
                if keep == 0 {
                    continue; // all positions masked ⇒ full -1 row (caller's causal guarantee)
                }
                // Largest `keep` by (score desc, index asc); partial-select then
                // sort only the winners, mirroring the ep's TopK introselect.
                if keep < candidates.len() {
                    candidates.select_nth_unstable_by(keep - 1, |a, b| score_order(*a, *b));
                    candidates.truncate(keep);
                }
                // IndexShare requires strictly-increasing positions per row.
                candidates.sort_unstable_by_key(|&(_, t)| t);
                let row_base = (b * q_seq + s) * self.top_k;
                for (slot, &(_, t)) in candidates.iter().enumerate() {
                    out[row_base + slot] = t as i64;
                }
            }
        }

        write_dense_bytes(
            &mut outputs[0],
            &out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )
    }
}

/// Total order over `(score, index)` candidates: highest score wins, ties broken
/// by lower position index (matching ONNX Runtime `TopK`). `total_cmp` keeps NaN
/// handling deterministic and identical to the CUDA kernel.
fn score_order(a: (f32, usize), b: (f32, usize)) -> Ordering {
    match b.0.total_cmp(&a.0) {
        Ordering::Equal => a.1.cmp(&b.1),
        order => order,
    }
}

fn validate_metadata(
    node: &Node,
    claim_metadata: Option<(&[Shape], &[DataType])>,
) -> Result<Attributes> {
    for name in node.attributes.keys() {
        if !matches!(name.as_str(), "top_k" | "scale" | "weights_scale") {
            return Err(error(format!(
                "attribute '{name}' is not part of the frozen v1 ABI"
            )));
        }
    }
    let top_k = required_positive_int(node, "top_k")?;
    let scale = required_finite_positive_float(node, "scale")?;
    let weights_scale = match node.attr("weights_scale") {
        Some(attribute) => {
            let value = attribute
                .as_float()
                .ok_or_else(|| error("attribute 'weights_scale' must be a float"))?;
            if !value.is_finite() || value <= 0.0 {
                return Err(error("attribute 'weights_scale' must be finite and > 0"));
            }
            value
        }
        None => 1.0,
    };
    let attrs = Attributes {
        top_k,
        scale,
        weights_scale,
    };
    if let Some((shapes, dtypes)) = claim_metadata {
        validate_claim_metadata(node, shapes, dtypes, attrs).map_err(error)?;
    }
    Ok(attrs)
}

fn validate_claim_metadata(
    node: &Node,
    shapes: &[Shape],
    dtypes: &[DataType],
    attrs: Attributes,
) -> std::result::Result<(), String> {
    if node.inputs.len() != 4 {
        return Err(format!(
            "expected 4 positional inputs, got {}",
            node.inputs.len()
        ));
    }
    if node.outputs.len() != 1 {
        return Err(format!("expected 1 output, got {}", node.outputs.len()));
    }
    if shapes.len() != node.inputs.len() || dtypes.len() != node.inputs.len() {
        return Err(format!(
            "claim metadata must cover all {} positional inputs",
            node.inputs.len()
        ));
    }
    for (index, name) in INPUT_NAMES.iter().enumerate() {
        if node.inputs[index].is_none() {
            return Err(format!("required input {index} ('{name}') is omitted"));
        }
        if !is_supported_float(dtypes[index]) {
            return Err(format!(
                "input {index} ('{name}') dtype {:?} unsupported; expected Float32, Float16, or BFloat16",
                dtypes[index]
            ));
        }
    }
    for index in [1, 2] {
        if dtypes[index] != dtypes[0] {
            return Err(format!(
                "input {index} ('{}') dtype {:?} must match query dtype {:?}",
                INPUT_NAMES[index], dtypes[index], dtypes[0]
            ));
        }
    }
    let expected_ranks = [4usize, 3, 3, 4];
    for (index, &rank) in expected_ranks.iter().enumerate() {
        if shapes[index].len() != rank {
            return Err(format!(
                "input {index} ('{}') rank {} unsupported; expected {rank}",
                INPUT_NAMES[index],
                shapes[index].len()
            ));
        }
    }
    // Static cross-input consistency (dynamic dims defer to run-time checks).
    require_same_static(&shapes[0], 0, &shapes[1], 0, "query/key batch")?;
    require_same_static(&shapes[0], 0, &shapes[2], 0, "query/weights batch")?;
    require_same_static(&shapes[0], 0, &shapes[3], 0, "query/bias batch")?;
    require_same_static(&shapes[0], 1, &shapes[2], 1, "query/weights seq")?;
    require_same_static(&shapes[0], 2, &shapes[2], 2, "query/weights heads")?;
    require_same_static(&shapes[0], 3, &shapes[1], 2, "query/key head_dim")?;
    require_same_static(&shapes[0], 1, &shapes[3], 2, "query/bias seq")?;
    require_same_static(&shapes[1], 1, &shapes[3], 3, "key/bias seq")?;
    check_static_dim(&shapes[3], 1, 1, "attention_bias head-broadcast")?;
    // Static output width, if declared, must equal `top_k`.
    let _ = attrs;
    Ok(())
}

fn is_supported_float(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    )
}

fn required_positive_int(node: &Node, name: &str) -> Result<usize> {
    let value = node
        .attr(name)
        .ok_or_else(|| error(format!("missing required integer attribute '{name}'")))?
        .as_int()
        .ok_or_else(|| error(format!("attribute '{name}' must be an integer")))?;
    usize::try_from(value)
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| error(format!("attribute '{name}' must be > 0")))
}

fn required_finite_positive_float(node: &Node, name: &str) -> Result<f32> {
    let value = node
        .attr(name)
        .ok_or_else(|| error(format!("missing required float attribute '{name}'")))?
        .as_float()
        .ok_or_else(|| error(format!("attribute '{name}' must be a float")))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(error(format!("attribute '{name}' must be finite and > 0")));
    }
    Ok(value)
}

fn require_eq(name: &str, left: usize, right: usize) -> Result<()> {
    if left != right {
        return Err(error(format!(
            "{name} dimensions differ: {left} vs {right}"
        )));
    }
    Ok(())
}

fn check_static_dim(
    shape: &Shape,
    axis: usize,
    expected: usize,
    name: &str,
) -> std::result::Result<(), String> {
    if let Some(actual) = shape[axis].as_static()
        && actual != expected
    {
        return Err(format!("{name} must be {expected}, got {actual}"));
    }
    Ok(())
}

fn require_same_static(
    left: &Shape,
    left_axis: usize,
    right: &Shape,
    right_axis: usize,
    name: &str,
) -> std::result::Result<(), String> {
    if let (Some(left), Some(right)) = (left[left_axis].as_static(), right[right_axis].as_static())
        && left != right
    {
        return Err(format!("{name} dimensions differ: {left} vs {right}"));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OP}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};

    const NEG: f32 = f32::NEG_INFINITY;

    #[derive(Clone, Copy)]
    struct Case {
        batch: usize,
        q_seq: usize,
        heads: usize,
        head_dim: usize,
        key_seq: usize,
        top_k: usize,
        scale: f32,
        weights_scale: Option<f32>,
    }

    fn node(case: Case, dtype: DataType, out_dtype: DataType) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let specs = [
            (
                dtype,
                vec![case.batch, case.q_seq, case.heads, case.head_dim],
            ),
            (dtype, vec![case.batch, case.key_seq, case.head_dim]),
            (dtype, vec![case.batch, case.q_seq, case.heads]),
            (
                DataType::Float32,
                vec![case.batch, 1, case.q_seq, case.key_seq],
            ),
        ];
        let inputs = specs
            .iter()
            .enumerate()
            .map(|(index, (dtype, shape))| {
                let value = graph.create_named_value(
                    format!("input_{index}"),
                    *dtype,
                    static_shape(shape.iter().copied()),
                );
                graph.add_input(value);
                Some(value)
            })
            .collect();
        let output = graph.create_named_value(
            "selected_indices",
            out_dtype,
            static_shape([case.batch, 1, case.q_seq, case.top_k]),
        );
        let mut n = Node::new(NodeId(0), OP, inputs, vec![output]);
        n.domain = "pkg.nxrt".into();
        n.attributes
            .insert("top_k".into(), Attribute::Int(case.top_k as i64));
        n.attributes
            .insert("scale".into(), Attribute::Float(case.scale));
        if let Some(ws) = case.weights_scale {
            n.attributes
                .insert("weights_scale".into(), Attribute::Float(ws));
        }
        let id = graph.insert_node(n);
        (graph, id)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_dtype(
        case: Case,
        dtype: DataType,
        query: &[f32],
        key: &[f32],
        weights: &[f32],
        bias: &[f32],
    ) -> Result<Vec<i64>> {
        let (graph, id) = node(case, dtype, DataType::Int64);
        let kernel = DsaIndexSelectFactory.create(graph.node(id), &[])?;
        let make = |shape: &[usize], data: &[f32]| match dtype {
            DataType::Float32 => Owned::f32(shape, data),
            DataType::Float16 => Owned::f16(shape, data),
            DataType::BFloat16 => Owned::bf16(shape, data),
            other => panic!("unsupported test dtype {other:?}"),
        };
        let query = make(&[case.batch, case.q_seq, case.heads, case.head_dim], query);
        let key = make(&[case.batch, case.key_seq, case.head_dim], key);
        let weights = make(&[case.batch, case.q_seq, case.heads], weights);
        let bias = Owned::f32(&[case.batch, 1, case.q_seq, case.key_seq], bias);
        let inputs = vec![query.view(), key.view(), weights.view(), bias.view()];
        let mut out = Owned::zeros(DataType::Int64, &[case.batch, 1, case.q_seq, case.top_k]);
        kernel.execute(&inputs, &mut [out.view_mut()])?;
        Ok(out.to_i64())
    }

    fn run(case: Case, query: &[f32], key: &[f32], weights: &[f32], bias: &[f32]) -> Vec<i64> {
        run_dtype(case, DataType::Float32, query, key, weights, bias).unwrap()
    }

    /// Independent brute-force reference (plain nested loops, no introselect) so
    /// randomized parity tests cross-check the kernel's partial-select path.
    fn reference(
        case: Case,
        query: &[f32],
        key: &[f32],
        weights: &[f32],
        bias: &[f32],
    ) -> Vec<i64> {
        let ws = case.weights_scale.unwrap_or(1.0);
        let mut out = vec![-1i64; case.batch * case.q_seq * case.top_k];
        for b in 0..case.batch {
            for s in 0..case.q_seq {
                let mut scored: Vec<(f32, usize)> = Vec::new();
                for t in 0..case.key_seq {
                    let bias_bt = bias[(b * case.q_seq + s) * case.key_seq + t];
                    let allowed = bias_bt > MASK_THRESHOLD;
                    if !allowed {
                        continue;
                    }
                    let mut weighted = 0.0f32;
                    for h in 0..case.heads {
                        let qb = ((b * case.q_seq + s) * case.heads + h) * case.head_dim;
                        let kb = (b * case.key_seq + t) * case.head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..case.head_dim {
                            dot += query[qb + d] * key[kb + d];
                        }
                        let relu = (case.scale * dot).max(0.0);
                        weighted += relu * (weights[(b * case.q_seq + s) * case.heads + h] * ws);
                    }
                    scored.push((weighted + bias_bt, t));
                }
                scored.sort_by(|a, b| match b.0.total_cmp(&a.0) {
                    Ordering::Equal => a.1.cmp(&b.1),
                    o => o,
                });
                scored.truncate(case.top_k);
                scored.sort_by_key(|&(_, t)| t);
                for (slot, &(_, t)) in scored.iter().enumerate() {
                    out[(b * case.q_seq + s) * case.top_k + slot] = t as i64;
                }
            }
        }
        out
    }

    /// (B,1,S,T) causal bias: 0 where `t <= past + s`, `-inf` otherwise.
    fn causal_bias(batch: usize, q_seq: usize, key_seq: usize, past: usize) -> Vec<f32> {
        let mut bias = vec![0.0f32; batch * q_seq * key_seq];
        for b in 0..batch {
            for s in 0..q_seq {
                for t in 0..key_seq {
                    if t > past + s {
                        bias[(b * q_seq + s) * key_seq + t] = NEG;
                    }
                }
            }
        }
        bias
    }

    fn base(top_k: usize) -> Case {
        Case {
            batch: 1,
            q_seq: 1,
            heads: 1,
            head_dim: 2,
            key_seq: 4,
            top_k,
            scale: 1.0,
            weights_scale: None,
        }
    }

    // --- Deterministic selection independent of scores -----------------------

    #[test]
    fn prefill_causal_padding_is_ascending_with_sentinels() {
        // top_k >= q_seq ⇒ every allowed position selected, sorted ascending,
        // right-padded with -1. Independent of the score ranking.
        let case = Case {
            q_seq: 3,
            key_seq: 3,
            top_k: 3,
            ..base(3)
        };
        let query = vec![1.0; case.q_seq * case.heads * case.head_dim];
        let key = vec![0.5; case.key_seq * case.head_dim];
        let weights = vec![1.0; case.q_seq * case.heads];
        let bias = causal_bias(1, case.q_seq, case.key_seq, 0);
        let out = run(case, &query, &key, &weights, &bias);
        assert_eq!(out, vec![0, -1, -1, 0, 1, -1, 0, 1, 2]);
    }

    #[test]
    fn sentinel_fill_when_causal_len_below_top_k() {
        // Row allows exactly 2 positions but top_k=4 ⇒ [x, y, -1, -1].
        let case = Case {
            q_seq: 1,
            key_seq: 6,
            top_k: 4,
            ..base(4)
        };
        let query = vec![1.0; case.heads * case.head_dim];
        let key = vec![0.3; case.key_seq * case.head_dim];
        let weights = vec![1.0; case.heads];
        // Only positions 0 and 1 allowed (past=1, s=0).
        let bias = causal_bias(1, 1, case.key_seq, 1);
        let out = run(case, &query, &key, &weights, &bias);
        assert_eq!(out, vec![0, 1, -1, -1]);
    }

    #[test]
    fn all_masked_row_is_all_sentinel() {
        let case = base(3);
        let query = vec![1.0; case.heads * case.head_dim];
        let key = vec![1.0; case.key_seq * case.head_dim];
        let weights = vec![1.0; case.heads];
        let bias = vec![NEG; case.key_seq];
        let out = run(case, &query, &key, &weights, &bias);
        assert_eq!(out, vec![-1, -1, -1]);
    }

    // --- Score-dependent selection -------------------------------------------

    #[test]
    fn selects_top_k_by_score_sorted_ascending() {
        // head_dim=1; key[t] = value ⇒ dot = query*key[t]. scale=1, query=1.
        // scores ∝ key[t]. Pick 2 highest → positions with largest key.
        let case = Case {
            head_dim: 1,
            key_seq: 4,
            top_k: 2,
            ..base(2)
        };
        let query = vec![1.0];
        // key values: t0=0.1, t1=0.9, t2=0.2, t3=0.7 ⇒ top2 = {1,3} → ascending [1,3]
        let key = vec![0.1, 0.9, 0.2, 0.7];
        let weights = vec![1.0];
        let bias = causal_bias(1, 1, case.key_seq, 3); // all allowed
        let out = run(case, &query, &key, &weights, &bias);
        assert_eq!(out, vec![1, 3]);
    }

    #[test]
    fn query_dependent_selection_changes() {
        // Same keys, different queries ⇒ different top-k (the property that makes
        // this query-dependent sparse attention, inexpressible by PagedAttention).
        let case = Case {
            head_dim: 2,
            key_seq: 3,
            top_k: 1,
            ..base(1)
        };
        // keys point along different axes.
        let key = vec![
            1.0, 0.0, /* t0 */ 0.0, 1.0, /* t1 */ 0.7, 0.7, /* t2 */
        ];
        let weights = vec![1.0];
        let bias = causal_bias(1, 1, case.key_seq, 2);
        let q_axis0 = vec![1.0, 0.0];
        let q_axis1 = vec![0.0, 1.0];
        assert_eq!(run(case, &q_axis0, &key, &weights, &bias), vec![0]);
        assert_eq!(run(case, &q_axis1, &key, &weights, &bias), vec![1]);
    }

    #[test]
    fn ties_prefer_lower_index() {
        // Two positions with identical score, top_k=1 ⇒ lower index wins.
        let case = Case {
            head_dim: 1,
            key_seq: 3,
            top_k: 1,
            ..base(1)
        };
        let query = vec![1.0];
        let key = vec![0.5, 0.5, 0.1]; // t0 and t1 tie highest
        let weights = vec![1.0];
        let bias = causal_bias(1, 1, case.key_seq, 2);
        assert_eq!(run(case, &query, &key, &weights, &bias), vec![0]);
    }

    #[test]
    fn relu_clamps_negative_scores() {
        // Negative dot ⇒ relu(0); positive dot ranks above. With all-allowed and
        // top_k covering positives, a negative-dot position is only chosen to fill.
        let case = Case {
            head_dim: 1,
            key_seq: 3,
            top_k: 2,
            ..base(2)
        };
        let query = vec![1.0];
        let key = vec![-5.0, 0.8, 0.2]; // t0 relu->0, t1=0.8, t2=0.2
        let weights = vec![1.0];
        let bias = causal_bias(1, 1, case.key_seq, 2);
        // top2 by score: t1(0.8), t2(0.2) ⇒ ascending [1,2]
        assert_eq!(run(case, &query, &key, &weights, &bias), vec![1, 2]);
    }

    #[test]
    fn weights_scale_and_per_head_weights_affect_ranking() {
        // 2 heads; head0 favors t0, head1 favors t1. Per-head weights tilt choice.
        let case = Case {
            heads: 2,
            head_dim: 2,
            key_seq: 2,
            top_k: 1,
            weights_scale: Some(0.5),
            ..base(1)
        };
        let key = vec![1.0, 0.0, /* t0 */ 0.0, 1.0 /* t1 */];
        // query head0 -> axis0 (t0), head1 -> axis1 (t1)
        let query = vec![1.0, 0.0, /* h0 */ 0.0, 1.0 /* h1 */];
        let bias = causal_bias(1, 1, case.key_seq, 1);
        // weight head0 >> head1 ⇒ pick t0
        assert_eq!(run(case, &query, &key, &[1.0, 0.0], &bias), vec![0]);
        // weight head1 >> head0 ⇒ pick t1
        assert_eq!(run(case, &query, &key, &[0.0, 1.0], &bias), vec![1]);
    }

    // --- Real tiny-GLM indexer dims (fixture: H=2, D=8, top_k=4) --------------

    #[test]
    fn real_glm_indexer_dims_prefill_and_decode() {
        // Exact tiny-glm52-qmoe-indexshare indexer geometry.
        let heads = 2;
        let head_dim = 8;
        let top_k = 4;
        let scale = (head_dim as f32).powf(-0.5);
        let weights_scale = (heads as f32).powf(-0.5);
        let key_seq = 6;

        // Prefill: S == T, per-row causal.
        let prefill = Case {
            batch: 1,
            q_seq: key_seq,
            heads,
            head_dim,
            key_seq,
            top_k,
            scale,
            weights_scale: Some(weights_scale),
        };
        let mut query = vec![0.0f32; prefill.q_seq * heads * head_dim];
        for (i, v) in query.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.11;
        }
        let mut key = vec![0.0f32; key_seq * head_dim];
        for (i, v) in key.iter_mut().enumerate() {
            *v = ((i % 5) as f32 - 2.0) * 0.13;
        }
        let mut weights = vec![0.0f32; prefill.q_seq * heads];
        for (i, v) in weights.iter_mut().enumerate() {
            *v = 0.2 + (i % 3) as f32 * 0.3;
        }
        let bias = causal_bias(1, prefill.q_seq, key_seq, 0);
        let got = run(prefill, &query, &key, &weights, &bias);
        assert_eq!(got, reference(prefill, &query, &key, &weights, &bias));
        // Row 0 allows only position 0 ⇒ [0,-1,-1,-1].
        assert_eq!(&got[0..top_k], &[0, -1, -1, -1]);
        // Every row: valid entries strictly increasing, sentinels trailing.
        assert_valid_rows(&got, prefill.q_seq, top_k, key_seq);

        // Decode: S=1, past grown so all key positions allowed.
        let decode = Case {
            q_seq: 1,
            ..prefill
        };
        let dq: Vec<f32> = query[..heads * head_dim].to_vec();
        let dw: Vec<f32> = weights[..heads].to_vec();
        let dbias = causal_bias(1, 1, key_seq, key_seq - 1);
        let dgot = run(decode, &dq, &key, &dw, &dbias);
        assert_eq!(dgot, reference(decode, &dq, &key, &dw, &dbias));
        assert_eq!(dgot.len(), top_k);
        assert_valid_rows(&dgot, 1, top_k, key_seq);
    }

    fn assert_valid_rows(out: &[i64], rows: usize, top_k: usize, key_seq: usize) {
        for r in 0..rows {
            let row = &out[r * top_k..(r + 1) * top_k];
            let mut seen_sentinel = false;
            let mut prev = -1i64;
            for &v in row {
                if v == -1 {
                    seen_sentinel = true;
                    continue;
                }
                assert!(!seen_sentinel, "valid index after sentinel: {row:?}");
                assert!(v > prev, "not strictly increasing: {row:?}");
                assert!((v as usize) < key_seq, "index out of range: {row:?}");
                prev = v;
            }
        }
    }

    // --- Randomized parity vs independent brute force ------------------------

    #[test]
    fn randomized_parity_against_reference() {
        let mut state = 0x1234_5678u64;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for _ in 0..40 {
            let case = Case {
                batch: 2,
                q_seq: 3,
                heads: 2,
                head_dim: 4,
                key_seq: 5,
                top_k: 3,
                scale: 0.35,
                weights_scale: Some(0.7),
            };
            let query: Vec<f32> = (0..case.batch * case.q_seq * case.heads * case.head_dim)
                .map(|_| rng())
                .collect();
            let key: Vec<f32> = (0..case.batch * case.key_seq * case.head_dim)
                .map(|_| rng())
                .collect();
            let weights: Vec<f32> = (0..case.batch * case.q_seq * case.heads)
                .map(|_| rng().abs())
                .collect();
            let bias = causal_bias(
                case.batch,
                case.q_seq,
                case.key_seq,
                case.key_seq - case.q_seq,
            );
            let got = run(case, &query, &key, &weights, &bias);
            let expect = reference(case, &query, &key, &weights, &bias);
            assert_eq!(got, expect);
        }
    }

    // --- Precision paths ------------------------------------------------------

    #[test]
    fn f16_and_bf16_match_selection() {
        let case = Case {
            head_dim: 2,
            key_seq: 4,
            top_k: 2,
            ..base(2)
        };
        let query = vec![0.5, 0.25];
        let key = vec![0.9, 0.1, 0.2, 0.2, 0.7, 0.6, 0.1, 0.8];
        let weights = vec![1.0];
        let bias = causal_bias(1, 1, case.key_seq, 3);
        let f32_out = run_dtype(case, DataType::Float32, &query, &key, &weights, &bias).unwrap();
        let f16_out = run_dtype(case, DataType::Float16, &query, &key, &weights, &bias).unwrap();
        let bf16_out = run_dtype(case, DataType::BFloat16, &query, &key, &weights, &bias).unwrap();
        assert_eq!(f16_out, f32_out);
        assert_eq!(bf16_out, f32_out);
    }

    // --- Typed rejection tests -----------------------------------------------

    fn err_of(result: Result<Vec<i64>>) -> String {
        match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    fn create_err(graph: &Graph, id: NodeId) -> String {
        match DsaIndexSelectFactory.create(graph.node(id), &[]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected create error, got Ok"),
        }
    }

    #[test]
    fn rejects_unknown_attribute() {
        let case = base(2);
        let (mut graph, id) = node(case, DataType::Float32, DataType::Int64);
        graph
            .node_mut(id)
            .attributes
            .insert("mystery".into(), Attribute::Int(1));
        let err = create_err(&graph, id);
        assert!(err.contains("not part of the frozen v1 ABI"), "{err}");
    }

    #[test]
    fn rejects_missing_top_k() {
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let v = graph.create_named_value("q", DataType::Float32, static_shape([1, 1, 1, 2]));
        graph.add_input(v);
        let out = graph.create_named_value("o", DataType::Int64, static_shape([1, 1, 1, 2]));
        let mut n = Node::new(NodeId(0), OP, vec![Some(v)], vec![out]);
        n.domain = "pkg.nxrt".into();
        n.attributes.insert("scale".into(), Attribute::Float(1.0));
        let id = graph.insert_node(n);
        let err = create_err(&graph, id);
        assert!(err.contains("top_k"), "{err}");
    }

    #[test]
    fn rejects_non_positive_scale() {
        let case = Case {
            scale: 0.0,
            ..base(2)
        };
        let (graph, id) = node(case, DataType::Float32, DataType::Int64);
        let err = create_err(&graph, id);
        assert!(err.contains("scale") && err.contains("> 0"), "{err}");
    }

    #[test]
    fn rejects_wrong_output_dtype() {
        let case = base(2);
        let query = vec![1.0; case.heads * case.head_dim];
        let key = vec![0.5; case.key_seq * case.head_dim];
        let weights = vec![1.0; case.heads];
        let bias = causal_bias(1, 1, case.key_seq, 3);
        let (graph, id) = node(case, DataType::Float32, DataType::Int32);
        let kernel = DsaIndexSelectFactory.create(graph.node(id), &[]).unwrap();
        let q = Owned::f32(&[case.batch, case.q_seq, case.heads, case.head_dim], &query);
        let k = Owned::f32(&[case.batch, case.key_seq, case.head_dim], &key);
        let w = Owned::f32(&[case.batch, case.q_seq, case.heads], &weights);
        let b = Owned::f32(&[case.batch, 1, case.q_seq, case.key_seq], &bias);
        let mut out = Owned::zeros(DataType::Int32, &[case.batch, 1, case.q_seq, case.top_k]);
        let err = err_of(
            kernel
                .execute(
                    &[q.view(), k.view(), w.view(), b.view()],
                    &mut [out.view_mut()],
                )
                .map(|()| Vec::new()),
        );
        assert!(err.contains("Int64"), "{err}");
    }

    #[test]
    fn rejects_dtype_mismatch_between_query_and_key() {
        let case = base(2);
        let query = vec![1.0; case.heads * case.head_dim];
        let key = vec![0.5; case.key_seq * case.head_dim];
        let weights = vec![1.0; case.heads];
        let bias = causal_bias(1, 1, case.key_seq, 3);
        let (graph, id) = node(case, DataType::Float32, DataType::Int64);
        let kernel = DsaIndexSelectFactory.create(graph.node(id), &[]).unwrap();
        let q = Owned::f32(&[case.batch, case.q_seq, case.heads, case.head_dim], &query);
        let k = Owned::f16(&[case.batch, case.key_seq, case.head_dim], &key); // mismatched
        let w = Owned::f32(&[case.batch, case.q_seq, case.heads], &weights);
        let b = Owned::f32(&[case.batch, 1, case.q_seq, case.key_seq], &bias);
        let mut out = Owned::zeros(DataType::Int64, &[case.batch, 1, case.q_seq, case.top_k]);
        let err = err_of(
            kernel
                .execute(
                    &[q.view(), k.view(), w.view(), b.view()],
                    &mut [out.view_mut()],
                )
                .map(|()| Vec::new()),
        );
        assert!(err.contains("must match query dtype"), "{err}");
    }

    #[test]
    fn rejects_bias_head_dim_not_one() {
        let case = base(2);
        let query = vec![1.0; case.heads * case.head_dim];
        let key = vec![0.5; case.key_seq * case.head_dim];
        let weights = vec![1.0; case.heads];
        let (graph, id) = node(case, DataType::Float32, DataType::Int64);
        let kernel = DsaIndexSelectFactory.create(graph.node(id), &[]).unwrap();
        let q = Owned::f32(&[case.batch, case.q_seq, case.heads, case.head_dim], &query);
        let k = Owned::f32(&[case.batch, case.key_seq, case.head_dim], &key);
        let w = Owned::f32(&[case.batch, case.q_seq, case.heads], &weights);
        // bias with head dim = 2 (must be 1)
        let bias = vec![0.0f32; case.batch * 2 * case.q_seq * case.key_seq];
        let b = Owned::f32(&[case.batch, 2, case.q_seq, case.key_seq], &bias);
        let mut out = Owned::zeros(DataType::Int64, &[case.batch, 1, case.q_seq, case.top_k]);
        let err = err_of(
            kernel
                .execute(
                    &[q.view(), k.view(), w.view(), b.view()],
                    &mut [out.view_mut()],
                )
                .map(|()| Vec::new()),
        );
        assert!(err.contains("dim 1 must be 1"), "{err}");
    }

    #[test]
    fn claim_gate_accepts_valid_and_rejects_bad_rank() {
        let case = base(4);
        let (graph, id) = node(case, DataType::Float32, DataType::Int64);
        let n = graph.node(id);
        let shapes: Vec<Shape> = vec![
            static_shape([1usize, 1, 1, 2]),
            static_shape([1usize, 4, 2]),
            static_shape([1usize, 1, 1]),
            static_shape([1usize, 1, 1, 4]),
        ];
        let dtypes = vec![DataType::Float32; 4];
        assert!(unsupported_reason(n, &shapes, &dtypes).is_none());
        // key given rank 4 instead of 3.
        let bad = vec![
            static_shape([1usize, 1, 1, 2]),
            static_shape([1usize, 4, 2, 1]),
            static_shape([1usize, 1, 1]),
            static_shape([1usize, 1, 1, 4]),
        ];
        let reason = unsupported_reason(n, &bad, &dtypes).unwrap();
        assert!(reason.contains("rank"), "{reason}");
    }
}
