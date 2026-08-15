//! Opt-in fp32→fp16 decoder precision rewrite (see [`DecodePrecision::Fp16`]).
//!
//! Selecting [`DecodePrecision::Fp16`] on a [`SessionBuilder`](crate::SessionBuilder)
//! casts an fp32-activation int4/block-32 decoder graph (e.g. the Foundry
//! `generic-cpu` Phi-3.5-mini export, whose `MatMulNBits` carry fp32 scales and
//! `accuracy_level=4`) to a fully fp16 graph. On a GPU backend that lands the
//! decoder on the fast fp16-fused decode kernels (half2 `MatMulNBits` GEMV,
//! fused gate/up SwiGLU, skip-RMSNorm, fp16 GQA) instead of the slower
//! fp32-activation path.
//!
//! ## Why here, in the session builder
//!
//! This runs on the freshly-loaded [`Graph`] *before* the session's I/O
//! signature ([`crate::IoMeta`]) is computed and before EP optimization. That
//! single early seam is required for consistency: the KV-cache bridge and logits
//! buffers are sized from the graph's declared input/output dtypes, while the
//! executor runs the (EP-fused) graph. Converting later — e.g. inside an EP pass
//! — would leave the KV/logits buffers fp32 while the executor graph is fp16,
//! and the forward pass would reject the mismatched KV tensors.
//!
//! ## Scope and safety
//!
//! The rewrite is a decoder-wide precision change, not a model-specific hack: it
//! down-converts *every* Float32 tensor (activations, residual, `MatMulNBits`
//! scales, norm gammas, GQA, KV cache, logits). It is gated three ways so the
//! default behaviour and every native fp16 model stay bit-identical:
//!
//! 1. no-op unless the caller selects [`DecodePrecision::Fp16`];
//! 2. no-op unless the session targets a GPU device;
//! 3. no-op unless the graph actually has an fp32-scale `MatMulNBits` (the
//!    fp32-activation quantized fingerprint). Native fp16 models carry fp16
//!    scales and never match.
//!
//! ## The rewrite
//!
//! Every `Float32` tensor becomes `Float16`: value type declarations (including
//! graph inputs/outputs — the KV cache and logits become fp16, exactly as a
//! native fp16 model already declares them), float initializers (embedding
//! table, `MatMulNBits` scales, norm gammas) have their bytes down-converted and
//! re-inlined, `Constant` float tensors are down-converted, and any
//! `Cast(to=float32)` is retargeted to fp16. Control-flow subgraphs (the
//! LongRoPE `If` cos/sin branches) are converted recursively. Non-float tensors
//! (int64 ids/mask, uint8 quant weights, int32 GQA seqlens) are untouched. The
//! rewrite only ever touches `Float32` data, so it is idempotent.

use onnx_runtime_ir::{Attribute, DataType, Graph, NodeId, TensorData, ValueId, WeightRef};
use onnx_runtime_loader::WeightStore;

use crate::DecodePrecision;

/// Apply the fp32→fp16 decoder rewrite to `graph` in place when the caller
/// selected [`DecodePrecision::Fp16`], the session targets a GPU, and the graph
/// is an fp32-activation quantized decoder. Returns `true` when the graph was
/// rewritten. Non-fatal on a missing initializer resolution (leaves the graph as
/// loaded and returns `false`).
pub(crate) fn maybe_convert_decode_fp16(
    graph: &mut Graph,
    weights: &WeightStore,
    precision: DecodePrecision,
    device_is_gpu: bool,
) -> bool {
    if precision != DecodePrecision::Fp16 || !device_is_gpu {
        return false;
    }
    if !graph_has_fp32_matmul_nbits(graph) {
        return false;
    }
    convert_graph_fp32_to_fp16(graph, weights);
    true
}

/// True when the graph has a `MatMulNBits` whose scales input (index 2) is a
/// `Float32` initializer — the fingerprint of an fp32-activation quantized
/// decoder. Native fp16 models carry fp16 scales and never match.
fn graph_has_fp32_matmul_nbits(graph: &Graph) -> bool {
    graph.nodes.iter().any(|(_, node)| {
        if node.op_type != "MatMulNBits" {
            return false;
        }
        let Some(&Some(scales)) = node.inputs.get(2) else {
            return false;
        };
        graph
            .initializers
            .get(&scales)
            .is_some_and(|weight| weight.dtype() == DataType::Float32)
    })
}

/// Down-convert every `Float32` tensor in `graph` (and its subgraphs) to
/// `Float16`, in place, resolving external initializer bytes through `weights`.
fn convert_graph_fp32_to_fp16(graph: &mut Graph, weights: &WeightStore) {
    // 1. Float initializers: resolve backing bytes, down-convert, re-inline.
    let float_inits: Vec<(ValueId, WeightRef)> = graph
        .initializers
        .iter()
        .filter(|(_, weight)| weight.dtype() == DataType::Float32)
        .map(|(id, weight)| (*id, weight.clone()))
        .collect();
    for (id, weight) in float_inits {
        let Some(bytes) = weights.bytes(&weight) else {
            // Should not happen for a well-formed model; leave this initializer
            // as fp32 and let downstream validation surface any inconsistency.
            continue;
        };
        let fp16 = f32_bytes_to_f16(bytes);
        let dims = weight.dims().to_vec();
        graph.set_initializer(
            id,
            WeightRef::Inline(TensorData::from_raw(DataType::Float16, dims, fp16)),
        );
    }

    // 2. Blanket-retype every remaining Float32 value declaration to Float16.
    retype_float_values(graph);

    // 3. Node-level rewrites: Constant float tensors, Cast(to=float32) targets,
    //    and recursion into inline subgraph attributes.
    let node_ids: Vec<NodeId> = graph.nodes.iter().map(|(id, _)| id).collect();
    for node_id in node_ids {
        rewrite_node_attributes_fp32_to_fp16(graph.node_mut(node_id));
    }

    // 4. Recurse into the indexed control-flow subgraphs (If cos/sin branches),
    //    so the later on-device select lowering materializes fp16 caches. These
    //    branches are pure `Constant` selections with no external initializers.
    let subgraph_keys: Vec<(NodeId, String)> = graph.subgraphs.keys().cloned().collect();
    for key in subgraph_keys {
        if let Some(mut sub) = graph.subgraphs.remove(&key) {
            retype_float_values(&mut sub);
            let sub_nodes: Vec<NodeId> = sub.nodes.iter().map(|(id, _)| id).collect();
            for node_id in sub_nodes {
                rewrite_node_attributes_fp32_to_fp16(sub.node_mut(node_id));
            }
            graph.subgraphs.insert(key, sub);
        }
    }
}

/// Retype every `Float32` value declaration in `graph` to `Float16`.
fn retype_float_values(graph: &mut Graph) {
    let float_values: Vec<ValueId> = graph
        .values
        .iter()
        .filter(|(_, value)| value.dtype == DataType::Float32)
        .map(|(id, _)| id)
        .collect();
    for id in float_values {
        if let Some(value) = graph.values.get_mut(id) {
            value.dtype = DataType::Float16;
        }
    }
}

/// Convert a node's inline attributes: `Constant`/tensor floats to fp16, a
/// `Cast(to=float32)` retargeted to fp16, and any nested subgraph attribute.
fn rewrite_node_attributes_fp32_to_fp16(node: &mut onnx_runtime_ir::Node) {
    let is_cast = node.op_type == "Cast" && node.is_default_domain();
    for (name, attr) in node.attributes.iter_mut() {
        match attr {
            Attribute::Int(v)
                if is_cast && name == "to" && *v == DataType::Float32.to_onnx() as i64 =>
            {
                *v = DataType::Float16.to_onnx() as i64;
            }
            Attribute::Tensor(tensor) => convert_tensor_fp32_to_fp16(tensor),
            Attribute::Tensors(tensors) => {
                for tensor in tensors.iter_mut() {
                    convert_tensor_fp32_to_fp16(tensor);
                }
            }
            Attribute::Graph(sub) => convert_inline_subgraph_fp32_to_fp16(sub),
            Attribute::Graphs(subs) => {
                for sub in subs.iter_mut() {
                    convert_inline_subgraph_fp32_to_fp16(sub);
                }
            }
            _ => {}
        }
    }
}

/// Convert the inline `Attribute::Graph` copy of a subgraph (kept consistent
/// with the indexed `graph.subgraphs` copy; the rewrite only touches `Float32`
/// data and is therefore idempotent).
fn convert_inline_subgraph_fp32_to_fp16(sub: &mut Graph) {
    retype_float_values(sub);
    let node_ids: Vec<NodeId> = sub.nodes.iter().map(|(id, _)| id).collect();
    for node_id in node_ids {
        rewrite_node_attributes_fp32_to_fp16(sub.node_mut(node_id));
    }
}

/// Down-convert an inline `TensorData` in place if it holds `Float32` bytes.
fn convert_tensor_fp32_to_fp16(tensor: &mut TensorData) {
    if tensor.dtype != DataType::Float32 {
        return;
    }
    tensor.data = f32_bytes_to_f16(&tensor.data);
    tensor.dtype = DataType::Float16;
}

/// Convert a little-endian `f32` byte buffer to the equivalent little-endian
/// `f16` buffer (round-to-nearest-even via the `half` crate).
fn f32_bytes_to_f16(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let half = half::f16::from_f32(value);
        out.extend_from_slice(&half.to_le_bytes());
    }
    out
}
