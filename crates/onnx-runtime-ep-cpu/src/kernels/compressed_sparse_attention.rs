//! Phase-1 f32 reference skeleton for
//! `pkg.nxrt::CompressedSparseAttention` v1.
//!
//! The registered operator exposes the complete frozen stateful v1 boundary.
//! Stateful compressor/carry updates remain an explicit Unsupported path until
//! their equations are wired into the runtime. The Phase-1 assembled-cache
//! gather/attention implementation remains a tested, unregistered reference.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::sparse_kv_gather::{
    checked_layout, checked_product, fallible_filled, read_dense_f32, read_dense_indices,
    sparse_kv_gather_masked_f32,
};
use super::{check_arity, write_dense_f32};

const OP: &str = "CompressedSparseAttention";
const LAYOUT_VERSION: i64 = 1;
const FROZEN_V1_REQUIRED_INPUTS: usize = 11;
const FROZEN_V1_MAX_INPUTS: usize = 20;
const FROZEN_V1_REQUIRED_OUTPUTS: usize = 3;
const FROZEN_V1_MAX_OUTPUTS: usize = 6;
const FROZEN_V1_REQUIRED_INPUT_NAMES: [&str; FROZEN_V1_REQUIRED_INPUTS] = [
    "query",
    "current_kv",
    "compressor_kv",
    "compressor_gate",
    "compressor_ape",
    "compressor_norm",
    "past_compressed_kv",
    "past_compression_carry",
    "seqlens_k",
    "total_sequence_length",
    "head_sink",
];

pub struct CompressedSparseAttentionFactory;

struct DeferredCompressedSparseAttentionKernel {
    compression_ratio: usize,
}

struct CompressedSparseAttentionKernel {
    num_heads: usize,
    head_dim: usize,
    compression_ratio: usize,
    index_num_heads: usize,
    scale: f32,
}

impl KernelFactory for CompressedSparseAttentionFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        validate_frozen_v1_schema(node)?;
        self.create_impl(node, input_shapes, false)
    }
}

impl CompressedSparseAttentionFactory {
    fn create_impl(
        &self,
        node: &Node,
        input_shapes: &[Vec<usize>],
        phase1_reference: bool,
    ) -> Result<Box<dyn Kernel>> {
        let num_heads = required_positive_int(node, "num_heads")?;
        let head_dim = required_positive_int(node, "head_dim")?;
        let compression_ratio = required_positive_int(node, "compression_ratio")?;
        if !matches!(compression_ratio, 4 | 128) {
            return Err(error(format!(
                "compression_ratio must be exactly 4 or 128, got {compression_ratio}"
            )));
        }
        let index_num_heads = optional_nonnegative_int(node, "index_num_heads", 0)?;
        let index_head_dim = optional_nonnegative_int(node, "index_head_dim", 0)?;
        let index_topk = optional_nonnegative_int(node, "index_topk", 0)?;
        if compression_ratio == 4
            && (index_num_heads == 0 || index_head_dim == 0 || index_topk == 0)
        {
            return Err(error(
                "ratio-4 requires positive index_num_heads, index_head_dim, and index_topk",
            ));
        }
        if compression_ratio == 128
            && (index_num_heads != 0 || index_head_dim != 0 || index_topk != 0)
        {
            return Err(error(
                "ratio-128 requires index_num_heads=index_head_dim=index_topk=0",
            ));
        }
        require_int_attr(node, "causal", 1)?;
        require_int_attr(node, "cache_layout_version", LAYOUT_VERSION)?;
        require_int_attr(node, "index_layout_version", LAYOUT_VERSION)?;
        let sink_mode = node
            .attr("sink_mode")
            .map(|attribute| {
                attribute
                    .as_str()
                    .ok_or_else(|| error("attribute sink_mode must be a UTF-8 string"))
            })
            .transpose()?
            .unwrap_or("logit_only");
        if sink_mode != "logit_only" {
            return Err(unsupported(format!(
                "sink_mode='{sink_mode}' is unsupported; v1 requires 'logit_only'"
            )));
        }
        let cache_format = node
            .attr("cache_format")
            .map(|attribute| {
                attribute
                    .as_str()
                    .ok_or_else(|| error("attribute cache_format must be a UTF-8 string"))
            })
            .transpose()?
            .unwrap_or("f32");
        if cache_format != "f32" {
            return Err(unsupported(format!(
                "cache_format='{cache_format}' requires Phase 2 FP4/FP8 compressed-KV dequantization"
            )));
        }
        let scale = node
            .attr("scale")
            .and_then(|attribute| attribute.as_float())
            .unwrap_or(0.0);
        if !scale.is_finite() || scale < 0.0 {
            return Err(error("scale must be finite and non-negative"));
        }

        if phase1_reference && input_shapes.len() >= 4 {
            infer_output_shape(
                &input_shapes[0],
                &input_shapes[1],
                &input_shapes[2],
                &input_shapes[3],
                num_heads,
                head_dim,
            )?;
        }
        if phase1_reference {
            Ok(Box::new(CompressedSparseAttentionKernel {
                num_heads,
                head_dim,
                compression_ratio,
                index_num_heads,
                scale,
            }))
        } else {
            Ok(Box::new(DeferredCompressedSparseAttentionKernel {
                compression_ratio,
            }))
        }
    }

    #[cfg(test)]
    fn create_phase1_reference(
        &self,
        node: &Node,
        input_shapes: &[Vec<usize>],
    ) -> Result<Box<dyn Kernel>> {
        self.create_impl(node, input_shapes, true)
    }
}

impl Kernel for DeferredCompressedSparseAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        validate_frozen_v1_runtime_arity(inputs, outputs)?;
        for (index, name) in FROZEN_V1_REQUIRED_INPUT_NAMES.iter().enumerate() {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required frozen-v1 input {index} ('{name}') is absent"
                )));
            }
        }

        let deferred = if self.compression_ratio == 4 {
            "stateful compressed-KV construction, compression-carry updates, index-key construction/carry updates, and top-k index selection are deferred in Phase 1"
        } else {
            "stateful compressed-KV construction and compression-carry updates are deferred in Phase 1"
        };
        Err(unsupported(deferred))
    }
}

fn validate_frozen_v1_schema(node: &Node) -> Result<()> {
    if !(FROZEN_V1_REQUIRED_INPUTS..=FROZEN_V1_MAX_INPUTS).contains(&node.inputs.len()) {
        return Err(error(format!(
            "frozen v1 requires {FROZEN_V1_REQUIRED_INPUTS}..={FROZEN_V1_MAX_INPUTS} positional inputs, got {}",
            node.inputs.len()
        )));
    }
    if !(FROZEN_V1_REQUIRED_OUTPUTS..=FROZEN_V1_MAX_OUTPUTS).contains(&node.outputs.len()) {
        return Err(error(format!(
            "frozen v1 requires {FROZEN_V1_REQUIRED_OUTPUTS}..={FROZEN_V1_MAX_OUTPUTS} outputs, got {}",
            node.outputs.len()
        )));
    }
    for (index, name) in FROZEN_V1_REQUIRED_INPUT_NAMES.iter().enumerate() {
        if node.inputs[index].is_none() {
            return Err(error(format!(
                "required frozen-v1 input {index} ('{name}') is omitted"
            )));
        }
    }
    Ok(())
}

fn validate_frozen_v1_runtime_arity(inputs: &[TensorView], outputs: &[TensorMut]) -> Result<()> {
    if !(FROZEN_V1_REQUIRED_INPUTS..=FROZEN_V1_MAX_INPUTS).contains(&inputs.len()) {
        return Err(error(format!(
            "frozen v1 requires {FROZEN_V1_REQUIRED_INPUTS}..={FROZEN_V1_MAX_INPUTS} positional inputs, got {}",
            inputs.len()
        )));
    }
    if !(FROZEN_V1_REQUIRED_OUTPUTS..=FROZEN_V1_MAX_OUTPUTS).contains(&outputs.len()) {
        return Err(error(format!(
            "frozen v1 requires {FROZEN_V1_REQUIRED_OUTPUTS}..={FROZEN_V1_MAX_OUTPUTS} outputs, got {}",
            outputs.len()
        )));
    }
    Ok(())
}

impl Kernel for CompressedSparseAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 4, 6, 1)?;
        for (name, input) in [
            ("query", &inputs[0]),
            ("cache", &inputs[1]),
            ("head_sink", &inputs[3]),
        ] {
            require_dtype(name, input.dtype, DataType::Float32)?;
        }
        if !matches!(inputs[2].dtype, DataType::Int32 | DataType::Int64) {
            return Err(error(format!(
                "indices must have dtype Int32 or Int64, got {:?}",
                inputs[2].dtype
            )));
        }
        require_dtype("Y", outputs[0].dtype, DataType::Float32)?;

        let expected_output = infer_output_shape(
            inputs[0].shape,
            inputs[1].shape,
            inputs[2].shape,
            inputs[3].shape,
            self.num_heads,
            self.head_dim,
        )?;
        if outputs[0].shape != expected_output {
            return Err(error(format!(
                "Y must have shape {expected_output:?}, got {:?}",
                outputs[0].shape
            )));
        }
        let query_shape = shape4("query", inputs[0].shape)?;
        let cache_shape = shape4("cache", inputs[1].shape)?;
        let indices_shape = shape4("indices", inputs[2].shape)?;
        let [batch, sequence, heads, dim] = query_shape;
        let groups = cache_shape[1];
        if groups != 1 && groups != heads {
            return Err(unsupported(format!(
                "cache/index groups must be 1 or num_heads ({heads}) in the Phase-1 skeleton, got {groups}"
            )));
        }
        if self.compression_ratio == 4 && groups != 1 && groups != self.index_num_heads {
            return Err(error(format!(
                "ratio-4 grouped indices must use 1 or index_num_heads={} groups, got {groups}",
                self.index_num_heads
            )));
        }
        debug_assert_eq!(dim, self.head_dim);
        debug_assert_eq!(heads, self.num_heads);
        let valid_lengths = inputs
            .get(4)
            .filter(|input| !input.is_absent())
            .map(|input| read_valid_lengths(input, batch, cache_shape[2]))
            .transpose()?;
        let attention_bias = inputs
            .get(5)
            .filter(|input| !input.is_absent())
            .map(|input| AttentionBias::new(input, [batch, heads, sequence, indices_shape[3]]))
            .transpose()?;

        let query = read_dense_f32(&inputs[0], "query")?;
        let cache = read_dense_f32(&inputs[1], "cache")?;
        let indices = read_dense_indices(&inputs[2], "indices")?;
        let sink = read_dense_f32(&inputs[3], "head_sink")?;
        let gathered = sparse_kv_gather_masked_f32(
            &cache,
            cache_shape,
            &indices,
            indices_shape,
            valid_lengths.as_deref(),
        )?;

        let selections = indices_shape[3];
        let output_elements = checked_layout(
            &[batch, sequence, heads, dim],
            std::mem::size_of::<f32>(),
            "Y",
        )?;
        let score_elements =
            checked_product(&[batch, heads, sequence, selections], "score element count")?;
        checked_layout(
            &[batch, heads, sequence, selections],
            std::mem::size_of::<f32>(),
            "scores",
        )?;
        let mut scores = fallible_filled(score_elements, f32::NEG_INFINITY, "attention scores")?;
        let mut output = fallible_filled(output_elements, 0.0f32, "attention output")?;
        let scale = if self.scale == 0.0 {
            1.0 / (dim as f32).sqrt()
        } else {
            self.scale
        };

        for b in 0..batch {
            for h in 0..heads {
                let group = if groups == 1 { 0 } else { h };
                for s in 0..sequence {
                    let score_row = flat4(
                        [b, h, s, 0],
                        [batch, heads, sequence, selections],
                        "score row",
                    )?;
                    let gathered_row = flat4(
                        [b, group, s, 0],
                        [batch, groups, sequence, selections],
                        "gathered row",
                    )?;
                    let query_row =
                        flat4([b, s, h, 0], [batch, sequence, heads, dim], "query row")?;
                    let mut maximum = f32::NEG_INFINITY;
                    for k in 0..selections {
                        let record = gathered_row
                            .checked_add(k)
                            .ok_or_else(|| error("gathered validity offset overflow"))?;
                        if !gathered.valid[record] {
                            continue;
                        }
                        let kv_row = record
                            .checked_mul(dim)
                            .ok_or_else(|| error("gathered KV offset overflow"))?;
                        let mut score = 0.0f32;
                        for d in 0..dim {
                            score += query[query_row + d] * gathered.values[kv_row + d];
                        }
                        score *= scale;
                        if let Some(bias) = &attention_bias {
                            score += bias.at(b, h, s, k)?;
                        }
                        scores[score_row + k] = score;
                        maximum = maximum.max(score);
                    }
                    if maximum == f32::NEG_INFINITY {
                        continue;
                    }

                    let mut denominator = 0.0f32;
                    for k in 0..selections {
                        let score = scores[score_row + k];
                        if score != f32::NEG_INFINITY {
                            denominator += (score - maximum).exp();
                        }
                    }
                    denominator += (sink[h] - maximum).exp();
                    if denominator == 0.0 || denominator.is_nan() {
                        return Err(error(format!(
                            "softmax denominator is invalid at [batch={b}, head={h}, query={s}]"
                        )));
                    }
                    let output_row =
                        flat4([b, s, h, 0], [batch, sequence, heads, dim], "output row")?;
                    for k in 0..selections {
                        let score = scores[score_row + k];
                        if score == f32::NEG_INFINITY {
                            continue;
                        }
                        let record = gathered_row
                            .checked_add(k)
                            .ok_or_else(|| error("gathered record offset overflow"))?;
                        let kv_row = record
                            .checked_mul(dim)
                            .ok_or_else(|| error("gathered KV offset overflow"))?;
                        let probability = (score - maximum).exp() / denominator;
                        for d in 0..dim {
                            output[output_row + d] += probability * gathered.values[kv_row + d];
                        }
                    }
                }
            }
        }
        write_dense_f32(&mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Infer `Y=[B,S,N,D]` while validating the Phase-1 assembled-cache boundary.
pub fn infer_output_shape(
    query: &[usize],
    cache: &[usize],
    indices: &[usize],
    head_sink: &[usize],
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<usize>> {
    let query = shape4("query", query)?;
    let cache = shape4("cache", cache)?;
    let indices = shape4("indices", indices)?;
    if query[2] != num_heads || query[3] != head_dim {
        return Err(error(format!(
            "query must end in [num_heads={num_heads}, head_dim={head_dim}], got {:?}",
            &query[2..]
        )));
    }
    if cache[0] != query[0] || cache[3] != head_dim {
        return Err(error(format!(
            "cache must have batch {} and head_dim {head_dim}, got {cache:?}",
            query[0]
        )));
    }
    if indices[0] != query[0] || indices[1] != cache[1] || indices[2] != query[1] {
        return Err(error(format!(
            "indices must have [B,G,S]=[{},{},{}], got {:?}",
            query[0],
            cache[1],
            query[1],
            &indices[..3]
        )));
    }
    if head_sink != [num_heads] {
        return Err(error(format!(
            "head_sink must have shape [{num_heads}], got {head_sink:?}"
        )));
    }
    let output = query.to_vec();
    checked_layout(&output, std::mem::size_of::<f32>(), "Y")?;
    Ok(output)
}

struct AttentionBias {
    data: Vec<f32>,
    shape: Vec<usize>,
    padded_shape: [usize; 4],
    target: [usize; 4],
}

impl AttentionBias {
    fn new(view: &TensorView, target: [usize; 4]) -> Result<Self> {
        if view.dtype == DataType::Bool {
            return Err(unsupported(
                "boolean attention_bias semantics are deferred; use additive f32 bias",
            ));
        }
        require_dtype("attention_bias", view.dtype, DataType::Float32)?;
        if view.shape.len() > 4 {
            return Err(error(format!(
                "attention_bias rank must be <= 4, got {:?}",
                view.shape
            )));
        }
        checked_layout(view.shape, std::mem::size_of::<f32>(), "attention_bias")?;
        let mut padded_shape = [1usize; 4];
        padded_shape[4 - view.shape.len()..].copy_from_slice(view.shape);
        for axis in 0..4 {
            if padded_shape[axis] != 1 && padded_shape[axis] != target[axis] {
                return Err(error(format!(
                    "attention_bias shape {:?} is not broadcastable to {target:?}",
                    view.shape
                )));
            }
        }
        Ok(Self {
            data: read_dense_f32(view, "attention_bias")?,
            shape: view.shape.to_vec(),
            padded_shape,
            target,
        })
    }

    fn at(&self, b: usize, h: usize, s: usize, k: usize) -> Result<f32> {
        let target_index = [b, h, s, k];
        let mut offset = 0usize;
        for axis in 0..4 {
            let coordinate = if self.padded_shape[axis] == 1 {
                0
            } else {
                target_index[axis]
            };
            offset = offset
                .checked_mul(self.padded_shape[axis])
                .and_then(|value| value.checked_add(coordinate))
                .ok_or_else(|| error("attention_bias offset overflow"))?;
        }
        self.data.get(offset).copied().ok_or_else(|| {
            error(format!(
                "attention_bias offset {offset} exceeds shape {:?} for target {:?}",
                self.shape, self.target
            ))
        })
    }
}

fn read_valid_lengths(view: &TensorView, batch: usize, cache_len: usize) -> Result<Vec<usize>> {
    if view.shape != [batch] {
        return Err(error(format!(
            "valid_lengths must have shape [{batch}], got {:?}",
            view.shape
        )));
    }
    let values = read_dense_indices(view, "valid_lengths")?;
    values
        .into_iter()
        .enumerate()
        .map(|(b, value)| {
            let value = usize::try_from(value)
                .map_err(|_| error(format!("valid_lengths[{b}] must be non-negative")))?;
            if value > cache_len {
                return Err(error(format!(
                    "valid_lengths[{b}]={value} exceeds cache length {cache_len}"
                )));
            }
            Ok(value)
        })
        .collect()
}

fn flat4(index: [usize; 4], shape: [usize; 4], what: &str) -> Result<usize> {
    index[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_add(index[1]))
        .and_then(|value| value.checked_mul(shape[2]))
        .and_then(|value| value.checked_add(index[2]))
        .and_then(|value| value.checked_mul(shape[3]))
        .and_then(|value| value.checked_add(index[3]))
        .ok_or_else(|| error(format!("{what} offset overflow")))
}

fn shape4(name: &str, shape: &[usize]) -> Result<[usize; 4]> {
    shape
        .try_into()
        .map_err(|_| error(format!("{name} must be rank 4, got shape {shape:?}")))
}

fn required_positive_int(node: &Node, name: &str) -> Result<usize> {
    let value = node
        .attr(name)
        .and_then(|attribute| attribute.as_int())
        .ok_or_else(|| error(format!("missing required integer attribute {name}")))?;
    usize::try_from(value)
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| error(format!("{name} must be positive, got {value}")))
}

fn optional_nonnegative_int(node: &Node, name: &str, default: i64) -> Result<usize> {
    let value = node
        .attr(name)
        .map(|attribute| {
            attribute
                .as_int()
                .ok_or_else(|| error(format!("attribute {name} must be an integer")))
        })
        .transpose()?
        .unwrap_or(default);
    usize::try_from(value).map_err(|_| error(format!("{name} must be non-negative, got {value}")))
}

fn require_int_attr(node: &Node, name: &str, expected: i64) -> Result<()> {
    let value = node
        .attr(name)
        .map(|attribute| {
            attribute
                .as_int()
                .ok_or_else(|| error(format!("attribute {name} must be an integer")))
        })
        .transpose()?
        .unwrap_or(expected);
    if value != expected {
        return Err(error(format!("{name} must be {expected}, got {value}")));
    }
    Ok(())
}

fn require_dtype(name: &str, got: DataType, expected: DataType) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have dtype {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn unsupported(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OP}: Unsupported: {}", message.into()))
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OP}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};

    fn kernel(
        ratio: i64,
        cache_format: Option<&str>,
        shapes: &[Vec<usize>],
    ) -> Result<Box<dyn Kernel>> {
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let input_specs = [
            ("query", DataType::Float32),
            ("cache", DataType::Float32),
            ("indices", DataType::Int32),
            ("head_sink", DataType::Float32),
        ];
        let inputs = input_specs
            .iter()
            .zip(shapes)
            .map(|((name, dtype), shape)| {
                Some(graph.create_named_value(*name, *dtype, static_shape(shape.iter().copied())))
            })
            .collect();
        let output = graph.create_named_value(
            "Y",
            DataType::Float32,
            static_shape(shapes[0].iter().copied()),
        );
        let mut node = onnx_runtime_ir::Node::new(NodeId(0), OP, inputs, vec![output]);
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(2));
        node.attributes.insert("head_dim".into(), Attribute::Int(2));
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(ratio));
        if ratio == 4 {
            node.attributes
                .insert("index_num_heads".into(), Attribute::Int(1));
            node.attributes
                .insert("index_head_dim".into(), Attribute::Int(128));
            node.attributes
                .insert("index_topk".into(), Attribute::Int(512));
        }
        if let Some(format) = cache_format {
            node.attributes
                .insert("cache_format".into(), Attribute::String(format.into()));
        }
        CompressedSparseAttentionFactory.create_phase1_reference(&node, shapes)
    }

    #[test]
    fn gathered_dense_fallback_matches_scalar_sink_oracle() {
        let shapes = vec![
            vec![1, 2, 2, 2],
            vec![1, 1, 3, 2],
            vec![1, 1, 2, 3],
            vec![2],
        ];
        let kernel = kernel(128, None, &shapes).unwrap();
        let query = Owned::f32(&shapes[0], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0]);
        let cache = Owned::f32(&shapes[1], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let indices = Owned::i32(&shapes[2], &[0, 1, -1, 2, 0, 1]);
        let sink = Owned::f32(&shapes[3], &[0.25, -0.5]);
        let mut output = Owned::zeros_f32(&shapes[0]);
        kernel
            .execute(
                &[query.view(), cache.view(), indices.view(), sink.view()],
                &mut [output.view_mut()],
            )
            .unwrap();

        let q = query.to_f32();
        let kv = cache.to_f32();
        let idx = indices.to_i32();
        let sinks = sink.to_f32();
        let scale = 1.0 / 2.0f32.sqrt();
        let mut expected = vec![0.0f32; 8];
        for s in 0..2 {
            for h in 0..2 {
                let mut scores = Vec::new();
                for k in 0..3 {
                    let selected = idx[s * 3 + k];
                    if selected < 0 {
                        continue;
                    }
                    let selected = selected as usize;
                    let dot = q[(s * 2 + h) * 2] * kv[selected * 2]
                        + q[(s * 2 + h) * 2 + 1] * kv[selected * 2 + 1];
                    scores.push((selected, dot * scale));
                }
                let maximum = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores
                    .iter()
                    .map(|(_, score)| (*score - maximum).exp())
                    .sum::<f32>()
                    + (sinks[h] - maximum).exp();
                for (selected, score) in scores {
                    let probability = (score - maximum).exp() / denominator;
                    expected[(s * 2 + h) * 2] += probability * kv[selected * 2];
                    expected[(s * 2 + h) * 2 + 1] += probability * kv[selected * 2 + 1];
                }
            }
        }
        for (actual, expected) in output.to_f32().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn quantized_cache_format_is_explicitly_unsupported() {
        let shapes = vec![
            vec![1, 1, 2, 2],
            vec![1, 1, 1, 2],
            vec![1, 1, 1, 1],
            vec![2],
        ];
        let result = kernel(128, Some("fp8_e4m3_block64"), &shapes);
        assert!(result.is_err());
        let message = result.err().unwrap().to_string();
        assert!(message.contains("Unsupported"));
        assert!(message.contains("FP4/FP8 compressed-KV dequantization"));
    }

    #[test]
    fn frozen_v1_stateful_path_is_explicitly_unsupported() {
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let inputs = FROZEN_V1_REQUIRED_INPUT_NAMES
            .iter()
            .map(|name| Some(graph.create_named_value(*name, DataType::Float32, static_shape([1]))))
            .collect();
        let outputs = ["Y", "present_compressed_kv", "present_compression_carry"]
            .into_iter()
            .map(|name| graph.create_named_value(name, DataType::Float32, static_shape([1])))
            .collect();
        let mut node = Node::new(NodeId(0), OP, inputs, outputs);
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(2));
        node.attributes.insert("head_dim".into(), Attribute::Int(2));
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(128));

        let shapes = vec![vec![1]; FROZEN_V1_REQUIRED_INPUTS];
        let kernel = CompressedSparseAttentionFactory
            .create(&node, &shapes)
            .unwrap();
        let owned_inputs = (0..FROZEN_V1_REQUIRED_INPUTS)
            .map(|_| Owned::f32(&[1], &[0.0]))
            .collect::<Vec<_>>();
        let input_views = owned_inputs.iter().map(Owned::view).collect::<Vec<_>>();
        let mut owned_outputs = (0..FROZEN_V1_REQUIRED_OUTPUTS)
            .map(|_| Owned::zeros_f32(&[1]))
            .collect::<Vec<_>>();
        let mut output_views = owned_outputs
            .iter_mut()
            .map(Owned::view_mut)
            .collect::<Vec<_>>();

        let message = kernel
            .execute(&input_views, &mut output_views)
            .unwrap_err()
            .to_string();
        assert!(message.contains("Unsupported"));
        assert!(message.contains("stateful compressed-KV construction"));
        assert!(message.contains("compression-carry updates"));
    }

    #[test]
    fn public_v1_rejects_phase1_reference_arity() {
        let mut graph = Graph::new();
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([1, 1, 2, 2]));
        let mut node = Node::new(NodeId(0), OP, vec![None; 4], vec![output]);
        node.attributes
            .insert("num_heads".into(), Attribute::Int(2));
        node.attributes.insert("head_dim".into(), Attribute::Int(2));
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(128));

        let shapes = vec![vec![]; 4];
        let message = CompressedSparseAttentionFactory
            .create(&node, &shapes)
            .err()
            .unwrap()
            .to_string();
        assert!(message.contains("frozen v1 requires 11..=20 positional inputs"));
    }
}
