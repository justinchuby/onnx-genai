//! CPU reference kernel for frozen `pkg.nxrt::IndexShare` v1.

use std::borrow::Cow;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, Shape};

use super::{check_arity, to_dense_f32, to_dense_i64, write_dense_f32};

const OP: &str = "IndexShare";
const INPUT_NAMES: [&str; 7] = [
    "query",
    "key",
    "value",
    "past_key",
    "past_value",
    "selected_indices",
    "attention_bias",
];

pub struct IndexShareFactory;

pub struct IndexShareKernel {
    num_heads: usize,
    kv_num_heads: usize,
    scale: Option<f32>,
}

#[derive(Clone, Copy)]
struct Attributes {
    num_heads: usize,
    kv_num_heads: usize,
    scale: Option<f32>,
}

impl KernelFactory for IndexShareFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let attrs = validate_metadata(node, None)?;
        Ok(Box::new(IndexShareKernel {
            num_heads: attrs.num_heads,
            kv_num_heads: attrs.kv_num_heads,
            scale: attrs.scale,
        }))
    }
}

pub(crate) fn unsupported_reason(
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
    q_heads: usize,
    kv_heads: usize,
    q_seq: usize,
    current_seq: usize,
    past_seq: usize,
    total_seq: usize,
    head_size: usize,
    index_heads: usize,
    selected_width: usize,
}

struct Bias {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Bias {
    fn at(&self, b: usize, h: usize, q: usize, k: usize) -> f32 {
        let logical = [b, h, q, k];
        let rank = self.shape.len();
        let mut offset = 0;
        for (axis, &dim) in self.shape.iter().enumerate() {
            let index = logical[4 - rank + axis];
            offset = offset * dim + if dim == 1 { 0 } else { index };
        }
        self.data[offset]
    }
}

impl Kernel for IndexShareKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 6, 7, 1)?;
        if !matches!(outputs.len(), 1 | 3) {
            return Err(error(format!(
                "expected 1 output or 3 outputs (paired present K/V), got {}",
                outputs.len()
            )));
        }
        for &index in &[0, 1, 2, 5] {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{}') is absent",
                    INPUT_NAMES[index]
                )));
            }
        }
        let has_past_key = optional_input(inputs, 3).is_some();
        let has_past_value = optional_input(inputs, 4).is_some();
        if has_past_key != has_past_value {
            return Err(error("past_key and past_value must be provided together"));
        }
        for &index in &[0, 1, 2] {
            require_dtype(index, &inputs[index], DataType::Float32)?;
        }
        for index in [3, 4, 6] {
            if let Some(input) = optional_input(inputs, index) {
                require_dtype(index, input, DataType::Float32)?;
            }
        }
        if !matches!(inputs[5].dtype, DataType::Int32 | DataType::Int64) {
            return Err(error(format!(
                "input 5 ('selected_indices') dtype {:?} unsupported; expected Int32 or Int64",
                inputs[5].dtype
            )));
        }
        for (index, output) in outputs.iter().enumerate() {
            if output.dtype != DataType::Float32 {
                return Err(error(format!(
                    "output {index} dtype {:?} unsupported; expected Float32",
                    output.dtype
                )));
            }
        }

        let dims = validate_runtime_shapes(inputs, outputs, self)?;
        let q = to_dense_f32(&inputs[0])?;
        let current_k = to_dense_f32(&inputs[1])?;
        let current_v = to_dense_f32(&inputs[2])?;
        let past_k = optional_input(inputs, 3).map(to_dense_f32).transpose()?;
        let past_v = optional_input(inputs, 4).map(to_dense_f32).transpose()?;
        let indices = to_dense_i64(&inputs[5])?;
        validate_indices(&indices, dims)?;
        let bias = optional_input(inputs, 6)
            .map(|view| {
                Ok::<Bias, EpError>(Bias {
                    data: to_dense_f32(view)?,
                    shape: view.shape.to_vec(),
                })
            })
            .transpose()?;

        let present_k = concatenate_cache(past_k.as_deref(), &current_k, dims);
        let present_v = concatenate_cache(past_v.as_deref(), &current_v, dims);
        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (dims.head_size as f32).sqrt());
        let sqrt_scale = scale.sqrt();
        let group = dims.q_heads / dims.kv_heads;
        let mut output = vec![0.0f32; dims.batch * dims.q_heads * dims.q_seq * dims.head_size];
        let mut scores = vec![0.0f32; dims.selected_width];

        for b in 0..dims.batch {
            for qh in 0..dims.q_heads {
                let kvh = qh / group;
                let ih = if dims.index_heads == 1 { 0 } else { qh };
                for qi in 0..dims.q_seq {
                    let row = ((b * dims.index_heads + ih) * dims.q_seq + qi) * dims.selected_width;
                    let valid = indices[row..row + dims.selected_width]
                        .iter()
                        .take_while(|&&index| index != -1)
                        .count();
                    for selected in 0..valid {
                        let key_position = indices[row + selected] as usize;
                        let mut score = 0.0f32;
                        for d in 0..dims.head_size {
                            let q_offset =
                                ((b * dims.q_heads + qh) * dims.q_seq + qi) * dims.head_size + d;
                            let k_offset = ((b * dims.kv_heads + kvh) * dims.total_seq
                                + key_position)
                                * dims.head_size
                                + d;
                            score +=
                                (q[q_offset] * sqrt_scale) * (present_k[k_offset] * sqrt_scale);
                        }
                        if let Some(bias) = &bias {
                            score += bias.at(b, qh, qi, key_position);
                        }
                        scores[selected] = score;
                    }

                    let max = scores[..valid]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    if max == f32::NEG_INFINITY {
                        continue;
                    }
                    let mut sum = 0.0f32;
                    for score in &mut scores[..valid] {
                        *score = (*score - max).exp();
                        sum += *score;
                    }
                    let inverse = 1.0 / sum;
                    for score in &mut scores[..valid] {
                        *score *= inverse;
                    }
                    let out_base = ((b * dims.q_heads + qh) * dims.q_seq + qi) * dims.head_size;
                    for d in 0..dims.head_size {
                        let mut value = 0.0f32;
                        for selected in 0..valid {
                            let key_position = indices[row + selected] as usize;
                            let v_offset = ((b * dims.kv_heads + kvh) * dims.total_seq
                                + key_position)
                                * dims.head_size
                                + d;
                            value += scores[selected] * present_v[v_offset];
                        }
                        output[out_base + d] = value;
                    }
                }
            }
        }

        write_dense_f32(&mut outputs[0], &output)?;
        if outputs.len() == 3 {
            write_dense_f32(&mut outputs[1], &present_k)?;
            write_dense_f32(&mut outputs[2], &present_v)?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn concatenate_cache(past: Option<&[f32]>, current: &[f32], dims: Dims) -> Vec<f32> {
    if dims.past_seq == 0 {
        return current.to_vec();
    }
    let past = past.expect("past_seq is nonzero only with a past cache");
    let mut present =
        Vec::with_capacity(dims.batch * dims.kv_heads * dims.total_seq * dims.head_size);
    let past_row = dims.past_seq * dims.head_size;
    let current_row = dims.current_seq * dims.head_size;
    for b in 0..dims.batch {
        for h in 0..dims.kv_heads {
            let past_base = (b * dims.kv_heads + h) * past_row;
            let current_base = (b * dims.kv_heads + h) * current_row;
            present.extend_from_slice(&past[past_base..past_base + past_row]);
            present.extend_from_slice(&current[current_base..current_base + current_row]);
        }
    }
    present
}

fn validate_indices(indices: &[i64], dims: Dims) -> Result<()> {
    for b in 0..dims.batch {
        for h in 0..dims.index_heads {
            for q in 0..dims.q_seq {
                let row = ((b * dims.index_heads + h) * dims.q_seq + q) * dims.selected_width;
                let mut previous = None;
                let mut padding = false;
                let mut count = 0;
                for (column, &index) in indices[row..row + dims.selected_width].iter().enumerate() {
                    if index == -1 {
                        padding = true;
                        continue;
                    }
                    if index < -1 {
                        return Err(index_error(
                            b,
                            h,
                            q,
                            column,
                            format!("invalid sentinel {index}"),
                        ));
                    }
                    if padding {
                        return Err(index_error(
                            b,
                            h,
                            q,
                            column,
                            format!("index {index} follows trailing -1 padding"),
                        ));
                    }
                    if index as usize >= dims.total_seq {
                        return Err(index_error(
                            b,
                            h,
                            q,
                            column,
                            format!(
                                "index {index} is out of range for cache length {}",
                                dims.total_seq
                            ),
                        ));
                    }
                    if let Some(previous) = previous
                        && index <= previous
                    {
                        let reason = if index == previous {
                            format!("duplicate index {index}")
                        } else {
                            format!("indices are not strictly increasing: {previous} then {index}")
                        };
                        return Err(index_error(b, h, q, column, reason));
                    }
                    previous = Some(index);
                    count += 1;
                }
                if count == 0 {
                    return Err(error(format!(
                        "selected_indices row [batch={b}, head={h}, query={q}] is all -1"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn index_error(batch: usize, head: usize, query: usize, column: usize, reason: String) -> EpError {
    error(format!(
        "selected_indices [batch={batch}, head={head}, query={query}, column={column}]: {reason}"
    ))
}

fn validate_runtime_shapes(
    inputs: &[TensorView],
    outputs: &[TensorMut],
    kernel: &IndexShareKernel,
) -> Result<Dims> {
    for &index in &[0, 1, 2, 5] {
        require_rank(index, inputs[index].shape, 4)?;
    }
    for index in [3, 4] {
        if let Some(input) = optional_input(inputs, index) {
            require_rank(index, input.shape, 4)?;
        }
    }
    let q = inputs[0].shape;
    let k = inputs[1].shape;
    let v = inputs[2].shape;
    let (batch, q_heads, q_seq, head_size) = (q[0], q[1], q[2], q[3]);
    if q_heads != kernel.num_heads {
        return Err(error(format!(
            "query head dimension {q_heads} must equal num_heads {}",
            kernel.num_heads
        )));
    }
    if k[0] != batch || v[0] != batch {
        return Err(error("query, key, and value batch dimensions must match"));
    }
    if k[1] != kernel.kv_num_heads || v[1] != kernel.kv_num_heads {
        return Err(error(format!(
            "key/value head dimensions must equal kv_num_heads {}",
            kernel.kv_num_heads
        )));
    }
    if k[2] != v[2] || k[3] != head_size || v[3] != head_size {
        return Err(error(
            "key/value sequence and head dimensions must match query/schema",
        ));
    }
    let current_seq = k[2];
    let mut past_seq = 0;
    if let (Some(past_k), Some(past_v)) = (optional_input(inputs, 3), optional_input(inputs, 4)) {
        if past_k.shape != past_v.shape {
            return Err(error("past_key and past_value shapes must match"));
        }
        if past_k.shape[0] != batch
            || past_k.shape[1] != kernel.kv_num_heads
            || past_k.shape[3] != head_size
        {
            return Err(error(
                "past key/value must have shape [B, kv_num_heads, S_past, H]",
            ));
        }
        past_seq = past_k.shape[2];
    }
    let total_seq = past_seq
        .checked_add(current_seq)
        .ok_or_else(|| error("total cache sequence length overflow"))?;
    let selected = inputs[5].shape;
    let index_heads = selected[1];
    if selected[0] != batch
        || !matches!(index_heads, 1) && index_heads != q_heads
        || selected[2] != q_seq
    {
        return Err(error(format!(
            "selected_indices must have shape [B, 1|N, S_q, K], got {selected:?}"
        )));
    }
    if selected[3] == 0 {
        return Err(error("selected_indices K dimension must be nonzero"));
    }
    if let Some(bias) = optional_input(inputs, 6) {
        validate_bias_shape(bias.shape, [batch, q_heads, q_seq, total_seq]).map_err(error)?;
    }
    if outputs[0].shape != q {
        return Err(error(format!(
            "output shape {:?} must equal query shape {q:?}",
            outputs[0].shape
        )));
    }
    if outputs.len() == 3 {
        let expected = [batch, kernel.kv_num_heads, total_seq, head_size];
        if outputs[1].shape != expected || outputs[2].shape != expected {
            return Err(error(format!(
                "present_key and present_value shapes must be {expected:?}"
            )));
        }
    }
    Ok(Dims {
        batch,
        q_heads,
        kv_heads: kernel.kv_num_heads,
        q_seq,
        current_seq,
        past_seq,
        total_seq,
        head_size,
        index_heads,
        selected_width: selected[3],
    })
}

fn validate_metadata(
    node: &Node,
    claim_metadata: Option<(&[Shape], &[DataType])>,
) -> Result<Attributes> {
    for name in node.attributes.keys() {
        if !matches!(name.as_str(), "num_heads" | "kv_num_heads" | "scale") {
            return Err(error(format!(
                "attribute '{name}' is not part of the frozen v1 ABI"
            )));
        }
    }
    let num_heads = required_positive_int(node, "num_heads")?;
    let kv_num_heads = optional_positive_int(node, "kv_num_heads")?.unwrap_or(num_heads);
    if num_heads % kv_num_heads != 0 {
        return Err(error(format!(
            "num_heads {num_heads} must be a multiple of kv_num_heads {kv_num_heads}"
        )));
    }
    let scale = node
        .attr("scale")
        .map(|attribute| {
            attribute
                .as_float()
                .ok_or_else(|| error("attribute 'scale' must be a float"))
        })
        .transpose()?;
    if scale.is_some_and(|scale| !scale.is_finite() || scale <= 0.0) {
        return Err(error("attribute 'scale' must be finite and > 0"));
    }
    let attrs = Attributes {
        num_heads,
        kv_num_heads,
        scale,
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
    if !(6..=7).contains(&node.inputs.len()) {
        return Err(format!(
            "expected 6 or 7 positional inputs, got {}",
            node.inputs.len()
        ));
    }
    if !matches!(node.outputs.len(), 1 | 3) {
        return Err(format!(
            "expected 1 output or 3 outputs, got {}",
            node.outputs.len()
        ));
    }
    if shapes.len() != node.inputs.len() || dtypes.len() != node.inputs.len() {
        return Err(format!(
            "claim metadata must cover all {} positional inputs",
            node.inputs.len()
        ));
    }
    for &index in &[0, 1, 2, 5] {
        if node.inputs[index].is_none() {
            return Err(format!(
                "required input {index} ('{}') is omitted",
                INPUT_NAMES[index]
            ));
        }
    }
    if node.inputs[3].is_some() != node.inputs[4].is_some() {
        return Err("past_key and past_value must be provided together".into());
    }
    for index in 0..node.inputs.len() {
        if node.inputs[index].is_none() {
            if dtypes[index] != DataType::Undefined {
                return Err(format!(
                    "omitted input {index} ('{}') must use dtype Undefined",
                    INPUT_NAMES[index]
                ));
            }
            continue;
        }
        let valid = if index == 5 {
            matches!(dtypes[index], DataType::Int32 | DataType::Int64)
        } else {
            dtypes[index] == DataType::Float32
        };
        if !valid {
            return Err(format!(
                "input {index} ('{}') dtype {:?} unsupported",
                INPUT_NAMES[index], dtypes[index]
            ));
        }
    }
    for &index in &[0, 1, 2, 5] {
        if shapes[index].len() != 4 {
            return Err(format!(
                "input {index} ('{}') rank {} unsupported; expected 4",
                INPUT_NAMES[index],
                shapes[index].len()
            ));
        }
    }
    for index in [3, 4] {
        if node.inputs[index].is_some() && shapes[index].len() != 4 {
            return Err(format!(
                "input {index} ('{}') rank {} unsupported; expected 4",
                INPUT_NAMES[index],
                shapes[index].len()
            ));
        }
    }
    if node.inputs.get(6).is_some_and(Option::is_some) && shapes[6].len() > 4 {
        return Err(format!(
            "input 6 ('attention_bias') rank {} unsupported; expected at most 4",
            shapes[6].len()
        ));
    }
    check_static_dim(&shapes[0], 1, attrs.num_heads, "query num_heads")?;
    check_static_dim(&shapes[1], 1, attrs.kv_num_heads, "key kv_num_heads")?;
    check_static_dim(&shapes[2], 1, attrs.kv_num_heads, "value kv_num_heads")?;
    require_same_static(&shapes[0], 0, &shapes[1], 0, "query/key batch")?;
    require_same_static(&shapes[0], 0, &shapes[2], 0, "query/value batch")?;
    require_same_static(&shapes[1], 2, &shapes[2], 2, "key/value sequence")?;
    require_same_static(&shapes[0], 3, &shapes[1], 3, "query/key head size")?;
    require_same_static(&shapes[0], 3, &shapes[2], 3, "query/value head size")?;
    require_same_static(&shapes[0], 0, &shapes[5], 0, "query/index batch")?;
    require_same_static(&shapes[0], 2, &shapes[5], 2, "query/index sequence")?;
    if let Some(index_heads) = shapes[5][1].as_static()
        && index_heads != 1
        && index_heads != attrs.num_heads
    {
        return Err(format!(
            "selected_indices head dimension must be 1 or {}, got {index_heads}",
            attrs.num_heads
        ));
    }
    if shapes[5][3].as_static() == Some(0) {
        return Err("selected_indices K dimension must be nonzero".into());
    }
    if node.inputs[3].is_some() {
        for index in [3, 4] {
            check_static_dim(&shapes[index], 1, attrs.kv_num_heads, "past kv_num_heads")?;
            require_same_static(&shapes[0], 0, &shapes[index], 0, "query/past batch")?;
            require_same_static(&shapes[0], 3, &shapes[index], 3, "query/past head size")?;
        }
        require_same_static(&shapes[3], 2, &shapes[4], 2, "past key/value sequence")?;
    }
    if node.inputs.get(6).is_some_and(Option::is_some) {
        validate_static_bias_shape(
            &shapes[6],
            &shapes[0],
            &shapes[1],
            node.inputs[3].is_some().then_some(&shapes[3]),
        )?;
    }
    Ok(())
}

fn validate_bias_shape(shape: &[usize], target: [usize; 4]) -> std::result::Result<(), String> {
    if shape.len() > 4 {
        return Err(format!("attention_bias rank {} exceeds 4", shape.len()));
    }
    for (axis, &dimension) in shape.iter().enumerate() {
        let expected = target[4 - shape.len() + axis];
        if dimension != 1 && dimension != expected {
            return Err(format!(
                "attention_bias dimension {dimension} is not broadcastable to {target:?}"
            ));
        }
    }
    Ok(())
}

fn validate_static_bias_shape(
    bias: &Shape,
    query: &Shape,
    key: &Shape,
    past: Option<&Shape>,
) -> std::result::Result<(), String> {
    for (axis, dim) in bias.iter().enumerate() {
        let Some(actual) = dim.as_static() else {
            continue;
        };
        if actual == 1 {
            continue;
        }
        let target_axis = 4 - bias.len() + axis;
        let expected = match target_axis {
            0 => query[0].as_static(),
            1 => query[1].as_static(),
            2 => query[2].as_static(),
            3 => match (past, key[2].as_static()) {
                (None, current) => current,
                (Some(past), Some(current)) => past[2]
                    .as_static()
                    .and_then(|past| past.checked_add(current)),
                _ => None,
            },
            _ => unreachable!("target rank is four"),
        };
        if let Some(expected) = expected
            && actual != expected
        {
            return Err(format!(
                "attention_bias dimension {actual} is not broadcastable to target axis {target_axis} size {expected}"
            ));
        }
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

fn optional_positive_int(node: &Node, name: &str) -> Result<Option<usize>> {
    node.attr(name)
        .map(|attribute| {
            let value = attribute
                .as_int()
                .ok_or_else(|| error(format!("attribute '{name}' must be an integer")))?;
            usize::try_from(value)
                .ok()
                .filter(|&value| value > 0)
                .ok_or_else(|| error(format!("attribute '{name}' must be > 0")))
        })
        .transpose()
}

fn require_rank(index: usize, shape: &[usize], expected: usize) -> Result<()> {
    if shape.len() != expected {
        return Err(error(format!(
            "input {index} ('{}') rank {} unsupported; expected {expected}",
            INPUT_NAMES[index],
            shape.len()
        )));
    }
    Ok(())
}

fn require_dtype(index: usize, input: &TensorView, expected: DataType) -> Result<()> {
    if input.dtype != expected {
        return Err(error(format!(
            "input {index} ('{}') dtype {:?} unsupported; expected {expected:?}",
            INPUT_NAMES[index], input.dtype
        )));
    }
    Ok(())
}

fn optional_input<'a>(inputs: &'a [TensorView<'a>], index: usize) -> Option<&'a TensorView<'a>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OP}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::{ExecutionProvider, TensorView};
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};

    #[derive(Clone, Copy)]
    struct Case {
        batch: usize,
        q_heads: usize,
        kv_heads: usize,
        q_seq: usize,
        current_seq: usize,
        past_seq: usize,
        head_size: usize,
        index_heads: usize,
        width: usize,
        scale: f32,
    }

    fn node(case: Case, with_past: bool, with_bias: bool, outputs: usize) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let specs = [
            Some((
                DataType::Float32,
                vec![case.batch, case.q_heads, case.q_seq, case.head_size],
            )),
            Some((
                DataType::Float32,
                vec![case.batch, case.kv_heads, case.current_seq, case.head_size],
            )),
            Some((
                DataType::Float32,
                vec![case.batch, case.kv_heads, case.current_seq, case.head_size],
            )),
            with_past.then_some((
                DataType::Float32,
                vec![case.batch, case.kv_heads, case.past_seq, case.head_size],
            )),
            with_past.then_some((
                DataType::Float32,
                vec![case.batch, case.kv_heads, case.past_seq, case.head_size],
            )),
            Some((
                DataType::Int64,
                vec![case.batch, case.index_heads, case.q_seq, case.width],
            )),
            with_bias.then_some((
                DataType::Float32,
                vec![
                    case.batch,
                    case.q_heads,
                    case.q_seq,
                    case.past_seq + case.current_seq,
                ],
            )),
        ];
        let inputs = specs
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                spec.as_ref().map(|(dtype, shape)| {
                    let value = graph.create_named_value(
                        format!("input_{index}"),
                        *dtype,
                        static_shape(shape.iter().copied()),
                    );
                    graph.add_input(value);
                    value
                })
            })
            .collect();
        let mut node_outputs = Vec::new();
        let output = graph.create_named_value(
            "output",
            DataType::Float32,
            static_shape([case.batch, case.q_heads, case.q_seq, case.head_size]),
        );
        node_outputs.push(output);
        if outputs == 3 {
            for name in ["present_key", "present_value"] {
                node_outputs.push(graph.create_named_value(
                    name,
                    DataType::Float32,
                    static_shape([
                        case.batch,
                        case.kv_heads,
                        case.past_seq + case.current_seq,
                        case.head_size,
                    ]),
                ));
            }
        }
        let mut node = Node::new(NodeId(0), OP, inputs, node_outputs);
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(case.q_heads as i64));
        node.attributes
            .insert("kv_num_heads".into(), Attribute::Int(case.kv_heads as i64));
        node.attributes
            .insert("scale".into(), Attribute::Float(case.scale));
        let id = graph.insert_node(node);
        (graph, id)
    }

    #[allow(clippy::too_many_arguments)]
    fn dense_oracle(
        case: Case,
        q: &[f32],
        present_k: &[f32],
        present_v: &[f32],
        indices: &[i64],
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let total = case.past_seq + case.current_seq;
        let group = case.q_heads / case.kv_heads;
        let sqrt_scale = case.scale.sqrt();
        let mut output = vec![0.0; case.batch * case.q_heads * case.q_seq * case.head_size];
        for b in 0..case.batch {
            for qh in 0..case.q_heads {
                let kvh = qh / group;
                let ih = if case.index_heads == 1 { 0 } else { qh };
                for qi in 0..case.q_seq {
                    let index_base = ((b * case.index_heads + ih) * case.q_seq + qi) * case.width;
                    let mut selected = vec![false; total];
                    for &index in &indices[index_base..index_base + case.width] {
                        if index >= 0 {
                            selected[index as usize] = true;
                        }
                    }
                    let mut scores = vec![f32::NEG_INFINITY; total];
                    for key_position in 0..total {
                        if !selected[key_position] {
                            continue;
                        }
                        let mut score = 0.0f32;
                        for d in 0..case.head_size {
                            let qo =
                                ((b * case.q_heads + qh) * case.q_seq + qi) * case.head_size + d;
                            let ko = ((b * case.kv_heads + kvh) * total + key_position)
                                * case.head_size
                                + d;
                            score += (q[qo] * sqrt_scale) * (present_k[ko] * sqrt_scale);
                        }
                        if let Some(bias) = bias {
                            score += bias[((b * case.q_heads + qh) * case.q_seq + qi) * total
                                + key_position];
                        }
                        scores[key_position] = score;
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    if max == f32::NEG_INFINITY {
                        continue;
                    }
                    let mut sum = 0.0f32;
                    for score in &mut scores {
                        *score = (*score - max).exp();
                        sum += *score;
                    }
                    for score in &mut scores {
                        *score *= 1.0 / sum;
                    }
                    for d in 0..case.head_size {
                        let mut value = 0.0f32;
                        for (key_position, &probability) in scores.iter().enumerate() {
                            let vo = ((b * case.kv_heads + kvh) * total + key_position)
                                * case.head_size
                                + d;
                            value += probability * present_v[vo];
                        }
                        output[((b * case.q_heads + qh) * case.q_seq + qi) * case.head_size + d] =
                            value;
                    }
                }
            }
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        case: Case,
        q: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        past_k: Option<&[f32]>,
        past_v: Option<&[f32]>,
        indices: &[i64],
        bias: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (graph, id) = node(case, past_k.is_some(), bias.is_some(), 3);
        let kernel = IndexShareFactory.create(graph.node(id), &[])?;
        let q = Owned::f32(&[case.batch, case.q_heads, case.q_seq, case.head_size], q);
        let current_k = Owned::f32(
            &[case.batch, case.kv_heads, case.current_seq, case.head_size],
            current_k,
        );
        let current_v = Owned::f32(
            &[case.batch, case.kv_heads, case.current_seq, case.head_size],
            current_v,
        );
        let past_k = past_k.map(|data| {
            Owned::f32(
                &[case.batch, case.kv_heads, case.past_seq, case.head_size],
                data,
            )
        });
        let past_v = past_v.map(|data| {
            Owned::f32(
                &[case.batch, case.kv_heads, case.past_seq, case.head_size],
                data,
            )
        });
        let indices = Owned::i64(
            &[case.batch, case.index_heads, case.q_seq, case.width],
            indices,
        );
        let bias = bias.map(|data| {
            Owned::f32(
                &[
                    case.batch,
                    case.q_heads,
                    case.q_seq,
                    case.past_seq + case.current_seq,
                ],
                data,
            )
        });
        let absent = TensorView::absent(DataType::Undefined);
        let inputs = vec![
            q.view(),
            current_k.view(),
            current_v.view(),
            past_k.as_ref().map_or(absent, Owned::view),
            past_v.as_ref().map_or(absent, Owned::view),
            indices.view(),
            bias.as_ref().map_or(absent, Owned::view),
        ];
        let mut output = Owned::zeros_f32(&[case.batch, case.q_heads, case.q_seq, case.head_size]);
        let mut present_k = Owned::zeros_f32(&[
            case.batch,
            case.kv_heads,
            case.past_seq + case.current_seq,
            case.head_size,
        ]);
        let mut present_v = Owned::zeros_f32(&[
            case.batch,
            case.kv_heads,
            case.past_seq + case.current_seq,
            case.head_size,
        ]);
        kernel.execute(
            &inputs,
            &mut [
                output.view_mut(),
                present_k.view_mut(),
                present_v.view_mut(),
            ],
        )?;
        Ok((output.to_f32(), present_k.to_f32(), present_v.to_f32()))
    }

    fn sequence(count: usize, offset: f32) -> Vec<f32> {
        (0..count)
            .map(|index| offset + index as f32 * 0.0625)
            .collect()
    }

    #[test]
    fn selected_subset_and_trailing_padding_match_dense_additive_mask_exactly() {
        let case = Case {
            batch: 1,
            q_heads: 2,
            kv_heads: 2,
            q_seq: 1,
            current_seq: 5,
            past_seq: 0,
            head_size: 3,
            index_heads: 2,
            width: 4,
            scale: 0.5,
        };
        let q = sequence(6, -0.25);
        let k = sequence(30, 0.125);
        let v = sequence(30, -1.0);
        let indices = [0, 2, 4, -1, 1, 3, -1, -1];
        let (actual, present_k, present_v) =
            run(case, &q, &k, &v, None, None, &indices, None).unwrap();
        let expected = dense_oracle(case, &q, &present_k, &present_v, &indices, None);
        assert_eq!(actual, expected);
    }

    #[test]
    fn gqa_shared_indices_match_dense_additive_mask_exactly() {
        let case = Case {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            q_seq: 2,
            current_seq: 4,
            past_seq: 1,
            head_size: 2,
            index_heads: 1,
            width: 3,
            scale: 0.25,
        };
        let q = sequence(16, -0.5);
        let past_k = sequence(4, 0.25);
        let past_v = sequence(4, -0.75);
        let k = sequence(16, 0.5);
        let v = sequence(16, 1.0);
        let indices = [0, 2, 4, 1, 3, 4];
        let (actual, present_k, present_v) = run(
            case,
            &q,
            &k,
            &v,
            Some(&past_k),
            Some(&past_v),
            &indices,
            None,
        )
        .unwrap();
        let expected = dense_oracle(case, &q, &present_k, &present_v, &indices, None);
        assert_eq!(actual, expected);
    }

    #[test]
    fn causal_and_padding_bias_composition_matches_dense_oracle_exactly() {
        let case = Case {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            q_seq: 2,
            current_seq: 3,
            past_seq: 2,
            head_size: 2,
            index_heads: 1,
            width: 4,
            scale: 0.5,
        };
        let q = sequence(4, 0.25);
        let past_k = sequence(4, -0.5);
        let past_v = sequence(4, 0.75);
        let k = sequence(6, 0.125);
        let v = sequence(6, -1.25);
        let indices = [0, 1, 2, 4, 0, 1, 3, 4];
        let mut bias = vec![0.0; 10];
        for qi in 0..case.q_seq {
            for key in 0..5 {
                if key > case.past_seq + qi || key == 1 {
                    bias[qi * 5 + key] = f32::NEG_INFINITY;
                }
            }
        }
        let (actual, present_k, present_v) = run(
            case,
            &q,
            &k,
            &v,
            Some(&past_k),
            Some(&past_v),
            &indices,
            Some(&bias),
        )
        .unwrap();
        let expected = dense_oracle(case, &q, &present_k, &present_v, &indices, Some(&bias));
        assert_eq!(actual, expected);
    }

    #[test]
    fn rejects_invalid_index_rows_at_execution() {
        let case = Case {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            q_seq: 1,
            current_seq: 3,
            past_seq: 0,
            head_size: 1,
            index_heads: 1,
            width: 3,
            scale: 1.0,
        };
        for (indices, expected) in [
            ([0, 0, -1], "duplicate"),
            ([0, 3, -1], "out of range"),
            ([1, 0, -1], "not strictly increasing"),
            ([-1, -1, -1], "all -1"),
        ] {
            let error = run(
                case,
                &[1.0],
                &[1.0, 2.0, 3.0],
                &[4.0, 5.0, 6.0],
                None,
                None,
                &indices,
                None,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn claim_gate_accepts_omitted_optionals_and_rejects_present_wrong_dtype() {
        let case = Case {
            batch: 1,
            q_heads: 2,
            kv_heads: 1,
            q_seq: 1,
            current_seq: 3,
            past_seq: 0,
            head_size: 2,
            index_heads: 1,
            width: 2,
            scale: 0.5,
        };
        let (graph, id) = node(case, false, false, 1);
        let node = graph.node(id);
        let shapes = node
            .inputs
            .iter()
            .map(|input| input.map_or_else(Vec::new, |value| graph.value(value).shape.clone()))
            .collect::<Vec<_>>();
        let mut dtypes = node
            .inputs
            .iter()
            .map(|input| input.map_or(DataType::Undefined, |value| graph.value(value).dtype))
            .collect::<Vec<_>>();
        let ep = CpuExecutionProvider::new();
        assert!(
            ep.supports_op(node, 1, &shapes, &dtypes, &[])
                .is_supported()
        );
        dtypes[5] = DataType::Float32;
        let rejected = ep.supports_op(node, 1, &shapes, &dtypes, &[]);
        assert!(rejected.reason().unwrap().contains("selected_indices"));
    }
}
