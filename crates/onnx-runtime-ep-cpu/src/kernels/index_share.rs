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

/// Claim-time gate shared with the CUDA execution provider so the device kernel
/// rejects exactly the dtype/layout/arity/shape combinations the CPU oracle
/// does (keeping the two backends' `supports_op` contracts in lockstep).
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
    q_heads: usize,
    kv_heads: usize,
    q_seq: usize,
    current_seq: usize,
    past_seq: usize,
    total_seq: usize,
    head_size: usize,
    index_heads: usize,
    selected_width: usize,
    /// Per-head row stride of the present cache the attention gathers from:
    /// `total_seq` for the growing concat present, or the fixed `past_seq`
    /// capacity when the present aliases `past` in place (capacity mode).
    cache_seq: usize,
    /// `true` when the 3-output present aliases the fixed-capacity `past`
    /// bindings (present sequence == past sequence) instead of growing via
    /// `past ⧺ current`. Selected for whole-step CUDA-graph capture.
    capacity_mode: bool,
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
        let bias = optional_input(inputs, 6)
            .map(|view| {
                Ok::<Bias, EpError>(Bias {
                    data: to_dense_f32(view)?,
                    shape: view.shape.to_vec(),
                })
            })
            .transpose()?;

        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (dims.head_size as f32).sqrt());
        let sqrt_scale = scale.sqrt();
        let mut output = vec![0.0f32; dims.batch * dims.q_heads * dims.q_seq * dims.head_size];

        if dims.capacity_mode {
            // The 3-output present aliases the fixed-capacity `past` bindings in
            // place (no growing `past ⧺ current`). The valid length — hence the
            // write position of the current token(s) and the index range — is
            // carried by the causal/padding `attention_bias` frontier, because
            // the capacity-sized `past` shape no longer encodes it.
            let bias = bias.as_ref().ok_or_else(|| {
                error(
                    "capacity-mode IndexShare (present aliases fixed-capacity past) requires attention_bias to carry the valid length",
                )
            })?;
            let past_k = past_k
                .as_deref()
                .expect("capacity mode requires a past cache");
            let past_v = past_v
                .as_deref()
                .expect("capacity mode requires a past cache");
            let valid_lens = capacity_valid_lens(bias, dims);
            validate_indices(&indices, dims, &valid_lens)?;
            let present_k = build_capacity_present(past_k, &current_k, dims, &valid_lens);
            let present_v = build_capacity_present(past_v, &current_v, dims, &valid_lens);
            attend_selected(
                &mut output,
                &present_k,
                &present_v,
                dims.cache_seq,
                &q,
                &indices,
                Some(bias),
                dims,
                sqrt_scale,
            );
            write_dense_f32(&mut outputs[0], &output)?;
            write_dense_f32(&mut outputs[1], &present_k)?;
            write_dense_f32(&mut outputs[2], &present_v)?;
            return Ok(());
        }

        validate_indices(&indices, dims, &vec![dims.total_seq; dims.batch])?;
        let present_k = concatenate_cache(past_k.as_deref(), &current_k, dims);
        let present_v = concatenate_cache(past_v.as_deref(), &current_v, dims);

        // The concat present packs exactly `total_seq` positions per head, so
        // the cache row stride equals `total_seq`.
        attend_selected(
            &mut output,
            &present_k,
            &present_v,
            dims.total_seq,
            &q,
            &indices,
            bias.as_ref(),
            dims,
            sqrt_scale,
        );

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

/// Compute IndexShare attention into `output` over a present cache whose
/// per-head row stride is `cache_seq` positions.
///
/// The concat present path passes `cache_seq == dims.total_seq`. A
/// fixed-capacity ("in-place") present — where the cache aliases `past` at
/// `max_len` and the current token's K/V is written at its absolute position —
/// passes `cache_seq == capacity`. The attention result is **byte-identical**
/// for both layouts because the loop only ever reads key positions named by
/// `selected_indices`, and those indices carry the same absolute positions
/// regardless of how many trailing capacity rows the cache reserves. This is
/// the numerical contract the device capacity-present kernel must match.
#[allow(clippy::too_many_arguments)]
fn attend_selected(
    output: &mut [f32],
    present_k: &[f32],
    present_v: &[f32],
    cache_seq: usize,
    q: &[f32],
    indices: &[i64],
    bias: Option<&Bias>,
    dims: Dims,
    sqrt_scale: f32,
) {
    let group = dims.q_heads / dims.kv_heads;
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
                        let k_offset = ((b * dims.kv_heads + kvh) * cache_seq + key_position)
                            * dims.head_size
                            + d;
                        score += (q[q_offset] * sqrt_scale) * (present_k[k_offset] * sqrt_scale);
                    }
                    if let Some(bias) = bias {
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
                        let v_offset = ((b * dims.kv_heads + kvh) * cache_seq + key_position)
                            * dims.head_size
                            + d;
                        value += scores[selected] * present_v[v_offset];
                    }
                    output[out_base + d] = value;
                }
            }
        }
    }
}

/// Build a fixed-capacity ("in-place") present cache that aliases `past` at
/// `capacity` positions instead of growing it via concat.
///
/// Positions `[0, past_seq)` hold `past`, the current token's K/V is written at
/// `[past_seq, past_seq + current_seq)`, and any trailing positions up to
/// `capacity` are left as `fill`. Those trailing rows are never read by
/// [`attend_selected`] because `selected_indices` only name positions
/// `< total_seq`, so attention over this layout matches the concat present
/// byte-for-byte. This mirrors the device kernel's capacity-present mode, where
/// present aliases the fixed-capacity `past_key`/`past_value` bindings.
#[cfg(test)]
fn capacity_present(
    past: Option<&[f32]>,
    current: &[f32],
    dims: Dims,
    capacity: usize,
    fill: f32,
) -> Vec<f32> {
    assert!(
        capacity >= dims.total_seq,
        "capacity {capacity} must hold at least total_seq {}",
        dims.total_seq
    );
    let mut present = vec![fill; dims.batch * dims.kv_heads * capacity * dims.head_size];
    let past_row = dims.past_seq * dims.head_size;
    let current_row = dims.current_seq * dims.head_size;
    let cap_row = capacity * dims.head_size;
    for b in 0..dims.batch {
        for h in 0..dims.kv_heads {
            let dst_base = (b * dims.kv_heads + h) * cap_row;
            if let Some(past) = past {
                let past_base = (b * dims.kv_heads + h) * past_row;
                present[dst_base..dst_base + past_row]
                    .copy_from_slice(&past[past_base..past_base + past_row]);
            }
            let current_base = (b * dims.kv_heads + h) * current_row;
            let write = dst_base + dims.past_seq * dims.head_size;
            present[write..write + current_row]
                .copy_from_slice(&current[current_base..current_base + current_row]);
        }
    }
    present
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

/// Per-batch valid length carried by the capacity-mode `attention_bias`
/// frontier: `valid_len[b] = 1 + max{ k : bias(b, ·, ·, k) is finite }`.
///
/// A causal/padding mask is finite for every attended (valid) key position and
/// `-inf` beyond it, so the rightmost finite column marks the logical cache
/// length. The current token(s) occupy `[valid_len - current_seq, valid_len)`.
/// This mirrors how default-domain `Attention` derives its valid length from
/// the additive mask at fixed capacity (see the session executor's
/// `kernel_input_uses_physical_capacity`).
fn capacity_valid_lens(bias: &Bias, dims: Dims) -> Vec<usize> {
    (0..dims.batch)
        .map(|b| {
            let mut valid = 0usize;
            for h in 0..dims.q_heads {
                for qi in 0..dims.q_seq {
                    for k in 0..dims.cache_seq {
                        if bias.at(b, h, qi, k).is_finite() {
                            valid = valid.max(k + 1);
                        }
                    }
                }
            }
            valid
        })
        .collect()
}

/// Build the fixed-capacity ("in-place") present that aliases `past` at
/// `dims.cache_seq` positions: every capacity row is copied from `past`, then
/// the current token(s) are written at `[valid_len - current_seq, valid_len)`.
///
/// Positions at or beyond `valid_len` are never read by [`attend_selected`]
/// (the indices only name positions `< valid_len`), so attention over this
/// layout matches the growing concat present byte-for-byte. `past` is supplied
/// at capacity (`past_seq == cache_seq`), so the present has the same shape as
/// the `past` binding — the aliasing the device capacity kernel performs.
fn build_capacity_present(
    past: &[f32],
    current: &[f32],
    dims: Dims,
    valid_lens: &[usize],
) -> Vec<f32> {
    let cap_row = dims.cache_seq * dims.head_size;
    let current_row = dims.current_seq * dims.head_size;
    let mut present = vec![0.0f32; dims.batch * dims.kv_heads * cap_row];
    for (b, &valid_len) in valid_lens.iter().enumerate() {
        let write_pos = valid_len.saturating_sub(dims.current_seq);
        for h in 0..dims.kv_heads {
            let dst = (b * dims.kv_heads + h) * cap_row;
            present[dst..dst + cap_row].copy_from_slice(&past[dst..dst + cap_row]);
            let current_base = (b * dims.kv_heads + h) * current_row;
            let at = dst + write_pos * dims.head_size;
            present[at..at + current_row]
                .copy_from_slice(&current[current_base..current_base + current_row]);
        }
    }
    present
}

fn validate_indices(indices: &[i64], dims: Dims, per_batch_bound: &[usize]) -> Result<()> {
    for (b, &bound) in per_batch_bound.iter().enumerate() {
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
                    if index as usize >= bound {
                        return Err(index_error(
                            b,
                            h,
                            q,
                            column,
                            format!("index {index} is out of range for cache length {bound}"),
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
    if outputs[0].shape != q {
        return Err(error(format!(
            "output shape {:?} must equal query shape {q:?}",
            outputs[0].shape
        )));
    }
    let mut cache_seq = total_seq;
    let mut capacity_mode = false;
    if outputs.len() == 3 {
        let concat = [batch, kernel.kv_num_heads, total_seq, head_size];
        let capacity = [batch, kernel.kv_num_heads, past_seq, head_size];
        // The present may either grow (`past ⧺ current`, sequence == total_seq)
        // or alias the fixed-capacity `past` in place (sequence == past_seq,
        // requires a past cache). `past_seq < total_seq` always (current_seq >=
        // 1), so the two shapes are unambiguous and no existing concat caller
        // trips the capacity branch.
        if outputs[1].shape == concat && outputs[2].shape == concat {
            // Growing concat present.
        } else if past_seq > 0 && outputs[1].shape == capacity && outputs[2].shape == capacity {
            capacity_mode = true;
            cache_seq = past_seq;
        } else {
            return Err(error(format!(
                "present_key and present_value shapes must be {concat:?} (growing) or {capacity:?} (fixed capacity)"
            )));
        }
    }
    // The bias spans the gathered cache: `total_seq` for the concat present, or
    // the fixed `cache_seq` capacity when the present aliases past in place.
    if let Some(bias) = optional_input(inputs, 6) {
        validate_bias_shape(bias.shape, [batch, q_heads, q_seq, cache_seq]).map_err(error)?;
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
        cache_seq,
        capacity_mode,
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

    #[derive(Clone, Copy, Debug)]
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
    fn rejects_index_after_interior_padding_at_execution() {
        let case = Case {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            q_seq: 1,
            current_seq: 6,
            past_seq: 0,
            head_size: 1,
            index_heads: 1,
            width: 3,
            scale: 1.0,
        };
        let error = run(
            case,
            &[1.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            None,
            None,
            &[2, -1, 5],
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("index 5 follows trailing -1 padding"),
            "{error}"
        );
    }

    /// Build a capacity-mode IndexShare node: `past`/`present` are sized to a
    /// fixed `capacity` (present aliases past), and `attention_bias` of width
    /// `capacity` carries the valid length via its finite frontier.
    #[allow(clippy::too_many_arguments)]
    fn run_capacity(
        case: Case,
        capacity: usize,
        q: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        past_k: &[f32],
        past_v: &[f32],
        indices: &[i64],
        bias: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let specs = [
            (
                DataType::Float32,
                vec![case.batch, case.q_heads, case.q_seq, case.head_size],
            ),
            (
                DataType::Float32,
                vec![case.batch, case.kv_heads, case.current_seq, case.head_size],
            ),
            (
                DataType::Float32,
                vec![case.batch, case.kv_heads, case.current_seq, case.head_size],
            ),
            (
                DataType::Float32,
                vec![case.batch, case.kv_heads, capacity, case.head_size],
            ),
            (
                DataType::Float32,
                vec![case.batch, case.kv_heads, capacity, case.head_size],
            ),
            (
                DataType::Int64,
                vec![case.batch, case.index_heads, case.q_seq, case.width],
            ),
            (
                DataType::Float32,
                vec![case.batch, case.q_heads, case.q_seq, capacity],
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
        let mut node_outputs = vec![graph.create_named_value(
            "output",
            DataType::Float32,
            static_shape([case.batch, case.q_heads, case.q_seq, case.head_size]),
        )];
        for name in ["present_key", "present_value"] {
            node_outputs.push(graph.create_named_value(
                name,
                DataType::Float32,
                static_shape([case.batch, case.kv_heads, capacity, case.head_size]),
            ));
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
        let past_k = Owned::f32(&[case.batch, case.kv_heads, capacity, case.head_size], past_k);
        let past_v = Owned::f32(&[case.batch, case.kv_heads, capacity, case.head_size], past_v);
        let indices = Owned::i64(
            &[case.batch, case.index_heads, case.q_seq, case.width],
            indices,
        );
        let bias = Owned::f32(&[case.batch, case.q_heads, case.q_seq, capacity], bias);
        let inputs = vec![
            q.view(),
            current_k.view(),
            current_v.view(),
            past_k.view(),
            past_v.view(),
            indices.view(),
            bias.view(),
        ];
        let mut output = Owned::zeros_f32(&[case.batch, case.q_heads, case.q_seq, case.head_size]);
        let mut present_k =
            Owned::zeros_f32(&[case.batch, case.kv_heads, capacity, case.head_size]);
        let mut present_v =
            Owned::zeros_f32(&[case.batch, case.kv_heads, capacity, case.head_size]);
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

    /// End-to-end through the public kernel API: a capacity-mode call (present
    /// aliases fixed-capacity past, valid length from the bias frontier) yields
    /// the byte-identical attention output and the same valid present region as
    /// the growing concat call. This is what keeps CPU==CUDA parity once the
    /// device kernel and executor bind the KV cache at capacity for capture.
    #[test]
    fn capacity_mode_execute_matches_concat_execute_byte_for_byte() {
        let case = Case {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            q_seq: 1,
            current_seq: 1,
            past_seq: 5,
            head_size: 3,
            index_heads: 4,
            width: 4,
            scale: 0.25,
        };
        let total = case.past_seq + case.current_seq;
        let q = sequence(case.batch * case.q_heads * case.q_seq * case.head_size, 0.25);
        let past_k = sequence(case.batch * case.kv_heads * case.past_seq * case.head_size, -0.5);
        let past_v = sequence(case.batch * case.kv_heads * case.past_seq * case.head_size, 0.75);
        let current_k =
            sequence(case.batch * case.kv_heads * case.current_seq * case.head_size, 0.125);
        let current_v =
            sequence(case.batch * case.kv_heads * case.current_seq * case.head_size, -1.25);
        // Strictly-increasing causal indices per (index-head, query).
        let mut indices = vec![-1i64; case.batch * case.index_heads * case.q_seq * case.width];
        for ih in 0..case.index_heads {
            let row = (ih * case.q_seq) * case.width;
            for k in 0..case.width.min(total) {
                indices[row + k] = (total - case.width.min(total) + k) as i64;
            }
        }
        // Concat reference with a causal bias over [0, total).
        let concat_bias = vec![0.0f32; case.batch * case.q_heads * case.q_seq * total];
        let (out_concat, pk_concat, pv_concat) = run(
            case,
            &q,
            &current_k,
            &current_v,
            Some(&past_k),
            Some(&past_v),
            &indices,
            Some(&concat_bias),
        )
        .unwrap();

        for capacity in [total, total + 3, total + 9] {
            // Capacity past: valid [0, past_seq) from past, garbage after.
            let mut cap_k =
                vec![987.0f32; case.batch * case.kv_heads * capacity * case.head_size];
            let mut cap_v =
                vec![-654.0f32; case.batch * case.kv_heads * capacity * case.head_size];
            let past_row = case.past_seq * case.head_size;
            let cap_row = capacity * case.head_size;
            for bh in 0..case.batch * case.kv_heads {
                cap_k[bh * cap_row..bh * cap_row + past_row]
                    .copy_from_slice(&past_k[bh * past_row..bh * past_row + past_row]);
                cap_v[bh * cap_row..bh * cap_row + past_row]
                    .copy_from_slice(&past_v[bh * past_row..bh * past_row + past_row]);
            }
            // Capacity bias: finite over [0, total), -inf beyond (the frontier).
            let mut cap_bias =
                vec![f32::NEG_INFINITY; case.batch * case.q_heads * case.q_seq * capacity];
            for bhq in 0..case.batch * case.q_heads * case.q_seq {
                for k in 0..total {
                    cap_bias[bhq * capacity + k] = 0.0;
                }
            }
            let (out_cap, pk_cap, pv_cap) = run_capacity(
                case, capacity, &q, &current_k, &current_v, &cap_k, &cap_v, &indices, &cap_bias,
            )
            .unwrap();
            assert_eq!(out_cap, out_concat, "capacity {capacity}: output diverged");
            // The valid present region [0, total) must equal the concat present.
            let cap_row = capacity * case.head_size;
            let concat_row = total * case.head_size;
            for bh in 0..case.batch * case.kv_heads {
                assert_eq!(
                    &pk_cap[bh * cap_row..bh * cap_row + concat_row],
                    &pk_concat[bh * concat_row..bh * concat_row + concat_row],
                    "capacity {capacity}: present_key valid region diverged"
                );
                assert_eq!(
                    &pv_cap[bh * cap_row..bh * cap_row + concat_row],
                    &pv_concat[bh * concat_row..bh * concat_row + concat_row],
                    "capacity {capacity}: present_value valid region diverged"
                );
            }
        }
    }

    fn dims_of(case: Case) -> Dims {
        Dims {
            batch: case.batch,
            q_heads: case.q_heads,
            kv_heads: case.kv_heads,
            q_seq: case.q_seq,
            current_seq: case.current_seq,
            past_seq: case.past_seq,
            total_seq: case.past_seq + case.current_seq,
            head_size: case.head_size,
            index_heads: case.index_heads,
            selected_width: case.width,
            cache_seq: case.past_seq + case.current_seq,
            capacity_mode: false,
        }
    }

    /// Locks in the S2 numerical contract: attention over a fixed-capacity
    /// ("in-place") present that aliases `past` at `capacity` and writes the
    /// current token at its absolute position is byte-identical to attention
    /// over the growing concat present, for every head layout / bias / index
    /// pattern. The device capacity-present kernel must match this exactly.
    #[test]
    fn capacity_present_attention_matches_concat_byte_for_byte() {
        let cases = [
            Case {
                batch: 1,
                q_heads: 1,
                kv_heads: 1,
                q_seq: 1,
                current_seq: 1,
                past_seq: 3,
                head_size: 2,
                index_heads: 1,
                width: 3,
                scale: 0.5,
            },
            Case {
                batch: 2,
                q_heads: 4,
                kv_heads: 2,
                q_seq: 2,
                current_seq: 3,
                past_seq: 5,
                head_size: 3,
                index_heads: 4,
                width: 4,
                scale: 0.25,
            },
        ];
        for case in cases {
            let dims = dims_of(case);
            let sqrt_scale = case.scale.sqrt();
            let total = dims.total_seq;
            let q = sequence(
                case.batch * case.q_heads * case.q_seq * case.head_size,
                0.25,
            );
            let past_k = sequence(
                case.batch * case.kv_heads * case.past_seq * case.head_size,
                -0.5,
            );
            let past_v = sequence(
                case.batch * case.kv_heads * case.past_seq * case.head_size,
                0.75,
            );
            let current_k = sequence(
                case.batch * case.kv_heads * case.current_seq * case.head_size,
                0.125,
            );
            let current_v = sequence(
                case.batch * case.kv_heads * case.current_seq * case.head_size,
                -1.25,
            );
            // Ascending, unique, causal indices per (batch, index-head, query).
            let mut indices = vec![-1i64; case.batch * case.index_heads * case.q_seq * case.width];
            for b in 0..case.batch {
                for ih in 0..case.index_heads {
                    for qi in 0..case.q_seq {
                        let row = ((b * case.index_heads + ih) * case.q_seq + qi) * case.width;
                        let limit = (case.past_seq + qi + 1).min(total);
                        let count = case.width.min(limit);
                        for k in 0..count {
                            indices[row + k] = (limit - count + k) as i64;
                        }
                    }
                }
            }

            for bias in [None, Some(())] {
                let bias = bias.map(|()| {
                    let data = (0..case.batch * case.q_heads * case.q_seq * total)
                        .map(|i| ((i % 5) as f32) * 0.1 - 0.2)
                        .collect::<Vec<_>>();
                    Bias {
                        data,
                        shape: vec![case.batch, case.q_heads, case.q_seq, total],
                    }
                });

                let concat_k = concatenate_cache(Some(&past_k), &current_k, dims);
                let concat_v = concatenate_cache(Some(&past_v), &current_v, dims);
                let mut expected =
                    vec![0.0f32; case.batch * case.q_heads * case.q_seq * case.head_size];
                attend_selected(
                    &mut expected,
                    &concat_k,
                    &concat_v,
                    total,
                    &q,
                    &indices,
                    bias.as_ref(),
                    dims,
                    sqrt_scale,
                );

                for capacity in [total, total + 1, total + 7] {
                    let cap_k = capacity_present(Some(&past_k), &current_k, dims, capacity, 987.0);
                    let cap_v = capacity_present(Some(&past_v), &current_v, dims, capacity, -654.0);
                    let mut actual =
                        vec![0.0f32; case.batch * case.q_heads * case.q_seq * case.head_size];
                    attend_selected(
                        &mut actual,
                        &cap_k,
                        &cap_v,
                        capacity,
                        &q,
                        &indices,
                        bias.as_ref(),
                        dims,
                        sqrt_scale,
                    );
                    assert_eq!(
                        actual, expected,
                        "capacity {capacity} diverged from concat for case {case:?}"
                    );
                }
            }
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
