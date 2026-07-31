//! Tensor `Value` manipulation and graph I/O name-classification helpers
//! shared across the decode submodules.
//!
//! Pure code motion from `decode.rs`.

use super::*;

pub(super) fn zero_state_value(info: &TensorInfo) -> anyhow::Result<Value> {
    let element_count = fixed_state_element_count(info)?;
    match info.dtype {
        DataType::Float32 => {
            Value::from_vec_f32(fallible_zeroed(element_count, 0.0, info)?, &info.shape)
        }
        DataType::Float16 => {
            Value::from_vec_f16_bits(fallible_zeroed(element_count, 0, info)?, &info.shape)
        }
        DataType::BFloat16 => {
            Value::from_vec_bf16_bits(fallible_zeroed(element_count, 0, info)?, &info.shape)
        }
        DataType::Int64 => {
            Value::from_vec_i64(fallible_zeroed(element_count, 0, info)?, &info.shape)
        }
        dtype => anyhow::bail!(
            "state input '{}' has unsupported zero-initialization dtype {:?}",
            info.name,
            dtype
        ),
    }
    .with_context(|| format!("failed to zero-initialize loop-state input '{}'", info.name))
}

fn fixed_state_element_count(info: &TensorInfo) -> anyhow::Result<usize> {
    info.shape.iter().try_fold(1_usize, |count, dimension| {
        let dimension = usize::try_from(*dimension).with_context(|| {
            format!(
                "state input '{}' has non-concrete dimension {}",
                info.name, dimension
            )
        })?;
        count
            .checked_mul(dimension)
            .context("state tensor element count overflow")
    })
}

fn fixed_state_bytes(info: &TensorInfo) -> anyhow::Result<u64> {
    let elements = u64::try_from(fixed_state_element_count(info)?)
        .context("state tensor element count exceeds u64")?;
    elements
        .checked_mul(info.dtype.size_of() as u64)
        .context("state tensor byte size overflow")
}

pub(super) fn validate_fixed_state_budget(
    session: &Session,
    state_pairs: &[(String, String)],
    budget_bytes: u64,
) -> anyhow::Result<()> {
    let mut required_bytes = 0_u64;
    for (input, _) in state_pairs {
        let info = session
            .inputs()
            .iter()
            .find(|info| info.name == *input)
            .with_context(|| format!("declared fixed-state input '{input}' disappeared"))?;
        required_bytes = required_bytes
            .checked_add(fixed_state_bytes(info)?)
            .context("total fixed-state allocation size overflow")?;
    }
    if required_bytes > budget_bytes {
        anyhow::bail!(
            "decoder fixed-state initialization requires {required_bytes} bytes, but the configured host RAM admission budget is {budget_bytes} bytes; reduce declared state dimensions or raise serving.memory.limits.host_ram_limit"
        );
    }
    Ok(())
}

fn fallible_zeroed<T: Clone>(
    element_count: usize,
    zero: T,
    info: &TensorInfo,
) -> anyhow::Result<Vec<T>> {
    let bytes = element_count
        .checked_mul(std::mem::size_of::<T>())
        .context("state tensor allocation byte size overflow")?;
    let mut data = Vec::new();
    data.try_reserve_exact(element_count).map_err(|error| {
        anyhow::anyhow!(
            "failed to allocate {bytes} bytes for loop-state input '{}': {error}",
            info.name
        )
    })?;
    data.resize(element_count, zero);
    Ok(data)
}

pub(super) fn ensure_i64(info: &TensorInfo) -> anyhow::Result<()> {
    if info.dtype != DataType::Int64 {
        anyhow::bail!("input '{}' must be Int64, got {:?}", info.name, info.dtype);
    }
    Ok(())
}

pub(super) fn empty_past_value(info: &TensorInfo) -> anyhow::Result<Value> {
    if !matches!(
        info.dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        anyhow::bail!(
            "KV input '{}' must be Float32, Float16, or BFloat16, got {:?}",
            info.name,
            info.dtype
        );
    }
    if info.shape.len() < 3 {
        anyhow::bail!(
            "KV input '{}' has unsupported shape {:?}",
            info.name,
            info.shape
        );
    }
    let seq_axis = info.shape.len() - 2;
    let mut shape = Vec::with_capacity(info.shape.len());
    for (axis, &dim) in info.shape.iter().enumerate() {
        let value = if axis == 0 {
            1
        } else if axis == seq_axis {
            0
        } else if dim > 0 {
            dim
        } else {
            anyhow::bail!(
                "cannot infer static dimension {} for empty KV input '{}' shape {:?}",
                axis,
                info.name,
                info.shape
            );
        };
        shape.push(value);
    }
    match info.dtype {
        DataType::Float32 => Value::from_slice_f32(&[], &shape),
        DataType::Float16 => Value::from_vec_f16_bits(Vec::new(), &shape),
        DataType::BFloat16 => Value::from_vec_bf16_bits(Vec::new(), &shape),
        _ => unreachable!("dtype checked above"),
    }
    .map_err(|e| anyhow::anyhow!("Failed to create empty KV input '{}': {}", info.name, e))
}

pub(crate) fn clone_value(value: &Value) -> anyhow::Result<Value> {
    // Per-step-invariant, read-only inputs (e.g. an encoder-decoder's static
    // cross-attention KV) are bound as no-copy aliases over a shared owner.
    // Re-alias them in O(1) instead of deep-copying the underlying buffer.
    if let Some(aliased) = value.try_alias_clone() {
        return aliased.map_err(|e| anyhow::anyhow!("Failed to alias-clone ORT value: {e}"));
    }
    match value.dtype() {
        DataType::Float32 => Value::from_slice_f32(&value.to_vec_f32()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone Float32 ORT value: {e}")),
        DataType::Float16 => Value::from_vec_f16_bits(value.to_vec_f16_bits()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone Float16 ORT value: {e}")),
        DataType::BFloat16 => Value::from_vec_bf16_bits(value.to_vec_bf16_bits()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone BFloat16 ORT value: {e}")),
        DataType::Int64 => Value::from_slice_i64(&value.to_vec_i64()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone Int64 ORT value: {e}")),
        // Generic byte-exact fallback for host-resident dtypes without a typed
        // accessor above (e.g. `Bool` modality masks, `Int32`). `to_raw_bytes`
        // copies the row-major allocation and `from_raw_bytes` reconstructs it
        // for the same shape/dtype, so the clone is bit-identical.
        dtype => Value::from_raw_bytes(value.to_raw_bytes()?, value.shape(), dtype)
            .map_err(|e| anyhow::anyhow!("Failed to clone {dtype:?} ORT value: {e}")),
    }
}

/// The single axis of `shape` whose extent is `extent`, or `None` when no axis
/// or more than one axis matches.
pub(super) fn sole_axis_with_extent(shape: &[i64], extent: usize) -> Option<usize> {
    let extent = i64::try_from(extent).ok()?;
    let mut found = None;
    for (axis, &dim) in shape.iter().enumerate() {
        if dim == extent {
            if found.is_some() {
                return None;
            }
            found = Some(axis);
        }
    }
    found
}

pub(super) fn slice_value_axis(
    value: &Value,
    axis: usize,
    start: usize,
    len: usize,
) -> anyhow::Result<Value> {
    let shape = value.shape();
    let axis_len = *shape.get(axis).context("KV slice axis is out of bounds")?;
    let axis_len = usize::try_from(axis_len).context("KV slice axis length is negative")?;
    if start > axis_len || len > axis_len - start {
        anyhow::bail!(
            "KV slice [{start}..{}) exceeds axis length {axis_len}",
            start + len
        );
    }
    let mut output_shape = shape.to_vec();
    output_shape[axis] = i64::try_from(len).context("KV slice length exceeds i64")?;

    fn copy_axis_slice<T: Copy>(
        input: &[T],
        shape: &[i64],
        axis: usize,
        start: usize,
        len: usize,
    ) -> Vec<T> {
        let inner = shape[axis + 1..]
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>();
        let outer = shape[..axis]
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>();
        let axis_len = shape[axis] as usize;
        let mut output = Vec::with_capacity(outer * len * inner);
        for outer_idx in 0..outer {
            let base = outer_idx * axis_len * inner + start * inner;
            output.extend_from_slice(&input[base..base + len * inner]);
        }
        output
    }

    match value.dtype() {
        DataType::Float32 => Value::from_vec_f32(
            copy_axis_slice(&value.to_vec_f32()?, shape, axis, start, len),
            &output_shape,
        ),
        DataType::Float16 => Value::from_vec_f16_bits(
            copy_axis_slice(&value.to_vec_f16_bits()?, shape, axis, start, len),
            &output_shape,
        ),
        DataType::BFloat16 => Value::from_vec_bf16_bits(
            copy_axis_slice(&value.to_vec_bf16_bits()?, shape, axis, start, len),
            &output_shape,
        ),
        dtype => anyhow::bail!("cannot slice cached KV tensor with dtype {dtype:?}"),
    }
    .map_err(|error| anyhow::anyhow!("Failed to slice cached KV tensor: {error}"))
}

/// Concatenate two KV tensors of identical shape (except along `axis`) into a
/// single tensor along that axis. Used to splice the pinned attention-sink rows
/// in front of the sliding-window rows.
pub(super) fn concat_value_axis(
    first: &Value,
    second: &Value,
    axis: usize,
) -> anyhow::Result<Value> {
    let first_shape = first.shape();
    let second_shape = second.shape();
    if first_shape.len() != second_shape.len() {
        anyhow::bail!("cannot concatenate KV tensors of differing rank");
    }
    for (dim, (a, b)) in first_shape.iter().zip(second_shape.iter()).enumerate() {
        if dim != axis && a != b {
            anyhow::bail!("cannot concatenate KV tensors: mismatched shape on axis {dim}");
        }
    }
    let mut output_shape = first_shape.to_vec();
    output_shape[axis] = first_shape[axis] + second_shape[axis];

    fn interleave<T: Copy>(
        first: &[T],
        second: &[T],
        shape_a: &[i64],
        shape_b: &[i64],
        axis: usize,
    ) -> Vec<T> {
        let inner = shape_a[axis + 1..]
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>();
        let outer = shape_a[..axis]
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>();
        let a_axis = shape_a[axis] as usize;
        let b_axis = shape_b[axis] as usize;
        let mut output = Vec::with_capacity(outer * (a_axis + b_axis) * inner);
        for outer_idx in 0..outer {
            let a_base = outer_idx * a_axis * inner;
            output.extend_from_slice(&first[a_base..a_base + a_axis * inner]);
            let b_base = outer_idx * b_axis * inner;
            output.extend_from_slice(&second[b_base..b_base + b_axis * inner]);
        }
        output
    }

    if first.dtype() != second.dtype() {
        anyhow::bail!("cannot concatenate KV tensors of differing dtype");
    }
    match first.dtype() {
        DataType::Float32 => Value::from_vec_f32(
            interleave(
                &first.to_vec_f32()?,
                &second.to_vec_f32()?,
                first_shape,
                second_shape,
                axis,
            ),
            &output_shape,
        ),
        DataType::Float16 => Value::from_vec_f16_bits(
            interleave(
                &first.to_vec_f16_bits()?,
                &second.to_vec_f16_bits()?,
                first_shape,
                second_shape,
                axis,
            ),
            &output_shape,
        ),
        DataType::BFloat16 => Value::from_vec_bf16_bits(
            interleave(
                &first.to_vec_bf16_bits()?,
                &second.to_vec_bf16_bits()?,
                first_shape,
                second_shape,
                axis,
            ),
            &output_shape,
        ),
        dtype => anyhow::bail!("cannot concatenate cached KV tensor with dtype {dtype:?}"),
    }
    .map_err(|error| anyhow::anyhow!("Failed to concatenate cached KV tensor: {error}"))
}

pub(crate) fn is_kv_input(name: &str) -> bool {
    name_contains_past_key_value(name)
}

pub(crate) fn is_gather_out_of_bounds(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("gather")
        && (lower.contains("indices element out of data bounds")
            || lower.contains("idx=") && lower.contains("out of"))
}

#[cfg(test)]
mod clone_value_tests {
    use super::clone_value;
    use onnx_genai_ort::{DataType, Value};

    #[test]
    fn clone_value_round_trips_bool_via_generic_fallback() {
        // `Bool` modality masks (e.g. gemma-3n audio `input_features_mask`) have
        // no typed clone accessor; the generic raw-bytes fallback must clone them
        // bit-exactly rather than erroring `unsupported cached ORT value dtype`.
        let bytes = vec![1u8, 0, 1, 1];
        let value = Value::from_raw_bytes(bytes.clone(), &[1, 4], DataType::Bool).unwrap();
        let cloned = clone_value(&value).expect("Bool clone must succeed via generic fallback");
        assert_eq!(cloned.dtype(), DataType::Bool);
        assert_eq!(cloned.shape(), value.shape());
        assert_eq!(cloned.to_raw_bytes().unwrap(), bytes);
    }
}
