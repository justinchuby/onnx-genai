//! Tensor `Value` manipulation and graph I/O name-classification helpers
//! shared across the decode submodules.
//!
//! Pure code motion from `decode.rs`.

use super::*;

pub(super) fn zero_state_value(info: &TensorInfo) -> anyhow::Result<Value> {
    let shape = concrete_fixed_state_shape(info)?;
    let element_count = fixed_state_element_count(info)?;
    match info.dtype {
        DataType::Float32 => {
            Value::from_vec_f32(fallible_zeroed(element_count, 0.0, info)?, &shape)
        }
        DataType::Float16 => {
            Value::from_vec_f16_bits(fallible_zeroed(element_count, 0, info)?, &shape)
        }
        DataType::BFloat16 => {
            Value::from_vec_bf16_bits(fallible_zeroed(element_count, 0, info)?, &shape)
        }
        DataType::Int64 => Value::from_vec_i64(fallible_zeroed(element_count, 0, info)?, &shape),
        dtype => anyhow::bail!(
            "state input '{}' has unsupported zero-initialization dtype {:?}",
            info.name,
            dtype
        ),
    }
    .with_context(|| format!("failed to zero-initialize loop-state input '{}'", info.name))
}

/// Concrete shape for zero-initializing a loop-carried fixed-state input.
///
/// The leading (batch) axis is commonly symbolic in exported decoder graphs;
/// single-sequence decode resolves it to `1`, mirroring the empty-KV convention
/// in [`empty_past_value`]. Every non-leading dimension must be concrete and
/// positive — a symbolic inner extent (e.g. a state channel count) cannot be
/// zero-initialized without guessing model DATA, so it is refused.
pub(super) fn concrete_fixed_state_shape(info: &TensorInfo) -> anyhow::Result<Vec<i64>> {
    if info.shape.is_empty() {
        anyhow::bail!(
            "state input '{}' has scalar shape; loop-carried state requires at least a batch axis",
            info.name
        );
    }
    info.shape
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            if axis == 0 && dim <= 0 {
                Ok(1)
            } else if dim > 0 {
                Ok(dim)
            } else {
                anyhow::bail!(
                    "state input '{}' has dynamic or invalid non-batch dimension {axis} in shape {:?}; \
                     zero initialization requires every non-batch fixed-state dimension to be concrete and positive",
                    info.name,
                    info.shape
                )
            }
        })
        .collect()
}

fn fixed_state_element_count(info: &TensorInfo) -> anyhow::Result<usize> {
    concrete_fixed_state_shape(info)?
        .iter()
        .try_fold(1_usize, |count, &dimension| {
            let dimension = usize::try_from(dimension).with_context(|| {
                format!(
                    "state input '{}' has non-concrete dimension {dimension}",
                    info.name
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
    session: &dyn GraphIo,
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
        // General, dtype-agnostic fallback: deep-copy the raw little-endian
        // element bytes for every remaining POD dtype (Bool, Int32, Int8,
        // Uint8/16/32/64, Int16, Float8*) instead of bailing per dtype. This is
        // what unblocks e.g. gemma-3n's Bool audio mask. `as_raw_bytes` returns a
        // precise error for a device-resident tensor rather than misreading a
        // device pointer as host memory, so a stray device value fails loudly
        // instead of corrupting silently.
        dtype => Value::from_raw_bytes(value.as_raw_bytes()?.to_vec(), value.shape(), dtype)
            .map_err(|e| anyhow::anyhow!("Failed to clone {dtype:?} ORT value: {e}")),
    }
}

/// The single axis of `shape` whose extent is `extent`, or `None` when no axis
/// or more than one axis matches. Only reachable from
/// [`DecodeState::truncate_past`](super::state::DecodeState::truncate_past),
/// which has no production caller while rollback is unwired.
#[cfg_attr(not(test), allow(dead_code))]
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
    use super::*;

    /// `clone_value` deep-copies a cached decode input. Before generalization it
    /// bailed with "unsupported cached ORT value dtype: Bool" (and likewise for
    /// Int32/Uint8/...), which blocked gemma-3n's Bool audio mask. It must now
    /// round-trip dtype + shape + raw bytes for every POD dtype.
    fn assert_clone_value_round_trips(bytes: Vec<u8>, shape: &[i64], dtype: DataType) {
        let original = Value::from_raw_bytes(bytes.clone(), shape, dtype)
            .unwrap_or_else(|e| panic!("build {dtype:?} cached input: {e}"));
        let cloned =
            clone_value(&original).unwrap_or_else(|e| panic!("clone_value {dtype:?}: {e}"));

        assert_eq!(cloned.dtype(), dtype, "{dtype:?}: dtype must round-trip");
        assert_eq!(cloned.shape(), shape, "{dtype:?}: shape must round-trip");
        assert_eq!(
            cloned.to_raw_bytes().expect("cloned bytes"),
            bytes,
            "{dtype:?}: raw bytes must round-trip identically"
        );
    }

    #[test]
    fn clone_value_round_trips_a_bool_cached_input() {
        // The gemma-3n audio-mask case: a Bool cached input that used to bail.
        assert_clone_value_round_trips(vec![1, 0, 1, 1, 0], &[5], DataType::Bool);
    }

    #[test]
    fn clone_value_round_trips_int32() {
        let bytes: Vec<u8> = [10i32, -20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_clone_value_round_trips(bytes, &[3], DataType::Int32);
    }

    #[test]
    fn clone_value_round_trips_an_empty_bool_input() {
        assert_clone_value_round_trips(Vec::new(), &[0], DataType::Bool);
    }

    #[test]
    fn clone_value_round_trips_a_multidim_bool_mask() {
        assert_clone_value_round_trips(vec![1, 0, 0, 1, 1, 0], &[2, 3], DataType::Bool);
    }

    #[test]
    fn clone_value_still_round_trips_the_typed_i64_fast_path() {
        // The generalization must not regress the existing typed dtypes.
        let original = Value::from_slice_i64(&[9, 8, 7], &[3]).expect("i64 cached input");
        let cloned = clone_value(&original).expect("clone i64");
        assert_eq!(cloned.dtype(), DataType::Int64);
        assert_eq!(cloned.to_vec_i64().expect("i64 out"), vec![9, 8, 7]);
    }
}

/// `clone_value` is the frame that aborted the `CLI ORT` job: it read a pooled
/// tensor back as its element type every decode step, and the pooled tensor had
/// been built from an empty `Vec<u8>` whose dangling pointer is 1-byte aligned.
///
/// The dtype matters here in a way that is easy to miss. `clone_value`'s sibling
/// path builds empty tensors from `Vec<u16>` / `Vec<f32>`, whose dangling
/// pointers are already 2- and 4-byte aligned, so those cases never aborted.
/// Only the `Vec<u8>`-backed constructor produced a pointer too weakly aligned
/// for its element type, which is why this covers every dtype `clone_value`
/// dispatches on rather than only the one that failed.
#[cfg(test)]
mod empty_value_clone_tests {
    use super::clone_value;
    use onnx_genai_ort::{DataType, Value};

    #[test]
    fn cloning_an_empty_raw_bytes_tensor_succeeds_for_every_dispatched_dtype() {
        for dtype in [
            DataType::Float32,
            DataType::Float16,
            DataType::BFloat16,
            DataType::Int64,
            DataType::Int32,
            DataType::Bool,
            DataType::Uint8,
        ] {
            let value = Value::from_raw_bytes(Vec::new(), &[0, 8], dtype)
                .unwrap_or_else(|e| panic!("build an empty {dtype:?} tensor: {e}"));
            let cloned = clone_value(&value)
                .unwrap_or_else(|e| panic!("clone an empty {dtype:?} tensor: {e:#}"));
            assert_eq!(cloned.dtype(), dtype);
            assert_eq!(cloned.shape(), &[0, 8], "{dtype:?}");
            assert_eq!(cloned.numel(), 0, "{dtype:?}");
        }
    }
}
