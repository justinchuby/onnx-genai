use super::*;

pub(super) fn to_f32_logits(logits: &Value) -> Result<Value> {
    let shape = logits.shape().to_vec();
    if logits.dtype() == DataType::Float32 {
        return Value::from_vec_f32(logits.to_vec_f32()?, &shape);
    }
    Value::from_vec_f32(logits.to_vec_f32_lossy()?, &shape)
}

/// Gather selected batch rows of a `[B, S, vocab]` Float32 logits tensor into a
/// compact `[rows.len(), S, vocab]` tensor, preserving the given row order.
pub(super) fn gather_logits_rows(logits: &Value, rows: &[usize]) -> Result<Value> {
    if logits.dtype() != DataType::Float32 || logits.shape().len() != 3 {
        return Err(OrtError::InvalidArgument(format!(
            "expected Float32 logits [B, S, V], got {:?} {:?}",
            logits.dtype(),
            logits.shape()
        )));
    }
    let shape = logits.shape();
    let batch = shape[0] as usize;
    let seq_len = shape[1] as usize;
    let vocab = shape[2] as usize;
    let data = logits.to_vec_f32()?;
    let row_stride = seq_len * vocab;
    let mut gathered = Vec::with_capacity(rows.len() * row_stride);
    for &row in rows {
        if row >= batch {
            return Err(OrtError::InvalidArgument(format!(
                "gather row {row} out of range for batch {batch}"
            )));
        }
        let start = row * row_stride;
        gathered.extend_from_slice(&data[start..start + row_stride]);
    }
    Value::from_vec_f32(gathered, &[rows.len() as i64, seq_len as i64, vocab as i64])
}

pub(super) fn empty_past_value(info: &TensorInfo) -> Result<Value> {
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
            return Err(OrtError::InvalidArgument(format!(
                "cannot infer static dimension {axis} for empty KV input '{}'",
                info.name
            )));
        };
        shape.push(value);
    }
    Value::empty(&shape, info.dtype)
}

/// Copy a device-resident `logits [1, 1, vocab]` row back into a host CPU
/// [`Value`] of the same dtype, for the non-greedy path that still consumes the
/// full vocabulary. This mirrors ORT's implicit device->host logits copy that
/// the on-device path otherwise skips.
#[cfg(feature = "cuda")]
pub(super) fn device_logits_to_host_value(
    device_sampler: &dyn DeviceSampler,
    dtype: DataType,
    dev_ptr: usize,
    vocab: usize,
) -> Result<Value> {
    let host = Value::empty(&[1, 1, vocab as i64], dtype)?;
    let nbytes = vocab
        .checked_mul(dtype.size_of())
        .ok_or_else(|| OrtError::InvalidArgument("logits byte size overflow".into()))?;
    let base = host.data_ptr_addr()? as *mut u8;
    // SAFETY: `host` is a freshly-allocated CPU tensor holding exactly `nbytes`
    // bytes; the slice aliases only that storage for the duration of the copy.
    let dst = unsafe { std::slice::from_raw_parts_mut(base, nbytes) };
    device_sampler.copy_row_to_host(dtype, dev_ptr, vocab, dst)?;
    Ok(host)
}

/// Copy an OrtValue's tensor data onto host-owned Rust buffers, producing a
/// new, session-independent CPU [`Value`]. Used to hand a KV cache between two
/// [`DecodeSession`]s (e.g. Metal-EP prefill → CPU-EP decode).
pub(super) fn clone_value_to_owned(value: &Value) -> Result<Value> {
    let shape = value.shape().to_vec();
    match value.dtype() {
        DataType::Float32 => Value::from_vec_f32(value.to_vec_f32()?, &shape),
        DataType::Float16 => Value::from_vec_f16_bits(value.to_vec_f16_bits()?, &shape),
        DataType::BFloat16 => Value::from_vec_bf16_bits(value.to_vec_bf16_bits()?, &shape),
        dtype => Err(OrtError::InvalidArgument(format!(
            "cannot export/clone KV tensor with dtype {dtype:?}"
        ))),
    }
}

pub(super) fn allocate_static_cache_buffers(
    batch_size: i64,
    pairs: &[StaticCachePair],
) -> Result<Vec<StaticCacheBuffer>> {
    if batch_size <= 0 {
        return Err(OrtError::InvalidArgument(format!(
            "batch_size must be positive, got {batch_size}"
        )));
    }
    let mut buffers = Vec::with_capacity(pairs.len() * 2);
    for pair in pairs {
        for (input, output) in [
            (&pair.key_input, &pair.key_output),
            (&pair.value_input, &pair.value_output),
        ] {
            let mut shape = input.shape.clone();
            shape[0] = batch_size;
            buffers.push(StaticCacheBuffer {
                input_name: input.name.clone(),
                output_name: output.clone(),
                current: Arc::new(zeroed_value(&shape, input.dtype)?),
                alternate: None,
            });
        }
    }
    Ok(buffers)
}

pub(super) fn zeroed_value(shape: &[i64], dtype: DataType) -> Result<Value> {
    let numel = shape.iter().try_fold(1usize, |acc, &dim| {
        if dim < 0 {
            return Err(OrtError::InvalidArgument(format!(
                "cannot allocate tensor with dynamic shape {shape:?}"
            )));
        }
        acc.checked_mul(dim as usize)
            .ok_or_else(|| OrtError::InvalidArgument(format!("tensor shape too large: {shape:?}")))
    })?;
    match dtype {
        DataType::Float32 => Value::from_vec_f32(vec![0.0; numel], shape),
        DataType::Float16 => Value::from_vec_f16_bits(vec![0; numel], shape),
        DataType::BFloat16 => Value::from_vec_bf16_bits(vec![0; numel], shape),
        // Every other dtype gets zeroed bytes. Zero is representable in every
        // ONNX tensor element type, including FP8, so allocating an empty cache
        // needs no per-dtype host representation. Whether a kernel can then
        // compute on those bytes is an execution-provider question, answered
        // when the session binds them; refusing to allocate here would report a
        // missing kernel as a missing allocator.
        dtype => Value::from_raw_bytes(vec![0u8; numel * dtype.size_of()], shape, dtype),
    }
}
