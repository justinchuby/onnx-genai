use super::*;

pub(crate) fn extract_logits(tensor: &Tensor) -> anyhow::Result<Vec<Vec<f32>>> {
    let values = tensor_to_f32(tensor)?;
    match tensor.shape.as_slice() {
        [vocab] if *vocab > 0 => Ok(vec![values]),
        [seq, vocab] if *seq > 0 && *vocab > 0 => Ok(values
            .chunks(*vocab)
            .take(*seq)
            .map(<[f32]>::to_vec)
            .collect()),
        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => Ok(values
            .chunks(*vocab)
            .take(*seq)
            .map(<[f32]>::to_vec)
            .collect()),
        shape => bail!("unsupported logits tensor shape: {shape:?}"),
    }
}

/// Extract per-*row* logits from a batched single-token forward whose logits
/// tensor is `[batch, 1, vocab]` (batch axis leading, query-seq collapsed to 1),
/// returning one `[vocab]` vector per batch row. Unlike [`extract_logits`],
/// which interprets a rank-3 tensor as `[1, seq, vocab]` and keeps the query-seq
/// rows of a single sequence, this keeps every batch row — the stage 2a
/// batch-N fused-forward result shape.
pub(crate) fn extract_batch_row_logits(
    tensor: &Tensor,
    batch: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let values = tensor_to_f32(tensor)?;
    let vocab = match tensor.shape.as_slice() {
        [b, seq, vocab] if *b == batch && *seq == 1 && *vocab > 0 => *vocab,
        [b, vocab] if *b == batch && *vocab > 0 => *vocab,
        shape => bail!(
            "unsupported batched logits tensor shape {shape:?} for batch {batch}; expected [{batch}, 1, vocab] or [{batch}, vocab]"
        ),
    };
    if values.len() < batch * vocab {
        bail!(
            "batched logits tensor {:?} yielded {} values, need {}",
            tensor.shape,
            values.len(),
            batch * vocab
        );
    }
    Ok(values
        .chunks(vocab)
        .take(batch)
        .map(<[f32]>::to_vec)
        .collect())
}

pub(crate) fn argmax_logits_tensor(tensor: &Tensor) -> anyhow::Result<TokenId> {
    let (value_count, vocab) = match tensor.shape.as_slice() {
        [vocab] if *vocab > 0 => (*vocab, *vocab),
        [seq, vocab] if *seq > 0 && *vocab > 0 => (seq * vocab, *vocab),
        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => (seq * vocab, *vocab),
        shape => bail!("unsupported logits tensor shape: {shape:?}"),
    };
    let row_start = value_count - vocab;
    let mut best = f32::NEG_INFINITY;
    let mut best_index = 0;
    let mut seen = 0;
    let mut visit = |value: f32| -> anyhow::Result<()> {
        if !value.is_finite() {
            bail!("native decoder produced non-finite logits");
        }
        if seen >= row_start && value > best {
            best = value;
            best_index = seen - row_start;
        }
        seen += 1;
        Ok(())
    };
    match tensor.dtype {
        DataType::Float32 => {
            if let Some(values) = tensor.try_as_slice_f32() {
                if values.len() < value_count {
                    bail!(
                        "native logits tensor shape {:?} requires {value_count} values, but only {} were readable",
                        tensor.shape,
                        values.len()
                    );
                }
                let values = &values[..value_count];
                if values.iter().any(|value| !value.is_finite()) {
                    bail!("native decoder produced non-finite logits");
                }
                return Ok(sample_greedy(&values[row_start..]) as TokenId);
            }
            for bytes in tensor
                .as_bytes()
                .as_chunks::<4>()
                .0
                .iter()
                .take(value_count)
            {
                visit(f32::from_le_bytes(*bytes))?;
            }
        }
        DataType::Float16 => {
            if let Some(bits) = tensor.try_as_slice_u16() {
                use half::slice::HalfBitsSliceExt;
                let halves: &[half::f16] = bits.reinterpret_cast();
                return argmax_finite_half_values(halves, value_count, row_start)
                    .map(|index| index as TokenId);
            }
            for bytes in tensor
                .as_bytes()
                .as_chunks::<2>()
                .0
                .iter()
                .take(value_count)
            {
                visit(f16_to_f32(u16::from_le_bytes(*bytes)))?;
            }
        }
        DataType::BFloat16 => {
            if let Some(bits) = tensor.try_as_slice_u16() {
                use half::slice::HalfBitsSliceExt;
                let halves: &[half::bf16] = bits.reinterpret_cast();
                return argmax_finite_half_values(halves, value_count, row_start)
                    .map(|index| index as TokenId);
            }
            for bytes in tensor
                .as_bytes()
                .as_chunks::<2>()
                .0
                .iter()
                .take(value_count)
            {
                visit(f32::from_bits(u32::from(u16::from_le_bytes(*bytes)) << 16))?;
            }
        }
        dtype => bail!("native logits must be Float32, Float16, or BFloat16, got {dtype:?}"),
    }
    if seen != value_count {
        bail!(
            "native logits tensor shape {:?} requires {value_count} values, but only {seen} were readable",
            tensor.shape
        );
    }
    Ok(best_index as TokenId)
}

fn argmax_finite_half_values<H>(
    halves: &[H],
    value_count: usize,
    row_start: usize,
) -> anyhow::Result<usize>
where
    [H]: half::slice::HalfFloatSliceExt,
{
    use half::slice::HalfFloatSliceExt;

    if halves.len() < value_count {
        bail!(
            "native logits tensor requires {value_count} values, but only {} were readable",
            halves.len()
        );
    }
    const CHUNK: usize = 4096;
    let mut scratch = [0.0f32; CHUNK];
    let mut best = f32::NEG_INFINITY;
    let mut best_index = 0;
    for (chunk_index, chunk) in halves[..value_count].chunks(CHUNK).enumerate() {
        let widened = &mut scratch[..chunk.len()];
        chunk.convert_to_f32_slice(widened);
        if widened.iter().any(|value| !value.is_finite()) {
            bail!("native decoder produced non-finite logits");
        }
        let chunk_start = chunk_index * CHUNK;
        let candidate_start = row_start.saturating_sub(chunk_start).min(chunk.len());
        if candidate_start == chunk.len() {
            continue;
        }
        let candidates = &widened[candidate_start..];
        let chunk_max = candidates.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if chunk_max > best {
            best = chunk_max;
            best_index = chunk_start
                + candidate_start
                + candidates
                    .iter()
                    .position(|&value| value == chunk_max)
                    .expect("finite non-empty chunk has a maximum");
        }
    }
    Ok(best_index - row_start)
}

pub(crate) fn extract_last_row(tensor: &Tensor) -> anyhow::Result<Vec<f32>> {
    let width = *tensor
        .shape
        .last()
        .context("tensor has no final feature dimension")?;
    if width == 0 {
        bail!("tensor final feature dimension must be positive");
    }
    let values = tensor_to_f32(tensor)?;
    if values.len() < width || !values.len().is_multiple_of(width) {
        bail!(
            "tensor element count {} is not a positive multiple of final feature width {width}",
            values.len()
        );
    }
    Ok(values[values.len() - width..].to_vec())
}

fn tensor_to_f32(tensor: &Tensor) -> anyhow::Result<Vec<f32>> {
    match tensor.dtype {
        DataType::Float32 => Ok(tensor.to_vec_f32()),
        DataType::Float16 => Ok(tensor
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect()),
        DataType::BFloat16 => Ok(tensor
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16))
            .collect()),
        dtype => bail!("native logits must be Float32, Float16, or BFloat16, got {dtype:?}"),
    }
}

pub(crate) fn tensor_from_f32_as(
    dtype: DataType,
    shape: &[usize],
    values: &[f32],
) -> anyhow::Result<Tensor> {
    match dtype {
        DataType::Float32 => Ok(Tensor::from_f32(shape, values)?),
        DataType::Float16 | DataType::BFloat16 => {
            let bytes = f32_slice_to_dtype_bytes(dtype, values)?;
            Ok(Tensor::from_raw(dtype, shape.to_vec(), &bytes)?)
        }
        other => bail!(
            "native embeddings input must be Float32, Float16, or BFloat16, got {other:?}; fix io.inputs_embeds_input or export a floating tensor"
        ),
    }
}

/// Encode `values` as the little-endian storage bytes for `dtype`, using the
/// `half` crate for the 16-bit narrowing so the result is bit-identical to the
/// ORT KV-inject path (`onnx_genai_ort::Value::from_f32_slice_as`,
/// `crates/onnx-genai-ort/src/value.rs`). Sharing this one encoder keeps the
/// native embedding-input path (`tensor_from_f32_as`) and the device paged
/// present-KV seed (`DecodeCudaState::seed_prefix`, GAP-3 Inc-D.1) byte-for-byte
/// aligned with ORT — the whole basis of the paged-KV byte-equality oracle.
pub(crate) fn f32_slice_to_dtype_bytes(dtype: DataType, values: &[f32]) -> anyhow::Result<Vec<u8>> {
    match dtype {
        DataType::Float32 => Ok(values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()),
        DataType::Float16 => Ok(values
            .iter()
            .flat_map(|value| half::f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()),
        DataType::BFloat16 => Ok(values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()),
        other => bail!(
            "cannot encode f32 values as {other:?} storage bytes; expected Float32, Float16, or BFloat16"
        ),
    }
}

/// Widen a device-resident KV read-out to `f32` using the `half` crate's
/// vectorized slice conversion (hardware F16C when available), matching ORT's
/// present-KV mirror path (`onnx_genai_ort::Value::to_vec_f32_lossy`,
/// `crates/onnx-genai-ort/src/value.rs`) bit-for-bit. IEEE f16→f32 is an exact
/// widening, so both paths land identical `f32` bits in the shared host paged
/// store — the paged-KV byte-equality oracle (GAP-3 Inc-C/D/D.1) depends on this
/// convert being the *same* routine ORT uses, not a hand-rolled bit-twiddle.
///
/// Only `f32` (Inc-D) and `f16` (Inc-D.1) rank-4 CUDA GQA caches reach here;
/// `bf16` and other dtypes stay gated to the non-paged fallback
/// ([`DecodeCudaState::kv_bindings_paged_rank4`]) and are rejected defensively.
pub(crate) fn kv_dtype_to_f32(tensor: &Tensor) -> anyhow::Result<Vec<f32>> {
    match tensor.dtype {
        DataType::Float32 => Ok(tensor.to_vec_f32()),
        DataType::Float16 => {
            use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
            if let Some(bits) = tensor.try_as_slice_u16() {
                let halves: &[half::f16] = bits.reinterpret_cast();
                Ok(halves.to_f32_vec())
            } else {
                Ok(tensor
                    .as_bytes()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|bytes| {
                        half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
                    })
                    .collect())
            }
        }
        other => bail!(
            "device present KV must be Float32 or Float16 to mirror into the f32 paged store \
             (bf16 / non-rank-4 stay gated to the non-paged path), got {other:?}"
        ),
    }
}

/// A native "past" input is a fixed-size recurrent state (e.g. the hybrid
/// linear-attention `conv_state` / `recurrent_state`) rather than a growable
/// key/value cache when its sequence axis — the penultimate axis, where the
/// decoder grows KV — is statically sized. Growable KV caches carry a symbolic
/// `past_sequence_length` on that axis and are concatenated each step; recurrent
/// states carry a concrete feature dimension there and are replaced wholesale.
/// This is a purely structural signal (RULES.md §2) — never a model-name gate.
pub(crate) fn is_recurrent_state_shape(shape: &[Dim]) -> bool {
    shape.len() >= 2 && shape[shape.len() - 2].is_static()
}

/// Bytes of fixed-size recurrent state one sequence needs.
///
/// The hybrid decoders (`conv_state`, `recurrent_state`) keep one of these per
/// recurrent layer for as long as a sequence lives, which makes it a
/// per-sequence cost that scales exactly the way KV does -- and unlike KV it was
/// never charged to anything. Admission decides how many sequences fit while
/// this is invisible to it, so the governor can admit a batch that does not fit
/// and the failure lands as an allocation error mid-generation.
///
/// Counted from the graph's own metadata, so a model with no recurrent layers
/// answers zero without being asked about by name (RULES.md §2).
///
/// The batch axis is counted as one: the reservation is per sequence, and the
/// scheduler multiplies. A symbolic non-batch axis means the export did not pin
/// a geometry this can size, and is reported rather than guessed at.
pub(crate) fn recurrent_state_bytes_per_sequence(
    session: &InferenceSession,
    present_to_past: &std::collections::HashMap<String, String>,
) -> anyhow::Result<u64> {
    // Only inputs the resolved I/O actually declares as loop-carried state.
    // Rediscovering that from shape alone -- "the penultimate axis is static" --
    // also matches a fixed-length KV input and any unrelated fixed-shape input,
    // which would charge memory the decoder never keeps, multiplied by the batch
    // size. `is_recurrent_state_shape` is the right test for *classifying* a
    // declared state input; it is not a test for *finding* them.
    let declared: std::collections::HashSet<&str> =
        present_to_past.values().map(String::as_str).collect();
    let mut total: u64 = 0;
    for meta in session.inputs() {
        if !declared.contains(meta.name.as_str()) || !is_recurrent_state_shape(&meta.shape) {
            continue;
        }
        total = total.saturating_add(recurrent_state_tensor_bytes(
            &meta.name,
            meta.dtype,
            &meta.shape,
        )?);
    }
    Ok(total)
}

#[cfg(feature = "native-cuda")]
pub(crate) fn recurrent_state_bytes_from_graph(
    graph: &onnx_runtime_ir::Graph,
    io: Option<&onnx_genai_metadata::ModelIoSpec>,
) -> anyhow::Result<u64> {
    let declared: std::collections::HashSet<&str> = io
        .and_then(|io| io.state_pairs.as_deref())
        .unwrap_or_default()
        .iter()
        .map(|pair| pair.input.as_str())
        .collect();
    let mut total = 0_u64;
    for &input in &graph.inputs {
        let value = graph.value(input);
        let Some(name) = value.name.as_deref() else {
            continue;
        };
        if !declared.contains(name) || !is_recurrent_state_shape(&value.shape) {
            continue;
        }
        total = total.saturating_add(recurrent_state_tensor_bytes(
            name,
            value.dtype,
            &value.shape,
        )?);
    }
    Ok(total)
}

fn recurrent_state_tensor_bytes(name: &str, dtype: DataType, shape: &[Dim]) -> anyhow::Result<u64> {
    let mut elements: u64 = 1;
    for (axis, dim) in shape.iter().copied().enumerate() {
        let extent = match dim {
            Dim::Static(value) => u64::try_from(value).unwrap_or(0),
            Dim::Symbolic(_) if axis == 0 => 1,
            Dim::Symbolic(_) => bail!(
                "cannot size recurrent state '{name}': dimension {axis} of {shape:?} is symbolic \
                 and is not the batch axis, so the export did not pin a geometry to reserve for"
            ),
        };
        elements = elements.saturating_mul(extent);
    }
    let bytes = dtype
        .checked_storage_bytes(usize::try_from(elements).unwrap_or(usize::MAX))
        .with_context(|| format!("unsupported recurrent-state dtype {dtype:?} for '{name}'"))?;
    Ok(u64::try_from(bytes).unwrap_or(u64::MAX))
}

/// Bytes of KV cache one sequence holds at full context, from the declared
/// `present_to_past` pairs.
///
/// # Why this is charged at all
///
/// On the ONNX Runtime path the KV pool leases its capacity at construction, so
/// the ledger sees it. The native path's page table is deliberately bookkeeping
/// only -- without per-layer geometry a page carries no storage -- and the real
/// KV lives in the session's own past/present tensors, which are allocated
/// through the execution provider and never reach the ledger. So the tier that
/// admission and every other consumer read was understated by the largest
/// per-sequence cost the decoder has.
///
/// # Why full context rather than current length
///
/// A lease is a reservation, and the point of reserving is that the memory is
/// there when the sequence grows into it. Charging the current length would
/// admit a sequence the device cannot actually carry to its context limit, and
/// discover that at a `cuMemAlloc` mid-generation -- which is the failure the
/// ledger exists to convert into a refusal at admission.
///
/// # Why the declared pairs rather than shape discovery
///
/// Same reason as [`recurrent_state_bytes_per_sequence`]: `present_to_past` is
/// what the resolved I/O says is loop-carried. Anything else that happens to
/// look like a KV tensor is not one, and charging it would reserve memory the
/// decoder never keeps.
pub(crate) fn kv_cache_bytes_per_sequence(
    session: &InferenceSession,
    present_to_past: &std::collections::HashMap<String, String>,
    max_context: usize,
) -> anyhow::Result<u64> {
    let declared: std::collections::HashSet<&str> =
        present_to_past.values().map(String::as_str).collect();
    let mut tensors = Vec::new();
    for meta in session.inputs() {
        // Recurrent state is loop-carried too, and is charged separately at its
        // own fixed size. Sizing it by context would be wrong by orders of
        // magnitude -- its whole point is that it does not grow.
        if !declared.contains(meta.name.as_str()) || is_recurrent_state_shape(&meta.shape) {
            continue;
        }
        tensors.push(crate::kv_sizing::KvTensorSpec {
            name: meta.name.clone(),
            dtype: match meta.dtype {
                DataType::Float32 => crate::kv_sizing::KvStorageType::Float32,
                DataType::Float16 => crate::kv_sizing::KvStorageType::Float16,
                DataType::BFloat16 => crate::kv_sizing::KvStorageType::BFloat16,
                dtype => bail!("unsupported KV dtype {dtype:?} for '{}'", meta.name),
            },
            shape: meta
                .shape
                .iter()
                .copied()
                .enumerate()
                .map(|(axis, dim)| match dim {
                    Dim::Static(value) => {
                        crate::kv_sizing::KvDimension::Fixed(u64::try_from(value).unwrap_or(0))
                    }
                    Dim::Symbolic(_) if axis == 0 => {
                        crate::kv_sizing::KvDimension::PerSequenceBatch
                    }
                    Dim::Symbolic(_) => crate::kv_sizing::KvDimension::Context,
                })
                .collect(),
        });
    }
    crate::kv_sizing::kv_cache_bytes_for_tensors(&tensors, u64::try_from(max_context).unwrap_or(0))
}

pub(crate) fn make_empty_input_tensor(
    session: &InferenceSession,
    name: &str,
) -> anyhow::Result<Tensor> {
    make_empty_input_tensor_batched(session, name, 1)
}

/// Batch-aware variant of [`make_empty_input_tensor`]: builds an empty (zero
/// sequence-length) past-KV / recurrent-state tensor with the batch axis (axis
/// 0) set to `batch` rows instead of the single-sequence default of `1`.
///
/// Stage 2a uses this for the stateless batch-N fused-forward probe
/// ([`NativeDecodeSession::run_fused_batch_prefill`]): a genuine batch axis with
/// an empty past, so weight streaming is exercised across `N` independent rows
/// without a batched KV *layout* (stage 2b). `batch == 1` is byte-identical to
/// [`make_empty_input_tensor`]. For a non-empty past see
/// [`make_past_input_tensor_batched`].
pub(crate) fn make_empty_input_tensor_batched(
    session: &InferenceSession,
    name: &str,
    batch: usize,
) -> anyhow::Result<Tensor> {
    make_past_input_tensor_batched(session, name, batch, 0)
}

/// Batch-aware past-KV builder that seeds a **non-empty** sequence axis of
/// `past_len` zero-filled positions (batch axis 0 = `batch` rows).
///
/// Stage 2b (#750) uses this to drive the batch-N fused forward with a genuine
/// length-`L` batched KV past ([`NativeDecodeSession::run_fused_batch_forward`]):
/// unlike the stage 2a empty-past probe, this exercises the ONNX attention
/// **batch coupling across QKV / mask / past-KV** at `N > 1` with real past
/// content, and — crucially for the memory trade — commits `batch × past_len`
/// worth of KV so the weight-residency reclaim under the elastic budget (#866)
/// becomes observable in `htod_bytes_per_token`. Growable seq axes are seeded to
/// `past_len`; fixed-size recurrent feature axes keep their static extent and
/// ignore `past_len`. `past_len == 0` is byte-identical to
/// [`make_empty_input_tensor_batched`].
pub(crate) fn make_past_input_tensor_batched(
    session: &InferenceSession,
    name: &str,
    batch: usize,
    past_len: usize,
) -> anyhow::Result<Tensor> {
    let meta = session
        .inputs()
        .iter()
        .find(|meta| meta.name == name)
        .with_context(|| format!("missing native KV metadata for '{name}'"))?;
    if meta.shape.len() < 3 {
        bail!(
            "native KV input '{name}' has unsupported shape {:?}; expected rank at least 3 with sequence on the penultimate axis",
            meta.shape
        );
    }
    let seq_axis = meta.shape.len() - 2;
    let mut shape = Vec::with_capacity(meta.shape.len());
    for (axis, dim) in meta.shape.iter().copied().enumerate() {
        let value = if axis == 0 {
            batch
        } else if axis == seq_axis {
            // Growable KV caches carry the requested `past_len` on the sequence
            // axis; fixed-size recurrent states (hybrid linear-attention
            // conv_state / recurrent_state) carry a static feature dim here and
            // must be seeded at full extent so the first forward sees a
            // zero-filled state.
            match dim {
                Dim::Static(value) => value,
                Dim::Symbolic(_) => past_len,
            }
        } else if let Dim::Static(value) = dim {
            value
        } else {
            bail!(
                "cannot create empty native KV input '{name}': dimension {axis} in shape {:?} is symbolic and is neither batch nor sequence; export a static cache geometry",
                meta.shape
            );
        };
        shape.push(value);
    }
    let numel: usize = shape.iter().product();
    // Sized rather than built: `from_raw` with a zeroed `Vec` allocates these
    // bytes twice and memcpys between them, and a hybrid decoder seeds one of
    // these per recurrent layer per sequence.
    meta.dtype
        .checked_storage_bytes(numel)
        .with_context(|| format!("unsupported KV dtype {:?} for '{name}'", meta.dtype))?;
    Tensor::zeros(meta.dtype, shape)
        .with_context(|| format!("create empty native KV tensor '{name}'"))
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = (fraction << (shift + 1)) & 0x03ff;
            sign | ((127 - 15 - shift) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

pub(crate) fn prefix_slice(tensor: &Tensor, axis: usize, len: usize) -> anyhow::Result<Tensor> {
    let axis_len = *tensor
        .shape
        .get(axis)
        .context("native KV slice axis out of bounds")?;
    if len > axis_len {
        bail!("native KV slice length {len} exceeds axis length {axis_len}");
    }

    let inner = tensor.shape[axis + 1..].iter().product::<usize>();
    let outer = tensor.shape[..axis].iter().product::<usize>();
    let elem_bytes = tensor
        .dtype
        .checked_storage_bytes(1)
        .context("native KV dtype has no fixed storage size")?;
    let source_stride = axis_len * inner * elem_bytes;
    let kept_stride = len * inner * elem_bytes;
    let source = tensor.as_bytes();
    let mut bytes = Vec::with_capacity(outer * kept_stride);
    for index in 0..outer {
        let start = index * source_stride;
        bytes.extend_from_slice(&source[start..start + kept_stride]);
    }
    let mut shape = tensor.shape.clone();
    shape[axis] = len;
    Tensor::from_raw(tensor.dtype, shape, &bytes).context("create sliced native KV tensor")
}

/// Whether the model's decode dispatches work through the shared CPU decode
/// pool -- i.e. it contains quantized `MatMulNBits` / `QMoE` projections whose
/// row-sharding runs on the persistent SPMD (or numa-split) decode workers.
///
/// A dense-f32 graph (no such nodes) instead has its dominant `MatMul`s serviced
/// by the multi-threaded MLAS GEMM, which gains nothing from the SPMD pool's
/// pinned, spinning workers and is slowed by them; such models take the bounded
/// dense decode pool instead (see
/// `onnx_runtime_ep_cpu::with_decode_pool_scope`). The check keys off a
/// structural graph property, never off a specific model, so it generalizes
/// across every quantized and f32 model. Subgraphs (e.g. `If` branches) are
/// scanned too so control-flow-wrapped decoders are classified correctly.
pub(crate) fn graph_uses_decode_pool(graph: &onnx_runtime_ir::Graph) -> bool {
    for (_, node) in graph.nodes.iter() {
        if onnx_runtime_session::is_plugin_fused_node(node) {
            return false;
        }
        if matches!(node.op_type.as_str(), "MatMulNBits" | "QMoE") {
            return true;
        }
        for attr in node.attributes.values() {
            if let onnx_runtime_ir::Attribute::Graph(subgraph) = attr
                && graph_uses_decode_pool(subgraph)
            {
                return true;
            }
        }
    }
    false
}

pub(crate) fn graph_has_plugin_fused(graph: &onnx_runtime_ir::Graph) -> bool {
    graph.nodes.iter().any(|(_, node)| {
        onnx_runtime_session::is_plugin_fused_node(node)
            || node.attributes.values().any(|attr| {
                matches!(attr, onnx_runtime_ir::Attribute::Graph(subgraph) if graph_has_plugin_fused(subgraph))
            })
    })
}

pub(crate) fn diagnose_native_failure(session: &InferenceSession, error: &str) -> String {
    if error.contains("f32 kernel input requires Float32, got Int64") {
        for (_, node) in session.graph().nodes.iter() {
            if node.op_type == "Gather"
                && let Some(data) = node.inputs.first().copied().flatten()
                && session.graph().value(data).dtype == DataType::Int64
            {
                return " (native CPU Gather lacks Int64 data support)".to_string();
            }
        }
    }
    String::new()
}

#[cfg(test)]
mod kv_convert_tests {
    use super::{f32_slice_to_dtype_bytes, kv_dtype_to_f32};
    use onnx_runtime_ir::DataType;
    use onnx_runtime_session::Tensor;

    fn f16_tensor(bits: &[u16]) -> Tensor {
        let bytes: Vec<u8> = bits.iter().flat_map(|bit| bit.to_le_bytes()).collect();
        Tensor::from_raw(DataType::Float16, vec![bits.len()], &bytes)
            .expect("build f16 tensor from bits")
    }

    /// The device f16 present-KV read-out ([`kv_dtype_to_f32`]) must widen with
    /// the exact same `half` routine ORT's paged mirror uses
    /// (`onnx_genai_ort::Value::to_vec_f32_lossy`,
    /// `crates/onnx-genai-ort/src/value.rs`, whose Float16 arm calls
    /// `half::slice::HalfFloatSliceExt::to_f32_vec`). Asserting equality against
    /// the reference `half::f16::to_f32` pins the native read-out to ORT's exact
    /// rounding — the whole basis of the paged-KV byte-equality oracle. A
    /// hand-rolled bit-twiddle that diverged on a subnormal / max-normal would
    /// fail here.
    #[test]
    fn kv_f16_widen_matches_half_reference() {
        // 0, ±small dyadics, a subnormal, and the f16 max-normal (65504).
        let seeds = [0.0f32, 0.5, 1.0, -2.25, 65504.0, 6.103_515_6e-5];
        let bits: Vec<u16> = seeds
            .iter()
            .map(|value| half::f16::from_f32(*value).to_bits())
            .collect();
        let widened = kv_dtype_to_f32(&f16_tensor(&bits)).expect("widen f16 KV");
        let reference: Vec<f32> = bits
            .iter()
            .map(|bit| half::f16::from_bits(*bit).to_f32())
            .collect();
        assert_eq!(
            widened, reference,
            "f16 KV widening must match the half crate (== ORT to_vec_f32_lossy) exactly"
        );
    }

    /// f16 -> f32 (read) -> f16 (seed) must be bit-exact for every finite f16
    /// value, because the source IS f16: widening is exact and the subsequent
    /// narrowing (`f32_slice_to_dtype_bytes`, the same `half::f16::from_f32` ORT
    /// injects with) reproduces the original bits. Covers all subnormals,
    /// normals, ±0, and ±inf; NaN payloads are excluded (narrowing canonicalizes
    /// them). Guards the reuse-seed path from a lossy round-trip.
    #[test]
    fn kv_f16_roundtrip_is_bit_exact() {
        let bits: Vec<u16> = (0..=u16::MAX)
            .filter(|bit| !half::f16::from_bits(*bit).is_nan())
            .collect();
        let source_bytes: Vec<u8> = bits.iter().flat_map(|bit| bit.to_le_bytes()).collect();
        let widened = kv_dtype_to_f32(&f16_tensor(&bits)).expect("widen f16 KV");
        let reencoded =
            f32_slice_to_dtype_bytes(DataType::Float16, &widened).expect("narrow f32 back to f16");
        assert_eq!(
            reencoded, source_bytes,
            "f16 -> f32 -> f16 round-trip must be bit-exact for f16-origin values"
        );
    }

    /// f32 KV stays a pass-through (no dtype change) so the Inc-D f32 device path
    /// is unaffected by the Inc-D.1 dtype branch.
    #[test]
    fn kv_f32_widen_is_identity() {
        let values = [1.5f32, -0.25, 12345.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let tensor = Tensor::from_raw(DataType::Float32, vec![values.len()], &bytes)
            .expect("build f32 tensor");
        assert_eq!(
            kv_dtype_to_f32(&tensor).expect("widen f32 KV"),
            values.to_vec()
        );
        assert_eq!(
            f32_slice_to_dtype_bytes(DataType::Float32, &values).expect("encode f32 KV"),
            bytes
        );
    }
}
