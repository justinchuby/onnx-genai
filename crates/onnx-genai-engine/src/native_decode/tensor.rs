use super::*;

fn extract_logits(tensor: &Tensor) -> anyhow::Result<Vec<Vec<f32>>> {
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

fn argmax_logits_tensor(tensor: &Tensor) -> anyhow::Result<TokenId> {
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
            for bytes in tensor.as_bytes().chunks_exact(4).take(value_count) {
                visit(f32::from_le_bytes(
                    bytes.try_into().expect("four-byte chunk"),
                ))?;
            }
        }
        DataType::Float16 => {
            if let Some(bits) = tensor.try_as_slice_u16() {
                use half::slice::HalfBitsSliceExt;
                let halves: &[half::f16] = bits.reinterpret_cast();
                return argmax_finite_half_values(halves, value_count, row_start)
                    .map(|index| index as TokenId);
            }
            for bytes in tensor.as_bytes().chunks_exact(2).take(value_count) {
                visit(f16_to_f32(u16::from_le_bytes(
                    bytes.try_into().expect("two-byte chunk"),
                )))?;
            }
        }
        DataType::BFloat16 => {
            if let Some(bits) = tensor.try_as_slice_u16() {
                use half::slice::HalfBitsSliceExt;
                let halves: &[half::bf16] = bits.reinterpret_cast();
                return argmax_finite_half_values(halves, value_count, row_start)
                    .map(|index| index as TokenId);
            }
            for bytes in tensor.as_bytes().chunks_exact(2).take(value_count) {
                visit(f32::from_bits(
                    u32::from(u16::from_le_bytes(
                        bytes.try_into().expect("two-byte chunk"),
                    )) << 16,
                ))?;
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

fn extract_last_row(tensor: &Tensor) -> anyhow::Result<Vec<f32>> {
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
            .chunks_exact(2)
            .map(|bytes| f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect()),
        DataType::BFloat16 => Ok(tensor
            .as_bytes()
            .chunks_exact(2)
            .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16))
            .collect()),
        dtype => bail!("native logits must be Float32, Float16, or BFloat16, got {dtype:?}"),
    }
}

fn tensor_from_f32_as(dtype: DataType, shape: &[usize], values: &[f32]) -> anyhow::Result<Tensor> {
    match dtype {
        DataType::Float32 => Ok(Tensor::from_f32(shape, values)?),
        DataType::Float16 => {
            let bytes = values
                .iter()
                .flat_map(|value| half::f16::from_f32(*value).to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            Ok(Tensor::from_raw(dtype, shape.to_vec(), &bytes)?)
        }
        DataType::BFloat16 => {
            let bytes = values
                .iter()
                .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            Ok(Tensor::from_raw(dtype, shape.to_vec(), &bytes)?)
        }
        other => bail!(
            "native embeddings input must be Float32, Float16, or BFloat16, got {other:?}; fix io.inputs_embeds_input or export a floating tensor"
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
fn is_recurrent_state_shape(shape: &[Dim]) -> bool {
    shape.len() >= 2 && shape[shape.len() - 2].is_static()
}

fn make_empty_input_tensor(session: &InferenceSession, name: &str) -> anyhow::Result<Tensor> {
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
            1
        } else if axis == seq_axis {
            // Growable KV caches start with an empty sequence axis; fixed-size
            // recurrent states (hybrid linear-attention conv_state /
            // recurrent_state) carry a static feature dim here and must be seeded
            // at full extent so the first forward sees a zero-filled state.
            match dim {
                Dim::Static(value) => value,
                Dim::Symbolic(_) => 0,
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
    let bytes = meta
        .dtype
        .checked_storage_bytes(shape.iter().product())
        .with_context(|| format!("unsupported KV dtype {:?} for '{name}'", meta.dtype))?;
    Tensor::from_raw(meta.dtype, shape, &vec![0; bytes])
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

fn prefix_slice(tensor: &Tensor, axis: usize, len: usize) -> anyhow::Result<Tensor> {
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
fn graph_uses_decode_pool(graph: &onnx_runtime_ir::Graph) -> bool {
    for (_, node) in graph.nodes.iter() {
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

fn diagnose_native_failure(session: &InferenceSession, error: &str) -> String {
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
