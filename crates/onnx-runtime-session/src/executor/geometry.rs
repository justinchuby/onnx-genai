use super::*;

/// The `[shape, strides, byte_offset]` storage-bounds gate (Holden's
/// precondition). Uses [`view_in_bounds`] for fixed-width dtypes and a
/// `storage_bytes` check for sub-byte packed dtypes (which have no integral
/// per-element byte size).
pub(super) fn view_bounds(
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    dtype: DataType,
    buffer_len: usize,
) -> Result<()> {
    let esize = dtype.byte_size();
    if esize == 0 {
        // Sub-byte (int4/uint4) or variable-width: size via `storage_bytes`.
        let storage_bytes =
            onnx_runtime_ir::checked_expected_bytes(dtype, shape).ok_or_else(|| {
                SessionError::ShapeOverflow {
                    value: "sub-byte tensor view".to_string(),
                    dims: shape.to_vec(),
                }
            })?;
        let need =
            byte_offset
                .checked_add(storage_bytes)
                .ok_or_else(|| SessionError::ShapeOverflow {
                    value: "sub-byte tensor view byte offset".to_string(),
                    dims: shape.to_vec(),
                })?;
        if need > buffer_len {
            return Err(SessionError::from(
                onnx_runtime_ep_api::EpError::InvalidTensorView {
                    reason: format!(
                        "sub-byte view needs {need} bytes but backing allocation is {buffer_len}"
                    ),
                },
            ));
        }
        return Ok(());
    }
    view_in_bounds(shape, strides, byte_offset, esize, buffer_len)?;
    Ok(())
}

/// Gather a strided view over `src` into a fresh contiguous row-major byte
/// buffer. `strides` are in **elements** (may be negative); `byte_offset` is the
/// byte position of the element origin within `src`. `esize` is the element
/// size in bytes (fixed-width types only — callers exclude sub-byte dtypes).
/// This is the materialization copy that turns a zero-copy view back into a
/// contiguous tensor for a strided-unaware consumer or the output boundary.
pub(super) fn gather_view(
    src: &[u8],
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    esize: usize,
) -> Vec<u8> {
    let n: usize = shape.iter().product();
    let mut out = vec![0u8; n * esize];
    if n == 0 {
        return out;
    }
    let rank = shape.len();
    let mut idx = vec![0usize; rank];
    let mut w = 0usize;
    loop {
        let mut off = byte_offset as i64;
        for d in 0..rank {
            off += strides[d] * idx[d] as i64 * esize as i64;
        }
        let s = off as usize;
        out[w..w + esize].copy_from_slice(&src[s..s + esize]);
        w += esize;
        // Advance the row-major index; stop when it wraps to all-zero.
        let mut carried = true;
        for axis in (0..rank).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                carried = false;
                break;
            }
            idx[axis] = 0;
        }
        if carried {
            break;
        }
    }
    out
}

/// Element count of a shape with overflow checking. A malicious or corrupt
/// shape whose dims multiply past `usize::MAX` would silently wrap under a plain
/// `iter().product()`, under-sizing the backing buffer. Returns
/// [`SessionError::ShapeOverflow`] instead so the caller allocates nothing.
pub(super) fn checked_numel(dims: &[usize], value: impl FnOnce() -> String) -> Result<usize> {
    let mut acc = 1usize;
    for &d in dims {
        acc = match acc.checked_mul(d) {
            Some(n) => n,
            None => {
                return Err(SessionError::ShapeOverflow {
                    value: value(),
                    dims: dims.to_vec(),
                });
            }
        };
    }
    Ok(acc)
}

/// Byte size of `numel` elements of `dtype` with overflow checking. Even when
/// the element *count* fits in `usize` (guarded by [`checked_numel`]), the
/// element-count → bytes multiply can still wrap for a fixed-width dtype and
/// under-size the backing buffer. Returns [`SessionError::ShapeOverflow`] so the
/// caller allocates nothing rather than a wrapped, undersized buffer.
pub(super) fn checked_storage_bytes(
    dtype: DataType,
    numel: usize,
    value: impl FnOnce() -> String,
    dims: &[usize],
) -> Result<usize> {
    dtype
        .checked_storage_bytes(numel)
        .ok_or_else(|| SessionError::ShapeOverflow {
            value: value(),
            dims: dims.to_vec(),
        })
}

/// The effective operator-set version governing `node`.
///
/// A node-local [`Node::version`] wins when set, which is how a rewrite emits a
/// newer standard operator without claiming every other node in that domain was
/// upgraded with it — the CPU provider's `Swish` fusion needs opset 24 in graphs
/// exported against older ones. `None` falls back to the graph's import, which
/// is ONNX's own behaviour and what every loaded node uses.
///
/// Loaded IR is canonical (the default domain is `""`, never `"ai.onnx"`; see
/// [`onnx_runtime_ir::normalize_domain`]), so the domain keys directly into the
/// opset-import map.
pub(super) fn effective_opset(graph: &Graph, node: &Node) -> u64 {
    graph.effective_opset(node).unwrap_or_else(|| {
        unreachable!(
            "internal invariant violated: node #{} ({}::{}) has no opset import",
            node.id.0,
            if node.domain.is_empty() {
                "ai.onnx"
            } else {
                &node.domain
            },
            node.op_type
        )
    })
}

/// Substitute concrete symbol bindings into a (possibly symbolic) shape.
/// Returns `None` if any dim is a symbol with no binding.
pub(super) fn substitute(shape: &Shape, bindings: &HashMap<SymbolId, usize>) -> Option<Vec<usize>> {
    shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Some(*n),
            Dim::Symbolic(s) => bindings.get(s).copied(),
        })
        .collect()
}

/// Like [`substitute`] but writes into `out` in place, reusing its existing
/// capacity (no heap allocation). Returns `false` (leaving `out` empty) if any
/// dim is an unbound symbol. Used by the decode-plan memo replay to refresh the
/// variant tail without allocating a fresh `Vec` per value per token.
pub(super) fn substitute_into(
    shape: &Shape,
    bindings: &HashMap<SymbolId, usize>,
    out: &mut Vec<usize>,
) -> bool {
    out.clear();
    for d in shape {
        match d {
            Dim::Static(n) => out.push(*n),
            Dim::Symbolic(s) => match bindings.get(s) {
                Some(&v) => out.push(v),
                None => {
                    out.clear();
                    return false;
                }
            },
        }
    }
    true
}

/// Decode raw little-endian integer bytes as `i64` for `dtype`, or `None` if the
/// dtype is not an integer the shape math understands. Shared by the owned-buffer
/// and materialized-view integer-input readers.
pub(super) fn bytes_as_i64(bytes: &[u8], dtype: DataType) -> Option<Vec<i64>> {
    match dtype {
        DataType::Int64 => onnx_runtime_ir::read_vec_le(bytes).ok(),
        DataType::Int32 => onnx_runtime_ir::read_vec_le::<i32>(bytes)
            .ok()
            .map(|values| values.into_iter().map(i64::from).collect()),
        _ => None,
    }
}

pub(super) fn bytes_as_f64(bytes: &[u8], dtype: DataType) -> Option<Vec<f64>> {
    match dtype {
        DataType::Float32 => onnx_runtime_ir::read_vec_le::<f32>(bytes)
            .ok()
            .map(|values| values.into_iter().map(f64::from).collect()),
        DataType::Float64 => onnx_runtime_ir::read_vec_le(bytes).ok(),
        _ => None,
    }
}

/// Whether a runtime input is small enough to materialize as shape-propagation
/// data. Keep this gate ahead of `contiguous_bytes`: unsupported tensors must
/// degrade to absent shape-data without allocating or copying their contents.
pub(super) fn bounded_shape_input(dtype: DataType, shape: &[usize]) -> bool {
    if !matches!(dtype, DataType::Int32 | DataType::Int64) {
        return false;
    }
    if shape.len() > 1 {
        return false;
    }
    shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .is_some_and(|count| count <= MAX_SHAPE_DATA_ELEMS)
}

/// Whether a node needs a float runtime input to resolve a data-dependent
/// output extent. The list is deliberately explicit, so shape propagation never
/// copies unrelated tensor data to host.
pub(super) fn reads_float_shape_input(node: &Node, input_index: usize, opset: u64) -> bool {
    node.is_default_domain()
        && ((node.op_type == "Resize" && input_index == if opset == 10 { 1 } else { 2 })
            || (node.op_type == "NonMaxSuppression" && matches!(input_index, 0 | 1 | 3 | 4)))
}

pub(super) fn kernel_input_uses_physical_capacity(node: &Node, input_index: usize) -> bool {
    // GQA treats the cache tensor extent as capacity and obtains the valid past
    // length from seqlens_k. Standard Attention instead derives past length from
    // the cache tensor extent itself.
    if node.domain == "com.microsoft"
        && node.op_type == "GroupQueryAttention"
        && matches!(input_index, 3 | 4)
    {
        return true;
    }
    // Default-domain `Attention` with an in-op KV cache (past_key=input 4,
    // past_value=input 5) can likewise treat the cache extent as physical
    // capacity, deriving the valid attended length on-device from the additive
    // attention mask (input 3) instead of the growing cache extent. This mirrors
    // the GQA treatment and is what lets the decode step bind the KV cache at a
    // fixed capacity so whole-step CUDA-graph capture stays shape-static. Gated
    // to the mask-driven, non-causal form (a present mask input and no
    // `is_causal` attribute): that path derives length from the mask frontier,
    // so the cache extent is pure capacity. Causal-attribute or mask-less
    // Attention still reads the cache extent as the valid length.
    node.is_default_domain()
        && node.op_type == "Attention"
        && matches!(input_index, 4 | 5)
        && node.inputs.get(3).is_some_and(Option::is_some)
        && node
            .attr("is_causal")
            .and_then(|attr| attr.as_int())
            .unwrap_or(0)
            == 0
        || (
            // `pkg.nxrt::IndexShare` mirrors the mask-driven Attention treatment.
            // Its capacity form emits the 3-output present that ALIASES the
            // fixed-capacity past bindings in place (past_key=input 3, past_value=
            // input 4) instead of a growing `past ⧺ current`, and carries the valid
            // length via the additive attention_bias (input 6) frontier — which the
            // kernel scans on-device to place the current token and bound the gather.
            // Binding those caches at physical capacity is what keeps whole-step
            // capture shape-static. Gated on the 3-output present + present bias so
            // the growing-concat form (1 output, or bias-less) still reads the cache
            // extent as the valid length.
            node.domain == "pkg.nxrt"
                && node.op_type == "IndexShare"
                && matches!(input_index, 3 | 4)
                && node.outputs.len() == 3
                && node.inputs.get(6).is_some_and(Option::is_some)
        )
}

pub(super) fn kernel_input_uses_padded_capacity(node: &Node, input_index: usize) -> bool {
    // Persistent decode masks have a zero-filled suffix. Capacity-oriented
    // graphs intentionally read Shape at the allocation extent and ReduceSum is
    // unchanged by that suffix; prefix-sensitive transforms such as CumSum must
    // instead see the logical valid length.
    node.is_default_domain()
        && input_index == 0
        && matches!(node.op_type.as_str(), "Shape" | "ReduceSum")
}
