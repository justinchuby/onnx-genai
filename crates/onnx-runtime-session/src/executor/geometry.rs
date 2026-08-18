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
/// size in bytes (fixed-width types only -- callers exclude sub-byte dtypes).
/// This is the materialization copy that turns a zero-copy view back into a
/// contiguous tensor for a strided-unaware consumer or the output boundary.
///
/// # Why this is not the obvious element loop
///
/// It used to be. A view is defined element-wise, so an odometer over the
/// logical index plus a rank-length dot product per element is the direct
/// transcription of the definition -- and it costs a carry chain, `rank`
/// multiply-adds and a one-element `copy_from_slice` for every element, however
/// much of the view is actually contiguous.
///
/// That cost lands on the two hottest structural paths in the executor:
/// materializing a **view graph output** (`contiguous_bytes`) and materializing
/// a **strided input** for a kernel that cannot take one. A `Transpose` whose
/// result is a graph output goes through the first, which is why a `Transpose`
/// benchmark measured 13.8x-120.8x behind ONNX Runtime while the `Transpose`
/// *kernel* was never on the critical path -- it correctly returns a zero-copy
/// view, and all of the time was here.
///
/// Almost none of that work is inherent. Whichever trailing axes of the view are
/// still contiguous in the source form runs of consecutive bytes that move
/// together, and adjacent axes whose strides compose can be merged into one
/// longer axis. So this collapses the geometry first and then copies the largest
/// contiguous run it can find, falling back to the element walk only for the
/// axes that genuinely interleave -- and even then with an incremental offset
/// rather than a fresh dot product per element.
pub(super) fn gather_view(
    src: &[u8],
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    esize: usize,
) -> Vec<u8> {
    let n: usize = shape.iter().product();
    let mut out = vec![0u8; n * esize];
    gather_view_into(&mut out, src, shape, strides, byte_offset, esize);
    out
}

/// [`gather_view`] writing into a caller-provided buffer.
///
/// `dst` must be exactly `numel * esize` bytes, and every one of them is
/// written, so the caller may pass freshly allocated memory it has not
/// initialized. That is what lets the graph-output path gather straight into the
/// tensor's own allocation instead of building a `Vec` and copying it in.
pub(super) fn gather_view_into(
    dst: &mut [u8],
    src: &[u8],
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    esize: usize,
) {
    let n: usize = shape.iter().product();
    debug_assert_eq!(dst.len(), n * esize);
    if n == 0 {
        return;
    }
    let (shape, strides) = collapse_view(shape, strides);
    gather_collapsed(src, dst, &shape, &strides, byte_offset, esize);
}

/// Drop size-1 axes and fuse adjacent axes whose strides compose.
///
/// Axes `i` and `i+1` address one contiguous run exactly when
/// `strides[i] == strides[i+1] * shape[i+1]`, because then advancing `i` by one
/// lands precisely where advancing `i+1` past its end would. Size-1 axes address
/// a single element wherever they sit, so their stride is irrelevant and they are
/// removed first -- which is also what lets otherwise-separated axes become
/// neighbours.
///
/// Collapsing changes no element's address; it only removes loop levels.
fn collapse_view(shape: &[usize], strides: &[i64]) -> (Vec<usize>, Vec<i64>) {
    let mut dims: Vec<(usize, i64)> = shape
        .iter()
        .zip(strides)
        .filter(|&(&dim, _)| dim != 1)
        .map(|(&dim, &stride)| (dim, stride))
        .collect();
    if dims.is_empty() {
        // Every axis was size 1: one element at the origin.
        return (vec![1], vec![0]);
    }
    let mut fused: Vec<(usize, i64)> = Vec::with_capacity(dims.len());
    for (dim, stride) in dims.drain(..) {
        match fused.last_mut() {
            Some((prev_dim, prev_stride)) if *prev_stride == stride * dim as i64 => {
                *prev_dim *= dim;
                *prev_stride = stride;
            }
            _ => fused.push((dim, stride)),
        }
    }
    fused.into_iter().unzip()
}

/// Minimum output bytes before splitting the gather across workers pays off.
///
/// Below roughly this size the copy finishes in less time than a `rayon` fan-out
/// costs to schedule.
const MIN_PARALLEL_GATHER_BYTES: usize = 256 * 1024;

/// Copy a collapsed view into `out`, choosing the largest contiguous unit.
fn gather_collapsed(
    src: &[u8],
    out: &mut [u8],
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    esize: usize,
) {
    let rank = shape.len();
    // The innermost axis is unit-stride exactly when the view's last dimension
    // is still laid out consecutively in the source, which is the common case
    // for every layout permutation that does not move the last axis -- including
    // every attention `(B,S,H,D) -> (B,H,S,D)` transpose.
    let (block_elems, outer_rank) = if strides[rank - 1] == 1 {
        (shape[rank - 1], rank - 1)
    } else {
        (1, rank)
    };
    let block = block_elems * esize;

    if outer_rank == 0 {
        // Fully contiguous after collapsing: one copy.
        out[..block].copy_from_slice(&src[byte_offset..byte_offset + block]);
        return;
    }

    let outer_shape = &shape[..outer_rank];
    let outer_strides = &strides[..outer_rank];
    let blocks: usize = outer_shape.iter().product();

    // Byte offset of output block `k`, computed from `k`'s mixed-radix digits.
    // Deriving it from `k` rather than carrying an odometer is what lets a
    // parallel task start at an arbitrary block.
    let src_offset_of = |mut k: usize| -> usize {
        let mut off = byte_offset as i64;
        for axis in (0..outer_rank).rev() {
            let digit = k % outer_shape[axis];
            k /= outer_shape[axis];
            off += digit as i64 * outer_strides[axis] * esize as i64;
        }
        off as usize
    };

    let body = |dst: &mut [u8], first_block: usize| {
        // Incremental odometer: the per-element dot product this function used
        // to pay is replaced by one add per block, with a carry only when an
        // axis wraps.
        let mut idx = vec![0usize; outer_rank];
        let mut off = src_offset_of(first_block);
        {
            let mut k = first_block;
            for axis in (0..outer_rank).rev() {
                idx[axis] = k % outer_shape[axis];
                k /= outer_shape[axis];
            }
        }
        let mut written = 0usize;
        loop {
            dst[written..written + block].copy_from_slice(&src[off..off + block]);
            written += block;
            if written == dst.len() {
                break;
            }
            let mut axis = outer_rank;
            loop {
                axis -= 1;
                idx[axis] += 1;
                off = (off as i64 + outer_strides[axis] * esize as i64) as usize;
                if idx[axis] < outer_shape[axis] {
                    break;
                }
                off = (off as i64 - outer_shape[axis] as i64 * outer_strides[axis] * esize as i64)
                    as usize;
                idx[axis] = 0;
                if axis == 0 {
                    break;
                }
            }
        }
    };

    let workers = rayon::current_num_threads().max(1);
    if workers > 1 && blocks > 1 && blocks * block >= MIN_PARALLEL_GATHER_BYTES {
        use rayon::prelude::*;
        let per_task = blocks.div_ceil(workers).max(1);
        out.par_chunks_mut(per_task * block)
            .enumerate()
            .for_each(|(task, chunk)| body(chunk, task * per_task));
    } else {
        body(out, 0);
    }
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

/// Decode raw little-endian integer or boolean bytes as `i64` for `dtype`, or
/// `None` if the dtype is not scalar data the shape math understands. Shared by
/// the owned-buffer and materialized-view shape-input readers.
pub(super) fn bytes_as_i64(bytes: &[u8], dtype: DataType) -> Option<Vec<i64>> {
    match dtype {
        DataType::Int64 => onnx_runtime_ir::read_vec_le(bytes).ok(),
        DataType::Int32 => onnx_runtime_ir::read_vec_le::<i32>(bytes)
            .ok()
            .map(|values| values.into_iter().map(i64::from).collect()),
        DataType::Bool => Some(bytes.iter().map(|&value| i64::from(value != 0)).collect()),
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
    if !matches!(dtype, DataType::Bool | DataType::Int32 | DataType::Int64) {
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

/// The default-domain ops that make up the standard additive causal-attention
/// mask builder (`attention_mask → CumSum/Unsqueeze/… → Where(0/-inf) → Cast →
/// Attention[input 3]`). When the mask binding's entire transitive consumer cone
/// is limited to these ops and every leaf is a capacity-form `Attention` mask
/// input (see [`is_capacity_form_attention_mask_input`]), the mask can be frozen
/// to physical width without changing the additive bias: valid columns `[0, L)`
/// are width-invariant (the CumSum prefix and fixed key positions are unchanged
/// by width) and padded columns `[L, max_len)` are forced to `-inf` by the
/// padding branch's `And`/`Where`, which dominates the CumSum suffix.
///
/// Broadcasting combiners that mix the mask *elementwise* with a logical-width
/// score — e.g. GLM-5.2's indexer `Add` — are deliberately **absent** from this
/// set, so a mask feeding such a consumer never classifies as padded-safe and
/// keeps exposing its logical valid length (freezing must never leak `max_len`
/// into a logical-width computation). Only ops actually observed in the standard
/// builder cone are listed; anything else disqualifies the binding conservatively.
pub(super) fn is_additive_mask_builder_op(node: &Node) -> bool {
    node.is_default_domain()
        && matches!(
            node.op_type.as_str(),
            "CumSum" | "Unsqueeze" | "Cast" | "GreaterOrEqual" | "And" | "Where" | "Slice" | "Sub"
        )
}

/// Whether `node`/`input_index` is the additive-mask input (input 3) of a
/// capacity-form `Attention`: a default-domain `Attention` whose KV cache
/// (inputs 4/5) is already bound at physical capacity — i.e. a present mask and
/// no `is_causal` attribute, so it derives the valid length from the mask
/// frontier (see [`kernel_input_uses_physical_capacity`]). Such a node is
/// *designed* to consume a physical-width additive mask, so it is a valid leaf
/// for the frozen-mask (padded-capacity) classification.
///
/// The KV cache inputs (`past_key` = input 4, `past_value` = input 5) must both
/// actually exist: `kernel_input_uses_physical_capacity` gates only on the mask
/// (input 3) presence and `is_causal == 0`, but the CUDA `Attention` kernel's
/// fixed-capacity append contract requires both past caches bound at physical
/// capacity (`standard_attention.rs`: `has_past_key`/`has_past_value` on inputs
/// 4/5, and `fixed_capacity_append` compares their capacity to the mask key
/// width). A masked, non-causal `Attention` with only q/k/v/mask and no KV
/// binding is NOT a capacity-form leaf, so require both KV inputs here rather
/// than blessing any masked non-causal `Attention` as a valid cone terminus.
pub(super) fn is_capacity_form_attention_mask_input(node: &Node, input_index: usize) -> bool {
    input_index == 3
        && node.is_default_domain()
        && node.op_type == "Attention"
        && node.inputs.get(4).is_some_and(Option::is_some)
        && node.inputs.get(5).is_some_and(Option::is_some)
        && kernel_input_uses_physical_capacity(node, 4)
}

/// Structural pattern-match: is `mask` an attention-mask binding whose entire
/// transitive consumer cone is the standard additive causal-mask builder, with
/// every leaf a capacity-form `Attention` mask input (input 3)?
///
/// Walks forward from the binding over graph edges. Each consumer must be one of:
/// - a physical-extent shape read (`Shape`/`ReduceSum`, a safe leaf),
/// - a capacity-form `Attention` mask input (a valid leaf — the `Attention` op
///   itself is not traversed),
/// - an additive-mask-builder op (traversed onward to its outputs),
///
/// otherwise the binding is disqualified (returns `false`). Requires at least one
/// capacity-form `Attention` leaf, so a mask that only feeds shape reads — or
/// escapes as a graph output — is never blessed by this path. Because the builder
/// op set ([`is_additive_mask_builder_op`]) excludes broadcasting logical-width
/// combiners (e.g. `Add`), GLM-5.2's indexer mask cone is rejected here and keeps
/// exposing its logical valid length.
///
/// When this holds, freezing the mask to physical (padded) width is byte-identical
/// to the logical-width mask: valid columns are width-invariant and padded columns
/// are forced to `-inf` by the padding branch — exactly the physical-width additive
/// mask a capacity-form `Attention` is designed to consume — which is what lets the
/// decode step keep a fixed-capacity, capture-stable mask binding.
pub(super) fn mask_binding_feeds_capacity_form_attention(graph: &Graph, mask: ValueId) -> bool {
    use std::collections::{HashMap, HashSet, VecDeque};

    // value → consumers as (node id, input slot).
    let mut consumers: HashMap<ValueId, Vec<(NodeId, usize)>> = HashMap::new();
    for (node_id, node) in graph.nodes.iter() {
        for (slot, value) in node.inputs.iter().enumerate() {
            if let Some(vid) = value {
                consumers.entry(*vid).or_default().push((node_id, slot));
            }
        }
    }
    let graph_outputs: HashSet<ValueId> = graph.outputs.iter().copied().collect();

    let mut visited: HashSet<ValueId> = HashSet::new();
    let mut frontier: VecDeque<ValueId> = VecDeque::new();
    frontier.push_back(mask);
    let mut reached_attention = false;
    while let Some(value) = frontier.pop_front() {
        if !visited.insert(value) {
            continue;
        }
        // A mask-derived value must not escape as a graph output under a frozen
        // (physical-width) mask. This includes the root mask binding itself: if
        // the binding is *also* a graph output, freezing it to physical width
        // would leak the padded `max_len` into whatever consumes that output, so
        // the binding must keep exposing its logical valid length. (Input
        // bindings can in principle also be graph outputs, so the root is not
        // exempt.)
        if graph_outputs.contains(&value) {
            return false;
        }
        for &(node_id, slot) in consumers.get(&value).map_or(&[][..], Vec::as_slice) {
            let node = graph.node(node_id);
            if kernel_input_uses_padded_capacity(node, slot) {
                // `Shape`/`ReduceSum`: reads the physical extent — safe leaf.
                continue;
            }
            if is_capacity_form_attention_mask_input(node, slot) {
                reached_attention = true;
                continue;
            }
            if is_additive_mask_builder_op(node) {
                for out in &node.outputs {
                    frontier.push_back(*out);
                }
                continue;
            }
            // Any other consumer (e.g. GLM-5.2's indexer `Add`, or a MatMul /
            // Gather / graph-output sink) observes the mask width: disqualify.
            return false;
        }
    }
    reached_attention
}

#[cfg(test)]
mod gather_tests {
    use super::*;

    /// The element-at-a-time walk `gather_view` used to be, kept verbatim as the
    /// reference every fast path is checked against.
    fn reference(
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

    /// Row-major strides in elements.
    fn contiguous(shape: &[usize]) -> Vec<i64> {
        let mut strides = vec![1i64; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1] as i64;
        }
        strides
    }

    /// The view a `perm` produces over a contiguous tensor of `base`.
    fn permuted(base: &[usize], perm: &[usize]) -> (Vec<usize>, Vec<i64>) {
        let strides = contiguous(base);
        (
            perm.iter().map(|&p| base[p]).collect(),
            perm.iter().map(|&p| strides[p]).collect(),
        )
    }

    fn permutations(rank: usize) -> Vec<Vec<usize>> {
        fn go(current: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
            if k == current.len() {
                out.push(current.clone());
                return;
            }
            for i in k..current.len() {
                current.swap(k, i);
                go(current, k + 1, out);
                current.swap(k, i);
            }
        }
        let mut result = Vec::new();
        let mut current: Vec<usize> = (0..rank).collect();
        go(&mut current, 0, &mut result);
        result
    }

    /// The correctness contract for the rewrite: every collapsed/blocked path
    /// must reproduce the original element walk exactly, for every permutation
    /// of every rank up to 4, including the size-1 axes that drive collapsing.
    #[test]
    fn every_gather_path_matches_the_element_walk() {
        let bases: &[&[usize]] = &[
            &[6],
            &[5, 3],
            &[1, 7],
            &[7, 1],
            &[2, 3, 4],
            &[1, 6, 5],
            &[6, 1, 5],
            &[6, 5, 1],
            &[2, 3, 4, 5],
            &[2, 1, 4, 5],
            &[1, 3, 1, 5],
            &[2, 8, 3, 16],
        ];
        for base in bases {
            let n: usize = base.iter().product();
            let src: Vec<u8> = (0..n * 4).map(|i| (i * 31 % 251) as u8).collect();
            for perm in permutations(base.len()) {
                let (shape, strides) = permuted(base, &perm);
                assert_eq!(
                    gather_view(&src, &shape, &strides, 0, 4),
                    reference(&src, &shape, &strides, 0, 4),
                    "base {base:?} perm {perm:?}"
                );
            }
        }
    }

    /// A non-zero origin must be honoured. A collapse that folded the offset
    /// into a stride would pass every zero-offset test and silently read the
    /// wrong window here.
    #[test]
    fn a_byte_offset_shifts_the_whole_gather() {
        let base = [4usize, 6];
        let src: Vec<u8> = (0..24 * 4 + 64).map(|i| (i % 241) as u8).collect();
        let (shape, strides) = permuted(&base, &[1, 0]);
        for byte_offset in [0usize, 4, 40, 64] {
            assert_eq!(
                gather_view(&src, &shape, &strides, byte_offset, 4),
                reference(&src, &shape, &strides, byte_offset, 4),
                "byte_offset={byte_offset}"
            );
        }
    }

    /// Negative strides are legal (a reversed `Slice` view), and the collapse's
    /// `stride * dim` fusion test has to keep working when the product is
    /// negative.
    #[test]
    fn negative_strides_are_gathered_in_the_right_order() {
        let src: Vec<u8> = (0..5 * 4 * 4).map(|i| (i % 251) as u8).collect();
        // A 5-element axis walked backwards from the last element.
        let shape = [5usize, 4];
        let strides = [-4i64, 1];
        let byte_offset = 4 * 4 * 4;
        assert_eq!(
            gather_view(&src, &shape, &strides, byte_offset, 4),
            reference(&src, &shape, &strides, byte_offset, 4)
        );
    }

    /// A zero stride (a broadcast axis) repeats one run, which the fusion test
    /// must not mistake for contiguity.
    #[test]
    fn a_zero_stride_axis_repeats_rather_than_advances() {
        let src: Vec<u8> = (0..4 * 4).map(|i| (i % 251) as u8).collect();
        let shape = [3usize, 4];
        let strides = [0i64, 1];
        assert_eq!(
            gather_view(&src, &shape, &strides, 0, 4),
            reference(&src, &shape, &strides, 0, 4)
        );
    }

    /// Collapsing must never change the element count or the addresses, only
    /// the number of loop levels; a fully contiguous view must reduce to one
    /// axis so the gather becomes a single `memcpy`.
    #[test]
    fn collapse_reduces_a_contiguous_view_to_one_axis() {
        assert_eq!(
            collapse_view(&[2, 3, 4], &contiguous(&[2, 3, 4])),
            (vec![24], vec![1])
        );
        assert_eq!(collapse_view(&[1, 5, 1], &[7, 1, 3]), (vec![5], vec![1]));
        assert_eq!(collapse_view(&[1, 1, 1], &[9, 9, 9]), (vec![1], vec![0]));
        // The attention layout keeps its innermost run and loses nothing else.
        let (shape, strides) = permuted(&[2, 8, 4, 64], &[0, 2, 1, 3]);
        assert_eq!(
            collapse_view(&shape, &strides),
            (vec![2, 4, 8, 64], vec![2048, 64, 256, 1])
        );
    }

    /// The parallel fan-out must be bit-identical to the serial path on a view
    /// large enough to cross the threshold, at thread counts that do not divide
    /// the block count evenly.
    #[test]
    fn the_parallel_gather_is_bit_identical_to_the_serial_one() {
        // (B=1, S=257, H=12, D=64) -> (B, H, S, D): 3084 blocks of 256 bytes.
        let base = [1usize, 257, 12, 64];
        let n: usize = base.iter().product();
        let src: Vec<u8> = (0..n * 4).map(|i| (i * 17 % 251) as u8).collect();
        let (shape, strides) = permuted(&base, &[0, 2, 1, 3]);
        let expect = reference(&src, &shape, &strides, 0, 4);
        for threads in [1usize, 3, 8] {
            let got = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| gather_view(&src, &shape, &strides, 0, 4));
            assert_eq!(got, expect, "threads={threads}");
        }
    }

    /// An empty view must allocate nothing and read nothing.
    #[test]
    fn an_empty_view_gathers_to_an_empty_buffer() {
        assert!(gather_view(&[], &[0, 4], &[4, 1], 0, 4).is_empty());
        assert!(gather_view(&[1, 2, 3, 4], &[3, 0], &[1, 1], 0, 4).is_empty());
    }
}
