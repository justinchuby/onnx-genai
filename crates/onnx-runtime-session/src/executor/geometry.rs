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

const MAX_COMPRESS_CONDITION_ELEMS: usize = 1 << 20;

/// `Compress` is data-dependent on its full boolean condition, which can be
/// larger than ordinary scalar/vector shape metadata for image models.
pub(super) fn bounded_compress_condition(dtype: DataType, shape: &[usize]) -> bool {
    dtype == DataType::Bool && shape.len() == 1 && shape[0] <= MAX_COMPRESS_CONDITION_ELEMS
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
    // only on a present mask input (input 3): the standard additive causal-mask
    // builder (`Where(And(attention_mask, causal), 0, -inf)`) is frozen to
    // physical capacity alongside the KV, and its last-row frontier is the true
    // valid length in BOTH the causal and non-causal form — at the last query
    // row the causal frontier and the padding frontier coincide. The CUDA
    // `Attention` kernel derives both the score-loop extent AND the causal offset
    // from that on-device length (`standard_attention.rs`, `dev_len`), so the
    // cache extent is pure capacity regardless of the op's `is_causal` attribute.
    // A mask-less `Attention` still reads the cache extent as the valid length.
    node.is_default_domain()
        && node.op_type == "Attention"
        && matches!(input_index, 4 | 5)
        && node.inputs.get(3).is_some_and(Option::is_some)
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
        || (
            // `pkg.nxrt::KvCacheCapacityAppend` (the CUDA-only rewritten form of a
            // decomposed-attention KV-cache growth `Concat`, see
            // `is_kv_cache_growth_concat`) writes `current` into a fixed-capacity
            // `past` in place at the offset(s) carried by its third operand
            // (`position_ids`), rather than growing a logical concatenation. Its
            // own kernel is therefore *designed* to receive `past` at physical
            // capacity — exactly like GQA/Attention/IndexShare above — so binding
            // it that way from the first (empty) step is what keeps whole-step
            // CUDA-graph capture shape-static. Gated on the `position_ids` operand
            // (input 2) actually being present: a malformed 2-input node with this
            // op/domain (which the rewrite never produces, but nothing else
            // enforces structurally) has no write-offset signal and must not be
            // treated as capacity-safe.
            node.domain == KV_CAPACITY_APPEND_DOMAIN
                && node.op_type == KV_CAPACITY_APPEND_OP
                && input_index == 0
                && node.inputs.get(2).is_some_and(Option::is_some)
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

pub(super) fn binding_shape_result_is_observed(graph: &Graph, input: ValueId) -> bool {
    let shape_outputs = graph
        .nodes
        .iter()
        .filter(|(_, node)| {
            node.is_default_domain()
                && node.op_type == "Shape"
                && node.inputs.first().copied().flatten() == Some(input)
        })
        .flat_map(|(_, node)| node.outputs.iter().copied())
        .collect::<HashSet<_>>();
    if shape_outputs.is_empty() {
        return false;
    }
    graph
        .outputs
        .iter()
        .any(|output| shape_outputs.contains(output))
        || graph.nodes.iter().any(|(_, node)| {
            node.inputs
                .iter()
                .flatten()
                .any(|input| shape_outputs.contains(input))
        })
}

/// Name a consumer that is *not* padded-capacity-safe, for attribution. Returns
/// `None` for a capacity-safe consumer so callers can filter and format in one
/// pass. Kept next to [`kernel_input_uses_padded_capacity`] so the allowlist and
/// the message that explains a decline cannot drift apart.
pub(super) fn describe_non_padded_consumer(node: &Node, input_index: usize) -> Option<String> {
    if kernel_input_uses_padded_capacity(node, input_index) {
        return None;
    }
    Some(format!("{}[input {input_index}]", describe_node(node)))
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
/// Broadcasting combiners that mix the mask *elementwise* with a score — `Add`,
/// and `Concat` appending a fixed-size extra column — are deliberately
/// **absent** from this unconditional set: whether they are safe depends on
/// where their *other* operand's length axis comes from, which
/// [`classify_mask_consumer`] decides case-by-case via
/// [`derives_from_kv_cache_growth`] rather than blessing every `Add`/`Concat`
/// here. GLM-5.2's indexer `Add` mixes the mask with a logical-width score
/// sourced from neither the mask nor a KV-cache append, so it is still
/// disqualified by that per-case check (freezing must never leak `max_len`
/// into a logical-width computation). Only ops that are *always* safe
/// regardless of their other operand's provenance are listed here; anything
/// else disqualifies the binding conservatively unless a narrower rule in
/// [`classify_mask_consumer`] applies.
pub(super) fn is_additive_mask_builder_op(node: &Node) -> bool {
    node.is_default_domain()
        && matches!(
            node.op_type.as_str(),
            "CumSum" | "Unsqueeze" | "Cast" | "GreaterOrEqual" | "And" | "Where" | "Slice" | "Sub"
        )
}

/// Whether `node`/`input_index` is the additive-mask input (input 3) of a
/// capacity-form `Attention`: a default-domain `Attention` whose KV cache
/// (inputs 4/5) is already bound at physical capacity — i.e. a present mask so
/// it derives the valid length from the mask frontier (see
/// [`kernel_input_uses_physical_capacity`]), in either the causal or non-causal
/// form. Such a node is *designed* to consume a physical-width additive mask, so
/// it is a valid leaf for the frozen-mask (padded-capacity) classification.
///
/// The KV cache inputs (`past_key` = input 4, `past_value` = input 5) must both
/// actually exist: `kernel_input_uses_physical_capacity` gates only on the mask
/// (input 3) presence, but the CUDA `Attention` kernel's fixed-capacity append
/// contract requires both past caches bound at physical capacity
/// (`standard_attention.rs`: `has_past_key`/`has_past_value` on inputs 4/5, and
/// `fixed_capacity_append` compares their capacity to the mask key width). A
/// masked `Attention` with only q/k/v/mask and no KV binding is NOT a
/// capacity-form leaf, so require both KV inputs here rather than blessing any
/// masked `Attention` as a valid cone terminus.
pub(super) fn is_capacity_form_attention_mask_input(node: &Node, input_index: usize) -> bool {
    input_index == 3
        && node.is_default_domain()
        && node.op_type == "Attention"
        && node.inputs.get(4).is_some_and(Option::is_some)
        && node.inputs.get(5).is_some_and(Option::is_some)
        && kernel_input_uses_physical_capacity(node, 4)
}

/// Domain of the native-only S3 capacity-write append (see
/// [`is_kv_cache_growth_concat`]'s second arm and
/// `kv_capacity_write_eligible_concats`). A CUDA-only load-time rewrite
/// replaces an eligible growth `Concat` with this op so the KV cache is
/// updated as an in-place capacity write instead of a growing concatenation,
/// which is what lets whole-step CUDA-graph capture stay shape-static across
/// decode steps. Kept as a bare literal (matching how `pkg.nxrt::IndexShare`
/// is already spelled independently in each crate that touches it) rather
/// than a cross-crate constant, since the rewrite and the kernel that
/// executes it live in different crates with no shared dependency edge.
pub(super) const KV_CAPACITY_APPEND_DOMAIN: &str = "pkg.nxrt";
/// Op type of the native-only S3 capacity-write append; see
/// [`KV_CAPACITY_APPEND_DOMAIN`].
pub(super) const KV_CAPACITY_APPEND_OP: &str = "KvCacheCapacityAppend";

/// Structural signature of a *decomposed*-attention KV-cache append: a
/// default-domain `Concat` whose growing operand is a graph **input** (the
/// persisted past cache) and whose result is a graph **output** (the
/// persisted present cache) — **or** the rewritten
/// [`KV_CAPACITY_APPEND_DOMAIN`]`::`[`KV_CAPACITY_APPEND_OP`] form of the same
/// node (same input/output role, plus a third `position_ids` operand; see
/// `kv_capacity_write_eligible_concats`). This recognizes the append node by
/// input/output *role* alone — not by name, axis, or tensor shape — so it
/// matches any decomposed self-attention export that grows its cache with a
/// plain `Concat` rather than an in-op cache (`Attention` /
/// `GroupQueryAttention` / `pkg.nxrt::IndexShare`), and continues to match
/// after the CUDA-only rewrite swaps the node for its capacity-write form —
/// otherwise the mask-cone walk's `Add`-rule provenance check
/// ([`derives_from_kv_cache_growth`]) would stop recognizing its own
/// rewritten output on the very graph the rewrite was proven safe for.
///
/// The KV-cache manager binds *both* ends of this edge — the past input and
/// the present output — at a fixed physical capacity regardless of node type
/// (see `dispatch.rs`'s "Fixed-capacity KV widening", which recognizes this
/// same predicate to keep the tracked present shape consistent with that
/// binding). A value whose length axis structurally derives from this node's
/// output (see [`derives_from_kv_cache_growth`]) is therefore exactly as
/// capacity-consistent as `present.*` itself, which is what lets a mask that
/// combines with such a value stay sound once frozen to physical width.
pub(super) fn is_kv_cache_growth_concat(graph: &Graph, node: &Node) -> bool {
    ((node.is_default_domain() && node.op_type == "Concat")
        || (node.domain == KV_CAPACITY_APPEND_DOMAIN && node.op_type == KV_CAPACITY_APPEND_OP))
        && node
            .inputs
            .first()
            .copied()
            .flatten()
            .is_some_and(|input| graph.inputs.contains(&input))
        && node
            .outputs
            .first()
            .is_some_and(|output| graph.outputs.contains(output))
}

/// The shape-relabelling/scaling op set that carries an operand's length axis
/// through unchanged, shared by both the backward provenance walk
/// ([`kv_cache_growth_concat_source`]) and the forward value-role walk
/// ([`forward_shape_preserving_matmul_consumers`]): `Transpose`, `Reshape`,
/// `Unsqueeze`, `Squeeze`, `Cast`, `Mul`, `Div`, `Slice`, `Expand`. One list,
/// walked in whichever direction the question needs, rather than a second
/// allowlist that could drift from this one.
fn is_shape_preserving_relabel_op(node: &Node) -> bool {
    node.is_default_domain()
        && matches!(
            node.op_type.as_str(),
            "Transpose"
                | "Reshape"
                | "Unsqueeze"
                | "Squeeze"
                | "Cast"
                | "Mul"
                | "Div"
                | "Slice"
                | "Expand"
        )
}

/// Bounded backward walk: does `value`'s length axis structurally derive from
/// an [`is_kv_cache_growth_concat`] node's output? Returns the source node so
/// callers that need to name *which* append it came from (the S3
/// capacity-write rewrite) do not have to re-walk to recover it.
///
/// Descends through [`is_shape_preserving_relabel_op`], following *every*
/// input, since for these ops no operand introduces a length axis foreign to
/// the one already being traced: an auxiliary shape/axes/scalar operand is
/// either a dead end or itself a legitimate `Shape(present.*)` read, which is
/// the same signal reached from the other side.
///
/// `MatMul` gets a narrower, semantics-derived rule: only input 1 is
/// followed, because ONNX defines a `MatMul` output's trailing axis as input
/// 1's trailing axis. Following input 0 as well (e.g. the query projection in
/// `Q @ Kᵀ`) would let an unrelated operand's provenance stand in for the one
/// that actually determines the result's length axis — restricting to input 1
/// is what keeps this a general structural fact about `MatMul`, not a
/// `Q`/`K`-specific convention.
pub(super) fn kv_cache_growth_concat_source(graph: &Graph, value: ValueId) -> Option<NodeId> {
    use std::collections::{HashMap, HashSet, VecDeque};

    let mut producer: HashMap<ValueId, NodeId> = HashMap::new();
    for (node_id, node) in graph.nodes.iter() {
        for out in &node.outputs {
            producer.insert(*out, node_id);
        }
    }

    let mut seen: HashSet<ValueId> = HashSet::new();
    let mut frontier: VecDeque<ValueId> = VecDeque::new();
    frontier.push_back(value);
    while let Some(current) = frontier.pop_front() {
        if !seen.insert(current) {
            continue;
        }
        let Some(&node_id) = producer.get(&current) else {
            continue;
        };
        let node = graph.node(node_id);
        if is_kv_cache_growth_concat(graph, node) {
            return Some(node_id);
        }
        if node.is_default_domain() && node.op_type == "MatMul" {
            if let Some(Some(rhs)) = node.inputs.get(1) {
                frontier.push_back(*rhs);
            }
            continue;
        }
        if is_shape_preserving_relabel_op(node) {
            for input in node.inputs.iter().flatten() {
                frontier.push_back(*input);
            }
        }
    }
    None
}

/// Whether `value`'s length axis structurally derives from an
/// [`is_kv_cache_growth_concat`] node's output. See
/// [`kv_cache_growth_concat_source`] for the walk and its rationale.
pub(super) fn derives_from_kv_cache_growth(graph: &Graph, value: ValueId) -> bool {
    kv_cache_growth_concat_source(graph, value).is_some()
}

/// Forward walk from a proven-safe `Softmax` sink's output, through
/// [`is_shape_preserving_relabel_op`], collecting every `MatMul` node it
/// reaches and which input slot the traced value occupies there.
///
/// This is the value-role counterpart of [`kv_cache_growth_concat_source`]'s
/// backward walk: the same op set, walked forward from the softmax output
/// instead of backward from a score operand, because a proven-safe softmax's
/// exact-zero padded probabilities are what make the *other* `MatMul` operand
/// (the "V" of `probs @ V`) safe regardless of its padded content — so the
/// question here is "where does this probability value get multiplied",
/// which is naturally asked forward. `MatMul` is a traversal terminus (not
/// followed through) because the identity of "this is still the mask's
/// length axis" ends at the matmul; the caller inspects the *other* operand.
fn forward_shape_preserving_matmul_consumers(
    graph: &Graph,
    start: ValueId,
) -> Vec<(NodeId, usize)> {
    use std::collections::{HashSet, VecDeque};

    let mut consumers: std::collections::HashMap<ValueId, Vec<(NodeId, usize)>> =
        std::collections::HashMap::new();
    for (node_id, node) in graph.nodes.iter() {
        for (slot, value) in node.inputs.iter().enumerate() {
            if let Some(vid) = value {
                consumers.entry(*vid).or_default().push((node_id, slot));
            }
        }
    }

    let mut seen: HashSet<ValueId> = HashSet::new();
    let mut frontier: VecDeque<ValueId> = VecDeque::new();
    frontier.push_back(start);
    let mut hits = Vec::new();
    while let Some(value) = frontier.pop_front() {
        if !seen.insert(value) {
            continue;
        }
        for &(node_id, slot) in consumers.get(&value).map_or(&[][..], Vec::as_slice) {
            let node = graph.node(node_id);
            if node.is_default_domain() && node.op_type == "MatMul" {
                hits.push((node_id, slot));
                continue;
            }
            if is_shape_preserving_relabel_op(node) {
                frontier.extend(node.outputs.iter().copied());
            }
        }
    }
    hits
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
    mask_cone_rejection(graph, mask, ShapeConsumptionPolicy::Disqualify).is_none()
}

/// Why the mask cone failed to classify as padded-capacity-safe, or `None` when
/// it succeeded. The predicates answer yes/no; bring-up needs the reason, so the
/// walk records it rather than leaving it to be re-derived from the graph.
pub(super) fn mask_cone_rejection(
    graph: &Graph,
    mask: ValueId,
    shape_policy: ShapeConsumptionPolicy,
) -> Option<String> {
    mask_binding_feeds_capacity_form_attention_impl(graph, mask, shape_policy).err()
}

/// Whether `mask` feeds *only* the standard additive causal-mask builder cone
/// terminating at capacity-form `Attention` leaves, **allowing** the builder's
/// window arithmetic to consume `Shape(mask)` (the DeepSeek-V2-Lite shape, which
/// [`mask_binding_feeds_capacity_form_attention`] disqualifies from *static*
/// freezing because `Shape(mask)` leaks the padded width into the multi-token
/// prefill query-position `Slice`).
///
/// This is the weaker "decode-freeze-safe" predicate. It holds exactly when the
/// mask's only consumers are the additive causal-mask builder + capacity-form
/// `Attention`, so a **single-token** decode step (`q_seq == 1`) can freeze the
/// mask to physical capacity even when prefill must expose the logical length:
///
/// - The query window is `Slice(CumSum(mask), start = Shape(mask) - q_seq,
///   end = Shape(mask))`. At `q_seq == 1` this is `[CumSum(mask)[Shape-1]]`,
///   which equals `total_len` for **any** `Shape >= total_len` because the
///   `CumSum` prefix-sum plateaus at `total_len` across the zero-padding —
///   freezing `Shape` to `max_len` yields the identical single query position.
/// - The key positions and padding branch are width-invariant on the valid
///   prefix and forced to `-inf` on the padded suffix, exactly as in the frozen
///   capacity-form case.
///
/// Broadcasting logical-width combiners (e.g. GLM-5.2's indexer `Add`) are still
/// excluded via [`is_additive_mask_builder_op`], so such masks are *not*
/// decode-freeze-safe and keep exposing their logical length every step.
pub(super) fn mask_binding_feeds_additive_causal_builder(graph: &Graph, mask: ValueId) -> bool {
    mask_binding_feeds_capacity_form_attention_impl(graph, mask, ShapeConsumptionPolicy::Allow)
        .is_ok()
}

/// Every [`is_kv_cache_growth_concat`] node in `graph` structurally proven
/// safe to rewrite from a growing concatenation into an in-place
/// [`KV_CAPACITY_APPEND_OP`] capacity write.
///
/// Reuses the exact decode-freeze-safe proof
/// [`mask_binding_feeds_additive_causal_builder`] already establishes (the
/// `ShapeConsumptionPolicy::Allow` walk): the rewrite only ever fires for a
/// single-token capture-eligible decode step, the same regime that proof is
/// scoped to. Two roles fall out of that one proof (see
/// `mask_binding_feeds_capacity_form_attention_impl`'s doc comment):
///
/// - **Score role**: the append feeds the non-mask operand of an `Add` the
///   walk classified `Propagate`.
/// - **Value role**: the append feeds one operand of a `MatMul` whose other
///   operand forward-derives from a `Softmax` the walk classified `Sink` —
///   safe because the sink's exact-zero padded probabilities already make
///   the padded rows of that operand irrelevant regardless of content.
///
/// Iterates every graph input as a mask candidate rather than assuming a
/// fixed binding-order convention (e.g. "bindings[0] is the mask"), and unions
/// the results — cheap at load-time graph sizes, and keeps this analysis
/// decoupled from how the engine layer happens to order its device bindings.
pub(super) fn kv_capacity_write_eligible_concats(graph: &Graph) -> HashSet<NodeId> {
    let mut eligible = HashSet::new();
    for &input in &graph.inputs {
        if let Ok(concats) = mask_binding_feeds_capacity_form_attention_impl(
            graph,
            input,
            ShapeConsumptionPolicy::Allow,
        ) {
            eligible.extend(concats);
        }
    }
    eligible
}

/// How the mask-cone walk treats a `Shape(mask)` consumer whose output is itself
/// consumed (i.e. `Shape` is not a dead-end physical-extent read).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ShapeConsumptionPolicy {
    /// Disqualify the binding from *static* padded-capacity freezing: a consumed
    /// `Shape(mask)` leaks the padded width into width-sensitive arithmetic
    /// (the multi-token prefill query-position window), which must see the
    /// logical length. Drives `expose_logical_input_shape`.
    Disqualify,
    /// Allow a consumed `Shape(mask)`: the cone is still the additive causal-mask
    /// builder, so a single-token decode step remains freeze-safe (the query
    /// window saturates). Drives the decode-freeze-safe classification.
    Allow,
}

/// What a single consumer edge does to the mask's length axis.
///
/// Freezing the mask to physical capacity is a **uniform substitution**: replace
/// the logical length `L` with `max_len` everywhere the mask's length axis
/// appears. That is sound exactly when no consumer sources that axis from
/// somewhere other than the mask, and when the padded lanes are neutralised
/// before reaching anything that is not padding-aware.
///
/// Every rule below is one of those two questions. Classifying the edge once,
/// here, is what keeps the walk from accumulating a separate mechanism per
/// model — the op allowlist, the `Shape` policy and the `Expand` shape-operand
/// check are all the *same* question asked about different operands.
enum ConsumerRole {
    /// A padding-aware sink: capacity-form `Attention` neutralises the padded
    /// keys itself, so the substitution terminates safely here. A default-domain
    /// `Softmax` over the mask's own (last) axis is the decomposed-attention
    /// analogue: `-inf` mask lanes become exact-zero probability, which is the
    /// same padding-neutralization `Attention` performs internally, just spelled
    /// out as an explicit node instead of happening inside a fused kernel.
    Sink,
    /// Padding-invariant read that does not propagate: zero padding contributes
    /// nothing (`ReduceSum`), or the physical extent is read and discarded
    /// (`Shape` as a dead end).
    InvariantLeaf,
    /// The op maps mask lanes to mask lanes without importing the length axis
    /// from another source, so the substitution stays uniform through it.
    Propagate,
    /// The op sources the mask's length axis from somewhere the substitution
    /// does not reach, so freezing would compare `max_len` against a logical
    /// length. Carries the explanation.
    Mixes(String),
}

fn classify_mask_consumer(
    graph: &Graph,
    node: &Node,
    slot: usize,
    mask: ValueId,
    consumers: &std::collections::HashMap<ValueId, Vec<(NodeId, usize)>>,
    shape_policy: ShapeConsumptionPolicy,
) -> ConsumerRole {
    if is_capacity_form_attention_mask_input(node, slot) {
        return ConsumerRole::Sink;
    }
    if kernel_input_uses_padded_capacity(node, slot) {
        // `ReduceSum` is unconditionally padding-invariant. `Shape` returns the
        // *physical* width, so it is only a safe leaf while its result is a dead
        // end; once consumed, the padded `max_len` enters width arithmetic. That
        // is fatal for multi-token prefill (the query-position window
        // `Slice(CumSum(mask), Shape-q_seq, Shape)` selects the wrong rows), but
        // harmless at `q_seq == 1`, where the window saturates to the same
        // position for any width >= the valid length — hence the policy split.
        if node.op_type == "Shape" && shape_policy == ShapeConsumptionPolicy::Disqualify {
            let consumed = node.outputs.iter().any(|out| consumers.contains_key(out));
            if consumed {
                return ConsumerRole::Mixes(format!(
                    "{} output is consumed, so the padded max_len would leak into \
                     width-sensitive arithmetic (the prefill query-position window)",
                    describe_node(node)
                ));
            }
        }
        return ConsumerRole::InvariantLeaf;
    }
    // `Expand` sets its output width from the *shape operand*, not from the mask.
    // The substitution therefore stays uniform only when that operand is itself
    // mask-derived: freezing the mask makes `Shape(mask)` report `max_len` too,
    // and the mask is broadcast to exactly its own frozen width. Sourced from
    // anywhere else, a frozen `max_len`-wide mask could not broadcast at all.
    if node.is_default_domain() && node.op_type == "Expand" && slot == 0 {
        return if expand_target_is_mask_derived(graph, node, mask) {
            ConsumerRole::Propagate
        } else {
            ConsumerRole::Mixes(format!(
                "{}[input {slot}] broadcasts the mask to a target shape not derived from \
                 Shape(mask), so the frozen width would not match the target",
                describe_node(node)
            ))
        };
    }
    if is_additive_mask_builder_op(node) {
        return ConsumerRole::Propagate;
    }
    // A decomposed attention score combines the additive mask with the raw
    // `Q @ Kᵀ` score via `Add` rather than an in-op `Attention`/`GroupQuery
    // Attention` cache. The substitution stays uniform through this `Add`
    // exactly when the *other* operand's length axis is itself capacity-
    // consistent — i.e. it structurally derives from the same KV-cache append
    // that the present-KV widening in `dispatch.rs` already treats as physical
    // capacity (see [`derives_from_kv_cache_growth`]). GLM-5.2's indexer `Add`
    // combines the mask with a logical-width score sourced from neither the
    // mask nor a KV-cache append, so it still falls to the `Mixes` branch below.
    if node.is_default_domain() && node.op_type == "Add" && node.inputs.len() == 2 {
        let other = node.inputs[1 - slot];
        return if other.is_some_and(|value| derives_from_kv_cache_growth(graph, value)) {
            ConsumerRole::Propagate
        } else {
            ConsumerRole::Mixes(format!(
                "{}[input {slot}] adds the mask to an operand that does not itself derive \
                 from a KV-cache append, so freezing would compare max_len against a value \
                 still at its logical length",
                describe_node(node)
            ))
        };
    }
    // A decomposed attention score may append a fixed-size extra column (e.g. a
    // per-head learned "attention sink" bias) onto the masked score *before* the
    // softmax that neutralises padding, via `Concat`. `Concat` only appends —
    // it never reorders or truncates the mask-derived operand — so the
    // substitution stays uniform as long as every *other* operand is
    // structurally independent of it: neither derived from the mask cone itself
    // nor from a KV-cache append (the two things whose size actually changes
    // under the substitution). A `Concat` operand that fails that independence
    // check would silently change size between the logical and frozen forms,
    // so it disqualifies the binding instead.
    if node.is_default_domain() && node.op_type == "Concat" {
        let independent = node.inputs.iter().enumerate().all(|(i, input)| {
            i == slot || input.is_none_or(|value| !derives_from_kv_cache_growth(graph, value))
        });
        return if independent {
            ConsumerRole::Propagate
        } else {
            ConsumerRole::Mixes(format!(
                "{}[input {slot}] concatenates the mask-derived value with an operand that \
                 also derives from a KV-cache append, so the two would not stay the same \
                 size once the mask is frozen",
                describe_node(node)
            ))
        };
    }
    // A default-domain `Softmax` that normalises over the mask's own (last)
    // axis is the decomposed-attention neutralisation point: `-inf` mask lanes
    // become exact-zero probability there, so nothing needs to be tracked past
    // it (mirrors capacity-form `Attention`'s internal padding treatment). A
    // `Softmax` over any other axis would not neutralise the length axis at
    // all, so it is deliberately excluded and falls to the catch-all below.
    //
    // The *default* when `axis` is absent is opset-dependent, not just a
    // spelling convenience: opset >= 13 defaults to `-1` with plain per-axis
    // normalization (identical to an explicit `-1`), but opset <= 12 defaults
    // to `1` with "coerce to 2D" semantics that merge every dim from `axis`
    // onward into one normalization group — for a rank > 2 tensor that mixes
    // in axes *other than* the length axis, so it would not purely neutralise
    // it. Since this call site has no tensor rank to confirm axis == rank - 1
    // in that case, an absent attribute is only ever treated as safe under
    // opset >= 13, where the default is unambiguously the last axis alone.
    if node.is_default_domain() && node.op_type == "Softmax" {
        let axis = node.attr("axis").and_then(Attribute::as_int);
        let normalizes_length_axis = match axis {
            Some(-1) => true,
            Some(_) => false,
            None => effective_opset(graph, node) >= 13,
        };
        return if normalizes_length_axis {
            ConsumerRole::Sink
        } else {
            ConsumerRole::Mixes(format!(
                "{} normalizes over an axis other than the last (or defaults to opset <= 12's \
                 axis=1 coerce-to-2D semantics, which is not confirmed to be the last axis \
                 alone), so it would not neutralise the mask's padded lanes before a \
                 non-padding-aware consumer",
                describe_node(node)
            ))
        };
    }
    // Everything else combines the mask with a value whose extent came from
    // elsewhere, or consumes it in a shape not recognised above.
    ConsumerRole::Mixes(format!(
        "{}[input {slot}] is outside the additive causal-mask builder set, so it \
         sources the mask length axis from another value",
        describe_node(node)
    ))
}

/// Whether `expand`'s shape operand (input 1) is derived from `Shape(mask)`.
///
/// Only the mask's own length axis has to come from the mask; the other axes of
/// the target (batch, query length — typically `Shape(input_ids)`) do not carry
/// the mask length and are unaffected by the substitution.
fn expand_target_is_mask_derived(graph: &Graph, expand: &Node, mask: ValueId) -> bool {
    use std::collections::{HashSet, VecDeque};

    let mut producer: std::collections::HashMap<ValueId, NodeId> = std::collections::HashMap::new();
    for (node_id, node) in graph.nodes.iter() {
        for out in &node.outputs {
            producer.insert(*out, node_id);
        }
    }
    let Some(Some(target)) = expand.inputs.get(1).copied() else {
        return false;
    };
    let mut seen: HashSet<ValueId> = HashSet::new();
    let mut frontier: VecDeque<ValueId> = VecDeque::new();
    frontier.push_back(target);
    while let Some(value) = frontier.pop_front() {
        if !seen.insert(value) {
            continue;
        }
        let Some(&node_id) = producer.get(&value) else {
            continue;
        };
        let node = graph.node(node_id);
        if node.op_type == "Shape" && node.inputs.first().copied().flatten() == Some(mask) {
            return true;
        }
        for input in node.inputs.iter().flatten() {
            frontier.push_back(*input);
        }
    }
    false
}

/// Walks the mask cone and, on success, also returns every
/// [`is_kv_cache_growth_concat`] node the walk structurally proved safe to
/// rewrite into an in-place capacity write (see
/// [`kv_capacity_write_eligible_concats`]). Kept as one walk rather than a
/// separate graph-wide scan: the K-role (score) and V-role (value) concats
/// are byproducts of the *same* padding-neutralization proof this function
/// already computes for the mask itself, not an independent question.
fn mask_binding_feeds_capacity_form_attention_impl(
    graph: &Graph,
    mask: ValueId,
    shape_policy: ShapeConsumptionPolicy,
) -> std::result::Result<HashSet<NodeId>, String> {
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
    // Score-role (K): an `Add` classified `Propagate` proves its *other*
    // operand safe by definition (see `classify_mask_consumer`'s `Add` rule) —
    // recover which append that operand derives from, if any.
    let mut score_role_concats: HashSet<NodeId> = HashSet::new();
    // Outputs of every `Softmax` this walk classified `Sink`, for the value-role
    // forward search below.
    let mut sink_softmax_outputs: Vec<ValueId> = Vec::new();
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
            return Err(format!(
                "mask-derived value {value:?} escapes as a graph output, so freezing the mask \
                 to physical width would leak the padded max_len to whatever consumes it"
            ));
        }
        for &(node_id, slot) in consumers.get(&value).map_or(&[][..], Vec::as_slice) {
            let node = graph.node(node_id);
            match classify_mask_consumer(graph, node, slot, mask, &consumers, shape_policy) {
                ConsumerRole::Sink => {
                    reached_attention = true;
                    if node.is_default_domain() && node.op_type == "Softmax" {
                        sink_softmax_outputs.extend(node.outputs.iter().copied());
                    }
                }
                ConsumerRole::InvariantLeaf => {}
                ConsumerRole::Propagate => {
                    if node.is_default_domain() && node.op_type == "Add" && node.inputs.len() == 2 {
                        let other = node.inputs[1 - slot];
                        if let Some(concat_id) =
                            other.and_then(|value| kv_cache_growth_concat_source(graph, value))
                        {
                            score_role_concats.insert(concat_id);
                        }
                    }
                    frontier.extend(node.outputs.iter().copied());
                }
                ConsumerRole::Mixes(reason) => return Err(reason),
            }
        }
    }
    if !reached_attention {
        return Err(String::from(
            "the mask cone never reached a capacity-form Attention mask input",
        ));
    }
    let mut eligible = score_role_concats;
    for output in sink_softmax_outputs {
        for (matmul_id, probs_slot) in forward_shape_preserving_matmul_consumers(graph, output) {
            let matmul = graph.node(matmul_id);
            if matmul.inputs.len() != 2 {
                continue;
            }
            let Some(other) = matmul.inputs[1 - probs_slot] else {
                continue;
            };
            if let Some(concat_id) = kv_cache_growth_concat_source(graph, other) {
                eligible.insert(concat_id);
            }
        }
    }
    Ok(eligible)
}

/// `name(OpType)`, or `<unnamed>(OpType)` — the shared spelling for every node
/// named in a capture-decline explanation.
fn describe_node(node: &Node) -> String {
    let name = if node.name.is_empty() {
        "<unnamed>"
    } else {
        node.name.as_str()
    };
    format!("{name}({})", node.op_type)
}

#[cfg(test)]
mod gather_tests {
    use super::*;
    use onnx_runtime_ir::static_shape;

    #[test]
    fn shape_capacity_is_not_assumed_when_the_result_is_observed() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let input = graph.create_named_value("mask", DataType::Int64, static_shape([1, 8]));
        graph.add_input(input);
        let shape = graph.create_named_value("shape", DataType::Int64, static_shape([2]));
        graph.insert_node(Node::new(
            NodeId(0),
            "Shape",
            vec![Some(input)],
            vec![shape],
        ));
        assert!(!binding_shape_result_is_observed(&graph, input));

        let observed = graph.create_named_value("observed", DataType::Int64, static_shape([2]));
        graph.insert_node(Node::new(
            NodeId(1),
            "Identity",
            vec![Some(shape)],
            vec![observed],
        ));
        graph.add_output(observed);
        assert!(binding_shape_result_is_observed(&graph, input));
    }

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
