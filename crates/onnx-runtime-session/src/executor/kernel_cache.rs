use super::*;

use std::sync::atomic::{AtomicU64, Ordering};

/// Global counters for kernel pre-binding path reachability.
/// Incremented on the pre-bound fast path; read by tests to prove the path fires.
#[cfg(test)]
pub(crate) static PREBIND_FAST_PATH_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static PREBIND_FALLBACK_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Cache key for a compiled kernel (§11.1). Keyed by the concrete node and its
/// **resolved** (concrete) input shapes: attributes are fixed per node, so this
/// is correct, and the shape component makes it *shape-keyed* — a re-run with
/// the same resolved shapes hits, a different shape (e.g. a new batch/seq)
/// misses and re-compiles. This preserves Chew's guarantee: a kernel is never
/// reused for a shape it was not compiled for.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(super) struct KernelKey {
    pub(super) node: u32,
    pub(super) shapes: Vec<Vec<usize>>,
}

impl KernelKey {
    /// Check whether the cached key matches the current input shapes **without
    /// allocating**. The caller's `input_shapes` is a `&[Vec<usize>]` (the reused
    /// scratch), so this comparison is a flat slice-of-slices equality.
    #[inline]
    pub(super) fn matches_shapes(&self, input_shapes: &[Vec<usize>]) -> bool {
        self.shapes.len() == input_shapes.len()
            && self
                .shapes
                .iter()
                .zip(input_shapes.iter())
                .all(|(a, b)| a.as_slice() == b.as_slice())
    }
}

/// The past-KV-cache **input** indices for a recognized attention node, if any.
///
/// These are the tensors whose sequence axis GROWS each decode step (the KV
/// cache), as opposed to a fixed-capacity recurrent state. The indices mirror
/// the authoritative structural list in
/// [`kernel_input_uses_physical_capacity`](super::geometry::kernel_input_uses_physical_capacity):
/// GQA past_key/value = inputs 3,4; default-domain `Attention` = inputs 4,5;
/// `pkg.nxrt::IndexShare` = inputs 3,4. Any other op returns `None`.
fn attention_past_kv_input_indices(node: &Node) -> Option<[usize; 2]> {
    if node.domain == "com.microsoft" && node.op_type == "GroupQueryAttention" {
        return Some([3, 4]);
    }
    if node.is_default_domain() && node.op_type == "Attention" {
        return Some([4, 5]);
    }
    if node.domain == "pkg.nxrt" && node.op_type == "IndexShare" {
        return Some([3, 4]);
    }
    None
}

/// If `shape` is a KV-cache tensor, the [`SymbolId`] on its **growing** sequence
/// axis. The KV layout is `[batch, kv_heads, sequence, head_dim]`, so the
/// sequence axis is the **penultimate** dim — this is the exact structural
/// convention used by `is_recurrent_state_shape` (a static penultimate axis is a
/// fixed recurrent state, a symbolic one is a growable KV cache). Returns `None`
/// for a static penultimate axis (recurrent state) or a rank < 2 shape.
fn kv_growing_symbol(shape: &Shape) -> Option<SymbolId> {
    if shape.len() < 2 {
        return None;
    }
    match shape[shape.len() - 2] {
        Dim::Symbolic(sym) => Some(sym),
        Dim::Static(_) => None,
    }
}

/// Structurally identify the set of GROWING symbols at build/graph-load time.
///
/// A growing symbol is one that lives on the sequence axis of an attention KV
/// cache (`past`/`present` key/value) — it increments each decode step, so any
/// buffer sized by it has a different extent every replay and MUST stay eager.
/// The QUERY `sequence_len` symbol is deliberately NOT collected here: it lives
/// on the query/hidden tensor, never on a KV slot, and the decode capture region
/// pins it constant (=1) across replays. This distinction is the whole point of
/// the classifier: GDN pointwise ops carry only batch/query-seq/heads (all
/// pinned) and no KV-length symbol → they become capture-eligible, while any op
/// carrying a KV-length symbol stays eager.
///
/// We gather symbols from BOTH the past-KV **inputs** and the present-KV
/// **outputs** (outputs[1]=present_key, outputs[2]=present_value) of every
/// recognized attention node. The loader interns same-named dim-params to one
/// `SymbolId` across layers, so the resulting set stays small.
pub(super) fn compute_capture_growing_symbols(graph: &Graph) -> HashSet<SymbolId> {
    let mut growing = HashSet::new();
    for node in graph.nodes.values() {
        let Some(past_inputs) = attention_past_kv_input_indices(node) else {
            continue;
        };
        for idx in past_inputs {
            if let Some(Some(vid)) = node.inputs.get(idx).copied()
                && let Some(value) = graph.try_value(vid)
                && let Some(sym) = kv_growing_symbol(&value.shape)
            {
                growing.insert(sym);
            }
        }
        // present_key = outputs[1], present_value = outputs[2] (see
        // dynamic_shapes.rs GQA present handling: `[query[0], kv_heads,
        // present_sequence, head_dim]`, seq axis = penultimate).
        for idx in [1usize, 2usize] {
            if let Some(&vid) = node.outputs.get(idx)
                && let Some(value) = graph.try_value(vid)
                && let Some(sym) = kv_growing_symbol(&value.shape)
            {
                growing.insert(sym);
            }
        }
    }
    if std::env::var("ONNX_GENAI_LOG_GROWING_SYMBOLS").is_ok() {
        eprintln!(
            "[onnx-genai-capture] build-time growing-symbol set: {} symbol(s): {:?}",
            growing.len(),
            growing
        );
    }
    growing
}

/// Whether `node` is capture-eligible under the **build-time growing-symbol**
/// classifier: every output dim must be free of any GROWING (KV-length) symbol.
///
/// This replaces the strict all-`Dim::Static` rule (fully-static output shapes),
/// which was too strict: it treats a PINNED symbolic dim (batch=1, query
/// seq_len=1, heads) the same as a GROWING one and so kept every GDN pointwise
/// op eager. Keying off the growing SET instead of dim staticness admits
/// `[Symbolic(batch), Symbolic(seq=1), Static(16), Static(128)]` (both symbolic
/// dims pinned) while correctly keeping anything with a KV-length dim eager.
/// This also resolves the heads-vs-tokens ambiguity structurally:
/// `[Symbolic(heads), Static(128)]` with `heads ∉ growing` → capturable;
/// `[Symbolic(seq_kv), Static(128)]` with `seq_kv ∈ growing` → eager.
///
/// When `growing` is empty (a pure-recurrent graph with no attention KV cache),
/// [`shape_references_any`] returns false for every op → all pointwise ops are
/// capture-eligible, which is correct: nothing grows step-to-step. Nodes with no
/// outputs are conservatively treated as not seq-independent.
pub(super) fn node_capture_seq_independent(
    graph: &Graph,
    node: &Node,
    growing: &HashSet<SymbolId>,
) -> bool {
    !node.outputs.is_empty()
        && node.outputs.iter().all(|&vid| {
            graph
                .try_value(vid)
                .is_some_and(|value| !shape_references_any(&value.shape, growing))
        })
}

/// Observable kernel-cache statistics (§11.1) — enough to prove reuse in tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Distinct compiled entries currently held.
    pub entries: usize,
    /// Lookups served from an existing entry.
    pub hits: u64,
    /// Lookups that compiled a new kernel.
    pub misses: u64,
    /// Lookups served via the pre-bound fast path (zero-alloc).
    pub prebind_hits: u64,
}

/// Shape-keyed kernel cache (§11.1). Owns the compiled kernels for the session.
#[derive(Default)]
pub(crate) struct KernelCache {
    pub(super) entries: HashMap<KernelKey, Box<dyn onnx_runtime_ep_api::Kernel>>,
    pub(super) hits: u64,
    pub(super) misses: u64,
    pub(super) prebind_hits: AtomicU64,
}

impl KernelCache {
    pub(super) fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            prebind_hits: self.prebind_hits.load(Ordering::Relaxed),
        }
    }

    /// Zero-allocation kernel lookup via a pre-stored binding key.
    ///
    /// Returns `Some` when `binding` shapes match the current `input_shapes`
    /// and the compiled kernel is present in the cache. Returns `None` on any
    /// mismatch — caller falls through to `get_or_create`. This is the
    /// **pre-bound fast path**: during steady-state decode (fixed shapes), it
    /// replaces the per-token HashMap-key allocation with a single
    /// pointer chase + slice comparison.
    #[inline]
    pub(super) fn get_prebound<'a>(
        &'a self,
        binding: &KernelKey,
        input_shapes: &[Vec<usize>],
    ) -> Option<&'a dyn onnx_runtime_ep_api::Kernel> {
        if !binding.matches_shapes(input_shapes) {
            return None;
        }
        let kernel = self.entries.get(binding)?.as_ref();
        self.prebind_hits.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        PREBIND_FAST_PATH_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(kernel)
    }

    /// Return the cached kernel for `(node, resolved_input_shapes)`, verifying
    /// EP support and compiling+inserting it on a miss. Also returns the
    /// [`KernelKey`] so the caller can store it as a pre-binding for future
    /// zero-alloc lookups.
    // Each argument is an independent part of the kernel-cache key or the EP
    // contract; bundling them into a context struct is tracked separately
    // (Dallas #5, kernel-dispatch decomposition).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn get_or_create(
        &mut self,
        node_id: NodeId,
        node: &Node,
        input_shapes: &[Vec<usize>],
        input_dtypes: &[DataType],
        constant_inputs: &[bool],
        opset: u64,
        capture_seq_independent: bool,
        ep: &dyn ExecutionProvider,
    ) -> Result<(&dyn onnx_runtime_ep_api::Kernel, KernelKey)> {
        let key = KernelKey {
            node: node_id.0,
            shapes: input_shapes.to_vec(),
        };
        if self.entries.contains_key(&key) {
            self.hits += 1;
        } else {
            // Verify the EP claims this op at these concrete shapes/layouts
            // before compiling — same gate the static path used at build.
            let shape_dims: Vec<Shape> = input_shapes
                .iter()
                .map(|s| s.iter().map(|&d| Dim::Static(d)).collect())
                .collect();
            let layouts = vec![TensorLayout::contiguous(); input_shapes.len()];
            if let KernelMatch::Unsupported { reason } =
                ep.supports_op(node, opset, &shape_dims, input_dtypes, &layouts)
            {
                return Err(SessionError::unsupported_op(
                    node,
                    node_id,
                    opset,
                    ep.name(),
                    reason,
                ));
            }
            let mut kernel = match ep.get_kernel(node, input_shapes, opset) {
                Ok(kernel) => kernel,
                Err(EpError::NoEpForOp {
                    domain,
                    op_type,
                    opset,
                }) => {
                    // Opset-aware claims should make this unreachable. Preserve
                    // the actionable diagnostic if an EP's claim drifts.
                    return Err(SessionError::unsupported_op(
                        node,
                        node_id,
                        opset,
                        ep.name(),
                        format!(
                            "no handler for {domain}::{op_type} at opset {opset} — add a claim+handler"
                        ),
                    ));
                }
                Err(error) => return Err(error.into()),
            };
            kernel.set_constant_inputs(constant_inputs);
            kernel.set_capture_seq_independent(capture_seq_independent);
            self.entries.insert(key.clone(), kernel);
            self.misses += 1;
        }
        #[cfg(test)]
        PREBIND_FALLBACK_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let kernel_ref = self.entries.get(&key).expect("just inserted").as_ref();
        Ok((kernel_ref, key))
    }
}
