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

/// The KV-cache I/O **slot indices** of a recognized stateful attention node.
///
/// `.past_inputs` are the input tensors whose sequence axis GROWS each decode
/// step (the KV cache), as opposed to a fixed-capacity recurrent state;
/// `.present_outputs` are the corresponding grown outputs. The indices mirror
/// the authoritative structural list in
/// [`kernel_input_uses_physical_capacity`](super::geometry::kernel_input_uses_physical_capacity):
/// GQA past_key/value = inputs 3,4; default-domain `Attention` = inputs 4,5;
/// `pkg.nxrt::IndexShare` = inputs 3,4. `present_key`/`present_value` are the
/// GQA-family `outputs[1]`/`outputs[2]`.
///
/// `pkg.nxrt`/`com.microsoft`::`CompressedSparseAttention` (CSA) has no separate
/// past-KV inputs (its cache is stateful device state); instead its
/// total-sequence-length-derived cache records land on `outputs[1]`/`outputs[3]`
/// (`[query[0], records, width]`, penultimate = `records`). Dynamic ratio-4 CSA
/// additionally mints a fresh, growing `selections` symbol on the **last** axis
/// of `outputs[5]` (`[query[0], index_heads, query_seq, selections]`); that axis
/// is captured via [`KvCacheSlots::last_axis_outputs`], since it is the trailing
/// (not penultimate) dim. See
/// `onnx-runtime-shape-inference/src/handlers/custom_ops.rs::compressed_sparse_attention`.
///
/// This op list is one of two complementary growing-symbol sources (see
/// [`compute_capture_growing_symbols`]); an attention variant not listed here is
/// still covered by the generic declared-KV-I/O scan (any `past…`/`present…`
/// rank-4 boundary tensor), so a missing entry degrades gracefully rather than
/// silently admitting a growing op.
#[derive(Default)]
struct KvCacheSlots {
    /// Inputs whose penultimate (sequence) axis grows each decode step.
    past_inputs: &'static [usize],
    /// Outputs whose penultimate (sequence/record) axis grows each decode step.
    present_outputs: &'static [usize],
    /// Outputs whose **last** axis is a growing symbol (CSA `selections`).
    last_axis_outputs: &'static [usize],
}

fn attention_kv_cache_slots(node: &Node) -> Option<KvCacheSlots> {
    if node.domain == "com.microsoft" && node.op_type == "GroupQueryAttention" {
        return Some(KvCacheSlots {
            past_inputs: &[3, 4],
            present_outputs: &[1, 2],
            last_axis_outputs: &[],
        });
    }
    if node.is_default_domain() && node.op_type == "Attention" {
        return Some(KvCacheSlots {
            past_inputs: &[4, 5],
            present_outputs: &[1, 2],
            last_axis_outputs: &[],
        });
    }
    if node.domain == "pkg.nxrt" && node.op_type == "IndexShare" {
        return Some(KvCacheSlots {
            past_inputs: &[3, 4],
            present_outputs: &[1, 2],
            last_axis_outputs: &[],
        });
    }
    if (node.domain == "pkg.nxrt" || node.domain == "com.microsoft")
        && node.op_type == "CompressedSparseAttention"
    {
        // CSA cache records (derived from total_sequence_length) live on the
        // penultimate axis of outputs 1 and 3; it has no past-KV inputs. The
        // ratio-4 variant additionally mints a growing `selections` symbol on
        // the LAST axis of output 5 (`[query[0], index_heads, query_seq,
        // selections]`), which the penultimate scan would miss.
        return Some(KvCacheSlots {
            past_inputs: &[],
            present_outputs: &[1, 3],
            last_axis_outputs: &[5],
        });
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

/// If `shape`'s **last** axis is symbolic, its [`SymbolId`]. Used for the CSA
/// ratio-4 `selections` axis, which is the trailing (not penultimate) dim and so
/// is not covered by [`kv_growing_symbol`].
fn last_axis_growing_symbol(shape: &Shape) -> Option<SymbolId> {
    match shape.last()? {
        Dim::Symbolic(sym) => Some(*sym),
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
/// Symbols are gathered from THREE complementary sources so coverage does not
/// depend on a fixed op allowlist (the finding-2 gap):
///   1. The past-KV **inputs** and present-KV **outputs** of every recognized
///      stateful attention node ([`attention_kv_cache_slots`]: GQA, default
///      `Attention`, `IndexShare`, and `CompressedSparseAttention`), including
///      CSA ratio-4's growing `selections` symbol on the **last** axis of
///      output 5 ([`KvCacheSlots::last_axis_outputs`]).
///   2. A GENERIC scan of the model's DECLARED past/present KV I/O — every graph
///      input named `past…`/output named `present…` with a rank-4 KV layout
///      `[batch, kv_heads, sequence, head_dim]` whose penultimate (sequence) axis
///      is symbolic. This catches an unrecognized attention variant's KV boundary
///      tensor without minting a fresh op entry, while the rank-4 guard excludes
///      fixed-capacity recurrent/conv states (rank 3, or a static penultimate).
///   3. The EQUIVALENCE-CLASS CLOSURE of (1)∪(2) under shape inference's symbol
///      unification. Broadcast inference replaces two distinct symbolic dims with
///      a single representative (the lower `SymbolId`, see
///      `onnx-runtime-shape-inference/src/context.rs`'s `broadcast_dim`), so a
///      growing KV symbol can be unified INTO a pinned-looking representative that
///      a downstream pointwise op then copies onto BOTH its edges — escaping an
///      exact-membership test (finding 1). Rather than re-derive a partial copy
///      of that unification per op (the enumeration drift that caused earlier
///      misses), we read the AUTHORITATIVE record inference itself keeps
///      ([`Graph::symbol_unifications`](onnx_runtime_ir::Graph::symbol_unifications)):
///      every handler that substitutes a symbol funnels through the single
///      `broadcast_dim` chokepoint that appends there, so the record covers
///      elementwise, `MatMul` batch dims, `Einsum` ellipsis, `Concat` non-concat
///      axes, `Expand`, and any future handler — complete by construction. We
///      close the growing set over those classes (see
///      [`close_growing_under_symbol_unification`]) and mark a whole class growing
///      if ANY of its members is growing. Because `node_capture_seq_independent`
///      tests exact membership on both edges, a closed set makes the growing
///      property transitively correct: a consumer carrying only the representative
///      is still rejected. Over-inclusion (a benign symbol unified with a growing
///      one) only costs perf — the op stays eager — it never admits a growing op.
///
/// The loader interns same-named dim-params to one `SymbolId` across layers, so
/// the resulting set stays small.
pub(super) fn compute_capture_growing_symbols(graph: &Graph) -> HashSet<SymbolId> {
    let mut growing = HashSet::new();
    for node in graph.nodes.values() {
        let Some(slots) = attention_kv_cache_slots(node) else {
            continue;
        };
        for &idx in slots.past_inputs {
            if let Some(Some(vid)) = node.inputs.get(idx).copied()
                && let Some(value) = graph.try_value(vid)
                && let Some(sym) = kv_growing_symbol(&value.shape)
            {
                growing.insert(sym);
            }
        }
        // present_key = outputs[1], present_value = outputs[2] (see
        // dynamic_shapes.rs GQA present handling: `[query[0], kv_heads,
        // present_sequence, head_dim]`, seq axis = penultimate). CSA cache
        // records land on outputs[1]/[3] (`[query[0], records, width]`).
        for &idx in slots.present_outputs {
            if let Some(&vid) = node.outputs.get(idx)
                && let Some(value) = graph.try_value(vid)
                && let Some(sym) = kv_growing_symbol(&value.shape)
            {
                growing.insert(sym);
            }
        }
        // CSA ratio-4 output 5 carries the growing `selections` symbol on its
        // LAST axis (`[query[0], index_heads, query_seq, selections]`).
        for &idx in slots.last_axis_outputs {
            if let Some(&vid) = node.outputs.get(idx)
                && let Some(value) = graph.try_value(vid)
                && let Some(sym) = last_axis_growing_symbol(&value.shape)
            {
                growing.insert(sym);
            }
        }
    }

    // Generic derivation from the DECLARED past/present KV I/O (ONNX GenAI's
    // standard `past_key_values.*` / `present.*` contract). A rank-4 boundary
    // tensor with a symbolic penultimate axis is a growable KV cache regardless
    // of which op produced/consumed it; a rank-3 conv state or a static
    // penultimate recurrent state is excluded.
    let boundary_is_growing_kv = |vid: ValueId| -> Option<SymbolId> {
        let value = graph.try_value(vid)?;
        let name = value.name.as_deref()?;
        let is_kv_boundary = name.starts_with("past") || name.starts_with("present");
        if is_kv_boundary && value.shape.len() == 4 {
            kv_growing_symbol(&value.shape)
        } else {
            None
        }
    };
    for &vid in graph.inputs.iter().chain(graph.outputs.iter()) {
        if let Some(sym) = boundary_is_growing_kv(vid) {
            growing.insert(sym);
        }
    }

    // Close the growing set under shape-inference symbol unification (finding 1).
    close_growing_under_symbol_unification(graph, &mut growing);

    if std::env::var("ONNX_GENAI_LOG_GROWING_SYMBOLS").is_ok() {
        eprintln!(
            "[onnx-genai-capture] build-time growing-symbol set: {} symbol(s): {:?}",
            growing.len(),
            growing
        );
    }
    growing
}

/// A minimal hash-map union-find over [`SymbolId`], used to rebuild the symbol
/// equivalence classes that shape inference's broadcast unification produces.
#[derive(Default)]
struct SymbolUnionFind {
    parent: HashMap<SymbolId, SymbolId>,
}

impl SymbolUnionFind {
    /// The class root of `s` (with path compression). A symbol never inserted is
    /// its own singleton root.
    fn find(&mut self, s: SymbolId) -> SymbolId {
        let mut root = s;
        while let Some(&p) = self.parent.get(&root) {
            if p == root {
                break;
            }
            root = p;
        }
        let mut cur = s;
        while let Some(&p) = self.parent.get(&cur) {
            if p == root {
                break;
            }
            self.parent.insert(cur, root);
            cur = p;
        }
        root
    }

    /// Union the classes of `a` and `b`. The lower `SymbolId` becomes the root,
    /// mirroring inference's lower-id-representative rule (`context.rs`
    /// `broadcast_dim`), though correctness here does not depend on the choice.
    fn union(&mut self, a: SymbolId, b: SymbolId) {
        self.parent.entry(a).or_insert(a);
        self.parent.entry(b).or_insert(b);
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            let (root, child) = if ra.0 <= rb.0 { (ra, rb) } else { (rb, ra) };
            self.parent.insert(child, root);
        }
    }
}

/// Grow `growing` to its equivalence-class closure under shape inference's
/// broadcast symbol unification (finding 1).
///
/// The closure is fed by the graph's AUTHORITATIVE symbol-unification record
/// ([`Graph::symbol_unifications`]), which shape inference populates from the
/// single `broadcast_dim` chokepoint every broadcasting handler funnels through.
/// That record therefore covers *every* op that can substitute one symbol for
/// another as the output representative — elementwise `broadcast`, `MatMul` batch
/// dims (`linalg.rs`), `Einsum` ellipsis (`einsum.rs`), `Concat` non-concat axes
/// (`movement/concat_slice.rs`), `Expand` (`movement/transform.rs`), and any
/// future handler — with zero per-op enumeration here (the enumeration drift that
/// caused earlier misses). The authoritative call sites are, for reference:
///
/// ```text
/// grep -rn "\.broadcast(\|\.broadcast_dim(" crates/onnx-runtime-shape-inference/src/handlers/
/// ```
///
/// We rebuild the union-find from those recorded `(loser, winner)` pairs, then
/// mark every symbol whose class contains a growing member as itself growing.
/// Because [`node_capture_seq_independent`] tests exact both-edge membership, a
/// closed set makes the growing property transitively correct: a downstream
/// consumer that only ever sees the pinned-looking representative on both its
/// edges still stays eager. Over-inclusion (a benign symbol unified with a
/// growing one) only costs perf — the op stays eager — it never admits a growing
/// op.
///
/// [`Graph::symbol_unifications`]: onnx_runtime_ir::Graph::symbol_unifications
fn close_growing_under_symbol_unification(graph: &Graph, growing: &mut HashSet<SymbolId>) {
    if graph.symbol_unifications.is_empty() {
        return;
    }
    let mut uf = SymbolUnionFind::default();
    for &(a, b) in &graph.symbol_unifications {
        if a != b {
            uf.union(a, b);
        }
    }
    if uf.parent.is_empty() {
        return;
    }
    // Roots of the classes that contain at least one growing symbol.
    let growing_roots: HashSet<SymbolId> = growing.iter().map(|&s| uf.find(s)).collect();
    // Every symbol that participated in a union and whose class is growing.
    let members: Vec<SymbolId> = uf.parent.keys().copied().collect();
    for sym in members {
        if growing_roots.contains(&uf.find(sym)) {
            growing.insert(sym);
        }
    }
}

/// Whether `node` is capture-eligible under the **build-time symbol** classifier.
///
/// An op is capture-eligible iff NEITHER any OUTPUT nor any INPUT references a
/// symbol in the GROWING set ([`compute_capture_growing_symbols`]) — a denylist
/// applied to BOTH sides of the op.
///
/// Checking the OUTPUT keeps eager any op whose result is sized by a KV-length
/// symbol (the direct dependence the PR targets). Checking the INPUT as well
/// catches a first-hop alias whose OUTPUT was rewritten to a pinned-looking
/// representative but whose INPUT still carries the raw growing KV symbol
/// (`[seq_kv, D]` ⊕ `[batch, D]` → `[batch, D]`).
///
/// The remaining, harder case — a DOWNSTREAM consumer that copies the already
/// aliased shape, so BOTH its edges show only the representative — is closed not
/// here but in the growing SET itself: [`compute_capture_growing_symbols`] takes
/// the equivalence-class closure of the growing symbols under shape inference's
/// broadcast unification, so the representative a growing symbol was unified into
/// is itself in the set. Exact both-edge membership on that closed set is
/// therefore transitively correct, and no per-call union-find is needed.
///
/// The growing set itself is derived robustly (attention-op set ∪ generic
/// declared KV I/O scan ∪ unification closure, see
/// [`compute_capture_growing_symbols`]) so finding-2 variants (CSA outputs incl.
/// the ratio-4 `selections` axis, and any `past…`/`present…` rank-4 boundary
/// tensor) are covered.
///
/// A growing DENYLIST (rather than a pinned ALLOWLIST) is deliberate: a
/// capture-eligible op legitimately consumes/produces intermediates carrying
/// benign FRESH symbols (warm-decode-seeded data-dependent extents on
/// Cast/Mul/QMoE/ScatterElements). Requiring every symbolic dim to be provably
/// *pinned* would keep all of those eager and dissolve the 154→34 capture
/// collapse that is this PR's entire purpose; only a genuinely GROWING dim on
/// either side is disqualifying.
///
/// Nodes with no outputs are conservatively treated as not seq-independent; a
/// value with an unresolved (missing) shape is treated as carrying no growing
/// symbol on that edge.
pub(super) fn node_capture_seq_independent(
    graph: &Graph,
    node: &Node,
    growing: &HashSet<SymbolId>,
) -> bool {
    if node.outputs.is_empty() {
        return false;
    }
    let edge_free_of_growing = |vid: ValueId| {
        graph
            .try_value(vid)
            .is_none_or(|value| !shape_references_any(&value.shape, growing))
    };
    node.outputs.iter().all(|&vid| edge_free_of_growing(vid))
        && node
            .inputs
            .iter()
            .all(|input| input.is_none_or(edge_free_of_growing))
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
