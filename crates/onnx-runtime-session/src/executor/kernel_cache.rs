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

pub(super) fn resolve_kernel_constant_inputs<'a>(
    graph: &'a Graph,
    weights: &'a onnx_runtime_loader::WeightStore,
    inputs: &[Option<ValueId>],
    input_shapes: &'a [Vec<usize>],
) -> Result<Vec<Option<KernelConstantInput<'a>>>> {
    inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let Some(value) = input else {
                return Ok(None);
            };
            let Some(weight) = graph.initializers.get(value) else {
                return Ok(None);
            };
            let bytes = weights.bytes(weight).ok_or_else(|| {
                SessionError::Internal(format!(
                    "initializer value {} could not be resolved for kernel preparation",
                    value.0
                ))
            })?;
            let shape = input_shapes.get(index).ok_or_else(|| {
                SessionError::Internal(format!(
                    "kernel preparation has no shape for initializer input {index}"
                ))
            })?;
            Ok(Some(KernelConstantInput {
                dtype: graph.value(*value).dtype,
                shape,
                bytes,
            }))
        })
        .collect()
}

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

/// Compute the DENYLIST disqualifying set: the growing KV-length symbols closed
/// over shape inference's lineage record. This is the fallback classifier mode
/// ([`CaptureClassifier::Denylist`]); the default entry point is
/// [`compute_capture_disqualifying_symbols`].
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
/// Symbols are gathered from complementary sources so coverage does not depend on
/// a fixed op allowlist (the finding-2 gap):
///   1. The past-KV **inputs** and present-KV **outputs** of every recognized
///      stateful attention node ([`attention_kv_cache_slots`]: GQA, default
///      `Attention`, `IndexShare`, and `CompressedSparseAttention`), including
///      CSA ratio-4's growing `selections` symbol on the **last** axis of
///      output 5 ([`KvCacheSlots::last_axis_outputs`]) — see
///      [`collect_structural_growing_symbols_excluding`].
///   2. A GENERIC scan of the model's DECLARED past/present KV I/O — every graph
///      input named `past…`/output named `present…` with a rank-4 KV layout
///      `[batch, kv_heads, sequence, head_dim]` whose penultimate (sequence) axis
///      is symbolic. This catches an unrecognized attention variant's KV boundary
///      tensor without minting a fresh op entry, while the rank-4 guard excludes
///      fixed-capacity recurrent/conv states (rank 3, or a static penultimate).
///   3. Any OPAQUE unknowable-extent symbol shape inference could not resolve
///      (overflow / negative degrade — [`Graph::symbol_opaque`]).
///   4. The TRANSITIVE LINEAGE CLOSURE of (1)∪(2)∪(3) over BOTH broadcast
///      unifications ([`Graph::symbol_unifications`]) AND derivation provenance
///      ([`Graph::symbol_derivations`], the round-4 fix): a growing KV symbol can
///      be unified INTO a pinned-looking representative (finding 1) OR be baked
///      into a DERIVED symbol like `seq_kv*8` by `Reshape`/`Flatten` (round-4).
///      Both are the AUTHORITATIVE records inference keeps at its two lineage
///      chokepoints (`broadcast_dim`, `lower`), so the closure
///      ([`close_disqualifying_set`]) covers every op that substitutes or derives
///      a symbol — complete by construction, no per-op enumeration.
///
/// [`Graph::symbol_opaque`]: onnx_runtime_ir::Graph::symbol_opaque
/// [`Graph::symbol_unifications`]: onnx_runtime_ir::Graph::symbol_unifications
/// [`Graph::symbol_derivations`]: onnx_runtime_ir::Graph::symbol_derivations
pub(super) fn compute_capture_growing_symbols(graph: &Graph) -> HashSet<SymbolId> {
    compute_capture_growing_symbols_excluding(graph, &HashSet::new())
}

/// As [`compute_capture_growing_symbols`], but treating every symbol in `pinned`
/// as a CONSTANT capacity axis: it is dropped from the structural seed (so the
/// lineage closure never propagates it) and force-excluded from the returned set.
/// See [`super::Executor::pin_fixed_capacity_kv_capture_symbols`] for why a
/// fixed-capacity device-valid-length KV seq axis is safe to pin.
pub(super) fn compute_capture_growing_symbols_excluding(
    graph: &Graph,
    pinned: &HashSet<SymbolId>,
) -> HashSet<SymbolId> {
    // Denylist seed: only the STRUCTURALLY-GROWING KV symbols (plus any opaque
    // unknowable-extent symbol shape inference could not resolve). A symbol that
    // is not proven growing is treated as capturable.
    let mut growing = collect_structural_growing_symbols_excluding(graph, pinned);
    growing.extend(graph.symbol_opaque.iter().copied());
    close_disqualifying_set(graph, &mut growing);
    if !pinned.is_empty() {
        growing.retain(|sym| !pinned.contains(sym));
    }
    growing
}

/// Collect the STRUCTURALLY-growing KV-length symbols from the graph — the raw
/// seed, before any lineage closure. See [`compute_capture_growing_symbols`] for
/// the source enumeration (recognized attention ops ∪ generic declared KV I/O).
/// Any symbol in `pinned` is removed from the seed (a fixed-capacity KV seq axis
/// pinned to a constant capacity).
fn collect_structural_growing_symbols_excluding(
    graph: &Graph,
    pinned: &HashSet<SymbolId>,
) -> HashSet<SymbolId> {
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
    if !pinned.is_empty() {
        growing.retain(|sym| !pinned.contains(sym));
    }
    growing
}

/// The KV sequence-axis symbols that a fixed-capacity, device-valid-length
/// attention path pins CONSTANT at the bound physical capacity. On this path the
/// kernel's launch grid is capacity-sized (bounded by the physically-allocated
/// `max_len`) and the valid attended length is read on-device (GQA's `seqlens_k`,
/// mask-driven `Attention`'s additive-mask frontier, `IndexShare`'s
/// attention-bias frontier), so a captured replay is shape-static: the seq axis
/// never changes extent between steps. These symbols are therefore safe to ADMIT
/// into capture — but only when the runtime actually binds the cache at fixed
/// physical capacity, which is why this is applied through the engine-gated
/// [`super::Executor::pin_fixed_capacity_kv_capture_symbols`] and never at build.
///
/// Only the CAPACITY form of each op contributes: the gate is
/// [`kernel_input_uses_physical_capacity`] over the node's past-KV inputs, which
/// is `false` for the growing-concat / paged / mask-less forms (and for
/// `CompressedSparseAttention`, which has no past-KV inputs). A growing/paged KV
/// path thus keeps its symbol in the disqualifying set and stays vetoed.
/// The sequence-axis symbol(s) of every *decode-freeze-safe* attention-mask
/// graph input — the mask / causal-bias length symbol that a single-token decode
/// step freezes to physical capacity (see
/// [`super::geometry::mask_binding_feeds_additive_causal_builder`]).
///
/// Unlike the KV cache seq axis, this symbol lives on the `attention_mask` input
/// and its additive-causal-builder cone (`CumSum`/`Slice`/`Where`/… → the
/// capacity-form `Attention` mask input), NOT on a `past`/`present` KV slot, so
/// [`collect_capacity_pinned_kv_symbols`] never captures it. Left unpinned it
/// keeps the *entire* mask-builder cone AND every `Attention` node that consumes
/// the bias as forced eager seams — for an MLA / HF-causal-mask model
/// (DeepSeek-V2-Lite) that is all 27 attention layers plus the shared cone,
/// producing a heavily-segmented capture whose eager-seam interleaving replays
/// incoherently.
///
/// Pinning it is safe under exactly the condition the runtime freezes the mask:
/// [`super::geometry::mask_binding_feeds_additive_causal_builder`] holds (the
/// same predicate that drives `DeviceIoBinding::mask_decode_freeze_safe`), so a
/// single-token decode step binds `logical == physical == max_len`. The frozen
/// width saturates the `CumSum` prefix to the true valid length and forces the
/// padded suffix to `-inf`, so a captured replay over the per-step-updated mask
/// buffer is byte-identical to the eager mask — the mask/bias axis is genuinely
/// a fixed-capacity constant across replays, exactly like a pinned KV seq axis.
/// CUDA-graph capture only ever engages on the single-token decode path, so this
/// never leaks the frozen width into multi-token prefill (which runs eager and
/// keeps exposing the logical length).
pub(super) fn collect_freeze_safe_mask_symbols(graph: &Graph) -> HashSet<SymbolId> {
    use super::geometry::{
        is_additive_mask_builder_op, is_capacity_form_attention_mask_input,
        mask_binding_feeds_additive_causal_builder,
    };
    use std::collections::HashMap;

    // value → consumers as (node id, input slot).
    let mut consumers: HashMap<ValueId, Vec<(NodeId, usize)>> = HashMap::new();
    for (node_id, node) in graph.nodes.iter() {
        for (slot, value) in node.inputs.iter().enumerate() {
            if let Some(vid) = value {
                consumers.entry(*vid).or_default().push((node_id, slot));
            }
        }
    }

    let mut symbols = HashSet::new();
    let collect_shape = |symbols: &mut HashSet<SymbolId>, vid: ValueId| {
        if let Some(value) = graph.try_value(vid) {
            for axis in 0..value.shape.len() {
                if let Dim::Symbolic(sym) = value.shape[axis] {
                    symbols.insert(sym);
                }
            }
        }
    };

    for &mask in &graph.inputs {
        if !mask_binding_feeds_additive_causal_builder(graph, mask) {
            continue;
        }
        // Walk the additive-mask-builder cone forward from the mask binding,
        // collecting every symbolic dim on every cone value (the mask input, the
        // `CumSum`/`Slice`/`Where`/… intermediates, and the causal-bias leaf the
        // capacity-form `Attention` consumes). Inference MINTS a fresh derived
        // symbol on the bias axis (distinct from the mask input's raw seq symbol
        // and not always recorded as a derivation of it), so pinning only the
        // input symbol leaves the bias/attention nodes disqualified — the whole
        // cone's symbol set must be pinned to admit them into capture.
        let mut visited: HashSet<ValueId> = HashSet::new();
        let mut frontier = vec![mask];
        while let Some(value) = frontier.pop() {
            if !visited.insert(value) {
                continue;
            }
            collect_shape(&mut symbols, value);
            for &(node_id, slot) in consumers.get(&value).map_or(&[][..], Vec::as_slice) {
                let node = graph.node(node_id);
                // The `Attention` op is a cone leaf: its bias input value is
                // already collected above; do not traverse into the attention.
                if is_capacity_form_attention_mask_input(node, slot) {
                    continue;
                }
                if is_additive_mask_builder_op(node) {
                    for out in &node.outputs {
                        frontier.push(*out);
                    }
                }
            }
        }
    }
    symbols
}

pub(super) fn collect_capacity_pinned_kv_symbols(graph: &Graph) -> HashSet<SymbolId> {
    let mut pinned = HashSet::new();
    for node in graph.nodes.values() {
        let Some(slots) = attention_kv_cache_slots(node) else {
            continue;
        };
        // Require that EVERY past-KV input is read as physical capacity (the
        // device-valid-length path). An op with no past inputs (CSA) or any
        // growing-concat input does not qualify.
        let capacity_form = !slots.past_inputs.is_empty()
            && slots
                .past_inputs
                .iter()
                .all(|&idx| super::geometry::kernel_input_uses_physical_capacity(node, idx));
        if !capacity_form {
            continue;
        }
        for &idx in slots.past_inputs {
            if let Some(Some(vid)) = node.inputs.get(idx).copied()
                && let Some(value) = graph.try_value(vid)
                && let Some(sym) = kv_growing_symbol(&value.shape)
            {
                pinned.insert(sym);
            }
        }
        for &idx in slots.present_outputs {
            if let Some(&vid) = node.outputs.get(idx)
                && let Some(value) = graph.try_value(vid)
                && let Some(sym) = kv_growing_symbol(&value.shape)
            {
                pinned.insert(sym);
            }
        }
    }
    pinned
}

/// The capture classifier's mode. See [`compute_capture_disqualifying_symbols`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CaptureClassifier {
    /// FAIL-SAFE (default): an op is capture-safe only if every symbol on its
    /// edges provably traces (via the lineage record) to graph-declared roots
    /// that are not growing. Any symbol tracing to a growing root OR that is an
    /// inference-minted symbol WITHOUT recorded provenance (untraceable /
    /// data-dependent / overflow) disqualifies the op — it stays eager. This
    /// structurally eliminates the whole "an unrecorded lineage site silently
    /// admits a growing-dependent symbol" bug class: unknown ⇒ eager ⇒ safe.
    FailSafe,
    /// DENYLIST (over provenance): a symbol not proven growing is capturable.
    /// Kept as a switchable fallback; only genuinely growing (or growing-derived,
    /// via the closure) symbols disqualify. Recovers the maximal capture collapse
    /// but relies on every growing-dependency being reachable through a recorded
    /// lineage edge (see the completeness argument in sebastian-728-revision.md).
    Denylist,
}

impl CaptureClassifier {
    /// Resolve the classifier from `ONNX_GENAI_CAPTURE_CLASSIFIER`
    /// (`failsafe`|`fail-safe`|`1` ⇒ fail-safe, `denylist`|`deny`|`0` ⇒
    /// denylist). Defaults to [`FailSafe`](Self::FailSafe): a false
    /// "capture-safe" is silent decode corruption, so the safe-by-default choice
    /// is to keep an op eager whenever its shape lineage is not provably pinned.
    fn from_env() -> Self {
        match std::env::var("ONNX_GENAI_CAPTURE_CLASSIFIER") {
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "denylist" | "deny" | "0" => Self::Denylist,
                _ => Self::FailSafe,
            },
            Err(_) => Self::FailSafe,
        }
    }
}

/// The set of symbols that DISQUALIFY an op from CUDA-graph capture — the single
/// production entry point ([`super::build`]). `node_capture_seq_independent`
/// tests exact both-edge membership against this set, so its meaning is "any op
/// whose shape references one of these must stay eager".
///
/// Which set is returned depends on [`CaptureClassifier`]:
///   * [`FailSafe`](CaptureClassifier::FailSafe) (default): the NOT-PINNED set
///     (structural growing ∪ opaque ∪ every inference-minted symbol without a
///     provenance trace to a pinned root), transitively closed.
///   * [`Denylist`](CaptureClassifier::Denylist): the GROWING set
///     ([`compute_capture_growing_symbols`]).
///
/// Both go through the same [`close_disqualifying_set`] closure; they differ ONLY
/// in the seed, which is why the classifier body and the closure are shared.
pub(super) fn compute_capture_disqualifying_symbols(graph: &Graph) -> HashSet<SymbolId> {
    let mode = CaptureClassifier::from_env();
    let set = match mode {
        CaptureClassifier::FailSafe => compute_not_pinned_symbols(graph),
        CaptureClassifier::Denylist => compute_capture_growing_symbols(graph),
    };
    if std::env::var("ONNX_GENAI_LOG_GROWING_SYMBOLS").is_ok() {
        eprintln!(
            "[onnx-genai-capture] classifier={mode:?} build-time disqualifying-symbol set:              {} symbol(s): {:?}",
            set.len(),
            set
        );
    }
    set
}

/// As [`compute_capture_disqualifying_symbols`], but with `pinned` treated as
/// constant capacity axes (see [`collect_capacity_pinned_kv_symbols`] and
/// [`super::Executor::pin_fixed_capacity_kv_capture_symbols`]): each pinned
/// symbol is dropped from the seed so the closure never propagates it, and is
/// force-excluded from the result.
pub(super) fn compute_capture_disqualifying_symbols_excluding(
    graph: &Graph,
    pinned: &HashSet<SymbolId>,
) -> HashSet<SymbolId> {
    let mode = CaptureClassifier::from_env();
    let set = match mode {
        CaptureClassifier::FailSafe => compute_not_pinned_symbols_excluding(graph, pinned),
        CaptureClassifier::Denylist => compute_capture_growing_symbols_excluding(graph, pinned),
    };
    if std::env::var("ONNX_GENAI_LOG_GROWING_SYMBOLS").is_ok() {
        eprintln!(
            "[onnx-genai-capture] classifier={mode:?} build-time disqualifying-symbol set: \
             {} symbol(s): {:?} (pinned-capacity KV: {:?})",
            set.len(),
            set,
            pinned,
        );
    }
    set
}

/// The FAIL-SAFE not-pinned symbol set (see
/// [`CaptureClassifier::FailSafe`]). Seeds the disqualifying set with the
/// structural growing symbols, the opaque unknowable-extent symbols, AND every
/// inference-minted symbol that lacks a recorded provenance edge (a
/// data-dependent / untraceable fresh symbol — `NonZero`/`Unique`/`Range`/
/// data-dependent `Slice`, a handler `fresh_dim`, etc.), then transitively
/// closes over the lineage record. A minted symbol whose provenance traces only
/// to pinned roots is NOT seeded, so — thanks to the derivation record — a fresh
/// symbol derived purely from pinned sources (`Reshape`/`Flatten` of
/// batch/heads) stays capturable and the capture collapse is preserved.
pub(super) fn compute_not_pinned_symbols(graph: &Graph) -> HashSet<SymbolId> {
    compute_not_pinned_symbols_excluding(graph, &HashSet::new())
}

/// As [`compute_not_pinned_symbols`], but with `pinned` dropped from the
/// structural seed and force-excluded from the result (fixed-capacity KV seq
/// axes pinned constant). A pinned KV seq symbol is a DECLARED root (id < floor,
/// with no provenance edge), so it can only enter via the structural seed; the
/// minted/opaque seeds below never introduce it.
pub(super) fn compute_not_pinned_symbols_excluding(
    graph: &Graph,
    pinned: &HashSet<SymbolId>,
) -> HashSet<SymbolId> {
    let mut set = collect_structural_growing_symbols_excluding(graph, pinned);
    set.extend(graph.symbol_opaque.iter().copied());

    // Without a persisted floor (inference did not run) we cannot tell a minted
    // symbol from a declared root; degrade to the denylist seed (structural ∪
    // opaque) — never less safe than the prior behavior.
    if let Some(floor) = graph.inference_symbol_floor {
        // A symbol is "provably rooted" only if it is a derived symbol (has at
        // least one recorded provenance edge) or a declared root (id < floor).
        // Any inference-minted symbol (id >= floor) with NO provenance edge is
        // untraceable ⇒ disqualifying.
        let has_provenance: HashSet<SymbolId> =
            graph.symbol_derivations.iter().map(|&(d, _)| d).collect();
        // Enumerate every symbol known to the graph: declared/minted symbols are
        // registered in `symbol_constraints`; also scan the lineage records in
        // case any endpoint is not (defensive).
        let candidates = graph
            .symbol_constraints
            .keys()
            .copied()
            .chain(graph.symbol_derivations.iter().flat_map(|&(d, s)| [d, s]))
            .chain(graph.symbol_unifications.iter().flat_map(|&(a, b)| [a, b]));
        for sym in candidates {
            if sym.0 >= floor && !has_provenance.contains(&sym) {
                set.insert(sym);
            }
        }
    }

    close_disqualifying_set(graph, &mut set);
    if !pinned.is_empty() {
        set.retain(|sym| !pinned.contains(sym));
    }
    set
}

/// Transitively close `set` over shape inference's symbol-lineage record so that
/// a symbol is disqualifying whenever it *depends on* a disqualifying symbol.
///
/// Two lineage edge kinds are followed, both read from the graph's
/// AUTHORITATIVE, complete-by-construction records:
///
///   * **Broadcast unifications** ([`Graph::symbol_unifications`], UNDIRECTED):
///     shape inference collapses two distinct symbolic dims onto one
///     representative at the single `broadcast_dim` chokepoint every broadcasting
///     handler funnels through — elementwise `broadcast`, `MatMul` batch dims
///     (`linalg.rs`), `Einsum` ellipsis (`einsum.rs`), `Concat` non-concat axes
///     (`movement/concat_slice.rs`), `Expand` (`movement/transform.rs`), and any
///     future handler. The two symbols denote the *same* dimension, so the
///     relation is symmetric: if either is disqualifying, so is the other.
///
///   * **Derivation provenance** ([`Graph::symbol_derivations`], DIRECTED
///     `source → derived`): when `SymbolInterner::lower` interns a derived
///     expression such as `seq_kv * 8` (`Reshape([-1])`, `Flatten`) to a fresh
///     symbol, it records `(derived, source)` for each constituent. A derived
///     symbol is disqualifying if ANY source is — but NOT the reverse (a pinned
///     `batch` must not be poisoned just because some `batch * seq_kv` product
///     also depends on the growing `seq_kv`). Hence this edge is directed
///     `source → derived` only.
///
/// The closure is a plain worklist BFS over the directed adjacency
/// `{a↔b for broadcast} ∪ {source→derived for derivation}` from the current
/// `set`. Because [`node_capture_seq_independent`] tests exact both-edge
/// membership, a closed set makes the disqualifying property transitively
/// correct: a downstream consumer that only ever sees a pinned-looking
/// representative or a derived symbol on both edges still stays eager.
/// Over-inclusion only costs perf (the op stays eager) — it never admits a
/// growing-dependent op.
///
/// Drift-detection greps for the two chokepoints, for reference:
/// ```text
/// grep -rn "\.broadcast(\|\.broadcast_dim(" crates/onnx-runtime-shape-inference/src/handlers/
/// grep -rn "\.lower(" crates/onnx-runtime-shape-inference/src/
/// ```
///
/// [`Graph::symbol_unifications`]: onnx_runtime_ir::Graph::symbol_unifications
/// [`Graph::symbol_derivations`]: onnx_runtime_ir::Graph::symbol_derivations
fn close_disqualifying_set(graph: &Graph, set: &mut HashSet<SymbolId>) {
    if graph.symbol_unifications.is_empty() && graph.symbol_derivations.is_empty() {
        return;
    }
    // Directed adjacency: disqualifying-ness flows along these edges.
    let mut adj: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new();
    for &(a, b) in &graph.symbol_unifications {
        if a != b {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
    }
    for &(derived, source) in &graph.symbol_derivations {
        if derived != source {
            adj.entry(source).or_default().push(derived);
        }
    }
    if adj.is_empty() {
        return;
    }
    let mut work: Vec<SymbolId> = set.iter().copied().collect();
    while let Some(sym) = work.pop() {
        let Some(neighbors) = adj.get(&sym) else {
            continue;
        };
        for &next in neighbors {
            if set.insert(next) {
                work.push(next);
            }
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
    /// Entries dropped by the per-node variant bound (issue #1362).
    pub evictions: u64,
}

/// Shape-keyed kernel cache (§11.1). Owns the compiled kernels for the session.
#[derive(Default)]
pub(crate) struct KernelCache {
    pub(super) entries: HashMap<KernelKey, Box<dyn onnx_runtime_ep_api::Kernel>>,
    /// Logical time each entry was last served, for the per-node bound below.
    /// Kept beside `entries` rather than inside them so a shared-reference hit
    /// can record its use without the map itself becoming interior-mutable.
    pub(super) last_used: HashMap<KernelKey, AtomicU64>,
    /// Monotonic tick handed out to `last_used`.
    pub(super) clock: AtomicU64,
    pub(super) hits: u64,
    pub(super) misses: u64,
    /// Entries dropped by the per-node bound.
    pub(super) evictions: u64,
    pub(super) prebind_hits: AtomicU64,
    block_quantized_moe_traffic_request: Option<u32>,
}

/// How many shape variants of one node the cache keeps (issue #1362).
///
/// The cache is keyed by node **and input shapes**, so every distinct prompt
/// length compiles a fresh kernel for every node in the graph. A compiled kernel
/// owns device workspaces (attention scratch, dequantization scratch, broadcast
/// metadata), so an unbounded cache turns "each request has a new prompt length"
/// into VRAM that grows without limit until the device is exhausted — memory the
/// resource governor never sees, because it belongs to the kernels rather than
/// to any binding or KV pool.
///
/// A per-node bound rather than a global one keeps the eviction proportional to
/// what actually varies: a graph keeps its hot decode and prefill variants, and
/// only a node that has genuinely seen many shapes gives one up.
///
/// The bound is small on purpose. A retained variant owns device scratch, so
/// raising it raises the floor under every long-context request: measured on a
/// 30B decoder, a bound of 10 pushed a 5.5k-token prompt past the mapped-memory
/// ceiling that a bound of 4 cleared with 20 GB to spare. The bound is the cap
/// on that scratch, and widening it defeats its own purpose.
///
/// Four is only enough because the shapes a request cycles through are kept
/// deliberately few: the native CUDA decoder rounds its prefill widths onto a
/// three-step ladder (`PREFILL_QUERY_WIDTH_STEPS`) so that a prompt's leftover
/// final chunk cannot invent a fresh width per request, which leaves exactly one
/// slot for the single-token decode shape.
const DEFAULT_VARIANTS_PER_NODE: usize = 4;

fn variants_per_node() -> usize {
    static RESOLVED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *RESOLVED.get_or_init(|| {
        std::env::var("ONNX_RUNTIME_KERNEL_CACHE_VARIANTS_PER_NODE")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|variants| *variants > 0)
            .unwrap_or(DEFAULT_VARIANTS_PER_NODE)
    })
}

impl KernelCache {
    #[inline]
    pub(super) fn contains(&self, key: &KernelKey) -> bool {
        self.entries.contains_key(key)
    }

    pub(super) fn arm_block_quantized_moe_traffic(&mut self, request_id: u32) -> Result<usize> {
        let mut visited = HashSet::new();
        let mut armed = 0;
        let mut failure = None;
        for (key, kernel) in &mut self.entries {
            if visited.insert(key.node) {
                match kernel.arm_block_quantized_moe_traffic(request_id) {
                    Ok(true) => armed += 1,
                    Ok(false) => {}
                    Err(error) => {
                        failure = Some(error);
                        break;
                    }
                }
            }
        }
        if let Some(error) = failure {
            let mut rollback = HashSet::new();
            for (key, kernel) in &mut self.entries {
                if rollback.insert(key.node) {
                    let _ = kernel.disarm_block_quantized_moe_traffic();
                }
            }
            return Err(error.into());
        }
        self.block_quantized_moe_traffic_request = (armed != 0).then_some(request_id);
        Ok(armed)
    }

    pub(super) fn reset_block_quantized_moe_traffic(&mut self) -> Result<()> {
        let mut visited = HashSet::new();
        for (key, kernel) in &mut self.entries {
            if visited.insert(key.node) {
                kernel.reset_block_quantized_moe_traffic()?;
            }
        }
        Ok(())
    }

    pub(super) fn snapshot_block_quantized_moe_traffic(
        &self,
    ) -> Result<onnx_runtime_ep_api::BlockQuantizedMoeTraffic> {
        let mut visited = HashSet::new();
        let mut total = onnx_runtime_ep_api::BlockQuantizedMoeTraffic::default();
        let mut physical_dram_bytes = 0_u64;
        let mut physical_complete = true;
        let mut observed = false;
        for (key, kernel) in &self.entries {
            if !visited.insert(key.node) {
                continue;
            }
            let Some(snapshot) = kernel.snapshot_block_quantized_moe_traffic()? else {
                continue;
            };
            observed = true;
            total.uploaded_whole_bank_bytes = total
                .uploaded_whole_bank_bytes
                .checked_add(snapshot.uploaded_whole_bank_bytes)
                .ok_or_else(|| SessionError::Internal("uploaded BQMoE bytes overflow".into()))?;
            total.committed_whole_bank_bytes = total
                .committed_whole_bank_bytes
                .checked_add(snapshot.committed_whole_bank_bytes)
                .ok_or_else(|| SessionError::Internal("committed BQMoE bytes overflow".into()))?;
            total.logical_route_demand_bytes = total
                .logical_route_demand_bytes
                .checked_add(snapshot.logical_route_demand_bytes)
                .ok_or_else(|| SessionError::Internal("logical BQMoE bytes overflow".into()))?;
            total.unique_selected_expert_bytes = total
                .unique_selected_expert_bytes
                .checked_add(snapshot.unique_selected_expert_bytes)
                .ok_or_else(|| SessionError::Internal("unique BQMoE bytes overflow".into()))?;
            total.page_ins = total
                .page_ins
                .checked_add(snapshot.page_ins)
                .ok_or_else(|| SessionError::Internal("BQMoE page-ins overflow".into()))?;
            match snapshot.physical_dram_bytes {
                Some(bytes) => {
                    physical_dram_bytes =
                        physical_dram_bytes.checked_add(bytes).ok_or_else(|| {
                            SessionError::Internal("physical BQMoE bytes overflow".into())
                        })?;
                }
                None => physical_complete = false,
            }
        }
        total.physical_dram_bytes = (observed && physical_complete).then_some(physical_dram_bytes);
        total.byte_hit_rate = total.physical_dram_bytes.and_then(|physical| {
            (total.logical_route_demand_bytes != 0)
                .then(|| 1.0 - (physical as f64 / total.logical_route_demand_bytes as f64).min(1.0))
        });
        Ok(total)
    }

    #[cfg(feature = "gpu-tests")]
    pub(super) fn inject_block_quantized_moe_traffic_fault_for_test(
        &self,
        fault: onnx_runtime_ep_cuda::kernels::block_quantized_moe::BlockQuantizedMoeTrafficFaultForTest,
    ) -> Result<()> {
        let mut visited = HashSet::new();
        let mut injected = 0usize;
        for (key, kernel) in &self.entries {
            if !visited.insert(key.node) {
                continue;
            }
            let Some(kernel) = kernel
                .as_any()
                .downcast_ref::<onnx_runtime_ep_cuda::kernels::block_quantized_moe::BlockQuantizedMoEKernel>()
            else {
                continue;
            };
            kernel.inject_route_telemetry_fault_for_test(fault)?;
            injected += 1;
        }
        if injected == 0 {
            return Err(SessionError::Internal(
                "no CUDA BlockQuantizedMoE kernel was available for traffic fault injection".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn disarm_block_quantized_moe_traffic(&mut self) -> Result<()> {
        let mut visited = HashSet::new();
        for (key, kernel) in &mut self.entries {
            if visited.insert(key.node) {
                kernel.disarm_block_quantized_moe_traffic()?;
            }
        }
        self.block_quantized_moe_traffic_request = None;
        Ok(())
    }

    /// Next logical tick.
    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    /// Record that `key` was just served, so eviction can tell the hot variants
    /// of a node from the ones a single past prompt length left behind.
    fn touch(&self, key: &KernelKey) {
        if let Some(slot) = self.last_used.get(key) {
            slot.store(self.tick(), Ordering::Relaxed);
        }
    }

    /// Drop the least recently used variants of `node` beyond the per-node bound.
    ///
    /// A kernel's device workspaces are freed by its `Drop`, so the EP's captured
    /// device graph is reset first: a capture can have baked the pointer of a
    /// workspace this eviction is about to free, and replaying it afterwards
    /// would read freed device memory. Resetting is what the device-binding drop
    /// path already does for the same reason.
    fn evict_surplus_variants(
        &mut self,
        node: u32,
        ep: &dyn ExecutionProvider,
        graph_tokens: [Option<DeviceGraphToken>; DeviceGraphSlot::COUNT],
    ) -> Result<()> {
        let bound = variants_per_node();
        let mut variants = self
            .entries
            .keys()
            .filter(|key| key.node == node)
            .map(|key| {
                let used = self
                    .last_used
                    .get(key)
                    .map(|slot| slot.load(Ordering::Relaxed))
                    .unwrap_or(0);
                (used, key.clone())
            })
            .collect::<Vec<_>>();
        if variants.len() <= bound {
            return Ok(());
        }
        variants.sort_by_key(|(used, _)| *used);
        let surplus = variants.len() - bound;
        // Evicting kernel variants can retire kernels baked into a captured
        // device graph in EITHER slot, so defensively reset both the Primary
        // (M=1 decode) and Verify (M=K speculative) slots here — resetting an
        // empty slot is a cheap no-op. This keeps the eviction path slot-correct
        // without threading the caller's active slot through the whole cache API.
        for token in graph_tokens.into_iter().flatten() {
            ep.reset_owned_device_graph(token)?;
        }
        for (_, key) in variants.into_iter().take(surplus) {
            self.entries.remove(&key);
            self.last_used.remove(&key);
            self.evictions += 1;
        }
        Ok(())
    }

    pub(super) fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            prebind_hits: self.prebind_hits.load(Ordering::Relaxed),
            evictions: self.evictions,
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
        self.touch(binding);
        self.prebind_hits.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        PREBIND_FAST_PATH_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(kernel)
    }

    #[inline]
    pub(super) fn has_prebound(&self, binding: &KernelKey, input_shapes: &[Vec<usize>]) -> bool {
        binding.matches_shapes(input_shapes) && self.entries.contains_key(binding)
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
        constant_values: &[Option<KernelConstantInput<'_>>],
        opset: u64,
        capture_seq_independent: bool,
        executor: ExecutorInstanceId,
        artifact_readiness: &mut ProviderArtifactReadiness,
        ep: &dyn ExecutionProvider,
        graph_tokens: [Option<DeviceGraphToken>; DeviceGraphSlot::COUNT],
    ) -> Result<(&dyn onnx_runtime_ep_api::Kernel, KernelKey)> {
        let key = KernelKey {
            node: node_id.0,
            shapes: input_shapes.to_vec(),
        };
        if self.entries.contains_key(&key) {
            self.hits += 1;
        } else {
            let shared_constant_state = self
                .entries
                .iter()
                .find(|(existing, _)| existing.node == key.node)
                .and_then(|(_, kernel)| kernel.shareable_constant_state());
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
            let mut kernel = match ep.get_kernel_for_executor(executor, node, input_shapes, opset) {
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
            let adopted = if let Some(state) = shared_constant_state {
                kernel.adopt_shareable_constant_state(state)?
            } else {
                false
            };
            if !adopted {
                kernel.prepare_constant_inputs(constant_values, ep)?;
            }
            if !adopted && let Some(request_id) = self.block_quantized_moe_traffic_request {
                kernel.arm_block_quantized_moe_traffic(request_id)?;
            }
            kernel.set_capture_seq_independent(capture_seq_independent);
            self.entries.insert(key.clone(), kernel);
            self.last_used
                .insert(key.clone(), AtomicU64::new(self.tick()));
            self.misses += 1;
            // Kernel creation is the single publication chokepoint for
            // executor-scoped provider artifacts. Invalidate permission here,
            // not in selected callers, so build preflight, binding preparation,
            // and runtime dispatch cannot disagree about readiness.
            artifact_readiness.advance_to(ExecutorArtifactReadinessEpoch::new(self.misses));
            self.evict_surplus_variants(key.node, ep, graph_tokens)?;
        }
        self.touch(&key);
        #[cfg(test)]
        PREBIND_FALLBACK_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let kernel_ref = self.entries.get(&key).expect("just inserted").as_ref();
        Ok((kernel_ref, key))
    }
}
