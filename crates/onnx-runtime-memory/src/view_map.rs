//! Zero-copy **view → source** aliasing supplied by the executor.
//!
//! The executor treats the outputs of layout/movement ops
//! (`Slice`/`Reshape`/`Squeeze`/`Unsqueeze`/`Transpose`/`Expand`, …) as
//! *zero-copy views*: they own no buffer and instead borrow (alias) a source
//! buffer with some strided geometry (see `ValueView` in
//! `onnx-runtime-session/src/executor.rs`). The source buffer is therefore
//! *pinned* — it must outlive every view that aliases it, or a reused buffer
//! would clobber a still-live alias (a use-after-free / silent-corruption bug).
//!
//! The planner does **not** hardcode op names. Instead the caller (the
//! executor, which already computes this) supplies a [`ViewMap`] of
//! `view → source` edges. The planner folds those edges transitively to a
//! *root* owner and extends the root's live interval to cover every view's last
//! use. This is exactly what prevents the greedy allocator from recycling a
//! buffer that a live view still points into.

use std::collections::HashMap;

use onnx_runtime_ir::ValueId;

/// A set of zero-copy `view → source` aliasing relationships.
///
/// Each entry means "`view` owns no activation slot; its bytes live inside
/// `source`'s buffer". A view of a view is allowed; [`ViewMap::root`] folds a
/// chain down to the single real buffer owner.
#[derive(Clone, Debug, Default)]
pub struct ViewMap {
    /// `view → immediate source`. The source may itself be a view.
    edges: HashMap<ValueId, ValueId>,
}

impl ViewMap {
    /// An empty map (no views — every value owns its own buffer).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a map from `(view, source)` pairs.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (ValueId, ValueId)>) -> Self {
        let mut m = Self::new();
        for (view, source) in pairs {
            m.insert(view, source);
        }
        m
    }

    /// Record that `view` aliases `source` (owns no buffer of its own).
    pub fn insert(&mut self, view: ValueId, source: ValueId) {
        self.edges.insert(view, source);
    }

    /// Whether `value` is a zero-copy view (aliases another value's buffer and
    /// therefore gets no slot of its own).
    pub fn is_view(&self, value: ValueId) -> bool {
        self.edges.contains_key(&value)
    }

    /// The immediate source `value` aliases, if it is a view.
    pub fn source_of(&self, value: ValueId) -> Option<ValueId> {
        self.edges.get(&value).copied()
    }

    /// Fold `value` to the **root** buffer owner by following `view → source`
    /// edges transitively. Returns `value` itself when it is not a view.
    ///
    /// A cycle in the (malformed) view map is broken defensively: the last node
    /// visited before the cycle closes is returned rather than looping forever.
    pub fn root(&self, value: ValueId) -> ValueId {
        let mut seen: Vec<ValueId> = Vec::new();
        let mut cur = value;
        while let Some(&next) = self.edges.get(&cur) {
            if seen.contains(&cur) {
                break;
            }
            seen.push(cur);
            cur = next;
        }
        cur
    }

    /// Number of recorded view edges.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether there are no view edges.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}
