//! Built-in operator inference rules.
//!
//! Each rule is a small pure function over an [`InferenceContext`]. Rules are
//! grouped by family and wired into the [`InferenceRegistry`] by
//! [`register_all`]. The Phase-1 coverage targets the transformer + CNN op set
//! needed to fully infer BERT-class graphs without executing them; see the
//! crate docs for the full list.

use crate::registry::InferenceRegistry;

mod custom_ops;
mod data_ops;
mod elementwise;
mod linalg;
mod movement;
mod norm;
mod pooling;
mod selection;
mod sequence;

/// Normalise an ONNX axis (which may be negative) into `0..rank`.
///
/// Negative axes count from the end. The result is clamped into `0..=rank-1`
/// for indexing an existing dimension (`rank` must be ≥ 1).
pub(crate) fn norm_axis(axis: i64, rank: usize) -> usize {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    a.clamp(0, r.saturating_sub(1)) as usize
}

/// Normalize an axis without clamping invalid values or overflowing.
pub(crate) fn checked_axis(axis: i64, rank: usize) -> Option<usize> {
    let rank = i64::try_from(rank).ok()?;
    let normalized = if axis < 0 {
        axis.checked_add(rank)?
    } else {
        axis
    };
    (0..rank)
        .contains(&normalized)
        .then(|| usize::try_from(normalized).ok())
        .flatten()
}

/// Populate `registry` with every built-in rule.
pub fn register_all(registry: &mut InferenceRegistry) {
    custom_ops::register(registry);
    elementwise::register(registry);
    linalg::register(registry);
    norm::register(registry);
    movement::register(registry);
    data_ops::register(registry);
    pooling::register(registry);
    selection::register(registry);
    sequence::register(registry);
}
