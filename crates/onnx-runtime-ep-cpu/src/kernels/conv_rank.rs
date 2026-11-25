//! The input-rank capability shared by Conv claim and construction.

use onnx_runtime_ir::Shape;

/// Empty claim shapes can also denote unknown rank in the current IR. Defer
/// those facts until concrete construction; per-axis symbolic dimensions keep
/// their known nonempty rank.
pub(super) fn claim_rejection(shapes: &[Shape]) -> Option<String> {
    let known_rank = |shape: &Shape| (!shape.is_empty()).then_some(shape.len());
    rank_rejection(
        shapes.first().and_then(known_rank),
        shapes.get(1).and_then(known_rank),
    )
}

pub(super) fn concrete_rejection(shapes: &[Vec<usize>]) -> Option<String> {
    rank_rejection(shapes.first().map(Vec::len), shapes.get(1).map(Vec::len))
}

fn rank_rejection(x: Option<usize>, w: Option<usize>) -> Option<String> {
    if x.is_some_and(|rank| !matches!(rank, 3 | 4))
        || w.is_some_and(|rank| !matches!(rank, 3 | 4))
        || matches!((x, w), (Some(x), Some(w)) if x != w)
    {
        let rank = |value: Option<usize>| {
            value.map_or_else(|| "unknown".into(), |value| value.to_string())
        };
        Some(format!(
            "Conv: requires matching rank-3 NCL or rank-4 NCHW tensors; got X rank={}, W rank={}",
            rank(x),
            rank(w),
        ))
    } else {
        None
    }
}
