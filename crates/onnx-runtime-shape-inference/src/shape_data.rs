//! Shape-DATA propagation: [`ShapeData`].
//!
//! The single most important idea carried over from the reference
//! implementation ([`justinchuby/onnx-shape-inference`]). Standard shape
//! inference propagates the *shape* of each tensor; it cannot resolve a
//! `Reshape` whose target vector is *computed* at runtime from a `Shape` op.
//! Shape-data propagation additionally tracks the known *element values* of the
//! small integer tensors that make up those shape-computation subgraphs —
//! `Shape → Slice → Concat → Gather → Unsqueeze → Reshape` — so the reshape
//! target resolves symbolically without executing the graph. This is exactly
//! what lets transformer graphs (BERT and friends) infer statically.
//!
//! A [`ShapeData`] models a rank-0 (scalar) or rank-1 (vector) integer tensor
//! whose elements are [`DimExpr`]s (so they may be concrete *or* symbolic —
//! e.g. the `batch`/`seq_len` dims read out of a `Shape` op). Only small
//! integer tensors are tracked; everything else has no shape-data.
//!
//! [`justinchuby/onnx-shape-inference`]: https://github.com/justinchuby/onnx-shape-inference

use onnx_runtime_ir::DataType;

use crate::dim_expr::DimExpr;
/// Upper bound on the element count a [`ShapeData`] will hold. Shape-vectors are
/// tiny (a handful of dims); this keeps propagation bounded and prevents an
/// accidental attempt to track a large weight tensor's contents.
pub const MAX_SHAPE_DATA_ELEMS: usize = 1024;

/// The known element values of a rank-0 or rank-1 integer tensor flowing
/// through a shape-computation subgraph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeData {
    /// The integer element type (`Int64`/`Int32`, or `Bool` for masks).
    pub dtype: DataType,
    /// Static dimensions. Empty == rank-0 scalar; `[n]` == rank-1 vector.
    pub dims: Vec<usize>,
    /// Row-major elements. Length is `1` for a scalar, `dims[0]` for a vector.
    pub elems: Vec<DimExpr>,
}

impl ShapeData {
    /// A rank-0 scalar holding `value`.
    pub fn scalar(dtype: DataType, value: DimExpr) -> Self {
        Self {
            dtype,
            dims: Vec::new(),
            elems: vec![value],
        }
    }

    /// A rank-1 vector holding `elems`.
    pub fn vector(dtype: DataType, elems: Vec<DimExpr>) -> Self {
        Self {
            dims: vec![elems.len()],
            dtype,
            elems,
        }
    }

    /// Whether this is a rank-0 scalar.
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Whether the element count is within [`MAX_SHAPE_DATA_ELEMS`].
    pub fn within_bounds(&self) -> bool {
        self.elems.len() <= MAX_SHAPE_DATA_ELEMS
    }

    /// This value's contents interpreted as a shape (each element is a dim).
    ///
    /// Used by `Reshape`/`Expand`/`ConstantOfShape` to turn a resolved shape
    /// vector into the output shape.
    pub fn as_shape(&self) -> Vec<DimExpr> {
        self.elems.clone()
    }

    /// Extract shape-data from a concrete constant tensor, if it is a small
    /// (rank ≤ 1, within [`MAX_SHAPE_DATA_ELEMS`]) integer or boolean tensor.
    ///
    /// Non-integer tensors, higher-rank tensors, and over-large tensors are not
    /// tracked (returns `None`) — only the shape-computation operands matter.
    pub fn from_tensor(dtype: DataType, dims: &[usize], data: &[u8]) -> Option<Self> {
        if dims.len() > 1 {
            return None;
        }
        let numel: usize = if dims.is_empty() {
            1
        } else {
            dims.iter().product()
        };
        if numel > MAX_SHAPE_DATA_ELEMS {
            return None;
        }
        let ints = read_ints(dtype, numel, data)?;
        let elems: Vec<DimExpr> = ints.into_iter().map(DimExpr::constant).collect();
        Some(Self {
            dtype,
            dims: dims.to_vec(),
            elems,
        })
    }
}

/// Read `numel` little-endian integer/boolean elements from `data`.
///
/// Returns `None` for non-integral dtypes (floating point, string) or a byte
/// length that does not match `numel` elements.
fn read_ints(dtype: DataType, numel: usize, data: &[u8]) -> Option<Vec<i64>> {
    macro_rules! read {
        ($ty:ty, $sz:expr) => {{
            if data.len() < numel * $sz {
                return None;
            }
            data.chunks_exact($sz)
                .take(numel)
                .map(|c| {
                    let arr: [u8; $sz] = c.try_into().ok()?;
                    Some(<$ty>::from_le_bytes(arr) as i64)
                })
                .collect::<Option<Vec<i64>>>()
        }};
    }
    match dtype {
        DataType::Int64 => read!(i64, 8),
        DataType::Uint64 => read!(u64, 8),
        DataType::Int32 => read!(i32, 4),
        DataType::Uint32 => read!(u32, 4),
        DataType::Int16 => read!(i16, 2),
        DataType::Uint16 => read!(u16, 2),
        DataType::Int8 => read!(i8, 1),
        DataType::Uint8 | DataType::Bool => read!(u8, 1),
        _ => None,
    }
}
