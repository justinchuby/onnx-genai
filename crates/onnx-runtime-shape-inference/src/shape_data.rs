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
///
/// Integer tensors carry their values in [`elems`](Self::elems) as
/// [`DimExpr`]s. Floating-point *scalar* constants (needed by `Range`, whose
/// float form the CPU kernel supports) are carried additively in
/// [`float_elems`](Self::float_elems); for those, `elems` is empty. `Eq` is not
/// derived because `f64` is not `Eq` — `ShapeData` is only ever used as a map
/// *value*, so structural equality is not required.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeData {
    /// The element type (`Int64`/`Int32`, or `Bool` for masks; `Float32`/
    /// `Float64` for the float-scalar side-channel).
    pub dtype: DataType,
    /// Static dimensions. Empty == rank-0 scalar; `[n]` == rank-1 vector.
    pub dims: Vec<usize>,
    /// Row-major integer elements. Length is `1` for a scalar, `dims[0]` for a
    /// vector. Empty when this is a floating-point scalar (see `float_elems`).
    pub elems: Vec<DimExpr>,
    /// Row-major floating-point elements for floating-point constants. Present
    /// only for float/double scalars captured for `Range`; `None` otherwise.
    pub float_elems: Option<Vec<f64>>,
}

impl ShapeData {
    /// A rank-0 scalar holding `value`.
    pub fn scalar(dtype: DataType, value: DimExpr) -> Self {
        Self {
            dtype,
            dims: Vec::new(),
            elems: vec![value],
            float_elems: None,
        }
    }

    /// A rank-1 vector holding `elems`.
    pub fn vector(dtype: DataType, elems: Vec<DimExpr>) -> Self {
        Self {
            dims: vec![elems.len()],
            dtype,
            elems,
            float_elems: None,
        }
    }

    /// A rank-0 floating-point scalar holding `value`.
    pub fn float_scalar(dtype: DataType, value: f64) -> Self {
        Self {
            dtype,
            dims: Vec::new(),
            elems: Vec::new(),
            float_elems: Some(vec![value]),
        }
    }

    /// This value interpreted as a floating-point scalar, if it is one.
    pub fn as_float_scalar(&self) -> Option<f64> {
        let floats = self.float_elems.as_ref()?;
        (self.is_scalar() && floats.len() == 1).then(|| floats[0])
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
        // Floating-point *scalars* are captured additively for `Range`; higher-
        // rank / vector float tensors (weights) are deliberately not tracked.
        if dtype.is_float() {
            if !dims.is_empty() {
                return None;
            }
            let value = read_float_scalar(dtype, data)?;
            return Some(Self::float_scalar(dtype, value));
        }
        let ints = read_ints(dtype, numel, data)?;
        let elems: Vec<DimExpr> = ints.into_iter().map(DimExpr::constant).collect();
        Some(Self {
            dtype,
            dims: dims.to_vec(),
            elems,
            float_elems: None,
        })
    }
}

/// Read a single little-endian floating-point scalar from `data`.
///
/// Supports `Float32` and `Float64` (the forms the `Range` kernel accepts);
/// returns `None` for other dtypes or a too-short buffer.
fn read_float_scalar(dtype: DataType, data: &[u8]) -> Option<f64> {
    match dtype {
        DataType::Float32 => {
            let arr: [u8; 4] = data.get(..4)?.try_into().ok()?;
            Some(f32::from_le_bytes(arr) as f64)
        }
        DataType::Float64 => {
            let arr: [u8; 8] = data.get(..8)?.try_into().ok()?;
            Some(f64::from_le_bytes(arr))
        }
        _ => None,
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
