//! Shapes with static and symbolic (dynamic) dimensions (see `docs/ORT2.md`
//! §3.2 and §11).

/// Unique identifier for a symbolic dimension.
///
/// Two dimensions that share the same protobuf dim-param name (e.g.
/// `"batch_size"`) are interned to the same `SymbolId` by the loader so that
/// shape reasoning can equate them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SymbolId(pub u32);

/// A single dimension: either a concrete size or a symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    /// A statically known extent.
    Static(usize),
    /// A dynamic extent identified by a [`SymbolId`].
    Symbolic(SymbolId),
}

impl Dim {
    /// The concrete size, if statically known.
    pub fn as_static(self) -> Option<usize> {
        match self {
            Dim::Static(n) => Some(n),
            Dim::Symbolic(_) => None,
        }
    }

    /// Whether this dimension is statically known.
    pub fn is_static(self) -> bool {
        matches!(self, Dim::Static(_))
    }
}

impl From<usize> for Dim {
    fn from(n: usize) -> Self {
        Dim::Static(n)
    }
}

impl From<SymbolId> for Dim {
    fn from(s: SymbolId) -> Self {
        Dim::Symbolic(s)
    }
}

/// A tensor shape: an ordered list of [`Dim`]s.
///
/// A rank-0 (scalar) shape is the empty vector. The IR `Shape` therefore always
/// denotes a *known* rank; it does not carry a distinct "unknown rank"
/// inhabitant (ONNX `tensor_type.shape` entirely absent). The loader maps an
/// absent `TensorShapeProto` to an empty `Shape`, so unknown-rank values are
/// currently indistinguishable from rank-0 scalars — an accepted limitation for
/// the dense fp / quantized models the runtime targets, where value types carry
/// shapes. Distinguishing the two would require making `Value::shape` optional
/// across the frozen IR contract and is deliberately out of scope here. Known,
/// per-axis unknown *dimensions* (of a known rank) are fully modeled via
/// [`Dim::Symbolic`].
pub type Shape = Vec<Dim>;

/// Convenience constructor for a fully-static shape.
pub fn static_shape(dims: impl IntoIterator<Item = usize>) -> Shape {
    dims.into_iter().map(Dim::Static).collect()
}

/// Returns `Some(dims)` if every dimension of `shape` is static.
pub fn as_static_shape(shape: &[Dim]) -> Option<Vec<usize>> {
    shape.iter().map(|d| d.as_static()).collect()
}

/// Whether every dimension of `shape` is static.
pub fn is_fully_static(shape: &[Dim]) -> bool {
    shape.iter().all(|d| d.is_static())
}

/// Constraints on a symbolic dimension, used by shape bucketing, kernel
/// specialization, and tiling (see `docs/ORT2.md` §11).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SymbolConstraints {
    pub id: Option<SymbolId>,
    /// Human-readable name, e.g. `"batch_size"`, `"seq_len"`.
    pub name: Option<String>,
    /// Inclusive minimum value.
    pub min: Option<usize>,
    /// Inclusive maximum value.
    pub max: Option<usize>,
    /// The value must be a multiple of this (for tiling / vectorization).
    pub divisible_by: Option<usize>,
}

impl SymbolConstraints {
    /// A fresh constraint set for `id` with an optional `name`.
    pub fn new(id: SymbolId, name: Option<String>) -> Self {
        Self {
            id: Some(id),
            name,
            ..Default::default()
        }
    }

    /// Whether `value` satisfies all present constraints.
    pub fn accepts(&self, value: usize) -> bool {
        self.min.is_none_or(|lo| value >= lo)
            && self.max.is_none_or(|hi| value <= hi)
            && self.divisible_by.is_none_or(|m| m != 0 && value.is_multiple_of(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_shape_helpers() {
        let s = static_shape([2, 3, 4]);
        assert_eq!(s.len(), 3);
        assert!(is_fully_static(&s));
        assert_eq!(as_static_shape(&s), Some(vec![2, 3, 4]));
    }

    #[test]
    fn symbolic_shape_is_not_static() {
        let s = vec![Dim::Symbolic(SymbolId(0)), Dim::Static(768)];
        assert!(!is_fully_static(&s));
        assert_eq!(as_static_shape(&s), None);
        assert_eq!(s[1].as_static(), Some(768));
    }

    #[test]
    fn dim_conversions() {
        assert_eq!(Dim::from(5usize), Dim::Static(5));
        assert_eq!(Dim::from(SymbolId(7)), Dim::Symbolic(SymbolId(7)));
    }

    #[test]
    fn constraints_accept() {
        let c = SymbolConstraints {
            id: Some(SymbolId(0)),
            name: Some("seq".into()),
            min: Some(1),
            max: Some(2048),
            divisible_by: Some(8),
        };
        assert!(c.accepts(8));
        assert!(c.accepts(2048));
        assert!(!c.accepts(0)); // below min
        assert!(!c.accepts(4096)); // above max
        assert!(!c.accepts(12)); // not divisible by 8
    }
}
