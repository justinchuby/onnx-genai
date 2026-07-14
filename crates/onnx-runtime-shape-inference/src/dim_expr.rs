//! Symbolic dimension arithmetic: [`DimExpr`].
//!
//! `onnx-runtime-ir`'s [`Dim`](onnx_runtime_ir::Dim) can only represent a
//! concrete extent ([`Dim::Static`](onnx_runtime_ir::Dim::Static)) or an opaque
//! symbol ([`Dim::Symbolic`](onnx_runtime_ir::Dim::Symbolic)). Shape inference
//! for reshape / conv / pool / flatten needs *derived* dimensions such as
//! `d0 * d1`, `d0 + k`, or `d0 / k`. We do **not** modify the frozen IR
//! contract; instead this crate reasons over dimensions using a small canonical
//! integer polynomial and only *lowers* back to an IR [`Dim`] when writing an
//! inferred shape into the graph (see [`crate::context::SymbolInterner`]).
//!
//! # Representation
//!
//! A [`DimExpr`] is a multivariate polynomial with integer coefficients over
//! the graph's [`SymbolId`]s: a sum of *monomials*, where each monomial is a
//! sorted product of symbols paired with an `i64` coefficient. Canonicalisation
//! (constant folding, term merging, sorted monomials) means two structurally
//! equal expressions compare equal — so identical derived dimensions (e.g. two
//! `batch * seq` products) intern to the *same* fresh symbol and stay unified.
//!
//! This is deliberately **not** a general CAS: it captures exactly the affine
//! and product forms the Phase-1 op set produces. Operations it cannot
//! represent exactly (floor division by a symbol, non-exact division) surface
//! as `None`, and the caller falls back to a fresh opaque symbol — the same
//! permissive degrade the reference implementation uses.

use std::collections::BTreeMap;

use onnx_runtime_ir::SymbolId;

/// A monomial: a sorted product of symbol ids. The empty vector is the constant
/// monomial (`1`). Stored as raw `u32`s so it is `Ord` for canonical keying.
type Monomial = Vec<u32>;

/// A canonical integer polynomial over symbolic dimensions.
///
/// Invariant: `terms` never contains a zero coefficient, and every key
/// ([`Monomial`]) is sorted ascending. The empty map is the integer `0`.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct DimExpr {
    terms: BTreeMap<Monomial, i64>,
}

impl DimExpr {
    /// The constant `n`.
    pub fn constant(n: i64) -> Self {
        let mut terms = BTreeMap::new();
        if n != 0 {
            terms.insert(Vec::new(), n);
        }
        Self { terms }
    }

    /// A single symbolic dimension.
    pub fn symbol(s: SymbolId) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(vec![s.0], 1);
        Self { terms }
    }

    /// The integer value, if this expression is a pure constant (includes `0`).
    pub fn as_const(&self) -> Option<i64> {
        match self.terms.len() {
            0 => Some(0),
            1 => self.terms.get(&Vec::new()).copied(),
            _ => None,
        }
    }

    /// The single symbol, if this expression is exactly one symbol with
    /// coefficient `1` (e.g. a bare `Dim::Symbolic` round-trips through this).
    pub fn as_symbol(&self) -> Option<SymbolId> {
        if self.terms.len() != 1 {
            return None;
        }
        let (mono, &coeff) = self.terms.iter().next()?;
        if coeff == 1 && mono.len() == 1 {
            Some(SymbolId(mono[0]))
        } else {
            None
        }
    }

    /// Whether this is a pure constant.
    pub fn is_const(&self) -> bool {
        self.as_const().is_some()
    }

    /// Drop any term whose coefficient collapsed to zero.
    fn prune(mut self) -> Self {
        self.terms.retain(|_, c| *c != 0);
        self
    }

    /// `self + other`.
    pub fn add(&self, other: &DimExpr) -> DimExpr {
        let mut terms = self.terms.clone();
        for (mono, &coeff) in &other.terms {
            *terms.entry(mono.clone()).or_insert(0) += coeff;
        }
        DimExpr { terms }.prune()
    }

    /// `self - other`.
    pub fn sub(&self, other: &DimExpr) -> DimExpr {
        let mut terms = self.terms.clone();
        for (mono, &coeff) in &other.terms {
            *terms.entry(mono.clone()).or_insert(0) -= coeff;
        }
        DimExpr { terms }.prune()
    }

    /// `self * other`.
    pub fn mul(&self, other: &DimExpr) -> DimExpr {
        let mut terms: BTreeMap<Monomial, i64> = BTreeMap::new();
        for (a_mono, &a_c) in &self.terms {
            for (b_mono, &b_c) in &other.terms {
                let mut mono = a_mono.clone();
                mono.extend_from_slice(b_mono);
                mono.sort_unstable();
                *terms.entry(mono).or_insert(0) += a_c * b_c;
            }
        }
        DimExpr { terms }.prune()
    }

    /// `self / other`, only when the division is *exact*.
    ///
    /// Handles the cases the op set actually produces: division by a non-zero
    /// constant (every coefficient must divide evenly) and division by a single
    /// monomial (symbol cancellation, as in `Reshape` `-1` inference where
    /// `total = b*s*768` is divided by `b*s*12` to yield `64`). Anything else —
    /// dividing by a multi-term polynomial, or a non-exact quotient — returns
    /// `None` so the caller can degrade to a fresh symbol.
    pub fn checked_div(&self, other: &DimExpr) -> Option<DimExpr> {
        if self.terms.is_empty() {
            return Some(DimExpr::constant(0));
        }
        // Divisor must be a single monomial with a non-zero coefficient.
        if other.terms.len() != 1 {
            return None;
        }
        let (div_mono, &div_coeff) = other.terms.iter().next()?;
        if div_coeff == 0 {
            return None;
        }
        let mut out: BTreeMap<Monomial, i64> = BTreeMap::new();
        for (mono, &coeff) in &self.terms {
            if coeff % div_coeff != 0 {
                return None;
            }
            // Subtract the divisor's symbol multiset from this monomial.
            let mut remaining = mono.clone();
            for sym in div_mono {
                let pos = remaining.iter().position(|s| s == sym)?;
                remaining.remove(pos);
            }
            out.insert(remaining, coeff / div_coeff);
        }
        Some(DimExpr { terms: out }.prune())
    }

    /// The product of a slice of expressions (`1` for an empty slice).
    pub fn product(exprs: &[DimExpr]) -> DimExpr {
        let mut acc = DimExpr::constant(1);
        for e in exprs {
            acc = acc.mul(e);
        }
        acc
    }
}

impl From<onnx_runtime_ir::Dim> for DimExpr {
    fn from(d: onnx_runtime_ir::Dim) -> Self {
        match d {
            onnx_runtime_ir::Dim::Static(n) => DimExpr::constant(n as i64),
            onnx_runtime_ir::Dim::Symbolic(s) => DimExpr::symbol(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(n: u32) -> DimExpr {
        DimExpr::symbol(SymbolId(n))
    }

    #[test]
    fn constant_folding() {
        let e = DimExpr::constant(3).add(&DimExpr::constant(4));
        assert_eq!(e.as_const(), Some(7));
        assert!(e.is_const());
    }

    #[test]
    fn zero_is_canonical() {
        assert_eq!(DimExpr::constant(0).as_const(), Some(0));
        assert_eq!(
            DimExpr::constant(5).sub(&DimExpr::constant(5)).as_const(),
            Some(0)
        );
    }

    #[test]
    fn symbol_roundtrip() {
        let e = sym(2);
        assert_eq!(e.as_symbol(), Some(SymbolId(2)));
        // 2*d is not a bare symbol.
        assert_eq!(e.add(&sym(2)).as_symbol(), None);
    }

    #[test]
    fn affine_expression() {
        // d0 + 5
        let e = sym(0).add(&DimExpr::constant(5));
        assert_eq!(e.as_const(), None);
        assert_eq!(e.as_symbol(), None);
        // (d0 + 5) - 5 == d0
        assert_eq!(e.sub(&DimExpr::constant(5)).as_symbol(), Some(SymbolId(0)));
    }

    #[test]
    fn product_of_symbols() {
        // d0 * d1
        let e = sym(0).mul(&sym(1));
        // commutative canonical form: d1 * d0 equals d0 * d1
        assert_eq!(e, sym(1).mul(&sym(0)));
    }

    #[test]
    fn exact_constant_division() {
        let e = DimExpr::constant(48);
        assert_eq!(
            e.checked_div(&DimExpr::constant(6)).unwrap().as_const(),
            Some(8)
        );
        // non-exact
        assert!(
            DimExpr::constant(7)
                .checked_div(&DimExpr::constant(2))
                .is_none()
        );
    }

    #[test]
    fn reshape_minus_one_cancellation() {
        // total = b * s * 768, known = b * s * 12  ->  64
        let b = sym(0);
        let s = sym(1);
        let total = DimExpr::product(&[b.clone(), s.clone(), DimExpr::constant(768)]);
        let known = DimExpr::product(&[b, s, DimExpr::constant(12)]);
        let missing = total.checked_div(&known).unwrap();
        assert_eq!(missing.as_const(), Some(64));
    }

    #[test]
    fn division_by_multiterm_is_none() {
        let total = sym(0).mul(&sym(1));
        let divisor = sym(0).add(&DimExpr::constant(1));
        assert!(total.checked_div(&divisor).is_none());
    }

    #[test]
    fn symbolic_product_division_keeps_symbol() {
        // (b * 768) / 768 == b
        let e = sym(0).mul(&DimExpr::constant(768));
        let q = e.checked_div(&DimExpr::constant(768)).unwrap();
        assert_eq!(q.as_symbol(), Some(SymbolId(0)));
    }

    #[test]
    fn from_ir_dim() {
        use onnx_runtime_ir::Dim;
        assert_eq!(DimExpr::from(Dim::Static(4)).as_const(), Some(4));
        assert_eq!(
            DimExpr::from(Dim::Symbolic(SymbolId(9))).as_symbol(),
            Some(SymbolId(9))
        );
    }
}
