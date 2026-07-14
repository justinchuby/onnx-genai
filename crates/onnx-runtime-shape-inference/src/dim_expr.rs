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
//!
//! # Overflow contract
//!
//! Coefficients are `i64`. A pathological (but not necessarily malicious) graph
//! can drive a concrete total past `i64::MAX` — e.g. a `Size`/`Reshape` product
//! over four `2^20` dims is `2^80`. Every coefficient combiner
//! ([`add`](DimExpr::add), [`sub`](DimExpr::sub), [`mul`](DimExpr::mul)) is
//! therefore **checked**: on overflow it does **not** panic (as unchecked debug
//! arithmetic would) and does **not** wrap to a bogus — possibly zero or
//! negative — static dim (as unchecked release arithmetic would). Instead the
//! result **degrades to an opaque unknown** ([`DimExpr::overflow`]): an
//! expression that reports as neither a constant nor a bare symbol, poisons any
//! further arithmetic it participates in, and lowers to a *fresh* symbol (see
//! [`crate::context::SymbolInterner::lower`]). This matches the crate's
//! permissive philosophy: a single pathological dim degrades to "unknown"
//! rather than aborting whole-graph inference. [`checked_div`](DimExpr::checked_div)
//! likewise returns `None` on overflow (including the `i64::MIN / -1` edge) so
//! the caller degrades to a fresh symbol.

use std::collections::BTreeMap;

use onnx_runtime_ir::SymbolId;

/// A monomial: a sorted product of symbol ids. The empty vector is the constant
/// monomial (`1`). Stored as raw `u32`s so it is `Ord` for canonical keying.
type Monomial = Vec<u32>;

/// A canonical integer polynomial over symbolic dimensions.
///
/// Invariant: `terms` never contains a zero coefficient, and every key
/// ([`Monomial`]) is sorted ascending. The empty map is the integer `0`.
///
/// The `overflow` flag marks an expression whose exact value could not be
/// represented because a coefficient combiner exceeded `i64` range. Such an
/// expression is an opaque unknown (see the module-level overflow contract):
/// it is never a constant or bare symbol, and it lowers to a fresh symbol.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct DimExpr {
    terms: BTreeMap<Monomial, i64>,
    overflow: bool,
}

impl DimExpr {
    /// The constant `n`.
    pub fn constant(n: i64) -> Self {
        let mut terms = BTreeMap::new();
        if n != 0 {
            terms.insert(Vec::new(), n);
        }
        Self {
            terms,
            overflow: false,
        }
    }

    /// A single symbolic dimension.
    pub fn symbol(s: SymbolId) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(vec![s.0], 1);
        Self {
            terms,
            overflow: false,
        }
    }

    /// An opaque unknown produced by an arithmetic overflow. See the
    /// module-level overflow contract. It reports as neither a constant nor a
    /// bare symbol, poisons any arithmetic it participates in, and lowers to a
    /// fresh symbol.
    pub fn overflow() -> Self {
        Self {
            terms: BTreeMap::new(),
            overflow: true,
        }
    }

    /// Whether this expression is the overflow/unknown sentinel.
    pub fn is_overflow(&self) -> bool {
        self.overflow
    }

    /// The integer value, if this expression is a pure constant (includes `0`).
    pub fn as_const(&self) -> Option<i64> {
        if self.overflow {
            return None;
        }
        match self.terms.len() {
            0 => Some(0),
            1 => self.terms.get(&Vec::new()).copied(),
            _ => None,
        }
    }

    /// The single symbol, if this expression is exactly one symbol with
    /// coefficient `1` (e.g. a bare `Dim::Symbolic` round-trips through this).
    pub fn as_symbol(&self) -> Option<SymbolId> {
        if self.overflow || self.terms.len() != 1 {
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
    ///
    /// Overflow-safe: an out-of-range coefficient sum degrades the result to
    /// [`DimExpr::overflow`] (never panics, never wraps). See the module-level
    /// overflow contract.
    pub fn add(&self, other: &DimExpr) -> DimExpr {
        if self.overflow || other.overflow {
            return DimExpr::overflow();
        }
        let mut terms = self.terms.clone();
        for (mono, &coeff) in &other.terms {
            let slot = terms.entry(mono.clone()).or_insert(0);
            match slot.checked_add(coeff) {
                Some(v) => *slot = v,
                None => return DimExpr::overflow(),
            }
        }
        DimExpr {
            terms,
            overflow: false,
        }
        .prune()
    }

    /// `self - other`.
    ///
    /// Overflow-safe: see [`add`](DimExpr::add).
    pub fn sub(&self, other: &DimExpr) -> DimExpr {
        if self.overflow || other.overflow {
            return DimExpr::overflow();
        }
        let mut terms = self.terms.clone();
        for (mono, &coeff) in &other.terms {
            let slot = terms.entry(mono.clone()).or_insert(0);
            match slot.checked_sub(coeff) {
                Some(v) => *slot = v,
                None => return DimExpr::overflow(),
            }
        }
        DimExpr {
            terms,
            overflow: false,
        }
        .prune()
    }

    /// `self * other`.
    ///
    /// Overflow-safe: an out-of-range coefficient product or accumulation
    /// degrades the result to [`DimExpr::overflow`]. See [`add`](DimExpr::add).
    pub fn mul(&self, other: &DimExpr) -> DimExpr {
        if self.overflow || other.overflow {
            return DimExpr::overflow();
        }
        let mut terms: BTreeMap<Monomial, i64> = BTreeMap::new();
        for (a_mono, &a_c) in &self.terms {
            for (b_mono, &b_c) in &other.terms {
                let Some(prod) = a_c.checked_mul(b_c) else {
                    return DimExpr::overflow();
                };
                let mut mono = a_mono.clone();
                mono.extend_from_slice(b_mono);
                mono.sort_unstable();
                let slot = terms.entry(mono).or_insert(0);
                match slot.checked_add(prod) {
                    Some(v) => *slot = v,
                    None => return DimExpr::overflow(),
                }
            }
        }
        DimExpr {
            terms,
            overflow: false,
        }
        .prune()
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
        // A poisoned operand has no representable value: degrade (caller mints a
        // fresh symbol on `None`).
        if self.overflow || other.overflow {
            return None;
        }
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
            // `checked_rem`/`checked_div` guard the `i64::MIN / -1` overflow
            // (divisor coefficients can be negative via `sub`): `None` degrades
            // to a fresh symbol rather than panicking.
            if coeff.checked_rem(div_coeff)? != 0 {
                return None;
            }
            // Subtract the divisor's symbol multiset from this monomial.
            let mut remaining = mono.clone();
            for sym in div_mono {
                let pos = remaining.iter().position(|s| s == sym)?;
                remaining.remove(pos);
            }
            out.insert(remaining, coeff.checked_div(div_coeff)?);
        }
        Some(
            DimExpr {
                terms: out,
                overflow: false,
            }
            .prune(),
        )
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

    #[test]
    fn mul_overflow_degrades_to_unknown() {
        // A 2^80-scale product (four 2^20 dims) exceeds i64: no panic (as debug
        // unchecked arithmetic would) and no wrap-to-zero (as release would).
        let big = DimExpr::constant(1 << 20);
        let total = DimExpr::product(&[big.clone(), big.clone(), big.clone(), big]);
        assert!(total.is_overflow());
        assert_eq!(total.as_const(), None); // never a bogus concrete dim
        assert_eq!(total.as_symbol(), None);
    }

    #[test]
    fn add_and_sub_overflow_degrade() {
        let max = DimExpr::constant(i64::MAX);
        assert!(max.add(&DimExpr::constant(1)).is_overflow());
        let min = DimExpr::constant(i64::MIN);
        assert!(min.sub(&DimExpr::constant(1)).is_overflow());
    }

    #[test]
    fn overflow_poisons_further_arithmetic() {
        let poisoned = DimExpr::overflow();
        assert!(poisoned.add(&DimExpr::constant(1)).is_overflow());
        assert!(poisoned.mul(&DimExpr::constant(2)).is_overflow());
        assert!(poisoned.sub(&DimExpr::constant(3)).is_overflow());
        assert!(poisoned.checked_div(&DimExpr::constant(4)).is_none());
        // An overflowed divisor also degrades.
        assert!(DimExpr::constant(8).checked_div(&poisoned).is_none());
    }

    #[test]
    fn checked_div_guards_i64_min_over_neg_one() {
        // i64::MIN / -1 overflows; the guard degrades to None rather than panic.
        let num = DimExpr::constant(i64::MIN);
        let div = DimExpr::constant(-1);
        assert!(num.checked_div(&div).is_none());
    }
}
