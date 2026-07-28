//! `Einsum` (opset 12): shape inference driven by the `equation` attribute.
//!
//! The equation's labels bind to the corresponding input dimensions; a repeated
//! label is a contraction (summed away) and a label appearing once survives.
//! An `...` term absorbs the remaining leading dimensions and broadcasts across
//! inputs like NumPy batch axes. The output order is taken from the explicit
//! right-hand side (after `->`) or, in the implicit form, from the broadcast
//! ellipsis followed by every once-only label in alphabetical order.
//!
//! Anything the equation cannot pin down from the resolved inputs (an unknown
//! input rank, a malformed term, or a missing output label) leaves the output
//! unresolved rather than guessing.

use std::collections::HashMap;

use onnx_runtime_ir::Attribute;

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;

/// A parsed einsum term: labels before an optional `...`, whether it has an
/// ellipsis, and the labels after it.
struct Term {
    before: Vec<char>,
    has_ellipsis: bool,
    after: Vec<char>,
}

impl Term {
    fn named_count(&self) -> usize {
        self.before.len() + self.after.len()
    }
}

/// Parse a single einsum term. Returns `None` for a term containing anything
/// other than letters and a single `...`.
fn parse_term(term: &str) -> Option<Term> {
    let letters_only = |slice: &str| slice.chars().all(|c| c.is_ascii_alphabetic());
    if let Some(position) = term.find("...") {
        let before = &term[..position];
        let after = &term[position + 3..];
        if !letters_only(before) || !letters_only(after) {
            return None;
        }
        Some(Term {
            before: before.chars().collect(),
            has_ellipsis: true,
            after: after.chars().collect(),
        })
    } else {
        if !letters_only(term) {
            return None;
        }
        Some(Term {
            before: term.chars().collect(),
            has_ellipsis: false,
            after: Vec::new(),
        })
    }
}

/// Record a label→dimension binding and its occurrence count, preferring a
/// concrete extent over a symbolic one when the same label recurs.
fn record_label(
    dims: &mut HashMap<char, DimExpr>,
    counts: &mut HashMap<char, usize>,
    label: char,
    dim: &DimExpr,
) {
    *counts.entry(label).or_insert(0) += 1;
    let keep_existing = dims
        .get(&label)
        .is_some_and(|existing| existing.as_const().is_some());
    if !keep_existing {
        dims.insert(label, dim.clone());
    }
}

fn einsum(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(equation) = ctx.node.attr("equation").and_then(Attribute::as_str) else {
        return Ok(());
    };
    let equation: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
    let (lhs, rhs) = match equation.split_once("->") {
        Some((left, right)) => (left.to_string(), Some(right.to_string())),
        None => (equation, None),
    };
    let terms: Vec<&str> = if lhs.is_empty() {
        Vec::new()
    } else {
        lhs.split(',').collect()
    };
    if terms.len() != ctx.num_inputs() {
        return Ok(());
    }
    let Some(dtype) = (0..ctx.num_inputs()).find_map(|i| ctx.input_dtype(i)) else {
        return Ok(());
    };

    let mut label_dims: HashMap<char, DimExpr> = HashMap::new();
    let mut label_counts: HashMap<char, usize> = HashMap::new();
    let mut ellipsis: Option<Vec<DimExpr>> = None;

    for (index, term) in terms.iter().enumerate() {
        let Some(term) = parse_term(term) else {
            return Ok(());
        };
        let Some(shape) = ctx.input_shape(index).map(<[DimExpr]>::to_vec) else {
            return Ok(());
        };
        if term.has_ellipsis {
            if shape.len() < term.named_count() {
                return Ok(());
            }
            let ellipsis_len = shape.len() - term.named_count();
            for (offset, &label) in term.before.iter().enumerate() {
                record_label(&mut label_dims, &mut label_counts, label, &shape[offset]);
            }
            let ellipsis_dims = shape[term.before.len()..term.before.len() + ellipsis_len].to_vec();
            ellipsis = Some(match ellipsis {
                Some(existing) => ctx.broadcast(&existing, &ellipsis_dims)?,
                None => ellipsis_dims,
            });
            let after_start = shape.len() - term.after.len();
            for (offset, &label) in term.after.iter().enumerate() {
                record_label(
                    &mut label_dims,
                    &mut label_counts,
                    label,
                    &shape[after_start + offset],
                );
            }
        } else {
            if term.before.len() != shape.len() {
                return Ok(());
            }
            for (offset, &label) in term.before.iter().enumerate() {
                record_label(&mut label_dims, &mut label_counts, label, &shape[offset]);
            }
        }
    }

    let out_shape = if let Some(rhs) = rhs {
        let Some(output) = parse_term(&rhs) else {
            return Ok(());
        };
        let mut dims = Vec::new();
        for &label in &output.before {
            let Some(dim) = label_dims.get(&label) else {
                return Ok(());
            };
            dims.push(dim.clone());
        }
        if output.has_ellipsis {
            dims.extend(ellipsis.clone().unwrap_or_default());
        }
        for &label in &output.after {
            let Some(dim) = label_dims.get(&label) else {
                return Ok(());
            };
            dims.push(dim.clone());
        }
        dims
    } else {
        let mut dims = ellipsis.clone().unwrap_or_default();
        let mut singletons: Vec<char> = label_counts
            .iter()
            .filter(|&(_, &count)| count == 1)
            .map(|(&label, _)| label)
            .collect();
        singletons.sort_unstable();
        for label in singletons {
            dims.push(label_dims[&label].clone());
        }
        dims
    };

    ctx.set_output(0, dtype, out_shape);
    Ok(())
}

/// Register the `Einsum` rule.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "Einsum", 12, einsum);
}
