use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Axis {
    Ellipsis(usize),
    Label(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Token {
    Ellipsis,
    Label(u8),
}

#[derive(Clone, Debug)]
struct Term {
    tokens: Vec<Token>,
    label_count: usize,
    has_ellipsis: bool,
}

/// Direct row-major input-axis mapping produced by the independent parser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperandLayout {
    shape: Vec<usize>,
    physical_axes: Vec<Axis>,
    strides: Vec<usize>,
}

impl OperandLayout {
    /// Concrete input shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn physical_axes(&self) -> &[Axis] {
        &self.physical_axes
    }

    pub(crate) fn strides(&self) -> &[usize] {
        &self.strides
    }
}

/// Fully resolved equation/shape analysis used only by the independent oracle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EquationAnalysis {
    normalized_equation: String,
    output_shape: Vec<usize>,
    output_axes: Vec<Axis>,
    iteration_axes: Vec<Axis>,
    iteration_extents: Vec<usize>,
    operands: Vec<OperandLayout>,
    reduction_elements: usize,
    work_items: usize,
}

impl EquationAnalysis {
    /// Equation after removing ONNX-permitted ASCII spaces.
    pub fn normalized_equation(&self) -> &str {
        &self.normalized_equation
    }

    /// Concrete output shape.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Number of output axes.
    pub fn output_rank(&self) -> usize {
        self.output_axes.len()
    }

    /// Direct input layouts.
    pub fn operands(&self) -> &[OperandLayout] {
        &self.operands
    }

    /// Number of lexicographic reduction tuples for each output element.
    pub fn reduction_elements(&self) -> usize {
        self.reduction_elements
    }

    /// Scalar operand visits performed by the direct evaluator.
    pub fn work_items(&self) -> usize {
        self.work_items
    }

    pub(crate) fn iteration_axes(&self) -> &[Axis] {
        &self.iteration_axes
    }

    pub(crate) fn iteration_extents(&self) -> &[usize] {
        &self.iteration_extents
    }
}

/// Infer the direct oracle's concrete output shape.
pub fn infer_output_shape(
    equation: &str,
    input_shapes: &[Vec<usize>],
) -> Result<Vec<usize>, EquationError> {
    Ok(analyze_equation(equation, input_shapes)?
        .output_shape
        .clone())
}

/// Parse and resolve an ONNX Einsum expression independently of production
/// `EinsumPlan`, including fixed-rank ellipsis and repeated-label diagonals.
pub fn analyze_equation(
    equation: &str,
    input_shapes: &[Vec<usize>],
) -> Result<EquationAnalysis, EquationError> {
    if input_shapes.is_empty() {
        return Err(EquationError::NoInputs);
    }
    let normalized: String = equation.chars().filter(|&ch| ch != ' ').collect();
    let arrows = normalized.match_indices("->").collect::<Vec<_>>();
    if arrows.len() > 1 {
        return Err(EquationError::MultipleOutputArrows);
    }
    let (input_text, explicit_output) = match arrows.first() {
        Some((offset, _)) => (&normalized[..*offset], Some(&normalized[*offset + 2..])),
        None => (normalized.as_str(), None),
    };
    let input_terms = input_text
        .split(',')
        .map(|text| parse_term(text, "input"))
        .collect::<Result<Vec<_>, _>>()?;
    if input_terms.len() != input_shapes.len() {
        return Err(EquationError::InputCount {
            terms: input_terms.len(),
            inputs: input_shapes.len(),
        });
    }

    let mut fixed_ellipsis_rank = None;
    for (input, (term, shape)) in input_terms.iter().zip(input_shapes).enumerate() {
        if shape.len() < term.label_count {
            return Err(EquationError::InputRank {
                input,
                rank: shape.len(),
                named_axes: term.label_count,
                has_ellipsis: term.has_ellipsis,
            });
        }
        let ellipsis_rank = shape.len() - term.label_count;
        if !term.has_ellipsis && ellipsis_rank != 0 {
            return Err(EquationError::InputRank {
                input,
                rank: shape.len(),
                named_axes: term.label_count,
                has_ellipsis: false,
            });
        }
        if term.has_ellipsis {
            if let Some((first_input, expected)) = fixed_ellipsis_rank {
                if ellipsis_rank != expected {
                    return Err(EquationError::FixedEllipsisRank {
                        first_input,
                        expected,
                        input,
                        found: ellipsis_rank,
                    });
                }
            } else {
                fixed_ellipsis_rank = Some((input, ellipsis_rank));
            }
        }
    }
    let ellipsis_rank = fixed_ellipsis_rank.map_or(0, |(_, rank)| rank);

    let mut label_dims = BTreeMap::<u8, usize>::new();
    let mut label_occurrences = BTreeMap::<u8, usize>::new();
    let mut ellipsis_dims = vec![None::<usize>; ellipsis_rank];
    let mut operands = Vec::with_capacity(input_shapes.len());
    for (input, (term, shape)) in input_terms.iter().zip(input_shapes).enumerate() {
        let physical_axes = expand_term(term, ellipsis_rank);
        debug_assert_eq!(physical_axes.len(), shape.len());
        let mut local_dims = BTreeMap::<u8, usize>::new();
        for (axis_index, (&axis, &dimension)) in physical_axes.iter().zip(shape).enumerate() {
            match axis {
                Axis::Label(label) => {
                    *label_occurrences.entry(label).or_default() += 1;
                    if let Some(first) = local_dims.insert(label, dimension)
                        && first != dimension
                    {
                        return Err(EquationError::DiagonalDimension {
                            input,
                            label: label as char,
                            first,
                            axis: axis_index,
                            found: dimension,
                        });
                    }
                    if let Some(&first) = label_dims.get(&label) {
                        if first != dimension {
                            return Err(EquationError::LabelDimension {
                                label: label as char,
                                first,
                                input,
                                axis: axis_index,
                                found: dimension,
                            });
                        }
                    } else {
                        label_dims.insert(label, dimension);
                    }
                }
                Axis::Ellipsis(index) => {
                    ellipsis_dims[index] = Some(match ellipsis_dims[index] {
                        None => dimension,
                        Some(first) => broadcast_extent(first, dimension).ok_or(
                            EquationError::EllipsisDimension {
                                axis: index,
                                first,
                                input,
                                physical_axis: axis_index,
                                found: dimension,
                            },
                        )?,
                    });
                }
            }
        }
        operands.push(OperandLayout {
            shape: shape.clone(),
            strides: contiguous_strides(shape)?,
            physical_axes,
        });
    }
    let ellipsis_dims = ellipsis_dims
        .into_iter()
        .map(|extent| extent.unwrap_or(1))
        .collect::<Vec<_>>();

    let output_axes = if let Some(output) = explicit_output {
        let term = parse_term(output, "output")?;
        let mut seen = BTreeSet::new();
        let mut axes = Vec::new();
        for token in term.tokens {
            match token {
                Token::Ellipsis => {
                    if fixed_ellipsis_rank.is_none() {
                        return Err(EquationError::UnknownOutputEllipsis);
                    }
                    for index in 0..ellipsis_rank {
                        axes.push(Axis::Ellipsis(index));
                    }
                }
                Token::Label(label) => {
                    if !seen.insert(label) {
                        return Err(EquationError::DuplicateOutputLabel(label as char));
                    }
                    if !label_dims.contains_key(&label) {
                        return Err(EquationError::UnknownOutputLabel(label as char));
                    }
                    axes.push(Axis::Label(label));
                }
            }
        }
        axes
    } else {
        let mut axes = (0..ellipsis_rank).map(Axis::Ellipsis).collect::<Vec<_>>();
        axes.extend(
            label_occurrences
                .iter()
                .filter_map(|(&label, &count)| (count == 1).then_some(Axis::Label(label))),
        );
        axes
    };

    let all_axes = (0..ellipsis_rank)
        .map(Axis::Ellipsis)
        .chain(label_dims.keys().copied().map(Axis::Label))
        .collect::<Vec<_>>();
    let output_set = output_axes.iter().copied().collect::<BTreeSet<_>>();
    let reduction_axes = all_axes
        .iter()
        .copied()
        .filter(|axis| !output_set.contains(axis))
        .collect::<Vec<_>>();
    let iteration_axes = output_axes
        .iter()
        .chain(&reduction_axes)
        .copied()
        .collect::<Vec<_>>();
    let extent = |axis: Axis| match axis {
        Axis::Ellipsis(index) => ellipsis_dims[index],
        Axis::Label(label) => label_dims[&label],
    };
    let output_shape = output_axes.iter().copied().map(extent).collect::<Vec<_>>();
    let iteration_extents = iteration_axes
        .iter()
        .copied()
        .map(extent)
        .collect::<Vec<_>>();
    let reduction_elements = checked_product(reduction_axes.iter().copied().map(extent))
        .ok_or(EquationError::SizeOverflow("reduction domain"))?;
    let iteration_elements = checked_product(iteration_extents.iter().copied())
        .ok_or(EquationError::SizeOverflow("iteration domain"))?;
    let work_items = iteration_elements
        .checked_mul(input_shapes.len())
        .ok_or(EquationError::SizeOverflow("oracle work"))?;

    Ok(EquationAnalysis {
        normalized_equation: normalized,
        output_shape,
        output_axes,
        iteration_axes,
        iteration_extents,
        operands,
        reduction_elements,
        work_items,
    })
}

fn parse_term(text: &str, side: &'static str) -> Result<Term, EquationError> {
    let mut tokens = Vec::new();
    let mut label_count = 0usize;
    let mut has_ellipsis = false;
    let mut offset = 0usize;
    while offset < text.len() {
        let rest = &text[offset..];
        if rest.starts_with("...") {
            if has_ellipsis {
                return Err(EquationError::MultipleEllipses(side));
            }
            has_ellipsis = true;
            tokens.push(Token::Ellipsis);
            offset += 3;
            continue;
        }
        let ch = rest.chars().next().expect("offset is inside the string");
        if ch.is_ascii_alphabetic() {
            tokens.push(Token::Label(ch as u8));
            label_count += 1;
            offset += 1;
        } else {
            return Err(EquationError::InvalidCharacter { ch, offset, side });
        }
    }
    Ok(Term {
        tokens,
        label_count,
        has_ellipsis,
    })
}

fn expand_term(term: &Term, ellipsis_rank: usize) -> Vec<Axis> {
    let mut axes =
        Vec::with_capacity(term.label_count + usize::from(term.has_ellipsis) * ellipsis_rank);
    for token in &term.tokens {
        match *token {
            Token::Label(label) => axes.push(Axis::Label(label)),
            Token::Ellipsis => axes.extend((0..ellipsis_rank).map(Axis::Ellipsis)),
        }
    }
    axes
}

fn broadcast_extent(left: usize, right: usize) -> Option<usize> {
    if left == right {
        Some(left)
    } else if left == 1 {
        Some(right)
    } else if right == 1 {
        Some(left)
    } else {
        None
    }
}

fn contiguous_strides(shape: &[usize]) -> Result<Vec<usize>, EquationError> {
    let mut strides = vec![1; shape.len()];
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(shape[axis])
            .ok_or(EquationError::SizeOverflow("input strides"))?;
    }
    Ok(strides)
}

fn checked_product(values: impl IntoIterator<Item = usize>) -> Option<usize> {
    values
        .into_iter()
        .try_fold(1usize, |product, value| product.checked_mul(value))
}

/// Independent equation/shape validation failure.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum EquationError {
    /// No tensor operands.
    #[error("Einsum expected at least one input, found none")]
    NoInputs,
    /// Multiple explicit arrows.
    #[error("Einsum equation contains more than one `->` output arrow")]
    MultipleOutputArrows,
    /// Invalid term byte.
    #[error(
        "Einsum {side} term contains invalid character `{ch}` at normalized byte offset {offset}"
    )]
    InvalidCharacter {
        /// Invalid character.
        ch: char,
        /// Normalized byte offset within the term.
        offset: usize,
        /// Input or output.
        side: &'static str,
    },
    /// Multiple ellipses in one side term.
    #[error("Einsum {0} term contains more than one ellipsis")]
    MultipleEllipses(&'static str),
    /// Equation terms and tensors disagree.
    #[error("Einsum equation declares {terms} input terms, but the node supplies {inputs} inputs")]
    InputCount {
        /// Equation term count.
        terms: usize,
        /// Tensor count.
        inputs: usize,
    },
    /// Input rank does not match its term.
    #[error(
        "Einsum input #{input} rank {rank} does not match {named_axes} named axes (ellipsis present: {has_ellipsis})"
    )]
    InputRank {
        /// Input index.
        input: usize,
        /// Tensor rank.
        rank: usize,
        /// Named axis count.
        named_axes: usize,
        /// Whether the term has ellipsis.
        has_ellipsis: bool,
    },
    /// Explicit ellipses do not all expand to one fixed rank.
    #[error(
        "Einsum input term #{input} explicit ellipsis has expansion rank {found}, but input term #{first_input} explicit ellipsis has expansion rank {expected}"
    )]
    FixedEllipsisRank {
        /// First ellipsis term.
        first_input: usize,
        /// Fixed rank.
        expected: usize,
        /// Mismatching input.
        input: usize,
        /// Mismatching rank.
        found: usize,
    },
    /// A repeated input label is not square.
    #[error(
        "Einsum input #{input} diagonal label `{label}` first has extent {first}, but axis {axis} has extent {found}"
    )]
    DiagonalDimension {
        /// Input index.
        input: usize,
        /// Repeated label.
        label: char,
        /// First extent.
        first: usize,
        /// Mismatching physical axis.
        axis: usize,
        /// Mismatching extent.
        found: usize,
    },
    /// Named extents disagree across inputs.
    #[error(
        "Einsum label `{label}` requires equal dimensions: first extent {first}, input #{input} axis {axis} extent {found}"
    )]
    LabelDimension {
        /// Named label.
        label: char,
        /// First extent.
        first: usize,
        /// Input index.
        input: usize,
        /// Physical axis.
        axis: usize,
        /// Found extent.
        found: usize,
    },
    /// Ellipsis dimensions cannot broadcast.
    #[error(
        "Einsum ellipsis axis #{axis} cannot broadcast extent {first} with input #{input} axis {physical_axis} extent {found}"
    )]
    EllipsisDimension {
        /// Ellipsis axis.
        axis: usize,
        /// First nontrivial extent.
        first: usize,
        /// Input index.
        input: usize,
        /// Physical axis.
        physical_axis: usize,
        /// Found extent.
        found: usize,
    },
    /// Duplicate explicit output label.
    #[error("Einsum output label `{0}` appears more than once")]
    DuplicateOutputLabel(char),
    /// Explicit output label is absent from inputs.
    #[error("Einsum output label `{0}` does not appear in any input")]
    UnknownOutputLabel(char),
    /// Output ellipsis has no input ellipsis.
    #[error("Einsum output ellipsis does not correspond to an input ellipsis")]
    UnknownOutputEllipsis,
    /// Shape arithmetic overflow.
    #[error("Einsum shape arithmetic overflowed while sizing {0}")]
    SizeOverflow(&'static str),
}
