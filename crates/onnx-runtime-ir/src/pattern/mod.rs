//! Compile-once, read-only ONNX graph recognition.
//!
//! [`Pattern::parse_ort`] and [`PatternLibrary::parse_ort`] accept the documented
//! subset of ORT's pattern dialect, **not** ONNX model/function text. See
//! `docs/architecture/IR_PATTERNS.md` for the interoperability contract.
//!
//! ```
//! use onnx_runtime_ir::{Graph, Node, NodeId, DataType};
//! use onnx_runtime_ir::pattern::{Pattern, MatchSession, MatchOptions};
//! let pattern = Pattern::parse_ort("(x) => return Identity(x)")?;
//! let mut graph = Graph::new();
//! let x = graph.create_value(DataType::Float32, vec![]);
//! graph.set_inputs(vec![x]);
//! let y = graph.create_value(DataType::Float32, vec![]);
//! let root = graph.insert_node(Node::new(NodeId(0), "Identity", vec![Some(x)], vec![y]));
//! graph.set_outputs(vec![y]);
//! let session = MatchSession::new(&graph)?;
//! let found = session.match_at(&pattern, root, MatchOptions::default())?.unwrap();
//! assert_eq!(found.value("x")?.unwrap().id(), x);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod compile;
mod execute;
mod syntax;

pub use execute::{
    AttributeCapture, BindingView, BoundaryPolicy, DimensionValue, Match, MatchBudget,
    MatchOptions, MatchSession, ScopedValue, ValueCapture,
};

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{DataType, TensorData};

/// A half-open UTF-8 byte range in the pattern source.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceSpan {
    pub start: usize,
    pub end: usize,
}

/// Machine-readable failure category. Ordinary mismatches are `Ok(None)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatternErrorKind {
    Syntax,
    Unsupported,
    InvalidPattern,
    InvalidGraph,
    BudgetExceeded,
    MissingMetadata,
    InvalidCapture,
    Predicate,
}

/// Actionable diagnostic, with a source span for parser/compiler errors.
#[derive(Clone, Debug, thiserror::Error)]
#[error("{kind:?} at bytes {start}..{end}: {message}", start=.span.start, end=.span.end)]
pub struct PatternError {
    pub kind: PatternErrorKind,
    pub span: SourceSpan,
    pub message: String,
}

impl PatternError {
    /// Report a failure from a read-only semantic predicate without disguising it
    /// as a structural mismatch.
    pub fn predicate(message: impl Into<String>) -> Self {
        Self::new(PatternErrorKind::Predicate, SourceSpan::default(), message)
    }

    fn new(kind: PatternErrorKind, span: SourceSpan, message: impl Into<String>) -> Self {
        Self {
            kind,
            span,
            message: message.into(),
        }
    }
}

type Result<T> = std::result::Result<T, PatternError>;

/// Upstream contract used for compatibility fixtures, not a released ORT version.
pub const ORT_REFERENCE_COMMIT: &str = "c992a5210bf5bf6a8f8ace36d8c6304ba70f1821";

/// Parsed and verified immutable plans. Cloning shares the plans.
#[derive(Clone, Debug)]
pub struct Pattern {
    plans: Arc<[Plan]>,
    operator_entry: bool,
}

impl Pattern {
    /// Parse an anonymous ORT pattern `(inputs) => { ... return outputs }`.
    /// Unsupported syntax is rejected, never interpreted as an ONNX model.
    pub fn parse_ort(source: &str) -> Result<Self> {
        let definition = syntax::parse_inline(source, false)?;
        compile::inline(definition)
    }

    /// Parse explicitly native extensions: operator `*`, ordered `...xs`
    /// operands, `values[min:*]`/`slots[min:*]`, and GRAPH-list references.
    pub fn parse_native(source: &str) -> Result<Self> {
        compile::inline(syntax::parse_inline(source, true)?)
    }
}

/// Owning named definitions with forward references, calls and alternatives.
#[derive(Clone, Debug)]
pub struct PatternLibrary {
    patterns: BTreeMap<String, Pattern>,
}

impl PatternLibrary {
    /// `operator_exists` distinguishes standard operators from misspelled
    /// unqualified pattern calls without adding a schema dependency to the IR.
    /// Explicitly qualified custom operators do not require registry lookup.
    pub fn parse_ort(source: &str, operator_exists: &dyn Fn(&str, &str) -> bool) -> Result<Self> {
        compile::library(syntax::parse_library(source, false)?, operator_exists)
    }

    pub fn parse_native(
        source: &str,
        operator_exists: &dyn Fn(&str, &str) -> bool,
    ) -> Result<Self> {
        compile::library(syntax::parse_library(source, true)?, operator_exists)
    }

    /// Retrieve a connected operator entry anchored at its last expanded node.
    pub fn pattern(&self, name: &str) -> Result<Pattern> {
        let mut pattern = self.graph_pattern(name)?;
        for plan in pattern.plans.iter() {
            compile::verify_entry(plan)?;
        }
        pattern.operator_entry = true;
        Ok(pattern)
    }

    /// Retrieve a whole-graph entry, including zero-node passthrough bodies.
    pub fn graph_pattern(&self, name: &str) -> Result<Pattern> {
        self.patterns.get(name).cloned().ok_or_else(|| {
            PatternError::new(
                PatternErrorKind::InvalidPattern,
                SourceSpan::default(),
                format!("pattern {name:?} is not defined; select a named library definition"),
            )
        })
    }
}

#[derive(Clone, Debug)]
struct Plan {
    names: BTreeMap<String, usize>,
    values: Vec<ValuePlan>,
    nodes: Vec<NodePlan>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    annotations: Vec<(usize, Vec<Constraint>)>,
    attribute_types: BTreeMap<String, (Option<String>, bool)>,
    attribute_names: BTreeMap<String, String>,
    attribute_conjunctions: Vec<(String, Option<String>, bool)>,
    parameters: Vec<Formal>,
    calls: usize,
    signature: Vec<syntax::Parameter>,
    output_signature: Vec<syntax::Parameter>,
    producers: Vec<Option<usize>>,
    tensor_formals: std::collections::BTreeSet<usize>,
}

impl Plan {
    fn empty() -> Self {
        Self {
            names: BTreeMap::new(),
            values: Vec::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            annotations: Vec::new(),
            attribute_types: BTreeMap::new(),
            attribute_names: BTreeMap::new(),
            parameters: Vec::new(),
            calls: 0,
            attribute_conjunctions: Vec::new(),
            signature: Vec::new(),
            output_signature: Vec::new(),
            producers: Vec::new(),
            tensor_formals: std::collections::BTreeSet::new(),
        }
    }
    fn value(&mut self, value: ValuePlan) -> usize {
        let id = self.values.len();
        self.values.push(value);
        id
    }
}

#[derive(Clone, Debug)]
enum Formal {
    Value(usize),
    Attribute(String),
}

#[derive(Clone, Debug)]
enum ValuePlan {
    Present,
    Optional,
    Absent,
    Literal(TensorData),
    Sequence {
        min: usize,
        max: Option<usize>,
        holes: bool,
    },
}

#[derive(Clone, Debug)]
struct NodePlan {
    domain: String,
    op: String,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    attributes: Vec<(String, AttributePlan)>,
    instance: String,
    rest: Option<usize>,
}

#[derive(Clone, Debug)]
enum AttributePlan {
    Literal(Literal),
    Capture(String),
    Graph(Pattern),
    Graphs(Vec<Pattern>),
}

#[derive(Clone, Debug, PartialEq)]
enum Literal {
    Int(i64),
    Float(f32),
    String(Vec<u8>),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Constraint {
    absent: bool,
    immutable: bool,
    element: Option<DataType>,
    element_variable: Option<String>,
    dimensions: Option<Vec<Dimension>>,
    prefix: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Dimension {
    Any,
    Fixed(usize),
    Named(String),
}
