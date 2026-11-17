use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::syntax::{Definition, Expr, ExprKind, Parameter, RawConstraint};
use super::*;

/// A value in one builder's scope, not an actual graph value or capture key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatternValue {
    owner: usize,
    index: usize,
}

/// An attribute binding in one builder's scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatternAttribute {
    owner: usize,
    index: usize,
}

/// An ordered sequence of positional slots, including absent slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatternSequence {
    owner: usize,
    index: usize,
}

/// Explicit positional slots. `None::<PatternValue>` also denotes absence.
#[derive(Clone, Copy, Debug)]
pub enum Input {
    Value(PatternValue),
    Sequence(PatternSequence),
    Absent,
    /// A fresh present wildcard at each occurrence, as in text `_`.
    Any,
}

impl From<PatternValue> for Input {
    fn from(value: PatternValue) -> Self {
        Self::Value(value)
    }
}

impl From<PatternSequence> for Input {
    fn from(value: PatternSequence) -> Self {
        Self::Sequence(value)
    }
}

impl From<Option<PatternValue>> for Input {
    fn from(value: Option<PatternValue>) -> Self {
        value.map_or(Self::Absent, Self::Value)
    }
}

/// The represented ONNX attribute kinds supported by capture annotations.
#[derive(Clone, Copy, Debug)]
pub enum AttributeType {
    Int,
    Float,
    String,
    Ints,
    Floats,
    Strings,
    Tensor,
}

impl AttributeType {
    fn name(self) -> &'static str {
        match self {
            Self::Int => "int",
            Self::Float => "float",
            Self::String => "string",
            Self::Ints => "ints",
            Self::Floats => "floats",
            Self::Strings => "strings",
            Self::Tensor => "tensor",
        }
    }
}

/// A verified whole-graph body with independent local bindings.
/// Clone it to constrain separate GRAPH attributes with the same structure.
#[derive(Clone, Debug)]
pub struct Body {
    pattern: Pattern,
    depth: usize,
}

/// Attribute constraints cannot be used as positional value handles.
#[derive(Clone, Debug)]
pub enum AttributeValue {
    Int(i64),
    Float(f32),
    String(Vec<u8>),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Capture(PatternAttribute),
    Graph(Body),
}

macro_rules! attribute_from {
    ($source:ty, $variant:ident) => {
        impl From<$source> for AttributeValue {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}
attribute_from!(i64, Int);
attribute_from!(f32, Float);
attribute_from!(Vec<i64>, Ints);
attribute_from!(Vec<f32>, Floats);
attribute_from!(PatternAttribute, Capture);
attribute_from!(Body, Graph);

impl From<&str> for AttributeValue {
    fn from(value: &str) -> Self {
        Self::String(value.as_bytes().to_vec())
    }
}

#[derive(Debug)]
struct Binding {
    name: String,
    explicit: bool,
    kind: BindingKind,
}

#[derive(Debug)]
enum BindingKind {
    Input(ValueInput),
    Sequence { min: usize, max: Option<usize> },
    Output,
    Attribute(AttributeType, bool),
}

#[derive(Clone, Copy, Debug)]
enum ValueInput {
    Required,
    OptionalTensor,
    OptionalValue,
}

#[derive(Clone, Copy)]
enum BindingClass {
    Value,
    Attribute,
    Sequence,
}

/// Compact, checked pattern construction. Methods record construction errors;
/// `finish`/`finish_body` return all of them before invoking the shared compiler.
///
/// ```
/// use onnx_runtime_ir::pattern::Pattern;
/// let mut p = Pattern::builder();
/// let [x, w, scale] = p.inputs(["x", "w", "scale"]);
/// let weight = p.op("DequantizeLinear", [w, scale]);
/// let y = p.op("MatMul", [x, weight]);
/// let pattern = p.finish(y)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Attribute and body handles are deliberately not tensor operands:
/// ```compile_fail
/// use onnx_runtime_ir::pattern::{Pattern, AttributeType};
/// let mut p = Pattern::builder();
/// let axis = p.attribute("axis", AttributeType::Int);
/// let y = p.op("Identity", [axis]);
/// ```
///
/// ```compile_fail
/// use onnx_runtime_ir::pattern::Pattern;
/// let mut branch = Pattern::builder();
/// let x = branch.input("x");
/// let body = branch.finish_body([x]).unwrap();
/// let mut p = Pattern::builder();
/// let y = p.op("Identity", [body]);
/// ```
///
/// ```compile_fail
/// use onnx_runtime_ir::pattern::Pattern;
/// let mut p = Pattern::builder();
/// let x = p.input("x");
/// let y = p.node("Identity", [x]).attr("axis", x).output();
/// ```
#[derive(Debug)]
pub struct PatternBuilder {
    owner: usize,
    bindings: Vec<Binding>,
    statements: Vec<(Vec<Parameter>, Expr)>,
    errors: Vec<String>,
    depth: usize,
}

impl PatternBuilder {
    pub(super) fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(1);
        let owner = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .unwrap_or(0);
        Self {
            owner,
            bindings: Vec::new(),
            statements: Vec::new(),
            errors: if owner == 0 {
                vec!["builder identity space exhausted; restart the process before building more patterns".into()]
            } else {
                Vec::new()
            },
            depth: 0,
        }
    }

    fn binding(&mut self, name: Option<&str>, kind: BindingKind) -> usize {
        let index = self.bindings.len();
        self.bindings.push(Binding {
            name: name.map_or_else(|| internal_name(index), str::to_owned),
            explicit: name.is_some(),
            kind,
        });
        index
    }

    /// Declare and capture a required input once.
    pub fn input(&mut self, name: &str) -> PatternValue {
        let index = self.binding(Some(name), BindingKind::Input(ValueInput::Required));
        PatternValue {
            owner: self.owner,
            index,
        }
    }

    pub fn inputs<const N: usize>(&mut self, names: [&str; N]) -> [PatternValue; N] {
        names.map(|name| self.input(name))
    }

    /// A tensor-or-absent input, including a truncated trailing slot.
    pub fn optional_input(&mut self, name: &str) -> PatternValue {
        let index = self.binding(Some(name), BindingKind::Input(ValueInput::OptionalTensor));
        PatternValue {
            owner: self.owner,
            index,
        }
    }

    /// An arbitrary value-or-absent slot, without requiring tensor metadata.
    pub fn optional_value(&mut self, name: &str) -> PatternValue {
        let index = self.binding(Some(name), BindingKind::Input(ValueInput::OptionalValue));
        PatternValue {
            owner: self.owner,
            index,
        }
    }

    /// Capture an ordered slot sequence with explicit inclusive length bounds.
    pub fn slot_sequence(&mut self, name: &str, min: usize, max: Option<usize>) -> PatternSequence {
        if let Some(max) = max
            && max < min
        {
            self.errors.push(format!(
                "slot sequence {name:?} has maximum {max} below minimum {min}; use min <= max"
            ));
        }
        let index = self.binding(Some(name), BindingKind::Sequence { min, max });
        PatternSequence {
            owner: self.owner,
            index,
        }
    }

    pub fn attribute(&mut self, name: &str, kind: AttributeType) -> PatternAttribute {
        self.attribute_binding(name, kind, false)
    }

    pub fn optional_attribute(&mut self, name: &str, kind: AttributeType) -> PatternAttribute {
        self.attribute_binding(name, kind, true)
    }

    fn attribute_binding(
        &mut self,
        name: &str,
        kind: AttributeType,
        optional: bool,
    ) -> PatternAttribute {
        let index = self.binding(Some(name), BindingKind::Attribute(kind, optional));
        PatternAttribute {
            owner: self.owner,
            index,
        }
    }

    /// Name an intermediate for normal `Match::value`/`instance` access.
    /// Inputs already have names; each binding may be named only once.
    pub fn capture(&mut self, name: &str, value: PatternValue) {
        if self.check(value.owner, value.index, BindingClass::Value, "capture") {
            let binding = &mut self.bindings[value.index];
            if binding.explicit {
                self.errors.push(format!(
                    "capture {name:?}: binding {:?} is already named; use its existing capture",
                    binding.name
                ));
            } else {
                binding.name = name.into();
                binding.explicit = true;
            }
        }
    }

    /// Start with an existing value; alternatives can add operator branches.
    pub fn alias(&mut self, value: PatternValue) -> NodeBuilder<'_> {
        let expr = self.value_expr(value, "alias");
        NodeBuilder {
            builder: self,
            current: PendingExpression::Value(expr),
            alternatives: Vec::new(),
            completed: false,
        }
    }

    /// Standard-domain operator, exact ordered inputs, one output.
    pub fn op<I: Into<Input>>(
        &mut self,
        op: &str,
        inputs: impl IntoIterator<Item = I>,
    ) -> PatternValue {
        self.node(op, inputs).output()
    }

    /// Configure only the uncommon parts, then call `output` or `outputs`.
    #[must_use = "call output() or outputs() to add the operator"]
    pub fn node<I: Into<Input>>(
        &mut self,
        op: &str,
        inputs: impl IntoIterator<Item = I>,
    ) -> NodeBuilder<'_> {
        let current = self.operation(op, inputs);
        NodeBuilder {
            builder: self,
            current,
            alternatives: Vec::new(),
            completed: false,
        }
    }

    fn operation<I: Into<Input>>(
        &mut self,
        op: &str,
        inputs: impl IntoIterator<Item = I>,
    ) -> PendingExpression {
        let inputs = inputs
            .into_iter()
            .enumerate()
            .map(|(index, input)| match input.into() {
                Input::Value(value) => {
                    self.value_expr(value, &format!("operator {op:?} input {index}"))
                }
                Input::Sequence(sequence) => {
                    self.check(
                        sequence.owner,
                        sequence.index,
                        BindingClass::Sequence,
                        &format!("operator {op:?} input {index}"),
                    );
                    expression(ExprKind::Spread(internal_name(sequence.index)))
                }
                Input::Absent => expression(ExprKind::Name("None".into())),
                Input::Any => expression(ExprKind::Name("_".into())),
            })
            .collect();
        PendingExpression::Operator {
            op: op.into(),
            domain: None,
            inputs,
            attributes: Vec::new(),
        }
    }

    fn check(&mut self, owner: usize, index: usize, expected: BindingClass, context: &str) -> bool {
        let error = if owner != self.owner {
            Some(
                "handle belongs to another builder/scope; declare a local input or pass a finished body",
            )
        } else if let Some(binding) = self.bindings.get(index) {
            let correct = matches!(
                (expected, &binding.kind),
                (
                    BindingClass::Value,
                    BindingKind::Input(_) | BindingKind::Output
                ) | (BindingClass::Attribute, BindingKind::Attribute(..))
                    | (BindingClass::Sequence, BindingKind::Sequence { .. })
            );
            if !correct {
                Some(
                    "handle has the wrong binding kind; use separate value, attribute and sequence handles",
                )
            } else {
                None
            }
        } else {
            Some("handle refers to an undefined binding; use a handle returned by this builder")
        };
        if let Some(error) = error {
            self.errors.push(format!("{context}: {error}"));
        }
        error.is_none()
    }

    fn value_expr(&mut self, value: PatternValue, context: &str) -> Expr {
        self.check(value.owner, value.index, BindingClass::Value, context);
        expression(ExprKind::Name(internal_name(value.index)))
    }

    pub fn finish(self, output: PatternValue) -> Result<Pattern> {
        self.finish_outputs([output])
    }

    /// Return distinct produced values in explicit order.
    pub fn finish_outputs(
        self,
        outputs: impl IntoIterator<Item = PatternValue>,
    ) -> Result<Pattern> {
        self.compile(outputs, true)
    }

    /// Whole-graph semantics: permits passthrough/duplicate returns and unused
    /// positional formals. No operator anchor is required.
    pub fn finish_body(self, outputs: impl IntoIterator<Item = PatternValue>) -> Result<Body> {
        let depth = self.depth;
        Ok(Body {
            pattern: self.compile(outputs, false)?,
            depth,
        })
    }

    fn compile(
        mut self,
        outputs: impl IntoIterator<Item = PatternValue>,
        operator_entry: bool,
    ) -> Result<Pattern> {
        let mut returns: Vec<_> = outputs
            .into_iter()
            .enumerate()
            .map(|(index, value)| self.value_expr(value, &format!("return {index}")))
            .collect();
        if returns.is_empty() {
            self.errors
                .push("return list is empty; select at least one output value".into());
        }
        let mut names = BTreeSet::new();
        for binding in &self.bindings {
            if binding.explicit && !identifier(&binding.name) {
                self.errors.push(format!(
                    "capture name {:?} is invalid; use an ASCII identifier other than _ or None",
                    binding.name
                ));
            }
            if !names.insert(&binding.name) {
                self.errors.push(format!(
                    "duplicate capture {:?}; give values and attributes distinct names",
                    binding.name
                ));
            }
        }
        if self.depth >= 32 {
            self.errors
                .push("GRAPH nesting reaches 32; flatten the scoped bodies".into());
        }
        if !self.errors.is_empty() {
            return Err(PatternError::new(
                PatternErrorKind::InvalidPattern,
                SourceSpan::default(),
                format!("pattern builder: {}", self.errors.join("; ")),
            ));
        }
        let mut parameters = Vec::new();
        for (index, binding) in self.bindings.iter().enumerate() {
            let annotation = match binding.kind {
                BindingKind::Input(ValueInput::Required) => None,
                BindingKind::Input(ValueInput::OptionalTensor) => {
                    let mut annotation = optional_annotation("tensor");
                    annotation[0].dimensions = None;
                    Some(annotation)
                }
                BindingKind::Input(ValueInput::OptionalValue) => {
                    let mut annotation = optional_annotation("value");
                    annotation[0].dimensions = None;
                    Some(annotation)
                }
                BindingKind::Sequence { min, max } => {
                    let mut annotation = raw("slots");
                    annotation.dimensions = None;
                    annotation.sequence = Some((min, max, true));
                    Some(vec![annotation])
                }
                BindingKind::Attribute(kind, optional) => {
                    let name = internal_name(index);
                    let used = self
                        .statements
                        .iter()
                        .any(|(_, expr)| uses_attribute(expr, &name));
                    if !used {
                        return Err(PatternError::new(
                            PatternErrorKind::InvalidPattern,
                            SourceSpan::default(),
                            format!(
                                "attribute capture {:?} is unused; attach it with node(...).attr(...) or remove it",
                                binding.name
                            ),
                        ));
                    }
                    Some(if optional {
                        optional_annotation(kind.name())
                    } else {
                        vec![raw(kind.name())]
                    })
                }
                BindingKind::Output => continue,
            };
            parameters.push(Parameter {
                name: binding.name.clone(),
                annotation,
            });
        }
        let renames: BTreeMap<_, _> = self
            .bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| (internal_name(index), binding.name.clone()))
            .collect();
        let rename = |name: &mut String| {
            if let Some(mapped) = renames.get(name) {
                *name = mapped.clone();
            }
        };
        for (targets, expr) in &mut self.statements {
            for target in targets {
                rename(&mut target.name);
            }
            rename_expr(expr, &rename);
        }
        for expr in &mut returns {
            rename_expr(expr, &rename);
        }
        compile::compile_body(
            Definition {
                named: false,
                interface: None,
                parameters,
                statements: self.statements,
                returns,
                output_signature: None,
                return_labels: None,
                call_parameters: None,
                require_outputs: false,
                span: SourceSpan::default(),
            },
            operator_entry,
        )
        .map_err(|mut error| {
            error.message = format!("pattern builder: {}", error.message);
            error
        })
    }
}

/// An operator under construction; it borrows its owning builder.
#[must_use = "call output() or outputs() to add the operator"]
pub struct NodeBuilder<'a> {
    builder: &'a mut PatternBuilder,
    current: PendingExpression,
    alternatives: Vec<Expr>,
    completed: bool,
}

#[derive(Clone)]
enum PendingExpression {
    Operator {
        op: String,
        domain: Option<String>,
        inputs: Vec<Expr>,
        attributes: Vec<(String, Expr)>,
    },
    Value(Expr),
}

impl PendingExpression {
    fn context(&self) -> String {
        match self {
            Self::Operator { op, .. } => format!("operator {op:?}"),
            Self::Value(_) => "value alternative".into(),
        }
    }

    fn compile(self, builder: &mut PatternBuilder) -> Expr {
        match self {
            Self::Value(expr) => expr,
            Self::Operator {
                op,
                domain,
                inputs,
                attributes,
            } => {
                let domain = domain.as_deref().unwrap_or("");
                if !identifier(&op) || (!domain.is_empty() && !domain.split('.').all(identifier)) {
                    builder.errors.push(format!(
                        "operator {op:?} or domain {domain:?} is invalid; pass an operator identifier and use .domain(\"vendor.name\") separately"
                    ));
                }
                let name = if domain.is_empty() {
                    op
                } else {
                    format!("{domain}::{op}")
                };
                expression(ExprKind::Call(name, inputs, attributes))
            }
        }
    }
}

impl NodeBuilder<'_> {
    /// Append an ordered operator alternative. Subsequent domain/attribute
    /// configuration applies only to this new branch.
    pub fn or_node<I: Into<Input>>(
        mut self,
        op: &str,
        inputs: impl IntoIterator<Item = I>,
    ) -> Self {
        let next = self.builder.operation(op, inputs);
        let previous = std::mem::replace(&mut self.current, next);
        self.alternatives.push(previous.compile(self.builder));
        self
    }

    /// Append a passthrough alternative, preserving the original value identity.
    pub fn or_value(mut self, value: PatternValue) -> Self {
        let next = self.builder.value_expr(value, "alternative");
        let previous = std::mem::replace(&mut self.current, PendingExpression::Value(next));
        self.alternatives.push(previous.compile(self.builder));
        self
    }

    pub fn domain(mut self, domain: &str) -> Self {
        let PendingExpression::Operator {
            op,
            domain: current,
            ..
        } = &mut self.current
        else {
            self.builder.errors.push(
                "value alternative has no operator domain; apply domain() after node() or or_node()".into(),
            );
            return self;
        };
        if current.replace(domain.into()).is_some() {
            self.builder.errors.push(format!(
                "operator {op:?} domain is specified twice; set its domain once"
            ));
        }
        self
    }

    pub fn attr(mut self, name: &str, value: impl Into<AttributeValue>) -> Self {
        let PendingExpression::Operator { op, attributes, .. } = &mut self.current else {
            self.builder.errors.push(
                "value alternative has no attributes; apply attr() after node() or or_node()"
                    .into(),
            );
            return self;
        };
        if !identifier(name) || attributes.iter().any(|(key, _)| key == name) {
            self.builder.errors.push(format!(
                "operator {op:?} attribute {name:?} is invalid or duplicated; use each attribute identifier once"
            ));
        }
        let kind = match value.into() {
            AttributeValue::Int(v) => ExprKind::Literal(Literal::Int(v)),
            AttributeValue::Float(v) => ExprKind::Literal(Literal::Float(v)),
            AttributeValue::String(v) => ExprKind::Literal(Literal::String(v)),
            AttributeValue::Ints(v) => ExprKind::Literal(Literal::Ints(v)),
            AttributeValue::Floats(v) => ExprKind::Literal(Literal::Floats(v)),
            AttributeValue::Capture(v) => {
                self.builder.check(
                    v.owner,
                    v.index,
                    BindingClass::Attribute,
                    &format!("operator {op:?} attribute {name:?}"),
                );
                ExprKind::Name(internal_name(v.index))
            }
            AttributeValue::Graph(body) => {
                self.builder.depth = self.builder.depth.max(body.depth + 1);
                ExprKind::Graph(body.pattern)
            }
        };
        attributes.push((name.into(), expression(kind)));
        self
    }

    pub fn output(self) -> PatternValue {
        self.outputs::<1>()[0]
    }

    /// Exact output arity and identity; positions are never interchangeable.
    pub fn outputs<const N: usize>(mut self) -> [PatternValue; N] {
        if N == 0 {
            self.builder.errors.push(format!(
                "{} declares zero outputs; request at least one positional output",
                self.current.context()
            ));
        }
        let outputs = std::array::from_fn(|_| {
            let index = self.builder.binding(None, BindingKind::Output);
            PatternValue {
                owner: self.builder.owner,
                index,
            }
        });
        let targets = outputs
            .iter()
            .map(|v| Parameter {
                name: internal_name(v.index),
                annotation: None,
            })
            .collect();
        let current = self.current.clone().compile(self.builder);
        let expr = if self.alternatives.is_empty() {
            current
        } else {
            self.alternatives.push(current);
            expression(ExprKind::Alternatives(std::mem::take(
                &mut self.alternatives,
            )))
        };
        self.builder.statements.push((targets, expr));
        self.completed = true;
        outputs
    }
}

impl Drop for NodeBuilder<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.builder.errors.push(format!(
                "{} was not completed; call output() or outputs() before finishing the pattern",
                self.current.context()
            ));
        }
    }
}

fn internal_name(index: usize) -> String {
    format!("#value:{index}")
}

fn identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    name != "_"
        && name != "None"
        && bytes
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

fn expression(kind: ExprKind) -> Expr {
    Expr {
        kind,
        span: SourceSpan::default(),
    }
}

fn raw(name: &str) -> RawConstraint {
    RawConstraint {
        name: name.into(),
        element_variable: None,
        dimensions: Some(Vec::new()),
        prefix: false,
        immutable: false,
        sequence: None,
    }
}

fn optional_annotation(name: &str) -> Vec<RawConstraint> {
    vec![raw(name), raw("None")]
}

fn rename_expr(expr: &mut Expr, rename: &impl Fn(&mut String)) {
    match &mut expr.kind {
        ExprKind::Name(name) | ExprKind::Spread(name) => rename(name),
        ExprKind::Call(_, inputs, attributes) => {
            for input in inputs {
                rename_expr(input, rename);
            }
            for (_, attribute) in attributes {
                rename_expr(attribute, rename);
            }
        }
        ExprKind::Alternatives(branches) => {
            for branch in branches {
                rename_expr(branch, rename);
            }
        }
        _ => {}
    }
}

fn uses_attribute(expr: &Expr, name: &str) -> bool {
    match &expr.kind {
        ExprKind::Call(_, _, attributes) => attributes
            .iter()
            .any(|(_, expr)| matches!(&expr.kind, ExprKind::Name(n) if n == name)),
        ExprKind::Alternatives(branches) => {
            branches.iter().any(|branch| uses_attribute(branch, name))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_internal_handles_fail_closed_instead_of_indexing_or_mixing_kinds() {
        let mut p = Pattern::builder();
        let attr = p.attribute("axis", AttributeType::Int);
        let wrong_kind = PatternValue {
            owner: attr.owner,
            index: attr.index,
        };
        let undefined = PatternValue {
            owner: attr.owner,
            index: usize::MAX,
        };
        p.capture("bad", undefined);
        let y = p.op("Add", [wrong_kind, undefined]);
        let error = p.finish(y).unwrap_err();
        assert!(error.message.contains("wrong binding kind"), "{error}");
        assert!(error.message.contains("undefined binding"), "{error}");
    }
}
