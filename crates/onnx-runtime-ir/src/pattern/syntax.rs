use super::*;

// Independently implemented from the public grammar/contract, not ported C++.
const MAX_SOURCE: usize = 1 << 20;
const MAX_TOKENS: usize = 65_536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Parameter {
    pub name: String,
    pub annotation: Option<Vec<RawConstraint>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RawConstraint {
    pub name: String,
    pub element_variable: Option<String>,
    pub dimensions: Option<Vec<Dimension>>,
    pub prefix: bool,
    pub immutable: bool,
    pub sequence: Option<(usize, Option<usize>, bool)>,
}

#[derive(Clone, Debug)]
pub(super) struct Definition {
    pub named: bool,
    pub interface: Option<String>,
    pub parameters: Vec<Parameter>,
    pub statements: Vec<(Vec<Parameter>, Expr)>,
    pub returns: Vec<Expr>,
    pub output_signature: Option<Vec<Parameter>>,
    pub return_labels: Option<Vec<String>>,
    pub call_parameters: Option<usize>,
    pub require_outputs: bool,
    pub span: SourceSpan,
}

#[derive(Clone, Debug)]
pub(super) enum DefinitionKind {
    Interface(Vec<Parameter>, Vec<Parameter>, SourceSpan),
    Body(Definition),
    Alternatives(Vec<String>, SourceSpan),
}

#[derive(Clone, Debug)]
pub(super) struct Expr {
    pub kind: ExprKind,
    pub span: SourceSpan,
}

#[derive(Clone, Debug)]
pub(super) enum ExprKind {
    Name(String),
    Call(String, Vec<Expr>, Vec<(String, Expr)>),
    Literal(Literal),
    Alternatives(Vec<Expr>),
    Spread(String),
    GraphList(Vec<String>),
    Graph(Pattern),
    AnonymousCall(Vec<String>, Vec<Expr>),
}

#[derive(Clone, Debug)]
struct Token {
    text: String,
    string: bool,
    span: SourceSpan,
}

pub(super) fn parse_inline(source: &str, native: bool) -> Result<Definition> {
    let mut parser = Parser::new(source, native)?;
    let definition = parser.body(false)?;
    parser.end()?;
    Ok(definition)
}

pub(super) fn parse_library(
    source: &str,
    native: bool,
) -> Result<BTreeMap<String, DefinitionKind>> {
    let mut parser = Parser::new(source, native)?;
    let mut definitions = BTreeMap::new();
    while parser.peek().is_some() {
        if parser.take(";") {
            continue;
        }
        let interface = parser.take("interface");
        if !interface {
            parser.expect("pattern")?;
        }
        let name = parser.identifier()?;
        let span = parser.span();
        let definition = if interface {
            let inputs = parser.parameters()?;
            parser.expect("->")?;
            DefinitionKind::Interface(inputs, parser.parameters()?, span)
        } else if parser.take("=") {
            let mut members = vec![parser.identifier()?];
            while parser.take("|") {
                let member = parser.identifier()?;
                if members.contains(&member) {
                    return Err(parser.error(
                        PatternErrorKind::InvalidPattern,
                        "duplicate alternative member; list each named member once",
                    ));
                }
                members.push(member);
            }
            parser.take(";");
            DefinitionKind::Alternatives(members, span)
        } else {
            DefinitionKind::Body(parser.body(true)?)
        };
        if definitions.insert(name.clone(), definition).is_some() {
            return Err(parser.error(
                PatternErrorKind::InvalidPattern,
                format!("duplicate definition {name:?}; give each pattern a unique name"),
            ));
        }
        if definitions.len() > 256 {
            return Err(parser.error(
                PatternErrorKind::BudgetExceeded,
                "library exceeds 256 definitions; split it into smaller libraries",
            ));
        }
    }
    Ok(definitions)
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
    length: usize,
    native: bool,
}

impl Parser {
    fn new(source: &str, native: bool) -> Result<Self> {
        let fail = |start, end, message| {
            PatternError::new(PatternErrorKind::Syntax, SourceSpan { start, end }, message)
        };
        if source.len() > MAX_SOURCE {
            return Err(fail(
                0,
                source.len(),
                "pattern source exceeds 1 MiB; split the library",
            ));
        }
        let mut tokens = Vec::new();
        let bytes = source.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let start = i;
            let byte = bytes[i];
            if byte.is_ascii_whitespace() {
                i += 1;
                continue;
            }
            if byte == b'#' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            let string = byte == b'"' || byte == b'\'';
            let text = if string {
                i += 1;
                let mut value = Vec::new();
                while i < bytes.len() && bytes[i] != byte {
                    if bytes[i] == b'\\' {
                        i += 1;
                        if i == bytes.len() || ![byte, b'\\', b'"', b'\''].contains(&bytes[i]) {
                            return Err(fail(
                                start,
                                i,
                                "unsupported string escape; escape quotes or backslashes",
                            ));
                        }
                    }
                    value.push(bytes[i]);
                    i += 1;
                }
                if i == bytes.len() {
                    return Err(fail(start, i, "unterminated string; add its closing quote"));
                }
                i += 1;
                String::from_utf8(value).map_err(|_| fail(start, i, "string is not valid UTF-8"))?
            } else if source[i..].starts_with("...") {
                i += 3;
                "...".into()
            } else if ["=>", "->", "::"]
                .iter()
                .any(|token| source[i..].starts_with(token))
            {
                i += 2;
                source[start..i].into()
            } else if byte.is_ascii_alphabetic() || byte == b'_' {
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                source[start..i].into()
            } else if byte.is_ascii_digit()
                || (byte == b'-' && bytes.get(i + 1).is_some_and(u8::is_ascii_digit))
            {
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || b".eE+-".contains(&bytes[i]))
                {
                    i += 1;
                }
                source[start..i].into()
            } else if b"(){}[],;:=|.<>?*".contains(&byte) {
                i += 1;
                source[start..i].into()
            } else {
                let width = source[start..].chars().next().map_or(1, char::len_utf8);
                return Err(fail(
                    start,
                    start + width,
                    "unexpected character; use the supported ORT pattern dialect",
                ));
            };
            tokens.push(Token {
                text,
                string,
                span: SourceSpan { start, end: i },
            });
            if tokens.len() > MAX_TOKENS {
                return Err(fail(
                    start,
                    i,
                    "pattern exceeds 65536 tokens; split the library",
                ));
            }
        }
        Ok(Self {
            tokens,
            position: 0,
            length: source.len(),
            native,
        })
    }

    fn peek(&self) -> Option<&str> {
        self.tokens
            .get(self.position)
            .map(|token| token.text.as_str())
    }
    fn span(&self) -> SourceSpan {
        self.tokens.get(self.position).map_or(
            SourceSpan {
                start: self.length,
                end: self.length,
            },
            |token| token.span,
        )
    }
    fn error(&self, kind: PatternErrorKind, message: impl Into<String>) -> PatternError {
        PatternError::new(kind, self.span(), message)
    }
    fn take(&mut self, text: &str) -> bool {
        if self.peek() == Some(text) && !self.tokens[self.position].string {
            self.position += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, text: &str) -> Result<()> {
        if self.take(text) {
            Ok(())
        } else {
            Err(self.error(
                PatternErrorKind::Syntax,
                format!(
                    "expected {text:?}, found {:?}; check the ORT-pattern entry syntax",
                    self.peek()
                ),
            ))
        }
    }
    fn identifier(&mut self) -> Result<String> {
        let Some(token) = self.tokens.get(self.position) else {
            return Err(self.error(
                PatternErrorKind::Syntax,
                "expected a name; complete the pattern",
            ));
        };
        if token.string
            || !token.text.as_bytes()[0].is_ascii_alphabetic() && !token.text.starts_with('_')
        {
            return Err(self.error(
                PatternErrorKind::Unsupported,
                "expected an unquoted identifier; quoted ONNX model names are not pattern bindings",
            ));
        }
        let text = token.text.clone();
        self.position += 1;
        Ok(text)
    }
    fn qualified(&mut self) -> Result<String> {
        let mut name = self.identifier()?;
        while self.peek() == Some(".") || self.peek() == Some("::") {
            let separator = self.peek().unwrap().to_string();
            self.position += 1;
            name.push_str(&separator);
            name.push_str(&self.identifier()?);
        }
        Ok(name)
    }
    fn end(&self) -> Result<()> {
        if self.peek().is_none() {
            Ok(())
        } else {
            Err(self.error(PatternErrorKind::Unsupported,
                "unexpected trailing syntax; use PatternLibrary for named patterns/interfaces; ONNX models and replacements require a different frontend"))
        }
    }
    fn parameters(&mut self) -> Result<Vec<Parameter>> {
        self.expect("(")?;
        let mut values = Vec::new();
        if !self.take(")") {
            loop {
                values.push(self.parameter()?);
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(values)
    }
    fn parameter(&mut self) -> Result<Parameter> {
        let name = self.identifier()?;
        if name == "None" || name == "_" {
            return Err(self.error(
                PatternErrorKind::InvalidPattern,
                "None and _ are reserved; choose a capture name",
            ));
        }
        let annotation = if self.take(":") {
            let mut alternatives = vec![self.constraint()?];
            while self.take("|") {
                alternatives.push(self.constraint()?);
            }
            if alternatives.len() > 64 {
                return Err(self.error(
                    PatternErrorKind::BudgetExceeded,
                    "annotation exceeds 64 alternatives; simplify its type contract",
                ));
            }
            Some(alternatives)
        } else {
            None
        };
        Ok(Parameter { name, annotation })
    }

    fn assignment_target(&mut self) -> Result<Parameter> {
        if matches!(self.peek(), Some("None" | "_")) {
            let name = self.identifier()?;
            if self.peek() == Some(":") {
                return Err(self.error(
                    PatternErrorKind::InvalidPattern,
                    "absent and wildcard output slots cannot have annotations",
                ));
            }
            return Ok(Parameter {
                name,
                annotation: None,
            });
        }
        self.parameter()
    }
    fn constraint(&mut self) -> Result<RawConstraint> {
        let immutable = self.take("const");
        let name = self.identifier()?;
        if name == "value" {
            if !self.native {
                return Err(self.error(
                    PatternErrorKind::Unsupported,
                    "untyped value annotations are a native extension; use parse_native, not parse_ort",
                ));
            }
            if matches!(self.peek(), Some("[" | "<")) {
                return Err(self.error(
                    PatternErrorKind::InvalidPattern,
                    "value has no dtype or shape contract; use tensor annotations or a metadata predicate",
                ));
            }
            return Ok(RawConstraint {
                name,
                sequence: None,
                element_variable: None,
                dimensions: None,
                prefix: false,
                immutable,
            });
        }
        if ["values", "slots"].contains(&name.as_str()) {
            if !self.native {
                return Err(self.error(
                    PatternErrorKind::Unsupported,
                    "sequence annotations are a native extension; use parse_native, not parse_ort",
                ));
            }
            self.expect("[")?;
            let min = self.extent()?;
            self.expect(":")?;
            let max = if self.take("*") {
                None
            } else {
                Some(self.extent()?)
            };
            self.expect("]")?;
            if max.is_some_and(|max| max < min) || immutable {
                return Err(self.error(
                    PatternErrorKind::InvalidPattern,
                    "sequence needs min <= max and no const qualifier",
                ));
            }
            return Ok(RawConstraint {
                sequence: Some((min, max, name == "slots")),
                name,
                element_variable: None,
                dimensions: None,
                prefix: false,
                immutable: false,
            });
        }
        let element_variable = if self.take("<") {
            let name = self.identifier()?;
            self.expect(">")?;
            Some(name)
        } else {
            None
        };
        let mut prefix = false;
        let dimensions = if self.take("[") {
            let mut dimensions = Vec::new();
            if !self.take("]") {
                loop {
                    if self.take("...") {
                        prefix = true;
                        self.expect("]")?;
                        break;
                    }
                    let dimension = if self.take("_") {
                        Dimension::Any
                    } else if self.tokens.get(self.position).is_some_and(|t| {
                        !t.string && t.text.starts_with(|c: char| c.is_ascii_digit())
                    }) {
                        let number = self.peek().unwrap().parse().map_err(|_| {
                            self.error(
                                PatternErrorKind::Syntax,
                                "dimension overflows usize; use a representable nonnegative extent",
                            )
                        })?;
                        self.position += 1;
                        Dimension::Fixed(number)
                    } else {
                        Dimension::Named(self.identifier()?)
                    };
                    dimensions.push(dimension);
                    if dimensions.len() > 64 {
                        return Err(self.error(PatternErrorKind::BudgetExceeded,
                            "annotation exceeds 64 dimensions; use an any-rank constraint and a predicate"));
                    }
                    if self.take("]") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            if dimensions.is_empty() {
                None
            } else {
                Some(dimensions)
            }
        } else {
            Some(Vec::new())
        };
        let prefix = prefix && dimensions.is_some();
        Ok(RawConstraint {
            name,
            dimensions,
            prefix,
            immutable,
            element_variable,
            sequence: None,
        })
    }
    fn extent(&mut self) -> Result<usize> {
        let value = self
            .peek()
            .and_then(|token| token.parse().ok())
            .ok_or_else(|| {
                self.error(
                    PatternErrorKind::Syntax,
                    "sequence extent must be a nonnegative usize",
                )
            })?;
        self.position += 1;
        Ok(value)
    }
    fn body(&mut self, named: bool) -> Result<Definition> {
        let span = self.span();
        let interface = if named && self.take("implements") {
            Some(self.identifier()?)
        } else {
            None
        };
        let parameters = if interface.is_some() {
            if self.take(":") {
                let extra = self.parameters()?;
                self.expect("->")?;
                extra
            } else {
                Vec::new()
            }
        } else {
            self.parameters()?
        };
        let output_signature = if named && interface.is_none() && self.take("->") {
            Some(self.parameters()?)
        } else {
            None
        };
        if !named {
            self.expect("=>")?;
        }
        let braces = self.take("{");
        if named && !braces {
            return Err(self.error(PatternErrorKind::Syntax, "named patterns need a { body }"));
        }
        let mut statements = Vec::new();
        while !self.take("return") {
            if self.take(";") {
                continue;
            }
            if self.peek().is_none() || self.peek() == Some("}") {
                return Err(self.error(
                    PatternErrorKind::Syntax,
                    "pattern needs explicit return values or a return operator call",
                ));
            }
            let mut targets = vec![self.assignment_target()?];
            while self.take(",") {
                targets.push(self.assignment_target()?);
            }
            self.expect("=")?;
            let expression = self.expression(0)?;
            statements.push((targets, expression));
            if statements.len() > 256 {
                return Err(self.error(
                    PatternErrorKind::BudgetExceeded,
                    "more than 256 statements; factor the pattern",
                ));
            }
            self.take(";");
        }
        let explicit_ports = named
            && self.peek() == Some("(")
            && self
                .tokens
                .get(self.position + 2)
                .is_some_and(|t| t.text == "=");
        let (returns, return_labels) = if explicit_ports {
            self.expect("(")?;
            let mut labels = Vec::new();
            let mut returns = Vec::new();
            loop {
                let label = self.identifier()?;
                if labels.contains(&label) {
                    return Err(self.error(
                        PatternErrorKind::InvalidPattern,
                        "duplicate return port; map each output label exactly once",
                    ));
                }
                labels.push(label);
                self.expect("=")?;
                let span = self.span();
                returns.push(Expr {
                    kind: ExprKind::Name(self.qualified()?),
                    span,
                });
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
            (returns, Some(labels))
        } else {
            let mut returns = vec![self.expression(0)?];
            while self.take(",") {
                returns.push(self.expression(0)?);
            }
            (returns, None)
        };
        self.take(";");
        if braces {
            self.expect("}")?;
        }
        Ok(Definition {
            named,
            require_outputs: output_signature.is_some() || interface.is_some(),
            interface,
            parameters,
            statements,
            returns,
            output_signature,
            return_labels,
            call_parameters: None,
            span,
        })
    }
    fn expression(&mut self, depth: usize) -> Result<Expr> {
        if depth >= 64 {
            return Err(self.error(
                PatternErrorKind::BudgetExceeded,
                "expression nesting reaches 64; factor or flatten the expression",
            ));
        }
        let span = self.span();
        let mut branches = vec![self.atom(depth + 1)?];
        while self.take("|") {
            branches.push(self.atom(depth + 1)?);
            if branches.len() > 64 {
                return Err(self.error(
                    PatternErrorKind::BudgetExceeded,
                    "more than 64 alternatives; split the pattern",
                ));
            }
        }
        if branches.len() == 1 {
            Ok(branches.pop().unwrap())
        } else {
            Ok(Expr {
                span,
                kind: ExprKind::Alternatives(branches),
            })
        }
    }
    fn atom(&mut self, depth: usize) -> Result<Expr> {
        let span = self.span();
        if self.take("...") {
            if !self.native {
                return Err(self.error(
                    PatternErrorKind::Unsupported,
                    "variadic operands require parse_native; ORT portable syntax has exact slots",
                ));
            }
            return Ok(Expr {
                span,
                kind: ExprKind::Spread(self.identifier()?),
            });
        }
        if self.peek() == Some("[")
            && self.tokens.get(self.position + 1).is_some_and(|token| {
                !token.string && token.text.starts_with(|c: char| c.is_ascii_alphabetic())
            })
        {
            if !self.native {
                return Err(self.error(
                    PatternErrorKind::Unsupported,
                    "GRAPH lists require parse_native; pinned ORT does not support GRAPHS",
                ));
            }
            self.expect("[")?;
            let mut names = vec![self.identifier()?];
            while self.take(",") {
                names.push(self.identifier()?);
            }
            self.expect("]")?;
            return Ok(Expr {
                span,
                kind: ExprKind::GraphList(names),
            });
        }
        if self.take("(") {
            let expression = self.expression(depth)?;
            self.expect(")")?;
            if self.peek() == Some("(") {
                let branches = match expression.kind {
                    ExprKind::Alternatives(branches) => branches,
                    kind => vec![Expr {
                        kind,
                        span: expression.span,
                    }],
                };
                let mut members = Vec::new();
                for branch in branches {
                    let ExprKind::Name(name) = branch.kind else {
                        return Err(self.error(PatternErrorKind::InvalidPattern,
                            "anonymous calls require named pattern members, not operator expressions"));
                    };
                    if members.contains(&name) {
                        return Err(self.error(
                            PatternErrorKind::InvalidPattern,
                            "duplicate anonymous alternative member; list each member once",
                        ));
                    }
                    members.push(name);
                }
                self.expect("(")?;
                let mut arguments = Vec::new();
                if !self.take(")") {
                    loop {
                        arguments.push(self.expression(depth)?);
                        if self.take(")") {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
                return Ok(Expr {
                    kind: ExprKind::AnonymousCall(members, arguments),
                    span,
                });
            }
            return Ok(expression);
        }
        if self.peek() == Some("[")
            || self.tokens.get(self.position).is_some_and(|t| {
                t.string || t.text.starts_with('-') || t.text.as_bytes()[0].is_ascii_digit()
            })
        {
            return Ok(Expr {
                span,
                kind: ExprKind::Literal(self.literal()?),
            });
        }
        let name = if self.take("*") {
            if !self.native {
                return Err(self.error(
                    PatternErrorKind::Unsupported,
                    "operator wildcards require parse_native",
                ));
            }
            "*".into()
        } else {
            self.qualified()?
        };
        if !self.take("(") {
            return Ok(Expr {
                span,
                kind: ExprKind::Name(name),
            });
        }
        let mut inputs = Vec::new();
        let mut attributes = Vec::new();
        if !self.take(")") {
            loop {
                if self
                    .tokens
                    .get(self.position + 1)
                    .is_some_and(|t| t.text == "=")
                {
                    let name = self.identifier()?;
                    self.expect("=")?;
                    if attributes.iter().any(|(key, _)| key == &name) {
                        return Err(self.error(
                            PatternErrorKind::InvalidPattern,
                            format!("attribute {name:?} listed twice; constrain it once"),
                        ));
                    }
                    attributes.push((name, self.expression(depth)?));
                } else {
                    if !attributes.is_empty() {
                        return Err(self.error(
                            PatternErrorKind::Syntax,
                            "positional inputs must precede keyword attributes",
                        ));
                    }
                    inputs.push(self.expression(depth)?);
                }
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(Expr {
            span,
            kind: ExprKind::Call(name, inputs, attributes),
        })
    }
    fn literal(&mut self) -> Result<Literal> {
        if self.take("[") {
            let mut values = Vec::new();
            if !self.take("]") {
                loop {
                    if self.peek() == Some("[") && !self.tokens[self.position].string {
                        return Err(self.error(PatternErrorKind::Unsupported,
                            "literal lists must be flat and uniformly int64 or float32; use a predicate for other tensors"));
                    }
                    values.push(self.literal()?);
                    if self.take("]") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            if values.iter().all(|v| matches!(v, Literal::Int(_))) {
                return Ok(Literal::Ints(
                    values
                        .into_iter()
                        .map(|v| {
                            if let Literal::Int(v) = v {
                                v
                            } else {
                                unreachable!()
                            }
                        })
                        .collect(),
                ));
            }
            if values.iter().all(|v| matches!(v, Literal::Float(_))) {
                return Ok(Literal::Floats(
                    values
                        .into_iter()
                        .map(|v| {
                            if let Literal::Float(v) = v {
                                v
                            } else {
                                unreachable!()
                            }
                        })
                        .collect(),
                ));
            }
            return Err(self.error(PatternErrorKind::Unsupported,
                "literal lists must be flat and uniformly int64 or float32; use a predicate for other tensors"));
        }
        let token = self.tokens.get(self.position).cloned().ok_or_else(|| {
            self.error(
                PatternErrorKind::Syntax,
                "expected a numeric or string literal",
            )
        })?;
        self.position += 1;
        if token.string {
            return Ok(Literal::String(token.text.into_bytes()));
        }
        if token.text.contains(['.', 'e', 'E']) {
            let value: f32 = token.text.parse().map_err(|_| {
                PatternError::new(
                    PatternErrorKind::Syntax,
                    token.span,
                    "invalid float literal; use a finite float32 number",
                )
            })?;
            if !value.is_finite() {
                return Err(PatternError::new(
                    PatternErrorKind::Syntax,
                    token.span,
                    "float literal is not finite float32; use a finite representable number",
                ));
            }
            Ok(Literal::Float(value))
        } else {
            Ok(Literal::Int(token.text.parse().map_err(|_| {
                PatternError::new(
                    PatternErrorKind::Syntax,
                    token.span,
                    "invalid int64 literal; use a signed 64-bit integer",
                )
            })?))
        }
    }
}
