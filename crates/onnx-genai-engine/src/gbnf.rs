//! Bounded translation of llama.cpp GBNF into model-token-aware llguidance Lark.
//!
//! This grammar describes the raw generated stream, independently of response
//! channel projection. Token references are resolved against the actual model.

pub(crate) mod lazy;
pub mod replay;
pub mod trigger;

use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, HashSet};

const MAX_DEPTH: usize = 256;
const MAX_REPETITION: usize = 2000;
const MAX_RULES: usize = 4096;
const MAX_ITEMS: usize = 65536;

#[derive(Debug)]
enum Atom {
    Literal(String),
    Class(String),
    Any,
    Reference(usize),
    Token { spelling: String, negate: bool },
    Group(Expression),
}

#[derive(Debug)]
struct Item {
    atom: Atom,
    // None is an unbounded upper extent, matching GBNF's {m,}.
    repeats: Vec<(usize, Option<usize>)>,
}

type Expression = Vec<Vec<Item>>;

struct Rule {
    name: String,
    offset: usize,
    expression: Option<Expression>,
}

struct Parser<'a> {
    source: &'a str,
    position: usize,
    rules: Vec<Rule>,
    names: BTreeMap<String, usize>,
    items: usize,
}

impl<'a> Parser<'a> {
    fn error(&self, message: impl std::fmt::Display) -> anyhow::Error {
        self.error_at(self.position, message)
    }

    fn error_at(&self, offset: usize, message: impl std::fmt::Display) -> anyhow::Error {
        let mut line = 1usize;
        let mut column = 1usize;
        let mut carriage_return = false;
        for character in self.source[..offset].chars() {
            match character {
                '\r' => {
                    line += 1;
                    column = 1;
                    carriage_return = true;
                }
                '\n' => {
                    if !carriage_return {
                        line += 1;
                    }
                    column = 1;
                    carriage_return = false;
                }
                _ => {
                    column += 1;
                    carriage_return = false;
                }
            }
        }
        anyhow::anyhow!(
            "grammar GBNF line {line}, column {column}: {message}; repair the grammar field"
        )
    }

    fn peek(&self) -> Option<char> {
        self.source[self.position..].chars().next()
    }

    fn take(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.position += character.len_utf8();
        Some(character)
    }

    fn space(&mut self, newline: bool) {
        loop {
            match self.peek() {
                Some(' ' | '\t') => {
                    self.take();
                }
                Some('\r' | '\n') if newline => {
                    self.take();
                }
                Some('#') => {
                    while self.peek().is_some_and(|c| c != '\r' && c != '\n') {
                        self.take();
                    }
                }
                _ => break,
            }
        }
    }

    fn expect(&mut self, expected: char) -> Result<()> {
        ensure!(
            self.peek() == Some(expected),
            self.error(format!("expected '{expected}'"))
        );
        self.take();
        Ok(())
    }

    fn name(&mut self) -> Result<String> {
        let start = self.position;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            self.take();
        }
        ensure!(
            self.position > start,
            self.error("expected a rule name (ASCII letters, digits or '-')")
        );
        Ok(self.source[start..self.position].to_owned())
    }

    fn rule_id(&mut self, name: String, offset: usize) -> Result<usize> {
        if let Some(&id) = self.names.get(&name) {
            return Ok(id);
        }
        ensure!(
            self.rules.len() < MAX_RULES,
            self.error(format!(
                "rule count exceeds {MAX_RULES}; reduce grammar complexity"
            ))
        );
        let id = self.rules.len();
        self.names.insert(name.clone(), id);
        self.rules.push(Rule {
            name,
            offset,
            expression: None,
        });
        Ok(id)
    }

    fn number(&mut self) -> Result<usize> {
        let start = self.position;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.take();
        }
        ensure!(
            self.position > start,
            self.error("expected a non-negative integer")
        );
        self.source[start..self.position]
            .parse()
            .map_err(|_| self.error_at(start, "integer exceeds this host's supported range"))
    }

    fn character(&mut self) -> Result<char> {
        let offset = self.position;
        let character = self
            .take()
            .ok_or_else(|| self.error("unexpected end of character"))?;
        if character != '\\' {
            return Ok(character);
        }
        let escaped = self
            .take()
            .ok_or_else(|| self.error("unfinished character escape"))?;
        match escaped {
            'n' => Ok('\n'),
            'r' => Ok('\r'),
            't' => Ok('\t'),
            '\\' | '"' | '[' | ']' | '-' => Ok(escaped),
            'x' | 'u' | 'U' => {
                let digits = match escaped {
                    'x' => 2,
                    'u' => 4,
                    _ => 8,
                };
                let mut scalar = 0u32;
                for _ in 0..digits {
                    let digit = self.take().and_then(|c| c.to_digit(16)).ok_or_else(|| {
                        self.error(format!("\\{escaped} requires {digits} hexadecimal digits"))
                    })?;
                    scalar = (scalar << 4) | digit;
                }
                char::from_u32(scalar).ok_or_else(|| {
                    self.error_at(
                        offset,
                        format!("U+{scalar:04X} is not a Unicode scalar value"),
                    )
                })
            }
            _ => Err(self.error_at(offset, format!("unsupported escape \\{escaped}"))),
        }
    }

    fn class(&mut self) -> Result<String> {
        self.expect('[')?;
        let negate = self.peek() == Some('^');
        if negate {
            self.take();
        }
        let mut class = if negate {
            String::from("[^")
        } else {
            String::from("[")
        };
        let mut count = 0usize;
        while self.peek() != Some(']') {
            let start = self.character()?;
            class.push_str(&format!("\\u{{{:x}}}", u32::from(start)));
            count += 1;
            if self.peek() == Some('-') && !self.source[self.position + 1..].starts_with(']') {
                self.take();
                let end = self.character()?;
                ensure!(
                    start <= end,
                    self.error("character range ends before it starts")
                );
                class.push_str(&format!("-\\u{{{:x}}}", u32::from(end)));
            }
        }
        ensure!(
            count > 0,
            self.error(
                "empty character class is not supported; specify characters or an empty string"
            )
        );
        self.expect(']')?;
        class.push(']');
        Ok(class)
    }

    fn atom(&mut self, depth: usize) -> Result<Atom> {
        self.items += 1;
        ensure!(
            self.items <= MAX_ITEMS,
            self.error(format!(
                "item count exceeds {MAX_ITEMS}; reduce grammar complexity"
            ))
        );
        match self.peek() {
            Some('"') => {
                self.take();
                let mut literal = String::new();
                while self.peek() != Some('"') {
                    literal.push(self.character()?);
                }
                self.expect('"')?;
                Ok(Atom::Literal(literal))
            }
            Some('[') => self.class().map(Atom::Class),
            Some('.') => {
                self.take();
                Ok(Atom::Any)
            }
            Some('(') => {
                ensure!(
                    depth < MAX_DEPTH,
                    self.error(format!("nesting exceeds {MAX_DEPTH}; reduce nested groups"))
                );
                self.take();
                self.space(true);
                let group = self.expression(true, depth + 1)?;
                self.expect(')')?;
                Ok(Atom::Group(group))
            }
            Some('<' | '!') => {
                let negate = self.peek() == Some('!');
                if negate {
                    self.take();
                }
                let start = self.position;
                self.expect('<')?;
                if self.peek() == Some('[') {
                    self.take();
                    self.number()?;
                    self.expect(']')?;
                } else {
                    while self.peek().is_some_and(|c| c != '>') {
                        self.take();
                    }
                }
                self.expect('>')?;
                Ok(Atom::Token {
                    spelling: self.source[start..self.position].to_owned(),
                    negate,
                })
            }
            Some(c) if c.is_ascii_alphanumeric() || c == '-' => {
                let offset = self.position;
                let name = self.name()?;
                self.rule_id(name, offset).map(Atom::Reference)
            }
            _ => {
                Err(self
                    .error("expected a literal, character class, rule, token reference or group"))
            }
        }
    }

    fn expression(&mut self, nested: bool, depth: usize) -> Result<Expression> {
        let mut branches = Vec::new();
        loop {
            let mut sequence = Vec::new();
            while !matches!(self.peek(), None | Some('|' | ')' | '\r' | '\n')) {
                let atom = self.atom(depth)?;
                self.space(nested);
                let mut repeats = Vec::new();
                let mut cost = 1usize;
                loop {
                    let bounds = match self.peek() {
                        Some('*') => {
                            self.take();
                            (0, None)
                        }
                        Some('+') => {
                            self.take();
                            (1, None)
                        }
                        Some('?') => {
                            self.take();
                            (0, Some(1))
                        }
                        Some('{') => {
                            self.take();
                            self.space(nested);
                            let min = self.number()?;
                            self.space(nested);
                            let max = if self.peek() == Some(',') {
                                self.take();
                                self.space(nested);
                                let max = if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                                    Some(self.number()?)
                                } else {
                                    None
                                };
                                self.space(nested);
                                max
                            } else {
                                Some(min)
                            };
                            self.expect('}')?;
                            ensure!(min <= MAX_REPETITION, self.error(format!("minimum repetition exceeds {MAX_REPETITION}; reduce repetitions")));
                            // llama.cpp treats an upper bound above its repetition budget as unbounded.
                            let max = max.filter(|&max| max <= MAX_REPETITION);
                            ensure!(
                                max.is_none_or(|max| max >= min),
                                self.error("maximum repetition is less than minimum")
                            );
                            (min, max)
                        }
                        _ => break,
                    };
                    ensure!(
                        !matches!(&atom, Atom::Literal(s) if s.is_empty()),
                        self.error("repetition requires a nonempty preceding item")
                    );
                    cost = cost
                        .checked_mul(bounds.1.unwrap_or(bounds.0.max(1)).max(1))
                        .ok_or_else(|| self.error("repetition complexity overflow"))?;
                    ensure!(
                        cost <= MAX_REPETITION,
                        self.error(format!(
                            "repetition complexity exceeds {MAX_REPETITION}; reduce repeated groups"
                        ))
                    );
                    ensure!(
                        repeats.len() < MAX_DEPTH,
                        self.error(
                            "too many consecutive repetition operators; simplify repeated items"
                        )
                    );
                    self.items += 1;
                    ensure!(
                        self.items <= MAX_ITEMS,
                        self.error("too many grammar items; simplify repeated items")
                    );
                    ensure!(
                        !repeats.last().is_some_and(|&(_, max)| max == Some(0)),
                        self.error("repetition requires a nonempty preceding item after {0}")
                    );
                    repeats.push(bounds);
                    self.space(nested);
                }
                sequence.push(Item { atom, repeats });
            }
            branches.push(sequence);
            if self.peek() != Some('|') {
                break;
            }
            self.take();
            self.space(true);
        }
        Ok(branches)
    }

    fn parse(mut self) -> Result<Self> {
        self.space(true);
        while self.peek().is_some() {
            let offset = self.position;
            let name = self.name()?;
            let id = self.rule_id(name, offset)?;
            self.space(false);
            for c in "::=".chars() {
                self.expect(c)?;
            }
            self.space(true);
            self.rules[id].expression = Some(self.expression(false, 0)?);
            ensure!(
                matches!(self.peek(), None | Some('\r' | '\n')),
                self.error("expected a newline after the rule")
            );
            self.space(true);
        }
        ensure!(
            self.names.contains_key("root"),
            self.error("missing root rule; define root ::= ...")
        );
        for rule in &self.rules {
            ensure!(
                rule.expression.is_some(),
                self.error_at(rule.offset, format!("undefined rule '{}'", rule.name))
            );
        }
        self.validate_structure()?;
        Ok(self)
    }

    fn validate_structure(&self) -> Result<()> {
        let mut nullable = vec![false; self.rules.len()];
        let mut iterations = 0usize;
        loop {
            iterations += 1;
            ensure!(iterations <= MAX_DEPTH + 1, self.error("nullable dependency chain exceeds the grammar depth budget; shorten chained rules"));
            let mut changed = false;
            for (id, rule) in self.rules.iter().enumerate() {
                if !nullable[id]
                    && expression_nullable(
                        rule.expression.as_ref().context("undefined GBNF rule")?,
                        &nullable,
                    )
                {
                    nullable[id] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut edges = Vec::new();
        for rule in &self.rules {
            let expression = rule.expression.as_ref().context("undefined GBNF rule")?;
            expression_cost(expression, &nullable)
                .map_err(|error| self.error_at(rule.offset, error))?;
            let mut references = Vec::new();
            first_references(expression, &nullable, &mut references);
            edges.push(references);
        }
        let mut active = vec![false; self.rules.len()];
        let mut depths = vec![None; self.rules.len()];
        for id in 0..self.rules.len() {
            reference_depth(id, &edges, &mut active, &mut depths, 0).map_err(|error| {
                self.error_at(
                    self.rules[id].offset,
                    format!("rule '{}': {error}", self.rules[id].name),
                )
            })?;
        }
        Ok(())
    }
}

fn atom_nullable(atom: &Atom, nullable: &[bool]) -> bool {
    match atom {
        Atom::Literal(value) => value.is_empty(),
        Atom::Reference(id) => nullable[*id],
        Atom::Group(expression) => expression_nullable(expression, nullable),
        _ => false,
    }
}

fn item_nullable(item: &Item, nullable: &[bool]) -> bool {
    item.repeats.iter().any(|&(min, _)| min == 0) || atom_nullable(&item.atom, nullable)
}

fn expression_nullable(expression: &Expression, nullable: &[bool]) -> bool {
    expression
        .iter()
        .any(|sequence| sequence.iter().all(|item| item_nullable(item, nullable)))
}

fn first_references(expression: &Expression, nullable: &[bool], references: &mut Vec<usize>) {
    for sequence in expression {
        for item in sequence {
            if !item.repeats.iter().any(|&(_, max)| max == Some(0)) {
                match &item.atom {
                    Atom::Reference(id) => references.push(*id),
                    Atom::Group(expression) => first_references(expression, nullable, references),
                    _ => {}
                }
            }
            if !item_nullable(item, nullable) {
                break;
            }
        }
    }
}

fn expression_cost(expression: &Expression, nullable: &[bool]) -> Result<usize> {
    let mut total = 0usize;
    for sequence in expression {
        for item in sequence {
            let mut cost = match &item.atom {
                Atom::Group(group) => expression_cost(group, nullable)?.max(1),
                _ => 1,
            };
            let mut empty = atom_nullable(&item.atom, nullable);
            for &(min, max) in &item.repeats {
                ensure!(
                    !empty || max.is_some(),
                    "unbounded repetition of a nullable item creates left recursion; make its repeated branch consume input"
                );
                empty |= min == 0;
                cost = cost.saturating_mul(max.unwrap_or(min.max(1)));
                ensure!(
                    cost <= MAX_ITEMS,
                    "expanded repetition exceeds the {MAX_ITEMS}-item grammar budget; reduce nested repetitions"
                );
            }
            total = total.saturating_add(cost);
            ensure!(
                total <= MAX_ITEMS,
                "expanded grammar exceeds the {MAX_ITEMS}-item budget; reduce alternatives or repetitions"
            );
        }
    }
    Ok(total)
}

fn reference_depth(
    id: usize,
    edges: &[Vec<usize>],
    active: &mut [bool],
    depths: &mut [Option<usize>],
    level: usize,
) -> Result<usize> {
    ensure!(
        level <= MAX_DEPTH,
        "rule chain exceeds {MAX_DEPTH}; shorten chained references"
    );
    ensure!(
        !active[id],
        "left recursion is unsupported by GBNF; make the recursive branch consume a character or token before referring back"
    );
    if let Some(depth) = depths[id] {
        return Ok(depth);
    }
    active[id] = true;
    let mut depth = 1;
    for &next in &edges[id] {
        depth = depth.max(1 + reference_depth(next, edges, active, depths, level + 1)?);
    }
    active[id] = false;
    ensure!(
        depth <= MAX_DEPTH,
        "rule chain exceeds {MAX_DEPTH}; shorten chained references"
    );
    depths[id] = Some(depth);
    Ok(depth)
}

/// Check GBNF syntax and resource bounds without loading a model vocabulary.
pub fn validate_gbnf(source: &str, max_bytes: usize) -> Result<()> {
    parse(source, max_bytes).map(|_| ())
}

fn parse(source: &str, max_bytes: usize) -> Result<Parser<'_>> {
    ensure!(
        max_bytes > 0,
        "GBNF max_constraint_bytes must be positive; configure a positive grammar byte budget"
    );
    ensure!(
        source.len() <= max_bytes,
        "grammar field is {} bytes, exceeding its {max_bytes}-byte budget; shorten the GBNF grammar",
        source.len()
    );
    Parser {
        source,
        position: 0,
        rules: Vec::new(),
        names: BTreeMap::new(),
        items: 0,
    }
    .parse()
}

fn render(
    expression: &Expression,
    resolve_token: &mut impl FnMut(&str) -> Result<u32>,
) -> Result<String> {
    expression.iter().map(|sequence| {
        sequence.iter().map(|item| {
            let mut rendered = match &item.atom {
                Atom::Literal(text) => serde_json::to_string(text)?,
                Atom::Class(class) => format!("/{class}/"),
                Atom::Any => "/(?s:.)/".to_owned(),
                Atom::Reference(id) => format!("r_{id}"),
                Atom::Group(group) => format!("({})", render(group, resolve_token)?),
                Atom::Token { spelling, negate } => {
                    let id = resolve_token(spelling).with_context(|| format!("resolve grammar token reference {spelling}; use an existing model token ID or a spelling that tokenizes to exactly one token"))?;
                    format!("<[{}{id}]>", if *negate { "^" } else { "" })
                }
            };
            for &(min, max) in &item.repeats {
                rendered = match max {
                    Some(0) => "\"\"".to_owned(),
                    Some(max) if min == max => format!("({rendered}){{{min}}}"),
                    Some(max) => format!("({rendered}){{{min},{max}}}"),
                    None => format!("({rendered}){{{min},}}"),
                };
            }
            Ok(rendered)
        }).collect::<Result<Vec<_>>>().map(|items| items.join(" "))
    }).collect::<Result<Vec<_>>>().map(|branches| branches.join(" | "))
}

pub(crate) fn compile(
    source: &str,
    max_bytes: usize,
    tokenizer: &onnx_genai_ort::Tokenizer,
) -> Result<String> {
    let parsed = parse(source, max_bytes)?;
    render_rules(&parsed, &mut token_resolver(tokenizer))
}

pub(crate) fn token_resolver(
    tokenizer: &onnx_genai_ort::Tokenizer,
) -> impl FnMut(&str) -> Result<u32> + '_ {
    let vocabulary: HashSet<_> = tokenizer.inner().get_vocab(true).into_values().collect();
    move |spelling: &str| -> Result<u32> {
        let id = if let Some(number) = spelling
            .strip_prefix("<[")
            .and_then(|s| s.strip_suffix("]>"))
        {
            number
                .parse::<u32>()
                .context("grammar token ID exceeds u32")?
        } else {
            let ids = tokenizer.encode_with_options(
                spelling,
                onnx_genai_ort::TokenizationOptions {
                    add_special: false,
                    parse_special: true,
                },
            )?;
            ensure!(
                ids.len() == 1,
                "grammar token {spelling} tokenizes to {} tokens instead of one",
                ids.len()
            );
            ids[0]
        };
        ensure!(
            vocabulary.contains(&id),
            "grammar token ID {id} is not present in the model vocabulary"
        );
        Ok(id)
    }
}

fn render_rules(
    parsed: &Parser<'_>,
    resolve: &mut impl FnMut(&str) -> Result<u32>,
) -> Result<String> {
    let root = parsed
        .names
        .get("root")
        .context("GBNF root was not resolved")?;
    // GBNF accepts every model-token spelling of a legal string, rather than
    // forcing one canonical tokenizer encoding of grammar-determined text.
    let mut lark = format!("%llguidance {{\"no_forcing\":true}}\nstart: r_{root}\n");
    for (id, rule) in parsed.rules.iter().enumerate() {
        let expression = rule
            .expression
            .as_ref()
            .context("GBNF rule was not defined")?;
        lark.push_str(&format!("r_{id}: {}\n", render(expression, resolve)?));
    }
    Ok(lark)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translated(source: &str) -> Result<String> {
        render_rules(&parse(source, 65536)?, &mut |spelling| match spelling {
            "<think>" => Ok(4),
            "</think>" => Ok(5),
            "<[19]>" => Ok(19),
            _ => anyhow::bail!("unknown test token {spelling}"),
        })
    }

    #[test]
    fn translates_recursive_unicode_ranges_and_raw_token_references() -> Result<()> {
        let grammar = translated(
            "# opening\r\nroot ::= <think> !</think>* </think> (word |\n \"done\")? <[19]>\nword ::= [ぁ-ゟ\\U0001f600] word? | [^\\n] .\n",
        )?;
        assert!(grammar.contains("<[4]> (<[^5]>){0,} <[5]>"), "{grammar}");
        assert!(
            grammar.contains("\\u{3041}-\\u{309f}\\u{1f600}"),
            "{grammar}"
        );
        assert!(grammar.contains("r_1"));
        assert!(grammar.contains("<[19]>"));
        Ok(())
    }

    #[test]
    fn preserves_literal_backslashes_and_repetition_contract() -> Result<()> {
        let grammar =
            translated(r#"root ::= "\\n\x41\u00e9\U0001f600"{2,4} [a-z]{0} "x"{3,} "y"{0,3000}"#)?;
        assert!(grammar.contains(r#"("\\nAé😀"){2,4}"#), "{grammar}");
        assert!(grammar.contains("\"\""));
        assert!(!grammar.contains("{0}"));
        assert!(grammar.contains("{3,}"));
        assert!(grammar.contains("{0,}"));
        Ok(())
    }

    #[test]
    fn syntax_failures_name_the_field_line_column_and_repair() {
        for source in [
            "root ::= [z-a]",
            "root ::= missing",
            "other ::= \"x\"",
            "root ::= \"\\q\"",
            "root ::= \"\\uD800\"",
            "root ::= x_name",
            "root ::= \"x\"{3,2}",
            "root ::= \"x\"{2001}",
            "root ::= !x",
            "root ::= (\"x\"",
            "root ::= <[no]>",
        ] {
            let error = validate_gbnf(source, 65536).unwrap_err().to_string();
            assert!(
                error.contains("grammar GBNF line")
                    && error.contains("column")
                    && error.contains("repair"),
                "{source}: {error}"
            );
        }
        let error = validate_gbnf("root ::=\n \"x\"\nnext ::= missing", 65536)
            .unwrap_err()
            .to_string();
        assert!(error.contains("line 3"), "{error}");
        let error = validate_gbnf("root ::= \"x\"\rnext ::= missing", 65536)
            .unwrap_err()
            .to_string();
        assert!(error.contains("line 2"), "{error}");
    }

    #[test]
    fn rejects_left_nullable_recursion_and_expanded_repetition_bombs() {
        for (grammar, cause) in [
            ("root ::= root | \"x\"", "left recursion"),
            ("root ::= a \"x\"\na ::= \"\" | root", "left recursion"),
            ("root ::= (\"\" | \"x\")*", "nullable"),
            ("root ::= (\"x\"{2000}){2000}", "expanded repetition"),
        ] {
            let error = validate_gbnf(grammar, 65536).unwrap_err().to_string();
            assert!(error.contains(cause), "{grammar}: {error}");
        }
    }

    #[test]
    fn enforces_source_nesting_and_repeat_complexity_budgets() {
        assert!(
            validate_gbnf("root ::= \"x\"", 0)
                .unwrap_err()
                .to_string()
                .contains("positive")
        );
        assert!(
            validate_gbnf("root ::= \"x\"", 2)
                .unwrap_err()
                .to_string()
                .contains("byte budget")
        );
        let nested = format!(
            "root ::= {}\"x\"{}",
            "(".repeat(MAX_DEPTH + 1),
            ")".repeat(MAX_DEPTH + 1)
        );
        assert!(
            validate_gbnf(&nested, 65536)
                .unwrap_err()
                .to_string()
                .contains("nesting")
        );
        assert!(
            validate_gbnf("root ::= \"x\"{100}{100}", 65536)
                .unwrap_err()
                .to_string()
                .contains("complexity")
        );
    }
}
