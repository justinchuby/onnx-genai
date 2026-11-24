//! Bounded byte-native ECMAScript trigger patterns for the native protocol.
//! Captures are reset on entry to every repetition, and an unset backreference
//! fails. These C++ semantics differ from the general Rust/JavaScript backends.
use super::{
    MAX_ITEMS,
    trigger_vm::{Id, Kind, Node, Program},
};
use anyhow::{Context, Result, ensure};
use std::ops::Range;
const MAX_DEPTH: usize = 128;
const MAX_REPETITION: usize = 2000;
const PROGRAM_BYTES: usize = 1 << 20;

pub struct EcmaBytePattern {
    program: Program,
    max_bytes: usize,
    program_bytes: usize,
}
impl EcmaBytePattern {
    pub fn compile(pattern: &str, max_bytes: usize) -> Result<Self> {
        ensure!(
            max_bytes > 0 && pattern.len() <= max_bytes,
            "grammar trigger pattern exceeds max_constraint_bytes={max_bytes}; shorten it or configure a positive sufficient budget"
        );
        let mut parser = Parser {
            bytes: pattern.as_bytes(),
            position: 0,
            captures: 0,
            open: Vec::new(),
            emitted: 0,
            nodes: Vec::new(),
        };
        let root = parser.expression(0)?;
        ensure!(
            parser.position == pattern.len(),
            "grammar trigger pattern has an unmatched ')' at byte {}; balance its groups",
            parser.position
        );
        let program_bytes = parser.emitted;
        let program = Program {
            root,
            nodes: parser.nodes,
            captures: parser.captures,
            full: pattern.starts_with('^') && pattern.ends_with('$'),
        };
        Ok(Self {
            program,
            max_bytes,
            program_bytes,
        })
    }
    pub(crate) fn program_bytes(&self) -> usize {
        self.program_bytes
    }
    /// First nonempty captured group, falling back to the complete match.
    pub fn find(&self, bytes: &[u8]) -> Result<Option<usize>> {
        ensure!(
            bytes.len() <= self.max_bytes,
            "grammar trigger buffer exceeds max_constraint_bytes={}; shorten output or increase the explicit byte budget",
            self.max_bytes
        );
        self.program.find(bytes)
    }
}
struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
    captures: usize,
    open: Vec<usize>,
    emitted: usize,
    nodes: Vec<Node>,
}
enum ClassAtom {
    Scalar(u32),
    Set(Vec<(u32, u32)>),
}
impl ClassAtom {
    fn ranges(self) -> Vec<(u32, u32)> {
        match self {
            Self::Scalar(c) => vec![(c, c)],
            Self::Set(ranges) => ranges,
        }
    }
}

impl Parser<'_> {
    fn node(&mut self, kind: Kind, captures: Range<usize>) -> Result<Id> {
        let extra = match &kind {
            Kind::Seq(items) | Kind::Alt(items) => items.len() * std::mem::size_of::<Id>(),
            _ => 0,
        };
        self.emitted = self
            .emitted
            .checked_add(std::mem::size_of::<Node>() + extra)
            .context("grammar trigger translated program size overflow; simplify the pattern")?;
        ensure!(
            self.emitted <= PROGRAM_BYTES && self.nodes.len() < MAX_ITEMS,
            "grammar trigger translated program exceeds its byte/node budget; simplify nested groups or character classes"
        );
        let id = Id(self.nodes.len());
        self.nodes.push(Node { kind, captures });
        Ok(id)
    }
    fn set(&mut self, ranges: &[(u32, u32)], negate: bool) -> Result<Id> {
        let mut set = [if negate { u64::MAX } else { 0 }; 4];
        for &(lo, hi) in ranges {
            for byte in lo..=hi.min(255) {
                let bit = 1u64 << (byte % 64);
                if negate {
                    set[byte as usize / 64] &= !bit;
                } else {
                    set[byte as usize / 64] |= bit;
                }
            }
        }
        self.node(Kind::Set(set), 0..0)
    }
    fn sequence(&mut self, items: Vec<Id>, span: Range<usize>) -> Result<Id> {
        if items.len() == 1 {
            Ok(items[0])
        } else {
            self.node(Kind::Seq(items), span)
        }
    }
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }
    fn take(&mut self) -> Result<u8> {
        let byte = self.peek().with_context(|| {
            format!(
                "grammar trigger pattern ends at byte {}; complete its escape, group or class",
                self.position
            )
        })?;
        self.position += 1;
        Ok(byte)
    }
    fn expect(&mut self, byte: u8) -> Result<()> {
        ensure!(
            self.take()? == byte,
            "grammar trigger pattern expected {:?} at byte {}; repair its syntax",
            char::from(byte),
            self.position.saturating_sub(1)
        );
        Ok(())
    }
    fn number(&mut self) -> Result<usize> {
        let start = self.position;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.position += 1;
        }
        ensure!(
            self.position > start,
            "grammar trigger repetition needs a decimal bound at byte {start}; supply a valid quantifier"
        );
        std::str::from_utf8(&self.bytes[start..self.position])?
            .parse()
            .context("grammar trigger decimal bound overflows usize; use a smaller bound")
    }
    fn expression(&mut self, depth: usize) -> Result<Id> {
        ensure!(
            depth <= MAX_DEPTH,
            "grammar trigger groups exceed depth {MAX_DEPTH}; simplify nested groups"
        );
        let start = self.captures + 1;
        let mut branch_start = start;
        let mut branches = Vec::new();
        let mut sequence = Vec::new();
        while let Some(byte) = self.peek() {
            if byte == b')' {
                break;
            }
            if byte == b'|' {
                self.position += 1;
                branches.push(self.sequence(
                    std::mem::take(&mut sequence),
                    branch_start..self.captures + 1,
                )?);
                branch_start = self.captures + 1;
                continue;
            }
            let (mut atom, assertion) = match self.take()? {
                b'(' => {
                    let kind = if self.peek() == Some(b'?') {
                        self.position += 1;
                        match self.take()? {
                            b':' => 1,
                            b'=' => 2,
                            b'!' => 3,
                            _ => anyhow::bail!(
                                "grammar trigger group at byte {} is not ECMAScript capturing, noncapturing or lookahead syntax; remove backend-specific group extensions",
                                self.position.saturating_sub(3)
                            ),
                        }
                    } else {
                        0
                    };
                    let capture = if kind == 0 {
                        self.captures += 1;
                        ensure!(
                            self.captures <= 1024,
                            "grammar trigger exceeds 1024 capture groups; use fewer capturing groups"
                        );
                        self.open.push(self.captures);
                        Some(self.captures)
                    } else {
                        None
                    };
                    let body = self.expression(depth + 1)?;
                    self.expect(b')')?;
                    let node = if let Some(group) = capture {
                        self.open.pop();
                        self.node(Kind::Capture { group, body }, group..self.captures + 1)?
                    } else if kind == 1 {
                        body
                    } else {
                        let span = self.nodes[body.0].captures.clone();
                        self.node(
                            Kind::Look {
                                positive: kind == 2,
                                body,
                            },
                            span,
                        )?
                    };
                    (node, kind >= 2)
                }
                b'[' => (self.character_class()?, false),
                b'.' => (self.set(&[(10, 10), (13, 13)], true)?, false),
                b'^' => (self.node(Kind::Start, 0..0)?, true),
                b'$' => (self.node(Kind::End, 0..0)?, true),
                b'\\' => self.escape()?,
                b'*' | b'+' | b'?' | b'{' | b'}' | b']' => anyhow::bail!(
                    "grammar trigger pattern has an unescaped syntax character at byte {}; escape literals or attach one valid quantifier to an atom",
                    self.position - 1
                ),
                byte => (self.node(Kind::Byte(byte), 0..0)?, false),
            };
            let bounds = match self.peek() {
                Some(b'*') => {
                    self.position += 1;
                    Some((0, None))
                }
                Some(b'+') => {
                    self.position += 1;
                    Some((1, None))
                }
                Some(b'?') => {
                    self.position += 1;
                    Some((0, Some(1)))
                }
                Some(b'{') => {
                    self.position += 1;
                    let min = self.number()?;
                    let max = if self.peek() == Some(b',') {
                        self.position += 1;
                        if self.peek() == Some(b'}') {
                            None
                        } else {
                            Some(self.number()?)
                        }
                    } else {
                        Some(min)
                    };
                    self.expect(b'}')?;
                    ensure!(
                        max.is_none_or(|m| min <= m),
                        "grammar trigger repetition ends below its minimum; repair the quantifier"
                    );
                    ensure!(
                        min <= MAX_REPETITION && max.is_none_or(|m| m <= MAX_REPETITION),
                        "grammar trigger repetition exceeds {MAX_REPETITION}; use a smaller bound"
                    );
                    Some((min, max))
                }
                _ => None,
            };
            if let Some((min, max)) = bounds {
                ensure!(
                    !assertion,
                    "grammar trigger quantifier follows an assertion; attach it to a consuming atom"
                );
                let greedy = self.peek() != Some(b'?');
                if !greedy {
                    self.position += 1;
                }
                let span = self.nodes[atom.0].captures.clone();
                atom = self.node(
                    Kind::Repeat {
                        body: atom,
                        min,
                        max,
                        greedy,
                    },
                    span,
                )?;
            }
            sequence.push(atom);
        }
        branches.push(self.sequence(sequence, branch_start..self.captures + 1)?);
        if branches.len() == 1 {
            Ok(branches[0])
        } else {
            self.node(Kind::Alt(branches), start..self.captures + 1)
        }
    }
    fn escape(&mut self) -> Result<(Id, bool)> {
        if matches!(self.peek(), Some(b'b' | b'B')) {
            let b = self.take()?;
            return Ok((self.node(Kind::Boundary(b == b'b'), 0..0)?, true));
        }
        if self.peek().is_some_and(|c| matches!(c, b'1'..=b'9')) {
            let group = self.number()?;
            ensure!(
                group <= self.captures,
                "grammar trigger backreference \\{group} precedes its capture; reference a preceding capture group"
            );
            let kind = if self.open.contains(&group) {
                Kind::Fail
            } else {
                Kind::Backref(group)
            };
            return Ok((self.node(kind, 0..0)?, false));
        }
        let ranges = self.class_escape()?.ranges();
        Ok((self.set(&ranges, false)?, false))
    }
    fn hex(&mut self, digits: usize) -> Result<u32> {
        let mut value = 0;
        for _ in 0..digits {
            let byte = self.take()?;
            let digit = char::from(byte)
                .to_digit(16)
                .context("grammar trigger escape requires hexadecimal digits; repair the escape")?;
            value = (value << 4) | digit;
        }
        Ok(value)
    }
    fn class_escape(&mut self) -> Result<ClassAtom> {
        let byte = self.take()?;
        let ranges = match byte {
            b'd' => Some(vec![(48, 57)]),
            b'D' => Some(vec![(0, 47), (58, 255)]),
            b'w' => Some(vec![(48, 57), (65, 90), (95, 95), (97, 122)]),
            b'W' => Some(vec![(0, 47), (58, 64), (91, 94), (96, 96), (123, 255)]),
            b's' => Some(vec![(9, 13), (32, 32)]),
            b'S' => Some(vec![(0, 8), (14, 31), (33, 255)]),
            _ => None,
        };
        if let Some(ranges) = ranges {
            return Ok(ClassAtom::Set(ranges));
        }
        let scalar = match byte {
            b'0' => 0,
            b'b' => 8,
            b'f' => 12,
            b'n' => 10,
            b'r' => 13,
            b't' => 9,
            b'v' => 11,
            b'x' => self.hex(2)?,
            // std::regex<char> narrows Unicode escapes to one byte.
            b'u' => self.hex(4)? & 0xff,
            b'c' => {
                let letter = self.take()?;
                ensure!(
                    letter.is_ascii_alphabetic(),
                    "grammar trigger control escape requires an ASCII letter; repair \\c escape"
                );
                u32::from(letter % 32)
            }
            byte if !byte.is_ascii_alphanumeric() => u32::from(byte),
            _ => anyhow::bail!(
                "grammar trigger escape at byte {} is invalid ECMAScript syntax; use a declared class, control/hex escape or escaped punctuation",
                self.position.saturating_sub(2)
            ),
        };
        Ok(ClassAtom::Scalar(scalar))
    }
    fn class_atom(&mut self) -> Result<ClassAtom> {
        match self.take()? {
            b'\\' => self.class_escape(),
            b'[' if self.peek() == Some(b':') => {
                self.position += 1;
                let start = self.position;
                while self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                    self.position += 1;
                }
                let name = std::str::from_utf8(&self.bytes[start..self.position])?;
                self.expect(b':')?;
                self.expect(b']')?;
                let ranges = match name {
                    "alnum" => vec![(48, 57), (65, 90), (97, 122)],
                    "alpha" => vec![(65, 90), (97, 122)],
                    "blank" => vec![(9, 9), (32, 32)],
                    "cntrl" => vec![(0, 31), (127, 127)],
                    "digit" | "d" => vec![(48, 57)],
                    "graph" => vec![(33, 126)],
                    "lower" => vec![(97, 122)],
                    "print" => vec![(32, 126)],
                    "punct" => vec![(33, 47), (58, 64), (91, 96), (123, 126)],
                    "space" | "s" => vec![(9, 13), (32, 32)],
                    "upper" => vec![(65, 90)],
                    "xdigit" => vec![(48, 57), (65, 70), (97, 102)],
                    "w" => vec![(48, 57), (65, 90), (95, 95), (97, 122)],
                    _ => anyhow::bail!(
                        "grammar trigger has unknown C-locale character class {name:?}; use a declared ECMAScript/POSIX character class"
                    ),
                };
                Ok(ClassAtom::Set(ranges))
            }
            b'[' if matches!(self.peek(), Some(b'.' | b'=')) => anyhow::bail!(
                "grammar trigger collating/equivalence bracket forms are not implemented yet; use explicit byte character classes"
            ),
            byte => Ok(ClassAtom::Scalar(u32::from(byte))),
        }
    }
    fn character_class(&mut self) -> Result<Id> {
        let negate = self.peek() == Some(b'^');
        if negate {
            self.position += 1;
        }
        let mut ranges = Vec::new();
        while self.peek() != Some(b']') {
            let first = self.class_atom()?;
            if self.peek() == Some(b'-') && self.bytes.get(self.position + 1) != Some(&b']') {
                self.position += 1;
                let last = self.class_atom()?;
                let (ClassAtom::Scalar(lo), ClassAtom::Scalar(hi)) = (first, last) else {
                    anyhow::bail!(
                        "grammar trigger class range endpoints must be characters; move class escapes outside the range"
                    );
                };
                ensure!(
                    lo <= hi,
                    "grammar trigger character range ends below its start; reverse or repair the endpoints"
                );
                ranges.push((lo, hi));
            } else {
                ranges.extend(first.ranges());
            }
            ensure!(
                ranges.len() <= MAX_ITEMS,
                "grammar trigger class exceeds {MAX_ITEMS} items; simplify duplicate character ranges"
            );
        }
        self.expect(b']')?;
        self.set(&ranges, negate)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_patterns_match_exhaustive_upstream_oracle() -> Result<()> {
        let mut checked = 0;
        for row in include_str!("trigger_oracle.tsv")
            .lines()
            .filter(|row| !row.starts_with('#'))
        {
            let columns: Vec<_> = row.split('\t').collect();
            assert_eq!(columns.len(), 3);
            let expected = columns[2].parse::<isize>()?;
            let actual = EcmaBytePattern::compile(columns[0], 4096)?.find(columns[1].as_bytes())?;
            assert_eq!(
                actual,
                usize::try_from(expected).ok(),
                "pattern {:?}, input {:?}",
                columns[0],
                columns[1]
            );
            checked += 1;
        }
        assert_eq!(checked, 2904);
        Ok(())
    }
    #[test]
    fn byte_patterns_match_pinned_upstream_capture_oracle() -> Result<()> {
        for (pattern, input, expected) in [
            ("(a*)b", "free b", Some(5)),
            ("x(a)?(b)", "xxb", Some(2)),
            ("(a(b)?)+", "aba", Some(2)),
            // Captures not participating in the final repetition are cleared
            // by the pinned std::regex oracle; retained captures move replay.
            ("x(?:(a)|b)+", "xab", Some(0)),
            ("x(?:a|(b))*", "xba", Some(0)),
            ("x(?:(a)|b){2}", "xab", Some(0)),
            (r"x(?:(a)|b)+\1", "xaba", None),
            ("a(?=(b))", "ab", Some(1)),
            ("^.*?(<tool>)", "junk<tool>tail", Some(4)),
            ("(é)", "pré", Some(2)),
            (r"(ab)\1", "xxabab", Some(2)),
            ("^a|b$", "b", Some(0)),
            ("^a|b$", "xb", None),
            (".", "\r", None),
            (r"\b(é)", "é", None),
            (r"(a)?b\1", "b", None),
            (r"(a\1)", "a", None),
            ("[]", "a", None),
            ("[^]", "a", Some(0)),
            (r"\u00E9", "é", None),
            (r"\u1234", "x", None),
            (r"\u1234", "4", Some(0)),
            (r"\_", "_", Some(0)),
            (r"\ ", " ", Some(0)),
            (r"(?=(a))\1", "a", Some(0)),
            (r"(?!(a))b\1", "b", None),
        ] {
            assert_eq!(
                EcmaBytePattern::compile(pattern, 4096)?.find(input.as_bytes())?,
                expected,
                "{pattern:?} on {input:?}"
            );
        }
        Ok(())
    }
    #[test]
    fn byte_patterns_reject_upstream_invalid_and_backend_extension_syntax() {
        for pattern in [
            r"\1(a)",
            "a{",
            "a**",
            "(?<=a)b",
            "(?<name>a)",
            "(?i:a)",
            "]",
            "}",
            r"[\1]",
            r"\a",
            "[z-a]",
            "(?=a)?a",
            r"\b?a",
            r"[0-\d]",
        ] {
            assert!(
                EcmaBytePattern::compile(pattern, 4096).is_err(),
                "{pattern}"
            );
        }
    }
    #[test]
    fn byte_patterns_preserve_nonunicode_classes_and_full_match_backtracking() -> Result<()> {
        assert_eq!(EcmaBytePattern::compile(r"\s", 64)?.find(&[0xa0])?, None);
        assert_eq!(EcmaBytePattern::compile(r"\S", 64)?.find(&[0xa0])?, Some(0));
        assert_eq!(EcmaBytePattern::compile("[é]", 64)?.find(&[0xc3])?, Some(0));
        assert_eq!(
            EcmaBytePattern::compile("^a|ab$", 64)?.find(b"ab")?,
            Some(0)
        );
        assert_eq!(
            EcmaBytePattern::compile(r"[\x00-\x02]", 64)?.find(&[1])?,
            Some(0)
        );
        Ok(())
    }
    #[test]
    fn byte_patterns_named_classes_keep_the_upstream_c_locale() -> Result<()> {
        assert_eq!(
            EcmaBytePattern::compile("[[:alpha:]]", 64)?.find("é".as_bytes())?,
            None
        );
        assert_eq!(
            EcmaBytePattern::compile("[[:alpha:]_]+", 64)?.find(b"2_Ab9")?,
            Some(1)
        );
        assert_eq!(
            EcmaBytePattern::compile("[[:xdigit:]]+", 64)?.find(b"zF0a")?,
            Some(1)
        );
        assert_eq!(
            EcmaBytePattern::compile("[^[:space:]]", 64)?.find(b"\r\nx")?,
            Some(2)
        );
        assert!(EcmaBytePattern::compile("[[:unknown:]]", 64).is_err());
        Ok(())
    }
    #[test]
    fn byte_patterns_bound_translated_program_depth_and_captures() {
        let expansion = EcmaBytePattern::compile(&r"\W".repeat(32768), 65536)
            .err()
            .unwrap();
        assert!(
            expansion.to_string().contains("translated program"),
            "{expansion:#}"
        );
        let nested = format!("{}a{}", "(".repeat(129), ")".repeat(129));
        assert!(
            EcmaBytePattern::compile(&nested, 4096)
                .err()
                .unwrap()
                .to_string()
                .contains("depth")
        );
        assert!(EcmaBytePattern::compile("a{2001}", 4096).is_err());
        assert!(
            EcmaBytePattern::compile(&"(a)".repeat(1025), 4096)
                .err()
                .unwrap()
                .to_string()
                .contains("capture groups")
        );
    }
    #[test]
    fn byte_patterns_bound_class_items_and_name_unimplemented_bracket_forms() {
        let class = format!("[{}]", "a".repeat(MAX_ITEMS + 1));
        assert!(
            EcmaBytePattern::compile(&class, class.len())
                .err()
                .unwrap()
                .to_string()
                .contains("class exceeds")
        );
        for pattern in ["[[.a.]]", "[[=a=]]"] {
            assert!(
                EcmaBytePattern::compile(pattern, 64)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("collating/equivalence")
            );
        }
    }
    #[test]
    fn byte_patterns_enforce_input_and_backtracking_budgets() -> Result<()> {
        assert!(
            EcmaBytePattern::compile("a", 1)?
                .find(b"aa")
                .unwrap_err()
                .to_string()
                .contains("max_constraint_bytes=1")
        );
        let regex = EcmaBytePattern::compile(r"^(a+)+\1$", 4096)?;
        let mut bytes = vec![b'a'; 512];
        bytes.push(b'b');
        assert!(regex.find(&bytes).is_err());
        Ok(())
    }
}
