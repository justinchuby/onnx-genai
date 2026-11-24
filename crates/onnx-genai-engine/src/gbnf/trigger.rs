//! Bounded byte-oriented ECMAScript trigger patterns for the native protocol.
//!
//! Encode each input byte as a Latin-1 scalar, so the regex backend's Unicode
//! storage cannot turn a model byte offset into a decoded-text offset. Translate
//! the accepted ECMAScript syntax explicitly; never expose backend extensions.
use super::MAX_ITEMS;
use anyhow::{Context, Result, ensure};
use fancy_regex::{Regex, RegexBuilder};

const MAX_DEPTH: usize = 128;
const MAX_REPETITION: usize = 2000;
const BACKTRACK_LIMIT: usize = 1_000_000;
const PROGRAM_BYTES: usize = 1 << 20;

pub struct EcmaBytePattern {
    regex: Regex,
    max_bytes: usize,
}
impl EcmaBytePattern {
    pub fn compile(pattern: &str, max_bytes: usize) -> Result<Self> {
        ensure!(
            max_bytes > 0 && pattern.len() <= max_bytes,
            "grammar trigger pattern exceeds max_constraint_bytes={max_bytes}; shorten it or configure a positive sufficient budget"
        );
        let mut parser = Translator {
            bytes: pattern.as_bytes(),
            position: 0,
            captures: 0,
            open: Vec::new(),
            emitted: 0,
        };
        let mut translated = parser.expression(0)?;
        ensure!(
            parser.position == pattern.len(),
            "grammar trigger pattern has an unmatched ')' at byte {}; balance its groups",
            parser.position
        );
        // The upstream uses regex_match, rather than search, only with both
        // anchors. Wrapping is essential: post-filtering a shorter first match
        // would lose a later alternative that satisfies the complete match.
        if pattern.starts_with('^') && pattern.ends_with('$') {
            translated = format!(r"\A(?:{translated})\z");
        }
        let regex = RegexBuilder::new(&translated)
            .backtrack_limit(BACKTRACK_LIMIT)
            .delegate_size_limit(1 << 20)
            .delegate_dfa_size_limit(1 << 20)
            .build().context("compile grammar trigger ECMAScript pattern; repair its syntax or simplify the expression")?;
        Ok(Self { regex, max_bytes })
    }

    /// First nonempty captured group's byte offset, falling back to the whole
    /// match. A capture in a lookahead may start after the whole match ends.
    pub fn find(&self, bytes: &[u8]) -> Result<Option<usize>> {
        ensure!(
            bytes.len() <= self.max_bytes,
            "grammar trigger buffer exceeds max_constraint_bytes={}; shorten output or increase the explicit byte budget",
            self.max_bytes
        );
        let input = bytes.iter().copied().map(char::from).collect::<String>();
        let Some(captures) = self.regex.captures(&input).context("match grammar trigger pattern within its backtracking budget; simplify the pattern or use a literal/token trigger")? else { return Ok(None); };
        let whole = captures.get(0).context("grammar trigger backend omitted its whole match; restart with a fresh compiled pattern")?;
        let offset = captures
            .iter()
            .skip(1)
            .flatten()
            .find(|m| m.start() != m.end())
            .map_or(whole.start(), |m| m.start());
        let prefix = input.get(..offset).context("grammar trigger backend returned a non-scalar Latin-1 offset; use a pattern over complete byte events")?;
        Ok(Some(prefix.chars().count()))
    }
}

struct Translator<'a> {
    bytes: &'a [u8],
    position: usize,
    captures: usize,
    open: Vec<usize>,
    emitted: usize,
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

fn class_text(ranges: &[(u32, u32)], negate: bool) -> String {
    let mut members = [false; 256];
    for &(lo, hi) in ranges {
        for byte in lo..=hi.min(255) {
            members[byte as usize] = true;
        }
    }
    let mut text = String::from("[");
    for (byte, &present) in members.iter().enumerate() {
        if present != negate {
            text.push_str(&format!(r"\u{{{byte:x}}}"));
        }
    }
    if text == "[" {
        return "(?!)".into();
    }
    text.push(']');
    text
}
fn literal(value: u32) -> String {
    if value > 255 {
        "(?!)".into()
    } else {
        format!(r"\u{{{value:x}}}")
    }
}
impl Translator<'_> {
    fn reserve(&mut self, bytes: usize) -> Result<()> {
        self.emitted = self
            .emitted
            .checked_add(bytes)
            .context("grammar trigger translated program size overflow; simplify the pattern")?;
        ensure!(
            self.emitted <= PROGRAM_BYTES,
            "grammar trigger translated program exceeds {PROGRAM_BYTES} bytes; simplify nested groups or character classes"
        );
        Ok(())
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
    fn expression(&mut self, depth: usize) -> Result<String> {
        ensure!(
            depth <= MAX_DEPTH,
            "grammar trigger groups exceed depth {MAX_DEPTH}; simplify nested groups"
        );
        let mut output = String::new();
        while let Some(byte) = self.peek() {
            if byte == b')' {
                break;
            }
            if byte == b'|' {
                self.position += 1;
                self.reserve(1)?;
                output.push('|');
                continue;
            }
            let (mut atom, assertion) = match self.take()? {
                b'(' => {
                    let kind = if self.peek() == Some(b'?') {
                        self.position += 1;
                        match self.take()? {
                            b':' => "?:",
                            b'=' => "?=",
                            b'!' => "?!",
                            _ => anyhow::bail!(
                                "grammar trigger group at byte {} is not ECMAScript capturing, noncapturing or lookahead syntax; remove backend-specific group extensions",
                                self.position.saturating_sub(3)
                            ),
                        }
                    } else {
                        ""
                    };
                    let capture = if kind.is_empty() {
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
                    if capture.is_some() {
                        self.open.pop();
                    }
                    (format!("({kind}{body})"), kind == "?=" || kind == "?!")
                }
                b'[' => (self.character_class()?, false),
                b'.' => (r"[^\r\n]".into(), false),
                b'^' => ("^".into(), true),
                b'$' => ("$".into(), true),
                b'\\' => self.escape()?,
                b'*' | b'+' | b'?' | b'{' | b'}' | b']' => anyhow::bail!(
                    "grammar trigger pattern has an unescaped syntax character at byte {}; escape literals or attach one valid quantifier to an atom",
                    self.position - 1
                ),
                byte => (literal(u32::from(byte)), false),
            };
            let quantifier = match self.peek() {
                Some(b'*' | b'+' | b'?') => Some(char::from(self.take()?).to_string()),
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
                        max.is_none_or(|max| min <= max),
                        "grammar trigger repetition ends below its minimum; repair the quantifier"
                    );
                    ensure!(
                        min <= MAX_REPETITION && max.is_none_or(|max| max <= MAX_REPETITION),
                        "grammar trigger repetition exceeds {MAX_REPETITION}; use a smaller bound"
                    );
                    Some(match max {
                        Some(max) if max == min => format!("{{{min}}}"),
                        Some(max) => format!("{{{min},{max}}}"),
                        None => format!("{{{min},}}"),
                    })
                }
                _ => None,
            };
            if let Some(quantifier) = quantifier {
                ensure!(
                    !assertion,
                    "grammar trigger quantifier follows an assertion; attach it to a consuming atom"
                );
                atom = format!("(?:{atom}){quantifier}");
                if self.peek() == Some(b'?') {
                    self.position += 1;
                    atom.push('?');
                }
            }
            self.reserve(atom.len())?;
            output.push_str(&atom);
        }
        Ok(output)
    }
    fn escape(&mut self) -> Result<(String, bool)> {
        if matches!(self.peek(), Some(b'b' | b'B')) {
            let b = self.take()?;
            let word = "[A-Za-z0-9_]";
            let boundary = if b == b'b' {
                format!("(?:(?<={word})(?!{word})|(?<!{word})(?={word}))")
            } else {
                format!("(?:(?<={word})(?={word})|(?<!{word})(?!{word}))")
            };
            return Ok((boundary, true));
        }
        if self.peek().is_some_and(|c| matches!(c, b'1'..=b'9')) {
            let group = self.number()?;
            ensure!(
                group <= self.captures,
                "grammar trigger backreference \\{group} precedes its capture; reference a preceding capture group"
            );
            return Ok((
                if self.open.contains(&group) {
                    "(?!)".into()
                } else {
                    format!(r"\{group}")
                },
                false,
            ));
        }
        Ok((class_text(&self.class_escape()?.ranges(), false), false))
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
    fn character_class(&mut self) -> Result<String> {
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
        Ok(class_text(&ranges, negate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_patterns_match_pinned_upstream_capture_oracle() -> Result<()> {
        for (pattern, input, expected) in [
            ("(a*)b", "free b", Some(5)),
            ("x(a)?(b)", "xxb", Some(2)),
            ("(a(b)?)+", "aba", Some(2)),
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
        let expansion = EcmaBytePattern::compile(&r"\W".repeat(4096), 16384)
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
