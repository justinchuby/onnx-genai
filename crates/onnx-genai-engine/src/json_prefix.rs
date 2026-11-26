//! JSON string spelling normalization, including unfinished UTF-16 escapes.
use crate::byte_grammar::{GrammarByteCursor, GrammarCandidate};
use anyhow::Context;
use std::ops::RangeInclusive;

#[derive(Debug, thiserror::Error)]
#[error(
    "invalid JSON string escape at byte {offset}: {reason}; supply a valid JSON escape and pair UTF-16 surrogates"
)]
pub(crate) struct InvalidJsonEscape {
    offset: usize,
    reason: &'static str,
}

/// The committed backend spelling excludes an unfinished escape. Its scalar
/// domain is checked against the same grammar before accepting another token.
pub(crate) struct JsonPrefix {
    pub(crate) canonical: String,
    pending: Option<PendingEscape>,
}

struct PendingEscape {
    ranges: Vec<RangeInclusive<u32>>,
    spelling: String,
}

fn hex_range(bytes: &[u8], offset: usize) -> Result<RangeInclusive<u32>, InvalidJsonEscape> {
    let mut value = 0u32;
    for &byte in bytes {
        let digit = char::from(byte).to_digit(16).ok_or(InvalidJsonEscape {
            offset,
            reason: "Unicode escapes require four ASCII hexadecimal digits",
        })?;
        value = value * 16 + digit;
    }
    let shift = 4 * (4 - bytes.len());
    Ok((value << shift)..=(((value + 1) << shift) - 1))
}

fn scalar_ranges(units: RangeInclusive<u32>) -> Vec<RangeInclusive<u32>> {
    let mut ranges = Vec::new();
    for valid in [0..=0xd7ff, 0xe000..=0xffff] {
        let lo = (*units.start()).max(*valid.start());
        let hi = (*units.end()).min(*valid.end());
        if lo <= hi {
            ranges.push(lo..=hi);
        }
    }
    let lo = (*units.start()).max(0xd800);
    let hi = (*units.end()).min(0xdbff);
    if lo <= hi {
        ranges.push((0x10000 + ((lo - 0xd800) << 10))..=(0x103ff + ((hi - 0xd800) << 10)));
    }
    ranges
}

impl JsonPrefix {
    pub(crate) fn has_pending_escape(&self) -> bool {
        self.pending.is_some()
    }

    /// Reject an impossible escape continuation without rescanning unrelated
    /// committed content. The real projection still checks every admitted piece.
    pub(crate) fn pending_piece_syntax_viable(&self, piece: &[u8]) -> bool {
        let Some(pending) = &self.pending else {
            return true;
        };
        let text = match std::str::from_utf8(piece) {
            Ok(text) => text,
            Err(error) if error.error_len().is_none() && error.valid_up_to() > 0 => {
                let Ok(text) = std::str::from_utf8(&piece[..error.valid_up_to()]) else {
                    return false;
                };
                text
            }
            Err(_) => return false,
        };
        let mut probe = String::with_capacity(1 + pending.spelling.len() + text.len());
        probe.push('"');
        probe.push_str(&pending.spelling);
        probe.push_str(text);
        Self::parse(&probe).is_ok_and(|prefix| {
            prefix
                .pending
                .is_none_or(|escape| !escape.ranges.is_empty())
        })
    }
    pub(crate) fn parse(text: &str) -> anyhow::Result<Self> {
        let bytes = text.as_bytes();
        let mut at = 0;
        let mut in_string = false;
        let mut pending = None;
        let mut canonical = String::with_capacity(text.len());
        let mut segment_start = 0;
        while at < bytes.len() {
            match bytes[at] {
                b'"' => in_string = !in_string,
                b'\\' if in_string => {
                    let start = at;
                    let rest = &bytes[at + 1..];
                    if rest.is_empty() {
                        pending = Some((start, std::iter::once(0..=0x10ffff).collect()));
                        break;
                    }
                    if rest[0] != b'u' {
                        if !b"\"\\/bfnrt".contains(&rest[0]) {
                            return Err(InvalidJsonEscape {
                                offset: start,
                                reason: "unrecognized backslash escape",
                            }
                            .into());
                        }
                        at += 2;
                        if rest[0] == b'/' {
                            canonical.push_str(&text[segment_start..start]);
                            canonical.push('/');
                            segment_start = at;
                        }
                        continue;
                    }
                    let digits = &rest[1..rest.len().min(5)];
                    let units = hex_range(digits, start)?;
                    if digits.len() < 4 {
                        pending = Some((start, scalar_ranges(units)));
                        break;
                    }
                    let unit = *units.start();
                    if (0xd800..=0xdbff).contains(&unit) {
                        let low = &bytes[start + 6..];
                        for (index, &expected) in b"\\u".iter().enumerate() {
                            if let Some(&actual) = low.get(index) {
                                if actual != expected {
                                    return Err(InvalidJsonEscape {
                                        offset: start + 6 + index,
                                        reason: "a high surrogate must be followed by a low-surrogate Unicode escape",
                                    }.into());
                                }
                            } else {
                                pending = Some((start, scalar_ranges(unit..=unit)));
                                break;
                            }
                        }
                        if pending.is_some() {
                            break;
                        }
                        let digits = &low[2..low.len().min(6)];
                        let low_units = hex_range(digits, start + 6)?;
                        let lo = (*low_units.start()).max(0xdc00);
                        let hi = (*low_units.end()).min(0xdfff);
                        if lo > hi {
                            return Err(InvalidJsonEscape {
                                offset: start + 6,
                                reason: "the second UTF-16 code unit is not a low surrogate",
                            }
                            .into());
                        }
                        if digits.len() < 4 {
                            let base = 0x10000 + ((unit - 0xd800) << 10);
                            pending = Some((
                                start,
                                std::iter::once((base + lo - 0xdc00)..=(base + hi - 0xdc00))
                                    .collect(),
                            ));
                            break;
                        }
                        canonical.push_str(&text[segment_start..start]);
                        canonical.push_str(&scalar_spelling(
                            0x10000 + ((unit - 0xd800) << 10) + lo - 0xdc00,
                        )?);
                        at += 12;
                        segment_start = at;
                        continue;
                    }
                    if (0xdc00..=0xdfff).contains(&unit) {
                        return Err(InvalidJsonEscape {
                            offset: start,
                            reason: "a low surrogate has no preceding high surrogate",
                        }
                        .into());
                    }
                    canonical.push_str(&text[segment_start..start]);
                    canonical.push_str(&scalar_spelling(unit)?);
                    at += 6;
                    segment_start = at;
                    continue;
                }
                0x7f if in_string => {
                    canonical.push_str(&text[segment_start..at]);
                    canonical.push_str("\\u007f");
                    segment_start = at + 1;
                }
                _ => {}
            }
            at += 1;
        }
        let stable_end = pending.as_ref().map_or(text.len(), |(at, _)| *at);
        canonical.push_str(&text[segment_start..stable_end]);
        let leading = canonical.len() - canonical.trim_start_matches([' ', '\t', '\r', '\n']).len();
        canonical.drain(..leading);
        // Only complete JSON lexemes can be followed by ignorable whitespace.
        // Do not turn an impossible prefix such as `1e ` or `tru ` into a
        // viable unfinished lexeme by removing its terminating whitespace.
        let trimmed = canonical.trim_end_matches([' ', '\t', '\r', '\n']);
        let complete_lexeme = matches!(
            trimmed.as_bytes().last(),
            Some(b'"' | b']' | b'}' | b'0'..=b'9')
        ) || ["true", "false", "null"]
            .iter()
            .any(|literal| trimmed.ends_with(literal));
        if !in_string && complete_lexeme {
            canonical.truncate(trimmed.len());
        }
        Ok(Self {
            canonical,
            pending: pending.map(|(_, ranges)| PendingEscape {
                ranges,
                spelling: text[stable_end..].into(),
            }),
        })
    }

    /// The token-trie backend consumes a literal existing prefill. Completed
    /// escapes use the same normalization as projected byte recognition.
    pub(crate) fn into_lexical(mut self) -> String {
        if let Some(pending) = self.pending {
            self.canonical.push_str(&pending.spelling);
        }
        self.canonical
    }

    pub(crate) fn preview(
        &self,
        cursor: &mut GrammarByteCursor,
        committed: &str,
        pending_utf8: &[u8],
    ) -> anyhow::Result<GrammarCandidate> {
        if let Some(pending) = &self.pending {
            // UTF-8 bytes cannot continue an ASCII JSON escape.
            if !pending_utf8.is_empty() {
                return Ok(GrammarCandidate::Rejected {
                    byte_offset: self.canonical.len(),
                });
            }
            let mut trial = cursor.empty_fork()?;
            if matches!(
                trial.preview(self.canonical.as_bytes())?,
                GrammarCandidate::Rejected { .. }
            ) {
                return Ok(GrammarCandidate::Rejected {
                    byte_offset: self.canonical.len(),
                });
            }
            trial.commit(self.canonical.as_bytes())?;
            for range in &pending.ranges {
                // Keep each UTF-8 width and the escaped ASCII region separate.
                // Binary subdivision prunes a whole scalar interval as soon as
                // its common byte prefix cannot continue the grammar.
                for domain in [
                    0..=31,
                    32..=126,
                    127..=127,
                    128..=2047,
                    2048..=0xd7ff,
                    0xe000..=0xffff,
                    0x10000..=0x10ffff,
                ] {
                    let lo = (*range.start()).max(*domain.start());
                    let hi = (*range.end()).min(*domain.end());
                    if lo <= hi && viable_scalar(&mut trial, lo, hi)? {
                        return Ok(GrammarCandidate::Viable { complete: false });
                    }
                }
            }
            return Ok(GrammarCandidate::Rejected {
                byte_offset: self.canonical.len(),
            });
        }
        let bytes = [self.canonical.as_bytes(), pending_utf8].concat();
        if let Some(suffix) = bytes.strip_prefix(committed.as_bytes()) {
            cursor.preview(suffix)
        } else {
            cursor.preview_root(&bytes)
        }
    }
}

fn scalar_spelling(scalar: u32) -> anyhow::Result<String> {
    let character =
        char::from_u32(scalar).context("JSON escape search lost its Unicode scalar range")?;
    let quoted = serde_json::to_string(&character.to_string())?;
    if character == '\u{7f}' {
        Ok("\\u007f".into())
    } else {
        Ok(quoted[1..quoted.len() - 1].into())
    }
}

fn viable_bytes(cursor: &mut GrammarByteCursor, bytes: &[u8]) -> anyhow::Result<bool> {
    match cursor.preview(bytes) {
        Ok(status) => Ok(matches!(status, GrammarCandidate::Viable { .. })),
        Err(error) if error.is::<crate::byte_grammar::GrammarByteBudgetExceeded>() => Ok(false),
        Err(error) => Err(error),
    }
}

fn viable_scalar(cursor: &mut GrammarByteCursor, lo: u32, hi: u32) -> anyhow::Result<bool> {
    let left = scalar_spelling(lo)?;
    let right = scalar_spelling(hi)?;
    let common = left
        .bytes()
        .zip(right.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    if (lo >= 128 || lo == hi) && common > 0 && !viable_bytes(cursor, &left.as_bytes()[..common])? {
        return Ok(false);
    }
    if lo == hi {
        return viable_bytes(cursor, left.as_bytes());
    }
    let middle = lo + (hi - lo) / 2;
    Ok(viable_scalar(cursor, lo, middle)? || viable_scalar(cursor, middle + 1, hi)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn accepts(schema: &serde_json::Value, text: &str) -> bool {
        let mut cursor = GrammarByteCursor::json_schema(schema, 1024).unwrap();
        let prefix = match JsonPrefix::parse(text) {
            Ok(prefix) => prefix,
            Err(error) if error.is::<InvalidJsonEscape>() => return false,
            Err(error) => panic!("{error:#}"),
        };
        matches!(
            prefix.preview(&mut cursor, "", b"").unwrap(),
            GrammarCandidate::Viable { .. }
        )
    }

    #[test]
    fn json_prefix_trailing_whitespace_preserves_string_values_and_rejects_second_values() {
        let schema = json!({"const":"A "});
        for text in ["\"A \"", " \t\"A \" \r\n", "\"A "] {
            assert!(accepts(&schema, text), "{text:?}");
        }
        assert!(!accepts(&schema, "\"A\" "));
        assert!(!accepts(&schema, "\"A \" \"second\""));
        assert_eq!(JsonPrefix::parse("\"A ").unwrap().canonical, "\"A ");
        assert_eq!(JsonPrefix::parse("{\"a\": ").unwrap().canonical, "{\"a\": ");
        assert!(!accepts(&json!({"const":1}), "1 2 "));
        assert!(accepts(&json!({"const":1}), "1 \n"));
        for invalid in ["1e ", "1. ", "1e- ", "tru ", "nul ", "fals "] {
            assert!(!accepts(&json!({}), invalid), "{invalid:?}");
        }
    }

    #[test]
    fn json_prefix_const_accepts_every_escape_split_and_preserves_rejection() {
        for (value, escaped) in [
            ("A", r#""\u0041""#),
            ("終", r#""\u7D42""#),
            ("😀", r#""\uD83D\uDE00""#),
            ("\\", r#""\u005c""#),
        ] {
            let schema = json!({"const":value});
            for end in 0..=escaped.len() {
                assert!(
                    accepts(&schema, &escaped[..end]),
                    "{value:?}: {:?}",
                    &escaped[..end]
                );
            }
        }
        for invalid in [
            r#""\u0042"#,
            r#""\u7e"#,
            r#""\ud83d\udf"#,
            r#""\udc"#,
            r#""\u00z"#,
            r#""\q"#,
            r#""\ud83dx"#,
        ] {
            assert!(!accepts(&json!({"const":"A"}), invalid), "{invalid:?}");
        }
    }

    #[test]
    fn json_prefix_known_keys_and_string_lengths_use_decoded_scalars() {
        let schema = json!({"type":"object", "properties":{"終":{"type":"string", "minLength":1, "maxLength":1}}, "required":["終"], "additionalProperties":false});
        let escaped = r#"{"\u7d42":"\ud83d\ude00"}"#;
        for end in 0..=escaped.len() {
            assert!(accepts(&schema, &escaped[..end]), "{:?}", &escaped[..end]);
        }
        assert!(!accepts(&schema, r#"{"\u7d43"#));
        assert!(!accepts(&schema, r#"{"終":"\ud83d\ude00A"#));
    }

    #[test]
    fn json_prefix_partial_escape_never_permits_eof_or_invalid_utf8_continuation() {
        let mut cursor = GrammarByteCursor::json_schema(&json!({"const":"A"}), 1024).unwrap();
        let prefix = JsonPrefix::parse(r#""\u004"#).unwrap();
        assert_eq!(
            prefix.preview(&mut cursor, "", b"").unwrap(),
            GrammarCandidate::Viable { complete: false }
        );
        assert!(matches!(
            prefix.preview(&mut cursor, "", &[0xc3]).unwrap(),
            GrammarCandidate::Rejected { .. }
        ));
        assert_eq!(cursor.committed_bytes(), 0);
        assert!(matches!(
            JsonPrefix::parse(r#""A""#)
                .unwrap()
                .preview(&mut cursor, "", b"")
                .unwrap(),
            GrammarCandidate::Viable { complete: true }
        ));
    }

    #[test]
    fn json_property_escapes_match_declared_key_identity_and_value_constraints() {
        let schema = json!({"type":"object","properties":{"\u{7f}":{"const":1}},"required":["\u{7f}"],"additionalProperties":false});
        for spelling in [r#"{"\u007f":1}"#, r#"{"\u007F":1}"#] {
            for end in 0..=spelling.len() {
                assert!(
                    accepts(&schema, &spelling[..end]),
                    "schema={schema}; prefix={:?}",
                    &spelling[..end]
                );
            }
        }
        assert!(!accepts(&schema, r#"{"\u007f":2}"#));
        let mixed = json!({"type":"object","properties":{"\u{7f}":{"const":1}},"additionalProperties":{"const":2}});
        assert!(accepts(&mixed, r#"{"\u007f":1}"#));
        assert!(
            !accepts(&mixed, r#"{"\u007f":2}"#),
            "an escaped known key must not bypass its schema through additionalProperties"
        );
        assert!(accepts(&mixed, r#"{"other":2}"#));
    }

    #[test]
    fn json_property_quoting_keeps_known_pattern_and_additional_ownership_aligned() {
        for key in [
            "\0",
            "\n",
            "\u{1f}",
            "\u{7f}",
            "\"",
            "\\",
            "/",
            "é",
            "😀",
            "\u{7f}x\u{7f}",
            r"\u007f",
        ] {
            let escaped = key
                .encode_utf16()
                .map(|unit| format!("\\u{unit:04X}"))
                .collect::<String>();
            for name in [
                serde_json::to_string(key).unwrap(),
                format!("\"{escaped}\""),
            ] {
                let correct = format!("{{{name}:1}}");
                let incorrect = format!("{{{name}:2}}");
                for schema in [
                    json!({"type":"object","properties":{key:{"const":1}},"additionalProperties":{"const":2}}),
                    json!({"type":"object","properties":{key:{"const":1}},"patternProperties":{".*":{"enum":[1,2]}}}),
                ] {
                    for end in correct
                        .char_indices()
                        .map(|(at, _)| at)
                        .chain(std::iter::once(correct.len()))
                    {
                        assert!(
                            accepts(&schema, &correct[..end]),
                            "key={key:?}; prefix={:?}",
                            &correct[..end]
                        );
                    }
                    assert!(
                        !accepts(&schema, &incorrect),
                        "escaped key must obey declared value: key={key:?}, {schema}"
                    );
                    assert!(accepts(&schema, r#"{"other":2}"#));
                }
                let unsatisfiable = json!({"type":"object","properties":{key:false},"additionalProperties":{"const":2}});
                assert!(
                    !accepts(&unsatisfiable, &incorrect),
                    "an impossible optional key must remain excluded: {key:?}"
                );
                assert!(accepts(&unsatisfiable, r#"{"other":2}"#));
                let union = json!({"anyOf":[
                    {"type":"object","properties":{key:{"const":1}},"additionalProperties":false},
                    {"type":"object","additionalProperties":{"const":2}}
                ]});
                assert!(accepts(&union, &correct));
                assert!(accepts(&union, &incorrect));
            }
        }
        let free = json!({"type":"object","additionalProperties":{"type":"string"}});
        assert!(accepts(&free, "{\"\u{7f}\":\"\u{7f}\"}"));
        assert!(
            !accepts(&free, "{\"x\":\"\u{1f}\"}"),
            "raw RFC 8259 control bytes remain invalid"
        );
    }

    #[test]
    fn json_prefix_control_escapes_preserve_free_and_literal_property_names() {
        for schema in [
            json!({"type":"object","additionalProperties":{"type":"string"}}),
            json!({"type":"object"}),
        ] {
            let spelling = r#"{"\u007f":"\u007f"}"#;
            for end in 0..=spelling.len() {
                assert!(accepts(&schema, &spelling[..end]), "{:?}", &spelling[..end]);
            }
        }
        // Known names are literal backend lexemes. Keep previously supported
        // raw DEL keys intact; classifying every string as a value breaks them.
        let known = json!({"type":"object","properties":{"\u{7f}":{"const":"A"}},"required":["\u{7f}"],"additionalProperties":false});
        assert!(accepts(&known, "{\"\u{7f}\":\"A\"}"));
        for character in ['\0', '\n', '\u{1f}', '\u{7f}'] {
            let schema = json!({"const":character.to_string()});
            let spelling = format!("\"\\u{:04x}\"", character as u32);
            for end in 0..=spelling.len() {
                assert!(accepts(&schema, &spelling[..end]), "{:?}", &spelling[..end]);
            }
        }
    }

    #[test]
    fn json_prefix_search_prunes_utf8_ranges_and_keeps_ascii_escape_alternatives() {
        let schema = json!({"type":"string", "pattern":"^[A-Z]$"});
        assert!(accepts(&schema, r#""\u00"#));
        assert!(!accepts(&schema, r#""\u01"#));
        // A longer escaped scalar exceeds this cursor's budget, but a printable
        // ASCII completion still fits. One rejected branch must not abort search.
        let mut cursor = GrammarByteCursor::json_schema(&json!({"const":"A"}), 4).unwrap();
        assert_eq!(
            JsonPrefix::parse(r#""\"#)
                .unwrap()
                .preview(&mut cursor, "", b"")
                .unwrap(),
            GrammarCandidate::Viable { complete: false }
        );
        let quoted = json!({"const":"\""});
        assert!(accepts(&quoted, r#""\u002"#));
        assert!(accepts(&quoted, r#""\u0022""#));
    }
}
