//! A generation-only XML spelling: exactly one invoke, double-quoted names,
//! XML-escaped JSON values. The inbound parser retains its broader raw-text API.
use super::{MAX_TOOL_NAME_BYTES, MAX_TOOL_PAYLOAD_BYTES, ToolParseOutcome, parse_atem_xml};
use std::collections::BTreeSet;

const INVOKE: &str = "<atem:invoke name=\"";
const PARAMETER: &str = "<atem:parameter name=\"";
const END_PARAMETER: &str = "</atem:parameter>";
const END_INVOKE: &str = "</atem:invoke>";
const ENTITIES: [(&str, char); 5] = [
    ("&amp;", '&'),
    ("&quot;", '"'),
    ("&apos;", '\''),
    ("&lt;", '<'),
    ("&gt;", '>'),
];

/// Logical prefixes checked against one already-compiled tool grammar. A partial
/// XML delimiter/entity can describe several future logical prefixes; none is
/// selected or committed until the actual bytes disambiguate it.
#[derive(Debug)]
pub struct CanonicalXmlPrefix {
    pub alternatives: Vec<String>,
    pub complete: bool,
    pub permits_pending_utf8: bool,
}

fn result(
    alternatives: Vec<String>,
    complete: bool,
    permits_pending_utf8: bool,
) -> Option<CanonicalXmlPrefix> {
    (!alternatives.is_empty()).then_some(CanonicalXmlPrefix {
        alternatives,
        complete,
        permits_pending_utf8,
    })
}
fn string_prefix(text: &str) -> Option<String> {
    let mut encoded = serde_json::to_string(text).ok()?;
    encoded.pop();
    Some(encoded)
}

// Complete entities are decoded exactly once, as in the authoritative parser.
// Unknown/literal entities are outside this generation spelling. A partial
// entity eagerly represents each possible decoded character, so schema-dead
// branches are rejected before its remaining fixed bytes are generated.
fn decode(text: &str) -> Option<(Vec<String>, bool)> {
    let mut decoded = String::new();
    let mut rest = text;
    while !rest.is_empty() {
        if rest.starts_with('&') {
            if let Some((entity, value)) =
                ENTITIES.iter().find(|(entity, _)| rest.starts_with(entity))
            {
                decoded.push(*value);
                rest = &rest[entity.len()..];
            } else {
                let alternatives = ENTITIES
                    .iter()
                    .filter(|(entity, _)| entity.starts_with(rest))
                    .map(|(_, value)| format!("{decoded}{value}"))
                    .collect();
                return Some((alternatives, false));
            }
        } else {
            let value = rest.chars().next()?;
            if matches!(value, '<' | '>' | '"' | '\'') {
                return None;
            }
            decoded.push(value);
            rest = &rest[value.len_utf8()..];
        }
    }
    Some((vec![decoded], true))
}

fn name_prefix(base: &str, raw: &str) -> Option<CanonicalXmlPrefix> {
    let (values, stable) = decode(raw)?;
    let alternatives = values
        .into_iter()
        .filter(|value| value.len() <= MAX_TOOL_NAME_BYTES)
        .filter_map(|value| string_prefix(&value).map(|prefix| format!("{base}{prefix}")))
        .collect();
    result(alternatives, false, stable)
}

/// Project the canonical JSON-valued XML generation language into the existing
/// tagged tool grammar. This does not change inbound XML admission. Schema
/// roots remain compiled once, independently of this spelling transformation.
/// Invalid syntax returns `None`, not a permissive raw-text fallback.
pub fn canonical_xml_prefix(input: &str) -> Option<CanonicalXmlPrefix> {
    if input.len() > MAX_TOOL_PAYLOAD_BYTES {
        return None;
    }
    let mut rest = input;
    let mut logical = String::from("<tool_call>\n{\"name\":");
    if INVOKE.starts_with(rest) && rest != INVOKE {
        logical.push('"');
        return result(vec![logical], false, false);
    }
    rest = rest.strip_prefix(INVOKE)?;
    let Some(end) = rest.find('"') else {
        return name_prefix(&logical, rest);
    };
    let (names, stable) = decode(&rest[..end])?;
    if !stable || names.len() != 1 {
        return None;
    }
    let name = &names[0];
    if name.is_empty() || name.len() > MAX_TOOL_NAME_BYTES {
        return None;
    }
    logical.push_str(&serde_json::to_string(name).ok()?);
    logical.push_str(",\"arguments\":{");
    rest = &rest[end..];
    let suffix = "\">";
    if suffix.starts_with(rest) {
        return result(vec![logical], false, false);
    }
    rest = rest.strip_prefix(suffix)?;
    let mut keys = BTreeSet::new();
    loop {
        let mut alternatives = Vec::new();
        if END_INVOKE.starts_with(rest) {
            alternatives.push(format!("{logical}}}}}\n</tool_call>"));
        }
        if PARAMETER.starts_with(rest) && rest != PARAMETER {
            alternatives.push(format!(
                "{logical}{}\"",
                if keys.is_empty() { "" } else { "," }
            ));
        }
        if !alternatives.is_empty() {
            let complete = rest == END_INVOKE;
            if complete && !matches!(parse_atem_xml(input), ToolParseOutcome::CompleteSoFar(_)) {
                return None;
            }
            return result(alternatives, complete, false);
        }
        rest = rest.strip_prefix(PARAMETER)?;
        let base = format!("{logical}{}", if keys.is_empty() { "" } else { "," });
        let Some(end) = rest.find('"') else {
            return name_prefix(&base, rest);
        };
        let (names, stable) = decode(&rest[..end])?;
        if !stable || names.len() != 1 {
            return None;
        }
        let key = &names[0];
        if key.is_empty() || key.len() > MAX_TOOL_NAME_BYTES || !keys.insert(key.clone()) {
            return None;
        }
        logical = format!("{base}{}:", serde_json::to_string(key).ok()?);
        rest = &rest[end..];
        if suffix.starts_with(rest) {
            return result(vec![logical], false, false);
        }
        rest = rest.strip_prefix(suffix)?;
        let Some(end) = rest.find('<') else {
            let (values, stable) = decode(rest)?;
            return result(
                values
                    .into_iter()
                    .map(|value| format!("{logical}{value}"))
                    .collect(),
                false,
                stable,
            );
        };
        let (values, stable) = decode(&rest[..end])?;
        if !stable || values.len() != 1 {
            return None;
        }
        // Closing the parameter fixes classification: only actual JSON is in
        // this generation language, including for string-valued parameters.
        let _: serde_json::Value = serde_json::from_str(values[0].trim()).ok()?;
        // Preserve the exact JSON spelling checked before the delimiter. A
        // completed token must not normalize an earlier rejected escape or
        // numeric prefix into a different backend language after the fact.
        logical.push_str(&values[0]);
        rest = &rest[end..];
        if END_PARAMETER.starts_with(rest) && rest != END_PARAMETER {
            return result(
                vec![
                    format!("{logical}}}}}\n</tool_call>"),
                    format!("{logical},\""),
                ],
                false,
                false,
            );
        }
        rest = rest.strip_prefix(END_PARAMETER)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_generation_is_a_subset_of_the_unchanged_inbound_xml_language() {
        let prefix = "<atem:invoke name=\"lookup\"><atem:parameter name=\"city\">";
        for value in [
            "1",
            "true",
            "null",
            "[1,true]",
            "{&quot;x&quot;:1}",
            "&quot;1&quot;",
            "&quot;Paris&quot;",
        ] {
            let text = format!("{prefix}{value}</atem:parameter></atem:invoke>");
            assert!(canonical_xml_prefix(&text).unwrap().complete, "{text}");
            assert!(matches!(
                parse_atem_xml(&text),
                ToolParseOutcome::CompleteSoFar(_)
            ));
        }
        // Inbound fallback remains available; generation deliberately has no
        // corresponding string fallback that could admit an impossible prefix.
        let raw = format!("{prefix}1x</atem:parameter></atem:invoke>");
        assert!(canonical_xml_prefix(&raw).is_none());
        let ToolParseOutcome::CompleteSoFar(calls) = parse_atem_xml(&raw) else {
            panic!("raw inbound text rejected");
        };
        assert_eq!(calls[0].arguments, r#"{"city":"1x"}"#);
    }
    #[test]
    fn canonical_generation_rejects_duplicate_nested_or_noncanonical_structure() {
        for text in [
            "<atem:invoke name=\"lookup\"><atem:parameter name=\"a\">1</atem:parameter><atem:parameter name=\"a\">",
            "<atem:invoke name=\"lookup\"><atem:parameter name=\"a\"><atem:invoke",
            "<atem:invoke name=\"lookup\"><atem:parameter name=\"a\">&unknown;",
            "<atem:invoke name='lookup'>",
        ] {
            assert!(canonical_xml_prefix(text).is_none(), "{text}");
        }
    }
}
