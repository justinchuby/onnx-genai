//! Exact-version, transport-neutral tool-call protocol parsing.
//!
//! Packages declare a single identity/version pair.  This module resolves that
//! pair without parser probing and classifies incremental model output before a
//! transport turns it into an API-specific tool-call shape.

use std::{collections::BTreeSet, fmt, path::Path};

use crate::ToolProtocolDeclaration;

/// Maximum complete protocol envelope size accepted from a model.
pub const MAX_TOOL_PAYLOAD_BYTES: usize = 64 * 1024;
/// Maximum calls in one completed protocol envelope sequence.
pub const MAX_TOOL_CALLS: usize = 32;
/// Maximum UTF-8 byte size of a tool name.
pub const MAX_TOOL_NAME_BYTES: usize = 256;
/// Maximum UTF-8 byte size of a protocol call identifier.
pub const MAX_TOOL_CALL_ID_BYTES: usize = 256;

/// A parsed function call independent of a request or response transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Canonical JSON encoding of the required object arguments.
    pub arguments: String,
}

/// A result while consuming model output one arbitrary chunk at a time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolParseOutcome {
    /// This protocol's opening envelope has not appeared.
    NoCall,
    /// An opening envelope appeared but cannot yet be decided.
    Incomplete,
    /// One or more complete envelopes have been parsed, but this v1 protocol
    /// has no sequence terminator, so more adjacent calls may still follow.
    CompleteSoFar(Vec<ToolCall>),
    /// The independent generation boundary ended a valid envelope sequence.
    TerminalComplete(Vec<ToolCall>),
    /// The envelope is complete enough to reject, with an actionable reason.
    Malformed(String),
}

/// Exact protocol implementation selected by metadata declaration.
///
/// The enum deliberately has no fallback or trial order: an identity/version
/// pair selects exactly one parser before any generated text is examined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolProtocol {
    TaggedJsonV1,
    AtemXmlV1,
}

impl ToolProtocol {
    pub fn declaration(self) -> (&'static str, &'static str) {
        match self {
            Self::TaggedJsonV1 => ("tagged-json", "v1"),
            Self::AtemXmlV1 => ("atem-xml", "v1"),
        }
    }

    pub fn parse(self, input: &str) -> ToolParseOutcome {
        if input.len() > MAX_TOOL_PAYLOAD_BYTES {
            return ToolParseOutcome::Malformed(format!(
                "tool output exceeds the {MAX_TOOL_PAYLOAD_BYTES}-byte protocol limit; reduce the model envelope"
            ));
        }
        match self {
            Self::TaggedJsonV1 => parse_tagged_json(input),
            Self::AtemXmlV1 => parse_atem_xml(input),
        }
    }

    /// Convert a terminal parser outcome into a protocol-specific error at the
    /// boundary where complete model output is known.
    pub fn output_error(
        self,
        outcome: &ToolParseOutcome,
        boundary: &str,
    ) -> Option<ToolProtocolError> {
        let (identity, version) = self.declaration();
        match outcome {
            ToolParseOutcome::NoCall
            | ToolParseOutcome::CompleteSoFar(_)
            | ToolParseOutcome::TerminalComplete(_) => None,
            ToolParseOutcome::Incomplete => Some(ToolProtocolError(format!(
                "declared tool protocol {identity}@{version} produced an incomplete envelope at the {boundary}; the opening envelope reached end of output"
            ))),
            ToolParseOutcome::Malformed(reason) => Some(ToolProtocolError(format!(
                "declared tool protocol {identity}@{version} produced a malformed envelope at the {boundary}: {reason}"
            ))),
        }
    }

    /// Reject only failures that are terminal while more chunks may arrive.
    pub fn incremental_output_error(
        self,
        outcome: &ToolParseOutcome,
        boundary: &str,
    ) -> Option<ToolProtocolError> {
        matches!(outcome, ToolParseOutcome::Malformed(_))
            .then(|| self.output_error(outcome, boundary))
            .flatten()
    }

    pub fn from_declaration(
        declaration: &ToolProtocolDeclaration,
    ) -> Result<Self, ToolProtocolError> {
        resolve(
            declaration,
            Path::new("the supplied tool protocol declaration"),
        )
    }
}

/// Incremental parsing state. Chunk boundaries have no semantic effect.
#[derive(Debug, Default, Clone)]
pub struct ToolCallStream {
    input: String,
    terminal_error: Option<String>,
}

impl ToolCallStream {
    pub fn push(&mut self, protocol: ToolProtocol, chunk: &str) -> ToolParseOutcome {
        if let Some(error) = &self.terminal_error {
            return ToolParseOutcome::Malformed(error.clone());
        }
        if self.input.len().saturating_add(chunk.len()) > MAX_TOOL_PAYLOAD_BYTES {
            let error = format!(
                "tool output exceeds the {MAX_TOOL_PAYLOAD_BYTES}-byte protocol limit; reduce the model envelope"
            );
            self.terminal_error = Some(error.clone());
            return ToolParseOutcome::Malformed(error);
        }
        self.input.push_str(chunk);
        let outcome = protocol.parse(&self.input);
        if let ToolParseOutcome::Malformed(error) = &outcome {
            self.terminal_error = Some(error.clone());
        }
        outcome
    }

    pub fn finish(self, protocol: ToolProtocol) -> ToolParseOutcome {
        if let Some(error) = self.terminal_error {
            return ToolParseOutcome::Malformed(error);
        }
        match protocol.parse(&self.input) {
            ToolParseOutcome::CompleteSoFar(calls) => ToolParseOutcome::TerminalComplete(calls),
            outcome => outcome,
        }
    }
}

/// Failure to resolve an exact protocol declaration or consume its output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolProtocolError(pub String);

impl fmt::Display for ToolProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ToolProtocolError {}

/// Resolve the exact declaration supplied by metadata, without fallback or
/// parser probing.
pub fn resolve(
    declaration: &ToolProtocolDeclaration,
    metadata_path: &Path,
) -> Result<ToolProtocol, ToolProtocolError> {
    match (declaration.identity.as_str(), declaration.version.as_str()) {
        ("tagged-json", "v1") => Ok(ToolProtocol::TaggedJsonV1),
        ("atem-xml", "v1") => Ok(ToolProtocol::AtemXmlV1),
        _ => Err(ToolProtocolError(format!(
            "{} declares package.tool_protocol identity {:?} version {:?}, but this runtime implements no such protocol. \
             Use one of tagged-json@v1 or atem-xml@v1, or install a runtime adapter for the declared protocol.",
            metadata_path.display(),
            declaration.identity,
            declaration.version,
        ))),
    }
}

fn parse_tagged_json(input: &str) -> ToolParseOutcome {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    if input.trim().is_empty() {
        return ToolParseOutcome::NoCall;
    }
    let Some((prefix, mut rest)) = input.split_once(OPEN) else {
        return if OPEN.starts_with(input.trim_start()) {
            ToolParseOutcome::Incomplete
        } else {
            ToolParseOutcome::NoCall
        };
    };
    if !prefix.trim().is_empty() {
        return ToolParseOutcome::Malformed(format!(
            "tagged-json@v1 has non-whitespace text before the first tool_call envelope: \
             {prefix:?}"
        ));
    }
    let mut values = Vec::new();
    loop {
        let Some((body, after)) = rest.split_once(CLOSE) else {
            return ToolParseOutcome::Incomplete;
        };
        match serde_json::from_str::<serde_json::Value>(body.trim()) {
            Ok(value) => values.push(value),
            Err(error) => {
                return ToolParseOutcome::Malformed(format!(
                    "tagged-json@v1 tool_call envelope contains invalid JSON: {error}"
                ));
            }
        }
        match after.split_once(OPEN) {
            Some((between, next)) if between.trim().is_empty() => rest = next,
            Some((between, _)) => {
                return ToolParseOutcome::Malformed(format!(
                    "tagged-json@v1 has non-whitespace text between tool_call envelopes: {between:?}"
                ));
            }
            None => {
                let trailing = after.trim_start();
                if trailing.is_empty() {
                    return values_to_calls("tagged-json@v1", values);
                }
                if OPEN.starts_with(trailing) {
                    return ToolParseOutcome::Incomplete;
                }
                return ToolParseOutcome::Malformed(
                    "tagged-json@v1 has trailing text after a tool_call envelope".to_string(),
                );
            }
        }
    }
}

fn parse_atem_xml(input: &str) -> ToolParseOutcome {
    const OPEN: &str = "<atem:invoke";
    const CLOSE: &str = "</atem:invoke>";
    let first = input.trim_start();
    if first.is_empty() {
        return ToolParseOutcome::NoCall;
    }
    let mut rest = if let Some(rest) = first.strip_prefix(OPEN) {
        rest
    } else if OPEN.starts_with(first) {
        return ToolParseOutcome::Incomplete;
    } else if let Some((prefix, _)) = input.split_once(OPEN) {
        return ToolParseOutcome::Malformed(format!(
            "atem-xml@v1 has non-whitespace text before the first invoke envelope: {prefix:?}"
        ));
    } else {
        return ToolParseOutcome::NoCall;
    };
    let mut values = Vec::new();
    loop {
        let Some((open_tag, body)) = rest.split_once('>') else {
            return ToolParseOutcome::Incomplete;
        };
        let Some(name) = xml_attribute(open_tag, "name") else {
            return ToolParseOutcome::Malformed(
                "atem-xml@v1 invoke envelope is missing its quoted name attribute".to_string(),
            );
        };
        let Some((body, after)) = body.split_once(CLOSE) else {
            return ToolParseOutcome::Incomplete;
        };
        let mut arguments = serde_json::Map::new();
        let mut parameters = body;
        while !parameters.trim().is_empty() {
            parameters = parameters.trim_start();
            let Some(after_open) = parameters.strip_prefix("<atem:parameter") else {
                return ToolParseOutcome::Malformed(
                    "atem-xml@v1 invoke envelope contains text outside parameter envelopes"
                        .to_string(),
                );
            };
            let Some((parameter_tag, after_tag)) = after_open.split_once('>') else {
                return ToolParseOutcome::Incomplete;
            };
            let Some(key) = xml_attribute(parameter_tag, "name") else {
                return ToolParseOutcome::Malformed(
                    "atem-xml@v1 parameter envelope is missing its quoted name attribute"
                        .to_string(),
                );
            };
            let Some((raw, after_parameter)) = after_tag.split_once("</atem:parameter>") else {
                return ToolParseOutcome::Incomplete;
            };
            if raw.contains("<atem:parameter") {
                return ToolParseOutcome::Malformed(
                    "atem-xml@v1 does not permit nested parameter envelopes".to_string(),
                );
            }
            let raw = xml_unescape(raw);
            let value =
                serde_json::from_str(raw.trim()).unwrap_or_else(|_| serde_json::Value::String(raw));
            if arguments.insert(key, value).is_some() {
                return ToolParseOutcome::Malformed(
                    "atem-xml@v1 invoke envelope has duplicate parameter names".to_string(),
                );
            }
            parameters = after_parameter;
        }
        values.push(serde_json::json!({ "name": name, "arguments": arguments }));
        match after.split_once(OPEN) {
            Some((between, next)) if between.trim().is_empty() => rest = next,
            Some((between, _)) => {
                return ToolParseOutcome::Malformed(format!(
                    "atem-xml@v1 has non-whitespace text between invoke envelopes: {between:?}"
                ));
            }
            None => {
                let trailing = after.trim_start();
                if trailing.is_empty() {
                    return values_to_calls("atem-xml@v1", values);
                }
                if OPEN.starts_with(trailing) {
                    return ToolParseOutcome::Incomplete;
                }
                return ToolParseOutcome::Malformed(format!(
                    "atem-xml@v1 has non-whitespace trailing text after an invoke envelope: {trailing:?}"
                ));
            }
        }
    }
}

fn xml_attribute(tag: &str, attribute: &str) -> Option<String> {
    let marker = format!("{attribute}=\"");
    let value = tag.split_once(&marker)?.1;
    let value = value.split_once('"')?.0;
    let value = xml_unescape(value);
    (!value.is_empty() && value.len() <= MAX_TOOL_NAME_BYTES).then_some(value)
}

fn xml_unescape(value: &str) -> String {
    value
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn values_to_calls(protocol: &str, values: Vec<serde_json::Value>) -> ToolParseOutcome {
    if values.is_empty() || values.len() > MAX_TOOL_CALLS {
        return ToolParseOutcome::Malformed(format!(
            "{protocol} must contain from 1 to {MAX_TOOL_CALLS} calls"
        ));
    }
    let mut ids = BTreeSet::new();
    let mut calls = Vec::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        let Some(name) = value.get("name").and_then(serde_json::Value::as_str) else {
            return ToolParseOutcome::Malformed(format!(
                "{protocol} call {index} is missing a string name"
            ));
        };
        if name.is_empty() || name.len() > MAX_TOOL_NAME_BYTES {
            return ToolParseOutcome::Malformed(format!(
                "{protocol} call {index} name must contain 1 to {MAX_TOOL_NAME_BYTES} bytes"
            ));
        }
        let id = value
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("call_{index}"));
        if id.is_empty() || id.len() > MAX_TOOL_CALL_ID_BYTES || !ids.insert(id.clone()) {
            return ToolParseOutcome::Malformed(format!(
                "{protocol} call {index} has an empty, oversized, or duplicate id"
            ));
        }
        let Some(arguments) = value.get("arguments").or_else(|| value.get("parameters")) else {
            return ToolParseOutcome::Malformed(format!(
                "{protocol} call {index} is missing required arguments or parameters"
            ));
        };
        if !arguments.is_object() {
            return ToolParseOutcome::Malformed(format!(
                "{protocol} call {index} arguments or parameters must be a JSON object"
            ));
        }
        let arguments = match serde_json::to_string(arguments) {
            Ok(arguments) if arguments.len() <= MAX_TOOL_PAYLOAD_BYTES => arguments,
            Ok(_) => {
                return ToolParseOutcome::Malformed(format!(
                    "{protocol} call {index} arguments exceed the {MAX_TOOL_PAYLOAD_BYTES}-byte limit"
                ));
            }
            Err(error) => {
                return ToolParseOutcome::Malformed(format!(
                    "{protocol} call {index} arguments cannot be encoded: {error}"
                ));
            }
        };
        calls.push(ToolCall {
            id,
            name: name.to_string(),
            arguments,
        });
    }
    ToolParseOutcome::CompleteSoFar(calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protocol(identity: &str) -> ToolProtocol {
        ToolProtocol::from_declaration(&ToolProtocolDeclaration {
            identity: identity.to_string(),
            version: "v1".to_string(),
        })
        .unwrap()
    }

    #[test]
    fn chunk_boundaries_do_not_change_declared_protocol_outcomes() {
        for (identity, input) in [
            (
                "tagged-json",
                "<tool_call>{\"id\":\"weather-1\",\"name\":\"weather\",\"arguments\":{\"city\":\"A\\\\\\\"B\"}}</tool_call>",
            ),
            (
                "atem-xml",
                "<atem:invoke name=\"weather\"><atem:parameter name=\"city\">\"Paris\"</atem:parameter></atem:invoke>",
            ),
        ] {
            let resolved = protocol(identity);
            let expected = resolved.parse(input);
            for split in 0..=input.len() {
                if !input.is_char_boundary(split) {
                    continue;
                }
                let mut stream = ToolCallStream::default();
                stream.push(resolved, &input[..split]);
                assert_eq!(
                    stream.push(resolved, &input[split..]),
                    expected,
                    "{identity} split at {split}"
                );
            }
        }
    }

    #[test]
    fn exact_identity_and_version_select_one_parser() {
        for declaration in [
            ToolProtocolDeclaration {
                identity: "tagged-json".to_string(),
                version: "v1".to_string(),
            },
            ToolProtocolDeclaration {
                identity: "atem-xml".to_string(),
                version: "v1".to_string(),
            },
        ] {
            assert!(resolve(&declaration, Path::new("fixture.yaml")).is_ok());
        }
        let error = resolve(
            &ToolProtocolDeclaration {
                identity: "tagged-json".to_string(),
                version: "v2".to_string(),
            },
            Path::new("fixture.yaml"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("fixture.yaml"), "{error}");
        assert!(error.contains("\"v2\""), "{error}");
    }

    #[test]
    fn incomplete_is_nonterminal_until_finish_and_malformed_latches() {
        let protocol = protocol("tagged-json");
        let mut stream = ToolCallStream::default();
        assert_eq!(
            stream.push(protocol, "<tool_cal"),
            ToolParseOutcome::Incomplete
        );
        assert_eq!(stream.finish(protocol), ToolParseOutcome::Incomplete);
        let mut stream = ToolCallStream::default();
        assert_eq!(
            stream.push(
                protocol,
                r#"<tool_call>{"name":"read","arguments":{"path":"src/"#
            ),
            ToolParseOutcome::Incomplete
        );
        assert!(
            protocol
                .incremental_output_error(&ToolParseOutcome::Incomplete, "chunk")
                .is_none()
        );
        assert!(matches!(
            stream.push(protocol, r#"lib.rs"}}</tool_call>"#),
            ToolParseOutcome::CompleteSoFar(_)
        ));

        let mut malformed = ToolCallStream::default();
        assert!(matches!(
            malformed.push(protocol, "<tool_call>{bad}</tool_call>"),
            ToolParseOutcome::Malformed(_)
        ));
        assert!(matches!(
            malformed.push(protocol, "<tool_call>{}</tool_call>"),
            ToolParseOutcome::Malformed(_)
        ));
    }

    #[test]
    fn empty_output_is_a_terminal_no_call_for_every_protocol() {
        for identity in ["tagged-json", "atem-xml"] {
            let protocol = protocol(identity);
            assert_eq!(protocol.parse(" \n\t"), ToolParseOutcome::NoCall);
            assert_eq!(
                ToolCallStream::default().finish(protocol),
                ToolParseOutcome::NoCall
            );
        }
    }

    #[test]
    fn tagged_json_requires_object_arguments_and_atem_requires_envelope_only_text() {
        let tagged = protocol("tagged-json");
        for input in [
            r#"<tool_call>{"name":"read"}</tool_call>"#,
            r#"<tool_call>{"name":"read","arguments":null}</tool_call>"#,
            r#"<tool_call>{"name":"read","arguments":[]}</tool_call>"#,
        ] {
            assert!(matches!(
                tagged.parse(input),
                ToolParseOutcome::Malformed(_)
            ));
        }
        assert!(matches!(
            tagged.parse(r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#),
            ToolParseOutcome::CompleteSoFar(_)
        ));

        let atem = protocol("atem-xml");
        let call = r#"<atem:invoke name="read"></atem:invoke>"#;
        for input in [format!("junk{call}"), format!("{call}junk")] {
            assert!(matches!(atem.parse(&input), ToolParseOutcome::Malformed(_)));
        }
    }

    #[test]
    fn multiple_calls_and_the_payload_bound_are_enforced() {
        let protocol = protocol("tagged-json");
        let input = concat!(
            r#"<tool_call>{"id":"one","name":"read","arguments":{}}</tool_call>"#,
            "\n",
            r#"<tool_call>{"id":"two","name":"write","parameters":{"path":"x"}}</tool_call>"#
        );
        let ToolParseOutcome::CompleteSoFar(calls) = protocol.parse(input) else {
            panic!("multiple calls must parse");
        };
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].name, "write");
        assert!(matches!(
            protocol.parse(&"x".repeat(MAX_TOOL_PAYLOAD_BYTES + 1)),
            ToolParseOutcome::Malformed(message) if message.contains("byte protocol limit")
        ));
    }

    #[test]
    fn adjacent_calls_are_chunk_independent_and_only_finish_makes_them_terminal() {
        for (identity, envelope) in [
            (
                "tagged-json",
                r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#,
            ),
            ("atem-xml", r#"<atem:invoke name="read"></atem:invoke>"#),
        ] {
            let protocol = protocol(identity);
            let input = format!("{envelope}\n{envelope}\t{envelope}");
            let mut stream = ToolCallStream::default();
            for character in input.chars() {
                stream.push(protocol, &character.to_string());
            }
            assert!(matches!(
                protocol.parse(&input),
                ToolParseOutcome::CompleteSoFar(ref calls) if calls.len() == 3
            ));
            assert!(matches!(
                stream.finish(protocol),
                ToolParseOutcome::TerminalComplete(ref calls) if calls.len() == 3
            ));
        }
    }

    #[test]
    fn a_complete_envelope_followed_by_a_partial_one_stays_incomplete() {
        for (identity, complete, partial) in [
            (
                "tagged-json",
                r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#,
                "<tool_call>",
            ),
            (
                "atem-xml",
                r#"<atem:invoke name="read"></atem:invoke>"#,
                "<atem:invoke",
            ),
        ] {
            let protocol = protocol(identity);
            let mut stream = ToolCallStream::default();
            assert!(matches!(
                stream.push(protocol, complete),
                ToolParseOutcome::CompleteSoFar(ref calls) if calls.len() == 1
            ));
            assert_eq!(stream.push(protocol, partial), ToolParseOutcome::Incomplete);
            assert_eq!(stream.finish(protocol), ToolParseOutcome::Incomplete);
        }
    }

    #[test]
    fn stream_byte_limit_covers_the_whole_pending_call_sequence() {
        let protocol = protocol("tagged-json");
        let envelope = r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#;
        let mut stream = ToolCallStream::default();
        assert!(matches!(
            stream.push(protocol, envelope),
            ToolParseOutcome::CompleteSoFar(ref calls) if calls.len() == 1
        ));
        assert!(matches!(
            stream.push(protocol, &" ".repeat(MAX_TOOL_PAYLOAD_BYTES - envelope.len())),
            ToolParseOutcome::CompleteSoFar(ref calls) if calls.len() == 1
        ));
        assert!(matches!(
            stream.push(protocol, " "),
            ToolParseOutcome::Malformed(message) if message.contains("byte protocol limit")
        ));
    }
}
