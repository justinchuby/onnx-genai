//! Provisional snapshots, not completed or schema-admitted tool calls.
use super::*;
use serde_json::Value;

/// A cumulative argument prefix belonging to one model envelope.
/// Consumers must admit the prefix and retain a stable wire identity. A final
/// protocol parse and complete batch admission are still required before use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallFragment {
    pub index: usize,
    pub id: Option<String>,
    pub name: String,
    pub arguments: String,
    pub envelope_complete: bool,
}

struct JsonEnvelope<'a> {
    body: &'a str,
    value: Option<Value>,
    closed: bool,
}

/// JSON, rather than a string delimiter search, determines the body boundary.
/// Thus a closing marker in a JSON string is ordinary argument data.
fn json_envelopes(input: &str) -> Result<Option<Vec<JsonEnvelope<'_>>>, String> {
    let mut rest = input.trim_start();
    let mut envelopes = Vec::new();
    if rest.is_empty() || TAGGED_JSON_MARKERS[0].starts_with(rest) {
        return Ok(Some(envelopes));
    }
    if !rest.starts_with(TAGGED_JSON_MARKERS[0]) {
        return if rest.contains(TAGGED_JSON_MARKERS[0]) {
            Err("tagged-json@v1 has non-whitespace text before its first envelope".into())
        } else {
            Ok(None)
        };
    }
    loop {
        if envelopes.len() == MAX_TOOL_CALLS {
            return Err(format!("tagged-json@v1 exceeds {MAX_TOOL_CALLS} calls"));
        }
        rest = rest
            .strip_prefix(TAGGED_JSON_MARKERS[0])
            .expect("checked opening marker")
            .trim_start();
        let mut values = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
        let value = match values.next() {
            Some(Ok(value)) => value,
            None => {
                envelopes.push(JsonEnvelope {
                    body: rest,
                    value: None,
                    closed: false,
                });
                break;
            }
            Some(Err(error)) if error.is_eof() => {
                envelopes.push(JsonEnvelope {
                    body: rest,
                    value: None,
                    closed: false,
                });
                break;
            }
            Some(Err(error)) => return Err(format!("tagged-json@v1 invalid JSON: {error}")),
        };
        let end = values.byte_offset();
        let body = &rest[..end];
        rest = rest[end..].trim_start();
        let Some(after) = rest.strip_prefix(TAGGED_JSON_MARKERS[1]) else {
            if TAGGED_JSON_MARKERS[1].starts_with(rest) {
                envelopes.push(JsonEnvelope {
                    body,
                    value: Some(value),
                    closed: false,
                });
                break;
            }
            return Err("tagged-json@v1 requires </tool_call> after its JSON object".into());
        };
        envelopes.push(JsonEnvelope {
            body,
            value: Some(value),
            closed: true,
        });
        rest = after.trim_start();
        if rest.is_empty() {
            break;
        }
        if TAGGED_JSON_MARKERS[0].starts_with(rest) {
            if envelopes.len() == MAX_TOOL_CALLS {
                return Err(format!("tagged-json@v1 exceeds {MAX_TOOL_CALLS} calls"));
            }
            envelopes.push(JsonEnvelope {
                body: "",
                value: None,
                closed: false,
            });
            break;
        }
        if !rest.starts_with(TAGGED_JSON_MARKERS[0]) {
            return Err("tagged-json@v1 has non-whitespace text between or after envelopes".into());
        }
    }
    Ok(Some(envelopes))
}

fn value_prefix(input: &str) -> Result<Option<(Value, usize)>, String> {
    let mut values = serde_json::Deserializer::from_str(input).into_iter::<Value>();
    match values.next() {
        Some(Ok(value)) => Ok(Some((value, values.byte_offset()))),
        None => Ok(None),
        Some(Err(error)) if error.is_eof() => Ok(None),
        Some(Err(error)) => Err(format!("tagged-json@v1 invalid field: {error}")),
    }
}

#[derive(Default)]
struct Fields<'a> {
    id: Option<String>,
    name: Option<String>,
    arguments: Option<&'a str>,
    parameters: Option<&'a str>,
    complete: bool,
}

fn json_fields(body: &str) -> Result<Fields<'_>, String> {
    let mut fields = Fields::default();
    let mut rest = body.trim_start();
    if rest.is_empty() {
        return Ok(fields);
    }
    rest = rest
        .strip_prefix('{')
        .ok_or("tagged-json@v1 call must be a JSON object")?
        .trim_start();
    let mut keys = BTreeSet::new();
    loop {
        if rest.starts_with('}') {
            fields.complete = true;
            return Ok(fields);
        }
        let Some((key, end)) = value_prefix(rest)? else {
            return Ok(fields);
        };
        let key = key
            .as_str()
            .ok_or("tagged-json@v1 field name must be a string")?;
        if !keys.insert(key.to_owned()) {
            return Err(format!("tagged-json@v1 duplicate call field {key:?}"));
        }
        rest = rest[end..].trim_start();
        if rest.is_empty() {
            return Ok(fields);
        }
        rest = rest
            .strip_prefix(':')
            .ok_or("tagged-json@v1 field requires ':'")?
            .trim_start();
        if key == "arguments" && !rest.is_empty() && !rest.starts_with('{') {
            return Err(
                "tagged-json@v1 arguments must be a JSON object; emit an object argument prefix"
                    .into(),
            );
        }
        let parsed = value_prefix(rest)?;
        if matches!(key, "arguments" | "parameters") && rest.starts_with('{') {
            let raw = &rest[..parsed.as_ref().map_or(rest.len(), |(_, end)| *end)];
            if key == "arguments" {
                fields.arguments = Some(raw);
            } else {
                fields.parameters = Some(raw);
            }
        }
        let Some((value, end)) = parsed else {
            return Ok(fields);
        };
        if matches!(key, "name" | "id")
            && let Some(value) = value.as_str()
        {
            let limit = if key == "id" {
                MAX_TOOL_CALL_ID_BYTES
            } else {
                MAX_TOOL_NAME_BYTES
            };
            if value.is_empty() || value.len() > limit {
                return Err(format!(
                    "tagged-json@v1 field {key:?} must contain 1 to {limit} bytes"
                ));
            }
            if key == "name" {
                fields.name = Some(value.into());
            } else {
                fields.id = Some(value.into());
            }
        }
        rest = rest[end..].trim_start();
        if rest.starts_with('}') {
            fields.complete = true;
            return Ok(fields);
        }
        if rest.is_empty() {
            return Ok(fields);
        }
        rest = rest
            .strip_prefix(',')
            .ok_or("tagged-json@v1 fields require a comma")?
            .trim_start();
    }
}

pub(super) fn parse_json(input: &str) -> ToolParseOutcome {
    match json_envelopes(input) {
        Ok(None) => ToolParseOutcome::NoCall,
        Ok(Some(envelopes)) if envelopes.is_empty() => {
            if input.trim().is_empty() {
                ToolParseOutcome::NoCall
            } else {
                ToolParseOutcome::Incomplete
            }
        }
        Ok(Some(envelopes)) => {
            for envelope in &envelopes {
                if let Err(error) = json_fields(envelope.body) {
                    return ToolParseOutcome::Malformed(error);
                }
            }
            if envelopes.iter().any(|envelope| !envelope.closed) {
                return ToolParseOutcome::Incomplete;
            }
            values_to_calls(
                "tagged-json@v1",
                envelopes
                    .into_iter()
                    .map(|envelope| envelope.value.expect("closed JSON"))
                    .collect(),
            )
        }
        Err(error) => ToolParseOutcome::Malformed(error),
    }
}

pub(super) fn snapshot(
    protocol: ToolProtocol,
    input: &str,
) -> Result<Vec<ToolCallFragment>, String> {
    match protocol {
        ToolProtocol::TaggedJsonV1 => {
            let mut fragments = Vec::new();
            for (index, envelope) in json_envelopes(input)?
                .unwrap_or_default()
                .into_iter()
                .enumerate()
            {
                let fields = json_fields(envelope.body)?;
                // `arguments` wins over `parameters` in this protocol. Until
                // the object closes, a later arguments field is still possible.
                let arguments = fields
                    .arguments
                    .or(fields.complete.then_some(fields.parameters).flatten())
                    .unwrap_or("");
                if let Some(name) = fields.name {
                    fragments.push(ToolCallFragment {
                        index,
                        id: fields.id,
                        name,
                        arguments: arguments.into(),
                        envelope_complete: envelope.closed,
                    });
                }
            }
            Ok(fragments)
        }
        ToolProtocol::AtemXmlV1 => xml_snapshot(input),
    }
}

enum XmlScan {
    NoCall,
    Incomplete(Vec<ToolCallFragment>),
    Complete(Vec<ToolCallFragment>),
}

fn xml_tag_end(text: &str) -> Option<usize> {
    let mut quote = None;
    for (index, byte) in text.bytes().enumerate() {
        if quote == Some(byte) {
            quote = None;
        } else if quote.is_none() && matches!(byte, b'\'' | b'"') {
            quote = Some(byte);
        } else if byte == b'>' && quote.is_none() {
            return Some(index);
        }
    }
    None
}

/// One scan supplies both authoritative full parsing and provisional prefixes.
/// A partial delimiter can still grow; incompatible text cannot be repaired by
/// appending more bytes and is rejected before publishing a provisional item.
fn scan_xml(input: &str) -> Result<XmlScan, String> {
    let mut rest = input.trim_start();
    let mut fragments = Vec::new();
    if rest.is_empty() {
        return Ok(XmlScan::NoCall);
    }
    if ATEM_XML_MARKERS[0].starts_with(rest) {
        return Ok(XmlScan::Incomplete(fragments));
    }
    if !rest.starts_with(ATEM_XML_MARKERS[0]) {
        return if rest.contains(ATEM_XML_MARKERS[0]) {
            Err("atem-xml@v1 has non-whitespace text before its first invoke envelope".into())
        } else {
            Ok(XmlScan::NoCall)
        };
    }
    loop {
        if fragments.len() == MAX_TOOL_CALLS {
            return Err(format!("atem-xml@v1 exceeds {MAX_TOOL_CALLS} calls"));
        }
        let after_open = rest
            .strip_prefix(ATEM_XML_MARKERS[0])
            .expect("checked invoke marker");
        let Some(end) = xml_tag_end(after_open) else {
            return Ok(XmlScan::Incomplete(fragments));
        };
        let name = xml_attribute(&after_open[..end], "name")
            .ok_or("atem-xml@v1 invoke requires one well-formed quoted name attribute, without duplicate attributes")?;
        let remainder = &after_open[end + 1..];
        let (body, after, closed) = match remainder.split_once(ATEM_XML_MARKERS[1]) {
            Some((body, after)) => (body, after, true),
            None => (remainder, "", false),
        };
        let mut parameters = body.trim_start();
        let mut keys = BTreeSet::new();
        let mut arguments = String::from("{");
        while !parameters.is_empty() {
            if ATEM_XML_MARKERS[2].starts_with(parameters)
                || ATEM_XML_MARKERS[1].starts_with(parameters)
            {
                if closed {
                    return Err(
                        "atem-xml@v1 invoke closes with an incomplete parameter envelope".into(),
                    );
                }
                break;
            }
            let Some(after_parameter) = parameters.strip_prefix(ATEM_XML_MARKERS[2]) else {
                return Err(
                    "atem-xml@v1 invoke envelope contains text outside parameter envelopes".into(),
                );
            };
            let Some(end) = xml_tag_end(after_parameter) else {
                if closed {
                    return Err(
                        "atem-xml@v1 invoke closes before its parameter tag is complete".into(),
                    );
                }
                break;
            };
            let key = xml_attribute(&after_parameter[..end], "name")
                .ok_or("atem-xml@v1 parameter requires one well-formed quoted name attribute, without duplicates")?;
            let value = &after_parameter[end + 1..];
            let Some((raw, after_parameter)) = value.split_once(ATEM_XML_MARKERS[3]) else {
                if closed {
                    return Err("atem-xml@v1 invoke closes before its parameter envelope".into());
                }
                if value.contains(ATEM_XML_MARKERS[2]) {
                    return Err("atem-xml@v1 does not permit nested parameter envelopes".into());
                }
                break;
            };
            if raw.contains(ATEM_XML_MARKERS[2]) {
                return Err("atem-xml@v1 does not permit nested parameter envelopes".into());
            }
            if !keys.insert(key.clone()) {
                return Err("atem-xml@v1 invoke envelope has duplicate parameter names".into());
            }
            let raw = xml_unescape(raw);
            let value = serde_json::from_str(raw.trim()).unwrap_or(Value::String(raw));
            if keys.len() > 1 {
                arguments.push(',');
            }
            arguments.push_str(&serde_json::to_string(&key).map_err(|error| error.to_string())?);
            arguments.push(':');
            arguments.push_str(&serde_json::to_string(&value).map_err(|error| error.to_string())?);
            parameters = after_parameter.trim_start();
        }
        if closed {
            arguments.push('}');
        }
        fragments.push(ToolCallFragment {
            index: fragments.len(),
            id: None,
            name,
            arguments,
            envelope_complete: closed,
        });
        if !closed {
            return Ok(XmlScan::Incomplete(fragments));
        }
        rest = after.trim_start();
        if rest.is_empty() {
            return Ok(XmlScan::Complete(fragments));
        }
        if fragments.len() == MAX_TOOL_CALLS {
            return Err(format!("atem-xml@v1 exceeds {MAX_TOOL_CALLS} calls"));
        }
        if ATEM_XML_MARKERS[0].starts_with(rest) {
            return Ok(XmlScan::Incomplete(fragments));
        }
        if !rest.starts_with(ATEM_XML_MARKERS[0]) {
            return Err(
                "atem-xml@v1 has non-whitespace text between or after invoke envelopes".into(),
            );
        }
    }
}

fn xml_snapshot(input: &str) -> Result<Vec<ToolCallFragment>, String> {
    match scan_xml(input)? {
        XmlScan::NoCall => Ok(Vec::new()),
        XmlScan::Incomplete(fragments) | XmlScan::Complete(fragments) => Ok(fragments),
    }
}

pub(super) fn parse_xml(input: &str) -> ToolParseOutcome {
    match scan_xml(input) {
        Ok(XmlScan::NoCall) => ToolParseOutcome::NoCall,
        Ok(XmlScan::Incomplete(_)) => ToolParseOutcome::Incomplete,
        Ok(XmlScan::Complete(fragments)) => {
            let values = fragments
                .into_iter()
                .map(|fragment| {
                    serde_json::from_str::<Value>(&fragment.arguments).map(
                        |arguments| serde_json::json!({"name":fragment.name,"arguments":arguments}),
                    )
                })
                .collect::<Result<Vec<_>, _>>();
            match values {
                Ok(values) => values_to_calls("atem-xml@v1", values),
                Err(error) => ToolParseOutcome::Malformed(format!(
                    "atem-xml@v1 could not encode its parsed parameter values: {error}"
                )),
            }
        }
        Err(error) => ToolParseOutcome::Malformed(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_chunk_independent(protocol: ToolProtocol, input: &str) {
        let mut whole = ToolCallStream::default();
        assert!(matches!(
            whole.push(protocol, input),
            ToolParseOutcome::CompleteSoFar(_)
        ));
        let expected = whole.fragments(protocol).unwrap();
        assert!(!expected.is_empty());
        let ToolParseOutcome::TerminalComplete(calls) = whole.finish(protocol) else {
            panic!("complete fixture");
        };
        for (fragment, call) in expected.iter().zip(&calls) {
            assert_eq!(fragment.name, call.name);
            assert_eq!(
                serde_json::from_str::<Value>(&fragment.arguments).unwrap(),
                serde_json::from_str::<Value>(&call.arguments).unwrap()
            );
        }

        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            let mut stream = ToolCallStream::default();
            stream.push(protocol, &input[..split]);
            let before = stream.fragments(protocol).unwrap();
            assert!(
                !matches!(
                    stream.push(protocol, &input[split..]),
                    ToolParseOutcome::Malformed(_)
                ),
                "split={split}"
            );
            let after = stream.fragments(protocol).unwrap();
            assert_eq!(after, expected, "split={split}");
            for fragment in before {
                let final_fragment = &after[fragment.index];
                assert_eq!(fragment.name, final_fragment.name);
                assert!(
                    final_fragment.arguments.starts_with(&fragment.arguments),
                    "split={split}: {fragment:?}"
                );
            }
            assert!(matches!(
                stream.finish(protocol),
                ToolParseOutcome::TerminalComplete(_)
            ));
        }
    }

    #[test]
    fn tagged_fragments_preserve_raw_spelling_names_ids_and_marker_strings_at_every_split() {
        assert_chunk_independent(
            ToolProtocol::TaggedJsonV1,
            r#"<tool_call>{"id":"first","name":"lookup","arguments":{ "z":"é</tool_call>","a":[{"k":"\u0041"}]}}</tool_call><tool_call>{"arguments":{"x":true},"name":"next","id":"late"}</tool_call>"#,
        );
        let mut stream = ToolCallStream::default();
        stream.push(
            ToolProtocol::TaggedJsonV1,
            r#"<tool_call>{"name":"lookup","arguments":{"city":"Pa"#,
        );
        let fragments = stream.fragments(ToolProtocol::TaggedJsonV1).unwrap();
        assert_eq!(fragments[0].arguments, r#"{"city":"Pa"#);
        assert_eq!(fragments[0].id, None);
        assert!(!fragments[0].envelope_complete);
        stream.push(
            ToolProtocol::TaggedJsonV1,
            r#"ris"},"id":"late"}</tool_call>"#,
        );
        let fragments = stream.fragments(ToolProtocol::TaggedJsonV1).unwrap();
        assert_eq!(fragments[0].id.as_deref(), Some("late"));
        assert_eq!(fragments[0].arguments, r#"{"city":"Paris"}"#);

        let mut alias = ToolCallStream::default();
        alias.push(
            ToolProtocol::TaggedJsonV1,
            r#"<tool_call>{"name":"lookup","parameters":{"ignored":1},"#,
        );
        assert_eq!(
            alias.fragments(ToolProtocol::TaggedJsonV1).unwrap()[0].arguments,
            ""
        );
        alias.push(
            ToolProtocol::TaggedJsonV1,
            r#""arguments":{"actual":2}}</tool_call>"#,
        );
        assert_eq!(
            alias.fragments(ToolProtocol::TaggedJsonV1).unwrap()[0].arguments,
            r#"{"actual":2}"#
        );
    }

    #[test]
    fn xml_fragments_wait_for_ambiguous_value_type_but_stream_completed_parameters() {
        let input = r#"<atem:invoke name="lookup"><atem:parameter name="z">1x</atem:parameter><atem:parameter name="a">&quot;é&quot;</atem:parameter></atem:invoke>"#;
        assert_chunk_independent(ToolProtocol::AtemXmlV1, input);
        let mut stream = ToolCallStream::default();
        stream.push(
            ToolProtocol::AtemXmlV1,
            r#"<atem:invoke name="lookup"><atem:parameter name="z">1"#,
        );
        assert_eq!(
            stream.fragments(ToolProtocol::AtemXmlV1).unwrap()[0].arguments,
            "{"
        );
        stream.push(ToolProtocol::AtemXmlV1, "x</atem:parameter>");
        assert_eq!(
            stream.fragments(ToolProtocol::AtemXmlV1).unwrap()[0].arguments,
            r#"{"z":"1x""#
        );

        assert_chunk_independent(
            ToolProtocol::AtemXmlV1,
            r#"<atem:invoke name="lookup>v"><atem:parameter name="a>b">&quot;é&lt;/atem:invoke&gt;&quot;</atem:parameter></atem:invoke>"#,
        );
        for invalid in [
            r#"<atem:invoke name="lookup">garbage"#,
            r#"<atem:invoke name="lookup"><atem:parameter name="x">v</atem:invoke>"#,
            r#"<atem:invoke not_name="lookup">"#,
            r#"<atem:invoke name="lookup" name="other">"#,
        ] {
            let mut stream = ToolCallStream::default();
            assert!(
                matches!(
                    stream.push(ToolProtocol::AtemXmlV1, invalid),
                    ToolParseOutcome::Malformed(_)
                ),
                "{invalid}"
            );
            assert!(stream.fragments(ToolProtocol::AtemXmlV1).is_err());
        }

        assert_chunk_independent(
            ToolProtocol::AtemXmlV1,
            r#"<atem:invoke name='lookup "v"' note='>'><atem:parameter name='a>b'>true</atem:parameter></atem:invoke>"#,
        );
        let mut bounded = ToolCallStream::default();
        let too_many = format!(
            "{}<atem:",
            r#"<atem:invoke name="lookup"></atem:invoke>"#.repeat(MAX_TOOL_CALLS)
        );
        assert!(matches!(
            bounded.push(ToolProtocol::AtemXmlV1, &too_many),
            ToolParseOutcome::Malformed(_)
        ));
    }

    #[test]
    fn impossible_argument_types_fail_before_a_provisional_call_can_be_published() {
        for suffix in ["[", "null", "\"text", "1"] {
            let mut stream = ToolCallStream::default();
            let input = format!(r#"<tool_call>{{"name":"lookup","arguments":{suffix}"#);
            assert!(
                matches!(
                    stream.push(ToolProtocol::TaggedJsonV1, &input),
                    ToolParseOutcome::Malformed(_)
                ),
                "{input}"
            );
            assert!(stream.fragments(ToolProtocol::TaggedJsonV1).is_err());
        }
    }

    #[test]
    fn duplicate_call_fields_and_payload_limits_are_terminal_before_fragment_publication() {
        for input in [
            r#"<tool_call>{"name":"a","name":"b","arguments":{}}</tool_call>"#,
            r#"<tool_call>{"name":"a","arguments":{},"arguments":{}}</tool_call>"#,
        ] {
            let mut stream = ToolCallStream::default();
            assert!(matches!(
                stream.push(ToolProtocol::TaggedJsonV1, input),
                ToolParseOutcome::Malformed(_)
            ));
            assert!(stream.fragments(ToolProtocol::TaggedJsonV1).is_err());
        }
        let mut stream = ToolCallStream::default();
        stream.push(
            ToolProtocol::TaggedJsonV1,
            &"x".repeat(MAX_TOOL_PAYLOAD_BYTES + 1),
        );
        assert!(stream.fragments(ToolProtocol::TaggedJsonV1).is_err());
    }
}
