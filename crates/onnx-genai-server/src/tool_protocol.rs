//! Server-owned adapters for declared tool-call wire protocols.
//!
//! `package.tool_protocol` is the portable declaration.  These adapters are one
//! runtime's implementation of that declaration; a package never names an
//! adapter, parser library, or model family.

use std::{collections::BTreeSet, path::Path};

use onnx_genai_engine::GenerateConstraint;
use onnx_genai_metadata::{
    MAX_TOOL_CALL_ID_BYTES, MAX_TOOL_CALLS, MAX_TOOL_NAME_BYTES, MAX_TOOL_PAYLOAD_BYTES,
    ToolProtocol as ParsedToolProtocol, ToolProtocolDeclaration, resolve_tool_protocol,
};

pub use onnx_genai_metadata::{ToolCallStream, ToolParseOutcome, ToolProtocolError};

use crate::types::{ChatCompletionRequest, ChatTool, ToolChoice, ToolChoiceMode};

/// How a protocol's forced tool choice changes normal response-format
/// constraints. This is adapter-owned because the tool-call envelope is part
/// of the protocol, not an OpenAI request's JSON representation.
#[derive(Debug)]
pub(crate) enum ToolOutputConstraint {
    /// No forced tool choice: retain the request's ordinary response constraint.
    Unchanged,
    /// A forced tool choice: use this protocol's constraint, if it has one.
    Set(Option<GenerateConstraint>),
}

/// Values a chat-template implementation receives after protocol rendering.
pub(crate) struct RenderedToolRequest {
    pub(crate) tools_json: Option<String>,
    pub(crate) fallback_prefix: String,
}

/// Server request-rendering policy for one shared exact parser.
///
/// The inner parser is transport-neutral and authoritative for all generated
/// output. This wrapper retains only OpenAI request rendering and post-parse
/// tool-choice validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolProtocol(ParsedToolProtocol);

impl ToolProtocol {
    pub fn declaration(&self) -> (&'static str, &'static str) {
        self.0.declaration()
    }

    pub(crate) fn render(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<RenderedToolRequest, ToolProtocolError> {
        match self.0 {
            ParsedToolProtocol::TaggedJsonV1 => render_tagged_json_request(request),
            ParsedToolProtocol::AtemXmlV1 => render_atem_xml_request(request),
        }
    }

    pub(crate) fn output_constraint(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ToolOutputConstraint, ToolProtocolError> {
        match self.0 {
            ParsedToolProtocol::TaggedJsonV1 => {
                let Some(schemas) = forced_tool_choice_schemas(request) else {
                    return Ok(ToolOutputConstraint::Unchanged);
                };
                let schema = if schemas.len() == 1 {
                    schemas.into_iter().next().expect("one schema")
                } else {
                    serde_json::json!({ "anyOf": schemas })
                };
                let schema = serde_json::to_string(&schema).map_err(|error| {
                    ToolProtocolError(format!(
                        "tagged-json@v1 cannot encode forced tool-choice schema: {error}"
                    ))
                })?;
                Ok(ToolOutputConstraint::Set(Some(GenerateConstraint::Lark(
                    format!(
                        "start: \"<tool_call>\\n\" tool \"\\n</tool_call>\"\ntool: %json {schema}\n"
                    ),
                ))))
            }
            ParsedToolProtocol::AtemXmlV1 => Ok(if forced_tool_choice_schemas(request).is_some() {
                ToolOutputConstraint::Set(None)
            } else {
                ToolOutputConstraint::Unchanged
            }),
        }
    }

    /// Convert a terminal parser outcome into a protocol-specific error at the
    /// boundary where the complete model output is known.
    pub fn output_error(
        &self,
        outcome: &ToolParseOutcome,
        boundary: &str,
    ) -> Option<ToolProtocolError> {
        self.0.output_error(outcome, boundary)
    }

    /// Reject only failures that are terminal while more chunks may arrive.
    pub fn incremental_output_error(
        &self,
        outcome: &ToolParseOutcome,
        boundary: &str,
    ) -> Option<ToolProtocolError> {
        self.0.incremental_output_error(outcome, boundary)
    }

    /// Enforce the request's tool-choice policy after parsing has reached its
    /// terminal boundary.
    pub fn validate_output(
        &self,
        request: &ChatCompletionRequest,
        outcome: &ToolParseOutcome,
        boundary: &str,
    ) -> Result<(), ToolProtocolError> {
        if let Some(error) = self.output_error(outcome, boundary) {
            return Err(error);
        }
        let (identity, version) = self.declaration();
        match (&request.tool_choice, outcome) {
            (Some(ToolChoice::Mode(ToolChoiceMode::Required)), ToolParseOutcome::NoCall) => {
                Err(ToolProtocolError(format!(
                    "declared tool protocol {identity}@{version} violated tool_choice required at the \
                 {boundary}: the model produced no tool call"
                )))
            }
            (Some(ToolChoice::Specific(choice)), ToolParseOutcome::NoCall) => {
                Err(ToolProtocolError(format!(
                    "declared tool protocol {identity}@{version} violated specific tool_choice \
                     {:?} at the {boundary}: the model produced no tool call",
                    choice.function.name
                )))
            }
            (Some(ToolChoice::Specific(choice)), ToolParseOutcome::Complete(calls)) => {
                let observed = calls
                    .iter()
                    .map(|call| call.name.as_str())
                    .collect::<BTreeSet<_>>();
                if observed.len() == 1 && observed.contains(choice.function.name.as_str()) {
                    Ok(())
                } else {
                    Err(ToolProtocolError(format!(
                        "declared tool protocol {identity}@{version} violated specific tool_choice \
                         {:?} at the {boundary}: parsed function name(s) were {:?}",
                        choice.function.name, observed
                    )))
                }
            }
            _ => Ok(()),
        }
    }

    pub fn parse(&self, input: &str) -> ToolParseOutcome {
        self.0.parse(input)
    }

    /// Resolve a declaration for callers that parse declared model output
    /// outside a loaded server.  Model admission uses [`resolve`] so its error
    /// includes the actual metadata path.
    pub fn from_declaration(
        declaration: &ToolProtocolDeclaration,
    ) -> Result<Self, ToolProtocolError> {
        resolve(
            declaration,
            Path::new("the supplied tool protocol declaration"),
        )
    }

    /// The shared, transport-neutral parser selected by this declaration.
    pub fn parser(&self) -> ParsedToolProtocol {
        self.0
    }
}

/// Resolve the exact declaration supplied by metadata, without fallback or
/// parser probing.  Keeping this registry at the adapter seam means another
/// runtime can implement the same portable declaration differently.
pub(crate) fn resolve(
    declaration: &ToolProtocolDeclaration,
    metadata_path: &Path,
) -> Result<ToolProtocol, ToolProtocolError> {
    resolve_tool_protocol(declaration, metadata_path).map(ToolProtocol)
}

pub(crate) fn validate_request(
    request: &ChatCompletionRequest,
    protocol: Option<&ToolProtocol>,
    declaration_path: Option<&Path>,
) -> Result<(), ToolProtocolError> {
    if !request.has_tool_context() {
        return Ok(());
    }
    let Some(protocol) = protocol else {
        let path = declaration_path
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "this package's metadata".to_string());
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference: {path} does not declare package.tool_protocol. \
             Add one exact protocol identity and version; a server never guesses from model output."
        )));
    };
    let (identity, version) = protocol.declaration();
    let rendered = protocol.render(request)?;
    if rendered
        .tools_json
        .as_ref()
        .is_some_and(|json| json.len() > MAX_TOOL_PAYLOAD_BYTES)
    {
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: rendered tools exceed the {MAX_TOOL_PAYLOAD_BYTES}-byte limit"
        )));
    }
    validate_untrusted_request_values(request, identity, version)
}

fn validate_untrusted_request_values(
    request: &ChatCompletionRequest,
    identity: &str,
    version: &str,
) -> Result<(), ToolProtocolError> {
    let tools = request.tools.as_deref().unwrap_or_default();
    if tools.len() > MAX_TOOL_CALLS {
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: at most {MAX_TOOL_CALLS} tools may be offered"
        )));
    }
    for (index, tool) in tools.iter().enumerate() {
        validate_tool(tool, index, identity, version)?;
    }
    for (index, message) in request.messages.iter().enumerate() {
        if message
            .tool_call_id
            .as_ref()
            .is_some_and(|id| id.is_empty() || id.len() > MAX_TOOL_CALL_ID_BYTES)
        {
            return Err(ToolProtocolError(format!(
                "tool request rejected before inference for {identity}@{version}: messages[{index}].tool_call_id must contain 1 to {MAX_TOOL_CALL_ID_BYTES} bytes"
            )));
        }
        if message
            .content
            .as_ref()
            .is_some_and(|content| content.text().len() > MAX_TOOL_PAYLOAD_BYTES)
        {
            return Err(ToolProtocolError(format!(
                "tool request rejected before inference for {identity}@{version}: messages[{index}].content exceeds the {MAX_TOOL_PAYLOAD_BYTES}-byte limit"
            )));
        }
        if message
            .name
            .as_ref()
            .is_some_and(|name| name.is_empty() || name.len() > MAX_TOOL_NAME_BYTES)
        {
            return Err(ToolProtocolError(format!(
                "tool request rejected before inference for {identity}@{version}: messages[{index}].name must contain 1 to {MAX_TOOL_NAME_BYTES} bytes"
            )));
        }
        if let Some(calls) = &message.tool_calls {
            if calls.len() > MAX_TOOL_CALLS {
                return Err(ToolProtocolError(format!(
                    "tool request rejected before inference for {identity}@{version}: messages[{index}].tool_calls may contain at most {MAX_TOOL_CALLS} calls"
                )));
            }
            for (call_index, call) in calls.iter().enumerate() {
                if call.id.is_empty()
                    || call.id.len() > MAX_TOOL_CALL_ID_BYTES
                    || call.function.name.is_empty()
                    || call.function.name.len() > MAX_TOOL_NAME_BYTES
                    || call.function.arguments.len() > MAX_TOOL_PAYLOAD_BYTES
                {
                    return Err(ToolProtocolError(format!(
                        "tool request rejected before inference for {identity}@{version}: messages[{index}].tool_calls[{call_index}] has an empty or oversized id, name, or arguments value"
                    )));
                }
            }
        }
    }
    if let Some(ToolChoice::Specific(choice)) = &request.tool_choice
        && (choice.function.name.is_empty() || choice.function.name.len() > MAX_TOOL_NAME_BYTES)
    {
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: tool_choice.function.name must contain 1 to {MAX_TOOL_NAME_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_tool(
    tool: &ChatTool,
    index: usize,
    identity: &str,
    version: &str,
) -> Result<(), ToolProtocolError> {
    if tool.kind != "function" {
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: tools[{index}].type must be \"function\""
        )));
    }
    if tool.function.name.is_empty() || tool.function.name.len() > MAX_TOOL_NAME_BYTES {
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: tools[{index}].function.name must contain 1 to {MAX_TOOL_NAME_BYTES} bytes"
        )));
    }
    let encoded = serde_json::to_vec(tool).map_err(|error| {
        ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: tools[{index}] cannot be rendered as JSON: {error}"
        ))
    })?;
    if encoded.len() > MAX_TOOL_PAYLOAD_BYTES {
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: tools[{index}] exceeds the {MAX_TOOL_PAYLOAD_BYTES}-byte limit"
        )));
    }
    Ok(())
}

fn render_tagged_json_request(
    request: &ChatCompletionRequest,
) -> Result<RenderedToolRequest, ToolProtocolError> {
    render_request(request, "<|tools|>", "<|tool_choice|>")
}

fn render_atem_xml_request(
    request: &ChatCompletionRequest,
) -> Result<RenderedToolRequest, ToolProtocolError> {
    // ATEM uses its own tools marker but retains the v1 common placement for
    // the caller's OpenAI tool_choice value.
    render_request(request, "<atem:tools>", "<|tool_choice|>")
}

fn render_request(
    request: &ChatCompletionRequest,
    tools_marker: &str,
    choice_marker: &str,
) -> Result<RenderedToolRequest, ToolProtocolError> {
    let tools = tools_offered_to_model(request);
    let tools_json = tools
        .map(serde_json::to_string)
        .transpose()
        .map_err(|error| {
            ToolProtocolError(format!("cannot render untrusted tool definitions: {error}"))
        })?;
    let mut fallback_prefix = String::new();
    if let Some(tools_json) = &tools_json {
        fallback_prefix.push_str(tools_marker);
        fallback_prefix.push('\n');
        fallback_prefix.push_str(tools_json);
        fallback_prefix.push('\n');
    }
    if let Some(choice) = &request.tool_choice {
        fallback_prefix.push_str(choice_marker);
        fallback_prefix.push('\n');
        fallback_prefix.push_str(&tool_choice_prompt(choice));
        fallback_prefix.push('\n');
    }

    Ok(RenderedToolRequest {
        tools_json,
        fallback_prefix,
    })
}

fn forced_tool_choice_schemas(request: &ChatCompletionRequest) -> Option<Vec<serde_json::Value>> {
    let tools = request
        .tools
        .as_ref()?
        .iter()
        .filter(|tool| tool.kind == "function");
    let selected = match request.tool_choice.as_ref()? {
        ToolChoice::Mode(ToolChoiceMode::Required) => tools.collect::<Vec<_>>(),
        ToolChoice::Specific(choice) if choice.kind == "function" => tools
            .filter(|tool| tool.function.name == choice.function.name)
            .collect::<Vec<_>>(),
        ToolChoice::Mode(ToolChoiceMode::Auto | ToolChoiceMode::None) | ToolChoice::Specific(_) => {
            Vec::new()
        }
    };
    let schemas = selected
        .into_iter()
        .map(tool_call_schema_for_tool)
        .collect::<Vec<_>>();
    (!schemas.is_empty()).then_some(schemas)
}

fn tool_call_schema_for_tool(tool: &ChatTool) -> serde_json::Value {
    let arguments_schema = tool
        .function
        .parameters
        .clone()
        .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
    serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "enum": [tool.function.name.clone()] },
            "arguments": arguments_schema
        },
        "required": ["name", "arguments"],
        "additionalProperties": false
    })
}

fn tools_offered_to_model(request: &ChatCompletionRequest) -> Option<&Vec<ChatTool>> {
    if matches!(
        request.tool_choice,
        Some(ToolChoice::Mode(ToolChoiceMode::None))
    ) {
        None
    } else {
        request.tools.as_ref().filter(|tools| !tools.is_empty())
    }
}

fn tool_choice_prompt(choice: &ToolChoice) -> String {
    match choice {
        ToolChoice::Mode(ToolChoiceMode::Auto) => "auto".to_string(),
        ToolChoice::Mode(ToolChoiceMode::None) => "none".to_string(),
        ToolChoice::Mode(ToolChoiceMode::Required) => "required".to_string(),
        ToolChoice::Specific(choice) => format!("function: {}", choice.function.name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protocol(identity: &str) -> ToolProtocol {
        resolve(
            &ToolProtocolDeclaration {
                identity: identity.to_string(),
                version: "v1".to_string(),
            },
            Path::new("fixtures/inference_metadata.yaml"),
        )
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
                stream.push(resolved.parser(), &input[..split]);
                let actual = stream.push(resolved.parser(), &input[split..]);
                assert_eq!(actual, expected, "{identity} split at {split}");
            }
        }
    }

    #[test]
    fn two_declared_protocols_share_the_same_stream_dispatch() {
        for (identity, input) in [
            (
                "tagged-json",
                "<tool_call>{\"id\":\"one\",\"name\":\"first\",\"arguments\":{}}</tool_call>\n<tool_call>{\"name\":\"second\",\"parameters\":{\"n\":2}}</tool_call>",
            ),
            (
                "atem-xml",
                "<atem:invoke name=\"first\"><atem:parameter name=\"text\">\"a &lt; b\"</atem:parameter></atem:invoke>",
            ),
        ] {
            let resolved = protocol(identity);
            let mut stream = ToolCallStream::default();
            let outcome = stream.push(resolved.parser(), input);
            let ToolParseOutcome::Complete(calls) = outcome else {
                panic!("expected complete {identity} call");
            };
            assert!(!calls.is_empty());
        }
    }

    #[test]
    fn incomplete_and_malformed_envelopes_are_distinct() {
        let resolved = protocol("tagged-json");
        assert_eq!(
            resolved.parse("<tool_call>{\"name\":\"missing-close\"}"),
            ToolParseOutcome::Incomplete
        );
        assert!(matches!(
            resolved.parse("<tool_call>{\"name\":}</tool_call>"),
            ToolParseOutcome::Malformed(_)
        ));
        let atem = protocol("atem-xml");
        assert!(matches!(
            atem.parse(
                "<atem:invoke name=\"tool\">unexpected<atem:parameter name=\"x\">1</atem:parameter></atem:invoke>"
            ),
            ToolParseOutcome::Malformed(_)
        ));
    }

    #[test]
    fn atem_xml_requires_the_complete_output_to_be_an_envelope_sequence() {
        let resolved = protocol("atem-xml");
        let call = r#"<atem:invoke name="read"></atem:invoke>"#;
        for input in [
            format!("junk{call}"),
            format!("{call}junk"),
            format!("junk{call}junk"),
        ] {
            assert!(
                matches!(resolved.parse(&input), ToolParseOutcome::Malformed(_)),
                "{input}"
            );
        }

        let valid = format!(
            " \n{call}\t<atem:invoke name=\"write\"><atem:parameter name=\"path\">\"src/lib.rs\"</atem:parameter></atem:invoke>\r\n"
        );
        let ToolParseOutcome::Complete(calls) = resolved.parse(&valid) else {
            panic!("surrounding whitespace and adjacent calls must be accepted");
        };
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[1].name, "write");
    }

    #[test]
    fn atem_xml_partial_openers_remain_incomplete_but_surrounding_text_is_terminal() {
        let resolved = protocol("atem-xml");
        let call = r#"<atem:invoke name="read"></atem:invoke>"#;
        let inputs = [
            "<atem:inv".to_string(),
            " \n<atem:inv".to_string(),
            format!("{call}\n<atem:inv"),
        ];
        for input in inputs {
            assert_eq!(
                resolved.parse(&input),
                ToolParseOutcome::Incomplete,
                "{input}"
            );
        }

        let mut stream = ToolCallStream::default();
        assert_eq!(
            stream.push(resolved.parser(), &format!("{call}junk")),
            ToolParseOutcome::Malformed(
                "atem-xml@v1 has non-whitespace trailing text after an invoke envelope: \"junk\""
                    .to_string()
            )
        );
        assert!(matches!(
            stream.push(resolved.parser(), "<atem:invoke name=\"ignored\">"),
            ToolParseOutcome::Malformed(message) if message.contains("trailing text")
        ));
    }

    #[test]
    fn tagged_json_requires_an_explicit_object_argument_field() {
        let resolved = protocol("tagged-json");
        for input in [
            r#"<tool_call>{"name":"read"}</tool_call>"#,
            r#"<tool_call>{"name":"read","arguments":null}</tool_call>"#,
            r#"<tool_call>{"name":"read","arguments":[]}</tool_call>"#,
            r#"<tool_call>{"name":"read","parameters":"path"}</tool_call>"#,
        ] {
            assert!(
                matches!(resolved.parse(input), ToolParseOutcome::Malformed(_)),
                "{input}"
            );
        }
        for input in [
            r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#,
            r#"<tool_call>{"name":"read","arguments":{"path":"src/lib.rs"}}</tool_call>"#,
            r#"<tool_call>{"name":"read","parameters":{}}</tool_call>"#,
        ] {
            assert!(
                matches!(resolved.parse(input), ToolParseOutcome::Complete(_)),
                "{input}"
            );
        }
    }

    #[test]
    fn incomplete_chunks_are_nonterminal_until_stream_finish() {
        let resolved = protocol("tagged-json");
        let mut stream = ToolCallStream::default();
        let partial = stream.push(
            resolved.parser(),
            r#"<tool_call>{"name":"read","arguments":{"path":"src/"#,
        );
        assert_eq!(partial, ToolParseOutcome::Incomplete);
        assert!(
            resolved
                .incremental_output_error(&partial, "SSE chunk ingestion")
                .is_none()
        );
        let complete = stream.push(resolved.parser(), r#"lib.rs"}}</tool_call>"#);
        assert!(matches!(complete, ToolParseOutcome::Complete(_)));
        assert!(
            resolved
                .incremental_output_error(&complete, "SSE chunk ingestion")
                .is_none()
        );
        assert_eq!(stream.finish(resolved.parser()), complete);
    }

    #[test]
    fn sse_boundary_turns_incomplete_envelopes_into_typed_protocol_errors() {
        for (identity, chunks) in [
            ("tagged-json", ["<tool_call>{\"name\":", "\"read\"}"]),
            (
                "atem-xml",
                [
                    "<atem:invoke name=\"read\"><atem:parameter",
                    " name=\"path\">src/lib.rs",
                ],
            ),
        ] {
            let resolved = protocol(identity);
            let mut stream = ToolCallStream::default();
            for chunk in chunks {
                assert!(matches!(
                    stream.push(resolved.parser(), chunk),
                    ToolParseOutcome::Incomplete
                ));
            }
            let outcome = stream.finish(resolved.parser());
            let error = resolved
                .output_error(&outcome, "SSE streaming boundary")
                .expect("an incomplete stream must fail closed")
                .to_string();
            assert!(error.contains(&format!("{identity}@v1")), "{error}");
            assert!(error.contains("incomplete"), "{error}");
            assert!(error.contains("SSE streaming boundary"), "{error}");
        }
    }

    #[test]
    fn unknown_declaration_names_its_metadata_path_and_exact_pair() {
        let error = resolve(
            &ToolProtocolDeclaration {
                identity: "unknown".to_string(),
                version: "v9".to_string(),
            },
            Path::new("fixtures/inference_metadata.yaml"),
        )
        .expect_err("unknown protocol must fail closed")
        .to_string();
        assert!(
            error.contains("fixtures/inference_metadata.yaml"),
            "{error}"
        );
        assert!(error.contains("\"unknown\""), "{error}");
        assert!(error.contains("\"v9\""), "{error}");
    }

    #[test]
    fn tool_request_without_a_declaration_is_rejected_before_inference() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "fixture",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {"name": "weather"}}]
        }))
        .unwrap();
        let error = validate_request(
            &request,
            None,
            Some(Path::new("fixtures/inference_metadata.yaml")),
        )
        .expect_err("tools require a declaration")
        .to_string();
        assert!(error.contains("before inference"), "{error}");
        assert!(
            error.contains("fixtures/inference_metadata.yaml"),
            "{error}"
        );
        assert!(error.contains("package.tool_protocol"), "{error}");
    }

    #[test]
    fn request_renderer_bounds_untrusted_descriptions_and_results() {
        let protocol = protocol("tagged-json");
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "fixture",
            "messages": [{
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "x".repeat(MAX_TOOL_PAYLOAD_BYTES + 1)
            }],
            "tools": [{"type": "function", "function": {
                "name": "weather",
                "description": "a description",
                "parameters": {"type": "object"}
            }}]
        }))
        .unwrap();
        let error = validate_request(&request, Some(&protocol), None)
            .expect_err("oversized tool result must be rejected")
            .to_string();
        assert!(error.contains("messages[0].content"), "{error}");

        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "fixture",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {"name": "weather"}}]
        }))
        .unwrap();
        let rendered = protocol.render(&request).expect("tool request renders");
        assert!(rendered.fallback_prefix.contains("<|tools|>"));
        assert!(
            rendered
                .tools_json
                .as_deref()
                .is_some_and(|json| json.contains("weather"))
        );
    }

    #[test]
    fn adapters_own_prompt_markers_and_forced_output_constraints() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "fixture",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {
                "name": "weather", "parameters": {"type": "object"}
            }}],
            "tool_choice": "required",
            "response_format": {"type": "json_object"}
        }))
        .unwrap();

        let tagged = protocol("tagged-json");
        let rendered = tagged.render(&request).expect("tagged request renders");
        assert!(rendered.fallback_prefix.contains("<|tools|>"));
        assert!(rendered.fallback_prefix.contains("<|tool_choice|>"));
        assert!(matches!(
            tagged.output_constraint(&request).unwrap(),
            ToolOutputConstraint::Set(Some(GenerateConstraint::Lark(_)))
        ));

        let atem = protocol("atem-xml");
        let rendered = atem.render(&request).expect("ATEM request renders");
        assert!(rendered.fallback_prefix.contains("<atem:tools>"));
        assert!(rendered.fallback_prefix.contains("<|tool_choice|>"));
        assert!(matches!(
            atem.output_constraint(&request).unwrap(),
            ToolOutputConstraint::Set(None)
        ));
    }

    #[test]
    fn model_output_and_incremental_stream_are_bounded() {
        let oversized = "x".repeat(MAX_TOOL_PAYLOAD_BYTES + 1);
        for identity in ["tagged-json", "atem-xml"] {
            let resolved = protocol(identity);
            assert!(matches!(
                resolved.parse(&oversized),
                ToolParseOutcome::Malformed(message) if message.contains("byte protocol limit")
            ));

            let mut stream = ToolCallStream::default();
            let outcome = stream.push(resolved.parser(), &oversized);
            assert!(matches!(
                outcome,
                ToolParseOutcome::Malformed(ref message) if message.contains("byte protocol limit")
            ));
            let error = resolved
                .output_error(&outcome, "SSE streaming boundary")
                .expect("terminal stream failure has typed protocol error")
                .to_string();
            assert!(error.contains(&format!("{identity}@v1")), "{error}");
            assert!(error.contains("SSE streaming boundary"), "{error}");
            assert!(matches!(
                stream.push(resolved.parser(), "<tool_call>{}</tool_call>"),
                ToolParseOutcome::Malformed(message) if message.contains("byte protocol limit")
            ));
        }
    }
}
