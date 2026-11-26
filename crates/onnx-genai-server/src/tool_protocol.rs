//! Server-owned adapters for declared tool-call wire protocols.
//!
//! `package.tool_protocol` is the portable declaration.  These adapters are one
//! runtime's implementation of that declaration; a package never names an
//! adapter, parser library, or model family.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use onnx_genai_engine::{GenerateConstraint, ToolCallPolicy};
use onnx_genai_metadata::{
    MAX_TOOL_CALL_ID_BYTES, MAX_TOOL_CALLS, MAX_TOOL_NAME_BYTES, MAX_TOOL_PAYLOAD_BYTES,
    ToolProtocol as ParsedToolProtocol, ToolProtocolDeclaration, resolve_tool_protocol,
};

pub use onnx_genai_metadata::ToolProtocolError;

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

/// Server request-rendering policy for one declared protocol.
///
/// Generated output is parsed and policy-checked inside the engine transaction;
/// this wrapper owns only OpenAI request rendering and constraint selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolProtocol(ParsedToolProtocol);

impl ToolProtocol {
    pub(crate) fn parsed(&self) -> ParsedToolProtocol {
        self.0
    }

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
            ParsedToolProtocol::TaggedJsonV1 | ParsedToolProtocol::AtemXmlV1 => {
                let Some(mut schemas) = forced_tool_choice_schemas(request)? else {
                    return Ok(ToolOutputConstraint::Unchanged);
                };
                if self.0 == ParsedToolProtocol::AtemXmlV1 {
                    for (tool, schema) in &mut schemas {
                        if tool.function.name.is_empty()
                            || tool.function.name.len() > onnx_genai_metadata::MAX_TOOL_NAME_BYTES
                        {
                            return Err(ToolProtocolError("canonical XML generation requires nonempty function names within the UTF-8 byte limit".into()));
                        }
                        // Preserve the actual standalone root, including its
                        // URI/dialect and existing compiler options. This owned
                        // option only restricts root keys to the inbound XML
                        // attribute encoding capability; nested JSON is intact.
                        if schema == &serde_json::Value::Bool(false) {
                            continue;
                        }
                        let root = schema.as_object_mut().ok_or_else(|| {
                            ToolProtocolError("XML argument schema must have an object root".into())
                        })?;
                        let options = root
                            .entry("x-guidance")
                            .or_insert_with(|| serde_json::json!({}));
                        let options = options.as_object_mut().ok_or_else(|| {
                            ToolProtocolError(
                                "XML argument schema x-guidance must be an options object".into(),
                            )
                        })?;
                        for (key, standard) in [("item_separator", ","), ("key_separator", ":")] {
                            if options
                                .get(key)
                                .is_some_and(|value| value.as_str() != Some(standard))
                            {
                                return Err(ToolProtocolError(format!(
                                    "canonical XML generation requires standard JSON {key}; remove its x-guidance lexical override"
                                )));
                            }
                        }
                        if options
                            .get("whitespace_pattern")
                            .is_some_and(|value| !value.is_null())
                        {
                            return Err(ToolProtocolError("canonical XML generation requires the default JSON whitespace language; remove x-guidance.whitespace_pattern".into()));
                        }
                        options.insert("root_object_key_policy".into(), serde_json::json!({"min":1,"max":onnx_genai_metadata::MAX_TOOL_NAME_BYTES}));
                    }
                }
                // JSON anyOf prunes backend-proven impossible branches. Retain
                // that behavior when each alternative owns a standalone root.
                // A single selected tool keeps its direct compilation path.
                if schemas.len() > 1 || self.0 == ParsedToolProtocol::AtemXmlV1 {
                    let mut viable = Vec::with_capacity(schemas.len());
                    for (tool, schema) in schemas {
                        let satisfiable =
                            onnx_genai_engine::byte_grammar::json_schema_is_satisfiable(&schema)
                                .map_err(|error| {
                                    ToolProtocolError(format!(
                                        "function {:?} parameters: {error:#}",
                                        tool.function.name
                                    ))
                                })?;
                        if satisfiable {
                            viable.push((tool, schema));
                        }
                    }
                    if viable.is_empty() {
                        let reason = if self.0 == ParsedToolProtocol::AtemXmlV1 {
                            "all required XML tool schemas have no instances within the named-parameter generation key-encoding capability (declared nonempty parameter names within the UTF-8 byte limit); revise tools[*].function.parameters or select a usable function"
                        } else {
                            "all required tool parameter schemas are unsatisfiable; revise tools[*].function.parameters or select a usable function"
                        };
                        return Err(ToolProtocolError(reason.into()));
                    }
                    schemas = viable;
                }
                // Every parameter document is compiled as its own %json root.
                // Its dialect, base URI, pointers, anchors and registry remain
                // exactly those of standalone parameter admission.
                let names = (0..schemas.len())
                    .map(|index| format!("tool_{index}"))
                    .collect::<Vec<_>>();
                let mut grammar = format!(
                    "start: \"<tool_call>\\n\" ws tool ws \"\\n</tool_call>\"\ntool: {}\n",
                    names.join(" | ")
                );
                grammar.push_str(concat!(
                    "ws: /[ \\t\\r\\n]+/?\n",
                    "name_key: %json {\"const\":\"name\"}\n",
                    "arguments_key: %json {\"const\":\"arguments\"}\n",
                    "id_key: %json {\"const\":\"id\"}\n",
                    "id_field: id_key ws \":\" ws call_id\n",
                ));
                grammar.push_str(&format!(
                    "call_id: %json {}\n",
                    serde_json::json!({"type":"string","minLength":1,"maxLength":onnx_genai_metadata::MAX_TOOL_CALL_ID_BYTES})
                ));
                for (rule, (tool, schema)) in names.iter().zip(schemas) {
                    let fields = [
                        format!("{rule}_name"),
                        "id_field".into(),
                        format!("{rule}_arguments"),
                    ];
                    let mut alternatives = vec![
                        format!("{} ws \",\" ws {}", fields[0], fields[2]),
                        format!("{} ws \",\" ws {}", fields[2], fields[0]),
                    ];
                    for first in 0..fields.len() {
                        for second in 0..fields.len() {
                            for third in 0..fields.len() {
                                if first != second && first != third && second != third {
                                    alternatives.push(format!(
                                        "{} ws \",\" ws {} ws \",\" ws {}",
                                        fields[first], fields[second], fields[third]
                                    ));
                                }
                            }
                        }
                    }
                    let schema = serde_json::to_string(&schema).map_err(|error| {
                        ToolProtocolError(format!(
                            "tagged-json@v1 cannot encode function {:?} parameter schema: {error}",
                            tool.function.name
                        ))
                    })?;
                    grammar.push_str(&format!(
                        "{rule}: \"{{\" ws ({}) ws \"}}\"\n{rule}_name: name_key ws \":\" ws {rule}_name_value\n{rule}_arguments: arguments_key ws \":\" ws {rule}_arguments_value\n{rule}_name_value: %json {}\n{rule}_arguments_value: %json {schema}\n",
                        alternatives.join(" | "), serde_json::json!({"enum":[tool.function.name]})
                    ));
                }
                let constraint = if self.0 == ParsedToolProtocol::AtemXmlV1 {
                    GenerateConstraint::XmlToolCall {
                        grammar,
                        prefix: String::new(),
                    }
                } else {
                    GenerateConstraint::Lark {
                        grammar,
                        prefix: String::new(),
                    }
                };
                Ok(ToolOutputConstraint::Set(Some(constraint)))
            }
        }
    }

    /// Resolve a declaration for request rendering outside a loaded server.
    /// Model admission uses [`resolve`] so its error includes the actual
    /// metadata path.
    pub fn from_declaration(
        declaration: &ToolProtocolDeclaration,
    ) -> Result<Self, ToolProtocolError> {
        resolve(
            declaration,
            Path::new("the supplied tool protocol declaration"),
        )
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

/// Responses and realtime share the same flattened function-tool vocabulary.
/// Lower it once into the canonical chat tool transaction input.
pub(crate) fn responses_tools(
    tools: serde_json::Value,
) -> Result<serde_json::Value, ToolProtocolError> {
    use serde_json::{Value, json};
    let Value::Array(tools) = tools else {
        return Err(ToolProtocolError("tools must be an array".into()));
    };
    let mut lowered = Vec::with_capacity(tools.len());
    for tool in tools {
        let Value::Object(mut object) = tool else {
            return Err(ToolProtocolError(
                "each Responses tool must be an object".into(),
            ));
        };
        if object.remove("type") != Some(json!("function")) {
            return Err(ToolProtocolError(
                "Responses tools must be type=function; hosted tools are not implemented".into(),
            ));
        }
        // The canonical tool transaction owns strict argument admission.
        object.remove("strict");
        lowered.push(json!({"type":"function","function":object}));
    }
    Ok(json!(lowered))
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

/// Translate OpenAI request vocabulary into the engine's transport-neutral
/// observation policy for this turn.
pub(crate) fn request_policy(request: &ChatCompletionRequest) -> ToolCallPolicy {
    let has_tools = request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty());
    if !has_tools
        || matches!(
            request.tool_choice,
            Some(ToolChoice::Mode(ToolChoiceMode::None))
        )
    {
        return ToolCallPolicy::Disabled;
    }
    match &request.tool_choice {
        Some(ToolChoice::Mode(ToolChoiceMode::Required)) => ToolCallPolicy::Required,
        Some(ToolChoice::Specific(choice)) => ToolCallPolicy::Specific {
            function: choice.function.name.clone(),
        },
        Some(ToolChoice::Mode(ToolChoiceMode::Auto)) | None => ToolCallPolicy::Auto,
        Some(ToolChoice::Mode(ToolChoiceMode::None)) => ToolCallPolicy::Disabled,
    }
}

fn validate_untrusted_request_values(
    request: &ChatCompletionRequest,
    identity: &str,
    version: &str,
) -> Result<(), ToolProtocolError> {
    let tools = request.tools.as_deref().unwrap_or_default();
    let mut offered_tool_names = BTreeSet::new();
    if tools.len() > MAX_TOOL_CALLS {
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: at most {MAX_TOOL_CALLS} tools may be offered"
        )));
    }
    for (index, tool) in tools.iter().enumerate() {
        validate_tool(tool, index, identity, version)?;
        if !offered_tool_names.insert(tool.function.name.as_str()) {
            return Err(ToolProtocolError(format!(
                "tool request rejected before inference for {identity}@{version}: tools[{index}].function.name '{}' duplicates an earlier offered tool name",
                tool.function.name
            )));
        }
    }
    validate_tool_call_history(request, identity, version, &offered_tool_names)?;
    if let Some(ToolChoice::Specific(choice)) = &request.tool_choice
        && (choice.function.name.is_empty() || choice.function.name.len() > MAX_TOOL_NAME_BYTES)
    {
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: tool_choice.function.name must contain 1 to {MAX_TOOL_NAME_BYTES} bytes"
        )));
    }
    Ok(())
}

/// Validate the caller-owned call/result history before a turn can reach the
/// engine. A result is associated by its typed call ID, never by its position
/// beside other calls or results.
fn validate_tool_call_history(
    request: &ChatCompletionRequest,
    identity: &str,
    version: &str,
    offered_tool_names: &BTreeSet<&str>,
) -> Result<(), ToolProtocolError> {
    #[derive(Clone)]
    struct PendingCall {
        name: String,
        message_index: usize,
        call_index: usize,
    }

    let mut issued = BTreeMap::<String, (usize, usize)>::new();
    let mut pending = BTreeMap::<String, PendingCall>::new();
    let mut received = BTreeMap::<String, usize>::new();

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

        match message.role.as_str() {
            "assistant" => {
                if message.tool_call_id.is_some() {
                    return Err(ToolProtocolError(format!(
                        "tool request rejected before inference for {identity}@{version}: messages[{index}].tool_call_id is only valid on a tool result"
                    )));
                }
                let Some(calls) = &message.tool_calls else {
                    continue;
                };
                if calls.is_empty() || calls.len() > MAX_TOOL_CALLS {
                    return Err(ToolProtocolError(format!(
                        "tool request rejected before inference for {identity}@{version}: messages[{index}].tool_calls must contain 1 to {MAX_TOOL_CALLS} calls"
                    )));
                }
                for (call_index, call) in calls.iter().enumerate() {
                    validate_history_call(
                        call,
                        index,
                        call_index,
                        identity,
                        version,
                        offered_tool_names,
                    )?;
                    if let Some((first_message, first_call)) =
                        issued.insert(call.id.clone(), (index, call_index))
                    {
                        return Err(ToolProtocolError(format!(
                            "tool request rejected before inference for {identity}@{version}: messages[{index}].tool_calls[{call_index}].id '{}' duplicates messages[{first_message}].tool_calls[{first_call}].id",
                            call.id
                        )));
                    }
                    pending.insert(
                        call.id.clone(),
                        PendingCall {
                            name: call.function.name.clone(),
                            message_index: index,
                            call_index,
                        },
                    );
                }
            }
            "tool" => {
                if message.tool_calls.is_some() {
                    return Err(ToolProtocolError(format!(
                        "tool request rejected before inference for {identity}@{version}: messages[{index}].tool_calls belongs on an assistant message, not a tool result"
                    )));
                }
                let Some(call_id) = message.tool_call_id.as_deref() else {
                    return Err(ToolProtocolError(format!(
                        "tool request rejected before inference for {identity}@{version}: messages[{index}] is a tool result and requires tool_call_id"
                    )));
                };
                if !matches!(
                    message.content.as_ref(),
                    Some(crate::types::ChatMessageContent::Text(_))
                ) {
                    return Err(ToolProtocolError(format!(
                        "tool request rejected before inference for {identity}@{version}: messages[{index}].content for tool_call_id '{call_id}' must be a text string"
                    )));
                }
                let Some(call) = pending.remove(call_id) else {
                    let reason = if let Some(result_index) = received.get(call_id) {
                        format!(
                            "duplicates the result already supplied at messages[{result_index}]"
                        )
                    } else if issued.contains_key(call_id) {
                        "does not answer an outstanding call".to_string()
                    } else {
                        "does not match any assistant tool call".to_string()
                    };
                    return Err(ToolProtocolError(format!(
                        "tool request rejected before inference for {identity}@{version}: messages[{index}].tool_call_id '{call_id}' {reason}"
                    )));
                };
                if let Some(name) = message.name.as_deref()
                    && name != call.name
                {
                    return Err(ToolProtocolError(format!(
                        "tool request rejected before inference for {identity}@{version}: messages[{index}].name '{name}' does not match tool_call_id '{call_id}' function '{}'",
                        call.name
                    )));
                }
                received.insert(call_id.to_string(), index);
            }
            _ => {
                if message.tool_calls.is_some() {
                    return Err(ToolProtocolError(format!(
                        "tool request rejected before inference for {identity}@{version}: messages[{index}].tool_calls is only valid on an assistant message"
                    )));
                }
                if message.tool_call_id.is_some() {
                    return Err(ToolProtocolError(format!(
                        "tool request rejected before inference for {identity}@{version}: messages[{index}].tool_call_id is only valid on a tool result"
                    )));
                }
            }
        }
    }

    if !pending.is_empty() {
        let calls = pending
            .iter()
            .map(|(id, call)| {
                format!(
                    "'{id}' from messages[{}].tool_calls[{}]",
                    call.message_index, call.call_index
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: missing tool result(s) for {calls}"
        )));
    }
    Ok(())
}

fn validate_history_call(
    call: &crate::types::ChatMessageToolCall,
    message_index: usize,
    call_index: usize,
    identity: &str,
    version: &str,
    offered_tool_names: &BTreeSet<&str>,
) -> Result<(), ToolProtocolError> {
    let field = format!("messages[{message_index}].tool_calls[{call_index}]");
    if call.kind != "function" {
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: {field}.type must be \"function\""
        )));
    }
    if call.id.is_empty()
        || call.id.len() > MAX_TOOL_CALL_ID_BYTES
        || call.function.name.is_empty()
        || call.function.name.len() > MAX_TOOL_NAME_BYTES
        || call.function.arguments.len() > MAX_TOOL_PAYLOAD_BYTES
    {
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: {field} has an empty or oversized id, name, or arguments value"
        )));
    }
    if !offered_tool_names.is_empty() && !offered_tool_names.contains(call.function.name.as_str()) {
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: {field}.function.name '{}' was not offered in tools",
            call.function.name
        )));
    }
    if !matches!(
        serde_json::from_str::<serde_json::Value>(&call.function.arguments),
        Ok(serde_json::Value::Object(_))
    ) {
        return Err(ToolProtocolError(format!(
            "tool request rejected before inference for {identity}@{version}: {field}.function.arguments must be a JSON object"
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

/// Both registered model protocols require object arguments. Constrain that
/// domain without relocating the supplied schema or its root definitions/refs.
pub(crate) fn object_argument_schema(
    schema: Option<&serde_json::Value>,
) -> Result<serde_json::Value, ToolProtocolError> {
    use serde_json::{Value, json};
    let domain = json!({"type":"object"});
    match schema {
        None | Some(Value::Null | Value::Bool(true)) => Ok(domain),
        Some(Value::Bool(false)) => Ok(Value::Bool(false)),
        Some(Value::Object(original)) => {
            let mut object = original.clone();
            let needs_domain = !matches!(object.get("type"), Some(Value::String(kind)) if kind == "object");
            match object.get_mut("allOf") {
                Some(Value::Array(terms)) => { if needs_domain { terms.push(domain); } }
                Some(_) => return Err(ToolProtocolError("tool parameters.allOf must be an array of schemas; supply a valid JSON Schema object or boolean".into())),
                None => { if needs_domain { object.insert("allOf".into(), json!([domain])); } }
            }
            Ok(Value::Object(object))
        }
        Some(_) => Err(ToolProtocolError("tool parameters must be a JSON Schema object or boolean; omit it (or use null) for unconstrained object arguments".into())),
    }
}

fn forced_tool_choice_schemas(
    request: &ChatCompletionRequest,
) -> Result<Option<Vec<(&ChatTool, serde_json::Value)>>, ToolProtocolError> {
    let (Some(tools), Some(choice)) = (request.tools.as_ref(), request.tool_choice.as_ref()) else {
        return Ok(None);
    };
    let tools = tools.iter().filter(|tool| tool.kind == "function");
    let selected = match choice {
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
        .map(|tool| tool_argument_schema(tool).map(|schema| (tool, schema)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((!schemas.is_empty()).then_some(schemas))
}

fn tool_argument_schema(tool: &ChatTool) -> Result<serde_json::Value, ToolProtocolError> {
    object_argument_schema(tool.function.parameters.as_ref()).map_err(|error| {
        ToolProtocolError(format!(
            "function {:?} parameters: {error}",
            tool.function.name
        ))
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
    fn argument_object_domain_retains_union_constraints_and_root_definitions() {
        use onnx_genai_engine::byte_grammar::{GrammarByteCursor, GrammarCandidate};
        for schema in [
            None,
            Some(serde_json::Value::Null),
            Some(serde_json::json!(true)),
            Some(serde_json::json!({})),
            Some(serde_json::json!({"type":["array","object"]})),
        ] {
            let constrained = object_argument_schema(schema.as_ref()).unwrap();
            let cursor = GrammarByteCursor::json_schema(&constrained, 4096).unwrap();
            assert!(matches!(
                cursor.preview_json_root(r#"{"x":1}"#).unwrap(),
                GrammarCandidate::Viable { complete: true }
            ));
            for scalar in ["[]", "1", "null", r#""text""#] {
                assert!(
                    matches!(
                        cursor.preview_json_root(scalar).unwrap(),
                        GrammarCandidate::Rejected { .. }
                    ),
                    "schema={schema:?} value={scalar}"
                );
            }
        }
        let schema = serde_json::json!({"$defs":{"Args":{"type":"object","properties":{"x":{"const":1}},"required":["x"],"additionalProperties":false}},"$ref":"#/$defs/Args","allOf":[{"minProperties":1}]});
        let constrained = object_argument_schema(Some(&schema)).unwrap();
        let cursor = GrammarByteCursor::json_schema(&constrained, 4096).unwrap();
        assert!(matches!(
            cursor.preview_json_root(r#"{"x":1}"#).unwrap(),
            GrammarCandidate::Viable { complete: true }
        ));
        for invalid in ["[]", "{}", r#"{"x":2}"#, r#"{"x":1,"y":2}"#] {
            assert!(
                matches!(
                    cursor.preview_json_root(invalid).unwrap(),
                    GrammarCandidate::Rejected { .. }
                ),
                "{invalid}"
            );
        }
    }

    #[test]
    fn argument_object_domain_does_not_turn_impossible_or_invalid_schemas_into_valid_tools() {
        use onnx_genai_engine::byte_grammar::GrammarByteCursor;
        for schema in [
            serde_json::json!(false),
            serde_json::json!({"type":"string"}),
            serde_json::json!({"const":[]}),
        ] {
            let constrained = object_argument_schema(Some(&schema)).unwrap();
            assert!(
                GrammarByteCursor::json_schema(&constrained, 4096).is_err(),
                "{schema}"
            );
        }
        for schema in [
            serde_json::json!(3),
            serde_json::json!([]),
            serde_json::json!({"type":"object","allOf":false}),
        ] {
            assert!(object_argument_schema(Some(&schema)).is_err(), "{schema}");
        }
    }

    #[test]
    fn forced_argument_resources_preserve_reference_scope_and_tool_isolation() {
        use onnx_genai_engine::byte_grammar::{GrammarByteCursor, GrammarCandidate};
        use serde_json::json;
        let object = |city: &str| json!({"type":"object","properties":{"city":{"const":city}},"required":["city"],"additionalProperties":false});
        let schemas = [
            json!({"$defs":{"Args":object("Paris")},"$ref":"json-schema:///#/$defs/Args"}),
            json!({"$defs":{"a/b~c":object("Paris")},"$ref":"#/$defs/a~1b~0c"}),
            json!({"$id":"https://example.invalid/args","$defs":{"Args":object("Paris")},"$ref":"#/$defs/Args"}),
            json!({"$defs":{"Args":{"$anchor":"Args","allOf":[object("Paris")]}},"$ref":"#Args"}),
            json!({"$id":"https://example.invalid/root/","$defs":{"Args":{"$id":"args", "allOf":[object("Paris")]}},"$ref":"args"}),
            json!({"type":"object","properties":{"city":{"const":"Paris"}},"required":["city"],"additionalProperties":false,"default":{"$ref":"literal-instance-data"}}),
            json!({"type":"object","properties":{"city":{"const":"Paris"},"next":{"anyOf":[{"type":"null"},{"$ref":"#"}]}},"required":["city"],"additionalProperties":false}),
            json!({"$schema":"http://json-schema.org/draft-04/schema#","definitions":{"Args":{"type":"object","properties":{"city":{"enum":["Paris"]}},"required":["city"],"additionalProperties":false}},"$ref":"#/definitions/Args"}),
        ];
        for parameters in schemas {
            // The explicit-ID case deliberately reuses the other tool's URI.
            // Separate registries must retain each tool's city constraint.
            let request: ChatCompletionRequest = serde_json::from_value(json!({
                "model":"fixture", "messages":[{"role":"user","content":"lookup"}],
                "tools":[{"type":"function","function":{"name":"lookup","parameters":parameters}},
                         {"type":"function","function":{"name":"other","parameters":{"$id":"https://example.invalid/args","allOf":[object("Rome")]}}}],
                "tool_choice":"required"
            })).unwrap();
            let ToolOutputConstraint::Set(Some(GenerateConstraint::Lark { grammar, .. })) =
                protocol("tagged-json").output_constraint(&request).unwrap()
            else {
                panic!("missing forced grammar")
            };
            let cursor = GrammarByteCursor::lark(&grammar, 16384)
                .unwrap_or_else(|e| panic!("{parameters}: {e:#}"));
            for (name, city, valid) in [
                ("lookup", "Paris", true),
                ("lookup", "Rome", false),
                ("other", "Rome", true),
                ("other", "Paris", false),
            ] {
                let text = format!(
                    "<tool_call>\n{}\n</tool_call>",
                    json!({"name":name,"arguments":{"city":city}})
                );
                let result = cursor.preview_root(text.as_bytes()).unwrap();
                assert_eq!(
                    matches!(result, GrammarCandidate::Viable { complete: true }),
                    valid,
                    "{parameters}: {text}: {result:?}"
                );
            }
            let spaced = "<tool_call>\n{ \"arguments\" : { \"city\" : \"Paris\" } , \"id\" : \"call_1\" , \"name\" : \"lookup\" }\n</tool_call>";
            let spaced_result = cursor.preview_root(spaced.as_bytes()).unwrap();
            assert!(
                matches!(spaced_result, GrammarCandidate::Viable { complete: true }),
                "{parameters}: {spaced_result:?}: {spaced}"
            );
            for body in [
                r#"{"name":"lookup","arguments":[],"id":"call_1"}"#.to_owned(),
                r#"{"name":"lookup","arguments":{"city":"Paris"},"extra":1}"#.to_owned(),
                r#"{"name":"lookup","name":"lookup","arguments":{"city":"Paris"}}"#.to_owned(),
                r#"{"name":"lookup"}"#.to_owned(),
                r#"{"name":"lookup","arguments":{"city":"Paris"},"id":""}"#.to_owned(),
                format!(
                    r#"{{"name":"lookup","arguments":{{"city":"Paris"}},"id":"{}"}}"#,
                    "x".repeat(onnx_genai_metadata::MAX_TOOL_CALL_ID_BYTES + 1)
                ),
            ] {
                let text = format!("<tool_call>\n{body}\n</tool_call>");
                assert!(
                    !matches!(
                        cursor.preview_root(text.as_bytes()).unwrap(),
                        GrammarCandidate::Viable { complete: true }
                    ),
                    "admitted invalid envelope: {text}"
                );
            }
        }
    }

    #[test]
    fn required_tool_choice_retains_viable_tools_beside_impossible_schemas() {
        use onnx_genai_engine::byte_grammar::{GrammarByteCursor, GrammarCandidate};
        use serde_json::json;
        for impossible in [
            json!(false),
            json!({"type":"string"}),
            json!({"type":"object","properties":{"x":false},"required":["x"]}),
        ] {
            let request: ChatCompletionRequest = serde_json::from_value(json!({
                "messages":[{"role":"user","content":"pick"}],"tool_choice":"required",
                "tools":[{"type":"function","function":{"name":"impossible","parameters":impossible}},
                         {"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"const":"Paris"}},"required":["city"],"additionalProperties":false}}}]
            })).unwrap();
            let ToolOutputConstraint::Set(Some(GenerateConstraint::Lark { grammar, .. })) =
                protocol("tagged-json").output_constraint(&request).unwrap()
            else {
                panic!("missing forced constraint")
            };
            let cursor = GrammarByteCursor::lark(&grammar, 4096)
                .unwrap_or_else(|error| panic!("{impossible}: {error:#}"));
            assert!(matches!(cursor.preview_root(b"<tool_call>\n{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>").unwrap(),GrammarCandidate::Viable{complete:true}));
        }
    }

    #[test]
    fn required_tool_schema_pruning_preserves_invalid_and_impossible_failures() {
        use onnx_genai_engine::byte_grammar::GrammarByteCursor;
        use serde_json::json;
        let request = |parameters: serde_json::Value,
                       choice: serde_json::Value,
                       other: serde_json::Value|
         -> ChatCompletionRequest {
            serde_json::from_value(
                json!({"messages":[{"role":"user","content":"pick"}],"tool_choice":choice,
                "tools":[{"type":"function","function":{"name":"broken","parameters":parameters}},
                         {"type":"function","function":{"name":"other","parameters":other}}]}),
            )
            .unwrap()
        };
        let error = protocol("tagged-json")
            .output_constraint(&request(json!(false), json!("required"), json!(false)))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("all required tool parameter schemas are unsatisfiable"),
            "{error}"
        );
        for invalid in [
            json!({"$ref":"#/$defs/missing"}),
            json!({"type":"not-a-json-type"}),
            json!({"oneOf":[{"type":"object"},{"type":"object"}]}),
        ] {
            let error = protocol("tagged-json")
                .output_constraint(&request(
                    invalid,
                    json!("required"),
                    json!({"type":"object"}),
                ))
                .unwrap_err();
            assert!(
                error.to_string().contains("function \"broken\" parameters"),
                "{error}"
            );
            assert!(!error.to_string().contains("all required"), "{error}");
        }
        let selected = request(
            json!(false),
            json!({"type":"function","function":{"name":"broken"}}),
            json!({"type":"object"}),
        );
        let ToolOutputConstraint::Set(Some(GenerateConstraint::Lark { grammar, .. })) =
            protocol("tagged-json")
                .output_constraint(&selected)
                .unwrap()
        else {
            panic!("missing specific-tool constraint")
        };
        assert!(GrammarByteCursor::lark(&grammar, 4096).is_err());
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
        assert!(error.contains("'unknown@v9'"), "{error}");
        assert!(error.contains("not registered"), "{error}");
        assert!(
            error.contains("do not add required_capabilities"),
            "{error}"
        );
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
    fn openai_tool_choice_translates_to_transport_neutral_engine_policy() {
        let request = |tool_choice: serde_json::Value, with_tools: bool| {
            serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({
                "model": "fixture",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": with_tools.then(|| serde_json::json!([
                    {"type": "function", "function": {"name": "weather"}}
                ])),
                "tool_choice": tool_choice
            }))
            .unwrap()
        };

        assert_eq!(
            request_policy(&request(serde_json::json!("auto"), true)),
            ToolCallPolicy::Auto
        );
        assert_eq!(
            request_policy(&request(serde_json::json!("required"), true)),
            ToolCallPolicy::Required
        );
        assert_eq!(
            request_policy(&request(serde_json::json!("none"), true)),
            ToolCallPolicy::Disabled
        );
        assert_eq!(
            request_policy(&request(serde_json::json!("auto"), false)),
            ToolCallPolicy::Disabled
        );
        assert_eq!(
            request_policy(&request(
                serde_json::json!({
                    "type": "function",
                    "function": {"name": "weather"}
                }),
                true,
            )),
            ToolCallPolicy::Specific {
                function: "weather".to_string()
            }
        );
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
            ToolOutputConstraint::Set(Some(GenerateConstraint::Lark { .. }))
        ));

        let atem = protocol("atem-xml");
        let rendered = atem.render(&request).expect("ATEM request renders");
        assert!(rendered.fallback_prefix.contains("<atem:tools>"));
        assert!(rendered.fallback_prefix.contains("<|tool_choice|>"));
        assert!(matches!(
            atem.output_constraint(&request).unwrap(),
            ToolOutputConstraint::Set(Some(GenerateConstraint::XmlToolCall { .. }))
        ));
    }

    #[test]
    fn xml_forcing_rejects_only_backend_proven_unencodable_roots_and_keeps_auto_unforced() {
        let request = |schema: serde_json::Value, choice: &str| {
            serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({"messages":[],"tools":[{"type":"function","function":{"name":"lookup","parameters":schema}}],"tool_choice":choice})).unwrap()
        };
        let xml = protocol("atem-xml");
        let schema = serde_json::json!({"$defs":{"Args":{"const":{"":1}}},"$ref":"#/$defs/Args"});
        let error = xml
            .output_constraint(&request(schema.clone(), "required"))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("generation key-encoding capability"),
            "{error}"
        );
        assert!(matches!(
            xml.output_constraint(&request(schema, "auto")).unwrap(),
            ToolOutputConstraint::Unchanged
        ));
        for minimum in [1, 2] {
            let dynamic = serde_json::json!({"type":"object","patternProperties":{"^a$":{"type":"integer"}},"additionalProperties":false,"minProperties":minimum});
            let error = xml
                .output_constraint(&request(dynamic, "required"))
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("generation key-encoding capability"),
                "{error}"
            );
        }
        let lexical = serde_json::json!({"type":"object","x-guidance":{"key_separator":"="}});
        let error = xml
            .output_constraint(&request(lexical, "required"))
            .unwrap_err();
        assert!(error.to_string().contains("standard JSON"), "{error}");
        let invalid = serde_json::json!({"$ref":"#/$defs/missing"});
        let error = xml
            .output_constraint(&request(invalid, "required"))
            .unwrap_err();
        assert!(
            !error
                .to_string()
                .contains("generation key-encoding capability"),
            "{error}"
        );
    }

    #[test]
    fn xml_mixed_tools_do_not_prune_policy_or_unsupported_value_errors() {
        let xml = protocol("atem-xml");
        let request = |schema: serde_json::Value, mixed: bool| {
            let mut tools = vec![
                serde_json::json!({"type":"function","function":{"name":"broken","parameters":schema}}),
            ];
            if mixed {
                tools.push(serde_json::json!({"type":"function","function":{"name":"usable","parameters":{"type":"object"}}}));
            }
            serde_json::from_value::<ChatCompletionRequest>(
                serde_json::json!({"messages":[],"tools":tools,"tool_choice":"required"}),
            )
            .unwrap()
        };
        let unsupported = serde_json::json!({"oneOf":[{"type":"number"},{"type":"integer"}]});
        for required in [false, true] {
            for referenced in [false, true] {
                let mut schema = serde_json::json!({"$id":"https://example.invalid/tool-policy","$defs":{"Value":unsupported},"type":"object","properties":{"":if referenced {serde_json::json!({"$ref":"#/$defs/Value"})} else {unsupported.clone()}},"additionalProperties":false});
                if required {
                    schema["required"] = serde_json::json!([""]);
                }
                let single = xml
                    .output_constraint(&request(schema.clone(), false))
                    .unwrap_err();
                let mixed = xml.output_constraint(&request(schema, true)).unwrap_err();
                assert_eq!(single.to_string(), mixed.to_string());
                assert!(
                    !mixed
                        .to_string()
                        .contains("generation key-encoding capability"),
                    "{mixed}"
                );
            }
        }
        let policy_only = serde_json::json!({"type":"object","properties":{"":{"type":"integer"}},"required":[""],"additionalProperties":false});
        let single = xml
            .output_constraint(&request(policy_only.clone(), false))
            .unwrap_err();
        let mixed = xml
            .output_constraint(&request(policy_only, true))
            .unwrap_err();
        assert_eq!(single.to_string(), mixed.to_string());
        assert!(
            mixed
                .to_string()
                .contains("generation key-encoding capability"),
            "{mixed}"
        );
        assert!(matches!(
            xml.output_constraint(&request(serde_json::Value::Bool(false), true))
                .unwrap(),
            ToolOutputConstraint::Set(Some(GenerateConstraint::XmlToolCall { .. }))
        ));
    }

    #[test]
    fn xml_mixed_tools_prune_only_original_empty_cardinality_languages() {
        let xml = protocol("atem-xml");
        for schema in [
            serde_json::json!({"type":"object","additionalProperties":false,"minProperties":1}),
            serde_json::json!({"type":"object","properties":{"ok":false},"additionalProperties":false,"minProperties":1}),
            serde_json::json!({"type":"object","minProperties":2,"maxProperties":1}),
            serde_json::json!({"anyOf":[false,{"type":"object","additionalProperties":false,"minProperties":1}]}),
        ] {
            let request = serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({"messages":[],"tools":[{"type":"function","function":{"name":"impossible","parameters":schema}},{"type":"function","function":{"name":"usable","parameters":{"type":"object"}}}],"tool_choice":"required"})).unwrap();
            let ToolOutputConstraint::Set(Some(GenerateConstraint::XmlToolCall {
                grammar, ..
            })) = xml.output_constraint(&request).unwrap()
            else {
                panic!("usable tool was not retained");
            };
            assert!(grammar.contains("usable"));
            assert!(!grammar.contains("impossible"));
        }
    }

    #[test]
    fn declared_adapters_match_reordered_tool_results_by_id() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "fixture",
            "tools": [
                {"type": "function", "function": {"name": "weather"}},
                {"type": "function", "function": {"name": "time"}}
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": "{\"city\":\"Paris\"}"
                            }
                        },
                        {
                            "id": "call_time",
                            "type": "function",
                            "function": {
                                "name": "time",
                                "arguments": "{\"timezone\":\"UTC\"}"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_time",
                    "name": "time",
                    "content": "UTC"
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather",
                    "name": "weather",
                    "content": "Paris"
                }
            ]
        }))
        .unwrap();

        for identity in ["tagged-json", "atem-xml"] {
            let protocol = protocol(identity);
            validate_request(&request, Some(&protocol), None)
                .unwrap_or_else(|error| panic!("{identity} rejected valid tool results: {error}"));
            let rendered = protocol.render(&request).unwrap();
            assert!(
                rendered
                    .fallback_prefix
                    .contains(if identity == "tagged-json" {
                        "<|tools|>"
                    } else {
                        "<atem:tools>"
                    }),
                "{identity} rendered the wrong tools marker"
            );
        }
    }
}
