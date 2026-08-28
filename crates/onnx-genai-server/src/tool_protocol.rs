//! Server-owned adapters for declared tool-call wire protocols.
//!
//! `package.tool_protocol` is the portable declaration.  These adapters are one
//! runtime's implementation of that declaration; a package never names an
//! adapter, parser library, or model family.

use std::{collections::BTreeSet, fmt, path::Path, sync::Arc};

use onnx_genai_engine::GenerateConstraint;
use onnx_genai_metadata::ToolProtocolDeclaration;

use crate::types::{
    ChatCompletionRequest, ChatMessageToolCall, ChatMessageToolCallFunction, ChatTool, ToolChoice,
    ToolChoiceMode,
};

const MAX_TOOL_PAYLOAD_BYTES: usize = 64 * 1024;
const MAX_TOOL_CALLS: usize = 32;
const MAX_TOOL_NAME_BYTES: usize = 256;
const MAX_TOOL_CALL_ID_BYTES: usize = 256;

/// A result while consuming model output one arbitrary chunk at a time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolParseOutcome {
    /// This protocol's opening envelope has not appeared.
    NoCall,
    /// An opening envelope appeared but cannot yet be decided.
    Incomplete,
    /// The complete envelope is valid and contains these calls.
    Complete(Vec<ChatMessageToolCall>),
    /// The envelope is complete enough to reject, with an actionable reason.
    Malformed(String),
}

/// Incremental parsing state.  Chunk boundaries have no semantic effect.
#[derive(Debug, Default)]
pub struct ToolCallStream {
    input: String,
    terminal_error: Option<String>,
}

impl ToolCallStream {
    pub fn push(&mut self, protocol: &ToolProtocol, chunk: &str) -> ToolParseOutcome {
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
        protocol.parse(&self.input)
    }

    pub fn finish(self, protocol: &ToolProtocol) -> ToolParseOutcome {
        if let Some(error) = self.terminal_error {
            return ToolParseOutcome::Malformed(error);
        }
        protocol.parse(&self.input)
    }
}

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

trait ToolProtocolAdapter: Send + Sync {
    fn declaration(&self) -> (&'static str, &'static str);
    fn render(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<RenderedToolRequest, ToolProtocolError>;
    fn output_constraint(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ToolOutputConstraint, ToolProtocolError>;
    fn parse(&self, input: &str) -> ToolParseOutcome;
}

/// A resolved declaration.  The dynamic adapter is deliberately private: it is
/// a runtime implementation detail, while the public metadata declaration is
/// portable.
#[derive(Clone)]
pub struct ToolProtocol(Arc<dyn ToolProtocolAdapter>);

impl fmt::Debug for ToolProtocol {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (identity, version) = self.declaration();
        formatter
            .debug_struct("ToolProtocol")
            .field("identity", &identity)
            .field("version", &version)
            .finish()
    }
}

impl ToolProtocol {
    pub fn declaration(&self) -> (&'static str, &'static str) {
        self.0.declaration()
    }

    pub(crate) fn render(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<RenderedToolRequest, ToolProtocolError> {
        self.0.render(request)
    }

    pub(crate) fn output_constraint(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ToolOutputConstraint, ToolProtocolError> {
        self.0.output_constraint(request)
    }

    /// Convert a terminal parser outcome into a protocol-specific error at the
    /// boundary where the complete model output is known.
    pub fn output_error(
        &self,
        outcome: &ToolParseOutcome,
        boundary: &str,
    ) -> Option<ToolProtocolError> {
        let (identity, version) = self.declaration();
        match outcome {
            ToolParseOutcome::NoCall | ToolParseOutcome::Complete(_) => None,
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
        &self,
        outcome: &ToolParseOutcome,
        boundary: &str,
    ) -> Option<ToolProtocolError> {
        matches!(outcome, ToolParseOutcome::Malformed(_))
            .then(|| self.output_error(outcome, boundary))
            .flatten()
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
                    .map(|call| call.function.name.as_str())
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
        if input.len() > MAX_TOOL_PAYLOAD_BYTES {
            return ToolParseOutcome::Malformed(format!(
                "tool output exceeds the {MAX_TOOL_PAYLOAD_BYTES}-byte protocol limit; reduce the model envelope"
            ));
        }
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
}

/// Resolve the exact declaration supplied by metadata, without fallback or
/// parser probing.  Keeping this registry at the adapter seam means another
/// runtime can implement the same portable declaration differently.
pub(crate) fn resolve(
    declaration: &ToolProtocolDeclaration,
    metadata_path: &Path,
) -> Result<ToolProtocol, ToolProtocolError> {
    let adapters: [Arc<dyn ToolProtocolAdapter>; 2] = [Arc::new(TaggedJsonV1), Arc::new(AtemXmlV1)];
    let matches = adapters
        .into_iter()
        .filter(|adapter| {
            let (identity, version) = adapter.declaration();
            identity == declaration.identity && version == declaration.version
        })
        .collect::<Vec<_>>();
    match matches.len() {
        1 => Ok(ToolProtocol(
            matches.into_iter().next().expect("one adapter"),
        )),
        0 => Err(ToolProtocolError(format!(
            "{} declares package.tool_protocol identity {:?} version {:?}, but this server implements no such protocol. \
             Use one of tagged-json@v1 or atem-xml@v1, or install a runtime adapter for the declared protocol.",
            metadata_path.display(),
            declaration.identity,
            declaration.version,
        ))),
        count => Err(ToolProtocolError(format!(
            "{} declares package.tool_protocol identity {:?} version {:?}, but {count} runtime adapters claim it. \
             The declaration is ambiguous; install exactly one adapter for that identity and version.",
            metadata_path.display(),
            declaration.identity,
            declaration.version,
        ))),
    }
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

#[derive(Debug)]
pub struct ToolProtocolError(String);

impl fmt::Display for ToolProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ToolProtocolError {}

struct TaggedJsonV1;

impl ToolProtocolAdapter for TaggedJsonV1 {
    fn declaration(&self) -> (&'static str, &'static str) {
        ("tagged-json", "v1")
    }

    fn render(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<RenderedToolRequest, ToolProtocolError> {
        render_tagged_json_request(request)
    }

    fn output_constraint(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ToolOutputConstraint, ToolProtocolError> {
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
            format!("start: \"<tool_call>\\n\" tool \"\\n</tool_call>\"\ntool: %json {schema}\n"),
        ))))
    }

    fn parse(&self, input: &str) -> ToolParseOutcome {
        parse_tagged_json(input)
    }
}

struct AtemXmlV1;

impl ToolProtocolAdapter for AtemXmlV1 {
    fn declaration(&self) -> (&'static str, &'static str) {
        ("atem-xml", "v1")
    }

    fn render(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<RenderedToolRequest, ToolProtocolError> {
        render_atem_xml_request(request)
    }

    fn output_constraint(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ToolOutputConstraint, ToolProtocolError> {
        Ok(if forced_tool_choice_schemas(request).is_some() {
            // ATEM is XML, and the engine's JSON grammar cannot describe its
            // quoted attributes and escaped parameter bodies. Explicitly clear
            // a response-format constraint rather than emitting tagged JSON.
            ToolOutputConstraint::Set(None)
        } else {
            ToolOutputConstraint::Unchanged
        })
    }

    fn parse(&self, input: &str) -> ToolParseOutcome {
        parse_atem_xml(input)
    }
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

fn parse_tagged_json(input: &str) -> ToolParseOutcome {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let Some(mut rest) = input.split_once(OPEN).map(|(_, rest)| rest) else {
        return ToolParseOutcome::NoCall;
    };
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
                if !after.trim().is_empty() {
                    return ToolParseOutcome::Malformed(
                        "tagged-json@v1 has trailing text after a tool_call envelope".to_string(),
                    );
                }
                return values_to_calls("tagged-json@v1", values);
            }
        }
    }
}

fn parse_atem_xml(input: &str) -> ToolParseOutcome {
    const OPEN: &str = "<atem:invoke";
    const CLOSE: &str = "</atem:invoke>";
    let Some(mut rest) = input.split_once(OPEN).map(|(_, rest)| rest) else {
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
            None => return values_to_calls("atem-xml@v1", values),
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
        let arguments = match serde_json::to_string(&arguments) {
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
        calls.push(ChatMessageToolCall {
            id,
            kind: "function".to_string(),
            function: ChatMessageToolCallFunction {
                name: name.to_string(),
                arguments,
            },
        });
    }
    ToolParseOutcome::Complete(calls)
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
                stream.push(&resolved, &input[..split]);
                let actual = stream.push(&resolved, &input[split..]);
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
            let outcome = stream.push(&resolved, input);
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
            &resolved,
            r#"<tool_call>{"name":"read","arguments":{"path":"src/"#,
        );
        assert_eq!(partial, ToolParseOutcome::Incomplete);
        assert!(
            resolved
                .incremental_output_error(&partial, "SSE chunk ingestion")
                .is_none()
        );
        let complete = stream.push(&resolved, r#"lib.rs"}}</tool_call>"#);
        assert!(matches!(complete, ToolParseOutcome::Complete(_)));
        assert!(
            resolved
                .incremental_output_error(&complete, "SSE chunk ingestion")
                .is_none()
        );
        assert_eq!(stream.finish(&resolved), complete);
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
                    stream.push(&resolved, chunk),
                    ToolParseOutcome::Incomplete
                ));
            }
            let outcome = stream.finish(&resolved);
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
            let outcome = stream.push(&resolved, &oversized);
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
                stream.push(&resolved, "<tool_call>{}</tool_call>"),
                ToolParseOutcome::Malformed(message) if message.contains("byte protocol limit")
            ));
        }
    }
}
