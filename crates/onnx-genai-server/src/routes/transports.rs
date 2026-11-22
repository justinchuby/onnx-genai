//! Responses and Anthropic are wire adapters over the same chat transaction.
//! Model admission, media processing, tool validation and sampling stay in chat.
use super::*;
use axum::body::{Body, to_bytes};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use tokio_stream::StreamExt;

#[derive(Clone, Copy)]
enum Protocol {
    Responses,
    Anthropic,
}

pub(crate) async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<Map<String, Value>>,
) -> Result<Response, ApiError> {
    serve(state, headers, response_request(body)?, Protocol::Responses).await
}
pub(crate) async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<Map<String, Value>>,
) -> Result<Response, ApiError> {
    serve(
        state,
        headers,
        anthropic_request(body)?,
        Protocol::Anthropic,
    )
    .await
}
pub(crate) async fn responses_tokens(
    State(state): State<AppState>,
    ApiJson(body): ApiJson<Map<String, Value>>,
) -> Result<Json<Value>, ApiError> {
    super::llama::input_tokens(State(state), ApiJson(response_request(body)?)).await
}
pub(crate) async fn messages_tokens(
    State(state): State<AppState>,
    ApiJson(mut body): ApiJson<Map<String, Value>>,
) -> Result<Json<Value>, ApiError> {
    body.entry("max_tokens").or_insert(json!(1));
    let request = anthropic_request(body)?;
    Ok(Json(
        json!({"input_tokens": super::llama::count_input_tokens(&state, &request).await?}),
    ))
}

fn anthropic_error_kind(status: StatusCode) -> &'static str {
    match status {
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
        status if status.is_client_error() => "invalid_request_error",
        _ => "api_error",
    }
}

/// Wrap handler and extractor failures alike in the Messages error envelope.
pub(crate) async fn anthropic_error_response(response: Response) -> Response {
    if !response.status().is_client_error() && !response.status().is_server_error() {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let message = match to_bytes(body,8*1024*1024).await {
        Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
            Ok(value) => value["error"]["message"].as_str().unwrap_or("server returned an error without a message; inspect server logs before retrying").to_owned(),
            Err(err) => { parts.status=StatusCode::INTERNAL_SERVER_ERROR; format!("could not encode the Messages error response: invalid internal error JSON: {err}; inspect server logs") },
        },
        Err(err) => { parts.status=StatusCode::INTERNAL_SERVER_ERROR; format!("could not read the Messages error response: {err}; reduce request size or inspect server logs") },
    };
    let value = json!({"type":"error","error":{"type":anthropic_error_kind(parts.status),"message":message}});
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.insert(
        header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );
    Response::from_parts(parts, Body::from(value.to_string()))
}

fn required(body: &mut Map<String, Value>, key: &str) -> Result<Value, ApiError> {
    body.remove(key)
        .ok_or_else(|| ApiError::invalid_request(format!("request field '{key}' is required")))
}
fn string<'a>(value: &'a Value, field: &str) -> Result<&'a str, ApiError> {
    value
        .as_str()
        .ok_or_else(|| ApiError::invalid_request(format!("{field} must be a string")))
}
fn array(value: Value, field: &str) -> Result<Vec<Value>, ApiError> {
    value
        .as_array()
        .cloned()
        .ok_or_else(|| ApiError::invalid_request(format!("{field} must be an array")))
}
fn chat(body: Map<String, Value>) -> Result<ChatCompletionRequest, ApiError> {
    serde_path_to_error::deserialize(Value::Object(body)).map_err(|err| {
        ApiError::invalid_request(format!(
            "invalid generation request: {err}; check messages, tools and sampling fields"
        ))
    })
}
fn neutral(body: &mut Map<String, Value>, key: &str, accepted: &Value) -> Result<(), ApiError> {
    if let Some(value) = body.remove(key)
        && !value.is_null()
        && &value != accepted
    {
        return Err(ApiError::invalid_request(format!(
            "{key}={value} is unsupported; omit it or use {accepted}"
        )));
    }
    Ok(())
}

fn response_request(mut body: Map<String, Value>) -> Result<ChatCompletionRequest, ApiError> {
    let input = required(&mut body, "input")?;
    let mut messages = Vec::new();
    if let Some(instructions) = body.remove("instructions")
        && !instructions.is_null()
    {
        messages.push(json!({"role":"system","content":string(&instructions,"instructions")?}));
    }
    match input {
        Value::String(text) => messages.push(json!({"role":"user","content":text})),
        Value::Array(items) => {
            for item in items {
                match item.get("type").and_then(Value::as_str).unwrap_or("message") {
                "message" => {
                    let role = string(&item["role"],"input message role")?;
                    let message = json!({"role":role,"content":response_content(&item["content"])?});
                    if role == "assistant" { push_assistant(&mut messages, message); } else { messages.push(message); }
                }
                "reasoning" => {
                    if item.get("encrypted_content").is_some_and(|value| !value.is_null()) {
                        return Err(ApiError::invalid_request("encrypted reasoning cannot be rendered by this package; provide visible summary text"));
                    }
                    let mut text = String::new();
                    for part in array(item.get("summary").cloned().unwrap_or_else(|| json!([])), "reasoning.summary")? {
                        if part["type"] != "summary_text" { return Err(ApiError::invalid_request("reasoning.summary requires summary_text items")); }
                        text.push_str(string(&part["text"], "reasoning.summary.text")?);
                    }
                    if !text.is_empty() { push_assistant(&mut messages, json!({"role":"assistant","content":null,"reasoning_content":text})); }
                }
                "function_call" => push_assistant(&mut messages, json!({"role":"assistant","content":null,"tool_calls":[{
                    "id":string(&item["call_id"],"function_call.call_id")?,"type":"function","function":{
                        "name":string(&item["name"],"function_call.name")?,"arguments":string(&item["arguments"],"function_call.arguments")?
                    }}]})),
                "function_call_output" => messages.push(json!({"role":"tool","tool_call_id":string(&item["call_id"],"function_call_output.call_id")?,"content":string(&item["output"],"function_call_output.output")?})),
                other => return Err(ApiError::invalid_request(format!("Responses input item type '{other}' is unsupported; use message/reasoning/function_call/function_call_output"))),
            }
            }
        }
        _ => {
            return Err(ApiError::invalid_request(
                "input must be text or a Responses item array",
            ));
        }
    }
    body.insert("messages".into(), json!(messages));
    if let Some(budget) = body.remove("max_output_tokens") {
        body.insert("max_completion_tokens".into(), budget);
    }
    if let Some(reasoning) = body.remove("reasoning") {
        let mut reasoning = reasoning
            .as_object()
            .cloned()
            .ok_or_else(|| ApiError::invalid_request("reasoning must be an object"))?;
        if let Some(effort) = reasoning.remove("effort") {
            body.insert("reasoning_effort".into(), effort);
        }
        neutral(&mut reasoning, "summary", &json!("auto"))?;
        if !reasoning.is_empty() {
            return Err(ApiError::invalid_request(
                "unsupported reasoning fields; use reasoning.effort and optional summary=auto",
            ));
        }
    }
    if let Some(text) = body.remove("text") {
        let format = text.get("format").ok_or_else(|| {
            ApiError::invalid_request("text.format is required when text settings are supplied")
        })?;
        let format = if format["type"] == "json_schema" {
            json!({"type":"json_schema","json_schema":format})
        } else {
            format.clone()
        };
        body.insert("response_format".into(), format);
    }
    if let Some(tools) = body.remove("tools") {
        let mut lowered = Vec::new();
        for mut tool in array(tools, "tools")? {
            let object = tool.as_object_mut().ok_or_else(|| {
                ApiError::invalid_request("each Responses tool must be an object")
            })?;
            if object.remove("type") != Some(json!("function")) {
                return Err(ApiError::invalid_request(
                    "Responses tools must be type=function; hosted tools are not implemented",
                ));
            }
            // Strictness is enforced by the package's typed tool transaction.
            object.remove("strict");
            lowered.push(json!({"type":"function","function":object}));
        }
        body.insert("tools".into(), json!(lowered));
    }
    if let Some(choice) = body.remove("tool_choice") {
        let choice = if choice.is_object() {
            json!({"type":"function","function":{"name":string(&choice["name"],"tool_choice.name")?}})
        } else {
            choice
        };
        body.insert("tool_choice".into(), choice);
    }
    neutral(&mut body, "store", &json!(false))?;
    neutral(&mut body, "background", &json!(false))?;
    neutral(&mut body, "parallel_tool_calls", &json!(true))?;
    neutral(&mut body, "truncation", &json!("disabled"))?;
    neutral(&mut body, "include", &json!([]))?;
    neutral(&mut body, "previous_response_id", &Value::Null)?;
    chat(body)
}
// Responses splits one assistant turn into reasoning/message/call items. Rejoin
// adjacent items before validating the shared chat tool transaction.
fn push_assistant(messages: &mut Vec<Value>, message: Value) {
    if let Some(previous) = messages
        .last_mut()
        .filter(|item| item["role"] == "assistant")
    {
        if let Some(reasoning) = message["reasoning_content"].as_str() {
            previous["reasoning_content"] = json!(format!(
                "{}{reasoning}",
                previous["reasoning_content"].as_str().unwrap_or_default()
            ));
        }
        if let Some(calls) = message["tool_calls"].as_array() {
            if !previous["tool_calls"].is_array() {
                previous["tool_calls"] = json!([]);
            }
            if let Some(existing) = previous["tool_calls"].as_array_mut() {
                existing.extend(calls.iter().cloned());
            }
        }
        if !message["content"].is_null() {
            let mut parts = previous["content"].as_array().cloned().unwrap_or_default();
            if let Some(text) = previous["content"].as_str() {
                parts.push(json!({"type":"text","text":text}));
            }
            if let Some(new_parts) = message["content"].as_array() {
                parts.extend(new_parts.iter().cloned());
            } else if let Some(text) = message["content"].as_str() {
                parts.push(json!({"type":"text","text":text}));
            }
            previous["content"] = json!(parts);
        }
    } else {
        messages.push(message);
    }
}
fn response_content(content: &Value) -> Result<Value, ApiError> {
    if content.is_string() {
        return Ok(content.clone());
    }
    let mut parts = Vec::new();
    for part in array(content.clone(), "message.content")? {
        parts.push(match part["type"].as_str() {
            Some("input_text"|"output_text") => json!({"type":"text","text":string(&part["text"],"content.text")?}),
            Some("input_image") => json!({"type":"image_url","image_url":{"url":string(&part["image_url"],"input_image.image_url")?}}),
            Some("input_audio") => json!({"type":"input_audio","input_audio":part["input_audio"]}),
            _ => return Err(ApiError::invalid_request("unsupported Responses content; use input_text, input_image or input_audio")),
        });
    }
    Ok(json!(parts))
}
fn anthropic_request(mut body: Map<String, Value>) -> Result<ChatCompletionRequest, ApiError> {
    let input = array(required(&mut body, "messages")?, "messages")?;
    let mut messages = Vec::new();
    if let Some(system) = body.remove("system") {
        messages.push(json!({"role":"system","content":text_blocks(&system,"system")?}));
    }
    for message in input {
        let role = string(&message["role"], "messages.role")?;
        if message["content"].is_string() {
            messages.push(json!({"role":role,"content":message["content"]}));
            continue;
        }
        let mut parts = Vec::new();
        let mut calls = Vec::new();
        let mut thinking = String::new();
        for part in array(message["content"].clone(), "messages.content")? {
            match part["type"].as_str() {
                Some("text") => parts.push(json!({"type":"text","text":string(&part["text"],"text block.text")?})),
                Some("image") => {
                    let source=&part["source"];
                    let url=match source["type"].as_str() {
                        Some("base64") => format!("data:{};base64,{}",string(&source["media_type"],"image.source.media_type")?,string(&source["data"],"image.source.data")?),
                        Some("url") => string(&source["url"],"image.source.url")?.to_string(),
                        _ => return Err(ApiError::invalid_request("image.source.type must be base64 or url")),
                    };
                    parts.push(json!({"type":"image_url","image_url":{"url":url}}));
                }
                Some("tool_use") => calls.push(json!({"id":string(&part["id"],"tool_use.id")?,"type":"function","function":{
                    "name":string(&part["name"],"tool_use.name")?,"arguments":part["input"].to_string()}})),
                Some("tool_result") => {
                    let content=part.get("content").cloned().unwrap_or_else(|| json!(""));
                    let mut text=text_blocks(&content,"tool_result.content")?;
                    if part["is_error"]==true {text=format!("Tool error: {text}");}
                    messages.push(json!({"role":"tool","tool_call_id":string(&part["tool_use_id"],"tool_result.tool_use_id")?,"content":text}));
                }
                Some("thinking") => thinking.push_str(string(&part["thinking"],"thinking.thinking")?),
                _ => return Err(ApiError::invalid_request("unsupported Anthropic content block; use text, image, thinking, tool_use or tool_result")),
            }
        }
        if !parts.is_empty() || !calls.is_empty() || !thinking.is_empty() {
            let mut lowered =
                json!({"role":role,"content":if parts.is_empty(){Value::Null}else{json!(parts)}});
            if !calls.is_empty() {
                lowered["tool_calls"] = json!(calls);
            }
            if !thinking.is_empty() {
                lowered["reasoning_content"] = json!(thinking);
            }
            messages.push(lowered);
        }
    }
    body.insert("messages".into(), json!(messages));
    if !body.contains_key("max_tokens") {
        return Err(ApiError::invalid_request(
            "max_tokens is required for Anthropic Messages",
        ));
    }
    if let Some(stops) = body.remove("stop_sequences") {
        body.insert("stop".into(), stops);
    }
    if let Some(tools) = body.remove("tools") {
        let mut lowered = Vec::new();
        for tool in array(tools, "tools")? {
            lowered.push(json!({"type":"function","function":{
            "name":string(&tool["name"],"tools.name")?,"description":tool.get("description"),"parameters":tool["input_schema"]}}));
        }
        body.insert("tools".into(), json!(lowered));
    }
    if let Some(choice) = body.remove("tool_choice") {
        if choice["disable_parallel_tool_use"] == true {
            return Err(ApiError::invalid_request(
                "disable_parallel_tool_use=true requires a single-call package policy; omit it for this server",
            ));
        }
        let choice = match choice["type"].as_str() {
            Some("auto") => json!("auto"),
            Some("any") => json!("required"),
            Some("none") => json!("none"),
            Some("tool") => {
                json!({"type":"function","function":{"name":string(&choice["name"],"tool_choice.name")?}})
            }
            _ => {
                return Err(ApiError::invalid_request(
                    "tool_choice.type must be auto, any, none or tool",
                ));
            }
        };
        body.insert("tool_choice".into(), choice);
    }
    if let Some(thinking) = body.remove("thinking") {
        match thinking["type"].as_str() {
            Some("disabled") => {
                body.insert(
                    "chat_template_kwargs".into(),
                    json!({"enable_thinking":false}),
                );
            }
            Some("adaptive") => {}
            _ => {
                return Err(ApiError::invalid_request(
                    "thinking must be type=disabled or adaptive; explicit thinking token budgets are not yet supported",
                ));
            }
        }
    }
    if let Some(metadata) = body.remove("metadata") {
        let mut metadata = metadata
            .as_object()
            .cloned()
            .ok_or_else(|| ApiError::invalid_request("metadata must be an object"))?;
        if let Some(user) = metadata.remove("user_id") {
            body.insert("user".into(), user);
        }
        if !metadata.is_empty() {
            return Err(ApiError::invalid_request(
                "Anthropic metadata supports user_id only",
            ));
        }
    }
    chat(body)
}
fn text_blocks(value: &Value, field: &str) -> Result<String, ApiError> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_string());
    }
    let mut result = String::new();
    for part in array(value.clone(), field)? {
        if part["type"] != "text" {
            return Err(ApiError::invalid_request(format!(
                "{field} must contain text blocks; convert other attachments to text first"
            )));
        }
        result.push_str(string(&part["text"], field)?);
    }
    Ok(result)
}

async fn serve(
    state: AppState,
    headers: HeaderMap,
    mut request: ChatCompletionRequest,
    protocol: Protocol,
) -> Result<Response, ApiError> {
    let streaming = request.stream;
    request.stream_options.include_usage = true;
    let response =
        super::completions::chat_completions(State(state), headers, ApiJson(request)).await?;
    if !streaming {
        let bytes = to_bytes(response.into_body(), 16 * 1024 * 1024)
            .await
            .map_err(|err| ApiError::internal(format!("chat response conversion failed: {err}")))?;
        let body: Value = serde_json::from_slice(&bytes)
            .map_err(|err| ApiError::internal(format!("chat response JSON was invalid: {err}")))?;
        return Ok(Json(buffered(&body, protocol)?).into_response());
    }
    Ok(stream_response(response, protocol))
}
fn buffered(chat: &Value, protocol: Protocol) -> Result<Value, ApiError> {
    let message = &chat["choices"][0]["message"];
    let usage = &chat["usage"];
    let calls = message["tool_calls"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut output = Vec::new();
    match protocol {
        Protocol::Responses => {
            if let Some(reasoning) = message["reasoning_content"].as_str() {
                output.push(json!({"id":format!("rs_{}",chat["id"].as_str().unwrap_or_default()),"type":"reasoning","summary":[{"type":"summary_text","text":reasoning}]}));
            }
            if let Some(text) = message["content"].as_str()
                && !text.is_empty()
            {
                output.push(json!({"id":format!("msg_{}",chat["id"].as_str().unwrap_or_default()),"type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":text,"annotations":[]}]}));
            }
            for call in calls {
                output.push(json!({"id":format!("fc_{}",call["id"].as_str().unwrap_or_default()),"type":"function_call","call_id":call["id"],"name":call["function"]["name"],"arguments":call["function"]["arguments"],"status":"completed"}));
            }
            let incomplete = chat["choices"][0]["finish_reason"] == "length";
            Ok(
                json!({"id":format!("resp_{}",chat["id"].as_str().unwrap_or_default()),"object":"response","created_at":chat["created"],"model":chat["model"],
                "status":if incomplete{"incomplete"}else{"completed"},"output":output,
                "incomplete_details":if incomplete{json!({"reason":"max_output_tokens"})}else{Value::Null},
                "error":null,"usage":{"input_tokens":usage["prompt_tokens"],"output_tokens":usage["completion_tokens"],"total_tokens":usage["total_tokens"]}}),
            )
        }
        Protocol::Anthropic => {
            if let Some(reasoning) = message["reasoning_content"].as_str() {
                output.push(json!({"type":"thinking","thinking":reasoning}));
            }
            if let Some(text) = message["content"].as_str()
                && !text.is_empty()
            {
                output.push(json!({"type":"text","text":text}));
            }
            for call in &calls {
                let args = call["function"]["arguments"].as_str().unwrap_or("{}");
                let input: Value = serde_json::from_str(args).map_err(|err| {
                    ApiError::internal(format!("committed tool arguments were invalid JSON: {err}"))
                })?;
                output.push(json!({"type":"tool_use","id":call["id"],"name":call["function"]["name"],"input":input}));
            }
            let reason = if !calls.is_empty() {
                "tool_use"
            } else if chat["choices"][0]["finish_reason"] == "length" {
                "max_tokens"
            } else if chat["choices"][0]["stop_sequence"].is_string() {
                "stop_sequence"
            } else {
                "end_turn"
            };
            Ok(
                json!({"id":format!("msg_{}",chat["id"].as_str().unwrap_or_default()),"type":"message","role":"assistant","model":chat["model"],"content":output,
                "stop_reason":reason,"stop_sequence":chat["choices"][0]["stop_sequence"],"usage":{"input_tokens":usage["prompt_tokens"],"output_tokens":usage["completion_tokens"]}}),
            )
        }
    }
}

struct StreamEncoder {
    protocol: Protocol,
    id: String,
    model: String,
    created: u64,
    text: String,
    reasoning: String,
    calls: BTreeMap<usize, Value>,
    usage: Value,
    finish: String,
    stop_sequence: Value,
    started: bool,
    text_started: bool,
    reasoning_started: bool,
    sequence: usize,
    positions: Vec<String>,
    input_tokens: Option<usize>,
    open_block: Option<usize>,
    retained_bytes: usize,
}
impl StreamEncoder {
    fn new(protocol: Protocol, input_tokens: Option<usize>) -> Self {
        Self {
            protocol,
            id: String::new(),
            model: String::new(),
            created: 0,
            text: String::new(),
            reasoning: String::new(),
            calls: BTreeMap::new(),
            usage: Value::Null,
            finish: String::new(),
            stop_sequence: Value::Null,
            started: false,
            text_started: false,
            reasoning_started: false,
            sequence: 0,
            positions: Vec::new(),
            input_tokens,
            open_block: None,
            retained_bytes: 0,
        }
    }
    fn position(&mut self, key: String) -> usize {
        if let Some(index) = self.positions.iter().position(|item| item == &key) {
            return index;
        }
        let index = self.positions.len();
        self.positions.push(key);
        index
    }
    fn start_block(&mut self, position: usize, events: &mut Vec<Event>) {
        if matches!(self.protocol, Protocol::Anthropic)
            && let Some(previous) = self.open_block.replace(position)
        {
            events.push(self.event("content_block_stop", json!({"index":previous})));
        }
    }
    fn retain(&mut self, bytes: usize) -> Result<(), String> {
        self.retained_bytes = self
            .retained_bytes
            .checked_add(bytes)
            .ok_or("adapted response byte count overflow")?;
        if self.retained_bytes > 16 * 1024 * 1024 {
            return Err("adapted response exceeded 16 MiB; reduce max_output_tokens".into());
        }
        Ok(())
    }
    fn event(&mut self, kind: &str, mut value: Value) -> Event {
        value["type"] = json!(kind);
        if matches!(self.protocol, Protocol::Responses) {
            value["sequence_number"] = json!(self.sequence);
            self.sequence += 1;
        }
        Event::default().event(kind).data(value.to_string())
    }
    fn chunk(&mut self, chunk: Value) -> Result<Vec<Event>, String> {
        let mut events = Vec::new();
        if !self.started {
            if matches!(self.protocol, Protocol::Anthropic) {
                self.input_tokens = chunk["nxrt_prompt_tokens"]
                    .as_u64()
                    .and_then(|count| usize::try_from(count).ok())
                    .or(self.input_tokens);
                if self.input_tokens.is_none() {
                    return Err("admitted chat stream lacks its exact prompt token count".into());
                }
            }
            self.id = chunk["id"]
                .as_str()
                .ok_or("chat stream initial chunk has no id")?
                .to_string();
            self.model = chunk["model"].as_str().unwrap_or_default().to_string();
            self.created = chunk["created"].as_u64().unwrap_or(0);
            self.started = true;
            match self.protocol {
                Protocol::Responses=>{
                    let response=json!({"id":format!("resp_{}",self.id),"object":"response","status":"in_progress","model":self.model,"output":[]});
                    events.push(self.event("response.created",json!({"response":response})));
                    events.push(self.event("response.in_progress",json!({"response":response})));
                }
                Protocol::Anthropic=>events.push(self.event("message_start",json!({"message":{"id":format!("msg_{}",self.id),"type":"message","role":"assistant","model":self.model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":self.input_tokens,"output_tokens":0}}}))),
            }
        }
        if !chunk["usage"].is_null() {
            self.usage = chunk["usage"].clone();
        }
        for choice in chunk["choices"].as_array().into_iter().flatten() {
            if let Some(reason) = choice["finish_reason"].as_str() {
                self.finish = reason.to_string();
                self.stop_sequence = choice["stop_sequence"].clone();
            }
            let delta = &choice["delta"];
            if let Some(text) = delta["content"].as_str()
                && !text.is_empty()
            {
                let position = self.position("text".into());
                self.retain(text.len())?;
                if !self.text_started {
                    self.start_block(position, &mut events);
                    self.text_started = true;
                    match self.protocol {
                        Protocol::Responses => {
                            events.push(self.event("response.output_item.added",json!({"output_index":position,"item":{"id":format!("msg_{}",self.id),"type":"message","status":"in_progress","role":"assistant","content":[]}})));
                            events.push(self.event("response.content_part.added",json!({"output_index":position,"item_id":format!("msg_{}",self.id),"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}})));
                        }
                        Protocol::Anthropic => events.push(self.event(
                            "content_block_start",
                            json!({"index":position,"content_block":{"type":"text","text":""}}),
                        )),
                    }
                }
                self.text.push_str(text);
                match self.protocol {
                    Protocol::Responses=>events.push(self.event("response.output_text.delta",json!({"output_index":position,"item_id":format!("msg_{}",self.id),"content_index":0,"delta":text}))),
                    Protocol::Anthropic=>events.push(self.event("content_block_delta",json!({"index":position,"delta":{"type":"text_delta","text":text}}))),
                }
            }
            if let Some(reasoning) = delta["reasoning_content"].as_str()
                && !reasoning.is_empty()
            {
                let position = self.position("reasoning".into());
                self.retain(reasoning.len())?;
                if !self.reasoning_started {
                    self.start_block(position, &mut events);
                    self.reasoning_started = true;
                    if matches!(self.protocol, Protocol::Responses) {
                        events.push(self.event("response.output_item.added", json!({"output_index":position,"item":{"id":format!("rs_{}",self.id),"type":"reasoning","summary":[]}})));
                        events.push(self.event("response.reasoning_summary_part.added", json!({"output_index":position,"item_id":format!("rs_{}",self.id),"summary_index":0,"part":{"type":"summary_text","text":""}})));
                    }
                    if matches!(self.protocol, Protocol::Anthropic) {
                        events.push(self.event(
                            "content_block_start",
                            json!({"index":position,"content_block":{"type":"thinking","thinking":""}}),
                        ));
                    }
                }
                self.reasoning.push_str(reasoning);
                if matches!(self.protocol, Protocol::Responses) {
                    events.push(self.event("response.reasoning_summary_text.delta", json!({"output_index":position,"item_id":format!("rs_{}",self.id),"summary_index":0,"delta":reasoning})));
                }
                if matches!(self.protocol, Protocol::Anthropic) {
                    events.push(self.event(
                        "content_block_delta",
                        json!({"index":position,"delta":{"type":"thinking_delta","thinking":reasoning}}),
                    ));
                }
            }
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                let index =
                    usize::try_from(call["index"].as_u64().ok_or("tool delta lacks index")?)
                        .map_err(|_| "tool index exceeds platform capacity")?;
                let position = self.position(format!("tool:{index}"));
                let new = !self.calls.contains_key(&index);
                self.retain(call.to_string().len())?;
                if new {
                    self.start_block(position, &mut events);
                }
                let current = self.calls.entry(index).or_insert_with(
                    || json!({"id":"","type":"function","function":{"name":"","arguments":""}}),
                );
                if let Some(id) = call["id"].as_str() {
                    current["id"] = json!(id);
                }
                if let Some(name) = call["function"]["name"].as_str() {
                    current["function"]["name"] = json!(name);
                }
                let arguments = call["function"]["arguments"].as_str().unwrap_or("");
                let previous = current["function"]["arguments"].as_str().unwrap_or("");
                current["function"]["arguments"] = json!(format!("{previous}{arguments}"));
                let id = current["id"].clone();
                let name = current["function"]["name"].clone();
                if new {
                    match self.protocol {
                    Protocol::Responses=>events.push(self.event("response.output_item.added",json!({"output_index":position,"item":{"id":format!("fc_{}",id.as_str().unwrap_or_default()),"type":"function_call","call_id":id,"name":name,"arguments":"","status":"in_progress"}}))),
                    Protocol::Anthropic=>events.push(self.event("content_block_start",json!({"index":position,"content_block":{"type":"tool_use","id":id,"name":name,"input":{}}}))),
                }
                }
                if !arguments.is_empty() {
                    match self.protocol {
                    Protocol::Responses=>events.push(self.event("response.function_call_arguments.delta",json!({"output_index":position,"item_id":format!("fc_{}",id.as_str().unwrap_or_default()),"delta":arguments}))),
                    Protocol::Anthropic=>events.push(self.event("content_block_delta",json!({"index":position,"delta":{"type":"input_json_delta","partial_json":arguments}}))),
                }
                }
            }
        }
        Ok(events)
    }
    fn done(&mut self) -> Result<Vec<Event>, String> {
        if self.finish.is_empty() || self.usage.is_null() {
            return Err("chat stream ended without a committed finish reason and usage".into());
        }
        let chat = json!({"id":self.id,"created":self.created,"model":self.model,"usage":self.usage,"choices":[{"finish_reason":self.finish,"stop_sequence":self.stop_sequence,"message":{"content":self.text,"reasoning_content":if self.reasoning.is_empty(){Value::Null}else{json!(self.reasoning)},"tool_calls":self.calls.values().collect::<Vec<_>>()}}]});
        let mut final_value = buffered(&chat, self.protocol).map_err(|err| err.message)?;
        let field = match self.protocol {
            Protocol::Responses => "output",
            Protocol::Anthropic => "content",
        };
        if let Some(items) = final_value[field].as_array_mut() {
            items.sort_by_key(|item| {
                let key = match item["type"].as_str() {
                    Some("message" | "text") => "text".to_string(),
                    Some("reasoning" | "thinking") => "reasoning".to_string(),
                    _ => {
                        let id = if matches!(self.protocol, Protocol::Responses) {
                            &item["call_id"]
                        } else {
                            &item["id"]
                        };
                        self.calls
                            .iter()
                            .find(|(_, call)| &call["id"] == id)
                            .map(|(index, _)| format!("tool:{index}"))
                            .unwrap_or_default()
                    }
                };
                self.positions
                    .iter()
                    .position(|position| position == &key)
                    .unwrap_or(usize::MAX)
            });
        }
        let mut events = Vec::new();
        match self.protocol {
            Protocol::Responses => {
                for (index, item) in final_value["output"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    if item["type"] == "message" {
                        events.push(self.event("response.output_text.done",json!({"output_index":index,"item_id":item["id"],"content_index":0,"text":self.text})));
                        events.push(self.event("response.content_part.done",json!({"output_index":index,"item_id":item["id"],"content_index":0,"part":item["content"][0]})));
                    } else if item["type"] == "reasoning" {
                        events.push(self.event("response.reasoning_summary_text.done",json!({"output_index":index,"item_id":item["id"],"summary_index":0,"text":self.reasoning})));
                        events.push(self.event("response.reasoning_summary_part.done",json!({"output_index":index,"item_id":item["id"],"summary_index":0,"part":item["summary"][0]})));
                    } else if item["type"] == "function_call" {
                        events.push(self.event("response.function_call_arguments.done",json!({"output_index":index,"item_id":item["id"],"arguments":item["arguments"]})));
                    }
                    events.push(self.event(
                        "response.output_item.done",
                        json!({"output_index":index,"item":item}),
                    ));
                }
                let kind = if final_value["status"] == "incomplete" {
                    "response.incomplete"
                } else {
                    "response.completed"
                };
                events.push(self.event(kind, json!({"response":final_value})));
            }
            Protocol::Anthropic => {
                if let Some(position) = self.open_block.take() {
                    events.push(self.event("content_block_stop", json!({"index":position})));
                }
                events.push(self.event("message_delta",json!({"delta":{"stop_reason":final_value["stop_reason"],"stop_sequence":final_value["stop_sequence"]},"usage":final_value["usage"]})));
                events.push(self.event("message_stop", json!({})));
            }
        }
        Ok(events)
    }
}
struct AdaptedStreamFailure {
    status: StatusCode,
    kind: Option<String>,
    message: String,
}
impl From<String> for AdaptedStreamFailure {
    fn from(message: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: None,
            message,
        }
    }
}
impl From<&str> for AdaptedStreamFailure {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

fn stream_response(response: Response, protocol: Protocol) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(16);
    tokio::spawn(async move {
        let mut source = response.into_body().into_data_stream();
        let mut pending = Vec::new();
        let mut encoder = StreamEncoder::new(protocol, None);
        let mut completed = false;
        let outcome: Result<(), AdaptedStreamFailure> = async {
            loop {
                let next = tokio::select! {_ = tx.closed()=>return Ok(()),next=source.next()=>next};
                let Some(bytes) = next else { break };
                let bytes = bytes.map_err(|err| format!("chat stream body failed: {err}"))?;
                pending.extend_from_slice(&bytes);
                if pending.len() > 2 * 1024 * 1024 {
                    return Err("chat SSE event exceeded 2 MiB; reduce response size".into());
                }
                while let Some(end) = pending.windows(2).position(|pair| pair == b"\n\n") {
                    let frame = pending.drain(..end + 2).collect::<Vec<_>>();
                    let frame = std::str::from_utf8(&frame)
                        .map_err(|err| format!("invalid chat SSE UTF-8: {err}"))?;
                    let data = frame
                        .lines()
                        .filter_map(|line| line.strip_prefix("data: "))
                        .collect::<Vec<_>>()
                        .join("\n");
                    if data.is_empty() {
                        continue;
                    }
                    let events = if data == "[DONE]" {
                        completed = true;
                        encoder.done()?
                    } else {
                        let value: Value = serde_json::from_str(&data)
                            .map_err(|err| format!("invalid chat SSE JSON: {err}"))?;
                        if !value["error"].is_null() {
                            let status=value["error"]["code"].as_u64().and_then(|code| u16::try_from(code).ok()).and_then(|code| StatusCode::from_u16(code).ok()).filter(|status| status.is_client_error() || status.is_server_error()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                            return Err(AdaptedStreamFailure {status,kind:value["error"]["type"].as_str().map(str::to_owned),message:value["error"]["message"].as_str().unwrap_or("chat generation failed; inspect the server log before retrying").to_string()});
                        }
                        encoder.chunk(value)?
                    };
                    for event in events {
                        if tx.send(Ok(event)).await.is_err() {
                            return Ok(());
                        }
                    }
                }
            }
            if !completed {
                return Err("chat stream ended before [DONE]; retry the request".into());
            }
            Ok(())
        }
        .await;
        if let Err(failure) = outcome {
            let kind = match protocol {
                Protocol::Anthropic => anthropic_error_kind(failure.status),
                Protocol::Responses => failure.kind.as_deref().unwrap_or("server_error"),
            };
            let mut error = json!({"type":kind,"message":failure.message});
            if matches!(protocol, Protocol::Responses) {
                error["code"] = json!(failure.status.as_u16());
            }
            let event = encoder.event("error", json!({"error":error}));
            let _ = tx.send(Ok(event)).await;
        }
    });
    Sse::new(ReceiverStream::new(rx)).into_response()
}

#[cfg(test)]
mod transport_tests {
    use super::*;

    #[test]
    fn protocols_lower_history_and_reject_unsupported_state() {
        let request = response_request(json!({"input":[{"type":"function_call","call_id":"call1","name":"weather","arguments":"{}"},{"type":"function_call_output","call_id":"call1","output":"sunny"}],"instructions":"help","max_output_tokens":12}).as_object().unwrap().clone()).unwrap();
        assert_eq!(request.messages.len(), 3);
        assert_eq!(request.messages[2].tool_call_id.as_deref(), Some("call1"));
        assert!(
            response_request(
                json!({"input":"hi","previous_response_id":"old"})
                    .as_object()
                    .unwrap()
                    .clone()
            )
            .is_err()
        );
        let request = anthropic_request(json!({"messages":[{"role":"user","content":"hi"}],"max_tokens":12,"system":[{"type":"text","text":"help"}]}).as_object().unwrap().clone()).unwrap();
        assert_eq!(request.messages.len(), 2);
        assert!(anthropic_request(json!({"messages":[],"max_tokens":12,"thinking":{"type":"enabled","budget_tokens":10}}).as_object().unwrap().clone()).is_err());
    }

    async fn wire(events: Vec<Event>) -> String {
        let response = Sse::new(tokio_stream::iter(
            events.into_iter().map(Ok::<_, Infallible>),
        ))
        .into_response();
        String::from_utf8(
            to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn streaming_indices_usage_and_stop_sequence_survive_conversion() {
        for protocol in [Protocol::Responses, Protocol::Anthropic] {
            let mut encoder = StreamEncoder::new(protocol, Some(19));
            let mut events = encoder.chunk(json!({"id":"x","model":"fixture","created":1,"choices":[{"delta":{"reasoning_content":"think"}}]})).unwrap();
            events.extend(
                encoder
                    .chunk(json!({"choices":[{"delta":{"content":"hello"}}]}))
                    .unwrap(),
            );
            events.extend(encoder.chunk(json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"weather","arguments":"{}"}}]}}]})).unwrap());
            events.extend(encoder.chunk(json!({"choices":[{"delta":{},"finish_reason":"stop","stop_sequence":"END"}],"usage":{"prompt_tokens":19,"completion_tokens":7,"total_tokens":26}})).unwrap());
            assert_eq!(encoder.positions, vec!["reasoning", "text", "tool:0"]);
            events.extend(encoder.done().unwrap());
            let result = wire(events).await;
            assert!(result.contains("\"input_tokens\":19"));
            match protocol {
                Protocol::Responses => {
                    assert!(result.contains("response.reasoning_summary_text.delta"));
                    assert!(result.contains("\"output_index\":2"));
                    assert!(result.contains("response.completed"));
                }
                Protocol::Anthropic => {
                    assert!(result.contains("\"index\":2"));
                    assert!(result.contains("\"stop_reason\":\"tool_use\""));
                    assert!(result.contains("\"stop_sequence\":\"END\""));
                    assert!(result.contains("message_stop"));
                }
            }
        }
    }

    #[test]
    fn responses_rejoins_reasoning_and_parallel_calls_into_one_assistant_turn() {
        let body = json!({"input":[
            {"type":"reasoning","summary":[{"type":"summary_text","text":"checking"}]},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"I will check"}]},
            {"type":"function_call","call_id":"a","name":"weather","arguments":"{}"},
            {"type":"function_call","call_id":"b","name":"time","arguments":"{}"},
            {"type":"function_call_output","call_id":"b","output":"noon"},
            {"type":"function_call_output","call_id":"a","output":"sunny"}
        ]});
        let request = response_request(body.as_object().unwrap().clone()).unwrap();
        assert_eq!(request.messages.len(), 3);
        assert_eq!(
            request.messages[0].reasoning_content.as_deref(),
            Some("checking")
        );
        assert_eq!(request.messages[0].tool_calls.as_ref().unwrap().len(), 2);
        assert_eq!(request.messages[1].tool_call_id.as_deref(), Some("b"));
    }

    #[tokio::test]
    async fn messages_error_conversion_preserves_retry_headers_and_valid_content_length() {
        let response = (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, "3")],
            Json(json!({"error":{"message":"queue is full","type":"resource_limit_error"}})),
        )
            .into_response();
        let response = anthropic_error_response(response).await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[header::RETRY_AFTER], "3");
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["type"], "rate_limit_error");
        assert_eq!(value["error"]["message"], "queue is full");
        let mut encoder = StreamEncoder::new(Protocol::Anthropic, Some(1));
        let event=encoder.event("error",json!({"error":{"type":anthropic_error_kind(StatusCode::INTERNAL_SERVER_ERROR),"message":"failed"}}));
        let event = wire(vec![event]).await;
        assert!(event.contains("\"type\":\"api_error\""));
    }

    #[tokio::test]
    async fn adapted_stream_errors_preserve_source_status_and_protocol_kind() {
        for protocol in [Protocol::Responses, Protocol::Anthropic] {
            let source = Response::new(Body::from(
                "event: error\ndata: {\"error\":{\"code\":429,\"type\":\"resource_limit_error\",\"message\":\"capacity exhausted\"}}\n\n",
            ));
            let response = stream_response(source, protocol);
            let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
            let text = std::str::from_utf8(&bytes).unwrap();
            let value: Value = serde_json::from_str(
                text.lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(value["type"], "error");
            assert_eq!(value["error"]["message"], "capacity exhausted");
            match protocol {
                Protocol::Responses => {
                    assert_eq!(value["error"]["type"], "resource_limit_error");
                    assert_eq!(value["error"]["code"], 429);
                }
                Protocol::Anthropic => {
                    assert_eq!(value["error"]["type"], "rate_limit_error");
                    assert!(value["error"].get("code").is_none());
                }
            }
            assert!(!text.contains("response.completed"));
        }
    }

    #[test]
    fn incomplete_upstream_stream_is_never_reported_as_success() {
        let mut encoder = StreamEncoder::new(Protocol::Responses, None);
        encoder.chunk(json!({"id":"x","model":"fixture","created":1,"choices":[{"delta":{"content":"partial"}}]})).unwrap();
        assert!(
            encoder
                .done()
                .unwrap_err()
                .contains("finish reason and usage")
        );
    }
}
