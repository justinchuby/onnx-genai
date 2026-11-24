//! llama.cpp host-tool discovery, invocation and bounded SSE transport.
use super::*;
use crate::tools::{BuiltinTool, Job, ResponseBytes, ToolValue, Tools};
use axum::{
    body::{Body, Bytes, to_bytes},
    extract::FromRequest,
};
use serde_json::{Value, json};
use std::{
    pin::Pin,
    task::{Context as TaskContext, Poll},
    time::Duration,
};
use tokio_stream::Stream;

const REQUEST_BYTES: usize = 32 * 1024 * 1024;
#[derive(Deserialize)]
struct Call {
    tool: String,
    #[serde(default)]
    params: Option<Box<serde_json::value::RawValue>>,
    #[serde(default)]
    stream: bool,
}
pub(crate) struct AdmittedCall {
    call: Call,
    tool: BuiltinTool,
    tools: Arc<Tools>,
    job: Arc<Job>,
    cwd: std::path::PathBuf,
    base64: bool,
}
fn owner(state: &AppState, headers: &HeaderMap) -> Result<Arc<Tools>, ApiError> {
    let tools=state.tools.clone().ok_or_else(||ApiError::forbidden("server tools are disabled; start with --tools all or a comma-separated list of trusted tools"))?;
    if let Some(origin) = headers.get(header::ORIGIN) {
        let origin = origin
            .to_str()
            .map_err(|_| ApiError::invalid_request("Origin must be a valid HTTP origin"))?;
        if !tools.origin_allowed(origin) {
            return Err(ApiError::forbidden(
                "host tools refuse this browser origin; serve the frontend from loopback or explicitly configure --tools-allow-origin",
            ));
        }
    }
    Ok(tools)
}
fn admission_error(error: anyhow::Error) -> ApiError {
    let mut failure = ApiError::unavailable(format!("{error:#}"));
    failure.retry_after_secs = Some(1);
    failure
}
fn header_text<'a>(headers: &'a HeaderMap, name: &str) -> Result<Option<&'a str>, ApiError> {
    headers
        .get(name)
        .map(|v| {
            v.to_str().map_err(|_| {
                ApiError::invalid_request(format!("{name} must be valid Unicode text"))
            })
        })
        .transpose()
}
impl FromRequest<AppState> for AdmittedCall {
    type Rejection = ApiError;
    async fn from_request(request: Request, state: &AppState) -> Result<Self, Self::Rejection> {
        let headers = request.headers();
        let tools = owner(state, headers)?;
        let content_type = header_text(headers, "content-type")?
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if content_type != "application/json"
            && !(content_type.starts_with("application/") && content_type.ends_with("+json"))
        {
            let mut error =
                ApiError::invalid_request("/tools expects Content-Type: application/json");
            error.status = StatusCode::UNSUPPORTED_MEDIA_TYPE;
            return Err(error);
        }
        if header_text(headers, "x-tool-runtime")?.is_some_and(|v| !v.is_empty()) {
            return Err(ApiError::invalid_request(
                "x-tool-runtime is not supported by the host executor; omit this header rather than silently falling back from an isolated runtime",
            ));
        }
        let cwd = tools.cwd(header_text(headers, "x-tool-cwd")?);
        let base64 = header_text(headers, "x-resp-type")? == Some("base64");
        let bound = header_text(headers, "content-length")?
            .map(|s| s.parse::<usize>())
            .transpose()
            .map_err(|_| {
                ApiError::invalid_request("Content-Length must be a nonnegative byte count")
            })?
            .unwrap_or_else(|| tools.request_bound(REQUEST_BYTES));
        if bound > REQUEST_BYTES {
            let mut error = ApiError::invalid_request(format!(
                "tool request exceeds {REQUEST_BYTES} bytes; split the write or edit"
            ));
            error.status = StatusCode::PAYLOAD_TOO_LARGE;
            return Err(error);
        }
        let mut job = tools.admit(bound).map_err(admission_error)?;
        let bytes = tokio::time::timeout(
            Duration::from_secs(60),
            to_bytes(request.into_body(), bound),
        )
        .await
        .map_err(|_| {
            ApiError::invalid_request(
                "tool request body timed out; send the complete JSON request within 60 seconds",
            )
        })?
        .map_err(|error| {
            let error=anyhow::Error::new(error);
            let oversized=error.chain().any(|cause|cause.is::<http_body_util::LengthLimitError>());
            let mut response=ApiError::invalid_request(if oversized {format!("tool request exceeds its admitted {bound} byte limit: {error:#}; correct Content-Length, split the request, or increase the host-tool budget")}else{format!("cannot read tool request body: {error:#}; resend a complete JSON request")});
            if oversized {response.status=StatusCode::PAYLOAD_TOO_LARGE;}response
        })?;
        let call: Call = serde_json::from_slice(&bytes).map_err(|error| {
            ApiError::invalid_request(format!(
                "invalid /tools request: {error}; supply tool, params and optional stream"
            ))
        })?;
        job.retain_body_credit(bytes.len())
            .map_err(admission_error)?;
        let tool = BuiltinTool::parse(&call.tool)
            .filter(|tool| tools.enabled(*tool))
            .ok_or_else(|| {
                let mut error = ApiError::invalid_request(
                    "unknown or disabled tool; GET /tools lists enabled tool names",
                );
                error.status = StatusCode::NOT_FOUND;
                error
            })?;
        if call.stream && !tool.streaming() {
            let mut error = ApiError::invalid_request(format!(
                "tool '{}' does not support stream=true",
                call.tool
            ));
            error.status = StatusCode::NOT_FOUND;
            return Err(error);
        }
        job.limit_time(
            if matches!(tool, BuiltinTool::FileGlobSearch | BuiltinTool::GrepSearch) {
                Duration::from_secs(15)
            } else {
                Duration::from_secs(60)
            },
        );
        Ok(Self {
            call,
            tool,
            tools,
            job: Arc::new(job),
            cwd,
            base64,
        })
    }
}
struct CancelOnDrop(Option<Arc<Job>>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(job) = &self.0 {
            job.cancel();
        }
    }
}
struct ToolStream {
    receiver: ReceiverStream<Value>,
    _cancel: CancelOnDrop,
}
impl Stream for ToolStream {
    type Item = Result<Event, Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.receiver)
            .poll_next(cx)
            .map(|value| value.map(|value| Ok(Event::default().data(value.to_string()))))
    }
}
fn encode(value: ToolValue) -> Result<Response, ApiError> {
    // A fallible writer avoids an unchecked serializer allocation on a large
    // base64 result. Credit follows the body bytes through clones and disconnect.
    struct Writer(Vec<u8>);
    impl std::io::Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0
                .try_reserve(bytes.len())
                .map_err(std::io::Error::other)?;
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = Writer(Vec::new());
    serde_json::to_writer(&mut writer, &value.value).map_err(|error| {
        ApiError::internal(format!(
            "serialize host-tool response: {error}; free host memory or reduce tool output"
        ))
    })?;
    let bytes = Bytes::from_owner(ResponseBytes {
        bytes: writer.0,
        _leases: value.leases,
        job: value.job,
    });
    Ok((
        [(header::CONTENT_TYPE, "application/json")],
        Body::from(bytes),
    )
        .into_response())
}
pub(crate) async fn list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let tools = owner(&state, &headers)?;
    let job = Arc::new(tools.admit(0).map_err(admission_error)?);
    encode(ToolValue {
        value: tools.definitions(),
        leases: Vec::new(),
        job,
    })
}
pub(crate) async fn invoke(admitted: AdmittedCall) -> Result<Response, ApiError> {
    let AdmittedCall {
        call,
        tool,
        tools,
        job,
        cwd,
        base64,
    } = admitted;
    let mut cancel = CancelOnDrop(Some(job.clone()));
    if call.stream {
        let (sender, receiver) = mpsc::channel(16);
        let producer_job = job.clone();
        tokio::spawn(async move {
            let params = call.params.as_ref().map_or("{}", |p| p.get());
            let result = tools
                .invoke(
                    tool,
                    params,
                    cwd,
                    base64,
                    producer_job,
                    Some(sender.clone()),
                )
                .await;
            let final_event = match result {
                Ok(_) => json!({"done":true}),
                Err(error) => json!({"done":true,"error":format!("{error:#}")}),
            };
            // A disconnected/backpressured consumer cannot retain an unbounded
            // producer. Pending data already occupies the fixed queue budget.
            let _ = tokio::time::timeout(Duration::from_secs(1), sender.send(final_event)).await;
        });
        return Ok(Sse::new(ToolStream {
            receiver: ReceiverStream::new(receiver),
            _cancel: cancel,
        })
        .into_response());
    }
    let params = call.params.as_ref().map_or("{}", |p| p.get());
    let result = tools
        .invoke(tool, params, cwd, base64, job.clone(), None)
        .await;
    let value = match result {
        Ok(result) => result,
        Err(error) if error.is::<onnx_runtime_memory_api::MemoryError>() => {
            return Err(admission_error(error));
        }
        Err(error) if error.is::<serde_json::Error>() => {
            return Err(ApiError::invalid_request(format!(
                "invalid tool parameters: {error:#}; GET /tools lists their schema"
            )));
        }
        Err(error) => ToolValue {
            value: json!({"error":format!("{error:#}")}),
            leases: Vec::new(),
            job,
        },
    };
    let response = encode(value)?;
    cancel.0.take();
    Ok(response)
}
