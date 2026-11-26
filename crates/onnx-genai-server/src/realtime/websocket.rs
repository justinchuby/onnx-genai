//! WebSocket transport over one exclusively owned native realtime connection.
use super::{
    RealtimeConnection, RealtimeInputOwned, RealtimeLimits,
    conditioning::{Conditioning, PromptOperation},
    conversation::{CompletedToolCall, Conversation, ItemId, ResponseId, ResponseKind},
    events::{ResponseEnd, ResponseEvents, TranscriptionEvents},
    pcm::PcmInput,
    settings::{Capabilities, Modalities, OutputBudget, ResponseOptions, SessionSettings},
    tools::ToolArguments,
};
use crate::{
    registry::ModelHandle,
    routes::{ApiError, resolve_model},
    state::AppState,
};
use anyhow::{Context, ensure};
use axum::{
    body::Body,
    extract::{Query, Request, State},
    response::Response,
};
use tokio_tungstenite::{
    WebSocketStream,
    tungstenite::{
        Message,
        protocol::{Role, WebSocketConfig},
    },
};
type WebSocket = WebSocketStream<hyper_util::rt::TokioIo<hyper::upgrade::Upgraded>>;
use futures_util::{SinkExt, StreamExt};
use onnx_genai_engine::{
    realtime::RealtimeTokenBudget,
    tokenization::{SpecialTokenHandling, TokenTextDecoder},
};
use onnx_genai_metadata::{
    RealtimeMode, ToolCallStream, ToolParseOutcome, realtime::RealtimeContract,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    num::{NonZeroU32, NonZeroUsize},
    sync::Arc,
};
use tokio::sync::{mpsc, oneshot};

#[derive(Deserialize)]
pub(crate) struct RealtimeQuery {
    #[serde(default)]
    model: String,
}

pub(crate) async fn upgrade(
    State(state): State<AppState>,
    Query(query): Query<RealtimeQuery>,
    mut request: Request,
) -> Result<Response, ApiError> {
    let mut handshake=tokio_tungstenite::tungstenite::handshake::server::create_response_with_body(&request,Body::empty)
        .map_err(|error|ApiError::invalid_request(format!("invalid realtime WebSocket upgrade: {error}; send a GET upgrade with Sec-WebSocket-Version: 13 and a valid Sec-WebSocket-Key")))?;
    if request
        .headers()
        .get("sec-websocket-protocol")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.split(',').any(|value| value.trim() == "realtime"))
    {
        handshake.headers_mut().insert(
            "sec-websocket-protocol",
            axum::http::HeaderValue::from_static("realtime"),
        );
    }
    let model = resolve_model(&state.registry, &query.model).await?;
    let contract=model.engine.realtime.clone().ok_or_else(||ApiError::invalid_request("selected model has no realtime profile; load a package declaring native realtime session quanta"))?;
    let limits = state.config.realtime.clone();
    if contract.spec().input_frame_samples > limits.native_samples.get() {
        return Err(ApiError::invalid_request(
            "model realtime input frame exceeds ServerConfig.realtime.native_samples; select a smaller frame or increase that audio memory budget",
        ));
    }
    let rate = NonZeroU32::new(contract.spec().input_sample_rate_hz).ok_or_else(|| {
        ApiError::internal("admitted realtime profile has zero input rate; repair the package")
    })?;
    let budget = NonZeroU32::new(state.config.max_output_tokens.min(i32::MAX as usize) as u32)
        .ok_or_else(|| ApiError::internal("server max_output_tokens must be positive"))?;
    let settings = SessionSettings::new(
        contract.spec().default_mode,
        rate,
        OutputBudget::Tokens(budget),
    );
    let conditioning = Conditioning::new(
        contract.spec(),
        onnx_genai_ort::TemplateLimits {
            max_output_bytes: limits.prompt_bytes,
            max_operations: limits.template_operations,
        },
        limits.prompt_tokens,
    );
    let initial_tokens = conditioning
        .tokens(
            &model.tokenizer,
            &contract,
            &settings,
            PromptOperation::SessionStart,
            "",
            None,
        )
        .map_err(|error| {
            ApiError::internal(format!("admit model realtime prompting: {error:#}"))
        })?;
    let (native, initial_output) = model
        .engine
        .open_realtime(settings.mode, initial_tokens)
        .await
        .map_err(|error| {
            ApiError::unavailable(format!("open native realtime session: {error:#}"))
        })?;
    let id = crate::completion_control::new_id("sess_")?;
    let max_message = limits.message_bytes.get();
    let runner = Runner {
        model,
        contract,
        limits,
        settings,
        conditioning,
        native,
        id,
        conversation: Conversation::default(),
        pcm: None,
        active: None,
        configured: true,
        needs_input: initial_output.needs_input,
        pending_commit: false,
        tool_arguments: ToolArguments::default(),
    };
    let upgraded = hyper::upgrade::on(&mut request);
    let mut websocket_config = WebSocketConfig::default();
    websocket_config.max_message_size = Some(max_message);
    websocket_config.max_frame_size = Some(max_message);
    websocket_config.write_buffer_size = 0;
    websocket_config.max_write_buffer_size=runner.limits.text_bytes.get().checked_mul(8).and_then(|bytes|bytes.checked_add(1024)).ok_or_else(||ApiError::internal("realtime text budget exceeds websocket write-buffer address space; reduce ServerConfig.realtime.text_bytes"))?;
    tokio::spawn(async move {
        match upgraded.await {
            Ok(stream) => {
                serve(
                    WebSocketStream::from_raw_socket(
                        hyper_util::rt::TokioIo::new(stream),
                        Role::Server,
                        Some(websocket_config),
                    )
                    .await,
                    runner,
                )
                .await
            }
            Err(error) => {
                tracing::debug!(%error,"realtime HTTP upgrade closed before websocket initialization")
            }
        }
    });
    Ok(handshake)
}
enum WireOutput {
    Event(Value),
    Close,
}
struct Outgoing {
    value: WireOutput,
    flushed: Option<oneshot::Sender<()>>,
}
type Incoming = Result<Value, String>;
async fn serve(socket: WebSocket, mut runner: Runner) {
    let (mut sink, mut source) = socket.split();
    let (input, mut incoming) = mpsc::channel::<Incoming>(1);
    let (output, mut outgoing) = mpsc::channel::<Outgoing>(16);
    let timeout = runner.limits.write_timeout;
    let close_output = output.clone();
    let read = async move {
        while let Some(message) = source.next().await {
            let event=match message? {
                Message::Text(text)=>serde_json::from_str(&text).map_err(|error|format!("realtime event must be valid JSON: {error}")),
                Message::Close(_)=>{
                    let (flushed,reply)=oneshot::channel();
                    let _=close_output.send(Outgoing {value:WireOutput::Close,flushed:Some(flushed)}).await;
                    let _=tokio::time::timeout(timeout,reply).await;
                    break;
                },
                Message::Ping(_)|Message::Pong(_)|Message::Frame(_)=>continue,
                Message::Binary(_)=>Err("realtime client events must be JSON text frames; send PCM16 as base64 in input_audio_buffer.append.audio".into()),
            };
            if input.send(event).await.is_err() {
                break;
            }
        }
        Ok::<(), anyhow::Error>(())
    };
    let write = async move {
        while let Some(event) = outgoing.recv().await {
            let closing = match event.value {
                WireOutput::Event(mut value) => {
                    value["event_id"] = json!(id("event_")?);
                    tokio::time::timeout(
                        timeout,
                        sink.send(Message::Text(value.to_string().into())),
                    )
                    .await
                    .context("realtime client did not consume output within its write timeout")??;
                    false
                }
                WireOutput::Close => {
                    tokio::time::timeout(timeout, sink.close())
                        .await
                        .context("realtime close handshake exceeded its write timeout")??;
                    true
                }
            };
            if let Some(flushed) = event.flushed {
                let _ = flushed.send(());
            }
            if closing {
                break;
            }
        }
        Ok::<(), anyhow::Error>(())
    };
    let protocol = async {
        runner.emit(&output,json!({"type":"session.created","session":runner.settings.snapshot(&runner.id,&runner.model.id,&runner.contract)})).await?;
        runner.run(&mut incoming, &output).await
    };
    // Dropping the losing protocol future drops its native connection. The
    // worker retains its reservation until cancellation and native cleanup end.
    let result = tokio::select! {result=read=>result,result=write=>result,result=protocol=>result};
    if let Err(error) = result {
        tracing::debug!(error=%format!("{error:#}"),"realtime connection closed");
    }
}
fn id(prefix: &str) -> anyhow::Result<String> {
    crate::completion_control::new_id(prefix)
        .map_err(|error| anyhow::anyhow!("allocate realtime ID from host randomness: {error:?}"))
}
fn string<'a>(object: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("realtime field {key} must be a string"))
}
enum FailureKind {
    Protocol(super::conversation::ProtocolErrorKind),
    Native,
}
struct Failure {
    message: String,
    kind: FailureKind,
}
impl Failure {
    fn fatal(&self) -> bool {
        matches!(self.kind, FailureKind::Native)
    }
    fn error_type(&self) -> &'static str {
        match self.kind {
            FailureKind::Protocol(kind) => kind.error_type(),
            FailureKind::Native => "server_error",
        }
    }
    fn invalid(error: impl std::fmt::Display) -> Self {
        Self {
            message: error.to_string(),
            kind: FailureKind::Protocol(super::conversation::ProtocolErrorKind::InvalidRequest),
        }
    }
    fn native(error: anyhow::Error) -> Self {
        Self {
            message: format!("{error:#}"),
            kind: FailureKind::Native,
        }
    }
}
impl From<super::conversation::ProtocolError> for Failure {
    fn from(error: super::conversation::ProtocolError) -> Self {
        Self {
            kind: FailureKind::Protocol(error.kind),
            message: error.to_string(),
        }
    }
}
struct ToolText {
    protocol: onnx_genai_metadata::ToolProtocol,
    stream: ToolCallStream,
    raw: String,
    published: usize,
    bindings: Vec<ToolBinding>,
}
struct ToolBinding {
    wire_id: String,
    item_id: String,
    name: String,
    model_id: Option<String>,
}
impl ToolText {
    fn fragments(
        &mut self,
        parameters: &ToolArguments,
        renderer: &mut ResponseEvents,
    ) -> anyhow::Result<Vec<Value>> {
        let snapshots = self.stream.fragments(self.protocol)?;
        let admitted = parameters.admit_fragments(&snapshots)?;
        let mut events = Vec::new();
        for proof in admitted {
            let fragment = proof.fragment();
            if self.bindings.len() == fragment.index {
                let wire_id = match &fragment.id {
                    Some(id) => id.clone(),
                    None => id("call_")?,
                };
                ensure!(
                    !self
                        .bindings
                        .iter()
                        .any(|binding| binding.wire_id == wire_id),
                    "realtime model call ID collides with an already projected call; emit distinct function IDs"
                );
                self.bindings.push(ToolBinding {
                    wire_id,
                    item_id: id("item_")?,
                    name: fragment.name.clone(),
                    model_id: fragment.id.clone(),
                });
            }
            let binding = self.bindings.get_mut(fragment.index)
                .context("realtime function fragment skipped a preceding call identity; repair the model protocol parser")?;
            ensure!(
                binding.name == fragment.name
                    && binding
                        .model_id
                        .as_ref()
                        .is_none_or(|model_id| fragment.id.as_ref() == Some(model_id)),
                "realtime function fragment changed an already decoded model identity; repair the model tool protocol"
            );
            if binding.model_id.is_none() {
                binding.model_id.clone_from(&fragment.id);
            }
            events.extend(renderer.tool_fragment(&proof, &binding.wire_id, &binding.item_id)?);
        }
        Ok(events)
    }
}
enum Rendered {
    Response(ResponseEvents),
    Transcription(TranscriptionEvents),
}
struct Active {
    id: ResponseId,
    renderer: Rendered,
    decoder: TokenTextDecoder,
    tools: Option<ToolText>,
    tool_arguments: ToolArguments,
    budget: OutputBudget,
    used: u64,
    text_enabled: bool,
    audio_enabled: bool,
}
struct Runner {
    model: Arc<ModelHandle>,
    contract: Arc<RealtimeContract>,
    limits: RealtimeLimits,
    settings: SessionSettings,
    conditioning: Conditioning,
    native: RealtimeConnection,
    id: String,
    conversation: Conversation,
    pcm: Option<PcmInput>,
    active: Option<Active>,
    configured: bool,
    needs_input: bool,
    pending_commit: bool,
    tool_arguments: ToolArguments,
}
impl Runner {
    async fn emit(&self, out: &mpsc::Sender<Outgoing>, value: Value) -> anyhow::Result<()> {
        out.send(Outgoing {
            value: WireOutput::Event(value),
            flushed: None,
        })
        .await
        .context("realtime output writer closed")
    }
    async fn error(&self, out: &mpsc::Sender<Outgoing>, failure: &Failure) -> anyhow::Result<()> {
        let value = json!({"type":"error","error":{"type":failure.error_type(),"code":null,"message":failure.message,"param":null}});
        if failure.fatal() {
            let (flushed, reply) = oneshot::channel();
            out.send(Outgoing {
                value: WireOutput::Event(value),
                flushed: Some(flushed),
            })
            .await
            .context("realtime error writer closed")?;
            let _ = tokio::time::timeout(self.limits.write_timeout, reply).await;
            return Ok(());
        }
        self.emit(out, value).await
    }
    async fn run(
        &mut self,
        input: &mut mpsc::Receiver<Incoming>,
        out: &mpsc::Sender<Outgoing>,
    ) -> anyhow::Result<()> {
        loop {
            let asr_active = self
                .conversation
                .active()
                .is_some_and(|active| matches!(active.kind, ResponseKind::Transcription { .. }));
            if !(asr_active && self.pending_commit) {
                match input.try_recv() {
                    Ok(event) => {
                        if !self.event_result(event, out).await? {
                            return Ok(());
                        }
                        continue;
                    }
                    Err(mpsc::error::TryRecvError::Disconnected) => return Ok(()),
                    Err(mpsc::error::TryRecvError::Empty) => {}
                }
            }
            match self.drive_ready(asr_active, out).await {
                Ok(true) => continue,
                Ok(false) => {}
                Err(failure) => {
                    self.error(out, &failure).await?;
                    if failure.fatal() {
                        return Ok(());
                    }
                }
            }
            let Some(event) = input.recv().await else {
                return Ok(());
            };
            if !self.event_result(event, out).await? {
                return Ok(());
            }
        }
    }
    async fn drive_ready(
        &mut self,
        asr_active: bool,
        out: &mpsc::Sender<Outgoing>,
    ) -> Result<bool, Failure> {
        if self.budget_exhausted() {
            self.quantum(RealtimeInputOwned::Cancel, false, out).await?;
            return Ok(true);
        }
        if !asr_active {
            let frame = self
                .pcm
                .as_mut()
                .map(PcmInput::next_frame)
                .transpose()
                .map_err(Failure::native)?
                .flatten();
            if let Some(frame) = frame {
                self.configure().await?;
                self.quantum(RealtimeInputOwned::Audio(frame), false, out)
                    .await?;
                return Ok(true);
            }
            if self.pending_commit {
                self.pending_commit = false;
                self.commit(out).await?;
                return Ok(true);
            }
        }
        if self.active.is_some() && !self.needs_input {
            self.quantum(RealtimeInputOwned::Poll, false, out).await?;
            return Ok(true);
        }
        Ok(false)
    }
    async fn event_result(
        &mut self,
        event: Incoming,
        out: &mpsc::Sender<Outgoing>,
    ) -> anyhow::Result<bool> {
        if self.budget_exhausted()
            && let Err(failure) = self.quantum(RealtimeInputOwned::Cancel, false, out).await
        {
            self.error(out, &failure).await?;
            return Ok(false);
        }
        let result = match event {
            Ok(event) => self.event(event, out).await,
            Err(error) => Err(Failure::invalid(error)),
        };
        if let Err(failure) = result {
            self.error(out, &failure).await?;
            return Ok(!failure.fatal());
        }
        Ok(true)
    }
    async fn configure(&mut self) -> Result<(), Failure> {
        if self.configured {
            return Ok(());
        }
        let tokens = self
            .conditioning
            .tokens(
                &self.model.tokenizer,
                &self.contract,
                &self.settings,
                PromptOperation::SessionStart,
                "",
                None,
            )
            .map_err(Failure::native)?;
        self.native.set_budget(RealtimeTokenBudget::Unlimited);
        let output = self
            .native
            .step(RealtimeInputOwned::Configure {
                mode: self.settings.mode,
                tokens,
            })
            .await
            .map_err(Failure::native)?;
        self.needs_input = output.needs_input;
        self.configured = true;
        Ok(())
    }
    fn response_constraint(
        &self,
        arguments: &ToolArguments,
        options: &ResponseOptions,
    ) -> anyhow::Result<Option<onnx_genai_engine::GenerateConstraint>> {
        let request: crate::types::ChatCompletionRequest = serde_json::from_value(json!({
            "messages": [],
            "tools": crate::tool_protocol::responses_tools(Value::Array(arguments.definitions().to_vec()))?,
            "tool_choice": super::settings::choice_name(&options.tool_choice),
        }))?;
        match self
            .model
            .tool_protocol
            .as_ref()
            .map(|protocol| protocol.output_constraint(&request))
            .transpose()?
        {
            Some(crate::tool_protocol::ToolOutputConstraint::Set(constraint)) => Ok(constraint),
            _ => Ok(None),
        }
    }
    fn duplex_constraint(&self) -> Result<Option<onnx_genai_engine::GenerateConstraint>, Failure> {
        let constraint = self
            .response_constraint(&self.tool_arguments, &self.settings.response)
            .map_err(Failure::invalid)?;
        if self.settings.response.tool_choice == crate::types::ToolChoiceMode::Required {
            if constraint.is_none() {
                return Err(Failure::invalid(
                    "input-driven duplex tool_choice=required needs a declared protocol with native constraint-guided sampling; this protocol has no supported forcing plan",
                ));
            }
            if self.contract.spec().inputs.token_mask.is_none() {
                return Err(Failure::invalid(format!(
                    "input-driven duplex tool_choice=required needs profiles.{}.realtime.inputs.token_mask; export a workflow consuming the mask before selecting text tokens",
                    self.contract.profile()
                )));
            }
        }
        Ok(constraint)
    }
    fn new_pcm(&self) -> anyhow::Result<PcmInput> {
        PcmInput::new(
            self.settings.input_rate,
            NonZeroU32::new(self.contract.spec().input_sample_rate_hz)
                .context("admitted realtime native rate is zero")?,
            NonZeroUsize::new(self.contract.spec().input_frame_samples)
                .context("admitted realtime native frame is zero")?,
            self.limits.source_bytes,
            self.limits.native_samples,
        )
    }
    async fn open_output(&mut self, out: &mpsc::Sender<Outgoing>) -> Result<(), Failure> {
        let active = self
            .conversation
            .active()
            .context("native response has no admitted conversation response")
            .map_err(Failure::native)?
            .clone();
        let mut preserved = BTreeSet::new();
        let tools = if matches!(active.kind, ResponseKind::Realtime)
            && !self.tool_arguments.definitions().is_empty()
            && active.options.tool_choice != crate::types::ToolChoiceMode::None
        {
            let protocol = self
                .model
                .tool_protocol
                .as_ref()
                .context("realtime tools need a package.tool_protocol declaration")
                .map_err(Failure::native)?
                .parsed();
            for marker in protocol.syntax_markers() {
                let ids = self
                    .model
                    .tokenizer
                    .encode_with_options(marker, Default::default())
                    .map_err(anyhow::Error::new)
                    .map_err(Failure::native)?;
                preserved.extend(ids);
            }
            Some(ToolText {
                protocol,
                stream: ToolCallStream::default(),
                raw: String::new(),
                published: 0,
                bindings: Vec::new(),
            })
        } else {
            None
        };
        let renderer = match &active.kind {
            ResponseKind::Realtime => {
                let (renderer, created) = ResponseEvents::new(
                    active.id.as_str().into(),
                    id("item_").map_err(Failure::native)?,
                    active.options.modalities.json(),
                    self.limits.text_bytes,
                    self.limits.audio_event_samples,
                );
                self.emit(out, created).await.map_err(Failure::native)?;
                Rendered::Response(renderer)
            }
            ResponseKind::Transcription { item } => Rendered::Transcription(
                TranscriptionEvents::new(item.as_str().into(), self.limits.text_bytes),
            ),
        };
        self.active = Some(Active {
            id: active.id,
            renderer,
            decoder: if tools.is_some() {
                TokenTextDecoder::with_preserved_tokens(&preserved)
            } else {
                TokenTextDecoder::new(SpecialTokenHandling::Skip)
            },
            tools,
            tool_arguments: self.tool_arguments.clone(),
            budget: active.options.budget,
            used: 0,
            text_enabled: !matches!(active.options.modalities, Modalities::Audio),
            audio_enabled: !matches!(active.options.modalities, Modalities::Text),
        });
        Ok(())
    }
    async fn drain_pcm(&mut self, out: &mpsc::Sender<Outgoing>) -> Result<(), Failure> {
        loop {
            if self.budget_exhausted() {
                self.quantum(RealtimeInputOwned::Cancel, false, out).await?;
            }
            let frame = self
                .pcm
                .as_mut()
                .map(PcmInput::next_frame)
                .transpose()
                .map_err(Failure::native)?
                .flatten();
            let Some(frame) = frame else {
                break;
            };
            self.quantum(RealtimeInputOwned::Audio(frame), false, out)
                .await?;
        }
        Ok(())
    }
    async fn commit(&mut self, out: &mpsc::Sender<Outgoing>) -> Result<(), Failure> {
        if self.settings.mode == RealtimeMode::Duplex {
            return Err(Failure::invalid(
                "input_audio_buffer.commit: duplex models answer while listening; there is no turn to commit",
            ));
        }
        let item = ItemId::new(id("item_").map_err(Failure::native)?);
        let events = self.conversation.commit_audio(item.clone())?;
        self.configure().await?;
        self.pcm
            .as_mut()
            .context("committable audio has no PCM buffer")
            .map_err(Failure::native)?
            .commit()
            .map_err(Failure::invalid)?;
        self.drain_pcm(out).await?;
        self.pcm = None;
        if self.settings.mode == RealtimeMode::Transcription {
            self.conversation.begin_transcription(
                ResponseId::new(id("resp_").map_err(Failure::native)?),
                item,
                &self.settings,
            )?;
            self.open_output(out).await?;
        }
        for event in events {
            self.emit(out, event).await.map_err(Failure::native)?;
        }
        self.quantum(RealtimeInputOwned::Commit, false, out).await
    }
    async fn event(&mut self, event: Value, out: &mpsc::Sender<Outgoing>) -> Result<(), Failure> {
        let kind = string(&event, "type").map_err(Failure::invalid)?;
        match kind {
            "session.update" | "transcription_session.update" => {
                let mut next = self.settings.clone();
                let caps = Capabilities {
                    contract: &self.contract,
                    tool_protocol: self.model.tool_protocol.as_ref(),
                };
                next.update(
                    event
                        .get("session")
                        .context("session.update requires session")
                        .map_err(Failure::invalid)?,
                    &caps,
                    self.conversation.started(),
                )
                .map_err(Failure::invalid)?;
                let tool_arguments = if next.tools != self.settings.tools {
                    Some(
                        ToolArguments::compile(&next.tools, self.limits.text_bytes.get())
                            .map_err(Failure::invalid)?,
                    )
                } else {
                    None
                };
                if !self.conversation.started() {
                    self.conditioning
                        .tokens(
                            &self.model.tokenizer,
                            &self.contract,
                            &next,
                            PromptOperation::SessionStart,
                            "",
                            None,
                        )
                        .map_err(Failure::invalid)?;
                }
                if next.input_rate != self.settings.input_rate && self.pcm.is_some() {
                    if self.conversation.active().is_some_and(|active| {
                        matches!(active.kind, ResponseKind::Transcription { .. })
                    }) {
                        return Err(Failure::invalid(
                            "finish the active transcription before changing the rate of queued next-turn PCM; retry after its completed event",
                        ));
                    }
                    self.configure().await?;
                    self.drain_pcm(out).await?;
                    self.pcm
                        .as_mut()
                        .context("PCM rate change lost its buffer")
                        .map_err(Failure::native)?
                        .change_input_rate(next.input_rate)
                        .map_err(Failure::invalid)?;
                }
                if !self.conversation.started() {
                    self.configured = false;
                }
                if let Some(tool_arguments) = tool_arguments {
                    self.tool_arguments = tool_arguments;
                }
                self.settings = next;
                self.emit(out,json!({"type":"session.updated","session":self.settings.snapshot(&self.id,&self.model.id,&self.contract)})).await.map_err(Failure::native)?;
            }
            "input_audio_buffer.append" => {
                let audio = string(&event, "audio").map_err(Failure::invalid)?;
                let starts_duplex = self.settings.mode == RealtimeMode::Duplex
                    && !audio.is_empty()
                    && self.conversation.active().is_none();
                let constraint = if starts_duplex {
                    self.duplex_constraint()?
                } else {
                    None
                };
                if self.pcm.is_none() {
                    self.pcm = Some(self.new_pcm().map_err(Failure::invalid)?);
                }
                self.pcm
                    .as_mut()
                    .context("new PCM buffer was not installed")
                    .map_err(Failure::native)?
                    .append(audio)
                    .map_err(Failure::invalid)?;
                if !audio.is_empty() {
                    self.conversation.audio_appended(true);
                }
                self.configure().await?;
                if starts_duplex {
                    self.native
                        .arm_constraint(constraint)
                        .await
                        .map_err(Failure::native)?;
                    if self.conversation.begin_duplex(
                        ResponseId::new(id("resp_").map_err(Failure::native)?),
                        &self.settings,
                    )? {
                        self.open_output(out).await?;
                    }
                }
            }
            "input_audio_buffer.clear" => self
                .emit(out, self.conversation.clear_audio())
                .await
                .map_err(Failure::native)?,
            "input_audio_buffer.commit" => {
                if self.conversation.has_audio_to_commit()
                    && self.conversation.active().is_some_and(|active| {
                        matches!(active.kind, ResponseKind::Transcription { .. })
                    })
                {
                    self.pending_commit = true;
                } else {
                    self.commit(out).await?;
                }
            }
            "conversation.item.create" => {
                let item = event
                    .get("item")
                    .context("conversation.item.create requires item")
                    .map_err(Failure::invalid)?;
                let item_id = ItemId::new(match item.get("id") {
                    Some(value) => value
                        .as_str()
                        .context("item.id must be a string")
                        .map_err(Failure::invalid)?
                        .into(),
                    None => id("item_").map_err(Failure::native)?,
                });
                let (text, call_id, events, operation) = match item
                    .get("type")
                    .and_then(Value::as_str)
                {
                    Some("message") => {
                        ensure_user_role(item).map_err(Failure::invalid)?;
                        let mut text = String::new();
                        for part in item
                            .get("content")
                            .and_then(Value::as_array)
                            .context("user item.content must be an input_text array")
                            .map_err(Failure::invalid)?
                        {
                            if part.get("type") != Some(&json!("input_text")) {
                                return Err(Failure::invalid(
                                    "user item content supports input_text only; send audio through input_audio_buffer.append",
                                ));
                            }
                            text.push_str(string(part, "text").map_err(Failure::invalid)?);
                        }
                        let events = self
                            .conversation
                            .user_text(item_id, &text, &self.settings)?;
                        (text, None, events, PromptOperation::UserText)
                    }
                    Some("function_call_output") => {
                        let call_id = string(item, "call_id")
                            .map_err(Failure::invalid)?
                            .to_owned();
                        let text = string(item, "output").map_err(Failure::invalid)?.to_owned();
                        let events = self.conversation.tool_result(
                            item_id,
                            &call_id,
                            &text,
                            &self.settings,
                        )?;
                        (
                            text,
                            Some(events.model_call_id),
                            events.events,
                            PromptOperation::ToolResult,
                        )
                    }
                    _ => {
                        return Err(Failure::invalid(
                            "conversation.item.create supports user messages and external function_call_output items only",
                        ));
                    }
                };
                self.configure().await?;
                let tokens = self
                    .conditioning
                    .tokens(
                        &self.model.tokenizer,
                        &self.contract,
                        &self.settings,
                        operation,
                        &text,
                        call_id.as_deref(),
                    )
                    .map_err(Failure::native)?;
                self.native.set_budget(RealtimeTokenBudget::Unlimited);
                let _ = self
                    .native
                    .step(RealtimeInputOwned::Text(tokens))
                    .await
                    .map_err(Failure::native)?;
                for event in events {
                    self.emit(out, event).await.map_err(Failure::native)?;
                }
            }
            "response.create" => {
                let admission = self.conversation.begin_response(
                    ResponseId::new(id("resp_").map_err(Failure::native)?),
                    ItemId::new(id("item_").map_err(Failure::native)?),
                    &self.settings,
                    event.get("response").unwrap_or(&json!({})),
                    self.contract.spec().output_sample_rate_hz.is_some(),
                )?;
                self.configure().await?;
                if admission.commit_input {
                    self.pcm
                        .as_mut()
                        .context("pending audio response has no PCM buffer")
                        .map_err(Failure::native)?
                        .commit()
                        .map_err(Failure::invalid)?;
                    self.drain_pcm(out).await?;
                    self.pcm = None;
                    let _ = self
                        .native
                        .step(RealtimeInputOwned::Commit)
                        .await
                        .map_err(Failure::native)?;
                }
                for event in admission.events {
                    self.emit(out, event).await.map_err(Failure::native)?;
                }
                self.open_output(out).await?;
                let mut settings = self.settings.clone();
                settings.response = self
                    .conversation
                    .active()
                    .context("admitted response disappeared")
                    .map_err(Failure::native)?
                    .options
                    .clone();
                let constraint = if self.contract.spec().inputs.token_mask.is_some() {
                    self.response_constraint(
                        &self
                            .active
                            .as_ref()
                            .context("admitted response disappeared")
                            .map_err(Failure::native)?
                            .tool_arguments,
                        &settings.response,
                    )
                    .map_err(Failure::native)?
                } else {
                    None
                };
                let tokens = self
                    .conditioning
                    .tokens(
                        &self.model.tokenizer,
                        &self.contract,
                        &settings,
                        PromptOperation::ResponseStart,
                        "",
                        None,
                    )
                    .map_err(Failure::native)?;
                self.quantum(
                    RealtimeInputOwned::Generate { tokens, constraint },
                    false,
                    out,
                )
                .await?;
            }
            "response.cancel" => {
                // Upstream cancellation is idempotent: an idle response has
                // nothing to cancel and emits no acknowledgement or error.
                if self.active.is_some() {
                    self.quantum(RealtimeInputOwned::Cancel, true, out).await?;
                }
            }
            _ => {
                return Err(Failure::invalid(format!(
                    "unknown realtime event type {kind:?}; use session.update, input_audio_buffer.*, conversation.item.create or response.create/cancel"
                )));
            }
        }
        Ok(())
    }
    fn budget_exhausted(&self) -> bool {
        self.active
            .as_ref()
            .is_some_and(|active| active.budget.exhausted(active.used))
    }
    async fn quantum(
        &mut self,
        mut input: RealtimeInputOwned,
        mut cancelled: bool,
        out: &mpsc::Sender<Outgoing>,
    ) -> Result<(), Failure> {
        let budget = match self.active.as_ref() {
            Some(active) => match active.budget {
                OutputBudget::Unlimited => RealtimeTokenBudget::Unlimited,
                OutputBudget::Tokens(limit) => {
                    let remaining = u64::from(limit.get()).saturating_sub(active.used);
                    if remaining == 0
                        && matches!(
                            input,
                            RealtimeInputOwned::Generate { .. } | RealtimeInputOwned::Poll
                        )
                    {
                        input = RealtimeInputOwned::Cancel;
                        cancelled = false;
                    }
                    RealtimeTokenBudget::Remaining(remaining)
                }
            },
            None => RealtimeTokenBudget::Unlimited,
        };
        let cancelling = matches!(input, RealtimeInputOwned::Cancel);
        self.native.set_budget(budget);
        let output = self.native.step(input).await.map_err(Failure::native)?;
        self.needs_input = output.needs_input;
        if cancelling && !output.done {
            return Err(Failure::native(anyhow::anyhow!(
                "native realtime Cancel did not complete its response; repair the workflow's control-operation contract"
            )));
        }
        let Some(mut active) = self.active.take() else {
            return Ok(());
        };
        active.used = active
            .used
            .checked_add(
                u64::try_from(output.output_tokens)
                    .map_err(anyhow::Error::new)
                    .map_err(Failure::native)?,
            )
            .context("realtime output usage overflow")
            .map_err(Failure::native)?;
        let mut text = String::new();
        for token in output.tokens {
            text.push_str(
                &active
                    .decoder
                    .step(&self.model.tokenizer, token)
                    .map_err(Failure::native)?,
            );
        }
        if output.done {
            text.push_str(
                &active
                    .decoder
                    .finish(&self.model.tokenizer)
                    .map_err(Failure::native)?,
            );
        }
        let mut events = Vec::new();
        match &mut active.renderer {
            Rendered::Response(renderer) => {
                renderer
                    .usage(output.output_tokens)
                    .map_err(Failure::native)?;
                if let Some(tools) = active.tools.as_mut() {
                    if tools.raw.len().saturating_add(text.len()) > self.limits.text_bytes.get() {
                        return Err(Failure::native(anyhow::anyhow!(
                            "realtime tool/text stream exceeds its retained text budget; request a shorter response"
                        )));
                    }
                    tools.raw.push_str(&text);
                    match tools.stream.push(tools.protocol, &text) {
                        ToolParseOutcome::NoCall if !tools.raw.trim().is_empty() => {
                            if active.text_enabled {
                                events.extend(
                                    renderer
                                        .text(&tools.raw[tools.published..])
                                        .map_err(Failure::native)?,
                                );
                            }
                            tools.published = tools.raw.len();
                        }
                        ToolParseOutcome::Malformed(error) => {
                            return Err(Failure::native(anyhow::anyhow!(
                                "realtime model tool protocol is malformed: {error}"
                            )));
                        }
                        _ => {}
                    }
                    events.extend(
                        tools
                            .fragments(&active.tool_arguments, renderer)
                            .map_err(Failure::native)?,
                    );
                } else if active.text_enabled {
                    events.extend(renderer.text(&text).map_err(Failure::native)?);
                }
                if active.audio_enabled {
                    events.extend(renderer.audio(&output.audio).map_err(Failure::native)?);
                }
            }
            Rendered::Transcription(renderer) => {
                if let Some(event) = renderer.text(&text).map_err(Failure::native)? {
                    events.push(event);
                }
                if !output.done && output.needs_input {
                    return Err(Failure::native(anyhow::anyhow!(
                        "committed transcription requested more input before completing; repair the realtime transcription workflow"
                    )));
                }
            }
        }
        for event in events {
            self.emit(out, event).await.map_err(Failure::native)?;
        }
        if !output.done {
            self.active = Some(active);
            return Ok(());
        }
        let end = if cancelled {
            ResponseEnd::Cancelled
        } else if active.budget.exhausted(active.used) {
            ResponseEnd::TokenLimit
        } else {
            ResponseEnd::Model
        };
        let mut calls = Vec::new();
        let mut completed_calls = Vec::new();
        let final_events = match active.renderer {
            Rendered::Response(mut renderer) => {
                if let Some(tools) = active.tools {
                    let final_fragments = tools
                        .stream
                        .fragments(tools.protocol)
                        .map_err(|error| Failure::native(error.into()))?;
                    match tools.stream.finish(tools.protocol) {
                        ToolParseOutcome::NoCall => {
                            if active.text_enabled {
                                for event in renderer
                                    .text(&tools.raw[tools.published..])
                                    .map_err(Failure::native)?
                                {
                                    self.emit(out, event).await.map_err(Failure::native)?;
                                }
                            }
                        }
                        ToolParseOutcome::TerminalComplete(parsed) => {
                            calls = parsed;
                            active
                                .tool_arguments
                                .admit_projected(&calls, &final_fragments)
                                .map_err(Failure::native)?;
                            if tools.bindings.len() != calls.len() {
                                return Err(Failure::native(anyhow::anyhow!(
                                    "realtime completed batch differs from the projected function identities; repair the model tool protocol parser"
                                )));
                            }
                            let mut call_events = Vec::new();
                            for (call, binding) in calls.iter().zip(&tools.bindings) {
                                if call.name != binding.name
                                    || !binding
                                        .model_id
                                        .as_ref()
                                        .is_none_or(|model_id| *model_id == call.id)
                                {
                                    return Err(Failure::native(anyhow::anyhow!(
                                        "realtime completed tool call changed its projected model identity; repair the model protocol"
                                    )));
                                }
                                let wire_call = onnx_genai_metadata::ToolCall {
                                    id: binding.wire_id.clone(),
                                    name: call.name.clone(),
                                    arguments: call.arguments.clone(),
                                };
                                call_events.extend(
                                    renderer
                                        .tool_call(&wire_call, binding.item_id.clone())
                                        .map_err(Failure::native)?,
                                );
                                completed_calls.push(CompletedToolCall {
                                    wire_id: binding.wire_id.clone(),
                                    model_call: call.clone(),
                                });
                            }
                            for event in call_events {
                                self.emit(out, event).await.map_err(Failure::native)?;
                            }
                        }
                        ToolParseOutcome::Incomplete if end.permits_unfinished_tools() => {}
                        other => {
                            return Err(Failure::native(anyhow::anyhow!(
                                "realtime tool protocol did not finish a valid envelope: {other:?}"
                            )));
                        }
                    }
                }
                if !end.permits_unfinished_tools()
                    && self.conversation.active().is_some_and(|response| {
                        response.options.tool_choice == crate::types::ToolChoiceMode::Required
                    })
                    && calls.is_empty()
                {
                    return Err(Failure::native(anyhow::anyhow!(
                        "native realtime response did not satisfy tool_choice=required; repair the model tool constraint workflow"
                    )));
                }
                renderer.finish(end)
            }
            Rendered::Transcription(renderer) => {
                self.configured = false;
                vec![renderer.finish()]
            }
        };
        let last = final_events
            .last()
            .and_then(|event| event.get("response"))
            .and_then(|response| response.get("output"))
            .and_then(Value::as_array)
            .and_then(|items| items.last())
            .and_then(|item| item.get("id"))
            .and_then(Value::as_str)
            .map(|id| ItemId::new(id.into()));
        self.conversation
            .complete(&active.id, &completed_calls, last)?;
        if self.settings.mode == RealtimeMode::Duplex {
            // llama.cpp's duplex cancel resets stream/gen and its next audio
            // starts a fresh KV/codec stream. Never replay queued old PCM.
            self.pcm = None;
            self.configured = false;
            self.conversation.discard_duplex_input();
        }
        for event in final_events {
            self.emit(out, event).await.map_err(Failure::native)?;
        }
        Ok(())
    }
}
fn ensure_user_role(item: &Value) -> anyhow::Result<()> {
    ensure!(
        item.get("role") == Some(&json!("user")),
        "conversation.item.create message.role must be user"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use tokio::net::{TcpListener, TcpStream};
    #[test]
    fn tool_batch_admission_rejects_a_late_undeclared_call_before_publication() {
        let tools = json!([{"type":"function","name":"lookup"}]);
        let mut calls = vec![onnx_genai_metadata::ToolCall {
            id: "call_1".into(),
            name: "lookup".into(),
            arguments: "{}".into(),
        }];
        let admitted = ToolArguments::compile(&tools, 4096).unwrap();
        admitted.admit(&calls).unwrap();
        calls.push(onnx_genai_metadata::ToolCall {
            id: "call_2".into(),
            name: "undeclared".into(),
            arguments: "{}".into(),
        });
        let error = admitted.admit(&calls).unwrap_err().to_string();
        assert!(
            error.contains("undeclared realtime tool \"undeclared\""),
            "{error}"
        );
    }
    type Client = WebSocketStream<TcpStream>;
    async fn event(client: &mut Client) -> Value {
        loop {
            let message = tokio::time::timeout(std::time::Duration::from_secs(5), client.next())
                .await
                .expect("realtime event timed out")
                .expect("websocket closed")
                .expect("read websocket");
            if let Message::Text(text) = message {
                let value: Value = serde_json::from_str(&text).unwrap();
                assert!(value["event_id"].as_str().is_some());
                return value;
            }
        }
    }
    async fn send(client: &mut Client, value: Value) {
        client
            .send(Message::Text(value.to_string().into()))
            .await
            .unwrap();
    }
    async fn server() -> (AppState, std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime");
        server_for_package(&root).await
    }
    async fn server_for_package(
        root: &std::path::Path,
    ) -> (AppState, std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        server_for_package_with_config(root, crate::ServerConfig::default()).await
    }
    async fn server_for_package_with_config(
        root: &std::path::Path,
        config: crate::ServerConfig,
    ) -> (AppState, std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        let state =
            AppState::load_with_config(root, Some("native-realtime".into()), config).unwrap();
        server_for_state(state).await
    }
    async fn server_for_state(
        state: AppState,
    ) -> (AppState, std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let router = crate::app(state.clone());
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (state, address, task)
    }
    async fn connect(address: std::net::SocketAddr) -> Client {
        connect_model(address, "native-realtime").await.unwrap()
    }
    async fn connect_model(
        address: std::net::SocketAddr,
        model: &str,
    ) -> Result<Client, tokio_tungstenite::tungstenite::Error> {
        let stream = TcpStream::connect(address).await?;
        tokio_tungstenite::client_async(format!("ws://{address}/v1/realtime?model={model}"), stream)
            .await
            .map(|(client, _)| client)
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_models_isolate_native_state_and_release_closed_worker() {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime");
        let package = tempfile::tempdir().unwrap();
        for file in ["model.onnx.textproto", "tokenizer.json"] {
            std::fs::copy(root.join(file), package.path().join(file)).unwrap();
        }
        let metadata = std::fs::read_to_string(root.join("inference_metadata.yaml"))
            .unwrap()
            .replacen("default_mode: realtime", "default_mode: duplex", 1);
        std::fs::write(package.path().join("inference_metadata.yaml"), metadata).unwrap();
        let specs = ["alpha", "beta"]
            .into_iter()
            .map(|id| crate::models_config::ModelSpec {
                package_variant: None,
                id: id.into(),
                path: package.path().to_owned(),
                eager: true,
                warmup: false,
            })
            .collect();
        let state = AppState::load_from_specs(specs, crate::ServerConfig::default()).unwrap();
        let (state, address, task) = server_for_state(state).await;
        let mut alpha = connect_model(address, "alpha").await.unwrap();
        let alpha_created = event(&mut alpha).await;
        assert_eq!(alpha_created["session"]["model"], "alpha");
        let mut beta = connect_model(address, "beta").await.unwrap();
        let beta_created = event(&mut beta).await;
        assert_eq!(beta_created["session"]["model"], "beta");
        assert_ne!(
            alpha_created["session"]["id"],
            beta_created["session"]["id"]
        );
        match connect_model(address, "alpha").await {
            Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
                assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                let body: Value =
                    serde_json::from_slice(response.body().as_ref().unwrap()).unwrap();
                assert!(
                    body["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains("realtime"),
                    "{body}"
                );
            }
            other => {
                panic!("an occupied model must reject a second realtime connection: {other:?}")
            }
        }
        async fn audio(client: &mut Client, sample: i16) -> Vec<u8> {
            let bytes: Vec<_> = (0..320).flat_map(|_| sample.to_le_bytes()).collect();
            send(
                client,
                json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(bytes)}),
            )
            .await;
            loop {
                let output = event(client).await;
                assert_ne!(output["type"], "error", "{output}");
                if output["type"] == "response.output_audio.delta" {
                    let bytes = STANDARD.decode(output["delta"].as_str().unwrap()).unwrap();
                    assert_eq!(
                        bytes.len(),
                        640,
                        "one native frame must contain 320 PCM16 samples"
                    );
                    return bytes;
                }
            }
        }
        assert!(audio(&mut alpha, 0).await.iter().all(|b| *b == 0));
        assert!(audio(&mut beta, 100).await.iter().any(|b| *b != 0));
        alpha.close(None).await.unwrap();
        let closed = tokio::time::timeout(std::time::Duration::from_secs(5), alpha.next())
            .await
            .expect("alpha close handshake timed out")
            .unwrap()
            .unwrap();
        assert!(matches!(closed, Message::Close(_)));
        drop(alpha);
        // The close ACK can precede native cleanup. Retry only transient busy
        // admission within a bounded deadline; other handshake failures fail.
        let mut alpha = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                match connect_model(address, "alpha").await {
                    Ok(client) => break client,
                    Err(tokio_tungstenite::tungstenite::Error::Http(response))
                        if response.status() == StatusCode::SERVICE_UNAVAILABLE =>
                    {
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                    Err(error) => panic!("reopen alpha after native cleanup: {error}"),
                }
            }
        })
        .await
        .expect("closed model worker was not released");
        let reopened = event(&mut alpha).await;
        assert_ne!(reopened["session"]["id"], alpha_created["session"]["id"]);
        assert_eq!(reopened["session"]["model"], "alpha");
        assert!(audio(&mut alpha, 0).await.iter().all(|b| *b == 0));
        // Beta keeps its own accumulated native audio state across Alpha's
        // disconnect/reset/reopen. A zero frame therefore still produces audio.
        assert!(audio(&mut beta, 0).await.iter().any(|b| *b != 0));
        alpha.close(None).await.unwrap();
        beta.close(None).await.unwrap();
        drop((alpha, beta));
        for id in ["alpha", "beta"] {
            let model = state.registry.resolve(id).unwrap().unwrap();
            tokio::task::spawn_blocking(move || model.engine.shutdown())
                .await
                .unwrap();
        }
        task.abort();
        let _ = task.await;
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_rejected_frames_and_pcm_pressure_preserve_native_transcription() {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime");
        let mut config = crate::ServerConfig::default();
        config.realtime.source_bytes = NonZeroUsize::new(640).unwrap();
        let (state, address, task) = server_for_package_with_config(&root, config).await;
        let mut client = connect(address).await;
        assert_eq!(event(&mut client).await["type"], "session.created");
        let rejected = [
            (Message::Text("{".into()), "valid JSON"),
            (Message::Binary(vec![0, 1].into()), "JSON text frames"),
            (Message::Text("null".into()), "type must be a string"),
            (
                Message::Text(json!({"type":"unknown"}).to_string().into()),
                "unknown realtime event",
            ),
            (
                Message::Text(
                    json!({"type":"input_audio_buffer.append","audio":"!!!"})
                        .to_string()
                        .into(),
                ),
                "valid base64",
            ),
            (
                Message::Text(
                    json!({"type":"input_audio_buffer.append","audio":STANDARD.encode([0])})
                        .to_string()
                        .into(),
                ),
                "even byte count",
            ),
            (
                Message::Text(
                    json!({"type":"input_audio_buffer.append","audio":STANDARD.encode([0;642])})
                        .to_string()
                        .into(),
                ),
                "640-byte PCM queue budget",
            ),
        ];
        let mut ids = BTreeSet::new();
        for (frame, diagnostic) in rejected {
            client.send(frame).await.unwrap();
            let failure = event(&mut client).await;
            assert_eq!(failure["type"], "error", "{failure}");
            assert_eq!(failure["error"]["type"], "invalid_request_error");
            assert!(
                failure["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains(diagnostic),
                "{failure}"
            );
            assert!(ids.insert(failure["event_id"].as_str().unwrap().to_owned()));
        }
        // Rejected audio must neither freeze session conditioning nor consume
        // input quota/state. A full-budget valid frame still starts an ASR turn.
        send(
            &mut client,
            json!({"type":"session.update","session":{"type":"transcription"}}),
        )
        .await;
        assert_eq!(event(&mut client).await["type"], "session.updated");
        send(
            &mut client,
            json!({"type":"input_audio_buffer.append","audio":STANDARD.encode([0;640])}),
        )
        .await;
        send(&mut client, json!({"type":"input_audio_buffer.commit"})).await;
        let committed = event(&mut client).await;
        assert_eq!(committed["type"], "input_audio_buffer.committed");
        let item = committed["item_id"].clone();
        let mut transcript = String::new();
        loop {
            let output = event(&mut client).await;
            assert_ne!(output["type"], "error", "{output}");
            if output["type"] == "conversation.item.input_audio_transcription.delta" {
                assert_eq!(output["item_id"], item);
                transcript.push_str(output["delta"].as_str().unwrap());
            }
            if output["type"] == "conversation.item.input_audio_transcription.completed" {
                assert_eq!(output["item_id"], item);
                assert_eq!(output["transcript"], "hello world");
                break;
            }
        }
        assert_eq!(transcript, "hello world");
        client.close(None).await.unwrap();
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }
    fn tool_package(envelope: &str) -> tempfile::TempDir {
        tool_package_from("realtime", envelope)
    }
    fn tool_package_from(fixture: &str, envelope: &str) -> tempfile::TempDir {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows")
            .join(fixture);
        let package = tempfile::tempdir().unwrap();
        std::fs::copy(
            root.join("model.onnx.textproto"),
            package.path().join("model.onnx.textproto"),
        )
        .unwrap();
        // One explicit fixture token preserves the complete model envelope;
        // inference still executes the native ONNX quantum, without a mock server.
        let mut tokenizer: Value =
            serde_json::from_slice(&std::fs::read(root.join("tokenizer.json")).unwrap()).unwrap();
        tokenizer["model"]["vocab"]
            .as_object_mut()
            .unwrap()
            .remove("tok16");
        tokenizer["model"]["vocab"][envelope] = json!(16);
        tokenizer["added_tokens"].as_array_mut().unwrap().push(json!({"id":16,"content":envelope,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}));
        std::fs::write(
            package.path().join("tokenizer.json"),
            serde_json::to_vec(&tokenizer).unwrap(),
        )
        .unwrap();
        let mut metadata: serde_yaml::Value = serde_yaml::from_str(
            &std::fs::read_to_string(root.join("inference_metadata.yaml")).unwrap(),
        )
        .unwrap();
        metadata["package"]["tool_protocol"] =
            serde_yaml::to_value(json!({"identity":"tagged-json","version":"v1"})).unwrap();
        metadata["profiles"]["live"]["realtime"]["prompt_template"] = serde_yaml::Value::String(
            format!(
                "{{% if operation == \"response_start\" %}}{envelope}{{% else %}}{{{{ text }}}}{{% endif %}}"
            ),
        );
        std::fs::write(
            package.path().join("inference_metadata.yaml"),
            serde_yaml::to_string(&metadata).unwrap(),
        )
        .unwrap();
        package
    }
    fn streaming_tool_package(native_budget_stop: bool) -> tempfile::TempDir {
        let package = tool_package_from("realtime-constrained", "<tool_call>\n");
        let path = package.path().join("tokenizer.json");
        let mut tokenizer: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let tail = "{\"name\":\"lookup\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>";
        tokenizer["model"]["vocab"]
            .as_object_mut()
            .unwrap()
            .remove("tok17");
        tokenizer["model"]["vocab"][tail] = json!(17);
        tokenizer["added_tokens"].as_array_mut().unwrap().push(json!({"id":17,"content":tail,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}));
        tokenizer["decoder"] = json!({"type":"Fuse"});
        std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
        let path = package.path().join("model.onnx.textproto");
        let mut graph = std::fs::read_to_string(&path).unwrap()
            .replace("node { input: \"generating\" input: \"asr\"", "initializer { name: \"polling_op\" data_type: 7 int64_data: 7 }\nnode { input: \"operation\" input: \"polling_op\" output: \"polling\" op_type: \"Equal\" }\nnode { input: \"generating\" input: \"polling\" output: \"decoding\" op_type: \"Or\" }\nnode { input: \"decoding\" input: \"asr\"")
            .replace("node { input: \"generating\" output: \"candidate_tokens\"", "node { input: \"decoding\" output: \"candidate_tokens\"")
            .replace("node { input: \"responding\" input: \"cancelling\" output: \"done\" op_type: \"Or\" }", "node { input: \"tokens_out\" output: \"emitted_id\" op_type: \"ReduceSum\" attribute { name: \"keepdims\" i: 0 type: INT } }\nnode { input: \"emitted_id\" input: \"three\" output: \"ending\" op_type: \"Equal\" }\nnode { input: \"ending\" input: \"cancelling\" output: \"model_done\" op_type: \"Or\" }");
        if native_budget_stop {
            graph = graph.replace("node { input: \"ending\" input: \"cancelling\" output: \"model_done\" op_type: \"Or\" }", "node { input: \"ending\" input: \"cancelling\" output: \"model_done\" op_type: \"Or\" }\nnode { input: \"output_token_budget\" input: \"zero\" output: \"limited\" op_type: \"GreaterOrEqual\" }\nnode { input: \"output_token_budget\" input: \"count\" output: \"at_limit\" op_type: \"Equal\" }\nnode { input: \"limited\" input: \"at_limit\" output: \"budget_done\" op_type: \"And\" }\nnode { input: \"budget_done\" input: \"model_done\" output: \"done\" op_type: \"Or\" }");
        } else {
            graph = graph.replace(
                "output: \"model_done\" op_type: \"Or\" }",
                "output: \"done\" op_type: \"Or\" }",
            );
        }
        std::fs::write(path, graph).unwrap();
        package
    }

    fn fragment_tool_package(paused: bool, late_id: bool) -> tempfile::TempDir {
        let package = streaming_tool_package(false);
        let prefix = if late_id {
            "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"city\":\"Pa"
        } else {
            "<tool_call>\n{\"id\":\"call_lookup\",\"name\":\"lookup\",\"arguments\":{\"city\":\"Pa"
        };
        let suffix = if late_id {
            "ris\"},\"id\":\"model_late\"}\n</tool_call>"
        } else {
            "ris\"}}\n</tool_call>"
        };
        let path = package.path().join("tokenizer.json");
        let mut tokenizer: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for (id, text) in [(16, prefix), (17, suffix)] {
            let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
            let old = vocab
                .iter()
                .find(|(_, value)| value.as_u64() == Some(id))
                .unwrap()
                .0
                .clone();
            vocab.remove(&old);
            vocab.insert(text.into(), json!(id));
            for token in tokenizer["added_tokens"].as_array_mut().unwrap() {
                if token["id"] == id {
                    token["content"] = json!(text);
                }
            }
        }
        std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
        let path = package.path().join("inference_metadata.yaml");
        let mut metadata: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let model_id = if late_id { "model_late" } else { "call_lookup" };
        metadata["profiles"]["live"]["realtime"]["prompt_template"] = serde_yaml::Value::String(
            format!(
                "{{% if operation == \"response_start\" %}}{prefix}{{% elif operation == \"tool_result\" and call_id != \"{model_id}\" %}}{{{{ raise_exception('tool result lost the original model call ID') }}}}{{% else %}}{{{{ text }}}}{{% endif %}}"
            ),
        );
        std::fs::write(path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
        if paused {
            let path = package.path().join("model.onnx.textproto");
            let graph = std::fs::read_to_string(&path).unwrap();
            let node = "node { input: \"responding\" output: \"needs_input\" op_type: \"Not\" }";
            assert!(graph.contains(node));
            std::fs::write(path, graph.replace(node, "initializer { name: \"fragment_wait\" data_type: 9 int32_data: 1 }\nnode { input: \"fragment_wait\" output: \"needs_input\" op_type: \"Identity\" }")).unwrap();
        }
        package
    }

    async fn configure_fragment_client(client: &mut Client) {
        assert_eq!(event(client).await["type"], "session.created");
        send(client,json!({"type":"session.update","session":{"output_modalities":["text"],"max_output_tokens":4,"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"city":{"const":"Paris"}},"required":["city"],"additionalProperties":false}}],"tool_choice":"required"}})).await;
        assert_eq!(event(client).await["type"], "session.updated");
        send(client,json!({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}})).await;
        assert_eq!(event(client).await["type"], "conversation.item.added");
        assert_eq!(event(client).await["type"], "conversation.item.done");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_tool_fragment_is_visible_while_native_response_waits_and_cancel_never_admits_it()
     {
        let package = fragment_tool_package(true, false);
        let (state, address, task) = server_for_package(package.path()).await;
        let mut client = connect(address).await;
        configure_fragment_client(&mut client).await;
        send(&mut client, json!({"type":"response.create"})).await;
        assert_eq!(event(&mut client).await["type"], "response.created");
        let item = event(&mut client).await;
        assert_eq!(item["type"], "response.output_item.added");
        assert_eq!(item["item"]["status"], "in_progress");
        assert_eq!(item["item"]["call_id"], "call_lookup");
        let delta = event(&mut client).await;
        assert_eq!(delta["type"], "response.function_call_arguments.delta");
        assert_eq!(delta["delta"], r#"{"city":"Pa"#);
        // The real workflow is waiting for input, so neither a completed call
        // nor response completion can have been synthesized behind these events.
        send(&mut client,json!({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"call_lookup","output":"hello"}})).await;
        assert_eq!(event(&mut client).await["type"], "error");
        send(&mut client, json!({"type":"response.cancel"})).await;
        loop {
            let output = event(&mut client).await;
            assert_ne!(output["type"], "error", "{output}");
            assert_ne!(
                output["type"], "response.function_call_arguments.done",
                "{output}"
            );
            if output["type"] == "response.done" {
                assert_eq!(output["response"]["status"], "cancelled");
                assert_eq!(output["response"]["output"][0]["status"], "cancelled");
                assert_eq!(
                    output["response"]["output"][0]["arguments"],
                    r#"{"city":"Pa"#
                );
                break;
            }
        }
        send(&mut client,json!({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"call_lookup","output":"hello"}})).await;
        assert_eq!(event(&mut client).await["type"], "error");
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_xml_parameter_streams_before_invoke_end_in_a_waiting_native_workflow() {
        let prefix =
            r#"<atem:invoke name="lookup"><atem:parameter name="city">"Paris"</atem:parameter>"#;
        let package = tool_package_from("realtime", prefix);
        let path = package.path().join("inference_metadata.yaml");
        let mut metadata: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        metadata["package"]["tool_protocol"] =
            serde_yaml::to_value(json!({"identity":"atem-xml","version":"v1"})).unwrap();
        std::fs::write(path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
        let path = package.path().join("model.onnx.textproto");
        let graph = std::fs::read_to_string(&path).unwrap();
        let done =
            "node { input: \"responding\" input: \"cancelling\" output: \"done\" op_type: \"Or\" }";
        let waiting = "node { input: \"responding\" output: \"needs_input\" op_type: \"Not\" }";
        assert!(graph.contains(done) && graph.contains(waiting));
        let graph = graph.replace(done, "node { input: \"cancelling\" output: \"done\" op_type: \"Identity\" }")
            .replace(waiting,"initializer { name: \"xml_wait\" data_type: 9 int32_data: 1 }\nnode { input: \"xml_wait\" output: \"needs_input\" op_type: \"Identity\" }");
        std::fs::write(path, graph).unwrap();
        let (state, address, task) = server_for_package(package.path()).await;
        let mut client = connect(address).await;
        configure_fragment_client(&mut client).await;
        send(&mut client, json!({"type":"response.create"})).await;
        assert_eq!(event(&mut client).await["type"], "response.created");
        let item = event(&mut client).await;
        assert_eq!(item["type"], "response.output_item.added", "{item}");
        assert_eq!(item["item"]["name"], "lookup");
        let delta = event(&mut client).await;
        assert_eq!(
            delta["type"], "response.function_call_arguments.delta",
            "{delta}"
        );
        assert_eq!(delta["delta"], r#"{"city":"Paris""#);
        send(&mut client, json!({"type":"response.cancel"})).await;
        loop {
            let output = event(&mut client).await;
            assert_ne!(output["type"], "error", "{output}");
            assert_ne!(
                output["type"], "response.function_call_arguments.done",
                "{output}"
            );
            if output["type"] == "response.done" {
                assert_eq!(output["response"]["status"], "cancelled");
                assert_eq!(
                    output["response"]["output"][0]["arguments"],
                    r#"{"city":"Paris""#
                );
                break;
            }
        }
        send(&mut client,json!({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":item["item"]["call_id"],"output":"hello"}})).await;
        assert_eq!(event(&mut client).await["type"], "error");
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_tool_fragments_keep_opaque_wire_id_and_condition_results_with_late_model_id()
    {
        let package = fragment_tool_package(false, true);
        let (state, address, task) = server_for_package(package.path()).await;
        let mut client = connect(address).await;
        configure_fragment_client(&mut client).await;
        send(&mut client, json!({"type":"response.create"})).await;
        let mut wire_id = None;
        let mut deltas = Vec::new();
        loop {
            let output = event(&mut client).await;
            assert_ne!(output["type"], "error", "{output}");
            if output["type"] == "response.output_item.added" {
                assert!(wire_id.is_none());
                wire_id = Some(output["item"]["call_id"].as_str().unwrap().to_owned());
                assert_ne!(wire_id.as_deref(), Some("model_late"));
            }
            if output["type"] == "response.function_call_arguments.delta" {
                assert_eq!(output["call_id"].as_str(), wire_id.as_deref());
                deltas.push(output["delta"].as_str().unwrap().to_owned());
            }
            if output["type"] == "response.function_call_arguments.done" {
                assert_eq!(output["arguments"], deltas.concat());
            }
            if output["type"] == "response.done" {
                assert_eq!(deltas, [r#"{"city":"Pa"#, r#"ris"}"#]);
                assert_eq!(
                    output["response"]["output"][0]["call_id"].as_str(),
                    wire_id.as_deref()
                );
                assert_eq!(
                    output["response"]["output"][0]["arguments"],
                    r#"{"city":"Paris"}"#
                );
                assert_eq!(output["response"]["usage"]["output_tokens"], 3);
                break;
            }
        }
        send(&mut client,json!({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"model_late","output":"hello"}})).await;
        assert_eq!(event(&mut client).await["type"], "error");
        send(&mut client,json!({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":wire_id.unwrap(),"output":"hello"}})).await;
        // The authored template raises if conditioning receives the wire ID
        // instead of model_late; these acknowledgements follow native execution.
        assert_eq!(event(&mut client).await["type"], "conversation.item.added");
        assert_eq!(event(&mut client).await["type"], "conversation.item.done");
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_tool_prefix_at_token_limit_completes_and_keeps_session_usable() {
        for native_budget_stop in [false, true] {
            let package = streaming_tool_package(native_budget_stop);
            let (state, address, task) = server_for_package(package.path()).await;
            let mut client = connect(address).await;
            assert_eq!(event(&mut client).await["type"], "session.created");
            send(&mut client,json!({"type":"session.update","session":{"output_modalities":["text"],"max_output_tokens":1,"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"city":{"const":"Paris"}},"required":["city"],"additionalProperties":false}}],"tool_choice":"required"}})).await;
            assert_eq!(event(&mut client).await["type"], "session.updated");
            let mut previous = None;
            for turn in 0..2 {
                send(&mut client,json!({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}})).await;
                assert_eq!(event(&mut client).await["type"], "conversation.item.added");
                assert_eq!(event(&mut client).await["type"], "conversation.item.done");
                send(&mut client,json!({"type":"response.create","response":{"max_output_tokens":if turn==0 {1} else {4}}})).await;
                loop {
                    let output = event(&mut client).await;
                    assert_ne!(
                        output["type"], "error",
                        "native_budget_stop={native_budget_stop}: {output}"
                    );
                    if output["type"] == "response.done" {
                        assert_eq!(output["response"]["status"], "completed");
                        if turn == 0 {
                            assert_eq!(output["response"]["usage"]["output_tokens"], 1);
                            assert_eq!(output["response"]["output"], json!([]));
                        } else {
                            assert_eq!(output["response"]["usage"]["output_tokens"], 3);
                            assert_eq!(output["response"]["output"][0]["name"], "lookup");
                        }
                        assert_ne!(previous, Some(output["response"]["id"].clone()));
                        previous = Some(output["response"]["id"].clone());
                        break;
                    }
                }
            }
            client.close(None).await.unwrap();
            drop(client);
            let model = state.registry.resolve("native-realtime").unwrap().unwrap();
            tokio::task::spawn_blocking(move || model.engine.shutdown())
                .await
                .unwrap();
            task.abort();
            let _ = task.await;
        }
    }

    async fn assert_tool_parameter_schema_masks_arrays(parameters: Value) {
        let package = tool_package_from(
            "realtime-constrained",
            "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
        );
        let path = package.path().join("tokenizer.json");
        let mut tokenizer: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let bad = "<tool_call>\n{\"name\":\"lookup\",\"arguments\":[]}\n</tool_call>";
        tokenizer["model"]["vocab"]
            .as_object_mut()
            .unwrap()
            .remove("tok17");
        tokenizer["model"]["vocab"][bad] = json!(17);
        tokenizer["added_tokens"].as_array_mut().unwrap().push(json!({"id":17,"content":bad,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}));
        std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
        let (state, address, task) = server_for_package(package.path()).await;
        let mut client = connect(address).await;
        assert_eq!(event(&mut client).await["type"], "session.created");
        send(&mut client,json!({"type":"session.update","session":{"output_modalities":["text"],"tools":[{"type":"function","name":"lookup","parameters":parameters}],"tool_choice":"required"}})).await;
        assert_eq!(event(&mut client).await["type"], "session.updated");
        send(&mut client,json!({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}})).await;
        assert_eq!(event(&mut client).await["type"], "conversation.item.added");
        assert_eq!(event(&mut client).await["type"], "conversation.item.done");
        send(&mut client, json!({"type":"response.create"})).await;
        loop {
            let output = event(&mut client).await;
            assert_ne!(
                output["type"], "error",
                "native mask must exclude higher-ranked array arguments: {output}"
            );
            if output["type"] == "response.done" {
                assert_eq!(
                    output["response"]["output"][0]["arguments"],
                    r#"{"city":"Paris"}"#
                );
                break;
            }
        }
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_unconstrained_tool_schema_masks_nonobject_protocol_arguments() {
        for parameters in [
            json!(true),
            json!({}),
            json!({"type":["array","object"]}),
            Value::Null,
        ] {
            assert_tool_parameter_schema_masks_arrays(parameters).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_tool_schema_root_references_survive_forced_envelope_scope() {
        for parameters in [
            json!({"$defs":{"Args":{"type":"object","properties":{"city":{"const":"Paris"}},"required":["city"],"additionalProperties":false}},"$ref":"#/$defs/Args"}),
            json!({"$defs":{"Args":{"type":"object","properties":{"city":{"const":"Paris"}},"required":["city"],"additionalProperties":false}},"$ref":"json-schema:///#/$defs/Args"}),
            json!({"$id":"https://example.invalid/args","$defs":{"a/b~c":{"type":"object","properties":{"city":{"const":"Paris"}},"required":["city"],"additionalProperties":false}},"$ref":"#/$defs/a~1b~0c"}),
            json!({"$schema":"http://json-schema.org/draft-04/schema#","definitions":{"Args":{"type":"object","properties":{"city":{"enum":["Paris"]}},"required":["city"],"additionalProperties":false}},"$ref":"#/definitions/Args"}),
        ] {
            assert_tool_parameter_schema_masks_arrays(parameters).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_required_tool_grammar_masks_native_selection() {
        for body in [
            r#"{"name":"lookup","arguments":{"city":"Paris"}}"#,
            r#"{"id":"call_lookup","name":"lookup","arguments":{"city":"Paris"}}"#,
            r#"{"id":"call_lookup","arguments":{"city":"Paris"},"name":"lookup"}"#,
            r#"{"name":"lookup","id":"call_lookup","arguments":{"city":"Paris"}}"#,
            r#"{"name":"lookup","arguments":{"city":"Paris"},"id":"call_lookup"}"#,
            r#"{"arguments":{"city":"Paris"},"id":"call_lookup","name":"lookup"}"#,
            r#"{"arguments":{"city":"Paris"},"name":"lookup","id":"call_lookup"}"#,
        ] {
            let package = tool_package_from(
                "realtime-constrained",
                &format!("<tool_call>\n{body}\n</tool_call>"),
            );
            let (state, address, task) = server_for_package(package.path()).await;
            let mut client = connect(address).await;
            assert_eq!(event(&mut client).await["type"], "session.created");
            send(&mut client, json!({"type":"session.update","session":{"output_modalities":["text"],"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"city":{"const":"Paris"}},"required":["city"],"additionalProperties":false}}],"tool_choice":"required"}})).await;
            assert_eq!(event(&mut client).await["type"], "session.updated");
            send(&mut client, json!({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}})).await;
            assert_eq!(event(&mut client).await["type"], "conversation.item.added");
            assert_eq!(event(&mut client).await["type"], "conversation.item.done");
            send(&mut client, json!({"type":"response.create"})).await;
            let mut observed = Vec::new();
            loop {
                let output = event(&mut client).await;
                observed.push(output.clone());
                assert_ne!(output["type"], "error", "{observed:?}");
                if output["type"] == "response.done" {
                    assert_eq!(output["response"]["usage"]["output_tokens"], 1);
                    assert_eq!(output["response"]["output"][0]["name"], "lookup");
                    assert_eq!(
                        serde_json::from_str::<Value>(
                            output["response"]["output"][0]["arguments"]
                                .as_str()
                                .unwrap()
                        )
                        .unwrap(),
                        json!({"city":"Paris"})
                    );
                    break;
                }
            }
            client.close(None).await.unwrap();
            drop(client);
            let model = state.registry.resolve("native-realtime").unwrap().unwrap();
            tokio::task::spawn_blocking(move || model.engine.shutdown())
                .await
                .unwrap();
            task.abort();
            let _ = task.await;
        }
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_initializes_native_session_with_rendered_prompt_tokens() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime");
        let package = tempfile::tempdir().unwrap();
        std::fs::copy(
            root.join("tokenizer.json"),
            package.path().join("tokenizer.json"),
        )
        .unwrap();
        let metadata = std::fs::read_to_string(root.join("inference_metadata.yaml")).unwrap()
            .replace("prompt_template: '{% if operation == \"response_start\" %}hello world{% else %}{{ text }}{% endif %}'", "prompt_template: 'hello'");
        std::fs::write(package.path().join("inference_metadata.yaml"), metadata).unwrap();
        // A native start must consume the first prompt token. Gather rejects
        // an empty initialization vector, unlike the ordinary echo fixture.
        let graph = std::fs::read_to_string(root.join("model.onnx.textproto")).unwrap()
            .replace("input: \"sum\" input: \"previous\" output: \"next\"", "input: \"seeded_sum\" input: \"previous\" output: \"next\"")
            .replace("node { input: \"starting\" input: \"operation\"", r#"node { input: "starting" output: "initial_token" op_type: "If"
attribute { name: "then_branch" type: GRAPH g { name: "required_prompt" node { input: "tokens" input: "zero" output: "first" op_type: "Gather" } output { name: "first" type { tensor_type { elem_type: 7 shape {} } } } } }
attribute { name: "else_branch" type: GRAPH g { name: "no_prompt" node { input: "zero" output: "none" op_type: "Identity" } output { name: "none" type { tensor_type { elem_type: 7 shape {} } } } } }
}
node { input: "initial_token" output: "initial_float" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
node { input: "sum" input: "initial_float" output: "seeded_sum" op_type: "Add" }
node { input: "starting" input: "operation""#);
        std::fs::write(package.path().join("model.onnx.textproto"), graph).unwrap();
        let (state, address, task) = server_for_package(package.path()).await;
        let mut client = connect(address).await;
        assert_eq!(event(&mut client).await["type"], "session.created");
        client.close(None).await.unwrap();
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_rejects_invalid_tool_arguments_before_any_call_publication() {
        let package = tool_package(
            r#"<tool_call>{"id":"call_lookup","name":"lookup","arguments":{"city":42}}</tool_call>"#,
        );
        let (state, address, task) = server_for_package(package.path()).await;
        let mut client = connect(address).await;
        assert_eq!(event(&mut client).await["type"], "session.created");
        send(&mut client, json!({"type":"session.update","session":{"output_modalities":["text"],"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"tool_choice":"required"}})).await;
        assert_eq!(event(&mut client).await["type"], "session.updated");
        send(&mut client, json!({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}})).await;
        assert_eq!(event(&mut client).await["type"], "conversation.item.added");
        assert_eq!(event(&mut client).await["type"], "conversation.item.done");
        send(&mut client, json!({"type":"response.create"})).await;
        assert_eq!(event(&mut client).await["type"], "response.created");
        // An error must be the next wire event: no call item or argument delta
        // may escape to a harness before the complete parameter admission.
        let failure = event(&mut client).await;
        assert_eq!(failure["type"], "error", "{failure}");
        assert_eq!(failure["error"]["type"], "server_error");
        assert!(
            failure["error"]["message"]
                .as_str()
                .unwrap()
                .contains("do not satisfy realtime tool")
        );
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_external_harness_tool_result_round_trip() {
        let package = tool_package(
            r#"<tool_call>{"id":"call_lookup","name":"lookup","arguments":{"city":"Paris"}}</tool_call>"#,
        );
        let (state, address, task) = server_for_package(package.path()).await;
        let mut client = connect(address).await;
        assert_eq!(event(&mut client).await["type"], "session.created");
        send(&mut client, json!({"type":"session.update","session":{"output_modalities":["text"],"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"tool_choice":"required"}})).await;
        assert_eq!(event(&mut client).await["type"], "session.updated");
        send(&mut client, json!({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}})).await;
        assert_eq!(event(&mut client).await["type"], "conversation.item.added");
        assert_eq!(event(&mut client).await["type"], "conversation.item.done");
        for turn in 0..2 {
            send(&mut client, json!({"type":"response.create"})).await;
            let mut argument_deltas = String::new();
            loop {
                let output = event(&mut client).await;
                assert_ne!(output["type"], "error", "{output}");
                if output["type"] == "response.function_call_arguments.delta" {
                    assert_eq!(output["call_id"], "call_lookup");
                    argument_deltas.push_str(output["delta"].as_str().unwrap());
                }
                if output["type"] == "response.done" {
                    assert_eq!(output["response"]["output"][0]["name"], "lookup");
                    assert_eq!(output["response"]["usage"]["output_tokens"], 1);
                    break;
                }
            }
            assert_eq!(
                serde_json::from_str::<Value>(&argument_deltas).unwrap(),
                json!({"city":"Paris"})
            );
            if turn == 0 {
                send(&mut client, json!({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"wrong","output":"world"}})).await;
                assert_eq!(event(&mut client).await["type"], "error");
                send(&mut client, json!({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"call_lookup","output":"world"}})).await;
                assert_eq!(event(&mut client).await["type"], "conversation.item.added");
                assert_eq!(event(&mut client).await["type"], "conversation.item.done");
                send(&mut client, json!({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"call_lookup","output":"world"}})).await;
                assert_eq!(event(&mut client).await["type"], "error");
            }
        }
        client.close(None).await.unwrap();
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_runs_native_text_with_exact_budget_and_preserves_nonfatal_errors() {
        let (state, address, task) = server().await;
        let mut client = connect(address).await;
        let created = event(&mut client).await;
        assert_eq!(created["type"], "session.created");
        assert_eq!(
            created["session"]["audio"]["input"]["format"]["rate"],
            16000
        );
        // Idle/repeated cancellation is silent, including before the first turn.
        send(&mut client, json!({"type":"response.cancel"})).await;
        send(&mut client, json!({"type":"response.cancel"})).await;
        send(&mut client, json!({"type":"response.create"})).await;
        let empty = event(&mut client).await;
        assert_eq!(empty["type"], "error");
        assert_eq!(empty["error"]["type"], "conversation_empty");
        assert!(empty["error"]["code"].is_null());
        send(&mut client, json!({"type":"input_audio_buffer.commit"})).await;
        let error = event(&mut client).await;
        assert_eq!(error["error"]["type"], "input_audio_buffer_commit_empty");
        assert!(error["error"]["code"].is_null());
        send(&mut client,json!({"type":"session.update","session":{"instructions":"hello","output_modalities":["text"],"max_output_tokens":1}})).await;
        assert_eq!(event(&mut client).await["type"], "session.updated");
        send(&mut client,json!({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"world"}]}})).await;
        assert_eq!(event(&mut client).await["type"], "conversation.item.added");
        assert_eq!(event(&mut client).await["type"], "conversation.item.done");
        send(&mut client, json!({"type":"response.create"})).await;
        let mut text = String::new();
        loop {
            let result = event(&mut client).await;
            assert_ne!(result["type"], "error", "{result}");
            if result["type"] == "response.output_text.delta" {
                text.push_str(result["delta"].as_str().unwrap());
            }
            if result["type"] == "response.done" {
                assert_eq!(result["response"]["usage"]["output_tokens"], 1);
                break;
            }
        }
        assert_eq!(text, "hello");
        send(&mut client, json!({"type":"response.cancel"})).await;
        send(&mut client, json!({"type":"response.cancel"})).await;
        send(
            &mut client,
            json!({"type":"session.update","session":{"max_output_tokens":2}}),
        )
        .await;
        assert_eq!(event(&mut client).await["type"], "session.updated");
        send(
            &mut client,
            json!({"type":"session.update","session":{"instructions":"changed"}}),
        )
        .await;
        let invalid = event(&mut client).await;
        assert_eq!(invalid["type"], "error");
        assert_eq!(invalid["error"]["type"], "invalid_request_error");
        assert!(invalid["error"]["code"].is_null());
        client.close(None).await.unwrap();
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
        drop(state);
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_transcription_uses_committed_item_and_actual_native_tokens() {
        use base64::Engine as _;
        let (state, address, task) = server().await;
        let mut client = connect(address).await;
        let _ = event(&mut client).await;
        send(&mut client,json!({"type":"session.update","session":{"type":"transcription","audio":{"input":{"transcription":{"language":"zh","prompt":"names"}}}}})).await;
        assert_eq!(
            event(&mut client).await["session"]["object"],
            "realtime.transcription_session"
        );
        send(&mut client,json!({"type":"input_audio_buffer.append","audio":base64::engine::general_purpose::STANDARD.encode([0u8;640])})).await;
        send(&mut client, json!({"type":"input_audio_buffer.commit"})).await;
        let committed = event(&mut client).await;
        assert_eq!(committed["type"], "input_audio_buffer.committed");
        let item = committed["item_id"].clone();
        let mut transcript = String::new();
        loop {
            let result = event(&mut client).await;
            assert_ne!(result["type"], "error", "{result}");
            if result["type"] == "conversation.item.input_audio_transcription.delta" {
                assert_eq!(result["item_id"], item);
                transcript.push_str(result["delta"].as_str().unwrap());
            }
            if result["type"] == "conversation.item.input_audio_transcription.completed" {
                assert_eq!(result["item_id"], item);
                assert_eq!(result["transcript"], "hello world");
                break;
            }
        }
        assert_eq!(transcript, "hello world");
        client.close(None).await.unwrap();
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
        drop(state);
    }
    #[cfg(feature = "native-backend")]
    fn duplex_constraint_package() -> tempfile::TempDir {
        let good =
            "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>";
        let bad =
            "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"city\":\"Rome\"}}\n</tool_call>";
        let package = tool_package_from("realtime-constrained", good);
        let path = package.path().join("tokenizer.json");
        let mut tokenizer: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        tokenizer["model"]["vocab"]
            .as_object_mut()
            .unwrap()
            .remove("tok17");
        tokenizer["model"]["vocab"][bad] = json!(17);
        tokenizer["added_tokens"].as_array_mut().unwrap().push(json!({"id":17,"content":bad,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}));
        std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
        let path = package.path().join("inference_metadata.yaml");
        let metadata = std::fs::read_to_string(&path).unwrap().replacen(
            "default_mode: realtime",
            "default_mode: duplex",
            1,
        );
        assert!(metadata.contains("default_mode: duplex"));
        std::fs::write(path, metadata).unwrap();
        let path = package.path().join("model.onnx.textproto");
        let graph = std::fs::read_to_string(&path).unwrap();
        let original = "node { input: \"operation\" input: \"generate\" output: \"generating\" op_type: \"Equal\" }";
        assert!(graph.contains(original));
        let graph = graph.replace(original,concat!(
            "initializer { name: \"duplex_mode_id\" data_type: 7 int64_data: 2 }\n",
            "node { input: \"mode\" input: \"duplex_mode_id\" output: \"is_duplex\" op_type: \"Equal\" }\n",
            "node { input: \"operation\" input: \"three\" output: \"is_audio\" op_type: \"Equal\" }\n",
            "node { input: \"is_duplex\" input: \"is_audio\" output: \"duplex_generating\" op_type: \"And\" }\n",
            "node { input: \"operation\" input: \"generate\" output: \"turn_generating\" op_type: \"Equal\" }\n",
            "node { input: \"turn_generating\" input: \"duplex_generating\" output: \"generating\" op_type: \"Or\" }",
        ));
        let original = graph
            .lines()
            .find(|line| line.starts_with("initializer { name: \"ranked_logits\""))
            .unwrap();
        let logits = (0..32)
            .map(|id| {
                format!(
                    " float_data: {}",
                    match id {
                        16 => 1,
                        17 => 2,
                        _ => -1000,
                    }
                )
            })
            .collect::<String>();
        let graph = graph.replace(
            original,
            &format!("initializer {{ name: \"ranked_logits\" data_type: 1 dims: 32{logits} }}"),
        );
        std::fs::write(path, graph).unwrap();
        package
    }

    #[cfg(feature = "native-backend")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_duplex_masks_higher_ranked_invalid_tool_before_first_audio_selection() {
        use base64::Engine as _;
        let package = duplex_constraint_package();
        let (state, address, task) = server_for_package_with_config(
            package.path(),
            crate::ServerConfig {
                engine_config: onnx_genai_engine::EngineConfig {
                    decode_backend: onnx_genai_engine::EngineDecodeBackend::Native,
                    native_device: Some(onnx_genai_engine::NativeDecodeDevice::Cpu),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await;
        let mut client = connect(address).await;
        assert_eq!(event(&mut client).await["type"], "session.created");
        send(&mut client,json!({"type":"session.update","session":{"output_modalities":["text"],"tools":[{"type":"function","name":"lookup","parameters":{"$defs":{"Args":{"type":"object","properties":{"city":{"const":"Paris"}},"required":["city"],"additionalProperties":false}},"$ref":"#/$defs/Args"}}],"tool_choice":"required"}})).await;
        assert_eq!(event(&mut client).await["type"], "session.updated");
        // No response.create: an actual native Audio quantum must consume the
        // schema mask before ArgMax chooses between logits Paris=1 and Rome=2.
        let pcm: Vec<u8> = (0..320).flat_map(|_| 100_i16.to_le_bytes()).collect();
        send(&mut client,json!({"type":"input_audio_buffer.append","audio":base64::engine::general_purpose::STANDARD.encode(pcm)})).await;
        loop {
            let output = event(&mut client).await;
            assert_ne!(
                output["type"], "error",
                "duplex must mask Rome before Native selection: {output}"
            );
            if output["type"] == "response.done" {
                assert_eq!(output["response"]["status"], "completed");
                assert_eq!(output["response"]["output"][0]["type"], "function_call");
                assert_eq!(
                    output["response"]["output"][0]["arguments"],
                    r#"{"city":"Paris"}"#
                );
                break;
            }
        }
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }

    #[cfg(feature = "native-backend")]
    fn fragmented_duplex_constraint_package() -> tempfile::TempDir {
        let package = duplex_constraint_package();
        let prefix = "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"city\":\"";
        let path = package.path().join("tokenizer.json");
        let mut tokenizer: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for (id, text) in [
            (16, prefix),
            (17, "Rome\"}}\n</tool_call>"),
            (18, "Paris\"}}\n</tool_call>"),
        ] {
            let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
            let old = vocab
                .iter()
                .find(|(_, value)| value.as_u64() == Some(id))
                .unwrap()
                .0
                .clone();
            vocab.remove(&old);
            vocab.insert(text.into(), json!(id));
            tokenizer["added_tokens"]
                .as_array_mut()
                .unwrap()
                .retain(|token| token["id"] != id);
            tokenizer["added_tokens"].as_array_mut().unwrap().push(json!({"id":id,"content":text,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}));
        }
        tokenizer["decoder"] = json!({"type":"Fuse"});
        std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
        let path = package.path().join("model.onnx.textproto");
        let graph = std::fs::read_to_string(&path).unwrap();
        let waiting = "node { input: \"responding\" output: \"needs_input\" op_type: \"Not\" }";
        let done =
            "node { input: \"responding\" input: \"cancelling\" output: \"done\" op_type: \"Or\" }";
        assert!(graph.contains(waiting) && graph.contains(done));
        let graph=graph.replace(waiting,"initializer { name: \"wait_audio\" data_type: 9 int32_data: 1 }\nnode { input: \"wait_audio\" output: \"needs_input\" op_type: \"Identity\" }").replace(done,"");
        // Derive completion from the actual selected token. The first Audio
        // selects prefix=16; the next Audio must preserve grammar progress and
        // choose a permitted suffix. Re-arming each quantum repeats the prefix.
        let graph=graph.replace("node { input: \"tokens_out\" output: \"count\" op_type: \"Size\" }",concat!(
            "node { input: \"tokens_out\" output: \"count\" op_type: \"Size\" }\n",
            "initializer { name: \"prefix_id\" data_type: 7 int64_data: 16 }\n",
            "node { input: \"tokens_out\" input: \"prefix_id\" output: \"is_prefix_vector\" op_type: \"Equal\" }\n",
            "node { input: \"is_prefix_vector\" output: \"any_prefix\" op_type: \"Cast\" attribute { name: \"to\" type: INT i: 7 } }\n",
            "node { input: \"any_prefix\" output: \"prefix_count\" op_type: \"ReduceSum\" attribute { name: \"keepdims\" type: INT i: 0 } }\n",
            "node { input: \"prefix_count\" input: \"zero\" output: \"complete_piece\" op_type: \"Equal\" }\n",
            "node { input: \"responding\" input: \"complete_piece\" output: \"completed_response\" op_type: \"And\" }\n",
            "node { input: \"completed_response\" input: \"cancelling\" output: \"done\" op_type: \"Or\" }",
        ));
        let line = graph
            .lines()
            .find(|line| line.starts_with("initializer { name: \"ranked_logits\""))
            .unwrap();
        let logits = (0..32)
            .map(|id| {
                format!(
                    " float_data: {}",
                    match id {
                        16 => 3,
                        17 => 2,
                        18 => 1,
                        _ => -1000,
                    }
                )
            })
            .collect::<String>();
        std::fs::write(
            path,
            graph.replace(
                line,
                &format!("initializer {{ name: \"ranked_logits\" data_type: 1 dims: 32{logits} }}"),
            ),
        )
        .unwrap();
        package
    }

    #[cfg(feature = "native-backend")]
    fn native_constraint_config() -> crate::ServerConfig {
        crate::ServerConfig {
            engine_config: onnx_genai_engine::EngineConfig {
                decode_backend: onnx_genai_engine::EngineDecodeBackend::Native,
                native_device: Some(onnx_genai_engine::NativeDecodeDevice::Cpu),
                ..Default::default()
            },
            ..Default::default()
        }
    }
    #[cfg(feature = "native-backend")]
    async fn duplex_audio(client: &mut Client) {
        duplex_audio_sample(client, 100).await;
    }
    #[cfg(feature = "native-backend")]
    async fn duplex_audio_sample(client: &mut Client, sample: i16) {
        use base64::Engine as _;
        let pcm: Vec<u8> = (0..320).flat_map(|_| sample.to_le_bytes()).collect();
        send(client,json!({"type":"input_audio_buffer.append","audio":base64::engine::general_purpose::STANDARD.encode(pcm)})).await;
    }
    #[cfg(feature = "native-backend")]
    async fn duplex_tools(client: &mut Client, city: &str) {
        send(client,json!({"type":"session.update","session":{"output_modalities":["text"],"max_output_tokens":1,"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"city":{"const":city}},"required":["city"],"additionalProperties":false}}],"tool_choice":"required"}})).await;
        let updated = event(client).await;
        assert_eq!(updated["type"], "session.updated", "{updated}");
    }
    #[cfg(feature = "native-backend")]
    async fn duplex_prefix(client: &mut Client) {
        duplex_audio(client).await;
        let mut response_id = None;
        loop {
            let output = event(client).await;
            assert_ne!(output["type"], "error", "{output}");
            assert_ne!(
                output["type"], "response.done",
                "prefix must stay incomplete: {output}"
            );
            if output["type"] == "response.created" {
                assert_eq!(
                    output["response"]["output_modalities"],
                    json!(["text"]),
                    "{output}"
                );
                response_id = Some(output["response"]["id"].clone());
            }
            if output["type"] == "response.function_call_arguments.delta" {
                assert_eq!(Some(output["response_id"].clone()), response_id, "{output}");
                assert_eq!(output["delta"], r#"{"city":""#, "{output}");
                break;
            }
        }
    }
    #[cfg(feature = "native-backend")]
    async fn duplex_done(client: &mut Client, status: &str) -> Value {
        loop {
            let output = event(client).await;
            assert_ne!(output["type"], "error", "{output}");
            if status == "cancelled" {
                assert_ne!(
                    output["type"], "response.function_call_arguments.done",
                    "{output}"
                );
            }
            if output["type"] == "response.done" {
                assert_eq!(output["response"]["status"], status);
                return output;
            }
        }
    }

    #[cfg(feature = "native-backend")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_duplex_freezes_tool_plan_across_updates_and_rearms_after_cancel() {
        let package = fragmented_duplex_constraint_package();
        let (state, address, task) =
            server_for_package_with_config(package.path(), native_constraint_config()).await;
        let mut client = connect(address).await;
        let created = event(&mut client).await;
        assert_eq!(created["type"], "session.created", "{created}");
        duplex_tools(&mut client, "Paris").await;
        duplex_prefix(&mut client).await;
        // Tool definitions are part of frozen model conditioning. A rejected
        // update must not alter either Native masking or parameter admission.
        send(&mut client,json!({"type":"session.update","session":{"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"city":{"const":"Rome"}},"required":["city"],"additionalProperties":false}}],"tool_choice":"none"}})).await;
        let rejected = event(&mut client).await;
        assert_eq!(rejected["type"], "error", "{rejected}");
        assert_eq!(
            rejected["error"]["type"], "invalid_request_error",
            "{rejected}"
        );
        assert!(
            rejected["error"]["message"]
                .as_str()
                .unwrap()
                .contains("session.tools cannot change"),
            "{rejected}"
        );
        // Only response defaults may change after input. The active response
        // retains its Required/Paris constraint and original text modality.
        send(&mut client,json!({"type":"session.update","session":{"tool_choice":"none","output_modalities":["audio"],"max_output_tokens":1}})).await;
        let updated = event(&mut client).await;
        assert_eq!(updated["type"], "session.updated", "{updated}");
        assert_eq!(updated["session"]["tool_choice"], "none", "{updated}");
        assert_eq!(
            updated["session"]["output_modalities"],
            json!(["audio"]),
            "{updated}"
        );
        assert_eq!(updated["session"]["max_output_tokens"], 1, "{updated}");
        duplex_audio(&mut client).await;
        let done = loop {
            let output = event(&mut client).await;
            assert_ne!(output["type"], "error", "{output}");
            assert_ne!(
                output["type"], "response.output_audio.delta",
                "active text modality must remain frozen: {output}"
            );
            if output["type"] == "response.done" {
                assert_eq!(output["response"]["status"], "completed", "{output}");
                break output;
            }
        };
        assert_eq!(
            done["response"]["output"][0]["arguments"],
            r#"{"city":"Paris"}"#
        );
        assert_eq!(done["response"]["usage"]["output_tokens"], 2, "{done}");
        send(&mut client,json!({"type":"session.update","session":{"tool_choice":"required","output_modalities":["text"]}})).await;
        let restored = event(&mut client).await;
        assert_eq!(restored["type"], "session.updated", "{restored}");
        assert_eq!(restored["session"]["tool_choice"], "required", "{restored}");
        assert_eq!(
            restored["session"]["output_modalities"],
            json!(["text"]),
            "{restored}"
        );
        duplex_prefix(&mut client).await;
        send(&mut client, json!({"type":"response.cancel"})).await;
        let cancelled = duplex_done(&mut client, "cancelled").await;
        let call = cancelled["response"]["output"][0]["call_id"].clone();
        send(&mut client,json!({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":call,"output":"not admitted"}})).await;
        let rejected_result = event(&mut client).await;
        assert_eq!(rejected_result["type"], "error", "{rejected_result}");
        duplex_prefix(&mut client).await;
        duplex_audio(&mut client).await;
        let done = duplex_done(&mut client, "completed").await;
        assert_eq!(
            done["response"]["output"][0]["arguments"],
            r#"{"city":"Paris"}"#
        );
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }

    #[cfg(feature = "native-backend")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_duplex_constraint_progress_survives_poll_without_rearming() {
        let package = fragmented_duplex_constraint_package();
        let path = package.path().join("model.onnx.textproto");
        let graph = std::fs::read_to_string(&path).unwrap();
        let waiting =
            "node { input: \"wait_audio\" output: \"needs_input\" op_type: \"Identity\" }";
        assert!(graph.contains(waiting));
        std::fs::write(
            path,
            graph.replace(
                waiting,
                "node { input: \"responding\" output: \"needs_input\" op_type: \"Not\" }",
            ),
        )
        .unwrap();
        // Poll must sample too, not echo the empty conditioning token vector.
        let path = package.path().join("model.onnx.textproto");
        let graph = std::fs::read_to_string(&path).unwrap();
        let original = "node { input: \"is_duplex\" input: \"is_audio\" output: \"duplex_generating\" op_type: \"And\" }";
        assert!(graph.contains(original));
        let replacement = concat!(
            "initializer { name: \"poll_op\" data_type: 7 int64_data: 7 }\n",
            "node { input: \"operation\" input: \"poll_op\" output: \"is_poll\" op_type: \"Equal\" }\n",
            "node { input: \"is_audio\" input: \"is_poll\" output: \"samples_text\" op_type: \"Or\" }\n",
            "node { input: \"is_duplex\" input: \"samples_text\" output: \"duplex_generating\" op_type: \"And\" }",
        );
        std::fs::write(path, graph.replace(original, replacement)).unwrap();
        let (state, address, task) =
            server_for_package_with_config(package.path(), native_constraint_config()).await;
        let mut client = connect(address).await;
        assert_eq!(event(&mut client).await["type"], "session.created");
        duplex_tools(&mut client, "Paris").await;
        duplex_audio(&mut client).await;
        let done = duplex_done(&mut client, "completed").await;
        assert_eq!(
            done["response"]["output"][0]["arguments"],
            r#"{"city":"Paris"}"#
        );
        assert_eq!(done["response"]["usage"]["output_tokens"], 2);
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }

    #[cfg(feature = "native-backend")]
    fn eos_duplex_constraint_package(ignore_mask: bool) -> tempfile::TempDir {
        let package = fragmented_duplex_constraint_package();
        let path = package.path().join("model.onnx.textproto");
        let graph = std::fs::read_to_string(&path).unwrap();
        let waiting =
            "node { input: \"wait_audio\" output: \"needs_input\" op_type: \"Identity\" }";
        let audio = "node { input: \"is_duplex\" input: \"is_audio\" output: \"duplex_generating\" op_type: \"And\" }";
        let turn = "node { input: \"operation\" input: \"generate\" output: \"turn_generating\" op_type: \"Equal\" }";
        let done = "node { input: \"completed_response\" input: \"cancelling\" output: \"done\" op_type: \"Or\" }";
        assert!(
            graph.contains(waiting)
                && graph.contains(audio)
                && graph.contains(turn)
                && graph.contains(done)
        );
        let graph=graph.replace(waiting,"node { input: \"responding\" output: \"needs_input\" op_type: \"Not\" }")
            .replace(audio,concat!(
                "initializer { name: \"poll_op\" data_type: 7 int64_data: 7 }\n",
                "node { input: \"operation\" input: \"poll_op\" output: \"is_poll\" op_type: \"Equal\" }\n",
                "node { input: \"is_audio\" input: \"is_poll\" output: \"samples_text\" op_type: \"Or\" }\n",
                "node { input: \"is_duplex\" input: \"samples_text\" output: \"duplex_generating\" op_type: \"And\" }",
            )).replace(turn,concat!(
                "node { input: \"operation\" input: \"generate\" output: \"turn_start\" op_type: \"Equal\" }\n",
                "node { input: \"turn_start\" input: \"is_poll\" output: \"turn_generating\" op_type: \"Or\" }",
            )).replace(done,concat!(
                "node { input: \"tokens_out\" output: \"selected_id\" op_type: \"ReduceSum\" attribute { name: \"keepdims\" i: 0 type: INT } }\n",
                "node { input: \"selected_id\" input: \"three\" output: \"selected_eos\" op_type: \"Equal\" }\n",
                "node { input: \"selected_eos\" input: \"cancelling\" output: \"done\" op_type: \"Or\" }",
            ));
        // EOS has the highest native logit at every step. The bool vocabulary
        // mask must forbid it until the complete schema-conforming envelope.
        let line = graph
            .lines()
            .find(|line| line.starts_with("initializer { name: \"ranked_logits\""))
            .unwrap();
        let logits = (0..32)
            .map(|id| {
                format!(
                    " float_data: {}",
                    match id {
                        3 => 10,
                        16 => 3,
                        17 => 2,
                        18 => 1,
                        _ => -1000,
                    }
                )
            })
            .collect::<String>();
        let mut graph = graph.replace(
            line,
            &format!("initializer {{ name: \"ranked_logits\" data_type: 1 dims: 32{logits} }}"),
        );
        if ignore_mask {
            let selection = "node { input: \"effective_mask\" input: \"ranked_logits\" input: \"banned\" output: \"masked_logits\" op_type: \"Where\" }";
            assert!(graph.contains(selection));
            graph=graph.replace(selection,"node { input: \"all_allowed\" input: \"ranked_logits\" input: \"banned\" output: \"masked_logits\" op_type: \"Where\" }");
        }
        std::fs::write(path, graph).unwrap();
        package
    }

    #[cfg(feature = "native-backend")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_duplex_masks_early_eos_and_admits_declared_eos_after_tool_completion() {
        let package = eos_duplex_constraint_package(false);
        let mut config = native_constraint_config();
        config.max_output_tokens = 1;
        let (state, address, task) = server_for_package_with_config(package.path(), config).await;
        let mut client = connect(address).await;
        assert_eq!(event(&mut client).await["type"], "session.created");
        duplex_tools(&mut client, "Paris").await;
        // Duplex is input-driven and intentionally unlimited, even when the
        // server/session turn defaults are one token. Native EOS ends it.
        duplex_audio(&mut client).await;
        let done = duplex_done(&mut client, "completed").await;
        assert_eq!(
            done["response"]["output"][0]["arguments"],
            r#"{"city":"Paris"}"#
        );
        assert_eq!(done["response"]["output"][0]["status"], "completed");
        assert_eq!(done["response"]["usage"]["output_tokens"], 3);
        // A second automatic response must arm from the beginning, after EOS
        // cleared the first grammar; no stale EOF state may select EOS first.
        duplex_audio(&mut client).await;
        let next = duplex_done(&mut client, "completed").await;
        assert_ne!(done["response"]["id"], next["response"]["id"]);
        assert_eq!(
            next["response"]["output"][0]["arguments"],
            r#"{"city":"Paris"}"#
        );
        assert_eq!(next["response"]["usage"]["output_tokens"], 3);
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }

    #[cfg(feature = "native-backend")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_duplex_rejects_native_eos_when_graph_ignores_required_mask() {
        let package = eos_duplex_constraint_package(true);
        let (state, address, task) =
            server_for_package_with_config(package.path(), native_constraint_config()).await;
        let mut client = connect(address).await;
        assert_eq!(event(&mut client).await["type"], "session.created");
        duplex_tools(&mut client, "Paris").await;
        duplex_audio(&mut client).await;
        loop {
            let output = event(&mut client).await;
            assert_ne!(
                output["type"], "response.done",
                "forbidden EOS cannot complete: {output}"
            );
            assert_ne!(
                output["type"], "response.function_call_arguments.done",
                "{output}"
            );
            assert_ne!(output["type"], "response.output_item.done", "{output}");
            if output["type"] == "error" {
                assert_eq!(output["error"]["type"], "server_error");
                assert!(
                    output["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains("token 3 forbidden by token_mask"),
                    "{output}"
                );
                break;
            }
        }
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }

    #[cfg(feature = "native-backend")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_native_turn_limit_truncates_tool_prefix_then_override_reaches_eos() {
        let package = eos_duplex_constraint_package(false);
        let path = package.path().join("inference_metadata.yaml");
        let metadata = std::fs::read_to_string(&path).unwrap().replacen(
            "default_mode: duplex",
            "default_mode: realtime",
            1,
        );
        assert!(metadata.contains("default_mode: realtime"));
        std::fs::write(path, metadata).unwrap();
        let (state, address, task) =
            server_for_package_with_config(package.path(), native_constraint_config()).await;
        let mut client = connect(address).await;
        assert_eq!(event(&mut client).await["type"], "session.created");
        duplex_tools(&mut client, "Paris").await;
        for limit in [1, 4] {
            send(&mut client,json!({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}})).await;
            assert_eq!(event(&mut client).await["type"], "conversation.item.added");
            assert_eq!(event(&mut client).await["type"], "conversation.item.done");
            send(
                &mut client,
                json!({"type":"response.create","response":{"max_output_tokens":limit}}),
            )
            .await;
            let done = loop {
                let output = event(&mut client).await;
                assert_ne!(output["type"], "error", "{output}");
                if limit == 1 {
                    assert_ne!(
                        output["type"], "response.function_call_arguments.done",
                        "truncated call is not admitted: {output}"
                    );
                }
                if output["type"] == "response.done" {
                    break output;
                }
            };
            assert_eq!(done["response"]["status"], "completed");
            let call = &done["response"]["output"][0];
            if limit == 1 {
                assert_eq!(done["response"]["usage"]["output_tokens"], 1);
                assert_eq!(call["status"], "incomplete");
                assert_eq!(call["arguments"], r#"{"city":""#);
                send(&mut client,json!({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":call["call_id"],"output":"not admitted"}})).await;
                assert_eq!(event(&mut client).await["type"], "error");
            } else {
                assert_eq!(done["response"]["usage"]["output_tokens"], 3);
                assert_eq!(call["status"], "completed");
                assert_eq!(call["arguments"], r#"{"city":"Paris"}"#);
            }
        }
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }

    #[cfg(feature = "native-backend")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_duplex_rejects_missing_mask_and_unsupported_forcing_before_audio() {
        use base64::Engine as _;
        for xml in [false, true] {
            let package = duplex_constraint_package();
            let path = package.path().join("inference_metadata.yaml");
            let mut metadata: serde_yaml::Value =
                serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            if xml {
                metadata["package"]["tool_protocol"] =
                    serde_yaml::to_value(json!({"identity":"atem-xml","version":"v1"})).unwrap();
            } else {
                metadata["profiles"]["live"]["realtime"]["inputs"]
                    .as_mapping_mut()
                    .unwrap()
                    .remove(serde_yaml::Value::String("token_mask".into()));
                metadata["pipeline"]["workflow"]["inputs"]
                    .as_mapping_mut()
                    .unwrap()
                    .remove(serde_yaml::Value::String("request.token_mask".into()));
                metadata["pipeline"]["workflow"]["steps"][0]["inputs"]
                    .as_mapping_mut()
                    .unwrap()
                    .remove(serde_yaml::Value::String("token_mask".into()));
                // Keep the exported graph valid: absent host binding becomes an
                // explicit empty optional input default on this fixture graph.
                let graph_path = package.path().join("model.onnx.textproto");
                let graph = std::fs::read_to_string(&graph_path).unwrap();
                let input = "input { name: \"token_mask\" type { tensor_type { elem_type: 9 shape { dim { dim_param: \"vocabulary_mask\" } } } } }";
                assert!(graph.contains(input));
                std::fs::write(
                    graph_path,
                    graph.replace(
                        input,
                        "initializer { name: \"token_mask\" data_type: 9 dims: 0 }",
                    ),
                )
                .unwrap();
            }
            std::fs::write(path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
            let (state, address, task) =
                server_for_package_with_config(package.path(), native_constraint_config()).await;
            let mut client = connect(address).await;
            let created = event(&mut client).await;
            assert_eq!(created["type"], "session.created", "xml={xml}: {created}");
            duplex_tools(&mut client, "Paris").await;
            duplex_audio(&mut client).await;
            let error = event(&mut client).await;
            assert_eq!(error["type"], "error", "xml={xml}: {error}");
            assert_eq!(
                error["error"]["type"], "invalid_request_error",
                "xml={xml}: {error}"
            );
            assert!(
                error["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains(if xml { "forcing plan" } else { "token_mask" }),
                "{error}"
            );
            // A legal instructions update proves the rejected append did not
            // start/freeze the conversation. Duplex commit is always rejected,
            // so an empty-commit error would not prove this property.
            send(&mut client,json!({"type":"session.update","session":{"instructions":"hello","tool_choice":"none","output_modalities":["audio"]}})).await;
            let updated = event(&mut client).await;
            assert_eq!(updated["type"], "session.updated", "xml={xml}: {updated}");
            assert_eq!(updated["session"]["instructions"], "hello", "{updated}");
            assert_eq!(updated["session"]["tool_choice"], "none", "{updated}");
            // Rejected PCM was 100. A single accepted 300-valued frame must be
            // the first Native output: queued rejected PCM would emit 100, and
            // previously executed PCM would contaminate the fixture's KV sum.
            duplex_audio_sample(&mut client, 300).await;
            let mut frames = Vec::new();
            let mut response_id = None;
            let done = loop {
                let output = event(&mut client).await;
                assert_ne!(output["type"], "error", "xml={xml}: {output}");
                assert_ne!(
                    output["type"], "response.function_call_arguments.done",
                    "{output}"
                );
                if output["type"] == "response.created" {
                    assert_eq!(
                        output["response"]["output_modalities"],
                        json!(["audio"]),
                        "{output}"
                    );
                    response_id = Some(output["response"]["id"].clone());
                }
                if output["type"] == "response.output_audio.delta" {
                    assert_eq!(Some(output["response_id"].clone()), response_id, "{output}");
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(output["delta"].as_str().unwrap())
                        .unwrap();
                    assert_eq!(bytes.len(), 320 * 2, "{output}");
                    frames.push(
                        bytes
                            .chunks_exact(2)
                            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                            .collect::<Vec<_>>(),
                    );
                }
                if output["type"] == "response.done" {
                    assert_eq!(
                        Some(output["response"]["id"].clone()),
                        response_id,
                        "{output}"
                    );
                    assert_eq!(output["response"]["status"], "completed", "{output}");
                    assert_eq!(output["response"]["usage"]["output_tokens"], 1, "{output}");
                    break output;
                }
            };
            assert_eq!(frames, vec![vec![300; 320]], "xml={xml}: {done}");
            drop(client);
            let model = state.registry.resolve("native-realtime").unwrap().unwrap();
            tokio::task::spawn_blocking(move || model.engine.shutdown())
                .await
                .unwrap();
            task.abort();
            let _ = task.await;
        }
    }

    #[cfg(feature = "native-backend")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_owner_arms_generic_constraint_without_consuming_a_quantum() {
        let package = duplex_constraint_package();
        let (state, _, task) =
            server_for_package_with_config(package.path(), native_constraint_config()).await;
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        let (mut native, initial) = model
            .engine
            .open_realtime(RealtimeMode::Duplex, vec![4])
            .await
            .unwrap();
        assert_eq!(initial.tokens, vec![4]);
        // A rejected compilation is an admission error, not a KV poison event.
        assert!(
            native
                .arm_constraint(Some(onnx_genai_engine::GenerateConstraint::Regex {
                    pattern: "[".into(),
                    prefix: String::new()
                }))
                .await
                .is_err()
        );
        let good =
            "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>";
        native
            .arm_constraint(Some(onnx_genai_engine::GenerateConstraint::Regex {
                pattern: regex::escape(good),
                prefix: String::new(),
            }))
            .await
            .unwrap();
        let output = native
            .step(RealtimeInputOwned::Audio(vec![0.1; 320]))
            .await
            .unwrap();
        assert_eq!(output.tokens, vec![16]);
        assert_eq!(output.output_tokens, 1);
        assert!(output.done);
        // Completion clears the response grammar; no plan means higher Rome wins.
        native.arm_constraint(None).await.unwrap();
        let output = native
            .step(RealtimeInputOwned::Audio(vec![0.1; 320]))
            .await
            .unwrap();
        assert_eq!(output.tokens, vec![17]);
        drop(native);
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn websocket_duplex_publishes_audio_before_commit_and_acknowledges_native_cancel() {
        use base64::Engine as _;
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime");
        let package = tempfile::tempdir().unwrap();
        for file in ["model.onnx.textproto", "tokenizer.json"] {
            std::fs::copy(root.join(file), package.path().join(file)).unwrap();
        }
        let metadata = std::fs::read_to_string(root.join("inference_metadata.yaml"))
            .unwrap()
            .replacen("default_mode: realtime", "default_mode: duplex", 1);
        std::fs::write(package.path().join("inference_metadata.yaml"), metadata).unwrap();
        let (state, address, task) = server_for_package(package.path()).await;
        let mut client = connect(address).await;
        assert_eq!(event(&mut client).await["type"], "session.created");
        let first_audio: Vec<u8> = (0..320).flat_map(|_| 100_i16.to_le_bytes()).collect();
        send(&mut client,json!({"type":"input_audio_buffer.append","audio":base64::engine::general_purpose::STANDARD.encode(first_audio)})).await;
        let mut response_id = None;
        loop {
            let result = event(&mut client).await;
            assert_ne!(result["type"], "error", "{result}");
            if result["type"] == "response.created" {
                response_id = Some(result["response"]["id"].clone());
            }
            if result["type"] == "response.output_audio.delta" {
                assert_eq!(
                    base64::engine::general_purpose::STANDARD
                        .decode(result["delta"].as_str().unwrap())
                        .unwrap()
                        .len(),
                    640
                );
                break;
            }
        }
        assert!(response_id.is_some());
        send(&mut client, json!({"type":"response.cancel"})).await;
        loop {
            let result = event(&mut client).await;
            assert_ne!(result["type"], "error", "{result}");
            if result["type"] == "response.done" {
                assert_eq!(Some(result["response"]["id"].clone()), response_id);
                assert_eq!(result["response"]["status"], "cancelled");
                break;
            }
        }
        send(&mut client,json!({"type":"input_audio_buffer.append","audio":base64::engine::general_purpose::STANDARD.encode([0u8;640])})).await;
        loop {
            let result = event(&mut client).await;
            assert_ne!(result["type"], "error", "{result}");
            if result["type"] == "response.created" {
                assert_ne!(Some(result["response"]["id"].clone()), response_id);
            }
            if result["type"] == "response.output_audio.delta" {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(result["delta"].as_str().unwrap())
                    .unwrap();
                assert!(
                    bytes.iter().all(|value| *value == 0),
                    "next duplex response must not retain the cancelled codec/history stream"
                );
                break;
            }
        }
        send(&mut client, json!({"type":"response.cancel"})).await;
        loop {
            let result = event(&mut client).await;
            assert_ne!(result["type"], "error", "{result}");
            if result["type"] == "response.done" {
                break;
            }
        }
        client.close(None).await.unwrap();
        let close = tokio::time::timeout(std::time::Duration::from_secs(5), client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(
            matches!(close, Message::Close(_)),
            "server must flush its close handshake"
        );
        drop(client);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
        drop(state);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ordinary_get_does_not_reserve_a_realtime_worker_or_pretend_to_upgrade() {
        use tower::ServiceExt;
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime");
        let state = AppState::load(&root, Some("native-realtime".into())).unwrap();
        let response = crate::app(state.clone())
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/realtime?model=native-realtime")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
        let model = state.registry.resolve("native-realtime").unwrap().unwrap();
        assert_eq!(model.engine.worker_statuses()[0].worker.active_turns, 0);
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
    }
}
