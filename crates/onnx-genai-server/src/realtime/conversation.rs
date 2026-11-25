//! Conversation/response transitions independent of native scheduling and I/O.
use super::settings::{ResponseOptions, SessionSettings};
use onnx_genai_metadata::{RealtimeMode, ToolCall};
use serde_json::{Value, json};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ResponseId(String);
impl ResponseId {
    pub(super) fn new(id: String) -> Self {
        Self(id)
    }
    pub(super) fn as_str(&self) -> &str {
        &self.0
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ItemId(String);
impl ItemId {
    pub(super) fn new(id: String) -> Self {
        Self(id)
    }
    pub(super) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProtocolErrorKind {
    InvalidRequest,
    EmptyConversation,
    EmptyAudioCommit,
}
impl ProtocolErrorKind {
    pub(super) fn error_type(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request_error",
            Self::EmptyConversation => "conversation_empty",
            Self::EmptyAudioCommit => "input_audio_buffer_commit_empty",
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub(super) struct ProtocolError {
    pub(super) kind: ProtocolErrorKind,
    message: String,
}
impl ProtocolError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            kind: ProtocolErrorKind::InvalidRequest,
            message: message.into(),
        }
    }
    fn typed(kind: ProtocolErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ResponseKind {
    Realtime,
    Transcription { item: ItemId },
}
#[derive(Debug, Clone)]
pub(super) struct ActiveResponse {
    pub(super) id: ResponseId,
    pub(super) kind: ResponseKind,
    pub(super) options: ResponseOptions,
}

#[derive(Debug)]
pub(super) struct ResponseAdmission {
    pub(super) commit_input: bool,
    pub(super) events: Vec<Value>,
}

#[derive(Default)]
pub(super) struct Conversation {
    started: bool,
    input_turn: bool,
    audio_uncommitted: bool,
    committed_audio: Option<ItemId>,
    tool_results: bool,
    last_item: Option<ItemId>,
    active: Option<ActiveResponse>,
    pending_calls: BTreeSet<String>,
}
impl Conversation {
    pub(super) fn discard_duplex_input(&mut self) {
        self.input_turn = false;
        self.audio_uncommitted = false;
        self.committed_audio = None;
        self.tool_results = false;
    }
    pub(super) fn has_audio_to_commit(&self) -> bool {
        self.audio_uncommitted
    }
    pub(super) fn started(&self) -> bool {
        self.started
    }
    pub(super) fn active(&self) -> Option<&ActiveResponse> {
        self.active.as_ref()
    }
    pub(super) fn audio_appended(&mut self, nonempty: bool) {
        if nonempty {
            self.started = true;
            self.input_turn = true;
            self.audio_uncommitted = true;
            self.pending_calls.clear();
        }
    }
    /// Upstream acknowledges clear without undoing already ingested audio.
    pub(super) fn clear_audio(&self) -> Value {
        json!({"type":"input_audio_buffer.cleared"})
    }
    pub(super) fn commit_audio(&mut self, item: ItemId) -> Result<Vec<Value>, ProtocolError> {
        if !self.audio_uncommitted {
            return Err(ProtocolError::typed(
                ProtocolErrorKind::EmptyAudioCommit,
                "input_audio_buffer.commit: the audio buffer is empty; append PCM audio before committing",
            ));
        }
        self.audio_uncommitted = false;
        self.input_turn = true;
        self.committed_audio = Some(item.clone());
        let mut events = vec![
            json!({"type":"input_audio_buffer.committed","previous_item_id":self.last_item.as_ref().map(ItemId::as_str),"item_id":item.as_str()}),
        ];
        events.extend(self.item_events(
            item,
            "message",
            json!({"role":"user","content":[{"type":"input_audio"}]}),
        ));
        Ok(events)
    }
    pub(super) fn user_text(
        &mut self,
        item: ItemId,
        text: &str,
        settings: &SessionSettings,
    ) -> Result<Vec<Value>, ProtocolError> {
        if settings.mode != RealtimeMode::Realtime {
            return Err(ProtocolError::invalid(
                "conversation.item.create only accepts text user messages in a turn-based realtime session",
            ));
        }
        self.started = true;
        self.input_turn = true;
        self.pending_calls.clear();
        Ok(self.item_events(
            item,
            "message",
            json!({"role":"user","content":[{"type":"input_text","text":text}]}),
        ))
    }
    pub(super) fn tool_result(
        &mut self,
        item: ItemId,
        call_id: &str,
        output: &str,
        settings: &SessionSettings,
    ) -> Result<Vec<Value>, ProtocolError> {
        if settings.mode != RealtimeMode::Realtime
            || self.active.is_some()
            || self.input_turn
            || !self.pending_calls.contains(call_id)
        {
            return Err(ProtocolError::invalid(format!(
                "conversation.item.create: function_call_output.call_id {call_id:?} must answer an outstanding call of the completed previous response, before a new user turn"
            )));
        }
        self.pending_calls.remove(call_id);
        self.tool_results = true;
        Ok(self.item_events(
            item,
            "function_call_output",
            json!({"call_id":call_id,"output":output}),
        ))
    }
    fn item_events(&mut self, id: ItemId, kind: &str, fields: Value) -> Vec<Value> {
        let previous = self.last_item.as_ref().map(ItemId::as_str);
        let mut item = fields;
        item["id"] = json!(id.as_str());
        item["object"] = json!("realtime.item");
        item["type"] = json!(kind);
        item["status"] = json!("completed");
        let events = vec![
            json!({"type":"conversation.item.added","previous_item_id":previous,"item":item}),
            json!({"type":"conversation.item.done","previous_item_id":previous,"item":item}),
        ];
        self.last_item = Some(id);
        events
    }
    /// Admit response options and consume any pending input commit atomically.
    /// The runner executes native Commit before Generate and these events.
    pub(super) fn begin_response(
        &mut self,
        id: ResponseId,
        audio_item: ItemId,
        settings: &SessionSettings,
        overrides: &Value,
        audio_output: bool,
    ) -> Result<ResponseAdmission, ProtocolError> {
        if settings.mode == RealtimeMode::Duplex {
            return Err(ProtocolError::invalid(
                "response.create: duplex models answer while listening; append audio to open a response",
            ));
        }
        if settings.mode == RealtimeMode::Transcription {
            return Err(ProtocolError::invalid(
                "response.create is not supported in a transcription session; commit input audio",
            ));
        }
        if self.active.is_some() {
            return Err(ProtocolError::invalid(
                "response.create: a response is already active; cancel or finish it before creating another",
            ));
        }
        if !self.input_turn && !self.tool_results {
            return Err(ProtocolError::typed(
                ProtocolErrorKind::EmptyConversation,
                "response.create: commit audio or create a message/tool result first",
            ));
        }
        let options = settings
            .response
            .updated(overrides, audio_output)
            .map_err(|error| ProtocolError::invalid(error.to_string()))?;
        let needs_commit = self.audio_uncommitted;
        let events = if needs_commit {
            self.commit_audio(audio_item)?
        } else {
            Vec::new()
        };
        self.active = Some(ActiveResponse {
            id,
            kind: ResponseKind::Realtime,
            options,
        });
        self.input_turn = false;
        self.tool_results = false;
        self.committed_audio = None;
        Ok(ResponseAdmission {
            commit_input: needs_commit,
            events,
        })
    }
    /// Duplex opens only with its first audio, remains open through silent
    /// publications, and is not capped by ordinary turn response defaults.
    pub(super) fn begin_duplex(
        &mut self,
        id: ResponseId,
        settings: &SessionSettings,
    ) -> Result<bool, ProtocolError> {
        if settings.mode != RealtimeMode::Duplex {
            return Err(ProtocolError::invalid(
                "duplex response requires a duplex realtime profile",
            ));
        }
        if self.active.is_some() {
            return Ok(false);
        }
        let mut options = settings.response.clone();
        options.budget = super::settings::OutputBudget::Unlimited;
        self.active = Some(ActiveResponse {
            id,
            kind: ResponseKind::Realtime,
            options,
        });
        Ok(true)
    }
    pub(super) fn begin_transcription(
        &mut self,
        id: ResponseId,
        item: ItemId,
        settings: &SessionSettings,
    ) -> Result<(), ProtocolError> {
        if settings.mode != RealtimeMode::Transcription
            || self.active.is_some()
            || self.audio_uncommitted
            || self.committed_audio.as_ref() != Some(&item)
        {
            return Err(ProtocolError::invalid(
                "audio transcription requires a transcription session with no active response; finish the previous transcription first",
            ));
        }
        self.active = Some(ActiveResponse {
            id,
            kind: ResponseKind::Transcription { item },
            options: settings.response.clone(),
        });
        self.input_turn = false;
        self.committed_audio = None;
        Ok(())
    }
    pub(super) fn complete(
        &mut self,
        id: &ResponseId,
        calls: &[ToolCall],
        last_output_item: Option<ItemId>,
    ) -> Result<(), ProtocolError> {
        let Some(active) = self.active.as_ref() else {
            return Err(ProtocolError::invalid(
                "native realtime completion has no active response; close the connection and inspect response scheduling",
            ));
        };
        if &active.id != id {
            return Err(ProtocolError::invalid(
                "native realtime completion belongs to a different response; close the connection rather than publishing stale output",
            ));
        }
        if matches!(active.kind, ResponseKind::Transcription { .. }) && !calls.is_empty() {
            return Err(ProtocolError::invalid(
                "transcription workflow emitted tool calls; select a compatible transcription profile",
            ));
        }
        let mut pending = BTreeSet::new();
        for call in calls {
            if call.id.is_empty() || !pending.insert(call.id.clone()) {
                return Err(ProtocolError::invalid(
                    "native realtime response contains empty or duplicate tool call IDs; repair the declared tool protocol",
                ));
            }
        }
        self.active = None;
        self.pending_calls = pending;
        if let Some(item) = last_output_item {
            self.last_item = Some(item);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::settings::{Modalities, OutputBudget};
    use super::*;
    use std::num::NonZeroU32;
    fn settings(mode: RealtimeMode) -> SessionSettings {
        SessionSettings::new(
            mode,
            NonZeroU32::new(16000).unwrap(),
            OutputBudget::Tokens(NonZeroU32::new(4096).unwrap()),
        )
    }
    fn item(id: &str) -> ItemId {
        ItemId::new(id.into())
    }
    fn response(id: &str) -> ResponseId {
        ResponseId::new(id.into())
    }
    #[test]
    fn clear_acknowledges_ingested_audio_without_losing_the_committable_turn() {
        let mut conversation = Conversation::default();
        assert_eq!(
            conversation.commit_audio(item("empty")).unwrap_err().kind,
            ProtocolErrorKind::EmptyAudioCommit
        );
        conversation.audio_appended(false);
        assert!(!conversation.started());
        conversation.audio_appended(true);
        assert!(conversation.started());
        assert_eq!(
            conversation.clear_audio()["type"],
            "input_audio_buffer.cleared"
        );
        let events = conversation.commit_audio(item("audio1")).unwrap();
        assert_eq!(events[0]["item_id"], "audio1");
        assert_eq!(events[1]["item"]["content"][0]["type"], "input_audio");
        assert_eq!(
            conversation.commit_audio(item("empty2")).unwrap_err().kind,
            ProtocolErrorKind::EmptyAudioCommit
        );
    }
    #[test]
    fn response_auto_commit_consumes_input_and_does_not_leave_a_phantom_turn() {
        let mut conversation = Conversation::default();
        let settings = settings(RealtimeMode::Realtime);
        assert_eq!(
            conversation
                .begin_response(
                    response("empty"),
                    item("unused"),
                    &settings,
                    &json!({}),
                    true
                )
                .unwrap_err()
                .kind,
            ProtocolErrorKind::EmptyConversation
        );
        conversation.audio_appended(true);
        let admission = conversation
            .begin_response(
                response("first"),
                item("audio"),
                &settings,
                &json!({"output_modalities":["text"]}),
                true,
            )
            .unwrap();
        assert!(admission.commit_input);
        assert_eq!(admission.events[0]["type"], "input_audio_buffer.committed");
        assert_eq!(
            conversation.active().unwrap().options.modalities,
            Modalities::Text
        );
        assert_eq!(conversation.active().unwrap().id.as_str(), "first");
        assert!(
            conversation
                .begin_response(
                    response("second"),
                    item("unused"),
                    &settings,
                    &json!({}),
                    true
                )
                .is_err()
        );
        assert!(
            conversation
                .complete(&response("stale"), &[], None)
                .is_err()
        );
        conversation
            .complete(&response("first"), &[], Some(item("assistant")))
            .unwrap();
        assert_eq!(
            conversation
                .begin_response(
                    response("empty2"),
                    item("unused"),
                    &settings,
                    &json!({}),
                    true
                )
                .unwrap_err()
                .kind,
            ProtocolErrorKind::EmptyConversation
        );
        assert!(
            conversation.started(),
            "prompt immutability lasts beyond response completion"
        );
        let user = conversation
            .user_text(item("user"), "next", &settings)
            .unwrap();
        assert_eq!(user[0]["previous_item_id"], "assistant");
    }
    #[test]
    fn external_tool_results_must_match_the_previous_completed_response_once() {
        let mut conversation = Conversation::default();
        let settings = settings(RealtimeMode::Realtime);
        conversation
            .user_text(item("user"), "weather", &settings)
            .unwrap();
        conversation
            .begin_response(
                response("first"),
                item("unused"),
                &settings,
                &json!({}),
                true,
            )
            .unwrap();
        assert!(
            conversation
                .tool_result(item("early"), "call1", "x", &settings)
                .is_err()
        );
        let calls = vec![ToolCall {
            id: "call1".into(),
            name: "weather".into(),
            arguments: "{}".into(),
        }];
        conversation
            .complete(&response("first"), &calls, Some(item("assistant_call")))
            .unwrap();
        assert!(
            conversation
                .tool_result(item("wrong"), "other", "x", &settings)
                .is_err()
        );
        let events = conversation
            .tool_result(item("result"), "call1", "sunny", &settings)
            .unwrap();
        assert_eq!(events[0]["item"]["type"], "function_call_output");
        assert_eq!(events[0]["item"]["call_id"], "call1");
        assert!(
            conversation
                .tool_result(item("duplicate"), "call1", "again", &settings)
                .is_err()
        );
        assert!(
            !conversation
                .begin_response(
                    response("answer"),
                    item("unused"),
                    &settings,
                    &json!({}),
                    true
                )
                .unwrap()
                .commit_input
        );
    }
    #[test]
    fn duplex_and_transcription_have_distinct_response_lifecycles() {
        let mut conversation = Conversation::default();
        let duplex = settings(RealtimeMode::Duplex);
        assert!(
            conversation
                .user_text(item("user"), "text", &duplex)
                .is_err()
        );
        assert!(
            conversation
                .begin_response(
                    response("invalid"),
                    item("unused"),
                    &duplex,
                    &json!({}),
                    true
                )
                .is_err()
        );
        conversation.audio_appended(true);
        assert!(
            conversation
                .begin_duplex(response("duplex1"), &duplex)
                .unwrap()
        );
        assert_eq!(
            conversation.active().unwrap().options.budget,
            OutputBudget::Unlimited
        );
        assert!(
            !conversation
                .begin_duplex(response("duplex2"), &duplex)
                .unwrap()
        );
        conversation
            .complete(&response("duplex1"), &[], None)
            .unwrap();
        assert!(
            conversation
                .begin_duplex(response("duplex2"), &duplex)
                .unwrap()
        );
        conversation
            .complete(&response("duplex2"), &[], None)
            .unwrap();
        let transcription = settings(RealtimeMode::Transcription);
        let mut conversation = Conversation::default();
        assert!(
            conversation
                .begin_transcription(
                    response("empty_asr"),
                    item("invented_audio"),
                    &transcription
                )
                .is_err()
        );
        conversation.audio_appended(true);
        conversation.commit_audio(item("audio")).unwrap();
        conversation
            .begin_transcription(response("asr"), item("audio"), &transcription)
            .unwrap();
        assert_eq!(
            conversation.active().unwrap().kind,
            ResponseKind::Transcription {
                item: item("audio")
            }
        );
        assert!(
            conversation
                .begin_transcription(response("asr2"), item("audio2"), &transcription)
                .is_err()
        );
        conversation.complete(&response("asr"), &[], None).unwrap();
    }
}
