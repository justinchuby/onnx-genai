//! Realtime wire events over already decoded, policy-admitted model output.
//! This module neither parses model tool syntax nor executes tools.
use anyhow::{Context, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use onnx_genai_metadata::{MAX_TOOL_CALLS, ToolCall};
use serde_json::{Value, json};
use std::{collections::BTreeMap, num::NonZeroUsize};

/// Consuming `finish` makes publishing a second terminal event impossible.
/// Event IDs are stamped once by the connection's common event sender.
pub(super) struct ResponseEvents {
    id: String,
    message_id: String,
    message_index: Option<usize>,
    text_index: Option<usize>,
    audio_index: Option<usize>,
    parts: Vec<Value>,
    output: Vec<Value>,
    calls: BTreeMap<String, usize>,
    retained_bytes: usize,
    limit: NonZeroUsize,
    max_audio_samples: NonZeroUsize,
    output_tokens: usize,
}
impl ResponseEvents {
    pub(super) fn new(
        id: String,
        message_id: String,
        modalities: Value,
        limit: NonZeroUsize,
        max_audio_samples: NonZeroUsize,
    ) -> (Self, Value) {
        let created = json!({"type":"response.created", "response":{"id":id,"object":"realtime.response","status":"in_progress","output_modalities":modalities,"output":[]}});
        (
            Self {
                id,
                message_id,
                message_index: None,
                text_index: None,
                audio_index: None,
                parts: Vec::new(),
                output: Vec::new(),
                calls: BTreeMap::new(),
                retained_bytes: 0,
                limit,
                max_audio_samples,
                output_tokens: 0,
            },
            created,
        )
    }
    fn reserve_text(&mut self, bytes: usize) -> anyhow::Result<()> {
        let total = self.retained_bytes.checked_add(bytes).context(
            "realtime response text size overflow; cancel the response and start a new turn",
        )?;
        ensure!(
            total <= self.limit.get(),
            "realtime response text/tool arguments exceed the {}-byte retained-text limit; cancel the response or configure a larger response budget",
            self.limit
        );
        self.retained_bytes = total;
        Ok(())
    }
    pub(super) fn usage(&mut self, output_tokens: usize) -> anyhow::Result<()> {
        self.output_tokens = self
            .output_tokens
            .checked_add(output_tokens)
            .context("realtime response output token count overflow; start a new response")?;
        Ok(())
    }
    fn message(&mut self, events: &mut Vec<Value>) -> usize {
        if let Some(index) = self.message_index {
            return index;
        }
        let index = self.output.len();
        let item = json!({"id":self.message_id,"object":"realtime.item","type":"message","role":"assistant","status":"in_progress","content":[]});
        events.push(json!({"type":"response.output_item.added","response_id":self.id,"output_index":index,"item":item}));
        self.output.push(item);
        self.message_index = Some(index);
        index
    }
    fn part(&mut self, audio: bool, events: &mut Vec<Value>) -> usize {
        if let Some(index) = if audio {
            self.audio_index
        } else {
            self.text_index
        } {
            return index;
        }
        let output_index = self.message(events);
        let index = self.parts.len();
        let part = if audio {
            json!({"type":"output_audio","transcript":""})
        } else {
            json!({"type":"output_text","text":""})
        };
        events.push(json!({"type":"response.content_part.added","response_id":self.id,"item_id":self.message_id,"output_index":output_index,"content_index":index,"part":part}));
        self.parts.push(part);
        if audio {
            self.audio_index = Some(index);
        } else {
            self.text_index = Some(index);
        }
        index
    }
    pub(super) fn text(&mut self, text: &str) -> anyhow::Result<Vec<Value>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        self.reserve_text(text.len())?;
        let mut events = Vec::new();
        let index = self.part(false, &mut events);
        let Value::String(buffer) = &mut self.parts[index]["text"] else {
            anyhow::bail!("realtime text part lost its text string; repair response event state");
        };
        buffer.push_str(text);
        events.push(json!({"type":"response.output_text.delta","response_id":self.id,"item_id":self.message_id,"output_index":self.message_index,"content_index":index,"delta":text}));
        Ok(events)
    }
    pub(super) fn audio(&mut self, samples: &[f32]) -> anyhow::Result<Vec<Value>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        ensure!(
            samples.len() <= self.max_audio_samples.get(),
            "realtime audio delta contains {} samples, exceeding the {}-sample event budget; reduce the model quantum size or increase the configured audio event budget",
            samples.len(),
            self.max_audio_samples
        );
        ensure!(
            samples.iter().all(|value| value.is_finite()),
            "realtime audio event contains nonfinite PCM; the workflow must emit finite mono samples"
        );
        let mut pcm = Vec::with_capacity(
            samples
                .len()
                .checked_mul(2)
                .context("realtime audio chunk byte length overflow")?,
        );
        for sample in samples {
            // llama-server's realtime PCM conversion uses a symmetric 32767
            // scale (including -1.0), distinct from the WAV full-range encoder.
            let value = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
            pcm.extend_from_slice(&value.to_le_bytes());
        }
        let mut events = Vec::new();
        let index = self.part(true, &mut events);
        events.push(json!({"type":"response.output_audio.delta","response_id":self.id,"item_id":self.message_id,"output_index":self.message_index,"content_index":index,"delta":STANDARD.encode(pcm)}));
        Ok(events)
    }
    /// The shared model protocol parser returns complete calls. Repeated
    /// cumulative snapshots emit no duplicate call items or argument bytes.
    pub(super) fn tool_call(
        &mut self,
        call: &ToolCall,
        item_id: String,
    ) -> anyhow::Result<Vec<Value>> {
        if let Some(index) = self.calls.get(&call.id) {
            let item = &self.output[*index];
            ensure!(
                item["name"] == call.name && item["arguments"] == call.arguments,
                "realtime tool call '{}' changed an already published name or argument object; repair the model tool protocol",
                call.id
            );
            return Ok(Vec::new());
        }
        ensure!(
            self.calls.len() < MAX_TOOL_CALLS,
            "realtime response exceeds the {MAX_TOOL_CALLS}-call tool protocol limit; split the response"
        );
        ensure!(
            !call.id.is_empty() && !call.name.is_empty(),
            "realtime tool call requires a nonempty call ID and name; use the shared model tool protocol parser"
        );
        let arguments: Value = serde_json::from_str(&call.arguments).context("realtime tool call arguments are not valid JSON; admit the complete call through the declared protocol parser before publishing")?;
        ensure!(
            arguments.is_object(),
            "realtime tool call arguments must be a JSON object"
        );
        self.reserve_text(
            call.arguments
                .len()
                .checked_add(call.name.len())
                .and_then(|n| n.checked_add(call.id.len()))
                .context("realtime tool call size overflow")?,
        )?;
        let index = self.output.len();
        let item = json!({"id":item_id,"object":"realtime.item","type":"function_call","status":"in_progress","name":call.name,"call_id":call.id,"arguments":""});
        let mut events = vec![
            json!({"type":"response.output_item.added","response_id":self.id,"output_index":index,"item":item}),
        ];
        events.push(json!({"type":"response.function_call_arguments.delta","response_id":self.id,"item_id":item_id,"output_index":index,"call_id":call.id,"delta":call.arguments}));
        self.calls.insert(call.id.clone(), index);
        self.output.push(item);
        self.output[index]["arguments"] = json!(call.arguments);
        Ok(events)
    }
    pub(super) fn finish(mut self, cancelled: bool) -> Vec<Value> {
        let mut events = Vec::new();
        let status = if cancelled { "cancelled" } else { "completed" };
        if let Some(output_index) = self.message_index {
            for (index, part) in self.parts.iter().enumerate() {
                let mut event = json!({"response_id":self.id,"item_id":self.message_id,"output_index":output_index,"content_index":index});
                if Some(index) == self.text_index {
                    event["type"] = json!("response.output_text.done");
                    event["text"] = part["text"].clone();
                } else {
                    event["type"] = json!("response.output_audio.done");
                }
                events.push(event.clone());
                event
                    .as_object_mut()
                    .expect("constructed event is an object")
                    .remove("text");
                event["type"] = json!("response.content_part.done");
                event["part"] = part.clone();
                events.push(event);
            }
            self.output[output_index]["content"] = json!(self.parts);
        }
        // Output index order is the wire order, independent of call ID sort.
        for (index, item) in self.output.iter().enumerate() {
            if item["type"] == "function_call" {
                events.push(json!({"type":"response.function_call_arguments.done","response_id":self.id,"item_id":item["id"],"output_index":index,"call_id":item["call_id"],"name":item["name"],"arguments":item["arguments"]}));
            }
        }
        for (index, item) in self.output.iter_mut().enumerate() {
            item["status"] = json!(status);
            events.push(json!({"type":"response.output_item.done","response_id":self.id,"output_index":index,"item":item}));
        }
        events.push(json!({"type":"response.done","response":{"id":self.id,"object":"realtime.response","status":status,"output":self.output,"usage":{"output_tokens":self.output_tokens}}}));
        events
    }
}

pub(super) struct TranscriptionEvents {
    item_id: String,
    transcript: String,
    limit: NonZeroUsize,
}
impl TranscriptionEvents {
    pub(super) fn new(item_id: String, limit: NonZeroUsize) -> Self {
        Self {
            item_id,
            transcript: String::new(),
            limit,
        }
    }
    pub(super) fn text(&mut self, text: &str) -> anyhow::Result<Option<Value>> {
        if text.is_empty() {
            return Ok(None);
        }
        ensure!(
            self.transcript.len().saturating_add(text.len()) <= self.limit.get(),
            "realtime transcription exceeds the {}-byte transcript limit; commit shorter audio turns",
            self.limit
        );
        self.transcript.push_str(text);
        Ok(Some(
            json!({"type":"conversation.item.input_audio_transcription.delta","item_id":self.item_id,"content_index":0,"delta":text}),
        ))
    }
    pub(super) fn finish(self) -> Value {
        json!({"type":"conversation.item.input_audio_transcription.completed","item_id":self.item_id,"content_index":0,"transcript":self.transcript})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn response(limit: usize) -> ResponseEvents {
        let (response, created) = ResponseEvents::new(
            "resp_test".into(),
            "item_message".into(),
            json!(["text", "audio"]),
            NonZeroUsize::new(limit).unwrap(),
            NonZeroUsize::new(1024).unwrap(),
        );
        assert_eq!(created["type"], "response.created");
        assert_eq!(created["response"]["output"], json!([]));
        response
    }
    fn kinds(events: &[Value]) -> Vec<&str> {
        events
            .iter()
            .map(|event| event["type"].as_str().unwrap())
            .collect()
    }
    #[test]
    fn mixed_response_parts_follow_first_delta_order_and_finalize_once() {
        let mut response = response(1024);
        let events = response.audio(&[-1.0, 0.0, 1.0]).unwrap();
        assert_eq!(
            kinds(&events),
            [
                "response.output_item.added",
                "response.content_part.added",
                "response.output_audio.delta"
            ]
        );
        assert_eq!(events[2]["content_index"], 0);
        assert_eq!(
            STANDARD
                .decode(events[2]["delta"].as_str().unwrap())
                .unwrap(),
            [1, 128, 0, 0, 255, 127]
        );
        let text = response.text("你好").unwrap();
        assert_eq!(text[0]["content_index"], 1);
        assert_eq!(
            kinds(&text),
            ["response.content_part.added", "response.output_text.delta"]
        );
        assert_eq!(
            kinds(&response.text("世界").unwrap()),
            ["response.output_text.delta"]
        );
        assert!(response.audio(&[]).unwrap().is_empty());
        assert!(response.text("").unwrap().is_empty());
        response.usage(3).unwrap();
        response.usage(2).unwrap();
        let terminal = response.finish(false);
        assert_eq!(
            kinds(&terminal),
            [
                "response.output_audio.done",
                "response.content_part.done",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.done"
            ]
        );
        let result = &terminal.last().unwrap()["response"];
        assert_eq!(result["status"], "completed");
        assert_eq!(result["output"][0]["content"][1]["text"], "你好世界");
        assert_eq!(result["usage"]["output_tokens"], 5);
        assert!(
            result.to_string().find("delta").is_none(),
            "audio chunks are not retained in the response object"
        );
    }
    #[test]
    fn canonical_tool_snapshots_deduplicate_and_preserve_output_order() {
        let mut response = response(4096);
        let protocol = onnx_genai_metadata::ToolProtocol::TaggedJsonV1;
        let mut stream = onnx_genai_metadata::ToolCallStream::default();
        assert!(matches!(
            stream.push(
                protocol,
                "<tool_call>{\"id\":\"z\",\"name\":\"weather\",\"arguments\":"
            ),
            onnx_genai_metadata::ToolParseOutcome::Incomplete
        ));
        let calls = match stream.push(protocol, "{\"city\":\"上海\"}}</tool_call><tool_call>{\"id\":\"a\",\"name\":\"clock\",\"arguments\":{}}</tool_call>") {
            onnx_genai_metadata::ToolParseOutcome::CompleteSoFar(calls) => calls,
            other => panic!("expected admitted calls, got {other:?}"),
        };
        for (index, call) in calls.iter().enumerate() {
            let events = response
                .tool_call(call, format!("item_call_{index}"))
                .unwrap();
            assert_eq!(events[0]["output_index"], index);
            assert_eq!(
                kinds(&events),
                [
                    "response.output_item.added",
                    "response.function_call_arguments.delta"
                ]
            );
            assert!(
                response
                    .tool_call(call, "ignored_duplicate_id".into())
                    .unwrap()
                    .is_empty()
            );
        }
        let text = response.text("done").unwrap();
        assert_eq!(text[0]["output_index"], 2);
        let terminal = response.finish(true);
        let done_calls: Vec<_> = terminal
            .iter()
            .filter(|event| event["type"] == "response.function_call_arguments.done")
            .collect();
        assert_eq!(done_calls[0]["call_id"], "z");
        assert_eq!(done_calls[1]["call_id"], "a");
        assert_eq!(terminal.last().unwrap()["response"]["status"], "cancelled");
        assert_eq!(
            terminal.last().unwrap()["response"]["output"][0]["arguments"],
            calls[0].arguments
        );
    }
    #[test]
    fn pcm16_saturates_finite_model_peaks_using_upstream_symmetric_scale() {
        let mut response = response(1024);
        let events = response.audio(&[-1.5, 1.5]).unwrap();
        let audio = events.last().unwrap();
        assert_eq!(
            STANDARD.decode(audio["delta"].as_str().unwrap()).unwrap(),
            [1, 128, 255, 127]
        );
    }
    #[test]
    fn response_limits_and_bad_pcm_fail_before_publication() {
        let mut response = response(4);
        assert!(response.text("12345").is_err());
        assert!(response.audio(&[f32::NAN]).is_err());
        assert!(
            response
                .audio(&[0.0; 1025])
                .unwrap_err()
                .to_string()
                .contains("event budget")
        );
        let terminal = response.finish(false);
        assert_eq!(kinds(&terminal), ["response.done"]);
        assert_eq!(terminal[0]["response"]["output"], json!([]));
    }
    #[test]
    fn transcription_uses_item_scoped_events_and_a_bounded_final_transcript() {
        let mut transcription =
            TranscriptionEvents::new("item_audio".into(), NonZeroUsize::new(6).unwrap());
        assert!(transcription.text("").unwrap().is_none());
        let event = transcription.text("你好").unwrap().unwrap();
        assert_eq!(
            event["type"],
            "conversation.item.input_audio_transcription.delta"
        );
        assert_eq!(event["item_id"], "item_audio");
        assert!(transcription.text("!").is_err());
        assert_eq!(transcription.finish()["transcript"], "你好");
    }
}
