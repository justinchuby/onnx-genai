use std::convert::Infallible;

use anyhow::Context;
use axum::response::sse::Event;
use serde::Serialize;
use tokio::sync::mpsc;

use crate::types::{ChatLogprobs, ChatMessageToolCall, CompletionLogprobs};

#[derive(Debug, Serialize)]
pub(crate) struct CompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CompletionChunkChoice {
    text: String,
    index: usize,
    finish_reason: Option<&'static str>,
    logprobs: Option<CompletionLogprobs>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChunkChoice {
    index: usize,
    delta: Delta,
    logprobs: Option<ChatLogprobs>,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// The model's private thinking, carried beside the answer rather than
    /// inside it so a client can show its progress without the two being
    /// mistaken for one another.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChunkToolCall {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    function: ChunkToolCallFunction,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChunkToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}
#[derive(Debug)]
pub(crate) struct StopBoundaryBuffer {
    stop_sequences: Vec<String>,
    pub(crate) pending: String,
}

impl StopBoundaryBuffer {
    pub(crate) fn new(stop_sequences: Vec<String>) -> Self {
        Self {
            stop_sequences: stop_sequences
                .into_iter()
                .filter(|sequence| !sequence.is_empty())
                .collect(),
            pending: String::new(),
        }
    }

    pub(crate) fn push(&mut self, text: &str) -> String {
        if self.stop_sequences.is_empty() {
            return text.to_string();
        }

        self.pending.push_str(text);
        if let Some(stop_start) = self.earliest_stop_start() {
            let safe = self.pending[..stop_start].to_string();
            self.pending.clear();
            return safe;
        }

        let keep = self.longest_stop_prefix_suffix_len();
        let emit_len = self.pending.len().saturating_sub(keep);
        if emit_len == 0 {
            return String::new();
        }

        let safe = self.pending[..emit_len].to_string();
        self.pending = self.pending[emit_len..].to_string();
        safe
    }

    pub(crate) fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    fn earliest_stop_start(&self) -> Option<usize> {
        self.stop_sequences
            .iter()
            .filter_map(|sequence| self.pending.find(sequence))
            .min()
    }

    fn longest_stop_prefix_suffix_len(&self) -> usize {
        let mut keep = 0;
        for sequence in &self.stop_sequences {
            for (prefix_len, _) in sequence.char_indices().skip(1) {
                if self.pending.ends_with(&sequence[..prefix_len]) {
                    keep = keep.max(prefix_len);
                }
            }
        }
        keep
    }
}
pub(crate) async fn send_stream_chunk(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    chunk: ChatCompletionChunk,
) -> anyhow::Result<()> {
    tx.send(Ok(Event::default().data(serde_json::to_string(&chunk)?)))
        .await
        .context("stream receiver closed")
}

pub(crate) async fn send_completion_stream_chunk(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    chunk: CompletionChunk,
) -> anyhow::Result<()> {
    tx.send(Ok(Event::default().data(serde_json::to_string(&chunk)?)))
        .await
        .context("stream receiver closed")
}

pub(crate) fn completion_chunk(
    id: &str,
    created: u64,
    model: &str,
    text: String,
    logprobs: Option<CompletionLogprobs>,
) -> CompletionChunk {
    CompletionChunk {
        id: id.to_string(),
        object: "text_completion",
        created,
        model: model.to_string(),
        choices: vec![CompletionChunkChoice {
            text,
            index: 0,
            finish_reason: None,
            logprobs,
        }],
    }
}

pub(crate) fn completion_done_chunk(
    id: &str,
    created: u64,
    model: &str,
    finish_reason: &'static str,
) -> CompletionChunk {
    CompletionChunk {
        id: id.to_string(),
        object: "text_completion",
        created,
        model: model.to_string(),
        choices: vec![CompletionChunkChoice {
            text: String::new(),
            index: 0,
            finish_reason: Some(finish_reason),
            logprobs: None,
        }],
    }
}

pub(crate) fn role_chunk(id: &str, created: u64, model: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant"),
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: None,
        }],
    }
}

pub(crate) fn content_chunk(
    id: &str,
    created: u64,
    model: &str,
    content: String,
    logprobs: Option<ChatLogprobs>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: Some(content),
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs,
            finish_reason: None,
        }],
    }
}

/// A chunk carrying the model's private thinking rather than its answer.
pub(crate) fn reasoning_chunk(
    id: &str,
    created: u64,
    model: &str,
    reasoning: String,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
                reasoning_content: Some(reasoning),
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: None,
        }],
    }
}

pub(crate) fn tool_call_delta_chunks(
    id: &str,
    created: u64,
    model: &str,
    tool_calls: Vec<ChatMessageToolCall>,
) -> Vec<ChatCompletionChunk> {
    tool_calls
        .into_iter()
        .enumerate()
        .flat_map(|(index, call)| {
            [
                tool_call_delta_chunk(
                    id,
                    created,
                    model,
                    ChunkToolCall {
                        index,
                        id: Some(call.id),
                        kind: Some(call.kind),
                        function: ChunkToolCallFunction {
                            name: Some(call.function.name),
                            arguments: Some(String::new()),
                        },
                    },
                ),
                tool_call_delta_chunk(
                    id,
                    created,
                    model,
                    ChunkToolCall {
                        index,
                        id: None,
                        kind: None,
                        function: ChunkToolCallFunction {
                            name: None,
                            arguments: Some(call.function.arguments),
                        },
                    },
                ),
            ]
        })
        .collect()
}

fn tool_call_delta_chunk(
    id: &str,
    created: u64,
    model: &str,
    tool_call: ChunkToolCall,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![tool_call]),
            },
            logprobs: None,
            finish_reason: None,
        }],
    }
}

pub(crate) fn done_chunk(
    id: &str,
    created: u64,
    model: &str,
    finish_reason: &'static str,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta::default(),
            logprobs: None,
            finish_reason: Some(finish_reason),
        }],
    }
}

#[cfg(test)]
mod reasoning_wire_tests {
    use super::{content_chunk, reasoning_chunk};

    // Reasoning rides on its own `reasoning_content` key and never touches
    // `content`, so an OpenAI-compatible client that ignores the extra field
    // sees the same answer stream as before while a reasoning-aware one can show
    // the model working.
    #[test]
    fn a_reasoning_chunk_carries_reasoning_content_and_no_content() {
        let chunk = reasoning_chunk("id", 0, "m", "weighing it".to_string());
        let json = serde_json::to_string(&chunk).expect("serialize");
        assert!(json.contains("\"reasoning_content\":\"weighing it\""), "{json}");
        assert!(!json.contains("\"content\""), "{json}");
    }

    // An ordinary answer chunk omits `reasoning_content` entirely, so its wire
    // shape is byte for byte what it was before the channel existed.
    #[test]
    fn a_content_chunk_omits_reasoning_content() {
        let chunk = content_chunk("id", 0, "m", "Hi".to_string(), None);
        let json = serde_json::to_string(&chunk).expect("serialize");
        assert!(json.contains("\"content\":\"Hi\""), "{json}");
        assert!(!json.contains("reasoning_content"), "{json}");
    }
}
