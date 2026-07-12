use std::convert::Infallible;

use anyhow::Context;
use axum::response::sse::Event;
use serde::Serialize;
use tokio::sync::mpsc;

use crate::types::{ChatMessageToolCall, ChatMessageToolCallFunction};

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
    logprobs: Option<serde_json::Value>,
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
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChunkToolCall {
    index: usize,
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: ChatMessageToolCallFunction,
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
            logprobs: None,
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
                tool_calls: None,
            },
            finish_reason: None,
        }],
    }
}

pub(crate) fn content_chunk(
    id: &str,
    created: u64,
    model: &str,
    content: String,
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
                tool_calls: None,
            },
            finish_reason: None,
        }],
    }
}

pub(crate) fn tool_calls_chunk(
    id: &str,
    created: u64,
    model: &str,
    tool_calls: Vec<ChatMessageToolCall>,
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
                tool_calls: Some(
                    tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(index, call)| ChunkToolCall {
                            index,
                            id: call.id,
                            kind: call.kind,
                            function: call.function,
                        })
                        .collect(),
                ),
            },
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
            finish_reason: Some(finish_reason),
        }],
    }
}
