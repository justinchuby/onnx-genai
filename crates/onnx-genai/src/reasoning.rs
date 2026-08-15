// Copyright (c) Microsoft Corporation.
//
//! Reasoning ("thinking") spans in generated text.
//!
//! Reasoning models emit a chain of thought bounded by delimiters before the
//! answer. Two things follow, and both matter to a multi-turn front end:
//!
//! 1. The span should be presented as reasoning, not as part of the answer.
//! 2. It must **not** be fed back into later turns. These models are trained
//!    with prior turns' thinking removed, so replaying it degrades quality and
//!    inflates the context — the reasoning of a long session can dwarf the
//!    conversation itself.
//!
//! The delimiters are not guessed from a model or vendor name (`RULES.md` §2).
//! They come from the package's own declared chat template: a template that
//! writes `<think>` is telling the runtime that is how this model marks
//! reasoning. Callers may also state them explicitly.

/// Delimiters bounding a model's reasoning span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningMarkers {
    pub start: String,
    pub end: String,
}

/// Delimiter pairs seen in published chat templates.
///
/// This is a table of *template syntax*, matched against the template a package
/// ships — never against a model name. A package whose template uses none of
/// these simply has no reasoning span as far as this runtime is concerned.
const TEMPLATE_MARKERS: &[(&str, &str)] = &[
    ("<think>", "</think>"),
    ("<|think|>", "<|/think|>"),
    ("<reasoning>", "</reasoning>"),
    ("<|begin_of_thought|>", "<|end_of_thought|>"),
];

impl ReasoningMarkers {
    /// Use explicitly stated delimiters.
    pub fn new(start: impl Into<String>, end: impl Into<String>) -> Self {
        Self {
            start: start.into(),
            end: end.into(),
        }
    }

    /// Detect the delimiters a chat template declares, if any.
    ///
    /// Either delimiter is enough. Templates differ in which they write: some
    /// render a prior turn's `</think>`, while others only append the opening
    /// `<think>` after the generation prompt and leave the close to the model.
    /// A template mentioning either is declaring the convention, and these
    /// strings are distinctive enough that a mention is not a coincidence.
    pub fn from_chat_template(template: &str) -> Option<Self> {
        TEMPLATE_MARKERS
            .iter()
            .find(|(start, end)| template.contains(start) || template.contains(end))
            .map(|(start, end)| Self::new(*start, *end))
    }

    /// Split generated text into its reasoning span and the answer.
    ///
    /// Uses the same marker state machine as streaming classification: at most
    /// one reasoning span is recognized, and text after the closing delimiter is
    /// answer even if it mentions the opening delimiter again. A template that
    /// opens the span itself means output begins inside reasoning with no opener
    /// to find.
    ///
    /// `opened_by_template` says whether the template already wrote the opener,
    /// which is what makes unclosed output reasoning rather than an answer.
    pub fn split(&self, text: &str, opened_by_template: bool) -> ReasoningSplit {
        let mut stream = ReasoningStream::new(self.clone(), opened_by_template);
        let mut events = stream.push_events(text);
        events.extend(stream.finish_events());

        let mut saw_reasoning = opened_by_template;
        let mut complete = !opened_by_template;
        let mut reasoning = String::new();
        let mut answer = String::new();
        for event in events {
            match event.kind {
                ReasoningEventKind::Text { is_reasoning } if is_reasoning => {
                    saw_reasoning = true;
                    reasoning.push_str(&event.text);
                }
                ReasoningEventKind::Text { .. } if saw_reasoning && complete => {
                    answer.push_str(&event.text);
                }
                ReasoningEventKind::Text { .. } if !saw_reasoning => {
                    answer.push_str(&event.text);
                }
                ReasoningEventKind::Text { .. } => {}
                ReasoningEventKind::StartMarker => {
                    saw_reasoning = true;
                    complete = false;
                    answer.clear();
                }
                ReasoningEventKind::EndMarker => {
                    if saw_reasoning {
                        complete = true;
                    }
                }
            }
        }

        if saw_reasoning {
            if !complete {
                answer.clear();
            }
            ReasoningSplit {
                reasoning: Some(reasoning.trim().to_string()),
                answer: answer.trim_start().to_string(),
                complete,
            }
        } else {
            ReasoningSplit {
                reasoning: None,
                answer,
                complete: true,
            }
        }
    }
}

/// Generated text separated into reasoning and answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningSplit {
    /// The chain of thought, if the model produced one.
    pub reasoning: Option<String>,
    /// The answer, which is the only part a later turn should see.
    pub answer: String,
    /// False when generation stopped inside the reasoning span, so there is no
    /// answer yet — the decode budget ran out mid-thought.
    pub complete: bool,
}

/// One piece of a streamed chunk with its reasoning classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningChunk {
    pub text: String,
    pub is_reasoning: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningEventKind {
    Text { is_reasoning: bool },
    StartMarker,
    EndMarker,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReasoningEvent {
    text: String,
    kind: ReasoningEventKind,
}

/// Tracks whether a token stream is currently inside a reasoning span.
///
/// Streaming cannot buffer to the end of a turn without destroying the point of
/// streaming, so the boundary is detected incrementally. Delimiters can be split
/// across tokens, so a small tail of recent text is retained.
#[derive(Debug)]
pub struct ReasoningStream {
    markers: ReasoningMarkers,
    inside: bool,
    closed: bool,
    /// Recent uncommitted output. A suffix that could still become the next
    /// delimiter is held back until the following chunk proves or completes it.
    pending: String,
}

impl ReasoningStream {
    /// Start a stream. `opened_by_template` is true when the template opens the
    /// reasoning span itself, so output begins inside it.
    pub fn new(markers: ReasoningMarkers, opened_by_template: bool) -> Self {
        Self {
            markers,
            inside: opened_by_template,
            closed: false,
            pending: String::new(),
        }
    }

    /// Whether the stream is currently inside a reasoning span.
    pub fn inside(&self) -> bool {
        self.inside
    }

    /// Feed the next chunk, returning the finalized reasoning classification
    /// for each subspan.
    ///
    /// A tokenizer may return `</think>` and the first answer word in one token.
    /// A single boolean for the whole token would dim that answer as reasoning,
    /// so this method splits at delimiter boundaries. A suffix that could be a
    /// partial delimiter is held until the next chunk; call [`finish`](Self::finish)
    /// when the stream ends to flush any remaining ordinary text.
    pub fn push_segments(&mut self, chunk: &str) -> Vec<ReasoningChunk> {
        self.push_events(chunk)
            .into_iter()
            .map(ReasoningEvent::into_chunk)
            .collect()
    }

    /// Flush text that was held because it looked like a partial delimiter.
    pub fn finish(&mut self) -> Vec<ReasoningChunk> {
        self.finish_events()
            .into_iter()
            .map(ReasoningEvent::into_chunk)
            .collect()
    }

    /// Feed the next chunk, returning whether it is reasoning.
    ///
    /// The state is updated after classifying, so the delimiter itself is
    /// attributed to the span it closes or opens.
    pub fn push(&mut self, chunk: &str) -> bool {
        let was_inside = self.inside;
        let segments = self.push_segments(chunk);
        segments.iter().any(|segment| segment.is_reasoning) || (segments.is_empty() && was_inside)
    }

    fn push_events(&mut self, chunk: &str) -> Vec<ReasoningEvent> {
        self.pending.push_str(chunk);
        self.drain_events(false)
    }

    fn finish_events(&mut self) -> Vec<ReasoningEvent> {
        self.drain_events(true)
    }

    fn drain_events(&mut self, finish: bool) -> Vec<ReasoningEvent> {
        let mut events = Vec::new();
        loop {
            if self.closed {
                let emit_len = self.pending.len();
                self.emit_text_prefix(emit_len, &mut events);
                break;
            }
            let (marker, marker_kind) = if self.inside {
                (self.markers.end.clone(), ReasoningEventKind::EndMarker)
            } else {
                (self.markers.start.clone(), ReasoningEventKind::StartMarker)
            };
            if let Some(index) = self.pending.find(marker.as_str()) {
                let marker_text = self.pending[index..index + marker.len()].to_string();
                self.emit_text_prefix(index, &mut events);
                self.pending.drain(..marker.len());
                events.push(ReasoningEvent {
                    text: marker_text,
                    kind: marker_kind,
                });
                self.inside = !self.inside;
                if marker_kind == ReasoningEventKind::EndMarker {
                    self.closed = true;
                }
                continue;
            }

            let keep = if finish {
                0
            } else {
                marker_prefix_suffix_len(&self.pending, &marker)
            };
            let emit_len = self.pending.len().saturating_sub(keep);
            self.emit_text_prefix(emit_len, &mut events);
            break;
        }
        events
    }

    fn emit_text_prefix(&mut self, bytes: usize, events: &mut Vec<ReasoningEvent>) {
        if bytes == 0 {
            return;
        }
        let text = self.pending[..bytes].to_string();
        self.pending.drain(..bytes);
        events.push(ReasoningEvent {
            text,
            kind: ReasoningEventKind::Text {
                is_reasoning: self.inside,
            },
        });
    }
}

impl ReasoningEvent {
    fn into_chunk(self) -> ReasoningChunk {
        ReasoningChunk {
            text: self.text,
            is_reasoning: match self.kind {
                ReasoningEventKind::Text { is_reasoning } => is_reasoning,
                ReasoningEventKind::StartMarker | ReasoningEventKind::EndMarker => true,
            },
        }
    }
}

fn marker_prefix_suffix_len(text: &str, marker: &str) -> usize {
    let max = text.len().min(marker.len().saturating_sub(1));
    for len in (1..=max).rev() {
        let start = text.len() - len;
        if text.is_char_boundary(start)
            && marker.is_char_boundary(len)
            && marker.starts_with(&text[start..])
        {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn think() -> ReasoningMarkers {
        ReasoningMarkers::new("<think>", "</think>")
    }

    fn chunk(text: &str, is_reasoning: bool) -> ReasoningChunk {
        ReasoningChunk {
            text: text.to_string(),
            is_reasoning,
        }
    }

    #[test]
    fn a_template_declares_its_own_reasoning_delimiters() {
        // A template that only opens the span still declares the convention:
        // it leaves the close to the model.
        assert_eq!(
            ReasoningMarkers::from_chat_template("{% for m in messages %}...{% endfor %}<think>\n"),
            Some(think())
        );
        // As does one that only renders the close of a prior turn.
        assert_eq!(
            ReasoningMarkers::from_chat_template("...</think>{{ content }}"),
            Some(think())
        );

        // A template with no reasoning syntax declares no span.
        assert_eq!(
            ReasoningMarkers::from_chat_template("<|user|>{{ content }}<|assistant|>"),
            None
        );
    }

    #[test]
    fn reasoning_is_separated_from_the_answer() {
        let split = think().split("<think>step one\nstep two</think>The answer is 4.", false);

        assert_eq!(split.reasoning.as_deref(), Some("step one\nstep two"));
        assert_eq!(split.answer, "The answer is 4.");
        assert!(split.complete);
    }

    #[test]
    fn a_template_opened_span_has_no_start_tag_in_the_output() {
        // Many templates append the opening tag after the generation prompt, so
        // the model only ever emits the close.
        let split = think().split("weighing the options</think>Yes.", true);

        assert_eq!(split.reasoning.as_deref(), Some("weighing the options"));
        assert_eq!(split.answer, "Yes.");
        assert!(split.complete);
    }

    #[test]
    fn an_opening_tag_after_the_close_belongs_to_the_answer() {
        // Seen in practice: a model closes its reasoning and then quotes the
        // opening tag again. Anchoring on the first opener would swallow the
        // whole answer as reasoning and leave history empty.
        let split = think().split("weighing it</think>Answer. See <think> for details.", true);

        assert_eq!(split.reasoning.as_deref(), Some("weighing it"));
        assert_eq!(split.answer, "Answer. See <think> for details.");
        assert!(split.complete);
    }

    #[test]
    fn text_without_reasoning_is_left_alone() {
        let split = think().split("Just an answer.", false);

        assert_eq!(split.reasoning.as_deref(), None);
        assert_eq!(split.answer, "Just an answer.");
        assert!(split.complete);
    }

    #[test]
    fn generation_stopped_mid_thought_reports_no_answer() {
        let split = think().split("<think>still working on it", false);

        assert_eq!(split.reasoning.as_deref(), Some("still working on it"));
        assert_eq!(split.answer, "");
        assert!(
            !split.complete,
            "an unterminated span must be visible to the caller"
        );
    }

    #[test]
    fn unclosed_output_under_a_template_opened_span_is_all_reasoning() {
        // The template wrote `<think>`, so the model began inside the span. A
        // budget that ran out before the close means there is no answer — and
        // storing this text as the answer would poison the next turn.
        let split = think().split("counting on my fingers", true);

        assert_eq!(split.reasoning.as_deref(), Some("counting on my fingers"));
        assert_eq!(split.answer, "");
        assert!(!split.complete);
    }

    #[test]
    fn a_stream_classifies_chunks_as_they_arrive() {
        let mut stream = ReasoningStream::new(think(), false);

        assert!(stream.push("<think>"));
        assert!(stream.push("thinking hard"));
        assert!(stream.inside());
        assert!(
            stream.push("</think>"),
            "the closing tag is still reasoning"
        );
        assert!(!stream.inside());
        assert!(!stream.push("The answer."));
    }

    #[test]
    fn a_stream_splits_an_answer_that_shares_the_closing_token() {
        let mut stream = ReasoningStream::new(think(), false);

        let segments = stream.push_segments("<think>because</think>Olympia");

        assert_eq!(
            segments,
            vec![
                chunk("<think>", true),
                chunk("because", true),
                chunk("</think>", true),
                chunk("Olympia", false),
            ]
        );
        assert!(!stream.inside());
    }

    #[test]
    fn a_split_opening_delimiter_is_held_until_complete() {
        let mut stream = ReasoningStream::new(think(), false);

        assert_eq!(stream.push_segments("<"), Vec::<ReasoningChunk>::new());
        assert_eq!(
            stream.push_segments("think>because"),
            vec![chunk("<think>", true), chunk("because", true)]
        );
        assert!(stream.inside());
    }

    #[test]
    fn a_split_closing_delimiter_leaves_same_token_answer_outside_reasoning() {
        let mut stream = ReasoningStream::new(think(), true);

        assert_eq!(stream.push_segments("</"), Vec::<ReasoningChunk>::new());
        assert_eq!(stream.push_segments("think"), Vec::<ReasoningChunk>::new());
        assert_eq!(
            stream.push_segments(">Olympia"),
            vec![chunk("</think>", true), chunk("Olympia", false)]
        );
        assert!(!stream.inside());
    }

    #[test]
    fn streaming_and_final_split_agree_on_reasoning_answer_boundary() {
        let markers = think();
        let text = "<think>because</think>Olympia";
        let mut stream = ReasoningStream::new(markers.clone(), false);
        let mut chunks = Vec::new();
        for piece in ["<", "think>because</", "think", ">Olympia"] {
            chunks.extend(stream.push_segments(piece));
        }
        chunks.extend(stream.finish());
        let rendered = chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<String>();
        let answer = chunks
            .iter()
            .filter(|chunk| !chunk.is_reasoning)
            .map(|chunk| chunk.text.as_str())
            .collect::<String>();
        let split = markers.split(text, false);

        assert_eq!(rendered, text);
        assert_eq!(answer, split.answer);
        assert_eq!(split.answer, "Olympia");
        assert!(split.complete);
    }

    #[test]
    fn streaming_classification_is_lossless_and_dims_markers_too() {
        let text = "<think>because</think>Olympia";
        let mut stream = ReasoningStream::new(think(), false);
        let chunks = stream.push_segments(text);

        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<String>(),
            text,
            "rendering may add color, but it must not swallow marker text"
        );
        assert_eq!(
            chunks,
            vec![
                chunk("<think>", true),
                chunk("because", true),
                chunk("</think>", true),
                chunk("Olympia", false),
            ]
        );
    }

    #[test]
    fn emoji_adjacent_to_marker_is_preserved_byte_for_byte() {
        let text = "🙂<think>思考🙂</think>✅";
        let mut stream = ReasoningStream::new(think(), false);
        let mut chunks = Vec::new();
        for piece in ["🙂<", "think>思", "考🙂</", "think>✅"] {
            chunks.extend(stream.push_segments(piece));
        }
        chunks.extend(stream.finish());

        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<String>()
                .as_bytes(),
            text.as_bytes(),
            "UTF-8 content around markers must be emitted unchanged"
        );
        assert_eq!(
            chunks
                .iter()
                .filter(|chunk| chunk.is_reasoning)
                .map(|chunk| chunk.text.as_str())
                .collect::<String>(),
            "<think>思考🙂</think>"
        );
    }

    #[test]
    fn a_delimiter_split_across_tokens_is_still_recognized() {
        let mut stream = ReasoningStream::new(think(), true);
        assert!(stream.inside());

        // Tokenizers routinely split `</think>` into several pieces.
        for piece in ["consider", "ing", "</", "think", ">"] {
            stream.push(piece);
        }

        assert!(
            !stream.inside(),
            "a delimiter arriving in pieces must still close the span"
        );
        assert!(!stream.push("Done."));
    }

    #[test]
    fn multibyte_output_does_not_split_a_character() {
        let mut stream = ReasoningStream::new(think(), true);

        // Long multi-byte content forces the retained tail to be trimmed.
        for _ in 0..50 {
            stream.push("思考中的内容");
        }
        stream.push("</think>");

        assert!(!stream.inside());
    }
}
