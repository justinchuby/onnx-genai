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
    /// The *closing* delimiter is located first. A template that opens the span
    /// itself means output begins inside the reasoning with no opener to find,
    /// and an answer may well mention the opening tag again — anchoring on the
    /// first opener would then swallow the answer as reasoning.
    ///
    /// `opened_by_template` says whether the template already wrote the opener,
    /// which is what makes unclosed output reasoning rather than an answer.
    pub fn split<'a>(&self, text: &'a str, opened_by_template: bool) -> ReasoningSplit<'a> {
        if let Some(end_index) = text.find(&self.end) {
            let head = &text[..end_index];
            // Drop the opener only when it actually begins the span.
            let reasoning = match head.find(&self.start) {
                Some(index) => &head[index + self.start.len()..],
                None => head,
            };
            return ReasoningSplit {
                reasoning: Some(reasoning.trim()),
                answer: text[end_index + self.end.len()..].trim_start(),
                complete: true,
            };
        }
        match text.find(&self.start) {
            // Opened but never closed: generation stopped mid-thought.
            Some(index) => ReasoningSplit {
                reasoning: Some(text[index + self.start.len()..].trim()),
                answer: "",
                complete: false,
            },
            // No opener either. When the template opened the span, the model
            // began inside it, so unclosed output is all reasoning; otherwise
            // this model simply did not think.
            None if opened_by_template => ReasoningSplit {
                reasoning: Some(text.trim()),
                answer: "",
                complete: false,
            },
            None => ReasoningSplit {
                reasoning: None,
                answer: text,
                complete: true,
            },
        }
    }
}

/// Generated text separated into reasoning and answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningSplit<'a> {
    /// The chain of thought, if the model produced one.
    pub reasoning: Option<&'a str>,
    /// The answer, which is the only part a later turn should see.
    pub answer: &'a str,
    /// False when generation stopped inside the reasoning span, so there is no
    /// answer yet — the decode budget ran out mid-thought.
    pub complete: bool,
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
    /// Recent output, bounded by the longest delimiter, so a delimiter split
    /// across token boundaries is still recognized.
    tail: String,
}

impl ReasoningStream {
    /// Start a stream. `opened_by_template` is true when the template opens the
    /// reasoning span itself, so output begins inside it.
    pub fn new(markers: ReasoningMarkers, opened_by_template: bool) -> Self {
        Self {
            markers,
            inside: opened_by_template,
            tail: String::new(),
        }
    }

    /// Whether the stream is currently inside a reasoning span.
    pub fn inside(&self) -> bool {
        self.inside
    }

    /// Feed the next chunk, returning whether it is reasoning.
    ///
    /// The state is updated after classifying, so the delimiter itself is
    /// attributed to the span it closes or opens.
    pub fn push(&mut self, chunk: &str) -> bool {
        let was_inside = self.inside;
        self.tail.push_str(chunk);
        loop {
            let marker = if self.inside {
                &self.markers.end
            } else {
                &self.markers.start
            };
            match self.tail.find(marker.as_str()) {
                Some(index) => {
                    let consumed = index + marker.len();
                    self.tail.drain(..consumed);
                    self.inside = !self.inside;
                }
                None => break,
            }
        }
        let longest = self.markers.start.len().max(self.markers.end.len());
        if self.tail.len() > longest {
            let drop_to = self.tail.len() - longest;
            // Never split a UTF-8 character when trimming the retained tail.
            let boundary = (0..=drop_to)
                .rev()
                .find(|index| self.tail.is_char_boundary(*index))
                .unwrap_or(0);
            self.tail.drain(..boundary);
        }
        was_inside || self.inside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn think() -> ReasoningMarkers {
        ReasoningMarkers::new("<think>", "</think>")
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

        assert_eq!(split.reasoning, Some("step one\nstep two"));
        assert_eq!(split.answer, "The answer is 4.");
        assert!(split.complete);
    }

    #[test]
    fn a_template_opened_span_has_no_start_tag_in_the_output() {
        // Many templates append the opening tag after the generation prompt, so
        // the model only ever emits the close.
        let split = think().split("weighing the options</think>Yes.", true);

        assert_eq!(split.reasoning, Some("weighing the options"));
        assert_eq!(split.answer, "Yes.");
        assert!(split.complete);
    }

    #[test]
    fn an_opening_tag_after_the_close_belongs_to_the_answer() {
        // Seen in practice: a model closes its reasoning and then quotes the
        // opening tag again. Anchoring on the first opener would swallow the
        // whole answer as reasoning and leave history empty.
        let split = think().split("weighing it</think>Answer. See <think> for details.", true);

        assert_eq!(split.reasoning, Some("weighing it"));
        assert_eq!(split.answer, "Answer. See <think> for details.");
        assert!(split.complete);
    }

    #[test]
    fn text_without_reasoning_is_left_alone() {
        let split = think().split("Just an answer.", false);

        assert_eq!(split.reasoning, None);
        assert_eq!(split.answer, "Just an answer.");
        assert!(split.complete);
    }

    #[test]
    fn generation_stopped_mid_thought_reports_no_answer() {
        let split = think().split("<think>still working on it", false);

        assert_eq!(split.reasoning, Some("still working on it"));
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

        assert_eq!(split.reasoning, Some("counting on my fingers"));
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
