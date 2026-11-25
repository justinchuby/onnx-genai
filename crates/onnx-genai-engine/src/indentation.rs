//! Stateful indentation boundaries for the pinned llama.cpp output contract.
//!
//! The first observed newline arms the check for a later token observation.
//! Spaces and tabs each count as one character. This intentionally tracks token
//! observations, rather than interpreting a completed string as a fresh stream.

/// State shared by generation stopping and transport text projection.
#[derive(Debug, Default, Clone)]
pub struct IndentationTracker {
    minimum: usize,
    last_newline: usize,
    has_newline: bool,
    cutoff: Option<usize>,
}

impl IndentationTracker {
    /// Construct request-owned state with immutable indentation policy.
    pub fn new(minimum: usize) -> Self {
        Self {
            minimum,
            ..Self::default()
        }
    }

    /// Observe the current stop-clipped generated text and newly visible output.
    ///
    /// Returns the byte boundary before insufficiently indented content. Leading
    /// whitespace on that line remains visible. Once stopped, the boundary is
    /// stable. Call exactly once per selected token, including empty fragments;
    /// a new request must use a new tracker.
    pub fn observe(&mut self, generated: &str, newly_visible: &str) -> Option<usize> {
        if self.minimum == 0 || self.cutoff.is_some() {
            return self.cutoff;
        }
        if self.has_newline {
            if self.last_newline > 0 {
                let mut pos = self.last_newline;
                while generated
                    .as_bytes()
                    .get(pos)
                    .is_some_and(|byte| matches!(byte, b' ' | b'\t'))
                {
                    pos += 1;
                }
                if pos < generated.len() && pos - self.last_newline < self.minimum {
                    self.cutoff = Some(pos);
                    return self.cutoff;
                }
            }
            if let Some(pos) = generated
                .get(self.last_newline..)
                .and_then(|suffix| suffix.find('\n'))
            {
                self.last_newline += pos + 1;
            }
        }
        if newly_visible.contains('\n') {
            self.has_newline = true;
        }
        self.cutoff
    }
}

/// Request-local projection of stop-clipped token text into indentation checks.
#[derive(Clone)]
pub(crate) struct IndentationProjection {
    tracker: IndentationTracker,
    observed_bytes: usize,
}
impl IndentationProjection {
    pub(crate) fn new(minimum: usize) -> Self {
        Self {
            tracker: IndentationTracker::new(minimum),
            observed_bytes: 0,
        }
    }
    pub(crate) fn observe(
        &mut self,
        text: &str,
        stops: &[crate::logits::StopSequence],
    ) -> Option<usize> {
        let boundary = crate::logits::text_stop_boundary(
            text,
            stops
                .iter()
                .enumerate()
                .filter_map(|(index, stop)| match stop {
                    crate::logits::StopSequence::Text(stop) => Some((index, stop.as_str())),
                    _ => None,
                }),
        );
        let visible = &text[..boundary.visible_bytes];
        // Full stop words are removed from generated text. A held partial stop
        // remains in generated text but does not arm visible-newline checks.
        let generated = if boundary.matched_sequence.is_some() {
            visible
        } else {
            text
        };
        let start = self.observed_bytes.min(visible.len());
        let cutoff = self.tracker.observe(generated, &visible[start..]);
        self.observed_bytes = visible.len();
        cutoff
    }
}

/// Output-only observer; it neither transforms logits nor changes acceptance.
pub(crate) struct IndentationLimit {
    state: std::sync::Mutex<IndentationObservation>,
    stops: Vec<crate::logits::StopSequence>,
}
struct IndentationObservation {
    projection: IndentationProjection,
    baseline: IndentationProjection,
    observed_tokens: usize,
    observed_raw_bytes: usize,
    signal: Option<crate::logits::ProcessorSignal>,
}
impl IndentationLimit {
    pub(crate) fn new(minimum: usize, stops: Vec<crate::logits::StopSequence>) -> Self {
        let projection = IndentationProjection::new(minimum);
        Self {
            state: std::sync::Mutex::new(IndentationObservation {
                baseline: projection.clone(),
                projection,
                observed_tokens: 0,
                observed_raw_bytes: 0,
                signal: None,
            }),
            stops,
        }
    }
}
impl crate::logits::LogitProcessor for IndentationLimit {
    fn process(&self, _: &mut [f32], _: &crate::logits::ProcessorContext) -> anyhow::Result<()> {
        Ok(())
    }
    fn only_observes_committed_output(&self) -> bool {
        true
    }
    fn preserves_argmax(&self) -> bool {
        true
    }
    fn signal(
        &self,
        context: &crate::logits::ProcessorContext,
    ) -> Option<crate::logits::ProcessorSignal> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if context.generated_tokens.is_empty() {
            return None;
        }
        if context.generated_tokens.len() == state.observed_tokens {
            if context.generated_text.len() == state.observed_raw_bytes {
                return state.signal.clone();
            }
            // EOF may complete this token's UTF-8. Replay that observation from
            // its baseline rather than advancing the newline scan twice.
            state.projection = state.baseline.clone();
        } else {
            state.baseline = state.projection.clone();
        }
        state.signal = state
            .projection
            .observe(&context.generated_text, &self.stops)
            .map(
                |visible_bytes| crate::logits::ProcessorSignal::IndentationLimit { visible_bytes },
            );
        state.observed_tokens = context.generated_tokens.len();
        state.observed_raw_bytes = context.generated_text.len();
        state.signal.clone()
    }
    fn name(&self) -> &str {
        "indentation_limit"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_newline_arms_a_later_observation_and_preserves_leading_whitespace() {
        let mut state = IndentationTracker::new(2);
        assert_eq!(state.observe("head\n", "head\n"), None);
        // The pinned implementation first finds the newline on this observation.
        assert_eq!(state.observe("head\n ", " "), None);
        assert_eq!(state.observe("head\n x", "x"), Some(6));
        assert_eq!(state.observe("head\n xyz", "yz"), Some(6));
    }

    #[test]
    fn tabs_count_once_and_blank_or_first_lines_follow_the_pinned_scan_order() {
        let mut state = IndentationTracker::new(2);
        assert_eq!(state.observe("first", "first"), None);
        assert_eq!(state.observe("first\n", "\n"), None);
        assert_eq!(state.observe("first\n\t", "\t"), None);
        assert_eq!(state.observe("first\n\t body\n", " body\n"), None);
        assert_eq!(state.observe("first\n\t body\n\n", "\n"), Some(13));
    }

    #[test]
    fn fragments_with_multiple_newlines_are_not_reinterpreted_as_multiple_tokens() {
        let mut state = IndentationTracker::new(1);
        assert_eq!(state.observe("a\nb\nc", "a\nb\nc"), None);
        assert_eq!(state.observe("a\nb\nc", ""), None);
        assert_eq!(state.observe("a\nb\nc", ""), Some(2));
    }

    #[test]
    fn hidden_newlines_do_not_arm_and_unicode_cutoffs_are_byte_boundaries() {
        let mut state = IndentationTracker::new(1);
        assert_eq!(state.observe("α", ""), None);
        assert_eq!(state.observe("α\n", ""), None);
        assert_eq!(state.observe("α\nβ", ""), None);
        assert_eq!(state.observe("α\nβ", "\nβ"), None);
        assert_eq!(state.observe("α\nβ", ""), None);
        assert_eq!(state.observe("α\nβ", ""), Some(3));
        assert!("α\nβ".is_char_boundary(3));
    }

    #[test]
    fn disabled_control_is_inert() {
        let mut state = IndentationTracker::new(0);
        for text in ["x\n", "x\ny", "x\ny\nz"] {
            assert_eq!(state.observe(text, text), None);
        }
    }
}

#[cfg(test)]
mod processor_tests {
    use super::*;
    use crate::logits::{LogitProcessor, ProcessorContext, ProcessorSignal, StopSequence};
    fn context(text: &str, count: usize) -> ProcessorContext {
        ProcessorContext {
            generated_text: text.into(),
            generated_tokens: vec![0; count],
            ..Default::default()
        }
    }
    #[test]
    fn repeated_observation_and_eof_completion_do_not_advance_scan_twice() {
        let limit = IndentationLimit::new(2, vec![]);
        assert_eq!(limit.signal(&context("head\n", 1)), None);
        assert_eq!(limit.signal(&context("head\n ", 2)), None);
        assert_eq!(limit.signal(&context("head\n ", 2)), None);
        assert_eq!(limit.signal(&context("head\n bad", 2)), None);
        assert_eq!(
            limit.signal(&context("head\n bad", 3)),
            Some(ProcessorSignal::IndentationLimit { visible_bytes: 6 })
        );
    }
    #[test]
    fn full_stop_is_removed_but_a_held_prefix_remains_in_generated_text() {
        let stops = vec![StopSequence::Text("END\n".into())];
        let mut projection = IndentationProjection::new(2);
        assert_eq!(projection.observe("head\n", &stops), None);
        assert_eq!(projection.observe("head\n ", &stops), None);
        assert_eq!(projection.observe("head\n END", &stops), Some(6));
        let mut projection = IndentationProjection::new(2);
        assert_eq!(projection.observe("head\n", &stops), None);
        assert_eq!(projection.observe("head\n ", &stops), None);
        assert_eq!(projection.observe("head\n END\n", &stops), None);
    }
    #[test]
    fn eos_label_overrides_indentation_without_skipping_its_observation() {
        let options = crate::GenerateOptions {
            minimum_indent: 2,
            eos_token_id: Some(0),
            stop_on_eos: true,
            ..Default::default()
        };
        let mut chain = crate::logits::ProcessorChain::new();
        crate::processors::append_output_observers(&mut chain, &options);
        for (text, count) in [("head\n", 1), ("head\n ", 2), ("head\n bad", 3)] {
            assert_eq!(
                crate::processors::finish_reason_after_token(
                    0,
                    &options,
                    &chain,
                    &context(text, count)
                ),
                Some(crate::FinishReason::EosToken)
            );
        }
        assert_eq!(
            chain.signal(&context("head\n bad", 3)),
            Some(ProcessorSignal::IndentationLimit { visible_bytes: 6 })
        );
    }
}
