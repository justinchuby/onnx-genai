//! Request-owned normal prediction stop; never a cancellation authority.
use crate::logits::{LogitProcessor, ProcessorContext, ProcessorSignal, StopSequence};
use std::{
    sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

pub(crate) struct PredictionTimeLimit {
    limit: Duration,
    started: OnceLock<Instant>,
    observed_bytes: AtomicUsize,
    stops: Vec<StopSequence>,
}
impl PredictionTimeLimit {
    pub(crate) fn new(limit: Duration, stops: Vec<StopSequence>) -> Self {
        Self {
            limit,
            started: OnceLock::new(),
            observed_bytes: AtomicUsize::new(0),
            stops,
        }
    }
    fn signal_at(&self, context: &ProcessorContext, now: Instant) -> Option<ProcessorSignal> {
        if context.generated_tokens.is_empty() {
            return None;
        }
        let started = *self.started.get_or_init(|| now);
        let text = &context.generated_text;
        // The native protocol checks the text fragment after removing a matched
        // stop word. A newline that belongs only to the stop word is not emitted.
        let end = crate::logits::text_stop_boundary(
            text,
            self.stops
                .iter()
                .enumerate()
                .filter_map(|(index, stop)| match stop {
                    StopSequence::Text(stop) => Some((index, stop.as_str())),
                    _ => None,
                }),
        )
        .visible_bytes;
        let previous = self.observed_bytes.swap(end, Ordering::Relaxed);
        if now.saturating_duration_since(started) > self.limit
            && text
                .as_bytes()
                .get(previous..end)
                .is_some_and(|new| new.contains(&b'\n'))
        {
            Some(ProcessorSignal::PredictionTimeLimit)
        } else {
            None
        }
    }
}
impl LogitProcessor for PredictionTimeLimit {
    fn process(&self, _: &mut [f32], _: &ProcessorContext) -> anyhow::Result<()> {
        Ok(())
    }
    fn only_observes_committed_output(&self) -> bool {
        true
    }
    fn preserves_argmax(&self) -> bool {
        true
    }
    fn signal(&self, context: &ProcessorContext) -> Option<ProcessorSignal> {
        self.signal_at(context, Instant::now())
    }
    fn name(&self) -> &str {
        "prediction_time_limit"
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn context(text: &str, tokens: usize) -> ProcessorContext {
        ProcessorContext {
            generated_tokens: vec![1; tokens],
            generated_text: text.into(),
            ..Default::default()
        }
    }
    #[test]
    fn starts_at_first_token_and_stops_only_on_a_fresh_newline_after_expiry() {
        let limit = PredictionTimeLimit::new(Duration::from_millis(10), vec![]);
        let start = Instant::now();
        assert_eq!(limit.signal_at(&context("", 0), start), None);
        let start = start + Duration::from_secs(60);
        assert_eq!(limit.signal_at(&context("first\n", 1), start), None);
        assert_eq!(
            limit.signal_at(&context("first\n", 1), start + Duration::from_millis(20)),
            None
        );
        assert_eq!(
            limit.signal_at(
                &context("first\nnext", 2),
                start + Duration::from_millis(30)
            ),
            None
        );
        assert_eq!(
            limit.signal_at(
                &context("first\nnext\n", 3),
                start + Duration::from_millis(31)
            ),
            Some(ProcessorSignal::PredictionTimeLimit)
        );
    }
    #[test]
    fn held_stop_prefix_newlines_wait_until_visible_and_hidden_trailing_bytes_never_trigger() {
        let start = Instant::now();
        let limit = PredictionTimeLimit::new(
            Duration::from_millis(10),
            vec![StopSequence::Text("END\nTAIL".into())],
        );
        assert_eq!(limit.signal_at(&context("head", 1), start), None);
        assert_eq!(
            limit.signal_at(&context("headEND\n", 2), start + Duration::from_millis(20)),
            None
        );
        assert_eq!(
            limit.signal_at(&context("headEND\nX", 3), start + Duration::from_millis(30)),
            Some(ProcessorSignal::PredictionTimeLimit)
        );
        let limit = PredictionTimeLimit::new(
            Duration::from_millis(10),
            vec![StopSequence::Text("END\nTAIL".into())],
        );
        assert_eq!(limit.signal_at(&context("head", 1), start), None);
        assert_eq!(
            limit.signal_at(
                &context("headEND\nTAILsuffix\n", 2),
                start + Duration::from_millis(20)
            ),
            None
        );
    }
    #[test]
    fn deadline_is_strict_and_stop_word_newlines_are_not_emitted_newlines() {
        let start = Instant::now();
        let limit = PredictionTimeLimit::new(Duration::from_millis(10), vec![]);
        assert_eq!(limit.signal_at(&context("x", 1), start), None);
        assert_eq!(
            limit.signal_at(&context("x\n", 2), start + Duration::from_millis(10)),
            None
        );
        let limit = PredictionTimeLimit::new(
            Duration::from_millis(10),
            vec![StopSequence::Text("END\n".into())],
        );
        assert_eq!(limit.signal_at(&context("x", 1), start), None);
        assert_eq!(
            limit.signal_at(&context("xEND\n", 2), start + Duration::from_millis(20)),
            None
        );
    }
}
