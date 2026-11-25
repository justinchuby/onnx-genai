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
        let previous = self.observed_bytes.swap(text.len(), Ordering::Relaxed);
        // The native protocol checks the text fragment after removing a matched
        // stop word. A newline that belongs only to the stop word is not emitted.
        let mut end = text.len();
        for stop in &self.stops {
            match stop {
                StopSequence::Text(stop) if !stop.is_empty() && text.ends_with(stop) => {
                    end -= stop.len();
                    break;
                }
                StopSequence::Tokens(stop)
                    if !stop.is_empty() && context.generated_tokens.ends_with(stop) =>
                {
                    break;
                }
                _ => {}
            }
        }
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
