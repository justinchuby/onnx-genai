//! Logit processor chain.
//!
//! Phase 1 uses the documented processor order:
//! repetition penalties -> constraints/stop checks -> temperature -> top-k -> top-p.

use std::collections::HashSet;

/// Token id used by the generation engine.
pub type TokenId = u32;

/// A stop sequence expressed either as generated text or token ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopSequence {
    /// Stop when generated text ends with this string.
    Text(String),
    /// Stop when generated token ids end with this sequence.
    Tokens(Vec<TokenId>),
}

impl StopSequence {
    fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Tokens(tokens) => tokens.is_empty(),
        }
    }
}

/// Context passed to logit processors.
#[derive(Debug, Clone, Default)]
pub struct ProcessorContext {
    /// Prompt token ids for the sequence.
    pub prompt_tokens: Vec<TokenId>,
    /// Tokens generated so far in this sequence.
    pub generated_tokens: Vec<TokenId>,
    /// Generated text so far, if detokenization is available.
    pub generated_text: String,
    /// Current step index.
    pub step: usize,
}

/// Non-logit side-channel emitted by processors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessorSignal {
    /// A configured stop sequence matched the current generated output.
    StopSequence { index: usize },
}

/// A logit processor modifies the logit distribution before sampling.
pub trait LogitProcessor: Send + Sync {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext);
    fn name(&self) -> &str;

    /// Return a termination signal for the current context, if this processor owns one.
    fn signal(&self, _context: &ProcessorContext) -> Option<ProcessorSignal> {
        None
    }
}

/// Ordered chain of logit processors.
#[derive(Default)]
pub struct ProcessorChain {
    processors: Vec<Box<dyn LogitProcessor>>,
}

impl ProcessorChain {
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    pub fn add(&mut self, processor: Box<dyn LogitProcessor>) {
        self.processors.push(processor);
    }

    /// Apply processors in insertion order.
    pub fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
        for proc in &self.processors {
            proc.process(logits, context);
        }
    }

    /// Return the first termination signal from the ordered chain.
    pub fn signal(&self, context: &ProcessorContext) -> Option<ProcessorSignal> {
        self.processors.iter().find_map(|proc| proc.signal(context))
    }

    /// Processor names in configured order, useful for diagnostics and tests.
    pub fn names(&self) -> Vec<&str> {
        self.processors.iter().map(|proc| proc.name()).collect()
    }
}

// --- Built-in processors ---

pub struct TemperatureProcessor {
    pub temperature: f32,
}

impl LogitProcessor for TemperatureProcessor {
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if self.temperature.is_finite() && self.temperature > 0.0 && self.temperature != 1.0 {
            for logit in logits.iter_mut() {
                *logit /= self.temperature;
            }
        }
    }

    fn name(&self) -> &str {
        "temperature"
    }
}

pub struct RepetitionPenaltyProcessor {
    pub penalty: f32,
}

impl LogitProcessor for RepetitionPenaltyProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
        if !self.penalty.is_finite() || self.penalty <= 0.0 || self.penalty == 1.0 {
            return;
        }

        let mut seen = HashSet::new();
        for &token_id in context
            .prompt_tokens
            .iter()
            .chain(context.generated_tokens.iter())
        {
            if !seen.insert(token_id) {
                continue;
            }
            if let Some(logit) = logits.get_mut(token_id as usize) {
                if *logit > 0.0 {
                    *logit /= self.penalty;
                } else {
                    *logit *= self.penalty;
                }
            }
        }
    }

    fn name(&self) -> &str {
        "repetition_penalty"
    }
}

pub struct StopSequenceProcessor {
    pub sequences: Vec<StopSequence>,
}

impl StopSequenceProcessor {
    pub fn new(sequences: Vec<StopSequence>) -> Self {
        Self { sequences }
    }
}

impl LogitProcessor for StopSequenceProcessor {
    fn process(&self, _logits: &mut [f32], _context: &ProcessorContext) {}

    fn signal(&self, context: &ProcessorContext) -> Option<ProcessorSignal> {
        self.sequences
            .iter()
            .enumerate()
            .find_map(|(index, sequence)| {
                if sequence.is_empty() {
                    return None;
                }

                let matched = match sequence {
                    StopSequence::Text(text) => context.generated_text.ends_with(text),
                    StopSequence::Tokens(tokens) => context.generated_tokens.ends_with(tokens),
                };

                matched.then_some(ProcessorSignal::StopSequence { index })
            })
    }

    fn name(&self) -> &str {
        "stop_sequence"
    }
}

pub struct TopKProcessor {
    pub top_k: usize,
}

impl LogitProcessor for TopKProcessor {
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if self.top_k == 0 || self.top_k >= logits.len() {
            return;
        }

        let mut sorted: Vec<f32> = logits.iter().copied().filter(|v| !v.is_nan()).collect();
        if sorted.is_empty() {
            return;
        }
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[self.top_k.saturating_sub(1).min(sorted.len() - 1)];

        for logit in logits.iter_mut() {
            if logit.is_nan() || *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &str {
        "top_k"
    }
}

pub struct TopPProcessor {
    pub top_p: f32,
}

impl LogitProcessor for TopPProcessor {
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if !self.top_p.is_finite() || self.top_p >= 1.0 || logits.is_empty() {
            return;
        }

        let max_logit = logits
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .fold(f32::NEG_INFINITY, f32::max);
        if !max_logit.is_finite() {
            return;
        }

        let exp_sum: f32 = logits
            .iter()
            .map(|&l| {
                if l.is_nan() {
                    0.0
                } else {
                    (l - max_logit).exp()
                }
            })
            .sum();
        if !exp_sum.is_finite() || exp_sum <= 0.0 {
            return;
        }

        let mut probs: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let prob = if l.is_nan() {
                    0.0
                } else {
                    (l - max_logit).exp() / exp_sum
                };
                (i, prob)
            })
            .collect();

        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumulative = 0.0;
        let mut keep_count = 0;
        let cutoff = self.top_p.max(0.0);
        for &(_, prob) in &probs {
            keep_count += 1;
            cumulative += prob;
            if cumulative >= cutoff {
                break;
            }
        }

        for &(idx, _) in probs.iter().skip(keep_count) {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    fn name(&self) -> &str {
        "top_p"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(prompt_tokens: Vec<TokenId>, generated_tokens: Vec<TokenId>) -> ProcessorContext {
        ProcessorContext {
            prompt_tokens,
            generated_tokens,
            generated_text: String::new(),
            step: 0,
        }
    }

    #[test]
    fn repetition_penalty_applies_once_per_seen_token() {
        let processor = RepetitionPenaltyProcessor { penalty: 2.0 };
        let mut logits = vec![4.0, -4.0, 8.0];
        processor.process(&mut logits, &context(vec![0, 0], vec![1, 1]));
        assert_eq!(logits, vec![2.0, -8.0, 8.0]);
    }

    #[test]
    fn top_k_masks_tokens_below_threshold() {
        let processor = TopKProcessor { top_k: 2 };
        let mut logits = vec![0.0, 5.0, 1.0, 4.0];
        processor.process(&mut logits, &ProcessorContext::default());
        assert_eq!(logits, vec![f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY, 4.0]);
    }

    #[test]
    fn top_p_keeps_minimal_nucleus_and_at_least_one_token() {
        let processor = TopPProcessor { top_p: 0.6 };
        let mut logits = vec![3.0, 2.0, 1.0];
        processor.process(&mut logits, &ProcessorContext::default());
        assert!(logits[0].is_finite());
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn temperature_scales_logits() {
        let processor = TemperatureProcessor { temperature: 2.0 };
        let mut logits = vec![2.0, -4.0];
        processor.process(&mut logits, &ProcessorContext::default());
        assert_eq!(logits, vec![1.0, -2.0]);
    }

    #[test]
    fn stop_sequence_signals_token_suffix_and_text_suffix() {
        let processor = StopSequenceProcessor::new(vec![
            StopSequence::Tokens(vec![2, 3]),
            StopSequence::Text("END".to_string()),
        ]);
        let token_context = ProcessorContext {
            generated_tokens: vec![1, 2, 3],
            ..Default::default()
        };
        assert_eq!(
            processor.signal(&token_context),
            Some(ProcessorSignal::StopSequence { index: 0 })
        );

        let text_context = ProcessorContext {
            generated_text: "hello END".to_string(),
            ..Default::default()
        };
        assert_eq!(
            processor.signal(&text_context),
            Some(ProcessorSignal::StopSequence { index: 1 })
        );
    }
}
