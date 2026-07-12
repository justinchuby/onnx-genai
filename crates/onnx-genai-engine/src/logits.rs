//! Logit processor chain.

/// Context passed to logit processors.
pub struct ProcessorContext {
    /// Tokens generated so far in this sequence.
    pub generated_tokens: Vec<u32>,
    /// Current step index.
    pub step: usize,
}

/// A logit processor modifies the logit distribution before sampling.
pub trait LogitProcessor: Send + Sync {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext);
    fn name(&self) -> &str;
}

/// Ordered chain of logit processors.
pub struct ProcessorChain {
    processors: Vec<Box<dyn LogitProcessor>>,
}

impl ProcessorChain {
    pub fn new() -> Self {
        Self { processors: Vec::new() }
    }

    pub fn add(&mut self, processor: Box<dyn LogitProcessor>) {
        self.processors.push(processor);
    }

    pub fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
        for proc in &self.processors {
            proc.process(logits, context);
        }
    }
}

impl Default for ProcessorChain {
    fn default() -> Self {
        Self::new()
    }
}

// --- Built-in processors ---

pub struct TemperatureProcessor {
    pub temperature: f32,
}

impl LogitProcessor for TemperatureProcessor {
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if self.temperature != 1.0 && self.temperature > 0.0 {
            for logit in logits.iter_mut() {
                *logit /= self.temperature;
            }
        }
    }

    fn name(&self) -> &str { "temperature" }
}

pub struct RepetitionPenaltyProcessor {
    pub penalty: f32,
}

impl LogitProcessor for RepetitionPenaltyProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
        for &token_id in &context.generated_tokens {
            if let Some(logit) = logits.get_mut(token_id as usize) {
                if *logit > 0.0 {
                    *logit /= self.penalty;
                } else {
                    *logit *= self.penalty;
                }
            }
        }
    }

    fn name(&self) -> &str { "repetition_penalty" }
}

pub struct TopKProcessor {
    pub top_k: usize,
}

impl LogitProcessor for TopKProcessor {
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if self.top_k == 0 || self.top_k >= logits.len() {
            return;
        }

        // Find the top-k threshold
        let mut sorted: Vec<f32> = logits.to_vec();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[self.top_k - 1];

        // Mask everything below threshold
        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &str { "top_k" }
}

pub struct TopPProcessor {
    pub top_p: f32,
}

impl LogitProcessor for TopPProcessor {
    fn process(&self, logits: &mut [f32], _context: &ProcessorContext) {
        if self.top_p >= 1.0 {
            return;
        }

        // Softmax to get probabilities
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let mut probs: Vec<(usize, f32)> = logits.iter()
            .enumerate()
            .map(|(i, &l)| (i, (l - max_logit).exp() / exp_sum))
            .collect();

        // Sort by probability descending
        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find cumulative probability cutoff
        let mut cumulative = 0.0;
        let mut cutoff_idx = probs.len();
        for (i, &(_, prob)) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative > self.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Mask tokens beyond cutoff
        for &(idx, _) in probs.iter().skip(cutoff_idx) {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    fn name(&self) -> &str { "top_p" }
}
