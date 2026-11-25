//! llama.cpp timing fields rendered from engine-owned measurements.
use onnx_genai_engine::GenerationTimings;
use serde_json::{Value, json};

pub(crate) fn render(sample: GenerationTimings) -> Value {
    let prompt_ms = sample.prompt_elapsed.as_secs_f64() * 1000.0;
    let predicted_ms = if sample.generated_tokens > 0 {
        (sample.generation_elapsed.as_secs_f64() * 1000.0).max(0.001)
    } else {
        0.0
    };
    // The first token comes from the final prompt forward, so generation
    // throughput counts subsequent decode steps, as in pinned llama.cpp.
    let steps = sample.generated_tokens.saturating_sub(1);
    let per_token = |ms: f64, count: usize| if count > 0 { ms / count as f64 } else { 0.0 };
    let per_second = |ms: f64, count: usize| {
        if ms > 0.0 {
            count as f64 * 1000.0 / ms
        } else {
            0.0
        }
    };
    json!({
        "cache_n":sample.cached_prompt_tokens,
        "prompt_n":sample.prompt_tokens,
        "prompt_ms":prompt_ms,
        "prompt_per_token_ms":per_token(prompt_ms,sample.prompt_tokens),
        "prompt_per_second":per_second(prompt_ms,sample.prompt_tokens),
        "predicted_n":sample.generated_tokens,
        "predicted_ms":predicted_ms,
        "predicted_per_token_ms":per_token(predicted_ms,steps),
        "predicted_per_second":per_second(predicted_ms,steps)
    })
}

pub(crate) fn render_prompt_progress(
    progress: onnx_genai_engine::pipeline::PromptProgress,
) -> Value {
    json!({"total":progress.total,"cache":progress.cache,"processed":progress.processed,
        "time_ms":u64::try_from(progress.elapsed.as_millis()).unwrap_or(u64::MAX)})
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    #[test]
    fn llama_timings_rates_count_the_first_token_as_prompt_work() {
        let mut sample = GenerationTimings {
            prompt_tokens: 3,
            cached_prompt_tokens: 4,
            generated_tokens: 3,
            prompt_elapsed: Duration::from_millis(6),
            generation_elapsed: Duration::from_millis(8),
        };
        let wire = render(sample);
        assert_eq!(wire["cache_n"], 4);
        assert_eq!(wire["prompt_n"], 3);
        assert_eq!(wire["prompt_per_token_ms"], 2.0);
        assert_eq!(wire["prompt_per_second"], 500.0);
        assert_eq!(wire["predicted_n"], 3);
        assert_eq!(wire["predicted_per_token_ms"], 4.0);
        assert_eq!(wire["predicted_per_second"], 250.0);
        sample.generated_tokens = 1;
        sample.generation_elapsed = Duration::ZERO;
        sample.prompt_elapsed = Duration::ZERO;
        let wire = render(sample);
        assert_eq!(wire["predicted_ms"], 0.001);
        assert_eq!(wire["predicted_per_token_ms"], 0.0);
        assert_eq!(wire["predicted_per_second"], 0.0);
        assert_eq!(wire["prompt_per_second"], 0.0);
        sample.generated_tokens = 0;
        assert_eq!(render(sample)["predicted_ms"], 0.0);
    }
}
