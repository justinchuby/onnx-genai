use onnx_genai_engine::GenerateResult;

/// Differential generation compares semantic output and cache behavior;
/// independent executions have different measured durations. Validate measured
/// token ownership before excluding only this nondeterministic field.
pub fn without_timings(mut result: GenerateResult) -> GenerateResult {
    if let Some(sample) = result.timings.take() {
        assert_eq!(sample.generated_tokens, result.token_ids.len());
        assert_eq!(sample.cached_prompt_tokens, result.prefix_cache_hit_len);
    }
    result
}

pub fn results_without_timings(results: Vec<GenerateResult>) -> Vec<GenerateResult> {
    results.into_iter().map(without_timings).collect()
}
