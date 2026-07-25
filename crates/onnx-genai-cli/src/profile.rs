//! Performance reporting for `--profile`.
//!
//! Answers the questions a user actually has after a run: how long until the
//! first token appeared, how fast tokens came after that, and where the time
//! went. Latency percentiles are reported alongside the mean because a mean
//! hides the stalls — a run that averages 20 ms/token but pauses for 400 ms
//! mid-sentence feels broken, and only the tail shows it.
//!
//! The measurement lives here rather than in the core crate because the server
//! already has its own telemetry layer (Prometheus counters in
//! `onnx_genai_server::metrics`); a third home for the same numbers would be
//! the duplication this CLI has otherwise been removing.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

/// Wall-clock timings collected across one generation.
#[derive(Debug, Default)]
pub(crate) struct TokenTimings {
    started: Option<Instant>,
    first_token: Option<Duration>,
    last_token_at: Option<Instant>,
    /// Gap before each token after the first: the inter-token latencies.
    gaps: Vec<Duration>,
    total: Option<Duration>,
    tokens: usize,
}

impl TokenTimings {
    pub(crate) fn start(&mut self) {
        let now = Instant::now();
        self.started = Some(now);
        self.last_token_at = Some(now);
    }

    /// Record that one token reached the caller.
    pub(crate) fn token(&mut self) {
        let now = Instant::now();
        let Some(started) = self.started else {
            return;
        };
        if self.first_token.is_none() {
            self.first_token = Some(now.duration_since(started));
        } else if let Some(previous) = self.last_token_at {
            self.gaps.push(now.duration_since(previous));
        }
        self.last_token_at = Some(now);
        self.tokens += 1;
    }

    pub(crate) fn finish(&mut self) {
        if let Some(started) = self.started {
            self.total = Some(started.elapsed());
        }
    }

    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }

    /// Time to first token: how long the user waited before anything appeared.
    pub(crate) fn time_to_first_token(&self) -> Option<Duration> {
        self.first_token
    }

    pub(crate) fn total(&self) -> Option<Duration> {
        self.total
    }

    /// Decode throughput, excluding the prefill wait.
    ///
    /// Reported separately from the end-to-end rate because prefill and decode
    /// scale differently: a long prompt inflates the end-to-end number without
    /// the model decoding any faster or slower.
    pub(crate) fn decode_tokens_per_second(&self) -> Option<f64> {
        let total = self.total?;
        let first = self.first_token?;
        let decode = total.checked_sub(first)?.as_secs_f64();
        let decoded = self.tokens.checked_sub(1)? as f64;
        (decode > 0.0 && decoded > 0.0).then(|| decoded / decode)
    }

    /// End-to-end throughput, including the wait for the first token.
    pub(crate) fn end_to_end_tokens_per_second(&self) -> Option<f64> {
        let total = self.total?.as_secs_f64();
        (total > 0.0 && self.tokens > 0).then(|| self.tokens as f64 / total)
    }

    /// Inter-token latency percentiles, in milliseconds.
    pub(crate) fn inter_token_latency(&self) -> Option<LatencySummary> {
        LatencySummary::from_gaps(&self.gaps)
    }
}

/// Distribution of inter-token latencies, in milliseconds.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LatencySummary {
    pub(crate) mean_ms: f64,
    pub(crate) p50_ms: f64,
    pub(crate) p90_ms: f64,
    pub(crate) p99_ms: f64,
    pub(crate) max_ms: f64,
}

impl LatencySummary {
    fn from_gaps(gaps: &[Duration]) -> Option<Self> {
        if gaps.is_empty() {
            return None;
        }
        let mut millis: Vec<f64> = gaps.iter().map(|gap| gap.as_secs_f64() * 1000.0).collect();
        millis.sort_by(|left, right| left.total_cmp(right));
        Some(Self {
            mean_ms: millis.iter().sum::<f64>() / millis.len() as f64,
            p50_ms: percentile(&millis, 0.50),
            p90_ms: percentile(&millis, 0.90),
            p99_ms: percentile(&millis, 0.99),
            max_ms: *millis.last().expect("non-empty"),
        })
    }
}

/// Nearest-rank percentile of an ascending slice.
fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (fraction * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

/// One measured phase of a run, e.g. model load or image preprocessing.
#[derive(Debug, Clone)]
pub(crate) struct Phase {
    pub(crate) name: &'static str,
    pub(crate) duration: Duration,
}

/// A named number reported alongside the timings, e.g. denoise steps.
#[derive(Debug, Clone)]
pub(crate) struct Counter {
    pub(crate) name: &'static str,
    pub(crate) value: f64,
    pub(crate) unit: &'static str,
}

/// Everything `--profile` reports for one run.
#[derive(Debug, Default)]
pub(crate) struct RunProfile {
    pub(crate) model: String,
    pub(crate) execution_provider: String,
    pub(crate) phases: Vec<Phase>,
    pub(crate) counters: Vec<Counter>,
    pub(crate) timings: TokenTimings,
    pub(crate) prompt_tokens: Option<usize>,
    pub(crate) finish_reason: Option<String>,
    pub(crate) prefix_cache_hit: Option<usize>,
}

impl RunProfile {
    pub(crate) fn new(model: String) -> Self {
        Self {
            model,
            execution_provider: std::env::var("ONNX_GENAI_EP")
                .unwrap_or_else(|_| "cpu (default)".to_string()),
            ..Self::default()
        }
    }

    pub(crate) fn phase(&mut self, name: &'static str, duration: Duration) {
        self.phases.push(Phase { name, duration });
    }

    pub(crate) fn counter(&mut self, name: &'static str, value: f64, unit: &'static str) {
        self.counters.push(Counter { name, value, unit });
    }

    /// Render the human-readable report.
    pub(crate) fn to_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "\n── profile ──────────────────────────────────");
        let _ = writeln!(out, "{:<24} {}", "model", self.model);
        let _ = writeln!(
            out,
            "{:<24} {}",
            "execution provider", self.execution_provider
        );

        for phase in &self.phases {
            let _ = writeln!(
                out,
                "{:<24} {:>10.1} ms",
                phase.name,
                phase.duration.as_secs_f64() * 1000.0
            );
        }
        if let Some(prompt_tokens) = self.prompt_tokens {
            let _ = writeln!(out, "{:<24} {:>10}", "prompt tokens", prompt_tokens);
        }
        if self.timings.tokens() > 0 {
            let _ = writeln!(
                out,
                "{:<24} {:>10}",
                "generated tokens",
                self.timings.tokens()
            );
        }
        if let Some(hit) = self.prefix_cache_hit.filter(|hit| *hit > 0) {
            let _ = writeln!(out, "{:<24} {:>10} tokens", "prefix cache reuse", hit);
        }
        if let Some(ttft) = self.timings.time_to_first_token() {
            let _ = writeln!(
                out,
                "{:<24} {:>10.1} ms",
                "time to first token",
                ttft.as_secs_f64() * 1000.0
            );
        }
        if let Some(total) = self.timings.total() {
            let _ = writeln!(
                out,
                "{:<24} {:>10.1} ms",
                "generation wall time",
                total.as_secs_f64() * 1000.0
            );
        }
        if let Some(rate) = self.timings.decode_tokens_per_second() {
            let _ = writeln!(out, "{:<24} {:>10.2} tok/s", "decode throughput", rate);
        }
        if let Some(rate) = self.timings.end_to_end_tokens_per_second() {
            let _ = writeln!(out, "{:<24} {:>10.2} tok/s", "end-to-end throughput", rate);
        }
        if let Some(latency) = self.timings.inter_token_latency() {
            let _ = writeln!(
                out,
                "{:<24} mean {:.1} / p50 {:.1} / p90 {:.1} / p99 {:.1} / max {:.1} ms",
                "inter-token latency",
                latency.mean_ms,
                latency.p50_ms,
                latency.p90_ms,
                latency.p99_ms,
                latency.max_ms
            );
        }
        for counter in &self.counters {
            let _ = writeln!(
                out,
                "{:<24} {:>10.2} {}",
                counter.name, counter.value, counter.unit
            );
        }
        if let Some(reason) = &self.finish_reason {
            let _ = writeln!(out, "{:<24} {}", "finish reason", reason);
        }

        // The per-stage breakdown (ORT kernels versus our own orchestration)
        // only exists when the engine's stage profiler was enabled.
        let stages = onnx_genai::ort::profile::report(self.timings.tokens() as u64);
        if stages.lines().count() > 2 {
            let _ = writeln!(out, "\nper-stage breakdown:");
            let _ = write!(out, "{stages}");
        }
        out
    }

    /// Render the machine-readable report, for diffing runs or plotting in CI.
    pub(crate) fn to_json(&self) -> String {
        let mut fields: Vec<String> = vec![
            format!("\"model\":{}", json_string(&self.model)),
            format!(
                "\"execution_provider\":{}",
                json_string(&self.execution_provider)
            ),
        ];
        for phase in &self.phases {
            fields.push(format!(
                "\"{}_ms\":{:.3}",
                phase.name.replace(' ', "_"),
                phase.duration.as_secs_f64() * 1000.0
            ));
        }
        if let Some(prompt_tokens) = self.prompt_tokens {
            fields.push(format!("\"prompt_tokens\":{prompt_tokens}"));
        }
        if self.timings.tokens() > 0 {
            fields.push(format!("\"generated_tokens\":{}", self.timings.tokens()));
        }
        if let Some(hit) = self.prefix_cache_hit {
            fields.push(format!("\"prefix_cache_hit_tokens\":{hit}"));
        }
        if let Some(ttft) = self.timings.time_to_first_token() {
            fields.push(format!(
                "\"time_to_first_token_ms\":{:.3}",
                ttft.as_secs_f64() * 1000.0
            ));
        }
        if let Some(total) = self.timings.total() {
            fields.push(format!(
                "\"generation_wall_ms\":{:.3}",
                total.as_secs_f64() * 1000.0
            ));
        }
        if let Some(rate) = self.timings.decode_tokens_per_second() {
            fields.push(format!("\"decode_tokens_per_second\":{rate:.4}"));
        }
        if let Some(rate) = self.timings.end_to_end_tokens_per_second() {
            fields.push(format!("\"end_to_end_tokens_per_second\":{rate:.4}"));
        }
        if let Some(latency) = self.timings.inter_token_latency() {
            fields.push(format!(
                "\"inter_token_latency_ms\":{{\"mean\":{:.3},\"p50\":{:.3},\"p90\":{:.3},\"p99\":{:.3},\"max\":{:.3}}}",
                latency.mean_ms, latency.p50_ms, latency.p90_ms, latency.p99_ms, latency.max_ms
            ));
        }
        for counter in &self.counters {
            fields.push(format!(
                "\"{}\":{:.4}",
                counter.name.replace(' ', "_"),
                counter.value
            ));
        }
        if let Some(reason) = &self.finish_reason {
            fields.push(format!("\"finish_reason\":{}", json_string(reason)));
        }
        format!("{{{}}}", fields.join(","))
    }
}

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            control if control < ' ' => {
                escaped.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => escaped.push(other),
        }
    }
    escaped.push('"');
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timings(first_ms: u64, gaps_ms: &[u64]) -> TokenTimings {
        // Build the timings directly: sleeping for real would make the test slow
        // and flaky, and the arithmetic is what is under test.
        let mut timings = TokenTimings {
            started: Some(Instant::now()),
            first_token: Some(Duration::from_millis(first_ms)),
            tokens: 1 + gaps_ms.len(),
            gaps: gaps_ms
                .iter()
                .map(|millis| Duration::from_millis(*millis))
                .collect(),
            ..TokenTimings::default()
        };
        let total: u64 = first_ms + gaps_ms.iter().sum::<u64>();
        timings.total = Some(Duration::from_millis(total));
        timings
    }

    #[test]
    fn decode_throughput_excludes_the_prefill_wait() {
        // 1 s of prefill, then 4 tokens 100 ms apart.
        let timings = timings(1_000, &[100, 100, 100, 100]);

        let decode = timings.decode_tokens_per_second().unwrap();
        let end_to_end = timings.end_to_end_tokens_per_second().unwrap();

        // Four gaps over 400 ms of decoding.
        assert!((decode - 10.0).abs() < 1e-6, "decode: {decode}");
        // Five tokens over 1.4 s wall time.
        assert!(
            (end_to_end - 5.0 / 1.4).abs() < 1e-6,
            "end to end: {end_to_end}"
        );
        assert!(
            decode > end_to_end,
            "prefill must drag the end-to-end rate below the decode rate"
        );
    }

    #[test]
    fn a_single_token_has_no_decode_rate_but_still_reports_time_to_first_token() {
        let timings = timings(250, &[]);

        assert_eq!(
            timings.time_to_first_token(),
            Some(Duration::from_millis(250))
        );
        assert!(timings.decode_tokens_per_second().is_none());
        assert!(timings.inter_token_latency().is_none());
        assert!(timings.end_to_end_tokens_per_second().is_some());
    }

    #[test]
    fn latency_percentiles_expose_a_stall_the_mean_hides() {
        // Nineteen fast tokens and one long stall.
        let mut gaps = vec![10_u64; 19];
        gaps.push(400);
        let timings = timings(0, &gaps);

        let latency = timings.inter_token_latency().unwrap();

        assert!((latency.p50_ms - 10.0).abs() < 1e-6, "{latency:?}");
        assert!((latency.max_ms - 400.0).abs() < 1e-6, "{latency:?}");
        assert!(
            latency.p99_ms > latency.p50_ms * 10.0,
            "the tail must surface the stall: {latency:?}"
        );
        // The mean alone would look merely mediocre.
        assert!(latency.mean_ms < 30.0, "{latency:?}");
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let sorted = [1.0, 2.0, 3.0, 4.0];

        assert_eq!(percentile(&sorted, 0.5), 2.0);
        assert_eq!(percentile(&sorted, 0.9), 4.0);
        assert_eq!(percentile(&sorted, 1.0), 4.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn the_json_report_is_parseable_and_carries_the_headline_numbers() {
        let mut profile = RunProfile::new("m".to_string());
        profile.phase("model load", Duration::from_millis(120));
        profile.counter("denoise steps", 25.0, "steps");
        profile.timings = timings(200, &[50, 50]);
        profile.prompt_tokens = Some(7);
        profile.finish_reason = Some("stop".to_string());

        let value: serde_json::Value =
            serde_json::from_str(&profile.to_json()).expect("the report must be valid JSON");

        assert_eq!(value["model"], "m");
        assert_eq!(value["prompt_tokens"], 7);
        assert_eq!(value["generated_tokens"], 3);
        assert_eq!(value["finish_reason"], "stop");
        assert!((value["model_load_ms"].as_f64().unwrap() - 120.0).abs() < 1e-6);
        assert!((value["denoise_steps"].as_f64().unwrap() - 25.0).abs() < 1e-6);
        assert!((value["time_to_first_token_ms"].as_f64().unwrap() - 200.0).abs() < 1e-6);
        assert!(value["decode_tokens_per_second"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn the_text_report_names_the_headline_numbers() {
        let mut profile = RunProfile::new("m".to_string());
        profile.timings = timings(200, &[50, 50]);

        let text = profile.to_text();

        for expected in [
            "time to first token",
            "decode throughput",
            "inter-token latency",
            "tok/s",
        ] {
            assert!(text.contains(expected), "missing {expected} in:\n{text}");
        }
    }
}
