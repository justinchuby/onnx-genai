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

use crate::memory::{format_bytes, peak_resident_bytes};

/// Wall-clock timings collected across one generation.
#[derive(Debug, Default)]
pub(crate) struct TokenTimings {
    started: Option<Instant>,
    first_token: Option<Duration>,
    last_token_at: Option<Instant>,
    /// Gap before each token after the first: the inter-token latencies.
    gaps: Vec<Duration>,
    /// Elapsed time at the last token, so decode throughput measures decoding
    /// rather than any teardown that happens after the final callback.
    last_token: Option<Duration>,
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
        self.last_token = Some(now.duration_since(started));
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
        let last = self.last_token?;
        let first = self.first_token?;
        let decode = last.checked_sub(first)?.as_secs_f64();
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
    /// Decode backend actually in use (`ort`, `native`, or `auto`'s choice).
    pub(crate) decode_backend: Option<String>,
    pub(crate) phases: Vec<Phase>,
    pub(crate) counters: Vec<Counter>,
    pub(crate) timings: TokenTimings,
    pub(crate) prompt_tokens: Option<usize>,
    pub(crate) finish_reason: Option<String>,
    pub(crate) prefix_cache_hit: Option<usize>,
    pub(crate) memory: MemoryUsage,
    pub(crate) pages: Option<PageActivity>,
    /// Reuse across a multi-component (multimodal) pipeline's generations.
    pub(crate) multimodal_reuse: Option<MultimodalReuse>,
}

/// What a multimodal pipeline avoided recomputing for this generation.
///
/// An image costs twice: the encoder forward pass, and a prompt in which that
/// one image expanded into hundreds of tokens. This reports how much of each
/// was carried over from a previous turn.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MultimodalReuse {
    pub(crate) encoder_hits: u64,
    pub(crate) encoder_misses: u64,
    pub(crate) encoder_bytes: u64,
    pub(crate) prefix_reused_tokens: u64,
    pub(crate) prefill_tokens: u64,
}

impl TokenTimings {
    /// The numbers worth watching while a reply is still being written.
    ///
    /// Only what is already known mid-turn: totals that need the turn to finish
    /// are left to the summary line printed afterwards.
    pub(crate) fn live_summary(&self) -> String {
        let mut parts = Vec::new();
        if self.tokens() > 0 {
            parts.push(format!("{} tok", self.tokens()));
        }
        if let Some(rate) = self.decode_tokens_per_second() {
            parts.push(format!("{rate:.1} tok/s"));
        }
        if let Some(ttft) = self.time_to_first_token() {
            parts.push(format!("ttft {:.0} ms", ttft.as_secs_f64() * 1000.0));
        }
        parts.join(" · ")
    }
}

impl RunProfile {
    /// One line of per-turn numbers for an interactive session.
    ///
    /// The full report is a page long and ends in a per-stage table, which is
    /// the wrong shape to print after every REPL turn. This keeps the numbers a
    /// reader actually watches turn to turn — how much went in, how much came
    /// out, how fast, and how much was not recomputed — on a single line.
    ///
    /// Fields that were never measured are omitted rather than shown as zero.
    pub(crate) fn to_stats_line(&self) -> String {
        let mut parts = Vec::new();
        if let Some(prompt_tokens) = self.prompt_tokens {
            parts.push(format!("{prompt_tokens} in"));
        }
        if self.timings.tokens() > 0 {
            parts.push(format!("{} out", self.timings.tokens()));
        }
        if let Some(rate) = self.timings.decode_tokens_per_second() {
            parts.push(format!("{rate:.1} tok/s"));
        }
        if let Some(ttft) = self.timings.time_to_first_token() {
            parts.push(format!("ttft {:.0} ms", ttft.as_secs_f64() * 1000.0));
        }

        // Reuse is the whole point of the cache, so it is reported whenever
        // there was any, from whichever cache served it.
        let reused = self.prefix_cache_hit.unwrap_or(0) as u64
            + self
                .multimodal_reuse
                .map_or(0, |reuse| reuse.prefix_reused_tokens);
        if reused > 0 {
            parts.push(format!("{reused} reused"));
        }
        if let Some(reuse) = self.multimodal_reuse
            && reuse.encoder_hits + reuse.encoder_misses > 0
        {
            parts.push(format!(
                "encoder {}/{}",
                reuse.encoder_hits,
                reuse.encoder_hits + reuse.encoder_misses
            ));
        }
        if let Some(peak) = self.memory.peak_resident_bytes {
            parts.push(format!("rss {}", format_bytes(peak)));
        }

        if parts.is_empty() {
            "[ no measurements for this turn ]".to_string()
        } else {
            format!("[ {} ]", parts.join(" · "))
        }
    }
}

impl MultimodalReuse {
    fn is_idle(&self) -> bool {
        self.encoder_hits == 0 && self.encoder_misses == 0 && self.prefix_reused_tokens == 0
    }
}

/// KV page pool activity over the run.
///
/// Reported as deltas across the generation, not lifetime totals: what matters
/// is what *this* run did to the pool.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PageActivity {
    pub(crate) allocations: u64,
    pub(crate) frees: u64,
    pub(crate) hot_evictions: u64,
    pub(crate) prefix_evictions: u64,
    pub(crate) allocation_failures: u64,
}

impl PageActivity {
    /// Difference between two samples, so the report covers only this run.
    pub(crate) fn since(
        before: onnx_genai::kv::PageStats,
        after: onnx_genai::kv::PageStats,
    ) -> Self {
        Self {
            allocations: after.allocations.saturating_sub(before.allocations),
            frees: after.frees.saturating_sub(before.frees),
            hot_evictions: after.hot_evictions.saturating_sub(before.hot_evictions),
            prefix_evictions: after
                .prefix_evictions
                .saturating_sub(before.prefix_evictions),
            allocation_failures: after
                .allocation_failures
                .saturating_sub(before.allocation_failures),
        }
    }

    fn is_idle(&self) -> bool {
        self.allocations == 0
            && self.frees == 0
            && self.hot_evictions == 0
            && self.prefix_evictions == 0
            && self.allocation_failures == 0
    }
}

/// Memory the run needed, from the kernel and from the engine's own accounting.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MemoryUsage {
    /// Process high-water mark: weights, KV pages, ORT arenas and transients
    /// together, which is what decides whether a model fits on a machine.
    pub(crate) peak_resident_bytes: Option<u64>,
    /// KV cache budget the engine sized for this model.
    pub(crate) kv_budget_bytes: Option<u64>,
    /// Tokens that budget holds.
    pub(crate) kv_max_tokens: Option<u64>,
    /// Host RAM the engine's governor accounts as in use. Zero means the
    /// governor tracks nothing on this path, not that nothing is resident —
    /// peak resident memory is the number to trust there.
    pub(crate) host_ram_used_bytes: Option<u64>,
    /// Device memory the engine's governor accounts as in use, and its ceiling.
    ///
    /// This is the engine's own bookkeeping, not the driver's. On a discrete
    /// GPU it is the only device figure reported here, because device
    /// allocations do not appear in the host process's resident set.
    pub(crate) device_used_bytes: Option<u64>,
    pub(crate) device_limit_bytes: Option<u64>,
    /// What the device ceiling is carved into. Reporting only a total answers
    /// "how much" but never "why", which is the question when a model does not
    /// fit: weights are fixed, but the KV budget is what a longer context eats.
    pub(crate) composition: Option<DeviceComposition>,
}

/// How the device memory ceiling is divided.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct DeviceComposition {
    pub(crate) model_weights_bytes: u64,
    pub(crate) activations_bytes: u64,
    pub(crate) runtime_overhead_bytes: u64,
    pub(crate) kv_bytes: u64,
    pub(crate) kv_pages: u64,
    pub(crate) kv_page_bytes: u64,
}

impl DeviceComposition {
    /// Whether the fixed (non-KV) reservation was actually measured.
    ///
    /// The engine currently reserves zero for weights, activations, and runtime
    /// overhead (a documented TODO in its resource governor). Printing those as
    /// `0 B` would read as "this model has no weights", so the composition is
    /// withheld until the engine measures it — the KV figures, which are real,
    /// are still reported.
    fn fixed_reservation_measured(&self) -> bool {
        self.model_weights_bytes > 0
            || self.activations_bytes > 0
            || self.runtime_overhead_bytes > 0
    }
}

impl MemoryUsage {
    /// Sample the kernel's high-water mark. Call after the work is done.
    pub(crate) fn sample_peak(&mut self) {
        self.peak_resident_bytes = peak_resident_bytes();
    }

    fn is_empty(&self) -> bool {
        self.peak_resident_bytes.is_none()
            && self.kv_budget_bytes.is_none()
            && self.host_ram_used_bytes.is_none()
            && self.device_used_bytes.is_none()
            && self.composition.is_none()
    }
}

impl RunProfile {
    pub(crate) fn new(model: String) -> Self {
        Self {
            model,
            // Filled in by the caller, which knows what was actually resolved:
            // the environment is only one of the inputs to that decision.
            execution_provider: "cpu".to_string(),
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
        if let Some(backend) = &self.decode_backend {
            let _ = writeln!(out, "{:<24} {:>10}", "decode backend", backend);
        }

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
        if let Some(reuse) = self.multimodal_reuse.filter(|reuse| !reuse.is_idle()) {
            if reuse.encoder_hits + reuse.encoder_misses > 0 {
                let _ = writeln!(
                    out,
                    "{:<24} {:>10} hit / {} run",
                    "encoder cache", reuse.encoder_hits, reuse.encoder_misses
                );
            }
            if reuse.prefix_reused_tokens > 0 {
                let _ = writeln!(
                    out,
                    "{:<24} {:>10} tokens",
                    "multimodal prefix reuse", reuse.prefix_reused_tokens
                );
            }
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
        if let Some(pages) = self.pages.filter(|pages| !pages.is_idle()) {
            let _ = writeln!(out, "kv page activity:");
            let _ = writeln!(out, "{:<24} {:>10}", "  allocated", pages.allocations);
            let _ = writeln!(out, "{:<24} {:>10}", "  freed", pages.frees);
            if pages.hot_evictions > 0 {
                let _ = writeln!(
                    out,
                    "{:<24} {:>10}  (pool under pressure)",
                    "  evicted from hot tier", pages.hot_evictions
                );
            }
            if pages.prefix_evictions > 0 {
                let _ = writeln!(
                    out,
                    "{:<24} {:>10}",
                    "  reclaimed from prefixes", pages.prefix_evictions
                );
            }
            if pages.allocation_failures > 0 {
                let _ = writeln!(
                    out,
                    "{:<24} {:>10}  (pool exhausted)",
                    "  allocation failures", pages.allocation_failures
                );
            }
        }
        if !self.memory.is_empty() {
            if let Some(peak) = self.memory.peak_resident_bytes {
                let _ = writeln!(
                    out,
                    "{:<24} {:>10}",
                    "peak resident memory",
                    format_bytes(peak)
                );
            }
            if let Some(kv) = self.memory.kv_budget_bytes {
                let tokens = self
                    .memory
                    .kv_max_tokens
                    .map(|tokens| format!(" ({tokens} tokens)"))
                    .unwrap_or_default();
                let _ = writeln!(
                    out,
                    "{:<24} {:>10}{tokens}",
                    "kv cache budget",
                    format_bytes(kv)
                );
            }
            if let Some(host) = self.memory.host_ram_used_bytes.filter(|bytes| *bytes > 0) {
                let _ = writeln!(
                    out,
                    "{:<24} {:>10}",
                    "engine host ram in use",
                    format_bytes(host)
                );
            }
            if let Some(composition) = self
                .memory
                .composition
                .filter(DeviceComposition::fixed_reservation_measured)
            {
                let _ = writeln!(out, "device memory breakdown:");
                for (label, bytes) in [
                    ("  model weights", composition.model_weights_bytes),
                    ("  activations", composition.activations_bytes),
                    ("  runtime overhead", composition.runtime_overhead_bytes),
                    ("  kv cache", composition.kv_bytes),
                ] {
                    // An unmeasured component is omitted: "0 B" beside a real
                    // figure reads as "this model has none", which is wrong.
                    if bytes == 0 {
                        continue;
                    }
                    let share = self
                        .memory
                        .device_limit_bytes
                        .filter(|limit| *limit > 0)
                        .map(|limit| format!("  {:>5.1}%", bytes as f64 / limit as f64 * 100.0))
                        .unwrap_or_default();
                    let _ = writeln!(out, "{label:<24} {:>10}{share}", format_bytes(bytes));
                }
                if composition.kv_pages > 0 {
                    let _ = writeln!(
                        out,
                        "{:<24} {:>10} x {}",
                        "  kv pages",
                        composition.kv_pages,
                        format_bytes(composition.kv_page_bytes)
                    );
                }
                let unmeasured: Vec<&str> = [
                    ("activations", composition.activations_bytes),
                    ("runtime overhead", composition.runtime_overhead_bytes),
                ]
                .into_iter()
                .filter(|(_, bytes)| *bytes == 0)
                .map(|(label, _)| label)
                .collect();
                if !unmeasured.is_empty() {
                    let _ = writeln!(
                        out,
                        "  ({} not yet measured by the engine)",
                        unmeasured.join(" and ")
                    );
                }
            }
            if let Some(device) = self.memory.device_used_bytes.filter(|bytes| *bytes > 0) {
                let limit = self
                    .memory
                    .device_limit_bytes
                    .filter(|bytes| *bytes > 0)
                    .map(|limit| format!(" of {}", format_bytes(limit)))
                    .unwrap_or_default();
                let _ = writeln!(
                    out,
                    "{:<24} {:>10}{limit}",
                    "device memory in use",
                    format_bytes(device)
                );
            }
        }

        // The per-stage breakdown (ORT kernels versus our own orchestration)
        // only exists when the engine's stage profiler was enabled.
        // The native executor's own phase profiler is a separate registry behind
        // a separate switch, so its rows are appended rather than interleaved:
        // presenting them as one table would imply they were measured together.
        let executor_phases = onnx_genai::engine::executor_phase_stats();
        if !executor_phases.is_empty() {
            let _ = writeln!(out, "\nnative executor phases:");
            let _ = writeln!(out, "{:<34} {:>12} {:>10}", "phase", "total_ms", "calls");
            let _ = writeln!(out, "{}", "-".repeat(58));
            for (phase, total_ns, calls) in executor_phases {
                let _ = writeln!(
                    out,
                    "{:<34} {:>12.3} {:>10}",
                    phase,
                    total_ns as f64 / 1e6,
                    calls
                );
            }
        }

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
        if let Some(reuse) = self.multimodal_reuse.filter(|reuse| !reuse.is_idle()) {
            fields.push(format!(
                "\"multimodal_reuse\":{{\"encoder_hits\":{},\"encoder_misses\":{},\"encoder_bytes\":{},\"prefix_reused_tokens\":{},\"prefill_tokens\":{}}}",
                reuse.encoder_hits,
                reuse.encoder_misses,
                reuse.encoder_bytes,
                reuse.prefix_reused_tokens,
                reuse.prefill_tokens
            ));
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
        if let Some(pages) = self.pages.filter(|pages| !pages.is_idle()) {
            fields.push(format!(
                "\"kv_pages\":{{\"allocated\":{},\"freed\":{},\"hot_evictions\":{},\"prefix_evictions\":{},\"allocation_failures\":{}}}",
                pages.allocations,
                pages.frees,
                pages.hot_evictions,
                pages.prefix_evictions,
                pages.allocation_failures
            ));
        }
        if let Some(peak) = self.memory.peak_resident_bytes {
            fields.push(format!("\"peak_resident_bytes\":{peak}"));
        }
        if let Some(kv) = self.memory.kv_budget_bytes {
            fields.push(format!("\"kv_cache_budget_bytes\":{kv}"));
        }
        if let Some(tokens) = self.memory.kv_max_tokens {
            fields.push(format!("\"kv_cache_max_tokens\":{tokens}"));
        }
        if let Some(host) = self.memory.host_ram_used_bytes.filter(|bytes| *bytes > 0) {
            fields.push(format!("\"engine_host_ram_bytes\":{host}"));
        }
        if let Some(device) = self.memory.device_used_bytes.filter(|bytes| *bytes > 0) {
            fields.push(format!("\"device_memory_bytes\":{device}"));
        }
        if let Some(limit) = self.memory.device_limit_bytes.filter(|bytes| *bytes > 0) {
            fields.push(format!("\"device_memory_limit_bytes\":{limit}"));
        }
        if let Some(composition) = self
            .memory
            .composition
            .filter(DeviceComposition::fixed_reservation_measured)
        {
            fields.push(format!(
                "\"device_memory_breakdown\":{{\"model_weights_bytes\":{},\"activations_bytes\":{},\"runtime_overhead_bytes\":{},\"kv_bytes\":{},\"kv_pages\":{},\"kv_page_bytes\":{}}}",
                composition.model_weights_bytes,
                composition.activations_bytes,
                composition.runtime_overhead_bytes,
                composition.kv_bytes,
                composition.kv_pages,
                composition.kv_page_bytes
            ));
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
            last_token: Some(Duration::from_millis(
                first_ms + gaps_ms.iter().sum::<u64>(),
            )),
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
    fn decode_throughput_ignores_teardown_after_the_last_token() {
        // 100 ms of prefill, four tokens 100 ms apart, then 5 s of cleanup
        // before the call returns. Decoding did not get slower.
        let mut timings = timings(100, &[100, 100, 100, 100]);
        timings.total = Some(Duration::from_millis(100 + 400 + 5_000));

        let decode = timings.decode_tokens_per_second().unwrap();

        assert!((decode - 10.0).abs() < 1e-6, "decode: {decode}");
        assert!(
            timings.end_to_end_tokens_per_second().unwrap() < 1.5,
            "end-to-end still reflects the whole wall time"
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
    fn page_activity_is_reported_as_a_delta_and_flags_pressure() {
        let before = onnx_genai::kv::PageStats {
            allocations: 100,
            frees: 90,
            hot_evictions: 0,
            prefix_evictions: 0,
            allocation_failures: 0,
        };
        let after = onnx_genai::kv::PageStats {
            allocations: 140,
            frees: 120,
            hot_evictions: 3,
            prefix_evictions: 7,
            allocation_failures: 1,
        };

        let activity = PageActivity::since(before, after);

        // Only this run's activity, not the pool's lifetime totals.
        assert_eq!(activity.allocations, 40);
        assert_eq!(activity.frees, 30);

        let mut profile = RunProfile::new("m".to_string());
        profile.timings = timings(10, &[10]);
        profile.pages = Some(activity);
        let text = profile.to_text();
        assert!(text.contains("kv page activity"), "{text}");
        assert!(text.contains("pool under pressure"), "{text}");
        assert!(text.contains("pool exhausted"), "{text}");

        let value: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();
        assert_eq!(value["kv_pages"]["allocated"], 40);
        assert_eq!(value["kv_pages"]["hot_evictions"], 3);
    }

    #[test]
    fn an_idle_page_pool_is_not_reported() {
        let mut profile = RunProfile::new("m".to_string());
        profile.timings = timings(10, &[10]);
        // A run that touched no pages says nothing rather than printing zeros.
        profile.pages = Some(PageActivity::default());

        assert!(!profile.to_text().contains("kv page activity"));
        let value: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();
        assert!(value.get("kv_pages").is_none());
    }

    #[test]
    fn memory_is_reported_when_measured_and_omitted_when_not() {
        let mut profile = RunProfile::new("m".to_string());
        profile.timings = timings(10, &[10]);
        assert!(
            !profile.to_text().contains("peak resident memory"),
            "an unmeasured platform must not invent a number"
        );

        profile.memory.peak_resident_bytes = Some(3 * 1024 * 1024);
        profile.memory.kv_budget_bytes = Some(1024 * 1024);
        profile.memory.kv_max_tokens = Some(4096);
        profile.memory.device_used_bytes = Some(2 * 1024 * 1024 * 1024);
        profile.memory.device_limit_bytes = Some(8 * 1024 * 1024 * 1024);

        let text = profile.to_text();
        assert!(text.contains("peak resident memory"), "{text}");
        assert!(text.contains("3.0 MiB"), "{text}");
        assert!(text.contains("4096 tokens"), "{text}");
        // Device memory is reported separately: on a discrete GPU it is invisible
        // to the host resident set.
        assert!(text.contains("device memory in use"), "{text}");
        assert!(text.contains("2.0 GiB of 8.0 GiB"), "{text}");

        let value: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();
        assert_eq!(value["peak_resident_bytes"], 3 * 1024 * 1024);
        assert_eq!(value["kv_cache_max_tokens"], 4096);
        assert_eq!(value["device_memory_bytes"], 2u64 * 1024 * 1024 * 1024);
    }

    #[test]
    fn the_device_breakdown_names_where_memory_went() {
        let mut profile = RunProfile::new("m".to_string());
        profile.timings = timings(10, &[10]);
        profile.memory.device_limit_bytes = Some(8 * 1024 * 1024 * 1024);
        profile.memory.composition = Some(DeviceComposition {
            model_weights_bytes: 2 * 1024 * 1024 * 1024,
            activations_bytes: 512 * 1024 * 1024,
            runtime_overhead_bytes: 256 * 1024 * 1024,
            kv_bytes: 4 * 1024 * 1024 * 1024,
            kv_pages: 2048,
            kv_page_bytes: 2 * 1024 * 1024,
        });

        let text = profile.to_text();

        assert!(text.contains("device memory breakdown"), "{text}");
        assert!(text.contains("model weights"), "{text}");
        assert!(text.contains("kv cache"), "{text}");
        assert!(
            !text.contains("not yet measured"),
            "every component was measured here:\n{text}"
        );
        // Shares make it obvious which component to shrink first.
        assert!(text.contains("25.0%"), "weights share missing:\n{text}");
        assert!(text.contains("50.0%"), "kv share missing:\n{text}");
        assert!(text.contains("2048 x 2.0 MiB"), "{text}");

        let value: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();
        let breakdown = &value["device_memory_breakdown"];
        assert_eq!(breakdown["model_weights_bytes"], 2u64 * 1024 * 1024 * 1024);
        assert_eq!(breakdown["kv_pages"], 2048);
    }

    #[test]
    fn unmeasured_components_are_named_rather_than_shown_as_zero() {
        let mut profile = RunProfile::new("m".to_string());
        profile.timings = timings(10, &[10]);
        profile.memory.device_limit_bytes = Some(4 * 1024 * 1024 * 1024);
        // What the engine reports today: weights and KV measured, the rest not.
        profile.memory.composition = Some(DeviceComposition {
            model_weights_bytes: 1024 * 1024 * 1024,
            activations_bytes: 0,
            runtime_overhead_bytes: 0,
            kv_bytes: 3 * 1024 * 1024 * 1024,
            kv_pages: 100,
            kv_page_bytes: 32 * 1024 * 1024,
        });

        let text = profile.to_text();

        assert!(text.contains("model weights"), "{text}");
        assert!(
            !text.contains("activations                   0 B"),
            "a zero must not read as 'no activations':\n{text}"
        );
        assert!(
            text.contains("activations and runtime overhead not yet measured"),
            "the gap must be named:\n{text}"
        );
    }

    #[test]
    fn an_unmeasured_fixed_reservation_withholds_the_breakdown() {
        let mut profile = RunProfile::new("m".to_string());
        profile.timings = timings(10, &[10]);
        profile.memory.kv_budget_bytes = Some(1024);
        // What the engine reports today: KV is real, the fixed reservation is
        // still a zeroed placeholder.
        profile.memory.composition = Some(DeviceComposition {
            model_weights_bytes: 0,
            activations_bytes: 0,
            runtime_overhead_bytes: 0,
            kv_bytes: 1024,
            kv_pages: 4,
            kv_page_bytes: 256,
        });

        let text = profile.to_text();

        assert!(
            !text.contains("model weights"),
            "an unmeasured reservation must not read as a model with no weights:\n{text}"
        );
        // The KV budget is measured, so it is still reported.
        assert!(text.contains("kv cache budget"), "{text}");
        let value: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();
        assert!(value.get("device_memory_breakdown").is_none());
    }

    #[test]
    fn an_unaccounted_device_is_omitted_rather_than_reported_as_zero() {
        let mut profile = RunProfile::new("m".to_string());
        profile.timings = timings(10, &[10]);
        profile.memory.peak_resident_bytes = Some(1024);
        // A CPU run: the governor tracks no device bytes. Printing "0 B" would
        // read as "the GPU used nothing" rather than "nothing was measured".
        profile.memory.device_used_bytes = Some(0);

        let text = profile.to_text();

        assert!(!text.contains("device memory"), "{text}");
        let value: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();
        assert!(value.get("device_memory_bytes").is_none());
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
