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
    pub(crate) context: Option<ContextUsage>,
    pub(crate) budget_cap: Option<BudgetCap>,
    pub(crate) finish_reason: Option<String>,
    /// The sampling policy the decode loop actually used this turn.
    pub(crate) sampling_policy: Option<SamplingPolicy>,
    pub(crate) prefix_cache_hit: Option<usize>,
    pub(crate) memory: MemoryUsage,
    pub(crate) pages: Option<PageActivity>,
    /// Reuse across a multi-component (multimodal) pipeline's generations.
    pub(crate) multimodal_reuse: Option<MultimodalReuse>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ContextUsage {
    pub(crate) used_tokens: usize,
    pub(crate) max_tokens: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BudgetCap {
    pub(crate) requested_max_new_tokens: usize,
    pub(crate) admitted_max_new_tokens: usize,
}

/// The sampling policy the decode loop actually used for a turn.
///
/// This is captured from the `GenerateOptions` *after*
/// [`resolve_sampling_defaults`](onnx_genai::config::GenerateOptions::resolve_sampling_defaults)
/// — the exact struct handed to generation — so surfacing it reports what
/// generation did rather than resolving the policy a second time (which is the
/// `/session`-summary defect this exists to catch: a display-side resolution
/// that can silently disagree with the decode loop, #385/#392).
#[derive(Debug, Clone, Copy)]
pub(crate) struct SamplingPolicy {
    pub(crate) greedy: bool,
    pub(crate) temperature: f32,
    pub(crate) top_p: f32,
    pub(crate) top_k: usize,
}

impl SamplingPolicy {
    /// Compact, machine-parseable rendering for the `--stats` / `--profile`
    /// output. Pure ASCII, so its display width equals its scalar count.
    pub(crate) fn to_stats_part(self) -> String {
        format!(
            "sampling greedy={} temperature={} top_p={} top_k={}",
            self.greedy, self.temperature, self.top_p, self.top_k
        )
    }
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
    /// Per-turn numbers for an interactive session.
    ///
    /// The full report is a page long and ends in a per-stage table, which is
    /// the wrong shape to print after every REPL turn. This keeps the numbers a
    /// reader actually watches turn to turn — how much went in, how much came
    /// out, how fast, and how much was not recomputed — in a compact block.
    ///
    /// Single-line layout kept for non-TTY/scripted opt-in output.
    pub(crate) fn to_stats_line(&self) -> String {
        let mut parts = Vec::new();
        if let Some(prompt_tokens) = self.prompt_tokens {
            parts.push(format!("{prompt_tokens} in"));
        }
        if self.timings.tokens() > 0 {
            parts.push(format!("{} out", self.timings.tokens()));
        }
        if let Some(policy) = &self.sampling_policy {
            parts.push(policy.to_stats_part());
        }
        if let Some(hit) = self.prefix_cache_hit {
            if let Some(prompt_tokens) = self.prompt_tokens.filter(|tokens| *tokens > 0) {
                parts.push(format!(
                    "cache {}/{} ({})",
                    format_token_count(hit),
                    format_token_count(prompt_tokens),
                    format_percent(hit, prompt_tokens)
                ));
            } else if hit > 0 {
                parts.push(format!("cache {}", format_token_count(hit)));
            }
        }
        if let Some(context) = self.context {
            parts.push(format!("ctx {context}"));
        }
        if let Some(cap) = self.budget_cap {
            parts.push(format!(
                "max-new capped {} -> {}",
                format_token_count(cap.requested_max_new_tokens),
                format_token_count(cap.admitted_max_new_tokens)
            ));
        }
        if let Some(backend) = &self.decode_backend {
            parts.push(format!("backend {backend}"));
        }
        if let Some(rate) = self.timings.decode_tokens_per_second() {
            parts.push(format!("{rate:.1} tok/s"));
        }
        if let Some(ttft) = self.timings.time_to_first_token() {
            parts.push(format!("ttft {:.0} ms", ttft.as_secs_f64() * 1000.0));
        }
        if let Some(reuse) = self.multimodal_reuse
            && reuse.prefix_reused_tokens > 0
        {
            parts.push(format!(
                "mm-prefix {}",
                format_token_count(reuse.prefix_reused_tokens as usize)
            ));
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
        if let Some(pages) = self.pages.filter(|pages| !pages.is_idle()) {
            parts.push(pages.to_verbose_stats_part());
        }
        if let Some(peak) = self.memory.peak_resident_bytes {
            parts.push(format!("rss {}", format_bytes(peak)));
        }
        Self::format_stats_parts(parts)
    }

    /// TTY layout with an explicit field-boundary break.
    ///
    /// The first line is generation performance and termination; the second is
    /// cache, context, scheduler and memory behavior. Keeping the break
    /// deliberate avoids terminal-width-dependent accidental wrapping.
    pub(crate) fn to_stats_block(&self) -> String {
        let (headline, resources) = self.stats_parts();
        match (headline.is_empty(), resources.is_empty()) {
            (true, true) => "[ no measurements for this turn ]".to_string(),
            (false, true) => Self::format_stats_parts(headline),
            (true, false) => Self::format_stats_parts(resources),
            (false, false) => format!(
                "{}\n{}",
                Self::format_stats_parts(headline),
                Self::format_stats_parts(resources)
            ),
        }
    }

    fn stats_parts(&self) -> (Vec<String>, Vec<String>) {
        let mut headline = Vec::new();
        let mut resources = Vec::new();
        if let Some(prompt_tokens) = self.prompt_tokens {
            headline.push(format!("{prompt_tokens} in"));
        }
        if self.timings.tokens() > 0 {
            headline.push(format!("{} out", self.timings.tokens()));
        }
        if let Some(policy) = &self.sampling_policy {
            headline.push(policy.to_stats_part());
        }
        if let Some(backend) = &self.decode_backend {
            headline.push(format!("backend {backend}"));
        }
        if let Some(rate) = self.timings.decode_tokens_per_second() {
            headline.push(format!("{rate:.1} tok/s"));
        }
        if let Some(rate) = self.timings.end_to_end_tokens_per_second() {
            headline.push(format!("e2e {rate:.1} tok/s"));
        }
        if let Some(ttft) = self.timings.time_to_first_token() {
            headline.push(format!("ttft {:.0} ms", ttft.as_secs_f64() * 1000.0));
        }
        if let Some(reason) = &self.finish_reason {
            headline.push(format!("finish {}", compact_finish_reason(reason)));
        }
        if let Some(hit) = self.prefix_cache_hit {
            if let Some(prompt_tokens) = self.prompt_tokens.filter(|tokens| *tokens > 0) {
                resources.push(format!(
                    "cache {}/{} {}",
                    format_token_count(hit),
                    format_token_count(prompt_tokens),
                    format_percent(hit, prompt_tokens)
                ));
            } else if hit > 0 {
                resources.push(format!("cache {}", format_token_count(hit)));
            }
        }
        if let Some(context) = self.context {
            resources.push(format!(
                "ctx {}/{}",
                format_token_count(context.used_tokens),
                format_token_count(context.max_tokens)
            ));
        }
        if let Some(cap) = self.budget_cap {
            headline.push(format!(
                "cap {}->{}",
                format_token_count(cap.requested_max_new_tokens),
                format_token_count(cap.admitted_max_new_tokens)
            ));
        }
        if let Some(reuse) = self.multimodal_reuse
            && reuse.prefix_reused_tokens > 0
        {
            resources.push(format!(
                "mm {}",
                format_token_count(reuse.prefix_reused_tokens as usize)
            ));
        }
        if let Some(reuse) = self.multimodal_reuse
            && reuse.encoder_hits + reuse.encoder_misses > 0
        {
            resources.push(format!(
                "enc {}/{}",
                reuse.encoder_hits,
                reuse.encoder_hits + reuse.encoder_misses
            ));
        }
        if let Some(pages) = self.pages.filter(|pages| !pages.is_idle()) {
            resources.push(pages.to_stats_part());
        }
        if let Some(peak) = self.memory.peak_resident_bytes {
            resources.push(format!("rss {}", format_bytes(peak)));
        }
        (headline, resources)
    }

    fn format_stats_parts(parts: Vec<String>) -> String {
        if parts.is_empty() {
            "[ no measurements for this turn ]".to_string()
        } else {
            format!("[ {} ]", parts.join(" · "))
        }
    }
}

impl std::fmt::Display for ContextUsage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} / {}",
            format_token_count(self.used_tokens),
            format_token_count(self.max_tokens)
        )
    }
}

fn format_token_count(tokens: usize) -> String {
    if tokens >= 1000 {
        format!("{:.1}k", tokens as f64 / 1000.0)
    } else {
        tokens.to_string()
    }
}

fn format_percent(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        return "0%".to_string();
    }
    format!("{:.0}%", numerator as f64 / denominator as f64 * 100.0)
}

fn compact_finish_reason(reason: &str) -> &str {
    match reason {
        "MaxTokens" => "max",
        "StopSequence" => "stop-seq",
        "Eos" | "EOS" => "eos",
        "Stop" => "stop",
        "Interrupted" => "interrupt",
        other => other,
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

    fn to_stats_part(self) -> String {
        let mut part = format!("pg +{}/-{}", self.allocations, self.frees);
        if self.hot_evictions > 0 {
            part.push_str(&format!(" hot {}", self.hot_evictions));
        }
        if self.prefix_evictions > 0 {
            part.push_str(&format!(" pref {}", self.prefix_evictions));
        }
        if self.allocation_failures > 0 {
            part.push_str(&format!(" fail {}", self.allocation_failures));
        }
        part
    }

    fn to_verbose_stats_part(self) -> String {
        let mut part = format!("pages +{} / -{}", self.allocations, self.frees);
        if self.hot_evictions > 0 {
            part.push_str(&format!(" hot-evict {}", self.hot_evictions));
        }
        if self.prefix_evictions > 0 {
            part.push_str(&format!(" prefix-evict {}", self.prefix_evictions));
        }
        if self.allocation_failures > 0 {
            part.push_str(&format!(" fail {}", self.allocation_failures));
        }
        part
    }
}

/// Memory the run needed, from the kernel and from the engine's own accounting.
#[derive(Debug, Default, Clone)]
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
    /// Non-zero means the device ledger is over its live ceiling.
    pub(crate) device_oversubscribed_bytes: Option<u64>,
    /// What the device ceiling is carved into. Reporting only a total answers
    /// "how much" but never "why", which is the question when a model does not
    /// fit: weights are fixed, but the KV budget is what a longer context eats.
    pub(crate) composition: Option<DeviceComposition>,
    /// Latest native activation planner result. This is separate from the
    /// device-reservation breakdown: it measures what activation sharing would
    /// save inside the executor, even before the allocator uses it.
    pub(crate) activation_plan: Option<ActivationPlanMemory>,
    /// Load-time static weight placement plan, when the loader found pageable
    /// layer regions. The row is the honesty check for `device_policy`: if a
    /// user asks for `gpu_layers:N`, `--profile` must show the translated byte
    /// plan instead of silently accepting a knob that nothing reached.
    pub(crate) weight_placement: Option<WeightPlacementMemory>,
    /// Authoritative load-time memory strategy, including inference and
    /// override provenance.
    pub(crate) memory_strategy_plan: Option<onnx_genai::engine::MemoryStrategyPlan>,
    /// What the virtual-memory arena did to physical memory, when this build
    /// can have one. See [`VmmArena`].
    pub(crate) vmm_arena: Option<VmmArena>,
}

/// Virtual-memory arena counters, as reported by the engine.
///
/// # Why this is printed even when it is all zero
///
/// The arena once logged that it was installed and committed **zero bytes**
/// for an entire generation (#659). A field that disappears when nothing
/// happened cannot distinguish "no arena" from "an arena doing nothing", which
/// is precisely the bug. So the row is printed whenever the build could have
/// an arena, and `reserved 0 B` is a visible answer rather than an absence.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct VmmArena {
    pub(crate) commits: u64,
    pub(crate) releases: u64,
    pub(crate) committed_bytes: u64,
    pub(crate) reserved_bytes: u64,
    pub(crate) peak_committed_bytes: u64,
    pub(crate) allocations: u64,
    /// Non-zero means the arena's granule reference counts do not balance.
    pub(crate) ref_underflows: u64,
    /// Non-zero means a byte counter was decremented below zero and clamped.
    pub(crate) byte_underflows: u64,
    /// Non-zero means committed device bytes were not recorded in the adopted governor.
    pub(crate) unaccounted_committed_bytes: u64,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ActivationPlanMemory {
    pub(crate) complete: bool,
    pub(crate) peak_bytes: u64,
    pub(crate) naive_bytes: u64,
    pub(crate) savings_ratio: f64,
    pub(crate) unknown_sizes: usize,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct WeightPlacementMemory {
    pub(crate) coordinated_weight_budget_bytes: u64,
    pub(crate) effective_budget_bytes: u64,
    pub(crate) device_bytes: u64,
    pub(crate) host_bytes: u64,
    pub(crate) explanation: String,
}

/// How the device memory ceiling is divided.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct DeviceComposition {
    pub(crate) model_weights_bytes: u64,
    /// None when the engine never measured it, which is not zero.
    pub(crate) activations_bytes: Option<u64>,
    /// None when the engine never measured it, which is not zero.
    pub(crate) runtime_overhead_bytes: Option<u64>,
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
            || self.activations_bytes.is_some()
            || self.runtime_overhead_bytes.is_some()
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
            && self.device_limit_bytes.is_none()
            && self.device_oversubscribed_bytes.is_none()
            && self.composition.is_none()
            && self.activation_plan.is_none()
            && self.weight_placement.is_none()
            && self.memory_strategy_plan.is_none()
            && self.vmm_arena.is_none()
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
            let share = self
                .prompt_tokens
                .filter(|tokens| *tokens > 0)
                .map(|tokens| format!(" ({})", format_percent(hit, tokens)))
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "{:<24} {:>10} tokens{share}",
                "prefix cache reuse", hit
            );
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
        if let Some(policy) = &self.sampling_policy {
            let _ = writeln!(
                out,
                "{:<24} {}",
                "sampling policy",
                if policy.greedy {
                    "greedy".to_string()
                } else {
                    format!(
                        "temperature {} / top_p {} / top_k {}",
                        policy.temperature, policy.top_p, policy.top_k
                    )
                }
            );
        }
        if let Some(cap) = self.budget_cap {
            let _ = writeln!(
                out,
                "{:<24} {} -> {} max_new_tokens (KV budget)",
                "scheduler cap", cap.requested_max_new_tokens, cap.admitted_max_new_tokens
            );
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
                    ("  model weights", Some(composition.model_weights_bytes)),
                    ("  activations", composition.activations_bytes),
                    ("  runtime overhead", composition.runtime_overhead_bytes),
                    ("  kv cache", Some(composition.kv_bytes)),
                ] {
                    // An unmeasured component is omitted: "0 B" beside a real
                    // figure reads as "this model has none", which is wrong.
                    // `None` now says that outright, so a genuine measurement of
                    // zero is printed rather than swallowed with it.
                    let Some(bytes) = bytes.filter(|bytes| *bytes > 0) else {
                        continue;
                    };
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
                .filter(|(_, bytes)| bytes.is_none())
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
            if let Some(bytes) = self
                .memory
                .device_oversubscribed_bytes
                .filter(|bytes| *bytes > 0)
            {
                let _ = writeln!(
                    out,
                    "{:<24} FAULT: device tier oversubscribed by {}",
                    "memory ledger",
                    format_bytes(bytes)
                );
            }
            if let Some(plan) = self.memory.activation_plan {
                if plan.complete {
                    let _ = writeln!(
                        out,
                        "{:<24} {} vs {} ({:.1}% saved)",
                        "activation plan",
                        format_bytes(plan.peak_bytes),
                        format_bytes(plan.naive_bytes),
                        plan.savings_ratio * 100.0
                    );
                } else {
                    let _ = writeln!(
                        out,
                        "{:<24} deferred ({} unknown sizes)",
                        "activation plan", plan.unknown_sizes
                    );
                }
            }
            if let Some(plan) = &self.memory.weight_placement {
                let _ = writeln!(
                    out,
                    "{:<24} {} device / {} host (budget {})",
                    "weight placement",
                    format_bytes(plan.device_bytes),
                    format_bytes(plan.host_bytes),
                    format_bytes(plan.effective_budget_bytes)
                );
                let _ = writeln!(out, "{:<24} {}", "  explanation", plan.explanation);
            }
            if let Some(plan) = &self.memory.memory_strategy_plan {
                let _ = writeln!(
                    out,
                    "{:<24} {:?} (inferred {:?}, access {:?})",
                    "memory strategy",
                    plan.strategy,
                    plan.inferred_strategy,
                    plan.weight_access_pattern
                );
                for decision in &plan.decisions {
                    let inferred = decision
                        .inferred_value
                        .as_deref()
                        .map(|value| format!(", inferred {value}"))
                        .unwrap_or_default();
                    let _ = writeln!(
                        out,
                        "{:<24} {}={} [{:?}{inferred}] — {}",
                        "  strategy decision",
                        decision.field,
                        decision.value,
                        decision.source,
                        decision.reason
                    );
                }
            }
            if let Some(arena) = self.memory.vmm_arena {
                if arena.reserved_bytes == 0 && arena.commits == 0 {
                    let _ = writeln!(
                        out,
                        "{:<24} not installed (set ONNX_GENAI_CUDA_VMM=1)",
                        "vmm arena"
                    );
                } else {
                    let _ = writeln!(
                        out,
                        "{:<24} {} committed of {} reserved (peak {})",
                        "vmm arena",
                        format_bytes(arena.committed_bytes),
                        format_bytes(arena.reserved_bytes),
                        format_bytes(arena.peak_committed_bytes)
                    );
                    // Allocations per commit is the suballocation ratio: 1.0
                    // means every allocation took its own granule, which is the
                    // regression that would make 2 MiB granularity unaffordable.
                    let _ = writeln!(
                        out,
                        "{:<24} {} allocs, {} commits, {} releases",
                        "vmm arena activity", arena.allocations, arena.commits, arena.releases
                    );
                    // Printed only when non-zero, and phrased as a fault
                    // rather than a statistic: a balanced arena has nothing to
                    // say here, and an unbalanced one has already unmapped
                    // memory some other allocation believes it owns.
                    if arena.ref_underflows > 0 {
                        let _ = writeln!(
                            out,
                            "{:<24} BUG: {} granule release(s) with a zero reference count",
                            "vmm arena", arena.ref_underflows
                        );
                    }
                    if arena.byte_underflows > 0 {
                        let _ = writeln!(
                            out,
                            "{:<24} BUG: {} byte counter underflow(s); committed bytes above \
                             are a lower bound",
                            "vmm arena", arena.byte_underflows
                        );
                    }
                    if arena.unaccounted_committed_bytes > 0 {
                        let _ = writeln!(
                            out,
                            "{:<24} FAULT: {} committed byte(s) not recorded in the memory ledger",
                            "vmm arena", arena.unaccounted_committed_bytes
                        );
                    }
                }
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
            for (phase, total_ns, calls) in &executor_phases {
                if !is_executor_phase_time_row(phase) {
                    continue;
                }
                let _ = writeln!(
                    out,
                    "{:<34} {:>12.3} {:>10}",
                    phase,
                    *total_ns as f64 / 1e6,
                    calls
                );
            }
            let byte_rows = executor_phases
                .iter()
                .filter(|(phase, _, _)| !is_executor_phase_time_row(phase))
                .collect::<Vec<_>>();
            if !byte_rows.is_empty() {
                let _ = writeln!(out, "\nnative executor byte counters:");
                let _ = writeln!(
                    out,
                    "{:<34} {:>12} {:>10} {:>12}",
                    "counter", "total_mb", "calls", "mb/call"
                );
                let _ = writeln!(out, "{}", "-".repeat(72));
                for (phase, total_bytes, calls) in byte_rows {
                    let total_mb = *total_bytes as f64 / (1024.0 * 1024.0);
                    let mb_per_call = if *calls > 0 {
                        total_mb / *calls as f64
                    } else {
                        0.0
                    };
                    let _ = writeln!(
                        out,
                        "{phase:<34} {total_mb:>12.3} {calls:>10} {mb_per_call:>12.3}"
                    );
                }
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
        if let Some(backend) = &self.decode_backend {
            fields.push(format!("\"decode_backend\":{}", json_string(backend)));
        }
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
            if let Some(prompt_tokens) = self.prompt_tokens.filter(|tokens| *tokens > 0) {
                fields.push(format!(
                    "\"prefix_cache_hit_percent\":{:.4}",
                    hit as f64 / prompt_tokens as f64 * 100.0
                ));
            }
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
        if let Some(cap) = self.budget_cap {
            fields.push(format!(
                "\"budget_cap\":{{\"requested_max_new_tokens\":{},\"admitted_max_new_tokens\":{}}}",
                cap.requested_max_new_tokens, cap.admitted_max_new_tokens
            ));
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
        if let Some(bytes) = self
            .memory
            .device_oversubscribed_bytes
            .filter(|bytes| *bytes > 0)
        {
            fields.push(format!("\"device_oversubscribed_bytes\":{bytes}"));
        }
        if let Some(composition) = self
            .memory
            .composition
            .filter(DeviceComposition::fixed_reservation_measured)
        {
            // Mirror the text form: an unmeasured component is omitted rather
            // than emitted as 0, and the gap is named. Emitting a bare 0 tells a
            // machine reader "this model has no activations", which is a
            // confident wrong answer -- and it disagreed with the text output
            // rendered from the same data.
            let mut parts = vec![format!(
                "\"model_weights_bytes\":{}",
                composition.model_weights_bytes
            )];
            let mut unmeasured = Vec::new();
            for (key, bytes) in [
                ("activations_bytes", composition.activations_bytes),
                ("runtime_overhead_bytes", composition.runtime_overhead_bytes),
            ] {
                match bytes {
                    // Absent, not zero. A machine reader told "0" concludes the
                    // model has no activations; the distinction now survives
                    // from the engine all the way here rather than being
                    // reconstructed from a sentinel.
                    None => unmeasured.push(format!("\"{key}\"")),
                    Some(bytes) => parts.push(format!("\"{key}\":{bytes}")),
                }
            }
            parts.push(format!("\"kv_bytes\":{}", composition.kv_bytes));
            parts.push(format!("\"kv_pages\":{}", composition.kv_pages));
            parts.push(format!("\"kv_page_bytes\":{}", composition.kv_page_bytes));
            if !unmeasured.is_empty() {
                parts.push(format!("\"unmeasured\":[{}]", unmeasured.join(",")));
            }
            fields.push(format!(
                "\"device_memory_breakdown\":{{{}}}",
                parts.join(",")
            ));
        }
        if let Some(plan) = self.memory.activation_plan {
            fields.push(format!(
                "\"activation_memory_plan\":{{\"complete\":{},\"peak_bytes\":{},\"naive_bytes\":{},\"savings_ratio\":{:.6},\"unknown_sizes\":{}}}",
                plan.complete,
                plan.peak_bytes,
                plan.naive_bytes,
                plan.savings_ratio,
                plan.unknown_sizes
            ));
        }
        if let Some(plan) = &self.memory.weight_placement {
            fields.push(format!(
                "\"weight_placement\":{{\"coordinated_weight_budget_bytes\":{},\"effective_budget_bytes\":{},\"device_bytes\":{},\"host_bytes\":{},\"explanation\":{}}}",
                plan.coordinated_weight_budget_bytes,
                plan.effective_budget_bytes,
                plan.device_bytes,
                plan.host_bytes,
                json_string(&plan.explanation)
            ));
        }
        if let Some(plan) = &self.memory.memory_strategy_plan {
            fields.push(format!(
                "\"memory_strategy\":{}",
                serde_json::to_string(plan).expect("memory strategy plan must serialize")
            ));
        }
        if let Some(arena) = self.memory.vmm_arena {
            fields.push(format!(
                "\"vmm_arena\":{{\"commits\":{},\"releases\":{},\"committed_bytes\":{},\"reserved_bytes\":{},\"peak_committed_bytes\":{},\"allocations\":{},\"ref_underflows\":{},\"byte_underflows\":{},\"unaccounted_committed_bytes\":{}}}",
                arena.commits,
                arena.releases,
                arena.committed_bytes,
                arena.reserved_bytes,
                arena.peak_committed_bytes,
                arena.allocations,
                arena.ref_underflows,
                arena.byte_underflows,
                arena.unaccounted_committed_bytes
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

fn is_executor_phase_time_row(phase: &str) -> bool {
    !phase.ends_with("_bytes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_width::UnicodeWidthStr;

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
    fn executor_phase_table_excludes_byte_counters_from_millisecond_rows() {
        assert!(is_executor_phase_time_row("run_scoped.setup_total.top"));
        assert!(!is_executor_phase_time_row("activation_plan.peak_bytes"));
        assert!(!is_executor_phase_time_row(
            "collect_outputs.top_host_bytes"
        ));
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
            activations_bytes: Some(512 * 1024 * 1024),
            runtime_overhead_bytes: Some(256 * 1024 * 1024),
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
        // Everything was measured here, so nothing is withheld and no gap is
        // named -- the counterpart to the unmeasured case below.
        assert_eq!(breakdown["activations_bytes"], 512u64 * 1024 * 1024);
        assert_eq!(breakdown["runtime_overhead_bytes"], 256u64 * 1024 * 1024);
        assert!(breakdown.get("unmeasured").is_none(), "{breakdown}");
    }

    #[test]
    fn unmeasured_components_are_named_rather_than_shown_as_zero() {
        let mut profile = RunProfile::new("m".to_string());
        profile.timings = timings(10, &[10]);
        profile.memory.device_limit_bytes = Some(4 * 1024 * 1024 * 1024);
        // What the engine reports today: weights and KV measured, the rest not.
        profile.memory.composition = Some(DeviceComposition {
            model_weights_bytes: 1024 * 1024 * 1024,
            activations_bytes: None,
            runtime_overhead_bytes: None,
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

        // The JSON is rendered from the same data and must not contradict it.
        // It used to emit "activations_bytes":0 unconditionally, so a machine
        // reader was told "this model has no activations" while a human reading
        // the text output was told the figure was never measured. This test
        // previously only looked at the text, which is how that survived.
        let value: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();
        let breakdown = &value["device_memory_breakdown"];
        assert_eq!(breakdown["model_weights_bytes"], 1024u64 * 1024 * 1024);
        assert!(
            breakdown.get("activations_bytes").is_none(),
            "an unmeasured component must be absent, not zero: {breakdown}"
        );
        assert!(
            breakdown.get("runtime_overhead_bytes").is_none(),
            "an unmeasured component must be absent, not zero: {breakdown}"
        );
        let unmeasured = breakdown["unmeasured"]
            .as_array()
            .expect("the gap must be named for machine readers too");
        assert!(unmeasured.iter().any(|name| name == "activations_bytes"));
        assert!(
            unmeasured
                .iter()
                .any(|name| name == "runtime_overhead_bytes")
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
            activations_bytes: None,
            runtime_overhead_bytes: None,
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
    fn weight_placement_explanation_is_visible_in_profile_text_and_json() {
        let mut profile = RunProfile::new("m".to_string());
        profile.timings = timings(10, &[10]);
        profile.memory.weight_placement = Some(WeightPlacementMemory {
            coordinated_weight_budget_bytes: 2048,
            effective_budget_bytes: 1024,
            device_bytes: 512,
            host_bytes: 1536,
            explanation: "VRAM placement: source=gpu_layers:1".to_string(),
        });

        let text = profile.to_text();
        assert!(text.contains("weight placement"), "{text}");
        assert!(text.contains("source=gpu_layers:1"), "{text}");

        let value: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();
        assert_eq!(value["weight_placement"]["device_bytes"], 512);
        assert_eq!(
            value["weight_placement"]["explanation"],
            "VRAM placement: source=gpu_layers:1"
        );
    }

    #[test]
    fn memory_strategy_and_provenance_are_visible_in_profile_text_and_json() {
        let mut profile = RunProfile::new("m".to_string());
        profile.timings = timings(10, &[10]);
        let mut plan = onnx_genai::engine::MemoryStrategyPlan::unknown(128, None, "test strategy");
        plan.strategy = onnx_genai::engine::MemoryStrategy::Compatibility;
        plan.inferred_strategy = onnx_genai::engine::MemoryStrategy::DynamicWeightResidency;
        plan.decisions.push(
            onnx_genai::engine::MemoryStrategyDecision::new(
                "strategy",
                "Compatibility",
                onnx_genai::engine::DecisionSource::CompatibilityDefault,
                "automatic activation remains gated",
                "no explicit budget",
            )
            .with_inferred_value("DynamicWeightResidency"),
        );
        profile.memory.memory_strategy_plan = Some(plan);

        let text = profile.to_text();
        assert!(text.contains("memory strategy"), "{text}");
        assert!(text.contains("CompatibilityDefault"), "{text}");
        assert!(text.contains("inferred DynamicWeightResidency"), "{text}");

        let value: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();
        assert_eq!(value["memory_strategy"]["strategy"], "Compatibility");
        assert_eq!(
            value["memory_strategy"]["inferred_strategy"],
            "DynamicWeightResidency"
        );
        assert_eq!(
            value["memory_strategy"]["decisions"][1]["source"],
            "CompatibilityDefault"
        );
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

    #[test]
    fn stats_line_reports_context_usage() {
        let mut profile = RunProfile::new("m".to_string());
        profile.prompt_tokens = Some(3100);
        profile.context = Some(ContextUsage {
            used_tokens: 3128,
            max_tokens: 8192,
        });

        let line = profile.to_stats_line();

        assert!(line.contains("ctx 3.1k / 8.2k"), "{line}");
    }

    #[test]
    fn stats_line_and_reports_expose_prefix_cache_reuse() {
        let mut profile = RunProfile::new("m".to_string());
        profile.prompt_tokens = Some(613);
        profile.prefix_cache_hit = Some(598);

        let line = profile.to_stats_line();
        let text = profile.to_text();
        let json: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();

        assert!(line.contains("cache 598/613 (98%)"), "{line}");
        assert!(
            text.contains("prefix cache reuse") && text.contains("598 tokens (98%)"),
            "{text}"
        );
        assert_eq!(json["prefix_cache_hit_tokens"], 598);
        assert!((json["prefix_cache_hit_percent"].as_f64().unwrap() - 97.553).abs() < 0.01);
    }

    #[test]
    fn stats_block_has_a_deliberate_two_line_full_layout() {
        let mut profile = RunProfile::new("m".to_string());
        profile.prompt_tokens = Some(613);
        profile.timings = timings(116, &vec![24; 63]);
        profile.decode_backend = Some("native".to_string());
        profile.finish_reason = Some("StopSequence".to_string());
        profile.prefix_cache_hit = Some(598);
        profile.context = Some(ContextUsage {
            used_tokens: 677,
            max_tokens: 8192,
        });
        profile.budget_cap = Some(BudgetCap {
            requested_max_new_tokens: 3584,
            admitted_max_new_tokens: 128,
        });
        profile.multimodal_reuse = Some(MultimodalReuse {
            encoder_hits: 1,
            encoder_misses: 1,
            encoder_bytes: 1_048_576,
            prefix_reused_tokens: 120,
            prefill_tokens: 64,
        });
        profile.pages = Some(PageActivity {
            allocations: 5,
            frees: 2,
            hot_evictions: 1,
            prefix_evictions: 3,
            allocation_failures: 1,
        });
        profile.memory.peak_resident_bytes = Some(2_684_354_560);

        let block = profile.to_stats_block();
        let lines = block.lines().collect::<Vec<_>>();

        assert_eq!(
            block,
            "[ 613 in · 64 out · backend native · 41.7 tok/s · e2e 39.3 tok/s · ttft 116 ms · finish stop-seq · cap 3.6k->128 ]\n\
             [ cache 598/613 98% · ctx 677/8.2k · mm 120 · enc 1/2 · pg +5/-2 hot 1 pref 3 fail 1 · rss 2.5 GiB ]"
        );
        assert_eq!(lines.len(), 2, "rendered stats block:\n{block}");
        assert_eq!(
            UnicodeWidthStr::width(lines[0]),
            114,
            "headline display width changed; rendered stats block:\n{block}"
        );
        assert_eq!(
            UnicodeWidthStr::width(lines[1]),
            100,
            "resource display width changed; rendered stats block:\n{block}"
        );
    }

    #[test]
    fn stats_block_width_uses_display_width_not_scalar_count() {
        // An unknown finish reason passes through `compact_finish_reason`
        // unchanged, so a CJK character in a model's stop token reaches the
        // rendered line. "字" is U+5B57: one scalar value, two terminal columns.
        // This test exists to prove that `UnicodeWidthStr::width` and
        // `chars().count()` disagree for such input — if they agreed, the
        // width assertion in `stats_block_has_a_deliberate_two_line_full_layout`
        // above would pass even with `chars().count()`, giving false confidence.
        let mut profile = RunProfile::new("m".to_string());
        profile.finish_reason = Some("字".to_string());

        let block = profile.to_stats_block();
        let line = block.lines().next().unwrap_or("");

        let char_count = line.chars().count();
        let display_width = UnicodeWidthStr::width(line);

        assert!(
            display_width > char_count,
            "a line containing a wide character must have display_width > chars().count(); \
             got display_width={display_width}, chars={char_count}; line: {line:?}"
        );
    }

    #[test]
    fn stats_line_names_page_activity_when_it_happened() {
        let mut profile = RunProfile::new("m".to_string());
        profile.pages = Some(PageActivity {
            allocations: 5,
            frees: 2,
            hot_evictions: 1,
            prefix_evictions: 3,
            allocation_failures: 1,
        });

        let line = profile.to_stats_line();

        assert!(line.contains("pages +5 / -2"), "{line}");
        assert!(line.contains("hot-evict 1"), "{line}");
        assert!(line.contains("prefix-evict 3"), "{line}");
        assert!(line.contains("fail 1"), "{line}");
    }

    #[test]
    fn stats_line_reports_resolved_backend() {
        let mut profile = RunProfile::new("m".to_string());
        profile.decode_backend = Some("ort".to_string());

        let line = profile.to_stats_line();
        let json: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();

        assert!(line.contains("backend ort"), "{line}");
        assert!(!line.contains("backend auto"), "{line}");
        assert_eq!(json["decode_backend"], "ort");
    }

    #[test]
    fn stats_line_and_reports_expose_scheduler_budget_cap() {
        let mut profile = RunProfile::new("m".to_string());
        profile.budget_cap = Some(BudgetCap {
            requested_max_new_tokens: 3584,
            admitted_max_new_tokens: 128,
        });

        let line = profile.to_stats_line();
        let text = profile.to_text();
        let json: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();

        assert!(line.contains("max-new capped 3.6k -> 128"), "{line}");
        assert!(
            text.contains("scheduler cap") && text.contains("3584 -> 128"),
            "{text}"
        );
        assert_eq!(json["budget_cap"]["requested_max_new_tokens"], 3584);
        assert_eq!(json["budget_cap"]["admitted_max_new_tokens"], 128);
    }

    /// A measured zero and an unmeasured component are different things.
    ///
    /// They used to be the same `u64`, so the reports reconstructed the
    /// distinction from `== 0` -- which meant a component genuinely measured at
    /// zero was reported as "not yet measured", and before #629 an unmeasured
    /// one was published to machine readers as a hard `0`. The type now carries
    /// it, so neither reconstruction is needed.
    #[test]
    fn a_measured_zero_is_not_reported_as_unmeasured() {
        let mut profile = RunProfile::new("m".to_string());
        profile.timings = timings(10, &[10]);
        profile.memory.device_limit_bytes = Some(4 * 1024 * 1024 * 1024);
        profile.memory.composition = Some(DeviceComposition {
            model_weights_bytes: 1024 * 1024 * 1024,
            // Measured, and genuinely zero.
            activations_bytes: Some(0),
            // Never measured.
            runtime_overhead_bytes: None,
            kv_bytes: 2 * 1024 * 1024 * 1024,
            kv_pages: 100,
            kv_page_bytes: 32 * 1024 * 1024,
        });

        let text = profile.to_text();
        assert!(
            text.contains("runtime overhead not yet measured"),
            "the unmeasured one must be named:\n{text}"
        );
        assert!(
            !text.contains("activations and runtime overhead not yet measured"),
            "activations were measured, so they must not be listed as unmeasured:\n{text}"
        );

        let value: serde_json::Value = serde_json::from_str(&profile.to_json()).unwrap();
        let breakdown = &value["device_memory_breakdown"];
        assert_eq!(
            breakdown["activations_bytes"], 0,
            "a measured zero is a number and belongs in the JSON"
        );
        let unmeasured = breakdown["unmeasured"].as_array().expect("named");
        assert!(
            unmeasured
                .iter()
                .any(|name| name == "runtime_overhead_bytes")
        );
        assert!(
            !unmeasured.iter().any(|name| name == "activations_bytes"),
            "activations were measured: {breakdown}"
        );
    }
}
