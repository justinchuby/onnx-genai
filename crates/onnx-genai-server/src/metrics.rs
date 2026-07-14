use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

#[cfg(feature = "metrics")]
use std::fmt::Write;

use axum::http::StatusCode;

const ENDPOINTS: [&str; 14] = [
    "/health",
    "/v1/models",
    "/v1/sessions",
    "/v1/sessions/{id}",
    "/v1/completions",
    "/v1/chat/completions",
    "/v1/status",
    "/metrics",
    "/v1/debug/config",
    "/v1/debug/sessions",
    "/v1/debug/kv",
    "/v1/debug/trace",
    "/v1/debug/trace/perfetto",
    "unknown",
];
const STATUS_CODES: usize = 600;
const LATENCY_BUCKETS_NS: [u64; 14] = [
    5_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
    75_000_000,
    100_000_000,
    150_000_000,
    200_000_000,
    300_000_000,
    500_000_000,
    750_000_000,
    1_000_000_000,
    2_500_000_000,
    5_000_000_000,
];

struct Histogram {
    buckets: [AtomicU64; LATENCY_BUCKETS_NS.len()],
    count: AtomicU64,
    sum_ns: AtomicU64,
}

impl Histogram {
    const fn new() -> Self {
        Self {
            buckets: [const { AtomicU64::new(0) }; LATENCY_BUCKETS_NS.len()],
            count: AtomicU64::new(0),
            sum_ns: AtomicU64::new(0),
        }
    }

    fn observe(&self, duration: Duration) {
        let ns = duration.as_nanos().min(u64::MAX as u128) as u64;
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_ns.fetch_add(ns, Ordering::Relaxed);
        for (bound, bucket) in LATENCY_BUCKETS_NS.iter().zip(&self.buckets) {
            if ns <= *bound {
                bucket.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

struct Registry {
    requests: [[AtomicU64; STATUS_CODES]; ENDPOINTS.len()],
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    ttft: Histogram,
    e2e: Histogram,
    active_sessions: AtomicU64,
    pending: AtomicU64,
    batch_size: AtomicU64,
    prefix_cache_hits: AtomicU64,
    prefix_cache_lookups: AtomicU64,
    rejections: AtomicU64,
    trace_ids: AtomicU64,
}

static REGISTRY: Registry = Registry {
    requests: [const { [const { AtomicU64::new(0) }; STATUS_CODES] }; ENDPOINTS.len()],
    prompt_tokens: AtomicU64::new(0),
    completion_tokens: AtomicU64::new(0),
    ttft: Histogram::new(),
    e2e: Histogram::new(),
    active_sessions: AtomicU64::new(0),
    pending: AtomicU64::new(0),
    batch_size: AtomicU64::new(0),
    prefix_cache_hits: AtomicU64::new(0),
    prefix_cache_lookups: AtomicU64::new(0),
    rejections: AtomicU64::new(0),
    trace_ids: AtomicU64::new(1),
};

pub(crate) struct GenerationMetrics {
    started: Instant,
    first_token_seen: bool,
}

impl GenerationMetrics {
    pub(crate) fn start() -> Self {
        decrement(&REGISTRY.pending);
        REGISTRY.batch_size.fetch_add(1, Ordering::Relaxed);
        Self {
            started: Instant::now(),
            first_token_seen: false,
        }
    }

    pub(crate) fn token(&mut self) {
        if !self.first_token_seen {
            REGISTRY.ttft.observe(self.started.elapsed());
            self.first_token_seen = true;
        }
    }

    pub(crate) fn result(&mut self, completion_tokens: usize, prefix_cache_hit_len: usize) {
        if completion_tokens > 0 {
            self.token();
        }
        REGISTRY
            .completion_tokens
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        REGISTRY
            .prefix_cache_lookups
            .fetch_add(1, Ordering::Relaxed);
        if prefix_cache_hit_len > 0 {
            REGISTRY.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Drop for GenerationMetrics {
    fn drop(&mut self) {
        REGISTRY.e2e.observe(self.started.elapsed());
        decrement(&REGISTRY.batch_size);
    }
}

pub(crate) fn request_finished(path: &str, status: StatusCode) {
    let endpoint = endpoint_index(path);
    let code = usize::from(status.as_u16());
    if code < STATUS_CODES {
        REGISTRY.requests[endpoint][code].fetch_add(1, Ordering::Relaxed);
    }
    if status == StatusCode::TOO_MANY_REQUESTS {
        REGISTRY.rejections.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn request_started() -> u64 {
    REGISTRY.trace_ids.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn latest_trace_id() -> u64 {
    REGISTRY.trace_ids.load(Ordering::Relaxed).saturating_sub(1)
}

pub(crate) fn generation_queued() {
    REGISTRY.pending.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn generation_queue_cancelled() {
    decrement(&REGISTRY.pending);
}

pub(crate) fn add_prompt_tokens(count: usize) {
    REGISTRY
        .prompt_tokens
        .fetch_add(count as u64, Ordering::Relaxed);
}

pub(crate) fn active_sessions_added(count: usize) {
    REGISTRY
        .active_sessions
        .fetch_add(count as u64, Ordering::Relaxed);
}

pub(crate) fn active_sessions_removed(count: usize) {
    for _ in 0..count {
        decrement(&REGISTRY.active_sessions);
    }
}

pub(crate) fn snapshot() -> MetricsSnapshot {
    let prompt_tokens = REGISTRY.prompt_tokens.load(Ordering::Relaxed);
    let completion_tokens = REGISTRY.completion_tokens.load(Ordering::Relaxed);
    MetricsSnapshot {
        active_sessions: REGISTRY.active_sessions.load(Ordering::Relaxed),
        pending_requests: REGISTRY.pending.load(Ordering::Relaxed),
        current_batch_size: REGISTRY.batch_size.load(Ordering::Relaxed),
        prefix_cache_hits: REGISTRY.prefix_cache_hits.load(Ordering::Relaxed),
        prefix_cache_lookups: REGISTRY.prefix_cache_lookups.load(Ordering::Relaxed),
        rejections: REGISTRY.rejections.load(Ordering::Relaxed),
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
    }
}

pub(crate) struct MetricsSnapshot {
    pub(crate) active_sessions: u64,
    pub(crate) pending_requests: u64,
    pub(crate) total_tokens: u64,
    pub(crate) current_batch_size: u64,
    pub(crate) prefix_cache_hits: u64,
    pub(crate) prefix_cache_lookups: u64,
    pub(crate) rejections: u64,
}

#[cfg(feature = "metrics")]
pub(crate) fn encode_prometheus() -> String {
    let mut output = String::with_capacity(4096);
    output.push_str("# HELP onnx_genai_requests_total Total HTTP requests.\n");
    output.push_str("# TYPE onnx_genai_requests_total counter\n");
    for (endpoint_index, statuses) in REGISTRY.requests.iter().enumerate() {
        for (status, value) in statuses.iter().enumerate() {
            let value = value.load(Ordering::Relaxed);
            if value != 0 {
                writeln!(
                    output,
                    "onnx_genai_requests_total{{endpoint=\"{}\",status=\"{status}\"}} {value}",
                    ENDPOINTS[endpoint_index]
                )
                .expect("writing to String cannot fail");
            }
        }
    }
    counter(
        &mut output,
        "onnx_genai_prompt_tokens_total",
        "Prompt tokens processed.",
        REGISTRY.prompt_tokens.load(Ordering::Relaxed),
    );
    counter(
        &mut output,
        "onnx_genai_completion_tokens_total",
        "Completion tokens generated.",
        REGISTRY.completion_tokens.load(Ordering::Relaxed),
    );
    let snapshot = snapshot();
    counter(
        &mut output,
        "onnx_genai_tokens_generated_total",
        "Total prompt and completion tokens processed.",
        snapshot.total_tokens,
    );
    histogram(
        &mut output,
        "onnx_genai_time_to_first_token_seconds",
        "Time to first generated token.",
        &REGISTRY.ttft,
    );
    histogram(
        &mut output,
        "onnx_genai_e2e_request_latency_seconds",
        "End-to-end generation latency.",
        &REGISTRY.e2e,
    );
    gauge(
        &mut output,
        "onnx_genai_sessions_active",
        "Currently active persistent sessions.",
        snapshot.active_sessions,
    );
    gauge(
        &mut output,
        "onnx_genai_requests_waiting",
        "Generation requests waiting for driver execution.",
        snapshot.pending_requests,
    );
    gauge(
        &mut output,
        "onnx_genai_batch_size_current",
        "Current generation batch size.",
        REGISTRY.batch_size.load(Ordering::Relaxed),
    );
    let hits = REGISTRY.prefix_cache_hits.load(Ordering::Relaxed);
    let lookups = REGISTRY.prefix_cache_lookups.load(Ordering::Relaxed);
    counter(
        &mut output,
        "onnx_genai_prefix_cache_hits_total",
        "Generation requests with a prefix-cache hit.",
        hits,
    );
    counter(
        &mut output,
        "onnx_genai_prefix_cache_lookups_total",
        "Generation requests checked for prefix-cache reuse.",
        lookups,
    );
    output.push_str("# HELP onnx_genai_prefix_cache_hit_rate Prefix-cache hit ratio.\n");
    output.push_str("# TYPE onnx_genai_prefix_cache_hit_rate gauge\n");
    let rate = if lookups == 0 {
        0.0
    } else {
        hits as f64 / lookups as f64
    };
    writeln!(output, "onnx_genai_prefix_cache_hit_rate {rate}").expect("String write");
    counter(
        &mut output,
        "onnx_genai_rejections_total",
        "HTTP requests rejected for overload.",
        REGISTRY.rejections.load(Ordering::Relaxed),
    );
    output
}

#[cfg(feature = "metrics")]
fn counter(output: &mut String, name: &str, help: &str, value: u64) {
    writeln!(output, "# HELP {name} {help}").expect("String write");
    writeln!(output, "# TYPE {name} counter").expect("String write");
    writeln!(output, "{name} {value}").expect("String write");
}

#[cfg(feature = "metrics")]
fn gauge(output: &mut String, name: &str, help: &str, value: u64) {
    writeln!(output, "# HELP {name} {help}").expect("String write");
    writeln!(output, "# TYPE {name} gauge").expect("String write");
    writeln!(output, "{name} {value}").expect("String write");
}

#[cfg(feature = "metrics")]
fn histogram(output: &mut String, name: &str, help: &str, histogram: &Histogram) {
    writeln!(output, "# HELP {name} {help}").expect("String write");
    writeln!(output, "# TYPE {name} histogram").expect("String write");
    for (bound, bucket) in LATENCY_BUCKETS_NS.iter().zip(&histogram.buckets) {
        writeln!(
            output,
            "{name}_bucket{{le=\"{}\"}} {}",
            *bound as f64 / 1_000_000_000.0,
            bucket.load(Ordering::Relaxed)
        )
        .expect("String write");
    }
    let count = histogram.count.load(Ordering::Relaxed);
    writeln!(output, "{name}_bucket{{le=\"+Inf\"}} {count}").expect("String write");
    writeln!(
        output,
        "{name}_sum {}",
        histogram.sum_ns.load(Ordering::Relaxed) as f64 / 1_000_000_000.0
    )
    .expect("String write");
    writeln!(output, "{name}_count {count}").expect("String write");
}

fn endpoint_index(path: &str) -> usize {
    match path {
        "/health" => 0,
        "/v1/models" => 1,
        "/v1/sessions" => 2,
        path if path.starts_with("/v1/sessions/") => 3,
        "/v1/completions" => 4,
        "/v1/chat/completions" => 5,
        "/v1/status" => 6,
        "/metrics" => 7,
        "/v1/debug/config" => 8,
        "/v1/debug/sessions" => 9,
        "/v1/debug/kv" => 10,
        "/v1/debug/trace/perfetto" => 12,
        "/v1/debug/trace" => 11,
        _ => 13,
    }
}

fn decrement(value: &AtomicU64) {
    let _ = value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_sub(1))
    });
}
