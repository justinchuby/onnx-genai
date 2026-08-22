use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

#[cfg(feature = "metrics")]
use std::fmt::Write;

use axum::http::StatusCode;
#[cfg(feature = "metrics")]
use onnx_genai_engine::GovernorSnapshot;

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
    codecoded_forward_steps: AtomicU64,
    codecoded_rows: AtomicU64,
    codecoded_rows_peak: AtomicU64,
    codecoded_max_batch: AtomicU64,
    prefix_cache_hits: AtomicU64,
    prefix_cache_lookups: AtomicU64,
    rejections: AtomicU64,
    trace_ids: AtomicU64,
}

impl Registry {
    const fn new() -> Self {
        Self {
            requests: [const { [const { AtomicU64::new(0) }; STATUS_CODES] }; ENDPOINTS.len()],
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
            ttft: Histogram::new(),
            e2e: Histogram::new(),
            active_sessions: AtomicU64::new(0),
            pending: AtomicU64::new(0),
            batch_size: AtomicU64::new(0),
            codecoded_forward_steps: AtomicU64::new(0),
            codecoded_rows: AtomicU64::new(0),
            codecoded_rows_peak: AtomicU64::new(0),
            codecoded_max_batch: AtomicU64::new(0),
            prefix_cache_hits: AtomicU64::new(0),
            prefix_cache_lookups: AtomicU64::new(0),
            rejections: AtomicU64::new(0),
            trace_ids: AtomicU64::new(1),
        }
    }

    fn request_finished(&self, path: &str, status: StatusCode) {
        let endpoint = endpoint_index(path);
        let code = usize::from(status.as_u16());
        if code < STATUS_CODES {
            self.requests[endpoint][code].fetch_add(1, Ordering::Relaxed);
        }
        if status == StatusCode::TOO_MANY_REQUESTS {
            self.rejections.fetch_add(1, Ordering::Relaxed);
        }
    }
}

static REGISTRY: Registry = Registry::new();

pub(crate) struct GenerationMetrics {
    started: Instant,
    first_token_seen: bool,
}

impl GenerationMetrics {
    pub(crate) fn start() -> Self {
        decrement(&REGISTRY.pending);
        // NOTE (issue #750): this counts *admitted generations*, not sequences
        // that are actually co-decoded in one batched forward pass. On a backend
        // that cannot batch (native, or a legacy / non-shared-buffer ORT model)
        // this gauge still climbs with concurrent requests even though each is
        // decoded one at a time via the per-request fallback path. That is why
        // `onnx_genai_batch_size_current` alone never revealed the missing
        // batching — read `/v1/resources` `batching.supported` for the truth.
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
    REGISTRY.request_finished(path, status);
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
        codecoded_forward_steps: REGISTRY.codecoded_forward_steps.load(Ordering::Relaxed),
        codecoded_rows: REGISTRY.codecoded_rows.load(Ordering::Relaxed),
        codecoded_rows_peak: REGISTRY.codecoded_rows_peak.load(Ordering::Relaxed),
        codecoded_max_batch: REGISTRY.codecoded_max_batch.load(Ordering::Relaxed),
        prefix_cache_hits: REGISTRY.prefix_cache_hits.load(Ordering::Relaxed),
        prefix_cache_lookups: REGISTRY.prefix_cache_lookups.load(Ordering::Relaxed),
        rejections: REGISTRY.rejections.load(Ordering::Relaxed),
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
    }
}

/// Record the rows a batched forward actually carried.
///
/// This is the honest counterpart to `batch_size`: that gauge counts admitted
/// generations, which climbs with concurrency even on a backend that decodes
/// them one at a time, whereas this only moves when a single forward advanced
/// several sequences (issue #750).
///
/// `steps`/`rows` are cumulative sums, so a caller that only has totals can
/// report them without inventing a per-step series; `peak_rows` is tracked
/// separately because a mean cannot recover it.
pub(crate) fn batch_forwards_observed(steps: u64, rows: u64, peak_rows: usize, max_batch: usize) {
    if steps == 0 {
        return;
    }
    REGISTRY
        .codecoded_forward_steps
        .fetch_add(steps, Ordering::Relaxed);
    REGISTRY.codecoded_rows.fetch_add(rows, Ordering::Relaxed);
    REGISTRY
        .codecoded_rows_peak
        .fetch_max(peak_rows as u64, Ordering::Relaxed);
    REGISTRY
        .codecoded_max_batch
        .fetch_max(max_batch as u64, Ordering::Relaxed);
}

pub(crate) struct MetricsSnapshot {
    pub(crate) active_sessions: u64,
    pub(crate) pending_requests: u64,
    pub(crate) total_tokens: u64,
    pub(crate) current_batch_size: u64,
    pub(crate) codecoded_forward_steps: u64,
    pub(crate) codecoded_rows: u64,
    pub(crate) codecoded_rows_peak: u64,
    pub(crate) codecoded_max_batch: u64,
    pub(crate) prefix_cache_hits: u64,
    pub(crate) prefix_cache_lookups: u64,
    pub(crate) rejections: u64,
}

impl MetricsSnapshot {
    /// Mean fraction of the physical decode batch that carried a sequence, over
    /// every batched forward this process issued, or `None` before any ran.
    pub(crate) fn batch_utilization(&self) -> Option<f64> {
        if self.codecoded_forward_steps == 0 || self.codecoded_max_batch == 0 {
            return None;
        }
        let mean_rows = self.codecoded_rows as f64 / self.codecoded_forward_steps as f64;
        Some(mean_rows / self.codecoded_max_batch as f64)
    }
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
        // Counts admitted generations, not sequences co-decoded in one batched
        // pass; it moves even on non-batching backends (issue #750).
        "Current number of admitted generations (not necessarily co-batched).",
        REGISTRY.batch_size.load(Ordering::Relaxed),
    );
    counter(
        &mut output,
        "onnx_genai_batch_forward_steps_total",
        "Batched decode forward passes issued by the continuous batch driver.",
        snapshot.codecoded_forward_steps,
    );
    counter(
        &mut output,
        "onnx_genai_batch_rows_codecoded_total",
        "Sequences carried by those forward passes, summed over steps.",
        snapshot.codecoded_rows,
    );
    gauge(
        &mut output,
        "onnx_genai_batch_rows_peak",
        "Most sequences ever co-decoded in a single forward pass.",
        snapshot.codecoded_rows_peak,
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
    output.push_str(&encode_weight_offload());
    output
}

#[cfg(all(feature = "metrics", feature = "native-cuda"))]
fn encode_weight_offload() -> String {
    let mut output = String::new();
    let stats = onnx_runtime_ep_cuda::global_offload_stats();
    counter(
        &mut output,
        "onnx_genai_cuda_weight_offload_page_ins_total",
        "Process-wide cumulative CUDA weight residency page-ins.",
        stats.page_ins,
    );
    counter(
        &mut output,
        "onnx_genai_cuda_weight_offload_hits_total",
        "Process-wide cumulative CUDA weight residency cache hits.",
        stats.hits,
    );
    counter(
        &mut output,
        "onnx_genai_cuda_weight_offload_evictions_total",
        "Process-wide cumulative CUDA weight residency evictions.",
        stats.evictions,
    );
    gauge(
        &mut output,
        "onnx_genai_cuda_weight_offload_content_resident_bytes",
        "Canonical weight content bytes currently held by CUDA residency caches.",
        stats.content_resident_bytes,
    );
    gauge(
        &mut output,
        "onnx_genai_cuda_weight_offload_physical_owned_bytes",
        "Authority-owned physical bytes across live CUDA VMM handle pools.",
        stats.physical_owned_bytes,
    );
    gauge(
        &mut output,
        "onnx_genai_cuda_weight_offload_mapped_physical_bytes",
        "Physical bytes mapped and attributed to CUDA weight residency zones.",
        stats.mapped_physical_bytes,
    );
    output.push_str("# HELP onnx_genai_cuda_weight_offload_hit_rate Process-wide CUDA weight residency hit ratio derived from cumulative hits and page-ins.\n");
    output.push_str("# TYPE onnx_genai_cuda_weight_offload_hit_rate gauge\n");
    let lookups = stats.hits + stats.page_ins;
    let hit_rate = if lookups == 0 {
        0.0
    } else {
        stats.hits as f64 / lookups as f64
    };
    writeln!(output, "onnx_genai_cuda_weight_offload_hit_rate {hit_rate}").expect("String write");
    output
}

#[cfg(not(all(feature = "metrics", feature = "native-cuda")))]
fn encode_weight_offload() -> String {
    String::new()
}

/// Name of the gauge that says whether the governor family below is real.
///
/// Exported so the handler can emit the `0` case without duplicating the
/// string; a typo here would split one series into two and neither would
/// alarm.
#[cfg(feature = "metrics")]
pub(crate) const RESOURCE_GOVERNOR_AVAILABLE: &str = "onnx_genai_resource_governor_available";

#[cfg(feature = "metrics")]
const RESOURCE_GOVERNOR_AVAILABLE_HELP: &str = "1 when the resource governor was readable for this scrape and the \
     onnx_genai_vram_*/host_ram_*/kv_* gauges below are present; 0 when it was \
     not and those gauges are absent for that reason.";

/// Emits the availability marker alone, for scrapes where the governor could
/// not be read.
///
/// A Prometheus series that simply STOPS is indistinguishable from a scrape
/// gap, a restart, or a relabel: the graph just ends. Omitting the governor
/// family on error therefore hides the failure in the one shape an operator
/// reads as "nothing to see". This publishes the absence instead.
#[cfg(feature = "metrics")]
pub(crate) fn encode_resource_governor_unavailable() -> String {
    let mut output = String::new();
    gauge(
        &mut output,
        RESOURCE_GOVERNOR_AVAILABLE,
        RESOURCE_GOVERNOR_AVAILABLE_HELP,
        0,
    );
    output
}

const KV_PAGING_APPLICABLE: &str = "onnx_genai_kv_paging_applicable";
const KV_PAGING_APPLICABLE_HELP: &str = concat!(
    "Whether paged KV is the mechanism the decoder uses: ",
    "1 applicable, 0 not applicable, -1 not yet determined. ",
    "Every other kv_pages_* series is a truthful reading of a real pool even ",
    "when this is 0, so a non-zero capacity is not evidence the pool is in use."
);

/// Publish the KV page pool's aggregate state.
///
/// `applicable` is emitted first and deliberately, because the counters below
/// it are honest reads of a real structure even on a model that never consults
/// the pool. `onnx_genai_kv_pages_capacity` is *non-zero* on a continuous-batching
/// model, so a consumer that charted utilisation without checking applicability
/// would be charting a mechanism that is not running -- and nothing in the
/// numbers themselves would reveal it.
///
/// The pending state is `-1` rather than an omitted series: the decode path is
/// chosen asynchronously at startup, so a scrape can genuinely arrive before
/// the answer exists, and reporting `0` then would state "not applicable" with
/// full confidence about a pool that may well be paged.
#[cfg(feature = "metrics")]
pub(crate) fn encode_kv_telemetry(
    applicability: onnx_genai_engine::Applicability,
    snapshot: &onnx_genai_engine::KvTelemetrySnapshot,
) -> String {
    use onnx_genai_engine::Applicability;

    let mut output = String::new();
    let applicable = match applicability {
        Applicability::Applicable => 1i64,
        Applicability::NotApplicable => 0,
        Applicability::Unknown => -1,
    };
    signed_gauge(
        &mut output,
        KV_PAGING_APPLICABLE,
        KV_PAGING_APPLICABLE_HELP,
        applicable,
    );
    gauge(
        &mut output,
        "onnx_genai_kv_pages_in_use",
        "KV pages with at least one reference, on any tier. May exceed capacity: eviction demotes a page to the cold tier without dropping its reference.",
        snapshot.pages_in_use as u64,
    );
    gauge(
        &mut output,
        "onnx_genai_kv_pages_shared",
        "KV pages with more than one reference, i.e. shared by copy-on-write or prefix reuse.",
        snapshot.pages_shared as u64,
    );
    gauge(
        &mut output,
        "onnx_genai_kv_pages_capacity",
        "Hot-tier live KV page capacity.",
        snapshot.hot_capacity as u64,
    );
    gauge(
        &mut output,
        "onnx_genai_kv_page_size_tokens",
        "Token slots per KV page.",
        snapshot.page_size as u64,
    );
    counter(
        &mut output,
        "onnx_genai_kv_page_allocations_total",
        "KV pages handed out.",
        snapshot.allocations,
    );
    counter(
        &mut output,
        "onnx_genai_kv_page_allocation_failures_total",
        "KV page allocations that found no page. The honest signal that the pool is under real pressure.",
        snapshot.allocation_failures,
    );
    counter(
        &mut output,
        "onnx_genai_kv_page_frees_total",
        "KV pages returned to the free list.",
        snapshot.frees,
    );
    counter(
        &mut output,
        "onnx_genai_kv_hot_evictions_total",
        "KV pages demoted from the hot tier.",
        snapshot.hot_evictions,
    );
    counter(
        &mut output,
        "onnx_genai_kv_prefix_evictions_total",
        "KV pages dropped from the prefix cache.",
        snapshot.prefix_evictions,
    );
    output
}

#[cfg(feature = "metrics")]
pub(crate) fn encode_mapped_growth(metrics: &onnx_genai_engine::MappedGrowthMetrics) -> String {
    let values = [
        (
            "onnx_genai_mapped_growth_attempts_total",
            "Mapped growth grant attempts.",
            metrics.attempts,
            "counter",
        ),
        (
            "onnx_genai_mapped_growth_bytes_transferred_total",
            "Allowance bytes transferred by committed mapped growth grants.",
            metrics.bytes_transferred,
            "counter",
        ),
        (
            "onnx_genai_mapped_growth_failures_total",
            "Mapped growth grant preparation failures.",
            metrics.failures,
            "counter",
        ),
        (
            "onnx_genai_mapped_growth_rollbacks_total",
            "Mapped growth grants rolled back before commit.",
            metrics.rollbacks,
            "counter",
        ),
        (
            "onnx_genai_mapped_weight_bytes",
            "Currently mapped reloadable weight bytes.",
            metrics.weight_mapped,
            "gauge",
        ),
        (
            "onnx_genai_mapped_kv_bytes",
            "Currently mapped native KV bytes.",
            metrics.kv_mapped,
            "gauge",
        ),
        (
            "onnx_genai_mapped_workspace_bytes",
            "Currently mapped governed workspace bytes.",
            metrics.workspace_mapped,
            "gauge",
        ),
        (
            "onnx_genai_mapped_total_owned_bytes",
            "Total physical bytes owned by mapped-growth authorities.",
            metrics.total_owned,
            "gauge",
        ),
        (
            "onnx_genai_mapped_growth_registered_holders",
            "Live reclaimable mapped holders.",
            metrics.live_holders,
            "gauge",
        ),
    ];
    let mut output = String::new();
    for (name, help, value, metric_type) in values {
        let _ = writeln!(output, "# HELP {name} {help}");
        let _ = writeln!(output, "# TYPE {name} {metric_type}");
        let _ = writeln!(output, "{name} {value}");
    }
    output
}

#[cfg(feature = "metrics")]
pub(crate) fn encode_resource_governor(snapshot: &GovernorSnapshot) -> String {
    let mut output = String::new();
    gauge(
        &mut output,
        RESOURCE_GOVERNOR_AVAILABLE,
        RESOURCE_GOVERNOR_AVAILABLE_HELP,
        1,
    );
    gauge(
        &mut output,
        "onnx_genai_vram_used_bytes",
        "VRAM currently used.",
        snapshot.vram.used,
    );
    gauge(
        &mut output,
        "onnx_genai_vram_limit_bytes",
        "Configured VRAM ceiling.",
        snapshot.vram.limit,
    );
    gauge(
        &mut output,
        "onnx_genai_vram_headroom_bytes",
        "VRAM bytes below the configured ceiling.",
        snapshot.vram.headroom,
    );
    gauge(
        &mut output,
        "onnx_genai_host_ram_used_bytes",
        "Host RAM currently used.",
        snapshot.host_ram.used,
    );
    gauge(
        &mut output,
        "onnx_genai_host_ram_limit_bytes",
        "Configured host RAM ceiling.",
        snapshot.host_ram.limit,
    );
    gauge(
        &mut output,
        "onnx_genai_host_ram_headroom_bytes",
        "Host RAM bytes below the configured ceiling.",
        snapshot.host_ram.headroom,
    );
    if let Some(disk) = snapshot.disk_spill {
        gauge(
            &mut output,
            "onnx_genai_disk_spill_used_bytes",
            "Disk spill currently used.",
            disk.used,
        );
        gauge(
            &mut output,
            "onnx_genai_disk_spill_limit_bytes",
            "Configured disk spill ceiling.",
            disk.limit,
        );
        gauge(
            &mut output,
            "onnx_genai_disk_spill_headroom_bytes",
            "Disk spill bytes below the configured ceiling.",
            disk.headroom,
        );
    }
    gauge(
        &mut output,
        "onnx_genai_kv_budget_bytes",
        "Derived VRAM budget available to KV cache.",
        snapshot.derived_budget.kv_bytes,
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

/// A gauge that can go negative, for tri-state values where the third state is
/// "not yet known" and must not be reported as either of the other two.
#[cfg(feature = "metrics")]
fn signed_gauge(output: &mut String, name: &str, help: &str, value: i64) {
    writeln!(output, "# HELP {name} {help}").expect("String write");
    writeln!(output, "# TYPE {name} gauge").expect("String write");
    writeln!(output, "{name} {value}").expect("String write");
}

#[cfg(test)]
mod request_metric_tests {
    use super::*;

    #[test]
    fn overload_response_increments_rejections_exactly_once() {
        let registry = Registry::new();

        registry.request_finished("/v1/chat/completions", StatusCode::TOO_MANY_REQUESTS);

        assert_eq!(registry.rejections.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn unrelated_error_does_not_increment_rejections() {
        let registry = Registry::new();

        registry.request_finished("/v1/chat/completions", StatusCode::INTERNAL_SERVER_ERROR);

        assert_eq!(registry.rejections.load(Ordering::Relaxed), 0);
    }
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

#[cfg(test)]
mod batch_occupancy_tests {
    use super::*;

    fn snapshot_with(steps: u64, rows: u64, peak: u64, max_batch: u64) -> MetricsSnapshot {
        MetricsSnapshot {
            active_sessions: 0,
            pending_requests: 0,
            total_tokens: 0,
            current_batch_size: 0,
            codecoded_forward_steps: steps,
            codecoded_rows: rows,
            codecoded_rows_peak: peak,
            codecoded_max_batch: max_batch,
            prefix_cache_hits: 0,
            prefix_cache_lookups: 0,
            rejections: 0,
        }
    }

    #[test]
    fn utilization_is_unknown_before_any_batched_forward() {
        // A backend that never batches must not report a fabricated 0.5.
        assert_eq!(snapshot_with(0, 0, 0, 8).batch_utilization(), None);
        assert_eq!(snapshot_with(4, 4, 1, 0).batch_utilization(), None);
    }

    #[test]
    fn utilization_is_mean_rows_over_the_physical_batch() {
        // 10 forwards carrying 40 rows on a batch of 8 -> mean 4 rows -> 0.5.
        assert_eq!(snapshot_with(10, 40, 8, 8).batch_utilization(), Some(0.5));
        // Serialized decode: every forward carried one row of eight.
        assert_eq!(snapshot_with(10, 10, 1, 8).batch_utilization(), Some(0.125));
        // Fully packed.
        assert_eq!(snapshot_with(10, 80, 8, 8).batch_utilization(), Some(1.0));
    }

    #[test]
    fn observing_forwards_accumulates_totals_and_keeps_the_peak() {
        // Cumulative deltas are added, but the peak is a maximum: averaging it
        // away would hide that a forward ever carried the whole batch.
        let before = snapshot();
        batch_forwards_observed(3, 9, 4, 8);
        batch_forwards_observed(2, 2, 1, 8);
        let after = snapshot();
        assert_eq!(
            after.codecoded_forward_steps - before.codecoded_forward_steps,
            5
        );
        assert_eq!(after.codecoded_rows - before.codecoded_rows, 11);
        assert!(after.codecoded_rows_peak >= 4);
        assert!(after.codecoded_max_batch >= 8);
    }

    #[test]
    fn a_zero_step_delta_records_nothing() {
        let before = snapshot();
        batch_forwards_observed(0, 0, 0, 8);
        let after = snapshot();
        assert_eq!(
            after.codecoded_forward_steps,
            before.codecoded_forward_steps
        );
    }
}

#[cfg(all(test, feature = "metrics"))]
mod tests {
    use super::*;
    use onnx_genai_engine::{Applicability, KvTelemetrySnapshot};

    fn sample() -> KvTelemetrySnapshot {
        KvTelemetrySnapshot {
            pages_in_use: 12,
            pages_shared: 3,
            hot_capacity: 64,
            page_size: 16,
            allocations: 100,
            allocation_failures: 2,
            frees: 88,
            hot_evictions: 5,
            prefix_evictions: 1,
        }
    }

    #[test]
    fn every_snapshot_field_reaches_the_exposition() {
        // A field added to the snapshot but never emitted would be invisible
        // with no symptom at the point of the omission, so pin all of them.
        let output = encode_kv_telemetry(Applicability::Applicable, &sample());
        for expected in [
            "onnx_genai_kv_pages_in_use 12",
            "onnx_genai_kv_pages_shared 3",
            "onnx_genai_kv_pages_capacity 64",
            "onnx_genai_kv_page_size_tokens 16",
            "onnx_genai_kv_page_allocations_total 100",
            "onnx_genai_kv_page_allocation_failures_total 2",
            "onnx_genai_kv_page_frees_total 88",
            "onnx_genai_kv_hot_evictions_total 5",
            "onnx_genai_kv_prefix_evictions_total 1",
        ] {
            assert!(
                output.contains(expected),
                "missing {expected} in:\n{output}"
            );
        }
    }

    /// Every emitted metric name uses the documented `onnx_genai_` prefix.
    ///
    /// # Why the other tests do not cover this
    ///
    /// The exposition tests pin metric names as **literals**, so they agree
    /// with whatever the emitter spells and stay green under a prefix that no
    /// documentation mentions. The whole memory-governor and KV-paging family
    /// was emitted as `onnxgenai_` -- no underscore -- for long enough that
    /// nobody could scrape it: the numbers were real, correct, and exported
    /// under a name that appears zero times in `docs/`.
    ///
    /// This asserts the *convention* rather than a list, so a family added
    /// tomorrow with the wrong prefix fails here instead of quietly producing
    /// an empty Grafana panel.
    #[test]
    fn every_emitted_metric_name_uses_the_documented_prefix() {
        let mut exposition = String::new();
        exposition.push_str(&encode_prometheus());
        exposition.push_str(&encode_kv_telemetry(Applicability::Applicable, &sample()));
        exposition.push_str(&encode_resource_governor_unavailable());

        let mut offenders = Vec::new();
        for line in exposition.lines() {
            // `# HELP <name> ...` / `# TYPE <name> ...` carry the name in the
            // third field; a sample line carries it first, possibly followed
            // by `{labels}`.
            let name = if let Some(rest) = line
                .strip_prefix("# HELP ")
                .or_else(|| line.strip_prefix("# TYPE "))
            {
                rest.split_whitespace().next()
            } else if line.starts_with('#') || line.trim().is_empty() {
                None
            } else {
                line.split_whitespace()
                    .next()
                    .map(|token| token.split('{').next().unwrap_or(token))
            };
            if let Some(name) = name
                && !name.starts_with("onnx_genai_")
            {
                offenders.push(name.to_string());
            }
        }

        assert!(
            offenders.is_empty(),
            "these metric names do not use the documented `onnx_genai_` prefix, \
             so nothing scraping the documented names will find them: {offenders:?}"
        );
    }

    #[test]
    fn prometheus_does_not_export_non_aggregate_weight_offload_gauges() {
        let output = encode_prometheus();
        for stale_name in [
            "onnx_genai_cuda_weight_offload_budget_bytes",
            "onnx_genai_cuda_weight_offload_peak_resident_bytes",
        ] {
            assert!(
                !output.contains(stale_name),
                "{stale_name} is per-residency state, not process truth, and must not be exported"
            );
        }
    }

    #[cfg(feature = "native-cuda")]
    #[test]
    fn cuda_weight_offload_metrics_are_process_activity_only() {
        let output = encode_weight_offload();
        for expected in [
            "# TYPE onnx_genai_cuda_weight_offload_page_ins_total counter",
            "# TYPE onnx_genai_cuda_weight_offload_hits_total counter",
            "# TYPE onnx_genai_cuda_weight_offload_evictions_total counter",
            "# TYPE onnx_genai_cuda_weight_offload_hit_rate gauge",
        ] {
            assert!(
                output.contains(expected),
                "missing expected process-wide offload metric {expected} in:\n{output}"
            );
        }
        for stale_name in [
            "onnx_genai_cuda_weight_offload_budget_bytes",
            "onnx_genai_cuda_weight_offload_peak_resident_bytes",
        ] {
            assert!(
                !output.contains(stale_name),
                "{stale_name} is per-residency state, not process truth, and must not be exported"
            );
        }
    }

    #[test]
    fn applicability_is_tri_state_and_never_omitted() {
        // The pending state must be reported, not dropped. A series that simply
        // stops is read as a scrape gap; and reporting 0 while the decode path
        // is still being chosen would assert "not applicable" about a pool that
        // may well be paged.
        for (state, expected) in [
            (
                Applicability::Applicable,
                "onnx_genai_kv_paging_applicable 1",
            ),
            (
                Applicability::NotApplicable,
                "onnx_genai_kv_paging_applicable 0",
            ),
            (Applicability::Unknown, "onnx_genai_kv_paging_applicable -1"),
        ] {
            let output = encode_kv_telemetry(state, &sample());
            assert!(
                output.contains(expected),
                "expected {expected} for {state:?} in:\n{output}"
            );
        }
    }

    #[test]
    fn counters_and_gauges_are_typed_correctly() {
        // Cumulative totals must be counters, live readings gauges: Prometheus
        // rate() on a gauge, or a counter reset alarm on a gauge, are both
        // silent misreadings rather than errors.
        let output = encode_kv_telemetry(Applicability::Applicable, &sample());
        assert!(output.contains("# TYPE onnx_genai_kv_page_allocations_total counter"));
        assert!(output.contains("# TYPE onnx_genai_kv_page_allocation_failures_total counter"));
        assert!(output.contains("# TYPE onnx_genai_kv_pages_in_use gauge"));
        assert!(output.contains("# TYPE onnx_genai_kv_paging_applicable gauge"));
    }

    #[test]
    fn counters_are_still_published_when_not_applicable() {
        // The numbers stay truthful readings of a real pool; what changes is
        // that the applicability flag tells a consumer not to chart them as a
        // live mechanism. Dropping them instead would look like a scrape gap.
        let output = encode_kv_telemetry(Applicability::NotApplicable, &sample());
        assert!(output.contains("onnx_genai_kv_pages_capacity 64"));
        assert!(output.contains("onnx_genai_kv_paging_applicable 0"));
    }
}
