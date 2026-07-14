//! Env-gated per-stage decode profiler.
//!
//! Enabled with `ONNX_GENAI_PROFILE=1`. When disabled every entry point is a
//! single relaxed-atomic load plus an early return, so production paths pay no
//! measurable cost. When enabled, [`Span`] accumulates wall-clock nanoseconds
//! and a call count per named stage into a process-global registry that
//! [`report`] renders as a table.
//!
//! This exists to answer one question: for each generated token, how much wall
//! time is spent inside the ORT kernels (`session.run`) versus our own
//! orchestration (tensor binding, KV rotation, logits copy, sampling,
//! detokenization). See `docs/benchmarks` and the CPU profiling decision note.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Returns whether profiling is enabled, reading `ONNX_GENAI_PROFILE` once.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("ONNX_GENAI_PROFILE").ok().as_deref(),
            Some("1") | Some("true") | Some("yes")
        )
    })
}

#[derive(Default, Clone, Copy)]
struct StageStat {
    total_ns: u128,
    count: u64,
}

fn registry() -> &'static Mutex<BTreeMap<&'static str, StageStat>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<&'static str, StageStat>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Add a measured duration for `stage`. No-op unless profiling is enabled.
pub fn record(stage: &'static str, nanos: u128) {
    if !enabled() {
        return;
    }
    if let Ok(mut reg) = registry().lock() {
        let entry = reg.entry(stage).or_default();
        entry.total_ns += nanos;
        entry.count += 1;
    }
}

/// Path to write a Chrome Trace Event (Perfetto) timeline to, from
/// `ONNX_GENAI_TRACE`. When set, each [`Span`] emits one timestamped
/// `complete` event so the run can be opened in <https://ui.perfetto.dev>.
fn trace_path() -> Option<&'static str> {
    static PATH: OnceLock<Option<String>> = OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var("ONNX_GENAI_TRACE")
            .ok()
            .filter(|value| !value.is_empty())
    })
    .as_deref()
}

/// Whether timeline tracing is enabled (a non-empty `ONNX_GENAI_TRACE`).
pub fn tracing_enabled() -> bool {
    trace_path().is_some()
}

/// The common time origin for trace timestamps, fixed on first use so the
/// first recorded event starts near t=0 on the Perfetto timeline.
fn trace_epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// One recorded timeline event, rendered later as a Chrome `X` (complete) event.
struct TraceEvent {
    name: &'static str,
    tid: u64,
    ts_us: u64,
    dur_us: u64,
}

/// Bound on retained events so a very long run cannot grow memory without limit.
const MAX_TRACE_EVENTS: usize = 1_000_000;

fn trace_sink() -> &'static Mutex<Vec<TraceEvent>> {
    static SINK: OnceLock<Mutex<Vec<TraceEvent>>> = OnceLock::new();
    SINK.get_or_init(|| Mutex::new(Vec::new()))
}

/// A small, stable per-thread id for the trace's thread lanes.
fn thread_trace_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    thread_local! {
        static TID: u64 = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    TID.with(|id| *id)
}

/// Record a single timeline event. No-op unless tracing is enabled.
fn record_trace(stage: &'static str, start: Instant, dur: Duration) {
    if !tracing_enabled() {
        return;
    }
    let ts_us = start.saturating_duration_since(trace_epoch()).as_micros() as u64;
    let event = TraceEvent {
        name: stage,
        tid: thread_trace_id(),
        ts_us,
        dur_us: dur.as_micros() as u64,
    };
    if let Ok(mut sink) = trace_sink().lock() {
        if sink.len() >= MAX_TRACE_EVENTS {
            return;
        }
        sink.push(event);
    }
}

/// Build the accumulated timeline as a Chrome Trace Event Format (Perfetto)
/// JSON document, openable in <https://ui.perfetto.dev> or `chrome://tracing`.
///
/// The span category is the stage-name prefix before the first `.` (e.g.
/// `ort`, `engine`, `loop`), so Perfetto can colour and group lanes by
/// subsystem. All events share one pid; each OS thread that opened a span gets
/// its own tid lane. Returns an empty (but well-formed) `traceEvents` array
/// when no spans have been recorded — the profiler only fills the in-memory
/// sink while `ONNX_GENAI_TRACE` is set, so callers get an honest empty trace
/// rather than fabricated events.
///
/// The recorded events carry only stage names and timing — never session IDs,
/// prompt text, or other user data — so the document is safe to expose without
/// redaction.
#[must_use]
pub fn trace_document() -> serde_json::Value {
    let trace_events: Vec<serde_json::Value> = match trace_sink().lock() {
        Ok(events) => events
            .iter()
            .map(|event| {
                let category = event.name.split('.').next().unwrap_or(event.name);
                serde_json::json!({
                    "name": event.name,
                    "cat": category,
                    "ph": "X",
                    "ts": event.ts_us,
                    "dur": event.dur_us,
                    "pid": 1,
                    "tid": event.tid,
                })
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    serde_json::json!({
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
    })
}

/// Number of timeline events currently retained in the in-memory sink.
#[must_use]
pub fn trace_event_count() -> usize {
    trace_sink().lock().map(|sink| sink.len()).unwrap_or(0)
}

/// Write the accumulated timeline to the `ONNX_GENAI_TRACE` path as a Chrome
/// Trace Event Format (Perfetto) JSON document. No-op (returns `Ok(())`) when
/// tracing is disabled. See [`trace_document`] for the emitted schema.
pub fn write_trace() -> std::io::Result<()> {
    let Some(path) = trace_path() else {
        return Ok(());
    };
    std::fs::write(path, serde_json::to_vec(&trace_document())?)?;
    Ok(())
}

/// A scoped timer that records its lifetime to `stage` on drop.
pub struct Span {
    stage: &'static str,
    start: Instant,
    /// Whether aggregate profiling (`ONNX_GENAI_PROFILE`) is active.
    aggregate: bool,
    /// Whether timeline tracing (`ONNX_GENAI_TRACE`) is active.
    trace: bool,
}

impl Span {
    /// Start a span. Cheap and inert when neither profiling nor tracing is on.
    #[must_use]
    pub fn new(stage: &'static str) -> Self {
        Self {
            stage,
            start: Instant::now(),
            aggregate: enabled(),
            trace: tracing_enabled(),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if !self.aggregate && !self.trace {
            return;
        }
        let elapsed = self.start.elapsed();
        if self.aggregate {
            record(self.stage, elapsed.as_nanos());
        }
        if self.trace {
            record_trace(self.stage, self.start, elapsed);
        }
    }
}

/// Open a profiling [`Span`] for the given static stage name.
#[macro_export]
macro_rules! prof_span {
    ($stage:expr) => {
        $crate::profile::Span::new($stage)
    };
}

/// Clear all accumulated stage statistics and any recorded timeline events.
pub fn reset() {
    if let Ok(mut reg) = registry().lock() {
        reg.clear();
    }
    if let Ok(mut sink) = trace_sink().lock() {
        sink.clear();
    }
}

/// Render the accumulated per-stage statistics as a text table.
///
/// `tokens` scales the per-token column; pass the number of generated tokens.
pub fn report(tokens: u64) -> String {
    let reg = match registry().lock() {
        Ok(reg) => reg,
        Err(_) => return String::from("<profiler registry poisoned>"),
    };
    let mut rows: Vec<(&'static str, StageStat)> =
        reg.iter().map(|(name, stat)| (*name, *stat)).collect();
    rows.sort_by_key(|row| std::cmp::Reverse(row.1.total_ns));

    let tokens = tokens.max(1);
    let mut out = String::new();
    out.push_str(&format!(
        "{:<26} {:>12} {:>10} {:>14} {:>12}\n",
        "stage", "total_ms", "calls", "us/call", "us/token"
    ));
    out.push_str(&format!("{}\n", "-".repeat(78)));
    for (name, stat) in &rows {
        let total_ms = stat.total_ns as f64 / 1_000_000.0;
        let us_per_call = if stat.count > 0 {
            (stat.total_ns as f64 / 1_000.0) / stat.count as f64
        } else {
            0.0
        };
        let us_per_token = (stat.total_ns as f64 / 1_000.0) / tokens as f64;
        out.push_str(&format!(
            "{:<26} {:>12.3} {:>10} {:>14.2} {:>12.2}\n",
            name, total_ms, stat.count, us_per_call, us_per_token
        ));
    }
    out
}
