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

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use onnx_genai_runtime_config::runtime_config;

/// How much detail a timeline records. Re-exported so callers can name a level
/// without depending on the tracer crate directly.
pub use onnx_runtime_tracer::TraceVerbosity;

/// Returns whether profiling is enabled, reading `ONNX_GENAI_PROFILE` once.
pub fn enabled() -> bool {
    runtime_config().profile
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
/// A destination set after startup, taking precedence over the environment.
///
/// The environment is read once into a `OnceLock`, which is right for a
/// one-shot run but wrong for an interactive session: a user who decides
/// mid-conversation that they want a timeline cannot restart the process to
/// get one. This is the override that lets them ask for it in place.
/// `None` = no override (use the environment); `Some(None)` = explicitly off;
/// `Some(Some(path))` = write there.
static RUNTIME_TRACE_PATH: std::sync::RwLock<Option<Option<std::path::PathBuf>>> =
    std::sync::RwLock::new(None);
/// Fast path for [`trace_path`] so an untraced run never takes the lock.
static RUNTIME_TRACE_SET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Direct the timeline to `path` from now on.
///
/// Three states, not two. `Some(path)` writes there; `None` turns the timeline
/// off *even when the environment asked for one*, because a session that says
/// "off" means off and would otherwise keep writing the startup destination;
/// [`clear_trace_override`] is how you go back to the environment's setting.
pub fn set_trace_path(path: Option<std::path::PathBuf>) {
    // Written under the same lock that guards the value, so a concurrent
    // setter cannot leave the flag describing a different write's value.
    let mut guard = RUNTIME_TRACE_PATH
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = Some(path);
    RUNTIME_TRACE_SET.store(true, std::sync::atomic::Ordering::Release);
}

/// Forget any runtime destination, deferring to the environment again.
pub fn clear_trace_override() {
    let mut guard = RUNTIME_TRACE_PATH
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = None;
    RUNTIME_TRACE_SET.store(false, std::sync::atomic::Ordering::Release);
}

/// Where the timeline will be written, if anywhere.
#[must_use]
pub fn trace_destination() -> Option<std::path::PathBuf> {
    if RUNTIME_TRACE_SET.load(std::sync::atomic::Ordering::Acquire)
        && let Some(override_value) = RUNTIME_TRACE_PATH
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    {
        return override_value;
    }
    runtime_config().trace.clone()
}

fn trace_path() -> Option<std::path::PathBuf> {
    trace_destination()
}

/// Whether timeline tracing is enabled (a non-empty `ONNX_GENAI_TRACE`).
pub fn tracing_enabled() -> bool {
    trace_path().is_some()
}

/// Convert a monotonic [`Instant`] to the shared absolute trace axis.
///
/// These spans share a timeline with the ones the runtime and its execution
/// providers record through `onnx-runtime-tracer`, so they have to be stamped
/// against the same origin rather than a private one. This module used to fix
/// its own epoch on first use; that only ever agreed with the runtime's by
/// coincidence, because both happened to be created near process start.
fn absolute_us(at: Instant) -> u64 {
    let ago = Instant::now().saturating_duration_since(at).as_micros() as u64;
    onnx_runtime_tracer::absolute_now_us().saturating_sub(ago)
}

/// One recorded timeline event, rendered later as a Chrome `X` (complete) event.
struct TraceEvent {
    name: &'static str,
    tid: u64,
    ts_us: u64,
    dur_us: u64,
    /// Perfetto `args` for this event. Usually empty: only spans that carry
    /// something a reader cannot infer from the name populate it.
    args: Vec<(&'static str, serde_json::Value)>,
}

/// Bound on retained events so a very long run cannot grow memory without limit.
const MAX_TRACE_EVENTS: usize = 1_000_000;

fn trace_sink() -> &'static Mutex<Vec<TraceEvent>> {
    static SINK: OnceLock<Mutex<Vec<TraceEvent>>> = OnceLock::new();
    SINK.get_or_init(|| Mutex::new(Vec::new()))
}

/// A small, stable per-thread id for the trace's thread lanes.
///
/// Deliberately the tracer's allocator rather than one of our own. These
/// events share a document with the ones the runtime and its providers record,
/// and a Chrome trace groups lanes by `pid` then `tid`: numbering threads
/// separately put the same thread on two lanes and two threads on one.
fn thread_trace_id() -> u64 {
    onnx_runtime_tracer::thread_lane_id()
}

/// Record a single timeline event. No-op unless tracing is enabled.
fn record_trace(
    stage: &'static str,
    start: Instant,
    dur: Duration,
    args: Vec<(&'static str, serde_json::Value)>,
) {
    if !tracing_enabled() {
        return;
    }
    let ts_us = absolute_us(start);
    let event = TraceEvent {
        name: stage,
        tid: thread_trace_id(),
        ts_us,
        dur_us: dur.as_micros() as u64,
        args,
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
                let mut value = serde_json::json!({
                    "name": event.name,
                    "cat": category,
                    "ph": "X",
                    "ts": event.ts_us,
                    "dur": event.dur_us,
                    "pid": onnx_runtime_tracer::process_id(),
                    "tid": event.tid,
                });
                if !event.args.is_empty() {
                    let args: serde_json::Map<String, serde_json::Value> = event
                        .args
                        .iter()
                        .map(|(key, value)| ((*key).to_string(), value.clone()))
                        .collect();
                    value["args"] = serde_json::Value::Object(args);
                }
                value
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
    /// Metadata to attach to this span. An empty `Vec` does not allocate, so
    /// the untraced path is unaffected by this field existing.
    args: Vec<(&'static str, serde_json::Value)>,
    /// Where this span was opened; a compile-time `&'static`.
    location: &'static std::panic::Location<'static>,
}

impl Span {
    /// Start a span. Cheap and inert when neither profiling nor tracing is on.
    ///
    /// Records the source location that opened it. `#[track_caller]` resolves
    /// that at compile time, so it costs nothing at run time — unlike a real
    /// backtrace, which measures 5.1us unresolved and 26.7us symbolised
    /// against the ~0.3ns this takes.
    #[must_use]
    #[track_caller]
    pub fn new(stage: &'static str) -> Self {
        Self {
            stage,
            start: Instant::now(),
            aggregate: enabled(),
            trace: tracing_enabled(),
            args: Vec::new(),
            location: std::panic::Location::caller(),
        }
    }

    /// Attach a key/value to this span, to be emitted as a Perfetto `arg`.
    ///
    /// For facts a reader cannot recover from the span's name and timing — the
    /// token a decode step produced, say. Ignored unless timeline tracing is
    /// on, so `value` should be cheap to produce; build anything expensive
    /// behind [`Span::is_tracing`].
    pub fn set_arg(&mut self, key: &'static str, value: impl Into<serde_json::Value>) {
        if !self.trace {
            return;
        }
        self.args.push((key, value.into()));
    }

    /// Whether this span will actually be recorded to the timeline.
    ///
    /// Guard the construction of expensive argument values with this.
    #[must_use]
    pub fn is_tracing(&self) -> bool {
        self.trace
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
            let mut args = std::mem::take(&mut self.args);
            args.push((
                "source",
                serde_json::Value::from(format!(
                    "{}:{}",
                    self.location.file(),
                    self.location.line()
                )),
            ));
            record_trace(self.stage, self.start, elapsed, args);
        }
    }
}

/// Open a profiling [`Span`] for the given static stage name.
#[macro_export]
macro_rules! prof_span {
    ($stage:expr) => {
        $crate::profile::Span::new($stage)
    };
    ($stage:expr, $key:expr => $value:expr) => {{
        let mut span = $crate::profile::Span::new($stage);
        span.set_arg($key, $value);
        span
    }};
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
/// One stage's accumulated cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageSnapshot {
    pub stage: &'static str,
    pub total_ns: u128,
    pub calls: u64,
}

/// Every recorded stage, most expensive first.
///
/// The structured counterpart to [`report`], so callers that are not writing to
/// a terminal — an HTTP endpoint, a JSON report — do not have to parse a table
/// that exists for human eyes.
pub fn snapshot() -> Vec<StageSnapshot> {
    let Ok(reg) = registry().lock() else {
        return Vec::new();
    };
    let mut rows = reg
        .iter()
        .map(|(name, stat)| StageSnapshot {
            stage: name,
            total_ns: stat.total_ns,
            calls: stat.count,
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| std::cmp::Reverse(row.total_ns));
    rows
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_document_renders_event_args() {
        reset();
        trace_sink().lock().unwrap().push(TraceEvent {
            name: "diffusion.denoise_step",
            tid: 7,
            ts_us: 123,
            dur_us: 45,
            args: vec![("step", serde_json::json!(3_u64))],
        });

        let document = trace_document();
        let event = &document["traceEvents"][0];
        assert_eq!(event["cat"], "diffusion");
        assert_eq!(event["name"], "diffusion.denoise_step");
        assert_eq!(event["args"]["step"], 3);
        reset();
    }
}
