//! The [`Tracer`] collector and its RAII [`SpanGuard`].
//!
//! A [`Tracer`] is a cheap, cloneable handle over a shared, thread-safe event
//! sink. Clone it freely (it is `Arc`-backed) and hand copies to worker
//! threads; every clone records into the same trace. Timestamps are derived
//! from a single monotonic epoch fixed when the tracer is created, so events
//! from different threads share one timeline.
//!
//! When a tracer is **disabled**, span creation and every `complete`/`instant`
//! entry point is a single relaxed atomic load plus an early return — no
//! allocation, no lock, no clock read. Production code can leave a disabled
//! tracer wired in at negligible cost.

use crate::args::Args;
use crate::error::{Result, TracerError};
use crate::event::{Event, Phase};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

/// Default cap on retained events so an unbounded run cannot exhaust memory.
pub const DEFAULT_MAX_EVENTS: usize = 1_000_000;

struct Inner {
    enabled: AtomicBool,
    epoch: Instant,
    pid: u64,
    max_events: usize,
    events: Mutex<Vec<Event>>,
    /// Maps each OS thread to a small, stable, per-tracer numeric lane id.
    tids: Mutex<HashMap<ThreadId, u64>>,
}

/// A thread-safe collector of Chrome Trace events.
///
/// See the [module docs](crate::tracer) for the enabled/disabled cost model.
#[derive(Clone)]
pub struct Tracer {
    inner: Arc<Inner>,
}

impl Tracer {
    /// Create a new, **enabled** tracer whose epoch is now.
    #[must_use]
    pub fn new() -> Self {
        Self::with_options(true, std::process::id() as u64, DEFAULT_MAX_EVENTS)
    }

    /// Create a **disabled** tracer. All recording is a no-op fast path until
    /// [`set_enabled(true)`](Tracer::set_enabled) is called.
    #[must_use]
    pub fn disabled() -> Self {
        Self::with_options(false, std::process::id() as u64, DEFAULT_MAX_EVENTS)
    }

    /// Create a tracer with an explicit enabled flag, process id, and event
    /// cap. Prefer [`Tracer::new`] unless you need to override these.
    #[must_use]
    pub fn with_options(enabled: bool, pid: u64, max_events: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                enabled: AtomicBool::new(enabled),
                epoch: Instant::now(),
                pid,
                max_events,
                events: Mutex::new(Vec::new()),
                tids: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Whether recording is currently enabled.
    #[must_use]
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.inner.enabled.load(Ordering::Relaxed)
    }

    /// Enable or disable recording. Cheap and safe to toggle from any thread.
    pub fn set_enabled(&self, enabled: bool) {
        self.inner.enabled.store(enabled, Ordering::Relaxed);
    }

    /// The process id stamped onto every event from this tracer.
    #[must_use]
    pub fn pid(&self) -> u64 {
        self.inner.pid
    }

    /// The stable lane id assigned to the current OS thread by this tracer.
    ///
    /// The first thread to touch a given tracer gets `0`, the next `1`, and so
    /// on; repeat calls from the same thread return the same id.
    #[must_use]
    pub fn current_tid(&self) -> u64 {
        let id = std::thread::current().id();
        let mut tids = self
            .inner
            .tids
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let next = tids.len() as u64;
        *tids.entry(id).or_insert(next)
    }

    /// Microseconds elapsed from the tracer epoch to `at`.
    fn ts_of(&self, at: Instant) -> u64 {
        at.saturating_duration_since(self.inner.epoch).as_micros() as u64
    }

    /// Append an event, honouring the retention cap. Assumes the caller has
    /// already checked [`is_enabled`](Tracer::is_enabled) where a fast path
    /// matters.
    fn push(&self, event: Event) {
        let mut events = self
            .inner
            .events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if events.len() >= self.inner.max_events {
            return;
        }
        events.push(event);
    }

    /// Record a complete (`"X"`) event that started at `start` and lasted
    /// `dur`. No-op when the tracer is disabled.
    ///
    /// Use this when you already have explicit timing. For scoped timing,
    /// prefer [`span`](Tracer::span), which measures duration for you.
    pub fn complete(
        &self,
        name: impl Into<String>,
        cat: impl Into<String>,
        start: Instant,
        dur: Duration,
        args: Option<Args>,
    ) {
        if !self.is_enabled() {
            return;
        }
        let tid = self.current_tid();
        self.push(Event {
            name: name.into(),
            cat: cat.into(),
            ph: Phase::Complete,
            ts: self.ts_of(start),
            dur: Some(dur.as_micros() as u64),
            pid: self.inner.pid,
            tid,
            scope: None,
            args: args.map(Args::into_value),
        });
    }

    /// Record an instant (`"i"`) event at the current time with thread scope.
    /// No-op when the tracer is disabled.
    pub fn instant(&self, name: impl Into<String>, cat: impl Into<String>, args: Option<Args>) {
        if !self.is_enabled() {
            return;
        }
        let now = Instant::now();
        let tid = self.current_tid();
        self.push(Event {
            name: name.into(),
            cat: cat.into(),
            ph: Phase::Instant,
            ts: self.ts_of(now),
            dur: None,
            pid: self.inner.pid,
            tid,
            scope: Some('t'),
            args: args.map(Args::into_value),
        });
    }

    /// Open a scoped span. The returned [`SpanGuard`] records a complete event
    /// covering its lifetime when it is dropped (or [`finish`](SpanGuard::finish)ed).
    ///
    /// When the tracer is disabled the guard is inert: it holds no owned
    /// strings and does nothing on drop.
    pub fn span(&self, name: impl Into<String>, cat: impl Into<String>) -> SpanGuard {
        if !self.is_enabled() {
            return SpanGuard::inert();
        }
        SpanGuard {
            state: Some(SpanState {
                tracer: self.clone(),
                name: name.into(),
                cat: cat.into(),
                start: Instant::now(),
                args: None,
            }),
        }
    }

    /// Emit a `process_name` metadata (`"M"`) event naming this process lane.
    /// No-op when disabled.
    pub fn set_process_name(&self, name: impl Into<String>) {
        self.metadata("process_name", name.into());
    }

    /// Emit a `thread_name` metadata (`"M"`) event naming the current thread's
    /// lane. No-op when disabled.
    pub fn set_thread_name(&self, name: impl Into<String>) {
        self.metadata("thread_name", name.into());
    }

    fn metadata(&self, kind: &str, name: String) {
        if !self.is_enabled() {
            return;
        }
        let tid = self.current_tid();
        let args = Args::new().with("name", name);
        self.push(Event {
            name: kind.to_string(),
            cat: "__metadata".to_string(),
            ph: Phase::Metadata,
            ts: 0,
            dur: None,
            pid: self.inner.pid,
            tid,
            scope: None,
            args: Some(args.into_value()),
        });
    }

    /// Number of events currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner
            .events
            .lock()
            .map(|events| events.len())
            .unwrap_or_else(|poisoned| poisoned.into_inner().len())
    }

    /// Whether no events have been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop all recorded events. The epoch and thread-lane assignments are
    /// preserved so ids stay stable across a clear.
    pub fn clear(&self) {
        self.inner
            .events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    /// A snapshot copy of the recorded events, in insertion order.
    #[must_use]
    pub fn events(&self) -> Vec<Event> {
        self.inner
            .events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Serialize the recorded events as a compact Chrome Trace JSON **array**,
    /// loadable in <https://ui.perfetto.dev> or `chrome://tracing`.
    #[must_use]
    pub fn to_chrome_json(&self) -> String {
        let events = self.events();
        // The built-in `Event` model always serializes successfully.
        serde_json::to_string(&events).expect("Event array serialization is infallible")
    }

    /// Like [`to_chrome_json`](Tracer::to_chrome_json) but pretty-printed.
    #[must_use]
    pub fn to_chrome_json_pretty(&self) -> String {
        let events = self.events();
        serde_json::to_string_pretty(&events).expect("Event array serialization is infallible")
    }

    /// Write the recorded events to `path` as a Chrome Trace JSON array.
    ///
    /// # Errors
    ///
    /// Returns [`TracerError::Write`] if the file cannot be created or written,
    /// with the offending path and underlying I/O cause included in the
    /// message.
    pub fn write_chrome_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let json = self.to_chrome_json();
        std::fs::write(path, json).map_err(|source| TracerError::Write {
            path: path.to_path_buf(),
            source,
        })
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Tracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tracer")
            .field("enabled", &self.is_enabled())
            .field("pid", &self.inner.pid)
            .field("events", &self.len())
            .finish()
    }
}

/// The live state of an enabled span. Absent for inert (disabled) guards.
struct SpanState {
    tracer: Tracer,
    name: String,
    cat: String,
    start: Instant,
    args: Option<Args>,
}

/// An RAII guard that records a complete event covering its lifetime.
///
/// Created by [`Tracer::span`]. The event is recorded when the guard is
/// dropped, or eagerly via [`finish`](SpanGuard::finish). Attach metadata
/// before the guard ends with [`set_args`](SpanGuard::set_args) or the
/// chaining [`with_args`](SpanGuard::with_args).
///
/// An inert guard (produced when the tracer was disabled) owns nothing and
/// does nothing on drop.
#[must_use = "a SpanGuard records its span only while it is alive; drop it at the end of the region to time"]
pub struct SpanGuard {
    state: Option<SpanState>,
}

impl SpanGuard {
    /// An inert guard that records nothing.
    fn inert() -> Self {
        Self { state: None }
    }

    /// Attach args to the span, consuming and returning the guard for chaining.
    pub fn with_args(mut self, args: Args) -> Self {
        if let Some(state) = self.state.as_mut() {
            state.args = Some(args);
        }
        self
    }

    /// Attach or replace the span's args in place.
    pub fn set_args(&mut self, args: Args) {
        if let Some(state) = self.state.as_mut() {
            state.args = Some(args);
        }
    }

    /// The tracer epoch-relative start timestamp of this span in microseconds,
    /// or `None` for an inert guard.
    #[must_use]
    pub fn start_ts(&self) -> Option<u64> {
        self.state
            .as_ref()
            .map(|state| state.tracer.ts_of(state.start))
    }

    /// Finish the span now, recording its event immediately instead of on drop.
    pub fn finish(mut self) {
        self.record();
    }

    fn record(&mut self) {
        if let Some(state) = self.state.take() {
            let dur = state.start.elapsed();
            state
                .tracer
                .complete(state.name, state.cat, state.start, dur, state.args);
        }
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        self.record();
    }
}
