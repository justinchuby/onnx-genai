//! The shared [`TraceContext`] and its RAII [`SpanGuard`] (§48.3 / §48.5).
//!
//! A [`TraceContext`] is the single instrumentation handle both the runtime and
//! genai layers hold. It bundles a shared [`TraceClock`], a
//! [`TraceSessionId`], the output [`collector`](TraceCollector), the default
//! [`TraceFormat`], and a [`TraceVerbosity`]. It is cheap to
//! [`clone`](Clone) (everything is behind an [`Arc`]) so the *same* context —
//! and therefore the *same* timeline and sink — can be handed to multiple
//! layers (§48.5).
//!
//! ## Enabled / disabled cost
//!
//! A context carries an atomic enable flag. When disabled — as with
//! [`TraceContext::noop`] — every entry point is a single relaxed atomic load
//! followed by an early return: **no clock read, no allocation, no lock, no
//! collector call**. In particular the metadata helpers check the flag
//! *before* converting their `impl Into<String>` arguments, so a disabled
//! context never allocates a name string. Production code can leave a disabled
//! context wired in at negligible cost and flip it on only when profiling.

use crate::args::Args;
use crate::clock::{TraceClock, TraceSessionId};
use crate::collector::{MemoryCollector, NoopCollector, TraceCollector};
use crate::error::Result;
use crate::event::{TraceEvent, TracePhase};
use crate::format::{TraceFormat, TraceVerbosity};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

thread_local! {
    static ACTIVE_SPANS: RefCell<Vec<Weak<Mutex<Args>>>> = const { RefCell::new(Vec::new()) };
}

/// Merge metadata into the innermost active span on the current thread.
///
/// Kernel implementations use this to enrich the executor-created operation
/// span without needing a tracing handle in the kernel ABI. This is a no-op
/// when no enabled span is active.
pub fn annotate_current_span(args: Args) {
    ACTIVE_SPANS.with(|spans| {
        let mut spans = spans.borrow_mut();
        loop {
            match spans.last().and_then(Weak::upgrade) {
                Some(active) => {
                    active
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .merge(args);
                    break;
                }
                None if !spans.is_empty() => {
                    spans.pop();
                }
                None => break,
            }
        }
    });
}

struct Inner {
    enabled: AtomicBool,
    clock: Arc<TraceClock>,
    session_id: TraceSessionId,
    collector: Arc<dyn TraceCollector>,
    format: TraceFormat,
    verbosity: TraceVerbosity,
    pid: u64,
    /// Maps each OS thread to a small, stable, per-context numeric lane id.
    tids: Mutex<HashMap<ThreadId, u64>>,
}

/// The shared tracing context (§48.3).
///
/// See the [module docs](crate::context) for the clone semantics and the
/// enabled/disabled cost model.
#[derive(Clone)]
pub struct TraceContext {
    inner: Arc<Inner>,
}

impl TraceContext {
    /// Build an **enabled** context around an explicit collector and format.
    ///
    /// Share `collector` (via [`Arc`]) if you also need typed access to it —
    /// e.g. keep an `Arc<MemoryCollector>` to read events back while also
    /// passing it here as `Arc<dyn TraceCollector>`.
    #[must_use]
    pub fn with_collector(collector: Arc<dyn TraceCollector>, format: TraceFormat) -> Self {
        Self {
            inner: Arc::new(Inner {
                enabled: AtomicBool::new(true),
                clock: Arc::new(TraceClock::new()),
                session_id: TraceSessionId::next(),
                collector,
                format,
                verbosity: TraceVerbosity::default(),
                pid: std::process::id() as u64,
                tids: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// A **disabled**, zero-overhead context backed by a [`NoopCollector`]
    /// (§48.3). Nothing is recorded until [`set_enabled(true)`](TraceContext::set_enabled).
    ///
    /// This allocates only the small shared `Inner` once; the hot path is a
    /// single relaxed atomic load.
    #[must_use]
    pub fn noop() -> Self {
        Self {
            inner: Arc::new(Inner {
                enabled: AtomicBool::new(false),
                clock: Arc::new(TraceClock::new()),
                session_id: TraceSessionId::next(),
                collector: Arc::new(NoopCollector),
                format: TraceFormat::ChromeJson,
                verbosity: TraceVerbosity::default(),
                pid: std::process::id() as u64,
                tids: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Build an enabled context backed by a fresh [`MemoryCollector`], returning
    /// both the context and a shared handle to the collector so callers can read
    /// events back or export them.
    ///
    /// Defaults to [`TraceFormat::ChromeJson`].
    #[must_use]
    pub fn in_memory() -> (Self, Arc<MemoryCollector>) {
        let collector = Arc::new(MemoryCollector::new());
        let ctx = Self::with_collector(collector.clone(), TraceFormat::ChromeJson);
        (ctx, collector)
    }

    /// Set the default output format, consuming and returning the context for
    /// chaining. No-op if the context is shared (other clones keep the old
    /// format); call before cloning.
    #[must_use]
    pub fn with_format(mut self, format: TraceFormat) -> Self {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.format = format;
        }
        self
    }

    /// Set the capture verbosity, consuming and returning the context for
    /// chaining. No-op if the context is already shared.
    #[must_use]
    pub fn with_verbosity(mut self, verbosity: TraceVerbosity) -> Self {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.verbosity = verbosity;
        }
        self
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

    /// The session id stamped onto this context.
    #[must_use]
    pub fn session_id(&self) -> TraceSessionId {
        self.inner.session_id
    }

    /// The default output format.
    #[must_use]
    pub fn format(&self) -> TraceFormat {
        self.inner.format
    }

    /// The capture verbosity.
    #[must_use]
    pub fn verbosity(&self) -> TraceVerbosity {
        self.inner.verbosity
    }

    /// The shared clock.
    #[must_use]
    pub fn clock(&self) -> &Arc<TraceClock> {
        &self.inner.clock
    }

    /// The shared output collector.
    #[must_use]
    pub fn collector(&self) -> &Arc<dyn TraceCollector> {
        &self.inner.collector
    }

    /// The process id stamped onto every event from this context.
    #[must_use]
    pub fn pid(&self) -> u64 {
        self.inner.pid
    }

    /// The stable lane id assigned to the current OS thread by this context.
    ///
    /// The first thread to touch a given context gets `0`, the next `1`, and so
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

    /// Flush the underlying collector, persisting any buffered events.
    ///
    /// # Errors
    ///
    /// Propagates any [`TracerError`](crate::TracerError) from the collector
    /// (e.g. a file write failure, with the path named).
    pub fn flush(&self) -> Result<()> {
        self.inner.collector.flush()
    }

    /// Emit an already-built event to the collector, honouring the enable flag.
    pub fn emit(&self, event: &TraceEvent) {
        if !self.is_enabled() {
            return;
        }
        self.inner.collector.emit(event);
    }

    /// Record a complete (`"X"`) event that started at `start` and lasted
    /// `dur`. No-op when the context is disabled.
    ///
    /// Use this when you already have explicit timing. For scoped timing,
    /// prefer [`span`](TraceContext::span), which measures duration for you.
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
        let event = TraceEvent {
            name: name.into(),
            cat: cat.into(),
            ph: TracePhase::Complete,
            ts: self.inner.clock.micros_at(start),
            dur: Some(dur.as_micros() as u64),
            pid: self.inner.pid,
            tid,
            scope: None,
            args: args.map(Args::into_value),
        };
        self.inner.collector.emit(&event);
    }

    /// Record an instant (`"i"`) event at the current time with thread scope.
    /// No-op when the context is disabled.
    pub fn instant(&self, name: impl Into<String>, cat: impl Into<String>, args: Option<Args>) {
        if !self.is_enabled() {
            return;
        }
        let ts = self.inner.clock.now_micros();
        let tid = self.current_tid();
        let event = TraceEvent {
            name: name.into(),
            cat: cat.into(),
            ph: TracePhase::Instant,
            ts,
            dur: None,
            pid: self.inner.pid,
            tid,
            scope: Some('t'),
            args: args.map(Args::into_value),
        };
        self.inner.collector.emit(&event);
    }

    /// Open a scoped span. The returned [`SpanGuard`] records a complete event
    /// covering its lifetime when it is dropped (or [`finish`](SpanGuard::finish)ed).
    ///
    /// When the context is disabled the guard is inert: it holds no owned
    /// strings and does nothing on drop.
    pub fn span(&self, name: impl Into<String>, cat: impl Into<String>) -> SpanGuard {
        if !self.is_enabled() {
            return SpanGuard::inert();
        }
        let args = Arc::new(Mutex::new(Args::new()));
        ACTIVE_SPANS.with(|spans| spans.borrow_mut().push(Arc::downgrade(&args)));
        SpanGuard {
            state: Some(SpanState {
                ctx: self.clone(),
                name: name.into(),
                cat: cat.into(),
                start: Instant::now(),
                args,
            }),
        }
    }

    /// Emit a `process_name` metadata (`"M"`) event naming this process lane.
    /// No-op when disabled — and it does **not** allocate the name in that case.
    pub fn set_process_name(&self, name: impl Into<String>) {
        if !self.is_enabled() {
            return;
        }
        self.metadata("process_name", name.into());
    }

    /// Emit a `thread_name` metadata (`"M"`) event naming the current thread's
    /// lane. No-op when disabled — and it does **not** allocate the name in that
    /// case.
    pub fn set_thread_name(&self, name: impl Into<String>) {
        if !self.is_enabled() {
            return;
        }
        self.metadata("thread_name", name.into());
    }

    /// Emit a metadata event. Callers must have already checked the enable flag
    /// (so the disabled fast path never allocates a name string).
    fn metadata(&self, kind: &str, name: String) {
        let tid = self.current_tid();
        let args = Args::new().with("name", name);
        let event = TraceEvent {
            name: kind.to_string(),
            cat: "__metadata".to_string(),
            ph: TracePhase::Metadata,
            ts: 0,
            dur: None,
            pid: self.inner.pid,
            tid,
            scope: None,
            args: Some(args.into_value()),
        };
        self.inner.collector.emit(&event);
    }
}

impl std::fmt::Debug for TraceContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraceContext")
            .field("enabled", &self.is_enabled())
            .field("session_id", &self.inner.session_id)
            .field("format", &self.inner.format)
            .field("verbosity", &self.inner.verbosity)
            .field("pid", &self.inner.pid)
            .finish()
    }
}

/// The live state of an enabled span. Absent for inert (disabled) guards.
struct SpanState {
    ctx: TraceContext,
    name: String,
    cat: String,
    start: Instant,
    args: Arc<Mutex<Args>>,
}

/// An RAII guard that records a complete event covering its lifetime.
///
/// Created by [`TraceContext::span`]. The event is recorded when the guard is
/// dropped, or eagerly via [`finish`](SpanGuard::finish). Attach metadata
/// before the guard ends with [`set_args`](SpanGuard::set_args) or the chaining
/// [`with_args`](SpanGuard::with_args).
///
/// An inert guard (produced when the context was disabled) owns nothing and
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
            *state
                .args
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = args;
        }
        self
    }

    /// Attach or replace the span's args in place.
    pub fn set_args(&mut self, args: Args) {
        if let Some(state) = self.state.as_mut() {
            *state
                .args
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = args;
        }
    }

    /// The clock epoch-relative start timestamp of this span in microseconds,
    /// or `None` for an inert guard.
    #[must_use]
    pub fn start_ts(&self) -> Option<u64> {
        self.state
            .as_ref()
            .map(|state| state.ctx.inner.clock.micros_at(state.start))
    }

    /// Finish the span now, recording its event immediately instead of on drop.
    pub fn finish(mut self) {
        self.record();
    }

    fn record(&mut self) {
        if let Some(state) = self.state.take() {
            ACTIVE_SPANS.with(|spans| {
                let mut spans = spans.borrow_mut();
                if let Some(index) = spans.iter().rposition(|active| {
                    active
                        .upgrade()
                        .is_some_and(|args| Arc::ptr_eq(&args, &state.args))
                }) {
                    spans.remove(index);
                }
            });
            let dur = state.start.elapsed();
            let args = state
                .args
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            state
                .ctx
                .complete(
                    state.name,
                    state.cat,
                    state.start,
                    dur,
                    (!args.is_empty()).then_some(args),
                );
        }
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        self.record();
    }
}
