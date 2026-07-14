//! The [`TraceCollector`] trait and the built-in collectors (§48.2 / §48.8).
//!
//! A collector is the output sink a [`TraceContext`](crate::TraceContext) emits
//! into. The runtime and genai layers annotate code once; *which* backends are
//! active is a configuration choice expressed as a collector (or a
//! [`CompositeCollector`] fan-out).
//!
//! ## `emit(&self, event: &TraceEvent)` — by reference, on purpose
//!
//! The §48.3 sketch showed `fn emit(&self, event: TraceEvent)` (by value), but
//! the §48.8.1 [`CompositeCollector`] sketch takes `&TraceEvent`. We reconcile
//! the two in favor of **by reference**: a single event must fan out to N
//! backends, and passing `&TraceEvent` lets [`CompositeCollector`] hand the
//! *same* event to every collector with **zero clones on the fan-out path**.
//! Collectors that need to retain the event (e.g. [`MemoryCollector`],
//! [`FileCollector`]) clone it themselves — so exactly the collectors that keep
//! data pay for a copy, and pure-forwarding or side-effecting collectors
//! ([`NoopCollector`], a future ITT collector) pay nothing.

use crate::error::{Result, TracerError};
use crate::event::TraceEvent;
use crate::format::TraceFormat;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Default cap on events retained by a [`MemoryCollector`] so an unbounded run
/// cannot exhaust memory. See [`MemoryCollector`] for the overflow contract.
pub const DEFAULT_MAX_EVENTS: usize = 1_000_000;

/// An output sink for [`TraceEvent`]s.
///
/// Implementors must be `Send + Sync`: a single collector is shared (behind an
/// [`Arc`](std::sync::Arc)) across every thread and layer that traces. See the
/// [module docs](crate::collector) for why [`emit`](TraceCollector::emit) takes
/// the event by reference.
pub trait TraceCollector: Send + Sync {
    /// Record one event. Must be cheap and non-blocking on the hot path;
    /// buffering/serialization should be deferred to [`flush`](TraceCollector::flush)
    /// where practical.
    fn emit(&self, event: &TraceEvent);

    /// Persist or finalize any buffered events.
    ///
    /// # Errors
    ///
    /// Returns a [`TracerError`] if the sink could not be finalized (for a
    /// file-backed sink, an I/O failure — reported with the offending path).
    fn flush(&self) -> Result<()>;
}

/// A zero-overhead collector that discards every event (§48.2).
///
/// This is the default sink for [`TraceContext::noop`](crate::TraceContext::noop):
/// [`emit`](TraceCollector::emit) is an empty function the optimizer can inline
/// away, and [`flush`](TraceCollector::flush) is infallible.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopCollector;

impl TraceCollector for NoopCollector {
    #[inline]
    fn emit(&self, _event: &TraceEvent) {}

    #[inline]
    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

/// A collector that captures events into an in-memory [`Vec`] (§48.2).
///
/// Intended for programmatic access and tests: after emitting, read the events
/// back with [`events`](MemoryCollector::events) or export them via the
/// convenience helpers. Share it by cloning an [`Arc<MemoryCollector>`] — all
/// clones observe the same buffer.
///
/// ## Overflow contract (bounded, no silent drop)
///
/// The buffer is **bounded** at [`capacity`](MemoryCollector::capacity)
/// (default [`DEFAULT_MAX_EVENTS`]). This is a deliberate memory safety valve,
/// not a ring buffer: once full, further events are **dropped** (the earliest
/// events, which usually carry setup context, are preserved). Dropping is
/// **never silent** — the first drop emits a single one-time warning to stderr,
/// and the running total is available via [`dropped`](MemoryCollector::dropped).
/// For unbounded capture, raise the cap with
/// [`with_capacity`](MemoryCollector::with_capacity) or stream to a
/// [`FileCollector`] instead.
#[derive(Debug)]
pub struct MemoryCollector {
    events: Mutex<Vec<TraceEvent>>,
    capacity: usize,
    dropped: AtomicUsize,
    warned: AtomicBool,
}

impl MemoryCollector {
    /// Create a collector with the default [`DEFAULT_MAX_EVENTS`] cap.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_EVENTS)
    }

    /// Create a collector that retains at most `capacity` events. A capacity of
    /// `0` disables capture (every event is counted as dropped).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            capacity,
            dropped: AtomicUsize::new(0),
            warned: AtomicBool::new(false),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<TraceEvent>> {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A snapshot copy of the captured events, in insertion order.
    #[must_use]
    pub fn events(&self) -> Vec<TraceEvent> {
        self.lock().clone()
    }

    /// Number of events currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether no events are currently retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// The retention cap.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// How many events have been dropped because the cap was reached.
    #[must_use]
    pub fn dropped(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Drop all captured events. The drop counter and warn-once latch are reset
    /// so the collector is reusable.
    pub fn clear(&self) {
        self.lock().clear();
        self.dropped.store(0, Ordering::Relaxed);
        self.warned.store(false, Ordering::Relaxed);
    }

    /// Export the captured events as a Chrome Trace JSON array.
    #[must_use]
    pub fn to_chrome_json(&self) -> String {
        crate::chrome::to_chrome_json(&self.events())
    }

    /// Export the captured events as a JSONL document.
    #[must_use]
    pub fn to_jsonl(&self) -> String {
        crate::jsonl::to_jsonl(&self.events())
    }

    /// Export the captured events as a Perfetto protobuf `Trace`.
    ///
    /// Only available with the `perfetto` cargo feature.
    #[cfg(feature = "perfetto")]
    #[must_use]
    pub fn to_perfetto_proto(&self) -> Vec<u8> {
        crate::perfetto::to_perfetto_proto(&self.events(), None)
    }
}

impl Default for MemoryCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceCollector for MemoryCollector {
    fn emit(&self, event: &TraceEvent) {
        let mut events = self.lock();
        if events.len() >= self.capacity {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            // Warn exactly once, naming the cap and how to lift it, so a full
            // buffer is never a silent data loss (RULES.md #1).
            if !self.warned.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "onnx-runtime-tracer: MemoryCollector reached its {} event cap; \
                     further events are being dropped. Raise it with \
                     MemoryCollector::with_capacity(n), or stream to a FileCollector \
                     for unbounded traces.",
                    self.capacity
                );
            }
            return;
        }
        events.push(event.clone());
    }

    fn flush(&self) -> Result<()> {
        // Nothing to persist — events live in memory until read back.
        Ok(())
    }
}

/// A collector that writes captured events to a file in a chosen
/// [`TraceFormat`] (§48.2).
///
/// Events are **buffered in memory** as they are emitted and the complete
/// document is written to the target path on [`flush`](TraceCollector::flush)
/// (Chrome JSON and Perfetto proto are whole-document formats, so a single
/// atomic write is both simplest and safest; JSONL is written the same way for
/// uniformity). Call [`flush`](TraceCollector::flush) once at the end of a run
/// — for example via [`TraceContext::flush`](crate::TraceContext::flush).
///
/// The target file is created (and truncated) up front by
/// [`new`](FileCollector::new) so an unwritable path fails **early**, with the
/// path named, instead of only surfacing at flush time.
#[derive(Debug)]
pub struct FileCollector {
    path: PathBuf,
    format: TraceFormat,
    buffer: Mutex<Vec<TraceEvent>>,
}

impl FileCollector {
    /// Create a collector that will write `format` to `path`.
    ///
    /// The file is created/truncated immediately to validate writability.
    ///
    /// # Errors
    ///
    /// Returns [`TracerError::UnsupportedFormat`] if Perfetto protobuf output
    /// was requested in a build without the `perfetto` cargo feature, or
    /// [`TracerError::Write`] if the file cannot be created. Both errors explain
    /// how to fix the configuration.
    pub fn new(path: impl AsRef<Path>, format: TraceFormat) -> Result<Self> {
        if format == TraceFormat::PerfettoProto && !cfg!(feature = "perfetto") {
            return Err(TracerError::UnsupportedFormat { format });
        }

        let path = path.as_ref().to_path_buf();
        std::fs::File::create(&path).map_err(|source| TracerError::Write {
            path: path.clone(),
            format,
            source,
        })?;
        Ok(Self {
            path,
            format,
            buffer: Mutex::new(Vec::new()),
        })
    }

    /// The path this collector writes to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The format this collector writes.
    #[must_use]
    pub fn format(&self) -> TraceFormat {
        self.format
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<TraceEvent>> {
        self.buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn serialized(&self, events: &[TraceEvent]) -> Vec<u8> {
        match self.format {
            TraceFormat::ChromeJson => crate::chrome::to_chrome_json(events).into_bytes(),
            TraceFormat::Jsonl => crate::jsonl::to_jsonl(events).into_bytes(),
            TraceFormat::PerfettoProto => {
                #[cfg(feature = "perfetto")]
                {
                    crate::perfetto::to_perfetto_proto(events, None)
                }
                #[cfg(not(feature = "perfetto"))]
                {
                    unreachable!(
                        "FileCollector::new rejects PerfettoProto when the `perfetto` \
                         cargo feature is disabled"
                    )
                }
            }
        }
    }
}

impl TraceCollector for FileCollector {
    fn emit(&self, event: &TraceEvent) {
        self.lock().push(event.clone());
    }

    fn flush(&self) -> Result<()> {
        let events = self.lock();
        let bytes = self.serialized(&events);
        std::fs::write(&self.path, bytes).map_err(|source| TracerError::Write {
            path: self.path.clone(),
            format: self.format,
            source,
        })
    }
}

/// Fan-out collector: emit one event to many backends at once (§48.8.1).
///
/// This is how "multiple tools simultaneously" (e.g. a Chrome JSON file *and* a
/// Perfetto file, or a live ITT feed *and* an on-disk trace) is expressed. The
/// event is passed to each inner collector by reference, so the fan-out itself
/// never clones.
#[derive(Default)]
pub struct CompositeCollector {
    collectors: Vec<Box<dyn TraceCollector>>,
}

impl CompositeCollector {
    /// Create an empty fan-out.
    #[must_use]
    pub fn new() -> Self {
        Self {
            collectors: Vec::new(),
        }
    }

    /// Add a backend to fan out to.
    pub fn add(&mut self, collector: Box<dyn TraceCollector>) {
        self.collectors.push(collector);
    }

    /// Builder form of [`add`](CompositeCollector::add) for chaining.
    #[must_use]
    pub fn with(mut self, collector: Box<dyn TraceCollector>) -> Self {
        self.add(collector);
        self
    }

    /// Number of backends attached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.collectors.len()
    }

    /// Whether no backends are attached (all emits are no-ops).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.collectors.is_empty()
    }
}

impl TraceCollector for CompositeCollector {
    fn emit(&self, event: &TraceEvent) {
        for collector in &self.collectors {
            collector.emit(event);
        }
    }

    /// Flush **every** backend, even if one fails.
    ///
    /// A failing backend must not prevent the others from being persisted, so
    /// we attempt all flushes and return the **first** error encountered (if
    /// any) once every backend has been given the chance to flush. This avoids
    /// silently skipping later collectors when an earlier one errors.
    fn flush(&self) -> Result<()> {
        let mut first_err = None;
        for collector in &self.collectors {
            if let Err(err) = collector.flush() {
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
        }
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl std::fmt::Debug for CompositeCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeCollector")
            .field("collectors", &self.collectors.len())
            .finish()
    }
}
