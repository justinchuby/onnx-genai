//! One timeline for the engine and everything an execution provider does inside it.
//!
//! The engine records its own spans (`loop.step`, `native.session_run`, …)
//! through [`onnx_genai_ort::profile`], while the native runtime and its
//! execution providers record theirs — per-operator spans, and the per-kernel
//! device/bytes/flops annotations the EPs attach — through
//! [`onnx_runtime_tracer`]. Those are two sinks, and the Perfetto export only
//! ever read the first, so a native run showed `native.session_run` as one
//! opaque block with nothing inside it.
//!
//! This installs a collector on the second sink and hands its events to the
//! first, so both land on one timeline.
//!
//! # Why a collector rather than a call into the profiler
//!
//! Every execution provider is becoming a plugin, so the set of things that
//! record spans is open-ended and none of them can be expected to depend on the
//! engine's profiler. [`onnx_runtime_tracer`] is the foundational crate they all
//! already share: an EP annotates through it, and whatever the host installed
//! collects the result. Bridging at that seam means a new plugin needs no
//! bridging code of its own — it is already on the timeline.

use std::sync::{Arc, OnceLock};

use onnx_runtime_tracer::{MemoryCollector, TraceContext, TraceFormat, TraceVerbosity};

/// Retained so the events can be drained at export time.
static COLLECTOR: OnceLock<Arc<MemoryCollector>> = OnceLock::new();
/// Built once and shared, so every session lands on one set of thread lanes.
static CONTEXT: OnceLock<Option<TraceContext>> = OnceLock::new();

/// Opt in to worker-lane spans inside a fanned-out operator.
///
/// Off by default: a per-worker span costs a few hundred nanoseconds, which is
/// nothing against a whole node but real against a slice of one, and it
/// multiplies the event count by the pool width.
const VERBOSITY_ENV: &str = "ONNX_GENAI_TRACE_VERBOSITY";

fn verbosity() -> TraceVerbosity {
    match std::env::var(VERBOSITY_ENV).ok().as_deref() {
        Some("full") => TraceVerbosity::Full,
        Some("decisions") => TraceVerbosity::Decisions,
        _ => TraceVerbosity::Ops,
    }
}

/// The shared runtime trace context, or `None` when tracing is off.
///
/// Built once. Sessions created later join the same timeline, which is what
/// makes spans from different providers comparable — a context owns its
/// thread-lane numbering, so handing out fresh contexts would scatter one
/// thread across several lanes.
pub fn context() -> Option<TraceContext> {
    CONTEXT
        .get_or_init(|| {
            let collector = COLLECTOR.get_or_init(|| Arc::new(MemoryCollector::new()));
            let context = TraceContext::with_collector(
                collector.clone() as Arc<dyn onnx_runtime_tracer::TraceCollector>,
                TraceFormat::ChromeJson,
            )
            .with_verbosity(verbosity());
            // Installed whether or not tracing is on right now, and switched
            // later with `set_recording`. A session that started untraced
            // would otherwise hold a no-op context for its whole life, so
            // turning tracing on mid-session would produce a timeline with no
            // operator spans in it — and an interactive session is exactly
            // where someone decides they want a timeline only after seeing
            // something odd. A disabled context costs one relaxed atomic load.
            context.set_enabled(onnx_genai_ort::profile::tracing_enabled());
            // Publish it as the ambient context so provider worker threads —
            // which have neither an active span nor a handle to pass one
            // through — can open spans on their own lanes.
            onnx_runtime_tracer::set_global_context(Some(context.clone()));
            Some(context)
        })
        .clone()
}

/// Everything the runtime and its providers recorded, as Chrome trace events.
///
/// Empty when tracing was never enabled — an empty timeline rather than a
/// fabricated one.
pub fn collected_events() -> Vec<onnx_runtime_tracer::TraceEvent> {
    COLLECTOR
        .get()
        .map(|collector| collector.events())
        .unwrap_or_default()
}

/// Start or stop recording, and choose how much detail to record.
///
/// Takes effect immediately on the live context, so an interactive session can
/// turn a timeline on between turns without reloading the model.
pub fn set_recording(enabled: bool, verbosity: TraceVerbosity) {
    if let Some(context) = context() {
        context.set_verbosity(verbosity);
        context.set_enabled(enabled);
    }
}

/// Discard everything recorded so far, so the next export covers only what
/// follows it.
pub fn reset() {
    if let Some(collector) = COLLECTOR.get() {
        collector.clear();
    }
}
