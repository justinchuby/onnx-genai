//! # `onnx-runtime-tracer`
//!
//! Unified, pure-Rust tracing for the ORT 2.0 runtime. It implements the
//! **`TraceCollector` collector architecture** from `docs/ORT2.md` §48: code is
//! annotated **once** through a shared [`TraceContext`], and *which* backends
//! consume those events is a configuration choice expressed as a
//! [`TraceCollector`] (or a [`CompositeCollector`] fan-out to several at once).
//!
//! It is a **foundational** crate, like [`onnx-runtime-ir`]: pure, safe Rust
//! (`#![forbid(unsafe_code)]` on every default build) with no dependency on the
//! ORT / genai stack, so any layer (runtime executor, EPs, or the genai engine)
//! can adopt it and share the exact same [`TraceContext`] type and one timeline
//! (§48.2). The one exception is the optional `cupti` feature, whose GPU-kernel
//! collector needs `unsafe` FFI/dlopen; enabling it relaxes the crate attribute
//! to `#![cfg_attr(not(feature = "cupti"), forbid(unsafe_code))]`, confining all
//! `unsafe` to [`cupti`].
//!
//! ## Architecture
//!
//! * [`TraceContext`] — the shared context (§48.3): a monotonic
//!   [`TraceClock`], a [`TraceSessionId`], an output [`collector`](TraceCollector),
//!   a default [`TraceFormat`], and a [`TraceVerbosity`]. Clone-shared (all
//!   `Arc`-backed) so multiple layers record into one sink;
//!   [`TraceContext::noop`] is the zero-overhead disabled path.
//! * [`TraceEvent`] / [`TracePhase`] — the unified event model, serialized to
//!   Chrome Trace Event Format on the wire.
//! * [`TraceCollector`] — the output sink trait. Built-ins:
//!   [`NoopCollector`] (default, zero overhead), [`MemoryCollector`] (capture
//!   into a `Vec` for API/tests), [`FileCollector`] (write a chosen format to a
//!   file), and [`CompositeCollector`] (fan-out to many).
//! * Serializers: [`chrome`] (always available), [`jsonl`] (always available),
//!   and [`perfetto`] (behind the `perfetto` feature) for
//!   `perfetto.protos.Trace` protobuf.
//!
//! ## Quick start
//!
//! ```
//! use onnx_runtime_tracer::{Args, TraceContext};
//!
//! // A context backed by an in-memory collector; keep the collector handle to
//! // read events back or export them.
//! let (ctx, mem) = TraceContext::in_memory();
//! ctx.set_process_name("nxrt");
//!
//! {
//!     // RAII span: records a complete event covering this scope on drop.
//!     let _span = ctx
//!         .span("MatMul_0", "compute")
//!         .with_args(Args::new().device("cpu").shapes(vec![vec![1_i64, 4096, 4096]]));
//!     // ... do the work being timed ...
//! }
//!
//! let json = mem.to_chrome_json();
//! assert!(json.starts_with('['));
//! ```
//!
//! ## Fan-out to multiple backends
//!
//! ```
//! use std::sync::Arc;
//! use onnx_runtime_tracer::{
//!     CompositeCollector, MemoryCollector, TraceContext, TraceFormat,
//! };
//!
//! let mem = Arc::new(MemoryCollector::new());
//! let composite = CompositeCollector::new().with(Box::new(MemoryCollector::new()));
//! # let _ = composite;
//! let ctx = TraceContext::with_collector(mem.clone(), TraceFormat::ChromeJson);
//! { let _s = ctx.span("op", "compute"); }
//! assert_eq!(mem.len(), 1);
//! ```
//!
//! ## Enabled / disabled cost
//!
//! When a [`TraceContext`] is disabled (e.g. [`TraceContext::noop`]), every
//! entry point is a single relaxed atomic load followed by an early return —
//! no clock read, no allocation, no lock. In particular the metadata helpers
//! check the flag *before* converting their arguments, so a disabled context
//! never allocates. See [`context`] for details.
//!
//! ## Cargo features
//!
//! | Feature | Default | Effect |
//! |---------|---------|--------|
//! | `perfetto` | ✅ | Perfetto protobuf export via `prost` ([`perfetto`]). |
//! | `chrome` | | Names the always-available Chrome JSON backend (no dep). |
//! | `itt` | | Capability flag for the future Intel ITT collector ([`itt`], stub). |
//! | `otel` | | Capability flag for the future OpenTelemetry export. |
//! | `cupti` | | GPU kernel tracing via a runtime-`dlopen`'d CUPTI Activity API collector ([`cupti`]). |
//!
//! The crate compiles with default features, with `--no-default-features`, and
//! with any combination of the above.
//!
//! ## Scope (Phase 2)
//!
//! This crate ships the standalone collector architecture + Chrome/JSONL/
//! Perfetto export, plus the `cupti` GPU-kernel collector (Activity-API path,
//! §48.8.3/§49). It is intentionally **not** wired into the session executor
//! yet (§48.5); the ITT (§48.8.2) and OpenTelemetry backends are declared but
//! deferred, and CUPTI PM-counter metrics (§49.3) are a documented Phase-2 stub.
//!
//! [`onnx-runtime-ir`]: https://docs.rs/onnx-runtime-ir

#![cfg_attr(not(feature = "cupti"), forbid(unsafe_code))]
#![warn(missing_docs)]

pub mod args;
pub mod chrome;
pub mod clock;
pub mod collector;
pub mod context;
pub mod error;
pub mod event;
pub mod format;
pub mod jsonl;

#[cfg(feature = "itt")]
pub mod itt;

#[cfg(feature = "cupti")]
pub mod cupti;

#[cfg(feature = "perfetto")]
pub mod perfetto;

pub use args::Args;
pub use clock::{TraceClock, TraceSessionId};
pub use collector::{
    CompositeCollector, DEFAULT_MAX_EVENTS, FileCollector, MemoryCollector, NoopCollector,
    TraceCollector,
};
pub use context::{SpanGuard, TraceContext};
pub use error::{Result, TracerError};
pub use event::{TraceEvent, TracePhase};
pub use format::{TraceFormat, TraceVerbosity};
