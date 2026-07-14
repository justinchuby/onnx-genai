//! # `onnx-runtime-tracer`
//!
//! Unified, pure-Rust tracing for the ORT 2.0 runtime. It does one job well:
//! **collect timed spans from anywhere in the runtime and export them as
//! [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
//! JSON** — the format that both [Perfetto](https://ui.perfetto.dev) and
//! `chrome://tracing` load directly.
//!
//! It is a **foundational** crate, like [`onnx-runtime-ir`]: pure, safe Rust
//! (`#![forbid(unsafe_code)]`) with no dependency on the ORT / genai stack, so
//! any layer (runtime executor, EPs, or the genai engine) can adopt it without
//! pulling in a heavier tree. The only dependencies are `serde` and
//! `serde_json`.
//!
//! This corresponds to the tracing crate in the `docs/ORT2.md` crate map
//! (`crates/onnx-runtime-tracer/  # Unified tracing (shared by runtime +
//! genai)`) and the Chrome Trace export shape in §17.2.
//!
//! ## Opening a trace in Perfetto
//!
//! Write a trace to disk, then open the JSON file at
//! <https://ui.perfetto.dev> (drag-and-drop, or "Open trace file") or at
//! `chrome://tracing` in a Chromium browser. The exported document is a plain
//! JSON **array** of events, which both viewers accept.
//!
//! ## Quick start
//!
//! ```
//! use onnx_runtime_tracer::{Args, Tracer};
//!
//! // One tracer per run. Clone it freely across threads — all clones record
//! // into the same timeline.
//! let tracer = Tracer::new();
//! tracer.set_process_name("nxrt");
//!
//! {
//!     // RAII span: records a complete event covering this scope on drop.
//!     let _span = tracer
//!         .span("MatMul_0", "compute")
//!         .with_args(Args::new().device("cpu").shapes(vec![vec![1_i64, 4096, 4096]]));
//!     // ... do the work being timed ...
//! }
//!
//! let json = tracer.to_chrome_json();
//! assert!(json.starts_with('['));
//! // tracer.write_chrome_json("trace.json")?; // then open in Perfetto
//! ```
//!
//! ## Enabled / disabled cost
//!
//! A [`Tracer`] carries an atomic enable flag. When disabled, [`Tracer::span`],
//! [`Tracer::complete`], and the other entry points are a single relaxed
//! atomic load followed by an early return — no clock read, no allocation, no
//! lock. Production builds can leave a disabled tracer wired in at negligible
//! cost and flip it on only when profiling. See [`tracer`] for details.
//!
//! ## Scope (Phase 1)
//!
//! Phase 1 ships the standalone collector + Chrome/Perfetto export only. It is
//! intentionally **not** wired into the session executor yet — that touches the
//! shared session crate and is a deliberate follow-up. Future work described
//! in `docs/ORT2.md` (a `tracing`-crate [`Layer`] bridge for §17.1, the §46
//! execution trace log, and §43 metrics/observability) can build on this API
//! without changing it.
//!
//! [`onnx-runtime-ir`]: https://docs.rs/onnx-runtime-ir
//! [`Layer`]: https://docs.rs/tracing-subscriber/latest/tracing_subscriber/layer/trait.Layer.html

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod args;
pub mod error;
pub mod event;
pub mod tracer;

pub use args::Args;
pub use error::{Result, TracerError};
pub use event::{Event, Phase};
pub use tracer::{DEFAULT_MAX_EVENTS, SpanGuard, Tracer};
