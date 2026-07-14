//! Intel ITT (VTune / Inspector) collector — **stub** (§48.8.2), behind the
//! `itt` cargo feature.
//!
//! The design (§48.8.2) bridges [`TraceEvent`](crate::TraceEvent)s into Intel
//! ITT API annotations so that, when VTune is attached, spans appear on its
//! timeline. That requires the upstream `ittapi` crate and a live VTune
//! collector, neither of which is available in this offline build, so the real
//! collector body is deferred to a later phase.
//!
//! For now the `itt` feature is a **capability flag with no dependency**: it
//! gates this documented placeholder so the flag exists and compiles. When the
//! `ittapi` dependency is wired in, [`IttCollector`] will implement
//! [`TraceCollector`](crate::TraceCollector) as sketched in §48.8.2 (begin/end
//! tasks, instant markers, a `DashMap` string-handle cache).

/// Placeholder for the future Intel ITT collector (§48.8.2).
///
/// Constructing one and using it is intentionally unsupported until the real
/// `ittapi` integration lands. [`new`](IttCollector::new) panics with a clear,
/// actionable message rather than silently producing a collector that records
/// nothing.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct IttCollector;

impl IttCollector {
    /// Not yet implemented.
    ///
    /// # Panics
    ///
    /// Always — the ITT collector body is deferred (see the module docs). This
    /// is a `todo!` marker, never reached on any supported path because nothing
    /// constructs it yet.
    #[must_use]
    pub fn new(_domain_name: &str) -> Self {
        todo!(
            "IttCollector is a deferred phase: the `itt` feature is currently a \
             capability flag only, with no `ittapi` dependency. Wire in the \
             `ittapi` crate and implement TraceCollector per docs/ORT2.md §48.8.2 \
             before constructing this."
        )
    }
}
