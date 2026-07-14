//! Chrome Trace Event Format JSON serialization (§48.2 / §48.3).
//!
//! Always available (no cargo feature required). A Chrome trace document is a
//! top-level JSON **array** of [`TraceEvent`] objects; both
//! <https://ui.perfetto.dev> and `chrome://tracing` load it directly.

use crate::event::TraceEvent;

/// Serialize events as a compact Chrome Trace JSON array.
#[must_use]
pub fn to_chrome_json(events: &[TraceEvent]) -> String {
    // The built-in `TraceEvent` model always serializes successfully.
    serde_json::to_string(events).expect("TraceEvent array serialization is infallible")
}

/// Serialize events as a pretty-printed Chrome Trace JSON array.
#[must_use]
pub fn to_chrome_json_pretty(events: &[TraceEvent]) -> String {
    serde_json::to_string_pretty(events).expect("TraceEvent array serialization is infallible")
}
