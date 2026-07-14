//! JSON Lines serialization (§48.3).
//!
//! One [`TraceEvent`] JSON object per line. Unlike the Chrome array format,
//! JSONL is append-streamable and trivially `grep`/`jq`-able, which is why the
//! [`FileCollector`](crate::FileCollector) can stream it live. Each line is the
//! event's Chrome-shape object, so a JSONL file is a Chrome array with the
//! surrounding `[`, `]`, and commas stripped.

use crate::event::TraceEvent;

/// Serialize a single event to one JSONL line (no trailing newline).
#[must_use]
pub fn event_to_line(event: &TraceEvent) -> String {
    event.to_json()
}

/// Serialize events as a JSONL document (one object per line, trailing
/// newline after each). An empty slice yields an empty string.
#[must_use]
pub fn to_jsonl(events: &[TraceEvent]) -> String {
    let mut out = String::new();
    for event in events {
        out.push_str(&event.to_json());
        out.push('\n');
    }
    out
}
