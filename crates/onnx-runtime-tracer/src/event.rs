//! The unified trace event model (§48.3).
//!
//! [`TraceEvent`] is the single event type both the runtime and genai layers
//! record. On the wire it serializes to the
//! [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
//! object shape (the exact form in `docs/ORT2.md` §17.2), so a
//! [`ChromeJson`](crate::TraceFormat::ChromeJson) or
//! [`Jsonl`](crate::TraceFormat::Jsonl) export is just a serialization of these
//! events, and the [`perfetto`](crate::perfetto) exporter maps the same fields
//! into `perfetto.protos.Trace` packets.
//!
//! Only the fields the runtime actually uses are modelled. Optional fields
//! (`dur`, `args`, `scope`) are omitted from the JSON when unset, keeping the
//! output compact and schema-valid.

use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

/// The event phase — the `ph` field in Chrome Trace Event Format.
///
/// Only the subset the runtime needs is modelled. [`Complete`](TracePhase::Complete)
/// (`"X"`) is the workhorse: a single self-contained event with a start `ts`
/// and a `dur`. The begin/end pair and instant/metadata phases are provided for
/// callers that need them (and for backends such as ITT that pair begin/end).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TracePhase {
    /// `"X"` — a complete event with a duration (`ts` .. `ts + dur`).
    Complete,
    /// `"B"` — the begin of a duration event (paired with [`TracePhase::End`]).
    Begin,
    /// `"E"` — the end of a duration event (paired with [`TracePhase::Begin`]).
    End,
    /// `"i"` — an instant event with no duration.
    Instant,
    /// `"M"` — a metadata event (e.g. `process_name`, `thread_name`).
    Metadata,
}

impl TracePhase {
    /// The single-character Chrome Trace phase code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            TracePhase::Complete => "X",
            TracePhase::Begin => "B",
            TracePhase::End => "E",
            TracePhase::Instant => "i",
            TracePhase::Metadata => "M",
        }
    }

    /// Parse a phase from its Chrome Trace code, if recognised.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "X" => Some(TracePhase::Complete),
            "B" => Some(TracePhase::Begin),
            "E" => Some(TracePhase::End),
            "i" => Some(TracePhase::Instant),
            "M" => Some(TracePhase::Metadata),
            _ => None,
        }
    }
}

impl Serialize for TracePhase {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.code())
    }
}

impl<'de> Deserialize<'de> for TracePhase {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let code = String::deserialize(deserializer)?;
        TracePhase::from_code(&code)
            .ok_or_else(|| de::Error::custom(format!("unknown Chrome trace phase code {code:?}")))
    }
}

/// A single unified trace event.
///
/// Timestamps ([`ts`](TraceEvent::ts)) and durations ([`dur`](TraceEvent::dur))
/// are in **microseconds** relative to the owning
/// [`TraceContext`](crate::TraceContext)'s [`TraceClock`](crate::TraceClock)
/// epoch. Construct events through the [`TraceContext`](crate::TraceContext) API
/// rather than by hand where possible; the public fields exist so exported
/// traces round-trip through `serde_json`.
///
/// Note: data-dependency (`flow_id`) and GPU-kernel correlation (`node_id`)
/// fields from §48.4 / §49 are intentionally **not** modelled yet — those land
/// with the executor-wiring and CUPTI phases.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceEvent {
    /// Human-readable event name (e.g. `"MatMul_0"`).
    pub name: String,
    /// Category used for colouring/grouping lanes (e.g. `"compute"`).
    pub cat: String,
    /// The event phase.
    pub ph: TracePhase,
    /// Start timestamp in microseconds, relative to the clock epoch.
    pub ts: u64,
    /// Duration in microseconds. Present only for [`TracePhase::Complete`].
    pub dur: Option<u64>,
    /// Process id (lane group).
    pub pid: u64,
    /// Thread id (lane within a process).
    pub tid: u64,
    /// Instant-event scope: `'g'` global, `'p'` process, `'t'` thread.
    pub scope: Option<char>,
    /// Arbitrary structured metadata rendered by the trace viewer.
    pub args: Option<Value>,
}

impl TraceEvent {
    /// Serialize this single event to a JSON object string (its Chrome Trace
    /// wire shape). Used by the [`Jsonl`](crate::TraceFormat::Jsonl) exporter.
    #[must_use]
    pub fn to_json(&self) -> String {
        // `TraceEvent` always serializes successfully, so the unwrap is infallible.
        serde_json::to_string(self).expect("TraceEvent serialization is infallible")
    }
}

impl Serialize for TraceEvent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Count only the fields we will actually emit so the struct length is
        // correct for formats that need it.
        let mut len = 6; // name, cat, ph, ts, pid, tid
        if self.dur.is_some() {
            len += 1;
        }
        if self.scope.is_some() {
            len += 1;
        }
        if self.args.is_some() {
            len += 1;
        }
        let mut state = serializer.serialize_struct("TraceEvent", len)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("cat", &self.cat)?;
        state.serialize_field("ph", &self.ph)?;
        state.serialize_field("ts", &self.ts)?;
        if let Some(dur) = self.dur {
            state.serialize_field("dur", &dur)?;
        }
        state.serialize_field("pid", &self.pid)?;
        state.serialize_field("tid", &self.tid)?;
        if let Some(scope) = self.scope {
            state.serialize_field("s", &scope)?;
        }
        if let Some(args) = &self.args {
            state.serialize_field("args", args)?;
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for TraceEvent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EventVisitor;

        impl<'de> Visitor<'de> for EventVisitor {
            type Value = TraceEvent;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a Chrome Trace event object")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<TraceEvent, M::Error> {
                let mut name = None;
                let mut cat = None;
                let mut ph = None;
                let mut ts = None;
                let mut dur = None;
                let mut pid = None;
                let mut tid = None;
                let mut scope = None;
                let mut args = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "name" => name = Some(map.next_value()?),
                        "cat" => cat = Some(map.next_value()?),
                        "ph" => ph = Some(map.next_value()?),
                        "ts" => ts = Some(map.next_value()?),
                        "dur" => dur = Some(map.next_value()?),
                        "pid" => pid = Some(map.next_value()?),
                        "tid" => tid = Some(map.next_value()?),
                        "s" => scope = Some(map.next_value()?),
                        "args" => args = Some(map.next_value()?),
                        _ => {
                            // Ignore unknown fields for forward compatibility
                            // with richer traces.
                            let _: de::IgnoredAny = map.next_value()?;
                        }
                    }
                }

                Ok(TraceEvent {
                    name: name.ok_or_else(|| de::Error::missing_field("name"))?,
                    cat: cat.unwrap_or_default(),
                    ph: ph.ok_or_else(|| de::Error::missing_field("ph"))?,
                    ts: ts.unwrap_or(0),
                    dur,
                    pid: pid.unwrap_or(0),
                    tid: tid.unwrap_or(0),
                    scope,
                    args,
                })
            }
        }

        deserializer.deserialize_map(EventVisitor)
    }
}
