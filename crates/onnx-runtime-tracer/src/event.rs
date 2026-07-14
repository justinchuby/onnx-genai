//! The trace event model — a typed mirror of the
//! [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview).
//!
//! An exported trace is a JSON **array** of [`Event`] objects (the exact shape
//! shown in `docs/ORT2.md` §17.2). Perfetto (<https://ui.perfetto.dev>) and
//! `chrome://tracing` both ingest that array directly.
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
/// Only the subset the runtime needs is modelled. [`Complete`](Phase::Complete)
/// (`"X"`) is the workhorse: a single self-contained event with a start `ts`
/// and a `dur`. The begin/end pair and instant/metadata phases are provided
/// for callers that need them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    /// `"X"` — a complete event with a duration (`ts` .. `ts + dur`).
    Complete,
    /// `"B"` — the begin of a duration event (paired with [`Phase::End`]).
    Begin,
    /// `"E"` — the end of a duration event (paired with [`Phase::Begin`]).
    End,
    /// `"i"` — an instant event with no duration.
    Instant,
    /// `"M"` — a metadata event (e.g. `process_name`, `thread_name`).
    Metadata,
}

impl Phase {
    /// The single-character Chrome Trace phase code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Phase::Complete => "X",
            Phase::Begin => "B",
            Phase::End => "E",
            Phase::Instant => "i",
            Phase::Metadata => "M",
        }
    }

    /// Parse a phase from its Chrome Trace code, if recognised.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "X" => Some(Phase::Complete),
            "B" => Some(Phase::Begin),
            "E" => Some(Phase::End),
            "i" => Some(Phase::Instant),
            "M" => Some(Phase::Metadata),
            _ => None,
        }
    }
}

impl Serialize for Phase {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.code())
    }
}

impl<'de> Deserialize<'de> for Phase {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let code = String::deserialize(deserializer)?;
        Phase::from_code(&code)
            .ok_or_else(|| de::Error::custom(format!("unknown Chrome trace phase code {code:?}")))
    }
}

/// A single Chrome Trace event.
///
/// Timestamps ([`ts`](Event::ts)) and durations ([`dur`](Event::dur)) are in
/// **microseconds** relative to the owning [`Tracer`](crate::Tracer)'s epoch.
/// Construct events through the [`Tracer`](crate::Tracer) API rather than by
/// hand where possible; the public fields exist so exported traces round-trip
/// through `serde_json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// Human-readable event name (e.g. `"MatMul_0"`).
    pub name: String,
    /// Category used for colouring/grouping lanes (e.g. `"compute"`).
    pub cat: String,
    /// The event phase.
    pub ph: Phase,
    /// Start timestamp in microseconds, relative to the tracer epoch.
    pub ts: u64,
    /// Duration in microseconds. Present only for [`Phase::Complete`].
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

impl Event {
    /// Serialize this single event to a JSON object string.
    ///
    /// Rarely needed directly — callers usually export the whole trace via
    /// [`Tracer::to_chrome_json`](crate::Tracer::to_chrome_json).
    #[must_use]
    pub fn to_json(&self) -> String {
        // `Event` always serializes successfully, so the unwrap is infallible.
        serde_json::to_string(self).expect("Event serialization is infallible")
    }
}

impl Serialize for Event {
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
        let mut state = serializer.serialize_struct("Event", len)?;
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

impl<'de> Deserialize<'de> for Event {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EventVisitor;

        impl<'de> Visitor<'de> for EventVisitor {
            type Value = Event;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a Chrome Trace event object")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Event, M::Error> {
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

                Ok(Event {
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
