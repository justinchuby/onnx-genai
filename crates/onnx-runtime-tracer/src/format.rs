//! Trace output [`TraceFormat`] and [`TraceVerbosity`] (§48.2 / §48.3).

use std::fmt;

/// The wire format a trace is serialized to (§48.3).
///
/// A [`TraceContext`](crate::TraceContext) carries a default format; individual
/// export helpers ([`chrome`](crate::chrome), [`jsonl`](crate::jsonl),
/// [`perfetto`](crate::perfetto)) can also be called directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TraceFormat {
    /// Chrome Trace Event Format JSON. The most portable option: both
    /// <https://ui.perfetto.dev> and `chrome://tracing` load it directly.
    ChromeJson,
    /// Perfetto native protobuf (`perfetto.protos.Trace`). Streams better for
    /// large traces and is the format Perfetto ingests natively. Requires the
    /// `perfetto` cargo feature to serialize.
    PerfettoProto,
    /// JSON Lines — one event object per line. `grep`/`jq`-friendly and
    /// append-streamable.
    Jsonl,
}

impl TraceFormat {
    /// A short, stable, lowercase name (matches the backend/registry key in
    /// §48.8.4 and the `--trace-format` CLI value in §48.6).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            TraceFormat::ChromeJson => "chrome",
            TraceFormat::PerfettoProto => "perfetto",
            TraceFormat::Jsonl => "jsonl",
        }
    }

    /// The conventional file extension for this format.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            TraceFormat::ChromeJson => "json",
            TraceFormat::PerfettoProto => "perfetto",
            TraceFormat::Jsonl => "jsonl",
        }
    }
}

impl fmt::Display for TraceFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How much detail a trace captures (§48.2).
///
/// Ordered from least to most detail: `Decisions < Ops < Full`. A higher level
/// includes everything a lower one would. This drives future event filtering
/// once the executor is wired in (§48.5); it is stored on the
/// [`TraceContext`](crate::TraceContext) today so callers can set intent, and
/// consumers can honor it as instrumentation lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TraceVerbosity {
    /// Only high-level scheduling/placement decisions (cheapest).
    Decisions,
    /// Decisions plus per-op execution spans.
    Ops,
    /// Everything: ops, transfers, counters, and fine-grained internals.
    Full,
}

impl TraceVerbosity {
    /// Whether a trace at `self` should include events tagged at `other`.
    ///
    /// True when `other` is no more detailed than `self` (e.g. a `Full` trace
    /// includes `Decisions`-level events, but a `Decisions` trace does not
    /// include `Ops`-level events).
    #[must_use]
    pub fn includes(self, other: TraceVerbosity) -> bool {
        other <= self
    }
}

impl Default for TraceVerbosity {
    fn default() -> Self {
        TraceVerbosity::Full
    }
}

impl fmt::Display for TraceVerbosity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TraceVerbosity::Decisions => "decisions",
            TraceVerbosity::Ops => "ops",
            TraceVerbosity::Full => "full",
        };
        f.write_str(s)
    }
}
