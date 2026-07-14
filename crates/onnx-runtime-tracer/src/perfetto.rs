//! Perfetto native protobuf export (§48.7 / §48.8.6), behind the `perfetto`
//! cargo feature.
//!
//! Perfetto ingests the standard `perfetto.protos.Trace` message. Rather than
//! compile the full (very large) upstream `.proto` schema — which would require
//! `protoc` at build time — we hand-write a **minimal but valid subset** of the
//! schema as [`prost`] messages. Perfetto's UI and `trace_processor` accept a
//! trace built from just these messages.
//!
//! ## Covered subset
//!
//! * [`Trace`] → repeated [`TracePacket`].
//! * [`TrackDescriptor`] packets that declare one process track and one thread
//!   track per OS thread lane (named from `process_name` / `thread_name`
//!   metadata events when present — no vendor names are hardcoded).
//! * [`TrackEvent`] packets: `SLICE_BEGIN` / `SLICE_END` pairs for
//!   [`Complete`](crate::TracePhase::Complete) and
//!   [`Begin`](crate::TracePhase::Begin)/[`End`](crate::TracePhase::End)
//!   events, and `INSTANT` for [`Instant`](crate::TracePhase::Instant) events.
//!   Each event carries its `name` and a single `category`.
//! * All timestamps are absolute nanoseconds from the trace clock epoch
//!   (Chrome-model microseconds × 1000).
//!
//! ## Deferred (not yet emitted)
//!
//! Counter tracks (§48.3 counters), flow events / arrows (§48.4), interned data
//! (string/category interning for size), GPU kernel tracks (§49), and custom
//! clock snapshots. These are additive and can be layered on without changing
//! the covered subset. Chrome JSON export remains the fuller-fidelity path for
//! `args`, which the `TrackEvent` subset here does not carry.

use crate::clock::TraceSessionId;
use crate::event::{TraceEvent, TracePhase};
use prost::Message;
use std::collections::BTreeMap;

// ── Minimal hand-written subset of perfetto.protos ──
// Field numbers match the upstream schema so Perfetto can parse the output.

/// `perfetto.protos.Trace` — the top-level trace container.
#[derive(Clone, PartialEq, Message)]
pub struct Trace {
    /// The ordered stream of trace packets.
    #[prost(message, repeated, tag = "1")]
    pub packet: Vec<TracePacket>,
}

/// `perfetto.protos.TracePacket` (subset).
#[derive(Clone, PartialEq, Message)]
pub struct TracePacket {
    /// Absolute timestamp in nanoseconds (default clock).
    #[prost(uint64, optional, tag = "8")]
    pub timestamp: Option<u64>,
    /// Packet sequence this event belongs to (track events require one).
    #[prost(uint32, optional, tag = "10")]
    pub trusted_packet_sequence_id: Option<u32>,
    /// A track event (slice begin/end/instant).
    #[prost(message, optional, tag = "11")]
    pub track_event: Option<TrackEvent>,
    /// A track descriptor (declares a process/thread track).
    #[prost(message, optional, tag = "60")]
    pub track_descriptor: Option<TrackDescriptor>,
}

/// `perfetto.protos.TrackEvent` (subset).
#[derive(Clone, PartialEq, Message)]
pub struct TrackEvent {
    /// The track this event belongs to.
    #[prost(uint64, optional, tag = "11")]
    pub track_uuid: Option<u64>,
    /// Event categories (we emit exactly one — the Chrome `cat`).
    #[prost(string, repeated, tag = "22")]
    pub categories: Vec<String>,
    /// Event name.
    #[prost(string, optional, tag = "23")]
    pub name: Option<String>,
    /// Event type (slice begin/end/instant).
    #[prost(enumeration = "TrackEventType", optional, tag = "9")]
    pub r#type: Option<i32>,
}

/// `perfetto.protos.TrackEvent.Type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ::prost::Enumeration)]
#[repr(i32)]
pub enum TrackEventType {
    /// Unset.
    Unspecified = 0,
    /// Opens a slice on the track.
    SliceBegin = 1,
    /// Closes the most recent open slice on the track.
    SliceEnd = 2,
    /// A zero-duration marker.
    Instant = 3,
    /// A counter sample (unused here; declared for completeness).
    Counter = 4,
}

/// `perfetto.protos.TrackDescriptor` (subset).
#[derive(Clone, PartialEq, Message)]
pub struct TrackDescriptor {
    /// Stable unique id for this track.
    #[prost(uint64, optional, tag = "1")]
    pub uuid: Option<u64>,
    /// Human-readable track name.
    #[prost(string, optional, tag = "2")]
    pub name: Option<String>,
    /// Present when this track represents a process.
    #[prost(message, optional, tag = "3")]
    pub process: Option<ProcessDescriptor>,
    /// Present when this track represents a thread.
    #[prost(message, optional, tag = "4")]
    pub thread: Option<ThreadDescriptor>,
}

/// `perfetto.protos.ProcessDescriptor` (subset).
#[derive(Clone, PartialEq, Message)]
pub struct ProcessDescriptor {
    /// OS process id.
    #[prost(int32, optional, tag = "1")]
    pub pid: Option<i32>,
    /// Process name.
    #[prost(string, optional, tag = "6")]
    pub process_name: Option<String>,
}

/// `perfetto.protos.ThreadDescriptor` (subset).
#[derive(Clone, PartialEq, Message)]
pub struct ThreadDescriptor {
    /// Owning process id (groups the thread under its process track).
    #[prost(int32, optional, tag = "1")]
    pub pid: Option<i32>,
    /// OS thread id (our per-context lane id).
    #[prost(int32, optional, tag = "2")]
    pub tid: Option<i32>,
    /// Thread name.
    #[prost(string, optional, tag = "5")]
    pub thread_name: Option<String>,
}

/// One trusted sequence covers the whole trace.
const SEQUENCE_ID: u32 = 1;

/// Deterministic, non-zero uuid for the process track.
fn process_track_uuid(pid: u64) -> u64 {
    // High bit set keeps it clear of thread-track uuids and non-zero.
    0x8000_0000_0000_0000 | (pid & 0x7FFF_FFFF)
}

/// Deterministic, non-zero uuid for a thread lane's track.
fn thread_track_uuid(pid: u64, tid: u64) -> u64 {
    0x4000_0000_0000_0000 | ((pid & 0x00FF_FFFF) << 24) | (tid & 0x00FF_FFFF)
}

/// Microseconds → nanoseconds (Perfetto's default time unit).
fn ns(micros: u64) -> u64 {
    micros.saturating_mul(1_000)
}

/// Serialize `events` to a Perfetto `Trace` protobuf byte stream.
///
/// `session` is used only to derive a stable, human-readable default process
/// track name when no `process_name` metadata event is present. Encoding to a
/// `Vec<u8>` is infallible in `prost`.
#[must_use]
pub fn to_perfetto_proto(events: &[TraceEvent], session: Option<TraceSessionId>) -> Vec<u8> {
    let mut trace = Trace { packet: Vec::new() };

    // Derive the process id and any track names from metadata events.
    let pid = events.first().map(|e| e.pid).unwrap_or(0);
    let mut process_name: Option<String> = None;
    // tid -> thread name, preserving discovery order via BTreeMap on tid.
    let mut thread_names: BTreeMap<u64, String> = BTreeMap::new();
    // Every tid we see any event for gets a track, named or not.
    let mut thread_tids: BTreeMap<u64, ()> = BTreeMap::new();

    for e in events {
        thread_tids.insert(e.tid, ());
        if e.ph == TracePhase::Metadata {
            let name = e
                .args
                .as_ref()
                .and_then(|v| v.get("name"))
                .and_then(|v| v.as_str())
                .map(str::to_string);
            match (e.name.as_str(), name) {
                ("process_name", Some(n)) => process_name = Some(n),
                ("thread_name", Some(n)) => {
                    thread_names.insert(e.tid, n);
                }
                _ => {}
            }
        }
    }

    // Process track descriptor.
    let process_track = process_track_uuid(pid);
    trace.packet.push(TracePacket {
        timestamp: None,
        trusted_packet_sequence_id: Some(SEQUENCE_ID),
        track_event: None,
        track_descriptor: Some(TrackDescriptor {
            uuid: Some(process_track),
            name: process_name.clone(),
            process: Some(ProcessDescriptor {
                pid: Some(pid as i32),
                process_name: Some(process_name.unwrap_or_else(|| match session {
                    Some(id) => format!("session {id}"),
                    None => format!("pid {pid}"),
                })),
            }),
            thread: None,
        }),
    });

    // Thread track descriptors.
    for tid in thread_tids.keys().copied() {
        let uuid = thread_track_uuid(pid, tid);
        let name = thread_names.get(&tid).cloned();
        trace.packet.push(TracePacket {
            timestamp: None,
            trusted_packet_sequence_id: Some(SEQUENCE_ID),
            track_event: None,
            track_descriptor: Some(TrackDescriptor {
                uuid: Some(uuid),
                name: name.clone(),
                process: None,
                thread: Some(ThreadDescriptor {
                    pid: Some(pid as i32),
                    tid: Some(tid as i32),
                    thread_name: name,
                }),
            }),
        });
    }

    // Event packets.
    for e in events {
        let track = thread_track_uuid(pid, e.tid);
        match e.ph {
            TracePhase::Complete => {
                let end = e.ts + e.dur.unwrap_or(0);
                trace
                    .packet
                    .push(slice(track, e, TrackEventType::SliceBegin, e.ts));
                trace
                    .packet
                    .push(slice(track, e, TrackEventType::SliceEnd, end));
            }
            TracePhase::Begin => {
                trace
                    .packet
                    .push(slice(track, e, TrackEventType::SliceBegin, e.ts));
            }
            TracePhase::End => {
                trace
                    .packet
                    .push(slice(track, e, TrackEventType::SliceEnd, e.ts));
            }
            TracePhase::Instant => {
                trace
                    .packet
                    .push(slice(track, e, TrackEventType::Instant, e.ts));
            }
            // Metadata is expressed through track descriptors above.
            TracePhase::Metadata => {}
        }
    }

    trace.encode_to_vec()
}

/// Build a single track-event packet.
fn slice(track: u64, e: &TraceEvent, kind: TrackEventType, ts_micros: u64) -> TracePacket {
    // SliceEnd carries no name/category in Perfetto (it closes the open slice).
    let (name, categories) = match kind {
        TrackEventType::SliceEnd => (None, Vec::new()),
        _ => (Some(e.name.clone()), vec![e.cat.clone()]),
    };
    TracePacket {
        timestamp: Some(ns(ts_micros)),
        trusted_packet_sequence_id: Some(SEQUENCE_ID),
        track_event: Some(TrackEvent {
            track_uuid: Some(track),
            categories,
            name,
            r#type: Some(kind as i32),
        }),
        track_descriptor: None,
    }
}
