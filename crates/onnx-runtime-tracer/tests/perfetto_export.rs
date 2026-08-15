//! Integration test for Perfetto protobuf export (§48.7). Compiled only with
//! the `perfetto` feature.
//!
//! Validates that the exporter produces **non-empty, valid** `Trace` protobuf
//! bytes that decode back to the expected track descriptors and track events.

#![cfg(feature = "perfetto")]

use onnx_runtime_tracer::perfetto::{self, Trace, TrackEventType};
use onnx_runtime_tracer::{Args, MemoryCollector, TraceContext, TraceFormat};
use prost::Message;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn perfetto_export_is_non_empty_and_valid() {
    let mem = Arc::new(MemoryCollector::new());
    let ctx = TraceContext::with_collector(mem.clone(), TraceFormat::PerfettoProto);
    ctx.set_process_name("nxrt");
    ctx.set_thread_name("main");
    ctx.complete(
        "MatMul_0",
        "compute",
        Instant::now(),
        Duration::from_micros(50),
        Some(Args::new().device("cpu")),
    );
    ctx.instant("checkpoint", "marker", None);

    let bytes = perfetto::to_perfetto_proto(&mem.events(), Some(ctx.session_id()));
    assert!(!bytes.is_empty(), "perfetto export must produce bytes");

    // The bytes must decode back as a valid Trace.
    let trace = Trace::decode(&bytes[..]).expect("emitted bytes must be a valid Trace protobuf");
    assert!(!trace.packet.is_empty(), "trace must contain packets");

    // At least one process track descriptor and one thread track descriptor.
    let descriptors: Vec<_> = trace
        .packet
        .iter()
        .filter_map(|p| p.track_descriptor.as_ref())
        .collect();
    assert!(
        descriptors.iter().any(|d| d.process.is_some()),
        "a process track descriptor must be present"
    );
    assert!(
        descriptors.iter().any(|d| d.thread.is_some()),
        "a thread track descriptor must be present"
    );
    // The process name we set must survive into the descriptor.
    assert!(
        descriptors.iter().any(|d| d.name.as_deref() == Some("nxrt")
            || d.process.as_ref().and_then(|p| p.process_name.as_deref()) == Some("nxrt")),
        "process_name metadata must name the process track"
    );

    // The complete event must yield a SLICE_BEGIN and a SLICE_END; the instant
    // must yield an INSTANT.
    let types: Vec<i32> = trace
        .packet
        .iter()
        .filter_map(|p| p.track_event.as_ref())
        .filter_map(|e| e.r#type)
        .collect();
    assert!(types.contains(&(TrackEventType::SliceBegin as i32)));
    assert!(types.contains(&(TrackEventType::SliceEnd as i32)));
    assert!(types.contains(&(TrackEventType::Instant as i32)));

    // The MemoryCollector's convenience export agrees.
    let via_mem = mem.to_perfetto_proto();
    assert!(!via_mem.is_empty());
}

#[test]
fn empty_perfetto_trace_still_decodes() {
    let bytes = perfetto::to_perfetto_proto(&[], None);
    // Even with no events we emit a process track descriptor, so the trace is
    // non-empty and valid.
    let trace = Trace::decode(&bytes[..]).expect("must decode");
    assert!(!trace.packet.is_empty());
}
