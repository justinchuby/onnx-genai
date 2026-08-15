//! Integration tests for the Chrome Trace export path via the collector
//! architecture.
//!
//! These exercise the public API the way a runtime would: build spans across
//! multiple threads through a shared [`TraceContext`], capture into a
//! [`MemoryCollector`], export to Chrome Trace JSON, and validate the exported
//! document parses back and is schema-correct. Timing assertions are
//! deliberately robust — we assert ordering and presence, never exact
//! durations.

use onnx_runtime_tracer::{
    Args, MemoryCollector, TraceContext, TraceEvent, TraceFormat, TracePhase,
    annotate_current_span, annotate_current_span_with, tracing_active,
};
use serde_json::Value;
use std::cell::Cell;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Parse the exported JSON and assert it is a Chrome Trace array of objects,
/// each carrying the required keys. Returns the parsed events.
fn parse_and_validate(json: &str) -> Vec<Value> {
    let value: Value = serde_json::from_str(json).expect("exported trace must be valid JSON");
    let array = value
        .as_array()
        .expect("a Chrome Trace document is a top-level JSON array")
        .clone();
    for event in &array {
        let obj = event.as_object().expect("each event is a JSON object");
        for key in ["name", "cat", "ph", "ts", "pid", "tid"] {
            assert!(
                obj.contains_key(key),
                "event missing required key {key:?}: {obj:?}"
            );
        }
        let ph = obj["ph"].as_str().expect("ph is a string");
        assert!(
            TracePhase::from_code(ph).is_some(),
            "unknown phase code {ph:?}"
        );
    }
    array
}

#[test]
fn complete_event_has_expected_shape() {
    let (ctx, mem) = TraceContext::in_memory();
    let start = Instant::now();
    ctx.complete(
        "MatMul_0",
        "compute",
        start,
        Duration::from_micros(50),
        Some(
            Args::new()
                .device("cuda:0")
                .shapes(vec![vec![1_i64, 4096, 4096]]),
        ),
    );

    let json = mem.to_chrome_json();
    let events = parse_and_validate(&json);
    assert_eq!(events.len(), 1);

    let event = &events[0];
    assert_eq!(event["name"], "MatMul_0");
    assert_eq!(event["cat"], "compute");
    assert_eq!(event["ph"], "X");
    assert_eq!(event["dur"], 50);
    assert_eq!(event["args"]["device"], "cuda:0");
    assert_eq!(event["args"]["shapes"][0][0], 1);
    assert_eq!(event["args"]["shapes"][0][1], 4096);
}

#[test]
fn round_trips_through_serde() {
    let (ctx, mem) = TraceContext::in_memory();
    let start = Instant::now();
    ctx.complete(
        "transfer",
        "transfer",
        start,
        Duration::from_micros(200),
        Some(Args::new().bytes(16_384).src("cuda:0").dst("cpu")),
    );

    let json = mem.to_chrome_json();
    let parsed: Vec<TraceEvent> =
        serde_json::from_str(&json).expect("events must deserialize back");
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].name, "transfer");
    assert_eq!(parsed[0].ph, TracePhase::Complete);
    assert_eq!(parsed[0].dur, Some(200));
    let args = parsed[0].args.as_ref().expect("args preserved");
    assert_eq!(args["bytes"], 16_384);
    assert_eq!(args["src"], "cuda:0");
    assert_eq!(args["dst"], "cpu");
}

#[test]
fn span_guard_records_on_drop_with_monotonic_ts_and_dur() {
    let (ctx, mem) = TraceContext::in_memory();
    {
        let _s = ctx.span("layer_norm", "compute");
        thread::sleep(Duration::from_millis(2));
    }

    {
        let _s = ctx
            .span("softmax", "compute")
            .with_args(Args::new().device("cpu"));
        thread::sleep(Duration::from_millis(2));
    }

    let events = mem.events();
    assert_eq!(events.len(), 2);
    for event in &events {
        assert_eq!(event.ph, TracePhase::Complete);
        assert!(event.dur.is_some());
    }
    assert!(events[0].ts <= events[1].ts);
    assert_eq!(events[1].args.as_ref().unwrap()["device"], "cpu");
}

#[test]
fn active_span_accepts_kernel_metrics() {
    let (ctx, mem) = TraceContext::in_memory();
    {
        let _span = ctx
            .span("MatMul", "compute")
            .with_args(Args::new().device("cpu"));
        assert!(tracing_active());
        annotate_current_span(Args::new().bytes(96).flops(48));
    }

    let events = mem.events();
    let args = events[0].args.as_ref().expect("span args");
    assert_eq!(args["device"], "cpu");
    assert_eq!(args["bytes"], 96);
    assert_eq!(args["flops"], 48);
}

#[test]
fn inactive_span_skips_lazy_kernel_metrics() {
    assert!(!tracing_active());
    let formula_calls = Cell::new(0);
    annotate_current_span_with(|| {
        formula_calls.set(formula_calls.get() + 1);
        Args::new().bytes(96).flops(48)
    });
    assert_eq!(formula_calls.get(), 0);
}

#[test]
fn explicit_finish_records_before_drop() {
    let (ctx, mem) = TraceContext::in_memory();
    let span = ctx.span("prefill", "compute");
    span.finish();
    assert_eq!(mem.len(), 1);
    assert_eq!(mem.events()[0].name, "prefill");
}

#[test]
fn multiple_threads_get_distinct_tids() {
    let (ctx, mem) = TraceContext::in_memory();
    let mut handles = Vec::new();
    for i in 0..4 {
        let c = ctx.clone();
        handles.push(thread::spawn(move || {
            let _s = c.span(format!("op_{i}"), "compute");
            thread::sleep(Duration::from_millis(1));
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let events = mem.events();
    assert_eq!(events.len(), 4);

    let json = mem.to_chrome_json();
    let parsed = parse_and_validate(&json);
    assert_eq!(parsed.len(), 4);

    let pid = ctx.pid();
    for event in &events {
        assert_eq!(event.pid, pid);
    }

    let mut tids: Vec<u64> = events.iter().map(|e| e.tid).collect();
    tids.sort_unstable();
    tids.dedup();
    assert_eq!(tids.len(), 4, "each thread must get a distinct tid");
}

#[test]
fn disabled_context_emits_nothing() {
    let mem = Arc::new(MemoryCollector::new());
    let ctx = TraceContext::with_collector(mem.clone(), TraceFormat::ChromeJson);
    ctx.set_enabled(false);

    let start = Instant::now();
    ctx.complete("x", "compute", start, Duration::from_micros(10), None);
    ctx.instant("marker", "compute", None);
    {
        let _s = ctx.span("y", "compute");
    }
    ctx.set_process_name("nxrt");

    assert!(mem.is_empty());
    assert_eq!(mem.to_chrome_json(), "[]");
}

#[test]
fn noop_context_records_nothing() {
    let ctx = TraceContext::noop();
    assert!(!ctx.is_enabled());
    ctx.complete(
        "x",
        "compute",
        Instant::now(),
        Duration::from_micros(1),
        None,
    );
    {
        let _s = ctx.span("y", "compute");
    }
    // Even when force-enabled, the NoopCollector discards everything.
    ctx.set_enabled(true);
    {
        let _s = ctx.span("z", "compute");
    }
    ctx.flush().expect("noop flush is infallible");
}

#[test]
fn enable_flag_can_be_toggled_at_runtime() {
    let mem = Arc::new(MemoryCollector::new());
    let ctx = TraceContext::with_collector(mem.clone(), TraceFormat::ChromeJson);
    ctx.set_enabled(false);
    {
        let _s = ctx.span("ignored", "compute");
    }
    assert!(mem.is_empty());

    ctx.set_enabled(true);
    {
        let _s = ctx.span("recorded", "compute");
    }
    assert_eq!(mem.len(), 1);
    assert_eq!(mem.events()[0].name, "recorded");

    ctx.set_enabled(false);
    {
        let _s = ctx.span("ignored_again", "compute");
    }
    assert_eq!(mem.len(), 1);
}

#[test]
fn metadata_events_name_process_and_thread() {
    let (ctx, mem) = TraceContext::in_memory();
    ctx.set_process_name("nxrt");
    ctx.set_thread_name("main");
    ctx.complete(
        "op",
        "compute",
        Instant::now(),
        Duration::from_micros(5),
        None,
    );

    let events = mem.events();
    let metadata: Vec<&TraceEvent> = events
        .iter()
        .filter(|e| e.ph == TracePhase::Metadata)
        .collect();
    assert_eq!(metadata.len(), 2);

    let names: Vec<&str> = metadata.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"process_name"));
    assert!(names.contains(&"thread_name"));

    let process = metadata.iter().find(|e| e.name == "process_name").unwrap();
    assert_eq!(process.args.as_ref().unwrap()["name"], "nxrt");

    parse_and_validate(&mem.to_chrome_json());
}

#[test]
fn instant_event_carries_thread_scope() {
    let (ctx, mem) = TraceContext::in_memory();
    ctx.instant("checkpoint", "marker", Some(Args::new().with("step", 3)));

    let events = mem.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].ph, TracePhase::Instant);
    assert_eq!(events[0].scope, Some('t'));
    assert_eq!(events[0].args.as_ref().unwrap()["step"], 3);

    let json = mem.to_chrome_json();
    let value: Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value[0]["s"], "t");
}

#[test]
fn empty_collector_exports_valid_empty_array() {
    let mem = MemoryCollector::new();
    let json = mem.to_chrome_json();
    assert_eq!(json, "[]");
    let events = parse_and_validate(&json);
    assert!(events.is_empty());
}

#[test]
fn clear_drops_events_but_keeps_context_usable() {
    let (ctx, mem) = TraceContext::in_memory();
    ctx.complete(
        "a",
        "compute",
        Instant::now(),
        Duration::from_micros(1),
        None,
    );
    assert_eq!(mem.len(), 1);
    mem.clear();
    assert!(mem.is_empty());
    ctx.complete(
        "b",
        "compute",
        Instant::now(),
        Duration::from_micros(1),
        None,
    );
    assert_eq!(mem.len(), 1);
    assert_eq!(mem.events()[0].name, "b");
}
