//! Integration tests for `onnx-runtime-tracer`.
//!
//! These exercise the public API the way a runtime would: build spans across
//! multiple threads, export to Chrome Trace JSON, and validate the exported
//! document parses back and is schema-correct. Timing assertions are
//! deliberately robust — we assert ordering and presence, never exact
//! durations.

use onnx_runtime_tracer::{Args, Event, Phase, Tracer};
use serde_json::Value;
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
            assert!(obj.contains_key(key), "event missing required key {key:?}: {obj:?}");
        }
        // `ph` must be a known single-character phase code.
        let ph = obj["ph"].as_str().expect("ph is a string");
        assert!(
            Phase::from_code(ph).is_some(),
            "unknown phase code {ph:?}"
        );
    }
    array
}

#[test]
fn complete_event_has_expected_shape() {
    let tracer = Tracer::new();
    let start = Instant::now();
    tracer.complete(
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

    let json = tracer.to_chrome_json();
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
    let tracer = Tracer::new();
    let start = Instant::now();
    tracer.complete(
        "transfer",
        "transfer",
        start,
        Duration::from_micros(200),
        Some(Args::new().bytes(16_384).src("cuda:0").dst("cpu")),
    );

    let json = tracer.to_chrome_json();
    let parsed: Vec<Event> = serde_json::from_str(&json).expect("events must deserialize back");
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].name, "transfer");
    assert_eq!(parsed[0].ph, Phase::Complete);
    assert_eq!(parsed[0].dur, Some(200));
    let args = parsed[0].args.as_ref().expect("args preserved");
    assert_eq!(args["bytes"], 16_384);
    assert_eq!(args["src"], "cuda:0");
    assert_eq!(args["dst"], "cpu");
}

#[test]
fn span_guard_records_on_drop_with_monotonic_ts_and_dur() {
    let tracer = Tracer::new();
    {
        let _s = tracer.span("layer_norm", "compute");
        thread::sleep(Duration::from_millis(2));
    }
    {
        let _s = tracer
            .span("softmax", "compute")
            .with_args(Args::new().device("cpu"));
        thread::sleep(Duration::from_millis(2));
    }

    let events = tracer.events();
    assert_eq!(events.len(), 2);
    // Both are complete events with a recorded duration.
    for event in &events {
        assert_eq!(event.ph, Phase::Complete);
        assert!(event.dur.is_some());
    }
    // Timestamps are monotonically non-decreasing in record order.
    assert!(events[0].ts <= events[1].ts);
    // The second span's args survived.
    assert_eq!(events[1].args.as_ref().unwrap()["device"], "cpu");
}

#[test]
fn explicit_finish_records_before_drop() {
    let tracer = Tracer::new();
    let span = tracer.span("prefill", "compute");
    span.finish();
    assert_eq!(tracer.len(), 1);
    assert_eq!(tracer.events()[0].name, "prefill");
}

#[test]
fn multiple_threads_get_distinct_tids() {
    let tracer = Tracer::new();
    let mut handles = Vec::new();
    for i in 0..4 {
        let t = tracer.clone();
        handles.push(thread::spawn(move || {
            let _s = t.span(format!("op_{i}"), "compute");
            thread::sleep(Duration::from_millis(1));
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let events = tracer.events();
    assert_eq!(events.len(), 4);

    let json = tracer.to_chrome_json();
    let parsed = parse_and_validate(&json);
    assert_eq!(parsed.len(), 4);

    // Every event shares the one process id.
    let pid = tracer.pid();
    for event in &events {
        assert_eq!(event.pid, pid);
    }

    // Four threads each recorded once, so we see four distinct tids.
    let mut tids: Vec<u64> = events.iter().map(|e| e.tid).collect();
    tids.sort_unstable();
    tids.dedup();
    assert_eq!(tids.len(), 4, "each thread must get a distinct tid");
}

#[test]
fn disabled_tracer_emits_nothing() {
    let tracer = Tracer::disabled();
    let start = Instant::now();
    tracer.complete("x", "compute", start, Duration::from_micros(10), None);
    tracer.instant("marker", "compute", None);
    {
        let _s = tracer.span("y", "compute");
    }
    tracer.set_process_name("nxrt");

    assert!(tracer.is_empty());
    assert_eq!(tracer.to_chrome_json(), "[]");
}

#[test]
fn enable_flag_can_be_toggled_at_runtime() {
    let tracer = Tracer::disabled();
    {
        let _s = tracer.span("ignored", "compute");
    }
    assert!(tracer.is_empty());

    tracer.set_enabled(true);
    {
        let _s = tracer.span("recorded", "compute");
    }
    assert_eq!(tracer.len(), 1);
    assert_eq!(tracer.events()[0].name, "recorded");

    tracer.set_enabled(false);
    {
        let _s = tracer.span("ignored_again", "compute");
    }
    assert_eq!(tracer.len(), 1);
}

#[test]
fn metadata_events_name_process_and_thread() {
    let tracer = Tracer::new();
    tracer.set_process_name("nxrt");
    tracer.set_thread_name("main");
    tracer.complete(
        "op",
        "compute",
        Instant::now(),
        Duration::from_micros(5),
        None,
    );

    let events = tracer.events();
    let metadata: Vec<&Event> = events
        .iter()
        .filter(|e| e.ph == Phase::Metadata)
        .collect();
    assert_eq!(metadata.len(), 2);

    let names: Vec<&str> = metadata.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"process_name"));
    assert!(names.contains(&"thread_name"));

    let process = metadata.iter().find(|e| e.name == "process_name").unwrap();
    assert_eq!(process.args.as_ref().unwrap()["name"], "nxrt");

    // Metadata events still validate as part of the exported array.
    parse_and_validate(&tracer.to_chrome_json());
}

#[test]
fn instant_event_carries_thread_scope() {
    let tracer = Tracer::new();
    tracer.instant("checkpoint", "marker", Some(Args::new().with("step", 3)));

    let events = tracer.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].ph, Phase::Instant);
    assert_eq!(events[0].scope, Some('t'));
    assert_eq!(events[0].args.as_ref().unwrap()["step"], 3);

    // The instant scope surfaces as the Chrome `s` field.
    let json = tracer.to_chrome_json();
    let value: Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value[0]["s"], "t");
}

#[test]
fn empty_tracer_exports_valid_empty_array() {
    let tracer = Tracer::new();
    let json = tracer.to_chrome_json();
    assert_eq!(json, "[]");
    let events = parse_and_validate(&json);
    assert!(events.is_empty());
}

#[test]
fn write_chrome_json_round_trips_from_disk() {
    let tracer = Tracer::new();
    tracer.complete(
        "op",
        "compute",
        Instant::now(),
        Duration::from_micros(7),
        Some(Args::new().device("cpu")),
    );

    // Use the cargo-provided temp dir, never /tmp.
    let mut path = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push("tracer_write_round_trip.json");
    tracer.write_chrome_json(&path).expect("write should succeed");

    let json = std::fs::read_to_string(&path).unwrap();
    let events = parse_and_validate(&json);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["args"]["device"], "cpu");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn write_chrome_json_reports_path_on_failure() {
    let tracer = Tracer::new();
    // A path whose parent directory does not exist forces an I/O error.
    let mut path = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push("definitely-missing-subdir");
    path.push("trace.json");

    let err = tracer.write_chrome_json(&path).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("trace.json"),
        "error message must name the path, got: {message}"
    );
    assert!(
        message.contains("failed to write"),
        "error message must say what failed, got: {message}"
    );
}

#[test]
fn clear_drops_events_but_keeps_timeline() {
    let tracer = Tracer::new();
    tracer.complete("a", "compute", Instant::now(), Duration::from_micros(1), None);
    assert_eq!(tracer.len(), 1);
    tracer.clear();
    assert!(tracer.is_empty());
    // Still usable after a clear.
    tracer.complete("b", "compute", Instant::now(), Duration::from_micros(1), None);
    assert_eq!(tracer.len(), 1);
    assert_eq!(tracer.events()[0].name, "b");
}
