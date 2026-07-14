//! Integration tests for the collector architecture (§48.2 / §48.8.1):
//! `NoopCollector`, `MemoryCollector`, `FileCollector`, and the
//! `CompositeCollector` fan-out.

use onnx_runtime_tracer::{
    CompositeCollector, FileCollector, MemoryCollector, NoopCollector, TraceCollector,
    TraceContext, TraceEvent, TraceFormat, TracePhase,
};
use std::sync::Arc;

fn sample_event(name: &str) -> TraceEvent {
    TraceEvent {
        name: name.to_string(),
        cat: "compute".to_string(),
        ph: TracePhase::Complete,
        ts: 10,
        dur: Some(5),
        pid: 1,
        tid: 0,
        scope: None,
        args: None,
    }
}

#[test]
fn noop_collector_captures_and_emits_nothing() {
    let noop = NoopCollector;
    // Emitting is a no-op and flush is infallible; there is no observable state.
    noop.emit(&sample_event("op"));
    noop.flush().expect("noop flush is infallible");
}

#[test]
fn memory_collector_captures_in_order() {
    let mem = MemoryCollector::new();
    mem.emit(&sample_event("a"));
    mem.emit(&sample_event("b"));
    mem.emit(&sample_event("c"));

    let events = mem.events();
    let names: Vec<&str> = events.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["a", "b", "c"]);
    assert_eq!(mem.len(), 3);
    assert_eq!(mem.dropped(), 0);
}

#[test]
fn memory_collector_bounded_overflow_drops_and_counts() {
    // Capacity 2: the third event is dropped (not overwritten), and the drop is
    // counted (a single warn-once also goes to stderr).
    let mem = MemoryCollector::with_capacity(2);
    mem.emit(&sample_event("a"));
    mem.emit(&sample_event("b"));
    mem.emit(&sample_event("c"));
    mem.emit(&sample_event("d"));

    let events = mem.events();
    let names: Vec<&str> = events.iter().map(|e| e.name.as_str()).collect();
    // Earliest events are preserved.
    assert_eq!(names, ["a", "b"]);
    assert_eq!(mem.len(), 2);
    assert_eq!(mem.dropped(), 2);
    assert_eq!(mem.capacity(), 2);
}

#[test]
fn composite_fans_out_to_all_backends() {
    let a = Arc::new(MemoryCollector::new());
    let b = Arc::new(MemoryCollector::new());

    let mut composite = CompositeCollector::new();
    composite.add(Box::new(SharedMemory(a.clone())));
    composite.add(Box::new(SharedMemory(b.clone())));
    assert_eq!(composite.len(), 2);

    composite.emit(&sample_event("x"));
    composite.emit(&sample_event("y"));

    // The same events reached both backends.
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 2);
    assert_eq!(a.events()[0].name, "x");
    assert_eq!(b.events()[1].name, "y");

    composite.flush().expect("all backends flush cleanly");
}

/// A thin `TraceCollector` that forwards to a shared `MemoryCollector`, so a
/// test can hold a typed handle to what a boxed composite member captured.
struct SharedMemory(Arc<MemoryCollector>);

impl TraceCollector for SharedMemory {
    fn emit(&self, event: &TraceEvent) {
        self.0.emit(event);
    }
    fn flush(&self) -> onnx_runtime_tracer::Result<()> {
        self.0.flush()
    }
}

#[test]
fn composite_context_writes_two_files() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let chrome_path = dir.join("composite_trace.json");
    let jsonl_path = dir.join("composite_trace.jsonl");

    let chrome = FileCollector::new(&chrome_path, TraceFormat::ChromeJson).unwrap();
    let jsonl = FileCollector::new(&jsonl_path, TraceFormat::Jsonl).unwrap();

    let composite = Arc::new(
        CompositeCollector::new()
            .with(Box::new(chrome))
            .with(Box::new(jsonl)),
    );
    let ctx = TraceContext::with_collector(composite, TraceFormat::ChromeJson);
    {
        let _s = ctx.span("op_a", "compute");
    }
    {
        let _s = ctx.span("op_b", "compute");
    }
    ctx.flush().expect("both files must flush");

    // Chrome file: a JSON array of two events.
    let chrome_json = std::fs::read_to_string(&chrome_path).unwrap();
    let arr: Vec<TraceEvent> = serde_json::from_str(&chrome_json).unwrap();
    assert_eq!(arr.len(), 2);

    // JSONL file: two lines, each a valid event object.
    let jsonl_text = std::fs::read_to_string(&jsonl_path).unwrap();
    let lines: Vec<&str> = jsonl_text.lines().collect();
    assert_eq!(lines.len(), 2);
    for line in lines {
        let ev: TraceEvent = serde_json::from_str(line).unwrap();
        assert_eq!(ev.ph, TracePhase::Complete);
    }

    let _ = std::fs::remove_file(&chrome_path);
    let _ = std::fs::remove_file(&jsonl_path);
}

#[test]
fn file_collector_reports_path_on_unwritable_target() {
    // A path whose parent directory does not exist forces an I/O error at
    // construction time — and the message must name the path and say how to fix.
    let mut path = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push("definitely-missing-subdir");
    path.push("trace.json");

    let err = FileCollector::new(&path, TraceFormat::ChromeJson).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("trace.json"),
        "error must name the path, got: {message}"
    );
    assert!(
        message.contains("failed to write"),
        "error must say what failed, got: {message}"
    );
    assert!(
        message.contains("parent directory"),
        "error must offer a remediation, got: {message}"
    );
}
