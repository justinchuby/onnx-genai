//! Live dlopen smoke test (ignored by default; run explicitly). Proves that
//! CuptiProfiler::new() dlopens libcupti and enable/flush do not crash.
#![cfg(feature = "cupti")]

use onnx_runtime_tracer::cupti::CuptiProfiler;

#[test]
#[ignore]
fn live_dlopen_and_flush() {
    let p = CuptiProfiler::new().expect("infallible");
    eprintln!("cupti available = {}", p.available());
    p.start_activity_tracing().expect("start ok");
    let records = p.stop_and_flush().expect("flush ok");
    eprintln!("drained {} kernel records", records.len());
}
