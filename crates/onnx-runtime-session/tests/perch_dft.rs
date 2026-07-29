//! Integration: verify the Perch model runs end-to-end on native CPU EP
//! and that the vDSP DFT fast path is exercised (macOS/iOS only).
use std::time::Instant;

#[cfg(any(target_os = "macos", target_os = "ios"))]
use onnx_runtime_ep_cpu::kernels::dft::DFT_VDSP_TEST_HITS;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::sync::atomic::Ordering;

#[test]
#[ignore = "requires models/perch-onnx/perch_v2.onnx at repo root"]
fn perch_model_runs() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = format!("{manifest}/../../models/perch-onnx/perch_v2.onnx");
    if !std::path::Path::new(&path).exists() {
        eprintln!("Skipping: model not found at {path}");
        return;
    }

    let mut session = onnx_runtime_session::load(&path).expect("load perch model");

    let input_data = vec![0.0f32; 160000];
    let input =
        onnx_runtime_session::Tensor::from_f32(&[1, 160000], &input_data).expect("create tensor");

    // Record vDSP counter before inference (macOS only).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    let vdsp_before = DFT_VDSP_TEST_HITS.load(Ordering::Relaxed);

    // Warmup
    let _ = session.run(&[("inputs", &input)]);

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let vdsp_after_warmup = DFT_VDSP_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            vdsp_after_warmup > vdsp_before,
            "vDSP DFT path was NOT exercised during Perch inference (counter stayed at {vdsp_before})"
        );
    }

    let start = Instant::now();
    let outputs = session.run(&[("inputs", &input)]).expect("run perch");
    let elapsed = start.elapsed();

    println!("Perch native CPU EP: {elapsed:.2?}");
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    println!(
        "vDSP DFT dispatches: {} (before={vdsp_before})",
        DFT_VDSP_TEST_HITS.load(Ordering::Relaxed),
    );
    assert_eq!(outputs.len(), 4, "Expected 4 outputs");
    for (i, out) in outputs.iter().enumerate() {
        println!("  output[{i}]: shape={:?}", out.shape);
    }
}
