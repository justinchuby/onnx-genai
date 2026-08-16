//! Load an arbitrary ONNX model on the native CUDA execution provider and
//! report whether the whole graph placed on CUDA.
//!
//! With `ONNX_GENAI_REQUIRE_CUDA=1` the session build fails when any node would
//! fall back to the CPU EP, and the error names the unclaimed nodes. This makes
//! it a direct op-coverage probe for the pure-Rust CUDA EP on a real device.
//!
//! Usage:
//!   ONNX_GENAI_REQUIRE_CUDA=1 CUDA_VISIBLE_DEVICES=0 \
//!     cargo run --release -p onnx-genai-bench --features cuda,bench-native \
//!     --bin cuda_place -- --model path/to/model.onnx [--device 0]

use std::path::PathBuf;

use anyhow::{Context, Result};
use onnx_runtime_session::{DevicePreference, SessionBuilder};

fn main() -> Result<()> {
    let mut model: Option<PathBuf> = None;
    let mut device: u32 = 0;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(it.next().expect("--model needs a value"))),
            "--device" => {
                device = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--device needs an integer");
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    let model = model.expect("--model is required");

    println!(
        "cuda_place: loading {} on CUDA device {device}",
        model.display()
    );
    let session = SessionBuilder::new()
        .model(&model)
        .device(DevicePreference::Gpu {
            index: Some(device),
        })
        .build()
        .with_context(|| format!("build native CUDA session for {}", model.display()))?;

    println!("cuda_place: PLACED OK (whole graph on CUDA)");
    println!("inputs:");
    for io in session.inputs() {
        println!("  {} {:?} {:?}", io.name, io.dtype, io.shape);
    }
    println!("outputs:");
    for io in session.outputs() {
        println!("  {} {:?} {:?}", io.name, io.dtype, io.shape);
    }
    Ok(())
}
