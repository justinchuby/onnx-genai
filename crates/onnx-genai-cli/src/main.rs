//! `onnx-genai` binary entry point for local development.
//!
//! All logic lives in the library crate ([`_onnx_genai_server::run`]) so it can
//! be shared with the `onnx-genai-server` wheel's PyO3 entry point. The
//! published wheel does not ship this binary; it invokes the same `run` through
//! a Python console script that first loads ONNX Runtime from the `onnxruntime`
//! wheel.
fn main() -> anyhow::Result<()> {
    _onnx_genai_server::run(std::env::args().collect())
}
