//! CPU execution provider exported as an ORT plugin-EP cdylib.
//!
//! This crate produces `libonnx_runtime_ep_cpu_plugin.so` (or platform
//! equivalent) that upstream ONNX Runtime can load via `dlopen` and use as a
//! real execution provider.
//!
//! The crate is intentionally thin: construct the EP, invoke the macro, done.

use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_plugin::export_ep_factories;

// Generates #[unsafe(no_mangle)] CreateEpFactories and ReleaseEpFactory.
//
// Export symbol name: `CreateEpFactories` (see onnx_runtime_ep_plugin::EXPORT_SYMBOL_CREATE).
// If Challenger finds ORT 1.27 uses `CreateEpApiFactories`, update that constant.
export_ep_factories!(|| Box::new(CpuExecutionProvider::new()));
