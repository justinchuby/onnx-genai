//! CUDA execution provider exported as an ORT plugin-EP cdylib.
//!
//! This crate mirrors `onnx-runtime-ep-cpu-plugin` for the CUDA EP.
//! Without the `cuda` feature (default), it compiles as a no-op stub so the
//! workspace builds on hosts without a CUDA toolkit.
//!
//! With `cuda` enabled and `onnx-runtime-ep-cuda` linked, it exports
//! `CreateEpFactories` and `ReleaseEpFactory` that project the CUDA EP
//! through the ORT plugin-EP C ABI.

#[cfg(feature = "cuda")]
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_plugin::export_ep_factories;

#[cfg(feature = "cuda")]
export_ep_factories!(|| CudaExecutionProvider::new());

// When built without `cuda` feature: export stubs that return zero factories
// so ORT's dlopen path gets a clean "no providers" response rather than a
// missing-symbol crash.
#[cfg(not(feature = "cuda"))]
export_ep_factories!(|| {
    panic!(
        "onnx-runtime-ep-cuda-plugin built without `cuda` feature; \
         CUDA EP is not available on this host"
    )
});
