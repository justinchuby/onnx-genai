use onnx_runtime_ep_api::{TensorMut, TensorView};

/// Record this kernel's device, bytes and flops on the active op span.
///
/// CUDA kernels should use the same trace vocabulary as every other execution
/// provider: executor spans describe graph nodes, while provider kernels only
/// annotate those spans with comparable device/work estimates. Keeping the
/// helper crate-local pins the device label and avoids scattering the shared
/// standard's stringly-typed entry point through hot kernel code.
#[inline]
pub(crate) fn record_kernel_metrics(
    inputs: &[TensorView<'_>],
    outputs: &[TensorMut<'_>],
    flops: impl FnOnce() -> u64,
) {
    onnx_runtime_ep_api::record_kernel_metrics("cuda", inputs, outputs, flops);
}

/// Open a span for one worker's slice of a fanned-out node (`TraceVerbosity::Full`).
///
/// CUDA work normally fans out on GPU streams/kernels rather than CPU worker
/// threads, so most call sites should not need this. It is here for the same
/// reason as the metrics wrapper: if a future host-side CUDA fallback shards one
/// graph node across a CPU pool, it can opt into the shared worker-span contract
/// without taking a direct dependency on the tracer crate.
// Kept for a planned host-side CUDA fallback that shards work across CPU workers.
#[allow(dead_code)]
#[inline]
pub(crate) fn worker_span(label: &'static str) -> Option<impl Drop> {
    onnx_runtime_ep_api::kernel_worker_span(label)
}

pub(crate) fn product(values: impl IntoIterator<Item = usize>) -> u64 {
    values
        .into_iter()
        .fold(1_u64, |total, value| total.saturating_mul(value as u64))
}
