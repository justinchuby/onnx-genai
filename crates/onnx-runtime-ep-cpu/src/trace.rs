use onnx_runtime_ep_api::{TensorMut, TensorView};

/// Record this kernel's device, bytes and flops on the active op span.
///
/// Delegates to the shared implementation in `onnx-runtime-ep-api` so every
/// provider reports the same keys with the same meaning; this wrapper only
/// pins the device name and keeps the call sites short.
#[cfg(feature = "tracing")]
#[inline]
pub(crate) fn record_kernel_metrics(
    inputs: &[TensorView<'_>],
    outputs: &[TensorMut<'_>],
    flops: impl FnOnce() -> u64,
) {
    onnx_runtime_ep_api::record_kernel_metrics("cpu", inputs, outputs, flops);
}

#[cfg(not(feature = "tracing"))]
#[inline]
pub(crate) fn record_kernel_metrics(
    _inputs: &[TensorView<'_>],
    _outputs: &[TensorMut<'_>],
    _flops: impl FnOnce() -> u64,
) {
}

/// Open a span for one worker's slice of a fanned-out node (`TraceVerbosity::Full`).
///
/// Returns `None` when tracing is off or below `Full`, so the decode path pays
/// one relaxed atomic load. See `onnx_runtime_ep_api::kernel_worker_span` for
/// why this is a separate span rather than an annotation.
#[cfg(feature = "tracing")]
#[inline]
pub(crate) fn worker_span(label: &'static str) -> Option<onnx_runtime_tracer::SpanGuard> {
    onnx_runtime_ep_api::kernel_worker_span(label)
}

#[cfg(not(feature = "tracing"))]
#[inline]
pub(crate) fn worker_span(_label: &'static str) -> Option<()> {
    None
}

pub(crate) fn product(values: impl IntoIterator<Item = usize>) -> u64 {
    values
        .into_iter()
        .fold(1_u64, |total, value| total.saturating_mul(value as u64))
}
