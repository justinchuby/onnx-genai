use onnx_runtime_ep_api::{TensorMut, TensorView};

#[cfg(feature = "tracing")]
use onnx_runtime_tracer::{Args, annotate_current_span_with};

#[cfg(feature = "tracing")]
#[inline]
pub(crate) fn record_kernel_metrics(
    inputs: &[TensorView<'_>],
    outputs: &[TensorMut<'_>],
    flops: impl FnOnce() -> u64,
) {
    annotate_current_span_with(|| {
        let input_bytes = inputs
            .iter()
            .filter(|input| !input.is_absent())
            .fold(0_u64, |total, input| {
                total.saturating_add(input.byte_size() as u64)
            });
        let bytes = outputs.iter().fold(input_bytes, |total, output| {
            total.saturating_add(output.byte_size() as u64)
        });
        Args::new().device("cpu").bytes(bytes).flops(flops())
    });
}

#[cfg(not(feature = "tracing"))]
#[inline]
pub(crate) fn record_kernel_metrics(
    _inputs: &[TensorView<'_>],
    _outputs: &[TensorMut<'_>],
    _flops: impl FnOnce() -> u64,
) {
}

pub(crate) fn product(values: impl IntoIterator<Item = usize>) -> u64 {
    values
        .into_iter()
        .fold(1_u64, |total, value| total.saturating_mul(value as u64))
}
