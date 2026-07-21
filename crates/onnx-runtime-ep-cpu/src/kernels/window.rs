//! ONNX window generators: `HannWindow`, `HammingWindow`, and `BlackmanWindow`.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::dtype::{NumericElem, write_dense};

use super::{check_arity, to_dense_i64};

#[derive(Clone, Copy)]
enum WindowKind {
    Hann,
    Hamming,
    Blackman,
}

impl WindowKind {
    fn name(self) -> &'static str {
        match self {
            Self::Hann => "HannWindow",
            Self::Hamming => "HammingWindow",
            Self::Blackman => "BlackmanWindow",
        }
    }
}

pub struct HannWindowFactory;
pub struct HammingWindowFactory;
pub struct BlackmanWindowFactory;

macro_rules! impl_factory {
    ($factory:ty, $kind:expr) => {
        impl KernelFactory for $factory {
            fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
                Ok(Box::new(WindowKernel {
                    kind: $kind,
                    periodic: node.attr("periodic").and_then(|a| a.as_int()).unwrap_or(1) != 0,
                    output_dtype: node
                        .attr("output_datatype")
                        .and_then(|a| a.as_int())
                        .map(|value| DataType::from_onnx(value as i32))
                        .unwrap_or(Some(DataType::Float32)),
                }))
            }
        }
    };
}

impl_factory!(HannWindowFactory, WindowKind::Hann);
impl_factory!(HammingWindowFactory, WindowKind::Hamming);
impl_factory!(BlackmanWindowFactory, WindowKind::Blackman);

struct WindowKernel {
    kind: WindowKind,
    periodic: bool,
    output_dtype: Option<DataType>,
}

impl Kernel for WindowKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = self.kind.name();
        check_arity(op, inputs, outputs, 1, 1, 1)?;
        if !inputs[0].shape.is_empty() {
            return Err(EpError::KernelFailed(format!(
                "{op}: size input must be a scalar"
            )));
        }
        let size = to_dense_i64(&inputs[0])?[0];
        let size = usize::try_from(size)
            .map_err(|_| EpError::KernelFailed(format!("{op}: size must be non-negative")))?;
        if outputs[0].shape != [size] {
            return Err(EpError::KernelFailed(format!(
                "{op}: output shape must be [{size}]"
            )));
        }
        let output_dtype = self.output_dtype.ok_or_else(|| {
            EpError::KernelFailed(format!("{op}: unsupported output_datatype attribute"))
        })?;
        if outputs[0].dtype != output_dtype {
            return Err(EpError::KernelFailed(format!(
                "{op}: output dtype {:?} does not match output_datatype {output_dtype:?}",
                outputs[0].dtype
            )));
        }

        if output_dtype == DataType::Float64 {
            let values = generate_f64(self.kind, size, self.periodic);
            return write_dense::<f64>(&mut outputs[0], &values);
        }
        crate::dispatch_arith!(output_dtype, op, T => {
            let values = generate_f32(self.kind, size, self.periodic)
                .into_iter()
                .map(T::from_f32_scalar)
                .collect::<Vec<_>>();
            write_dense::<T>(&mut outputs[0], &values)
        })
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

fn generate_f32(kind: WindowKind, size: usize, periodic: bool) -> Vec<f32> {
    let denominator = if periodic {
        size as f32
    } else {
        size as f32 - 1.0
    };
    (0..size)
        .map(|n| {
            let n = n as f32;
            match kind {
                WindowKind::Hann => (n * std::f32::consts::PI / denominator).sin().powi(2),
                WindowKind::Hamming => {
                    let alpha = 25.0_f32 / 46.0;
                    alpha - (n * std::f32::consts::PI * 2.0 / denominator).cos() * (1.0 - alpha)
                }
                WindowKind::Blackman => {
                    let mut value = (n * (std::f32::consts::PI * 2.0) / denominator).cos() * -0.5;
                    value += (n * (std::f32::consts::PI * 4.0) / denominator).cos() * 0.08;
                    value += 0.42;
                    value
                }
            }
        })
        .collect()
}

fn generate_f64(kind: WindowKind, size: usize, periodic: bool) -> Vec<f64> {
    let denominator = if periodic {
        size as f64
    } else {
        size as f64 - 1.0
    };
    (0..size)
        .map(|n| {
            let n = n as f64;
            match kind {
                WindowKind::Hann => (n * std::f64::consts::PI / denominator).sin().powi(2),
                WindowKind::Hamming => {
                    let alpha = 25.0_f64 / 46.0;
                    alpha - (n * std::f64::consts::PI * 2.0 / denominator).cos() * (1.0 - alpha)
                }
                WindowKind::Blackman => {
                    let mut value = (n * (std::f64::consts::PI * 2.0) / denominator).cos() * -0.5;
                    value += (n * (std::f64::consts::PI * 4.0) / denominator).cos() * 0.08;
                    value += 0.42;
                    value
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }

    fn run(kind: WindowKind, periodic: bool) -> Vec<f32> {
        let size = Owned::i64(&[], &[5]);
        let mut output = Owned::zeros_f32(&[5]);
        WindowKernel {
            kind,
            periodic,
            output_dtype: Some(DataType::Float32),
        }
        .execute(&[size.view()], &mut [output.view_mut()])
        .unwrap();
        output.to_f32()
    }

    #[test]
    fn hann_periodic_and_symmetric_match_onnx_reference() {
        assert_close(
            &run(WindowKind::Hann, true),
            &[0.0, 0.34549153, 0.90450853, 0.904_508_5, 0.34549144],
        );
        assert_close(
            &run(WindowKind::Hann, false),
            &[0.0, 0.5, 1.0, 0.5, 7.642743e-15],
        );
    }

    #[test]
    fn hamming_periodic_and_symmetric_match_onnx_reference() {
        assert_close(
            &run(WindowKind::Hamming, true),
            &[0.08695652, 0.4024053, 0.9128121, 0.9128121, 0.40240523],
        );
        assert_close(
            &run(WindowKind::Hamming, false),
            &[0.08695652, 0.5434783, 1.0, 0.54347825, 0.08695652],
        );
    }

    #[test]
    fn blackman_periodic_and_symmetric_match_onnx_reference() {
        assert_close(
            &run(WindowKind::Blackman, true),
            &[0.0, 0.20077014, 0.8492299, 0.8492299, 0.20077014],
        );
        assert_close(
            &run(WindowKind::Blackman, false),
            &[0.0, 0.34, 1.0, 0.34, 0.0],
        );
    }
}
