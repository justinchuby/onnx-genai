use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView, det};
use onnx_runtime_ir::Node;

use crate::dtype::{NumericElem, to_dense, write_dense};

pub struct Factory;

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Det(node.version.unwrap_or(11))))
    }
}

struct Det(i64);

trait DetElem: NumericElem {
    fn widen(self) -> f64;
    fn narrow(value: f64) -> Self;
}

macro_rules! det_elem {
    ($ty:ty, $widen:expr, $narrow:expr) => {
        impl DetElem for $ty {
            fn widen(self) -> f64 {
                ($widen)(self)
            }
            fn narrow(value: f64) -> Self {
                ($narrow)(value)
            }
        }
    };
}
det_elem!(f64, |v| v, |v| v);
det_elem!(f32, f64::from, |v| v as f32);
det_elem!(half::f16, half::f16::to_f64, half::f16::from_f64);
det_elem!(half::bf16, half::bf16::to_f64, half::bf16::from_f64);

impl Kernel for Det {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let geometry = det::validate(inputs, outputs, self.0)?;
        let input = &inputs[0];
        let output = &mut outputs[0];
        if !input.device.is_host_accessible() || !output.device.is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "CPU Det: expected host-accessible input and output, got {:?} and {:?}; explicitly transfer to CPU",
                input.device, output.device
            )));
        }
        crate::dispatch_float!(input.dtype, "Det", T => run::<T>(input, output, geometry))
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

fn run<T: DetElem>(
    input: &TensorView,
    output: &mut TensorMut,
    geometry: det::Geometry,
) -> Result<()> {
    match (geometry.batch, geometry.order) {
        (0, _) => write_dense::<T>(output, &[]),
        (batch, 0) => write_dense(output, &vec![T::narrow(1.0); batch]),
        (_, 1) => write_dense(output, &to_dense::<T>(input)?),
        (batch, n) => {
            let dense = to_dense::<T>(input)?;
            let mut work = vec![0.0; geometry.matrix_elements];
            let mut result = Vec::with_capacity(batch);
            for matrix in dense.chunks_exact(geometry.matrix_elements) {
                for (dst, &src) in work.iter_mut().zip(matrix) {
                    *dst = src.widen();
                }
                result.push(T::narrow(determinant(&mut work, n)));
            }
            write_dense(output, &result)
        }
    }
}

fn determinant(a: &mut [f64], n: usize) -> f64 {
    if a.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    // Keep an independent exponent so intermediate pivot products cannot
    // overflow or underflow when the final result is representable.
    let (mut mantissa, mut exponent) = (1.0_f64, 0_i64);
    for k in 0..n {
        let mut pivot = k;
        let mut max = a[k * n + k].abs();
        for r in k + 1..n {
            let candidate = a[r * n + k].abs();
            if candidate.is_nan() {
                return f64::NAN;
            }
            if candidate > max {
                pivot = r;
                max = candidate;
            }
        }
        if max.is_nan() {
            return f64::NAN;
        }
        if max == 0.0 {
            return if mantissa.is_infinite() {
                f64::NAN
            } else {
                0.0
            };
        }
        if pivot != k {
            for c in k..n {
                a.swap(k * n + c, pivot * n + c);
            }
            mantissa = -mantissa;
        }
        let diagonal = a[k * n + k];
        if diagonal.is_infinite() {
            mantissa *= diagonal;
        } else {
            let (fraction, power) = libm::frexp(diagonal);
            let (normalized, correction) = libm::frexp(mantissa * fraction);
            mantissa = normalized;
            exponent += i64::from(power) + i64::from(correction);
        }
        for r in k + 1..n {
            let factor = a[r * n + k] / diagonal;
            if factor == 0.0 {
                continue;
            }
            for c in k + 1..n {
                a[r * n + c] -= factor * a[k * n + c];
            }
        }
    }
    libm::scalbn(mantissa, exponent.clamp(-4096, 4096) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuExecutionProvider, kernels::testutil::Owned};
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{DataType, NodeId, ValueId, static_shape};

    fn node() -> Node {
        Node::new(NodeId(0), "Det", vec![Some(ValueId(0))], vec![ValueId(1)])
    }

    fn execute(input: &Owned, dtype: DataType, shape: &[usize], opset: u64) -> Owned {
        let ep = CpuExecutionProvider::new();
        let view = input.view();
        let claim = ep.supports_op(
            &node(),
            opset,
            &[static_shape(view.shape.iter().copied())],
            &[dtype],
            &[],
        );
        assert!(claim.is_supported(), "{:?}", claim.reason());
        let kernel = ep
            .get_kernel(&node(), &[view.shape.to_vec()], opset)
            .unwrap();
        let mut output = Owned::zeros(dtype, shape);
        kernel.execute(&[view], &mut [output.view_mut()]).unwrap();
        output
    }

    #[test]
    fn native_det_cpu() {
        use DataType::*;
        // Pinned ONNX test_det_2d and test_det_nd.
        let x = Owned::f32(&[2, 2], &[0., 1., 2., 3.]);
        assert_eq!(execute(&x, Float32, &[], 11).to_f32(), [-2.]);
        let x = Owned::f32(
            &[3, 2, 2],
            &[1., 2., 3., 4., 1., 2., 2., 1., 1., 3., 3., 1.],
        );
        let y = execute(&x, Float32, &[3], 22);
        for (got, want) in y.to_f32().into_iter().zip([-2., -3., -8.]) {
            assert!((got - want).abs() <= 1e-6);
        }

        let values = [0., 2., 3., 0., 1., 2., 1., 2.];
        let x = Owned::f16(&[1, 2, 2, 2], &values);
        assert_eq!(execute(&x, Float16, &[1, 2], 21).to_f16_as_f32(), [-6., 0.]);
        let x = Owned::bf16(&[1, 2, 2, 2], &values);
        assert_eq!(
            execute(&x, BFloat16, &[1, 2], 22).to_bf16_as_f32(),
            [-6., 0.]
        );

        let mut diag = [0.; 16];
        for (i, v) in [1e200, 1e200, 1e-200, 1e-200].into_iter().enumerate() {
            diag[i * 5] = v;
        }
        let x = Owned::f64(&[4, 4], &diag);
        let result = execute(&x, Float64, &[], 11).to_f64()[0];
        assert!((result - 1.0).abs() < 1e-14);
        let x = Owned::f32(&[2, 0, 0], &[]);
        assert_eq!(execute(&x, Float32, &[2], 22).to_f32(), [1., 1.]);
        let x = Owned::f32(&[0, 3, 3], &[]);
        assert!(execute(&x, Float32, &[0], 22).to_f32().is_empty());
    }

    #[test]
    fn native_det_cpu_rejects_bad_schema_and_views() {
        use DataType::*;
        let ep = CpuExecutionProvider::new();
        for (opset, dtype, shape, reason) in [
            (10, Float32, vec![2, 2], "since opset 11"),
            (21, BFloat16, vec![2, 2], "requires opset 22"),
            (22, Int32, vec![2, 2], "Int32"),
            (22, Float32, vec![2], "rank"),
            (22, Float32, vec![2, 3], "square"),
        ] {
            let claim = ep.supports_op(&node(), opset, &[static_shape(shape)], &[dtype], &[]);
            assert!(
                claim.reason().is_some_and(|s| s.contains(reason)),
                "{claim:?}"
            );
        }
        let kernel = ep.get_kernel(&node(), &[vec![2, 2]], 21).unwrap();
        let x = Owned::bf16(&[2, 2], &[0., 1., 2., 3.]);
        let mut y = Owned::zeros(BFloat16, &[]);
        let err = kernel
            .execute(&[x.view()], &mut [y.view_mut()])
            .unwrap_err();
        assert!(err.to_string().contains("requires opset 22"), "{err}");

        let x = Owned::f32(&[2, 2], &[0., 1., 2., 3.]);
        let mut y = Owned::zeros(Float32, &[1]);
        let err = kernel
            .execute(&[x.view()], &mut [y.view_mut()])
            .unwrap_err();
        assert!(err.to_string().contains("batch shape"), "{err}");
    }
}
