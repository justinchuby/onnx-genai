use crate::dtype::{to_dense, to_dense_f32_widen, write_dense, write_dense_f32_narrow};
use onnx_runtime_ep_api::standard_math::{Scan, validate};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, standard_math::Math};

pub(super) struct Factory;

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(MathKernel(
            Math::from_node(node).map_err(EpError::KernelFailed)?,
        )))
    }
}

struct MathKernel(Math);

macro_rules! evaluate {
    ($name:ident, $float:ty, $erf:path) => {
        fn $name(
            op: Math,
            data: &mut [$float],
            b: &[$float],
            scan: Option<Scan>,
            round: impl Fn($float) -> $float,
        ) {
            match op {
                Math::Expm1 => data.iter_mut().for_each(|x| *x = x.exp_m1()),
                Math::Log1p => data.iter_mut().for_each(|x| *x = x.ln_1p()),
                Math::SwiGlu { alpha } => {
                    let alpha = round(alpha as $float);
                    for (a, &b) in data.iter_mut().zip(b) {
                        let z = round(alpha * *a);
                        let e = (-z.abs()).exp();
                        let sigmoid = round(if z >= 0. { 1. / (1. + e) } else { e / (1. + e) });
                        *a = round(*a * sigmoid) * b;
                    }
                }
                Math::GeGlu { tanh } => {
                    for (a, &b) in data.iter_mut().zip(b) {
                        let x = *a;
                        let v = if tanh {
                            ((0.797_884_560_802_865_4 as $float)
                                * (x + (0.044715 as $float) * x * x * x))
                                .tanh()
                        } else {
                            $erf(x / (std::f64::consts::SQRT_2 as $float))
                        };
                        *a = (0.5 * x * (1. + v)) * b;
                    }
                }
                Math::LogCumSumExp { .. } => {
                    let Scan {
                        lanes,
                        length,
                        inner,
                    } = scan.expect("validated scan axis");
                    for lane in 0..lanes {
                        let start = (lane / inner) * length * inner + lane % inner;
                        let mut max = <$float>::NEG_INFINITY;
                        // Keep the leading 1 separate so log1p retains small prefixes.
                        let mut rest = 0.;
                        for k in 0..length {
                            let x = &mut data[start + k * inner];
                            if x.is_nan() {
                                max = *x;
                            } else if !max.is_nan() {
                                if *x == <$float>::INFINITY {
                                    max = *x;
                                } else if x.is_finite() && max != <$float>::INFINITY {
                                    if *x > max {
                                        rest = (1. + rest) * (max - *x).exp();
                                        max = *x;
                                    } else {
                                        rest += (*x - max).exp();
                                    }
                                }
                            }
                            *x = if max.is_finite() {
                                max + rest.ln_1p()
                            } else {
                                max
                            };
                        }
                    }
                }
            }
        }
    };
}
evaluate!(eval_f32, f32, libm::erff);
evaluate!(eval_f64, f64, libm::erf);

impl Kernel for MathKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let scan = validate(self.0, inputs, outputs)?;
        let op = self.0.name();
        let x = &inputs[0];
        if x.numel() == 0 {
            return Ok(());
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use crate::kernels::testutil::Owned;

            #[test]
            fn scan_preserves_small_prefix_and_handles_interleaved_lanes() {
                let x = Owned::f32(&[2, 2], &[0., 1000., -30., 1000.]);
                let mut y = Owned::zeros_f32(&[2, 2]);
                MathKernel(Math::LogCumSumExp { axis: -2 })
                    .execute(&[x.view()], &mut [y.view_mut()])
                    .unwrap();
                let y = y.to_f32();
                assert_eq!(&y[..2], &[0., 1000.]);
                assert!((y[2] - 9.357_623e-14).abs() < 1e-19);
                assert!((y[3] - 1000.693_2).abs() < 1e-4);
            }

            #[test]
            fn gated_activation_rejects_broadcasting_before_writing() {
                let a = Owned::f32(&[2, 1], &[1., 2.]);
                let b = Owned::f32(&[1, 2], &[1., 2.]);
                let mut y = Owned::zeros_f32(&[2, 1]);
                let err = MathKernel(Math::GeGlu { tanh: false })
                    .execute(&[a.view(), b.view()], &mut [y.view_mut()])
                    .unwrap_err();
                assert!(err.to_string().contains("no broadcasting"), "{err}");
                assert_eq!(y.to_f32(), &[0., 0.]);
            }
        }
        if x.dtype == DataType::Float64 {
            let mut a = to_dense::<f64>(x)?;
            let b = inputs
                .get(1)
                .map(to_dense::<f64>)
                .transpose()?
                .unwrap_or_default();
            eval_f64(self.0, &mut a, &b, scan, |x| x);
            write_dense(&mut outputs[0], &a)
        } else {
            let mut a = to_dense_f32_widen(op, x)?.into_owned();
            let b = inputs
                .get(1)
                .map(|b| to_dense_f32_widen(op, b))
                .transpose()?
                .unwrap_or_default();
            eval_f32(self.0, &mut a, &b, scan, |v| match x.dtype {
                DataType::Float16 => half::f16::from_f32(v).to_f32(),
                DataType::BFloat16 => half::bf16::from_f32(v).to_f32(),
                _ => v,
            });
            write_dense_f32_narrow(op, &mut outputs[0], &a)
        }
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}
