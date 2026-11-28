use std::ops::Neg;

use onnx_runtime_ep_api::negative_log_likelihood_loss::{Dimensions, Loss, OP, Reduction, weight};
use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::dispatch_float;
use crate::dtype::{ComputeDomain, NumericElem, to_dense, write_dense};

pub struct Factory(pub u64);

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(NativeLoss(Loss::from_node(node, self.0)?)))
    }
}

struct NativeLoss(Loss);

impl Kernel for NativeLoss {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let dims = self.0.validate(inputs, outputs)?;
        dispatch_float!(inputs[0].dtype, OP, T => {
            if inputs[1].dtype == DataType::Int32 {
                self.compute::<T, i32>(inputs, &mut outputs[0], dims)
            } else {
                self.compute::<T, i64>(inputs, &mut outputs[0], dims)
            }
        })
    }
}

impl NativeLoss {
    fn compute<T: NumericElem, I: NumericElem + Into<i64>>(
        &self,
        inputs: &[TensorView],
        output: &mut TensorMut,
        dims: Dimensions,
    ) -> Result<()>
    where
        T::Acc: Neg<Output = T::Acc>,
    {
        let x = to_dense::<T>(&inputs[0])?;
        let targets = to_dense::<I>(&inputs[1])?;
        let weights = weight(inputs).map(to_dense::<T>).transpose()?;
        let mut losses = Vec::new();
        if self.0.reduction == Reduction::None {
            losses.reserve(dims.elements);
        }
        let (mut sum, mut weight_sum) = (T::Acc::default(), T::Acc::default());
        let mut count = 0;
        for (i, label) in targets.into_iter().enumerate() {
            let mut value = T::Acc::default();
            if let Some(class) = self.0.class(label.into(), i, dims.classes)? {
                let offset =
                    (i / dims.spatial * dims.classes + class) * dims.spatial + i % dims.spatial;
                value = -x[offset].to_acc();
                if let Some(weights) = &weights {
                    let w = weights[class].to_acc();
                    value = value.c_mul(w);
                    if self.0.reduction == Reduction::Mean {
                        weight_sum = weight_sum.c_add(w);
                    }
                }
                count += 1;
            }
            if self.0.reduction == Reduction::None {
                losses.push(T::from_acc(value));
            } else {
                sum = sum.c_add(value);
            }
        }
        match self.0.reduction {
            Reduction::None => write_dense(output, &losses),
            Reduction::Sum => write_dense(output, &[T::from_acc(sum)]),
            Reduction::Mean => {
                let mean = if weights.is_some() {
                    sum.c_div(weight_sum)
                } else {
                    sum.c_div_usize(count)
                };
                write_dense(output, &[T::from_acc(mean)])
            }
        }
    }
}
