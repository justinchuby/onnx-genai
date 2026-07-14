//! Value selection kernels: `Clip`, `ArgMax`, `ArgMin`, `TopK`, and `NonZero`.

use core::cmp::Ordering;
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, to_dense_f32, to_dense_i64, write_dense_bytes, write_dense_f32};
use crate::strided::numel;

pub struct ClipKernel {
    min: Option<f32>,
    max: Option<f32>,
}
pub struct ClipFactory;
impl KernelFactory for ClipFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ClipKernel {
            min: node.attr("min").and_then(|a| a.as_float()),
            max: node.attr("max").and_then(|a| a.as_float()),
        }))
    }
}
impl Kernel for ClipKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Clip", inputs, outputs, 1, 3, 1)?;
        if inputs[0].dtype != DataType::Float32 || outputs[0].dtype != DataType::Float32 {
            return Err(EpError::KernelFailed(
                "Clip: currently supports Float32".into(),
            ));
        }
        let min = if inputs.len() > 1 {
            scalar_f32("Clip min", &inputs[1])?
        } else {
            self.min.unwrap_or(f32::NEG_INFINITY)
        };
        let max = if inputs.len() > 2 {
            scalar_f32("Clip max", &inputs[2])?
        } else {
            self.max.unwrap_or(f32::INFINITY)
        };
        if min > max {
            return Err(EpError::KernelFailed(
                "Clip: min must not exceed max".into(),
            ));
        }
        let y: Vec<f32> = to_dense_f32(&inputs[0])?
            .into_iter()
            .map(|x| if x.is_nan() { x } else { x.max(min).min(max) })
            .collect();
        write_dense_f32(&mut outputs[0], &y)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

#[derive(Clone, Copy)]
enum ArgOp {
    Max,
    Min,
}
pub struct ArgKernel {
    op: ArgOp,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
}
pub struct ArgMaxFactory;
pub struct ArgMinFactory;
fn arg_factory(node: &Node, op: ArgOp) -> Box<dyn Kernel> {
    Box::new(ArgKernel {
        op,
        axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(0),
        keepdims: node.attr("keepdims").and_then(|a| a.as_int()).unwrap_or(1) != 0,
        select_last_index: node
            .attr("select_last_index")
            .and_then(|a| a.as_int())
            .unwrap_or(0)
            != 0,
    })
}
impl KernelFactory for ArgMaxFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(arg_factory(node, ArgOp::Max))
    }
}
impl KernelFactory for ArgMinFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(arg_factory(node, ArgOp::Min))
    }
}
impl Kernel for ArgKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let name = match self.op {
            ArgOp::Max => "ArgMax",
            ArgOp::Min => "ArgMin",
        };
        check_arity(name, inputs, outputs, 1, 1, 1)?;
        if outputs[0].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(format!(
                "{name}: output must be Int64"
            )));
        }
        let x = to_dense_f32(&inputs[0])?;
        let axis = axis(name, self.axis, inputs[0].shape.len())?;
        let width = inputs[0].shape[axis];
        if width == 0 {
            return Err(EpError::KernelFailed(format!(
                "{name}: reduced axis must be non-empty"
            )));
        }
        let inner = numel(&inputs[0].shape[axis + 1..]);
        let mut out = Vec::with_capacity(numel(outputs[0].shape));
        for outer in 0..numel(&inputs[0].shape[..axis]) {
            for i in 0..inner {
                let mut best = 0;
                for d in 1..width {
                    let candidate = x[(outer * width + d) * inner + i];
                    let value = x[(outer * width + best) * inner + i];
                    let better = match self.op {
                        ArgOp::Max => candidate > value,
                        ArgOp::Min => candidate < value,
                    };
                    if better || (self.select_last_index && candidate == value) {
                        best = d;
                    }
                }
                out.push(best as i64);
            }
        }
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = self.keepdims;
        write_dense_bytes(&mut outputs[0], &bytes)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

pub struct TopKKernel {
    axis: i64,
    largest: bool,
    sorted: bool,
}
pub struct TopKFactory;
impl KernelFactory for TopKFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(TopKKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1),
            largest: node.attr("largest").and_then(|a| a.as_int()).unwrap_or(1) != 0,
            sorted: node.attr("sorted").and_then(|a| a.as_int()).unwrap_or(1) != 0,
        }))
    }
}
impl Kernel for TopKKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("TopK", inputs, outputs, 2, 2, 2)?;
        if outputs[1].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(
                "TopK: indices output must be Int64".into(),
            ));
        }
        let x = to_dense_f32(&inputs[0])?;
        let k_values = to_dense_i64(&inputs[1])?;
        if k_values.len() != 1 || k_values[0] < 0 {
            return Err(EpError::KernelFailed(
                "TopK: K must be a non-negative scalar".into(),
            ));
        }
        let axis = axis("TopK", self.axis, inputs[0].shape.len())?;
        let width = inputs[0].shape[axis];
        let k = k_values[0] as usize;
        if k > width {
            return Err(EpError::KernelFailed(
                "TopK: K exceeds selected axis".into(),
            ));
        }
        let inner = numel(&inputs[0].shape[axis + 1..]);
        let mut values = Vec::with_capacity(numel(outputs[0].shape));
        let mut indices = Vec::with_capacity(numel(outputs[1].shape));
        for outer in 0..numel(&inputs[0].shape[..axis]) {
            for i in 0..inner {
                let mut candidates: Vec<usize> = (0..width).collect();
                candidates.sort_by(|&a, &b| {
                    topk_order(
                        x[(outer * width + a) * inner + i],
                        x[(outer * width + b) * inner + i],
                        a,
                        b,
                        self.largest,
                    )
                });
                if !self.sorted {
                    candidates.truncate(k);
                }
                for d in candidates.into_iter().take(k) {
                    values.push(x[(outer * width + d) * inner + i]);
                    indices.push(d as i64);
                }
            }
        }
        write_dense_f32(&mut outputs[0], &values)?;
        write_dense_bytes(
            &mut outputs[1],
            &indices
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}
fn topk_order(a: f32, b: f32, ia: usize, ib: usize, largest: bool) -> Ordering {
    let order = if largest {
        b.total_cmp(&a)
    } else {
        a.total_cmp(&b)
    };
    if order == Ordering::Equal {
        ia.cmp(&ib)
    } else {
        order
    }
}

pub struct NonZeroKernel;
pub struct NonZeroFactory;
impl KernelFactory for NonZeroFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(NonZeroKernel))
    }
}
impl Kernel for NonZeroKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("NonZero", inputs, outputs, 1, 1, 1)?;
        if outputs[0].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(
                "NonZero: output must be Int64".into(),
            ));
        }
        let x = to_dense_f32(&inputs[0])?;
        let rank = inputs[0].shape.len();
        let strides = contiguous(inputs[0].shape);
        let mut coordinates = vec![Vec::new(); rank];
        for (linear, &v) in x.iter().enumerate() {
            if v != 0. {
                let mut rem = linear;
                for d in 0..rank {
                    coordinates[d].push((rem / strides[d]) as i64);
                    rem %= strides[d];
                }
            }
        }
        let bytes: Vec<u8> = coordinates
            .into_iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        write_dense_bytes(&mut outputs[0], &bytes)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}
fn scalar_f32(name: &str, view: &TensorView) -> Result<f32> {
    let x = to_dense_f32(view)?;
    if x.len() == 1 {
        Ok(x[0])
    } else {
        Err(EpError::KernelFailed(format!("{name} must be a scalar")))
    }
}
fn axis(name: &str, raw: i64, rank: usize) -> Result<usize> {
    let a = if raw < 0 { raw + rank as i64 } else { raw };
    if a < 0 || a as usize >= rank {
        Err(EpError::KernelFailed(format!("{name}: axis out of range")))
    } else {
        Ok(a as usize)
    }
}
fn contiguous(shape: &[usize]) -> Vec<usize> {
    let mut r = vec![1; shape.len()];
    for d in (0..shape.len()).rev().skip(1) {
        r[d] = r[d + 1] * shape[d + 1];
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    #[test]
    fn clip_tensor_bounds() {
        let x = Owned::f32(&[3], &[-2., 0.5, 3.]);
        let lo = Owned::f32(&[], &[0.]);
        let hi = Owned::f32(&[], &[1.]);
        let mut y = Owned::zeros_f32(&[3]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(&[x.view(), lo.view(), hi.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_f32(), vec![0., 0.5, 1.]);
    }
    #[test]
    fn argmax_last_tie_negative_axis() {
        let x = Owned::f32(&[2, 3], &[1., 4., 4., 3., 2., 1.]);
        let mut y = Owned::zeros(DataType::Int64, &[2]);
        ArgKernel {
            op: ArgOp::Max,
            axis: -1,
            keepdims: false,
            select_last_index: true,
        }
        .execute(&[x.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_i64(), vec![2, 0]);
    }
    #[test]
    fn argmin_keepdims_selects_last_tie() {
        let x = Owned::f32(&[2, 3], &[3., 1., 1., 2., 0., 0.]);
        let mut y = Owned::zeros(DataType::Int64, &[2, 1]);
        ArgKernel {
            op: ArgOp::Min,
            axis: 1,
            keepdims: true,
            select_last_index: true,
        }
        .execute(&[x.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_i64(), vec![2, 2]);
    }
    #[test]
    fn topk_and_nonzero() {
        let x = Owned::f32(&[4], &[2., 5., 1., 4.]);
        let k = Owned::i64(&[], &[2]);
        let mut v = Owned::zeros_f32(&[2]);
        let mut i = Owned::zeros(DataType::Int64, &[2]);
        TopKKernel {
            axis: -1,
            largest: true,
            sorted: true,
        }
        .execute(&[x.view(), k.view()], &mut [v.view_mut(), i.view_mut()])
        .unwrap();
        assert_eq!(v.to_f32(), vec![5., 4.]);
        assert_eq!(i.to_i64(), vec![1, 3]);
        let z = Owned::f32(&[2, 2], &[0., 3., 4., 0.]);
        let mut o = Owned::zeros(DataType::Int64, &[2, 2]);
        NonZeroKernel
            .execute(&[z.view()], &mut [o.view_mut()])
            .unwrap();
        assert_eq!(o.to_i64(), vec![0, 1, 1, 0]);
    }
}
