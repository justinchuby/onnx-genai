//! Value selection kernels: `Clip`, `ArgMax`, `ArgMin`, `TopK`, and `NonZero`.

use core::cmp::Ordering;
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::add::require_same_dtype;
use super::{check_arity, to_dense_f32, to_dense_i64, write_dense_bytes, write_dense_f32};
use crate::dispatch_arith;
use crate::dtype::{NumericElem, to_dense, write_dense};
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
        dispatch_arith!(inputs[0].dtype, "Clip", T => {
            clip_typed::<T>(self, inputs, outputs)
        })
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

fn clip_typed<T: NumericElem + PartialOrd>(
    kernel: &ClipKernel,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    if outputs[0].dtype != T::DTYPE {
        return Err(EpError::KernelFailed(format!(
            "Clip: output dtype {:?} must match input dtype {:?}",
            outputs[0].dtype,
            T::DTYPE
        )));
    }
    let min = if inputs.len() > 1 && !inputs[1].is_absent() {
        require_same_dtype("Clip", &inputs[1], T::DTYPE)?;
        Some(scalar_typed::<T>("Clip min", &inputs[1])?)
    } else {
        kernel.min.map(T::from_f32_scalar)
    };
    let max = if inputs.len() > 2 && !inputs[2].is_absent() {
        require_same_dtype("Clip", &inputs[2], T::DTYPE)?;
        Some(scalar_typed::<T>("Clip max", &inputs[2])?)
    } else {
        kernel.max.map(T::from_f32_scalar)
    };
    if let (Some(min), Some(max)) = (min, max) {
        if min > max {
            return Err(EpError::KernelFailed(
                "Clip: min must not exceed max".into(),
            ));
        }
    }

    let y = to_dense::<T>(&inputs[0])?
        .into_iter()
        .map(|x| {
            let x = if let Some(min) = min {
                if x < min { min } else { x }
            } else {
                x
            };
            if let Some(max) = max {
                if x > max { max } else { x }
            } else {
                x
            }
        })
        .collect::<Vec<_>>();
    write_dense::<T>(&mut outputs[0], &y)
}

fn scalar_typed<T: NumericElem>(name: &str, view: &TensorView) -> Result<T> {
    let x = to_dense::<T>(view)?;
    if x.len() == 1 {
        Ok(x[0])
    } else {
        Err(EpError::KernelFailed(format!("{name} must be a scalar")))
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
    fn clip_supports_int8_defaults_and_f16_tensor_bounds() {
        let x = Owned {
            bytes: vec![(-3i8) as u8, 0, 4],
            shape: vec![3],
            strides: vec![1],
            dtype: DataType::Int8,
        };
        let mut int_out = Owned::zeros(DataType::Int8, &[3]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(&[x.view()], &mut [int_out.view_mut()])
        .unwrap();
        assert_eq!(int_out.bytes, x.bytes);

        let f16 = Owned::f16(&[3], &[-2., 0.5, 3.]);
        let lo = Owned::f16(&[], &[0.]);
        let hi = Owned::f16(&[], &[1.]);
        let mut f16_out = Owned::zeros(DataType::Float16, &[3]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(
            &[f16.view(), lo.view(), hi.view()],
            &mut [f16_out.view_mut()],
        )
        .unwrap();
        assert_eq!(f16_out.to_f16_as_f32(), vec![0., 0.5, 1.]);
    }

    #[test]
    fn clip_clamps_negative_i32_values() {
        let x = Owned::i32(&[5], &[-5, -1, 0, 3, 9]);
        let lo = Owned::i32(&[], &[-2]);
        let hi = Owned::i32(&[], &[4]);
        let mut y = Owned::zeros(DataType::Int32, &[5]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(&[x.view(), lo.view(), hi.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_i32(), vec![-2, -1, 0, 3, 4]);
    }

    #[test]
    fn clip_honors_absent_integer_bound_slots() {
        let x = Owned::i32(&[5], &[-5, -1, 0, 3, 9]);
        let lo = Owned::i32(&[], &[-2]);
        let hi = Owned::i32(&[], &[4]);

        let mut lower_only = Owned::zeros(DataType::Int32, &[5]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(
            &[x.view(), lo.view(), TensorView::absent(DataType::Int32)],
            &mut [lower_only.view_mut()],
        )
        .unwrap();
        assert_eq!(lower_only.to_i32(), vec![-2, -1, 0, 3, 9]);

        let mut upper_only = Owned::zeros(DataType::Int32, &[5]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(
            &[x.view(), TensorView::absent(DataType::Int32), hi.view()],
            &mut [upper_only.view_mut()],
        )
        .unwrap();
        assert_eq!(upper_only.to_i32(), vec![-5, -1, 0, 3, 4]);
    }

    #[test]
    fn clip_supports_i64_tensor_bounds() {
        let x = Owned::i64(&[5], &[-9, -2, 0, 4, 12]);
        let lo = Owned::i64(&[], &[-3]);
        let hi = Owned::i64(&[], &[5]);
        let mut y = Owned::zeros(DataType::Int64, &[5]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(&[x.view(), lo.view(), hi.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_i64(), vec![-3, -2, 0, 4, 5]);
    }

    #[test]
    fn clip_supports_f64_tensor_bounds() {
        let f64_owned = |shape: &[usize], data: &[f64]| Owned {
            bytes: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.to_vec(),
            strides: onnx_runtime_ir::compute_contiguous_strides(shape),
            dtype: DataType::Float64,
        };
        let x = f64_owned(&[5], &[-3.5, -1.0, 0.5, 2.0, 8.5]);
        let lo = f64_owned(&[], &[-2.0]);
        let hi = f64_owned(&[], &[4.0]);
        let mut y = Owned::zeros(DataType::Float64, &[5]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(&[x.view(), lo.view(), hi.view()], &mut [y.view_mut()])
        .unwrap();
        let values: Vec<f64> = y
            .bytes
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![-2.0, -1.0, 0.5, 2.0, 4.0]);
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
