//! Indexed data kernels: `GatherElements`, `GatherND`, `ScatterElements`, and
//! `OneHot`.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{check_arity, elem_size, to_dense_bytes, to_dense_i64, write_dense_bytes};
use crate::dispatch_arith;
use crate::dtype::{ComputeDomain, NumericElem, to_dense, write_dense};
use crate::strided::numel;

pub struct GatherElementsKernel {
    axis: i64,
}
pub struct GatherElementsFactory;
impl KernelFactory for GatherElementsFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(GatherElementsKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(0),
        }))
    }
}
impl Kernel for GatherElementsKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("GatherElements", inputs, outputs, 2, 2, 1)?;
        let data = &inputs[0];
        let indices = &inputs[1];
        if data.shape.len() != indices.shape.len() {
            return Err(EpError::KernelFailed(
                "GatherElements: data and indices must have equal rank".into(),
            ));
        }
        let rank = data.shape.len();
        let axis = normalize_axis("GatherElements", self.axis, rank)?;
        for d in 0..rank {
            if d != axis && indices.shape[d] > data.shape[d] {
                return Err(EpError::KernelFailed(format!(
                    "GatherElements: indices dimension {} exceeds data dimension {} at axis {d}",
                    indices.shape[d], data.shape[d]
                )));
            }
        }
        let esize = elem_size(data.dtype)?;
        if outputs[0].dtype != data.dtype {
            return Err(EpError::KernelFailed(
                "GatherElements: output dtype must match data".into(),
            ));
        }
        let src = to_dense_bytes(data)?;
        let idx = to_dense_i64(indices)?;
        let n = numel(indices.shape);
        let strides = contiguous(indices.shape);
        let data_strides = contiguous(data.shape);
        let mut out = vec![0; n * esize];
        for linear in 0..n {
            let mut rem = linear;
            let mut data_off = 0;
            for d in 0..rank {
                let coordinate = rem / strides[d];
                rem %= strides[d];
                let c = if d == axis {
                    let raw = idx[linear];
                    let adjusted = if raw < 0 {
                        raw + data.shape[d] as i64
                    } else {
                        raw
                    };
                    if adjusted < 0 || adjusted as usize >= data.shape[d] {
                        return Err(EpError::KernelFailed(format!(
                            "GatherElements: index {raw} out of range at axis {d}"
                        )));
                    }
                    adjusted as usize
                } else {
                    coordinate
                };
                data_off += c * data_strides[d];
            }
            out[linear * esize..(linear + 1) * esize]
                .copy_from_slice(&src[data_off * esize..(data_off + 1) * esize]);
        }
        write_dense_bytes(&mut outputs[0], &out)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

pub struct GatherNDKernel {
    batch_dims: i64,
}
pub struct GatherNDFactory;
impl KernelFactory for GatherNDFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(GatherNDKernel {
            batch_dims: node
                .attr("batch_dims")
                .and_then(|a| a.as_int())
                .unwrap_or(0),
        }))
    }
}
impl Kernel for GatherNDKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("GatherND", inputs, outputs, 2, 2, 1)?;
        let data = &inputs[0];
        let indices = &inputs[1];
        if indices.shape.is_empty() {
            return Err(EpError::KernelFailed(
                "GatherND: indices must have rank >= 1".into(),
            ));
        }
        let batch = self.batch_dims;
        if batch < 0 || batch as usize > data.shape.len() || batch as usize >= indices.shape.len() {
            return Err(EpError::KernelFailed("GatherND: invalid batch_dims".into()));
        }
        let batch = batch as usize;
        if data.shape[..batch] != indices.shape[..batch] {
            return Err(EpError::KernelFailed(
                "GatherND: batch dimensions must match".into(),
            ));
        }
        let k = *indices.shape.last().unwrap();
        if k > data.shape.len() - batch {
            return Err(EpError::KernelFailed(
                "GatherND: index tuple is longer than data suffix rank".into(),
            ));
        }
        if outputs[0].dtype != data.dtype {
            return Err(EpError::KernelFailed(
                "GatherND: output dtype must match data".into(),
            ));
        }
        let src = to_dense_bytes(data)?;
        let idx = to_dense_i64(indices)?;
        let esize = elem_size(data.dtype)?;
        let batches = numel(&data.shape[..batch]);
        let tuples_per_batch = numel(&indices.shape[batch..indices.shape.len() - 1]);
        let data_batch_len = numel(&data.shape[batch..]);
        let tail_len = numel(&data.shape[batch + k..]);
        let mut out = Vec::with_capacity(batches * tuples_per_batch * tail_len * esize);
        for b in 0..batches {
            for t in 0..tuples_per_batch {
                let mut base = b * data_batch_len;
                for d in 0..k {
                    let raw = idx[(b * tuples_per_batch + t) * k + d];
                    let dim = data.shape[batch + d];
                    let v = if raw < 0 { raw + dim as i64 } else { raw };
                    if v < 0 || v as usize >= dim {
                        return Err(EpError::KernelFailed(format!(
                            "GatherND: index {raw} out of range at tuple dimension {d}"
                        )));
                    }
                    base += v as usize * numel(&data.shape[batch + d + 1..]);
                }
                out.extend_from_slice(&src[base * esize..(base + tail_len) * esize]);
            }
        }
        write_dense_bytes(&mut outputs[0], &out)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

fn normalize_axis(op: &str, axis: i64, rank: usize) -> Result<usize> {
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    if axis < 0 || axis as usize >= rank {
        Err(EpError::KernelFailed(format!("{op}: axis out of range")))
    } else {
        Ok(axis as usize)
    }
}
fn contiguous(shape: &[usize]) -> Vec<usize> {
    let mut result = vec![1; shape.len()];
    for d in (0..shape.len()).rev().skip(1) {
        result[d] = result[d + 1] * shape[d + 1];
    }
    result
}

#[derive(Clone, Copy)]
enum ScatterReduction {
    None,
    Add,
    Mul,
    Max,
    Min,
}

pub struct ScatterElementsKernel {
    axis: i64,
    reduction: ScatterReduction,
}

/// `ScatterND`: update slices selected by the trailing index tuples.
pub struct ScatterNDKernel {
    reduction: ScatterReduction,
}

pub struct ScatterNDFactory;

impl KernelFactory for ScatterNDFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ScatterNDKernel {
            reduction: parse_scatter_reduction("ScatterND", node)?,
        }))
    }
}

pub(crate) fn scatter_nd_unsupported_reason(input_dtypes: &[DataType]) -> Option<String> {
    if input_dtypes.is_empty() {
        return None;
    }
    if input_dtypes.len() != 3 {
        return Some(format!(
            "ScatterND requires 3 inputs, got {}",
            input_dtypes.len()
        ));
    }
    let data_dtype = input_dtypes[0];
    if !matches!(
        data_dtype,
        DataType::Float32
            | DataType::Float16
            | DataType::BFloat16
            | DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Uint8
            | DataType::Uint16
            | DataType::Uint32
            | DataType::Uint64
    ) {
        return Some(format!(
            "ScatterND: data dtype {data_dtype:?} is not implemented by the CPU kernel"
        ));
    }
    if input_dtypes[1] != DataType::Int64 {
        return Some(format!(
            "ScatterND: indices must have Int64 dtype, got {:?}",
            input_dtypes[1]
        ));
    }
    if input_dtypes[2] != data_dtype {
        return Some(format!(
            "ScatterND: updates dtype {:?} must match data dtype {data_dtype:?}",
            input_dtypes[2]
        ));
    }
    None
}

fn parse_scatter_reduction(op: &str, node: &Node) -> Result<ScatterReduction> {
    match node.attr("reduction").and_then(Attribute::as_str) {
        None | Some("none") => Ok(ScatterReduction::None),
        Some("add") => Ok(ScatterReduction::Add),
        Some("mul") => Ok(ScatterReduction::Mul),
        Some("max") => Ok(ScatterReduction::Max),
        Some("min") => Ok(ScatterReduction::Min),
        Some(value) => Err(EpError::KernelFailed(format!(
            "{op}: unsupported reduction {value:?}"
        ))),
    }
}

impl Kernel for ScatterNDKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("ScatterND", inputs, outputs, 3, 3, 1)?;
        dispatch_arith!(inputs[0].dtype, "ScatterND", T => {
            scatter_nd_typed::<T>(self, inputs, outputs)
        })
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}
pub struct ScatterElementsFactory;
impl KernelFactory for ScatterElementsFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ScatterElementsKernel {
            axis: node.attr("axis").and_then(Attribute::as_int).unwrap_or(0),
            reduction: parse_scatter_reduction("ScatterElements", node)?,
        }))
    }
}
impl Kernel for ScatterElementsKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("ScatterElements", inputs, outputs, 3, 3, 1)?;
        dispatch_arith!(inputs[0].dtype, "ScatterElements", T => {
            scatter_elements_typed::<T>(self, inputs, outputs)
        })
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

fn scatter_elements_typed<T: NumericElem>(
    kernel: &ScatterElementsKernel,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()>
where
    T::Acc: PartialOrd,
{
    let data = &inputs[0];
    let indices = &inputs[1];
    let updates = &inputs[2];
    if indices.dtype != DataType::Int64 {
        return Err(EpError::KernelFailed(
            "ScatterElements: indices must be Int64".into(),
        ));
    }
    if updates.dtype != T::DTYPE || outputs[0].dtype != T::DTYPE {
        return Err(EpError::KernelFailed(
            "ScatterElements: data, updates, and output must share a dtype".into(),
        ));
    }
    if indices.shape != updates.shape || indices.shape.len() != data.shape.len() {
        return Err(EpError::KernelFailed(
            "ScatterElements: indices and updates must have equal rank and shape".into(),
        ));
    }
    if outputs[0].shape != data.shape {
        return Err(EpError::KernelFailed(
            "ScatterElements: output shape must match data".into(),
        ));
    }
    let rank = data.shape.len();
    let axis = normalize_axis("ScatterElements", kernel.axis, rank)?;
    for d in 0..rank {
        if d != axis && indices.shape[d] > data.shape[d] {
            return Err(EpError::KernelFailed(format!(
                "ScatterElements: indices dimension {} exceeds data dimension {} at axis {d}",
                indices.shape[d], data.shape[d]
            )));
        }
    }

    let mut out = to_dense::<T>(data)?;
    let indices = to_dense_i64(indices)?;
    let updates = to_dense::<T>(updates)?;
    let index_strides = contiguous(inputs[1].shape);
    let data_strides = contiguous(data.shape);
    for (linear, update) in updates.into_iter().enumerate() {
        let mut rem = linear;
        let mut data_offset = 0;
        for d in 0..rank {
            let coordinate = rem / index_strides[d];
            rem %= index_strides[d];
            let coordinate = if d == axis {
                let raw = indices[linear];
                let adjusted = if raw < 0 {
                    raw + data.shape[d] as i64
                } else {
                    raw
                };
                if adjusted < 0 || adjusted as usize >= data.shape[d] {
                    return Err(EpError::KernelFailed(format!(
                        "ScatterElements: index {raw} out of range at axis {d}"
                    )));
                }
                adjusted as usize
            } else {
                coordinate
            };
            data_offset += coordinate * data_strides[d];
        }
        out[data_offset] = match kernel.reduction {
            ScatterReduction::None => update,
            ScatterReduction::Add => T::from_acc(out[data_offset].to_acc().c_add(update.to_acc())),
            ScatterReduction::Mul => T::from_acc(out[data_offset].to_acc().c_mul(update.to_acc())),
            ScatterReduction::Max => T::from_acc(out[data_offset].to_acc().c_max(update.to_acc())),
            ScatterReduction::Min => T::from_acc(out[data_offset].to_acc().c_min(update.to_acc())),
        };
    }
    write_dense::<T>(&mut outputs[0], &out)
}

fn scatter_nd_typed<T: NumericElem>(
    kernel: &ScatterNDKernel,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()>
where
    T::Acc: PartialOrd,
{
    let data = &inputs[0];
    let indices = &inputs[1];
    let updates = &inputs[2];
    if indices.dtype != DataType::Int64 {
        return Err(EpError::KernelFailed(
            "ScatterND: indices must be Int64".into(),
        ));
    }
    if updates.dtype != T::DTYPE || outputs[0].dtype != T::DTYPE {
        return Err(EpError::KernelFailed(
            "ScatterND: data, updates, and output must share a dtype".into(),
        ));
    }
    if indices.shape.is_empty() {
        return Err(EpError::KernelFailed(
            "ScatterND: indices must have rank at least 1".into(),
        ));
    }
    if outputs[0].shape != data.shape {
        return Err(EpError::KernelFailed(
            "ScatterND: output shape must match data".into(),
        ));
    }

    let index_depth = *indices.shape.last().expect("rank checked above");
    if index_depth > data.shape.len() {
        return Err(EpError::KernelFailed(format!(
            "ScatterND: index tuple length {index_depth} exceeds data rank {}",
            data.shape.len()
        )));
    }
    let mut expected_updates_shape = indices.shape[..indices.shape.len() - 1].to_vec();
    expected_updates_shape.extend_from_slice(&data.shape[index_depth..]);
    if updates.shape != expected_updates_shape {
        return Err(EpError::KernelFailed(format!(
            "ScatterND: updates shape {:?} must be {:?}",
            updates.shape, expected_updates_shape
        )));
    }

    let tuple_count = numel(&indices.shape[..indices.shape.len() - 1]);
    let slice_len = numel(&data.shape[index_depth..]);
    let mut out = to_dense::<T>(data)?;
    let indices = to_dense_i64(indices)?;
    let updates = to_dense::<T>(updates)?;
    let data_strides = contiguous(data.shape);
    for tuple in 0..tuple_count {
        let mut destination = 0usize;
        for dimension in 0..index_depth {
            let raw = indices[tuple * index_depth + dimension];
            let dim = data.shape[dimension];
            let coordinate = if raw < 0 { raw + dim as i64 } else { raw };
            if coordinate < 0 || coordinate as usize >= dim {
                return Err(EpError::KernelFailed(format!(
                    "ScatterND: index {raw} out of range at tuple dimension {dimension}"
                )));
            }
            destination += coordinate as usize * data_strides[dimension];
        }
        for element in 0..slice_len {
            let destination = destination + element;
            let update = updates[tuple * slice_len + element];
            out[destination] = match kernel.reduction {
                ScatterReduction::None => update,
                ScatterReduction::Add => {
                    T::from_acc(out[destination].to_acc().c_add(update.to_acc()))
                }
                ScatterReduction::Mul => {
                    T::from_acc(out[destination].to_acc().c_mul(update.to_acc()))
                }
                ScatterReduction::Max => {
                    T::from_acc(out[destination].to_acc().c_max(update.to_acc()))
                }
                ScatterReduction::Min => {
                    T::from_acc(out[destination].to_acc().c_min(update.to_acc()))
                }
            };
        }
    }
    write_dense::<T>(&mut outputs[0], &out)
}

pub struct OneHotKernel {
    axis: i64,
}
pub struct OneHotFactory;
impl KernelFactory for OneHotFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(OneHotKernel {
            axis: node.attr("axis").and_then(Attribute::as_int).unwrap_or(-1),
        }))
    }
}
impl Kernel for OneHotKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("OneHot", inputs, outputs, 3, 3, 1)?;
        let indices = &inputs[0];
        if indices.dtype != DataType::Int64 || inputs[1].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(
                "OneHot: indices and depth must be Int64".into(),
            ));
        }
        let depth = to_dense_i64(&inputs[1])?;
        if depth.len() != 1 || depth[0] < 0 {
            return Err(EpError::KernelFailed(
                "OneHot: depth must be a non-negative scalar".into(),
            ));
        }
        let depth = usize::try_from(depth[0]).map_err(|_| {
            EpError::KernelFailed("OneHot: depth exceeds addressable memory".into())
        })?;
        let values = &inputs[2];
        if values.shape != [2] || outputs[0].dtype != values.dtype {
            return Err(EpError::KernelFailed(
                "OneHot: values must have shape [2] and output must match its dtype".into(),
            ));
        }
        let output_rank = indices.shape.len() + 1;
        let axis = if self.axis < 0 {
            self.axis + output_rank as i64
        } else {
            self.axis
        };
        if axis < 0 || axis as usize >= output_rank {
            return Err(EpError::KernelFailed("OneHot: axis out of range".into()));
        }
        let axis = axis as usize;
        let mut expected_shape = indices.shape.to_vec();
        expected_shape.insert(axis, depth);
        if outputs[0].shape != expected_shape {
            return Err(EpError::KernelFailed(
                "OneHot: output shape does not match indices, depth, and axis".into(),
            ));
        }
        let element_size = elem_size(values.dtype)?;
        let values = to_dense_bytes(values)?;
        let indices = to_dense_i64(indices)?;
        let off_value = &values[..element_size];
        let on_value = &values[element_size..];
        let mut out = vec![0; numel(&expected_shape) * element_size];
        for chunk in out.chunks_exact_mut(element_size) {
            chunk.copy_from_slice(off_value);
        }
        let output_strides = contiguous(&expected_shape);
        let index_strides = contiguous(inputs[0].shape);
        for (index_linear, index) in indices.into_iter().enumerate() {
            if index < 0 || index as usize >= depth {
                continue;
            }
            let mut rem = index_linear;
            let mut output_linear = 0;
            for (d, &output_stride) in output_strides.iter().enumerate() {
                if d == axis {
                    output_linear += index as usize * output_stride;
                } else {
                    let index_dimension = if d < axis { d } else { d - 1 };
                    let stride = index_strides[index_dimension];
                    let coordinate = rem / stride;
                    rem %= stride;
                    output_linear += coordinate * output_stride;
                }
            }
            out[output_linear * element_size..(output_linear + 1) * element_size]
                .copy_from_slice(on_value);
        }
        write_dense_bytes(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    #[test]
    fn gather_elements_negative_axis_and_index() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let i = Owned::i64(&[2, 2], &[2, 0, -1, 1]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        GatherElementsKernel { axis: -1 }
            .execute(&[x.view(), i.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![3., 1., 6., 5.]);
    }
    #[test]
    fn gather_nd_tuples_and_negative_indices() {
        let x = Owned::f32(&[2, 2, 2], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let i = Owned::i64(&[2, 2], &[0, 1, -1, 0]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        GatherNDKernel { batch_dims: 0 }
            .execute(&[x.view(), i.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![2., 3., 4., 5.]);
    }
    #[test]
    fn gather_nd_with_batch_dims() {
        let x = Owned::f32(&[2, 2, 2], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let i = Owned::i64(&[2, 2, 1], &[1, 0, 0, 1]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        GatherNDKernel { batch_dims: 1 }
            .execute(&[x.view(), i.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![2., 3., 0., 1., 4., 5., 6., 7.]);
    }

    #[test]
    fn scatter_elements_negative_axis_and_add_reduction() {
        let data = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let indices = Owned::i64(&[2, 2], &[2, 0, -1, 1]);
        let updates = Owned::f32(&[2, 2], &[10., 20., 30., 40.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        ScatterElementsKernel {
            axis: -1,
            reduction: ScatterReduction::Add,
        }
        .execute(
            &[data.view(), indices.view(), updates.view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        assert_eq!(out.to_f32(), vec![21., 2., 13., 4., 45., 36.]);
    }

    #[test]
    fn scatter_elements_overwrites_duplicate_indices() {
        let data = Owned::i32(&[3], &[1, 2, 3]);
        let indices = Owned::i64(&[3], &[1, 1, 0]);
        let updates = Owned::i32(&[3], &[7, 8, 9]);
        let mut out = Owned::zeros(DataType::Int32, &[3]);
        ScatterElementsKernel {
            axis: 0,
            reduction: ScatterReduction::None,
        }
        .execute(
            &[data.view(), indices.view(), updates.view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        assert_eq!(out.to_i32(), vec![9, 8, 3]);
    }

    #[test]
    fn scatter_nd_replaces_slices_at_multidimensional_tuples() {
        // ONNX ScatterND: indices [2,2] address the first two dimensions of
        // data [2,3,2], so each update is a trailing slice of length two.
        let data = Owned::f32(
            &[2, 3, 2],
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        );
        let indices = Owned::i64(&[2, 2], &[1, 0, 0, 1]);
        let updates = Owned::f32(&[2, 2], &[10., 11., 20., 21.]);
        let mut out = Owned::zeros_f32(&[2, 3, 2]);
        ScatterNDKernel {
            reduction: ScatterReduction::None,
        }
        .execute(
            &[data.view(), indices.view(), updates.view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        assert_eq!(
            out.to_f32(),
            vec![0., 1., 20., 21., 4., 5., 10., 11., 8., 9., 10., 11.]
        );
    }

    #[test]
    fn scatter_nd_add_reduces_duplicate_indices() {
        // Duplicate coordinates are well-defined with reduction=add.
        let data = Owned::i32(&[2, 2], &[1, 2, 3, 4]);
        let indices = Owned::i64(&[3, 1], &[0, 1, 1]);
        let updates = Owned::i32(&[3, 2], &[10, 20, 30, 40, 50, 60]);
        let mut out = Owned::zeros(DataType::Int32, &[2, 2]);
        ScatterNDKernel {
            reduction: ScatterReduction::Add,
        }
        .execute(
            &[data.view(), indices.view(), updates.view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        assert_eq!(out.to_i32(), vec![11, 22, 83, 104]);
    }

    #[test]
    fn scatter_nd_mul_reduces_duplicate_indices() {
        let data = Owned::i32(&[2, 2], &[2, 3, 4, 5]);
        let indices = Owned::i64(&[3, 1], &[0, 1, 1]);
        let updates = Owned::i32(&[3, 2], &[7, 11, 2, 3, 5, 7]);
        let mut out = Owned::zeros(DataType::Int32, &[2, 2]);
        ScatterNDKernel {
            reduction: ScatterReduction::Mul,
        }
        .execute(
            &[data.view(), indices.view(), updates.view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        assert_eq!(out.to_i32(), vec![14, 33, 40, 105]);
    }

    #[test]
    fn one_hot_negative_index_is_all_off_and_negative_axis() {
        let indices = Owned::i64(&[2], &[0, -1]);
        let depth = Owned::i64(&[], &[3]);
        let values = Owned::i32(&[2], &[2, 5]);
        let mut out = Owned::zeros(DataType::Int32, &[2, 3]);
        OneHotKernel { axis: -1 }
            .execute(
                &[indices.view(), depth.view(), values.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_i32(), vec![5, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn one_hot_inserts_axis_before_indices_dimensions() {
        let indices = Owned::i64(&[2, 2], &[0, 1, 1, 0]);
        let depth = Owned::i64(&[], &[2]);
        let values = Owned::f32(&[2], &[0., 1.]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        OneHotKernel { axis: 1 }
            .execute(
                &[indices.view(), depth.view(), values.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 0., 0., 1., 0., 1., 1., 0.]);
    }
    #[test]
    fn indexed_bf16_movement_preserves_bits() {
        let x = Owned::bf16(&[2, 2], &[1., -2., 3., 4.]);
        let indices = Owned::i64(&[2, 1], &[1, 0]);
        let mut ge = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 1]);
        GatherElementsKernel { axis: 1 }
            .execute(&[x.view(), indices.view()], &mut [ge.view_mut()])
            .unwrap();
        assert_eq!(
            ge.to_u16_bits(),
            vec![x.to_u16_bits()[1], x.to_u16_bits()[2]]
        );
        let nd_indices = Owned::i64(&[2, 2], &[0, 1, 1, 0]);
        let mut nd = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2]);
        GatherNDKernel { batch_dims: 0 }
            .execute(&[x.view(), nd_indices.view()], &mut [nd.view_mut()])
            .unwrap();
        assert_eq!(
            nd.to_u16_bits(),
            vec![x.to_u16_bits()[1], x.to_u16_bits()[2]]
        );
        let updates = Owned::bf16(&[2, 1], &[9., -8.]);
        let mut scatter = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 2]);
        ScatterElementsKernel {
            axis: 1,
            reduction: ScatterReduction::None,
        }
        .execute(
            &[x.view(), indices.view(), updates.view()],
            &mut [scatter.view_mut()],
        )
        .unwrap();
        assert_eq!(
            scatter.to_u16_bits(),
            vec![
                x.to_u16_bits()[0],
                updates.to_u16_bits()[0],
                updates.to_u16_bits()[1],
                x.to_u16_bits()[3]
            ]
        );
    }
}
