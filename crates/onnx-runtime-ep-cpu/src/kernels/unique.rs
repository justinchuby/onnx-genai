//! ONNX `Unique` (opset 11+) over flattened elements or slices along an axis.

use std::cmp::Ordering;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{elem_size, to_dense_bytes, write_dense_bytes};
use crate::strided::numel;

pub struct UniqueKernel {
    axis: Option<i64>,
    sorted: bool,
}

pub struct UniqueFactory;

impl KernelFactory for UniqueFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = optional_int_attr(node, "axis")?;
        let sorted = match optional_int_attr(node, "sorted")?.unwrap_or(1) {
            0 => false,
            1 => true,
            value => {
                return Err(EpError::KernelFailed(format!(
                    "Unique: `sorted` must be 0 or 1, got {value}"
                )));
            }
        };
        Ok(Box::new(UniqueKernel { axis, sorted }))
    }
}

impl Kernel for UniqueKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || !(1..=4).contains(&outputs.len()) {
            return Err(EpError::KernelFailed(format!(
                "Unique: expected 1 input and 1..=4 outputs, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }

        let input = &inputs[0];
        let element_size = elem_size(input.dtype)?;
        ensure_supported_dtype(input.dtype)?;
        let dense = to_dense_bytes(input)?;
        let plan = unique_plan(
            &dense,
            input.dtype,
            input.shape,
            element_size,
            self.axis,
            self.sorted,
        )?;

        let expected_y_shape = match plan.axis {
            Some(axis) => {
                let mut shape = input.shape.to_vec();
                shape[axis] = plan.first_indices.len();
                shape
            }
            None => vec![plan.first_indices.len()],
        };
        validate_output(&outputs[0], input.dtype, &expected_y_shape, "Y")?;
        let y = gather_y(
            &dense,
            input.shape,
            element_size,
            plan.axis,
            &plan.first_indices,
        );
        write_dense_bytes(&mut outputs[0], &y)?;

        let unique_shape = [plan.first_indices.len()];
        if outputs.len() >= 2 {
            validate_output(&outputs[1], DataType::Int64, &unique_shape, "indices")?;
            write_i64(&mut outputs[1], &plan.first_indices)?;
        }
        if outputs.len() >= 3 {
            let inverse_shape = [plan.inverse_indices.len()];
            validate_output(
                &outputs[2],
                DataType::Int64,
                &inverse_shape,
                "inverse_indices",
            )?;
            write_i64(&mut outputs[2], &plan.inverse_indices)?;
        }
        if outputs.len() == 4 {
            validate_output(&outputs[3], DataType::Int64, &unique_shape, "counts")?;
            write_i64(&mut outputs[3], &plan.counts)?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx == 0
    }
}

struct UniquePlan {
    axis: Option<usize>,
    first_indices: Vec<usize>,
    inverse_indices: Vec<usize>,
    counts: Vec<usize>,
}

fn unique_plan(
    dense: &[u8],
    dtype: DataType,
    shape: &[usize],
    element_size: usize,
    axis: Option<i64>,
    sorted: bool,
) -> Result<UniquePlan> {
    let axis = axis
        .map(|axis| normalize_axis(axis, shape.len()))
        .transpose()?;
    let item_count = axis.map_or_else(|| numel(shape), |axis| shape[axis]);
    let items = make_items(dense, shape, element_size, axis);

    let mut unique_items: Vec<Vec<u8>> = Vec::new();
    let mut first_indices = Vec::new();
    let mut inverse_indices = Vec::with_capacity(item_count);
    let mut counts = Vec::new();
    for (index, item) in items.into_iter().enumerate() {
        let group = unique_items
            .iter()
            .position(|unique| compare_items(dtype, unique, &item) == Ordering::Equal);
        let group = match group {
            Some(group) => {
                counts[group] += 1;
                group
            }
            None => {
                let group = unique_items.len();
                unique_items.push(item);
                first_indices.push(index);
                counts.push(1);
                group
            }
        };
        inverse_indices.push(group);
    }

    if sorted {
        let mut order: Vec<usize> = (0..unique_items.len()).collect();
        order.sort_by(|&a, &b| compare_items(dtype, &unique_items[a], &unique_items[b]));
        let mut old_to_new = vec![0; order.len()];
        for (new, &old) in order.iter().enumerate() {
            old_to_new[old] = new;
        }
        first_indices = order.iter().map(|&old| first_indices[old]).collect();
        counts = order.iter().map(|&old| counts[old]).collect();
        for group in &mut inverse_indices {
            *group = old_to_new[*group];
        }
    }

    Ok(UniquePlan {
        axis,
        first_indices,
        inverse_indices,
        counts,
    })
}

fn make_items(
    dense: &[u8],
    shape: &[usize],
    element_size: usize,
    axis: Option<usize>,
) -> Vec<Vec<u8>> {
    let Some(axis) = axis else {
        return dense
            .chunks_exact(element_size)
            .map(<[u8]>::to_vec)
            .collect();
    };
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let block_bytes = inner * element_size;
    let mut items = vec![Vec::with_capacity(outer * block_bytes); axis_len];
    for outer_index in 0..outer {
        for (axis_index, item) in items.iter_mut().enumerate() {
            let start = (outer_index * axis_len + axis_index) * block_bytes;
            item.extend_from_slice(&dense[start..start + block_bytes]);
        }
    }
    items
}

fn gather_y(
    dense: &[u8],
    shape: &[usize],
    element_size: usize,
    axis: Option<usize>,
    first_indices: &[usize],
) -> Vec<u8> {
    let Some(axis) = axis else {
        let mut output = Vec::with_capacity(first_indices.len() * element_size);
        for &index in first_indices {
            let start = index * element_size;
            output.extend_from_slice(&dense[start..start + element_size]);
        }
        return output;
    };
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let block_bytes = inner * element_size;
    let mut output = Vec::with_capacity(outer * first_indices.len() * block_bytes);
    for outer_index in 0..outer {
        for &axis_index in first_indices {
            let start = (outer_index * axis_len + axis_index) * block_bytes;
            output.extend_from_slice(&dense[start..start + block_bytes]);
        }
    }
    output
}

fn compare_items(dtype: DataType, a: &[u8], b: &[u8]) -> Ordering {
    let size = dtype.byte_size();
    for (a, b) in a.chunks_exact(size).zip(b.chunks_exact(size)) {
        let ordering = compare_element(dtype, a, b);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

fn compare_element(dtype: DataType, a: &[u8], b: &[u8]) -> Ordering {
    macro_rules! compare {
        ($ty:ty) => {{
            let a = <$ty>::from_le_bytes(a.try_into().unwrap());
            let b = <$ty>::from_le_bytes(b.try_into().unwrap());
            a.cmp(&b)
        }};
    }
    macro_rules! compare_float {
        ($ty:ty) => {{
            let a = <$ty>::from_le_bytes(a.try_into().unwrap());
            let b = <$ty>::from_le_bytes(b.try_into().unwrap());
            a.partial_cmp(&b).unwrap_or_else(|| a.total_cmp(&b))
        }};
    }
    match dtype {
        DataType::Bool | DataType::Uint8 => a[0].cmp(&b[0]),
        DataType::Int8 => (a[0] as i8).cmp(&(b[0] as i8)),
        DataType::Uint16 => compare!(u16),
        DataType::Int16 => compare!(i16),
        DataType::Uint32 => compare!(u32),
        DataType::Int32 => compare!(i32),
        DataType::Uint64 => compare!(u64),
        DataType::Int64 => compare!(i64),
        DataType::Float16 => {
            let a = half::f16::from_le_bytes(a.try_into().unwrap());
            let b = half::f16::from_le_bytes(b.try_into().unwrap());
            a.partial_cmp(&b)
                .unwrap_or_else(|| a.to_bits().cmp(&b.to_bits()))
        }
        DataType::BFloat16 => {
            let a = half::bf16::from_le_bytes(a.try_into().unwrap());
            let b = half::bf16::from_le_bytes(b.try_into().unwrap());
            a.partial_cmp(&b)
                .unwrap_or_else(|| a.to_bits().cmp(&b.to_bits()))
        }
        DataType::Float32 => compare_float!(f32),
        DataType::Float64 => compare_float!(f64),
        _ => unreachable!("unsupported Unique dtype was validated"),
    }
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize> {
    let rank = i64::try_from(rank)
        .map_err(|_| EpError::KernelFailed("Unique: input rank is too large".into()))?;
    let axis = if axis < 0 { axis + rank } else { axis };
    if !(0..rank).contains(&axis) {
        return Err(EpError::KernelFailed(format!(
            "Unique: axis {axis} is out of range for rank {rank}"
        )));
    }
    Ok(axis as usize)
}

fn ensure_supported_dtype(dtype: DataType) -> Result<()> {
    match dtype {
        DataType::Bool
        | DataType::Uint8
        | DataType::Int8
        | DataType::Uint16
        | DataType::Int16
        | DataType::Uint32
        | DataType::Int32
        | DataType::Uint64
        | DataType::Int64
        | DataType::Float16
        | DataType::BFloat16
        | DataType::Float32
        | DataType::Float64 => Ok(()),
        _ => Err(EpError::KernelFailed(format!(
            "Unique: dtype {dtype:?} is unsupported"
        ))),
    }
}

fn validate_output(output: &TensorMut, dtype: DataType, shape: &[usize], name: &str) -> Result<()> {
    if output.dtype != dtype || output.shape != shape {
        return Err(EpError::KernelFailed(format!(
            "Unique: {name} must have dtype {dtype:?} and shape {shape:?}, got {:?}{:?}",
            output.dtype, output.shape
        )));
    }
    Ok(())
}

fn write_i64(output: &mut TensorMut, values: &[usize]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for &value in values {
        let value = i64::try_from(value)
            .map_err(|_| EpError::KernelFailed("Unique: index exceeds i64 range".into()))?;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    write_dense_bytes(output, &bytes)
}

fn optional_int_attr(node: &Node, name: &str) -> Result<Option<i64>> {
    match node.attr(name) {
        None => Ok(None),
        Some(Attribute::Int(value)) => Ok(Some(*value)),
        Some(_) => Err(EpError::KernelFailed(format!(
            "Unique: `{name}` must be an integer"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn sorts_axis_slices_lexicographically() {
        let input = Owned::f32(
            &[2, 4, 2],
            &[
                1., 1., 0., 1., 2., 1., 0., 1., 1., 1., 0., 1., 2., 1., 0., 1.,
            ],
        );
        let mut y = Owned::zeros_f32(&[2, 3, 2]);
        let mut indices = Owned::zeros(DataType::Int64, &[3]);
        let mut inverse = Owned::zeros(DataType::Int64, &[4]);
        let mut counts = Owned::zeros(DataType::Int64, &[3]);

        UniqueKernel {
            axis: Some(1),
            sorted: true,
        }
        .execute(
            &[input.view()],
            &mut [
                y.view_mut(),
                indices.view_mut(),
                inverse.view_mut(),
                counts.view_mut(),
            ],
        )
        .unwrap();

        assert_eq!(
            y.to_f32(),
            vec![0., 1., 1., 1., 2., 1., 0., 1., 1., 1., 2., 1.]
        );
        assert_eq!(indices.to_i64(), vec![1, 0, 2]);
        assert_eq!(inverse.to_i64(), vec![1, 0, 2, 0]);
        assert_eq!(counts.to_i64(), vec![2, 1, 1]);
    }

    #[test]
    fn unsorted_flattened_values_keep_first_appearance() {
        let input = Owned::f32(&[6], &[2., 1., 1., 3., 4., 3.]);
        let mut y = Owned::zeros_f32(&[4]);
        let mut indices = Owned::zeros(DataType::Int64, &[4]);
        let mut inverse = Owned::zeros(DataType::Int64, &[6]);
        let mut counts = Owned::zeros(DataType::Int64, &[4]);

        UniqueKernel {
            axis: None,
            sorted: false,
        }
        .execute(
            &[input.view()],
            &mut [
                y.view_mut(),
                indices.view_mut(),
                inverse.view_mut(),
                counts.view_mut(),
            ],
        )
        .unwrap();

        assert_eq!(y.to_f32(), vec![2., 1., 3., 4.]);
        assert_eq!(indices.to_i64(), vec![0, 1, 3, 4]);
        assert_eq!(inverse.to_i64(), vec![0, 1, 1, 2, 3, 2]);
        assert_eq!(counts.to_i64(), vec![1, 2, 2, 1]);
    }
}
