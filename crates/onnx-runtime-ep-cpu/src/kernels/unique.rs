//! ONNX `Unique` (opset 11+) over flattened elements or slices along an axis.

use std::cmp::Ordering;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{elem_size, to_dense_bytes, write_dense_bytes};
use crate::strided::{elem_offset, next_index, numel};

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
        ensure_supported_dtype(input.dtype)?;
        if input.dtype == DataType::String {
            return execute_strings(input, outputs, self.axis, self.sorted);
        }

        let element_size = elem_size(input.dtype)?;
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

fn execute_strings(
    input: &TensorView,
    outputs: &mut [TensorMut],
    axis: Option<i64>,
    sorted: bool,
) -> Result<()> {
    let dense = to_dense_strings(input)?;
    let axis = axis
        .map(|axis| normalize_axis(axis, input.shape.len()))
        .transpose()?;
    let items = make_string_items(&dense, input.shape, axis);
    let item_count = items.len();
    let (first_indices, inverse_indices, counts) =
        unique_groups_by(item_count, sorted, |a, b| items[a].cmp(&items[b]));

    let expected_y_shape = match axis {
        Some(axis) => {
            let mut shape = input.shape.to_vec();
            shape[axis] = first_indices.len();
            shape
        }
        None => vec![first_indices.len()],
    };
    validate_output(&outputs[0], DataType::String, &expected_y_shape, "Y")?;
    let y = gather_string_y(&dense, input.shape, axis, &first_indices);
    write_dense_strings(&mut outputs[0], &y)?;

    let unique_shape = [first_indices.len()];
    if outputs.len() >= 2 {
        validate_output(&outputs[1], DataType::Int64, &unique_shape, "indices")?;
        write_i64(&mut outputs[1], &first_indices)?;
    }
    if outputs.len() >= 3 {
        let inverse_shape = [inverse_indices.len()];
        validate_output(
            &outputs[2],
            DataType::Int64,
            &inverse_shape,
            "inverse_indices",
        )?;
        write_i64(&mut outputs[2], &inverse_indices)?;
    }
    if outputs.len() == 4 {
        validate_output(&outputs[3], DataType::Int64, &unique_shape, "counts")?;
        write_i64(&mut outputs[3], &counts)?;
    }
    Ok(())
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

    let (first_indices, inverse_indices, counts) = unique_groups(&items, dtype, item_count, sorted);

    Ok(UniquePlan {
        axis,
        first_indices,
        inverse_indices,
        counts,
    })
}

fn unique_groups(
    items: &[Vec<u8>],
    dtype: DataType,
    item_count: usize,
    sorted: bool,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    unique_groups_by(item_count, sorted, |a, b| {
        compare_items(dtype, &items[a], &items[b])
    })
}

fn unique_groups_by(
    item_count: usize,
    sorted: bool,
    compare: impl Fn(usize, usize) -> Ordering,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..item_count).collect();
    order.sort_unstable_by(|&a, &b| compare(a, b).then_with(|| a.cmp(&b)));

    let mut first_indices = Vec::new();
    let mut inverse_indices = vec![0; item_count];
    let mut counts = Vec::new();
    let mut previous_index: Option<usize> = None;
    for &index in &order {
        let new_group =
            previous_index.is_none_or(|previous| compare(previous, index) != Ordering::Equal);
        if new_group {
            first_indices.push(index);
            counts.push(0);
        } else if index < *first_indices.last().unwrap() {
            *first_indices.last_mut().unwrap() = index;
        }
        let group = counts.len() - 1;
        inverse_indices[index] = group;
        counts[group] += 1;
        previous_index = Some(index);
    }

    if !sorted {
        let mut group_order: Vec<usize> = (0..first_indices.len()).collect();
        group_order.sort_unstable_by_key(|&group| first_indices[group]);
        let mut sorted_to_unsorted = vec![0; group_order.len()];
        for (unsorted, &sorted) in group_order.iter().enumerate() {
            sorted_to_unsorted[sorted] = unsorted;
        }
        first_indices = group_order
            .iter()
            .map(|&group| first_indices[group])
            .collect();
        counts = group_order.iter().map(|&group| counts[group]).collect();
        for group in &mut inverse_indices {
            *group = sorted_to_unsorted[*group];
        }
    }

    (first_indices, inverse_indices, counts)
}

fn to_dense_strings(view: &TensorView) -> Result<Vec<String>> {
    validate_string_view(
        view.data.is_null(),
        view.dtype,
        view.shape,
        view.strides,
        view.byte_offset,
        view.device.is_host_accessible(),
    )?;
    let count = numel(view.shape);
    let origin = view.data_ptr::<String>();
    let mut output = Vec::with_capacity(count);
    if count == 0 {
        return Ok(output);
    }
    let mut index = vec![0usize; view.shape.len()];
    loop {
        let offset = elem_offset(view.strides, &index);
        // SAFETY: string tensor views point at an out-of-band array of initialized
        // `String` values; the validated shape/strides describe live elements.
        output.push(unsafe { (*origin.offset(offset)).clone() });
        if !next_index(view.shape, &mut index) {
            break;
        }
    }
    Ok(output)
}

fn write_dense_strings(output: &mut TensorMut, values: &[String]) -> Result<()> {
    validate_string_view(
        output.data.is_null(),
        output.dtype,
        output.shape,
        output.strides,
        output.byte_offset,
        output.device.is_host_accessible(),
    )?;
    if !output.is_contiguous() {
        return Err(EpError::InvalidTensorView {
            reason: "Unique: String output must be contiguous".into(),
        });
    }
    if values.len() != numel(output.shape) {
        return Err(EpError::KernelFailed(format!(
            "Unique: String output element count {} does not match produced {}",
            numel(output.shape),
            values.len()
        )));
    }
    let destination = output.data_ptr_mut::<String>();
    for (index, value) in values.iter().enumerate() {
        // SAFETY: string output storage is an out-of-band array of initialized
        // `String` slots with one slot per logical output element.
        unsafe {
            (*destination.add(index)).clone_from(value);
        }
    }
    Ok(())
}

fn validate_string_view(
    data_is_null: bool,
    dtype: DataType,
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    host_accessible: bool,
) -> Result<()> {
    if dtype != DataType::String
        || data_is_null
        || shape.len() != strides.len()
        || !byte_offset.is_multiple_of(std::mem::align_of::<String>())
        || !host_accessible
    {
        return Err(EpError::InvalidTensorView {
            reason: "Unique: invalid out-of-band String tensor view".into(),
        });
    }
    Ok(())
}

fn make_string_items(dense: &[String], shape: &[usize], axis: Option<usize>) -> Vec<Vec<String>> {
    let Some(axis) = axis else {
        return dense.iter().cloned().map(|value| vec![value]).collect();
    };
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut items = vec![Vec::with_capacity(outer * inner); axis_len];
    for outer_index in 0..outer {
        for (axis_index, item) in items.iter_mut().enumerate() {
            let start = (outer_index * axis_len + axis_index) * inner;
            item.extend_from_slice(&dense[start..start + inner]);
        }
    }
    items
}

fn gather_string_y(
    dense: &[String],
    shape: &[usize],
    axis: Option<usize>,
    first_indices: &[usize],
) -> Vec<String> {
    let Some(axis) = axis else {
        return first_indices
            .iter()
            .map(|&index| dense[index].clone())
            .collect();
    };
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut output = Vec::with_capacity(outer * first_indices.len() * inner);
    for outer_index in 0..outer {
        for &axis_index in first_indices {
            let start = (outer_index * axis_len + axis_index) * inner;
            output.extend_from_slice(&dense[start..start + inner]);
        }
    }
    output
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
    if dtype == DataType::String {
        return a.cmp(b);
    }
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
            match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => a.partial_cmp(&b).unwrap(),
            }
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
            match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => a.partial_cmp(&b).unwrap(),
            }
        }
        DataType::BFloat16 => {
            let a = half::bf16::from_le_bytes(a.try_into().unwrap());
            let b = half::bf16::from_le_bytes(b.try_into().unwrap());
            match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => a.partial_cmp(&b).unwrap(),
            }
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
        | DataType::Float64
        | DataType::String => Ok(()),
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
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
    use onnx_runtime_ir::{DeviceId, compute_contiguous_strides};

    struct StringOwned {
        values: Vec<String>,
        shape: Vec<usize>,
        strides: Vec<i64>,
    }

    impl StringOwned {
        fn new(shape: &[usize], values: &[&str]) -> Self {
            Self {
                values: values.iter().map(|value| (*value).to_owned()).collect(),
                shape: shape.to_vec(),
                strides: compute_contiguous_strides(shape),
            }
        }

        fn zeros(shape: &[usize]) -> Self {
            Self {
                values: vec![String::new(); numel(shape)],
                shape: shape.to_vec(),
                strides: compute_contiguous_strides(shape),
            }
        }

        fn view(&self) -> TensorView<'_> {
            TensorView::new(
                DevicePtr(self.values.as_ptr().cast()),
                DataType::String,
                &self.shape,
                &self.strides,
                DeviceId::cpu(),
            )
        }

        fn view_mut(&mut self) -> TensorMut<'_> {
            TensorMut::new(
                DevicePtrMut(self.values.as_mut_ptr().cast()),
                DataType::String,
                &self.shape,
                &self.strides,
                DeviceId::cpu(),
            )
        }
    }

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

    #[test]
    fn collapses_all_nan_payloads_and_signed_zero() {
        let first_nan = f32::from_bits(0x7fc0_0001);
        let second_nan = f32::from_bits(0x7fc0_1234);
        let input = Owned::f32(&[4], &[first_nan, second_nan, -0.0, 0.0]);
        let mut y = Owned::zeros_f32(&[2]);
        let mut indices = Owned::zeros(DataType::Int64, &[2]);
        let mut inverse = Owned::zeros(DataType::Int64, &[4]);
        let mut counts = Owned::zeros(DataType::Int64, &[2]);

        UniqueKernel {
            axis: None,
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

        let values = y.to_f32();
        assert_eq!(values[0], -0.0);
        assert!(values[1].is_nan());
        assert_eq!(indices.to_i64(), vec![2, 0]);
        assert_eq!(inverse.to_i64(), vec![1, 1, 0, 0]);
        assert_eq!(counts.to_i64(), vec![2, 2]);
    }

    #[test]
    fn string_tensor_runs_through_kernel_sorted_and_unsorted() {
        let input = StringOwned::new(
            &[9],
            &[
                "pear", "apple", "pear", "banana", "apple", "pear", "pear", "apple", "pear",
            ],
        );

        for (sorted, expected_y, expected_indices, expected_inverse, expected_counts) in [
            (
                true,
                vec!["apple", "banana", "pear"],
                vec![1, 3, 0],
                vec![2, 0, 2, 1, 0, 2, 2, 0, 2],
                vec![3, 1, 5],
            ),
            (
                false,
                vec!["pear", "apple", "banana"],
                vec![0, 1, 3],
                vec![0, 1, 0, 2, 1, 0, 0, 1, 0],
                vec![5, 3, 1],
            ),
        ] {
            let mut y = StringOwned::zeros(&[3]);
            let mut indices = Owned::zeros(DataType::Int64, &[3]);
            let mut inverse = Owned::zeros(DataType::Int64, &[9]);
            let mut counts = Owned::zeros(DataType::Int64, &[3]);

            UniqueKernel { axis: None, sorted }
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

            assert_eq!(y.values, expected_y);
            assert_eq!(indices.to_i64(), expected_indices);
            assert_eq!(inverse.to_i64(), expected_inverse);
            assert_eq!(counts.to_i64(), expected_counts);
        }
    }

    #[test]
    fn large_unique_input_uses_sort_and_linear_grouping() {
        let item_count = 50_000usize;
        let items: Vec<Vec<u8>> = (0..item_count)
            .rev()
            .map(|value| (value as u64).to_le_bytes().to_vec())
            .collect();

        let (indices, inverse, counts) = unique_groups(&items, DataType::Uint64, item_count, true);
        assert_eq!(indices.len(), item_count);
        assert_eq!(inverse.len(), item_count);
        assert!(counts.iter().all(|&count| count == 1));
        assert_eq!(indices[0], item_count - 1);
        assert_eq!(indices[item_count - 1], 0);
    }
}
