//! ONNX `Unique` (opset 11+) over flattened elements or slices along an axis.

use std::cmp::Ordering;

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, KernelSizedOutput, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{elem_size, to_dense_bytes, write_dense_bytes};
use crate::dtype::unsupported_dtype;
use crate::strided::numel;

#[cfg(test)]
std::thread_local! {
    static UNIQUE_PLAN_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

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
        let requested_outputs: Vec<bool> =
            outputs.iter().map(|output| !output.is_absent()).collect();
        let owned = self.compute_owned_outputs(inputs, &requested_outputs)?;
        if owned.len() != outputs.len() {
            return Err(EpError::KernelFailed(format!(
                "Unique: internal output count mismatch: produced {}, expected {}",
                owned.len(),
                outputs.len()
            )));
        }

        for (slot, (output, value)) in outputs.iter_mut().zip(owned).enumerate() {
            let Some(value) = value else {
                if !output.is_absent() {
                    return Err(EpError::KernelFailed(format!(
                        "Unique: output slot {slot} was requested but not produced"
                    )));
                }
                continue;
            };
            validate_output(output, value.dtype, &value.shape, output_name(slot))?;
            write_dense_bytes(output, &value.bytes)?;
        }
        Ok(())
    }

    fn has_kernel_sized_outputs(&self) -> bool {
        true
    }

    fn execute_kernel_sized(
        &self,
        inputs: &[TensorView],
        requested_outputs: &[bool],
    ) -> Result<Vec<Option<KernelSizedOutput>>> {
        self.compute_owned_outputs(inputs, requested_outputs)
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx == 0
    }
}

impl UniqueKernel {
    fn compute_owned_outputs(
        &self,
        inputs: &[TensorView],
        requested_outputs: &[bool],
    ) -> Result<Vec<Option<KernelSizedOutput>>> {
        if inputs.len() != 1 || !(1..=4).contains(&requested_outputs.len()) {
            return Err(EpError::KernelFailed(format!(
                "Unique: expected 1 input and 1..=4 output slots, got {} inputs and {} slots",
                inputs.len(),
                requested_outputs.len()
            )));
        }
        if !requested_outputs[0] {
            return Err(EpError::KernelFailed(
                "Unique: required output Y (slot 0) is absent".into(),
            ));
        }

        let input = &inputs[0];
        if !input.device.is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "Unique: kernel-sized output execution is host-only, but input 0 is on {:?}; \
                 place Unique on a host EP instead of copying device payloads implicitly",
                input.device
            )));
        }
        ensure_supported_dtype(input.dtype)?;

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
        let unique_len = plan.first_indices.len();
        let y_shape = match plan.axis {
            Some(axis) => {
                let mut shape = input.shape.to_vec();
                shape[axis] = unique_len;
                shape
            }
            None => vec![unique_len],
        };
        let mut outputs = Vec::with_capacity(requested_outputs.len());
        for (slot, &requested) in requested_outputs.iter().enumerate() {
            if !requested {
                outputs.push(None);
                continue;
            }
            let output = match slot {
                0 => KernelSizedOutput {
                    shape: y_shape.clone(),
                    dtype: input.dtype,
                    bytes: gather_y(
                        &dense,
                        input.shape,
                        element_size,
                        plan.axis,
                        &plan.first_indices,
                    ),
                },
                1 => KernelSizedOutput {
                    shape: vec![unique_len],
                    dtype: DataType::Int64,
                    bytes: encode_i64(&plan.first_indices)?,
                },
                2 => KernelSizedOutput {
                    shape: vec![plan.inverse_indices.len()],
                    dtype: DataType::Int64,
                    bytes: encode_i64(&plan.inverse_indices)?,
                },
                3 => KernelSizedOutput {
                    shape: vec![unique_len],
                    dtype: DataType::Int64,
                    bytes: encode_i64(&plan.counts)?,
                },
                _ => unreachable!("output slot count was checked"),
            };
            outputs.push(Some(output));
        }
        Ok(outputs)
    }
}

fn output_name(slot: usize) -> &'static str {
    match slot {
        0 => "Y",
        1 => "indices",
        2 => "inverse_indices",
        3 => "counts",
        _ => "unknown",
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
    #[cfg(test)]
    UNIQUE_PLAN_RUNS.with(|runs| runs.set(runs.get() + 1));

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
        | DataType::Float64 => Ok(()),
        _ => Err(unsupported_dtype("Unique", dtype)),
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

fn encode_i64(values: &[usize]) -> Result<Vec<u8>> {
    let capacity = values
        .len()
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| EpError::KernelFailed("Unique: int64 output byte length overflow".into()))?;
    let mut bytes = Vec::with_capacity(capacity);
    for &value in values {
        let value = i64::try_from(value)
            .map_err(|_| EpError::KernelFailed("Unique: index exceeds i64 range".into()))?;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
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

    fn output_f32(output: &KernelSizedOutput) -> Vec<f32> {
        output
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn output_i64(output: &KernelSizedOutput) -> Vec<i64> {
        output
            .bytes
            .chunks_exact(8)
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
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

    #[test]
    fn kernel_sized_all_outputs_plan_once_and_preserve_mixed_dtypes() {
        UNIQUE_PLAN_RUNS.with(|runs| runs.set(0));
        let input = Owned::f32(&[6], &[2., 1., 1., 3., 4., 3.]);
        let outputs = UniqueKernel {
            axis: None,
            sorted: false,
        }
        .execute_kernel_sized(&[input.view()], &[true, true, true, true])
        .unwrap();

        assert_eq!(
            UNIQUE_PLAN_RUNS.with(std::cell::Cell::get),
            1,
            "one deferred execution must run Unique planning exactly once"
        );
        let outputs: Vec<_> = outputs
            .iter()
            .map(|output| output.as_ref().unwrap())
            .collect();
        assert_eq!(outputs[0].shape, [4]);
        assert_eq!(outputs[0].dtype, DataType::Float32);
        assert_eq!(output_f32(outputs[0]), [2., 1., 3., 4.]);
        assert_eq!(outputs[1].dtype, DataType::Int64);
        assert_eq!(output_i64(outputs[1]), [0, 1, 3, 4]);
        assert_eq!(outputs[2].shape, [6]);
        assert_eq!(output_i64(outputs[2]), [0, 1, 1, 2, 3, 2]);
        assert_eq!(output_i64(outputs[3]), [1, 2, 2, 1]);
    }

    #[test]
    fn kernel_sized_optional_subset_does_not_materialize_absent_slots() {
        let input = Owned::f32(&[5], &[3., 1., 3., 2., 1.]);
        let outputs = UniqueKernel {
            axis: None,
            sorted: true,
        }
        .execute_kernel_sized(&[input.view()], &[true, false, true, false])
        .unwrap();

        assert_eq!(output_f32(outputs[0].as_ref().unwrap()), [1., 2., 3.]);
        assert!(outputs[1].is_none());
        assert_eq!(output_i64(outputs[2].as_ref().unwrap()), [2, 0, 2, 1, 0]);
        assert!(outputs[3].is_none());
    }

    #[test]
    fn kernel_sized_empty_all_equal_and_all_distinct() {
        for (values, expected) in [
            (&[][..], &[][..]),
            (&[7., 7., 7.][..], &[7.][..]),
            (&[3., 1., 2.][..], &[1., 2., 3.][..]),
        ] {
            let input = Owned::f32(&[values.len()], values);
            let outputs = UniqueKernel {
                axis: None,
                sorted: true,
            }
            .execute_kernel_sized(&[input.view()], &[true])
            .unwrap();
            assert_eq!(output_f32(outputs[0].as_ref().unwrap()), expected);
        }
    }

    #[test]
    fn kernel_sized_accepts_strided_input() {
        let input = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]).with_view(&[3, 2], &[1, 3]);
        let outputs = UniqueKernel {
            axis: None,
            sorted: false,
        }
        .execute_kernel_sized(&[input.view()], &[true])
        .unwrap();
        assert_eq!(
            output_f32(outputs[0].as_ref().unwrap()),
            [1., 4., 2., 5., 3., 6.]
        );
    }

    #[test]
    fn kernel_sized_rejects_device_input_before_planning() {
        use onnx_runtime_ep_api::DevicePtr;
        use onnx_runtime_ir::DeviceId;

        UNIQUE_PLAN_RUNS.with(|runs| runs.set(0));
        let values = [1.0f32, 2.0];
        let view = TensorView::new(
            DevicePtr(values.as_ptr().cast()),
            DataType::Float32,
            &[2],
            &[1],
            DeviceId::cuda(0),
        );
        let error = UniqueKernel {
            axis: None,
            sorted: true,
        }
        .execute_kernel_sized(&[view], &[true])
        .unwrap_err();
        assert!(error.to_string().contains("host-only"));
        assert_eq!(UNIQUE_PLAN_RUNS.with(std::cell::Cell::get), 0);
    }
}
