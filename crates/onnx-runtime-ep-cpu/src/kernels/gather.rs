//! `Gather`: index fixed-width `data` along `axis` with an integer `indices` tensor
//! (`docs/ORT2.md` §4.4). Output shape is
//! `data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:]`.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Node, is_contiguous};

use super::{check_arity, elem_size, to_dense_bytes, to_dense_i64, write_dense_bytes};
use crate::strided::numel;

fn checked_product(shape: &[usize], what: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| {
            EpError::KernelFailed(format!("Gather: {what} overflowed for shape {shape:?}"))
        })
    })
}

fn normalize_index(raw: i64, axis_dim: usize) -> Result<usize> {
    let normalized = if raw < 0 {
        raw as i128 + axis_dim as i128
    } else {
        raw as i128
    };
    if normalized < 0 || normalized >= axis_dim as i128 {
        return Err(EpError::KernelFailed(format!(
            "Gather: index {raw} out of range for axis dim {axis_dim}"
        )));
    }
    Ok(normalized as usize)
}

/// Dtype-agnostic Gather kernel carrying the resolved `axis`.
pub struct GatherKernel {
    /// Raw axis attribute (may be negative); normalized against data rank at run.
    axis: i64,
}

/// Factory reading the `axis` attribute (default 0) from the node.
pub struct GatherFactory;

impl KernelFactory for GatherFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(0);
        Ok(Box::new(GatherKernel { axis }))
    }
}

impl Kernel for GatherKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Gather", inputs, outputs, 2, 2, 1)?;
        let data = &inputs[0];
        if outputs[0].dtype != data.dtype {
            return Err(EpError::KernelFailed(format!(
                "Gather: output dtype {:?} must match data dtype {:?}",
                outputs[0].dtype, data.dtype
            )));
        }
        let esize = elem_size(data.dtype)?;
        let indices = to_dense_i64(&inputs[1])?;
        let data_shape = inputs[0].shape;
        let idx_shape = inputs[1].shape;
        let rank = data_shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(
                "Gather: data must have rank >= 1".into(),
            ));
        }

        // Normalize axis into [0, rank).
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if axis < 0 || axis as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "Gather: axis {} out of range for rank {rank}",
                self.axis
            )));
        }
        let axis = axis as usize;

        let axis_dim = data_shape[axis];
        let outer: usize = data_shape[..axis].iter().product();
        let inner: usize = data_shape[axis + 1..].iter().product();
        let num_idx = numel(idx_shape);

        if axis == 0
            && is_contiguous(data.shape, data.strides)
            && is_contiguous(outputs[0].shape, outputs[0].strides)
        {
            data.validate()?;
            outputs[0].validate()?;
            let row_elements = checked_product(&data_shape[1..], "axis-0 row element count")?;
            let expected_output_elements =
                indices.len().checked_mul(row_elements).ok_or_else(|| {
                    EpError::KernelFailed("Gather: axis-0 output element count overflowed".into())
                })?;
            let output_elements = checked_product(outputs[0].shape, "destination element count")?;
            if output_elements != expected_output_elements {
                return Err(EpError::KernelFailed(format!(
                    "Gather: axis-0 destination has {output_elements} elements, expected \
                     {expected_output_elements} for {} indices and {row_elements} elements per row",
                    indices.len()
                )));
            }
            let row_bytes = row_elements.checked_mul(esize).ok_or_else(|| {
                EpError::KernelFailed("Gather: axis-0 row byte count overflowed".into())
            })?;
            let source_elements = axis_dim.checked_mul(row_elements).ok_or_else(|| {
                EpError::KernelFailed("Gather: axis-0 source element count overflowed".into())
            })?;
            let source_bytes = source_elements.checked_mul(esize).ok_or_else(|| {
                EpError::KernelFailed("Gather: axis-0 source byte count overflowed".into())
            })?;
            let expected_output_bytes =
                expected_output_elements.checked_mul(esize).ok_or_else(|| {
                    EpError::KernelFailed("Gather: axis-0 output byte count overflowed".into())
                })?;

            // Validate every index before the first write, so an invalid later
            // index cannot leave a partially populated output.
            for &raw in &indices {
                normalize_index(raw, axis_dim)?;
            }

            let src = data.data_ptr::<u8>();
            let dst = outputs[0].data_ptr_mut::<u8>();
            for (output_row, &raw) in indices.iter().enumerate() {
                let idx = normalize_index(raw, axis_dim)?;
                let src_offset = idx.checked_mul(row_bytes).ok_or_else(|| {
                    EpError::KernelFailed("Gather: axis-0 source byte offset overflowed".into())
                })?;
                let dst_offset = output_row.checked_mul(row_bytes).ok_or_else(|| {
                    EpError::KernelFailed(
                        "Gather: axis-0 destination byte offset overflowed".into(),
                    )
                })?;
                let src_end = src_offset.checked_add(row_bytes).ok_or_else(|| {
                    EpError::KernelFailed("Gather: axis-0 source row end overflowed".into())
                })?;
                let dst_end = dst_offset.checked_add(row_bytes).ok_or_else(|| {
                    EpError::KernelFailed("Gather: axis-0 destination row end overflowed".into())
                })?;
                if src_end > source_bytes || dst_end > expected_output_bytes {
                    return Err(EpError::KernelFailed(
                        "Gather: axis-0 row copy exceeds validated tensor bounds".into(),
                    ));
                }
                // SAFETY: validated contiguous views plus the checked source index
                // and exact destination element count keep both row-sized regions
                // in bounds; executor SSA makes them disjoint.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.add(src_offset),
                        dst.add(dst_offset),
                        row_bytes,
                    );
                }
            }
            return Ok(());
        }

        let data = to_dense_bytes(data)?;
        let mut out = vec![0u8; outer * num_idx * inner * esize];
        let mut w = 0usize;
        for o in 0..outer {
            for &raw in &indices {
                let idx = if raw < 0 { raw + axis_dim as i64 } else { raw };
                if idx < 0 || idx as usize >= axis_dim {
                    return Err(EpError::KernelFailed(format!(
                        "Gather: index {raw} out of range for axis dim {axis_dim}"
                    )));
                }
                let base = (o * axis_dim + idx as usize) * inner;
                let len = inner * esize;
                let base = base * esize;
                out[w..w + len].copy_from_slice(&data[base..base + len]);
                w += len;
            }
        }
        write_dense_bytes(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{
        Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
    };
    use onnx_runtime_loader::Model;

    fn run_result(
        axis: i64,
        data: &Owned,
        idx: &Owned,
        out: &mut Owned,
    ) -> onnx_runtime_ep_api::Result<()> {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 13);
        let data_value =
            graph.create_named_value("data", data.dtype, static_shape(data.shape.iter().copied()));
        let indices_value = graph.create_named_value(
            "indices",
            idx.dtype,
            static_shape(idx.shape.iter().copied()),
        );
        let output =
            graph.create_named_value("output", out.dtype, static_shape(out.shape.iter().copied()));
        graph.add_input(data_value);
        graph.add_input(indices_value);
        let mut node = Node::new(
            NodeId(0),
            "Gather",
            vec![Some(data_value), Some(indices_value)],
            vec![output],
        );
        node.attributes.insert("axis".into(), Attribute::Int(axis));
        let node = graph.insert_node(node);
        graph.add_output(output);
        let model = Model::new(&graph);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 13)
            .unwrap()
            .execute(&[data.view(), idx.view()], &mut [out.view_mut()])
    }

    fn run(axis: i64, data: &Owned, idx: &Owned, out: &mut Owned) {
        run_result(axis, data, idx, out).unwrap();
    }

    fn take_rows(data: &[f32], columns: usize, indices: &[i64]) -> Vec<f32> {
        let rows = data.len() / columns;
        let mut expected = Vec::with_capacity(indices.len() * columns);
        for &raw in indices {
            let row = if raw < 0 {
                (rows as i64 + raw) as usize
            } else {
                raw as usize
            };
            expected.extend_from_slice(&data[row * columns..(row + 1) * columns]);
        }
        expected
    }

    #[test]
    fn gather_axis0_decode_single_row() {
        let values = [1., 2., 3., 4., 5., 6.];
        let indices = [2];
        let data = Owned::f32(&[3, 2], &values);
        let idx = Owned::i64(&[1], &indices);
        let mut out = Owned::zeros_f32(&[1, 2]);
        run(0, &data, &idx, &mut out);
        assert_eq!(out.to_f32(), take_rows(&values, 2, &indices));
    }

    #[test]
    fn gather_axis0_multiple_rows() {
        let values = [1., 2., 3., 4., 5., 6.];
        let indices = [2, 0];
        let data = Owned::f32(&[3, 2], &values);
        let idx = Owned::i64(&[2], &indices);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run(0, &data, &idx, &mut out);
        assert_eq!(out.to_f32(), take_rows(&values, 2, &indices));
    }

    #[test]
    fn gather_columns_axis1() {
        // data [2,3], indices [2] = [0,2], axis 1 -> [2,2]
        let data = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let idx = Owned::i64(&[2], &[0, 2]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run(1, &data, &idx, &mut out);
        // cols 0 and 2: [1,3] and [4,6]
        assert_eq!(out.to_f32(), vec![1., 3., 4., 6.]);
    }

    #[test]
    fn gather_negative_index() {
        let values = [1., 2., 3., 4., 5., 6.];
        let indices = [-1, -3];
        let data = Owned::f32(&[3, 2], &values);
        let idx = Owned::i64(&[2], &indices);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run(0, &data, &idx, &mut out);
        assert_eq!(out.to_f32(), take_rows(&values, 2, &indices));
    }

    #[test]
    fn gather_axis0_rejects_out_of_range_before_writing() {
        let data = Owned::f32(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let idx = Owned::i64(&[2], &[0, 3]);
        let mut out = Owned::f32(&[2, 2], &[9., 9., 9., 9.]);
        let err = run_result(0, &data, &idx, &mut out).unwrap_err();
        assert!(err.to_string().contains("index 3 out of range"));
        assert_eq!(out.to_f32(), vec![9., 9., 9., 9.]);
    }

    #[test]
    fn gather_axis0_rejects_mismatched_output_before_writing() {
        let data = Owned::f32(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let idx = Owned::i64(&[2], &[2, 0]);
        let mut out = Owned::f32(&[4, 2], &[9.; 8]);
        out.shape = vec![1, 2];
        out.strides = compute_contiguous_strides(&out.shape);

        let err = run_result(0, &data, &idx, &mut out).unwrap_err();
        assert!(
            err.to_string()
                .contains("destination has 2 elements, expected 4")
        );
        assert_eq!(out.to_f32(), vec![9.; 8]);
    }

    #[test]
    fn gather_noncontiguous_axis1_uses_general_path() {
        // Physical storage is column-major; strides expose logical rows
        // [[1, 2, 3], [4, 5, 6]].
        let mut data = Owned::f32(&[2, 3], &[1., 4., 2., 5., 3., 6.]);
        data.strides = vec![1, 2];
        let idx = Owned::i64(&[2], &[2, 0]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run(1, &data, &idx, &mut out);

        let logical = [[1., 2., 3.], [4., 5., 6.]];
        let expected: Vec<f32> = logical.iter().flat_map(|row| [row[2], row[0]]).collect();
        assert_eq!(out.to_f32(), expected);
    }

    #[test]
    fn gather_2d_indices_embedding() {
        // Embedding-style: data [4,2] (vocab x dim), indices [1,3] -> [1,3,2]
        let data = Owned::f32(&[4, 2], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let idx = Owned::i64(&[1, 3], &[0, 2, 3]);
        let mut out = Owned::zeros_f32(&[1, 3, 2]);
        run(0, &data, &idx, &mut out);
        assert_eq!(out.to_f32(), vec![0., 1., 4., 5., 6., 7.]);
    }

    #[test]
    fn gather_int64_shape_dimension_with_int32_indices() {
        let data = Owned::i64(&[4], &[8, 16, 32, 64]);
        let idx = Owned::i32(&[1], &[2]);
        let mut out = Owned::zeros(DataType::Int64, &[1]);
        run(0, &data, &idx, &mut out);
        assert_eq!(out.to_i64(), vec![32]);
    }

    #[test]
    fn gather_int64_multidim_indices_negative_axis() {
        // data [2,3], indices [2,2], axis -1 -> [2,2,2].
        let data = Owned::i64(&[2, 3], &[10, 20, 30, 40, 50, 60]);
        let idx = Owned::i64(&[2, 2], &[2, 0, 1, 2]);
        let mut out = Owned::zeros(DataType::Int64, &[2, 2, 2]);
        run(-1, &data, &idx, &mut out);
        assert_eq!(out.to_i64(), vec![30, 10, 20, 30, 60, 40, 50, 60]);
    }

    #[test]
    fn gather_int64_negative_index_wraps() {
        let data = Owned::i64(&[3], &[11, 22, 33]);
        let idx = Owned::i64(&[1], &[-1]);
        let mut out = Owned::zeros(DataType::Int64, &[1]);
        run(0, &data, &idx, &mut out);
        assert_eq!(out.to_i64(), vec![33]);
    }
    #[test]
    fn gather_bf16_preserves_element_bits() {
        let data = Owned::bf16(&[3], &[1., -2., 3.]);
        let indices = Owned::i64(&[2], &[2, 0]);
        let mut out = Owned::zeros(DataType::BFloat16, &[2]);
        run(0, &data, &indices, &mut out);
        assert_eq!(
            out.to_u16_bits(),
            vec![data.to_u16_bits()[2], data.to_u16_bits()[0]]
        );
    }
}
