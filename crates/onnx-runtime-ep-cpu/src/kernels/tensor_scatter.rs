//! `TensorScatter`: fixed-capacity key/value-cache updates with per-batch cursors.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{check_arity, elem_size, to_dense_bytes, to_dense_i64, write_dense_bytes};
use crate::strided::numel;

pub struct TensorScatterKernel {
    axis: i64,
}

pub struct TensorScatterFactory;

impl KernelFactory for TensorScatterFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        if let Some(mode) = node.attr("mode").and_then(Attribute::as_str)
            && mode != "linear"
        {
            return Err(EpError::KernelFailed(format!(
                "TensorScatter: unsupported mode {mode:?}; only \"linear\" is implemented"
            )));
        }
        Ok(Box::new(TensorScatterKernel {
            axis: node.attr("axis").and_then(Attribute::as_int).unwrap_or(-2),
        }))
    }
}

impl Kernel for TensorScatterKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("TensorScatter", inputs, outputs, 3, 3, 1)?;
        let cache = &inputs[0];
        let updates = &inputs[1];
        let write_indices = &inputs[2];

        if !matches!(cache.dtype, DataType::Float32 | DataType::Float16) {
            return Err(EpError::KernelFailed(format!(
                "TensorScatter: cache dtype {:?} is not implemented; expected Float32 or Float16",
                cache.dtype
            )));
        }
        if updates.dtype != cache.dtype || outputs[0].dtype != cache.dtype {
            return Err(EpError::KernelFailed(
                "TensorScatter: cache, updates, and output must share a dtype".into(),
            ));
        }
        if !matches!(write_indices.dtype, DataType::Int32 | DataType::Int64) {
            return Err(EpError::KernelFailed(format!(
                "TensorScatter: write_indices must be Int32 or Int64, got {:?}",
                write_indices.dtype
            )));
        }
        if cache.shape.len() < 2 {
            return Err(EpError::KernelFailed(
                "TensorScatter: cache rank must be at least 2".into(),
            ));
        }
        if updates.shape.len() != cache.shape.len() {
            return Err(EpError::KernelFailed(format!(
                "TensorScatter: updates rank {} must match cache rank {}",
                updates.shape.len(),
                cache.shape.len()
            )));
        }
        if outputs[0].shape != cache.shape {
            return Err(EpError::KernelFailed(
                "TensorScatter: output shape must match cache shape".into(),
            ));
        }

        let rank = cache.shape.len();
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if axis <= 0 || axis as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "TensorScatter: axis {} is out of range for rank {rank}; axis 0 is the batch dimension",
                self.axis
            )));
        }
        let axis = axis as usize;
        for dimension in 0..rank {
            if dimension != axis && updates.shape[dimension] != cache.shape[dimension] {
                return Err(EpError::KernelFailed(format!(
                    "TensorScatter: updates dimension {} at axis {dimension} must match cache dimension {}",
                    updates.shape[dimension], cache.shape[dimension]
                )));
            }
        }

        let batch_size = cache.shape[0];
        if write_indices.shape != [batch_size] {
            return Err(EpError::KernelFailed(format!(
                "TensorScatter: write_indices shape {:?} must be [{batch_size}]",
                write_indices.shape
            )));
        }

        let write_indices = to_dense_i64(write_indices)?;
        let sequence_length = updates.shape[axis];
        let max_sequence_length = cache.shape[axis];
        for (batch, &write_index) in write_indices.iter().enumerate() {
            if write_index < 0 {
                return Err(EpError::KernelFailed(format!(
                    "TensorScatter: write_indices[{batch}] = {write_index} is negative"
                )));
            }
            let position = usize::try_from(write_index).map_err(|_| {
                EpError::KernelFailed(format!(
                    "TensorScatter: write_indices[{batch}] = {write_index} exceeds addressable memory"
                ))
            })?;
            let end = position.checked_add(sequence_length).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "TensorScatter: write_indices[{batch}] + sequence length overflows"
                ))
            })?;
            if end > max_sequence_length {
                return Err(EpError::KernelFailed(format!(
                    "TensorScatter: write_indices[{batch}] + sequence length ({position} + {sequence_length}) exceeds cache axis length {max_sequence_length}"
                )));
            }
        }

        let element_size = elem_size(cache.dtype)?;
        let suffix_elements = numel(&cache.shape[axis + 1..]);
        let suffix_bytes = suffix_elements
            .checked_mul(element_size)
            .ok_or_else(|| EpError::KernelFailed("TensorScatter: byte size overflow".into()))?;
        let prefixes_per_batch = numel(&cache.shape[1..axis]);
        let prefix_count = batch_size
            .checked_mul(prefixes_per_batch)
            .ok_or_else(|| EpError::KernelFailed("TensorScatter: prefix count overflow".into()))?;
        let update_prefix_bytes = sequence_length
            .checked_mul(suffix_bytes)
            .ok_or_else(|| EpError::KernelFailed("TensorScatter: update size overflow".into()))?;
        let cache_prefix_bytes = max_sequence_length
            .checked_mul(suffix_bytes)
            .ok_or_else(|| EpError::KernelFailed("TensorScatter: cache size overflow".into()))?;

        let mut output = to_dense_bytes(cache)?;
        let updates = to_dense_bytes(updates)?;
        for prefix in 0..prefix_count {
            let batch = prefix / prefixes_per_batch;
            let destination_sequence = write_indices[batch] as usize;
            let source_byte = prefix.checked_mul(update_prefix_bytes).ok_or_else(|| {
                EpError::KernelFailed("TensorScatter: source offset overflow".into())
            })?;
            let destination_byte = prefix
                .checked_mul(cache_prefix_bytes)
                .and_then(|offset| {
                    destination_sequence
                        .checked_mul(suffix_bytes)
                        .and_then(|sequence_offset| offset.checked_add(sequence_offset))
                })
                .ok_or_else(|| {
                    EpError::KernelFailed("TensorScatter: destination offset overflow".into())
                })?;
            output[destination_byte..destination_byte + update_prefix_bytes]
                .copy_from_slice(&updates[source_byte..source_byte + update_prefix_bytes]);
        }
        write_dense_bytes(&mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::NodeId;

    fn execute(
        kernel: &TensorScatterKernel,
        cache: &Owned,
        updates: &Owned,
        write_indices: &Owned,
        output: &mut Owned,
    ) -> Result<()> {
        kernel.execute(
            &[cache.view(), updates.view(), write_indices.view()],
            &mut [output.view_mut()],
        )
    }

    #[test]
    fn tensor_scatter_f32_sequence_one_uses_distinct_int64_batch_cursors() {
        let cache = Owned::f32(
            &[2, 4, 1, 2],
            &[
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
            ],
        );
        let updates = Owned::f32(&[2, 1, 1, 2], &[100., 101., 200., 201.]);
        let write_indices = Owned::i64(&[2], &[1, 3]);
        let mut output = Owned::f32(&[2, 4, 1, 2], &[0.; 16]);

        execute(
            &TensorScatterKernel { axis: 1 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap();

        assert_eq!(
            output.to_f32(),
            vec![
                0., 1., 100., 101., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 200., 201.
            ]
        );
    }

    #[test]
    fn tensor_scatter_f16_sequence_many_preserves_bits_with_int32_indices() {
        let cache_bits: Vec<u16> = (0..24).map(|value| 0x3000 + value).collect();
        let update_bits = vec![
            0x7e01, 0x0001, 0x7c00, 0xfc00, 0x3555, 0x3c00, 0x4000, 0x4200,
        ];
        let cache = Owned::f16_bits(&[2, 6, 2], &cache_bits);
        let updates = Owned::f16_bits(&[2, 2, 2], &update_bits);
        let write_indices = Owned::i32(&[2], &[1, 4]);
        let mut output = Owned::f16_bits(&[2, 6, 2], &[0; 24]);

        execute(
            &TensorScatterKernel { axis: -2 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap();

        let mut expected = cache_bits;
        expected[2..6].copy_from_slice(&update_bits[..4]);
        expected[20..24].copy_from_slice(&update_bits[4..]);
        assert_eq!(output.to_u16_bits(), expected);
    }

    #[test]
    fn tensor_scatter_rejects_invalid_axis() {
        let cache = Owned::f32(&[1, 2], &[0., 1.]);
        let updates = Owned::f32(&[1, 1], &[2.]);
        let write_indices = Owned::i64(&[1], &[0]);
        let mut output = Owned::f32(&[1, 2], &[0.; 2]);
        let error = execute(
            &TensorScatterKernel { axis: 0 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap_err();
        assert!(error.to_string().contains("axis"));
    }

    #[test]
    fn tensor_scatter_rejects_non_axis_shape_mismatch() {
        let cache = Owned::f32(&[2, 3, 2], &[0.; 12]);
        let updates = Owned::f32(&[2, 1, 1], &[0.; 2]);
        let write_indices = Owned::i64(&[2], &[0, 0]);
        let mut output = Owned::f32(&[2, 3, 2], &[0.; 12]);
        let error = execute(
            &TensorScatterKernel { axis: 1 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap_err();
        assert!(error.to_string().contains("updates dimension"));
    }

    #[test]
    fn tensor_scatter_rejects_write_indices_shape_mismatch() {
        let cache = Owned::f32(&[2, 3], &[0.; 6]);
        let updates = Owned::f32(&[2, 1], &[0.; 2]);
        let write_indices = Owned::i64(&[2, 1], &[0, 0]);
        let mut output = Owned::f32(&[2, 3], &[0.; 6]);
        let error = execute(
            &TensorScatterKernel { axis: 1 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap_err();
        assert!(error.to_string().contains("write_indices shape"));
    }

    #[test]
    fn tensor_scatter_rejects_negative_write_index() {
        let cache = Owned::f32(&[1, 3], &[0.; 3]);
        let updates = Owned::f32(&[1, 1], &[1.]);
        let write_indices = Owned::i64(&[1], &[-1]);
        let mut output = Owned::f32(&[1, 3], &[0.; 3]);
        let error = execute(
            &TensorScatterKernel { axis: 1 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap_err();
        assert!(error.to_string().contains("negative"));
    }

    #[test]
    fn tensor_scatter_rejects_write_past_cache() {
        let cache = Owned::f32(&[1, 3], &[0.; 3]);
        let updates = Owned::f32(&[1, 2], &[1., 2.]);
        let write_indices = Owned::i64(&[1], &[2]);
        let mut output = Owned::f32(&[1, 3], &[0.; 3]);
        let error = execute(
            &TensorScatterKernel { axis: 1 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap_err();
        assert!(error.to_string().contains("exceeds cache axis length"));
    }

    #[test]
    fn tensor_scatter_rejects_unsupported_cache_dtype() {
        let cache = Owned::i32(&[1, 2], &[0, 1]);
        let updates = Owned::i32(&[1, 1], &[2]);
        let write_indices = Owned::i64(&[1], &[0]);
        let mut output = Owned::i32(&[1, 2], &[0; 2]);
        let error = execute(
            &TensorScatterKernel { axis: 1 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap_err();
        assert!(error.to_string().contains("cache dtype"));
    }

    #[test]
    fn tensor_scatter_rejects_mismatched_update_dtype() {
        let cache = Owned::f32(&[1, 2], &[0., 1.]);
        let updates = Owned::f16(&[1, 1], &[2.]);
        let write_indices = Owned::i64(&[1], &[0]);
        let mut output = Owned::f32(&[1, 2], &[0.; 2]);
        let error = execute(
            &TensorScatterKernel { axis: 1 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap_err();
        assert!(error.to_string().contains("share a dtype"));
    }

    #[test]
    fn tensor_scatter_rejects_invalid_cache_rank() {
        let cache = Owned::f32(&[2], &[0., 1.]);
        let updates = Owned::f32(&[1], &[2.]);
        let write_indices = Owned::i64(&[2], &[0, 0]);
        let mut output = Owned::f32(&[2], &[0.; 2]);
        let error = execute(
            &TensorScatterKernel { axis: -1 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap_err();
        assert!(error.to_string().contains("rank must be at least 2"));
    }

    #[test]
    fn tensor_scatter_rejects_updates_rank_mismatch() {
        let cache = Owned::f32(&[1, 2, 1], &[0., 1.]);
        let updates = Owned::f32(&[1, 1], &[2.]);
        let write_indices = Owned::i64(&[1], &[0]);
        let mut output = Owned::f32(&[1, 2, 1], &[0.; 2]);
        let error = execute(
            &TensorScatterKernel { axis: 1 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap_err();
        assert!(error.to_string().contains("updates rank"));
    }

    #[test]
    fn tensor_scatter_rejects_output_shape_mismatch() {
        let cache = Owned::f32(&[1, 2], &[0., 1.]);
        let updates = Owned::f32(&[1, 1], &[2.]);
        let write_indices = Owned::i64(&[1], &[0]);
        let mut output = Owned::f32(&[1, 1], &[0.]);
        let error = execute(
            &TensorScatterKernel { axis: 1 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap_err();
        assert!(error.to_string().contains("output shape"));
    }

    #[test]
    fn tensor_scatter_rejects_unsupported_write_indices_dtype() {
        let cache = Owned::f32(&[1, 2], &[0., 1.]);
        let updates = Owned::f32(&[1, 1], &[2.]);
        let write_indices = Owned::f32(&[1], &[0.]);
        let mut output = Owned::f32(&[1, 2], &[0.; 2]);
        let error = execute(
            &TensorScatterKernel { axis: 1 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap_err();
        assert!(error.to_string().contains("write_indices must be"));
    }

    #[test]
    fn tensor_scatter_factory_rejects_unsupported_mode() {
        let mut node = Node::new(NodeId(0), "TensorScatter", vec![], vec![]);
        node.attributes
            .insert("mode".into(), Attribute::String("circular".into()));
        let error = match TensorScatterFactory.create(&node, &[]) {
            Ok(_) => panic!("circular mode must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("unsupported mode"));
    }

    #[test]
    fn tensor_scatter_is_registered_for_exported_and_compatibility_domains() {
        let registry = crate::kernels::build_cpu_registry();
        assert!(registry.supports("TensorScatter", "", 24));
        assert!(registry.supports("TensorScatter", "com.microsoft", 1));
    }

    #[test]
    fn tensor_scatter_accepts_strided_input_contract() {
        let mut cache = Owned::f32(&[1, 3, 2], &[0., 1., 2., 3., 4., 5.]);
        cache.strides = vec![6, 1, 3];
        let updates = Owned::f32(&[1, 1, 2], &[10., 11.]);
        let write_indices = Owned::i64(&[1], &[1]);
        let mut output = Owned::f32(&[1, 3, 2], &[0.; 6]);

        execute(
            &TensorScatterKernel { axis: 1 },
            &cache,
            &updates,
            &write_indices,
            &mut output,
        )
        .unwrap();

        assert_eq!(output.to_f32(), vec![0., 3., 10., 11., 2., 5.]);
    }
}
