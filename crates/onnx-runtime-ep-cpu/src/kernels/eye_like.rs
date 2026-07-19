//! ONNX `EyeLike`: an identity-like tensor using an input only for shape and dtype.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{check_arity, write_dense_bytes};
use crate::dtype::{NumericElem, write_dense};

pub struct EyeLikeKernel {
    k: i64,
    dtype: Option<DataType>,
}
pub struct EyeLikeFactory;

impl KernelFactory for EyeLikeFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = match node.attr("k") {
            None => 0,
            Some(Attribute::Int(k)) => *k,
            Some(_) => {
                return Err(EpError::KernelFailed(
                    "EyeLike: `k` must be an integer".into(),
                ));
            }
        };
        let dtype = match node.attr("dtype") {
            None => None,
            Some(Attribute::Int(dtype)) => {
                let dtype = i32::try_from(*dtype).map_err(|_| {
                    EpError::KernelFailed(format!("EyeLike: invalid dtype {dtype}"))
                })?;
                Some(DataType::from_onnx(dtype).ok_or_else(|| {
                    EpError::KernelFailed(format!("EyeLike: invalid dtype {dtype}"))
                })?)
            }
            Some(_) => {
                return Err(EpError::KernelFailed(
                    "EyeLike: `dtype` must be an integer".into(),
                ));
            }
        };
        Ok(Box::new(EyeLikeKernel { k, dtype }))
    }
}

impl Kernel for EyeLikeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("EyeLike", inputs, outputs, 1, 1, 1)?;
        if inputs[0].shape.len() != 2 {
            return Err(EpError::KernelFailed(
                "EyeLike: input must be rank 2".into(),
            ));
        }
        if outputs[0].shape != inputs[0].shape {
            return Err(EpError::KernelFailed(
                "EyeLike: output shape must match input".into(),
            ));
        }
        let dtype = self.dtype.unwrap_or(inputs[0].dtype);
        if outputs[0].dtype != dtype {
            return Err(EpError::KernelFailed(format!(
                "EyeLike: output dtype {:?} does not match expected {:?}",
                outputs[0].dtype, dtype
            )));
        }
        if dtype == DataType::Bool {
            return eye_bool(self.k, outputs);
        }
        crate::dispatch_arith!(dtype, "EyeLike", T => eye_typed::<T>(self.k, outputs))
    }
}

fn eye_typed<T: NumericElem>(k: i64, outputs: &mut [TensorMut]) -> Result<()> {
    let [rows, cols] = [outputs[0].shape[0], outputs[0].shape[1]];
    let mut values = vec![T::from_f32_scalar(0.0); rows * cols];
    for row in 0..rows {
        if let Some(col) = diagonal_col(row, cols, k) {
            values[row * cols + col] = T::from_f32_scalar(1.0);
        }
    }
    write_dense::<T>(&mut outputs[0], &values)
}

fn eye_bool(k: i64, outputs: &mut [TensorMut]) -> Result<()> {
    let [rows, cols] = [outputs[0].shape[0], outputs[0].shape[1]];
    let mut values = vec![0u8; rows * cols];
    for row in 0..rows {
        if let Some(col) = diagonal_col(row, cols, k) {
            values[row * cols + col] = 1;
        }
    }
    write_dense_bytes(&mut outputs[0], &values)
}

fn diagonal_col(row: usize, cols: usize, k: i64) -> Option<usize> {
    let row = i64::try_from(row).ok()?;
    let col = row.checked_add(k)?;
    usize::try_from(col).ok().filter(|&col| col < cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::NodeId;

    #[test]
    fn populates_offset_diagonal() {
        let input = Owned::f32(&[3, 4], &[0.; 12]);
        let mut output = Owned::zeros_f32(&[3, 4]);
        let mut node = Node::new(NodeId(0), "EyeLike", vec![], vec![]);
        node.attributes.insert("k".into(), Attribute::Int(1));
        EyeLikeFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(
            output.to_f32(),
            vec![0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]
        );
    }

    #[test]
    fn dtype_override_controls_output() {
        let input = Owned::f32(&[2, 3], &[0.; 6]);
        let mut output = Owned::zeros(DataType::Int32, &[2, 3]);
        let mut node = Node::new(NodeId(0), "EyeLike", vec![], vec![]);
        node.attributes
            .insert("dtype".into(), Attribute::Int(DataType::Int32 as i64));
        EyeLikeFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_i32(), vec![1, 0, 0, 0, 1, 0]);
    }

    #[test]
    fn defaults_to_input_dtype() {
        let input = Owned::f32(&[2, 2], &[0.; 4]);
        let mut output = Owned::zeros_f32(&[2, 2]);
        EyeLikeFactory
            .create(&Node::new(NodeId(0), "EyeLike", vec![], vec![]), &[])
            .unwrap()
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_f32(), vec![1., 0., 0., 1.]);
    }

    #[test]
    fn extreme_offsets_produce_an_all_zero_matrix() {
        let input = Owned::f32(&[3, 3], &[0.; 9]);
        for k in [i64::MAX, i64::MIN] {
            let mut output = Owned::zeros_f32(&[3, 3]);
            let mut node = Node::new(NodeId(0), "EyeLike", vec![], vec![]);
            node.attributes.insert("k".into(), Attribute::Int(k));
            EyeLikeFactory
                .create(&node, &[])
                .unwrap()
                .execute(&[input.view()], &mut [output.view_mut()])
                .unwrap();
            assert_eq!(output.to_f32(), vec![0.; 9], "k = {k}");
        }
    }

    #[test]
    fn large_in_range_offset_populates_the_diagonal() {
        let input = Owned::f32(&[3, 128], &[0.; 384]);
        let mut output = Owned::zeros_f32(&[3, 128]);
        let mut node = Node::new(NodeId(0), "EyeLike", vec![], vec![]);
        node.attributes.insert("k".into(), Attribute::Int(125));
        EyeLikeFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();

        let mut expected = vec![0.; 384];
        expected[125] = 1.;
        expected[128 + 126] = 1.;
        expected[256 + 127] = 1.;
        assert_eq!(output.to_f32(), expected);
    }

    #[test]
    fn dtype_override_writes_explicit_zero_and_one_for_every_supported_type() {
        let input = Owned::f32(&[2, 3], &[0.; 6]);
        for dtype in [
            DataType::Bool,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::Uint8,
            DataType::Uint16,
            DataType::Uint32,
            DataType::Uint64,
            DataType::Float16,
            DataType::BFloat16,
            DataType::Float32,
            DataType::Float64,
        ] {
            let mut output = Owned::zeros(dtype, &[2, 3]);
            let mut node = Node::new(NodeId(0), "EyeLike", vec![], vec![]);
            node.attributes
                .insert("dtype".into(), Attribute::Int(i64::from(dtype.to_onnx())));
            EyeLikeFactory
                .create(&node, &[])
                .unwrap()
                .execute(&[input.view()], &mut [output.view_mut()])
                .unwrap();

            let one = match dtype {
                DataType::Bool | DataType::Int8 | DataType::Uint8 => vec![1],
                DataType::Int16 | DataType::Uint16 => 1u16.to_le_bytes().to_vec(),
                DataType::Int32 | DataType::Uint32 => 1u32.to_le_bytes().to_vec(),
                DataType::Int64 | DataType::Uint64 => 1u64.to_le_bytes().to_vec(),
                DataType::Float16 => 0x3c00u16.to_le_bytes().to_vec(),
                DataType::BFloat16 => 0x3f80u16.to_le_bytes().to_vec(),
                DataType::Float32 => 1.0f32.to_le_bytes().to_vec(),
                DataType::Float64 => 1.0f64.to_le_bytes().to_vec(),
                _ => unreachable!("unsupported EyeLike dtype {dtype:?}"),
            };
            for (index, value) in output.bytes.chunks_exact(dtype.byte_size()).enumerate() {
                assert_eq!(
                    value,
                    if index == 0 || index == 4 {
                        one.as_slice()
                    } else {
                        &[0; 8][..dtype.byte_size()]
                    },
                    "dtype {dtype:?}, index {index}"
                );
            }
        }
    }

    #[test]
    fn rejects_out_of_range_dtype_attribute_without_truncating() {
        let mut node = Node::new(NodeId(0), "EyeLike", vec![], vec![]);
        node.attributes
            .insert("dtype".into(), Attribute::Int(i64::MAX));
        assert!(EyeLikeFactory.create(&node, &[]).is_err());
    }
}
