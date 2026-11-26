mod affine;
mod native;
mod packed;
use affine::Affine;
use native::Native;
use onnx_runtime_ep_api::packed_routed::fail;
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorView};
use onnx_runtime_ir::dequantize_packed::Metadata;
use onnx_runtime_ir::{DataType, Node, packed_routed::Op};
use packed::Packed;

pub(super) struct Factory;

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Native(
            Op::from_node(node).map_err(EpError::KernelFailed)?,
        )))
    }
}

struct RouteSource<'view, 'storage> {
    op: Op,
    view: &'view TensorView<'storage>,
    count: usize,
}
impl<'view, 'storage> RouteSource<'view, 'storage> {
    fn new(op: Op, view: &'view TensorView<'storage>) -> Result<Self> {
        let count = match view.dtype {
            DataType::Int32 => {
                crate::strided::validate_host_storage::<i32>(op.name(), "group_indices", view)?
            }
            DataType::Int64 => {
                crate::strided::validate_host_storage::<i64>(op.name(), "group_indices", view)?
            }
            dtype => {
                return Err(fail(
                    op,
                    format!(
                        "group_indices must be INT32/INT64, got {dtype:?}; bind integer routes"
                    ),
                ));
            }
        };
        std::alloc::Layout::array::<usize>(count).map_err(|_| {
            fail(
                op,
                "group_indices exceed addressable route storage; reduce route dimensions",
            )
        })?;
        Ok(Self { op, view, count })
    }
    fn copy_into(&self, groups: usize, out: &mut [usize]) -> Result<()> {
        let op = self.op;
        if out.len() != self.count {
            return Err(fail(
                op,
                format!(
                    "group_indices workspace has {} elements, expected {}; bind the complete route workspace",
                    out.len(),
                    self.count
                ),
            ));
        }
        let mut convert = |i, value| {
            out[i] = usize::try_from(value)
                .ok()
                .filter(|&g| g < groups)
                .ok_or_else(|| {
                    fail(
                        op,
                        format!("group_indices[{i}] is {value}; select a group in [0, {groups})"),
                    )
                })?;
            Ok(())
        };
        if self.view.dtype == DataType::Int32 {
            crate::dtype::try_visit_storage::<i32>(op.name(), self.view, |i, v| {
                convert(i, i64::from(v))
            })
        } else {
            crate::dtype::try_visit_storage::<i64>(op.name(), self.view, convert)
        }
    }
}
#[cfg(test)]
fn route_indices(op: Op, idx: &TensorView, groups: usize) -> Result<Vec<usize>> {
    let source = RouteSource::new(op, idx)?;
    let mut routes = vec![0usize; source.count];
    source.copy_into(groups, &mut routes)?;
    Ok(routes)
}

pub(super) struct AffineFactory;

impl KernelFactory for AffineFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(
            onnx_runtime_ep_api::packed_routed::AffineGrouped::from_node(node).map(Affine)?,
        ))
    }
}

pub(super) struct PackedFactory;

impl KernelFactory for PackedFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Packed {
            metadata: Metadata::from_node(node).map_err(|detail| {
                EpError::KernelFailed(format!("PackedGroupedMatMul: {detail}"))
            })?,
        }))
    }
}

#[cfg(test)]
mod packed_host_tests {
    use super::*;
    use crate::kernels::testutil::{Owned, PreparedTestKernel};
    use onnx_runtime_ep_api::DevicePtr;
    use onnx_runtime_ir::{Attribute, DeviceId, NodeId};

    fn kernel() -> Packed {
        let mut node = Node::new(NodeId(0), "PackedGroupedMatMul", vec![], vec![]);
        node.attributes
            .insert("format".into(), Attribute::String("gguf:q4_0".into()));
        node.attributes
            .insert("shape".into(), Attribute::Ints(vec![1, 1, 32]));
        Packed {
            metadata: Metadata::from_node(&node).unwrap(),
        }
    }
    #[test]
    fn rejects_unreadable_packed_storage_before_any_route_or_activation_read() {
        let k = kernel();
        let x = Owned::f32(&[1, 32], &[1.; 32]);
        let packed = Owned::u8(&[18], &[0; 18]);
        let indices = Owned::i64(&[1, 1], &[0]);
        let mut inputs = [x.view(), packed.view(), indices.view()];
        // These Host descriptors are valid metadata but deliberately unreadable;
        // refusing the bank must precede any activation or index dereference.
        inputs[0].data = DevicePtr(4usize as *const std::ffi::c_void);
        inputs[2].data = DevicePtr(8usize as *const std::ffi::c_void);
        let mut output = Owned::f32(&[1, 1, 1], &[123.]);
        for kind in 0..3 {
            inputs[1] = packed.view();
            match kind {
                0 => {
                    inputs[1].device = DeviceId::cuda(0);
                    inputs[1].data = DevicePtr(std::ptr::dangling());
                }
                1 => inputs[1].data = DevicePtr((usize::MAX - 8) as *const std::ffi::c_void),
                _ => {
                    inputs[1].data = DevicePtr((usize::MAX - 3) as *const std::ffi::c_void);
                    inputs[1].byte_offset = 8;
                }
            }
            let message = k
                .execute(&inputs, &mut [output.view_mut()])
                .unwrap_err()
                .to_string();
            assert!(
                message.contains("PackedGroupedMatMul") && message.contains("packed weights"),
                "{message}"
            );
            assert!(
                message.contains(if kind == 0 {
                    "Host-accessible"
                } else {
                    "unaddressable span"
                }),
                "{message}"
            );
            assert_eq!(output.to_f32(), [123.]);
        }
    }
    #[test]
    fn packed_host_storage_supports_byte_offsets_and_empty_execution_remains_read_free() {
        let k = kernel();
        let x = Owned::f32(&[1, 32], &[1.; 32]);
        let indices = Owned::i64(&[1, 1], &[0]);
        // Q4_0: scale one, all encoded values nine => every weight is one.
        let mut bytes = vec![0xcc, 0, 0x3c];
        bytes.extend_from_slice(&[0x99; 16]);
        let packed = Owned::u8(&[bytes.len()], &bytes);
        let view = TensorView::new(
            packed.view().data,
            DataType::Uint8,
            &[18],
            &[1],
            DeviceId::cpu(),
        )
        .with_byte_offset(1);
        let mut out = Owned::zeros_f32(&[1, 1, 1]);
        k.execute_prepared(&[x.view(), view, indices.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [32.]);
        let x = Owned::f32(&[0, 32], &[]);
        let indices = Owned::i64(&[0, 1], &[]);
        let mut out = Owned::zeros_f32(&[0, 1, 1]);
        let view = TensorView::new(
            DevicePtr(std::ptr::dangling()),
            DataType::Uint8,
            &[18],
            &[1],
            DeviceId::cuda(0),
        );
        k.execute(&[x.view(), view, indices.view()], &mut [out.view_mut()])
            .unwrap();
    }
}
