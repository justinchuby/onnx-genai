mod affine;
use crate::dtype::{to_dense, to_dense_f32_widen, write_dense, write_dense_f32_narrow};
use affine::Affine;
use onnx_runtime_ep_api::packed_routed::{
    GroupLayout, fail, group_layout, packed_group_layout, validate,
};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::dequantize_packed::Metadata;
use onnx_runtime_ir::{DataType, Node, packed_routed::Op};

pub(super) struct Factory;

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Native(
            Op::from_node(node).map_err(EpError::KernelFailed)?,
        )))
    }
}

struct Native(Op);

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
fn route_indices(op: Op, idx: &TensorView, groups: usize) -> Result<Vec<usize>> {
    let source = RouteSource::new(op, idx)?;
    let mut routes = vec![0usize; source.count];
    source.copy_into(groups, &mut routes)?;
    Ok(routes)
}

fn grouped<F>(
    l: GroupLayout,
    x: &[F],
    weight: impl Fn(usize) -> F,
    groups: &[usize],
    b: Option<&[F]>,
    round: impl Fn(F) -> F,
) -> Vec<F>
where
    F: Copy + Default + std::ops::Add<Output = F> + std::ops::Mul<Output = F>,
{
    let mut y = vec![F::default(); l.lanes];
    for (route, row) in y.chunks_exact_mut(l.n).enumerate() {
        let g = groups[route];
        let token = route / l.selections;
        for (n, out) in row.iter_mut().enumerate() {
            let mut sum = F::default();
            for k in 0..l.k {
                let i = if l.transposed {
                    n * l.k + k
                } else {
                    k * l.n + n
                };
                sum = sum + x[token * l.k + k] * weight(g * l.k * l.n + i);
            }
            // The ONNX function applies Add after the MatMul result is stored.
            *out = if let Some(b) = b {
                round(sum) + b[g * l.n + n]
            } else {
                sum
            };
        }
    }
    y
}

impl Kernel for Native {
    fn execute(&self, x: &[TensorView], y: &mut [TensorMut]) -> Result<()> {
        let op = self.0;
        validate(op, x, y)?;
        let l = group_layout(op, x, &y[0])?;
        let groups = route_indices(op, &x[2], l.groups)?;
        if l.lanes == 0 {
            return Ok(());
        }
        let b = l.bias.then(|| &x[3]);
        if x[0].dtype == DataType::Float64 {
            let bias = b.map(to_dense::<f64>).transpose()?;
            let weights = to_dense::<f64>(&x[1])?;
            let data = grouped(
                l,
                &to_dense::<f64>(&x[0])?,
                |i| weights[i],
                &groups,
                bias.as_deref(),
                |v| v,
            );
            write_dense(&mut y[0], &data)
        } else {
            let widen = |t| to_dense_f32_widen(op.name(), t);
            let bias = b.map(widen).transpose()?;
            let weights = widen(&x[1])?;
            let data = grouped(
                l,
                &widen(&x[0])?,
                |i| weights[i],
                &groups,
                bias.as_deref(),
                |v| match x[0].dtype {
                    DataType::Float16 => half::f16::from_f32(v).to_f32(),
                    DataType::BFloat16 => half::bf16::from_f32(v).to_f32(),
                    _ => v,
                },
            );
            write_dense_f32_narrow(op.name(), &mut y[0], &data)
        }
    }
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

struct Packed {
    metadata: Metadata,
}

impl Kernel for Packed {
    fn execute(&self, x: &[TensorView], y: &mut [TensorMut]) -> Result<()> {
        let l = packed_group_layout(&self.metadata, x, y)?;
        if l.lanes == 0 {
            return Ok(());
        }
        if x.iter().any(|tensor| !tensor.is_contiguous()) || !y[0].is_contiguous() {
            return Err(EpError::KernelFailed(
                "PackedGroupedMatMul: inputs and output must use contiguous storage".into(),
            ));
        }
        crate::strided::validate_host_storage::<u8>(
            "PackedGroupedMatMul",
            "packed weights",
            &x[1],
        )?;
        let routes = route_indices(Op::GroupedMatMul { transposed: true }, &x[2], l.groups)?;
        let activation = super::to_dense_f32(&x[0])?;
        // SAFETY: packed_group_layout validated the one-dimensional UINT8
        // extent, and the Host span proof validated representable addresses.
        // The owning EP guarantees live backing for this contiguous byte view.
        let packed = unsafe {
            std::slice::from_raw_parts(x[1].data_ptr::<u8>(), self.metadata.packed_bytes())
        };
        let (width, bytes) = self.metadata.format().block();
        let blocks = l.k / width;
        let decode = super::dequantize_packed::decoder(self.metadata.format());
        let mut values = [0.0f32; 256];
        let mut output = vec![0.0f32; l.lanes];
        for (route, row) in output.chunks_exact_mut(l.n).enumerate() {
            let expert = routes[route];
            let token = route / l.selections;
            for (column, out) in row.iter_mut().enumerate() {
                let packed_row = (expert * l.n + column) * blocks * bytes;
                let mut sum = 0.0f32;
                for block in 0..blocks {
                    let source = packed_row + block * bytes;
                    decode(&packed[source..source + bytes], &mut values[..width]);
                    let activation =
                        &activation[token * l.k + block * width..token * l.k + (block + 1) * width];
                    for (&a, &weight) in activation.iter().zip(&values[..width]) {
                        sum += a * weight;
                    }
                }
                *out = sum;
            }
        }
        write_dense(&mut y[0], &output)
    }
}

#[cfg(test)]
mod packed_host_tests {
    use super::*;
    use crate::kernels::testutil::Owned;
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
        k.execute(&[x.view(), view, indices.view()], &mut [out.view_mut()])
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
