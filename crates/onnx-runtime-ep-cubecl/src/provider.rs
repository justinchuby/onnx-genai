//! The [`ExecutionProvider`] implementation shared by both CubeCL backends.
//!
//! One type serves `cubecl-webgpu` and `cubecl-vulkan`; they differ only in the
//! CubeCL runtime type parameter `R` and the [`CubeclBackend`] identity carried
//! alongside it. Nothing here branches on the backend except for reporting.

use std::ffi::c_void;
use std::sync::Arc;

use cubecl::prelude::*;
use onnx_runtime_ep_api::{
    DeviceBuffer, EpConfig, EpError, ExecutionProvider, Fence, HostToDeviceCopier, Kernel,
    KernelMatch, OpRegistry, Result,
};
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};

use crate::backend::CubeclBackend;
use crate::context::CubeclContext;
use crate::kernels::{self, CubeclOpDescriptor};
use crate::memory::HandleTable;

/// Every operator this EP advertises, for the plugin ABIs to publish.
pub fn build_cubecl_registry_descriptors(f16_available: bool) -> Vec<CubeclOpDescriptor> {
    kernels::supported_ops_for(f16_available)
}

/// A CubeCL-backed execution provider.
pub struct CubeclExecutionProvider<R: Runtime> {
    name: String,
    backend: CubeclBackend,
    context: Arc<CubeclContext<R>>,
    registry: OpRegistry,
    live: bool,
}

impl<R: Runtime<Device = cubecl_wgpu::WgpuDevice>> CubeclExecutionProvider<R> {
    /// Open `backend` on `ordinal` and build a ready-to-initialise provider.
    pub fn new(backend: CubeclBackend, ordinal: u32) -> Result<Self> {
        let client = crate::runtime::open_client::<R>(backend, ordinal)?;
        Ok(Self::from_client(backend, ordinal, client))
    }

    /// Build a provider around an already-open client, for tests and for hosts
    /// that share one CubeCL client across several components.
    pub fn from_client(backend: CubeclBackend, ordinal: u32, client: ComputeClient<R>) -> Self {
        let device = DeviceId::new(backend.device_type(), ordinal);
        let f16 = crate::runtime::supports_f16(&client);
        let context = Arc::new(CubeclContext {
            client,
            table: HandleTable::new(),
            device,
            backend,
            f16,
        });
        let registry = kernels::build_registry(context.clone());
        Self {
            name: backend.ep_name().to_string(),
            backend,
            context,
            registry,
            // Live from construction, not from `initialize`. Every device
            // resource this provider needs -- the CubeCL client, the device
            // handle, the f16 probe -- is already open by the time we get here,
            // so there is nothing for `initialize` to set up. The ORT plugin EP
            // ABI has no initialize hook at all and dispatches straight into
            // `get_kernel`; gating on an explicit `initialize` made every node
            // fail under real ORT while the direct-call tests passed.
            live: true,
        }
    }

    pub fn backend(&self) -> CubeclBackend {
        self.backend
    }

    pub fn context(&self) -> &Arc<CubeclContext<R>> {
        &self.context
    }

    /// Whether this provider's device reported usable f16, as probed at open.
    ///
    /// Exposed so a host can report the dtype surface it actually got instead of
    /// discovering it from a rejected node, and so tests can assert the accept
    /// and refuse paths against the same probe the provider used.
    pub fn supports_f16(&self) -> bool {
        self.context.f16
    }

    /// Fail early, with the operation named, when a call arrives after
    /// `shutdown`. The client is drained there, so continuing would dispatch
    /// against buffers that may already have been recycled.
    fn require_live(&self, what: &str) -> Result<()> {
        if self.live {
            return Ok(());
        }
        Err(EpError::KernelFailed(format!(
            "{}: {what} was called after shutdown(); the provider released its device \
             resources and must be reconstructed before further use.",
            self.name
        )))
    }

    /// Reject a buffer that belongs to a different device than this provider.
    fn require_own_buffer(&self, buffer: &DeviceBuffer, what: &str) -> Result<()> {
        if buffer.device() == self.context.device {
            return Ok(());
        }
        Err(EpError::KernelFailed(format!(
            "{}: {what} received a buffer on {:?}, but this provider owns {:?}. Buffers must \
             be used by the EP that allocated them.",
            self.name,
            buffer.device(),
            self.context.device,
        )))
    }
}

impl<R: Runtime<Device = cubecl_wgpu::WgpuDevice>> ExecutionProvider
    for CubeclExecutionProvider<R>
{
    fn name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> DeviceType {
        self.backend.device_type()
    }

    fn device_id(&self) -> DeviceId {
        self.context.device
    }

    fn host_to_device_copier(&self) -> Option<Arc<dyn HostToDeviceCopier>> {
        Some(Arc::new(CubeclHostCopier {
            context: self.context.clone(),
            name: self.name.clone(),
        }))
    }

    fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
        // Idempotent, and not a precondition for anything: the device is opened
        // in the constructor because the ORT plugin path never calls this.
        // Re-arming after a `shutdown` is deliberately not offered -- the
        // CubeCL client is drained there and is not restartable in place.
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        // Outstanding work must drain before the client is dropped, otherwise
        // buffers can be recycled underneath an in-flight dispatch.
        self.sync()?;
        self.live = false;
        Ok(())
    }

    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        _layouts: &[TensorLayout],
    ) -> KernelMatch {
        if !self.registry.supports(&op.op_type, &op.domain, opset) {
            let domain = if op.domain.is_empty() {
                "ai.onnx"
            } else {
                &op.domain
            };
            return KernelMatch::unsupported(format!(
                "{}: no handler for {domain}::{} at opset {opset}",
                self.name, op.op_type
            ));
        }
        // f16 is accepted only when this adapter actually reported f16 buffer
        // and arithmetic support. The probe result differs between machines
        // running the same binary, so the refusal names the device feature
        // rather than implying the EP lacks the kernel.
        if let Some(bad) = input_dtypes.iter().find(|dtype| {
            **dtype != DataType::Float32 && !(**dtype == DataType::Float16 && self.context.f16)
        }) {
            let detail = if *bad == DataType::Float16 {
                "this adapter does not report f16 buffer and arithmetic support (WebGPU \
                 'shader-f16')"
            } else {
                "the cubecl backends implement f32 and f16 only"
            };
            return KernelMatch::unsupported(format!(
                "{}: {} input dtype {bad:?} is unsupported; {detail}",
                self.name, op.op_type
            ));
        }
        KernelMatch::Supported {
            cost: dispatch_cost(shapes, input_dtypes),
            required_input_layouts: None,
            output_layouts: vec![TensorLayout::default()],
        }
    }

    fn get_kernel(&self, op: &Node, shapes: &[Vec<usize>], opset: u64) -> Result<Box<dyn Kernel>> {
        self.require_live("get_kernel")?;
        let factory = self
            .registry
            .lookup(&op.op_type, &op.domain, opset)
            .ok_or_else(|| EpError::NoEpForOp {
                domain: if op.domain.is_empty() {
                    "ai.onnx".to_string()
                } else {
                    op.domain.clone()
                },
                op_type: op.op_type.clone(),
                opset,
            })?;
        factory.create(op, shapes)
    }

    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer> {
        self.require_live("allocate")?;
        // A zero-byte allocation still has to yield a distinct, non-null,
        // freeable address, because callers store and later deallocate it.
        let handle = self.context.client.empty(size.max(1));
        let ptr = self.context.table.insert(handle, size, alignment);
        // SAFETY: `ptr` is a synthetic address owned by this provider's table,
        // is non-null, is never dereferenced on the host, and stays valid until
        // the matching `deallocate`.
        Ok(unsafe { DeviceBuffer::from_raw_parts(ptr, self.context.device, size, alignment) })
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()> {
        if buffer.is_borrowed() {
            return Ok(());
        }
        self.require_own_buffer(&buffer, "deallocate")?;
        self.context.table.remove(buffer.as_ptr().cast_mut().cast())
    }

    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()> {
        self.require_live("copy")?;
        self.require_own_buffer(src, "copy source")?;
        self.require_own_buffer(dst, "copy destination")?;
        if size == 0 {
            return Ok(());
        }
        if size > src.len() || size > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "{}: copy of {size} bytes exceeds source ({}) or destination ({}) allocation",
                self.name,
                src.len(),
                dst.len()
            )));
        }
        // CubeCL exposes no device-to-device copy, so this stages through the
        // host. That is a real cost, and it is deliberate: the alternative is a
        // copy kernel, which would still be a full dispatch and would silently
        // reinterpret the bytes as a numeric element type. Device-to-device
        // copies are rare in the graphs these kernels currently claim; if that
        // changes, replace this with a u32 copy kernel rather than hiding the
        // cost.
        let resolved = self.context.table.resolve(src.as_ptr(), size)?;
        let bytes = self
            .context
            .client
            .read_one(resolved.handle)
            .map_err(|error| {
                EpError::KernelFailed(format!("{}: device read failed: {error:?}", self.name))
            })?;
        // SAFETY: `dst` is a live allocation of this provider holding at least
        // `size` bytes, as checked above.
        unsafe { self.copy_bytes_to_device(&bytes[..size], dst.as_mut_ptr()) }
    }

    fn copy_async(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<Fence> {
        // The staged copy above is synchronous by construction, so the fence it
        // returns is already complete. Reporting a pending fence would promise
        // an overlap that does not exist.
        self.copy(src, dst, size)?;
        Ok(Fence::signalled())
    }

    fn copy_from_host(&self, src: &[u8], dst: &mut DeviceBuffer) -> Result<()> {
        self.require_live("copy_from_host")?;
        self.require_own_buffer(dst, "copy_from_host destination")?;
        if src.len() > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "{}: host upload of {} bytes exceeds the {}-byte destination allocation",
                self.name,
                src.len(),
                dst.len()
            )));
        }
        // SAFETY: bounds and ownership are checked above.
        unsafe { self.copy_bytes_to_device(src, dst.as_mut_ptr()) }
    }

    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> Result<()> {
        self.require_live("copy_to_host")?;
        self.require_own_buffer(src, "copy_to_host source")?;
        if dst.is_empty() {
            return Ok(());
        }
        if dst.len() > src.len() {
            return Err(EpError::KernelFailed(format!(
                "{}: host download of {} bytes exceeds the {}-byte source allocation",
                self.name,
                dst.len(),
                src.len()
            )));
        }
        let resolved = self.context.table.resolve(src.as_ptr(), dst.len())?;
        let bytes = self
            .context
            .client
            .read_one(resolved.handle)
            .map_err(|error| {
                EpError::KernelFailed(format!("{}: device read failed: {error:?}", self.name))
            })?;
        dst.copy_from_slice(&bytes[..dst.len()]);
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        cubecl::future::block_on(self.context.client.sync()).map_err(|error| {
            EpError::KernelFailed(format!("{}: device sync failed: {error:?}", self.name))
        })
    }
}

impl<R: Runtime<Device = cubecl_wgpu::WgpuDevice>> CubeclExecutionProvider<R> {
    /// Write host bytes into the allocation that `dst` addresses.
    ///
    /// # Safety
    ///
    /// `dst` must be a live address from this provider's table with at least
    /// `src.len()` bytes available from it.
    unsafe fn copy_bytes_to_device(&self, src: &[u8], dst: *mut c_void) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let resolved = self.context.table.resolve(dst, src.len())?;
        self.context.client.write(
            &resolved.handle,
            cubecl::bytes::Bytes::from_elems(src.to_vec()),
        );
        Ok(())
    }
}

/// The staging path ORT and nxrt use to push host tensors onto the device.
struct CubeclHostCopier<R: Runtime> {
    context: Arc<CubeclContext<R>>,
    name: String,
}

impl<R: Runtime> HostToDeviceCopier for CubeclHostCopier<R> {
    unsafe fn copy_host_to_device(&self, src: &[u8], dst: *mut c_void) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let resolved = self
            .context
            .table
            .resolve(dst, src.len())
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "{}: host-to-device staging failed: {error}",
                    self.name
                ))
            })?;
        self.context.client.write(
            &resolved.handle,
            cubecl::bytes::Bytes::from_elems(src.to_vec()),
        );
        Ok(())
    }
}

/// Fixed GPU dispatch latency in microseconds.
///
/// A WebGPU/Vulkan compute dispatch costs tens of microseconds of command
/// encoding and submission before a single lane runs, which is why a tiny node
/// is cheaper on the CPU even though the GPU would finish the arithmetic first.
/// The value is a conservative order-of-magnitude figure, not a measurement: it
/// exists so the planner has a real reason to leave small nodes alone, and it is
/// the number to replace once dispatch latency is actually benchmarked here.
const DISPATCH_LATENCY_US: f64 = 30.0;

/// Cost of running one node on this provider.
///
/// Only the launch term is modelled. Throughput is deliberately left at zero
/// rather than filled with an invented bandwidth constant: a wrong throughput
/// model would silently mis-place large nodes, whereas an absent one just makes
/// the planner size-agnostic above the launch threshold. `bytes_moved` is still
/// reported so a roofline model can be layered on without changing callers.
fn dispatch_cost(shapes: &[Shape], dtypes: &[DataType]) -> onnx_runtime_ep_api::Cost {
    // A symbolic dimension contributes nothing rather than a guessed extent:
    // under-reporting traffic is recoverable, inventing an extent is not.
    let elements: u64 = shapes
        .iter()
        .map(|shape| {
            shape
                .iter()
                .map(|dim| dim.as_static().unwrap_or(0) as u64)
                .product::<u64>()
        })
        .sum();
    // f16 halves the traffic for the same element count, which is the whole
    // reason to run it, so the width comes from the node's dtype rather than a
    // hardcoded 4.
    let width = match dtypes.first() {
        Some(DataType::Float16) => 2,
        _ => 4,
    };
    let mut cost = onnx_runtime_ep_api::Cost::ZERO;
    cost.launch_us = DISPATCH_LATENCY_US;
    cost.bytes_moved = elements.saturating_mul(width);
    cost
}

#[cfg(test)]
mod tests {
    use super::{DISPATCH_LATENCY_US, dispatch_cost};
    use onnx_runtime_ir::{DataType, Dim, Shape};

    fn static_shape(dims: &[usize]) -> Shape {
        Shape::from(dims.iter().copied().map(Dim::Static).collect::<Vec<_>>())
    }

    #[test]
    fn a_node_always_pays_the_dispatch_latency() {
        let cost = dispatch_cost(&[static_shape(&[1])], &[DataType::Float32]);
        assert_eq!(cost.launch_us, DISPATCH_LATENCY_US);
        assert_eq!(cost.bytes_moved, 4);
    }

    #[test]
    fn traffic_sums_every_operand() {
        let cost = dispatch_cost(
            &[static_shape(&[2, 3]), static_shape(&[3, 4])],
            &[DataType::Float32],
        );
        assert_eq!(cost.bytes_moved, (2 * 3 + 3 * 4) * 4);
    }

    #[test]
    fn f16_moves_half_the_bytes_of_f32() {
        let shape = [static_shape(&[128])];
        let f32_cost = dispatch_cost(&shape, &[DataType::Float32]);
        let f16_cost = dispatch_cost(&shape, &[DataType::Float16]);
        assert_eq!(f16_cost.bytes_moved * 2, f32_cost.bytes_moved);
    }

    #[test]
    fn a_symbolic_extent_is_not_guessed() {
        let shape = Shape::from(vec![
            Dim::Symbolic(onnx_runtime_ir::SymbolId(0)),
            Dim::Static(8),
        ]);
        assert_eq!(dispatch_cost(&[shape], &[DataType::Float32]).bytes_moved, 0);
    }
}
