//! The [`CpuExecutionProvider`]: a host execution provider backed by pure-Rust
//! reference kernels (`docs/ORT2.md` §4.4).
//!
//! # Memory & safety invariants (ep-api safety review)
//!
//! This EP is the allocator/deallocator for every buffer it hands out. It
//! upholds the five must-hold invariants from the ep-api safety review:
//!
//! 1. **View bounds** — kernels only read/write within the extent a
//!    [`TensorView`](onnx_runtime_ep_api::TensorView)'s shape/strides/offset
//!    describe; the caller that owns the backing buffer verifies storage bounds
//!    via [`crate::strided::view_in_bounds`] before dispatch (a `TensorView`
//!    cannot see its allocation size).
//! 2. **Single-free** — every [`allocate`](CpuExecutionProvider::allocate)
//!    pairs with exactly one [`deallocate`](CpuExecutionProvider::deallocate);
//!    `DeviceBuffer` has no `Drop`, so a dropped handle leaks but never
//!    double-frees.
//! 3. **No cross-EP free** — `deallocate`/`copy` assert the buffer's device
//!    matches this EP's device.
//! 4. **`copy` size** — `copy`/`copy_async` reject `size` larger than either
//!    endpoint.
//! 5. **Thread-affine allocators** — N/A: host `malloc` addresses are portable,
//!    so `DeviceBuffer` is soundly `Send`/`Sync` (documented in ep-api).

use std::alloc::{Layout, alloc, dealloc};
use std::ffi::c_void;

use onnx_runtime_ep_api::{
    Cost, DeviceBuffer, EpConfig, EpError, ExecutionProvider, Fence, Kernel, KernelMatch,
    OpRegistry, Result,
};
use onnx_runtime_ir::{DeviceId, DeviceType, Node, Shape, TensorLayout};

use crate::kernels::build_cpu_registry;
use crate::optimizer::cpu_optimization_passes;

/// CPU execution provider. Always available; the fallback EP for any op.
///
/// Holds the CPU op → kernel-factory registry, built once at construction. The
/// registry is also exposed to the session (Track D) so placement and kernel
/// instantiation share one source of truth.
pub struct CpuExecutionProvider {
    device: DeviceId,
    initialized: bool,
    registry: OpRegistry,
}

impl std::fmt::Debug for CpuExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuExecutionProvider")
            .field("device", &self.device)
            .field("initialized", &self.initialized)
            .field("registered_ops", &self.registry.len())
            .finish()
    }
}

impl Default for CpuExecutionProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuExecutionProvider {
    /// Construct a CPU EP bound to `CPU:0` with all Phase-1 kernels registered.
    pub fn new() -> Self {
        Self {
            device: DeviceId::cpu(),
            initialized: false,
            registry: build_cpu_registry(),
        }
    }

    /// Borrow the CPU op registry (shared with the session layer).
    pub fn registry(&self) -> &OpRegistry {
        &self.registry
    }
}

impl ExecutionProvider for CpuExecutionProvider {
    fn name(&self) -> &str {
        "cpu_ep"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn device_id(&self) -> DeviceId {
        self.device
    }

    fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
        // Pure-Rust kernels need no device resources or external libraries.
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }

    fn supports_op(&self, op: &Node, shapes: &[Shape], _layouts: &[TensorLayout]) -> KernelMatch {
        // Model-agnostic: keyed on (op_type, domain) via the registry — the
        // single source of truth for "is this an op/domain we support". This
        // accepts standard default-domain (`""`/`ai.onnx`) ops and any contrib
        // ops (e.g. fused `com.microsoft` ops) the registry knows, without a
        // hardcoded op/domain whitelist.
        if !self.registry.supports(&op.op_type, &op.domain) {
            return KernelMatch::Unsupported;
        }
        // The reference kernels produce contiguous row-major outputs and accept
        // strided inputs, so no input layout is required.
        let output_layouts = vec![TensorLayout::contiguous(); op.outputs.len()];
        // Rough cost estimate from the input element counts; the real cost model
        // (Phase 2) refines this. Kept monotonic in problem size so placement
        // still prefers a smaller CPU op over a larger one.
        let elems: u64 = shapes
            .iter()
            .map(|s| {
                s.iter()
                    .map(|d| d.as_static().unwrap_or(1) as u64)
                    .product::<u64>()
            })
            .sum();
        let cost = Cost::new(elems as f64, elems as f64, 0.0)
            .with_launch_us(0.1)
            .with_bytes_moved(elems.saturating_mul(4));
        KernelMatch::Supported {
            cost,
            required_input_layouts: None,
            output_layouts,
        }
    }

    fn get_kernel(&self, op: &Node, shapes: &[Vec<usize>], opset: u64) -> Result<Box<dyn Kernel>> {
        // Select the highest registered `since_version` that is <= the graph's
        // effective opset for this op's domain. Ops with a single registration
        // (since_version 1) always match; opset-specialized ops (e.g. Softmax,
        // registered at both 1 and 13) get the version-correct kernel.
        let factory = self
            .registry
            .lookup(&op.op_type, &op.domain, opset)
            .ok_or_else(|| EpError::NoEpForOp {
                op_type: op.op_type.clone(),
            })?;
        factory.create(op, shapes)
    }

    fn custom_passes(&self) -> Vec<Box<dyn onnx_runtime_optimizer::OptimizationPass>> {
        cpu_optimization_passes()
    }

    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(EpError::AlignmentError);
        }
        // std::alloc rejects zero-sized layouts; allocate at least one byte so
        // the base pointer is non-null, but record the requested `size`.
        let alloc_size = size.max(1);
        let layout =
            Layout::from_size_align(alloc_size, alignment).map_err(|_| EpError::AlignmentError)?;
        // SAFETY: `layout` has non-zero size (bumped to >= 1) and a valid
        // power-of-two alignment. We check the returned pointer for null below
        // and treat OOM as an error rather than dereferencing.
        let ptr = unsafe { alloc(layout) } as *mut c_void;
        if ptr.is_null() {
            return Err(EpError::OutOfMemory {
                requested: size,
                available: 0,
            });
        }
        // SAFETY: `ptr` is a fresh, unique, non-null allocation of `alloc_size`
        // (>= `size`) bytes aligned to `alignment`, owned by this EP and freed
        // exactly once in `deallocate` (invariant #2). No other handle aliases
        // it. We record the caller-requested `size`.
        Ok(unsafe { DeviceBuffer::from_raw_parts(ptr, self.device, size, alignment) })
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()> {
        // Invariant #3: never free a buffer that belongs to another EP/device.
        assert_eq!(
            buffer.device(),
            self.device,
            "cpu_ep: refusing to deallocate a buffer from device {:?}",
            buffer.device()
        );
        // Borrowed buffers alias foreign memory (e.g. an mmap'd weight file)
        // that this EP never allocated — freeing it would be undefined
        // behavior. The real owner outlives the buffer and frees it itself.
        if buffer.is_borrowed() {
            return Ok(());
        }
        let size = buffer.len();
        let align = buffer.alignment();
        let ptr = buffer.into_raw() as *mut u8;
        // Reconstruct the exact layout used in `allocate` (same `max(1)` bump).
        let layout = Layout::from_size_align(size.max(1), align)
            .expect("cpu_ep: layout was valid at allocation time");
        // SAFETY: `ptr` came from this EP's `alloc` with `layout` (invariant #2),
        // `into_raw` consumed the owning handle so no alias remains, and this is
        // the single free of that allocation.
        unsafe { dealloc(ptr, layout) };
        Ok(())
    }

    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()> {
        // Invariant #3: both endpoints must belong to this EP.
        assert_eq!(
            src.device(),
            self.device,
            "cpu_ep::copy: foreign src buffer"
        );
        assert_eq!(
            dst.device(),
            self.device,
            "cpu_ep::copy: foreign dst buffer"
        );
        // Invariant #4: never read/write past either endpoint.
        if size > src.len() || size > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "cpu_ep::copy: size {size} exceeds src {} or dst {}",
                src.len(),
                dst.len()
            )));
        }
        if size == 0 {
            return Ok(());
        }
        let src_ptr = src.as_ptr() as *const u8;
        let dst_ptr = dst.as_mut_ptr() as *mut u8;
        // SAFETY: both pointers are valid host allocations of at least `size`
        // bytes (checked above). They name distinct `DeviceBuffer`s — `dst` is
        // borrowed `&mut`, so it cannot alias `src` — hence non-overlapping.
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
        }
        Ok(())
    }

    fn copy_async(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<Fence> {
        // Host copies are synchronous; perform it and return a signaled fence.
        self.copy(src, dst, size)?;
        Ok(Fence::default())
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_lifecycle() {
        let mut ep = CpuExecutionProvider::new();
        assert_eq!(ep.name(), "cpu_ep");
        assert_eq!(ep.device_type(), DeviceType::Cpu);
        assert_eq!(ep.device_id(), DeviceId::cpu());
        ep.initialize(&EpConfig::default()).unwrap();
        assert!(ep.initialized);
        ep.shutdown().unwrap();
        assert!(!ep.initialized);
    }

    #[test]
    fn allocate_deallocate_single_free_and_aligned() {
        let ep = CpuExecutionProvider::new();
        let buf = ep.allocate(256, 64).unwrap();
        assert_eq!(buf.len(), 256);
        assert_eq!(buf.alignment(), 64);
        assert_eq!(buf.device(), DeviceId::cpu());
        // 64-byte aligned base.
        assert_eq!(buf.as_ptr() as usize % 64, 0);
        // Single free — a double free would trip ASan/Miri.
        ep.deallocate(buf).unwrap();
    }

    #[test]
    fn allocate_zero_size_is_nonnull() {
        let ep = CpuExecutionProvider::new();
        let buf = ep.allocate(0, 16).unwrap();
        assert_eq!(buf.len(), 0);
        assert!(!buf.as_ptr().is_null());
        ep.deallocate(buf).unwrap();
    }

    /// `deallocate` must be a no-op free for a borrowed buffer: it aliases
    /// memory the EP never allocated (here a `Vec`), so freeing it would be UB.
    /// After deallocation the backing must remain fully valid.
    #[test]
    fn deallocate_borrowed_buffer_is_a_noop_free() {
        let ep = CpuExecutionProvider::new();
        let mut backing = vec![42u8; 128];
        let ptr = backing.as_mut_ptr() as *mut c_void;
        // SAFETY: `ptr`/`len` name `backing`'s live allocation; `backing`
        // outlives the buffer, we never write through it, and `deallocate` must
        // skip the free because the buffer is borrowed.
        let buf = unsafe { DeviceBuffer::from_borrowed_parts(ptr, ep.device_id(), 128, 1) };
        assert!(buf.is_borrowed());
        ep.deallocate(buf).unwrap();
        // No free happened: the Vec is still valid and unmodified.
        assert!(backing.iter().all(|&b| b == 42));
        backing[0] = 1; // proves the allocation is live (would be UAF if freed)
        assert_eq!(backing[0], 1);
    }

    #[test]
    fn allocate_rejects_bad_alignment() {
        let ep = CpuExecutionProvider::new();
        assert!(matches!(ep.allocate(16, 0), Err(EpError::AlignmentError)));
        assert!(matches!(
            ep.allocate(16, 24), // not a power of two
            Err(EpError::AlignmentError)
        ));
    }

    #[test]
    fn copy_moves_bytes_and_checks_size() {
        let ep = CpuExecutionProvider::new();
        let mut src = ep.allocate(16, 16).unwrap();
        let mut dst = ep.allocate(16, 16).unwrap();
        // Write a pattern into src.
        // SAFETY: host buffer of 16 bytes, unique &mut access.
        unsafe {
            let p = src.as_mut_ptr() as *mut u8;
            for i in 0..16u8 {
                *p.add(i as usize) = i;
            }
        }
        ep.copy(&src, &mut dst, 16).unwrap();
        // SAFETY: dst is a valid 16-byte host buffer.
        unsafe {
            let p = dst.as_ptr() as *const u8;
            for i in 0..16u8 {
                assert_eq!(*p.add(i as usize), i);
            }
        }
        // Oversized copy is rejected.
        assert!(ep.copy(&src, &mut dst, 32).is_err());
        ep.deallocate(src).unwrap();
        ep.deallocate(dst).unwrap();
    }

    #[test]
    #[should_panic(expected = "device")]
    fn deallocate_rejects_cross_device_buffer() {
        let ep = CpuExecutionProvider::new();
        // Fabricate a buffer tagged with a CUDA device to trip invariant #3.
        let boxed = vec![0u8; 8].into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut c_void;
        // SAFETY: valid 8-byte host allocation; we only use it to exercise the
        // device assert. It leaks on the panic path, which is fine in a test.
        let foreign = unsafe { DeviceBuffer::from_raw_parts(ptr, DeviceId::cuda(0), 8, 8) };
        let _ = ep.deallocate(foreign); // must panic before freeing
    }

    #[test]
    fn get_kernel_dispatches_phase1_ops() {
        let ep = CpuExecutionProvider::new();
        for (i, op) in crate::kernels::PHASE1_OPS.iter().enumerate() {
            let node = Node::new(onnx_runtime_ir::NodeId(i as u32), *op, vec![], vec![]);
            assert!(ep.get_kernel(&node, &[], 17).is_ok(), "no kernel for {op}");
        }
        let bad = Node::new(onnx_runtime_ir::NodeId(99), "Conv", vec![], vec![]);
        assert!(ep.get_kernel(&bad, &[], 17).is_err());
    }

    #[test]
    fn supports_op_reports_phase1_only() {
        let ep = CpuExecutionProvider::new();
        let mm = Node::new(onnx_runtime_ir::NodeId(0), "MatMul", vec![], vec![]);
        assert!(ep.supports_op(&mm, &[], &[]).is_supported());
        let conv = Node::new(onnx_runtime_ir::NodeId(1), "Conv", vec![], vec![]);
        assert!(!ep.supports_op(&conv, &[], &[]).is_supported());
    }

    #[test]
    fn supports_fused_contrib_domain_layernorm() {
        let ep = CpuExecutionProvider::new();
        // The optimizer emits fused LayerNormalization in `com.microsoft`; the
        // EP must accept it (bound to the same kernel as the standard op).
        let mut fused = Node::new(
            onnx_runtime_ir::NodeId(0),
            "LayerNormalization",
            vec![],
            vec![],
        );
        fused.domain = "com.microsoft".to_string();
        assert!(ep.supports_op(&fused, &[], &[]).is_supported());
        assert!(ep.get_kernel(&fused, &[], 1).is_ok());

        // The fused `FusedMatMulBias` (MatMul+Add) now has a contrib-domain
        // kernel, so it is supported and instantiable.
        let mut fmb = Node::new(
            onnx_runtime_ir::NodeId(1),
            "FusedMatMulBias",
            vec![],
            vec![],
        );
        fmb.domain = "com.microsoft".to_string();
        assert!(ep.supports_op(&fmb, &[], &[]).is_supported());
        assert!(ep.get_kernel(&fmb, &[], 1).is_ok());

        // The fused `FusedGemm` (MatMul+Add+Relu) now has a contrib-domain
        // kernel too, so it is supported and instantiable.
        let mut fg = Node::new(onnx_runtime_ir::NodeId(2), "FusedGemm", vec![], vec![]);
        fg.domain = "com.microsoft".to_string();
        assert!(ep.supports_op(&fg, &[], &[]).is_supported());
        assert!(ep.get_kernel(&fg, &[], 1).is_ok());

        // The fused `FusedAttention` (SDPA core) is supported in the contrib
        // domain; its factory needs the synthesized `scale` attribute to
        // instantiate.
        let mut fa = Node::new(onnx_runtime_ir::NodeId(4), "FusedAttention", vec![], vec![]);
        fa.domain = "com.microsoft".to_string();
        assert!(ep.supports_op(&fa, &[], &[]).is_supported());
        fa.attributes
            .insert("scale".to_string(), onnx_runtime_ir::Attribute::Float(0.5));
        assert!(ep.get_kernel(&fa, &[], 1).is_ok());

        // A contrib op with no kernel is still rejected — support is keyed on
        // (op_type, domain).
        let mut unknown = Node::new(
            onnx_runtime_ir::NodeId(3),
            "NotARealFusedOp",
            vec![],
            vec![],
        );
        unknown.domain = "com.microsoft".to_string();
        assert!(!ep.supports_op(&unknown, &[], &[]).is_supported());
    }
}
