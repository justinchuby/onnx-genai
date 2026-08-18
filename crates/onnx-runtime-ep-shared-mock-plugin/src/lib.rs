//! Test-only shared-EP plugin: a CPU-backed execution provider exported
//! through the **shared-EP** factory path (`create_ep_factories_for_shared_ep`).
//!
//! # Why this crate exists
//!
//! The shared-EP ownership model (one `Arc<Mutex<Box<dyn ExecutionProvider +
//! Send>>>` handed to the allocator, sync stream, data transfer, *and* every
//! `OrtEp` a factory produces via `EpHandle::Shared`) is the path a real CUDA
//! plugin takes. Until now it was only
//! covered by tests that call our own `extern "C"` vtable entries directly.
//! That proves the function pointers work; it does **not** prove that a real
//! `OrtSession` can be created against a shared-EP factory, that ORT will
//! partition nodes onto it, that `Run()` produces correct values, or that
//! ORT's real release ordering tears the shared EP down exactly once.
//!
//! This crate is the missing conformance vehicle. It is deliberately
//! **CPU-typed** (`DeviceSupport::cpu_only()`): `factory_get_supported_devices`
//! only matches hardware devices that ORT actually enumerates, so on a
//! GPU-less host a GPU-typed mock would never be selected and no session could
//! ever be created. A CPU-typed shared EP exercises every shared-ownership
//! code path that matters (single instance across sessions, governed kernel
//! workspaces, fused-subgraph intermediates, factory-level teardown) on
//! ordinary CI hardware.
//!
//! **This proves shared-EP ownership, workspace plumbing, and teardown
//! ordering. It proves nothing about CUDA correctness or device memory.**
//! See `docs/CUDA_EP_STATUS.md` and issue #768.
//!
//! # Falsification design
//!
//! [`WorkspaceAddKernel`] (op `Add`) declares a non-zero
//! [`WorkspaceRequirement`] and its [`Kernel::execute`] **always fails**. The
//! only way a model containing `Add` can run is if the executor honours the
//! `workspace_requirement` / `execute_with_workspace` contract and hands over
//! a non-null, correctly-sized, correctly-aligned workspace. A regression that
//! drops workspace plumbing turns every E2E test in this crate red.
//!
//! [`PlainMulKernel`] (op `Mul`) declares [`WorkspaceRequirement::NONE`] and
//! implements only `execute`, so the "no workspace needed" path stays covered.
//!
//! Setting [`nxrt_mock_shared_ep_set_persistent_workspace`] flips the `Add`
//! kernel's declared lifetime to [`WorkspaceLifetime::SessionPersistent`],
//! which the executor has no arena for. It must then pass `None` — routing the
//! kernel to its own self-owned scratch — and must **never** hand over a
//! step-scoped block that ORT recycles when `Compute` returns. The falsifier is
//! `nxrt_mock_shared_ep_persistent_downgraded == 0` while
//! `nxrt_mock_shared_ep_persistent_declined > 0`: any executor that "helpfully"
//! serves the persistent request from scratch memory flips the first counter
//! and fails the test.
//!
//! All observable state is exported through `#[no_mangle]` C accessors so the
//! integration test — which reaches this library through ORT's `dlopen` — can
//! read the same statics ORT's copy mutates.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use onnx_runtime_ep_api::{
    Cost, DeviceBuffer, EpConfig, EpError, ExecutionProvider, Fence, Kernel, KernelMatch,
    Result as EpResult, TensorMetadata, TensorMut, TensorView, WorkspaceLifetime,
    WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ep_plugin::device::DeviceSupport;
use onnx_runtime_ep_plugin::ep::KernelRegistryEntry;
use onnx_runtime_ep_plugin::factory::create_ep_factories_for_shared_ep;
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};
use onnx_runtime_memory_governor::MemoryRole;

/// Alignment the mock `Add` kernel demands of its workspace. Deliberately much
/// larger than the adapter's default 16-byte device alignment so a plumbing
/// bug that ignores `WorkspaceRequirement::alignment` is caught.
pub const MOCK_WORKSPACE_ALIGNMENT: usize = 256;

// ─── Observable counters ─────────────────────────────────────────────────────
//
// These are process-global because the test reads them through `dlsym` on the
// very library ORT loaded (dlopen of an already-loaded path returns the same
// mapping), so both sides observe one set of statics.

static EP_INSTANCES_CREATED: AtomicUsize = AtomicUsize::new(0);
static EP_INSTANCES_LIVE: AtomicUsize = AtomicUsize::new(0);
static EP_SHUTDOWN_CALLS: AtomicUsize = AtomicUsize::new(0);
static EP_ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static EP_DEALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static EP_ALLOC_LIVE: AtomicUsize = AtomicUsize::new(0);
static KERNEL_WORKSPACE_OK: AtomicUsize = AtomicUsize::new(0);
static KERNEL_WORKSPACE_MISSING: AtomicUsize = AtomicUsize::new(0);
static KERNEL_EXECUTE_WITHOUT_WORKSPACE: AtomicUsize = AtomicUsize::new(0);
static KERNEL_MUL_EXECUTED: AtomicUsize = AtomicUsize::new(0);
static KERNEL_PERSISTENT_DECLINED: AtomicUsize = AtomicUsize::new(0);
static KERNEL_PERSISTENT_DOWNGRADED: AtomicUsize = AtomicUsize::new(0);
static KERNEL_WORKSPACE_PLANS: AtomicUsize = AtomicUsize::new(0);

/// When set, `WorkspaceAddKernel` declares a `SessionPersistent` workspace.
static PERSISTENT_WORKSPACE: AtomicBool = AtomicBool::new(false);

macro_rules! counter_export {
    ($fn_name:ident, $static_name:ident, $doc:literal) => {
        #[doc = $doc]
        #[unsafe(no_mangle)]
        pub extern "C" fn $fn_name() -> usize {
            $static_name.load(Ordering::SeqCst)
        }
    };
}

counter_export!(
    nxrt_mock_shared_ep_instances_created,
    EP_INSTANCES_CREATED,
    "Number of `SharedMockEp` values ever constructed. Must stay `1` across \
     multiple sessions if the shared-EP `Arc` is genuinely shared."
);
counter_export!(
    nxrt_mock_shared_ep_instances_live,
    EP_INSTANCES_LIVE,
    "Number of `SharedMockEp` values not yet dropped."
);
counter_export!(
    nxrt_mock_shared_ep_shutdown_calls,
    EP_SHUTDOWN_CALLS,
    "Number of `ExecutionProvider::shutdown()` calls observed."
);
counter_export!(
    nxrt_mock_shared_ep_alloc_calls,
    EP_ALLOC_CALLS,
    "Cumulative `ExecutionProvider::allocate` calls. Workspaces and subgraph \
     intermediates come from ORT scratch (`KernelContext_GetScratchBuffer`), \
     **not** from here, so this must stay `0` during `Run()` — a non-zero value \
     means per-dispatch EP allocate/free came back, which synchronises the \
     device and is illegal during CUDA-graph capture."
);
counter_export!(
    nxrt_mock_shared_ep_dealloc_calls,
    EP_DEALLOC_CALLS,
    "Cumulative `ExecutionProvider::deallocate` calls."
);
counter_export!(
    nxrt_mock_shared_ep_alloc_live,
    EP_ALLOC_LIVE,
    "Allocations made through the EP that have not been freed. Must be `0` \
     after a `Run()` completes — a non-zero value is a per-call leak."
);
counter_export!(
    nxrt_mock_shared_ep_workspace_ok,
    KERNEL_WORKSPACE_OK,
    "Dispatches where the mock `Add` kernel received a valid workspace."
);
counter_export!(
    nxrt_mock_shared_ep_workspace_missing,
    KERNEL_WORKSPACE_MISSING,
    "Dispatches where `execute_with_workspace` was called with `None` despite \
     a non-zero `workspace_requirement` (plumbing regression)."
);
counter_export!(
    nxrt_mock_shared_ep_execute_without_workspace,
    KERNEL_EXECUTE_WITHOUT_WORKSPACE,
    "Dispatches that bypassed `execute_with_workspace` entirely and called \
     `execute` (plumbing regression)."
);
counter_export!(
    nxrt_mock_shared_ep_mul_executed,
    KERNEL_MUL_EXECUTED,
    "Dispatches of the zero-workspace `Mul` kernel."
);
counter_export!(
    nxrt_mock_shared_ep_persistent_declined,
    KERNEL_PERSISTENT_DECLINED,
    "Dispatches where the kernel declared a `SessionPersistent` workspace and \
     the executor correctly passed `None`, routing it to its own self-owned \
     scratch. Must be `> 0` in the persistent scenario."
);
counter_export!(
    nxrt_mock_shared_ep_persistent_downgraded,
    KERNEL_PERSISTENT_DOWNGRADED,
    "Dispatches where the kernel declared a `SessionPersistent` workspace and \
     the executor handed over a workspace anyway. Must stay `0`: ORT reclaims \
     `KernelContext_GetScratchBuffer` memory when `Compute` returns, so that \
     block is recycled behind the kernel's back on the next `Run`."
);
counter_export!(
    nxrt_mock_shared_ep_workspace_plans,
    KERNEL_WORKSPACE_PLANS,
    "Cumulative `Kernel::workspace_requirement` calls the executor actually \
     made. Stands in for the cuBLASLt heuristic search the CUDA GEMM kernels \
     run inside that method: repeated `Run`s of an unchanged shape must not \
     grow this counter, or every decode step pays for planning it already did."
);

/// Cumulative count of node-operand placement **resolutions** the executor
/// performed (`onnx_runtime_ep_plugin::compute::workspace_placement_queries`).
///
/// Not a mock-side counter: it reads the executor's own static, because the
/// property under test is that the *executor* does not work out where a node's
/// operands live unless it is about to place a workspace there. A dispatch that
/// needs no workspace, or whose `SessionPersistent` request is declined, must
/// not grow this.
///
/// A resolution is not the same as an ORT call: a node whose operands are all
/// fused-subgraph intermediates resolves placement from the subgraph memory
/// info without calling ORT at all. This counts decisions, not FFI.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_mock_shared_ep_placement_queries() -> usize {
    onnx_runtime_ep_plugin::compute::workspace_placement_queries()
}

/// The same counter under the name the CUDA and CPU plugins export, so one
/// validation harness reads the same symbol whichever cdylib it is pointed at.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_workspace_placement_queries() -> usize {
    onnx_runtime_ep_plugin::compute::workspace_placement_queries()
}

/// Number of nodes this EP compiled, under the name the CUDA and CPU plugins
/// export. Distinguishes "ORT selected our EP" from "ORT gave our EP a node".
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_compiled_node_count() -> usize {
    onnx_runtime_ep_plugin::ep::compiled_node_count()
}

/// Select the workspace lifetime the mock `Add` kernel declares.
///
/// `0` = `StepScoped` (the request the executor serves from ORT scratch),
/// anything else = `SessionPersistent` (the request the executor must decline
/// with `None` rather than silently downgrade). Call before `CreateSession`,
/// since `workspace_requirement` is consulted per dispatch but the kernel is
/// built at compile time.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_mock_shared_ep_set_persistent_workspace(on: usize) {
    PERSISTENT_WORKSPACE.store(on != 0, Ordering::SeqCst);
}

/// Reset every counter. Tests call this before each scenario.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_mock_shared_ep_reset_counters() {
    for c in [
        &EP_ALLOC_CALLS,
        &EP_DEALLOC_CALLS,
        &KERNEL_WORKSPACE_OK,
        &KERNEL_WORKSPACE_MISSING,
        &KERNEL_EXECUTE_WITHOUT_WORKSPACE,
        &KERNEL_MUL_EXECUTED,
        &KERNEL_PERSISTENT_DECLINED,
        &KERNEL_PERSISTENT_DOWNGRADED,
        &KERNEL_WORKSPACE_PLANS,
    ] {
        c.store(0, Ordering::SeqCst);
    }
    onnx_runtime_ep_plugin::compute::reset_workspace_placement_queries();
    PERSISTENT_WORKSPACE.store(false, Ordering::SeqCst);
    // `EP_INSTANCES_CREATED`, `EP_INSTANCES_LIVE`, `EP_SHUTDOWN_CALLS` and
    // `EP_ALLOC_LIVE` are lifetime invariants, not per-scenario counters, and
    // are deliberately NOT reset.
}

// ─── The mock shared EP ──────────────────────────────────────────────────────

/// A host-memory execution provider exported through the shared-EP path.
#[derive(Debug, Default)]
pub struct SharedMockEp {
    _private: (),
}

impl SharedMockEp {
    /// Construct the (single) shared instance.
    pub fn new() -> Self {
        EP_INSTANCES_CREATED.fetch_add(1, Ordering::SeqCst);
        EP_INSTANCES_LIVE.fetch_add(1, Ordering::SeqCst);
        Self { _private: () }
    }
}

impl Drop for SharedMockEp {
    fn drop(&mut self) {
        EP_INSTANCES_LIVE.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Ops this EP claims. `Add` requires a workspace; `Mul` does not.
const SUPPORTED_OPS: &[&str] = &["Add", "Mul"];

const SUPPORTED_DTYPES: &[DataType] = &[DataType::Float32];

impl ExecutionProvider for SharedMockEp {
    fn name(&self) -> &str {
        "nxrt_shared_mock_ep"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn device_id(&self) -> DeviceId {
        DeviceId::cpu()
    }

    fn initialize(&mut self, _config: &EpConfig) -> EpResult<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> EpResult<()> {
        EP_SHUTDOWN_CALLS.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn supports_op(
        &self,
        op: &Node,
        _opset: u64,
        _shapes: &[Shape],
        input_dtypes: &[DataType],
        _layouts: &[TensorLayout],
    ) -> KernelMatch {
        if !op.domain.is_empty() {
            return KernelMatch::unsupported(format!(
                "shared mock EP only serves the default ONNX domain, got {:?}",
                op.domain
            ));
        }
        if !SUPPORTED_OPS.contains(&op.op_type.as_str()) {
            return KernelMatch::unsupported(format!(
                "shared mock EP supports {SUPPORTED_OPS:?}, got {:?}",
                op.op_type
            ));
        }
        if input_dtypes.iter().any(|d| *d != DataType::Float32) {
            return KernelMatch::unsupported("shared mock EP is f32-only");
        }
        KernelMatch::Supported {
            cost: Cost::ZERO,
            required_input_layouts: None,
            output_layouts: vec![TensorLayout::contiguous()],
        }
    }

    fn get_kernel(
        &self,
        op: &Node,
        _shapes: &[Vec<usize>],
        _opset: u64,
    ) -> EpResult<Box<dyn Kernel>> {
        match op.op_type.as_str() {
            "Add" => Ok(Box::new(WorkspaceAddKernel)),
            "Mul" => Ok(Box::new(PlainMulKernel)),
            other => Err(EpError::KernelFailed(format!(
                "shared mock EP has no kernel for {other}"
            ))),
        }
    }

    fn allocate(&self, size: usize, alignment: usize) -> EpResult<DeviceBuffer> {
        // Mirror the real adapter contract: a zero-size request must still
        // yield a unique non-null pointer, and Rust's global allocator forbids
        // a zero-size `Layout`.
        let bytes = size.max(1);
        let align = alignment.max(1);
        if !align.is_power_of_two() {
            return Err(EpError::AlignmentError);
        }
        let layout = std::alloc::Layout::from_size_align(bytes, align)
            .map_err(|_| EpError::AlignmentError)?;
        // SAFETY: `layout` has non-zero size (bytes >= 1).
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(EpError::OutOfMemory {
                requested: size,
                available: 0,
            });
        }
        EP_ALLOC_CALLS.fetch_add(1, Ordering::SeqCst);
        EP_ALLOC_LIVE.fetch_add(1, Ordering::SeqCst);
        // SAFETY: `ptr` is a live allocation of `size`/`alignment` as reported.
        Ok(unsafe { DeviceBuffer::from_raw_parts(ptr.cast(), DeviceId::cpu(), size, align) })
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> EpResult<()> {
        let ptr = buffer.as_ptr();
        let size = buffer.len();
        let align = buffer.alignment().max(1);
        if !ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(size.max(1), align)
                .map_err(|_| EpError::AlignmentError)?;
            // SAFETY: identical layout to the one used in `allocate`.
            unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
        }
        EP_DEALLOC_CALLS.fetch_add(1, Ordering::SeqCst);
        EP_ALLOC_LIVE.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> EpResult<()> {
        if size > 0 {
            // SAFETY: both buffers are host allocations of at least `size`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().cast::<u8>(),
                    dst.as_mut_ptr().cast::<u8>(),
                    size,
                );
            }
        }
        Ok(())
    }

    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> EpResult<Fence> {
        self.copy(src, dst, size)?;
        Ok(Fence::signalled())
    }

    fn sync(&self) -> EpResult<()> {
        Ok(())
    }
}

// ─── Kernels ─────────────────────────────────────────────────────────────────

fn f32_operands<'a>(
    inputs: &'a [TensorView<'a>],
    op: &str,
) -> EpResult<(&'a TensorView<'a>, &'a TensorView<'a>, usize)> {
    if inputs.len() != 2 {
        return Err(EpError::KernelFailed(format!(
            "{op}: expected 2 inputs, got {}",
            inputs.len()
        )));
    }
    let (a, b) = (&inputs[0], &inputs[1]);
    if a.dtype != DataType::Float32 || b.dtype != DataType::Float32 {
        return Err(EpError::KernelFailed(format!("{op}: f32 inputs only")));
    }
    if a.numel() != b.numel() {
        return Err(EpError::KernelFailed(format!(
            "{op}: shared mock EP does not broadcast ({:?} vs {:?})",
            a.shape, b.shape
        )));
    }
    Ok((a, b, a.numel()))
}

/// Elementwise `Add` that **only** works when handed a governed workspace.
///
/// The sums are staged through the workspace and copied to the output, so a
/// workspace that is too small, misaligned, or absent cannot produce a correct
/// result — the kernel returns an error rather than silently degrading.
pub struct WorkspaceAddKernel;

impl Kernel for WorkspaceAddKernel {
    fn execute(&self, _inputs: &[TensorView], _outputs: &mut [TensorMut]) -> EpResult<()> {
        KERNEL_EXECUTE_WITHOUT_WORKSPACE.fetch_add(1, Ordering::SeqCst);
        Err(EpError::KernelFailed(
            "WorkspaceAddKernel::execute called directly: this kernel declares a non-zero \
             WorkspaceRequirement and must be dispatched through execute_with_workspace(). \
             The executor is not honouring the workspace contract."
                .to_string(),
        ))
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
    ) -> EpResult<WorkspaceRequirement> {
        // Stands in for the cuBLASLt heuristic search a real CUDA GEMM kernel
        // runs here. The executor must not repeat it for a shape it has already
        // planned; `workspace_plans_do_not_repeat_for_an_unchanged_shape`
        // asserts on this counter.
        KERNEL_WORKSPACE_PLANS.fetch_add(1, Ordering::SeqCst);
        let numel: usize = inputs
            .first()
            .map(|m| m.shape.iter().product())
            .unwrap_or(0usize);
        let persistent = PERSISTENT_WORKSPACE.load(Ordering::SeqCst);
        Ok(WorkspaceRequirement {
            bytes: (numel * std::mem::size_of::<f32>()) as u64,
            alignment: MOCK_WORKSPACE_ALIGNMENT,
            lifetime: if persistent {
                WorkspaceLifetime::SessionPersistent
            } else {
                WorkspaceLifetime::StepScoped
            },
            role: MemoryRole::Workspace {
                step_scoped: !persistent,
            },
        })
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> EpResult<()> {
        if PERSISTENT_WORKSPACE.load(Ordering::SeqCst) {
            // Mirrors a real persistent-declaring kernel with a self-owned
            // fallback (`GroupQueryAttention`, or `StandardAttention` on its
            // single-token decode geometry): the executor has no
            // session-persistent device arena, so it must decline with `None`
            // and let the kernel fall back to scratch it owns and controls the
            // lifetime of.
            if workspace.is_some() {
                KERNEL_PERSISTENT_DOWNGRADED.fetch_add(1, Ordering::SeqCst);
                return Err(EpError::KernelFailed(
                    "WorkspaceAddKernel: the executor served a SessionPersistent workspace \
                     request from step-scoped memory. ORT reclaims that block when Compute \
                     returns, so the kernel would reuse recycled memory on the next Run."
                        .to_string(),
                ));
            }
            KERNEL_PERSISTENT_DECLINED.fetch_add(1, Ordering::SeqCst);
            let (a, b, numel) = f32_operands(inputs, "Add")?;
            if outputs.len() != 1 {
                return Err(EpError::KernelFailed(format!(
                    "Add: expected 1 output, got {}",
                    outputs.len()
                )));
            }
            // Self-owned scratch, allocated and freed by the kernel.
            let mut owned = vec![0f32; numel];
            // SAFETY: this EP is CPU-typed, so operand pointers are host
            // resident, and `f32_operands` checked the element count.
            unsafe {
                let ap = a.data_ptr::<f32>();
                let bp = b.data_ptr::<f32>();
                for (i, slot) in owned.iter_mut().enumerate() {
                    *slot = *ap.add(i) + *bp.add(i);
                }
                let out = outputs[0].data_ptr_mut::<f32>();
                std::ptr::copy_nonoverlapping(owned.as_ptr(), out, numel);
            }
            return Ok(());
        }
        let (a, b, numel) = f32_operands(inputs, "Add")?;
        let need = numel * std::mem::size_of::<f32>();

        let Some(ws) = workspace else {
            KERNEL_WORKSPACE_MISSING.fetch_add(1, Ordering::SeqCst);
            return Err(EpError::KernelFailed(format!(
                "WorkspaceAddKernel: workspace_requirement asked for {need} bytes but the \
                 executor passed None"
            )));
        };
        let ws_ptr = ws.ptr().as_ptr::<f32>();
        if ws_ptr.is_null() {
            return Err(EpError::KernelFailed(
                "WorkspaceAddKernel: executor passed a null workspace pointer".to_string(),
            ));
        }
        if ws.bytes() < need {
            return Err(EpError::KernelFailed(format!(
                "WorkspaceAddKernel: workspace too small — need {need} bytes, got {}",
                ws.bytes()
            )));
        }
        if !(ws_ptr as usize).is_multiple_of(MOCK_WORKSPACE_ALIGNMENT) {
            return Err(EpError::KernelFailed(format!(
                "WorkspaceAddKernel: workspace pointer {ws_ptr:p} is not {MOCK_WORKSPACE_ALIGNMENT}-byte \
                 aligned as requested by workspace_requirement"
            )));
        }
        if outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "Add: expected 1 output, got {}",
                outputs.len()
            )));
        }

        // SAFETY: all pointers are host-resident (this EP is CPU-typed) and
        // sized for at least `numel` f32 elements, checked above.
        unsafe {
            let ap = a.data_ptr::<f32>();
            let bp = b.data_ptr::<f32>();
            for i in 0..numel {
                *ws_ptr.add(i) = *ap.add(i) + *bp.add(i);
            }
            let out = outputs[0].data_ptr_mut::<f32>();
            std::ptr::copy_nonoverlapping(ws_ptr as *const f32, out, numel);
        }
        KERNEL_WORKSPACE_OK.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Elementwise `Mul` that needs no workspace — keeps the
/// `WorkspaceRequirement::NONE` path covered by the same E2E run.
pub struct PlainMulKernel;

impl Kernel for PlainMulKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> EpResult<()> {
        let (a, b, numel) = f32_operands(inputs, "Mul")?;
        if outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "Mul: expected 1 output, got {}",
                outputs.len()
            )));
        }
        // SAFETY: host-resident buffers sized for at least `numel` f32s.
        unsafe {
            let ap = a.data_ptr::<f32>();
            let bp = b.data_ptr::<f32>();
            let out = outputs[0].data_ptr_mut::<f32>();
            for i in 0..numel {
                *out.add(i) = *ap.add(i) * *bp.add(i);
            }
        }
        KERNEL_MUL_EXECUTED.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

// ─── Plugin entry points ─────────────────────────────────────────────────────

/// Kernel-registry entries advertised to ORT.
pub fn kernel_registry_entries() -> Vec<KernelRegistryEntry> {
    SUPPORTED_OPS
        .iter()
        .map(|op| KernelRegistryEntry {
            op_type: op,
            domain: "",
            since_version: 7,
            end_version: i32::MAX,
            supported_dtypes: SUPPORTED_DTYPES,
            input_dtype_constraints: &[],
        })
        .collect()
}

/// ORT plugin-EP entry point: create the shared-EP factory.
///
/// # Safety
///
/// Called by ORT's plugin loader; all pointer arguments must satisfy the ORT
/// plugin-EP C ABI contract.
///
/// # Panic safety
///
/// Panics are caught and converted into a failure `OrtStatus`; unwinding into
/// ORT would be undefined behaviour.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn CreateEpFactories(
    _registration_name: *const std::ffi::c_char,
    api_base: *const onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtApiBase,
    _logger: *const onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtLogger,
    out_factories: *mut *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtEpFactory,
    max_factories: usize,
    out_num: *mut usize,
) -> *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtStatus {
    let out_factories_raw = out_factories;
    let out_num_raw = out_num;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ep: Box<dyn ExecutionProvider + Send> = Box::new(SharedMockEp::new());
        let shared = Arc::new(Mutex::new(ep));
        unsafe {
            create_ep_factories_for_shared_ep(
                api_base,
                out_factories_raw,
                max_factories,
                out_num_raw,
                "nxrt_shared_mock_ep",
                shared,
                kernel_registry_entries(),
                DeviceSupport::cpu_only(),
                std::ptr::null_mut(),
            )
        }
    }));
    match result {
        Ok(status) => status,
        Err(_) => {
            if !out_num_raw.is_null() {
                unsafe { *out_num_raw = 0 };
            }
            onnx_runtime_ep_plugin::panic_to_fail_status(
                "CreateEpFactories: shared mock EP construction panicked (fail-closed)",
            )
        }
    }
}

/// ORT plugin-EP entry point: release the factory and tear down the shared EP.
///
/// Uses [`onnx_runtime_ep_plugin::factory::release_ep_factory`], which performs
/// the explicit shared-EP `shutdown()` when the factory holds the last
/// reference. ORT invokes this from `UnregisterExecutionProviderLibrary`,
/// after every session, allocator, stream and `OrtEp` has been released — so
/// the explicit path is the one normal teardown takes.
///
/// # Safety
///
/// `factory` must be a pointer produced by `CreateEpFactories` in this library
/// and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ReleaseEpFactory(
    factory: *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtEpFactory,
) -> *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtStatus {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: caller guarantees provenance per the ABI contract.
        unsafe { onnx_runtime_ep_plugin::factory::release_ep_factory(factory) }
    }));
    match result {
        Ok(status) => status,
        Err(_) => onnx_runtime_ep_plugin::panic_to_fail_status(
            "ReleaseEpFactory: panic during shared mock factory release (fail-closed)",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
    use onnx_runtime_ir::compute_contiguous_strides;

    #[test]
    fn add_kernel_requires_workspace_and_execute_fails_closed() {
        let k = WorkspaceAddKernel;
        let shape = [4usize];
        let strides = compute_contiguous_strides(&shape);
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];

        let av = TensorView::new(
            DevicePtr(a.as_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let bv = TensorView::new(
            DevicePtr(b.as_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let mut ov = TensorMut::new(
            DevicePtrMut(out.as_mut_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );

        let err = k
            .execute(&[av, bv], std::slice::from_mut(&mut ov))
            .expect_err("execute() must fail — the kernel demands a workspace");
        assert!(
            err.to_string().contains("execute_with_workspace"),
            "error must name the contract it needs: {err}"
        );
    }

    #[test]
    fn add_kernel_reports_exact_workspace_bytes_and_alignment() {
        let k = WorkspaceAddKernel;
        let shape = [2usize, 4];
        let meta = TensorMetadata::new(DataType::Float32, &shape, true);
        let req = k.workspace_requirement(&[meta, meta]).unwrap();
        assert_eq!(req.bytes, 8 * 4);
        assert_eq!(req.alignment, MOCK_WORKSPACE_ALIGNMENT);
    }

    #[test]
    fn add_kernel_rejects_undersized_workspace() {
        let k = WorkspaceAddKernel;
        let shape = [4usize];
        let strides = compute_contiguous_strides(&shape);
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        // Aligned, but only big enough for one element.
        let mut scratch = vec![0u8; MOCK_WORKSPACE_ALIGNMENT * 2];
        let base = scratch.as_mut_ptr();
        let offset = base.align_offset(MOCK_WORKSPACE_ALIGNMENT);
        // SAFETY: `offset` is within `scratch`'s allocation by construction.
        let aligned = unsafe { base.add(offset) };

        let av = TensorView::new(
            DevicePtr(a.as_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let bv = TensorView::new(
            DevicePtr(b.as_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let mut ov = TensorMut::new(
            DevicePtrMut(out.as_mut_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );

        let ws = WorkspaceView::new(DevicePtrMut(aligned.cast()), 4);
        let err = k
            .execute_with_workspace(&[av, bv], std::slice::from_mut(&mut ov), Some(ws))
            .expect_err("undersized workspace must be rejected");
        assert!(err.to_string().contains("too small"), "{err}");
    }

    #[test]
    fn add_kernel_computes_through_a_valid_workspace() {
        let k = WorkspaceAddKernel;
        let shape = [4usize];
        let strides = compute_contiguous_strides(&shape);
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        let mut scratch = vec![0u8; MOCK_WORKSPACE_ALIGNMENT * 2];
        let base = scratch.as_mut_ptr();
        let offset = base.align_offset(MOCK_WORKSPACE_ALIGNMENT);
        // SAFETY: `offset` is within `scratch`'s allocation by construction.
        let aligned = unsafe { base.add(offset) };

        let av = TensorView::new(
            DevicePtr(a.as_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let bv = TensorView::new(
            DevicePtr(b.as_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let mut ov = TensorMut::new(
            DevicePtrMut(out.as_mut_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );

        let ws = WorkspaceView::new(DevicePtrMut(aligned.cast()), 4 * 4);
        k.execute_with_workspace(&[av, bv], std::slice::from_mut(&mut ov), Some(ws))
            .expect("valid workspace must succeed");
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn ep_allocate_zero_bytes_yields_a_freeable_non_null_pointer() {
        let ep = SharedMockEp::new();
        let buf = ep.allocate(0, 16).expect("zero-size allocation must work");
        assert!(!buf.as_ptr().is_null());
        ep.deallocate(buf).expect("free must succeed");
    }

    #[test]
    fn registry_entries_cover_both_ops() {
        let entries = kernel_registry_entries();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().any(|e| e.op_type == "Add"));
        assert!(entries.iter().any(|e| e.op_type == "Mul"));
        assert!(entries.iter().all(|e| e.domain.is_empty()));
    }
}
