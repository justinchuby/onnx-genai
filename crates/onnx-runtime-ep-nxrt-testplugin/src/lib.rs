//! Test-fixture nxrt plugin built on the **real shipped ABI** from
//! `onnx-runtime-ep-nxrt-abi`.
//!
//! This crate is a `cdylib` exporting `NxrtNegotiate` + `NxrtCreateEpFactories`
//! via the `export_nxrt_ep_factories!` macro. It implements a trivial CPU-backed
//! EP for integration testing purposes only.
//!
//! # Negative-test control
//!
//! Environment variables (read at runtime) produce broken behaviour from the
//! same binary:
//! - `NXRT_TEST_PANIC=1`: panics inside the EP constructor (containment test)
//! - `NXRT_TEST_FACTORY_ERROR=1`: EP constructor returns an error

use std::sync::atomic::{AtomicUsize, Ordering};

use onnx_runtime_ep_api::{
    DeviceBuffer, EpConfig, EpError, ExecutionProvider, Fence, Kernel, KernelMatch, Result,
};
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};

/// Global counter tracking live EP instances. Tests assert this returns to zero.
pub static LIVE_EP_COUNT: AtomicUsize = AtomicUsize::new(0);

// ─── Minimal test EP implementation ─────────────────────────────────────────

/// A trivial CPU-backed EP for testing the nxrt ABI round trip.
pub struct TestNxrtEp {
    initialized: bool,
}

impl Default for TestNxrtEp {
    fn default() -> Self {
        Self::new()
    }
}

impl TestNxrtEp {
    pub fn new() -> Self {
        LIVE_EP_COUNT.fetch_add(1, Ordering::SeqCst);
        Self { initialized: false }
    }
}

impl Drop for TestNxrtEp {
    fn drop(&mut self) {
        LIVE_EP_COUNT.fetch_sub(1, Ordering::SeqCst);
    }
}

impl ExecutionProvider for TestNxrtEp {
    fn consume_route_residency_at_boundary(&self) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "NxrtTestPlugin"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn device_id(&self) -> DeviceId {
        DeviceId::new(DeviceType::Cpu, 0)
    }

    fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }

    fn supports_op(
        &self,
        _op: &Node,
        _opset: u64,
        _shapes: &[Shape],
        _input_dtypes: &[DataType],
        _layouts: &[TensorLayout],
    ) -> KernelMatch {
        KernelMatch::Unsupported {
            reason: "test plugin declines all ops (fail-closed)".into(),
        }
    }

    fn get_kernel(
        &self,
        _op: &Node,
        _shapes: &[Vec<usize>],
        _opset: u64,
    ) -> Result<Box<dyn Kernel>> {
        Err(EpError::KernelFailed("test plugin has no kernels".into()))
    }

    fn allocate(&self, _size: usize, _alignment: usize) -> Result<DeviceBuffer> {
        Err(EpError::KernelFailed("test plugin cannot allocate".into()))
    }

    fn deallocate(&self, _buffer: DeviceBuffer) -> Result<()> {
        Err(EpError::KernelFailed(
            "test plugin cannot deallocate".into(),
        ))
    }

    fn copy(&self, _src: &DeviceBuffer, _dst: &mut DeviceBuffer, _size: usize) -> Result<()> {
        Err(EpError::KernelFailed("test plugin cannot copy".into()))
    }

    fn copy_async(
        &self,
        _src: &DeviceBuffer,
        _dst: &mut DeviceBuffer,
        _size: usize,
    ) -> Result<Fence> {
        Err(EpError::KernelFailed(
            "test plugin cannot copy_async".into(),
        ))
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}

// ─── ABI export via the real macro ──────────────────────────────────────────

fn construct_ep() -> Box<dyn ExecutionProvider> {
    if std::env::var("NXRT_TEST_PANIC").is_ok() {
        panic!("deliberate test panic in NxrtCreateEpFactories");
    }
    if std::env::var("NXRT_TEST_FACTORY_ERROR").is_ok() {
        // We can't return an error from the constructor closure (it returns
        // a Box<dyn EP>), so panicking is the mechanism the macro catches.
        panic!("NXRT_TEST_FACTORY_ERROR: simulated factory failure");
    }
    Box::new(TestNxrtEp::new())
}

onnx_runtime_ep_nxrt_abi::export_nxrt_ep_factories!(construct_ep);

/// Query the live EP count (for lifetime tests). Exported for test access.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_test_live_ep_count() -> usize {
    LIVE_EP_COUNT.load(Ordering::SeqCst)
}
