//! The [`CpuExecutionProvider`] — a Phase 1 stub implementation of the EP
//! trait. Method bodies are `todo!()` pending the oneDNN kernel bindings.

use onnx_runtime_ep_api::{
    DeviceBuffer, EpConfig, ExecutionProvider, Fence, Kernel, KernelMatch, Result,
};
use onnx_runtime_ir::{DeviceId, DeviceType, Node, Shape, TensorLayout};

/// CPU execution provider. Always available; the fallback EP for any op.
#[derive(Debug, Default)]
pub struct CpuExecutionProvider {
    device: DeviceId,
    initialized: bool,
}

impl CpuExecutionProvider {
    /// Construct a CPU EP bound to `CPU:0`.
    pub fn new() -> Self {
        Self {
            device: DeviceId::cpu(),
            initialized: false,
        }
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
        // CPU EP needs no device resources; oneDNN engine setup lands here.
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }

    fn supports_op(&self, _op: &Node, _shapes: &[Shape], _layouts: &[TensorLayout]) -> KernelMatch {
        // Real impl consults the CPU op registry + oneDNN capabilities.
        todo!("ort2-ep-cpu: report supported CPU ops with cost + layouts")
    }

    fn get_kernel(&self, _op: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        todo!("ort2-ep-cpu: instantiate oneDNN-backed kernel for op")
    }

    fn allocate(&self, _size: usize, _alignment: usize) -> Result<DeviceBuffer> {
        todo!("ort2-ep-cpu: aligned host allocation")
    }

    fn deallocate(&self, _buffer: DeviceBuffer) -> Result<()> {
        todo!("ort2-ep-cpu: free host allocation")
    }

    fn copy(&self, _src: &DeviceBuffer, _dst: &mut DeviceBuffer, _size: usize) -> Result<()> {
        todo!("ort2-ep-cpu: host memcpy")
    }

    fn copy_async(
        &self,
        _src: &DeviceBuffer,
        _dst: &mut DeviceBuffer,
        _size: usize,
    ) -> Result<Fence> {
        todo!("ort2-ep-cpu: host copy is synchronous; return a signaled fence")
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
}
