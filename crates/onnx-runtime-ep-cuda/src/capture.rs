//! CUDA graph capture eligibility.

use onnx_runtime_ep_api::Kernel;
use onnx_runtime_ep_api::{EpError, Result};

/// Returns whether every kernel in a subgraph is eligible for CUDA graph capture.
///
/// CUDA graph capture paths must use this gate before beginning capture. A
/// single incompatible kernel makes the entire subgraph ineligible.
pub fn subgraph_graph_capturable(kernels: &[&dyn Kernel]) -> bool {
    kernels.iter().all(|kernel| kernel.cuda_graph_compatible())
}

/// Reject a kernel sequence before stream capture unless every kernel has
/// explicitly opted into the CUDA graph contract.
pub fn require_subgraph_graph_capturable(kernels: &[&dyn Kernel]) -> Result<()> {
    let incompatible = kernels
        .iter()
        .enumerate()
        .filter_map(|(index, kernel)| (!kernel.cuda_graph_compatible()).then_some(index))
        .collect::<Vec<_>>();
    if incompatible.is_empty() {
        return Ok(());
    }

    Err(EpError::KernelFailed(format!(
        "cuda_ep: CUDA graph capture rejected before begin_capture: kernels at sequence indices \
         {incompatible:?} are not graph-compatible (capture forbids device allocation/free, \
         lazy NVRTC or library initialization, D2H validation, stream synchronization, and \
         changing tensor shapes or addresses)"
    )))
}

#[cfg(test)]
mod tests {
    use onnx_runtime_ep_api::{Kernel, Result, TensorMut, TensorView};

    use super::{require_subgraph_graph_capturable, subgraph_graph_capturable};

    struct TestKernel {
        capturable: bool,
    }

    impl Kernel for TestKernel {
        fn execute(&self, _inputs: &[TensorView], _outputs: &mut [TensorMut]) -> Result<()> {
            Ok(())
        }

        fn cuda_graph_compatible(&self) -> bool {
            self.capturable
        }
    }

    #[test]
    fn empty_subgraph_is_capturable() {
        assert!(subgraph_graph_capturable(&[]));
    }

    #[test]
    fn all_kernels_must_be_capturable() {
        let compatible_a = TestKernel { capturable: true };
        let compatible_b = TestKernel { capturable: true };
        let incompatible = TestKernel { capturable: false };

        assert!(subgraph_graph_capturable(&[&compatible_a, &compatible_b]));
        assert!(!subgraph_graph_capturable(&[
            &compatible_a,
            &incompatible,
            &compatible_b,
        ]));
    }

    #[test]
    fn audit_returns_a_clear_error_for_incompatible_kernels() {
        let compatible = TestKernel { capturable: true };
        let incompatible = TestKernel { capturable: false };

        let error = require_subgraph_graph_capturable(&[&compatible, &incompatible]).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("rejected before begin_capture"));
        assert!(message.contains("[1]"));
        assert!(message.contains("allocation/free"));
    }
}
