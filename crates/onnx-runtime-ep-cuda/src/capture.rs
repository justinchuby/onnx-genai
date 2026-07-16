//! CUDA graph capture eligibility.

use onnx_runtime_ep_api::Kernel;

/// Returns whether every kernel in a subgraph is eligible for CUDA graph capture.
///
/// CUDA graph capture paths must use this gate before beginning capture. A
/// single incompatible kernel makes the entire subgraph ineligible.
pub fn subgraph_graph_capturable(kernels: &[&dyn Kernel]) -> bool {
    kernels.iter().all(|kernel| kernel.cuda_graph_compatible())
}

#[cfg(test)]
mod tests {
    use onnx_runtime_ep_api::{Kernel, Result, TensorMut, TensorView};

    use super::subgraph_graph_capturable;

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
}
