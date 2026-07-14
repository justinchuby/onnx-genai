//! The [`Kernel`] trait and kernel-match / cost types (§4.2).

use onnx_runtime_ir::TensorLayout;

use crate::error::Result;
use crate::tensor::{TensorMut, TensorView};

/// A cost estimate for running a kernel, used by the placement cost model
/// (`docs/ORT2.md` §6). Times are in microseconds; a fuller model lands in
/// `onnx-runtime-cost-model` (Phase 2).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost {
    /// Estimated compute time (µs).
    pub compute_us: f64,
    /// Estimated memory-traffic time (µs).
    pub memory_us: f64,
    /// Estimated layout-conversion / copy time at boundaries (µs).
    pub transfer_us: f64,
}

impl Cost {
    /// Total estimated wall time (µs).
    pub fn total_us(&self) -> f64 {
        self.compute_us + self.memory_us + self.transfer_us
    }
}

/// Result of [`crate::ExecutionProvider::supports_op`].
pub enum KernelMatch {
    Supported {
        cost: Cost,
        /// Layouts the kernel requires for each input, if constrained.
        required_input_layouts: Option<Vec<TensorLayout>>,
        /// Layouts the kernel produces for each output.
        output_layouts: Vec<TensorLayout>,
    },
    Unsupported,
}

impl KernelMatch {
    /// Whether the op is supported.
    pub fn is_supported(&self) -> bool {
        matches!(self, KernelMatch::Supported { .. })
    }
}

/// A kernel ready to execute a specific op with specific shapes (§4.2).
pub trait Kernel: Send {
    /// Execute over device-resident inputs/outputs.
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()>;

    /// Estimated FLOPs, if known (for the cost model).
    fn estimated_flops(&self) -> Option<u64> {
        None
    }

    /// Whether the kernel accepts a non-contiguous (strided) input at `idx`.
    fn supports_strided_input(&self, input_idx: usize) -> bool {
        let _ = input_idx;
        false
    }

    /// The layout the kernel writes most efficiently, if it has a preference.
    fn preferred_output_layout(&self) -> Option<TensorLayout> {
        None
    }

    /// Whether this kernel can be captured inside a CUDA graph.
    fn cuda_graph_compatible(&self) -> bool {
        false
    }
}
