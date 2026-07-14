//! The [`Kernel`] trait and kernel-match / cost types (§4.2).

use onnx_runtime_ir::TensorLayout;

use crate::error::Result;
use crate::tensor::{TensorMut, TensorView};

/// A cost estimate for running a kernel, consumed by the placement cost model
/// (`docs/ORT2.md` §6). All time fields are in **microseconds**; a fuller model
/// (roofline, calibration) lands in `onnx-runtime-cost-model` (Phase 2).
///
/// The struct is `#[non_exhaustive]`: the Phase-2 cost model may add fields
/// (e.g. energy, occupancy) without breaking EP crates. Construct it via
/// [`Cost::ZERO`], [`Cost::new`], or the `with_*` builders rather than a struct
/// literal so those additions stay source-compatible.
///
/// The three time components (`compute_us`, `memory_us`, `transfer_us`) map onto
/// the design's *compute*, *memory-traffic*, and *layout/transfer* estimates;
/// `launch_us` captures fixed dispatch latency (§6.2 `launch_overhead`) and
/// `bytes_moved` carries the raw memory-traffic figure a roofline model needs
/// (§6.3, mirroring the design `Cost::memory_bytes`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct Cost {
    /// Estimated compute time (µs).
    pub compute_us: f64,
    /// Estimated memory-traffic time (µs).
    pub memory_us: f64,
    /// Estimated layout-conversion / cross-device copy time at boundaries (µs).
    pub transfer_us: f64,
    /// Fixed kernel-launch / dispatch latency (µs), independent of size.
    pub launch_us: f64,
    /// Estimated bytes of memory traffic (for roofline / bandwidth models).
    pub bytes_moved: u64,
}

impl Cost {
    /// The zero cost (free op).
    pub const ZERO: Cost = Cost {
        compute_us: 0.0,
        memory_us: 0.0,
        transfer_us: 0.0,
        launch_us: 0.0,
        bytes_moved: 0,
    };

    /// A cost from its three time components; `launch_us` and `bytes_moved`
    /// default to zero (set them via the builders).
    pub fn new(compute_us: f64, memory_us: f64, transfer_us: f64) -> Self {
        Self {
            compute_us,
            memory_us,
            transfer_us,
            ..Self::ZERO
        }
    }

    /// Set the fixed launch/dispatch latency.
    pub fn with_launch_us(mut self, launch_us: f64) -> Self {
        self.launch_us = launch_us;
        self
    }

    /// Set the estimated memory-traffic volume.
    pub fn with_bytes_moved(mut self, bytes_moved: u64) -> Self {
        self.bytes_moved = bytes_moved;
        self
    }

    /// Total estimated wall time (µs): the sum of all time components.
    pub fn total_us(&self) -> f64 {
        self.compute_us + self.memory_us + self.transfer_us + self.launch_us
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_zero_and_total() {
        assert_eq!(Cost::ZERO.total_us(), 0.0);
        let c = Cost::new(10.0, 5.0, 2.0);
        assert_eq!(c.total_us(), 17.0);
        assert_eq!(c.bytes_moved, 0);
        assert_eq!(c.launch_us, 0.0);
    }

    #[test]
    fn cost_builders_are_additive() {
        let c = Cost::new(10.0, 5.0, 2.0)
            .with_launch_us(3.0)
            .with_bytes_moved(4096);
        // launch time folds into the total; bytes_moved is metadata for roofline.
        assert_eq!(c.total_us(), 20.0);
        assert_eq!(c.bytes_moved, 4096);
    }

    #[test]
    fn kernel_match_reports_support() {
        let supported = KernelMatch::Supported {
            cost: Cost::new(1.0, 0.0, 0.0),
            required_input_layouts: None,
            output_layouts: vec![],
        };
        assert!(supported.is_supported());
        assert!(!KernelMatch::Unsupported.is_supported());
    }
}
