//! The [`Kernel`] trait and kernel-match / cost types (§4.2).

use onnx_runtime_ir::TensorLayout;

use crate::error::Result;
use crate::tensor::{TensorMut, TensorView};
use crate::weight::WeightHandle;

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

/// A zero-copy **view output**: a kernel's declaration that one of its outputs
/// is a strided view aliasing one of its inputs' buffers, rather than freshly
/// computed bytes (`docs/ORT2.md` §5.4, lazy PyTorch-style views).
///
/// The `shape` / `strides` / `byte_offset` describe the output tensor relative
/// to the **same base pointer** as the referenced input view (i.e. relative to
/// the input's backing allocation, honoring any offset that input itself
/// already carried). Strides are in **elements** and may be negative (DLPack).
/// The executor records this metadata against the output value and does **not**
/// allocate a buffer or invoke the compute path for that slot; the source
/// buffer is kept alive until the view's consumers have run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewOutput {
    /// Positional index (into the kernel's `inputs`) of the aliased input.
    pub input_index: usize,
    /// Output shape.
    pub shape: Vec<usize>,
    /// Output element strides relative to the aliased input's base pointer.
    pub strides: Vec<i64>,
    /// Byte offset of the output element origin from the aliased input's base.
    pub byte_offset: usize,
}

/// An executor-delivered kernel input. Existing EPs receive `Tensor` variants;
/// an EP advertising the `nxrt` capability may receive a lazy `Weight` at the
/// `pkg.nxrt::BlockQuantizedMoE` boundary.
pub enum KernelInput<'a> {
    Tensor(TensorView<'a>),
    Weight(&'a WeightHandle),
}

impl<'a> KernelInput<'a> {
    pub fn tensor(&self) -> Option<&TensorView<'a>> {
        match self {
            Self::Tensor(view) => Some(view),
            Self::Weight(_) => None,
        }
    }

    pub fn weight(&self) -> Option<&WeightHandle> {
        match self {
            Self::Tensor(_) => None,
            Self::Weight(weight) => Some(weight),
        }
    }
}

/// A kernel ready to execute a specific op with specific shapes (§4.2).
pub trait Kernel: Send {
    /// Tell the kernel which positional inputs are immutable graph constants.
    ///
    /// The session calls this exactly once, immediately after construction.
    /// Kernels may use it to prepack or memoize those inputs. Runtime inputs must
    /// never be marked constant: caching them would return stale results.
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        let _ = constant_inputs;
    }

    /// Execute over device-resident inputs/outputs.
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()>;

    /// Execute through the general weight-delivery seam.
    ///
    /// The default adapter accepts only resident tensor inputs and forwards to
    /// [`Kernel::execute`], so existing EPs compile and behave identically.
    /// Paging-aware kernels override this method to consume lazy handles.
    fn execute_with_inputs(
        &self,
        inputs: &[KernelInput<'_>],
        outputs: &mut [TensorMut],
    ) -> Result<()> {
        let views = inputs
            .iter()
            .map(|input| {
                input.tensor().copied().ok_or_else(|| {
                    crate::EpError::KernelFailed(
                        "kernel received a lazy WeightHandle without implementing \
                         execute_with_inputs"
                            .into(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.execute(&views, outputs)
    }

    /// Estimated FLOPs, if known (for the cost model).
    fn estimated_flops(&self) -> Option<u64> {
        None
    }

    /// Attempt to express this node's outputs as zero-copy [`ViewOutput`]s over
    /// its inputs instead of computing bytes (the layout/movement-op fast path).
    ///
    /// `inputs` carries the real (possibly already-strided) input views and
    /// `num_outputs` is the node's output arity. Returning:
    /// * `None` — the default — means "compute normally": the executor allocates
    ///   output buffers and calls [`Kernel::execute`].
    /// * `Some(specs)` means every output is a view; `specs.len()` MUST equal
    ///   `num_outputs`. A kernel that can view some but not all outputs must
    ///   return `None` (all-or-nothing) so correctness never regresses.
    ///
    /// When `Some` is returned, [`Kernel::execute`] is **not** invoked.
    fn view_outputs(&self, inputs: &[TensorView], num_outputs: usize) -> Option<Vec<ViewOutput>> {
        let _ = (inputs, num_outputs);
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use crate::{DeviceId, DevicePtr};

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

    struct LegacyKernel {
        called: Arc<AtomicBool>,
    }

    impl Kernel for LegacyKernel {
        fn execute(&self, inputs: &[TensorView], _outputs: &mut [TensorMut]) -> Result<()> {
            assert_eq!(inputs.len(), 1);
            assert_eq!(inputs[0].shape, &[4]);
            self.called.store(true, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn legacy_kernel_adapter_receives_the_resident_tensor_path() {
        let called = Arc::new(AtomicBool::new(false));
        let kernel = LegacyKernel {
            called: Arc::clone(&called),
        };
        let bytes = [1u8, 2, 3, 4];
        let shape = [4usize];
        let strides = [1i64];
        let inputs = [KernelInput::Tensor(TensorView::new(
            DevicePtr(bytes.as_ptr().cast()),
            onnx_runtime_ir::DataType::Uint8,
            &shape,
            &strides,
            DeviceId::cpu(),
        ))];

        kernel.execute_with_inputs(&inputs, &mut []).unwrap();
        assert!(called.load(Ordering::Relaxed));
    }
}
