//! ONNX `DFT` operator (opset 17+): the discrete Fourier transform.
//!
//! Supports real and complex input, forward and inverse, full and onesided output.
//! On macOS/iOS the fast path uses Accelerate's vDSP DFT for power-of-two lengths;
//! Float64 and other portable paths use a Cooley–Tukey radix-2 FFT for
//! power-of-two lengths or a direct O(N²) DFT for other lengths.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use num_traits::{Float, FloatConst};
use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView,
    WorkspaceExecutionPreparation, WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ir::{
    DataType, Node,
    dft::{self, DftMode},
};

use super::float_workspace::FloatWorkspace;
use crate::dtype::{
    scalar_i64, to_dense_f32_widen, validate_f32_widen_input, write_dense_f32_narrow,
};

/// Dispatch counter for the vDSP Accelerate DFT fast path.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub static DFT_VDSP_TEST_HITS: AtomicU64 = AtomicU64::new(0);

/// Dispatch counter for the radix-2 FFT fallback path.
pub static DFT_FFT_TEST_HITS: AtomicU64 = AtomicU64::new(0);

/// Total transforms that took *a* fast path rather than the naive O(n^2) DFT.
///
/// The two counters above are mutually exclusive per call and which of them
/// fires is a platform detail: on Apple targets [`DftPlan::new`] requests a
/// vDSP setup for every power-of-two `n >= 4`, and when vDSP returns one
/// [`DftPlan::transform`] returns from that arm before the radix-2 counter is
/// reached. A null setup falls through to radix-2, so neither counter alone is
/// a reliable answer even on one target. A caller that means
/// "this transform did not fall back to the naive path" must therefore read the
/// sum. Reading `DFT_FFT_TEST_HITS` alone states an x86-shaped implementation
/// detail, and on Apple targets it is unsatisfiable for any eligible `n` --
/// which is exactly how #2089 turned an STFT assertion into a deterministic red
/// on `Rust coverage (macOS arm64)`.
///
/// A test that specifically wants the *radix-2* path is a different question
/// and must not use this: pick an `n` below the vDSP minimum and read
/// `DFT_FFT_TEST_HITS`, as `fft_fallback_reachability` does with `n = 2`.
///
/// Process-global and monotonic. Prefer [`dft_fast_path_hits_this_thread`] for
/// a delta around a specific call: this counter is shared by every thread, and
/// the unit-test binary runs tests in parallel, so a delta taken from it is a
/// lower bound on the caller's own transforms rather than a measurement of
/// them.
pub fn dft_fast_path_hits() -> u64 {
    let hits = DFT_FFT_TEST_HITS.load(Ordering::Relaxed);
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    let hits = hits + DFT_VDSP_TEST_HITS.load(Ordering::Relaxed);
    hits
}

thread_local! {
    /// Per-thread mirror of [`dft_fast_path_hits`].
    static LOCAL_FAST_PATH_HITS: Cell<u64> = const { Cell::new(0) };
}

/// Fast-path transforms performed *on the calling thread*.
///
/// The process-global counters cannot express "this call took a fast path".
/// A test differences them around a call and asserts `after - before >= k`,
/// but the true delta is `own + other`, where `other` is whatever concurrent
/// tests transformed inside the same window. `own >= k` implies the assertion
/// holds, so a global delta never yields a false red -- but the converse is
/// what a regression guard needs, and it does not hold: a frame dropping to
/// the naive path (`own = k - 1`) is masked whenever a concurrent test
/// contributes one hit. The assertion then passes without having measured
/// anything about the code under test.
///
/// This counter is not shared, so a delta around a call is exactly that
/// call's fast-path count -- provided the work happens on the calling thread,
/// which holds for `Dft` and `Stft`: neither fans out. Callers that fan out
/// must use the global counters and accept the lower bound.
pub fn dft_fast_path_hits_this_thread() -> u64 {
    LOCAL_FAST_PATH_HITS.with(Cell::get)
}

fn record_fast_path_hit() {
    LOCAL_FAST_PATH_HITS.with(|hits| hits.set(hits.get().wrapping_add(1)));
}

pub struct DftFactory;

impl KernelFactory for DftFactory {
    fn unsupported_reason(
        &self,
        node: &Node,
        _opset: u64,
        shapes: &[onnx_runtime_ir::Shape],
        _dtypes: &[onnx_runtime_ir::DataType],
    ) -> Option<String> {
        let mode = match DftMode::parse(node) {
            Ok(mode) => mode,
            Err(reason) => return Some(reason),
        };
        shapes
            .first()
            .and_then(|shape| shape.last())
            .and_then(|dim| dim.as_static())
            .and_then(|components| mode.validate_components(components).err())
    }

    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let mode = DftMode::parse(node).map_err(EpError::KernelFailed)?;
        if let Some(&components) = input_shapes.first().and_then(|shape| shape.last()) {
            mode.validate_components(components)
                .map_err(EpError::KernelFailed)?;
        }
        let opset = node.local_opset().unwrap_or(17);
        Ok(Box::new(DftKernel {
            mode,
            opset,
            default_axis: dft::default_axis(node, opset),
        }))
    }
}

struct DftKernel {
    mode: DftMode,
    opset: u64,
    default_axis: i64,
}

struct DftGeometry {
    axis: usize,
    signal_len: usize,
    n: usize,
    output_len: usize,
    output_components: usize,
    batches: usize,
    output_elements: usize,
    dtype: DataType,
}

impl DftGeometry {
    fn workspace<T: Float>(&self) -> Result<FloatWorkspace<4, T>> {
        let len = if self.batches == 0 { 0 } else { self.n };
        FloatWorkspace::new(
            "DFT",
            [
                ("real input", len),
                ("imaginary input", len),
                ("real output", len),
                ("imaginary output", len),
            ],
        )
    }

    fn requirement(&self) -> Result<onnx_runtime_ep_api::WorkspaceRequirement> {
        Ok(if self.dtype == DataType::Float64 {
            self.workspace::<f64>()?.requirement()
        } else {
            self.workspace::<f32>()?.requirement()
        })
    }
}

impl DftKernel {
    fn max_inputs(&self) -> usize {
        if self.opset >= 20 { 3 } else { 2 }
    }

    fn geometry(
        &self,
        inputs: &[TensorMetadata<'_>],
        views: Option<&[TensorView<'_>]>,
    ) -> Result<Option<DftGeometry>> {
        let max_inputs = self.max_inputs();
        if !(1..=max_inputs).contains(&inputs.len()) || !inputs[0].present {
            return Err(EpError::KernelFailed(format!(
                "DFT opset {} needs 1 to {max_inputs} input slots with present signal, got {}; use a scalar dft_length and supply an axis input only from opset 20",
                self.opset,
                inputs.len()
            )));
        }
        let input = &inputs[0];
        if input.shape.len() < 2 {
            return Err(EpError::KernelFailed(format!(
                "DFT: input must have rank >= 2, got {:?}",
                input.shape
            )));
        }
        if !super::FLOAT_DTYPES.contains(&input.dtype) {
            return Err(EpError::KernelFailed(format!(
                "DFT: signal must have floating dtype, got {:?}",
                input.dtype
            )));
        }
        let rank = input.shape.len();
        let last = rank - 1;
        self.mode
            .validate_components(input.shape[last])
            .map_err(EpError::KernelFailed)?;
        let present = |i: usize| inputs.get(i).is_some_and(|input| input.present);
        for (i, name) in [(1, "dft_length"), (2, "axis")] {
            if present(i) {
                let control = inputs[i];
                if !control.shape.is_empty() {
                    return Err(EpError::KernelFailed(format!(
                        "DFT: `{name}` must be a scalar, got shape {:?}",
                        control.shape
                    )));
                }
                if !(if i == 1 {
                    dft::is_length_dtype(control.dtype)
                } else {
                    control.dtype == DataType::Int64
                }) {
                    return Err(EpError::KernelFailed(format!(
                        "DFT: `{name}` dtype {:?} is invalid; use int32/int64 for dft_length and int64 for axis",
                        control.dtype
                    )));
                }
            }
        }
        if views.is_none() && (present(1) || present(2)) {
            return Ok(None);
        }
        let read_control = |i, name| {
            let view = views.and_then(|views| views.get(i)).ok_or_else(|| EpError::KernelFailed(format!("DFT: `{name}` metadata is present but its Host view is missing; bind the scalar before execution")))?;
            scalar_i64("DFT", name, view)
        };
        let axis = dft::normalize_axis(
            if present(2) {
                read_control(2, "axis")?
            } else {
                self.default_axis
            },
            rank,
        )
        .map_err(EpError::KernelFailed)?;
        let signal_len = input.shape[axis];
        let n = if present(1) {
            dft::explicit_length(read_control(1, "dft_length")?).map_err(EpError::KernelFailed)?
        } else {
            self.mode
                .default_length(signal_len)
                .map_err(EpError::KernelFailed)?
        };
        let batch_dims = || {
            input
                .shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis && *i != last)
                .map(|(_, &d)| d)
        };
        let batches = if batch_dims().any(|d| d == 0) {
            0
        } else {
            batch_dims()
                .try_fold(1usize, |count, d| count.checked_mul(d))
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "DFT: batch dimensions in {:?} overflow; reduce the input shape",
                        input.shape
                    ))
                })?
        };
        let output_len = self.mode.output_length(n);
        let output_components = self.mode.output_components();
        let output_elements = batches.checked_mul(output_len).and_then(|count| count.checked_mul(output_components)).ok_or_else(|| EpError::KernelFailed(format!("DFT: output for {batches} batches of length {n} overflows; reduce shape or dft_length")))?;
        let input_elements =
            onnx_runtime_ir::gemm::checked_elements(input.shape).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "DFT: input shape {:?} overflows element count; reduce the input shape",
                    input.shape
                ))
            })?;
        for (name, count) in [
            ("input materialization", input_elements),
            ("output staging", output_elements),
        ] {
            let layout = if input.dtype == DataType::Float64 {
                std::alloc::Layout::array::<f64>(count)
            } else {
                std::alloc::Layout::array::<f32>(count)
            };
            layout.map_err(|_| EpError::KernelFailed(format!("DFT: {name} of {count} elements for {:?} exceeds addressable capacity; reduce shape or dft_length", input.dtype)))?;
        }
        Ok(Some(DftGeometry {
            axis,
            signal_len,
            n,
            output_len,
            output_components,
            batches,
            output_elements,
            dtype: input.dtype,
        }))
    }

    fn live_geometry(&self, inputs: &[TensorView<'_>]) -> Result<DftGeometry> {
        let metadata = std::array::from_fn::<_, 3, _>(|i| {
            inputs.get(i).map_or(
                TensorMetadata::new(onnx_runtime_ir::DataType::Undefined, &[], false),
                |input| TensorMetadata::new(input.dtype, input.shape, !input.is_absent()),
            )
        });
        if inputs.len() > metadata.len() {
            return Err(EpError::KernelFailed(format!(
                "DFT opset {} accepts at most {} input slots, got {}",
                self.opset,
                self.max_inputs(),
                inputs.len()
            )));
        }
        self.geometry(&metadata[..inputs.len()], Some(inputs))?.ok_or_else(|| EpError::KernelFailed("DFT: runtime controls are unreadable; bind Host dft_length/axis scalars before execution".into()))
    }
}

impl Kernel for DftKernel {
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        match self.geometry(inputs, None)? {
            Some(geometry) => Ok(WorkspacePreparation::Ready(geometry.requirement()?)),
            None => Ok(WorkspacePreparation::Deferred),
        }
    }

    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        _outputs: &[TensorMut],
        metadata: &[TensorMetadata<'_>],
        _output_metadata: &[onnx_runtime_ep_api::OutputMetadata<'_>],
        _cached: Option<WorkspacePreparation>,
    ) -> Result<WorkspaceExecutionPreparation> {
        if inputs.len() != metadata.len() {
            return Err(EpError::KernelFailed("DFT: views and metadata have different slot counts; bind every input before workspace preparation".into()));
        }
        for (i, (view, meta)) in inputs.iter().zip(metadata).enumerate() {
            // Signal values are not needed for sizing. A logically present
            // intermediate may still lack backing; only controls must be read.
            if (!meta.present && !view.is_absent()) || (i > 0 && meta.present && view.is_absent()) {
                return Err(EpError::KernelFailed(format!(
                    "DFT: input {i} has inconsistent logical presence or an unreadable control; bind required Host scalars before workspace preparation"
                )));
            }
            if meta.present
                && !view.is_absent()
                && (meta.shape != view.shape || meta.dtype != view.dtype)
            {
                return Err(EpError::KernelFailed(format!(
                    "DFT: input {i} view differs from its metadata; prepare from the current shape and dtype"
                )));
            }
        }
        let geometry = self.geometry(metadata, Some(inputs))?.ok_or_else(|| {
            EpError::KernelFailed(
                "DFT: runtime controls remain unreadable; bind Host scalar inputs before admission"
                    .into(),
            )
        })?;
        Ok(WorkspaceExecutionPreparation::Ready(
            geometry.requirement()?,
        ))
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }
}

impl DftKernel {
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        super::check_arity("DFT", inputs, outputs, 1, self.max_inputs(), 1)?;
        let geometry = self.live_geometry(inputs)?;
        let input = &inputs[0];
        let output = &mut outputs[0];
        let rank = input.shape.len();
        let axis = geometry.axis;
        let last = rank - 1;
        let expected_extent = |i| {
            if i == axis {
                geometry.output_len
            } else if i == last {
                geometry.output_components
            } else {
                input.shape[i]
            }
        };
        if output.shape.len() != rank
            || output
                .shape
                .iter()
                .enumerate()
                .any(|(i, &d)| d != expected_extent(i))
        {
            let expected_shape = (0..rank).map(expected_extent).collect::<Vec<_>>();
            return Err(EpError::KernelFailed(format!(
                "DFT: mode {:?} output shape mismatch: expected {expected_shape:?}, got {:?}",
                self.mode, output.shape
            )));
        }
        validate_f32_widen_input("DFT", input)?;
        if !output.device.is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "DFT: output device {:?} is not Host accessible; bind a CPU output",
                output.device
            )));
        }
        if output.dtype != input.dtype {
            return Err(EpError::KernelFailed(format!(
                "DFT: output dtype must match input {:?}, got {:?}; bind an output with the signal dtype",
                input.dtype, output.dtype
            )));
        }
        if geometry.batches == 0 {
            return super::write_dense_bytes(output, &[]);
        }
        if input.dtype == DataType::Float64 {
            let inverse = self.mode.inverse();
            let result = self.transform_batches(
                &geometry,
                input,
                output,
                workspace,
                || {
                    let bytes = super::to_dense_bytes(input)?;
                    Ok(bytes
                        .chunks_exact(8)
                        .map(|b| f64::from_ne_bytes(b.try_into().expect("8-byte chunk")))
                        .collect::<Vec<_>>())
                },
                |real, imag, real_out, imag_out| {
                    if real.len().is_power_of_two() && real.len() > 1 {
                        real_out.copy_from_slice(real);
                        imag_out.copy_from_slice(imag);
                        fft_radix2_in_place(real_out, imag_out, inverse);
                    } else {
                        naive_dft_into(real, imag, real_out, imag_out, inverse);
                    }
                },
            )?;
            let bytes: Vec<_> = result.into_iter().flat_map(f64::to_ne_bytes).collect();
            super::write_dense_bytes(output, &bytes)
        } else {
            let mut dft = None;
            let result = self.transform_batches(
                &geometry,
                input,
                output,
                workspace,
                || to_dense_f32_widen("DFT", input),
                |r, i, ro, io| {
                    dft.get_or_insert_with(|| DftPlan::new(geometry.n, self.mode.inverse()))
                        .transform(r, i, ro, io)
                },
            )?;
            write_dense_f32_narrow("DFT", output, &result)
        }
    }

    fn transform_batches<T: Float, D: AsRef<[T]>>(
        &self,
        geometry: &DftGeometry,
        input: &TensorView<'_>,
        output: &TensorMut<'_>,
        workspace: Option<WorkspaceView>,
        read: impl FnOnce() -> Result<D>,
        mut transform: impl FnMut(&[T], &[T], &mut [T], &mut [T]),
    ) -> Result<Vec<T>> {
        let axis = geometry.axis;
        let last = input.shape.len() - 1;
        let input_strides = row_major_strides(input.shape)?;
        let output_strides = row_major_strides(output.shape)?;
        let batch_shape = input
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis && *i != last)
            .map(|(_, &d)| d)
            .collect::<Vec<_>>();
        let plan = geometry.workspace::<T>()?;
        // SAFETY: CPU execution owns exclusive live Host backing until this
        // synchronous closure returns. The shared helper checks/initializes
        // four disjoint typed regions before any references are formed.
        unsafe {
            plan.with_buffers(workspace, |[real_in, imag_in, real_out, imag_out]| {
                let data = read()?;
                let input_data = data.as_ref();
                let mut output_data = vec![T::zero(); geometry.output_elements];
                let mut batch_indices = vec![0usize; batch_shape.len()];
                let copied = geometry.signal_len.min(self.mode.input_length(geometry.n));
                for _ in 0..geometry.batches {
                    let (in_base, out_base) = compute_bases(
                        &batch_indices,
                        input.shape,
                        output.shape,
                        axis,
                        last,
                        &input_strides,
                        &output_strides,
                    );
                    for i in 0..copied {
                        let idx = in_base + i * input_strides[axis];
                        real_in[i] = input_data[idx];
                        if input.shape[last] == 2 {
                            imag_in[i] = input_data[idx + 1];
                        }
                    }
                    // The plan never mutates its input. Positive-frequency padding
                    // stays zero between batches; mirrors touch only negative bins.
                    if self.mode == DftMode::RealInverse {
                        imag_in[0] = T::zero();
                        if geometry.n.is_multiple_of(2) {
                            imag_in[geometry.n / 2] = T::zero();
                        }
                        for i in 1..geometry.n.div_ceil(2) {
                            real_in[geometry.n - i] = real_in[i];
                            imag_in[geometry.n - i] = -imag_in[i];
                        }
                    }
                    transform(real_in, imag_in, real_out, imag_out);
                    for i in 0..geometry.output_len {
                        let idx = out_base + i * output_strides[axis];
                        output_data[idx] = real_out[i];
                        if geometry.output_components == 2 {
                            output_data[idx + 1] = imag_out[i];
                        }
                    }
                    advance_indices(&mut batch_indices, &batch_shape);
                }
                Ok(output_data)
            })
        }
    }
}

fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].checked_mul(shape[i + 1]).ok_or_else(|| EpError::KernelFailed(format!("DFT: contiguous strides for shape {shape:?} overflow; reduce the input/output shape")))?;
    }
    Ok(strides)
}

fn compute_bases(
    batch_indices: &[usize],
    input_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
    last: usize,
    input_strides: &[usize],
    output_strides: &[usize],
) -> (usize, usize) {
    let _ = (input_shape, output_shape);
    let mut in_base = 0usize;
    let mut out_base = 0usize;
    let mut bi = 0;
    for d in 0..input_strides.len() {
        if d == axis || d == last {
            continue;
        }
        in_base += batch_indices[bi] * input_strides[d];
        out_base += batch_indices[bi] * output_strides[d];
        bi += 1;
    }
    (in_base, out_base)
}

fn advance_indices(indices: &mut [usize], shape: &[usize]) {
    for i in (0..indices.len()).rev() {
        indices[i] += 1;
        if indices[i] < shape[i] {
            return;
        }
        indices[i] = 0;
    }
}

/// Owned convenience for transform-level numerical/dispatch tests.
#[cfg(test)]
fn compute_dft(
    real_in: &[f32],
    imag_in: &[f32],
    n: usize,
    inverse: bool,
    onesided: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut real_out = vec![0.0; n];
    let mut imag_out = vec![0.0; n];
    DftPlan::new(n, inverse).transform(real_in, imag_in, &mut real_out, &mut imag_out);
    if onesided {
        let half = n / 2 + 1;
        real_out.truncate(half);
        imag_out.truncate(half);
    }
    (real_out, imag_out)
}

#[cfg(test)]
thread_local! {
    static LOCAL_PLAN_CREATES: Cell<usize> = const { Cell::new(0) };
}

/// A transform plan reusable across batches or STFT frames of one fixed length.
pub(super) struct DftPlan {
    n: usize,
    inverse: bool,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    vdsp_setup: VdspDftSetup,
}

impl DftPlan {
    pub(super) fn new(n: usize, inverse: bool) -> Self {
        #[cfg(test)]
        LOCAL_PLAN_CREATES.with(|count| count.set(count.get() + 1));
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let vdsp_setup = if n.is_power_of_two() && n >= 4 {
            let direction = if inverse {
                VDSP_DFT_INVERSE
            } else {
                VDSP_DFT_FORWARD
            };
            unsafe { vDSP_DFT_zop_CreateSetup(std::ptr::null(), n as u64, direction) }
        } else {
            std::ptr::null()
        };
        Self {
            n,
            inverse,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            vdsp_setup,
        }
    }

    pub(super) fn transform(
        &self,
        real_in: &[f32],
        imag_in: &[f32],
        real_out: &mut [f32],
        imag_out: &mut [f32],
    ) {
        debug_assert_eq!(real_in.len(), self.n);
        debug_assert_eq!(imag_in.len(), self.n);
        debug_assert_eq!(real_out.len(), self.n);
        debug_assert_eq!(imag_out.len(), self.n);

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if !self.vdsp_setup.is_null() {
            unsafe {
                vDSP_DFT_Execute(
                    self.vdsp_setup,
                    real_in.as_ptr(),
                    imag_in.as_ptr(),
                    real_out.as_mut_ptr(),
                    imag_out.as_mut_ptr(),
                );
            }
            if self.inverse {
                let scale = 1.0 / self.n as f32;
                for value in real_out.iter_mut().chain(imag_out.iter_mut()) {
                    *value *= scale;
                }
            }
            DFT_VDSP_TEST_HITS.fetch_add(1, Ordering::Relaxed);
            record_fast_path_hit();
            return;
        }

        if self.n.is_power_of_two() && self.n > 1 {
            DFT_FFT_TEST_HITS.fetch_add(1, Ordering::Relaxed);
            record_fast_path_hit();
            real_out.copy_from_slice(real_in);
            imag_out.copy_from_slice(imag_in);
            fft_radix2_in_place(real_out, imag_out, self.inverse);
        } else {
            naive_dft_into(real_in, imag_in, real_out, imag_out, self.inverse);
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl Drop for DftPlan {
    fn drop(&mut self) {
        if !self.vdsp_setup.is_null() {
            unsafe {
                vDSP_DFT_DestroySetup(self.vdsp_setup);
            }
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
const VDSP_DFT_FORWARD: i32 = 1;
#[cfg(any(target_os = "macos", target_os = "ios"))]
const VDSP_DFT_INVERSE: i32 = -1;

#[cfg(any(target_os = "macos", target_os = "ios"))]
type VdspDftSetup = *const std::ffi::c_void;

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vDSP_DFT_zop_CreateSetup(
        previous: VdspDftSetup,
        length: u64,
        direction: i32,
    ) -> VdspDftSetup;

    fn vDSP_DFT_Execute(
        setup: VdspDftSetup,
        ir: *const f32,
        ii: *const f32,
        or_: *mut f32,
        oi: *mut f32,
    );

    fn vDSP_DFT_DestroySetup(setup: VdspDftSetup);
}

/// Cooley–Tukey radix-2 FFT for power-of-two lengths.
#[cfg(test)]
fn fft_radix2(
    real_in: &[f32],
    imag_in: &[f32],
    n: usize,
    inverse: bool,
    onesided: bool,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert!(n.is_power_of_two() && n > 1);

    let mut real = real_in.to_vec();
    let mut imag = imag_in.to_vec();
    fft_radix2_in_place(&mut real, &mut imag, inverse);

    if onesided {
        let half = n / 2 + 1;
        real.truncate(half);
        imag.truncate(half);
    }

    (real, imag)
}

fn fft_radix2_in_place<T: Float + FloatConst>(real: &mut [T], imag: &mut [T], inverse: bool) {
    let n = real.len();
    debug_assert!(n.is_power_of_two() && n > 1);
    debug_assert_eq!(imag.len(), n);

    // Bit-reversal permutation.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }

    // Butterfly passes.
    let turn = if inverse { T::TAU() } else { -T::TAU() };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = turn / T::from(len).unwrap();
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let phase = angle * T::from(k).unwrap();
                let w_re = phase.cos();
                let w_im = phase.sin();
                let i1 = start + k;
                let i2 = start + k + half;
                let tr = real[i2] * w_re - imag[i2] * w_im;
                let ti = real[i2] * w_im + imag[i2] * w_re;
                real[i2] = real[i1] - tr;
                imag[i2] = imag[i1] - ti;
                real[i1] = real[i1] + tr;
                imag[i1] = imag[i1] + ti;
            }
        }
        len <<= 1;
    }

    if inverse {
        let scale = T::one() / T::from(n).unwrap();
        for x in real.iter_mut() {
            *x = *x * scale;
        }
        for x in imag.iter_mut() {
            *x = *x * scale;
        }
    }
}

/// Naive O(N²) DFT for arbitrary lengths.
#[cfg(test)]
fn naive_dft(
    real_in: &[f32],
    imag_in: &[f32],
    n: usize,
    inverse: bool,
    onesided: bool,
) -> (Vec<f32>, Vec<f32>) {
    let out_len = if onesided { n / 2 + 1 } else { n };
    let mut real_out = vec![0.0f32; out_len];
    let mut imag_out = vec![0.0f32; out_len];
    naive_dft_into(real_in, imag_in, &mut real_out, &mut imag_out, inverse);
    (real_out, imag_out)
}

fn naive_dft_into<T: Float + FloatConst>(
    real_in: &[T],
    imag_in: &[T],
    real_out: &mut [T],
    imag_out: &mut [T],
    inverse: bool,
) {
    let n = real_in.len();
    debug_assert_eq!(imag_in.len(), n);
    debug_assert_eq!(real_out.len(), imag_out.len());
    debug_assert!(real_out.len() <= n);
    let turn = if inverse { T::TAU() } else { -T::TAU() };
    let length = T::from(n).unwrap();
    for k in 0..real_out.len() {
        let mut sum_re = T::zero();
        let mut sum_im = T::zero();
        for t in 0..n {
            let angle = turn * T::from(k).unwrap() * T::from(t).unwrap() / length;
            let w_re = angle.cos();
            let w_im = angle.sin();
            sum_re = sum_re + (real_in[t] * w_re - imag_in[t] * w_im);
            sum_im = sum_im + (real_in[t] * w_im + imag_in[t] * w_re);
        }
        if inverse {
            let scale = T::one() / length;
            sum_re = sum_re * scale;
            sum_im = sum_im * scale;
        }
        real_out[k] = sum_re;
        imag_out[k] = sum_im;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::{Owned, PreparedTestKernel};
    use onnx_runtime_ir::{Attribute, Node, NodeId};

    fn make_dft_node(axis: i64, onesided: i64) -> Node {
        let mut node = Node::new(NodeId(0), "DFT", vec![], vec![]);
        node.attributes.insert("axis".into(), Attribute::Int(axis));
        node.attributes
            .insert("onesided".into(), Attribute::Int(onesided));
        node
    }

    fn versioned_dft_node(version: i64, axis: Option<i64>) -> Node {
        let mut node = Node::new(NodeId(0), "DFT", vec![], vec![]);
        node.version = Some(version);
        node.attributes.insert("onesided".into(), Attribute::Int(1));
        if let Some(axis) = axis {
            node.attributes.insert("axis".into(), Attribute::Int(axis));
        }
        node
    }

    #[test]
    fn versioned_default_axis_and_explicit_overrides_are_discriminating() {
        let input = Owned::f32(&[1, 8, 6, 1], &[0.0; 48]);

        let mut opset17_output = Owned::zeros_f32(&[1, 5, 6, 2]);
        DftFactory
            .create(&versioned_dft_node(17, None), &[vec![1, 8, 6, 1]])
            .unwrap()
            .execute_prepared(&[input.view()], &mut [opset17_output.view_mut()])
            .unwrap();

        let mut opset20_output = Owned::zeros_f32(&[1, 8, 4, 2]);
        DftFactory
            .create(&versioned_dft_node(20, None), &[vec![1, 8, 6, 1]])
            .unwrap()
            .execute_prepared(&[input.view()], &mut [opset20_output.view_mut()])
            .unwrap();

        let mut attr_output = Owned::zeros_f32(&[1, 8, 4, 2]);
        DftFactory
            .create(&versioned_dft_node(17, Some(2)), &[vec![1, 8, 6, 1]])
            .unwrap()
            .execute_prepared(&[input.view()], &mut [attr_output.view_mut()])
            .unwrap();

        let axis = Owned::i64(&[], &[1]);
        let mut input_axis_output = Owned::zeros_f32(&[1, 5, 6, 2]);
        DftFactory
            .create(&versioned_dft_node(20, None), &[vec![1, 8, 6, 1]])
            .unwrap()
            .execute_prepared(
                &[
                    input.view(),
                    TensorView::absent(onnx_runtime_ir::DataType::Int64),
                    axis.view(),
                ],
                &mut [input_axis_output.view_mut()],
            )
            .unwrap();
    }

    fn irfft_node(version: i64) -> Node {
        let mut n = make_dft_node(-2, 1);
        n.version = Some(version);
        n.attributes.insert("inverse".into(), Attribute::Int(1));
        n
    }

    #[test]
    fn real_inverse_defaults_to_full_real_signal_and_ignores_endpoint_imaginary_parts() {
        for version in [17, 20] {
            let input = Owned::f32(&[1, 3, 2], &[10., 99., -2., 2., -2., -77.]);
            let mut output = Owned::zeros_f32(&[1, 4, 1]);
            DftFactory
                .create(&irfft_node(version), &[vec![1, 3, 2]])
                .unwrap()
                .execute_prepared(&[input.view()], &mut [output.view_mut()])
                .unwrap();
            for (actual, expected) in output.to_f32().iter().zip([1., 2., 3., 4.]) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "{version}: {actual} versus {expected}"
                );
            }
        }
    }

    #[test]
    fn real_inverse_even_odd_padded_and_truncated_spectra_match_independent_f64_sums() {
        for n in [1usize, 2, 3, 4, 5, 6, 9, 10] {
            for bins in [0usize, 1, 3, 7] {
                let values = (0..2)
                    .flat_map(|batch| {
                        (0..bins).flat_map(move |k| {
                            [
                                (batch + 1) as f32 * 0.375 + k as f32 * 0.25,
                                0.125 - k as f32 * 0.125,
                            ]
                        })
                    })
                    .collect::<Vec<_>>();
                let input = Owned::f32(&[2, bins, 2], &values);
                let length = Owned::i64(&[], &[n as i64]);
                let mut output = Owned::zeros_f32(&[2, n, 1]);
                DftFactory
                    .create(&irfft_node(20), &[vec![2, bins, 2]])
                    .unwrap()
                    .execute_prepared(&[input.view(), length.view()], &mut [output.view_mut()])
                    .unwrap();
                let actual = output.to_f32();
                for batch in 0..2 {
                    for t in 0..n {
                        let mut expected = 0f64;
                        for k in 0..bins.min(n / 2 + 1) {
                            let re = values[(batch * bins + k) * 2] as f64;
                            let im = values[(batch * bins + k) * 2 + 1] as f64;
                            let angle = 2. * std::f64::consts::PI * (k * t) as f64 / n as f64;
                            expected += if k == 0 || (n.is_multiple_of(2) && k == n / 2) {
                                re * angle.cos()
                            } else {
                                2. * (re * angle.cos() - im * angle.sin())
                            };
                        }
                        expected /= n as f64;
                        assert!(
                            (actual[batch * n + t] as f64 - expected).abs() < 1e-4,
                            "n={n}, bins={bins}, batch={batch}, t={t}: {} versus {expected}",
                            actual[batch * n + t]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn real_inverse_rejects_real_spectrum_and_wrong_output_geometry_before_writing() {
        let n = irfft_node(20);
        let shapes = [onnx_runtime_ir::static_shape([1, 3, 1])];
        assert!(
            DftFactory
                .unsupported_reason(&n, 20, &shapes, &[])
                .unwrap()
                .contains("RealInverse")
        );
        assert!(DftFactory.create(&n, &[vec![1, 3, 1]]).is_err());
        let input = Owned::f32(&[1, 3, 2], &[1.; 6]);
        let kernel = DftFactory.create(&n, &[]).unwrap();
        for shape in [vec![1], vec![2, 4, 1], vec![1, 3, 2]] {
            let count = shape.iter().product();
            let mut output = Owned::f32(&shape, &vec![123.; count]);
            let error = kernel
                .execute_prepared(&[input.view()], &mut [output.view_mut()])
                .unwrap_err();
            assert!(
                error.to_string().contains("output shape mismatch"),
                "{error}"
            );
            assert!(output.to_f32().iter().all(|&v| v == 123.));
        }
    }

    #[test]
    fn dft_real_inverse_preserves_float64_precision_in_both_opsets() {
        let delta = 2f64.powi(-40);
        let spectrum = Owned::f64(
            &[2, 2, 2],
            &[1.0 + delta, 9.0, 1.0, 11.0, 1.0, 7.0, 1.0 + delta, 13.0],
        );
        let inputs = [spectrum.view()];
        let expected = Owned::f64(
            &[2, 2, 1],
            &[
                1.0 + delta / 2.0,
                delta / 2.0,
                1.0 + delta / 2.0,
                -delta / 2.0,
            ],
        );
        for opset in [17, 20] {
            let kernel = DftFactory.create(&irfft_node(opset), &[]).unwrap();
            let requirement = kernel
                .workspace_requirement(&metadata(&inputs), &[])
                .unwrap();
            let WorkspacePreparation::Ready(requirement) = requirement else {
                panic!("implicit controls should give a static requirement for opset {opset}");
            };
            assert_eq!(
                requirement.bytes, 64,
                "four f64 buffers for one two-point transform, not two batches"
            );
            let mut output = Owned::f64(&[2, 2, 1], &[0.; 4]);
            assert!(kernel.execute(&inputs, &mut [output.view_mut()]).is_err());
            kernel
                .execute_prepared(&inputs, &mut [output.view_mut()])
                .unwrap();
            // NumPy irfft ignores the imaginary components of DC and even Nyquist.
            assert_eq!(output.bytes, expected.bytes, "opset {opset}");
        }
    }

    #[test]
    fn dft_v20_real_inverse_accepts_rank_zero_int32_length_on_nondefault_axis() {
        let spectrum = Owned::f32(&[2, 2, 2], &[1., 9., 2., -8., 3., 4., -2., 1.]);
        let length = Owned::i32(&[], &[5]);
        let axis = Owned::i64(&[], &[-3]);
        let kernel = DftFactory.create(&irfft_node(20), &[]).unwrap();
        let mut output = Owned::zeros_f32(&[5, 2, 1]);
        kernel
            .execute_prepared(
                &[spectrum.view(), length.view(), axis.view()],
                &mut [output.view_mut()],
            )
            .unwrap();
        // NumPy irfft([[1+9j, 2-8j], [3+4j, -2+1j]], n=5, axis=0).
        let expected = [
            1.4,
            -0.4,
            -0.95087004,
            -0.2276362,
            -1.7112768,
            0.8120995,
            0.16963601,
            1.2823277,
            2.0925107,
            0.533209,
        ];
        for (i, (actual, expected)) in output.to_f32().into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2e-6,
                "index {i}: expected {expected}, got {actual}"
            );
        }

        let spectrum = Owned::f64(&[2, 2, 2], &[1., 9., 2., -8., 3., 4., -2., 1.]);
        let mut output = Owned::f64(&[5, 2, 1], &[0.; 10]);
        kernel
            .execute_prepared(
                &[spectrum.view(), length.view(), axis.view()],
                &mut [output.view_mut()],
            )
            .unwrap();
        for (i, (bytes, expected)) in output.bytes.chunks_exact(8).zip(expected).enumerate() {
            let actual = f64::from_ne_bytes(bytes.try_into().unwrap());
            assert!(
                (actual - f64::from(expected)).abs() < 2e-6,
                "index {i}: expected {expected}, got {actual}"
            );
        }
    }

    fn metadata<'a>(inputs: &[TensorView<'a>]) -> Vec<TensorMetadata<'a>> {
        inputs
            .iter()
            .map(|input| TensorMetadata::new(input.dtype, input.shape, !input.is_absent()))
            .collect()
    }

    #[test]
    fn batches_share_one_fourier_plan_and_one_frame_scratch() {
        let kernel = DftFactory.create(&make_dft_node(-2, 1), &[]).unwrap();
        let input = Owned::f32(
            &[3, 4, 1],
            &[1., 2., 3., 4., 10., 20., 30., 40., -1., -2., -3., -4.],
        );
        let inputs = [input.view()];
        let request = kernel
            .workspace_requirement(&metadata(&inputs), &[])
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(
            request.bytes, 64,
            "four transform-length buffers, not batch times that size"
        );
        let before = LOCAL_PLAN_CREATES.with(Cell::get);
        let _control = DftPlan::new(2, false);
        assert_eq!(
            LOCAL_PLAN_CREATES.with(Cell::get),
            before + 1,
            "constructor positive control"
        );
        let mut output = Owned::zeros_f32(&[3, 3, 2]);
        kernel
            .execute_prepared(&inputs, &mut [output.view_mut()])
            .unwrap();
        assert_eq!(
            LOCAL_PLAN_CREATES.with(Cell::get),
            before + 2,
            "one plan for all three transforms"
        );
        for (frame, scale) in output.to_f32().chunks_exact(6).zip([1., 10., -1.]) {
            for (actual, expected) in frame.iter().zip([10., 0., -2., 2., -2., 0.]) {
                assert!((actual - scale * expected).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn live_scalar_controls_refine_deferred_metadata_and_reject_stale_backing() {
        let kernel = DftFactory.create(&make_dft_node(-2, 0), &[]).unwrap();
        let signal = Owned::f32(&[2, 7, 1], &[1.; 14]);
        let mut length = Owned::i64(&[], &[3]);
        let mut backing = [0x7fc0_1234u32; 38];
        let workspace = WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(unsafe { backing.as_mut_ptr().add(1).cast() }),
            144,
        );
        let mut old = None;
        for n in [3usize, 9, 3] {
            length.bytes.copy_from_slice(&(n as i64).to_le_bytes());
            let inputs = [signal.view(), length.view()];
            let meta = metadata(&inputs);
            assert_eq!(
                kernel.workspace_requirement(&meta, &[]).unwrap(),
                WorkspacePreparation::Deferred
            );
            let request = kernel
                .workspace_requirement_for_execution(
                    &inputs,
                    &[],
                    &meta,
                    &[],
                    Some(WorkspacePreparation::Ready(
                        onnx_runtime_ep_api::WorkspaceRequirement::NONE,
                    )),
                )
                .unwrap()
                .require_ready()
                .unwrap();
            assert_eq!(request.bytes as usize, 16 * n);
            let mut output = Owned::f32(&[2, n, 2], &vec![123.; 4 * n]);
            let calls = LOCAL_PLAN_CREATES.with(Cell::get);
            if n == 9 {
                let error = kernel
                    .execute_with_workspace(&inputs, &mut [output.view_mut()], old)
                    .unwrap_err();
                assert!(error.to_string().contains("workspace needs"), "{error}");
                assert!(output.to_f32().iter().all(|&value| value == 123.));
                assert_eq!(
                    LOCAL_PLAN_CREATES.with(Cell::get),
                    calls,
                    "no DSP plan before capacity validation"
                );
            }
            let current = WorkspaceView::new(workspace.ptr(), request.bytes as usize);
            kernel
                .execute_with_workspace(&inputs, &mut [output.view_mut()], Some(current))
                .unwrap();
            let actual = output.to_f32();
            for batch in 0..2 {
                for k in 0..n {
                    let mut re = 0f64;
                    let mut im = 0f64;
                    for t in 0..7.min(n) {
                        let angle = -2. * std::f64::consts::PI * (k * t) as f64 / n as f64;
                        re += angle.cos();
                        im += angle.sin();
                    }
                    assert!((actual[(batch * n + k) * 2] as f64 - re).abs() < 1e-4);
                    assert!((actual[(batch * n + k) * 2 + 1] as f64 - im).abs() < 1e-4);
                }
            }
            assert_eq!(backing[0], 0x7fc0_1234);
            assert_eq!(backing[37], 0x7fc0_1234);
            old = Some(current);
            backing[1..37].fill(0x7fc0_1234);
        }
    }

    #[test]
    fn axis_changes_at_one_address_and_unreadable_logical_controls_are_not_cached() {
        let mut node = make_dft_node(-2, 0);
        node.version = Some(20);
        let kernel = DftFactory.create(&node, &[]).unwrap();
        let signal = Owned::f32(&[2, 4, 1], &[1., 2., 3., 4., 10., 20., 30., 40.]);
        let mut axis = Owned::i64(&[], &[1]);
        let mut output = Owned::zeros_f32(&[2, 4, 2]);
        for (value, bytes, dc) in [(1i64, 64, 10.), (0, 32, 11.), (1, 64, 10.)] {
            axis.bytes.copy_from_slice(&value.to_le_bytes());
            let inputs = [
                signal.view(),
                TensorView::absent(onnx_runtime_ir::DataType::Int64),
                axis.view(),
            ];
            let meta = metadata(&inputs);
            assert_eq!(
                kernel.workspace_requirement(&meta, &[]).unwrap(),
                WorkspacePreparation::Deferred
            );
            let request = kernel
                .workspace_requirement_for_execution(&inputs, &[], &meta, &[], None)
                .unwrap()
                .require_ready()
                .unwrap();
            assert_eq!(request.bytes, bytes);
            kernel
                .execute_prepared(&inputs, &mut [output.view_mut()])
                .unwrap();
            assert_eq!(output.to_f32()[0], dc);
            let missing = [
                signal.view(),
                inputs[1],
                TensorView::absent(onnx_runtime_ir::DataType::Int64),
            ];
            assert!(
                kernel
                    .workspace_requirement_for_execution(&missing, &[], &meta, &[], None)
                    .unwrap_err()
                    .to_string()
                    .contains("logical presence")
            );
        }
    }

    #[test]
    fn metadata_only_signal_can_prepare_without_reading_unmaterialized_storage() {
        let kernel = DftFactory.create(&make_dft_node(-2, 0), &[]).unwrap();
        let length = Owned::i64(&[], &[5]);
        let inputs = [
            TensorView::absent(onnx_runtime_ir::DataType::Float32),
            length.view(),
        ];
        let meta = [
            TensorMetadata::new(onnx_runtime_ir::DataType::Float32, &[2, 4, 1], true),
            TensorMetadata::new(onnx_runtime_ir::DataType::Int64, &[], true),
        ];
        let request = kernel
            .workspace_requirement_for_execution(&inputs, &[], &meta, &[], None)
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(request.bytes, 80);
        let missing_control = [
            inputs[0],
            TensorView::absent(onnx_runtime_ir::DataType::Int64),
        ];
        assert!(
            kernel
                .workspace_requirement_for_execution(&missing_control, &[], &meta, &[], None)
                .is_err()
        );
        let mut output = Owned::f32(&[2, 5, 2], &[123.; 20]);
        assert!(kernel.execute(&inputs, &mut [output.view_mut()]).is_err());
        assert_eq!(output.to_f32(), [123.; 20]);
    }

    #[test]
    fn invalid_workspace_and_empty_batches_keep_output_and_view_guards() {
        let kernel = DftFactory.create(&make_dft_node(-2, 0), &[]).unwrap();
        let signal = Owned::f32(&[1, 4, 1], &[1., 2., 3., 4.]);
        let mut output = Owned::f32(&[1, 4, 2], &[123.; 8]);
        let mut backing = [0u32; 17];
        for workspace in [
            None,
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(backing.as_mut_ptr().cast()),
                63,
            )),
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(unsafe {
                    backing.as_mut_ptr().cast::<u8>().add(1).cast()
                }),
                64,
            )),
        ] {
            let error = kernel
                .execute_with_workspace(&[signal.view()], &mut [output.view_mut()], workspace)
                .unwrap_err();
            assert!(error.to_string().contains("workspace needs"), "{error}");
            assert_eq!(output.to_f32(), [123.; 8]);
        }
        let empty = Owned::f32(&[0, 4, 1], &[]);
        let mut output = Owned::zeros_f32(&[0, 4, 2]);
        let before = LOCAL_PLAN_CREATES.with(Cell::get);
        kernel
            .execute(&[empty.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(LOCAL_PLAN_CREATES.with(Cell::get), before);
        let mut invalid = empty.view();
        invalid.device = onnx_runtime_ir::DeviceId::cuda(0);
        assert!(
            kernel
                .execute(&[invalid], &mut [output.view_mut()])
                .unwrap_err()
                .to_string()
                .contains("explicitly transfer")
        );
        for shape in [&[0usize][..], &[1][..]] {
            let bad = Owned::i64(shape, &vec![1; shape[0]]);
            assert!(
                kernel
                    .execute(&[empty.view(), bad.view()], &mut [output.view_mut()])
                    .unwrap_err()
                    .to_string()
                    .contains("must be a scalar")
            );
        }
        assert!(row_major_strides(&[1, usize::MAX, 2]).is_err());
        assert!(
            kernel
                .workspace_requirement(
                    &[TensorMetadata::new(
                        onnx_runtime_ir::DataType::Float32,
                        &[usize::MAX, 4, 1],
                        true
                    )],
                    &[]
                )
                .is_err()
        );
    }

    #[test]
    fn governed_fourier_buffers_reuse_one_lane_without_warm_issuer_callbacks() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{
            HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
        };
        use std::sync::Arc;
        let books = Arc::new(RecordingGovernor::new_host(64));
        let manager = ProcessMemoryManager::new().unwrap();
        let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
        let resources = crate::CpuMemoryResources::new(
            books.weight_memory(HolderId::new(13)),
            Default::default(),
        )
        .unwrap()
        .with_workspace_pool(pool.clone())
        .unwrap();
        let provider = crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
        let kernel = DftFactory.create(&make_dft_node(-2, 0), &[]).unwrap();
        let signal = Owned::f32(&[2, 4, 1], &[1., 2., 3., 4., 2., 4., 6., 8.]);
        let inputs = [signal.view()];
        let meta = metadata(&inputs);
        let mut output = Owned::zeros_f32(&[2, 4, 2]);
        let request = kernel
            .workspace_requirement_for_execution(&inputs, &[], &meta, &[], None)
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(request.bytes, 64);
        let first = provider
            .allocate_workspace(64, request.alignment, request.role())
            .unwrap();
        let identity = first.managed_owner().unwrap().identity();
        assert!(
            provider
                .allocate_workspace(64, request.alignment, request.role())
                .is_err(),
            "an active lane cannot be reused by a concurrent call"
        );
        provider.deallocate_workspace(first).unwrap();
        let before = (books.claims(), books.releases());
        drop(
            books
                .reserve(Tier::Host, 0, request.role(), HolderId::new(99))
                .unwrap(),
        );
        books.probe_zero_release();
        assert_eq!(
            (books.claims(), books.releases()),
            (before.0 + 1, before.1 + 1),
            "zero-byte observer positive controls"
        );
        let calls = (books.claims(), books.releases());
        for _ in 0..4096 {
            let request = kernel
                .workspace_requirement_for_execution(&inputs, &[], &meta, &[], None)
                .unwrap()
                .require_ready()
                .unwrap();
            let mut allocation = provider
                .allocate_workspace(request.bytes as usize, request.alignment, request.role())
                .unwrap();
            assert_eq!(allocation.managed_owner().unwrap().identity(), identity);
            let workspace = WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(allocation.as_mut_ptr()),
                allocation.len(),
            );
            kernel
                .execute_with_workspace(&inputs, &mut [output.view_mut()], Some(workspace))
                .unwrap();
            provider.deallocate_workspace(allocation).unwrap();
        }
        for (frame, scale) in output.to_f32().chunks_exact(8).zip([1., 2.]) {
            for (actual, expected) in frame.iter().zip([10., 0., -2., 2., -2., 0., -2., -2.]) {
                assert!((actual - scale * expected).abs() < 1e-4);
            }
        }
        assert_eq!(
            (books.claims(), books.releases()),
            calls,
            "including zero-byte callbacks"
        );
        assert_eq!(books.used(Tier::Host), 64);
        assert_eq!(manager.process_used(Tier::Host), 64);
        pool.trim_idle(64).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }

    /// Verify the radix-2 FFT matches the naive DFT within tolerance.
    #[test]
    fn fft_matches_naive_forward() {
        let n = 16;
        let real: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let imag = vec![0.0f32; n];

        let (r_naive, i_naive) = naive_dft(&real, &imag, n, false, false);
        let (r_fft, i_fft) = fft_radix2(&real, &imag, n, false, false);

        for k in 0..n {
            assert!(
                (r_naive[k] - r_fft[k]).abs() < 1e-4,
                "real[{k}]: naive={} fft={}",
                r_naive[k],
                r_fft[k]
            );
            assert!(
                (i_naive[k] - i_fft[k]).abs() < 1e-4,
                "imag[{k}]: naive={} fft={}",
                i_naive[k],
                i_fft[k]
            );
        }
    }

    /// Verify onesided produces N/2+1 bins.
    #[test]
    fn fft_onesided() {
        let n = 8;
        let real: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let imag = vec![0.0f32; n];

        let (r, i) = fft_radix2(&real, &imag, n, false, true);
        assert_eq!(r.len(), 5); // 8/2 + 1
        assert_eq!(i.len(), 5);
    }

    /// Verify inverse DFT recovers the original signal.
    #[test]
    fn fft_inverse_roundtrip() {
        let n = 32;
        let real: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).cos()).collect();
        let imag = vec![0.0f32; n];

        let (r_fwd, i_fwd) = fft_radix2(&real, &imag, n, false, false);
        let (r_inv, i_inv) = fft_radix2(&r_fwd, &i_fwd, n, true, false);

        for k in 0..n {
            assert!(
                (real[k] - r_inv[k]).abs() < 1e-4,
                "roundtrip real[{k}]: orig={} recovered={}",
                real[k],
                r_inv[k]
            );
            assert!(
                imag[k].abs() < 1e-4 && i_inv[k].abs() < 1e-4,
                "roundtrip imag[{k}]"
            );
        }
    }

    /// Verify the vDSP path fires on macOS and matches a double-precision reference.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn vdsp_matches_fft() {
        let n = 1024;
        let real: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin()).collect();
        let imag = vec![0.0f32; n];

        let before = DFT_VDSP_TEST_HITS.load(Ordering::Relaxed);
        let (r_vdsp, i_vdsp) = compute_dft(&real, &imag, n, false, true);
        let after = DFT_VDSP_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "vDSP path was not taken for N=1024 forward onesided DFT"
        );

        // Compare against double-precision naive DFT for ground truth.
        let out_len = n / 2 + 1;
        let mut r_ref = vec![0.0f64; out_len];
        let mut i_ref = vec![0.0f64; out_len];
        for k in 0..out_len {
            let mut sr = 0.0f64;
            let mut si = 0.0f64;
            for (t, &r) in real.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / (n as f64);
                sr += r as f64 * angle.cos();
                si += r as f64 * angle.sin();
            }
            r_ref[k] = sr;
            i_ref[k] = si;
        }

        let max_err: f32 = r_vdsp
            .iter()
            .zip(&r_ref)
            .chain(i_vdsp.iter().zip(&i_ref))
            .map(|(a, b)| (*a as f64 - b).abs() as f32)
            .fold(0.0f32, f32::max);
        // vDSP f32 accumulation against f64 reference: expect < 1e-3 absolute error.
        assert!(
            max_err < 1e-2,
            "vDSP vs f64-reference max error = {max_err} (threshold 1e-2)"
        );
    }

    /// Kernel-level test: real input, onesided, axis=-2.
    #[test]
    fn dft_kernel_real_onesided() {
        // Shape: [1, 8, 1] — batch=1, signal_len=8, complex_dim=1
        let n = 8usize;
        let input_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let input = Owned::f32(&[1, n, 1], &input_data);

        // dft_length = 8 (scalar i64)
        let dft_len = Owned::i64(&[], &[n as i64]);

        // Output: [1, 5, 2] (onesided: 8/2+1 = 5)
        let out_len = n / 2 + 1;
        let mut output = Owned::zeros_f32(&[1, out_len, 2]);

        let node = make_dft_node(-2, 1);
        let kernel = DftFactory
            .create(&node, &[vec![1, n, 1]])
            .expect("factory create");

        kernel
            .execute_prepared(&[input.view(), dft_len.view()], &mut [output.view_mut()])
            .expect("kernel execute");

        let out = output.to_f32();
        // Verify DC component: sum of 0..7 = 28
        assert!(
            (out[0] - 28.0).abs() < 1e-3,
            "DC real = {} expected 28.0",
            out[0]
        );
        assert!(out[1].abs() < 1e-3, "DC imag = {} expected 0.0", out[1]);
    }

    #[test]
    fn dft_kernel_rejects_onesided_complex_input() {
        let input = Owned::f32(&[1, 4, 2], &[0.0; 8]);
        let mut output = Owned::zeros_f32(&[1, 3, 2]);
        let node = make_dft_node(-2, 1);
        assert!(DftFactory.create(&node, &[vec![1, 4, 2]]).is_err());
        let kernel = DftFactory
            .create(&node, &[])
            .expect("unresolved shape defers validation");
        let error = kernel
            .execute_prepared(&[input.view()], &mut [output.view_mut()])
            .expect_err("onesided complex DFT must be rejected");
        assert!(error.to_string().contains("valid only for real input"));
    }

    /// Verify radix-2 FFT fallback fires (n=2, which is pow2 but below vDSP minimum).
    #[test]
    fn fft_fallback_reachability() {
        let n = 2usize;
        let real = vec![1.0f32, -1.0];
        let imag = vec![0.0f32; n];

        let before = DFT_FFT_TEST_HITS.load(Ordering::Relaxed);
        let (r, i) = compute_dft(&real, &imag, n, false, false);
        let after = DFT_FFT_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "DFT_FFT_TEST_HITS counter did not fire for n=2 (before={before}, after={after})"
        );
        // DC = 1 + (-1) = 0, bin[1] = 1 + (-1)*e^{-i*pi} = 1 - (-1) = 2
        assert!((r[0]).abs() < 1e-6, "DC real = {}", r[0]);
        assert!((r[1] - 2.0).abs() < 1e-6, "bin[1] real = {}", r[1]);
        assert!(i[0].abs() < 1e-6 && i[1].abs() < 1e-6);
    }

    /// The cross-platform counterpart of [`fft_fallback_reachability`], and the
    /// regression guard for #2089.
    ///
    /// That test deliberately uses `n = 2`, which is below the vDSP minimum, so
    /// it says nothing about the sizes production actually transforms. `n = 4`
    /// is the *first* length for which [`DftPlan::new`] builds a vDSP setup, so
    /// it is exactly the size at which the two platforms diverge: Linux and
    /// Windows reach the radix-2 counter, Apple returns from the vDSP arm
    /// before it. Asserting on the fast-path counter is what makes the claim
    /// -- "an eligible power-of-two length does not fall back to the naive
    /// O(n^2) transform" -- true on every target rather than on x86 only.
    ///
    /// This fails on macOS against the pre-#2089 code and passes here, which is
    /// the only property that distinguishes the fix from deleting the
    /// assertion.
    #[test]
    fn an_eligible_power_of_two_takes_a_fast_path_on_every_target() {
        let n = 4usize;
        let real = vec![1.0f32, 2.0, 3.0, 4.0];
        let imag = vec![0.0f32; n];

        let before = dft_fast_path_hits_this_thread();
        let (r, i) = compute_dft(&real, &imag, n, false, false);
        let after = dft_fast_path_hits_this_thread();
        assert_eq!(
            after,
            before + 1,
            "n={n} is vDSP-eligible and power-of-two, so it must take a fast path on \
             every target (before={before}, after={after})"
        );

        // Independent of which fast path served it: DC is the sum, and bin 2 is
        // the alternating sum.
        assert!((r[0] - 10.0).abs() < 1e-5, "DC real = {}", r[0]);
        assert!(i[0].abs() < 1e-5, "DC imag = {}", i[0]);
        assert!((r[2] + 2.0).abs() < 1e-5, "bin[2] real = {}", r[2]);
    }

    /// The per-thread counter has to actually be per-thread.
    ///
    /// Every dispatch assertion that survived #2093 -- the one above, and the
    /// STFT frame test -- differences [`dft_fast_path_hits_this_thread`] around
    /// a call and asserts an **exact equality**. That is sound only while the
    /// counter is unshared. If it ever reads the process-global instead, every
    /// one of those equalities quietly becomes `own + whatever concurrent tests
    /// transformed in the same window`, which is precisely the lower bound the
    /// per-thread counter was introduced to escape -- and no test would fail,
    /// because inflation is the direction that satisfies them.
    ///
    /// That is the shape this repository calls a false oracle: the instrument's
    /// broken reading is indistinguishable from its working one. The doc
    /// comment on `dft_fast_path_hits_this_thread` argues the property at
    /// length; this executes it.
    #[test]
    fn the_per_thread_counter_does_not_see_another_threads_transforms() {
        const CONCURRENT_DFTS: u64 = 64;

        let global_before = dft_fast_path_hits();
        let local_before = dft_fast_path_hits_this_thread();

        std::thread::spawn(|| {
            for _ in 0..CONCURRENT_DFTS {
                // n = 2 is a power of two below the vDSP minimum of 4, so this
                // takes the radix-2 fast path on every target, Apple included.
                compute_dft(&[1.0, -1.0], &[0.0, 0.0], 2, false, false);
            }
        })
        .join()
        .expect("the contaminating thread must not panic");

        // Anti-vacuity: without this, a thread that silently did nothing would
        // satisfy the assertion below just as well as an isolated counter does.
        let global_after = dft_fast_path_hits();
        assert!(
            global_after >= global_before + CONCURRENT_DFTS,
            "the shared counter must have absorbed the other thread's {CONCURRENT_DFTS} \
             transforms ({global_before} -> {global_after}); if it did not, this test is \
             no longer demonstrating the contamination it exists to rule out"
        );

        assert_eq!(
            dft_fast_path_hits_this_thread(),
            local_before,
            "this thread performed no transform, so its own counter must not have moved \
             -- a per-thread counter that sees another thread's work turns every exact \
             dispatch equality in this crate back into a lower bound, silently"
        );
    }
}
