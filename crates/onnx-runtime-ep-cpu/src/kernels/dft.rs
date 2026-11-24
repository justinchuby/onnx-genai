//! ONNX `DFT` operator (opset 17+): the discrete Fourier transform.
//!
//! Supports real and complex input, forward and inverse, full and onesided output.
//! On macOS/iOS the fast path uses Accelerate's vDSP DFT for power-of-two lengths;
//! all other cases fall back to a Cooley–Tukey radix-2 FFT (power-of-two) or naive
//! O(N²) DFT (arbitrary length).

use std::cell::Cell;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::dft::{self as dft_rules, Spectrum};
use onnx_runtime_ir::{DataType, Node};
use rustfft::{FftPlanner, num_complex::Complex};

use super::{check_arity, to_dense_bytes, write_dense_bytes};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::numel;

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
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let inverse = node.attr("inverse").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        let onesided = node.attr("onesided").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        let opset = node.local_opset().unwrap_or(17);
        Ok(Box::new(DftKernel {
            inverse,
            spectrum: Spectrum::from_flags(opset, inverse, onesided),
            opset,
            default_axis: dft_rules::default_axis(node, opset),
        }))
    }
}

struct DftKernel {
    inverse: bool,
    spectrum: Spectrum,
    opset: u64,
    default_axis: i64,
}

impl Kernel for DftKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // DFT opset 17: inputs = [input, dft_length?]
        // DFT opset 20: inputs = [input, dft_length?, axis?]
        let min_in = 1;
        let max_in = if self.opset >= 20 { 3 } else { 2 };
        check_arity("DFT", inputs, outputs, min_in, max_in, 1)?;

        let input = &inputs[0];
        let rank = input.shape.len();
        if rank < 2 {
            return Err(EpError::KernelFailed(
                "DFT: input must have rank >= 2".into(),
            ));
        }

        // Determine the signal axis.
        let axis_raw = scalar(inputs, 2, "axis", |dtype| dtype == DataType::Int64)?
            .unwrap_or(self.default_axis);
        let axis = dft_rules::normalize_axis(axis_raw, rank).map_err(dft_error)?;
        let last = rank - 1;
        self.spectrum
            .validate_components(input.shape[last])
            .map_err(dft_error)?;
        let explicit = scalar(inputs, 1, "dft_length", dft_rules::is_length_dtype)?
            .map(dft_rules::explicit_length)
            .transpose()
            .map_err(dft_error)?;
        let dft_length = self
            .spectrum
            .transform_length(explicit, input.shape[axis])
            .map_err(dft_error)?;
        let mut expected = input.shape.to_vec();
        expected[axis] = self.spectrum.output_extent(dft_length);
        expected[last] = self.spectrum.output_components();
        let output = &mut outputs[0];
        if output.shape != expected || output.dtype != input.dtype {
            return Err(dft_error(format!(
                "expected output {:?} {expected:?}, got {:?} {:?}",
                input.dtype, output.dtype, output.shape
            )));
        }

        let geometry = DftGeometry {
            input_shape: input.shape,
            output_shape: output.shape,
            axis,
            dft_length,
            spectrum: self.spectrum,
        };
        if input.dtype == DataType::Float64 && self.opset >= 20 {
            let bytes = to_dense_bytes(input)?;
            let data: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|chunk| f64::from_ne_bytes(chunk.try_into().expect("8-byte chunk")))
                .collect();
            let mut planner = FftPlanner::<f64>::new();
            let plan = if self.inverse {
                planner.plan_fft_inverse(dft_length)
            } else {
                planner.plan_fft_forward(dft_length)
            };
            let mut scratch = vec![Complex::default(); plan.get_inplace_scratch_len()];
            let mut work = vec![Complex::default(); dft_length];
            let scale = if self.inverse {
                1.0 / dft_length as f64
            } else {
                1.0
            };
            let data = geometry.transform(&data, |real, imag, real_out, imag_out| {
                for (i, value) in work.iter_mut().enumerate() {
                    *value = Complex::new(real[i], imag[i]);
                }
                plan.process_with_scratch(&mut work, &mut scratch);
                for (i, value) in work.iter().enumerate() {
                    real_out[i] = value.re * scale;
                    imag_out[i] = value.im * scale;
                }
            });
            let bytes: Vec<u8> = data.into_iter().flat_map(f64::to_ne_bytes).collect();
            write_dense_bytes(output, &bytes)
        } else {
            let data = to_dense_f32_widen("DFT", input)?;
            let plan = DftPlan::new(dft_length, self.inverse);
            let data = geometry.transform(&data, |r, i, ro, io| plan.transform(r, i, ro, io));
            write_dense_f32_narrow("DFT", output, &data)
        }
    }
}

fn dft_error(reason: String) -> EpError {
    EpError::KernelFailed(format!("DFT: {reason}"))
}

fn scalar(
    inputs: &[TensorView],
    index: usize,
    name: &str,
    valid_type: impl Fn(DataType) -> bool,
) -> Result<Option<i64>> {
    let Some(input) = inputs.get(index).filter(|input| !input.is_absent()) else {
        return Ok(None);
    };
    if !input.shape.is_empty() || !valid_type(input.dtype) {
        return Err(dft_error(format!(
            "{name} must be a scalar of type {}, got {:?} {:?}",
            if index == 1 {
                "Int32 or Int64"
            } else {
                "Int64"
            },
            input.dtype,
            input.shape
        )));
    }
    Ok(Some(super::to_dense_i64(input)?[0]))
}

struct DftGeometry<'a> {
    input_shape: &'a [usize],
    output_shape: &'a [usize],
    axis: usize,
    dft_length: usize,
    spectrum: Spectrum,
}

impl DftGeometry<'_> {
    fn transform<T: Copy + Default + std::ops::Neg<Output = T>>(
        &self,
        input: &[T],
        mut transform: impl FnMut(&[T], &[T], &mut [T], &mut [T]),
    ) -> Vec<T> {
        let axis = self.axis;
        let last = self.input_shape.len() - 1;
        let n = self.dft_length;
        let bins = self.spectrum.input_extent(n);
        let keep = bins.min(self.input_shape[axis]);
        let real_output = self.spectrum == Spectrum::OneSidedInput;
        let batch_shape: Vec<_> = self.input_shape[..last]
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (i != axis).then_some(d))
            .collect();
        let batch_count: usize = batch_shape.iter().product();
        let input_strides = row_major_strides(self.input_shape);
        let output_strides = row_major_strides(self.output_shape);
        let mut output = vec![T::default(); numel(self.output_shape)];
        if batch_count == 0 {
            return output;
        }
        let mut real = vec![T::default(); n];
        let mut imag = vec![T::default(); n];
        let mut real_out = vec![T::default(); n];
        let mut imag_out = vec![T::default(); n];
        let mut batch_indices = vec![0usize; batch_shape.len()];
        for _ in 0..batch_count {
            let (in_base, out_base) = compute_bases(
                &batch_indices,
                self.input_shape,
                self.output_shape,
                axis,
                last,
                &input_strides,
                &output_strides,
            );

            for i in 0..keep {
                let idx = in_base + i * input_strides[axis];
                real[i] = input[idx];
                if self.input_shape[last] == 2 {
                    imag[i] = input[idx + 1];
                }
            }
            if real_output {
                // DC and the even-length Nyquist bin are their own conjugates.
                imag[0] = T::default();
                if n % 2 == 0 {
                    imag[n / 2] = T::default();
                }
                for i in bins..n {
                    real[i] = real[n - i];
                    imag[i] = -imag[n - i];
                }
            }
            transform(&real, &imag, &mut real_out, &mut imag_out);
            for i in 0..self.output_shape[axis] {
                let idx = out_base + i * output_strides[axis];
                output[idx] = real_out[i];
                if !real_output {
                    output[idx + 1] = imag_out[i];
                }
            }
            advance_indices(&mut batch_indices, &batch_shape);
        }
        output
    }
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
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

/// Compute the DFT using the best available method.
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

/// A transform plan reusable across STFT frames of one fixed length.
pub(super) struct DftPlan {
    n: usize,
    inverse: bool,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    vdsp_setup: VdspDftSetup,
}

impl DftPlan {
    pub(super) fn new(n: usize, inverse: bool) -> Self {
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

fn fft_radix2_in_place(real: &mut [f32], imag: &mut [f32], inverse: bool) {
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
    let sign: f32 = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = sign * 2.0 * PI / len as f32;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let w_re = (angle * k as f32).cos();
                let w_im = (angle * k as f32).sin();
                let i1 = start + k;
                let i2 = start + k + half;
                let tr = real[i2] * w_re - imag[i2] * w_im;
                let ti = real[i2] * w_im + imag[i2] * w_re;
                real[i2] = real[i1] - tr;
                imag[i2] = imag[i1] - ti;
                real[i1] += tr;
                imag[i1] += ti;
            }
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f32;
        for x in real.iter_mut() {
            *x *= scale;
        }
        for x in imag.iter_mut() {
            *x *= scale;
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

fn naive_dft_into(
    real_in: &[f32],
    imag_in: &[f32],
    real_out: &mut [f32],
    imag_out: &mut [f32],
    inverse: bool,
) {
    let n = real_in.len();
    debug_assert_eq!(imag_in.len(), n);
    debug_assert_eq!(real_out.len(), imag_out.len());
    debug_assert!(real_out.len() <= n);
    let sign: f32 = if inverse { 1.0 } else { -1.0 };
    for k in 0..real_out.len() {
        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        for t in 0..n {
            let angle = sign * 2.0 * PI * (k as f32) * (t as f32) / (n as f32);
            let w_re = angle.cos();
            let w_im = angle.sin();
            sum_re += real_in[t] * w_re - imag_in[t] * w_im;
            sum_im += real_in[t] * w_im + imag_in[t] * w_re;
        }
        if inverse {
            let scale = 1.0 / n as f32;
            sum_re *= scale;
            sum_im *= scale;
        }
        real_out[k] = sum_re;
        imag_out[k] = sum_im;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
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
            .execute(&[input.view()], &mut [opset17_output.view_mut()])
            .unwrap();

        let mut opset20_output = Owned::zeros_f32(&[1, 8, 4, 2]);
        DftFactory
            .create(&versioned_dft_node(20, None), &[vec![1, 8, 6, 1]])
            .unwrap()
            .execute(&[input.view()], &mut [opset20_output.view_mut()])
            .unwrap();

        let mut attr_output = Owned::zeros_f32(&[1, 8, 4, 2]);
        DftFactory
            .create(&versioned_dft_node(17, Some(2)), &[vec![1, 8, 6, 1]])
            .unwrap()
            .execute(&[input.view()], &mut [attr_output.view_mut()])
            .unwrap();

        let axis = Owned::i64(&[], &[1]);
        let mut input_axis_output = Owned::zeros_f32(&[1, 5, 6, 2]);
        DftFactory
            .create(&versioned_dft_node(20, None), &[vec![1, 8, 6, 1]])
            .unwrap()
            .execute(
                &[
                    input.view(),
                    TensorView::absent(onnx_runtime_ir::DataType::Int64),
                    axis.view(),
                ],
                &mut [input_axis_output.view_mut()],
            )
            .unwrap();
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
            .execute(&[input.view(), dft_len.view()], &mut [output.view_mut()])
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
        let kernel = DftFactory
            .create(&node, &[vec![1, 4, 2]])
            .expect("factory create");
        let error = kernel
            .execute(&[input.view()], &mut [output.view_mut()])
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
    fn irfft20() -> Box<dyn Kernel> {
        let mut node = versioned_dft_node(20, None);
        node.attributes.insert("inverse".into(), Attribute::Int(1));
        DftFactory.create(&node, &[]).unwrap()
    }

    #[test]
    fn dft_v20_irfft_odd_int32_pads_missing_bins_on_a_non_last_axis() {
        let signal = Owned::f32(&[2, 2, 2], &[1., 9., 2., -8., 3., 4., -2., 1.]);
        let length = Owned::i32(&[], &[5]);
        let axis = Owned::i64(&[], &[-3]);
        let mut output = Owned::zeros_f32(&[5, 2, 1]);
        irfft20()
            .execute(
                &[signal.view(), length.view(), axis.view()],
                &mut [output.view_mut()],
            )
            .unwrap();
        // np.fft.irfft([[1+9j, 2-8j], [3+4j, -2+1j]], n=5, axis=0)
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
        let actual = to_dense_f32_widen("DFT test", &output.view()).unwrap();
        for (i, (&a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 2e-6, "index {i}: expected {e}, got {a}");
        }
    }

    #[test]
    fn dft_v20_irfft_default_length_preserves_float64_precision() {
        let delta = 2f64.powi(-40);
        let signal = Owned::f64(&[2, 2], &[1.0 + delta, 9.0, 1.0, 11.0]);
        let mut output = Owned::f64(&[2, 1], &[0.; 2]);
        irfft20()
            .execute(&[signal.view()], &mut [output.view_mut()])
            .unwrap();
        // np.fft.irfft([1+2**-40+9j, 1+11j]); both imaginary parts are ignored.
        assert_eq!(
            output.bytes,
            Owned::f64(&[2, 1], &[1.0 + delta / 2.0, delta / 2.0]).bytes
        );
    }

    #[test]
    fn dft_v20_irfft_accepts_lower_precision_and_int64_length() {
        let f16 = Owned::f16(&[3, 2], &[1., 9., 3., 4., 5., -6.]);
        let bf16 = Owned::bf16(&[3, 2], &[1., 9., 3., 4., 5., -6.]);
        let length = Owned::i64(&[], &[4]);
        for (signal, mut output) in [
            (f16, Owned::f16(&[4, 1], &[0.; 4])),
            (bf16, Owned::bf16(&[4, 1], &[0.; 4])),
        ] {
            irfft20()
                .execute(&[signal.view(), length.view()], &mut [output.view_mut()])
                .unwrap();
            // np.fft.irfft([1+9j, 3+4j, 5-6j])
            assert_eq!(
                to_dense_f32_widen("DFT test", &output.view()).unwrap(),
                [3., -3., 0., 1.]
            );
        }
    }
}
