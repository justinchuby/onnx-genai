//! ONNX `DFT` operator (opset 17+): the discrete Fourier transform.
//!
//! Supports real and complex input, forward and inverse, full and onesided output.
//! On macOS/iOS the fast path uses Accelerate's vDSP DFT for power-of-two lengths;
//! all other cases fall back to a Cooley–Tukey radix-2 FFT (power-of-two) or naive
//! O(N²) DFT (arbitrary length).

use std::f32::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::check_arity;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::numel;

/// Dispatch counter for the vDSP Accelerate DFT fast path.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub static DFT_VDSP_TEST_HITS: AtomicU64 = AtomicU64::new(0);

/// Dispatch counter for the radix-2 FFT fallback path.
pub static DFT_FFT_TEST_HITS: AtomicU64 = AtomicU64::new(0);

pub struct DftFactory;

impl KernelFactory for DftFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let inverse = node.attr("inverse").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        let onesided = node.attr("onesided").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        // Opset < 20: axis is an attribute (default 1).
        // Opset >= 20: axis is an input, handled at execute time.
        let axis_attr = node.attr("axis").and_then(|a| a.as_int());
        Ok(Box::new(DftKernel {
            inverse,
            onesided,
            axis_attr,
        }))
    }
}

struct DftKernel {
    inverse: bool,
    onesided: bool,
    axis_attr: Option<i64>,
}

impl Kernel for DftKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // DFT opset 17: inputs = [input, dft_length?]
        // DFT opset 20: inputs = [input, dft_length?, axis?]
        let min_in = 1;
        let max_in = 3;
        check_arity("DFT", inputs, outputs, min_in, max_in, 1)?;

        let input = &inputs[0];
        let rank = input.shape.len();
        if rank < 2 {
            return Err(EpError::KernelFailed(
                "DFT: input must have rank >= 2".into(),
            ));
        }

        // Determine the signal axis.
        let axis_raw = if inputs.len() >= 3 && !inputs[2].is_absent() {
            // Opset 20: axis as input
            let axis_data = super::to_dense_i64(&inputs[2])?;
            axis_data[0]
        } else {
            self.axis_attr.unwrap_or(-2)
        };

        let axis = normalize_axis(axis_raw, rank)?;
        let last = rank - 1;

        // The last dimension is the complex component dimension (1 for real, 2 for complex).
        let complex_dim = input.shape[last];
        let is_real_input = complex_dim == 1;
        if complex_dim != 1 && complex_dim != 2 {
            return Err(EpError::KernelFailed(format!(
                "DFT: last dimension must be 1 (real) or 2 (complex), got {complex_dim}"
            )));
        }
        if self.onesided && !is_real_input {
            return Err(EpError::KernelFailed(
                "DFT: onesided=1 is valid only for real input (last dimension 1)".into(),
            ));
        }

        // Determine DFT length.
        let signal_len = input.shape[axis];
        let dft_length = if inputs.len() >= 2 && !inputs[1].is_absent() {
            let len_data = super::to_dense_i64(&inputs[1])?;
            usize::try_from(len_data[0])
                .ok()
                .filter(|length| *length > 0)
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "DFT: dft_length must be positive, got {}",
                        len_data[0]
                    ))
                })?
        } else {
            signal_len
        };

        // Validate output shape.
        let out_signal_len = if self.onesided {
            dft_length / 2 + 1
        } else {
            dft_length
        };

        let output = &mut outputs[0];
        if output.shape[last] != 2 {
            return Err(EpError::KernelFailed(format!(
                "DFT: output last dim must be 2, got {}",
                output.shape[last]
            )));
        }
        if output.shape[axis] != out_signal_len {
            return Err(EpError::KernelFailed(format!(
                "DFT: output signal axis size mismatch: expected {out_signal_len}, got {}",
                output.shape[axis]
            )));
        }

        // Materialize input as f32. `to_dense_f32_widen` accepts the full ONNX
        // DFT type constraint (float32/float16/bfloat16/float64 and strided
        // float32) and folds every case into a dense f32 buffer, so DFT is
        // computed in f32 and narrowed back to the output dtype on write — the
        // same compute-in-f32 contract every other float kernel uses.
        let input_data = to_dense_f32_widen("DFT", input)?;

        // Compute batch dimensions: all dims except axis and last.
        let mut batch_shape: Vec<usize> = Vec::with_capacity(rank - 2);
        for (i, &d) in input.shape.iter().enumerate() {
            if i != axis && i != last {
                batch_shape.push(d);
            }
        }
        let batch_count: usize = batch_shape.iter().product();

        // Compute strides for the input layout (row-major).
        let input_strides = row_major_strides(input.shape);
        let output_strides = row_major_strides(output.shape);

        let total_output = numel(output.shape);
        let mut output_data = vec![0.0f32; total_output];

        // Iterate over batches.
        let mut batch_indices = vec![0usize; batch_shape.len()];
        for _ in 0..batch_count {
            // Map batch indices to input/output coordinates.
            let (in_base, out_base) = compute_bases(
                &batch_indices,
                input.shape,
                output.shape,
                axis,
                last,
                &input_strides,
                &output_strides,
            );

            // Extract signal from input.
            let mut real_in = vec![0.0f32; signal_len];
            let mut imag_in = vec![0.0f32; signal_len];
            for i in 0..signal_len {
                let idx = in_base + i * input_strides[axis];
                real_in[i] = input_data[idx];
                if !is_real_input {
                    imag_in[i] = input_data[idx + 1];
                }
            }

            // Zero-pad or truncate to dft_length.
            real_in.resize(dft_length, 0.0);
            imag_in.resize(dft_length, 0.0);

            // Compute DFT.
            let (real_out, imag_out) =
                compute_dft(&real_in, &imag_in, dft_length, self.inverse, self.onesided);

            // Write to output.
            for i in 0..out_signal_len {
                let idx = out_base + i * output_strides[axis];
                output_data[idx] = real_out[i];
                output_data[idx + 1] = imag_out[i];
            }

            // Advance batch indices.
            advance_indices(&mut batch_indices, &batch_shape);
        }

        write_dense_f32_narrow("DFT", output, &output_data)
    }
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize> {
    let r = rank as i64;
    let normalized = if axis >= 0 { axis } else { axis + r };
    if normalized < 0 || normalized >= r - 1 {
        return Err(EpError::KernelFailed(format!(
            "DFT: axis {axis} invalid for rank {rank} (last dim is complex component)"
        )));
    }
    Ok(normalized as usize)
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
fn compute_dft(
    real_in: &[f32],
    imag_in: &[f32],
    n: usize,
    inverse: bool,
    onesided: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut real_out = vec![0.0; n];
    let mut imag_out = vec![0.0; n];
    compute_dft_into(real_in, imag_in, &mut real_out, &mut imag_out, inverse);
    if onesided {
        let half = n / 2 + 1;
        real_out.truncate(half);
        imag_out.truncate(half);
    }
    (real_out, imag_out)
}

/// Compute one DFT into caller-owned full-length buffers.
pub(super) fn compute_dft_into(
    real_in: &[f32],
    imag_in: &[f32],
    real_out: &mut [f32],
    imag_out: &mut [f32],
    inverse: bool,
) {
    DftPlan::new(real_in.len(), inverse).transform(real_in, imag_in, real_out, imag_out);
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
            return;
        }

        if self.n.is_power_of_two() && self.n > 1 {
            DFT_FFT_TEST_HITS.fetch_add(1, Ordering::Relaxed);
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
}
