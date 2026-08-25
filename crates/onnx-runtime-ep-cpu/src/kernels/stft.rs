//! ONNX `STFT` operator (opset 17).
//!
//! ONNX slides complete, uncentered frames over `[batch, signal, 1|2]`; it
//! does not pad the signal. Power-of-two frame lengths share DFT's O(N log N)
//! radix-2/vDSP path. Other lengths currently use the shared O(N²) scalar DFT.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::check_arity;
use super::dft::DftPlan;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

pub struct StftFactory;

pub(crate) fn unsupported_dtype_reason(input_dtypes: &[DataType]) -> Option<String> {
    let signal = input_dtypes.first().copied().unwrap_or(DataType::Undefined);
    if !matches!(
        signal,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Some(format!(
            "STFT signal must be float32, float16, or bfloat16 because this CPU \
             kernel computes in f32, got {signal:?}"
        ));
    }
    let frame_step = input_dtypes.get(1).copied().unwrap_or(DataType::Undefined);
    if !matches!(frame_step, DataType::Int32 | DataType::Int64) {
        return Some(format!(
            "STFT frame_step must be int32 or int64, got {frame_step:?}"
        ));
    }
    if let Some(&window) = input_dtypes.get(2)
        && window != DataType::Undefined
        && window != signal
    {
        return Some(format!(
            "STFT window dtype {window:?} must match signal dtype {signal:?}"
        ));
    }
    if let Some(&frame_length) = input_dtypes.get(3)
        && frame_length != DataType::Undefined
        && !matches!(frame_length, DataType::Int32 | DataType::Int64)
    {
        return Some(format!(
            "STFT frame_length must be int32 or int64, got {frame_length:?}"
        ));
    }
    None
}

impl KernelFactory for StftFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let onesided = node.attr("onesided").and_then(|a| a.as_int()).unwrap_or(1);
        if !matches!(onesided, 0 | 1) {
            return Err(EpError::KernelFailed(format!(
                "STFT: attribute `onesided` must be 0 or 1, got {onesided}"
            )));
        }
        Ok(Box::new(StftKernel {
            onesided: onesided != 0,
        }))
    }
}

struct StftKernel {
    onesided: bool,
}

impl Kernel for StftKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("STFT", inputs, outputs, 2, 4, 1)?;

        let signal = &inputs[0];
        require_compute_dtype("signal", signal.dtype)?;
        if signal.shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "STFT: `signal` must have rank 3 [batch, signal_length, 1|2], got rank {}",
                signal.shape.len()
            )));
        }
        let components = signal.shape[2];
        if components != 1 && components != 2 {
            return Err(EpError::KernelFailed(format!(
                "STFT: `signal` last dimension must be 1 (real) or 2 (complex), got {components}"
            )));
        }
        if self.onesided && components == 2 {
            return Err(EpError::KernelFailed(
                "STFT: `onesided=1` requires a real signal (last dimension 1); \
                 use `onesided=0` for complex input"
                    .into(),
            ));
        }

        let frame_step = positive_scalar("frame_step", &inputs[1])?;
        let window = inputs.get(2).filter(|input| !input.is_absent());
        let frame_length_input = inputs.get(3).filter(|input| !input.is_absent());

        let window_length = if let Some(window) = window {
            require_compute_dtype("window", window.dtype)?;
            if window.dtype != signal.dtype {
                return Err(EpError::KernelFailed(format!(
                    "STFT: `window` dtype {:?} must match `signal` dtype {:?}",
                    window.dtype, signal.dtype
                )));
            }
            if window.shape.len() != 1 {
                return Err(EpError::KernelFailed(format!(
                    "STFT: `window` must have rank 1, got rank {}",
                    window.shape.len()
                )));
            }
            if window.shape[0] == 0 {
                return Err(EpError::KernelFailed(
                    "STFT: `window` length must be greater than zero".into(),
                ));
            }
            Some(window.shape[0])
        } else {
            None
        };
        let explicit_frame_length = frame_length_input
            .map(|input| positive_scalar("frame_length", input))
            .transpose()?;

        let frame_length = match (window_length, explicit_frame_length) {
            (Some(window_length), Some(frame_length)) => {
                if window_length != frame_length {
                    return Err(EpError::KernelFailed(format!(
                        "STFT: `window` length {window_length} must equal `frame_length` {frame_length}"
                    )));
                }
                frame_length
            }
            (Some(window_length), None) => window_length,
            (None, Some(frame_length)) => frame_length,
            (None, None) => {
                return Err(EpError::KernelFailed(
                    "STFT: either optional `window` or `frame_length` must be provided".into(),
                ));
            }
        };

        let signal_length = signal.shape[1];
        if frame_length > signal_length {
            return Err(EpError::KernelFailed(format!(
                "STFT: frame length {frame_length} exceeds signal length {signal_length}; \
                 STFT uses complete unpadded frames"
            )));
        }
        let frames = (signal_length - frame_length) / frame_step + 1;
        let bins = if self.onesided {
            frame_length / 2 + 1
        } else {
            frame_length
        };

        let output = &mut outputs[0];
        require_compute_dtype("output", output.dtype)?;
        if output.dtype != signal.dtype {
            return Err(EpError::KernelFailed(format!(
                "STFT: output dtype {:?} must match `signal` dtype {:?}",
                output.dtype, signal.dtype
            )));
        }
        let expected_shape = [signal.shape[0], frames, bins, 2];
        if output.shape != expected_shape {
            return Err(EpError::KernelFailed(format!(
                "STFT: output shape mismatch: expected {expected_shape:?}, got {:?}",
                output.shape
            )));
        }

        let signal_data = to_dense_f32_widen("STFT signal", signal)?;
        let window_data = window
            .map(|window| to_dense_f32_widen("STFT window", window))
            .transpose()?;
        let mut output_data = vec![0.0f32; signal.shape[0] * frames * bins * 2];

        // Four frame-sized buffers are allocated once and reused for every
        // batch/frame pair. The materialized signal/window and final output are
        // the only input/output-sized allocations.
        let mut frame_real = vec![0.0f32; frame_length];
        let mut frame_imag = vec![0.0f32; frame_length];
        let mut spectrum_real = vec![0.0f32; frame_length];
        let mut spectrum_imag = vec![0.0f32; frame_length];
        let dft = DftPlan::new(frame_length, false);

        for batch in 0..signal.shape[0] {
            for frame in 0..frames {
                let signal_start =
                    batch * signal_length * components + frame * frame_step * components;
                for sample in 0..frame_length {
                    let weight = window_data.as_ref().map_or(1.0, |values| values[sample]);
                    let input = signal_start + sample * components;
                    frame_real[sample] = signal_data[input] * weight;
                    frame_imag[sample] = if components == 2 {
                        signal_data[input + 1] * weight
                    } else {
                        0.0
                    };
                }

                dft.transform(
                    &frame_real,
                    &frame_imag,
                    &mut spectrum_real,
                    &mut spectrum_imag,
                );

                let output_start = (batch * frames + frame) * bins * 2;
                for bin in 0..bins {
                    output_data[output_start + bin * 2] = spectrum_real[bin];
                    output_data[output_start + bin * 2 + 1] = spectrum_imag[bin];
                }
            }
        }

        write_dense_f32_narrow("STFT", output, &output_data)
    }
}

fn require_compute_dtype(argument: &str, dtype: DataType) -> Result<()> {
    if matches!(
        dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        Ok(())
    } else {
        Err(EpError::KernelFailed(format!(
            "STFT: `{argument}` dtype must be float32, float16, or bfloat16 \
             (the CPU kernel computes in f32); got {dtype:?}"
        )))
    }
}

fn positive_scalar(name: &str, input: &TensorView<'_>) -> Result<usize> {
    if input.is_absent() {
        return Err(EpError::KernelFailed(format!(
            "STFT: required `{name}` input is absent"
        )));
    }
    if !input.shape.is_empty() {
        return Err(EpError::KernelFailed(format!(
            "STFT: `{name}` must be a scalar, got shape {:?}",
            input.shape
        )));
    }
    let value = super::to_dense_i64(input)?[0];
    if value <= 0 {
        return Err(EpError::KernelFailed(format!(
            "STFT: `{name}` must be greater than zero, got {value}"
        )));
    }
    usize::try_from(value).map_err(|_| {
        EpError::KernelFailed(format!(
            "STFT: `{name}` value {value} does not fit this platform's index size"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::dft::fast_path_hits;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::TensorView;
    use onnx_runtime_ir::{Attribute, NodeId};

    fn node(onesided: i64) -> Node {
        let mut node = Node::new(NodeId(0), "STFT", vec![], vec![]);
        node.attributes
            .insert("onesided".into(), Attribute::Int(onesided));
        node
    }

    fn reference(
        signal: &[f64],
        components: usize,
        frame_step: usize,
        frame_length: usize,
        window: Option<&[f64]>,
        onesided: bool,
    ) -> Vec<f32> {
        let signal_length = signal.len() / components;
        let frames = (signal_length - frame_length) / frame_step + 1;
        let bins = if onesided {
            frame_length / 2 + 1
        } else {
            frame_length
        };
        let mut output = Vec::with_capacity(frames * bins * 2);
        for frame in 0..frames {
            for bin in 0..bins {
                let mut real = 0.0f64;
                let mut imag = 0.0f64;
                for sample in 0..frame_length {
                    let weight = window.map_or(1.0, |values| values[sample]);
                    let index = (frame * frame_step + sample) * components;
                    let input_real = signal[index] * weight;
                    let input_imag = if components == 2 {
                        signal[index + 1] * weight
                    } else {
                        0.0
                    };
                    let angle = -2.0 * std::f64::consts::PI * bin as f64 * sample as f64
                        / frame_length as f64;
                    real += input_real * angle.cos() - input_imag * angle.sin();
                    imag += input_real * angle.sin() + input_imag * angle.cos();
                }
                output.push(real as f32);
                output.push(imag as f32);
            }
        }
        output
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-4,
                "value {index}: got {actual}, expected {expected}"
            );
        }
    }

    fn execute(
        signal: &Owned,
        step: &Owned,
        window: Option<&Owned>,
        length: Option<&Owned>,
        onesided: i64,
        output_shape: &[usize],
    ) -> Result<Owned> {
        let mut output = Owned::zeros_f32(output_shape);
        let absent_window = TensorView::absent(DataType::Float32);
        let absent_length = TensorView::absent(DataType::Int64);
        let window_view = window.map_or(absent_window, Owned::view);
        let length_view = length.map_or(absent_length, Owned::view);
        StftFactory.create(&node(onesided), &[])?.execute(
            &[signal.view(), step.view(), window_view, length_view],
            &mut [output.view_mut()],
        )?;
        Ok(output)
    }

    #[test]
    fn real_unwindowed_overlapping_frames_match_independent_reference() {
        let values: Vec<f32> = (1..=8).map(|value| value as f32).collect();
        let signal = Owned::f32(&[1, 8, 1], &values);
        let step = Owned::i64(&[], &[2]);
        let length = Owned::i64(&[], &[4]);
        let before = fast_path_hits();
        let output = execute(&signal, &step, None, Some(&length), 0, &[1, 3, 4, 2]).unwrap();
        let after = fast_path_hits();

        let input: Vec<f64> = values.iter().map(|&value| value as f64).collect();
        assert_close(&output.to_f32(), &reference(&input, 1, 2, 4, None, false));
        assert_eq!(output.shape[1], 3, "the last eligible frame must be kept");
        assert!(
            after >= before + 3,
            "each power-of-two frame must take a fast path rather than the naive DFT \
             (radix-2 everywhere, vDSP on Apple targets); before={before} after={after}"
        );
        // The middle frame starts at sample 2. A non-overlapping increment
        // would instead transform samples 4..8 and fail this comparison.
        let middle_dc = output.to_f32()[4 * 2];
        assert_eq!(middle_dc, 18.0);
    }

    #[test]
    fn strided_signal_and_nontrivial_strided_window_are_honored() {
        let signal_storage = Owned::f32(
            &[1, 12, 1],
            &[
                1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0, 5.0, 99.0, 6.0, 99.0,
            ],
        )
        .with_view(&[1, 6, 1], &[12, 2, 1]);
        let window_storage =
            Owned::f32(&[8], &[0.25, 77.0, 0.5, 77.0, 1.5, 77.0, 2.0, 77.0]).with_view(&[4], &[2]);
        let step = Owned::i32(&[], &[2]);
        let output = execute(
            &signal_storage,
            &step,
            Some(&window_storage),
            None,
            0,
            &[1, 2, 4, 2],
        )
        .unwrap();

        let expected = reference(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            1,
            2,
            4,
            Some(&[0.25, 0.5, 1.5, 2.0]),
            false,
        );
        assert_close(&output.to_f32(), &expected);
        assert_ne!(
            output.to_f32()[0],
            10.0,
            "an implementation that ignores the window must fail"
        );
    }

    #[test]
    fn complex_signal_is_supported_only_for_two_sided_output() {
        let values = [1.0, 0.5, 2.0, -1.0, 0.0, 3.0, -2.0, 0.25];
        let signal = Owned::f32(&[1, 4, 2], &values);
        let step = Owned::i64(&[], &[4]);
        let length = Owned::i64(&[], &[4]);
        let output = execute(&signal, &step, None, Some(&length), 0, &[1, 1, 4, 2]).unwrap();
        let input: Vec<f64> = values.iter().map(|&value| value as f64).collect();
        assert_close(&output.to_f32(), &reference(&input, 2, 4, 4, None, false));

        let err = execute(&signal, &step, None, Some(&length), 1, &[1, 1, 3, 2])
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires a real signal"), "{err}");
    }

    #[test]
    fn onesided_keeps_nyquist_and_full_output_contains_conjugate_half() {
        let values = [0.5f32, 1.0, -0.25, 2.0, 0.0, -1.0, 0.75, 0.25];
        let signal = Owned::f32(&[1, 8, 1], &values);
        let step = Owned::i64(&[], &[8]);
        let length = Owned::i64(&[], &[8]);
        let one = execute(&signal, &step, None, Some(&length), 1, &[1, 1, 5, 2]).unwrap();
        let full = execute(&signal, &step, None, Some(&length), 0, &[1, 1, 8, 2]).unwrap();
        let one = one.to_f32();
        let full = full.to_f32();

        assert_close(&one, &full[..10]);
        assert!((full[2] - full[14]).abs() < 1e-5);
        assert!((full[3] + full[15]).abs() < 1e-5);
        assert!(
            one[8].abs() > 1e-3,
            "the N/2 Nyquist bin must be present; N/2 bins is off by one"
        );
    }

    #[test]
    fn exact_frame_succeeds_and_short_signal_fails_without_padding() {
        let step = Owned::i64(&[], &[2]);
        let length = Owned::i64(&[], &[4]);
        let exact = Owned::f32(&[1, 4, 1], &[1.0, 2.0, 3.0, 4.0]);
        let output = execute(&exact, &step, None, Some(&length), 1, &[1, 1, 3, 2]).unwrap();
        assert_eq!(output.shape, [1, 1, 3, 2]);

        let short = Owned::f32(&[1, 3, 1], &[1.0, 2.0, 3.0]);
        let err = execute(&short, &step, None, Some(&length), 1, &[1, 0, 3, 2])
            .unwrap_err()
            .to_string();
        assert!(err.contains("complete unpadded frames"), "{err}");
    }

    #[test]
    fn invalid_step_window_length_complex_dimension_and_dtype_fail_loudly() {
        let signal = Owned::f32(&[1, 4, 1], &[1.0, 2.0, 3.0, 4.0]);
        let zero_step = Owned::i64(&[], &[0]);
        let length = Owned::i64(&[], &[4]);
        let err = execute(&signal, &zero_step, None, Some(&length), 1, &[1, 1, 3, 2])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`frame_step` must be greater than zero"),
            "{err}"
        );

        let step = Owned::i64(&[], &[2]);
        let wrong_window = Owned::f32(&[3], &[1.0, 0.5, 1.0]);
        let err = execute(
            &signal,
            &step,
            Some(&wrong_window),
            Some(&length),
            1,
            &[1, 1, 3, 2],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("must equal `frame_length`"), "{err}");

        let bad_components = Owned::f32(&[1, 4, 3], &[0.0; 12]);
        let err = execute(
            &bad_components,
            &step,
            None,
            Some(&length),
            0,
            &[1, 1, 4, 2],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("last dimension must be 1"), "{err}");

        let f64_signal = Owned::f64(&[1, 4, 1], &[1.0, 2.0, 3.0, 4.0]);
        let err = execute(&f64_signal, &step, None, Some(&length), 0, &[1, 1, 4, 2])
            .unwrap_err()
            .to_string();
        assert!(err.contains("computes in f32"), "{err}");
    }

    #[test]
    fn frame_length_and_window_are_not_distinct_fft_sizes() {
        let signal = Owned::f32(&[1, 8, 1], &[0.0; 8]);
        let step = Owned::i64(&[], &[2]);
        let window = Owned::f32(&[4], &[1.0; 4]);
        let length = Owned::i64(&[], &[8]);
        let err = execute(
            &signal,
            &step,
            Some(&window),
            Some(&length),
            1,
            &[1, 1, 5, 2],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("must equal `frame_length`"), "{err}");
    }

    #[test]
    fn float16_and_bfloat16_compute_in_f32_and_narrow_once() {
        let step = Owned::i64(&[], &[4]);
        let length = Owned::i64(&[], &[4]);
        for (dtype, signal) in [
            (
                DataType::Float16,
                Owned::f16(&[1, 4, 1], &[1.0, 2.0, 3.0, 4.0]),
            ),
            (
                DataType::BFloat16,
                Owned::bf16(&[1, 4, 1], &[1.0, 2.0, 3.0, 4.0]),
            ),
        ] {
            let mut output = Owned::zeros(dtype, &[1, 1, 3, 2]);
            StftFactory
                .create(&node(1), &[])
                .unwrap()
                .execute(
                    &[
                        signal.view(),
                        step.view(),
                        TensorView::absent(dtype),
                        length.view(),
                    ],
                    &mut [output.view_mut()],
                )
                .unwrap();
            let dc_bits = u16::from_le_bytes([output.bytes[0], output.bytes[1]]);
            let dc = if dtype == DataType::Float16 {
                half::f16::from_bits(dc_bits).to_f32()
            } else {
                half::bf16::from_bits(dc_bits).to_f32()
            };
            assert_eq!(dc, 10.0, "{dtype:?} output must contain the f32 DC sum");
        }
    }

    #[test]
    fn arbitrary_frame_length_matches_naive_reference() {
        let values = [1.0f32, -2.0, 0.5, 3.0, 1.5];
        let signal = Owned::f32(&[1, 5, 1], &values);
        let step = Owned::i64(&[], &[1]);
        let length = Owned::i64(&[], &[3]);
        let output = execute(&signal, &step, None, Some(&length), 0, &[1, 3, 3, 2]).unwrap();
        let input: Vec<f64> = values.iter().map(|&value| value as f64).collect();
        assert_close(&output.to_f32(), &reference(&input, 1, 1, 3, None, false));
    }

    #[test]
    fn invalid_onesided_attribute_is_rejected_by_the_factory() {
        let err = match StftFactory.create(&node(2), &[]) {
            Ok(_) => panic!("onesided=2 must be rejected"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("must be 0 or 1"), "{err}");
    }

    #[test]
    fn onesided_defaults_to_one() {
        let signal = Owned::f32(&[1, 4, 1], &[1.0, 2.0, 3.0, 4.0]);
        let step = Owned::i64(&[], &[4]);
        let length = Owned::i64(&[], &[4]);
        let mut output = Owned::zeros_f32(&[1, 1, 3, 2]);
        StftFactory
            .create(&Node::new(NodeId(0), "STFT", vec![], vec![]), &[])
            .unwrap()
            .execute(
                &[
                    signal.view(),
                    step.view(),
                    TensorView::absent(DataType::Float32),
                    length.view(),
                ],
                &mut [output.view_mut()],
            )
            .unwrap();
        assert_eq!(output.shape[2], 3);
    }
}
