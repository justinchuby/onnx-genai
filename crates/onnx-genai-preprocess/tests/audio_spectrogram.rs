//! Independent DFT and executable dataflow coverage for authored power spectra.
use onnx_genai_preprocess::audio::{AudioProgram, AudioTensorData, encode_wav_pcm16};
use serde_json::{Value, json};

fn spectrum(n_fft: usize, hop: usize, window: usize) -> Value {
    json!({"op":"spectrogram","mode":"power","window":"hann","sample_rate":8000,"n_fft":n_fft,"hop_length":hop,"win_length":window,"outputs":["spectrum"]})
}
fn output(name: &str, source: &str, content: &str, dtype: &str) -> Value {
    json!({"name":name,"source":source,"content":content,"dtype":dtype})
}
fn compile(
    transforms: Value,
    outputs: Value,
) -> Result<AudioProgram, onnx_genai_preprocess::audio::AudioPreprocessError> {
    AudioProgram::from_program(
        &serde_json::from_value(json!({"transforms":transforms,"outputs":outputs})).unwrap(),
    )
}
fn features_output() -> Value {
    json!([output("features", "spectrum", "audio_features", "float32")])
}
fn encoded(samples: &[f32]) -> Vec<u8> {
    encode_wav_pcm16(samples, 8000, 1).unwrap()
}
fn floats(tensor: &onnx_genai_preprocess::audio::NamedAudioTensor) -> &[f32] {
    let AudioTensorData::Fp32(data) = &tensor.data else {
        panic!("float output required")
    };
    data
}
/// Slow double-precision DFT with independently constructed reflected indices.
fn reference(
    samples: &[f32],
    fft: usize,
    hop: usize,
    window: usize,
    mode: &str,
    window_kind: &str,
) -> Vec<f64> {
    let frames = samples.len() / hop;
    let mut expected = Vec::new();
    for bin in 0..=fft / 2 {
        for frame in 0..frames {
            let (mut real, mut imaginary) = (0.0, 0.0);
            for offset in 0..window {
                let position = (fft - window) / 2 + offset;
                let mut source = (frame * hop + position) as i64 - (fft / 2) as i64;
                if samples.len() == 1 {
                    source = 0;
                }
                while source < 0 || source >= samples.len() as i64 {
                    if source < 0 {
                        source = -source;
                    }
                    if source >= samples.len() as i64 {
                        source = 2 * samples.len() as i64 - 2 - source;
                    }
                }
                let phase = std::f64::consts::TAU * offset as f64 / window as f64;
                let coefficient = match window_kind {
                    "hann" => 0.5 - 0.5 * phase.cos(),
                    "hamming" => 0.54 - 0.46 * phase.cos(),
                    "rectangular" => 1.0,
                    _ => panic!("reference window must be explicit"),
                };
                let value = f64::from(samples[source as usize]) * coefficient;
                let phase = std::f64::consts::TAU * (bin * position) as f64 / fft as f64;
                real += value * phase.cos();
                imaginary -= value * phase.sin();
            }
            expected.push(match mode {
                "power" => real * real + imaginary * imaginary,
                "magnitude" => real.hypot(imaginary),
                _ => panic!("reference mode must be explicit"),
            });
        }
    }
    expected
}
#[test]
fn authored_power_spectrogram_matches_independent_dft_for_varied_geometry() {
    let samples = [
        -0.5, 0.25, 0.5, -0.25, 0.0, 0.5, 0.25, -0.5, 0.25, 0.5, 0.0, -0.5, 0.25, 0.5, -0.25, 0.5,
        0.0,
    ];
    for (fft, hop, window) in [(3, 2, 3), (8, 3, 5), (12, 4, 10)] {
        let program = compile(
            json!([{"op":"decode"},spectrum(fft,hop,window)]),
            features_output(),
        )
        .unwrap();
        let result = program
            .run(
                &[&encoded(&samples)],
                onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
            )
            .unwrap();
        assert_eq!(
            result[0].shape,
            [1, (fft / 2 + 1) as i64, (samples.len() / hop) as i64]
        );
        let expected = reference(&samples, fft, hop, window, "power", "hann");
        for (actual, expected) in floats(&result[0]).iter().zip(expected) {
            assert!(
                (f64::from(*actual) - expected).abs() < 1e-5 * (1.0 + expected),
                "geometry {fft}/{hop}/{window}: {actual} != {expected}"
            );
        }
    }
}
#[test]
fn power_branches_frame_validity_masks_and_input_windows_keep_their_sources() {
    let mut spec = spectrum(8, 2, 6);
    spec["inputs"] = json!(["padded"]);
    let mut scaled_spec = spec.clone();
    scaled_spec["inputs"] = json!(["scaled"]);
    scaled_spec["outputs"] = json!(["scaled_spectrum"]);
    let program = compile(
        json!([
            {"op":"decode","outputs":["raw"]},
            {"op":"resample","sample_rate":8000,"outputs":["resampled"]},
            {"op":"pad","mode":"fixed_window","target_length":8,"outputs":["padded"]},
            {"op":"rescale","scale":2.0,"outputs":["scaled"]}, spec, scaled_spec,
            {"op":"emit_validity_mask","inputs":["spectrum"],"outputs":["mask"]},
            {"op":"emit_valid_frames","inputs":["spectrum"],"outputs":["frames"]}
        ]),
        json!([
            output("features", "spectrum", "audio_features", "float32"),
            output("scaled", "scaled_spectrum", "features", "float32"),
            output("mask", "mask", "validity_mask", "int64"),
            output("frames", "frames", "valid_frames", "int64")
        ]),
    )
    .unwrap();
    assert_eq!(
        program.maximum_input_samples(8000).unwrap().unwrap().get(),
        8
    );
    assert_eq!(
        program.maximum_input_samples(16000).unwrap().unwrap().get(),
        16
    );
    let result = program
        .run(
            &[&encoded(&[0.25, 0.5, 0.25]), &encoded(&[0.25; 8])],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap();
    assert_eq!(result[0].shape, [2, 5, 4]);
    for (raw, scaled) in floats(&result[0]).iter().zip(floats(&result[1])) {
        assert!((scaled - 4.0 * raw).abs() < 1e-5);
    }
    assert_eq!(
        result[2].data,
        AudioTensorData::Int64(vec![1, 1, 0, 0, 1, 1, 1, 1])
    );
    assert_eq!(result[3].data, AudioTensorData::Int64(vec![2, 4]));
}
#[test]
fn spectrum_geometry_modes_types_and_analysis_rate_fail_with_context() {
    for (field, value, needle) in [
        ("mode", json!("complex"), "mode: power or magnitude"),
        (
            "window",
            json!("blackman"),
            "window: hann, hamming or rectangular",
        ),
        ("mode", json!(null), "mode: power or magnitude"),
        (
            "window",
            json!(null),
            "window: hann, hamming or rectangular",
        ),
        ("n_fft", json!(0), "n_fft"),
        ("hop_length", json!(null), "hop_length"),
        ("win_length", json!(100), "win_length"),
    ] {
        let mut spec = spectrum(8, 2, 8);
        spec[field] = value;
        let error = compile(json!([{"op":"decode"},spec]), features_output())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("transform[1]") && error.contains(needle),
            "{error}"
        );
    }
    let error=compile(json!([{"op":"decode"},spectrum(8,2,8),{"op":"normalize","mode":"whisper_log_mel","outputs":["normalized"]}]),json!([output("out","normalized","audio_features","float32")])).unwrap_err().to_string();
    assert!(
        error.contains("normalization") && error.contains("log-mel"),
        "{error}"
    );
    let program = compile(json!([{"op":"decode"},spectrum(8,2,8)]), features_output()).unwrap();
    let wrong_rate = encode_wav_pcm16(&[0.25; 8], 16000, 1).unwrap();
    let error = program
        .run(
            &[&wrong_rate],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("spectrogram") && error.contains("resample"),
        "{error}"
    );
}
#[test]
fn empty_and_single_sample_spectra_remain_finite_and_ragged_batches_require_padding() {
    let program = compile(json!([{"op":"decode"},spectrum(8,1,8)]), features_output()).unwrap();
    for samples in [vec![], vec![0.5]] {
        let result = program
            .run(
                &[&encoded(&samples)],
                onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
            )
            .unwrap();
        assert_eq!(result[0].shape, [1, 5, samples.len() as i64]);
        assert!(floats(&result[0]).iter().all(|v| v.is_finite()));
        let expected = reference(&samples, 8, 1, 8, "power", "hann");
        for (actual, expected) in floats(&result[0]).iter().zip(expected) {
            assert!((f64::from(*actual) - expected).abs() < 1e-5);
        }
    }
    let error = program
        .run(
            &[&encoded(&[0.5]), &encoded(&[0.5; 2])],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("ragged") && error.contains("pad"), "{error}");
}

#[test]
fn every_explicit_projection_and_window_matches_independent_dft() {
    let samples = [
        -0.5, 0.25, 0.5, -0.25, 0.0, 0.5, 0.25, -0.5, 0.25, 0.5, 0.0, -0.5, 0.25, 0.5, -0.25, 0.5,
        0.0,
    ];
    for mode in ["power", "magnitude"] {
        for window_kind in ["hann", "hamming", "rectangular"] {
            for (fft, hop, window) in [(3, 2, 3), (8, 3, 5), (12, 4, 10), (6, 1, 1)] {
                let mut spec = spectrum(fft, hop, window);
                spec["mode"] = json!(mode);
                spec["window"] = json!(window_kind);
                let program = compile(json!([{"op":"decode"},spec]), features_output()).unwrap();
                let result = program
                    .run(
                        &[&encoded(&samples)],
                        onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
                    )
                    .unwrap();
                assert_eq!(
                    result[0].shape,
                    [1, (fft / 2 + 1) as i64, (samples.len() / hop) as i64]
                );
                for (actual, expected) in floats(&result[0]).iter().zip(reference(
                    &samples,
                    fft,
                    hop,
                    window,
                    mode,
                    window_kind,
                )) {
                    assert!(
                        (f64::from(*actual) - expected).abs() < 1e-5 * (1.0 + expected),
                        "{mode}/{window_kind} geometry {fft}/{hop}/{window}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}
#[test]
fn magnitude_remains_finite_when_squared_power_would_overflow() {
    let build = |mode: &str, scale: f64| {
        let mut spec = spectrum(8, 1, 8);
        spec["mode"] = json!(mode);
        compile(
            json!([{"op":"decode"},{"op":"rescale","scale":scale},spec]),
            features_output(),
        )
        .unwrap()
    };
    let bytes = encoded(&[0.25; 8]);
    let scale = 1e20;
    let result = build("magnitude", scale)
        .run(
            &[&bytes],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap();
    let values = floats(&result[0]);
    assert!(values.iter().all(|value| value.is_finite()));
    // Periodic Hann(8) has DC gain 4, and the encoded 0.25 is exact.
    let expected = scale as f32;
    assert!(
        (values[0] - expected).abs() < expected * 1e-5,
        "{} != {expected}",
        values[0]
    );
    let error = build("power", scale)
        .run(
            &[&bytes],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("power spectrum")
            && error.contains("nonfinite")
            && error.contains("rescale"),
        "{error}"
    );
    for mode in ["power", "magnitude"] {
        let error = build(mode, f64::from(f32::MAX))
            .run(
                &[&encoded(&[0.5; 8])],
                onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
            )
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(mode) && error.contains("nonfinite") && error.contains("rescale"),
            "{error}"
        );
    }
}
