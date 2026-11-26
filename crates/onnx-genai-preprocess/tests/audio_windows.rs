//! Admission follows executable waveform length and validity, not mel heuristics.
use onnx_genai_preprocess::audio::{AudioProgram, AudioTensorData, encode_wav_pcm16};
use serde_json::{Value, json};
fn program(transforms: Value, outputs: Value) -> AudioProgram {
    AudioProgram::from_program(
        &serde_json::from_value(json!({"transforms":transforms,"outputs":outputs})).unwrap(),
    )
    .unwrap()
}
fn wave(source: &str) -> Value {
    json!({"name":"wave","source":source,"content":"waveform","dtype":"float32"})
}
fn length_program(window: bool) -> AudioProgram {
    let mut transforms = json!([
        {"op":"decode","outputs":["raw"]},
        {"op":"resample","sample_rate":7,"outputs":["a"]},
        {"op":"resample","sample_rate":13,"outputs":["b"]}
    ])
    .as_array()
    .unwrap()
    .clone();
    if window {
        transforms.push(json!({"op":"trim","target_length":9,"outputs":["limited"]}));
    }
    transforms.push(json!({"op":"emit_valid_samples","outputs":["count"]}));
    program(
        json!(transforms),
        json!([{"name":"count","source":"count","content":"valid_samples","dtype":"int64"}]),
    )
}
#[test]
fn window_boundary_matches_actual_multi_resample_execution_at_different_source_rates() {
    let bounded = length_program(true);
    let reference = length_program(false);
    for rate in [5, 11, 19, 37] {
        let maximum = bounded.maximum_input_samples(rate).unwrap().unwrap().get();
        let mut observed = 0;
        for samples in 1..80 {
            let bytes = encode_wav_pcm16(&vec![0.25; samples], rate, 1).unwrap();
            let actual = bounded
                .run(
                    &[&bytes],
                    onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
                )
                .unwrap();
            let untrimmed = reference
                .run(
                    &[&bytes],
                    onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
                )
                .unwrap();
            let fits = actual[0].data == untrimmed[0].data;
            if fits {
                observed = samples;
            }
            assert_eq!(
                samples <= maximum,
                fits,
                "{samples} samples at {rate} Hz, maximum {maximum}"
            );
        }
        assert_eq!(maximum, observed);
    }
}
#[test]
fn limits_follow_live_branches_and_ignore_unpublished_truncation() {
    let transforms = json!([
        {"op":"decode","outputs":["raw"]},
        {"op":"trim","target_length":1,"outputs":["unused"]},
        {"op":"pad","inputs":["raw"],"mode":"fixed_window","target_length":12,"outputs":["a"]},
        {"op":"resample","inputs":["raw"],"sample_rate":16000,"outputs":["fast"]},
        {"op":"trim","target_length":14,"outputs":["b"]}
    ]);
    let a = program(transforms.clone(), json!([wave("a")]));
    assert_eq!(a.maximum_input_samples(8000).unwrap().unwrap().get(), 12);
    let both = program(
        transforms,
        json!([wave("a"),{"name":"other","source":"b","content":"waveform","dtype":"float32"}]),
    );
    assert_eq!(both.maximum_input_samples(8000).unwrap().unwrap().get(), 7);
    assert_eq!(
        both.maximum_input_samples(16000).unwrap().unwrap().get(),
        12
    );
}
#[test]
fn windows_before_resampling_bind_source_samples_and_padding_is_not_real_audio() {
    let p = program(
        json!([
            {"op":"decode","outputs":["raw"]},
            {"op":"pad","target_length":32,"outputs":["padded"]},
            {"op":"trim","target_length":12,"outputs":["clipped"]},
            {"op":"resample","sample_rate":8000,"outputs":["out"]}
        ]),
        json!([wave("out")]),
    );
    assert_eq!(p.maximum_input_samples(16000).unwrap().unwrap().get(), 12);
    assert_eq!(p.maximum_input_samples(8000).unwrap().unwrap().get(), 12);
    let bytes = encode_wav_pcm16(&[0.25; 12], 8000, 1).unwrap();
    let actual = p
        .run(
            &[&bytes],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap();
    let AudioTensorData::Fp32(samples) = &actual[0].data else {
        panic!("float waveform")
    };
    assert_eq!(samples, &[0.25; 12]);
}
#[test]
fn unbounded_and_unrepresentable_windows_are_explicit() {
    let unbounded = program(
        json!([{"op":"decode","outputs":["raw"]},{"op":"pad","target_length":12,"outputs":["out"]}]),
        json!([wave("out")]),
    );
    assert!(unbounded.maximum_input_samples(8000).unwrap().is_none());
    assert!(
        unbounded
            .maximum_input_samples(0)
            .unwrap_err()
            .to_string()
            .contains("sample rate")
    );
    let tiny = program(
        json!([{"op":"decode","outputs":["raw"]},{"op":"resample","sample_rate":48000,"outputs":["fast"]},{"op":"trim","target_length":1,"outputs":["out"]}]),
        json!([wave("out")]),
    );
    let error = tiny.maximum_input_samples(1000).unwrap_err().to_string();
    assert!(
        error.contains("one source sample") && error.contains("enlarge"),
        "{error}"
    );
}

#[test]
fn fixed_feature_windows_follow_configured_hops_without_a_special_frame_count() {
    use onnx_genai_preprocess::audio::{LogMelConfig, LogMelExtractor};
    use std::num::NonZeroUsize;
    for (hop, frames) in [(53, 5), (64, 4), (97, 3)] {
        let extractor = LogMelExtractor::from_config(LogMelConfig {
            n_mels: 7,
            sample_rate: 8000,
            n_fft: 128,
            hop_length: hop,
            win_length: 96,
        })
        .unwrap();
        for length in [0, 80, 400] {
            let samples = (0..length)
                .map(|i| (i as f32 * 0.2).sin() * 0.5)
                .collect::<Vec<_>>();
            let actual = extractor
                .extract_fixed_frames(&samples, 8000, NonZeroUsize::new(frames).unwrap())
                .unwrap();
            let mut window = samples;
            window.resize(frames * hop, 0.0);
            let reference = extractor.extract(&window, 8000).unwrap();
            assert_eq!(actual, reference);
            assert_eq!(actual.shape(), [1, 7, frames]);
        }
        for source_rate in [8000, 11025, 16000, 44100] {
            let maximum = extractor
                .maximum_input_samples_for_frames(source_rate, NonZeroUsize::new(frames).unwrap())
                .unwrap()
                .get();
            let counter = program(
                json!([
                    {"op":"decode","outputs":["raw"]},
                    {"op":"resample","sample_rate":8000,"outputs":["resampled"]},
                    {"op":"emit_valid_samples","outputs":["count"]}
                ]),
                json!([{"name":"count","source":"count","content":"valid_samples","dtype":"int64"}]),
            );
            for (count, fits) in [(maximum, true), (maximum + 1, false)] {
                let bytes = encode_wav_pcm16(&vec![0.5; count], source_rate, 1).unwrap();
                let actual = counter
                    .run(
                        &[&bytes],
                        onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
                    )
                    .unwrap();
                let AudioTensorData::Int64(samples) = &actual[0].data else {
                    panic!("sample count")
                };
                assert_eq!(
                    samples[0] <= (frames * hop) as i64,
                    fits,
                    "source {source_rate}, frames {frames}, hop {hop}, count {count}"
                );
            }
        }
        let error = extractor
            .extract_fixed_frames(&[], 8000, NonZeroUsize::new(usize::MAX).unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("overflows") && error.contains("reduce"),
            "{error}"
        );
    }
}
