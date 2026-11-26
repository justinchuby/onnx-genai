//! Named processor-local values, explicit normalization and waveform validity.
use onnx_genai_preprocess::audio::{
    AudioProgram, AudioTensorData, LogMelConfig, LogMelExtractor, NamedAudioTensor,
    encode_wav_pcm16,
};
use serde_json::{Value, json};
fn program(
    transforms: Value,
    outputs: Value,
) -> Result<AudioProgram, onnx_genai_preprocess::audio::AudioPreprocessError> {
    let p = serde_json::from_value(json!({"transforms":transforms,"outputs":outputs})).unwrap();
    AudioProgram::from_program(&p)
}
fn wav(values: &[f32]) -> Vec<u8> {
    encode_wav_pcm16(values, 8000, 1).unwrap()
}
fn output(name: &str, source: &str, content: &str, dtype: &str) -> Value {
    json!({"name":name,"source":source,"content":content,"dtype":dtype})
}
fn floats(t: &NamedAudioTensor) -> &[f32] {
    let AudioTensorData::Fp32(data) = &t.data else {
        panic!("expected float output")
    };
    data
}
#[test]
fn waveform_branches_and_implicit_previous_values_are_independent() {
    let p = program(
        json!([
            {"op":"decode","outputs":["raw"]},
            {"op":"rescale","scale":2.0,"outputs":["doubled"]},
            {"op":"rescale","inputs":["raw"],"outputs":["halved"],"scale":0.5}
        ]),
        json!([
            output("raw", "raw", "waveform", "float32"),
            output("two", "doubled", "waveform", "float32"),
            output("half", "halved", "waveform", "float32")
        ]),
    )
    .unwrap();
    let actual = p
        .run(
            &[&wav(&[-0.5, 0.0, 0.5])],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap();
    assert_eq!(floats(&actual[0]), [-0.5, 0.0, 0.5]);
    assert_eq!(floats(&actual[1]), [-1.0, 0.0, 1.0]);
    assert_eq!(floats(&actual[2]), [-0.25, 0.0, 0.25]);
}
#[test]
fn batch_maximum_left_padding_keeps_masks_and_sample_lengths_aligned() {
    let p = program(
        json!([
            {"op":"decode","outputs":["raw"]},
            {"op":"pad","mode":"left","pad_value":0.75,"outputs":["padded"]},
            {"op":"emit_validity_mask","inputs":["padded"],"outputs":["mask"]},
            {"op":"emit_valid_samples","inputs":["padded"],"outputs":["length"]},
            {"op":"emit_valid_frames","inputs":["padded"],"hop_length":2,"outputs":["frames"]}
        ]),
        json!([
            output("wave", "padded", "waveform", "float32"),
            output("mask", "mask", "validity_mask", "int64"),
            output("length", "length", "valid_samples", "int64"),
            output("frames", "frames", "valid_frames", "int64")
        ]),
    )
    .unwrap();
    let actual = p
        .run(
            &[&wav(&[0.25, 0.5]), &wav(&[-0.5, 0.0, 0.25, 0.5, 0.75])],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap();
    assert_eq!(actual[0].shape, [2, 5]);
    assert_eq!(
        floats(&actual[0]),
        [
            0.75,
            0.75,
            0.75,
            0.25,
            0.5,
            -0.5,
            0.0,
            0.25,
            0.5,
            24575.0 / 32768.0
        ]
    );
    assert_eq!(
        actual[1].data,
        AudioTensorData::Int64(vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    );
    assert_eq!(actual[2].data, AudioTensorData::Int64(vec![2, 5]));
    assert_eq!(actual[3].data, AudioTensorData::Int64(vec![1, 2]));
    let empty = p
        .run(
            &[&wav(&[]), &wav(&[0.5, 0.5])],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap();
    assert_eq!(empty[2].data, AudioTensorData::Int64(vec![0, 2]));
    assert_eq!(empty[3].data, AudioTensorData::Int64(vec![0, 1]));
}
#[test]
fn normalization_uses_valid_samples_and_preserves_padding() {
    let p = program(
        json!([
            {"op":"decode","outputs":["raw"]},
            {"op":"pad","mode":"right","target_length":5,"pad_value":0.75,"outputs":["padded"]},
            {"op":"zero_mean_unit_variance","epsilon":1e-7,"outputs":["normal"]},
            {"op":"trim","target_length":1,"outputs":["trimmed"]},
            {"op":"emit_sample_lengths","outputs":["length"]}
        ]),
        json!([
            output("padded", "padded", "waveform", "float32"),
            output("normal", "normal", "waveform", "float32"),
            output("trimmed", "trimmed", "waveform", "float32"),
            output("length", "length", "sample_lengths", "int64")
        ]),
    )
    .unwrap();
    let tensors = p
        .run(
            &[&wav(&[-0.5, 0.5])],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap();
    assert_eq!(floats(&tensors[0]), [-0.5, 0.5, 0.75, 0.75, 0.75]);
    assert!((floats(&tensors[1])[0] + 1.0).abs() < 1e-6);
    assert!((floats(&tensors[1])[1] - 1.0).abs() < 1e-6);
    assert_eq!(&floats(&tensors[1])[2..], [0.75, 0.75, 0.75]);
    assert_eq!(tensors[2].shape, [1, 1]);
    assert_eq!(tensors[3].data, AudioTensorData::Int64(vec![1]));
}
#[test]
fn raw_log_mel_and_explicit_normalization_publish_different_sources() {
    let p=program(json!([
        {"op":"decode","outputs":["wave"]},
        {"op":"log_mel","sample_rate":8000,"num_mel_bins":4,"n_fft":64,"hop_length":16,"outputs":["log"]},
        {"op":"normalize","mode":"whisper_log_mel","outputs":["normal"]}
    ]),json!([output("raw","log","audio_features","float32"),output("normal","normal","audio_features","float32")])).unwrap();
    let samples = (0..128)
        .map(|i| (i as f32 * 0.31).sin() * 0.3)
        .collect::<Vec<_>>();
    let encoded = wav(&samples);
    let tensors = p
        .run(
            &[&encoded],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap();
    assert_ne!(floats(&tensors[0]), floats(&tensors[1]));
    let decoded = onnx_genai_preprocess::audio::decode_audio(
        &encoded,
        onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
    )
    .unwrap();
    let expected = LogMelExtractor::from_config(LogMelConfig {
        n_mels: 4,
        sample_rate: 8000,
        n_fft: 64,
        hop_length: 16,
        win_length: 64,
    })
    .unwrap()
    .extract(&decoded.samples, 8000)
    .unwrap();
    assert_eq!(floats(&tensors[1]), expected.data);
}
#[test]
fn resampling_branches_keep_their_own_lengths() {
    let p = program(
        json!([
            {"op":"decode","outputs":["raw"]},
            {"op":"resample","sample_rate":16000,"outputs":["upsampled"]},
            {"op":"emit_valid_samples","outputs":["up_length"]},
            {"op":"emit_valid_samples","inputs":["raw"],"outputs":["raw_length"]}
        ]),
        json!([
            output("up", "upsampled", "waveform", "float32"),
            output("up_length", "up_length", "valid_samples", "int64"),
            output("raw_length", "raw_length", "valid_samples", "int64")
        ]),
    )
    .unwrap();
    let tensors = p
        .run(
            &[&wav(&[0.0, 0.25, 0.5, 0.75])],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap();
    assert_eq!(tensors[0].shape, [1, 8]);
    assert_eq!(tensors[1].data, AudioTensorData::Int64(vec![8]));
    assert_eq!(tensors[2].data, AudioTensorData::Int64(vec![4]));
}
#[test]
fn wrong_sources_types_lengths_and_parameters_fail_before_execution() {
    for case in 0..9 {
        let mut transforms = json!([{"op":"decode","outputs":["raw"]},{"op":"rescale","scale":2.0,"inputs":["raw"],"outputs":["scaled"]}]);
        let mut out = output("out", "scaled", "waveform", "float32");
        match case {
            0 => transforms[1]["inputs"] = json!(["missing"]),
            1 => transforms[1]["outputs"] = json!(["raw"]),
            2 => out["source"] = json!("missing"),
            3 => out["dtype"] = json!("int64"),
            4 => out["content"] = json!("audio_features"),
            5 => transforms[1]["op"] = json!("spectrogram"),
            6 => {
                transforms[1]["op"] = json!("emit_valid_frames");
            }
            7 => {
                transforms[1]["op"] = json!("zero_mean_unit_variance");
                transforms[1]["epsilon"] = json!(0.0);
            }
            8 => {
                transforms[1]["op"] = json!("emit_valid_samples");
                out = output("out", "scaled", "valid_frames", "int64");
            }
            _ => unreachable!(),
        }
        let error = program(transforms, json!([out])).unwrap_err();
        assert!(error.to_string().contains("audio"), "case {case}: {error}");
        assert!(error.to_string().len() > 60, "case {case}: {error}");
    }
}
#[test]
fn log_mel_does_not_silently_change_input_sample_rate() {
    let p=program(json!([{"op":"decode","outputs":["wave"]},{"op":"log_mel","sample_rate":16000,"num_mel_bins":4,"outputs":["mel"]}]),json!([output("mel","mel","audio_features","float32")])).unwrap();
    let error = p
        .run(
            &[&wav(&[0.5; 400])],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("insert") || error.to_string().contains("declare resample"),
        "{error}"
    );
}
#[test]
fn resampling_an_empty_real_interval_keeps_it_empty_after_left_padding() {
    let p = program(
        json!([
            {"op":"decode","outputs":["wave"]},
            {"op":"pad","mode":"left","target_length":5,"pad_value":0.75,"outputs":["padded"]},
            {"op":"resample","sample_rate":11025,"outputs":["resampled"]},
            {"op":"emit_valid_samples","outputs":["length"]},
            {"op":"emit_validity_mask","inputs":["resampled"],"outputs":["mask"]}
        ]),
        json!([
            output("length", "length", "valid_samples", "int64"),
            output("mask", "mask", "validity_mask", "int64")
        ]),
    )
    .unwrap();
    let tensors = p
        .run(
            &[&wav(&[])],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap();
    assert_eq!(tensors[0].data, AudioTensorData::Int64(vec![0]));
    let AudioTensorData::Int64(mask) = &tensors[1].data else {
        panic!("expected mask")
    };
    assert!(mask.iter().all(|v| *v == 0));
}
#[test]
fn nonfinite_rescale_results_are_actionable_runtime_errors() {
    let p = program(
        json!([
            {"op":"decode","outputs":["wave"]},
            {"op":"rescale","scale":4.0,"outputs":["loud"]},
            {"op":"rescale","scale":f32::MAX,"outputs":["overflow"]}
        ]),
        json!([output("out", "overflow", "waveform", "float32")]),
    )
    .unwrap();
    let error = p
        .run(
            &[&wav(&[0.5])],
            onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("transform[2]") && error.to_string().contains("reduce scale"),
        "{error}"
    );
}
#[test]
fn empty_output_names_and_zero_rate_wav_are_rejected() {
    let error = program(
        json!([{"op":"decode","outputs":["wave"]}]),
        json!([output(" ", "wave", "waveform", "float32")]),
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("output.name is empty"),
        "{error}"
    );
    let mut encoded = wav(&[0.0, 0.5]);
    // PCM WAV fmt chunk: zero sample and byte rates are malformed metadata.
    encoded[24..28].copy_from_slice(&0u32.to_le_bytes());
    encoded[28..32].copy_from_slice(&0u32.to_le_bytes());
    let error = onnx_genai_preprocess::audio::decode_audio(
        &encoded,
        onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
    )
    .unwrap_err();
    assert!(error.to_string().contains("sample rate"), "{error}");
}
