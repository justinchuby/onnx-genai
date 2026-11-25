//! Content-based MP3 decoding against an independent ffmpeg waveform oracle.
use onnx_genai_preprocess::audio::{AudioProgram, AudioTensorData, decode_audio, encode_wav_pcm16};
use serde_json::json;
const MP3: &[u8] = include_bytes!("../../../tests/fixtures/audio/stereo-tone.mp3");
#[test]
fn mp3_gapless_stereo_mean_matches_independent_pcm_reference() {
    let actual = decode_audio(MP3).unwrap();
    assert_eq!(actual.sample_rate, 16000);
    let reference = include_bytes!("../../../tests/fixtures/audio/stereo-tone-reference.f32le")
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual.samples.len(), reference.len());
    assert_eq!(reference.len(), 1600, "gapless 100 ms fixture");
    let maximum = actual
        .samples
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(maximum < 0.0001, "MP3 decoder/downmix error {maximum}");
    let rms =
        (actual.samples.iter().map(|v| v * v).sum::<f32>() / actual.samples.len() as f32).sqrt();
    assert!(
        (0.20..0.30).contains(&rms),
        "arithmetic stereo mean RMS {rms}"
    );
}
#[test]
fn authored_decode_program_accepts_mp3_and_preserves_valid_samples() {
    let declaration = serde_json::from_value(json!({
        "transforms":[{"op":"decode","outputs":["wave"]},{"op":"emit_valid_samples","outputs":["count"]}],
        "outputs":[{"name":"wave","source":"wave","content":"waveform","dtype":"float32"},{"name":"count","source":"count","content":"valid_samples","dtype":"int64"}]
    })).unwrap();
    let program = AudioProgram::from_program(&declaration).unwrap();
    let outputs = program.run(&[MP3]).unwrap();
    assert_eq!(outputs[0].shape, [1, 1600]);
    let AudioTensorData::Fp32(actual) = &outputs[0].data else {
        panic!("waveform dtype")
    };
    assert_eq!(actual, &decode_audio(MP3).unwrap().samples);
    let AudioTensorData::Int64(count) = &outputs[1].data else {
        panic!("count dtype")
    };
    assert_eq!(count, &[1600]);
}
#[test]
fn malformed_encoded_audio_fails_actionably_and_pcm16_wav_is_unchanged() {
    for bytes in [&[][..], &b"ID3bad"[..], &MP3[..20]] {
        let error = decode_audio(bytes).unwrap_err().to_string();
        assert!(
            error.contains("encoded audio") && error.contains("supported audio"),
            "{error}"
        );
    }
    let wav = encode_wav_pcm16(&[-0.5, 0.0, 0.5], 8000, 1).unwrap();
    let actual = decode_audio(&wav).unwrap();
    assert_eq!(actual.sample_rate, 8000);
    assert_eq!(actual.samples, [-0.5, 0.0, 0.5]);
}
#[test]
fn symphonia_codec_matrix_matches_independent_pcm_references() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/audio/codec-matrix");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let mut failures = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["file"].as_str().unwrap();
        let decoded = match decode_audio(&std::fs::read(root.join(name)).unwrap()) {
            Ok(value) => value,
            Err(error) => {
                failures.push(format!("{name}: {error}"));
                continue;
            }
        };
        let reference = std::fs::read(root.join(case["reference"].as_str().unwrap()))
            .unwrap()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        if decoded.sample_rate as u64 != case["sample_rate"].as_u64().unwrap()
            || decoded.samples.len() != reference.len()
        {
            failures.push(format!(
                "{name}: {} Hz/{} frames, expected {}/{}",
                decoded.sample_rate,
                decoded.samples.len(),
                case["sample_rate"],
                reference.len()
            ));
            continue;
        }
        let maximum = decoded
            .samples
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let tolerance = case["max_error"].as_f64().unwrap() as f32;
        if maximum > tolerance {
            failures.push(format!(
                "{name}: maximum PCM error {maximum}, tolerance {tolerance}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
#[test]
fn float_wav_transport_preserves_precision_empty_clips_and_finite_guards() {
    use onnx_genai_preprocess::audio::encode_wav_f32;
    let values = [0.1234567, -0.0000001, 1.25, -1.5];
    let mut bytes = encode_wav_f32(&values, 8000, 1).unwrap();
    assert_eq!(decode_audio(&bytes).unwrap().samples, values);
    let data = bytes.windows(4).position(|chunk| chunk == b"data").unwrap() + 8;
    bytes[data..data + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(
        decode_audio(&bytes)
            .unwrap_err()
            .to_string()
            .contains("nonfinite")
    );
    for bytes in [
        encode_wav_f32(&[], 8000, 1).unwrap(),
        encode_wav_pcm16(&[], 8000, 1).unwrap(),
    ] {
        let decoded = decode_audio(&bytes).unwrap();
        assert_eq!(decoded.sample_rate, 8000);
        assert!(decoded.samples.is_empty());
    }
    assert!(encode_wav_f32(&[f32::NAN], 8000, 1).is_err());
    assert!(encode_wav_pcm16(&[f32::INFINITY], 8000, 1).is_err());
    assert!(
        encode_wav_f32(&[0.5; 3], 8000, 2)
            .unwrap_err()
            .to_string()
            .contains("interleaved")
    );
}
#[test]
fn unsupported_codec_preserves_the_decoder_cause_and_repair_guidance() {
    let bytes = include_bytes!("../../../tests/fixtures/audio/codec-matrix/unsupported-opus.ogg");
    let error = decode_audio(bytes).unwrap_err();
    assert!(
        std::error::Error::source(&error).is_some(),
        "codec error cause must survive"
    );
    let text = error.to_string();
    assert!(
        text.contains("codec") && text.contains("convert") && text.contains("WAV"),
        "{text}"
    );
}
