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
            error.contains("MP3") && error.contains("provide valid"),
            "{error}"
        );
    }
    let wav = encode_wav_pcm16(&[-0.5, 0.0, 0.5], 8000, 1).unwrap();
    let actual = decode_audio(&wav).unwrap();
    assert_eq!(actual.sample_rate, 8000);
    assert_eq!(actual.samples, [-0.5, 0.0, 0.5]);
}
