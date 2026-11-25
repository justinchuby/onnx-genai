//! Public process boundary: native frame decoding, stream/file segmentation and profiles.
#![cfg(feature = "native-backend")]
use serde_json::Value;
use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};
fn model(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows")
        .join(name)
}
fn command(model: &Path, source: &Path, extra: &[&str], profile: Option<&Path>) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_onnx-genai"));
    if let Some(path) = profile {
        command.arg("--profile-json").arg(path);
    }
    command
        .arg("transcribe")
        .arg(model)
        .arg(source)
        .args([
            "--format",
            "json",
            "--min-segment-seconds",
            "0",
            "--cpu-cores",
            "1",
        ])
        .args(extra)
        .env_remove("ONNX_GENAI_BACKEND")
        .env("ONNX_GENAI_EP", "cpu")
        .env(
            "ONNX_GENAI_ORT_LIB",
            "/missing/nxrt-native-transcription-reference-test/onnxruntime",
        );
    command
}
fn transcripts(output: Output) -> Vec<Value> {
    assert!(
        output.status.success(),
        "CLI failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
#[test]
fn native_frame_transcription_process_handles_files_wav_stdin_and_raw_pcm_without_ort() {
    let scratch = tempfile::tempdir().unwrap();
    let samples = [-0.5, -0.5, 0.125, -0.5, 0.0, 0.5, 0.5, 0.125, 0.5];
    let wav = onnx_genai::preprocess::audio::encode_wav_pcm16(&samples, 8000, 1).unwrap();
    let file = scratch.path().join("clip.wav");
    std::fs::write(&file, &wav).unwrap();
    let profile = scratch.path().join("profile.json");
    let expected = transcripts(
        command(&model("transcription_ctc"), &file, &[], Some(&profile))
            .output()
            .unwrap(),
    );
    assert_eq!(expected.len(), 1);
    assert_eq!(expected[0]["text"], "bb aa");
    assert_eq!(expected[0]["index"], 0);
    let status: Value = serde_json::from_slice(&std::fs::read(profile).unwrap()).unwrap();
    assert_eq!(status["decode_backend"], "native");
    assert_eq!(status["execution_provider"], "native-cpu");
    let raw = samples
        .iter()
        .flat_map(|v| ((*v * 32767.0_f32).round() as i16).to_le_bytes())
        .collect::<Vec<_>>();
    for bytes in [&wav, &raw] {
        let mut child = command(
            &model("transcription_ctc"),
            Path::new("-"),
            &["--sample-rate", "8000"],
            None,
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
        child.stdin.take().unwrap().write_all(bytes).unwrap();
        assert_eq!(transcripts(child.wait_with_output().unwrap()), expected);
    }
    let truncated = transcripts(
        command(
            &model("transcription_ctc"),
            &file,
            &["--max-new-tokens", "1"],
            None,
        )
        .output()
        .unwrap(),
    );
    assert_eq!(truncated[0]["text"], "b");
}
#[test]
fn native_generative_mp3_process_uses_the_authored_window_for_every_segment() {
    let input =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/audio/stereo-tone.mp3");
    let rows = transcripts(
        command(&model("transcription_tokens"), &input, &[], None)
            .output()
            .unwrap(),
    );
    assert_eq!(
        rows.len(),
        4,
        "100 ms recording uses three 32 ms windows and one tail"
    );
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row["index"], index);
        assert_eq!(row["text"], "hello");
    }
    assert_eq!(rows[0]["end"], 0.032);
    assert_eq!(rows[3]["start"], 0.096);
    assert_eq!(rows[3]["end"], 0.1);
}
