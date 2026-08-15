//! End-to-end coverage for `onnx-genai transcribe`, including live streaming.
//!
//! Drives the real binary against the committed `tiny-whisper` fixture, whose
//! decoder always predicts the same token, so the assertions are about the
//! *plumbing*: segmentation, timestamps, formats, stream framing, and the
//! promise that a piped stream is transcribed as it arrives rather than after
//! it ends.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model() -> PathBuf {
    repository_root().join("tests/fixtures/tiny-whisper")
}

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_onnx-genai"))
        .args(arguments)
        .env("ONNX_GENAI_EP", "cpu")
        .output()
        .expect("the onnx-genai binary must run")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Three 0.2s bursts separated by 0.2s of silence, at 16 kHz mono.
fn bursts() -> Vec<i16> {
    let rate = 16_000_f32;
    let tone = |count: usize| -> Vec<i16> {
        (0..count)
            .map(|index| {
                (16_000.0 * (2.0 * std::f32::consts::PI * 440.0 * index as f32 / rate).sin()) as i16
            })
            .collect()
    };
    let mut samples = tone(3_200);
    samples.extend(std::iter::repeat_n(0, 3_200));
    samples.extend(tone(3_200));
    samples.extend(std::iter::repeat_n(0, 3_200));
    samples.extend(tone(3_200));
    samples
}

fn pcm_bytes() -> Vec<u8> {
    bursts()
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect()
}

fn wav_bytes() -> Vec<u8> {
    let samples: Vec<f32> = bursts()
        .iter()
        .map(|sample| f32::from(*sample) / 32768.0)
        .collect();
    onnx_genai::preprocess::audio::encode_wav_pcm16(&samples, 16_000, 1)
        .expect("the fixture waveform must encode")
}

/// Feed `input` to `transcribe` on stdin and return its output.
fn pipe(arguments: &[&str], input: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_onnx-genai"))
        .args(arguments)
        .env("ONNX_GENAI_EP", "cpu")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the onnx-genai binary must start");
    child
        .stdin
        .as_mut()
        .expect("stdin is piped")
        .write_all(input)
        .expect("the stream must be accepted");
    child.wait_with_output().expect("transcribe must exit")
}

#[test]
fn transcribes_a_wav_file() {
    let output = run(&[
        "transcribe",
        model().to_str().unwrap(),
        model().join("tiny.wav").to_str().unwrap(),
        "--max-new-tokens",
        "2",
    ]);

    assert!(
        output.status.success(),
        "transcription failed: {}",
        stderr(&output)
    );
    assert!(
        !stdout(&output).trim().is_empty(),
        "a transcript is expected"
    );
    // The real-time factor tells a user whether the model keeps up with audio.
    assert!(
        stderr(&output).contains("real-time factor"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn logs_never_pollute_the_transcript_on_stdout() {
    let output = run(&[
        "transcribe",
        model().to_str().unwrap(),
        model().join("tiny.wav").to_str().unwrap(),
        "--max-new-tokens",
        "1",
    ]);

    // stdout is routinely piped into a file or another tool, so diagnostics
    // must go to stderr only.
    for line in stdout(&output).lines() {
        assert!(!line.contains("INFO"), "log line on stdout: {line}");
        assert!(
            !line.contains("Loading model"),
            "log line on stdout: {line}"
        );
    }
}

#[test]
fn json_transcripts_carry_indices_and_timings() {
    let output = pipe(
        &[
            "transcribe",
            model().to_str().unwrap(),
            "-",
            "--sample-rate",
            "16000",
            "--max-new-tokens",
            "1",
            "--format",
            "json",
        ],
        &pcm_bytes(),
    );

    assert!(output.status.success(), "failed: {}", stderr(&output));
    let segments: Vec<serde_json::Value> = stdout(&output)
        .lines()
        .map(|line| serde_json::from_str(line).expect("each line must be a JSON object"))
        .collect();
    assert!(segments.len() > 1, "the bursts must produce many segments");

    for (position, segment) in segments.iter().enumerate() {
        assert_eq!(segment["index"], position as u64);
        let start = segment["start"].as_f64().expect("start");
        let end = segment["end"].as_f64().expect("end");
        assert!(end > start, "segment {position}: {start} -> {end}");
        assert!(!segment["text"].as_str().expect("text").is_empty());
    }
    // Timestamps must advance monotonically through the stream.
    for pair in segments.windows(2) {
        assert!(pair[1]["start"].as_f64().unwrap() >= pair[0]["start"].as_f64().unwrap());
    }
}

#[test]
fn silence_between_bursts_is_skipped_rather_than_transcribed() {
    let output = pipe(
        &[
            "transcribe",
            model().to_str().unwrap(),
            "-",
            "--sample-rate",
            "16000",
            "--max-new-tokens",
            "1",
            "--format",
            "json",
            "--silence-seconds",
            "0.05",
        ],
        &pcm_bytes(),
    );

    let segments: Vec<serde_json::Value> = stdout(&output)
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let spoken: f64 = segments
        .iter()
        .map(|segment| segment["end"].as_f64().unwrap() - segment["start"].as_f64().unwrap())
        .sum();

    // The clip is 1.0s long but only 0.6s is speech; the 0.4s of silence must
    // not be sent to the model.
    assert!(
        spoken < 0.8,
        "silence appears to have been transcribed: {spoken}s of {}s",
        1.0
    );
    // A gap must be visible as a jump between a segment's end and the next start.
    assert!(
        segments.windows(2).any(|pair| {
            pair[1]["start"].as_f64().unwrap() - pair[0]["end"].as_f64().unwrap() > 0.1
        }),
        "expected a silence gap in {segments:?}"
    );
}

#[test]
fn a_wav_stream_declares_its_own_sample_rate() {
    // No --sample-rate: the header must supply it.
    let output = pipe(
        &[
            "transcribe",
            model().to_str().unwrap(),
            "-",
            "--max-new-tokens",
            "1",
        ],
        &wav_bytes(),
    );

    assert!(output.status.success(), "failed: {}", stderr(&output));
    assert!(
        stderr(&output).contains("16000 Hz"),
        "the header's rate must be reported: {}",
        stderr(&output)
    );
    assert!(!stdout(&output).trim().is_empty());
}

#[test]
fn raw_pcm_and_wav_streams_transcribe_identically() {
    let arguments = |rate: &'static str| {
        vec![
            "transcribe".to_string(),
            model().to_str().unwrap().to_string(),
            "-".to_string(),
            "--max-new-tokens".to_string(),
            "1".to_string(),
            "--format".to_string(),
            "json".to_string(),
            "--sample-rate".to_string(),
            rate.to_string(),
        ]
    };
    fn borrow(owned: &[String]) -> Vec<&str> {
        owned.iter().map(String::as_str).collect()
    }

    let raw_arguments = arguments("16000");
    let raw = pipe(&borrow(&raw_arguments), &pcm_bytes());
    // A deliberately wrong flag proves the WAV header overrides it.
    let wav_arguments = arguments("8000");
    let wav = pipe(&borrow(&wav_arguments), &wav_bytes());

    assert_eq!(
        stdout(&raw),
        stdout(&wav),
        "the same audio must transcribe the same whether or not it is framed"
    );
}

#[test]
fn srt_output_is_well_formed() {
    let output = run(&[
        "transcribe",
        model().to_str().unwrap(),
        model().join("tiny.wav").to_str().unwrap(),
        "--max-new-tokens",
        "1",
        "--format",
        "srt",
    ]);

    let text = stdout(&output);
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("1"), "subtitles are numbered from one");
    let timing = lines.next().expect("a timing line");
    assert!(timing.contains(" --> "), "timing line: {timing}");
    // HH:MM:SS,mmm on both sides.
    for stamp in timing.split(" --> ") {
        assert_eq!(stamp.len(), 12, "malformed timestamp: {stamp}");
        assert_eq!(&stamp[2..3], ":");
        assert_eq!(&stamp[8..9], ",");
    }
}

#[test]
fn a_model_without_an_audio_input_is_rejected() {
    let output = run(&[
        "transcribe",
        repository_root()
            .join("tests/fixtures/tiny-llm")
            .to_str()
            .unwrap(),
        model().join("tiny.wav").to_str().unwrap(),
    ]);

    assert!(!output.status.success(), "a text model cannot transcribe");
    let message = stderr(&output);
    assert!(message.contains("What:"), "message: {message}");
    assert!(message.contains("How:"), "message: {message}");
}

#[test]
fn a_segment_longer_than_the_model_window_is_rejected() {
    let output = run(&[
        "transcribe",
        model().to_str().unwrap(),
        model().join("tiny.wav").to_str().unwrap(),
        "--segment-seconds",
        "30",
    ]);

    assert!(!output.status.success(), "the window is the hard maximum");
    let message = stderr(&output);
    assert!(message.contains("What:"), "message: {message}");
    // The message must name the model's actual window rather than a guess.
    assert!(message.contains("0.08s"), "message: {message}");
}

#[test]
fn an_unreadable_audio_file_names_the_path() {
    let output = run(&[
        "transcribe",
        model().to_str().unwrap(),
        "definitely-not-here.wav",
    ]);

    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("definitely-not-here.wav"),
        "message: {}",
        stderr(&output)
    );
}
