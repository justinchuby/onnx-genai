#![cfg(feature = "bench-native")]

use std::process::Command;

#[test]
fn native_cpu_synthetic_profile_reports_throughput() {
    let output = Command::new(env!("CARGO_BIN_EXE_profile_native"))
        .args([
            "--synthetic",
            "--tokens",
            "2",
            "--warmups",
            "1",
            "--runs",
            "1",
            "--ep",
            "cpu",
        ])
        .output()
        .expect("run profile_native");

    assert!(
        output.status.success(),
        "profile_native failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let throughput = stdout
        .lines()
        .find_map(|line| line.strip_prefix("throughput: "))
        .and_then(|line| line.split_whitespace().next())
        .and_then(|value| value.parse::<f64>().ok());
    assert!(
        throughput.is_some_and(|value| value > 0.0) && stdout.contains("tok/s"),
        "missing throughput number:\n{stdout}"
    );
    let header = stdout
        .lines()
        .find(|line| line.starts_with("profile_native: model="))
        .expect("profile header");
    assert!(
        !header.contains("backend="),
        "default native header changed:\n{header}"
    );
}

#[test]
fn native_cpu_synthetic_profile_reports_decode_stages_when_enabled() {
    let output = Command::new(env!("CARGO_BIN_EXE_profile_native"))
        .env("ONNX_GENAI_PROFILE", "1")
        .args([
            "--synthetic",
            "--tokens",
            "2",
            "--warmups",
            "1",
            "--runs",
            "1",
            "--ep",
            "cpu",
        ])
        .output()
        .expect("run profile_native with stage profiling");

    assert!(
        output.status.success(),
        "profile_native failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("stage")
            && stdout.contains("us/token")
            && stdout.contains("loop.next_logits")
            && stdout.contains("loop.sampling"),
        "missing per-stage decode profile:\n{stdout}"
    );
}
