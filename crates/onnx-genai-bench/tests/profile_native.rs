#![cfg(feature = "bench-native")]

use std::path::Path;
use std::process::Command;

#[test]
fn native_cpu_profile_reports_throughput() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../onnx-runtime-session/tests/fixtures/bert_toy");
    let output = Command::new(env!("CARGO_BIN_EXE_profile_native"))
        .args([
            "--model",
            model.to_str().unwrap(),
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
        throughput.is_some_and(|value| value > 0.0) && stdout.contains("runs/s"),
        "missing throughput number:\n{stdout}"
    );
}
