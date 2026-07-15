#![cfg(feature = "bench-native")]

use std::path::Path;
use std::process::Command;

#[test]
fn native_cpu_profile_reports_throughput_or_actionable_op_gap() {
    let model =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-gemma4-assistant");
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

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("native CPU Gather lacks Int64 data support"),
            "profile_native failed without an actionable native-op diagnosis:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            stderr
        );
        return;
    }
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
}
