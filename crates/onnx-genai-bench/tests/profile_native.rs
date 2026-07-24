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

/// End-to-end token-exactness guard for `numa-split`: on a real int4 model the
/// `numa-split` two-level layout row-shards every projection's output rows, and
/// because a GEMV row is an independent dot product over the whole K dimension,
/// the concatenated shards must reproduce the flat/`compact` result bit-for-bit.
/// This asserts the *generated token sequence* is identical between `compact`
/// and `numa-split` decode of the same greedy prompt.
///
/// `#[ignore]` + env-gated on a real model path (`ONNX_GENAI_NUMA_E2E_MODEL`),
/// since it needs a downloaded model and a multi-node host to exercise the
/// split; on a single-node host `numa-split` falls back to the flat path, so the
/// sequences still match (the test stays valid, it just does not exercise the
/// cross-node join). Run with:
///   ONNX_GENAI_NUMA_E2E_MODEL=/path/to/model_dir \
///     cargo test -p onnx-genai-bench --features bench-native,mlas \
///     --test profile_native -- --ignored numa_split_tokens_match_compact
#[test]
#[ignore = "needs a real int4 model via ONNX_GENAI_NUMA_E2E_MODEL and a multi-node host"]
fn numa_split_tokens_match_compact_end_to_end() {
    let Ok(model) = std::env::var("ONNX_GENAI_NUMA_E2E_MODEL") else {
        eprintln!("ONNX_GENAI_NUMA_E2E_MODEL unset; skipping numa-split e2e token-exactness test");
        return;
    };

    let tokens_for = |affinity: &str| -> String {
        let output = Command::new(env!("CARGO_BIN_EXE_profile_native"))
            .env("ONNX_GENAI_CPU_DECODE_AFFINITY", affinity)
            .args([
                "--model",
                &model,
                "--steady",
                "--tokens",
                "48",
                "--decode-skip",
                "8",
                "--warmups",
                "0",
                "--runs",
                "1",
                "--backend",
                "native",
                "--prompt",
                "The capital of France is",
            ])
            .output()
            .expect("run profile_native");
        assert!(
            output.status.success(),
            "profile_native ({affinity}) failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout
            .lines()
            .find_map(|line| line.strip_prefix("generated_token_ids: "))
            .map(str::to_string)
            .unwrap_or_else(|| panic!("no generated_token_ids in output:\n{stdout}"))
    };

    let compact = tokens_for("compact");
    let numa_split = tokens_for("numa-split");
    assert_eq!(
        compact, numa_split,
        "numa-split decode diverged from compact (row-sharding must be bit-exact):\n\
         compact:    {compact}\n\
         numa-split: {numa_split}"
    );
}
