//! Checked-in graph fixtures must remain reviewable ONNX TextFormat.

use std::path::Path;
use std::process::Command;

#[test]
fn checked_in_graph_fixtures_are_not_binary_onnx() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new("git")
        .args(["ls-files", "*.onnx"])
        .current_dir(&root)
        .output()
        .expect("git must be available to audit checked-in fixtures");
    assert!(
        output.status.success(),
        "git fixture audit failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let binaries = String::from_utf8(output.stdout)
        .expect("git paths are UTF-8")
        .lines()
        .filter(|path| root.join(path).is_file())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert!(
        binaries.is_empty(),
        "checked-in ONNX graphs must use *.onnx.textproto, found: {}",
        binaries.join(", ")
    );
}
