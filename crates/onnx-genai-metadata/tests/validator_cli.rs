use std::path::PathBuf;
use std::process::Command;

#[test]
fn validator_cli_accepts_checked_in_conformance_package() {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/north_star_minimal.yaml");
    let output = Command::new(env!("CARGO_BIN_EXE_validate_metadata"))
        .arg(&fixture)
        .output()
        .expect("validator CLI runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn validator_cli_fails_closed() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let output = Command::new(env!("CARGO_BIN_EXE_validate_metadata"))
        .arg(fixture.join("invalid.yaml"))
        .output()
        .expect("validator CLI runs");
    assert!(!output.status.success());
}
