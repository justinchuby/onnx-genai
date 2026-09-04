use std::path::Path;

#[path = "support/dflash_admission_fixture.rs"]
mod dflash_admission_fixture;

use dflash_admission_fixture::{PROPOSER_FILE, TARGET_FILE, check, documents};
use onnx_genai_ort::{DataType, Environment, Session, SessionOptions, Value};

fn fixture_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dflash-admission")
}

#[test]
fn dflash_admission_fixture_generation_is_deterministic() {
    let first = documents().expect("first DFlash fixture generation");
    let second = documents().expect("second DFlash fixture generation");
    assert_eq!(first, second);
    for (file, document) in [(PROPOSER_FILE, first.proposer), (TARGET_FILE, first.target)] {
        assert!(
            std::str::from_utf8(&document).is_ok(),
            "{file} must be canonical UTF-8"
        );
        assert!(
            !document.contains(&b'\r'),
            "{file} must use LF even on Windows"
        );
    }
}

#[test]
fn maintained_dflash_admission_fixtures_match_generator() {
    let generated = documents().expect("DFlash fixture generation");
    let root = fixture_root();
    for (file, actual, expected) in [
        (
            PROPOSER_FILE,
            std::fs::read(root.join(PROPOSER_FILE)),
            generated.proposer,
        ),
        (
            TARGET_FILE,
            std::fs::read(root.join(TARGET_FILE)),
            generated.target,
        ),
    ] {
        let actual = actual.unwrap_or_else(|error| {
            panic!("failed to read {}: {error}", root.join(file).display())
        });
        assert_eq!(
            actual, expected,
            "{file} is stale; regenerate with \
             `cargo run -p onnx-genai-engine --example generate_dflash_admission_fixture`"
        );
    }
    check(&root).expect("DFlash generator exact-byte check");
}

#[test]
fn maintained_dflash_admission_models_execute() -> anyhow::Result<()> {
    let root = fixture_root();
    let environment = Environment::new("dflash-maintained-fixture")?;
    let target = Session::new(
        &environment,
        &root.join(TARGET_FILE),
        SessionOptions::default(),
    )?;
    let tokens = Value::from_slice_i64(&[1, 2], &[1, 2])?;
    let past_target = Value::from_slice_f32(&[0.0; 3], &[1, 1, 3])?;
    let token_history = Value::from_slice_i64(&[0], &[1, 1])?;
    let recurrent = Value::from_slice_f32(&[0.5, -0.5], &[1, 2])?;
    let target_outputs = target.run(&[
        ("tokens", &tokens),
        ("past_target", &past_target),
        ("token_history", &token_history),
        ("recurrent", &recurrent),
    ])?;
    assert_eq!(target_outputs[0].to_vec_f32()?.len(), 6);
    assert_eq!(target_outputs[3].to_vec_f32()?.len(), 22);
    assert_eq!(target_outputs[4].to_vec_f32()?.len(), 9);
    assert_eq!(target_outputs[5].to_vec_i64()?, vec![0, 1, 2]);
    assert_eq!(target_outputs[6].to_vec_f32()?, vec![0.5, -0.5]);
    assert_eq!(target_outputs[7].to_vec_f32()?, vec![0.5, -0.5, 0.5, -0.5]);

    let proposer = Session::new(
        &environment,
        &root.join(PROPOSER_FILE),
        SessionOptions::default(),
    )?;
    let target_features = Value::from_slice_f32(&[0.0; 18], &[1, 2, 9])?;
    let noise_embeddings = Value::from_slice_f32(&[0.0; 12], &[1, 4, 3])?;
    let masked_positions = Value::from_raw_bytes(vec![0; 4], &[1, 4], DataType::Bool)?;
    let position_ids = Value::from_slice_i64(&[0, 1], &[1, 2])?;
    let attention_mask = Value::from_slice_i64(&[1, 1], &[1, 2])?;
    let output_projection = Value::from_slice_f32(&[0.0; 33], &[3, 11])?;
    let past_draft = Value::from_slice_f32(&[0.0; 3], &[1, 1, 3])?;
    let proposer_outputs = proposer.run(&[
        ("target_features", &target_features),
        ("noise_embeddings", &noise_embeddings),
        ("masked_positions", &masked_positions),
        ("position_ids", &position_ids),
        ("attention_mask", &attention_mask),
        ("output_projection", &output_projection),
        ("past_draft", &past_draft),
    ])?;
    assert_eq!(proposer_outputs[0].to_vec_i64()?, vec![1, 2, 3]);
    assert_eq!(proposer_outputs[1].to_vec_f32()?.len(), 33);
    assert_eq!(proposer_outputs[2].to_vec_f32()?.len(), 9);
    Ok(())
}
