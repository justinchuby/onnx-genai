use std::collections::HashMap;
use std::path::{Path, PathBuf};

use onnx_genai_ort::{
    DataType, DecodeKvMode, DecodeSession, DecodeSessionOptions, Environment, Session,
    SessionOptions, TensorInfo, Value,
};

fn tiny_llm() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm/model.onnx")
}

fn load_session() -> (Environment, Session) {
    let env = Environment::new("decode-session-test").expect("env");
    let session = Session::new(&env, &tiny_llm(), SessionOptions::default()).expect("session");
    (env, session)
}

#[test]
fn fp16_value_round_trips_bits() {
    let bits = vec![0x0000, 0x3c00, 0xc000, 0x7bff];
    let value = Value::from_slice_f16_bits(&bits, &[2, 2]).expect("f16 tensor");
    assert_eq!(value.dtype(), DataType::Float16);
    assert_eq!(value.shape(), &[2, 2]);
    assert_eq!(value.to_vec_f16_bits().expect("f16 bits"), bits);
}

#[test]
fn bound_decode_logits_match_naive_repass() {
    let (_env, session) = load_session();
    let tokens = [1_i64, 5, 7];
    let naive = naive_logits(&session, &tokens);

    let mut decode =
        DecodeSession::new(&session, DecodeSessionOptions::default()).expect("decode session");
    assert_eq!(decode.mode(), DecodeKvMode::ZeroCopyRebind);

    for (index, token) in tokens.iter().enumerate() {
        let total = index + 1;
        let logits = decode
            .step(&[*token], &vec![1; total], &[index as i64])
            .expect("bound step");
        assert_close(&logits.to_vec_f32().expect("bound logits"), &naive[index]);
    }
}

#[test]
fn bound_decode_rewind_matches_replay() {
    let (_env, session) = load_session();
    let mut decode =
        DecodeSession::new(&session, DecodeSessionOptions::default()).expect("decode session");

    let first = decode.step(&[1], &[1], &[0]).expect("first step");
    let first_logits = first.to_vec_f32().expect("first logits");
    decode.step(&[5], &[1, 1], &[1]).expect("second step");
    decode.rewind(1).expect("rewind to first token");
    assert_eq!(decode.past_len(), 1);
    let replayed = decode
        .step(&[5], &[1, 1], &[1])
        .expect("replayed second step");

    let naive = naive_logits(&session, &[1, 5]);
    assert_close(&first_logits, &naive[0]);
    assert_close(&replayed.to_vec_f32().expect("replayed logits"), &naive[1]);
}

fn naive_logits(session: &Session, tokens: &[i64]) -> Vec<Vec<f32>> {
    let mut past = HashMap::new();
    let mut logits = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        let mut owned = vec![
            (
                "input_ids".to_string(),
                Value::from_slice_i64(&[*token], &[1, 1]).expect("input ids"),
            ),
            (
                "attention_mask".to_string(),
                Value::from_slice_i64(&vec![1; index + 1], &[1, (index + 1) as i64])
                    .expect("attention mask"),
            ),
            (
                "position_ids".to_string(),
                Value::from_slice_i64(&[index as i64], &[1, 1]).expect("position ids"),
            ),
        ];
        for input in session.inputs() {
            if is_kv_input(&input.name) {
                let value = past
                    .remove(&input.name)
                    .unwrap_or_else(|| empty_past_value(input).expect("empty past"));
                owned.push((input.name.clone(), value));
            }
        }
        let refs = owned
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect::<Vec<_>>();
        let outputs = session.run(&refs).expect("naive run");
        let mut next_past = HashMap::new();
        for (name, value) in session.output_names().iter().zip(outputs) {
            if name.contains("logits") {
                logits.push(value.to_vec_f32().expect("naive logits"));
            } else if let Some(past_name) = present_to_past(name) {
                next_past.insert(past_name, value);
            }
        }
        past = next_past;
    }
    logits
}

fn empty_past_value(info: &TensorInfo) -> onnx_genai_ort::Result<Value> {
    let seq_axis = info.shape.len() - 2;
    let mut shape = info.shape.clone();
    shape[0] = 1;
    shape[seq_axis] = 0;
    Value::empty(&shape, info.dtype)
}

fn is_kv_input(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("past") && (lower.contains("key") || lower.contains("value"))
}

fn present_to_past(name: &str) -> Option<String> {
    name.strip_prefix("present.")
        .map(|suffix| format!("past_key_values.{suffix}"))
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "logit {index}: {actual} != {expected}"
        );
    }
}
