use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use onnx_genai_ort::{
    BatchedStaticCacheDecodeSession, DataType, DecodeKvMode, DecodeSession, DecodeSessionOptions,
    Environment, Session, SessionOptions, StaticCacheBindingMode, StaticCacheDecodeOptions,
    StaticCacheDecodeSession, TensorInfo, Value,
};

fn tiny_llm() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm/model.onnx")
}

fn tiny_scatter_llm() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm-scatter/model.onnx")
}

fn deterministic_session_options() -> SessionOptions {
    SessionOptions::default().with_intra_op_threads(1)
}

fn test_environment() -> &'static Environment {
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    ENVIRONMENT.get_or_init(|| Environment::new("decode-session-test").expect("env"))
}

fn ort_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn load_session() -> Session {
    Session::new(
        test_environment(),
        &tiny_llm(),
        deterministic_session_options(),
    )
    .expect("session")
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
fn to_vec_f32_lossy_widens_fp16_and_passes_through_fp32() {
    // 0.0, 1.0, -2.0, 65504.0 (fp16 max) as IEEE-754 half bit patterns.
    let bits = vec![0x0000, 0x3c00, 0xc000, 0x7bff];
    let fp16 = Value::from_slice_f16_bits(&bits, &[2, 2]).expect("f16 tensor");
    let widened = fp16.to_vec_f32_lossy().expect("fp16 widen");
    assert_eq!(widened, vec![0.0_f32, 1.0, -2.0, 65504.0]);

    let fp32 = Value::from_slice_f32(&[0.5, -1.25], &[2]).expect("f32 tensor");
    assert_eq!(fp32.to_vec_f32_lossy().expect("f32 passthrough"), vec![0.5, -1.25]);
}

#[test]
fn bound_decode_logits_match_naive_repass() {
    let _guard = ort_test_lock().lock().expect("ORT test lock");
    let session = load_session();
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
    let _guard = ort_test_lock().lock().expect("ORT test lock");
    let session = load_session();
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

#[test]
fn static_cache_decode_reuses_buffers_and_rewinds_deterministically() {
    let _guard = ort_test_lock().lock().expect("ORT test lock");
    let session = Session::new(
        test_environment(),
        &tiny_scatter_llm(),
        deterministic_session_options(),
    )
    .expect("session");
    let signature = StaticCacheDecodeSession::detect(&session)
        .expect("detect")
        .expect("static-cache signature");
    assert_eq!(signature.layers, 1);
    assert_eq!(signature.max_len, 16);
    assert_eq!(signature.kv_dim, 16);
    assert!(!signature.has_position_ids);

    let mut decode =
        StaticCacheDecodeSession::new(&session, StaticCacheDecodeOptions { batch_size: 1 })
            .expect("static decode session");
    let initial_buffers = decode.buffer_infos().expect("initial buffers");
    assert_eq!(initial_buffers.len(), 2);

    let prefill = decode.prefill(&[1, 5], &[0, 1]).expect("prefill");
    assert_eq!(prefill.shape(), &[1, 2, 32]);
    assert_eq!(decode.current_len(), 2);
    assert_eq!(decode.max_len(), 16);
    assert_eq!(
        decode.buffer_infos().expect("after prefill"),
        initial_buffers
    );

    let first = decode.step(&[7], &[2]).expect("first step");
    let first_logits = first.to_vec_f32().expect("first logits");
    let first_token = argmax(&first_logits) as i64;
    assert_eq!(decode.current_len(), 3);
    assert_eq!(decode.binding_mode(), StaticCacheBindingMode::InPlaceAlias);
    assert_eq!(decode.buffer_infos().expect("after first"), initial_buffers);

    let second = decode.step(&[first_token], &[3]).expect("second step");
    let second_logits = second.to_vec_f32().expect("second logits");
    let second_token = argmax(&second_logits) as i64;
    assert_eq!(
        decode.buffer_infos().expect("after second"),
        initial_buffers
    );

    let third = decode.step(&[second_token], &[4]).expect("third step");
    let third_logits = third.to_vec_f32().expect("third logits");
    let third_token = argmax(&third_logits) as i64;
    assert_eq!(decode.buffer_infos().expect("after third"), initial_buffers);
    eprintln!("static-cache tiny tokens: [{first_token}, {second_token}, {third_token}]");

    decode.rewind(3).expect("rewind to first generated token");
    assert_eq!(decode.current_len(), 3);
    assert_eq!(
        decode.buffer_infos().expect("after rewind"),
        initial_buffers
    );

    let replay_second = decode
        .step(&[first_token], &[3])
        .expect("replay second step");
    let replay_second_logits = replay_second.to_vec_f32().expect("replay second logits");
    assert_close(&replay_second_logits, &second_logits);
    assert_eq!(argmax(&replay_second_logits) as i64, second_token);

    let replay_third = decode
        .step(&[second_token], &[4])
        .expect("replay third step");
    let replay_third_logits = replay_third.to_vec_f32().expect("replay third logits");
    assert_close(&replay_third_logits, &third_logits);
    assert_eq!(argmax(&replay_third_logits) as i64, third_token);
    assert_eq!(
        decode.buffer_infos().expect("after replay"),
        initial_buffers
    );
}

#[test]
fn batched_static_cache_matches_unbatched_rows_and_reuses_slots() {
    let _guard = ort_test_lock().lock().expect("ORT test lock");
    let session = Session::new(
        test_environment(),
        &tiny_scatter_llm(),
        deterministic_session_options(),
    )
    .expect("session");
    let prompts = [vec![1_i64, 5], vec![2_i64, 6, 7], vec![3_i64]];
    let generated = 3;

    let expected = prompts
        .iter()
        .map(|prompt| static_cache_greedy_trace(&session, prompt, generated))
        .collect::<Vec<_>>();
    let max_steps = expected
        .iter()
        .map(|trace| trace.input_tokens.len())
        .max()
        .expect("traces");

    let mut batched =
        BatchedStaticCacheDecodeSession::new(&session, StaticCacheDecodeOptions { batch_size: 3 })
            .expect("batched static decode session");
    let initial_buffers = batched.buffer_infos().expect("initial buffers");
    assert_eq!(initial_buffers.len(), 2);
    assert!(
        initial_buffers
            .iter()
            .all(|buffer| buffer.shape == [3, 16, 16])
    );

    for step in 0..max_steps {
        let mut ids = vec![0_i64; prompts.len()];
        let mut positions = vec![0_i64; prompts.len()];
        let mut advance = vec![false; prompts.len()];
        for (row, trace) in expected.iter().enumerate() {
            if step < trace.input_tokens.len() {
                ids[row] = trace.input_tokens[step];
                positions[row] = batched.row_len(row).expect("row len") as i64;
                advance[row] = true;
            }
        }

        let logits = batched
            .step_select(&ids, &positions, &advance)
            .expect("batched step");
        for (row, trace) in expected.iter().enumerate() {
            if advance[row] {
                let row_logits =
                    BatchedStaticCacheDecodeSession::row_logits(&logits, row, 0).expect("row");
                assert_batched_matches_individual(&row_logits, &trace.logits[step]);
            }
        }
        assert_eq!(batched.buffer_infos().expect("after step"), initial_buffers);
    }
    assert_eq!(batched.row_lens(), &[5, 6, 4]);
    assert_eq!(batched.active_rows(), vec![0, 1, 2]);

    batched.rewind_row(1, 2).expect("rewind row 1");
    let replay = batched
        .step_select(
            &[0, expected[1].input_tokens[2], 0],
            &[0, 2, 0],
            &[false, true, false],
        )
        .expect("replay row 1");
    let replay_row = BatchedStaticCacheDecodeSession::row_logits(&replay, 1, 0).expect("row 1");
    assert_batched_matches_individual(&replay_row, &expected[1].logits[2]);
    assert_eq!(batched.row_len(1).expect("row 1 len"), 3);

    batched.deactivate_row(2).expect("deactivate row 2");
    assert!(!batched.is_active(2).expect("row 2 active"));
    batched.assign_row(2).expect("reuse row 2");
    assert!(batched.is_active(2).expect("row 2 active"));
    assert_eq!(batched.row_len(2).expect("row 2 reset"), 0);
    let replacement = static_cache_greedy_trace(&session, &[4_i64, 9], 0);
    for (index, token) in replacement.input_tokens.iter().enumerate() {
        let logits = batched
            .step_select(
                &[0, 0, *token],
                &[0, 0, index as i64],
                &[false, false, true],
            )
            .expect("replacement row step");
        let row_logits =
            BatchedStaticCacheDecodeSession::row_logits(&logits, 2, 0).expect("replacement row");
        assert_batched_matches_individual(&row_logits, &replacement.logits[index]);
    }
    assert_eq!(batched.row_len(2).expect("replacement len"), 2);
    assert_eq!(
        batched.buffer_infos().expect("after reuse"),
        initial_buffers
    );
}

#[test]
fn batched_static_cache_active_compaction_skips_inactive_rows_and_admits_replacement() {
    let _guard = ort_test_lock().lock().expect("ORT test lock");
    let session = Session::new(
        test_environment(),
        &tiny_scatter_llm(),
        deterministic_session_options(),
    )
    .expect("session");
    let prompts = [
        vec![1_i64, 5],
        vec![2_i64, 6],
        vec![3_i64, 7],
        vec![4_i64, 8],
    ];
    let expected = prompts
        .iter()
        .map(|prompt| static_cache_greedy_trace(&session, prompt, 2))
        .collect::<Vec<_>>();

    let mut batched =
        BatchedStaticCacheDecodeSession::new(&session, StaticCacheDecodeOptions { batch_size: 4 })
            .expect("batched static decode session");

    for prompt_index in 0..2 {
        let ids = prompts
            .iter()
            .map(|prompt| prompt[prompt_index])
            .collect::<Vec<_>>();
        let positions = vec![prompt_index as i64; prompts.len()];
        let logits = batched.step(&ids, &positions).expect("prompt step");
        for (row, trace) in expected.iter().enumerate() {
            let row_logits =
                BatchedStaticCacheDecodeSession::row_logits(&logits, row, 0).expect("row logits");
            assert_batched_matches_individual(&row_logits, &trace.logits[prompt_index]);
        }
    }

    let first_generated_ids = [0_usize, 1, 2, 3]
        .into_iter()
        .map(|row| argmax(&expected[row].logits[1]) as i64)
        .collect::<Vec<_>>();
    let full_logits = batched
        .step(&first_generated_ids, &[2, 2, 2, 2])
        .expect("full first generated step");
    for &row in &[0_usize, 2] {
        let row_logits =
            BatchedStaticCacheDecodeSession::row_logits(&full_logits, row, 0).expect("row logits");
        assert_batched_matches_individual(&row_logits, &expected[row].logits[2]);
    }
    let mut full_reference =
        BatchedStaticCacheDecodeSession::new(&session, StaticCacheDecodeOptions { batch_size: 4 })
            .expect("full reference");
    for prompt_index in 0..2 {
        let ids = prompts
            .iter()
            .map(|prompt| prompt[prompt_index])
            .collect::<Vec<_>>();
        full_reference
            .step(&ids, &vec![prompt_index as i64; prompts.len()])
            .expect("reference prompt step");
    }
    full_reference
        .step(&first_generated_ids, &[2, 2, 2, 2])
        .expect("reference first generated step");

    batched.deactivate_row(1).expect("deactivate row 1");
    batched.deactivate_row(3).expect("deactivate row 3");
    assert_eq!(batched.compact().expect("compact active rows"), 2);
    assert_eq!(batched.active_rows(), vec![0, 2]);
    assert_eq!(batched.physical_slot(0).expect("row 0 slot"), Some(0));
    assert_eq!(batched.physical_slot(2).expect("row 2 slot"), Some(1));
    assert_eq!(batched.physical_slot(1).expect("row 1 slot"), None);
    assert_eq!(
        batched.logical_row_for_physical_slot(1).expect("slot 1"),
        Some(2)
    );
    assert!((batched.inactive_compute_fraction() - 0.5).abs() < f32::EPSILON);

    let second_generated_ids = [0_usize, 2]
        .into_iter()
        .map(|row| argmax(&expected[row].logits[2]) as i64)
        .collect::<Vec<_>>();
    let full_second_ids = [0_usize, 1, 2, 3]
        .into_iter()
        .map(|row| argmax(&expected[row].logits[2]) as i64)
        .collect::<Vec<_>>();
    let full_reference_logits = full_reference
        .step(&full_second_ids, &[3, 3, 3, 3])
        .expect("reference second generated step");
    let active_logits = batched
        .step_active(&second_generated_ids, &[3, 3])
        .expect("active second generated step");
    assert_eq!(active_logits.shape(), &[2, 1, 32]);
    for (active_index, row) in [0_usize, 2].into_iter().enumerate() {
        let row_logits =
            BatchedStaticCacheDecodeSession::row_logits(&active_logits, active_index, 0)
                .expect("active row logits");
        assert_batched_matches_individual(&row_logits, &expected[row].logits[3]);
        let full_row_logits =
            BatchedStaticCacheDecodeSession::row_logits(&full_reference_logits, row, 0)
                .expect("full reference row logits");
        assert_close(&row_logits, &full_row_logits);
    }

    batched.admit_row(1).expect("admit replacement row 1");
    assert_eq!(batched.physical_slot(1).expect("replacement slot"), Some(2));
    assert_eq!(batched.row_len(1).expect("replacement len"), 0);
    assert_eq!(batched.active_rows(), vec![0, 2, 1]);

    let replacement = static_cache_greedy_trace(&session, &[9_i64, 10], 0);
    for (index, token) in replacement.input_tokens.iter().enumerate() {
        let logits = batched
            .step_active_select(
                &[0, 0, *token],
                &[
                    batched.row_len(0).expect("row 0 len") as i64,
                    0,
                    index as i64,
                ],
                &[false, false, true],
            )
            .expect("replacement active prefill");
        let row_logits =
            BatchedStaticCacheDecodeSession::row_logits(&logits, 2, 0).expect("replacement logits");
        assert_batched_matches_individual(&row_logits, &replacement.logits[index]);
    }
    assert_eq!(batched.row_len(1).expect("replacement final len"), 2);
    assert_eq!(batched.active_batch_size(), 3);
    assert!((batched.inactive_compute_fraction() - 0.25).abs() < f32::EPSILON);
}

struct StaticTrace {
    input_tokens: Vec<i64>,
    logits: Vec<Vec<f32>>,
}

fn static_cache_greedy_trace(session: &Session, prompt: &[i64], generated: usize) -> StaticTrace {
    let mut decode =
        StaticCacheDecodeSession::new(session, StaticCacheDecodeOptions { batch_size: 1 })
            .expect("static decode session");
    let mut input_tokens = Vec::new();
    let mut logits = Vec::new();
    for (position, token) in prompt.iter().enumerate() {
        let value = decode
            .step(&[*token], &[position as i64])
            .expect("prompt step");
        input_tokens.push(*token);
        logits.push(value.to_vec_f32().expect("prompt logits"));
    }
    for _ in 0..generated {
        let next = argmax(logits.last().expect("previous logits")) as i64;
        let value = decode
            .step(&[next], &[decode.current_len() as i64])
            .expect("generated step");
        input_tokens.push(next);
        logits.push(value.to_vec_f32().expect("generated logits"));
    }
    StaticTrace {
        input_tokens,
        logits,
    }
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

fn assert_batched_matches_individual(batched: &[f32], individual: &[f32]) {
    const TOLERANCE: f32 = 1e-4;

    assert_eq!(batched.len(), individual.len());
    for (index, (batched, individual)) in batched.iter().zip(individual).enumerate() {
        assert!(
            (batched - individual).abs() <= TOLERANCE,
            "logit {index}: batched {batched} != individual {individual}"
        );
    }

    let batched_argmax = argmax(batched);
    let individual_argmax = argmax(individual);
    if batched_argmax != individual_argmax {
        // Batched and individual GEMMs have different reduction structures even with one ORT
        // thread. An ordering change is acceptable only when both outputs show a near-tie.
        assert!(
            (batched[batched_argmax] - batched[individual_argmax]).abs() <= TOLERANCE
                && (individual[individual_argmax] - individual[batched_argmax]).abs() <= TOLERANCE,
            "argmax mismatch: batched {batched_argmax}, individual {individual_argmax}"
        );
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("non-empty logits")
}
