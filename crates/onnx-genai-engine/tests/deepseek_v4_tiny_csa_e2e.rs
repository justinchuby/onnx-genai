//! Native CPU end-to-end coverage for canonical compressed-attention state.
#![cfg(feature = "native-backend")]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    CompressedRecordStateInfo, NativeDecodeDevice, NativeDecodeSession, NativeStateOperation,
    NativeStateOperationRefusal,
};
use onnx_genai_metadata::{CompressionRatio, StateGroupProperties, StateKind, StatePortRole};
use onnx_runtime_ir::DataType;
use onnx_runtime_memory_governor::Tier;
#[cfg(feature = "native-cuda")]
use onnx_runtime_session::{DevicePreference, InferenceSession};

fn fixture_dir() -> PathBuf {
    let dir = std::env::var_os("DEEPSEEK_V4_TINY_CSA_E2E_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-deepseek-v4-csa")
        });
    for artifact in ["model.onnx.textproto", "inference_metadata.yaml"] {
        assert!(
            dir.join(artifact).is_file(),
            "required compressed-attention fixture artifact is missing: {}",
            dir.join(artifact).display()
        );
    }
    assert!(
        !dir.join("model.onnx").exists() && !dir.join("model.onnx.data").exists(),
        "the fixture must use the repository-governed textproto representation"
    );
    dir
}

fn build_cpu_session(dir: &Path) -> NativeDecodeSession {
    NativeDecodeSession::load_with_resolved_io(
        dir.join("model.onnx.textproto"),
        NativeDecodeDevice::Cpu,
    )
    .expect("load canonical workflow metadata and native CPU model")
}

#[cfg(feature = "native-cuda")]
fn remove_textproto_message(text: &mut String, marker: &str) {
    let start = text
        .find(marker)
        .unwrap_or_else(|| panic!("missing textproto message marker {marker:?}"));
    let mut depth = 0usize;
    let mut end = None;
    for (offset, byte) in text.as_bytes()[start..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1).expect("balanced textproto braces");
                if depth == 0 {
                    end = Some(start + offset + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let mut end = end.expect("complete textproto message");
    if text.as_bytes().get(end) == Some(&b'\n') {
        end += 1;
    }
    text.replace_range(start..end, "");
}

#[cfg(feature = "native-cuda")]
fn five_output_fixture_dir(source: &Path) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/tiny-deepseek-v4-csa-five-output");
    std::fs::create_dir_all(&dir).expect("create governed five-output fixture directory");
    let mut model = std::fs::read_to_string(source.join("model.onnx.textproto"))
        .expect("read canonical CSA fixture");
    let node_output = "    output: \"selected_indices.0\"\n";
    assert_eq!(
        model.matches(node_output).count(),
        1,
        "selected_indices must be one CSA node output"
    );
    model = model.replacen(node_output, "", 1);
    remove_textproto_message(&mut model, "  output {\n    name: \"selected_indices.0\"\n");
    std::fs::write(dir.join("model.onnx.textproto"), model).expect("write five-output CSA fixture");
    std::fs::copy(
        source.join("inference_metadata.yaml"),
        dir.join("inference_metadata.yaml"),
    )
    .expect("copy canonical CSA metadata");
    dir
}

#[cfg(feature = "native-cuda")]
fn build_cuda_session(dir: &Path) -> NativeDecodeSession {
    let metadata = onnx_genai_metadata::load_metadata_from_dir(dir)
        .expect("load CSA fixture metadata")
        .expect("CSA fixture metadata");
    let io = metadata.decoder_io().expect("derive canonical decoder ABI");
    let session = InferenceSession::builder()
        .model(dir.join("model.onnx.textproto"))
        .device(DevicePreference::Gpu { index: Some(0) })
        .option("optimization", "basic")
        .build()
        .expect("build native CUDA session over the CSA fixture");
    NativeDecodeSession::from_session_with_io(session, io)
        .expect("wrap canonical CSA decoder with fixed-stride CUDA state")
}

fn schedule_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-deepseek-v4-csa-schedule")
}

#[cfg(feature = "native-cuda")]
#[test]
fn cuda_native_five_output_ratio4_scatter_preserves_fixed_stride_state() {
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping native fixed-stride five-output CSA test: {error}");
        return;
    }
    let dir = five_output_fixture_dir(&fixture_dir());
    let graph =
        onnx_runtime_loader::load_model(dir.join("model.onnx.textproto")).expect("load fixture");
    let ratio4 = graph
        .nodes
        .iter()
        .map(|(_, node)| node)
        .find(|node| {
            node.domain == onnx_runtime_ir::RUNTIME_DOMAIN
                && node.op_type == "CompressedSparseAttention"
                && node
                    .attr("compression_ratio")
                    .and_then(|value| value.as_int())
                    == Some(4)
        })
        .expect("ratio-4 CSA node");
    assert_eq!(
        ratio4.outputs.len(),
        5,
        "fixture must exercise the valid five-output CSA schema"
    );

    let mut cuda = build_cuda_session(&dir);
    let initial = cuda.cuda_kv_debug_stats().expect("CUDA state stats");
    assert_eq!(initial.csa_record_device_ptrs.len(), 3);
    assert!(
        initial
            .csa_record_logical_shapes
            .iter()
            .all(|shape| shape[0] == 1)
    );
    let pointers = initial.csa_record_device_ptrs;

    for total in 1..=9 {
        let past = total - 1;
        let token = ((total * 3) % 97 + 1) as u32;
        let logits = cuda
            .decode(&[token], past)
            .unwrap_or_else(|error| panic!("five-output CUDA token {total} failed: {error:#}"));
        assert!(
            logits[0].iter().all(|value| value.is_finite()),
            "five-output CUDA token {total} produced non-finite logits"
        );
    }

    let stats = cuda.cuda_kv_debug_stats().expect("CUDA state stats");
    assert_eq!(stats.csa_record_device_ptrs, pointers);
    assert!(
        stats
            .csa_record_logical_shapes
            .iter()
            .zip(&stats.csa_record_physical_shapes)
            .all(|(logical, physical)| {
                logical[0] == 1
                    && logical[1] <= physical[1]
                    && logical[2] == physical[2]
                    && physical[1] > logical[1]
            })
    );
    assert!(
        stats.csa_record_growth_events >= 2,
        "ratio-4 record cursors must grow while fixed physical strides stay stable"
    );

    cuda.reset().expect("reset fixed-stride CSA state");
    let reset = cuda.cuda_kv_debug_stats().expect("reset CUDA state stats");
    assert_eq!(reset.csa_record_device_ptrs, pointers);
    assert!(
        reset
            .csa_record_logical_shapes
            .iter()
            .all(|shape| shape[1] == 0)
    );
    let logits = cuda.decode(&[17], 0).expect("decode after CSA reset");
    assert!(logits[0].iter().all(|value| value.is_finite()));
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index as u32)
        .expect("logits row must not be empty")
}

fn records(session: &NativeDecodeSession, input: &str) -> usize {
    session
        .compressed_record_state()
        .unwrap()
        .into_iter()
        .find(|entry| entry.input == input)
        .unwrap_or_else(|| panic!("missing compressed state '{input}'"))
        .records
}

#[derive(Debug, PartialEq, Eq)]
struct RequestOutcome {
    generated: Vec<u32>,
    current_len: usize,
    state: Vec<CompressedRecordStateInfo>,
}

fn run_persistent_request(dir: &Path, prompt: &[u32]) -> RequestOutcome {
    let mut session = build_cpu_session(dir);
    let mut logits = session.decode(prompt, 0).unwrap().pop().unwrap();
    let mut generated = Vec::with_capacity(16);
    for step in 0..16 {
        let token = argmax(&logits);
        generated.push(token);
        let past = session.current_len();
        logits = session
            .decode(&[token], past)
            .unwrap_or_else(|error| panic!("solo decode step {step} failed: {error:#}"))
            .pop()
            .unwrap();
    }
    RequestOutcome {
        generated,
        current_len: session.current_len(),
        state: session.compressed_record_state().unwrap(),
    }
}

#[test]
fn cpu_prefill_and_16_decode_steps_advance_real_record_state() {
    let dir = fixture_dir();
    let mut session = build_cpu_session(&dir);
    assert_eq!(
        session.kv_reservation(512).unwrap(),
        (4_285_824, Tier::Host),
        "dense KV and cadence-sized compressed records share governed accounting"
    );
    assert_eq!(
        session.recurrent_state_reservation().unwrap(),
        (606_208, Tier::Host),
        "fixed compressor carries use the governed recurrent-state charge"
    );
    let prompt = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut logits = session.decode(&prompt, 0).unwrap().pop().unwrap();

    for step in 0..16 {
        let past = session.current_len();
        logits = session
            .decode(&[argmax(&logits)], past)
            .unwrap_or_else(|error| panic!("decode step {step} failed: {error:#}"))
            .pop()
            .unwrap();
    }
    assert_eq!(session.current_len(), 24);

    let state = session.compressed_record_state().unwrap();
    assert_eq!(state.len(), 3);
    assert!(state.iter().any(|entry| {
        entry.input == "past_compressed_kv.0"
            && entry.ratio == CompressionRatio::Ratio4
            && entry.dtype == DataType::Uint8
            && entry.records == 6
    }));
    assert!(state.iter().any(|entry| {
        entry.input == "past_index_key.0"
            && entry.ratio == CompressionRatio::Ratio4
            && entry.dtype == DataType::Uint8
            && entry.records == 6
    }));
    assert!(state.iter().any(|entry| {
        entry.input == "past_compressed_kv.1"
            && entry.ratio == CompressionRatio::Ratio128
            && entry.dtype == DataType::Float32
            && entry.records == 0
    }));
}

#[test]
fn exact_21_ratio4_20_ratio128_schedule_advances_through_session() {
    let dir = schedule_fixture_dir();
    let metadata = onnx_genai_metadata::load_metadata_from_dir(&dir)
        .unwrap()
        .expect("schedule fixture metadata");
    let abi = metadata.decoder_io().expect("derived decoder ABI");
    let mut schedule = abi
        .state_groups
        .iter()
        .filter(|group| {
            group.kind == StateKind::CompressedAttention
                && group
                    .ports
                    .iter()
                    .any(|port| port.role == Some(StatePortRole::CompressedKv))
        })
        .map(|group| {
            let layer = group.ports[0].layer.expect("compressed layer index");
            let Some(StateGroupProperties::CompressedAttention { ratio, .. }) = group.properties
            else {
                panic!("compressed group requires typed properties");
            };
            (layer, ratio)
        })
        .collect::<Vec<_>>();
    schedule.sort_unstable_by_key(|(layer, _)| *layer);
    assert_eq!(schedule.len(), 41);
    assert_eq!(
        schedule.iter().map(|(layer, _)| *layer).collect::<Vec<_>>(),
        (2..=42).collect::<Vec<_>>()
    );
    assert!(schedule.iter().enumerate().all(|(index, (_, ratio))| {
        *ratio
            == if index % 2 == 0 {
                CompressionRatio::Ratio4
            } else {
                CompressionRatio::Ratio128
            }
    }));

    let mut session = build_cpu_session(&dir);
    let mut logits = session.decode(&[1, 2, 3, 4], 0).unwrap().pop().unwrap();
    for step in 0..16 {
        let past = session.current_len();
        logits = session
            .decode(&[argmax(&logits)], past)
            .unwrap_or_else(|error| panic!("full-schedule decode step {step} failed: {error:#}"))
            .pop()
            .unwrap();
    }

    let state = session.compressed_record_state().unwrap();
    let ratio4_kv = state
        .iter()
        .filter(|entry| {
            entry.input.starts_with("past_compressed_kv.")
                && entry.ratio == CompressionRatio::Ratio4
                && entry.dtype == DataType::Uint8
                && entry.records == 5
        })
        .count();
    let ratio4_index = state
        .iter()
        .filter(|entry| {
            entry.input.starts_with("past_index_key.")
                && entry.ratio == CompressionRatio::Ratio4
                && entry.dtype == DataType::Uint8
                && entry.records == 5
        })
        .count();
    let ratio128_kv = state
        .iter()
        .filter(|entry| {
            entry.input.starts_with("past_compressed_kv.")
                && entry.ratio == CompressionRatio::Ratio128
                && entry.dtype == DataType::Float32
                && entry.records == 0
        })
        .count();
    assert_eq!(ratio4_kv, 21);
    assert_eq!(ratio4_index, 21);
    assert_eq!(ratio128_kv, 20);
}

#[test]
fn ratio128_record_cursor_is_exact_around_128_256_and_257() {
    let dir = fixture_dir();
    let mut session = build_cpu_session(&dir);
    let prompt = (0..127)
        .map(|index| (index % 97 + 1) as u32)
        .collect::<Vec<_>>();
    session.decode(&prompt, 0).expect("127-token prefill");
    assert_eq!(records(&session, "past_compressed_kv.1"), 0);

    session.decode(&[3], 127).expect("token 128");
    assert_eq!(records(&session, "past_compressed_kv.1"), 1);
    session.decode(&[5], 128).expect("token 129");
    assert_eq!(records(&session, "past_compressed_kv.1"), 1);

    let to_255 = (0..126)
        .map(|index| (index % 89 + 1) as u32)
        .collect::<Vec<_>>();
    session.decode(&to_255, 129).expect("tokens through 255");
    assert_eq!(records(&session, "past_compressed_kv.1"), 1);
    session.decode(&[7], 255).expect("token 256");
    assert_eq!(records(&session, "past_compressed_kv.1"), 2);
    session.decode(&[11], 256).expect("token 257");
    assert_eq!(records(&session, "past_compressed_kv.1"), 2);

    let hca = session
        .compressed_record_state()
        .unwrap()
        .into_iter()
        .find(|entry| entry.input == "past_compressed_kv.1")
        .unwrap();
    assert_eq!(hca.ratio, CompressionRatio::Ratio128);
    assert_eq!(hca.dtype, DataType::Float32);
    assert_eq!(hca.batch, 1);
    assert_eq!(hca.layer, 1);
    assert_eq!(hca.record_width_bytes, 512 * std::mem::size_of::<f32>());
}

#[test]
fn unsupported_state_operations_are_typed_transactional_refusals() {
    let dir = fixture_dir();
    let mut session = build_cpu_session(&dir);
    session.decode(&[3, 1, 4, 1, 5], 0).unwrap();
    let before_len = session.current_len();
    let before_state = session.compressed_record_state().unwrap();

    let snapshot_error = match session.snapshot_recurrent_state_public() {
        Ok(_) => panic!("compressed record state must not be snapshot-capable"),
        Err(error) => error,
    };
    assert!(
        snapshot_error
            .downcast_ref::<NativeStateOperationRefusal>()
            .is_some_and(|reason| reason.operation == NativeStateOperation::Snapshot)
    );
    for operation in [
        NativeStateOperation::Restore,
        NativeStateOperation::Rollback,
        NativeStateOperation::Fork,
    ] {
        let error = session
            .ensure_state_operation_supported(operation, Some(before_len))
            .expect_err("compressed record state operation must be refused");
        assert!(
            error
                .downcast_ref::<NativeStateOperationRefusal>()
                .is_some_and(|reason| reason.operation == operation)
        );
        assert_eq!(session.current_len(), before_len);
        assert_eq!(session.compressed_record_state().unwrap(), before_state);
    }

    let rewind_error = session
        .rewind(before_len - 1)
        .expect_err("compressed record state is not prefix-rewindable");
    assert!(
        rewind_error
            .downcast_ref::<NativeStateOperationRefusal>()
            .is_some_and(|reason| {
                reason.operation == NativeStateOperation::Rewind
                    && reason.target_len == Some(before_len - 1)
            })
    );
    assert_eq!(session.current_len(), before_len);
    assert_eq!(session.compressed_record_state().unwrap(), before_state);

    session
        .reset()
        .expect("reset to empty state remains supported");
    assert_eq!(session.current_len(), 0);
}

#[test]
fn persistent_cpu_request_sets_cover_b1_b2_b3_without_cross_talk() {
    let dir = fixture_dir();
    let prompts = [
        (0..8)
            .map(|index| (index % 31 + 1) as u32)
            .collect::<Vec<_>>(),
        (0..127)
            .map(|index| (index % 43 + 7) as u32)
            .collect::<Vec<_>>(),
        (0..255)
            .map(|index| (index % 59 + 13) as u32)
            .collect::<Vec<_>>(),
    ];

    for batch in 1..=3 {
        let expected = prompts[..batch]
            .iter()
            .map(|prompt| run_persistent_request(&dir, prompt))
            .collect::<Vec<_>>();
        let mut sessions = (0..batch)
            .map(|_| build_cpu_session(&dir))
            .collect::<Vec<_>>();
        let mut logits = sessions
            .iter_mut()
            .zip(&prompts[..batch])
            .map(|(session, prompt)| session.decode(prompt, 0).unwrap().pop().unwrap())
            .collect::<Vec<_>>();
        let mut generated = vec![Vec::with_capacity(16); batch];

        for step in 0..16 {
            for row in 0..batch {
                let token = argmax(&logits[row]);
                generated[row].push(token);
                let past = sessions[row].current_len();
                logits[row] = sessions[row]
                    .decode(&[token], past)
                    .unwrap_or_else(|error| {
                        panic!("batch {batch} row {row} decode step {step} failed: {error:#}")
                    })
                    .pop()
                    .unwrap();
            }
        }

        for row in 0..batch {
            let actual = RequestOutcome {
                generated: generated[row].clone(),
                current_len: sessions[row].current_len(),
                state: sessions[row].compressed_record_state().unwrap(),
            };
            assert_eq!(
                actual, expected[row],
                "interleaving {batch} persistent requests changed row {row}"
            );
            assert_eq!(actual.current_len, prompts[row].len() + 16);
            assert_eq!(
                records(&sessions[row], "past_compressed_kv.0"),
                actual.current_len / 4
            );
            assert_eq!(
                records(&sessions[row], "past_index_key.0"),
                actual.current_len / 4
            );
            assert_eq!(
                records(&sessions[row], "past_compressed_kv.1"),
                actual.current_len / 128
            );
        }
    }
}
