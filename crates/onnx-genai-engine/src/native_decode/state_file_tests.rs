use super::*;
use crate::{Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest};

fn path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}
fn load(path: &Path) -> anyhow::Result<Engine> {
    Engine::from_dir(
        path,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(crate::NativeDecodeDevice::Cpu),
            native_state_files: crate::NativeStateFilePolicy::Required,
            ..EngineConfig::default()
        },
    )
}
fn request(tokens: Vec<TokenId>, count: usize) -> GenerateRequest {
    let mut r = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
    r.options.max_new_tokens = count;
    r.options.greedy = true;
    r.options.stop_on_eos = false;
    r
}
fn save(engine: &mut Engine, session: SessionId) -> anyhow::Result<Vec<u8>> {
    let mut file = Vec::new();
    let report =
        engine.write_native_session_state(session, &mut file, NativeStateFileLimits::default())?;
    assert_eq!(report.bytes, file.len() as u64);
    assert_eq!(report.logical_tokens, engine.session_token_count(session)?);
    assert!(report.materialized_tokens < report.logical_tokens);
    Ok(file)
}
fn assert_state(
    engine: &mut Engine,
    baseline: &NativePastSnapshot,
    tokens: &[TokenId],
    target: SessionId,
) -> anyhow::Result<()> {
    let actual = engine.native_session.as_mut().unwrap().snapshot_past()?;
    assert_eq!(actual.len, baseline.len);
    assert_eq!(actual.host_contents(), baseline.host_contents());
    assert_eq!(actual.last_hidden, baseline.last_hidden);
    if let (Some(a), Some(b)) = (&actual.cpu_kv, &baseline.cpu_kv) {
        assert_eq!(a.logical_len, b.logical_len);
        assert_eq!(a.entries.len(), b.entries.len());
        for (a, b) in a.entries.iter().zip(&b.entries) {
            assert_eq!(a.pair, b.pair);
            assert_eq!(a.dtype, b.dtype);
            assert_eq!(a.rows, b.rows);
            assert!(
                a.shape
                    .iter()
                    .zip(&b.shape)
                    .enumerate()
                    .all(|(axis, (a, b))| axis == 2 || a == b)
            );
        }
    } else {
        assert_eq!(actual.cpu_kv.is_some(), baseline.cpu_kv.is_some());
    }
    assert_eq!(engine.native_sessions[&target].tokens, tokens);
    Ok(())
}

#[test]
fn state_file_cross_restart_preserves_kv_hidden_pending_token_and_continuation()
-> anyhow::Result<()> {
    for fixture in [
        "tiny-native-engine",
        "tiny-native-scalar-gqa",
        "tiny-mtp-full",
    ] {
        let root = path(fixture);
        let mut original = load(&root)?;
        let source = original.create_session()?;
        original.generate_in_session(source, request(vec![0], 3))?;
        let tokens = original.native_sessions[&source].tokens.clone();
        let baseline = original.native_session.as_mut().unwrap().snapshot_past()?;
        let file = save(&mut original, source)?;
        let expected = original
            .generate_in_session(source, request(tokens.clone(), 3))?
            .token_ids;
        drop(original);
        let mut restarted = load(&root)?;
        let target = restarted.create_session()?;
        let report = restarted.restore_native_session_state(
            target,
            &mut file.as_slice(),
            NativeStateFileLimits::default(),
        )?;
        assert_eq!(report.bytes, file.len() as u64);
        assert_eq!(report.logical_tokens, tokens.len());
        assert_eq!(report.materialized_tokens, baseline.len);
        assert_state(&mut restarted, &baseline, &tokens, target)?;
        assert_eq!(
            restarted
                .generate_in_session(target, request(tokens, 3))?
                .token_ids,
            expected
        );
    }
    Ok(())
}

#[test]
fn state_file_restores_live_cpu_rows_into_a_smaller_initial_capacity() -> anyhow::Result<()> {
    let root = path("tiny-native-scalar-gqa");
    let mut original = load(&root)?;
    let source = original.create_session()?;
    original.generate_in_session(source, request(vec![0], 6))?;
    let file = save(&mut original, source)?;
    let tokens = original.native_sessions[&source].tokens.clone();
    let baseline = original.native_session.as_mut().unwrap().snapshot_past()?;
    let expected = original
        .generate_in_session(source, request(tokens.clone(), 2))?
        .token_ids;
    drop(original);
    let mut restarted = load(&root)?;
    let native = restarted.native_session.as_mut().unwrap();
    native.cpu_kv = cpu::DecodeCpuKvState::new(&mut native.session, &native.present_to_past, 2)?;
    assert_eq!(native.cpu_kv.as_ref().unwrap().max_len, 2);
    let target = restarted.create_session()?;
    restarted.restore_native_session_state(
        target,
        &mut file.as_slice(),
        NativeStateFileLimits::default(),
    )?;
    assert!(
        restarted
            .native_session
            .as_ref()
            .unwrap()
            .cpu_kv
            .as_ref()
            .unwrap()
            .max_len
            >= baseline.len
    );
    assert_state(&mut restarted, &baseline, &tokens, target)?;
    assert_eq!(
        restarted
            .generate_in_session(target, request(tokens, 2))?
            .token_ids,
        expected
    );
    Ok(())
}

fn edit_manifest(
    file: &[u8],
    edit: impl FnOnce(&mut serde_json::Value),
) -> anyhow::Result<Vec<u8>> {
    let len = usize::try_from(u64::from_le_bytes(file[40..48].try_into()?))?;
    let mut value = serde_json::from_slice(&file[HEADER as usize..HEADER as usize + len])?;
    edit(&mut value);
    let json = serde_json::to_vec(&value)?;
    let mut result = file[..HEADER as usize].to_vec();
    result[40..48].copy_from_slice(&(json.len() as u64).to_le_bytes());
    result.extend(json);
    result.extend_from_slice(&file[HEADER as usize + len..file.len() - 32]);
    let digest = Sha256::digest(&result);
    result.extend_from_slice(&digest);
    Ok(result)
}

#[test]
fn state_file_rejects_corruption_and_forged_layouts_without_touching_live_state()
-> anyhow::Result<()> {
    let mut engine = load(&path("tiny-native-engine"))?;
    let target = engine.create_session()?;
    engine.generate_in_session(target, request(vec![0], 3))?;
    let file = save(&mut engine, target)?;
    let mut candidates = vec![
        (file[..file.len() - 1].to_vec(), "checksum"),
        (
            edit_manifest(&file, |v| v["past"][0]["dtype"] = serde_json::json!(255))?,
            "dtype",
        ),
        (
            edit_manifest(&file, |v| v["past"][0]["dtype"] = serde_json::json!(12))?,
            "dtype/rank",
        ),
        (
            edit_manifest(&file, |v| v["past"][0]["shape"][0] = serde_json::json!(2))?,
            "byte count",
        ),
        (
            edit_manifest(&file, |v| v["len"] = serde_json::json!(4))?,
            "axis",
        ),
        (
            edit_manifest(&file, |v| v["tokens"][0] = serde_json::json!(u32::MAX))?,
            "vocabulary",
        ),
        (
            edit_manifest(&file, |v| {
                v["extra"] = serde_json::json!(true);
            })?,
            "unknown field",
        ),
        (
            edit_manifest(&file, |v| {
                v["past"].as_array_mut().unwrap().pop();
            })?,
            "tensor count",
        ),
        (
            edit_manifest(&file, |v| {
                let first = v["past"][0].clone();
                v["past"][1] = first;
            })?,
            "duplicated",
        ),
    ];
    let mut checksum = file.clone();
    let last = checksum.len() - 33;
    checksum[last] ^= 1;
    candidates.push((checksum, "checksum mismatch"));
    let mut trailing = file.clone();
    trailing.push(0);
    candidates.push((trailing, "trailing data"));
    let mut magic = file.clone();
    magic[7] = b'2';
    candidates.push((magic, "magic/version"));
    let mut identity = file.clone();
    identity[8] ^= 1;
    candidates.push((identity, "model identity"));
    let mut length = file.clone();
    length[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
    candidates.push((length, "overflow"));
    let mut payload = file.clone();
    payload[48..56].copy_from_slice(&0_u64.to_le_bytes());
    candidates.push((payload, "payload length"));
    let baseline = engine.native_session.as_mut().unwrap().snapshot_past()?;
    let tokens = engine.native_sessions[&target].tokens.clone();
    let stamp = engine.native_access_counter;
    let generation = engine.native_sessions[&target].generation;
    let used = MemoryGovernor::used(engine.governor.memory(), Tier::Host);
    for (candidate, reason) in candidates {
        let error = engine
            .restore_native_session_state(
                target,
                &mut candidate.as_slice(),
                NativeStateFileLimits::default(),
            )
            .unwrap_err();
        assert!(
            format!("{error:#}").contains(reason),
            "expected {reason:?}, got {error:#}"
        );
        assert_state(&mut engine, &baseline, &tokens, target)?;
        assert_eq!(engine.native_sessions[&target].generation, generation);
        assert_eq!(engine.native_access_counter, stamp);
        assert_eq!(
            MemoryGovernor::used(engine.governor.memory(), Tier::Host),
            used
        );
    }
    Ok(())
}

#[test]
fn state_file_enforces_limits_memory_admission_and_opt_in() -> anyhow::Result<()> {
    let mut engine = load(&path("tiny-native-engine"))?;
    let target = engine.create_session()?;
    engine.generate_in_session(target, request(vec![0], 3))?;
    let file = save(&mut engine, target)?;
    let baseline = engine.native_session.as_mut().unwrap().snapshot_past()?;
    let tokens = engine.native_sessions[&target].tokens.clone();
    for limits in [
        NativeStateFileLimits {
            max_bytes: 100,
            ..Default::default()
        },
        NativeStateFileLimits {
            max_manifest_bytes: 1,
            ..Default::default()
        },
        NativeStateFileLimits {
            max_tokens: 1,
            ..Default::default()
        },
    ] {
        assert!(
            engine
                .restore_native_session_state(target, &mut file.as_slice(), limits)
                .is_err()
        );
        let mut output = Vec::new();
        assert!(
            engine
                .write_native_session_state(target, &mut output, limits)
                .is_err()
        );
        assert!(
            output.is_empty(),
            "admission failure must precede the first byte"
        );
        assert_state(&mut engine, &baseline, &tokens, target)?;
    }
    let memory = engine.governor.memory().clone();
    let guard = reserve(&memory, engine.governor.snapshot().host_ram.headroom)?;
    let error = engine
        .restore_native_session_state(target, &mut file.as_slice(), Default::default())
        .unwrap_err();
    assert!(format!("{error:#}").contains("reserve bounded Native state file buffers"));
    assert_state(&mut engine, &baseline, &tokens, target)?;
    drop(guard);
    let mut disabled = Engine::from_dir(&path("tiny-native-engine"), EngineConfig::default())?;
    let session = disabled.create_session()?;
    disabled.generate_in_session(session, request(vec![0], 1))?;
    assert!(
        disabled
            .write_native_session_state(session, &mut Vec::new(), Default::default())
            .unwrap_err()
            .to_string()
            .contains("native_state_files=Required")
    );
    assert!(
        disabled
            .restore_native_session_state(session, &mut file.as_slice(), Default::default())
            .unwrap_err()
            .to_string()
            .contains("native_state_files=Required")
    );
    Ok(())
}

#[test]
fn state_file_identity_is_fixed_at_load_and_rejects_changed_metadata_after_restart()
-> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    for name in [
        "model.onnx.textproto",
        "tokenizer.json",
        "inference_metadata.yaml",
    ] {
        std::fs::copy(
            path("tiny-native-engine").join(name),
            root.path().join(name),
        )?;
    }
    let mut engine = load(root.path())?;
    let session = engine.create_session()?;
    engine.generate_in_session(session, request(vec![0], 2))?;
    let file = save(&mut engine, session)?;
    let metadata_path = root.path().join("inference_metadata.yaml");
    let mut metadata: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata_path)?)?;
    metadata["model"]["max_sequence_length"] = serde_yaml::Value::from(1000);
    std::fs::write(metadata_path, serde_yaml::to_string(&metadata)?)?;
    assert_eq!(
        save(&mut engine, session)?,
        file,
        "saving must use loaded identity rather than re-read changed files"
    );
    drop(engine);
    let mut restarted = load(root.path())?;
    let target = restarted.create_session()?;
    let error = restarted
        .restore_native_session_state(target, &mut file.as_slice(), Default::default())
        .unwrap_err();
    assert!(error.to_string().contains("model identity"));
    assert_eq!(restarted.session_token_count(target)?, 0);
    Ok(())
}

fn compressed_package() -> anyhow::Result<tempfile::TempDir> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-deepseek-v4-csa");
    let dir = tempfile::tempdir()?;
    for name in ["inference_metadata.yaml", "model.onnx.textproto"] {
        std::fs::copy(source.join(name), dir.path().join(name))?;
    }
    let metadata: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(
        source.join("inference_metadata.yaml"),
    )?)?;
    let count = metadata["package"]["tokenizer"]["vocab_size"]
        .as_u64()
        .context("compressed fixture vocabulary")?;
    let vocabulary: serde_json::Map<String, serde_json::Value> = (0..count)
        .map(|i| (format!("t{i}"), serde_json::json!(i)))
        .collect();
    std::fs::write(
        dir.path().join("tokenizer.json"),
        serde_json::to_vec(&serde_json::json!({
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null,"pre_tokenizer":{"type":"Whitespace"}, "post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":vocabulary,"unk_token":"t0"}
        }))?,
    )?;
    Ok(dir)
}

#[test]
fn state_file_compressed_records_cursors_and_partial_carries_survive_restart() -> anyhow::Result<()>
{
    let root = compressed_package()?;
    for prompt in [vec![1], vec![1, 2, 3], vec![1, 2, 3, 4, 5, 6, 7]] {
        let mut engine = load(root.path())?;
        let session = engine.create_session()?;
        engine.generate_in_session(session, request(prompt, 1))?;
        let native = engine.native_session.as_ref().unwrap();
        let record = native
            .compressed_state
            .records()
            .next()
            .context("fixture needs compressed record")?;
        let name = record.input.clone();
        let axis = record.sequence_axis;
        let file = save(&mut engine, session)?;
        let tokens = engine.native_sessions[&session].tokens.clone();
        let baseline = engine.native_session.as_mut().unwrap().snapshot_past()?;
        let expected = engine
            .generate_in_session(session, request(tokens.clone(), 3))?
            .token_ids;
        drop(engine);
        let mut restarted = load(root.path())?;
        let target = restarted.create_session()?;
        restarted.restore_native_session_state(target, &mut file.as_slice(), Default::default())?;
        assert_state(&mut restarted, &baseline, &tokens, target)?;
        let forged = edit_manifest(&file, |v| {
            let spec = v["past"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|s| s["name"] == name)
                .unwrap();
            spec["shape"][axis] = serde_json::json!(spec["shape"][axis].as_u64().unwrap() + 1);
            let shape = spec["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            spec["bytes"] = serde_json::json!(
                bytes(
                    dtype(spec["dtype"].as_u64().unwrap() as u8).unwrap(),
                    &shape
                )
                .unwrap()
            );
        })?;
        let error = restarted
            .restore_native_session_state(target, &mut forged.as_slice(), Default::default())
            .unwrap_err();
        assert!(
            error
                .downcast_ref::<crate::native_decode::CompressedStateTransitionRefusal>()
                .is_some(),
            "{error:#}"
        );
        assert_state(&mut restarted, &baseline, &tokens, target)?;
        assert_eq!(
            restarted
                .generate_in_session(target, request(tokens, 3))?
                .token_ids,
            expected
        );
    }
    Ok(())
}

#[test]
fn state_file_invalid_inplace_binding_is_rejected_before_capacity_or_state_changes()
-> anyhow::Result<()> {
    let mut engine = load(&path("tiny-native-scalar-gqa"))?;
    let session = engine.create_session()?;
    engine.generate_in_session(session, request(vec![0], 3))?;
    let file = save(&mut engine, session)?;
    let baseline = engine.native_session.as_mut().unwrap().snapshot_past()?;
    let tokens = engine.native_sessions[&session].tokens.clone();
    let forged = edit_manifest(&file, |v| {
        v["cpu"][1]["pair"][1] = serde_json::json!("wrong-port")
    })?;
    let error = engine
        .restore_native_session_state(session, &mut forged.as_slice(), Default::default())
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("mismatched ports/dtype/geometry/cursor")
    );
    assert_state(&mut engine, &baseline, &tokens, session)?;
    Ok(())
}

#[test]
fn state_file_writer_io_failure_does_not_change_or_retain_session_state() -> anyhow::Result<()> {
    struct FullDisk;
    impl Write for FullDisk {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("test disk is full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut engine = load(&path("tiny-native-engine"))?;
    let session = engine.create_session()?;
    engine.generate_in_session(session, request(vec![0], 3))?;
    let baseline = engine.native_session.as_mut().unwrap().snapshot_past()?;
    let tokens = engine.native_sessions[&session].tokens.clone();
    let before = MemoryGovernor::used(engine.governor.memory(), Tier::Host);
    let error = engine
        .write_native_session_state(session, &mut FullDisk, Default::default())
        .unwrap_err();
    let message = format!("{error:#}");
    assert!(
        message.contains("storage space and permissions") && message.contains("disk is full"),
        "{message}"
    );
    assert_state(&mut engine, &baseline, &tokens, session)?;
    assert_eq!(
        MemoryGovernor::used(engine.governor.memory(), Tier::Host),
        before
    );
    Ok(())
}

#[test]
fn state_file_configuration_rejects_unsupported_execution_paths_at_load() -> anyhow::Result<()> {
    let error = Engine::from_dir(
        &path("tiny-native-engine"),
        EngineConfig {
            decode_backend: EngineDecodeBackend::Ort,
            native_state_files: crate::NativeStateFilePolicy::Required,
            ..Default::default()
        },
    )
    .err()
    .context("ORT state files must fail before loading a decoder")?;
    assert!(
        error.to_string().contains("requires backend=native"),
        "{error:#}"
    );
    let error = Engine::from_dir(
        &path("onnx_genai_workflows/reranking"),
        EngineConfig {
            native_state_files: crate::NativeStateFilePolicy::Required,
            ..Default::default()
        },
    )
    .err()
    .context("interpreted state must not silently ignore state files")?;
    assert!(
        error.to_string().contains("interpreted workflow state"),
        "{error:#}"
    );
    Ok(())
}
