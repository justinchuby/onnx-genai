use onnx_genai_engine::{
    ContinuousBatchDiagnostic, ContinuousBatchEvent, Engine, EngineConfig, GenerateOptions,
    GeneratePrompt, GenerateRequest, GenerateResult,
};
use onnx_genai_ort::SessionOptions;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Once;

fn shared_buffer_fixture() -> anyhow::Result<tempfile::TempDir> {
    static ENABLE_SHARED_PRESENT: Once = Once::new();
    ENABLE_SHARED_PRESENT.call_once(|| {
        // This isolated test binary sets the CPU fixed-present opt-in before
        // runtime configuration is first read.
        unsafe { std::env::set_var("ONNX_GENAI_SHARED_KV_PRESENT_BINDING", "1") };
    });
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm-sharedbuffer")
        .canonicalize()?;
    let fixture_parent = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/test-fixtures");
    fs::create_dir_all(&fixture_parent)?;
    let root = tempfile::Builder::new()
        .prefix("live-row-shared-")
        .tempdir_in(fixture_parent)?;
    for entry in fs::read_dir(&source)? {
        let entry = entry?;
        if entry.file_name() != "model.onnx" && entry.file_name() != "inference_metadata.yaml" {
            fs::copy(entry.path(), root.path().join(entry.file_name()))?;
        }
    }
    let metadata = fs::read_to_string(source.join("inference_metadata.yaml"))?;
    fs::write(
        root.path().join("inference_metadata.yaml"),
        metadata.replacen("aliasing: forbidden", "aliasing: permitted", 1),
    )?;
    let textproto = fs::read_to_string(source.join("model.onnx.textproto"))?;
    fs::write(
        root.path().join("model.onnx"),
        onnx_std::textproto::to_binary(&textproto)?,
    )?;
    Ok(root)
}

fn cpu_engine(model_dir: &Path) -> anyhow::Result<Engine> {
    Engine::from_dir_with_session_options(
        model_dir,
        EngineConfig::default(),
        SessionOptions::default().with_intra_op_threads(1),
    )
}

fn ragged_requests() -> Vec<GenerateRequest> {
    [
        (vec![1], 1),
        (vec![1, 2, 3], 4),
        (vec![2, 1], 2),
        (vec![3, 2, 1, 2], 3),
    ]
    .into_iter()
    .map(|(tokens, max_new_tokens)| GenerateRequest {
        prompt: GeneratePrompt::TokenIds(tokens),
        options: GenerateOptions {
            greedy: true,
            stop_on_eos: false,
            max_new_tokens,
            ..GenerateOptions::default()
        },
    })
    .collect()
}

fn isolated_results(
    fixture: &Path,
    requests: &[GenerateRequest],
) -> anyhow::Result<Vec<GenerateResult>> {
    requests
        .iter()
        .cloned()
        .map(|request| cpu_engine(fixture)?.generate(request))
        .collect()
}

fn assert_live_row_ownership(diagnostic: &ContinuousBatchDiagnostic) {
    let mut owners = HashSet::new();
    for row in &diagnostic.rows {
        if let Some(handle) = row.handle {
            assert!(
                owners.insert(handle),
                "handle {handle:?} owns more than one physical row: {diagnostic:?}"
            );
            assert!(
                row.resident_tokens <= row.logical_tokens,
                "physical row {} is resident beyond its logical request position: {row:?}",
                row.physical_row
            );
            assert!(
                row.generated_tokens <= row.logical_tokens,
                "row {} published more tokens than it owns: {row:?}",
                row.physical_row
            );
        } else {
            assert_eq!(
                row.logical_tokens, 0,
                "vacant physical row {} retains logical request ownership: {row:?}",
                row.physical_row
            );
        }
    }
    for handle in &diagnostic.queued_handles {
        assert!(
            !owners.contains(handle),
            "queued handle {handle:?} is also resident: {diagnostic:?}"
        );
    }
}

#[test]
fn static_shared_batch_matches_isolated_ragged_rows() -> anyhow::Result<()> {
    let fixture = shared_buffer_fixture()?;
    let requests = ragged_requests();
    let expected = isolated_results(fixture.path(), &requests)?;
    let mut engine = cpu_engine(fixture.path())?;
    assert!(
        engine.batching_capability().supports_batching(),
        "fixture must exercise the production shared-forward path"
    );

    assert_eq!(engine.generate_batched_static(requests.clone())?, expected);
    assert_eq!(
        engine.generate_batched_static(requests)?,
        expected,
        "a second static batch on the same engine must not inherit row state, output, or residency"
    );
    Ok(())
}

#[test]
fn scheduled_shared_batch_matches_isolated_through_backfill_and_row_reuse() -> anyhow::Result<()> {
    let fixture = shared_buffer_fixture()?;
    let requests = ragged_requests();
    let expected = isolated_results(fixture.path(), &requests)?;
    let mut engine = cpu_engine(fixture.path())?;
    assert!(
        engine.batching_capability().supports_batching(),
        "fixture must exercise the production shared-forward path"
    );

    assert_eq!(
        engine.run_continuous_batch_scheduled(requests.clone(), 2)?,
        expected
    );
    assert_eq!(
        engine.run_continuous_batch_scheduled(requests, 2)?,
        expected,
        "two physical rows must retire unequal requests, backfill and reuse slots without \
         carrying prior output, RNG, completion, state ownership, or residency"
    );
    Ok(())
}

#[test]
fn live_row_residency_journal_and_completion_ownership_survive_backfill() -> anyhow::Result<()> {
    let fixture = shared_buffer_fixture()?;
    let requests = ragged_requests();
    let expected = isolated_results(fixture.path(), &requests)?;
    let mut engine = cpu_engine(fixture.path())?;
    let mut manager = engine.continuous_batch_manager(2)?;
    let handles = requests
        .into_iter()
        .map(|request| manager.submit(request))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let initial = manager.diagnostic()?;
    assert_eq!(initial.queued_handles, handles);
    assert!(initial.rows.iter().all(|row| row.handle.is_none()));
    assert_live_row_ownership(&initial);

    let mut completed = HashMap::new();
    let mut saw_journal = false;
    while manager.has_pending_work() {
        manager.step()?;
        let diagnostic = manager.diagnostic()?;
        assert_live_row_ownership(&diagnostic);
        saw_journal |= diagnostic
            .committed_output_journal
            .iter()
            .any(|tokens| !tokens.is_empty());
        for event in manager.poll() {
            if let ContinuousBatchEvent::Finished { handle, result } = event {
                assert!(
                    completed.insert(handle, result).is_none(),
                    "handle {handle:?} completed more than once"
                );
            }
        }
    }
    manager.drain()?;

    assert!(
        saw_journal,
        "the authored output journal must record tokens before pass completion"
    );
    assert_eq!(completed.len(), handles.len());
    for (index, handle) in handles.into_iter().enumerate() {
        assert_eq!(
            completed.remove(&handle),
            Some(expected[index].clone()),
            "backfill or row reuse leaked state into request {index}"
        );
    }
    let final_state = manager.diagnostic()?;
    assert!(final_state.rows.iter().all(|row| row.handle.is_none()));
    assert!(final_state.queued_handles.is_empty());
    assert_live_row_ownership(&final_state);
    Ok(())
}
