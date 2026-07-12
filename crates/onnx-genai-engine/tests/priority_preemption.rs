use onnx_genai_engine::{
    Engine, EngineConfig, FinishReason, GeneratePrompt, GenerateRequest,
    PrioritizedGenerateRequest, ScheduledGenerateArrival,
};
use onnx_genai_scheduler::{PreemptionPolicy, Priority, PriorityPolicy, SchedulerConfig};
use std::path::{Path, PathBuf};

fn tiny_fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm")
        .canonicalize()?)
}

fn priority_config() -> EngineConfig {
    EngineConfig {
        scheduler: SchedulerConfig {
            max_batch_size: 1,
            max_total_tokens: 1024,
            priority_policy: PriorityPolicy::Priority,
            preemption_policy: PreemptionPolicy::Swap,
        },
        ..Default::default()
    }
}

fn token_request(tokens: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    request
}

#[test]
fn higher_priority_request_runs_before_earlier_lower_priority_request() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&tiny_fixture()?, priority_config())?;
    let low_session = engine.create_session()?;
    let high_session = engine.create_session()?;

    let results = engine.drive_prioritized_requests(vec![
        PrioritizedGenerateRequest {
            session_id: low_session,
            request: token_request(vec![2, 4, 3], 1),
            priority: Priority::Low,
        },
        PrioritizedGenerateRequest {
            session_id: high_session,
            request: token_request(vec![2, 5, 3], 1),
            priority: Priority::High,
        },
    ])?;

    assert_eq!(results[0].session_id, high_session);
    assert_eq!(results[1].session_id, low_session);
    assert_eq!(results[0].result.finish_reason, FinishReason::MaxTokens);
    assert_eq!(results[1].result.finish_reason, FinishReason::MaxTokens);
    engine.close_session(low_session)?;
    engine.close_session(high_session)?;
    Ok(())
}

#[test]
fn high_priority_arrival_preempts_low_priority_and_both_complete() -> anyhow::Result<()> {
    let fixture = tiny_fixture()?;

    let low_expected = {
        let mut engine = Engine::from_dir(&fixture, priority_config())?;
        let session = engine.create_session()?;
        engine.generate_in_session(session, token_request(vec![2, 4, 3], 4))?
    };
    let high_expected = {
        let mut engine = Engine::from_dir(&fixture, priority_config())?;
        let session = engine.create_session()?;
        engine.generate_in_session(session, token_request(vec![2, 5, 3], 2))?
    };

    let mut engine = Engine::from_dir(&fixture, priority_config())?;
    let low_session = engine.create_session()?;
    let high_session = engine.create_session()?;

    let results = engine.drive_prioritized_arrivals(vec![
        ScheduledGenerateArrival {
            arrival_step: 0,
            request: PrioritizedGenerateRequest {
                session_id: low_session,
                request: token_request(vec![2, 4, 3], 4),
                priority: Priority::Low,
            },
        },
        ScheduledGenerateArrival {
            arrival_step: 1,
            request: PrioritizedGenerateRequest {
                session_id: high_session,
                request: token_request(vec![2, 5, 3], 2),
                priority: Priority::High,
            },
        },
    ])?;

    let high = results
        .iter()
        .find(|result| result.session_id == high_session)
        .expect("high-priority result missing");
    let low = results
        .iter()
        .find(|result| result.session_id == low_session)
        .expect("low-priority result missing");

    assert_eq!(results[0].session_id, high_session);
    assert_eq!(high.result.token_ids, high_expected.token_ids);
    assert_eq!(low.result.token_ids, low_expected.token_ids);
    assert_eq!(high.result.token_ids.len(), 2);
    assert_eq!(low.result.token_ids.len(), 4);
    assert_eq!(high.result.finish_reason, FinishReason::MaxTokens);
    assert_eq!(low.result.finish_reason, FinishReason::MaxTokens);
    engine.close_session(low_session)?;
    engine.close_session(high_session)?;
    Ok(())
}
