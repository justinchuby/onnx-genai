use onnx_genai_engine::{
    Engine, EngineConfig, FinishReason, GeneratePrompt, GenerateRequest,
    PrioritizedGenerateRequest, ScheduledGenerateArrival, SpeculativeMode,
};
use onnx_genai_scheduler::{PreemptionPolicy, Priority, PriorityPolicy, SchedulerConfig};
use std::path::{Path, PathBuf};

fn tiny_fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm")
        .canonicalize()?)
}

fn cursor_fallback_fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm-cursor-fallback")
        .canonicalize()?)
}

fn priority_config() -> EngineConfig {
    EngineConfig {
        scheduler: SchedulerConfig {
            max_batch_size: 1,
            max_total_tokens: 1024,
            priority_policy: PriorityPolicy::Priority,
            preemption_policy: PreemptionPolicy::Swap,
            bytes_per_token: None,
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
fn cursor_ineligible_topology_dispatches_through_generic_generation() -> anyhow::Result<()> {
    let request = token_request(vec![2, 4, 3], 2);
    let expected = {
        let mut engine = Engine::from_dir(&tiny_fixture()?, priority_config())?;
        let session = engine.create_session()?;
        engine.generate_in_session(session, request.clone())?
    };

    let mut engine = Engine::from_dir(&cursor_fallback_fixture()?, priority_config())?;
    let session = engine.create_session()?;
    let results = engine.drive_prioritized_requests(vec![PrioritizedGenerateRequest {
        session_id: session,
        request,
        priority: Priority::High,
    }])?;

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session_id, session);
    assert_eq!(results[0].result.token_ids, expected.token_ids);
    assert_eq!(results[0].result.text, expected.text);
    assert_eq!(results[0].result.finish_reason, expected.finish_reason);
    engine.close_session(session)?;
    Ok(())
}

#[test]
fn valid_speculative_request_dispatches_through_generic_generation_and_preserves_session()
-> anyhow::Result<()> {
    let fixture = tiny_fixture()?;
    let prompt = vec![3, 26, 11, 9, 29, 3, 26, 11, 9, 29];
    let mut speculative = token_request(prompt, 4);
    speculative.options.speculative_mode = Some(SpeculativeMode::PromptLookup {
        ngram: 1,
        max_tokens: 4,
    });

    let mut direct = Engine::from_dir(&fixture, priority_config())?;
    let direct_session = direct.create_session()?;
    let expected = direct.generate_in_session(direct_session, speculative.clone())?;
    assert!(
        direct.last_speculative_stats().proposed_tokens > 0,
        "the direct generic path must execute speculative proposal/verification"
    );

    let mut prioritized = Engine::from_dir(&fixture, priority_config())?;
    let prioritized_session = prioritized.create_session()?;
    let results = prioritized.drive_prioritized_requests(vec![PrioritizedGenerateRequest {
        session_id: prioritized_session,
        request: speculative,
        priority: Priority::High,
    }])?;

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].result.token_ids, expected.token_ids);
    assert_eq!(results[0].result.text, expected.text);
    assert_eq!(results[0].result.finish_reason, expected.finish_reason);
    assert!(
        prioritized.last_speculative_stats().proposed_tokens > 0,
        "typed cursor ineligibility must dispatch through the generic speculative path"
    );

    let continuation = token_request(vec![7], 2);
    let expected_continuation = direct.generate_in_session(direct_session, continuation.clone())?;
    let actual_continuation = prioritized.generate_in_session(prioritized_session, continuation)?;
    assert_eq!(
        actual_continuation.token_ids, expected_continuation.token_ids,
        "cursor preflight must not mutate session state before generic fallback"
    );
    assert_eq!(actual_continuation.text, expected_continuation.text);
    assert_eq!(
        actual_continuation.finish_reason,
        expected_continuation.finish_reason
    );

    direct.close_session(direct_session)?;
    prioritized.close_session(prioritized_session)?;
    Ok(())
}

#[test]
fn invalid_speculative_request_does_not_enter_cursor_fallback() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&tiny_fixture()?, priority_config())?;
    let session = engine.create_session()?;
    let mut request = token_request(vec![2, 4, 3], 2);
    request.options.speculative_mode = Some(SpeculativeMode::PromptLookup {
        ngram: 0,
        max_tokens: 4,
    });

    let error = match engine.drive_prioritized_requests(vec![PrioritizedGenerateRequest {
        session_id: session,
        request,
        priority: Priority::High,
    }]) {
        Ok(_) => panic!("an invalid speculative contract must fail closed"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("prompt-lookup ngram must be greater than zero"),
        "{error:#}"
    );

    let recovered = engine.generate_in_session(session, token_request(vec![2, 4, 3], 1))?;
    assert_eq!(recovered.finish_reason, FinishReason::MaxTokens);
    engine.close_session(session)?;
    Ok(())
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

/// The engine must actually MOVE the preempted sequence's KV off the hot tier
/// (not just skip running it), and the preempted-then-restored sequence must
/// still decode byte-identical tokens. Eviction stats prove the movement
/// happened; the token equality proves preemption did not change the output.
#[test]
fn preemption_evicts_kv_and_preserves_output() -> anyhow::Result<()> {
    let fixture = tiny_fixture()?;

    let low_expected = {
        let mut engine = Engine::from_dir(&fixture, priority_config())?;
        let session = engine.create_session()?;
        engine.generate_in_session(session, token_request(vec![2, 4, 3], 6))?
    };

    let mut engine = Engine::from_dir(&fixture, priority_config())?;
    let low_session = engine.create_session()?;
    let high_session = engine.create_session()?;

    let baseline_evictions = engine.page_stats().hot_evictions;

    let results = engine.drive_prioritized_arrivals(vec![
        ScheduledGenerateArrival {
            arrival_step: 0,
            request: PrioritizedGenerateRequest {
                session_id: low_session,
                request: token_request(vec![2, 4, 3], 6),
                priority: Priority::Low,
            },
        },
        ScheduledGenerateArrival {
            arrival_step: 2,
            request: PrioritizedGenerateRequest {
                session_id: high_session,
                request: token_request(vec![2, 5, 3], 2),
                priority: Priority::High,
            },
        },
    ])?;

    let low = results
        .iter()
        .find(|result| result.session_id == low_session)
        .expect("low-priority result missing");

    // The engine executed real KV eviction when it preempted the low sequence.
    assert!(
        engine.page_stats().hot_evictions > baseline_evictions,
        "expected the engine to evict the preempted sequence's KV off the hot tier \
         (baseline {baseline_evictions}, after {})",
        engine.page_stats().hot_evictions
    );

    // Preemption is a memory optimization, not an output change.
    assert_eq!(low.result.token_ids, low_expected.token_ids);
    assert_eq!(low.result.finish_reason, FinishReason::MaxTokens);

    engine.close_session(low_session)?;
    engine.close_session(high_session)?;
    Ok(())
}

/// The single-sequence / no-pressure path must be behavior-identical: no
/// preemption is emitted, so the engine performs no KV eviction and produces
/// the same tokens as a plain in-session generate.
#[test]
fn no_preemption_path_does_not_evict_kv() -> anyhow::Result<()> {
    let fixture = tiny_fixture()?;

    let expected = {
        let mut engine = Engine::from_dir(&fixture, priority_config())?;
        let session = engine.create_session()?;
        engine.generate_in_session(session, token_request(vec![2, 4, 3], 5))?
    };

    let mut engine = Engine::from_dir(&fixture, priority_config())?;
    let session = engine.create_session()?;
    let baseline_evictions = engine.page_stats().hot_evictions;

    let results = engine.drive_prioritized_requests(vec![PrioritizedGenerateRequest {
        session_id: session,
        request: token_request(vec![2, 4, 3], 5),
        priority: Priority::Normal,
    }])?;

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].result.token_ids, expected.token_ids);
    assert_eq!(
        engine.page_stats().hot_evictions,
        baseline_evictions,
        "the no-preemption path must not evict any KV"
    );

    engine.close_session(session)?;
    Ok(())
}
