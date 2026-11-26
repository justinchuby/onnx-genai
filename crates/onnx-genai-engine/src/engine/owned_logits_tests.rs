//! Real Engine turn/row admission, not a fake inference backend.
use super::*;
use onnx_runtime_ep_cpu::{CpuBufferMemory, CpuCacheRetention, CpuExecutionProvider};
use onnx_runtime_memory_api::Tier;
use onnx_runtime_memory_governor::MemoryGovernor;
use onnx_runtime_session::{HostTensorAllocator, InferenceSession, OwnedHostArray};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

fn managed_engine() -> anyhow::Result<(Engine, HostTensorAllocator)> {
    let mut engine =
        super::tests::model_free_rewind_test_engine(PagedKvCache::new(1, 1), HashMap::new())?;
    let limits = ResourceLimits {
        host_ram_limit: ResourceLimit::Bytes(1024 * 1024),
        ..ResourceLimits::default()
    };
    let governor = Arc::new(EngineResourceGovernor::new(
        limits,
        false,
        ModelKvConfig::known(1, 1),
        0,
    )?);
    let memory = CpuBufferMemory::governed(
        &governor.process_memory_manager(),
        Arc::new(governor.memory().clone()),
    )?;
    let resources = governor
        .cpu_memory_resources(CpuCacheRetention::default())
        .with_buffer_memory(memory)?;
    let cpu = Arc::new(CpuExecutionProvider::initialized_with_resources(resources)?);
    let original = super::tests::tiny_dense_native_decoder();
    let graph = original.graph().clone();
    drop(original);
    let session = InferenceSession::from_graph_with_provider(
        graph,
        Arc::new(onnx_runtime_loader::WeightStore::new()),
        Path::new("."),
        cpu,
        None,
    )?;
    let context = session.host_tensor_allocator()?;
    engine.governor = governor;
    engine.decode_backend = EngineDecodeBackend::Native;
    engine.native_session = Some(
        crate::native_decode::NativeDecodeSession::from_session_with_io(
            session,
            &super::tests::tiny_dense_decoder_io(),
        )?,
    );
    Ok((engine, context))
}
fn request(source: crate::LogprobSource) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![1, 2]));
    request.options.max_new_tokens = 1;
    request.options.stop_on_eos = false;
    request.options.seed = Some(7);
    request.options.greedy = false;
    request.options.max_constraint_bytes = 4096;
    request.options.temperature = 1.0;
    request.options.top_k = 0;
    request.options.top_p = 1.0;
    request.options.min_p = 0.0;
    request.options.top_logprobs = Some(2);
    request.options.logprob_source = source;
    request
        .options
        .logit_bias
        .insert(1, crate::logits::LogitBias::Add(1.0));
    request
}
#[test]
fn actual_native_row_copy_denial_after_forward_rolls_back_and_same_engine_retry_preserves_model_probabilities()
-> anyhow::Result<()> {
    let (mut engine, context) = managed_engine()?;
    let session = engine.create_session()?;
    let memory = engine.governor.memory().clone();
    let initial_used = memory.used(Tier::Host);
    let fired = Arc::new(AtomicBool::new(false));
    let forwarded_used = Arc::new(AtomicU64::new(0));
    let processed = Arc::new(AtomicUsize::new(0));
    let blocker = Arc::new(Mutex::new(None::<OwnedHostArray<u8>>));
    let control = crate::GenerationControl::new()
        .with_checkpoint({
            let fired = Arc::clone(&fired);
            let blocker = Arc::clone(&blocker);
            let context = context.clone();
            let forwarded_used = Arc::clone(&forwarded_used);
            move |boundary| {
                if boundary == crate::GenerationBoundary::AfterDecoderForward
                    && !fired.swap(true, Ordering::Relaxed)
                {
                    // This is the production checkpoint after the real decoder
                    // advanced KV. Do not fail the checkpoint: fill remaining
                    // actual headroom so the following COW row request is denied.
                    forwarded_used.store(memory.used(Tier::Host), Ordering::Relaxed);
                    let available = memory.available(Tier::Host);
                    anyhow::ensure!(
                        available > 8,
                        "fixture must reach sampling with meaningful Host headroom"
                    );
                    *blocker.lock().unwrap() =
                        Some(context.array::<u8>(usize::try_from(available)?)?);
                }
                Ok(())
            }
        })
        .with_prompt_progress({
            let processed = Arc::clone(&processed);
            move |progress| {
                processed.fetch_max(progress.processed, Ordering::Relaxed);
                Ok(())
            }
        });
    let failed = engine
        .generate_in_session_with_control_callbacks(
            session,
            request(crate::LogprobSource::Model),
            control,
            None,
            None,
        )
        .unwrap_err();
    assert!(
        format!("{failed:#}").contains("admit mutable Native logits row"),
        "{failed:#}"
    );
    assert!(fired.load(Ordering::Relaxed));
    assert_eq!(
        processed.load(Ordering::Relaxed),
        2,
        "actual prompt forward must finish before row-copy denial"
    );
    assert!(forwarded_used.load(Ordering::Relaxed) > initial_used);
    assert_eq!(engine.session_token_count(session)?, 0);
    assert_eq!(engine.native_session.as_ref().unwrap().current_len(), 0);
    drop(blocker.lock().unwrap().take());
    let retried = engine.generate_in_session(session, request(crate::LogprobSource::Model))?;
    let (mut cold, _) = managed_engine()?;
    let cold_session = cold.create_session()?;
    let reference = cold.generate_in_session(cold_session, request(crate::LogprobSource::Model))?;
    assert_eq!(
        retried.token_ids, reference.token_ids,
        "denial/retry must preserve the seeded request stream"
    );
    assert_eq!(retried.token_ids.len(), 1);
    let model = retried.logprobs.as_ref().unwrap()[0].logprob;
    assert!(
        (model + std::f32::consts::LN_2).abs() < 1e-6,
        "raw duplicate logits must have model probability 1/2"
    );
    let (mut processed_engine, _) = managed_engine()?;
    let processed_session = processed_engine.create_session()?;
    let processed_result = processed_engine
        .generate_in_session(processed_session, request(crate::LogprobSource::Processed))?;
    assert_eq!(processed_result.token_ids, retried.token_ids);
    let token = processed_result.token_ids[0];
    let normalizer = (1.0_f64 + 1.0_f64.exp()).ln();
    let expected = if token == 1 {
        (1.0 - normalizer) as f32
    } else {
        -normalizer as f32
    };
    let actual = processed_result.logprobs.as_ref().unwrap()[0].logprob;
    assert!(
        (actual - expected).abs() < 1e-6,
        "processed probabilities must include the +1 bias while Model preserves the original row"
    );
    Ok(())
}

#[test]
fn native_token_history_matching_managed_domain_funds_exactly_one_owner_and_aliases()
-> anyhow::Result<()> {
    let (engine, context) = managed_engine()?;
    let memory = engine.governor.memory().clone();
    let manager = engine.governor.process_memory_manager();
    let funding = context
        .host_payload_funding()
        .expect("real managed fixture must report its allocation domain");
    assert!(funding.matches(memory.authority_id(Tier::Host), &manager));
    let unrelated = onnx_runtime_memory_governor::LedgerGovernor::new(
        onnx_runtime_memory_governor::LeaseLedger::new_for_device(
            onnx_runtime_memory_api::DeviceKey::HOST,
            0,
            1024 * 1024,
            0,
        ),
    );
    assert!(
        !funding.matches(unrelated.authority_id(Tier::Host), &manager),
        "matching manager alone must not establish this Engine's funding authority"
    );
    let before = memory.used(Tier::Host);
    let physical = manager.process_used(Tier::Host);
    let history = engine.admit_native_token_history(&[1, 2], &[3])?;
    assert_eq!(history.as_slice(), [1, 2, 3]);
    assert_eq!(memory.used(Tier::Host) - before, 12);
    assert_eq!(manager.process_used(Tier::Host) - physical, 12);
    let alias = history.clone();
    assert_eq!(memory.used(Tier::Host) - before, 12);
    drop(history);
    assert_eq!(alias.as_slice(), [1, 2, 3]);
    assert_eq!(memory.used(Tier::Host) - before, 12);
    drop(alias);
    assert_eq!(memory.used(Tier::Host), before);
    assert_eq!(manager.process_used(Tier::Host), physical);
    Ok(())
}
#[test]
fn native_token_history_foreign_allocator_uses_own_shared_process_capacity() -> anyhow::Result<()> {
    let (mut engine, _context) = managed_engine()?;
    let memory = engine.governor.memory().clone();
    let manager = engine.governor.process_memory_manager();
    let foreign = onnx_runtime_memory_governor::ProcessMemoryManager::new()?;
    assert!(!manager.is_same_instance(&foreign));
    let resources = onnx_runtime_ep_cpu::CpuMemoryResources::default().with_buffer_memory(
        CpuBufferMemory::governed(&foreign, Arc::new(memory.clone()))?,
    )?;
    let cpu = Arc::new(CpuExecutionProvider::initialized_with_resources(resources)?);
    let graph = engine
        .native_session
        .as_ref()
        .unwrap()
        .inference_session()
        .graph()
        .clone();
    let session = InferenceSession::from_graph_with_provider(
        graph,
        Arc::new(onnx_runtime_loader::WeightStore::new()),
        Path::new("."),
        cpu,
        None,
    )?;
    let funding = session
        .host_tensor_allocator()?
        .host_payload_funding()
        .unwrap();
    assert!(!funding.matches(memory.authority_id(Tier::Host), &manager));
    assert!(funding.matches(memory.authority_id(Tier::Host), &foreign));
    engine.native_session = Some(
        crate::native_decode::NativeDecodeSession::from_session_with_io(
            session,
            &super::tests::tiny_dense_decoder_io(),
        )?,
    );
    let before = memory.used(Tier::Host);
    let local = manager.process_used(Tier::Host);
    let other = foreign.process_used(Tier::Host);
    manager.set_process_limit(Tier::Host, local + 11)?;
    let denied = engine.admit_native_token_history(&[1, 2], &[3]);
    assert!(
        denied.is_err(),
        "the owning process quota must deny before allocation"
    );
    assert_eq!(memory.used(Tier::Host), before);
    assert_eq!(manager.process_used(Tier::Host), local);
    assert_eq!(foreign.process_used(Tier::Host), other);
    manager.set_process_limit(Tier::Host, local + 12)?;
    let history = engine.admit_native_token_history(&[1, 2], &[3])?;
    assert_eq!(history.as_slice(), [1, 2, 3]);
    assert_eq!(memory.used(Tier::Host) - before, 12);
    assert_eq!(manager.process_used(Tier::Host), local + 12);
    assert_eq!(
        foreign.process_used(Tier::Host),
        other,
        "foreign manager must not receive an allocation mistaken for this Engine's managed domain"
    );
    let alias = history.clone();
    drop(history);
    assert_eq!(alias.as_slice(), [1, 2, 3]);
    assert_eq!(manager.process_used(Tier::Host), local + 12);
    assert_eq!(memory.used(Tier::Host) - before, 12);
    drop(alias);
    assert_eq!(memory.used(Tier::Host), before);
    assert_eq!(manager.process_used(Tier::Host), local);
    Ok(())
}
