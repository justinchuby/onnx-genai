//! Main generation engine.

pub(crate) use crate::FimConfig;
pub(crate) use crate::config::{ResolvedMtpConfig, validate_resolved_mtp_config};
pub(crate) use crate::connector_bridge::ConnectorBridge;
pub(crate) use crate::decode::{
    DecodeState, ModelDecodePath, detect_model_decode_path, next_session_token_argmax,
    next_session_token_logits, next_session_token_sampled,
};
pub(crate) use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, exceeded_context_limit};
pub(crate) use crate::kv_bridge::{
    KvModelInfo, PlacedPayload, RewindRequest, RewindRunnerPolicy, attach_pages_to_sequence,
    chunk_payload_from_exported, common_prefix_len, exported_layers_from_runner,
    infer_kv_model_info, kv_model_past_is_f32, load_materialized_past,
    ort_session_has_recurrent_state, past_kv_from_payloads, rewind_draft_state_to_len,
    rewind_target_state_to_len, sequence_pages_for_len, validate_draft_state_rewind_to_len,
    validate_target_state_rewind_to_len,
};
pub(crate) use crate::logits::{StopSequence, TokenId};
pub(crate) use crate::processors::{
    build_processor_chain, ensure_constrained_finish, load_fim_config_from_model_dir,
    push_unique_stop_sequence,
};
pub(crate) use crate::sampling::{Sampler, SamplingRng};
pub(crate) use crate::session::{ActiveGenerate, DraftModel, DraftSession, EngineSession};
pub(crate) use anyhow::Context;
pub(crate) use onnx_genai_kv::{
    Device, KvCacheOps, KvDType, LocalTieredConnector, PagedKvCache, PrefixCache,
};
pub(crate) use onnx_genai_metadata::InferenceMetadata;
pub(crate) use onnx_genai_ort::{
    DataType, Eagle3DecodeSession, Environment, ModelDirectory, MtpDecodeSession, Session,
    SessionOptions, Tokenizer,
};
pub(crate) use onnx_genai_scheduler::{
    CapacityProvider, CapacityProviders, FixedCapacity, GovernorReconfigureOutcome,
    GovernorSnapshot, ModelKvConfig, Priority, ResourceError, ResourceGovernor, ResourceLimit,
    ResourceLimits, ScheduleDecision, ScheduledBudgetCap, ScheduledRequest, Scheduler,
    VramBreakdown,
};
pub(crate) use onnx_std::{MetadataHints, MetadataWarning, PlacementStrength};
pub(crate) use std::collections::HashMap;
pub(crate) use std::path::Path;
pub(crate) use std::sync::Arc;

pub use crate::config::{
    DecisionSource, DevicePolicy, DevicePolicyParseError, DryConfig, Eagle3Config, EngineConfig,
    EngineConfigError, EngineDecodeBackend, FinishReason, GenerateConstraint, GenerateOptions,
    GeneratePrompt, GenerateRequest, GenerateResult, GenerateToken, GenerateTokenCallback,
    GenerationBudgetCap, KvConnectorBackend, KvConnectorConfig, LayerWeightBytes, LimitParseError,
    MemoryPolicyApplication, MemoryStrategy, MemoryStrategyDecision, MemoryStrategyPlan,
    MirostatConfig, MirostatVersion, MtpCacheScope, MtpConfig, MtpHiddenLayout, MtpWeightSource,
    PrioritizedGenerateRequest, PrioritizedGenerateResult, RecurrentPrefixCacheStats,
    RewindTokenCount, SamplingOverrides, ScheduledGenerateArrival, SessionCheckpoint,
    SessionForkCapability, SessionId, SessionPosition, SpeculativeMode, TokenLogprob,
    WeightAccessPattern, WeightPlacementReport, XtcConfig, parse_device_policy,
    parse_resource_limit,
};
pub use crate::connector_bridge::{ConnectorLookupOutcome, ConnectorStats};
pub(crate) use crate::speculative::{
    LinearEmbedder, LinearLmHead, MtpEmbedder, MtpLmHead, SpeculativeStats,
    load_target_initializer_adapters,
};
// The MTP proposer is driven only from the native decode path; an ORT-only
// build has no consumer for it and would see an unused import. Its only runtime
// use is the native cold-generation path.
#[cfg(feature = "native-backend")]
pub(crate) use crate::speculative::MtpProposer;

mod capability;
mod decode_backend;
mod governor;
mod ids;
mod load;
pub(crate) mod memory_plan;
mod memory_strategy;
mod metadata;
mod model;
#[cfg(feature = "native-backend")]
mod placement;
mod runtime;
pub(crate) use runtime::apply_eos_policy;
pub(crate) mod session_state;
mod speculative_load;
mod workflow_api;
pub use capability::{PackageCapabilityError, SessionPrefillCarry, package_capability_error};
pub use metadata::graph_port_contracts;

pub(crate) use decode_backend::*;
pub(crate) use governor::*;
pub use governor::{EngineGovernorError, EngineResourceGovernor, resolve_device_vram_limit_bytes};
pub(crate) use ids::SharedSessionIds;
pub(crate) use load::{
    force_managed_weight_streaming_enabled, session_device_domain, validate_shared_authority_limit,
};
pub(crate) use memory_plan::Holder;
pub(crate) use memory_strategy::*;
pub(crate) use metadata::*;
pub use model::Engine;
pub(crate) use model::*;
#[cfg(feature = "native-backend")]
pub(crate) use placement::plan_static_weight_placement;
pub(crate) use speculative_load::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_loop::logprob_for_token;
    use crate::logits::ProcessorContext;
    use crate::processors::{
        finish_reason_after_token, select_next_token, select_next_token_with_sampler,
    };
    use crate::sampling::Sampler;
    #[cfg(feature = "native-backend")]
    use onnx_genai_metadata::{DecoderAbi, KvOwnership, SequenceInputKind};
    #[cfg(feature = "native-backend")]
    use onnx_runtime_ir::{Attribute, DataType as IrDataType, Graph, Node, NodeId, Shape};
    use proptest::prelude::*;
    #[cfg(feature = "native-backend")]
    use std::collections::BTreeMap;
    use std::collections::HashMap;

    #[test]
    fn paged_kv_fork_shares_prefix_then_diverges_copy_on_write() -> anyhow::Result<()> {
        let mut cache = PagedKvCache::new(4, 8);
        let parent = cache.create_sequence();
        cache.append(parent, 6)?;
        let parent_pages = cache.page_table.get_sequence(parent).unwrap().to_vec();

        let child = cache.fork(parent, 6)?;
        let child_pages = cache.page_table.get_sequence(child).unwrap().to_vec();

        assert_eq!(child_pages, parent_pages);
        for page_id in &parent_pages {
            assert_eq!(cache.page_table.pages[page_id].ref_count, 2);
        }

        cache.append(child, 1)?;
        let diverged_child_pages = cache.page_table.get_sequence(child).unwrap().to_vec();
        assert_eq!(cache.len(parent)?, 6);
        assert_eq!(cache.len(child)?, 7);
        assert_eq!(diverged_child_pages[0], parent_pages[0]);
        assert_ne!(diverged_child_pages[1], parent_pages[1]);
        assert_eq!(cache.page_table.pages[&parent_pages[0]].ref_count, 2);
        assert_eq!(cache.page_table.pages[&parent_pages[1]].ref_count, 1);
        assert_eq!(
            cache.page_table.pages[&diverged_child_pages[1]].ref_count,
            1
        );
        Ok(())
    }

    #[test]
    fn cached_prefix_pages_survive_rewind_and_divergent_write() -> anyhow::Result<()> {
        let mut cache = PagedKvCache::new(4, 8);
        let mut prefixes = PrefixCache::new();
        let seq = cache.create_sequence();
        let tokens = vec![10, 11, 12, 13, 14, 15];
        cache.append(seq, tokens.len())?;
        let cached_pages = cache.page_table.get_sequence(seq).unwrap().to_vec();
        prefixes.insert_pages(&tokens, &cached_pages, &mut cache.page_table);

        cache.rewind_to(seq, 2)?;
        cache.append(seq, 1)?;

        let matched = prefixes.lookup_shared(&tokens, &mut cache.page_table);
        assert_eq!(matched.matched_tokens, tokens.len());
        assert_eq!(matched.page_ids, cached_pages);
        for page_id in &matched.page_ids {
            assert!(
                cache
                    .page_table
                    .pages
                    .get(page_id)
                    .is_some_and(|page| page.ref_count > 0),
                "cached prefix referenced reclaimed page {page_id}"
            );
        }
        prefixes.release_shared(&tokens, matched.matched_tokens, &mut cache.page_table);
        Ok(())
    }

    fn model_free_rewind_test_engine(
        kv_cache: PagedKvCache,
        sessions: HashMap<SessionId, EngineSession>,
    ) -> anyhow::Result<Engine> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json")
            .canonicalize()?;
        let tokenizer = Tokenizer::from_file(&fixture)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        let governor = EngineResourceGovernor::new(
            ResourceLimits::default(),
            false,
            ModelKvConfig::known(1, 1),
            0,
        )?;

        Ok(Engine {
            workflow: Box::new(crate::pipeline::generation::test_decoder_runtime()?),
            decode_backend: EngineDecodeBackend::Ort,
            metadata: InferenceMetadata::default(),
            metadata_hints: MetadataHints::default(),
            kv_cache,
            prefix_cache: PrefixCache::new(),
            token_prefix_cache: Vec::new(),
            kv_model: None,
            decode_path: ModelDecodePath::Generic,
            scheduler: Scheduler::new(onnx_genai_scheduler::SchedulerConfig::default()),
            governor,
            sessions,
            workflow_sessions: HashMap::new(),
            workflow_session_ids: SharedSessionIds::new(),
            session: None,
            #[cfg(feature = "native-backend")]
            native_session: None,
            #[cfg(feature = "native-backend")]
            weight_placement: None,
            memory_strategy_plan: MemoryStrategyPlan::unknown(0, None, "test engine fixture"),
            #[cfg(feature = "native-backend")]
            native_sessions: HashMap::new(),
            #[cfg(feature = "native-backend")]
            native_active_session: None,
            #[cfg(feature = "native-backend")]
            native_session_ids: SharedSessionIds::new(),
            #[cfg(feature = "native-backend")]
            native_access_counter: 0,
            #[cfg(feature = "native-backend")]
            native_default_session: None,
            #[cfg(feature = "native-backend")]
            native_max_sessions: 8,
            #[cfg(feature = "native-backend")]
            #[cfg(feature = "native-backend")]
            native_recurrent_prefix_stats: RecurrentPrefixCacheStats::default(),
            draft: None,
            mtp: None,
            eagle3: None,
            tokenizer: Some(tokenizer),
            fim_config: None,
            num_speculative_tokens: 1,
            speculative_mode: SpeculativeMode::None,
            last_speculative_stats: SpeculativeStats::default(),
            connector: ConnectorBridge::null(),
            _environment: None,
        })
    }

    #[cfg(feature = "native-backend")]
    fn insert_test_op(
        graph: &mut Graph,
        op_type: &str,
        inputs: Vec<onnx_runtime_ir::ValueId>,
        output: onnx_runtime_ir::ValueId,
        attributes: &[(&str, Attribute)],
    ) {
        let mut node = Node::new(
            NodeId(0),
            op_type,
            inputs.into_iter().map(Some).collect(),
            vec![output],
        );
        for (name, value) in attributes {
            node.attributes.insert((*name).to_string(), value.clone());
        }
        graph.insert_node(node);
    }

    #[cfg(feature = "native-backend")]
    fn tiny_hybrid_native_decoder() -> onnx_runtime_session::InferenceSession {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 11);
        let batch = graph.intern_symbol("batch");
        let sequence = graph.intern_symbol("sequence");
        let total = graph.intern_symbol("total");
        let past = graph.intern_symbol("past");
        let shape = |dims: &[onnx_runtime_ir::Dim]| -> Shape { dims.to_vec() };

        let input_ids = graph.create_named_value(
            "input_ids",
            IrDataType::Int64,
            shape(&[batch.into(), sequence.into()]),
        );
        let attention_mask = graph.create_named_value(
            "attention_mask",
            IrDataType::Int64,
            shape(&[batch.into(), total.into()]),
        );
        let position_ids = graph.create_named_value(
            "position_ids",
            IrDataType::Int64,
            shape(&[batch.into(), sequence.into()]),
        );
        let conv_state = graph.create_named_value(
            "past_key_values.0.conv_state",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), 1.into()]),
        );
        let recurrent_state = graph.create_named_value(
            "past_key_values.1.recurrent_state",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), 1.into()]),
        );
        let past_key = graph.create_named_value(
            "past_key_values.2.key",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), past.into(), 1.into()]),
        );
        let past_value = graph.create_named_value(
            "past_key_values.2.value",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), past.into(), 1.into()]),
        );
        for input in [
            input_ids,
            attention_mask,
            position_ids,
            conv_state,
            recurrent_state,
            past_key,
            past_value,
        ] {
            graph.add_input(input);
        }

        let cast = graph.create_value(IrDataType::Float32, shape(&[batch.into(), sequence.into()]));
        insert_test_op(
            &mut graph,
            "Cast",
            vec![input_ids],
            cast,
            &[("to", Attribute::Int(1))],
        );
        let current_kv = graph.create_value(
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), sequence.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Unsqueeze",
            vec![cast],
            current_kv,
            &[("axes", Attribute::Ints(vec![1, 3]))],
        );
        let token_sum = graph.create_value(IrDataType::Float32, shape(&[batch.into(), 1.into()]));
        insert_test_op(
            &mut graph,
            "ReduceSum",
            vec![cast],
            token_sum,
            &[
                ("axes", Attribute::Ints(vec![1])),
                ("keepdims", Attribute::Int(1)),
            ],
        );
        let token_state = graph.create_value(
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Unsqueeze",
            vec![token_sum],
            token_state,
            &[("axes", Attribute::Ints(vec![2]))],
        );
        let conv_plus_token = graph.create_value(
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Add",
            vec![conv_state, token_state],
            conv_plus_token,
            &[],
        );
        let present_conv = graph.create_named_value(
            "present.0.conv_state",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Add",
            vec![conv_plus_token, token_state],
            present_conv,
            &[],
        );
        let recurrent_plus_token = graph.create_value(
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Add",
            vec![recurrent_state, token_state],
            recurrent_plus_token,
            &[],
        );
        let recurrent_plus_two_tokens = graph.create_value(
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Add",
            vec![recurrent_plus_token, token_state],
            recurrent_plus_two_tokens,
            &[],
        );
        let present_recurrent = graph.create_named_value(
            "present.1.recurrent_state",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Add",
            vec![recurrent_plus_two_tokens, token_state],
            present_recurrent,
            &[],
        );
        let logits = graph.create_named_value(
            "logits",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), 2.into()]),
        );
        insert_test_op(
            &mut graph,
            "Concat",
            vec![present_conv, present_recurrent],
            logits,
            &[("axis", Attribute::Int(2))],
        );
        let present_key = graph.create_named_value(
            "present.2.key",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), total.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Concat",
            vec![past_key, current_kv],
            present_key,
            &[("axis", Attribute::Int(2))],
        );
        let present_value = graph.create_named_value(
            "present.2.value",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), total.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Concat",
            vec![past_value, current_kv],
            present_value,
            &[("axis", Attribute::Int(2))],
        );
        for output in [
            logits,
            present_conv,
            present_recurrent,
            present_key,
            present_value,
        ] {
            graph.add_output(output);
        }
        onnx_runtime_session::InferenceSession::from_graph(graph).expect("tiny hybrid")
    }

    #[cfg(feature = "native-backend")]
    fn tiny_dense_native_decoder() -> onnx_runtime_session::InferenceSession {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 11);
        let batch = graph.intern_symbol("batch");
        let sequence = graph.intern_symbol("sequence");
        let total = graph.intern_symbol("total");
        let past = graph.intern_symbol("past");
        let shape = |dims: &[onnx_runtime_ir::Dim]| -> Shape { dims.to_vec() };

        let input_ids = graph.create_named_value(
            "input_ids",
            IrDataType::Int64,
            shape(&[batch.into(), sequence.into()]),
        );
        let attention_mask = graph.create_named_value(
            "attention_mask",
            IrDataType::Int64,
            shape(&[batch.into(), total.into()]),
        );
        let position_ids = graph.create_named_value(
            "position_ids",
            IrDataType::Int64,
            shape(&[batch.into(), sequence.into()]),
        );
        let past_key = graph.create_named_value(
            "past_key_values.0.key",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), past.into(), 1.into()]),
        );
        let past_value = graph.create_named_value(
            "past_key_values.0.value",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), past.into(), 1.into()]),
        );
        for input in [
            input_ids,
            attention_mask,
            position_ids,
            past_key,
            past_value,
        ] {
            graph.add_input(input);
        }

        let cast = graph.create_value(IrDataType::Float32, shape(&[batch.into(), sequence.into()]));
        insert_test_op(
            &mut graph,
            "Cast",
            vec![input_ids],
            cast,
            &[("to", Attribute::Int(1))],
        );
        let token_logits = graph.create_value(
            IrDataType::Float32,
            shape(&[batch.into(), sequence.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Unsqueeze",
            vec![cast],
            token_logits,
            &[("axes", Attribute::Ints(vec![2]))],
        );
        let logits = graph.create_named_value(
            "logits",
            IrDataType::Float32,
            shape(&[batch.into(), sequence.into(), 2.into()]),
        );
        insert_test_op(
            &mut graph,
            "Concat",
            vec![token_logits, token_logits],
            logits,
            &[("axis", Attribute::Int(2))],
        );
        let current_kv = graph.create_value(
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), sequence.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Unsqueeze",
            vec![cast],
            current_kv,
            &[("axes", Attribute::Ints(vec![1, 3]))],
        );
        let present_key = graph.create_named_value(
            "present.0.key",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), total.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Concat",
            vec![past_key, current_kv],
            present_key,
            &[("axis", Attribute::Int(2))],
        );
        let present_value = graph.create_named_value(
            "present.0.value",
            IrDataType::Float32,
            shape(&[batch.into(), 1.into(), total.into(), 1.into()]),
        );
        insert_test_op(
            &mut graph,
            "Concat",
            vec![past_value, current_kv],
            present_value,
            &[("axis", Attribute::Int(2))],
        );
        for output in [logits, present_key, present_value] {
            graph.add_output(output);
        }
        onnx_runtime_session::InferenceSession::from_graph(graph).expect("tiny dense")
    }

    #[cfg(feature = "native-backend")]
    fn tiny_dense_decoder_io() -> DecoderAbi {
        DecoderAbi {
            sequence_source: Some(SequenceInputKind::TokenIds),
            kv_ownership: Some(KvOwnership::Owned),
            kv_layout: None,
            token_input: Some("input_ids".into()),
            inputs_embeds_input: None,
            attention_mask_input: Some("attention_mask".into()),
            position_ids_input: Some("position_ids".into()),
            logits_output: Some("logits".into()),
            hidden_output: None,
            kv_inputs: Some(vec![
                "past_key_values.0.key".into(),
                "past_key_values.0.value".into(),
            ]),
            kv_outputs: Some(vec!["present.0.key".into(), "present.0.value".into()]),
            encoder_hidden_states_input: None,
            audio_features_input: None,
            cross_kv_inputs: None,
            cross_kv_outputs: None,
            state_pairs: None,
            optional_inputs: BTreeMap::new(),
            static_cache: None,
            aliasing: None,
        }
    }

    #[cfg(feature = "native-backend")]
    fn native_prefix_snapshot_test_engine() -> anyhow::Result<Engine> {
        let mut engine = model_free_rewind_test_engine(PagedKvCache::new(1, 1), HashMap::new())?;
        engine.decode_backend = EngineDecodeBackend::Native;
        engine.native_session = Some(crate::native_decode::NativeDecodeSession::from_session(
            tiny_hybrid_native_decoder(),
        )?);
        Ok(engine)
    }

    #[cfg(feature = "native-backend")]
    fn native_dense_test_engine() -> anyhow::Result<Engine> {
        let mut engine = model_free_rewind_test_engine(PagedKvCache::new(1, 1), HashMap::new())?;
        engine.decode_backend = EngineDecodeBackend::Native;
        engine.native_session = Some(
            crate::native_decode::NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
                tiny_dense_native_decoder(),
                None,
                Some(&tiny_dense_decoder_io()),
            )?,
        );
        Ok(engine)
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn native_recurrent_prefix_hit_matches_cold_output() -> anyhow::Result<()> {
        let prompt = vec![1, 2, 3, 4, 5];
        let mut cold = native_prefix_snapshot_test_engine()?;
        let mut cold_options = GenerateOptions {
            max_new_tokens: 2,
            stop_on_eos: false,
            cold_start: true,
            ..GenerateOptions::default()
        };
        let cold_result = cold.generate(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(prompt.clone()),
            options: cold_options.clone(),
        })?;

        let mut cached = native_prefix_snapshot_test_engine()?;
        cold_options.cold_start = false;
        cold_options.semantic_prefix_len = Some(3);
        let first = cached.create_session()?;
        let second = cached.create_session()?;
        cached.generate_in_session(
            first,
            GenerateRequest {
                prompt: GeneratePrompt::TokenIds(prompt.clone()),
                options: cold_options.clone(),
            },
        )?;
        let incremental_hits_before =
            crate::native_decode::NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS
                .load(std::sync::atomic::Ordering::Relaxed);
        let hit_result = cached.generate_in_session(
            second,
            GenerateRequest {
                prompt: GeneratePrompt::TokenIds(prompt),
                options: cold_options,
            },
        )?;
        let incremental_hits_after =
            crate::native_decode::NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS
                .load(std::sync::atomic::Ordering::Relaxed);

        assert_eq!(hit_result.token_ids, cold_result.token_ids);
        assert_eq!(hit_result.text, cold_result.text);
        assert_eq!(incremental_hits_after, incremental_hits_before + 1);
        assert_eq!(
            cached.recurrent_prefix_cache_stats(),
            RecurrentPrefixCacheStats {
                lookups: 2,
                hits: 1,
                stores: 1,
                restored_tokens: 3,
            }
        );
        Ok(())
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn native_semantic_prefix_on_dense_session_generates_without_snapshot_support()
    -> anyhow::Result<()> {
        let mut engine = native_dense_test_engine()?;
        let session = engine.create_session()?;
        let result = engine.generate_in_session(
            session,
            GenerateRequest {
                prompt: GeneratePrompt::TokenIds(vec![1, 2, 3, 4]),
                options: GenerateOptions {
                    max_new_tokens: 1,
                    stop_on_eos: false,
                    semantic_prefix_len: Some(2),
                    ..GenerateOptions::default()
                },
            },
        )?;

        assert_eq!(result.token_ids.len(), 1);
        assert_eq!(
            engine.recurrent_prefix_cache_stats(),
            RecurrentPrefixCacheStats::default()
        );
        Ok(())
    }

    #[test]
    fn failed_rewind_of_windowed_evicted_position_leaves_session_unchanged() -> anyhow::Result<()> {
        let mut kv_cache = PagedKvCache::new(4, 8);
        let session_id = kv_cache.create_sequence();
        kv_cache.append(session_id, 5)?;
        let original_kv_len = kv_cache.len(session_id)?;
        let state = EngineSession {
            tokens: vec![10, 11, 12, 13, 14],
            kv_token_count: 5,
            decode_state: DecodeState::for_test_windowed(2, 2),
            draft: None,
            sampled_fastpath_failed: false,
        };
        let mut sessions = HashMap::new();
        sessions.insert(session_id, state);
        let mut engine = model_free_rewind_test_engine(kv_cache, sessions)?;

        let error = engine
            .rewind_session_to(session_id, SessionPosition::new(2))
            .expect_err("evicted sliding-window rewind must fail");

        assert!(
            error.to_string().contains("were evicted"),
            "unexpected error: {error:#}"
        );
        let state = engine.sessions.get(&session_id).expect("session remains");
        assert_eq!(state.tokens, [10, 11, 12, 13, 14]);
        assert_eq!(state.kv_token_count, 5);
        assert_eq!(engine.kv_cache.len(session_id)?, original_kv_len);
        Ok(())
    }

    #[test]
    fn failed_rewind_of_ort_owned_kv_leaves_session_unchanged() -> anyhow::Result<()> {
        let mut kv_cache = PagedKvCache::new(4, 8);
        let session_id = kv_cache.create_sequence();
        kv_cache.append(session_id, 3)?;
        let original_kv_len = kv_cache.len(session_id)?;
        let state = EngineSession {
            tokens: vec![20, 21, 22],
            kv_token_count: 3,
            decode_state: DecodeState::for_test_with_past(HashMap::new()),
            draft: None,
            sampled_fastpath_failed: false,
        };
        let mut sessions = HashMap::new();
        sessions.insert(session_id, state);
        let mut engine = model_free_rewind_test_engine(kv_cache, sessions)?;

        let error = engine
            .rewind_session_to(session_id, SessionPosition::new(1))
            .expect_err("ORT-owned KV without paged materialization must fail");

        assert!(
            error
                .to_string()
                .contains("without paged KV materialization"),
            "unexpected error: {error:#}"
        );
        let state = engine.sessions.get(&session_id).expect("session remains");
        assert_eq!(state.tokens, [20, 21, 22]);
        assert_eq!(state.kv_token_count, 3);
        assert_eq!(engine.kv_cache.len(session_id)?, original_kv_len);
        Ok(())
    }

    #[test]
    fn failed_rewind_of_runner_backed_state_leaves_session_unchanged() -> anyhow::Result<()> {
        let mut kv_cache = PagedKvCache::new(4, 8);
        let session_id = kv_cache.create_sequence();
        kv_cache.append(session_id, 4)?;
        let original_kv_len = kv_cache.len(session_id)?;
        let original_pages = kv_cache
            .page_table
            .get_sequence(session_id)
            .expect("sequence exists")
            .to_vec();
        let state = EngineSession {
            tokens: vec![30, 31, 32, 33],
            kv_token_count: 4,
            decode_state: DecodeState::for_test_runner_backed(),
            draft: None,
            sampled_fastpath_failed: false,
        };
        let mut sessions = HashMap::new();
        sessions.insert(session_id, state);
        let mut engine = model_free_rewind_test_engine(kv_cache, sessions)?;

        let error = engine
            .rewind_session_to(session_id, SessionPosition::new(2))
            .expect_err("runner-backed rewind must fail closed");

        assert!(
            error.to_string().contains("runner-backed decoder state"),
            "unexpected error: {error:#}"
        );
        let state = engine.sessions.get(&session_id).expect("session remains");
        assert_eq!(state.tokens, [30, 31, 32, 33]);
        assert_eq!(state.kv_token_count, 4);
        assert_eq!(engine.kv_cache.len(session_id)?, original_kv_len);
        assert_eq!(
            engine
                .kv_cache
                .page_table
                .get_sequence(session_id)
                .expect("sequence remains"),
            original_pages.as_slice()
        );
        Ok(())
    }

    #[derive(Debug, Clone)]
    enum KvOp {
        Append { session: usize, tokens: usize },
        Rewind { session: usize, position: usize },
        Fork { session: usize, position: usize },
        Remove { session: usize },
    }

    fn kv_op_strategy() -> impl Strategy<Value = KvOp> {
        prop_oneof![
            (0usize..32, 1usize..=3).prop_map(|(session, tokens)| KvOp::Append { session, tokens }),
            (0usize..32, 0usize..96)
                .prop_map(|(session, position)| KvOp::Rewind { session, position }),
            (0usize..32, 0usize..96)
                .prop_map(|(session, position)| KvOp::Fork { session, position }),
            (0usize..32).prop_map(|session| KvOp::Remove { session }),
        ]
    }

    fn assert_live_sequence_refcounts_match_pages(
        cache: &PagedKvCache,
        live: &[(SessionId, usize)],
    ) {
        let mut expected_refs = HashMap::new();
        for &(seq, expected_len) in live {
            assert_eq!(cache.len(seq).unwrap(), expected_len);
            for &page_id in cache.page_table.get_sequence(seq).unwrap() {
                *expected_refs.entry(page_id).or_insert(0u32) += 1;
            }
        }

        for (&page_id, &expected_ref_count) in &expected_refs {
            let page = cache
                .page_table
                .pages
                .get(&page_id)
                .unwrap_or_else(|| panic!("live sequence references missing page {page_id}"));
            assert_eq!(
                page.ref_count, expected_ref_count,
                "page {page_id} refcount must match live sequence references"
            );
        }

        for (&page_id, page) in &cache.page_table.pages {
            let expected = expected_refs.get(&page_id).copied().unwrap_or(0);
            assert_eq!(
                page.ref_count, expected,
                "page {page_id} has a refcount without matching live references"
            );
        }
    }

    proptest! {
        #[test]
        fn paged_kv_refcounts_match_live_sequences_for_random_fork_rewind_interleavings(
            ops in prop::collection::vec(kv_op_strategy(), 1..80)
        ) {
            let mut cache = PagedKvCache::new(4, 256);
            let root = cache.create_sequence();
            let mut live = vec![(root, 0usize)];

            for op in ops {
                match op {
                    KvOp::Append { session, tokens } => {
                        let index = session % live.len();
                        let (seq, len) = &mut live[index];
                        cache.append(*seq, tokens).unwrap();
                        *len += tokens;
                    }
                    KvOp::Rewind { session, position } => {
                        let index = session % live.len();
                        let (seq, len) = &mut live[index];
                        let target = position % (*len + 1);
                        cache.rewind_to(*seq, target).unwrap();
                        *len = target;
                    }
                    KvOp::Fork { session, position } => {
                        let index = session % live.len();
                        let (source, source_len) = live[index];
                        let target = position % (source_len + 1);
                        let child = cache.fork(source, target).unwrap();
                        live.push((child, target));
                    }
                    KvOp::Remove { session } => {
                        if live.len() > 1 {
                            let index = session % live.len();
                            let (seq, _) = live.swap_remove(index);
                            cache.remove(seq).unwrap();
                        }
                    }
                }
                assert_live_sequence_refcounts_match_pages(&cache, &live);
            }
        }
    }

    fn test_model_path(label: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static NEXT_PATH_ID: AtomicUsize = AtomicUsize::new(0);
        std::env::current_dir()
            .expect("current directory")
            .join(format!(
                ".onnx-genai-{label}-{}-{}.onnx",
                std::process::id(),
                NEXT_PATH_ID.fetch_add(1, Ordering::Relaxed)
            ))
    }

    fn write_scan_model(nodes: &[(&str, &str)]) -> std::path::PathBuf {
        write_scan_model_with_weights(nodes, 0)
    }

    /// Build a valid ONNX model whose graph has the given `(domain, op_type)`
    /// nodes, plus `weight_floats` f32 elements of inline initializer data to
    /// simulate an inline-weight export (the qwen3 case). The prost scan must
    /// skip past this initializer (via `Buf::advance`) rather than reading it.
    fn write_scan_model_with_weights(
        nodes: &[(&str, &str)],
        weight_floats: usize,
    ) -> std::path::PathBuf {
        use onnx::ir::{DataType, Dim, Graph, Node, NodeId, TensorData, WeightRef, static_shape};
        use onnx_std as onnx;
        use prost::Message;

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        for (index, &(domain, op_type)) in nodes.iter().enumerate() {
            let output = graph.create_named_value(
                format!("output_{index}"),
                DataType::Float32,
                static_shape([]),
            );
            let mut node = Node::new(NodeId(index as u32), op_type, vec![], vec![output]);
            node.domain = domain.to_string();
            graph.insert_node(node);
            graph.add_output(output);
        }

        if weight_floats > 0 {
            let weight = graph.create_named_value(
                "inline_weight",
                DataType::Float32,
                vec![Dim::from(weight_floats)],
            );
            let bytes = vec![0u8; weight_floats * std::mem::size_of::<f32>()];
            graph.set_initializer(
                weight,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Float32,
                    vec![weight_floats],
                    bytes,
                )),
            );
        }

        let path = test_model_path("control-flow");
        std::fs::write(
            &path,
            onnx::Model::new(graph)
                .to_proto()
                .expect("serialize test model")
                .encode_to_vec(),
        )
        .expect("write test model");
        path
    }

    #[test]
    fn control_flow_scan_ignores_regular_ops() {
        let plain = write_scan_model(&[("", "MatMul"), ("", "Add"), ("", "GroupQueryAttention")]);
        assert!(!model_has_control_flow_nodes(&plain));
        std::fs::remove_file(&plain).ok();
    }

    #[test]
    fn ort_cuda_graph_configuration_is_opt_in() {
        let model = write_scan_model(&[("", "MatMul")]);
        let mut options =
            SessionOptions::with_execution_provider(onnx_genai_ort::ep_selection("cuda"));

        options.graph_capture = false;
        configure_ort_cuda_graph(&mut options, &model);
        assert!(
            !options.graph_capture,
            "ORT capture must stay off by default"
        );

        options.graph_capture = true;
        configure_ort_cuda_graph(&mut options, &model);
        assert!(
            options.graph_capture,
            "an explicit ORT capture opt-in must be preserved"
        );
        std::fs::remove_file(&model).ok();
    }

    #[test]
    fn control_flow_scan_detects_standard_onnx_control_flow_ops() {
        for domain in ["", "ai.onnx"] {
            for op_type in ["If", "Loop", "Scan"] {
                let path = write_scan_model(&[(domain, op_type)]);
                assert!(
                    model_has_control_flow_nodes(&path),
                    "expected standard-domain control-flow op '{domain}:{op_type}' to be detected"
                );
                std::fs::remove_file(&path).ok();
            }
        }
    }

    #[test]
    fn control_flow_scan_ignores_custom_domain_control_flow_names() {
        for op_type in ["If", "Loop", "Scan"] {
            let path = write_scan_model(&[("com.example", op_type)]);
            assert!(
                !model_has_control_flow_nodes(&path),
                "custom-domain op 'com.example:{op_type}' must not disable capture"
            );
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn control_flow_scan_conservatively_skips_uninspectable_models() {
        let missing = std::path::Path::new("does-not-exist-onnx-genai.onnx");
        assert!(model_has_control_flow_nodes(missing));

        let garbage = test_model_path("garbage");
        std::fs::write(&garbage, b"not a protobuf").expect("write garbage model");
        assert!(model_has_control_flow_nodes(&garbage));
        std::fs::remove_file(&garbage).ok();
    }

    #[test]
    fn control_flow_scan_reads_nodes_past_large_inline_weights() {
        // Simulate an inline-weight export (like the qwen3 models, whose
        // `model.onnx` embeds >1 GB of weights): the graph carries a large
        // initializer alongside its nodes. The prost scan must still find
        // the control-flow op — and, for a plain graph, must NOT be fooled into
        // conservatively reporting control flow just because the file is large.
        // 4 Mi f32 elements = 16 MiB of inline initializer data.
        let weight_floats = 4 * 1024 * 1024;

        let with_control_flow =
            write_scan_model_with_weights(&[("", "MatMul"), ("", "If")], weight_floats);
        assert!(
            model_has_control_flow_nodes(&with_control_flow),
            "control-flow op must be detected even behind a large inline initializer"
        );
        std::fs::remove_file(&with_control_flow).ok();

        let plain = write_scan_model_with_weights(
            &[("", "MatMul"), ("", "GroupQueryAttention")],
            weight_floats,
        );
        assert!(
            !model_has_control_flow_nodes(&plain),
            "a large inline-weight model without control flow must remain capture-eligible"
        );
        std::fs::remove_file(&plain).ok();
    }

    #[test]
    fn control_flow_scan_conservatively_handles_truncated_control_flow_model() {
        // A control-flow model whose bytes are truncated anywhere must never
        // parse cleanly into a "no control flow" verdict (which would wrongly
        // enable CUDA graph capture and trigger ORT's ~6x slower per-Run path).
        // Truncation either cuts the graph payload (its length header points past
        // EOF -> None) or stops before the graph is ever seen (no-graph -> None),
        // so every prefix (including the empty file) must fall back to
        // conservative `true`.
        let full = write_scan_model(&[("", "MatMul"), ("", "If")]);
        let bytes = std::fs::read(&full).expect("read full model");
        std::fs::remove_file(&full).ok();

        for truncated_len in 0..bytes.len() {
            let truncated = test_model_path(&format!("truncated-{truncated_len}"));
            std::fs::write(&truncated, &bytes[..truncated_len]).expect("write truncated model");
            assert!(
                model_has_control_flow_nodes(&truncated),
                "a truncated control-flow model (len {truncated_len}) must stay conservative"
            );
            std::fs::remove_file(&truncated).ok();
        }
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn backend_and_control_flow_scans_parse_textproto_fixture() -> anyhow::Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/model.onnx.textproto");

        assert!(!model_requires_native_backend(&path)?);
        assert!(!model_has_control_flow_nodes(&path));
        Ok(())
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn auto_backend_detection_reads_onnx_node_types_not_incidental_strings() {
        use onnx_runtime_loader::proto::{
            ModelProto,
            onnx::{GraphProto, NodeProto, OperatorSetIdProto},
        };

        let mut model = ModelProto {
            producer_name: "BlockQuantizedMatMul appears only in metadata".to_string(),
            graph: Some(GraphProto::default()),
            ..ModelProto::default()
        };
        assert!(!model_proto_requires_native_backend(&model));

        model.graph.as_mut().unwrap().node.push(NodeProto {
            domain: "pkg.nxrt".to_string(),
            op_type: "BlockQuantizedMatMul".to_string(),
            ..NodeProto::default()
        });
        model.opset_import.push(OperatorSetIdProto {
            domain: "pkg.nxrt".to_string(),
            version: 1,
        });
        assert!(model_proto_requires_native_backend(&model));

        model.graph.as_mut().unwrap().node[0].domain = "example.wrong.domain".to_string();
        assert!(!model_proto_requires_native_backend(&model));

        model.graph.as_mut().unwrap().node[0].domain = "pkg.nxrt".to_string();
        model.opset_import[0].version = 2;
        assert!(!model_proto_requires_native_backend(&model));
    }

    #[test]
    fn backend_env_values_are_case_insensitive_and_reject_unknown_values() {
        assert_eq!(
            parse_backend_env("AuTo").unwrap(),
            EngineDecodeBackend::Auto
        );
        assert_eq!(parse_backend_env("ORT").unwrap(), EngineDecodeBackend::Ort);
        assert_eq!(
            parse_backend_env("native").unwrap(),
            EngineDecodeBackend::Native
        );
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Auto, || {
                Ok(Some("nAtIvE".to_owned()))
            })
            .unwrap(),
            EngineDecodeBackend::Native
        );

        let error = parse_backend_env("cuda").unwrap_err().to_string();
        assert!(error.contains("ONNX_GENAI_BACKEND"), "{error}");
        assert!(error.contains("auto, ort, native"), "{error}");
    }

    #[test]
    fn explicit_backend_ignores_env_and_auto_honors_it() {
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Ort, || {
                Err(anyhow::anyhow!("unreadable environment value"))
            })
            .unwrap(),
            EngineDecodeBackend::Ort
        );
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Native, || {
                panic!("explicit backend must not read ONNX_GENAI_BACKEND")
            })
            .unwrap(),
            EngineDecodeBackend::Native
        );
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Auto, || {
                Ok(Some("ort".to_owned()))
            })
            .unwrap(),
            EngineDecodeBackend::Ort
        );
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Auto, || Ok(None)).unwrap(),
            EngineDecodeBackend::Auto
        );
    }

    #[test]
    fn forced_ort_load_failure_includes_native_switch_hint() {
        let error = augment_backend_error::<()>(
            Err(anyhow::anyhow!("simulated native-only model load failure")),
            EngineDecodeBackend::Ort,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("EngineDecodeBackend::Native"), "{error}");
        assert!(error.contains("ONNX_GENAI_BACKEND=native"), "{error}");
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn forced_native_load_or_run_failure_includes_ort_switch_hint() {
        let error = augment_backend_error::<()>(
            Err(anyhow::anyhow!("simulated native decoder load/run failure")),
            EngineDecodeBackend::Native,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("EngineDecodeBackend::Ort"), "{error}");
        assert!(error.contains("ONNX_GENAI_BACKEND=ort"), "{error}");
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn device_policy_reaches_native_load_and_profiles_placement() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-glm52-qmoe-indexshare");
        let config = EngineConfig::from_yaml(
            "serving:\n  memory:\n    weights:\n      device_policy: gpu_layers:1\n",
        )
        .unwrap();
        let config = EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(crate::native_decode::NativeDecodeDevice::Cpu),
            ..config
        };

        let engine = Engine::from_dir(&model_dir, config).unwrap();
        let report = engine
            .weight_placement_report()
            .expect("QMoE fixture should produce a static weight placement report");

        assert!(
            report.explanation.contains("gpu_layers:1"),
            "{}",
            report.explanation
        );
        assert!(
            report.host_bytes > 0,
            "CPU load has no governed device residency budget, so the plan should visibly keep bytes on host: {report:?}"
        );
    }

    #[cfg(not(feature = "native-backend"))]
    #[test]
    fn forced_native_without_feature_reports_how_to_switch() {
        let error = resolve_decode_backend(Path::new("unused.onnx"), EngineDecodeBackend::Native)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ONNX_GENAI_BACKEND=ort"), "{error}");
        assert!(error.contains("EngineDecodeBackend::Ort"), "{error}");
    }

    fn test_capacities() -> CapacityProviders {
        CapacityProviders {
            vram: Arc::new(FixedCapacity::new(1_000, 1_000)),
            host_ram: Arc::new(FixedCapacity::new(2_000, 2_000)),
            disk_spill: None,
        }
    }

    #[test]
    fn governor_handle_reflects_configured_limits_without_a_model() {
        let limits = ResourceLimits {
            vram_limit: ResourceLimit::Fraction(0.5),
            host_ram_limit: ResourceLimit::Auto,
            disk_spill_limit: None,
        };
        let governor = EngineResourceGovernor::new_with_capacities(
            limits.clone(),
            true,
            test_capacities(),
            ModelKvConfig::known(100, 16),
            0,
        )
        .unwrap();
        let snapshot = governor.snapshot();
        assert_eq!(snapshot.configured_limits, limits);
        assert_eq!(snapshot.resolved_limits.vram_bytes, Some(500));
        assert_eq!(snapshot.derived_budget.total_pages, 5);
        assert_eq!(snapshot.vram.headroom, 500);
        assert_eq!(snapshot.host_ram.used, 0);
        assert_eq!(snapshot.host_ram.limit, 500);
        assert_eq!(snapshot.host_ram.headroom, 500);
        assert_eq!(snapshot.disk_spill, None);

        let outcome = governor.set_vram_limit(ResourceLimit::Bytes(800)).unwrap();
        assert_eq!(outcome.new_limits.vram_bytes, Some(800));
        assert_eq!(
            governor.snapshot().configured_limits.vram_limit,
            ResourceLimit::Bytes(800)
        );
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn engine_governors_keep_disabled_host_cache_budgets_governed() {
        let first = EngineResourceGovernor::new_with_capacities(
            ResourceLimits {
                host_ram_limit: ResourceLimit::Bytes(400),
                ..ResourceLimits::default()
            },
            false,
            test_capacities(),
            ModelKvConfig::known(100, 16),
            0,
        )
        .unwrap();
        let second = EngineResourceGovernor::new_with_capacities(
            ResourceLimits {
                host_ram_limit: ResourceLimit::Bytes(900),
                ..ResourceLimits::default()
            },
            false,
            test_capacities(),
            ModelKvConfig::known(100, 16),
            0,
        )
        .unwrap();

        assert_eq!(
            first.weight_offload_host_cache().configured_budget_bytes(),
            0
        );
        assert_eq!(first.weight_offload_host_cache().budget(), (0, true));
        assert_eq!(
            second.weight_offload_host_cache().configured_budget_bytes(),
            0
        );
        assert_eq!(second.weight_offload_host_cache().budget(), (0, true));
    }

    #[test]
    fn an_explicit_byte_limit_is_honored_without_a_device_query() {
        // There is no device-capacity probe on this path, so a fraction of the
        // (unknown) device capacity is unknown. But an absolute byte limit is
        // the caller's authoritative statement and must be honoured exactly,
        // capacity query or not (#947). Host RAM and disk are measured from the
        // OS, so their fractions/auto resolve against real, non-fabricated
        // numbers rather than the old provisional constants.
        const EXPLICIT_VRAM_BYTES: u64 = (40u64 << 30) + 1;
        let limits = ResourceLimits {
            vram_limit: ResourceLimit::Bytes(EXPLICIT_VRAM_BYTES),
            host_ram_limit: ResourceLimit::Fraction(0.5),
            disk_spill_limit: Some(ResourceLimit::Auto),
        };
        let governor =
            EngineResourceGovernor::new(limits, false, ModelKvConfig::known(1, 1), 0).unwrap();
        let snapshot = governor.snapshot();
        assert_eq!(
            snapshot.resolved_limits.vram_bytes,
            Some(EXPLICIT_VRAM_BYTES),
            "an explicit byte limit must be honoured exactly, with no device query"
        );
        let host_ram_bytes = snapshot.resolved_limits.host_ram_bytes;
        assert!(
            host_ram_bytes > 0,
            "a fraction of measured host RAM must be a real, positive number"
        );
        assert_ne!(
            host_ram_bytes,
            (16u64 << 30) / 2,
            "host RAM must be measured, not half of the old fabricated 16 GiB constant"
        );
        assert!(
            snapshot.resolved_limits.disk_spill_bytes.unwrap_or(0) > 0,
            "auto disk spill must resolve against measured free space"
        );
    }

    #[test]
    fn governor_snapshot_reports_usage_limit_and_headroom_for_each_enabled_tier() {
        let capacities = CapacityProviders {
            vram: Arc::new(FixedCapacity::new(1_000, 900)),
            host_ram: Arc::new(FixedCapacity::new(2_000, 1_500)),
            disk_spill: Some(Arc::new(FixedCapacity::new(4_000, 3_000))),
        };
        let governor = EngineResourceGovernor::new_with_capacities(
            ResourceLimits {
                vram_limit: ResourceLimit::Bytes(800),
                host_ram_limit: ResourceLimit::Bytes(1_200),
                disk_spill_limit: Some(ResourceLimit::Bytes(3_000)),
            },
            false,
            capacities,
            ModelKvConfig::known(100, 16),
            0,
        )
        .unwrap();
        use onnx_runtime_memory_governor::{HolderId, MemoryGovernor as _, MemoryRole, Tier};

        let _device = governor
            .memory()
            .reserve(Tier::Device, 300, MemoryRole::KvCache, HolderId::new(1))
            .unwrap();
        let _host = governor
            .memory()
            .reserve(Tier::Host, 500, MemoryRole::KvCache, HolderId::new(2))
            .unwrap();
        let _disk = governor
            .memory()
            .reserve(Tier::Disk, 1_000, MemoryRole::KvCache, HolderId::new(3))
            .unwrap();

        let snapshot = governor.snapshot();
        assert_eq!(
            snapshot.vram,
            onnx_genai_scheduler::TierSnapshot {
                used: 300,
                limit: 800,
                headroom: 500,
            }
        );
        assert_eq!(
            snapshot.host_ram,
            onnx_genai_scheduler::TierSnapshot {
                used: 500,
                limit: 1_200,
                headroom: 700,
            }
        );
        assert_eq!(
            snapshot.disk_spill,
            Some(onnx_genai_scheduler::TierSnapshot {
                used: 1_000,
                limit: 3_000,
                headroom: 2_000,
            })
        );
    }

    #[test]
    fn governor_handle_rejects_disabled_runtime_override() {
        let governor = EngineResourceGovernor::new_with_capacities(
            ResourceLimits::default(),
            false,
            test_capacities(),
            ModelKvConfig::known(100, 16),
            0,
        )
        .unwrap();
        assert!(matches!(
            governor.set_vram_limit(ResourceLimit::Bytes(800)),
            Err(EngineGovernorError::RuntimeOverrideDisabled)
        ));
    }

    #[test]
    fn token_logprobs_use_log_softmax_and_sorted_top_tokens() {
        let logits = [1.0, f32::NEG_INFINITY, 3.0, 2.0];
        let result = logprob_for_token(&logits, 3, 2);
        let logsumexp = 3.0 + ((1.0_f32 - 3.0).exp() + 1.0 + (2.0_f32 - 3.0).exp()).ln();

        assert_eq!(result.token_id, 3);
        assert_eq!(result.logprob, 2.0 - logsumexp);
        assert!(result.logprob <= 0.0);
        assert!(result.top.windows(2).all(|pair| pair[0].1 >= pair[1].1));
        assert!(result.top.iter().any(|(token_id, _)| *token_id == 3));
        assert!(result.top.iter().all(|(token_id, _)| *token_id != 1));
    }

    #[test]
    fn processor_chain_uses_documented_order() {
        let options = GenerateOptions {
            // `GenerateOptions::default()` is greedy, and a greedy request
            // builds no sampling warpers at all. This test is about the order
            // the warpers take when they *are* built, so it must ask to sample.
            greedy: false,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 10,
            min_p: 0.05,
            top_a: 0.4,
            typical_p: 0.8,
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.3,
            dry: Some(crate::config::DryConfig {
                multiplier: 0.5,
                base: 1.75,
                allowed_length: 2,
                sequence_breakers: vec![13],
            }),
            mirostat: Some(crate::config::MirostatConfig {
                tau: 5.0,
                eta: 0.1,
                version: crate::config::MirostatVersion::V2,
            }),
            xtc: Some(crate::config::XtcConfig {
                probability: 0.5,
                threshold: 0.1,
            }),
            stop_sequences: vec![StopSequence::Tokens(vec![42])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false).unwrap();
        assert_eq!(
            chain.names(),
            vec![
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
                "dry",
                "stop_sequence",
                "temperature",
                // top_k and top_p are fused when both are configured: running
                // them separately makes top_p rescan the whole vocabulary that
                // top_k just reduced. The fused processor occupies the same
                // position in the order and produces the same surviving set.
                "top_k_top_p",
                "min_p",
                "top_a",
                "typical_p",
                "mirostat_v2",
                "xtc"
            ]
        );
    }

    #[test]
    fn processor_chain_includes_json_constraint_before_sampling_filters() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json")
            .canonicalize()?;
        let tokenizer = Tokenizer::from_file(&fixture)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        let options = GenerateOptions {
            greedy: false,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 10,
            min_p: 0.05,
            top_a: 0.4,
            typical_p: 0.8,
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.3,
            constraint: Some(GenerateConstraint::Json),
            ..Default::default()
        };

        let chain = build_processor_chain(&options, Some(&tokenizer), false)?;

        assert_eq!(
            chain.names(),
            vec![
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
                "json_constraint",
                "temperature",
                "top_k_top_p",
                "min_p",
                "top_a",
                "typical_p"
            ]
        );
        Ok(())
    }

    #[test]
    fn greedy_selection_uses_argmax_after_processors() {
        let options = GenerateOptions {
            greedy: true,
            top_k: 2,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false).unwrap();
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 2.0, 4.0, 3.0];
        assert_eq!(
            select_next_token(&mut logits, &context, &options, &chain, 0.0),
            2
        );
    }

    #[test]
    fn sampled_selection_can_pick_non_argmax() {
        let options = GenerateOptions {
            greedy: false,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false).unwrap();
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 0.0];
        assert_eq!(
            select_next_token(&mut logits, &context, &options, &chain, 0.75),
            1
        );
    }

    struct LastTokenSampler;

    impl Sampler for LastTokenSampler {
        fn sample(&mut self, logits: &[f32], _context: &ProcessorContext) -> TokenId {
            logits.len().saturating_sub(1) as TokenId
        }

        fn name(&self) -> &str {
            "last_token"
        }
    }

    #[test]
    fn custom_sampler_can_select_after_default_processors() {
        let options = GenerateOptions {
            top_k: 2,
            ..Default::default()
        };
        // A custom sampler is not necessarily greedy, so it keeps the warpers
        // even on these (greedy-by-default) options.
        let chain = build_processor_chain(&options, None, true).unwrap();
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 2.0, 4.0, 3.0];
        let mut sampler = LastTokenSampler;

        assert_eq!(
            select_next_token_with_sampler(&mut logits, &context, &chain, &mut sampler),
            3
        );
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn default_processor_chain_is_empty_for_unchanged_defaults() {
        let options = GenerateOptions::default();
        let chain = build_processor_chain(&options, None, false).unwrap();
        assert!(chain.names().is_empty());
    }

    #[test]
    fn finish_reason_detects_eos_before_stop_sequence() {
        let options = GenerateOptions {
            eos_token_id: Some(7),
            stop_sequences: vec![StopSequence::Tokens(vec![7])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false).unwrap();
        let context = ProcessorContext {
            generated_tokens: vec![7],
            ..Default::default()
        };
        assert_eq!(
            finish_reason_after_token(7, &options, &chain, &context),
            Some(FinishReason::EosToken)
        );
    }

    #[test]
    fn finish_reason_detects_stop_sequence() {
        let options = GenerateOptions {
            stop_sequences: vec![StopSequence::Tokens(vec![2, 3])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false).unwrap();
        let context = ProcessorContext {
            generated_tokens: vec![1, 2, 3],
            ..Default::default()
        };
        assert_eq!(
            finish_reason_after_token(3, &options, &chain, &context),
            Some(FinishReason::StopSequence { index: 0 })
        );
    }

    #[test]
    fn json_constraint_defers_stop_until_value_is_complete() {
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Json),
            stop_sequences: vec![StopSequence::Text("}".to_string())],
            ..Default::default()
        };
        let chain_options = GenerateOptions {
            stop_sequences: options.stop_sequences.clone(),
            ..Default::default()
        };
        let chain = build_processor_chain(&chain_options, None, false).unwrap();
        let incomplete = ProcessorContext {
            generated_text: "{\"value\":".to_string(),
            ..Default::default()
        };
        let complete = ProcessorContext {
            generated_text: "{\"value\":1}".to_string(),
            ..Default::default()
        };

        assert_eq!(
            finish_reason_after_token(1, &options, &chain, &incomplete),
            None
        );
        assert_eq!(
            finish_reason_after_token(1, &options, &chain, &complete),
            Some(FinishReason::StopSequence { index: 0 })
        );
    }

    #[test]
    fn incomplete_json_constraint_rejects_length_finishes() {
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Json),
            ..Default::default()
        };
        for reason in [FinishReason::MaxTokens, FinishReason::Length] {
            let error = ensure_constrained_finish(&options, "{\"value\":", reason).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("stopped before a complete JSON value")
            );
        }
        ensure_constrained_finish(&options, "{\"value\":1}", FinishReason::MaxTokens).unwrap();
        ensure_constrained_finish(&options, "", FinishReason::EosToken).unwrap();
    }

    #[test]
    fn common_prefix_len_stops_before_rejected_draft_token() {
        assert_eq!(common_prefix_len(&[1, 2, 3, 4], &[1, 2, 9]), 2);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(common_prefix_len(&[7], &[8]), 0);
    }

    #[test]
    fn tiny_fixture_generates_requested_tokens_end_to_end() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids.len(), 3);
        assert_eq!(result.finish_reason, FinishReason::MaxTokens);
        assert!(engine.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn tiny_fixture_returns_opt_in_per_token_logprobs() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir_with_session_options(
            &fixture,
            EngineConfig::default(),
            SessionOptions::default().with_intra_op_threads(1),
        )?;
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.top_logprobs = Some(3);

        let result = engine.generate(request)?;
        let logprobs = result.logprobs.as_ref().expect("logprobs requested");

        assert_eq!(logprobs.len(), result.token_ids.len());
        for (token_id, token_logprob) in result.token_ids.iter().zip(logprobs) {
            assert_eq!(*token_id, token_logprob.token_id);
            assert!(token_logprob.logprob <= 0.0);
            assert!(
                token_logprob
                    .top
                    .windows(2)
                    .all(|pair| pair[0].1 >= pair[1].1)
            );
            assert!(
                token_logprob
                    .top
                    .iter()
                    .any(|(top_token_id, _)| top_token_id == token_id)
            );
        }

        let mut disabled = GenerateRequest::new("hello");
        disabled.options.max_new_tokens = 1;
        disabled.options.temperature = 0.0;
        disabled.options.stop_on_eos = false;
        assert!(engine.generate(disabled)?.logprobs.is_none());
        Ok(())
    }

    #[test]
    fn tiny_fixture_uses_past_present_decode_session_with_stable_greedy_output()
    -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        assert!(matches!(
            engine.decode_path,
            ModelDecodePath::PastPresent {
                shared_buffer: false,
                ..
            }
        ));
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids, vec![22, 22, 20]);
        Ok(())
    }

    #[test]
    fn arc_owned_session_preserves_w1_generation_and_continuation_parity() -> anyhow::Result<()> {
        fn request(tokens: Vec<TokenId>) -> GenerateRequest {
            let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
            request.options.max_new_tokens = 2;
            request.options.temperature = 0.0;
            request.options.stop_on_eos = false;
            request
        }

        fn two_turns(engine: &mut Engine) -> anyhow::Result<(Vec<TokenId>, Vec<TokenId>, usize)> {
            let session = engine.create_session()?;
            let first = engine.generate_in_session(session, request(vec![2, 4, 3]))?;
            let second = engine.generate_in_session(session, request(vec![5, 7]))?;
            let count = engine.session_token_count(session)?;
            engine.close_session(session)?;
            Ok((first.token_ids, second.token_ids, count))
        }

        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
        let candidate = Engine::from_dir(&fixture, EngineConfig::default())?;
        let shared_session = std::sync::Arc::clone(
            candidate
                .session
                .as_ref()
                .context("ORT candidate must own a decoder session")?,
        );
        let session_address = std::sync::Arc::as_ptr(&shared_session);
        let concurrent_run_support = shared_session.concurrent_run_support();

        // Moving the holder and cloning only the immutable session resource must
        // not move the pointee or change its provider capability signal.
        let mut candidate = candidate;
        let moved_session = candidate
            .session
            .as_ref()
            .context("moved ORT candidate must retain its decoder session")?;
        assert_eq!(std::sync::Arc::as_ptr(moved_session), session_address);
        assert_eq!(
            moved_session.concurrent_run_support(),
            concurrent_run_support
        );
        assert!(std::sync::Arc::ptr_eq(moved_session, &shared_session));
        drop(shared_session);

        assert_eq!(two_turns(&mut candidate)?, two_turns(&mut baseline)?);
        Ok(())
    }

    #[test]
    fn scatter_fixture_does_not_select_static_cache_from_metadata() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm-scatter")
            .canonicalize()?;
        let engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        assert!(!matches!(
            engine.decode_path,
            ModelDecodePath::StaticCache { .. }
        ));
        Ok(())
    }

    #[test]
    fn tiny_fixture_speculative_matches_plain_greedy_with_k_gt_one() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut speculative = Engine::from_dir(
            &fixture,
            EngineConfig {
                draft_model: Some(fixture.clone()),
                num_speculative_tokens: 3,
                ..Default::default()
            },
        )?;

        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 6;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.num_speculative_tokens = Some(3);

        let baseline_result = baseline.generate(request.clone())?;
        let speculative_result = speculative.generate(request)?;

        assert_eq!(speculative_result.token_ids, baseline_result.token_ids);
        assert_eq!(
            speculative_result.finish_reason,
            baseline_result.finish_reason
        );
        assert_eq!(speculative_result.token_ids.len(), 6);
        Ok(())
    }

    #[test]
    fn tiny_fixture_stops_at_explicit_context_length_without_ort_error() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3]));
        request.options.max_new_tokens = 32;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.max_context = Some(16);

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids.len(), 13);
        assert_eq!(result.finish_reason, FinishReason::Length);
        assert!(engine.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn tiny_fixture_session_stops_at_explicit_context_length_without_ort_error()
    -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let session_id = engine.create_session()?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3]));
        request.options.max_new_tokens = 32;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.max_context = Some(16);

        let result = engine.generate_in_session(session_id, request)?;

        assert_eq!(result.token_ids.len(), 13);
        assert_eq!(result.finish_reason, FinishReason::Length);
        assert_eq!(engine.session_token_count(session_id)?, 16);
        engine.close_session(session_id)?;
        Ok(())
    }

    #[test]
    fn tiny_fixture_session_persists_context_across_turns() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let session_id = engine.create_session()?;

        let mut first = GenerateRequest::new("hello");
        first.options.max_new_tokens = 2;
        first.options.temperature = 0.0;
        first.options.stop_on_eos = false;
        let first_result = engine.generate_in_session(session_id, first)?;
        let first_count = engine.session_token_count(session_id)?;

        let mut second = GenerateRequest::new(" world");
        second.options.max_new_tokens = 2;
        second.options.temperature = 0.0;
        second.options.stop_on_eos = false;
        let second_result = engine.generate_in_session(session_id, second)?;
        let second_count = engine.session_token_count(session_id)?;

        assert_eq!(first_result.token_ids.len(), 2);
        assert_eq!(second_result.token_ids.len(), 2);
        assert!(second_count > first_count);
        assert!(engine.sessions[&session_id].kv_token_count > 0);
        engine.close_session(session_id)?;
        assert!(engine.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn tiny_fixture_session_rewind_to_checkpoint_reports_unsupported_runner_state()
    -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        assert!(matches!(
            engine.decode_path,
            ModelDecodePath::PastPresent { .. }
        ));
        let session_id = engine.create_session()?;
        let checkpoint = engine.checkpoint_session(session_id)?;

        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3]));
        request.options.max_new_tokens = 4;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let first = engine.generate_in_session(session_id, request.clone())?;
        let first_count = engine.session_token_count(session_id)?;
        assert!(first_count > checkpoint.position.get());

        let error = engine
            .restore_session(checkpoint)
            .expect_err("PastPresent runner-backed session rewind is not supported");
        assert!(
            error.to_string().contains("runner-backed decoder state"),
            "unexpected restore error: {error:#}"
        );
        assert_eq!(
            engine.session_token_count(session_id)?,
            first_count,
            "failed public rewind must leave session unchanged"
        );
        assert_eq!(first.token_ids.len(), 4);
        engine.close_session(session_id)?;
        Ok(())
    }

    #[test]
    fn tiny_fixture_session_fork_fails_closed_until_runner_state_supported() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let session_id = engine.create_session()?;
        assert!(engine.session_fork_capability().is_none());

        let error = engine
            .fork_session(
                &SessionForkCapability { _private: () },
                session_id,
                SessionPosition::new(0),
            )
            .expect_err("fork must fail closed until decode state can be cloned safely");

        assert!(
            error
                .to_string()
                .contains("session fork is not yet enabled"),
            "unexpected fork error: {error}"
        );
        engine.close_session(session_id)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires ONNX_GENAI_FIM_MODEL_DIR to point at a FIM-capable coder model"]
    fn fim_generation_runs_with_fim_capable_model() -> anyhow::Result<()> {
        let Some(model_dir) = onnx_genai_runtime_config::runtime_config()
            .fim_model_dir
            .as_deref()
        else {
            eprintln!("set ONNX_GENAI_FIM_MODEL_DIR to a Qwen2.5-Coder/StarCoder-style model");
            return Ok(());
        };
        let mut engine = Engine::from_dir(model_dir, EngineConfig::default())?;
        assert!(
            engine.fim_config().is_some(),
            "model tokenizer_config.json must expose recognized FIM tokens"
        );

        let mut options = GenerateOptions {
            max_new_tokens: 16,
            temperature: 0.0,
            ..Default::default()
        };
        options
            .stop_sequences
            .push(StopSequence::Text("\n\n".into()));

        let result =
            engine.generate_fim("fn add(a: i32, b: i32) -> i32 {\n    ", "\n}", options)?;

        assert!(!result.token_ids.is_empty());
        Ok(())
    }

    fn local_tiered_engine_config(chunk_size: usize) -> EngineConfig {
        EngineConfig {
            kv_connector: KvConnectorConfig {
                backend: KvConnectorBackend::LocalTiered(onnx_genai_kv::LocalTieredConfig {
                    chunk_size,
                    page_size: chunk_size,
                    ..onnx_genai_kv::LocalTieredConfig::default()
                }),
                chunk_size,
                ..KvConnectorConfig::default()
            },
            ..EngineConfig::default()
        }
    }

    #[test]
    fn null_connector_default_leaves_behavior_unchanged() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
        assert!(!baseline.connector.is_active());

        let mut request =
            GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3, 5, 6, 7, 8, 9]));
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = baseline.generate(request)?;
        // With the default Null connector, no external activity happens at all.
        assert_eq!(baseline.last_connector_stats(), ConnectorStats::default());
        assert_eq!(result.token_ids.len(), 3);
        Ok(())
    }

    #[test]
    fn local_tiered_connector_stores_prefill_chunks() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, local_tiered_engine_config(2))?;
        assert!(engine.connector.is_active());

        let mut request =
            GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3, 5, 6, 7, 8, 9]));
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let baseline_ids = {
            let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
            baseline.generate(request.clone())?.token_ids
        };

        let result = engine.generate(request)?;

        // Store-after-prefill ran: complete chunks were pushed to the connector.
        assert!(
            engine.last_connector_stats().stores > 0,
            "expected connector store path to push chunks, got {:?}",
            engine.last_connector_stats()
        );
        // The store path is a pure side effect for a first, unseen request:
        // nothing is resident to fetch, so output matches full recompute.
        assert_eq!(result.token_ids, baseline_ids);
        Ok(())
    }

    #[test]
    fn local_tiered_connector_fetch_reuse_is_token_identical() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, local_tiered_engine_config(2))?;

        // Request 1 populates the connector with the prompt's KV chunks.
        let prompt = vec![10, 11, 12, 13, 14, 15];
        let mut warm = GenerateRequest::new(GeneratePrompt::TokenIds(prompt.clone()));
        warm.options.max_new_tokens = 1;
        warm.options.temperature = 0.0;
        warm.options.stop_on_eos = false;
        engine.generate(warm)?;
        assert!(engine.last_connector_stats().stores > 0);

        // Drop the in-process caches so the connector is the ONLY source of
        // cross-session reuse — simulating a fresh process / different node that
        // shares nothing but the connector.
        engine.token_prefix_cache.clear();
        engine.prefix_cache = PrefixCache::new();

        // Request 2 shares the whole prefix (≥ 1 chunk) with request 1.
        let mut reuse = GenerateRequest::new(GeneratePrompt::TokenIds(prompt.clone()));
        reuse.options.max_new_tokens = 4;
        reuse.options.temperature = 0.0;
        reuse.options.stop_on_eos = false;
        let reuse_result = engine.generate(reuse)?;
        let stats = engine.last_connector_stats();

        // (a) Prefill was genuinely shortened: real KV bytes were fetched and
        // injected into the runner.
        assert!(
            stats.fetched_tokens > 0 && stats.chunk_hits > 0,
            "expected connector fetch to materialize KV, got {stats:?}"
        );
        // At least one prompt token is always left to feed the decoder.
        assert!(stats.fetched_tokens < prompt.len());

        // (b) Output is byte-for-byte identical to full recompute with a Null
        // connector — proving the materialized KV is correct, not just present.
        let baseline_ids = {
            let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
            let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt.clone()));
            request.options.max_new_tokens = 4;
            request.options.temperature = 0.0;
            request.options.stop_on_eos = false;
            baseline.generate(request)?.token_ids
        };
        assert_eq!(
            reuse_result.token_ids, baseline_ids,
            "connector-reuse output must match full recompute exactly"
        );
        Ok(())
    }
}
