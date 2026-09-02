//! Core `Engine` struct and speculative model holder structs.

use super::*;

/// Error message used when the ORT decode backend is unexpectedly missing its
/// decoder session. Shared by [`Engine::ort_session`] and the few call sites
/// that must borrow the session field disjointly from other mutable fields.
pub(crate) const MISSING_ORT_SESSION: &str = "ORT backend must own a decoder session";

/// The one generation runtime.
///
/// Every package declares `pipeline.workflow`, and every package executes it
/// through the interpreter held in [`Self::workflow`]. What differs between a
/// bare decoder and a composite pipeline is not *which runtime* executes it but
/// *which executor implements a declared step*: a component naming a contract
/// this runtime registered — `onnx-genai.autoregressive-decode`, above all — is
/// run by the fused decode session in the fields below, which owns paged KV,
/// the device sampling fast paths and the captured CUDA graph. A component
/// naming none is invoked generically from its artifact.
///
/// Callers (server, CLI, C ABI, benchmarks) hold one handle and never branch on
/// which kind of package they loaded, which is what the old
/// `Engine` / `PipelineEngine` split forced them to do.
pub struct Engine {
    /// Runtime-only authority that binds prepared session forks to this engine.
    pub(crate) session_fork_origin: SessionForkOrigin,
    /// The interpreter that executes this package's declared workflow.
    ///
    /// Not optional: a package that declares no workflow does not load, so
    /// there is no state in which a request could reach a path that no longer
    /// exists.
    pub(crate) workflow: Box<crate::pipeline::WorkflowRuntime>,
    /// Resolved decoder execution backend.
    pub(crate) decode_backend: EngineDecodeBackend,
    /// Model inference metadata.
    pub(crate) metadata: InferenceMetadata,
    /// Validated execution hints embedded in the ONNX graph.
    pub(crate) metadata_hints: MetadataHints,
    /// KV cache manager.
    pub(crate) kv_cache: PagedKvCache,
    /// Shared-prefix cache for reusing paged KV across sessions.
    pub(crate) prefix_cache: PrefixCache,
    /// Token-only prefix index used by ORT-owned decode sessions until page import/export lands.
    pub(crate) token_prefix_cache: Vec<Vec<TokenId>>,
    /// KV tensor layout inferred from model present/past TensorInfo.
    pub(crate) kv_model: Option<KvModelInfo>,
    /// ORT decode path selected by model I/O introspection.
    pub(crate) decode_path: ModelDecodePath,
    /// Batch scheduler.
    pub(crate) scheduler: Scheduler,
    /// Per-device resource ceilings and shared scheduler byte budget.
    ///
    /// Exactly one exists per runtime — a second would double-count every
    /// reservation — which is why the workflow interpreter beside it holds
    /// none and this is not an `Option`.
    pub(crate) governor: Arc<EngineResourceGovernor>,
    /// Persistent multi-turn session state, keyed by session id.
    pub(crate) sessions: HashMap<SessionId, EngineSession>,
    /// Conversations continued through the interpreter's session-scoped state.
    ///
    /// A package whose components the interpreter invokes has no decode core to
    /// hold a paged KV sequence, but it still has sessions: its workflow may
    /// declare `scope: session` state, and the runtime keys that state by the
    /// id handed out here. The value is the logical token count so far, which
    /// is what a caller asking about a session wants to know.
    pub(crate) workflow_sessions: HashMap<SessionId, usize>,
    /// Mints worker-local interpreter session ids. The server qualifies these
    /// with `WorkerId`, so different workers may intentionally mint the same
    /// local id without aliasing conversations.
    pub(crate) workflow_session_ids: SharedSessionIds,
    /// Immutable ORT decoder model resource.
    ///
    /// Workers may clone this `Arc` without cloning mutable KV, bindings,
    /// allocators, device values, or graph-capture state.
    pub(crate) session: Option<Arc<Session>>,
    /// Native decoder session. Native execution is single-request and serialized
    /// by the server's fallback driver in this first milestone.
    #[cfg(feature = "native-backend")]
    pub(crate) native_session: Option<crate::native_decode::NativeDecodeSession>,
    /// Load-time static device/host placement plan for pageable weight layers.
    #[cfg(feature = "native-backend")]
    pub(crate) weight_placement: Option<WeightPlacementReport>,
    /// Memory strategy inferred at load before applying any strategy-specific policy.
    pub(crate) memory_strategy_plan: MemoryStrategyPlan,
    /// Multi-session native state: per-session token history keyed by session id.
    /// The active session (whose KV is loaded in `native_session`) is tracked by
    /// `native_active_session`. When switching, the engine re-prefills from the
    /// target session's token history.
    #[cfg(feature = "native-backend")]
    pub(crate) native_sessions: HashMap<SessionId, NativeSessionState>,
    /// Which native session currently has its KV state loaded in `native_session`.
    #[cfg(feature = "native-backend")]
    pub(crate) native_active_session: Option<SessionId>,
    /// Mints native session ids, in their own shared namespace.
    #[cfg(feature = "native-backend")]
    pub(crate) native_session_ids: SharedSessionIds,
    /// Monotonic counter for LRU access stamps. Kept separate from the id
    /// counter so that touching a session does not consume session ids.
    #[cfg(feature = "native-backend")]
    pub(crate) native_access_counter: u64,
    /// Implicit "default" native session used by the stateless `generate()` path
    /// for transparent KV reuse.
    #[cfg(feature = "native-backend")]
    pub(crate) native_default_session: Option<SessionId>,
    /// Maximum retained native session token histories before LRU eviction.
    #[cfg(feature = "native-backend")]
    pub(crate) native_max_sessions: usize,
    /// Native shared-KV proposer loaded from the same metadata contract.
    #[cfg(feature = "native-backend")]
    /// Native recurrent/past snapshots keyed by semantic token prefixes.
    #[cfg(feature = "native-backend")]
    pub(crate) native_recurrent_prefix_stats: RecurrentPrefixCacheStats,
    /// Optional draft model used by the speculative decoding path.
    pub(crate) draft: Option<DraftModel>,
    /// Optional MTP head and target-side projections.
    pub(crate) mtp: Option<MtpModel>,
    /// Optional EAGLE-3 head and target-side embedding.
    pub(crate) eagle3: Option<Eagle3Model>,
    /// Optional shared-KV draft proposer.
    /// Tokenizer loaded from the model directory.
    ///
    /// Absent for a workflow package that ships none (an image-generation
    /// pipeline, for instance): such a package never reaches the token decode
    /// path, and inventing an empty tokenizer would make that a runtime
    /// surprise instead of a load-time fact.
    pub(crate) tokenizer: Option<Arc<Tokenizer>>,
    /// Auto-detected fill-in-the-middle token configuration.
    pub(crate) fim_config: Option<FimConfig>,
    /// Default speculative draft width K.
    pub(crate) num_speculative_tokens: usize,
    /// Default speculative candidate source.
    pub(crate) speculative_mode: SpeculativeMode,
    /// Diagnostics from the most recent generation call.
    pub(crate) last_speculative_stats: SpeculativeStats,
    /// Optional distributed KV connector bridge (DESIGN §38, K3). Inert when
    /// configured as `Null` (the default), preserving in-process-only behavior.
    pub(crate) connector: ConnectorBridge,
    /// Keeps the one fixed-weight memory reservation alive until every engine
    /// sharing the immutable ORT session has dropped.
    pub(crate) _shared_memory_plan:
        Option<Arc<std::sync::Mutex<crate::engine::memory_plan::ModelMemoryPlan>>>,
    /// ORT environment — MUST be the LAST field so it (and the plugin EP factory it owns via
    /// RegisterExecutionProviderLibrary) drops AFTER every Session/draft/mtp/eagle3 field above.
    /// Rust drops struct fields in declaration order; if the env dropped first, ORT would tear down
    /// the plugin EP factory before the sessions, causing a teardown use-after-free (segfault) in
    /// the Metal/MLX plugin EP's allocator/data-transfer/context release path.
    ///
    /// Tests that exercise pre-session validation may set this to `None` so they
    /// stay model-free and do not touch the local ORT library.
    pub(crate) _environment: Option<Arc<Environment>>,
}

impl Engine {
    /// Build the one runtime around an already-constructed workflow interpreter.
    ///
    /// The decode-core fields below it are inert for a workflow package: it owns
    /// no paged KV, no decoder session, and no scheduler of its own — its
    /// components own their caches and the interpreter drives them. They are
    /// real (not `Option`) values so the decode core needs no null checks on a
    /// path it never runs; the governor, memory plan, and backend are read back
    /// from the workflow so a caller sees one authoritative answer.
    pub(crate) fn from_workflow(
        workflow: crate::pipeline::WorkflowRuntime,
        governor: EngineResourceGovernor,
    ) -> anyhow::Result<Self> {
        let decode_backend = workflow.decode_backend();
        let memory_strategy_plan = workflow.memory_strategy_plan().clone();
        let metadata = workflow
            .models()
            .directory
            .metadata
            .clone()
            .unwrap_or_default();
        // The package's tokenizer stays owned by `PipelineModels`; text
        // tokenization for a workflow package is served from there rather than
        // duplicated into the decode core.
        Ok(Engine {
            session_fork_origin: SessionForkOrigin::new(),
            workflow: Box::new(workflow),
            decode_backend,
            metadata,
            metadata_hints: MetadataHints::default(),
            kv_cache: PagedKvCache::new(1, 1),
            prefix_cache: PrefixCache::new(),
            token_prefix_cache: Vec::new(),
            kv_model: None,
            decode_path: ModelDecodePath::Generic,
            scheduler: Scheduler::new(onnx_genai_scheduler::SchedulerConfig::default()),
            governor: Arc::new(governor),
            sessions: HashMap::new(),
            workflow_sessions: HashMap::new(),
            workflow_session_ids: SharedSessionIds::new(),
            session: None,
            #[cfg(feature = "native-backend")]
            native_session: None,
            #[cfg(feature = "native-backend")]
            weight_placement: None,
            memory_strategy_plan,
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
            native_max_sessions: 0,
            #[cfg(feature = "native-backend")]
            native_recurrent_prefix_stats: RecurrentPrefixCacheStats::default(),
            draft: None,
            mtp: None,
            eagle3: None,
            tokenizer: None,
            fim_config: None,
            num_speculative_tokens: 1,
            speculative_mode: SpeculativeMode::None,
            last_speculative_stats: SpeculativeStats::default(),
            connector: ConnectorBridge::null(),
            _shared_memory_plan: None,
            _environment: None,
        })
    }
}

#[cfg(feature = "native-backend")]
pub(crate) struct NativePrefixSnapshot {
    pub(crate) snapshot: crate::native_decode::NativePastSnapshot,
    pub(crate) _lease: onnx_runtime_memory_governor::MemoryLease,
}

/// Per-conversation state for native session-persistent KV reuse.
/// The authoritative token history lives here; the `NativeDecodeSession`'s
/// `current_len` represents the *KV-materialized* position.
#[cfg(feature = "native-backend")]
pub(crate) struct NativeSessionState {
    /// Full token history of this session (prompt + generated tokens from all turns).
    pub(crate) tokens: Vec<TokenId>,
    /// Monotonically increasing access stamp, used to pick an LRU victim.
    pub(crate) last_access: u64,
    /// Reset advances this generation while retaining the public session id.
    pub(crate) generation: u64,
    /// A failed atomic rollback poisons only this exact logical generation.
    pub(crate) poison: Option<String>,
}

pub(crate) struct MtpModel {
    pub(crate) config: MtpConfig,
    pub(crate) runtime_config: ResolvedMtpConfig,
    pub(crate) session: Arc<Session>,
    pub(crate) embedder: MtpEmbedder,
    pub(crate) lm_head: MtpLmHead,
    pub(crate) hidden_output: String,
    pub(crate) kv_mode: onnx_genai_ort::MtpDraftKvMode,
    pub(crate) num_speculative_tokens: usize,
}

pub(crate) struct Eagle3Model {
    pub(crate) config: Eagle3Config,
    pub(crate) session: Arc<Session>,
    pub(crate) embedder: LinearEmbedder,
    pub(crate) token_map: Option<Vec<TokenId>>,
    pub(crate) hidden_outputs: Vec<String>,
    pub(crate) kv_mode: onnx_genai_ort::Eagle3DraftKvMode,
    pub(crate) num_speculative_tokens: usize,
}

#[cfg(test)]
mod ownership_tests {
    use super::*;

    fn assert_arc_session(_: &Arc<Session>) {}

    fn assert_engine_session_holders(
        engine: &Engine,
        draft: &DraftModel,
        mtp: &MtpModel,
        eagle3: &Eagle3Model,
    ) {
        if let Some(session) = &engine.session {
            assert_arc_session(session);
        }
        assert_arc_session(&draft.session);
        assert_arc_session(&mtp.session);
        assert_arc_session(&eagle3.session);
    }

    #[test]
    fn primary_and_speculative_ort_session_holders_are_arc_owned() {
        let _ = assert_engine_session_holders as fn(&Engine, &DraftModel, &MtpModel, &Eagle3Model);
    }
}
