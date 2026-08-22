//! Core `Engine` struct and speculative model holder structs.

use super::*;

/// Error message used when the ORT decode backend is unexpectedly missing its
/// decoder session. Shared by [`Engine::ort_session`] and the few call sites
/// that must borrow the session field disjointly from other mutable fields.
pub(crate) const MISSING_ORT_SESSION: &str = "ORT backend must own a decoder session";

/// The generation engine.
pub struct Engine {
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
    pub(crate) governor: EngineResourceGovernor,
    /// Persistent multi-turn session state, keyed by session id.
    pub(crate) sessions: HashMap<SessionId, EngineSession>,
    /// ORT session for decoder execution.
    pub(crate) session: Option<Box<Session>>,
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
    /// Monotonic counter for native session id generation.
    #[cfg(feature = "native-backend")]
    pub(crate) native_session_counter: u64,
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
    pub(crate) native_shared_kv_proposer: Option<NativeSharedKvProposerModel>,
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
    pub(crate) shared_kv_proposer: Option<SharedKvProposerModel>,
    /// Tokenizer loaded from the model directory.
    pub(crate) tokenizer: Tokenizer,
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
    /// ORT environment — MUST be the LAST field so it (and the plugin EP factory it owns via
    /// RegisterExecutionProviderLibrary) drops AFTER every Session/draft/mtp/eagle3 field above.
    /// Rust drops struct fields in declaration order; if the env dropped first, ORT would tear down
    /// the plugin EP factory before the sessions, causing a teardown use-after-free (segfault) in
    /// the Metal/MLX plugin EP's allocator/data-transfer/context release path.
    ///
    /// Tests that exercise pre-session validation may set this to `None` so they
    /// stay model-free and do not touch the local ORT library.
    pub(crate) _environment: Option<Environment>,
}

#[cfg(feature = "native-backend")]
pub(crate) struct NativePrefixSnapshot {
    pub(crate) snapshot: crate::native_decode::NativePastSnapshot,
    pub(crate) _lease: onnx_runtime_memory_governor::MemoryLease,
}

// SAFETY: `Engine` owns every ORT or native-runtime handle reachable through
// its sessions and decode state. Neither runtime's sessions, values, bindings,
// allocators, or CPU tensors have thread affinity. Moving the engine transfers
// exclusive ownership; mutation still requires `&mut Engine`. Self-references
// in ORT decode runners point into boxed `Session` allocations, whose addresses
// remain stable when the owning `Engine` moves. This would stop being sound if
// an execution provider introduced thread-affine handles or a field gained
// unsynchronized shared mutation.
unsafe impl Send for Engine {}

/// Per-conversation state for native session-persistent KV reuse.
/// The authoritative token history lives here; the `NativeDecodeSession`'s
/// `current_len` represents the *KV-materialized* position.
#[cfg(feature = "native-backend")]
pub(crate) struct NativeSessionState {
    /// Full token history of this session (prompt + generated tokens from all turns).
    pub(crate) tokens: Vec<TokenId>,
    /// Monotonically increasing access stamp, used to pick an LRU victim.
    pub(crate) last_access: u64,
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
    pub(crate) session: Box<Session>,
    pub(crate) embedder: LinearEmbedder,
    pub(crate) token_map: Option<Vec<TokenId>>,
    pub(crate) hidden_outputs: Vec<String>,
    pub(crate) kv_mode: onnx_genai_ort::Eagle3DraftKvMode,
    pub(crate) num_speculative_tokens: usize,
}

pub(crate) struct SharedKvProposerModel {
    pub(crate) config: SharedKvProposerConfig,
    pub(crate) session: Box<Session>,
    /// Target input-token embedding table, used to build the token-embedding
    /// half of each draft step's `inputs_embeds`.
    pub(crate) embedder: LinearEmbedder,
    pub(crate) num_speculative_tokens: usize,
}

#[cfg(feature = "native-backend")]
pub(crate) struct NativeSharedKvProposerModel {
    pub(crate) session: crate::native_decode::NativeProposerSession,
    pub(crate) embedder: LinearEmbedder,
    pub(crate) groups: Vec<onnx_genai_metadata::SharedKvGroup>,
    pub(crate) hidden_size: usize,
}
