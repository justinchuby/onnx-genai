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
    /// Native shared-KV proposer loaded from the same metadata contract.
    #[cfg(feature = "native-backend")]
    pub(crate) native_shared_kv_proposer: Option<NativeSharedKvProposerModel>,
    /// Native-LoRA manager (design §D, **P4**). Owns the decoded PEFT adapter
    /// for a single-fixed-adapter session and records the active selection.
    /// `None` when no adapter was configured. Present so engine-level
    /// activate/deactivate can toggle the injected override buffers.
    #[cfg(feature = "native-backend")]
    pub(crate) lora_manager: Option<crate::lora::manager::LoraManager>,
    /// Selectable name of the single LoRA adapter that collapsed to the DIRECT
    /// fast path (design §D/§J). When exactly one `--adapters NAME=PATH` is
    /// configured it loads on the always-on single-adapter path but keeps its
    /// user-facing NAME here, so a `--select-adapter NAME` request for that same
    /// adapter resolves to a no-op (already applied) instead of failing as if no
    /// adapter were loaded. `None` for base-only or grouped multi-adapter
    /// sessions.
    #[cfg(feature = "native-backend")]
    pub(crate) lora_single_adapter_name: Option<String>,
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
    pub(crate) _environment: Environment,
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
