use std::sync::Arc;

/// Prototype ONNX opaque-state kinds supported by nxrt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpaqueStateKind {
    KVCache,
    LinearAttentionState,
    CausalConvState,
}

impl OpaqueStateKind {
    pub(crate) fn onnx_name(self) -> &'static str {
        match self {
            Self::KVCache => "KVCache",
            Self::LinearAttentionState => "LinearAttentionState",
            Self::CausalConvState => "CausalConvState",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionCacheMode {
    Dense,
    Paged { page_size: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearAttentionStateMode {
    Contiguous,
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalConvStateMode {
    Contiguous,
    Ring,
}

/// Runtime-only physical policy. It never changes the serialized ONNX graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateRuntimeConfig {
    pub attention: AttentionCacheMode,
    pub linear_attention: LinearAttentionStateMode,
    pub causal_conv: CausalConvStateMode,
}

/// Work counters from the most recent inference call containing
/// `AttentionWithCache`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttentionTraversalStats {
    /// Key/query/head pairs in the logical post-append cache before visibility.
    pub logical_key_candidates: u64,
    /// Dot products actually evaluated after compact visibility and position
    /// constraints.
    pub evaluated_key_candidates: u64,
    /// Full visibility-block visits, including broadcast across query heads.
    pub full_block_visits: u64,
    /// Partial visibility-block visits, including broadcast across query heads.
    pub partial_block_visits: u64,
}

impl Default for StateRuntimeConfig {
    fn default() -> Self {
        Self {
            attention: AttentionCacheMode::Dense,
            linear_attention: LinearAttentionStateMode::Contiguous,
            causal_conv: CausalConvStateMode::Contiguous,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum StatePayload {
    Kv(KvState),
    Linear(LinearState),
    Conv(ConvState),
}

#[derive(Clone, Debug)]
pub(crate) struct KvRecord {
    pub position: i64,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(crate) enum KvRows {
    Dense(Arc<Vec<KvRecord>>),
    Paged {
        page_size: usize,
        pages: Arc<Vec<Arc<Vec<KvRecord>>>>,
        len: usize,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct KvState {
    pub batch: usize,
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub rows: Arc<Vec<KvRows>>,
}

#[derive(Clone, Debug)]
pub(crate) enum LinearStorage {
    Contiguous(Arc<Vec<f32>>),
    Blocked(Arc<Vec<Arc<Vec<f32>>>>),
}

#[derive(Clone, Debug)]
pub(crate) struct LinearState {
    pub batch: usize,
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub storage: LinearStorage,
}

#[derive(Clone, Debug)]
pub(crate) enum ConvStorage {
    Contiguous(Arc<Vec<f32>>),
    Ring { data: Arc<Vec<f32>>, start: usize },
}

#[derive(Clone, Debug)]
pub(crate) struct ConvState {
    pub batch: usize,
    pub channels: usize,
    pub history: usize,
    pub storage: ConvStorage,
}

/// An immutable, Arc-shared ONNX opaque value returned by one inference call
/// and accepted by a later call to the same state slot.
#[derive(Clone, Debug)]
pub struct OpaqueValue {
    pub(crate) owner: u64,
    pub(crate) node: u32,
    pub(crate) kind: OpaqueStateKind,
    pub(crate) payload: Arc<StatePayload>,
}

impl OpaqueValue {
    pub fn kind(&self) -> OpaqueStateKind {
        self.kind
    }
}

/// Named input accepted by [`crate::InferenceSession::run_values`].
#[derive(Clone, Copy)]
pub enum SessionInput<'a> {
    Tensor(&'a crate::Tensor),
    Opaque(&'a OpaqueValue),
}
