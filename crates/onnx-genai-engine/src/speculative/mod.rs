//! Speculative decoding engine.

//! Greedy requests can propose candidates with a draft model, an MTP head, or
//! model-free prompt lookup. All sources feed the same target verification,
//! longest-prefix acceptance, correction-token, and KV rewind path.

use crate::TokenId;
use crate::config::{MtpCacheScope, MtpHiddenLayout};
use crate::decode::{
    DraftProposalRequest, apply_paged_sliding_window, extract_logits_sequence_with_io,
    next_session_token_logits, next_session_token_logits_and_hidden,
    next_session_token_logits_and_hiddens, propose_draft_tokens, run_decode_session_logits,
    run_decode_step,
};
use crate::decode_loop::{
    DecodeLoopState, commit_selected_token, logprob_for_token, reached_context_limit,
};
use crate::engine::{Engine, MISSING_ORT_SESSION};
use crate::kv_bridge::{
    RewindRequest, RewindRunnerPolicy, common_prefix_len, mirror_present_kv_to_pages,
    rewind_draft_state_to_len, rewind_target_state_to_len, trim_overmaterialized_target_kv,
};
use crate::logits::{ProcessorChain, ProcessorContext};
use crate::processors::{ensure_constrained_finish, select_next_token_with_rng};
use crate::sampling::SamplingRng;
use crate::session::{DraftModel, DraftSession, EngineSession};
use crate::{
    FinishReason, GenerateOptions, GenerateResult, GenerateTokenCallback, SessionId,
    SpeculativeMode,
};
use anyhow::Context;
use onnx_genai_kv::KvCacheOps;
use onnx_genai_ort::{
    Eagle3DecodeOptions, Eagle3DecodeSession, MtpDecodeOptions, MtpDecodeSession, Session,
    SharedKvInput,
};
use onnx_runtime_ir::{DataType as IrDataType, WeightRef};
use onnx_runtime_loader::WeightStore;
use std::path::Path;
use std::sync::Arc;

pub mod tree;
pub use tree::{
    KvRetentionPlan, SpecTree, SpecTreeBuilder, Topology, TreeNode, TreeScorer,
    ancestor_attention_mask, relative_position_ids, verify_tree,
};

/// Produces a target-model token embedding for an MTP proposal step.
pub trait TokenEmbedder {
    fn hidden_size(&self) -> usize;
    /// Number of token ids this embedder can embed. A token id at or beyond it
    /// has no embedding row, so it can be neither embedded nor verified.
    fn vocab_size(&self) -> usize;
    fn embed(&self, token: TokenId, out: &mut [f32]) -> anyhow::Result<()>;
}

/// Projects a target-model hidden state to vocabulary logits.
pub trait LmHead {
    fn vocab_size(&self) -> usize;
    fn logits(&self, hidden: &[f32], out: &mut [f32]) -> anyhow::Result<()>;
}

/// Dense target embedding table in row-major `[vocab, hidden]` order.
#[derive(Debug, Clone)]
pub struct LinearEmbedder {
    weight: Vec<f32>,
    vocab: usize,
    hidden: usize,
}

impl LinearEmbedder {
    pub fn new(weight: Vec<f32>, vocab: usize, hidden: usize) -> anyhow::Result<Self> {
        if weight.len() != vocab * hidden {
            anyhow::bail!(
                "embedder weight length {} != vocab {vocab} * hidden {hidden}",
                weight.len()
            );
        }
        Ok(Self {
            weight,
            vocab,
            hidden,
        })
    }
}

impl TokenEmbedder for LinearEmbedder {
    fn hidden_size(&self) -> usize {
        self.hidden
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn embed(&self, token: TokenId, out: &mut [f32]) -> anyhow::Result<()> {
        let token = token as usize;
        if token >= self.vocab {
            anyhow::bail!("token {token} out of range for vocab {}", self.vocab);
        }
        if out.len() != self.hidden {
            anyhow::bail!(
                "embed output length {} != hidden {}",
                out.len(),
                self.hidden
            );
        }
        let start = token * self.hidden;
        out.copy_from_slice(&self.weight[start..start + self.hidden]);
        Ok(())
    }
}

/// Dense target LM-head projection in row-major `[hidden, vocab]` order.
#[derive(Debug, Clone)]
pub struct LinearLmHead {
    weight: Vec<f32>,
    hidden: usize,
    vocab: usize,
}

#[derive(Debug, Clone)]
struct TargetInitializerMatrix {
    store: Arc<WeightStore>,
    weight: WeightRef,
    rows: usize,
    cols: usize,
}

impl TargetInitializerMatrix {
    fn new(
        store: Arc<WeightStore>,
        weight: WeightRef,
        rows: usize,
        cols: usize,
    ) -> anyhow::Result<Self> {
        let dtype = weight.dtype();
        if !matches!(
            dtype,
            IrDataType::Float32 | IrDataType::Float16 | IrDataType::BFloat16
        ) {
            anyhow::bail!(
                "MTP target initializer dtype {dtype:?} is not supported; Phase 1 supports Float32, Float16, and BFloat16 shared weights"
            );
        }
        let bytes = store
            .bytes(&weight)
            .context("target initializer bytes are not available")?;
        let expected = dtype.storage_bytes(
            rows.checked_mul(cols)
                .context("target initializer element count overflow")?,
        );
        if bytes.len() != expected {
            anyhow::bail!(
                "target initializer byte length {} != expected {expected} for [{rows}, {cols}] {dtype:?}",
                bytes.len()
            );
        }
        Ok(Self {
            store,
            weight,
            rows,
            cols,
        })
    }

    fn value(&self, row: usize, col: usize) -> anyhow::Result<f32> {
        let index = row
            .checked_mul(self.cols)
            .and_then(|start| start.checked_add(col))
            .context("target initializer index overflow")?;
        let bytes = self
            .store
            .bytes(&self.weight)
            .context("target initializer bytes are not available")?;
        Ok(match self.weight.dtype() {
            IrDataType::Float32 => {
                let start = index * 4;
                f32::from_le_bytes(bytes[start..start + 4].try_into().expect("four bytes"))
            }
            IrDataType::Float16 => {
                let start = index * 2;
                half::f16::from_bits(u16::from_le_bytes(
                    bytes[start..start + 2].try_into().expect("two bytes"),
                ))
                .to_f32()
            }
            IrDataType::BFloat16 => {
                let start = index * 2;
                half::bf16::from_bits(u16::from_le_bytes(
                    bytes[start..start + 2].try_into().expect("two bytes"),
                ))
                .to_f32()
            }
            dtype => anyhow::bail!("unsupported target initializer dtype {dtype:?}"),
        })
    }
}

/// Target embedding adapter backed directly by an ONNX initializer.
#[derive(Debug, Clone)]
pub(crate) struct TargetInitializerEmbedder {
    matrix: TargetInitializerMatrix,
}

impl TokenEmbedder for TargetInitializerEmbedder {
    fn hidden_size(&self) -> usize {
        self.matrix.cols
    }

    fn vocab_size(&self) -> usize {
        self.matrix.rows
    }

    fn embed(&self, token: TokenId, out: &mut [f32]) -> anyhow::Result<()> {
        let token = token as usize;
        if token >= self.matrix.rows {
            anyhow::bail!(
                "token {token} out of range for target initializer vocabulary {}",
                self.matrix.rows
            );
        }
        if out.len() != self.matrix.cols {
            anyhow::bail!(
                "embed output length {} != hidden {}",
                out.len(),
                self.matrix.cols
            );
        }
        for (column, value) in out.iter_mut().enumerate() {
            *value = self.matrix.value(token, column)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum LmHeadInitializerLayout {
    HiddenVocab,
    VocabHidden,
}

/// Target LM-head adapter backed directly by an ONNX initializer.
#[derive(Debug, Clone)]
pub(crate) struct TargetInitializerLmHead {
    matrix: TargetInitializerMatrix,
    layout: LmHeadInitializerLayout,
    hidden: usize,
    vocab: usize,
}

impl LmHead for TargetInitializerLmHead {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn logits(&self, hidden: &[f32], out: &mut [f32]) -> anyhow::Result<()> {
        if hidden.len() != self.hidden {
            anyhow::bail!(
                "lm-head input length {} != hidden {}",
                hidden.len(),
                self.hidden
            );
        }
        if out.len() != self.vocab {
            anyhow::bail!(
                "lm-head output length {} != vocab {}",
                out.len(),
                self.vocab
            );
        }
        for (vocab_index, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (hidden_index, &value) in hidden.iter().enumerate() {
                let weight = match self.layout {
                    LmHeadInitializerLayout::HiddenVocab => {
                        self.matrix.value(hidden_index, vocab_index)?
                    }
                    LmHeadInitializerLayout::VocabHidden => {
                        self.matrix.value(vocab_index, hidden_index)?
                    }
                };
                acc += value * weight;
            }
            *slot = acc;
        }
        Ok(())
    }
}

/// Embedding implementation selected by legacy file or package metadata.
#[derive(Debug, Clone)]
pub(crate) enum MtpEmbedder {
    Linear(LinearEmbedder),
    TargetInitializer(TargetInitializerEmbedder),
}

impl TokenEmbedder for MtpEmbedder {
    fn hidden_size(&self) -> usize {
        match self {
            Self::Linear(embedder) => embedder.hidden_size(),
            Self::TargetInitializer(embedder) => embedder.hidden_size(),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Linear(embedder) => embedder.vocab_size(),
            Self::TargetInitializer(embedder) => embedder.vocab_size(),
        }
    }

    fn embed(&self, token: TokenId, out: &mut [f32]) -> anyhow::Result<()> {
        match self {
            Self::Linear(embedder) => embedder.embed(token, out),
            Self::TargetInitializer(embedder) => embedder.embed(token, out),
        }
    }
}

/// LM-head implementation selected by legacy file or package metadata.
#[derive(Debug, Clone)]
pub(crate) enum MtpLmHead {
    Linear(LinearLmHead),
    TargetInitializer(TargetInitializerLmHead),
    /// int4 `MatMulNBits` shared LM-head projected on the GPU (or CPU) via a
    /// standalone single-node session that reuses the target's quantised
    /// initializers zero-copy. Selected when the target LM-head is quantised.
    #[cfg(feature = "native-backend")]
    Quantized(QuantizedDraftLmHead),
}

impl LmHead for MtpLmHead {
    fn vocab_size(&self) -> usize {
        match self {
            Self::Linear(lm_head) => lm_head.vocab_size(),
            Self::TargetInitializer(lm_head) => lm_head.vocab_size(),
            #[cfg(feature = "native-backend")]
            Self::Quantized(lm_head) => lm_head.vocab_size(),
        }
    }

    fn logits(&self, hidden: &[f32], out: &mut [f32]) -> anyhow::Result<()> {
        match self {
            Self::Linear(lm_head) => lm_head.logits(hidden, out),
            Self::TargetInitializer(lm_head) => lm_head.logits(hidden, out),
            #[cfg(feature = "native-backend")]
            Self::Quantized(lm_head) => lm_head.logits(hidden, out),
        }
    }
}

/// Device on which the int4 draft LM-head `MatMulNBits` projection runs.
///
/// The projection is a standalone single-node session built from the target's
/// own quantised LM-head initializers. It runs *outside* the captured native
/// decode step (during proposal), so it never affects CUDA-graph capture of the
/// target decode step.
#[derive(Debug, Clone, Copy)]
// An ORT-only build still threads `Option<DraftProjectionDevice>` through the
// adapter loader, but only the native path ever constructs one; outside that
// feature, constructions are limited to tests.
#[cfg_attr(not(feature = "native-backend"), allow(dead_code))]
pub(crate) enum DraftProjectionDevice {
    /// Project on the CPU int4 `MatMulNBits` kernel (used by unit tests and the
    /// native CPU backend).
    Cpu,
    /// Project on the native CUDA int4 `MatMulNBits` kernel at the given ordinal.
    Cuda {
        #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
        index: u32,
    },
}

fn is_dense_float_dtype(dtype: IrDataType) -> bool {
    matches!(
        dtype,
        IrDataType::Float32 | IrDataType::Float16 | IrDataType::BFloat16
    )
}

pub(crate) fn load_target_initializer_adapters(
    model_path: &Path,
    embedding_name: &str,
    lm_head_name: &str,
    hidden_size: usize,
    projection: Option<DraftProjectionDevice>,
) -> anyhow::Result<(MtpEmbedder, MtpLmHead, usize)> {
    let (graph, store) =
        onnx_runtime_loader::load_model_with_weights(model_path).with_context(|| {
            format!(
                "Failed to load target initializers from '{}'",
                model_path.display()
            )
        })?;
    let find_weight = |name: &str| -> anyhow::Result<WeightRef> {
        graph
            .initializers
            .iter()
            .find_map(|(&value_id, weight)| {
                (graph.value(value_id).name.as_deref() == Some(name)).then(|| weight.clone())
            })
            .with_context(|| format!("target model initializer '{name}' was not found"))
    };
    let embedding_weight = find_weight(embedding_name)?;
    let lm_head_weight = find_weight(lm_head_name)?;
    let embedding_dims = embedding_weight.dims();
    if embedding_dims.len() != 2 || embedding_dims[1] != hidden_size {
        anyhow::bail!(
            "target embedding initializer '{embedding_name}' shape {embedding_dims:?} must be [vocab, {hidden_size}]"
        );
    }
    let vocab_size = embedding_dims[0];
    let embedder = TargetInitializerEmbedder {
        matrix: TargetInitializerMatrix::new(
            Arc::clone(&store),
            embedding_weight,
            vocab_size,
            hidden_size,
        )?,
    };
    let lm_head_dims = lm_head_weight.dims();
    // The MTP draft head reuses the target's shared LM-head to project its hidden
    // state into draft logits. A *dense* [vocab, hidden] / [hidden, vocab] weight
    // (f32/f16/bf16) is projected host-side by `TargetInitializerLmHead`. An int4
    // MatMulNBits-quantised lm_head is stored as a 3-D packed uint8 blob with
    // companion scales / zero_points: dequantising the full matrix is multi-GB
    // and a host-side draft GEMV per step is not throughput-viable, so it is
    // projected on the GPU (or CPU) via a standalone single-node MatMulNBits
    // session that reuses the target's own quantised initializers zero-copy —
    // running *outside* the captured decode step, so capture-safety is preserved.
    if lm_head_dims.len() != 2 || !is_dense_float_dtype(lm_head_weight.dtype()) {
        #[cfg(feature = "native-backend")]
        if let Some(projection) = projection {
            let (lm_head, head_vocab) = build_quantized_draft_lm_head(
                &graph,
                Arc::clone(&store),
                lm_head_name,
                hidden_size,
                projection,
            )?;
            if head_vocab != vocab_size {
                anyhow::bail!(
                    "target quantised LM-head vocabulary {head_vocab} does not match embedding vocabulary {vocab_size}"
                );
            }
            return Ok((
                MtpEmbedder::TargetInitializer(embedder),
                MtpLmHead::Quantized(lm_head),
                vocab_size,
            ));
        }
        anyhow::bail!(
            "target LM-head initializer '{lm_head_name}' has shape {lm_head_dims:?} dtype {:?}, \
             which is not a dense f32/f16/bf16 [vocab, hidden] matrix. A quantised (e.g. int4 \
             MatMulNBits) shared LM-head requires a GPU/CPU projection device (only the native \
             backend supplies one); it cannot be dequantised host-side.",
            lm_head_weight.dtype()
        );
    }
    let _ = projection;
    let (layout, rows, cols) = if lm_head_dims == [hidden_size, vocab_size] {
        (
            LmHeadInitializerLayout::HiddenVocab,
            hidden_size,
            vocab_size,
        )
    } else if lm_head_dims == [vocab_size, hidden_size] {
        (
            LmHeadInitializerLayout::VocabHidden,
            vocab_size,
            hidden_size,
        )
    } else {
        anyhow::bail!(
            "target LM-head initializer '{lm_head_name}' shape {lm_head_dims:?} must be [{hidden_size}, {vocab_size}] or [{vocab_size}, {hidden_size}]"
        );
    };
    let lm_head = TargetInitializerLmHead {
        matrix: TargetInitializerMatrix::new(store, lm_head_weight, rows, cols)?,
        layout,
        hidden: hidden_size,
        vocab: vocab_size,
    };
    Ok((
        MtpEmbedder::TargetInitializer(embedder),
        MtpLmHead::TargetInitializer(lm_head),
        vocab_size,
    ))
}

/// A shared int4 `MatMulNBits` LM-head projected on-device (or on the CPU int4
/// kernel) for MTP draft-token generation.
///
/// The projection is a standalone single-node `InferenceSession` built from the
/// target model's *own* quantised LM-head initializers (weight / scales /
/// zero_points), reused zero-copy through the same [`WeightStore`] mmap — no
/// re-export, no host-side dequantisation. `logits` feeds one hidden vector per
/// draft step and reads back the full vocabulary logits. This runs during
/// proposal, *outside* the captured native decode step, so it never perturbs
/// CUDA-graph capture of the target step (fallbacks stay at zero). Draft-token
/// exactness is not required for correctness — drafts are verified against the
/// target, so an imperfect projection only lowers the acceptance rate.
#[cfg(feature = "native-backend")]
pub(crate) struct QuantizedDraftLmHead {
    session: Arc<std::sync::Mutex<onnx_runtime_session::InferenceSession>>,
    input_name: String,
    hidden: usize,
    vocab: usize,
}

#[cfg(feature = "native-backend")]
impl std::fmt::Debug for QuantizedDraftLmHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedDraftLmHead")
            .field("hidden", &self.hidden)
            .field("vocab", &self.vocab)
            .finish()
    }
}

#[cfg(feature = "native-backend")]
impl Clone for QuantizedDraftLmHead {
    fn clone(&self) -> Self {
        Self {
            session: Arc::clone(&self.session),
            input_name: self.input_name.clone(),
            hidden: self.hidden,
            vocab: self.vocab,
        }
    }
}

#[cfg(feature = "native-backend")]
impl LmHead for QuantizedDraftLmHead {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn logits(&self, hidden: &[f32], out: &mut [f32]) -> anyhow::Result<()> {
        if hidden.len() != self.hidden {
            anyhow::bail!(
                "quantised lm-head input length {} != hidden {}",
                hidden.len(),
                self.hidden
            );
        }
        if out.len() != self.vocab {
            anyhow::bail!(
                "quantised lm-head output length {} != vocab {}",
                out.len(),
                self.vocab
            );
        }
        let input = onnx_runtime_session::Tensor::from_f32(&[1, self.hidden], hidden)
            .context("build draft lm-head projection input")?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("draft lm-head projection session lock poisoned"))?;
        let outputs = session
            .run(&[(self.input_name.as_str(), &input)])
            .context("run draft lm-head projection")?;
        let logits = outputs
            .into_iter()
            .next()
            .context("draft lm-head projection produced no output")?;
        match logits.dtype {
            IrDataType::Float32 => {
                let values = logits.to_vec_f32();
                if values.len() != self.vocab {
                    anyhow::bail!(
                        "draft lm-head projection returned {} logits, expected {}",
                        values.len(),
                        self.vocab
                    );
                }
                out.copy_from_slice(&values);
            }
            other => anyhow::bail!("draft lm-head projection returned unsupported dtype {other:?}"),
        }
        Ok(())
    }
}

/// Build a [`QuantizedDraftLmHead`] from a loaded target `graph` by locating the
/// `MatMulNBits` node that consumes `lm_head_name` and cloning it into a
/// standalone single-node projection session on `projection`.
///
/// All shape/quantisation parameters (`K`, `N`, `bits`, `block_size`, the
/// scales / zero_points inputs) are derived from the discovered node — nothing
/// is hardcoded. The three quantised initializers are reused zero-copy from the
/// same [`WeightStore`], which is handed to the projection session so its mmap
/// stays alive.
#[cfg(feature = "native-backend")]
fn build_quantized_draft_lm_head(
    graph: &onnx_runtime_ir::Graph,
    store: Arc<WeightStore>,
    lm_head_name: &str,
    hidden_size: usize,
    projection: DraftProjectionDevice,
) -> anyhow::Result<(QuantizedDraftLmHead, usize)> {
    use onnx_runtime_ir::{Attribute, Node, NodeId, static_shape};

    // The native loader lowers int4 `MatMulNBits` into an explicit dequant +
    // dense `MatMul` subgraph (BitShift / BitwiseAnd / Cast / Sub / Mul), so the
    // loaded IR contains no `MatMulNBits` node to clone. The three quantised
    // initializers (weight / scales / zero_points) do persist, however, so we
    // reconstruct a clean single-node `MatMulNBits` projection from them and let
    // the projection session's EP re-run the quantised GEMV. `weight`/`scales`
    // are required; `zero_points` is optional (symmetric quantisation).
    let find_initializer = |name: &str| -> Option<WeightRef> {
        graph.initializers.iter().find_map(|(&vid, weight)| {
            (graph.value(vid).name.as_deref() == Some(name)).then(|| weight.clone())
        })
    };
    let weight_ref = find_initializer(lm_head_name)
        .with_context(|| format!("shared LM-head initializer '{lm_head_name}' was not found"))?;
    // Companion scales / zero_points follow the `MatMulNBits` export convention:
    // the same prefix as the packed weight with a `.scales` / `.zero_points`
    // suffix (e.g. `lm_head.weight` -> `lm_head.scales`).
    let prefix = lm_head_name
        .rsplit_once('.')
        .map(|(head, _)| head)
        .unwrap_or(lm_head_name);
    let scales_name = format!("{prefix}.scales");
    let zero_points_name = format!("{prefix}.zero_points");
    let scales_ref = find_initializer(&scales_name).with_context(|| {
        format!(
            "quantised LM-head '{lm_head_name}' companion scales initializer '{scales_name}' \
             was not found"
        )
    })?;
    let zero_points_ref = find_initializer(&zero_points_name);

    // Derive all quantisation geometry from the initializer shapes and the
    // configured hidden size — nothing is hardcoded. The packed weight is
    // [N, k_blocks, blob] uint8; K = hidden_size; block_size = K / k_blocks;
    // bits = blob_bytes * 8 / block_size (int4 packs two weights per byte).
    let weight_dims = weight_ref.dims();
    if weight_dims.len() != 3 {
        anyhow::bail!(
            "quantised LM-head weight '{lm_head_name}' must be a 3-D packed [N, k_blocks, blob] \
             tensor, got shape {weight_dims:?}"
        );
    }
    let n = weight_dims[0];
    let k_blocks = weight_dims[1];
    let blob = weight_dims[2];
    let k = hidden_size;
    if k_blocks == 0 || !k.is_multiple_of(k_blocks) {
        anyhow::bail!(
            "quantised LM-head weight '{lm_head_name}' k_blocks {k_blocks} does not divide hidden \
             size {k}"
        );
    }
    let block_size = (k / k_blocks) as i64;
    if block_size == 0 || (blob * 8) % (block_size as usize) != 0 {
        anyhow::bail!(
            "quantised LM-head weight '{lm_head_name}' blob {blob} incompatible with block_size \
             {block_size}"
        );
    }
    let bits = (blob * 8 / block_size as usize) as i64;
    let scales_dims = scales_ref.dims().to_vec();
    if scales_dims != [n, k_blocks] {
        anyhow::bail!(
            "quantised LM-head scales '{scales_name}' shape {scales_dims:?} does not match \
             weight-derived [N={n}, k_blocks={k_blocks}]"
        );
    }
    // The MatMulNBits CUDA kernel requires Float32 scales; the exported artifact
    // stores them as BFloat16 (or Float16). Convert once, at build time, into an
    // inline Float32 initializer so the reconstructed projection node satisfies
    // the kernel's dtype contract. Float32 scales are passed through untouched.
    let scales_ref = match scales_ref.dtype() {
        IrDataType::Float32 => scales_ref,
        dtype @ (IrDataType::BFloat16 | IrDataType::Float16) => {
            let raw = store.bytes(&scales_ref).with_context(|| {
                format!("resolve bytes for quantised LM-head scales '{scales_name}'")
            })?;
            if raw.len() % 2 != 0 {
                anyhow::bail!(
                    "quantised LM-head scales '{scales_name}' byte length {} is not 2-byte aligned",
                    raw.len()
                );
            }
            let mut f32_bytes = Vec::with_capacity(raw.len() * 2);
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let value = match dtype {
                    IrDataType::BFloat16 => half::bf16::from_bits(bits).to_f32(),
                    _ => half::f16::from_bits(bits).to_f32(),
                };
                f32_bytes.extend_from_slice(&value.to_le_bytes());
            }
            WeightRef::Inline(onnx_runtime_ir::TensorData::from_raw(
                IrDataType::Float32,
                scales_dims.to_vec(),
                f32_bytes,
            ))
        }
        other => anyhow::bail!(
            "quantised LM-head scales '{scales_name}' has unsupported dtype {other:?}; \
             expected Float32, BFloat16, or Float16"
        ),
    };
    let mut projection_graph = onnx_runtime_ir::Graph::default();
    projection_graph
        .opset_imports
        .insert("com.microsoft".to_string(), 1);
    let hidden_value = projection_graph.create_named_value(
        "draft_hidden",
        IrDataType::Float32,
        static_shape([1, k]),
    );
    projection_graph.add_input(hidden_value);
    let add_initializer = |graph: &mut onnx_runtime_ir::Graph, name: &str, weight: &WeightRef| {
        let value = graph.create_named_value(
            name,
            weight.dtype(),
            static_shape(weight.dims().iter().copied()),
        );
        graph.set_initializer(value, weight.clone());
        value
    };
    let weight_value = add_initializer(&mut projection_graph, "draft_lm_head.weight", &weight_ref);
    let scales_value = add_initializer(&mut projection_graph, "draft_lm_head.scales", &scales_ref);
    let mut inputs = vec![Some(hidden_value), Some(weight_value), Some(scales_value)];
    if let Some(zero_points_ref) = &zero_points_ref {
        let zero_points_value = add_initializer(
            &mut projection_graph,
            "draft_lm_head.zero_points",
            zero_points_ref,
        );
        inputs.push(Some(zero_points_value));
    }
    let logits_value = projection_graph.create_named_value(
        "draft_logits",
        IrDataType::Float32,
        static_shape([1, n]),
    );
    let mut projection_node = Node::new(NodeId(0), "MatMulNBits", inputs, vec![logits_value]);
    projection_node.domain = "com.microsoft".to_string();
    projection_node
        .attributes
        .insert("K".to_string(), Attribute::Int(k as i64));
    projection_node
        .attributes
        .insert("N".to_string(), Attribute::Int(n as i64));
    projection_node
        .attributes
        .insert("bits".to_string(), Attribute::Int(bits));
    projection_node
        .attributes
        .insert("block_size".to_string(), Attribute::Int(block_size));
    projection_graph.insert_node(projection_node);
    projection_graph.add_output(logits_value);

    let provider: Arc<dyn onnx_runtime_ep_api::ExecutionProvider> = match projection {
        DraftProjectionDevice::Cpu => Arc::new(onnx_runtime_ep_cpu::CpuExecutionProvider::new()),
        #[cfg(feature = "native-cuda")]
        DraftProjectionDevice::Cuda { index } => Arc::new(
            onnx_runtime_ep_cuda::CudaExecutionProvider::initialized(index)
                .context("initialize CUDA EP for int4 draft LM-head projection")?,
        ),
        #[cfg(not(feature = "native-cuda"))]
        DraftProjectionDevice::Cuda { .. } => anyhow::bail!(
            "int4 draft LM-head projection on CUDA requires the `native-cuda` feature"
        ),
    };
    let session = onnx_runtime_session::InferenceSession::from_graph_with_provider(
        projection_graph,
        store,
        Path::new("."),
        provider,
    )
    .context("build int4 draft LM-head projection session")?;
    Ok((
        QuantizedDraftLmHead {
            session: Arc::new(std::sync::Mutex::new(session)),
            input_name: "draft_hidden".to_string(),
            hidden: hidden_size,
            vocab: n,
        },
        n,
    ))
}

impl LinearLmHead {
    pub fn new(weight: Vec<f32>, hidden: usize, vocab: usize) -> anyhow::Result<Self> {
        if weight.len() != hidden * vocab {
            anyhow::bail!(
                "lm-head weight length {} != hidden {hidden} * vocab {vocab}",
                weight.len()
            );
        }
        Ok(Self {
            weight,
            hidden,
            vocab,
        })
    }
}

impl LmHead for LinearLmHead {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn logits(&self, hidden: &[f32], out: &mut [f32]) -> anyhow::Result<()> {
        if hidden.len() != self.hidden {
            anyhow::bail!(
                "lm-head input length {} != hidden {}",
                hidden.len(),
                self.hidden
            );
        }
        if out.len() != self.vocab {
            anyhow::bail!(
                "lm-head output length {} != vocab {}",
                out.len(),
                self.vocab
            );
        }
        for (col, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (row, &value) in hidden.iter().enumerate() {
                acc += value * self.weight[row * self.vocab + col];
            }
            *slot = acc;
        }
        Ok(())
    }
}

/// Index of the maximum logit, resolving ties to the lowest index.
pub fn argmax(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .fold(None, |best, (index, &value)| match best {
            Some((_, best_value)) if value <= best_value => best,
            _ => Some((index, value)),
        })
        .map(|(index, _)| index)
}

/// Speculative acceptance rule implemented by the Phase 3 engine path.
///
/// `Greedy` is the rule exercised by the live target-verification loop today.
/// `RejectionSampling` and `Typical` are declared per DESIGN §3.5 and consumed
/// by the tree-verification core in [`tree`]; the linear path is unaffected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcceptanceRule {
    /// Accept a draft token iff it matches the target model's greedy argmax.
    Greedy,
    /// Accept a draft token via the speculative rejection-sampling test.
    RejectionSampling,
    /// Accept a draft token iff its target probability clears `threshold`.
    Typical { threshold: f32 },
}

/// Result of a single greedy speculative verification step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreedyStep {
    /// Number of proposed draft tokens accepted before the first mismatch.
    pub accepted_prefix_len: usize,
    /// Whether every proposed draft token was accepted.
    pub fully_accepted: bool,
}

/// Inputs a proposer needs to draft speculative candidates for one verify pass.
pub struct SpeculativeProposerContext<'a> {
    pub width: usize,
    pub context_tokens: &'a [TokenId],
    pub generated_tokens: &'a [TokenId],
    pub generated_text: &'a str,
    pub first_step: usize,
    pub options: &'a GenerateOptions,
    pub chain: &'a ProcessorChain,
    /// Target decoder's last hidden state, when required by the proposer.
    pub target_hidden: Option<&'a [f32]>,
    /// Target decoder's selected low/middle/high last-token hidden states.
    ///
    /// EAGLE-3 concatenates these in order to form its `fused_hidden` input.
    pub target_hidden_layers: Option<&'a [Vec<f32>]>,
    /// Target model's unprocessed greedy next token.
    pub guaranteed_token: Option<TokenId>,
    /// Target KV slices bound to a shared-KV proposer's `shared_kv.*` inputs.
    pub shared_kv_slices: Option<&'a [SharedKvInput]>,
}

/// Aggregate diagnostics for one speculative generation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeStats {
    pub verification_steps: usize,
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub multi_token_accepts: usize,
}

/// Candidate tokens proposed for a target-model verification pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeProposal {
    pub tokens: Vec<TokenId>,
    pub positions: Option<Vec<usize>>,
    pub tree: Option<Vec<Vec<usize>>>,
}

impl SpeculativeProposal {
    pub fn linear(tokens: Vec<TokenId>) -> Self {
        Self {
            tokens,
            positions: None,
            tree: None,
        }
    }
}

/// Outcome reported back to the proposer after verification and commit.
pub struct SpeculativeAcceptContext<'a> {
    pub accepted_prefix_len: usize,
    pub committed_tokens: &'a [TokenId],
    pub target_tokens: &'a [TokenId],
}

/// Source of speculative draft tokens.
pub trait SpeculativeProposer {
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal>;

    fn accept(&mut self, _context: &SpeculativeAcceptContext<'_>) -> anyhow::Result<()> {
        Ok(())
    }

    fn rewind(&mut self, _target_tokens: &[TokenId]) -> anyhow::Result<()> {
        Ok(())
    }

    fn name(&self) -> &str;
}

/// Model-free proposer that copies the continuation after the most recent
/// earlier occurrence of the current context suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramProposer {
    ngram: usize,
    max_tokens: usize,
}

impl NgramProposer {
    pub fn new(ngram: usize, max_tokens: usize) -> anyhow::Result<Self> {
        if ngram == 0 {
            anyhow::bail!("ngram must be greater than zero");
        }
        if max_tokens == 0 {
            anyhow::bail!("max_tokens must be greater than zero");
        }
        Ok(Self { ngram, max_tokens })
    }
}

impl SpeculativeProposer for NgramProposer {
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal> {
        let tokens = context.context_tokens;
        if tokens.len() <= self.ngram {
            return Ok(SpeculativeProposal::linear(Vec::new()));
        }

        let suffix_start = tokens.len() - self.ngram;
        let suffix = &tokens[suffix_start..];
        let Some(match_start) = (0..suffix_start).rev().find(|&start| {
            start + self.ngram < tokens.len() && &tokens[start..start + self.ngram] == suffix
        }) else {
            return Ok(SpeculativeProposal::linear(Vec::new()));
        };

        let continuation_start = match_start + self.ngram;
        let continuation_len = context
            .width
            .min(self.max_tokens)
            .min(tokens.len() - continuation_start);
        Ok(SpeculativeProposal::linear(
            tokens[continuation_start..continuation_start + continuation_len].to_vec(),
        ))
    }

    fn name(&self) -> &str {
        "prompt_lookup"
    }
}

/// Multi-token-prediction proposer backed by an ORT MTP-head session.
pub struct MtpProposer<'a, E = LinearEmbedder, L = LinearLmHead> {
    session: MtpDecodeSession<'a>,
    embedder: E,
    lm_head: L,
    cache_scope: MtpCacheScope,
}

impl<'a, E, L> MtpProposer<'a, E, L>
where
    E: TokenEmbedder,
    L: LmHead,
{
    pub fn new(
        head: &'a Session,
        options: MtpDecodeOptions,
        embedder: E,
        lm_head: L,
    ) -> anyhow::Result<Self> {
        Self::new_with_cache_scope(
            head,
            options,
            embedder,
            lm_head,
            MtpCacheScope::ProposalLocal,
        )
    }

    pub fn new_with_cache_scope(
        head: &'a Session,
        options: MtpDecodeOptions,
        embedder: E,
        lm_head: L,
        cache_scope: MtpCacheScope,
    ) -> anyhow::Result<Self> {
        let session = MtpDecodeSession::new(head, options)
            .map_err(|error| anyhow::anyhow!("Failed to create MTP decode session: {error}"))?;
        if session.signature().hidden_size != embedder.hidden_size() {
            anyhow::bail!(
                "MTP head hidden size {} does not match target embedding hidden size {}",
                session.signature().hidden_size,
                embedder.hidden_size()
            );
        }
        Ok(Self {
            session,
            embedder,
            lm_head,
            cache_scope,
        })
    }
}

impl<E, L> MtpProposer<'static, E, L>
where
    E: TokenEmbedder,
    L: LmHead,
{
    pub fn new_owned(
        head: Arc<Session>,
        options: MtpDecodeOptions,
        embedder: E,
        lm_head: L,
        cache_scope: MtpCacheScope,
    ) -> anyhow::Result<Self> {
        let session = MtpDecodeSession::new_owned(head, options)
            .map_err(|error| anyhow::anyhow!("Failed to create MTP decode session: {error}"))?;
        if session.signature().hidden_size != embedder.hidden_size() {
            anyhow::bail!(
                "MTP head hidden size {} does not match target embedding hidden size {}",
                session.signature().hidden_size,
                embedder.hidden_size()
            );
        }
        Ok(Self {
            session,
            embedder,
            lm_head,
            cache_scope,
        })
    }
}

impl<E, L> SpeculativeProposer for MtpProposer<'_, E, L>
where
    E: TokenEmbedder,
    L: LmHead,
{
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal> {
        let hidden = context
            .target_hidden
            .context("MTP proposer requires the target model's last hidden state")?;
        let guaranteed_token = context
            .guaranteed_token
            .context("MTP proposer requires the target model's greedy next token")?;
        let draft_count = context.width.saturating_sub(1);
        let expected_state_len = self
            .session
            .signature()
            .hidden_size
            .checked_mul(self.session.hc_mult())
            .context("MTP HC state width overflow")?;
        if hidden.len() != expected_state_len {
            anyhow::bail!(
                "target_hidden length {} != hc_mult {} * hidden {}",
                hidden.len(),
                self.session.hc_mult(),
                self.session.signature().hidden_size
            );
        }
        if self.cache_scope == MtpCacheScope::ProposalLocal {
            self.session.reset();
        } else if context.first_step > 0 {
            anyhow::bail!(
                "MTP accepted_prefix KV reuse is not enabled: the frozen Mobius contract does not define correction-token/cache alignment"
            );
        }
        let mut tokens = Vec::with_capacity(draft_count + 1);
        tokens.push(guaranteed_token);
        let mut running_state = hidden.to_vec();
        let mut previous_token = guaranteed_token;
        let mut embedding = vec![0.0f32; self.session.signature().hidden_size];
        let mut logits = vec![0.0f32; self.lm_head.vocab_size()];
        for draft_index in 0..draft_count {
            self.embedder.embed(previous_token, &mut embedding)?;
            let position = i64::try_from(
                context
                    .context_tokens
                    .len()
                    .checked_add(draft_index)
                    .context("MTP absolute position overflow")?,
            )
            .context("MTP position exceeds i64")?;
            let output = self
                .session
                .step_with_state(&embedding, &running_state, position)
                .map_err(|error| anyhow::anyhow!("MTP proposal step failed: {error}"))?;
            self.lm_head.logits(&output.hidden, &mut logits)?;
            let token = argmax(&logits).context("lm-head produced empty logits")? as TokenId;
            tokens.push(token);
            running_state = output.state;
            previous_token = token;
        }
        Ok(SpeculativeProposal {
            tokens,
            positions: None,
            tree: None,
        })
    }

    fn accept(&mut self, context: &SpeculativeAcceptContext<'_>) -> anyhow::Result<()> {
        if self.cache_scope == MtpCacheScope::ProposalLocal
            || self.session.mode() == onnx_genai_ort::MtpDraftKvMode::HiddenThreaded
        {
            self.session.reset();
            return Ok(());
        }
        self.session
            .rewind(context.accepted_prefix_len.saturating_sub(1))
            .map_err(|error| anyhow::anyhow!("Failed to rewind MTP proposal: {error}"))
    }

    fn rewind(&mut self, _target_tokens: &[TokenId]) -> anyhow::Result<()> {
        self.session.reset();
        Ok(())
    }

    fn name(&self) -> &str {
        "mtp"
    }
}

/// EAGLE-3 proposer backed by an autoregressive ORT draft-head session.
pub struct Eagle3Proposer<'a, E = LinearEmbedder> {
    session: Eagle3DecodeSession<'a>,
    embedder: E,
    token_map: Option<Vec<TokenId>>,
}

impl<'a, E> Eagle3Proposer<'a, E>
where
    E: TokenEmbedder,
{
    pub fn new(
        head: &'a Session,
        options: Eagle3DecodeOptions,
        embedder: E,
    ) -> anyhow::Result<Self> {
        let session = Eagle3DecodeSession::new(head, options)
            .map_err(|error| anyhow::anyhow!("Failed to create EAGLE-3 decode session: {error}"))?;
        if session.signature().hidden_size != embedder.hidden_size() {
            anyhow::bail!(
                "EAGLE-3 head hidden size {} does not match target embedding hidden size {}",
                session.signature().hidden_size,
                embedder.hidden_size()
            );
        }
        Ok(Self {
            session,
            embedder,
            token_map: None,
        })
    }

    /// Translate proposer vocabulary ids before embedding and verification.
    pub fn with_token_map(mut self, token_map: Vec<TokenId>) -> anyhow::Result<Self> {
        if token_map.len() < self.session.signature().draft_vocab_size {
            anyhow::bail!(
                "proposer token map has {} entries but logits expose {} ids",
                token_map.len(),
                self.session.signature().draft_vocab_size
            );
        }
        // Every mapped id is embedded through the target embedding table and
        // verified against the target vocabulary, so an id at or beyond that
        // vocabulary has no row and cannot be a real target token. Reject the
        // whole map at load time rather than faulting mid-proposal on whichever
        // draft id happens to select the out-of-range entry.
        let vocab = self.embedder.vocab_size();
        if let Some((index, mapped)) = token_map
            .iter()
            .copied()
            .enumerate()
            .find(|&(_, id)| id as usize >= vocab)
        {
            anyhow::bail!(
                "proposer token map entry {index} maps to target id {mapped}, but the \
                 target/embedder vocabulary has only {vocab} ids; every mapped id must be a \
                 valid target token"
            );
        }
        self.token_map = Some(token_map);
        Ok(self)
    }
}

impl<E> SpeculativeProposer for Eagle3Proposer<'_, E>
where
    E: TokenEmbedder,
{
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal> {
        let layers = context
            .target_hidden_layers
            .context("EAGLE-3 proposer requires low/middle/high target hidden states")?;
        let guaranteed_token = context
            .guaranteed_token
            .context("EAGLE-3 proposer requires the target model's greedy next token")?;
        let hidden_size = self.session.signature().hidden_size;
        if layers.len() != 3 || layers.iter().any(|layer| layer.len() != hidden_size) {
            anyhow::bail!(
                "EAGLE-3 requires exactly three target hidden states of width {hidden_size}"
            );
        }
        let mut fused_hidden = Vec::with_capacity(self.session.signature().fused_hidden_size);
        for layer in layers {
            fused_hidden.extend_from_slice(layer);
        }
        if fused_hidden.len() != self.session.signature().fused_hidden_size {
            anyhow::bail!(
                "fused target hidden length {} != EAGLE-3 head fused hidden {}",
                fused_hidden.len(),
                self.session.signature().fused_hidden_size
            );
        }
        let mut running_hidden = context
            .target_hidden
            .map(<[f32]>::to_vec)
            .unwrap_or_else(|| layers[2].clone());
        if running_hidden.len() != hidden_size {
            anyhow::bail!(
                "EAGLE-3 recycled target hidden length {} != hidden {hidden_size}",
                running_hidden.len()
            );
        }

        self.session.reset();
        let draft_count = context.width.saturating_sub(1);
        let mut tokens = Vec::with_capacity(draft_count + 1);
        tokens.push(guaranteed_token);
        let mut previous_token = guaranteed_token;
        let mut embedding = vec![0.0f32; hidden_size];
        for step in 0..draft_count {
            self.embedder.embed(previous_token, &mut embedding)?;
            let position = i64::try_from(context.context_tokens.len() + step)
                .context("EAGLE-3 position exceeds i64")?;
            let output = self
                .session
                .step(&embedding, &fused_hidden, &running_hidden, position)
                .map_err(|error| anyhow::anyhow!("EAGLE-3 proposal step failed: {error}"))?;
            let draft_id =
                argmax(&output.logits).context("EAGLE-3 head produced empty draft logits")?;
            let token = if let Some(token_map) = &self.token_map {
                *token_map
                    .get(draft_id)
                    .context("proposer token id is absent from the declared vocabulary map")?
            } else {
                TokenId::try_from(draft_id).context("EAGLE-3 token id exceeds u32 range")?
            };
            tokens.push(token);
            previous_token = token;
            running_hidden = output.hidden;
        }
        Ok(SpeculativeProposal::linear(tokens))
    }

    fn accept(&mut self, _context: &SpeculativeAcceptContext<'_>) -> anyhow::Result<()> {
        // Every verification pass receives a fresh target low/mid/high anchor.
        // Keeping draft-only KV across passes would make rejected features stale.
        self.session.reset();
        Ok(())
    }

    fn rewind(&mut self, _target_tokens: &[TokenId]) -> anyhow::Result<()> {
        self.session.reset();
        Ok(())
    }

    fn name(&self) -> &str {
        "eagle3"
    }
}

pub(crate) struct DraftModelProposer<'a> {
    draft_model: &'a mut DraftModel,
    draft_state: &'a mut DraftSession,
    rng: Option<&'a mut SamplingRng>,
}

impl<'a> DraftModelProposer<'a> {
    fn new(draft_model: &'a mut DraftModel, draft_state: &'a mut DraftSession) -> Self {
        Self {
            draft_model,
            draft_state,
            rng: None,
        }
    }

    fn with_rng(
        draft_model: &'a mut DraftModel,
        draft_state: &'a mut DraftSession,
        rng: &'a mut SamplingRng,
    ) -> Self {
        Self {
            draft_model,
            draft_state,
            rng: Some(rng),
        }
    }

    fn align_to_target_prefix(
        &mut self,
        target_tokens: &[TokenId],
        prefix_len: usize,
    ) -> anyhow::Result<()> {
        self.draft_state.tokens = target_tokens[..prefix_len].to_vec();
        if self.draft_state.kv_token_count > prefix_len {
            rewind_draft_state_to_len(
                self.draft_model,
                self.draft_state,
                RewindRequest::new(prefix_len, RewindRunnerPolicy::AllowRunnerRewind),
            )?;
        }
        Ok(())
    }
}

impl SpeculativeProposer for DraftModelProposer<'_> {
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal> {
        let mut fallback_rng = SamplingRng::new(context.options.seed);
        let rng = self.rng.as_deref_mut().unwrap_or(&mut fallback_rng);
        let tokens = propose_draft_tokens(DraftProposalRequest {
            draft_model: self.draft_model,
            draft_state: self.draft_state,
            width: context.width,
            generated_tokens: context.generated_tokens,
            generated_text: context.generated_text,
            first_step: context.first_step,
            options: context.options,
            chain: context.chain,
            rng,
        })?;
        Ok(SpeculativeProposal::linear(tokens))
    }

    fn rewind(&mut self, target_tokens: &[TokenId]) -> anyhow::Result<()> {
        let common_len = common_prefix_len(&self.draft_state.tokens, target_tokens);
        if self.draft_state.kv_token_count > common_len {
            rewind_draft_state_to_len(
                self.draft_model,
                self.draft_state,
                RewindRequest::new(common_len, RewindRunnerPolicy::AllowRunnerRewind),
            )?;
        }
        self.draft_state.tokens = target_tokens.to_vec();
        Ok(())
    }

    fn name(&self) -> &str {
        "draft_model"
    }
}

/// Bundled inputs for [`Engine::generate_speculative_loop`]: the session
/// identifier and its mutable engine state, the generation options and
/// processor chain, the context limit and prefix-cache hit length, the mutable
/// output accumulators (generated tokens, text, and log-probabilities), the
/// sampling RNG, and the optional per-token callback.
pub(crate) struct SpeculativeLoopState<'state, 'callback> {
    pub(crate) session_id: SessionId,
    pub(crate) state: &'state mut EngineSession,
    pub(crate) options: &'state GenerateOptions,
    pub(crate) chain: &'state ProcessorChain,
    pub(crate) max_context: Option<usize>,
    pub(crate) prefix_cache_hit_len: usize,
    pub(crate) generated_tokens: &'state mut Vec<TokenId>,
    pub(crate) generated_text: &'state mut String,
    pub(crate) generated_logprobs: &'state mut Option<Vec<crate::config::TokenLogprob>>,
    pub(crate) rng: &'state mut SamplingRng,
    pub(crate) callback: Option<&'state mut GenerateTokenCallback<'callback>>,
}

/// Target-model forward result for one speculative step: the base next-token
/// logits, any hidden state(s) the active proposer consumes, and the target's
/// unprocessed greedy next token (used by hidden-state proposers).
struct TargetPrediction {
    base_logits: Vec<f32>,
    target_hidden: Option<Vec<f32>>,
    target_hidden_layers: Option<Vec<Vec<f32>>>,
    guaranteed_token: Option<TokenId>,
}

/// Immutable per-step inputs threaded into [`Engine::propose_candidates`].
struct CandidateProposalInputs<'a> {
    width: usize,
    step: usize,
    base_len: usize,
    prediction: &'a TargetPrediction,
    shared_kv_slices: Option<&'a [SharedKvInput]>,
    generated_tokens: &'a [TokenId],
    generated_text: &'a str,
    options: &'a GenerateOptions,
    chain: &'a ProcessorChain,
}

/// Longest-accepted-prefix decision for one verification pass: the number of
/// accepted draft tokens, the correction token when the first mismatch occurred,
/// and the per-candidate log-probabilities captured during selection.
struct AcceptedPrefix {
    accepted: usize,
    replacement: Option<TokenId>,
    candidate_logprobs: Option<Vec<crate::config::TokenLogprob>>,
}

impl Engine {
    fn speculative_mode(&self, options: &GenerateOptions) -> SpeculativeMode {
        options
            .speculative_mode
            .clone()
            .unwrap_or_else(|| self.speculative_mode.clone())
    }

    /// Whether the package permits the runtime to turn speculation on by itself.
    ///
    /// Speculation replaces one implementation of a contract with an equivalent
    /// one, so the package's declared equivalence class decides whether the swap
    /// is silently allowed. Only a `bitwise` or `distribution_preserving`
    /// contract may be substituted without the caller asking: a merely
    /// `semantic` equivalence is free to change the output distribution, which
    /// is exactly what a caller who did not ask for speculation did not agree to.
    ///
    /// A component that declares nothing is treated as `semantic` — the schema
    /// default — so silence never buys an automatic optimization. Note that an
    /// absent contract must be *counted* as semantic rather than skipped:
    /// filtering undeclared components out would make `all` vacuously true for a
    /// package whose components declare no contracts at all.
    pub(crate) fn permits_automatic_speculation(&self) -> bool {
        let Some(pipeline) = &self.metadata.pipeline else {
            return false;
        };
        components_permit_automatic_speculation(&pipeline.workflow.components)
    }

    pub(crate) fn should_use_speculative(&self, options: &GenerateOptions) -> bool {
        // Naming a mode — on the request or in the engine configuration — is an
        // explicit opt-in. Anything else is the runtime substituting a different
        // implementation of the same contract on its own, which only a
        // bitwise or distribution-preserving equivalence class permits.
        let opted_in = options.speculative_mode.is_some()
            || !matches!(self.speculative_mode, SpeculativeMode::None);
        if !opted_in && !self.permits_automatic_speculation() {
            return false;
        }
        let mode_available = match self.speculative_mode(options) {
            SpeculativeMode::None => false,
            SpeculativeMode::DraftModel => self.draft.is_some(),
            SpeculativeMode::PromptLookup { ngram, max_tokens } => ngram > 0 && max_tokens > 0,
            SpeculativeMode::Mtp(config) => {
                self.mtp.as_ref().is_some_and(|mtp| mtp.config == config)
            }
            SpeculativeMode::Eagle3(config) => self
                .eagle3
                .as_ref()
                .is_some_and(|eagle3| eagle3.config == config),
        };
        mode_available
            // Grammar processors carry per-request parser state; draft/verify
            // would need separate parser branches for speculative candidates.
            && options.constraint.is_none()
            && options.selects_greedily()
            && self.kv_model.is_some()
    }

    pub(crate) fn generate_speculative_loop(
        &mut self,
        loop_state: SpeculativeLoopState<'_, '_>,
    ) -> anyhow::Result<GenerateResult> {
        let SpeculativeLoopState {
            session_id,
            state,
            options,
            chain,
            max_context,
            prefix_cache_hit_len,
            generated_tokens,
            generated_text,
            generated_logprobs,
            rng,
            mut callback,
        } = loop_state;
        let speculative_mode = self.speculative_mode(options);
        let draft_width = self.resolve_draft_width(options, &speculative_mode)?;
        let mut mtp_proposer = if matches!(&speculative_mode, SpeculativeMode::Mtp(_)) {
            let mtp = self
                .mtp
                .as_ref()
                .context("MTP speculation requested without a loaded MTP head")?;
            Some(MtpProposer::new_owned(
                Arc::clone(&mtp.session),
                MtpDecodeOptions {
                    kv_mode: mtp.kv_mode,
                    batch_size: 1,
                    hc_mult: mtp.runtime_config.hc_mult,
                    hidden_state_rank4: mtp.runtime_config.target_hidden_layout
                        == MtpHiddenLayout::Bshc,
                    hidden_output: mtp.runtime_config.mtp_hidden_output.clone(),
                    state_output: mtp.runtime_config.mtp_state_output.clone(),
                },
                mtp.embedder.clone(),
                mtp.lm_head.clone(),
                mtp.runtime_config.cache_scope,
            )?)
        } else {
            None
        };
        let mut step = 0;

        loop {
            if let Some(result) = self.check_speculative_termination(
                generated_tokens,
                generated_text,
                generated_logprobs,
                state.tokens.len(),
                options,
                max_context,
                prefix_cache_hit_len,
            )? {
                return Ok(result);
            }

            let remaining_tokens = options.max_new_tokens - generated_tokens.len();
            let remaining_context = max_context
                .map(|limit| limit.saturating_sub(state.tokens.len()))
                .unwrap_or(remaining_tokens);
            let width = draft_width
                .min(remaining_tokens)
                .min(remaining_context)
                .max(1);

            let base_len = state.tokens.len();
            let base_generated_len = generated_tokens.len();

            let prediction = self.load_target_prediction(session_id, state, &speculative_mode)?;

            let shared_kv_slices: Option<Vec<SharedKvInput>> = None;

            let draft_tokens = self.propose_candidates(
                &speculative_mode,
                state,
                &mut mtp_proposer,
                rng,
                CandidateProposalInputs {
                    width,
                    step,
                    base_len,
                    prediction: &prediction,
                    shared_kv_slices: shared_kv_slices.as_deref(),
                    generated_tokens,
                    generated_text,
                    options,
                    chain,
                },
            )?;

            // Snapshot the pre-draft recurrent/conv state of a hybrid native
            // target BEFORE the verify window destructively advances it, so the
            // accept path can commit it to exactly the accepted prefix. Inert for
            // every non-native / pure-dense target (returns `None`).
            #[cfg(feature = "native-backend")]
            let recurrent_snapshot = match state.decode_state.native_recurrent_runner_mut() {
                Some(native) => Some(native.snapshot_recurrent_state()?),
                None => None,
            };

            let verified_logits =
                self.run_target_verification(session_id, state, &draft_tokens, base_len)?;

            let mut target_logits = Vec::with_capacity(draft_tokens.len() + 1);
            target_logits.push(prediction.base_logits);
            target_logits.extend(verified_logits);

            let accepted_prefix = self.choose_accepted_prefix(
                &mut target_logits,
                &draft_tokens,
                base_len,
                state,
                generated_tokens,
                options,
                chain,
                rng,
                step,
            )?;
            let accepted = accepted_prefix.accepted;

            // Commit the hybrid native target's recurrent/conv state to the
            // accepted prefix. Attention KV keeps the ordinary prefix-slice rewind
            // in `assemble_commit_tokens`; this only rebuilds the destructive
            // rolling caches (snapshot + accepted-token re-advance), then squares
            // the paged length bookkeeping so that rewind becomes a no-op.
            #[cfg(feature = "native-backend")]
            if let Some(snapshot) = recurrent_snapshot.as_ref() {
                self.commit_native_recurrent_target(
                    session_id,
                    state,
                    base_len,
                    &draft_tokens[..accepted],
                    snapshot,
                )?;
            }

            // Rewind the target KV to the accepted prefix and pick any correction
            // or bonus token. The rewind happens inside assemble_commit_tokens
            // BEFORE the token push in commit_speculative_tokens, so the commit
            // starts from exactly the accepted boundary (base_len + accepted).
            let (commit_tokens, commit_logprobs) = self.assemble_commit_tokens(
                &accepted_prefix,
                &draft_tokens,
                base_len,
                state,
                &mut target_logits,
                generated_tokens,
                options,
                chain,
                rng,
                step,
                max_context,
                session_id,
            )?;

            if matches!(&speculative_mode, SpeculativeMode::DraftModel) {
                self.notify_draft_acceptance(state, accepted, &commit_tokens)?;
            } else if let Some(proposer) = mtp_proposer.as_mut() {
                proposer.accept(&SpeculativeAcceptContext {
                    accepted_prefix_len: accepted,
                    committed_tokens: &commit_tokens,
                    target_tokens: &state.tokens,
                })?;
            }

            if let Some(result) = self.commit_speculative_tokens(
                session_id,
                state,
                generated_tokens,
                generated_text,
                generated_logprobs,
                commit_tokens,
                commit_logprobs,
                accepted,
                base_len,
                prefix_cache_hit_len,
                options,
                chain,
                &speculative_mode,
                &mut step,
                &mut callback,
                max_context,
            )? {
                return Ok(result);
            }

            if matches!(&speculative_mode, SpeculativeMode::DraftModel) {
                self.sync_draft_to_target(state)?;
            }

            if generated_tokens.len() == base_generated_len {
                anyhow::bail!("speculative decoding made no progress");
            }
        }
    }

    /// Resolve the configured maximum draft width for the active speculative
    /// mode, clamped to at least one token per step.
    fn resolve_draft_width(
        &self,
        options: &GenerateOptions,
        speculative_mode: &SpeculativeMode,
    ) -> anyhow::Result<usize> {
        let draft_width = match speculative_mode {
            SpeculativeMode::PromptLookup { max_tokens, .. } => *max_tokens,
            SpeculativeMode::Mtp(_) => {
                self.mtp
                    .as_ref()
                    .map(|mtp| {
                        options
                            .num_speculative_tokens
                            .unwrap_or(mtp.num_speculative_tokens)
                    })
                    .context("MTP speculation requested without a loaded MTP head")?
                    + 1
            }
            SpeculativeMode::Eagle3(_) => {
                self.eagle3
                    .as_ref()
                    .map(|eagle3| {
                        options
                            .num_speculative_tokens
                            .unwrap_or(eagle3.num_speculative_tokens)
                    })
                    .context("EAGLE-3 speculation requested without a loaded EAGLE-3 head")?
                    + 1
            }
            _ => options
                .num_speculative_tokens
                .unwrap_or(self.num_speculative_tokens),
        }
        .max(1);
        Ok(draft_width)
    }

    /// Emit an early [`GenerateResult`] when the max-token or context limit is
    /// reached before proposing this step, mirroring the greedy loop's stops.
    #[allow(clippy::too_many_arguments)]
    fn check_speculative_termination(
        &self,
        generated_tokens: &[TokenId],
        generated_text: &str,
        generated_logprobs: &Option<Vec<crate::config::TokenLogprob>>,
        token_count: usize,
        options: &GenerateOptions,
        max_context: Option<usize>,
        prefix_cache_hit_len: usize,
    ) -> anyhow::Result<Option<GenerateResult>> {
        if generated_tokens.len() >= options.max_new_tokens {
            ensure_constrained_finish(options, generated_text, FinishReason::MaxTokens)?;
            return Ok(Some(self.finish_result(
                generated_tokens,
                FinishReason::MaxTokens,
                prefix_cache_hit_len,
                generated_logprobs.as_deref(),
            )?));
        }
        if reached_context_limit(token_count, max_context) {
            ensure_constrained_finish(options, generated_text, FinishReason::Length)?;
            return Ok(Some(self.finish_result(
                generated_tokens,
                FinishReason::Length,
                prefix_cache_hit_len,
                generated_logprobs.as_deref(),
            )?));
        }
        Ok(None)
    }

    /// Run the target model forward for the current base position, returning its
    /// next-token logits, any proposer hidden state(s), and greedy next token.
    fn load_target_prediction(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
        speculative_mode: &SpeculativeMode,
    ) -> anyhow::Result<TargetPrediction> {
        let (base_logits, target_hidden, target_hidden_layers) =
            if let SpeculativeMode::Mtp(_) = speculative_mode {
                let hidden_output = self
                    .mtp
                    .as_ref()
                    .context("MTP speculation requested without a loaded MTP head")?
                    .hidden_output
                    .clone();
                let (logits, hidden) = next_session_token_logits_and_hidden(
                    self.session
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!(MISSING_ORT_SESSION))?,
                    self.kv_model.as_ref(),
                    &mut self.kv_cache,
                    session_id,
                    state,
                    &hidden_output,
                )?;
                (logits, Some(hidden), None)
            } else if let SpeculativeMode::Eagle3(_) = speculative_mode {
                let hidden_outputs = self
                    .eagle3
                    .as_ref()
                    .context("EAGLE-3 speculation requested without a loaded EAGLE-3 head")?
                    .hidden_outputs
                    .clone();
                let (logits, layers) = next_session_token_logits_and_hiddens(
                    self.session
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!(MISSING_ORT_SESSION))?,
                    self.kv_model.as_ref(),
                    &mut self.kv_cache,
                    session_id,
                    state,
                    &hidden_outputs,
                )?;
                let last_hidden = layers
                    .last()
                    .cloned()
                    .context("EAGLE-3 target hidden-state list was empty")?;
                (logits, Some(last_hidden), Some(layers))
            } else {
                (
                    next_session_token_logits(
                        self.session
                            .as_deref()
                            .ok_or_else(|| anyhow::anyhow!(MISSING_ORT_SESSION))?,
                        self.kv_model.as_ref(),
                        &mut self.kv_cache,
                        session_id,
                        state,
                    )?,
                    None,
                    None,
                )
            };
        let guaranteed_token = target_hidden
            .as_ref()
            .map(|_| argmax(&base_logits).context("target logits were empty"))
            .transpose()?
            .map(TokenId::try_from)
            .transpose()
            .context("target token id exceeds u32 range")?;
        Ok(TargetPrediction {
            base_logits,
            target_hidden,
            target_hidden_layers,
            guaranteed_token,
        })
    }

    /// Invoke the active proposer to produce this step's draft tokens.
    fn propose_candidates(
        &mut self,
        speculative_mode: &SpeculativeMode,
        state: &mut EngineSession,
        mtp_proposer: &mut Option<MtpProposer<'_, MtpEmbedder, MtpLmHead>>,
        rng: &mut SamplingRng,
        inputs: CandidateProposalInputs<'_>,
    ) -> anyhow::Result<Vec<TokenId>> {
        let proposer_context = SpeculativeProposerContext {
            width: inputs.width,
            context_tokens: &state.tokens,
            generated_tokens: inputs.generated_tokens,
            generated_text: inputs.generated_text,
            first_step: inputs.step,
            options: inputs.options,
            chain: inputs.chain,
            target_hidden: inputs.prediction.target_hidden.as_deref(),
            target_hidden_layers: inputs.prediction.target_hidden_layers.as_deref(),
            guaranteed_token: inputs.prediction.guaranteed_token,
            shared_kv_slices: inputs.shared_kv_slices,
        };
        let draft_tokens = match speculative_mode {
            SpeculativeMode::None => Vec::new(),
            SpeculativeMode::DraftModel => {
                let draft_model = self
                    .draft
                    .as_mut()
                    .context("speculative decoding requested without a draft model")?;
                let draft_state = state
                    .draft
                    .as_mut()
                    .context("speculative session missing draft state")?;
                let mut proposer = DraftModelProposer::with_rng(draft_model, draft_state, rng);
                proposer.align_to_target_prefix(&state.tokens, inputs.base_len)?;
                proposer.propose(&proposer_context)?.tokens
            }
            SpeculativeMode::PromptLookup { ngram, max_tokens } => {
                NgramProposer::new(*ngram, *max_tokens)?
                    .propose(&proposer_context)?
                    .tokens
            }
            SpeculativeMode::Mtp(_) => {
                mtp_proposer
                    .as_mut()
                    .context("MTP proposer state was not initialized")?
                    .propose(&proposer_context)?
                    .tokens
            }
            SpeculativeMode::Eagle3(_) => {
                let eagle3 = self
                    .eagle3
                    .as_ref()
                    .context("EAGLE-3 speculation requested without a loaded EAGLE-3 head")?;
                let mut proposer = Eagle3Proposer::new(
                    &eagle3.session,
                    Eagle3DecodeOptions {
                        kv_mode: eagle3.kv_mode,
                        batch_size: 1,
                    },
                    eagle3.embedder.clone(),
                )?;
                if let Some(token_map) = &eagle3.token_map {
                    proposer = proposer.with_token_map(token_map.clone())?;
                }
                proposer.propose(&proposer_context)?.tokens
            }
        };
        Ok(draft_tokens)
    }

    /// Append the draft tokens to the target sequence and run the target
    /// verification forward, returning one logits row per proposed token.
    fn run_target_verification(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
        draft_tokens: &[TokenId],
        base_len: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        self.last_speculative_stats.verification_steps += 1;
        self.last_speculative_stats.proposed_tokens += draft_tokens.len();

        state.tokens.extend_from_slice(draft_tokens);
        let verified_logits = if draft_tokens.is_empty() {
            Vec::new()
        } else if state.decode_state.has_runner() {
            let logits =
                run_decode_session_logits(&mut state.decode_state, draft_tokens, base_len)?;
            self.kv_cache
                .append(session_id, draft_tokens.len())
                .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {session_id}: {e}"))?;
            state.kv_token_count += draft_tokens.len();
            logits
        } else {
            let retained_base_len = state.decode_state.retained_kv_len(base_len);
            let outputs = run_decode_step(
                self.session
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!(MISSING_ORT_SESSION))?,
                &mut state.decode_state,
                draft_tokens,
                base_len,
            )?;
            if state.decode_state.use_kv {
                if let Some(kv_model) = &self.kv_model {
                    mirror_present_kv_to_pages(
                        self.session
                            .as_deref()
                            .ok_or_else(|| anyhow::anyhow!(MISSING_ORT_SESSION))?,
                        kv_model,
                        &mut self.kv_cache,
                        session_id,
                        &outputs,
                        retained_base_len,
                        draft_tokens.len(),
                    )?;
                } else {
                    self.kv_cache
                        .append(session_id, draft_tokens.len())
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to advance KV sequence {session_id}: {e}")
                        })?;
                }
                state.kv_token_count += draft_tokens.len();
                apply_paged_sliding_window(
                    &mut self.kv_cache,
                    session_id,
                    state.decode_state.sliding_window(),
                    state.decode_state.sink_tokens(),
                )?;
            }
            extract_logits_sequence_with_io(
                self.session
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!(MISSING_ORT_SESSION))?,
                outputs,
                state.decode_state.io.logits_output.as_deref(),
            )?
        };
        Ok(verified_logits)
    }

    /// Commit a hybrid native target's recurrent/conv state to the accepted
    /// prefix after a verify window advanced it over every draft token.
    ///
    /// Gated-DeltaNet SSM + conv1d state is a destructive rolling cache with no
    /// per-step history to prefix-slice, so — following vLLM — the committed
    /// state is rebuilt from the pre-draft `snapshot`: attention KV is
    /// prefix-sliced back to `base_len` and the recurrent/conv bindings are
    /// restored, then exactly `accepted_tokens` are re-run so the state equals
    /// feeding only the accepted continuation from the snapshot. The re-run also
    /// leaves the native attention KV at `base_len + accepted`, so this squares
    /// the paged length bookkeeping the runner mirrors, letting the ordinary
    /// rewind in [`Self::assemble_commit_tokens`] become a no-op.
    #[cfg(feature = "native-backend")]
    fn commit_native_recurrent_target(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
        base_len: usize,
        accepted_tokens: &[TokenId],
        snapshot: &crate::native_decode::RecurrentStateSnapshot,
    ) -> anyhow::Result<()> {
        {
            let native = state
                .decode_state
                .native_recurrent_runner_mut()
                .context("native recurrent target is no longer available for commit")?;
            native.commit_recurrent_state_to_accepted(snapshot, base_len, accepted_tokens)?;
        }
        let committed = base_len + accepted_tokens.len();
        self.kv_cache
            .rewind_to(session_id, committed)
            .map_err(|e| {
                anyhow::anyhow!("Failed to rewind KV sequence {session_id} to {committed}: {e}")
            })?;
        state.kv_token_count = committed;
        state.tokens.truncate(committed);
        Ok(())
    }

    /// Select the target token for each proposed position and count the longest
    /// accepted prefix, recording the correction token at the first mismatch.
    #[allow(clippy::too_many_arguments)]
    fn choose_accepted_prefix(
        &mut self,
        target_logits: &mut [Vec<f32>],
        draft_tokens: &[TokenId],
        base_len: usize,
        state: &EngineSession,
        generated_tokens: &[TokenId],
        options: &GenerateOptions,
        chain: &ProcessorChain,
        rng: &mut SamplingRng,
        step: usize,
    ) -> anyhow::Result<AcceptedPrefix> {
        let mut accepted = 0;
        let mut replacement = None;
        let mut candidate_logprobs = options.top_logprobs.map(|_| Vec::new());
        for idx in 0..draft_tokens.len() {
            let mut context = ProcessorContext {
                prompt_tokens: state.tokens[..base_len].to_vec(),
                generated_tokens: generated_tokens
                    .iter()
                    .copied()
                    .chain(draft_tokens[..idx].iter().copied())
                    .collect(),
                generated_text: self
                    .tokenizer
                    .decode(
                        &generated_tokens
                            .iter()
                            .copied()
                            .chain(draft_tokens[..idx].iter().copied())
                            .collect::<Vec<_>>(),
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to detokenize speculative context: {e}")
                    })?,
                step: step + idx,
            };
            let target_token =
                select_next_token_with_rng(&mut target_logits[idx], &context, options, chain, rng);
            if let (Some(top_logprobs), Some(logprobs)) =
                (options.top_logprobs, candidate_logprobs.as_mut())
            {
                logprobs.push(logprob_for_token(
                    &target_logits[idx],
                    target_token,
                    top_logprobs,
                ));
            }
            if target_token == draft_tokens[idx] {
                accepted += 1;
            } else {
                replacement = Some(target_token);
                context.generated_tokens.push(target_token);
                break;
            }
        }
        self.last_speculative_stats.accepted_tokens += accepted;
        if accepted >= 2 {
            self.last_speculative_stats.multi_token_accepts += 1;
        }
        Ok(AcceptedPrefix {
            accepted,
            replacement,
            candidate_logprobs,
        })
    }

    /// Build the committed token list for this step. This rewinds the target KV
    /// to the accepted prefix (base_len + accepted) BEFORE selecting any
    /// correction or bonus token, so downstream commit pushes resume from
    /// exactly the accepted boundary. Do not reorder the rewind relative to the
    /// correction-token selection below.
    #[allow(clippy::too_many_arguments)]
    fn assemble_commit_tokens(
        &mut self,
        accepted_prefix: &AcceptedPrefix,
        draft_tokens: &[TokenId],
        base_len: usize,
        state: &mut EngineSession,
        target_logits: &mut [Vec<f32>],
        generated_tokens: &[TokenId],
        options: &GenerateOptions,
        chain: &ProcessorChain,
        rng: &mut SamplingRng,
        step: usize,
        max_context: Option<usize>,
        session_id: SessionId,
    ) -> anyhow::Result<(Vec<TokenId>, Option<Vec<crate::config::TokenLogprob>>)> {
        let accepted = accepted_prefix.accepted;
        let candidate_logprobs = accepted_prefix.candidate_logprobs.as_ref();

        let mut commit_tokens = draft_tokens[..accepted].to_vec();
        let mut commit_logprobs = candidate_logprobs.map(|logprobs| logprobs[..accepted].to_vec());
        let rewind_len = base_len + accepted;
        rewind_target_state_to_len(
            self.session
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!(MISSING_ORT_SESSION))?,
            self.kv_model.as_ref(),
            &mut self.kv_cache,
            session_id,
            state,
            RewindRequest::new(rewind_len, RewindRunnerPolicy::AllowRunnerRewind),
        )?;

        if let Some(token) = accepted_prefix.replacement {
            commit_tokens.push(token);
            if let (Some(source), Some(commit)) = (candidate_logprobs, commit_logprobs.as_mut()) {
                commit.push(source[accepted].clone());
            }
        } else if generated_tokens.len() + commit_tokens.len() < options.max_new_tokens
            && !reached_context_limit(base_len + commit_tokens.len(), max_context)
        {
            let mut context = ProcessorContext {
                prompt_tokens: state.tokens[..base_len].to_vec(),
                generated_tokens: generated_tokens
                    .iter()
                    .copied()
                    .chain(draft_tokens.iter().copied())
                    .collect(),
                generated_text: self
                    .tokenizer
                    .decode(
                        &generated_tokens
                            .iter()
                            .copied()
                            .chain(draft_tokens.iter().copied())
                            .collect::<Vec<_>>(),
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to detokenize speculative context: {e}")
                    })?,
                step: step + draft_tokens.len(),
            };
            let token = select_next_token_with_rng(
                target_logits
                    .last_mut()
                    .context("target verification did not produce next-token logits")?,
                &context,
                options,
                chain,
                rng,
            );
            if let (Some(top_logprobs), Some(logprobs)) =
                (options.top_logprobs, commit_logprobs.as_mut())
            {
                logprobs.push(logprob_for_token(
                    target_logits
                        .last()
                        .context("target verification did not produce next-token logits")?,
                    token,
                    top_logprobs,
                ));
            }
            context.generated_tokens.push(token);
            commit_tokens.push(token);
        }
        Ok((commit_tokens, commit_logprobs))
    }

    /// Commit the accepted (and any correction/bonus) tokens: push not-yet-seen
    /// tokens onto the target sequence, emit the per-token callback, and advance
    /// counters. On a finish signal, trim the over-materialized target KV, sync
    /// the draft state, and return the terminal [`GenerateResult`].
    #[allow(clippy::too_many_arguments)]
    fn commit_speculative_tokens(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
        generated_tokens: &mut Vec<TokenId>,
        generated_text: &mut String,
        generated_logprobs: &mut Option<Vec<crate::config::TokenLogprob>>,
        commit_tokens: Vec<TokenId>,
        commit_logprobs: Option<Vec<crate::config::TokenLogprob>>,
        accepted: usize,
        base_len: usize,
        prefix_cache_hit_len: usize,
        options: &GenerateOptions,
        chain: &ProcessorChain,
        speculative_mode: &SpeculativeMode,
        step: &mut usize,
        callback: &mut Option<&mut GenerateTokenCallback<'_>>,
        max_context: Option<usize>,
    ) -> anyhow::Result<Option<GenerateResult>> {
        for (commit_idx, token_id) in commit_tokens.into_iter().enumerate() {
            if generated_tokens.len() >= options.max_new_tokens
                || (commit_idx >= accepted
                    && reached_context_limit(state.tokens.len(), max_context))
            {
                break;
            }
            if commit_idx >= accepted {
                state.tokens.push(token_id);
            }
            self.scheduler.advance(session_id);
            let prompt_tokens = state.tokens[..base_len.min(state.tokens.len())].to_vec();
            let mut commit_state = DecodeLoopState {
                generated_tokens: std::mem::take(generated_tokens),
                generated_text: std::mem::take(generated_text),
                step: *step,
                prefix_cache_hit_len,
                logprobs: generated_logprobs.take(),
                rng: SamplingRng::new(options.seed),
                custom_sampler: None,
            };
            if let (Some(all_logprobs), Some(step_logprobs)) =
                (commit_state.logprobs.as_mut(), commit_logprobs.as_ref())
            {
                all_logprobs.push(step_logprobs[commit_idx].clone());
            }
            let finish_reason = commit_selected_token(
                &mut commit_state,
                &prompt_tokens,
                token_id,
                options,
                chain,
                &self.tokenizer,
                callback.as_deref_mut(),
            )?;
            *generated_tokens = commit_state.generated_tokens;
            *generated_text = commit_state.generated_text;
            *generated_logprobs = commit_state.logprobs;
            *step = commit_state.step;
            if let Some(finish_reason) = finish_reason {
                trim_overmaterialized_target_kv(
                    self.session
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!(MISSING_ORT_SESSION))?,
                    self.kv_model.as_ref(),
                    &mut self.kv_cache,
                    session_id,
                    state,
                    RewindRunnerPolicy::AllowRunnerRewind,
                )?;
                if matches!(speculative_mode, SpeculativeMode::DraftModel) {
                    self.sync_draft_to_target(state)?;
                }
                return Ok(Some(self.finish_result(
                    generated_tokens,
                    finish_reason,
                    prefix_cache_hit_len,
                    generated_logprobs.as_deref(),
                )?));
            }
        }
        Ok(None)
    }

    pub(crate) fn sync_draft_to_target(&mut self, state: &mut EngineSession) -> anyhow::Result<()> {
        if let (Some(draft_model), Some(draft_state)) = (&mut self.draft, &mut state.draft) {
            DraftModelProposer::new(draft_model, draft_state).rewind(&state.tokens)?;
        }
        Ok(())
    }

    fn notify_draft_acceptance(
        &mut self,
        state: &mut EngineSession,
        accepted_prefix_len: usize,
        committed_tokens: &[TokenId],
    ) -> anyhow::Result<()> {
        if let (Some(draft_model), Some(draft_state)) = (&mut self.draft, &mut state.draft) {
            DraftModelProposer::new(draft_model, draft_state).accept(
                &SpeculativeAcceptContext {
                    accepted_prefix_len,
                    committed_tokens,
                    target_tokens: &state.tokens,
                },
            )?;
        }
        Ok(())
    }
}

/// Whether every component may be silently swapped for an equivalent one.
///
/// An absent contract counts as `EquivalenceClass::Semantic` — the schema
/// default — rather than being skipped. Filtering undeclared components out
/// would make the `all` vacuously true for a package whose components declare no
/// contracts at all, which is precisely the package that promised nothing.
fn components_permit_automatic_speculation(
    components: &std::collections::BTreeMap<String, onnx_genai_metadata::WorkflowComponent>,
) -> bool {
    if components.is_empty() {
        return false;
    }
    components.values().all(|component| {
        component
            .contract
            .as_ref()
            .map(|contract| contract.equivalence)
            .unwrap_or_default()
            .permits_automatic_speculation()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_ort::{Environment, SessionOptions};
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    /// The int4 draft LM-head projection primitive must reproduce a `MatMulNBits`
    /// GEMV from the target's own quantised initializers, argmax-correctly, on
    /// the CPU int4 kernel — the GPU path shares this exact builder. A one-hot
    /// hidden vector isolates a single weight column: setting one column's lead
    /// nibble to the maximum code (15, i.e. +7 after the symmetric zero-point)
    /// while every other column stays at the zero code (8) makes that column the
    /// unique argmax, independent of the exact scale, proving the projection
    /// routes hidden→logits through the real quantised kernel rather than a stub.
    #[cfg(feature = "native-backend")]
    #[test]
    fn quantized_draft_lm_head_projects_int4_argmax() -> anyhow::Result<()> {
        use onnx_runtime_ir::{
            Attribute, DataType, Graph, Node, NodeId, TensorData, WeightRef, static_shape,
        };

        // K=32 (one block), N=4 columns, symmetric int4 (no zero_points input).
        let k = 32usize;
        let n = 4usize;
        let block_size = 32i64;
        let k_blocks = 1usize;
        let blob = 16usize; // block_size / 2, two int4 weights per byte.
        let target_col = 2usize;

        // Every packed byte is 0x88 → both nibbles are code 8 (zero after the
        // symmetric zero-point). For the target column, the first weight (k=0,
        // the low nibble of byte 0) is code 15 (+7).
        let mut weight = vec![0x88u8; n * k_blocks * blob];
        weight[target_col * k_blocks * blob] = 0x8F;
        let scales = vec![1.0f32; n * k_blocks];

        let mut graph = Graph::default();
        graph.opset_imports.insert("com.microsoft".to_string(), 1);
        let activation = graph.create_named_value("A", DataType::Float32, static_shape([1, k]));
        let weight_value = graph.create_named_value(
            "lm_head.weight",
            DataType::Uint8,
            static_shape([n, k_blocks, blob]),
        );
        graph.set_initializer(
            weight_value,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Uint8,
                vec![n, k_blocks, blob],
                weight,
            )),
        );
        let scales_value = graph.create_named_value(
            "lm_head.scales",
            DataType::Float32,
            static_shape([n, k_blocks]),
        );
        graph.set_initializer(
            scales_value,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![n, k_blocks],
                scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )),
        );
        let logits_value =
            graph.create_named_value("logits", DataType::Float32, static_shape([1, n]));
        let mut node = Node::new(
            NodeId(0),
            "MatMulNBits",
            vec![Some(activation), Some(weight_value), Some(scales_value)],
            vec![logits_value],
        );
        node.domain = "com.microsoft".to_string();
        node.attributes
            .insert("K".to_string(), Attribute::Int(k as i64));
        node.attributes
            .insert("N".to_string(), Attribute::Int(n as i64));
        node.attributes
            .insert("bits".to_string(), Attribute::Int(4));
        node.attributes
            .insert("block_size".to_string(), Attribute::Int(block_size));
        graph.insert_node(node);

        let (lm_head, vocab) = build_quantized_draft_lm_head(
            &graph,
            Arc::new(WeightStore::new()),
            "lm_head.weight",
            k,
            DraftProjectionDevice::Cpu,
        )?;
        assert_eq!(vocab, n);
        assert_eq!(lm_head.vocab_size(), n);

        // One-hot hidden at k=0 selects each column's lead weight.
        let mut hidden = vec![0.0f32; k];
        hidden[0] = 1.0;
        let mut logits = vec![0.0f32; n];
        lm_head.logits(&hidden, &mut logits)?;

        let best = argmax(&logits).expect("argmax over non-empty logits");
        assert_eq!(
            best, target_col,
            "int4 draft projection argmax {best} != expected column {target_col}; logits {logits:?}"
        );
        for (col, &value) in logits.iter().enumerate() {
            if col != target_col {
                assert!(
                    logits[target_col] > value,
                    "target column logit {} not strictly greatest vs col {col} = {value}",
                    logits[target_col]
                );
            }
        }
        Ok(())
    }

    struct StubProposer {
        tokens: Vec<TokenId>,
        accepted: Option<usize>,
        rewound_to: Option<Vec<TokenId>>,
    }

    impl SpeculativeProposer for StubProposer {
        fn propose(
            &mut self,
            _context: &SpeculativeProposerContext<'_>,
        ) -> anyhow::Result<SpeculativeProposal> {
            Ok(SpeculativeProposal::linear(self.tokens.clone()))
        }

        fn accept(&mut self, context: &SpeculativeAcceptContext<'_>) -> anyhow::Result<()> {
            self.accepted = Some(context.accepted_prefix_len);
            Ok(())
        }

        fn rewind(&mut self, target_tokens: &[TokenId]) -> anyhow::Result<()> {
            self.rewound_to = Some(target_tokens.to_vec());
            Ok(())
        }

        fn name(&self) -> &str {
            "stub"
        }
    }

    #[test]
    fn speculative_proposer_trait_supports_non_draft_sources() -> anyhow::Result<()> {
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let mut proposer = StubProposer {
            tokens: vec![3, 5],
            accepted: None,
            rewound_to: None,
        };

        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 2,
            context_tokens: &[1],
            generated_tokens: &[1],
            generated_text: "a",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: None,
            target_hidden_layers: None,
            guaranteed_token: None,
            shared_kv_slices: None,
        })?;
        proposer.accept(&SpeculativeAcceptContext {
            accepted_prefix_len: 1,
            committed_tokens: &[3, 4],
            target_tokens: &[1, 3, 4],
        })?;
        proposer.rewind(&[1, 3, 4])?;

        assert_eq!(proposal.tokens, vec![3, 5]);
        assert_eq!(proposer.accepted, Some(1));
        assert_eq!(proposer.rewound_to, Some(vec![1, 3, 4]));
        Ok(())
    }

    #[test]
    fn ngram_proposer_copies_most_recent_matching_continuation() -> anyhow::Result<()> {
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let mut proposer = NgramProposer::new(2, 4)?;
        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 3,
            context_tokens: &[7, 8, 9, 4, 7, 8],
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: None,
            target_hidden_layers: None,
            guaranteed_token: None,
            shared_kv_slices: None,
        })?;

        assert_eq!(proposal.tokens, vec![9, 4, 7]);
        Ok(())
    }

    #[test]
    fn ngram_proposer_validates_configuration_and_empty_matches() -> anyhow::Result<()> {
        assert_eq!(
            NgramProposer::new(0, 1).unwrap_err().to_string(),
            "ngram must be greater than zero"
        );
        assert_eq!(
            NgramProposer::new(1, 0).unwrap_err().to_string(),
            "max_tokens must be greater than zero"
        );

        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let context = |tokens| SpeculativeProposerContext {
            width: 4,
            context_tokens: tokens,
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: None,
            target_hidden_layers: None,
            guaranteed_token: None,
            shared_kv_slices: None,
        };
        let mut proposer = NgramProposer::new(2, 4)?;

        assert!(proposer.propose(&context(&[1, 2]))?.tokens.is_empty());
        assert!(proposer.propose(&context(&[1, 2, 3, 4]))?.tokens.is_empty());
        assert_eq!(proposer.name(), "prompt_lookup");
        Ok(())
    }

    #[test]
    fn ngram_proposer_respects_request_and_config_widths() -> anyhow::Result<()> {
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let tokens = [1, 2, 3, 4, 1, 2];
        let mut proposer = NgramProposer::new(2, 2)?;
        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 8,
            context_tokens: &tokens,
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: None,
            target_hidden_layers: None,
            guaranteed_token: None,
            shared_kv_slices: None,
        })?;
        assert_eq!(proposal.tokens, vec![3, 4]);

        let mut proposer = NgramProposer::new(2, 8)?;
        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 1,
            context_tokens: &tokens,
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: None,
            target_hidden_layers: None,
            guaranteed_token: None,
            shared_kv_slices: None,
        })?;
        assert_eq!(proposal.tokens, vec![3]);
        Ok(())
    }

    fn lcg_weights(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 33) as u32;
                (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn eagle3_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn load_eagle3_head() -> anyhow::Result<Session> {
        static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
        let environment = ENVIRONMENT
            .get_or_init(|| Environment::new("engine-eagle3-test").expect("environment"));
        let head_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-eagle3/model.onnx.textproto");
        Ok(Session::new(
            environment,
            &head_path,
            SessionOptions::default().with_intra_op_threads(1),
        )?)
    }

    #[test]
    fn mtp_proposer_uses_real_head_and_returns_guaranteed_plus_k_drafts() -> anyhow::Result<()> {
        const HIDDEN: usize = 16;
        const VOCAB: usize = 32;
        static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
        let environment =
            ENVIRONMENT.get_or_init(|| Environment::new("engine-mtp-test").expect("environment"));
        let head_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen35-mtp/model.onnx.textproto");
        let head = Session::new(
            environment,
            &head_path,
            SessionOptions::default().with_intra_op_threads(1),
        )?;
        let embedder =
            LinearEmbedder::new(lcg_weights(0x1111_2222, VOCAB * HIDDEN), VOCAB, HIDDEN)?;
        let lm_head = LinearLmHead::new(lcg_weights(0x3333_4444, HIDDEN * VOCAB), HIDDEN, VOCAB)?;
        let hidden = lcg_weights(0xA5A5_1234, HIDDEN);
        let mut logits = vec![0.0; VOCAB];
        LmHead::logits(&lm_head, &hidden, &mut logits)?;
        let guaranteed = argmax(&logits).context("target logits were empty")? as TokenId;
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let mut proposer = MtpProposer::new(&head, MtpDecodeOptions::default(), embedder, lm_head)?;

        fn assert_speculative_proposer<T: SpeculativeProposer>(_proposer: &T) {}
        assert_speculative_proposer(&proposer);
        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 5,
            context_tokens: &[1],
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: Some(&hidden),
            target_hidden_layers: None,
            guaranteed_token: Some(guaranteed),
            shared_kv_slices: None,
        })?;

        assert_eq!(proposer.name(), "mtp");
        assert_eq!(guaranteed, 13);
        assert_eq!(proposal.tokens.len(), 5);
        assert_eq!(proposal.tokens.first(), Some(&guaranteed));
        assert_eq!(proposal.tokens, vec![guaranteed, 27, 11, 2, 27]);
        Ok(())
    }

    #[derive(Clone)]
    struct ConstantEmbedder;

    impl TokenEmbedder for ConstantEmbedder {
        fn hidden_size(&self) -> usize {
            2
        }

        fn vocab_size(&self) -> usize {
            usize::MAX
        }

        fn embed(&self, _token: TokenId, out: &mut [f32]) -> anyhow::Result<()> {
            out.copy_from_slice(&[1.0, 0.0]);
            Ok(())
        }
    }

    #[derive(Clone)]
    struct RecordingLmHead {
        inputs: Arc<Mutex<Vec<Vec<f32>>>>,
    }

    impl LmHead for RecordingLmHead {
        fn vocab_size(&self) -> usize {
            1
        }

        fn logits(&self, hidden: &[f32], out: &mut [f32]) -> anyhow::Result<()> {
            self.inputs
                .lock()
                .map_err(|_| anyhow::anyhow!("recording LM-head lock poisoned"))?
                .push(hidden.to_vec());
            out[0] = 1.0;
            Ok(())
        }
    }

    #[test]
    fn mtp_proposer_threads_rank4_hc_state_across_drafts() -> anyhow::Result<()> {
        static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
        let environment = ENVIRONMENT
            .get_or_init(|| Environment::new("engine-mtp-hc-test").expect("environment"));
        let head_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny-hc-mtp/model.onnx.textproto");
        let head = Session::new(
            environment,
            &head_path,
            SessionOptions::default().with_intra_op_threads(1),
        )?;
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let mut proposer = MtpProposer::new(
            &head,
            MtpDecodeOptions {
                kv_mode: onnx_genai_ort::MtpDraftKvMode::HiddenThreaded,
                batch_size: 1,
                hc_mult: 2,
                hidden_state_rank4: true,
                hidden_output: "mtp_hidden".into(),
                state_output: Some("mtp_state".into()),
            },
            ConstantEmbedder,
            RecordingLmHead {
                inputs: Arc::clone(&recorded),
            },
        )?;
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let target_hc = vec![0.0; 4];

        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 3,
            context_tokens: &[4, 5],
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: Some(&target_hc),
            target_hidden_layers: None,
            guaranteed_token: Some(0),
            shared_kv_slices: None,
        })?;

        assert_eq!(proposal.tokens, vec![0, 0, 0]);
        assert_eq!(
            *recorded
                .lock()
                .map_err(|_| anyhow::anyhow!("recording LM-head lock poisoned"))?,
            vec![vec![2.0, 0.0], vec![4.0, 0.0]]
        );
        Ok(())
    }

    #[test]
    fn mtp_package_references_borrow_target_initializers() -> anyhow::Result<()> {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-mtp-full");
        let (embedder, lm_head, vocab_size) = load_target_initializer_adapters(
            &fixture.join("model.onnx.textproto"),
            "transformer.wte.weight",
            "lm_head.weight_t",
            16,
            None,
        )?;
        assert_eq!(vocab_size, 32);

        let mut embedded = vec![0.0; 16];
        embedder.embed(7, &mut embedded)?;
        let raw_embedding = std::fs::read(fixture.join("embedding.f32"))?;
        let expected_embedding = raw_embedding
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        assert_eq!(embedded, expected_embedding[7 * 16..8 * 16]);

        let hidden = lcg_weights(0xDEAD_BEEF, 16);
        let mut package_logits = vec![0.0; 32];
        lm_head.logits(&hidden, &mut package_logits)?;
        let raw_lm_head = std::fs::read(fixture.join("lm_head.f32"))?;
        let linear_lm_head = LinearLmHead::new(
            raw_lm_head
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect(),
            16,
            32,
        )?;
        let mut expected_logits = vec![0.0; 32];
        linear_lm_head.logits(&hidden, &mut expected_logits)?;
        assert_eq!(package_logits, expected_logits);
        Ok(())
    }

    #[test]
    fn eagle3_proposer_loads_fixture_and_returns_shape_correct_proposal() -> anyhow::Result<()> {
        const HIDDEN: usize = 16;
        const VOCAB: usize = 32;
        let _guard = eagle3_test_lock()
            .lock()
            .map_err(|_| anyhow::anyhow!("EAGLE-3 test lock poisoned"))?;
        let head = load_eagle3_head()?;
        let embedder =
            LinearEmbedder::new(lcg_weights(0x5555_6666, VOCAB * HIDDEN), VOCAB, HIDDEN)?;
        let layers = vec![
            lcg_weights(0x1000_0001, HIDDEN),
            lcg_weights(0x2000_0002, HIDDEN),
            lcg_weights(0x3000_0003, HIDDEN),
        ];
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let mut proposer = Eagle3Proposer::new(&head, Eagle3DecodeOptions::default(), embedder)?;
        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 4,
            context_tokens: &[1, 2],
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: Some(&layers[2]),
            target_hidden_layers: Some(&layers),
            guaranteed_token: Some(7),
            shared_kv_slices: None,
        })?;

        assert_eq!(proposer.name(), "eagle3");
        assert_eq!(proposal.tokens.len(), 4);
        assert_eq!(proposal.tokens.first(), Some(&7));
        assert!(proposal.tokens.iter().all(|&token| token < VOCAB as u32));
        assert!(proposal.positions.is_none());
        assert!(proposal.tree.is_none());
        Ok(())
    }

    #[test]
    fn chained_proposer_applies_declared_vocabulary_mapping() -> anyhow::Result<()> {
        const HIDDEN: usize = 16;
        const VOCAB: usize = 32;
        let _guard = eagle3_test_lock()
            .lock()
            .map_err(|_| anyhow::anyhow!("chained proposer test lock poisoned"))?;
        let head = load_eagle3_head()?;
        let layers = vec![
            lcg_weights(0x1000_0001, HIDDEN),
            lcg_weights(0x2000_0002, HIDDEN),
            lcg_weights(0x3000_0003, HIDDEN),
        ];
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let context = SpeculativeProposerContext {
            width: 2,
            context_tokens: &[1, 2],
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: Some(&layers[2]),
            target_hidden_layers: Some(&layers),
            guaranteed_token: Some(7),
            shared_kv_slices: None,
        };
        let weights = lcg_weights(0x5555_6666, VOCAB * HIDDEN);
        let mut plain = Eagle3Proposer::new(
            &head,
            Eagle3DecodeOptions::default(),
            LinearEmbedder::new(weights.clone(), VOCAB, HIDDEN)?,
        )?;
        let plain = plain.propose(&context)?;
        let map = (0..VOCAB as TokenId)
            .map(|token| token ^ 1)
            .collect::<Vec<_>>();
        let mut mapped = Eagle3Proposer::new(
            &head,
            Eagle3DecodeOptions::default(),
            LinearEmbedder::new(weights, VOCAB, HIDDEN)?,
        )?
        .with_token_map(map.clone())?;
        let mapped = mapped.propose(&context)?;
        assert_eq!(mapped.tokens[0], plain.tokens[0]);
        assert_eq!(mapped.tokens[1], map[plain.tokens[1] as usize]);
        Ok(())
    }

    /// Load-closed vocabulary bound: a token-map entry that indexes at or past
    /// the target/embedder vocabulary has no embedding row and could never be a
    /// real target token, so the map is rejected when it is installed — not
    /// mid-proposal on whichever draft id happens to select it.
    #[test]
    fn eagle3_token_map_rejects_ids_beyond_target_vocab() -> anyhow::Result<()> {
        const HIDDEN: usize = 16;
        const VOCAB: usize = 32;
        let _guard = eagle3_test_lock()
            .lock()
            .map_err(|_| anyhow::anyhow!("eagle3 token-map test lock poisoned"))?;
        let head = load_eagle3_head()?;
        let weights = lcg_weights(0x2468_1357, VOCAB * HIDDEN);
        // A full-length map (so it clears the length check) whose last entry
        // indexes exactly at the vocabulary size — one past the last valid id.
        let mut out_of_range = (0..VOCAB as TokenId).collect::<Vec<_>>();
        *out_of_range.last_mut().expect("non-empty map") = VOCAB as TokenId;
        let error = Eagle3Proposer::new(
            &head,
            Eagle3DecodeOptions::default(),
            LinearEmbedder::new(weights, VOCAB, HIDDEN)?,
        )?
        .with_token_map(out_of_range)
        .err()
        .expect("an out-of-range token map must be rejected at load time");
        let message = error.to_string();
        assert!(
            message.contains("target/embedder vocabulary") && message.contains(&VOCAB.to_string()),
            "error must name the vocabulary bound: {message}"
        );
        Ok(())
    }

    #[test]
    fn eagle3_proposer_accept_then_propose_resets_draft_state() -> anyhow::Result<()> {
        const HIDDEN: usize = 16;
        const VOCAB: usize = 32;
        let _guard = eagle3_test_lock()
            .lock()
            .map_err(|_| anyhow::anyhow!("EAGLE-3 test lock poisoned"))?;
        let head = load_eagle3_head()?;
        let embedder =
            LinearEmbedder::new(lcg_weights(0x7777_8888, VOCAB * HIDDEN), VOCAB, HIDDEN)?;
        let layers = vec![
            lcg_weights(0x4000_0004, HIDDEN),
            lcg_weights(0x5000_0005, HIDDEN),
            lcg_weights(0x6000_0006, HIDDEN),
        ];
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let context = SpeculativeProposerContext {
            width: 3,
            context_tokens: &[4, 5, 6],
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: Some(&layers[2]),
            target_hidden_layers: Some(&layers),
            guaranteed_token: Some(9),
            shared_kv_slices: None,
        };
        let mut proposer = Eagle3Proposer::new(&head, Eagle3DecodeOptions::default(), embedder)?;

        let first = proposer.propose(&context)?;
        proposer.accept(&SpeculativeAcceptContext {
            accepted_prefix_len: 2,
            committed_tokens: &first.tokens[..2],
            target_tokens: &[4, 5, 6, first.tokens[0], first.tokens[1]],
        })?;
        let second = proposer.propose(&context)?;

        assert_eq!(first, second);
        Ok(())
    }

    #[test]
    fn mtp_mode_selects_mtp_proposer_contract() {
        let mode = SpeculativeMode::Mtp(crate::config::MtpConfig {
            head_model: "mtp.onnx".into(),
            target_hidden_output: "hidden_states".into(),
            embedding_weights: "embed.f32".into(),
            lm_head_weights: "lm_head.f32".into(),
            vocab_size: 32,
            hidden_size: 16,
            kv_mode: onnx_genai_ort::MtpDraftKvMode::HiddenThreaded,
            num_speculative_tokens: 4,
        });
        let selected = match mode {
            SpeculativeMode::Mtp(_) => "mtp",
            SpeculativeMode::Eagle3(_) => "eagle3",
            SpeculativeMode::DraftModel => "draft_model",
            SpeculativeMode::PromptLookup { .. } => "prompt_lookup",
            SpeculativeMode::None => "none",
        };
        assert_eq!(selected, "mtp");
    }

    #[test]
    fn eagle3_mode_selects_eagle3_proposer_contract() {
        let mode = SpeculativeMode::Eagle3(crate::config::Eagle3Config {
            head_model: "eagle3.onnx".into(),
            target_hidden_outputs: vec!["low".into(), "mid".into(), "high".into()],
            embedding_weights: "embed.f32".into(),
            token_map: None,
            vocab_size: 32,
            hidden_size: 16,
            kv_mode: onnx_genai_ort::Eagle3DraftKvMode::HiddenThreaded,
            num_speculative_tokens: 4,
        });
        let selected = match mode {
            SpeculativeMode::Eagle3(_) => "eagle3",
            SpeculativeMode::Mtp(_) => "mtp",
            SpeculativeMode::DraftModel => "draft_model",
            SpeculativeMode::PromptLookup { .. } => "prompt_lookup",
            SpeculativeMode::None => "none",
        };
        assert_eq!(selected, "eagle3");
    }
}

#[cfg(test)]
mod equivalence_gate_tests {
    use super::components_permit_automatic_speculation;
    use onnx_genai_metadata::WorkflowComponent;
    use std::collections::BTreeMap;

    fn component(equivalence: Option<&str>) -> WorkflowComponent {
        let contract = equivalence
            .map(|class| {
                format!(
                    "\ncontract:\n  id: onnx-genai.decoder\n  version: \"1\"\n  \
                     equivalence: {class}\n"
                )
            })
            .unwrap_or_default();
        serde_yaml::from_str(&format!(
            "implementation: {{ kind: onnx, artifact: decoder.onnx }}\nports: {{}}{contract}"
        ))
        .expect("component parses")
    }

    fn gate(components: &[(&str, Option<&str>)]) -> bool {
        let map = components
            .iter()
            .map(|(name, class)| ((*name).to_string(), component(*class)))
            .collect::<BTreeMap<_, _>>();
        components_permit_automatic_speculation(&map)
    }

    /// Silence must never buy an automatic optimization. A package whose
    /// components declare no contract at all promised nothing, so it must be
    /// read as `semantic` — not skipped, which would make `all` vacuously true.
    #[test]
    fn a_package_that_declares_no_contracts_permits_nothing() {
        assert!(!gate(&[("decoder", None)]));
        assert!(!gate(&[("decoder", None), ("draft", None)]));
        assert!(!gate(&[]), "an empty package promised nothing either");
    }

    /// One undeclared component is enough to withhold consent, even when every
    /// component that did declare one is distribution-preserving.
    #[test]
    fn one_undeclared_component_withholds_consent() {
        assert!(gate(&[("decoder", Some("distribution_preserving"))]));
        assert!(!gate(&[
            ("decoder", Some("distribution_preserving")),
            ("draft", None),
        ]));
    }

    /// Only bitwise and distribution-preserving equivalence permit a silent
    /// swap; merely semantic equivalence is free to change the output
    /// distribution, which an unasking caller did not agree to.
    #[test]
    fn only_distribution_preserving_classes_permit_a_silent_swap() {
        assert!(gate(&[("decoder", Some("bitwise"))]));
        assert!(gate(&[("decoder", Some("distribution_preserving"))]));
        assert!(!gate(&[("decoder", Some("semantic"))]));
        assert!(!gate(&[
            ("decoder", Some("bitwise")),
            ("draft", Some("semantic")),
        ]));
    }
}
