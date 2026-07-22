//! Native nxrt adapter for the engine's existing decode loop.

use crate::config::{GenerateOptions, GenerateResult, GenerateTokenCallback};
use crate::decode::DecodeBackend;
use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, run_decode_loop};
use crate::logits::{ProcessorChain, TokenId};
use crate::sampling::sample_greedy;
use anyhow::{Context, bail};
use onnx_genai_metadata::{KvOwnership, ModelIoSpec, SequenceInputKind, SharedKvGroup};
use onnx_genai_ort::Tokenizer;
use onnx_runtime_ir::{DataType, DeviceType, Dim, SymbolId};
use onnx_runtime_session::{
    CaptureDeclineReport, DeviceAllocationCounts, DeviceBindingTransferStats,
    DeviceGraphCaptureResult, DeviceIoBinding, DevicePreference, InferenceSession, Tensor,
};
use onnx_runtime_tracer::{Args, TraceContext, capture_rejected};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

/// Device requested for a native decode session.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NativeDecodeDevice {
    #[default]
    Cpu,
    Cuda {
        index: Option<u32>,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeDecodeCudaOptions {
    pub kv_max_len: Option<usize>,
    pub graph_capture: Option<bool>,
}

/// Purely structural signals that gate whether whole-step CUDA graph capture is
/// *auto-attempted* on the native decode path. Never derived from a model or
/// architecture name (RULES.md §2/§2.1) — only from device placement and the
/// declared KV-ownership metadata. When these hold, per-step decode topology is
/// static and the KV cache is device-resident and owned, so a captured graph can
/// replay safely. The runtime decline machinery in `DecodeCudaState::new`
/// remains the final safety net: if a would-be capture still carries a dynamic
/// auxiliary seam it is transparently declined and decode continues eagerly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphCaptureStructuralSafety {
    /// Decode runs on a CUDA device (device-resident, replayable bindings).
    device_is_cuda: bool,
    /// KV cache is owned/device-resident (not a borrowed shared-KV proposer).
    kv_ownership: KvOwnership,
}

impl GraphCaptureStructuralSafety {
    /// True when structural conditions make whole-step capture safe to attempt.
    fn is_capture_safe(self) -> bool {
        self.device_is_cuda && self.kv_ownership == KvOwnership::Owned
    }
}

/// Resolve whether whole-step CUDA graph capture should be attempted for the
/// native decode path, honoring explicit overrides before the structural
/// auto-decision.
///
/// Precedence:
/// 1. Programmatic `NativeDecodeCudaOptions::graph_capture` (`Some`) always wins.
/// 2. An explicitly-set `ONNX_GENAI_CUDA_GRAPH` env var (`=0` forces OFF, `=1`
///    forces ON) is honored next.
/// 3. When neither is set, auto-decide from `structural` safety: attempt capture
///    only when the decode topology is structurally graph-safe.
fn resolve_graph_capture_enabled(
    programmatic: Option<bool>,
    env_explicit: bool,
    env_value: bool,
    structural: GraphCaptureStructuralSafety,
) -> bool {
    if let Some(explicit) = programmatic {
        return explicit;
    }
    if env_explicit {
        return env_value;
    }
    structural.is_capture_safe()
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum NativeGqaSequenceLengthsPolicy {
    #[default]
    PerBatchOnly,
    AllowUnitBatchScalar,
}

const DEFAULT_CUDA_KV_MAX_LEN: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaKvDebugStats {
    pub logical_len: usize,
    pub max_len: usize,
    pub device_ptrs: Vec<usize>,
    pub kv_transfers: DeviceBindingTransferStats,
    pub graph: CudaGraphDebugStats,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CudaGraphDebugStats {
    pub enabled: bool,
    pub captures: u64,
    pub replays: u64,
    pub fallbacks: u64,
    pub allocation_counts: DeviceAllocationCounts,
    /// Structured reasons from the most recent capture fallback.
    pub fallback_report: Option<CaptureDeclineReport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeCudaGraphPhase {
    NeedsWarmup,
    Armed,
    Ready,
    Unsupported,
}

struct DecodeCudaState {
    logical_len: usize,
    max_len: usize,
    bindings: Vec<DeviceIoBinding>,
    base_binding_count: usize,
    kv_binding_range: std::ops::Range<usize>,
    auxiliary_binding_range: std::ops::Range<usize>,
    input_ids_binding: usize,
    position_ids_binding: Option<usize>,
    logits_binding: usize,
    logits_shape: Vec<usize>,
    logits_dtype: DataType,
    greedy_result: DeviceIoBinding,
    graph_enabled: bool,
    graph_phase: DecodeCudaGraphPhase,
    graph_captures: u64,
    graph_replays: u64,
    graph_fallbacks: u64,
    graph_fallback_reason: Option<String>,
    graph_fallback_report: Option<CaptureDeclineReport>,
    /// Structural reasons, recorded at binding time, why one or more auxiliary
    /// graph outputs could not be persistently bound (an unresolved symbolic
    /// dimension that is not batch or query-seq). Non-empty here means CUDA
    /// graph capture was declined up front and the eager device path is in
    /// force for this generation. Empty when every auxiliary output was
    /// statically bindable.
    auxiliary_bind_declines: Vec<String>,
    /// When `false` (today's default), `NativeDecodeSession::rewind` invalidates
    /// the captured decode graph before rolling the device KV back — correct for
    /// the eager M=K verify path (option (b)), which captures nothing.
    ///
    /// When `true`, rewind performs a *contents-only* mutation (zero the mask
    /// tail + truncate the KV logical length) and **retains** the captured graph.
    /// This is the option (c) invariant: a single fixed-topology M=maxK graph
    /// whose device-binding pointers stay invariant while only buffer contents /
    /// logical shapes change across steps — exactly the data-driven mutation the
    /// captured graph already tolerates on the M=1 replay path. Kept dormant
    /// (default `false`) until WP4 graduates verify to the captured path.
    retain_graph_on_rewind: bool,
    /// Dormant option (c) scaffolding: the fixed query-row capacity (M=maxK) a
    /// padded single-capture verify graph would be captured at. `None` today —
    /// the eager verify path (option (b)) captures nothing. Set only by the
    /// dormant `configure_padded_verify_capture` switch (not on the hot path).
    #[allow(dead_code)]
    padded_query_capacity: Option<usize>,
}

struct DecodeCudaIo<'a> {
    input_ids: &'a str,
    attention_mask: &'a str,
    position_ids: Option<&'a str>,
    logits: &'a str,
}

fn trace_capture_declines(trace: &TraceContext, report: &CaptureDeclineReport) {
    for decline in &report.entries {
        if let Some(node_id) = decline.node_id {
            capture_rejected(
                trace,
                node_id,
                decline.op_type.as_str(),
                decline.domain.as_str(),
                decline.reason.as_str(),
            );
        }
    }
}

/// Stateful decoder-with-past adapter over the pure-Rust native runtime.
pub struct NativeDecodeSession {
    session: InferenceSession,
    input_ids: String,
    attention_mask: String,
    position_ids: Option<String>,
    logits: String,
    hidden_output: Option<String>,
    kv_inputs: Vec<String>,
    present_to_past: HashMap<String, String>,
    past: HashMap<String, Tensor>,
    cuda: Option<DecodeCudaState>,
    trace: TraceContext,
    current_len: usize,
    last_hidden: Option<Vec<f32>>,
}

impl NativeDecodeSession {
    /// Load a decoder-with-past ONNX model on the requested native device.
    pub fn load(path: impl AsRef<Path>, device: NativeDecodeDevice) -> anyhow::Result<Self> {
        Self::load_with_cuda_options(path, device, NativeDecodeCudaOptions::default())
    }
}

/// Semantic outputs of one native proposer forward.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NativeProposerOutput {
    pub logits: Option<Vec<Vec<f32>>>,
    pub projected_state: Option<Vec<f32>>,
}

/// Metadata-driven native execution adapter for speculative proposer graphs.
///
/// Unlike [`NativeDecodeSession`], this adapter accepts either token ids or
/// precomputed embeddings and supports both graph-owned past/present KV and
/// target-owned shared-KV inputs.
pub(crate) struct NativeProposerSession {
    session: InferenceSession,
    sequence_source: SequenceInputKind,
    sequence_input: String,
    attention_mask: Option<String>,
    position_ids: Option<String>,
    logits_output: Option<String>,
    projected_state_output: Option<String>,
    kv_ownership: KvOwnership,
    kv_inputs: Vec<String>,
    present_to_past: Vec<(String, String)>,
    past: HashMap<String, Tensor>,
    current_len: usize,
}

impl NativeProposerSession {
    pub(crate) fn load(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        let preference = match device {
            NativeDecodeDevice::Cpu => DevicePreference::Cpu,
            NativeDecodeDevice::Cuda { index } => DevicePreference::Gpu { index },
        };
        let session = InferenceSession::builder()
            .model(path)
            .device(preference)
            .build()
            .context("load native proposer model")?;
        Self::from_session(session, io)
    }

    pub(crate) fn from_session(
        session: InferenceSession,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        let input_names = session
            .inputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let output_names = session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let sequence_source = io
            .and_then(|io| io.sequence_source)
            .unwrap_or(SequenceInputKind::TokenIds);
        let sequence_input = match sequence_source {
            SequenceInputKind::TokenIds => declared_or_detected_input(
                &input_names,
                io.and_then(|io| io.token_input.as_deref()),
                &["input_ids", "decoder_input_ids"],
                "token_input",
            )?,
            SequenceInputKind::InputsEmbeds => declared_or_detected_input(
                &input_names,
                io.and_then(|io| io.inputs_embeds_input.as_deref()),
                &["inputs_embeds"],
                "inputs_embeds_input",
            )?,
        };
        let attention_mask = optional_declared_or_detected_input(
            &input_names,
            io.and_then(|io| io.attention_mask_input.as_deref()),
            &["attention_mask"],
            "attention_mask_input",
        )?;
        let position_ids = optional_declared_or_detected_input(
            &input_names,
            io.and_then(|io| io.position_ids_input.as_deref()),
            &["position_ids"],
            "position_ids_input",
        )?;
        let logits_output = optional_declared_or_detected_output(
            &output_names,
            io.and_then(|io| io.logits_output.as_deref()),
            &["logits"],
            "logits_output",
        )?;
        let projected_state_output = optional_declared_or_detected_output(
            &output_names,
            io.and_then(|io| io.hidden_output.as_deref()),
            &["projected_state"],
            "hidden_output",
        )?;
        if logits_output.is_none() && projected_state_output.is_none() {
            bail!(
                "native proposer metadata must declare at least one semantic output role: io.logits_output or io.hidden_output"
            );
        }

        let kv_ownership = io
            .and_then(|io| io.kv_ownership)
            .unwrap_or(KvOwnership::Owned);
        let (kv_inputs, present_to_past) = match kv_ownership {
            KvOwnership::Owned => {
                let (inputs, outputs) = match io {
                    Some(io) => match (&io.kv_inputs, &io.kv_outputs) {
                        (Some(inputs), Some(outputs)) => (inputs.clone(), outputs.clone()),
                        (None, None) => (Vec::new(), Vec::new()),
                        _ => bail!(
                            "native proposer metadata must declare io.kv_inputs and io.kv_outputs together for owned KV"
                        ),
                    },
                    None => {
                        let inputs = input_names
                            .iter()
                            .filter(|name| is_past_name(name))
                            .cloned()
                            .collect::<Vec<_>>();
                        let outputs = output_names
                            .iter()
                            .filter(|name| is_present_name(name))
                            .cloned()
                            .collect::<Vec<_>>();
                        (inputs, outputs)
                    }
                };
                if inputs.len() != outputs.len() {
                    bail!(
                        "native proposer owned-KV contract has {} past inputs but {} present outputs; declare equal positional lists",
                        inputs.len(),
                        outputs.len()
                    );
                }
                let pairs = if io.is_some() {
                    outputs.into_iter().zip(inputs.iter().cloned()).collect()
                } else {
                    outputs
                            .into_iter()
                            .map(|output| {
                                matching_past_name(&output, &inputs)
                                    .map(|input| (output.clone(), input))
                                    .with_context(|| {
                                        format!(
                                            "native proposer present output '{output}' has no matching past input"
                                        )
                                    })
                            })
                            .collect::<anyhow::Result<Vec<_>>>()?
                };
                (inputs, pairs)
            }
            KvOwnership::Shared => {
                if io.is_some_and(|io| io.kv_outputs.as_ref().is_some_and(|v| !v.is_empty())) {
                    bail!(
                        "native proposer metadata declares kv_ownership 'shared' but also declares io.kv_outputs; shared-KV proposers reference target cache and must not emit owned present KV"
                    );
                }
                (Vec::new(), Vec::new())
            }
        };

        Ok(Self {
            session,
            sequence_source,
            sequence_input,
            attention_mask,
            position_ids,
            logits_output,
            projected_state_output,
            kv_ownership,
            kv_inputs,
            present_to_past,
            past: HashMap::new(),
            current_len: 0,
        })
    }

    pub(crate) fn reset(&mut self) {
        self.past.clear();
        self.current_len = 0;
    }

    #[allow(dead_code)]
    pub(crate) fn step_token_ids(
        &mut self,
        token_ids: &[TokenId],
    ) -> anyhow::Result<NativeProposerOutput> {
        if self.sequence_source != SequenceInputKind::TokenIds {
            bail!(
                "native proposer contract requires inputs_embeds, but token ids were supplied; build embeddings and call step_inputs_embeds"
            );
        }
        if token_ids.is_empty() {
            bail!("native proposer token input must contain at least one token");
        }
        let values = token_ids
            .iter()
            .map(|&token| i64::from(token))
            .collect::<Vec<_>>();
        let sequence = Tensor::from_i64(&[1, token_ids.len()], &values)?;
        self.run_step(sequence, token_ids.len(), self.current_len, &[])
    }

    pub(crate) fn step_inputs_embeds(
        &mut self,
        inputs_embeds: &[f32],
        position_start: usize,
        shared_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<NativeProposerOutput> {
        if self.sequence_source != SequenceInputKind::InputsEmbeds {
            bail!(
                "native proposer contract requires token_ids, but embeddings were supplied; call step_token_ids"
            );
        }
        let meta = self
            .session
            .inputs()
            .iter()
            .find(|meta| meta.name == self.sequence_input)
            .context("native proposer sequence input metadata disappeared")?;
        let width = match meta.shape.last() {
            Some(Dim::Static(width)) if *width > 0 => *width,
            _ => bail!(
                "native proposer inputs_embeds '{}' must declare a positive static final width, got {:?}; export the embedding width in the ONNX type",
                self.sequence_input,
                meta.shape
            ),
        };
        if inputs_embeds.is_empty() || !inputs_embeds.len().is_multiple_of(width) {
            bail!(
                "native proposer inputs_embeds length {} must be a non-zero multiple of declared width {width}",
                inputs_embeds.len()
            );
        }
        let sequence_len = inputs_embeds.len() / width;
        let sequence = tensor_from_f32_as(meta.dtype, &[1, sequence_len, width], inputs_embeds)
            .with_context(|| {
                format!(
                    "build native proposer embeddings for input '{}'",
                    self.sequence_input
                )
            })?;
        self.run_step(sequence, sequence_len, position_start, shared_inputs)
    }

    fn run_step(
        &mut self,
        sequence: Tensor,
        sequence_len: usize,
        position_start: usize,
        shared_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<NativeProposerOutput> {
        let total_len = self
            .current_len
            .checked_add(sequence_len)
            .context("native proposer context length overflow")?;
        let shared_kv_len = shared_inputs
            .iter()
            .filter_map(|(_, tensor)| {
                tensor
                    .shape
                    .len()
                    .checked_sub(2)
                    .map(|axis| tensor.shape[axis])
            })
            .max()
            .unwrap_or(total_len);
        let mut owned = vec![(self.sequence_input.clone(), sequence)];
        if let Some(name) = &self.attention_mask {
            owned.push((
                name.clone(),
                Tensor::from_i64(&[1, shared_kv_len], &vec![1; shared_kv_len])?,
            ));
        }
        if let Some(name) = &self.position_ids {
            let position_end = position_start
                .checked_add(sequence_len)
                .context("native proposer position range overflow")?;
            let positions = (position_start..position_end)
                .map(|position| i64::try_from(position).context("position id exceeds i64 range"))
                .collect::<anyhow::Result<Vec<_>>>()?;
            owned.push((
                name.clone(),
                Tensor::from_i64(&[1, sequence_len], &positions)?,
            ));
        }
        match self.kv_ownership {
            KvOwnership::Owned => {
                for name in &self.kv_inputs {
                    let tensor = match self.past.remove(name) {
                        Some(tensor) => tensor,
                        None => make_empty_input_tensor(&self.session, name)?,
                    };
                    owned.push((name.clone(), tensor));
                }
            }
            KvOwnership::Shared => {
                for (name, tensor) in shared_inputs {
                    if !self.session.inputs().iter().any(|meta| meta.name == *name) {
                        bail!(
                            "native proposer shared-KV input '{name}' is not exposed by the graph; graph inputs: {:?}",
                            self.session
                                .inputs()
                                .iter()
                                .map(|meta| meta.name.as_str())
                                .collect::<Vec<_>>()
                        );
                    }
                    owned.push((name.clone(), tensor.clone()));
                }
            }
        }
        let bindings = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let outputs = self
                .session
                .run(&bindings)
                .context("native proposer forward pass failed; verify metadata port names, sequence_source, kv_ownership, and tensor shapes")?;
        let names = self
            .session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let mut named = names.into_iter().zip(outputs).collect::<HashMap<_, _>>();
        let logits = self
            .logits_output
            .as_ref()
            .map(|name| {
                let tensor = named.remove(name).with_context(|| {
                    format!("native proposer omitted declared logits output '{name}'")
                })?;
                extract_logits(&tensor)
            })
            .transpose()?;
        let projected_state = self
            .projected_state_output
            .as_ref()
            .map(|name| {
                let tensor = named.remove(name).with_context(|| {
                    format!("native proposer omitted declared hidden output '{name}'")
                })?;
                extract_last_row(&tensor)
            })
            .transpose()?;
        if self.kv_ownership == KvOwnership::Owned {
            let mut next = HashMap::with_capacity(self.present_to_past.len());
            for (present, past) in &self.present_to_past {
                let tensor = named.remove(present).with_context(|| {
                    format!("native proposer omitted declared present output '{present}'")
                })?;
                next.insert(past.clone(), tensor);
            }
            self.past = next;
            self.current_len = total_len;
        }
        Ok(NativeProposerOutput {
            logits,
            projected_state,
        })
    }
}

impl NativeDecodeSession {
    pub(crate) fn load_with_weight_offload_host_cache(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        host_cache: onnx_runtime_ep_cpu::WeightOffloadHostCache,
        gqa_sequence_lengths_policy: NativeGqaSequenceLengthsPolicy,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        let preference = match device {
            NativeDecodeDevice::Cpu => DevicePreference::Cpu,
            NativeDecodeDevice::Cuda { index } => DevicePreference::Gpu { index },
        };
        let mut builder = InferenceSession::builder().model(path).device(preference);
        if device == NativeDecodeDevice::Cpu {
            let ep =
                onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_weight_offload_host_cache(
                    host_cache,
                )
                .context("initialize native CPU execution provider")?;
            builder = builder.execution_provider(Arc::new(ep));
        }
        #[cfg(not(feature = "cuda"))]
        let _ = gqa_sequence_lengths_policy;
        #[cfg(feature = "cuda")]
        if let NativeDecodeDevice::Cuda { index } = device {
            let gqa_sequence_lengths_policy = match gqa_sequence_lengths_policy {
                NativeGqaSequenceLengthsPolicy::PerBatchOnly => {
                    onnx_runtime_ep_cuda::GqaSequenceLengthsPolicy::PerBatchOnly
                }
                NativeGqaSequenceLengthsPolicy::AllowUnitBatchScalar => {
                    onnx_runtime_ep_cuda::GqaSequenceLengthsPolicy::AllowUnitBatchScalar
                }
            };
            let ep = onnx_runtime_ep_cuda::CudaExecutionProvider::initialized_with_options(
                index.unwrap_or(0),
                onnx_runtime_ep_cuda::CudaExecutionProviderOptions {
                    gqa_sequence_lengths_policy,
                },
            )
            .context("initialize native CUDA execution provider")?;
            builder = builder.execution_provider(Arc::new(ep));
        }
        let session = builder.build().context("load native decoder model")?;
        Self::from_session_with_cuda_kv_max_len_and_io(session, None, io)
    }

    /// Load with an explicit CUDA KV capacity. `None` uses
    /// `ONNX_GENAI_CUDA_KV_MAX_LEN`, then the 4096-token default.
    pub fn load_with_cuda_kv_max_len(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        cuda_kv_max_len: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::load_with_cuda_options(
            path,
            device,
            NativeDecodeCudaOptions {
                kv_max_len: cuda_kv_max_len,
                graph_capture: None,
            },
        )
    }

    /// Load with explicit native-CUDA decode options. When graph capture is
    /// unspecified (`None`) and `ONNX_GENAI_CUDA_GRAPH` is unset, capture is
    /// auto-enabled whenever the decode topology is structurally graph-safe
    /// (CUDA device with owned, device-resident KV), and transparently declines
    /// to eager execution otherwise. An explicit `ONNX_GENAI_CUDA_GRAPH=0`/`=1`
    /// or a programmatic `graph_capture` value always overrides the
    /// auto-decision.
    pub fn load_with_cuda_options(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        options: NativeDecodeCudaOptions,
    ) -> anyhow::Result<Self> {
        let preference = match device {
            NativeDecodeDevice::Cpu => DevicePreference::Cpu,
            NativeDecodeDevice::Cuda { index } => DevicePreference::Gpu { index },
        };
        let session = InferenceSession::builder()
            .model(path)
            .device(preference)
            .build()
            .context("load native decoder model")?;
        Self::from_session_with_cuda_options(session, options)
    }

    /// Wrap an already-built native session, validating its decoder-with-past I/O.
    pub fn from_session(session: InferenceSession) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_options(session, NativeDecodeCudaOptions::default())
    }

    fn from_session_with_cuda_kv_max_len_and_io(
        session: InferenceSession,
        cuda_kv_max_len: Option<usize>,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_options_and_io(
            session,
            NativeDecodeCudaOptions {
                kv_max_len: cuda_kv_max_len,
                graph_capture: None,
            },
            io,
        )
    }

    fn from_session_with_cuda_options(
        session: InferenceSession,
        cuda_options: NativeDecodeCudaOptions,
    ) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_options_and_io(session, cuda_options, None)
    }

    fn from_session_with_cuda_options_and_io(
        mut session: InferenceSession,
        cuda_options: NativeDecodeCudaOptions,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        let input_names = session
            .inputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let output_names = session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();

        let sequence_source = io
            .and_then(|io| io.sequence_source)
            .unwrap_or(SequenceInputKind::TokenIds);
        if sequence_source != SequenceInputKind::TokenIds {
            bail!(
                "native target decoder requires metadata sequence_source 'token_ids'; got '{sequence_source:?}'. Embedding-driven graphs are supported as proposer sessions, while generation still requires a token-driven target decoder"
            );
        }
        let input_ids = declared_or_detected_input(
            &input_names,
            io.and_then(|io| io.token_input.as_deref()),
            &["input_ids", "decoder_input_ids"],
            "token_input",
        )?;
        let attention_mask = declared_or_detected_input(
            &input_names,
            io.and_then(|io| io.attention_mask_input.as_deref()),
            &["attention_mask"],
            "attention_mask_input",
        )?;
        let position_ids = optional_declared_or_detected_input(
            &input_names,
            io.and_then(|io| io.position_ids_input.as_deref()),
            &["position_ids"],
            "position_ids_input",
        )?;
        let logits = declared_or_detected_output(
            &output_names,
            io.and_then(|io| io.logits_output.as_deref()),
            &["logits"],
            "logits_output",
        )?;
        let hidden_output = optional_declared_or_detected_output(
            &output_names,
            io.and_then(|io| io.hidden_output.as_deref()),
            &[],
            "hidden_output",
        )?;
        let kv_ownership = io
            .and_then(|io| io.kv_ownership)
            .unwrap_or(KvOwnership::Owned);
        if kv_ownership != KvOwnership::Owned {
            bail!(
                "native target decoder requires metadata kv_ownership 'owned'; got '{kv_ownership:?}'. Shared KV is valid for proposer graphs that reference this target's cache"
            );
        }
        let (kv_inputs, present_outputs) = match io {
            Some(io) => match (&io.kv_inputs, &io.kv_outputs) {
                (Some(inputs), Some(outputs)) => (inputs.clone(), outputs.clone()),
                (None, None) => (Vec::new(), Vec::new()),
                _ => bail!(
                    "native target decoder metadata must declare io.kv_inputs and io.kv_outputs together"
                ),
            },
            None => (
                input_names
                    .iter()
                    .filter(|name| is_past_name(name))
                    .cloned()
                    .collect(),
                output_names
                    .iter()
                    .filter(|name| is_present_name(name))
                    .cloned()
                    .collect(),
            ),
        };
        if kv_inputs.is_empty() || present_outputs.is_empty() {
            bail!(
                "native decode requires decoder-with-past I/O; past inputs: {:?}, present outputs: {:?}",
                kv_inputs,
                present_outputs
            );
        }

        let mut present_to_past = HashMap::new();
        if io.is_some() {
            present_to_past.extend(
                present_outputs
                    .iter()
                    .cloned()
                    .zip(kv_inputs.iter().cloned()),
            );
        } else {
            for output in &present_outputs {
                let Some(input) = matching_past_name(output, &kv_inputs) else {
                    bail!(
                        "native decoder present output '{output}' has no matching past input; inputs: {:?}",
                        kv_inputs
                    );
                };
                present_to_past.insert(output.clone(), input);
            }
        }
        if present_to_past.len() != kv_inputs.len() {
            bail!(
                "native decoder has incomplete past/present pairs; past inputs: {:?}, present outputs: {:?}",
                kv_inputs,
                present_outputs
            );
        }

        let cuda = if session.device_id().device_type == DeviceType::Cuda {
            let max_len = match cuda_options.kv_max_len {
                Some(0) => bail!("CUDA KV max length must be greater than zero"),
                Some(value) => value,
                None => cuda_kv_max_len_from_env()?,
            };
            let runtime_config = onnx_genai_runtime_config::runtime_config();
            let graph_enabled = resolve_graph_capture_enabled(
                cuda_options.graph_capture,
                runtime_config.cuda_graph_explicit,
                runtime_config.cuda_graph,
                GraphCaptureStructuralSafety {
                    device_is_cuda: true,
                    kv_ownership,
                },
            );
            Some(DecodeCudaState::new(
                &mut session,
                DecodeCudaIo {
                    input_ids: &input_ids,
                    attention_mask: &attention_mask,
                    position_ids: position_ids.as_deref(),
                    logits: &logits,
                },
                &present_to_past,
                max_len,
                graph_enabled,
            )?)
        } else {
            None
        };

        Ok(Self {
            session,
            input_ids,
            attention_mask,
            position_ids,
            logits,
            hidden_output,
            kv_inputs,
            present_to_past,
            past: HashMap::new(),
            cuda,
            trace: TraceContext::noop(),
            current_len: 0,
            last_hidden: None,
        })
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn kv_layer_count(&self) -> usize {
        self.kv_inputs.len() / 2
    }

    /// Last target hidden-state row produced by the most recent forward.
    pub(crate) fn last_hidden(&self) -> Option<&[f32]> {
        self.last_hidden.as_deref()
    }

    /// Materialize metadata-declared shared-KV references from the target's
    /// current host cache. Native CUDA keeps KV device-resident and is not yet
    /// exposed through this CPU tensor contract.
    pub(crate) fn shared_kv_inputs(
        &self,
        groups: &[SharedKvGroup],
    ) -> anyhow::Result<Vec<(String, Tensor)>> {
        if self.cuda.is_some() {
            bail!(
                "native shared-KV proposer execution currently requires a CPU target session; CUDA target KV references need device-binding alias support"
            );
        }
        let mut inputs = Vec::with_capacity(groups.len() * 2);
        for group in groups {
            let key_target = group.target_key_input.as_deref().with_context(|| {
                format!(
                    "shared_kv group '{}' is missing target_key_input; declare the exact target decoder KV input name",
                    group.name
                )
            })?;
            let value_target = group.target_value_input.as_deref().with_context(|| {
                format!(
                    "shared_kv group '{}' is missing target_value_input; declare the exact target decoder KV input name",
                    group.name
                )
            })?;
            let key_input = group.key_input.as_deref().with_context(|| {
                format!(
                    "shared_kv group '{}' is missing key_input; declare the exact proposer input name",
                    group.name
                )
            })?;
            let value_input = group.value_input.as_deref().with_context(|| {
                format!(
                    "shared_kv group '{}' is missing value_input; declare the exact proposer input name",
                    group.name
                )
            })?;
            let key = self.past.get(key_target).with_context(|| {
                format!(
                    "target shared-KV key '{}' for group '{}' is unavailable; run the target decoder before invoking the proposer and ensure io.kv_inputs names this cache",
                    key_target, group.name
                )
            })?;
            let value = self.past.get(value_target).with_context(|| {
                format!(
                    "target shared-KV value '{}' for group '{}' is unavailable; run the target decoder before invoking the proposer and ensure io.kv_inputs names this cache",
                    value_target, group.name
                )
            })?;
            inputs.push((key_input.to_owned(), key.clone()));
            inputs.push((value_input.to_owned(), value.clone()));
        }
        Ok(inputs)
    }

    pub fn cuda_kv_debug_stats(&self) -> Option<CudaKvDebugStats> {
        self.cuda
            .as_ref()
            .map(|state| state.debug_stats(&self.session))
    }

    pub fn cuda_graph_fallback_reason(&self) -> Option<&str> {
        self.cuda
            .as_ref()
            .and_then(|state| state.graph_fallback_reason.as_deref())
    }

    /// Structural reasons, if any, why CUDA graph capture was declined at
    /// binding time because an auxiliary graph output carries an unresolved
    /// symbolic dimension (not batch or query-seq) that cannot be collapsed to
    /// a fixed persistent device binding. Empty when every auxiliary output was
    /// statically bindable. Decode still runs eagerly when this is non-empty.
    pub fn cuda_auxiliary_bind_declines(&self) -> &[String] {
        self.cuda
            .as_ref()
            .map(|state| state.auxiliary_bind_declines.as_slice())
            .unwrap_or(&[])
    }

    /// Structured reasons from the most recent CUDA graph fallback.
    pub fn cuda_graph_fallback_report(&self) -> Option<&CaptureDeclineReport> {
        self.cuda
            .as_ref()
            .and_then(|state| state.graph_fallback_report.as_ref())
    }

    /// Attach the shared runtime trace context used for capture-fallback events
    /// and per-op executor spans (kernel-variant + capture-rejection reasons).
    pub fn set_trace_context(&mut self, trace: TraceContext) {
        self.session.set_trace_context(trace.clone());
        self.trace = trace;
    }

    /// Dormant option (c) bring-up control (WP4): arm the padded single M=maxK
    /// captured verify graph and retain the captured graph across `rewind`. No-op
    /// on non-CUDA sessions. Not wired into any live decode path yet; exercised
    /// only by the option-(c) rewind-correctness tests.
    #[cfg(test)]
    fn configure_padded_verify_capture(&mut self, max_query_rows: usize) {
        if let Some(state) = self.cuda.as_mut() {
            state.configure_padded_verify_capture(max_query_rows);
        }
    }

    /// Toggle the option (c) "rewind retains the captured graph" guard directly.
    /// Dormant: bring-up / correctness tests only.
    #[cfg(test)]
    fn set_retain_graph_on_rewind(&mut self, retain: bool) {
        if let Some(state) = self.cuda.as_mut() {
            state.set_retain_graph_on_rewind(retain);
        }
    }

    /// Fixed query-row capacity of the dormant padded verify capture, or `None`.
    #[cfg(test)]
    fn padded_query_capacity(&self) -> Option<usize> {
        self.cuda
            .as_ref()
            .and_then(DecodeCudaState::padded_query_capacity)
    }

    pub fn decode(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        <Self as DecodeBackend>::decode(self, token_ids, past_len)
    }

    /// Rewind by prefix-slicing every carried host KV tensor.
    pub fn rewind(&mut self, target_len: usize) -> anyhow::Result<()> {
        <Self as DecodeBackend>::rewind(self, target_len)
    }

    pub fn reset(&mut self) -> anyhow::Result<()> {
        <Self as DecodeBackend>::reset(self)
    }

    /// Generate through the engine's shared token loop, not a backend-local loop.
    pub fn generate(
        &mut self,
        prompt_tokens: &[TokenId],
        options: &GenerateOptions,
        chain: &ProcessorChain,
        tokenizer: &Tokenizer,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_with_callback(prompt_tokens, options, chain, tokenizer, None)
    }

    /// Generate through the shared loop and optionally stream generated tokens.
    pub(crate) fn generate_with_callback(
        &mut self,
        prompt_tokens: &[TokenId],
        options: &GenerateOptions,
        chain: &ProcessorChain,
        tokenizer: &Tokenizer,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        if prompt_tokens.is_empty() {
            bail!("native generation requires at least one prompt token");
        }
        self.reset()?;
        let mut backend = NativeLoopAdapter {
            session: self,
            prompt_tokens: prompt_tokens.to_vec(),
            pending_tokens: prompt_tokens.to_vec(),
        };
        let mut state = DecodeLoopState::new(0, options.seed, options.top_logprobs);
        run_decode_loop(
            &mut backend,
            &mut state,
            options,
            chain,
            tokenizer,
            options.max_context,
            callback,
        )
    }

    fn make_empty_past(&self, name: &str) -> anyhow::Result<Tensor> {
        make_empty_input_tensor(&self.session, name)
    }

    fn decode_cuda(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let total_len = past_len
            .checked_add(token_ids.len())
            .context("native decode context length overflow")?;
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.max_len {
            bail!(
                "CUDA KV capacity exceeded: requested context length {total_len}, configured max_len {} (set ONNX_GENAI_CUDA_KV_MAX_LEN or use load_with_cuda_kv_max_len)",
                state.max_len
            );
        }
        state.extend_mask(past_len, total_len)?;

        if token_ids.len() == 1 {
            state.write_decode_inputs(token_ids[0], past_len)?;
            if let Err(error) = state.run_one_token(&mut self.session, &self.trace) {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                bail!("native CUDA decoder forward pass failed{diagnosis}: {error}");
            }
            let logits = state.read_logits()?;
            // Detection-before-consumption: the logits read above is the single
            // per-step device→host sync. Piggyback on it to poll the shared
            // capture-error word (no extra synchronize). If a captured replay
            // violates a device-side bound, kernels latch the flag and avoid the
            // unsafe access, so fail hard before consuming the produced token.
            let capture_error = self.session.check_device_capture_error()?;
            if capture_error != 0 {
                let _ = state.invalidate_graph(&mut self.session);
                bail!(
                    "native CUDA decoder aborted: device capture validation violation (flags=0x{capture_error:x}) detected during captured graph replay; the produced token was rejected before consumption and the decode graph was invalidated"
                );
            }
            if logits.iter().flatten().any(|value| !value.is_finite()) {
                bail!("native decoder produced non-finite logits");
            }
            state.set_logical_len(total_len)?;
            self.current_len = total_len;
            return Ok(logits);
        }

        state.invalidate_graph(&mut self.session)?;
        let ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let input_ids = Tensor::from_i64(&[1, token_ids.len()], &ids)?;
        let mut owned = Vec::with_capacity(2);
        owned.push((self.input_ids.clone(), input_ids));
        if let Some(position_ids_name) = &self.position_ids {
            let positions = (past_len..total_len)
                .map(|position| i64::try_from(position).context("position id exceeds i64 range"))
                .collect::<anyhow::Result<Vec<_>>>()?;
            owned.push((
                position_ids_name.clone(),
                Tensor::from_i64(&[1, token_ids.len()], &positions)?,
            ));
        }
        let bindings = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let outputs = match self
            .session
            .run_with_device_bindings(&bindings, &mut state.bindings[..state.base_binding_count])
        {
            Ok(outputs) => outputs,
            Err(error) => {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                bail!("native CUDA decoder forward pass failed{diagnosis}: {error}");
            }
        };
        let names = self
            .session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let mut named = names
            .into_iter()
            .zip(outputs)
            .filter_map(|(name, tensor)| tensor.map(|tensor| (name, tensor)))
            .collect::<HashMap<_, _>>();
        let logits = named
            .remove(&self.logits)
            .with_context(|| format!("native decoder omitted logits output '{}'", self.logits))?;
        if !named.is_empty() {
            bail!(
                "native CUDA decoder unexpectedly materialized bound outputs: {:?}",
                named.keys().collect::<Vec<_>>()
            );
        }
        let logits = extract_logits(&logits)?;
        if logits.iter().flatten().any(|value| !value.is_finite()) {
            bail!("native decoder produced non-finite logits");
        }
        state.set_logical_len(total_len)?;
        self.current_len = total_len;
        Ok(logits)
    }

    /// Speculative **verify** primitive (option (b): the safe eager M=K path).
    ///
    /// Runs the `draft` candidate tokens (K = `draft.len()`) through the target
    /// in a single eager forward and returns `[K, vocab]` host logits — one
    /// predicted-distribution row per draft position. This is the primitive
    /// WP2/WP3 build on: the driver compares each row's argmax against `draft`
    /// to find the accepted prefix (plus the free bonus token) and then rewinds
    /// the device KV to the committed length.
    ///
    /// It never enters the M=1 captured-graph greedy hot path — it always takes
    /// the eager multi-token forward (`decode_cuda_eager`) so the 762 tok/s plain
    /// path stays byte-identical. Greedy is the target regime, but returning raw
    /// logits also lets a driver fall back to host sampling for non-greedy
    /// requests. `past` must equal the committed length (`current_len`).
    pub fn decode_verify(
        &mut self,
        draft: &[TokenId],
        past: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if draft.is_empty() {
            bail!("native decode_verify requires at least one draft token");
        }
        if past != self.current_len {
            bail!(
                "native decode_verify past length mismatch: caller supplied {past}, adapter holds {}",
                self.current_len
            );
        }
        if self.cuda.is_some() {
            return self.decode_cuda_eager(draft, past);
        }
        // CPU sessions already run any M>1 forward eagerly through the shared
        // decode path, which returns the full [K, vocab] rows verify needs.
        <Self as DecodeBackend>::decode(self, draft, past)
    }

    /// Eager multi-token (M=K) CUDA forward used by the verify primitive.
    ///
    /// Self-contained on purpose: it mirrors `decode_cuda`'s eager branch but is
    /// a *separate* method so the M=1 captured-graph hot path in `decode_cuda`
    /// stays byte-identical and out of verify's blast radius. It invalidates any
    /// captured graph (option (b) captures nothing), rebuilds host `[1,K]`
    /// input/position tensors, runs against the device KV/mask bindings, and
    /// advances the KV logical length to `past_len + K`.
    ///
    /// The whole pass is wrapped in its own trace span so Deckard's per-op
    /// timings under it remain attributable to the verify forward.
    fn decode_cuda_eager(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let _verify_span = self
            .trace
            .span("native_decode_verify", "spec")
            .with_args(Args::new().with("rows", token_ids.len() as u64));
        let total_len = past_len
            .checked_add(token_ids.len())
            .context("native decode context length overflow")?;
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.max_len {
            bail!(
                "CUDA KV capacity exceeded: requested context length {total_len}, configured max_len {} (set ONNX_GENAI_CUDA_KV_MAX_LEN or use load_with_cuda_kv_max_len)",
                state.max_len
            );
        }
        state.extend_mask(past_len, total_len)?;
        state.invalidate_graph(&mut self.session)?;
        let ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let input_ids = Tensor::from_i64(&[1, token_ids.len()], &ids)?;
        let mut owned = Vec::with_capacity(2);
        owned.push((self.input_ids.clone(), input_ids));
        if let Some(position_ids_name) = &self.position_ids {
            let positions = (past_len..total_len)
                .map(|position| i64::try_from(position).context("position id exceeds i64 range"))
                .collect::<anyhow::Result<Vec<_>>>()?;
            owned.push((
                position_ids_name.clone(),
                Tensor::from_i64(&[1, token_ids.len()], &positions)?,
            ));
        }
        let bindings = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let outputs = match self
            .session
            .run_with_device_bindings(&bindings, &mut state.bindings[..state.base_binding_count])
        {
            Ok(outputs) => outputs,
            Err(error) => {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                bail!("native CUDA verify forward pass failed{diagnosis}: {error}");
            }
        };
        let names = self
            .session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let mut named = names
            .into_iter()
            .zip(outputs)
            .filter_map(|(name, tensor)| tensor.map(|tensor| (name, tensor)))
            .collect::<HashMap<_, _>>();
        let logits = named
            .remove(&self.logits)
            .with_context(|| format!("native decoder omitted logits output '{}'", self.logits))?;
        if !named.is_empty() {
            bail!(
                "native CUDA verify unexpectedly materialized bound outputs: {:?}",
                named.keys().collect::<Vec<_>>()
            );
        }
        let logits = extract_logits(&logits)?;
        if logits.iter().flatten().any(|value| !value.is_finite()) {
            bail!("native decoder produced non-finite logits");
        }
        state.set_logical_len(total_len)?;
        self.current_len = total_len;
        Ok(logits)
    }

    fn decode_cuda_greedy(
        &mut self,
        token_id: TokenId,
        past_len: usize,
    ) -> anyhow::Result<TokenId> {
        let total_len = past_len
            .checked_add(1)
            .context("native decode context length overflow")?;
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.max_len {
            bail!(
                "CUDA KV capacity exceeded: requested context length {total_len}, configured max_len {} (set ONNX_GENAI_CUDA_KV_MAX_LEN or use load_with_cuda_kv_max_len)",
                state.max_len
            );
        }
        state.extend_mask(past_len, total_len)?;
        state.write_decode_inputs(token_id, past_len)?;
        if let Err(error) = state.run_one_token(&mut self.session, &self.trace) {
            let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
            bail!("native CUDA decoder forward pass failed{diagnosis}: {error}");
        }
        let (token_id, capture_error) = state.read_greedy_result()?;
        if capture_error != 0 {
            let _ = state.invalidate_graph(&mut self.session);
            bail!(
                "native CUDA decoder aborted: device capture validation violation (flags=0x{capture_error:x}) detected during captured graph replay; the produced token was rejected before consumption and the decode graph was invalidated"
            );
        }
        state.set_logical_len(total_len)?;
        self.current_len = total_len;
        Ok(token_id)
    }
}

impl DecodeCudaState {
    /// Collect the symbolic dimension ids that the native decoder structurally
    /// pins to `1` at decode time. Batch (axis 0 of every input) and query-seq
    /// (the remaining `input_ids` / `position_ids` axes, which are bound to a
    /// single token) are the only symbols that [`persistent_output_shape`] may
    /// safely collapse to `1`. `batch_only` restricts collection to axis 0 for
    /// inputs whose non-batch axes grow with the sequence (attention_mask and
    /// the past-KV tensors, whose total_seq axis is *not* a decode unit).
    fn collect_unit_symbols(shape: &[Dim], batch_only: bool, out: &mut HashSet<SymbolId>) {
        for (axis, dim) in shape.iter().enumerate() {
            if batch_only && axis != 0 {
                continue;
            }
            if let Dim::Symbolic(symbol) = dim {
                out.insert(*symbol);
            }
        }
    }

    /// First structurally-unresolved symbolic axis of an auxiliary output: a
    /// `Dim::Symbolic` that is *not* one of the decode-unit (batch / query-seq)
    /// symbols. Such a dimension is data-dependent (e.g. an accumulator indexed
    /// by total_seq / past+1), so collapsing it to `1` in a persistent device
    /// binding would under-allocate. Returns `(axis, symbol)` of the offender.
    fn unresolved_symbolic_axis(
        shape: &[Dim],
        unit_symbols: &HashSet<SymbolId>,
    ) -> Option<(usize, SymbolId)> {
        shape.iter().enumerate().find_map(|(axis, dim)| match dim {
            Dim::Symbolic(symbol) if !unit_symbols.contains(symbol) => Some((axis, *symbol)),
            _ => None,
        })
    }

    fn persistent_output_shape(
        name: &str,
        dtype: DataType,
        shape: &[Dim],
    ) -> anyhow::Result<Vec<usize>> {
        if matches!(dtype, DataType::Undefined | DataType::String) {
            bail!(
                "cannot bind auxiliary CUDA graph output '{name}' persistently: dtype {dtype:?} does not have fixed-size device tensor storage, but CUDA graph capture requires every declared graph output to use stable device storage; export this output as a numeric tensor or remove the unused graph output"
            );
        }
        let shape = shape
            .iter()
            .map(|dim| match dim {
                Dim::Static(value) => *value,
                Dim::Symbolic(_) => 1,
            })
            .collect::<Vec<_>>();
        let elements = shape.iter().try_fold(1usize, |product, &dim| {
            product.checked_mul(dim).with_context(|| {
                format!(
                    "cannot bind auxiliary CUDA graph output '{name}' persistently: shape {shape:?} overflows the device allocation size; export a bounded output shape or remove the unused graph output"
                )
            })
        })?;
        dtype.checked_storage_bytes(elements).with_context(|| {
            format!(
                "cannot bind auxiliary CUDA graph output '{name}' persistently: dtype {dtype:?} shape {shape:?} has no representable device allocation size; export a fixed-size numeric tensor or remove the unused graph output"
            )
        })?;
        Ok(shape)
    }

    fn new(
        session: &mut InferenceSession,
        io: DecodeCudaIo<'_>,
        present_to_past: &HashMap<String, String>,
        max_len: usize,
        graph_enabled: bool,
    ) -> anyhow::Result<Self> {
        let mut mask = session.allocate_device_binding(
            io.attention_mask,
            None::<String>,
            DataType::Int64,
            vec![1, max_len],
            vec![1, 0],
        )?;
        mask.write_bytes(0, &vec![0; max_len * std::mem::size_of::<i64>()])?;

        let mut pairs = present_to_past
            .iter()
            .map(|(present, past)| (present.clone(), past.clone()))
            .collect::<Vec<_>>();
        pairs.sort_unstable_by(|left, right| left.1.cmp(&right.1));
        let mut bindings = Vec::with_capacity(4 + pairs.len());
        bindings.push(mask);
        let kv_start = bindings.len();
        for (present, past) in pairs {
            let meta = session
                .inputs()
                .iter()
                .find(|meta| meta.name == past)
                .with_context(|| format!("missing CUDA KV input metadata for '{past}'"))?;
            if !matches!(meta.dtype, DataType::Float32 | DataType::Float16) || meta.shape.len() != 4
            {
                bail!(
                    "CUDA KV input '{past}' must be rank-4 f32 or f16, got {:?} {:?}",
                    meta.dtype,
                    meta.shape
                );
            }
            let mut physical_shape = Vec::with_capacity(4);
            for (axis, dim) in meta.shape.iter().copied().enumerate() {
                let value = if axis == 0 {
                    1
                } else if axis == 2 {
                    max_len
                } else if let Dim::Static(value) = dim {
                    value
                } else {
                    bail!(
                        "cannot infer CUDA KV dimension {axis} for '{past}' shape {:?}",
                        meta.shape
                    );
                };
                physical_shape.push(value);
            }
            let mut logical_shape = physical_shape.clone();
            logical_shape[2] = 0;
            bindings.push(session.allocate_device_binding(
                past,
                Some(present),
                meta.dtype,
                physical_shape,
                logical_shape,
            )?);
        }
        let kv_end = bindings.len();

        let logits_meta = session
            .outputs()
            .iter()
            .find(|meta| meta.name == io.logits)
            .with_context(|| format!("missing CUDA logits output metadata for '{}'", io.logits))?;
        if !matches!(logits_meta.dtype, DataType::Float32 | DataType::Float16)
            || logits_meta.shape.is_empty()
        {
            bail!(
                "CUDA logits output '{}' must be non-scalar f32 or f16, got {:?} {:?}",
                io.logits,
                logits_meta.dtype,
                logits_meta.shape
            );
        }
        let logits_dtype = logits_meta.dtype;
        let logits_shape =
            Self::persistent_output_shape(io.logits, logits_dtype, &logits_meta.shape)?;
        let logits_device_binding = session.allocate_device_output_binding(
            io.logits,
            logits_dtype,
            logits_shape.clone(),
            logits_shape.clone(),
        )?;

        let present_outputs = present_to_past.keys().cloned().collect::<HashSet<_>>();
        let auxiliary_meta = session
            .outputs()
            .iter()
            .filter(|meta| meta.name != io.logits && !present_outputs.contains(&meta.name))
            .cloned()
            .collect::<Vec<_>>();

        // Structural safe-collapse analysis for auxiliary outputs. The native
        // decoder pins batch and query-seq to `1` at decode, so a symbolic aux
        // dimension is only safe to collapse to `1` when it is one of those
        // structurally-unit axes. Gather every symbol the decoder binds to `1`:
        // input_ids / position_ids (bound to `[1, 1]`) on all axes, plus the
        // batch axis (axis 0) of attention_mask and each past-KV input. Any
        // other symbolic aux dim (e.g. one indexed by total_seq / past+1) is
        // data-dependent and must not be collapsed. See RULES.md §2 — this is a
        // purely structural signal, never a model-name gate.
        let mut unit_symbols: HashSet<SymbolId> = HashSet::new();
        if let Some(meta) = session
            .inputs()
            .iter()
            .find(|meta| meta.name == io.input_ids)
        {
            Self::collect_unit_symbols(&meta.shape, false, &mut unit_symbols);
        }
        if let Some(position_ids) = io.position_ids
            && let Some(meta) = session
                .inputs()
                .iter()
                .find(|meta| meta.name == position_ids)
        {
            Self::collect_unit_symbols(&meta.shape, false, &mut unit_symbols);
        }
        if let Some(meta) = session
            .inputs()
            .iter()
            .find(|meta| meta.name == io.attention_mask)
        {
            Self::collect_unit_symbols(&meta.shape, true, &mut unit_symbols);
        }
        for past in present_to_past.values() {
            if let Some(meta) = session.inputs().iter().find(|meta| &meta.name == past) {
                Self::collect_unit_symbols(&meta.shape, true, &mut unit_symbols);
            }
        }

        let auxiliary_start = bindings.len();
        let mut declined_auxiliary: Vec<String> = Vec::new();
        for meta in auxiliary_meta {
            if let Some((axis, symbol)) = Self::unresolved_symbolic_axis(&meta.shape, &unit_symbols)
            {
                // The output's extent on this axis is data-dependent and not
                // structurally identifiable as batch or query-seq. Collapsing
                // it to `1` (as a persistent device binding requires) would
                // under-allocate, so we deliberately do NOT bind it. The eager
                // executor JIT-sizes and materializes this output every step,
                // so decode still works; only CUDA graph capture is forfeited
                // (capture demands a stable device address for every output).
                let symbol_label = session
                    .graph()
                    .symbol_constraints
                    .get(&symbol)
                    .and_then(|constraints| constraints.name.clone())
                    .unwrap_or_else(|| format!("symbol#{}", symbol.0));
                declined_auxiliary.push(format!(
                    "'{}' (axis {axis} is symbolic dim '{symbol_label}', not structurally batch or query-seq)",
                    meta.name
                ));
                continue;
            }
            let shape = Self::persistent_output_shape(&meta.name, meta.dtype, &meta.shape)?;
            bindings.push(
                session
                    .allocate_device_output_binding(
                        &meta.name,
                        meta.dtype,
                        shape.clone(),
                        shape,
                    )
                    .with_context(|| {
                        format!(
                            "failed to allocate persistent CUDA device binding for auxiliary graph output '{}'; CUDA graph capture requires every declared output to keep a stable device address",
                            meta.name
                        )
                    })?,
            );
        }
        let auxiliary_end = bindings.len();
        let base_binding_count = bindings.len();

        // If any auxiliary output could not be persistently bound, CUDA graph
        // capture is impossible (an unbound output would materialize on the
        // host mid-capture). Decline capture up front, with a clear structural
        // reason, and fall back to the eager device path — which still decodes
        // correctly by dynamically allocating the unbindable output each step.
        let graph_enabled = if !declined_auxiliary.is_empty() {
            if graph_enabled {
                tracing::warn!(
                    "native CUDA decode graph capture disabled: auxiliary output(s) {} carry unresolved symbolic dimensions that cannot be collapsed to a fixed persistent device binding; decode continues eagerly with dynamic allocation for those outputs",
                    declined_auxiliary.join(", ")
                );
            } else {
                tracing::debug!(
                    "native CUDA decode leaving auxiliary output(s) {} unbound (unresolved symbolic dimensions); eager path allocates them dynamically",
                    declined_auxiliary.join(", ")
                );
            }
            false
        } else {
            graph_enabled
        };

        let input_ids_binding = bindings.len();
        bindings.push(session.allocate_device_binding(
            io.input_ids,
            None::<String>,
            DataType::Int64,
            vec![1, 1],
            vec![1, 1],
        )?);
        let position_ids_binding = if let Some(position_ids) = io.position_ids {
            let index = bindings.len();
            bindings.push(session.allocate_device_binding(
                position_ids,
                None::<String>,
                DataType::Int64,
                vec![1, 1],
                vec![1, 1],
            )?);
            Some(index)
        } else {
            None
        };
        let logits_binding = bindings.len();
        bindings.push(logits_device_binding);

        #[cfg(feature = "cuda")]
        let argmax_words = {
            let vocab = *logits_shape
                .last()
                .context("CUDA logits shape has no vocabulary dimension")?;
            2 + onnx_runtime_ep_cuda::device_argmax_scratch_words(vocab)
        };
        #[cfg(not(feature = "cuda"))]
        let argmax_words = 2;
        let greedy_result = session.allocate_device_output_binding(
            "__native_greedy_argmax",
            DataType::Uint32,
            vec![argmax_words],
            vec![2],
        )?;

        Ok(Self {
            logical_len: 0,
            max_len,
            bindings,
            base_binding_count,
            kv_binding_range: kv_start..kv_end,
            auxiliary_binding_range: auxiliary_start..auxiliary_end,
            input_ids_binding,
            position_ids_binding,
            logits_binding,
            logits_shape,
            logits_dtype,
            greedy_result,
            graph_enabled,
            graph_phase: DecodeCudaGraphPhase::NeedsWarmup,
            graph_captures: 0,
            graph_replays: 0,
            graph_fallbacks: 0,
            graph_fallback_reason: None,
            graph_fallback_report: None,
            auxiliary_bind_declines: declined_auxiliary,
            retain_graph_on_rewind: false,
            padded_query_capacity: None,
        })
    }

    fn extend_mask(&mut self, start: usize, end: usize) -> anyhow::Result<()> {
        if end > self.max_len || start > end {
            bail!(
                "invalid CUDA mask update {start}..{end} for capacity {}",
                self.max_len
            );
        }
        let ones = (start..end)
            .flat_map(|_| 1i64.to_le_bytes())
            .collect::<Vec<_>>();
        self.bindings[0].write_bytes(start * std::mem::size_of::<i64>(), &ones)?;
        self.bindings[0].set_logical_shape(vec![1, end])?;
        Ok(())
    }

    fn set_logical_len(&mut self, len: usize) -> anyhow::Result<()> {
        for binding in &mut self.bindings[self.kv_binding_range.clone()] {
            let mut shape = binding.physical_shape().to_vec();
            shape[2] = len;
            binding.set_logical_shape(shape)?;
        }
        self.logical_len = len;
        Ok(())
    }

    fn rewind(&mut self, target_len: usize) -> anyhow::Result<()> {
        if target_len < self.logical_len {
            let zeros = vec![0u8; (self.logical_len - target_len) * std::mem::size_of::<i64>()];
            self.bindings[0].write_bytes(target_len * std::mem::size_of::<i64>(), &zeros)?;
        }
        self.bindings[0].set_logical_shape(vec![1, target_len])?;
        self.set_logical_len(target_len)
    }

    fn write_decode_inputs(&mut self, token_id: TokenId, position: usize) -> anyhow::Result<()> {
        self.bindings[self.input_ids_binding].write_bytes(0, &i64::from(token_id).to_le_bytes())?;
        if let Some(index) = self.position_ids_binding {
            let position = i64::try_from(position).context("position id exceeds i64 range")?;
            self.bindings[index].write_bytes(0, &position.to_le_bytes())?;
        }
        Ok(())
    }

    fn run_one_token(
        &mut self,
        session: &mut InferenceSession,
        trace: &TraceContext,
    ) -> anyhow::Result<()> {
        debug_assert!(self.auxiliary_binding_range.end <= self.base_binding_count);
        if !self.graph_enabled {
            session.run_with_device_bindings(&[], &mut self.bindings)?;
            return Ok(());
        }

        match self.graph_phase {
            DecodeCudaGraphPhase::NeedsWarmup => {
                session.run_with_device_bindings(&[], &mut self.bindings)?;
                self.graph_phase = DecodeCudaGraphPhase::Armed;
            }
            DecodeCudaGraphPhase::Armed => {
                match session.try_capture_with_device_bindings(&[], &mut self.bindings)? {
                    DeviceGraphCaptureResult::Captured(outputs) => {
                        if outputs.iter().any(Option::is_some) {
                            bail!("captured CUDA decode unexpectedly materialized a host output");
                        }
                        self.graph_captures += 1;
                        self.graph_phase = DecodeCudaGraphPhase::Ready;
                    }
                    DeviceGraphCaptureResult::NotCapturable(report) => {
                        self.graph_fallbacks += 1;
                        self.graph_phase = DecodeCudaGraphPhase::Unsupported;
                        trace_capture_declines(trace, &report);
                        let reason = report.to_string();
                        self.graph_fallback_reason = Some(reason.clone());
                        self.graph_fallback_report = Some(report);
                        tracing::warn!(
                            "native CUDA decode graph capture disabled for this generation: {reason}"
                        );
                        session.run_with_device_bindings(&[], &mut self.bindings)?;
                    }
                }
            }
            DecodeCudaGraphPhase::Ready => {
                session.replay_device_graph(&mut self.bindings)?;
                self.graph_replays += 1;
            }
            DecodeCudaGraphPhase::Unsupported => {
                session.run_with_device_bindings(&[], &mut self.bindings)?;
            }
        }
        Ok(())
    }

    fn read_logits(&mut self) -> anyhow::Result<Vec<Vec<f32>>> {
        let bytes = self.bindings[self.logits_binding].read_bytes()?;
        let logits = Tensor::from_raw(self.logits_dtype, self.logits_shape.clone(), &bytes)?;
        extract_logits(&logits)
    }

    fn greedy_fastpath_supported(&self) -> bool {
        self.bindings[self.logits_binding].device_argmax_supported()
    }

    fn read_greedy_result(&mut self) -> anyhow::Result<(TokenId, u32)> {
        let vocab = *self
            .logits_shape
            .last()
            .context("CUDA logits shape has no vocabulary dimension")?;
        self.bindings[self.logits_binding].device_argmax(vocab, &mut self.greedy_result)?;
        let mut bytes = [0_u8; 2 * std::mem::size_of::<u32>()];
        self.greedy_result.read_bytes_into(&mut bytes)?;
        Ok((
            u32::from_ne_bytes(bytes[..4].try_into().expect("four token-id bytes")),
            u32::from_ne_bytes(bytes[4..].try_into().expect("four capture-error bytes")),
        ))
    }

    fn invalidate_graph(&mut self, session: &mut InferenceSession) -> anyhow::Result<()> {
        session.reset_device_graph()?;
        self.graph_phase = DecodeCudaGraphPhase::NeedsWarmup;
        Ok(())
    }

    /// Dormant option (c) switch (kept off until WP4). Arm the padded single
    /// M=maxK captured verify graph: fix the query-row capacity at `max_query_rows`
    /// and retain the captured graph across `rewind` (contents-only mutation)
    /// instead of invalidating it. Not reachable from the plain M=1 hot path nor
    /// the eager (option (b)) verify path; only a future WP4 driver flips it on.
    #[allow(dead_code)]
    fn configure_padded_verify_capture(&mut self, max_query_rows: usize) {
        self.padded_query_capacity = Some(max_query_rows);
        self.retain_graph_on_rewind = true;
    }

    /// Toggle whether `rewind` retains the captured decode graph (option (c),
    /// contents-only mutation) or invalidates it (option (b), the eager default).
    /// Dormant: only exercised by option-(c) correctness tests until WP4.
    #[allow(dead_code)]
    fn set_retain_graph_on_rewind(&mut self, retain: bool) {
        self.retain_graph_on_rewind = retain;
    }

    /// Fixed query-row capacity (M=maxK) of the dormant padded verify capture, or
    /// `None` while the eager (option (b)) verify path is in force.
    #[allow(dead_code)]
    fn padded_query_capacity(&self) -> Option<usize> {
        self.padded_query_capacity
    }

    fn debug_stats(&self, session: &InferenceSession) -> CudaKvDebugStats {
        let mut transfers = DeviceBindingTransferStats::default();
        let device_ptrs = self.bindings[self.kv_binding_range.clone()]
            .iter()
            .map(|binding| {
                let stats = binding.transfer_stats();
                transfers.host_upload_calls += stats.host_upload_calls;
                transfers.host_upload_bytes += stats.host_upload_bytes;
                transfers.host_download_calls += stats.host_download_calls;
                transfers.host_download_bytes += stats.host_download_bytes;
                binding.device_ptr() as usize
            })
            .collect();
        CudaKvDebugStats {
            logical_len: self.logical_len,
            max_len: self.max_len,
            device_ptrs,
            kv_transfers: transfers,
            graph: CudaGraphDebugStats {
                enabled: self.graph_enabled,
                captures: self.graph_captures,
                replays: self.graph_replays,
                fallbacks: self.graph_fallbacks,
                allocation_counts: session.device_allocation_counts().unwrap_or_default(),
                fallback_report: self.graph_fallback_report.clone(),
            },
        }
    }
}

fn cuda_kv_max_len_from_env() -> anyhow::Result<usize> {
    match std::env::var("ONNX_GENAI_CUDA_KV_MAX_LEN") {
        Ok(value) => {
            let parsed = value.trim().parse::<usize>().with_context(|| {
                format!("invalid ONNX_GENAI_CUDA_KV_MAX_LEN={value:?}: expected a positive integer")
            })?;
            if parsed == 0 {
                bail!("ONNX_GENAI_CUDA_KV_MAX_LEN must be greater than zero");
            }
            Ok(parsed)
        }
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_CUDA_KV_MAX_LEN),
        Err(error) => Err(error).context("read ONNX_GENAI_CUDA_KV_MAX_LEN"),
    }
}

impl DecodeBackend for NativeDecodeSession {
    fn current_len(&self) -> usize {
        self.current_len
    }

    fn max_context(&self) -> Option<usize> {
        self.cuda.as_ref().map(|state| state.max_len)
    }

    fn decode(&mut self, token_ids: &[TokenId], past_len: usize) -> anyhow::Result<Vec<Vec<f32>>> {
        if token_ids.is_empty() {
            bail!("native decode requires at least one token");
        }
        if past_len != self.current_len {
            bail!(
                "native decode past length mismatch: caller supplied {past_len}, adapter holds {}",
                self.current_len
            );
        }
        if self.cuda.is_some() {
            return self.decode_cuda(token_ids, past_len);
        }
        let total_len = past_len
            .checked_add(token_ids.len())
            .context("native decode context length overflow")?;
        let ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let input_ids = Tensor::from_i64(&[1, token_ids.len()], &ids)?;
        let attention_mask = Tensor::from_i64(&[1, total_len], &vec![1; total_len])?;

        let mut owned = Vec::with_capacity(3 + self.kv_inputs.len());
        owned.push((self.input_ids.clone(), input_ids));
        owned.push((self.attention_mask.clone(), attention_mask));
        if let Some(position_ids_name) = &self.position_ids {
            let positions = (past_len..total_len)
                .map(|position| i64::try_from(position).context("position id exceeds i64 range"))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let position_ids = Tensor::from_i64(&[1, token_ids.len()], &positions)?;
            owned.push((position_ids_name.clone(), position_ids));
        }
        for name in &self.kv_inputs {
            let tensor = match self.past.remove(name) {
                Some(tensor) => tensor,
                None => self.make_empty_past(name)?,
            };
            owned.push((name.clone(), tensor));
        }
        let bindings = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        // Single-token CPU decode: run the whole forward inside one decode-pool
        // installation so the ~121 per-op `MatMulNBits` projections execute
        // inline on resident workers instead of each re-installing the pool
        // (eliminating the per-op fork-join crossing). Prefill (M>1) and the CUDA
        // path (handled above) must keep using the global pool, so gate on M==1.
        let run_result: anyhow::Result<_> = if token_ids.len() == 1 {
            onnx_runtime_ep_cpu::with_decode_pool_scope(|| {
                self.session.run(&bindings).map_err(anyhow::Error::from)
            })
        } else {
            self.session.run(&bindings).map_err(anyhow::Error::from)
        };
        let outputs = match run_result {
            Ok(outputs) => outputs,
            Err(error) => {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                bail!("native decoder forward pass failed{diagnosis}: {error}");
            }
        };
        let names = self
            .session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let mut named = names.into_iter().zip(outputs).collect::<HashMap<_, _>>();
        let logits = named
            .remove(&self.logits)
            .with_context(|| format!("native decoder omitted logits output '{}'", self.logits))?;
        let logits = extract_logits(&logits)?;
        if logits.iter().flatten().any(|value| !value.is_finite()) {
            bail!("native decoder produced non-finite logits");
        }
        self.last_hidden = self
            .hidden_output
            .as_ref()
            .map(|name| {
                let tensor = named.remove(name).with_context(|| {
                    format!("native decoder omitted declared hidden output '{name}'")
                })?;
                extract_last_row(&tensor)
                    .with_context(|| format!("read native decoder hidden output '{name}'"))
            })
            .transpose()?;

        let mut next_past = HashMap::with_capacity(self.kv_inputs.len());
        for (present, past) in &self.present_to_past {
            let tensor = named
                .remove(present)
                .with_context(|| format!("native decoder omitted present output '{present}'"))?;
            let seq_axis =
                tensor.shape.len().checked_sub(2).with_context(|| {
                    format!("native present tensor '{present}' rank is below 2")
                })?;
            if tensor.shape[seq_axis] != total_len {
                bail!(
                    "native present tensor '{present}' sequence length {} does not match {total_len}",
                    tensor.shape[seq_axis]
                );
            }
            next_past.insert(past.clone(), tensor);
        }
        self.past = next_past;
        self.current_len = total_len;
        Ok(logits)
    }

    fn rewind(&mut self, target_len: usize) -> anyhow::Result<()> {
        if target_len > self.current_len {
            bail!(
                "cannot rewind native KV from {} forward to {target_len}",
                self.current_len
            );
        }
        if target_len == self.current_len {
            return Ok(());
        }
        if let Some(state) = &mut self.cuda {
            // Option (b) default: invalidate the captured decode graph before the
            // KV roll-back (the eager verify path captures nothing, and the plain
            // M=1 path re-warms cleanly). Option (c) (dormant until WP4) retains
            // the single fixed-topology M=maxK graph and rewinds contents only —
            // `state.rewind` mutates just the mask tail + KV logical length, the
            // same data-driven mutation the captured graph already tolerates.
            if !state.retain_graph_on_rewind {
                state.invalidate_graph(&mut self.session)?;
            }
            state.rewind(target_len)?;
            self.current_len = target_len;
            return Ok(());
        }
        if target_len == 0 {
            self.past.clear();
            self.current_len = 0;
            self.last_hidden = None;
            return Ok(());
        }
        for (name, tensor) in &mut self.past {
            let axis = tensor
                .shape
                .len()
                .checked_sub(2)
                .with_context(|| format!("native KV tensor '{name}' rank is below 2"))?;
            *tensor = prefix_slice(tensor, axis, target_len)
                .with_context(|| format!("rewind native KV tensor '{name}'"))?;
        }
        self.current_len = target_len;
        Ok(())
    }
}

impl Drop for NativeDecodeSession {
    fn drop(&mut self) {
        if let Some(state) = &mut self.cuda {
            let _ = state.invalidate_graph(&mut self.session);
        }
    }
}

struct NativeLoopAdapter<'a> {
    session: &'a mut NativeDecodeSession,
    prompt_tokens: Vec<TokenId>,
    pending_tokens: Vec<TokenId>,
}

impl DecodeLoopBackend for NativeLoopAdapter<'_> {
    fn context_len(&self) -> usize {
        self.session.current_len() + self.pending_tokens.len()
    }

    fn processor_prompt_tokens(&self) -> Vec<TokenId> {
        self.prompt_tokens.clone()
    }

    fn next_logits(&mut self) -> anyhow::Result<Vec<f32>> {
        let past_len = self.session.current_len();
        self.session
            .decode(&self.pending_tokens, past_len)?
            .pop()
            .context("native decoder produced no logits")
    }

    fn greedy_fastpath_supported(&self) -> bool {
        self.session
            .cuda
            .as_ref()
            .is_some_and(DecodeCudaState::greedy_fastpath_supported)
    }

    fn next_token_greedy(&mut self) -> anyhow::Result<TokenId> {
        if self.pending_tokens.len() != 1 {
            return Ok(sample_greedy(&self.next_logits()?));
        }
        let past_len = self.session.current_len();
        self.session
            .decode_cuda_greedy(self.pending_tokens[0], past_len)
    }

    fn commit_token(&mut self, token_id: TokenId) -> anyhow::Result<()> {
        self.pending_tokens.clear();
        self.pending_tokens.push(token_id);
        Ok(())
    }
}

fn find_name(names: &[String], candidates: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        let lower = name.to_ascii_lowercase();
        candidates
            .iter()
            .any(|candidate| lower == *candidate || lower.ends_with(&format!(".{candidate}")))
            .then(|| name.clone())
    })
}

fn declared_or_detected_input(
    names: &[String],
    declared: Option<&str>,
    candidates: &[&str],
    field: &str,
) -> anyhow::Result<String> {
    if let Some(name) = declared {
        if names.iter().any(|candidate| candidate == name) {
            return Ok(name.to_owned());
        }
        bail!(
            "native graph metadata io.{field} declares input '{name}', but the graph exposes inputs {names:?}; fix the metadata port name"
        );
    }
    find_name(names, candidates).with_context(|| {
        format!(
            "native graph is missing {}; declare io.{field} explicitly or export one of {candidates:?}",
            candidates.first().copied().unwrap_or("the required input")
        )
    })
}

fn optional_declared_or_detected_input(
    names: &[String],
    declared: Option<&str>,
    candidates: &[&str],
    field: &str,
) -> anyhow::Result<Option<String>> {
    declared
        .map(|name| declared_or_detected_input(names, Some(name), candidates, field))
        .transpose()
        .map(|declared| declared.or_else(|| find_name(names, candidates)))
}

fn declared_or_detected_output(
    names: &[String],
    declared: Option<&str>,
    candidates: &[&str],
    field: &str,
) -> anyhow::Result<String> {
    if let Some(name) = declared {
        if names.iter().any(|candidate| candidate == name) {
            return Ok(name.to_owned());
        }
        bail!(
            "native graph metadata io.{field} declares output '{name}', but the graph exposes outputs {names:?}; fix the metadata port name"
        );
    }
    find_name(names, candidates).with_context(|| {
        format!(
            "native graph is missing {}; declare io.{field} explicitly or export one of {candidates:?}",
            candidates.first().copied().unwrap_or("the required output")
        )
    })
}

fn optional_declared_or_detected_output(
    names: &[String],
    declared: Option<&str>,
    candidates: &[&str],
    field: &str,
) -> anyhow::Result<Option<String>> {
    declared
        .map(|name| declared_or_detected_output(names, Some(name), candidates, field))
        .transpose()
        .map(|declared| declared.or_else(|| find_name(names, candidates)))
}

fn is_past_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("past_key_values.") || lower.starts_with("past.")
}

fn is_present_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("present_key_values.") || lower.starts_with("present.")
}

fn matching_past_name(output: &str, inputs: &[String]) -> Option<String> {
    let lower = output.to_ascii_lowercase();
    let suffix = lower
        .strip_prefix("present_key_values.")
        .or_else(|| lower.strip_prefix("present."))?;
    inputs.iter().find_map(|input| {
        let input_lower = input.to_ascii_lowercase();
        (input_lower.strip_prefix("past_key_values.") == Some(suffix)
            || input_lower.strip_prefix("past.") == Some(suffix))
        .then(|| input.clone())
    })
}

fn extract_logits(tensor: &Tensor) -> anyhow::Result<Vec<Vec<f32>>> {
    let values = tensor_to_f32(tensor)?;
    match tensor.shape.as_slice() {
        [vocab] if *vocab > 0 => Ok(vec![values]),
        [seq, vocab] if *seq > 0 && *vocab > 0 => Ok(values
            .chunks(*vocab)
            .take(*seq)
            .map(<[f32]>::to_vec)
            .collect()),
        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => Ok(values
            .chunks(*vocab)
            .take(*seq)
            .map(<[f32]>::to_vec)
            .collect()),
        shape => bail!("unsupported logits tensor shape: {shape:?}"),
    }
}

fn extract_last_row(tensor: &Tensor) -> anyhow::Result<Vec<f32>> {
    let width = *tensor
        .shape
        .last()
        .context("tensor has no final feature dimension")?;
    if width == 0 {
        bail!("tensor final feature dimension must be positive");
    }
    let values = tensor_to_f32(tensor)?;
    if values.len() < width || !values.len().is_multiple_of(width) {
        bail!(
            "tensor element count {} is not a positive multiple of final feature width {width}",
            values.len()
        );
    }
    Ok(values[values.len() - width..].to_vec())
}

fn tensor_to_f32(tensor: &Tensor) -> anyhow::Result<Vec<f32>> {
    match tensor.dtype {
        DataType::Float32 => Ok(tensor.to_vec_f32()),
        DataType::Float16 => Ok(tensor
            .as_bytes()
            .chunks_exact(2)
            .map(|bytes| f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect()),
        DataType::BFloat16 => Ok(tensor
            .as_bytes()
            .chunks_exact(2)
            .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16))
            .collect()),
        dtype => bail!("native logits must be Float32, Float16, or BFloat16, got {dtype:?}"),
    }
}

fn tensor_from_f32_as(dtype: DataType, shape: &[usize], values: &[f32]) -> anyhow::Result<Tensor> {
    match dtype {
        DataType::Float32 => Ok(Tensor::from_f32(shape, values)?),
        DataType::Float16 => {
            let bytes = values
                .iter()
                .flat_map(|value| half::f16::from_f32(*value).to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            Ok(Tensor::from_raw(dtype, shape.to_vec(), &bytes)?)
        }
        DataType::BFloat16 => {
            let bytes = values
                .iter()
                .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            Ok(Tensor::from_raw(dtype, shape.to_vec(), &bytes)?)
        }
        other => bail!(
            "native embeddings input must be Float32, Float16, or BFloat16, got {other:?}; fix io.inputs_embeds_input or export a floating tensor"
        ),
    }
}

fn make_empty_input_tensor(session: &InferenceSession, name: &str) -> anyhow::Result<Tensor> {
    let meta = session
        .inputs()
        .iter()
        .find(|meta| meta.name == name)
        .with_context(|| format!("missing native KV metadata for '{name}'"))?;
    if meta.shape.len() < 3 {
        bail!(
            "native KV input '{name}' has unsupported shape {:?}; expected rank at least 3 with sequence on the penultimate axis",
            meta.shape
        );
    }
    let seq_axis = meta.shape.len() - 2;
    let mut shape = Vec::with_capacity(meta.shape.len());
    for (axis, dim) in meta.shape.iter().copied().enumerate() {
        let value = if axis == 0 {
            1
        } else if axis == seq_axis {
            0
        } else if let Dim::Static(value) = dim {
            value
        } else {
            bail!(
                "cannot create empty native KV input '{name}': dimension {axis} in shape {:?} is symbolic and is neither batch nor sequence; export a static cache geometry",
                meta.shape
            );
        };
        shape.push(value);
    }
    let bytes = meta
        .dtype
        .checked_storage_bytes(shape.iter().product())
        .with_context(|| format!("unsupported KV dtype {:?} for '{name}'", meta.dtype))?;
    Tensor::from_raw(meta.dtype, shape, &vec![0; bytes])
        .with_context(|| format!("create empty native KV tensor '{name}'"))
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = (fraction << (shift + 1)) & 0x03ff;
            sign | ((127 - 15 - shift) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

fn prefix_slice(tensor: &Tensor, axis: usize, len: usize) -> anyhow::Result<Tensor> {
    let axis_len = *tensor
        .shape
        .get(axis)
        .context("native KV slice axis out of bounds")?;
    if len > axis_len {
        bail!("native KV slice length {len} exceeds axis length {axis_len}");
    }

    let inner = tensor.shape[axis + 1..].iter().product::<usize>();
    let outer = tensor.shape[..axis].iter().product::<usize>();
    let elem_bytes = tensor
        .dtype
        .checked_storage_bytes(1)
        .context("native KV dtype has no fixed storage size")?;
    let source_stride = axis_len * inner * elem_bytes;
    let kept_stride = len * inner * elem_bytes;
    let source = tensor.as_bytes();
    let mut bytes = Vec::with_capacity(outer * kept_stride);
    for index in 0..outer {
        let start = index * source_stride;
        bytes.extend_from_slice(&source[start..start + kept_stride]);
    }
    let mut shape = tensor.shape.clone();
    shape[axis] = len;
    Tensor::from_raw(tensor.dtype, shape, &bytes).context("create sliced native KV tensor")
}

fn diagnose_native_failure(session: &InferenceSession, error: &str) -> String {
    if error.contains("f32 kernel input requires Float32, got Int64") {
        for (_, node) in session.graph().nodes.iter() {
            if node.op_type == "Gather"
                && let Some(data) = node.inputs.first().copied().flatten()
                && session.graph().value(data).dtype == DataType::Int64
            {
                return " (native CPU Gather lacks Int64 data support)".to_string();
            }
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_metadata::{KvOwnership, ModelIoSpec, SequenceInputKind};
    use onnx_runtime_ir::{Attribute, Graph, Node, NodeId, Shape, SymbolId, TensorData};
    use prost::Message;

    #[test]
    fn graph_capture_auto_enables_for_owned_cuda_kv() {
        let structural = GraphCaptureStructuralSafety {
            device_is_cuda: true,
            kv_ownership: KvOwnership::Owned,
        };
        assert!(structural.is_capture_safe());
        // Env unset, no programmatic override -> auto-enable from structure.
        assert!(resolve_graph_capture_enabled(
            None, false, false, structural
        ));
    }

    #[test]
    fn graph_capture_auto_declines_for_non_owned_or_non_cuda() {
        let shared = GraphCaptureStructuralSafety {
            device_is_cuda: true,
            kv_ownership: KvOwnership::Shared,
        };
        assert!(!shared.is_capture_safe());
        assert!(!resolve_graph_capture_enabled(None, false, false, shared));

        let cpu_owned = GraphCaptureStructuralSafety {
            device_is_cuda: false,
            kv_ownership: KvOwnership::Owned,
        };
        assert!(!cpu_owned.is_capture_safe());
        assert!(!resolve_graph_capture_enabled(
            None, false, false, cpu_owned
        ));
    }

    #[test]
    fn graph_capture_env_explicit_overrides_auto_decision() {
        let safe = GraphCaptureStructuralSafety {
            device_is_cuda: true,
            kv_ownership: KvOwnership::Owned,
        };
        let unsafe_structural = GraphCaptureStructuralSafety {
            device_is_cuda: true,
            kv_ownership: KvOwnership::Shared,
        };
        // ONNX_GENAI_CUDA_GRAPH=0 forces OFF even when structurally safe.
        assert!(!resolve_graph_capture_enabled(None, true, false, safe));
        // ONNX_GENAI_CUDA_GRAPH=1 forces ON even when structure would decline
        // (the runtime decline machinery is still the final safety net).
        assert!(resolve_graph_capture_enabled(
            None,
            true,
            true,
            unsafe_structural
        ));
    }

    #[test]
    fn graph_capture_programmatic_override_wins_over_env_and_structure() {
        let safe = GraphCaptureStructuralSafety {
            device_is_cuda: true,
            kv_ownership: KvOwnership::Owned,
        };
        // Programmatic Some(false) beats explicit env=1 and safe structure.
        assert!(!resolve_graph_capture_enabled(
            Some(false),
            true,
            true,
            safe
        ));
        // Programmatic Some(true) beats explicit env=0.
        assert!(resolve_graph_capture_enabled(Some(true), true, false, safe));
    }

    #[test]
    fn capture_fallback_emits_each_structured_decline_to_tracer() {
        let report = CaptureDeclineReport {
            entries: vec![onnx_runtime_session::CaptureDecline {
                node_id: Some(12),
                op_type: "GroupQueryAttention".to_string(),
                domain: "com.microsoft".to_string(),
                reason:
                    "requires warmed f32 q_seq==1 k_seq==1 fixed-capacity device-KV reference path"
                        .to_string(),
            }],
        };
        let (trace, events) = TraceContext::in_memory();

        trace_capture_declines(&trace, &report);

        let events = events.events();
        assert_eq!(events.len(), 1);
        let args = events[0].args.as_ref().unwrap();
        assert_eq!(args[onnx_runtime_tracer::ARG_CAPTURE_REJECTED_NODE], 12);
        assert_eq!(
            args[onnx_runtime_tracer::ARG_CAPTURE_REJECTED_OP],
            "GroupQueryAttention"
        );
        assert_eq!(
            args[onnx_runtime_tracer::ARG_CAPTURE_REJECTED_REASON],
            report.entries[0].reason
        );
    }

    fn insert_op(
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

    fn tiny_decoder(last_token_logits: bool) -> InferenceSession {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 11);
        let batch = graph.intern_symbol("batch");
        let sequence = graph.intern_symbol("sequence");
        let total = graph.intern_symbol("total");
        let past = graph.intern_symbol("past");
        let shape = |dims: &[Dim]| -> Shape { dims.to_vec() };

        let input_ids = graph.create_named_value(
            "input_ids",
            DataType::Int64,
            shape(&[batch.into(), sequence.into()]),
        );
        let attention_mask = graph.create_named_value(
            "attention_mask",
            DataType::Int64,
            shape(&[batch.into(), total.into()]),
        );
        let position_ids = graph.create_named_value(
            "position_ids",
            DataType::Int64,
            shape(&[batch.into(), sequence.into()]),
        );
        let past_key = graph.create_named_value(
            "past_key_values.0.key",
            DataType::Float32,
            shape(&[batch.into(), 1.into(), past.into(), 1.into()]),
        );
        let past_value = graph.create_named_value(
            "past_key_values.0.value",
            DataType::Float32,
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

        let cast = graph.create_value(DataType::Float32, shape(&[batch.into(), sequence.into()]));
        insert_op(
            &mut graph,
            "Cast",
            vec![input_ids],
            cast,
            &[("to", Attribute::Int(1))],
        );
        let current_kv = graph.create_value(
            DataType::Float32,
            shape(&[batch.into(), 1.into(), sequence.into(), 1.into()]),
        );
        insert_op(
            &mut graph,
            "Unsqueeze",
            vec![cast],
            current_kv,
            &[("axes", Attribute::Ints(vec![1, 3]))],
        );

        let logits = if last_token_logits {
            let logits = graph.create_named_value(
                "logits",
                DataType::Float32,
                shape(&[1.into(), 1.into(), 2.into()]),
            );
            let data = [10.0f32, 20.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            insert_op(
                &mut graph,
                "Constant",
                vec![],
                logits,
                &[(
                    "value",
                    Attribute::Tensor(TensorData::from_raw(DataType::Float32, vec![1, 1, 2], data)),
                )],
            );
            logits
        } else {
            let logits = graph.create_named_value(
                "logits",
                DataType::Float32,
                shape(&[batch.into(), sequence.into(), 1.into()]),
            );
            insert_op(
                &mut graph,
                "Unsqueeze",
                vec![cast],
                logits,
                &[("axes", Attribute::Ints(vec![2]))],
            );
            logits
        };
        let present_key = graph.create_named_value(
            "present.0.key",
            DataType::Float32,
            shape(&[batch.into(), 1.into(), total.into(), 1.into()]),
        );
        insert_op(
            &mut graph,
            "Concat",
            vec![past_key, current_kv],
            present_key,
            &[("axis", Attribute::Int(2))],
        );
        let present_value = graph.create_named_value(
            "present.0.value",
            DataType::Float32,
            shape(&[batch.into(), 1.into(), total.into(), 1.into()]),
        );
        insert_op(
            &mut graph,
            "Concat",
            vec![past_value, current_kv],
            present_value,
            &[("axis", Attribute::Int(2))],
        );
        for output in [logits, present_key, present_value] {
            graph.add_output(output);
        }
        InferenceSession::from_graph(graph).expect("build tiny decoder")
    }

    fn session_from_graph(graph: Graph) -> InferenceSession {
        let bytes = onnx_std::Model::new(graph)
            .to_proto()
            .expect("serialize ONNX model")
            .encode_to_vec();
        InferenceSession::builder()
            .model_bytes(&bytes)
            .build()
            .expect("load ONNX model")
    }

    fn proposer_io(sequence_source: SequenceInputKind, kv_ownership: KvOwnership) -> ModelIoSpec {
        ModelIoSpec {
            sequence_source: Some(sequence_source),
            kv_ownership: Some(kv_ownership),
            token_input: (sequence_source == SequenceInputKind::TokenIds)
                .then(|| "input_ids".into()),
            inputs_embeds_input: (sequence_source == SequenceInputKind::InputsEmbeds)
                .then(|| "embeddings".into()),
            attention_mask_input: Some("mask".into()),
            position_ids_input: Some("positions".into()),
            logits_output: Some("draft_scores".into()),
            hidden_output: (sequence_source == SequenceInputKind::InputsEmbeds)
                .then(|| "next_state".into()),
            kv_inputs: (kv_ownership == KvOwnership::Owned)
                .then(|| vec!["cache_key".into(), "cache_value".into()]),
            kv_outputs: (kv_ownership == KvOwnership::Owned)
                .then(|| vec!["next_key".into(), "next_value".into()]),
            encoder_hidden_states_input: None,
            cross_kv_inputs: None,
            cross_kv_outputs: None,
            kv_update: None,
            state_pairs: None,
        }
    }

    fn tiny_owned_kv_proposer() -> InferenceSession {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 11);
        let batch = graph.intern_symbol("batch");
        let sequence = graph.intern_symbol("sequence");
        let total = graph.intern_symbol("total");
        let past = graph.intern_symbol("past");
        let input_ids = graph.create_named_value(
            "input_ids",
            DataType::Int64,
            vec![batch.into(), sequence.into()],
        );
        let mask =
            graph.create_named_value("mask", DataType::Int64, vec![batch.into(), total.into()]);
        let positions = graph.create_named_value(
            "positions",
            DataType::Int64,
            vec![batch.into(), sequence.into()],
        );
        let cache_key = graph.create_named_value(
            "cache_key",
            DataType::Float32,
            vec![batch.into(), 1.into(), past.into(), 1.into()],
        );
        let cache_value = graph.create_named_value(
            "cache_value",
            DataType::Float32,
            vec![batch.into(), 1.into(), past.into(), 1.into()],
        );
        for input in [input_ids, mask, positions, cache_key, cache_value] {
            graph.add_input(input);
        }
        let cast = graph.create_named_value(
            "token_values",
            DataType::Float32,
            vec![batch.into(), sequence.into()],
        );
        insert_op(
            &mut graph,
            "Cast",
            vec![input_ids],
            cast,
            &[("to", Attribute::Int(DataType::Float32 as i64))],
        );
        let draft_scores = graph.create_named_value(
            "draft_scores",
            DataType::Float32,
            vec![batch.into(), sequence.into(), 1.into()],
        );
        insert_op(
            &mut graph,
            "Unsqueeze",
            vec![cast],
            draft_scores,
            &[("axes", Attribute::Ints(vec![2]))],
        );
        let current = graph.create_named_value(
            "current_cache",
            DataType::Float32,
            vec![batch.into(), 1.into(), sequence.into(), 1.into()],
        );
        insert_op(
            &mut graph,
            "Unsqueeze",
            vec![cast],
            current,
            &[("axes", Attribute::Ints(vec![1, 3]))],
        );
        let next_key = graph.create_named_value(
            "next_key",
            DataType::Float32,
            vec![batch.into(), 1.into(), total.into(), 1.into()],
        );
        insert_op(
            &mut graph,
            "Concat",
            vec![cache_key, current],
            next_key,
            &[("axis", Attribute::Int(2))],
        );
        let next_value = graph.create_named_value(
            "next_value",
            DataType::Float32,
            vec![batch.into(), 1.into(), total.into(), 1.into()],
        );
        insert_op(
            &mut graph,
            "Concat",
            vec![cache_value, current],
            next_value,
            &[("axis", Attribute::Int(2))],
        );
        for output in [draft_scores, next_key, next_value] {
            graph.add_output(output);
        }
        session_from_graph(graph)
    }

    fn tiny_shared_kv_embed_proposer(width: usize) -> InferenceSession {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 13);
        let batch = graph.intern_symbol("batch");
        let sequence = graph.intern_symbol("sequence");
        let kv_len = graph.intern_symbol("kv_len");
        let embeddings = graph.create_named_value(
            "embeddings",
            DataType::Float32,
            vec![batch.into(), sequence.into(), width.into()],
        );
        let mask =
            graph.create_named_value("mask", DataType::Int64, vec![batch.into(), kv_len.into()]);
        let positions = graph.create_named_value(
            "positions",
            DataType::Int64,
            vec![batch.into(), sequence.into()],
        );
        let shared_key = graph.create_named_value(
            "external.key",
            DataType::Float32,
            vec![batch.into(), 1.into(), kv_len.into(), width.into()],
        );
        let shared_value = graph.create_named_value(
            "external.value",
            DataType::Float32,
            vec![batch.into(), 1.into(), kv_len.into(), width.into()],
        );
        for input in [embeddings, mask, positions, shared_key, shared_value] {
            graph.add_input(input);
        }
        let draft_scores = graph.create_named_value(
            "draft_scores",
            DataType::Float32,
            vec![batch.into(), sequence.into(), width.into()],
        );
        insert_op(&mut graph, "Identity", vec![embeddings], draft_scores, &[]);
        let next_state = graph.create_named_value(
            "next_state",
            DataType::Float32,
            vec![batch.into(), sequence.into(), width.into()],
        );
        insert_op(&mut graph, "Identity", vec![embeddings], next_state, &[]);
        graph.add_output(draft_scores);
        graph.add_output(next_state);
        session_from_graph(graph)
    }

    #[cfg(feature = "cuda")]
    #[derive(Clone, Copy)]
    enum AuxOutput {
        /// Static `[1, 1]` auxiliary output produced by `Cast(input_ids)`. The
        /// original capture-safe smoke case: no symbolic dims to reason about.
        StaticUnit,
        /// `[batch, 1]` auxiliary output whose leading dim is a *genuine*
        /// symbolic `batch` dim shared with `input_ids`. It resolves to `1` at
        /// decode, so it is structurally a decode unit and capture must still
        /// succeed with the output persistently bound (collapsed to `[1, 1]`).
        SymbolicBatch,
        /// `[1, total_seq]` auxiliary output produced by `Cast(attention_mask)`,
        /// whose trailing dim is the symbolic `total_seq` dim. That dim grows
        /// with the sequence and is NOT batch/query-seq, so F1 must decline to
        /// persistently bind it (collapsing to `[1, 1]` would under-allocate)
        /// and fall back to eager, where the executor JIT-sizes it each step.
        SymbolicTotalSeq,
    }

    #[cfg(feature = "cuda")]
    fn capture_safe_cuda_decoder(
        graph_capture: bool,
        max_len: usize,
    ) -> anyhow::Result<NativeDecodeSession> {
        build_cuda_decoder(graph_capture, max_len, AuxOutput::StaticUnit)
    }

    #[cfg(feature = "cuda")]
    fn build_cuda_decoder(
        graph_capture: bool,
        max_len: usize,
        aux: AuxOutput,
    ) -> anyhow::Result<NativeDecodeSession> {
        use prost::Message;

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 13);
        let batch = graph.intern_symbol("batch");
        let total = graph.intern_symbol("total");
        let past = graph.intern_symbol("past");

        // input_ids declares a symbolic batch dim for the SymbolicBatch case so
        // the aux output can legitimately *share* it; it is bound to `[1, 1]` at
        // decode regardless, so the other cases are unaffected.
        let input_ids_shape = match aux {
            AuxOutput::SymbolicBatch => vec![batch.into(), 1.into()],
            _ => vec![1.into(), 1.into()],
        };
        let input_ids = graph.create_named_value("input_ids", DataType::Int64, input_ids_shape);
        let attention_mask = graph.create_named_value(
            "attention_mask",
            DataType::Int64,
            vec![1.into(), total.into()],
        );
        let position_ids =
            graph.create_named_value("position_ids", DataType::Int64, vec![1.into(), 1.into()]);
        let past_key = graph.create_named_value(
            "past_key_values.0.key",
            DataType::Float32,
            vec![1.into(), 1.into(), past.into(), 1.into()],
        );
        let past_value = graph.create_named_value(
            "past_key_values.0.value",
            DataType::Float32,
            vec![1.into(), 1.into(), past.into(), 1.into()],
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

        let logits =
            graph.create_named_value("logits", DataType::Float32, vec![1.into(), 1.into()]);
        insert_op(
            &mut graph,
            "Cast",
            vec![input_ids],
            logits,
            &[("to", Attribute::Int(DataType::Float32 as i64))],
        );
        let present_key = graph.create_named_value(
            "present.0.key",
            DataType::Float32,
            vec![1.into(), 1.into(), past.into(), 1.into()],
        );
        insert_op(
            &mut graph,
            "Cast",
            vec![past_key],
            present_key,
            &[("to", Attribute::Int(DataType::Float32 as i64))],
        );
        let present_value = graph.create_named_value(
            "present.0.value",
            DataType::Float32,
            vec![1.into(), 1.into(), past.into(), 1.into()],
        );
        insert_op(
            &mut graph,
            "Cast",
            vec![past_value],
            present_value,
            &[("to", Attribute::Int(DataType::Float32 as i64))],
        );
        // Auxiliary output geometry drives the F1 structural analysis.
        let (aux_shape, aux_source): (Vec<Dim>, _) = match aux {
            AuxOutput::StaticUnit => (vec![1.into(), 1.into()], input_ids),
            AuxOutput::SymbolicBatch => (vec![batch.into(), 1.into()], input_ids),
            AuxOutput::SymbolicTotalSeq => (vec![1.into(), total.into()], attention_mask),
        };
        let auxiliary = graph.create_named_value("auxiliary_state", DataType::Float32, aux_shape);
        insert_op(
            &mut graph,
            "Cast",
            vec![aux_source],
            auxiliary,
            &[("to", Attribute::Int(DataType::Float32 as i64))],
        );
        for output in [logits, present_key, present_value, auxiliary] {
            graph.add_output(output);
        }

        let model = onnx_std::Model::new(graph).to_proto()?.encode_to_vec();
        let session = InferenceSession::builder()
            .model_bytes(&model)
            .device(DevicePreference::Gpu { index: Some(0) })
            .build()
            .context("build capture-safe CUDA decoder")?;
        NativeDecodeSession::from_session_with_cuda_options(
            session,
            NativeDecodeCudaOptions {
                kv_max_len: Some(max_len),
                graph_capture: Some(graph_capture),
            },
        )
    }

    #[cfg(feature = "cuda")]
    fn binding_addresses(session: &NativeDecodeSession) -> Vec<usize> {
        session
            .cuda
            .as_ref()
            .expect("CUDA state")
            .bindings
            .iter()
            .map(|binding| binding.device_ptr() as usize)
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn input_update_stats(session: &NativeDecodeSession) -> [DeviceBindingTransferStats; 3] {
        let state = session.cuda.as_ref().expect("CUDA state");
        [
            state.bindings[0].transfer_stats(),
            state.bindings[state.input_ids_binding].transfer_stats(),
            state.bindings[state.position_ids_binding.expect("position_ids binding")]
                .transfer_stats(),
        ]
    }

    #[cfg(feature = "cuda")]
    fn assert_single_value_uploads(
        before: [DeviceBindingTransferStats; 3],
        after: [DeviceBindingTransferStats; 3],
    ) {
        for (before, after) in before.into_iter().zip(after) {
            assert_eq!(after.host_upload_calls, before.host_upload_calls + 1);
            assert_eq!(
                after.host_upload_bytes,
                before.host_upload_bytes + std::mem::size_of::<i64>() as u64
            );
        }
    }

    #[cfg(feature = "cuda")]
    fn assert_decode_bindings(
        session: &mut NativeDecodeSession,
        addresses: &[usize],
        token: TokenId,
        position: usize,
        max_len: usize,
    ) -> anyhow::Result<()> {
        assert_eq!(binding_addresses(session), addresses);
        let state = session.cuda.as_mut().expect("CUDA state");

        let input = state.bindings[state.input_ids_binding].read_bytes()?;
        assert_eq!(
            i64::from_le_bytes(input.try_into().expect("one input id")),
            i64::from(token)
        );

        let position_bytes = state.bindings
            [state.position_ids_binding.expect("position_ids binding")]
        .read_bytes()?;
        assert_eq!(
            i64::from_le_bytes(position_bytes.try_into().expect("one position id")),
            position as i64
        );

        let mask = state.bindings[0]
            .read_bytes()?
            .chunks_exact(std::mem::size_of::<i64>())
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(mask.len(), max_len);
        assert!(mask[..=position].iter().all(|&value| value == 1));
        assert!(mask[position + 1..].iter().all(|&value| value == 0));
        assert_eq!(state.bindings[0].logical_shape(), &[1, position + 1]);
        for binding in &state.bindings[state.kv_binding_range.clone()] {
            assert_eq!(binding.logical_shape()[2], position + 1);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn run_capture_safe_decode(
        session: &mut NativeDecodeSession,
        tokens: &[TokenId],
        addresses: &[usize],
        max_len: usize,
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        let mut logits = Vec::with_capacity(tokens.len());
        for (position, &token) in tokens.iter().enumerate() {
            let before = input_update_stats(session);
            let step = session.decode(&[token], position)?;
            let after = input_update_stats(session);
            assert_single_value_uploads(before, after);
            assert_decode_bindings(session, addresses, token, position, max_len)?;
            logits.push(
                step.into_iter()
                    .flatten()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
            );
        }
        Ok(logits)
    }

    #[test]
    fn native_decode_advances_kv_and_rewinds() {
        let mut session =
            NativeDecodeSession::from_session(tiny_decoder(false)).expect("load decoder");
        let logits = session.decode(&[1, 2, 3], 0).expect("prefill");
        assert_eq!(logits.len(), 3);
        assert_eq!(logits[0].len(), 1);
        assert_eq!(session.current_len(), 3);

        let logits = session.decode(&[4], 3).expect("decode");
        assert_eq!(logits.len(), 1);
        assert_eq!(logits[0].len(), 1);
        assert_eq!(session.current_len(), 4);

        session.rewind(2).expect("rewind");
        assert_eq!(session.current_len(), 2);
        session.decode(&[5], 2).expect("decode after rewind");
        assert_eq!(session.current_len(), 3);
    }

    #[test]
    fn native_proposer_defaults_preserve_token_ids_and_owned_kv() {
        let mut io = proposer_io(SequenceInputKind::TokenIds, KvOwnership::Owned);
        io.sequence_source = None;
        io.kv_ownership = None;
        let mut proposer = NativeProposerSession::from_session(tiny_owned_kv_proposer(), Some(&io))
            .expect("load token proposer");
        let first = proposer.step_token_ids(&[2, 4]).expect("first proposal");
        assert_eq!(first.logits, Some(vec![vec![2.0], vec![4.0]]));
        assert_eq!(first.projected_state, None);
        let second = proposer.step_token_ids(&[7]).expect("second proposal");
        assert_eq!(second.logits, Some(vec![vec![7.0]]));
        assert_eq!(proposer.current_len, 3);
    }

    #[test]
    fn native_proposer_runs_inputs_embeds_with_shared_kv_and_output_roles() {
        let width = 3;
        let io = proposer_io(SequenceInputKind::InputsEmbeds, KvOwnership::Shared);
        let mut proposer =
            NativeProposerSession::from_session(tiny_shared_kv_embed_proposer(width), Some(&io))
                .expect("load embedding proposer");
        let key = Tensor::from_f32(&[1, 1, 2, width], &[0.0; 6]).expect("shared key");
        let value = Tensor::from_f32(&[1, 1, 2, width], &[1.0; 6]).expect("shared value");
        let inputs = vec![
            ("external.key".to_string(), key),
            ("external.value".to_string(), value),
        ];
        let embeddings = [1.0, 2.0, 3.0];
        let output = proposer
            .step_inputs_embeds(&embeddings, 5, &inputs)
            .expect("shared-KV proposal");
        assert_eq!(output.logits, Some(vec![embeddings.to_vec()]));
        assert_eq!(output.projected_state, Some(embeddings.to_vec()));
        assert_eq!(proposer.current_len, 0, "shared KV is target-owned");
    }

    #[test]
    fn native_decode_verify_then_rewind_matches_fresh_decode() {
        // WP1 exit criterion (CPU logic coverage): verify K tokens, rewind to the
        // committed length, and prove a subsequent decode is bit-identical to a
        // fresh decode from the same committed prefix (no KV corruption). The
        // device-KV bit-identity variant is `native_cuda_verify_rewind_no_kv_corruption`.
        let mut session =
            NativeDecodeSession::from_session(tiny_decoder(false)).expect("load decoder");
        let prompt = [1, 2, 3];
        session.decode(&prompt, 0).expect("prefill");
        let past = session.current_len();
        assert_eq!(past, prompt.len());

        // Verify a K-token draft via the verify primitive: returns [K, vocab].
        let draft = [4, 5, 6];
        let rows = session.decode_verify(&draft, past).expect("verify");
        assert_eq!(rows.len(), draft.len());
        assert_eq!(rows[0].len(), 1);
        assert_eq!(session.current_len(), past + draft.len());

        // Accept j of the draft, rewind device/host KV to the committed length.
        let j = 1;
        session.rewind(past + j).expect("rewind");
        assert_eq!(session.current_len(), past + j);

        // Subsequent decode from the committed prefix.
        let feed = 9;
        let after = session
            .decode(&[feed], past + j)
            .expect("decode after rewind");

        // Fresh session decoded over the committed prefix prompt ++ draft[..j].
        let mut fresh =
            NativeDecodeSession::from_session(tiny_decoder(false)).expect("fresh decoder");
        let mut committed = prompt.to_vec();
        committed.extend_from_slice(&draft[..j]);
        fresh.decode(&committed, 0).expect("fresh prefill");
        let fresh_after = fresh
            .decode(&[feed], committed.len())
            .expect("fresh decode");

        let after_bits = after
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        let fresh_bits = fresh_after
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            after_bits, fresh_bits,
            "verify+rewind diverged from fresh decode"
        );
    }

    #[test]
    fn native_decode_verify_requires_matching_past_and_nonempty_draft() {
        let mut session =
            NativeDecodeSession::from_session(tiny_decoder(false)).expect("load decoder");
        session.decode(&[1, 2], 0).expect("prefill");
        assert!(
            session
                .decode_verify(&[], 2)
                .expect_err("empty draft must fail")
                .to_string()
                .contains("at least one draft token")
        );
        assert!(
            session
                .decode_verify(&[3], 5)
                .expect_err("past mismatch must fail")
                .to_string()
                .contains("past length mismatch")
        );
    }

    #[test]
    fn native_decode_option_c_scaffolding_is_dormant_by_default() {
        // The padded M=maxK capture + retain-graph-on-rewind switches (option (c))
        // must stay dormant. On a CPU session (no CUDA state) the controls are
        // inert no-ops and the capacity stays `None`.
        let mut session =
            NativeDecodeSession::from_session(tiny_decoder(false)).expect("load decoder");
        assert_eq!(session.padded_query_capacity(), None);
        session.set_retain_graph_on_rewind(true);
        session.configure_padded_verify_capture(8);
        assert_eq!(session.padded_query_capacity(), None);
    }

    #[test]
    fn persistent_auxiliary_output_shape_is_fixed_and_rejects_strings() {
        let shape = DecodeCudaState::persistent_output_shape(
            "auxiliary_state",
            DataType::Float16,
            &[Dim::Symbolic(SymbolId(0)), Dim::Static(1536)],
        )
        .expect("numeric auxiliary output must be bindable");
        assert_eq!(shape, [1, 1536]);

        let error = DecodeCudaState::persistent_output_shape(
            "auxiliary_text",
            DataType::String,
            &[Dim::Static(1)],
        )
        .expect_err("variable-width auxiliary output must fail explicitly");
        let message = error.to_string();
        assert!(message.contains("auxiliary_text"));
        assert!(message.contains("fixed-size device tensor storage"));
        assert!(message.contains("export this output as a numeric tensor"));
    }

    #[test]
    fn unit_symbol_collection_is_structural_and_batch_aware() {
        // input_ids / position_ids are bound to `[1, 1]` at decode, so *every*
        // symbolic axis is a decode-unit (batch or query-seq). attention_mask
        // and past-KV grow along their sequence axis, so only axis 0 (batch) is
        // a unit; the total_seq / past symbols must NOT be collected.
        let batch = SymbolId(0);
        let query_seq = SymbolId(1);
        let total = SymbolId(2);
        let past = SymbolId(3);

        let mut unit = HashSet::new();
        // input_ids: [batch, query_seq]
        DecodeCudaState::collect_unit_symbols(
            &[Dim::Symbolic(batch), Dim::Symbolic(query_seq)],
            false,
            &mut unit,
        );
        // attention_mask: [batch, total] — batch only.
        DecodeCudaState::collect_unit_symbols(
            &[Dim::Symbolic(batch), Dim::Symbolic(total)],
            true,
            &mut unit,
        );
        // past-KV: [batch, heads, past, head_dim] — batch only.
        DecodeCudaState::collect_unit_symbols(
            &[
                Dim::Symbolic(batch),
                Dim::Static(4),
                Dim::Symbolic(past),
                Dim::Static(8),
            ],
            true,
            &mut unit,
        );

        assert!(unit.contains(&batch));
        assert!(unit.contains(&query_seq));
        assert!(
            !unit.contains(&total),
            "total_seq must not be a decode unit"
        );
        assert!(!unit.contains(&past), "past must not be a decode unit");
    }

    #[test]
    fn unresolved_symbolic_axis_flags_only_non_unit_symbols() {
        let batch = SymbolId(0);
        let query_seq = SymbolId(1);
        let total = SymbolId(2);
        let unit = HashSet::from([batch, query_seq]);

        // Fully static aux output: always bindable.
        assert_eq!(
            DecodeCudaState::unresolved_symbolic_axis(&[Dim::Static(1), Dim::Static(1536)], &unit),
            None
        );
        // Symbolic dim that IS a decode unit (batch): safe to collapse to 1.
        assert_eq!(
            DecodeCudaState::unresolved_symbolic_axis(
                &[Dim::Symbolic(batch), Dim::Static(1536)],
                &unit
            ),
            None
        );
        // Symbolic dim that is NOT batch/query-seq (an accumulator indexed by
        // total_seq): flagged with its axis and symbol so decode declines to
        // persistently bind it.
        assert_eq!(
            DecodeCudaState::unresolved_symbolic_axis(
                &[Dim::Static(1), Dim::Symbolic(total), Dim::Static(64)],
                &unit
            ),
            Some((1, total))
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_cuda_capture_replay_is_bit_exact_and_refreshes_decode_inputs() -> anyhow::Result<()> {
        if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
            eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
            return Ok(());
        }

        const MAX_LEN: usize = 16;
        const TOKENS: [TokenId; 10] = [3, 17, 5, 29, 11, 23, 7, 31, 13, 2];

        let mut eager = capture_safe_cuda_decoder(false, MAX_LEN)?;
        let eager_addresses = binding_addresses(&eager);
        let eager_first = run_capture_safe_decode(&mut eager, &TOKENS, &eager_addresses, MAX_LEN)?;
        let eager_stats = eager.cuda_kv_debug_stats().expect("CUDA stats");
        assert!(!eager_stats.graph.enabled);
        assert_eq!(eager_stats.graph.captures, 0);
        assert_eq!(eager_stats.graph.replays, 0);
        assert_eq!(eager_stats.graph.fallbacks, 0);
        assert!(eager.cuda_graph_fallback_reason().is_none());

        let mut captured = capture_safe_cuda_decoder(true, MAX_LEN)?;
        {
            let state = captured.cuda.as_ref().expect("CUDA state");
            assert_eq!(state.auxiliary_binding_range.len(), 1);
            assert_eq!(
                state.bindings[state.auxiliary_binding_range.start].output_name(),
                Some("auxiliary_state")
            );
            assert!(state.auxiliary_binding_range.end <= state.base_binding_count);
        }
        let captured_addresses = binding_addresses(&captured);
        let captured_first =
            run_capture_safe_decode(&mut captured, &TOKENS, &captured_addresses, MAX_LEN)?;
        let first_stats = captured.cuda_kv_debug_stats().expect("CUDA stats");
        assert!(first_stats.graph.enabled);
        assert_eq!(first_stats.graph.captures, 1);
        assert_eq!(first_stats.graph.replays, TOKENS.len() as u64 - 2);
        assert_eq!(first_stats.graph.fallbacks, 0);
        assert!(captured.cuda_graph_fallback_reason().is_none());
        assert_eq!(captured_first, eager_first);
        assert_eq!(
            captured_first,
            TOKENS
                .iter()
                .map(|&token| vec![(token as f32).to_bits()])
                .collect::<Vec<_>>()
        );
        assert_eq!(captured_addresses, binding_addresses(&captured));
        assert_eq!(
            first_stats.kv_transfers,
            DeviceBindingTransferStats::default()
        );

        eager.reset()?;
        captured.reset()?;
        let eager_second = run_capture_safe_decode(&mut eager, &TOKENS, &eager_addresses, MAX_LEN)?;
        let captured_second =
            run_capture_safe_decode(&mut captured, &TOKENS, &captured_addresses, MAX_LEN)?;
        let second_stats = captured.cuda_kv_debug_stats().expect("CUDA stats");
        assert_eq!(captured_second, eager_second);
        assert_eq!(captured_second, captured_first);
        assert_eq!(second_stats.graph.captures, 2);
        assert_eq!(second_stats.graph.replays, 2 * (TOKENS.len() as u64 - 2));
        assert_eq!(second_stats.graph.fallbacks, 0);
        assert_eq!(captured_addresses, binding_addresses(&captured));
        assert_eq!(
            second_stats.kv_transfers,
            DeviceBindingTransferStats::default()
        );

        eprintln!(
            "native CUDA capture-safe decode parity: captures={} replays={} fallbacks={} steps_per_generation={}",
            second_stats.graph.captures,
            second_stats.graph.replays,
            second_stats.graph.fallbacks,
            TOKENS.len()
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_cuda_symbolic_batch_aux_captures_bit_exact() -> anyhow::Result<()> {
        // F2 positive case: an auxiliary output with a *genuinely symbolic* dim
        // (`batch`) that resolves to 1 at decode must remain persistently
        // bindable and fully capturable — capture succeeds, no fallback, and
        // replay is bit-exact with the eager device path.
        if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
            eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
            return Ok(());
        }

        const MAX_LEN: usize = 16;
        const TOKENS: [TokenId; 8] = [3, 17, 5, 29, 11, 23, 7, 31];

        let mut eager = build_cuda_decoder(false, MAX_LEN, AuxOutput::SymbolicBatch)?;
        let eager_addresses = binding_addresses(&eager);
        let eager_logits = run_capture_safe_decode(&mut eager, &TOKENS, &eager_addresses, MAX_LEN)?;

        let mut captured = build_cuda_decoder(true, MAX_LEN, AuxOutput::SymbolicBatch)?;
        {
            let state = captured.cuda.as_ref().expect("CUDA state");
            // The symbolic-batch aux output is structurally a decode unit, so it
            // is persistently bound (collapsed to [1, 1]) — F1 does NOT decline.
            assert!(state.graph_enabled);
            assert!(captured.cuda_auxiliary_bind_declines().is_empty());
            let state = captured.cuda.as_ref().unwrap();
            assert_eq!(state.auxiliary_binding_range.len(), 1);
            assert_eq!(
                state.bindings[state.auxiliary_binding_range.start].output_name(),
                Some("auxiliary_state")
            );
        }
        let captured_addresses = binding_addresses(&captured);
        let captured_logits =
            run_capture_safe_decode(&mut captured, &TOKENS, &captured_addresses, MAX_LEN)?;
        let stats = captured.cuda_kv_debug_stats().expect("CUDA stats");
        assert!(stats.graph.enabled);
        assert_eq!(stats.graph.captures, 1);
        assert_eq!(stats.graph.replays, TOKENS.len() as u64 - 2);
        assert_eq!(stats.graph.fallbacks, 0);
        assert!(captured.cuda_graph_fallback_reason().is_none());
        assert_eq!(captured_logits, eager_logits);
        assert_eq!(
            captured_logits,
            TOKENS
                .iter()
                .map(|&token| vec![(token as f32).to_bits()])
                .collect::<Vec<_>>()
        );
        assert_eq!(captured_addresses, binding_addresses(&captured));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_cuda_symbolic_total_seq_aux_declines_capture_but_decodes_eagerly()
    -> anyhow::Result<()> {
        // F2 negative case (the F1 path): an auxiliary output whose symbolic dim
        // is `total_seq` — NOT batch/query-seq — cannot be collapsed to a fixed
        // persistent binding without under-allocating. F1 must decline capture
        // at binding time (leaving the output unbound), yet decode MUST still
        // work via the eager device path, producing correct output.
        if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
            eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
            return Ok(());
        }

        const MAX_LEN: usize = 16;
        const TOKENS: [TokenId; 8] = [3, 17, 5, 29, 11, 23, 7, 31];

        // Request graph capture; F1 must decline it structurally.
        let mut declined = build_cuda_decoder(true, MAX_LEN, AuxOutput::SymbolicTotalSeq)?;
        {
            let state = declined.cuda.as_ref().expect("CUDA state");
            assert!(
                !state.graph_enabled,
                "F1 must disable capture for an unresolved-symbolic aux output"
            );
            // The unbindable aux output is left out of the persistent bindings.
            assert_eq!(state.auxiliary_binding_range.len(), 0);
        }
        let declines = declined.cuda_auxiliary_bind_declines();
        assert_eq!(declines.len(), 1);
        assert!(declines[0].contains("auxiliary_state"));
        assert!(declines[0].contains("total"));
        assert!(declines[0].contains("not structurally batch or query-seq"));

        let addresses = binding_addresses(&declined);
        let declined_logits = run_capture_safe_decode(&mut declined, &TOKENS, &addresses, MAX_LEN)?;
        let stats = declined.cuda_kv_debug_stats().expect("CUDA stats");
        assert!(!stats.graph.enabled);
        assert_eq!(stats.graph.captures, 0);
        assert_eq!(stats.graph.replays, 0);
        assert_eq!(stats.graph.fallbacks, 0);

        // Decode is bit-exact with a plain eager (graph_capture=false) run — the
        // unresolved aux output changes nothing about the decode result.
        let mut eager = build_cuda_decoder(false, MAX_LEN, AuxOutput::SymbolicTotalSeq)?;
        let eager_addresses = binding_addresses(&eager);
        let eager_logits = run_capture_safe_decode(&mut eager, &TOKENS, &eager_addresses, MAX_LEN)?;
        assert_eq!(declined_logits, eager_logits);
        assert_eq!(
            declined_logits,
            TOKENS
                .iter()
                .map(|&token| vec![(token as f32).to_bits()])
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn native_decode_accepts_last_token_only_logits_and_advances_kv() {
        let mut session =
            NativeDecodeSession::from_session(tiny_decoder(true)).expect("load decoder");

        let logits = session.decode(&[1, 2, 3], 0).expect("prefill");
        assert_eq!(logits, vec![vec![10.0, 20.0]]);
        assert_eq!(session.current_len(), 3);

        let logits = session.decode(&[4], 3).expect("decode");
        assert_eq!(logits, vec![vec![10.0, 20.0]]);
        assert_eq!(session.current_len(), 4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_cuda_qwen_decode_matches_cpu_tokens() -> anyhow::Result<()> {
        if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
            eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
            return Ok(());
        }

        let model_dir = Path::new("/home/justinchu/qwen2.5-0.5b-int4-onnx");
        if !model_dir.join("model.onnx").is_file() {
            eprintln!("skipping CUDA smoke; target model is not installed");
            return Ok(());
        }
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;
        let prompt = tokenizer.encode("Hello")?;
        const HORIZON: usize = 64;
        let generate = |session: &mut NativeDecodeSession| -> anyhow::Result<(Vec<TokenId>, u128)> {
            let mut logits = session
                .decode(&prompt, 0)?
                .pop()
                .context("prefill must produce logits")?;
            let mut tokens = Vec::with_capacity(HORIZON);
            let mut decode_nanos = 0u128;
            for step in 0..HORIZON {
                let token = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .map(|(index, _)| index as TokenId)
                    .context("logits must not be empty")?;
                tokens.push(token);
                if step + 1 == HORIZON {
                    break;
                }
                let start = std::time::Instant::now();
                logits = session
                    .decode(&[token], session.current_len())?
                    .pop()
                    .context("decode must produce logits")?;
                decode_nanos += start.elapsed().as_nanos();
            }
            Ok((tokens, decode_nanos))
        };

        let mut cpu =
            NativeDecodeSession::load(model_dir.join("model.onnx"), NativeDecodeDevice::Cpu)?;
        let (cpu_tokens, _) = generate(&mut cpu)?;
        drop(cpu);

        let mut eager = NativeDecodeSession::load_with_cuda_options(
            model_dir.join("model.onnx"),
            NativeDecodeDevice::Cuda { index: Some(0) },
            NativeDecodeCudaOptions {
                kv_max_len: Some(128),
                graph_capture: Some(false),
            },
        )?;
        let eager_before = eager
            .cuda_kv_debug_stats()
            .context("CUDA session must expose device KV stats")?;
        let (eager_tokens, eager_nanos) = generate(&mut eager)?;
        let eager_after = eager.cuda_kv_debug_stats().unwrap();
        assert!(!eager_after.graph.enabled);
        assert_eq!(eager_after.graph.captures, 0);
        assert_eq!(eager_after.graph.replays, 0);
        drop(eager);

        let mut captured = NativeDecodeSession::load_with_cuda_options(
            model_dir.join("model.onnx"),
            NativeDecodeDevice::Cuda { index: Some(0) },
            NativeDecodeCudaOptions {
                kv_max_len: Some(128),
                graph_capture: Some(true),
            },
        )?;
        let captured_before = captured.cuda_kv_debug_stats().unwrap();
        let (captured_tokens, captured_nanos) = generate(&mut captured)?;
        let captured_after = captured.cuda_kv_debug_stats().unwrap();

        assert_eq!(cpu_tokens.len(), HORIZON);
        assert_eq!(eager_tokens, cpu_tokens);
        assert_eq!(captured_tokens, eager_tokens);
        assert_eq!(
            &cpu_tokens[..8],
            &[11576, 42740, 11, 358, 614, 264, 3405, 911]
        );
        assert_eq!(eager_before.device_ptrs, eager_after.device_ptrs);
        assert_eq!(captured_before.device_ptrs, captured_after.device_ptrs);
        assert_eq!(
            captured_after.kv_transfers,
            DeviceBindingTransferStats::default()
        );
        assert!(captured_after.graph.enabled);
        assert_eq!(captured_after.graph.captures, 1);
        assert_eq!(captured_after.graph.replays, HORIZON as u64 - 2);
        assert_eq!(captured_after.graph.fallbacks, 0);
        assert!(captured.cuda_graph_fallback_reason().is_none());

        let eager_us = eager_nanos as f64 / (HORIZON - 1) as f64 / 1000.0;
        let captured_us = captured_nanos as f64 / (HORIZON - 1) as f64 / 1000.0;
        eprintln!(
            "native CUDA decode wall-time: eager={eager_us:.1} us/token, graph-flag={captured_us:.1} us/token, delta={:.1}%",
            (captured_us / eager_us - 1.0) * 100.0
        );

        captured.rewind(captured_after.logical_len - 2)?;
        let rewound = captured.cuda_kv_debug_stats().unwrap();
        assert_eq!(rewound.logical_len, captured_after.logical_len - 2);
        assert_eq!(rewound.device_ptrs, captured_before.device_ptrs);
        assert_eq!(rewound.kv_transfers, DeviceBindingTransferStats::default());

        captured.reset()?;
        let (second_tokens, _) = generate(&mut captured)?;
        assert_eq!(second_tokens, captured_tokens);

        captured.reset()?;
        let error = captured
            .decode(&vec![0; 129], 0)
            .expect_err("decode beyond configured KV capacity must fail");
        assert!(error.to_string().contains("CUDA KV capacity exceeded"));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_cuda_verify_rewind_no_kv_corruption() -> anyhow::Result<()> {
        // WP1 exit criterion on real device KV: decode K draft tokens through the
        // eager M=K verify primitive, rewind to the committed length (past+j), and
        // prove a subsequent M=1 decode is BIT-IDENTICAL to a fresh M=1 decode from
        // the same committed prefix. Bit-identity proves the rewind left no stale
        // KV columns attended. Both rewind regimes are exercised: option (b)
        // (invalidate-on-rewind, the default) and the dormant option (c) guard
        // (retain-graph-on-rewind), which must be equally KV-correct.
        if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
            eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
            return Ok(());
        }
        let model_dir = Path::new("/home/justinchu/qwen2.5-0.5b-int4-onnx");
        if !model_dir.join("model.onnx").is_file() {
            eprintln!("skipping CUDA smoke; target model is not installed");
            return Ok(());
        }
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;
        let prompt = tokenizer.encode("The quick brown fox")?;

        let argmax = |row: &[f32]| -> TokenId {
            row.iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index as TokenId)
                .expect("logits row must not be empty")
        };
        let make = |graph: bool| -> anyhow::Result<NativeDecodeSession> {
            NativeDecodeSession::load_with_cuda_options(
                model_dir.join("model.onnx"),
                NativeDecodeDevice::Cuda { index: Some(0) },
                NativeDecodeCudaOptions {
                    kv_max_len: Some(128),
                    graph_capture: Some(graph),
                },
            )
        };

        // Oracle: greedy-continue the prompt to obtain a deterministic draft.
        let mut oracle = make(true)?;
        let mut logits = oracle.decode(&prompt, 0)?.pop().context("prefill logits")?;
        let mut cont = Vec::new();
        for _ in 0..6 {
            let token = argmax(&logits);
            cont.push(token);
            logits = oracle
                .decode(&[token], oracle.current_len())?
                .pop()
                .context("oracle decode logits")?;
        }
        drop(oracle);

        let past = prompt.len();
        let draft = &cont[..4];
        let j = 2usize; // pretend the driver accepted 2 of the 4 draft tokens
        let feed = cont[j]; // deterministic next token fed after the committed prefix

        for retain in [false, true] {
            let mut verify_sess = make(true)?;
            verify_sess.decode(&prompt, 0)?;
            if retain {
                verify_sess.set_retain_graph_on_rewind(true);
            }
            assert_eq!(verify_sess.current_len(), past);

            // Eager M=K verify pass returns one logits row per draft position.
            let rows = verify_sess.decode_verify(draft, past)?;
            assert_eq!(rows.len(), draft.len());
            assert_eq!(verify_sess.current_len(), past + draft.len());

            // Rewind to the committed length; mask/KV logical shapes must follow.
            verify_sess.rewind(past + j)?;
            assert_eq!(verify_sess.current_len(), past + j);
            let stats = verify_sess.cuda_kv_debug_stats().unwrap();
            assert_eq!(stats.logical_len, past + j);

            let after = verify_sess.decode(&[feed], past + j)?;

            // Fresh M=1 reference from the committed prefix prompt ++ draft[..j].
            let mut fresh = make(true)?;
            let mut committed = prompt.clone();
            committed.extend_from_slice(&draft[..j]);
            fresh.decode(&committed, 0)?;
            let fresh_after = fresh.decode(&[feed], committed.len())?;

            let after_bits = after
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>();
            let fresh_bits = fresh_after
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>();
            assert_eq!(
                after_bits, fresh_bits,
                "verify+rewind (retain_graph_on_rewind={retain}) corrupted device KV vs fresh M=1 decode"
            );
        }
        Ok(())
    }

    #[test]
    fn native_logits_shapes_match_ort_semantics() {
        let cases = [
            (vec![3], 1),
            (vec![1, 3], 1),
            (vec![2, 3], 2),
            (vec![1, 1, 3], 1),
            (vec![1, 2, 3], 2),
            (vec![2, 2, 3], 2),
        ];
        for (shape, expected_rows) in cases {
            let values = (0..shape.iter().product::<usize>())
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            let tensor = Tensor::from_f32(&shape, &values).expect("create logits");
            let logits = extract_logits(&tensor).expect("extract logits");
            assert_eq!(logits.len(), expected_rows, "shape {shape:?}");
            assert_eq!(logits[0].len(), 3, "shape {shape:?}");
        }

        let tensor = Tensor::from_f32(&[1, 1, 1, 3], &[0.0; 3]).expect("create logits");
        let error = extract_logits(&tensor).expect_err("rank-four logits must be rejected");
        assert!(
            error
                .to_string()
                .contains("unsupported logits tensor shape")
        );
    }
}
