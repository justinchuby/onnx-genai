//! Stateful decoder seam for the flat autoregressive pipeline.
//!
//! Inc1 (#450) routed the *stateless* `every_step` components through the
//! backend-neutral [`ComponentSession`](onnx_genai_metadata::ComponentSession)
//! trait. The decoder cannot use that seam: it is **stateful** — its KV cache
//! grows across steps and, for the native backend, lives device-resident — so
//! driving it through a stateless host round-trip would drop KV continuity and
//! re-stage the whole cache every step.
//!
//! [`PipelineDecoderComponent`] is the stateful counterpart: the decode loop
//! calls [`step`](PipelineDecoderComponent::step) once per token and the
//! implementation retains its own per-step outputs, so the loop never touches a
//! backend tensor type. [`OrtPipelineDecoder`] is the ONNX Runtime
//! implementation, behaviour-identical to the previous inline
//! `run_decode_step_with_extra` path. A native implementation keeping KV
//! device-resident is the follow-up (Inc2b); see
//! `.squad/decisions/inbox/mary-pipeline-inc2-design.md`.

use super::*;
use crate::decode::{extract_next_token_logits_from_outputs, run_decode_step_with_extra};
#[cfg(feature = "native-backend")]
use crate::kv_bridge::extract_present_token;
use crate::kv_bridge::{KvModelInfo, mirror_present_kv_to_pages};
#[cfg(feature = "native-backend")]
use onnx_genai_kv::LayerKv;
use onnx_genai_kv::{PagedKvCache, SequenceId};

/// One decoder driven inside the pipeline decode loop, owning its KV state
/// across steps so the loop stays backend-agnostic.
///
/// The implementation retains the most recent step's outputs internally: the
/// loop calls [`step`](Self::step), then reads [`next_token_logits`](Self::next_token_logits)
/// and (when paging) [`mirror_last_present_kv`](Self::mirror_last_present_kv)
/// without ever handling a concrete tensor type.
pub(crate) trait PipelineDecoderComponent {
    /// Prepare any governed decoder workspace using the exact routed values for
    /// this step. Backends without a workspace contract remain a no-op.
    fn prepare_step(
        &mut self,
        _input_tokens: &[TokenId],
        _past_len: usize,
        _extras: &[(String, Value)],
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Run one decoder step over `input_tokens` at absolute `past_len`, binding
    /// the routed `extras` (every_step outputs, `inputs_embeds`, routed
    /// positions, static cross-attention KV), advancing the internal KV. The
    /// step's outputs are retained for [`next_token_logits`](Self::next_token_logits)
    /// and [`mirror_last_present_kv`](Self::mirror_last_present_kv).
    fn step(
        &mut self,
        input_tokens: &[TokenId],
        past_len: usize,
        extras: &[(String, Value)],
    ) -> anyhow::Result<()>;

    /// Next-token logits (final position) from the most recent step.
    fn next_token_logits(&self) -> anyhow::Result<Vec<f32>>;

    /// Mirror the most recent step's present KV into the paged cache so a later
    /// request opening with the same prefix can attach the pages. Takes `&mut
    /// self` because a device-resident native decoder (GAP-3 Inc-D) reads its
    /// present KV out of a device binding, which mutates transfer bookkeeping;
    /// the ORT and host-growable paths remain read-only in effect.
    fn mirror_last_present_kv(
        &mut self,
        kv_model: &KvModelInfo,
        cache: &mut PagedKvCache,
        seq: SequenceId,
        retained_past_len: usize,
        input_len: usize,
    ) -> anyhow::Result<()>;

    /// Whether the decoder carries KV across steps.
    fn use_kv(&self) -> bool;

    /// KV length in *retained* buffer space for the given absolute `past_len`.
    fn retained_kv_len(&self, past_len: usize) -> usize;

    /// Sliding-window span, if the decoder attends over a bounded window.
    fn sliding_window(&self) -> Option<usize>;

    /// Number of always-retained sink tokens under a sliding window.
    fn sink_tokens(&self) -> usize;

    /// Current KV prefix length retained by this decoder, when the backend owns
    /// a rewindable session-resident cache.
    fn current_kv_len(&self) -> Option<usize> {
        None
    }

    /// Rewind a session-resident KV cache to `target_len`. Returns `false` for
    /// backends that cannot retain their decoder object across requests.
    fn rewind_kv(&mut self, _target_len: usize) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Whether this decoder can mirror its present KV into the paged cache and be
    /// re-seeded from a materialized paged prefix, so the pipeline may drive it
    /// on the paged (cross-request KV reuse) path rather than the fresh-decode
    /// path. ORT decoders always can; a native decoder can only when its KV is
    /// host-resident and f32 (GAP-3 Inc-C) or device-resident CUDA f32 rank-4
    /// (GAP-3 Inc-D) — otherwise (f16 / in-place / non-rank-4) the pipeline keeps
    /// it on the non-paged path (Inc-A behaviour) with no regression.
    fn supports_paged_kv(&self) -> bool {
        true
    }

    /// Seed this decoder's KV state from a materialized shared paged prefix so a
    /// request that reuses a common prompt prefix resumes at `materialized.
    /// sequence_len` without recomputing it. Only invoked for decoders that
    /// report [`supports_paged_kv`](Self::supports_paged_kv); the default is
    /// unreachable and errors loudly.
    fn load_paged_prefix(
        &mut self,
        _kv_model: &KvModelInfo,
        _materialized: &onnx_genai_kv::MaterializedKv,
    ) -> anyhow::Result<()> {
        anyhow::bail!("this decoder does not support paged prefix reuse")
    }
}

/// ONNX Runtime [`PipelineDecoderComponent`]: wraps the borrowed decoder session
/// and its host-KV [`DecodeState`], forwarding to the existing ORT decode-step,
/// KV-mirror, and logits-extraction helpers so behaviour is identical to the
/// previous inline path.
pub(crate) struct OrtPipelineDecoder<'a> {
    session: &'a Session,
    state: &'a mut DecodeState,
    /// The most recent step's outputs (present KV + logits), retained so the
    /// loop can read logits and mirror KV without handling ORT `Value`s.
    last_outputs: Option<Vec<Value>>,
}

impl<'a> OrtPipelineDecoder<'a> {
    pub(crate) fn new(session: &'a Session, state: &'a mut DecodeState) -> Self {
        Self {
            session,
            state,
            last_outputs: None,
        }
    }

    fn last_outputs(&self) -> anyhow::Result<&[Value]> {
        self.last_outputs
            .as_deref()
            .context("decoder logits/KV requested before any decode step ran")
    }
}

impl PipelineDecoderComponent for OrtPipelineDecoder<'_> {
    fn step(
        &mut self,
        input_tokens: &[TokenId],
        past_len: usize,
        extras: &[(String, Value)],
    ) -> anyhow::Result<()> {
        let outputs =
            run_decode_step_with_extra(self.session, self.state, input_tokens, past_len, extras)?;
        self.last_outputs = Some(outputs);
        Ok(())
    }

    fn next_token_logits(&self) -> anyhow::Result<Vec<f32>> {
        // Read logits from the retained outputs without moving/cloning them: the
        // pipeline keeps them for KV mirroring, and the slice extractor borrows.
        extract_next_token_logits_from_outputs(
            self.session,
            self.last_outputs()?,
            self.state.io.logits_output.as_deref(),
        )
    }

    fn mirror_last_present_kv(
        &mut self,
        kv_model: &KvModelInfo,
        cache: &mut PagedKvCache,
        seq: SequenceId,
        retained_past_len: usize,
        input_len: usize,
    ) -> anyhow::Result<()> {
        mirror_present_kv_to_pages(
            self.session,
            kv_model,
            cache,
            seq,
            self.last_outputs()?,
            retained_past_len,
            input_len,
        )
    }

    fn use_kv(&self) -> bool {
        self.state.use_kv
    }

    fn retained_kv_len(&self, past_len: usize) -> usize {
        self.state.retained_kv_len(past_len)
    }

    fn sliding_window(&self) -> Option<usize> {
        self.state.sliding_window()
    }

    fn sink_tokens(&self) -> usize {
        self.state.sink_tokens()
    }
}

/// Native (pure-Rust nxrt) [`PipelineDecoderComponent`]: wraps a
/// [`NativeDecodeSession`](crate::native_decode::NativeDecodeSession) whose KV
/// cache stays session-resident across `step()` calls, so the expensive KV state
/// never round-trips through the host pipeline pool. Only the small per-step
/// routed inputs (e.g. one token's `inputs_embeds` produced by an every_step
/// component) cross the host seam each step; the decoder owns and grows its KV
/// internally.
///
/// This is the Inc2b (GAP 3) counterpart to [`OrtPipelineDecoder`]: the same
/// pipeline decode loop drives either backend through the trait with no forked
/// code path. Paged present-KV mirroring and prefix reuse are wired for the
/// host-resident growable f32 KV path (GAP-3 Inc-C, see
/// [`supports_paged_kv`](PipelineDecoderComponent::supports_paged_kv)); a
/// device-resident (CUDA) or in-place (GQA) / f16 KV store keeps the non-paged
/// fresh-decode path until Inc-D. Cross-attention / vision KV is Inc3.
#[cfg(feature = "native-backend")]
pub(crate) struct NativePipelineDecoder {
    session: crate::native_decode::NativeDecodeSession,
    /// Next-token logits (final position) from the most recent step, retained so
    /// the loop reads them without the native tensor type crossing the seam.
    last_logits: Option<Vec<f32>>,
}

#[cfg(feature = "native-backend")]
impl NativePipelineDecoder {
    fn native_step_inputs(
        extras: &[(String, Value)],
    ) -> anyhow::Result<Vec<(String, onnx_runtime_session::Tensor)>> {
        let mut step_inputs = Vec::with_capacity(extras.len());
        for (port, value) in extras {
            let component = value_to_component_tensor(value)?;
            let tensor =
                crate::native_component::component_tensor_to_native_tensor(port, &component)?;
            step_inputs.push((port.clone(), tensor));
        }
        Ok(step_inputs)
    }

    /// Load the decoder ONNX model as a native decode session on the given
    /// device, keeping its KV cache resident across the generation. The
    /// pipeline-declared `io` spec is threaded so an `inputs_embeds` decoder
    /// (no token input) binds its sequence source and KV pairs from metadata.
    pub(crate) fn load(
        path: &std::path::Path,
        device: crate::native_decode::NativeDecodeDevice,
        io: Option<&onnx_genai_metadata::ModelIoSpec>,
        metadata_max_len: Option<usize>,
        #[cfg(feature = "cuda")] offload_policy: onnx_runtime_ep_cuda::DeviceOffloadPolicy,
        #[cfg(feature = "cuda")] governor: std::sync::Arc<
            dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync,
        >,
        #[cfg(feature = "cuda")] manager: onnx_runtime_memory_governor::ProcessMemoryManager,
    ) -> anyhow::Result<Self> {
        #[cfg(feature = "cuda")]
        let session = crate::native_decode::NativeDecodeSession::load_with_io_and_cuda_governor(
            path,
            device,
            io,
            metadata_max_len,
            offload_policy,
            governor,
            manager,
        )
        .with_context(|| {
            format!(
                "failed to load native pipeline decoder '{}'",
                path.display()
            )
        })?;
        #[cfg(not(feature = "cuda"))]
        let session = crate::native_decode::NativeDecodeSession::load_with_io(
            path,
            device,
            io,
            metadata_max_len,
        )
        .with_context(|| {
            format!(
                "failed to load native pipeline decoder '{}'",
                path.display()
            )
        })?;
        Ok(Self {
            session,
            last_logits: None,
        })
    }
}

#[cfg(feature = "native-backend")]
impl PipelineDecoderComponent for NativePipelineDecoder {
    fn prepare_step(
        &mut self,
        input_tokens: &[TokenId],
        past_len: usize,
        extras: &[(String, Value)],
    ) -> anyhow::Result<()> {
        let step_inputs = Self::native_step_inputs(extras)?;
        self.session.prepare_generation_workspace_with_step_inputs(
            input_tokens,
            past_len,
            &step_inputs,
        )?;
        Ok(())
    }

    fn step(
        &mut self,
        input_tokens: &[TokenId],
        past_len: usize,
        extras: &[(String, Value)],
    ) -> anyhow::Result<()> {
        // Route each host-pool extra (an ort::Value, e.g. the every_step
        // embedding output) to the decoder's exact graph port as a native tensor.
        // value -> ComponentTensor -> native Tensor reuses the Inc1 value-type
        // seam; this is one token's embedding per decode step (small upload),
        // while the KV cache stays resident inside the native session.
        let step_inputs = Self::native_step_inputs(extras)?;
        let rows = self
            .session
            .decode_with_step_inputs(input_tokens, past_len, &step_inputs)?;
        // Native decode returns one logits row per input position; the final row
        // is the next-token distribution (matching the ORT extractor).
        self.last_logits = Some(
            rows.into_iter()
                .next_back()
                .context("native decoder produced no logits rows")?,
        );
        Ok(())
    }

    fn next_token_logits(&self) -> anyhow::Result<Vec<f32>> {
        self.last_logits
            .clone()
            .context("decoder logits requested before any decode step ran")
    }

    fn mirror_last_present_kv(
        &mut self,
        kv_model: &KvModelInfo,
        cache: &mut PagedKvCache,
        seq: SequenceId,
        retained_past_len: usize,
        input_len: usize,
    ) -> anyhow::Result<()> {
        // Read the most recent step's accumulated present KV out of the native
        // decoder — the growable host cache (Inc-C) or the device-resident CUDA
        // binding (Inc-D) — then publish the freshly-decoded tokens into pages
        // through the *same* geometry the ORT decoder uses
        // (`extract_present_token` + `append_token_kv`), so native and ORT
        // mirror byte-identical pages. The device read carries the physical
        // capacity shape so strides address the padded buffer; the host read
        // carries its compact shape. f16 / in-place / non-rank-4 KV stays gated
        // off the paged path (`supports_paged_kv`).
        let layer_data = kv_model
            .layers
            .iter()
            .map(|layer| {
                let (key, key_shape) =
                    self.session.present_kv(&layer.key_past)?.with_context(|| {
                        format!(
                            "native decoder produced no present KV for '{}'; a decode step must \
                             run before mirroring",
                            layer.key_past
                        )
                    })?;
                let (value, value_shape) = self
                    .session
                    .present_kv(&layer.value_past)?
                    .with_context(|| {
                        format!(
                            "native decoder produced no present KV for '{}'",
                            layer.value_past
                        )
                    })?;
                let to_i64 =
                    |shape: Vec<usize>| shape.iter().map(|&d| d as i64).collect::<Vec<_>>();
                Ok((key, to_i64(key_shape), value, to_i64(value_shape)))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        for offset in 0..input_len {
            let token_pos = retained_past_len + offset;
            let owned_layers = layer_data
                .iter()
                .enumerate()
                .map(|(layer_idx, (key, key_shape, value, value_shape))| {
                    let layer_config = kv_model.layer_tensor_config(layer_idx);
                    Ok((
                        extract_present_token(key, key_shape, layer_config, token_pos)?,
                        extract_present_token(value, value_shape, layer_config, token_pos)?,
                    ))
                })
                .collect::<anyhow::Result<Vec<(Vec<f32>, Vec<f32>)>>>()?;
            let borrowed = owned_layers
                .iter()
                .map(|(key, value)| LayerKv {
                    key: key.as_slice(),
                    value: value.as_slice(),
                })
                .collect::<Vec<_>>();
            cache
                .append_token_kv(seq, &borrowed)
                .context("Failed to mirror native present KV into pages")?;
        }
        Ok(())
    }

    fn use_kv(&self) -> bool {
        true
    }

    fn retained_kv_len(&self, past_len: usize) -> usize {
        // No sliding window in this increment: retained length is the absolute
        // past length, so the paged mirror indexes the present tensor in the
        // same absolute space the growable host cache grows in.
        past_len
    }

    fn sliding_window(&self) -> Option<usize> {
        None
    }

    fn sink_tokens(&self) -> usize {
        0
    }

    fn current_kv_len(&self) -> Option<usize> {
        Some(self.session.current_len())
    }

    fn rewind_kv(&mut self, target_len: usize) -> anyhow::Result<bool> {
        self.session.rewind(target_len)?;
        Ok(true)
    }

    fn supports_paged_kv(&self) -> bool {
        self.session.supports_host_kv_mirror() || self.session.supports_device_kv_mirror()
    }

    fn load_paged_prefix(
        &mut self,
        kv_model: &KvModelInfo,
        materialized: &onnx_genai_kv::MaterializedKv,
    ) -> anyhow::Result<()> {
        // Re-seed the growable host KV from the shared prefix using the exact
        // `[1, num_kv_heads, seq, head_dim]` layout the ORT decoder injects
        // (`kv_bridge::materialized_past_values` via `past_shape`), so native and
        // ORT prefix reuse are byte-identical. Discontinuous attention-sink
        // prefixes are Inc-D, matching the ORT path's own restriction.
        if materialized.start_position != 0 || materialized.sink_len != 0 {
            anyhow::bail!(
                "native paged prefix reuse cannot start at absolute position {} (sink_len {}); \
                 discontinuous attention-sink prefixes are Inc-D",
                materialized.start_position,
                materialized.sink_len
            );
        }
        let seq_len = materialized.sequence_len;
        let mut entries = Vec::with_capacity(kv_model.layers.len() * 2);
        for (layer_idx, layer) in kv_model.layers.iter().enumerate() {
            let config = kv_model.layer_tensor_config(layer_idx);
            let shape = vec![1_usize, config.num_kv_heads, seq_len, config.head_dim];
            let materialized_layer = materialized
                .layers
                .get(layer_idx)
                .with_context(|| format!("materialized prefix is missing layer {layer_idx} KV"))?;
            entries.push((
                layer.key_past.clone(),
                materialized_layer.key.clone(),
                shape.clone(),
            ));
            entries.push((
                layer.value_past.clone(),
                materialized_layer.value.clone(),
                shape,
            ));
        }
        self.session.seed_kv(entries, seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_ort::{PipelineModels, Value};
    use std::path::Path;

    /// Inc2a is a pure refactor: driving the decoder through the stateful
    /// [`PipelineDecoderComponent`] seam must be bit-identical to the previous
    /// inline `run_decode_step_with_extra` + `extract_next_token_logits_*` path.
    /// Run both over the same fixture, step for step, and assert the logits match
    /// exactly — the ORT decoder's token output cannot change.
    #[test]
    fn ort_decoder_component_matches_inline_step_path() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-multiaxis-state-decoder");
        let models = PipelineModels::load(&fixture)?;
        let session = models
            .session("decoder")
            .expect("fixture decoder session is loaded");
        let component = &models.directory.spec.models["decoder"];
        let positions = models
            .directory
            .spec
            .positions
            .as_ref()
            .expect("fixture position program");
        let extras_for = || -> anyhow::Result<Vec<(String, Value)>> {
            Ok(vec![(
                "routed_sequence".to_string(),
                Value::from_vec_f32(vec![0.0; 3], &[1, 3, 1])?,
            )])
        };
        // The exact steps the fixture's inline test exercises: a 3-token prefill
        // then two single-token decode steps, threading KV across each.
        let steps: [(Vec<TokenId>, usize); 3] = [(vec![1, 2, 3], 0), (vec![6], 3), (vec![15], 4)];

        // Golden: the previous inline path over its own state.
        let mut inline_state = DecodeState::new_with_io_and_positions(
            session,
            component.io.as_ref(),
            Some(positions),
        )?;
        let mut golden = Vec::new();
        for (tokens, past_len) in &steps {
            let extras = extras_for()?;
            let outputs =
                run_decode_step_with_extra(session, &mut inline_state, tokens, *past_len, &extras)?;
            golden.push(extract_next_token_logits_from_outputs(
                session,
                &outputs,
                inline_state.io.logits_output.as_deref(),
            )?);
        }

        // Under test: the same steps driven through the trait wrapper, which owns
        // its state and retains each step's outputs internally.
        let mut wrapped_state = DecodeState::new_with_io_and_positions(
            session,
            component.io.as_ref(),
            Some(positions),
        )?;
        let mut decoder = OrtPipelineDecoder::new(session, &mut wrapped_state);
        for ((tokens, past_len), expected) in steps.iter().zip(&golden) {
            let extras = extras_for()?;
            decoder.step(tokens, *past_len, &extras)?;
            let got = decoder.next_token_logits()?;
            assert_eq!(
                &got, expected,
                "trait-driven decoder logits must equal the inline path bit for bit"
            );
        }
        Ok(())
    }
}
