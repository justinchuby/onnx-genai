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
use crate::kv_bridge::{KvModelInfo, mirror_present_kv_to_pages};
use onnx_genai_kv::{PagedKvCache, SequenceId};

/// One decoder driven inside the pipeline decode loop, owning its KV state
/// across steps so the loop stays backend-agnostic.
///
/// The implementation retains the most recent step's outputs internally: the
/// loop calls [`step`](Self::step), then reads [`next_token_logits`](Self::next_token_logits)
/// and (when paging) [`mirror_last_present_kv`](Self::mirror_last_present_kv)
/// without ever handling a concrete tensor type.
pub(crate) trait PipelineDecoderComponent {
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
    /// request opening with the same prefix can attach the pages.
    fn mirror_last_present_kv(
        &self,
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
        &self,
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
