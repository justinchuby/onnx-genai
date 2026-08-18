//! Paged key/value decode-loop adapter for autoregressive pipelines.
//!
//! Pure code motion from `pipeline.rs`: the paged-sequence lifecycle helpers,
//! the `PagedMirror` that reflects each step's KV into shared pages, and the
//! `PipelineDecodeLoopBackend` that bridges the pipeline decoder into the
//! shared decode loop. Struct fields are exposed pub(crate) because the flat
//! autoregressive driver constructs these adapters across the module boundary.

use super::*;
use onnx_genai_metadata::ComponentSession;

/// Whether this error is the KV page pool being full, rather than a fault.
///
/// A full pool is a capacity condition the caller can degrade around; anything
/// else means the mirror is broken and must not be swallowed.
fn is_kv_out_of_memory(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<onnx_genai_kv::KvError>(),
            Some(onnx_genai_kv::KvError::OutOfMemory { .. })
        )
    })
}

impl PipelinePagedKv {
    /// Release the in-flight sequence, if any, returning its pages to the pool.
    ///
    /// Pages the prefix cache published are retained by it, so this frees only
    /// what nothing else refers to.
    pub(crate) fn discard_active(&mut self) {
        let Some(seq) = self.active.take() else {
            return;
        };
        for page_id in self.cache.page_table.remove_sequence(seq) {
            self.cache.page_table.free(page_id);
        }
    }

    /// Return pages to the pool by evicting unreferenced cached prefixes,
    /// least-recently-used first.
    ///
    /// Without this the cache would hold every prefix it ever published and the
    /// pool would run dry after enough distinct conversations. Only prefixes no
    /// live sequence is borrowing can go.
    pub(crate) fn evict_until_free(&mut self, wanted_pages: usize) {
        let free = self
            .cache
            .page_table
            .free_count(onnx_genai_kv::Device::Gpu(0));
        if free >= wanted_pages {
            return;
        }
        self.prefix
            .evict_lru(wanted_pages - free, &mut self.cache.page_table);
    }
}

/// Where a decode step's KV is written so later requests can share it.
pub(crate) struct PagedMirror<'a> {
    pub(crate) kv_model: &'a KvModelInfo,
    pub(crate) cache: &'a mut PagedKvCache,
    pub(crate) seq: SequenceId,
    /// Tokens whose KV actually reached the pages so far.
    ///
    /// Mirroring can stop early when the pool runs dry, and only this many
    /// tokens may then be published — the pages beyond it do not exist, and a
    /// key claiming them would hand a later request KV that was never written.
    pub(crate) mirrored_tokens: usize,
    /// Set once the pool refused a page, after which mirroring stops for the
    /// rest of this generation.
    pub(crate) exhausted: bool,
    /// Set once the sliding window dropped pages from this sequence.
    ///
    /// What remains is then `[sinks | recent window]`, which is not a prefix of
    /// anything. Publishing it under a key that says "the first N tokens" would
    /// hand a later request pages with a hole in the middle, so nothing from a
    /// sequence that has been windowed may be published at all.
    pub(crate) windowed: bool,
}

pub(crate) struct PipelineDecodeLoopBackend<'a, 'admission> {
    /// The decoder driven each step, owning its KV state behind a stateful,
    /// backend-neutral seam. Boxed behind [`PipelineDecoderComponent`] so the
    /// same loop drives the ORT decoder (via [`OrtPipelineDecoder`]) or, in a
    /// follow-up, a native decoder keeping KV device-resident — one code path,
    /// no forked native copy (mirrors the Inc1 every_step seam, but stateful).
    pub(crate) decoder: &'a mut dyn PipelineDecoderComponent,
    /// One-shot admission notification fired only after the first decoder step's
    /// exact routed inputs have successfully prepared governed workspace.
    pub(crate) admission: Option<&'admission mut dyn FnMut()>,
    /// Shared tensor pool: external inputs + prompt-phase outputs + the
    /// per-step outputs of the `every_step` components (refreshed each step).
    pub(crate) pool: &'a mut PipelineTensors,
    /// Declared `every_step` components (paired with their backend-neutral
    /// [`ComponentSession`]), executed in topological order on every step before
    /// the decoder runs. Boxed behind the trait so the same loop drives an ORT
    /// session (via `OrtComponentSessionRef`) or a native nxrt component with no
    /// forked code path — this is the value-type seam for the every_step slice.
    pub(crate) step_components: Vec<(StepComponentBinding, Box<dyn ComponentSession + 'a>)>,
    /// `(source_endpoint, decoder_input_port)` routing recomputed each step.
    pub(crate) decoder_in_edges: Vec<(String, String)>,
    /// Static encoder-produced cross-attention KV bound to the decoder every
    /// step: `(decoder_input_port, shared_value)`. Resolved once from the encoder
    /// prologue outputs and held behind an `Arc` so each step re-binds it as a
    /// no-copy alias (O(1)) rather than deep-copying the large invariant buffer
    /// (see `PipelineEngine::static_cross_kv_bindings`).
    pub(crate) static_cross_kv: Vec<(String, Arc<Value>)>,
    pub(crate) context_tokens: Vec<TokenId>,
    /// Leading tokens whose KV was carried over from the previous generation and
    /// so must not be prefilled again.
    pub(crate) retained_len: usize,
    pub(crate) prompt_len: usize,
    pub(crate) generated_count: usize,
    /// Paged KV to mirror each step's `present.*` outputs into, when the
    /// decoder's KV can be paged.
    pub(crate) paged: Option<PagedMirror<'a>>,
    /// Tokens the decoder has actually run, and therefore the exact length its
    /// KV covers.
    ///
    /// Tracked rather than derived from `context_tokens`, because the two differ:
    /// `commit_token` appends the sampled token, but that token is not fed to the
    /// decoder until the *next* step, so at the end of a generation the context
    /// is one token longer than the KV. Retaining the context length would claim
    /// KV that does not exist and corrupt the next turn's attention.
    pub(crate) kv_len: usize,
}

impl PipelineDecodeLoopBackend<'_, '_> {
    /// Run every declared `every_step` component over `seed` (the full prompt on
    /// prefill, the single running token on decode), publishing all of their
    /// outputs into the shared pool. Topological order ensures a component sees
    /// any upstream `every_step` output produced earlier in the same step.
    fn run_step_components(&mut self, seed: &[TokenId]) -> anyhow::Result<()> {
        if self.step_components.is_empty() {
            return Ok(());
        }
        let ids: Vec<i64> = seed.iter().map(|&t| i64::from(t)).collect();
        // Disjoint borrows: the step sessions run `&mut` while the pool is read
        // for inputs and written for outputs.
        let Self {
            step_components,
            pool,
            ..
        } = self;
        for (binding, session) in step_components.iter_mut() {
            let mut inputs: Vec<(String, onnx_genai_metadata::ComponentTensor)> =
                Vec::with_capacity(binding.routed_inputs.len() + 1);
            for routed in &binding.routed_inputs {
                let value = pool
                    .get(&routed.endpoint)
                    .or_else(|| {
                        routed
                            .routed_from
                            .as_deref()
                            .and_then(|from| pool.get(from))
                    })
                    .with_context(|| routed.missing_message.clone())?;
                let coerced = coerce_value_to_dtype(value, routed.dtype)?;
                inputs.push((routed.port.clone(), value_to_component_tensor(&coerced)?));
            }
            if let Some(port) = &binding.token_input {
                inputs.push((port.clone(), token_seed_component_tensor(&ids)?));
            }
            let refs = inputs
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect::<Vec<_>>();
            let outputs = session.run(&refs).map_err(|e| {
                anyhow::anyhow!("every_step component '{}' failed: {e}", binding.component)
            })?;
            for (name, tensor) in outputs {
                pool.insert(
                    format!("{}.{}", binding.component, name),
                    component_tensor_to_value(&tensor)?,
                );
            }
        }
        Ok(())
    }

    /// Build this step's decoder extra inputs by re-reading every routed source
    /// endpoint from the shared pool. `every_step` outputs are already fresh
    /// (just re-run); cached `prompt_only` conditioning is simply re-read. The
    /// static encoder cross-attention KV (resolved once from the prologue) is
    /// appended verbatim so the decoder's `past_*_cross_%d` inputs are bound.
    fn decoder_extras(&self) -> anyhow::Result<Vec<(String, Value)>> {
        let mut extras =
            Vec::with_capacity(self.decoder_in_edges.len() + self.static_cross_kv.len());
        for (from, port) in &self.decoder_in_edges {
            let value = self.pool.get(from).with_context(|| {
                format!("missing routed pipeline tensor '{from}' for decoder input '{port}'")
            })?;
            extras.push((port.clone(), clone_value(value)?));
        }
        for (port, value) in &self.static_cross_kv {
            // The static cross-KV buffer is invariant across the decode loop, so
            // re-bind it as a no-copy alias over the shared owner instead of
            // deep-copying the (large) tensor every step.
            let aliased = Value::alias_with_shape(Arc::clone(value), value.shape())?;
            extras.push((port.clone(), aliased));
        }
        Ok(extras)
    }
}

impl DecodeLoopBackend for PipelineDecodeLoopBackend<'_, '_> {
    fn context_len(&self) -> usize {
        self.context_tokens.len()
    }

    fn processor_prompt_tokens(&self) -> &[TokenId] {
        &self.context_tokens
    }

    fn next_logits(&mut self) -> anyhow::Result<Vec<f32>> {
        let use_kv = self.decoder.use_kv();
        let past_len = if use_kv {
            self.context_tokens
                .len()
                .saturating_sub(if self.generated_count == 0 {
                    self.prompt_len
                } else {
                    1
                })
        } else {
            0
        };
        // On the first step feed only the tokens not already covered by
        // retained KV (`prompt_len` is the uncovered suffix, and equals the
        // whole prompt when nothing was retained); afterwards, the running token.
        let input_tokens = if use_kv && self.generated_count > 0 {
            self.context_tokens[self.context_tokens.len() - 1..].to_vec()
        } else {
            self.context_tokens[self.retained_len..].to_vec()
        };
        // Refresh every `every_step` component over exactly the tokens the
        // decoder is about to consume, then route their (and any cached) outputs
        // into the decoder for this step.
        self.run_step_components(&input_tokens)?;
        let extras = self.decoder_extras()?;
        self.decoder
            .prepare_step(&input_tokens, past_len, &extras)?;
        if let Some(admitted) = self.admission.take() {
            admitted();
        }
        // Advance the decoder one step; it retains this step's outputs internally
        // so the loop never handles a concrete tensor type (see
        // `PipelineDecoderComponent`).
        self.decoder.step(&input_tokens, past_len, &extras)?;
        self.kv_len = past_len + input_tokens.len();
        // Mirror this step's KV into pages before the outputs are consumed, so
        // a later request opening with the same prefix can attach these pages
        // instead of recomputing them.
        if let Some(paged) = self.paged.as_mut().filter(|paged| !paged.exhausted) {
            // A windowed decoder's present tensor is indexed in *retained*
            // buffer space, not absolute position space: once the window has
            // evicted anything, an absolute index reads the wrong rows or runs
            // off the end. This is the same conversion the single-model decode
            // step does before mirroring.
            let retained_past_len = self.decoder.retained_kv_len(past_len);
            match self.decoder.mirror_last_present_kv(
                paged.kv_model,
                paged.cache,
                paged.seq,
                retained_past_len,
                input_tokens.len(),
            ) {
                Ok(()) => paged.mirrored_tokens = past_len + input_tokens.len(),
                // Mirroring exists so a *later* request can reuse this KV. The
                // pool running dry says nothing about whether this generation
                // is valid, so failing it would punish the caller for a cache
                // being full. Stop mirroring and keep decoding; only the
                // tokens already mirrored get published.
                Err(error) if is_kv_out_of_memory(&error) => {
                    paged.exhausted = true;
                    tracing::debug!(
                        "KV page pool exhausted after {} token(s); this generation stops \
                         publishing KV for reuse but continues normally ({error})",
                        paged.mirrored_tokens
                    );
                }
                Err(error) => return Err(error),
            }
            // Keep the paged sequence's window in step with the decoder's, so
            // the pages published for reuse describe what the decoder can
            // actually attend to.
            let pages_before = paged
                .cache
                .page_table
                .get_sequence(paged.seq)
                .map_or(0, <[_]>::len);
            apply_paged_sliding_window(
                paged.cache,
                paged.seq,
                self.decoder.sliding_window(),
                self.decoder.sink_tokens(),
            )?;
            let pages_after = paged
                .cache
                .page_table
                .get_sequence(paged.seq)
                .map_or(0, <[_]>::len);
            // Compared rather than inferred from the window size: only an
            // actual drop makes the sequence non-contiguous, so a windowed
            // model whose conversation still fits its window keeps publishing.
            if pages_after < pages_before {
                paged.windowed = true;
            }
        }
        self.decoder.next_token_logits()
    }

    fn commit_token(&mut self, token_id: TokenId) -> anyhow::Result<()> {
        self.context_tokens.push(token_id);
        self.generated_count += 1;
        Ok(())
    }
}
