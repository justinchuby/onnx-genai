use super::*;

pub(super) fn stage_host_prefix_rewind(
    past: &HashMap<String, Tensor>,
    skip_names: &HashSet<String>,
    target_len: usize,
) -> anyhow::Result<Vec<(String, Tensor)>> {
    let mut rewound = Vec::new();
    for (name, tensor) in past {
        if skip_names.contains(name) {
            continue;
        }
        let axis = tensor
            .shape
            .len()
            .checked_sub(2)
            .with_context(|| format!("native KV tensor '{name}' rank is below 2"))?;
        rewound.push((
            name.clone(),
            prefix_slice(tensor, axis, target_len)
                .with_context(|| format!("rewind native KV tensor '{name}'"))?,
        ));
    }
    Ok(rewound)
}

impl DecodeBackend for NativeDecodeSession {
    fn decode(&mut self, token_ids: &[TokenId], past_len: usize) -> anyhow::Result<Vec<Vec<f32>>> {
        self.decode_with_step_inputs(token_ids, past_len, &[])
    }

    fn decode_argmax(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Option<u32>> {
        if let Some(cuda) = &self.cuda {
            cuda.ensure_state_restore_healthy()?;
        }
        if token_ids.is_empty() {
            bail!("native decode requires at least one token");
        }
        if past_len != self.current_len {
            bail!(
                "native decode past length mismatch: caller supplied {past_len}, adapter holds {}",
                self.current_len
            );
        }
        let Some(chunk) = self
            .prefill_chunk_size
            .map(NonZeroUsize::get)
            .filter(|&chunk| token_ids.len() > chunk)
        else {
            return self.decode_argmax_forward(token_ids, past_len);
        };
        // Only the last chunk's token is the prompt's continuation; the earlier
        // forwards exist to populate KV. They are still run through the same
        // entry point, so nothing about a chunk differs from a short prompt.
        let mut token = None;
        for slice in token_ids.chunks(chunk) {
            let past_len = self.current_len;
            token = self.decode_argmax_forward(slice, past_len)?;
        }
        Ok(token)
    }

    fn supports_argmax(&self) -> bool {
        true
    }
}

impl NativeDecodeSession {
    /// Greedy sibling of [`Self::decode_with_step_inputs`] for a decoder whose
    /// graph declares per-step ports (an embedding input, and any other declared
    /// `Routed` port) that [`DecodeBackend::decode_argmax`] cannot carry.
    ///
    /// Returns `Ok(None)` whenever the step is not the captured single-token
    /// shape the device-argmax epilogue applies to — a multi-token prefill, a
    /// decoder whose step inputs are not capture-eligible, a non-CUDA session.
    /// The caller then falls back to logits, so this is a fast path and never a
    /// behaviour switch.
    pub(crate) fn decode_argmax_with_step_inputs(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        step_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<Option<TokenId>> {
        if token_ids.len() != 1 || !self.captured_step_input_greedy_supported() {
            return Ok(None);
        }
        if past_len != self.current_len {
            bail!(
                "native decode past length mismatch: caller supplied {past_len}, adapter holds {}",
                self.current_len
            );
        }
        let total_len = past_len
            .checked_add(1)
            .context("native decode context length overflow")?;
        self.maybe_enable_decode_inline(token_ids);
        self.decode_cuda_captured_step_inputs_greedy(token_ids, past_len, total_len, step_inputs)
            .map(Some)
    }

    /// One prefill or decode forward over `token_ids`, with no chunking.
    fn decode_argmax_forward(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Option<u32>> {
        self.maybe_enable_decode_inline(token_ids);
        if self.cuda.is_some() {
            if token_ids.len() == 1 {
                // A decoder that declares per-step `inputs_embeds`/routed ports
                // cannot take the token-id-only greedy step below: that step
                // writes only the token id, so those persistent bindings would
                // replay whatever bytes they last held. Offer the captured
                // step-input epilogue first; it returns `None` whenever the step
                // is not that shape, and otherwise fails naming the port that
                // was not supplied instead of decoding without it.
                if let Some(token) =
                    self.decode_argmax_with_step_inputs(token_ids, past_len, &[])?
                {
                    return Ok(Some(token));
                }
                if self.has_eager_step_inputs() {
                    bail!(
                        "native CUDA decoder declares per-step inputs_embeds/routed ports, which \
                         the greedy argmax fast path cannot supply; drive this decoder through \
                         the workflow bindings that route those ports, or read logits and sample \
                         on the host"
                    );
                }
                return self.decode_cuda_greedy(token_ids[0], past_len).map(Some);
            }
            let token = self
                .decode_cuda(token_ids, past_len, &[])?
                .pop()
                .map(|logits| sample_greedy(&logits))
                .context("native CUDA decoder produced no logits")?;
            return Ok(Some(token));
        }
        if self.cpu_kv.is_some() {
            return match self.decode_cpu_inplace(token_ids, past_len, true, &[])? {
                NativeCpuDecodeResult::Token(token) => Ok(Some(token)),
                NativeCpuDecodeResult::Logits(_) => unreachable!("greedy token decode requested"),
            };
        }
        match self.decode_cpu(token_ids, past_len, true, &[])? {
            NativeCpuDecodeResult::Token(token) => Ok(Some(token)),
            NativeCpuDecodeResult::Logits(_) => unreachable!("greedy token decode requested"),
        }
    }

    /// Bound the tokens per prefill forward.
    ///
    /// Set from `model.runtime_configurable.chunked_prefill.chunk_size` at load,
    /// which is the same metadata the flat ORT pipeline reads. Zero and `None`
    /// both mean "one forward for the whole prompt".
    pub(crate) fn set_prefill_chunk_size(&mut self, chunk_size: Option<usize>) {
        self.prefill_chunk_size = chunk_size.and_then(NonZeroUsize::new);
        tracing::debug!(
            prefill_chunk_size = ?self.prefill_chunk_size.map(NonZeroUsize::get),
            "native decode prefill chunking configured from model metadata"
        );
    }
}

impl NativeDecodeSession {
    pub fn activation_memory_plan_stats(&self) -> Option<crate::ActivationMemoryPlanSummary> {
        self.session.activation_memory_plan_stats().map(Into::into)
    }

    pub(super) fn rewind_inner(&mut self, target_len: usize) -> anyhow::Result<()> {
        #[cfg(test)]
        if let Some(remaining) = self.fail_rewind_after.as_mut() {
            if *remaining == 0 {
                self.fail_rewind_after = None;
                bail!("injected native decoder rewind failure");
            }
            *remaining -= 1;
        }
        if let Some(cuda) = &self.cuda {
            cuda.ensure_state_restore_healthy()?;
        }
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
            // M=1 path re-warms cleanly). Two dormant seams (both OFF by default)
            // would retain the graph across a contents-only rewind instead — the
            // rewind only zeros the mask tail + truncates the KV logical length,
            // leaving every binding's physical_shape/device_ptr fixed, so the
            // captured M=1 graph's replay signature stays valid: `retain_graph_on_rewind`
            // (option (c) padded verify capture) and `retain_decode_graph_across_spec`
            // (spec-decode retention). Retention on rewind alone is capture-safe,
            // but NOT sufficient for a speedup: the eager M>1 verify forward tears
            // the graph down every step regardless, and retaining across BOTH
            // sites is capture-unsafe until the verify workspace is pinned (see
            // the verify site + decision note). A full reset to `target_len == 0`
            // (between generations) always invalidates so a stale graph never
            // leaks into the next generation.
            let retain = state.retain_graph_on_rewind
                || (state.retain_decode_graph_across_spec && target_len != 0);
            if !retain {
                state.invalidate_graph(&mut self.session)?;
            }
            state.rewind(target_len)?;
            if target_len == 0 {
                // Generation-reset boundary: `rewind(0)` just re-zeroed the
                // recurrent/conv rolling caches, so a verify graph captured in the
                // previous generation would replay against reset state (and, on a
                // reused session across prompts, potentially stale bindings). Reset
                // the verify-dedicated sibling's captured graph and re-arm its
                // phase so the next generation re-warms + recaptures cleanly. This
                // is the ONLY place the verify sibling is reset — within a
                // generation it survives the per-step Primary teardowns because it
                // binds only fixed-capacity, non-moving external buffers plus its
                // own private interior arena (see `invalidate_graph`).
                self.session.reset_verify_sibling_device_graph()?;
                state.reset_verify_graph_phase();
            }
            self.current_len = target_len;
            return Ok(());
        }
        if let Some(state) = &mut self.cpu_kv {
            // Append-only persistent buffers preserve rows [0, target_len), so a
            // rewind is just shrinking the exposed logical length; the next
            // appended step overwrites [target_len, ...) in place.
            state.set_logical_len(target_len)?;
            if target_len == 0 {
                self.last_hidden = None;
            }
            self.current_len = target_len;
            return Ok(());
        }
        if target_len == 0 {
            self.past.clear();
            self.current_len = 0;
            self.last_hidden = None;
            return Ok(());
        }
        let mut skip_names: HashSet<String> = self
            .session
            .inputs()
            .iter()
            .filter(|meta| is_recurrent_state_shape(&meta.shape))
            .map(|meta| meta.name.clone())
            .collect();
        // Every typed CSA/HCA state tensor is restored atomically. Record
        // buffers advance on a compression cursor rather than token count, and
        // fixed carries have no sequence axis at all; neither may enter generic
        // penultimate-axis prefix slicing.
        if !self.compressed_state.is_empty() {
            skip_names.extend(self.compressed_state.state_past_names().cloned());
        }
        let rewound = stage_host_prefix_rewind(&self.past, &skip_names, target_len)?;
        for (name, tensor) in rewound {
            let slot = self
                .past
                .get_mut(&name)
                .with_context(|| format!("native KV tensor '{name}' disappeared during rewind"))?;
            *slot = tensor;
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

pub(crate) struct NativeLoopAdapter<'a> {
    pub(crate) session: &'a mut NativeDecodeSession,
    pub(crate) prompt_tokens: Vec<TokenId>,
    pub(crate) pending_tokens: Vec<TokenId>,
    /// Device-token-loop chain depth (`0` = disabled). When `>= 2`, a greedy
    /// step runs a device-resident chained replay that returns several tokens at
    /// once; `lookahead` buffers the tokens beyond the one returned to the loop
    /// so the shared loop keeps its per-token EOS / stop / callback semantics.
    pub(crate) device_loop_k: usize,
    pub(crate) lookahead: std::collections::VecDeque<TokenId>,
}

impl DecodeLoopBackend for NativeLoopAdapter<'_> {
    fn context_len(&self) -> usize {
        self.session.current_len() + self.pending_tokens.len()
    }

    fn processor_prompt_tokens(&self) -> &[TokenId] {
        &self.prompt_tokens
    }

    fn next_logits(&mut self) -> anyhow::Result<Vec<f32>> {
        let past_len = self.session.current_len();
        self.session
            .decode(&self.pending_tokens, past_len)?
            .pop()
            .context("native decoder produced no logits")
    }

    fn greedy_fastpath_supported(&self) -> bool {
        self.session.cuda.is_none()
            || self
                .session
                .cuda
                .as_ref()
                .is_some_and(DecodeCudaState::greedy_fastpath_supported)
    }

    fn next_token_greedy(&mut self) -> anyhow::Result<TokenId> {
        if self.pending_tokens.len() != 1 {
            let past_len = self.session.current_len();
            return self
                .session
                .decode_argmax(&self.pending_tokens, past_len)?
                .context("native decoder did not return an argmax token");
        }
        // Serve buffered tokens from a prior device-token-loop chain first; the
        // device already advanced past them, so no forward is run here.
        if let Some(token) = self.lookahead.pop_front() {
            return Ok(token);
        }
        if self.device_loop_k >= 2 {
            let past_len = self.session.current_len();
            let seed = self.pending_tokens[0];
            let mut tokens =
                self.session
                    .decode_cuda_greedy_loop(seed, past_len, self.device_loop_k)?;
            if tokens.is_empty() {
                bail!("native device token loop returned no tokens");
            }
            let first = tokens.remove(0);
            self.lookahead.extend(tokens);
            return Ok(first);
        }
        let past_len = self.session.current_len();
        self.session
            .decode_argmax(&self.pending_tokens, past_len)?
            .context("native decoder did not return an argmax token")
    }

    fn commit_token(&mut self, token_id: TokenId) -> anyhow::Result<()> {
        self.pending_tokens.clear();
        self.pending_tokens.push(token_id);
        Ok(())
    }
}
