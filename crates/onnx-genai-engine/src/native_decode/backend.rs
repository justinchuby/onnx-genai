use super::*;

impl DecodeBackend for NativeDecodeSession {
    fn decode(&mut self, token_ids: &[TokenId], past_len: usize) -> anyhow::Result<Vec<Vec<f32>>> {
        self.decode_with_step_inputs(token_ids, past_len, &[])
    }

    fn decode_argmax(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Option<u32>> {
        if token_ids.is_empty() {
            bail!("native decode requires at least one token");
        }
        if past_len != self.current_len {
            bail!(
                "native decode past length mismatch: caller supplied {past_len}, adapter holds {}",
                self.current_len
            );
        }
        self.maybe_enable_decode_inline(token_ids);
        if self.cuda.is_some() {
            if token_ids.len() == 1 {
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

    fn supports_argmax(&self) -> bool {
        true
    }
}

impl NativeDecodeSession {
    pub fn activation_memory_plan_stats(&self) -> Option<crate::ActivationMemoryPlanSummary> {
        self.session.activation_memory_plan_stats().map(Into::into)
    }

    pub(super) fn rewind_inner(&mut self, target_len: usize) -> anyhow::Result<()> {
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
        let recurrent_names: HashSet<String> = self
            .session
            .inputs()
            .iter()
            .filter(|meta| is_recurrent_state_shape(&meta.shape))
            .map(|meta| meta.name.clone())
            .collect();
        for (name, tensor) in &mut self.past {
            // Recurrent states are destructive rolling caches with no per-step
            // history to slice; leave them intact (greedy decode never rewinds,
            // and speculative rewind of a recurrent state is unsupported).
            if recurrent_names.contains(name) {
                continue;
            }
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

pub(crate) struct NativeLoopAdapter<'a> {
    pub(crate) session: &'a mut NativeDecodeSession,
    pub(crate) prompt_tokens: Vec<TokenId>,
    pub(crate) pending_tokens: Vec<TokenId>,
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
