//! Batched static-cache generation path.

use crate::config::{
    EngineDecodeBackend, FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest,
    GenerateResult,
};
use crate::decode::ModelDecodePath;
use crate::decode_loop::{
    DecodeLoopState, commit_selected_token, finish_result, logprob_for_token, reached_context_limit,
};
use crate::engine::Engine;
use crate::logits::{ProcessorChain, ProcessorContext, TokenId};
use crate::processors::{
    build_processor_chain, ensure_constrained_finish, select_next_token_with_rng,
};
use crate::sampling::SamplingRng;
use anyhow::Context;
use onnx_genai_ort::Tokenizer;
use onnx_genai_ort::decode::{
    BatchedDecodeSession, BatchedSharedBufferDecodeSession, SharedBufferBatchOptions,
};
use onnx_genai_ort::{BatchedStaticCacheDecodeSession, StaticCacheDecodeOptions};
use onnx_genai_scheduler::{PreemptionPolicy, Priority, PriorityPolicy, Scheduler};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContinuousBatchHandle {
    pub id: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContinuousBatchEvent {
    Token {
        handle: ContinuousBatchHandle,
        token: crate::config::GenerateToken,
    },
    Finished {
        handle: ContinuousBatchHandle,
        result: GenerateResult,
    },
}

#[derive(Debug)]
pub enum ContinuousBatchAdmission {
    Assigned {
        handle: ContinuousBatchHandle,
    },
    Rejected {
        handle: ContinuousBatchHandle,
        error: anyhow::Error,
    },
}

struct BatchRow {
    result_index: usize,
    physical_row: usize,
    context_tokens: Vec<TokenId>,
    options: GenerateOptions,
    chain: ProcessorChain,
    max_context: Option<usize>,
    state: DecodeLoopState,
    pending_logits: Option<Vec<f32>>,
    active: bool,
}

impl BatchRow {
    fn processor_context(&self) -> ProcessorContext {
        ProcessorContext {
            prompt_tokens: self.context_tokens.clone(),
            generated_tokens: self.state.generated_tokens.clone(),
            generated_text: self.state.generated_text.clone(),
            step: self.state.step,
        }
    }
}

struct PendingContinuousRequest {
    handle: ContinuousBatchHandle,
    prompt_tokens: Vec<TokenId>,
    options: GenerateOptions,
    chain: ProcessorChain,
    max_context: Option<usize>,
}

struct ContinuousBatchRow {
    handle: ContinuousBatchHandle,
    physical_row: usize,
    context_tokens: Vec<TokenId>,
    options: GenerateOptions,
    chain: ProcessorChain,
    max_context: Option<usize>,
    state: DecodeLoopState,
    pending_logits: Option<Vec<f32>>,
}

impl ContinuousBatchRow {
    fn processor_context(&self) -> ProcessorContext {
        ProcessorContext {
            prompt_tokens: self.context_tokens.clone(),
            generated_tokens: self.state.generated_tokens.clone(),
            generated_text: self.state.generated_text.clone(),
            step: self.state.step,
        }
    }
}

/// Operator-facing description of how many sequences the engine's decode path
/// can advance in a single shared forward pass.
///
/// This is sourced from the decode path itself — the resolved
/// [`EngineDecodeBackend`](crate::EngineDecodeBackend) plus the model-I/O
/// [`ModelDecodePath`] selection — **not** from whether an ORT decoder session
/// happens to be present. Before this type existed, the only signal an operator
/// had was [`Engine::continuous_batch_manager`] returning `Err`, which the
/// server swallowed into a debug/info-level "using per-request engine path" log
/// line. On the native backend that `Err` is structural and permanent (batch and
/// query-seq are pinned to 1 in `native_decode/cuda.rs`), so reporting it as a
/// first-class capability lets `--max-batch` and `/v1/resources` tell the truth
/// instead of accepting a batch width that silently has no effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchingCapability {
    /// Maximum number of sequences that can be decoded concurrently in one
    /// shared forward pass. `None` means "not structurally capped": batching is
    /// supported and bounded only by memory and the operator's `--max-batch`.
    /// `Some(1)` means the decode path can only ever advance one sequence, so
    /// continuous batching is unavailable regardless of the requested width.
    max_concurrent_sequences: Option<usize>,
    /// Operator-facing explanation naming the backend / decode path and the
    /// reason for the limit. Safe to log and to surface over `/v1/resources`.
    reason: String,
}

impl BatchingCapability {
    /// Whether more than one sequence can share a decode step.
    pub fn supports_batching(&self) -> bool {
        !matches!(self.max_concurrent_sequences, Some(cap) if cap <= 1)
    }

    /// The structural cap on concurrently-decoded sequences, if any. `None`
    /// means "bounded only by memory / configuration".
    pub fn max_concurrent_sequences(&self) -> Option<usize> {
        self.max_concurrent_sequences
    }

    /// Operator-facing reason string describing the backend / decode path.
    pub fn reason(&self) -> &str {
        &self.reason
    }

    /// Whether a requested batch width can be honored by this decode path.
    pub fn allows(&self, requested: usize) -> bool {
        match self.max_concurrent_sequences {
            None => true,
            Some(cap) => requested <= cap,
        }
    }

    /// The batch width that will actually take effect for `requested`: the
    /// request clamped to `[1, cap]` (or just floored at 1 when uncapped).
    pub fn effective_max_batch(&self, requested: usize) -> usize {
        match self.max_concurrent_sequences {
            None => requested.max(1),
            Some(cap) => requested.clamp(1, cap),
        }
    }
}

/// Synchronous continuous-batch manager for STATIC-CACHE models.
///
/// Requests are submitted into a FIFO queue and admitted into a fixed number of
/// physical decode rows. Each `step` samples one token for rows that have
/// pending logits, emits token/result events, evicts finished rows, admits queued
/// requests into freed slots, then prepares logits for the next step.
pub struct ContinuousBatchManager<'a> {
    decode: Box<dyn BatchedDecodeSession<'a> + 'a>,
    tokenizer: &'a Tokenizer,
    metadata_max_context: Option<usize>,
    static_max_len: usize,
    queue: VecDeque<PendingContinuousRequest>,
    rows: Vec<Option<ContinuousBatchRow>>,
    admissions: VecDeque<ContinuousBatchAdmission>,
    events: VecDeque<ContinuousBatchEvent>,
    next_handle: usize,
}

impl<'a> ContinuousBatchManager<'a> {
    fn new(
        mut decode: Box<dyn BatchedDecodeSession<'a> + 'a>,
        tokenizer: &'a Tokenizer,
        metadata_max_context: Option<usize>,
        max_batch: usize,
    ) -> anyhow::Result<Self> {
        if max_batch == 0 {
            anyhow::bail!("continuous batch max_batch must be greater than zero");
        }
        for row in 0..max_batch {
            decode
                .deactivate_row(row)
                .map_err(|e| anyhow::anyhow!("Failed to initialize continuous row {row}: {e}"))?;
        }
        let static_max_len = decode.max_len();
        Ok(Self {
            decode,
            tokenizer,
            metadata_max_context,
            static_max_len,
            queue: VecDeque::new(),
            rows: (0..max_batch).map(|_| None).collect(),
            admissions: VecDeque::new(),
            events: VecDeque::new(),
            next_handle: 0,
        })
    }

    /// Queue a request for the next available decode row.
    pub fn submit(&mut self, request: GenerateRequest) -> anyhow::Result<ContinuousBatchHandle> {
        let handle = ContinuousBatchHandle {
            id: self.next_handle,
        };
        self.next_handle += 1;
        request.options.validate()?;
        let mut options = request.options;
        if options.eos_token_id.is_none() {
            options.eos_token_id = self.tokenizer.eos_token_id();
        }
        let prompt_tokens = match request.prompt {
            GeneratePrompt::TokenIds(tokens) => tokens,
            GeneratePrompt::Text(text) => self
                .tokenizer
                .encode(&text)
                .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e}"))?,
        };
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        let max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(&options, Some(self.tokenizer))?;
        if reached_context_limit(prompt_tokens.len(), max_context) {
            self.admissions
                .push_back(ContinuousBatchAdmission::Assigned { handle });
            ensure_constrained_finish(&options, "", FinishReason::Length)?;
            self.events.push_back(ContinuousBatchEvent::Finished {
                handle,
                result: finish_result(self.tokenizer, &[], FinishReason::Length, 0, None)?,
            });
            return Ok(handle);
        }
        self.queue.push_back(PendingContinuousRequest {
            handle,
            prompt_tokens,
            options,
            chain,
            max_context,
        });
        Ok(handle)
    }

    /// Advance all rows with pending logits by one generated token.
    pub fn step(&mut self) -> anyhow::Result<()> {
        self.admit_available_rows();
        let ready_rows = self
            .rows
            .iter()
            .enumerate()
            .filter_map(|(row_index, row)| {
                row.as_ref()
                    .and_then(|row| row.pending_logits.is_some().then_some(row_index))
            })
            .collect::<Vec<_>>();

        for row_index in ready_rows {
            let mut row = self.rows[row_index]
                .take()
                .context("ready continuous row disappeared")?;
            let finished = self.advance_row(&mut row)?;
            if finished {
                self.decode
                    .deactivate_row(row.physical_row)
                    .map_err(|e| anyhow::anyhow!("Failed to deactivate continuous row: {e}"))?;
            } else {
                self.rows[row_index] = Some(row);
            }
        }

        self.decode_next_pending_rows()
    }

    /// Assign queued requests to currently available decode rows.
    pub fn admit_pending(&mut self) {
        self.admit_available_rows();
    }

    /// Drain token/result events emitted by previous `submit` or `step` calls.
    pub fn poll(&mut self) -> Vec<ContinuousBatchEvent> {
        self.events.drain(..).collect()
    }

    pub fn poll_admissions(&mut self) -> Vec<ContinuousBatchAdmission> {
        self.admissions.drain(..).collect()
    }

    pub fn cancel_pending(&mut self, handle: ContinuousBatchHandle) -> bool {
        let Some(index) = self
            .queue
            .iter()
            .position(|pending| pending.handle == handle)
        else {
            return false;
        };
        self.queue.remove(index);
        true
    }

    pub fn max_batch(&self) -> usize {
        self.rows.len()
    }

    pub fn pending_len(&self) -> usize {
        self.queue.len()
    }

    pub fn active_len(&self) -> usize {
        self.rows.iter().filter(|row| row.is_some()).count()
    }

    pub fn has_pending_work(&self) -> bool {
        !self.queue.is_empty() || self.active_len() > 0
    }

    pub fn is_idle(&self) -> bool {
        !self.has_pending_work() && self.events.is_empty()
    }

    fn max_context_for_request(&self, options: &GenerateOptions) -> Option<usize> {
        let configured = self.metadata_max_context.or(options.max_context);
        Some(configured.map_or(self.static_max_len, |limit| limit.min(self.static_max_len)))
    }

    fn admit_available_rows(&mut self) {
        while !self.queue.is_empty() {
            let Some(row_index) = self.rows.iter().position(|row| row.is_none()) else {
                break;
            };
            let pending = self.queue.pop_front().expect("queue checked non-empty");
            let handle = pending.handle;
            match self.admit_pending_into_row(pending, row_index) {
                Ok(()) => self
                    .admissions
                    .push_back(ContinuousBatchAdmission::Assigned { handle }),
                Err(error) => self
                    .admissions
                    .push_back(ContinuousBatchAdmission::Rejected { handle, error }),
            }
        }
    }

    fn admit_pending_into_row(
        &mut self,
        pending: PendingContinuousRequest,
        row_index: usize,
    ) -> anyhow::Result<()> {
        self.decode
            .assign_row(row_index)
            .map_err(|e| anyhow::anyhow!("Failed to assign continuous row: {e}"))?;
        let rng = SamplingRng::for_row(pending.options.seed, row_index);
        let loop_state = DecodeLoopState::with_rng(0, rng, pending.options.top_logprobs);
        let row = ContinuousBatchRow {
            handle: pending.handle,
            physical_row: row_index,
            context_tokens: pending.prompt_tokens,
            options: pending.options,
            chain: pending.chain,
            max_context: pending.max_context,
            state: loop_state,
            pending_logits: None,
        };
        self.rows[row_index] = Some(row);
        Ok(())
    }

    fn advance_row(&mut self, row: &mut ContinuousBatchRow) -> anyhow::Result<bool> {
        let mut logits = row
            .pending_logits
            .take()
            .context("active continuous row has no pending logits")?;
        let context = row.processor_context();
        let token_id = select_next_token_with_rng(
            &mut logits,
            &context,
            &row.options,
            &row.chain,
            &mut row.state.rng,
        );
        if let (Some(top_logprobs), Some(logprobs)) =
            (row.options.top_logprobs, row.state.logprobs.as_mut())
        {
            logprobs.push(logprob_for_token(&logits, token_id, top_logprobs));
        }
        row.context_tokens.push(token_id);

        let mut emitted_token = None;
        let mut callback = |token| {
            emitted_token = Some(token);
            Ok(())
        };
        let finish_reason = commit_selected_token(
            &mut row.state,
            &row.context_tokens,
            token_id,
            &row.options,
            &row.chain,
            self.tokenizer,
            Some(&mut callback),
        )?;
        if let Some(token) = emitted_token {
            self.events.push_back(ContinuousBatchEvent::Token {
                handle: row.handle,
                token,
            });
        }

        let finish_reason = match finish_reason {
            Some(reason) => Some(reason),
            None if row.state.generated_tokens.len() >= row.options.max_new_tokens => {
                ensure_constrained_finish(
                    &row.options,
                    &row.state.generated_text,
                    FinishReason::MaxTokens,
                )?;
                Some(FinishReason::MaxTokens)
            }
            None if reached_context_limit(row.context_tokens.len(), row.max_context) => {
                ensure_constrained_finish(
                    &row.options,
                    &row.state.generated_text,
                    FinishReason::Length,
                )?;
                Some(FinishReason::Length)
            }
            None => None,
        };

        if let Some(reason) = finish_reason {
            self.events.push_back(ContinuousBatchEvent::Finished {
                handle: row.handle,
                result: finish_result(
                    self.tokenizer,
                    &row.state.generated_tokens,
                    reason,
                    row.state.prefix_cache_hit_len,
                    row.state.logprobs.as_deref(),
                )?,
            });
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn decode_next_pending_rows(&mut self) -> anyhow::Result<()> {
        let advancing_rows = self
            .rows
            .iter()
            .flatten()
            .filter(|row| row.pending_logits.is_none())
            .map(|row| row.physical_row)
            .collect::<Vec<_>>();
        if advancing_rows.is_empty() {
            return Ok(());
        }
        let active_rows = self.decode.active_rows();
        if advancing_rows.len() == active_rows.len() {
            let mut input_ids = vec![0_i64; active_rows.len()];
            let mut position_ids = vec![0_i64; active_rows.len()];
            let mut completes_context = vec![false; active_rows.len()];
            for (active_index, &logical_row) in active_rows.iter().enumerate() {
                let row = self.rows[logical_row]
                    .as_ref()
                    .context("active continuous row is not assigned")?;
                let row_len = self
                    .decode
                    .row_len(logical_row)
                    .map_err(|e| anyhow::anyhow!("Failed to read continuous row length: {e}"))?;
                let token = *row.context_tokens.get(row_len).with_context(|| {
                    format!(
                        "continuous row {logical_row} has no token to advance at offset {row_len}"
                    )
                })?;
                input_ids[active_index] = i64::from(token);
                position_ids[active_index] = row_len as i64;
                completes_context[active_index] = row_len + 1 == row.context_tokens.len();
            }
            let logits = self
                .decode
                .step_active(&input_ids, &position_ids)
                .map_err(|e| anyhow::anyhow!("Continuous active static-cache step failed: {e}"))?;
            for (active_index, logical_row) in active_rows.into_iter().enumerate() {
                if completes_context[active_index] {
                    let row = self.rows[logical_row]
                        .as_mut()
                        .context("active continuous row is not assigned")?;
                    row.pending_logits = Some(row_logits(&logits, active_index, 0)?);
                }
            }
            return Ok(());
        }

        let mut input_ids = vec![0_i64; self.max_batch()];
        let mut position_ids = vec![0_i64; self.max_batch()];
        let mut advance_rows = vec![false; self.max_batch()];
        let mut completes_context = vec![false; self.max_batch()];
        for row in self.rows.iter().flatten() {
            if row.pending_logits.is_none() {
                let row_len = self
                    .decode
                    .row_len(row.physical_row)
                    .map_err(|e| anyhow::anyhow!("Failed to read continuous row length: {e}"))?;
                let token = *row.context_tokens.get(row_len).with_context(|| {
                    format!(
                        "continuous row {} has no token to advance at offset {row_len}",
                        row.physical_row
                    )
                })?;
                input_ids[row.physical_row] = i64::from(token);
                position_ids[row.physical_row] = row_len as i64;
                advance_rows[row.physical_row] = true;
                completes_context[row.physical_row] = row_len + 1 == row.context_tokens.len();
            }
        }
        let logits = self
            .decode
            .step_select(&input_ids, &position_ids, &advance_rows)
            .map_err(|e| anyhow::anyhow!("Continuous static-cache decode step failed: {e}"))?;
        for row in self.rows.iter_mut().flatten() {
            if advance_rows[row.physical_row] && completes_context[row.physical_row] {
                row.pending_logits = Some(row_logits(&logits, row.physical_row, 0)?);
            }
        }
        Ok(())
    }
}

impl Engine {
    /// Generate a fixed batch of independent requests on a STATIC-CACHE model.
    ///
    /// Each request owns its processor chain, sampling options, stop conditions,
    /// and context limit. Prompt prefill is batched by row, then every decode
    /// iteration runs one ORT forward for all active rows and demuxes row logits.
    /// Finished rows are deactivated so they are no longer sampled or committed;
    /// the current ORT static-cache runner still executes the original fixed
    /// physical batch until row-view compaction lands in the backend.
    pub fn generate_batched_static(
        &mut self,
        requests: Vec<GenerateRequest>,
    ) -> anyhow::Result<Vec<GenerateResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        if !matches!(self.decode_path, ModelDecodePath::StaticCache { .. }) {
            anyhow::bail!(
                "batched static generation requires a STATIC-CACHE model; past/present batching is deferred"
            );
        }

        let mut results = vec![None; requests.len()];
        let mut rows = Vec::new();
        for (result_index, request) in requests.into_iter().enumerate() {
            request.options.validate()?;
            let mut options = request.options;
            if options.eos_token_id.is_none() {
                options.eos_token_id = self.tokenizer.eos_token_id();
            }
            let prompt_tokens = match request.prompt {
                GeneratePrompt::TokenIds(tokens) => tokens,
                GeneratePrompt::Text(text) => self
                    .tokenizer
                    .encode(&text)
                    .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e}"))?,
            };
            if prompt_tokens.is_empty() {
                anyhow::bail!("prompt must contain at least one token");
            }
            let max_context = self.batched_max_context_for_request(&options);
            let chain = build_processor_chain(&options, Some(&self.tokenizer))?;
            if reached_context_limit(prompt_tokens.len(), max_context) {
                ensure_constrained_finish(&options, "", FinishReason::Length)?;
                results[result_index] = Some(finish_result(
                    &self.tokenizer,
                    &[],
                    FinishReason::Length,
                    0,
                    None,
                )?);
                continue;
            }
            let physical_row = rows.len();
            let rng = SamplingRng::for_row(options.seed, physical_row);
            let loop_state = DecodeLoopState::with_rng(0, rng, options.top_logprobs);
            rows.push(BatchRow {
                result_index,
                physical_row,
                context_tokens: prompt_tokens,
                options,
                chain,
                max_context,
                state: loop_state,
                pending_logits: None,
                active: true,
            });
        }

        if rows.is_empty() {
            return collect_batch_results(results);
        }

        let io = self
            .metadata
            .model
            .as_ref()
            .and_then(|model| model.io.as_ref());
        let mut decode = BatchedStaticCacheDecodeSession::new(
            self.session
                .as_deref()
                .context("ORT decoder session is unavailable")?,
            StaticCacheDecodeOptions {
                batch_size: i64::try_from(rows.len()).context("batch size exceeds i64")?,
            },
            io,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create batched static-cache session: {e}"))?;

        prefill_batched_rows(&mut decode, &mut rows)?;
        let mut active_rows = rows.len();
        while active_rows > 0 {
            for row in rows.iter_mut().filter(|row| row.active) {
                let mut logits = row
                    .pending_logits
                    .take()
                    .context("active batch row has no pending logits")?;
                let context = row.processor_context();
                let token_id = select_next_token_with_rng(
                    &mut logits,
                    &context,
                    &row.options,
                    &row.chain,
                    &mut row.state.rng,
                );
                if let (Some(top_logprobs), Some(logprobs)) =
                    (row.options.top_logprobs, row.state.logprobs.as_mut())
                {
                    logprobs.push(logprob_for_token(&logits, token_id, top_logprobs));
                }
                row.context_tokens.push(token_id);

                let finish_reason = commit_selected_token(
                    &mut row.state,
                    &row.context_tokens,
                    token_id,
                    &row.options,
                    &row.chain,
                    &self.tokenizer,
                    None,
                )?;

                let finish_reason = match finish_reason {
                    Some(reason) => Some(reason),
                    None if row.state.generated_tokens.len() >= row.options.max_new_tokens => {
                        ensure_constrained_finish(
                            &row.options,
                            &row.state.generated_text,
                            FinishReason::MaxTokens,
                        )?;
                        Some(FinishReason::MaxTokens)
                    }
                    None if reached_context_limit(row.context_tokens.len(), row.max_context) => {
                        ensure_constrained_finish(
                            &row.options,
                            &row.state.generated_text,
                            FinishReason::Length,
                        )?;
                        Some(FinishReason::Length)
                    }
                    None => None,
                };

                if let Some(reason) = finish_reason {
                    results[row.result_index] = Some(finish_result(
                        &self.tokenizer,
                        &row.state.generated_tokens,
                        reason,
                        row.state.prefix_cache_hit_len,
                        row.state.logprobs.as_deref(),
                    )?);
                    decode
                        .deactivate_row(row.physical_row)
                        .map_err(|e| anyhow::anyhow!("Failed to deactivate batch row: {e}"))?;
                    row.active = false;
                    active_rows -= 1;
                }
            }

            if active_rows > 0 {
                decode_next_batched_tokens(&mut decode, &mut rows)?;
            }
        }

        collect_batch_results(results)
    }

    /// Report how many sequences this engine's decode path can advance in a
    /// single shared forward pass, and why.
    ///
    /// The answer is derived from the resolved decode backend and the model-I/O
    /// [`ModelDecodePath`], **not** from whether an ORT decoder session happens
    /// to be present. That distinction is the whole point: on the native backend
    /// [`Self::continuous_batch_manager`] always fails because there is no ORT
    /// decoder session, but the honest reason is that native decode pins batch
    /// and query-seq to 1 as a structural invariant (`native_decode/cuda.rs`),
    /// not that a session is momentarily missing.
    pub fn batching_capability(&self) -> BatchingCapability {
        // The native decode path advances exactly one sequence per step
        // regardless of the model's KV I/O shape, so it is answered first and
        // unconditionally. `native_decode/cuda.rs` pins both the batch dimension
        // and the query-sequence dimension to 1 as a structural decode
        // invariant, which is not a tunable.
        if self.decode_backend == EngineDecodeBackend::Native {
            return BatchingCapability {
                max_concurrent_sequences: Some(1),
                reason: "the native decode backend advances exactly one sequence \
                         per step: batch and query-seq are pinned to 1 as a \
                         structural decode invariant (native_decode/cuda.rs), so \
                         continuous batching is unavailable"
                    .to_string(),
            };
        }
        match self.decode_path {
            ModelDecodePath::StaticCache { .. } => BatchingCapability {
                max_concurrent_sequences: None,
                reason: "ONNX Runtime static-cache decode advances a shared batch \
                         of sequences per step; concurrency is bounded only by \
                         memory and the configured maximum batch size"
                    .to_string(),
            },
            ModelDecodePath::PastPresent {
                shared_buffer: true,
                ..
            } => BatchingCapability {
                max_concurrent_sequences: None,
                reason: "shared-KV-buffer past/present decode advances a shared \
                         batch of sequences per step; concurrency is bounded only \
                         by memory and the configured maximum batch size"
                    .to_string(),
            },
            ModelDecodePath::PastPresent { .. } => BatchingCapability {
                max_concurrent_sequences: Some(1),
                reason: "this past/present model is not using a shared KV buffer: \
                         the execution provider did not report fixed-capacity \
                         present binding, or it was not opted into at launch, so \
                         only one sequence can be decoded at a time"
                    .to_string(),
            },
            ModelDecodePath::Legacy => BatchingCapability {
                max_concurrent_sequences: Some(1),
                reason: "this legacy past/present model has no shared KV buffer \
                         and cannot batch: continuous batching requires a \
                         static-cache or shared-buffer past/present model"
                    .to_string(),
            },
        }
    }

    /// Create a lower-level continuous-batch manager for incremental serving.
    pub fn continuous_batch_manager(
        &self,
        max_batch: usize,
    ) -> anyhow::Result<ContinuousBatchManager<'_>> {
        if max_batch == 0 {
            anyhow::bail!("continuous batch max_batch must be greater than zero");
        }
        // Native backend: refuse structurally with the honest reason, not the
        // misleading "session unavailable" (there is deliberately no ORT decoder
        // session on the native path). `ContinuousBatchManager` drives a *ragged*
        // `BatchedDecodeSession`: per-row `row_len`, `assign_row`/`deactivate_row`,
        // `step_select` with a per-row `advance_rows` mask and per-row
        // `position_ids`, and host `[B, 1, vocab]` logits fed to the host sampler
        // (`advance_row` -> `select_next_token_with_rng`). The native CUDA
        // persistent batch path is *uniform*: `DecodeCudaState::extend_mask`
        // writes one identical mask window to every row and sets a single shared
        // logical length, `decode_cuda_greedy_batch` steps every row at one shared
        // position (`vec![past_len; batch]`), there is no per-row length, and the
        // device-argmax fast path returns tokens, not logits. Genuinely different
        // requests can be batched only when they share a length and step in
        // lockstep (see `profile_native --solo-equivalence-prompts`, stage 2c);
        // mid-flight backfill of finished rows would make the batch ragged, which
        // the uniform path cannot represent. Wiring the manager onto native
        // requires per-row mask/position/length generalization in
        // native_decode/cuda.rs, tracked under #750 stage 2c.
        if self.decode_backend == EngineDecodeBackend::Native {
            anyhow::bail!(
                "continuous batching is unavailable on the native decode backend: the \
                 ContinuousBatchManager needs a ragged BatchedDecodeSession (per-row length, \
                 per-row advance/position, host logits for the sampler), but the native CUDA \
                 batch path is uniform (one shared mask window and one shared position per step, \
                 no per-row length, device-argmax tokens not logits). Same-length lockstep \
                 batching is exercised by `profile_native --solo-equivalence-prompts` (#750 \
                 stage 2c); ragged continuous batching needs per-row cuda.rs generalization."
            );
        }
        let session = self
            .session
            .as_deref()
            .context("ORT decoder session is unavailable")?;
        let batch_size = i64::try_from(max_batch).context("batch size exceeds i64")?;
        let decode: Box<dyn BatchedDecodeSession<'_> + '_> = match self.decode_path {
            ModelDecodePath::StaticCache { .. } => {
                let io = self
                    .metadata
                    .model
                    .as_ref()
                    .and_then(|model| model.io.as_ref());
                Box::new(
                    BatchedStaticCacheDecodeSession::new(
                        session,
                        StaticCacheDecodeOptions { batch_size },
                        io,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to create continuous static-cache session: {e}")
                    })?,
                )
            }
            ModelDecodePath::PastPresent {
                shared_buffer: true,
                max_len,
                ..
            } => {
                let max_len = max_len
                    .context("shared-buffer continuous batching requires a known max_len")?;
                let io = self
                    .metadata
                    .model
                    .as_ref()
                    .and_then(|model| model.io.as_ref());
                Box::new(
                    BatchedSharedBufferDecodeSession::new_with_io(
                        session,
                        SharedBufferBatchOptions {
                            batch_size,
                            max_len,
                        },
                        io,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to create continuous shared-buffer session: {e}")
                    })?,
                )
            }
            // These two refusals are reported separately because they have
            // OPPOSITE operator remedies. A past/present model reaching here has
            // `shared_buffer: false`, which is decided by
            // `supports_fixed_capacity_present_binding()` -- an execution-provider
            // capability plus an explicit opt-in -- so the same model on the same
            // disk can batch or not depending on how the server was launched. A
            // legacy model cannot batch under any launch. Collapsing them emits
            // one sentence that tells an operator to change the model when the
            // real fix may be an environment variable, and vice versa.
            ModelDecodePath::PastPresent { .. } => {
                anyhow::bail!(
                    "continuous batching requires a shared KV buffer, and this \
                     past/present model is not using one: the execution provider \
                     did not report fixed-capacity present binding, or it was not \
                     opted into at launch"
                );
            }
            ModelDecodePath::Legacy => {
                // This string is pinned CHARACTER BY CHARACTER by the README, by
                // check-perf-claims.test.js, and by batch_driver.rs's test. It
                // stays on one line and byte-identical: an operator matches a
                // quoted error against their own terminal, and reflowing it onto
                // a continuation would break every one of those without changing
                // a single thing a reader sees.
                anyhow::bail!(
                    "continuous batching requires a STATIC-CACHE or shared-buffer past/present model"
                );
            }
        };
        let metadata_max_context = self
            .metadata
            .model
            .as_ref()
            .and_then(|model| model.max_sequence_length);
        ContinuousBatchManager::new(decode, &self.tokenizer, metadata_max_context, max_batch)
    }

    /// Run requests to completion through a dynamic continuous batch.
    pub fn run_continuous_batch(
        &mut self,
        requests: Vec<GenerateRequest>,
        max_batch: usize,
    ) -> anyhow::Result<Vec<GenerateResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let expected_results = requests.len();
        let mut manager = self.continuous_batch_manager(max_batch)?;
        let mut results = vec![None; expected_results];
        for request in requests {
            manager.submit(request)?;
            collect_finished_events(manager.poll(), &mut results)?;
        }
        while results.iter().any(|result| result.is_none()) {
            if !manager.has_pending_work() {
                break;
            }
            manager.step()?;
            collect_finished_events(manager.poll(), &mut results)?;
        }
        collect_batch_results(results)
    }

    /// Run requests to completion through a continuous batch whose formation is
    /// driven by the [`Scheduler`], rather than by the manager's greedy
    /// self-admission.
    ///
    /// This is the engine-serving counterpart to [`Self::run_continuous_batch`]:
    /// instead of self-admitting every arrival, each iteration the scheduler
    /// decides which waiting requests are *eligible* to enter the batch — FCFS
    /// order gated by the shared KV byte budget and total-token ceiling — and
    /// only those requests are handed to the batch. Physical concurrency is
    /// bounded by the manager's `max_batch` decode rows, which run one *shared*
    /// batched forward pass per iteration; finished rows are backfilled from the
    /// admitted set so the batch stays continuously occupied.
    ///
    /// Because a request's tokens never depend on which rows share its batch,
    /// the per-request output is byte-identical to running each request on its
    /// own (`Engine::generate`) and to the greedy [`Self::run_continuous_batch`]
    /// — batching here is a throughput optimization, not an output change.
    ///
    /// The scheduler is run with preemption disabled: this batch owns its KV in
    /// the batched decode session's physical rows, which cannot be swapped out
    /// and resumed in place, so mid-flight eviction/swap of a running row is
    /// deferred (tracked with session-level continuous batching). Byte-budget
    /// admission still holds because each row reserves its worst-case footprint
    /// up front.
    pub fn run_continuous_batch_scheduled(
        &mut self,
        requests: Vec<GenerateRequest>,
        max_batch: usize,
    ) -> anyhow::Result<Vec<GenerateResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        if max_batch == 0 {
            anyhow::bail!("continuous batch max_batch must be greater than zero");
        }
        let expected_results = requests.len();

        // Tokenize every prompt up front so the scheduler and the batch manager
        // agree on the exact prompt length, and feed the manager token ids so no
        // re-tokenization can drift between the two.
        let mut prepared = Vec::with_capacity(expected_results);
        let mut token_requests = Vec::with_capacity(expected_results);
        for mut request in requests {
            let prompt_tokens = match &request.prompt {
                GeneratePrompt::TokenIds(tokens) => tokens.clone(),
                GeneratePrompt::Text(text) => self
                    .tokenizer
                    .encode(text)
                    .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e}"))?,
            };
            prepared.push((prompt_tokens.len(), request.options.max_new_tokens));
            request.prompt = GeneratePrompt::TokenIds(prompt_tokens);
            token_requests.push(request);
        }

        // A dedicated FCFS scheduler governs admission for this batch. Physical
        // concurrency is enforced by the manager's `max_batch` decode rows, so
        // the scheduler's own batch-size cap is opened up to the request count:
        // its role here is admission *eligibility* — ordering plus the shared
        // token/byte budget — not row count. Preemption is disabled because this
        // batch owns its KV in the decode session's physical rows, which cannot
        // be swapped out and resumed in place.
        let mut scheduler_config = self.scheduler.config().clone();
        scheduler_config.max_batch_size = expected_results.max(max_batch);
        scheduler_config.preemption_policy = PreemptionPolicy::Disabled;
        scheduler_config.priority_policy = PriorityPolicy::Fcfs;
        let mut scheduler = Scheduler::new(scheduler_config);

        // Every request waits in the scheduler keyed by its result index. Each
        // iteration the scheduler decides which waiting requests are eligible to
        // admit (subject to the token/byte budget); those — and only those — are
        // submitted into the manager, whose greedy `step()` then keeps its
        // physical rows continuously occupied via same-step backfill. Because the
        // eligible set is fed to the manager as spare queue entries (never fewer
        // rows than the manager can fill), the decode path is byte-identical to
        // `run_continuous_batch`; the scheduler simply gates which requests are
        // allowed into the batch.
        for (index, (prompt_len, max_new_tokens)) in prepared.iter().enumerate() {
            scheduler.enqueue_generate_request(
                index as u64,
                *prompt_len,
                (*max_new_tokens).max(1),
                Priority::Normal,
            );
        }

        let mut manager = self.continuous_batch_manager(max_batch)?;
        let mut results = vec![None; expected_results];
        let mut pending_requests: Vec<Option<GenerateRequest>> =
            token_requests.into_iter().map(Some).collect();
        // Manager handle id -> original result index (handles are minted lazily
        // as requests are admitted, so this preserves the caller's ordering).
        let mut handle_to_index: HashMap<usize, usize> = HashMap::new();

        while results.iter().any(|result| result.is_none()) {
            let decision = scheduler.schedule();
            let admitted_this_iter = decision.prefill.len();
            for seq_id in &decision.prefill {
                let index = *seq_id as usize;
                let request = pending_requests[index]
                    .take()
                    .with_context(|| format!("request {index} admitted twice"))?;
                let handle = manager.submit(request)?;
                handle_to_index.insert(handle.id, index);
            }

            // Progress guard: if nothing is running and the scheduler admitted
            // nothing this iteration, the batch cannot make progress (e.g. the
            // token budget is too small to fit even one queued request).
            if admitted_this_iter == 0 && manager.active_len() == 0 && !manager.has_pending_work() {
                if scheduler.waiting_count() > 0 {
                    anyhow::bail!(
                        "scheduler-driven continuous batch stalled: {} request(s) queued but none could be admitted (scheduler budget too small for max_batch={max_batch})",
                        scheduler.waiting_count()
                    );
                }
                break;
            }

            manager.step()?;
            for event in manager.poll() {
                if let ContinuousBatchEvent::Finished { handle, result } = event {
                    let index = *handle_to_index
                        .get(&handle.id)
                        .with_context(|| format!("continuous handle {} is unmapped", handle.id))?;
                    results[index] = Some(result);
                    scheduler.complete(index as u64);
                }
            }
        }

        collect_batch_results(results)
    }

    fn batched_max_context_for_request(&self, options: &GenerateOptions) -> Option<usize> {
        let configured = self
            .metadata
            .model
            .as_ref()
            .and_then(|model| model.max_sequence_length)
            .or(options.max_context);
        let runtime_max = match self.decode_path {
            ModelDecodePath::StaticCache { max_len } => Some(max_len),
            ModelDecodePath::PastPresent {
                shared_buffer: true,
                max_len,
                ..
            } => max_len,
            ModelDecodePath::PastPresent { .. } | ModelDecodePath::Legacy => None,
        };
        match runtime_max {
            Some(runtime_max) => {
                Some(configured.map_or(runtime_max, |limit| limit.min(runtime_max)))
            }
            None => configured,
        }
    }
}

fn collect_finished_events(
    events: Vec<ContinuousBatchEvent>,
    results: &mut [Option<GenerateResult>],
) -> anyhow::Result<()> {
    for event in events {
        if let ContinuousBatchEvent::Finished { handle, result } = event {
            let slot = results
                .get_mut(handle.id)
                .with_context(|| format!("continuous handle {} is out of range", handle.id))?;
            *slot = Some(result);
        }
    }
    Ok(())
}

fn prefill_batched_rows(
    decode: &mut BatchedStaticCacheDecodeSession<'_>,
    rows: &mut [BatchRow],
) -> anyhow::Result<()> {
    let prompt_len = rows[0].context_tokens.len();
    let equal_prompt_len = rows
        .iter()
        .all(|row| row.context_tokens.len() == prompt_len);
    if equal_prompt_len {
        let mut input_ids = Vec::with_capacity(rows.len() * prompt_len);
        let mut position_ids = Vec::with_capacity(rows.len() * prompt_len);
        for row in rows.iter() {
            input_ids.extend(row.context_tokens.iter().map(|&token| i64::from(token)));
            position_ids.extend((0..prompt_len).map(|pos| pos as i64));
        }
        let logits = decode
            .prefill(&input_ids, &position_ids)
            .map_err(|e| anyhow::anyhow!("Batched static-cache prefill failed: {e}"))?;
        for row in rows.iter_mut() {
            row.pending_logits = Some(row_logits(&logits, row.physical_row, prompt_len - 1)?);
        }
        return Ok(());
    }

    let max_prompt_len = rows
        .iter()
        .map(|row| row.context_tokens.len())
        .max()
        .unwrap_or(0);
    for offset in 0..max_prompt_len {
        let mut input_ids = vec![0_i64; rows.len()];
        let mut position_ids = vec![0_i64; rows.len()];
        let mut advance_rows = vec![false; rows.len()];
        for row in rows.iter() {
            if let Some(&token) = row.context_tokens.get(offset) {
                input_ids[row.physical_row] = i64::from(token);
                position_ids[row.physical_row] = decode
                    .row_len(row.physical_row)
                    .map_err(|e| anyhow::anyhow!("Failed to read batch row length: {e}"))?
                    as i64;
                advance_rows[row.physical_row] = true;
            }
        }
        let logits = decode
            .step_select(&input_ids, &position_ids, &advance_rows)
            .map_err(|e| anyhow::anyhow!("Batched static-cache ragged prefill failed: {e}"))?;
        for row in rows.iter_mut().filter(|row| advance_rows[row.physical_row]) {
            row.pending_logits = Some(row_logits(&logits, row.physical_row, 0)?);
        }
    }
    Ok(())
}

fn decode_next_batched_tokens(
    decode: &mut BatchedStaticCacheDecodeSession<'_>,
    rows: &mut [BatchRow],
) -> anyhow::Result<()> {
    let mut input_ids = vec![0_i64; rows.len()];
    let mut position_ids = vec![0_i64; rows.len()];
    let mut advance_rows = vec![false; rows.len()];
    for row in rows.iter().filter(|row| row.active) {
        let token = *row
            .context_tokens
            .last()
            .context("active batch row has empty context")?;
        input_ids[row.physical_row] = i64::from(token);
        position_ids[row.physical_row] = decode
            .row_len(row.physical_row)
            .map_err(|e| anyhow::anyhow!("Failed to read batch row length: {e}"))?
            as i64;
        advance_rows[row.physical_row] = true;
    }
    let logits = decode
        .step_select(&input_ids, &position_ids, &advance_rows)
        .map_err(|e| anyhow::anyhow!("Batched static-cache decode step failed: {e}"))?;
    for row in rows.iter_mut().filter(|row| row.active) {
        row.pending_logits = Some(row_logits(&logits, row.physical_row, 0)?);
    }
    Ok(())
}

fn row_logits(
    logits: &onnx_genai_ort::Value,
    row: usize,
    seq_index: usize,
) -> anyhow::Result<Vec<f32>> {
    BatchedStaticCacheDecodeSession::row_logits(logits, row, seq_index)
        .map_err(|e| anyhow::anyhow!("Failed to extract row logits: {e}"))
}

fn collect_batch_results(
    results: Vec<Option<GenerateResult>>,
) -> anyhow::Result<Vec<GenerateResult>> {
    results
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.with_context(|| format!("batch request {index} did not finish"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::sample_categorical;
    use onnx_genai_ort::decode::BatchedDecodeSession;
    use onnx_genai_ort::{OrtError, Value};
    use std::path::Path;

    struct RejectAssignDecode;

    impl<'a> BatchedDecodeSession<'a> for RejectAssignDecode {
        fn batch_size(&self) -> usize {
            1
        }

        fn max_len(&self) -> usize {
            32
        }

        fn row_len(&self, _row: usize) -> onnx_genai_ort::Result<usize> {
            Ok(0)
        }

        fn active_rows(&self) -> Vec<usize> {
            Vec::new()
        }

        fn deactivate_row(&mut self, _row: usize) -> onnx_genai_ort::Result<()> {
            Ok(())
        }

        fn assign_row(&mut self, _row: usize) -> onnx_genai_ort::Result<()> {
            Err(OrtError::InvalidArgument(
                "deliberate row assignment failure".to_string(),
            ))
        }

        fn step_select(
            &mut self,
            _next_token_ids: &[i64],
            _position_ids: &[i64],
            _advance_rows: &[bool],
        ) -> onnx_genai_ort::Result<Value> {
            unreachable!("a rejected row must never decode")
        }

        fn step_active(
            &mut self,
            _next_token_ids: &[i64],
            _position_ids: &[i64],
        ) -> onnx_genai_ort::Result<Value> {
            unreachable!("a rejected row must never decode")
        }
    }

    struct AcceptAssignDecode;

    impl<'a> BatchedDecodeSession<'a> for AcceptAssignDecode {
        fn batch_size(&self) -> usize {
            1
        }

        fn max_len(&self) -> usize {
            32
        }

        fn row_len(&self, _row: usize) -> onnx_genai_ort::Result<usize> {
            Ok(0)
        }

        fn active_rows(&self) -> Vec<usize> {
            Vec::new()
        }

        fn deactivate_row(&mut self, _row: usize) -> onnx_genai_ort::Result<()> {
            Ok(())
        }

        fn assign_row(&mut self, _row: usize) -> onnx_genai_ort::Result<()> {
            Ok(())
        }

        fn step_select(
            &mut self,
            _next_token_ids: &[i64],
            _position_ids: &[i64],
            _advance_rows: &[bool],
        ) -> onnx_genai_ort::Result<Value> {
            unreachable!("admission must not require backend generation")
        }

        fn step_active(
            &mut self,
            _next_token_ids: &[i64],
            _position_ids: &[i64],
        ) -> onnx_genai_ort::Result<Value> {
            unreachable!("admission must not require backend generation")
        }
    }

    fn test_tokenizer() -> Tokenizer {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json");
        Tokenizer::from_file(path).expect("load tiny tokenizer")
    }

    #[test]
    fn per_row_sampling_is_seedable_and_independent() {
        let options = GenerateOptions {
            greedy: false,
            ..Default::default()
        };
        let sequence = |row| {
            let mut rng = SamplingRng::for_row(Some(99), row);
            (0..32)
                .map(|_| sample_categorical(&[0.0, 0.0, 0.0], rng.value_for(&options)))
                .collect::<Vec<_>>()
        };

        assert_eq!(sequence(0), sequence(0));
        assert_eq!(sequence(1), sequence(1));
        assert_ne!(sequence(0), sequence(1));
    }

    #[test]
    fn row_assignment_failure_is_reported_for_the_submitted_handle() {
        let tokenizer = test_tokenizer();
        let mut manager =
            ContinuousBatchManager::new(Box::new(RejectAssignDecode), &tokenizer, None, 1).unwrap();
        let handle = manager
            .submit(GenerateRequest::new(GeneratePrompt::TokenIds(vec![1])))
            .unwrap();

        assert!(manager.poll_admissions().is_empty());
        manager.step().unwrap();
        let admissions = manager.poll_admissions();
        assert_eq!(admissions.len(), 1);
        match &admissions[0] {
            ContinuousBatchAdmission::Rejected {
                handle: rejected,
                error,
            } => {
                assert_eq!(*rejected, handle);
                assert!(
                    error
                        .to_string()
                        .contains("deliberate row assignment failure")
                );
            }
            ContinuousBatchAdmission::Assigned { .. } => panic!("failed row was acknowledged"),
        }
        assert!(!manager.has_pending_work());
    }

    #[test]
    fn successful_row_admission_is_observable_before_backend_generation() {
        let tokenizer = test_tokenizer();
        let mut manager =
            ContinuousBatchManager::new(Box::new(AcceptAssignDecode), &tokenizer, None, 1).unwrap();
        let handle = manager
            .submit(GenerateRequest::new(GeneratePrompt::TokenIds(vec![1])))
            .unwrap();

        manager.admit_pending();
        let admissions = manager.poll_admissions();
        assert!(matches!(
            admissions.as_slice(),
            [ContinuousBatchAdmission::Assigned { handle: assigned }] if *assigned == handle
        ));
    }

    #[test]
    fn pending_request_can_be_cancelled_without_an_admission_event() {
        let tokenizer = test_tokenizer();
        let mut manager =
            ContinuousBatchManager::new(Box::new(RejectAssignDecode), &tokenizer, None, 1).unwrap();
        let handle = manager
            .submit(GenerateRequest::new(GeneratePrompt::TokenIds(vec![1])))
            .unwrap();

        assert!(manager.cancel_pending(handle));
        assert!(!manager.has_pending_work());
        assert!(manager.poll_admissions().is_empty());
    }
}
