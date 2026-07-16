//! Batched static-cache generation path.

use crate::config::{
    FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult,
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
use onnx_genai_ort::{BatchedStaticCacheDecodeSession, StaticCacheDecodeOptions};
use onnx_genai_ort::{Session, Tokenizer};
use std::collections::VecDeque;

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

/// Synchronous continuous-batch manager for STATIC-CACHE models.
///
/// Requests are submitted into a FIFO queue and admitted into a fixed number of
/// physical decode rows. Each `step` samples one token for rows that have
/// pending logits, emits token/result events, evicts finished rows, admits queued
/// requests into freed slots, then prepares logits for the next step.
pub struct ContinuousBatchManager<'a> {
    decode: BatchedStaticCacheDecodeSession<'a>,
    tokenizer: &'a Tokenizer,
    metadata_max_context: Option<usize>,
    static_max_len: usize,
    queue: VecDeque<PendingContinuousRequest>,
    rows: Vec<Option<ContinuousBatchRow>>,
    events: VecDeque<ContinuousBatchEvent>,
    next_handle: usize,
}

impl<'a> ContinuousBatchManager<'a> {
    fn new(
        session: &'a Session,
        tokenizer: &'a Tokenizer,
        metadata_max_context: Option<usize>,
        max_batch: usize,
    ) -> anyhow::Result<Self> {
        if max_batch == 0 {
            anyhow::bail!("continuous batch max_batch must be greater than zero");
        }
        let mut decode = BatchedStaticCacheDecodeSession::new(
            session,
            StaticCacheDecodeOptions {
                batch_size: i64::try_from(max_batch).context("batch size exceeds i64")?,
            },
        )
        .map_err(|e| anyhow::anyhow!("Failed to create continuous static-cache session: {}", e))?;
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
                .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e))?,
        };
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        let max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(&options, Some(self.tokenizer))?;
        if reached_context_limit(prompt_tokens.len(), max_context) {
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
        self.admit_available_rows()?;
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
                    .map_err(|e| anyhow::anyhow!("Failed to deactivate continuous row: {}", e))?;
            } else {
                self.rows[row_index] = Some(row);
            }
        }

        self.admit_available_rows()?;
        self.decode_next_pending_rows()
    }

    /// Drain token/result events emitted by previous `submit` or `step` calls.
    pub fn poll(&mut self) -> Vec<ContinuousBatchEvent> {
        self.events.drain(..).collect()
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

    fn admit_available_rows(&mut self) -> anyhow::Result<()> {
        while !self.queue.is_empty() {
            let Some(row_index) = self.rows.iter().position(|row| row.is_none()) else {
                break;
            };
            let pending = self.queue.pop_front().expect("queue checked non-empty");
            self.decode
                .assign_row(row_index)
                .map_err(|e| anyhow::anyhow!("Failed to assign continuous row: {}", e))?;
            let rng = SamplingRng::for_row(pending.options.seed, row_index);
            let loop_state = DecodeLoopState::with_rng(0, rng, pending.options.top_logprobs);
            let mut row = ContinuousBatchRow {
                handle: pending.handle,
                physical_row: row_index,
                context_tokens: pending.prompt_tokens,
                options: pending.options,
                chain: pending.chain,
                max_context: pending.max_context,
                state: loop_state,
                pending_logits: None,
            };
            prefill_continuous_row(&mut self.decode, &mut row)?;
            self.rows[row_index] = Some(row);
        }
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
            row.context_tokens.clone(),
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
            for (active_index, &logical_row) in active_rows.iter().enumerate() {
                let row = self.rows[logical_row]
                    .as_ref()
                    .context("active continuous row is not assigned")?;
                let token = *row
                    .context_tokens
                    .last()
                    .context("continuous row has empty context")?;
                input_ids[active_index] = i64::from(token);
                position_ids[active_index] =
                    self.decode.row_len(logical_row).map_err(|e| {
                        anyhow::anyhow!("Failed to read continuous row length: {}", e)
                    })? as i64;
            }
            let logits = self
                .decode
                .step_active(&input_ids, &position_ids)
                .map_err(|e| {
                    anyhow::anyhow!("Continuous active static-cache step failed: {}", e)
                })?;
            for (active_index, logical_row) in active_rows.into_iter().enumerate() {
                let row = self.rows[logical_row]
                    .as_mut()
                    .context("active continuous row is not assigned")?;
                row.pending_logits = Some(row_logits(&logits, active_index, 0)?);
            }
            return Ok(());
        }

        let mut input_ids = vec![0_i64; self.max_batch()];
        let mut position_ids = vec![0_i64; self.max_batch()];
        let mut advance_rows = vec![false; self.max_batch()];
        for row in self.rows.iter().flatten() {
            if row.pending_logits.is_none() {
                let token = *row
                    .context_tokens
                    .last()
                    .context("continuous row has empty context")?;
                input_ids[row.physical_row] = i64::from(token);
                position_ids[row.physical_row] =
                    self.decode.row_len(row.physical_row).map_err(|e| {
                        anyhow::anyhow!("Failed to read continuous row length: {}", e)
                    })? as i64;
                advance_rows[row.physical_row] = true;
            }
        }
        let logits = self
            .decode
            .step_select(&input_ids, &position_ids, &advance_rows)
            .map_err(|e| anyhow::anyhow!("Continuous static-cache decode step failed: {}", e))?;
        for row in self.rows.iter_mut().flatten() {
            if advance_rows[row.physical_row] {
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
                    .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e))?,
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

        let mut decode = BatchedStaticCacheDecodeSession::new(
            self.session
                .as_deref()
                .context("ORT decoder session is unavailable")?,
            StaticCacheDecodeOptions {
                batch_size: i64::try_from(rows.len()).context("batch size exceeds i64")?,
            },
        )
        .map_err(|e| anyhow::anyhow!("Failed to create batched static-cache session: {}", e))?;

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
                    row.context_tokens.clone(),
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
                        .map_err(|e| anyhow::anyhow!("Failed to deactivate batch row: {}", e))?;
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

    /// Create a lower-level continuous-batch manager for incremental serving.
    pub fn continuous_batch_manager(
        &self,
        max_batch: usize,
    ) -> anyhow::Result<ContinuousBatchManager<'_>> {
        if !matches!(self.decode_path, ModelDecodePath::StaticCache { .. }) {
            anyhow::bail!(
                "continuous batching requires a STATIC-CACHE model; past/present batching is deferred"
            );
        }
        let metadata_max_context = self
            .metadata
            .model
            .as_ref()
            .and_then(|model| model.max_sequence_length);
        ContinuousBatchManager::new(
            self.session
                .as_deref()
                .context("ORT decoder session is unavailable")?,
            &self.tokenizer,
            metadata_max_context,
            max_batch,
        )
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

fn prefill_continuous_row(
    decode: &mut BatchedStaticCacheDecodeSession<'_>,
    row: &mut ContinuousBatchRow,
) -> anyhow::Result<()> {
    for offset in 0..row.context_tokens.len() {
        let mut input_ids = vec![0_i64; decode.batch_size()];
        let mut position_ids = vec![0_i64; decode.batch_size()];
        let mut advance_rows = vec![false; decode.batch_size()];
        input_ids[row.physical_row] = i64::from(row.context_tokens[offset]);
        position_ids[row.physical_row] = decode
            .row_len(row.physical_row)
            .map_err(|e| anyhow::anyhow!("Failed to read continuous row length: {}", e))?
            as i64;
        advance_rows[row.physical_row] = true;
        let logits = decode
            .step_select(&input_ids, &position_ids, &advance_rows)
            .map_err(|e| anyhow::anyhow!("Continuous static-cache prefill failed: {}", e))?;
        row.pending_logits = Some(row_logits(&logits, row.physical_row, 0)?);
    }
    Ok(())
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
            .map_err(|e| anyhow::anyhow!("Batched static-cache prefill failed: {}", e))?;
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
                    .map_err(|e| anyhow::anyhow!("Failed to read batch row length: {}", e))?
                    as i64;
                advance_rows[row.physical_row] = true;
            }
        }
        let logits = decode
            .step_select(&input_ids, &position_ids, &advance_rows)
            .map_err(|e| anyhow::anyhow!("Batched static-cache ragged prefill failed: {}", e))?;
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
            .map_err(|e| anyhow::anyhow!("Failed to read batch row length: {}", e))?
            as i64;
        advance_rows[row.physical_row] = true;
    }
    let logits = decode
        .step_select(&input_ids, &position_ids, &advance_rows)
        .map_err(|e| anyhow::anyhow!("Batched static-cache decode step failed: {}", e))?;
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
        .map_err(|e| anyhow::anyhow!("Failed to extract row logits: {}", e))
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
}
