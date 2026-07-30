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
use onnx_genai_ort::Tokenizer;
use onnx_genai_ort::decode::{
    BatchedDecodeSession, BatchedSharedBufferDecodeSession, SharedBufferBatchOptions,
};
use onnx_genai_ort::{BatchedStaticCacheDecodeSession, StaticCacheDecodeOptions};
use std::collections::VecDeque;

/// The `lora.segments` routing id for a base-only row — the kernel's null route
/// (design §J.3): a row with this id gets no adapter delta. A row bound to an
/// adapter carries that adapter's [`AdapterId`](onnx_runtime_ep_api::AdapterId)
/// value instead.
const BASE_LORA_ROUTE: i32 = -1;

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
    /// The grouped-LoRA route this request was admitted with, resolved once at
    /// submit (design §J.4): the selected adapter's id, or [`BASE_LORA_ROUTE`]
    /// for a base-only request. Never re-resolved per step, so a row cannot
    /// inherit another request's adapter.
    lora_route: i32,
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
    /// The grouped-LoRA route this running sequence is bound to for every decode
    /// step (design §J.4 P2e): the admitted adapter's id, or [`BASE_LORA_ROUTE`]
    /// for base-only. Written into `lora.segments[physical_row]` each step so one
    /// continuous batch can carry several adapters at once.
    lora_route: i32,
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
    decode: Box<dyn BatchedDecodeSession<'a> + 'a>,
    tokenizer: &'a Tokenizer,
    metadata_max_context: Option<usize>,
    static_max_len: usize,
    queue: VecDeque<PendingContinuousRequest>,
    rows: Vec<Option<ContinuousBatchRow>>,
    events: VecDeque<ContinuousBatchEvent>,
    next_handle: usize,
    /// Name → `lora.segments` route id for every adapter this session admits
    /// (design §J.4 P2e). Empty for a base-only / non-grouped session, which
    /// keeps the routing machinery dormant: no per-step `segments` buffer is
    /// built or fed, so the base path is byte-for-byte unchanged.
    lora_adapter_routes: Vec<(String, i32)>,
    /// Reused per-step routing buffer (design perf gate). Refilled in place each
    /// decode step so the `lora.segments` payload is not reallocated per token.
    lora_route_scratch: Vec<i32>,
}

impl<'a> ContinuousBatchManager<'a> {
    fn new(
        decode: Box<dyn BatchedDecodeSession<'a> + 'a>,
        tokenizer: &'a Tokenizer,
        metadata_max_context: Option<usize>,
        max_batch: usize,
    ) -> anyhow::Result<Self> {
        Self::with_lora_adapter_routes(
            decode,
            tokenizer,
            metadata_max_context,
            max_batch,
            Vec::new(),
        )
    }

    pub(crate) fn with_lora_adapter_routes(
        mut decode: Box<dyn BatchedDecodeSession<'a> + 'a>,
        tokenizer: &'a Tokenizer,
        metadata_max_context: Option<usize>,
        max_batch: usize,
        lora_adapter_routes: Vec<(String, i32)>,
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
            events: VecDeque::new(),
            next_handle: 0,
            lora_adapter_routes,
            lora_route_scratch: Vec::new(),
        })
    }

    /// Whether this session routes per-row grouped-LoRA adapters (design §J.4
    /// P2e). False for a base-only / non-grouped session, in which case the
    /// per-step `lora.segments` feed is skipped entirely (fast-path preservation).
    fn lora_routing_enabled(&self) -> bool {
        !self.lora_adapter_routes.is_empty()
    }

    /// Resolve a request's selected adapter name to its `lora.segments` route id
    /// (design §J.4). `None` ⇒ base-only ([`BASE_LORA_ROUTE`]). An unknown name
    /// fails loud at admission — never a silent base fallback — and a session
    /// with no grouped pool rejects any explicit adapter selection.
    fn resolve_lora_route(&self, adapter: Option<&str>) -> anyhow::Result<i32> {
        match adapter {
            None => Ok(BASE_LORA_ROUTE),
            Some(name) if self.lora_adapter_routes.is_empty() => anyhow::bail!(
                "per-request LoRA adapter {name:?} was requested, but this continuous-batch \
                 session was not loaded with a grouped-LoRA pool (configure \
                 EngineConfig::lora_adapters)"
            ),
            Some(name) => self
                .lora_adapter_routes
                .iter()
                .find(|(adapter_name, _)| adapter_name == name)
                .map(|(_, route)| *route)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "unknown LoRA adapter {name:?}; loaded adapters: [{}]",
                        self.lora_adapter_routes
                            .iter()
                            .map(|(adapter_name, _)| adapter_name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }),
        }
    }

    /// Feed the physical-row-indexed `lora.segments` for the next `step_select` /
    /// `prefill` call: `segments[physical_row]` is that row's sequence adapter
    /// route, empty slots route to base ([`BASE_LORA_ROUTE`]). A no-op (no
    /// allocation, no session call) unless this session routes adapters.
    fn feed_physical_lora_routes(&mut self) -> anyhow::Result<()> {
        if !self.lora_routing_enabled() {
            return Ok(());
        }
        self.lora_route_scratch.clear();
        self.lora_route_scratch.resize(self.rows.len(), BASE_LORA_ROUTE);
        for row in self.rows.iter().flatten() {
            self.lora_route_scratch[row.physical_row] = row.lora_route;
        }
        self.decode
            .set_lora_routes(&self.lora_route_scratch)
            .map_err(|e| anyhow::anyhow!("Failed to feed continuous grouped-LoRA segments: {e}"))
    }

    /// Feed the active-row-ordered `lora.segments` for the next `step_active`
    /// call: `segments[i]` is the adapter route of the i-th active row in
    /// `active_rows` order, matching how `step_active` orders its inputs/logits.
    /// A no-op unless this session routes adapters.
    fn feed_active_lora_routes(&mut self, active_rows: &[usize]) -> anyhow::Result<()> {
        if !self.lora_routing_enabled() {
            return Ok(());
        }
        self.lora_route_scratch.clear();
        self.lora_route_scratch.reserve(active_rows.len());
        for &logical_row in active_rows {
            let route = self.rows[logical_row]
                .as_ref()
                .context("active continuous row is not assigned")?
                .lora_route;
            self.lora_route_scratch.push(route);
        }
        self.decode
            .set_lora_routes(&self.lora_route_scratch)
            .map_err(|e| anyhow::anyhow!("Failed to feed continuous grouped-LoRA segments: {e}"))
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
        let lora_route = self
            .resolve_lora_route(options.adapter.as_deref())
            .context("resolve per-request LoRA adapter for continuous batch")?;
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
            lora_route,
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
                    .map_err(|e| anyhow::anyhow!("Failed to deactivate continuous row: {e}"))?;
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
                lora_route: pending.lora_route,
            };
            self.rows[row_index] = Some(row);
            self.prefill_continuous_row(row_index)?;
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
            for (active_index, &logical_row) in active_rows.iter().enumerate() {
                let row = self.rows[logical_row]
                    .as_ref()
                    .context("active continuous row is not assigned")?;
                let token = *row
                    .context_tokens
                    .last()
                    .context("continuous row has empty context")?;
                input_ids[active_index] = i64::from(token);
                position_ids[active_index] = self
                    .decode
                    .row_len(logical_row)
                    .map_err(|e| anyhow::anyhow!("Failed to read continuous row length: {e}"))?
                    as i64;
            }
            self.feed_active_lora_routes(&active_rows)?;
            let logits = self
                .decode
                .step_active(&input_ids, &position_ids)
                .map_err(|e| anyhow::anyhow!("Continuous active static-cache step failed: {e}"))?;
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
                position_ids[row.physical_row] = self
                    .decode
                    .row_len(row.physical_row)
                    .map_err(|e| anyhow::anyhow!("Failed to read continuous row length: {e}"))?
                    as i64;
                advance_rows[row.physical_row] = true;
            }
        }
        self.feed_physical_lora_routes()?;
        let logits = self
            .decode
            .step_select(&input_ids, &position_ids, &advance_rows)
            .map_err(|e| anyhow::anyhow!("Continuous static-cache decode step failed: {e}"))?;
        for row in self.rows.iter_mut().flatten() {
            if advance_rows[row.physical_row] {
                row.pending_logits = Some(row_logits(&logits, row.physical_row, 0)?);
            }
        }
        Ok(())
    }

    /// Prefill a freshly admitted row one prompt token at a time. The row is
    /// already stored at `self.rows[row_index]`, so the per-step `lora.segments`
    /// feed can route every physical slot to its own sequence's adapter (design
    /// §J.4 P2e) — the prefilling row to its admitted adapter, other live rows to
    /// theirs, empty slots to base.
    fn prefill_continuous_row(&mut self, row_index: usize) -> anyhow::Result<()> {
        let context_len = self.rows[row_index]
            .as_ref()
            .context("continuous row disappeared before prefill")?
            .context_tokens
            .len();
        for offset in 0..context_len {
            let batch_size = self.decode.batch_size();
            let mut input_ids = vec![0_i64; batch_size];
            let mut position_ids = vec![0_i64; batch_size];
            let mut advance_rows = vec![false; batch_size];
            let physical_row = {
                let row = self.rows[row_index]
                    .as_ref()
                    .context("continuous row disappeared during prefill")?;
                input_ids[row.physical_row] = i64::from(row.context_tokens[offset]);
                advance_rows[row.physical_row] = true;
                row.physical_row
            };
            position_ids[physical_row] = self
                .decode
                .row_len(physical_row)
                .map_err(|e| anyhow::anyhow!("Failed to read continuous row length: {e}"))?
                as i64;
            self.feed_physical_lora_routes()?;
            let logits = self
                .decode
                .step_select(&input_ids, &position_ids, &advance_rows)
                .map_err(|e| anyhow::anyhow!("Continuous static-cache prefill failed: {e}"))?;
            let extracted = row_logits(&logits, physical_row, 0)?;
            self.rows[row_index]
                .as_mut()
                .context("continuous row disappeared after prefill step")?
                .pending_logits = Some(extracted);
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

        let mut decode = BatchedStaticCacheDecodeSession::new(
            self.session
                .as_deref()
                .context("ORT decoder session is unavailable")?,
            StaticCacheDecodeOptions {
                batch_size: i64::try_from(rows.len()).context("batch size exceeds i64")?,
            },
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

    /// Create a lower-level continuous-batch manager for incremental serving.
    pub fn continuous_batch_manager(
        &self,
        max_batch: usize,
    ) -> anyhow::Result<ContinuousBatchManager<'_>> {
        if max_batch == 0 {
            anyhow::bail!("continuous batch max_batch must be greater than zero");
        }
        let session = self
            .session
            .as_deref()
            .context("ORT decoder session is unavailable")?;
        let batch_size = i64::try_from(max_batch).context("batch size exceeds i64")?;
        let decode: Box<dyn BatchedDecodeSession<'_> + '_> = match self.decode_path {
            ModelDecodePath::StaticCache { .. } => Box::new(
                BatchedStaticCacheDecodeSession::new(
                    session,
                    StaticCacheDecodeOptions { batch_size },
                )
                .map_err(|e| {
                    anyhow::anyhow!("Failed to create continuous static-cache session: {e}")
                })?,
            ),
            ModelDecodePath::PastPresent {
                shared_buffer: true,
                max_len,
                ..
            } => {
                let max_len = max_len
                    .context("shared-buffer continuous batching requires a known max_len")?;
                Box::new(
                    BatchedSharedBufferDecodeSession::new(
                        session,
                        SharedBufferBatchOptions {
                            batch_size,
                            max_len,
                        },
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to create continuous shared-buffer session: {e}")
                    })?,
                )
            }
            ModelDecodePath::PastPresent { .. } | ModelDecodePath::Legacy => {
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

    /// Run requests to completion through a **native** continuous batch that can
    /// carry mixed LoRA adapters per row (design §J.4/§J.5 P2e).
    ///
    /// Unlike [`Self::run_continuous_batch`] (ORT `BatchedStaticCacheDecodeSession`,
    /// no grouped op), this drives a [`NativeBatchedDecodeSession`] over the
    /// engine's native grouped `InferenceSession`, so each row's logits carry that
    /// row's own adapter delta. Per-request adapter names are resolved to their
    /// `lora.segments` ids from the loaded grouped pool; an unknown name fails
    /// loud at admission.
    #[cfg(feature = "native-backend")]
    pub fn run_native_continuous_batch(
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
        let metadata_max_context = self
            .metadata
            .model
            .as_ref()
            .and_then(|model| model.max_sequence_length);
        let max_len = metadata_max_context.unwrap_or(4096).max(1);
        // Borrow the native session and tokenizer as disjoint fields so the
        // manager can hold the mutable session borrow alongside the shared
        // tokenizer borrow for its whole lifetime.
        let native_session = self.native_session.as_mut().context(
            "native continuous batching requires a native decoder session; load the engine with \
             ONNX_GENAI_BACKEND=native",
        )?;
        let lora_adapter_routes = native_session.lora_adapter_routes();
        let inference_session = native_session.inference_session_mut();
        let decode = crate::native_decode::NativeBatchedDecodeSession::new(
            inference_session,
            max_batch,
            max_len,
        )
        .context("build native batched decode session")?;
        let mut manager = ContinuousBatchManager::with_lora_adapter_routes(
            Box::new(decode),
            &self.tokenizer,
            metadata_max_context,
            max_batch,
            lora_adapter_routes,
        )?;

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

/// Acceptance coverage for design §J.4 P2e — mixed-adapter rows within ONE
/// continuous batch. Drives the real [`ContinuousBatchManager`] (submit → step →
/// poll) over a grouped-LoRA session and asserts that one batch carrying three
/// sequences bound to adapter A, adapter B, and base produces per-row-correct
/// outputs. The `GroupedProbeSession` is a decode-session test double for the KV
/// runner only: it runs the **real** `GroupedLoraDelta` kernel (via
/// [`InferenceSession`]) on the per-row `lora.segments` the manager builds, so a
/// whole-batch-to-base manager fails this test (row A and row B would collapse to
/// base). The kernel's own per-row math is proven separately by
/// `onnx-runtime-session`'s `grouped_two_adapters_route_per_row`.
#[cfg(all(test, feature = "native-backend"))]
mod lora_continuous_batch_tests {
    use super::*;
    use onnx_genai_ort::{OrtError, Value};
    use onnx_runtime_ir::{
        Attribute, DataType as IrDataType, Dim, Graph, Node, NodeId, TensorData, WeightRef,
        static_shape,
    };
    use onnx_runtime_loader::encoder::{Model, write_model};
    use onnx_runtime_session::lora_inject::{LoraAdapterSpec, LoraModuleSpec};
    use onnx_runtime_session::{InferenceSession, Tensor};
    use crate::native_decode::NativeBatchedDecodeSession;

    const K: usize = 16;
    const N: usize = 3;
    const RANK: usize = 2;

    fn f32_le(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn i32_le(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// `scale · ((x · A_t) · B_t)` from the injection factors (fp32 accumulators),
    /// the authoritative per-adapter delta a routed row must reproduce.
    fn reference_delta(x: &[f32], a_t: &[f32], b_t: &[f32], scale: f32) -> Vec<f32> {
        let mut mid = vec![0.0f64; RANK];
        for j in 0..RANK {
            let mut acc = 0.0f64;
            for p in 0..K {
                acc += x[p] as f64 * a_t[p * RANK + j] as f64;
            }
            mid[j] = acc;
        }
        let mut delta = vec![0.0f32; N];
        for j in 0..N {
            let mut acc = 0.0f64;
            for p in 0..RANK {
                acc += mid[p] * b_t[p * N + j] as f64;
            }
            delta[j] = (acc * scale as f64) as f32;
        }
        delta
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    }

    /// A single-layer `q_proj` int4 base projection that dequantizes to zero
    /// (every nibble is the affine zero point `0x8`), so the observable output
    /// isolates the grouped-LoRA delta. The activation `x[batch, K]` uses a
    /// symbolic batch dim so the same session runs any batch width.
    fn write_zero_base_batch_model(path: &std::path::Path) {
        const BITS: usize = 4;
        const BLOCK_SIZE: usize = 16;
        let k_blocks = K / BLOCK_SIZE;
        let blob_size = BLOCK_SIZE * BITS / 8;

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert("com.microsoft".to_string(), 1);

        let batch = graph.intern_symbol("batch");
        let x = graph.create_named_value(
            "x",
            IrDataType::Float32,
            vec![Dim::from(batch), Dim::from(K)],
        );
        graph.add_input(x);

        let weight = graph.create_named_value(
            "model.layers.0.attn.q_proj.MatMulNBits.qweight",
            IrDataType::Uint8,
            static_shape([N, k_blocks, blob_size]),
        );
        graph.set_initializer(
            weight,
            WeightRef::Inline(TensorData::from_raw(
                IrDataType::Uint8,
                vec![N, k_blocks, blob_size],
                vec![0x88u8; N * k_blocks * blob_size],
            )),
        );
        let scales = graph.create_named_value(
            "model.layers.0.attn.q_proj.MatMulNBits.scales",
            IrDataType::Float32,
            static_shape([N, k_blocks]),
        );
        graph.set_initializer(
            scales,
            WeightRef::Inline(TensorData::from_raw(
                IrDataType::Float32,
                vec![N, k_blocks],
                f32_le(&vec![1.0f32; N * k_blocks]),
            )),
        );

        let y = graph.create_named_value(
            "y",
            IrDataType::Float32,
            vec![Dim::from(batch), Dim::from(N)],
        );
        let mut node = Node::new(
            NodeId(0),
            "MatMulNBits",
            vec![Some(x), Some(weight), Some(scales)],
            vec![y],
        );
        node.name = "/model/layers.0/attn/q_proj/MatMulNBits".to_string();
        node.domain = "com.microsoft".to_string();
        node.attributes.insert("K".to_string(), Attribute::Int(K as i64));
        node.attributes.insert("N".to_string(), Attribute::Int(N as i64));
        node.attributes.insert("bits".to_string(), Attribute::Int(BITS as i64));
        node.attributes
            .insert("block_size".to_string(), Attribute::Int(BLOCK_SIZE as i64));
        graph.insert_node(node);
        graph.add_output(y);

        write_model(&Model::new(&graph), path).unwrap();
    }

    fn lora_spec(name: &str, a_t: &[f32], b_t: &[f32], scale: f32) -> LoraAdapterSpec {
        LoraAdapterSpec {
            name: name.to_string(),
            modules: vec![LoraModuleSpec {
                module_name: "self_attn.q_proj".to_string(),
                layer_index: 0,
                rank: RANK,
                scale,
                a_t: TensorData::from_raw(IrDataType::Float32, vec![K, RANK], f32_le(a_t)),
                b_t: TensorData::from_raw(IrDataType::Float32, vec![RANK, N], f32_le(b_t)),
            }],
        }
    }

    /// A `BatchedDecodeSession` that runs the real grouped-LoRA `InferenceSession`
    /// on a fixed activation, so each physical row's logits depend ONLY on the
    /// per-row `lora.segments` route the manager feeds. KV-free (the base model is
    /// a pure projection), so `row_len`/positions are bookkeeping only.
    struct GroupedProbeSession {
        session: InferenceSession,
        segments_input: String,
        fixed_x: Vec<f32>,
        batch_size: usize,
        row_lens: Vec<usize>,
        active: Vec<bool>,
        routes: Vec<i32>,
    }

    impl GroupedProbeSession {
        fn run_rows(&mut self, rows: usize) -> onnx_genai_ort::Result<Value> {
            if self.routes.len() != rows {
                return Err(OrtError::InvalidArgument(format!(
                    "probe fed {} routes for {rows} activation rows",
                    self.routes.len()
                )));
            }
            let mut x = Vec::with_capacity(rows * K);
            for _ in 0..rows {
                x.extend_from_slice(&self.fixed_x);
            }
            let x_tensor = Tensor::from_f32(&[rows, K], &x)
                .map_err(|e| OrtError::InvalidArgument(format!("probe x tensor: {e}")))?;
            let seg_tensor =
                Tensor::from_raw(IrDataType::Int32, vec![rows], &i32_le(&self.routes))
                    .map_err(|e| OrtError::InvalidArgument(format!("probe segments tensor: {e}")))?;
            let outputs = self
                .session
                .run(&[("x", &x_tensor), (self.segments_input.as_str(), &seg_tensor)])
                .map_err(|e| OrtError::InvalidArgument(format!("probe grouped run: {e}")))?;
            let y = outputs[0].to_vec_f32();
            Value::from_slice_f32(&y, &[rows as i64, 1, N as i64])
        }
    }

    impl<'a> BatchedDecodeSession<'a> for GroupedProbeSession {
        fn batch_size(&self) -> usize {
            self.batch_size
        }
        fn max_len(&self) -> usize {
            1 << 20
        }
        fn row_len(&self, row: usize) -> onnx_genai_ort::Result<usize> {
            Ok(self.row_lens[row])
        }
        fn active_rows(&self) -> Vec<usize> {
            (0..self.batch_size).filter(|&row| self.active[row]).collect()
        }
        fn deactivate_row(&mut self, row: usize) -> onnx_genai_ort::Result<()> {
            self.active[row] = false;
            Ok(())
        }
        fn assign_row(&mut self, row: usize) -> onnx_genai_ort::Result<()> {
            self.active[row] = true;
            self.row_lens[row] = 0;
            Ok(())
        }
        fn set_lora_routes(&mut self, routes: &[i32]) -> onnx_genai_ort::Result<()> {
            self.routes.clear();
            self.routes.extend_from_slice(routes);
            Ok(())
        }
        fn step_select(
            &mut self,
            next_token_ids: &[i64],
            _position_ids: &[i64],
            advance_rows: &[bool],
        ) -> onnx_genai_ort::Result<Value> {
            let logits = self.run_rows(next_token_ids.len())?;
            for row in 0..self.batch_size {
                if self.active[row] && advance_rows[row] {
                    self.row_lens[row] += 1;
                }
            }
            Ok(logits)
        }
        fn step_active(
            &mut self,
            next_token_ids: &[i64],
            _position_ids: &[i64],
        ) -> onnx_genai_ort::Result<Value> {
            let logits = self.run_rows(next_token_ids.len())?;
            for row in 0..self.batch_size {
                if self.active[row] {
                    self.row_lens[row] += 1;
                }
            }
            Ok(logits)
        }
    }

    fn tokenizer() -> Tokenizer {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm-scatter/tokenizer.json");
        Tokenizer::from_file(path).expect("load fixture tokenizer")
    }

    fn base_request(adapter: Option<&str>) -> GenerateRequest {
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![1]));
        request.options.max_new_tokens = 1;
        request.options.greedy = true;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.eos_token_id = Some(1_000_000);
        request.options.adapter = adapter.map(str::to_string);
        request
    }

    #[test]
    fn continuous_batch_routes_mixed_adapters_per_row() {
        // Adapter A projects onto vocab index 1; adapter B onto index 2; base is
        // the exact zero vector (argmax 0). Distinct nonzero argmaxes make the
        // routing observable in the emitted token and non-tautological: a
        // whole-batch-to-base manager emits token 0 for every row and fails.
        let fixed_x: Vec<f32> = vec![1.0; K];
        let scale = 1.0f32;
        // A_t is [K, RANK]; column 0 sums the activation into hidden unit 0.
        let mut a_t = vec![0.0f32; K * RANK];
        for p in 0..K {
            a_t[p * RANK] = 0.1; // -> mid[0] = 0.1 * sum(x) = 0.1 * K
        }
        // B_t is [RANK, N]; hidden unit 0 drives one distinct vocab column each.
        let mut b_t_a = vec![0.0f32; RANK * N];
        b_t_a[1] = 1.0; // row 0 (hidden 0) -> vocab 1
        let mut b_t_b = vec![0.0f32; RANK * N];
        b_t_b[2] = 1.0; // row 0 (hidden 0) -> vocab 2

        let delta_a = reference_delta(&fixed_x, &a_t, &b_t_a, scale);
        let delta_b = reference_delta(&fixed_x, &a_t, &b_t_b, scale);
        let token_a = argmax(&delta_a);
        let token_b = argmax(&delta_b);
        assert_ne!(token_a, 0, "adapter A must route off the base argmax");
        assert_ne!(token_b, 0, "adapter B must route off the base argmax");
        assert_ne!(token_a, token_b, "adapters A and B must be distinguishable");

        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.onnx");
        write_zero_base_batch_model(&model_path);

        let session = InferenceSession::builder()
            .model(&model_path)
            .lora_adapters(vec![
                lora_spec("adapter_a", &a_t, &b_t_a, scale),
                lora_spec("adapter_b", &a_t, &b_t_b, scale),
            ])
            .build()
            .expect("build grouped multi-adapter session");

        let segments_input = session
            .lora_segments_input()
            .expect("grouped session exposes a segments input")
            .to_string();
        let route_a = session.resolve_lora_adapter("adapter_a").expect("resolve A").0 as i32;
        let route_b = session.resolve_lora_adapter("adapter_b").expect("resolve B").0 as i32;

        let max_batch = 3;
        let probe = GroupedProbeSession {
            session,
            segments_input,
            fixed_x,
            batch_size: max_batch,
            row_lens: vec![0; max_batch],
            active: vec![false; max_batch],
            routes: Vec::new(),
        };

        let tokenizer = tokenizer();
        let mut manager = ContinuousBatchManager::with_lora_adapter_routes(
            Box::new(probe),
            &tokenizer,
            None,
            max_batch,
            vec![
                ("adapter_a".to_string(), route_a),
                ("adapter_b".to_string(), route_b),
            ],
        )
        .expect("build continuous-batch manager");

        // Handle order: 0 -> adapter A, 1 -> adapter B, 2 -> base.
        let handle_a = manager.submit(base_request(Some("adapter_a"))).unwrap();
        let handle_b = manager.submit(base_request(Some("adapter_b"))).unwrap();
        let handle_base = manager.submit(base_request(None)).unwrap();

        let mut first_token = std::collections::HashMap::new();
        // Drain the initial admission events, then step to completion.
        let record = |events: Vec<ContinuousBatchEvent>,
                      sink: &mut std::collections::HashMap<usize, u32>| {
            for event in events {
                if let ContinuousBatchEvent::Token { handle, token } = event {
                    sink.entry(handle.id).or_insert(token.token_id);
                }
            }
        };
        record(manager.poll(), &mut first_token);
        let mut guard = 0;
        while manager.has_pending_work() {
            manager.step().unwrap();
            record(manager.poll(), &mut first_token);
            guard += 1;
            assert!(guard < 64, "continuous batch failed to drain");
        }

        assert_eq!(
            first_token.get(&handle_a.id).copied(),
            Some(token_a as u32),
            "row bound to adapter A must emit A's argmax token"
        );
        assert_eq!(
            first_token.get(&handle_b.id).copied(),
            Some(token_b as u32),
            "row bound to adapter B must emit B's argmax token"
        );
        assert_eq!(
            first_token.get(&handle_base.id).copied(),
            Some(0u32),
            "base row must emit the zero-base argmax token"
        );
    }

    /// An unknown adapter name fails loud at admission (never a silent base
    /// fallback), and a base-only request is admitted with the null route.
    #[test]
    fn continuous_batch_unknown_adapter_fails_loud() {
        let fixed_x: Vec<f32> = vec![1.0; K];
        let scale = 1.0f32;
        let mut a_t = vec![0.0f32; K * RANK];
        for p in 0..K {
            a_t[p * RANK] = 0.1;
        }
        let mut b_t_a = vec![0.0f32; RANK * N];
        b_t_a[1] = 1.0;

        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.onnx");
        write_zero_base_batch_model(&model_path);
        let session = InferenceSession::builder()
            .model(&model_path)
            .lora_adapters(vec![lora_spec("adapter_a", &a_t, &b_t_a, scale)])
            .build()
            .expect("build grouped session");
        let segments_input = session.lora_segments_input().unwrap().to_string();
        let route_a = session.resolve_lora_adapter("adapter_a").unwrap().0 as i32;

        let max_batch = 2;
        let probe = GroupedProbeSession {
            session,
            segments_input,
            fixed_x,
            batch_size: max_batch,
            row_lens: vec![0; max_batch],
            active: vec![false; max_batch],
            routes: Vec::new(),
        };
        let tokenizer = tokenizer();
        let mut manager = ContinuousBatchManager::with_lora_adapter_routes(
            Box::new(probe),
            &tokenizer,
            None,
            max_batch,
            vec![("adapter_a".to_string(), route_a)],
        )
        .expect("build manager");

        let error = manager
            .submit(base_request(Some("nope")))
            .expect_err("unknown adapter must fail loud at admission");
        let message = format!("{error:#}");
        assert!(
            message.contains("unknown LoRA adapter"),
            "unexpected error: {message}"
        );
        // Base-only still admits fine.
        manager.submit(base_request(None)).expect("base-only admits");
    }

    fn i64_le(values: &[i64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A token-driven sibling of [`write_zero_base_batch_model`]: the same
    /// zero-dequantizing `q_proj` int4 projection, but fed by
    /// `input_ids -> Gather(embedding) -> Reshape[-1, K] -> x` so the graph is
    /// driven by real token ids (a decoder-shaped, KV-free step) instead of a
    /// pre-baked activation. Token id 1 embeds to `ones(K)` so `x == fixed_x`,
    /// isolating the grouped-LoRA delta exactly like the probe model. The single
    /// output is named `logits` so the native batched session resolves it.
    fn write_zero_base_token_model(path: &std::path::Path) {
        const BITS: usize = 4;
        const BLOCK_SIZE: usize = 16;
        const VOCAB_EMBED: usize = 8;
        let k_blocks = K / BLOCK_SIZE;
        let blob_size = BLOCK_SIZE * BITS / 8;

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert("com.microsoft".to_string(), 1);

        let batch = graph.intern_symbol("batch");
        let input_ids = graph.create_named_value(
            "input_ids",
            IrDataType::Int64,
            vec![Dim::from(batch), Dim::from(1usize)],
        );
        graph.add_input(input_ids);

        // Embedding table: row for token id 1 is ones(K); every other row is
        // zero, so a driven token 1 reproduces the probe's fixed activation.
        let mut embedding_data = vec![0.0f32; VOCAB_EMBED * K];
        for column in 0..K {
            embedding_data[K + column] = 1.0;
        }
        let embedding = graph.create_named_value(
            "model.embed_tokens.weight",
            IrDataType::Float32,
            static_shape([VOCAB_EMBED, K]),
        );
        graph.set_initializer(
            embedding,
            WeightRef::Inline(TensorData::from_raw(
                IrDataType::Float32,
                vec![VOCAB_EMBED, K],
                f32_le(&embedding_data),
            )),
        );
        let gathered = graph.create_named_value(
            "gathered",
            IrDataType::Float32,
            vec![Dim::from(batch), Dim::from(1usize), Dim::from(K)],
        );
        let mut gather = Node::new(
            NodeId(0),
            "Gather",
            vec![Some(embedding), Some(input_ids)],
            vec![gathered],
        );
        gather.name = "/model/embed_tokens/Gather".to_string();
        gather.attributes.insert("axis".to_string(), Attribute::Int(0));
        graph.insert_node(gather);

        // Reshape [batch, 1, K] -> [batch, K] so the base MatMulNBits (and the
        // injected GroupedLoraDelta that shadows it) sees a rank-2 activation.
        let reshape_shape = graph.create_named_value(
            "reshape_shape",
            IrDataType::Int64,
            static_shape([2]),
        );
        graph.set_initializer(
            reshape_shape,
            WeightRef::Inline(TensorData::from_raw(
                IrDataType::Int64,
                vec![2],
                i64_le(&[-1, K as i64]),
            )),
        );
        let x = graph.create_named_value(
            "x",
            IrDataType::Float32,
            vec![Dim::from(batch), Dim::from(K)],
        );
        let mut reshape = Node::new(
            NodeId(1),
            "Reshape",
            vec![Some(gathered), Some(reshape_shape)],
            vec![x],
        );
        reshape.name = "/model/embed_tokens/Reshape".to_string();
        graph.insert_node(reshape);

        let weight = graph.create_named_value(
            "model.layers.0.attn.q_proj.MatMulNBits.qweight",
            IrDataType::Uint8,
            static_shape([N, k_blocks, blob_size]),
        );
        graph.set_initializer(
            weight,
            WeightRef::Inline(TensorData::from_raw(
                IrDataType::Uint8,
                vec![N, k_blocks, blob_size],
                vec![0x88u8; N * k_blocks * blob_size],
            )),
        );
        let scales = graph.create_named_value(
            "model.layers.0.attn.q_proj.MatMulNBits.scales",
            IrDataType::Float32,
            static_shape([N, k_blocks]),
        );
        graph.set_initializer(
            scales,
            WeightRef::Inline(TensorData::from_raw(
                IrDataType::Float32,
                vec![N, k_blocks],
                f32_le(&vec![1.0f32; N * k_blocks]),
            )),
        );
        let logits = graph.create_named_value(
            "logits",
            IrDataType::Float32,
            vec![Dim::from(batch), Dim::from(N)],
        );
        let mut node = Node::new(
            NodeId(2),
            "MatMulNBits",
            vec![Some(x), Some(weight), Some(scales)],
            vec![logits],
        );
        node.name = "/model/layers.0/attn/q_proj/MatMulNBits".to_string();
        node.domain = "com.microsoft".to_string();
        node.attributes.insert("K".to_string(), Attribute::Int(K as i64));
        node.attributes.insert("N".to_string(), Attribute::Int(N as i64));
        node.attributes.insert("bits".to_string(), Attribute::Int(BITS as i64));
        node.attributes
            .insert("block_size".to_string(), Attribute::Int(BLOCK_SIZE as i64));
        graph.insert_node(node);
        graph.add_output(logits);

        write_model(&Model::new(&graph), path).unwrap();
    }

    /// THE gap-closing acceptance test (design §J.9): drive the REAL
    /// [`NativeBatchedDecodeSession`] — not a decode-session double — through the
    /// production [`ContinuousBatchManager`], carrying three sequences bound to
    /// adapter A, adapter B, and base within ONE native continuous batch, and
    /// assert each row emits its own adapter's argmax token. Non-tautological: a
    /// whole-batch-to-base session collapses rows A and B to token 0 and fails.
    #[test]
    fn native_continuous_batch_routes_mixed_adapters_per_row() {
        let fixed_x: Vec<f32> = vec![1.0; K];
        let scale = 1.0f32;
        let mut a_t = vec![0.0f32; K * RANK];
        for p in 0..K {
            a_t[p * RANK] = 0.1;
        }
        let mut b_t_a = vec![0.0f32; RANK * N];
        b_t_a[1] = 1.0; // adapter A -> vocab 1
        let mut b_t_b = vec![0.0f32; RANK * N];
        b_t_b[2] = 1.0; // adapter B -> vocab 2

        let token_a = argmax(&reference_delta(&fixed_x, &a_t, &b_t_a, scale));
        let token_b = argmax(&reference_delta(&fixed_x, &a_t, &b_t_b, scale));
        assert_ne!(token_a, 0, "adapter A must route off the base argmax");
        assert_ne!(token_b, 0, "adapter B must route off the base argmax");
        assert_ne!(token_a, token_b, "adapters A and B must be distinguishable");

        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.onnx");
        write_zero_base_token_model(&model_path);

        let mut session = InferenceSession::builder()
            .model(&model_path)
            .lora_adapters(vec![
                lora_spec("adapter_a", &a_t, &b_t_a, scale),
                lora_spec("adapter_b", &a_t, &b_t_b, scale),
            ])
            .build()
            .expect("build grouped multi-adapter native session");
        let route_a = session.resolve_lora_adapter("adapter_a").expect("resolve A").0 as i32;
        let route_b = session.resolve_lora_adapter("adapter_b").expect("resolve B").0 as i32;

        let max_batch = 3;
        let decode = NativeBatchedDecodeSession::new(&mut session, max_batch, 1 << 16)
            .expect("build native batched decode session");

        let tokenizer = tokenizer();
        let mut manager = ContinuousBatchManager::with_lora_adapter_routes(
            Box::new(decode),
            &tokenizer,
            None,
            max_batch,
            vec![
                ("adapter_a".to_string(), route_a),
                ("adapter_b".to_string(), route_b),
            ],
        )
        .expect("build native continuous-batch manager");

        let handle_a = manager.submit(base_request(Some("adapter_a"))).unwrap();
        let handle_b = manager.submit(base_request(Some("adapter_b"))).unwrap();
        let handle_base = manager.submit(base_request(None)).unwrap();

        let mut first_token = std::collections::HashMap::new();
        let record = |events: Vec<ContinuousBatchEvent>,
                      sink: &mut std::collections::HashMap<usize, u32>| {
            for event in events {
                if let ContinuousBatchEvent::Token { handle, token } = event {
                    sink.entry(handle.id).or_insert(token.token_id);
                }
            }
        };
        record(manager.poll(), &mut first_token);
        let mut guard = 0;
        while manager.has_pending_work() {
            manager.step().unwrap();
            record(manager.poll(), &mut first_token);
            guard += 1;
            assert!(guard < 64, "native continuous batch failed to drain");
        }

        assert_eq!(
            first_token.get(&handle_a.id).copied(),
            Some(token_a as u32),
            "native row bound to adapter A must emit A's argmax token"
        );
        assert_eq!(
            first_token.get(&handle_b.id).copied(),
            Some(token_b as u32),
            "native row bound to adapter B must emit B's argmax token"
        );
        assert_eq!(
            first_token.get(&handle_base.id).copied(),
            Some(0u32),
            "native base row must emit the zero-base argmax token"
        );
    }

    /// The native production path rejects an unknown adapter loud at admission
    /// (never a silent base fallback), mirroring the probe path.
    #[test]
    fn native_continuous_batch_unknown_adapter_fails_loud() {
        let scale = 1.0f32;
        let mut a_t = vec![0.0f32; K * RANK];
        for p in 0..K {
            a_t[p * RANK] = 0.1;
        }
        let mut b_t_a = vec![0.0f32; RANK * N];
        b_t_a[1] = 1.0;

        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.onnx");
        write_zero_base_token_model(&model_path);
        let mut session = InferenceSession::builder()
            .model(&model_path)
            .lora_adapters(vec![lora_spec("adapter_a", &a_t, &b_t_a, scale)])
            .build()
            .expect("build grouped native session");
        let route_a = session.resolve_lora_adapter("adapter_a").unwrap().0 as i32;

        let max_batch = 2;
        let decode = NativeBatchedDecodeSession::new(&mut session, max_batch, 1 << 16)
            .expect("build native batched decode session");
        let tokenizer = tokenizer();
        let mut manager = ContinuousBatchManager::with_lora_adapter_routes(
            Box::new(decode),
            &tokenizer,
            None,
            max_batch,
            vec![("adapter_a".to_string(), route_a)],
        )
        .expect("build native manager");

        let error = manager
            .submit(base_request(Some("nope")))
            .expect_err("unknown adapter must fail loud at admission");
        let message = format!("{error:#}");
        assert!(
            message.contains("unknown LoRA adapter"),
            "unexpected error: {message}"
        );
        manager.submit(base_request(None)).expect("base-only admits");
    }
}
