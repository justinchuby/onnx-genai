//! Batched static-cache generation path.

use crate::config::{
    FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult,
};
use crate::decode::ModelDecodePath;
use crate::decode_loop::{
    DecodeLoopState, commit_selected_token, finish_result, reached_context_limit,
};
use crate::engine::Engine;
use crate::logits::{ProcessorChain, ProcessorContext, TokenId};
use crate::processors::{build_processor_chain, ensure_constrained_finish, select_next_token};
use anyhow::Context;
use onnx_genai_ort::{BatchedStaticCacheDecodeSession, StaticCacheDecodeOptions};

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
                )?);
                continue;
            }
            rows.push(BatchRow {
                result_index,
                physical_row: rows.len(),
                context_tokens: prompt_tokens,
                options,
                chain,
                max_context,
                state: DecodeLoopState::new(0),
                pending_logits: None,
                active: true,
            });
        }

        if rows.is_empty() {
            return collect_batch_results(results);
        }

        let mut decode = BatchedStaticCacheDecodeSession::new(
            &self.session,
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
                let token_id = select_next_token(
                    &mut logits,
                    &row.processor_context(),
                    &row.options,
                    &row.chain,
                    0.0,
                );
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
