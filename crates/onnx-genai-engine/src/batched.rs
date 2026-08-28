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
use onnx_genai_metadata::decoder_workflow::{CONTINUOUS_BATCH_CONTRACT, IterationPolicy};
use onnx_genai_ort::Tokenizer;
use onnx_genai_ort::decode::{
    BatchedDecodeSession, BatchedSharedBufferDecodeSession, SharedBufferBatchOptions,
};
use onnx_genai_ort::{BatchedStaticCacheDecodeSession, DataType, StaticCacheDecodeOptions, Value};
use onnx_genai_scheduler::{PreemptionPolicy, Priority, PriorityPolicy, Scheduler};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::pipeline::generation::CONTINUOUS_BATCH_CONTRACTS;
use crate::pipeline::{
    WorkflowGenerationCursor, WorkflowNodeHost, WorkflowNodeRequest, WorkflowRuntime,
};

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

struct PendingContinuousRequest {
    handle: ContinuousBatchHandle,
    prompt_tokens: Vec<TokenId>,
    options: GenerateOptions,
    chain: ProcessorChain,
    max_context: Option<usize>,
}

struct ContinuousBatchRow {
    handle: ContinuousBatchHandle,
    context_tokens: Vec<TokenId>,
    options: GenerateOptions,
    chain: ProcessorChain,
    max_context: Option<usize>,
    state: DecodeLoopState,
    pending: Option<RowPending>,
}

/// A row's prepared next-token input after one decode step.
///
/// The per-row router decides this at extraction time so `advance_row` never
/// has to reason about device residency: a device-portable row already holds
/// the token id selected on the device (no host logits ever materialized),
/// while a host-required row carries the full `[vocab]` logits its chain and
/// logprobs need.
enum RowPending {
    /// Full host logits; the row's processor chain and sampler run in
    /// `advance_row`.
    HostLogits(Vec<f32>),
    /// The token id already selected entirely on the device.
    DeviceToken(TokenId),
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
    /// Every token id this model ends on, resolved once when the manager is
    /// built. Held here because the manager cannot reach the package's workflow
    /// itself, and reading only the tokenizer's first id is what made a
    /// multi-EOS model run past its end on the batched route while stopping
    /// correctly on the single-row one.
    eos_token_ids: Vec<TokenId>,
    metadata_max_context: Option<usize>,
    static_max_len: usize,
    queue: VecDeque<PendingContinuousRequest>,
    rows: Vec<Option<ContinuousBatchRow>>,
    admissions: VecDeque<ContinuousBatchAdmission>,
    events: VecDeque<ContinuousBatchEvent>,
    next_handle: usize,
    /// Device→host logits transfer accounting for the per-row router. Populated
    /// only when the backend hands back a device-resident buffer
    /// ([`onnx_genai_ort::decode::BatchStepLogits::Device`]); it counts the full
    /// vocabulary copies paid for host-required rows separately from the 4-byte
    /// token ids read back for device-sampled rows, so a caller can prove a
    /// mixed batch moved `(host-required rows) x vocab x 4` bytes rather than
    /// `(all rows) x vocab x 4`.
    routed_stats: onnx_genai_ort::decode::LogitsD2hStats,
    used_device_routing: bool,
    occupancy: BatchOccupancy,
    /// The interpreter over this package's loop, re-authored to declare
    /// `onnx-genai.continuous-batch`.
    ///
    /// Every `step` is one iteration of *that* loop: the bound, the liveness
    /// predicate, the carried cells and the emit that publishes the token
    /// stream are read from the authored document, and the row-major forward
    /// pass below is reached because the body's node names the contract this
    /// manager registers an executor for. Nothing here asks what kind of
    /// package it holds.
    runtime: Rc<WorkflowRuntime>,
    /// The in-flight interpretation, held between externally driven steps.
    ///
    /// A server drives a batch one step at a time and admits work between
    /// steps, so the manager cannot call a function that loops to completion —
    /// but it must run *the same loop*, which is what the cursor gives it.
    cursor: Option<WorkflowGenerationCursor>,
    /// Iteration bound the authored loop is started with.
    iteration_bound: usize,
    /// Tokens this pass's executor committed, per positional row.
    ///
    /// Held against what the authored emit published — row by row, because a
    /// batch's emit accumulates one stream per row — so the declared emit stays
    /// load-bearing: a row whose tokens stopped reaching the stream would
    /// otherwise be invisible behind the event queue the caller actually reads.
    committed_per_row: Vec<Vec<i64>>,
    /// Declared passes this manager has completed.
    ///
    /// A pass that ended while work remained would restart transparently on the
    /// next `step`, so premature termination self-heals and is invisible in the
    /// token stream. Counting passes is what makes it visible: a batch driven
    /// to completion is one pass.
    passes: usize,
}

/// How many sequences actually shared each batched forward pass.
///
/// A server can admit many concurrent generations and still decode them one at
/// a time — an admission gauge counts requests in flight, not rows co-decoded,
/// so it cannot tell continuous batching from serialization. This counts the
/// rows carried by each forward the manager issues, so the distinction is
/// observable rather than inferred from latency.
///
/// `steps` counts forwards, not tokens: prompt-context steps advance rows
/// without emitting a token, and they are batched too.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchOccupancy {
    /// Batched forward passes issued by this manager.
    pub steps: u64,
    /// Sum over steps of the rows advanced by that step.
    pub rows_advanced: u64,
    /// Largest number of rows advanced by any single step.
    pub max_rows_in_step: usize,
    /// Physical decode rows the manager owns.
    pub max_batch: usize,
}

impl BatchOccupancy {
    /// Mean rows per batched forward, or `None` before any forward ran.
    ///
    /// Strictly greater than 1.0 exactly when some forward carried more than
    /// one sequence, which is the property that distinguishes real continuous
    /// batching from serialized decode.
    pub fn mean_rows_per_step(&self) -> Option<f64> {
        (self.steps > 0).then(|| self.rows_advanced as f64 / self.steps as f64)
    }

    /// Peak fraction of the physical batch that was ever co-decoded, in `0.0..=1.0`.
    pub fn peak_utilization(&self) -> f64 {
        if self.max_batch == 0 {
            return 0.0;
        }
        self.max_rows_in_step as f64 / self.max_batch as f64
    }

    fn record_step(&mut self, rows: usize) {
        self.steps += 1;
        self.rows_advanced += rows as u64;
        self.max_rows_in_step = self.max_rows_in_step.max(rows);
    }
}

impl<'a> ContinuousBatchManager<'a> {
    fn new(
        mut decode: Box<dyn BatchedDecodeSession<'a> + 'a>,
        tokenizer: &'a Tokenizer,
        eos_token_ids: Vec<TokenId>,
        metadata_max_context: Option<usize>,
        max_batch: usize,
        runtime: Rc<WorkflowRuntime>,
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
            eos_token_ids,
            metadata_max_context,
            static_max_len,
            queue: VecDeque::new(),
            rows: (0..max_batch).map(|_| None).collect(),
            admissions: VecDeque::new(),
            events: VecDeque::new(),
            next_handle: 0,
            routed_stats: onnx_genai_ort::decode::LogitsD2hStats::default(),
            used_device_routing: false,
            occupancy: BatchOccupancy {
                max_batch,
                ..Default::default()
            },
            runtime,
            cursor: None,
            // A continuous batch has no global token budget: each row carries
            // its own `max_new_tokens`, which the executor's token policy
            // applies, and the batch itself runs while any row is live. The
            // authored loop still declares a bound, so it is given the longest
            // sequence the package says it can hold — a bound that is real
            // rather than a number invented to be large.
            iteration_bound: metadata_max_context.unwrap_or(static_max_len).max(1),
            committed_per_row: vec![Vec::new(); max_batch],
            passes: 0,
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
        crate::engine::apply_eos_policy(&mut options, &self.eos_token_ids);
        let prompt_tokens = match request.prompt {
            GeneratePrompt::TokenIds(tokens) => tokens,
            GeneratePrompt::TokenRows(_) => {
                anyhow::bail!("multi-row prompts are supported only by workflow pipelines")
            }
            GeneratePrompt::Text(text) => self
                .tokenizer
                .encode(&text)
                .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e}"))?,
        };
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        let max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(&options, Some(self.tokenizer), false)?;
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

    /// Advance the authored loop by one iteration.
    ///
    /// The iteration is the workflow's, not this method's: the cursor walks the
    /// re-authored body, the body's node names
    /// `onnx-genai.continuous-batch`, and the interpreter routes that node here
    /// because this manager registered an executor for that contract. What used
    /// to be the loop is now [`Self::run_batch_node`], which advances the rows
    /// once and publishes the predicate the loop reads.
    pub fn step(&mut self) -> anyhow::Result<()> {
        // Admitted before the pass is started so the loop's first liveness read
        // sees the rows a caller submitted, rather than ending a batch that has
        // work waiting.
        self.admit_available_rows();
        let runtime = Rc::clone(&self.runtime);
        let mut cursor = match self.cursor.take() {
            Some(cursor) => cursor,
            None => {
                let request = self.iteration_request()?;
                let mut host: Option<&mut dyn WorkflowNodeHost> = Some(self);
                WorkflowGenerationCursor::start(
                    &runtime,
                    request,
                    CONTINUOUS_BATCH_CONTRACTS,
                    &mut host,
                )?
            }
        };
        let ran = {
            let mut host: Option<&mut dyn WorkflowNodeHost> = Some(self);
            cursor.advance(&runtime, &mut host)?
        };
        if ran {
            self.cursor = Some(cursor);
            return Ok(());
        }
        // The loop ended: no row will decode again, or the declared bound is
        // spent. A later `submit` starts a new pass rather than resuming a
        // finished one, which is what keeps the emitted stream a statement
        // about one batch. Reading the emitted rows back is deliberately done
        // here and not per step: it materializes every row's accumulated
        // stream, which on the per-token path would be quadratic in the tokens
        // it is only checking.
        let emitted = cursor.emitted_token_rows(&runtime)?;
        self.finish_pass(&emitted)
    }

    /// Cross-check the pass's declared emit against what the executor committed.
    ///
    /// Row by row. An aggregate comparison would be satisfied by a body that
    /// published one row's tokens twice and another's not at all, which is
    /// exactly the failure a recycled batch row can produce.
    fn finish_pass(&mut self, emitted: &[Vec<i64>]) -> anyhow::Result<()> {
        for (row, committed) in self.committed_per_row.iter().enumerate() {
            let published = emitted.get(row).map_or(&[][..], Vec::as_slice);
            anyhow::ensure!(
                published == committed,
                "the authored continuous-batch loop emitted {published:?} into row {row} \
                 while its executor committed {committed:?}; the declared token stream and the \
                 generated one must be the same stream"
            );
        }
        self.cursor = None;
        self.committed_per_row = vec![Vec::new(); self.rows.len()];
        self.passes += 1;
        Ok(())
    }

    /// The request the authored loop is bound with.
    ///
    /// Its prompt is a one-token placeholder per physical row, and that is a
    /// statement rather than a shortcut: every package input this body consumes
    /// is consumed by the hosted node, so the plan classifies them as
    /// host-supplied and the executor answers for them out of the rows it
    /// holds. What the placeholder does is fix the row axis — the loop's
    /// row-scoped cells are seeded at the batch's width, so a row that stops
    /// keeps its position instead of shifting its neighbours'.
    ///
    /// The bound is the loop's own, read from the authored document.
    fn iteration_request(&self) -> anyhow::Result<crate::pipeline::PipelineGenerateRequest> {
        let rows = self.rows.len();
        let mut request = crate::pipeline::PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenRows(vec![vec![0]; rows]),
            options: GenerateOptions {
                max_new_tokens: self.iteration_bound,
                ..GenerateOptions::default()
            },
        });
        let declared = &self.runtime.workflow_spec().inputs;
        let rows_i64 = i64::try_from(rows).context("continuous batch width exceeds int64")?;
        for (cell, seed) in [
            ("active", RowSeed::Flag(true)),
            ("done", RowSeed::Flag(false)),
            ("accepted_len", RowSeed::Count(0)),
            ("cache_lengths", RowSeed::Count(0)),
        ] {
            let name = format!("package.{cell}");
            if !declared.contains_key(&name) {
                continue;
            }
            let value = match seed {
                RowSeed::Flag(flag) => row_flags(&vec![flag; rows])?,
                RowSeed::Count(count) => Value::from_slice_i64(&vec![count; rows], &[rows_i64])?,
            };
            request = request.with_input(&name, value);
        }
        Ok(request)
    }

    /// Advance every live row by one shared forward pass, then say which rows
    /// are still generating.
    ///
    /// One iteration of the declared loop. It holds nothing about the loop
    /// itself: how many iterations may run, whether another may, and what the
    /// emit publishes are the interpreter's, read from the authored document.
    fn run_batch_node(&mut self, request: &mut WorkflowNodeRequest<'_>) -> anyhow::Result<()> {
        self.admit_available_rows();
        let ready_rows = self
            .rows
            .iter()
            .enumerate()
            .filter_map(|(row_index, row)| {
                row.as_ref()
                    .and_then(|row| row.pending.is_some().then_some(row_index))
            })
            .collect::<Vec<_>>();

        let mut committed: Vec<Option<TokenId>> = vec![None; self.rows.len()];
        for row_index in ready_rows {
            let mut row = self.rows[row_index]
                .take()
                .context("ready continuous row disappeared")?;
            let (token, finished) = self.advance_row(&mut row)?;
            committed[row_index] = Some(token);
            self.committed_per_row[row_index].push(i64::from(token));
            if finished {
                self.decode
                    .deactivate_row(row_index)
                    .map_err(|e| anyhow::anyhow!("Failed to deactivate continuous row: {e}"))?;
            } else {
                self.rows[row_index] = Some(row);
            }
        }

        self.decode_next_pending_rows()?;
        self.publish_row_outputs(request, &committed)
    }

    /// Rows that will decode again on the next iteration.
    ///
    /// This is the *batch's* liveness, published for every slot, and it has to
    /// be: the loop reads its predicate before the body runs and preserves an
    /// inactive row's carried state afterwards, so a slot that published
    /// `false` can never be brought back within the pass. A slot that retires
    /// while another row is still generating may be backfilled before the next
    /// body runs — by an admission this node performs, or by a `submit` between
    /// two steps — and a per-slot `false` would retire it permanently, dropping
    /// the backfilled request's tokens out of the declared stream and, once its
    /// neighbours finish, ending a pass that still has work.
    ///
    /// The per-row question the emit actually needs — *did this slot contribute
    /// a token this iteration?* — is answered by `accepted_len`, which is zero
    /// for a slot that sat the step out. So a retired slot reported live here
    /// publishes nothing.
    ///
    /// When no slot holds a request and nothing is queued, no row is live and
    /// the loop's own predicate ends the pass, which is precisely the batch's
    /// stop condition, stated once, in the document.
    fn live_rows(&self) -> Vec<bool> {
        vec![self.has_pending_work(); self.rows.len()]
    }

    /// Define the SSA values the batch node declares.
    ///
    /// One entry per *physical* row, always, so the row axis the emit slices is
    /// the same width on every iteration: a row nobody occupies is inactive and
    /// contributes nothing, rather than shifting its neighbours' positions.
    ///
    /// State ports the block also declares — the KV halves the decoder bound —
    /// are deliberately left undefined: their buffers belong to the decode
    /// session this executor owns, exactly as they do for the single-row
    /// decode core, and materializing them here would mean copying a cache the
    /// loop never reads.
    fn publish_row_outputs(
        &self,
        request: &mut WorkflowNodeRequest<'_>,
        committed: &[Option<TokenId>],
    ) -> anyhow::Result<()> {
        let rows = self.rows.len();
        for (port, value) in request.outputs.iter() {
            let tensor = match port.as_str() {
                "token" => {
                    let tokens = committed
                        .iter()
                        .map(|token| i64::from(token.unwrap_or(0)))
                        .collect::<Vec<_>>();
                    Value::from_slice_i64(&tokens, &[rows as i64, 1])?
                }
                // A row is live while it will decode again: it holds a
                // request, or the batch has one queued that this slot will be
                // backfilled with. Reporting the *request*'s liveness instead
                // would retire a slot the batch is about to refill, and the
                // loop preserves an inactive row's carried state — so a
                // recycled row would stay retired for the rest of the pass.
                "active" => row_flags(&self.live_rows())?,
                "done" => row_flags(
                    &self
                        .live_rows()
                        .into_iter()
                        .map(|live| !live)
                        .collect::<Vec<_>>(),
                )?,
                "accepted_len" => {
                    let accepted = committed
                        .iter()
                        .map(|token| i64::from(token.is_some()))
                        .collect::<Vec<_>>();
                    Value::from_slice_i64(&accepted, &[rows as i64])?
                }
                "cache_lengths" | "lengths" => {
                    let lengths = (0..rows)
                        .map(|row| {
                            self.decode
                                .row_len(row)
                                .map_err(|e| {
                                    anyhow::anyhow!("Failed to read continuous row length: {e}")
                                })
                                .and_then(|len| {
                                    i64::try_from(len).context("cache length exceeds int64")
                                })
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    Value::from_slice_i64(&lengths, &[rows as i64])?
                }
                _ => continue,
            };
            request.values.insert(value.clone(), tensor);
        }
        Ok(())
    }

    /// End the current pass, reconciling the declared emit with what the
    /// executor committed.
    ///
    /// A caller that stops stepping as soon as its results are in leaves the
    /// authored loop one liveness read short of ending, and the cross-check in
    /// [`Self::finish_pass`] would never run — a check nobody reaches is not a
    /// check. Draining is what makes the declared emit load-bearing for the
    /// run-to-completion routes, which is where most batches are driven.
    pub fn drain(&mut self) -> anyhow::Result<()> {
        while self.cursor.is_some() {
            anyhow::ensure!(
                !self.has_pending_work(),
                "a continuous batch cannot be drained while {} rows and {} queued requests are \
                 still live",
                self.active_len(),
                self.pending_len()
            );
            self.step()?;
        }
        Ok(())
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

    /// Rows actually co-decoded per batched forward so far.
    ///
    /// Read this instead of an admitted-request gauge to tell whether the
    /// backend really advanced several sequences per step.
    pub fn occupancy(&self) -> BatchOccupancy {
        self.occupancy
    }

    /// Cumulative device→host logits transfer cost of the backend driving this
    /// manager.
    ///
    /// When the backend hands back a device-resident logits buffer, the per-row
    /// router owns the transfers and this returns the router's accounting: full
    /// vocabulary copies charged only to host-required rows, plus the 4-byte
    /// token ids read back for device-sampled rows. Otherwise it reports the
    /// backend's own cost — the native host-logits seam round-trips every row
    /// each step, and the ORT backends keep logits host-side and report `None`,
    /// so a caller can quote the manager's honest D2H cost or tell that the
    /// backend pays none.
    pub fn logits_d2h_stats(&self) -> Option<onnx_genai_ort::decode::LogitsD2hStats> {
        if self.used_device_routing {
            Some(self.routed_stats)
        } else {
            self.decode.logits_d2h_stats()
        }
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
            context_tokens: pending.prompt_tokens,
            options: pending.options,
            chain: pending.chain,
            max_context: pending.max_context,
            state: loop_state,
            pending: None,
        };
        self.rows[row_index] = Some(row);
        Ok(())
    }

    fn advance_row(&mut self, row: &mut ContinuousBatchRow) -> anyhow::Result<(TokenId, bool)> {
        let token_id = match row
            .pending
            .take()
            .context("active continuous row has no pending token")?
        {
            // The row was sampled entirely on the device; its token id is
            // already selected and no host logits were ever materialized. The
            // request RNG was advanced when the token was drawn (`take_row_pending`),
            // so the seeded stream matches the host path.
            RowPending::DeviceToken(token_id) => token_id,
            RowPending::HostLogits(mut logits) => {
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
                token_id
            }
        };

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
            Ok((token_id, true))
        } else {
            Ok((token_id, false))
        }
    }

    /// Note that a decode step produced device-resident logits so the manager's
    /// D2H accounting (not the backend's) is authoritative for `logits_d2h_stats`.
    fn account_device_step(&mut self, logits: &onnx_genai_ort::decode::BatchStepLogits) {
        if matches!(logits, onnx_genai_ort::decode::BatchStepLogits::Device(_)) {
            self.used_device_routing = true;
            self.routed_stats.steps += 1;
        }
    }

    /// Route one row's step logits: sample device-portable rows entirely on the
    /// device (only a 4-byte token id crosses the bus) and copy the full
    /// `[vocab]` row to the host only for rows that need it (history-dependent
    /// processors, logprobs). `buf_index` indexes the step's logits buffer
    /// (active-row order after `step_active`, physical row after `step_select`);
    /// `row_index` indexes `self.rows`.
    ///
    /// For a host-only backend (`Ort`/`HostRows`) every row is `Host`, so this
    /// is exactly the previous demux with no behavior change. The per-row win is
    /// realized only when the backend hands back a `Device` buffer.
    fn take_row_pending(
        &mut self,
        logits: &mut onnx_genai_ort::decode::BatchStepLogits,
        row_index: usize,
        buf_index: usize,
    ) -> anyhow::Result<()> {
        let device_available = matches!(logits, onnx_genai_ort::decode::BatchStepLogits::Device(_));
        let row = self.rows[row_index]
            .as_mut()
            .context("continuous row is not assigned")?;
        let plan = crate::processors::device_sampling_plan(
            &row.chain,
            &row.options,
            row.state.custom_sampler.is_some(),
            device_available,
            device_available,
        );
        match plan {
            crate::processors::DeviceSamplingPlan::Greedy
            | crate::processors::DeviceSamplingPlan::Sampled => {
                let params = onnx_genai_ort::DeviceSampleParams {
                    temperature: row.options.temperature,
                    top_k: row.options.top_k,
                    top_p: row.options.top_p,
                    min_p: row.options.min_p,
                    greedy: matches!(plan, crate::processors::DeviceSamplingPlan::Greedy),
                    // Draw from the same request RNG as the host categorical path
                    // so the seeded token stream is identical whether the row is
                    // routed to the device or the host.
                    rng_value: row.state.rng.value_for(&row.options),
                };
                let token = device_sample_row(logits, buf_index, &params)?;
                row.pending = Some(RowPending::DeviceToken(token));
                self.routed_stats.rows_device_sampled += 1;
                self.routed_stats.token_id_bytes += u128::from(TOKEN_ID_BYTES);
            }
            crate::processors::DeviceSamplingPlan::Host => {
                let host = take_row_logits(logits, buf_index, 0)?;
                if device_available {
                    self.routed_stats.bytes += host.len() as u128 * u128::from(LOGIT_BYTES);
                    self.routed_stats.rows_host_copied += 1;
                }
                row.pending = Some(RowPending::HostLogits(host));
            }
        }
        Ok(())
    }

    /// Record one batched forward and log the rows it carried.
    ///
    /// `seam` names which backend entry point issued it, because the two seams
    /// index their logits buffers differently and a wrong-seam row count would
    /// otherwise be invisible.
    fn record_forward(&mut self, rows: usize, seam: &'static str) {
        self.occupancy.record_step(rows);
        tracing::debug!(
            seam,
            rows,
            max_batch = self.occupancy.max_batch,
            queued = self.queue.len(),
            step = self.occupancy.steps,
            "continuous batch forward"
        );
    }

    fn decode_next_pending_rows(&mut self) -> anyhow::Result<()> {
        let advancing_rows = self
            .rows
            .iter()
            .enumerate()
            .filter_map(|(row_index, row)| {
                row.as_ref()
                    .is_some_and(|row| row.pending.is_none())
                    .then_some(row_index)
            })
            .collect::<Vec<_>>();
        if advancing_rows.is_empty() {
            return Ok(());
        }
        let active_rows = self.decode.active_rows();
        if advancing_rows.len() == active_rows.len() {
            self.record_forward(active_rows.len(), "step_active");
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
            let mut logits = self
                .decode
                .step_active(&input_ids, &position_ids)
                .map_err(|e| anyhow::anyhow!("Continuous active static-cache step failed: {e}"))?;
            self.account_device_step(&logits);
            for (active_index, logical_row) in active_rows.into_iter().enumerate() {
                if completes_context[active_index] {
                    self.take_row_pending(&mut logits, logical_row, active_index)?;
                }
            }
            return Ok(());
        }

        let mut input_ids = vec![0_i64; self.max_batch()];
        let mut position_ids = vec![0_i64; self.max_batch()];
        let mut advance_rows = vec![false; self.max_batch()];
        let mut completes_context = vec![false; self.max_batch()];
        self.record_forward(advancing_rows.len(), "step_select");
        for (row_index, row) in self
            .rows
            .iter()
            .enumerate()
            .filter_map(|(row_index, row)| row.as_ref().map(|row| (row_index, row)))
        {
            if row.pending.is_none() {
                let row_len = self
                    .decode
                    .row_len(row_index)
                    .map_err(|e| anyhow::anyhow!("Failed to read continuous row length: {e}"))?;
                let token = *row.context_tokens.get(row_len).with_context(|| {
                    format!(
                        "continuous row {row_index} has no token to advance at offset {row_len}"
                    )
                })?;
                input_ids[row_index] = i64::from(token);
                position_ids[row_index] = row_len as i64;
                advance_rows[row_index] = true;
                completes_context[row_index] = row_len + 1 == row.context_tokens.len();
            }
        }
        let mut logits = self
            .decode
            .step_select(&input_ids, &position_ids, &advance_rows)
            .map_err(|e| anyhow::anyhow!("Continuous static-cache decode step failed: {e}"))?;
        self.account_device_step(&logits);
        let extract_rows = self
            .rows
            .iter()
            .enumerate()
            .filter_map(|(row_index, row)| {
                row.as_ref()
                    .is_some_and(|_| advance_rows[row_index] && completes_context[row_index])
                    .then_some(row_index)
            })
            .collect::<Vec<_>>();
        for physical_row in extract_rows {
            self.take_row_pending(&mut logits, physical_row, physical_row)?;
        }
        Ok(())
    }
}

impl Engine {
    /// Why this engine must use isolated generation rather than a shared
    /// live-row forward.
    ///
    /// This is deliberately an eligibility check rather than package
    /// validation.  The workflow remains executable when the selected backend
    /// cannot share its forward, when the graph shape is unsuitable for the
    /// specialized runner, or when its declared loop has no compactable
    /// row-major form.
    fn shared_batch_ineligible_reason(&self) -> Option<String> {
        if !self.holds_decode_core() {
            return Some(
                "the declared workflow has no fused decode executor that can issue a shared \
                 live-row forward"
                    .to_string(),
            );
        }
        if let Err(error) =
            crate::pipeline::validate_generation_workflow(self.workflow.workflow_spec())
        {
            return Some(format!(
                "the declared generation loop is not eligible for the specialized shared \
                 live-row executor: {error:#}"
            ));
        }
        if self.workflow.workflow_spec().inputs.values().any(|input| {
            matches!(
                 &input.role,
                 onnx_genai_metadata::SemanticInputRole::Runtime { role, .. }
                     if *role == onnx_genai_metadata::RuntimeInputRole::RowSelection
            )
        }) {
            return Some(
                "the declared workflow supplies a positional row plan, but the specialized \
                  shared decoder cannot apply that plan atomically to backend KV state, request \
                  values, semantic state, effects, RNG, and output ownership; use isolated \
                  workflow execution"
                    .to_string(),
            );
        }
        let capability = self.batching_capability();
        (!capability.supports_batching()).then(|| capability.reason().to_string())
    }

    /// Execute each request through the ordinary single-request path.
    ///
    /// Formation, row placement, and shared-forward support are optimization
    /// concerns.  This is therefore the only fallback: it preserves the normal
    /// scheduler admission, request-local state, callbacks, EOS handling, and
    /// result order without inventing a second isolated token policy.
    fn run_isolated_batch(
        &mut self,
        requests: Vec<GenerateRequest>,
    ) -> anyhow::Result<Vec<GenerateResult>> {
        requests
            .into_iter()
            .enumerate()
            .map(|(index, request)| {
                self.generate(request)
                    .with_context(|| format!("isolated batch request {index}"))
            })
            .collect()
    }

    /// Refuse a package whose declared workflow the batched routes cannot run.
    ///
    /// Batching advances N rows through *one* decoder forward pass, so it needs
    /// a package whose declared graph is that decoder and whose stop policy is
    /// the runtime's. A package that declares an in-graph sampler or several
    /// components has a loop of its own shape, and the row-major batch would be
    /// executing something other than what it declared.
    ///
    /// Checked at the routes' single admission point, before a slot is taken.
    fn assert_batched_generation_declared(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.holds_decode_core(),
            "batched generation runs N rows through one decoder forward pass, which requires the \
             package's declared decode step to be one this runtime has a fused executor for; \
             this package declares a workflow whose components the interpreter invokes \
             individually, so its declared loop is not a row-major batch"
        );
        crate::pipeline::validate_generation_workflow(self.workflow.workflow_spec())
    }
    /// Generate a fixed batch of independent requests.
    ///
    /// A fixed batch and a continuously backfilled one are the *same* iteration
    /// algorithm with different admission: N rows advance through one shared
    /// forward pass, each row keeps its own processor chain, sampling options,
    /// stop conditions and context limit, and a finished row stops being
    /// sampled. So this runs the same authored body — the one whose node
    /// declares `onnx-genai.continuous-batch` — through the same executor,
    /// admitting every request up front and never backfilling.
    ///
    /// Expressing it this way is what removes the second token policy: before,
    /// this method selected, committed and stopped with its own copy of the
    /// per-row rules, which is exactly the kind of duplicate that drifts.
    ///
    /// Two deliberate consequences of sharing the route:
    ///
    /// * It now serves every batchable decode path, not only a fixed-capacity
    ///   cache. The old refusal named STATIC-CACHE because this method built
    ///   that session itself; the shared route asks
    ///   [`Self::continuous_batch_manager`], which has served shared-buffer
    ///   past/present packages since they became batchable. A package that used
    ///   to be refused here now runs, and reports the same tokens a single-row
    ///   generation would.
    /// * A physical row is reserved per request, including a request whose
    ///   prompt already fills the context and therefore finishes at submission
    ///   without occupying its row. Reserving the caller's own batch width is
    ///   the honest bound; narrowing it would mean tokenizing every prompt twice
    ///   to find out.
    pub fn generate_batched_static(
        &mut self,
        requests: Vec<GenerateRequest>,
    ) -> anyhow::Result<Vec<GenerateResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let max_batch = requests.len();
        self.run_continuous_batch(requests, max_batch)
    }

    /// Report how many sequences this engine's decode path can advance in a
    /// single shared forward pass, and why.
    ///
    /// The answer is derived from the resolved decode backend, the model-I/O
    /// [`ModelDecodePath`], and — on the native backend — the actually-bound CUDA
    /// persistent batch extent, **not** from whether an ORT decoder session
    /// happens to be present. On the native backend the answer stays consistent
    /// with what [`Self::continuous_batch_manager`] will build: batch 1 pins to a
    /// single sequence (a structural invariant of `native_decode/cuda.rs`), while
    /// a session pinned to batch >= 2 reports that many concurrent sequences.
    pub fn batching_capability(&self) -> BatchingCapability {
        // The native decode path advances one sequence per step *unless* it is
        // running a CUDA persistent batch session pinned to batch >= 2 (via
        // `ONNX_GENAI_NATIVE_DECODE_BATCH`), in which case #750 wired the
        // continuous-batch manager onto it and it advances that many sequences per
        // fused forward. Without such a session the batch and query-seq axes are
        // pinned to 1 as a structural decode invariant (`native_decode/cuda.rs`),
        // which is not a tunable — so the answer is derived from the actual bound
        // session, not the model's KV I/O shape, and it stays consistent with what
        // `continuous_batch_manager` will actually build.
        if self.decode_backend == EngineDecodeBackend::Native {
            #[cfg(feature = "native-backend")]
            if let Some(batch) = self
                .native_session
                .as_ref()
                .map(crate::native_decode::NativeDecodeSession::native_decode_batch)
                .filter(|&batch| batch >= 2)
            {
                return BatchingCapability {
                    max_concurrent_sequences: Some(batch),
                    reason: format!(
                        "the native CUDA persistent decode session is pinned to batch {batch}: \
                         the continuous-batch manager advances up to {batch} sequences per fused \
                         forward (ONNX_GENAI_NATIVE_DECODE_BATCH)"
                    ),
                };
            }
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
                reason: "this past/present model is not using a shared KV buffer, \
                         so only one sequence can be decoded at a time. A shared \
                         buffer needs three things to agree: the package must \
                         declare `aliasing` as `permitted` or `required` \
                         (silence means forbidden), the package must declare \
                         `model.max_sequence_length` to size the reservation, and \
                         the execution provider must report fixed-capacity \
                         present binding"
                    .to_string(),
            },
            ModelDecodePath::Generic => BatchingCapability {
                max_concurrent_sequences: Some(1),
                reason: "this generic graph-I/O decode path has no declared \
                         batchable KV service and cannot batch"
                    .to_string(),
            },
        }
    }

    /// Create a lower-level continuous-batch manager for incremental serving.
    ///
    /// Takes `&mut self` because the native backend drives the manager over a
    /// `&mut NativeDecodeSession` — every decode step mutates the persistent
    /// device bindings — while the ORT backends only need `&self.session`; a
    /// single `&mut self` entry point serves both without disadvantaging either
    /// (the ORT branch takes a shared borrow of `self.session` out of the `&mut`,
    /// the native branch a mutable borrow of `self.native_session`, disjoint from
    /// the shared borrow of `self.tokenizer`).
    pub fn continuous_batch_manager(
        &mut self,
        max_batch: usize,
    ) -> anyhow::Result<ContinuousBatchManager<'_>> {
        if max_batch == 0 {
            anyhow::bail!("continuous batch max_batch must be greater than zero");
        }
        // The single admission point for both continuous-batch entry points, so
        // the precondition is checked once, here, rather than once per caller
        // where a new caller could forget it.
        self.assert_batched_generation_declared()?;
        // The authored document this batch iterates. Built before the session
        // borrows below because it is read from the package's declared
        // workflow, which the manager cannot reach once it holds a mutable
        // borrow of the decode session.
        let runtime = self
            .workflow
            .iteration_runtime(IterationPolicy::ContinuousBatch)?;
        // Resolved before the session borrows below, and from the same source
        // the single-row path uses: the manager cannot reach the package's
        // workflow once it holds a mutable session borrow.
        let eos_token_ids = self.default_eos_token_ids()?;
        // Native backend (#750 stage 4): wire the manager onto the native CUDA
        // persistent batch path. Stage 3a made the fused forward ragged (per-row
        // `row_lens`/`position_ids`/mask window) and stage 3b built the two seams
        // this needs — host `[B, 1, vocab]` logits via
        // `decode_greedy_batch_ragged_logits` (the device-argmax fast path is
        // untouched and stays the greedy default) and mid-flight
        // `assign_batch_row`/`deactivate_batch_row` backfill that leaves peers'
        // captured graph intact. The remaining gap was purely the trait shape:
        // `BatchedDecodeSession::step_*` used to return an ORT `Value`, whereas
        // the native seam returns host `Vec<Vec<f32>>`. That is now reconciled by
        // `BatchStepLogits`, whose `HostRows` variant carries the native seam's
        // per-row logits and is *moved* out by `take_row` — so the throughput path
        // adds no logits copy on top of its single D2H read — while the ORT
        // `Value` variant keeps the exact previous demux. `NativeBatchedDecodeSession`
        // adapts the `&mut NativeDecodeSession` seams to the trait.
        #[cfg(feature = "native-backend")]
        if self.decode_backend == EngineDecodeBackend::Native {
            let metadata_max_context = self
                .metadata
                .model
                .as_ref()
                .and_then(|model| model.max_sequence_length);
            let tokenizer = self.tokenizer.as_ref().context(
                "continuous batching requires a tokenizer, which this package does not declare",
            )?;
            let session = self.native_session.as_mut().context(
                "continuous batching on the native backend requires an engine-owned native \
                 decode session; none is loaded",
            )?;
            let decode: Box<dyn BatchedDecodeSession<'_> + '_> =
                Box::new(NativeBatchedDecodeSession::new(session, max_batch)?);
            return ContinuousBatchManager::new(
                decode,
                tokenizer,
                eos_token_ids,
                metadata_max_context,
                max_batch,
                runtime,
            );
        }
        #[cfg(not(feature = "native-backend"))]
        if self.decode_backend == EngineDecodeBackend::Native {
            anyhow::bail!(
                "continuous batching on the native decode backend requires building \
                 onnx-genai-engine with the 'native-backend' feature"
            );
        }
        let session = self
            .session
            .as_deref()
            .context("ORT decoder session is unavailable")?;
        let batch_size = i64::try_from(max_batch).context("batch size exceeds i64")?;
        let decode: Box<dyn BatchedDecodeSession<'_> + '_> = match self.decode_path {
            ModelDecodePath::StaticCache { .. } => {
                let io = self.metadata.decoder_io();
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
                let io = self.metadata.decoder_io();
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
            // `shared_buffer: false`, which is decided jointly by the PACKAGE
            // (`aliasing`, plus a declared `model.max_sequence_length`
            // to size the reservation) and by the DEPLOYMENT
            // (`supports_fixed_capacity_present_binding()`), so the same model
            // may batch or not depending on both the package's declaration and
            // the execution provider. A legacy model cannot batch under any
            // launch. Collapsing them emits one sentence that tells an operator
            // to change the model when the real fix may be the provider, and
            // vice versa.
            ModelDecodePath::PastPresent { .. } => {
                anyhow::bail!(
                    "continuous batching requires a shared KV buffer, and this \
                     past/present model is not using one: the package must declare \
                     `aliasing: permitted` (silence means forbidden) and a \
                     `model.max_sequence_length`, and the execution provider must \
                     report fixed-capacity present binding"
                );
            }
            ModelDecodePath::Generic => {
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
        ContinuousBatchManager::new(
            decode,
            self.tokenizer.as_ref().context(
                "continuous batching requires a tokenizer, which this package does not declare",
            )?,
            eos_token_ids,
            metadata_max_context,
            max_batch,
            runtime,
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
        if max_batch == 0 {
            anyhow::bail!("continuous batch max_batch must be greater than zero");
        }
        if let Some(reason) = self.shared_batch_ineligible_reason() {
            tracing::debug!(%reason, "declining shared live-row batching; using isolated execution");
            return self.run_isolated_batch(requests);
        }
        // Constructing a shared session only allocates its private execution
        // resources.  If a backend rejects its shape, capture, or binding ABI
        // here, nothing request-visible has run yet and isolation remains
        // correct.  Errors after construction are never retried in isolation:
        // a forward may already have changed state or published output.
        let fallback_requests = requests.clone();
        let expected_results = requests.len();
        let manager_result = self.continuous_batch_manager(max_batch);
        if let Err(error) = &manager_result {
            tracing::debug!(
                %error,
                "shared live-row setup declined; using isolated execution"
            );
        }
        if manager_result.is_err() {
            drop(manager_result);
            return self.run_isolated_batch(fallback_requests);
        }
        let mut manager = manager_result.expect("checked shared manager construction");
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
        manager.drain()?;
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
        if let Some(reason) = self.shared_batch_ineligible_reason() {
            tracing::debug!(%reason, "declining scheduled shared live-row batching; using isolated execution");
            return self.run_isolated_batch(requests);
        }
        let fallback_requests = requests.clone();
        let expected_results = requests.len();

        // Tokenize every prompt up front so the scheduler and the batch manager
        // agree on the exact prompt length, and feed the manager token ids so no
        // re-tokenization can drift between the two.
        let mut prepared = Vec::with_capacity(expected_results);
        let mut token_requests = Vec::with_capacity(expected_results);
        for mut request in requests {
            let prompt_tokens = match &request.prompt {
                GeneratePrompt::TokenIds(tokens) => tokens.clone(),
                GeneratePrompt::TokenRows(_) => {
                    anyhow::bail!("multi-row prompts are supported only by workflow pipelines")
                }
                GeneratePrompt::Text(text) => self
                    .require_tokenizer()?
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

        let manager_result = self.continuous_batch_manager(max_batch);
        if let Err(error) = &manager_result {
            tracing::debug!(
                %error,
                "scheduled shared live-row setup declined; using isolated execution"
            );
        }
        if manager_result.is_err() {
            drop(manager_result);
            return self.run_isolated_batch(fallback_requests);
        }
        let mut manager = manager_result.expect("checked shared manager construction");
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
        manager.drain()?;

        collect_batch_results(results)
    }
}

/// The row-major iteration executor, as the interpreter reaches it.
///
/// One node per call, and nothing about the loop: the manager holds the rows,
/// the decode session and the per-row token policy, while the bound, the
/// liveness predicate and the emit stay the authored document's.
impl WorkflowNodeHost for ContinuousBatchManager<'_> {
    fn hosted_contracts(&self) -> &'static [&'static str] {
        CONTINUOUS_BATCH_CONTRACTS
    }

    fn execute_contract_node(
        &mut self,
        mut request: WorkflowNodeRequest<'_>,
    ) -> anyhow::Result<bool> {
        match request.contract {
            CONTINUOUS_BATCH_CONTRACT => {
                self.run_batch_node(&mut request)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

/// The seed one row-scoped serving cell starts a batch pass with.
enum RowSeed {
    Flag(bool),
    Count(i64),
}

/// One boolean per physical batch row.
fn row_flags(flags: &[bool]) -> anyhow::Result<Value> {
    Value::from_raw_bytes(
        flags.iter().copied().map(u8::from).collect(),
        &[flags.len() as i64],
        DataType::Bool,
    )
    .map_err(Into::into)
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

/// Extract one row's logits from a [`BatchStepLogits`] produced by the batched
/// decode trait. This is a copy on the ORT path (unchanged demux cost) and a
/// move on the native host-logits path, so wiring the manager onto the native
/// backend adds no logits copy beyond the single device→host read the seam
/// already performs.
fn take_row_logits(
    logits: &mut onnx_genai_ort::decode::BatchStepLogits,
    row: usize,
    seq_index: usize,
) -> anyhow::Result<Vec<f32>> {
    logits
        .take_row(row, seq_index)
        .map_err(|e| anyhow::anyhow!("Failed to extract row logits: {e}"))
}

/// Bytes copied device→host for one full `f32` logit.
const LOGIT_BYTES: u32 = 4;
/// Bytes copied device→host for one device-sampled token id (`u32`).
const TOKEN_ID_BYTES: u32 = 4;

/// Sample one row entirely on the device from a device-resident step buffer,
/// returning the selected token id (only 4 bytes cross the bus). Only ever
/// called for the [`onnx_genai_ort::decode::BatchStepLogits::Device`] variant,
/// which the per-row router selects via [`crate::processors::device_sampling_plan`].
fn device_sample_row(
    logits: &mut onnx_genai_ort::decode::BatchStepLogits,
    row: usize,
    params: &onnx_genai_ort::DeviceSampleParams,
) -> anyhow::Result<TokenId> {
    match logits {
        onnx_genai_ort::decode::BatchStepLogits::Device(device) => device
            .sample_row(row, params)
            .map_err(|e| anyhow::anyhow!("Failed to device-sample continuous row {row}: {e}")),
        _ => anyhow::bail!("device_sample_row requires a device-resident logits buffer"),
    }
}

/// Adapts the native CUDA persistent batch decode seams
/// (`NativeDecodeSession::decode_greedy_batch_ragged_logits`,
/// `assign_batch_row`/`deactivate_batch_row`, `active_batch_rows`,
/// `batch_row_len`) to [`BatchedDecodeSession`] so a [`ContinuousBatchManager`]
/// can drive the native backend exactly as it drives the ORT sessions.
///
/// The per-row host logits the seam already reads back are handed to the manager
/// as [`onnx_genai_ort::decode::BatchStepLogits::HostRows`], which `take_row`
/// *moves* out — the throughput path pays only its single `[B, 1, vocab]`
/// device→host read, no extra copy. The cumulative cost of that read is exposed
/// through [`onnx_genai_ort::decode::BatchedDecodeSession::logits_d2h_stats`] so
/// a caller can report the manager's honest D2H cost.
#[cfg(feature = "native-backend")]
struct NativeBatchedDecodeSession<'a> {
    session: &'a mut crate::native_decode::NativeDecodeSession,
    batch: usize,
    max_len: usize,
    d2h: onnx_genai_ort::decode::LogitsD2hStats,
}

#[cfg(feature = "native-backend")]
impl<'a> NativeBatchedDecodeSession<'a> {
    fn new(
        session: &'a mut crate::native_decode::NativeDecodeSession,
        max_batch: usize,
    ) -> anyhow::Result<Self> {
        let batch = session.native_decode_batch();
        if batch < 2 {
            anyhow::bail!(
                "continuous batching on the native backend requires a CUDA persistent decode \
                 session pinned to batch >= 2 (set ONNX_GENAI_NATIVE_DECODE_BATCH); this session \
                 bound batch {batch}"
            );
        }
        if max_batch != batch {
            anyhow::bail!(
                "native continuous batch max_batch {max_batch} must equal the pinned native \
                 decode batch extent {batch} (set ONNX_GENAI_NATIVE_DECODE_BATCH={max_batch}); \
                 the manager owns every physical decode row"
            );
        }
        let max_len = session.batch_kv_max_len().context(
            "native continuous batching requires a CUDA decode session with a known KV max_len",
        )?;
        Ok(Self {
            session,
            batch,
            max_len,
            d2h: onnx_genai_ort::decode::LogitsD2hStats::default(),
        })
    }

    fn accumulate_d2h(&mut self, step: &crate::native_decode::RaggedLogitsStep) {
        self.d2h.bytes += step.d2h_bytes as u128;
        self.d2h.time += step.d2h_time;
        self.d2h.steps += 1;
    }
}

#[cfg(feature = "native-backend")]
fn native_token_ids(next_token_ids: &[i64]) -> onnx_genai_ort::Result<Vec<crate::logits::TokenId>> {
    next_token_ids
        .iter()
        .map(|&id| {
            crate::logits::TokenId::try_from(id).map_err(|_| {
                onnx_genai_ort::OrtError::InvalidArgument(format!(
                    "native continuous batch token id {id} is out of range for u32"
                ))
            })
        })
        .collect()
}

#[cfg(feature = "native-backend")]
fn native_past_lens(position_ids: &[i64]) -> onnx_genai_ort::Result<Vec<usize>> {
    position_ids
        .iter()
        .map(|&pos| {
            usize::try_from(pos).map_err(|_| {
                onnx_genai_ort::OrtError::InvalidArgument(format!(
                    "native continuous batch position id {pos} is negative or out of range"
                ))
            })
        })
        .collect()
}

#[cfg(feature = "native-backend")]
fn native_err(context: &str, err: anyhow::Error) -> onnx_genai_ort::OrtError {
    onnx_genai_ort::OrtError::InvalidArgument(format!("{context}: {err:#}"))
}

#[cfg(feature = "native-backend")]
impl<'a> BatchedDecodeSession<'a> for NativeBatchedDecodeSession<'a> {
    fn batch_size(&self) -> usize {
        self.batch
    }

    fn max_len(&self) -> usize {
        self.max_len
    }

    fn row_len(&self, row: usize) -> onnx_genai_ort::Result<usize> {
        self.session
            .batch_row_len(row)
            .map_err(|e| native_err("native continuous row_len", e))
    }

    fn active_rows(&self) -> Vec<usize> {
        self.session.active_batch_rows()
    }

    fn deactivate_row(&mut self, row: usize) -> onnx_genai_ort::Result<()> {
        self.session
            .deactivate_batch_row(row)
            .map_err(|e| native_err("native continuous deactivate_row", e))
    }

    fn assign_row(&mut self, row: usize) -> onnx_genai_ort::Result<()> {
        self.session
            .assign_batch_row(row)
            .map_err(|e| native_err("native continuous assign_row", e))
    }

    fn step_select(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
        advance_rows: &[bool],
    ) -> onnx_genai_ort::Result<onnx_genai_ort::decode::BatchStepLogits> {
        let tokens = native_token_ids(next_token_ids)?;
        let past_lens = native_past_lens(position_ids)?;
        let step = self
            .session
            .decode_greedy_batch_ragged_logits(&tokens, &past_lens, advance_rows)
            .map_err(|e| native_err("native continuous step_select", e))?;
        self.accumulate_d2h(&step);
        // Physical-row indexed: `logits[slot]` is slot `slot`'s row. Move each
        // row into the trait's owned container — no per-row copy.
        Ok(onnx_genai_ort::decode::BatchStepLogits::HostRows(
            step.logits.into_iter().map(Some).collect(),
        ))
    }

    fn step_active(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
    ) -> onnx_genai_ort::Result<onnx_genai_ort::decode::BatchStepLogits> {
        let active = self.session.active_batch_rows();
        if next_token_ids.len() != active.len() || position_ids.len() != active.len() {
            return Err(onnx_genai_ort::OrtError::InvalidArgument(format!(
                "native continuous step_active expects one input per active row: {} active, \
                 {} tokens, {} positions",
                active.len(),
                next_token_ids.len(),
                position_ids.len()
            )));
        }
        // Expand active-order inputs to full physical-slot arrays the ragged seam
        // expects (empty slots are held at advance=false so they neither grow nor
        // perturb the active rows).
        let tokens_active = native_token_ids(next_token_ids)?;
        let past_active = native_past_lens(position_ids)?;
        let mut tokens = vec![0_u32; self.batch];
        let mut past_lens = vec![0_usize; self.batch];
        let mut advances = vec![false; self.batch];
        for (active_index, &slot) in active.iter().enumerate() {
            tokens[slot] = tokens_active[active_index];
            past_lens[slot] = past_active[active_index];
            advances[slot] = true;
        }
        let mut step = self
            .session
            .decode_greedy_batch_ragged_logits(&tokens, &past_lens, &advances)
            .map_err(|e| native_err("native continuous step_active", e))?;
        self.accumulate_d2h(&step);
        // Gather into active-row order, moving each row out of the physical-slot
        // vector (no per-row copy).
        let mut rows = Vec::with_capacity(active.len());
        for &slot in &active {
            rows.push(Some(std::mem::take(&mut step.logits[slot])));
        }
        Ok(onnx_genai_ort::decode::BatchStepLogits::HostRows(rows))
    }

    fn logits_d2h_stats(&self) -> Option<onnx_genai_ort::decode::LogitsD2hStats> {
        Some(self.d2h)
    }
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
    /// The authored row-major body every manager in this module iterates.
    ///
    /// Built from the same canonical decoder workflow a real minimal package
    /// declares, re-authored by the metadata emitter into the body whose node
    /// names `onnx-genai.continuous-batch`. Using it here rather than a bespoke
    /// document is what keeps these cases on the production path: a manager
    /// that stopped selecting the declared executor would fail here first.
    fn batch_runtime() -> std::rc::Rc<crate::pipeline::WorkflowRuntime> {
        crate::pipeline::generation::test_decoder_runtime()
            .expect("a minimal decoder is expressible as a workflow")
            .iteration_runtime(
                onnx_genai_metadata::decoder_workflow::IterationPolicy::ContinuousBatch,
            )
            .expect("its loop is re-authorable as a row-major batch")
    }

    use super::*;
    use crate::sampling::sample_categorical;
    use onnx_genai_ort::OrtError;
    use onnx_genai_ort::decode::BatchedDecodeSession;
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
        ) -> onnx_genai_ort::Result<onnx_genai_ort::decode::BatchStepLogits> {
            unreachable!("a rejected row must never decode")
        }

        fn step_active(
            &mut self,
            _next_token_ids: &[i64],
            _position_ids: &[i64],
        ) -> onnx_genai_ort::Result<onnx_genai_ort::decode::BatchStepLogits> {
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
        ) -> onnx_genai_ort::Result<onnx_genai_ort::decode::BatchStepLogits> {
            unreachable!("admission must not require backend generation")
        }

        fn step_active(
            &mut self,
            _next_token_ids: &[i64],
            _position_ids: &[i64],
        ) -> onnx_genai_ort::Result<onnx_genai_ort::decode::BatchStepLogits> {
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
        assert_eq!(sequence(0), sequence(1));
    }

    #[test]
    fn batch_one_sampling_matches_the_independent_sequence() {
        let options = GenerateOptions {
            greedy: false,
            temperature: 0.8,
            seed: Some(17),
            ..Default::default()
        };
        let mut batched_rng = SamplingRng::for_row(options.seed, 0);
        let batched = (0..16)
            .map(|_| sample_categorical(&[0.0, 1.0, 2.0], batched_rng.value_for(&options)))
            .collect::<Vec<_>>();
        let mut independent_rng = SamplingRng::for_row(options.seed, 0);
        let independent = (0..16)
            .map(|_| sample_categorical(&[0.0, 1.0, 2.0], independent_rng.value_for(&options)))
            .collect::<Vec<_>>();
        assert_eq!(batched, independent);
    }

    #[test]
    fn heterogeneous_batch_sampling_matches_independent_active_row_runs() {
        let options = [
            GenerateOptions {
                greedy: false,
                temperature: 0.7,
                seed: Some(11),
                ..Default::default()
            },
            GenerateOptions {
                greedy: true,
                seed: Some(22),
                ..Default::default()
            },
            GenerateOptions {
                greedy: false,
                temperature: 1.3,
                seed: Some(33),
                ..Default::default()
            },
        ];
        let active_steps = [
            [true, true, true],
            [true, false, true],
            [false, false, true],
            [true, false, true],
        ];
        let logits = [[0.0, 1.0, 2.0], [3.0, 2.0, 1.0], [0.5, 0.5, 0.5]];

        let mut batched_rng = options
            .iter()
            .enumerate()
            .map(|(row, options)| SamplingRng::for_row(options.seed, row))
            .collect::<Vec<_>>();
        let mut batched = vec![Vec::new(); options.len()];
        for active in active_steps {
            for row in 0..options.len() {
                if active[row] {
                    let rng = batched_rng[row].value_for(&options[row]);
                    batched[row].push(sample_categorical(&logits[row], rng));
                }
            }
        }

        for row in 0..options.len() {
            let mut rng = SamplingRng::new(options[row].seed);
            let expected = active_steps
                .iter()
                .filter(|active| active[row])
                .map(|_| sample_categorical(&logits[row], rng.value_for(&options[row])))
                .collect::<Vec<_>>();
            assert_eq!(batched[row], expected);
        }
        assert_eq!(batched[1], vec![0]);
    }

    #[test]
    fn row_assignment_failure_is_reported_for_the_submitted_handle() {
        let tokenizer = test_tokenizer();
        let mut manager = ContinuousBatchManager::new(
            Box::new(RejectAssignDecode),
            &tokenizer,
            Vec::new(),
            None,
            1,
            batch_runtime(),
        )
        .unwrap();
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
        let mut manager = ContinuousBatchManager::new(
            Box::new(AcceptAssignDecode),
            &tokenizer,
            Vec::new(),
            None,
            1,
            batch_runtime(),
        )
        .unwrap();
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
        let mut manager = ContinuousBatchManager::new(
            Box::new(RejectAssignDecode),
            &tokenizer,
            Vec::new(),
            None,
            1,
            batch_runtime(),
        )
        .unwrap();
        let handle = manager
            .submit(GenerateRequest::new(GeneratePrompt::TokenIds(vec![1])))
            .unwrap();

        assert!(manager.cancel_pending(handle));
        assert!(!manager.has_pending_work());
        assert!(manager.poll_admissions().is_empty());
    }

    // ---- Per-row device routing (#1022) ---------------------------------

    use crate::sampling::sample_greedy;
    use onnx_genai_ort::decode::{BatchStepLogits, DeviceRowLogits};
    use std::collections::HashMap;

    /// A device-resident logits buffer stand-in.
    ///
    /// It runs the *host* reference selection so a device-routed row is provably
    /// byte-identical to the host path for the same RNG draw (the routing seam,
    /// not the device kernel, is what this exercises — the CUDA filter math has
    /// its own host-oracle parity in `device_sampler.rs`). Accounting lives in
    /// the manager, so this only has to select tokens and hand back rows.
    struct MockDeviceRowLogits {
        logits: Vec<Vec<f32>>,
        vocab: usize,
    }

    impl DeviceRowLogits for MockDeviceRowLogits {
        fn rows(&self) -> usize {
            self.logits.len()
        }

        fn vocab(&self) -> usize {
            self.vocab
        }

        fn sample_row(
            &mut self,
            row: usize,
            params: &onnx_genai_ort::DeviceSampleParams,
        ) -> onnx_genai_ort::Result<u32> {
            let logits = &self.logits[row];
            let token = if params.greedy {
                sample_greedy(logits)
            } else {
                sample_categorical(logits, params.rng_value)
            };
            Ok(token)
        }

        fn copy_row_to_host(&mut self, row: usize) -> onnx_genai_ort::Result<Vec<f32>> {
            Ok(self.logits[row].clone())
        }
    }

    /// Scripted batched backend that returns fixed per-row logits every step,
    /// either as a device-resident buffer (`device = true`) or as host rows.
    /// A one-token prompt makes the first decode step complete each row's
    /// context, so `step()` immediately produces routed pending tokens.
    struct ScriptedBatchDecode {
        batch: usize,
        vocab: usize,
        max_len: usize,
        row_len: Vec<usize>,
        active: Vec<bool>,
        row_logits: Vec<Vec<f32>>,
        device: bool,
    }

    impl ScriptedBatchDecode {
        fn new(row_logits: Vec<Vec<f32>>, vocab: usize, device: bool) -> Self {
            let batch = row_logits.len();
            Self {
                batch,
                vocab,
                max_len: 4096,
                row_len: vec![0; batch],
                active: vec![false; batch],
                row_logits,
                device,
            }
        }

        fn wrap(&self, buffer: Vec<Vec<f32>>) -> BatchStepLogits {
            if self.device {
                BatchStepLogits::Device(Box::new(MockDeviceRowLogits {
                    logits: buffer,
                    vocab: self.vocab,
                }))
            } else {
                BatchStepLogits::HostRows(buffer.into_iter().map(Some).collect())
            }
        }
    }

    impl<'a> BatchedDecodeSession<'a> for ScriptedBatchDecode {
        fn batch_size(&self) -> usize {
            self.batch
        }

        fn max_len(&self) -> usize {
            self.max_len
        }

        fn row_len(&self, row: usize) -> onnx_genai_ort::Result<usize> {
            Ok(self.row_len[row])
        }

        fn active_rows(&self) -> Vec<usize> {
            (0..self.batch).filter(|&r| self.active[r]).collect()
        }

        fn deactivate_row(&mut self, row: usize) -> onnx_genai_ort::Result<()> {
            self.active[row] = false;
            Ok(())
        }

        fn assign_row(&mut self, row: usize) -> onnx_genai_ort::Result<()> {
            self.active[row] = true;
            self.row_len[row] = 0;
            Ok(())
        }

        fn step_select(
            &mut self,
            _next_token_ids: &[i64],
            _position_ids: &[i64],
            advance_rows: &[bool],
        ) -> onnx_genai_ort::Result<BatchStepLogits> {
            // Physical-row indexed buffer.
            let buffer = (0..self.batch)
                .map(|r| self.row_logits[r].clone())
                .collect();
            for row in 0..self.batch {
                if self.active[row] && advance_rows.get(row).copied().unwrap_or(false) {
                    self.row_len[row] += 1;
                }
            }
            Ok(self.wrap(buffer))
        }

        fn step_active(
            &mut self,
            _next_token_ids: &[i64],
            _position_ids: &[i64],
        ) -> onnx_genai_ort::Result<BatchStepLogits> {
            // Active-row ordered buffer.
            let active = self.active_rows();
            let buffer = active.iter().map(|&r| self.row_logits[r].clone()).collect();
            for &row in &active {
                self.row_len[row] += 1;
            }
            Ok(self.wrap(buffer))
        }
    }

    fn one_hot(vocab: usize, token: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; vocab];
        v[token] = 10.0;
        v
    }

    fn greedy_req(prompt: TokenId) -> GenerateRequest {
        GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![prompt]),
            options: GenerateOptions {
                greedy: true,
                stop_on_eos: false,
                max_new_tokens: 6,
                ..Default::default()
            },
        }
    }

    /// A MIXED batch: some rows are device-portable and some genuinely need the
    /// full host vocabulary. This is the property #1022 locks in — it fails
    /// today (all rows go to the host), fails again under a naive whole-batch
    /// fallback, and passes only under per-row routing: exactly the
    /// host-required rows pay the `vocab x 4` copy while the device-portable
    /// rows copy back only a 4-byte token id.
    #[test]
    fn mixed_batch_moves_only_host_required_rows() {
        let tokenizer = test_tokenizer();
        let vocab = 8usize;
        let row_logits = vec![
            one_hot(vocab, 1),
            one_hot(vocab, 2),
            one_hot(vocab, 3),
            one_hot(vocab, 4),
        ];
        let decode = ScriptedBatchDecode::new(row_logits, vocab, true);
        let mut manager = ContinuousBatchManager::new(
            Box::new(decode),
            &tokenizer,
            Vec::new(),
            None,
            4,
            batch_runtime(),
        )
        .unwrap();

        // Rows 0,1: greedy → device-portable. Row 2: repetition penalty (a
        // history-dependent processor) → host-required. Row 3: top_logprobs →
        // host-required. Submission order fills physical rows 0..3 in order.
        manager.submit(greedy_req(1)).unwrap();
        manager.submit(greedy_req(1)).unwrap();
        let mut penalty = greedy_req(1);
        penalty.options.repetition_penalty = 1.2;
        manager.submit(penalty).unwrap();
        let mut logprobs = greedy_req(1);
        logprobs.options.top_logprobs = Some(2);
        manager.submit(logprobs).unwrap();

        manager.step().unwrap();

        let stats = manager
            .logits_d2h_stats()
            .expect("device-routing manager reports its own D2H accounting");
        let vocab_bytes = vocab as u128 * u128::from(LOGIT_BYTES);
        assert_eq!(stats.rows_host_copied, 2, "only the 2 host rows are copied");
        assert_eq!(
            stats.rows_device_sampled, 2,
            "the 2 greedy rows sample on device"
        );
        assert_eq!(
            stats.bytes,
            2 * vocab_bytes,
            "moved bytes must equal (host-required rows) x vocab x 4"
        );
        assert_ne!(
            stats.bytes,
            4 * vocab_bytes,
            "must NOT copy the whole batch (that is the defect #1022 removes)"
        );
        assert_eq!(stats.token_id_bytes, 2 * u128::from(TOKEN_ID_BYTES));
        assert_eq!(stats.steps, 1);
    }

    /// An all-host batch keeps paying the full per-row copy — routing does not
    /// change behavior when nothing qualifies (the naive whole-batch cost).
    #[test]
    fn all_host_batch_still_copies_every_row() {
        let tokenizer = test_tokenizer();
        let vocab = 8usize;
        let decode =
            ScriptedBatchDecode::new(vec![one_hot(vocab, 1), one_hot(vocab, 2)], vocab, true);
        let mut manager = ContinuousBatchManager::new(
            Box::new(decode),
            &tokenizer,
            Vec::new(),
            None,
            2,
            batch_runtime(),
        )
        .unwrap();
        for _ in 0..2 {
            let mut req = greedy_req(1);
            req.options.repetition_penalty = 1.2; // forces host for every row
            manager.submit(req).unwrap();
        }
        manager.step().unwrap();
        let stats = manager.logits_d2h_stats().unwrap();
        assert_eq!(stats.rows_host_copied, 2);
        assert_eq!(stats.rows_device_sampled, 0);
        assert_eq!(stats.bytes, 2 * vocab as u128 * u128::from(LOGIT_BYTES));
    }

    fn drive_to_completion(manager: &mut ContinuousBatchManager) -> HashMap<usize, Vec<TokenId>> {
        let mut tokens: HashMap<usize, Vec<TokenId>> = HashMap::new();
        let mut guard = 0;
        while manager.has_pending_work() {
            manager.step().unwrap();
            for event in manager.poll() {
                if let ContinuousBatchEvent::Token { handle, token } = event {
                    tokens.entry(handle.id).or_default().push(token.token_id);
                }
            }
            guard += 1;
            assert!(guard < 1000, "runaway decode loop");
        }
        tokens
    }

    /// A slot that retires into an empty queue must still be usable.
    ///
    /// This is the failure the row-liveness rule exists to prevent. The
    /// interpreter reads the loop's predicate *before* the body runs and
    /// preserves an inactive row's carried state afterwards, so a slot that
    /// once published `false` can never come back within the pass. Drive a
    /// short row to completion while the queue is empty, submit a second
    /// request into the freed slot, and keep stepping: the backfilled request
    /// must generate its own tokens, its tokens must reach the *declared* emit
    /// (which `finish_pass` checks row by row when the pass ends), and the pass
    /// must not end while it is still running.
    ///
    /// The teeth are the `drain()` below and the pass count, not the token
    /// counts: the caller's event stream is populated by the *executor*, so a
    /// backfilled request's tokens still arrive there even when they never
    /// reach the declared emit. Deleting either of those two assertions removes
    /// the coverage while leaving a green test, so neither is decoration.
    #[test]
    fn a_slot_freed_into_an_empty_queue_is_reused_by_a_later_submission() {
        let vocab = 8;
        let tokenizer = test_tokenizer();
        let mut manager = ContinuousBatchManager::new(
            Box::new(ScriptedBatchDecode::new(
                vec![one_hot(vocab, 3), one_hot(vocab, 4)],
                vocab,
                false,
            )),
            &tokenizer,
            Vec::new(),
            None,
            2,
            batch_runtime(),
        )
        .unwrap();

        let mut short = greedy_req(1);
        short.options.max_new_tokens = 2;
        let short = manager.submit(short).unwrap();
        let mut long = greedy_req(1);
        long.options.max_new_tokens = 6;
        let long = manager.submit(long).unwrap();

        let mut tokens: HashMap<usize, Vec<TokenId>> = HashMap::new();
        let mut late = None;
        let mut guard = 0;
        while manager.has_pending_work() {
            manager.step().expect("a pass with live work must not fail");
            for event in manager.poll() {
                match event {
                    ContinuousBatchEvent::Token { handle, token } => {
                        tokens.entry(handle.id).or_default().push(token.token_id);
                    }
                    ContinuousBatchEvent::Finished { handle, .. } if handle == short => {
                        // The queue is empty at this moment, so the freed slot
                        // is exactly the one a per-slot `false` would retire.
                        assert!(manager.pending_len() == 0);
                        let mut request = greedy_req(1);
                        request.options.max_new_tokens = 3;
                        late = Some(manager.submit(request).unwrap());
                    }
                    ContinuousBatchEvent::Finished { .. } => {}
                }
            }
            guard += 1;
            assert!(guard < 1000, "runaway decode loop");
        }

        // Ending the pass is what reconciles the declared emit with the
        // executor's commits, row by row. Without it the latch below is
        // invisible: the caller's event stream comes from the executor, so a
        // row whose tokens stopped reaching the *emit* still looks correct.
        manager
            .drain()
            .expect("the declared emit must hold every row's committed tokens");

        let late = late.expect("the short row must have finished");
        // One declared pass. A loop that ended while a row was still generating
        // would restart on the next `step` and produce the right tokens anyway,
        // so the token counts below cannot see it — the pass count can.
        assert_eq!(
            manager.passes, 1,
            "a batch driven to completion runs one declared pass"
        );
        assert_eq!(tokens.get(&short.id).map(Vec::len), Some(2));
        assert_eq!(tokens.get(&long.id).map(Vec::len), Some(6));
        assert_eq!(
            tokens.get(&late.id).map(Vec::len),
            Some(3),
            "the backfilled request must generate its own tokens"
        );
    }

    /// A continuous-batch row stops on a declared end token that is not the
    /// first.
    ///
    /// No CPU fixture can batch — a shared KV buffer needs an execution provider
    /// that reports fixed-capacity present binding — so an end-to-end batched
    /// test on this machine would skip, and a skipped test that reports success
    /// is exactly the evidence this must not be. The scripted decode gives the
    /// manager a real row loop with a known token stream, so the assertion is
    /// about the manager's own stop policy rather than about a backend.
    ///
    /// The unreachable id is declared *first*: a manager that kept only `ids[0]`
    /// would never stop and would run to the budget.
    #[test]
    fn a_continuous_row_stops_on_a_non_first_declared_end_token() {
        let vocab = 8;
        let tokenizer = test_tokenizer();
        let stop = 3u32;
        let unreachable = 7u32;

        let mut manager = ContinuousBatchManager::new(
            Box::new(ScriptedBatchDecode::new(
                vec![one_hot(vocab, 3)],
                vocab,
                false,
            )),
            &tokenizer,
            vec![unreachable, stop],
            None,
            1,
            batch_runtime(),
        )
        .unwrap();

        let mut request = greedy_req(16);
        request.options.stop_on_eos = true;
        manager.submit(request).unwrap();
        let tokens = drive_to_completion(&mut manager);

        assert_eq!(
            tokens.values().cloned().collect::<Vec<_>>(),
            vec![vec![stop]],
            "the row must end at the declared end token rather than at the budget"
        );
    }

    /// Device-routed rows must produce the SAME token stream as the pure-host
    /// path for the same RNG draw — both greedy (argmax) and seeded categorical
    /// rows, alongside a host-required repetition-penalty row that is never
    /// device-routed in either configuration.
    #[test]
    fn device_routing_matches_host_token_stream() {
        let tokenizer = test_tokenizer();
        let vocab = 8usize;
        // Row 0: greedy. Row 1: seeded categorical (device-portable, empty
        // chain). Row 2: repetition penalty (host-required in both configs).
        let build_requests = || {
            let mut greedy = greedy_req(1);
            greedy.options.seed = Some(7);
            let categorical = GenerateRequest {
                prompt: GeneratePrompt::TokenIds(vec![1]),
                options: GenerateOptions {
                    greedy: false,
                    temperature: 1.0,
                    seed: Some(7),
                    stop_on_eos: false,
                    max_new_tokens: 6,
                    ..Default::default()
                },
            };
            let mut penalty = greedy_req(1);
            penalty.options.repetition_penalty = 1.3;
            penalty.options.seed = Some(7);
            vec![greedy, categorical, penalty]
        };
        let logits = || {
            vec![
                one_hot(vocab, 3),
                vec![1.0, 2.0, 0.5, 3.0, 2.5, 0.1, 1.5, 0.2],
                one_hot(vocab, 5),
            ]
        };

        let mut device_manager = ContinuousBatchManager::new(
            Box::new(ScriptedBatchDecode::new(logits(), vocab, true)),
            &tokenizer,
            Vec::new(),
            None,
            3,
            batch_runtime(),
        )
        .unwrap();
        let mut host_manager = ContinuousBatchManager::new(
            Box::new(ScriptedBatchDecode::new(logits(), vocab, false)),
            &tokenizer,
            Vec::new(),
            None,
            3,
            batch_runtime(),
        )
        .unwrap();
        for req in build_requests() {
            device_manager.submit(req).unwrap();
        }
        for req in build_requests() {
            host_manager.submit(req).unwrap();
        }

        let device_tokens = drive_to_completion(&mut device_manager);
        let host_tokens = drive_to_completion(&mut host_manager);

        assert_eq!(
            device_tokens, host_tokens,
            "per-row device routing must not change the token stream vs the host path"
        );
        // The device run must have actually routed the two portable rows on the
        // device (otherwise this would trivially pass by taking the host path).
        let stats = device_manager.logits_d2h_stats().unwrap();
        assert!(
            stats.rows_device_sampled > 0,
            "expected device sampling to occur; got {stats:?}"
        );
        assert!(host_manager.logits_d2h_stats().is_none() || !host_manager.used_device_routing);
    }

    // ---- Co-decoded batch occupancy -------------------------------------

    /// Drive to completion recording, per handle, the tokens it received *and*
    /// the order they arrived in, so a routing mix-up is detectable.
    fn drive_recording_rows(
        manager: &mut ContinuousBatchManager,
        mut on_step: impl FnMut(&mut ContinuousBatchManager),
    ) -> HashMap<usize, Vec<TokenId>> {
        let mut tokens: HashMap<usize, Vec<TokenId>> = HashMap::new();
        let mut guard = 0;
        while manager.has_pending_work() {
            manager.step().unwrap();
            for event in manager.poll() {
                if let ContinuousBatchEvent::Token { handle, token } = event {
                    tokens.entry(handle.id).or_default().push(token.token_id);
                }
            }
            on_step(manager);
            guard += 1;
            assert!(guard < 1000, "runaway decode loop");
        }
        tokens
    }

    fn capped_greedy_req(prompt: TokenId, max_new_tokens: usize) -> GenerateRequest {
        let mut req = greedy_req(prompt);
        req.options.max_new_tokens = max_new_tokens;
        req
    }

    /// The whole point of the counter: a batch of three concurrent requests must
    /// report forwards that carried three rows, not three forwards of one row.
    #[test]
    fn concurrent_rows_are_reported_as_co_decoded_in_one_forward() {
        let tokenizer = test_tokenizer();
        let vocab = 8usize;
        let decode = ScriptedBatchDecode::new(
            vec![one_hot(vocab, 1), one_hot(vocab, 2), one_hot(vocab, 3)],
            vocab,
            false,
        );
        let mut manager = ContinuousBatchManager::new(
            Box::new(decode),
            &tokenizer,
            Vec::new(),
            None,
            3,
            batch_runtime(),
        )
        .unwrap();
        for _ in 0..3 {
            manager.submit(capped_greedy_req(1, 4)).unwrap();
        }

        let tokens = drive_recording_rows(&mut manager, |_| {});
        let occupancy = manager.occupancy();

        assert_eq!(occupancy.max_batch, 3);
        assert_eq!(
            occupancy.max_rows_in_step, 3,
            "three concurrent requests must share a forward"
        );
        assert!(
            occupancy.mean_rows_per_step().unwrap() > 1.0,
            "mean rows per forward proves the batch was not serialized: {occupancy:?}"
        );
        assert_eq!(occupancy.peak_utilization(), 1.0);
        // Each row's scripted logits are one-hot on a different token, so
        // identical streams would mean the demux crossed rows.
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[&0], vec![1; 4]);
        assert_eq!(tokens[&1], vec![2; 4]);
        assert_eq!(tokens[&2], vec![3; 4]);
    }

    /// One request on an eight-row manager must report occupancy 1, so the
    /// counter cannot be mistaken for "the configured maximum".
    #[test]
    fn a_lone_request_reports_occupancy_of_one() {
        let tokenizer = test_tokenizer();
        let vocab = 8usize;
        let decode = ScriptedBatchDecode::new(vec![one_hot(vocab, 1); 8], vocab, false);
        let mut manager = ContinuousBatchManager::new(
            Box::new(decode),
            &tokenizer,
            Vec::new(),
            None,
            8,
            batch_runtime(),
        )
        .unwrap();
        manager.submit(capped_greedy_req(1, 3)).unwrap();

        drive_recording_rows(&mut manager, |_| {});
        let occupancy = manager.occupancy();

        assert_eq!(occupancy.max_rows_in_step, 1);
        assert_eq!(occupancy.mean_rows_per_step(), Some(1.0));
        assert_eq!(occupancy.peak_utilization(), 0.125);
    }

    /// Ragged rows: when a short row retires early the remaining forwards carry
    /// fewer rows, so the mean lands strictly between the peak and 1.
    #[test]
    fn ragged_rows_shrink_the_batch_as_they_retire() {
        let tokenizer = test_tokenizer();
        let vocab = 8usize;
        let decode = ScriptedBatchDecode::new(
            vec![one_hot(vocab, 1), one_hot(vocab, 2), one_hot(vocab, 3)],
            vocab,
            false,
        );
        let mut manager = ContinuousBatchManager::new(
            Box::new(decode),
            &tokenizer,
            Vec::new(),
            None,
            3,
            batch_runtime(),
        )
        .unwrap();
        manager.submit(capped_greedy_req(1, 1)).unwrap();
        manager.submit(capped_greedy_req(1, 2)).unwrap();
        manager.submit(capped_greedy_req(1, 9)).unwrap();

        let tokens = drive_recording_rows(&mut manager, |_| {});
        let occupancy = manager.occupancy();

        assert_eq!(tokens[&0].len(), 1, "row 0 stops after its own budget");
        assert_eq!(tokens[&1].len(), 2);
        assert_eq!(tokens[&2].len(), 9);
        assert_eq!(tokens[&2], vec![3; 9], "the long row keeps its own token");
        assert_eq!(occupancy.max_rows_in_step, 3);
        let mean = occupancy.mean_rows_per_step().unwrap();
        assert!(
            mean > 1.0 && mean < 3.0,
            "a ragged batch decodes fewer rows once short rows retire: {occupancy:?}"
        );
    }

    /// A request queued behind a full batch must be admitted into the slot a
    /// retired row frees, and must then receive *that physical row's* tokens.
    #[test]
    fn a_queued_request_reuses_the_slot_a_retired_row_freed() {
        let tokenizer = test_tokenizer();
        let vocab = 8usize;
        // Physical row 0 emits token 1, row 1 emits token 2.
        let decode =
            ScriptedBatchDecode::new(vec![one_hot(vocab, 1), one_hot(vocab, 2)], vocab, false);
        let mut manager = ContinuousBatchManager::new(
            Box::new(decode),
            &tokenizer,
            Vec::new(),
            None,
            2,
            batch_runtime(),
        )
        .unwrap();
        let short = manager.submit(capped_greedy_req(1, 1)).unwrap();
        let long = manager.submit(capped_greedy_req(1, 6)).unwrap();
        // Third request cannot be assigned: both physical rows are taken.
        let late = manager.submit(capped_greedy_req(1, 2)).unwrap();
        manager.admit_pending();
        assert_eq!(manager.active_len(), 2);
        assert_eq!(manager.pending_len(), 1, "the third request must wait");

        let mut admitted_late_while_long_ran = false;
        let tokens = drive_recording_rows(&mut manager, |manager| {
            if manager.pending_len() == 0 && manager.active_len() == 2 {
                admitted_late_while_long_ran = true;
            }
        });
        let occupancy = manager.occupancy();

        assert!(
            admitted_late_while_long_ran,
            "the queued request must enter the freed slot while the long row still decodes"
        );
        assert_eq!(tokens[&short.id], vec![1]);
        assert_eq!(tokens[&long.id], vec![2; 6]);
        assert_eq!(
            tokens[&late.id],
            vec![1; 2],
            "the late request must take over physical row 0 and read its logits"
        );
        assert_eq!(occupancy.max_rows_in_step, 2);
        assert!(occupancy.mean_rows_per_step().unwrap() > 1.0);
    }

    /// Before any forward there is nothing honest to report.
    #[test]
    fn occupancy_is_empty_before_the_first_forward() {
        let tokenizer = test_tokenizer();
        let manager = ContinuousBatchManager::new(
            Box::new(AcceptAssignDecode),
            &tokenizer,
            Vec::new(),
            None,
            1,
            batch_runtime(),
        )
        .unwrap();
        let occupancy = manager.occupancy();
        assert_eq!(occupancy.steps, 0);
        assert_eq!(occupancy.mean_rows_per_step(), None);
        assert_eq!(occupancy.peak_utilization(), 0.0);
        assert_eq!(occupancy.max_batch, 1);
    }
}
