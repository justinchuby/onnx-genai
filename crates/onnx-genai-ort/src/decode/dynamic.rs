use super::*;
use std::collections::HashSet;

/// Prompt and prefill runs use CUDA-graph annotation id `-1` (no capture) so
/// only the fixed-shape decode step is captured and replayed. Each
/// [`DecodeSession`] that enables capture claims a process-unique positive id
/// (see [`next_capture_graph_id`]) so that reusing the underlying ORT session
/// for a new generation never re-captures a different graph under an id ORT
/// already holds — which corrupts ORT's per-id CUDA-graph bookkeeping.
static NEXT_CAPTURE_GRAPH_ID: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(1);

#[cfg(feature = "cuda")]
fn device_argmax_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ONNX_GENAI_DEVICE_ARGMAX")
            .map(|value| value != "0")
            .unwrap_or(true)
    })
}

fn next_capture_graph_id() -> i32 {
    // Ids must be unique across concurrently-live sessions and strictly positive
    // so they never collide with the `-1` no-capture sentinel. Masking off the
    // sign bit keeps them positive and unique within each 2^31 cycle; the lone
    // zero per cycle is remapped. A wrap would only reuse an id after 2^31
    // generations, by which point the prior holder is long dropped.
    let raw = NEXT_CAPTURE_GRAPH_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    match raw & i32::MAX {
        0 => i32::MAX,
        id => id,
    }
}

enum CapturedStepError {
    PreRun(OrtError),
    RunInvoked(OrtError),
}

/// Map a phased run failure onto the captured-step retry classification.
///
/// A [`RunPhaseError::Setup`] failure happens before the ORT `Run` call, so the
/// model has not advanced and the step is safe to replay through the standard
/// path — classified [`CapturedStepError::PreRun`]. A [`RunPhaseError::Invoked`]
/// failure happens at or after the model invocation, which may have advanced KV
/// state, so it must propagate without a replay — [`CapturedStepError::RunInvoked`].
fn classify_run_phase(err: RunPhaseError) -> CapturedStepError {
    match err {
        RunPhaseError::Setup(err) => CapturedStepError::PreRun(err),
        RunPhaseError::Invoked(err) => CapturedStepError::RunInvoked(err),
    }
}

fn retry_pre_run_captured_failure<T>(
    result: std::result::Result<T, CapturedStepError>,
    retry: impl FnOnce(OrtError) -> Result<T>,
) -> Result<T> {
    match result {
        Ok(value) => Ok(value),
        Err(CapturedStepError::PreRun(err)) => retry(err),
        Err(CapturedStepError::RunInvoked(err)) => Err(err),
    }
}

/// A stateful IoBinding decode runner that keeps KV OrtValues inside ORT.
pub struct DecodeSession<'a> {
    session: &'a Session,
    binding: IoBinding,
    kv_pairs: Vec<KvPair>,
    token_input: String,
    attention_mask_input: Option<String>,
    position_ids_input: Option<String>,
    logits_output: String,
    current_kv: HashMap<String, Arc<Value>>,
    current_len: usize,
    mode: DecodeKvMode,
    /// Owned device allocator that backs the shared-buffer KV `Value`s in
    /// `current_kv`. OrtValues created through an allocator free their memory
    /// via that allocator on release, so it MUST outlive the `Value`s. This
    /// field is declared after `current_kv` so Rust drops the KV `Value`s first
    /// and releases this allocator afterwards; releasing it earlier caused a
    /// use-after-free SIGSEGV at session close.
    kv_allocator: Option<crate::Allocator>,
    /// Static-shape captured-decode state, populated lazily on the first
    /// single-token step when the session has CUDA graph capture enabled.
    /// Holds the persistent, fixed-address I/O buffers a captured graph replays
    /// against: `input_ids [1,1]`, `position_ids [1,1]`, a max-length
    /// `attention_mask [1, max_len]`, and the `logits [1,1,vocab]` output.
    capture: Option<CaptureState>,
    /// Hard cap on logical context length (the model-declared shared-buffer
    /// `max_length`). The KV buffers may be sized *below* this via bucketing
    /// (see [`kv_capacity_bucket`] and [`DecodeSession::kv_capacity`]); this is
    /// only the ceiling growth can reach. `None` outside shared-buffer mode.
    max_length: Option<usize>,
    /// Current allocated capacity (sequence-axis length) of the shared KV
    /// buffers, i.e. the active bucket. Starts small and grows by re-capture (see
    /// [`DecodeSession::ensure_kv_capacity`]).
    ///
    /// WHY this exists: onnxruntime-genai's CUDA captured decode scales its
    /// per-step attention cost with the *actual* sequence length, but allocating
    /// the shared KV buffers at the full `max_length` made onnx-genai's per-token
    /// cost scale with that declared capacity instead — a large-`max_length`
    /// model (e.g. Mistral-7B's 32768) paid an ~O(capacity) attention tax every
    /// step regardless of how few tokens were actually generated (measured ~30%
    /// slower than og: 40.6 vs 51.9 tok/s on an RTX 4060; temporarily shrinking
    /// the model's `max_length` to 4096 closed the gap). Bucketing the KV
    /// buffers to ~O(actual length) removes that tax while re-capturing only
    /// O(log length) times. `0` outside shared-buffer mode.
    kv_capacity: usize,
    /// Whether the persistent captured-decode I/O is currently bound. Captured
    /// graphs require stable bindings across replays, so we bind the persistent
    /// buffers once and only rebind after a non-captured step clears them.
    capture_bound: bool,
    /// Process-unique CUDA-graph annotation id claimed lazily when this session
    /// first captures its decode graph. `None` until the first captured step.
    capture_graph_id: Option<i32>,
    /// Set when a captured decode step fails and we fall back to the standard
    /// decode path for the rest of this generation. Once set, the captured fast
    /// path is skipped even though the underlying session still reports
    /// `graph_capture() == true`, so graceful degradation persists per decode
    /// loop without mutating the shared session.
    graph_capture_disabled: bool,
    /// Backend-provided device-side token selection for the captured path.
    /// `Some` only when the captured `logits` buffer is device-resident (see
    /// [`CaptureState::logits_on_device`]); it keeps the full vocabulary on the
    /// device and reduces it there, so no per-token full-vocab host copy occurs.
    #[cfg(feature = "cuda")]
    device_sampler: Option<Box<dyn DeviceSampler>>,
}

/// Outcome of a single decode step: either the logits Value (default path) or,
/// on the greedy fast path, only the argmax token id read directly from the
/// persistent logits buffer without materializing the full vocabulary on the
/// host.
enum StepLogits {
    Value(Value),
    Token(u32),
}

/// Persistent I/O buffers for the static-shape captured decode graph.
struct CaptureState {
    input_ids: Value,
    position_ids: Value,
    attention_mask: Value,
    logits: Value,
    mask_len: usize,
    /// Number of leading `attention_mask` entries currently set to 1. The valid
    /// region only grows within a generation, so each step fills just the delta
    /// `[mask_valid_len, valid_len)` instead of rewriting the whole prefix,
    /// keeping the captured-decode step O(1) rather than O(context). Reset to 0
    /// by [`DecodeSession::reset_captured_mask`] on rewind/reset.
    mask_valid_len: usize,
    /// Vocabulary width of the `logits [1, 1, vocab]` output.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    vocab: usize,
    /// Whether `logits` is CUDA device-resident (allocated on the session's CUDA
    /// allocator). When set, the greedy fast path reduces it with an on-device
    /// argmax kernel instead of copying the full vocabulary to the host.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    logits_on_device: bool,
    /// Whether `input_ids` / `position_ids` / `attention_mask` are CUDA
    /// device-resident (allocated on the session's CUDA allocator). When set,
    /// each step refreshes them with a host->device copy in place and the
    /// captured graph reads the updated device bytes on replay, so the IoBinding
    /// set is bound **once** and never cleared/re-bound per token. When unset
    /// (host inputs / CPU-argmax path) the loop falls back to the per-step
    /// `clear_inputs` + re-bind that a captured graph needs to observe fresh CPU
    /// inputs (see the `step_captured` binding comment and issue
    /// microsoft/onnxruntime#29782).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    inputs_on_device: bool,
    /// CUDA device id backing the device-resident capture buffers. Used to pin
    /// the per-step host->device input copies to the correct device.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    device_id: i32,
}

impl CaptureState {
    /// Refresh the per-step dynamic inputs in their persistent **host** buffers.
    ///
    /// `input_ids` / `position_ids` are overwritten; the attention mask's valid
    /// region only grows within a generation (rewind/reset clear it) and prior
    /// entries are already 1, so only the newly-valid tail is filled — typically
    /// a single element — keeping this step O(1) in context rather than
    /// rewriting the whole prefix.
    fn write_step_inputs_host(
        &mut self,
        token: i64,
        position: i64,
        valid_len: usize,
    ) -> Result<()> {
        self.input_ids.write_i64_prefix(&[token])?;
        self.position_ids.write_i64_prefix(&[position])?;
        if valid_len > self.mask_valid_len {
            self.attention_mask.fill_i64_range(
                self.mask_valid_len,
                valid_len - self.mask_valid_len,
                1,
            )?;
        } else if valid_len < self.mask_valid_len {
            // Defensive: a shrink without an intervening reset — clear the tail
            // that is no longer valid so it does not leak into this step.
            self.attention_mask
                .fill_i64_range(valid_len, self.mask_valid_len - valid_len, 0)?;
        }
        self.mask_valid_len = valid_len;
        Ok(())
    }

    /// Refresh the per-step dynamic inputs in their persistent **CUDA
    /// device-resident** buffers with in-place host->device copies. The captured
    /// graph reads these device buffers directly, so the update is observed on
    /// the next replay without any re-bind (see `step_captured`). Mirrors the
    /// O(1) mask-tail growth of [`write_step_inputs_host`](Self::write_step_inputs_host).
    ///
    /// The copies are issued on cudart's default stream and, for pageable host
    /// sources, may return before the DMA to device memory has completed (CUDA
    /// API synchronization contract). ORT replays the captured graph on a
    /// `cudaStreamNonBlocking` stream that does not serialize against the default
    /// stream, so this synchronizes the device before returning to guarantee the
    /// fresh input bytes are globally visible before the replay reads them (RAW
    /// ordering — microsoft/onnxruntime#29782). At this point the device is
    /// otherwise idle (the prior step's device sampler already synchronized it),
    /// so the sync only drains these few tiny transfers.
    #[cfg(feature = "cuda")]
    fn write_step_inputs_device(
        &mut self,
        token: i64,
        position: i64,
        valid_len: usize,
    ) -> Result<()> {
        let device_id = self.device_id;
        self.input_ids
            .write_i64_prefix_device(&[token], device_id)?;
        self.position_ids
            .write_i64_prefix_device(&[position], device_id)?;
        if valid_len > self.mask_valid_len {
            self.attention_mask.fill_i64_range_device(
                self.mask_valid_len,
                valid_len - self.mask_valid_len,
                1,
                device_id,
            )?;
        } else if valid_len < self.mask_valid_len {
            self.attention_mask.fill_i64_range_device(
                valid_len,
                self.mask_valid_len - valid_len,
                0,
                device_id,
            )?;
        }
        self.mask_valid_len = valid_len;
        // Order the host->device input copies above before ORT's non-blocking
        // captured-graph replay reads these buffers (see the doc comment).
        let _sync_span = crate::prof_span!("ort.write_step_inputs.device_sync");
        let _guard = crate::cuda_rt::DeviceGuard::set(device_id)?;
        crate::cuda_rt::device_synchronize()?;
        Ok(())
    }

    /// Zero the currently-valid prefix of the attention mask (device- or
    /// host-resident) and reset the valid-length counter. Only the previously
    /// valid prefix is cleared — the rest is already zero.
    fn clear_valid_mask(&mut self) -> Result<()> {
        if self.mask_valid_len == 0 {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if self.inputs_on_device {
            self.attention_mask
                .fill_i64_range_device(0, self.mask_valid_len, 0, self.device_id)?;
            self.mask_valid_len = 0;
            return Ok(());
        }
        self.attention_mask
            .fill_i64_range(0, self.mask_valid_len, 0)?;
        self.mask_valid_len = 0;
        Ok(())
    }
}

impl Drop for DecodeSession<'_> {
    fn drop(&mut self) {
        // If this session captured a decode graph, release it now — while this
        // session's fixed-address I/O buffers are still alive (fields are
        // dropped after this method returns). The captured graph references
        // those buffers; leaving it registered on the shared ORT session would
        // let a later release clean up a graph whose buffers were already freed,
        // corrupting the heap.
        if let Some(graph_id) = self.capture_graph_id {
            let _ = self.session.release_captured_graph(graph_id);
        }
    }
}

impl<'a> DecodeSession<'a> {
    /// Create a decode session using unambiguous tensor shapes.
    ///
    /// Models with shape-ambiguous roles must use [`Self::new_with_io`].
    pub fn new(session: &'a Session, options: DecodeSessionOptions) -> Result<Self> {
        Self::new_with_io(session, options, None)
    }

    /// Create a decode session using declarative graph-port roles when present.
    pub fn new_with_io(
        session: &'a Session,
        options: DecodeSessionOptions,
        io: Option<&onnx_genai_metadata::ModelIoSpec>,
    ) -> Result<Self> {
        let kv_pairs = infer_kv_pairs(session, io)?;
        let excluded = kv_pairs
            .iter()
            .flat_map(|pair| [pair.past.as_str(), pair.present.as_str()])
            .chain(
                [
                    io.and_then(|io| io.inputs_embeds_input.as_deref()),
                    io.and_then(|io| io.attention_mask_input.as_deref()),
                    io.and_then(|io| io.position_ids_input.as_deref()),
                    io.and_then(|io| io.hidden_output.as_deref()),
                ]
                .into_iter()
                .flatten(),
            )
            .collect::<HashSet<_>>();
        let resolve_input = |declared, role, structural: fn(&TensorInfo) -> bool| {
            crate::io_roles::resolve_port(session.inputs(), declared, role, |tensor| {
                !excluded.contains(tensor.name.as_str()) && structural(tensor)
            })
            .map_err(OrtError::InvalidArgument)
            .map(|port| port.map(|port| port.name))
        };
        let never = |_: &TensorInfo| false;
        let token_input = resolve_input(
            io.and_then(|io| io.token_input.as_deref()),
            "model.io.token_input",
            crate::io_roles::is_rank_one_or_two_sequence,
        )?
        .ok_or_else(|| {
            OrtError::InvalidArgument(
                "cannot resolve token input from tensor shape; declare model.io.token_input".into(),
            )
        })?;
        let attention_mask_input = resolve_input(
            io.and_then(|io| io.attention_mask_input.as_deref()),
            "model.io.attention_mask_input",
            never,
        )?;
        let position_ids_input = resolve_input(
            io.and_then(|io| io.position_ids_input.as_deref()),
            "model.io.position_ids_input",
            never,
        )?;
        let logits_output = crate::io_roles::resolve_port(
            session.outputs(),
            io.and_then(|io| io.logits_output.as_deref()),
            "model.io.logits_output",
            |tensor| {
                !excluded.contains(tensor.name.as_str())
                    && crate::io_roles::is_rank_one_to_three_output(tensor)
            },
        )
        .map_err(OrtError::InvalidArgument)?
        .map(|port| port.name)
        .ok_or_else(|| {
            OrtError::InvalidArgument(
                "cannot resolve logits output from tensor shape; declare model.io.logits_output"
                    .into(),
            )
        })?;
        let assigned_inputs = [
            Some(token_input.as_str()),
            attention_mask_input.as_deref(),
            position_ids_input.as_deref(),
            io.and_then(|io| io.inputs_embeds_input.as_deref()),
        ]
        .into_iter()
        .flatten()
        .chain(kv_pairs.iter().map(|pair| pair.past.as_str()))
        .collect::<HashSet<_>>();
        let assigned_outputs = [
            Some(logits_output.as_str()),
            io.and_then(|io| io.hidden_output.as_deref()),
        ]
        .into_iter()
        .flatten()
        .chain(kv_pairs.iter().map(|pair| pair.present.as_str()))
        .collect::<HashSet<_>>();
        let unassigned_state_inputs = session
            .inputs()
            .iter()
            .filter(|tensor| {
                tensor.shape.len() >= 3 && !assigned_inputs.contains(tensor.name.as_str())
            })
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();
        let unassigned_state_outputs = session
            .outputs()
            .iter()
            .filter(|tensor| {
                tensor.shape.len() >= 3 && !assigned_outputs.contains(tensor.name.as_str())
            })
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();
        if !unassigned_state_inputs.is_empty() || !unassigned_state_outputs.is_empty() {
            return Err(OrtError::InvalidArgument(format!(
                "cannot resolve decoder state from tensor shapes (inputs: {unassigned_state_inputs:?}, outputs: {unassigned_state_outputs:?}); declare model.io.kv_inputs and model.io.kv_outputs"
            )));
        }
        let share_buffer = options.past_present_share_buffer.unwrap_or(false);
        let mode = if share_buffer {
            DecodeKvMode::SharedBuffer
        } else {
            DecodeKvMode::ZeroCopyRebind
        };
        let mut this = Self {
            session,
            binding: IoBinding::new(session)?,
            kv_pairs,
            token_input,
            attention_mask_input,
            position_ids_input,
            logits_output,
            current_kv: HashMap::new(),
            current_len: 0,
            mode,
            kv_allocator: None,
            capture: None,
            max_length: None,
            kv_capacity: 0,
            capture_bound: false,
            capture_graph_id: None,
            graph_capture_disabled: false,
            #[cfg(feature = "cuda")]
            device_sampler: None,
        };
        if mode == DecodeKvMode::SharedBuffer {
            let max_length = options.max_length.ok_or_else(|| {
                OrtError::InvalidArgument(
                    "DecodeSession shared-buffer mode requires max_length".into(),
                )
            })?;
            // Allocate the shared KV buffers at a small starting bucket rather
            // than the full `max_length`. The prompt/prefill and decode steps
            // grow the buckets on demand (see `ensure_kv_capacity`), so a model
            // that declares a huge context but generates few tokens never pays
            // the O(max_length) captured-decode attention tax up front.
            let initial_capacity = kv_capacity_bucket(0, max_length);
            this.allocate_shared_buffers(options.batch_size, initial_capacity)?;
            this.max_length = Some(max_length);
            this.kv_capacity = initial_capacity;
        }
        Ok(this)
    }

    /// The selected KV binding strategy.
    pub fn mode(&self) -> DecodeKvMode {
        self.mode
    }

    /// Current logical KV length in tokens.
    pub fn past_len(&self) -> usize {
        self.current_len
    }

    /// Run one incremental decode step and return the logits OrtValue.
    ///
    /// `attention_mask` is the full `[batch, past + new]` mask flattened row-major,
    /// while `position_ids` covers only `new_input_ids`.
    pub fn step(
        &mut self,
        new_input_ids: &[i64],
        attention_mask: &[i64],
        position_ids: &[i64],
    ) -> Result<Value> {
        match self.step_dispatch(new_input_ids, attention_mask, position_ids, None)? {
            StepLogits::Value(logits) => Ok(logits),
            // `params` is None, so the captured path returns a Value.
            StepLogits::Token(_) => {
                Err(OrtError::InvalidArgument("step produced a token id".into()))
            }
        }
    }

    /// Run one incremental decode step and return only the greedy (argmax)
    /// token id of the final logits row.
    ///
    /// This is the greedy decode fast path: on the captured single-token path
    /// the argmax is computed directly on the persistent logits buffer, so the
    /// full ~150K-entry vocabulary never leaves the tensor (no owned clone, no
    /// host `Vec`, no separate CPU argmax scan). Callers that apply logit
    /// processors or non-greedy sampling must use [`Self::step`] instead.
    pub fn step_argmax(
        &mut self,
        new_input_ids: &[i64],
        attention_mask: &[i64],
        position_ids: &[i64],
    ) -> Result<u32> {
        match self.step_dispatch(
            new_input_ids,
            attention_mask,
            position_ids,
            Some(&DeviceSampleParams::greedy()),
        )? {
            StepLogits::Token(token) => Ok(token),
            // The standard (non-captured) path returns a Value; reduce it here.
            StepLogits::Value(logits) => {
                let _sampling_span = crate::prof_span!("ort.sampling");
                logits.argmax_last_row()
            }
        }
    }

    /// Run one incremental decode step and return the token id selected by the
    /// device-portable sampling pipeline described by `params`.
    ///
    /// This is the sampling analogue of [`Self::step_argmax`]: when `params` is
    /// greedy it is exactly the argmax fast path. Non-greedy device sampling
    /// (temperature/top-k/top-p/min-p + categorical draw) runs entirely on the
    /// device logits pointer via the [`DeviceSampler`]. If the logits are not on
    /// the device (CPU path), the non-greedy request returns an error so the
    /// engine can fall back to its host sampler — this method never copies the
    /// full vocabulary to the host to sample.
    pub fn step_sampled(
        &mut self,
        new_input_ids: &[i64],
        attention_mask: &[i64],
        position_ids: &[i64],
        params: &DeviceSampleParams,
    ) -> Result<u32> {
        if params.greedy {
            return self.step_argmax(new_input_ids, attention_mask, position_ids);
        }
        match self.step_dispatch(new_input_ids, attention_mask, position_ids, Some(params))? {
            StepLogits::Token(token) => Ok(token),
            // The captured device path yields a token; a Value means the logits
            // stayed on the host (non-captured / CPU path), where this method
            // does not sample. Signal the engine to fall back to its host sampler.
            StepLogits::Value(_) => Err(OrtError::InvalidArgument(
                "device sampled (non-greedy) decode requires on-device logits; \
                 logits are on the host"
                    .into(),
            )),
        }
    }

    /// Whether a subsequent single-token [`Self::step_sampled`] call will select
    /// the token entirely on the device (returning a token id) rather than
    /// producing host logits.
    ///
    /// The non-greedy device sample path requires *both* the captured decode
    /// fast path (`SharedBuffer` KV, graph capture enabled) *and* on-device
    /// logits (CUDA, device argmax enabled, a device KV allocator). When either
    /// is missing, `step_sampled` runs the standard step — which advances KV
    /// state — and then reports that logits are on the host. Callers must not
    /// invoke the sampled path in that case: the standard step's side effects
    /// would be replayed by the host fallback, double-advancing the KV cache.
    /// This predicate lets the caller route straight to host sampling instead.
    pub fn will_sample_on_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            let captured = self.mode == DecodeKvMode::SharedBuffer
                && self.session.graph_capture()
                && !self.graph_capture_disabled;
            let device_logits = device_argmax_enabled()
                && self.session.cuda_device_id().is_some()
                && self.kv_allocator.is_some();
            captured && device_logits
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    fn step_dispatch(
        &mut self,
        new_input_ids: &[i64],
        attention_mask: &[i64],
        position_ids: &[i64],
        params: Option<&DeviceSampleParams>,
    ) -> Result<StepLogits> {
        if new_input_ids.is_empty() {
            return Err(OrtError::InvalidArgument(
                "decode step requires at least one input id".into(),
            ));
        }
        if position_ids.len() != new_input_ids.len() {
            return Err(OrtError::InvalidArgument(
                "position_ids length must match input_ids length".into(),
            ));
        }

        // Grow the shared KV buckets before binding if this step's present KV
        // (positions `[0, current_len + seq_len)`) would overrun the current
        // capacity. This covers both the prompt/prefill step (large `seq_len`)
        // and ordinary decode steps that cross a bucket boundary. Growing here
        // — before either the captured fast path or the standard path binds the
        // buffers — keeps the fixed-address captured graph valid at its bound
        // capacity and forces a one-time re-capture at the new size.
        if self.mode == DecodeKvMode::SharedBuffer {
            let required = self
                .current_len
                .checked_add(new_input_ids.len())
                .ok_or_else(|| OrtError::InvalidArgument("decode length overflow".into()))?;
            self.ensure_kv_capacity(required)?;
        }

        // Static-shape captured decode fast path: once the prompt has been
        // consumed, every decode step feeds one token with fixed-shape inputs
        // and fixed-address KV buffers, so a single CUDA graph can be captured
        // and replayed to eliminate per-kernel launch overhead.
        if self.mode == DecodeKvMode::SharedBuffer
            && self.session.graph_capture()
            && !self.graph_capture_disabled
            && new_input_ids.len() == 1
            && self.current_len > 0
        {
            let captured =
                self.step_captured(new_input_ids[0], attention_mask, position_ids[0], params);
            return retry_pre_run_captured_failure(captured, |err| {
                // Failures before ORT starts the graph run cannot have advanced
                // KV state, so disable capture and retry this step through the
                // standard path. Once the graph run is invoked, errors propagate
                // without retry because KV may already have been mutated.
                {
                    tracing::warn!(
                        error = %err,
                        "CUDA graph decode setup failed; disabling graph capture and \
                         falling back to the standard decode path for the rest of this session"
                    );
                    self.graph_capture_disabled = true;
                    self.capture = None;
                }
                self.step_standard(new_input_ids, attention_mask, position_ids)
                    .map(StepLogits::Value)
            });
        }

        self.step_standard(new_input_ids, attention_mask, position_ids)
            .map(StepLogits::Value)
    }

    fn step_standard(
        &mut self,
        new_input_ids: &[i64],
        attention_mask: &[i64],
        position_ids: &[i64],
    ) -> Result<Value> {
        let seq_len = i64::try_from(new_input_ids.len())
            .map_err(|_| OrtError::InvalidArgument("input length exceeds i64".into()))?;
        let total_len = i64::try_from(attention_mask.len())
            .map_err(|_| OrtError::InvalidArgument("attention mask length exceeds i64".into()))?;

        let input_ids = Value::from_slice_i64(new_input_ids, &[1, seq_len])?;
        let attention_mask = Value::from_slice_i64(attention_mask, &[1, total_len])?;
        let position_ids = Value::from_slice_i64(position_ids, &[1, seq_len])?;

        let bind_span = crate::prof_span!("ort.bind_inputs");
        self.binding.clear()?;
        // This step re-binds fresh per-step Values, so any persistent captured
        // binding is now stale and must be re-established before the next
        // captured step.
        self.capture_bound = false;
        self.bind_standard_inputs(&input_ids, &attention_mask, &position_ids)?;
        self.bind_kv_inputs()?;
        let mut borrowed_outputs = Vec::new();
        for output in self.session.output_names() {
            if self.mode == DecodeKvMode::SharedBuffer
                && let Some(pair) = self.kv_pairs.iter().find(|pair| pair.present == *output)
            {
                let value = self.current_kv.get(&pair.past).ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "missing shared KV buffer for '{}'",
                        pair.past
                    ))
                })?;
                borrowed_outputs.push(value.raw_ptr_addr());
                self.binding.bind_output(output, value)?;
            } else {
                self.binding
                    .bind_output_to_device(output, &MemoryInfo::cpu()?)?;
            }
        }
        drop(bind_span);

        {
            let _run_span = crate::prof_span!("ort.session_run");
            // When graph capture is enabled, prompt/prefill runs use annotation
            // -1 so ORT executes them normally instead of capturing them as the
            // (differently-shaped) decode graph.
            if self.session.graph_capture() && !self.graph_capture_disabled {
                self.session.run_with_binding_graph(&self.binding, -1)?;
            } else {
                self.session.run_with_binding(&self.binding)?;
            }
        }
        let _extract_span = crate::prof_span!("ort.extract_outputs");
        let mut logits = None;
        if self.mode == DecodeKvMode::SharedBuffer {
            let outputs = self.binding.output_values_or_borrowed(&borrowed_outputs)?;
            for (name, value) in self.session.output_names().iter().zip(outputs) {
                if name == &self.logits_output {
                    logits = value;
                    break;
                }
            }
        } else {
            let outputs = self.binding.output_values()?;
            self.rotate_outputs(outputs, &mut logits)?;
        }
        self.current_len = self
            .current_len
            .checked_add(new_input_ids.len())
            .ok_or_else(|| OrtError::InvalidArgument("decode length overflow".into()))?;
        logits.ok_or_else(|| OrtError::InvalidArgument("model did not produce logits".into()))
    }

    /// Single-token decode step replayed through a captured CUDA graph.
    ///
    /// All inputs are bound to persistent, fixed-address buffers whose shapes
    /// never change across steps: `input_ids [1,1]`, `position_ids [1,1]`, and a
    /// full-capacity `attention_mask [1, max_len]` whose leading `valid_len`
    /// entries are 1 (the model derives GQA sequence lengths from the mask, so
    /// the trailing zeros mask the unused KV-buffer tail). KV buffers are the
    /// same fixed shared buffers bound in place as both past inputs and present
    /// outputs. Logits are written into a persistent output buffer. The first
    /// such step captures the graph; subsequent steps replay it.
    fn step_captured(
        &mut self,
        token: i64,
        attention_mask: &[i64],
        position: i64,
        params: Option<&DeviceSampleParams>,
    ) -> std::result::Result<StepLogits, CapturedStepError> {
        self.ensure_capture_state()
            .map_err(CapturedStepError::PreRun)?;
        // Move the capture buffers out of `self` for the duration of the step so
        // the `&mut self` bind helpers don't alias the borrow; restore on the
        // success path (an error aborts generation and drops the state).
        let mut cap = self.capture.take().ok_or_else(|| {
            CapturedStepError::PreRun(OrtError::InvalidArgument(
                "capture state initialized".into(),
            ))
        })?;
        let valid_len = attention_mask.len();
        if valid_len > cap.mask_len {
            return Err(CapturedStepError::PreRun(OrtError::InvalidArgument(
                format!(
                    "attention length {valid_len} exceeds captured mask capacity {}",
                    cap.mask_len
                ),
            )));
        }
        // Refresh the per-step dynamic inputs in their persistent, fixed-address
        // buffers. Device-resident inputs are updated in place with a
        // host->device copy (the captured graph reads them directly on replay);
        // host inputs are updated with a plain memcpy and ORT re-copies them on
        // the re-bind below.
        let write_span = crate::prof_span!("ort.write_step_inputs");
        if cap.inputs_on_device {
            #[cfg(feature = "cuda")]
            {
                cap.write_step_inputs_device(token, position, valid_len)
                    .map_err(CapturedStepError::PreRun)?;
            }
        } else {
            cap.write_step_inputs_host(token, position, valid_len)
                .map_err(CapturedStepError::PreRun)?;
        }
        drop(write_span);

        // On the first step under this capture id (and after any reset/rewind/KV
        // grow that calls `invalidate_captured_graph`), bind every input and
        // output so the graph is captured against these exact buffers. On later
        // steps that merely replay the captured graph, the output tensors (KV
        // shared buffers and logits) are device-resident and unchanged, so their
        // bindings persist untouched.
        //
        // How the *inputs* are refreshed depends on where they live:
        // - Device-resident inputs (the CUDA device-argmax path): the captured
        //   graph reads their device buffers directly, so mutating those buffers
        //   in place (above) is observed on the next replay. The one-time
        //   binding stands — nothing is cleared or re-bound per token. This is
        //   the fix for microsoft/onnxruntime#29782: binding tiny CPU inputs
        //   forced a per-step `ClearBoundInputs` + full re-bind of the entire set
        //   (including the large, unchanged KV buffers) purely to trigger the
        //   host->device copy.
        // - Host CPU inputs (CPU-argmax fallback): ORT only copies a bound CPU
        //   input host->device on (re)bind, so the mutated inputs must be cleared
        //   and re-bound each step; clearing inputs also drops the KV input
        //   bindings, so those are re-bound too (cheap: device-resident, no copy).
        let bind_span = crate::prof_span!("ort.bind_inputs");
        if self.capture_bound {
            if !cap.inputs_on_device {
                self.binding
                    .clear_inputs()
                    .map_err(CapturedStepError::PreRun)?;
                self.bind_standard_inputs(&cap.input_ids, &cap.attention_mask, &cap.position_ids)
                    .map_err(CapturedStepError::PreRun)?;
                self.bind_kv_inputs().map_err(CapturedStepError::PreRun)?;
            }
        } else {
            self.binding.clear().map_err(CapturedStepError::PreRun)?;
            self.bind_standard_inputs(&cap.input_ids, &cap.attention_mask, &cap.position_ids)
                .map_err(CapturedStepError::PreRun)?;
            self.bind_kv_inputs().map_err(CapturedStepError::PreRun)?;
            for output in self.session.output_names() {
                if let Some(pair) = self.kv_pairs.iter().find(|pair| pair.present == *output) {
                    let value = self.current_kv.get(&pair.past).ok_or_else(|| {
                        CapturedStepError::PreRun(OrtError::InvalidArgument(format!(
                            "missing shared KV buffer for '{}'",
                            pair.past
                        )))
                    })?;
                    self.binding
                        .bind_output(output, value)
                        .map_err(CapturedStepError::PreRun)?;
                } else if output == &self.logits_output {
                    self.binding
                        .bind_output(output, &cap.logits)
                        .map_err(CapturedStepError::PreRun)?;
                } else {
                    self.binding
                        .bind_output_to_device(
                            output,
                            &MemoryInfo::cpu().map_err(CapturedStepError::PreRun)?,
                        )
                        .map_err(CapturedStepError::PreRun)?;
                }
            }
        }
        drop(bind_span);

        {
            let _run_span = crate::prof_span!("ort.session_run");
            let graph_id = self.capture_graph_id.ok_or_else(|| {
                CapturedStepError::PreRun(OrtError::InvalidArgument(
                    "capture graph id assigned in ensure_capture_state".into(),
                ))
            })?;
            self.session
                .run_with_binding_graph_phased(&self.binding, graph_id)
                .map_err(classify_run_phase)?;
        }
        // A graph is now captured under `capture_graph_id`; mark it so reset /
        // rewind / drop release it before this session's buffers are freed.
        self.capture_bound = true;
        self.current_len = self.current_len.checked_add(1).ok_or_else(|| {
            CapturedStepError::RunInvoked(OrtError::InvalidArgument(
                "decode length overflow".into(),
            ))
        })?;

        // Reduce or copy the persistent logits buffer while it is still live.
        // With `params` set (greedy or non-greedy) the device sampler reads only
        // the winning token id straight from the buffer — the full vocabulary
        // never leaves the tensor. Without `params` (plain `step`) snapshot it
        // into an owned Value so the caller can consume it while the captured
        // buffer is reused next step.
        let _extract_span = crate::prof_span!("ort.extract_outputs");
        #[cfg(feature = "cuda")]
        if cap.logits_on_device {
            let dtype = cap.logits.dtype();
            let ptr = cap
                .logits
                .data_ptr_addr()
                .map_err(CapturedStepError::RunInvoked)?;
            let device_sampler = self
                .device_sampler
                .as_ref()
                .expect("device sampler initialized for device logits");
            let result = if let Some(params) = params {
                device_sampler
                    .sample_one(dtype, ptr, cap.vocab, params)
                    .map(StepLogits::Token)
            } else {
                device_logits_to_host_value(device_sampler.as_ref(), dtype, ptr, cap.vocab)
                    .map(StepLogits::Value)
            };
            self.capture = Some(cap);
            return result.map_err(CapturedStepError::RunInvoked);
        }
        // Host logits (no CUDA feature / CPU-allocated output). Greedy reduces to
        // the argmax token; non-greedy cannot sample here, so surface the logits
        // Value and let `step_sampled` signal a host fallback.
        let result = match params {
            None => cap.logits.clone_owned().map(StepLogits::Value),
            Some(p) if p.greedy => cap.logits.argmax_last_row().map(StepLogits::Token),
            Some(_) => cap.logits.clone_owned().map(StepLogits::Value),
        };
        self.capture = Some(cap);
        result.map_err(CapturedStepError::RunInvoked)
    }

    /// Lazily allocate the persistent captured-decode I/O buffers.
    fn ensure_capture_state(&mut self) -> Result<()> {
        if self.capture.is_some() {
            return Ok(());
        }
        // Size the captured attention mask to the current KV bucket, not the
        // hard `max_length`: the mask capacity must track the KV buffer's
        // sequence-axis capacity (the model derives GQA sequence lengths from
        // the mask), and both grow together in `grow_kv_buffers`.
        let mask_len = self.kv_capacity;
        if mask_len == 0 {
            return Err(OrtError::InvalidArgument(
                "captured decode requires an allocated KV bucket".into(),
            ));
        }
        let logits_info = self
            .session
            .outputs()
            .iter()
            .find(|info| info.name == self.logits_output)
            .ok_or_else(|| {
                OrtError::InvalidArgument(format!(
                    "resolved logits output '{}' is not exposed",
                    self.logits_output
                ))
            })?;
        let vocab = logits_info
            .shape
            .last()
            .copied()
            .filter(|dim| *dim > 0)
            .ok_or_else(|| {
                OrtError::InvalidArgument("logits output has no static vocab dim".into())
            })?;

        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut input_ids = Value::from_vec_i64(vec![0i64], &[1, 1])?;
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut position_ids = Value::from_vec_i64(vec![0i64], &[1, 1])?;
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut attention_mask = Value::from_vec_i64(vec![0i64; mask_len], &[1, mask_len as i64])?;
        let logits_dtype = logits_info.dtype;
        let vocab_usize = usize::try_from(vocab)
            .map_err(|_| OrtError::InvalidArgument("logits vocab dim is negative".into()))?;
        // Ends the immutable borrow of `self.session` (`logits_info`) before the
        // device-argmax setup below borrows `self` mutably.
        let _ = logits_info;

        // Keep logits on the CUDA device when the session runs on CUDA, so the
        // captured greedy path can argmax the full vocabulary on-device (one
        // 4-byte token id returns) instead of ORT copying it host-side every
        // token. `kv_allocator` is the retained CUDA device allocator (Some only
        // for device EPs in shared-buffer mode); reuse it so it outlives the
        // logits `Value` (mirroring the shared KV buffers).
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut logits_on_device = false;
        // When logits stay on-device we also keep the small dynamic inputs
        // (`input_ids` / `position_ids` / `attention_mask`) device-resident: the
        // captured graph then reads them in place on every replay, so the whole
        // IoBinding set is bound once and never cleared/re-bound per token. The
        // device sampler that this same branch installs fully synchronizes the
        // device at the end of each step, which orders the next step's in-place
        // input overwrite after the prior replay's read (see `step_captured`).
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut inputs_on_device = false;
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut device_id = 0i32;
        let logits;
        #[cfg(feature = "cuda")]
        {
            // Default on; `ONNX_GENAI_DEVICE_ARGMAX=0` forces the host argmax
            // path (CPU logits + full-vocab device->host copy) for A/B testing.
            if device_argmax_enabled() && self.session.cuda_device_id().is_some() {
                if let Some(allocator) = self.kv_allocator.as_ref() {
                    logits = Value::empty_in(&[1, 1, vocab], logits_dtype, allocator)?;
                    logits_on_device = true;
                    device_id = self.session.cuda_device_id().unwrap_or(0).max(0);
                    // Move the per-step dynamic inputs onto the CUDA device.
                    // `empty_in` memory is uninitialized: `input_ids` /
                    // `position_ids` are fully overwritten each step, but the
                    // attention mask's masked-out tail must read zero, so zero
                    // the whole mask buffer once here.
                    let device_input_ids = Value::empty_in(&[1, 1], DataType::Int64, allocator)?;
                    let device_position_ids = Value::empty_in(&[1, 1], DataType::Int64, allocator)?;
                    let device_attention_mask =
                        Value::empty_in(&[1, mask_len as i64], DataType::Int64, allocator)?;
                    device_input_ids.write_i64_prefix_device(&[0], device_id)?;
                    device_position_ids.write_i64_prefix_device(&[0], device_id)?;
                    device_attention_mask.fill_i64_range_device(0, mask_len, 0, device_id)?;
                    input_ids = device_input_ids;
                    position_ids = device_position_ids;
                    attention_mask = device_attention_mask;
                    inputs_on_device = true;
                } else {
                    logits = Value::empty(&[1, 1, vocab], logits_dtype)?;
                }
            } else {
                logits = Value::empty(&[1, 1, vocab], logits_dtype)?;
            }
            if logits_on_device && self.device_sampler.is_none() {
                let device = self.session.cuda_device_id().unwrap_or(0).max(0) as usize;
                let sampler = Box::new(CudaSampler::new(device)?) as Box<dyn DeviceSampler>;
                tracing::debug!(
                    sampler = sampler.name(),
                    device,
                    "initialized device sampler"
                );
                self.device_sampler = Some(sampler);
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            logits = Value::empty(&[1, 1, vocab], logits_dtype)?;
        }

        // Claim a process-unique annotation id so this session captures its own
        // graph rather than re-capturing under an id ORT may still hold from a
        // prior generation on the same underlying ORT session.
        self.capture_graph_id = Some(next_capture_graph_id());

        self.capture = Some(CaptureState {
            input_ids,
            position_ids,
            attention_mask,
            logits,
            mask_len,
            mask_valid_len: 0,
            vocab: vocab_usize,
            logits_on_device,
            inputs_on_device,
            device_id,
        });
        Ok(())
    }

    /// Rewind to a smaller logical KV length.
    ///
    /// In zero-copy-rebind mode this rebinds a compact prefix tensor for each
    /// current present buffer. This is no-copy when the prefix is contiguous in
    /// memory; otherwise rewind performs a one-time compacting slice copy for
    /// correctness. In shared-buffer mode only the logical cursor changes; stale
    /// data beyond `target_len` remains in the buffers and must be masked out by
    /// subsequent attention masks/position ids.
    pub fn rewind(&mut self, target_len: usize) -> Result<()> {
        if target_len > self.current_len {
            return Err(OrtError::InvalidArgument(format!(
                "cannot rewind from {} to larger length {}",
                self.current_len, target_len
            )));
        }
        if target_len == self.current_len {
            return Ok(());
        }
        if target_len == 0 {
            if self.mode == DecodeKvMode::ZeroCopyRebind {
                self.current_kv.clear();
            }
            self.current_len = 0;
            self.invalidate_captured_graph();
            return Ok(());
        }
        if self.mode == DecodeKvMode::ZeroCopyRebind {
            let mut rewound = HashMap::with_capacity(self.current_kv.len());
            for pair in &self.kv_pairs {
                let value = self.current_kv.get(&pair.past).ok_or_else(|| {
                    OrtError::InvalidArgument(format!("missing KV tensor '{}'", pair.past))
                })?;
                let mut shape = value.shape().to_vec();
                shape[pair.seq_axis] = i64::try_from(target_len).map_err(|_| {
                    OrtError::InvalidArgument("target rewind length exceeds i64".into())
                })?;
                rewound.insert(
                    pair.past.clone(),
                    Arc::new(prefix_value(value, &shape, pair.seq_axis)?),
                );
            }

            fn prefix_value(value: &Arc<Value>, shape: &[i64], seq_axis: usize) -> Result<Value> {
                let owner_shape = value.shape();
                let prefix_is_contiguous = owner_shape.iter().take(seq_axis).all(|&dim| dim == 1);
                if prefix_is_contiguous {
                    return Value::alias_with_shape(Arc::clone(value), shape);
                }

                match value.dtype() {
                    DataType::Float32 => {
                        let data = value.to_vec_f32()?;
                        let prefix = copy_prefix(&data, owner_shape, shape);
                        Value::from_vec_f32(prefix, shape)
                    }
                    DataType::Float16 => {
                        let data = value.to_vec_f16_bits()?;
                        let prefix = copy_prefix(&data, owner_shape, shape);
                        Value::from_vec_f16_bits(prefix, shape)
                    }
                    DataType::BFloat16 => {
                        let data = value.to_vec_bf16_bits()?;
                        let prefix = copy_prefix(&data, owner_shape, shape);
                        Value::from_vec_bf16_bits(prefix, shape)
                    }
                    dtype => Err(OrtError::InvalidArgument(format!(
                        "cannot rewind KV tensor with dtype {dtype:?}"
                    ))),
                }
            }

            fn copy_prefix<T: Copy>(
                data: &[T],
                input_shape: &[i64],
                output_shape: &[i64],
            ) -> Vec<T> {
                let output_len = output_shape.iter().product::<i64>() as usize;
                let mut output = Vec::with_capacity(output_len);
                let input_strides = row_major_strides(input_shape);
                for mut linear in 0..output_len {
                    let mut input_offset = 0usize;
                    for (axis, &dim) in output_shape.iter().enumerate().rev() {
                        let index = linear % dim as usize;
                        linear /= dim as usize;
                        input_offset += index * input_strides[axis];
                    }
                    output.push(data[input_offset]);
                }
                output
            }

            fn row_major_strides(shape: &[i64]) -> Vec<usize> {
                let mut strides = vec![1; shape.len()];
                for axis in (0..shape.len().saturating_sub(1)).rev() {
                    strides[axis] = strides[axis + 1] * shape[axis + 1] as usize;
                }
                strides
            }
            self.current_kv = rewound;
        }
        self.current_len = target_len;
        // The captured attention mask relies on the valid region growing
        // monotonically, so a rewind must clear the now-invalid tail and drop
        // the captured graph so the next step re-captures at the new position.
        self.invalidate_captured_graph();
        self.reset_captured_mask()?;
        Ok(())
    }

    /// Zero the valid region of the persistent captured attention mask, if
    /// allocated, and reset the valid-length counter. Called on rewind/reset so
    /// a shorter or restarted sequence never sees stale ones in the trailing
    /// (masked-out) region. Only the previously-valid prefix is cleared — the
    /// rest is already zero — so this stays O(previous context), not O(max_len).
    fn reset_captured_mask(&mut self) -> Result<()> {
        if let Some(cap) = self.capture.as_mut() {
            cap.clear_valid_mask()?;
        }
        Ok(())
    }

    /// Release any captured decode graph and force the next captured step to
    /// re-capture under a fresh annotation id. A captured CUDA graph replays
    /// against the exact buffers/positions seen at capture time; after a reset
    /// or rewind the sequence structure changes, so the stale graph must not be
    /// replayed. A fresh id avoids re-capturing under an id ORT may still hold.
    fn invalidate_captured_graph(&mut self) {
        if self.capture_bound {
            if let Some(graph_id) = self.capture_graph_id {
                let _ = self.session.release_captured_graph(graph_id);
            }
            // Re-capture under a new id if this session keeps decoding.
            self.capture_graph_id = Some(next_capture_graph_id());
            self.capture_bound = false;
        }
    }

    /// Reset the decode cursor and drop zero-copy-rebind KV state.
    pub fn reset(&mut self) {
        if self.mode == DecodeKvMode::ZeroCopyRebind {
            self.current_kv.clear();
        }
        self.current_len = 0;
        self.invalidate_captured_graph();
        let _ = self.reset_captured_mask();
    }

    /// Export the current KV cache as owned, session-independent CPU tensors.
    ///
    /// Each entry is `(past_key_values.* input name, materialized Value)` whose
    /// backing data is copied onto host-owned Rust buffers, so the returned
    /// values outlive the producing session and can be handed to a *different*
    /// [`DecodeSession`] loaded from the same model via [`Self::import_kv`].
    ///
    /// This is the KV-handoff primitive for hybrid execution (e.g. prefill on
    /// the Metal EP, decode on the CPU EP). On Apple-silicon unified memory the
    /// producing session's `present.*` outputs are already CPU-addressable, so
    /// the copy is a cheap host `memcpy`. Only supported in
    /// [`DecodeKvMode::ZeroCopyRebind`], where the runtime holds the present KV
    /// as materialized OrtValues; shared-buffer mode owns fixed max-length
    /// device buffers that are not portable across sessions.
    pub fn export_kv(&self) -> Result<Vec<(String, Value)>> {
        if self.mode != DecodeKvMode::ZeroCopyRebind {
            return Err(OrtError::InvalidArgument(
                "export_kv is only supported in ZeroCopyRebind mode".into(),
            ));
        }
        let mut exported = Vec::with_capacity(self.kv_pairs.len());
        for pair in &self.kv_pairs {
            let value = self.current_kv.get(&pair.past).ok_or_else(|| {
                OrtError::InvalidArgument(format!(
                    "cannot export KV: missing tensor '{}' (run a prefill/decode step first)",
                    pair.past
                ))
            })?;
            exported.push((pair.past.clone(), clone_value_to_owned(value)?));
        }
        Ok(exported)
    }

    /// Adopt a KV cache produced by another session (same model) and set the
    /// logical KV length to `len`.
    ///
    /// The counterpart to [`Self::export_kv`]: it replaces this session's KV
    /// state so the next [`Self::step`] continues generation from `len` tokens
    /// of context. Every `past_key_values.*` tensor this model expects must be
    /// present in `kv` and match the model's dtype; the sequence axis of each
    /// tensor must equal `len`. Only supported in
    /// [`DecodeKvMode::ZeroCopyRebind`].
    pub fn import_kv(&mut self, len: usize, kv: Vec<(String, Value)>) -> Result<()> {
        if self.mode != DecodeKvMode::ZeroCopyRebind {
            return Err(OrtError::InvalidArgument(
                "import_kv is only supported in ZeroCopyRebind mode".into(),
            ));
        }
        let mut incoming: HashMap<String, Value> = kv.into_iter().collect();
        let mut adopted = HashMap::with_capacity(self.kv_pairs.len());
        for pair in &self.kv_pairs {
            let value = incoming.remove(&pair.past).ok_or_else(|| {
                OrtError::InvalidArgument(format!("import_kv missing KV tensor '{}'", pair.past))
            })?;
            if value.dtype() != pair.input.dtype {
                return Err(OrtError::InvalidArgument(format!(
                    "import_kv dtype mismatch for '{}': got {:?}, expected {:?}",
                    pair.past,
                    value.dtype(),
                    pair.input.dtype
                )));
            }
            let seq_dim = value.shape().get(pair.seq_axis).copied().unwrap_or(-1);
            if seq_dim != i64::try_from(len).unwrap_or(-1) {
                return Err(OrtError::InvalidArgument(format!(
                    "import_kv length mismatch for '{}': seq axis {} = {}, expected {}",
                    pair.past, pair.seq_axis, seq_dim, len
                )));
            }
            adopted.insert(pair.past.clone(), Arc::new(value));
        }
        self.current_kv = adopted;
        self.current_len = len;
        Ok(())
    }

    fn bind_standard_inputs(
        &mut self,
        input_ids: &Value,
        attention_mask: &Value,
        position_ids: &Value,
    ) -> Result<()> {
        for input in self.session.inputs() {
            if input.name == self.token_input {
                self.binding.bind_input(&input.name, input_ids)?;
            } else if self.attention_mask_input.as_deref() == Some(input.name.as_str()) {
                self.binding.bind_input(&input.name, attention_mask)?;
            } else if self.position_ids_input.as_deref() == Some(input.name.as_str()) {
                self.binding.bind_input(&input.name, position_ids)?;
            }
        }
        Ok(())
    }

    fn bind_kv_inputs(&mut self) -> Result<()> {
        for pair in &self.kv_pairs {
            let value = if let Some(value) = self.current_kv.get(&pair.past) {
                Arc::clone(value)
            } else {
                Arc::new(empty_past_value(&pair.input)?)
            };
            self.binding.bind_input(&pair.past, &value)?;
        }
        Ok(())
    }

    fn rotate_outputs(&mut self, outputs: Vec<Value>, logits: &mut Option<Value>) -> Result<()> {
        if self.mode == DecodeKvMode::SharedBuffer {
            for (name, value) in self.session.output_names().iter().zip(outputs) {
                if name == &self.logits_output {
                    *logits = Some(value);
                    break;
                }
            }
            return Ok(());
        }

        let present_to_past = self
            .kv_pairs
            .iter()
            .map(|pair| (pair.present.as_str(), pair.past.as_str()))
            .collect::<HashMap<_, _>>();
        let mut next_kv = HashMap::with_capacity(self.kv_pairs.len());
        for (name, value) in self.session.output_names().iter().zip(outputs) {
            if let Some(past_name) = present_to_past.get(name.as_str()) {
                next_kv.insert((*past_name).to_string(), Arc::new(value));
            } else if name == &self.logits_output {
                *logits = Some(value);
            }
        }
        self.current_kv = next_kv;
        Ok(())
    }

    fn allocate_shared_buffers(&mut self, batch_size: i64, max_length: usize) -> Result<()> {
        let max_length = i64::try_from(max_length)
            .map_err(|_| OrtError::InvalidArgument("max_length exceeds i64".into()))?;
        // Prefer a device-resident (e.g. CUDA/WebGPU) allocator so the runtime-owned
        // max-length KV buffers live on the EP device. Bound as both
        // `past_key_values.*` inputs and `present.*` outputs, the KV cache then
        // stays on-device across decode steps (present aliased in place onto
        // past), eliminating the per-step host<->device KV copies the default
        // CPU allocator would force under an accelerator EP. Falls back to the
        // CPU allocator for CPU / non-device EPs.
        let device_allocator = self.session.device_kv_allocator()?;
        let cpu_allocator;
        let allocator = match device_allocator.as_ref() {
            Some(allocator) => allocator,
            None => {
                cpu_allocator = crate::Allocator::default_cpu()?;
                &cpu_allocator
            }
        };
        let mut allocated = Vec::with_capacity(self.kv_pairs.len());
        for pair in &self.kv_pairs {
            let mut shape = pair.input.shape.clone();
            for (axis, dim) in shape.iter_mut().enumerate() {
                if axis == 0 {
                    *dim = batch_size;
                } else if axis == pair.seq_axis {
                    *dim = max_length;
                } else if *dim < 0 {
                    return Err(OrtError::InvalidArgument(format!(
                        "cannot infer shared-buffer static dimension {axis} for '{}'",
                        pair.past
                    )));
                }
            }
            allocated.push((
                pair.past.clone(),
                Arc::new(Value::empty_in(&shape, pair.input.dtype, allocator)?),
            ));
        }
        // The `allocator` borrow of `device_allocator` ends here; retain the
        // owned device allocator so it outlives the KV `Value`s it just backed
        // (see `DecodeSession::kv_allocator`). Moving the wrapper does not change
        // the underlying `OrtAllocator*` the `Value`s reference. The CPU fallback
        // allocator is the process-owned default and needs no retention.
        for (past, value) in allocated {
            self.current_kv.insert(past, value);
        }
        self.kv_allocator = device_allocator;
        Ok(())
    }

    /// Ensure the shared KV buckets can hold `required` sequence positions,
    /// growing (and re-capturing) if they cannot.
    ///
    /// No-op unless `required` exceeds the current bucket. The new capacity is
    /// the next [`kv_capacity_bucket`], so per-step attention cost tracks
    /// ~O(actual length) while growth (and its one-time KV copy + graph
    /// re-capture) happens only O(log length) times per generation.
    fn ensure_kv_capacity(&mut self, required: usize) -> Result<()> {
        if self.mode != DecodeKvMode::SharedBuffer {
            return Ok(());
        }
        let _ = onnx_genai_kv::ensure_kv_capacity(self, required)?;
        Ok(())
    }
}

impl onnx_genai_kv::KvCapacityGrowthBackend for DecodeSession<'_> {
    type Error = OrtError;
    type GrownBuffers = HashMap<String, Arc<Value>>;
    type GrownMask = Value;

    fn current_capacity(&self) -> usize {
        self.kv_capacity
    }

    fn hard_max_capacity(&self) -> usize {
        self.max_length.unwrap_or(0)
    }

    fn valid_len(&self) -> usize {
        self.current_len
    }

    fn capacity_exceeded(&self, required: usize) -> Self::Error {
        OrtError::InvalidArgument(format!(
            "requested KV capacity {required} exceeds model max_length {}",
            self.hard_max_capacity()
        ))
    }

    fn build_grown_buffers(
        &mut self,
        new_capacity: usize,
        valid_len: usize,
    ) -> Result<Self::GrownBuffers> {
        // Resolve the transfer path once, failing fast (before any allocation)
        // if the buffers live on a device we have no copy primitive for.
        let device = self.grow_device()?;

        // Build the replacement buffers first; only swap them in once every KV
        // tensor has been copied successfully, so a mid-way failure leaves the
        // session's existing state intact.
        let old_capacity = self.kv_capacity;
        (|| {
            let mut grown = HashMap::with_capacity(self.kv_pairs.len());
            for pair in &self.kv_pairs {
                let old = self.current_kv.get(&pair.past).ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "cannot grow KV: missing shared buffer for '{}'",
                        pair.past
                    ))
                })?;
                let mut new_shape = old.shape().to_vec();
                new_shape[pair.seq_axis] = i64::try_from(new_capacity)
                    .map_err(|_| OrtError::InvalidArgument("KV capacity exceeds i64".into()))?;
                let new_value = grow_kv_value(
                    old,
                    &new_shape,
                    pair.seq_axis,
                    valid_len,
                    device,
                    self.kv_allocator.as_ref(),
                )?;
                grown.insert(pair.past.clone(), Arc::new(new_value));
            }
            Ok(grown)
        })()
        .map_err(|error| self.kv_growth_failed_error(old_capacity, new_capacity, error))
    }

    fn build_grown_mask(
        &mut self,
        new_capacity: usize,
        _valid_len: usize,
    ) -> Result<Option<Self::GrownMask>> {
        // The captured attention mask capacity must equal the KV buffer
        // capacity. Build its replacement before mutating the session so a
        // fallible allocation cannot leave the KV and capture state out of sync.
        // Preserve the mask's residency: a device-resident captured mask (the
        // CUDA device-argmax path) must stay on the device allocator so the
        // captured graph keeps reading it in place after re-capture.
        if let Some((valid_ones, on_device, mask_device_id)) = self
            .capture
            .as_ref()
            .map(|cap| (cap.mask_valid_len, cap.inputs_on_device, cap.device_id))
        {
            let mask_len = i64::try_from(new_capacity)
                .map_err(|_| OrtError::InvalidArgument("KV capacity exceeds i64".into()))?;
            self.build_grown_capture_mask(
                new_capacity,
                mask_len,
                valid_ones,
                on_device,
                mask_device_id,
            )
            .map(Some)
            .map_err(|error| self.kv_growth_failed_error(self.kv_capacity, new_capacity, error))
        } else {
            Ok(None)
        }
    }

    fn invalidate_capture(&mut self) -> Result<()> {
        // The shared grow driver has completed all fallible allocation/copy
        // work. Release the captured graph while its old buffers are still alive;
        // commit_grown_capacity atomically swaps in the replacement state.
        self.invalidate_captured_graph();
        Ok(())
    }

    fn commit_grown_capacity(
        &mut self,
        new_capacity: usize,
        grown: Self::GrownBuffers,
        grown_mask: Option<Self::GrownMask>,
    ) -> Result<()> {
        self.current_kv = grown;
        self.kv_capacity = new_capacity;
        if let (Some(cap), Some(attention_mask)) = (self.capture.as_mut(), grown_mask) {
            cap.attention_mask = attention_mask;
            cap.mask_len = new_capacity;
        }
        Ok(())
    }
}

impl DecodeSession<'_> {
    fn kv_growth_bytes_per_token(&self) -> Option<usize> {
        let mut bytes = if self.capture.is_some() {
            std::mem::size_of::<i64>()
        } else {
            0
        };
        for pair in &self.kv_pairs {
            let value = self.current_kv.get(&pair.past)?;
            let shape = value.shape();
            let mut elements = 1usize;
            for (axis, dim) in shape.iter().copied().enumerate() {
                if axis == pair.seq_axis {
                    continue;
                }
                let dim = usize::try_from(dim).ok()?;
                elements = elements.checked_mul(dim)?;
            }
            bytes = bytes.checked_add(elements.checked_mul(value.dtype().size_of())?)?;
        }
        Some(bytes)
    }

    fn kv_growth_failed_error(
        &self,
        old_capacity: usize,
        new_capacity: usize,
        error: OrtError,
    ) -> OrtError {
        let bytes_per_token = self.kv_growth_bytes_per_token();
        let byte_summary = bytes_per_token
            .map(|bytes_per_token| {
                let new_bytes = new_capacity.saturating_mul(bytes_per_token);
                let transient_bytes = old_capacity
                    .saturating_add(new_capacity)
                    .saturating_mul(bytes_per_token);
                format!(
                    "The attempted new KV allocation is approximately {new_bytes} bytes and the transient peak is approximately {transient_bytes} bytes because growth keeps the old bucket live until the new bucket and valid-prefix copy are complete. KV bytes/token: {bytes_per_token}."
                )
            })
            .unwrap_or_else(|| {
                "KV bytes/token could not be derived from the current shared buffers.".to_owned()
            });
        let memory = self
            .session
            .cuda_device_id()
            .and_then(|device_id| {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda_rt::device_memory_info(device_id)
                        .ok()
                        .map(|memory| {
                            format!(
                                "CUDA free={} bytes, total={} bytes",
                                memory.free_bytes, memory.total_bytes
                            )
                        })
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = device_id;
                    None
                }
            })
            .unwrap_or_else(|| "device free-memory query unavailable".to_owned());
        OrtError::InvalidArgument(format!(
            "shared-buffer KV capacity growth failed while growing from {old_capacity} to {new_capacity} tokens: {error}. \
             {byte_summary} {memory}. The session state was left unchanged; reset or retry with a shorter prompt/max_new_tokens, set an explicit KV max length cap, or free VRAM used by other processes."
        ))
    }

    /// Build a replacement captured attention mask of `capacity` elements
    /// (shape `[1, mask_len]`) with the leading `valid_ones` set to 1 and the
    /// rest zero, preserving the mask's residency: device-resident when
    /// `on_device` (allocated on the retained CUDA allocator and initialized with
    /// host->device fills), host-resident otherwise. Used by
    /// the shared [`onnx_genai_kv::ensure_kv_capacity`] driver when the KV
    /// capacity grows.
    fn build_grown_capture_mask(
        &self,
        capacity: usize,
        mask_len: i64,
        valid_ones: usize,
        on_device: bool,
        device_id: i32,
    ) -> Result<Value> {
        #[cfg(feature = "cuda")]
        if on_device {
            let allocator = self.kv_allocator.as_ref().ok_or_else(|| {
                OrtError::InvalidArgument(
                    "device-resident captured mask requires a CUDA device allocator".into(),
                )
            })?;
            let mask = Value::empty_in(&[1, mask_len], DataType::Int64, allocator)?;
            mask.fill_i64_range_device(0, capacity, 0, device_id)?;
            if valid_ones > 0 {
                mask.fill_i64_range_device(0, valid_ones, 1, device_id)?;
            }
            return Ok(mask);
        }
        #[cfg(not(feature = "cuda"))]
        let _ = (on_device, device_id);
        let mut mask = vec![0i64; capacity];
        for slot in mask.iter_mut().take(valid_ones) {
            *slot = 1;
        }
        Value::from_vec_i64(mask, &[1, mask_len])
    }

    /// Classify where the shared KV buffers live so [`grow_kv_buffers`] can pick
    /// the right prefix-copy transfer. CPU buffers copy on the host; CUDA
    /// buffers copy device-to-device via `cudart`. Any other device EP has no
    /// implemented copy path, so growth reports a clear error rather than
    /// silently corrupting the KV cache.
    ///
    /// [`grow_kv_buffers`]: DecodeSession::grow_kv_buffers
    fn grow_device(&self) -> Result<GrowDevice> {
        if self.kv_allocator.is_none() {
            return Ok(GrowDevice::Host);
        }
        if self.session.is_cuda() {
            return Ok(GrowDevice::Cuda);
        }
        Err(OrtError::InvalidArgument(
            "shared-buffer KV growth is only implemented for CPU and CUDA \
             devices; this session's device EP has no device-to-device KV copy \
             path"
                .into(),
        ))
    }
}

#[cfg(test)]
mod captured_step_retry_tests {
    use super::*;
    use std::cell::Cell;

    /// A fake model+sampler that counts how many times the model is invoked.
    ///
    /// Its [`Self::captured_step`] mirrors the phase ordering of the real
    /// [`DecodeSession::step_captured`]: a pre-run setup phase (input binding /
    /// run-option setup) that has *not* touched the model, then the single model
    /// invocation, then the post-run device-argmax readback / sampler. Failures
    /// are injected at a chosen phase so a test can prove exactly how many times
    /// the model runs across the fast path and any standard-step fallback.
    struct FakeRunner {
        model_invocations: Cell<usize>,
    }

    impl FakeRunner {
        fn new() -> Self {
            Self {
                model_invocations: Cell::new(0),
            }
        }

        fn invoke_model(&self) {
            self.model_invocations.set(self.model_invocations.get() + 1);
        }

        /// One captured decode step, failing at `fail_phase` if set.
        fn captured_step(
            &self,
            fail_phase: Option<StepPhase>,
        ) -> std::result::Result<i64, CapturedStepError> {
            // PRE-run: input/mask/KV binding and run-option setup. The model has
            // not executed, so a failure here is retryable.
            if fail_phase == Some(StepPhase::Setup) {
                return Err(classify_run_phase(RunPhaseError::Setup(
                    OrtError::InvalidArgument("injected binding failure".into()),
                )));
            }
            // The model invocation itself (advances KV state exactly once).
            self.invoke_model();
            if fail_phase == Some(StepPhase::Run) {
                return Err(classify_run_phase(RunPhaseError::Invoked(
                    OrtError::InvalidArgument("injected run failure".into()),
                )));
            }
            // POST-run: device-argmax readback / device sampler. The model has
            // already advanced, so a failure here must propagate, never retry.
            if fail_phase == Some(StepPhase::Sample) {
                return Err(CapturedStepError::RunInvoked(OrtError::InvalidArgument(
                    "injected sampler failure".into(),
                )));
            }
            Ok(7)
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum StepPhase {
        Setup,
        Run,
        Sample,
    }

    #[test]
    fn post_run_sampling_failure_runs_model_once_and_does_not_fall_back() {
        let runner = FakeRunner::new();
        let result =
            retry_pre_run_captured_failure(runner.captured_step(Some(StepPhase::Sample)), |_| {
                // step_standard fallback would re-invoke the model — forbidden
                // once the sampler failed post-invocation.
                runner.invoke_model();
                Ok(0)
            });

        assert!(result.is_err(), "post-run sampler failure must propagate");
        assert_eq!(
            runner.model_invocations.get(),
            1,
            "post-run failure must not replay the model via the standard step"
        );
    }

    #[test]
    fn post_run_invocation_failure_runs_model_once_and_does_not_fall_back() {
        let runner = FakeRunner::new();
        let result =
            retry_pre_run_captured_failure(runner.captured_step(Some(StepPhase::Run)), |_| {
                runner.invoke_model();
                Ok(0)
            });

        assert!(result.is_err(), "run-call failure must propagate");
        assert_eq!(
            runner.model_invocations.get(),
            1,
            "a failure at the model invocation must not be replayed"
        );
    }

    #[test]
    fn pre_run_setup_failure_falls_back_and_runs_model_once() {
        let runner = FakeRunner::new();
        let result =
            retry_pre_run_captured_failure(runner.captured_step(Some(StepPhase::Setup)), |_| {
                // Setup failed before the model ran, so the standard step safely
                // performs the one and only model invocation.
                runner.invoke_model();
                Ok(99)
            });

        assert_eq!(result.expect("pre-run failure should retry"), 99);
        assert_eq!(
            runner.model_invocations.get(),
            1,
            "pre-run failure must retry through the standard step exactly once"
        );
    }

    #[test]
    fn classify_run_phase_maps_setup_to_retryable_and_invoked_to_propagate() {
        // Setup failures are retryable (PreRun); invocation failures propagate.
        let runner = FakeRunner::new();
        let retried = retry_pre_run_captured_failure(
            Err(classify_run_phase(RunPhaseError::Setup(
                OrtError::InvalidArgument("setup".into()),
            ))),
            |_| {
                runner.invoke_model();
                Ok(1)
            },
        );
        assert_eq!(retried.expect("setup must retry"), 1);
        assert_eq!(runner.model_invocations.get(), 1);

        let propagated: Result<i64> = retry_pre_run_captured_failure(
            Err(classify_run_phase(RunPhaseError::Invoked(
                OrtError::InvalidArgument("invoked".into()),
            ))),
            |_| panic!("invoked failure must not retry"),
        );
        assert!(propagated.is_err());
    }
}
