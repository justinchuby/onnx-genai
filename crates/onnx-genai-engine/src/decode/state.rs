//! Per-session decode state (`DecodeState`).
//!
//! Pure code motion from `decode.rs`.

use super::resolved_io::ResolvedIo;
use super::step::stable_session_ref;
use super::values::{concat_value_axis, slice_value_axis, validate_fixed_state_budget};
use super::*;

pub(crate) struct DecodeState {
    pub(crate) use_kv: bool,
    /// The cached KV tensors. Private so it cannot be assigned without also
    /// updating `kv_len`: the two must move together or the length silently
    /// describes tokens the cache no longer holds. Every mutation routes
    /// through [`DecodeState::set_past`] (external code) or the length-aware
    /// methods on this type (`rewind_kv`, `apply_window_after_step`,
    /// `rewind_windowed`, `rewind_runner`).
    past: HashMap<String, Value>,
    /// Absolute number of KV tokens `past` represents. For a runner-backed
    /// state the runner owns the length instead (`runner_len`), so this stays
    /// `0` and [`DecodeState::current_kv_len`] reads the runner. For a windowed
    /// state this is the absolute position count, which is larger than the
    /// physically retained rows (`retained_kv_len`).
    kv_len: usize,
    pub(crate) present_to_past: HashMap<String, String>,
    pub(crate) kv_inputs: Vec<String>,
    pub(crate) io: ResolvedIo,
    pub(super) loop_state: HashMap<String, Value>,
    pub(super) positions: Option<PositionProgram>,
    pub(super) next_positions: Option<Vec<i64>>,
    pub(super) sliding_window: Option<usize>,
    pub(super) sink_tokens: usize,
    pub(super) retained_kv_len: usize,
    pub(super) runner: Option<DecodeRunner>,
    #[cfg(test)]
    pub(super) test_runner_marker: bool,
}

impl DecodeState {
    /// Construct decode state from metadata or unambiguous tensor shapes.
    pub(crate) fn new_with_io(
        session: &dyn GraphIo,
        io: Option<&onnx_genai_metadata::ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        Self::new_with_io_and_positions(session, io, None)
    }

    /// Construct generic decoder state from explicit graph I/O and the pipeline's
    /// declared position program.
    pub(crate) fn new_with_io_and_positions(
        session: &dyn GraphIo,
        io: Option<&onnx_genai_metadata::ModelIoSpec>,
        positions: Option<&PositionProgram>,
    ) -> anyhow::Result<Self> {
        Self::new_with_io_positions_and_state_budget(session, io, positions, u64::MAX)
    }

    pub(crate) fn new_with_io_positions_and_state_budget(
        session: &dyn GraphIo,
        io: Option<&onnx_genai_metadata::ModelIoSpec>,
        positions: Option<&PositionProgram>,
        fixed_state_budget_bytes: u64,
    ) -> anyhow::Result<Self> {
        let resolved = ResolvedIo::resolve_with_positions(session, io, positions)?;
        validate_fixed_state_budget(session, &resolved.state_pairs, fixed_state_budget_bytes)?;
        Self::from_resolved(resolved, positions.cloned())
    }

    fn from_resolved(
        resolved: ResolvedIo,
        positions: Option<PositionProgram>,
    ) -> anyhow::Result<Self> {
        let kv_inputs = resolved
            .kv_pairs
            .iter()
            .map(|(past, _)| past.clone())
            .collect::<Vec<_>>();
        let present_to_past = resolved
            .kv_pairs
            .iter()
            .map(|(past, present)| (present.clone(), past.clone()))
            .collect::<HashMap<_, _>>();
        let use_kv = !resolved.kv_pairs.is_empty();
        Ok(Self {
            use_kv,
            past: HashMap::new(),
            kv_len: 0,
            present_to_past,
            kv_inputs,
            io: resolved,
            loop_state: HashMap::new(),
            positions,
            next_positions: None,
            sliding_window: None,
            sink_tokens: 0,
            retained_kv_len: 0,
            runner: None,
            #[cfg(test)]
            test_runner_marker: false,
        })
    }

    /// Create decode state for a selected path, resolving ports from explicit
    /// metadata or unambiguous tensor shapes.
    pub(crate) fn new_for_path_with_io(
        session: &Session,
        path: &ModelDecodePath,
        io: Option<&onnx_genai_metadata::ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        Self::new_for_path_with_io_positions_and_state_budget(session, path, io, None, u64::MAX)
    }

    pub(crate) fn new_for_path_with_io_positions_and_state_budget(
        session: &Session,
        path: &ModelDecodePath,
        io: Option<&onnx_genai_metadata::ModelIoSpec>,
        positions: Option<&PositionProgram>,
        fixed_state_budget_bytes: u64,
    ) -> anyhow::Result<Self> {
        match path {
            ModelDecodePath::Legacy => Self::new_with_io_positions_and_state_budget(
                session,
                io,
                positions,
                fixed_state_budget_bytes,
            ),
            ModelDecodePath::StaticCache { .. } => {
                let resolved = ResolvedIo::resolve_with_positions(session, io, positions)?;
                if !resolved.state_pairs.is_empty() || positions.is_some() {
                    anyhow::bail!(
                        "static-cache decode does not support declared generic positions or fixed loop-carried state; select the past/present or legacy decode path"
                    );
                }
                Ok(Self {
                    use_kv: true,
                    past: HashMap::new(),
                    kv_len: 0,
                    present_to_past: HashMap::new(),
                    kv_inputs: Vec::new(),
                    io: resolved,
                    loop_state: HashMap::new(),
                    positions: None,
                    next_positions: None,
                    sliding_window: None,
                    sink_tokens: 0,
                    retained_kv_len: 0,
                    runner: Some(DecodeRunner::StaticCache(StaticCacheDecodeSession::new(
                        stable_session_ref(session),
                        StaticCacheDecodeOptions { batch_size: 1 },
                        io,
                    )?)),
                    #[cfg(test)]
                    test_runner_marker: false,
                })
            }
            ModelDecodePath::PastPresent {
                shared_buffer,
                max_len,
                sliding_window,
                sink_tokens,
            } => {
                let mut state = Self::new_with_io_positions_and_state_budget(
                    session,
                    io,
                    positions,
                    fixed_state_budget_bytes,
                )?;
                state.sliding_window = *sliding_window;
                state.sink_tokens = sink_tokens.unwrap_or(0);
                if state.use_kv
                    && sliding_window.is_none()
                    && state.io.state_pairs.is_empty()
                    && state.positions.is_none()
                {
                    state.runner = Some(DecodeRunner::PastPresent(DecodeSession::new_with_io(
                        stable_session_ref(session),
                        DecodeSessionOptions {
                            batch_size: 1,
                            max_length: *max_len,
                            past_present_share_buffer: Some(*shared_buffer),
                        },
                        io,
                    )?));
                }
                Ok(state)
            }
        }
    }

    pub(crate) fn has_runner(&self) -> bool {
        self.runner.is_some() || self.has_test_runner_marker()
    }

    #[cfg(test)]
    fn has_test_runner_marker(&self) -> bool {
        self.test_runner_marker
    }

    #[cfg(not(test))]
    fn has_test_runner_marker(&self) -> bool {
        false
    }

    pub(crate) fn is_windowed(&self) -> bool {
        self.sliding_window.is_some()
    }

    pub(crate) fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }

    /// Number of pinned leading attention-sink tokens (0 if disabled).
    pub(crate) fn sink_tokens(&self) -> usize {
        self.sink_tokens
    }

    pub(crate) fn uses_token_prefix_cache(&self) -> bool {
        self.has_runner() || self.is_windowed()
    }

    pub(crate) fn validate_rewind_to_len(
        &self,
        absolute_current_len: usize,
        target_len: usize,
        has_paged_materialization: bool,
        runner_policy: crate::kv_bridge::RewindRunnerPolicy,
    ) -> anyhow::Result<()> {
        if !self.use_kv || absolute_current_len == target_len {
            return Ok(());
        }
        if self.has_runner() {
            if runner_policy == crate::kv_bridge::RewindRunnerPolicy::AllowRunnerRewind {
                return Ok(());
            }
            anyhow::bail!(
                "cannot rewind runner-backed decoder state to token {target_len}; start a fresh session and replay the prefix instead"
            );
        }
        if self.is_windowed() {
            return self.validate_windowed_rewind(absolute_current_len, target_len);
        }
        if !has_paged_materialization {
            anyhow::bail!("cannot rewind ORT KV tensors without paged KV materialization");
        }
        Ok(())
    }

    fn validate_windowed_rewind(
        &self,
        absolute_current_len: usize,
        target_len: usize,
    ) -> anyhow::Result<()> {
        let _window_size = self
            .sliding_window
            .context("windowed rewind requires sliding-window state")?;
        if self.sink_tokens == 0 {
            let retained_start = absolute_current_len.saturating_sub(self.retained_kv_len);
            if target_len < retained_start {
                anyhow::bail!(
                    "cannot rewind sliding-window KV to absolute position {target_len}; positions before {retained_start} were evicted"
                );
            }
            return Ok(());
        }

        let sink = self.sink_tokens.min(self.retained_kv_len);
        let window_len = self.retained_kv_len - sink;
        let window_abs_start = absolute_current_len.saturating_sub(window_len);
        if target_len >= window_abs_start || target_len <= sink {
            return Ok(());
        }
        anyhow::bail!(
            "cannot rewind sliding-window KV to absolute position {target_len}; positions in the evicted gap [{sink}, {window_abs_start}) are unavailable"
        );
    }

    pub(crate) fn retained_kv_len(&self, absolute_past_len: usize) -> usize {
        if self.is_windowed() {
            self.retained_kv_len
        } else {
            absolute_past_len
        }
    }

    pub(crate) fn runner_len(&self) -> usize {
        match &self.runner {
            Some(DecodeRunner::StaticCache(session)) => session.current_len(),
            Some(DecodeRunner::PastPresent(session)) => session.past_len(),
            #[cfg(feature = "native-backend")]
            Some(DecodeRunner::Native(session)) => session.current_len(),
            None => 0,
        }
    }

    /// The KV tensors this state currently holds. Read-only; writes go through
    /// [`set_past`](Self::set_past) or the length-aware rewind/window methods so
    /// the tracked length can never diverge from the tensors it describes.
    pub(crate) fn past(&self) -> &HashMap<String, Value> {
        &self.past
    }

    /// Replace the cached KV and its absolute length together. This is the only
    /// way code outside this module can write `past`; the length is not
    /// optional, so a caller cannot leave it stale.
    pub(crate) fn set_past(&mut self, past: HashMap<String, Value>, kv_len: usize) {
        self.past = past;
        self.kv_len = kv_len;
    }

    /// Absolute KV length this state owns. A runner-backed state defers to the
    /// runner (which owns its own cursor); every other state tracks the length
    /// next to the `past` tensors it describes, so the pipeline no longer has
    /// to thread it in from the retained context.
    pub(crate) fn current_kv_len(&self) -> usize {
        match self.runner {
            Some(_) => self.runner_len(),
            None => self.kv_len,
        }
    }

    /// Drop the KV for everything past `target`, reusing the shared prefix a
    /// diverging prompt still has in common. Same name and signature as
    /// [`PipelineDecoderComponent::rewind_kv`](crate::pipeline::PipelineDecoderComponent::rewind_kv),
    /// so both backends' `KvPrefixStore` adapters are the same shape: the state
    /// owns its length, so it is not told `current_len` from outside.
    pub(crate) fn rewind_kv(&mut self, target: usize) -> anyhow::Result<bool> {
        self.truncate_past(self.current_kv_len(), target)
    }

    /// Whether the active runner can select the greedy token internally via
    /// [`DecodeBackend::decode_argmax`] without materializing host logits. Only
    /// the shared-buffer past/present runner supports this today; the check is
    /// side-effect-free so callers can decide before consuming any input.
    pub(crate) fn runner_supports_argmax(&self) -> bool {
        self.runner
            .as_ref()
            .is_some_and(DecodeRunner::supports_argmax)
    }

    pub(crate) fn runner_supports_sampled(&self) -> bool {
        self.runner
            .as_ref()
            .is_some_and(DecodeRunner::supports_sampled)
    }

    /// Drop the KV for everything past `target_len`, so a prompt that diverges
    /// from the retained context can still reuse the part it shares.
    ///
    /// Returns `false` when the state cannot be truncated soundly, which is a
    /// normal outcome, not an error: the caller recomputes instead.
    ///
    /// A pipeline component's past is an opaque per-graph tensor with no
    /// declared sequence axis, so the axis is identified by extent — the one
    /// dimension equal to the current KV length. When more than one axis
    /// matches, the choice is a guess, and guessing wrong silently corrupts
    /// attention, so this refuses instead.
    /// A bare state holding only `past`, for testing truncation mechanics.
    #[cfg(test)]
    pub(crate) fn for_test_with_past(past: HashMap<String, Value>) -> Self {
        Self {
            use_kv: true,
            past,
            kv_len: 0,
            present_to_past: HashMap::new(),
            kv_inputs: Vec::new(),
            io: ResolvedIo::default(),
            loop_state: HashMap::new(),
            positions: None,
            next_positions: None,
            sliding_window: None,
            sink_tokens: 0,
            retained_kv_len: 0,
            runner: None,
            test_runner_marker: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test_windowed(sliding_window: usize, retained_kv_len: usize) -> Self {
        Self {
            use_kv: true,
            past: HashMap::new(),
            kv_len: 0,
            present_to_past: HashMap::new(),
            kv_inputs: Vec::new(),
            io: ResolvedIo::default(),
            loop_state: HashMap::new(),
            positions: None,
            next_positions: None,
            sliding_window: Some(sliding_window),
            sink_tokens: 0,
            retained_kv_len,
            runner: None,
            test_runner_marker: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test_runner_backed() -> Self {
        Self {
            use_kv: true,
            past: HashMap::new(),
            kv_len: 0,
            present_to_past: HashMap::new(),
            kv_inputs: Vec::new(),
            io: ResolvedIo::default(),
            loop_state: HashMap::new(),
            positions: None,
            next_positions: None,
            sliding_window: None,
            sink_tokens: 0,
            retained_kv_len: 0,
            runner: None,
            test_runner_marker: true,
        }
    }

    /// Borrow the active native runner iff it carries recurrent (GDN SSM /
    /// conv1d) state, so the speculative loop can snapshot/commit that state
    /// around a verify window. `None` for every non-native or pure-dense target,
    /// keeping the commit path inert elsewhere.
    #[cfg(feature = "native-backend")]
    pub(crate) fn native_recurrent_runner_mut(
        &mut self,
    ) -> Option<&mut crate::native_decode::NativeDecodeSession> {
        self.runner
            .as_mut()
            .and_then(DecodeRunner::native_recurrent_mut)
    }

    pub(crate) fn rewind_runner(&mut self, target_len: usize) -> anyhow::Result<()> {
        if target_len != 0 && !self.loop_state.is_empty() {
            anyhow::bail!(
                "cannot rewind fixed loop-carried decoder state to token {target_len}; reset to zero and replay the prefix instead"
            );
        }
        match &mut self.runner {
            Some(DecodeRunner::StaticCache(session)) => session.rewind(target_len)?,
            Some(DecodeRunner::PastPresent(session)) => session.rewind(target_len)?,
            #[cfg(feature = "native-backend")]
            Some(DecodeRunner::Native(session)) => session.rewind(target_len)?,
            None => {
                self.past.clear();
                self.kv_len = 0;
            }
        }
        if target_len == 0 {
            self.loop_state.clear();
            self.next_positions = None;
        }
        Ok(())
    }

    /// Whether this state's runner can hand off its KV cache as owned host
    /// tensors (export/import) — true only for a `PastPresent` runner in
    /// [`DecodeKvMode::ZeroCopyRebind`]. Shared-buffer / static-cache runners own
    /// fixed device buffers that are not portable across sessions, so the
    /// connector cannot extract or inject their KV.
    pub(crate) fn runner_supports_kv_handoff(&self) -> bool {
        matches!(
            &self.runner,
            Some(DecodeRunner::PastPresent(session))
                if session.mode() == DecodeKvMode::ZeroCopyRebind
        )
    }

    /// Export the runner's current KV as owned `(past_key_values.* name, Value)`
    /// pairs covering `[0, runner_len())`. Only valid when
    /// [`runner_supports_kv_handoff`](Self::runner_supports_kv_handoff) is true.
    pub(crate) fn export_runner_kv(&self) -> anyhow::Result<Vec<(String, Value)>> {
        match &self.runner {
            Some(DecodeRunner::PastPresent(session)) => Ok(session.export_kv()?),
            _ => anyhow::bail!("no ZeroCopyRebind PastPresent runner to export KV from"),
        }
    }

    /// Replace the runner's KV with `kv` (covering `len` tokens) so the next
    /// decode step continues from `len` tokens of context. Only valid when
    /// [`runner_supports_kv_handoff`](Self::runner_supports_kv_handoff) is true.
    pub(crate) fn import_runner_kv(
        &mut self,
        len: usize,
        kv: Vec<(String, Value)>,
    ) -> anyhow::Result<()> {
        match &mut self.runner {
            Some(DecodeRunner::PastPresent(session)) => {
                session.import_kv(len, kv)?;
                Ok(())
            }
            _ => anyhow::bail!("no ZeroCopyRebind PastPresent runner to import KV into"),
        }
    }

    pub(crate) fn apply_window_after_step(
        &mut self,
        session: &Session,
        _absolute_total_len: usize,
        present_len: usize,
    ) -> anyhow::Result<()> {
        let Some(window_size) = self.sliding_window else {
            return Ok(());
        };
        let sink = self.sink_tokens.min(present_len);
        let window_start = present_len.saturating_sub(window_size);
        // The sink prefix and the window cover the whole present buffer: keep it.
        if window_start <= sink {
            self.retained_kv_len = present_len;
            return Ok(());
        }
        let window_len = present_len - window_start;
        for input_name in &self.kv_inputs {
            let info = session
                .inputs()
                .iter()
                .find(|info| info.name == *input_name)
                .with_context(|| format!("missing KV input metadata for '{input_name}'"))?;
            let seq_axis = info
                .shape
                .len()
                .checked_sub(2)
                .context("KV input rank must be at least 2")?;
            let value = self
                .past
                .get(input_name)
                .with_context(|| format!("missing cached KV tensor for '{input_name}'"))?;
            let trimmed = if sink == 0 {
                slice_value_axis(value, seq_axis, window_start, window_len)?
            } else {
                // StreamingLLM: pin sink rows, then the trailing window rows.
                let head = slice_value_axis(value, seq_axis, 0, sink)?;
                let tail = slice_value_axis(value, seq_axis, window_start, window_len)?;
                concat_value_axis(&head, &tail, seq_axis)?
            };
            self.past.insert(input_name.clone(), trimmed);
        }
        self.retained_kv_len = sink + window_len;
        Ok(())
    }

    pub(crate) fn rewind_windowed(
        &mut self,
        absolute_current_len: usize,
        target_len: usize,
    ) -> anyhow::Result<()> {
        let window_size = self
            .sliding_window
            .context("windowed rewind requires sliding-window state")?;

        if self.sink_tokens == 0 {
            let retained_start = absolute_current_len.saturating_sub(self.retained_kv_len);
            if target_len < retained_start {
                anyhow::bail!(
                    "cannot rewind sliding-window KV to absolute position {target_len}; positions before {retained_start} were evicted"
                );
            }
            let target_retained_len = target_len - retained_start;
            if target_retained_len < self.retained_kv_len {
                for value in self.past.values_mut() {
                    let seq_axis = value
                        .shape()
                        .len()
                        .checked_sub(2)
                        .context("KV tensor rank must be at least 2")?;
                    *value = slice_value_axis(value, seq_axis, 0, target_retained_len)?;
                }
            }
            self.retained_kv_len = target_retained_len.min(window_size);
            self.kv_len = target_len;
            return Ok(());
        }

        // Sink-aware layout: the buffer holds `sink` pinned rows followed by the
        // window rows, so the absolute retained set is `[0, sink) ∪ [ws, len)`.
        let sink = self.sink_tokens.min(self.retained_kv_len);
        let window_len = self.retained_kv_len - sink;
        let window_abs_start = absolute_current_len.saturating_sub(window_len);
        let new_retained = if target_len >= window_abs_start {
            sink + (target_len - window_abs_start)
        } else if target_len <= sink {
            target_len
        } else {
            anyhow::bail!(
                "cannot rewind sliding-window KV to absolute position {target_len}; positions in the evicted gap [{sink}, {window_abs_start}) are unavailable"
            );
        };
        if new_retained < self.retained_kv_len {
            for value in self.past.values_mut() {
                let seq_axis = value
                    .shape()
                    .len()
                    .checked_sub(2)
                    .context("KV tensor rank must be at least 2")?;
                *value = slice_value_axis(value, seq_axis, 0, new_retained)?;
            }
        }
        self.retained_kv_len = new_retained;
        self.kv_len = target_len;
        Ok(())
    }
}
