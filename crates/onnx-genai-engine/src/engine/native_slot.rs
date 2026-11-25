//! The one physical serial Native decoder, independent of conversation IDs.
use super::*;
use crate::native_decode::NativeStateFileLimits;

/// Inspectable retained-cache boundary of a single-sequence Native executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeSlotState {
    pub materialized_tokens: usize,
    pub max_context: Option<usize>,
    pub state_files: bool,
}

impl Engine {
    /// A slot exists only when a real Native decoder owns one sequence lane.
    /// Queue capacity and the number of logical conversations confer no slots.
    pub fn native_slot_state(&self) -> Option<NativeSlotState> {
        if self.decode_backend != EngineDecodeBackend::Native {
            return None;
        }
        let native = self.native_session.as_ref()?;
        if native.native_decode_batch() != 1 {
            return None;
        }
        Some(NativeSlotState {
            materialized_tokens: if self.native_active_session.is_some() {
                native.current_len()
            } else {
                0
            },
            max_context: self.effective_max_context(&GenerateOptions::default()),
            state_files: native.state_identity.is_some(),
        })
    }

    pub fn write_native_slot_state(
        &mut self,
        writer: &mut impl std::io::Write,
        limits: NativeStateFileLimits,
    ) -> anyhow::Result<NativeStateFileSummary> {
        self.require_native_slot()?;
        if let Some(id) = self.native_active_session {
            let logical = self
                .native_sessions
                .get(&id)
                .context("Native slot owner disappeared; reset the cache or reload the model")?;
            if !logical.tokens.is_empty() {
                return self.write_native_session_state(id, writer, limits);
            }
        }
        let native = self
            .native_session
            .as_ref()
            .expect("admitted Native decoder");
        let identity = native.state_identity.context("Native slot state files are disabled; enable Native state files for this host-state decoder")?;
        let _lease = super::native_session_snapshot::reserve(
            self.governor.memory(),
            native.empty_snapshot_memory_bytes()?,
        )?;
        let empty = native.empty_past_snapshot()?;
        let bytes = empty.write_file(&[], identity, writer, limits, self.governor.memory())?;
        Ok(NativeStateFileSummary {
            logical_tokens: 0,
            materialized_tokens: 0,
            bytes,
        })
    }

    /// Restore into the private stateless-request cache, never into a named
    /// conversation that happened to occupy the physical decoder previously.
    pub fn restore_native_slot_state(
        &mut self,
        reader: &mut impl std::io::Read,
        limits: NativeStateFileLimits,
    ) -> anyhow::Result<NativeStateFileSummary> {
        self.require_native_slot()?;
        let decoded = self.read_native_file(reader, limits)?;
        let report = NativeStateFileSummary {
            logical_tokens: decoded.tokens.len(),
            materialized_tokens: decoded.state.len(),
            bytes: decoded.encoded_bytes,
        };
        if let Some(id) = self
            .native_default_session
            .filter(|id| self.native_sessions.contains_key(id))
        {
            self.restore_native_session(
                id,
                NativeSessionSnapshot {
                    origin: self.session_fork_origin,
                    source: id,
                    tokens: decoded.tokens,
                    state: decoded.state,
                    _leases: decoded.leases,
                },
            )?;
        } else {
            // Reserve publication before physical writes. A failed decode or
            // install must not evict a live named conversation or stamp it.
            self.native_sessions
                .try_reserve(1)
                .context("reserve Native default-cache ownership; free host memory and retry")?;
            let id = self.next_native_session_id();
            self.install_native_physical_state(id, decoded.state)?;
            let stamp = self.touch_native_session();
            self.native_sessions.insert(
                id,
                NativeSessionState {
                    tokens: decoded.tokens,
                    last_access: stamp,
                    generation: 1,
                    poison: None,
                },
            );
            self.native_default_session = Some(id);
            self.native_active_session = Some(id);
            self.evict_native_sessions(id);
        }
        Ok(report)
    }

    /// Erase retained physical cache and private default history. Named logical
    /// conversations remain unchanged and can materialize their own history.
    pub fn erase_native_slot_state(&mut self) -> anyhow::Result<usize> {
        let before = self.require_native_slot()?.materialized_tokens;
        let generation = self
            .native_default_session
            .and_then(|id| self.native_sessions.get(&id))
            .map(|s| {
                s.generation
                    .checked_add(1)
                    .context("Native default-cache generation exhausted; reload the model")
            })
            .transpose()?;
        let native = self
            .native_session
            .as_ref()
            .expect("admitted Native decoder");
        anyhow::ensure!(
            native.supports_past_snapshots(),
            "Native slot erase requires an admitted host-state participant; select a host-state executor"
        );
        let _lease = super::native_session_snapshot::reserve(
            self.governor.memory(),
            native.empty_snapshot_memory_bytes()?,
        )?;
        let empty = native.empty_past_snapshot()?;
        let label = self
            .native_active_session
            .or(self.native_default_session)
            .unwrap_or_else(|| self.next_native_session_id());
        self.install_native_physical_state(label, empty)?;
        if let Some(id) = self.native_default_session
            && let Some(state) = self.native_sessions.get_mut(&id)
        {
            state.tokens.clear();
            state.generation = generation.expect("existing default generation was preflighted");
            state.poison = None;
        }
        self.native_active_session = None;
        Ok(before)
    }

    fn require_native_slot(&self) -> anyhow::Result<NativeSlotState> {
        self.require_native_snapshot_backend()?;
        self.native_slot_state().context("this executor has no admitted serial Native slot; use a single-sequence Native decoder")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn engine(fixture: &str) -> anyhow::Result<Engine> {
        Engine::from_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures")
                .join(fixture),
            EngineConfig {
                native_state_files: crate::NativeStateFilePolicy::WhenSupported,
                native_device: Some(crate::NativeDecodeDevice::Cpu),
                ..Default::default()
            },
        )
    }
    fn request() -> GenerateRequest {
        let mut r = GenerateRequest::new(GeneratePrompt::TokenIds(vec![0]));
        r.options.max_new_tokens = 3;
        r.options.greedy = true;
        r.options.stop_on_eos = false;
        r
    }
    #[test]
    fn native_slot_optional_files_do_not_reject_unrelated_workflows() -> anyhow::Result<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
        for (fixture, backend) in [
            ("tiny-llm", EngineDecodeBackend::Ort),
            (
                "onnx_genai_workflows/reranking",
                EngineDecodeBackend::Native,
            ),
        ] {
            let e = Engine::from_dir(
                &root.join(fixture),
                EngineConfig {
                    decode_backend: backend,
                    native_state_files: crate::NativeStateFilePolicy::WhenSupported,
                    ..Default::default()
                },
            )?;
            assert!(e.native_slot_state().is_none());
        }
        Ok(())
    }
    #[test]
    fn native_slot_empty_save_restore_and_erase_are_real_empty_cache_operations()
    -> anyhow::Result<()> {
        for fixture in ["tiny-native-engine", "tiny-native-scalar-gqa"] {
            let mut e = engine(fixture)?;
            assert_eq!(e.native_slot_state().unwrap().materialized_tokens, 0);
            let mut file = Vec::new();
            let report = e.write_native_slot_state(&mut file, Default::default())?;
            assert_eq!(report.logical_tokens, 0);
            e.generate(request())?;
            assert!(e.native_slot_state().unwrap().materialized_tokens > 0);
            e.restore_native_slot_state(&mut file.as_slice(), Default::default())?;
            assert_eq!(e.native_slot_state().unwrap().materialized_tokens, 0);
            assert_eq!(e.erase_native_slot_state()?, 0);
            let mut second = Vec::new();
            e.write_native_slot_state(&mut second, Default::default())?;
            assert_eq!(file, second);
            e.generate(request())?;
        }
        Ok(())
    }
    #[test]
    fn native_slot_files_and_erase_do_not_rewrite_named_conversations() -> anyhow::Result<()> {
        let mut e = engine("tiny-native-engine")?;
        e.generate(request())?;
        let mut file = Vec::new();
        let saved = e.write_native_slot_state(&mut file, Default::default())?;
        let named = e.create_session()?;
        let mut r = request();
        r.prompt = GeneratePrompt::TokenIds(vec![1]);
        e.generate_in_session(named, r)?;
        let tokens = e.native_sessions[&named].tokens.clone();
        let generation = e.native_sessions[&named].generation;
        e.restore_native_slot_state(&mut file.as_slice(), Default::default())?;
        assert_eq!(
            e.native_slot_state().unwrap().materialized_tokens,
            saved.materialized_tokens
        );
        assert_eq!(e.native_sessions[&named].tokens, tokens);
        assert_eq!(e.native_sessions[&named].generation, generation);
        assert_eq!(e.erase_native_slot_state()?, saved.materialized_tokens);
        assert_eq!(e.native_sessions[&named].tokens, tokens);
        Ok(())
    }
    #[test]
    fn native_slot_cold_start_skips_reuse_but_retains_committed_speculative_cache()
    -> anyhow::Result<()> {
        let mut e = engine("tiny-mtp-full")?;
        let mut cacheable = request();
        cacheable.prompt = GeneratePrompt::TokenIds(vec![0, 1]);
        let warm = e.generate(cacheable.clone())?;
        assert_eq!(warm.prefix_cache_hit_len, 0);
        assert!(warm.retained_cache_tokens > 0);
        assert_eq!(
            warm.retained_cache_tokens,
            e.native_slot_state().unwrap().materialized_tokens
        );
        let mut file = Vec::new();
        let saved = e.write_native_slot_state(&mut file, Default::default())?;
        let reused = e.generate(cacheable.clone())?;
        assert!(reused.prefix_cache_hit_len > 0);
        assert_eq!(reused.token_ids, warm.token_ids);
        cacheable.options.cold_start = true;
        let cold = e.generate(cacheable.clone())?;
        assert_eq!(cold.prefix_cache_hit_len, 0);
        assert_eq!(cold.token_ids, warm.token_ids);
        assert!(cold.retained_cache_tokens > 0);
        assert_eq!(
            cold.retained_cache_tokens,
            e.native_slot_state().unwrap().materialized_tokens
        );
        assert_eq!(e.erase_native_slot_state()?, cold.retained_cache_tokens);
        assert_eq!(e.native_slot_state().unwrap().materialized_tokens, 0);
        e.restore_native_slot_state(&mut file.as_slice(), Default::default())?;
        assert_eq!(
            e.native_slot_state().unwrap().materialized_tokens,
            saved.materialized_tokens
        );
        cacheable.options.cold_start = false;
        let restored = e.generate(cacheable)?;
        assert!(restored.prefix_cache_hit_len > 0);
        assert_eq!(restored.token_ids, warm.token_ids);
        Ok(())
    }
}
