//! Complete owned Native captures and model-bound state files. Transport path
//! admission remains above the engine; opaque captures cannot cross engines.

use super::*;
use onnx_runtime_memory_api::Tier;
use onnx_runtime_memory_governor::{MemoryGovernor, MemoryLease};

/// An owned committed Native session boundary, including live KV rows,
/// recurrent/convolution state, compressed-state cursors/carries, hidden
/// conditioning, and authoritative logical tokens. The physical cursor can lag
/// logical history by a selected token and is retained independently.
///
/// This is an engine-local capability, not a portable file format. Restore
/// consumes it and rejects a different engine instance or generation before
/// touching the destination. Capture and rollback copies are memory-admitted.
pub struct NativeSessionSnapshot {
    pub(super) origin: SessionForkOrigin,
    pub(super) source: SessionId,
    pub(super) tokens: Vec<TokenId>,
    pub(super) state: crate::native_decode::NativePastSnapshot,
    pub(super) _leases: Vec<MemoryLease>,
}

/// Logical history and executable cursor are independent in a saved file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeStateFileSummary {
    pub logical_tokens: usize,
    pub materialized_tokens: usize,
    pub bytes: u64,
}

/// A file was rejected before live decoder state publication. Resource errors
/// remain in the source chain so transports can distinguish overload from bad input.
#[derive(Debug, thiserror::Error)]
#[error("Native state file was rejected before publication: {0}")]
pub struct NativeStateFileReadError(String);

impl std::fmt::Debug for NativeSessionSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeSessionSnapshot")
            .field("source", &self.source)
            .field("logical_tokens", &self.tokens.len())
            .field("materialized_tokens", &self.state.len())
            .field("state_bytes", &self.state.bytes())
            .finish_non_exhaustive()
    }
}

impl NativeSessionSnapshot {
    pub fn source(&self) -> SessionId {
        self.source
    }

    pub fn logical_token_count(&self) -> usize {
        self.tokens.len()
    }

    pub fn materialized_token_count(&self) -> usize {
        self.state.len()
    }

    pub fn state_bytes(&self) -> u64 {
        self.state.bytes()
    }
}

impl Engine {
    /// Copy the committed executable state of the currently materialized
    /// Native conversation. Inactive histories cannot masquerade as KV state:
    /// generate in the conversation before capturing it. Device-owned
    /// state is rejected rather than copied implicitly to the host.
    ///
    /// This operation does not change history, access stamps, or decoder state.
    pub fn capture_native_session(
        &mut self,
        session: SessionId,
    ) -> anyhow::Result<NativeSessionSnapshot> {
        self.require_native_snapshot_backend()?;
        let logical = self.native_sessions.get(&session).with_context(|| {
            format!(
                "cannot capture native session {session}: session not found; create a session first"
            )
        })?;
        require_healthy(session, logical)?;
        anyhow::ensure!(
            self.native_active_session == Some(session),
            "cannot capture native session {session}: its executable state is not materialized; generate in this session before saving it"
        );
        let native = self.native_session.as_mut().context(
            "cannot capture native session: native decoder is unavailable; reload the model",
        )?;
        anyhow::ensure!(
            native.supports_past_snapshots(),
            "cannot capture native session {session}: state is device-owned; select a host-state executor or implement an explicit device snapshot participant"
        );
        anyhow::ensure!(
            native.current_len() <= logical.tokens.len(),
            "cannot capture native session {session}: physical cursor {} exceeds logical history {}; reset this session before saving it",
            native.current_len(),
            logical.tokens.len()
        );
        let token_bytes = u64::try_from(logical.tokens.len())?
            .checked_mul(u64::try_from(std::mem::size_of::<TokenId>())?)
            .context("native session token snapshot size overflow")?;
        let bytes = native
            .past_snapshot_bytes()?
            .checked_add(token_bytes)
            .context("native session snapshot size overflow")?;
        let lease = reserve(self.governor.memory(), bytes).with_context(|| {
            format!("reserve complete state capture for native session {session}")
        })?;
        let state = native.snapshot_past().with_context(|| {
            format!("capture complete executable state of native session {session}")
        })?;
        Ok(NativeSessionSnapshot {
            origin: self.session_fork_origin,
            source: session,
            tokens: logical.tokens.clone(),
            state,
            _leases: vec![lease],
        })
    }

    /// Install a complete saved boundary into an existing Native session.
    /// The source may have been reset, advanced, or closed since capture.
    /// All admission precedes writes, and an execution failure restores the
    /// previous physical state before returning. If that rollback also fails,
    /// affected logical generations are poisoned until explicitly reset.
    pub fn restore_native_session(
        &mut self,
        target: SessionId,
        snapshot: NativeSessionSnapshot,
    ) -> anyhow::Result<()> {
        self.require_native_snapshot_backend()?;
        anyhow::ensure!(
            snapshot.origin == self.session_fork_origin,
            "cannot restore native session {target}: snapshot belongs to a different engine instance or generation; capture state on the engine that owns the destination"
        );
        let logical = self.native_sessions.get(&target)
            .with_context(|| format!("cannot restore native session {target}: destination session not found; create the destination first"))?;
        require_healthy(target, logical)?;
        let next_generation = logical.generation.checked_add(1)
            .with_context(|| format!("cannot restore native session {target}: logical generation exhausted; close it and create a new session"))?;
        self.install_native_physical_state(target, snapshot.state)?;
        let stamp = self.native_access_counter.saturating_add(1);
        let logical = self
            .native_sessions
            .get_mut(&target)
            .expect("exclusive restore admission retains destination");
        logical.tokens = snapshot.tokens;
        logical.generation = next_generation;
        logical.last_access = stamp;
        self.native_access_counter = stamp;
        self.native_active_session = Some(target);
        // Host-only admission makes bind infallible: there is no device owner
        // registry to update. The public owner is committed only after writes.
        Ok(())
    }

    pub(super) fn install_native_physical_state(
        &mut self,
        target: SessionId,
        state: crate::native_decode::NativePastSnapshot,
    ) -> anyhow::Result<()> {
        let native = self.native_session.as_mut().context(
            "cannot restore native session: native decoder is unavailable; reload the model",
        )?;
        anyhow::ensure!(
            native.supports_past_snapshots(),
            "cannot restore native session {target}: state is device-owned; select the original host-state executor"
        );
        // Captures are opaque and engine-bound, so geometry and ports cannot
        // be supplied by a caller. Capacity only grows within this decoder.
        // Keep a full undo even when another conversation currently owns it.
        let undo = if native.current_len() == 0 {
            None
        } else {
            let lease = reserve(self.governor.memory(), native.past_snapshot_bytes()?)
                .with_context(|| {
                    format!("reserve rollback state before restoring native session {target}")
                })?;
            Some((
                native.snapshot_past().with_context(|| {
                    format!("capture rollback state before restoring native session {target}")
                })?,
                lease,
            ))
        };
        if let Err(error) = native.restore_owned_past_snapshot(state) {
            let rollback = match undo {
                Some((state, _lease)) => native.restore_owned_past_snapshot(state),
                None => native.reset(),
            };
            if let Err(rollback) = rollback {
                let reason = format!(
                    "restore native session {target} failed: {error:#}; rollback also failed: {rollback:#}"
                );
                for id in [Some(target), self.native_active_session]
                    .into_iter()
                    .flatten()
                {
                    if let Some(state) = self.native_sessions.get_mut(&id) {
                        state.poison = Some(reason.clone());
                    }
                }
                self.native_active_session = None;
                return Err(error.context(format!("native state restore rollback failed: {rollback:#}; reset affected sessions or reload the model")));
            }
            return Err(error.context(format!("restore native session {target}; previous state was restored; retry with a fresh capture")));
        }
        Ok(())
    }

    /// Write a bounded, checksummed Native state file. The writer must be
    /// staged/atomically published by the transport; an I/O error can leave a
    /// partial output. Load with `native_state_files=Required` to enable identity.
    pub fn write_native_session_state(
        &mut self,
        session: SessionId,
        writer: &mut impl std::io::Write,
        mut limits: crate::native_decode::NativeStateFileLimits,
    ) -> anyhow::Result<NativeStateFileSummary> {
        self.require_native_snapshot_backend()?;
        let identity = self
            .native_session
            .as_ref()
            .and_then(|s| s.state_identity)
            .context(
                "Native state files are disabled; reload with EngineConfig.native_state_files=Required",
            )?;
        if let Some(context) = self.effective_max_context(&GenerateOptions::default()) {
            limits.max_tokens = limits.max_tokens.min(context);
        }
        let capture = self.capture_native_session(session)?;
        let bytes = capture.state.write_file(
            &capture.tokens,
            identity,
            writer,
            limits,
            self.governor.memory(),
            &self
                .native_session
                .as_ref()
                .context("Native decoder disappeared during state capture; reload the model")?
                .inference_session()
                .host_tensor_allocator()?,
        )?;
        Ok(NativeStateFileSummary {
            logical_tokens: capture.tokens.len(),
            materialized_tokens: capture.state.len(),
            bytes,
        })
    }

    /// Verify and restore a complete state file from the same model package,
    /// including across engine instances/restarts. Admission and checksum
    /// validation complete before the transactional restore touches live state.
    pub fn restore_native_session_state(
        &mut self,
        target: SessionId,
        reader: &mut impl std::io::Read,
        limits: crate::native_decode::NativeStateFileLimits,
    ) -> anyhow::Result<NativeStateFileSummary> {
        self.require_native_snapshot_backend()?;
        let logical = self.native_sessions.get(&target)
            .with_context(|| format!("cannot read state for native session {target}: destination not found; create it first"))?;
        require_healthy(target, logical)?;
        let decoded = self.read_native_file(reader, limits)?;
        let report = NativeStateFileSummary {
            logical_tokens: decoded.tokens.len(),
            materialized_tokens: decoded.state.len(),
            bytes: decoded.encoded_bytes,
        };
        self.restore_native_session(
            target,
            NativeSessionSnapshot {
                origin: self.session_fork_origin,
                source: target,
                tokens: decoded.tokens,
                state: decoded.state,
                _leases: decoded.leases,
            },
        )?;
        Ok(report)
    }

    pub(super) fn read_native_file(
        &self,
        reader: &mut impl std::io::Read,
        mut limits: crate::native_decode::NativeStateFileLimits,
    ) -> anyhow::Result<crate::native_decode::DecodedState> {
        (|| -> anyhow::Result<_> {
        if let Some(context) = self.effective_max_context(&GenerateOptions::default()) {
            limits.max_tokens = limits.max_tokens.min(context);
        }
        let native = self
            .native_session
            .as_ref()
            .context("Native decoder is unavailable; reload the model")?;
        let decoded = native.read_state_file(reader, limits, self.governor.memory())?;
        let tokenizer = self.require_tokenizer()?;
        for (index, token) in decoded.tokens.iter().enumerate() {
            anyhow::ensure!(
                tokenizer.inner().id_to_token(*token).is_some(),
                "Native state token[{index}]={token} is absent from the package vocabulary; restore a file from the original tokenizer"
            );
        }
        Ok(decoded)
        })().map_err(|error| { let reason = error.to_string(); error.context(NativeStateFileReadError(reason)) })
    }

    pub(super) fn require_native_snapshot_backend(&self) -> anyhow::Result<()> {
        self.require_workflow_execution_admitted()?;
        anyhow::ensure!(
            self.holds_decode_core() && self.decode_backend == EngineDecodeBackend::Native,
            "complete native session capture/restore requires the Native decoder backend; load the model with backend=native and a host-state executor"
        );
        Ok(())
    }
}

fn require_healthy(session: SessionId, state: &NativeSessionState) -> anyhow::Result<()> {
    anyhow::ensure!(
        state.poison.is_none(),
        "cannot snapshot/restore native session {session} generation {}: previous rollback failed: {}; reset this session or reload the model",
        state.generation,
        state.poison.as_deref().unwrap_or("")
    );
    Ok(())
}

pub(super) fn reserve(governor: &dyn MemoryGovernor, bytes: u64) -> anyhow::Result<MemoryLease> {
    MemoryGovernor::reserve(
        governor,
        Tier::Host,
        bytes,
        Holder::NativeSessionSnapshot.role(),
        Holder::NativeSessionSnapshot.id(),
    )
    .map_err(anyhow::Error::from)
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
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(crate::NativeDecodeDevice::Cpu),
                ..EngineConfig::default()
            },
        )
    }

    fn request(tokens: Vec<TokenId>, count: usize) -> GenerateRequest {
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
        request.options.max_new_tokens = count;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request
    }

    fn assert_same_physical(
        engine: &mut Engine,
        expected: &crate::native_decode::NativePastSnapshot,
    ) -> anyhow::Result<()> {
        let actual = engine.native_session.as_mut().unwrap().snapshot_past()?;
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual.host_contents(), expected.host_contents());
        assert_eq!(actual.cpu_kv_snapshot(), expected.cpu_kv_snapshot());
        Ok(())
    }

    #[test]
    fn native_session_snapshot_roundtrip_restores_real_state_and_pending_token()
    -> anyhow::Result<()> {
        let mut hidden_verified = false;
        for fixture in [
            "tiny-native-engine",
            "tiny-native-scalar-gqa",
            "tiny-mtp-full",
        ] {
            let mut engine = engine(fixture)?;
            let source = engine.create_session()?;
            engine.generate_in_session(source, request(vec![0], 3))?;
            let original = engine.native_sessions[&source].tokens.clone();
            let baseline = engine.native_session.as_mut().unwrap().snapshot_past()?;
            let hidden = engine
                .native_session
                .as_ref()
                .unwrap()
                .last_hidden()
                .map(<[f32]>::to_vec);
            hidden_verified |= hidden.is_some();
            let capture = engine.capture_native_session(source)?;
            assert_eq!(capture.source(), source);
            assert_eq!(capture.logical_token_count(), original.len());
            assert_eq!(capture.materialized_token_count(), baseline.len());
            assert!(
                capture.materialized_token_count() < capture.logical_token_count(),
                "sampled final token must remain pending across restore"
            );
            assert!(capture.state_bytes() > 0);
            let continuation = request(original.clone(), 3);
            let expected = engine.generate_in_session(source, continuation.clone())?;
            let target = engine.create_session()?;
            engine.restore_native_session(target, capture)?;
            assert_same_physical(&mut engine, &baseline)?;
            assert_eq!(
                engine.native_session.as_ref().unwrap().last_hidden(),
                hidden.as_deref()
            );
            assert_eq!(engine.native_sessions[&target].tokens, original);
            assert_eq!(engine.native_active_session, Some(target));
            assert_eq!(engine.native_sessions[&target].generation, 2);
            assert_eq!(
                engine.generate_in_session(target, continuation)?.token_ids,
                expected.token_ids
            );
            // Source and destination do not alias mutable state.
            assert_eq!(
                engine.native_sessions[&source].tokens.len(),
                original.len() + 3
            );
            engine.reset_session(target)?;
            assert_eq!(
                engine.native_sessions[&source].tokens.len(),
                original.len() + 3
            );
        }
        assert!(
            hidden_verified,
            "roundtrip fixtures must exercise retained hidden conditioning"
        );
        Ok(())
    }

    #[test]
    fn native_session_snapshot_roundtrip_restores_recurrent_participants() -> anyhow::Result<()> {
        let mut engine = super::super::tests::native_prefix_snapshot_test_engine()?;
        let source = engine.create_session()?;
        engine.generate_in_session(source, request(vec![1, 2, 3], 2))?;
        assert!(
            engine
                .native_session
                .as_ref()
                .unwrap()
                .has_recurrent_state()
        );
        let baseline = engine.native_session.as_mut().unwrap().snapshot_past()?;
        let tokens = engine.native_sessions[&source].tokens.clone();
        let capture = engine.capture_native_session(source)?;
        engine.generate_in_session(source, request(tokens.clone(), 2))?;
        engine.restore_native_session(source, capture)?;
        assert_same_physical(&mut engine, &baseline)?;
        assert_eq!(engine.native_sessions[&source].tokens, tokens);
        engine.generate_in_session(source, request(tokens, 1))?;
        Ok(())
    }

    #[test]
    fn native_session_snapshot_restores_compressed_cursors_and_partial_carries()
    -> anyhow::Result<()> {
        let source =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-deepseek-v4-csa");
        let directory = tempfile::tempdir()?;
        for name in ["inference_metadata.yaml", "model.onnx.textproto"] {
            std::fs::copy(source.join(name), directory.path().join(name))?;
        }
        let metadata: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(
            source.join("inference_metadata.yaml"),
        )?)?;
        let count = metadata["package"]["tokenizer"]["vocab_size"]
            .as_u64()
            .context("compressed fixture vocabulary size")?;
        let vocabulary: serde_json::Map<String, serde_json::Value> = (0..count)
            .map(|id| (format!("t{id}"), serde_json::json!(id)))
            .collect();
        std::fs::write(
            directory.path().join("tokenizer.json"),
            serde_json::to_vec(
                &serde_json::json!({"version":"1.0", "truncation":null, "padding":null,
                "added_tokens":[], "normalizer":null, "pre_tokenizer":{"type":"Whitespace"},
                "post_processor":null, "decoder":null,
                "model":{"type":"WordLevel", "vocab":vocabulary, "unk_token":"t0"}}),
            )?,
        )?;
        let mut engine = Engine::from_dir(
            directory.path(),
            EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(crate::NativeDecodeDevice::Cpu),
                ..EngineConfig::default()
            },
        )?;
        let session = engine.create_session()?;
        engine.generate_in_session(session, request(vec![1, 2, 3, 4, 5, 6, 7], 1))?;
        assert!(
            !engine
                .native_session
                .as_ref()
                .unwrap()
                .compressed_record_state()?
                .is_empty()
        );
        let baseline = engine.native_session.as_mut().unwrap().snapshot_past()?;
        let tokens = engine.native_sessions[&session].tokens.clone();
        let capture = engine.capture_native_session(session)?;
        let expected = engine
            .generate_in_session(session, request(tokens.clone(), 3))?
            .token_ids;
        engine.restore_native_session(session, capture)?;
        assert_same_physical(&mut engine, &baseline)?;
        assert_eq!(engine.native_sessions[&session].tokens, tokens);
        assert_eq!(
            engine
                .generate_in_session(session, request(tokens, 3))?
                .token_ids,
            expected
        );
        Ok(())
    }

    #[test]
    fn native_session_snapshot_memory_denial_is_nonmutating() -> anyhow::Result<()> {
        let mut engine = engine("tiny-native-engine")?;
        let source = engine.create_session()?;
        engine.generate_in_session(source, request(vec![0], 3))?;
        let memory = engine.governor.memory().clone();
        let baseline = engine.native_session.as_mut().unwrap().snapshot_past()?;
        let tokens = engine.native_sessions[&source].tokens.clone();
        let stamp = engine.native_access_counter;
        let guard = reserve(&memory, engine.governor.snapshot().host_ram.headroom)?;
        let error = engine.capture_native_session(source).unwrap_err();
        assert!(format!("{error:#}").contains("reserve complete state capture"));
        assert_same_physical(&mut engine, &baseline)?;
        assert_eq!(engine.native_sessions[&source].tokens, tokens);
        assert_eq!(engine.native_access_counter, stamp);
        drop(guard);
        let capture = engine.capture_native_session(source)?;
        let guard = reserve(&memory, engine.governor.snapshot().host_ram.headroom)?;
        let error = engine.restore_native_session(source, capture).unwrap_err();
        assert!(format!("{error:#}").contains("reserve rollback state"));
        assert_same_physical(&mut engine, &baseline)?;
        assert_eq!(engine.native_sessions[&source].tokens, tokens);
        assert_eq!(engine.native_access_counter, stamp);
        drop(guard);
        Ok(())
    }

    #[test]
    fn native_session_snapshot_survives_source_reset_and_close() -> anyhow::Result<()> {
        let mut engine = engine("tiny-native-engine")?;
        let source = engine.create_session()?;
        engine.generate_in_session(source, request(vec![0], 2))?;
        let capture = engine.capture_native_session(source)?;
        let tokens = engine.native_sessions[&source].tokens.clone();
        engine.reset_session(source)?;
        engine.close_session(source)?;
        let target = engine.create_session()?;
        engine.restore_native_session(target, capture)?;
        assert_eq!(engine.native_sessions[&target].tokens, tokens);
        engine.generate_in_session(target, request(tokens, 2))?;
        Ok(())
    }

    #[test]
    fn native_session_snapshot_rejects_inactive_missing_and_foreign_before_mutation()
    -> anyhow::Result<()> {
        let mut a = engine("tiny-native-engine")?;
        let mut b = engine("tiny-native-engine")?;
        let source = a.create_session()?;
        let target = b.create_session()?;
        assert_eq!(source, target, "exercise colliding public session IDs");
        let inactive = a.capture_native_session(source).unwrap_err();
        assert!(inactive.to_string().contains("generate in this session"));
        a.generate_in_session(source, request(vec![0], 2))?;
        b.generate_in_session(target, request(vec![1], 3))?;
        let before = b.native_session.as_mut().unwrap().snapshot_past()?;
        let tokens = b.native_sessions[&target].tokens.clone();
        let stamp = b.native_access_counter;
        let error = b
            .restore_native_session(target, a.capture_native_session(source)?)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("different engine instance or generation")
        );
        assert_same_physical(&mut b, &before)?;
        assert_eq!(b.native_sessions[&target].tokens, tokens);
        assert_eq!(b.native_access_counter, stamp);
        a.advance_session_fork_generation_for_test();
        // A snapshot becomes stale when the engine ownership generation changes.
        let old = a.capture_native_session(source)?;
        a.advance_session_fork_generation_for_test();
        assert!(
            a.restore_native_session(source, old)
                .unwrap_err()
                .to_string()
                .contains("different engine instance or generation")
        );
        b.close_session(target)?;
        assert!(
            b.capture_native_session(target)
                .unwrap_err()
                .to_string()
                .contains("session not found")
        );
        Ok(())
    }

    #[test]
    fn native_session_snapshot_lease_accounts_capture_and_releases_on_drop() -> anyhow::Result<()> {
        let mut engine = engine("tiny-native-engine")?;
        let source = engine.create_session()?;
        engine.generate_in_session(source, request(vec![0], 3))?;
        let used = || MemoryGovernor::used(engine.governor.memory(), Tier::Host);
        let before = used();
        let token_bytes = engine.native_sessions[&source].tokens.len() as u64
            * std::mem::size_of::<TokenId>() as u64;
        let stamp = engine.native_access_counter;
        let capture = engine.capture_native_session(source)?;
        assert_eq!(
            MemoryGovernor::used(engine.governor.memory(), Tier::Host) - before,
            capture.state_bytes() + token_bytes
        );
        assert_eq!(engine.native_access_counter, stamp);
        drop(capture);
        assert_eq!(
            MemoryGovernor::used(engine.governor.memory(), Tier::Host),
            before
        );
        Ok(())
    }

    #[test]
    fn native_session_snapshot_restore_failure_rolls_back_other_live_owner() -> anyhow::Result<()> {
        let mut engine = engine("tiny-native-scalar-gqa")?;
        let source = engine.create_session()?;
        engine.generate_in_session(source, request(vec![0], 2))?;
        let capture = engine.capture_native_session(source)?;
        let target = engine.create_session()?;
        engine.generate_in_session(target, request(vec![1], 4))?;
        let before = engine.native_session.as_mut().unwrap().snapshot_past()?;
        let tokens = engine.native_sessions[&target].tokens.clone();
        let generation = engine.native_sessions[&target].generation;
        let stamp = engine.native_access_counter;
        let inactive = engine.create_session()?;
        engine
            .native_session
            .as_mut()
            .unwrap()
            .fail_past_restores_for_test(1);
        let error = engine
            .restore_native_session(inactive, capture)
            .unwrap_err();
        assert!(format!("{error:#}").contains("after CPU KV writes"));
        assert_same_physical(&mut engine, &before)?;
        assert_eq!(engine.native_active_session, Some(target));
        assert_eq!(engine.native_sessions[&target].tokens, tokens);
        assert_eq!(engine.native_sessions[&target].generation, generation);
        assert_eq!(
            engine.native_access_counter,
            stamp + 1,
            "only create_session touched access"
        );
        assert!(engine.native_sessions[&inactive].tokens.is_empty());
        assert_eq!(engine.native_sessions[&inactive].generation, 1);
        engine.generate_in_session(target, request(tokens, 1))?;
        Ok(())
    }

    #[test]
    fn native_session_snapshot_failed_undo_poisons_only_affected_sessions() -> anyhow::Result<()> {
        let mut engine = engine("tiny-native-scalar-gqa")?;
        let source = engine.create_session()?;
        engine.generate_in_session(source, request(vec![0], 2))?;
        let capture = engine.capture_native_session(source)?;
        let target = engine.create_session()?;
        engine.generate_in_session(target, request(vec![1], 4))?;
        engine
            .native_session
            .as_mut()
            .unwrap()
            .fail_past_restores_for_test(2);
        let error = engine.restore_native_session(source, capture).unwrap_err();
        assert!(format!("{error:#}").contains("rollback failed"));
        assert_eq!(engine.native_active_session, None);
        for session in [source, target] {
            assert!(engine.native_sessions[&session].poison.is_some());
            assert!(
                engine
                    .generate_in_session(session, request(vec![0], 1))
                    .is_err()
            );
            engine.reset_session(session)?;
            assert!(engine.native_sessions[&session].poison.is_none());
        }
        engine.generate_in_session(source, request(vec![0], 1))?;
        Ok(())
    }
}
