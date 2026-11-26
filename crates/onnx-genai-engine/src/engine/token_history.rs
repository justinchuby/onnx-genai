//! Immutable logical Native history. Snapshot aliases share payload funding;
//! shorter logical views retain the whole backing until the final owner drops.
use super::*;
use onnx_runtime_memory_governor::{MemoryGovernor, MemoryLease};

#[derive(Clone, Default)]
pub(crate) struct NativeTokenHistory {
    storage: Option<Arc<HistoryStorage>>,
    length: usize,
}
struct HistoryStorage {
    data: onnx_runtime_session::OwnedHostArray<TokenId>,
    // Only absent/foreign domains need a paired legacy Engine grant. A matching
    // allocator already retains actual funding; neither case is a complete
    // managed-worker capability.
    _admission: Option<MemoryLease>,
}
impl NativeTokenHistory {
    pub(crate) fn as_slice(&self) -> &[TokenId] {
        self.storage
            .as_ref()
            .map_or(&[], |storage| &storage.data.as_slice()[..self.length])
    }
    pub(crate) fn truncate(&mut self, length: usize) {
        self.length = self.length.min(length);
    }
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
}
impl std::ops::Deref for NativeTokenHistory {
    type Target = [TokenId];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
impl std::fmt::Debug for NativeTokenHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}
impl PartialEq for NativeTokenHistory {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl Eq for NativeTokenHistory {}
impl Engine {
    pub(super) fn admit_native_token_history(
        &self,
        prompt: &[TokenId],
        generated: &[TokenId],
    ) -> anyhow::Result<NativeTokenHistory> {
        let length = prompt
            .len()
            .checked_add(generated.len())
            .context("Native logical history length overflows; shorten prompt or generation")?;
        if length == 0 {
            return Ok(NativeTokenHistory::default());
        }
        let bytes = length
            .checked_mul(std::mem::size_of::<TokenId>())
            .context("Native logical history byte count overflows; shorten prompt or generation")?;
        let session = self
            .native_session
            .as_ref()
            .context(
                "Native decoder is unavailable while admitting logical history; reload the model",
            )?
            .inference_session();
        let mut context = crate::native_decode::host_tensors_for_session(session)?;
        let funding = context.host_payload_funding();
        let actual_domain = funding.as_ref().is_some_and(|domain| {
            domain.matches(
                self.governor
                    .memory()
                    .authority_id(onnx_runtime_memory_api::Tier::Host),
                &self.governor.process_memory_manager(),
            )
        });
        let admission = if actual_domain {
            None
        } else {
            // A foreign managed provider is not this Engine's domain. Keep its
            // allocation authority untouched and use explicit legacy Host
            // staging, funded by the Engine grant, until B4 binds all paths.
            if funding.is_some() {
                context = onnx_runtime_session::HostTensorAllocator::standalone();
            }
            let holder = memory_plan::Holder::NativeTokenHistory;
            Some(self.governor.memory().reserve(onnx_runtime_memory_api::Tier::Host,u64::try_from(bytes)?,holder.role(),holder.id()).context("admit Native logical history before publication; release retained sessions/snapshots or increase Host headroom")?)
        };
        let mut data=context.array::<TokenId>(length).context("allocate owned Native logical history before publication; release Host headroom or shorten the request")?;
        data.as_mut_slice()[..prompt.len()].copy_from_slice(prompt);
        data.as_mut_slice()[prompt.len()..].copy_from_slice(generated);
        Ok(NativeTokenHistory {
            storage: Some(Arc::new(HistoryStorage {
                data,
                _admission: admission,
            })),
            length,
        })
    }
}

impl AsRef<[TokenId]> for NativeTokenHistory {
    fn as_ref(&self) -> &[TokenId] {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
        ResourceLimit,
    };
    use onnx_runtime_memory_api::Tier;
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    fn engine() -> anyhow::Result<Engine> {
        let mut config = EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cpu),
            ..Default::default()
        };
        config.limits.host_ram_limit = ResourceLimit::Bytes(64 * 1024 * 1024);
        Engine::from_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine"),
            config,
        )
    }
    fn request(tokens: Vec<TokenId>) -> GenerateRequest {
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
        request.options.max_new_tokens = 1;
        request.options.greedy = true;
        request.options.temperature = 0.;
        request.options.stop_on_eos = false;
        request
    }
    #[test]
    fn native_history_quota_failure_after_forward_preserves_boundary_and_same_engine_retry()
    -> anyhow::Result<()> {
        let mut engine = engine()?;
        let session = engine.create_session()?;
        engine.generate_in_session(session, request(vec![4]))?;
        assert_eq!(engine.native_sessions[&session].tokens.as_slice(), [4, 1]);
        let prior = engine.native_sessions[&session].tokens.clone();
        let past = engine.native_session.as_mut().unwrap().snapshot_past()?;
        let stamp = engine.native_access_counter;
        let memory = engine.governor.memory().clone();
        let blocker = Arc::new(Mutex::new(None));
        let pressure = blocker.clone();
        let authority = memory.clone();
        let forwards = Arc::new(AtomicUsize::new(0));
        let observed = forwards.clone();
        let control = crate::GenerationControl::new().with_checkpoint(move |boundary| {
            if boundary == crate::GenerationBoundary::AfterDecoderForward {
                observed.fetch_add(1, Ordering::SeqCst);
                let mut guard = pressure.lock().unwrap();
                if guard.is_none() {
                    let bytes = authority.available(Tier::Host).saturating_sub(1);
                    *guard = Some(authority.reserve(
                        Tier::Host,
                        bytes,
                        memory_plan::Holder::NativeTokenHistory.role(),
                        memory_plan::Holder::NativeTokenHistory.id(),
                    )?);
                }
            }
            Ok(())
        });
        let mut published = Vec::new();
        let mut callback = |token: crate::GenerateToken| {
            published.push(token);
            Ok(())
        };
        let error = engine
            .generate_in_session_with_control_callbacks(
                session,
                request(vec![5, 4]),
                control,
                None,
                Some(&mut callback),
            )
            .expect_err("history grant must fail before durable publication");
        assert!(
            format!("{error:#}").contains("Native logical history"),
            "{error:#}"
        );
        assert!(
            error
                .chain()
                .any(|cause| cause.is::<onnx_runtime_memory_api::MemoryError>()),
            "history quota must preserve its typed resource cause: {error:#}"
        );
        assert!(forwards.load(Ordering::SeqCst) > 0);
        assert!(published.is_empty());
        assert_eq!(engine.native_sessions[&session].tokens, prior);
        assert_eq!(engine.native_access_counter, stamp);
        assert!(engine.native_sessions[&session].poison.is_none());
        let restored = engine.native_session.as_mut().unwrap().snapshot_past()?;
        assert_eq!(restored.host_contents(), past.host_contents());
        assert_eq!(restored.cpu_kv_snapshot(), past.cpu_kv_snapshot());
        drop(restored);
        drop(past);
        drop(blocker.lock().unwrap().take());
        let result = engine.generate_in_session(session, request(vec![5, 4]))?;
        assert_eq!(result.token_ids, vec![1]);
        assert_eq!(
            engine.native_sessions[&session].tokens.as_slice(),
            [5, 4, 1]
        );
        assert_eq!(
            prior.as_slice(),
            [4, 1],
            "new history must not mutate shared snapshot data"
        );
        drop(prior);
        drop(engine);
        assert_eq!(memory.used(Tier::Host), 0);
        Ok(())
    }
    #[test]
    fn native_history_snapshots_share_funding_and_shorter_views_keep_capacity() -> anyhow::Result<()>
    {
        let mut engine = engine()?;
        let session = engine.create_session()?;
        engine.generate_in_session(session, request(vec![4, 5]))?;
        let memory = engine.governor.memory().clone();
        let before = memory.used(Tier::Host);
        let history = engine.native_sessions[&session].tokens.clone();
        let mut shorter = history.clone();
        shorter.truncate(1);
        assert_eq!(shorter.as_slice(), [4]);
        assert_eq!(history.as_slice(), [4, 5, 1]);
        assert!(std::ptr::eq(
            shorter.as_slice().as_ptr(),
            history.as_slice().as_ptr()
        ));
        assert_eq!(memory.used(Tier::Host), before);
        engine.reset_session(session)?;
        assert!(engine.native_sessions[&session].tokens.is_empty());
        assert_eq!(history.as_slice(), [4, 5, 1]);
        assert_eq!(shorter.as_slice(), [4]);
        drop(engine);
        let retained = memory.used(Tier::Host);
        assert!(
            retained >= 12,
            "shared history capacity must remain funded after source reset/drop"
        );
        drop(history);
        assert_eq!(
            memory.used(Tier::Host),
            retained,
            "short view must retain full capacity funding"
        );
        drop(shorter);
        assert_eq!(memory.used(Tier::Host), 0);
        Ok(())
    }
}
