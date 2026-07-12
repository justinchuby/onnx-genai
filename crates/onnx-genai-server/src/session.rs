use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use anyhow::Context;
use onnx_genai::SessionId;

#[derive(Clone)]
pub(crate) struct SessionRegistry {
    inner: Arc<Mutex<SessionRegistryInner>>,
    max_sessions: usize,
}

#[derive(Debug)]
struct SessionRegistryInner {
    sessions: HashMap<String, SessionEntry>,
    access_clock: u64,
}

#[derive(Debug)]
struct SessionEntry {
    engine_session_id: SessionId,
    last_access: u64,
}

impl SessionRegistry {
    pub(crate) fn new(max_sessions: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SessionRegistryInner {
                sessions: HashMap::new(),
                access_clock: 0,
            })),
            max_sessions,
        }
    }

    pub(crate) fn insert(
        &self,
        client_id: String,
        engine_session_id: SessionId,
    ) -> anyhow::Result<Option<SessionId>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?;
        let evicted = if inner.sessions.len() >= self.max_sessions {
            inner
                .sessions
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(id, _)| id.clone())
                .and_then(|id| {
                    inner
                        .sessions
                        .remove(&id)
                        .map(|entry| entry.engine_session_id)
                })
        } else {
            None
        };
        inner.access_clock = inner.access_clock.saturating_add(1);
        let last_access = inner.access_clock;
        inner.sessions.insert(
            client_id,
            SessionEntry {
                engine_session_id,
                last_access,
            },
        );
        Ok(evicted)
    }

    pub(crate) fn get(&self, client_id: &str) -> anyhow::Result<Option<SessionId>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?;
        if !inner.sessions.contains_key(client_id) {
            return Ok(None);
        }
        inner.access_clock = inner.access_clock.saturating_add(1);
        let last_access = inner.access_clock;
        let entry = inner
            .sessions
            .get_mut(client_id)
            .expect("entry checked above");
        entry.last_access = last_access;
        Ok(Some(entry.engine_session_id))
    }

    pub(crate) fn remove(&self, client_id: &str) -> anyhow::Result<Option<SessionId>> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?
            .sessions
            .remove(client_id)
            .map(|entry| entry.engine_session_id))
    }

    pub(crate) fn next_client_id(&self) -> anyhow::Result<String> {
        let mut bytes = [0_u8; 16];
        getrandom::fill(&mut bytes).context("OS CSPRNG failed")?;
        Ok(format!("sess-{}", hex_token(&bytes)))
    }
}

fn hex_token(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
