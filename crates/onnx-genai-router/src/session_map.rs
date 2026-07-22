//! Session → node affinity table with optional JSON persistence and migration
//! accounting (see `docs/DESIGN.md` §34.4 and §34.6).
//!
//! Pure and unit-testable: no async, no I/O beyond explicit `save`/`load`.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::node::NodeId;

/// Session → node affinity table. Keeps a running log of migration events for
/// observability (§34.12 metrics feed off this).
#[derive(Debug, Clone, Default)]
pub struct SessionMap {
    map: HashMap<String, NodeId>,
    migrations: Vec<MigrationEvent>,
}

/// A recorded session migration (affinity broke and the session moved).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MigrationEvent {
    /// Opaque session id that moved.
    pub session_id: String,
    /// Node the session was previously pinned to.
    pub from_node: NodeId,
    /// Node the session moved to.
    pub to_node: NodeId,
    /// Why the migration happened.
    pub reason: MigrationReason,
    /// Estimated re-prefill cost in tokens (0 if unknown).
    pub reprefill_tokens: u64,
}

/// Why a session migrated off its affinity node (§34.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationReason {
    /// Original node is down/unhealthy.
    NodeDown,
    /// Original node is overloaded (KV usage over threshold).
    Overloaded,
    /// Manual/operator-triggered rebalancing.
    Rebalance,
}

/// On-disk representation of the session map (persistence format).
#[derive(Debug, Serialize, Deserialize)]
struct PersistedSessionMap {
    sessions: HashMap<String, NodeId>,
}

impl SessionMap {
    /// Create an empty session map.
    pub fn new() -> Self {
        SessionMap::default()
    }

    /// Pin a session to a node (records affinity, no migration).
    pub fn assign(&mut self, session_id: impl Into<String>, node: NodeId) {
        self.map.insert(session_id.into(), node);
    }

    /// Look up the node a session is currently pinned to.
    pub fn get(&self, session_id: &str) -> Option<&NodeId> {
        self.map.get(session_id)
    }

    /// Whether a session already has an affinity entry.
    pub fn contains(&self, session_id: &str) -> bool {
        self.map.contains_key(session_id)
    }

    /// Move a session to `to_node`, recording a [`MigrationEvent`] if it was
    /// previously pinned to a *different* node. Returns the recorded event (if
    /// any). If the session had no prior affinity, this is just an assignment
    /// and no migration is recorded.
    pub fn migrate(
        &mut self,
        session_id: impl Into<String>,
        to_node: NodeId,
        reason: MigrationReason,
        reprefill_tokens: u64,
    ) -> Option<MigrationEvent> {
        let session_id = session_id.into();
        let event = match self.map.get(&session_id) {
            Some(from) if *from != to_node => Some(MigrationEvent {
                session_id: session_id.clone(),
                from_node: from.clone(),
                to_node: to_node.clone(),
                reason,
                reprefill_tokens,
            }),
            _ => None,
        };
        self.map.insert(session_id, to_node);
        if let Some(ev) = &event {
            self.migrations.push(ev.clone());
        }
        event
    }

    /// Drop a session's affinity entry.
    pub fn remove(&mut self, session_id: &str) -> Option<NodeId> {
        self.map.remove(session_id)
    }

    /// Recorded migration history.
    pub fn migrations(&self) -> &[MigrationEvent] {
        &self.migrations
    }

    /// Number of pinned sessions (exposed for the §34.12 metric).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether no sessions are pinned.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterate the current session → node pins.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &NodeId)> {
        self.map.iter()
    }

    /// Serialize the affinity table to pretty JSON (migration log excluded —
    /// only the routing table is persistent state).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let persisted = PersistedSessionMap {
            sessions: self.map.clone(),
        };
        serde_json::to_string_pretty(&persisted)
    }

    /// Rebuild an affinity table from JSON produced by [`SessionMap::to_json`].
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let persisted: PersistedSessionMap = serde_json::from_str(json)?;
        Ok(SessionMap {
            map: persisted.sessions,
            migrations: Vec::new(),
        })
    }

    /// Persist the affinity table to a JSON file (parent dirs must exist).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PersistError> {
        let path = path.as_ref();
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(|source| PersistError::Io {
            path: path.display().to_string(),
            source,
        })
    }

    /// Load an affinity table from a JSON file previously written by
    /// [`SessionMap::save`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PersistError> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path).map_err(|source| PersistError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Ok(Self::from_json(&json)?)
    }
}

/// Errors from session-map persistence.
#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    /// Filesystem error reading/writing the session map.
    #[error("session map I/O error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// JSON (de)serialization error.
    #[error("session map serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assign_and_get() {
        let mut m = SessionMap::new();
        assert!(m.get("s1").is_none());
        m.assign("s1", NodeId::new("gpu-0"));
        assert_eq!(m.get("s1"), Some(&NodeId::new("gpu-0")));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn migrate_records_event_on_node_change() {
        let mut m = SessionMap::new();
        m.assign("s1", NodeId::new("gpu-0"));
        let ev = m
            .migrate("s1", NodeId::new("gpu-1"), MigrationReason::NodeDown, 512)
            .expect("migration recorded");
        assert_eq!(ev.from_node, NodeId::new("gpu-0"));
        assert_eq!(ev.to_node, NodeId::new("gpu-1"));
        assert_eq!(ev.reason, MigrationReason::NodeDown);
        assert_eq!(ev.reprefill_tokens, 512);
        assert_eq!(m.get("s1"), Some(&NodeId::new("gpu-1")));
        assert_eq!(m.migrations().len(), 1);
    }

    #[test]
    fn migrate_to_same_node_is_noop() {
        let mut m = SessionMap::new();
        m.assign("s1", NodeId::new("gpu-0"));
        let ev = m.migrate("s1", NodeId::new("gpu-0"), MigrationReason::Rebalance, 0);
        assert!(ev.is_none());
        assert!(m.migrations().is_empty());
    }

    #[test]
    fn migrate_new_session_does_not_record() {
        let mut m = SessionMap::new();
        let ev = m.migrate(
            "s-new",
            NodeId::new("gpu-2"),
            MigrationReason::Overloaded,
            100,
        );
        assert!(ev.is_none());
        assert_eq!(m.get("s-new"), Some(&NodeId::new("gpu-2")));
        assert!(m.migrations().is_empty());
    }

    #[test]
    fn json_roundtrip_preserves_affinity() {
        let mut m = SessionMap::new();
        m.assign("s1", NodeId::new("gpu-0"));
        m.assign("s2", NodeId::new("gpu-1"));
        let json = m.to_json().unwrap();
        let restored = SessionMap::from_json(&json).unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.get("s1"), Some(&NodeId::new("gpu-0")));
        assert_eq!(restored.get("s2"), Some(&NodeId::new("gpu-1")));
    }

    #[test]
    fn file_persist_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        let mut m = SessionMap::new();
        m.assign("s1", NodeId::new("gpu-0"));
        m.assign("s2", NodeId::new("gpu-1"));
        m.save(&path).unwrap();

        let restored = SessionMap::load(&path).unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.get("s1"), Some(&NodeId::new("gpu-0")));
        assert_eq!(restored.get("s2"), Some(&NodeId::new("gpu-1")));
    }

    #[test]
    fn migration_reason_serializes_snake_case() {
        let json = serde_json::to_string(&MigrationReason::NodeDown).unwrap();
        assert_eq!(json, "\"node_down\"");
    }
}
