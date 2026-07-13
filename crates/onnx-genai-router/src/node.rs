//! Node identity, live node state, and the `/v1/status` deserialization
//! contract (see `docs/DESIGN.md` §34.4 and §34.8).
//!
//! Everything here is pure and unit-testable. The actual async HTTP polling
//! loop is deferred to R3; this module only defines the [`NodeStatusFetcher`]
//! seam and the pure [`NodeState::apply_status`] / [`NodeState::record_miss`]
//! transitions that a poller drives.

use std::time::Instant;

use serde::Deserialize;

/// Opaque node identifier.
///
/// The router assigns no meaning to node ids beyond equality — they are opaque
/// strings supplied by configuration and echoed back by nodes in their
/// `/v1/status` response (`node_id`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    /// Construct a node id from anything string-like.
    pub fn new(id: impl Into<String>) -> Self {
        NodeId(id.into())
    }

    /// Borrow the underlying id string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for NodeId {
    fn from(s: &str) -> Self {
        NodeId(s.to_string())
    }
}

impl From<String> for NodeId {
    fn from(s: String) -> Self {
        NodeId(s)
    }
}

/// Live snapshot of a node's state, refreshed via `GET /v1/status`.
///
/// `address` is stored as an opaque string (may be `host:port` or `ip:port`)
/// so the router does not need DNS resolution to make decisions — the R3 proxy
/// resolves it when it actually connects.
#[derive(Debug, Clone)]
pub struct NodeState {
    /// Opaque node id.
    pub id: NodeId,
    /// Upstream address (`host:port`), used by the R3 proxy to connect.
    pub address: String,
    /// Whether the node is currently considered routable.
    pub healthy: bool,
    /// KV cache utilization, 0.0..=1.0.
    pub kv_usage: f32,
    /// Number of requests waiting to be admitted.
    pub queue_depth: u32,
    /// Number of sessions currently resident on the node.
    pub active_sessions: u32,
    /// Recent decode throughput (tokens/sec).
    pub tokens_per_second: f64,
    /// When the last successful poll landed (`None` until first poll).
    pub last_poll: Option<Instant>,
    /// Consecutive failed/missed polls since the last success.
    pub consecutive_misses: u32,
}

impl NodeState {
    /// Create a fresh node that has not yet been polled.
    ///
    /// A node starts `healthy = true` and is demoted only after
    /// `unhealthy_after_misses` consecutive missed polls or when a poll
    /// explicitly reports `healthy = false`.
    pub fn new(id: impl Into<NodeId>, address: impl Into<String>) -> Self {
        NodeState {
            id: id.into(),
            address: address.into(),
            healthy: true,
            kv_usage: 0.0,
            queue_depth: 0,
            active_sessions: 0,
            tokens_per_second: 0.0,
            last_poll: None,
            consecutive_misses: 0,
        }
    }

    /// Apply a freshly fetched `/v1/status` payload.
    ///
    /// Pure state transition (aside from reading the current `Instant`): a
    /// successful poll clears the miss counter and refreshes the load signals.
    pub fn apply_status(&mut self, status: NodeStatus) {
        self.healthy = status.healthy;
        self.kv_usage = status.kv_usage;
        self.queue_depth = status.queue_depth;
        self.active_sessions = status.active_sessions;
        self.tokens_per_second = status.tokens_per_second;
        self.last_poll = Some(Instant::now());
        self.consecutive_misses = 0;
    }

    /// Same as [`NodeState::apply_status`] but with an injectable timestamp, so
    /// tests (and R3) can drive the clock deterministically.
    pub fn apply_status_at(&mut self, status: NodeStatus, now: Instant) {
        self.apply_status(status);
        self.last_poll = Some(now);
    }

    /// Record a missed/failed poll.
    ///
    /// After `unhealthy_after_misses` consecutive misses the node is marked
    /// unhealthy and drops out of routing until a successful poll restores it.
    /// Returns `true` if this call flipped the node to unhealthy.
    pub fn record_miss(&mut self, unhealthy_after_misses: u32) -> bool {
        self.consecutive_misses = self.consecutive_misses.saturating_add(1);
        let was_healthy = self.healthy;
        if self.consecutive_misses >= unhealthy_after_misses {
            self.healthy = false;
        }
        was_healthy && !self.healthy
    }

    /// Whether this node can accept a session that wants affinity, given the
    /// overload threshold (`kv_usage` strictly below the threshold).
    pub fn accepts_affinity(&self, overload_threshold: f32) -> bool {
        self.healthy && self.kv_usage < overload_threshold
    }
}

/// Deserialization mirror of the inference server's `GET /v1/status` response
/// (see `docs/DESIGN.md` §34.8).
///
/// This is intentionally a **copy** of the server's contract rather than a
/// shared type: the router must not depend on the server/engine crates. If the
/// two ever drift, a shared `onnx-genai-node-contract` crate can be extracted
/// (tracked in `.squad/decisions`). Unknown fields are ignored and most
/// numeric fields default so the router degrades gracefully across versions.
#[derive(Debug, Clone, Deserialize)]
pub struct NodeStatus {
    /// Opaque node id the router keys on.
    pub node_id: String,
    /// Node self-reported health.
    #[serde(default = "default_true")]
    pub healthy: bool,
    /// KV cache utilization, 0.0..=1.0.
    #[serde(default)]
    pub kv_usage: f32,
    #[serde(default)]
    pub kv_pages_used: u32,
    #[serde(default)]
    pub kv_pages_total: u32,
    #[serde(default)]
    pub kv_pages_shared: u32,
    #[serde(default)]
    pub queue_depth: u32,
    #[serde(default)]
    pub active_sessions: u32,
    #[serde(default)]
    pub paused_sessions: u32,
    #[serde(default)]
    pub tokens_per_second: f64,
    #[serde(default)]
    pub batch_utilization: f32,
    /// Per-session summaries the node currently holds (opaque to the router).
    #[serde(default)]
    pub sessions: Vec<SessionSummary>,
    /// Prefix hashes resident on the node (hex strings, opaque to the router).
    #[serde(default)]
    pub prefix_hashes: Vec<String>,
}

/// One entry of the `/v1/status` `sessions` array. Fields are opaque to the
/// router but preserved for observability / future rebalancing heuristics.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionSummary {
    pub id: String,
    #[serde(default)]
    pub priority: String,
    #[serde(default)]
    pub kv_pages: u32,
    #[serde(default)]
    pub state: String,
}

fn default_true() -> bool {
    true
}

/// Async seam for fetching a node's status.
///
/// The pure core never performs I/O. R3's background poller implements this
/// trait (e.g. over `hyper`) and feeds the returned [`NodeStatus`] into
/// [`crate::router::Router::update_node`]. Kept here as a documented interface
/// so the proxy milestone can slot in cleanly.
pub trait NodeStatusFetcher: Send + Sync {
    /// Fetch `/v1/status` from the node at `address`. Implementations should
    /// treat timeouts/connection errors as a miss (return `Err`).
    fn fetch(
        &self,
        address: &str,
    ) -> impl std::future::Future<Output = Result<NodeStatus, FetchError>> + Send;
}

/// Error returned by a [`NodeStatusFetcher`]; a fetch error counts as a missed
/// poll for health tracking.
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    /// The node did not respond in time.
    #[error("timed out polling node at {0}")]
    Timeout(String),
    /// Transport or protocol error while polling.
    #[error("transport error polling node at {address}: {source}")]
    Transport {
        address: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The node responded but the body could not be parsed as [`NodeStatus`].
    #[error("failed to decode /v1/status from {address}: {source}")]
    Decode {
        address: String,
        #[source]
        source: serde_json::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(node_id: &str, healthy: bool, kv: f32, queue: u32) -> NodeStatus {
        NodeStatus {
            node_id: node_id.to_string(),
            healthy,
            kv_usage: kv,
            kv_pages_used: 0,
            kv_pages_total: 0,
            kv_pages_shared: 0,
            queue_depth: queue,
            active_sessions: 0,
            paused_sessions: 0,
            tokens_per_second: 0.0,
            batch_utilization: 0.0,
            sessions: vec![],
            prefix_hashes: vec![],
        }
    }

    #[test]
    fn deserializes_full_status_contract() {
        // The exact JSON shape from DESIGN.md §34.8.
        let json = r#"{
            "node_id": "gpu-2",
            "healthy": true,
            "kv_usage": 0.73,
            "kv_pages_used": 1496,
            "kv_pages_total": 2048,
            "kv_pages_shared": 256,
            "queue_depth": 2,
            "active_sessions": 8,
            "paused_sessions": 3,
            "tokens_per_second": 167.6,
            "batch_utilization": 0.82,
            "sessions": [
                { "id": "agent-worker-3", "priority": "standard", "kv_pages": 64, "state": "paused" },
                { "id": "agent-worker-7", "priority": "interactive", "kv_pages": 128, "state": "decoding" }
            ],
            "prefix_hashes": ["a1b2c3d4", "e5f6a7b8"]
        }"#;
        let s: NodeStatus = serde_json::from_str(json).expect("parse status");
        assert_eq!(s.node_id, "gpu-2");
        assert!(s.healthy);
        assert_eq!(s.kv_pages_total, 2048);
        assert_eq!(s.sessions.len(), 2);
        assert_eq!(s.sessions[1].state, "decoding");
        assert_eq!(s.prefix_hashes, vec!["a1b2c3d4", "e5f6a7b8"]);
    }

    #[test]
    fn deserializes_minimal_status_with_defaults() {
        // Forward/backward tolerant: only node_id is required.
        let s: NodeStatus = serde_json::from_str(r#"{ "node_id": "n1" }"#).unwrap();
        assert_eq!(s.node_id, "n1");
        assert!(s.healthy); // defaults to true
        assert_eq!(s.kv_usage, 0.0);
        assert!(s.sessions.is_empty());
    }

    #[test]
    fn apply_status_refreshes_and_clears_misses() {
        let mut n = NodeState::new("gpu-0", "10.0.0.1:8000");
        n.consecutive_misses = 2;
        n.healthy = false;
        n.apply_status(status("gpu-0", true, 0.5, 4));
        assert!(n.healthy);
        assert_eq!(n.kv_usage, 0.5);
        assert_eq!(n.queue_depth, 4);
        assert_eq!(n.consecutive_misses, 0);
        assert!(n.last_poll.is_some());
    }

    #[test]
    fn record_miss_marks_unhealthy_after_n() {
        let mut n = NodeState::new("gpu-0", "10.0.0.1:8000");
        assert!(!n.record_miss(3)); // 1
        assert!(n.healthy);
        assert!(!n.record_miss(3)); // 2
        assert!(n.healthy);
        let flipped = n.record_miss(3); // 3 -> unhealthy
        assert!(flipped);
        assert!(!n.healthy);
        // Further misses stay unhealthy but don't re-report the flip.
        assert!(!n.record_miss(3));
        assert!(!n.healthy);
    }

    #[test]
    fn record_miss_reports_flip_only_once() {
        let mut m = NodeState::new("n", "a:1");
        assert!(m.record_miss(1)); // threshold 1 flips immediately
        assert!(!m.record_miss(1)); // already unhealthy, no second flip
    }
}
