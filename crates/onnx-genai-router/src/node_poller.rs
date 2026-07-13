//! Background node poller: periodically fetches each node's `GET /v1/status`
//! and feeds the result into the shared [`Router`] (see `docs/DESIGN.md` §34.8).
//!
//! The [`NodeStatusFetcher`] seam from [`crate::node`] is implemented here by
//! [`HttpStatusFetcher`] over a `hyper-util` legacy client. The loop itself is
//! generic over the fetcher so tests can drive it with an in-memory stub and no
//! real network (see the module tests).
//!
//! Lock discipline: the shared `Router` `Mutex` is only ever held for short,
//! synchronous sections (snapshot the addresses, or apply one status/miss). It
//! is **never** held across the `.await` on `fetch`.

use std::time::Duration;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Method, Request};
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;

use crate::node::{FetchError, NodeId, NodeStatus, NodeStatusFetcher};
use crate::state::SharedState;

/// Shared hyper client body type: a fully-buffered body. Status polls send an
/// empty body; the proxy reuses the same client type with a real body.
pub type ProxyClient = Client<HttpConnector, Full<Bytes>>;

/// Build the shared hyper-util client used by both the poller and the proxy.
pub fn build_client() -> ProxyClient {
    Client::builder(TokioExecutor::new()).build_http()
}

/// [`NodeStatusFetcher`] backed by a hyper-util HTTP client with a per-poll
/// timeout. A timeout or transport/decoding failure is surfaced as a
/// [`FetchError`], which the poll loop treats as a missed poll.
pub struct HttpStatusFetcher {
    client: ProxyClient,
    timeout: Duration,
}

impl HttpStatusFetcher {
    /// Create a fetcher with the given per-request timeout.
    pub fn new(client: ProxyClient, timeout: Duration) -> Self {
        HttpStatusFetcher { client, timeout }
    }
}

impl NodeStatusFetcher for HttpStatusFetcher {
    async fn fetch(&self, address: &str) -> Result<NodeStatus, FetchError> {
        let uri = format!("http://{address}/v1/status");
        let request = Request::builder()
            .method(Method::GET)
            .uri(&uri)
            .body(Full::new(Bytes::new()))
            .map_err(|source| FetchError::Transport {
                address: address.to_string(),
                source: Box::new(source),
            })?;

        let response = tokio::time::timeout(self.timeout, self.client.request(request))
            .await
            .map_err(|_| FetchError::Timeout(address.to_string()))?
            .map_err(|source| FetchError::Transport {
                address: address.to_string(),
                source: Box::new(source),
            })?;

        let body = response
            .into_body()
            .collect()
            .await
            .map_err(|source| FetchError::Transport {
                address: address.to_string(),
                source: Box::new(source),
            })?
            .to_bytes();

        serde_json::from_slice(&body).map_err(|source| FetchError::Decode {
            address: address.to_string(),
            source,
        })
    }
}

/// Run a single poll pass over every configured node. Exposed (not just used by
/// [`run`]) so tests can drive one deterministic sweep at a time.
pub async fn poll_once<F: NodeStatusFetcher>(state: &SharedState, fetcher: &F) {
    // Snapshot (id, address) under a short lock, then release before any I/O.
    let targets: Vec<(NodeId, String)> = {
        let router = state.router.lock().expect("router mutex poisoned");
        router
            .nodes()
            .iter()
            .map(|n| (n.id.clone(), n.address.clone()))
            .collect()
    };

    for (id, address) in targets {
        match fetcher.fetch(&address).await {
            Ok(status) => {
                let mut router = state.router.lock().expect("router mutex poisoned");
                if !router.update_node(status) {
                    tracing::warn!(node = %id, "poll returned status for an unknown node id");
                }
            }
            Err(err) => {
                let flipped = {
                    let mut router = state.router.lock().expect("router mutex poisoned");
                    router.record_node_miss(&id)
                };
                if flipped {
                    tracing::warn!(node = %id, error = %err, "node marked unhealthy after missed polls");
                } else {
                    tracing::debug!(node = %id, error = %err, "missed node poll");
                }
            }
        }
    }
}

/// Poll loop: sweep every node every `poll_interval_ms` until cancelled.
///
/// The `tokio::interval` schedules ticks; each tick runs a full [`poll_once`]
/// sweep. Runs forever — spawn it as a task and drop/abort it on shutdown.
pub async fn run<F: NodeStatusFetcher>(state: SharedState, fetcher: F) {
    let period = Duration::from_millis(state.poll_interval_ms.max(1));
    let mut ticker = tokio::time::interval(period);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        ticker.tick().await;
        poll_once(&state, &fetcher).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    use crate::config::RoutingPolicy;
    use crate::node::NodeState;
    use crate::router::Router;
    use crate::state::AppState;

    /// In-memory fetcher: returns queued results per address, no network.
    struct StubFetcher {
        // Per-address queue of poll outcomes (Ok(status) or Err(miss)).
        script: StdMutex<std::collections::HashMap<String, Vec<Result<NodeStatus, ()>>>>,
    }

    impl StubFetcher {
        fn new() -> Self {
            StubFetcher {
                script: StdMutex::new(std::collections::HashMap::new()),
            }
        }

        fn push_ok(&self, address: &str, kv_usage: f32) {
            let status = NodeStatus {
                node_id: address_to_id(address),
                healthy: true,
                kv_usage,
                kv_pages_used: 0,
                kv_pages_total: 0,
                kv_pages_shared: 0,
                queue_depth: 0,
                active_sessions: 0,
                paused_sessions: 0,
                tokens_per_second: 0.0,
                batch_utilization: 0.0,
                sessions: vec![],
                prefix_hashes: vec![],
            };
            self.script
                .lock()
                .unwrap()
                .entry(address.to_string())
                .or_default()
                .push(Ok(status));
        }

        fn push_miss(&self, address: &str) {
            self.script
                .lock()
                .unwrap()
                .entry(address.to_string())
                .or_default()
                .push(Err(()));
        }
    }

    // Test node ids are derived from their address for convenience.
    fn address_to_id(address: &str) -> String {
        address.replace(':', "-")
    }

    impl NodeStatusFetcher for StubFetcher {
        async fn fetch(&self, address: &str) -> Result<NodeStatus, FetchError> {
            let mut script = self.script.lock().unwrap();
            let queue = script.get_mut(address);
            match queue.and_then(|q| if q.is_empty() { None } else { Some(q.remove(0)) }) {
                Some(Ok(status)) => Ok(status),
                _ => Err(FetchError::Timeout(address.to_string())),
            }
        }
    }

    fn state_with(addresses: &[&str]) -> SharedState {
        let nodes = addresses
            .iter()
            .map(|a| NodeState::new(address_to_id(a), *a))
            .collect();
        let router = Router::new(nodes, RoutingPolicy::AffinityThenLoad);
        AppState::new(router, 10)
    }

    #[tokio::test]
    async fn poll_once_applies_status() {
        let state = state_with(&["10.0.0.1:8000"]);
        let fetcher = StubFetcher::new();
        fetcher.push_ok("10.0.0.1:8000", 0.42);
        poll_once(&state, &fetcher).await;
        let router = state.router.lock().unwrap();
        let node = &router.nodes()[0];
        assert!((node.kv_usage - 0.42).abs() < 1e-6);
        assert_eq!(node.consecutive_misses, 0);
        assert!(node.healthy);
    }

    #[tokio::test]
    async fn repeated_misses_mark_node_unhealthy() {
        let state = state_with(&["10.0.0.1:8000"]);
        let fetcher = StubFetcher::new();
        // Default unhealthy_after_misses = 3.
        for _ in 0..3 {
            fetcher.push_miss("10.0.0.1:8000");
        }
        for _ in 0..3 {
            poll_once(&state, &fetcher).await;
        }
        let router = state.router.lock().unwrap();
        assert!(!router.nodes()[0].healthy);
        assert_eq!(router.nodes()[0].consecutive_misses, 3);
    }

    #[tokio::test]
    async fn successful_poll_recovers_after_misses() {
        let state = state_with(&["10.0.0.1:8000"]);
        let fetcher = StubFetcher::new();
        fetcher.push_miss("10.0.0.1:8000");
        fetcher.push_miss("10.0.0.1:8000");
        fetcher.push_miss("10.0.0.1:8000"); // -> unhealthy
        fetcher.push_ok("10.0.0.1:8000", 0.1); // -> recovers
        for _ in 0..4 {
            poll_once(&state, &fetcher).await;
        }
        let router = state.router.lock().unwrap();
        assert!(router.nodes()[0].healthy);
        assert_eq!(router.nodes()[0].consecutive_misses, 0);
    }
}
