//! Background node poller: periodically fetches each node's `GET /v1/status`
//! and feeds the result into the shared [`Router`] (see `docs/architecture/DESIGN.md` §34.8).
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

        parse_status(address, &body)
    }
}

/// Parses one `/v1/status` body, refusing a status that omits a field the
/// router steers on.
///
/// Separate from [`HttpStatusFetcher::fetch`] so it can be tested on raw bytes.
/// Testing the omission rule through a hand-built `NodeStatus` would prove
/// nothing: by the time a `NodeStatus` exists, serde has ALREADY substituted
/// the zero this function exists to prevent.
fn parse_status(address: &str, body: &[u8]) -> Result<NodeStatus, FetchError> {
    let raw: serde_json::Value =
        serde_json::from_slice(body).map_err(|source| FetchError::Decode {
            address: address.to_string(),
            source,
        })?;

    // An omitted load-bearing field is a MISSING MEASUREMENT, and serde's
    // `default` turns it into `0` -- the single most attractive value the
    // scorer can see. Treated as a miss instead, reusing the health machinery
    // that already exists for a node that did not answer at all.
    if let Some(field) = missing_load_bearing_field(&raw) {
        return Err(FetchError::Decode {
            address: address.to_string(),
            source: serde::de::Error::custom(format!(
                "`{field}` is absent from /v1/status; a node that cannot \
                 measure itself is counted as a missed poll, because \
                 defaulting it to zero would make it the most attractive \
                 routing target on the fleet"
            )),
        });
    }

    serde_json::from_value(raw).map_err(|source| FetchError::Decode {
        address: address.to_string(),
        source,
    })
}

/// Status fields the router actually STEERS ON.
///
/// Deliberately NOT every field. `tokens_per_second`, `batch_utilization` and
/// the session arrays are observability: losing them costs a graph, and
/// refusing a whole status over one of them would take a node out of service
/// for a field no routing decision reads. These two are different --
/// `kv_usage` gates the overload check AND prefix affinity, `queue_depth` is
/// the other scoring term -- so a zero in either does not degrade routing, it
/// INVERTS it.
const LOAD_BEARING_STATUS_FIELDS: [&str; 2] = ["kv_usage", "queue_depth"];

/// The first load-bearing field this status does not actually report.
///
/// `null` counts as absent: it deserialises to the same default as omission,
/// so treating it differently here would leave the identical hole open under a
/// different spelling.
fn missing_load_bearing_field(raw: &serde_json::Value) -> Option<&'static str> {
    let object = raw.as_object()?;
    LOAD_BEARING_STATUS_FIELDS
        .into_iter()
        .find(|field| !object.get(*field).is_some_and(|value| !value.is_null()))
}

/// Run a single poll pass over every configured node. Exposed (not just used by
/// [`run`]) so tests can drive one deterministic sweep at a time.
///
/// Fetches fan out **concurrently**: we snapshot every node's `(id, address)`
/// under one short lock, release it, then drive all `fetch` futures at once via
/// [`futures_util::future::join_all`]. This keeps a single overrunning (hung) node from serializing
/// the whole sweep — with `MissedTickBehavior::Delay`, a serial sweep that took
/// K×timeout would delay the next tick and staleness-degrade the *healthy*
/// nodes' health/load refresh exactly when the cluster is already degraded. Each
/// fetch keeps its own per-request timeout inside `fetcher.fetch`. Results are
/// applied under short locks; the `Mutex` is never held across an `.await`.
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

    // Fan every fetch out concurrently on this task. Each future borrows the
    // shared fetcher and carries its node id, so results can be applied (or a
    // miss counted) as they land. `join_all` polls them all on one task — no
    // spawning, so no `'static` bound is imposed on the borrowed fetcher.
    let fetches = targets.into_iter().map(|(id, address)| async move {
        let result = fetcher.fetch(&address).await;
        (id, result)
    });

    for (id, result) in futures_util::future::join_all(fetches).await {
        apply_poll_result(state, &id, result);
    }
}

/// Apply one node's poll outcome under a short lock.
///
/// A status whose self-reported `node_id` is not in config (`update_node`
/// returns `false`) is treated as a **miss** for the polled target: otherwise
/// an unrecognized-id response would leave the node in its initial healthy,
/// zero-load state forever, letting least-loaded routing favour a node we can't
/// actually observe.
fn apply_poll_result(state: &SharedState, id: &NodeId, result: Result<NodeStatus, FetchError>) {
    match result {
        Ok(status) => {
            let recognized = {
                let mut router = state.router.lock().expect("router mutex poisoned");
                router.update_node(status)
            };
            if !recognized {
                tracing::warn!(
                    node = %id,
                    "poll returned status for an unknown node id; counting as a miss"
                );
                let flipped = {
                    let mut router = state.router.lock().expect("router mutex poisoned");
                    router.record_node_miss(id)
                };
                if flipped {
                    tracing::warn!(node = %id, "node marked unhealthy after missed polls");
                }
            }
        }
        Err(err) => {
            let flipped = {
                let mut router = state.router.lock().expect("router mutex poisoned");
                router.record_node_miss(id)
            };
            if flipped {
                tracing::warn!(node = %id, error = %err, "node marked unhealthy after missed polls");
            } else {
                tracing::debug!(node = %id, error = %err, "missed node poll");
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
            match queue.and_then(|q| {
                if q.is_empty() {
                    None
                } else {
                    Some(q.remove(0))
                }
            }) {
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

    /// A poll returning a status whose `node_id` is not in config must be
    /// treated as a miss for the polled target, not silently dropped — else the
    /// node would keep its initial healthy, zero-load state and attract routing.
    #[tokio::test]
    async fn unrecognized_id_response_records_miss() {
        let state = state_with(&["10.0.0.1:8000"]);
        // Queue an Ok status whose node_id does NOT match the configured id.
        let status = NodeStatus {
            node_id: "some-other-node".to_string(),
            healthy: true,
            kv_usage: 0.0,
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
        let fetcher = StubFetcher::new();
        fetcher
            .script
            .lock()
            .unwrap()
            .entry("10.0.0.1:8000".to_string())
            .or_default()
            .push(Ok(status));

        poll_once(&state, &fetcher).await;

        let router = state.router.lock().unwrap();
        // The mismatched status was NOT applied and a miss WAS recorded.
        assert_eq!(router.nodes()[0].consecutive_misses, 1);
    }

    /// Fetcher that blocks each call until *all* expected calls are in flight,
    /// deterministically proving the sweep runs fetches concurrently: under a
    /// serial sweep the first call would never see the others start and would
    /// hang (caught by the surrounding timeout).
    struct BarrierFetcher {
        started: std::sync::atomic::AtomicUsize,
        expected: usize,
        // Addresses that should return an error instead of Ok.
        fail: std::collections::HashSet<String>,
    }

    impl BarrierFetcher {
        fn new(expected: usize, fail: &[&str]) -> Self {
            BarrierFetcher {
                started: std::sync::atomic::AtomicUsize::new(0),
                expected,
                fail: fail.iter().map(|s| s.to_string()).collect(),
            }
        }
    }

    impl NodeStatusFetcher for BarrierFetcher {
        async fn fetch(&self, address: &str) -> Result<NodeStatus, FetchError> {
            use std::sync::atomic::Ordering;
            self.started.fetch_add(1, Ordering::SeqCst);
            // Spin cooperatively until every fetch has started. Only possible
            // if the sweep polls them concurrently.
            while self.started.load(Ordering::SeqCst) < self.expected {
                tokio::task::yield_now().await;
            }
            if self.fail.contains(address) {
                return Err(FetchError::Timeout(address.to_string()));
            }
            let status = NodeStatus {
                node_id: address_to_id(address),
                healthy: true,
                kv_usage: 0.25,
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
            Ok(status)
        }
    }

    /// A single sweep fans the per-node fetches out concurrently: with one node
    /// erroring and two succeeding, all three are resolved in the same sweep
    /// (the healthy two update, the failing one records a miss). The barrier
    /// fetcher would deadlock under a serial sweep, so the enclosing timeout
    /// guards against a regression to sequential polling.
    #[tokio::test]
    async fn poll_once_fans_out_concurrently() {
        let state = state_with(&["10.0.0.1:8000", "10.0.0.2:8000", "10.0.0.3:8000"]);
        let fetcher = BarrierFetcher::new(3, &["10.0.0.2:8000"]);

        tokio::time::timeout(Duration::from_secs(5), poll_once(&state, &fetcher))
            .await
            .expect("concurrent sweep must complete; serial polling would deadlock the barrier");

        let router = state.router.lock().unwrap();
        let by_id = |id: &str| router.nodes().iter().find(|n| n.id.as_str() == id).unwrap();
        // The two healthy nodes were updated in the same sweep.
        assert!((by_id("10.0.0.1-8000").kv_usage - 0.25).abs() < 1e-6);
        assert_eq!(by_id("10.0.0.1-8000").consecutive_misses, 0);
        assert!((by_id("10.0.0.3-8000").kv_usage - 0.25).abs() < 1e-6);
        assert_eq!(by_id("10.0.0.3-8000").consecutive_misses, 0);
        // The failing node recorded a miss.
        assert_eq!(by_id("10.0.0.2-8000").consecutive_misses, 1);
    }

    /// A status that omits `kv_usage` must NOT be accepted as `kv_usage: 0.0`.
    ///
    /// Zero is below every overload threshold and is the minimum of any
    /// candidate set, so the defaulted value does not merely lose information
    /// -- it makes the node that cannot measure itself the most attractive
    /// target on the fleet. Losing telemetry would attract load.
    #[test]
    fn an_omitted_load_bearing_field_is_not_a_zero_reading() {
        for field in ["kv_usage", "queue_depth"] {
            let mut body = serde_json::json!({
                "node_id": "n1",
                "healthy": true,
                "kv_usage": 0.9,
                "queue_depth": 40,
            });
            body.as_object_mut().unwrap().remove(field);
            assert_eq!(
                missing_load_bearing_field(&body),
                Some(field),
                "omitting {field} must be reported as a missing measurement"
            );
        }
    }

    /// `null` deserialises to the same default as omission.
    #[test]
    fn a_null_load_bearing_field_is_also_missing() {
        let body = serde_json::json!({
            "node_id": "n1",
            "kv_usage": serde_json::Value::Null,
            "queue_depth": 0,
        });
        assert_eq!(missing_load_bearing_field(&body), Some("kv_usage"));
    }

    /// The control that keeps this fix from becoming an outage.
    ///
    /// Observability fields are NOT load-bearing: refusing a whole status
    /// because a graph is missing would take a healthy node out of service
    /// over a field no routing decision reads.
    #[test]
    fn a_status_missing_only_observability_fields_is_still_accepted() {
        let body = serde_json::json!({
            "node_id": "n1",
            "kv_usage": 0.5,
            "queue_depth": 2,
        });
        assert_eq!(
            missing_load_bearing_field(&body),
            None,
            "tokens_per_second, batch_utilization and the session arrays are \
             absent here and none of them steer routing"
        );
        let status: NodeStatus =
            serde_json::from_value(body).expect("the status still deserialises");
        assert!((status.kv_usage - 0.5).abs() < 1e-6);
        assert_eq!(status.queue_depth, 2);
    }

    /// End-to-end over the REAL parse path, on raw bytes.
    ///
    /// The unit tests above call the predicate directly, which cannot see
    /// whether it is wired into anything. This one goes through the function
    /// `fetch` actually calls, so deleting the check reddens it.
    #[test]
    fn a_body_omitting_kv_usage_is_refused_by_the_parser() {
        let body = br#"{"node_id":"n1","healthy":true,"queue_depth":3}"#;
        let error = parse_status("10.0.0.1:8000", body)
            .expect_err("a status with no kv_usage must not parse into kv_usage: 0.0");
        let rendered = error.to_string();
        assert!(
            rendered.contains("kv_usage"),
            "the refusal must name the field: {rendered}"
        );

        // The whole point: serde WOULD have accepted it, silently, as the most
        // attractive score on the fleet.
        let defaulted: NodeStatus =
            serde_json::from_slice(body).expect("serde accepts it by design");
        assert_eq!(
            defaulted.kv_usage, 0.0,
            "this is the value the router would have steered on"
        );
    }

    /// The control at the same level: a complete body still parses.
    #[test]
    fn a_complete_body_still_parses() {
        let body = br#"{"node_id":"n1","kv_usage":0.75,"queue_depth":9}"#;
        let status = parse_status("10.0.0.1:8000", body).expect("a complete status is accepted");
        assert_eq!(status.node_id, "n1");
        assert!((status.kv_usage - 0.75).abs() < 1e-6);
        assert_eq!(status.queue_depth, 9);
    }

    /// A zero that the node ACTUALLY REPORTED is a measurement and must be
    /// honoured -- an idle node is genuinely the best target.
    #[test]
    fn an_explicit_zero_is_a_measurement_not_a_miss() {
        let body = serde_json::json!({
            "node_id": "n1",
            "kv_usage": 0.0,
            "queue_depth": 0,
        });
        assert_eq!(
            missing_load_bearing_field(&body),
            None,
            "the node said zero; that is the honest best score, not an absence"
        );
    }
}
