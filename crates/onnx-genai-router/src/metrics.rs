//! Prometheus metrics for the router (see `docs/DESIGN.md` §34.12).
//!
//! Following the inference server's convention, metrics are a hand-rolled
//! atomic/`Mutex` registry rendered to Prometheus text exposition — the repo
//! does not pull the `prometheus` crate, so neither does the router.
//!
//! Only `requests_total{node,decision}` is *stored* here (it is the one signal
//! not already present in the routing core). Migration counts, re-prefill
//! tokens, and the node/health gauges are **derived from the live [`Router`]
//! snapshot** at scrape time via [`Metrics::encode`], so there is a single
//! source of truth and no double counting.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Mutex;

use crate::router::{Router, RoutingDecision};
use crate::session_map::MigrationReason;

/// Stringly label for a [`RoutingDecision`] (the §34.12 `decision` label).
pub fn decision_label(decision: RoutingDecision) -> &'static str {
    match decision {
        RoutingDecision::Affinity => "affinity",
        RoutingDecision::Prefix => "prefix",
        RoutingDecision::LeastLoaded => "least_loaded",
    }
}

fn migration_reason_label(reason: MigrationReason) -> &'static str {
    match reason {
        MigrationReason::NodeDown => "node_down",
        MigrationReason::Overloaded => "overloaded",
        MigrationReason::Rebalance => "rebalance",
    }
}

/// Router metrics registry.
#[derive(Default)]
pub struct Metrics {
    /// `onnx_genai_router_requests_total{node,decision}`. Keyed by
    /// `(node_id, decision_label)`; `BTreeMap` keeps the exposition stable.
    requests_total: Mutex<BTreeMap<(String, &'static str), u64>>,
}

impl Metrics {
    /// Create an empty registry.
    pub fn new() -> Self {
        Metrics::default()
    }

    /// Record one routed request to `node` decided via `decision`.
    pub fn record_request(&self, node: &str, decision: RoutingDecision) {
        let mut guard = self.requests_total.lock().expect("metrics mutex poisoned");
        *guard
            .entry((node.to_string(), decision_label(decision)))
            .or_insert(0) += 1;
    }

    /// Render the full Prometheus exposition. Reads the live router snapshot for
    /// the derived counters/gauges (§34.12). The caller passes an already-locked
    /// `&Router` so metrics never take the router lock themselves.
    pub fn encode(&self, router: &Router) -> String {
        let mut out = String::with_capacity(2048);

        // --- requests_total (stored) ---
        out.push_str(
            "# HELP onnx_genai_router_requests_total Total routed requests by node and decision.\n",
        );
        out.push_str("# TYPE onnx_genai_router_requests_total counter\n");
        {
            let guard = self.requests_total.lock().expect("metrics mutex poisoned");
            for ((node, decision), value) in guard.iter() {
                writeln!(
                    out,
                    "onnx_genai_router_requests_total{{node=\"{}\",decision=\"{decision}\"}} {value}",
                    escape(node)
                )
                .expect("write to String");
            }
        }

        // --- migrations + reprefill (derived from the session-map log) ---
        let mut migrations: BTreeMap<&'static str, u64> = BTreeMap::new();
        let mut reprefill_tokens: u64 = 0;
        for ev in router.session_map().migrations() {
            *migrations
                .entry(migration_reason_label(ev.reason))
                .or_insert(0) += 1;
            reprefill_tokens = reprefill_tokens.saturating_add(ev.reprefill_tokens);
        }
        out.push_str(
            "# HELP onnx_genai_router_session_migrations_total Session migrations by reason.\n",
        );
        out.push_str("# TYPE onnx_genai_router_session_migrations_total counter\n");
        for (reason, value) in migrations.iter() {
            writeln!(
                out,
                "onnx_genai_router_session_migrations_total{{reason=\"{reason}\"}} {value}"
            )
            .expect("write to String");
        }
        out.push_str(
            "# HELP onnx_genai_router_reprefill_tokens_total Tokens re-prefilled after migration.\n",
        );
        out.push_str("# TYPE onnx_genai_router_reprefill_tokens_total counter\n");
        writeln!(
            out,
            "onnx_genai_router_reprefill_tokens_total {reprefill_tokens}"
        )
        .expect("write to String");

        // --- node gauges (derived from live node state) ---
        out.push_str("# HELP onnx_genai_router_node_healthy Node health (1 = healthy).\n");
        out.push_str("# TYPE onnx_genai_router_node_healthy gauge\n");
        for node in router.nodes() {
            writeln!(
                out,
                "onnx_genai_router_node_healthy{{node=\"{}\"}} {}",
                escape(node.id.as_str()),
                u8::from(node.healthy)
            )
            .expect("write to String");
        }
        out.push_str("# HELP onnx_genai_router_node_kv_usage Node KV cache utilization (0..1).\n");
        out.push_str("# TYPE onnx_genai_router_node_kv_usage gauge\n");
        for node in router.nodes() {
            writeln!(
                out,
                "onnx_genai_router_node_kv_usage{{node=\"{}\"}} {}",
                escape(node.id.as_str()),
                node.kv_usage
            )
            .expect("write to String");
        }
        out.push_str("# HELP onnx_genai_router_node_queue_depth Node admission queue depth.\n");
        out.push_str("# TYPE onnx_genai_router_node_queue_depth gauge\n");
        for node in router.nodes() {
            writeln!(
                out,
                "onnx_genai_router_node_queue_depth{{node=\"{}\"}} {}",
                escape(node.id.as_str()),
                node.queue_depth
            )
            .expect("write to String");
        }

        // --- router internals ---
        out.push_str(
            "# HELP onnx_genai_router_session_map_size Entries in the session->node map.\n",
        );
        out.push_str("# TYPE onnx_genai_router_session_map_size gauge\n");
        writeln!(
            out,
            "onnx_genai_router_session_map_size {}",
            router.session_map().len()
        )
        .expect("write to String");
        out.push_str("# HELP onnx_genai_router_prefix_map_size Entries in the prefix->node map.\n");
        out.push_str("# TYPE onnx_genai_router_prefix_map_size gauge\n");
        writeln!(
            out,
            "onnx_genai_router_prefix_map_size {}",
            router.prefix_map().len()
        )
        .expect("write to String");

        out
    }
}

/// Escape a Prometheus label value (backslash, double-quote, newline).
fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RoutingPolicy;
    use crate::node::{NodeId, NodeState};
    use crate::router::{RouteRequest, Router};

    fn router_with_nodes() -> Router {
        let mut n0 = NodeState::new("gpu-0", "10.0.0.1:8000");
        n0.kv_usage = 0.5;
        n0.queue_depth = 2;
        let n1 = NodeState::new("gpu-1", "10.0.0.2:8000");
        Router::new(vec![n0, n1], RoutingPolicy::AffinityThenLoad)
    }

    #[test]
    fn encodes_expected_metric_families() {
        let m = Metrics::new();
        let router = router_with_nodes();
        m.record_request("gpu-0", RoutingDecision::LeastLoaded);
        m.record_request("gpu-0", RoutingDecision::Affinity);
        let text = m.encode(&router);
        assert!(text.contains(
            "onnx_genai_router_requests_total{node=\"gpu-0\",decision=\"least_loaded\"} 1"
        ));
        assert!(
            text.contains(
                "onnx_genai_router_requests_total{node=\"gpu-0\",decision=\"affinity\"} 1"
            )
        );
        assert!(text.contains("onnx_genai_router_node_healthy{node=\"gpu-0\"} 1"));
        assert!(text.contains("onnx_genai_router_node_kv_usage{node=\"gpu-0\"} 0.5"));
        assert!(text.contains("onnx_genai_router_node_queue_depth{node=\"gpu-0\"} 2"));
        assert!(text.contains("onnx_genai_router_reprefill_tokens_total 0"));
    }

    #[test]
    fn derives_migration_counts_from_session_map() {
        let mut router = router_with_nodes();
        // Pin s1 to gpu-1, then knock it out so the next route migrates.
        let req = RouteRequest {
            session_id: Some("s1".into()),
            system_prompt_hash: None,
        };
        let pinned = router.route(&req).unwrap();
        router.node_mut(&pinned).unwrap().healthy = false;
        router.route_decision(&req).unwrap();
        let text = Metrics::new().encode(&router);
        assert!(
            text.contains("onnx_genai_router_session_migrations_total{reason=\"node_down\"} 1")
        );
    }

    #[test]
    fn decision_labels_match_design() {
        assert_eq!(decision_label(RoutingDecision::Affinity), "affinity");
        assert_eq!(decision_label(RoutingDecision::Prefix), "prefix");
        assert_eq!(decision_label(RoutingDecision::LeastLoaded), "least_loaded");
    }

    #[test]
    fn label_values_are_escaped() {
        let m = Metrics::new();
        let router = Router::new(
            vec![NodeState::new(NodeId::new("we\"ird"), "a:1")],
            RoutingPolicy::AffinityThenLoad,
        );
        m.record_request("we\"ird", RoutingDecision::Affinity);
        let text = m.encode(&router);
        assert!(text.contains("node=\"we\\\"ird\""));
    }
}
