//! Core routing logic (see `docs/DESIGN.md` §34.4 and §34.5).
//!
//! [`Router::route`] is a **pure decision function** over the current node
//! snapshot: given a [`RouteRequest`] it returns the chosen [`NodeId`] (or
//! `None` when no node can serve it). It performs no I/O. The R3 poller feeds
//! fresh state in via [`Router::update_node`] / [`Router::record_node_miss`],
//! and the R3 proxy consumes the returned node's `address`.
//!
//! Unlike the illustrative pseudo-code in the design (which uses
//! `.expect("no healthy nodes")`), this implementation returns `Option` and
//! never panics — a clean pre-release contract.

use crate::config::{RouterConfig, RoutingPolicy, WeightConfig};
use crate::node::{NodeId, NodeState, NodeStatus};
use crate::prefix_map::PrefixMap;
use crate::session_map::{MigrationReason, SessionMap};

use std::collections::HashSet;

/// KV-usage ceiling above which a node stops accepting prefix co-location
/// (design §34.5). Distinct from the affinity `overload_threshold` so we keep a
/// little headroom for prefix sharing.
const PREFIX_KV_THRESHOLD: f32 = 0.85;

/// A routing request. Deliberately minimal and model-agnostic: an opaque
/// session id (for affinity) and an opaque prefix hash (for co-location).
#[derive(Debug, Clone, Default)]
pub struct RouteRequest {
    /// Opaque session id, if the caller is continuing a conversation.
    pub session_id: Option<String>,
    /// Stable hash of the system prompt, if prefix co-location is desired.
    /// Produced by [`crate::prefix_map::hash_system_prompt`].
    pub system_prompt_hash: Option<u64>,
}

/// How a routing decision was reached (feeds the §34.12 `decision` label).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingDecision {
    /// Routed to the session's existing affinity node.
    Affinity,
    /// Routed to a node already holding the same prefix.
    Prefix,
    /// Routed to the least-loaded healthy node.
    LeastLoaded,
}

/// The session-aware router.
pub struct Router {
    nodes: Vec<NodeState>,
    session_map: SessionMap,
    prefix_map: PrefixMap,
    policy: RoutingPolicy,
    /// KV usage at/above which affinity breaks (config `overload_threshold`).
    overload_threshold: f32,
    /// Whether prefix co-location is enabled.
    prefix_colocate: bool,
    /// Missed polls before a node is demoted (config `unhealthy_after_misses`).
    unhealthy_after_misses: u32,
    /// Nodes being gracefully drained: they keep serving sessions that already
    /// have affinity, but accept no *new* sessions (via prefix or least-loaded).
    draining: HashSet<NodeId>,
}

impl Router {
    /// Build a router from an explicit set of nodes and a policy, using default
    /// thresholds (overload 0.95, prefix co-location on, 3-miss health).
    pub fn new(nodes: Vec<NodeState>, policy: RoutingPolicy) -> Self {
        Router {
            nodes,
            session_map: SessionMap::new(),
            prefix_map: PrefixMap::new(),
            policy,
            overload_threshold: 0.95,
            prefix_colocate: true,
            unhealthy_after_misses: 3,
            draining: HashSet::new(),
        }
    }

    /// Build a router from a validated [`RouterConfig`]. Nodes start healthy
    /// and un-polled; the R3 poller populates live state.
    pub fn from_config(config: &RouterConfig) -> Self {
        let nodes = config
            .nodes
            .iter()
            .map(|n| NodeState::new(n.name.clone(), n.address.clone()))
            .collect();
        Router {
            nodes,
            session_map: SessionMap::new(),
            prefix_map: PrefixMap::new(),
            policy: config.routing.policy.clone(),
            overload_threshold: config.routing.overload_threshold,
            prefix_colocate: config.routing.prefix_colocate,
            unhealthy_after_misses: config.health.unhealthy_after_misses,
            draining: HashSet::new(),
        }
    }

    /// Core routing decision. Returns the chosen node id, or `None` if there is
    /// no node able to serve the request (e.g. all nodes unhealthy).
    pub fn route(&mut self, request: &RouteRequest) -> Option<NodeId> {
        self.route_decision(request).map(|(node, _)| node)
    }

    /// Like [`Router::route`] but also returns *why* the node was chosen, for
    /// metrics. Records/updates session affinity and prefix co-location as a
    /// side effect (and a [`crate::session_map::MigrationEvent`] when affinity
    /// broke and the session moved).
    pub fn route_decision(
        &mut self,
        request: &RouteRequest,
    ) -> Option<(NodeId, RoutingDecision)> {
        let (node, decision) = self.pick(request)?;
        self.record_routing(request, &node);
        Some((node, decision))
    }

    /// Pure selection: no side effects, honors the policy ordering.
    fn pick(&self, request: &RouteRequest) -> Option<(NodeId, RoutingDecision)> {
        for step in self.policy_steps() {
            match step {
                Step::Affinity => {
                    if let Some(node) = self.try_affinity(request) {
                        return Some((node, RoutingDecision::Affinity));
                    }
                }
                Step::Prefix => {
                    if let Some(node) = self.try_prefix(request) {
                        return Some((node, RoutingDecision::Prefix));
                    }
                }
            }
        }
        self.weighted_fallback_node(request)
            .map(|node| (node, RoutingDecision::LeastLoaded))
    }

    /// Preference order of the primary (non-fallback) steps per policy.
    fn policy_steps(&self) -> &'static [Step] {
        match self.policy {
            RoutingPolicy::AffinityThenLoad => &[Step::Affinity, Step::Prefix],
            // Weighted scores affinity as a continuous bonus in the fallback;
            // no separate binary gate here.
            RoutingPolicy::Weighted(_) => &[Step::Prefix],
            RoutingPolicy::PrefixThenLoad => &[Step::Prefix, Step::Affinity],
            // Always route to least KV usage: skip affinity/prefix entirely.
            RoutingPolicy::LeastKvUsage => &[],
        }
    }

    fn try_affinity(&self, request: &RouteRequest) -> Option<NodeId> {
        let session_id = request.session_id.as_ref()?;
        let node = self.session_map.get(session_id)?.clone();
        let state = self.node_by_id(&node)?;
        if state.accepts_affinity(self.overload_threshold) {
            Some(node)
        } else {
            None
        }
    }

    fn try_prefix(&self, request: &RouteRequest) -> Option<NodeId> {
        if !self.prefix_colocate {
            return None;
        }
        let hash = request.system_prompt_hash?;
        let node = self.prefix_map.get(hash)?.clone();
        // A draining node accepts no new sessions, even for prefix sharing.
        if self.draining.contains(&node) {
            return None;
        }
        let state = self.node_by_id(&node)?;
        if state.healthy && state.kv_usage < PREFIX_KV_THRESHOLD {
            Some(node)
        } else {
            None
        }
    }

    /// Least-loaded healthy node under the active policy's load score.
    /// Draining nodes are excluded (they accept no new sessions). Returns `None`
    /// when no node is healthy and routable.
    pub fn least_loaded_node(&self) -> Option<NodeId> {
        self.nodes
            .iter()
            .filter(|n| n.healthy && !self.draining.contains(&n.id))
            .min_by(|a, b| {
                self.load_score(a)
                    .partial_cmp(&self.load_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|n| n.id.clone())
    }

    /// Fallback node selection used by [`Self::pick`].
    ///
    /// For [`RoutingPolicy::Weighted`] the session's current affinity node
    /// receives a score reduction of `affinity_weight`, making it "stickier"
    /// under moderate load (§34.5).  The bonus is withheld when the affinity
    /// node is at or above `overload_threshold` so a saturated node cannot win
    /// solely because of affinity.  Unhealthy and draining nodes are always
    /// excluded.
    ///
    /// For all other policies delegates to [`Self::least_loaded_node`].
    fn weighted_fallback_node(&self, request: &RouteRequest) -> Option<NodeId> {
        let RoutingPolicy::Weighted(w) = &self.policy else {
            return self.least_loaded_node();
        };
        let affinity_target = request
            .session_id
            .as_ref()
            .and_then(|s| self.session_map.get(s));
        self.nodes
            .iter()
            .filter(|n| n.healthy && !self.draining.contains(&n.id))
            .min_by(|a, b| {
                weighted_node_score(a, affinity_target, w, self.overload_threshold)
                    .partial_cmp(&weighted_node_score(b, affinity_target, w, self.overload_threshold))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|n| n.id.clone())
    }

    /// Lower is better. Scoring depends on policy (§34.5).
    fn load_score(&self, node: &NodeState) -> f32 {
        let normalized_queue = node.queue_depth as f32 / 10.0;
        match &self.policy {
            RoutingPolicy::LeastKvUsage => node.kv_usage,
            RoutingPolicy::Weighted(w) => {
                node.kv_usage * w.kv_weight + normalized_queue * w.queue_weight
            }
            RoutingPolicy::AffinityThenLoad | RoutingPolicy::PrefixThenLoad => {
                node.kv_usage * 0.6 + normalized_queue * 0.4
            }
        }
    }

    /// Record affinity/prefix mappings after a routing decision. If the session
    /// had a prior affinity to a *different* node, record a migration.
    fn record_routing(&mut self, request: &RouteRequest, node: &NodeId) {
        if let Some(session_id) = &request.session_id {
            match self.session_map.get(session_id).cloned() {
                Some(prev) if prev != *node => {
                    let reason = match self.node_by_id(&prev) {
                        Some(state) if !state.healthy => MigrationReason::NodeDown,
                        _ => MigrationReason::Overloaded,
                    };
                    self.session_map
                        .migrate(session_id.clone(), node.clone(), reason, 0);
                }
                _ => self.session_map.assign(session_id.clone(), node.clone()),
            }
        }
        if self.prefix_colocate && let Some(hash) = request.system_prompt_hash {
            self.prefix_map.assign(hash, node.clone());
        }
    }

    // ---- Node state management (driven by the R3 poller) -------------------

    /// Apply a freshly polled `/v1/status` for the matching node id. Returns
    /// `false` if the status is for an unknown node.
    pub fn update_node(&mut self, status: NodeStatus) -> bool {
        let id = NodeId::new(status.node_id.clone());
        match self.node_index(&id) {
            Some(idx) => {
                self.nodes[idx].apply_status(status);
                true
            }
            None => false,
        }
    }

    /// Record a missed poll for a node. Returns `true` if this flipped the node
    /// to unhealthy.
    pub fn record_node_miss(&mut self, id: &NodeId) -> bool {
        match self.node_index(id) {
            Some(idx) => self.nodes[idx].record_miss(self.unhealthy_after_misses),
            None => false,
        }
    }

    /// Immutable view of all nodes.
    pub fn nodes(&self) -> &[NodeState] {
        &self.nodes
    }

    // ---- Draining & rebalancing (driven by the R3 /router/* API) -----------

    /// Mark a node as draining (or clear it). A draining node keeps serving
    /// sessions that already have affinity to it, but is excluded from prefix
    /// and least-loaded selection so no *new* sessions land on it. Returns
    /// `false` if the node id is unknown.
    pub fn set_draining(&mut self, id: &NodeId, draining: bool) -> bool {
        if self.node_index(id).is_none() {
            return false;
        }
        if draining {
            self.draining.insert(id.clone());
        } else {
            self.draining.remove(id);
        }
        true
    }

    /// Whether a node is currently draining.
    pub fn is_draining(&self, id: &NodeId) -> bool {
        self.draining.contains(id)
    }

    /// Rebalance sessions off nodes that are unhealthy, draining, or at/above
    /// the overload threshold onto the current least-loaded node. Records a
    /// [`MigrationReason::Rebalance`] migration for each session actually moved
    /// and returns the number moved.
    ///
    /// Scope: this is the minimal operator-triggered reassignment described in
    /// §34.7. It reassigns affinity in the router's own table; the affected
    /// sessions re-prefill lazily on their next request (the node fleet is not
    /// notified out-of-band).
    pub fn rebalance(&mut self) -> usize {
        let pinned: Vec<(String, NodeId)> = self
            .session_map
            .iter()
            .map(|(s, n)| (s.clone(), n.clone()))
            .collect();
        let mut moved = 0;
        for (session_id, node) in pinned {
            let needs_move = match self.node_by_id(&node) {
                Some(state) => {
                    !state.healthy
                        || self.draining.contains(&node)
                        || state.kv_usage >= self.overload_threshold
                }
                None => true,
            };
            if !needs_move {
                continue;
            }
            if let Some(target) = self.least_loaded_node()
                && target != node
            {
                self.session_map
                    .migrate(session_id, target, MigrationReason::Rebalance, 0);
                moved += 1;
            }
        }
        moved
    }

    /// Look up a node by id.
    pub fn node_by_id(&self, id: &NodeId) -> Option<&NodeState> {
        self.node_index(id).map(|idx| &self.nodes[idx])
    }

    /// Mutable access to a node by id (used by the poller and by tests).
    pub fn node_mut(&mut self, id: &NodeId) -> Option<&mut NodeState> {
        self.node_index(id).map(|idx| &mut self.nodes[idx])
    }

    /// Number of currently healthy nodes.
    pub fn healthy_node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.healthy).count()
    }

    /// Read-only access to the session affinity table.
    pub fn session_map(&self) -> &SessionMap {
        &self.session_map
    }

    /// Record affinity for a session whose id was assigned out-of-band (e.g. by
    /// the node in a `POST /v1/sessions` response rather than carried in the
    /// request). A plain assignment — no migration accounting.
    pub fn record_session_affinity(&mut self, session_id: impl Into<String>, node: NodeId) {
        self.session_map.assign(session_id, node);
    }

    /// Read-only access to the prefix co-location map.
    pub fn prefix_map(&self) -> &PrefixMap {
        &self.prefix_map
    }

    fn node_index(&self, id: &NodeId) -> Option<usize> {
        self.nodes.iter().position(|n| &n.id == id)
    }
}

/// Score a node for the [`RoutingPolicy::Weighted`] fallback path.
///
/// `score = kv_usage × kv_weight + normalized_queue × queue_weight − bonus`
///
/// The `affinity_weight` bonus is applied only when the node is the session's
/// current affinity target *and* its KV usage is below `overload_threshold`.
/// Withholding the bonus for overloaded nodes prevents the affinity discount
/// from routing a request to a saturated node.
fn weighted_node_score(
    node: &NodeState,
    affinity_target: Option<&NodeId>,
    w: &WeightConfig,
    overload_threshold: f32,
) -> f32 {
    let normalized_queue = node.queue_depth as f32 / 10.0;
    let base = node.kv_usage * w.kv_weight + normalized_queue * w.queue_weight;
    let is_affinity_target = affinity_target == Some(&node.id);
    let bonus = if is_affinity_target && node.kv_usage < overload_threshold {
        w.affinity_weight
    } else {
        0.0
    };
    base - bonus
}

/// Primary routing steps (before the least-loaded fallback).
#[derive(Debug, Clone, Copy)]
enum Step {
    Affinity,
    Prefix,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WeightConfig;

    fn node(id: &str, healthy: bool, kv: f32, queue: u32) -> NodeState {
        let mut n = NodeState::new(id, format!("{id}:8000"));
        n.healthy = healthy;
        n.kv_usage = kv;
        n.queue_depth = queue;
        n
    }

    fn router(nodes: Vec<NodeState>, policy: RoutingPolicy) -> Router {
        Router::new(nodes, policy)
    }

    #[test]
    fn affinity_hit_returns_pinned_node() {
        let mut r = router(
            vec![node("gpu-0", true, 0.2, 0), node("gpu-1", true, 0.1, 0)],
            RoutingPolicy::AffinityThenLoad,
        );
        // First route (no affinity yet) records affinity to the least loaded.
        let req = RouteRequest {
            session_id: Some("s1".into()),
            system_prompt_hash: None,
        };
        let first = r.route_decision(&req).unwrap();
        assert_eq!(first.1, RoutingDecision::LeastLoaded);
        let pinned = first.0.clone();
        // Second route with the same session sticks to the pinned node.
        let second = r.route_decision(&req).unwrap();
        assert_eq!(second.0, pinned);
        assert_eq!(second.1, RoutingDecision::Affinity);
    }

    #[test]
    fn affinity_breaks_when_node_unhealthy() {
        let mut r = router(
            vec![node("gpu-0", true, 0.2, 0), node("gpu-1", true, 0.1, 0)],
            RoutingPolicy::AffinityThenLoad,
        );
        // Pin s1 to gpu-1 (least loaded).
        let req = RouteRequest {
            session_id: Some("s1".into()),
            system_prompt_hash: None,
        };
        let pinned = r.route(&req).unwrap();
        // Knock the pinned node offline.
        r.node_mut(&pinned).unwrap().healthy = false;
        let decision = r.route_decision(&req).unwrap();
        assert_ne!(decision.0, pinned);
        assert_eq!(decision.1, RoutingDecision::LeastLoaded);
        // A migration was recorded with NodeDown reason.
        let migrations = r.session_map().migrations();
        assert_eq!(migrations.len(), 1);
        assert_eq!(migrations[0].from_node, pinned);
        assert_eq!(migrations[0].reason, MigrationReason::NodeDown);
    }

    #[test]
    fn affinity_breaks_when_node_overloaded() {
        let mut r = router(
            vec![node("gpu-0", true, 0.2, 0), node("gpu-1", true, 0.1, 0)],
            RoutingPolicy::AffinityThenLoad,
        );
        let req = RouteRequest {
            session_id: Some("s1".into()),
            system_prompt_hash: None,
        };
        let pinned = r.route(&req).unwrap();
        // Push the pinned node above the overload threshold (0.95).
        r.node_mut(&pinned).unwrap().kv_usage = 0.99;
        let decision = r.route_decision(&req).unwrap();
        assert_ne!(decision.0, pinned);
        assert_eq!(decision.1, RoutingDecision::LeastLoaded);
        assert_eq!(
            r.session_map().migrations()[0].reason,
            MigrationReason::Overloaded
        );
    }

    #[test]
    fn prefix_colocation_routes_to_shared_node() {
        let mut r = router(
            vec![node("gpu-0", true, 0.5, 0), node("gpu-1", true, 0.1, 0)],
            RoutingPolicy::PrefixThenLoad,
        );
        let hash = crate::prefix_map::hash_system_prompt("shared prompt");
        // Seed a prefix on the busier node gpu-0.
        let req_a = RouteRequest {
            session_id: Some("s-a".into()),
            system_prompt_hash: Some(hash),
        };
        // First request goes to least loaded (gpu-1) and seeds prefix->gpu-1.
        let first = r.route_decision(&req_a).unwrap();
        assert_eq!(first.1, RoutingDecision::LeastLoaded);
        let colocated = first.0.clone();
        // A different session with the same prefix co-locates.
        let req_b = RouteRequest {
            session_id: Some("s-b".into()),
            system_prompt_hash: Some(hash),
        };
        let second = r.route_decision(&req_b).unwrap();
        assert_eq!(second.0, colocated);
        assert_eq!(second.1, RoutingDecision::Prefix);
    }

    #[test]
    fn prefix_skipped_when_target_overloaded() {
        let mut r = router(
            vec![node("gpu-0", true, 0.1, 0), node("gpu-1", true, 0.1, 0)],
            RoutingPolicy::PrefixThenLoad,
        );
        let hash = 42u64;
        // Manually pin the prefix to gpu-0, then overload gpu-0 beyond 0.85.
        let req = RouteRequest {
            session_id: None,
            system_prompt_hash: Some(hash),
        };
        let first = r.route(&req).unwrap();
        r.node_mut(&first).unwrap().kv_usage = 0.9;
        let decision = r.route_decision(&req).unwrap();
        // Prefix target too full -> falls back to least loaded (the other node).
        assert_eq!(decision.1, RoutingDecision::LeastLoaded);
        assert_ne!(decision.0, first);
    }

    #[test]
    fn least_loaded_default_scoring_prefers_lower_kv_and_queue() {
        let r = router(
            vec![
                node("gpu-0", true, 0.8, 0),
                node("gpu-1", true, 0.2, 1),
                node("gpu-2", true, 0.5, 0),
            ],
            RoutingPolicy::AffinityThenLoad,
        );
        // Scores: gpu-0=0.48, gpu-1=0.16, gpu-2=0.30 -> gpu-1 wins.
        assert_eq!(r.least_loaded_node(), Some(NodeId::new("gpu-1")));
    }

    #[test]
    fn least_kv_usage_policy_ignores_queue() {
        let r = router(
            vec![
                node("gpu-0", true, 0.3, 100),
                node("gpu-1", true, 0.5, 0),
            ],
            RoutingPolicy::LeastKvUsage,
        );
        // Despite huge queue, gpu-0 has lower KV usage.
        assert_eq!(r.least_loaded_node(), Some(NodeId::new("gpu-0")));
    }

    #[test]
    fn weighted_policy_scoring_selects_expected_node() {
        // `least_loaded_node` uses `load_score` (no affinity bonus — that path
        // is exercised by `weighted_fallback_node` via `route`).
        // Scores: gpu-0 = 0.9×0.3 = 0.27, gpu-1 = 0.1×0.3 + 0.5×0.2 = 0.13.
        let w = WeightConfig {
            affinity_weight: 0.5,
            kv_weight: 0.3,
            queue_weight: 0.2,
        };
        let r = router(
            vec![
                node("gpu-0", true, 0.9, 0),
                node("gpu-1", true, 0.1, 5),
            ],
            RoutingPolicy::Weighted(w),
        );
        assert_eq!(r.least_loaded_node(), Some(NodeId::new("gpu-1")));
    }

    #[test]
    fn no_healthy_nodes_returns_none() {
        let mut r = router(
            vec![node("gpu-0", false, 0.1, 0), node("gpu-1", false, 0.1, 0)],
            RoutingPolicy::AffinityThenLoad,
        );
        let req = RouteRequest {
            session_id: Some("s1".into()),
            system_prompt_hash: None,
        };
        assert_eq!(r.route(&req), None);
        assert_eq!(r.least_loaded_node(), None);
    }

    #[test]
    fn update_node_applies_status_by_id() {
        let mut r = router(
            vec![node("gpu-0", true, 0.0, 0)],
            RoutingPolicy::AffinityThenLoad,
        );
        let status = NodeStatus {
            node_id: "gpu-0".into(),
            healthy: true,
            kv_usage: 0.42,
            kv_pages_used: 0,
            kv_pages_total: 0,
            kv_pages_shared: 0,
            queue_depth: 3,
            active_sessions: 0,
            paused_sessions: 0,
            tokens_per_second: 0.0,
            batch_utilization: 0.0,
            sessions: vec![],
            prefix_hashes: vec![],
        };
        assert!(r.update_node(status));
        assert!((r.node_by_id(&NodeId::new("gpu-0")).unwrap().kv_usage - 0.42).abs() < 1e-6);
        // Unknown node id is a no-op.
        let unknown = NodeStatus {
            node_id: "ghost".into(),
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
        assert!(!r.update_node(unknown));
    }

    #[test]
    fn record_node_miss_demotes_after_threshold() {
        let mut r = router(
            vec![node("gpu-0", true, 0.1, 0)],
            RoutingPolicy::AffinityThenLoad,
        );
        let id = NodeId::new("gpu-0");
        assert!(!r.record_node_miss(&id)); // 1
        assert!(!r.record_node_miss(&id)); // 2
        assert!(r.record_node_miss(&id)); // 3 -> unhealthy (default threshold 3)
        assert_eq!(r.healthy_node_count(), 0);
    }

    #[test]
    fn from_config_builds_nodes_and_thresholds() {
        let yaml = r#"
listen: "0.0.0.0:8080"
nodes:
  - address: "10.0.0.1:8000"
    name: "gpu-0"
  - address: "10.0.0.2:8000"
    name: "gpu-1"
routing:
  policy: least_kv_usage
  overload_threshold: 0.8
  prefix_colocate: false
health:
  unhealthy_after_misses: 2
"#;
        let cfg = RouterConfig::from_yaml_str(yaml).unwrap();
        let r = Router::from_config(&cfg);
        assert_eq!(r.nodes().len(), 2);
        assert_eq!(r.policy, RoutingPolicy::LeastKvUsage);
        assert!(!r.prefix_colocate);
        assert!((r.overload_threshold - 0.8).abs() < 1e-6);
        assert_eq!(r.unhealthy_after_misses, 2);
    }

    #[test]
    fn draining_node_excluded_from_new_sessions_but_keeps_affinity() {
        let mut r = router(
            vec![node("gpu-0", true, 0.1, 0), node("gpu-1", true, 0.9, 0)],
            RoutingPolicy::AffinityThenLoad,
        );
        // s1 pins to gpu-0 (least loaded).
        let req = RouteRequest {
            session_id: Some("s1".into()),
            system_prompt_hash: None,
        };
        let pinned = r.route(&req).unwrap();
        assert_eq!(pinned, NodeId::new("gpu-0"));
        // Drain gpu-0: existing session s1 still sticks (affinity honored)...
        assert!(r.set_draining(&pinned, true));
        assert!(r.is_draining(&pinned));
        let again = r.route_decision(&req).unwrap();
        assert_eq!(again.0, pinned);
        assert_eq!(again.1, RoutingDecision::Affinity);
        // ...but a brand new session avoids the draining node.
        let req2 = RouteRequest {
            session_id: Some("s2".into()),
            system_prompt_hash: None,
        };
        let new = r.route(&req2).unwrap();
        assert_eq!(new, NodeId::new("gpu-1"));
    }

    #[test]
    fn set_draining_unknown_node_is_false() {
        let mut r = router(
            vec![node("gpu-0", true, 0.1, 0)],
            RoutingPolicy::AffinityThenLoad,
        );
        assert!(!r.set_draining(&NodeId::new("ghost"), true));
    }

    #[test]
    fn draining_all_nodes_yields_no_route_for_new_session() {
        let mut r = router(
            vec![node("gpu-0", true, 0.1, 0)],
            RoutingPolicy::AffinityThenLoad,
        );
        assert!(r.set_draining(&NodeId::new("gpu-0"), true));
        let req = RouteRequest {
            session_id: Some("s1".into()),
            system_prompt_hash: None,
        };
        assert_eq!(r.route(&req), None);
    }

    #[test]
    fn rebalance_moves_sessions_off_draining_node() {
        let mut r = router(
            vec![node("gpu-0", true, 0.2, 0), node("gpu-1", true, 0.1, 0)],
            RoutingPolicy::AffinityThenLoad,
        );
        // Pin s1 to gpu-1 (least loaded).
        let req = RouteRequest {
            session_id: Some("s1".into()),
            system_prompt_hash: None,
        };
        let pinned = r.route(&req).unwrap();
        assert_eq!(pinned, NodeId::new("gpu-1"));
        // Drain gpu-1 and rebalance: s1 should move to gpu-0.
        r.set_draining(&pinned, true);
        let moved = r.rebalance();
        assert_eq!(moved, 1);
        assert_eq!(r.session_map().get("s1"), Some(&NodeId::new("gpu-0")));
        let migrations = r.session_map().migrations();
        assert_eq!(migrations.last().unwrap().reason, MigrationReason::Rebalance);
    }

    #[test]
    fn rebalance_noop_when_nodes_healthy_and_unloaded() {
        let mut r = router(
            vec![node("gpu-0", true, 0.2, 0), node("gpu-1", true, 0.1, 0)],
            RoutingPolicy::AffinityThenLoad,
        );
        let req = RouteRequest {
            session_id: Some("s1".into()),
            system_prompt_hash: None,
        };
        r.route(&req);
        assert_eq!(r.rebalance(), 0);
    }

    // ---- Weighted affinity bonus tests -------------------------------------

    /// With zero affinity_weight the lower-load node wins even though the
    /// session is pinned to the higher-load node.  With a high affinity_weight
    /// the pinned node gets a scoring discount and wins instead.
    #[test]
    fn weighted_affinity_bonus_changes_decision() {
        let make_router = |affinity_weight: f32| {
            let w = WeightConfig {
                affinity_weight,
                kv_weight: 0.3,
                queue_weight: 0.2,
            };
            router(
                vec![
                    node("gpu-0", true, 0.6, 0), // affinity target; score = 0.6×0.3 = 0.18
                    node("gpu-1", true, 0.3, 0), // competitor;      score = 0.3×0.3 = 0.09
                ],
                RoutingPolicy::Weighted(w),
            )
        };

        // No bonus: gpu-1 (score 0.09) beats gpu-0 (score 0.18).
        let mut r = make_router(0.0);
        r.record_session_affinity("s1", NodeId::new("gpu-0"));
        let req = RouteRequest { session_id: Some("s1".into()), system_prompt_hash: None };
        let d = r.route_decision(&req).unwrap();
        assert_eq!(d.0, NodeId::new("gpu-1"), "without bonus, lower-load gpu-1 wins");
        assert_eq!(d.1, RoutingDecision::LeastLoaded);

        // With affinity_weight=0.5: gpu-0 score = 0.18 − 0.5 = −0.32 < 0.09 → gpu-0 wins.
        let mut r = make_router(0.5);
        r.record_session_affinity("s1", NodeId::new("gpu-0"));
        let d = r.route_decision(&req).unwrap();
        assert_eq!(d.0, NodeId::new("gpu-0"), "with bonus, affinity gpu-0 wins");
        assert_eq!(d.1, RoutingDecision::LeastLoaded);
    }

    /// A maximum affinity_weight must not route to an *unhealthy* affinity node.
    #[test]
    fn weighted_affinity_bonus_skipped_for_unhealthy_node() {
        let w = WeightConfig { affinity_weight: 1.0, kv_weight: 0.3, queue_weight: 0.2 };
        let mut r = router(
            vec![
                node("gpu-0", false, 0.1, 0), // affinity target, but unhealthy
                node("gpu-1", true, 0.9, 0),  // only healthy candidate
            ],
            RoutingPolicy::Weighted(w),
        );
        r.record_session_affinity("s1", NodeId::new("gpu-0"));
        let req = RouteRequest { session_id: Some("s1".into()), system_prompt_hash: None };
        let d = r.route_decision(&req).unwrap();
        assert_eq!(d.0, NodeId::new("gpu-1"), "unhealthy affinity node excluded despite max bonus");
    }

    /// A maximum affinity_weight must not route to an *overloaded* affinity node
    /// (KV usage at or above the overload threshold).
    #[test]
    fn weighted_affinity_bonus_skipped_for_overloaded_node() {
        let w = WeightConfig { affinity_weight: 1.0, kv_weight: 0.3, queue_weight: 0.2 };
        let mut r = router(
            vec![
                node("gpu-0", true, 0.99, 0), // affinity target, but overloaded (> 0.95)
                node("gpu-1", true, 0.5, 0),  // healthy, below threshold
            ],
            RoutingPolicy::Weighted(w),
        );
        r.record_session_affinity("s1", NodeId::new("gpu-0"));
        let req = RouteRequest { session_id: Some("s1".into()), system_prompt_hash: None };
        let d = r.route_decision(&req).unwrap();
        // Without the bonus gpu-0 scores 0.99×0.3=0.297 vs gpu-1's 0.5×0.3=0.15 → gpu-1 wins.
        assert_eq!(d.0, NodeId::new("gpu-1"), "overloaded affinity node loses without bonus");
    }
}
