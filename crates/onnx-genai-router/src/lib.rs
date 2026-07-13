//! # onnx-genai-router
//!
//! A lightweight, **model-agnostic**, session-aware router for onnx-genai
//! inference clusters (see `docs/DESIGN.md` §34).
//!
//! The router sits behind a standard load balancer (Nginx/Envoy) and in front
//! of a fleet of inference nodes. It makes smart routing decisions that a
//! generic round-robin LB cannot:
//!
//! - **Session affinity** — a conversation's KV cache lives on one node; keep
//!   subsequent turns on that node to avoid re-prefilling.
//! - **Prefix co-location** — sessions that share a system prompt are placed on
//!   the same node so they can share KV pages.
//! - **Load-aware fallback** — when affinity/prefix don't apply (or the target
//!   node is down/overloaded) fall back to the least-loaded healthy node.
//!
//! ## What this crate knows nothing about
//!
//! The router is strictly model-agnostic. Node ids and session ids are opaque
//! strings. There are **no** model names, tokenizers, or inference concepts in
//! this crate — it routes purely by session affinity, prefix hash, and node
//! load reported over the `/v1/status` contract.
//!
//! ## Milestone scope (R3 — networking/runtime)
//!
//! R3 turns the pure core into a runnable binary:
//!
//! - [`node_poller`] implements [`node::NodeStatusFetcher`] over a `hyper-util`
//!   client and drives [`router::Router::update_node`] /
//!   [`router::Router::record_node_miss`] on a `tokio` interval.
//! - [`proxy`] is the model-agnostic reverse proxy (buffers the request to
//!   extract session id + prefix hash, streams the response back).
//! - [`api`] serves the `/router/*` endpoints and falls through to the proxy.
//! - [`metrics`] renders the §34.12 Prometheus exposition.
//! - [`state`] holds the shared [`router::Router`] behind an `Arc<Mutex>`.

pub mod api;
pub mod config;
pub mod metrics;
pub mod node;
pub mod node_poller;
pub mod prefix_map;
pub mod proxy;
pub mod router;
pub mod session_map;
pub mod state;

pub use api::build_app;
pub use config::{
    ConfigError, HealthConfig, NodeConfig, RouterConfig, RoutingConfig, RoutingPolicy,
    SessionMapConfig, WeightConfig,
};
pub use metrics::Metrics;
pub use node::{NodeId, NodeState, NodeStatus, NodeStatusFetcher, SessionSummary};
pub use node_poller::{HttpStatusFetcher, build_client};
pub use prefix_map::{PrefixMap, hash_system_prompt};
pub use router::{RouteRequest, RoutingDecision, Router};
pub use session_map::{MigrationEvent, MigrationReason, SessionMap};
pub use state::{AppState, SharedState};
