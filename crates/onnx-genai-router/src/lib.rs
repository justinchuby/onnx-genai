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
//! ## Milestone scope (R2 — pure core)
//!
//! This milestone ships the pure routing logic, configuration model, node
//! polling data model, and comprehensive unit tests. The hyper-based reverse
//! proxy (`proxy.rs`), the `/router/*` HTTP API (`api.rs`), and the binary
//! entry point (`main.rs`) are deferred to R3. The core here is structured so
//! R3 can bolt those on without touching this logic:
//!
//! - [`node::NodeStatusFetcher`] is the async seam the R3 poller implements.
//! - [`router::Router::update_node`] / [`router::Router::record_node_miss`]
//!   feed poll results into the pure state machine.
//! - [`router::Router::route`] is a pure decision function over the current
//!   node snapshot.

pub mod config;
pub mod node;
pub mod prefix_map;
pub mod router;
pub mod session_map;

pub use config::{
    ConfigError, HealthConfig, NodeConfig, RouterConfig, RoutingConfig, RoutingPolicy,
    SessionMapConfig, WeightConfig,
};
pub use node::{NodeId, NodeState, NodeStatus, NodeStatusFetcher, SessionSummary};
pub use prefix_map::{PrefixMap, hash_system_prompt};
pub use router::{RouteRequest, RoutingDecision, Router};
pub use session_map::{MigrationEvent, MigrationReason, SessionMap};
