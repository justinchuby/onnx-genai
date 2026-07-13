//! Shared runtime state for the router's HTTP layer.
//!
//! The pure [`Router`] is wrapped in a `std::sync::Mutex` behind an `Arc`. Lock
//! discipline (see `.squad/decisions`): critical sections are short and purely
//! synchronous — the guard is always dropped *before* any `.await` (network
//! I/O). We deliberately use a std `Mutex` (not a tokio one) precisely so the
//! type system nudges us away from holding it across an await point.

use std::sync::{Arc, Mutex};

use crate::metrics::Metrics;
use crate::node_poller::{ProxyClient, build_client};
use crate::router::Router;

/// Handle shared across the axum app, the proxy fallback, and the poller task.
pub struct AppState {
    /// The pure routing core. Locked briefly for each decision / state update.
    pub router: Mutex<Router>,
    /// Metrics registry (internally synchronized; no router lock required).
    pub metrics: Metrics,
    /// Shared hyper client used for both upstream proxying and status polls.
    pub client: ProxyClient,
    /// How often the background poller refreshes node status.
    pub poll_interval_ms: u64,
}

/// Convenience alias for the reference-counted [`AppState`].
pub type SharedState = Arc<AppState>;

impl AppState {
    /// Build shared state around a constructed [`Router`], creating a fresh
    /// shared hyper client.
    pub fn new(router: Router, poll_interval_ms: u64) -> SharedState {
        Self::with_client(router, poll_interval_ms, build_client())
    }

    /// Build shared state with an explicit hyper client (used by tests).
    pub fn with_client(
        router: Router,
        poll_interval_ms: u64,
        client: ProxyClient,
    ) -> SharedState {
        Arc::new(AppState {
            router: Mutex::new(router),
            metrics: Metrics::new(),
            client,
            poll_interval_ms,
        })
    }
}
