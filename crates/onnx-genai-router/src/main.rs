//! `onnx-genai-router` binary entry point (see `docs/DESIGN.md` §34.10).
//!
//! Loads a [`RouterConfig`] from YAML, builds the pure [`Router`], spawns the
//! background node poller, and serves the `/router/*` API plus the reverse-proxy
//! fallback on the configured listen address. Shuts down gracefully on SIGINT.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use onnx_genai_router::node_poller::{self, HttpStatusFetcher};
use onnx_genai_router::{AppState, Router, RouterConfig, build_app};

/// Per-poll HTTP timeout. Kept comfortably under a typical `poll_interval_ms`
/// so a hung node cannot stall the sweep; misses accrue toward the health cutoff.
const POLL_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Parser)]
#[command(
    name = "onnx-genai-router",
    about = "Model-agnostic, session-aware router for onnx-genai inference clusters"
)]
struct Cli {
    /// Path to the router YAML config (see docs/DESIGN.md §34.11).
    #[arg(long, env = "ONNX_GENAI_ROUTER_CONFIG")]
    config: PathBuf,

    /// Override the `listen` address from the config (host:port).
    #[arg(long)]
    listen: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let config = RouterConfig::from_path(&cli.config)?;
    let listen: SocketAddr = cli
        .listen
        .as_deref()
        .unwrap_or(config.listen.as_str())
        .parse()?;
    let poll_interval_ms = config.routing.poll_interval_ms;

    let router = Router::from_config(&config);
    let state = AppState::new(router, poll_interval_ms);

    // Spawn the background poller sharing the same hyper client as the proxy.
    let fetcher = HttpStatusFetcher::new(state.client.clone(), POLL_TIMEOUT);
    let poller = tokio::spawn(node_poller::run(state.clone(), fetcher));

    let app = build_app(state);
    let listener = tokio::net::TcpListener::bind(listen).await?;
    tracing::info!(
        %listen,
        nodes = config.nodes.len(),
        poll_interval_ms,
        "starting onnx-genai-router"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    poller.abort();
    tracing::info!("onnx-genai-router shut down");
    Ok(())
}

/// Resolve when SIGINT (Ctrl-C) is received.
async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::error!(error = %err, "failed to install SIGINT handler");
    }
    tracing::info!("shutdown signal received");
}
