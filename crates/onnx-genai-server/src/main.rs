use clap::Parser;
use onnx_genai_server::{ServeArgs, run_serve};

/// Standalone entry point for the OpenAI-compatible server. The unified CLI
/// exposes the same flags via `onnx-genai serve`; both share [`ServeArgs`] and
/// [`run_serve`] from the library crate.
#[derive(Debug, Parser)]
#[command(
    name = "onnx-genai-server",
    about = "OpenAI-compatible HTTP server for onnx-genai"
)]
struct Cli {
    #[command(flatten)]
    serve: ServeArgs,
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
    run_serve(cli.serve).await
}
