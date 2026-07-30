use clap::Parser;
use onnx_genai_server::{ServeArgs, run_serve};

/// Standalone entry point for the OpenAI-compatible server. The unified CLI
/// exposes the same flags via `onnx-genai serve`; both share [`ServeArgs`] and
/// [`run_serve`] from the library crate.
///
/// `--version` reports the COMMIT the binary was built from, not a semver
/// string. A release number would answer a question nobody asks of a demo
/// binary; the question actually asked -- repeatedly, for hours, with no way
/// to answer it -- was "which tree produced the process I am talking to".
/// `CARGO_TARGET_DIR` is shared across worktrees here, so the binary's PATH
/// names a directory rather than a history and `ls`/`stat` cannot answer it.
///
/// This is deliberately clap's built-in `--version` rather than a bespoke
/// flag: it short-circuits BEFORE required-argument validation, so provenance
/// can be read from a binary without supplying `--model`, which is exactly the
/// situation a launcher is in when it wants to check before starting anything.
#[derive(Debug, Parser)]
#[command(
    name = "onnx-genai-server",
    about = "OpenAI-compatible HTTP server for onnx-genai",
    version = env!("ONNX_GENAI_BUILD_SHA")
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
