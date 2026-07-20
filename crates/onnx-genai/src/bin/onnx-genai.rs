use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use anyhow::Context;
use clap::{Args, Parser, Subcommand, ValueEnum};
use onnx_genai::{
    Engine, EngineConfig, FinishReason, GenerateOptions, GenerateRequest, GenerateToken,
    SpeculativeGenerationTrace, SpeculativeMode, SpeculativeTraceFamily, StopSequence,
};
use serde_json::{Map, Value, json};

#[derive(Debug, Parser)]
#[command(
    name = "onnx-genai",
    about = "Run generative AI models with ONNX Runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Generate text from a prompt.
    Generate(GenerateArgs),
    /// Write a provenance-bound token capture for inference-sim conformance.
    Capture(CaptureArgs),
}

#[derive(Debug, Args)]
struct GenerateArgs {
    /// Model directory containing the ONNX model, tokenizer, and optional metadata.
    #[arg(long)]
    model: PathBuf,

    /// Maximum number of new tokens to generate.
    #[arg(long)]
    max_new_tokens: Option<usize>,

    /// Temperature applied before token selection.
    #[arg(long)]
    temperature: Option<f32>,

    /// Nucleus sampling probability. Values >= 1 disable top-p filtering.
    #[arg(long)]
    top_p: Option<f32>,

    /// Keep only the top-k logits before token selection. Zero disables top-k filtering.
    #[arg(long)]
    top_k: Option<usize>,

    /// Text stop sequence. May be provided multiple times.
    #[arg(long)]
    stop: Vec<String>,

    /// Print generated tokens as they arrive.
    #[arg(long)]
    stream: bool,

    /// Prompt text.
    prompt: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CaptureRole {
    TargetOnly,
    Speculative,
}

#[derive(Debug, Args)]
struct CaptureArgs {
    /// Model directory containing the ONNX model, tokenizer, and optional metadata.
    #[arg(long)]
    model: PathBuf,

    /// Atomically published JSON capture path.
    #[arg(long)]
    output: PathBuf,

    /// Whether this is the target-only baseline or speculative run.
    #[arg(long, value_enum)]
    role: CaptureRole,

    /// Unique run identity. Use a different value for each role.
    #[arg(long)]
    id: String,

    /// Exact runtime revision, such as an onnx-genai git commit.
    #[arg(long)]
    runtime_revision: String,

    /// Content fingerprint of the target model artifact.
    #[arg(long)]
    model_fingerprint: String,

    /// Content fingerprint of the tokenizer artifact.
    #[arg(long)]
    tokenizer_fingerprint: String,

    /// Fingerprint of this controlled generation configuration.
    #[arg(long)]
    generation_config_fingerprint: String,

    /// Content fingerprint of the proposer artifact. Required for speculative runs.
    #[arg(long)]
    proposer_fingerprint: Option<String>,

    /// Use prompt-lookup speculation with this n-gram size instead of package metadata.
    #[arg(long)]
    prompt_lookup_ngram: Option<usize>,

    /// Prompt-lookup proposal width when --prompt-lookup-ngram is set.
    #[arg(long, default_value_t = 4)]
    prompt_lookup_max_tokens: usize,

    /// Exact number of output tokens. Capture disables EOS and stop sequences.
    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,

    /// Prompt text.
    prompt: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate(args) => generate(args),
        Commands::Capture(args) => capture(args),
    }
}

fn generate(args: GenerateArgs) -> anyhow::Result<()> {
    let mut options = GenerateOptions::default();
    if let Some(max_new_tokens) = args.max_new_tokens {
        options.max_new_tokens = max_new_tokens;
    }
    if let Some(temperature) = args.temperature {
        options.temperature = temperature;
    }
    if let Some(top_p) = args.top_p {
        options.top_p = top_p;
    }
    if let Some(top_k) = args.top_k {
        options.top_k = top_k;
    }
    options.stop_sequences = args.stop.into_iter().map(StopSequence::Text).collect();

    let request = GenerateRequest {
        prompt: args.prompt.into(),
        options,
    };

    let mut engine = Engine::from_dir(&args.model, EngineConfig::default())?;

    if args.stream {
        let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
            print!("{}", token.text);
            io::stdout().flush()?;
            Ok(())
        };
        engine.generate_with_callback(request, Some(&mut callback))?;
    } else {
        let result = engine.generate(request)?;
        print!("{}", result.text);
        io::stdout().flush()?;
    }

    Ok(())
}

fn capture(args: CaptureArgs) -> anyhow::Result<()> {
    validate_capture_args(&args)?;
    let mut options = GenerateOptions {
        max_new_tokens: args.max_new_tokens,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        ..Default::default()
    };
    if args.role == CaptureRole::TargetOnly {
        options.speculative_mode = Some(SpeculativeMode::None);
    } else if let Some(ngram) = args.prompt_lookup_ngram {
        options.speculative_mode = Some(SpeculativeMode::PromptLookup {
            ngram,
            max_tokens: args.prompt_lookup_max_tokens,
        });
    }
    let request = GenerateRequest {
        prompt: args.prompt.clone().into(),
        options,
    };
    let mut engine = Engine::from_dir(&args.model, EngineConfig::default())?;
    let (_result, trace) = engine.generate_with_speculative_trace(request)?;
    let document = capture_document(&args, &trace)?;
    write_json_atomically(&args.output, &document)
}

fn validate_capture_args(args: &CaptureArgs) -> anyhow::Result<()> {
    for (name, value) in [
        ("id", args.id.as_str()),
        ("runtime_revision", args.runtime_revision.as_str()),
        ("model_fingerprint", args.model_fingerprint.as_str()),
        ("tokenizer_fingerprint", args.tokenizer_fingerprint.as_str()),
        (
            "generation_config_fingerprint",
            args.generation_config_fingerprint.as_str(),
        ),
    ] {
        if value.trim().is_empty() {
            anyhow::bail!("{name} must be non-empty");
        }
    }
    if args.max_new_tokens == 0 {
        anyhow::bail!("max_new_tokens must be greater than zero");
    }
    if args.prompt_lookup_ngram == Some(0) {
        anyhow::bail!("prompt_lookup_ngram must be greater than zero");
    }
    if args.prompt_lookup_ngram.is_some() && args.prompt_lookup_max_tokens == 0 {
        anyhow::bail!("prompt_lookup_max_tokens must be greater than zero");
    }
    if args.role == CaptureRole::TargetOnly && args.prompt_lookup_ngram.is_some() {
        anyhow::bail!("target-only capture must not configure prompt lookup");
    }
    match (args.role, args.proposer_fingerprint.as_deref()) {
        (CaptureRole::TargetOnly, Some(_)) => {
            anyhow::bail!("target-only capture must not declare a proposer fingerprint");
        }
        (CaptureRole::Speculative, Some(value)) if !value.trim().is_empty() => {}
        (CaptureRole::Speculative, _) => {
            anyhow::bail!("speculative capture requires a non-empty proposer fingerprint");
        }
        (CaptureRole::TargetOnly, None) => {}
    }
    Ok(())
}

fn capture_document(
    args: &CaptureArgs,
    trace: &SpeculativeGenerationTrace,
) -> anyhow::Result<Value> {
    if trace.finish_reason != FinishReason::MaxTokens {
        anyhow::bail!(
            "conformance capture requires max-token completion, got {:?}",
            trace.finish_reason
        );
    }
    let mut capture = Map::from_iter([
        ("revision".into(), json!(1)),
        ("id".into(), json!(args.id)),
        (
            "role".into(),
            json!(match args.role {
                CaptureRole::TargetOnly => "target_only",
                CaptureRole::Speculative => "speculative",
            }),
        ),
        (
            "provenance".into(),
            json!({
                "source": "onnx-genai-cli",
                "runtime_revision": args.runtime_revision,
                "model_fingerprint": args.model_fingerprint,
                "tokenizer_fingerprint": args.tokenizer_fingerprint,
                "generation_config_fingerprint": args.generation_config_fingerprint,
            }),
        ),
        ("completion_reason".into(), json!("max_tokens")),
        ("prompt_token_ids".into(), json!(trace.prompt_token_ids)),
        ("output_token_ids".into(), json!(trace.output_token_ids)),
    ]);
    match args.role {
        CaptureRole::TargetOnly => {
            if trace.family.is_some()
                || trace.max_additional_tokens.is_some()
                || !trace.iterations.is_empty()
            {
                anyhow::bail!("target-only capture unexpectedly entered speculative decode");
            }
        }
        CaptureRole::Speculative => {
            let family = trace
                .family
                .context("speculative capture did not enter speculative decode")?;
            let max_additional_tokens = trace
                .max_additional_tokens
                .context("speculative capture did not report its proposal width")?;
            if trace.iterations.is_empty() {
                anyhow::bail!("speculative capture produced no verification iterations");
            }
            capture.insert("family".into(), json!(family_name(family)));
            capture.insert(
                "proposer_fingerprint".into(),
                json!(
                    args.proposer_fingerprint
                        .as_deref()
                        .expect("validated speculative proposer fingerprint")
                ),
            );
            capture.insert("max_additional_tokens".into(), json!(max_additional_tokens));
            capture.insert(
                "iterations".into(),
                Value::Array(
                    trace
                        .iterations
                        .iter()
                        .enumerate()
                        .map(|(index, iteration)| {
                            json!({
                                "id": format!("iteration-{index}"),
                                "output_offset": iteration.output_offset,
                                "proposal_token_ids": iteration.proposal_token_ids,
                                "target_token_ids": iteration.target_token_ids,
                                "committed_token_ids": iteration.committed_token_ids,
                            })
                        })
                        .collect(),
                ),
            );
        }
    }
    capture.insert(
        "terminal".into(),
        json!({
            "status": "completed",
            "output_token_count": trace.output_token_ids.len(),
            "iteration_count": trace.iterations.len(),
        }),
    );
    Ok(json!({ "runtime_token_capture": capture }))
}

fn family_name(family: SpeculativeTraceFamily) -> &'static str {
    match family {
        SpeculativeTraceFamily::DraftModel => "draft_model",
        SpeculativeTraceFamily::PromptLookup => "prompt_lookup",
        SpeculativeTraceFamily::Mtp => "mtp",
        SpeculativeTraceFamily::Eagle3 => "eagle3",
        SpeculativeTraceFamily::SharedKv => "shared_kv",
    }
}

fn write_json_atomically(path: &Path, document: &Value) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("capture output path must have a UTF-8 file name")?;
    let temporary = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(document)?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes)?;
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai::SpeculativeIterationTrace;

    fn args(role: CaptureRole) -> CaptureArgs {
        CaptureArgs {
            model: "model".into(),
            output: "capture.json".into(),
            role,
            id: "run-001".into(),
            runtime_revision: "onnx-genai@test".into(),
            model_fingerprint: "sha256:model".into(),
            tokenizer_fingerprint: "sha256:tokenizer".into(),
            generation_config_fingerprint: "sha256:config".into(),
            proposer_fingerprint: (role == CaptureRole::Speculative)
                .then(|| "sha256:proposer".into()),
            prompt_lookup_ngram: None,
            prompt_lookup_max_tokens: 4,
            max_new_tokens: 2,
            prompt: "hello".into(),
        }
    }

    #[test]
    fn target_only_document_has_a_complete_zero_iteration_terminal() {
        let trace = SpeculativeGenerationTrace {
            prompt_token_ids: vec![1],
            output_token_ids: vec![2, 3],
            finish_reason: FinishReason::MaxTokens,
            family: None,
            max_additional_tokens: None,
            iterations: Vec::new(),
        };
        let document = capture_document(&args(CaptureRole::TargetOnly), &trace).unwrap();
        assert_eq!(
            document["runtime_token_capture"]["terminal"],
            json!({
                "status": "completed",
                "output_token_count": 2,
                "iteration_count": 0,
            })
        );
    }

    #[test]
    fn speculative_document_preserves_executed_iteration_facts() {
        let trace = SpeculativeGenerationTrace {
            prompt_token_ids: vec![1],
            output_token_ids: vec![2, 4],
            finish_reason: FinishReason::MaxTokens,
            family: Some(SpeculativeTraceFamily::PromptLookup),
            max_additional_tokens: Some(2),
            iterations: vec![SpeculativeIterationTrace {
                output_offset: 0,
                proposal_token_ids: vec![2, 3],
                target_token_ids: vec![2, 4],
                committed_token_ids: vec![2, 4],
            }],
        };
        let document = capture_document(&args(CaptureRole::Speculative), &trace).unwrap();
        let capture = &document["runtime_token_capture"];
        assert_eq!(capture["family"], "prompt_lookup");
        assert_eq!(capture["max_additional_tokens"], 2);
        assert_eq!(capture["iterations"][0]["target_token_ids"], json!([2, 4]));
        assert_eq!(
            capture["iterations"][0]["committed_token_ids"],
            json!([2, 4])
        );
    }
}
