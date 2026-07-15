//! Native nxrt single-inference profiler.
//!
//! The native session API currently exposes one named-input `run` call and is
//! CPU-only. It does not yet expose I/O binding, KV-cache rotation, generation,
//! or session-level CUDA graph capture/replay, so this tool measures repeated
//! whole-model inference rather than token generation.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use onnx_runtime_session::{InferenceSession, Tensor};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ExecutionProvider {
    Cpu,
    Cuda,
}

#[derive(Debug, Parser)]
#[command(about = "Profile repeated whole-model inference through native nxrt")]
struct Args {
    /// ONNX model file, or a directory containing model.onnx.
    #[arg(long)]
    model: PathBuf,

    /// Repeated inference calls per measured run (not generated tokens).
    #[arg(long, default_value_t = 128)]
    tokens: usize,

    #[arg(long, default_value_t = 1)]
    warmups: usize,

    #[arg(long, default_value_t = 1)]
    runs: usize,

    #[arg(long, value_enum, default_value_t = ExecutionProvider::Cpu)]
    ep: ExecutionProvider,

    /// Request native CUDA graph capture/replay when it becomes available.
    #[arg(long)]
    cuda_graph: bool,
}

fn model_file(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("model.onnx")
    } else {
        path.to_path_buf()
    }
}

fn zero_inputs(session: &InferenceSession) -> Result<Vec<(String, Tensor)>> {
    session
        .inputs()
        .iter()
        .map(|meta| {
            let shape: Vec<usize> = meta
                .shape
                .iter()
                .map(|dim| dim.as_static().unwrap_or(1))
                .collect();
            let elements = shape
                .iter()
                .try_fold(1usize, |n, &dim| n.checked_mul(dim))
                .with_context(|| format!("input {} shape overflows usize", meta.name))?;
            let bytes = meta
                .dtype
                .checked_storage_bytes(elements)
                .filter(|&size| size != 0)
                .with_context(|| {
                    format!(
                        "cannot synthesize input {} with dtype {:?}",
                        meta.name, meta.dtype
                    )
                })?;
            let tensor = Tensor::from_raw(meta.dtype, shape, &vec![0; bytes])
                .with_context(|| format!("create zero input {}", meta.name))?;
            Ok((meta.name.clone(), tensor))
        })
        .collect()
}

fn run_once(session: &mut InferenceSession, inputs: &[(String, Tensor)]) -> Result<()> {
    let bindings: Vec<(&str, &Tensor)> = inputs
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    let outputs = session.run(&bindings).context(
        "native session.run failed (an unsupported operator is expected while native op coverage grows)",
    )?;
    std::hint::black_box(outputs);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.tokens == 0 || args.runs == 0 {
        bail!("--tokens and --runs must be greater than zero");
    }

    if matches!(args.ep, ExecutionProvider::Cuda) {
        if args.cuda_graph {
            eprintln!(
                "native CUDA graph capture not yet wired at session level \
                 (individual kernels expose cuda_graph_compatible() gating)"
            );
        }
        bail!(
            "native onnx-runtime-session is currently CPU-only; CUDA EP selection is not yet \
             wired into SessionBuilder"
        );
    }
    if args.cuda_graph {
        eprintln!("--cuda-graph applies only to --ep cuda; continuing on CPU without capture");
    }

    let model = model_file(&args.model);
    println!(
        "profile_native: model={} ep=cpu repeated_inferences={} warmups={} runs={}",
        model.display(),
        args.tokens,
        args.warmups,
        args.runs
    );
    println!(
        "mode: native whole-model session.run; no I/O binding/KV decode loop is available, \
         so tok/s is not measurable"
    );

    let mut session = InferenceSession::load(&model)
        .with_context(|| format!("load native model {}", model.display()))?;
    let inputs = zero_inputs(&session)?;
    println!(
        "inputs: {} (symbolic dimensions synthesized as 1); outputs: {}",
        inputs.len(),
        session.outputs().len()
    );

    for _ in 0..args.warmups {
        for _ in 0..args.tokens {
            run_once(&mut session, &inputs)?;
        }
    }

    let calls = args
        .tokens
        .checked_mul(args.runs)
        .context("measured inference count overflows usize")?;
    let start = Instant::now();
    for _ in 0..calls {
        run_once(&mut session, &inputs)?;
    }
    let elapsed = start.elapsed();
    let runs_per_second = calls as f64 / elapsed.as_secs_f64();
    let milliseconds_per_run = elapsed.as_secs_f64() * 1_000.0 / calls as f64;
    println!(
        "throughput: {:.2} runs/s, {:.3} ms/run ({} session.run calls in {:.3} ms)",
        runs_per_second,
        milliseconds_per_run,
        calls,
        elapsed.as_secs_f64() * 1_000.0
    );
    println!("tok/s: unavailable (native generation/decode loop is not implemented)");
    Ok(())
}
