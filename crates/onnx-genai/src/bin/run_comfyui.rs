//! Convert a ComfyUI workflow, then run it on the generic workflow engine.
//!
//! ```text
//! run_comfyui --package <dir> [options] <workflow.json>
//! ```
//!
//! This is a thin wrapper and nothing more. It converts the workflow with
//! [`onnx_genai_comfyui_config`], writes the canonical
//! `inference_metadata.yaml` into the package, and then hands the package to
//! [`Engine::from_pipeline_dir`] like any other workflow package. Every step of
//! the diffusion loop — schedule, guidance, solver, decode — is executed by the
//! generic workflow runtime from the emitted metadata. There is no diffusion
//! logic here, and no code path that reads the ComfyUI document at run time.

use std::path::{Path, PathBuf};
use std::time::Instant;

use onnx_genai::engine::PipelineGenerateRequest;
use onnx_genai::engine::pipeline::{PipelineOutputs, WorkflowOutputRole};
use onnx_genai::ort::Value;
use onnx_genai::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_comfyui_config::{ComponentLayout, ConvertOptions, convert_file, to_yaml};

const USAGE: &str = "usage: run_comfyui --package <dir> [options] <workflow.json>\n\n\
     Converts a ComfyUI API-format workflow into canonical inference metadata and executes it \
     on the generic workflow engine.\n\n\
     --package <dir>       workflow package holding the ONNX components\n\
     --metadata <path>     where to write the converted metadata \
     (default: <package>/inference_metadata.yaml)\n\
     --overwrite           replace an existing metadata document\n\
     --convert-only        convert and write metadata without executing\n\
     --adapters <path>     the package's own `adapters` contract, required for LoRA workflows\n\
     --textproto           reference `*.onnx.textproto` component artifacts\n\
     --prompt-tokens a,b   positive prompt token ids (default: 1,2)\n\
     --negative-tokens a,b negative prompt token ids (default: zeros of the same length)\n\
     --steps <n>           override the converted iteration count\n\
     --output <path>       write the decoded image as a binary PPM";

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

struct Args {
    workflow: PathBuf,
    package: PathBuf,
    metadata: Option<PathBuf>,
    adapters: Option<PathBuf>,
    output: Option<PathBuf>,
    prompt_tokens: Vec<i64>,
    negative_tokens: Option<Vec<i64>>,
    steps: Option<usize>,
    overwrite: bool,
    convert_only: bool,
    textproto: bool,
}

fn run() -> anyhow::Result<()> {
    let args = parse_args()?;
    let options = ConvertOptions {
        layout: if args.textproto {
            ComponentLayout::textproto()
        } else {
            ComponentLayout::default()
        },
        adapters: match &args.adapters {
            Some(path) => {
                let value: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(path)?)?;
                Some(value.get("adapters").cloned().unwrap_or(value))
            }
            None => None,
        },
    };

    let (_, document, report) = convert_file(&args.workflow, &options)?;
    let metadata_path = args
        .metadata
        .clone()
        .unwrap_or_else(|| args.package.join("inference_metadata.yaml"));
    if metadata_path.exists() && !args.overwrite {
        anyhow::bail!(
            "{} already exists. Why: the converted document becomes the package's source of \
             execution truth, so replacing one silently would change what the package means. \
             How to fix: pass --overwrite, or point --metadata at a new path",
            metadata_path.display()
        );
    }
    std::fs::write(&metadata_path, to_yaml(&document)?)?;
    eprintln!(
        "converted {} -> {} ({} steps, solver={}, spacing={}, guidance={})",
        args.workflow.display(),
        metadata_path.display(),
        report.plan.iterations(),
        report.plan.solver.as_str(),
        report.plan.spacing.as_str(),
        report
            .plan
            .guidance
            .as_ref()
            .map_or("off".to_owned(), |guidance| guidance.scale.to_string()),
    );
    for ignored in &report.ignored_nodes {
        eprintln!("ignored (cannot reach the saved image): {ignored}");
    }
    if args.convert_only {
        return Ok(());
    }

    let iterations = args.steps.unwrap_or(report.plan.iterations() as usize);
    let mut engine = Engine::from_pipeline_dir(&args.package, EngineConfig::default())?;

    let options = GenerateOptions {
        max_new_tokens: iterations,
        seed: Some(report.plan.seed.unsigned_abs()),
        ..GenerateOptions::default()
    };
    let width = i64::try_from(args.prompt_tokens.len())?;
    let negative = args
        .negative_tokens
        .clone()
        .unwrap_or_else(|| vec![0; args.prompt_tokens.len()]);
    let mut request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(Vec::new()),
        options,
    })
    .with_input(
        "request.input_ids",
        Value::from_slice_i64(&args.prompt_tokens, &[1, width])?,
    )
    .with_input(
        "request.seed",
        Value::from_slice_i64(&[report.plan.seed], &[1])?,
    );
    if report.plan.uses_guidance() {
        let scale = report.plan.guidance.as_ref().map_or(1.0, |g| g.scale) as f32;
        request = request
            .with_input(
                "request.negative_input_ids",
                Value::from_slice_i64(&negative, &[1, i64::try_from(negative.len())?])?,
            )
            .with_input(
                "request.guidance_scale",
                Value::from_slice_f32(&[scale], &[1])?,
            );
    }

    let started = Instant::now();
    let outputs = engine.run_pipeline_outputs(request)?;
    let elapsed = started.elapsed();

    let image = image_output(&engine, &outputs)?;
    let shape = image.shape().to_vec();
    eprintln!(
        "executed {iterations} steps in {:.3}s ({:.2} steps/s); image shape {shape:?}",
        elapsed.as_secs_f64(),
        iterations as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
    );
    if let Some(path) = &args.output {
        write_ppm(path, &image.to_vec_f32()?, &shape)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

/// The workflow's image-role output, which is the only thing this tool reads.
///
/// The role is read from the emitted metadata rather than from an output name,
/// so nothing here depends on how the converter spelled it.
fn image_output<'a>(
    engine: &onnx_genai::Engine,
    outputs: &'a PipelineOutputs,
) -> anyhow::Result<&'a Value> {
    engine
        .structured_output_for_role(outputs, WorkflowOutputRole::Image)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "the converted workflow produced no image-role output. Why: a converted ComfyUI \
                 workflow always emits one, so this means the package's metadata was replaced \
                 after conversion"
            )
        })
}

/// Write a `[batch, 3, height, width]` float tensor's first row as binary PPM.
fn write_ppm(path: &Path, pixels: &[f32], shape: &[i64]) -> anyhow::Result<()> {
    let [_, channels, height, width] = shape else {
        anyhow::bail!("expected a rank-4 image output, got shape {shape:?}");
    };
    if *channels != 3 {
        anyhow::bail!("expected a 3-channel image output, got {channels} channels");
    }
    let (height, width) = (*height as usize, *width as usize);
    let plane = height * width;
    let mut out = format!("P6\n{width} {height}\n255\n").into_bytes();
    for index in 0..plane {
        for channel in 0..3 {
            // Diffusion decoders emit [-1, 1]; clamping keeps a partially
            // trained or tiny model from wrapping around instead of saturating.
            let value = pixels[channel * plane + index];
            out.push((((value + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn parse_args() -> anyhow::Result<Args> {
    let mut workflow = None;
    let mut package = None;
    let mut metadata = None;
    let mut adapters = None;
    let mut output = None;
    let mut prompt_tokens = None;
    let mut negative_tokens = None;
    let mut steps = None;
    let mut overwrite = false;
    let mut convert_only = false;
    let mut textproto = false;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = |flag: &str| -> anyhow::Result<String> {
            arguments
                .next()
                .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
        };
        match argument.as_str() {
            "--package" => package = Some(PathBuf::from(value("--package")?)),
            "--metadata" => metadata = Some(PathBuf::from(value("--metadata")?)),
            "--adapters" => adapters = Some(PathBuf::from(value("--adapters")?)),
            "--output" => output = Some(PathBuf::from(value("--output")?)),
            "--prompt-tokens" => prompt_tokens = Some(parse_tokens(&value("--prompt-tokens")?)?),
            "--negative-tokens" => {
                negative_tokens = Some(parse_tokens(&value("--negative-tokens")?)?)
            }
            "--steps" => steps = Some(value("--steps")?.parse()?),
            "--overwrite" => overwrite = true,
            "--convert-only" => convert_only = true,
            "--textproto" => textproto = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other if other.starts_with('-') => anyhow::bail!("unknown option: {other}"),
            other => workflow = Some(PathBuf::from(other)),
        }
    }
    Ok(Args {
        workflow: workflow.ok_or_else(|| anyhow::anyhow!("{USAGE}"))?,
        package: package.ok_or_else(|| anyhow::anyhow!("--package is required\n\n{USAGE}"))?,
        metadata,
        adapters,
        output,
        prompt_tokens: prompt_tokens.unwrap_or_else(|| vec![1, 2]),
        negative_tokens,
        steps,
        overwrite,
        convert_only,
        textproto,
    })
}

fn parse_tokens(value: &str) -> anyhow::Result<Vec<i64>> {
    value
        .split(',')
        .map(|token| token.trim().parse::<i64>().map_err(anyhow::Error::from))
        .collect()
}
