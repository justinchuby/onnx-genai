use std::path::Path;

use anyhow::Context as _;
use onnx_genai::EngineConfig;
use onnx_genai::engine::PipelineEngine;
use onnx_genai::ort::{ChatMessage, Tokenizer};
use onnx_genai::text_to_audio;
use onnx_genai::text_to_image;
use onnx_genai_server::multimodal;

use super::commands::resolved_default_providers;
use super::interactive::{
    Backend, EXIT_INTERRUPTED, TurnInput, apply_context_sized_max_new_tokens,
    context_exhaustion_error, context_window_is_full, install_ctrlc_handler, is_interrupt_error,
    warn_missing_context_limit,
};
use super::output::{build_turn_prompt, detect_reasoning, load_chat_template, run_generation_turn};
use super::profile::{self, RunProfile};
use super::{GenerateArgs, ProfileArgs, resolve_model_dir};

pub(super) fn generate(args: GenerateArgs, profiling: &ProfileArgs) -> anyhow::Result<()> {
    install_ctrlc_handler();
    args.cpu.apply()?;
    let model_dir = resolve_model_dir(&args.model);
    let mut profile = RunProfile::new(model_dir.display().to_string());
    profile.execution_provider = resolved_default_providers();
    if args.image_output.output_image.is_some() && args.audio_output.output_audio.is_some() {
        anyhow::bail!(
            "What: --output-image and --output-audio were combined. \
             Why: one invocation produces one kind of output. \
             How: run the command once per output."
        );
    }
    if args.image_output.output_image.is_some() {
        return generate_image(&model_dir, args, profiling, profile);
    }
    if args.audio_output.output_audio.is_some() {
        return generate_audio(&model_dir, args, profiling, profile);
    }
    let options = args.sampling.to_options();

    let template = load_chat_template(&model_dir, args.sampling.raw);
    let history = vec![ChatMessage::user(args.prompt)];
    let prompt = build_turn_prompt(template.as_ref(), &history)?;
    let mut turn = TurnInput {
        prompt,
        images: args.attachments.images.clone(),
        audio: args.attachments.audio.clone(),
        options,
        prompt_tokens: None,
        context_limit: None,
    };

    let load_started = std::time::Instant::now();
    let mut backend = Backend::load(&model_dir, args.engine.to_config())?;
    profile.phase("model load", load_started.elapsed());
    let prompt_tokens = backend.prompt_tokens(&turn.prompt).unwrap_or_default();
    let effective_max_context = backend.effective_max_context(&turn.options);
    if let Some(limit) = context_window_is_full(prompt_tokens, effective_max_context) {
        return Err(context_exhaustion_error(prompt_tokens, limit));
    }
    let used_fallback = apply_context_sized_max_new_tokens(
        &mut turn.options,
        args.sampling.max_new_tokens.is_some(),
        prompt_tokens,
        effective_max_context,
    );
    if used_fallback {
        warn_missing_context_limit(turn.options.max_new_tokens);
    }
    turn.prompt_tokens = Some(prompt_tokens);
    turn.context_limit = effective_max_context;
    profile.prompt_tokens = Some(prompt_tokens);
    profile.context = effective_max_context.map(|max_tokens| profile::ContextUsage {
        used_tokens: prompt_tokens,
        max_tokens,
    });
    if let Some(memory) = backend.kv_usage() {
        profile.memory = memory;
    }
    let pages_before = backend.page_stats();
    let reasoning = detect_reasoning(template.as_ref());
    match run_generation_turn(
        &mut backend,
        turn,
        args.stream,
        Some(&mut profile),
        reasoning.as_ref(),
        None,
    ) {
        Ok(output) => {
            if args.stream {
                println!();
            } else {
                println!("{output}");
            }
            if let (Some(before), Some(after)) = (pages_before, backend.page_stats()) {
                profile.pages = Some(profile::PageActivity::since(before, after));
            }
            profiling.emit(&mut profile)?;
            Ok(())
        }
        Err(error) if is_interrupt_error(&error) => {
            // A Ctrl-C during a one-shot generation aborts and exits non-zero.
            eprintln!("\n^C interrupted");
            std::process::exit(EXIT_INTERRUPTED);
        }
        Err(error) => Err(error),
    }
}

/// Render `--prompt` to PNG(s) through the model's declared diffusion pipeline.
fn generate_image(
    model_dir: &Path,
    args: GenerateArgs,
    profiling: &ProfileArgs,
    mut profile: RunProfile,
) -> anyhow::Result<()> {
    let output = args
        .image_output
        .output_image
        .clone()
        .expect("image output path checked by the caller");
    let request = args.image_output.to_request(args.prompt.clone());
    let load_started = std::time::Instant::now();
    let mut engine = PipelineEngine::from_dir_with_config(model_dir, EngineConfig::default())
        .map_err(|error| {
            anyhow::anyhow!(
                "What: {} could not be loaded as a diffusion pipeline. \
                 Why: {error:#}. \
                 How: point --output-image at a package whose inference metadata declares a `pipeline` with `strategy.kind: iterative`.",
                model_dir.display()
            )
        })?;

    profile.phase("model load", load_started.elapsed());
    let render_started = std::time::Instant::now();
    let images = text_to_image::render(model_dir, &mut engine, &request)?;
    let render_elapsed = render_started.elapsed();
    profile.phase("render", render_elapsed);
    if let Some(steps) = request.steps.or(engine.spec().strategy.num_steps) {
        profile.counter("denoise steps", steps as f64, "steps");
        if steps > 0 {
            profile.counter(
                "per step",
                render_elapsed.as_secs_f64() * 1000.0 / steps as f64,
                "ms",
            );
        }
    }
    if images.is_empty() {
        anyhow::bail!(
            "What: no image was produced. \
             Why: the pipeline returned fewer images than the requested batch size of {}. \
             How: render with --batch-size 1, or report this as a pipeline output-shape bug.",
            request.batch_size
        );
    }
    for (index, image) in images.iter().enumerate() {
        let path = if images.len() == 1 {
            output.clone()
        } else {
            let stem = output
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| "out".to_string());
            let extension = output
                .extension()
                .map(|extension| extension.to_string_lossy().into_owned())
                .unwrap_or_else(|| "png".to_string());
            output.with_file_name(format!("{stem}_{index}.{extension}"))
        };
        text_to_image::save_png(image, &path)?;
        println!(
            "saved {} ({}x{})",
            path.display(),
            image.width,
            image.height
        );
    }
    profiling.emit(&mut profile)?;
    Ok(())
}

/// Synthesize `--prompt` to a WAV file through the model's declared TTS pipeline.
fn generate_audio(
    model_dir: &Path,
    args: GenerateArgs,
    profiling: &ProfileArgs,
    mut profile: RunProfile,
) -> anyhow::Result<()> {
    let output = args
        .audio_output
        .output_audio
        .clone()
        .expect("audio output path checked by the caller");
    let setup = multimodal::load(model_dir)?.with_context(|| {
        format!(
            "What: {} could not be loaded as a speech pipeline. \
             Why: it declares no `pipeline`, so it has no vocoder stage to run. \
             How: point --output-audio at a text-to-speech package.",
            model_dir.display()
        )
    })?;
    let tokenizer = Tokenizer::from_file(&setup.tokenizer_path).map_err(|error| {
        anyhow::anyhow!(
            "What: the package's tokenizer could not be loaded from {}. \
             Why: {error}. \
             How: verify the package ships a valid tokenizer.json.",
            setup.tokenizer_path.display()
        )
    })?;
    let load_started = std::time::Instant::now();
    let mut engine = PipelineEngine::from_dir_with_config(model_dir, args.engine.to_config())?;
    profile.phase("model load", load_started.elapsed());

    let request = args
        .audio_output
        .to_request(args.prompt.clone(), &args.sampling);
    let synthesis_started = std::time::Instant::now();
    let audio = text_to_audio::synthesize(&mut engine, &tokenizer, &request)?;
    let synthesis_elapsed = synthesis_started.elapsed();
    profile.phase("synthesis", synthesis_elapsed);
    profile.counter("audio produced", audio.duration_secs() as f64, "s");
    if audio.duration_secs() > 0.0 {
        profile.counter(
            "real-time factor",
            synthesis_elapsed.as_secs_f64() / audio.duration_secs() as f64,
            "x",
        );
    }
    text_to_audio::save_wav(&audio, &output)?;
    println!(
        "saved {} ({:.2}s, {} Hz, {} channel{})",
        output.display(),
        audio.duration_secs(),
        audio.sample_rate,
        audio.channels,
        if audio.channels == 1 { "" } else { "s" }
    );
    profiling.emit(&mut profile)?;
    Ok(())
}
