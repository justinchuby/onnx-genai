use std::io::{self, IsTerminal};
use std::path::Path;

use anyhow::Context as _;
use onnx_genai::engine::PipelineEngine;
use onnx_genai::ort::{ChatMessage, Tokenizer};
use onnx_genai::text_to_audio;
use onnx_genai::text_to_image;
use onnx_genai_server::multimodal;

use super::commands::resolved_default_providers;
use super::interactive::{
    Backend, EXIT_INTERRUPTED, ReplInputMode, TurnInput, apply_context_sized_max_new_tokens,
    context_exhaustion_error, context_window_is_full, initial_repl_show_stats,
    install_ctrlc_handler, is_interrupt_error, repl_input_mode, warn_missing_context_limit,
};
use super::output::{
    build_turn_prompt, detect_reasoning, emit_stats_line, load_chat_template, run_generation_turn,
};
use super::profile::{self, RunProfile};
use super::{GenerateArgs, ProfileArgs, decode_backend_name, resolve_model_dir};

pub(super) fn generate(args: GenerateArgs, profiling: &ProfileArgs) -> anyhow::Result<()> {
    install_ctrlc_handler();
    args.cpu.apply()?;
    let model_dir = resolve_model_dir(&args.model);
    let mut profile = RunProfile::new(model_dir.display().to_string());
    profile.execution_provider = resolved_default_providers();
    let output_kind = generate_output_kind(&args)?;
    let input_mode = repl_input_mode(io::stdin().is_terminal(), io::stdout().is_terminal());
    let show_stats = initial_generate_show_stats(input_mode, args.no_stats, output_kind);
    if matches!(output_kind, GenerateOutputKind::Image) {
        return generate_image(&model_dir, args, profiling, profile);
    }
    if matches!(output_kind, GenerateOutputKind::Audio) {
        return generate_audio(&model_dir, args, profiling, profile);
    }
    generate_text(&model_dir, args, profiling, profile, show_stats)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerateOutputKind {
    Text,
    Image,
    Audio,
}

fn generate_output_kind(args: &GenerateArgs) -> anyhow::Result<GenerateOutputKind> {
    if args.image_output.output_image.is_some() && args.audio_output.output_audio.is_some() {
        anyhow::bail!(
            "What: --output-image and --output-audio were combined. \
             Why: one invocation produces one kind of output. \
             How: run the command once per output."
        );
    }
    if args.image_output.output_image.is_some() {
        return Ok(GenerateOutputKind::Image);
    }
    if args.audio_output.output_audio.is_some() {
        return Ok(GenerateOutputKind::Audio);
    }
    Ok(GenerateOutputKind::Text)
}

fn initial_generate_show_stats(
    mode: ReplInputMode,
    no_stats: bool,
    output_kind: GenerateOutputKind,
) -> bool {
    matches!(output_kind, GenerateOutputKind::Text) && initial_repl_show_stats(mode, no_stats)
}

fn generate_text(
    model_dir: &Path,
    args: GenerateArgs,
    profiling: &ProfileArgs,
    mut profile: RunProfile,
    show_stats: bool,
) -> anyhow::Result<()> {
    let options = args.sampling.to_options();

    let template = load_chat_template(model_dir, args.sampling.raw);
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
    let mut backend = Backend::load(model_dir, args.engine.to_config())?;
    profile.decode_backend = Some(decode_backend_name(backend.decode_backend()).to_string());
    profile.phase("model load", load_started.elapsed());
    // Honor the model's declared sampling regime (e.g. a reasoning model that
    // ships do_sample=true) now that metadata is loaded; explicit CLI flags
    // still win. Without this a model that degenerates under greedy would loop.
    turn.options.resolve_sampling_defaults(
        backend.generation_defaults(),
        &args.sampling.sampling_overrides(),
    );
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
    let offload_before = cuda_offload_stats();
    let mut reasoning = detect_reasoning(template.as_ref());
    backend.bind_reasoning_marker_tokens(&mut reasoning);
    match run_generation_turn(
        &mut backend,
        turn,
        args.stream,
        Some(&mut profile),
        reasoning.as_ref(),
        None,
    ) {
        Ok(output) => {
            if !args.stream {
                println!("{output}");
            }
            if let (Some(before), Some(after)) = (pages_before, backend.page_stats()) {
                profile.pages = Some(profile::PageActivity::since(before, after));
            }
            record_cuda_offload_counters(&mut profile, offload_before);
            profiling.emit(&mut profile)?;
            emit_stats_line(show_stats, profiling.profile, &mut profile);
            Ok(())
        }
        Err(error) if is_interrupt_error(&error) => {
            // A Ctrl-C during a one-shot generation aborts and exits non-zero.
            eprintln!("^C interrupted");
            std::process::exit(EXIT_INTERRUPTED);
        }
        Err(error) => Err(error),
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CudaOffloadSnapshot {
    page_ins: u64,
    hits: u64,
    evictions: u64,
    prefetch_issued: u64,
    prefetch_declined_guard: u64,
    prefetch_joined: u64,
}

fn cuda_offload_stats() -> CudaOffloadSnapshot {
    #[cfg(feature = "native-cuda")]
    {
        let stats = onnx_runtime_ep_cuda::global_offload_stats();
        CudaOffloadSnapshot {
            page_ins: stats.page_ins,
            hits: stats.hits,
            evictions: stats.evictions,
            prefetch_issued: stats.prefetch_issued,
            prefetch_declined_guard: stats.prefetch_declined_guard,
            prefetch_joined: stats.prefetch_joined,
        }
    }
    #[cfg(not(feature = "native-cuda"))]
    {
        CudaOffloadSnapshot::default()
    }
}

fn record_cuda_offload_counters(profile: &mut RunProfile, before: CudaOffloadSnapshot) {
    #[cfg(feature = "native-cuda")]
    let after = onnx_runtime_ep_cuda::global_offload_stats();
    #[cfg(not(feature = "native-cuda"))]
    let after = CudaOffloadSnapshot::default();

    let page_ins = after.page_ins.saturating_sub(before.page_ins);
    let hits = after.hits.saturating_sub(before.hits);
    let evictions = after.evictions.saturating_sub(before.evictions);
    let prefetch_issued = after.prefetch_issued.saturating_sub(before.prefetch_issued);
    let prefetch_declined_guard = after
        .prefetch_declined_guard
        .saturating_sub(before.prefetch_declined_guard);
    let prefetch_joined = after.prefetch_joined.saturating_sub(before.prefetch_joined);
    if page_ins > 0
        || hits > 0
        || evictions > 0
        || prefetch_issued > 0
        || prefetch_declined_guard > 0
        || prefetch_joined > 0
    {
        profile.counter(
            "weight offload prefetch issued",
            prefetch_issued as f64,
            "prefetches",
        );
        profile.counter(
            "weight offload prefetch declined guard",
            prefetch_declined_guard as f64,
            "prefetches",
        );
        profile.counter(
            "weight offload prefetch joined",
            prefetch_joined as f64,
            "prefetches",
        );
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
    let mut engine = PipelineEngine::from_dir_with_config(model_dir, args.engine.to_config())
        .map_err(|error| {
            anyhow::anyhow!(
                "What: {} could not be loaded as a diffusion pipeline. \
                 Why: {error:#}. \
                 How: point --output-image at a package whose inference metadata declares a `pipeline` with `strategy.kind: iterative`.",
                model_dir.display()
            )
        })?;

    profile.decode_backend = Some(decode_backend_name(engine.decode_backend()).to_string());
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
    profile.decode_backend = Some(decode_backend_name(engine.decode_backend()).to_string());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_generate_defaults_to_stats_only_on_the_repl_tty_path() {
        assert!(initial_generate_show_stats(
            ReplInputMode::Tty,
            false,
            GenerateOutputKind::Text
        ));
        assert!(!initial_generate_show_stats(
            ReplInputMode::Plain,
            false,
            GenerateOutputKind::Text
        ));
        assert!(!initial_generate_show_stats(
            ReplInputMode::Tty,
            true,
            GenerateOutputKind::Text
        ));
    }

    #[test]
    fn non_text_generate_outputs_do_not_default_to_token_stats() {
        assert!(!initial_generate_show_stats(
            ReplInputMode::Tty,
            false,
            GenerateOutputKind::Image
        ));
        assert!(!initial_generate_show_stats(
            ReplInputMode::Tty,
            false,
            GenerateOutputKind::Audio
        ));
    }
}
