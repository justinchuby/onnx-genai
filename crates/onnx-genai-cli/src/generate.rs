use std::io::{self, IsTerminal};
use std::path::Path;

use anyhow::Context as _;
use onnx_genai::engine::PipelineEngine;
use onnx_genai::ort::{ChatMessage, Tokenizer};
use onnx_genai::text_to_audio;
use onnx_genai::text_to_image;
use onnx_genai_server::multimodal;

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
    let profile = RunProfile::new(model_dir.display().to_string());
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
    profile.execution_provider = backend.execution_provider_status();
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
            // `kv_usage` replaces `profile.memory` wholesale, so it must run
            // before anything that writes into that struct.
            if let Some(memory) = backend.kv_usage() {
                profile.memory = memory;
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
    materialize_ns: u64,
    htod_ns: u64,
    admit_sync_ns: u64,
    staging_fill_bytes: u64,
    staging_fill_regions: u64,
    staging_fill_calls: u64,
    materialize_fallback_calls: u64,
    htod_bytes: u64,
    vram_alloc_ns: u64,
    vram_free_ns: u64,
    budget_bytes: u64,
    peak_resident_bytes: u64,
    content_resident_bytes: u64,
    mapped_physical_bytes: u64,
    physical_owned_bytes: u64,
}

fn cuda_offload_stats() -> CudaOffloadSnapshot {
    #[cfg(feature = "native-cuda")]
    {
        let stats = onnx_runtime_ep_cuda::global_offload_stats();
        CudaOffloadSnapshot {
            page_ins: stats.page_ins,
            hits: stats.hits,
            evictions: stats.evictions,
            materialize_ns: stats.materialize_ns,
            htod_ns: stats.htod_ns,
            admit_sync_ns: stats.admit_sync_ns,
            staging_fill_bytes: stats.staging_fill_bytes,
            staging_fill_regions: stats.staging_fill_regions,
            staging_fill_calls: stats.staging_fill_calls,
            materialize_fallback_calls: stats.materialize_fallback_calls,
            htod_bytes: stats.htod_bytes,
            vram_alloc_ns: stats.vram_alloc_ns,
            vram_free_ns: stats.vram_free_ns,
            budget_bytes: stats.budget_bytes,
            peak_resident_bytes: stats.peak_resident_bytes,
            content_resident_bytes: stats.content_resident_bytes,
            mapped_physical_bytes: stats.mapped_physical_bytes,
            physical_owned_bytes: stats.physical_owned_bytes,
        }
    }
    #[cfg(not(feature = "native-cuda"))]
    {
        CudaOffloadSnapshot::default()
    }
}

fn record_cuda_offload_counters(profile: &mut RunProfile, before: CudaOffloadSnapshot) {
    let after = cuda_offload_stats();
    emit_cuda_offload_counters(profile, before, after);
}

/// Emit the weight-offload activity counters for the delta between two
/// snapshots. Split out from [`record_cuda_offload_counters`] so it is testable
/// without a CUDA device: the source of `after` is the only part that needs one.
///
/// Every counter this consults to decide whether the section is interesting
/// MUST also be emitted. An earlier version computed `page_ins`, `hits` and
/// `evictions`, gated the section on them, and then printed only the prefetch
/// counters — so the section could appear with every printed row reading zero
/// while the counters that actually fired stayed invisible. That made the
/// residency cache look inert when it was busy.
///
/// Every emitted row must also have a live writer in `weight_paging.rs`. Do not
/// keep schema-compatibility zeros: removing a dead column is safer than making
/// "not measured" look like "measured zero".
fn emit_cuda_offload_counters(
    profile: &mut RunProfile,
    before: CudaOffloadSnapshot,
    after: CudaOffloadSnapshot,
) {
    let page_ins = after.page_ins.saturating_sub(before.page_ins);
    let hits = after.hits.saturating_sub(before.hits);
    let evictions = after.evictions.saturating_sub(before.evictions);
    let materialize_ns = after.materialize_ns.saturating_sub(before.materialize_ns);
    let htod_ns = after.htod_ns.saturating_sub(before.htod_ns);
    let admit_sync_ns = after.admit_sync_ns.saturating_sub(before.admit_sync_ns);
    let staging_fill_bytes = after
        .staging_fill_bytes
        .saturating_sub(before.staging_fill_bytes);
    let staging_fill_regions = after
        .staging_fill_regions
        .saturating_sub(before.staging_fill_regions);
    let staging_fill_calls = after
        .staging_fill_calls
        .saturating_sub(before.staging_fill_calls);
    let materialize_fallback_calls = after
        .materialize_fallback_calls
        .saturating_sub(before.materialize_fallback_calls);
    let htod_bytes = after.htod_bytes.saturating_sub(before.htod_bytes);
    let vram_alloc_ns = after.vram_alloc_ns.saturating_sub(before.vram_alloc_ns);
    let vram_free_ns = after.vram_free_ns.saturating_sub(before.vram_free_ns);
    let budget_bytes = after.budget_bytes;
    let peak_resident_bytes = after.peak_resident_bytes.max(before.peak_resident_bytes);
    let content_resident_bytes = after.content_resident_bytes;
    let mapped_physical_bytes = after.mapped_physical_bytes;
    let physical_owned_bytes = after.physical_owned_bytes;
    if page_ins > 0
        || hits > 0
        || evictions > 0
        || materialize_ns > 0
        || htod_ns > 0
        || admit_sync_ns > 0
        || staging_fill_bytes > 0
        || staging_fill_regions > 0
        || staging_fill_calls > 0
        || materialize_fallback_calls > 0
        || htod_bytes > 0
        || vram_alloc_ns > 0
        || vram_free_ns > 0
        || budget_bytes > 0
        || peak_resident_bytes > 0
        || content_resident_bytes > 0
        || mapped_physical_bytes > 0
        || physical_owned_bytes > 0
    {
        profile.counter("weight offload page-ins", page_ins as f64, "page-ins");
        profile.counter("weight offload cache hits", hits as f64, "hits");
        profile.counter("weight offload evictions", evictions as f64, "evictions");
        let lookups = page_ins.saturating_add(hits);
        if lookups > 0 {
            profile.counter(
                "weight offload hit rate",
                (hits as f64 / lookups as f64) * 100.0,
                "%",
            );
        }
        profile.counter(
            "weight offload staging fill",
            materialize_ns as f64 / 1_000_000.0,
            "ms",
        );
        profile.counter(
            "weight offload staging fill bytes",
            staging_fill_bytes as f64,
            "bytes",
        );
        profile.counter(
            "weight offload staging fill regions",
            staging_fill_regions as f64,
            "regions",
        );
        profile.counter(
            "weight offload staging fill calls",
            staging_fill_calls as f64,
            "calls",
        );
        profile.counter(
            "weight offload materialize fallback calls",
            materialize_fallback_calls as f64,
            "calls",
        );
        profile.counter(
            "weight offload H2D copy",
            htod_ns as f64 / 1_000_000.0,
            "ms",
        );
        profile.counter("weight offload H2D bytes", htod_bytes as f64, "bytes");
        profile.counter(
            "weight offload VRAM alloc",
            vram_alloc_ns as f64 / 1_000_000.0,
            "ms",
        );
        profile.counter(
            "weight offload VRAM free",
            vram_free_ns as f64 / 1_000_000.0,
            "ms",
        );
        profile.counter("weight offload budget", budget_bytes as f64, "bytes");
        profile.counter(
            "weight offload peak resident",
            peak_resident_bytes as f64,
            "bytes",
        );
        profile.counter(
            "weight offload content resident",
            content_resident_bytes as f64,
            "bytes",
        );
        profile.counter(
            "weight offload mapped physical",
            mapped_physical_bytes as f64,
            "bytes",
        );
        profile.counter(
            "weight offload physical owned",
            physical_owned_bytes as f64,
            "bytes",
        );
        profile.counter(
            "weight offload admit sync",
            admit_sync_ns as f64 / 1_000_000.0,
            "ms",
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

    profile.execution_provider = engine.execution_provider_status();
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
    profile.execution_provider = engine.execution_provider_status();
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

    /// The gate in `emit_cuda_offload_counters` consults activity counters to
    /// decide the section is worth printing, so those counters must be printed.
    /// Before this test, `page_ins`, `hits` and `evictions` were consulted and
    /// then dropped: a run whose residency cache was paging hard printed a
    /// weight-offload section in which every visible row read `0.00`.
    #[test]
    fn every_counter_the_offload_gate_consults_is_also_emitted() {
        let before = CudaOffloadSnapshot::default();
        for (label, after) in [
            (
                "page_ins",
                CudaOffloadSnapshot {
                    page_ins: 7,
                    ..Default::default()
                },
            ),
            (
                "hits",
                CudaOffloadSnapshot {
                    hits: 7,
                    ..Default::default()
                },
            ),
            (
                "evictions",
                CudaOffloadSnapshot {
                    evictions: 7,
                    ..Default::default()
                },
            ),
        ] {
            let mut profile = RunProfile::new("test".to_string());
            emit_cuda_offload_counters(&mut profile, before, after);
            assert!(
                profile.counters.iter().any(|counter| counter.value == 7.0),
                "{label} made the offload section print, but no emitted row \
                 carries its value — the counter is invisible to the operator"
            );
        }
    }

    #[test]
    fn dead_offload_counters_are_not_emitted_as_compatibility_zeroes() {
        let mut profile = RunProfile::new("test".to_string());
        emit_cuda_offload_counters(
            &mut profile,
            CudaOffloadSnapshot::default(),
            CudaOffloadSnapshot {
                page_ins: 1,
                hits: 1,
                evictions: 1,
                materialize_ns: 1,
                htod_ns: 1,
                admit_sync_ns: 1,
                staging_fill_bytes: 1,
                staging_fill_regions: 1,
                staging_fill_calls: 1,
                materialize_fallback_calls: 1,
                htod_bytes: 1,
                vram_alloc_ns: 1,
                vram_free_ns: 1,
                budget_bytes: 1,
                peak_resident_bytes: 1,
                ..Default::default()
            },
        );

        for dead_name_fragment in ["copy fence", "prefetch", "pinned staging"] {
            assert!(
                !profile
                    .counters
                    .iter()
                    .any(|counter| counter.name.contains(dead_name_fragment)),
                "{dead_name_fragment} counters have no writer and must not be emitted as 0.00"
            );
        }
    }

    #[test]
    fn offload_hit_rate_is_reported_as_a_percentage_of_lookups() {
        let mut profile = RunProfile::new("test".to_string());
        emit_cuda_offload_counters(
            &mut profile,
            CudaOffloadSnapshot::default(),
            CudaOffloadSnapshot {
                page_ins: 3,
                hits: 1,
                ..Default::default()
            },
        );
        let rate = profile
            .counters
            .iter()
            .find(|counter| counter.name == "weight offload hit rate")
            .expect("hit rate is emitted when the cache was looked up");
        assert_eq!(rate.value, 25.0);
        assert_eq!(rate.unit, "%");
    }

    /// A cache that was never consulted has no hit rate; reporting `0%` would
    /// be indistinguishable from a cache that missed on everything.
    #[test]
    fn offload_hit_rate_is_omitted_when_the_cache_was_never_looked_up() {
        let mut profile = RunProfile::new("test".to_string());
        emit_cuda_offload_counters(
            &mut profile,
            CudaOffloadSnapshot::default(),
            CudaOffloadSnapshot {
                evictions: 2,
                ..Default::default()
            },
        );
        assert!(
            !profile
                .counters
                .iter()
                .any(|counter| counter.name == "weight offload hit rate")
        );
    }
}
