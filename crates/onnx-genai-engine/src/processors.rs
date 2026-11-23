//! Request logit processor construction and token selection helpers.

use crate::FimConfig;
use crate::config::{
    FinishReason, GenerateConstraint, GenerateOptions, SamplerSequence, SamplerStage,
};
use crate::logits::{
    AdaptivePProcessor, ConstraintProcessor, DryProcessor, GrammarConstraintKind, InfillContract,
    InfillProcessor, JsonConstraint, LlguidanceConstraint, MinPProcessor, MirostatProcessor,
    PenaltiesProcessor, ProcessorChain, ProcessorContext, ProcessorSignal, StopSequence,
    StopSequenceProcessor, TemperatureProcessor, TokenId, TopAProcessor, TopKProcessor,
    TopNSigmaProcessor, TopPProcessor, TypicalPProcessor, XtcProcessor,
};
use crate::sampling::{Sampler, default_sampler_for_options};
use anyhow::Context;
use onnx_genai_ort::Tokenizer;
use std::path::Path;

/// Build mandatory safety processors and the effective ordered sampler list.
/// An explicit greedy policy bypasses stochastic warpers; a temperature-zero
/// stage instead creates its point mass at the requested position. Hard bias,
/// grammar and stop contracts cannot be omitted through a custom stage list.
/// A caller-supplied terminal sampler still receives the configured distribution.
pub(crate) fn build_processor_chain(
    options: &GenerateOptions,
    tokenizer: Option<&Tokenizer>,
    has_custom_sampler: bool,
) -> anyhow::Result<ProcessorChain> {
    let mut chain = ProcessorChain::new();
    // A temperature-zero stage still runs at its requested position. Only an
    // explicit greedy policy bypasses stochastic warpers before terminal selection.
    let samples = !options.greedy || has_custom_sampler;
    if !options.logit_bias.is_empty() {
        if let Some(tokenizer) = tokenizer {
            options.validate_logit_bias(tokenizer)?;
        }
        chain.add(Box::new(crate::logits::LogitBiasProcessor {
            biases: options.logit_bias.clone(),
        }));
    }
    if !options.stop_sequences.is_empty() {
        chain.add(Box::new(StopSequenceProcessor::new(
            options.stop_sequences.clone(),
        )));
    }

    if let Some(constraint) = &options.constraint {
        let tokenizer = tokenizer.context("constrained decoding requires a tokenizer")?;
        let token_texts = tokenizer_token_texts(tokenizer);
        match constraint {
            GenerateConstraint::Json { prefix } => {
                chain.add(Box::new(ConstraintProcessor::new(
                    Box::new(JsonConstraint::new(prefix.clone())?),
                    token_texts,
                    terminal_token_ids(options),
                )));
            }
            GenerateConstraint::JsonSchema(schema) => {
                chain.add(Box::new(ConstraintProcessor::new(
                    build_llguidance_constraint(
                        GrammarConstraintKind::JsonSchema,
                        schema,
                        tokenizer,
                        &token_texts,
                        options.eos_token_id,
                    )?,
                    token_texts,
                    terminal_token_ids(options),
                )));
            }
            GenerateConstraint::Regex(regex) => {
                chain.add(Box::new(ConstraintProcessor::new(
                    build_llguidance_constraint(
                        GrammarConstraintKind::Regex,
                        regex,
                        tokenizer,
                        &token_texts,
                        options.eos_token_id,
                    )?,
                    token_texts,
                    terminal_token_ids(options),
                )));
            }
            GenerateConstraint::Lark(grammar) => {
                chain.add(Box::new(ConstraintProcessor::new(
                    build_llguidance_constraint(
                        GrammarConstraintKind::Lark,
                        grammar,
                        tokenizer,
                        &token_texts,
                        options.eos_token_id,
                    )?,
                    token_texts,
                    terminal_token_ids(options),
                )));
            }
        }
    }

    chain.begin_optional_stages();
    if let Some(mirostat) = options.mirostat.filter(|_| samples) {
        // Mirostat owns selection policy and, like upstream, replaces the
        // ordered warper list with fixed temperature followed by Mirostat.
        if options.temperature != 1.0 {
            chain.add(Box::new(TemperatureProcessor {
                temperature: options.temperature,
                dynamic: None,
            }));
        }
        chain.add(Box::new(MirostatProcessor::new(
            mirostat.tau,
            mirostat.eta,
            mirostat.version,
        )));
        return Ok(chain);
    }
    let stages = options
        .sampler_order
        .as_ref()
        .map_or(SamplerSequence::DEFAULT, |sequence| sequence.stages());
    let infill = if samples && stages.contains(&SamplerStage::Infill) {
        let tokenizer = tokenizer.context("infill sampling requires a declared tokenizer; load a package with tokenizer byte pieces")?;
        let endings = terminal_token_ids(options);
        let fallback = options.eot_token_id.or(options.eos_token_id).or_else(||endings.first().copied())
            .context("infill sampling requires a declared EOS/end-of-turn role; set package.tokenizer.special_tokens.eos_token_id")?;
        Some(std::sync::Arc::new(InfillContract::new(
            tokenizer, endings, fallback,
        )?))
    } else {
        None
    };
    for &stage in stages {
        append_sampler_stage(&mut chain, options, stage, samples, infill.as_ref())?;
    }
    // Adaptive-p owns terminal selection regardless of its listed position;
    // duplicate selectors still create one terminal transform, as upstream.
    if samples && stages.contains(&SamplerStage::AdaptiveP) && options.adaptive_p.target >= 0.0 {
        chain.add(Box::new(AdaptivePProcessor::new(options.adaptive_p)?));
    }
    Ok(chain)
}

fn append_sampler_stage(
    chain: &mut ProcessorChain,
    options: &GenerateOptions,
    stage: SamplerStage,
    samples: bool,
    infill: Option<&std::sync::Arc<InfillContract>>,
) -> anyhow::Result<()> {
    if !samples && !matches!(stage, SamplerStage::Penalties | SamplerStage::Dry) {
        return Ok(());
    }
    match stage {
        SamplerStage::AdaptiveP => {}
        SamplerStage::Infill => {
            chain.add(Box::new(InfillProcessor {
                contract: infill
                    .context("infill sampling contract was not resolved before stage construction")?
                    .clone(),
            }));
        }
        SamplerStage::Penalties => {
            if options.penalty_window != Some(0)
                && (options.repetition_penalty != 1.0
                    || options.frequency_penalty != 0.0
                    || options.presence_penalty != 0.0)
            {
                chain.add(Box::new(PenaltiesProcessor {
                    repetition_penalty: options.repetition_penalty,
                    frequency_penalty: options.frequency_penalty,
                    presence_penalty: options.presence_penalty,
                    window: options.penalty_window,
                }));
            }
        }
        SamplerStage::Dry => {
            if let Some(dry) = &options.dry
                && dry.multiplier > 0.0
                && dry.window != Some(0)
            {
                chain.add(Box::new(DryProcessor {
                    multiplier: dry.multiplier,
                    base: dry.base,
                    allowed_length: dry.allowed_length,
                    sequence_breakers: dry.sequence_breakers.clone(),
                    window: dry.window,
                }));
            }
        }
        SamplerStage::Temperature => {
            if options.temperature != 1.0
                || options
                    .dynamic_temperature
                    .is_some_and(|config| config.range > 0.0)
            {
                chain.add(Box::new(TemperatureProcessor {
                    temperature: options.temperature,
                    dynamic: options.dynamic_temperature,
                }));
            }
        }
        SamplerStage::TopNSigma => {
            if options.top_n_sigma > 0.0 {
                chain.add(Box::new(TopNSigmaProcessor {
                    n: options.top_n_sigma,
                }));
            }
        }
        SamplerStage::TopK => {
            if options.top_k > 0 {
                chain.add(Box::new(TopKProcessor {
                    top_k: options.top_k,
                }));
            }
        }
        SamplerStage::TopP => {
            if options.top_p < 1.0 && !chain.fuse_last_top_k_with_top_p(options.top_p) {
                chain.add(Box::new(TopPProcessor {
                    top_p: options.top_p,
                }));
            }
        }
        SamplerStage::MinP => {
            if options.min_p > 0.0 {
                chain.add(Box::new(MinPProcessor {
                    min_p: options.min_p,
                }));
            }
        }
        SamplerStage::TopA => {
            if options.top_a > 0.0 {
                chain.add(Box::new(TopAProcessor {
                    top_a: options.top_a,
                }));
            }
        }
        SamplerStage::TypicalP => {
            if options.typical_p < 1.0 {
                chain.add(Box::new(TypicalPProcessor {
                    typical_p: options.typical_p,
                }));
            }
        }
        SamplerStage::Xtc => {
            if let Some(xtc) = options.xtc.filter(|config| config.probability > 0.0) {
                chain.add(Box::new(XtcProcessor::new(
                    xtc.probability,
                    xtc.threshold,
                    options.seed,
                )));
            }
        }
    }
    Ok(())
}

/// Expand cached typed transforms into the fixed device sampler's operations.
/// Processor names are diagnostics and carry no dispatch authority.
fn flattened_device_transforms(
    chain: &ProcessorChain,
) -> Option<impl Iterator<Item = crate::logits::DeviceSamplingTransform> + '_> {
    use crate::logits::DeviceSamplingTransform as Transform;
    Some(
        chain
            .device_transforms()?
            .iter()
            .flat_map(|&transform| match transform {
                Transform::TopKTopP { top_k, top_p } => {
                    [Some(Transform::TopK(top_k)), Some(Transform::TopP(top_p))]
                }
                transform => [Some(transform), None],
            })
            .flatten(),
    )
}

pub(crate) fn is_device_portable_chain(chain: &ProcessorChain) -> bool {
    use crate::logits::DeviceSamplingTransform as Transform;
    let Some(transforms) = flattened_device_transforms(chain) else {
        return false;
    };
    let mut previous = None;
    for transform in transforms {
        let rank = match transform {
            Transform::Temperature(_) => 0,
            Transform::TopK(_) => 1,
            Transform::TopP(_) => 2,
            Transform::MinP(_) => 3,
            Transform::TopKTopP { .. } => unreachable!("fused transforms expanded above"),
        };
        if previous.is_some_and(|previous| previous >= rank) {
            return false;
        }
        previous = Some(rank);
    }
    true
}

fn device_pipeline_matches_options(chain: &ProcessorChain, options: &GenerateOptions) -> bool {
    use crate::logits::DeviceSamplingTransform as Transform;
    let Some(actual) = flattened_device_transforms(chain) else {
        return false;
    };
    let expected = [
        (options.temperature != 1.0).then_some(Transform::Temperature(options.temperature)),
        (options.top_k > 0).then_some(Transform::TopK(options.top_k)),
        (options.top_p < 1.0).then_some(Transform::TopP(options.top_p)),
        (options.min_p > 0.0).then_some(Transform::MinP(options.min_p)),
    ];
    actual.into_iter().eq(expected.into_iter().flatten())
}

/// How a single request's next token can be selected: entirely on the device
/// (copying back only a 4-byte token id) or on the host from the full
/// vocabulary.
///
/// This is the one authoritative answer to "can this row be sampled on device".
/// The single-request decode loop ([`crate::decode_loop`]) and the continuous
/// batch manager's per-row router both call [`device_sampling_plan`], so a new
/// disqualifying condition is added in exactly one place instead of drifting
/// between two inline copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeviceSamplingPlan {
    /// Greedy (argmax) selection on the device.
    Greedy,
    /// Categorical selection with the device-portable filter pipeline
    /// (temperature / top-k / top-p / min-p) on the device.
    Sampled,
    /// The row needs the full host logits: a history-dependent processor
    /// (repetition/frequency/presence penalty, grammar, stop sequences), a
    /// `top_logprobs` request, a custom sampler, a non-portable chain, or a
    /// backend that cannot sample on the device.
    Host,
}

/// Decide whether `options` + `chain` can be sampled on the device.
///
/// `has_custom_sampler` disqualifies the device path because a foreign sampler
/// must see processed host logits. `greedy_fastpath_supported` and
/// `sampled_fastpath_supported` are the backend's device-sampling capabilities;
/// when both are false the verdict is always [`DeviceSamplingPlan::Host`].
pub(crate) fn device_sampling_plan(
    chain: &ProcessorChain,
    options: &GenerateOptions,
    has_custom_sampler: bool,
    greedy_fastpath_supported: bool,
    sampled_fastpath_supported: bool,
) -> DeviceSamplingPlan {
    // A custom sampler replaces the default greedy/categorical selection, so the
    // device fast paths must be bypassed to give the sampler processed logits.
    //
    // The greedy test is argmax-equivalence, not emptiness: a chain of pure
    // truncation processors cannot move the maximum, and greedy selection reads
    // nothing but the maximum. `top_logprobs` is still excluded because it
    // reports the processed distribution, which the fast path never forms.
    let greedy = chain.preserves_argmax()
        && options.top_logprobs.is_none()
        && options.selects_greedily()
        && !has_custom_sampler
        && greedy_fastpath_supported;
    if greedy {
        return DeviceSamplingPlan::Greedy;
    }
    // Keep greedy behavior on its existing argmax path. The sampled path is only
    // for categorical decoding whose processor chain the device sampler supports.
    let sampled = !options.selects_greedily()
        && options.top_logprobs.is_none()
        && !has_custom_sampler
        && options.temperature > 0.0
        && is_device_portable_chain(chain)
        && device_pipeline_matches_options(chain, options)
        && sampled_fastpath_supported;
    if sampled {
        DeviceSamplingPlan::Sampled
    } else {
        DeviceSamplingPlan::Host
    }
}

pub(crate) fn load_fim_config_from_model_dir(
    model_dir: &Path,
) -> anyhow::Result<Option<FimConfig>> {
    let tokenizer_config = model_dir.join("tokenizer_config.json");
    if !tokenizer_config.is_file() {
        return Ok(None);
    }

    let text = std::fs::read_to_string(&tokenizer_config)
        .with_context(|| format!("failed to read {}", tokenizer_config.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("invalid JSON in {}", tokenizer_config.display()))?;
    Ok(FimConfig::from_tokenizer_config(&value))
}

pub(crate) fn push_unique_stop_sequence(
    stop_sequences: &mut Vec<StopSequence>,
    stop: StopSequence,
) {
    if !stop_sequences.contains(&stop) {
        stop_sequences.push(stop);
    }
}

fn build_llguidance_constraint(
    kind: GrammarConstraintKind,
    grammar: &str,
    tokenizer: &Tokenizer,
    token_texts: &[Option<String>],
    eos_token_id: Option<TokenId>,
) -> anyhow::Result<Box<dyn crate::logits::Constraint>> {
    match LlguidanceConstraint::from_hf_tokenizer(
        kind,
        grammar,
        tokenizer.inner(),
        token_texts.len(),
        eos_token_id,
    ) {
        Ok(constraint) => Ok(Box::new(constraint)),
        Err(hf_error) => LlguidanceConstraint::from_token_texts(
            kind,
            grammar,
            token_texts,
            eos_token_id,
        )
        .map(|constraint| Box::new(constraint) as Box<dyn crate::logits::Constraint>)
        .with_context(|| {
            format!(
                "failed to initialize llguidance with HuggingFace tokenizer ({hf_error}) or decoded-token fallback"
            )
        }),
    }
}

fn constrained_json_is_complete(options: &GenerateOptions, generated: &str) -> Option<bool> {
    let GenerateConstraint::Json { prefix } = options.constraint.as_ref()? else {
        return None;
    };
    if prefix.is_empty() {
        Some(JsonConstraint::is_complete(generated))
    } else {
        Some(JsonConstraint::is_complete(&format!("{prefix}{generated}")))
    }
}

pub(crate) fn ensure_constrained_finish(
    options: &GenerateOptions,
    generated_text: &str,
    finish_reason: FinishReason,
) -> anyhow::Result<()> {
    if matches!(
        finish_reason,
        FinishReason::MaxTokens | FinishReason::Length
    ) && constrained_json_is_complete(options, generated_text) == Some(false)
    {
        anyhow::bail!(
            "JSON constrained decoding stopped before a complete JSON value; increase max_new_tokens or max_context"
        );
    }
    Ok(())
}

fn tokenizer_token_texts(tokenizer: &Tokenizer) -> Vec<Option<String>> {
    let vocab = tokenizer.inner().get_vocab(true);
    let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
    let mut token_texts = vec![None; max_id + 1];
    for (id, token_text) in token_texts.iter_mut().enumerate() {
        *token_text = tokenizer.decode(&[id as TokenId]).ok();
    }
    token_texts
}

pub(crate) fn select_next_token(
    logits: &mut [f32],
    context: &ProcessorContext,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    rng_value: f32,
) -> anyhow::Result<TokenId> {
    let mut sampler = default_sampler_for_options(options, rng_value);
    select_next_token_with_sampler(logits, context, chain, &mut sampler)
}

pub(crate) fn select_next_token_with_rng(
    logits: &mut [f32],
    context: &ProcessorContext,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    rng: &mut crate::sampling::SamplingRng,
) -> anyhow::Result<TokenId> {
    select_next_token(logits, context, options, chain, rng.value_for(options))
}

pub(crate) fn select_next_token_with_sampler(
    logits: &mut [f32],
    context: &ProcessorContext,
    chain: &ProcessorChain,
    sampler: &mut dyn Sampler,
) -> anyhow::Result<TokenId> {
    chain.process(logits, context)?;
    let token = sampler.sample(logits, context);
    chain.on_selected(token, context)?;
    Ok(token)
}

pub(crate) fn finish_reason_after_token(
    token_id: TokenId,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    context: &ProcessorContext,
) -> Option<FinishReason> {
    if options.terminates(token_id) {
        return Some(FinishReason::EosToken);
    }

    match chain.signal(context) {
        Some(ProcessorSignal::StopSequence { index })
            if constrained_json_is_complete(options, &context.generated_text) != Some(false) =>
        {
            Some(FinishReason::StopSequence { index })
        }
        Some(ProcessorSignal::StopSequence { .. }) => None,
        None => None,
    }
}

#[cfg(test)]
mod device_portability_tests {
    use super::*;
    use crate::config::GenerateOptions;

    /// A *sampling* request with the given truncation controls.
    ///
    /// `GenerateOptions::default()` is greedy (`greedy: true`), so `..Default`
    /// alone would build no warpers at all now and these device-sampler tests
    /// would vacuously pass on an empty chain. Every test here that is about
    /// the sampled path has to say so.
    fn options_with(top_k: usize, top_p: f32) -> GenerateOptions {
        GenerateOptions {
            top_k,
            top_p,
            greedy: false,
            ..Default::default()
        }
    }

    #[test]
    fn dynamic_temperature_requires_the_host_distribution_and_valid_controls() {
        let mut options = GenerateOptions {
            greedy: false,
            temperature: 1.0,
            dynamic_temperature: Some(crate::config::DynamicTemperatureConfig {
                range: 0.4,
                exponent: 0.0,
            }),
            ..Default::default()
        };
        options.validate().unwrap();
        let chain = build_processor_chain(&options, None, false).unwrap();
        assert_eq!(chain.names(), vec!["dynamic_temperature"]);
        assert!(!is_device_portable_chain(&chain));
        assert_eq!(
            device_sampling_plan(&chain, &options, false, true, true),
            DeviceSamplingPlan::Host
        );
        for (range, exponent, field) in [
            (-0.1, 1.0, "dynatemp_range"),
            (f32::INFINITY, 1.0, "dynatemp_range"),
            (0.4, f32::NAN, "dynatemp_exponent"),
            (0.4, -1.0, "dynatemp_exponent"),
        ] {
            options.dynamic_temperature =
                Some(crate::config::DynamicTemperatureConfig { range, exponent });
            assert!(options.validate().unwrap_err().to_string().contains(field));
        }
    }

    /// The standard sampling configuration must stay eligible for the device
    /// sampling fast path.
    ///
    /// Fused top-k/top-p carries the same typed transform proof as its separate
    /// stages, preserving the standard device sampling configuration.
    #[test]
    fn a_top_k_top_p_chain_is_device_portable() {
        let chain = build_processor_chain(&options_with(40, 0.95), None, false)
            .expect("standard sampling options build a chain");
        assert!(
            chain.names().contains(&"top_k_top_p"),
            "expected the fused processor, got {:?}",
            chain.names()
        );
        assert!(
            is_device_portable_chain(&chain),
            "the most common sampling configuration must reach the device \
             sampler, got {:?}",
            chain.names()
        );
    }

    fn greedy_options(base: GenerateOptions) -> GenerateOptions {
        GenerateOptions {
            greedy: true,
            ..base
        }
    }

    /// Every sampling warper a request can carry, all at once, on a *sampling*
    /// request.
    fn every_warper() -> GenerateOptions {
        GenerateOptions {
            greedy: false,
            temperature: 0.7,
            dynamic_temperature: Some(crate::config::DynamicTemperatureConfig {
                range: 0.2,
                exponent: 1.0,
            }),
            top_k: 64,
            top_p: 0.95,
            min_p: 0.05,
            top_a: 0.2,
            typical_p: 0.5,
            xtc: Some(crate::config::XtcConfig {
                probability: 0.5,
                threshold: 0.1,
            }),
            ..Default::default()
        }
    }

    const WARPER_NAMES: &[&str] = &[
        "temperature",
        "dynamic_temperature",
        "top_k",
        "top_p",
        "top_k_top_p",
        "min_p",
        "top_a",
        "typical_p",
        "mirostat_v2",
        "xtc",
    ];

    /// A greedy request builds no sampling warpers, whatever the model's
    /// `generation` defaults say.
    ///
    /// This is the whole point: greedy means the maximum logit wins, so a
    /// request that asks for greedy and *also* carries `top_k` / `top_p` /
    /// `typical_p` (because the model card ships them and they ride along on
    /// every request) must not have its distribution reshaped on the way to an
    /// argmax. `transformers` draws the same line -- `do_sample=False` never
    /// calls `_get_logits_warper`.
    #[test]
    fn a_greedy_request_builds_no_sampling_warpers() {
        let chain = build_processor_chain(&greedy_options(every_warper()), None, false)
            .expect("greedy options build a chain");
        assert!(
            chain.names().is_empty(),
            "greedy selection must not reshape the distribution, got {:?}",
            chain.names()
        );
    }

    /// Negative control for the test above.
    ///
    /// Without it, a `build_processor_chain` that returned an empty chain for
    /// *every* request would pass, and sampling would silently become greedy.
    #[test]
    fn a_sampling_request_builds_every_warper_it_asked_for() {
        for dynamic_temperature in [None, every_warper().dynamic_temperature] {
            let options = GenerateOptions {
                dynamic_temperature,
                ..every_warper()
            };
            let chain = build_processor_chain(&options, None, false)
                .expect("sampling options build a chain");
            let names = chain.names();
            let temperature_stage = if dynamic_temperature.is_some() {
                "dynamic_temperature"
            } else {
                "temperature"
            };
            assert!(names.contains(&temperature_stage), "{names:?}");
            for warper in WARPER_NAMES {
                if matches!(
                    *warper,
                    "top_k_top_p" | "mirostat_v2" | "temperature" | "dynamic_temperature"
                ) {
                    // Typical-p intervenes here, so K and P remain separate; Mirostat owns a distinct selection mode.
                    continue;
                }
                assert!(
                    names.contains(warper),
                    "sampling must keep {warper}, got {names:?}"
                );
            }
        }
    }

    /// The controls that are *not* warpers survive a greedy request, because
    /// they express what the caller wants generated rather than how to sample.
    #[test]
    fn a_greedy_request_keeps_penalties_and_stop_sequences() {
        let options = greedy_options(GenerateOptions {
            top_k: 64,
            top_p: 0.95,
            repetition_penalty: 1.1,
            presence_penalty: 0.5,
            stop_sequences: vec![crate::logits::StopSequence::Text("<|end|>".into())],
            ..Default::default()
        });
        let chain = build_processor_chain(&options, None, false).expect("options build a chain");
        assert_eq!(
            chain.names(),
            vec!["stop_sequence", "penalties"],
            "only the sampling warpers are dropped under greedy selection"
        );
    }

    #[test]
    fn zero_penalty_window_keeps_the_greedy_device_path_available() {
        let options = greedy_options(GenerateOptions {
            repetition_penalty: 1.5,
            frequency_penalty: 2.0,
            presence_penalty: 2.0,
            penalty_window: Some(0),
            ..Default::default()
        });
        let chain = build_processor_chain(&options, None, false).unwrap();
        assert!(chain.is_empty());
        assert_eq!(
            device_sampling_plan(&chain, &options, false, true, true),
            DeviceSamplingPlan::Greedy
        );
    }

    /// A custom sampler replaces the default selection, so it is not
    /// necessarily greedy and must still receive the configured distribution
    /// even when `greedy` is set on the request.
    #[test]
    fn a_custom_sampler_keeps_the_warpers_a_greedy_request_would_drop() {
        let options = greedy_options(every_warper());
        let chain =
            build_processor_chain(&options, None, true).expect("custom-sampler chain builds");
        assert!(
            chain.names().contains(&"top_k") && chain.names().contains(&"top_p"),
            "a custom sampler samples; it needs the distribution, got {:?}",
            chain.names()
        );
    }

    /// Temperature zero is a point-mass stage at its requested position;
    /// an explicit global greedy policy bypasses those warpers entirely.
    #[test]
    fn zero_temperature_runs_at_its_requested_stage() {
        let options = GenerateOptions {
            greedy: false,
            temperature: 0.0,
            sampler_order: Some(
                SamplerSequence::new(vec![SamplerStage::Temperature, SamplerStage::TopK]).unwrap(),
            ),
            top_k: 2,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false).unwrap();
        assert!(chain.names().contains(&"temperature"));
        assert!(options.selects_greedily());
        let mut logits = vec![3.0, 1.0, 2.0];
        chain
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits, vec![3.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        let raw = build_processor_chain(&greedy_options(options), None, false).unwrap();
        assert!(raw.is_empty());
    }

    /// The chat-model shape -- greedy plus the model's sampler defaults plus a
    /// template stop sequence -- reaches the device argmax.
    ///
    /// Two independent gates used to close this path. `chain.is_empty()` was
    /// one: it read a stop sequence as "sampling is configured". The other was
    /// `PipelineDecodeLoopBackend` never overriding `greedy_fastpath_supported`.
    #[test]
    fn a_greedy_chat_request_plans_the_device_argmax() {
        let options = greedy_options(GenerateOptions {
            top_k: 64,
            top_p: 0.95,
            stop_sequences: vec![crate::logits::StopSequence::Text("<|end|>".into())],
            ..Default::default()
        });
        let chain = build_processor_chain(&options, None, false).expect("options build a chain");
        assert_eq!(
            chain.names(),
            vec!["stop_sequence"],
            "the warpers should already be gone"
        );
        assert_eq!(
            device_sampling_plan(&chain, &options, false, true, true),
            DeviceSamplingPlan::Greedy,
            "the stop-sequence processor never touches logits"
        );
    }

    /// Negative control: a processor that survives greedy *and* rewrites logits
    /// must force the host path, because the device argmax reads raw logits.
    #[test]
    fn a_penalty_that_survives_greedy_denies_the_device_argmax() {
        for options in [
            greedy_options(GenerateOptions {
                repetition_penalty: 1.1,
                ..Default::default()
            }),
            greedy_options(GenerateOptions {
                presence_penalty: 0.5,
                ..Default::default()
            }),
        ] {
            let chain =
                build_processor_chain(&options, None, false).expect("options build a chain");
            assert!(!chain.is_empty(), "the penalty must survive greedy");
            assert_eq!(
                device_sampling_plan(&chain, &options, false, true, true),
                DeviceSamplingPlan::Host,
                "a per-token logit rewrite decides the argmax, so it must be applied"
            );
        }
    }

    /// Dropping the warpers must not change which token greedy selects. Checked
    /// against the processors themselves rather than argued from their names.
    #[test]
    fn dropping_the_truncation_warpers_does_not_move_the_greedy_token() {
        let raw: Vec<f32> = vec![0.5, -3.0, 9.25, 1.0, 8.5, -1.0, 2.0, 0.0];
        let sampling_chain = build_processor_chain(&options_with(4, 0.5), None, false)
            .expect("sampling options build a chain");
        let mut processed = raw.clone();
        sampling_chain
            .process(&mut processed, &ProcessorContext::default())
            .unwrap();
        assert_ne!(
            processed, raw,
            "this test is only meaningful if the warpers actually rewrote logits"
        );
        assert_eq!(
            crate::sampling::sample_greedy(&processed),
            crate::sampling::sample_greedy(&raw),
            "top-k/top-p mask relative to the maximum, so the maximum survives"
        );
    }

    /// Each half alone must remain portable too.
    #[test]
    fn top_k_alone_and_top_p_alone_stay_device_portable() {
        for (top_k, top_p, expected) in [(40usize, 1.0f32, "top_k"), (0, 0.95, "top_p")] {
            let chain = build_processor_chain(&options_with(top_k, top_p), None, false)
                .expect("chain builds");
            assert!(
                chain.names().contains(&expected),
                "expected {expected}, got {:?}",
                chain.names()
            );
            assert!(
                is_device_portable_chain(&chain),
                "{expected} alone must stay device portable, got {:?}",
                chain.names()
            );
        }
    }
}

#[cfg(test)]
mod device_sampling_plan_tests {
    use super::*;
    use crate::config::GenerateOptions;

    fn empty_chain() -> ProcessorChain {
        build_processor_chain(&GenerateOptions::default(), None, false).expect("empty chain builds")
    }

    fn history_chain() -> ProcessorChain {
        let options = GenerateOptions {
            repetition_penalty: 1.2,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false).expect("penalty chain builds");
        assert!(!chain.is_empty(), "repetition penalty must add a processor");
        assert!(
            !is_device_portable_chain(&chain),
            "repetition penalty must be non-portable"
        );
        chain
    }

    fn greedy_options() -> GenerateOptions {
        GenerateOptions {
            greedy: true,
            ..Default::default()
        }
    }

    fn sampled_options() -> GenerateOptions {
        GenerateOptions {
            greedy: false,
            temperature: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn greedy_options_with_support_and_empty_chain_plan_greedy() {
        assert_eq!(
            device_sampling_plan(&empty_chain(), &greedy_options(), false, true, true),
            DeviceSamplingPlan::Greedy
        );
    }

    #[test]
    fn sampled_options_with_support_and_portable_chain_plan_sampled() {
        assert_eq!(
            device_sampling_plan(&empty_chain(), &sampled_options(), false, true, true),
            DeviceSamplingPlan::Sampled
        );
    }

    // Each disqualifying condition, applied on its own, must force the host
    // verdict. These pin the predicate so a new caller (the continuous-batch
    // per-row router) cannot silently route a host-only request to the device.

    #[test]
    fn top_logprobs_forces_host() {
        let mut greedy = greedy_options();
        greedy.top_logprobs = Some(3);
        assert_eq!(
            device_sampling_plan(&empty_chain(), &greedy, false, true, true),
            DeviceSamplingPlan::Host
        );
        let mut sampled = sampled_options();
        sampled.top_logprobs = Some(3);
        assert_eq!(
            device_sampling_plan(&empty_chain(), &sampled, false, true, true),
            DeviceSamplingPlan::Host
        );
    }

    #[test]
    fn custom_sampler_forces_host() {
        assert_eq!(
            device_sampling_plan(&empty_chain(), &greedy_options(), true, true, true),
            DeviceSamplingPlan::Host
        );
        assert_eq!(
            device_sampling_plan(&empty_chain(), &sampled_options(), true, true, true),
            DeviceSamplingPlan::Host
        );
    }

    #[test]
    fn non_empty_or_non_portable_chain_forces_host() {
        // A history-dependent processor makes the chain non-empty (disqualifies
        // greedy) and non-portable (disqualifies sampled).
        assert_eq!(
            device_sampling_plan(&history_chain(), &greedy_options(), false, true, true),
            DeviceSamplingPlan::Host
        );
        assert_eq!(
            device_sampling_plan(&history_chain(), &sampled_options(), false, true, true),
            DeviceSamplingPlan::Host
        );
    }

    #[test]
    fn backend_without_support_forces_host() {
        assert_eq!(
            device_sampling_plan(&empty_chain(), &greedy_options(), false, false, false),
            DeviceSamplingPlan::Host
        );
        assert_eq!(
            device_sampling_plan(&empty_chain(), &sampled_options(), false, false, false),
            DeviceSamplingPlan::Host
        );
    }

    #[test]
    fn greedy_needs_greedy_support_specifically() {
        // Sampled support alone does not license the greedy path, and greedy
        // support alone does not license the sampled path.
        assert_eq!(
            device_sampling_plan(&empty_chain(), &greedy_options(), false, false, true),
            DeviceSamplingPlan::Host
        );
        assert_eq!(
            device_sampling_plan(&empty_chain(), &sampled_options(), false, true, false),
            DeviceSamplingPlan::Host
        );
    }
}

/// Every token id that ends generation for this request.
///
/// A constrained decode asks "may this candidate finish the output?", which is
/// the same question [`GenerateOptions::terminates`] answers for the committed
/// token. Deriving both from one set is what keeps a grammar from treating a
/// model's second end token as ordinary text while the loop treats it as
/// terminal.
fn terminal_token_ids(options: &GenerateOptions) -> Vec<TokenId> {
    let mut ids = options.eos_token_ids.clone();
    if let Some(id) = options.eos_token_id
        && !ids.contains(&id)
    {
        ids.push(id);
    }
    ids
}

#[cfg(test)]
mod logit_bias_tests {
    use super::*;
    use crate::logits::LogitBias;
    #[test]
    fn bias_changes_greedy_argmax_and_excludes_device_shortcuts() {
        let options = GenerateOptions {
            logit_bias: [(0, LogitBias::Ban), (2, LogitBias::Add(10.0))].into(),
            ..GenerateOptions::default()
        };
        let chain = build_processor_chain(&options, None, false).unwrap();
        assert!(!is_device_portable_chain(&chain));
        let mut logits = vec![9.0, 3.0, 1.0];
        chain
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits, vec![f32::NEG_INFINITY, 3.0, 11.0]);
        assert_eq!(crate::sampling::sample_greedy(&logits), 2);
        let invalid = GenerateOptions {
            logit_bias: [(0, LogitBias::Add(f32::NAN))].into(),
            ..GenerateOptions::default()
        };
        assert!(
            invalid
                .validate()
                .unwrap_err()
                .to_string()
                .contains("logit_bias[0]")
        );
    }
}

#[cfg(test)]
mod ordered_sampler_tests {
    use super::*;
    use crate::logits::{LogitProcessor, TemperatureProcessor};
    fn options(stages: Vec<SamplerStage>) -> GenerateOptions {
        GenerateOptions {
            greedy: false,
            temperature: 2.0,
            top_p: 0.6,
            sampler_order: Some(SamplerSequence::new(stages).unwrap()),
            ..Default::default()
        }
    }
    #[test]
    fn sampler_order_changes_the_actual_processed_distribution() {
        let mut before = vec![2.0, 1.0, 0.0];
        let mut after = before.clone();
        build_processor_chain(
            &options(vec![SamplerStage::Temperature, SamplerStage::TopP]),
            None,
            false,
        )
        .unwrap()
        .process(&mut before, &ProcessorContext::default())
        .unwrap();
        build_processor_chain(
            &options(vec![SamplerStage::TopP, SamplerStage::Temperature]),
            None,
            false,
        )
        .unwrap()
        .process(&mut after, &ProcessorContext::default())
        .unwrap();
        assert_eq!(before.iter().filter(|value| value.is_finite()).count(), 2);
        assert_eq!(after.iter().filter(|value| value.is_finite()).count(), 1);
    }
    #[test]
    fn sampler_order_omissions_and_repetitions_are_load_bearing() {
        let mut options = options(vec![]);
        options.temperature = 0.0;
        let chain = build_processor_chain(&options, None, false).unwrap();
        assert!(chain.is_empty());
        assert!(
            !options.selects_greedily(),
            "omitting temperature cannot turn a raw categorical distribution into greedy"
        );
        assert_eq!(
            device_sampling_plan(&chain, &options, false, true, true),
            DeviceSamplingPlan::Host
        );
        options.temperature = 2.0;
        options.sampler_order =
            Some(SamplerSequence::new(vec![SamplerStage::Temperature; 2]).unwrap());
        let chain = build_processor_chain(&options, None, false).unwrap();
        let mut logits = vec![4.0, 0.0];
        chain
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits, vec![1.0, 0.0]);
        assert_eq!(
            device_sampling_plan(&chain, &options, false, true, true),
            DeviceSamplingPlan::Host
        );
    }
    #[test]
    fn sampler_device_capability_checks_order_parameters_and_real_implementation() {
        let ordered = options(vec![SamplerStage::Temperature, SamplerStage::TopP]);
        let fixed = build_processor_chain(&ordered, None, false).unwrap();
        assert_eq!(
            device_sampling_plan(&fixed, &ordered, false, true, true),
            DeviceSamplingPlan::Sampled
        );
        let reverse = options(vec![SamplerStage::TopP, SamplerStage::Temperature]);
        let chain = build_processor_chain(&reverse, None, false).unwrap();
        assert_eq!(
            device_sampling_plan(&chain, &reverse, false, true, true),
            DeviceSamplingPlan::Host
        );
        let mut wrong_params = ProcessorChain::new();
        wrong_params.add(Box::new(TemperatureProcessor {
            temperature: 3.0,
            dynamic: None,
        }));
        assert_eq!(
            device_sampling_plan(&wrong_params, &ordered, false, true, true),
            DeviceSamplingPlan::Host
        );
        struct Impostor;
        impl LogitProcessor for Impostor {
            fn process(&self, logits: &mut [f32], _: &ProcessorContext) -> anyhow::Result<()> {
                logits.reverse();
                Ok(())
            }
            fn name(&self) -> &str {
                "temperature"
            }
        }
        let mut impostor = ProcessorChain::new();
        impostor.add(Box::new(Impostor));
        assert!(!is_device_portable_chain(&impostor));
    }
    #[test]
    fn sampler_fusion_respects_adjacency_and_temperature_rounding_order() {
        let options = GenerateOptions {
            greedy: false,
            temperature: 2.0,
            top_k: 2,
            sampler_order: Some(
                SamplerSequence::new(vec![SamplerStage::TopK, SamplerStage::Temperature]).unwrap(),
            ),
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false).unwrap();
        assert_eq!(chain.names(), vec!["top_k", "temperature"]);
        assert_eq!(
            device_sampling_plan(&chain, &options, false, true, true),
            DeviceSamplingPlan::Host
        );
        // Positive real scaling preserves ranks, but f32 division can merge
        // unequal logits. Moving temperature ahead of an exactly-k tie quota
        // would choose token 0 instead of the original maximum at token 1.
        let mut precision_options = options.clone();
        precision_options.temperature = f32::MAX;
        precision_options.top_k = 1;
        let mut ranked_first = vec![1e-30, 2e-30];
        build_processor_chain(&precision_options, None, false)
            .unwrap()
            .process(&mut ranked_first, &ProcessorContext::default())
            .unwrap();
        assert_eq!(ranked_first, vec![f32::NEG_INFINITY, 0.0]);
        precision_options.sampler_order = Some(
            SamplerSequence::new(vec![SamplerStage::Temperature, SamplerStage::TopK]).unwrap(),
        );
        let mut scaled_first = vec![1e-30, 2e-30];
        let chain = build_processor_chain(&precision_options, None, false).unwrap();
        chain
            .process(&mut scaled_first, &ProcessorContext::default())
            .unwrap();
        assert_eq!(scaled_first, vec![0.0, f32::NEG_INFINITY]);
        assert_eq!(
            device_sampling_plan(&chain, &precision_options, false, true, true),
            DeviceSamplingPlan::Sampled
        );
        let mut options = options;
        options.top_p = 0.6;
        options.sampler_order = Some(
            SamplerSequence::new(vec![
                SamplerStage::TopK,
                SamplerStage::TypicalP,
                SamplerStage::TopP,
                SamplerStage::Temperature,
            ])
            .unwrap(),
        );
        let chain = build_processor_chain(&options, None, false).unwrap();
        assert_eq!(chain.names(), vec!["top_k_top_p", "temperature"]);
        assert_eq!(
            device_sampling_plan(&chain, &options, false, true, true),
            DeviceSamplingPlan::Host
        );
    }
    #[test]
    fn sampler_dynamic_entropy_uses_compacted_candidates_but_preserves_mask_only_cardinality() {
        let mut options = GenerateOptions {
            greedy: false,
            temperature: 1.0,
            top_k: 2,
            dynamic_temperature: Some(crate::DynamicTemperatureConfig {
                range: 0.5,
                exponent: 1.0,
            }),
            sampler_order: Some(
                SamplerSequence::new(vec![SamplerStage::TopK, SamplerStage::Temperature]).unwrap(),
            ),
            ..Default::default()
        };
        let mut logits = vec![3.0, 3.0, -100.0, -100.0];
        build_processor_chain(&options, None, false)
            .unwrap()
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits, vec![2.0, 2.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        options.top_n_sigma = 1.0;
        options.sampler_order = Some(
            SamplerSequence::new(vec![SamplerStage::TopNSigma, SamplerStage::Temperature]).unwrap(),
        );
        let mut logits = vec![3.0, 3.0, -100.0, -100.0];
        build_processor_chain(&options, None, false)
            .unwrap()
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits, vec![3.0, 3.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
    }

    #[test]
    fn selected_token_feedback_disables_device_bypass_even_for_rank_preserving_stages() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        struct Feedback(Arc<AtomicUsize>);
        impl crate::logits::LogitProcessor for Feedback {
            fn process(&self, _: &mut [f32], _: &ProcessorContext) -> anyhow::Result<()> {
                Ok(())
            }
            fn name(&self) -> &str {
                "feedback"
            }
            fn preserves_argmax(&self) -> bool {
                true
            }
            fn device_sampling_transform(&self) -> Option<crate::logits::DeviceSamplingTransform> {
                Some(crate::logits::DeviceSamplingTransform::Temperature(1.0))
            }
            fn needs_selection_feedback(&self) -> bool {
                true
            }
            fn on_selected(&self, _: TokenId, _: &ProcessorContext) -> anyhow::Result<()> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }
        let accepted = Arc::new(AtomicUsize::new(0));
        let mut chain = ProcessorChain::new();
        chain.add(Box::new(Feedback(accepted.clone())));
        let options = GenerateOptions::default();
        assert_eq!(
            device_sampling_plan(&chain, &options, false, true, true),
            DeviceSamplingPlan::Host
        );
        let token = select_next_token(
            &mut [0.0, 1.0],
            &ProcessorContext::default(),
            &options,
            &chain,
            0.0,
        )
        .unwrap();
        assert_eq!(token, 1);
        assert_eq!(accepted.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn adaptive_p_is_one_terminal_stage_and_requires_explicit_selection() {
        let mut options = GenerateOptions {
            greedy: false,
            temperature: 2.0,
            adaptive_p: crate::AdaptivePConfig {
                target: 0.2,
                decay: 0.9,
            },
            sampler_order: Some(
                SamplerSequence::new(vec![
                    SamplerStage::AdaptiveP,
                    SamplerStage::Temperature,
                    SamplerStage::AdaptiveP,
                ])
                .unwrap(),
            ),
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false).unwrap();
        assert_eq!(chain.names(), ["temperature", "adaptive_p"]);
        assert_eq!(
            device_sampling_plan(&chain, &options, false, true, true),
            DeviceSamplingPlan::Host
        );
        options.adaptive_p.target = -1.0;
        assert_eq!(
            build_processor_chain(&options, None, false)
                .unwrap()
                .names(),
            ["temperature"]
        );
        options.adaptive_p.target = 0.2;
        options.sampler_order = None;
        assert!(
            !build_processor_chain(&options, None, false)
                .unwrap()
                .names()
                .contains(&"adaptive_p")
        );
        options.sampler_order = Some(SamplerSequence::new(vec![SamplerStage::AdaptiveP]).unwrap());
        options.mirostat = Some(crate::MirostatConfig {
            tau: 5.0,
            eta: 0.1,
            version: crate::MirostatVersion::V2,
        });
        assert_eq!(
            build_processor_chain(&options, None, false)
                .unwrap()
                .names(),
            ["temperature", "mirostat_v2"]
        );
        assert_eq!(
            serde_json::from_str::<SamplerSequence>(r#"["ADAPTIVE-P"]"#)
                .unwrap()
                .stages(),
            [SamplerStage::AdaptiveP]
        );
    }

    #[test]
    fn sampler_mirostat_replaces_the_ordered_warper_list() {
        let options = GenerateOptions {
            greedy: false,
            temperature: 0.7,
            top_k: 40,
            mirostat: Some(crate::MirostatConfig {
                tau: 5.0,
                eta: 0.1,
                version: crate::MirostatVersion::V2,
            }),
            sampler_order: Some(SamplerSequence::new(vec![SamplerStage::TopK]).unwrap()),
            ..Default::default()
        };
        assert_eq!(
            build_processor_chain(&options, None, false)
                .unwrap()
                .names(),
            vec!["temperature", "mirostat_v2"]
        );
    }

    #[test]
    fn sampler_candidate_cardinality_handles_disabled_k_and_xtc_removal() {
        use crate::logits::TopKTopPProcessor;
        let context = ProcessorContext::default();
        let initial = vec![0.6_f32.ln(), 0.3_f32.ln(), 0.1_f32.ln(), f32::NEG_INFINITY];
        let mut logits = initial.clone();
        assert_eq!(
            TopKTopPProcessor {
                top_k: 0,
                top_p: 0.7
            }
            .process_with_candidate_count(&mut logits, &context, 4)
            .unwrap(),
            2
        );
        let mut logits = initial;
        assert_eq!(
            XtcProcessor::new(1.0, 0.2, Some(17))
                .process_with_candidate_count(&mut logits, &context, 4)
                .unwrap(),
            3
        );
        assert_eq!(logits.iter().filter(|value| value.is_finite()).count(), 2);
        assert!(
            TemperatureProcessor {
                temperature: f32::INFINITY,
                dynamic: None
            }
            .device_sampling_transform()
            .is_none()
        );
    }

    #[test]
    fn sampler_zero_temperature_keeps_argmax_shortcut_with_rank_preserving_filters() {
        let options = GenerateOptions {
            greedy: false,
            temperature: 0.0,
            top_k: 3,
            top_p: 0.9,
            min_p: 0.1,
            top_a: 0.2,
            top_n_sigma: 1.0,
            sampler_order: Some(
                SamplerSequence::new(vec![
                    SamplerStage::TopNSigma,
                    SamplerStage::TopK,
                    SamplerStage::TopP,
                    SamplerStage::MinP,
                    SamplerStage::TopA,
                    SamplerStage::Temperature,
                ])
                .unwrap(),
            ),
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false).unwrap();
        assert_eq!(
            device_sampling_plan(&chain, &options, false, true, true),
            DeviceSamplingPlan::Greedy
        );
        for mut logits in [
            vec![3.0, 2.0, 1.0, 0.0],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![f32::NAN, -0.0, 0.0, f32::INFINITY, 3.0, 2.0],
            vec![3e38, 2e38, 0.0],
        ] {
            let expected = crate::sampling::sample_greedy(&logits);
            chain
                .process(&mut logits, &ProcessorContext::default())
                .unwrap();
            assert_eq!(crate::sampling::sample_greedy(&logits), expected);
        }
    }

    #[test]
    fn sampler_stage_aliases_bounds_and_top_n_sigma_are_real() {
        let order: SamplerSequence = serde_json::from_value(serde_json::json!([
            "TOP-K",
            "nucleus",
            "temp",
            "typ",
            "topnsigma"
        ]))
        .unwrap();
        assert_eq!(
            order.stages(),
            [
                SamplerStage::TopK,
                SamplerStage::TopP,
                SamplerStage::Temperature,
                SamplerStage::TypicalP,
                SamplerStage::TopNSigma
            ]
        );
        assert_eq!(
            serde_json::to_value(SamplerStage::TopNSigma).unwrap(),
            "top_n_sigma"
        );
        assert!(SamplerSequence::new(vec![SamplerStage::TopK; 65]).is_err());
        assert!(serde_json::from_value::<SamplerSequence>(serde_json::json!(["unknown"])).is_err());
        let mut logits = vec![3.0, 2.0, 1.0, f32::NEG_INFINITY];
        TopNSigmaProcessor { n: 1.0 }
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(
            logits,
            vec![3.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]
        );
    }
}

#[cfg(test)]
mod json_continuation_tests {
    use super::*;
    #[test]
    fn json_continuation_finish_and_stop_use_prefix_plus_suffix() {
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Json { prefix: "{".into() }),
            ..Default::default()
        };
        ensure_constrained_finish(&options, "}", FinishReason::MaxTokens).unwrap();
        assert!(ensure_constrained_finish(&options, "", FinishReason::Length).is_err());
        let mut chain = ProcessorChain::new();
        chain.add(Box::new(StopSequenceProcessor::new(vec![
            StopSequence::Text("}".into()),
        ])));
        let context = ProcessorContext {
            generated_text: "}".into(),
            ..Default::default()
        };
        assert_eq!(
            finish_reason_after_token(5, &options, &chain, &context),
            Some(FinishReason::StopSequence { index: 0 })
        );
    }
}
