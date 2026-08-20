//! Request logit processor construction and token selection helpers.

use crate::FimConfig;
use crate::config::{FinishReason, GenerateConstraint, GenerateOptions};
use crate::logits::{
    ConstraintProcessor, DryProcessor, FrequencyPenaltyProcessor, GrammarConstraintKind,
    JsonConstraint, LlguidanceConstraint, MinPProcessor, MirostatProcessor,
    PresencePenaltyProcessor, ProcessorChain, ProcessorContext, ProcessorSignal,
    RepetitionPenaltyProcessor, StopSequence, StopSequenceProcessor, TemperatureProcessor, TokenId,
    TopAProcessor, TopKProcessor, TopKTopPProcessor, TopPProcessor, TypicalPProcessor,
    XtcProcessor,
};
use crate::sampling::{Sampler, default_sampler_for_options};
use anyhow::Context;
use onnx_genai_ort::Tokenizer;
use std::path::Path;

pub(crate) fn build_processor_chain(
    options: &GenerateOptions,
    tokenizer: Option<&Tokenizer>,
) -> anyhow::Result<ProcessorChain> {
    let mut chain = ProcessorChain::new();

    if options.repetition_penalty != 1.0 {
        chain.add(Box::new(RepetitionPenaltyProcessor {
            penalty: options.repetition_penalty,
            window: options.repetition_window,
        }));
    }

    if options.frequency_penalty != 0.0 {
        chain.add(Box::new(FrequencyPenaltyProcessor {
            frequency_penalty: options.frequency_penalty,
        }));
    }

    if options.presence_penalty != 0.0 {
        chain.add(Box::new(PresencePenaltyProcessor {
            presence_penalty: options.presence_penalty,
        }));
    }

    if let Some(dry) = &options.dry
        && dry.multiplier > 0.0
    {
        chain.add(Box::new(DryProcessor {
            multiplier: dry.multiplier,
            base: dry.base,
            allowed_length: dry.allowed_length,
            sequence_breakers: dry.sequence_breakers.clone(),
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
            GenerateConstraint::Json => {
                chain.add(Box::new(ConstraintProcessor::new(
                    Box::new(JsonConstraint),
                    token_texts,
                    options.eos_token_id,
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
                    options.eos_token_id,
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
                    options.eos_token_id,
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
                    options.eos_token_id,
                )));
            }
        }
    }

    if options.temperature > 0.0 && options.temperature != 1.0 {
        chain.add(Box::new(TemperatureProcessor {
            temperature: options.temperature,
        }));
    }

    // Fused when both are configured, which is the common case. Running them
    // separately makes top-p rescan the whole vocabulary that top-k has
    // already reduced to `top_k` entries; the fused processor keeps the
    // survivors and runs the nucleus search on those alone.
    match (options.top_k > 0, options.top_p < 1.0) {
        (true, true) => {
            chain.add(Box::new(TopKTopPProcessor {
                top_k: options.top_k,
                top_p: options.top_p,
            }));
        }
        (true, false) => {
            chain.add(Box::new(TopKProcessor {
                top_k: options.top_k,
            }));
        }
        (false, true) => {
            chain.add(Box::new(TopPProcessor {
                top_p: options.top_p,
            }));
        }
        (false, false) => {}
    }

    if options.min_p > 0.0 {
        chain.add(Box::new(MinPProcessor {
            min_p: options.min_p,
        }));
    }

    if options.top_a > 0.0 {
        chain.add(Box::new(TopAProcessor {
            top_a: options.top_a,
        }));
    }

    if options.typical_p < 1.0 {
        chain.add(Box::new(TypicalPProcessor {
            typical_p: options.typical_p,
        }));
    }

    if let Some(mirostat) = options.mirostat {
        chain.add(Box::new(MirostatProcessor::new(
            mirostat.tau,
            mirostat.eta,
            mirostat.version,
        )));
    }

    if let Some(xtc) = options.xtc
        && xtc.probability > 0.0
    {
        chain.add(Box::new(XtcProcessor::new(
            xtc.probability,
            xtc.threshold,
            options.seed,
        )));
    }

    Ok(chain)
}

/// Whether every processor in `chain` is implemented by the device sampler.
///
/// `top_k_top_p` is the fused form of `top_k` followed by `top_p`. It performs
/// exactly the same filtering, so a chain containing it is portable wherever
/// one containing the two separately would have been. Omitting it here would
/// silently disable the device sampling fast path for the most common
/// configuration -- the chain would still produce correct tokens, just on the
/// host, with no error to indicate why.
pub(crate) fn is_device_portable_chain(chain: &ProcessorChain) -> bool {
    chain.names().into_iter().all(|name| {
        matches!(
            name,
            "temperature" | "top_k" | "top_p" | "top_k_top_p" | "min_p"
        )
    })
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
        && (options.greedy || options.temperature == 0.0)
        && !has_custom_sampler
        && greedy_fastpath_supported;
    if greedy {
        return DeviceSamplingPlan::Greedy;
    }
    // Keep greedy behavior on its existing argmax path. The sampled path is only
    // for categorical decoding whose processor chain the device sampler supports.
    let sampled = !options.greedy
        && options.temperature != 0.0
        && options.top_logprobs.is_none()
        && !has_custom_sampler
        && is_device_portable_chain(chain)
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

pub(crate) fn ensure_constrained_finish(
    options: &GenerateOptions,
    generated_text: &str,
    finish_reason: FinishReason,
) -> anyhow::Result<()> {
    if matches!(
        (&options.constraint, finish_reason),
        (
            Some(GenerateConstraint::Json),
            FinishReason::MaxTokens | FinishReason::Length
        )
    ) && !JsonConstraint::is_complete(generated_text)
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
) -> TokenId {
    let mut sampler = default_sampler_for_options(options, rng_value);
    select_next_token_with_sampler(logits, context, chain, &mut sampler)
}

pub(crate) fn select_next_token_with_rng(
    logits: &mut [f32],
    context: &ProcessorContext,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    rng: &mut crate::sampling::SamplingRng,
) -> TokenId {
    select_next_token(logits, context, options, chain, rng.value_for(options))
}

pub(crate) fn select_next_token_with_sampler(
    logits: &mut [f32],
    context: &ProcessorContext,
    chain: &ProcessorChain,
    sampler: &mut dyn Sampler,
) -> TokenId {
    chain.process(logits, context);
    sampler.sample(logits, context)
}

pub(crate) fn finish_reason_after_token(
    token_id: TokenId,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    context: &ProcessorContext,
) -> Option<FinishReason> {
    if options.stop_on_eos && options.eos_token_id == Some(token_id) {
        return Some(FinishReason::EosToken);
    }

    match chain.signal(context) {
        Some(ProcessorSignal::StopSequence { index })
            if !matches!(&options.constraint, Some(GenerateConstraint::Json))
                || JsonConstraint::is_complete(&context.generated_text) =>
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

    fn options_with(top_k: usize, top_p: f32) -> GenerateOptions {
        GenerateOptions {
            top_k,
            top_p,
            ..Default::default()
        }
    }

    /// The standard sampling configuration must stay eligible for the device
    /// sampling fast path.
    ///
    /// `is_device_portable_chain` matches on processor *names*, so renaming or
    /// fusing a processor silently drops the chain off the device path: tokens
    /// stay correct, they are just sampled on the host instead, with no error
    /// to say why. Fusing top_k and top_p broke exactly this, and only two
    /// decode-loop tests noticed. This pins it directly.
    #[test]
    fn a_top_k_top_p_chain_is_device_portable() {
        let chain = build_processor_chain(&options_with(40, 0.95), None)
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

    fn plan_for(options: &GenerateOptions) -> DeviceSamplingPlan {
        let chain = build_processor_chain(options, None).expect("options build a chain");
        device_sampling_plan(&chain, options, false, true, true)
    }

    /// A greedy request that also carries the model's own `generation` defaults
    /// must still reach the device argmax.
    ///
    /// This is the shape almost every chat model presents: `top_k` / `top_p`
    /// come from the model card and ride along on the request even when the
    /// caller asked for greedy decoding, plus a stop sequence from the chat
    /// template. The old `chain.is_empty()` test read that as "sampling is
    /// configured" and sent the request to the host, so the device argmax --
    /// implemented, tested, and measured -- was dead for real models while
    /// passing every test that used bare `GenerateOptions::default()`.
    #[test]
    fn a_greedy_request_carrying_model_sampler_defaults_still_plans_the_device_argmax() {
        let options = greedy_options(GenerateOptions {
            top_k: 64,
            top_p: 0.95,
            stop_sequences: vec![crate::logits::StopSequence::Text("<|end|>".into())],
            ..Default::default()
        });
        let chain = build_processor_chain(&options, None).expect("options build a chain");
        assert!(
            !chain.is_empty(),
            "this test is only meaningful with a non-empty chain"
        );
        assert_eq!(
            device_sampling_plan(&chain, &options, false, true, true),
            DeviceSamplingPlan::Greedy,
            "truncation and stop-sequence processors cannot move the argmax, got {:?}",
            chain.names()
        );
    }

    /// Negative control for the test above: a processor that *can* move the
    /// argmax must force the host path.
    ///
    /// Without this, `preserves_argmax` returning `true` unconditionally would
    /// pass every other test here while silently changing generated text.
    #[test]
    fn a_repetition_penalty_denies_the_device_argmax() {
        let options = greedy_options(GenerateOptions {
            top_k: 64,
            top_p: 0.95,
            repetition_penalty: 1.1,
            ..Default::default()
        });
        assert_eq!(
            plan_for(&options),
            DeviceSamplingPlan::Host,
            "a per-token logit rewrite changes which token is the maximum"
        );
    }

    /// Same control for the truncation processors that keep a *relative*
    /// threshold but can still discard the maximum.
    #[test]
    fn typical_p_and_xtc_deny_the_device_argmax() {
        for options in [
            greedy_options(GenerateOptions {
                typical_p: 0.5,
                ..Default::default()
            }),
            greedy_options(GenerateOptions {
                xtc: Some(crate::config::XtcConfig {
                    probability: 0.5,
                    threshold: 0.1,
                }),
                ..Default::default()
            }),
        ] {
            assert_eq!(
                plan_for(&options),
                DeviceSamplingPlan::Host,
                "this processor may mask the top token itself"
            );
        }
    }

    /// The claim `preserves_argmax` makes about top-k/top-p, checked against the
    /// processor rather than assumed from its name.
    #[test]
    fn top_k_top_p_leaves_the_argmax_where_it_was() {
        let chain = build_processor_chain(&options_with(4, 0.5), None)
            .expect("standard sampling options build a chain");
        let raw: Vec<f32> = vec![0.5, -3.0, 9.25, 1.0, 8.5, -1.0, 2.0, 0.0];
        let expected = crate::sampling::sample_greedy(&raw);
        let mut processed = raw.clone();
        chain.process(&mut processed, &ProcessorContext::default());
        assert_ne!(
            processed, raw,
            "this test is only meaningful if the chain actually rewrote logits"
        );
        assert_eq!(
            crate::sampling::sample_greedy(&processed),
            expected,
            "truncation must not move the maximum"
        );
    }

    /// Each half alone must remain portable too.
    #[test]
    fn top_k_alone_and_top_p_alone_stay_device_portable() {
        for (top_k, top_p, expected) in [(40usize, 1.0f32, "top_k"), (0, 0.95, "top_p")] {
            let chain =
                build_processor_chain(&options_with(top_k, top_p), None).expect("chain builds");
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
        build_processor_chain(&GenerateOptions::default(), None).expect("empty chain builds")
    }

    fn history_chain() -> ProcessorChain {
        let options = GenerateOptions {
            repetition_penalty: 1.2,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).expect("penalty chain builds");
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
