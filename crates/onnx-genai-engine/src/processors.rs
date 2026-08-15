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
