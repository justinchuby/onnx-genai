//! Request logit processor construction and token selection helpers.

use crate::FimConfig;
use crate::config::{FinishReason, GenerateConstraint, GenerateOptions};
use crate::logits::{
    ConstraintProcessor, FrequencyPenaltyProcessor, GrammarConstraintKind, JsonConstraint,
    LlguidanceConstraint, MinPProcessor, PresencePenaltyProcessor, ProcessorChain,
    ProcessorContext, ProcessorSignal, RepetitionPenaltyProcessor, StopSequence,
    StopSequenceProcessor, TemperatureProcessor, TokenId, TopKProcessor, TopPProcessor,
};
use crate::sampling::{sample_categorical, sample_greedy};
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

    if options.top_k > 0 {
        chain.add(Box::new(TopKProcessor {
            top_k: options.top_k,
        }));
    }

    if options.top_p < 1.0 {
        chain.add(Box::new(TopPProcessor {
            top_p: options.top_p,
        }));
    }

    if options.min_p > 0.0 {
        chain.add(Box::new(MinPProcessor {
            min_p: options.min_p,
        }));
    }

    Ok(chain)
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
    for id in 0..=max_id {
        token_texts[id] = tokenizer.decode(&[id as TokenId]).ok();
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
    chain.process(logits, context);
    if options.greedy || options.temperature == 0.0 {
        sample_greedy(logits)
    } else {
        sample_categorical(logits, rng_value)
    }
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
