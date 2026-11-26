//! Request logit processor construction and token selection helpers.

use crate::FimConfig;
use crate::config::{
    FinishReason, GenerateConstraint, GenerateOptions, SamplerSequence, SamplerStage,
};
use crate::logits::{
    AdaptivePProcessor, ConstraintProcessor, DryProcessor, GrammarConstraintKind, InfillContract,
    InfillProcessor, LlguidanceConstraint, MinPProcessor, MirostatProcessor, PenaltiesProcessor,
    ProcessorChain, ProcessorContext, ProcessorSignal, StopSequence, StopSequenceProcessor,
    TemperatureProcessor, TokenId, TopAProcessor, TopKProcessor, TopNSigmaProcessor, TopPProcessor,
    TypicalPProcessor, XtcProcessor,
};
use crate::sampling::{Sampler, default_sampler_for_options};
use anyhow::Context;
use onnx_genai_ort::Tokenizer;
use std::path::Path;

/// Preserve one output-stop priority for ordinary and authored generation.
pub(crate) fn append_output_observers(chain: &mut ProcessorChain, options: &GenerateOptions) {
    if let Some(limit) = options.max_prediction_time.filter(|limit| !limit.is_zero()) {
        chain.add(Box::new(crate::prediction_limit::PredictionTimeLimit::new(
            limit,
            options.stop_sequences.clone(),
        )));
    }
    if options.minimum_indent > 0 {
        chain.add(Box::new(crate::indentation::IndentationLimit::new(
            options.minimum_indent,
            options.stop_sequences.clone(),
        )));
    }
    if !options.stop_sequences.is_empty() {
        chain.add(Box::new(StopSequenceProcessor::new(
            options.stop_sequences.clone(),
        )));
    }
}

/// Build mandatory safety processors and the effective ordered sampler list.
/// An explicit greedy policy bypasses stochastic warpers; a temperature-zero
/// stage instead creates its point mass at the requested position. Hard bias,
/// grammar and stop contracts cannot be omitted through a custom stage list.
/// A caller-supplied terminal sampler still receives the configured distribution.
pub(crate) fn build_processor_chain(
    options: &GenerateOptions,
    tokenizer: Option<&Tokenizer>,
    has_custom_sampler: bool,
    decoder: Option<std::sync::Arc<crate::byte_stream::ByteStreamContract>>,
) -> anyhow::Result<ProcessorChain> {
    anyhow::ensure!(
        options.minimum_indent == 0 || tokenizer.is_some(),
        "minimum_indent requires the model tokenizer; declare a tokenizer before requesting indentation stopping"
    );
    let mut chain = ProcessorChain::new();
    chain.bind_publication_decoder(decoder.clone());
    // A temperature-zero stage still runs at its requested position. Only an
    // explicit greedy policy bypasses stochastic warpers before terminal selection.
    let samples = !options.greedy || has_custom_sampler;
    if !options.effective_preserved_tokens().is_empty() || options.response_projection.is_some() {
        options.validate_preserved_tokens(tokenizer.context("preserved_tokens requires the model tokenizer; provide a tokenizer before requesting special-token output")?)?;
    }
    if !options.logit_bias.is_empty() {
        if let Some(tokenizer) = tokenizer {
            options.validate_logit_bias(tokenizer)?;
        }
        chain.add(Box::new(crate::logits::LogitBiasProcessor {
            biases: options.logit_bias.clone(),
        }));
    }
    append_output_observers(&mut chain, options);

    if let Some(constraint) = &options.constraint {
        let tokenizer = tokenizer.context("constrained decoding requires a tokenizer")?;
        let token_texts = match constraint {
            // Raw constraints resolve their own exact byte vocabulary and consume
            // candidate IDs. A token decoded in isolation has no stream context
            // and may even panic in an otherwise valid composed Strip decoder.
            GenerateConstraint::Gbnf { .. }
            | GenerateConstraint::RawLark { .. }
            | GenerateConstraint::LazyJsonSchema { .. } => Vec::new(),
            _ => tokenizer_token_texts(tokenizer)?,
        };
        let projection = &options.response_projection;
        let plain_prefix = match constraint {
            GenerateConstraint::Json { prefix }
            | GenerateConstraint::JsonSchema { prefix, .. }
            | GenerateConstraint::Regex { prefix, .. }
            | GenerateConstraint::Lark { prefix, .. }
            | GenerateConstraint::XmlToolCall { prefix, .. } => prefix.as_str(),
            GenerateConstraint::Gbnf { .. }
            | GenerateConstraint::RawLark { .. }
            | GenerateConstraint::LazyJsonSchema { .. } => "",
        };
        anyhow::ensure!(
            projection.is_none() || plain_prefix.is_empty(),
            "constraint prefix conflicts with response_projection ownership; leave the constraint prefix empty and put the complete assistant prefill in response_projection"
        );
        let max_bytes = projection
            .as_ref()
            .map_or(options.max_constraint_bytes, |projection| {
                projection.max_bytes().min(options.max_constraint_bytes)
            });
        let prefix = |plain: &str| {
            projection.as_ref().map_or_else(
                || plain.to_owned(),
                |projection| projection.prefix().to_owned(),
            )
        };
        let kind = || {
            projection.as_ref().map_or(
                crate::text_projection::ProjectionKind::Plain,
                |projection| {
                    crate::text_projection::ProjectionKind::Declared(projection.template().clone())
                },
            )
        };
        let endings = || {
            if options.stop_on_eos {
                terminal_token_ids(options)
            } else {
                Vec::new()
            }
        };
        let grammar: Box<dyn crate::logits::Constraint> = match constraint {
            GenerateConstraint::Gbnf { grammar, .. } | GenerateConstraint::RawLark { grammar } => {
                let contract = crate::byte_stream::ByteStreamContract::resolve(
                    tokenizer, crate::tokenization::SpecialTokenHandling::Preserve,
                ).context("resolve tokenizer.decoder byte contract for raw grammar")?.context("raw grammar requires a resolved lossless tokenizer.decoder byte contract; use a tokenizer with supported byte decoding instead of contextual decoded-token guesses")?;
                // GBNF inspects raw pieces, including declared special tokens.
                // Only explicitly resolved ending roles receive an EOS frame.
                // The shared byte contract proves density and exact decoder
                // transforms without adapter guesses based on token spelling.
                let endings = terminal_token_ids(options);
                if let GenerateConstraint::Gbnf {
                    activation: Some(activation),
                    ..
                } = constraint
                {
                    Box::new(crate::gbnf::lazy::LazyGrammarConstraint::new(
                        grammar,
                        tokenizer,
                        options,
                        contract.pieces,
                        endings,
                        activation,
                    )?)
                } else {
                    let lark = match constraint {
                        GenerateConstraint::RawLark { .. } => {
                            anyhow::ensure!(
                                options.max_constraint_bytes > 0
                                    && grammar.len() <= options.max_constraint_bytes,
                                "raw Lark grammar is {} bytes with max_constraint_bytes={}; shorten grammar or configure a positive sufficient budget",
                                grammar.len(),
                                options.max_constraint_bytes
                            );
                            grammar.clone()
                        }
                        _ => {
                            crate::gbnf::compile(grammar, options.max_constraint_bytes, tokenizer)?
                        }
                    };
                    let constraint = onnx_genai_llguidance::panic_utils::catch_unwind(
                        std::panic::AssertUnwindSafe(|| {
                            LlguidanceConstraint::from_token_bytes(
                                GrammarConstraintKind::Lark, &lark, contract.pieces, &endings, "",
                            )
                        }),
                    ).context("compile raw grammar against the resolved model byte vocabulary; repair the grammar or tokenizer.decoder")?;
                    Box::new(constraint.with_completion_only_eos(endings))
                }
            }

            GenerateConstraint::LazyJsonSchema { schema, activation } => {
                let contract = crate::byte_stream::ByteStreamContract::resolve(
                    tokenizer, crate::tokenization::SpecialTokenHandling::Preserve,
                ).context("resolve tokenizer.decoder byte contract for lazy JSON")?
                    .context("lazy JSON requires a resolved lossless tokenizer.decoder byte contract; use a supported raw byte decoder")?;
                Box::new(crate::gbnf::lazy::LazyGrammarConstraint::json_schema(
                    schema,
                    tokenizer,
                    options,
                    contract,
                    terminal_token_ids(options),
                    activation,
                )?)
            }

            GenerateConstraint::Json { prefix: plain } => {
                Box::new(crate::response_constraint::ContentConstraint::json_schema(
                    tokenizer,
                    &serde_json::Value::Bool(true),
                    kind(),
                    &prefix(plain),
                    max_bytes,
                    crate::response_constraint::ConstraintTokens::decoded(
                        endings(),
                        &options.effective_preserved_tokens(),
                        decoder.clone(),
                    ),
                )?)
            }
            GenerateConstraint::JsonSchema {
                schema,
                prefix: plain,
            } => {
                let schema = serde_json::from_str(schema)
                    .context("parse JSON Schema constraint; supply a valid JSON schema value")?;
                Box::new(crate::response_constraint::ContentConstraint::json_schema(
                    tokenizer,
                    &schema,
                    kind(),
                    &prefix(plain),
                    max_bytes,
                    crate::response_constraint::ConstraintTokens::decoded(
                        endings(),
                        &options.effective_preserved_tokens(),
                        decoder.clone(),
                    ),
                )?)
            }
            GenerateConstraint::Regex {
                pattern,
                prefix: plain,
            } => {
                if projection.is_some()
                    || decoder.is_some()
                    || crate::byte_stream::ByteStreamContract::requests_literal_stream(tokenizer)?
                {
                    Box::new(crate::response_constraint::ContentConstraint::text(
                        tokenizer,
                        &crate::config::TextGrammar::Regex(pattern.clone()),
                        kind(),
                        &prefix(plain),
                        max_bytes,
                        crate::response_constraint::ConstraintTokens::decoded(
                            endings(),
                            &options.effective_preserved_tokens(),
                            decoder.clone(),
                        ),
                    )?)
                } else {
                    build_llguidance_constraint(
                        GrammarConstraintKind::Regex,
                        pattern,
                        plain,
                        tokenizer,
                        &token_texts,
                        options.eos_token_id,
                        &terminal_token_ids(options),
                    )?
                }
            }
            GenerateConstraint::XmlToolCall {
                grammar,
                prefix: plain,
            } => {
                let (kind, prefix) = if let Some(projection) = projection {
                    (
                        crate::text_projection::ProjectionKind::Declared(
                            projection.template().clone(),
                        ),
                        projection.prefix(),
                    )
                } else {
                    (
                        crate::text_projection::ProjectionKind::Plain,
                        plain.as_str(),
                    )
                };
                Box::new(
                    crate::response_constraint::ContentConstraint::xml_tool_call(
                        tokenizer,
                        grammar,
                        kind,
                        prefix,
                        max_bytes,
                        crate::response_constraint::ConstraintTokens::decoded(
                            endings(),
                            &options.effective_preserved_tokens(),
                            decoder.clone(),
                        ),
                    )?,
                )
            }
            GenerateConstraint::Lark {
                grammar,
                prefix: plain,
            } => {
                if projection.is_some()
                    || decoder.is_some()
                    || crate::byte_stream::ByteStreamContract::requests_literal_stream(tokenizer)?
                {
                    Box::new(crate::response_constraint::ContentConstraint::text(
                        tokenizer,
                        &crate::config::TextGrammar::Lark(grammar.clone()),
                        kind(),
                        &prefix(plain),
                        max_bytes,
                        crate::response_constraint::ConstraintTokens::decoded(
                            endings(),
                            &options.effective_preserved_tokens(),
                            decoder.clone(),
                        ),
                    )?)
                } else {
                    build_llguidance_constraint(
                        GrammarConstraintKind::Lark,
                        grammar,
                        plain,
                        tokenizer,
                        &token_texts,
                        options.eos_token_id,
                        &terminal_token_ids(options),
                    )?
                }
            }
        };
        chain.add(Box::new(ConstraintProcessor::new(
            grammar,
            token_texts,
            terminal_token_ids(options),
        )));
    }

    if let Some(budget) = &options.reasoning_budget {
        for id in budget.forced_tokens() {
            anyhow::ensure!(
                !matches!(
                    options.logit_bias.get(id),
                    Some(crate::logits::LogitBias::Ban)
                ),
                "reasoning closer token {id} is banned by logit_bias; allow all declared forced closing tokens before enabling the budget or live control"
            );
        }
        chain.add(Box::new(budget.processor()?));
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
    metadata: &onnx_genai_metadata::InferenceMetadata,
    tokenizer: &Tokenizer,
) -> anyhow::Result<Option<FimConfig>> {
    if let Some(declaration) = metadata
        .package
        .as_ref()
        .and_then(|package| package.infill.as_ref())
    {
        for (role, spelling) in [
            ("prefix_token", &declaration.prefix_token),
            ("middle_token", &declaration.middle_token),
            ("suffix_token", &declaration.suffix_token),
        ]
        .into_iter()
        .chain(
            declaration
                .additional_stop_tokens
                .iter()
                .map(|spelling| ("additional_stop_tokens", spelling)),
        ) {
            let id = tokenizer.token_id(spelling).with_context(||format!("package.infill.{role} {spelling:?} is absent from the package tokenizer; declare an exact registered control token"))?;
            anyhow::ensure!(
                tokenizer.encode_with_options(
                    spelling,
                    onnx_genai_ort::TokenizationOptions {
                        add_special: false,
                        parse_special: true
                    }
                ).with_context(|| format!("encode package.infill.{role} {spelling:?}; repair the declared tokenizer control token"))? == [id],
                "package.infill.{role} must encode to its one declared tokenizer ID; register the control token explicitly"
            );
        }
        return Ok(Some(FimConfig {
            prefix_token: declaration.prefix_token.clone(),
            middle_token: declaration.middle_token.clone(),
            suffix_token: declaration.suffix_token.clone(),
            format: declaration.order,
            additional_stop_tokens: declaration.additional_stop_tokens.clone(),
            extra_context_template: declaration.extra_context_template.clone(),
        }));
    }
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
    prefix: &str,
    tokenizer: &Tokenizer,
    token_texts: &[Option<String>],
    eos_token_id: Option<TokenId>,
    endings: &[TokenId],
) -> anyhow::Result<Box<dyn crate::logits::Constraint>> {
    if !prefix.is_empty() {
        let ids = (0..token_texts.len())
            .map(TokenId::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let mut bytes=crate::tokenization::token_pieces(tokenizer,&ids)
            .context("prefilled grammar requires lossless tokenizer byte pieces; repair the package tokenizer decoder")?;
        for (id, token) in tokenizer.inner().get_added_tokens_decoder() {
            if token.special {
                let piece = bytes
                    .get_mut(usize::try_from(id)?)
                    .context("special grammar token is outside the declared byte vocabulary")?;
                piece.insert(
                    0,
                    onnx_genai_llguidance::toktrie::TokTrie::SPECIAL_TOKEN_MARKER,
                );
            }
        }
        return LlguidanceConstraint::from_token_bytes_with_prefix(
            kind, grammar, bytes, endings, prefix,
        )
        .map(|constraint| Box::new(constraint) as Box<dyn crate::logits::Constraint>);
    }
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

fn tokenizer_token_texts(tokenizer: &Tokenizer) -> anyhow::Result<Vec<Option<String>>> {
    let vocab = tokenizer.inner().get_vocab(true);
    let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
    let mut token_texts = vec![None; max_id + 1];
    for &id in vocab.values() {
        token_texts[id as usize] = Some(
            onnx_genai_llguidance::panic_utils::catch_unwind(std::panic::AssertUnwindSafe(|| tokenizer.decode(&[id]).map_err(anyhow::Error::from)))
                .with_context(|| format!("decode vocabulary token {id} for sampling; repair tokenizer.decoder or this vocabulary entry"))?
        );
    }
    Ok(token_texts)
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
    let signal = chain.signal(context);
    if options.terminates(token_id) {
        return Some(FinishReason::EosToken);
    }

    signal.map(|signal| match signal {
        ProcessorSignal::StopSequence { index } => FinishReason::StopSequence { index },
        ProcessorSignal::PredictionTimeLimit => FinishReason::PredictionTimeLimit,
        ProcessorSignal::IndentationLimit { .. } => FinishReason::IndentationLimit,
    })
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
        let chain = build_processor_chain(&options, None, false, None).unwrap();
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
        let chain = build_processor_chain(&options_with(40, 0.95), None, false, None)
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
        let chain = build_processor_chain(&greedy_options(every_warper()), None, false, None)
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
            let chain = build_processor_chain(&options, None, false, None)
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
        let chain =
            build_processor_chain(&options, None, false, None).expect("options build a chain");
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
        let chain = build_processor_chain(&options, None, false, None).unwrap();
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
            build_processor_chain(&options, None, true, None).expect("custom-sampler chain builds");
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
        let chain = build_processor_chain(&options, None, false, None).unwrap();
        assert!(chain.names().contains(&"temperature"));
        assert!(options.selects_greedily());
        let mut logits = vec![3.0, 1.0, 2.0];
        chain
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits, vec![3.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        let raw = build_processor_chain(&greedy_options(options), None, false, None).unwrap();
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
        let chain =
            build_processor_chain(&options, None, false, None).expect("options build a chain");
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
                build_processor_chain(&options, None, false, None).expect("options build a chain");
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
        let sampling_chain = build_processor_chain(&options_with(4, 0.5), None, false, None)
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
            let chain = build_processor_chain(&options_with(top_k, top_p), None, false, None)
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
        build_processor_chain(&GenerateOptions::default(), None, false, None)
            .expect("empty chain builds")
    }

    fn history_chain() -> ProcessorChain {
        let options = GenerateOptions {
            repetition_penalty: 1.2,
            ..Default::default()
        };
        let chain =
            build_processor_chain(&options, None, false, None).expect("penalty chain builds");
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
        let chain = build_processor_chain(&options, None, false, None).unwrap();
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
            None,
        )
        .unwrap()
        .process(&mut before, &ProcessorContext::default())
        .unwrap();
        build_processor_chain(
            &options(vec![SamplerStage::TopP, SamplerStage::Temperature]),
            None,
            false,
            None,
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
        let chain = build_processor_chain(&options, None, false, None).unwrap();
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
        let chain = build_processor_chain(&options, None, false, None).unwrap();
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
        let fixed = build_processor_chain(&ordered, None, false, None).unwrap();
        assert_eq!(
            device_sampling_plan(&fixed, &ordered, false, true, true),
            DeviceSamplingPlan::Sampled
        );
        let reverse = options(vec![SamplerStage::TopP, SamplerStage::Temperature]);
        let chain = build_processor_chain(&reverse, None, false, None).unwrap();
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
        let chain = build_processor_chain(&options, None, false, None).unwrap();
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
        build_processor_chain(&precision_options, None, false, None)
            .unwrap()
            .process(&mut ranked_first, &ProcessorContext::default())
            .unwrap();
        assert_eq!(ranked_first, vec![f32::NEG_INFINITY, 0.0]);
        precision_options.sampler_order = Some(
            SamplerSequence::new(vec![SamplerStage::Temperature, SamplerStage::TopK]).unwrap(),
        );
        let mut scaled_first = vec![1e-30, 2e-30];
        let chain = build_processor_chain(&precision_options, None, false, None).unwrap();
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
        let chain = build_processor_chain(&options, None, false, None).unwrap();
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
        build_processor_chain(&options, None, false, None)
            .unwrap()
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(logits, vec![2.0, 2.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        options.top_n_sigma = 1.0;
        options.sampler_order = Some(
            SamplerSequence::new(vec![SamplerStage::TopNSigma, SamplerStage::Temperature]).unwrap(),
        );
        let mut logits = vec![3.0, 3.0, -100.0, -100.0];
        build_processor_chain(&options, None, false, None)
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
        let chain = build_processor_chain(&options, None, false, None).unwrap();
        assert_eq!(chain.names(), ["temperature", "adaptive_p"]);
        assert_eq!(
            device_sampling_plan(&chain, &options, false, true, true),
            DeviceSamplingPlan::Host
        );
        options.adaptive_p.target = -1.0;
        assert_eq!(
            build_processor_chain(&options, None, false, None,)
                .unwrap()
                .names(),
            ["temperature"]
        );
        options.adaptive_p.target = 0.2;
        options.sampler_order = None;
        assert!(
            !build_processor_chain(&options, None, false, None,)
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
            build_processor_chain(&options, None, false, None,)
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
            build_processor_chain(&options, None, false, None,)
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
        let chain = build_processor_chain(&options, None, false, None).unwrap();
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
    fn json_continuation_obeys_stop_before_the_value_is_complete() {
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Json { prefix: "{".into() }),
            ..Default::default()
        };
        let mut chain = ProcessorChain::new();
        chain.add(Box::new(StopSequenceProcessor::new(vec![
            StopSequence::Text(":".into()),
        ])));
        let context = ProcessorContext {
            generated_text: "\"key\":".into(),
            ..Default::default()
        };
        assert_eq!(
            finish_reason_after_token(5, &options, &chain, &context),
            Some(FinishReason::StopSequence { index: 0 })
        );
    }
}

#[cfg(test)]
mod lark_prefill_tests {
    use super::*;

    #[test]
    fn raw_grammars_match_actual_unicode_bytes_and_tokenizations() {
        let mut value: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        let vocab = value["model"]["vocab"].as_object_mut().unwrap();
        for (id, text) in [
            (22, "é"),
            (23, "😀"),
            (24, "é😀"),
            (25, r"\u00E9"),
            (26, "<0xC3>"),
            (27, "<0xA9>"),
        ] {
            let old = vocab
                .iter()
                .find(|(_, value)| value.as_u64() == Some(id))
                .unwrap()
                .0
                .clone();
            vocab.remove(&old);
            vocab.insert(text.into(), serde_json::json!(id));
        }
        value["decoder"] = serde_json::json!({"type":"ByteFallback"});
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let tokenizer = Tokenizer::from_file(path).unwrap();
        for (grammar, history, expected) in [
            (
                "%llguidance {\"no_forcing\":true}\nstart: \"é😀\"",
                vec![],
                vec![22, 24, 26],
            ),
            (
                "%llguidance {\"no_forcing\":true}\nstart: \"é😀\"",
                vec![26],
                vec![27],
            ),
            (
                "%llguidance {\"no_forcing\":true}\nstart: <[22]> <[23]>",
                vec![22],
                vec![23],
            ),
            ("root ::=", vec![], vec![3]),
            (r"root ::= [é😀]{1,2}", vec![26], vec![27]),
            (r"root ::= [é😀]{1,2}", vec![26, 27], vec![3, 22, 23, 26]),
            ("root ::= \"é\" |", vec![], vec![3, 22, 26]),
            ("root ::= \"é\"{0}", vec![], vec![3]),
            (r"root ::= [é😀]{1,2}", vec![], vec![22, 23, 24, 26]),
            (r"root ::= [é😀]{1,2}", vec![22], vec![3, 22, 23, 26]),
            (r"root ::= [é😀]{1,2}", vec![24], vec![3]),
            ("root ::= \"é\" root | \"\"", vec![], vec![3, 22, 26]),
            ("root ::= \"é\" root | \"\"", vec![22], vec![3, 22, 26]),
            ("root ::= .", vec![], vec![14, 15, 22, 23, 26]),
            (r#"root ::= "\u00E9\U0001F600""#, vec![], vec![22, 24, 26]),
            (r#"root ::= "\u00E9\U0001F600""#, vec![22], vec![23]),
            (r#"root ::= "\\u00E9""#, vec![], vec![25]),
        ] {
            let options = GenerateOptions {
                constraint: Some(if grammar.starts_with("%llguidance") {
                    GenerateConstraint::RawLark {
                        grammar: grammar.into(),
                    }
                } else {
                    GenerateConstraint::Gbnf {
                        activation: None,
                        grammar: grammar.into(),
                    }
                }),
                eos_token_id: Some(3),
                greedy: true,
                ..Default::default()
            };
            let chain = build_processor_chain(&options, Some(&tokenizer), false, None).unwrap();
            let mut logits = vec![0.0; 32];
            chain
                .process(
                    &mut logits,
                    &ProcessorContext {
                        generated_tokens: history.clone(),
                        ..Default::default()
                    },
                )
                .unwrap();
            let allowed = logits
                .iter()
                .enumerate()
                .filter_map(|(id, score)| score.is_finite().then_some(id))
                .collect::<Vec<_>>();
            assert_eq!(allowed, expected, "{grammar} after {history:?}");
        }
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::RawLark {
                grammar: "%llguidance {}\nstart: \"é\"".into(),
            }),
            max_constraint_bytes: 2,
            ..Default::default()
        };
        let error = match build_processor_chain(&options, Some(&tokenizer), false, None) {
            Ok(_) => panic!("oversized raw Lark must be rejected"),
            Err(error) => error,
        };
        assert!(format!("{error:#}").contains("max_constraint_bytes=2"));
    }

    #[test]
    fn raw_grammars_constrain_pieces_before_publication_stripping() {
        let original: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        let scratch = tempfile::tempdir().unwrap();
        for (start, stop) in [(1, 0), (0, 1), (1, 1)] {
            let mut value = original.clone();
            let vocab = value["model"]["vocab"].as_object_mut().unwrap();
            vocab.remove("tok22");
            vocab.insert("ĠxĠ".into(), serde_json::json!(22));
            value["decoder"] = serde_json::json!({"type":"Sequence","decoders":[
                {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
                {"type":"Strip","content":" ","start":start,"stop":stop}
            ]});
            let path = scratch.path().join(format!("{start}-{stop}.json"));
            std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            let tokenizer = Tokenizer::from_file(path).unwrap();
            let expected_text = match (start, stop) {
                (1, 0) => "x ",
                (0, 1) => " x",
                _ => "x",
            };
            assert_eq!(
                tokenizer.inner().decode(&[22], false).unwrap(),
                expected_text
            );
            for constraint in [
                GenerateConstraint::Gbnf {
                    grammar: "root ::= \" x \"".into(),
                    activation: None,
                },
                GenerateConstraint::Gbnf {
                    grammar: "root ::= <[22]>".into(),
                    activation: None,
                },
                GenerateConstraint::RawLark {
                    grammar: "start: \" x \"".into(),
                },
                GenerateConstraint::Gbnf {
                    grammar: "root ::= \" x \"".into(),
                    activation: Some(
                        crate::GrammarActivation::new(
                            vec![crate::GrammarTrigger::Token(22)],
                            crate::GrammarCollation::Libcxx,
                        )
                        .unwrap(),
                    ),
                },
            ] {
                let options = GenerateOptions {
                    constraint: Some(constraint),
                    eos_token_id: Some(3),
                    ..Default::default()
                };
                let chain = build_processor_chain(&options, Some(&tokenizer), false, None).unwrap();
                let mut logits = [0.0; 32];
                chain
                    .process(&mut logits, &ProcessorContext::default())
                    .unwrap();
                assert!(
                    logits[22].is_finite(),
                    "raw piece must retain stripped spaces"
                );
                let history = ProcessorContext {
                    generated_tokens: vec![22],
                    ..Default::default()
                };
                let mut completed = [0.0; 32];
                chain.process(&mut completed, &history).unwrap();
                assert!(completed[3].is_finite());
                assert!(!completed[22].is_finite());
                let mut reset = [0.0; 32];
                chain
                    .process(&mut reset, &ProcessorContext::default())
                    .unwrap();
                assert!(reset[22].is_finite());
            }
        }
    }

    #[test]
    fn gbnf_masks_actual_model_tokens_negation_and_history_rewinds() {
        let mut value: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        value["model"]["vocab"]
            .as_object_mut()
            .unwrap()
            .remove("tok31");
        value["model"]["vocab"]["<think>"] = serde_json::json!(31);
        value["model"]["vocab"]
            .as_object_mut()
            .unwrap()
            .remove("tok29");
        value["model"]["vocab"]["<ordinary>"] = serde_json::json!(29);
        value["added_tokens"] = serde_json::json!([{
            "id":31,"content":"<think>","single_word":false,"lstrip":false,
            "rstrip":false,"normalized":false,"special":true
        }, {"id":29,"content":"<ordinary>","single_word":false,"lstrip":false,
            "rstrip":false,"normalized":false,"special":false}]);
        value["decoder"] = serde_json::json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false});
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let tokenizer = Tokenizer::from_file(path).unwrap();
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Gbnf {
                activation: None,
                grammar: "root ::= <think> !<[31]> <[31]>".into(),
            }),
            eos_token_id: Some(3),
            eos_token_ids: vec![30],
            greedy: true,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, Some(&tokenizer), false, None).unwrap();
        for (history, expected) in [
            (vec![], Some(31)),
            (vec![31], None),
            (vec![31, 4], Some(31)),
            (vec![31, 4, 31], Some(3)),
            (vec![], Some(31)),
            (vec![31, 5], Some(31)),
        ] {
            let mut logits = vec![0.0; 32];
            chain
                .process(
                    &mut logits,
                    &ProcessorContext {
                        generated_tokens: history,
                        ..Default::default()
                    },
                )
                .unwrap();
            let allowed = logits
                .iter()
                .enumerate()
                .filter_map(|(id, score)| score.is_finite().then_some(id))
                .collect::<Vec<_>>();
            if let Some(expected) = expected {
                assert_eq!(
                    allowed,
                    if expected == 3 {
                        vec![3, 30]
                    } else {
                        vec![expected]
                    }
                );
            } else {
                assert!(!allowed.contains(&31));
                assert!(allowed.contains(&4));
                assert!(!allowed.contains(&3));
                assert!(!allowed.contains(&30));
            }
        }
        for (literal, expected) in [("<think>", 31), ("<ordinary>", 29)] {
            let mut literal_options = options.clone();
            literal_options.constraint = Some(GenerateConstraint::Gbnf {
                activation: None,
                grammar: format!("root ::= {literal:?}"),
            });
            let literal_chain =
                build_processor_chain(&literal_options, Some(&tokenizer), false, None).unwrap();
            let mut logits = [0.0; 32];
            literal_chain
                .process(&mut logits, &ProcessorContext::default())
                .unwrap();
            let allowed = logits
                .iter()
                .enumerate()
                .filter_map(|(id, score)| score.is_finite().then_some(id))
                .collect::<Vec<_>>();
            assert_eq!(
                allowed,
                [expected],
                "literal {literal} must follow actual token bytes and special flags"
            );
        }
        {
            let (case, expected) = ("sparse", "dense");
            let mut invalid_vocab = value.clone();
            invalid_vocab["model"]["vocab"]
                .as_object_mut()
                .unwrap()
                .remove("tok28");
            let invalid_path = scratch.path().join(format!("{case}.json"));
            std::fs::write(&invalid_path, serde_json::to_vec(&invalid_vocab).unwrap()).unwrap();
            let invalid_tokenizer = Tokenizer::from_file(invalid_path).unwrap();
            let error = build_processor_chain(&options, Some(&invalid_tokenizer), false, None)
                .err()
                .expect("unsupported byte contract must be explicit");
            assert!(format!("{error:#}").contains(expected), "{case}: {error:#}");
        }
        let invalid_history = ProcessorContext {
            generated_tokens: vec![31, 3],
            ..Default::default()
        };
        assert!(chain.process(&mut [0.0; 32], &invalid_history).is_err());
        let mut reset_logits = vec![0.0; 32];
        chain
            .process(&mut reset_logits, &ProcessorContext::default())
            .unwrap();
        assert!(reset_logits[31].is_finite() && !reset_logits[3].is_finite());
        for grammar in ["root ::= <[32]>", "root ::= <not-in-vocabulary>"] {
            let mut invalid = options.clone();
            invalid.constraint = Some(GenerateConstraint::Gbnf {
                activation: None,
                grammar: grammar.into(),
            });
            let error = build_processor_chain(&invalid, Some(&tokenizer), false, None)
                .err()
                .expect("invalid token reference");
            assert!(format!("{error:#}").contains("grammar token"), "{error:#}");
        }
    }

    #[test]
    fn lark_prefill_keeps_explicit_special_token_transitions() {
        let mut value: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        value["added_tokens"] = serde_json::json!([{
            "id":31,"content":"tok31","single_word":false,"lstrip":false,
            "rstrip":false,"normalized":false,"special":true
        }]);
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let tokenizer = Tokenizer::from_file(path).unwrap();
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Lark {
                grammar: "start: \"hello\" (<[31]> | \"tok30\") \"world\"\n".into(),
                prefix: "hello".into(),
            }),
            eos_token_id: Some(3),
            greedy: true,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, Some(&tokenizer), false, None).unwrap();
        let mut logits = vec![0.0; 32];
        chain
            .process(&mut logits, &ProcessorContext::default())
            .unwrap();
        assert_eq!(
            logits
                .iter()
                .enumerate()
                .filter_map(|(id, score)| score.is_finite().then_some(id))
                .collect::<Vec<_>>(),
            [30, 31]
        );
        for token in [31, 30] {
            let mut logits = vec![0.0; 32];
            chain
                .process(
                    &mut logits,
                    &ProcessorContext {
                        generated_tokens: vec![token],
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(
                logits
                    .iter()
                    .enumerate()
                    .filter_map(|(id, score)| score.is_finite().then_some(id))
                    .collect::<Vec<_>>(),
                [5]
            );
        }
    }

    #[test]
    fn lark_prefill_enters_the_existing_grammar_position_without_resampling_it() {
        let tokenizer = Tokenizer::from_file(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-llm/tokenizer.json"),
        )
        .unwrap();
        let options = |prefix: &str| GenerateOptions {
            constraint: Some(GenerateConstraint::Lark {
                grammar: "start: \"hello\" \"world\"\n".into(),
                prefix: prefix.into(),
            }),
            eos_token_id: Some(3),
            greedy: true,
            ..Default::default()
        };
        for (prefix, expected) in [("", 4), ("hello", 5), ("helloworld", 3)] {
            let chain =
                build_processor_chain(&options(prefix), Some(&tokenizer), false, None).unwrap();
            let mut logits = vec![0.0; 32];
            chain
                .process(&mut logits, &ProcessorContext::default())
                .unwrap();
            let allowed = logits
                .iter()
                .enumerate()
                .filter_map(|(id, score)| score.is_finite().then_some(id))
                .collect::<Vec<_>>();
            assert_eq!(allowed, [expected], "prefix={prefix:?}");
            if prefix == "hello" {
                let mut logits = vec![0.0; 32];
                chain
                    .process(
                        &mut logits,
                        &ProcessorContext {
                            generated_tokens: vec![5],
                            generated_text: "world".into(),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert!(logits[3].is_finite());
                assert!(!logits[4].is_finite() && !logits[5].is_finite());
            }
        }
        let error = build_processor_chain(&options("world"), Some(&tokenizer), false, None)
            .err()
            .expect("invalid grammar prefill must be rejected");
        assert!(
            error.is::<crate::logits::ConstraintPrefixError>(),
            "{error:#}"
        );
    }
}
