//! Request-local raw grammar activation and token-aware history replay.
use super::{MAX_ITEMS, replay::GbnfReplay, token_resolver, trigger::EcmaBytePattern};
use crate::{
    config::{GenerateOptions, GrammarActivation, GrammarTrigger},
    logits::{Constraint, ProcessorContext, TokenCandidate, TokenId},
};
use anyhow::{Context, Result, ensure};
use onnx_genai_ort::{TokenizationOptions, Tokenizer};
use std::{collections::HashSet, sync::Mutex};

struct Pattern {
    request_index: usize,
    matcher: EcmaBytePattern,
}

struct Position {
    token: TokenId,
    start: usize,
    end: usize,
}
struct Active {
    cursor: GbnfReplay,
    ended: bool,
}
#[derive(Default)]
struct State {
    history: Vec<TokenId>,
    bytes: Vec<u8>,
    positions: Vec<Position>,
    active: Option<Active>,
}

pub(crate) struct LazyGbnfConstraint {
    initial: GbnfReplay,
    pieces: Vec<Vec<u8>>,
    endings: HashSet<TokenId>,
    tokens: HashSet<TokenId>,
    patterns: Vec<Pattern>,
    max_bytes: usize,
    gate: Option<Mutex<crate::reasoning::GrammarCursor>>,
    state: Mutex<State>,
}
struct ResolvedTriggers {
    tokens: HashSet<TokenId>,
    patterns: Vec<Pattern>,
}
pub(crate) fn validate_activation(
    tokenizer: &Tokenizer,
    activation: &GrammarActivation,
    max_bytes: usize,
    preserved: &std::collections::BTreeSet<TokenId>,
) -> Result<()> {
    resolve_activation(tokenizer, activation, max_bytes, preserved).map(drop)
}
fn resolve_word(
    tokenizer: &Tokenizer,
    word: &str,
    index: usize,
    preserved: &std::collections::BTreeSet<TokenId>,
) -> Result<Option<TokenId>> {
    let ids = tokenizer
        .encode_with_options(
            word,
            TokenizationOptions {
                add_special: false,
                parse_special: true,
            },
        )
        .with_context(|| {
            format!("tokenize grammar_triggers[{index}] word against the model tokenizer")
        })?;
    if let [id] = ids.as_slice() {
        ensure!(
            preserved.contains(id),
            "grammar_triggers[{index}] word {word:?} resolves to token {id} which is absent from preserved_tokens; preserve the word token before using it as a trigger"
        );
        Ok(Some(*id))
    } else {
        Ok(None)
    }
}
pub(crate) fn validate_words(
    tokenizer: &Tokenizer,
    activation: &GrammarActivation,
    preserved: &std::collections::BTreeSet<TokenId>,
) -> Result<()> {
    for (index, trigger) in activation.triggers().iter().enumerate() {
        if let GrammarTrigger::Word(word) = trigger {
            resolve_word(tokenizer, word, index, preserved)?;
        }
    }
    Ok(())
}
fn resolve_activation(
    tokenizer: &Tokenizer,
    activation: &GrammarActivation,
    max_bytes: usize,
    preserved: &std::collections::BTreeSet<TokenId>,
) -> Result<ResolvedTriggers> {
    let mut tokens = HashSet::new();
    let mut patterns = Vec::new();
    let mut source_bytes = 0usize;
    let mut program_bytes = 0usize;
    for (index, trigger) in activation.triggers().iter().enumerate() {
        let mut token = None;
        let pattern = match trigger {
            GrammarTrigger::Token(id) => {
                token = Some(*id);
                None
            }
            GrammarTrigger::Word(word) => {
                if let Some(id) = resolve_word(tokenizer, word, index, preserved)? {
                    token = Some(id);
                    None
                } else {
                    Some(regex::escape(word))
                }
            }
            GrammarTrigger::Pattern(pattern) => Some(pattern.clone()),
            GrammarTrigger::PatternFull(pattern) => Some(if pattern.is_empty() {
                "^$".into()
            } else {
                format!(
                    "{}{}{}",
                    if pattern.starts_with('^') { "" } else { "^" },
                    pattern,
                    if pattern.ends_with('$') { "" } else { "$" }
                )
            }),
        };
        if let Some(id) = token {
            ensure!(
                tokenizer.inner().id_to_token(id).is_some(),
                "grammar_triggers[{index}] token {id} is absent from the model vocabulary; choose an existing token ID"
            );
            tokens.insert(id);
        }
        if let Some(pattern) = pattern {
            source_bytes = source_bytes.checked_add(pattern.len()).context(
                "grammar trigger source byte count overflow; use fewer or shorter patterns",
            )?;
            ensure!(
                source_bytes <= max_bytes,
                "grammar trigger patterns exceed the shared max_constraint_bytes={max_bytes} budget; shorten their combined source"
            );
            let pattern = EcmaBytePattern::compile(&pattern, max_bytes)
                .with_context(|| format!("compile grammar_triggers[{index}]"))?;
            program_bytes = program_bytes
                .checked_add(pattern.program_bytes())
                .context("grammar trigger compiled byte count overflow; simplify the patterns")?;
            ensure!(
                program_bytes <= 1 << 20,
                "grammar trigger patterns exceed the shared one-MiB translated program budget; combine or simplify the patterns"
            );
            patterns.push(Pattern {
                request_index: index,
                matcher: pattern,
            });
        }
    }
    Ok(ResolvedTriggers { tokens, patterns })
}

impl LazyGbnfConstraint {
    pub(crate) fn new(
        grammar: &str,
        tokenizer: &Tokenizer,
        options: &GenerateOptions,
        pieces: Vec<Vec<u8>>,
        endings: Vec<TokenId>,
        activation: &GrammarActivation,
    ) -> Result<Self> {
        let ResolvedTriggers { tokens, patterns } = resolve_activation(
            tokenizer,
            activation,
            options.max_constraint_bytes,
            &options.effective_preserved_tokens(),
        )?;
        for &id in &tokens {
            ensure!(
                (id as usize) < pieces.len(),
                "raw grammar byte vocabulary omits trigger token {id}; repair the tokenizer byte contract"
            );
        }
        Ok(Self {
            initial: GbnfReplay::compile(
                grammar,
                options.max_constraint_bytes,
                token_resolver(tokenizer),
            )?,
            pieces,
            endings: endings.into_iter().collect(),
            tokens,
            patterns,
            max_bytes: options.max_constraint_bytes,
            gate: options
                .reasoning_budget
                .clone()
                .map(crate::reasoning::GrammarCursor::new)
                .transpose()?
                .map(Mutex::new),
            state: Mutex::new(State::default()),
        })
    }

    fn accept(&self, state: &mut State, token: TokenId) -> Result<()> {
        let piece = self.pieces.get(token as usize).with_context(|| format!("lazy grammar history token {token} is outside the model byte vocabulary; repair the generated history"))?;
        if let Some(active) = &mut state.active {
            ensure!(
                !active.ended,
                "lazy grammar history token {token} continues after an ending; discard the stale suffix or start a new completion"
            );
            if self.endings.contains(&token) {
                ensure!(
                    active.cursor.is_accepting(),
                    "lazy grammar ending token {token} precedes root completion; discard the invalid history suffix"
                );
                active.ended = true;
            } else {
                ensure!(
                    active.cursor.consume(token, piece)?,
                    "lazy grammar history token {token} rejects its supplied byte piece; replay only history admitted by the same grammar"
                );
            }
            return Ok(());
        }
        if self.tokens.contains(&token) {
            let mut cursor = self.initial.clone();
            ensure!(
                cursor.consume(token, piece)?,
                "lazy grammar token trigger {token} is rejected at the root; include the triggering token itself in the grammar"
            );
            state.active = Some(Active {
                cursor,
                ended: self.endings.contains(&token),
            });
            state.bytes.clear();
            state.positions.clear();
            return Ok(());
        }
        let end = state
            .bytes
            .len()
            .checked_add(piece.len())
            .context("lazy grammar trigger buffer overflow; shorten generated output")?;
        ensure!(
            end <= self.max_bytes,
            "lazy grammar awaiting-trigger buffer exceeds max_constraint_bytes={}; shorten output or increase its explicit byte budget",
            self.max_bytes
        );
        state.positions.push(Position {
            token,
            start: state.bytes.len(),
            end,
        });
        state.bytes.extend_from_slice(piece);
        for pattern in &self.patterns {
            let index = pattern.request_index;
            if let Some(start) = pattern
                .matcher
                .find(&state.bytes)
                .with_context(|| format!("search grammar_triggers[{index}]"))?
            {
                ensure!(
                    start <= state.bytes.len(),
                    "grammar_triggers[{index}] returned an invalid replay offset; restart with a freshly compiled pattern"
                );
                let mut cursor = self.initial.clone();
                for position in &state.positions {
                    if position.end <= start {
                        continue;
                    }
                    let piece = &state.bytes[position.start.max(start)..position.end];
                    ensure!(
                        cursor
                            .consume(position.token, piece)
                            .with_context(|| format!(
                                "replay lazy grammar trigger piece for token {} at byte {}",
                                position.token,
                                position.start.max(start)
                            ))?,
                        "grammar_triggers[{index}] replay rejects token {} at byte {}; include the captured trigger and its suffix in the grammar",
                        position.token,
                        position.start.max(start)
                    );
                }
                state.active = Some(Active {
                    cursor,
                    ended: false,
                });
                state.bytes.clear();
                state.positions.clear();
                break;
            }
        }
        Ok(())
    }
}
impl Constraint for LazyGbnfConstraint {
    fn allowed_next_tokens(
        &self,
        context: &ProcessorContext,
        candidates: &[TokenCandidate],
    ) -> Result<Vec<bool>> {
        ensure!(
            context.generated_tokens.len() <= MAX_ITEMS,
            "lazy grammar raw history exceeds {MAX_ITEMS} tokens; shorten the completion"
        );
        let mut gate = self
            .gate
            .as_ref()
            .map(|gate| {
                gate.lock().map_err(|_| {
                    anyhow::anyhow!(
                        "lazy grammar reasoning gate is poisoned; restart this completion"
                    )
                })
            })
            .transpose()?;
        let view = gate
            .as_mut()
            .map(|gate| gate.synchronize(&context.generated_tokens, MAX_ITEMS))
            .transpose()?;
        let history = view
            .as_ref()
            .map_or(context.generated_tokens.as_slice(), |view| view.tokens);
        let apply = view.as_ref().is_none_or(|view| view.apply);
        let mut state = self.state.lock().map_err(|_| {
            anyhow::anyhow!("lazy grammar state is poisoned; restart this completion")
        })?;
        if !history.starts_with(&state.history) {
            *state = State::default();
        }
        let result = (|| {
            ensure!(
                history.len() <= MAX_ITEMS,
                "lazy grammar history exceeds {MAX_ITEMS} tokens; shorten the completion"
            );
            for &token in &history[state.history.len()..] {
                self.accept(&mut state, token)?;
                state.history.push(token);
            }
            candidates
                .iter()
                .map(|candidate| {
                    let Some(piece) = self.pieces.get(candidate.token_id as usize) else {
                        return Ok(false);
                    };
                    if !apply {
                        return Ok(true);
                    }
                    let Some(active) = &state.active else {
                        return Ok(true);
                    };
                    if self.endings.contains(&candidate.token_id) {
                        return Ok(active.cursor.is_accepting());
                    }
                    if active.ended || piece.is_empty() || !active.cursor.fits_piece(piece.len()) {
                        return Ok(false);
                    }
                    active
                        .cursor
                        .clone()
                        .consume(candidate.token_id, piece)
                        .with_context(|| {
                            format!(
                                "preview lazy grammar candidate token {}",
                                candidate.token_id
                            )
                        })
                })
                .collect()
        })();
        if result.is_err() {
            *state = State::default();
        }
        result
    }
    fn name(&self) -> &str {
        "lazy_gbnf_constraint"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::GenerateConstraint, logits::ProcessorChain, processors::build_processor_chain,
    };
    fn tokenizer() -> Tokenizer {
        let mut value: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        let vocab = value["model"]["vocab"].as_object_mut().unwrap();
        for (id, text) in [
            (22, "free"),
            (23, "prefixOPENtail"),
            (24, "B"),
            (25, "OPEN"),
            (26, "tail"),
            (27, " "),
            (28, "END"),
            (29, "op"),
            (30, "en"),
            (31, "é"),
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
        Tokenizer::from_file(path).unwrap()
    }
    fn options(grammar: &str, triggers: Vec<GrammarTrigger>) -> GenerateOptions {
        GenerateOptions {
            constraint: Some(GenerateConstraint::Gbnf {
                grammar: grammar.into(),
                activation: Some(GrammarActivation::new(triggers).unwrap()),
            }),
            eos_token_id: Some(3),
            greedy: true,
            ..Default::default()
        }
    }
    fn chain(grammar: &str, triggers: Vec<GrammarTrigger>) -> ProcessorChain {
        build_processor_chain(&options(grammar, triggers), Some(&tokenizer()), false).unwrap()
    }
    fn allowed(chain: &ProcessorChain, history: &[TokenId]) -> Result<Vec<usize>> {
        let mut logits = vec![0.; 32];
        chain.process(
            &mut logits,
            &ProcessorContext {
                generated_tokens: history.to_vec(),
                ..Default::default()
            },
        )?;
        Ok(logits
            .iter()
            .enumerate()
            .filter_map(|(id, score)| score.is_finite().then_some(id))
            .collect())
    }
    #[test]
    fn lazy_chain_token_activation_recovers_after_rollback_and_invalid_suffix() {
        let chain = chain(r#"root ::= <[25]> "B""#, vec![GrammarTrigger::Token(25)]);
        assert_eq!(allowed(&chain, &[22]).unwrap().len(), 32);
        assert_eq!(allowed(&chain, &[22, 25]).unwrap(), [24]);
        assert_eq!(allowed(&chain, &[22, 25, 24]).unwrap(), [3]);
        assert!(allowed(&chain, &[22, 25, 22]).is_err());
        assert_eq!(allowed(&chain, &[22]).unwrap().len(), 32);
        assert_eq!(allowed(&chain, &[22, 22, 25]).unwrap(), [24]);
        assert!(allowed(&chain, &[25, 24, 3, 24]).is_err());
        assert_eq!(allowed(&chain, &[25]).unwrap(), [24]);
    }
    #[test]
    fn lazy_chain_pattern_replays_truncated_piece_with_original_identity() {
        for grammar in [r#"root ::= <[23]> "B""#, r#"root ::= "OPENtailB""#] {
            let chain = chain(
                grammar,
                vec![GrammarTrigger::Pattern("prefix(OPEN)tail".into())],
            );
            assert_eq!(allowed(&chain, &[22, 23]).unwrap(), [24]);
            assert_eq!(allowed(&chain, &[22, 23, 24]).unwrap(), [3]);
        }
        for grammar in ["root ::= <[31]> <[24]>", r#"root ::= <[31]> "B""#] {
            let chain = chain(grammar, vec![GrammarTrigger::Pattern(r"\xA9".into())]);
            assert_eq!(allowed(&chain, &[31]).unwrap(), [24]);
            assert_eq!(allowed(&chain, &[31, 24]).unwrap(), [3]);
        }
    }
    #[test]
    fn lazy_chain_prioritizes_token_triggers_then_pattern_configuration_order() {
        let chain = chain(
            r#"root ::= "prefixOPENtailB""#,
            vec![
                GrammarTrigger::Pattern("tail".into()),
                GrammarTrigger::Token(23),
            ],
        );
        assert_eq!(allowed(&chain, &[23]).unwrap(), [24]);
        let first_tail = self::chain(
            r#"root ::= "tailB""#,
            vec![
                GrammarTrigger::Pattern("tail".into()),
                GrammarTrigger::Pattern("OPEN".into()),
            ],
        );
        assert_eq!(allowed(&first_tail, &[23]).unwrap(), [24]);
        let first_open = self::chain(
            r#"root ::= "tailB""#,
            vec![
                GrammarTrigger::Token(25),
                GrammarTrigger::Pattern("OPEN".into()),
                GrammarTrigger::Pattern("tail".into()),
            ],
        );
        let error = allowed(&first_open, &[23]).unwrap_err();
        assert!(
            format!("{error:#}").contains("grammar_triggers[1]"),
            "{error:#}"
        );
    }
    #[test]
    fn lazy_chain_full_patterns_and_word_resolution_use_actual_tokenizer() {
        let full = chain(
            r#"root ::= "OPENB""#,
            vec![GrammarTrigger::PatternFull("OPEN".into())],
        );
        assert_eq!(allowed(&full, &[22, 25]).unwrap().len(), 32);
        assert_eq!(allowed(&full, &[25]).unwrap(), [24]);
        let mut single = options(
            r#"root ::= <[25]> "B""#,
            vec![GrammarTrigger::Word("OPEN".into())],
        );
        let tokenizer = tokenizer();
        let error = build_processor_chain(&single, Some(&tokenizer), false)
            .err()
            .unwrap();
        assert!(error.to_string().contains("preserved_tokens"), "{error:#}");
        single.preserved_tokens.insert(25);
        let single = build_processor_chain(&single, Some(&tokenizer), false).unwrap();
        assert_eq!(allowed(&single, &[25]).unwrap(), [24]);
        let multi = chain(
            r#"root ::= "op enB""#,
            vec![GrammarTrigger::Word("op en".into())],
        );
        assert_eq!(allowed(&multi, &[22, 29, 27, 30]).unwrap(), [24]);
    }
    #[test]
    fn lazy_chain_validates_triggers_before_sampling_and_bounds_pending_history() {
        assert!(GrammarActivation::new(vec![]).is_err());
        let tokenizer = tokenizer();
        for trigger in [
            GrammarTrigger::Token(99),
            GrammarTrigger::Pattern("(?<=x)y".into()),
        ] {
            assert!(
                build_processor_chain(
                    &options(r#"root ::= "B""#, vec![trigger]),
                    Some(&tokenizer),
                    false
                )
                .is_err()
            );
        }
        let mut bounded = options(
            r#"root ::= "B""#,
            vec![GrammarTrigger::Pattern("NEVER".into())],
        );
        bounded.max_constraint_bytes = 32;
        let bounded = build_processor_chain(&bounded, Some(&tokenizer), false).unwrap();
        let error = allowed(&bounded, &[23, 23, 23]).unwrap_err();
        assert!(format!("{error:#}").contains("awaiting-trigger buffer"));
        assert_eq!(allowed(&bounded, &[]).unwrap().len(), 32);
        assert!(allowed(&bounded, &[99]).is_err());
    }
    fn reasoning_options(
        grammar: &str,
        initially_active: bool,
        prefix_tokens: Vec<TokenId>,
        limit: Option<usize>,
        trigger: TokenId,
    ) -> GenerateOptions {
        let tokenizer = tokenizer();
        let mut options = options(grammar, vec![GrammarTrigger::Token(trigger)]);
        options.reasoning_budget = Some(
            crate::reasoning::ReasoningBudget::resolve(
                &tokenizer,
                crate::reasoning::ReasoningBudgetSpec {
                    start_sequences: vec![vec![29]],
                    initially_active,
                    end_sequences: vec![vec![30, 28], vec![28]],
                    forced_tokens: vec![30, 28],
                    prefix_tokens,
                    max_tokens: limit,
                },
            )
            .unwrap(),
        );
        options
    }
    #[test]
    fn lazy_chain_hides_triggers_inside_reasoning_then_resumes_after_closer() {
        let options = reasoning_options(r#"root ::= "OPENB""#, false, vec![], None, 25);
        let chain = build_processor_chain(&options, Some(&tokenizer()), false).unwrap();
        assert_eq!(allowed(&chain, &[29]).unwrap().len(), 32);
        assert_eq!(allowed(&chain, &[29, 25]).unwrap().len(), 32);
        assert_eq!(allowed(&chain, &[29, 25, 28]).unwrap().len(), 32);
        assert_eq!(allowed(&chain, &[29, 25, 28, 25]).unwrap(), [24]);
        assert_eq!(allowed(&chain, &[29, 25, 28, 25, 24]).unwrap(), [3]);
    }
    #[test]
    fn lazy_chain_replays_complete_longest_reasoning_end_sequence() {
        let grammar = r#"root ::= <[30]> <[28]> "B""#;
        let options = reasoning_options(grammar, false, vec![], None, 30);
        let chain = build_processor_chain(&options, Some(&tokenizer()), false).unwrap();
        assert_eq!(allowed(&chain, &[29, 25, 30]).unwrap().len(), 32);
        assert_eq!(allowed(&chain, &[29, 25, 30, 28]).unwrap(), [24]);
        let prefilled = reasoning_options(grammar, true, vec![30], None, 30);
        let chain = build_processor_chain(&prefilled, Some(&tokenizer()), false).unwrap();
        assert_eq!(allowed(&chain, &[]).unwrap().len(), 32);
        assert_eq!(allowed(&chain, &[28]).unwrap(), [24]);
    }
    #[test]
    fn lazy_chain_budget_forces_closer_without_grammar_conflict_and_rolls_back() {
        let options = reasoning_options(r#"root ::= <[30]> <[28]> "B""#, true, vec![], Some(1), 30);
        let chain = build_processor_chain(&options, Some(&tokenizer()), false).unwrap();
        assert_eq!(allowed(&chain, &[22]).unwrap(), [30]);
        assert_eq!(allowed(&chain, &[22, 30]).unwrap(), [28]);
        assert_eq!(allowed(&chain, &[22, 30, 28]).unwrap(), [24]);
        assert_eq!(allowed(&chain, &[22]).unwrap(), [30]);
        assert_eq!(allowed(&chain, &[22, 30, 28]).unwrap(), [24]);
    }
    #[test]
    fn lazy_chain_live_control_does_not_reinterpret_past_hidden_tokens() {
        let options = reasoning_options(r#"root ::= <[30]> <[28]> "B""#, true, vec![], None, 30);
        let control = options.reasoning_budget.as_ref().unwrap().control();
        let chain = build_processor_chain(&options, Some(&tokenizer()), false).unwrap();
        assert_eq!(allowed(&chain, &[22, 25]).unwrap().len(), 32);
        control.request_end().unwrap();
        assert_eq!(allowed(&chain, &[22, 25]).unwrap(), [30]);
        assert_eq!(allowed(&chain, &[22, 25, 30]).unwrap(), [28]);
        assert_eq!(allowed(&chain, &[22, 25, 30, 28]).unwrap(), [24]);
        assert_eq!(allowed(&chain, &[22, 25]).unwrap(), [30]);
        assert_eq!(allowed(&chain, &[22, 25, 30, 28]).unwrap(), [24]);
    }
    #[test]
    fn lazy_chain_masks_over_budget_candidates_without_losing_legal_eos() {
        let mut options = options(r#"root ::= "OPEN" "B"*"#, vec![GrammarTrigger::Token(25)]);
        options.max_constraint_bytes = 32;
        let chain = build_processor_chain(&options, Some(&tokenizer()), false).unwrap();
        let mut history = vec![25];
        history.extend([24; 28]);
        assert_eq!(allowed(&chain, &history).unwrap(), [3]);
    }
}
