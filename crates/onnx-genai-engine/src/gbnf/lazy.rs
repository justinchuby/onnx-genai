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
enum Compiled {
    Gbnf(GbnfReplay),
    Json(Box<crate::response_constraint::ContentConstraint>),
}
enum Cursor {
    Gbnf(GbnfReplay),
    Json { history: Vec<TokenId> },
}
struct Active {
    cursor: Cursor,
    ended: bool,
}
#[derive(Default)]
struct State {
    history: Vec<TokenId>,
    bytes: Vec<u8>,
    positions: Vec<Position>,
    active: Option<Active>,
}

pub(crate) struct LazyGrammarConstraint {
    initial: Compiled,
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
            let pattern = EcmaBytePattern::compile(&pattern, max_bytes, activation.collation())
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

impl LazyGrammarConstraint {
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
            initial: Compiled::Gbnf(GbnfReplay::compile(
                grammar,
                options.max_constraint_bytes,
                token_resolver(tokenizer),
            )?),
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

    pub(crate) fn json_schema(
        schema: &str,
        tokenizer: &Tokenizer,
        options: &GenerateOptions,
        contract: crate::byte_stream::ByteStreamContract,
        endings: Vec<TokenId>,
        activation: &GrammarActivation,
    ) -> Result<Self> {
        ensure!(
            options.max_constraint_bytes > 0 && schema.len() <= options.max_constraint_bytes,
            "lazy JSON Schema source has {} bytes with max_constraint_bytes={}; shorten the schema or configure a positive sufficient budget",
            schema.len(),
            options.max_constraint_bytes
        );
        let schema = serde_json::from_str(schema)
            .context("parse lazy JSON Schema; supply a valid JSON schema value")?;
        let pieces = contract.pieces.clone();
        let initial = crate::response_constraint::ContentConstraint::raw_json_schema(
            tokenizer,
            &schema,
            options.max_constraint_bytes,
            contract,
        )
        .context("compile lazy JSON Schema before sampling")?;
        let ResolvedTriggers { tokens, patterns } = resolve_activation(
            tokenizer,
            activation,
            options.max_constraint_bytes,
            &options.effective_preserved_tokens(),
        )?;
        Ok(Self {
            initial: Compiled::Json(Box::new(initial)),
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

    fn start(&self) -> Cursor {
        match &self.initial {
            Compiled::Gbnf(initial) => Cursor::Gbnf(initial.clone()),
            Compiled::Json(_) => Cursor::Json {
                history: Vec::new(),
            },
        }
    }

    fn consume(
        &self,
        cursor: &mut Cursor,
        token: TokenId,
        piece: &[u8],
        replay: bool,
    ) -> Result<bool> {
        match (cursor, &self.initial) {
            (Cursor::Gbnf(cursor), _) => cursor.consume(token, piece),
            (Cursor::Json { history }, Compiled::Json(initial)) => {
                if replay {
                    // All captured bytes are seeded together by activation.
                    initial.seed_raw_bytes(piece)?;
                } else {
                    let mut selected = history.clone();
                    selected.push(token);
                    initial.allowed_next_tokens(
                        &ProcessorContext {
                            generated_tokens: selected,
                            ..Default::default()
                        },
                        &[],
                    )?;
                    history.push(token);
                }
                Ok(true)
            }
            _ => anyhow::bail!(
                "lazy grammar cursor does not match its compiled recognizer; restart this completion"
            ),
        }
    }

    fn complete(&self, cursor: &Cursor) -> Result<bool> {
        match (cursor, &self.initial) {
            (Cursor::Gbnf(cursor), _) => Ok(cursor.is_accepting()),
            (Cursor::Json { history }, Compiled::Json(initial)) => {
                initial.can_finish(&ProcessorContext {
                    generated_tokens: history.clone(),
                    ..Default::default()
                })
            }
            _ => anyhow::bail!(
                "lazy grammar cursor does not match its compiled recognizer; restart this completion"
            ),
        }
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
                    self.complete(&active.cursor)?,
                    "lazy grammar ending token {token} precedes root completion; discard the invalid history suffix"
                );
                active.ended = true;
            } else {
                ensure!(
                    self.consume(&mut active.cursor, token, piece, false)?,
                    "lazy grammar history token {token} rejects its supplied byte piece; replay only history admitted by the same grammar"
                );
            }
            return Ok(());
        }
        if self.tokens.contains(&token) {
            let mut cursor = self.start();
            ensure!(
                self.consume(&mut cursor, token, piece, true)?,
                "lazy grammar token trigger {token} is rejected at the root; include the triggering token itself in the grammar"
            );
            let ended = self.endings.contains(&token);
            ensure!(
                !ended || self.complete(&cursor)?,
                "lazy grammar ending trigger token {token} precedes root completion; include a complete root in the triggering token or choose another trigger"
            );
            state.active = Some(Active { cursor, ended });
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
                let mut cursor = self.start();
                if matches!(cursor, Cursor::Json { .. }) {
                    self.consume(&mut cursor, token, &state.bytes[start..], true)
                        .with_context(|| {
                            format!("replay grammar_triggers[{index}] JSON suffix at byte {start}")
                        })?;
                } else {
                    for position in &state.positions {
                        if position.end <= start {
                            continue;
                        }
                        let piece = &state.bytes[position.start.max(start)..position.end];
                        ensure!(
                            self.consume(&mut cursor, position.token, piece, true)
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
                }
                let ended = self.endings.contains(&token);
                ensure!(
                    !ended || self.complete(&cursor)?,
                    "grammar_triggers[{index}] activates on ending token {token} before root completion; use a trigger whose replay completes the root before ending"
                );
                state.active = Some(Active { cursor, ended });
                state.bytes.clear();
                state.positions.clear();
                break;
            }
        }
        Ok(())
    }
}
impl Constraint for LazyGrammarConstraint {
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
            if apply
                && let Some(Active {
                    cursor: Cursor::Json { history },
                    ended,
                }) = &state.active
            {
                let Compiled::Json(initial) = &self.initial else {
                    anyhow::bail!(
                        "lazy JSON cursor has no compiled schema; restart this completion"
                    )
                };
                if *ended {
                    return Ok(vec![false; candidates.len()]);
                }
                let context = ProcessorContext {
                    generated_tokens: history.clone(),
                    ..Default::default()
                };
                let mut mask = initial.allowed_next_tokens(&context, candidates)?;
                let complete = initial.can_finish(&context)?;
                for (candidate, allowed) in candidates.iter().zip(&mut mask) {
                    if self.endings.contains(&candidate.token_id) {
                        *allowed = complete;
                    }
                }
                return Ok(mask);
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
                    if active.ended {
                        return Ok(false);
                    }
                    if self.endings.contains(&candidate.token_id) {
                        return self.complete(&active.cursor);
                    }
                    let Cursor::Gbnf(cursor) = &active.cursor else {
                        anyhow::bail!("lazy grammar cursor has no compiled GBNF recognizer; restart this completion")
                    };
                    if piece.is_empty() || !cursor.fits_piece(piece.len()) {
                        return Ok(false);
                    }
                    cursor
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
    fn needs_selection_feedback(&self) -> bool {
        true
    }

    fn on_selected(&self, token: TokenId, context: &ProcessorContext) -> Result<()> {
        ensure!(
            context.generated_tokens.len() < MAX_ITEMS,
            "lazy grammar selected token exceeds the {MAX_ITEMS}-token history budget; shorten the completion"
        );
        let mut history = context.generated_tokens.clone();
        history.push(token);
        self.allowed_next_tokens(
            &ProcessorContext {
                generated_tokens: history,
                ..Default::default()
            },
            &[],
        )
        .with_context(|| {
            format!("validate selected lazy grammar token {token} before publishing output")
        })?;
        Ok(())
    }

    fn name(&self) -> &str {
        "lazy_grammar_constraint"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::GenerateConstraint, logits::ProcessorChain, processors::build_processor_chain,
    };
    fn tokenizer() -> Tokenizer {
        tokenizer_pieces(&[
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
        ])
    }
    fn tokenizer_pieces(pieces: &[(u64, &str)]) -> Tokenizer {
        tokenizer_with_decoder(pieces, serde_json::json!({"type":"ByteFallback"}))
    }
    fn tokenizer_with_decoder(pieces: &[(u64, &str)], decoder: serde_json::Value) -> Tokenizer {
        let mut value: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        let vocab = value["model"]["vocab"].as_object_mut().unwrap();
        for &(id, text) in pieces {
            let old = vocab
                .iter()
                .find(|(_, value)| value.as_u64() == Some(id))
                .unwrap()
                .0
                .clone();
            vocab.remove(&old);
            vocab.insert(text.into(), serde_json::json!(id));
        }
        value["decoder"] = decoder;
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        Tokenizer::from_file(path).unwrap()
    }
    fn options(grammar: &str, triggers: Vec<GrammarTrigger>) -> GenerateOptions {
        GenerateOptions {
            constraint: Some(GenerateConstraint::Gbnf {
                grammar: grammar.into(),
                activation: Some(
                    GrammarActivation::new(triggers, crate::GrammarCollation::Libcxx).unwrap(),
                ),
            }),
            eos_token_id: Some(3),
            greedy: true,
            ..Default::default()
        }
    }
    fn chain(grammar: &str, triggers: Vec<GrammarTrigger>) -> ProcessorChain {
        build_processor_chain(&options(grammar, triggers), Some(&tokenizer()), false, None).unwrap()
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
    fn json_chain(schema: &str, triggers: Vec<GrammarTrigger>) -> ProcessorChain {
        let tokenizer = tokenizer_pieces(&[
            (22, "free"),
            (23, "prefix{\"n\":"),
            (24, "1}"),
            (25, "{\"n\":"),
            (26, "2}"),
            (27, "{\"n\":2}"),
            (28, "\"A\""),
            (29, "\"\\u0041\""),
            (30, "{\"n\":1}"),
            (31, "é"),
        ]);
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::LazyJsonSchema {
                schema: schema.into(),
                activation: GrammarActivation::new(triggers, crate::GrammarCollation::Libcxx)
                    .unwrap(),
            }),
            eos_token_id: Some(3),
            greedy: true,
            ..Default::default()
        };
        build_processor_chain(&options, Some(&tokenizer), false, None).unwrap()
    }

    #[test]
    fn lazy_json_uses_raw_bytes_after_trigger_despite_publication_stripping() {
        let tokenizer = tokenizer_with_decoder(
            &[(22, "\""), (23, "A"), (24, "B")],
            serde_json::json!({"type":"Sequence","decoders":[
                {"type":"ByteFallback"}, {"type":"Fuse"},
                {"type":"Strip","content":"\"","start":1,"stop":1}
            ]}),
        );
        assert_eq!(tokenizer.inner().decode(&[22, 23, 22], false).unwrap(), "A");
        let projected = crate::response_constraint::ContentConstraint::json_schema(
            &tokenizer,
            &serde_json::json!({"const":"A"}),
            crate::text_projection::ProjectionKind::Plain,
            "",
            4096,
            vec![3],
            &Default::default(),
            None,
        )
        .unwrap();
        assert!(
            projected
                .allowed_next_tokens(
                    &ProcessorContext {
                        generated_tokens: vec![22, 23, 22],
                        ..Default::default()
                    },
                    &[],
                )
                .is_err(),
            "published A lacks JSON quotes and must remain invalid for content constraints"
        );

        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::LazyJsonSchema {
                schema: r#"{"const":"A"}"#.into(),
                activation: GrammarActivation::new(
                    vec![GrammarTrigger::Token(22)],
                    crate::GrammarCollation::Libcxx,
                )
                .unwrap(),
            }),
            eos_token_id: Some(3),
            ..Default::default()
        };
        let chain = build_processor_chain(&options, Some(&tokenizer), false, None).unwrap();
        assert!(allowed(&chain, &[22]).unwrap().contains(&23));
        assert!(!allowed(&chain, &[22]).unwrap().contains(&24));
        assert!(allowed(&chain, &[22, 23]).unwrap().contains(&22));
        assert_eq!(allowed(&chain, &[22, 23, 22]).unwrap(), [3]);
        assert!(allowed(&chain, &[22, 24]).is_err());
        assert!(allowed(&chain, &[22]).unwrap().contains(&23));
    }

    #[test]
    fn lazy_json_schema_replays_capture_and_resets_after_rewind_and_rejection() {
        let chain = json_chain(
            r#"{"type":"object","properties":{"n":{"const":1}},"required":["n"],"additionalProperties":false}"#,
            vec![GrammarTrigger::Pattern(r"prefix(\{)".into())],
        );
        assert_eq!(allowed(&chain, &[22]).unwrap().len(), 32);
        let mask = allowed(&chain, &[22, 23]).unwrap();
        assert!(mask.contains(&24), "{mask:?}");
        assert!(!mask.contains(&26) && !mask.contains(&3), "{mask:?}");
        assert_eq!(allowed(&chain, &[22, 23, 24]).unwrap(), [3]);
        assert!(allowed(&chain, &[22, 23, 26]).is_err());
        assert_eq!(allowed(&chain, &[22]).unwrap().len(), 32);
        assert!(allowed(&chain, &[23]).unwrap().contains(&24));
    }

    #[test]
    fn lazy_json_schema_preserves_equivalent_escapes_and_terminal_feedback() {
        let chain = json_chain(r#"{"const":"A"}"#, vec![GrammarTrigger::Token(29)]);
        chain.on_selected(29, &ProcessorContext::default()).unwrap();
        assert_eq!(allowed(&chain, &[29]).unwrap(), [3]);
        chain
            .on_selected(
                3,
                &ProcessorContext {
                    generated_tokens: vec![29],
                    ..Default::default()
                },
            )
            .unwrap();
        let invalid = json_chain(r#"{"const":"B"}"#, vec![GrammarTrigger::Token(29)]);
        assert!(
            invalid
                .on_selected(29, &ProcessorContext::default())
                .is_err()
        );
        assert_eq!(allowed(&invalid, &[]).unwrap().len(), 32);
    }

    #[test]
    fn lazy_chain_checks_last_selected_trigger_before_publication_and_recovers() {
        let chain = chain(r#"root ::= "OPENB""#, vec![GrammarTrigger::Token(23)]);
        let context = ProcessorContext {
            generated_tokens: vec![22],
            ..Default::default()
        };
        assert_eq!(allowed(&chain, &[22]).unwrap().len(), 32);
        let error = chain.on_selected(23, &context).unwrap_err();
        assert!(
            format!("{error:#}").contains("token trigger 23"),
            "{error:#}"
        );
        assert_eq!(allowed(&chain, &[22]).unwrap().len(), 32);
        chain.on_selected(25, &context).unwrap();
        assert_eq!(allowed(&chain, &[22, 25]).unwrap().len(), 32);
    }

    #[test]
    fn lazy_chain_selected_trigger_validates_root_and_selected_eos() {
        let chain = chain(r#"root ::= "OPENB""#, vec![GrammarTrigger::Token(25)]);
        chain.on_selected(25, &ProcessorContext::default()).unwrap();
        let context = ProcessorContext {
            generated_tokens: vec![25],
            ..Default::default()
        };
        assert_eq!(allowed(&chain, &[25]).unwrap(), [24]);
        let error = chain.on_selected(3, &context).unwrap_err();
        assert!(
            format!("{error:#}").contains("precedes root completion"),
            "{error:#}"
        );
        chain.on_selected(24, &context).unwrap();
        assert_eq!(allowed(&chain, &[25, 24]).unwrap(), [3]);
        chain
            .on_selected(
                3,
                &ProcessorContext {
                    generated_tokens: vec![25, 24],
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(allowed(&chain, &[25, 24, 3]).is_err());
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
        let error = build_processor_chain(&single, Some(&tokenizer), false, None)
            .err()
            .unwrap();
        assert!(error.to_string().contains("preserved_tokens"), "{error:#}");
        single.preserved_tokens.insert(25);
        let single = build_processor_chain(&single, Some(&tokenizer), false, None).unwrap();
        assert_eq!(allowed(&single, &[25]).unwrap(), [24]);
        let multi = chain(
            r#"root ::= "op enB""#,
            vec![GrammarTrigger::Word("op en".into())],
        );
        assert_eq!(allowed(&multi, &[22, 29, 27, 30]).unwrap(), [24]);
    }
    #[test]
    fn lazy_chain_validates_triggers_before_sampling_and_bounds_pending_history() {
        assert!(GrammarActivation::new(vec![], crate::GrammarCollation::Libcxx).is_err());
        let tokenizer = tokenizer();
        for trigger in [
            GrammarTrigger::Token(99),
            GrammarTrigger::Pattern("(?<=x)y".into()),
        ] {
            assert!(
                build_processor_chain(
                    &options(r#"root ::= "B""#, vec![trigger]),
                    Some(&tokenizer),
                    false,
                    None,
                )
                .is_err()
            );
        }
        let mut bounded = options(
            r#"root ::= "B""#,
            vec![GrammarTrigger::Pattern("NEVER".into())],
        );
        bounded.max_constraint_bytes = 32;
        let bounded = build_processor_chain(&bounded, Some(&tokenizer), false, None).unwrap();
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
        let chain = build_processor_chain(&options, Some(&tokenizer()), false, None).unwrap();
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
        let chain = build_processor_chain(&options, Some(&tokenizer()), false, None).unwrap();
        assert_eq!(allowed(&chain, &[29, 25, 30]).unwrap().len(), 32);
        assert_eq!(allowed(&chain, &[29, 25, 30, 28]).unwrap(), [24]);
        let prefilled = reasoning_options(grammar, true, vec![30], None, 30);
        let chain = build_processor_chain(&prefilled, Some(&tokenizer()), false, None).unwrap();
        assert_eq!(allowed(&chain, &[]).unwrap().len(), 32);
        assert_eq!(allowed(&chain, &[28]).unwrap(), [24]);
    }
    #[test]
    fn lazy_chain_budget_forces_closer_without_grammar_conflict_and_rolls_back() {
        let options = reasoning_options(r#"root ::= <[30]> <[28]> "B""#, true, vec![], Some(1), 30);
        let chain = build_processor_chain(&options, Some(&tokenizer()), false, None).unwrap();
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
        let chain = build_processor_chain(&options, Some(&tokenizer()), false, None).unwrap();
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
        let chain = build_processor_chain(&options, Some(&tokenizer()), false, None).unwrap();
        let mut history = vec![25];
        history.extend([24; 28]);
        assert_eq!(allowed(&chain, &history).unwrap(), [3]);
    }
}
