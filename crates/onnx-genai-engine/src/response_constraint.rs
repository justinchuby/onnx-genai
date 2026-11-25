//! Model-token constraints over the shared declared response projection.
use crate::byte_grammar::{
    GrammarByteBudgetExceeded, GrammarByteCursor, GrammarCandidate, GrammarTokenVocabulary,
};
use crate::byte_stream::{ByteStreamContract, ByteStreamState};
use crate::json_prefix::{InvalidJsonEscape, JsonPrefix};
use crate::logits::{Constraint, ConstraintPrefixError, ProcessorContext, TokenCandidate, TokenId};
use crate::response_template::{
    IncompleteResponseUtf8, MissingResponseField, NonMonotonicProjection, ResponseBufferExceeded,
    projection_extension,
};
use crate::text_projection::{ProjectionKind, TextProjection, TextView};
use crate::tokenization::{NonMonotonicText, TokenTextDecoder};
use anyhow::{Context, ensure};
use onnx_genai_ort::Tokenizer;
use std::{collections::HashSet, sync::Mutex};

#[derive(Debug, thiserror::Error)]
#[error(
    "response content candidate evaluation failed and state recovery also failed: {recovery:#}"
)]
struct ResponseConstraintRecovery {
    #[source]
    operation: anyhow::Error,
    recovery: anyhow::Error,
}

enum Codec {
    Bytes(ByteStreamContract),
    Canonical(Box<Tokenizer>),
}

#[derive(Clone)]
enum DecoderState {
    Bytes(ByteStreamState),
    Canonical(TokenTextDecoder),
}

impl Codec {
    fn new_state(&self, preserved: &std::collections::BTreeSet<u32>) -> DecoderState {
        match self {
            Self::Bytes(contract) => DecoderState::Bytes(contract.new_state()),
            Self::Canonical(_) => {
                DecoderState::Canonical(TokenTextDecoder::with_preserved_tokens(preserved))
            }
        }
    }
}

impl DecoderState {
    fn is_identity(&self) -> bool {
        matches!(self, Self::Bytes(state) if state.is_identity())
    }
}

#[derive(Clone, Copy)]
enum Normalization {
    Json,
    Literal,
}
impl Normalization {
    fn parse(self, text: &str) -> anyhow::Result<LogicalPrefix> {
        match self {
            Self::Json => JsonPrefix::parse(text).map(LogicalPrefix::Json),
            Self::Literal => Ok(LogicalPrefix::Literal(text.to_owned())),
        }
    }
}
enum LogicalPrefix {
    Json(JsonPrefix),
    Literal(String),
}
impl LogicalPrefix {
    fn canonical(&self) -> &str {
        match self {
            Self::Json(prefix) => &prefix.canonical,
            Self::Literal(text) => text,
        }
    }
    fn has_pending_escape(&self) -> bool {
        matches!(self, Self::Json(prefix) if prefix.has_pending_escape())
    }
    fn pending_piece_syntax_viable(&self, piece: &[u8]) -> bool {
        match self {
            Self::Json(prefix) => prefix.pending_piece_syntax_viable(piece),
            Self::Literal(_) => true,
        }
    }
    fn preview(
        &self,
        cursor: &mut GrammarByteCursor,
        committed: &str,
        pending: &[u8],
    ) -> anyhow::Result<GrammarCandidate> {
        match self {
            Self::Json(prefix) => prefix.preview(cursor, committed, pending),
            Self::Literal(text) => {
                if let Some(suffix) = text.strip_prefix(committed) {
                    let mut bytes = suffix.as_bytes().to_vec();
                    bytes.extend_from_slice(pending);
                    cursor.preview(&bytes)
                } else {
                    let mut bytes = text.as_bytes().to_vec();
                    bytes.extend_from_slice(pending);
                    cursor.preview_root(&bytes)
                }
            }
        }
    }
}

struct State {
    projection: TextProjection,
    decoder: DecoderState,
    json: GrammarByteCursor,
    history: Vec<TokenId>,
    raw: String,
    normalized: LogicalPrefix,
    ended: bool,
}

struct ConstraintTokens {
    endings: Vec<TokenId>,
    preserved: std::collections::BTreeSet<u32>,
}

pub(crate) struct ContentConstraint {
    normalization: Normalization,
    projection_kind: ProjectionKind,
    prefix: String,
    max_bytes: usize,
    endings: HashSet<TokenId>,
    vocabulary: HashSet<TokenId>,
    codec: Codec,
    plain_vocabulary: Option<GrammarTokenVocabulary>,
    preserved: std::collections::BTreeSet<u32>,
    state: Mutex<State>,
}

impl ContentConstraint {
    pub(crate) fn json_schema(
        tokenizer: &Tokenizer,
        schema: &serde_json::Value,
        projection_kind: ProjectionKind,
        prefix: &str,
        max_bytes: usize,
        endings: Vec<TokenId>,
        preserved: &std::collections::BTreeSet<u32>,
    ) -> anyhow::Result<Self> {
        Self::new(
            tokenizer,
            GrammarByteCursor::json_schema(schema, max_bytes)?,
            Normalization::Json,
            projection_kind,
            prefix,
            max_bytes,
            ConstraintTokens {
                endings,
                preserved: preserved.clone(),
            },
        )
    }
    pub(crate) fn text(
        tokenizer: &Tokenizer,
        grammar: &crate::config::TextGrammar,
        projection_kind: ProjectionKind,
        prefix: &str,
        max_bytes: usize,
        endings: Vec<TokenId>,
        preserved: &std::collections::BTreeSet<u32>,
    ) -> anyhow::Result<Self> {
        let cursor = match grammar {
            crate::config::TextGrammar::Regex(pattern) => {
                GrammarByteCursor::regex(pattern, max_bytes)?
            }
            crate::config::TextGrammar::Lark(source) => GrammarByteCursor::lark(source, max_bytes)?,
        };
        Self::new(
            tokenizer,
            cursor,
            Normalization::Literal,
            projection_kind,
            prefix,
            max_bytes,
            ConstraintTokens {
                endings,
                preserved: preserved.clone(),
            },
        )
    }
    fn new(
        tokenizer: &Tokenizer,
        mut json: GrammarByteCursor,
        normalization: Normalization,
        projection_kind: ProjectionKind,
        prefix: &str,
        max_bytes: usize,
        tokens: ConstraintTokens,
    ) -> anyhow::Result<Self> {
        let ConstraintTokens { endings, preserved } = tokens;
        let vocabulary = tokenizer
            .inner()
            .get_vocab(true)
            .values()
            .copied()
            .collect::<HashSet<_>>();
        let codec = match ByteStreamContract::resolve_output(tokenizer, &preserved)? {
            Some(contract) => Codec::Bytes(contract),
            None => Codec::Canonical(Box::new(tokenizer.clone())),
        };
        let mut projection = projection_kind.create(prefix, max_bytes)?;
        let raw = projection.content(false)?;
        let normalized = (|| -> anyhow::Result<LogicalPrefix> {
            let normalized = normalization.parse(&raw)?;
            if let GrammarCandidate::Rejected { byte_offset } = normalized.preview(&mut json, "", b"")? {
                anyhow::bail!("content constraint prefix is rejected by its grammar at logical byte {byte_offset}; provide a grammar-valid partial content value or adjust the grammar");
            }
            json.commit(normalized.canonical().as_bytes())
                .context("commit normalized assistant prefill to the JSON recognizer")?;
            Ok(normalized)
        })().map_err(|source| ConstraintPrefixError { source })?;
        let plain_vocabulary = match (&projection_kind, &codec) {
            (ProjectionKind::Plain, Codec::Bytes(contract)) => {
                Some(GrammarTokenVocabulary::new(&contract.pieces)?)
            }
            _ => None,
        };
        let decoder = codec.new_state(&preserved);
        Ok(Self {
            normalization,
            projection_kind,
            prefix: prefix.into(),
            max_bytes,
            endings: endings.into_iter().collect(),
            vocabulary,
            codec,
            plain_vocabulary,
            preserved: preserved.clone(),
            state: Mutex::new(State {
                projection,
                decoder,
                json,
                history: Vec::new(),
                raw,
                normalized,
                ended: false,
            }),
        })
    }

    fn reset(&self, state: &mut State) -> anyhow::Result<()> {
        state.projection = self.projection_kind.create(&self.prefix, self.max_bytes)?;
        state.decoder = self.codec.new_state(&self.preserved);
        state.history.clear();
        state.raw = state.projection.content(false)?;
        state.normalized = self.normalization.parse(&state.raw)?;
        state.json.reset();
        state.json.commit(state.normalized.canonical().as_bytes())?;
        state.ended = false;
        Ok(())
    }

    fn inspect_grammar(
        view: TextView<'_>,
        json: &mut GrammarByteCursor,
        base_raw: &str,
        base_canonical: &str,
        terminal: bool,
        normalization: Normalization,
    ) -> anyhow::Result<bool> {
        let published = view.content(terminal);
        if projection_extension(&published, base_raw, terminal, "content")?.is_none() {
            return Ok(true);
        }
        let grammar = if terminal {
            published.clone()
        } else {
            view.grammar_content()
        };
        let normalized = normalization.parse(&grammar)?;
        let pending = view.pending_bytes()?;
        let status = normalized.preview(json, base_canonical, pending)?;
        if let GrammarCandidate::Viable { complete } = status
            && (!terminal || complete)
        {
            return Ok(true);
        }
        // End trimming can hide whitespace only after a complete value. An
        // incomplete JSON string must validate its provisional trailing data.
        if pending.is_empty() || view.pending_can_be_trimmed() {
            let trimmed = normalization.parse(&published)?;
            let status = trimmed.preview(json, base_canonical, b"")?;
            return Ok(matches!(
                status,
                GrammarCandidate::Viable { complete: true }
            ));
        }
        Ok(false)
    }

    fn inspect(&self, state: &mut State, id: TokenId) -> anyhow::Result<bool> {
        if !self.vocabulary.contains(&id) || state.ended {
            return Ok(false);
        }
        let terminal = self.endings.contains(&id);
        let mut fork = state.decoder.clone();
        let bytes = self.decode(&mut fork, id, terminal, true)?;
        let State {
            projection,
            json,
            raw,
            normalized,
            ..
        } = state;
        projection.with_preview(&bytes, terminal, |view| {
            Self::inspect_grammar(
                view,
                json,
                raw,
                normalized.canonical(),
                terminal,
                self.normalization,
            )
        })
    }

    fn token_allowed(&self, state: &mut State, id: TokenId) -> anyhow::Result<bool> {
        Self::admit_preview(self.inspect(state, id))
    }

    fn admit_preview(result: anyhow::Result<bool>) -> anyhow::Result<bool> {
        match result {
            Err(error)
                if error.is::<std::str::Utf8Error>()
                    || error.is::<IncompleteResponseUtf8>()
                    || error.is::<MissingResponseField>()
                    || error.is::<ResponseBufferExceeded>()
                    || error.is::<GrammarByteBudgetExceeded>()
                    || error.is::<NonMonotonicText>()
                    || error.is::<InvalidJsonEscape>()
                    || error.is::<NonMonotonicProjection>() =>
            {
                Ok(false)
            }
            result => result,
        }
    }

    /// Test end of the logical decoded text without inventing an EOS token.
    /// Ordinary ending tokens may contain real grammar characters; checking a
    /// candidate ending is therefore different from checking existing output.
    pub(crate) fn can_finish(&self, context: &ProcessorContext) -> anyhow::Result<bool> {
        self.allowed_next_tokens(context, &[])?;
        let mut state = self.state.lock().map_err(|_| anyhow::anyhow!("response content constraint state is poisoned; restart generation with a fresh constraint"))?;
        if state.ended {
            return Ok(true);
        }
        let mut decoder = state.decoder.clone();
        let pending = match (&self.codec, &mut decoder) {
            (Codec::Bytes(_), DecoderState::Bytes(state)) => state.finish(),
            (Codec::Canonical(tokenizer), DecoderState::Canonical(state)) => {
                state.finish(tokenizer)?.into_bytes()
            }
            _ => anyhow::bail!(
                "logical completion decoder does not match its tokenizer contract; restart generation with a fresh constraint"
            ),
        };
        let State {
            projection,
            json,
            raw,
            normalized,
            ..
        } = &mut *state;
        Self::admit_preview(projection.with_preview(&pending, true, |view| {
            Self::inspect_grammar(
                view,
                json,
                raw,
                normalized.canonical(),
                true,
                self.normalization,
            )
        }))
    }

    fn decode(
        &self,
        decoder: &mut DecoderState,
        id: TokenId,
        terminal: bool,
        preview: bool,
    ) -> anyhow::Result<Vec<u8>> {
        match (&self.codec, decoder) {
            (Codec::Bytes(contract), DecoderState::Bytes(state)) => {
                let piece = contract
                    .pieces
                    .get(id as usize)
                    .context("content grammar token has no declared byte piece")?;
                let mut bytes = state.step(piece).into_owned();
                if terminal {
                    bytes.extend(state.finish());
                }
                Ok(bytes)
            }
            (Codec::Canonical(tokenizer), DecoderState::Canonical(state)) => {
                let mut text = state.step(tokenizer, id)?;
                if terminal {
                    text.push_str(&state.finish(tokenizer)?);
                } else if preview {
                    text.push_str(&state.clone().finish(tokenizer)?);
                }
                Ok(text.into_bytes())
            }
            _ => anyhow::bail!(
                "content grammar decoder state does not match its resolved tokenizer contract; restart generation with a fresh constraint"
            ),
        }
    }

    fn advance(&self, state: &mut State, id: TokenId) -> anyhow::Result<()> {
        let bytes = self.decode(&mut state.decoder, id, self.endings.contains(&id), false)?;
        state.projection.commit(&bytes)?;
        let raw = if self.endings.contains(&id) {
            state.projection.content(true)?
        } else {
            state.projection.content(false)?
        };
        if projection_extension(&raw, &state.raw, self.endings.contains(&id), "content")?.is_some()
        {
            let normalized = self.normalization.parse(&raw)?;
            if let Some(suffix) = normalized
                .canonical()
                .strip_prefix(state.normalized.canonical())
            {
                state.json.commit(suffix.as_bytes())?;
            } else {
                state
                    .json
                    .replace_prefix(normalized.canonical().as_bytes())?;
            }
            state.raw = raw;
            state.normalized = normalized;
        }
        state.history.push(id);
        state.ended = self.endings.contains(&id);
        Ok(())
    }
}

impl Constraint for ContentConstraint {
    fn allowed_next_tokens(
        &self,
        context: &ProcessorContext,
        candidates: &[TokenCandidate],
    ) -> anyhow::Result<Vec<bool>> {
        let mut state = self.state.lock().map_err(|_|anyhow::anyhow!(
            "response content constraint state is poisoned; restart this generation with a fresh constraint"))?;
        if !context.generated_tokens.starts_with(&state.history) {
            self.reset(&mut state)?;
        }
        let result = (|| {
            for &id in &context.generated_tokens[state.history.len()..] {
                ensure!(
                    self.token_allowed(&mut state, id)?,
                    "response content history token {id} at generated offset {} is rejected; supply history admitted by the same tokenizer, template and grammar",
                    state.history.len()
                );
                self.advance(&mut state, id)?;
            }
            if state.ended {
                return Ok(vec![false; candidates.len()]);
            }
            if let (Some(vocabulary), Codec::Bytes(contract)) =
                (&self.plain_vocabulary, &self.codec)
                && state.decoder.is_identity()
                && !state.normalized.has_pending_escape()
            {
                let pending = state
                    .projection
                    .plain_pending()?
                    .context("compiled plain JSON vocabulary requires a plain text projection")?
                    .to_vec();
                let mask = state.json.vocabulary_mask(vocabulary, &pending)?;
                return candidates
                    .iter()
                    .map(|candidate| {
                        let id = candidate.token_id;
                        let piece = contract.pieces.get(id as usize);
                        if !self.endings.contains(&id)
                            && let Some(piece) =
                                piece.filter(|piece| !piece.is_empty() && !piece.contains(&b'\\'))
                            && !(state.normalized.canonical().is_empty()
                                && piece.first().is_some_and(u8::is_ascii_whitespace))
                        {
                            Ok((id as usize) < mask.len()
                                && mask.get(id as usize)
                                && state.projection.can_append_plain(piece.len()))
                        } else {
                            self.token_allowed(&mut state, id)
                        }
                    })
                    .collect();
            }
            candidates
                .iter()
                .map(|candidate| {
                    if self.plain_vocabulary.is_some()
                        && state.normalized.has_pending_escape()
                        && state.decoder.is_identity()
                        && let Codec::Bytes(contract) = &self.codec
                        && !contract
                            .pieces
                            .get(candidate.token_id as usize)
                            .is_some_and(|piece| {
                                state.normalized.pending_piece_syntax_viable(piece)
                            })
                    {
                        return Ok(false);
                    }
                    self.token_allowed(&mut state, candidate.token_id)
                })
                .collect()
        })();
        match result {
            Err(operation) => {
                if let Err(recovery) = self.reset(&mut state) {
                    return Err(ResponseConstraintRecovery {
                        operation,
                        recovery,
                    }
                    .into());
                }
                Err(operation)
            }
            result => result,
        }
    }
    fn name(&self) -> &str {
        match self.normalization {
            Normalization::Json => "json_constraint",
            Normalization::Literal => "response_grammar_constraint",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{GenerateConstraint, GenerateOptions};
    use crate::processors::build_processor_chain;
    use crate::response_template::ResponseTemplate;
    use serde_json::{Value, json};

    fn tokenizer(byte_level: bool) -> Tokenizer {
        let mut value: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        let vocab = value["model"]["vocab"].as_object_mut().unwrap();
        for (old, new, id) in [
            ("hello", if byte_level { "Ã" } else { "hello" }, 4),
            ("world", if byte_level { "©" } else { "world" }, 5),
            ("tok16", if byte_level { "ÿ" } else { "tok16" }, 16),
            ("tok22", "[/r]{\"n\":1}", 22),
            ("tok23", "[/r]{\"n\":2}", 23),
            ("tok24", "[/r]bad", 24),
            ("tok25", "thought", 25),
        ] {
            vocab.remove(old);
            vocab.insert(new.into(), json!(id));
        }
        if byte_level {
            value["model"] = json!({"type":"BPE","unk_token":"<unk>","byte_fallback":false,
                "vocab":value["model"]["vocab"],"merges":[]});
            value["decoder"] = json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false});
        }
        value["added_tokens"] = json!([{"id":3,"content":"<eos>","single_word":false,"lstrip":false,
            "rstrip":false,"normalized":false,"special":true}]);
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        Tokenizer::from_file(path).unwrap()
    }

    #[test]
    fn logical_text_grammar_checks_ordinary_eos_without_special_registration() {
        let tokenizer = Tokenizer::from_file(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/onnx_genai_workflows/realtime/tokenizer.json"),
        )
        .unwrap();
        let grammar = ContentConstraint::text(
            &tokenizer,
            &crate::config::TextGrammar::Regex("hello world <eos>".into()),
            ProjectionKind::Plain,
            "",
            4096,
            vec![3],
            &Default::default(),
        )
        .unwrap();
        let eos = [TokenCandidate {
            token_id: 3,
            text: "<eos>".into(),
            is_eos: true,
        }];
        for (tokens, expected) in [(vec![], false), (vec![4], false), (vec![4, 5], true)] {
            let context = ProcessorContext {
                generated_tokens: tokens,
                ..Default::default()
            };
            assert_eq!(
                grammar.allowed_next_tokens(&context, &eos).unwrap(),
                [expected]
            );
        }
        let plain = ContentConstraint::text(
            &tokenizer,
            &crate::config::TextGrammar::Regex("hello world".into()),
            ProjectionKind::Plain,
            "",
            4096,
            vec![3],
            &Default::default(),
        )
        .unwrap();
        let context = ProcessorContext {
            generated_tokens: vec![4, 5],
            ..Default::default()
        };
        assert!(plain.can_finish(&context).unwrap());
        assert_eq!(
            plain.allowed_next_tokens(&context, &eos).unwrap(),
            [false],
            "an ordinary ending piece would add unwanted text after the completed root"
        );
    }
    #[test]
    fn projected_text_grammars_allow_reasoning_then_constrain_literal_content_and_rewinds() {
        use crate::config::TextGrammar;
        for byte_level in [false, true] {
            let tokenizer = tokenizer(byte_level);
            for grammar in [
                TextGrammar::Regex(r#"\{"n":1\}"#.into()),
                TextGrammar::Lark(r#"start: "{\"n\":1}""#.into()),
            ] {
                let mut options = options(&tokenizer);
                options.constraint = Some(match grammar {
                    TextGrammar::Regex(pattern) => GenerateConstraint::Regex {
                        pattern,
                        prefix: String::new(),
                    },
                    TextGrammar::Lark(grammar) => GenerateConstraint::Lark {
                        grammar,
                        prefix: String::new(),
                    },
                });
                let mut ambiguous = options.clone();
                if let Some(
                    GenerateConstraint::Regex { prefix, .. }
                    | GenerateConstraint::Lark { prefix, .. },
                ) = &mut ambiguous.constraint
                {
                    *prefix = "unowned prefix".to_owned();
                }
                assert!(
                    build_processor_chain(&ambiguous, Some(&tokenizer), false)
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("response_projection")
                );
                let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
                let initial = mask(&chain, &[]);
                assert!(initial[22].is_finite());
                assert!(!initial[23].is_finite());
                assert!(!initial[24].is_finite());
                assert!(initial[25].is_finite());
                assert!(!initial[3].is_finite());
                let after_thought = mask(&chain, &[25]);
                assert!(after_thought[22].is_finite());
                assert!(!after_thought[23].is_finite());
                assert!(mask(&chain, &[25, 22])[3].is_finite());
                assert_eq!(mask(&chain, &[]), initial);
            }
        }
    }

    fn project(
        options: &mut GenerateOptions,
        template: std::sync::Arc<ResponseTemplate>,
        prefix: &str,
        tokenizer: &Tokenizer,
    ) {
        options.response_projection = Some(
            crate::response_template::ResponseProjection::resolve(
                template,
                prefix.to_owned(),
                options.max_constraint_bytes,
                Some(tokenizer),
            )
            .unwrap(),
        );
    }

    fn options(tokenizer: &Tokenizer) -> GenerateOptions {
        GenerateOptions {
            max_constraint_bytes: 4096,
            eos_token_id: Some(3),
            greedy: true,
            response_projection: Some(crate::response_template::ResponseProjection::resolve(ResponseTemplate::compile(&json!({"start_anchor":"A","fields":{
                    "reasoning_content":{"open":"[r]","close":"[/r]"},"content":{"close":"<eos>"}
                }}))
                .unwrap(), "A[r]old".into(), 4096, Some(tokenizer)).unwrap()),
            constraint: Some(GenerateConstraint::JsonSchema { schema: (json!({"type":"object","properties":{"n":{"const":1}},"required":["n"],"additionalProperties":false})).to_string(), prefix: String::new() }),
            ..Default::default()
        }
    }

    fn byte_tokenizer(fallback: bool, overrides: &[(u32, &str)]) -> Tokenizer {
        let mut value: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        let vocab = value["model"]["vocab"].as_object_mut().unwrap();
        for (id, text) in overrides {
            let old = vocab
                .iter()
                .find(|(_, value)| value.as_u64() == Some(u64::from(*id)))
                .unwrap()
                .0
                .clone();
            vocab.remove(&old);
            assert!(
                vocab.insert((*text).into(), json!(id)).is_none(),
                "duplicate test token piece {text:?}"
            );
        }
        value["model"] = json!({"type":"BPE","unk_token":"<unk>","byte_fallback":fallback,
            "vocab":value["model"]["vocab"],"merges":[]});
        value["decoder"] = if fallback {
            json!({"type":"ByteFallback"})
        } else {
            json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false})
        };
        value["added_tokens"] = json!([{"id":3,"content":"<eos>","single_word":false,"lstrip":false,
            "rstrip":false,"normalized":false,"special":true}]);
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        Tokenizer::from_file(path).unwrap()
    }

    fn mask(chain: &crate::ProcessorChain, ids: &[u32]) -> Vec<f32> {
        let mut logits = vec![0.0; 32];
        chain
            .process(
                &mut logits,
                &ProcessorContext {
                    generated_tokens: ids.into(),
                    ..Default::default()
                },
            )
            .unwrap();
        logits
    }

    #[test]
    fn json_decoder_panics_report_token_and_configuration_without_unwinding() {
        let base = byte_tokenizer(true, &[(4, " ")]);
        let mut value: Value =
            serde_json::from_str(&base.inner().to_string(false).unwrap()).unwrap();
        value["decoder"] = json!({"type":"Sequence","decoders":[
            {"type":"Strip","content":" ","start":0,"stop":100},
            {"type":"ByteFallback"}
        ]});
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let tokenizer = Tokenizer::from_file(path).unwrap();
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Json {
                prefix: String::new(),
            }),
            ..Default::default()
        };
        let error = build_processor_chain(&options, Some(&tokenizer), false)
            .err()
            .unwrap();
        let message = format!("{error:#}");
        assert!(
            message.contains("vocabulary token") && message.contains("tokenizer.decoder"),
            "{message}"
        );
        let mut decoder = TokenTextDecoder::default();
        let error = decoder.step(&tokenizer, 4).unwrap_err();
        assert!(format!("{error:#}").contains("token 4"));
        let error = decoder.finish(&tokenizer).unwrap_err();
        assert!(format!("{error:#}").contains("pending terminal token text"));
    }

    #[test]
    fn plain_json_schema_preserves_composed_byte_fallback_decoder_support() {
        let base = byte_tokenizer(
            true,
            &[(4, "<0xC3>"), (5, "<0xA9>"), (22, "▁\""), (23, "\"")],
        );
        let mut value: Value =
            serde_json::from_str(&base.inner().to_string(false).unwrap()).unwrap();
        value["decoder"] = json!({"type":"Sequence","decoders":[
            {"type":"Replace","pattern":{"String":"▁"},"content":" "},
            {"type":"ByteFallback"},{"type":"Fuse"},
            {"type":"Strip","content":" ","start":1,"stop":0}
        ]});
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let tokenizer = Tokenizer::from_file(path).unwrap();
        let schema = json!({"const":"é"});
        let legacy = crate::logits::LlguidanceConstraint::from_hf_tokenizer(
            crate::logits::GrammarConstraintKind::JsonSchema,
            &schema.to_string(),
            tokenizer.inner(),
            32,
            Some(3),
        )
        .unwrap();
        let candidates = (0..32)
            .map(|token_id| TokenCandidate {
                token_id,
                text: String::new(),
                is_eos: token_id == 3,
            })
            .collect::<Vec<_>>();
        assert!(
            legacy
                .allowed_next_tokens(
                    &ProcessorContext {
                        generated_tokens: vec![23],
                        ..Default::default()
                    },
                    &candidates
                )
                .unwrap()[4]
        );
        let options = GenerateOptions {
            eos_token_id: Some(3),
            constraint: Some(GenerateConstraint::JsonSchema {
                schema: schema.to_string(),
                prefix: String::new(),
            }),
            ..Default::default()
        };
        let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
        for opening in [22u32, 23] {
            assert!(mask(&chain, &[])[opening as usize].is_finite());
            assert!(mask(&chain, &[opening])[4].is_finite());
            assert!(mask(&chain, &[opening, 4])[5].is_finite());
            assert!(mask(&chain, &[opening, 4, 5])[23].is_finite());
            assert!(mask(&chain, &[opening, 4, 5, 23])[3].is_finite());
            assert_eq!(tokenizer.decode(&[opening, 4, 5, 23, 3]).unwrap(), "\"é\"");
        }
        let declared_options = GenerateOptions {
            eos_token_id: Some(3),
            response_projection: Some(
                crate::response_template::ResponseProjection::resolve(
                    ResponseTemplate::compile(&json!({"start_anchor":"A","fields":{
                        "content":{"open":"<c>","close":"<eos>"}
                    }}))
                    .unwrap(),
                    "A<c>".into(),
                    4096,
                    Some(&tokenizer),
                )
                .unwrap(),
            ),
            constraint: Some(GenerateConstraint::JsonSchema {
                schema: (schema.clone()).to_string(),
                prefix: String::new(),
            }),
            ..Default::default()
        };
        let chain = build_processor_chain(&declared_options, Some(&tokenizer), false).unwrap();
        for opening in [22u32, 23] {
            assert!(mask(&chain, &[])[opening as usize].is_finite());
            assert!(mask(&chain, &[opening])[4].is_finite());
            assert!(mask(&chain, &[opening, 4])[5].is_finite());
            assert!(mask(&chain, &[opening, 4, 5])[23].is_finite());
            assert!(!mask(&chain, &[opening, 4, 5])[3].is_finite());
            assert!(mask(&chain, &[opening, 4, 5, 23])[3].is_finite());
        }
    }

    #[test]
    fn plain_json_vocabulary_masks_match_semantic_projection_across_rewinds_and_escapes() {
        let tokenizer = byte_tokenizer(
            true,
            &[
                (4, "\\"),
                (5, "u"),
                (6, "0"),
                (7, "4"),
                (8, "1"),
                (9, "2"),
                (16, "\""),
                (17, "<0xFF>"),
                (18, "<0xC0>"),
                (19, "<0x80>"),
                (20, "<0xED>"),
                (21, "<0xA0>"),
                (22, "{"),
                (23, "}"),
                (24, "\"k\":"),
                (25, "123"),
                (26, " \n"),
                (27, "<0xC3>"),
                (28, "<0xA9>"),
                (29, "<0x31>"),
            ],
        );
        let fast = ContentConstraint::json_schema(
            &tokenizer,
            &json!(true),
            ProjectionKind::Plain,
            "",
            4096,
            vec![3],
            &Default::default(),
        )
        .unwrap();
        let mut reference = ContentConstraint::json_schema(
            &tokenizer,
            &json!(true),
            ProjectionKind::Plain,
            "",
            4096,
            vec![3],
            &Default::default(),
        )
        .unwrap();
        assert!(fast.plain_vocabulary.is_some());
        reference.plain_vocabulary = None;
        let candidates = (0..32)
            .map(|token_id| TokenCandidate {
                token_id,
                text: String::new(),
                is_eos: token_id == 3,
            })
            .collect::<Vec<_>>();
        for history in [
            vec![],
            vec![22],
            vec![22, 24],
            vec![22, 24, 25],
            vec![22, 24, 25, 23],
            vec![16],
            vec![16, 27],
            vec![16, 27, 28],
            vec![16, 20],
            vec![16, 4, 5, 6, 6, 7],
            vec![16, 4, 5, 6, 6, 7, 8, 16],
            vec![16],
            vec![],
        ] {
            let context = ProcessorContext {
                generated_tokens: history.clone(),
                ..Default::default()
            };
            assert_eq!(
                fast.allowed_next_tokens(&context, &candidates).unwrap(),
                reference
                    .allowed_next_tokens(&context, &candidates)
                    .unwrap(),
                "{history:?}"
            );
        }
    }

    #[test]
    fn plain_json_schema_processor_accepts_split_escapes_and_checks_ordinary_eos_text() {
        let tokenizer = byte_tokenizer(
            true,
            &[
                (4, "\\"),
                (5, "u"),
                (6, "0"),
                (7, "4"),
                (8, "1"),
                (9, "2"),
                (16, "\""),
                (22, "\"A"),
            ],
        );
        let mut options = GenerateOptions {
            eos_token_id: Some(3),
            constraint: Some(GenerateConstraint::JsonSchema {
                schema: json!({"const":"A"}).to_string(),
                prefix: "\"".into(),
            }),
            ..Default::default()
        };
        let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
        assert!(mask(&chain, &[])[4].is_finite());
        let pending = mask(&chain, &[4, 5, 6, 6, 7]);
        assert!(pending[8].is_finite() && !pending[9].is_finite() && !pending[3].is_finite());
        assert!(mask(&chain, &[4, 5, 6, 6, 7, 8])[16].is_finite());
        assert!(mask(&chain, &[4, 5, 6, 6, 7, 8, 16])[3].is_finite());
        assert_eq!(pending, mask(&chain, &[4, 5, 6, 6, 7]));

        options.eos_token_id = Some(16);
        if let Some(GenerateConstraint::JsonSchema { prefix, .. }) = &mut options.constraint {
            *prefix = "\"A".into();
        }
        let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
        assert!(
            mask(&chain, &[])[16].is_finite(),
            "ordinary EOS quote finishes the actual JSON text"
        );
        if let Some(GenerateConstraint::JsonSchema { prefix, .. }) = &mut options.constraint {
            *prefix = "\"A\"".into();
        }
        let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
        assert!(
            !mask(&chain, &[])[16].is_finite(),
            "EOS roles do not erase ordinary quote bytes"
        );
    }

    #[test]
    fn response_json_processor_checks_split_escapes_in_generation_and_existing_prefill() {
        let tokenizer = byte_tokenizer(
            true,
            &[
                (4, "\\"),
                (5, "u"),
                (6, "0"),
                (7, "4"),
                (8, "1"),
                (9, "2"),
                (16, "\""),
                (22, "[/r]\""),
                (23, "\\ud800\\u0000"),
            ],
        );
        let mut options = options(&tokenizer);
        if let Some(GenerateConstraint::JsonSchema { schema, .. }) = &mut options.constraint {
            *schema = json!({"const":"A"}).to_string();
        }
        let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
        for history in [
            vec![22],
            vec![22, 4],
            vec![22, 4, 5],
            vec![22, 4, 5, 6],
            vec![22, 4, 5, 6, 6],
            vec![22, 4, 5, 6, 6, 7],
        ] {
            let logits = mask(&chain, &history);
            assert!(
                !logits[3].is_finite() && !logits[16].is_finite() && !logits[23].is_finite(),
                "{history:?}"
            );
        }
        let pending = mask(&chain, &[22, 4, 5, 6, 6, 7]);
        assert!(pending[8].is_finite() && !pending[9].is_finite());
        assert!(mask(&chain, &[22, 4, 5, 6, 6, 7, 8])[16].is_finite());
        assert!(mask(&chain, &[22, 4, 5, 6, 6, 7, 8, 16])[3].is_finite());
        // Replacing the generated history must recover the pending scalar domain.
        assert_eq!(pending, mask(&chain, &[22, 4, 5, 6, 6, 7]));

        let spelling = r#""\u0041""#;
        let ids = [16, 4, 5, 6, 6, 7, 8, 16];
        for split in 0..=spelling.len() {
            let template = options
                .response_projection
                .as_ref()
                .unwrap()
                .template()
                .clone();
            project(
                &mut options,
                template,
                &format!("A[r]old[/r]{}", &spelling[..split]),
                &tokenizer,
            );
            let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
            let mut history = Vec::new();
            for &id in &ids[split..] {
                assert!(
                    mask(&chain, &history)[id as usize].is_finite(),
                    "split {split}, history {history:?}, next {id}"
                );
                history.push(id);
            }
            assert!(mask(&chain, &history)[3].is_finite(), "split {split}");
        }
    }

    #[test]
    fn response_json_processor_accepts_split_unicode_data_without_allowing_invalid_bytes() {
        let tokenizer = byte_tokenizer(
            false,
            &[(4, "Ã"), (5, "©"), (16, "ÿ"), (22, "[/r]\""), (23, "\"")],
        );
        let mut options = options(&tokenizer);
        if let Some(GenerateConstraint::JsonSchema { schema, .. }) = &mut options.constraint {
            *schema = json!({"const":"é"}).to_string();
        }
        let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
        let first = mask(&chain, &[22]);
        assert!(first[4].is_finite());
        assert!(
            !first[5].is_finite()
                && !first[16].is_finite()
                && !first[3].is_finite()
                && !first[23].is_finite()
        );
        let pending = mask(&chain, &[22, 4]);
        assert!(pending[5].is_finite());
        assert!(!pending[3].is_finite() && !pending[23].is_finite());
        assert!(mask(&chain, &[22, 4, 5])[23].is_finite());
        assert!(mask(&chain, &[22, 4, 5, 23])[3].is_finite());
    }

    #[test]
    fn response_json_processor_eos_can_finish_a_split_unicode_framing_boundary() {
        let tokenizer = byte_tokenizer(
            true,
            &[
                (4, "<0xE7>"),
                (5, "<0xB5>"),
                (16, "<0x82>"),
                (22, "{\"n\":1}"),
            ],
        );
        let mut options = options(&tokenizer);
        options.eos_token_id = Some(16);
        {
            let template = ResponseTemplate::compile(&json!({"start_anchor":"A","fields":{
                "reasoning_content":{"open":"[r]","close":"[/r]"},"content":{"close":"終"}
            }}))
            .unwrap();
            let prefix: String = "A[r]old[/r]".into();
            project(&mut options, template, &prefix, &tokenizer);
        }
        let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
        assert!(mask(&chain, &[22])[4].is_finite());
        assert!(!mask(&chain, &[22])[16].is_finite());
        assert!(mask(&chain, &[22, 4])[5].is_finite());
        assert!(mask(&chain, &[22, 4, 5])[16].is_finite());
    }

    #[test]
    fn response_json_processor_checks_provisional_whitespace_inside_string_constants() {
        let tokenizer = byte_tokenizer(true, &[(6, " "), (22, "[/r]\"a"), (23, "\"")]);
        for (value, space_allowed) in [("a", false), ("a ", true)] {
            let mut options = options(&tokenizer);
            if let Some(GenerateConstraint::JsonSchema { schema, .. }) = &mut options.constraint {
                *schema = json!({"const":value}).to_string();
            }
            let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
            assert_eq!(mask(&chain, &[22])[6].is_finite(), space_allowed);
            if space_allowed {
                assert!(mask(&chain, &[22, 6])[23].is_finite());
                assert!(mask(&chain, &[22, 6, 23])[3].is_finite());
            }
        }
    }

    #[test]
    fn response_json_processor_resolves_overlapping_closers_before_partial_unicode() {
        let tokenizer = byte_tokenizer(false, &[(4, "Ã"), (5, "©"), (22, "{\"n\":1}"), (23, "X")]);
        for regex in [false, true] {
            let mut options = options(&tokenizer);
            {
                let field = if regex {
                    json!({"open_pattern":"<c>","close_pattern":"X終|X"})
                } else {
                    json!({"open":"<c>","close":["X","X終"]})
                };
                let template = ResponseTemplate::compile(&json!({"start_anchor":"A","fields":{
                    "content":field
                }}))
                .unwrap();
                let prefix: String = "A<c>".into();
                project(&mut options, template, &prefix, &tokenizer);
            }
            let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
            // C3 cannot extend X終, so the complete X closes content before it.
            assert!(mask(&chain, &[22, 23])[4].is_finite());
            assert!(mask(&chain, &[22, 23, 4])[5].is_finite());
            assert!(mask(&chain, &[22, 23, 4, 5])[3].is_finite());
        }
    }

    #[test]
    fn response_json_processor_budget_rejections_do_not_abort_other_candidates() {
        let tokenizer = tokenizer(false);
        let mut options = options(&tokenizer);
        {
            let template = ResponseTemplate::compile(&json!({"start_anchor":"A","fields":{
                "reasoning_content":{"open":"[r]","close":"[/r]"},"content":{"close":"<eos>"}
            },"defaults":{"content":format!("{{\"extra\":\"{}\"}}","x".repeat(200))}}))
            .unwrap();
            let schema = json!({"type":"object"}).to_string();
            let prefix = options
                .response_projection
                .as_ref()
                .unwrap()
                .prefix()
                .to_owned();
            project(&mut options, template, &prefix, &tokenizer);
            options.constraint = Some(GenerateConstraint::JsonSchema {
                schema,
                prefix: String::new(),
            });
        }
        options.max_constraint_bytes = 64;
        let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
        let initial = mask(&chain, &[]);
        assert!(
            !initial[3].is_finite(),
            "oversized default is an EOS rejection, not a backend failure"
        );
        assert!(initial[22].is_finite());
        assert!(mask(&chain, &[22])[3].is_finite());
    }

    #[test]
    fn response_json_processor_masks_projected_content_and_actual_eos() {
        for byte_level in [false, true] {
            let tokenizer = tokenizer(byte_level);
            for regex in [false, true] {
                let mut options = options(&tokenizer);
                if regex {
                    let template = ResponseTemplate::compile(&json!({"start_anchor":"A","fields":{
                        "reasoning_content":{"open_pattern":r"\[r\]","close_pattern":r"\[/r\]\s*"},
                        "content":{"close_pattern":"<eos>"}
                    }}))
                    .unwrap();
                    let prefix = options
                        .response_projection
                        .as_ref()
                        .unwrap()
                        .prefix()
                        .to_owned();
                    project(&mut options, template, &prefix, &tokenizer);
                }
                let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
                let mut logits = vec![0.0; 34];
                chain
                    .process(&mut logits, &ProcessorContext::default())
                    .unwrap();
                assert!(logits[22].is_finite() && logits[25].is_finite());
                assert!(
                    !logits[23].is_finite() && !logits[24].is_finite() && !logits[3].is_finite()
                );
                assert!(!logits[32].is_finite() && !logits[33].is_finite());
                let context = ProcessorContext {
                    generated_tokens: vec![22],
                    ..Default::default()
                };
                let mut logits = vec![0.0; 34];
                chain.process(&mut logits, &context).unwrap();
                assert!(
                    logits[3].is_finite(),
                    "EOS must project its declared closing marker"
                );
            }
        }
    }

    #[test]
    fn response_json_processor_replays_history_replacements_and_recovers_rejections() {
        let tokenizer = tokenizer(false);
        let options = options(&tokenizer);
        let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
        for ids in [vec![], vec![25], vec![22], vec![25], vec![], vec![22]] {
            let context = ProcessorContext {
                generated_tokens: ids,
                ..Default::default()
            };
            let mut actual = vec![0.0; 32];
            let mut expected = actual.clone();
            chain.process(&mut actual, &context).unwrap();
            build_processor_chain(&options, Some(&tokenizer), false)
                .unwrap()
                .process(&mut expected, &context)
                .unwrap();
            assert_eq!(actual, expected);
        }
        let error = chain
            .process(
                &mut [0.0; 32],
                &ProcessorContext {
                    generated_tokens: vec![23],
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("history token 23 at generated offset 0"),
            "{error:#}"
        );
        let mut logits = vec![0.0; 32];
        chain
            .process(
                &mut logits,
                &ProcessorContext {
                    generated_tokens: vec![22],
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(logits[3].is_finite());
    }
}
