//! Model-token constraints over the shared declared response projection.
use crate::json_cursor::{JsonByteBudgetExceeded, JsonByteCursor, JsonCandidate};
use crate::logits::{Constraint, ProcessorContext, TokenCandidate, TokenId, canonical_json_prefix};
use crate::response_template::{
    IncompleteResponseUtf8, MissingResponseField, NonMonotonicProjection, PrefixOutput,
    ResponseBufferExceeded, ResponseParser, ResponseTemplate, projection_extension,
};
use crate::tokenization::{NonMonotonicText, SpecialTokenHandling, TokenTextDecoder};
use anyhow::{Context, ensure};
use onnx_genai_ort::Tokenizer;
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};
use tokenizers::decoders::DecoderWrapper;

#[derive(Debug, thiserror::Error)]
#[error("response JSON candidate evaluation failed and state recovery also failed: {recovery:#}")]
struct ResponseConstraintRecovery {
    #[source]
    operation: anyhow::Error,
    recovery: anyhow::Error,
}

enum Codec {
    Bytes(Vec<Vec<u8>>),
    Canonical(Box<Tokenizer>),
}

struct State {
    parser: ResponseParser,
    decoder: TokenTextDecoder,
    json: JsonByteCursor,
    history: Vec<TokenId>,
    raw: String,
    canonical: String,
    ended: bool,
}

pub(crate) struct ResponseJsonConstraint {
    template: Arc<ResponseTemplate>,
    prefix: String,
    max_bytes: usize,
    endings: HashSet<TokenId>,
    vocabulary: HashSet<TokenId>,
    codec: Codec,
    state: Mutex<State>,
}

impl ResponseJsonConstraint {
    pub(crate) fn new(
        tokenizer: &Tokenizer,
        schema: &serde_json::Value,
        template: Arc<ResponseTemplate>,
        prefix: &str,
        max_bytes: usize,
        endings: Vec<TokenId>,
    ) -> anyhow::Result<Self> {
        template.require_json_boundaries()?;
        let vocabulary = tokenizer
            .inner()
            .get_vocab(true)
            .values()
            .copied()
            .collect::<HashSet<_>>();
        let codec = match tokenizer.inner().get_decoder() {
            Some(DecoderWrapper::ByteLevel(_) | DecoderWrapper::ByteFallback(_)) => {
                let size = vocabulary.len();
                ensure!(
                    vocabulary
                        .iter()
                        .all(|id| usize::try_from(*id).is_ok_and(|id| id < size)),
                    "projected JSON requires dense byte vocabulary IDs; repair tokenizer.json before enabling this sampling policy"
                );
                let ids = (0..size)
                    .map(u32::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                Codec::Bytes(
                    crate::tokenization::token_pieces(tokenizer, &ids)
                        .context("resolve lossless model pieces for response JSON sampling")?,
                )
            }
            decoder => {
                ensure!(
                    decoder.is_none_or(|decoder| !crate::tokenization::has_byte_decoder(decoder)),
                    "projected JSON cannot yet prove contextual byte-decoder sequences; this combination requires a resolved byte-concatenation decoder contract"
                );
                Codec::Canonical(Box::new(tokenizer.clone()))
            }
        };
        let parser =
            ResponseParser::new(template.clone(), prefix, PrefixOutput::Include, max_bytes)?;
        let raw = parser.json_content(false);
        let canonical = canonical_json_prefix(&raw)?.into_owned();
        let mut json = JsonByteCursor::new(schema, max_bytes)?;
        json.commit(canonical.as_bytes()).context("rendered assistant prefill is not a viable projected JSON prefix; repair the selected content prefix or schema")?;
        Ok(Self {
            template,
            prefix: prefix.into(),
            max_bytes,
            endings: endings.into_iter().collect(),
            vocabulary,
            codec,
            state: Mutex::new(State {
                parser,
                decoder: TokenTextDecoder::new(SpecialTokenHandling::Preserve),
                json,
                history: Vec::new(),
                raw,
                canonical,
                ended: false,
            }),
        })
    }

    fn reset(&self, state: &mut State) -> anyhow::Result<()> {
        state.parser = ResponseParser::new(
            self.template.clone(),
            &self.prefix,
            PrefixOutput::Include,
            self.max_bytes,
        )?;
        state.decoder = TokenTextDecoder::new(SpecialTokenHandling::Preserve);
        state.history.clear();
        state.raw = state.parser.json_content(false);
        state.canonical = canonical_json_prefix(&state.raw)?.into_owned();
        state.json.reset();
        state.json.commit(state.canonical.as_bytes())?;
        state.ended = false;
        Ok(())
    }

    fn inspect_json(
        view: &ResponseParser,
        json: &mut JsonByteCursor,
        base_raw: &str,
        base_canonical: &str,
        terminal: bool,
    ) -> anyhow::Result<bool> {
        let published = view.json_content(terminal);
        if projection_extension(&published, base_raw, terminal, "content")?.is_none() {
            return Ok(true);
        }
        let grammar = if terminal {
            published.clone()
        } else {
            view.grammar_json_content()
        };
        let canonical = canonical_json_prefix(&grammar)?;
        let pending = view.pending_json_bytes().unwrap_or_default();
        let combined;
        let bytes = if pending.is_empty() {
            canonical.as_bytes()
        } else {
            combined = [canonical.as_bytes(), pending].concat();
            &combined
        };
        let status = if let Some(suffix) = bytes.strip_prefix(base_canonical.as_bytes()) {
            json.preview(suffix)?
        } else {
            json.preview_root(bytes)?
        };
        if let JsonCandidate::Viable { complete } = status
            && (!terminal || complete)
        {
            return Ok(true);
        }
        // End trimming can hide whitespace only after a complete value. An
        // incomplete JSON string must validate its provisional trailing data.
        if pending.is_empty() || view.pending_can_be_trimmed() {
            let trimmed = canonical_json_prefix(&published)?;
            let status = if let Some(suffix) = trimmed.strip_prefix(base_canonical) {
                json.preview(suffix.as_bytes())?
            } else {
                json.preview_root(trimmed.as_bytes())?
            };
            return Ok(matches!(status, JsonCandidate::Viable { complete: true }));
        }
        Ok(false)
    }

    fn inspect(&self, state: &mut State, id: TokenId) -> anyhow::Result<bool> {
        if !self.vocabulary.contains(&id) || state.ended {
            return Ok(false);
        }
        let terminal = self.endings.contains(&id);
        let mut fork = state.decoder.clone();
        let text;
        let bytes = match &self.codec {
            Codec::Bytes(pieces) => pieces
                .get(id as usize)
                .context("response JSON token has no declared byte piece")?
                .as_slice(),
            Codec::Canonical(tokenizer) => {
                text = {
                    let mut text = fork.step(tokenizer, id)?;
                    if terminal {
                        text.push_str(&fork.finish(tokenizer)?);
                    } else {
                        let mut materialized = fork.clone();
                        text.push_str(&materialized.finish(tokenizer)?);
                    }
                    text
                };
                text.as_bytes()
            }
        };
        let State {
            parser,
            json,
            raw,
            canonical,
            ..
        } = state;
        parser.with_preview(bytes, terminal, |view| {
            Self::inspect_json(view, json, raw, canonical, terminal)
        })
    }

    fn token_allowed(&self, state: &mut State, id: TokenId) -> anyhow::Result<bool> {
        match self.inspect(state, id) {
            Err(error)
                if error.is::<std::str::Utf8Error>()
                    || error.is::<IncompleteResponseUtf8>()
                    || error.is::<MissingResponseField>()
                    || error.is::<ResponseBufferExceeded>()
                    || error.is::<JsonByteBudgetExceeded>()
                    || error.is::<NonMonotonicText>()
                    || error.is::<NonMonotonicProjection>() =>
            {
                Ok(false)
            }
            result => result,
        }
    }

    fn advance(&self, state: &mut State, id: TokenId) -> anyhow::Result<()> {
        match &self.codec {
            Codec::Bytes(pieces) => state.parser.commit_unpublished(&pieces[id as usize])?,
            Codec::Canonical(tokenizer) => {
                let mut text = state.decoder.step(tokenizer, id)?;
                if self.endings.contains(&id) {
                    text.push_str(&state.decoder.finish(tokenizer)?);
                }
                state.parser.commit_unpublished(text.as_bytes())?;
            }
        }
        let raw = if self.endings.contains(&id) {
            state
                .parser
                .with_preview(b"", true, |view| Ok(view.json_content(true)))?
        } else {
            state.parser.json_content(false)
        };
        if projection_extension(&raw, &state.raw, self.endings.contains(&id), "content")?.is_some()
        {
            let canonical = canonical_json_prefix(&raw)?.into_owned();
            if let Some(suffix) = canonical.strip_prefix(&state.canonical) {
                state.json.commit(suffix.as_bytes())?;
            } else {
                state.json.replace_prefix(canonical.as_bytes())?;
            }
            state.raw = raw;
            state.canonical = canonical;
        }
        state.history.push(id);
        state.ended = self.endings.contains(&id);
        Ok(())
    }
}

impl Constraint for ResponseJsonConstraint {
    fn allowed_next_tokens(
        &self,
        context: &ProcessorContext,
        candidates: &[TokenCandidate],
    ) -> anyhow::Result<Vec<bool>> {
        let mut state = self.state.lock().map_err(|_|anyhow::anyhow!(
            "response JSON constraint state is poisoned; restart this generation with a fresh constraint"))?;
        if !context.generated_tokens.starts_with(&state.history) {
            self.reset(&mut state)?;
        }
        let result = (|| {
            for &id in &context.generated_tokens[state.history.len()..] {
                ensure!(
                    self.token_allowed(&mut state, id)?,
                    "response JSON history token {id} at generated offset {} is rejected; supply history admitted by the same tokenizer, template and schema",
                    state.history.len()
                );
                self.advance(&mut state, id)?;
            }
            candidates
                .iter()
                .map(|candidate| self.token_allowed(&mut state, candidate.token_id))
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
        "response_json_schema"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{GenerateConstraint, GenerateOptions};
    use crate::processors::build_processor_chain;
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

    fn options() -> GenerateOptions {
        GenerateOptions {
            eos_token_id: Some(3),
            greedy: true,
            constraint: Some(GenerateConstraint::ResponseJson {
                schema: json!({"type":"object","properties":{"n":{"const":1}},"required":["n"],"additionalProperties":false}),
                template: ResponseTemplate::compile(&json!({"start_anchor":"A","fields":{
                    "reasoning_content":{"open":"[r]","close":"[/r]"},"content":{"close":"<eos>"}
                }}))
                .unwrap(),
                prefix: "A[r]old".into(),
                max_bytes: 4096,
            }),
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
            vocab.insert((*text).into(), json!(id));
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
    fn response_json_processor_accepts_split_unicode_data_without_allowing_invalid_bytes() {
        let tokenizer = byte_tokenizer(
            false,
            &[(4, "Ã"), (5, "©"), (16, "ÿ"), (22, "[/r]\""), (23, "\"")],
        );
        let mut options = options();
        if let Some(GenerateConstraint::ResponseJson { schema, .. }) = &mut options.constraint {
            *schema = json!({"const":"é"});
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
        let mut options = options();
        options.eos_token_id = Some(16);
        if let Some(GenerateConstraint::ResponseJson {
            template, prefix, ..
        }) = &mut options.constraint
        {
            *template = ResponseTemplate::compile(&json!({"start_anchor":"A","fields":{
                "reasoning_content":{"open":"[r]","close":"[/r]"},"content":{"close":"終"}
            }}))
            .unwrap();
            *prefix = "A[r]old[/r]".into();
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
            let mut options = options();
            if let Some(GenerateConstraint::ResponseJson { schema, .. }) = &mut options.constraint {
                *schema = json!({"const":value});
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
    fn response_json_processor_resolves_overlapping_literal_closers_before_partial_unicode() {
        let tokenizer = byte_tokenizer(false, &[(4, "Ã"), (5, "©"), (22, "{\"n\":1}"), (23, "X")]);
        let mut options = options();
        if let Some(GenerateConstraint::ResponseJson {
            template, prefix, ..
        }) = &mut options.constraint
        {
            *template = ResponseTemplate::compile(&json!({"start_anchor":"A","fields":{
                "content":{"open":"<c>","close":["X","X終"]}
            }}))
            .unwrap();
            *prefix = "A<c>".into();
        }
        let chain = build_processor_chain(&options, Some(&tokenizer), false).unwrap();
        // C3 cannot extend X終, so the complete X closes content before it.
        assert!(mask(&chain, &[22, 23])[4].is_finite());
        assert!(mask(&chain, &[22, 23, 4])[5].is_finite());
        assert!(mask(&chain, &[22, 23, 4, 5])[3].is_finite());
    }

    #[test]
    fn response_json_processor_budget_rejections_do_not_abort_other_candidates() {
        let tokenizer = tokenizer(false);
        let mut options = options();
        if let Some(GenerateConstraint::ResponseJson {
            template,
            schema,
            max_bytes,
            ..
        }) = &mut options.constraint
        {
            *template = ResponseTemplate::compile(&json!({"start_anchor":"A","fields":{
                "reasoning_content":{"open":"[r]","close":"[/r]"},"content":{"close":"<eos>"}
            },"defaults":{"content":format!("{{\"extra\":\"{}\"}}","x".repeat(200))}}))
            .unwrap();
            *schema = json!({"type":"object"});
            *max_bytes = 64;
        }
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
            let chain = build_processor_chain(&options(), Some(&tokenizer), false).unwrap();
            let mut logits = vec![0.0; 34];
            chain
                .process(&mut logits, &ProcessorContext::default())
                .unwrap();
            assert!(logits[22].is_finite() && logits[25].is_finite());
            assert!(!logits[23].is_finite() && !logits[24].is_finite() && !logits[3].is_finite());
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

    #[test]
    fn response_json_processor_replays_history_replacements_and_recovers_rejections() {
        let tokenizer = tokenizer(false);
        let options = options();
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
