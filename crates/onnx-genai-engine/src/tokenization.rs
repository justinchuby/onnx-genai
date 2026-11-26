//! Lossless tokenizer pieces for protocol consumers and byte-oriented tooling.
use anyhow::{Context, bail, ensure};
use onnx_genai_llguidance::toktrie::TokTrie;
use onnx_genai_ort::Tokenizer;
use std::{collections::BTreeSet, sync::Arc};
use tokenizers::Decoder as _;
use tokenizers::decoders::DecoderWrapper;

#[derive(Debug, thiserror::Error)]
#[error(
    "incremental token decoding changed an already published prefix; use buffered output or repair the package tokenizer decoder"
)]
pub(crate) struct NonMonotonicText;

/// Whether protocol markers participate in incremental output text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpecialTokenHandling {
    #[default]
    Skip,
    Preserve,
}

/// Request-local incremental detokenization state. Retains the decoder's
/// necessary lookbehind rather than repeatedly decoding the whole generation.
/// Cloning forks that lookbehind and token policy for isolated candidate decoding.
#[derive(Default, Clone)]
pub struct TokenTextDecoder {
    special_tokens: SpecialTokenHandling,
    preserved_tokens: Arc<BTreeSet<u32>>,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}
impl TokenTextDecoder {
    pub fn new(special_tokens: SpecialTokenHandling) -> Self {
        Self {
            special_tokens,
            ..Default::default()
        }
    }
    /// Preserve selected special IDs without changing the shared tokenizer.
    pub fn with_preserved_tokens(preserved: &BTreeSet<u32>) -> Self {
        Self {
            preserved_tokens: Arc::new(preserved.clone()),
            ..Default::default()
        }
    }
    pub fn step(&mut self, tokenizer: &Tokenizer, id: u32) -> anyhow::Result<String> {
        if self.special_tokens == SpecialTokenHandling::Skip
            && !self.preserved_tokens.contains(&id)
            && tokenizer
                .inner()
                .get_added_vocabulary()
                .get_added_tokens_decoder()
                .get(&id)
                .is_some_and(|token| {
                    tokenizer
                        .inner()
                        .get_added_vocabulary()
                        .is_special_token(&token.content)
                })
        {
            return Ok(String::new());
        }

        onnx_genai_llguidance::panic_utils::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tokenizers::tokenizer::step_decode_stream(
                tokenizer.inner(), vec![id], false, &mut self.ids, &mut self.prefix, &mut self.prefix_index,
            ).map(|text| text.unwrap_or_default()).map_err(|error| {
                if error.is::<tokenizers::tokenizer::DecodeStreamError>() {
                    anyhow::Error::new(NonMonotonicText)
                } else {
                    anyhow::Error::from_boxed(error)
                }
            })
        })).with_context(|| format!("incremental decoding of token {id} failed; repair tokenizer.decoder or its token vocabulary"))
    }
    /// Emit pending bytes at the terminal boundary, including the tokenizer's
    /// replacement text for an incomplete final UTF-8 sequence.
    pub fn finish(&mut self, tokenizer: &Tokenizer) -> anyhow::Result<String> {
        let decoded =
            onnx_genai_llguidance::panic_utils::catch_unwind(std::panic::AssertUnwindSafe(|| {
                tokenizer
                    .decode_with_special_tokens(&self.ids)
                    .map_err(anyhow::Error::from)
            }))
            .context(
                "decode pending terminal token text; repair tokenizer.decoder if decoding panicked",
            )?;
        let tail = decoded
            .strip_prefix(&self.prefix)
            .ok_or(NonMonotonicText)?
            .to_owned();
        self.prefix = decoded;
        self.prefix_index = self.ids.len();
        Ok(tail)
    }
}

/// Decode output while retaining only the requested special-token IDs.
pub fn decode_output(
    tokenizer: &Tokenizer,
    ids: &[u32],
    preserved: &BTreeSet<u32>,
) -> anyhow::Result<String> {
    onnx_genai_llguidance::panic_utils::catch_unwind(std::panic::AssertUnwindSafe(|| {
    if preserved.is_empty() {
        return tokenizer.decode(ids).map_err(Into::into);
    }
    let specials = tokenizer
        .inner()
        .get_added_vocabulary()
        .get_added_tokens_decoder();
    let visible = ids
        .iter()
        .copied()
        .filter(|id| {
            preserved.contains(id)
                || !specials.get(id).is_some_and(|token| {
                    tokenizer
                        .inner()
                        .get_added_vocabulary()
                        .is_special_token(&token.content)
                })
        })
        .collect::<Vec<_>>();
    tokenizer
        .decode_with_special_tokens(&visible)
        .map_err(Into::into)
    })).context("decode generated token text; repair tokenizer.decoder or its vocabulary if decoding failed")
}

/// Decode final text with request-owned indentation clipping. This replay is
/// independent of the finish label so an EOS/time override keeps the boundary.
pub(crate) fn decode_generation_output(
    tokenizer: &Tokenizer,
    ids: &[u32],
    options: &crate::GenerateOptions,
) -> anyhow::Result<String> {
    let preserved = options.effective_preserved_tokens();
    if options.minimum_indent == 0 {
        return decode_output(tokenizer, ids, &preserved);
    }
    let mut decoder = TokenTextDecoder::with_preserved_tokens(&preserved);
    let mut projection = crate::indentation::IndentationProjection::new(options.minimum_indent);
    let mut text = String::new();
    let mut cutoff = None;
    for (index, &id) in ids.iter().enumerate() {
        text.push_str(&decoder.step(tokenizer, id)?);
        if index + 1 == ids.len() {
            text.push_str(&decoder.finish(tokenizer)?);
        }
        if let Some(boundary) = projection.observe(&text, &options.stop_sequences) {
            cutoff.get_or_insert(boundary);
        }
    }
    if let Some(cutoff) = cutoff {
        anyhow::ensure!(
            text.is_char_boundary(cutoff),
            "indentation output boundary {cutoff} is not a UTF-8 boundary; repair output observer/tokenizer consistency"
        );
        text.truncate(cutoff);
    }
    Ok(text)
}

/// Decode committed token boundaries using the same request policy as live output.
pub fn token_texts(
    tokenizer: Option<&Tokenizer>,
    ids: &[u32],
    preserved: &BTreeSet<u32>,
) -> anyhow::Result<Vec<String>> {
    let Some(tokenizer) = tokenizer else {
        return Ok(vec![String::new(); ids.len()]);
    };
    // Staged output knows the final text before publishing any callback. A
    // decoder may revise an earlier prefix (e.g. an invalid final byte run);
    // defer that batch's text to its last token instead of publishing a prefix
    // that the immutable final result would later have to retract.
    let final_text = decode_output(tokenizer, ids, preserved)?;
    let attempt = || -> anyhow::Result<Vec<String>> {
        let mut decoder = TokenTextDecoder::with_preserved_tokens(preserved);
        let mut offset = 0;
        let mut pieces = Vec::with_capacity(ids.len());
        for id in ids {
            let piece = decoder.step(tokenizer, *id)?;
            ensure!(
                final_text
                    .get(offset..)
                    .is_some_and(|rest| rest.starts_with(&piece)),
                NonMonotonicText
            );
            offset += piece.len();
            pieces.push(piece);
        }
        if let Some(last) = pieces.last_mut() {
            last.push_str(&decoder.finish(tokenizer)?);
        }
        ensure!(pieces.concat() == final_text, NonMonotonicText);
        Ok(pieces)
    };
    match attempt() {
        Ok(pieces) => Ok(pieces),
        Err(error) if error.is::<NonMonotonicText>() => {
            let mut pieces = vec![String::new(); ids.len()];
            if let Some(last) = pieces.last_mut() {
                *last = final_text;
            }
            Ok(pieces)
        }
        Err(error) => Err(error),
    }
}

pub(crate) fn has_byte_decoder(decoder: &DecoderWrapper) -> bool {
    match decoder {
        DecoderWrapper::ByteFallback(_) | DecoderWrapper::ByteLevel(_) => true,
        DecoderWrapper::Sequence(sequence) => sequence.get_decoders().iter().any(has_byte_decoder),
        _ => false,
    }
}

// A vocabulary piece has no first-token context. Metaspace's initial-space
// stripping belongs to full-text decoding, not byte-prefix matching.
fn context_free_decoder(decoder: &DecoderWrapper) -> Option<DecoderWrapper> {
    use tokenizers::pre_tokenizers::metaspace::PrependScheme;
    match decoder {
        DecoderWrapper::Metaspace(space) if space.get_prepend_scheme() != PrependScheme::Never => {
            let mut space = space.clone();
            space.set_prepend_scheme(PrependScheme::Never);
            Some(DecoderWrapper::Metaspace(space))
        }
        DecoderWrapper::Sequence(sequence) => {
            let mut changed = false;
            let decoders = sequence
                .get_decoders()
                .iter()
                .map(|decoder| match context_free_decoder(decoder) {
                    Some(decoder) => {
                        changed = true;
                        decoder
                    }
                    None => decoder.clone(),
                })
                .collect();
            changed.then(|| {
                DecoderWrapper::Sequence(tokenizers::decoders::sequence::Sequence::new(decoders))
            })
        }
        _ => None,
    }
}

/// The grammar adapter's stable JSON interface isolates its tokenizer version.
/// Runtime encoding and decoding keep using the workspace tokenizer.
pub(crate) fn grammar_byte_tokenizer(
    tokenizer: &tokenizers::Tokenizer,
) -> anyhow::Result<toktrie_hf_tokenizers::ByteTokenizer> {
    let json = tokenizer
        .to_string(false)
        .map_err(anyhow::Error::from_boxed)
        .context("serialize declared tokenizer for grammar byte conversion")?;
    toktrie_hf_tokenizers::ByteTokenizer::from_json_bytes(json.as_bytes())
        .context("grammar adapter cannot read the declared tokenizer JSON; check decoder support in toktrie_hf_tokenizers")
}

/// Return exact per-token byte pieces, retaining incomplete UTF-8 fragments.
///
/// Byte codecs use the same vocabulary adapter as constrained decoding. Other
/// codecs already produce Unicode pieces. The vocabulary is built once per
/// batch of requested ids, never once per token, and no shared tokenizer state
/// is changed.
pub fn token_pieces(tokenizer: &Tokenizer, ids: &[u32]) -> anyhow::Result<Vec<Vec<u8>>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let spellings =
        ids.iter()
            .map(|id| {
                tokenizer.inner().id_to_token(*id).with_context(|| {
                    format!("token id {id} is absent from the tokenizer vocabulary")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
    if !tokenizer
        .inner()
        .get_decoder()
        .is_some_and(has_byte_decoder)
    {
        let normalized = tokenizer
            .inner()
            .get_decoder()
            .and_then(context_free_decoder);
        let decoder = normalized
            .as_ref()
            .or_else(|| tokenizer.inner().get_decoder());
        let specials = tokenizer
            .inner()
            .get_added_vocabulary()
            .get_added_tokens_decoder();
        return ids.iter().zip(spellings).map(|(id,spelling)| {
            if specials.get(id).is_some_and(|token|token.special) { return Ok(spelling.into_bytes()); }
            match decoder {
                Some(decoder)=>decoder.decode(vec![spelling]).map(String::into_bytes)
                    .map_err(|error|anyhow::anyhow!("token {id} byte-piece decoding failed: {error}; repair the declared tokenizer decoder")),
                None=>Ok(spelling.into_bytes()),
            }
        }).collect();
    }
    let size = tokenizer.inner().get_vocab_size(true);
    for (token, id) in tokenizer.inner().get_vocab(true) {
        ensure!(
            usize::try_from(id).is_ok_and(|id| id < size),
            "byte piece conversion requires dense vocabulary ids; token {token:?} has id {id} outside the {size} vocabulary entries; repair tokenizer.json or disable the operation that requires lossless token bytes"
        );
    }
    let vocabulary=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| grammar_byte_tokenizer(tokenizer.inner())))
        .map_err(|_| anyhow::anyhow!("byte piece conversion rejected invalid tokenizer codec data; repair tokenizer.json or disable the operation that requires lossless token bytes"))?
        .context("byte piece conversion cannot interpret the declared decoder; repair tokenizer.json or disable the operation that requires lossless token bytes")?
        .token_bytes();
    ids.iter().zip(spellings).map(|(id,spelling)| {
        let bytes=vocabulary.get(usize::try_from(*id)?).with_context(|| format!("token id {id} has no byte vocabulary entry"))?;
        // The grammar adapter frames special tokens with a marker. Strip only
        // that exact marker+spelling frame, never an ordinary 0xff byte piece.
        let bytes=if bytes.first()==Some(&TokTrie::SPECIAL_TOKEN_MARKER) && bytes.get(1..)==Some(spelling.as_bytes()) {&bytes[1..]} else {bytes.as_slice()};
        if bytes.is_empty() && !spelling.is_empty() {
            bail!("token id {id} ({spelling:?}) has no lossless byte piece for the declared decoder; repair tokenizer.json or disable the operation that requires lossless token bytes");
        }
        Ok(bytes.to_vec())
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    fn tokenizer(byte_level: bool) -> Tokenizer {
        let mut value: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        let mut vocab = value["model"]["vocab"].as_object().unwrap().clone();
        vocab.remove("hello");
        vocab.remove("world");
        vocab.remove("tok16");
        vocab.insert(if byte_level { "Ã" } else { "<0xC3>" }.into(), json!(4));
        vocab.insert(if byte_level { "©" } else { "<0xA9>" }.into(), json!(5));
        vocab.insert(if byte_level { "ÿ" } else { "<0xFF>" }.into(), json!(16));
        value["model"] = json!({"type":"BPE","unk_token":"<unk>","byte_fallback":!byte_level,"vocab":vocab,"merges":[]});
        value["decoder"] = if byte_level {
            json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false})
        } else {
            json!({"type":"ByteFallback"})
        };
        if byte_level {
            value["pre_tokenizer"] = value["decoder"].clone();
        }
        value["added_tokens"] = json!([{"id":31,"content":"tok31","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}]);
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        Tokenizer::from_file(path).unwrap()
    }
    #[test]
    fn selected_special_output_is_request_local_and_keeps_unicode_lookbehind() {
        for byte_level in [false, true] {
            let tokenizer = tokenizer(byte_level);
            let ids = [31, 4, 5, 31];
            let preserved = BTreeSet::from([31]);
            let mut decoder = TokenTextDecoder::with_preserved_tokens(&preserved);
            let mut text = String::new();
            text.push_str(&decoder.step(&tokenizer, 31).unwrap());
            assert_eq!(decoder.step(&tokenizer, 4).unwrap(), "");
            let mut fork = decoder.clone();
            text.push_str(&decoder.step(&tokenizer, 5).unwrap());
            text.push_str(&decoder.step(&tokenizer, 31).unwrap());
            text.push_str(&decoder.finish(&tokenizer).unwrap());
            assert_eq!(text, "tok31étok31");
            assert_eq!(fork.step(&tokenizer, 5).unwrap(), "é");
            assert_eq!(fork.finish(&tokenizer).unwrap(), "");
            assert_eq!(decode_output(&tokenizer, &ids, &preserved).unwrap(), text);
            assert_eq!(
                token_texts(Some(&tokenizer), &ids, &preserved)
                    .unwrap()
                    .concat(),
                text
            );
            assert_eq!(
                decode_output(&tokenizer, &ids, &BTreeSet::new()).unwrap(),
                "é"
            );
            assert_eq!(tokenizer.decode(&ids).unwrap(), "é");
            let options = crate::GenerateOptions {
                preserved_tokens: BTreeSet::from([9999]),
                ..Default::default()
            };
            let error = options.validate_processors(&tokenizer).unwrap_err();
            assert!(
                format!("{error:#}").contains("preserved_tokens contains absent token ID 9999"),
                "{error:#}"
            );
        }
    }
    #[test]
    fn declared_regex_frame_fragments_can_preserve_explicit_model_symbols() {
        let tokenizer = tokenizer(false);
        let template = crate::response_template::ResponseTemplate::compile(&json!({
            "start_anchor":"A","preserved_tokens":["tok31"],
            "fields":{"content":{"open_pattern":"tok31tok22"}}
        }))
        .unwrap();
        let projection = crate::response_template::ResponseProjection::resolve(
            template.clone(),
            "A".into(),
            4096,
            Some(&tokenizer),
        )
        .unwrap();
        let options = crate::GenerateOptions {
            response_projection: Some(projection),
            ..Default::default()
        };
        let selected = options.effective_preserved_tokens();
        assert!(selected.contains(&31));
        let text = decode_output(&tokenizer, &[31, 22, 23], &selected).unwrap();
        let mut parser = crate::response_template::ResponseParser::new(
            template,
            "A",
            crate::response_template::PrefixOutput::Include,
            4096,
        )
        .unwrap();
        let mut output = parser.feed(&text).unwrap();
        output.content.push_str(&parser.finish().unwrap().content);
        assert_eq!(output.content, "tok23");
        assert_eq!(tokenizer.decode(&[31, 22, 23]).unwrap(), "tok22tok23");
    }
    #[test]
    fn candidate_decoder_forks_preserve_partial_unicode_and_special_policy() {
        for byte_level in [false, true] {
            let tokenizer = tokenizer(byte_level).clone();
            for policy in [SpecialTokenHandling::Skip, SpecialTokenHandling::Preserve] {
                let mut live = TokenTextDecoder::new(policy);
                assert_eq!(
                    live.step(&tokenizer, 31).unwrap(),
                    if policy == SpecialTokenHandling::Preserve {
                        "tok31"
                    } else {
                        ""
                    }
                );
                assert_eq!(live.step(&tokenizer, 4).unwrap(), "");
                let mut terminal = live.clone();
                assert_eq!(terminal.finish(&tokenizer).unwrap(), "�");
                let mut accepted = live.clone();
                assert_eq!(accepted.step(&tokenizer, 5).unwrap(), "é");
                assert_eq!(live.step(&tokenizer, 5).unwrap(), "é");
                assert_eq!(
                    accepted.finish(&tokenizer).unwrap(),
                    live.finish(&tokenizer).unwrap()
                );
            }
        }
    }

    #[test]
    fn candidate_decoder_failure_does_not_change_live_lookbehind() {
        let tokenizer = tokenizer(false);
        let mut live = TokenTextDecoder::new(SpecialTokenHandling::Preserve);
        live.step(&tokenizer, 4).unwrap();
        assert_eq!(live.step(&tokenizer, 5).unwrap(), "é");
        let mut rejected = live.clone();
        assert_eq!(rejected.step(&tokenizer, 16).unwrap(), "");
        assert!(
            rejected
                .finish(&tokenizer)
                .unwrap_err()
                .is::<NonMonotonicText>()
        );
        assert_eq!(live.step(&tokenizer, 31).unwrap(), "tok31");
        assert_eq!(live.finish(&tokenizer).unwrap(), "");
    }

    #[test]
    fn candidate_decoder_forks_keep_contextual_word_separators() {
        let tokenizer = Tokenizer::from_file(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-llm/tokenizer.json"),
        )
        .unwrap();
        let mut live = TokenTextDecoder::new(SpecialTokenHandling::Preserve);
        assert_eq!(live.step(&tokenizer, 4).unwrap(), "hello");
        let mut first = live.clone();
        let mut second = live.clone();
        assert_eq!(first.step(&tokenizer, 5).unwrap(), " world");
        assert_eq!(second.step(&tokenizer, 6).unwrap(), " the");
        assert_eq!(live.step(&tokenizer, 5).unwrap(), " world");
        assert_eq!(live.finish(&tokenizer).unwrap(), "");
    }

    #[test]
    fn protocol_decoder_preserves_specials_and_split_unicode() {
        for byte_level in [false, true] {
            let tokenizer = tokenizer(byte_level);
            for ids in [vec![31, 4, 5, 31], vec![4]] {
                let mut decoder = TokenTextDecoder::new(SpecialTokenHandling::Preserve);
                let mut text = String::new();
                for id in &ids {
                    text.push_str(&decoder.step(&tokenizer, *id).unwrap());
                }
                text.push_str(&decoder.finish(&tokenizer).unwrap_or_else(|error| {
                    panic!("byte_level={byte_level}, ids={ids:?}, emitted={text:?}: {error:#}")
                }));
                assert_eq!(text, tokenizer.decode_with_special_tokens(&ids).unwrap());
            }
        }
    }
    #[test]
    fn protocol_decoder_refuses_to_retract_published_unicode() {
        let tokenizer = tokenizer(false);
        let mut decoder = TokenTextDecoder::new(SpecialTokenHandling::Preserve);
        assert_eq!(decoder.step(&tokenizer, 4).unwrap(), "");
        assert_eq!(decoder.step(&tokenizer, 5).unwrap(), "é");
        assert_eq!(decoder.step(&tokenizer, 16).unwrap(), "");
        assert!(
            decoder
                .finish(&tokenizer)
                .unwrap_err()
                .is::<NonMonotonicText>()
        );
    }
    #[test]
    fn metaspace_pieces_preserve_initial_space_without_mutating_text_decoding() {
        let mut value: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        let vocab = value["model"]["vocab"].as_object_mut().unwrap();
        vocab.remove("world");
        vocab.remove("the");
        vocab.remove("tok31");
        vocab.insert("▁world".into(), json!(5));
        vocab.insert("world".into(), json!(6));
        vocab.insert("▁SPECIAL".into(), json!(31));
        value["decoder"] =
            json!({"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true});
        value["added_tokens"] = json!([{"id":31,"content":"▁SPECIAL","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}]);
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let tokenizer = Tokenizer::from_file(path).unwrap();
        assert_eq!(tokenizer.decode_with_special_tokens(&[5]).unwrap(), "world");
        assert_eq!(
            token_pieces(&tokenizer, &[5, 6, 31]).unwrap(),
            vec![
                b" world".to_vec(),
                b"world".to_vec(),
                "▁SPECIAL".as_bytes().to_vec()
            ]
        );
        assert_eq!(tokenizer.decode_with_special_tokens(&[5]).unwrap(), "world");
    }

    #[test]
    fn incremental_text_preserves_unicode_boundaries_and_terminal_fragments() {
        for byte_level in [false, true] {
            let tokenizer = tokenizer(byte_level);
            let mut decoder = TokenTextDecoder::default();
            assert_eq!(decoder.step(&tokenizer, 4).unwrap(), "");
            assert_eq!(decoder.step(&tokenizer, 5).unwrap(), "é");
            assert_eq!(decoder.finish(&tokenizer).unwrap(), "");
            if !byte_level {
                decoder.step(&tokenizer, 4).unwrap();
                let error = decoder.finish(&tokenizer).unwrap_err();
                assert!(error.is::<NonMonotonicText>());
            }
            for ids in [vec![4], vec![4, 4], vec![4, 5, 4], vec![4, 5, 16]] {
                let chunks = token_texts(Some(&tokenizer), &ids, &Default::default())
                    .unwrap_or_else(|error| {
                        panic!("byte_level={byte_level} ids={ids:?}: {error:#}")
                    });
                assert_eq!(chunks.concat(), tokenizer.decode(&ids).unwrap(), "{ids:?}");
            }
        }
        let tokenizer = Tokenizer::from_file(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-llm/tokenizer.json"),
        )
        .unwrap();
        assert_eq!(
            token_texts(Some(&tokenizer), &[22, 22], &Default::default()).unwrap(),
            vec!["tok22", " tok22"]
        );
    }
    #[test]
    fn byte_codecs_preserve_utf8_fragments_and_literal_ff_without_special_framing() {
        for byte_level in [false, true] {
            let tokenizer = tokenizer(byte_level);
            let ids = tokenizer
                .encode_with_options("é", Default::default())
                .unwrap();
            assert_eq!(ids, vec![4, 5]);
            assert_eq!(
                token_pieces(&tokenizer, &ids).unwrap(),
                vec![vec![195], vec![169]]
            );
            assert_eq!(
                token_pieces(&tokenizer, &[16, 31]).unwrap(),
                vec![vec![255], b"tok31".to_vec()]
            );
            assert_eq!(tokenizer.decode_with_special_tokens(&ids).unwrap(), "é");
            assert!(
                token_pieces(&tokenizer, &[999999])
                    .unwrap_err()
                    .to_string()
                    .contains("absent")
            );
        }
    }
}
