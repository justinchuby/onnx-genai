//! Lossless tokenizer pieces for protocol consumers and byte-oriented tooling.
use anyhow::{Context, bail, ensure};
use llguidance::toktrie::TokTrie;
use onnx_genai_ort::Tokenizer;
use tokenizers::decoders::DecoderWrapper;

#[derive(Debug, thiserror::Error)]
#[error(
    "incremental token decoding changed an already published prefix; use buffered output or repair the package tokenizer decoder"
)]
struct NonMonotonicText;

/// Request-local incremental detokenization state. Retains the decoder's
/// necessary lookbehind rather than repeatedly decoding the whole generation.
#[derive(Default)]
pub(crate) struct TokenTextDecoder {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}
impl TokenTextDecoder {
    pub(crate) fn step(&mut self, tokenizer: &Tokenizer, id: u32) -> anyhow::Result<String> {
        tokenizers::tokenizer::step_decode_stream(
            tokenizer.inner(), id, true, &mut self.ids, &mut self.prefix, &mut self.prefix_index,
        ).map(|text| text.unwrap_or_default()).map_err(|error| {
            if error.is::<tokenizers::tokenizer::DecodeStreamError>() {
                anyhow::Error::new(NonMonotonicText)
            } else {
                anyhow::anyhow!("incremental decoding of token {id} failed: {error}; repair the package tokenizer decoder or its token vocabulary")
            }
        })
    }
    /// Emit pending bytes at the terminal boundary, including the tokenizer's
    /// replacement text for an incomplete final UTF-8 sequence.
    pub(crate) fn finish(&mut self, tokenizer: &Tokenizer) -> anyhow::Result<String> {
        let decoded = tokenizer
            .decode(&self.ids)
            .context("decode pending terminal token text")?;
        let tail = decoded
            .strip_prefix(&self.prefix)
            .ok_or(NonMonotonicText)?
            .to_owned();
        self.prefix = decoded;
        self.prefix_index = self.ids.len();
        Ok(tail)
    }
}

/// Decode a committed token batch with the same boundaries as live generation.
pub(crate) fn token_texts(
    tokenizer: Option<&Tokenizer>,
    ids: &[u32],
) -> anyhow::Result<Vec<String>> {
    let Some(tokenizer) = tokenizer else {
        return Ok(vec![String::new(); ids.len()]);
    };
    // Staged output knows the final text before publishing any callback. A
    // decoder may revise an earlier prefix (e.g. an invalid final byte run);
    // defer that batch's text to its last token instead of publishing a prefix
    // that the immutable final result would later have to retract.
    let final_text = tokenizer.decode(ids)?;
    let attempt = || -> anyhow::Result<Vec<String>> {
        let mut decoder = TokenTextDecoder::default();
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

fn has_byte_decoder(decoder: &DecoderWrapper) -> bool {
    match decoder {
        DecoderWrapper::ByteFallback(_) | DecoderWrapper::ByteLevel(_) => true,
        DecoderWrapper::Sequence(sequence) => sequence.get_decoders().iter().any(has_byte_decoder),
        _ => false,
    }
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
        return ids
            .iter()
            .map(|id| {
                tokenizer
                    .decode_with_special_tokens(&[*id])
                    .map(String::into_bytes)
                    .map_err(Into::into)
            })
            .collect();
    }
    let size = tokenizer.inner().get_vocab_size(true);
    for (token, id) in tokenizer.inner().get_vocab(true) {
        ensure!(
            usize::try_from(id).is_ok_and(|id| id < size),
            "byte piece conversion requires dense vocabulary ids; token {token:?} has id {id} outside the {size} vocabulary entries; use with_pieces=false or repair tokenizer.json"
        );
    }
    let vocabulary=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| toktrie_hf_tokenizers::ByteTokenizer::from_tokenizer(tokenizer.inner().clone())))
        .map_err(|_| anyhow::anyhow!("byte piece conversion rejected invalid tokenizer codec data; use with_pieces=false or repair tokenizer.json"))?
        .context("byte piece conversion cannot interpret the declared decoder; use with_pieces=false or repair tokenizer.json")?
        .token_bytes();
    ids.iter().zip(spellings).map(|(id,spelling)| {
        let bytes=vocabulary.get(usize::try_from(*id)?).with_context(|| format!("token id {id} has no byte vocabulary entry"))?;
        // The grammar adapter frames special tokens with a marker. Strip only
        // that exact marker+spelling frame, never an ordinary 0xff byte piece.
        let bytes=if bytes.first()==Some(&TokTrie::SPECIAL_TOKEN_MARKER) && bytes.get(1..)==Some(spelling.as_bytes()) {&bytes[1..]} else {bytes.as_slice()};
        if bytes.is_empty() && !spelling.is_empty() {
            bail!("token id {id} ({spelling:?}) has no lossless byte piece for the declared decoder; use with_pieces=false or repair tokenizer.json");
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
                let chunks = token_texts(Some(&tokenizer), &ids).unwrap_or_else(|error| {
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
            token_texts(Some(&tokenizer), &[22, 22]).unwrap(),
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
