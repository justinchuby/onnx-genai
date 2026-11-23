//! Resolved streaming contracts for token-local transforms and byte decoders.
use crate::tokenization::{SpecialTokenHandling, has_byte_decoder};
use anyhow::{Context, ensure};
use onnx_genai_ort::Tokenizer;
use std::{borrow::Cow, collections::HashMap, num::NonZeroUsize};
use tokenizers::{Decoder as _, decoders::DecoderWrapper};

pub(crate) struct ByteStreamContract {
    pub(crate) pieces: Vec<Vec<u8>>,
    initial_strips: Vec<InitialStrip>,
}

#[derive(Clone, Copy)]
struct InitialStrip {
    character: char,
    count: NonZeroUsize,
}

#[derive(Clone)]
pub(crate) struct ByteStreamState {
    strips: Vec<StripState>,
}

#[derive(Clone)]
enum StripState {
    Matching {
        character: char,
        remaining: NonZeroUsize,
        matched: Vec<u8>,
    },
    Done,
}

fn flatten<'a>(
    decoder: &'a DecoderWrapper,
    path: String,
    stages: &mut Vec<(String, &'a DecoderWrapper)>,
) {
    if let DecoderWrapper::Sequence(sequence) = decoder {
        for (index, child) in sequence.get_decoders().iter().enumerate() {
            flatten(child, format!("{path}.decoders[{index}]"), stages);
        }
    } else {
        stages.push((path, decoder));
    }
}

fn byte_level_alphabet() -> anyhow::Result<HashMap<char, u8>> {
    // Ask the dependency's public byte adapter for its protocol mapping rather
    // than infer byte values from glyph code points or a model identity.
    let alphabet = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect::<Vec<_>>();
    let vocab = alphabet
        .iter()
        .enumerate()
        .map(|(id, character)| (character.to_string(), serde_json::json!(id)))
        .collect::<serde_json::Map<_, _>>();
    let config = serde_json::json!({"version":"1.0","added_tokens":[],
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "model":{"type":"BPE","vocab":vocab,"merges":[],"unk_token":null,"byte_fallback":false}});
    let bytes = serde_json::to_vec(&config)?;
    let pieces = llguidance::panic_utils::catch_unwind(std::panic::AssertUnwindSafe(|| {
        toktrie_hf_tokenizers::ByteTokenizer::from_json_bytes(&bytes)
    }))
    .context("resolve the ByteLevel protocol alphabet from the tokenizer adapter")?
    .token_bytes();
    ensure!(
        pieces.len() == alphabet.len(),
        "ByteLevel adapter returned a different alphabet size; use matching tokenizer and byte-adapter dependencies"
    );
    alphabet.into_iter().zip(pieces).map(|(character,piece)| {
        ensure!(piece.len()==1,"ByteLevel glyph {character:?} did not resolve to one byte; use matching tokenizer and byte-adapter dependencies");
        Ok((character,piece[0]))
    }).collect()
}

impl ByteStreamContract {
    /// Token-local stages preserve input-token boundaries. Once byte chunks are
    /// fused, only finite initial character stripping is admitted here. Other
    /// stages need a separate contextual-stream proof rather than a guessed map.
    pub(crate) fn resolve(
        tokenizer: &Tokenizer,
        policy: SpecialTokenHandling,
    ) -> anyhow::Result<Option<Self>> {
        let Some(decoder) = tokenizer
            .inner()
            .get_decoder()
            .filter(|decoder| has_byte_decoder(decoder))
        else {
            return Ok(None);
        };
        let mut stages = Vec::new();
        flatten(decoder, "tokenizer.decoder".into(), &mut stages);
        let byte_index = stages
            .iter()
            .position(|(_, stage)| {
                matches!(
                    stage,
                    DecoderWrapper::ByteFallback(_) | DecoderWrapper::ByteLevel(_)
                )
            })
            .context("byte decoder contract contains no byte stage; repair tokenizer.decoder")?;
        for (path, stage) in &stages[..byte_index] {
            ensure!(
                matches!(stage, DecoderWrapper::Replace(_) | DecoderWrapper::Strip(_)),
                "{path} changes token context before the byte decoder; JSON output requires token-local transforms before one byte stage, or a separately resolved contextual decoder contract"
            );
        }
        let byte_level = matches!(stages[byte_index].1, DecoderWrapper::ByteLevel(_));
        let mut fused = byte_level;
        let mut initial_strips = Vec::new();
        for (path, stage) in &stages[byte_index + 1..] {
            match stage {
                DecoderWrapper::Fuse(_) => fused = true,
                DecoderWrapper::Strip(strip) if strip.start == 0 && strip.stop == 0 => {}
                DecoderWrapper::Strip(strip) if fused && strip.stop == 0 => {
                    if let Some(count) = NonZeroUsize::new(strip.start) {
                        initial_strips.push(InitialStrip {
                            character: strip.content,
                            count,
                        });
                    }
                }
                _ => anyhow::bail!(
                    "{path} needs a contextual byte-stream contract: after a byte decoder, JSON output currently supports Fuse and initial Strip on a fused stream; trailing stripping and other context-changing stages need explicit streaming support"
                ),
            }
        }
        let mapping = byte_level.then(byte_level_alphabet).transpose()?;
        let vocab = tokenizer.inner().get_vocab(true);
        let size = vocab.len();
        ensure!(
            vocab
                .values()
                .all(|id| usize::try_from(*id).is_ok_and(|id| id < size)),
            "JSON byte decoding requires dense tokenizer.model.vocab IDs; repair tokenizer.json before enabling this byte-stream contract"
        );
        let specials = tokenizer.inner().get_added_tokens_decoder();
        let mut pieces = Vec::with_capacity(size);
        for index in 0..size {
            let id = u32::try_from(index).context("JSON byte vocabulary exceeds token IDs")?;
            if policy == SpecialTokenHandling::Skip
                && specials.get(&id).is_some_and(|token| token.special)
            {
                pieces.push(Vec::new());
                continue;
            }
            let mut token=tokenizer.inner().id_to_token(id).with_context(||format!("tokenizer.model.vocab is missing byte token ID {id}; repair the dense vocabulary"))?;
            for (path, stage) in &stages[..byte_index] {
                let mut transformed =
                    llguidance::panic_utils::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        stage
                            .decode_chain(vec![token])
                            .map_err(anyhow::Error::from_boxed)
                    }))
                    .with_context(|| {
                        format!(
                            "decode token {id} through {path}; repair the token-local decoder stage"
                        )
                    })?;
                ensure!(
                    transformed.len() == 1,
                    "{path} changed token {id} into multiple pieces; use a token-local byte decoder contract"
                );
                token = transformed
                    .pop()
                    .context("validated token-local decoder lost its only piece")?;
            }
            let piece = if let Some(mapping) = &mapping {
                token
                    .chars()
                    .map(|character| mapping.get(&character).copied())
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_else(|| token.as_bytes().to_vec())
            } else if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                match u8::from_str_radix(&token[3..5], 16) {
                    Ok(byte) => vec![byte],
                    Err(_) => token.into_bytes(),
                }
            } else {
                token.into_bytes()
            };
            pieces.push(piece);
        }
        Ok(Some(Self {
            pieces,
            initial_strips,
        }))
    }

    pub(crate) fn new_state(&self) -> ByteStreamState {
        ByteStreamState {
            strips: self
                .initial_strips
                .iter()
                .map(|strip| StripState::Matching {
                    character: strip.character,
                    remaining: strip.count,
                    matched: Vec::new(),
                })
                .collect(),
        }
    }
}

impl ByteStreamState {
    pub(crate) fn has_pending_prefix(&self) -> bool {
        self.strips
            .iter()
            .any(|strip| matches!(strip,StripState::Matching {matched,..} if !matched.is_empty()))
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.strips
            .iter()
            .all(|strip| matches!(strip, StripState::Done))
    }

    pub(crate) fn step<'a>(&mut self, bytes: &'a [u8]) -> Cow<'a, [u8]> {
        if self.is_identity() {
            return Cow::Borrowed(bytes);
        }
        let mut output = bytes.to_vec();
        for strip in &mut self.strips {
            output = strip.apply(&output);
        }
        Cow::Owned(output)
    }

    pub(crate) fn finish(&mut self) -> Vec<u8> {
        let mut output = Vec::new();
        for strip in &mut self.strips {
            let mut next = strip.apply(&output);
            next.extend(strip.finish());
            output = next;
        }
        output
    }
}

impl StripState {
    fn apply(&mut self, bytes: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            match self {
                Self::Done => {
                    output.extend_from_slice(&bytes[at..]);
                    break;
                }
                Self::Matching {
                    character,
                    remaining,
                    matched,
                } => {
                    let mut encoded = [0; 4];
                    let target = character.encode_utf8(&mut encoded).as_bytes();
                    if target.get(matched.len()) != Some(&bytes[at]) {
                        output.extend_from_slice(matched);
                        output.extend_from_slice(&bytes[at..]);
                        *self = Self::Done;
                        break;
                    }
                    matched.push(bytes[at]);
                    at += 1;
                    if matched.len() == target.len() {
                        matched.clear();
                        if let Some(next) = NonZeroUsize::new(remaining.get() - 1) {
                            *remaining = next;
                        } else {
                            *self = Self::Done;
                        }
                    }
                }
            }
        }
        output
    }
    fn finish(&mut self) -> Vec<u8> {
        match std::mem::replace(self, Self::Done) {
            Self::Matching { matched, .. } => matched,
            Self::Done => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn tokenizer(decoder: Value, overrides: &[(u32, &str)]) -> Tokenizer {
        let mut config: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tiny-llm/tokenizer.json"
        )))
        .unwrap();
        let vocab = config["model"]["vocab"].as_object_mut().unwrap();
        for (id, piece) in overrides {
            let old = vocab
                .iter()
                .find(|(_, value)| value.as_u64() == Some(u64::from(*id)))
                .unwrap()
                .0
                .clone();
            vocab.remove(&old);
            assert!(vocab.insert((*piece).into(), json!(id)).is_none());
        }
        config["model"] = json!({"type":"BPE","vocab":config["model"]["vocab"],"merges":[],"unk_token":"<unk>","byte_fallback":true});
        config["decoder"] = decoder;
        config["added_tokens"] = json!([{"id":3,"content":"<eos>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}]);
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
        Tokenizer::from_file(path).unwrap()
    }

    fn matches_decoding(
        tokenizer: &Tokenizer,
        policy: SpecialTokenHandling,
        ids: &[u32],
    ) -> Vec<u8> {
        let contract = ByteStreamContract::resolve(tokenizer, policy)
            .unwrap()
            .unwrap();
        let mut state = contract.new_state();
        let mut output = Vec::new();
        for (index, &id) in ids.iter().enumerate() {
            output.extend_from_slice(&state.step(&contract.pieces[id as usize]));
            let mut at_end = state.clone();
            let mut bytes = output.clone();
            bytes.extend(at_end.finish());
            match std::str::from_utf8(&bytes) {
                Ok(text) => {
                    let actual = match policy {
                        SpecialTokenHandling::Skip => tokenizer.decode(&ids[..=index]),
                        SpecialTokenHandling::Preserve => {
                            tokenizer.decode_with_special_tokens(&ids[..=index])
                        }
                    }
                    .unwrap();
                    assert_eq!(text, actual, "{policy:?}, {:?}", &ids[..=index]);
                }
                Err(error) => assert!(
                    error.error_len().is_none(),
                    "invalid test byte stream {bytes:?}"
                ),
            }
        }
        output.extend(state.finish());
        output
    }

    #[test]
    fn composed_byte_fallback_stream_matches_token_local_replacement_and_prefix_stripping() {
        let tokenizer = tokenizer(
            json!({"type":"Sequence","decoders":[
                {"type":"Replace","pattern":{"String":"¤"},"content":" "},
                {"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}
            ]}),
            &[
                (4, "<0xE7>"),
                (5, "<0xB5>"),
                (16, "<0x82>"),
                (22, "¤\""),
                (23, "\""),
                (26, "A"),
            ],
        );
        assert_eq!(
            matches_decoding(
                &tokenizer,
                SpecialTokenHandling::Skip,
                &[22, 4, 5, 16, 23, 3]
            ),
            "\"終\"".as_bytes()
        );
        assert_eq!(
            matches_decoding(&tokenizer, SpecialTokenHandling::Skip, &[3, 22, 26, 23]),
            b"\"A\""
        );
        assert_eq!(
            matches_decoding(&tokenizer, SpecialTokenHandling::Preserve, &[3, 22, 26, 23]),
            b"<eos> \"A\""
        );
    }

    #[test]
    fn composed_prefix_strips_hold_unicode_and_compose_in_declared_order() {
        let unicode = tokenizer(
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},
            {"type":"Strip","content":"終","start":2,"stop":0}]}),
            &[
                (4, "<0xE7>"),
                (5, "<0xB5>"),
                (16, "<0x82>"),
                (23, "\""),
                (26, "A"),
            ],
        );
        assert_eq!(
            matches_decoding(
                &unicode,
                SpecialTokenHandling::Skip,
                &[4, 5, 16, 4, 5, 16, 23, 26, 23]
            ),
            b"\"A\""
        );
        let ordered = tokenizer(
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},
            {"type":"Strip","content":" ","start":1,"stop":0},{"type":"Strip","content":"_","start":1,"stop":0}]}),
            &[(22, " _"), (23, "\""), (26, "A")],
        );
        assert_eq!(
            matches_decoding(&ordered, SpecialTokenHandling::Skip, &[22, 23, 26, 23]),
            b"\"A\""
        );
    }

    #[test]
    fn composed_byte_level_uses_its_fused_scope_and_unmapped_glyph_fallback() {
        let tokenizer = tokenizer(
            json!({"type":"Sequence","decoders":[
                {"type":"Replace","pattern":{"String":"_"},"content":"Ġ"},
                {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
                {"type":"Strip","content":" ","start":1,"stop":0}
            ]}),
            &[(4, "Ã"), (5, "©"), (22, "_\""), (23, "\""), (25, "😀")],
        );
        assert_eq!(
            matches_decoding(&tokenizer, SpecialTokenHandling::Skip, &[22, 4, 5, 23]),
            "\"é\"".as_bytes()
        );
        assert_eq!(
            matches_decoding(&tokenizer, SpecialTokenHandling::Skip, &[23, 25, 23]),
            "\"😀\"".as_bytes()
        );
    }

    #[test]
    fn byte_stream_contract_rejects_chunk_local_and_trailing_stripping_with_paths() {
        for decoder in [
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Strip","content":" ","start":1,"stop":0}]}),
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":0,"stop":1}]}),
        ] {
            let tokenizer = tokenizer(decoder, &[]);
            let error = ByteStreamContract::resolve(&tokenizer, SpecialTokenHandling::Skip)
                .err()
                .unwrap();
            let message = format!("{error:#}");
            assert!(
                message.contains("tokenizer.decoder.decoders[")
                    && message.contains("streaming support"),
                "{message}"
            );
        }
    }
}
