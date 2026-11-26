//! Resolved streaming contracts for token-local transforms and byte decoders.
use crate::tokenization::{SpecialTokenHandling, has_byte_decoder};
use anyhow::{Context, ensure};
use onnx_genai_ort::Tokenizer;
use std::{
    borrow::Cow,
    collections::{HashMap, VecDeque},
    num::NonZeroUsize,
};
use tokenizers::{Decoder as _, decoders::DecoderWrapper};

pub(crate) struct ByteStreamContract {
    pub(crate) pieces: Vec<Vec<u8>>,
    strips: Vec<StreamStripSpec>,
}

#[derive(Clone, Copy)]
enum StreamStripSpec {
    Initial {
        character: char,
        count: NonZeroUsize,
    },
    Trailing {
        character: char,
        count: NonZeroUsize,
    },
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ByteStreamState {
    strips: Vec<StreamStripState>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum StripState {
    Matching {
        character: char,
        remaining: NonZeroUsize,
        matched: Vec<u8>,
    },
    Done,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum StreamStripState {
    Initial(StripState),
    Trailing {
        character: char,
        count: NonZeroUsize,
        held: VecDeque<u8>,
    },
}

#[derive(Debug, thiserror::Error)]
#[error(
    "tokenizer.decoder retained {retained} byte(s), exceeding the {limit}-byte response buffer; reduce the Strip stop count or increase the response buffer"
)]
pub(crate) struct ByteStreamBufferExceeded {
    retained: usize,
    limit: usize,
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
    /// Raw grammars consume vocabulary pieces before publication-only stripping.
    /// Only an already resolved contract can produce this identity codec.
    pub(crate) fn into_raw(mut self) -> Self {
        self.strips.clear();
        self
    }

    /// Filter special IDs before any stream-context transforms run.
    pub(crate) fn resolve_output(
        tokenizer: &Tokenizer,
        preserved: &std::collections::BTreeSet<u32>,
    ) -> anyhow::Result<Option<Self>> {
        let Some(mut contract) = Self::resolve(tokenizer, SpecialTokenHandling::Preserve)? else {
            return Ok(None);
        };
        let added = tokenizer.inner().get_added_vocabulary();
        for (&id, token) in added.get_added_tokens_decoder() {
            if added.is_special_token(&token.content) && !preserved.contains(&id) {
                contract.pieces.get_mut(id as usize).with_context(||format!("special token {id} is absent from the byte vocabulary; repair tokenizer.json"))?.clear();
            }
        }
        Ok(Some(contract))
    }

    /// Token-local stages preserve input-token boundaries. Once byte chunks are
    /// fused, finite initial/trailing character stripping is admitted here. Other
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
                "{path} changes token context before the byte decoder; streaming protocol output requires token-local transforms before one byte stage, or a separately resolved contextual decoder contract"
            );
        }
        let byte_level = matches!(stages[byte_index].1, DecoderWrapper::ByteLevel(_));
        let mut fused = byte_level;
        let mut strips = Vec::new();
        let mut retention = 0usize;
        for (path, stage) in &stages[byte_index + 1..] {
            match stage {
                DecoderWrapper::Fuse(_) => fused = true,
                DecoderWrapper::Strip(strip) if strip.start == 0 && strip.stop == 0 => {}
                DecoderWrapper::Strip(strip) if fused => {
                    if let Some(count) = NonZeroUsize::new(strip.start) {
                        strips.push(StreamStripSpec::Initial {
                            character: strip.content,
                            count,
                        });
                        retention = retention
                            .checked_add(strip.content.len_utf8() - 1)
                            .context("initial Strip retention overflows")?;
                    }
                    if let Some(count) = NonZeroUsize::new(strip.stop) {
                        let bytes = count.get().checked_mul(strip.content.len_utf8())
                            .and_then(|bytes| bytes.checked_add(strip.content.len_utf8()-1))
                            .context("trailing Strip retention overflows; reduce tokenizer.decoder Strip stop")?;
                        retention = retention
                            .checked_add(bytes)
                            .context("composed Strip retention overflows")?;
                        strips.push(StreamStripSpec::Trailing {
                            character: strip.content,
                            count,
                        });
                    }
                    ensure!(
                        retention <= 4 * 1024 * 1024,
                        "{path} requires {retention} retained bytes, exceeding the 4 MiB streaming decoder budget; reduce Strip stop counts"
                    );
                }
                _ => anyhow::bail!(
                    "{path} needs a contextual byte-stream contract: after a byte decoder, streaming protocol output requires Fuse before contextual Strip; other context-changing stages need explicit streaming support"
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
            "streaming byte decoding requires dense tokenizer.model.vocab IDs; repair tokenizer.json before enabling this byte-stream contract"
        );
        let specials = tokenizer.inner().get_added_tokens_decoder();
        let mut pieces = Vec::with_capacity(size);
        for index in 0..size {
            let id = u32::try_from(index).context("byte-stream vocabulary exceeds token IDs")?;
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
        Ok(Some(Self { pieces, strips }))
    }

    pub(crate) fn has_trailing_strip(&self) -> bool {
        self.strips
            .iter()
            .any(|strip| matches!(strip, StreamStripSpec::Trailing { .. }))
    }
    /// Prove a marker is visible after its final token in the given prefix
    /// context. Initial strips must not consume it, and no suffix may delay it.
    pub(crate) fn marker_stable(&self, tokens: &[u32], prefix: &[u32]) -> anyhow::Result<bool> {
        let mut state = self.new_state();
        for &id in prefix {
            let piece = self
                .pieces
                .get(id as usize)
                .context("reasoning prefix token is absent from the resolved byte vocabulary")?;
            let _ = state.step(piece);
        }
        let mut expected = Vec::new();
        let mut published = Vec::new();
        for &id in tokens {
            let piece = self
                .pieces
                .get(id as usize)
                .context("reasoning marker token is absent from the resolved byte vocabulary")?;
            expected.extend_from_slice(piece);
            published.extend_from_slice(&state.step(piece));
        }
        Ok(published.ends_with(&expected) && state.retained_bytes() == 0)
    }

    pub(crate) fn new_state(&self) -> ByteStreamState {
        ByteStreamState {
            strips: self
                .strips
                .iter()
                .map(|strip| match strip {
                    StreamStripSpec::Initial { character, count } => {
                        StreamStripState::Initial(StripState::Matching {
                            character: *character,
                            remaining: *count,
                            matched: Vec::new(),
                        })
                    }
                    StreamStripSpec::Trailing { character, count } => StreamStripState::Trailing {
                        character: *character,
                        count: *count,
                        held: VecDeque::new(),
                    },
                })
                .collect(),
        }
    }
}

impl ByteStreamState {
    pub(crate) fn has_pending_prefix(&self) -> bool {
        self.strips.iter().any(|strip| match strip {
            StreamStripState::Initial(StripState::Matching { matched, .. }) => !matched.is_empty(),
            StreamStripState::Trailing {
                character, held, ..
            } => held.len() % character.len_utf8() != 0,
            _ => false,
        })
    }
    pub(crate) fn is_identity(&self) -> bool {
        self.strips
            .iter()
            .all(|strip| matches!(strip, StreamStripState::Initial(StripState::Done)))
    }
    pub(crate) fn retained_bytes(&self) -> usize {
        self.strips
            .iter()
            .map(|strip| match strip {
                StreamStripState::Initial(StripState::Matching { matched, .. }) => matched.len(),
                StreamStripState::Trailing { held, .. } => held.len(),
                _ => 0,
            })
            .sum()
    }
    pub(crate) fn storage_bytes(&self) -> usize {
        self.retained_bytes() + self.strips.len() * std::mem::size_of::<StreamStripState>()
    }
    /// Raw continuations which can release only part of a delayed suffix.
    /// Include completion of split matching scalars as well as full characters.
    pub(crate) fn suffix_continuations(&self) -> Vec<Vec<u8>> {
        let mut pieces = std::collections::BTreeSet::new();
        let single_suffix = self
            .strips
            .iter()
            .filter(|strip| matches!(strip, StreamStripState::Trailing { .. }))
            .count()
            == 1
            && self.strips.iter().all(|strip| {
                !matches!(
                    strip,
                    StreamStripState::Initial(StripState::Matching { .. })
                )
            });
        for strip in &self.strips {
            let (character, partial) = match strip {
                StreamStripState::Initial(StripState::Matching {
                    character, matched, ..
                }) => (*character, matched.len()),
                StreamStripState::Trailing {
                    character, held, ..
                } => (*character, held.len() % character.len_utf8()),
                _ => continue,
            };
            let mut encoded = [0; 4];
            let target = character.encode_utf8(&mut encoded).as_bytes();
            let mut jumped = false;
            if let StreamStripState::Trailing { count, held, .. } = strip
                && partial == 0
                && held.len() / target.len() < count.get()
            {
                // For one suffix, every shorter filling run has identical
                // terminal output and flushing it cannot repair an already
                // rejected grammar prefix. Jump directly to its first release.
                pieces.insert(target.repeat(count.get() - held.len() / target.len() + 1));
                jumped = single_suffix;
            }
            if !jumped {
                pieces.insert(target.to_vec());
            }
            if partial > 0 {
                pieces.insert(target[partial..].to_vec());
            }
        }
        pieces.into_iter().collect()
    }
    pub(crate) fn check_buffer(&self, limit: usize) -> anyhow::Result<()> {
        let retained = self.retained_bytes();
        if retained > limit {
            return Err(ByteStreamBufferExceeded { retained, limit }.into());
        }
        Ok(())
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
    fn flush(&mut self, terminal: bool) -> Vec<u8> {
        let mut output = Vec::new();
        for strip in &mut self.strips {
            let mut next = strip.apply(&output);
            next.extend(strip.finish(terminal));
            output = next;
        }
        output
    }
    /// Bytes which must become visible if more non-stripped text follows.
    /// Do not confuse this provisional suffix with terminal stripped output.
    pub(crate) fn provisional_tail(&self) -> Vec<u8> {
        self.clone().flush(false)
    }
    pub(crate) fn finish(&mut self) -> Vec<u8> {
        self.flush(true)
    }
}

impl StreamStripState {
    fn apply(&mut self, bytes: &[u8]) -> Vec<u8> {
        match self {
            Self::Initial(strip) => strip.apply(bytes),
            Self::Trailing {
                character,
                count,
                held,
            } => {
                let mut encoded = [0; 4];
                let target = character.encode_utf8(&mut encoded).as_bytes();
                let mut output = Vec::new();
                for &byte in bytes {
                    if byte == target[held.len() % target.len()] {
                        held.push_back(byte);
                        if held.len() / target.len() > count.get() {
                            for _ in 0..target.len() {
                                output.push(
                                    held.pop_front().expect("retained complete Strip character"),
                                );
                            }
                        }
                    } else {
                        output.extend(held.drain(..));
                        if byte == target[0] {
                            held.push_back(byte);
                        } else {
                            output.push(byte);
                        }
                    }
                }
                output
            }
        }
    }
    fn finish(&mut self, terminal: bool) -> Vec<u8> {
        match self {
            Self::Initial(strip) => {
                if terminal {
                    strip.finish()
                } else {
                    Vec::new()
                }
            }
            Self::Trailing {
                character, held, ..
            } => {
                let partial = held.len() % character.len_utf8();
                if terminal && partial == 0 {
                    held.clear();
                    Vec::new()
                } else {
                    held.drain(..).collect()
                }
            }
        }
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
    fn trailing_strips_match_fused_decoder_and_hold_only_a_bounded_suffix() {
        for character in ['x', 'é', '終'] {
            let tokenizer = tokenizer(
                json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},
                {"type":"Strip","content":character.to_string(),"start":0,"stop":2}]}),
                &[],
            );
            let contract = ByteStreamContract::resolve(&tokenizer, SpecialTokenHandling::Skip)
                .unwrap()
                .unwrap();
            for text in [
                format!("abc{character}{character}{character}"),
                format!("abc{character}z{character}"),
                format!("abc{character}{character}z"),
            ] {
                for chunk in 1..=text.len() {
                    let mut state = contract.new_state();
                    let mut output = Vec::new();
                    for bytes in text.as_bytes().chunks(chunk) {
                        output.extend_from_slice(&state.step(bytes));
                        assert!(state.retained_bytes() < 3 * character.len_utf8());
                    }
                    let mut preview = output.clone();
                    preview.extend(state.provisional_tail());
                    assert_eq!(preview, text.as_bytes());
                    output.extend(state.finish());
                    let expected = tokenizer
                        .inner()
                        .get_decoder()
                        .unwrap()
                        .decode_chain(vec![text.clone()])
                        .unwrap()
                        .join("");
                    assert_eq!(
                        output,
                        expected.as_bytes(),
                        "{character:?}, {text:?}, {chunk}"
                    );
                }
            }
        }
    }

    #[test]
    fn composed_trailing_strips_match_canonical_decoding_without_retracting_bytes() {
        for stages in [
            json!([{ "type":"Strip","content":"x","start":1,"stop":2},{"type":"Strip","content":"y","start":0,"stop":1}]),
            json!([{ "type":"Strip","content":"é","start":0,"stop":2},{"type":"Strip","content":"x","start":0,"stop":1}]),
            json!([{ "type":"Strip","content":"x","start":0,"stop":1},{"type":"Strip","content":"é","start":0,"stop":2}]),
        ] {
            let mut decoders = vec![json!({"type":"ByteFallback"}), json!({"type":"Fuse"})];
            decoders.extend(stages.as_array().unwrap().iter().cloned());
            let tokenizer = tokenizer(json!({"type":"Sequence","decoders":decoders}), &[]);
            let contract = ByteStreamContract::resolve(&tokenizer, SpecialTokenHandling::Skip)
                .unwrap()
                .unwrap();
            for text in ["xxxAyyyxx", "abcééxx", "abcxxéé", "abcéxéx", "xéAxxéé"] {
                let expected = tokenizer
                    .inner()
                    .get_decoder()
                    .unwrap()
                    .decode_chain(vec![text.into()])
                    .unwrap()
                    .join("");
                for chunk in 1..=text.len() {
                    let mut state = contract.new_state();
                    let mut output = Vec::new();
                    for bytes in text.as_bytes().chunks(chunk) {
                        output.extend_from_slice(&state.step(bytes));
                        assert!(expected.as_bytes().starts_with(&output));
                    }
                    output.extend(state.finish());
                    assert_eq!(output, expected.as_bytes(), "{stages}, {text}, {chunk}");
                }
            }
        }
        let oversized = tokenizer(
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},
            {"type":"Strip","content":"x","start":0,"stop":5*1024*1024}]}),
            &[],
        );
        let message = format!(
            "{:#}",
            ByteStreamContract::resolve(&oversized, SpecialTokenHandling::Skip)
                .err()
                .unwrap()
        );
        assert!(
            message.contains("4 MiB") && message.contains("Strip stop"),
            "{message}"
        );
    }

    #[test]
    fn trailing_strips_preserve_partial_unicode_and_account_for_a_full_window() {
        let tokenizer = tokenizer(
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},
            {"type":"Strip","content":"é","start":0,"stop":1}]}),
            &[],
        );
        let contract = ByteStreamContract::resolve(&tokenizer, SpecialTokenHandling::Skip)
            .unwrap()
            .unwrap();
        let mut state = contract.new_state();
        assert_eq!(&*state.step(b"abc\xc3"), b"abc");
        assert!(state.has_pending_prefix());
        assert_eq!(state.provisional_tail(), b"\xc3");
        assert_eq!(state.clone().finish(), b"\xc3");
        assert!(state.suffix_continuations().contains(&vec![0xa9]));
        assert!(state.step(b"\xa9").is_empty());
        assert!(!state.has_pending_prefix());
        assert!(state.step(b"\xc3").is_empty());
        let mut completion = state.clone();
        assert_eq!(&*completion.step(&[0xa9]), "é".as_bytes());
        assert!(completion.finish().is_empty());
        assert!(
            state
                .check_buffer(2)
                .unwrap_err()
                .is::<ByteStreamBufferExceeded>()
        );
        assert_eq!(&*state.step(b"\xa9"), "é".as_bytes());
        assert!(state.finish().is_empty());
    }

    #[test]
    fn byte_stream_contract_rejects_unresolved_chunk_local_stages_with_paths() {
        for decoder in [
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Strip","content":" ","start":1,"stop":0}]}),
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"String":"x"},"content":"y"}]}),
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
