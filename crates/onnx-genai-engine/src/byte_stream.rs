//! Resolved streaming contracts for token-local transforms and byte decoders.
use crate::tokenization::{SpecialTokenHandling, has_byte_decoder};
use anyhow::{Context, ensure};
use onnx_genai_ort::Tokenizer;
use std::{
    borrow::Cow,
    collections::{HashMap, VecDeque},
    num::NonZeroUsize,
    sync::Arc,
};
use tokenizers::{Decoder as _, decoders::DecoderWrapper};

#[derive(Debug, thiserror::Error)]
#[error(
    "{path} exceeded its {limit}-byte transform-work budget; simplify the literal decoder chain or reduce candidate/output work"
)]
pub(crate) struct StreamWorkExceeded {
    path: String,
    limit: usize,
}
pub(crate) struct StreamWork {
    remaining: usize,
    limit: usize,
}
impl StreamWork {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            remaining: limit,
            limit,
        }
    }
    pub(crate) fn charge(&mut self, bytes: usize, path: &str) -> anyhow::Result<()> {
        if bytes > self.remaining {
            return Err(StreamWorkExceeded {
                path: path.into(),
                limit: self.limit,
            }
            .into());
        }
        self.remaining -= bytes;
        Ok(())
    }
    pub(crate) fn copies(&mut self, bytes: usize, times: usize, path: &str) -> anyhow::Result<()> {
        let bytes = bytes
            .checked_mul(times)
            .with_context(|| format!("{path} work-size overflow"))?;
        self.charge(bytes, path)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BytePieceKind {
    Text,
    FallbackBytes,
    Skipped,
}

pub(crate) struct ByteStreamContract {
    pub(crate) pieces: Vec<Vec<u8>>,
    pub(crate) kinds: Vec<BytePieceKind>,
    transforms: Vec<StreamTransformSpec>,
}

#[derive(Clone)]
enum StreamTransformSpec {
    Fallback,
    Utf8,
    Replace(Arc<LiteralReplacement>),
    Initial {
        character: char,
        count: NonZeroUsize,
    },
    Trailing {
        character: char,
        count: NonZeroUsize,
    },
}

#[derive(Clone)]
struct FallbackRun {
    bytes: Arc<Vec<u8>>,
    length: usize,
    invalid: bool,
    digest: u64,
}
impl Default for FallbackRun {
    fn default() -> Self {
        Self {
            bytes: Arc::new(Vec::new()),
            length: 0,
            invalid: false,
            digest: 14695981039346656037,
        }
    }
}
impl FallbackRun {
    fn len(&self) -> usize {
        self.length
    }
    fn is_empty(&self) -> bool {
        self.length == 0
    }
    fn output_length(&self) -> anyhow::Result<usize> {
        if !self.invalid && std::str::from_utf8(&self.bytes).is_ok() {
            Ok(self.length)
        } else {
            self.length
                .checked_mul(3)
                .context("ByteFallback replacement byte count overflow")
        }
    }
    fn appended(&self, bytes: &[u8], limit: usize) -> anyhow::Result<Self> {
        let length = self
            .length
            .checked_add(bytes.len())
            .context("ByteFallback run size overflow")?;
        ensure!(
            length <= limit,
            "ByteFallback run exceeds its {limit}-byte retained response budget; increase buffer or end the byte run"
        );
        if self.invalid {
            return Ok(Self {
                length,
                ..self.clone()
            });
        }
        let mut next = Vec::new();
        next.try_reserve_exact(length).context("reserve bounded ByteFallback run before append; free host memory or shorten generation")?;
        next.extend_from_slice(&self.bytes);
        next.extend_from_slice(bytes);
        // An irreparable invalid run's entire eventual text is U+FFFD per byte;
        // its original payload is no longer semantically relevant to any future.
        if std::str::from_utf8(&next).is_err_and(|e| e.error_len().is_some()) {
            return Ok(Self {
                bytes: Arc::new(Vec::new()),
                length,
                invalid: true,
                digest: 0,
            });
        }
        let digest = bytes.iter().fold(self.digest, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(1099511628211)
        });
        Ok(Self {
            bytes: Arc::new(next),
            length,
            invalid: false,
            digest,
        })
    }
}
impl PartialEq for FallbackRun {
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length
            && self.invalid == other.invalid
            && (self.invalid
                || (self.digest == other.digest
                    && (Arc::ptr_eq(&self.bytes, &other.bytes) || self.bytes == other.bytes)))
    }
}
impl Eq for FallbackRun {}
impl std::hash::Hash for FallbackRun {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&(self.length, self.invalid, self.digest), state);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReplacePatternKind {
    Literal,
    Empty,
    Regex,
}
#[derive(Clone, Copy)]
enum ProbeMode {
    Root,
    Pattern,
    Text,
}
#[derive(Debug)]
struct ProbeError;
impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "unexpected canonical Replace serialization; use a supported tokenizers version",
        )
    }
}
impl std::error::Error for ProbeError {}
impl serde::ser::Error for ProbeError {
    fn custom<T: std::fmt::Display>(_: T) -> Self {
        Self
    }
}
struct PatternProbe {
    mode: ProbeMode,
    kind: Option<ReplacePatternKind>,
}
impl serde::ser::SerializeStruct for &mut PatternProbe {
    type Ok = ();
    type Error = ProbeError;
    fn serialize_field<T: serde::Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), ProbeError> {
        if key == "pattern" {
            let mode = self.mode;
            self.mode = ProbeMode::Pattern;
            let result = value.serialize(&mut **self);
            self.mode = mode;
            result
        } else {
            Ok(())
        }
    }
    fn end(self) -> Result<(), ProbeError> {
        Ok(())
    }
}
impl<'a> serde::Serializer for &'a mut PatternProbe {
    type Ok = ();
    type Error = ProbeError;
    type SerializeSeq = serde::ser::Impossible<(), ProbeError>;
    type SerializeTuple = serde::ser::Impossible<(), ProbeError>;
    type SerializeTupleStruct = serde::ser::Impossible<(), ProbeError>;
    type SerializeTupleVariant = serde::ser::Impossible<(), ProbeError>;
    type SerializeMap = serde::ser::Impossible<(), ProbeError>;
    type SerializeStruct = Self;
    type SerializeStructVariant = serde::ser::Impossible<(), ProbeError>;
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self, ProbeError> {
        if matches!(self.mode, ProbeMode::Root) {
            Ok(self)
        } else {
            Err(ProbeError)
        }
    }
    fn serialize_str(self, value: &str) -> Result<(), ProbeError> {
        if matches!(self.mode, ProbeMode::Text) {
            self.kind = Some(if value.is_empty() {
                ReplacePatternKind::Empty
            } else {
                ReplacePatternKind::Literal
            });
            Ok(())
        } else {
            Err(ProbeError)
        }
    }
    fn serialize_newtype_variant<T: serde::Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), ProbeError> {
        if !matches!(self.mode, ProbeMode::Pattern) {
            return Err(ProbeError);
        }
        match variant {
            "String" => {
                self.mode = ProbeMode::Text;
                value.serialize(self)
            }
            "Regex" => {
                self.kind = Some(ReplacePatternKind::Regex);
                Ok(())
            }
            _ => Err(ProbeError),
        }
    }
    fn serialize_bool(self, _: bool) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_i8(self, _: i8) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_i16(self, _: i16) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_i32(self, _: i32) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_i64(self, _: i64) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_u8(self, _: u8) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_u16(self, _: u16) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_u32(self, _: u32) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_u64(self, _: u64) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_f32(self, _: f32) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_f64(self, _: f64) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_char(self, _: char) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_none(self) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_some<T: serde::Serialize + ?Sized>(self, _: &T) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_unit(self) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
    ) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_newtype_struct<T: serde::Serialize + ?Sized>(
        self,
        _: &'static str,
        _: &T,
    ) -> Result<(), ProbeError> {
        Err(ProbeError)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, ProbeError> {
        Err(ProbeError)
    }
    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, ProbeError> {
        Err(ProbeError)
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, ProbeError> {
        Err(ProbeError)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, ProbeError> {
        Err(ProbeError)
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, ProbeError> {
        Err(ProbeError)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, ProbeError> {
        Err(ProbeError)
    }
}
fn replacement_kind(
    replace: &tokenizers::normalizers::replace::Replace,
) -> anyhow::Result<ReplacePatternKind> {
    use serde::Serialize as _;
    let mut probe = PatternProbe {
        mode: ProbeMode::Root,
        kind: None,
    };
    replace
        .serialize(&mut probe)
        .context("inspect canonical Replace pattern kind without allocating a decoder plan")?;
    probe.kind.context(
        "canonical Replace serializer omitted its pattern; use matching tokenizers dependency",
    )
}

/// A literal, nonrecursive leftmost replacement resolved from decoder metadata.
struct LiteralReplacement {
    pattern: Vec<u8>,
    content: Vec<u8>,
    failure: Vec<usize>,
    path: String,
}
// State keys use immutable per-contract allocation identity, not a repeated
// hash of multi-megabyte pattern/content tables at every candidate fork.
impl PartialEq for LiteralReplacement {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}
impl Eq for LiteralReplacement {}
impl std::hash::Hash for LiteralReplacement {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(self, state);
    }
}
impl LiteralReplacement {
    fn resolve(
        replace: &tokenizers::normalizers::replace::Replace,
        path: &str,
    ) -> anyhow::Result<Self> {
        match replacement_kind(replace)? {
            ReplacePatternKind::Literal => {}
            ReplacePatternKind::Empty => anyhow::bail!(
                "{path} empty literal Replace needs explicit zero-width streaming support; use a nonempty String pattern"
            ),
            ReplacePatternKind::Regex => anyhow::bail!(
                "{path} regex Replace needs separately resolved streaming support; use a nonempty String pattern"
            ),
        }
        #[derive(serde::Deserialize)]
        struct Config {
            pattern: Pattern,
            content: String,
        }
        #[derive(serde::Deserialize)]
        enum Pattern {
            String(String),
            Regex(String),
        }
        let config: Config = serde_json::from_value(
            serde_json::to_value(replace)
                .with_context(|| format!("inspect {path} literal Replace configuration"))?,
        )?;
        let pattern = match config.pattern {
            Pattern::String(pattern) if !pattern.is_empty() => pattern.into_bytes(),
            Pattern::String(_) => anyhow::bail!(
                "{path} empty literal Replace needs explicit zero-width stream semantics; use a nonempty literal"
            ),
            Pattern::Regex(pattern) => anyhow::bail!(
                "{path} regex Replace {pattern:?} needs separately resolved streaming support for regex Replace; use a nonempty literal String pattern"
            ),
        };
        let content = config.content.into_bytes();
        ensure!(
            pattern
                .len()
                .checked_mul(size_of::<usize>() + 1)
                .and_then(|n| n.checked_add(content.len()))
                .is_some_and(|n| n <= 4 * 1024 * 1024),
            "{path} literal Replace configuration exceeds its 4 MiB byte/state budget; shorten pattern/content"
        );
        let mut failure = vec![0; pattern.len()];
        for i in 1..pattern.len() {
            let mut j = failure[i - 1];
            while j > 0 && pattern[i] != pattern[j] {
                j = failure[j - 1];
            }
            if pattern[i] == pattern[j] {
                j += 1;
            }
            failure[i] = j;
        }
        Ok(Self {
            pattern,
            content,
            failure,
            path: path.into(),
        })
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ByteStreamState {
    transforms: Vec<StreamTransformState>,
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
enum StreamTransformState {
    Fallback {
        held: FallbackRun,
    },
    Utf8 {
        held: Vec<u8>,
    },
    Replace {
        spec: Arc<LiteralReplacement>,
        matched: usize,
    },
    Initial(StripState),
    Trailing {
        character: char,
        count: NonZeroUsize,
        held: VecDeque<u8>,
    },
}

#[derive(Debug, thiserror::Error)]
#[error(
    "tokenizer.decoder retained {retained} byte(s), exceeding the {limit}-byte response buffer; reduce decoder retention or increase the response buffer"
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
    let pieces =
        onnx_genai_llguidance::panic_utils::catch_unwind(std::panic::AssertUnwindSafe(|| {
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

pub(crate) fn prepare_literal_output(
    tokenizer: &Tokenizer,
    preserved: &std::collections::BTreeSet<u32>,
) -> anyhow::Result<Option<Arc<ByteStreamContract>>> {
    if !ByteStreamContract::requests_literal_stream(tokenizer)? {
        return Ok(None);
    }
    Ok(Some(Arc::new(
        ByteStreamContract::resolve_output(tokenizer, preserved)?
            .context("literal publication requires its resolved byte decoder contract")?,
    )))
}

fn tokenizer_bytes(tokenizer: &Tokenizer) -> anyhow::Result<u64> {
    struct Count(u64);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| std::io::Error::other("tokenizer serialized size overflow"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut source = Count(0);
    serde_json::to_writer(&mut source, tokenizer.inner())
        .context("count tokenizer configuration for contextual decoder admission")?;
    Ok(source.0)
}
fn literal_catalog_limit(tokenizer: &Tokenizer) -> anyhow::Result<usize> {
    usize::try_from(
        tokenizer_bytes(tokenizer)?
            .checked_mul(4)
            .and_then(|n| {
                u64::try_from(tokenizer.inner().get_vocab_size(true))
                    .ok()
                    .and_then(|size| size.checked_mul(128))
                    .and_then(|slots| n.checked_add(slots))
            })
            .context("literal byte catalogue budget overflow")?,
    )
    .context("literal byte catalogue exceeds host address space")
}

pub(crate) fn reserve_workspace(
    tokenizer: &Tokenizer,
    options: &crate::GenerateOptions,
    authority: &dyn onnx_runtime_memory_governor::MemoryGovernor,
) -> anyhow::Result<Option<onnx_runtime_memory_governor::MemoryLease>> {
    if !ByteStreamContract::requests_literal_stream(tokenizer)? {
        return Ok(None);
    }
    let map = u64::try_from(literal_catalog_limit(tokenizer)?)?
        .checked_mul(4)
        .context("literal decoder vocabulary reservation overflow")?;
    let full_constraint = matches!(
        options.constraint.as_ref(),
        Some(
            crate::GenerateConstraint::Json { .. }
                | crate::GenerateConstraint::JsonSchema { .. }
                | crate::GenerateConstraint::Regex { .. }
                | crate::GenerateConstraint::Lark { .. }
                | crate::GenerateConstraint::XmlToolCall { .. }
        )
    );
    let buffers = u64::try_from(options.max_constraint_bytes)?
        .checked_mul(if full_constraint { 32 } else { 8 })
        .context("literal decoder response reservation overflow")?;
    let frontier = if full_constraint {
        u64::try_from(options.max_constraint_bytes)?
            .saturating_mul(4)
            .clamp(4096, 16 * 1024 * 1024)
    } else {
        0
    };
    let bytes = map
        .checked_add(buffers)
        .and_then(|n| frontier.checked_mul(4).and_then(|f| n.checked_add(f)))
        .and_then(|n| n.checked_add(24 * 1024 * 1024))
        .context("literal decoder workspace reservation overflow")?;
    let holder = crate::engine::Holder::DecoderTransforms;
    let lease=onnx_runtime_memory_governor::MemoryGovernor::reserve(authority,onnx_runtime_memory_api::Tier::Host,bytes,holder.role(),holder.id()).with_context(||format!("reserve {bytes} host bytes for bounded contextual literal decoder before generation; shorten tokenizer/response buffer, free host memory or raise the host limit"))?;
    Ok(Some(lease))
}

impl ByteStreamContract {
    /// Raw grammars consume vocabulary pieces before publication-only stripping.
    /// Only an already resolved contract can produce this identity codec.
    pub(crate) fn into_raw(mut self) -> Self {
        self.transforms.clear();
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
                contract.kinds[id as usize] = BytePieceKind::Skipped;
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
        // Reject an unsupported later stage before compiling any preceding
        // literal tables. Pure preflight validation never allocates half a plan.
        let mut proven_fused = byte_level;
        for (path, stage) in &stages[byte_index + 1..] {
            match stage {
                DecoderWrapper::Fuse(_) => proven_fused = true,
                DecoderWrapper::Strip(strip)
                    if proven_fused || (strip.start == 0 && strip.stop == 0) => {}
                DecoderWrapper::Replace(replace) if proven_fused => {
                    match replacement_kind(replace)? {
                        ReplacePatternKind::Literal => {}
                        ReplacePatternKind::Empty => anyhow::bail!(
                            "{path} empty literal Replace requires separately resolved streaming support; use a nonempty String pattern"
                        ),
                        ReplacePatternKind::Regex => anyhow::bail!(
                            "{path} regex Replace requires separately resolved streaming support; use a nonempty String pattern"
                        ),
                    }
                }
                _ => anyhow::bail!(
                    "{path} needs a contextual byte-stream contract: streaming support requires only Fuse, fused Strip and nonempty literal Replace after one byte stage"
                ),
            }
        }
        let mut fused = byte_level;
        let mut transforms = Vec::new();
        let mut retention = 0usize;
        for (path, stage) in &stages[byte_index + 1..] {
            match stage {
                DecoderWrapper::Fuse(_) => fused = true,
                DecoderWrapper::Strip(strip) if strip.start == 0 && strip.stop == 0 => {}
                DecoderWrapper::Strip(strip) if fused => {
                    if let Some(count) = NonZeroUsize::new(strip.start) {
                        transforms.push(StreamTransformSpec::Initial {
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
                        transforms.push(StreamTransformSpec::Trailing {
                            character: strip.content,
                            count,
                        });
                    }
                    ensure!(
                        retention <= 4 * 1024 * 1024,
                        "{path} requires {retention} retained bytes, exceeding the 4 MiB streaming decoder budget; reduce Strip stop counts"
                    );
                }
                DecoderWrapper::Replace(replace) if fused => {
                    let spec = LiteralReplacement::resolve(replace, path)?;
                    retention = retention
                        .checked_add(spec.pattern.len() - 1)
                        .context("literal Replace retention overflow")?;
                    ensure!(
                        retention <= 4 * 1024 * 1024,
                        "{path} requires too much decoder retention; shorten Replace patterns or Strip counts"
                    );
                    transforms.push(StreamTransformSpec::Replace(Arc::new(spec)));
                }
                _ => anyhow::bail!(
                    "{path} needs a contextual byte-stream contract: after a byte decoder, streaming protocol output requires Fuse before contextual Strip; other context-changing stages need explicit streaming support"
                ),
            }
        }
        if transforms
            .iter()
            .any(|s| matches!(s, StreamTransformSpec::Replace(_)))
        {
            transforms.insert(
                0,
                if byte_level {
                    StreamTransformSpec::Utf8
                } else {
                    StreamTransformSpec::Fallback
                },
            );
        }
        let spec_bytes = transforms.iter().try_fold(0usize, |sum, stage| {
            let bytes = size_of::<StreamTransformSpec>()
                .checked_add(
                    match stage {
                        StreamTransformSpec::Replace(spec) => spec
                            .pattern
                            .len()
                            .checked_mul(size_of::<usize>() + 1)
                            .and_then(|n| n.checked_add(spec.content.len()))
                            .and_then(|n| n.checked_add(spec.path.len())),
                        _ => Some(0),
                    }
                    .context("composed decoder specification size overflow")?,
                )
                .context("composed decoder specification size overflow")?;
            sum.checked_add(bytes)
                .context("composed decoder specification size overflow")
        })?;
        ensure!(
            spec_bytes <= 4 * 1024 * 1024,
            "tokenizer.decoder composed transform specifications exceed the 4 MiB contract budget; reduce stages, literal patterns or replacement content"
        );
        let catalog_limit = if transforms
            .iter()
            .any(|s| matches!(s, StreamTransformSpec::Replace(_)))
        {
            Some(literal_catalog_limit(tokenizer)?)
        } else {
            None
        };
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
        let mut kinds = Vec::with_capacity(size);
        let mut catalog_bytes = size
            .checked_mul(64)
            .context("literal byte catalogue index size overflow")?;
        for index in 0..size {
            let id = u32::try_from(index).context("byte-stream vocabulary exceeds token IDs")?;
            if policy == SpecialTokenHandling::Skip
                && specials.get(&id).is_some_and(|token| token.special)
            {
                pieces.push(Vec::new());
                kinds.push(BytePieceKind::Skipped);
                continue;
            }
            let mut token=tokenizer.inner().id_to_token(id).with_context(||format!("tokenizer.model.vocab is missing byte token ID {id}; repair the dense vocabulary"))?;
            for (path, stage) in &stages[..byte_index] {
                if let Some(limit) = catalog_limit {
                    let bound = match stage {
                        DecoderWrapper::Replace(replace) => token
                            .len()
                            .checked_add(1)
                            .and_then(|n| n.checked_mul(replace.content.len()))
                            .and_then(|n| n.checked_add(token.len())),
                        _ => Some(token.len()),
                    }
                    .context("token-local decoder output bound overflow")?;
                    ensure!(
                        bound
                            .checked_mul(2)
                            .and_then(|n| n.checked_add(catalog_bytes))
                            .is_some_and(|n| n <= limit),
                        "{path} token {id} replacement upper bound exceeds the {limit}-byte literal vocabulary workspace; shorten replacement content or token-local stages"
                    );
                    ensure!(
                        token
                            .len()
                            .checked_mul(96)
                            .and_then(|n| n.checked_add(48))
                            .is_some_and(|n| n <= 16 * 1024 * 1024),
                        "{path} token {id} needs more than 16 MiB canonical match scratch; shorten the token or replace token-local regex preprocessing"
                    );
                }
                let mut transformed = onnx_genai_llguidance::panic_utils::catch_unwind(
                    std::panic::AssertUnwindSafe(|| {
                        stage
                            .decode_chain(vec![token])
                            .map_err(anyhow::Error::from_boxed)
                    }),
                )
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
            let fallback_byte = !byte_level
                && token.len() == 6
                && token.starts_with("<0x")
                && token.ends_with('>')
                && u8::from_str_radix(&token[3..5], 16).is_ok();
            kinds.push(if fallback_byte {
                BytePieceKind::FallbackBytes
            } else {
                BytePieceKind::Text
            });
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
            if let Some(limit) = catalog_limit {
                catalog_bytes = catalog_bytes
                    .checked_add(piece.capacity())
                    .context("literal byte catalogue size overflow")?;
                ensure!(
                    catalog_bytes <= limit,
                    "literal decoded vocabulary exceeds its {limit}-byte compiled catalogue budget; simplify token-local expansion"
                );
            }
            pieces.push(piece);
        }
        Ok(Some(Self {
            pieces,
            kinds,
            transforms,
        }))
    }

    pub(crate) fn has_literal_replace(&self) -> bool {
        self.transforms
            .iter()
            .any(|stage| matches!(stage, StreamTransformSpec::Replace(_)))
    }
    /// Publication needs delayed literal-prefix handling only for the resolved
    /// fused literal subset; other canonical decoders keep their existing path.
    pub(crate) fn requests_literal_stream(tokenizer: &Tokenizer) -> anyhow::Result<bool> {
        struct Capability {
            byte: bool,
            fused: bool,
            literal: bool,
            supported: bool,
        }
        fn visit(stage: &DecoderWrapper, state: &mut Capability) -> anyhow::Result<()> {
            if let DecoderWrapper::Sequence(sequence) = stage {
                for stage in sequence.get_decoders() {
                    visit(stage, state)?;
                }
                return Ok(());
            }
            match stage {
                DecoderWrapper::ByteLevel(_) if !state.byte => {
                    state.byte = true;
                    state.fused = true;
                }
                DecoderWrapper::ByteFallback(_) if !state.byte => state.byte = true,
                DecoderWrapper::Replace(_) | DecoderWrapper::Strip(_) if !state.byte => {}
                DecoderWrapper::Fuse(_) if state.byte => state.fused = true,
                DecoderWrapper::Strip(strip)
                    if state.byte && (state.fused || (strip.start == 0 && strip.stop == 0)) => {}
                DecoderWrapper::Replace(replace) if state.byte && state.fused => {
                    if replacement_kind(replace)? == ReplacePatternKind::Literal {
                        state.literal = true;
                    } else {
                        state.supported = false;
                    }
                }
                _ => state.supported = false,
            }
            Ok(())
        }
        let mut state = Capability {
            byte: false,
            fused: false,
            literal: false,
            supported: true,
        };
        if let Some(decoder) = tokenizer.inner().get_decoder() {
            visit(decoder, &mut state)?;
        }
        Ok(state.byte && state.literal && state.supported)
    }
    pub(crate) fn has_trailing_strip(&self) -> bool {
        self.transforms
            .iter()
            .any(|strip| matches!(strip, StreamTransformSpec::Trailing { .. }))
    }
    /// Prove a marker is visible after its final token in the given prefix
    /// context. Initial transforms must not consume it, and no suffix may delay it.
    pub(crate) fn marker_stable(&self, tokens: &[u32], prefix: &[u32]) -> anyhow::Result<bool> {
        let mut state = self.new_state();
        for &id in prefix {
            let piece = self
                .pieces
                .get(id as usize)
                .context("reasoning prefix token is absent from the resolved byte vocabulary")?;
            let _ = state.step(piece, self.kinds[id as usize], 4 * 1024 * 1024)?;
        }
        let mut expected = Vec::new();
        let mut published = Vec::new();
        for &id in tokens {
            let piece = self
                .pieces
                .get(id as usize)
                .context("reasoning marker token is absent from the resolved byte vocabulary")?;
            expected.extend_from_slice(piece);
            published.extend_from_slice(&state.step(
                piece,
                self.kinds[id as usize],
                4 * 1024 * 1024,
            )?);
        }
        Ok(published.ends_with(&expected) && state.retained_bytes() == 0)
    }

    pub(crate) fn new_state(&self) -> ByteStreamState {
        ByteStreamState {
            transforms: self
                .transforms
                .iter()
                .map(|strip| match strip {
                    StreamTransformSpec::Fallback => StreamTransformState::Fallback {
                        held: FallbackRun::default(),
                    },
                    StreamTransformSpec::Utf8 => StreamTransformState::Utf8 { held: Vec::new() },
                    StreamTransformSpec::Replace(spec) => StreamTransformState::Replace {
                        spec: spec.clone(),
                        matched: 0,
                    },
                    StreamTransformSpec::Initial { character, count } => {
                        StreamTransformState::Initial(StripState::Matching {
                            character: *character,
                            remaining: *count,
                            matched: Vec::new(),
                        })
                    }
                    StreamTransformSpec::Trailing { character, count } => {
                        StreamTransformState::Trailing {
                            character: *character,
                            count: *count,
                            held: VecDeque::new(),
                        }
                    }
                })
                .collect(),
        }
    }
}

impl ByteStreamState {
    pub(crate) fn has_pending_prefix(&self) -> bool {
        self.transforms.iter().any(|strip| match strip {
            StreamTransformState::Fallback { held } => !held.is_empty(),
            StreamTransformState::Utf8 { held } => !held.is_empty(),
            StreamTransformState::Replace { matched, .. } => *matched > 0,
            StreamTransformState::Initial(StripState::Matching { matched, .. }) => {
                !matched.is_empty()
            }
            StreamTransformState::Trailing {
                character, held, ..
            } => held.len() % character.len_utf8() != 0,
            _ => false,
        })
    }
    pub(crate) fn is_identity(&self) -> bool {
        self.transforms
            .iter()
            .all(|strip| matches!(strip, StreamTransformState::Initial(StripState::Done)))
    }
    pub(crate) fn has_literal_replace(&self) -> bool {
        self.transforms
            .iter()
            .any(|stage| matches!(stage, StreamTransformState::Replace { .. }))
    }
    pub(crate) fn retained_bytes(&self) -> usize {
        self.transforms
            .iter()
            .map(|strip| match strip {
                StreamTransformState::Fallback { held } => held.len(),
                StreamTransformState::Utf8 { held } => held.len(),
                StreamTransformState::Replace { matched, .. } => *matched,
                StreamTransformState::Initial(StripState::Matching { matched, .. }) => {
                    matched.len()
                }
                StreamTransformState::Trailing { held, .. } => held.len(),
                _ => 0,
            })
            .sum()
    }
    pub(crate) fn storage_bytes(&self) -> usize {
        self.retained_bytes() + self.transforms.len() * std::mem::size_of::<StreamTransformState>()
    }
    /// Raw continuations which can release only part of a delayed suffix.
    /// Include completion of split matching scalars as well as full characters.
    pub(crate) fn suffix_continuations(&self) -> Vec<Vec<u8>> {
        let mut pieces = std::collections::BTreeSet::new();
        let single_suffix = self
            .transforms
            .iter()
            .filter(|strip| matches!(strip, StreamTransformState::Trailing { .. }))
            .count()
            == 1
            && self.transforms.iter().all(|strip| {
                !matches!(
                    strip,
                    StreamTransformState::Initial(StripState::Matching { .. })
                )
            });
        for strip in &self.transforms {
            let (character, partial) = match strip {
                StreamTransformState::Initial(StripState::Matching {
                    character, matched, ..
                }) => (*character, matched.len()),
                StreamTransformState::Trailing {
                    character, held, ..
                } => (*character, held.len() % character.len_utf8()),
                _ => continue,
            };
            let mut encoded = [0; 4];
            let target = character.encode_utf8(&mut encoded).as_bytes();
            let mut jumped = false;
            if let StreamTransformState::Trailing { count, held, .. } = strip
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
    pub(crate) fn step<'a>(
        &mut self,
        bytes: &'a [u8],
        kind: BytePieceKind,
        limit: usize,
    ) -> anyhow::Result<Cow<'a, [u8]>> {
        let mut work = StreamWork::new(limit.saturating_mul(64).max(4096));
        self.step_with_work(bytes, kind, limit, &mut work)
    }
    pub(crate) fn step_with_work<'a>(
        &mut self,
        bytes: &'a [u8],
        kind: BytePieceKind,
        limit: usize,
        work: &mut StreamWork,
    ) -> anyhow::Result<Cow<'a, [u8]>> {
        work.charge(self.storage_bytes(), "tokenizer.decoder state fork")?;
        if self.has_literal_replace() {
            let mut next = self.clone();
            let output = next.step_inner(bytes, kind, limit, work)?;
            *self = next;
            return Ok(output);
        }
        self.step_inner(bytes, kind, limit, work)
    }
    fn step_inner<'a>(
        &mut self,
        bytes: &'a [u8],
        kind: BytePieceKind,
        limit: usize,
        work: &mut StreamWork,
    ) -> anyhow::Result<Cow<'a, [u8]>> {
        if kind == BytePieceKind::Skipped {
            return Ok(Cow::Borrowed(&[]));
        }
        if self.is_identity() {
            ensure!(
                bytes.len() <= limit,
                "decoded byte delta exceeds its {limit}-byte response budget; increase the response buffer"
            );
            return Ok(Cow::Borrowed(bytes));
        }
        ensure!(
            bytes.len() <= limit,
            "raw byte delta exceeds its {limit}-byte decoder budget; increase the response buffer"
        );
        work.charge(bytes.len(), "tokenizer.decoder input copy")?;
        let mut output = Vec::new();
        output.try_reserve_exact(bytes.len()).context(
            "allocate bounded decoder input delta; free memory or reduce response buffer",
        )?;
        output.extend_from_slice(bytes);
        for strip in &mut self.transforms {
            output = if let StreamTransformState::Fallback { held } = strip {
                if kind == BytePieceKind::FallbackBytes {
                    work.copies(
                        held.len()
                            .checked_add(output.len())
                            .context("ByteFallback run work overflow")?,
                        4,
                        "tokenizer.decoder ByteFallback run",
                    )?;
                    *held = held.appended(&output, limit)?;
                    Vec::new()
                } else {
                    work.copies(
                        held.output_length()?
                            .checked_add(output.len())
                            .context("ByteFallback boundary work overflow")?,
                        4,
                        "tokenizer.decoder ByteFallback boundary",
                    )?;
                    let mut joined = fallback_output(held, limit)?;
                    ensure!(
                        joined
                            .len()
                            .checked_add(output.len())
                            .is_some_and(|n| n <= limit),
                        "ByteFallback boundary exceeds its {limit}-byte response budget; increase buffer"
                    );
                    joined.try_reserve_exact(output.len()).context("grow bounded ByteFallback boundary delta; free memory or reduce response buffer")?;
                    joined.extend(output);
                    *held = FallbackRun::default();
                    joined
                }
            } else {
                strip.apply(&output, limit, work)?
            };
        }
        Ok(Cow::Owned(output))
    }
    fn flush(
        &mut self,
        terminal: bool,
        limit: usize,
        work: &mut StreamWork,
    ) -> anyhow::Result<Vec<u8>> {
        work.charge(
            self.storage_bytes(),
            "tokenizer.decoder terminal state scan",
        )?;
        let mut output = Vec::new();
        for strip in &mut self.transforms {
            let mut next = if matches!(strip, StreamTransformState::Fallback { .. }) {
                output
            } else {
                strip.apply(&output, limit, work)?
            };
            let tail_bound = match &*strip {
                StreamTransformState::Fallback { held } if terminal => held.output_length()?,
                StreamTransformState::Utf8 { held } if terminal && !held.is_empty() => 3,
                StreamTransformState::Replace { matched, .. } => *matched,
                StreamTransformState::Initial(StripState::Matching { matched, .. }) if terminal => {
                    matched.len()
                }
                StreamTransformState::Trailing {
                    character, held, ..
                } if !terminal || held.len() % character.len_utf8() != 0 => held.len(),
                _ => 0,
            };
            ensure!(
                next.len()
                    .checked_add(tail_bound)
                    .is_some_and(|n| n <= limit),
                "decoder EOF exceeds its {limit}-byte response budget; increase the response buffer"
            );
            work.copies(tail_bound, 4, "tokenizer.decoder EOF release")?;
            let tail = if let StreamTransformState::Fallback { held } = strip {
                if terminal {
                    let tail = fallback_output(held, limit.saturating_sub(next.len()))?;
                    *held = FallbackRun::default();
                    tail
                } else {
                    Vec::new()
                }
            } else {
                strip.finish(terminal)
            };
            ensure!(
                next.len()
                    .checked_add(tail.len())
                    .is_some_and(|n| n <= limit),
                "decoder EOF exceeds its {limit}-byte response budget; increase the response buffer"
            );
            next.extend(tail);
            output = next;
        }
        Ok(output)
    }
    /// Bytes which must become visible if more non-stripped text follows.
    /// Do not confuse this provisional suffix with terminal stripped output.
    pub(crate) fn provisional_tail(&self, limit: usize) -> anyhow::Result<Vec<u8>> {
        self.clone().flush(
            false,
            limit,
            &mut StreamWork::new(limit.saturating_mul(64).max(4096)),
        )
    }
    pub(crate) fn finish(&mut self, limit: usize) -> anyhow::Result<Vec<u8>> {
        self.finish_with_work(
            limit,
            &mut StreamWork::new(limit.saturating_mul(64).max(4096)),
        )
    }
    pub(crate) fn finish_with_work(
        &mut self,
        limit: usize,
        work: &mut StreamWork,
    ) -> anyhow::Result<Vec<u8>> {
        work.charge(self.storage_bytes(), "tokenizer.decoder EOF fork")?;
        let mut next = self.clone();
        let output = next.flush(true, limit, work)?;
        *self = next;
        Ok(output)
    }
}

fn fallback_output(run: &FallbackRun, limit: usize) -> anyhow::Result<Vec<u8>> {
    let length = run.output_length()?;
    ensure!(
        length <= limit,
        "ByteFallback replacement run exceeds its {limit}-byte response budget; increase buffer or repair byte sequence"
    );
    let mut output = Vec::new();
    output.try_reserve_exact(length).context(
        "allocate bounded ByteFallback boundary output; free memory or shorten generation",
    )?;
    if !run.invalid && std::str::from_utf8(&run.bytes).is_ok() {
        output.extend_from_slice(&run.bytes);
    } else {
        for _ in 0..run.length {
            output.extend_from_slice("�".as_bytes());
        }
    }
    Ok(output)
}

fn replacement_byte(
    spec: &LiteralReplacement,
    matched: &mut usize,
    byte: u8,
    mut emit: impl FnMut(&[u8]) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    while *matched > 0 && spec.pattern[*matched] != byte {
        let next = spec.failure[*matched - 1];
        emit(&spec.pattern[..*matched - next])?;
        *matched = next;
    }
    if spec.pattern[*matched] == byte {
        *matched += 1;
        if *matched == spec.pattern.len() {
            emit(&spec.content)?;
            *matched = 0;
        }
    } else {
        emit(&[byte])?;
    }
    Ok(())
}

impl StreamTransformState {
    fn apply(
        &mut self,
        bytes: &[u8],
        limit: usize,
        work: &mut StreamWork,
    ) -> anyhow::Result<Vec<u8>> {
        if let Self::Utf8 { held } = self {
            let length = held
                .len()
                .checked_add(bytes.len())
                .context("canonical UTF8 input overflow")?;
            work.copies(length, 12, "tokenizer.decoder canonical UTF8")?;
            let mut input = Vec::new();
            input.try_reserve_exact(length).context(
                "allocate bounded canonical UTF8 input; free memory or shorten response",
            )?;
            input.extend_from_slice(held);
            input.extend_from_slice(bytes);
            let mut length = 0usize;
            let mut at = 0;
            while at < input.len() {
                match std::str::from_utf8(&input[at..]) {
                    Ok(valid) => {
                        length = length
                            .checked_add(valid.len())
                            .context("canonical UTF8 length overflow")?;
                        break;
                    }
                    Err(error) => {
                        length = length
                            .checked_add(error.valid_up_to())
                            .context("canonical UTF8 length overflow")?;
                        at += error.valid_up_to();
                        if let Some(n) = error.error_len() {
                            length = length
                                .checked_add(3)
                                .context("canonical UTF8 replacement overflow")?;
                            at += n;
                        } else {
                            break;
                        }
                    }
                }
            }
            ensure!(
                length <= limit,
                "canonical UTF8 output exceeds its {limit}-byte decoder budget; increase response buffer"
            );
            let mut output = Vec::new();
            output
                .try_reserve_exact(length)
                .context("allocate bounded canonical UTF8 delta; free memory or reduce output")?;
            at = 0;
            held.clear();
            while at < input.len() {
                match std::str::from_utf8(&input[at..]) {
                    Ok(valid) => {
                        output.extend_from_slice(valid.as_bytes());
                        break;
                    }
                    Err(error) => {
                        output.extend_from_slice(&input[at..at + error.valid_up_to()]);
                        at += error.valid_up_to();
                        if let Some(n) = error.error_len() {
                            output.extend_from_slice("�".as_bytes());
                            at += n;
                        } else {
                            held.extend_from_slice(&input[at..]);
                            break;
                        }
                    }
                }
            }
            return Ok(output);
        }
        if let Self::Replace { spec, matched } = self {
            // Count on a private scalar state before allocating or mutating live state.
            work.copies(
                bytes
                    .len()
                    .checked_add(*matched)
                    .context("literal Replace work overflow")?,
                4,
                &spec.path,
            )?;
            let mut cursor = *matched;
            let mut length = 0usize;
            for &byte in bytes {
                replacement_byte(spec, &mut cursor, byte, |part| {
                    work.charge(part.len(), &spec.path)?;
                    length = length
                        .checked_add(part.len())
                        .context("Replace output length overflow")?;
                    ensure!(
                        length <= limit,
                        "{} replacement expands past the {limit}-byte response budget; shorten content or increase the response buffer",
                        spec.path
                    );
                    Ok(())
                })?;
            }
            work.charge(length, &spec.path)?;
            let mut output = Vec::new();
            output.try_reserve_exact(length).with_context(||format!("allocate bounded replacement output at {}; reduce response budget/content or free memory",spec.path))?;
            for &byte in bytes {
                replacement_byte(spec, matched, byte, |part| {
                    output.extend_from_slice(part);
                    Ok(())
                })?;
            }
            return Ok(output);
        }
        ensure!(
            bytes
                .len()
                .checked_add(match self {
                    Self::Initial(StripState::Matching { matched, .. }) => matched.len(),
                    Self::Trailing { held, .. } => held.len(),
                    _ => 0,
                })
                .is_some_and(|n| n <= limit),
            "Strip input plus retained bytes exceeds its {limit}-byte decoder budget; increase the response buffer"
        );
        let retained = match self {
            Self::Initial(StripState::Matching { matched, .. }) => matched.len(),
            Self::Trailing { held, .. } => held.len(),
            _ => 0,
        };
        work.copies(
            bytes
                .len()
                .checked_add(retained)
                .context("Strip work overflow")?,
            4,
            "tokenizer.decoder Strip",
        )?;
        Ok(self.apply_strip(bytes))
    }
    fn apply_strip(&mut self, bytes: &[u8]) -> Vec<u8> {
        match self {
            Self::Fallback { .. } | Self::Utf8 { .. } | Self::Replace { .. } => {
                unreachable!("Replace handled by bounded transform")
            }
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
            Self::Fallback { .. } => unreachable!("ByteFallback EOF uses bounded fallible flush"),
            Self::Utf8 { held } => {
                if terminal && !held.is_empty() {
                    held.clear();
                    "�".as_bytes().to_vec()
                } else {
                    Vec::new()
                }
            }
            Self::Replace { spec, matched } => {
                let tail = spec.pattern[..*matched].to_vec();
                *matched = 0;
                tail
            }
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
            output.extend_from_slice(
                &state
                    .step(
                        &contract.pieces[id as usize],
                        contract.kinds[id as usize],
                        4096,
                    )
                    .unwrap(),
            );
            let mut at_end = state.clone();
            let mut bytes = output.clone();
            bytes.extend(at_end.finish(4096).unwrap());
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
        output.extend(state.finish(4096).unwrap());
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
    fn composed_prefix_transforms_hold_unicode_and_compose_in_declared_order() {
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
    fn trailing_transforms_match_fused_decoder_and_hold_only_a_bounded_suffix() {
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
                        output.extend_from_slice(
                            &state.step(bytes, BytePieceKind::Text, 4096).unwrap(),
                        );
                        assert!(state.retained_bytes() < 3 * character.len_utf8());
                    }
                    let mut preview = output.clone();
                    preview.extend(state.provisional_tail(4096).unwrap());
                    assert_eq!(preview, text.as_bytes());
                    output.extend(state.finish(4096).unwrap());
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
    fn composed_trailing_transforms_match_canonical_decoding_without_retracting_bytes() {
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
                        output.extend_from_slice(
                            &state.step(bytes, BytePieceKind::Text, 4096).unwrap(),
                        );
                        assert!(expected.as_bytes().starts_with(&output));
                    }
                    output.extend(state.finish(4096).unwrap());
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
    fn trailing_transforms_preserve_partial_unicode_and_account_for_a_full_window() {
        let tokenizer = tokenizer(
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},
            {"type":"Strip","content":"é","start":0,"stop":1}]}),
            &[],
        );
        let contract = ByteStreamContract::resolve(&tokenizer, SpecialTokenHandling::Skip)
            .unwrap()
            .unwrap();
        let mut state = contract.new_state();
        assert_eq!(
            &*state.step(b"abc\xc3", BytePieceKind::Text, 4096).unwrap(),
            b"abc"
        );
        assert!(state.has_pending_prefix());
        assert_eq!(state.provisional_tail(4096).unwrap(), b"\xc3");
        assert_eq!(state.clone().finish(4096).unwrap(), b"\xc3");
        assert!(state.suffix_continuations().contains(&vec![0xa9]));
        assert!(
            state
                .step(b"\xa9", BytePieceKind::Text, 4096)
                .unwrap()
                .is_empty()
        );
        assert!(!state.has_pending_prefix());
        assert!(
            state
                .step(b"\xc3", BytePieceKind::Text, 4096)
                .unwrap()
                .is_empty()
        );
        let mut completion = state.clone();
        assert_eq!(
            &*completion.step(&[0xa9], BytePieceKind::Text, 4096).unwrap(),
            "é".as_bytes()
        );
        assert!(completion.finish(4096).unwrap().is_empty());
        assert!(
            state
                .check_buffer(2)
                .unwrap_err()
                .is::<ByteStreamBufferExceeded>()
        );
        assert_eq!(
            &*state.step(b"\xa9", BytePieceKind::Text, 4096).unwrap(),
            "é".as_bytes()
        );
        assert!(state.finish(4096).unwrap().is_empty());
    }

    #[test]
    fn byte_stream_contract_rejects_unresolved_chunk_local_stages_with_paths() {
        for decoder in [
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Strip","content":" ","start":1,"stop":0}]}),
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"Regex":"x+"},"content":"y"}]}),
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
    #[test]
    fn fused_literal_replace_matches_canonical_decoder_at_every_byte_split() {
        for (pattern, content, text) in [
            ("aa", "X", "aaaaa"),
            ("aba", "$1", "abababa"),
            ("é終", "Y", "aé終é"),
            ("x", "", "xaxx"),
            ("END", "end", "EN"),
        ] {
            let decoder = json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"String":pattern},"content":content}]});
            let tokenizer = self::tokenizer(decoder, &[]);
            let contract = ByteStreamContract::resolve(&tokenizer, SpecialTokenHandling::Skip)
                .unwrap()
                .unwrap();
            let expected = tokenizer
                .inner()
                .get_decoder()
                .unwrap()
                .decode(vec![text.to_owned()])
                .unwrap();
            for split in 0..=text.len() {
                let mut state = contract.new_state();
                let mut output = state
                    .step(
                        &text.as_bytes()[..split],
                        BytePieceKind::FallbackBytes,
                        4096,
                    )
                    .unwrap()
                    .into_owned();
                output.extend_from_slice(
                    &state
                        .step(
                            &text.as_bytes()[split..],
                            BytePieceKind::FallbackBytes,
                            4096,
                        )
                        .unwrap(),
                );
                output.extend(state.finish(4096).unwrap());
                assert_eq!(output, expected.as_bytes(), "{pattern:?}, split={split}");
            }
        }
    }
    #[test]
    fn literal_replace_composes_in_order_and_rejects_expansion_before_mutation() {
        let tokenizer = self::tokenizer(
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"String":"ab"},"content":"xx"},{"type":"Replace","pattern":{"String":"xx"},"content":"END"},{"type":"Strip","content":"D","start":0,"stop":1}]}),
            &[],
        );
        let contract = ByteStreamContract::resolve(&tokenizer, SpecialTokenHandling::Skip)
            .unwrap()
            .unwrap();
        let mut state = contract.new_state();
        assert!(
            state
                .step(b"a", BytePieceKind::Text, 4096)
                .unwrap()
                .is_empty()
        );
        let mut output = state
            .step(b"b", BytePieceKind::Text, 4096)
            .unwrap()
            .into_owned();
        output.extend(state.finish(4096).unwrap());
        assert_eq!(output, b"EN");
        let tokenizer = self::tokenizer(
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"String":"x"},"content":"12345678"}]}),
            &[],
        );
        let contract = ByteStreamContract::resolve(&tokenizer, SpecialTokenHandling::Skip)
            .unwrap()
            .unwrap();
        let mut state = contract.new_state();
        let original = state.clone();
        assert!(
            state
                .step(b"x", BytePieceKind::Text, 7)
                .unwrap_err()
                .to_string()
                .contains("expands past")
        );
        assert!(state == original);
        assert_eq!(
            &*state.step(b"x", BytePieceKind::Text, 8).unwrap(),
            b"12345678"
        );
    }

    #[test]
    fn literal_stream_preserves_canonical_byte_groups_and_special_boundaries() {
        for byte_level in [false, true] {
            let byte_decoder = if byte_level {
                json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false})
            } else {
                json!({"type":"ByteFallback"})
            };
            let overrides = if byte_level {
                [(4, "Ã"), (5, "©"), (6, "ÿ"), (7, "Z")]
            } else {
                [(4, "<0xC3>"), (5, "<0xA9>"), (6, "<0xFF>"), (7, "Z")]
            };
            let tokenizer = self::tokenizer(
                json!({"type":"Sequence","decoders":[byte_decoder,{"type":"Fuse"},{"type":"Replace","pattern":{"String":"�"},"content":"!"}]}),
                &overrides,
            );
            for policy in [SpecialTokenHandling::Skip, SpecialTokenHandling::Preserve] {
                let contract = ByteStreamContract::resolve(&tokenizer, policy)
                    .unwrap()
                    .unwrap();
                for ids in [
                    vec![4],
                    vec![4, 5],
                    vec![4, 5, 6],
                    vec![4, 5, 6, 7],
                    vec![4, 3, 5],
                    vec![4, 5, 3, 6],
                    vec![4, 7, 5],
                ] {
                    let expected = if policy == SpecialTokenHandling::Skip {
                        tokenizer.decode(&ids).unwrap()
                    } else {
                        tokenizer.decode_with_special_tokens(&ids).unwrap()
                    };
                    let mut state = contract.new_state();
                    let mut output = Vec::new();
                    for &id in &ids {
                        output.extend_from_slice(
                            &state
                                .step(
                                    &contract.pieces[id as usize],
                                    contract.kinds[id as usize],
                                    4096,
                                )
                                .unwrap(),
                        );
                    }
                    output.extend(state.finish(4096).unwrap());
                    assert_eq!(
                        output,
                        expected.as_bytes(),
                        "bytelevel={byte_level} ids={ids:?} policy={policy:?}"
                    );
                }
            }
        }
    }
    #[test]
    fn replacement_second_stage_failure_preserves_the_entire_fork() {
        let tokenizer = self::tokenizer(
            json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"String":"ab"},"content":"x"},{"type":"Replace","pattern":{"String":"x"},"content":"12345678"}]}),
            &[],
        );
        let contract = ByteStreamContract::resolve(&tokenizer, SpecialTokenHandling::Skip)
            .unwrap()
            .unwrap();
        let mut state = contract.new_state();
        state.step(b"a", BytePieceKind::Text, 4096).unwrap();
        let before = state.clone();
        assert!(state.step(b"b", BytePieceKind::Text, 7).is_err());
        assert!(state == before);
        assert_eq!(
            &*state.step(b"b", BytePieceKind::Text, 8).unwrap(),
            b"12345678"
        );
    }

    #[test]
    fn literal_work_budget_counts_intermediate_expansion_before_later_deletion() {
        let tokenizer = self::tokenizer(
            json!({"type":"Sequence","decoders":[{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},{"type":"Replace","pattern":{"String":"a"},"content":"b".repeat(2048)},{"type":"Replace","pattern":{"String":"b"},"content":""}]}),
            &[],
        );
        let contract = ByteStreamContract::resolve(&tokenizer, SpecialTokenHandling::Skip)
            .unwrap()
            .unwrap();
        let mut state = contract.new_state();
        let before = state.clone();
        let mut work = StreamWork::new(10_000);
        let error = state
            .step_with_work(b"a", BytePieceKind::Text, 4096, &mut work)
            .unwrap_err();
        assert!(error.is::<StreamWorkExceeded>(), "{error:#}");
        assert!(state == before);
        assert!(
            state
                .step_with_work(
                    b"a",
                    BytePieceKind::Text,
                    4096,
                    &mut StreamWork::new(100_000)
                )
                .unwrap()
                .is_empty()
        );
    }
}
