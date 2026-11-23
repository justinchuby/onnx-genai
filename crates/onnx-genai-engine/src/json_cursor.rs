//! Reversible byte-oriented JSON Schema recognition for projected response text.
use anyhow::{Context, ensure};
use llguidance::api::TopLevelGrammar;
use llguidance::earley::{BiasComputer, ParserRecognizer};
use llguidance::toktrie::{
    ApproximateTokEnv, InferenceCapabilities, SimpleVob, TokEnv, TokRxInfo, TokTrie,
};
use llguidance::{ParserFactory, TokenParser};
use std::{fmt, num::NonZeroUsize, sync::Arc};

/// Recognition of a candidate after the committed logical JSON bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonCandidate {
    /// Every byte is viable; complete additionally permits grammar EOF.
    Viable { complete: bool },
    /// The first rejected byte's offset in the combined logical JSON text.
    Rejected { byte_offset: usize },
}

#[derive(Debug, thiserror::Error)]
#[error(
    "JSON byte candidate would exceed the {limit}-byte response budget at committed offset {offset}; shorten the response or increase its explicit budget"
)]
pub(crate) struct JsonByteBudgetExceeded {
    limit: usize,
    offset: usize,
}

/// A request-local compiled schema recognizer independent of model token IDs.
///
/// Bytes use the backend's JSON lexical language. Response-channel projection
/// and escape normalization belong before this cursor. User stops and length
/// limits do not require completeness; grammar EOF does.
pub struct JsonByteCursor {
    matcher: ByteMatcher,
    initial: ByteMatcher,
    committed: Vec<u32>,
    max_bytes: NonZeroUsize,
    failure: Option<CursorFailure>,
}

/// A model vocabulary walked against the cursor's already compiled grammar.
/// Model IDs never enter the byte parser's history or its internal byte mask.
pub(crate) struct JsonTokenVocabulary(TokTrie);

impl JsonTokenVocabulary {
    pub(crate) fn new(pieces: &[Vec<u8>]) -> anyhow::Result<Self> {
        let mut pieces = pieces.to_vec();
        let eos = u32::try_from(pieces.len()).context("JSON model vocabulary exceeds token IDs")?;
        pieces.push(b"\xff<json-eos>".to_vec());
        let info = TokRxInfo {
            vocab_size: u32::try_from(pieces.len())?,
            tok_eos: eos,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        Ok(Self(TokTrie::from(&info, &pieces)))
    }
}

impl BiasComputer for JsonTokenVocabulary {
    fn compute_bias(&self, recognizer: &mut ParserRecognizer<'_>, start: &[u8]) -> SimpleVob {
        let mut mask = self.0.alloc_token_set();
        self.0.add_bias(recognizer, &mut mask, start);
        mask
    }
    fn trie(&self) -> &TokTrie {
        &self.0
    }
}

/// Keeps parser access panic-fenced while permitting a bounded external trie
/// walk. The public Matcher wrapper does not expose this backend operation.
enum ByteMatcher {
    Ready(Box<TokenParser>),
    Failed(ByteMatcherFailure),
}

#[derive(Debug, Clone)]
struct ByteMatcherFailure(Arc<anyhow::Error>);
impl fmt::Display for ByteMatcherFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "JSON recognizer backend operation failed: {:#}; retry with a fresh cursor",
            self.0
        )
    }
}
impl std::error::Error for ByteMatcherFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref().as_ref())
    }
}

impl ByteMatcher {
    fn new(mut parser: TokenParser) -> Self {
        parser.start_without_prompt();
        Self::Ready(Box::new(parser))
    }
    fn deep_clone(&self) -> Self {
        match self {
            Self::Ready(parser) => Self::Ready(Box::new(parser.deep_clone())),
            Self::Failed(failure) => Self::Failed(failure.clone()),
        }
    }
    fn get_error(&mut self) -> Option<ByteMatcherFailure> {
        let _ = self.run(|parser| {
            if let Some(error) = parser.parser.get_error() {
                anyhow::bail!("{}", error.message());
            }
            Ok(())
        });
        match self {
            Self::Failed(failure) => Some(failure.clone()),
            Self::Ready(_) => None,
        }
    }
    fn is_stopped(&self) -> bool {
        match self {
            Self::Ready(parser) => parser.stopped(),
            Self::Failed(_) => true,
        }
    }
    fn run<T>(
        &mut self,
        operation: impl FnOnce(&mut TokenParser) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let result = match self {
            Self::Failed(failure) => return Err(failure.clone().into()),
            Self::Ready(parser) => {
                llguidance::panic_utils::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    operation(parser)
                }))
            }
        };
        match result {
            Ok(value) => Ok(value),
            Err(operation) => {
                let failure = ByteMatcherFailure(Arc::new(operation));
                *self = Self::Failed(failure.clone());
                Err(failure.into())
            }
        }
    }
    fn try_consume_tokens(&mut self, tokens: &[u32]) -> anyhow::Result<usize> {
        self.run(|parser| {
            for (index, &token) in tokens.iter().enumerate() {
                if !parser.validate_token(token)? { return Ok(index); }
                let backtrack = parser.consume_token(token)?;
                parser.check_stop()?;
                ensure!(backtrack == 0, "JSON byte recognizer requested {backtrack} token backtracks; use a grammar supporting append-only byte recognition");
            }
            Ok(tokens.len())
        })
    }
    fn rollback(&mut self, count: usize) -> anyhow::Result<()> {
        self.run(|parser| parser.rollback(count))
    }
    fn is_accepting(&mut self) -> anyhow::Result<bool> {
        self.run(|parser| Ok(parser.is_accepting()))
    }
    fn vocabulary_mask(&mut self, vocabulary: &JsonTokenVocabulary) -> anyhow::Result<SimpleVob> {
        self.run(|parser| {
            if parser.stopped() { return Ok(vocabulary.trie().alloc_token_set()); }
            let forced_prefix = parser.force_bytes();
            // The backend cache keys grammar state, not vocabulary identity.
            // Neither model-ID masks nor byte-ID masks may leak across walks.
            parser.invalidate_bias_cache();
            let mask = parser.parser.compute_bias(vocabulary, &forced_prefix);
            parser.invalidate_bias_cache();
            if let Some(error) = parser.parser.get_error() {
                anyhow::bail!("JSON vocabulary recognition failed: {}; repair the schema or increase its explicit grammar limits", error.message());
            }
            Ok(mask)
        })
    }
    #[cfg(test)]
    fn test_trigger_lexer_error(&mut self) -> anyhow::Result<()> {
        self.run(|parser| parser.parser.test_trigger_lexer_error())
    }
}

#[derive(Debug, Clone)]
struct CursorFailure(Arc<anyhow::Error>);
impl fmt::Display for CursorFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "JSON byte cursor could not restore its committed prefix: {:#}; create a fresh cursor",
            self.0
        )
    }
}
impl std::error::Error for CursorFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref().as_ref())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("JSON candidate operation failed and committed-prefix recovery also failed: {recovery:#}")]
struct RecoveryFailure {
    #[source]
    operation: anyhow::Error,
    recovery: anyhow::Error,
}

struct Trial {
    tokens: Vec<u32>,
    accepted: usize,
    status: JsonCandidate,
}

impl JsonByteCursor {
    /// Compile one schema with an explicit bound on committed and candidate bytes.
    pub fn new(schema: &serde_json::Value, max_bytes: usize) -> anyhow::Result<Self> {
        let max_bytes = NonZeroUsize::new(max_bytes).context(
            "JSON byte cursor max_bytes must be positive; configure a nonzero response budget",
        )?;
        // Each real token is one byte. 0xff can never occur in valid UTF-8 and
        // is rejected before recognition; its empty cell is never consumed.
        let mut pieces = (0..u8::MAX).map(|byte| vec![byte]).collect::<Vec<_>>();
        pieces.push(Vec::new());
        let eos = u32::try_from(pieces.len()).context("JSON byte vocabulary exceeds token IDs")?;
        let mut eos_piece = vec![TokTrie::SPECIAL_TOKEN_MARKER];
        eos_piece.extend_from_slice(b"<eos>");
        pieces.push(eos_piece);
        let info = TokRxInfo {
            vocab_size: u32::try_from(pieces.len())
                .context("JSON byte vocabulary size exceeds token IDs")?,
            tok_eos: eos,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &pieces)));
        let factory = ParserFactory::new(&env, InferenceCapabilities::default(), &[])
            .context("initialize JSON byte grammar factory")?;
        let parser = factory.create_parser(TopLevelGrammar::from_json_schema(schema.clone()))
            .context("compile JSON Schema for projected response bytes; repair the schema if unsupported")?;
        let matcher = ByteMatcher::new(parser);
        Ok(Self {
            initial: matcher.deep_clone(),
            matcher,
            committed: Vec::new(),
            max_bytes,
            failure: None,
        })
    }

    fn check_input(&self, bytes: &[u8]) -> anyhow::Result<()> {
        if let Some(failure) = &self.failure {
            return Err(failure.clone().into());
        }
        if !self
            .committed
            .len()
            .checked_add(bytes.len())
            .is_some_and(|size| size <= self.max_bytes.get())
        {
            return Err(JsonByteBudgetExceeded {
                limit: self.max_bytes.get(),
                offset: self.committed.len(),
            }
            .into());
        }
        Ok(())
    }

    fn trial(&mut self, bytes: &[u8]) -> anyhow::Result<Trial> {
        if let Some(error) = self.matcher.get_error() {
            return Err(anyhow::Error::new(error).context(format!(
                "JSON byte recognizer failed at committed offset {}; retry with a fresh cursor",
                self.committed.len()
            )));
        }
        let tokens = bytes.iter().copied().map(u32::from).collect::<Vec<_>>();
        let accepted = if self.matcher.is_stopped() {
            0
        } else {
            let valid_utf8_part = bytes
                .iter()
                .position(|byte| *byte == 0xff)
                .unwrap_or(bytes.len());
            self.matcher
                .try_consume_tokens(&tokens[..valid_utf8_part])
                .context("check candidate bytes against the compiled JSON Schema")?
        };
        let status = if accepted == bytes.len() {
            JsonCandidate::Viable {
                complete: self
                    .matcher
                    .is_accepting()
                    .context("check JSON grammar EOF eligibility")?,
            }
        } else {
            JsonCandidate::Rejected {
                byte_offset: self.committed.len() + accepted,
            }
        };
        Ok(Trial {
            tokens,
            accepted,
            status,
        })
    }

    fn restore_committed(&mut self) -> anyhow::Result<()> {
        let mut fresh = self.initial.deep_clone();
        let accepted = fresh
            .try_consume_tokens(&self.committed)
            .context("replay committed JSON bytes after a backend failure")?;
        ensure!(
            accepted == self.committed.len(),
            "JSON recovery accepted only {accepted} of {} committed bytes; create a fresh generation instead of reusing this cursor",
            self.committed.len()
        );
        self.matcher = fresh;
        Ok(())
    }

    fn recover(&mut self, operation: anyhow::Error) -> anyhow::Error {
        match self.restore_committed() {
            Ok(()) => operation,
            Err(recovery) => {
                let failure = CursorFailure(Arc::new(
                    RecoveryFailure {
                        operation,
                        recovery,
                    }
                    .into(),
                ));
                self.failure = Some(failure.clone());
                failure.into()
            }
        }
    }

    fn rollback(&mut self, accepted: usize) -> anyhow::Result<()> {
        self.matcher
            .rollback(accepted)
            .context("restore JSON grammar state after candidate inspection")
            .map_err(|error| self.recover(error))
    }

    /// Inspect a candidate without consuming any of its bytes, including on rejection.
    /// Backend failures recover the committed state and retain their causal error.
    pub fn preview(&mut self, bytes: &[u8]) -> anyhow::Result<JsonCandidate> {
        self.check_input(bytes)?;
        let trial = self.trial(bytes).map_err(|error| self.recover(error))?;
        self.rollback(trial.accepted)?;
        Ok(trial.status)
    }

    /// Compute canonical byte-piece candidates together. Pending UTF-8 is
    /// provisional: both this walk and failures preserve the committed prefix.
    pub(crate) fn vocabulary_mask(
        &mut self,
        vocabulary: &JsonTokenVocabulary,
        pending: &[u8],
    ) -> anyhow::Result<SimpleVob> {
        self.check_input(pending)?;
        let trial = self.trial(pending).map_err(|error| self.recover(error))?;
        if matches!(trial.status, JsonCandidate::Rejected { .. }) {
            self.rollback(trial.accepted)?;
            anyhow::bail!(
                "committed JSON UTF-8 tail cannot continue the schema; rebuild the projection before computing model candidates"
            );
        }
        match self.matcher.vocabulary_mask(vocabulary) {
            Ok(mask) => {
                self.rollback(trial.accepted)?;
                Ok(mask)
            }
            Err(error) => Err(self.recover(
                error.context("compute model-token mask using the compiled JSON byte grammar"),
            )),
        }
    }

    /// Commit an entirely viable prefix atomically. Rejection leaves the prefix intact.
    pub fn commit(&mut self, bytes: &[u8]) -> anyhow::Result<()> {
        self.check_input(bytes)?;
        let trial = self.trial(bytes).map_err(|error| self.recover(error))?;
        match trial.status {
            JsonCandidate::Viable { .. } => {
                self.committed.extend_from_slice(&trial.tokens);
                Ok(())
            }
            JsonCandidate::Rejected { byte_offset } => {
                self.rollback(trial.accepted)?;
                anyhow::bail!(
                    "JSON Schema rejects the content byte at offset {byte_offset}; supply a schema-valid partial value before continuing"
                );
            }
        }
    }

    /// Discard the current prefix and begin again with the already compiled schema.
    pub fn reset(&mut self) {
        self.matcher = self.initial.deep_clone();
        self.committed.clear();
        self.failure = None;
    }

    pub(crate) fn empty_fork(&self) -> anyhow::Result<Self> {
        if let Some(failure) = &self.failure {
            return Err(failure.clone().into());
        }
        Ok(Self {
            matcher: self.initial.deep_clone(),
            initial: self.initial.deep_clone(),
            committed: Vec::new(),
            max_bytes: self.max_bytes,
            failure: None,
        })
    }

    /// Check a replacement representation against the same compiled schema.
    pub fn preview_root(&self, bytes: &[u8]) -> anyhow::Result<JsonCandidate> {
        self.empty_fork()?.preview(bytes)
    }

    /// Atomically replace the logical prefix after a representation change.
    pub fn replace_prefix(&mut self, bytes: &[u8]) -> anyhow::Result<()> {
        let mut fresh = self.empty_fork()?;
        fresh.commit(bytes)?;
        *self = fresh;
        Ok(())
    }

    /// Number of committed logical bytes, including any unfinished UTF-8 scalar.
    pub fn committed_bytes(&self) -> usize {
        self.committed.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object_cursor(max_bytes: usize) -> JsonByteCursor {
        JsonByteCursor::new(
            &json!({
                "type":"object", "properties":{"n":{"enum":[1,2]}},
                "required":["n"], "additionalProperties":false
            }),
            max_bytes,
        )
        .unwrap()
    }

    #[test]
    fn json_byte_cursor_model_masks_keep_vocabulary_ids_and_byte_caches_separate() {
        let mut pieces = vec![Vec::new(); 300];
        pieces[255] = b"1".to_vec();
        pieces[270] = b"2".to_vec();
        pieces[12] = b"}".to_vec();
        let first = JsonTokenVocabulary::new(&pieces).unwrap();
        pieces.swap(255, 270);
        let second = JsonTokenVocabulary::new(&pieces).unwrap();
        let mut cursor = JsonByteCursor::new(&json!({"type":"object","properties":{"n":{"const":1}},"required":["n"],"additionalProperties":false}),128).unwrap();
        cursor.commit(br#"{"n":"#).unwrap();
        for _ in 0..2 {
            let mask = cursor.vocabulary_mask(&first, b"").unwrap();
            assert!(mask.get(255) && !mask.get(270) && !mask.get(12));
            let mask = cursor.vocabulary_mask(&second, b"").unwrap();
            assert!(!mask.get(255) && mask.get(270));
            assert_eq!(
                cursor.preview(b"1}").unwrap(),
                JsonCandidate::Viable { complete: true }
            );
            assert!(matches!(
                cursor.preview(b"2}").unwrap(),
                JsonCandidate::Rejected { .. }
            ));
            assert_eq!(cursor.committed_bytes(), 5);
        }
        cursor.commit(b"1").unwrap();
        assert!(cursor.vocabulary_mask(&first, b"").unwrap().get(12));
        cursor.commit(b"}").unwrap();
        assert_eq!(
            cursor.preview(b"").unwrap(),
            JsonCandidate::Viable { complete: true }
        );
    }

    #[test]
    fn json_byte_cursor_model_masks_restore_pending_utf8_and_backend_failures() {
        let vocabulary = JsonTokenVocabulary::new(&[vec![0xa9, b'"'], vec![0xa9, b'A']]).unwrap();
        let mut cursor = JsonByteCursor::new(&json!({"const":"é"}), 32).unwrap();
        cursor.commit(b"\"").unwrap();
        for _ in 0..2 {
            let mask = cursor.vocabulary_mask(&vocabulary, &[0xc3]).unwrap();
            assert!(mask.get(0) && !mask.get(1));
            assert_eq!(cursor.committed_bytes(), 1);
            assert_eq!(
                cursor.preview(&[0xc3, 0xa9, b'"']).unwrap(),
                JsonCandidate::Viable { complete: true }
            );
        }
        let _ = cursor.matcher.test_trigger_lexer_error();
        assert!(cursor.vocabulary_mask(&vocabulary, &[0xc3]).is_err());
        assert!(cursor.vocabulary_mask(&vocabulary, &[0xc3]).unwrap().get(0));
    }

    #[test]
    fn json_byte_cursor_replacement_probes_reuse_the_schema_and_preserve_the_old_prefix() {
        let mut cursor = object_cursor(64);
        cursor.commit(br#"{"n":1"#).unwrap();
        assert_eq!(
            cursor.preview_root(br#"{"n":2}"#).unwrap(),
            JsonCandidate::Viable { complete: true }
        );
        assert_eq!(cursor.committed_bytes(), 6);
        assert!(cursor.replace_prefix(br#"{"n":3}"#).is_err());
        assert_eq!(
            cursor.preview(b"}").unwrap(),
            JsonCandidate::Viable { complete: true }
        );
        cursor.replace_prefix(br#"{"n":2}"#).unwrap();
        assert_eq!(
            cursor.preview(b"").unwrap(),
            JsonCandidate::Viable { complete: true }
        );
    }

    #[test]
    fn json_byte_cursor_trials_restore_valid_rejected_and_complete_states() {
        let mut cursor = object_cursor(128);
        cursor.commit(br#"{"n":"#).unwrap();
        for _ in 0..3 {
            assert_eq!(
                cursor.preview(b"1}").unwrap(),
                JsonCandidate::Viable { complete: true }
            );
            assert_eq!(
                cursor.preview(b"3}").unwrap(),
                JsonCandidate::Rejected { byte_offset: 5 }
            );
            assert_eq!(
                cursor.preview(b"1").unwrap(),
                JsonCandidate::Viable { complete: false }
            );
            assert_eq!(
                cursor.preview(b"2}").unwrap(),
                JsonCandidate::Viable { complete: true }
            );
            assert_eq!(cursor.committed_bytes(), 5);
        }
        cursor.commit(b"1}").unwrap();
        assert_eq!(
            cursor.preview(b"").unwrap(),
            JsonCandidate::Viable { complete: true }
        );
        assert!(matches!(
            cursor.preview(b"x").unwrap(),
            JsonCandidate::Rejected { byte_offset: 7 }
        ));
        assert!(
            cursor
                .commit(b"x")
                .unwrap_err()
                .to_string()
                .contains("offset 7")
        );
        assert_eq!(cursor.committed_bytes(), 7);
        cursor.reset();
        cursor.commit(br#"{"n":2}"#).unwrap();
        assert_eq!(
            cursor.preview(b"").unwrap(),
            JsonCandidate::Viable { complete: true }
        );
    }

    #[test]
    fn json_byte_cursor_rejected_commit_and_budget_errors_leave_prefix_intact() {
        let mut cursor = object_cursor(7);
        cursor.commit(b"{").unwrap();
        assert!(
            cursor
                .commit(br#""bad":"#)
                .unwrap_err()
                .to_string()
                .contains("offset")
        );
        assert_eq!(cursor.committed_bytes(), 1);
        assert!(
            cursor
                .preview(b"1234567")
                .unwrap_err()
                .to_string()
                .contains("7-byte")
        );
        assert_eq!(cursor.committed_bytes(), 1);
        assert_eq!(
            cursor.preview(br#""n":1}"#).unwrap(),
            JsonCandidate::Viable { complete: true }
        );
        cursor.commit(br#""n":1}"#).unwrap();
        assert!(
            cursor
                .commit(b"x")
                .unwrap_err()
                .to_string()
                .contains("budget")
        );
        assert_eq!(
            cursor.preview(b"").unwrap(),
            JsonCandidate::Viable { complete: true }
        );
        assert!(
            JsonByteCursor::new(&json!({}), 0)
                .err()
                .unwrap()
                .to_string()
                .contains("positive")
        );
        let error = JsonByteCursor::new(&json!({"type":"invalid-json-type"}), 32)
            .err()
            .expect("invalid schema type must be rejected");
        assert!(
            error.to_string().contains("compile JSON Schema"),
            "{error:#}"
        );
        assert!(error.chain().count() > 1);
    }

    #[test]
    fn json_byte_cursor_handles_split_utf8_and_rejects_invalid_byte_candidates() {
        let mut cursor = JsonByteCursor::new(&json!({"type":"string"}), 128).unwrap();
        cursor.commit(b"\"").unwrap();
        cursor.commit(&[0xe7]).unwrap();
        for _ in 0..2 {
            assert_eq!(
                cursor.preview(&[0xb5, 0x82, b'"']).unwrap(),
                JsonCandidate::Viable { complete: true }
            );
            assert!(matches!(
                cursor.preview(&[0x20]).unwrap(),
                JsonCandidate::Rejected { .. }
            ));
            assert_eq!(cursor.committed_bytes(), 2);
        }
        cursor.commit(&[0xb5, 0x82]).unwrap();
        assert_eq!(
            cursor.preview(b"\"").unwrap(),
            JsonCandidate::Viable { complete: true }
        );
        assert_eq!(
            cursor.preview(&[0xff]).unwrap(),
            JsonCandidate::Rejected { byte_offset: 4 }
        );
        cursor.commit(b"\"").unwrap();
        assert_eq!(cursor.committed_bytes(), 5);
    }

    #[test]
    fn json_byte_cursor_backend_errors_recover_without_losing_committed_bytes() {
        let mut cursor = object_cursor(128);
        cursor.commit(br#"{"n":"#).unwrap();
        let _ = cursor.matcher.test_trigger_lexer_error();
        let error = cursor.preview(b"1}").unwrap_err();
        assert!(format!("{error:#}").contains("JSON"), "{error:#}");
        assert_eq!(cursor.committed_bytes(), 5);
        assert_eq!(
            cursor.preview(b"2}").unwrap(),
            JsonCandidate::Viable { complete: true }
        );
    }

    #[test]
    fn json_byte_cursor_unrecoverable_state_retains_causes_until_explicit_reset() {
        let mut cursor = object_cursor(128);
        cursor.commit(b"{").unwrap();
        cursor.committed.push(u32::from(b'x'));
        let _ = cursor.matcher.test_trigger_lexer_error();
        let error = cursor.preview(b"").unwrap_err();
        assert!(format!("{error:#}").contains("recovery"), "{error:#}");
        assert!(error.chain().count() > 1);
        assert!(cursor.preview(b"").unwrap_err().is::<CursorFailure>());
        cursor.reset();
        assert_eq!(
            cursor.preview(br#"{"n":1}"#).unwrap(),
            JsonCandidate::Viable { complete: true }
        );
    }
}
