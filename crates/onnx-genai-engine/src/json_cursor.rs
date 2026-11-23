//! Reversible byte-oriented JSON Schema recognition for projected response text.
use anyhow::{Context, ensure};
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, InferenceCapabilities, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory};
use std::{fmt, num::NonZeroUsize, sync::Arc};

/// Recognition of a candidate after the committed logical JSON bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonCandidate {
    /// Every byte is viable; complete additionally permits grammar EOF.
    Viable { complete: bool },
    /// The first rejected byte's offset in the combined logical JSON text.
    Rejected { byte_offset: usize },
}

/// A request-local compiled schema recognizer independent of model token IDs.
///
/// Bytes use the backend's JSON lexical language. Response-channel projection
/// and escape normalization belong before this cursor. User stops and length
/// limits do not require completeness; grammar EOF does.
pub struct JsonByteCursor {
    matcher: Matcher,
    initial: Matcher,
    committed: Vec<u32>,
    max_bytes: NonZeroUsize,
    failure: Option<CursorFailure>,
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
        let matcher = Matcher::new(Ok(parser));
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
        ensure!(
            self.committed
                .len()
                .checked_add(bytes.len())
                .is_some_and(|size| size <= self.max_bytes.get()),
            "JSON byte candidate would exceed the {}-byte response budget at committed offset {}; shorten the response or increase its explicit budget",
            self.max_bytes,
            self.committed.len()
        );
        Ok(())
    }

    fn trial(&mut self, bytes: &[u8]) -> anyhow::Result<Trial> {
        if let Some(error) = self.matcher.get_error() {
            anyhow::bail!(
                "JSON byte recognizer failed at committed offset {}: {error}; retry with a fresh cursor",
                self.committed.len()
            );
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
