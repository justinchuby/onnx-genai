//! Transactional response text before and after declared response-channel projection.
use crate::response_template::{
    IncompleteResponseUtf8, PrefixOutput, ResponseBufferExceeded, ResponseParser, ResponseTemplate,
};
use crate::tokenization::SpecialTokenHandling;
use anyhow::Context;
use std::{num::NonZeroUsize, sync::Arc};

pub(crate) enum ProjectionKind {
    Plain,
    Declared(Arc<ResponseTemplate>),
}

impl ProjectionKind {
    pub(crate) fn special_tokens(&self) -> SpecialTokenHandling {
        match self {
            Self::Plain => SpecialTokenHandling::Skip,
            Self::Declared(_) => SpecialTokenHandling::Preserve,
        }
    }

    pub(crate) fn create(&self, prefix: &str, max_bytes: usize) -> anyhow::Result<TextProjection> {
        match self {
            Self::Plain => {
                let limit = NonZeroUsize::new(max_bytes).context(
                    "constraint text byte budget must be positive; configure max_constraint_bytes",
                )?;
                if prefix.len() > limit.get() {
                    return Err(ResponseBufferExceeded { limit: limit.get() }.into());
                }
                Ok(TextProjection::Plain(PlainText {
                    bytes: prefix.as_bytes().into(),
                    limit,
                }))
            }
            Self::Declared(template) => {
                template.require_constraint_boundaries()?;
                Ok(TextProjection::Declared(Box::new(ResponseParser::new(
                    template.clone(),
                    prefix,
                    PrefixOutput::Include,
                    max_bytes,
                )?)))
            }
        }
    }
}

pub(crate) enum TextProjection {
    Plain(PlainText),
    Declared(Box<ResponseParser>),
}

pub(crate) enum TextView<'a> {
    Plain { text: &'a str, pending: &'a [u8] },
    Declared(&'a ResponseParser),
}

impl TextView<'_> {
    pub(crate) fn content(&self, terminal: bool) -> String {
        match self {
            Self::Plain { text, .. } => (*text).into(),
            Self::Declared(parser) => parser.content_text(terminal),
        }
    }

    pub(crate) fn grammar_content(&self) -> String {
        match self {
            Self::Plain { text, .. } => (*text).into(),
            Self::Declared(parser) => parser.grammar_content_text(),
        }
    }

    pub(crate) fn pending_bytes(&self) -> anyhow::Result<&[u8]> {
        match self {
            Self::Plain { pending, .. } => Ok(pending),
            Self::Declared(parser) => Ok(parser.pending_content_bytes()?.unwrap_or_default()),
        }
    }

    pub(crate) fn pending_can_be_trimmed(&self) -> bool {
        match self {
            Self::Plain { .. } => false,
            Self::Declared(parser) => parser.pending_can_be_trimmed(),
        }
    }
}

impl TextProjection {
    pub(crate) fn plain_pending(&self) -> anyhow::Result<Option<&[u8]>> {
        match self {
            Self::Declared(_) => Ok(None),
            Self::Plain(text) => {
                let (_, pending) = split_text_utf8(&text.bytes)?;
                Ok(Some(pending))
            }
        }
    }

    pub(crate) fn can_append_plain(&self, bytes: usize) -> bool {
        match self {
            Self::Declared(_) => false,
            Self::Plain(text) => text
                .bytes
                .len()
                .checked_add(bytes)
                .is_some_and(|len| len <= text.limit.get()),
        }
    }
    pub(crate) fn content(&mut self, terminal: bool) -> anyhow::Result<String> {
        self.with_preview(b"", terminal, |view| Ok(view.content(terminal)))
    }

    pub(crate) fn with_preview<T>(
        &mut self,
        bytes: &[u8],
        terminal: bool,
        inspect: impl FnOnce(TextView<'_>) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        match self {
            Self::Declared(parser) => parser.with_preview(bytes, terminal, |parser| {
                inspect(TextView::Declared(parser))
            }),
            Self::Plain(text) => {
                let trial = text.append(bytes)?;
                inspect(trial.view(terminal)?)
            }
        }
    }

    pub(crate) fn commit(&mut self, bytes: &[u8]) -> anyhow::Result<()> {
        match self {
            Self::Declared(parser) => parser.commit_unpublished(bytes),
            Self::Plain(text) => {
                let mut trial = text.append(bytes)?;
                trial.view(false)?;
                trial.original = trial.bytes.len();
                Ok(())
            }
        }
    }
}

pub(crate) struct PlainText {
    bytes: Vec<u8>,
    limit: NonZeroUsize,
}

impl PlainText {
    fn append(&mut self, bytes: &[u8]) -> anyhow::Result<PlainTrial<'_>> {
        if !self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_some_and(|len| len <= self.limit.get())
        {
            return Err(ResponseBufferExceeded {
                limit: self.limit.get(),
            }
            .into());
        }
        let original = self.bytes.len();
        self.bytes.extend_from_slice(bytes);
        Ok(PlainTrial {
            bytes: &mut self.bytes,
            original,
        })
    }
}

struct PlainTrial<'a> {
    bytes: &'a mut Vec<u8>,
    original: usize,
}

fn split_text_utf8(bytes: &[u8]) -> anyhow::Result<(&str, &[u8])> {
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok((text, &[])),
        Err(error) if error.error_len().is_none() => {
            let at = error.valid_up_to();
            let text = std::str::from_utf8(&bytes[..at])
                .context("read validated response UTF-8 text before pending scalar")?;
            Ok((text, &bytes[at..]))
        }
        Err(error) => Err(anyhow::Error::new(error)
            .context("model token produces invalid UTF-8; use valid byte pieces or repair the tokenizer decoder")),
    }
}

impl PlainTrial<'_> {
    fn view(&self, terminal: bool) -> anyhow::Result<TextView<'_>> {
        let (text, pending) = split_text_utf8(self.bytes)?;
        if terminal && !pending.is_empty() {
            return Err(IncompleteResponseUtf8 {
                bytes: pending.len(),
                offset: text.len(),
            }
            .into());
        }
        Ok(TextView::Plain { text, pending })
    }
}

impl Drop for PlainTrial<'_> {
    fn drop(&mut self) {
        self.bytes.truncate(self.original);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_projection_trials_preserve_partial_utf8_and_restore_errors() {
        let mut projection = ProjectionKind::Plain.create("\"", 16).unwrap();
        projection.commit(&[0xe7]).unwrap();
        assert!(
            projection
                .content(true)
                .unwrap_err()
                .is::<IncompleteResponseUtf8>()
        );
        for _ in 0..2 {
            projection
                .with_preview(&[0xb5, 0x82, b'"'], true, |view| {
                    assert_eq!(view.content(true), "\"終\"");
                    assert!(view.pending_bytes()?.is_empty());
                    Ok(())
                })
                .unwrap();
            assert_eq!(projection.content(false).unwrap(), "\"");
            assert!(
                projection
                    .commit(b"X")
                    .unwrap_err()
                    .is::<std::str::Utf8Error>()
            );
            assert_eq!(projection.content(false).unwrap(), "\"");
        }
        projection.commit(&[0xb5, 0x82, b'"']).unwrap();
        assert_eq!(projection.content(true).unwrap(), "\"終\"");
    }

    #[test]
    fn plain_text_projection_rolls_back_budget_inspection_errors_and_panics() {
        let mut projection = ProjectionKind::Plain.create("{", 2).unwrap();
        let error = projection
            .with_preview(b"}", true, |_| -> anyhow::Result<()> {
                anyhow::bail!("inspection failed")
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "inspection failed");
        assert_eq!(projection.content(false).unwrap(), "{");
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: anyhow::Result<()> =
                projection.with_preview(b"}", true, |_| panic!("trial panic"));
        }));
        assert!(panic.is_err());
        assert_eq!(projection.content(false).unwrap(), "{");
        assert!(
            projection
                .commit(b"} ")
                .unwrap_err()
                .is::<ResponseBufferExceeded>()
        );
        assert_eq!(projection.content(false).unwrap(), "{");
        projection.commit(b"}").unwrap();
        assert_eq!(projection.content(true).unwrap(), "{}");
    }
}
