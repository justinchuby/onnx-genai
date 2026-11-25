//! Model-authored generative ASR conditioning and bounded transcript projection.
use super::{TranscriptionLimits, bounded};
use crate::response_template::{
    PrefixOutput, ResponseParser, ResponseProjection, ResponseTemplate,
};
use anyhow::{Context, ensure};
use onnx_genai_ort::{ChatTemplate, TemplateLimits, TokenizationOptions, Tokenizer};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    num::{NonZeroU64, NonZeroUsize},
    path::Path,
    sync::Arc,
};

pub const CONFIG_FILE: &str = "transcription_config.json";
#[derive(Debug, Clone, Copy)]
pub struct PromptLimits {
    pub max_input_tokens: NonZeroUsize,
    pub template: TemplateLimits,
    pub max_text_bytes: NonZeroUsize,
}
impl Default for PromptLimits {
    fn default() -> Self {
        Self {
            max_input_tokens: NonZeroUsize::new(16384).unwrap(),
            template: TemplateLimits {
                max_output_bytes: NonZeroUsize::new(1024 * 1024).unwrap(),
                max_operations: NonZeroU64::new(1_000_000).unwrap(),
            },
            max_text_bytes: TranscriptionLimits::default().max_text_bytes,
        }
    }
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    version: u32,
    prompt_template: String,
    #[serde(default)]
    languages: Languages,
    #[serde(default)]
    prompt_conditioning: bool,
    #[serde(default)]
    response_template: Option<serde_json::Value>,
}
#[derive(Debug, Default, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Languages {
    #[default]
    Unsupported,
    Any,
    Choices {
        values: Vec<String>,
    },
}
#[derive(Debug)]
pub struct GenerativeTranscription {
    template: ChatTemplate,
    languages: Languages,
    prompt_conditioning: bool,
    response: Option<Arc<ResponseTemplate>>,
}
impl GenerativeTranscription {
    pub fn load(directory: &Path) -> anyhow::Result<Option<Arc<Self>>> {
        let path = directory.join(CONFIG_FILE);
        let bytes=match std::fs::read(&path) {
            Ok(bytes)=>bytes,
            Err(error) if error.kind()==std::io::ErrorKind::NotFound=>return Ok(None),
            Err(error)=>return Err(error).with_context(||format!("read generative transcription contract {}; repair the path and read permissions",path.display())),
        };
        Self::compile(&bytes).map(|contract|Some(Arc::new(contract)))
            .with_context(||format!("load generative transcription contract {}; repair its declared prompting and response boundaries",path.display()))
    }
    pub fn compile(bytes: &[u8]) -> anyhow::Result<Self> {
        let declaration:Declaration=serde_json::from_slice(bytes)
            .context("parse transcription_config.json; provide version, prompt_template and supported conditioning fields as valid JSON")?;
        ensure!(
            declaration.version == 1,
            "transcription_config.json version {} is unsupported; use version 1",
            declaration.version
        );
        ensure!(
            !declaration.prompt_template.trim().is_empty(),
            "transcription prompt_template is empty; declare the model's exact decoder prompt"
        );
        if let Languages::Choices { values } = &declaration.languages {
            ensure!(
                !values.is_empty()
                    && values.iter().all(|value| !value.trim().is_empty())
                    && values
                        .iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        == values.len(),
                "transcription languages.choices requires distinct nonempty values; list accepted language labels or use kind unsupported/any"
            );
        }
        let response=declaration.response_template.as_ref().map(ResponseTemplate::compile).transpose()
            .context("compile transcription response_template; declare the transcript content channel and its boundaries")?;
        Ok(Self {
            template: ChatTemplate::from_source(declaration.prompt_template),
            languages: declaration.languages,
            prompt_conditioning: declaration.prompt_conditioning,
            response,
        })
    }
    pub fn render(
        &self,
        tokenizer: &Tokenizer,
        language: Option<&str>,
        prompt: Option<&str>,
        limits: PromptLimits,
    ) -> anyhow::Result<RenderedPrompt> {
        let language = language.filter(|value| !value.is_empty());
        let prompt = prompt.filter(|value| !value.is_empty());
        match (&self.languages, language) {
            (Languages::Unsupported, Some(_)) => anyhow::bail!(
                "transcription language conditioning is not declared; omit language or select a contract with languages.kind any/choices"
            ),
            (Languages::Choices { values }, Some(value)) => ensure!(
                values.iter().any(|allowed| allowed == value),
                "transcription language '{value}' is not declared; choose one of {values:?} or omit language for the model default"
            ),
            _ => {}
        }
        ensure!(
            self.prompt_conditioning || prompt.is_none(),
            "transcription prompt conditioning is not declared; omit prompt or select a contract with prompt_conditioning true"
        );
        let kwargs: BTreeMap<_, _> = [
            ("language".into(), serde_json::json!(language)),
            ("prompt".into(), serde_json::json!(prompt)),
        ]
        .into_iter()
        .collect();
        let rendered=self.template.render_with_limited_kwargs(&[],None,false,None,&kwargs,limits.template)
            .context("render transcription prompt_template; repair the template or shorten language/prompt conditioning")?;
        let token_ids = tokenizer
            .encode_with_options(
                &rendered,
                TokenizationOptions {
                    add_special: false,
                    parse_special: true,
                },
            )
            .context("tokenize the declared transcription prompt; repair the package tokenizer")?;
        ensure!(
            !token_ids.is_empty(),
            "transcription prompt_template produced no decoder tokens; declare a nonempty decoder prompt"
        );
        bounded("input tokens", token_ids.len(), limits.max_input_tokens)?;
        Ok(RenderedPrompt {
            token_ids,
            text: rendered,
        })
    }
    pub fn prepare(
        &self,
        tokenizer: &Tokenizer,
        language: Option<&str>,
        prompt: Option<&str>,
        limits: PromptLimits,
    ) -> anyhow::Result<PreparedPrompt> {
        let RenderedPrompt {
            token_ids,
            text: rendered,
        } = self.render(tokenizer, language, prompt, limits)?;
        let projection = self
            .response
            .as_ref()
            .map(|response| {
                ResponseProjection::resolve(
                    response.clone(),
                    rendered.clone(),
                    limits.max_text_bytes.get(),
                    Some(tokenizer),
                )
            })
            .transpose()?;
        let output =
            TranscriptRenderer::new(self.response.clone(), &rendered, limits.max_text_bytes)?;
        Ok(PreparedPrompt {
            token_ids,
            projection,
            output,
        })
    }
}
pub struct RenderedPrompt {
    pub token_ids: Vec<u32>,
    pub text: String,
}
pub struct PreparedPrompt {
    pub token_ids: Vec<u32>,
    pub projection: Option<ResponseProjection>,
    pub output: TranscriptRenderer,
}
/// Raw and projected bytes are bounded independently. Prefix text is parser
/// context, never newly recognized speech; committed output reconciles deltas.
pub struct TranscriptRenderer {
    parser: Option<ResponseParser>,
    raw: String,
    text: String,
    maximum: NonZeroUsize,
}
pub struct TranscriptOutput {
    pub text: String,
    pub delta: String,
}
impl TranscriptRenderer {
    pub fn plain(maximum: NonZeroUsize) -> Self {
        Self {
            parser: None,
            raw: String::new(),
            text: String::new(),
            maximum,
        }
    }
    fn new(
        response: Option<Arc<ResponseTemplate>>,
        prefix: &str,
        maximum: NonZeroUsize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            parser: response
                .map(|response| {
                    ResponseParser::new(response, prefix, PrefixOutput::Exclude, maximum.get())
                })
                .transpose()?,
            ..Self::plain(maximum)
        })
    }
    fn append_text(&mut self, delta: &str) -> anyhow::Result<()> {
        let length = self
            .text
            .len()
            .checked_add(delta.len())
            .context("transcription projected text size overflows; shorten the recording")?;
        bounded("projected text bytes", length, self.maximum)?;
        self.text.push_str(delta);
        Ok(())
    }
    pub fn push(&mut self, raw: &str) -> anyhow::Result<String> {
        let length = self
            .raw
            .len()
            .checked_add(raw.len())
            .context("transcription raw text size overflows; shorten the recording")?;
        bounded("raw text bytes", length, self.maximum)?;
        let delta = match &mut self.parser {
            Some(parser) => parser.feed(raw)?.content,
            None => raw.to_owned(),
        };
        self.append_text(&delta)?;
        self.raw.push_str(raw);
        Ok(delta)
    }
    pub fn finish(mut self, committed: &str) -> anyhow::Result<TranscriptOutput> {
        let tail=committed.strip_prefix(&self.raw).context("committed transcription does not extend its published raw deltas; repair the tokenizer's incremental output contract")?;
        let mut delta = self.push(tail)?;
        if let Some(parser) = self.parser.take() {
            let remaining = parser.finish()?.content;
            self.append_text(&remaining)?;
            delta.push_str(&remaining);
        }
        Ok(TranscriptOutput {
            text: self.text,
            delta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tokenizer() -> Tokenizer {
        Tokenizer::from_file(
            Path::new(env!("CARGO_MANIFEST_DIR")).join(
                "../../tests/fixtures/onnx_genai_workflows/transcription_tokens/tokenizer.json",
            ),
        )
        .unwrap()
    }
    fn contract() -> GenerativeTranscription {
        GenerativeTranscription::compile(br#"{"version":1,"prompt_template":"hello{% if language %} {{ language }}{% endif %}{% if prompt %} {{ prompt }}{% endif %}","languages":{"kind":"choices","values":["world"]},"prompt_conditioning":true}"#).unwrap()
    }
    #[test]
    fn authored_prompt_uses_exact_tokens_conditioning_and_limits() {
        let tokenizer = tokenizer();
        let contract = contract();
        let limits = PromptLimits::default();
        assert_eq!(
            contract
                .prepare(&tokenizer, Some("world"), Some("dog"), limits)
                .unwrap()
                .token_ids,
            vec![4, 5, 13]
        );
        assert!(
            contract
                .prepare(&tokenizer, Some("absent"), None, limits)
                .is_err()
        );
        let mut short = limits;
        short.max_input_tokens = NonZeroUsize::new(1).unwrap();
        assert!(
            contract
                .prepare(&tokenizer, None, Some("world"), short)
                .is_err()
        );
        short.template.max_output_bytes = NonZeroUsize::new(2).unwrap();
        assert!(contract.prepare(&tokenizer, None, None, short).is_err());
        let none = GenerativeTranscription::compile(br#"{"version":1,"prompt_template":"hello"}"#)
            .unwrap();
        assert!(
            none.prepare(&tokenizer, Some("world"), None, limits)
                .is_err()
        );
        assert!(
            none.prepare(&tokenizer, None, Some("world"), limits)
                .is_err()
        );
        assert!(
            GenerativeTranscription::compile(br#"{"version":2,"prompt_template":"hello"}"#)
                .is_err()
        );
        assert!(
            GenerativeTranscription::compile(br#"{"version":1,"prompt_template":""}"#).is_err()
        );
    }
    #[test]
    fn transcript_projection_handles_split_boundaries_and_committed_tail() {
        let spec=ResponseTemplate::compile(&serde_json::json!({"start_anchor":"dog","fields":{"content":{"open":"tok16 ","close":" tok17","optional":false}}})).unwrap();
        let raw = "tok16 hello你好 tok17";
        for split in (0..=raw.len()).filter(|split| raw.is_char_boundary(*split)) {
            let mut output = TranscriptRenderer::new(
                Some(spec.clone()),
                "dog",
                NonZeroUsize::new(1024).unwrap(),
            )
            .unwrap();
            let first = output.push(&raw[..split]).unwrap();
            let result = output.finish(raw).unwrap();
            assert_eq!(result.text, "hello你好");
            assert_eq!(format!("{first}{}", result.delta), result.text);
        }
        let mut output = TranscriptRenderer::plain(NonZeroUsize::new(3).unwrap());
        assert!(output.push("four").is_err());
        let mut output = TranscriptRenderer::plain(NonZeroUsize::new(10).unwrap());
        output.push("old").unwrap();
        assert!(output.finish("new").is_err());
    }
}
