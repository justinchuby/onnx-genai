//! Bounded model-authored realtime prompting through the canonical renderer.
use super::settings::SessionSettings;
use anyhow::{Context, ensure};
use onnx_genai_metadata::{RealtimeSpec, realtime::RealtimeContract};
use onnx_genai_ort::{ChatTemplate, TemplateLimits, TokenizationOptions, Tokenizer};
use serde_json::json;
use std::{collections::BTreeMap, num::NonZeroUsize};

#[derive(Clone, Copy)]
pub(super) enum PromptOperation {
    SessionStart,
    UserText,
    ToolResult,
    ResponseStart,
}
impl PromptOperation {
    fn name(self) -> &'static str {
        match self {
            Self::SessionStart => "session_start",
            Self::UserText => "user_text",
            Self::ToolResult => "tool_result",
            Self::ResponseStart => "response_start",
        }
    }
}
pub(super) struct Conditioning {
    template: ChatTemplate,
    limits: TemplateLimits,
    max_tokens: NonZeroUsize,
}
impl Conditioning {
    pub(super) fn new(
        spec: &RealtimeSpec,
        limits: TemplateLimits,
        max_tokens: NonZeroUsize,
    ) -> Self {
        Self {
            template: ChatTemplate::from_source(spec.prompt_template.clone()),
            limits,
            max_tokens,
        }
    }
    pub(super) fn tokens(
        &self,
        tokenizer: &Tokenizer,
        contract: &RealtimeContract,
        settings: &SessionSettings,
        operation: PromptOperation,
        text: &str,
        call_id: Option<&str>,
    ) -> anyhow::Result<Vec<u32>> {
        let kwargs: BTreeMap<_, _> = [
            ("operation".into(), json!(operation.name())),
            ("text".into(), json!(text)),
            ("call_id".into(), json!(call_id)),
            ("mode".into(), json!(settings.mode)),
            ("instructions".into(), json!(settings.instructions)),
            ("voice".into(), json!(settings.voice)),
            ("language".into(), json!(settings.output_language)),
            (
                "transcription_language".into(),
                json!(settings.input_language),
            ),
            ("transcription_prompt".into(), json!(settings.input_prompt)),
            ("session".into(), settings.snapshot("", "", contract)),
        ]
        .into_iter()
        .collect();
        let tools = crate::tool_protocol::responses_tools(settings.tools.clone())?.to_string();
        let rendered=self.template.render_with_limited_kwargs(&[],Some(&tools),false,None,&kwargs,self.limits)
            .with_context(||format!("render realtime profile '{}' {} prompt; repair realtime.prompt_template or reduce the supplied text/settings",contract.profile(),operation.name()))?;
        let tokens = tokenizer
            .encode_with_options(
                &rendered,
                TokenizationOptions {
                    add_special: false,
                    parse_special: true,
                },
            )
            .with_context(|| {
                format!(
                    "tokenize realtime {} prompt using the package tokenizer",
                    operation.name()
                )
            })?;
        ensure!(
            tokens.len() <= self.max_tokens.get(),
            "realtime {} prompt has {} tokens, exceeding its {}-token admission budget; shorten text/instructions or increase the configured realtime prompt budget",
            operation.name(),
            tokens.len(),
            self.max_tokens
        );
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::super::settings::{Capabilities, OutputBudget};
    use super::*;
    use std::num::{NonZeroU32, NonZeroU64};
    fn package() -> (RealtimeContract, Tokenizer) {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime");
        let metadata =
            onnx_genai_metadata::load_metadata(&root.join("inference_metadata.yaml")).unwrap();
        (
            onnx_genai_metadata::realtime::realtime_contract(&metadata)
                .unwrap()
                .unwrap(),
            Tokenizer::from_file(root.join("tokenizer.json")).unwrap(),
        )
    }
    fn renderer(spec: &RealtimeSpec, max_tokens: usize) -> Conditioning {
        Conditioning::new(
            spec,
            TemplateLimits {
                max_output_bytes: NonZeroUsize::new(4096).unwrap(),
                max_operations: NonZeroU64::new(10_000).unwrap(),
            },
            NonZeroUsize::new(max_tokens).unwrap(),
        )
    }
    #[test]
    fn model_template_carries_conditioning_and_exact_stream_tokens_without_inserted_bos() {
        let (contract, tokenizer) = package();
        let mut spec = contract.spec().clone();
        spec.prompt_template="{% if operation == 'session_start' %}{{ instructions }}{% if voice == 'alice' and language == 'zh' %} world{% endif %}{% else %}{{ text }}{% endif %}".into();
        let mut settings = SessionSettings::new(
            spec.default_mode,
            NonZeroU32::new(spec.input_sample_rate_hz).unwrap(),
            OutputBudget::Unlimited,
        );
        let caps = Capabilities {
            contract: &contract,
            tool_protocol: None,
        };
        settings.update(&json!({"instructions":"hello","audio":{"output":{"voice":"alice","language":"zh"}}}),&caps,false).unwrap();
        let renderer = renderer(&spec, 100);
        assert_eq!(
            renderer
                .tokens(
                    &tokenizer,
                    &contract,
                    &settings,
                    PromptOperation::SessionStart,
                    "",
                    None
                )
                .unwrap(),
            [4, 5]
        );
        assert_eq!(
            renderer
                .tokens(
                    &tokenizer,
                    &contract,
                    &settings,
                    PromptOperation::UserText,
                    "the quick",
                    None
                )
                .unwrap(),
            [6, 7]
        );
        assert_eq!(
            renderer
                .tokens(
                    &tokenizer,
                    &contract,
                    &settings,
                    PromptOperation::ToolResult,
                    "brown fox",
                    Some("call1")
                )
                .unwrap(),
            [8, 9]
        );
        assert_eq!(
            renderer
                .tokens(
                    &tokenizer,
                    &contract,
                    &settings,
                    PromptOperation::ResponseStart,
                    "hello",
                    None
                )
                .unwrap(),
            [4]
        );
    }
    #[test]
    fn model_prompt_expansion_and_token_counts_are_admitted_before_native_execution() {
        let (contract, tokenizer) = package();
        let mut spec = contract.spec().clone();
        let settings = SessionSettings::new(
            spec.default_mode,
            NonZeroU32::new(16000).unwrap(),
            OutputBudget::Unlimited,
        );
        assert!(
            renderer(&spec, 1)
                .tokens(
                    &tokenizer,
                    &contract,
                    &settings,
                    PromptOperation::UserText,
                    "hello world",
                    None
                )
                .unwrap_err()
                .to_string()
                .contains("2 tokens")
        );
        spec.prompt_template = "{{ 'hello' * 10000 }}".into();
        assert!(
            renderer(&spec, 100)
                .tokens(
                    &tokenizer,
                    &contract,
                    &settings,
                    PromptOperation::UserText,
                    "",
                    None
                )
                .is_err()
        );
    }
}
