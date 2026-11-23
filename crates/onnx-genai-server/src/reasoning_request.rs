//! Wire reasoning controls lowered through explicit tokenizer/channel contracts.
use crate::types::ChatCompletionRequest;
use anyhow::{Context, ensure};
use onnx_genai_engine::{
    GenerateOptions,
    config::GenerateConstraint,
    reasoning::{ReasoningBudget, ReasoningBudgetSpec},
    response_template::{ReasoningBoundary, ResponseTemplate},
};
use onnx_genai_ort::{TokenizationOptions, Tokenizer};
use serde::Deserialize;
use std::sync::Arc;

const FIELDS: &[&str] = &[
    "reasoning_budget_tokens",
    "thinking_budget_tokens",
    "reasoning_budget_start_tag",
    "reasoning_budget_end_tags",
    "reasoning_budget_end_tag",
    "reasoning_budget_message",
    "reasoning_control",
    "generation_prompt",
];

#[derive(Default, Deserialize)]
pub(crate) struct ReasoningRequest {
    reasoning_budget_tokens: Option<i64>,
    thinking_budget_tokens: Option<i64>,
    reasoning_budget_start_tag: Option<String>,
    reasoning_budget_end_tags: Option<Vec<String>>,
    reasoning_budget_end_tag: Option<String>,
    #[serde(default)]
    reasoning_budget_message: String,
    #[serde(default)]
    pub(crate) reasoning_control: bool,
    generation_prompt: Option<String>,
}
pub(crate) enum BudgetContext<'a> {
    Raw,
    Chat {
        template: Option<&'a Arc<ResponseTemplate>>,
        prompt: &'a str,
    },
}
impl ReasoningRequest {
    pub(crate) fn is_field(name: &str) -> bool {
        FIELDS.contains(&name)
    }
    pub(crate) fn extract(request: &ChatCompletionRequest) -> anyhow::Result<Self> {
        let fields = request
            .extensions
            .iter()
            .filter(|(name, _)| Self::is_field(name))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect::<serde_json::Map<_, _>>();
        let parsed:Self=serde_path_to_error::deserialize(serde_json::Value::Object(fields))
            .context("parse reasoning controls; supply integer budgets, text delimiters and a boolean reasoning_control")?;
        for (name, value) in [
            ("reasoning_budget_tokens", parsed.reasoning_budget_tokens),
            ("thinking_budget_tokens", parsed.thinking_budget_tokens),
        ] {
            if let Some(value) = value {
                ensure!(
                    (-1..=i64::from(i32::MAX)).contains(&value),
                    "{name} must be -1 for the default/unlimited budget or an integer in 0..={}; received {value}",
                    i32::MAX
                );
            }
        }
        Ok(parsed)
    }
    fn limit(&self) -> Option<usize> {
        self.reasoning_budget_tokens
            .or(self.thinking_budget_tokens)
            .filter(|value| *value >= 0)
            .map(|value| value as usize)
    }
    fn end_override(&self) -> Option<Vec<String>> {
        self.reasoning_budget_end_tags.clone().or_else(|| {
            self.reasoning_budget_end_tag
                .as_ref()
                .map(|tag| vec![tag.clone()])
        })
    }
    fn apply_generation_prompt(
        &self,
        options: &mut GenerateOptions,
        context: &BudgetContext<'_>,
    ) -> anyhow::Result<()> {
        let Some(explicit) = self.generation_prompt.as_deref() else {
            return Ok(());
        };
        match context {
            BudgetContext::Chat {
                template: Some(template),
                prompt,
            } => {
                ensure!(
                    template.generation_prefix(prompt)? == explicit,
                    "generation_prompt conflicts with the final assistant prefill from response_template; omit it or supply that exact suffix"
                );
            }
            BudgetContext::Chat { prompt, .. } => ensure!(
                prompt.ends_with(explicit),
                "generation_prompt is not a suffix of the rendered chat prompt; supply the actual assistant prefill"
            ),
            BudgetContext::Raw => {}
        }
        match options.constraint.as_mut() {
            Some(
                GenerateConstraint::Json { prefix }
                | GenerateConstraint::JsonSchema { prefix, .. }
                | GenerateConstraint::Regex { prefix, .. }
                | GenerateConstraint::Lark { prefix, .. },
            ) => {
                if matches!(context, BudgetContext::Chat { .. }) && !prefix.is_empty() {
                    ensure!(
                        prefix == explicit,
                        "generation_prompt conflicts with the prepared constrained-content prefix; preserve the existing assistant continuation"
                    );
                }
                *prefix = explicit.to_owned();
            }
            Some(GenerateConstraint::ResponseJson { prefix, .. })
                if matches!(context, BudgetContext::Raw) =>
            {
                *prefix = explicit.to_owned()
            }
            _ => {}
        }
        Ok(())
    }

    fn has_explicit_protocol(&self) -> bool {
        self.reasoning_budget_start_tag.is_some()
            || self.reasoning_budget_end_tags.is_some()
            || self.reasoning_budget_end_tag.is_some()
    }

    pub(crate) fn apply(
        &self,
        options: &mut GenerateOptions,
        tokenizer: &Tokenizer,
        context: BudgetContext<'_>,
    ) -> anyhow::Result<()> {
        let hint_only = self.reasoning_control
            && self.limit().is_none()
            && !self.has_explicit_protocol()
            && self.generation_prompt.is_none()
            && self.reasoning_budget_message.is_empty();
        match self.apply_protocol(options, tokenizer, context) {
            Err(error) if hint_only => {
                tracing::debug!(reason=%format!("{error:#}"),"optional reasoning control is unavailable for this tokenizer/template contract");
                options.reasoning_budget = None;
                Ok(())
            }
            result => result,
        }
    }

    fn apply_protocol(
        &self,
        options: &mut GenerateOptions,
        tokenizer: &Tokenizer,
        context: BudgetContext<'_>,
    ) -> anyhow::Result<()> {
        self.apply_generation_prompt(options, &context)?;
        if self.limit().is_none() && !self.reasoning_control {
            return Ok(());
        }
        // The upstream frontend requests optional control on every model. A
        // model with no declared/explicit reasoning protocol has no applicable
        // block to count or arm; do not infer delimiters from template text.
        let explicit_protocol = self.has_explicit_protocol();
        if !explicit_protocol {
            match &context {
                BudgetContext::Raw => return Ok(()),
                BudgetContext::Chat { template, .. }
                    if template.is_none_or(|template| !template.has_reasoning_channel()) =>
                {
                    return Ok(());
                }
                _ => {}
            }
        }
        let (starts, ends, prefix, initially_active) = match context {
            BudgetContext::Chat {
                template: Some(template),
                prompt,
            } => {
                let boundaries = template.reasoning_boundaries()?;
                let prefix = template.generation_prefix(prompt)?;
                ensure!(
                    self.generation_prompt
                        .as_deref()
                        .is_none_or(|explicit| explicit == prefix),
                    "generation_prompt conflicts with the final assistant prefix derived from response_template; omit it for chat requests or provide that exact prefix"
                );
                let starts = if let Some(start) = boundaries.start.as_ref() {
                    marker_alternatives(
                        start,
                        self.reasoning_budget_start_tag
                            .as_ref()
                            .map(|tag| vec![tag.clone()]),
                        "reasoning_budget_start_tag",
                    )?
                } else {
                    ensure!(
                        self.reasoning_budget_start_tag.is_none(),
                        "response_template declares an implicit reasoning channel; omit reasoning_budget_start_tag instead of inventing an opener"
                    );
                    Vec::new()
                };
                let ends = marker_alternatives(
                    &boundaries.end,
                    self.end_override(),
                    "reasoning_budget_end_tags",
                )?;
                (starts, ends, prefix.to_owned(), boundaries.start.is_none())
            }
            context => {
                let start=self.reasoning_budget_start_tag.clone().context("reasoning budget needs a declared response_template or explicit reasoning_budget_start_tag; supply the model's actual reasoning protocol")?;
                let ends=self.end_override().context("reasoning budget needs a declared response_template or explicit reasoning_budget_end_tags; supply a nonempty closer list")?;
                let prefix = match context {
                    BudgetContext::Raw => self.generation_prompt.clone().unwrap_or_default(),
                    BudgetContext::Chat { prompt, .. } => {
                        let prefix=self.generation_prompt.clone().context("chat reasoning controls without a response_template require explicit generation_prompt matching the actual assistant prefill; declare response_template or supply that suffix")?;
                        ensure!(
                            prompt.ends_with(&prefix),
                            "generation_prompt must be a suffix of the rendered chat prompt; use its actual final assistant prefill"
                        );
                        prefix
                    }
                };
                (vec![start], ends, prefix, false)
            }
        };
        let start_sequences = starts
            .iter()
            .map(|tag| encode_delimiter(tokenizer, tag, "reasoning_budget_start_tag"))
            .collect::<anyhow::Result<_>>()?;
        let end_sequences = ends
            .iter()
            .map(|tag| encode_delimiter(tokenizer, tag, "reasoning_budget_end_tags"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        ensure!(
            !end_sequences.is_empty(),
            "reasoning_budget_end_tags must contain a nonempty closer; provide the declared end-of-reasoning token sequence"
        );
        let mut forced_tokens = tokenizer
            .encode_with_options(
                &self.reasoning_budget_message,
                TokenizationOptions {
                    add_special: false,
                    parse_special: true,
                },
            )
            .context("tokenize reasoning_budget_message")?;
        forced_tokens.extend_from_slice(&end_sequences[0]);
        let prefix_tokens = tokenizer
            .encode_with_options(
                &prefix,
                TokenizationOptions {
                    add_special: false,
                    parse_special: true,
                },
            )
            .context("tokenize existing reasoning generation_prompt")?;
        options.reasoning_budget = Some(
            ReasoningBudget::resolve(
                tokenizer,
                ReasoningBudgetSpec {
                    start_sequences,
                    end_sequences,
                    forced_tokens,
                    prefix_tokens,
                    max_tokens: self.limit(),
                    initially_active,
                },
            )
            .context("resolve request reasoning budget against the declared tokenizer protocol")?,
        );
        Ok(())
    }
}
fn marker_alternatives(
    boundary: &ReasoningBoundary<'_>,
    explicit: Option<Vec<String>>,
    field: &str,
) -> anyhow::Result<Vec<String>> {
    let alternatives=explicit.or_else(||boundary.literal_alternatives().map(<[String]>::to_vec))
        .with_context(||format!("{field} uses a patterned response_template boundary; supply an explicit matching delimiter instead of guessing one"))?;
    ensure!(
        !alternatives.is_empty(),
        "{field} requires a nonempty delimiter list"
    );
    for literal in &alternatives {
        boundary
            .validate_literal(literal)
            .with_context(|| format!("validate {field} against response_template"))?;
    }
    Ok(alternatives)
}
fn encode_delimiter(tokenizer: &Tokenizer, text: &str, field: &str) -> anyhow::Result<Vec<u32>> {
    ensure!(
        !text.is_empty(),
        "{field} must not contain an empty delimiter"
    );
    let tokens = tokenizer
        .encode_with_options(
            text,
            TokenizationOptions {
                add_special: false,
                parse_special: true,
            },
        )
        .with_context(|| format!("tokenize {field}"))?;
    ensure!(
        !tokens.is_empty(),
        "{field} encodes to no tokens; repair the tokenizer protocol"
    );
    let decoded = tokenizer
        .decode_with_special_tokens(&tokens)
        .with_context(|| format!("validate {field} tokenizer round-trip"))?;
    ensure!(
        decoded == text,
        "{field} encodes as {decoded:?}, not the declared {text:?}; provide a lossless delimiter or repair the tokenizer vocabulary/decoder"
    );
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    fn request(fields: Value) -> ChatCompletionRequest {
        let mut body = json!({"messages":[{"role":"user","content":"hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(body).unwrap()
    }
    fn tokenizer() -> Tokenizer {
        Tokenizer::from_file(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-llm/tokenizer.json"),
        )
        .unwrap()
    }
    #[test]
    fn reasoning_budget_wire_precedence_and_range_match_upstream_fields() {
        let parsed = ReasoningRequest::extract(&request(
            json!({"reasoning_budget_tokens":3,"thinking_budget_tokens":9,
            "reasoning_budget_end_tags":["tok23"],"reasoning_budget_end_tag":"tok24"}),
        ))
        .unwrap();
        assert_eq!(parsed.limit(), Some(3));
        assert_eq!(parsed.end_override().unwrap(), ["tok23"]);
        assert_eq!(
            ReasoningRequest::extract(&request(json!({"thinking_budget_tokens":0})))
                .unwrap()
                .limit(),
            Some(0)
        );
        for fields in [
            json!({"reasoning_budget_tokens":-2}),
            json!({"thinking_budget_tokens":2147483648i64}),
            json!({"reasoning_control":"true"}),
        ] {
            assert!(ReasoningRequest::extract(&request(fields)).is_err());
        }
    }
    #[test]
    fn reasoning_budget_wire_resolves_explicit_raw_protocol_and_rejects_lossy_markers() {
        let tokenizer = tokenizer();
        let mut options = GenerateOptions::default();
        let request = request(
            json!({"thinking_budget_tokens":0,"reasoning_budget_start_tag":"tok22",
            "reasoning_budget_end_tag":"tok23","generation_prompt":"tok22"}),
        );
        ReasoningRequest::extract(&request)
            .unwrap()
            .apply(&mut options, &tokenizer, BudgetContext::Raw)
            .unwrap();
        assert!(options.reasoning_budget.is_some());
        options.validate_processors(&tokenizer).unwrap();
        let bad = request_with_tags("not-in-vocabulary");
        assert!(
            ReasoningRequest::extract(&bad)
                .unwrap()
                .apply(&mut options, &tokenizer, BudgetContext::Raw)
                .unwrap_err()
                .to_string()
                .contains("encodes as")
        );
    }
    fn request_with_tags(start: &str) -> ChatCompletionRequest {
        request(
            json!({"reasoning_budget_tokens":0,"reasoning_budget_start_tag":start,"reasoning_budget_end_tag":"tok23"}),
        )
    }
    #[test]
    fn reasoning_budget_wire_derives_declared_chat_boundaries_and_final_prefill() {
        let tokenizer = tokenizer();
        let mut options = GenerateOptions::default();
        let template = ResponseTemplate::compile(&json!({"start_anchor":"anchor","fields":{
            "reasoning_content":{"open":"tok22","close":["tok23","tok24"]},"content":{}
        }}))
        .unwrap();
        ReasoningRequest::extract(&request(json!({"reasoning_budget_tokens":0})))
            .unwrap()
            .apply(
                &mut options,
                &tokenizer,
                BudgetContext::Chat {
                    template: Some(&template),
                    prompt: "anchor tok22 prior tok23 user anchor tok22",
                },
            )
            .unwrap();
        assert!(options.reasoning_budget.is_some());
        options.validate_processors(&tokenizer).unwrap();
    }
    #[test]
    fn reasoning_budget_wire_supports_implicit_and_explicit_pattern_delimiters() {
        let tokenizer = tokenizer();
        let mut options = GenerateOptions::default();
        let template = ResponseTemplate::compile(&json!({"start_anchor":"anchor","fields":{
            "reasoning_content":{},"content":{"open_pattern":"tok2[34]"}
        }}))
        .unwrap();
        let explicit = ReasoningRequest::extract(&request(
            json!({"reasoning_budget_tokens":0,"reasoning_budget_end_tags":["tok23"]}),
        ))
        .unwrap();
        explicit
            .apply(
                &mut options,
                &tokenizer,
                BudgetContext::Chat {
                    template: Some(&template),
                    prompt: "anchor",
                },
            )
            .unwrap();
        assert!(options.reasoning_budget.is_some());
        options.validate_processors(&tokenizer).unwrap();
        let missing =
            ReasoningRequest::extract(&request(json!({"reasoning_budget_tokens":0}))).unwrap();
        assert!(
            missing
                .apply(
                    &mut options,
                    &tokenizer,
                    BudgetContext::Chat {
                        template: Some(&template),
                        prompt: "anchor"
                    }
                )
                .unwrap_err()
                .to_string()
                .contains("patterned")
        );
    }
    #[test]
    fn reasoning_budget_wire_disabled_budget_does_not_require_a_protocol() {
        let tokenizer = tokenizer();
        let mut options = GenerateOptions::default();
        ReasoningRequest::extract(&request(json!({"reasoning_budget_tokens":-1})))
            .unwrap()
            .apply(&mut options, &tokenizer, BudgetContext::Raw)
            .unwrap();
        assert!(options.reasoning_budget.is_none());
        let enabled =
            ReasoningRequest::extract(&request(json!({"reasoning_budget_tokens":0}))).unwrap();
        enabled
            .apply(&mut options, &tokenizer, BudgetContext::Raw)
            .unwrap();
        assert!(options.reasoning_budget.is_none());
        let partial = ReasoningRequest::extract(&request(
            serde_json::json!({"reasoning_budget_tokens":0,"reasoning_budget_start_tag":"tok22"}),
        ))
        .unwrap();
        assert!(
            partial
                .apply(&mut options, &tokenizer, BudgetContext::Raw)
                .is_err()
        );
    }
    #[test]
    fn reasoning_budget_wire_frontend_control_hint_is_inapplicable_without_a_reasoning_channel() {
        let tokenizer = tokenizer();
        let template =
            ResponseTemplate::compile(&json!({"start_anchor":"anchor","fields":{"content":{}}}))
                .unwrap();
        for declared in [None, Some(&template)] {
            let controls = ReasoningRequest::extract(&request(
                json!({"reasoning_control":true,"thinking_budget_tokens":0}),
            ))
            .unwrap();
            let mut options = GenerateOptions::default();
            controls
                .apply(
                    &mut options,
                    &tokenizer,
                    BudgetContext::Chat {
                        template: declared,
                        prompt: "anchor hello",
                    },
                )
                .unwrap();
            assert!(options.reasoning_budget.is_none());
        }
    }
    #[test]
    fn reasoning_budget_wire_generation_prompt_seeds_grammar_even_without_a_budget() {
        let tokenizer = tokenizer();
        let controls =
            ReasoningRequest::extract(&request(json!({"generation_prompt":"tok22"}))).unwrap();
        let mut options = GenerateOptions {
            constraint: Some(GenerateConstraint::Regex {
                pattern: "tok22tok23".into(),
                prefix: String::new(),
            }),
            ..Default::default()
        };
        controls
            .apply(&mut options, &tokenizer, BudgetContext::Raw)
            .unwrap();
        assert!(
            matches!(options.constraint,Some(GenerateConstraint::Regex {ref prefix,..}) if prefix=="tok22")
        );
        assert!(options.reasoning_budget.is_none());
        options.constraint = Some(GenerateConstraint::Json {
            prefix: String::new(),
        });
        let controls =
            ReasoningRequest::extract(&request(json!({"generation_prompt":"{\"value\":"})))
                .unwrap();
        controls
            .apply(&mut options, &tokenizer, BudgetContext::Raw)
            .unwrap();
        assert!(
            matches!(options.constraint,Some(GenerateConstraint::Json {ref prefix}) if prefix=="{\"value\":")
        );
    }
    fn model_state() -> (tempfile::TempDir, crate::AppState) {
        model_state_with_config(crate::ServerConfig::default())
    }
    fn model_state_with_config(
        config: crate::ServerConfig,
    ) -> (tempfile::TempDir, crate::AppState) {
        let scratch = tempfile::tempdir().unwrap();
        let source =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
        for entry in std::fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                std::fs::copy(entry.path(), scratch.path().join(entry.file_name())).unwrap();
            }
        }
        std::fs::write(
            scratch.path().join("tokenizer_config.json"),
            json!({
                "chat_template":"{% for m in messages %}{{ m.content }} {% endfor %}anchor tok22",
                "response_template":{"start_anchor":"anchor","fields":{
                    "reasoning_content":{"open":"tok22","close":"tok23"},"content":{}
                }}
            })
            .to_string(),
        )
        .unwrap();
        let state =
            crate::AppState::load_with_config(scratch.path(), Some("budget-wire".into()), config)
                .unwrap();
        (scratch, state)
    }
    async fn post(router: axum::Router, body: Value) -> axum::response::Response {
        use tower::ServiceExt as _;
        router
            .oneshot(
                axum::http::Request::post("/v1/chat/completions")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }
    #[tokio::test]
    async fn reasoning_budget_wire_real_onnx_closes_reasoning_in_buffered_and_logprob_sse() {
        let (_scratch, state) = model_state();
        let router = crate::app(state);
        for budget in [0, 1] {
            for stream in [false, true] {
                for logprobs in [false, true] {
                    let response=post(router.clone(),json!({"messages":[{"role":"user","content":"hello"}],
                        "thinking_budget_tokens":budget,"reasoning_control":true,"max_tokens":budget+2,
                        "temperature":0,"logit_bias":{"4":1000},"stream":stream,"logprobs":logprobs})).await;
                    assert_eq!(response.status(), axum::http::StatusCode::OK);
                    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                        .await
                        .unwrap();
                    let expected = if budget == 0 { "" } else { "hello" };
                    if stream {
                        let text = std::str::from_utf8(&bytes).unwrap();
                        let mut content = String::new();
                        let mut reasoning = String::new();
                        for line in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
                            if line == "[DONE]" {
                                continue;
                            }
                            let chunk: Value = serde_json::from_str(line).unwrap();
                            assert!(chunk.get("error").is_none(), "{chunk}");
                            content.push_str(
                                chunk["choices"][0]["delta"]["content"]
                                    .as_str()
                                    .unwrap_or(""),
                            );
                            reasoning.push_str(
                                chunk["choices"][0]["delta"]["reasoning_content"]
                                    .as_str()
                                    .unwrap_or(""),
                            );
                        }
                        assert_eq!(content, "hello", "{text}");
                        assert_eq!(reasoning, expected, "{text}");
                    } else {
                        let body: Value = serde_json::from_slice(&bytes).unwrap();
                        assert_eq!(body["choices"][0]["message"]["content"], "hello", "{body}");
                        assert_eq!(
                            body["choices"][0]["message"]["reasoning_content"]
                                .as_str()
                                .unwrap_or(""),
                            expected,
                            "{body}"
                        );
                        assert_eq!(body["usage"]["completion_tokens"], budget + 2, "{body}");
                    }
                }
            }
        }
    }
    #[tokio::test]
    async fn reasoning_budget_wire_banned_closer_is_rejected_before_sse_headers() {
        let (_scratch, state) = model_state();
        let response = post(
            crate::app(state),
            json!({"messages":[{"role":"user","content":"hello"}],
            "reasoning_budget_tokens":0,"stream":true,"logit_bias":{"23":false},"max_tokens":2}),
        )
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
        assert!(
            !response.headers()[axum::http::header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .contains("event-stream")
        );
        let bytes = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("closer token 23 is banned"));
    }
    #[tokio::test]
    async fn reasoning_budget_wire_native_and_legacy_routes_keep_raw_protocol_text() {
        use tower::ServiceExt as _;
        let (_scratch, state) = model_state();
        let router = crate::app(state);
        for path in ["/completion", "/v1/completions"] {
            for stream in [false, true] {
                let mut body = json!({"prompt":"hello tok22","temperature":0,"stream":stream,
                    "reasoning_budget_tokens":0,"reasoning_budget_start_tag":"tok22",
                    "reasoning_budget_end_tag":"tok23","generation_prompt":"tok22","logit_bias":{"4":1000}});
                body[if path == "/completion" {
                    "n_predict"
                } else {
                    "max_tokens"
                }] = json!(2);
                let response = router
                    .clone()
                    .oneshot(
                        axum::http::Request::post(path)
                            .header(axum::http::header::CONTENT_TYPE, "application/json")
                            .body(axum::body::Body::from(body.to_string()))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), axum::http::StatusCode::OK, "{path}");
                let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                    .await
                    .unwrap();
                if stream {
                    let text = std::str::from_utf8(&bytes).unwrap();
                    assert!(
                        text.contains("tok23") && text.contains("hello"),
                        "{path}: {text}"
                    );
                    assert!(!text.contains("\"error\":"), "{path}: {text}");
                } else {
                    let result: Value = serde_json::from_slice(&bytes).unwrap();
                    let output = if path == "/completion" {
                        &result["content"]
                    } else {
                        &result["choices"][0]["text"]
                    };
                    assert_eq!(output, "tok23 hello", "{path}: {result}");
                }
            }
        }
    }
    #[tokio::test]
    async fn reasoning_budget_wire_anthropic_enabled_uses_the_real_sampler() {
        use tower::ServiceExt as _;
        let (_scratch, state) = model_state();
        let router = crate::app(state);
        for stream in [false, true] {
            let body = json!({"model":"budget-wire","messages":[{"role":"user","content":"hello"}],
                "thinking":{"type":"enabled","budget_tokens":0},"max_tokens":2,
                "temperature":0,"logit_bias":{"4":1000},"stream":stream});
            let response = router
                .clone()
                .oneshot(
                    axum::http::Request::post("/v1/messages")
                        .header(axum::http::header::CONTENT_TYPE, "application/json")
                        .body(axum::body::Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap();
            if stream {
                let text = std::str::from_utf8(&bytes).unwrap();
                assert!(text.contains("hello"), "{text}");
                assert!(!text.contains("\"type\":\"error\""), "{text}");
            } else {
                let result: Value = serde_json::from_slice(&bytes).unwrap();
                assert!(
                    result["content"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|part| part["type"] == "text" && part["text"] == "hello"),
                    "{result}"
                );
                assert_eq!(result["usage"]["output_tokens"], 2, "{result}");
            }
        }
    }
    #[test]
    fn reasoning_budget_wire_optional_control_does_not_break_parser_only_models() {
        let tokenizer = tokenizer();
        let mut options = GenerateOptions::default();
        let template = ResponseTemplate::compile(&json!({"start_anchor":"anchor","fields":{
            "reasoning_content":{"open":"[r]","close":"[/r]"},"content":{}
        }}))
        .unwrap();
        let hint = ReasoningRequest::extract(&request(json!({"reasoning_control":true}))).unwrap();
        hint.apply(
            &mut options,
            &tokenizer,
            BudgetContext::Chat {
                template: Some(&template),
                prompt: "anchor[r]",
            },
        )
        .unwrap();
        assert!(options.reasoning_budget.is_none());
        let finite = ReasoningRequest::extract(&request(
            json!({"reasoning_control":true,"reasoning_budget_tokens":0}),
        ))
        .unwrap();
        assert!(
            finite
                .apply(
                    &mut options,
                    &tokenizer,
                    BudgetContext::Chat {
                        template: Some(&template),
                        prompt: "anchor[r]"
                    }
                )
                .is_err()
        );
    }

    #[cfg(feature = "native-backend")]
    #[tokio::test]
    async fn reasoning_budget_wire_native_backend_observes_the_mandatory_budget() {
        let (_scratch, state) = model_state_with_config(crate::ServerConfig {
            engine_config: onnx_genai::EngineConfig {
                decode_backend: onnx_genai::engine::EngineDecodeBackend::Native,
                ..Default::default()
            },
            ..Default::default()
        });
        let router = crate::app(state);
        for budget in [0, 1] {
            let response=post(router.clone(),json!({"messages":[{"role":"user","content":"hello"}],
                "reasoning_budget_tokens":budget,"max_tokens":budget+2,"temperature":0,"logit_bias":{"4":1000}})).await;
            assert_eq!(response.status(), axum::http::StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap();
            let result: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                result["choices"][0]["message"]["content"], "hello",
                "{result}"
            );
            assert_eq!(
                result["choices"][0]["message"]["reasoning_content"]
                    .as_str()
                    .unwrap_or(""),
                if budget == 0 { "" } else { "hello" },
                "{result}"
            );
        }
    }
}
