//! SystemOne transport. Decision execution is a declared non-generative workflow.
use super::*;
use onnx_genai_engine::decision::{
    DecisionAnswer, DecisionInput, DecisionLimits, DecisionQuestion, PreparedDecision,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, num::NonZeroUsize};

#[derive(Deserialize)]
pub(crate) struct DecisionRequest {
    #[serde(default)]
    model: String,
    #[serde(deserialize_with = "present_value")]
    state: Value,
    questions: BTreeMap<String, Value>,
    #[serde(default)]
    files: Option<Vec<Value>>,
    #[serde(default)]
    images: Option<Vec<Value>>,
    #[serde(default)]
    videos: Option<Vec<Value>>,
}
fn present_value<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Value, D::Error> {
    Value::deserialize(deserializer)
}
fn image_url(value: &Value, field: &str) -> Result<String, ApiError> {
    let url = value.as_str().ok_or_else(|| {
        ApiError::invalid_request(format!("{field} must be an image data URI string"))
    })?;
    if !url.starts_with("data:image/") {
        return Err(ApiError::invalid_request(format!(
            "{field} must be a data:image/...;base64,... URI; decision image inputs do not fetch remote URLs or accept audio/video"
        )));
    }
    Ok(url.to_owned())
}
/// Upstream order: files, images, then chat-state image parts in message order.
/// Images leave textual state; their position is retained in the native inputs.
fn state_images(request: &mut DecisionRequest) -> Result<Vec<String>, ApiError> {
    if request.videos.as_ref().is_some_and(|x| !x.is_empty()) {
        return Err(ApiError::invalid_request(
            "videos are not supported by the decision image contract; use a package declaring the required media capabilities",
        ));
    }
    let mut urls = Vec::new();
    for (field, values) in [("files", &request.files), ("images", &request.images)] {
        for (index, value) in values.iter().flatten().enumerate() {
            urls.push(image_url(value, &format!("{field}[{index}]"))?);
        }
    }
    let messages = match &mut request.state {
        Value::Object(object) => object.get_mut("messages"),
        value @ Value::Array(_) => Some(value),
        _ => None,
    };
    if let Some(Value::Array(messages)) = messages {
        for (index, message) in messages.iter_mut().enumerate() {
            let Some(Value::Array(parts)) = message.get_mut("content") else {
                continue;
            };
            let mut text = Vec::new();
            for (part_index, part) in std::mem::take(parts).into_iter().enumerate() {
                let field = format!("state.messages[{index}].content[{part_index}]");
                match part.get("type").and_then(Value::as_str) {
                    Some("image_url") => {
                        let value = part.get("image_url").ok_or_else(|| {
                            ApiError::invalid_request(format!("{field}.image_url is required"))
                        })?;
                        let value = value.get("url").unwrap_or(value);
                        urls.push(image_url(value, &format!("{field}.image_url"))?);
                    }
                    Some("input_audio" | "video_url") => {
                        return Err(ApiError::invalid_request(format!(
                            "{field} requires an audio/video decision contract; the declared image path does not accept this media"
                        )));
                    }
                    _ => text.push(part),
                }
            }
            *parts = text;
        }
    }
    Ok(urls)
}
pub(crate) async fn decide(
    State(state): State<AppState>,
    ApiJson(mut request): ApiJson<DecisionRequest>,
) -> Result<Json<Value>, ApiError> {
    if request.questions.is_empty()
        || request.questions.len() > state.config.max_decision_questions.get()
    {
        return Err(ApiError::invalid_request(format!(
            "questions must contain 1..={} entries; split the request or raise --max-decision-questions",
            state.config.max_decision_questions
        )));
    }
    let image_urls = state_images(&mut request)?;
    let handle = resolve_model(&state.registry, &request.model).await?;
    let contract=handle.engine.decision.as_ref().ok_or_else(||ApiError{status:StatusCode::NOT_IMPLEMENTED,kind:"not_supported_error",message:format!("model '{}' declares no decision profile; select a package with profiles.<name>.kind: decision and a calibrated probability workflow",handle.id),retry_after_secs:None})?;
    if let Some(max) = contract.input_policy().max_questions
        && request.questions.len() > max.get()
    {
        return Err(ApiError::invalid_request(format!(
            "questions exceeds model profile '{}' max_questions {max}; split the request",
            contract.profile(),
        )));
    }
    contract
        .admit_images(image_urls.len())
        .map_err(|error| ApiError::invalid_request(error.to_string()))?;
    let questions = request
        .questions
        .into_iter()
        .map(|(id, value)| {
            DecisionQuestion::from_value(contract, value, state.config.max_decision_options)
                .map(|q| (id.clone(), q))
                .map_err(|e| ApiError::invalid_request(format!("questions.{id}: {e:#}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let limit = NonZeroUsize::new(
        state
            .config
            .max_decision_input_tokens
            .get()
            .min(handle.model_max_context.unwrap_or(usize::MAX)),
    )
    .ok_or_else(|| {
        ApiError::internal("decision input budget resolved to zero; repair model context metadata")
    })?;
    let mut media = crate::image_input::ImageRequest::new(
        state.config.max_decision_media_bytes,
        "--max-decision-media-bytes",
    );
    let images = media.load(&image_urls, "decision images").await?;
    // Prepare all variants/questions before admitting any model work.
    let prepared = questions
        .into_iter()
        .map(|(id, q)| {
            PreparedDecision::prepare(
                &handle.tokenizer,
                DecisionInput {
                    state: &request.state,
                    images: &images,
                },
                q,
                DecisionLimits {
                    max_input_tokens: limit,
                    template: onnx_genai_ort::TemplateLimits {
                        max_output_bytes: state.config.max_decision_prompt_bytes,
                        max_operations: state.config.max_decision_template_operations,
                    },
                },
            )
            .map(|q| (id.clone(), q))
            .map_err(|e| ApiError::invalid_request(format!("questions.{id}: {e:#}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut answers = serde_json::Map::new();
    let mut tokens = 0usize;
    for (id, q) in prepared {
        let result = handle
            .engine
            .decide(q)
            .await
            .map_err(map_generate_submit_error)?;
        tokens = tokens
            .checked_add(result.prompt_tokens())
            .ok_or_else(|| ApiError::internal("decision input usage overflow; split questions"))?;
        let probabilities = result
            .options()
            .iter()
            .zip(result.probabilities())
            .map(|(option, p)| (option.key.clone(), json!(p)))
            .collect::<serde_json::Map<_, _>>();
        let answer = match result.answer() {
            DecisionAnswer::Choice { key, confidence } => {
                json!({"type":"choice","choice":key,"confidence":confidence,"probabilities":probabilities})
            }
            DecisionAnswer::Score { score, confidence } => {
                let legend = result.legend().map_err(|e| {
                    ApiError::internal(format!("format decision score legend: {e:#}"))
                })?;
                json!({"type":"score","score":score,"confidence":confidence,"probabilities":probabilities,"legend":legend})
            }
            DecisionAnswer::Noul { probability } => json!({"type":"noul","noul":probability}),
        };
        answers.insert(id, answer);
    }
    Ok(Json(
        json!({"model":handle.id,"answers":answers,"usage":{"input_tokens":tokens,"output_tokens":0}}),
    ))
}
