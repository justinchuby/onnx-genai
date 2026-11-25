//! SystemOne transport. Decision execution is a declared non-generative workflow.
use super::*;
use onnx_genai_engine::decision::{
    DecisionAnswer, DecisionInput, DecisionLimits, DecisionMedia, DecisionQuestion,
    PreparedDecision,
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
struct MediaUrl {
    kind: onnx_genai_metadata::EncodedMediaKind,
    source: String,
    field: String,
}
fn media_uri(value: &Value, field: &str) -> Result<MediaUrl, ApiError> {
    let source = value.as_str().ok_or_else(|| {
        ApiError::invalid_request(format!("{field} must be an image/audio data URI string"))
    })?;
    let kind = if source.starts_with("data:image/") {
        onnx_genai_metadata::EncodedMediaKind::Image
    } else if source.starts_with("data:audio/") {
        onnx_genai_metadata::EncodedMediaKind::Audio
    } else {
        return Err(ApiError::invalid_request(format!(
            "{field} must be a data:image/... or data:audio/... base64 URI; remote URLs and video are not supported by this decision media contract"
        )));
    };
    Ok(MediaUrl {
        kind,
        source: source.to_owned(),
        field: field.to_owned(),
    })
}
fn audio_url(value: &Value, field: &str) -> Result<MediaUrl, ApiError> {
    let source = value
        .get("data")
        .or_else(|| value.get("url"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ApiError::invalid_request(format!(
                "{field}.data or .url must contain base64 PCM16 WAV or MP3 bytes or an audio data URI"
            ))
        })?;
    if source.contains("://")
        || source.starts_with("file:")
        || (source.starts_with("data:") && !source.starts_with("data:audio/"))
    {
        return Err(ApiError::invalid_request(format!(
            "{field} must contain base64 PCM16 WAV or MP3 audio; remote/file URLs and non-audio data URIs are not fetched"
        )));
    }
    Ok(MediaUrl {
        kind: onnx_genai_metadata::EncodedMediaKind::Audio,
        source: source.to_owned(),
        field: field.to_owned(),
    })
}
fn spoken_url(question: &Value, id: &str) -> Result<Option<MediaUrl>, ApiError> {
    let Some(instructions) = question
        .get("instructions")
        .filter(|v| v.get("type").and_then(Value::as_str) == Some("input_audio"))
    else {
        return Ok(None);
    };
    let value = instructions.get("input_audio").ok_or_else(|| {
        ApiError::invalid_request(format!(
            "questions.{id}.instructions.input_audio is required for spoken instructions"
        ))
    })?;
    audio_url(value, &format!("questions.{id}.instructions.input_audio")).map(Some)
}
/// Upstream order: files, images alias, then typed chat-state media parts.
fn state_media(request: &mut DecisionRequest) -> Result<Vec<MediaUrl>, ApiError> {
    if request.videos.as_ref().is_some_and(|v| !v.is_empty()) {
        return Err(ApiError::invalid_request(
            "videos need a declared executable video decision contract; this transport currently admits image/audio state only",
        ));
    }
    let mut media = Vec::new();
    for (field, values) in [("files", &request.files), ("images", &request.images)] {
        for (index, value) in values.iter().flatten().enumerate() {
            media.push(media_uri(value, &format!("{field}[{index}]"))?);
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
                        let entry = media_uri(
                            value.get("url").unwrap_or(value),
                            &format!("{field}.image_url"),
                        )?;
                        if entry.kind != onnx_genai_metadata::EncodedMediaKind::Image {
                            return Err(ApiError::invalid_request(format!(
                                "{field}.image_url must contain image data; use input_audio for audio"
                            )));
                        }
                        media.push(entry);
                    }
                    Some("input_audio") => {
                        let value = part.get("input_audio").ok_or_else(|| {
                            ApiError::invalid_request(format!("{field}.input_audio is required"))
                        })?;
                        media.push(audio_url(value, &format!("{field}.input_audio"))?);
                    }
                    Some("video_url") => {
                        return Err(ApiError::invalid_request(format!(
                            "{field} requires executable video decision support; image/audio profiles cannot consume video"
                        )));
                    }
                    _ => text.push(part),
                }
            }
            *parts = text;
        }
    }
    Ok(media)
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
    let media_urls = state_media(&mut request)?;
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
        .admit_media(&media_urls.iter().map(|url| url.kind).collect::<Vec<_>>())
        .map_err(|error| ApiError::invalid_request(error.to_string()))?;
    let questions = request
        .questions
        .into_iter()
        .map(|(id, value)| {
            let spoken = spoken_url(&value, &id)?;
            DecisionQuestion::from_value(contract, value, state.config.max_decision_options)
                .map(|q| (id.clone(), q, spoken))
                .map_err(|error| ApiError::invalid_request(format!("questions.{id}: {error:#}")))
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
    let mut owner = crate::media_input::MediaRequest::new(
        state.config.max_decision_media_bytes,
        "--max-decision-media-bytes",
    );
    let mut media = Vec::with_capacity(media_urls.len());
    for url in &media_urls {
        media.push(DecisionMedia {
            kind: url.kind,
            bytes: owner.load_one(url.kind, &url.source, &url.field).await?,
        });
    }
    let mut spoken_inputs = Vec::with_capacity(questions.len());
    for (_, _, url) in &questions {
        spoken_inputs.push(match url {
            Some(url) => Some(owner.load_one(url.kind, &url.source, &url.field).await?),
            None => None,
        });
    }
    // Prepare all variants/questions before admitting any model work.
    let prepared = questions
        .into_iter()
        .zip(spoken_inputs)
        .map(|((id, q, _), spoken)| {
            PreparedDecision::prepare(
                &handle.tokenizer,
                DecisionInput {
                    state: &request.state,
                    media: &media,
                    spoken: spoken.as_ref(),
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
        let result = handle.engine.decide(q).await.map_err(|error| {
            let mut error = map_generate_submit_error(error);
            error.message = format!("questions.{id}: {}", error.message);
            error
        })?;
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
