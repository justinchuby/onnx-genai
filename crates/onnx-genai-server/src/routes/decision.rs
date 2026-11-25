//! SystemOne transport. Decision execution is a declared non-generative workflow.
use super::*;
use onnx_genai_engine::decision::{
    DecisionAnswer, DecisionLimits, DecisionQuestion, PreparedDecision,
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
fn embedded_media(state: &Value) -> bool {
    state.as_array().is_some_and(|messages| {
        messages.iter().any(|m| {
            m.get("content")
                .and_then(Value::as_array)
                .is_some_and(|parts| {
                    parts.iter().any(|part| {
                        part.get("type")
                            .and_then(Value::as_str)
                            .is_some_and(|kind| {
                                matches!(kind, "image_url" | "input_audio" | "video_url")
                            })
                    })
                })
        })
    })
}
pub(crate) async fn decide(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<DecisionRequest>,
) -> Result<Json<Value>, ApiError> {
    if request.questions.is_empty()
        || request.questions.len() > state.config.max_decision_questions.get()
    {
        return Err(ApiError::invalid_request(format!(
            "questions must contain 1..={} entries; split the request or raise --max-decision-questions",
            state.config.max_decision_questions
        )));
    }
    if request.files.as_ref().is_some_and(|x| !x.is_empty())
        || request.images.as_ref().is_some_and(|x| !x.is_empty())
        || request.videos.as_ref().is_some_and(|x| !x.is_empty())
        || embedded_media(&request.state)
    {
        return Err(ApiError::invalid_request(
            "decision v1 accepts text/structured state without media; this package needs a declared multimodal decision contract before files/images/audio/video can be evaluated",
        ));
    }
    let handle = resolve_model(&state.registry, &request.model).await?;
    let contract=handle.engine.decision.as_ref().ok_or_else(||ApiError{status:StatusCode::NOT_IMPLEMENTED,kind:"not_supported_error",message:format!("model '{}' declares no decision v1 profile; select a package with profiles.<name>.kind: decision and a calibrated probability workflow",handle.id),retry_after_secs:None})?;
    if let Some(max) = contract.input_policy().max_questions
        && request.questions.len() > max.get()
    {
        return Err(ApiError::invalid_request(format!(
            "questions exceeds model profile '{}' max_questions {max}; split the request",
            contract.profile(),
        )));
    }
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
    // Prepare all variants/questions before admitting any model work.
    let prepared = questions
        .into_iter()
        .map(|(id, q)| {
            PreparedDecision::prepare(
                &handle.tokenizer,
                &request.state,
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
