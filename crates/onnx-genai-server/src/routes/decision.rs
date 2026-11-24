//! SystemOne transport. Decision execution is a declared non-generative workflow.
use super::*;
use onnx_genai_engine::decision::{
    DecisionAnswer, DecisionKind, DecisionLimits, DecisionOption, DecisionQuestion,
    PreparedDecision,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, num::NonZeroUsize};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
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
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Question {
    #[serde(rename = "type")]
    kind: DecisionKind,
    #[serde(deserialize_with = "present_value")]
    instructions: Value,
    #[serde(default)]
    criteria: Value,
}
fn present_value<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Value, D::Error> {
    Value::deserialize(deserializer)
}
fn question(id: &str, value: Value, limit: usize) -> Result<DecisionQuestion, ApiError> {
    let q: Question = serde_json::from_value(value).map_err(|e| {
        ApiError::invalid_request(format!(
            "questions.{id}: {e}; supply type, instructions and criteria"
        ))
    })?;
    let invalid = |message: &str| ApiError::invalid_request(format!("questions.{id}: {message}"));
    if q.instructions.get("type").and_then(Value::as_str) == Some("input_audio") {
        return Err(invalid(
            "spoken instructions require a multimodal decision contract; supply text instructions",
        ));
    }
    let options = match q.kind {
        DecisionKind::Choice => {
            let object = q
                .criteria
                .as_object()
                .filter(|x| !x.is_empty())
                .ok_or_else(|| {
                    invalid(
                        "choice criteria must be a nonempty object mapping keys to descriptions",
                    )
                })?;
            let mut options = object
                .iter()
                .map(|(key, description)| DecisionOption {
                    key: key.clone(),
                    description: description.clone(),
                })
                .collect::<Vec<_>>();
            options.sort_by(|a, b| a.key.cmp(&b.key));
            options
        }
        DecisionKind::Score => {
            let array = q
                .criteria
                .as_array()
                .filter(|x| (2..=10).contains(&x.len()))
                .ok_or_else(|| invalid("score criteria must contain 2..=10 level descriptions"))?;
            array
                .iter()
                .enumerate()
                .map(|(i, description)| DecisionOption {
                    key: i.to_string(),
                    description: description.clone(),
                })
                .collect()
        }
        DecisionKind::Noul => {
            if !q.criteria.is_null() && !q.criteria.is_object() {
                return Err(invalid(
                    "noul criteria must be an object describing false/true or null",
                ));
            }
            if q.criteria.as_object().is_some_and(|object| {
                object
                    .keys()
                    .any(|key| !matches!(key.as_str(), "false" | "true"))
            }) {
                return Err(invalid(
                    "noul criteria may only describe false and true; use those explicit keys",
                ));
            }
            ["false", "true"]
                .into_iter()
                .map(|key| DecisionOption {
                    key: key.into(),
                    description: q.criteria.get(key).cloned().unwrap_or(Value::Null),
                })
                .collect()
        }
    };
    if options.len() > limit {
        return Err(invalid(&format!(
            "{} options exceeds configured maximum {limit}; reduce criteria or increase --max-decision-options",
            options.len()
        )));
    }
    Ok(DecisionQuestion {
        kind: q.kind,
        instructions: q.instructions,
        options,
    })
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
    let questions = request
        .questions
        .into_iter()
        .map(|(id, value)| {
            question(&id, value, state.config.max_decision_options.get()).map(|q| (id, q))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let handle = resolve_model(&state.registry, &request.model).await?;
    let contract=handle.engine.decision.as_deref().ok_or_else(||ApiError{status:StatusCode::NOT_IMPLEMENTED,kind:"not_supported_error",message:format!("model '{}' declares no decision v1 profile; select a package with profiles.<name>.kind: decision and a calibrated probability workflow",handle.id),retry_after_secs:None})?;
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
                contract,
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
            .checked_add(result.prompt_tokens)
            .ok_or_else(|| ApiError::internal("decision input usage overflow; split questions"))?;
        let probabilities = result
            .options
            .iter()
            .zip(&result.probabilities)
            .map(|(option, p)| (option.key.clone(), json!(p)))
            .collect::<serde_json::Map<_, _>>();
        let answer = match result.answer {
            DecisionAnswer::Choice { key, confidence } => {
                json!({"type":"choice","choice":key,"confidence":confidence,"probabilities":probabilities})
            }
            DecisionAnswer::Score { score, confidence } => {
                let legend = result
                    .options
                    .iter()
                    .map(|o| (o.key.clone(), o.description.clone()))
                    .collect::<serde_json::Map<_, _>>();
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
