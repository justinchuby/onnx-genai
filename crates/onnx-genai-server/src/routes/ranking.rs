//! Native Jina/TEI reranking over declared token-pair scoring workflows.
use super::*;
use onnx_genai_engine::ranking::PreparedTokenPair;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RerankRequest {
    #[serde(default)]
    model: String,
    query: Value,
    #[serde(default)]
    documents: Option<Vec<Value>>,
    #[serde(default)]
    texts: Option<Vec<Value>>,
    #[serde(default)]
    top_n: Option<usize>,
    #[serde(default)]
    return_text: bool,
}
fn item_text(item: Value, field: &str) -> Result<String, ApiError> {
    if let Value::String(text) = item {
        return Ok(text);
    }
    let Value::Object(mut object) = item else {
        return Err(ApiError::invalid_request(format!(
            "{field} must be text or an object with text and optional image fields"
        )));
    };
    let text = match object.remove("text") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text,
        Some(Value::Array(parts)) => parts
            .into_iter()
            .enumerate()
            .map(|(index, part)| {
                part.as_str().map(str::to_owned).ok_or_else(|| {
                    ApiError::invalid_request(format!("{field}.text[{index}] must be a string"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .concat(),
        Some(_) => {
            return Err(ApiError::invalid_request(format!(
                "{field}.text must be a string or string array"
            )));
        }
    };
    if let Some(image) = object.remove("image")
        && !image.is_null()
        && image != json!([])
    {
        return Err(ApiError::invalid_request(format!(
            "{field}.image requires a multimodal reranking contract; this package's reranking v1 profile accepts text pairs"
        )));
    }
    if !object.is_empty() {
        return Err(ApiError::invalid_request(format!(
            "{field} contains unsupported fields {:?}; use text and omit media for a text-pair profile",
            object.keys().collect::<Vec<_>>()
        )));
    }
    Ok(text)
}

pub(crate) async fn rerank(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<RerankRequest>,
) -> Result<Json<Value>, ApiError> {
    let tei = request.texts.is_some();
    // Upstream selects response shape from texts, while documents takes input
    // precedence when both keys are present.
    let documents = request.documents.or(request.texts).ok_or_else(|| {
        ApiError::invalid_request(
            "documents or texts must provide a nonempty reranking input array",
        )
    })?;
    if documents.is_empty() || documents.len() > state.config.max_rerank_documents.get() {
        return Err(ApiError::invalid_request(format!(
            "documents/texts must contain 1..={} items; split larger requests or raise --max-rerank-documents",
            state.config.max_rerank_documents
        )));
    }
    let query = item_text(request.query, "query")?;
    let documents = documents
        .into_iter()
        .enumerate()
        .map(|(index, value)| item_text(value, &format!("documents[{index}]")))
        .collect::<Result<Vec<_>, _>>()?;
    let handle = resolve_model(&state.registry, &request.model).await?;
    let contract=handle.engine.reranking.as_deref().ok_or_else(||ApiError::invalid_request(format!("model '{}' declares no reranking v1 profile; load a package with a token-pair workflow and profiles.<name>.outputs.score",handle.id)))?;
    let limit = state
        .config
        .max_rerank_input_tokens
        .get()
        .min(handle.model_max_context.unwrap_or(usize::MAX));
    let limit=std::num::NonZeroUsize::new(limit).ok_or_else(||ApiError::internal("reranking resolved a zero input budget; repair model context metadata or server limits"))?;
    // Validate every pair before submitting any model work. Each execution owns
    // one ordinary driver permit, so cancellation never strands queued pairs.
    let pairs = documents
        .iter()
        .enumerate()
        .map(|(index, document)| {
            PreparedTokenPair::prepare(&handle.tokenizer, contract, &query, document, limit)
                .map_err(|err| {
                    ApiError::invalid_request(format!(
                        "documents[{index}] could not be prepared for reranking: {err:#}"
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut ranked = Vec::with_capacity(pairs.len());
    let mut prompt_tokens = 0usize;
    for (index, pair) in pairs.into_iter().enumerate() {
        let score = handle
            .engine
            .score_pair(pair)
            .await
            .map_err(map_generate_submit_error)?;
        prompt_tokens = prompt_tokens
            .checked_add(score.prompt_tokens)
            .ok_or_else(|| {
                ApiError::internal("reranking token count overflow; split the document request")
            })?;
        ranked.push((index, score.probability));
    }
    ranked.sort_by(|(left_index, left), (right_index, right)| {
        right
            .total_cmp(left)
            .then_with(|| left_index.cmp(right_index))
    });
    ranked.truncate(request.top_n.unwrap_or(ranked.len()));
    let rows = ranked
        .into_iter()
        .map(|(index, score)| {
            if tei {
                let mut row = json!({"index":index,"score":score});
                if request.return_text {
                    row["text"] = json!(documents[index]);
                }
                row
            } else {
                json!({"index":index,"relevance_score":score})
            }
        })
        .collect::<Vec<_>>();
    Ok(Json(if tei {
        json!(rows)
    } else {
        json!({"model":if request.model.is_empty(){&handle.id}else{&request.model},"object":"list","usage":{"prompt_tokens":prompt_tokens,"total_tokens":prompt_tokens},"results":rows})
    }))
}
