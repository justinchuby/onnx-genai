# Reranking packages and APIs

The server executes query/document pairs through a package-declared scoring
workflow. A `profiles.<name>` entry with `kind: reranking`, `version: '1'` and an
explicit `outputs.score` binding enables the API. The model workflow computes the
score; it must expose a finite match probability in `[0, 1]`.

The v1 profile is a text token-pair contract:

- The package tokenizer encodes `(query, document)` with its declared pair
  post-processor, special tokens, padding and truncation policy.
- One or more `prompt_tokens` runtime inputs receive the paired token IDs.
- Application inputs named `attention_mask` and `token_type_ids` receive the
  tokenizer's corresponding feature rows. Their physical workflow port names
  are arbitrary; the semantic application source identifies their values.
- Pair inputs are int64 `[batch, sequence]` tensors accepting one batch row.
  A fixed sequence length requires matching tokenizer padding/truncation.
- `outputs.score` names a floating scalar or one scalar per batch row, with
  workflow role `tensor`. Supported score dtypes are float16, bfloat16 and
  float32. A multi-class/logit head must perform its declared activation and
  class selection inside the workflow before emitting this score.

Required unsupported profile versions, ambiguous profiles and invalid bindings
fail package admission. An unsupported version marked `ignorable` does not
advertise reranking capability. The public capability flag is derived from the
admitted contract, independent of model name or execution provider.

The executable example is
[`tests/fixtures/onnx_genai_workflows/reranking`](../tests/fixtures/onnx_genai_workflows/reranking).
Its ONNX classifier jointly consumes paired IDs, masks and segment IDs. This is
a tiny protocol fixture rather than a trained relevance model.

## HTTP contract

`POST /reranking`, `/rerank`, `/v1/rerank` and `/v1/reranking` share one handler.

With `documents`, the response uses Jina format:

```json
{"model":"ranker","object":"list","usage":{"prompt_tokens":10,"total_tokens":10},"results":[{"index":1,"relevance_score":0.8},{"index":0,"relevance_score":0.2}]}
```

With `texts`, the response is a TEI array of `{index, score}`. `return_text: true`
adds the normalized document text to TEI rows. `query` and each document can be
a string or an object whose `text` is a string/string array; arrays concatenate
in order. Media requires a different scoring contract and is rejected by v1.

Results sort by descending probability; equal scores retain input order.
`top_n` defaults to all documents and can be zero. Every submitted pair is
scored before truncating the response, and usage counts the non-padding tokens
of every scored pair. Like the pinned upstream behavior, `texts` selects TEI
format while `documents` takes input precedence if both keys are supplied.

`--max-rerank-input-tokens` defaults to 8192 per pair and
`--max-rerank-documents` defaults to 128. Both must be positive. Model context
limits also apply. All pairs are prepared and validated before any model work;
individual scoring calls use the ordinary driver admission permit. Cancelling
a request stops submission of subsequent pairs.
