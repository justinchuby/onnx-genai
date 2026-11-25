# Reranking text and images

`POST /rerank`, `/reranking`, `/v1/rerank` and `/v1/reranking` execute the model
package's declared pair-scoring workflow. There is no architecture or tensor-name
inference. The workflow must return a final scalar relevance probability in
`[0,1]`; raw logits require explicit sigmoid/softmax in the model.

## Requests

```json
{
  "model": "vision-ranker",
  "query": {"text": "a red object", "image": "https://example.org/query.png"},
  "documents": [
    {"text": ["first ", "candidate"], "image": ["https://example.org/a-front.png", "https://example.org/a-back.png"]},
    {"text": "second candidate", "image": ["https://example.org/b-front.png", "https://example.org/b-back.png"]}
  ],
  "top_n": 2
}
```

A query or document is a string, or an object with `text` and `image`. Text arrays
concatenate in order. An image is an HTTP(S) URL, image data URI or raw base64;
an array supplies multiple ordered images. PNG and JPEG are supported. Image-only
objects are valid when the package admits them. Video, `fps` and `max_frames`
remain invalid reranking fields.

Image capacity and required positions come from the package. A text-only v1
profile rejects images before fetching them. A v2 profile maps query and document
image positions to explicit workflow inputs. Missing required images and excess
images fail before any remote fetch or model submission. Optional images follow
the workflow's declared presence/default behavior; the server does not invent a
fallback image. Encoded image contents are fully validated before model work;
malformed pixel data receives a request error naming its query/document position.

`documents` produces Jina-style `results` with `index` and `relevance_score`,
ordered by descending score and stable input index for ties. `top_n` defaults to
all documents and may be zero. Every pair is scored before result truncation;
usage covers all scored pairs. Cancelling a request stops submission of later
pairs. Supplying `texts`
selects TEI rows with `index` and `score`; `return_text: true` includes the
concatenated text. When both input keys are present, `documents` supplies the
items while `texts` still selects the TEI response shape, matching upstream.

For v1 text pairs, usage counts non-padding tokenizer inputs. For v2, the workflow
must emit `outputs.prompt_tokens`, an int64 count covering its complete input,
including any media-token expansion it performs. The server sums these actual
per-pair counts; image byte lengths are not token estimates.

## Package contract

A text-only v1 profile declares `kind: reranking`, `version: '1'` and
`outputs.score`. Its tokenizer encodes `(query, document)` with its authored pair
post-processor, special tokens, padding and truncation. One or more request
`prompt_tokens` inputs receive the paired IDs. Application sources
`attention_mask` and `token_type_ids` identify tokenizer feature rows; physical
workflow input names remain arbitrary. These inputs are int64 `[batch, sequence]`
tensors accepting one batch row. Fixed sequence lengths require matching
padding/truncation. Supported score dtypes are float16, bfloat16 and float32.
Required unsupported profile versions, ambiguous profiles and invalid bindings
fail package admission; unsupported ignorable profiles advertise no capability.
The [text fixture](../tests/fixtures/onnx_genai_workflows/reranking) remains an
executable example.

A multimodal package uses core schema v1.12 and reranking profile v2:

```yaml
profiles:
  relevance:
    kind: reranking
    version: '2'
    outputs:
      score: probability
      prompt_tokens: usage
    reranking:
      query_images: [request.query_picture]
      document_images: [request.document_primary, request.document_secondary]
```

Each image endpoint names an opaque, application-sourced uint8 vector of encoded
bytes. Each position has a distinct endpoint. Names and dimensions are package
data. Required/default/presence semantics come from those workflow inputs, and
all decode, resize, encoding, feature fusion and calibration belong to the
workflow. Tokenizer pair features retain their declared attention-mask and token
segment bindings.

`onnx-genai.image-preprocess@2` uses component-local output names in
`preprocessing.image.outputs[].name`. Invocations bind those ports to their own
SSA values, so one declared preprocessing component can run for query and
multiple document images. Every declared output port must be bound on every
invocation. ABI v1 retains its workflow-scoped output names and single-invocation
contract.

The executable native fixture is
[`reranking_multimodal`](../tests/fixtures/onnx_genai_workflows/reranking_multimodal).
It decodes real PNGs, resizes/rescales them, combines text and three image inputs,
and makes the primary/secondary image ordering observable in the final score.
It is a correctness fixture, not an accuracy or performance benchmark.

## Resource bounds and discovery

- `--max-rerank-documents` defaults to 128.
- `--max-rerank-input-tokens` defaults to 8192 tokenizer tokens per pair and is capped by the model
  context limit.
- `--max-rerank-media-bytes` defaults to 64 MiB of unique retained encoded images
  per request. Reused source strings share their allocation across pairs.
- An individual encoded image is bounded at 20 MiB; JSON request bodies use the
  existing 25 MiB media-upload limit. Remote responses are streamed under both
  the image and remaining request bounds.

All CLI limits must be positive. `/props` advertises `nxrt_capabilities.reranking`
and the model's ordered image capacities under `reranking_image_limits`. Chat
vision capability requires a separate runtime-owned media input; the existence
of an internal image preprocessing program does not grant a chat image endpoint.
