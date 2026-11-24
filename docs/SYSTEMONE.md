# Non-generative SystemOne decisions

`POST /v1/systemone` evaluates a model-declared decision head. It does not ask a
chat model to generate an answer. A package must declare a `decision` v1 task
profile; ordinary generation/embedding/reranking packages receive actionable
`501 not_supported_error`. `/props` advertises `nxrt_capabilities.decision`.

```json
{
  "model": "decision-model",
  "state": "The text or structured context to assess",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Choose the appropriate destination",
      "criteria": {"billing": "Account and invoice questions", "support": "Technical help"}
    },
    "quality": {
      "type": "score",
      "instructions": "Rate the quality",
      "criteria": ["poor", "adequate", "excellent"]
    },
    "acceptable": {"type": "noul", "instructions": "Is this acceptable?"}
  }
}
```

The response contains `model`, `answers` keyed by question ID, and
`usage: {input_tokens, output_tokens: 0}`. A choice answer has `choice`,
`probabilities` and `confidence`. A score has `score`, `probabilities`,
`confidence` and `legend`. A `noul` answer has its probability under `noul`.
Choice keys are lexically ordered; score levels are zero-based in array order;
`noul` options are `false`, then `true`. Ties select the first option. All prompt
variants count toward input usage; tokenizer padding does not.

## Package contract

The new core declaration requires `schema_version: v1.10`. Both parsed documents
and programmatically constructed metadata enforce this floor. Older schema
versions retain their previous meanings.

```yaml
profiles:
  decisions:
    kind: decision
    version: '1'
    outputs:
      probabilities: calibrated.probabilities
    decision:
      prompt_template: |
        {{ state | tojson }}
        {{ instructions | tojson }}
        {% for option in options %}{{ option.key }}: {{ option.description | tojson }}
        {% endfor %}
      max_options: 16
      variants: direct
      choice_confidence: uniform_margin
      score_confidence: mode_distance
      score: expected_level
```

The model author supplies the prompt; there is no model-name or architecture
lookup. Its Jinja context contains `state`, `instructions`, `type`, `options`
(an ordered array of key/description objects), and zero-based `variant`.
The tokenizer and its postprocessor produce the actual input IDs and feature
rows. The profile uses a request-sourced `prompt_tokens` runtime workflow input.
Optional application inputs are `attention_mask`, `token_type_ids` (int64
`[1,sequence]` rows), and `option_count` (int64 scalar or `[1]`). Unknown required
inputs fail admission instead of receiving guessed values.

The workflow emits a host-readable float16/bfloat16/float32 probability tensor
with shape `[options]` or `[1,options]`. The runtime checks cardinality, range,
finiteness and total probability mass. Small dtype-rounding discrepancies are
normalized. Raw logits or independent sigmoid scores must be converted into a
distribution by the authored workflow; invalid mass/range, malformed shapes and
missing outputs fail. Calibration, head selection, option selection, temperature and
bias belong to the authored workflow. Fixed head/sequence dimensions constrain
requests explicitly; dynamic shapes support varying option/token counts.

`reverse_average` evaluates both direct and reversed option orders, realigns the
second result and averages probabilities. `direct` evaluates one variant.
Confidence policies are explicit: `uniform_margin`, `mode_distance`,
`normalized_entropy`, or `maximum_probability`. Score policies are
`expected_level` or `median_level`. No policy is inferred from model identity.
The decision workflow must not declare text/token outputs; its generation token
budget is zero. Model execution uses the ordinary bounded engine worker queue,
and dropped clients do not execute pending commands. Converted probabilities
retain canonical host-memory ownership through the returned engine result.

## Limits and remaining work

`--max-decision-input-tokens` defaults to 8192 per prompt variant (also capped by
model context); `--max-decision-questions` defaults to 64 and
`--max-decision-options` to 128 (also capped by the package/head). They must be
positive. `--max-decision-prompt-bytes` defaults to 4 MiB before tokenization,
and `--max-decision-template-operations` to 1,000,000 Jinja operations per variant.
Rendered-byte and interpreter-fuel limits fail before model execution, including
large template loops. All questions/variants are prepared before any inference is submitted.
The package's exact tokenizer padding/truncation contract remains authoritative.

This v1 execution path supports text and structured JSON state/instructions.
Files/images/video, chat media parts and spoken instructions are refused before
execution. Multimodal decision preparation, model-specific input/default/score-range policies,
canonical legend policies, joint/packed decision heads and shared generative /
decision workflow entry points still need explicit metadata and runtime support.
GGUF decision-head import is not implemented by registering this HTTP route.
The small native ONNX fixture proves actual input-dependent head execution,
not compatibility with every upstream decision model. These remain part of the
overall compatibility work; tool execution remains delegated to external harnesses.
