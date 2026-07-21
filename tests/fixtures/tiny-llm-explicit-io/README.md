# tiny-llm-explicit-io

Regression fixture for the metadata-driven graph-port binding (Phase 1 of the
"no assumed tensor names" rule).

It is the `tiny-llm` GPT-2 decoder graph with every decode-step port renamed to
a deliberately non-conventional name:

| role            | conventional name       | renamed to        |
| --------------- | ----------------------- | ----------------- |
| token input     | `input_ids`             | `tokens`          |
| attention mask  | `attention_mask`        | `attn_mask_port`  |
| position ids    | `position_ids`          | `pos_port`        |
| logits output   | `logits`                | `out_logits`      |
| past KV inputs  | `past_key_values.0.*`   | `cache_k_in.0`, `cache_v_in.0`   |
| present KV out  | `present.0.*`           | `cache_k_out.0`, `cache_v_out.0` |

Because none of these match the historical name conventions, the runtime can
only decode this graph by reading the explicit `model.io` block in
`inference_metadata.yaml`. The graph was produced from `tiny-llm` by renaming
the tensor names consistently across the textproto; no Mobius regeneration is
required.
