# Native CUDA decode graph capture

Native decode enables CUDA graph capture only when `ONNX_GENAI_CUDA_GRAPH=1`.
`NativeDecodeCudaOptions::graph_capture` can explicitly override the environment
for an individual session. The default remains eager execution.

Only the steady one-token shape is eligible: fixed `[1,1]` input/position
buffers, a fixed-capacity attention mask, fixed-address shared KV, and a
persistent `[1,1,vocab]` logits output. The first one-token step warms kernels
and buffers; the next eligible step records and immediately launches the graph;
later steps update the scalar device inputs and mask delta before replay. Logits
are copied to the host only after capture or replay.

Set `ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1` to print each eager seam's structural
path kind (`host-seam` or `eager-device-seam`) alongside its existing detailed
capture-decline reason.

Every compiled kernel is passed through `subgraph_graph_capturable` before stream
capture. Any kernel that can allocate/free, compile lazily, perform D2H
validation, or synchronize the stream rejects the whole step and native decode
continues eagerly without changing tokens. The current Qwen int4 decode graph
still falls back because kernels including `MatMulNBits`, GQA, Gather, and
broadcast elementwise operations are deliberately marked non-capturable.

The installed executable is owned by the session CUDA runtime and is destroyed
before its referenced buffers. Reset, rewind, multi-token/prefill shape changes,
binding address/shape changes, and session drop invalidate it. A later
generation warms and captures a fresh executable; a live executable is never
reused across generations or incompatible bindings.

## Multi-component / routed decoders (step-inputs capture, default-on)

A multi-component pipeline decoder (e.g. gemma-3n / gemma4-e2b, and the
GQA layers of the 35B-A3B class) consumes per-step `inputs_embeds` and/or routed
ports supplied by upstream components (the embedding model) each decode step.
The captured single-token fast path (`run_one_token`) writes those one-token
tensors into **persistent** device bindings instead of rebuilding owned uploads,
so the routed decode step reuses the captured graph exactly like a plain
token-id decode — recovering the graph-capture perf that the eager owned-input
path forfeits.

This path is **on by default** for capture-eligible decoders. The environment
variable `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` is an **opt-out escape
hatch**:

- unset (or any truthy/unrecognized value) — capture-on (the default);
- `0` / `false` / `no` / `off` — force the eager owned-input path.

The structural eligibility gates still apply: a decoder whose bindings expose a
growing logical prefix, or whose attention mask is exposed to a non-capacity-aware
consumer (e.g. a GLM-style indexer), clears `graph_enabled` and **auto-declines
to the eager path** regardless of this flag, so default-on never captures an
ineligible decoder and never changes tokens. Recurrent Scan / LinearAttention
hybrids decline through the same structural gate until their capture lane lands.
The process-global counter `NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES`
increments once per captured routed decode step, providing a non-vacuous
capture-hit signal for tests.
