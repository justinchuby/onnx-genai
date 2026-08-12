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

## Interaction with weight offload and VMM remap

Two measured facts constrain how capture composes with the memory system; the
full analysis lives in
[`MEMORY_ARCHITECTURE.md`](./MEMORY_ARCHITECTURE.md).

- **Weight offload and CUDA graph capture are mutually exclusive today.** The
  weight pager's alloc/copy/free operations are capture-illegal, so enabling
  offload disables capture (module docs in
  `crates/onnx-genai-engine/src/native_decode/cuda.rs`). A model large enough to
  need offload therefore gets **none** of the capture-fragmentation wins that
  #708 and #728 landed (they took 35B-A3B decode from **154 to 34** graph
  segments). Issue **#716** is the fix: page weights under a stable virtual
  address so page-in remaps physical granules at the same VA instead of returning
  a new pointer that would invalidate the capture.
- **A captured graph survives a VMM remap at the same virtual address (#727).**
  A graph instantiated before `cuMemUnmap`/`cuMemCreate`/`cuMemMap` at the same
  VA replays correctly afterwards and writes into the **new** physical pages —
  sentinel-proven in
  `crates/onnx-runtime-cuda-memory/tests/vmm_graph_remap_gpu.rs`; one physical
  handle mapped at two VAs is readable through either. **Not proven, treated as
  unsafe:** unmapping while a replay is in flight; and `cuMemMap` *during* capture
  returns `CUDA_SUCCESS` but is not proven replayable, so growth is issued outside
  the captured segment. This survivability is the premise under #716.
