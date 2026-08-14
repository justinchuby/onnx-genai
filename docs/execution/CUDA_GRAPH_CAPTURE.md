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

Several measured facts constrain how capture composes with the memory system;
the full analysis and the audit table live in
[`MEMORY_ARCHITECTURE.md`](../memory/MEMORY_ARCHITECTURE.md), which is the single source
for KV geometry and residency. This section only summarises the results that
touch capture.

- **Offload and capture are no longer mutually exclusive (#796).** They were:
  the weight pager's alloc/copy/free operations are capture-illegal, so enabling
  offload used to disable capture, and the models that most needed offload lost
  the 154 → 34 graph-segment collapse landed by #708/#728/#757. #796 removed the
  exclusion by paging weights under a **stable virtual address** — a page-in
  remaps physical granules at the same VA instead of returning a new pointer that
  would invalidate the capture. Capture-under-offload is now gated on
  `weight_offload_enabled && !weight_offload_stable_va`, i.e. it is allowed on the
  stable-VA path and pinned by the #796 unit tests
  (`weight_offload_on_stable_va_path_keeps_graph_capture`).
- **Managed no-spill VMM is the default (#798).** On native CUDA the
  authority-governed VMM path is selected by default with automatic weight
  streaming when a model exceeds the resolved budget. A model that fits does
  **not** page: the default-flag run measures `FullResident`, offload off, **0
  page-ins**. Growth on this default path is in-place VMM at a stable base VA, so
  a captured decode graph survives KV bucket growth (below).
- **A captured graph survives a VMM remap at the same virtual address (#727).**
  A graph instantiated before `cuMemUnmap`/`cuMemCreate`/`cuMemMap` at the same
  VA replays correctly afterwards and writes into the **new** physical pages —
  sentinel-proven in
  `crates/onnx-runtime-cuda-memory/tests/vmm_graph_remap_gpu.rs`; one physical
  handle mapped at two VAs is readable through either. **Not proven, treated as
  unsafe:** unmapping while a replay is in flight; and `cuMemMap` *during*
  capture returns `CUDA_SUCCESS` but is not proven replayable, so growth is
  issued outside the captured segment. This survivability is the premise under
  #796.
- **Growth invalidation is now conditional (#811).** The engine no longer
  invalidates the captured graph on every KV bucket growth. On the seq-major
  fixed full-context-stride path, growth commits token stripes on demand while
  device pointers and physical shapes stay unchanged, batch-0 addressing is
  capacity-independent, and the mask is fully committed — so the graph is
  **kept**, with a named keep reason listing those four checked dependencies, and
  growth-attributable invalidations fall **3 → 0**. Head-major still invalidates,
  correctly, because it re-strides every head stripe and moves 688,576 bytes per
  growth. A defense-in-depth signature check (device pointer + physical shape per
  binding, compared before/after the commit) forces invalidation if anything
  unexpectedly moved; a negative-oracle test fails if a graph is ever kept across
  a growth that genuinely moved bytes.

### A `captures=0` reading that was a process artifact, not a model gate (#804/#807)

The process-wide `RuntimeConfig` is a snapshot frozen on first read. A
capture-OFF phase that set `ONNX_GENAI_CUDA_GRAPH=0` *after* that snapshot was
already frozen had no effect, and a long-lived process then reported `captures=0`
— which #801 (and a merged PR body) misattributed to a model capture gate. #804
found the real cause was the cached, frozen env value; **two PRs drew wrong
conclusions from that number**. The lesson: anything that reads process-frozen
config must run in its own process, and capture counters must be read per
process. #807 added a debug-only freeze guard that panics if a variable feeding
the frozen snapshot is mutated after freeze.
