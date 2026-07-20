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
