# Decode timeline traces

Real per-span timelines captured from the onnx-genai decode loop with the
env-gated profiler. Each file is a [Chrome Trace Event Format][ctf] document
that [Perfetto][perfetto] and `chrome://tracing` read directly.

Set `ONNX_GENAI_TRACE=<path>` to turn on timeline capture (zero-cost when
unset); add `ONNX_GENAI_PROFILE=1` to also print the aggregate per-stage table.
`ONNX_GENAI_PROFILE` on its own prints a **text table**, not JSON — the JSON
document here is produced by `ONNX_GENAI_TRACE`. Warmup-run events are discarded,
so a trace contains only the measured run.

## `onnx-genai-cuda-device-argmax.perfetto.json`

Qwen2.5-0.5B-Instruct on the CUDA EP with CUDA graph capture and the
**on-device greedy argmax** path (`ONNX_GENAI_DEVICE_ARGMAX=1`, the default when
the `cuda` feature is built). The logits stay on the GPU and are reduced by a
custom kernel, so only the winning token id returns to the host.

### View it

1. Open <https://ui.perfetto.dev>.
2. Drag the `.perfetto.json` file onto the window (or **Open trace file**).
3. Zoom into the per-token band. Each generated token is one
   `loop.next_logits` span nesting its sub-stages.

### What you are looking at

Spans are grouped by category (the name prefix before the first `.`):

| Category | Span | Meaning |
| --- | --- | --- |
| `loop` | `loop.next_logits` | Whole per-token forward + logits fetch |
| `ort` | `ort.bind_inputs` | Bind CPU/device inputs for the step |
| `ort` | `ort.session_run` | ONNX Runtime `Run` (the model forward) |
| `ort` | `ort.extract_outputs` | Pull logits handle back from ORT (+ device argmax) |
| `loop` | `loop.commit_selected` / `loop.commit_token` | Append token to the KV/sequence |
| `loop` | `loop.detokenize` | Turn the token id into text |

The story the trace tells: **`ort.session_run` dominates each token**, and the
per-token host `engine.logits_to_vec` copy is **absent** — with device argmax the
full-vocabulary logits are never pulled to the host, so only `ort.extract_outputs`
(which now launches the argmax kernel and copies back the 4-byte token id)
remains. Compare with `onnx-genai-cuda-decode.perfetto.json` (below), captured on
the host-argmax path, where `engine.logits_to_vec` is a per-token span.

### Regenerate

```powershell
# Windows / PowerShell; NVRTC must be on PATH (nvidia-cuda-nvrtc-cu12 wheel)
$env:ORT_ROOT = "...\onnxruntime-win-x64-gpu_cuda12-1.27.0"
cargo build --release -p onnx-genai-bench `
  --features bench-ort,onnx-genai-ort/cuda --bin profile_decode

$env:ONNX_GENAI_TRACE = "examples\traces\onnx-genai-cuda-device-argmax.perfetto.json"
$env:ONNX_GENAI_PROFILE = "1"
$env:ONNX_GENAI_CUDA_GRAPH = "1"
$env:ONNX_GENAI_DEVICE_ARGMAX = "1"
.\target\release\profile_decode.exe --model "<model dir>" --tokens 24 --warmups 1 --runs 1
```

```bash
# Linux / macOS
ONNX_GENAI_TRACE=examples/traces/onnx-genai-cuda-device-argmax.perfetto.json \
ONNX_GENAI_PROFILE=1 ONNX_GENAI_CUDA_GRAPH=1 ONNX_GENAI_DEVICE_ARGMAX=1 \
./target/release/profile_decode --model "$FDL_MODEL" --tokens 24 --warmups 1 --runs 1
```

The event *timings* reflect the machine and thermal state at capture time (this
one was taken on a throttled RTX 4060 Laptop); it is the span *structure* — which
stages exist per token — that the trace is meant to show.

## `onnx-genai-cuda-decode.perfetto.json`

An earlier capture on the **host-argmax** path: Qwen2.5-0.5B-Instruct, CUDA EP,
CUDA graph + device KV, on an NVIDIA H200. Same viewing steps as above.

Here `engine.logits_to_vec` (the device→host logits copy) *is* a per-token span
and is the second-largest slice after `ort.session_run` — the copy the
device-argmax path above eliminates. Its extra spans:

| Category | Span | Meaning |
| --- | --- | --- |
| `engine` | `engine.logits_to_vec` | Copy logits to host for sampling |
| `loop` | `loop.sampling` | Greedy argmax over the vocabulary (host) |

[ctf]: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
[perfetto]: https://ui.perfetto.dev
