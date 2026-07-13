# Examples

## `onnx-genai-cuda-decode.perfetto.json` — decode timeline trace

A real per-span timeline captured from an onnx-genai CUDA decode loop
(Qwen2.5-0.5B-Instruct, CUDA EP, CUDA graph + device KV enabled) on an
NVIDIA H200. It is a [Chrome Trace Event Format][ctf] document, which
[Perfetto][perfetto] and `chrome://tracing` read directly.

### View it

1. Open <https://ui.perfetto.dev>.
2. Drag `onnx-genai-cuda-decode.perfetto.json` onto the window
   (or **Open trace file**).
3. Zoom into the per-token band. Each generated token is one
   `loop.next_logits` span that nests its sub-stages.

### What you are looking at

Spans are grouped by category (the name prefix before the first `.`):

| Category | Span | Meaning |
| --- | --- | --- |
| `loop` | `loop.next_logits` | Whole per-token forward + logits fetch |
| `ort` | `ort.bind_inputs` | Bind CPU/device inputs for the step |
| `ort` | `ort.session_run` | ONNX Runtime `Run` (the model forward) |
| `ort` | `ort.extract_outputs` | Pull logits handle back from ORT |
| `engine` | `engine.logits_to_vec` | Copy logits to host for sampling |
| `loop` | `loop.sampling` | Greedy argmax over the vocabulary |
| `loop` | `loop.commit_selected` / `loop.commit_token` | Append token to the KV/sequence |
| `loop` | `loop.detokenize` | Turn the token id into text |

The story the trace tells: **`ort.session_run` dominates each token**, with
`engine.logits_to_vec` (the device→host logits copy) the next largest slice.
Sampling, binding, and detokenization are comparatively tiny. This is why the
decode-side optimization work targets the model forward and the logits copy.

### Regenerate

```bash
source .cudaenv.sh   # sets cargo on PATH and $FDL_MODEL
cargo build --release -p onnx-genai-bench \
  --features bench-ort,onnx-genai-ort/cuda --bin profile_decode

ONNX_GENAI_TRACE=examples/onnx-genai-cuda-decode.perfetto.json \
ONNX_GENAI_PROFILE=1 \
ONNX_GENAI_EP=cuda ONNX_GENAI_CUDA_GRAPH=1 ONNX_GENAI_DEVICE_KV=1 \
./target/release/profile_decode --model "$FDL_MODEL" \
  --tokens 24 --warmups 1 --runs 1
```

`ONNX_GENAI_TRACE=<path>` turns on timeline capture in the profiler; it is
zero-cost when the variable is unset. `ONNX_GENAI_PROFILE=1` additionally
prints the aggregate per-stage table. Warmup-run events are discarded, so the
trace contains only the measured run.

[ctf]: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
[perfetto]: https://ui.perfetto.dev
