# Profile samples

Real `--profile` reports captured from Qwen2.5-0.5B-Instruct, one per execution
provider, so the shape of the output can be read without running anything — and
so the numbers can be compared against a later change.

| File | Provider |
| --- | --- |
| [`qwen2.5-0.5b-cpu.txt`](qwen2.5-0.5b-cpu.txt) | CPU |
| [`qwen2.5-0.5b-metal.txt`](qwen2.5-0.5b-metal.txt) | MLX/Metal plugin |
| `*.json` | the same runs via `--profile-json`, for diffing or plotting |

Captured on an Apple M1 Max (32 GiB, macOS 26.5.2) with a release build. The
machine was not idle — load average ≈ 4 — so treat the absolute milliseconds as
indicative and the *ratios* as the point.

## Regenerating

```bash
cargo build --release -p onnx-genai-cli

ONNX_GENAI_EP=cpu ./target/release/onnx-genai --profile \
  generate models/qwen2.5-0.5b \
  --prompt "Write a short Rust function that reverses a string." \
  --max-new-tokens 64 --temperature 0
```

For the Metal run, point at the MLX plugin the Python packages ship. It is then
auto-selected on macOS:

```bash
ONNX_GENAI_METAL_EP_LIB=$(python -c 'import onnxruntime_mlx, os;
print(os.path.join(os.path.dirname(onnxruntime_mlx.__file__), "libonnxruntime_mlx_ep.dylib"))') \
./target/release/onnx-genai --profile generate models/qwen2.5-0.5b ...
```

Add `--profile-trace out.json` for a Perfetto timeline instead of these
aggregates; see [`../traces/`](../traces/).

## What the samples show

```
                        CPU         Metal
model load           1489 ms        500 ms
time to first token   112 ms        513 ms
decode throughput   45.3 tok/s    34.1 tok/s
end-to-end          42.0 tok/s    26.8 tok/s
```

**Metal is slower here, and that is the sample's most useful property.** The
report is worth having precisely because it contradicts the assumption that the
GPU path wins. ONNX Runtime says why while loading — on stderr, so it is not in
the captured reports below:

```
Some nodes were not assigned to the preferred execution providers
```

The MLX EP claims only part of the graph, so every decode step crosses back and
forth between the GPU and CPU for the nodes it did not take. At batch 1 a 0.5B
decode is memory-bound anyway, and that traffic costs more than the kernels
save. Metal loads the model much faster, so it wins on start-up and loses on
steady state — which is exactly the trade-off the per-stage table exists to make
visible.

Note also that `ort.session_run` accounts for essentially all of `loop.step` in
both runs — 1517.8 ms against 1503.4 ms on CPU, the difference being the prefill
call the decode loop does not own. Everything this crate does itself, summed
across 64 tokens, is under 3 ms: sampling 1.17 ms, KV bookkeeping 0.42 ms,
detokenizing 0.25 ms. Any real speed-up has to come from the model forward.

## Reading the rest of the report

* **Decode throughput excludes the prefill wait; end-to-end includes it.** A long
  prompt inflates the second without the model decoding any faster.
* **Percentiles sit next to the mean** because a run averaging 21 ms/token but
  stalling for 98 ms mid-sentence feels broken, and only the tail shows it — see
  the Metal run's `p99 64.2` against its `p50 31.2`.
* **`kv page activity`** is a delta for this run, not lifetime totals. Evictions
  and allocation failures appear only when they happen.
* **`device memory breakdown`** splits the ceiling into weights and KV. The line
  about activations says they are *not yet measured* rather than folding them
  into another number or reporting them as zero.
* **`execution provider`** names what actually resolved, not what was requested:
  on macOS the Metal plugin is auto-selected without appearing in any
  environment variable the caller set.
