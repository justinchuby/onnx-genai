# Profile samples

Real `--profile` reports captured from Qwen2.5-0.5B-Instruct, one per execution
provider, so the shape of the output can be read without running anything — and
so the numbers can be compared against a later change.

| File | Backend | Provider |
| --- | --- | --- |
| [`qwen2.5-0.5b-cpu.txt`](qwen2.5-0.5b-cpu.txt) | ONNX Runtime | CPU |
| [`qwen2.5-0.5b-metal.txt`](qwen2.5-0.5b-metal.txt) | ONNX Runtime | MLX/Metal plugin |
| [`qwen2.5-0.5b-native.txt`](qwen2.5-0.5b-native.txt) | native | CPU |
| `*.json` | | the same runs via `--profile-json`, for diffing or plotting |

Captured on an Apple M1 Max (32 GiB, macOS 26.5.2) with a release build. The
machine was not idle — load average ≈ 4 — so treat the absolute milliseconds as
indicative and the *ratios* as the point.

## Regenerating

```bash
cargo build --release -p onnx-genai-cli

ONNX_GENAI_EP=cpu ./target/release/onnx-genai --profile \
  generate models/qwen2.5-0.5b \
  --prompt "Write a short Rust function that reverses a string." \
  --max-new-tokens 48 --temperature 0
```

For the Metal run, point at the MLX plugin the Python packages ship. It is then
auto-selected on macOS:

```bash
ONNX_GENAI_METAL_EP_LIB=$(python -c 'import onnxruntime_mlx, os;
print(os.path.join(os.path.dirname(onnxruntime_mlx.__file__), "libonnxruntime_mlx_ep.dylib"))') \
./target/release/onnx-genai --profile generate models/qwen2.5-0.5b ...
```

For the native backend, set `ONNX_GENAI_BACKEND=native`.

Add `--profile-trace out.json` for a Perfetto timeline instead of these
aggregates; see [`../traces/`](../traces/).

## What the samples show

```
                        CPU        Metal       native
model load           1497 ms       600 ms       131 ms
time to first token   117 ms       690 ms      2414 ms
decode throughput   45.9 tok/s   67.4 tok/s    3.4 tok/s
end-to-end          41.3 tok/s   34.2 tok/s    3.0 tok/s
```

**No single configuration wins every row, which is the point of keeping all
three.** Each is fastest at something and slowest at something else:

* **Metal has the fastest steady-state decode** (1.47x CPU) but the slowest
  start: 690 ms to first token against CPU's 117 ms. It only pulls ahead on
  end-to-end once a generation runs past roughly 90 tokens; these samples stop
  at 48, so CPU still wins the end-to-end column.
* **The native backend loads in 131 ms**, an order of magnitude faster than
  ORT's 1497 ms, because it maps weights instead of building a session graph. It
  then decodes ~13x slower — it is a from-scratch executor without ORT's kernel
  library, and is included as an honest baseline, not a recommendation.

Earlier revisions of this sample recorded Metal as *slower* than CPU. That was
real, and the cause was visible in the trace: the MLX plugin declined every
`Attention` node (it refused a causal mask combined with an explicit mask), so
attention ran on CPU and the graph was cut into 28 subgraphs. Fixing the plugin
to claim those nodes is what moved the number from 36 to 67 tok/s.

## Why a stage's `us/call` can look nothing like the latency line

`loop.step` on the Metal run reports 28894 us/call, which would be 34 tok/s —
yet `inter-token latency` says 14.8 ms and the machine really does decode at 67
tok/s. Both are correct: **the first `loop.step` carries the prefill.** Back the
one-off cost out and the rest is uniform:

```
Metal   (1386.95 ms total - 690.3 ms prefill) / 47 remaining steps = 14.8 ms  -> 67.4 tok/s
CPU     (1139.84 ms total - 116.7 ms prefill) / 47 remaining steps = 21.8 ms  -> 45.9 tok/s
```

The mean is averaging a bimodal distribution. Read `inter-token latency` and its
percentiles for steady-state decode, and `us/call` only for stages that are not
called once per token.

Note also that `ort.session_run` accounts for essentially all of `loop.step` in
every run -- 1155.8 ms against 1139.8 ms on CPU, the difference being the prefill
call the decode loop does not own. Everything this crate does itself, summed
across 48 tokens, is under 2 ms: sampling 0.90 ms, KV bookkeeping 0.36 ms,
detokenizing 0.40 ms. Any real speed-up has to come from the model forward.

## Reading the rest of the report

* **Decode throughput excludes the prefill wait; end-to-end includes it.** A long
  prompt inflates the second without the model decoding any faster.
* **Percentiles sit next to the mean** because a run whose typical token takes
  268 ms but which stalls for 1.26 s mid-sentence feels broken, and only the
  tail shows it — see the native run's `p99 1259.8` against its `p50 267.5`, a
  4.7x spread that the mean (294.3) almost entirely hides.
* **`kv page activity`** is a delta for this run, not lifetime totals. Evictions
  and allocation failures appear only when they happen.
* **`device memory breakdown`** splits the ceiling into weights and KV. The line
  about activations says they are *not yet measured* rather than folding them
  into another number or reporting them as zero.
* **`execution provider`** names what actually resolved, not what was requested:
  on macOS the Metal plugin is auto-selected without appearing in any
  environment variable the caller set.
