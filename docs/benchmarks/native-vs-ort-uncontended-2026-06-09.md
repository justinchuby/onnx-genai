# Native CUDA vs. ORT CUDA uncontended ladder — 2026-06-09

## Method

- Source: `origin/main` at `b5b2084841be2f31c3d9cbb8475e0e30c950743e`.
- GPU: physical NVIDIA H200 5, pinned with `CUDA_VISIBLE_DEVICES=5`.
  It had 0 MiB allocated and 0% utilization before every model. Between-run
  snapshots showed 0 MiB allocated and 0–2% residual utilization. No competing
  process appeared on GPU 5. A compute-idle 129.6-GB allocation remained on GPU
  1 throughout, but did not use the selected device.
- Harness: release `profile_native`, built with
  `mlas,bench-ort,bench-native,cuda`.
- Decode: greedy prompt `The capital of France is`, EOS stopping disabled,
  one warmup, three measured runs, `--steady --decode-skip 8`.
- Values are medians of the three measured steady-decode throughputs. Ratio is
  native / ORT.
- Both 128- and 1024-token generations were inspected. Every native backend
  produced readable, non-garbled output. Some models became repetitive at the
  long horizon, but remained linguistically coherent.
- The Qwen3-0.6B rows were remeasured on 2026-07-24 from `ea452be0`, using
  `/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda-postfix` (the
  metadata-fixed `-postfix` artifact), rather than the generic Qwen3 artifact.
  Unlike the original ladder, this was a shared, CPU-loaded host: CPU was pinned
  to core 1 and GPU 3 was allocation-free before/after, but host load averaged
  6.93–16.77 during the recorded runs. These are on-same-host before/after
  results, not an uncontended absolute claim. Three one-run process samples per
  backend were used; table values are their medians.

## Results

| model | native tok/s | ORT tok/s | native / ORT | @tokens | contended? |
|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B int4 | 908.19 | 594.45 | **1.528×** | 128 | No |
| Qwen2.5-0.5B int4 | 795.06 | 496.82 | **1.600×** | 1024 | No |
| Qwen2.5-1.5B int4 | 628.95 | 443.64 | **1.418×** | 128 | No |
| Qwen2.5-1.5B int4 | 556.13 | 370.08 | **1.503×** | 1024 | No |
| Qwen2.5-7B int4 | 301.62 | 273.46 | **1.103×** | 128 | No |
| Qwen2.5-7B int4 | 274.98 | 240.41 | **1.144×** | 1024 | No |
| Phi-4-mini int4/int8 | 320.78 | 230.70 | **1.390×** | 128 | No |
| Phi-4-mini int4/int8 | 289.47 | 203.58 | **1.422×** | 1024 | No |
| Qwen3-0.6B int4 (`qwen3-0.6b-int4-cuda-postfix`)† | 530.68 | 443.54 | **1.196×** | 128 | Shared host; CPU load |
| Qwen3-0.6B int4 (`qwen3-0.6b-int4-cuda-postfix`)† | 479.42 | 384.05 | **1.248×** | 1024 | Shared host; CPU load |
| DeepSeek-Coder-1.3B int4 | 796.26 | 635.59 | **1.253×** | 128 | No |
| DeepSeek-Coder-1.3B int4 | 627.39 | 512.59 | **1.224×** | 1024 | No |
| DeepSeek-R1-Distill-Qwen-1.5B int4 | 634.26 | 439.97 | **1.442×** | 128 | No |
| DeepSeek-R1-Distill-Qwen-1.5B int4 | 556.82 | 370.74 | **1.502×** | 1024 | No |
| GLM-4-9B int4 | 120.88 | N/A | N/A | 128 | No |
| GLM-4-9B int4 | 114.27 | N/A | N/A | 1024 | No |

**Verdict:** native CUDA beat ORT CUDA in every comparable original
uncontended row, by 1.103–1.600×; GLM-4-9B ran coherently natively but ORT
could not load it. The post-fix Qwen3 ratio is 1.196× at 128 tokens and 1.248×
at 1024 tokens on a shared host, so it is not included in that uncontended
range.

† Qwen3 remeasurement source: `ea452be0740e0a9894fb6df29245e2bbe453d15e`,
after the multi-group Q/K RMSNorm capture fix (decode seams 29→1; Chew's review
measured about +18% native). This shared-host result is +16.6% versus the prior
454.97 tok/s ladder value. H200 GPU 3, `CUDA_VISIBLE_DEVICES=3`, `taskset -c 1`,
`--ep cuda`, prompt `The capital of France is`, `--steady --decode-skip 8`, one
warmup, and three separate one-measurement runs per backend. Raw tok/s: 128 native
530.51/533.44/530.68, ORT 451.84/443.54/438.01; 1024 native
479.42/480.29/478.96, ORT 388.41/384.05/380.17. Native and ORT both produced
the same coherent `Paris` / `Rome` continuation in the 128-token check.

## Coherence and failures

- Qwen2.5 outputs began with `Paris` and continued in readable English.
- Phi-4-mini answered `Paris` before continuing into its packaged prompt
  material.
- Qwen3-0.6B (`qwen3-0.6b-int4-cuda-postfix`) continued with `Paris` and
  `Rome`.
- DeepSeek-Coder emitted a readable list of national capitals.
- DeepSeek-R1 was repetitive and its packaged template emitted `C iter`, but
  the continuation remained readable rather than numerically corrupted.
- GLM-4-9B answered in mixed English and Chinese (`法国的首都是巴黎`) before
  becoming repetitive. It was not token garbage.
- ORT rejected GLM-4-9B's partial-RoPE GQA graph with
  `Unrecognized attribute: rotary_embedding_dim for operator
  GroupQueryAttention`; this is a schema/load limitation, not an OOM.

The first Qwen2.5-1.5B native 128-token process launch produced an isolated
374.65 tok/s median despite GPU 5 remaining allocation-free. Two immediate
independent relaunches produced 628.95 and 630.74 tok/s with tight within-launch
ranges. The table uses the first reproducible relaunch (628.95 tok/s); the
second confirms it. No other row needed a rerun.

## Commands

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
export CUDA_VISIBLE_DEVICES=5

cargo build --release -p onnx-genai-bench \
  --features mlas,bench-ort,bench-native,cuda --bin profile_native

target/release/profile_native \
  --model <model-dir> --backend <native|ort> --ep cuda \
  --prompt "The capital of France is" \
  --tokens <128|1024> --steady --decode-skip 8 --warmups 1 --runs 3
```

Foundry package roots were resolved to `v4` for Qwen2.5 and `v5` for
Phi-4-mini. The Mobius/GLM artifact directories contained `model.onnx`
directly.
