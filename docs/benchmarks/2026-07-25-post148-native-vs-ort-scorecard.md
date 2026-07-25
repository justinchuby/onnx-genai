# Post-#148 native CUDA vs. ORT CUDA scorecard — 2026-07-25

## Headline

On the three requested Foundry `cuda-gpu` int4 artifacts, native CUDA remains
faster than a confirmed ORT CUDA baseline across the board:

- Phi-4-mini: **321.91 vs. 232.27 tok/s, 1.386× (+38.6%)**.
- Qwen2.5-1.5B: **700.88 vs. 436.61 tok/s, 1.605× (+60.5%)**.
- Qwen2.5-7B: **308.34 vs. 274.67 tok/s, 1.123× (+12.3%)**.

The exact-commit native A/B attributes a **10.5%** Qwen2.5-1.5B gain and a
**2.1%** Qwen2.5-7B gain to #148. Phi-4-mini was neutral (-0.09%).

## Post-#148 scorecard

Values are medians of three measured runs. Parentheses show the run min–max
and full-range span as a percentage of the median. Every process used two
warmups, generated 128 tokens, excluded the first eight emitted tokens from
the steady-decode interval, pinned CPU 1, and ran on physical H200 GPU 5.

| Model | Native CUDA | ORT CUDA | Native / ORT |
|---|---:|---:|---:|
| Phi-4-mini | 321.91 (321.27–322.05; 0.24%) | 232.27 (231.06–233.32; 0.97%) | **1.386×** |
| Qwen2.5-1.5B | 700.88 (700.73–701.03; 0.04%) | 436.61 (436.33–436.62; 0.07%) | **1.605×** |
| Qwen2.5-7B | 308.34 (307.96–308.46; 0.16%) | 274.67 (272.82–275.11; 0.83%) | **1.123×** |

The scorecard binary was built from `6a64ee3c`, current `origin/main`, which
contains #148 (`7a2cd87d`) and the subsequent GLM logical-mask fix (#149).

## ORT-on-GPU confirmation

GPU 5 was sampled every 100 ms throughout each ORT process. All artifacts
loaded `CUDAExecutionProvider`, showed sustained device activity, and emitted
no `Memcpy nodes are added` warning.

| Model | Peak GPU util | Peak GPU memory | Nonzero-util samples | `Memcpy` warning | Verdict |
|---|---:|---:|---:|---|---|
| Phi-4-mini | 87% | 4,775 MiB | 47 | None | **Valid GPU run** |
| Qwen2.5-1.5B | 86% | 2,853 MiB | 26 | None | **Valid GPU run** |
| Qwen2.5-7B | 91% | 6,309 MiB | 41 | None | **Valid GPU run** |

ORT printed its standard notice that shape-related nodes may intentionally be
assigned to CPU. In combination with 86–91% peak GPU utilization and the
absence of inserted-`Memcpy` warnings, this is not the partial-EP fallback
pattern that invalidated earlier `generic-cpu` comparisons.

## Exact #148 cross-model effect

To isolate the kernel change, separate release binaries were built from the
commit immediately before #148 (`04c85242`) and the exact #148 commit
(`7a2cd87d`). Native-only runs were performed back-to-back on the same idle
GPU, with implementation order alternated across models. Token IDs were
identical before and after #148 for all three models. The A/B held the model,
prompt, 128 generated tokens, two warmups, three measured runs, steady-decode
window (first eight tokens excluded), CPU-1 `taskset` pinning, and physical GPU
constant; the only executable change was #148's `down_tpl<COLS>` grid-fill
kernel.

| Model | Pre-#148 native | Exact #148 native | Change | Verdict |
|---|---:|---:|---:|---|
| Phi-4-mini | 322.03 tok/s | 321.75 tok/s | **-0.09%** | Neutral |
| Qwen2.5-1.5B | 635.26 tok/s | 702.11 tok/s | **+10.52%** | Helped |
| Qwen2.5-7B | 301.92 tok/s | 308.17 tok/s | **+2.07%** | Helped |

The result confirms that the SM-count-selected down-GEMV grid-fill is not
7B-specific. It strongly helps the 1.5B Qwen shape, reproduces the expected
approximately 2.1% 7B gain, and is effectively inert on Phi-4-mini.

The ordering is physically plausible rather than an ORT-jitter artifact: this
is a native-only, same-GPU A/B, and the smaller Qwen1.5B down projection has
smaller K/N and therefore fewer baseline 8-column CTAs than 7B. It is more
grid-starved on a many-SM H200, so splitting `down_tpl<COLS>` into more CTAs
can hide more latency and produce a larger relative gain. The 10.52% result is
nevertheless awaiting one additional clean-idle-GPU confirmation; the review
fleet was busy when this scorecard was reviewed, so no contended rerun was
attempted.

For context, versus the earlier scorecard's separate pre-#148 session, native
changed from 322.04 to 321.91 tok/s on Phi, 632.62 to 700.88 on Qwen 1.5B,
and 302.24 to 308.34 on Qwen 7B. Those indicative changes agree with the
controlled A/B, but the exact-commit A/B above is the attribution authority.

## Correctness and caveats

- Phi-4-mini and Qwen2.5-7B produced identical native/ORT token IDs and
  coherent continuations. Exact pre/post-#148 native token IDs also matched.
- Qwen2.5-1.5B retained the known repetitive continuation on both backends.
  Native and ORT differ in an early attribution, as in the prior scorecard;
  its throughput row is valid, but this is not a clean coherence result.
- Native spreads were at most 0.50 tok/s in the scorecard. Qwen 1.5B ORT was
  similarly tight; Phi and Qwen 7B ORT showed 0.8–1.0% within-process jitter.
- Because those two ORT spans exceeded roughly 1 tok/s, they were repeated on
  the same still-idle GPU. The repeat medians were lower (229.85 and
  271.77 tok/s), with Qwen 7B tightening to a 0.32 tok/s span. The table keeps
  the original, faster ORT medians, which is conservative for native/ORT
  ratios and avoids selecting a favorable rerun.
- At selection, GPU 5 had 0 MiB allocated and 0% utilization. It returned to
  0 MiB and 0% after the sweep and after the exact A/B. GPU 1 retained the
  other team's approximately 129 GB allocation and was never selected.
- Other physical GPUs were busy when preparation began. The sweep waited
  until GPU 5 became genuinely idle rather than benchmarking under contention.

## Reproduction

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
cd /home/justinchu/wt-hicks-sweep

CARGO_TARGET_DIR=target/post148 cargo build --release \
  -p onnx-genai-bench --features bench-native,bench-ort,cuda \
  --bin profile_native

BIN=target/post148/release/profile_native
COMMON="--ep cuda --steady --warmups 2 --runs 3 --tokens 128"
PHI=$HOME/.foundry/cache/models/Microsoft/Phi-4-mini-instruct-cuda-gpu-5/v5
Q15=$HOME/.foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-cuda-gpu-4/v4
Q7=$HOME/.foundry/cache/models/Microsoft/qwen2.5-7b-instruct-cuda-gpu-4/v4

for MODEL in "$PHI" "$Q15" "$Q7"; do
  CUDA_VISIBLE_DEVICES=5 taskset -c 1 "$BIN" \
    --model "$MODEL" $COMMON --backend native
  CUDA_VISIBLE_DEVICES=5 taskset -c 1 "$BIN" \
    --model "$MODEL" $COMMON --backend ort
done
```

Each process was independently monitored with:

```bash
nvidia-smi -i 5 \
  --query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits -lms 100
```
