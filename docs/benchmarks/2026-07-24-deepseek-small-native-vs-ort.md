# Small DeepSeek native CUDA vs. ORT CUDA — 2026-07-24

This controlled steady-decode comparison covers real CUDA-int4
DeepSeek-Coder-1.3B and DeepSeek-R1-Distill-Qwen-1.5B artifacts on an
8× NVIDIA H200 shared box (GPU 1). The box was contended: a resident
129.6-GB VLLM process was compute-idle, and other teams were running CPU/GPU
work. The values below are therefore three-run medians under that contention,
with the observed run ranges retained for context.

Native and ORT used the same decode window: fixed greedy prompt, one warmup,
32 tokens, the first eight generated tokens skipped, and three steady runs.
Native was run with `profile_native --ep cuda --steady --tokens 32 --warmups 1
--runs 3 --decode-skip 8` and `CUDA_VISIBLE_DEVICES=1`.

| Model | Native tok/s (median of 3) | ORT tok/s (median of 3) | Native/ORT | Native run range | ORT run range |
|---|---:|---:|---:|---:|---:|
| DeepSeek-Coder-1.3B int4 | 824.08 | 618.02 | **1.333×** | 823.26–824.47 | 616.23–619.28 |
| DeepSeek-R1-Distill-Qwen-1.5B int4 | 647.73 | 430.51 | **1.505×** | 647.55–648.71 | 428.74–434.03 |

Both native runs reported healthy capture: `enabled=true`, two captures, 58
replays, and zero fallbacks before measurement; the measured generation added
one capture, 29 replays, and zero fallbacks.

## Greedy output and parity

DeepSeek-Coder-1.3B produced coherent greedy text, and native and ORT emitted
the exact same 32 output IDs.

For DeepSeek-R1-Distill-Qwen-1.5B, the supplied chat template produced a
coherent native continuation. The fixed parity prompt first diverged at
generated index 7 (the eighth token): native selected token `374`, while ORT
selected `315`. The native-more-accurate result was adjudicated with the fp32
oracle and locked by the ignored real-model regression at `30ab9c7f`; this
throughput comparison does not treat the R1 native/ORT outputs as exact parity.
