# Cohaagen: fused QMoE SwiGLU decode kernel

Date: 2026-08-11
Branch: `squad/moe-profile-nextlever`

## Finding

The 35B-A3B decode artifact uses fused SwiGLU FC1 (`fc3` absent): FC1 emits `2 * intermediate` gate/up rows, then `qmoe_activate` reduces that scratch to `activated`, and FC2/down remains `qmoe_linear_f32`.

A conservative small-route decode fusion (`rows == 1`, `routes <= 16`) now computes the two FC1 reductions and SwiGLU activation in one `qmoe_gate_up_activate_*` kernel. The fused path is deterministic, argmax-stable, and within GPU/CPU parity tolerance. It removes the separate `qmoe_activate` launch and the `fc1_output` global scratch round-trip for this regime. For Qwen3.6-35B-A3B (`routes=8`, `inter=512`) that is one launch plus ~64 KiB/layer/token of FC1 scratch write/read traffic eliminated; `activated` remains global because FC2/down is still a separate GEMV.

## Measurement

H200 GPU 1, shared host, `profile_native --pipeline --steady --warmups 1 --runs 5 --tokens 128`:

- Baseline from #764 iteration: 11.511 ms/token median (prior same-code spread 11.38–11.56).
- Fused SwiGLU decode: 11.126 ms/token median (runs: 11.126, 11.122, 11.146, 11.174, 11.124).
- Improvement: ~3.3% vs 11.511 and below the prior 11.38 noise floor.

Nsight Systems kernel mix after fusion (`--cuda-graph-trace=node`, 64 tokens):

- `qmoe_linear_f32` (down): 13.3%, 27.76 us median, 5120 instances.
- `qmoe_gate_up_activate_f16`: 9.8%, 20.45 us median, 5120 instances.
- `qmoe_activate` no longer appears on the QMoE decode path.

Nsight Compute on `qmoe_gate_up_activate_f16`:

- No eligible: ~35%.
- Long scoreboard: ~22.5% of active warp cycles.
- Not selected: ~18.0%.
- Wait: ~16.1%.
- Short scoreboard: ~10.2%.
- Barrier: ~8.2%.
- Active warps: ~40.6% of peak sustained active.

## Correctness

QMoE GPU suite passed. The 35B fresh-engine teacher-forced oracle lock passed with argmax 33803 and margin `logprob(33803)-logprob(5342)=0.09375`, inside the 0.04..0.14 band.

## Decision

Ship the conservative FC1/SwiGLU decode fusion. It clears the >3% model-level gate while keeping FC2/down separate and avoiding VMM/offload paths. The next structural lever remains a larger down/FC2 fusion or persistent decode kernel, but that requires a more invasive parallelization design because a single CTA-per-route shared-memory intermediate would underparallelize hidden-output FC2.
