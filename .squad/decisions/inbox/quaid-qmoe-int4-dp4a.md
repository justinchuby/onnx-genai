# Quaid memo: QMoE int4 DP4A / vectorized weight-read decode experiment

Date: 2026-08-11
Branch: `squad/moe-int4-dp4a-experiment` (fresh off `origin/main` @ 4dd5119b, i.e. #765 tip)
GPU: H200, `CUDA_VISIBLE_DEVICES=4`
Authorized by: Justin (@justinchuby) as a risk/reward branch experiment. NO-SHIP is an acceptable outcome; the goal is DATA.

## TL;DR — NO-SHIP (perf regression, oracle test fails on the #722 tripwire)

The highest-ceiling remaining QMoE decode lever named in Cohaagen's latency-floor memo
(*"int4 DP4A / vectorized weight-read"*) was prototyped as a **128-bit (`uint4`) vectorized
int4 weight-stream** for the `rows==1 && routes<=16` decode GEMV (FC1 fused gate/up +
SwiGLU, and FC2/down). It **regressed** the model by **+4.90% ms/token** and grew the QMoE
op by **+25%**. The teacher-forced correctness lock was numerically **unchanged** (token
33803 holds, margin **0.09375** — bit-identical to baseline), but the oracle test still
**FAILS** because the *autoregressive* #722 tripwire drifted to token 13 (outside the narrow
benign set {33803, 46283}). Both ship gates (oracle PASS + >3% ms/tok win) are missed, so
the code is reverted and only this memo is kept.

## What was built

A vectorized int4 GEMV chunk (`qmoe_int4_vec32`): one aligned **128-bit `uint4` load streams
32 packed int4 weights** (four 8-value sub-chunks) per memory transaction, reusing each quant
block's scale/zero-point across the sub-chunks it covers. Accumulation is **byte-order-identical
within a thread** to four sequential scalar `qmoe_int4_chunk` calls, so the only numerical
change is the block-reduction fan-in (fewer, wider per-thread partials).

Wired behind new NVRTC entries (`qmoe_linear_vec_*`, `qmoe_gate_up_activate_vec_*`, templated on
a `bool Vectorized`), guarded to 4-bit layouts with `in_features % 32 == 0`, and applied to the
decode path only: fused FC1 gate/up (in=hidden=2048) and FC2/down (in=inter=512). Launch width
drops to one thread per 32-weight superchunk, rounded to whole warps and clamped `[32, 256]`
(FC1 → 64 threads, FC2 → 32 threads) so the warp-shuffle block reduction stays complete.

Sub-experiment coverage:
- **(2) memory-coalesce (vectorized wide weight read):** BUILT + MEASURED — this memo's result.
- **(1) DP4A compute-quant (int8 activation × int8 weight, `__dp4a`):** NOT built. Analytically
  dominated on this kernel (see "Why DP4A was not pursued").
- **int4 tensor-core / mma:** NOT built (same reasoning as DP4A; multiply is not the bottleneck).

## Risk / reward table

| Metric | Baseline (#765 tip) | After (vec int4) | Delta |
| --- | ---: | ---: | ---: |
| Decode ms/token (steady, 3 runs) | **11.157** | **11.704** | **+0.547 ms / +4.90% (REGRESSION)** |
| QMoE op, per-token window | 67.5–67.9 ms, ~13.15% | 84.5 ms, ~15.8% | **+25% QMoE op time** |
| MatMulNBits op (control) | ~70.0 ms, ~13.1% | ~70.0 ms, ~13.1% | unchanged |
| Oracle: teacher-forced argmax @119 | **33803** | **33803** | HOLDS |
| Oracle: primary margin logprob(33803)−logprob(5342) | **0.09375** | **0.09375** | **UNCHANGED (bit-identical)** |
| Oracle: autoregressive token @119 (#722 tripwire) | 33803 / 46283 (benign coin-flip) | **13** | **out of benign set → test FAILS** |
| QMoE GPU unit suite | 31/31 | 31/31 (incl. 2 new vec parity tests) | green |

Profile: `profile_native --pipeline --steady --warmups 1 --runs 3 --tokens 128`,
`ONNX_GENAI_PROFILE_OPS=1 ONNX_GENAI_CUDA_KV_MAX_LEN=262144`, model
`/home/justinchu/qwen36-35b-a3b-qmoe-artifacts`.
Oracle: `qwen36_35b_a3b_qmoe_native_cuda_matches_fp32_oracle` (`ONNX_GENAI_CUDA_GRAPH=1
ONNX_GENAI_CUDA_KV_MAX_LEN=4096`). The change-invariant dense-int4 (step 3) and full-fp32-CPU
(step 4) cross-checks were skipped in the after-run to save ~40 min of CPU oracle time — they
do not touch the CUDA QMoE kernel and passed on baseline; steps 1–2 (the QMoE CUDA path) ran.

## Why it regressed (root cause)

Cohaagen's NCU data already showed `qmoe_linear_*` is **latency / occupancy / barrier bound,
not bandwidth bound** (DRAM ~4% of peak, barrier ~37%, no-eligible ~22%). At `in_features=2048`
the scalar path already issues a coalesced 32-bit (`uint32` = 8 int4) load per thread with 256
threads/block — the weight read is *already* coalesced. Going to a 128-bit `uint4` load forces
**fewer threads** (64 for FC1, 32 for FC2) to cover the same K, which **lowers memory-level
parallelism (fewer outstanding requests) and occupancy** — exactly the resource this
latency-bound kernel depends on to hide load latency. The wider transactions do not compensate.
Net: QMoE op +25%. This mirrors the earlier ILP-2 unroll regression (#764 follow-up): any
transform that trades away warps/MLP for per-thread work loses on this M=1 GEMV.

## Why DP4A was not pursued

DP4A (int8×int8→int32 `__dp4a`) accelerates the **multiply**, but the multiply is not the
bottleneck: the kernel is barrier/occupancy/latency bound with DRAM at ~4% and low useful math
intensity. DP4A would (a) not reduce the dominant int4 **weight-read** stream (~503 MB/token,
still read as int4 then unpacked), (b) add per-row int8 activation-quant overhead and dynamic
scale bookkeeping, and (c) introduce accumulation-order / dequant-scale numerical risk on a
teacher-forced-locked path. On a kernel that is not multiply-bound, it is analytically dominated
by the memory-coalesce attempt that already regressed. Building it would consume another ~1 h
oracle/profile cycle for a predetermined-negative result.

## Numerical-risk finding (the useful positive datum)

The **capture-independent primary lock is completely insensitive** to this class of change: the
vectorized reduction reordering left `logprob(33803)−logprob(5342) = 0.09375` **bit-identical**
to baseline, and the new `rows==1` GPU parity tests conform to the CPU reference within the
2e-5 abs / 1e-4 rel band (block_size=16 with a 32-weight superchunk straddling two quant blocks
is exercised). So the *teacher-forced* numerical risk of int4 GEMV reduction restructuring is
effectively **zero** here. The oracle test failure is confined to the **autoregressive** #722
tripwire (token 13 ∉ {33803, 46283}) — a manifestation of the documented fp16 near-tie
coin-flip whose winner shifts under any tiny decode perturbation; the allowed-set is simply too
narrow to include this perturbation's outcome. It is not evidence of a corrupt kernel, but it
does mean the oracle **does not PASS** as written, so the ship gate is not met regardless of perf.

## Recommendation

**This lever is NOT worth a bigger investment in its current form.** Three independent surgical
attempts on the existing per-output-element block-reduction CTA shape have now all failed to
clear the >3% gate: FC2/down+combine fusion (~0.08%, Cohaagen), ILP-2 K-unroll (regressed,
Cohaagen), and this vectorized 128-bit weight read (**+4.9% regression, Quaid**). The consistent
signal is that the QMoE decode GEMV is **occupancy/MLP/barrier bound**, and any transform that
reduces active warps to buy per-thread width or ILP loses.

The remaining real headroom (the kernel is ~10× above the ~0.1–0.24 ms/token int4 bandwidth
floor) requires a **fundamentally different decode kernel structure**, not a chunk-level tweak:
a persistent / tiled QMoE decode worker that keeps many warps resident and busy across the
k=8 experts and the FC1→SwiGLU→FC2 chain (amortizing launch/graph-node overhead and the
per-output block reduction), while preserving fp32 accumulation order enough to hold the 33803
lock. That is a multi-day design with high integration risk against the ORT-style per-op
executor + graph-capture model — schedule it deliberately, not as a surgical follow-up.

DP4A / int4 tensor-core specifically should be **shelved** until the kernel is first made
compute-bound by such a restructure; on today's memory/latency-bound shape it cannot help.
