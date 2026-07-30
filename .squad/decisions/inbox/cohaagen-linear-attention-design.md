# Design note — CUDA EP `com.microsoft::LinearAttention` (Gated DeltaNet)

**Author:** Cohaagen (CUDA EP) · **Issue:** #67 (refs #384) · **Date:** 2026-07-30

## Why

`LinearAttention` (Gated DeltaNet / gated delta-rule linear attention, as used by
the Qwen3.5 / Qwen3-Next *hybrid* family) is the second-ranked CUDA coverage gap
after `CausalConvWithState` (#480). Empirical placement probe over the real
Foundry text decoders (all decline today — "no handler for
com.microsoft::LinearAttention"):

| model (text decoder)        | LinearAttention nodes | placed on CUDA (before) |
|-----------------------------|:---------------------:|:-----------------------:|
| qwen3.5-0.8b-generic-cpu-2  | 18                    | 0                       |
| qwen3.5-2b-text-generic-cpu | 18                    | 0                       |
| qwen3.5-9b-generic-cpu-2    | 24                    | 0                       |

Real node config (from the probe): `update_rule=gated_delta`, `scale=1.0`,
`q_num_heads=kv_num_heads=16` (0.8b/2b) or `q=16, kv=32` (9b, **inverse GQA**),
`d_k=d_v=128`, all six inputs present, decay `[B,T,H_kv]` and beta `[B,T,H_kv]`
(both per-head), all Float32. Pairs with `CausalConvWithState` to land the hybrid
decode path on CUDA.

## Op contract (oracle = CPU EP `linear_attention.rs`, faithful port of ORT
`contrib_ops/cpu/bert/linear_attention.cc`)

Domain `com.microsoft`, opset 1. Inputs (channels-last, trailing optionals):

* `query`  `[B, T, H_q·d_k]`
* `key`    `[B, T, n_k·d_k]`
* `value`  `[B, T, H_kv·d_v]`
* `past_state` (opt) `[B, H_kv, d_k, d_v]` — zeros when absent
* `decay` (opt) `[B, T, H_kv]` (per-head) or `[B, T, H_kv·d_k]` (per-key-dim)
* `beta`  (opt) `[B, T, H_kv]` (per-head) or `[B, T, 1]` (shared)

Outputs: `output` `[B, T, max(H_q,H_kv)·d_v]`, `present_state` `[B,H_kv,d_k,d_v]`.

Attributes: `q_num_heads`, `kv_num_heads` (positive ints), `update_rule` ∈
{linear, gated, delta, gated_delta (default)}, `scale` (0/absent → `1/sqrt(d_k)`).

Per timestep `t`, per kv-head `h` (state `S[d_k,d_v]`, row-major `S[i·d_v+j]`):

```
decay      (gated/gated_delta):  S[i,j] *= exp(g_t[i or head])
retrieval  (delta/gated_delta):  r[j]   = Σ_i S[i,j]·k_t[i]
delta      (delta/gated_delta):  d[j]   = beta_t·(v_t[j] − r[j]);  S[i,j] += k_t[i]·d[j]
linear     (linear/gated):       S[i,j] += k_t[i]·v_t[j]
readout:                         o_t[j] = scale · Σ_i q_t[i]·S[i,j]   (updated S)
```

GQA both directions: `H_q ≥ H_kv` → `heads_per_group = H_q/H_kv` query heads
share one kv state; `H_q < H_kv` (inverse) → `heads_per_group = 0`, output slot
is `h_kv`, query head `h_q = h_kv·H_q/H_kv`. `n_k ≤ H_kv` via
`kv_per_k_head = H_kv/n_k` (`h_k = h_kv/kv_per_k_head`).

## State handling & parallelization

**Key structural fact:** column `j` of the state matrix evolves *independently* of
every other column — retrieval, delta, linear update and readout for output
element `j` touch only `S[:, j]`, `k_t`, `v_t[j]`, `g_t`, `beta_t`, `q_t`. So the
whole op is embarrassingly parallel across `(b, h_kv, j)`; each such triple is one
independent sequential scan over `t`.

**Mapping:** one CUDA thread per `(b, h_kv, j)` (`grid×block` covers
`B·H_kv·d_v` threads, block=256, grid-stride). Each thread:

1. Loads its state column into a **per-thread f32 register/local array**
   `float sc[MAX_DK]` (init from `past_state` widened to f32, else 0).
2. Runs the full `t`-loop entirely in **f32** (matching the CPU/ORT `float`
   kernel exactly — no per-step narrowing), reading `k/v/q/decay/beta` widened
   per-access via `__half2float`/`__bfloat162float`.
3. On exit writes its column back to `present_state` (narrowing to the I/O dtype).

Keeping `sc` in f32 across all timesteps reproduces ORT's f32 state precisely;
`f16`/`bf16` inputs are widened on read and outputs narrowed on write. `MAX_DK =
256` bounds the local array (real `d_k = 128`); the claim gate rejects `d_k > 256`
so we never claim an op we can't run. No shared memory, no cross-thread reduction,
no scratch alloc.

## Dtype / accumulation plan

* Supported I/O dtypes: `Float32`, `Float16`, `BFloat16` (all inputs + outputs
  same dtype — the claim gate enforces this).
* All accumulation (retrieval `r`, readout `o`, decayed/updated state) in **f32**.
* `scale` resolved on the host (`1/sqrt(d_k)` when attr is 0/absent), passed as f32.
* 3 NVRTC entry points (`_f32/_f16/_bf16`) from one C++ template; half headers
  required only for the non-f32 stems.

## Wiring / verification

Factory + `unsupported_reason` claim gate in `provider.rs`; register
`("LinearAttention","com.microsoft",1)`; add to `CUDA_COVERED_OPS`; dedicated GPU
parity suite `linear_attention_gpu.rs` vs the CPU EP oracle (all four rules;
standard + inverse GQA; key-head sharing; per-head + per-key-dim decay; per-head +
shared beta; f32/f16/bf16; a chained-vs-full state-carry proof) plus a
`dedicated(...)` entry in `cuda_conformance_gpu.rs`. Placement probe confirms the
18/18/24 nodes now place on CUDA.

## Folded-in follow-ups

Deferred (keep this PR focused on LinearAttention): `com.microsoft` domain
`RotaryEmbedding` registration and `Bool`-input `NonZero`. Both are independent of
LinearAttention and carry their own semantic checks (com.microsoft RotaryEmbedding
interleaving/attribute differences vs ai.onnx; NonZero bool path) — tracked as
separate small follow-ups so the hybrid-attention PR stays reviewable.
