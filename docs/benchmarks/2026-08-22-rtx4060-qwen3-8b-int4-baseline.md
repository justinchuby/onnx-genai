# Qwen3 int4 decode baseline — RTX 4060 Laptop (native CUDA EP)

**Author:** Sebastian (performance) · **Date:** 2026-08-22 · **Box:** primary
Windows dev laptop (shared with build agents).

This is a *baseline measurement*, not an optimization. The goal was a trustworthy
decode number for later kernel work to be measured against. The honest headline
is that **this hardware tier cannot give a stable absolute decode number for a
model this size** — thermal/power throttling swings decode ~2–4× within a single
sitting — so the durable deliverables here are the *contention-invariant*
structural facts and an explicit methodology mandate for future work, not a
single tok/s figure.

Everything below is split into **measured** (I ran it) and **inferred** (derived
from measured facts). Where a number is thermally unstable it is given as a range
with the state it was observed in.

---

## 0. What model this actually is (read this first)

The team's target is "Qwen3.8" — the 27B Gated-DeltaNet + MTP hybrid (GGUF arch
`qwen35`) that prior H200 sessions measured at ~61 tok/s. **That model does not
fit this 8 GB card** (int4 weights alone ≈ 13–14 GB), so per the task's explicit
instruction I did **not** silently substitute it. I used the **largest Qwen3.x
int4 that actually fits 8 GB**:

- **Model: Qwen3-8B (dense), int4, block-32.** Source GGUF
  `Qwen/Qwen3-8B-GGUF → Qwen3-8B-Q4_K_M.gguf` (5.03 GB), converted with mobius
  `build-gguf … --dtype f16 --runtime onnx-genai`.
- It is a **dense** transformer: **no Gated-DeltaNet, no MTP/nextn head**. So the
  decode path here is standard dense int4 M=1 GEMV + GQA attention — it does **not**
  exercise the GDN recurrence / MTP levers the 27B work was about. Numbers here do
  not transfer to the 27B hybrid.
- ONNX int4 weights: **4.53 GB** (`model.onnx.data`); loads with ~1.76 GB VRAM
  headroom on the 8 GB card.

### Block size: only block-32 is reachable through mobius

mobius **normalizes every supported int4 GGUF form (Q4_0/Q4_1/Q4_K/Q6_K) to a
4-bit, block-32 asymmetric** `MatMulNBits` target (`integrations/gguf/_builder.py`
"Q4_K-containing mixed presets target 4-bit, block-32 asymmetric";
`_repacker.py` repacks 256-element Q4_K super-blocks to `(4, 32)`). **There is no
mobius path to a block-128 model.** So the block-128 requantization the task asked
me to try is *not feasible through this exporter*, and the fast `GENERAL_BS`
(block ≠ 32) CUDA GEMV variants are unreachable for any mobius-built model. This
baseline is therefore block-32, which is also the only thing a mobius user can
ship today.

---

## 1. Machine (measured)

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 **Laptop**, `sm_89`, 8 GiB, WDDM |
| Driver / CUDA | 591.55 / reports CUDA 13.1 |
| Peak HBM bandwidth | 256 GB/s (spec) |
| CUDA libs | pip wheels (cu13) under anaconda `site-packages\nvidia`; no toolkit |
| Runtime | `origin/main` `6fdc04d75` + local unblocks (§2), `--features native-cuda,bench-ort`, native decode backend |
| ORT support lib | 1.28.0 (loaded for `bench-ort`; **decode ran on the native EP**) |

Box was verified compute-idle before timing (three resident python PIDs, each
0 s CPU-time delta over 4 s — held state, no active compute). GPU device memory
was **not** polled during any timing window (the ~30% perturbation trap).

---

## 2. Four blockers hit on the way to a number (all reproduced)

Getting *any* coherent Qwen3 decode required working around four separate defects.
These are the most actionable output of this session; each deserves its own issue.

1. **`origin/main` does not compile with `native-cuda`.** The #1579 memory-stack
   merge (`a36964280`) added `let device = context.device();` in
   `onnx-runtime-ep-cuda/src/provider.rs`, but `RegisteredMemoryContext` has no
   `device()` accessor → `E0599`. CI does not build this feature, so it landed
   red. **Fix applied (in this PR):** add the one-line accessor
   `RegisteredMemoryContext::device()` returning `self.record.registered.device()`
   (the underlying `RegisteredProviderContext::device()` already exists). Clearly
   correct and required to build at all.

2. **`#1704` regressed `#1715` — the native loader rejects mobius's own metadata.**
   Current mobius (`--runtime onnx-genai`) emits the SSA "north-star" workflow
   metadata (`workflow_ssa`, `serving_service_contract`, `bounded_state_recurrence`,
   …). PR #1715 (Aug 21 20:19) deliberately made the loader **log-and-continue**
   on those capabilities so "the mobius-emitted workflow metadata … loads and
   decodes (7.233 ms/token)". PR #1704 (Aug 22 07:44, ~12 h later) reintroduced a
   **fatal** `admit_inference_metadata` that bails on unsupported capabilities,
   plus tests locking that rejection. Net effect on today's HEAD: a package that
   decoded yesterday now fails at load with *"Unsupported inference metadata
   capabilities: bounded_state_recurrence, …"*. **Local unblock for measurement:**
   restored #1715's log-and-continue in `admit_inference_metadata` (the bare
   decoder runs off `metadata.decoder_io()` and never executes those workflow
   features; coherence was verified independently, §3). This is a genuine
   #1704↔#1715 conflict the team must adjudicate — I did not include it in this
   PR; it is reported for decision.

3. **mobius drops Qwen3 QK-norm → incoherent output.** With the metadata unblocked,
   the model loaded and ran on CUDA (graph capture engaged) but produced garbage
   ("*capital of of of capital capital…*"). Root cause: mobius's GGUF tensor map
   puts `qwen3` in `_LLAMA_FAMILY` and only adds the per-head Q/K RMSNorm mappings
   (`_QWEN35_HYBRID_EXTRAS`) when `arch == "qwen35"` — **never for plain `qwen3`** —
   and the ArchitectureConfig never sets `attn_qk_norm`. So `blk.N.attn_q_norm` /
   `attn_k_norm` were logged "Unmapped … (skipped)" and the exported graph had
   **0 QK-norm nodes**. Qwen3 requires per-head QK-normalization, so attention was
   mathematically wrong. **Fix applied in mobius (local):** infer
   `attn_qk_norm` from tensor presence (mirroring the existing `_infer_attn_qkv_bias`)
   and add a `_QWEN3_EXTRAS` qk-norm mapping for `arch == "qwen3"`. After
   reconversion the graph has **72 QK-norm nodes** (36 layers × q,k) and decode is
   **coherent** ("*Paris. The capital of the United States is Washington, D.C. …*").
   This belongs in the mobius repo; the patch is recorded in §7.

4. **Native CUDA prefill of a real prompt fails on the 8 GB VMM path.** One-token
   decode-time KV growth works fine, but batched prefill of a multi-hundred-token
   prompt errors with `Error: size native CUDA KV mapped-growth transaction`
   (`--prefill-sweep`), and the full 27.9k-token fixture is additionally rejected
   by a legitimate KV-budget admission failure (8 GB tier: 4.1 GB KV ceiling vs
   1.76 GB free after weights — Rule 9, not a bug). Consequence: **prefill
   per-token was not cleanly measurable** here (§5).

---

## 3. Correctness gate (measured)

Greedy, native CUDA, `--prompt "The capital of France is"`, 128 tokens:

> " Paris. The capital of the United States is Washington, D.C. The capital of
> Japan is Tokyo. The capital of Brazil is Brasília. The capital of Canada is
> Ottawa. … The capital of China is Beijing. The capital of the United Kingdom is
> London. …"

Coherent and factual — the QK-norm fix (§2.3) is the difference between this and
the pre-fix garbage. Not byte-locked to an oracle (relaxed golden-lock bar);
coherence + factual correctness is the gate.

---

## 4. Decode (measured) — and why it is a range, not a number

`profile_native --ep cuda --steady`, greedy, native backend, block-32 int4.

**Decode throughput is thermally unstable and degrades *within* a single run.**
The RTX 4060 Laptop cannot sustain the ~4.4 GB/token weight stream (§6) without
throttling. Representative observations, all this session, same binary/model:

| condition | decode ms/token | tok/s | notes |
|---|---|---|---|
| first cold 32-tok smoke | **22.7** | **44.1** | best ever seen (coolest, shortest) |
| cold anchor, 96 tok, 180 s cooldown | **28.3** | **35.3** | most reproducible "cold" point |
| runs=5×128 tok, one sitting | 27.5 → **42.8** → 53.1 | 36.3 → **23.4** → 18.8 | monotonic slowdown within the sitting |
| hot (back-to-back after a run) | 60 – 88 | 11 – 17 | fully throttled |

Within one `runs=5` invocation the spread is **~1.9×**; across sittings it reaches
**~4×** (18.8 → 44.1 tok/s). Per the measurement-discipline skill, an effect
smaller than an arm's own spread is unmeasured — here that noise floor is enormous.
**No single absolute decode number from this box is trustworthy for a model this
size.** The least-throttled, most-reproducible anchor is **~28 ms/token
(~35 tok/s)**; treat that as an upper-ish bound, not a stable baseline.

**CUDA graph capture engages cleanly** (measured, and this *is* load-invariant):
`cuda_graph: enabled=true … fallbacks=0`, no `cuda_graph_decline_reason` ever
emitted. Captures/invalidations track KV-growth boundaries (e.g. 7 captures / 7
invalidations / 881 replays over a 5×128 run) and replays dominate — decode runs
captured.

---

## 5. Prefill (measured, but not separable)

Prefill per-token could **not** be isolated on this box:

- `--prefill-sweep {256,768,1536}` and even `{128,256,384}` → `Error: size native
  CUDA KV mapped-growth transaction` (§2.4).
- Steady `--no-prefix-cache` prefill of a **332-token** prompt: 718 / 1427 /
  1741 ms across 3 runs (**2.4× spread**), dominated by one-time graph-capture cost.
- The **5-token** prompt's cold prefill (~1330 ms) was *slower* than the 332-token
  prompt's best (718 ms), because prefill time is dominated by fixed capture +
  thermal state, not prompt length. Differencing therefore yields a nonsensical
  (negative) slope: **prefill per-token is unmeasured here.**

Inferred: on this tier prefill is fixed-cost-bound (graph capture ≈ 0.7–1.7 s) and
KV-budget/growth-limited; a clean marginal-prefill rate needs a quiet, thermally
stable box and a fix for the mapped-growth prefill path.

---

## 6. Structural facts (measured, fully load-invariant)

These do not move with thermal state and are the trustworthy core of this report.

### Per-op ranking — uncaptured run (`ONNX_GENAI_CUDA_GRAPH=0 ONNX_GENAI_PROFILE_OPS=1`)

> Capture hides ops inside the graph; this profile is from an **uncaptured** run,
> so its absolute latency (it reported ~800 ms/token with per-op instrumentation)
> is **not** the decode number — only the op-type *ranking/percentages* are
> meaningful, and those were stable across every forward pass:

| op type | % of decode step | calls/step |
|---|---|---|
| **MatMulNBits** | **75.8%** (73.7–76.8) | 217 |
| SkipSimplifiedLayerNormalization | 11.0% | 35 |
| Attention | 4.4% | 36 |
| Reshape | 3.2% | 144 |
| RMSNormalization | 2.1% | 73 |
| RotaryEmbedding | 2.0% | 72 |

Same shape as the prior hunyuan profile, with int4 GEMV *more* dominant here
(~76% vs 69%). Decode is unambiguously **int4 `MatMulNBits`-bound.** The 73
`RMSNormalization` calls include the 72 new QK-norm ops (36 layers × q,k) from
the §2.3 fix; they cost ~2% — cheap and correct.

### Dominant MatMulNBits shapes (exact, from the ONNX graph)

253 `MatMulNBits` nodes, all block-32 4-bit. Per **decode step** (M=1):

| K | N | count | role | bytes/op |
|---|---|---|---|---|
| 4096 | 12288 | 72 | gate + up | 29.10 MB |
| 12288 | 4096 | 36 | down | 29.10 MB |
| 4096 | 4096 | 72 | q + o | 9.70 MB |
| 4096 | 151936 | 1 | lm_head | 359.78 MB |
| 4096 | 1024 | 72 | k + v (GQA, 8 kv-heads) | 2.42 MB |

Bytes/op includes packed int4 weights + f16 scales + packed int4 zero-points
(block-32 asymmetric). **Total int4 weight streamed per token = 4.375 GB.**
(KV-cache reads at short context add ≈ 10 MB/step and are negligible; embeddings
are a single `GatherBlockQuantized`.)

### Memory roofline (inferred from the two measured facts above)

- Floor = 4.375 GB / 256 GB/s = **17.1 ms/token → 58.5 tok/s** memory-bound ceiling.
- Achieved fraction of the 256 GB/s peak, by state:
  - cold anchor (28.3 ms): 4.375 GB / 0.0283 s = **154 GB/s ≈ 60%** of peak;
  - best 32-tok smoke (22.7 ms): 193 GB/s ≈ **75%**;
  - throttled (52–88 ms): 50–84 GB/s ≈ **20–33%**.

The 60–75% at the least-throttled points is *already good* utilization for a
block-32 M=1 GEMV (better than the ~50% quoted for hunyuan block-128). The large
low-end is throttling: under sustained load the card's delivered bandwidth itself
collapses, so the "% of peak" is only meaningful at the cold anchor. **The lever
implied by the roofline (17.1 ms floor vs ~28 ms cold) is ~1.6×**, but capturing
it requires a thermally stable box — on this laptop the throttle swing dwarfs it.

---

## 7. Reproduce

**Model (requires the mobius QK-norm fix, §2.3):**
```
# in mobius, patch integrations/gguf/_config_mapping.py + _tensor_mapping.py:
#   - add _infer_attn_qk_norm(model) (mirrors _infer_attn_qkv_bias) and pass
#     attn_qk_norm=_infer_attn_qk_norm(model) into ArchitectureConfig
#   - add _QWEN3_EXTRAS = {attn_q_norm→self_attn.q_norm, attn_k_norm→self_attn.k_norm}
#     and `elif arch == "qwen3": result.update(_QWEN3_EXTRAS)` in _build_mapping
python -m mobius build-gguf Qwen3-8B-Q4_K_M.gguf -o qwen3-8b-int4-b32 \
    --dtype f16 --runtime onnx-genai
```

**Binary (requires the §2.1 build fix; §2.2 admission unblock to load the model):**
```
cargo build --release -p onnx-genai-bench --features native-cuda,bench-ort --bin profile_native
```

**Env (Windows, pip-wheel CUDA — see `windows-cuda-runbook.md` §1):**
```powershell
$sp='…\site-packages\nvidia'; $cu="$sp\cu13"
$env:CUDA_PATH=$cu; $env:CUDA_HOME=$cu
$env:PATH="$cu\bin\x86_64;$sp\cudnn\bin;"+$env:PATH
```

**Decode (expect thermal variance — cool the GPU, take the first cold burst):**
```powershell
.\target\release\profile_native.exe --model qwen3-8b-int4-b32 --ep cuda --steady `
    --tokens 96 --warmups 1 --runs 1 --prompt "The capital of France is"
```

**Per-op ranking (uncaptured):**
```powershell
$env:ONNX_GENAI_PROFILE_OPS='1'; $env:ONNX_GENAI_CUDA_GRAPH='0'
.\target\release\profile_native.exe --model qwen3-8b-int4-b32 --ep cuda --steady --tokens 48 --runs 2
```

---

## 8. Bottom line for later kernel work

- **Do not quote an absolute decode tok/s from this box for an 8 B model.** The
  thermal swing is ~2–4×. Future kernel A/B **must** be alternating arms in one
  sitting with a control arm the change cannot touch (measurement-discipline
  skill), reading the *ratio*, not the absolutes.
- The cost to attack is confirmed: **int4 `MatMulNBits` M=1 GEMV, ~76% of the
  step, 4.375 GB/token, block-32** (the fast block-≠32 kernels are unreachable via
  mobius). Cold utilization is already ~60–75% of the 256 GB/s roofline, so the
  realistic decode headroom on this tier is ~1.6× (17.1 ms floor vs ~28 ms cold),
  not a large multiple — and it is gated behind thermal stability, not just kernel
  work.
- Four defects block the *path* to this number today (§2). The build fix (§2.1) is
  in this PR; the other three (admission #1704↔#1715, mobius Qwen3 QK-norm, native
  CUDA prefill KV mapped-growth) are reported for their owners.
