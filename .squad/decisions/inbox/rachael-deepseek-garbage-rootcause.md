# DeepSeek-V2-Lite real-weight garbage decode — root-cause diagnosis (Rachael)

**Date:** 2026-07-23
**Author:** Rachael (worker)
**Branch/build:** clean `origin/main` @ `569507c`, worktree `/home/justinchu/wt-ds-semantic`, `dump_intermediates` bin (native CUDA EP).
**Model:** `/home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4-blk32/model.onnx` (27 layers; L0 dense, L1–26 QMoE).
**Prompt:** "The capital of France is" → ids `[549,6077,280,7239,317]`, M=5 greedy prefill.
**GPU:** pinned `CUDA_VISIBLE_DEVICES=3` (idle, 4 MiB, 0 %).

> Status: **DIAGNOSIS ONLY — no fix implemented.** Route per "Recommended fix + owner".

---

## (a) Export-vs-engine verdict: **EXPORT CORRECT — ENGINE BUG** (high confidence)

Ran the *identical* exported ONNX through plain `onnxruntime` 1.27 (CPUExecutionProvider, fp32-capable) as a per-node/logits reference:

| Position | ORT argmax (reference) | Native argmax | ORT top logit | Native top logit |
|---|---|---|---|---|
| pos0 | 1022 | 207 (near-tie w/ 1022) | 11.72 | 14.38 |
| pos1 | 280  | 207 | 22.64 | 6.05 |
| pos2 | 254  | 2629 | 23.62 | 11.9 |
| pos3 | 317  | 11 | 22.45 | 14.2 |
| **pos4 (next token)** | **8913 (' Paris')** | **245** | **25.02** | **7.48** |

ORT on our ONNX decodes the coherent, confident continuation (pos4 = ' Paris', logit 25.0). Native produces garbage AND the logits are **heavily attenuated** — ORT's confident predictions (logit 22–25) collapse to ~6–14 in native. Combined with Batty's weight verification (dequantized L0 q/o_proj reconstruct the checkpoint), the export is correct; the **native CUDA engine miscomputes the forward pass**.

### Determinism correction (overturns an earlier finding)
An earlier pass reported "non-deterministic / uninitialized-memory read (deterministic ≤L17)". **That was an artifact of not pinning the GPU** on a shared box (another team's job occupies GPU 1 with 129 GiB). **Pinned to an idle GPU, native is deterministic** (identical logits across 8 launches; CUDA-graph on == off). The only residual non-determinism is a tiny (~0.1–0.4-logit) jitter that occasionally reorders *near-tie* top tokens (1022 vs 207) — a minor, separate fp-reduction-order issue, **not** the garbage bug. The garbage is a **deterministic compute error**.

---

## (b) First diverging op — clean per-position cosine vs ORT (deterministic, pinned GPU)

Method: expose one internal tensor at a time as an extra graph output (validated non-perturbing when logits stay at the dominant value), dump native to f32, cosine vs ORT per position.

| Tensor (layer 0) | overall cos | per-position cos [p0..p4] |
|---|---|---|
| `input_layernorm.RMSNorm_19` (embed+norm) | **1.0000** | [1.0, 1.0, 1.0, 1.0, 1.0] |
| `q_proj.MatMulNBits_20` (M=5) | **1.0000** | [1.0, 1.0, 1.0, 1.0, 1.0] |
| `kv_a_proj_with_mqa.MatMulNBits_24` | **1.0000** | [1.0, 1.0, 1.0, 1.0, 1.0] |
| `kv_b_proj.MatMulNBits_27` | **1.0000** | [1.0, 1.0, 1.0, 1.0, 1.0] |
| `RotaryEmbedding_32` (K-RoPE) | **1.0000** | [1.0, 1.0, 1.0, 1.0, 1.0] |
| `Reshape_39` (K into Attention) | 0.9972 | [0.998, 0.996, 0.999, 0.994, 0.999] |
| **`o_proj.MatMulNBits_41`** | **0.8802** | **[1.0, 0.829, 0.856, 0.850, 0.812]** |

**The first clean divergence is at the attention output: position 0 is exact (cos 1.0), positions 1–4 diverge (cos ≈ 0.83).** This is a **M>1 prefill, position-dependent** error (position 0 = single causal key / identity RoPE is trivially correct; positions 1–4 involve non-trivial rotation and multi-key causal attention).

Downstream this compounds through the QMoE stack (per-layer residual `Add` cosine vs ORT): L1 = 0.31, staying 0.03–0.46 through L23, with native activations **severely attenuated** (ORT residual std reaches 28 at L3 / 18 at L25; native stays ~0.07–2.7). The QMoE layers **amplify** the upstream attention error; they are not the origin.

### What this exonerates (corrects prior hypotheses)
- **MatMulNBits fp16 M>1 prefill — EXONERATED.** q_proj/kv_a/kv_b are all cos = 1.0 at *every* position (M=5). Standalone MatMulNBits nodes (q_proj, o_proj, down_proj — no fused `gamma`) all use the *same* plain f16 prefill GEMM; since q_proj is perfect at all rows, the plain GEMM handles all M>1 rows correctly, so **o_proj's pos1–4 error must come from its input (the attention output)**, not from o_proj itself. → Contradicts Marsten's "fp16 MatMulNBits prefill" localization.
- **"Uniform softmax / attention collapse"** (an earlier session) — was a *perturbed-probe artifact*; discard.
- **QMoE kernel** — not the primary bug; QMoE scratch buffers are correctly guarded (gather writes every routed row with a `route<routes ? … : 0` sentinel; the grouped-GEMM/GEMV partition by `expert_counts[expert] >= gemm_min_tokens` is exact).
- **Uninitialized memory / non-determinism** — artifact of an un-pinned shared GPU (see above).

---

## (c) Root cause

The **native attention path miscomputes query rows 1..M-1 during M>1 prefill** on real weights. All measurable attention *inputs* are correct at all positions (Q/K/V projections cos 1.0; K-RoPE cos 1.0; K-into-attention cos 0.997; V is a reshape of cos-1.0 `kv_b`; Q into attention = concat of cos-1.0 q_proj "nope" part + Q-RoPE). The attention *output* (seen through the proven-correct o_proj) is exact at position 0 and wrong at positions 1–4.

The exact defect could not be split below the Q-RoPE→Attention block because **the native EP zeroes the Q, V, and attention-output buffers** to any observer (they read all-zeros through graph-output/Mul-probe/truncation, while the K path reads normally). This Q/V-buffer-zeroing asymmetry is itself a lead: the MLA attention op appears to consume/alias the Q and V buffers in place. Two candidates, both in the **engine attention path**:

1. **Q-RoPE multi-head path** — `crates/onnx-runtime-ep-cuda/src/kernels/rotary_embedding.rs`. Q-RoPE (`RotaryEmbedding_31`) rotates the 16-head, 1024-dim query (16×64 rope lanes); K-RoPE (`RotaryEmbedding_32`) is single-head 64-dim and is **verified correct at all positions**. The untested link is the *multi-head* Q rotation at non-zero positions — a head-stride/pairing bug there would leave position 0 (rotation ≈ identity) correct and grow with position. **Leading suspect** (the moderate ~0.83 cosine is more consistent with a rotation error than a fully broken attention).
2. **MLA `attention_row` / `build_kv` multi-key path** — `crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs`. Scrutinize the per-query causal frontier (`causal_limit = i + offsets[b]`; verify `offset==0` for empty-past prefill), the `build_kv` 3D→4D gather for keys/values at `j>0`, the Q `q_is_3d` offset, and the asymmetric MLA head sizes (`head_size=192` for QK vs `v_head_size=128` for V).

**Why real weights but not synthetic:** the DeepSeek path was only ever validated *structurally* with synthetic random weights (which produce finite-but-meaningless tokens), so the position-dependent attention error was never compared against a trusted reference and went undetected.

---

## (d) Recommended fix + owner

**Owner: ENGINE / attention-kernel owner** (NOT `matmul_nbits.rs`/Deckard — exonerated; NOT `qmoe.rs` — only amplifies).

1. **First, confirm the split** the diagnosis couldn't (Q/V buffers are unreadable via probes): add a temporary debug copy so Q-RoPE (`RotaryEmbedding_31`) output can be read, and compare per-position to the ORT reference (`ort_allres`/probe harness in `/home/justinchu/onnx-genai/.rachael-diag`). If Q-RoPE pos1–4 diverges → fix the **multi-head Q rotation** in `rotary_embedding.rs`. If Q-RoPE is correct → fix the **MLA multi-key prefill** in `standard_attention.rs` (`attention_row`/`build_kv`), per the checklist in (c).
2. Add a **regression test**: real-weight (or reference-checked) DeepSeek-V2-Lite prefill must match an `onnxruntime`-on-our-ONNX reference at pos4 (= token 8913 ' Paris') and per-position argmax, with a fp16-tolerance cosine gate on layer-0 `o_proj` (pos1–4). The current synthetic-weight structural test passes on garbage and must be superseded.

### Reproduce
```
source /home/justinchu/onnx-genai/.cudaenv.sh ; export CUDA_VISIBLE_DEVICES=3   # idle GPU
cd /home/justinchu/wt-ds-semantic
cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin dump_intermediates
D=/home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4-blk32
./target/release/dump_intermediates $D/model.onnx "549,6077,280,7239,317" logits   # native garbage, pos4=245
# ORT reference (coherent, pos4=8913): .rachael-diag/ortvenv + ort_allres.py
```

---

# UPDATE — CONFIRMED ROOT CAUSE + FIX (VERIFIED)

**Status: FIXED.** Branch `fix/deepseek-mla-attention-prefill` (commit `1fe314f`, off clean origin/main `569507c`), pushed, NOT merged (awaiting fresh reviewer).

## Root cause (corrects the (c)/(d) speculation above)

RoPE and `attention_row` are **both correct** — exonerated. The real defect is a **device-stream ordering race**, not an attention-math bug.

- Kernels run on the EP's **dedicated non-default (non-blocking) stream**, but `CudaRuntime::dtod` issues `cuMemcpyDtoD` on the **legacy default stream**. Non-blocking streams are **not implicitly ordered** against the default stream, so the synchronous D2D copy can run **before a producer kernel finishes writing its source buffer**.
- This is hit by `copy_reshape` (Reshape/Squeeze, `movement.rs:271`), which materializes `Reshape_39`(K) / `Reshape_30`(V) feeding MLA attention. The fast **k_rope** columns are always ready; the slower **kv_b_proj → Reshape_28 → Split_29 → Concat_37 → Reshape_39** path producing **k_nope + V** is frequently read **stale (zero)** → attention garbage. DeepSeek's MLA assembles K/V through a tight `Concat→Reshape→Attention` chain that reliably exposes the race; dense models (Phi/Qwen) happened to hide it. `dtoh` already synchronizes for this exact reason; `dtod` did not.

### Proof it is a race (not compute)
- Same binary, no instrumentation: identical runs gave **different** logits (run1/2 pos0 argmax=207 garbage, run3 pos0=1022 correct).
- `CUDA_LAUNCH_BLOCKING=1` (serializes launches) → **deterministic and exactly matches ORT**: pos0=1022, pos1=280, pos2=254, pos3=317, **pos4=8913 (' Paris', 25.0)**.
- Kernel pointer dump showed `present_key`/`present_value` do **not** alias `k_cur`/`v_cur` → ruled out in-place aliasing, leaving stream ordering.

## Fix (one line, general, DRY)

`crates/onnx-runtime-ep-cuda/src/runtime.rs` — `CudaRuntime::dtod` now calls `self.synchronize()?` before `memcpy_dtod_sync`, draining the EP stream so the copy always sees fully-produced bytes (mirrors `dtoh`). General: covers every `dtod` caller (movement `copy_reshape`, provider `copy`, CSA checkpoint). Capture-safe: `cuMemcpyDtoD` is illegal during stream capture, so `dtod` is never on the capture path (Qwen cuda-graph decode confirmed unaffected). **Owner: ENGINE / runtime** (not matmul_nbits/Deckard, not qmoe).

## Verification (native CUDA, GPU3)

Before → `"Grants. Links Choir SAC Candle CSP..."` (blk128) / `"to成33333333..."` (blk32) — garbage.

After (deterministic, 4/4 runs match ORT):
- **block-32** `deepseek-v2-lite-real-int4-blk32`: `" Paris.\nThe capital of France is Paris.\nThe capital of France is Paris. ..."`
- **block-128** `deepseek-v2-lite-real-int4`: `" Paris.\n\nThe Eiffel Tower is in Paris.\n\nThe Eiffel Tower is in Paris. ..."`
- Per-position prefill argmax == ORT-on-our-ONNX: pos0=1022, pos1=280, pos2=254, pos3=317, **pos4=8913 (' Paris')**.

## Regression test

`crates/onnx-runtime-ep-cuda/src/runtime.rs` → `runtime::tests::dtod_waits_for_pending_stream_writes`: enqueues a slow (clock-spin) producer kernel on the EP stream, then `dtod` without an explicit sync, and asserts the copy sees the sentinel (not the pre-launch poison). **Verified FAIL without the fix** (`got -999`), **PASS with it**. General (guards the stream-ordering invariant for all `dtod` users; not DeepSeek-specific).

## Gate (source .cudaenv.sh; CUDA_VISIBLE_DEVICES=3; clean worktree off origin/main)

- `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` → **202 passed / 0 failed** (201 baseline + new test).
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda --lib -- -D warnings` → clean.
- Qwen2.5-0.5b native decode → coherent (`" Paris. It is the largest city in the country ..."`), cuda_graph enabled captures=1 replays=21 fallbacks=0 → **no dense-model regression**.

Branch: `fix/deepseek-mla-attention-prefill` @ `1fe314f`. Do NOT merge — reviewer to verify.
