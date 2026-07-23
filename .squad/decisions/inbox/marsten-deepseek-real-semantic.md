### 2026-07-23: Clean-main real DeepSeek-V2-Lite semantic validation fails for block-128 and block-32
**By:** Marsten
**What:** ❌ Both real int4 exports run entirely on native CUDA from clean
`origin/main` `569507c`, but neither is semantically coherent for the exact
prompt `The capital of France is`. Block-128 produces multilingual garbage at
25.58 tok/s; block-32 immediately collapses to repeated `3` tokens at
26.01 tok/s. Both diverge from HF at token 1.
**Why:** This supersedes Marsten's earlier stale-checkout reports. The clean-main
binary proves block-128 execution support and zero EP fallbacks, but not semantic
fidelity. Isolated real-input checks clear QMoE packing/execution and MLA
Attention; the first demonstrated native-vs-ORT numerical divergence is the
layer-0 fp16 `MatMulNBits` q-projection, whose small prefill error is amplified
through MoE routing across 27 layers.

## Authoritative environment

- Binary:
  `/home/justinchu/wt-ds-semantic/target/release/profile_native`
- Source worktree: `/home/justinchu/wt-ds-semantic`
- Revision: clean `origin/main` at `569507c`
- GPU: physical GPU 0, idle NVIDIA H200
- Strict placement: `ONNX_GENAI_REQUIRE_CUDA=1`
- Prompt: `The capital of France is`
- Prompt IDs: `[549, 6077, 280, 7239, 317]`
- Greedy, 48 new tokens, one warmup, one measured run
- HF bf16 reference:

```text
 Paris. The most famous landmark in Paris is the Eiffel Tower.
```

## A. Real block-128 export

Artifact:
`/home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4/`

- Verdict: ❌ **broken**
- Throughput: **25.58 tok/s**, 39.088 ms/token
- Generated: 48/48 valid vocabulary IDs
- Finite: all top-20 token-0 log-probabilities finite; no NaN/inf surfaced
- CPU EP fallbacks: **0** (strict CUDA placement accepted)
- CUDA graph: `enabled=false`, captures 0, replays 0, fallbacks **0**
- MLA capture status: not attempted because graph capture is disabled; importantly,
  the stale-checkout MLA capture fallback is absent
- First HF divergence: token **1**

Actual decoded text:

```text
 Grants. Links Choir SAC Candle CSP CSP Kir Kirpt充满 ... Kir三位季充满正是日本优生活是指从优cons我们充满充满地充满充满从这些充满建议华式悠生活从充满生活。充满严到今天支配
```

Early comparison:

| # | Block-128 native | HF bf16 |
|---:|---|---|
| 1 | `67545` / `" Grants"` | `8913` / `" Paris"` |
| 2 | `13` / `"."` | `13` / `"."` |
| 3 | `32556` / `" Links"` | `429` / `" The"` |
| 4 | `73915` / `" Choir"` | `1094` / `" most"` |
| 5 | `85499` / `" SAC"` | `9679` / `" famous"` |
| 6 | `82234` / `" Candle"` | `44872` / `" landmark"` |
| 7 | `95910` / `" CSP"` | `279` / `" in"` |
| 8 | `95910` / `" CSP"` | `8913` / `" Paris"` |
| 9 | `27421` / `" Kir"` | `317` / `" is"` |
| 10 | `27421` / `" Kir"` | `254` / `" the"` |
| 11 | `462` / `"pt"` | `427` / `" E"` |
| 12 | `17057` / `"充满"` | `96575` / `"iffel"` |

## B. Block-32 cross-check

Artifact:
`/home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4-blk32/`

- Verdict: ❌ **broken**
- Throughput: **26.01 tok/s**, 38.444 ms/token
- Generated: 48/48 valid vocabulary IDs
- Finite: all top-20 token-0 log-probabilities finite; no NaN/inf surfaced
- CPU EP fallbacks: **0** (strict CUDA placement accepted)
- CUDA graph: `enabled=false`, captures 0, replays 0, fallbacks **0**
- MLA capture status: not attempted; no capture fallback
- First HF divergence: token **1**

Actual decoded text:

```text
 to成3333333333333333333333333333333333333333333333
```

Early comparison:

| # | Block-32 native | HF bf16 |
|---:|---|---|
| 1 | `276` / `" to"` | `8913` / `" Paris"` |
| 2 | `1114` / `"成"` | `13` / `"."` |
| 3 | `18` / `"3"` | `429` / `" The"` |
| 4 | `18` / `"3"` | `1094` / `" most"` |
| 5 | `18` / `"3"` | `9679` / `" famous"` |
| 6 | `18` / `"3"` | `44872` / `" landmark"` |
| 7 | `18` / `"3"` | `279` / `" in"` |
| 8 | `18` / `"3"` | `8913` / `" Paris"` |
| 9 | `18` / `"3"` | `317` / `" is"` |
| 10 | `18` / `"3"` | `254` / `" the"` |
| 11 | `18` / `"3"` | `427` / `" E"` |
| 12 | `18` / `"3"` | `96575` / `"iffel"` |

## Root-cause isolation

The following probes used the block-32 model, exact real prefill tensors produced
by ORT for the same prompt, and clean-main native CUDA:

1. **MLA Attention is not the semantic fault.** Isolated layer-0 standard
   `Attention` with Q/K width 192 and V width 128 matches ORT:
   output MAE `3.09e-6`, maximum error `2.44e-4`, cosine `1.0000001`;
   present K and V are bit-exact. All values are finite.

2. **QMoE packing/kernel is not the semantic fault on exact inputs.** Isolated
   layer-1 64-expert/top-6 QMoE matches ORT:
   MAE `1.45e-8`, maximum error `1.49e-7`, cosine `1.0000001`.
   Separately, checkpoint dequantization gave cosine `0.99670` for interleaved
   gate/up FC1 and `0.99674` for FC2, confirming exporter packing.

3. **The first demonstrated divergence is fp16 `MatMulNBits` prefill.**
   Isolated layer-0 `q_proj` on the exact ORT RMSNorm input differs from ORT:

   | Export | MAE | RMSE | Max error | Finite |
   |---|---:|---:|---:|---|
   | block-128 | `2.79e-4` | `5.78e-4` | `0.015625` | yes |
   | block-32 | `2.52e-4` | `5.37e-4` | `0.015625` | yes |

   The individual projection remains high-cosine, but DeepSeek's 26 routed
   layers make router top-k decisions discontinuous; repeated fp16 projection
   drift can change expert selection and cascade into completely different
   logits. Replacing either every QMoE output or every Attention output with
   ORT intermediates restores token 1 to `8913` (`" Paris"`), consistent with
   accumulated upstream projection drift rather than a QMoE packing or MLA
   latent-cache implementation error.

## Final verdict

❌ **Shipping-code semantic milestone does not pass for this prompt.**
Block-128 support itself is proven operational, both exports have zero EP and
graph fallbacks, and all inspected values are finite. However, both outputs are
systematic garbage from token 1. Dispatch the numerical investigation to the
native fp16 `MatMulNBits` prefill/reduction owner, with MoE router top-k
sensitivity as the amplification mechanism; do not dispatch QMoE packing or MLA.
