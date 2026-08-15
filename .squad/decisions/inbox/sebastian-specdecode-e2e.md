# Decision drop — Stage-4 speculative decode with CAPTURED fused verify (native CUDA EP)

- **Author:** Sebastian (Performance Engineer, CUDA & Perf)
- **Date:** 2026-08-15
- **Branch:** `squad/spec-decode-e2e` — head `a3fbd41c`
- **PR:** draft (references #957 Stage 4)
- **GPU:** single idle H200, `CUDA_VISIBLE_DEVICES=7` (re-checked `nvidia-smi` idle before every run); `source .cudaenv.sh`
- **Feature gate:** `ONNX_GENAI_SPEC_CAPTURED_VERIFY=1` **and** `ONNX_GENAI_MARLIN_M_GT_1=1` (Marlin M>1 makes the M=width verify a single capturable graph). OFF by default → zero risk to the shipped M=1 decode path.

## TL;DR
The captured fused verify path is **built, correct (byte-identical to plain greedy on glm AND qwen), and delivers a real multiplicative decode win on high-acceptance prompts** — up to **215 tok/s on glm-4-9b-int4 (1.93× the 111 tok/s non-speculative baseline)**. It does **not** yet beat stock ORT (~250 tok/s) even on favorable prompts, and it **regresses on generic prose** (67 tok/s). Both the ORT gap and the generic-prompt regression trace to a single root cause: **the session has ONE device-graph slot**, so every lookup *miss* must invalidate + re-warm the width-W verify graph (running an eager forward while a graph is installed corrupts device memory — `CUDA_ERROR_ILLEGAL_ADDRESS`). A **dedicated second graph slot** (EP work) removes the re-warm penalty and is the scoped follow-on that would let realistic prompts win and clear ORT.

## Design (what shipped)
Fused, prompt-lookup-only, steady-state-only path in `native_speculative.rs::generate`:
1. **Propose before the forward** (prompt-lookup needs only context, not base logits).
2. **One fused `[1, 1+draft_width]` captured forward** over `[bonus ⊕ draft]`, padded to a **constant width** so a single constant-signature CUDA graph replays across steps: row 0 = base next-token distribution, rows `1..=k` = verify rows (numerically identical to the eager base+verify forwards).
3. **Rewind to the accepted prefix** with `retain_graph_on_rewind=true` so the graph survives the per-step rewind and *replays* instead of re-capturing.
4. **Empty-draft (lookup miss) → cheap M=1 eager base forward** (`decode_base_eager_keep_verify_graph`) rather than paying the ~2.2× width-W cost to commit a single token. This invalidates the graph (eager-alongside-installed-graph is unsafe), so the next hit re-warms.
5. **Numerically-safe near-tie acceptance guard** (`greedy_accept`/`TieGuard`, from the earlier layer) — accepted stream is byte-identical to plain greedy; `near_tie_rejections` observes acceptance (not correctness) lost to tie-prone logits.

## BEFORE/AFTER measurements — glm-4-9b-int4 (H200 GPU7, `--steady --tokens 200 --decode-skip 40 --runs 3 --warmups 1`, `ONNX_GENAI_MARLIN_M_GT_1=1`)

| Prompt regime | Mode | K | mean-accepted B | acceptance | **decode tok/s** | vs non-spec |
|---|---|---|---|---|---|---|
| — | non-spec (plain M=1) | — | — | — | **111.2** | 1.00× |
| Verbatim copy | captured-spec | 4 | 4.87 | 98.1% | 157.7 | 1.42× |
| Verbatim copy | captured-spec | 8 | 7.92 | 89.7% | **205.9** | **1.85×** |
| Near-pure repetition | captured-spec | 8 | 6.70 | 72.8% | **215.4** | **1.93×** |
| Generic prose | captured-spec | 8 | 3.40 | 30.0% | 66.8 | 0.60× (regress) |
| — | stock ORT-genai (prior session) | — | — | — | ~250.3 | — |

## qwen2.5-14b-int4 control (block-32) — correctness + no-regression
| Mode | K | B | acceptance | tok/s | tokens vs non-spec |
|---|---|---|---|---|---|
| non-spec | — | — | — | 126.7 | — |
| captured-spec | 8 | 5.00 | 100% | 130.6 | **byte-identical ✓** |

qwen's M>1 forward is costlier relative to M=1 (14B, block-32), so even at 100% acceptance it is ~flat (no regression). Marlin M>1 correctness holds across both block sizes.

## Correctness (non-negotiable gate) — PASS
- **glm** captured-spec token stream **== non-spec** on a mixed/generic prompt (heavy empty-draft interleave): byte-identical.
- **qwen** captured-spec token stream **== non-spec** on a copy prompt (100% acceptance): byte-identical.
- CUDA smoke test `native_prompt_lookup_matches_plain_greedy_cuda` (with `ONNX_GENAI_SPEC_CAPTURED_VERIFY=1`) — **green**.
- 6 GPU-free `greedy_accept`/tie-guard unit tests — **green**.
- `cargo fmt --all -- --check` clean; `clippy -p onnx-genai-engine --features cuda,native-backend --all-targets -D warnings` clean **except 2 pre-existing `platform_capacity.rs` `u64→u64` casts unrelated to this change** (last touched #947/#952).

## The #1 lever to go PAST ORT (recommendation)
A **second dedicated device-graph slot** in the session/EP so the M=1 base decode graph and the M=width verify graph can BOTH stay installed and be replayed independently. Today the shared single slot forces an invalidate + 2-forward re-warm on every miss→hit transition, which (a) caps the favorable-prompt win below ORT and (b) turns low-acceptance prompts into a regression. With no re-warm, the theoretical ceiling at 100% acceptance is ~`111 × W / B*` ≈ `111 × 9 / 2.2 ≈ 450` tok/s for K=8 — comfortably past ORT's 250. This is measurement-backed and is the logical next increment.

Interim guidance: keep the feature **opt-in** (env-gated OFF). It is a clear win for high-acceptance / repetitive / copy / code-continuation workloads and must not be enabled globally until the second-slot work removes the generic-prompt regression.

## Reproduce (exact commands)
```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
export CUDA_VISIBLE_DEVICES=7   # re-check nvidia-smi idle first
cd .worktrees/sebastian-specdecode-e2e
cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native
GLM=/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda

# non-spec baseline
ONNX_GENAI_MARLIN_M_GT_1=1 ./target/release/profile_native --model "$GLM" --ep cuda \
  --steady --tokens 200 --decode-skip 40 --runs 3 --warmups 1 --speculative none --prompt "<prompt>"

# captured speculative (favorable prompt, K=8)
ONNX_GENAI_MARLIN_M_GT_1=1 ONNX_GENAI_SPEC_CAPTURED_VERIFY=1 ./target/release/profile_native \
  --model "$GLM" --ep cuda --steady --tokens 200 --decode-skip 40 --runs 3 --warmups 1 \
  --speculative prompt-lookup --spec-ngram 3 --spec-tokens 8 --prompt "<repetitive/copy prompt>"

# correctness: token-identity vs non-spec + CUDA smoke
ONNX_GENAI_RUN_CUDA_SMOKE=1 ONNX_GENAI_MARLIN_M_GT_1=1 ONNX_GENAI_SPEC_CAPTURED_VERIFY=1 \
  cargo test -p onnx-genai-engine --features cuda,native-backend \
  --test native_speculative_driver native_prompt_lookup_matches_plain_greedy_cuda -- --nocapture
```
`ONNX_GENAI_SPEC_DEBUG=1` prints the per-step capture state machine (`phase`, `captures`, `replays`, `fallback`) for verification.
