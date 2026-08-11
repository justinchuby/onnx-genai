# cohaagen — History

## Role
CUDA performance and kernel specialist. Owns CUDA EP op-coverage (#67), weight-offload (#63/#87), kernel tuning (o_proj split-K, CausalConvWithState, LinearAttention), and decode profiling.

_Entries before 2026-08-11 archived to `history-archive.md` (Scribe rounds 1–9 + 2026-08-12 compaction). Archived: PR #380 re-review; 7B perf; #63/#87 weight pager; #480/#484/#525 kernels; #529; #535/#544/#552; GQA capture; 27B roofline; fused LinearAttention; Foundry sweep; DeepSeek/GLM; Thread-3 hetero; 35B-A3B full unblock chain through #618/#625/#676/#684/#700/#708._

## 2026-08-11T21:10:00Z — Upstream audit: SM-count columns-per-CTA for M=1 MatMulNBits

- Confirmed upstream hardcodes `kColsPerThreadBlock = 8` with `grid.y = 1` in `matmul_4bits_m1_impl.cuh:135`.
- No SM-count adaptation exists in the M=1 or batched GEMV paths.
- PR #29469 (online tuning of small-M cap) is orthogonal — tunes *which* kernel, not grid geometry.
- No colliding in-flight work found across 30+ PRs and recent commits.
- **Verdict: genuine uncovered gap.** Contribution is small (~25 LOC), bit-identical, upstream-idiomatic.
- Caveat: +2.08% claim has no provenance; fresh benchmarks on ≥2 GPU generations required before PR.
- Wrote `.squad/decisions/inbox/cohaagen-matmulnbits-upstream-audit.md`.

## 2026-08-11 — MatMulNBits SM-adaptive grid PR shipped

- Implemented `SelectColsPerBlock(n, sm_count)` in upstream ORT.
- Templated M=1 kernel on `cols_per_block` (8/4/2).
- 10 files changed, ~225 insertions, 62 deletions.
- Opened draft PR: microsoft/onnxruntime#31988.
- No performance numbers published; benchmark methodology documented in PR body.
- Leak check passed (no persona names, no squad files in committed content).

## 2026-08-12T00:15:00Z — MatMulNBits upstream workstream recap

- Upstream audit confirmed genuine gap: `kColsPerThreadBlock = 8` hardcoded, no SM adaptation anywhere in M=1 or batched GEMV paths.
- Implementation shipped as draft PR #31988: `SelectColsPerBlock(n, sm_count)` → 8/4/2, templated kernel, multiProcessorCount threaded through 3 layers.
- No performance claims published; benchmark methodology documented in PR.
- Routing guard added by Chew under lockout: accepted-shape set preserved exactly (n%8==0 required).
- PR stays draft until GPU benchmarks on ≥2 GPU generations.
- CPU AMX QNBit prefill: no PR (host has no AMX/VNNI).
- Split-K excluded: 2-way K_SPLIT=2 regressed 7B o_proj GEMV by −0.59%.
