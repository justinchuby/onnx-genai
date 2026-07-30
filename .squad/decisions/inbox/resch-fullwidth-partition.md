## 2026-07-30 — MLAS SQNBit full-width dynamic partition is not the default

Context: Deckard wired ORT-style dynamic block claiming into `mlas-sys`. I briefly
added a CPU EP toggle `ONNX_GENAI_CPU_MM_MLAS_FULLWIDTH=1` so constant
`MatMulNBits` weights could bypass the static SPMD N-shard path and run a single
full-width `sqnbit_gemm_into_with_workspace(..., multithread=true)` call through
the cached full-width MLAS pack/workspace.

Measurement host was `aarch64-pc-windows-msvc` on a contended Windows machine;
`cargo check -p onnx-runtime-ep-cpu --features mlas --target
x86_64-pc-windows-msvc` passes, but these are not x86 throughput numbers.

Qwen3 0.6B CPU int4 steady decode (`--tokens 96 --runs 15`, decode skip 8):

| path | tok/s best | tok/s p90 | tok/s median | MatMulNBits ms/token |
|---|---:|---:|---:|---:|
| native static-SPMD | 110.32 | 109.76 | 97.61 | 8.93 |
| native full-width dynamic | 91.72 | 91.64 | 81.65 | 8.29 |
| ORT CPU | 106.16 | 106.03 | 104.89 | n/a here |

Correctness: full-width and static native emitted identical 96 generated token
ids in the decode check. ORT emitted a different sequence on this model/backend.

Verdict: do **not** make full-width dynamic the default. The MLAS-only portion
improved on one profiled run (8.93 -> 8.29 ms/token) but end-to-end steady
decode lost badly and 15-run full-width processes intermittently hung/stalled
after several measured runs.

Follow-up: the live `ONNX_GENAI_CPU_MM_MLAS_FULLWIDTH` product/test toggle was
reverted. Its process-global `OnceLock` env cache polluted in-process ARM64
tests that also mutate MatMulNBits routing env vars, making the suite
order-dependent. Keep the negative result as the artifact; keep the product on
the previously-green static-SPMD default and use isolated subprocess harnesses
for any future full-width A/B experiments.

Test hygiene follow-up: ARM64 QNBit/KAI route tests now avoid using shared hit
counters to infer the path taken after execution; they inspect the kernel's own
cache instead, so concurrent tests cannot make an MLAS run look like a KAI run.
The work-stealing parity child explicitly requests two steal tiles per worker
when it asserts extra dynamic segments; the production default remains one tile
per worker.
