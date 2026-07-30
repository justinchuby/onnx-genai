# 2026-07-29 — ARM64 MLAS QNBit decode uses SPMD sharding

By: Resch

## Decision

For non-Apple ARM64 Qwen-style `MatMulNBits` decode (bits4/bits8, block-128,
accuracy-4), MLAS QNBit is now default when the `mlas` feature is enabled.
`ONNX_GENAI_CPU_MM_MLAS_QNBIT=0` remains the A/B escape hatch back to the native
KAI fallback.

The MLAS QNBit decode hot path pre-packs one N-shard per persistent SPMD worker
and runs each shard with `multithread=false`, instead of issuing one full-width
MLAS `multithread=true` call per node through the Rayon-backed MLAS parallel-for.
`ONNX_GENAI_CPU_MM_MLAS_NO_SHARD=1` remains the full-width A/B/parity hatch.

## Root cause

The 2.1x gap was not the KleidiAI ukernel. It was our full-width MLAS threading
integration paying an MLAS/Rayon parallel-for dispatch for each of the 197
`MatMulNBits` nodes per decode token. On qwen3-0.6b, the same MLAS/KleidiAI
kernel dropped from ~102 tok/s class when driven by ORT or SPMD sharding to
~57 tok/s through our full-width Rayon hook.

## Measurement

Host was still somewhat variable/contended; use same-window relative numbers.
Command shape: `profile_native --model ...qwen3-0.6b-generic-cpu-4/v4 --ep cpu
--steady --warmups 1 --runs 1 --tokens 128`, interleaved 3 passes.

Median decode throughput:

- Native KAI fallback (`ONNX_GENAI_CPU_MM_MLAS_QNBIT=0`): 97.9 tok/s
- Native MLAS QNBit default (SPMD-sharded): 104.2 tok/s
- ORT CPU: 107.5 tok/s

The old full-width MLAS path measured ~56.9 tok/s in the same session, so the
SPMD-sharded MLAS route closes essentially all of the threadpool gap and lands
within ~3% of ORT while beating the native KAI fallback by ~6%.

Per-op profiling confirms the mechanism: full-width MLAS QNBit MatMulNBits was
typically ~16-17 ms/forward for 197 calls; SPMD-sharded MLAS is typically
~7-9 ms/forward for the same 197 calls.
