# Session consolidation — 2026-07-30

Consolidated the match-ORT-threadpool-tuning arc. Deckard reproduced ORT-style dynamic block claiming and proved it wins the isolated full-width QNBit microbench. Resch proved the same direction loses end-to-end Qwen3 decode because pool park/wake variance across many small ops dominates, then reverted the live full-width toggle and fixed ARM64 route-test hygiene. The durable result is that native static-SPMD now matches or slightly beats ORT on best-case and p90 throughput, while median remains variance-limited on the contended host.
