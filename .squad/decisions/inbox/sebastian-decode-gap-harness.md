# Decode-shaped benchmarking is now the default protocol for CPU scheduler work

**Author:** Sebastian (Performance Engineer)
**Context:** phase 21, `bench_decode_gap`

## Decision

Model-level CPU scheduler claims must come from `bench_decode_gap` with a
non-zero gap, not from `bench_generic`'s tight loop.

`bench_generic` remains correct for prefill and for throughput questions. It
is not correct for decode, because it measures the one regime decode never
occupies: with no gap between iterations the pool's workers never park, so
every dispatch lands on a spinning core. Its short default (7-10 runs) has the
opposite problem — it sits inside the pool warm-up transient, which on this
host runs to several hundred iterations at budget 16.

## Rules this implies

1. **Report a detected steady window, not a guessed `--warmups`.** Warm-up
   length scales with pool width; one constant cannot serve a 4-wide and a
   32-wide pool. If the harness says the series never settled, the run is too
   short — raise `--iters`, do not report the number.
2. **A/B arms run in separate processes.** The A/A null control measures a
   systematic ~8% advantage for whichever arm runs second within one process
   (warm allocator and caches). That is larger than most effects worth
   chasing.
3. **Quote the null control with any ratio.** On this host the band is
   0.81x-0.94x; per-cell readings inside it carry no information.
4. **Solo arms.** A co-resident ORT session depresses a native arm by up to
   4.8x (§38). `bench_decode_gap` drops the ORT session after the parity check
   so this cannot be forgotten.

## Non-decision

The adaptive spin window (`MIN_SPIN` 20us, `MAX_SPIN` 500us) is **left
unchanged**. Measured idle cost is ~6.5 CPU-ms per idle period at saturation,
but only +0.4 CPU-ms at a decode-shaped 100us gap — 0.4% of the cell's own
cost. Shortening it would trade real latency insurance for a saving that does
not exist in the regime decode occupies.
