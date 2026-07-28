# Decision: Republish profile figures with load context (2026-07-28)

**By:** Pris (test/bench engineer)
**Context:** PRs #342, #347, #349, #351, #352, #353, #354 merged since last profile publish

## What changed

The previously published headline (1.33× decode, 1.34× end-to-end for native FP16 vs ORT)
was measured at host load 4–5. Independent re-measurement at load 2.5–3.7 shows:

- **qwen2.5-0.5b-f16**: 1.72× decode, 1.68× end-to-end, 5.01× process-start→first-token
- **TinyStories-33M FP32**: 0.91× decode (ORT wins), 0.82× end-to-end (ORT wins), 2.47× cold-start

## Verification

Justin measured at load 2.5–2.9 (3 runs qwen, 5 runs tiny). Pris independently measured
at load 2.5–3.7 (3 runs qwen, 5 runs tiny) on the same machine, same commit (679837e1).

| Model | Metric | Justin | Pris | Agreement |
|---|---|---:|---:|---|
| qwen f16 | decode tok/s | 74.03 | 73.02 | ✅ <2% |
| qwen f16 | ORT decode tok/s | 42.68 | 42.56 | ✅ <1% |
| tiny-33m | decode tok/s | 297.5 | 297.4 | ✅ <1% |
| tiny-33m | ORT decode tok/s | 324.4 | 327.0 | ✅ <1% |

## Standing rule applied

Every number in the updated README carries its host load. The load sensitivity section
remains — readers can see that at load 5+, native decode drops to ~45 tok/s (1.11× ORT)
while at load 2.5 it reaches 73 tok/s (1.72× ORT). ORT is stable at 40–43 regardless.

## Regressions

No regressions detected. All metrics improved or held steady relative to the prior publish:
- Decode went from 53.1 tok/s (load 4–5) to 73.0 tok/s (load 3.7) — improvement is load, not code
- ORT baseline is stable (39.9–42.6 tok/s across all measurements)
- TinyStories-33M deficit (0.91×) is unchanged — this is a genuine gap, not a regression

## Unflattering numbers preserved

- TinyStories decode: 0.91× (ORT wins)
- TinyStories end-to-end: 0.82× (ORT wins)
- TinyStories TTFT: 2.86× slower than ORT
- Roofline% marked NOT BINDING for cache-resident models per #354
