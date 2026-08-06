# Challenger — History

Append-only. Each entry records a claim challenged, the verdict, and what the
challenge changed.

## 2026-08-06 — Hired

Requested by @justinchuby:

> "hire一个新人 叫挑战者 职责是challenge不符合常识或者直觉的claim，每次观察得到的重要、影响技术方向的结果，都让它想想是不是哪里有疏漏"

Created after a session in which several direction-setting measurements turned
out to be narrower or wronger than first reported, and were only caught because
someone happened to ask a second question:

- A weight-prefetch A/B that measured demand fallback against itself, because
  the prefetch guard had silently declined every opportunity (#673).
- A VMM arena that committed ~800 MB less and still needed a *higher* VRAM
  limit — granularity waste, not noise (#682).
- A lookahead-depth sweep whose best median sat inside the baseline's range, so
  no win was established at any depth (#673).
- A KV on-demand-commit test that would have passed unchanged if the feature had
  silently fallen back to eager commit (#682 review).

The common shape: a result was accepted without asking what *else* could produce
it. Challenger's remit is that question.
