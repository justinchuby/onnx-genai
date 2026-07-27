
# Ferro — History

## 2026-07-18T04-55-00Z — Scribe session update

- Audited GLM IndexShare selected-token attention versus DeepSeek CSA; made no code changes and blocked implementation pending a frozen GLM private-op ABI, index semantics, parity/order, and mask/cache contract.

- 2026-07-27T10:09:19Z: Roadmap wave landed: #239/#246/#249/#248/#256/#263/#259 plus fmt gate #264; reviewer-lockout protocol enforced where required.

## 2026-07-27T13:12:20+00:00 — Roadmap wave-5

- Reviewed PR #266: rejected the naive ReduceLogSumExp overflow path, then approved Deckard's stable two-pass CUDA reduction and large-value parity regression.
