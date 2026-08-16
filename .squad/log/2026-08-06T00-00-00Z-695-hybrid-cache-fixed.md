# Session log — #695 hybrid cache fixed

- Timestamp: 2026-08-06T00:00:00Z
- PR: #700 (merged)
- Issue: #695 (closed)
- Follow-up: #701

PR #700 fixes wrong continuation logits for hybrid Mamba/attention decoders by disabling native host/device KV-mirror prefix reuse when recurrent state is present, forcing full recompute. Validation includes single-shot byte identity, an always-on gate unit test, and an env-gated GPU continuation regression proving reused argmax matches the fresh oracle token `33803`. Harry approved the PR and flagged a minor ORT paged-reuse residual; coordinator filed #701 for that follow-up.
