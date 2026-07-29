# Benny — History

## 2026-07-29T03:45:00+0000 — PR #382 merged

- Continued #377 in PR #382, merged `85b9ba15`: ORT shared-buffer/static-cache decode adapters now consume declared KV pairs rather than exporter names.
- Repaired the latent #380 regression where `BatchedSharedBufferDecodeSession::new` received `None` and always failed.
- Lori required a CPU lock; Leon supplied the reassigned regression test under reviewer lockout.
