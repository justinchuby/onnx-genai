# S11 runtime state tiering

- Physical residency, page budgets, restore-cost thresholds, and isolation scope
  remain runtime configuration; inference metadata is unchanged.
- Page migration is transactional: capability preflight, allocation, and a
  complete physical copy finish before the authoritative store is replaced.
- Spillability and recomputability are independent inputs. Spillability reflects
  lossless backend support; recomputability reflects semantic legality.
- Reusable KV is session-private by default. Cross-session reuse requires an
  explicit opaque shared-domain configuration, which is part of every cache key.
- This slice claims the existing host-backed GPU/CPU residency abstraction and
  lossless F32 disk payload store. It does not claim native device or generic
  disk residency for arbitrary runtime values.
