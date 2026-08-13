# Decision: decode correctness does NOT depend on eviction order (#888)

**Date:** 2026-08-13
**Author:** Copilot (squad/eviction-order-correctness)
**Refs:** #888 (investigation), #886 (rejected byte-aware residency), #864, #866, #750, #837

## Question

#886 rejected a byte-aware residency policy that corrupts decode output (16→3
tokens) whenever weight offload engages, and speculated the cause might be a
*latent, order-dependent defect in the shipped offload path* — i.e. that "the
scan-resistant recency discipline is load-bearing for correctness." If literally
true, that would block #864's hybrid (whose thesis is choosing a better resident
hot set than the driver picks blind).

## Finding — explanation (1): the rejected change is buggy, NOT the shipped path

A new default-OFF probe (`ONNX_GENAI_WEIGHT_OFFLOAD_EVICT_ORDER`) changes only
the eviction *victim* on the size-blind path, keeping the shipped always-bypass
decision. On qwen14b-zp, managed streaming, each run solo with a verified-clear
GPU:

| arm | tokens | byte_hit_rate | evictions | verdict |
|---|---|---|---|---|
| baseline (LRU, byte-aware OFF) | 16 ✓ `[96347…752]` | 70.16% | 2672 | reference |
| **evict order = MRU** (reverse recency) | **16 ✓ identical** | 79.15% | 1952 | clean |
| **evict order = Smallest** (byte-aware's exact victim) | **16 ✓ identical** | 71.77% | 10192 | clean |
| byte-aware ON (graph ON) | 3 ✗ early EOS | — | — | corrupt |
| byte-aware ON (graph OFF) | 3 ✗ early EOS | — | — | corrupt |

Two independent, still-correct eviction *orders* — including byte-aware's exact
smallest-first victim under extreme churn (10,192 evictions) — are
**byte-identical** with clean ledgers. **Changing eviction order alone is
value-neutral.** The corruption is caused solely by byte-aware's *other* change:
the **retain-vs-bypass flip** — promoting a large tensor the shipped path streams
transiently into a *retained, stable-slot resident* that is then served as a
**hit** (no re-fill) across steps.

## Mechanism class (what it is NOT)

- **Not** captured-VA baking — graph-OFF corrupts too.
- **Not** a copy/compute fence hazard — a full drain of both streams before every
  page-in fill (`ONNX_GENAI_WEIGHT_OFFLOAD_SYNC_BEFORE_FILL=1`) does **not** fix it.
- One concrete consistency bug was found & confirmed to occur (a *slotted* key
  that later bypasses gets `stable_slot=true` but never rejoins `pages`;
  `ONNX_GENAI_WEIGHT_OFFLOAD_RETAIN_SLOTTED=1` closes it) — but closing it does
  **not** stop the corruption. So it is real-but-secondary; the primary
  value-corruption path is deeper in retaining/re-admitting large stable-slot
  tensors (granule-level checksums across steps are the remaining decider).

## Consequence for #864 / #866 / #750

The shipped size-blind path is **safe** — it never retains large tensors, so the
buggy path is unreachable (the #888 probe confirms it: eviction reorder + heavy
churn stays byte-identical). Therefore:

- **#864 hybrid is NOT blocked by an eviction-order invariant.** A hybrid that
  pins a **static** hot set (retain the chosen weights once, never evict/re-admit
  them; zero-copy the cold remainder — exactly #864's stated shape) does not
  exercise the corrupting retain-then-churn path.
- The hazard is specifically **evicting and re-admitting large stable-slot
  residents**. Any dynamic scheme that moves large weight residents in/out —
  byte-aware, and potentially #866 elastic reclaim or #750 admission if they
  churn large weight pages — must validate token identity, and should prefer a
  pinned non-churning hot set over a churning residency reorder.

## Status

Investigation only. Byte-aware stays rejected/default-OFF. No shipped behaviour
changed; all knobs added are default-OFF and byte-identical on the default path.
PR opened against #888 (do not close).
