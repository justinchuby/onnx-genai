# Decision: Multi-turn and batch benchmarks reveal structural native deficit

**Date:** 2026-07-28  
**Author:** Pris  
**Status:** Active  
**Affects:** Iran, Deckard (native backend architecture)

## Context

PR #351 established native's cold-start advantage (2.47–4.63× faster process
start → first token). Justin flagged that real usage is multi-turn: load once,
prefill every turn. We needed to know whether ORT's pre-packing cost amortises
across a session — and if so, what the fix is.

## Findings

### Multi-turn LLM

| Model | Break-even turn | ORT overall advantage (10 turns) |
|---|---|---|
| TinyStories-33M (f32) | 3 | 2.0× |
| Qwen2.5-0.5B (f16) | 5–8 | 1.2× |

**Root cause: NOT pre-packing amortisation.** The native backend has no
session-persistent KV cache. Each turn re-prefills the entire conversation
(O(context_length)). ORT's session API preserves KV, so each turn prefills only
new tokens (O(new_tokens)). At turn 10, native TTFT is 6–8× its turn-1 value
while ORT TTFT stays flat.

### Steady-state per-prefill (turns 3–10)

| Model | Native TTFT ms | ORT TTFT ms | Ratio |
|---|---|---|---|
| TinyStories-33M | 93.4 | 29.4 | 3.2× ORT faster |
| Qwen2.5-0.5B-f16 | 519 | 169 | 3.1× ORT faster |

### Batch vision (MobileNetV2)

- Batch=1: native 0.43× ORT (12 ms vs 5 ms)
- Batch>1: **native crashes (segfault)** — correctness bug
- ORT scales 1.9× from batch=1→16

### Cache survival (PR #353)

Weight transpose caches ARE correctly reused across turns:
- Qwen f16: 168 entries at load, stable across all turns
- TinyStories f32: lazily fills to 25 entries, then stable

This is NOT the cause of the deficit.

## Should we pre-pack?

**No — pre-packing would not address the dominant issue.**

The multi-turn deficit is caused by the absence of persistent KV, not by
slower per-token computation. Pre-packing could narrow the per-prefill gap at
equal context length (estimated 1.5–2× improvement), but it cannot eliminate
the O(context_length) vs O(new_tokens) structural disadvantage.

If persistent KV sessions are added to the native backend, THEN pre-packing
should be revisited to close any remaining per-prefill gap. The load-time cost
of pre-packing (estimated +200–400 ms for Qwen-0.5B-f16, based on ORT's 1.8 s
vs native's 340 ms load) would be acceptable for long-lived servers but
unacceptable for cold-start use cases — an opt-in mode would be needed.

## Decisions for Iran/Deckard

1. **Session-persistent KV for native backend** is the #1 priority for
   multi-turn competitiveness. Without it, no kernel optimization can close
   the gap beyond 3 turns.
2. **Batch>1 vision segfault** is a correctness bug that should be filed and
   fixed before any batch benchmark claims.
3. **Pre-packing** should be deferred until after persistent KV lands, then
   re-evaluated.

## Published conclusion changes

The PR #351 cold-start advantage **remains valid for one-shot use**. For
multi-turn sessions (≥3 turns on small models, ≥5–8 on large), ORT is
cumulatively faster. This is now documented in `examples/profiles/README.md`
with the multi-turn framing section.
