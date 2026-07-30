# Field Meaning Audit — what each telemetry field actually counts

**Every other safeguard in this project answers "is this number computed or fabricated?"
This document answers the different question: "does this number mean what its name says?"**

Seven separate defects tonight turned on the second question, and not one of them would
have been caught by the first. A hardcoded `0.0` is inert and greppable. A real,
correctly-computed value under a name describing a different quantity is live — it moves
when you exercise the feature, which is the strongest confirmation signal a developer can
get, and here it is precisely backwards. It carries `state: 'measured'`, `source: 'server'`,
a live endpoint and a plausible bounded value, so **the entire provenance apparatus
certifies it as healthy.**

## Rules for maintaining this file

1. **No line numbers anywhere in this document. Cite the file and the symbol.**
   Roughly forty moves of `HEAD` tonight rotted every line number quoted in a broadcast,
   several within minutes. Not one symbol name decayed. An audit document that rots is
   worse than none, because it will be trusted.
2. **The "What it actually counts" column states the quantity in plain language, never
   the field's own name restated.** "HTTP generations in flight", not "the current batch
   size". If the column can be produced by prettifying the field name, it is not doing
   its job.
3. **Re-verify existing rows before adding new ones.** Three provenance entries were found
   declaring now-real measurements to be hardcoded zeros — an honesty mechanism that
   expires the moment the code it indicts improves. This file is the same class of
   artefact and will decay the same way.

## The table

All rows below were re-verified against source at `feat/genai-demo-dashboard`,
2026-07-30 01:50 local.

| Field | Name implies | What it actually counts | Source of truth | Safe to render? |
|---|---|---|---|---|
| `prefix_cache_lookups` | cache lookups | **Completed generations.** Incremented unconditionally once per generation. | `metrics.rs` → `result` | ❌ never as "lookups" |
| `prefix_cache_hits` | blocks/tokens reused | **Generations that got any nonzero prefix overlap.** The increment is binary — a generation overlapping twelve blocks counts identically to one overlapping a single token. | `metrics.rs` → `prefix_reuse_increments`, which returns `(1, len)` for any `len > 0` | ❌ never as a hit count |
| `prefix_cache_hit_rate` | cache efficiency | **Fraction of generations that got any prefix overlap.** A real and useful quantity that is not a cache hit rate. Its own Prometheus HELP string calls it a hit ratio, so a reader who diligently checks the documentation is confirmed in the wrong belief. | `metrics.rs` gauge emission; ratio assembled in `routes/admin.rs` | ❌ forbidden as a displayed value |
| `prefix_cache_hit_len` | prefill work skipped | **Longest token overlap with any cached prompt.** On the branch both server entry points actually take, nothing is restored and no prefill is skipped. | `engine/runtime.rs` → `prepare_session_prefix`, token-prefix branch | ❌ |
| `batch_size_current` | engine batch rows | **In-flight HTTP generation requests.** Observed reading 8 while the real engine batch was 4. | `metrics.rs` → `REGISTRY.batch_size`, emitted as `onnx_genai_batch_size_current` | ⚠️ only as "requests in flight" |
| `batch_utilization` | engine occupancy | In-flight HTTP count ÷ `effective_batch_capacity`. The numerator is the mislabelled HTTP count above; the clamp to `1.0` then conceals the over-count and renders it as a confident "full". **The only case tonight where a correct safety mechanism destroys the evidence it sits on top of.** | `state.rs` → `effective_batch_capacity`, which is `min(max_batch, max_queue_depth)` | ⚠️ absolute count preferred |
| `ttft` | time to first token | **Time from request admission**, not from generation start — it includes queue time. Shares one `started` instant with the end-to-end timer. | `metrics.rs` → the request guard's `started` field, observed via `elapsed()` | ⚠️ label as including queue |
| `vram.used` | GPU device memory | **KV byte-budget accounting only.** Never device memory. | KV governor byte budget | ⚠️ label "KV budget in use" |
| `host_ram.used` | demo memory | **Whole machine**, including the browser, the editor and the second server. | host sampler | ⚠️ never attribute to the demo |
| `active_sessions` | concurrent requests | **Persistent `X-Session-Id` sessions.** Four concurrent requests with no session header display 0 while four token streams visibly interleave on screen. | session registry | ⚠️ Scenario B only |
| `decode_backend` | execution mode | **Decoder runtime** (`Auto`/`Ort`/`Native`) — orthogonal to static-vs-dynamic cache. The mode discriminator is `continuous_batch_supported`. | `config.rs` → `EngineDecodeBackend`; mode fork in `driver.rs` | ❌ never as the mode |
| `allocation_failures` | pool exhaustion | **Structurally pinned at zero.** The pool grows by demoting to the cold tier rather than failing, so this can never move. A panel keyed on it shows a pool under heavy pressure as perfectly healthy. Use `hot_evictions` as the pool-full signal. | `routes/mod.rs` → the pressure-signal docblock on `hot_evictions` / `allocation_failures` | ❌ reads as good news |
| `pages_in_use` | pages ≤ capacity | **May legitimately exceed `hot_capacity`**, because eviction demotes a page without dropping its reference. **This gauge must not be clamped** — deliberately the opposite of `batch_utilization`. Both facts are recorded here so nobody "harmonises" them later. | `routes/mod.rs` → KV block window fields | ✅ unclamped only |

## Two properties worth preserving

**`defined_ratio` returns `None`, not `0.0`, for an undefined ratio.** Its own comment
states the reason: `0/0` is not `0.0`, and emitting zero for an undefined ratio reports
"we have never measured this" as "we measured this and it was the worst possible value" —
indistinguishable downstream, because a genuine `0.0` is a legitimate reading.

**`NaN` is not neutral in this wire format.** It serialises to JSON `null`, and `null`
means `unavailable` under the ratified contract — so an arithmetic edge case would
silently masquerade as a missing measurement. Check your float paths.

## Why suspicion is the wrong instrument

`0 hits / 135 lookups` was implausible, so three agents investigated it. `19/20 = 95%`
looked exactly like a working cache, so it sailed through — and it would have been the
demo's proudest number.

**Suspicion tracks implausibility, not falsehood.** Attention is drawn by numbers that
look wrong, and a plausible lie is by definition the one that does not. The fields to
audit hardest are precisely the ones the team currently feels best about.

## Open work

This table should be mechanically checked rather than maintained by hand: every
"what it actually counts" claim that quotes source should fail loudly when that source
changes. That belongs in the existing merged citation harness — **do not build a second
harness.** Coordinate with the harness owner before adding checks here.
