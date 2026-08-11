# Harry — History

## Historical context
Older detailed dated entries through 2026-08-04T00:40:00Z — PR #625 loader review were moved to `history-archive.md` during Scribe compaction on 2026-08-11T03:25:00Z. Keep this live file focused on current routing-relevant context; full chronology is preserved in the archive.

## 2026-08-06T00:00:00Z — PR #676 review

- Harry picked up review after gpt-5.6-sol canary-looped, approved #676 with two nits only, and the coordinator squash-merged it.
- Review conclusion: native 3-D router-probs fix and token-119 oracle regression are sound; token-119 proves QMoE is more accurate than dense int4 for that step rather than exposing a native kernel bug.

## 2026-08-06T00:00:00Z — PR #700 review

- Approved #700 after verifying the recurrent-state gate disables host/device KV-mirror reuse for hybrid decoders without regressing single-shot behavior.
- Confirmed the env-gated GPU continuation regression compares reused continuation argmax against the fresh oracle (`33803`).
- Flagged a minor residual on the ORT paged-reuse path (`kv_bridge.rs:407`); coordinator filed #701.

## 2026-08-06T12:30:27Z — Reviews #684/#692 and cache-reuse bug

- Approved PR #684 after verifying the parallel `qmoe_route` top-k reduction is byte-exact with the old serial total-order scan and leaves softmax aggregation in original order.
- Found the pre-existing #676 oracle-test defect: reused-engine teacher forcing in the hybrid-Mamba model produced argmax 279 instead of oracle token 33803.
- Approved PR #692 and independently confirmed the underlying prefix-cache-reuse engine bug; issue #695 now tracks missing Mamba conv/recurrent state restoration.

## 2026-08-06T19:40:00Z — PR #708 review

- Approved #708 after verifying capture-safe Split sizes are statically inferred and byte-equivalent to the old host-read path; left nits for dead fallback docs and redundant re-locking.

## 2026-08-11T03:25:00Z — Reviewing GLM/DeepSeek native fixes

- Dispatched to review PR #770 and PR #771.
- Focus areas: KV precedence soundness for GLM native CUDA capacity, and Cast-backed QMoE scale placement dtype/region correctness for DeepSeek.
- Current status is running; no review decision recorded yet.
