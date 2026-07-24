# DeepSeek MLA capture-enablement — scoping & design (follow-up to the correctness fix)

**Author:** Rachael (worker) · 2026-07-24
**Depends on:** correctness fix `fix/deepseek-mla-kv-capacity` @ `621936f` (merged-pending)
**Status:** SCOPED — awaiting coordinator sequencing (do not merge with the correctness fix; separate branch e.g. `perf/deepseek-mla-capture-fixedcap` off current main `24531c4`).

## Goal
Make the default-domain `Attention` decode step CUDA-graph-**capturable**
(`captures>=1 replays=N fallbacks=0`) so DeepSeek MLA replays per-token like the
dense GQA models — target: lift DeepSeek from ~24 tok/s toward the dense range.
Today `StandardAttentionKernel::capture_support() == Unsupported`.

## Confirmed capture blockers (with line refs, standard_attention.rs @ 621936f)
1. **Host-derived, growing control arrays uploaded per step** (lines 971–1052):
   `offsets[b] = key_past_seq` and `pad_limits[b]` are computed **on the host**
   from the KV tensor extent (`key_past_seq`), then `htod`-uploaded each step
   (lines 1051–1052). `htod` is capture-illegal, and `key_past_seq` **grows every
   step**, so a captured graph would bake a stale past length → wrong output on
   replay. **Root blocker.**
2. **Growing tensor extent = changing shapes** (executor.rs:1122
   `kernel_input_uses_physical_capacity` is GQA-only): the default-domain
   Attention KV binding is exposed at **logical/growing** length, so the kernel's
   launch dims/shapes change every step — capture requires identical shapes per
   replay.
3. **Per-step `synchronize()`** at execute entry (line 717) and exit (line 1173)
   — illegal inside capture.
4. **The correctness copy-back uses synchronous `dtod`** (`cuMemcpyDtoD`,
   capture-illegal). NOTE: main `24531c4` now has stream-ordered
   `dtod_async`/copy_reshape (Gaff) — the copy-back must switch to the async,
   same-compute-stream variant under capture.
5. `dtoh` of `nonpad_kv_seqlen` (line 616) — NOT hit on the decode-with-past path
   (`nonpad` is mutually exclusive with `past_key`), so not a blocker for MLA
   decode, but must stay off the captured path.

## Proposed design (fixed-capacity KV + device-side valid length)
General for any default-domain Attention with a growing KV cache; must NOT touch
the working GroupQueryAttention path.

**A. Fixed-capacity KV binding.** Teach `kernel_input_uses_physical_capacity`
(executor.rs:1122) to recognize default-domain `Attention` KV inputs (4,5) as
physical-capacity (mirroring GQA inputs 3,4). KV is already physically allocated
at `max_len` (native_decode.rs `DecodeCudaState::new` ~1743–1772); this makes the
binding shape **stable across steps** (capacity, not logical length).

**B. Device-side valid-length scalar.** Provide the past/total valid length as a
**device scalar** (int32) that advances on the compute stream each step (mirrors
GQA's `seqlens_k`). `StandardAttentionKernel` computes `offsets`/`pad_limits`
**on-device** from that scalar instead of host `key_past_seq` + `htod`. Removes
blocker #1. Needs native_decode.rs to own/advance the scalar and executor
plumbing to pass it (or synthesize from GQA-style seqlens if present).

**C. Remove per-step host sync.** Drop the `synchronize()` at 717/1173 for the
capture path; rely on stream ordering (all work already on the EP compute
stream). The correctness argument is identical to the merged dtod-async change.

**D. Async copy-back.** Under capture, replace the synchronous `dtod` copy-back
(from the correctness fix) with the stream-ordered `dtod_async` on the compute
stream (now available on main). Keep the sync `dtod` for the eager path. Same
stream ⇒ implicitly ordered vs the producer `build_kv` ⇒ race-free AND
capture-legal. Preserves `dtod_waits_for_pending_stream_writes`.

**E. Flip `capture_support()` → Supported** once A–D land, and teach the
executor to treat the Attention KV/mask inputs as fixed-capacity for binding.

**F. build_kv at fixed capacity.** With capacity-strided KV, `build_kv` should
append the new token at the device-valid-length slot (like GQA) rather than
re-restriding the whole cache every step — this also removes the O(seq) per-step
restride (a perf win) and makes the correctness copy-back unnecessary on the
capture path (append is in-place-safe at a fixed slot). This is the larger part.

## Files touched (estimate)
- `crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs` — device-side
  offsets/pad_limits, remove sync, async copy / fixed-slot append, capture_support.
- `crates/onnx-runtime-session/src/executor.rs:1122` — recognize default-domain
  Attention KV as physical-capacity; plumb valid-length scalar.
- `crates/onnx-genai-engine/src/native_decode.rs` — own/advance the device
  valid-length scalar per decode step for the Attention path.

## Risk / effort
Medium-large, **coupled** rework (kernel + executor binding + engine state).
Higher regression risk than the correctness fix; must re-verify DeepSeek
determinism+coherence AND dense GQA non-regression + all 204 lib tests + new
capture test. Recommend landing as its own reviewed branch AFTER the correctness
fix merges, in stages: (F) fixed-slot append first (perf + simplifies), then
(A–E) capture enablement. **Correctness is already delivered independently and
must not wait on this.**

## Recommended owner
Engine (executor binding + native_decode scalar) + kernel (standard_attention).
Rachael can carry it once sequenced; needs coordinator go-ahead given the
executor/engine surface area beyond the isolated kernel fix.
