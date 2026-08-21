### 2026-08-21: MTP approach-B (skip re-advance on full accept) + device-to-device recurrent snapshot — 20.68 → 34.34 tok/s, token-identical, both slots replay

**By:** Gaff

**What:**
Two committed, GPU-validated, token-identical speedups on the real Qwen3.8-27B
int4 hybrid (GDN+GQA) MTP self-spec decode (k=1, 83.3% accept, 2.67 tok/step,
128-token run on an idle H200, median-of-5). Branch `squad/mtp-approach-b` off
origin/main `66e19c2b5`.

1. **approach-B — eliminate the redundant re-advance (commit `6c7f71650`).**
   When ALL draft tokens are accepted (~67% of steps at 83.3% accept, k=1), the
   eager M=K verify forward has ALREADY advanced KV + GDN/conv recurrent state to
   exactly `base+accepted` (== committed length). The snapshot→restore→re-advance
   commit is then pure redundancy, so it is skipped; only partial-accept keeps
   the snapshot→rewind→restore→re-advance path (intermediate per-position
   recurrent states are not exposed by the GDN kernel, so partial accept still
   rebuilds by re-running the accepted prefix). Result: **20.68 → 26.73 tok/s
   (+29%)**; Primary graph replays 504 → 248 (64 redundant re-advance M=1
   forwards/generation removed).

2. **device-to-device recurrent snapshot (commit `2f48f801c`).** The remaining
   snapshot copied every fixed-state (GDN SSM + conv1d) binding device→HOST every
   spec step via `read_bytes_range` (~10ms/step PCIe round-trip), unused on the
   full-accept steps. Replaced with a stream-ordered **device→device** snapshot
   into a per-binding device scratch arena:
   - `ExecutionProvider::copy_device_to_device` (default errors; CUDA overrides
     with `cuMemcpyDtoDAsync` on the EP stream → snapshot is ordered ahead of the
     verify forward that overwrites the state, no host sync).
   - `DeviceIoBinding::snapshot_device_into` / `restore_device_from` + an
     `allocator()` accessor; `DeviceBuffer` re-exported from onnx-runtime-session.
   - `DecodeCudaState.fixed_state_snapshot_scratch` allocated lazily and sized
     from `fixed_state_binding_range` (no hardcoded layer/dim numbers);
     `snapshot_fixed_states_device` / `restore_fixed_states_device` replace the
     host-bytes variants. `RecurrentStateSnapshot` carries a `device_scratch`
     flag; the CPU host-past path is unchanged. Result: **26.73 → 34.34 tok/s**.

Also fixed a pre-existing red on main (commit `c799a835b`): `InferenceSession`
gained `verify_exec` in #1658 but the two `executor/tests.rs` initializers were
never updated, breaking `cargo test -p onnx-runtime-session --features cuda`.

**Per-step target-forward budget (128-tok run, 48 verify steps):**
- Before approach-B: 48 base(M=1) + 80 re-advance(M=1) + 48 verify(M=2)  [Primary replays=504].
- After approach-B:  48 base(M=1) + 16 re-advance(M=1) + 48 verify(M=2)  [Primary replays=248].
- D2D snapshot does not change the forward count; it removes the per-step PCIe
  D2H transfer (state staging) that was the dominant remaining wall-clock cost.

**Validation (all green):**
- Both slots replay, fallbacks=0: `cuda_graph: captures=4 replays=248 fallbacks=0 invalidations=3`;
  `cuda_graph_verify: captures=4 replays=184 fallbacks=0 invalidations=0`.
- MTP token-identical to baseline: `generated_token_ids` md5 `be7ed565` across
  median-of-5, no NaN (finite-guard clean).
- Engine lib `--features native-backend` **579 passed / 0 failed** (greedy inert
  — the snapshot machinery only runs under spec-decode with recurrent state).
- ep-cuda `graph::tests` **8/8**; session `--features cuda` green (191 + suites).
- Median-of-5 A/B (H200, `CUDA_VISIBLE_DEVICES=5`, batch=1, tokens=128, warmups=3,
  int4 block-32, ORT 1.28 cuda13, `--release --features bench-native,native-cuda,cuda-13000`,
  `profile_native --steady`): **34.34 tok/s** (34.29–34.36, very tight).
  Progression: 20.68 (#1658) → 26.73 (approach-B) → 34.34 (D2D snapshot). Total +66%.

**Why:**
On a launch-bound decode, MTP only beats greedy if per accepted step it costs
fewer target forwards than tokens produced. approach-B removes lot (c) — the
re-advance — on the 67% full-accept path (correct because the M=K verify
post-state IS byte-equivalent to K sequential M=1 advances for the accepted
prefix; proven by md5 equality). The D2D snapshot removes the last big per-step
overhead (the PCIe state stage) with an exact device copy. Both are byte-identical
and inert for greedy, so they carry no correctness risk.

**Remaining gap to ~56 greedy (honest):** still below greedy. The structural
lever left is **fusing out the separate M=1 base decode** so per-step cost = a
single M=2 verify forward. It is NOT trivially fusible: circular dependency — the
MTP head needs the hidden state AFTER the last committed token to draft, but that
token/bonus is only processed by the base decode. EAGLE-style seeding the MTP
head from the previous verify's frontier hidden would break the cycle and stay
token-identical (the target verify corrects every draft, so drafts only affect
acceptance RATE, not correctness), but the speed benefit hinges on preserving the
83.3% acceptance through the changed seeding — needs MTP-head reseeding-semantics
verification. Deferred as the next (riskier, bigger) turn; approach-B + D2D are
landed and durable regardless.
