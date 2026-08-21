### 2026-08-21: MTP verify-dedicated interior arena + both CUDA-graph slots replay

**By:** Gaff

**What:** Gave the fixed M=k+1 speculative-verify forward its OWN executor
sibling with a private interior device-buffer arena (`build_verify_sibling`,
`Session::verify_exec` + verify-sibling methods, routed through
`run_verify_graph_phase`). This is the unlock for verify CUDA-graph
capture+replay: the sibling shares only immutable weights/EP + the persistent
external KV/recurrent-state bindings with the main executor, so the interleaved
M=1 base decode can no longer resize the interior scratch the M=2 verify graph
baked (the old `Slice [1,2] vs [1,1]` decline → eager fallback). Also fixed a
nondeterministic CUDA illegal-access (700): the sibling's captured graph bakes
its StepScoped workspace pointer, so the sibling now PINS its workspace
(`pin_step_workspace = true` in `build_verify_sibling`) — it only ever runs the
fixed M=K shape, so the workspace is reserved once and never freed back to the
shared EP arena between replays. The verify sibling's captured graph is reset
ONLY at the generation boundary (`rewind_inner` target_len==0), never in the
per-step `invalidate_graph`.

Then closed Blocker B (Primary M=1 decode graph was invalidated every spec
step): (1) `commit_recurrent_state_to_accepted` now re-advances accepted tokens
ONE at a time (M=1) instead of a batched M=num_accepted forward — sequential
recurrent advance is state-equivalent (unit tests stay byte-identical) but keeps
the Primary executor pinned at the [1,1] decode shape; (2) enabled the
`retain_decode_graph_across_spec` seam when verify-capture arms, so the
contents-only KV roll-back in the commit (fixed physical_shape/device_ptr) no
longer tears down the Primary graph. Its old caveat (unsafe until the verify
workspace is pinned) is resolved now the verify is a separate pinned sibling.

**Result (GPU-proven, H200 ord 5, qwen38-27b-int4-mtp-cuda, k=1, 83.3% accept,
median of 5, tokens=128 warmups=3, ORT 1.28 cuda13, int4 block-32, build
`--release --features bench-native,native-cuda,cuda-13000`):**
- **BOTH slots replay, fallbacks=0, invalidations→~0** (every one of 5 runs):
  - `cuda_graph:        captures=4 replays=504 fallbacks=0 invalidations=3`
  - `cuda_graph_verify: captures=4 replays=184 fallbacks=0 invalidations=0`
  - (Primary replays=504 exactly matches the established greedy baseline run.)
- Token-identical across all 5 runs (deterministic), no NaN, coherent text.
- MTP throughput: **median 20.63 tok/s** (20.52–20.64), up from 15.6 pre-fix.
- Suites: engine lib 579/0 (`native-backend`), ep-cuda graph::tests 8/8, the
  recurrent-commit + `native_verify_logits_require_restored_recurrent_state`
  byte-identity oracles pass.

**Why MTP is still < greedy (~56 tok/s) despite both slots graphed — honest
finding, NOT a speedup:** graphing the verify was necessary but not sufficient.
Approach-A (snapshot → restore → re-advance) redundantly RE-RUNS the accepted
tokens through the full model on commit, and MTP runs verify(M=2) + per-token
re-advance forwards per step for only ~2.67 tokens/step. On this launch-bound
q38 decode that is strictly more device work per accepted token than greedy's
single graphed M=1 forward, so the campaign's net-speedup target is not met by
graphing alone. The greedy baseline artifact is out-of-sandbox this session; the
number quoted is the campaign-established ~56 tok/s, corroborated by the Primary
replays=504 equivalence.

**Remaining gap / next lever:** move from approach-A to approach-B — thread
`num_accepted` through the verify so the committed recurrent state is SELECTED
from the verify's per-position post-states instead of re-scanned. That removes
the redundant re-advance forwards (the dominant remaining per-step cost) and is
the path to a net MTP win over greedy. The two-slot graphed executor delivered
here is the prerequisite for that work.
