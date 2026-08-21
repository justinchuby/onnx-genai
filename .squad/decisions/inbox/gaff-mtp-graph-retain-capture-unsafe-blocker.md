### 2026-08-21: MTP graph retention across verify+rewind is capture-unsafe — root cause localized, dormant seam landed, no speedup number

**By:** Gaff

**What:**
Investigated the "make MTP `replays>0`" speedup lever (turn the 78% acceptance into a real net win vs the 62.56 tok/s greedy baseline). Landed the two-site invalidation seam as a **documented dormant flag** (`retain_decode_graph_across_spec`, default OFF) plus test-only setters, mirroring the existing dormant `retain_graph_on_rewind`/option-c convention. **No speedup number exists** — naive retention is both insufficient and capture-unsafe (GPU-verified). Behavior is byte-inert vs origin/main `843b0bf7d` (flag never enabled).

Root cause of MTP `replays=0` (empirically localized, GPU, 10 verify-steps steady):
- The captured M=1 decode graph is torn down TWICE per spec step: **30× at `cuda.rs` `run_cuda_eager_rows_owned`** (the eager M>1 verify forward calls `invalidate_graph`) + **21× at `backend.rs:145` `rewind_inner`** (commit rewind). The capture state machine needs ≥2–3 consecutive same-shape M=1 steps to reach a replay; the per-step invalidations tear it down first. `binding_signature` (bindings.rs:1090) keys only on physical_shape+device_ptr (not logical length/data), which is why greedy replays fine as KV grows but MTP never does.

Why naive retention fails (GPU-verified via temporary A/B env toggles, since reverted):
- Retain-across-rewind-only (safe, replays=0, ~15 tok/s) and retain-across-verify-only (safe, replays=0, ~15 tok/s) each leave the *other* site invalidating, so no graph ever survives to replay.
- Retain across **BOTH** — the only config that lets a graph survive a full verify→rewind→re-advance cycle and replay — produces **non-finite logits (exit 1, finite-guard caught)**. Cause: the eager M=K verify reserves a larger StepScoped `step_workspace` that `release_step_workspace` (bindings.rs:793) frees after the run; the captured M=1 graph baked the old workspace pointer, so the next M=1 replay reads a stale/moved address → NaN. Greedy is immune because every step is the same M=1 shape → arena returns the same address.

Deeper structural blocker (the real reason there's no easy win):
- Even with the M=1 replay workspace issue solved, the **M=K verify forward stays eager (un-captured)** and pays full per-op launch overhead — exactly what graphed greedy avoids (55.91 vs eager). A real MTP speedup requires **capturing the verify itself** into a fixed-shape replayable graph: option-c "padded verify capture" (pad every forward to maxK rows, capture once at maxK, replay for base and verify), plus a pinned/isolated StepScoped workspace and a shape-keyed graph slot (the EP currently holds a single `device_graph_signature`, can't hold M=1 base + M=K verify at once). This is a multi-turn executor workstream; dormant scaffolding already exists (`configure_padded_verify_capture`, `padded_query_capacity`, `retain_graph_on_rewind`, all `#[cfg(test)]`).

**Why:**
The coordinator's fallback clause: if graph retention across differing shapes proves infeasible in one turn, land the root-cause analysis + any partial capture-safety improvement and report precisely — do not fabricate a speedup. Retention is not just infeasible in one turn, it's **wrong** as scoped (option-b eager verify can never beat graphed greedy no matter how the M=1 graph is retained). Landed the documented two-site seam + flag as the low-risk building block the future option-c work will flip on, keeping the exact GPU evidence inline at both invalidation sites so the next agent has the map. Recommend the next turn be scoped explicitly to **option-c padded verify capture** (capture the verify), not further M=1-retention tuning.

**Validation (GPU, H200 CUDA_VISIBLE_DEVICES=5, all 8 idle 0MiB/0%; build `--features bench-native,native-cuda,cuda-13000`; ORT 1.28 cuda13 `.ort-cuda-1.28/root`; int4 block-32; harness `profile_native`; branch `squad/mtp-retain-graph-on-rewind` off origin/main `843b0bf7d`):**
- Full lib suite `cargo test -p onnx-genai-engine --no-default-features --features native-backend --lib`: **575 passed, 0 failed, 1 ignored** (greedy inert).
- Inertness on real Qwen3.8-27B int4 hybrid artifact `/home/justinchu/qwen38-27b-int4-mtp-cuda` (short window, flag OFF = origin behavior):
  - MTP (`--steady`): 14.45 tok/s, acceptance 78.9%, `cuda_graph enabled=true captures=16 replays=0 fallbacks=0 invalidations=99`, no NaN, tokens_per_verify_step 2.58.
  - Greedy (plain): 55.39 tok/s, `cuda_graph captures=2 replays=92 fallbacks=0 invalidations=1` (healthy, matches origin ~55.91).
- **No MTP speedup number is reported — none exists** (no config safely produced replays>0). MTP stays ~3.8× slower than greedy under eager verify.
