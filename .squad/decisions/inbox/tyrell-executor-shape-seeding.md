### 2026-07-24: Seed warm JIT decode shapes + capture-recording quarantine (Stage 0 of DeepSeek whole-step capture)

**By:** Tyrell
**Branch:** `perf/capture-executor-shape-seeding` (off `perf/deepseek-mla-capture` @ `25dbb60` — the Attention capture foundation, currently in review). **Needs review before merge; not merged.** Rebase onto the merged MLA foundation when it lands. Headline tok/s bench is deferred to the integration pass on `bench/ort-vs-native-cuda` (GPU contention here makes the ~2 ms/token direct gain unmeasurable; the structural seam-count drop is the acceptance criterion).

**Scope:** `crates/onnx-runtime-session/src/executor.rs` ONLY. No kernel files, no `provider.rs`, no `standard_attention.rs`/`native_decode.rs`. This makes the executor *admit* already-capture-safe ops; it does not add/alter kernels.

#### Root cause (confirmed, Pris's finding reproduced exactly)
The executor rejects a node as an eager seam **before** consulting its kernel whenever any input/output shape is absent from `resolved` (EP `plan_capture_region` default policy declines on unresolved shapes). `resolve_soft` deliberately omits data-dependent (JIT) decode shapes, and only external/control-flow shapes were seeded for capture. So DeepSeek-V2-Lite decode ops that are ALREADY capture-safe (Cast, Mul, QMoE, ScatterElements — all advertise `Supported`, skip sync, pool scratch during capture) still fragmented into eager seams purely because their JIT output shapes weren't seeded. Measured: **727 distinct eager seam nodes** per decode step (matches Pris exactly).

#### Fix
1. **Warm decode shape seeding** (`seed_warm_decode_capture_shapes`). After an eager warmup step, snapshot the full resolved shape map (`capture_warm_shapes`) together with the persistent-binding signature it ran under (`ExternalBindings::capture_signature()` = sorted (vid, is_input, dtype, shape, ptr, len) of every persistent binding). On a later capture-mode run presenting the **identical** signature, seed each still-unresolved (non-external, non-initializer, non-sequence) value from the warm snapshot so its already-capture-safe consumers fold into captured segments. Guardrails, all honored:
   - Shapes are derived from a real eager warmup, never hardcoded/assumed.
   - A changed persistent pointer/capacity/shape → signature mismatch → **all seeds withheld** (nodes stay eager); `replay_device_graph`'s independent `binding_signature` check also retires the installed graph. Never replays a stale graph against changed shapes.
   - The capture pass re-resolves each node's true shape; any divergence from a seeded value retires the graph and declines (recapture) rather than baking a stale shape.
   - No per-step allocation when the signature matches; view/bounds validation untouched.
   - Seeding is valid ONLY for the exact warmed signature — anything varying across steps forces recapture or stays eager.

2. **Capture-recording quarantine + retry** (in the `RunMode::Capture` arm + `node_capture_reason`). Seeding surfaced a latent problem: a kernel can advertise `CaptureSupport::Supported` yet abort device-graph *recording* (e.g. `ai.onnx::Softmax`, the MoE gate — softmax.rs declares `Supported` but calls `synchronize()` unconditionally, which CUDA rejects mid-capture). Admitting one such node aborted the **entire** segmented capture → full eager fallback (0 captures). Fix: when `run_plan_segmented` (Capture) errors at a node, record it (`last_capture_failed_node`), reset the device graph, quarantine its `(domain, op_type)` (`capture_quarantine_ops`), and re-plan/re-record treating quarantined ops as forced `CaptureRecordingFailed` eager seams. Re-recording a fixed-capacity decode step is idempotent (same position/token → same values into the same slots), so retry is safe; bounded by node count; quarantine grows monotonically (a kernel that breaks recording breaks it every time), so recaptures converge immediately. New `SeamReason::CaptureRecordingFailed`.

#### Results — proof of effect (`ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1`, `--steady --decode-skip 8 --warmups 1 --runs 1 --tokens 12`, GPU 1)
Distinct eager seam nodes per decode step, **identical for both exports** (blk32 `deepseek-v2-lite-real-int4-blk32` and blk128 `deepseek-v2-lite-real-int4`):

| | seeding OFF (baseline) | seeding ON + quarantine |
|---|---|---|
| **distinct eager seam nodes** | **727** | **541** (−186, −25.6%) |
| eager node executions across run | 1454 | 1082 (−26%) |
| "data-dependent shape unresolved" seam class | 692 occ (Cast 106, Mul 104, QMoE 52, ScatterElements 52, MatMul 52, TopK 52, GatherElements 52, Softmax 52, …) | **0** — class eliminated |
| segmented-capture status | succeeds (191 seg / 190 seam) | **succeeds** (193 seg / 192 seam) |

**Cast, Mul, QMoE, ScatterElements stopped being seams** (fully folded into captured segments). The nodes still eager after seeding now report their **real kernel-capability decline** (not a spurious missing-shape rejection), which is exactly the signal kernel owners need — see below.

#### Correctness / determinism (HARD GATE — PASS, both exports)
Prompt "The capital of France is", 3× identical each export:
`[8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185]` = pos0 8913 ' Paris' → matches expected exactly. Capture engaged and clean: `cuda_graph: captures=2 replays=18 fallbacks=0` (no stale-graph corruption — the main risk of this change is disproven).

#### Dense non-regression (PASS)
Qwen2.5-0.5B int4 (`qwen2.5-0.5b-int4-onnx-native`): 3× identical, coherent (" Paris. It is the largest city in the country and the"), `captures=2 replays=18 fallbacks=0`. Dense graphs have statically-resolved decode shapes, so warm seeding is a no-op for them (nothing unresolved to seed) — no behavior change, no regression.

#### Ops that I EXPECTED to fold but did NOT (for the kernel-owner agents)
These now surface their true kernel decline (they were previously hidden as unresolved-shape seams). They stay eager until their kernel is made capture-safe:
- **`ai.onnx::Softmax` (MoE gate) — KERNEL BUG:** declares `CaptureSupport::Supported` but `run`/`run_nvrtc_f32` call `self.runtime.synchronize()` unconditionally (`crates/onnx-runtime-ep-cuda/src/kernels/softmax.rs:271,323`; `capture_support()` at :343). This aborts recording; my quarantine keeps capture working but Softmax stays a seam (52/step). **Fix the kernel to skip the sync during capture (mirror the Cast/Mul pattern) and it will fold for free.**
- `ai.onnx::Reshape` — copy path not a capture-validated zero-copy view.
- `ai.onnx::Split` — reads runtime split sizes on host + trailing stream sync.
- `ai.onnx::Concat` — trailing host stream sync.
- `ai.onnx::Expand` — per-call broadcast metadata alloc/upload/free + sync.
- `ai.onnx::TopK` — reads K D2H + host sync.
- `ai.onnx::GatherElements` — per-call indexing metadata + sync.
- `ai.onnx::MatMul` (M==1 GEMV) — cuBLASLt per-call workspace alloc/free + heuristic query not capturable.
- `ai.onnx::Where` — capture-safe only for invariant scalar-predicate select over equal-shaped operands; broadcast/non-scalar condition launches stay eager.
- `ai.onnx::Unsqueeze` / `Slice` / `CumSum` — host-side runtime axes/bounds + sync (structural host seams; not shape-gated).

#### Gates
- `cargo test -p onnx-runtime-session --features mlas --lib` → **65 / 0** (63 baseline + 2 new tests).
- `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` (GPU 1) → **208 / 0** (≥207, no regression).
- `cargo clippy -p onnx-runtime-session --features mlas --lib -- -D warnings` → clean. (Pre-existing repo test-only debt `let mut input_axes` in an unrelated executor test is not introduced here — same item Deckard noted.)
- `cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native` → ok.

#### Tests added (non-tautological)
- `warm_decode_seeding_admits_previously_unresolved_capture_safe_node`: a `Range`(runtime start/limit/delta)→`Cast` graph is an unresolved-shape seam before warmup; after one eager warmup the identical signature seeds the exact extent `[4]` and clears the unresolved-shape seam; a changed persistent-binding signature withholds the seed.
- `quarantined_op_type_is_forced_to_a_capture_recording_failed_seam`: a statically-shaped `Cast` is not a recording-failed seam until its `(domain, op_type)` is quarantined, after which `node_capture_reason` forces it to `CaptureRecordingFailed` regardless of resolved shapes/kernel capability.

**Files changed:** `crates/onnx-runtime-session/src/executor.rs` (+ 2 tests in-module).
