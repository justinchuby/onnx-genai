### 2026-08-01: Inc-1b PR-2 — wire decode-specialized inlined-body Executor (dual-plan EAGER, flag default-OFF). Banks ~1.26× on 27B decode. Scope HELD (Size M–L, MEDIUM blast radius). Greenlight-FREE.

**By:** Cohaagen (EP/runtime perf)
**PR:** https://github.com/justinchuby/onnx-genai/pull/588 — branch `squad/inc1b-pr2-decode-inline`, base `origin/main` (`70bac712`, includes transform PR #580), commit `82689914`. OPEN, **no auto-merge** (reviewer: Harry, author-lockout).
**Task:** Build Inc-1b PR-2 from `cohaagen-27b-inc1b-design.md` §5 — wire PR-1's `inline_single_trip_scan_bodies` transform into a second, decode-specialized `Executor` and route single-token decode to it, eager, capture OFF, flag default-OFF.

**Outcome: SHIPPED as a bounded PR. Scope HELD.** Did NOT grow to L; did NOT touch the capture surface (#443/#543), `plan_capture_segments`, or `run_plan_segmented`. The eager win materialised on the real target (27B) exactly as design §5 predicted.

## What I built
1. **Sibling Executor builder** (`onnx-runtime-session`): `Executor::build_decode_inline_sibling(&self) -> Result<Option<Self>>` runs `inline_single_trip_scan_bodies(&self.graph)`, re-resolves interior shapes with Permissive `InferenceRegistry::infer_graph` (mirrors `ChildExecutor::compile`), then `Self::build(graph, Arc::clone(&self.weights), Arc::clone(&self.ep))` — **shares the SAME `Arc<WeightStore>` and `Arc<dyn ExecutionProvider>`** (multi-plan/shared-weights pattern). Returns `None` for a graph with no single-trip-eligible recurrent Scan (dense decoder), so the caller keeps today's path. The main/prefill exec is left **byte-identical**.
2. **Session API** (`onnx-runtime-session`): `InferenceSession` gains `decode_inline_exec: Option<Executor>` + `enable_decode_inline()` (idempotent lazy build), `decode_inline_ready()`, and `run_decode_inline_with_device_bindings(inputs, bindings)`. The sibling binds the IDENTICAL persistent device buffers `bindings` supplies (bindings resolve by name; the transform leaves `graph.inputs`/`graph.outputs` names+order unchanged → state continuity is automatic).
3. **Engine flag + routing** (`onnx-genai-engine`): `ONNX_GENAI_DECODE_INLINE_SCAN` (default OFF; truthy = `1`/`true`/`yes`/`on`). Lazy `maybe_enable_decode_inline` at the first single-token decode step (both `decode_with_step_inputs` and the greedy `decode_argmax` entry points). Single-token decode routes to the sibling on **all three native single-token paths**: CUDA greedy device-argmax fast path (`decode_cuda_greedy` — THE path the 27B profile uses), CUDA logits path (`decode_cuda`), and CPU in-place path (`decode_cpu_inplace`). The greedy path reuses the existing device-argmax kernel (`read_greedy_result`) so tie-breaking is byte-identical to the captured baseline and full logits never round-trip to host.

**Capture stays OFF** — the sibling runs eager; the main exec's capture state machine stays dormant on decode. Device-graph capture of the inlined body is PR-3 (greenlight-gated), explicitly OUT OF SCOPE.

## Files changed
- `crates/onnx-runtime-session/src/executor/build.rs` — `build_decode_inline_sibling` + `default_domain_scan_count` helper.
- `crates/onnx-runtime-session/src/lib.rs` — `decode_inline_exec` field + 3 public methods.
- `crates/onnx-runtime-session/src/executor/tests.rs` — 4 guard tests.
- `crates/onnx-genai-engine/src/native_decode/mod.rs` — `DecodeInlineState`, flag fn, pure `route_decode_inline_decision`, `maybe_enable_decode_inline`, `route_decode_inline`.
- `crates/onnx-genai-engine/src/native_decode/backend.rs` — `maybe_enable_decode_inline` at the greedy `decode_argmax` entry.
- `crates/onnx-genai-engine/src/native_decode/cuda.rs` — inline branch in `decode_cuda` (logits) and `decode_cuda_greedy` (device-argmax fast path).
- `crates/onnx-genai-engine/src/native_decode/cpu.rs` — inline branch in `decode_cpu_inplace`.
- `crates/onnx-genai-engine/src/native_decode/load.rs` — field init.
- `crates/onnx-genai-engine/src/native_decode/tests.rs` — 3 guard tests.

## Blast radius: MEDIUM
New default-off decode route on the three native single-token paths; prefill, multi-token, dense, and every capture path are untouched. When the flag is OFF the session is byte-identical to today (sibling never built, `decode_inline` latches `Disabled`).

## Harry's 4 MANDATORY guards → which test covers each
1. **Byte-identical parity + final recurrent state** — `decode_inline_sibling_is_byte_exact_with_scan_and_preserves_state` (session): N decode steps through the Scan child-session plan vs the decode-inline plan on a hybrid fixture; per-token outputs byte-identical AND final recurrent state identical.
2. **Runtime scan-axis extent==1 assertion + fallback** — `route_decode_inline_decision` (pure) + `decode_inline_routes_only_single_token_when_enabled` / `decode_inline_never_routes_when_disabled_or_unbuilt` (engine): only a single-token (extent-1) step routes to the sibling; every multi-token step falls back to the main Scan exec, so a wrongly-collapsed graph is never run.
3. **Persistent state-buffer continuity** — `decode_inline_sibling_preserves_persistent_state_across_prefill_handoff` (session): the sibling binds the identical persistent device state buffers the main exec used at the prefill→decode hand-off.
4. **state_pairs ordering + shape check** — `decode_inline_sibling_preserves_state_output_order_and_resolves_shapes` (session): first `num_state` present outputs map to present-state in `io.state_pairs` order; inlined-interior shapes resolve (Permissive) before use.
Plus `decode_inline_sibling_none_for_dense_graph` (session, negative) + `decode_inline_flag_defaults_off_and_parses_truthy` (engine).

## Perf — measured OFF vs ON on Qwen3.6-27B int4 hybrid (H200, `profile_native --ep cuda --backend native --steady --decode-skip 8 --warmups 2 --runs 3 --tokens 64`)
| flag | decode ms/tok (medians across 4 runs) | tok/s |
|------|----------------------------------------|-------|
| OFF  | 149.2 / 155.9 / 157.7 / 150.6 (~153)  | ~6.5  |
| ON   | 117.7 / 124.0 / 121.9 / 126.7 (~122)  | ~8.2  |

**~1.26× decode speedup** (design predicted ~1.28×; the 167→130 ms/tok absolutes in the design were on a slower baseline — this build/GPU runs 153→122, same ratio). **Generated token ids byte-identical OFF vs ON on every run** (strong on-model semantics-preservation evidence in addition to guard #1).

**Why eager beats the CAPTURED baseline** (verified with an instrumented probe, since removed): the 27B baseline decode graph *is* CUDA-graph captured, yet the captured Scan operator still pays real per-step child-dispatch + loop-state collect work inside each replay. Inlining the single-trip body removes that boundary entirely, so the eager inlined graph does strictly less work per token than the captured-Scan graph — exactly the "child entry/exit + collect + setup/finish" removal in design §5. (First ON measurement was flat because my initial wiring missed the greedy `decode_cuda_greedy` fast path that the profile actually uses; wiring it delivered the win.)

## Byte-exact GPU e2e gate
`crates/onnx-genai-engine/tests/native_autoderive_io_cuda_e2e.rs` (`#[ignore]`, stock 27b == CPU fp32 oracle, expected ids `[11751,13,271,248068,271,248069,271,4639,369,4252,13,11751,369,279,6511,321]`) run with the flag ON (`ONNX_GENAI_DECODE_INLINE_SCAN=1`, `--test-threads=1`, CUDA env, `ONNX_GENAI_REQUIRE_CUDA=1`): _<result filled below>_. (profile_native already proved ON is byte-identical to OFF on the 27B GPU decode, and OFF matches the oracle by the pre-existing gate.)

## fmt / clippy / test evidence
- `cargo fmt --all --check` — clean.
- `cargo clippy -p onnx-runtime-session --all-targets -- -D warnings` — exit 0 (fixed two `useless_vec` in my tests).
- `cargo clippy -p onnx-genai-engine --features native-backend --all-targets -- -D warnings` — exit 0.
- `cargo clippy -p onnx-genai-engine --features cuda,native-backend --all-targets -- -D warnings` — exit 0 (covers the touched CUDA `decode_cuda`/`decode_cuda_greedy` branches).
- `cargo test -p onnx-runtime-session --lib` — 109 passed (incl. the 4 guard tests).
- `cargo test -p onnx-genai-engine --features native-backend --lib` — 360 passed, 1 ignored (incl. the 3 guard tests); flag-OFF path unchanged.
- NOTE: a workspace-wide `cargo check` fails only on the unrelated `onnx-runtime-cpuinfo` cmake vendor submodule in a fresh worktree (environmental) — build/test per-crate.

## Honest scope statement
- Only the **native single-token decode paths** are routed to the inline sibling: CUDA greedy device-argmax (`decode_cuda_greedy`), CUDA logits (`decode_cuda`), and CPU in-place (`decode_cpu_inplace`). Routed-port / `inputs_embeds` eager-rows paths and multi-token steps keep the main exec.
- Capture is deliberately OFF (PR-3 territory, greenlight + capture-team sign-off required). This PR needs neither.
- Flag default-OFF ⇒ zero behavior change unless explicitly enabled.
