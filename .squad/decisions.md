# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

### 2026-07-12T14:50:00-07:00: Decisions archive rollover
**By:** Scribe
**What:** Archived the fully merged canonical decision file to `decisions/archive/2026-07-12T14-50-00Z-decisions.md` after the publish/audit merge pushed `decisions.md` above 20480 bytes.
**Why:** Keep the hot decisions file small while retaining the complete merged inbox record, including publish, CI, audit, security, and speculative-runtime notes, under `.squad/decisions/archive/`.

---

### 2026-07-12T14:50:00-07:00: PUBLISHED to crates.io: onnx-genai v0.1.0 and seven sub-crates
**By:** Scribe
**What:** Published `onnx-genai v0.1.0` plus all seven sub-crates to crates.io: `onnx-genai-metadata`, `onnx-genai-kv`, `onnx-genai-scheduler`, `onnx-genai-ort`, `onnx-genai-ort-sys`, `onnx-genai-engine`, and `onnx-genai-server`. The release contract is `.github/workflows/publish.yml` using the protected `crates` environment and `CARGO_REGISTRY_TOKEN`; it publishes leaves-first, checks crates.io before each publish, skips versions already present, uses a User-Agent header for registry API calls, and is safe to re-run idempotently. Future releases are performed by bumping the workspace version and re-running the workflow.
**CI/Audit:** `.github/workflows/ci.yml` runs fmt/build/test on push and PR; clippy remains non-blocking until warning cleanup makes `-D warnings` viable. `.github/workflows/audit.yml` runs weekly cargo-audit and on dependency changes; fresh `cargo audit` found 0 vulnerabilities. Audit code regularly via scheduled cargo-audit plus periodic review passes.
**Security:** Batched-driver DoS findings were fixed with bounded active+pending admission (`max_pending`, HTTP 429 + `Retry-After`) and non-blocking bounded delivery that drops slow/closed clients instead of stalling the shared driver.
**Speculative runtime:** §27/§28 includes prompt-lookup n-gram speculation (`NgramProposer`, greedy-identical), `MtpProposer` and `SpeculativeMode::Mtp`, a full ignored tiny-MTP package/e2e fixture (`tiny-mtp-full`) that matches greedy, an EAGLE-3 fixture with proposer TBD, and speculator config auto-discovery for vLLM-style metadata. Remaining optional speculative work: EAGLE-3 proposer and optimized full-MTP hidden-output decode.
**Model policy:** Agents use task-appropriate models with `gpt-5.5` as the floor.
**Why:** The repository is now published, continuously validated, routinely audited, and has the complete runtime milestone set recorded in the canonical team log.

---

### 2026-07-12T14:50:00-07:00: Release and audit conventions
**By:** Scribe
**What:** Publishing is via `.github/workflows/publish.yml` using the protected `crates` environment; bump the workspace version and re-run the workflow for future releases. Audit code regularly through scheduled cargo-audit plus periodic review passes.
**Why:** These contracts should be durable and visible to future agents before they change release or security workflows.

---

### 2026-07-12T14:56:00-07:00: Make workspace clippy blocking
**By:** Batty
**What:** Cleared workspace clippy warnings covering dereferenced types matching their origin, collapsible conditionals, a derivable `Default`, needless range indexing, a large enum variant, explicit `.into_iter()`, needless `Ok`/`?`, and unnecessary mutable bindings. The private server driver command now boxes its oversized generation request variant. No `#[allow]` attributes were added. CI clippy now runs with `-D warnings` and is blocking.
**Why:** A warning-free workspace lets CI enforce Rust idioms and prevent new clippy warnings without changing generation behavior or public API ergonomics.

---

### 2026-07-12T15:40:00-07:00: Decode ownership boundary
**By:** Batty
**What:** Audited ORT `decode.rs`/`mtp.rs` against engine `decode.rs`, `decode_loop.rs`, and `kv_bridge.rs`. ORT decode sessions correctly owned single-forward tensor binding plus KV buffers/cursors/rewind, but `MtpDecodeSession::propose`, target embedding/LM-head abstractions, and argmax token selection were generation policy leaked into ORT. Moved that proposal loop and its target-side helpers into engine `speculative.rs`; ORT `MtpDecodeSession` now exposes only detection, one-step hidden-state execution, KV state, reset, and rewind. Documented ORT as single-forward/KV-buffer ownership and engine as token selection, stopping, multi-step generation, constraints, and logical KV policy. The explicit adapter seam is engine's `DecodeBackend` trait; `ModelDecodePath` is the model-I/O strategy enum used to select its implementation, not a trait.
**Why:** This keeps runtime tensor/KV mechanics close to ORT while ensuring every decision about which token to generate or when generation ends remains in the engine, without changing proposal behavior.

---

### 2026-07-12T15:20:00-07:00: Establish LLVM coverage baseline and target high-value gaps
**By:** Pris
**What:** Installed `llvm-tools-preview` and `cargo-llvm-cov` 0.8.7, added `scripts/coverage.sh`, documented coverage usage, and added deterministic tests for speculative prompt-lookup rejection/configuration edges, JSON constraint finish/stop behavior, paged-KV validation/eviction/prefetch errors, int8 copy-on-write rewrites, and prefix-cache active/missing release/eviction behavior. Comparable pre/post coverage excluding concurrently changing `onnx-genai-server` and newly added `onnx-genai-bench` rose from 73.83% to 74.60% lines and 73.15% to 73.85% regions. In that scope, engine rose 73.21%→73.80% lines / 70.32%→70.87% regions and KV rose 90.41%→93.63% lines / 91.04%→92.90% regions. Final full-workspace coverage is 75.63% lines and 74.34% regions: onnx-genai 0.00%/0.00%, onnx-genai-bench 0.00%/0.00%, engine 74.87%/71.81%, KV 93.63%/92.90%, metadata 80.87%/79.35%, ORT 68.67%/70.54%, scheduler 91.70%/88.48%, server 80.05%/76.63% (line/region). `onnx-genai-ort-sys` generated bindings do not appear as instrumented workspace source. The initial full-workspace run was blocked by concurrent incomplete server edits, so no fabricated server/bench “before” number is reported.
**Why:** The largest remaining meaningful gaps are engine KV/ORT bridge failure and rewind paths (30.48% lines), engine decode variants and ORT error paths, ORT decode/value/allocator/chat-template failures, server driver shutdown/overload/session-error/streaming-tool-call branches, and untested CLI/benchmark entry points. Add a CI coverage job using `scripts/coverage.sh --fail-under-lines 75`, publish an HTML or LCOV artifact, and ratchet the floor upward without reducing coverage; prioritize getting engine above 75% and adding targeted ORT decode error-fixture tests. `cargo test --workspace` and blocking `cargo clippy --workspace --all-targets -- -D warnings` pass.

---

### 2026-07-12T15:20:00-07:00: Add OpenAI-compatible legacy completions endpoint
**By:** Rachael
**What:** Added `POST /v1/completions` with OpenAI legacy text-completion request/response mapping, including `prompt`, optional `suffix`, sampler parameters, stop sequences, usage, and SSE streaming chunks followed by `[DONE]`. Requests without `suffix` use normal generation; requests with `suffix` route through the engine's `generate_fim_with_config` path using the model's auto-detected `FimConfig`. Models without recognized FIM tokens return HTTP 400 when `suffix` is supplied.
**Why:** This exposes the engine's existing FIM/infilling capability for coding-agent clients while sharing the server generation driver, admission cap, context/output limits, error mapping, and plain-completion session handling.

---

### 2026-07-12T15:56:00-07:00: OpenAI-compatible vision input routing
**By:** Rachael
**What:** `/v1/chat/completions` now accepts message content as either a string or OpenAI-style `text`/`image_url` parts. Image URLs support base64 `data:image/...` URIs and bounded, timeout-protected HTTP(S) fetches. Pipeline startup inspects the loaded encoder's `pixel_values` metadata to derive endpoint, layout, and fixed image size; decoded RGB images are resized and normalized to `[0, 1]` f32 tensors. Vision requests use `PipelineGenerateRequest`, while single-model requests containing images return HTTP 400.
**Why:** The engine already supports external pipeline tensors, so the server can add vision without changing engine or ORT internals. `[0, 1]` is the only generic normalization available because current pipeline metadata does not declare processor mean/std. A real production vision model still needs a real CLIP-style encoder plus compatible decoder/image-feature contract packaged as a VLM pipeline via mobius; Pris should supply that model package and its processor metadata so model-specific normalization can replace the generic fallback.

---

### 2026-07-12T15:40:00-07:00: Split the HTTP server library into focused modules
**By:** Rachael
**What:** Reduced `onnx-genai-server/src/lib.rs` to public re-exports plus `app()`/`serve()` wiring. Request/response schemas moved to `types.rs`, handlers and request preparation to `routes.rs`, batch-driver ownership and event routing to `driver.rs`, session registry and caps to `session.rs`, streaming buffers/chunks to `sse.rs`, and state/config/model loading to `state.rs`; unit tests moved to `tests.rs`.
**Why:** Issue #2 called for maintainable module boundaries around the 2,713-line server library. The crate-root re-exports preserve the existing public Rust API, and endpoint behavior, streaming, tool handling, admission/session caps, and driver logic are unchanged.

---

### 2026-07-12T15:20:00-07:00: Add a device-comparable Criterion benchmark crate
**By:** Sebastian
**What:** Added the non-published `crates/onnx-genai-bench` workspace member. Its default Criterion target covers tokenizer encode/decode, greedy/top-k/top-p/min-p sampling, paged KV allocation/deallocation, a seven-processor logit chain, and llguidance grammar masking without loading a model. The `bench-ort` target covers tiny-model E2E generation, scatter-cache prefill lengths, and scatter static batches. Run `scripts/run_benchmarks.sh` (or add `--model`) to print machine metadata and a comparable scenario/metric table while retaining Criterion reports under `target/criterion`.
**Why:** A fixed suite, stable scenario names, explicit units, and machine metadata allow measurements from the same commit to be saved and diffed across CPUs and devices without placing benchmarks on the default test execution path. A short validation run on an Apple M1 Max (10 cores, Darwin 25.5.0 arm64, rustc 1.97.0) measured encode 3.47M tokens/s, decode 26.92M tokens/s, greedy 80.4µs/token, top-k 456.8µs/token, top-p 718.0µs/token, min-p 327.0µs/token, KV page cycling 532.8K pages/s, the processor chain 687.4µs/step, and grammar masking 78.5µs/step. These are smoke-run samples, not publication-quality measurements; use the runner's default Criterion duration for comparisons.

---

### 2026-07-12T16:06:00-07:00: Add low-overhead server observability core
**By:** Sebastian
**What:** Added a process-wide atomic metrics registry with `onnx_genai_*` request, token, TTFT, end-to-end latency, session, queue, batch, prefix-cache, and rejection metrics. Added unauthenticated `GET /metrics` behind the default-on `metrics` Cargo feature and always-on `GET /v1/status`. Instrumentation covers HTTP request spans/trace IDs, route status counters, driver queue/batch lifecycle, generated tokens, prefix-cache results, sessions, TTFT, and generation latency.
**Why:** DESIGN §§31–32 require production metrics and quick status introspection with negligible hot-path overhead. The implementation uses relaxed atomics and fixed-size histogram/status arrays with no per-request registry allocations or background exporter. Perfetto tracing, OTLP, and the full `/v1/debug/*` suite remain future work.

---

### 2026-07-12T16:14:00-07:00: Planning batch closed issues #2-5 and recorded next gaps
**By:** Scribe
**What:** Issues #2-5 are closed. #2 split `onnx-genai-server/src/lib.rs` into focused `routes`, `driver`, `sse`, `types`, `state`, and `session` modules while preserving public re-exports. #3 clarified decode ownership: ORT owns forward execution plus KV buffers/cursors/rewind, while the engine owns generation loops, token policy, stopping, constraints, and the `DecodeBackend` seam. #4 added OpenAI-compatible `POST /v1/completions` for legacy completions and FIM via `suffix`. #5 added the non-published `onnx-genai-bench` Criterion crate and device-comparable benchmark runner.
**OpenAI surface:** Current surface covers chat completions, tool calls, FIM through `/v1/completions`, image input parts for `/v1/chat/completions`, and streaming. Vision accepts base64 data URIs and bounded HTTP(S) image URLs, preprocesses to pipeline tensors, routes VLM pipeline requests, and returns HTTP 400 for image input on non-VLM single-model paths. Real vision quality still requires a production mobius CLIP+decoder VLM package and processor metadata.
**Observability:** Added low-overhead atomic metrics, request spans and trace IDs, default-on `metrics` feature with `GET /metrics` Prometheus output, and always-on `GET /v1/status`. Instrumentation covers request/status counts, driver queue/batch/session state, generated tokens, TTFT, end-to-end latency, prefix-cache hits, and 429 rejections. Perfetto, OTLP, and full debug endpoints remain deferred.
**Coverage:** `cargo-llvm-cov` is available via `scripts/coverage.sh`. Full workspace baseline is 75.63% line / 74.34% region overall: KV 93.63, Scheduler 91.70, Server 80.05, Engine 74.87, ORT 68.67 line coverage. Proposed CI floor is `scripts/coverage.sh --fail-under-lines 75`; biggest high-value gap is engine `kv_bridge`.
**Why:** This records the completed planning batch, preserves endpoint/metrics/benchmark/coverage contracts, and makes remaining follow-ups visible to future routing.

---

### 2026-07-12T16:14:00-07:00: CI and deterministic real-model test conventions
**By:** Scribe
**What:** CI clippy is blocking and must run as `cargo clippy --workspace --all-targets -- -D warnings`. Real-model exact-equality tests that compare floating-point generation outputs should force ORT single-thread execution with `intra_op_threads=1` to avoid reduction-order tie flakes; production defaults remain unchanged.
**Why:** Future agents need to treat clippy warnings as build failures and keep deterministic exact-equality tests stable across ORT scheduling differences.
