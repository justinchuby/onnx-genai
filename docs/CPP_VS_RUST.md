# C++ vs Rust for onnx-genai and the ORT rewrite — an honest assessment

**Audience:** the C++ engineers on this team who will (rightly) be skeptical of "rewrite it in Rust."
**Purpose:** make the *honest* case — where Rust has already paid for itself in this codebase, where
C++ is still the right tool, and how a migration can be incremental and low-risk instead of a
religious big-bang rewrite.

This is deliberately not a hype piece. If you are an expert, hype is what makes you distrust the
argument. Everything below is grounded in code we have already shipped in this repo (and the sibling
`onnxruntime-mlx` plugin EP), with real numbers you can reproduce.

---

## 0. TL;DR

- We are **not** proposing to rewrite ONNX Runtime's MLAS/CUDA kernels in Rust. That would be a
  decade of work thrown away. We keep and *reuse* the fast C/C++/CUDA kernels through FFI.
- We **are** proposing that the *orchestration* layers — IR, graph optimizer, partitioner, memory
  planner, session/EP glue, KV cache, scheduler, sampling, serving — are better written in Rust, and
  that new components should start in Rust.
- The evidence is that we already did a lot of this and it works: **~119k lines of Rust across 27
  crates, 1468 tests**, a pure-Rust CPU EP with real kernels, a pure-Rust paged-KV cache, and a
  **pure-Rust ONNX Runtime plugin EP** (`onnxruntime-ep-mlx`) that loads into *stock* ORT and is
  published to crates.io + PyPI.
- The honest cost: **452 `unsafe` blocks**, almost all of them at C-ABI / CUDA boundaries, plus Rust's
  compile times and a less mature ML-kernel ecosystem.

If you read nothing else, read §3 (where C++ still wins) and §6 (the migration that doesn't bet the
company).

---

## 1. What we have actually built in Rust (the receipts)

| Layer | Crate(s) | Rust LOC | `unsafe` | Tests | Nature |
|---|---|---:|---:|---:|---|
| ONNX IR | `onnx-runtime-ir` | 2,113 | 1 | 40 | pure logic |
| Graph optimizer | `onnx-runtime-optimizer` | 3,273 | 1 | 47 | pure logic |
| Memory planner | `onnx-runtime-memory` | 1,190 | 2 | 13 | pure logic |
| CPU EP + kernels | `onnx-runtime-ep-cpu` | 16,122 | 29 | 328 | **real compute** (elementwise, pooling, softmax, …) |
| Paged KV cache | `onnx-genai-kv` | 5,718 | **0** | 56 | pure logic (paged + quantized KV) |
| Scheduler | `onnx-genai-scheduler` | 965 | **0** | 13 | pure logic |
| C-ABI / CUDA bridges | `onnx-runtime-capi`, `onnx-genai-ort`, `onnx-runtime-ep-cuda` | — | 139 / 98 / 70 | — | **FFI boundary** |
| MLX plugin EP (sibling repo) | `onnxruntime-ep-mlx` | 22,180 | 521 | 1000+ | **FFI-heavy** (every MLX op is a C call) |

The shape of the numbers is the whole argument: **the pieces that are pure logic are almost entirely
safe Rust** (KV cache: 0 `unsafe` in 5.7k lines; optimizer: 1 in 3.3k; CPU kernels: 29 in 16k), and
**`unsafe` is concentrated where it belongs — at the C boundary.** Rust does not pretend the hardware
is safe; it *contains* the danger to auditable, minority regions instead of letting it diffuse
through the whole codebase the way `reinterpret_cast` and raw pointers do in C++.

### Concrete capabilities we shipped in Rust

- A **pure-Rust ORT plugin EP** for Apple MLX: it implements the ORT plugin-EP C ABI, is loaded by an
  *unmodified* ONNX Runtime, claims ~130 op types, and covers `MatMulNBits`, `GroupQueryAttention`,
  `PagedAttention` (block-paged KV, packed var-length batches — CUDA-only in ORT itself, so Rust is
  the *only* way it runs on Apple Silicon), quantized Mixture-of-Experts (`QMoE`), `GatherND`, and a
  compiled-subgraph fusion path. Published to crates.io and PyPI via OIDC trusted publishing.
- A **paged-KV cache** with per-page quantization, single-token-append invariants, and 56 tests —
  0 `unsafe`.
- A **CPU execution provider with real kernels** written in safe Rust.
- A graph **optimizer**, **shape-inference**, and **partitioner** as ordinary, testable Rust modules.

None of this is theoretical. It is compiled, tested, and (for the EP) released to public registries.

---

## 2. Where Rust demonstrably helped *in this codebase*

These are not textbook claims; they are things that happened while building the above.

1. **Compiler-enforced invariants that used to be documentation.** The EP's op-claim predicates return
   a `Result<(), Cow<'static, str>>` with `require!`/`deny!` macros. Before, the "why did this op
   decline" reason lived in a *separate table* that silently drifted out of sync with the code. The
   Rust type system made the reason and the check the same expression — drift became impossible. That
   is an ownership/`enum`/`Result` property, not a coding-standard we have to police in review.

2. **The crashes we hit were never Rust UB.** In this codebase's recent work, every hard crash was
   either (a) inside the C++ MLX library (`mlx_compile` aborting on a bad closure), or (b) an
   FFI-contract mistake we made in *one* `unsafe` function (reading an `int32` tensor's bytes as
   `i64`). Both were localized in seconds because the safe/unsafe boundary told us exactly where to
   look. We spent zero time chasing use-after-free, double-free, or data races in the ~119k lines of
   safe Rust. In an equivalent C++ codebase those are the bugs that eat weeks.

3. **Fearless concurrency for serving.** The MLX EP is thread-affine; the rule "one session per
   thread" is *enforced* by the borrow checker and `Send`/`Sync` bounds, and shared caches are
   `Mutex`-guarded with the compiler refusing to let you forget. The server/scheduler crates lean on
   this: async batching without a class of heisenbugs.

4. **`cargo` as the whole build system.** No CMake, no protobuf compiler, no vendored toolchain matrix.
   `cargo build` cross-compiles; `cargo test` runs 1468 tests; the EP publishes to two package
   registries from one CI workflow with OIDC and no long-lived tokens. Onboarding a new component is
   `cargo new`, not a CMakeLists archaeology dig.

5. **Refactors that are actually safe.** We renamed the entire env-var flag namespace, refactored 129
   claim predicates from `bool` to a rich result type, and swept a codebase clippy-clean — each time
   the compiler enumerated every call site. "If it compiles, the rename is complete" is a real
   productivity multiplier on a large surface.

---

## 3. Where C++ is still the right answer (the honest cons)

If we pretend these away, the experts are right to ignore us.

1. **The kernels.** ORT's MLAS, the CUDA/cuDNN kernels, oneDNN, the fused attention kernels — these
   are the product of many engineer-years of hand-tuned assembly, intrinsics, and vendor libraries.
   Rust has **no MLAS-equivalent** and no mature GEMM/conv kernel ecosystem. Any honest plan **reuses
   these through FFI** and only rewrites a kernel when there is a specific, measured reason. Writing a
   competitive int8 GEMM in Rust today is a losing use of time.

2. **FFI is still `unsafe`, and there is a lot of it.** The MLX EP has **521 `unsafe` calls** —
   because every `mlx_*` op is a C function. Rust did not delete the danger there; it drew a box
   around it. If your component is 90% FFI (a thin kernel shim), Rust buys you much less than if it is
   90% logic (a scheduler). Be honest about which one you are writing.

3. **The ML ecosystem is C++/Python.** Model exporters, quantization tools, profilers, the reference
   ONNX implementation, most research code — all assume C++/Python. We interoperate constantly, and
   every interop point is an FFI/unsafe surface.

4. **Compile times and toolchain churn.** A cold workspace build is minutes, not seconds. We are on
   `edition = 2024`, which means a recent toolchain and occasional churn (let-chains, new lints).
   C++ incremental builds and the maturity of its tooling are real advantages day-to-day.

5. **Rewrite risk and opportunity cost.** ORT is millions of lines of *correct, fast, shipping* C++.
   "Rewrite it" is, at face value, the single most classic engineering mistake there is. A Rust ORT
   that is 95% as fast and 100% rewritten is a **strategic loss**, not a win. The only defensible
   version is incremental and reuse-heavy (see §6).

6. **Team expertise.** Our deep experts are C++ experts. Ramping the whole team on Rust's borrow
   checker, async, and macro ecosystem is a real cost, paid in calendar time.

---

## 4. Feature-by-feature: does Rust actually help *this* kind of code?

| Component type | Rust advantage | Verdict |
|---|---|---|
| Graph IR / optimizer / partitioner | Sum types + exhaustive `match` model op-graphs perfectly; refactors are compiler-checked | **Strong Rust win** (see `onnx-runtime-ir`, `-optimizer`) |
| KV cache / scheduler / batching | Ownership models buffer lifetimes and aliasing; 0 `unsafe`, no data races | **Strong Rust win** (`onnx-genai-kv`: 0 unsafe) |
| Session / EP glue / C-ABI | Must speak C ABI → `unsafe`, but contained; safe wrappers above it | **Rust win, with a caveat** |
| Sampling / detokenize / serving | Async + `Result` error handling + no GC pauses | **Rust win** |
| Elementwise / shape / movement kernels | Safe Rust is competitive and far more maintainable | **Rust win** (`onnx-runtime-ep-cpu`) |
| GEMM / conv / attention hot kernels | No mature Rust ecosystem; vendor libs win | **C++ / reuse via FFI** |
| CUDA / Metal device kernels | Written in CUDA/Metal regardless of host language | **Neutral** (host glue can be Rust) |

The pattern: **Rust wins the orchestration and correctness-critical logic; C++/vendor kernels win the
raw FLOPs.** A good architecture puts the Rust/kernel boundary exactly at the FFI line — which is what
we already do.

---

## 5. The `unsafe` question, answered honestly

The reflexive objection is: "you still write `unsafe`, so what did safety buy you?" Two answers:

1. **Containment is the point.** 452 `unsafe` blocks in 119k lines means **>99.6% of the code is
   compiler-verified memory-safe**, and the dangerous 0.4% is greppable, reviewable, and lives at
   named boundaries (`onnx-runtime-capi`, `onnx-genai-ort`, `onnx-runtime-ep-cuda`). In C++,
   *100%* of the code is in the danger zone; there is no `unsafe` keyword to grep for because there is
   no safe subset. Reviewers can spend their attention where it matters.

2. **The unsafe you keep is honest about its contract.** When we mis-used an FFI contract (int32 vs
   int64 read), the bug was in an `unsafe fn` whose job is exactly that boundary. The fix was local
   and the blast radius was one function. That is the whole value proposition: not zero danger, but
   *localized, labeled* danger.

---

## 6. A migration that doesn't bet the company

This is the part to actually argue about. The plan is **strangler-fig, not big-bang**:

1. **New components start in Rust.** KV cache, scheduler, engine, server, router, sampling — done, in
   Rust, today. This is free: you were going to write them anyway.
2. **Wrap, don't rewrite, ORT first.** `onnx-genai-ort` binds the existing ORT via its C ABI. You get
   ORT's kernels and EPs immediately, with a safe Rust surface on top. (This is exactly how the MLX EP
   ships: stock ORT + a Rust plugin.)
3. **Reimplement the runtime layer behind the ORT C API** (`nxrt` / the `onnx-runtime-*` crates), one
   piece at a time: IR → optimizer → partitioner → memory planner → session. Each piece is validated
   against ORT's own conformance tests. Because it presents the same C API, consumers don't notice.
4. **Reuse kernels via FFI; replace selectively.** Call MLAS/CUDA from Rust. Rewrite a kernel only
   when you have a measured win (e.g. the MLX path, where there was no ORT kernel at all on Apple
   Silicon — so Rust wasn't competing with a mature kernel, it was the *only* option).
5. **Plugin EPs are the wedge.** A Rust EP loads into *unmodified* ONNX Runtime. That means you can
   ship Rust into a C++ ORT deployment with zero rewrite, prove it in production, and expand from
   there. We already did this end-to-end.

At every step the system is shippable, ORT-compatible, and reuses the expensive kernels. Nobody has to
approve a two-year rewrite; they approve one crate at a time, each with a conformance gate.

---

## 7. Recommendation, and what would change it

**Recommendation:** default new onnx-genai components to Rust; continue the incremental `nxrt`
runtime behind the ORT C API; keep and reuse C++/CUDA kernels through FFI indefinitely, replacing them
only on measured evidence. Treat "rewrite a hot kernel in Rust" as guilty-until-proven, and "write the
orchestration in Rust" as the default.

**What would change this recommendation (intellectual honesty):**
- If a Rust runtime piece cannot match ORT's latency within a small margin on our conformance suite,
  it stays a C++/FFI wrapper. We measure, we don't assume.
- If FFI overhead at the Rust↔kernel boundary shows up in decode-step profiles, that boundary gets
  redrawn or the hot path stays native.
- If team velocity drops because of Rust ramp-up rather than rising after it, we slow the migration.

The bet is not "Rust is faster than C++." It is: **for the 80% of this system that is orchestration
and correctness-critical state, Rust removes an entire category of bug at the cost of some `unsafe` at
the edges and some compile time — and for the 20% that is raw kernels, we keep using the best C/C++/CUDA
we can get.** The receipts in §1 say we can execute that split, because we already have.

---

### Appendix: reproduce the numbers

```bash
# Rust LOC across the workspace
find crates -name '*.rs' -not -path '*/target/*' | xargs wc -l | tail -1
# unsafe blocks
grep -rn 'unsafe' crates --include='*.rs' | grep -v test | wc -l
# tests
grep -rn '#\[test\]\|#\[tokio::test\]' crates --include='*.rs' | wc -l
# per-crate unsafe (shows the boundary concentration)
for c in onnx-runtime-ir onnx-runtime-optimizer onnx-genai-kv onnx-runtime-capi onnx-runtime-ep-cuda; do
  echo "$c: $(grep -rn unsafe crates/$c --include='*.rs' | wc -l) unsafe"; done
```
