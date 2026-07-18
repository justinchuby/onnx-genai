# C++ vs Rust for ONNX Runtime — an honest assessment

**Audience:** the ONNX Runtime maintainers and engineers at Microsoft who own the C++ codebase and
will (rightly) be skeptical of "rewrite it in Rust."
**Purpose:** make the *honest* case — where Rust would demonstrably help ORT, where C++ is still the
right tool, and how adoption can be incremental and low-risk instead of a religious big-bang rewrite.

This is deliberately not a hype piece. If you are an expert, hype is what makes you distrust the
argument. Everything below is grounded in a **working existence proof**: `onnx-genai`, an independent
Rust reimplementation of the ONNX Runtime GenAI stack (and a partial `onnx-runtime-*` runtime layer),
plus `onnxruntime-ep-mlx`, a **pure-Rust ONNX Runtime plugin execution provider that loads into stock,
unmodified ORT**. Those two codebases are used here only as *evidence that the approach is executable* —
not as products anyone is asking ORT to adopt. Every number below is reproducible (see appendix).

---

## 0. TL;DR

- This is **not** a proposal to rewrite ORT's MLAS/CUDA kernels in Rust. That would throw away a
  decade of hand-tuned work. Keep and *reuse* the fast C/C++/CUDA kernels through FFI.
- It **is** an argument that the *orchestration* layers — IR, graph optimizer, partitioner, memory
  planner, session/EP glue, and the GenAI stack (KV cache, batching, sampling, serving) — are a
  natural fit for Rust, and that new components and new EPs are a low-risk place to start.
- The evidence that this is executable: a proof-of-concept of **~119k lines of Rust across 27 crates,
  1468 tests**, including a pure-Rust CPU EP with real kernels, a pure-Rust paged-KV cache, and a
  pure-Rust ORT plugin EP published to crates.io + PyPI that runs under an unmodified ONNX Runtime.
- The honest cost: **452 `unsafe` blocks**, almost all at C-ABI / CUDA boundaries, plus Rust's
  compile times and a less mature ML-kernel ecosystem.
- **Agentic development** is a first-class consideration here (§6): Rust's compiler-as-oracle feedback
  loop makes AI agents converge fast and blocks whole classes of latent bug — but today's agents are
  more fluent in C++ and in the existing ORT idioms. Both effects are real.

If you read nothing else, read §3 (where C++ still wins), §6 (agentic development), and §7 (the
adoption path that doesn't bet the runtime).

---

## 1. The existence proof (the receipts)

These numbers come from `onnx-genai` + `onnxruntime-ep-mlx` — an independent, working reimplementation
built to test exactly this question. They are offered as *proof the approach compiles, runs, tests,
and ships*, not as code for ORT to consume.

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

### Concrete capabilities the experiment covers

- A **pure-Rust ORT plugin EP** for Apple MLX: it implements the ORT plugin-EP C ABI, is loaded by an
  *unmodified* ONNX Runtime, claims ~130 op types, and covers `MatMulNBits`, `GroupQueryAttention`,
  `PagedAttention` (block-paged KV, packed var-length batches — CUDA-only in ORT itself, so Rust is
  the *only* way it runs on Apple Silicon), quantized Mixture-of-Experts (`QMoE`), `GatherND`, and a
  compiled-subgraph fusion path. Published to crates.io and PyPI via OIDC trusted publishing.
- A **paged-KV cache** with per-page quantization, single-token-append invariants, and 56 tests —
  0 `unsafe`.
- A **CPU execution provider with real kernels** written in safe Rust.
- A graph **optimizer**, **shape-inference**, and **partitioner** as ordinary, testable Rust modules.

None of this is theoretical. It is compiled, tested, and (for the EP) released to public registries by
one person as a side experiment — which is itself a data point about the approach's tractability.

---

## 2. Where Rust demonstrably helped (from building the experiment)

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
   version is incremental and reuse-heavy (see §7).

6. **The ORT team's expertise is in C++.** ORT's deepest experts are C++ experts with years of context
   in *this specific codebase*. Ramping a team on Rust's borrow checker, async, and macro ecosystem is
   a real cost paid in calendar time — and an agent (see §6) working inside the existing C++ ORT has
   far more precedent to draw on than in a greenfield Rust runtime.

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
raw FLOPs.** A good architecture puts the Rust/kernel boundary exactly at the FFI line — which is
precisely where the experiment draws it.

---

## 5. The `unsafe` question, answered honestly

The reflexive objection is: "you still write `unsafe`, so what did safety buy you?" Two answers:

1. **Containment is the point.** 452 `unsafe` blocks in 119k lines means **>99.6% of the code is
   compiler-verified memory-safe**, and the dangerous 0.4% is greppable, reviewable, and lives at
   named boundaries (`onnx-runtime-capi`, `onnx-genai-ort`, `onnx-runtime-ep-cuda`). In C++,
   *100%* of the code is in the danger zone; there is no `unsafe` keyword to grep for because there is
   no safe subset. Reviewers can spend their attention where it matters.

2. **The unsafe you keep is honest about its contract.** When the experiment mis-used an FFI contract
   (int32 vs int64 read), the bug was in an `unsafe fn` whose job is exactly that boundary. The fix was
   local and the blast radius was one function. That is the whole value proposition: not zero danger,
   but *localized, labeled* danger.

---

## 6. Agentic development: which codebase is easier for AI agents?

This is no longer a footnote. A growing share of ORT changes will be written or drafted by coding
agents, and the language's ergonomics *for an agent* now matter as much as for a human. This section
is grounded in direct evidence: the entire experiment above — `PagedAttention`, `QMoE`, `GatherND`,
the 129-predicate `ClaimResult` refactor, a crate-wide clippy sweep, an env-var namespace rename —
was implemented by an AI agent in a tight edit→check loop.

**Where Rust is markedly better for agents:**

1. **The compiler is a precise, machine-readable oracle.** `cargo check` / `clippy --message-format=json`
   emit structured diagnostics with exact spans *and suggested fixes*. An agent's wrong guess usually
   **fails to compile with a pointer to the exact fix**, and it self-corrects in the next loop
   iteration. In C++ the same class of mistake frequently *compiles* and fails at runtime as UB or a
   segfault — which gives an agent **no actionable signal** (a stack-less abort is nearly useless
   feedback). Concretely, in this experiment the compiler caught agent mistakes — an unstable
   expression-attribute (`E0658`), a bad `if let` collapse, borrow/lifetime errors, non-exhaustive
   `match` — each as a localized error the agent fixed immediately. The only *hard* crashes were in the
   C++ MLX library and one FFI-contract slip; the safe-Rust logic gave the agent no UB to chase.

2. **"If it compiles, the change is complete."** When an agent renames or re-types an API, the compiler
   enumerates *every* call site. The agent gets a complete, verifiable worklist and knows when it is
   done. In C++, silent breakage, link-time errors, and ODR give an agent an incomplete and often
   misleading picture — the failure surfaces far from the edit.

3. **One uniform, scriptable toolchain.** `cargo build/test/clippy` is the same everywhere, with
   parseable output. An agent doesn't have to reverse-engineer CMake/Bazel targets, generated headers,
   or include paths to know how to build and test. Fewer files per change, too — no `.h`/`.cpp` to keep
   in sync, no forward declarations.

4. **`unsafe` is a labeled blast radius.** An agent (and the human reviewing the agent's PR) can `grep`
   for exactly the regions that need scrutiny. C++ has no safe subset to exclude, so review attention
   can't be focused — a real problem when triaging machine-written diffs at volume.

5. **Clippy is a built-in senior reviewer.** It flags non-idiomatic and footgun patterns in agent
   output automatically, before a human looks.

6. **Bugs are caught, not shipped.** An agent lacks full global context by construction. In C++ that
   makes it prone to introducing latent use-after-free / aliasing bugs that pass review and tests; in
   Rust those simply do not compile. In this experiment, a speculative optimization that would have
   been a silent latent bug in C++ instead surfaced as a *reproducible* `mlx_compile` abort, was caught
   by the conformance suite, and was cleanly reverted — exactly the failure mode you want with
   machine-authored change.

**Where C++ is better for agents (honest):**

1. **Training-data asymmetry.** Agents have seen vastly more C++ than Rust, and specifically far more
   *ORT* C++. An agent modifying the existing ORT codebase has enormous in-distribution precedent;
   the same agent in a greenfield Rust runtime has less to pattern-match against. For incremental
   changes *inside today's ORT*, C++ is the agent's native habitat.
2. **The borrow checker can force non-local restructuring.** Lifetime and aliasing errors sometimes
   require refactoring beyond the edit site — agents can thrash on these the same way juniors do.
   Macro-heavy code and elaborate trait bounds also degrade agent accuracy.
3. **Compile/link latency slows the loop.** A cold Rust workspace build is minutes; that lengthens the
   agent's edit→check cycle (though `cargo check` and per-crate builds mitigate it).

**Net for agents:** for orchestration and correctness-critical logic, Rust's tight, precise,
machine-readable feedback loop plus memory safety make it an *excellent* target for agentic
development — agents converge fast and cannot ship whole categories of latent bug. The countervailing
force is that today's agents are simply more fluent in C++ and in the existing ORT idioms, which
matters most for incremental work inside the current codebase. That asymmetry shrinks over time; the
compiler-as-oracle advantage does not.

---

## 7. An adoption path that doesn't bet the runtime

This is the part to actually argue about. The path is **strangler-fig, not big-bang**:

1. **New components start in Rust.** KV cache, scheduler, engine, server, router, sampling — the
   experiment shows these are comfortable in Rust today, and they're greenfield anyway.
2. **Wrap, don't rewrite, ORT first.** Bind the existing ORT via its C ABI (the experiment's
   `onnx-genai-ort` does this). You get ORT's kernels and EPs immediately, with a safe Rust surface on
   top — exactly how the MLX EP ships: stock ORT + a Rust plugin.
3. **Reimplement the runtime layer behind the ORT C API**, one piece at a time: IR → optimizer →
   partitioner → memory planner → session. Each piece is validated against ORT's own conformance
   tests. Because it presents the same C API, consumers don't notice.
4. **Reuse kernels via FFI; replace selectively.** Call MLAS/CUDA from Rust. Rewrite a kernel only on a
   measured win (e.g. the MLX path, where there was no ORT kernel at all on Apple Silicon — so Rust
   wasn't competing with a mature kernel, it was the *only* option).
5. **Plugin EPs are the wedge.** A Rust EP loads into *unmodified* ONNX Runtime. You can ship Rust into
   a C++ ORT deployment with zero rewrite, prove it, and expand from there. The experiment does this
   end-to-end.

At every step the system is shippable, ORT-compatible, and reuses the expensive kernels. Nobody has to
approve a multi-year rewrite; they approve one crate at a time, each behind a conformance gate.

---

## 8. Recommendation, and what would change it

**Recommendation (for discussion, not a mandate):** treat Rust as the default for *new* orchestration
and GenAI components and for *new* plugin EPs; where the runtime is reimplemented, do it incrementally
behind the ORT C API; keep and reuse C++/CUDA kernels through FFI indefinitely, replacing them only on
measured evidence. Treat "rewrite a hot kernel in Rust" as guilty-until-proven, and "write the
orchestration/EP glue in Rust" as the low-risk default.

**What would change this recommendation (intellectual honesty):**
- If a Rust runtime piece cannot match ORT's latency within a small margin on the ORT conformance
  suite, it stays a C++/FFI wrapper. Measure, don't assume.
- If FFI overhead at the Rust↔kernel boundary shows up in decode-step profiles, that boundary gets
  redrawn or the hot path stays native.
- If agent/human velocity drops because of Rust ramp-up rather than rising after it, slow down.

The bet is not "Rust is faster than C++." It is: **for the majority of this system that is
orchestration and correctness-critical state, Rust removes an entire category of bug — and is a
strong substrate for agentic development — at the cost of some `unsafe` at the edges and some compile
time; and for the raw kernels, keep using the best C/C++/CUDA available.** The receipts in §1 say one
person (plus an agent) can execute that split, because the experiment already did.

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
