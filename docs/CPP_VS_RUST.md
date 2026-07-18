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
- The evidence that this is executable: a proof-of-concept of **~204k lines of Rust across 31 crates,
  2,200+ tests**, including a pure-Rust CPU EP with real kernels, a pure-Rust paged-KV cache, and a
  Rust-authored ORT plugin EP published to crates.io + PyPI that runs under an unmodified ONNX Runtime.
- The honest cost: **~650 `unsafe` occurrences**, almost all at C-ABI / CUDA boundaries, plus Rust's
  compile times and a less mature ML-kernel ecosystem.
- **Agentic development** is a first-class consideration here (§6–§8): Rust's compiler-as-oracle
  feedback loop makes AI agents converge fast and blocks whole classes of latent bug, it makes
  human-defined contracts *machine-enforceable* for agent implementations, and it shrinks what a
  reviewer must actually check — but today's agents are more fluent in C++ and in the existing ORT
  idioms. All of these are real.

If you read nothing else, read §3 (where C++ still wins), §6–§8 (agentic development, contract-first
workflow, and reviewing agent-authored Rust), §9 (composability / à-la-carte reuse for collaborators),
§10 (how the biggest C++ projects already adopt Rust without breaking users), and §11 (the adoption
path that doesn't bet the runtime).

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

- A **Rust-authored ORT plugin EP** for Apple MLX (its compute calls the native MLX C API): it
  implements the ORT plugin-EP C ABI, is loaded by an *unmodified* ONNX Runtime, claims ~130 op types,
  and covers `MatMulNBits`, `GroupQueryAttention`, `PagedAttention` (block-paged KV, packed var-length
  batches — CUDA-only in ORT itself, so this EP is currently the only way to run it under ORT on Apple
  Silicon; MLX/Metal does the compute, and a C++ plugin could do the same — the point is the path didn't
  exist, not that only Rust could build it), quantized Mixture-of-Experts (`QMoE`), `GatherND`, and a
  compiled-subgraph fusion path. Published to crates.io and PyPI via OIDC trusted publishing.
- A **paged-KV cache** with per-page quantization, single-token-append invariants, and 56 tests —
  0 `unsafe`.
- A **CPU execution provider with real kernels** written in safe Rust.
- A graph **optimizer**, **shape-inference**, and **partitioner** as ordinary, testable Rust modules.

### Extensibility: the rewrite adds what OGA doesn't have

The strongest test of "too model-specific / not extensible" is whether the successor onboards
*new capabilities* fast. `onnx-genai` — the standard-driven Rust reimplementation of OGA — is being
built by an **agent team** and is already reaching past OGA's surface into modalities OGA does not
support: **diffusion, multimodal (vision + audio), and audio-to-audio** pipelines, added as
declared-metadata + trait-plugin extensions rather than new bespoke model dispatch. This is the
extensibility argument in practice: because behavior is derived from declared inference metadata and
à-la-carte traits (see §9), a new architecture or modality is mostly *declaration + fixture + a plugin*,
and it is tractable for agents to add. (Scope is honest: these are in-progress capabilities in a
personal experiment, not a shipped, conformance-gated product — but the point is the *shape* of
extension, which is what OGA's per-model-family style makes hard.)

None of this is theoretical. It is compiled, tested, and (for the EP) released to public registries by
one person as a side experiment — which is itself a data point about the approach's tractability.

### Measured performance vs onnxruntime-genai (OGA)

Early signs are that the Rust reimplementation is **not** paying a decode tax versus OGA — and on CUDA
is ahead — but the *clean* runtime-to-runtime numbers are still being collected. The comparisons below
use **Foundry Local** as the OGA proxy (its decode path *is* `OgaGenerator::GenerateNextToken` /
onnxruntime-genai), and **Foundry Local runs as a daemon/server, so its HTTP figures carry server
overhead that this document should not launder into a pure engine-vs-engine result.** Treat these as
*indicative*, not definitive:

- **CPU (Apple M1 Max, Qwen2.5-0.5B int4).** Warm decode roughly parity on short, ahead on long
  (~175 vs ~160 tok/s) after the fp32-GQA shared-KV fix; feeding OGA's *own* `model.onnx` through our
  runtime reproduces parity, so the delta is the runtime, not the model.
  (`docs/benchmarks/2026-07-13-foundry-local-analysis.md`.)
- **CUDA (H200, Qwen2.5-0.5B int4).** Ahead of Foundry Local / onnxruntime-genai CUDA on decode, TTFT,
  and total in the measured run. (`docs/benchmarks/2026-07-13-H200-cuda-onnxgenai-vs-ollama-foundry.md`.)

> **[PLACEHOLDER — clean OGA head-to-head, to be filled in]**
> Definitive numbers will compare **onnx-genai against onnxruntime-genai's C API directly** (no Foundry
> Local server in the loop), on the **same ORT build, model, EP, tokenizer, sampling, and hardware**,
> reporting TTFT and inter-token latency (median + p95/p99), tokens/sec under concurrent load, and peak
> memory — on **CPU and CUDA**. Author's summary of current results: *CPU decode already exceeds OGA;
> CUDA is on par (fuller CUDA numbers to be added).* These land here once the server-overhead-free
> harness is run; see the bake-off methodology in §12.

**Honest scope:** one model family, two machines, batch-1 greedy decode, an ORT-version delta (FL 1.26 /
ours 1.27), and a server-mediated OGA proxy. The narrow, defensible claim today is: *where it has been
measured, the Rust orchestration layer meets-or-beats OGA on decode and is ahead on CUDA* — not "faster
everywhere." The clean bake-off in §12 is what turns that from indicative into settled.

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
   look. We spent zero time chasing use-after-free, double-free, or data races in the ~204k lines of
   safe Rust. In an equivalent C++ codebase those are the bugs that eat weeks.

3. **Fearless concurrency for serving.** The MLX EP is thread-affine; the rule "one session per
   thread" is *enforced* by the borrow checker and `Send`/`Sync` bounds, and shared caches are
   `Mutex`-guarded with the compiler refusing to let you forget. The server/scheduler crates lean on
   this: async batching without *data races* — a whole class of heisenbug the compiler rules out.
   (Honestly: this is data-race freedom, not deadlock/starvation/cancellation freedom — those still
   need design and review; see §3.)

4. **`cargo` as the whole build system.** No CMake, no protobuf compiler, no vendored toolchain matrix.
   `cargo build` cross-compiles; `cargo test` runs 2,200+ tests; the EP publishes to two package
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
   version is incremental and reuse-heavy (see §10–§11).

6. **The ORT team's expertise is in C++.** ORT's deepest experts are C++ experts with years of context
   in *this specific codebase*. Ramping a team on Rust's borrow checker, async, and macro ecosystem is
   a real cost paid in calendar time — and an agent (see §6) working inside the existing C++ ORT has
   far more precedent to draw on than in a greenfield Rust runtime.

7. **Modern C++ has closed part of the safety gap.** A fair comparison is against *today's* C++, not
   1990s C++: RAII, `unique_ptr`/`shared_ptr`, `std::span`, `std::variant`/`std::expected`, `gsl`, plus
   AddressSanitizer / ThreadSanitizer / UBSan, `clang-tidy`, and continuous fuzzing catch a large share
   of memory and concurrency bugs. The honest difference is not "C++ is unsafe, Rust is safe" — it is
   *default and coverage*: sanitizers/fuzzers find bugs only on code paths and inputs they actually
   execute (runtime, probabilistic, and off in production builds), and the guidelines are opt-in and
   unenforced; Rust's guarantees are compile-time, total over all paths, and on by default. That is a
   real, but *narrower and more nuanced*, advantage than a naïve pitch implies — and for a team already
   running ASan/TSan/fuzzing well, the marginal safety gain is smaller than for one that isn't. (The
   honest counter-asymmetry: standing up and *continuously running* that tooling — separate sanitizer
   build variants, fuzzing infra, keeping them green in CI — is itself an ongoing build-pipeline cost,
   whereas Rust's guarantees are on by default with no extra pipeline. So the gap is smaller for a
   well-tooled team, but that tooling is not free either.)

8. **Async runtime and thread-pool coexistence is unsolved integration work.** Adopting `async` Rust
   means adopting a runtime (e.g. Tokio) that owns threads — which must then coexist with ORT's and each
   EP's own thread pools and, for GPU EPs, their stream/queue model. Getting this wrong causes
   oversubscription, priority inversion, or executor threads blocked on FFI calls. It is tractable, but
   it is a concrete adoption cost and design question the experiment has only partially exercised, not a
   free lunch.

---

## 4. Feature-by-feature: does Rust actually help *this* kind of code?

**Read the "Verdict" column as a qualitative assessment of *fit and maintainability*, drawn from
building the experiment — not a benchmarked claim of performance parity.** "Rust win" here means the
code was safer to write and refactor and had zero memory-safety defects, not that it out-runs a tuned
C++ equivalent. On raw performance there are now *preliminary* head-to-head numbers vs OGA (see §1:
CPU meets-or-beats, CUDA ahead in the measured runs) — but via a server-mediated proxy (Foundry Local)
and on a narrow model/hardware set, so a *settled* performance claim still needs the clean bake-off in
§12: decode/prefill latency (median + p95/p99), throughput under concurrent load, peak memory, binary
size, cold/warm build time, and model/platform coverage against OGA/ORT. Until that lands, treat each
verdict as "promising fit, with early-but-favorable perf signal, pending the full bake-off."

| Component type | Rust advantage | Verdict (qualitative) |
|---|---|---|
| Graph IR / optimizer / partitioner | Sum types + exhaustive `match` model op-graphs perfectly; refactors are compiler-checked | **Strong Rust fit** (see `onnx-runtime-ir`, `-optimizer`) |
| KV cache / scheduler / batching | Ownership models buffer lifetimes and aliasing; 0 `unsafe`, no data races | **Strong Rust fit** (`onnx-genai-kv`: 0 unsafe) |
| Session / EP glue / C-ABI | Must speak C ABI → `unsafe`, but contained; safe wrappers above it | **Rust fit, with a caveat** |
| Sampling / detokenize / serving | Async + `Result` error handling + no GC pauses | **Rust fit** |
| Elementwise / shape / movement kernels | Safe Rust is maintainable; perf parity for these simple kernels is plausible but unbenchmarked here | **Likely Rust fit** (`onnx-runtime-ep-cpu`) |
| GEMM / conv / attention hot kernels | No mature Rust ecosystem; vendor libs win | **C++ / reuse via FFI** |
| CUDA / Metal device kernels | Written in CUDA/Metal regardless of host language | **Neutral** (host glue can be Rust) |

The pattern: **Rust fits the orchestration and correctness-critical logic; C++/vendor kernels own the
raw FLOPs.** A good architecture puts the Rust/kernel boundary exactly at the FFI line — which is
precisely where the experiment draws it.

---

## 5. The `unsafe` question, answered honestly

The reflexive objection is: "you still write `unsafe`, so what did safety buy you?" Two answers:

1. **Containment is the point.** The ~650 `unsafe` occurrences are *concentrated at FFI boundaries*, not
   spread through the logic. Five C-ABI / kernel crates (`onnx-runtime-ep-cuda`, `onnx-genai-ort`,
   `onnx-runtime-capi`, `onnx-runtime-ep-cpu`, `onnx-runtime-dlpack`) hold roughly 80% of them, while the
   pure-logic crates are near-zero: `onnx-genai-engine` has 2 `unsafe` in ~18k lines, `onnx-runtime-ir` /
   `-optimizer` / `-shape-inference` have 1 each, `onnx-genai-kv` has 0. `unsafe` is greppable and lives
   at named boundaries, so reviewers spend their memory-safety attention where it actually applies. This
   is *not* a claim that the rest is bug-free — safe Rust still has logic, panic, arithmetic, and
   concurrency bugs that need ordinary review — but it does remove the *use-after-free / data-race*
   review burden from the large majority of crates that carry little or no `unsafe`. In C++ there is no
   safe subset to grep for: every line is in scope for that class of review.

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

6. **Whole categories of bug are caught, not shipped.** An agent lacks full global context by construction. In C++ that
   makes it prone to introducing latent use-after-free / aliasing bugs that pass review and tests; in
   Rust those simply do not compile. In this experiment, a speculative optimization that would have
   been a silent latent bug in C++ instead surfaced as a *reproducible* `mlx_compile` abort, was caught
   by the conformance suite, and was cleanly reverted — exactly the failure mode you want with
   machine-authored change.

7. **Runtime failures are *diagnosable*, not undefined.** When Rust logic does fail, a `panic` is a
   *defined* event: a message plus the exact `file:line` (with `#[track_caller]`, `unwrap`/`expect`/index
   panics point at the caller), and a symbolicated backtrace under `RUST_BACKTRACE=1`. It is
   deterministic and reproducible for a given input, and with `panic = "unwind"` it can be caught at an
   FFI/request boundary (`catch_unwind`) — turning a would-be crash into a logged, handled error while
   the process stays up. The C++ counterpart to the same mistake (out-of-bounds, use-after-free) is
   *undefined*: it may corrupt silently, crash far from the cause, or segfault with no message — the
   "no actionable signal" problem from point 1, now at runtime. **Honest caveats:** a panic escaping a
   plain `extern "C"` boundary is itself UB (so FFI entry points must wrap in `catch_unwind`, as this
   EP does); `panic = "abort"` builds forgo unwind/catch; and interactive stepping, debugger
   pretty-printers, and especially *async* backtraces are still weaker than C++'s mature tooling (§3).
   Net: for the UB/segfault/corruption class, Rust failures are markedly easier to *diagnose*; for
   interactive debugging depth, C++ tooling still leads.

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

**Evidence status (important):** the above is grounded in *one* agent-built project, not a controlled
comparison. It shows the loop is *feasible and pleasant* in Rust; it does **not** measure that Rust
beats modern ORT C++ (with clang diagnostics, sanitizers, static analysis, and huge in-repo precedent)
as an agent substrate. Read it as a hypothesis. What would actually settle it: a matched study — the
same set of representative tasks implemented by agents in both a Rust crate and ORT C++, measuring
iterations-to-green, wall-clock time, escaped-defect rate, and human review time. That experiment is
worth running before treating "better for agents" as established rather than plausible.

---

## 7. The workflow you actually want: contract-first, agent-implemented, machine-verified

The target operating model is: **a team member specifies the *contract* and the *expected behavior*;
an agent writes the implementation; the toolchain verifies the implementation against the contract.**
Rust is an unusually strong substrate for exactly this division of labor, because in Rust the contract
is *machine-checked*, not aspirational.

- **The contract lives in the type system, not a comment.** A trait signature, a `Result<T, ErrKind>`,
  a newtype that enforces an invariant, an exhaustive `enum`, `#[must_use]` — these are promises the
  agent's body is *forced* to honor. A C++ header declares a signature but cannot stop the body from
  violating the invariant with a `reinterpret_cast` or a raw pointer; a Rust signature constrains what
  the implementation can even express. When the human owns the types, the agent cannot silently break
  the interface.
- **Expected behavior lives in oracles the agent must pass.** In the experiment, `PagedAttention` was
  specified by a numpy port of ORT's *own* reference kernel; the agent iterated the MLX implementation
  until it matched the oracle to ~1e-4 — the reviewer never had to read the MLX plumbing line by line,
  because they defined the acceptance test. `QMoE` used ORT's CPU kernel as ground truth. Op-claim
  validity is a typed `ClaimResult`, so "is this input admissible" is a value the compiler tracks.
- **Verification is mechanical, not a matter of trust.** The agent's PR is not "trust me"; it is "the
  typed contract you wrote compiles, and the behavior you specified passes its oracle" — build +
  `clippy` + the conformance suite. This does *not* prove universal correctness (tests are finite, and
  the types encode only the invariants you chose to express), but it does mean the declared interface
  holds and the acceptance oracles pass without a human reading every line of the implementation.
- **The agent optimizes *within* the contract.** Choosing the fused-vs-eager path, the chunked
  algorithm, the shapeless-vs-shape-keyed capture — the agent explores the implementation space (this
  is where "optimize per the latest research" happens) while the human owns the interface and the
  acceptance oracle. The contract is the guardrail that makes autonomous exploration safe.

This maps cleanly onto contract/spec-first development with property-based testing — a workflow that is
*more* effective in a language where the compiler can reject any implementation that doesn't fit the
declared shape.

---

## 8. "Can we review Rust we don't know?" — the reviewability concern, honestly

A colleague's worry — *"I'm not fluent in Rust, so I can't effectively review it"* — is the right
question to ask about machine-authored code at volume. Here is the honest answer, both sides.

**Why the concern is smaller than it feels:**

1. **The review model changes.** In *safe* Rust the compiler already guarantees the absence of
   use-after-free, double-free, data races, buffer overflows, and uninitialized reads. That entire
   category — which consumes a large fraction of C++ review attention — is simply *off the table*.
   The reviewer's budget shifts to the one thing that matters: **does this implementation satisfy the
   contract and the expected behavior you specified?** That is a higher-level review, anchored on the
   contract the reviewer themselves wrote — and it is largely language-agnostic.
2. **The part that truly needs Rust+FFI expertise is small and labeled.** It's the `unsafe` blocks —
   ~0.4% of the code, greppable, at named boundaries. Point your scarce deep-Rust review there and let
   the compiler own the rest.
3. **Gate before review, not during.** Require agent PRs to be green build + `clippy`-clean +
   conformance-passing in CI. The human then reviews a *pre-validated* diff; the machine has already
   rejected the mechanical mistakes.
4. **Reading Rust is much easier than writing it.** A reviewer never fights the borrow checker — the
   compiler already resolved the lifetimes. Explicit types, `Result`, and exhaustive `match` make
   intent unusually legible; idiomatic Rust is arguably *more* reviewable than template/overload-heavy
   modern C++, where the behavior of a line can depend on invisible ADL and SFINAE.
5. **Agent-assisted review closes the fluency gap now.** The same compiler/clippy signal that guides
   the authoring agent, plus an agent that summarizes a diff, explains a lifetime, or answers "does
   this satisfy the contract," lets a C++-fluent engineer review Rust effectively *today* while ramping
   their own fluency.

**The honest costs:**

- Reviewing `unsafe` FFI genuinely requires Rust+FFI skill. It's a small, concentrated capability, but
  it must exist on the team.
- A reviewer new to Rust idioms *is* slower at first. This is a real ramp cost — mitigated by the
  shrunken surface (points 1–2) and the automated gates (point 3), but not zero.
- Macro-heavy or trait-bound-heavy code can be opaque to a newcomer. That's a reason to hold
  agent-authored code to an idiomatic, macro-light standard (enforced by `clippy` and review norms),
  which is good practice regardless.

**Net:** the thing you *can't* review well in unfamiliar Rust — memory safety — is exactly the thing
the compiler already reviewed for you. The thing you *must* review — does the implementation meet the
contract and behave correctly — is language-agnostic and anchored on the contract you wrote. The
unfamiliarity cost is real, but it is concentrated in the small `unsafe` surface and the initial ramp,
not spread across every line the way C++'s memory-safety review burden is.

---

## 9. Composability: à la carte components and user plugins

A core goal is that collaborators and users can **reuse the pieces they want and replace the pieces
they want** — take the IR crate just to do graph fusion on ONNX models, take the ORT or GenAI crate as
a runtime, use the built-in sampler or drop in their own, plug in a custom tokenizer or logit
processor — and contribute their own optimized components where they have an edge. Rust makes this
substantially easier for Rust consumers, and no worse for everyone else.

**Why Rust helps:**

- **Cargo + crates.io is frictionless library reuse.** Every component is an independently versioned
  crate. A collaborator who only wants graph fusion writes `onnx-runtime-ir = "…"` (+ `-optimizer`) in
  `Cargo.toml` and has it — no vendoring, no CMake, no ABI matching, no system deps, no "which ORT
  build did you link against" (this holds for the pure-Rust crates like `-ir`/`-optimizer`; crates that
  bind ORT/CUDA/MLX still carry those native toolchain requirements). The experiment already proved the
  publish path end-to-end: the MLX EP is on crates.io, installable with `cargo add`. The C++ story is
  heavier — consume ORT and you link (or vendor) the monolith and match its ABI and compiler.
- **Fine-grained crates = take only what you need.** The workspace is 31 crates with real boundaries
  (`onnx-runtime-ir`, `-memory`, `-optimizer`, `-ep-cpu`, `-ep-cuda`, `onnx-genai-kv`, `-scheduler`,
  `-engine`, `-preprocess`, `-router`…). Depend on the three you use; the rest never enters your build.
  Feature flags (`default-features = false` + opt-in) trim further — a CUDA-free build, a tracer-free
  build. This cuts the *other* way too, and it's a lived contrast from building the experiment:
  **modular crates make a subsystem extractable and reusable by construction**, whereas pulling a
  coherent module *out* of the ORT C++ monolith — shared headers, template entanglement, a coupled
  build graph — is real surgery. "Reuse just the optimizer" is a one-line dependency here; in the C++
  codebase it's a detangling project.
- **Traits are clean, zero-cost plugin points — and they already exist here.** "Bring your own
  component" is a trait the user implements. The experiment already exposes `Sampler`, `LogitProcessor`,
  `Constraint`, `TokenEmbedder`, `LmHead`, and `SpeculativeProposer` as public traits: a user writes
  `impl Sampler for MySampler` and drops it in. It's compile-time-checked against the trait; depending
  on how the extension point is wired it is either monomorphized (static dispatch, no runtime cost) or a
  cheap `Box<dyn _>` trait object (one vtable indirection, comparable to a C++ virtual call — the
  experiment uses trait objects for `LogitProcessor`/`Constraint`/`Sampler`), and it needs no
  registration boilerplate. Note these are *source-level* extension points (you rebuild with your impl),
  not dynamically-loaded ABI plugins. Users provide custom-optimized components by
  implementing the trait, and the compiler guarantees they fit the contract.
- **A source-level, semver'd, documented API.** `cargo doc` + crates.io semver give consumers a stable,
  discoverable surface instead of a C++ header they must match to a compiler and an STL ABI — no
  symbol-mangling or ABI-skew failures at link time.
- **Strong external precedent.** The two most widely reused pieces of the modern LLM stack —
  HuggingFace `tokenizers` and `safetensors` — are *Rust libraries* consumed everywhere (Python, JS,
  Rust). Rust components getting adopted à la carte across an ecosystem is not hypothetical; it is how
  today's tokenizer and model-format layers already work.

**Honest caveats:**

- **Rust source-level reuse is for Rust consumers.** A collaborator writing in C++/Python cannot `impl`
  a Rust trait; they consume the same **C ABI** the runtime already exposes (`onnx-runtime-capi`, the
  plugin-EP ABI). So Rust makes reuse *much* easier for Rust users and *no harder* for cross-language
  users — but the cross-language plugin story is the C ABI, exactly as today (and it works: the MLX EP
  is a C-ABI plugin loaded by unmodified C++ ORT). Python bindings (PyO3) can re-export the ergonomic
  pieces, which is how `tokenizers`/`safetensors` reach Python.
- **Pre-1.0 semver churns.** While the crates are `0.x`, breaking API changes are frequent and
  consumers must pin versions. A *stable* third-party plugin ecosystem needs a committed-stable
  interface — a stabilized trait or the C ABI — the same discipline ORT already applies to its C API.
- **Cargo builds from source.** Consumers compile the crate and its deps (fast for pure-Rust libs like
  the IR/optimizer; the usual FFI build cost for crates with C kernels). Prebuilt-binary distribution
  is possible but less turnkey than dropping in a prebuilt `.so`.

**Net:** for the "pick the components that fit you, replace the ones where you have a better idea" goal,
Rust's crate granularity + traits + cargo make **Rust-native reuse dramatically more ergonomic** — with
real traits already in place and strong ecosystem precedent — while cross-language consumers keep the
C-ABI path that already works.

---

## 10. Precedent: how the biggest C++ projects adopt Rust *without breaking users*

This is the concern that actually matters for us: **not breaking existing users, and staying compatible
with their build process.** It is worth being precise, because the largest C++ codebases on earth have
already solved exactly this, and their playbook is the one we're proposing — not a leap of faith.

**Chromium / Microsoft Edge.** Chromium began landing production Rust in 2023. The mechanics are the
point:
- Rust enters the **existing GN/Ninja build** as ordinary library targets (`rust_static_library`)
  that C++ `static_library` targets depend on. There is no separate build the downstream must adopt;
  Ninja links the Rust output like any other object file.
- Interop goes through **CXX** (a binding generator that produces checked, bidirectional glue and
  eliminates a class of hand-written FFI mismatch) at a reviewed boundary. CXX reduces the boilerplate
  and the type-mismatch surface; it does not by itself validate a falsely-declared foreign contract or
  make arbitrary C++ internals safe, and it is not a versioned cross-version dynamic ABI (that role
  belongs to a deliberately-stable C ABI).
- The policy is explicitly **interop-only / new-code-first**: Rust is *not* used to force-rewrite
  established C++. No top-level feature is Rust-only, and integration "must not significantly slow
  existing build workflows."
- **Edge is built on Chromium and inherits all of this.** It already ships upstream Rust (e.g. font and
  image-decoding paths) into a browser used by hundreds of millions — and no user, and no downstream
  build, had to change anything. That is the existence proof for "you can put Rust into a giant, shipping
  C++ product invisibly."

**Windows.** Microsoft landed the **first Rust in the Windows kernel** (the Win32k graphics subsystem —
`win32kbase_rs.sys`) and in **DWriteCore**, dropped into the existing Windows build and binary-compatible
with everything around them; users never saw a seam. The motivation is MSRC's finding that **~70% of
Microsoft CVEs are memory-safety bugs** — the same category safe Rust removes by construction. Again:
incremental interop, no rewrite, no user-visible break.

**Android.** Google's data is the most useful, because it quantifies *why new-code-first works*: the
share of Android's memory-safety vulnerabilities fell from **~76% (2019) to ~24% (2024)** — achieved
**not by rewriting old C/C++**, but by writing new and refactored code in Rust alongside it.
Vulnerabilities concentrate in *new* code and decay as code ages, so shifting only new code to a safe
language captures most of the safety benefit without touching the mature codebase.

### The most instructive data point: Firefox migrated its *own* engine to Rust — yet reuses ONNX Runtime in C++

Firefox is the strongest possible test of "shouldn't a Rust shop just write it in Rust?" — Mozilla
*created* Rust, and Gecko already contains large Rust subsystems (Stylo, WebRender). Yet for its
on-device ML features Firefox **reuses ONNX Runtime rather than writing a Rust inference engine.** It
started with ORT compiled to WebAssembly (via Transformers.js) and in 2024–2025 moved to a **native C++
ONNX Runtime backend**, vendored as precompiled libraries and exposed to the JS front-end through a thin
WebIDL shim — a 2–10× speedup, transparent to the calling code. The ML backend is explicitly C++; the
orchestration around it is JS/Transformers.js. That fact cuts both ways, and both directions matter:

- **It validates the "keep and reuse the kernels/runtime" half — strongly.** The people with the deepest
  Rust expertise on earth chose to reuse a mature C++ inference runtime rather than reimplement one — and
  when they optimized, they moved *toward* native C++ ORT, not away from it. Reimplementing a kernel-heavy
  runtime *for its own sake* is exactly the guilty-until-proven case §3 and §13 warn against. If ORT
  already runs your model well, the highest-value move is to *wrap* it, not re-derive MLAS.
- **It is the honest counterweight to Phase 2 (incrementally porting the runtime to Rust).** If Mozilla
  didn't rewrite ORT, we cannot justify porting the runtime on "Rust is nicer." Any runtime seam must
  earn its way on measured value and long-term (agent-)maintainability, one conformance-gated piece at a
  time — and for many consumers the right answer is *"never; keep wrapping ORT."* That is already the
  plan; Firefox is the reason to hold that line.
- **It also shows the integration pattern we're proposing, in production.** Firefox vendors the C++
  runtime behind a thin, stable shim (WebIDL there; the C ABI / plugin-EP boundary here) so the compute
  stays reused and unmodified while the surrounding code evolves independently. That is exactly the split
  proposed here — reuse ORT for compute behind a stable boundary, and keep the new correctness-critical
  orchestration code in a memory-safe language.
- **The workload that pulls all of this is on-device, interactive inference.** Firefox's ML push is
  entirely local (privacy, no server round-trip) — the same class of workload (concurrent,
  latency-sensitive, on-device) where the orchestration layer (KV/memory management, scheduling,
  sampling) is a natural place for a safe, concurrent Rust substrate, without touching the reused kernels.
  (Firefox doesn't itself prove *where* the bottleneck sits; that's a claim we'd back with our own
  decode-step profiles, not borrow from Mozilla.)

#### How Firefox *itself* migrated to Rust (the "Oxidation" playbook)

The same organization is also the best-documented example of incrementally moving a giant, shipping C++
codebase (Gecko, ~tens of millions of lines) to Rust — and the method is exactly the strangler-fig
approach this document argues for. The sequence:

- **Incubate risky work out-of-tree first.** Rust and the **Servo** research engine (a from-scratch,
  all-Rust browser engine) were the proving ground. Components were hardened in Servo, then transplanted
  into Gecko — Firefox never bet the shipping product on unproven code. *(Our analog: `onnx-genai` /
  `onnxruntime-mlx` are the out-of-tree experiment; proven pieces graduate into the mainline.)*
- **Start with a small, self-contained, security-sensitive component.** The first Rust to ship in
  Firefox was the **MP4 metadata parser, in Firefox 48 (Aug 2016)** — a leaf that parses *untrusted*
  media, a classic memory-safety hotspot, with a narrow interface. Low blast radius, high safety payoff,
  easy to A/B. *(Our analog: a plugin EP, or a leaf orchestration crate — the MLX EP is precisely this
  wedge.)*
- **Roll in more parsers/libraries incrementally, coexisting via FFI.** e.g. **`encoding_rs`** (character
  encoding) replaced its C++ counterpart piece by piece, living behind an FFI boundary next to unchanged
  C++. No big-bang cutover.
- **Then take on a major subsystem where Rust's strengths uniquely pay off.** **Stylo / Quantum CSS
  (Firefox 57, Nov 2017)** replaced the CSS style system with a Rust engine that runs styling **in
  parallel across cores**. Mozilla has noted that parallel styling had been attempted in C++ before and
  abandoned because the data races were unmanageable; Rust's compile-time data-race freedom is what
  finally made a *parallel* engine shippable. **WebRender (Firefox 67, May 2019)**, a GPU renderer, came
  from the same Servo lineage. *(Our analog: the concurrent, lifetime-heavy **scheduler / KV-memory /
  continuous-batching** layer is our "Stylo" — the subsystem being redesigned anyway, where fearless
  concurrency is the point.)*
- **Coexistence is permanent, not a deadline.** A decade in, Gecko is still majority C++ with growing
  Rust; the goal was never a full rewrite, only to move the seams where Rust earns its place. That is the
  same posture §13 recommends here.

The honest, load-bearing detail is Stylo: it is a concrete case where Rust's safety didn't just prevent
bugs, it **enabled a capability C++ couldn't ship** (safe parallelism at that scale). That is the
strongest form of the argument — not "Rust is nicer," but "Rust let us build the thing we otherwise
couldn't." Our bet is that an LLM scheduler + paged-KV memory manager is the same kind of
concurrency-heavy subsystem.

Net: Firefox does not weaken the case; it sharpens it. It is a real-world confirmation of the exact
boundary this document draws — **reuse the C++ runtime/kernels through a stable ABI; write the new
orchestration and on-device serving layer in Rust; port the runtime itself only incrementally and only
on evidence.**

### The shared playbook (identical across all of them):
1. Rust enters through the **existing build system** as normal library targets — no new toolchain the
   consumer must adopt (Cargo can emit a `staticlib`/`cdylib` that CMake/MSBuild/GN links like any
   other `.a`/`.lib`/`.so`; `cxx`/`cbindgen` generate the headers).
2. A **reviewed interop boundary** — CXX (generated bindings that cut FFI-mismatch bugs) or, for a
   *versioned, stable* cross-version contract, a plain **C ABI** — keeps callers unchanged.
3. **New-code-first, no forced rewrite;** mature code stays until there's a reason.
4. **Never a user-visible break** — the language underneath is invisible across the boundary.

### Why our transition is *easier* than these precedents

The hard constraint that dominates Chromium/Windows Rust adoption is *"don't disturb the one enormous,
monolithic C++ build that ships to everyone."* We have that constraint on **one** side and not the other:

- **Runtime (ORT) side — as hard as theirs, and solved the same way.** We bind ORT through its **C ABI**,
  exactly as those projects use CXX/C-ABI, and plugin EPs load into *unmodified* ORT (the MLX EP is a
  stock-ORT `.dylib` — no fork, no relink). Consumers keep the same C API and the same build. So even our
  harder half mirrors the proven precedent.
- **GenAI (OGA) side — materially easier, because the giant C++ consumers aren't here.** Edge, Windows,
  and Android — the very codebases whose build compatibility makes Rust adoption delicate — **do not
  depend on onnxruntime-genai at all.** OGA's consumers are a smaller, newer, higher-level set (apps
  calling the GenAI API in C/C++/C#/Python), not OS/browser build pipelines. So the "don't break the
  monolithic C++ build" constraint that shapes the precedents **largely doesn't apply to the GenAI
  layer.** OGA-next can present a compatible API surface, reach parity, and only then deprecate anything —
  without ever threatening a browser or OS build. That makes the **GenAI layer the easiest place to go
  Rust-first**, arguably easier than the precedents, which had to thread Rust into a single giant binary.

### The one real cost the precedents all paid (so should we)

The honest tax is **the Rust toolchain in CI** — bootstrapping `rustc`/`cargo` into hermetic/enterprise
build systems — plus some build-time increase. Chromium gates adoption on "must not significantly slow
existing workflows," and Windows/Android absorbed the toolchain-integration work up front. We should
adopt the same bar: Rust artifacts must link into a consumer's existing build with no new *runtime*
dependency and a bounded *build-time* cost, or the boundary stays where it is.

**Bottom line:** "we don't want to break existing users / we need build-process compatibility" is not a
reason to avoid Rust — it is precisely the requirement Chromium/Edge, Windows, and Android already met
with the strangler-fig + stable-ABI playbook. We inherit that playbook for ORT via the C ABI, and the
GenAI layer is *less* constrained than any of them because its consumers aren't the monolithic C++
builds that made the precedents hard.

---

## 11. An adoption path that doesn't bet the runtime

This is the part to actually argue about. The path is **strangler-fig, not big-bang**:

1. **New components start in Rust.** KV cache, scheduler, engine, server, router, sampling — the
   experiment shows these are comfortable in Rust today, and they're greenfield anyway.
2. **Wrap, don't rewrite, ORT first.** Bind the existing ORT via its C ABI (the experiment's
   `onnx-genai-ort` does this). You get ORT's kernels and EPs immediately, with a safe Rust surface on
   top — exactly how the MLX EP ships: stock ORT + a Rust plugin.
3. **(Hardest tier, explicitly experimental) reimplement *internal* runtime seams behind the ORT C
   API** — IR → optimizer → partitioner → memory planner → session — one at a time. Two honest caveats
   separate this from step 2's proven "wrap ORT": **(a)** presenting the same C API preserves the
   *external* surface, but ORT's internals are not cleanly pluggable seams today, so "swap one piece" is
   real architectural work, not a drop-in; and **(b)** ORT's conformance corpus gates *numerical/op*
   behavior but does not by itself establish parity of threading, allocation, memory planning,
   performance, or platform/model coverage. Treat each seam as a hypothesis with its *own* parity gate
   (numerics **and** latency **and** coverage); several will rightly stay C++/FFI wrappers. Steps 1–2
   and 5 are validated by the experiment; **step 3 is the research bet, not a settled plan** — keep the
   two clearly separate when discussing risk.
4. **Reuse kernels via FFI; replace selectively.** Call MLAS/CUDA from Rust. Rewrite a kernel only on a
   measured win (e.g. the MLX path, where there was no ORT kernel at all on Apple Silicon — so this EP
   is the only current way to run those ops under ORT there; MLX does the compute, and a C++ plugin
   could have too).
5. **Plugin EPs are the wedge.** A Rust EP loads into *unmodified* ONNX Runtime. You can ship Rust into
   a C++ ORT deployment with zero rewrite, prove it, and expand from there. The experiment does this
   end-to-end.

At every step the system is shippable, ORT-compatible, and reuses the expensive kernels. Nobody has to
approve a multi-year rewrite; they approve one crate at a time, each behind a conformance gate.

---

## 12. Proposed experiment: the matched bake-off and its decision gates

The doc should be *falsifiable*, not just persuasive. Here is the concrete experiment that would turn
"credible case to try" into "greenlight" or "stand down." It is designed to answer the exact objections
a skeptical senior ORT engineer raises: no clean head-to-head numbers, async/thread integration
unproven, agentic claim anecdotal, OGA parity undefined.

**Step 0 — Define the ABI first (prerequisite, no experiment without it).** Any cross-language reuse or
plugin story rides on a **deliberately-stable C ABI**. Before porting or cross-language claims, freeze
the interface: the runtime C API surface and the plugin-EP ABI, versioned, with a compatibility policy.
This is cheap, uncontroversial, and unblocks everything else.

**A. The clean performance bake-off (replaces the server-mediated Foundry Local numbers in §1).**
- **Contestants:** `onnx-genai` (Rust) vs **onnxruntime-genai's C API directly** — *no* Foundry Local
  server in the loop, so no daemon/HTTP overhead on either side.
- **Held identical:** ORT build, model weights, EP, tokenizer, sampling, batching policy, hardware.
- **Report:** TTFT and inter-token latency (median **+ p95/p99**); tokens/sec at batch-1 **and under
  concurrent load**; peak + steady-state **host and device memory**; CPU utilization + thread count;
  allocation and Rust↔C transition profiles; cancellation/overload behavior; plus **binary size** and
  **cold/warm build time**.
- **Coverage:** **CPU and CUDA** at minimum, plus one constrained on-device target — and **more than one
  model family** (not just Qwen2.5-0.5B), including at least one larger model.
- *Author's current read (to be confirmed by this harness): CPU decode already exceeds OGA; CUDA is on
  par-to-ahead, with fuller CUDA numbers to be added.*

**B. The async / thread-pool integration proof.** A design + trace for the Rust serving slice showing:
bounded thread count under load, **no blocking FFI on executor workers**, explicit ownership of CPU
scheduling vs ORT's intra/inter-op pools, GPU-completion integration without polling storms, and clean
cancellation/backpressure — compared directly against a C++ slice using ORT's existing facilities.

**C. The agentic-development study (matched, blind).** The *same* set of representative tasks
implemented by agents in both a Rust crate and ORT C++, reviewed by ORT engineers **unfamiliar with each
implementation**. Measure iterations-to-green, wall-clock time, **escaped defects found by fuzzing**,
human review minutes, and a six-month follow-up modification cost. Rule: don't count compiler errors
fixed during generation as "defects avoided" unless the C++ arm gets equivalent compiler + sanitizer +
static-analysis feedback.

**D. The extensibility study.** Time-to-add a capability OGA lacks — e.g. an **audio-to-audio** or
**diffusion** pipeline — in each stack, from spec to passing fixture. This tests the "standard-driven,
declaration-not-dispatch" thesis directly (the `onnx-genai` agent-team work is the Rust data point).

**E. The OGA parity matrix (prerequisite deliverable for proposition (b)).** A requirements matrix with
explicit pass/fail gates: OGA public APIs, model families, multimodal preprocessing, structured
generation, adapters, device combinations, packaging, language bindings, telemetry, support
obligations. "The experiment implements X" is not evidence of product equivalence until this is filled.

### Decision gates (what each outcome greenlights)

| Result | Conclusion |
|---|---|
| Rust slice hits **feature parity, within ~2% latency/throughput, no thread/memory regression**, and **materially less implement+review time and fewer fuzz-found defects** | Greenlight proposition (b): refactor OGA to the Rust layer, on a schedule, parity-gated |
| Perf parity but agentic/maintenance benefits unproven | Continue (a) (new EPs + isolated components); keep (b) a pilot |
| Perf regression outside ~2%, or async/thread integration unsolved | Hold (b); the Rust layer stays experimental; keep wrapping ORT |
| Any runtime-internal seam can't match ORT on numerics **and** latency **and** coverage | That seam stays C++/FFI — proposition (c) is refused for it |

The bar is deliberately strict and symmetric: this experiment can **fail** and tell us to keep C++.
That is the point — it replaces advocacy with a measurement everyone has pre-agreed to believe.

---

## 13. Recommendation, and what would change it

**Recommendation (for discussion, not a mandate):** treat Rust as the default for *new* orchestration
and GenAI components and for *new* plugin EPs; where the runtime is reimplemented, do it incrementally
behind the ORT C API; keep and reuse C++/CUDA kernels through FFI indefinitely, replacing them only on
measured evidence. Treat "rewrite a hot kernel in Rust" as guilty-until-proven, and "write the
orchestration/EP glue in Rust" as the low-risk default.

**What would change this recommendation (intellectual honesty):** the **bake-off and decision gates in
§12** are the concrete, pre-agreed test — this recommendation is contingent on their outcome. In short:
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

Numbers below are from the `onnx-genai` workspace (excluding the sibling `onnxruntime-mlx` repo, whose
22,180 LOC / 521 `unsafe` are counted separately); "unsafe" counts *lines containing the keyword*
(occurrences, not syntactic blocks). Re-run against `HEAD` — they grow as the experiment does.

```bash
# Rust LOC across the workspace
find crates -name '*.rs' -not -path '*/target/*' | xargs wc -l | tail -1
# unsafe occurrences (lines containing the keyword, excluding tests)
grep -rn 'unsafe' crates --include='*.rs' | grep -v test | wc -l
# tests
grep -rn '#\[test\]\|#\[tokio::test\]' crates --include='*.rs' | wc -l
# per-crate unsafe (shows the boundary concentration)
for c in onnx-runtime-ir onnx-runtime-optimizer onnx-genai-kv onnx-runtime-capi onnx-runtime-ep-cuda; do
  echo "$c: $(grep -rn unsafe crates/$c --include='*.rs' | wc -l) unsafe"; done
```
