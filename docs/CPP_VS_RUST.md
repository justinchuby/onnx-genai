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
   version is incremental and reuse-heavy (see §10–§11).

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
- **Verification is automatic and total.** The agent's PR is not "trust me"; it is "the contract you
  wrote compiles, and the behavior you specified passes" — build + `clippy` + the conformance suite.
  That is precisely the human-defines / agent-implements / machine-verifies loop you want, and Rust
  makes the "contract" half enforceable at compile time rather than a convention policed in review.
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
  build did you link against." The experiment already proved the publish path end-to-end: the MLX EP
  is on crates.io, installable with `cargo add`. The C++ story is heavier — consume ORT and you link
  (or vendor) the monolith and match its ABI and compiler.
- **Fine-grained crates = take only what you need.** The workspace is 27 crates with real boundaries
  (`onnx-runtime-ir`, `-memory`, `-optimizer`, `-ep-cpu`, `-ep-cuda`, `onnx-genai-kv`, `-scheduler`,
  `-engine`, `-preprocess`, `-router`…). Depend on the three you use; the rest never enters your build.
  Feature flags (`default-features = false` + opt-in) trim further — a CUDA-free build, a tracer-free
  build. C++ *can* be modular, but in practice ORT ships as one large artifact and cherry-picking a
  subsystem is surgery.
- **Traits are clean, zero-cost plugin points — and they already exist here.** "Bring your own
  component" is a trait the user implements. The experiment already exposes `Sampler`, `LogitProcessor`,
  `Constraint`, `TokenEmbedder`, `LmHead`, and `SpeculativeProposer` as public traits: a user writes
  `impl Sampler for MySampler` and drops it in. It's compile-time-checked, monomorphized (the
  abstraction costs nothing at runtime — unlike a virtual-dispatch or FFI-callback plugin), and needs
  no registration boilerplate or ABI contract. Users provide custom-optimized components by
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
- Interop goes through **CXX** (safe, generated, bidirectional bindings) at a reviewed FFI boundary.
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

**The shared playbook (identical across all three):**
1. Rust enters through the **existing build system** as normal library targets — no new toolchain the
   consumer must adopt (Cargo can emit a `staticlib`/`cdylib` that CMake/MSBuild/GN links like any
   other `.a`/`.lib`/`.so`; `cxx`/`cbindgen` generate the headers).
2. A **stable ABI / interop boundary** (CXX, or a plain C ABI) keeps callers unchanged.
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

## 12. Recommendation, and what would change it

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
