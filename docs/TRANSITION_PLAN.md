# Transition plan — evolving ONNX Runtime and OGA toward a Rust, agent-native runtime

**Audience:** the ONNX Runtime and ONNX Runtime GenAI (OGA) teams at Microsoft.
**Companion:** [`CPP_VS_RUST.md`](./CPP_VS_RUST.md) (the honest C++/Rust technical case),
[`ORT2.md`](./ORT2.md) (runtime design), [`DESIGN.md`](./DESIGN.md) (GenAI layer).

This is a proposal for *how* we could move, incrementally and low-risk, toward a Rust,
agent-native inference stack — **without a big-bang rewrite, without throwing away the kernels, and
with the OGA and ORT teams driving it.** It is written to be argued with. Nothing here is a criticism
of the people who built OGA and ORT; it is an argument about where the *next* era of models and
development workflows is pulling the architecture.

---

## 0. Framing (read this first)

OGA did something genuinely hard and valuable: it made generative models *run well on ORT*, shipped a
large catalog of model families, and built up the config, recipe, and conformance knowledge that any
successor depends on. None of that is wasted — it is the foundation the transition stands on.

The honest observation is narrow and forward-looking: OGA's design grew up around a **per-model-family
dispatch** style (each new architecture tends to need bespoke wiring), which was exactly the right
call for bootstrapping — and which now bumps against an extensibility ceiling as the model zoo
explodes (hybrid attention/SSM, MoE, linear attention, long context, multimodal) and as *agents*, not
just humans, become the ones extending it. The goal is to evolve past that ceiling **with** the OGA
team, reusing their work, not around them.

Two independent moves make this safe:

1. **Make every EP a plugin EP; the EP's language stays a free choice.** The direction is to move
   *all* execution providers behind ORT's plugin-EP C ABI (the same interface the MLX EP uses), so EPs
   become independently versioned, loadable modules instead of in-tree code. Once an EP is a plugin,
   **what language it's written in is an implementation detail**: a large, complex existing C++ EP can
   stay C++ behind the ABI (no forced port), while new EPs can be Rust where that's simpler. Proven: a
   pure-Rust MLX EP loads into *unmodified* ORT and ships `PagedAttention`, `QMoE`, GQA, and a fused
   compiled path — on hardware ORT had no kernels for. Zero risk to existing ORT.
2. **Rewrite the layers that need an LLM redesign anyway, in Rust.** The serving layer — scheduler,
   KV/memory management, batching — is *not* a port; it needs a ground-up redesign for LLM and agentic
   workloads regardless of language (see the fact-check in §1a). Doing that redesign in Rust is
   therefore nearly free, and it's exactly the layer where Rust's safety and concurrency pay off most.
   Model behavior in this layer comes from *declared metadata* (per
   [onnx/onnx#8184](https://github.com/onnx/onnx/issues/8184)) rather than hardcoded model-type
   branches — so adding a model is data, not code.

---

## 1. Principles (the guardrails on every phase)

- **Strangler-fig, never big-bang.** The system is shippable and ORT-compatible at every step. We
  replace one seam at a time, each behind a conformance gate.
- **Keep the kernels.** MLAS, CUDA/cuDNN, fused attention — reused through FFI. We rewrite a
  kernel only on measured evidence, or where none exists (e.g. Apple/MLX).
- **Contracts first, agents implement, the machine verifies.** Humans (the OGA/ORT experts) own the
  interfaces, the invariants, and the acceptance oracles; agents fill in implementations; build +
  clippy + the ORT conformance suite decide whether it's correct. (See `CPP_VS_RUST.md` §7–§8.)
- **Standard-driven, not model-specific.** Behavior is derived from declared inference metadata, so
  new architectures are onboarded as declarations + tests, not a new dispatch branch.
- **EPs are plugins; EP language is a per-EP choice.** Every EP sits behind the plugin-EP C ABI. That
  decouples EP releases from the core and makes the language of any given EP an implementation detail —
  keep C++ where it's mature, use Rust where it's easier.
- **C++ and Rust coexist indefinitely.** This is not a deadline to delete C++; it is a gradient where
  new and refactored seams default to Rust and the mature kernels stay where they are.
- **Conformance is the contract with ORT.** Every Rust seam presents ORT's C API and passes ORT's own
  node/model conformance corpus, so downstream consumers never notice the language underneath.

---

## 1a. Fact-check: the serving layer needs an LLM redesign regardless of language

The claim behind Phase 1 is that the **scheduler / KV-memory / batching** layer is not a port but a
*redesign* the field has already had to do for LLMs — so rewriting it (in Rust) costs little beyond the
redesign we'd owe anyway. That claim checks out; classic DNN serving and LLM serving are structurally
different systems:

| Concern | Classic DNN inference (what ORT's scheduler grew up on) | LLM / agentic serving (what's needed now) |
|---|---|---|
| Batching | Request-level: gather N requests, one forward pass, all finish together | **Iteration-level / continuous batching**: batch re-formed every decode step; requests join and leave mid-flight (Orca, OSDI '22) |
| Per-request state | Stateless, one-shot | **Growing per-sequence KV cache** dominates memory and lifetime |
| Memory | Fixed activations, no fragmentation problem | **Paged KV** to avoid fragmentation and pack many concurrent sequences (vLLM, SOSP '23) |
| Phases | One forward pass | Distinct **prefill (compute-bound) vs decode (memory-bound)** phases scheduled differently (chunked prefill, prefill/decode disaggregation) |
| Scheduling policy | FIFO / round-robin over independent requests | **Preemption, priority, prefix/KV reuse, fairness** across many long-running, variable-length generations |
| Advanced paths | n/a | **Speculative decoding**, multi-turn/long-context concurrency as the primary workload |

Continuous batching + paged attention are a documented paradigm shift, not a tuning tweak: they are the
core contributions of Orca and vLLM precisely because the classic request-batching scheduler leaves the
GPU idle (padding, head-of-line blocking) on autoregressive, variable-length generation. So the
scheduler and memory manager are the **most-redesigned** components in the move to LLMs — and the ones
where Rust's ownership model and fearless concurrency pay off most (a paged KV allocator and a
continuous-batching scheduler are exactly the concurrent, lifetime-heavy state machines Rust checks at
compile time). Conversely, the **kernels and EPs** (matmul, attention, conv) are *not* redesigned by
this shift — which is why they stay in C++ behind the plugin ABI and get reused, not rewritten.

**Net:** rewrite what's being redesigned anyway (scheduler/memory/serving) in Rust; keep what's stable
(kernels/EPs) in place. The redesign is the cost; the language is nearly free on top of it.

---


### Phase 0 — Every EP becomes a plugin EP (now; zero risk; language-agnostic)

The wedge that requires nobody's permission. The move is architectural first, linguistic second: put
**all** EPs behind ORT's plugin-EP C ABI so each is an independently loadable, independently versioned
module (`.dylib`/`.so`) that stock ORT discovers and loads. This decouples EP release cadence from the
core and shrinks the in-tree surface — a win even for EPs that never change language.

Once an EP is a plugin, **its language is a free choice**:

- A large, complex, heavily-tuned C++ EP (CUDA, for instance) can **stay C++** behind the ABI — no
  forced port, no risk.
- New or from-scratch EPs can be Rust where that's simpler and safer. **Proof:** `onnxruntime-ep-mlx`
  — a pure-Rust MLX EP on crates.io/PyPI, ~130 claimed ops, `PagedAttention`/`QMoE`/GQA,
  compiled-subgraph fusion — runs models ORT could not run on Apple Silicon at all, loaded into
  *unmodified* ORT.

- **Ask of the team:** commit to the plugin-EP boundary as the standard packaging for EPs, and treat
  Rust as an accepted language for *new* EPs. Both ship into existing C++ ORT deployments unchanged.
- **Exit criteria:** EPs (C++ or Rust) load through the plugin ABI and pass the EP conformance suite;
  an EP's language is invisible to the core.

### Phase 1 — A refactored, extensible GenAI layer (OGA-next) in Rust

Reimagine the GenAI layer around declared metadata and trait-based plugins, with ORT as the compute
backend (via its C ABI). This is where "too model-specific / not extensible" gets fixed.

- **Standard-driven models:** a model is onboarded by its declared inference metadata + a conformance
  fixture, not a new code path. New architectures (hybrid/SSM, MoE, linear attention) are declarations.
- **Trait-based, à-la-carte components** (already real in the experiment): `Sampler`, `LogitProcessor`,
  `Constraint`, `TokenEmbedder`, `LmHead`, `SpeculativeProposer`. Users and partners reuse the pieces
  they want and drop in their own (`impl Sampler for MySampler`) — see `CPP_VS_RUST.md` §9.
- **ORT stays the engine:** all NN compute is delegated to ORT sessions/EPs; the GenAI layer owns
  everything above the session (KV, batching, scheduling, sampling, serving).
- **Ask of the team:** co-own the metadata contracts and the model-onboarding fixtures. OGA's recipe
  and config knowledge is exactly what defines these contracts.
- **Exit criteria:** parity with OGA on a target model set (quality + latency) behind the same public
  surface, model onboarding reduced to declaration + fixture.

### Phase 2 — Agent-driven, incremental ORT → Rust port (behind the ORT C API)

Reimplement the runtime layer one seam at a time — IR → loader → shape inference → optimizer →
partitioner → memory planner → session — each presenting the ORT C API and gated on ORT conformance.
The `onnx-runtime-*` crates in the experiment are a running sketch of this (a real `onnx-runtime-ir`
contract, CPU EP with kernels, optimizer, session skeleton).

- **How the porting happens:** agents do the mechanical translation against the C++ reference and the
  conformance corpus; humans own the seam's contract and review the (small, labeled) `unsafe` at the
  FFI edges. This is the workflow Rust is unusually good at (precise compiler feedback, "if it
  compiles the rename is complete", conformance as the oracle).
- **Kernels are reused, not ported,** until there is a measured reason otherwise.
- **Ask of the team:** point the conformance suite at each Rust seam as it lands; treat a green
  conformance run as the merge bar.
- **Exit criteria:** each seam matches ORT latency within a small margin on conformance, or it stays a
  C++/FFI wrapper. We measure; we don't assume.

### Phase 3 — New-era capabilities as first-class features

The reason to do any of this is not to re-draw the same runtime in a new language — it is to build the
capabilities the current design makes hard, which the next generation of models and agentic workloads
demand. The experiment already prototyped these, which is the evidence they're tractable:

- **Advanced memory management:** paged + quantized KV cache, tiered/device-resident KV, a global
  liveness-based memory planner, zero-copy mmap weight streaming, DLPack zero-copy import/export.
- **Agent-native serving:** concurrent multi-turn, long-context, continuous batching, speculative
  decoding — as the *primary* workload, not an add-on.
- **Modern architectures by declaration:** hybrid attention/SSM, MoE (`QMoE`), linear attention,
  block-paged attention (`PagedAttention`).
- **Placement & capture:** global-optimal device placement, CUDA-graph capture, async transfer and
  compute/communication overlap.

These land incrementally on top of Phases 1–2, each behind the same contracts and conformance.

---

## 3. Who does what (this is collaborative, not a takeover)

| Role | Owned by | Why |
|---|---|---|
| Kernels (MLAS/CUDA/cuDNN/fused attn) | **ORT team, in C++** | Decade of tuning; reused via FFI, not rewritten |
| Execution providers (behind the plugin ABI) | **EP owners; language is their choice** | Complex C++ EPs stay C++; new EPs may be Rust (MLX proof) |
| Scheduler / KV-memory / continuous batching | **Joint, redesigned in Rust** | Being redesigned for LLMs regardless (§1a); safety+concurrency payoff |
| Metadata contracts + model onboarding fixtures | **OGA team** | Their recipe/config knowledge *is* the contract |
| Conformance corpus (the acceptance oracle) | **ORT team** | The bar every Rust seam must clear |
| Seam contracts & `unsafe`/FFI review | **ORT/OGA engineers** | Small, concentrated, high-value review surface |
| Mechanical implementation & porting | **Agents, supervised** | Contract-bounded, machine-verified |
| New-era features (memory, placement, serving) | **Joint** | Designed with the teams, prototyped in the experiment |

The experts are *more* central in this model, not less: they move up the stack from writing every line
to defining contracts, owning the kernels, and reviewing the parts that actually need judgment.

---

## 4. Risks and how we bound them

- **"It's a rewrite, and rewrites fail."** Mitigation: it is explicitly *not* a rewrite — kernels stay,
  each seam is conformance-gated behind the C API, and the system ships at every step. A seam that
  can't match ORT stays C++.
- **Perf regressions at the Rust↔kernel boundary.** Mitigation: FFI overhead is measured in decode-step
  profiles; hot boundaries are redrawn or kept native.
- **Reviewability for engineers new to Rust.** Mitigation: safe Rust removes the entire
  memory-safety review category; review shifts to contract-fit (language-agnostic); deep-Rust review
  concentrates on the small `unsafe` surface; agent-assisted review closes the fluency gap while the
  team ramps. (See `CPP_VS_RUST.md` §8.)
- **Ecosystem/interop.** Mitigation: cross-language consumers keep the C ABI (proven by the MLX plugin);
  Python reaches the ergonomic pieces via PyO3, exactly as `tokenizers`/`safetensors` do.
- **OGA continuity.** Mitigation: OGA-next presents a compatible surface and reaches parity before
  anything is deprecated; nothing is turned off until the successor is provably better.

---

## 5. What would tell us to stop or slow down

- A Rust seam that cannot reach ORT's latency within a small margin on conformance → it stays a C++/FFI
  wrapper, and we learn where the boundary belongs.
- Team velocity dropping because of Rust ramp-up rather than rising after it → we slow the cadence and
  invest in enablement first.
- FFI overhead showing up as a real decode-step cost → we redraw the boundary or keep the hot path
  native.

Intellectual honesty is the point: the plan is a set of hypotheses, each with a conformance-and-latency
gate that can fail loudly and cheaply.

---

## 6. Concrete next steps (small, reversible)

1. **Adopt the plugin-EP boundary as standard, and accept Rust for new EPs.** Land the plugin-EP path
   officially so every EP (C++ or Rust) loads through the ABI; the MLX EP is the reference.
2. **Stand up the metadata/onboarding contract** for OGA-next with the OGA team, and port one or two
   representative model families as declarations + fixtures to validate the shape of the contract.
3. **Pick one runtime seam** (e.g. the graph optimizer or shape inference) and port it to Rust behind
   the ORT C API, gated on the existing conformance corpus — a self-contained proof of the Phase-2
   loop.
4. **Prototype one new-era feature end-to-end** (e.g. the paged/tiered KV memory manager) against a
   real serving workload to show the capability upside, not just parity.

Each step is independently valuable, independently reversible, and driven by the teams who own the
code today.

---

### Appendix: what already exists as evidence (all reproducible in this repo)

- A pure-Rust ORT **plugin EP** (`onnxruntime-ep-mlx`) on crates.io + PyPI: `PagedAttention`, `QMoE`,
  GQA, GatherND, compiled fusion.
- A Rust **GenAI layer** with declared-metadata onboarding and trait plugins (`Sampler`,
  `LogitProcessor`, `Constraint`, …).
- A Rust **runtime layer** sketch behind the ORT C API: `onnx-runtime-ir` (stable contract), a CPU EP
  with real kernels, optimizer, shape inference, session.
- New-era features already prototyped: paged/quantized KV, device-resident KV, zero-copy weight
  streaming, DLPack, CUDA-graph capture.
- Scale/quality: ~119k lines of Rust across 27 crates, 1468 tests, `unsafe` concentrated at FFI
  boundaries (see `CPP_VS_RUST.md` §1).
