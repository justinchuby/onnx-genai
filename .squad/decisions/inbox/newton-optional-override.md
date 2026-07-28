### 2026-07-28: OptionalOverride classification is explicit opt-in, not structural

**By:** Newton

**What:** Implemented the P1 "overridable optional input" executor mechanism
(design §B.3) — a graph value that is simultaneously (a) a graph input feedable
by name, (b) an initializer carrying a concrete default used when unfed, and (c)
the holder of a declared *dynamic* dimension so a different-rank override can be
fed. It is a general executor capability with nothing LoRA-specific in the
executor.

A value is classified as an override **only** when its `ValueId` is passed
explicitly into a new build entry point,
`Executor::build_with_overrides(graph, weights, ep, &override_ids)`. Classification
is never inferred structurally. The descriptor (`OptionalOverride` in
`executor/state.rs`) holds the declared (symbolic) shape, the concrete default
shape, and owned default bytes; it is stored in a new
`Executor::optional_overrides: HashMap<ValueId, OptionalOverride>` that is empty
for every un-injected graph. Every new code path is gated on membership in that
(empty-by-default) map.

Changed files/functions (all in `crates/onnx-runtime-session/src/executor/`):
- `state.rs` — added `OptionalOverride` descriptor + `optional_overrides` field;
  documented `required_inputs` as one of three input categories (required /
  plain-initializer / overridable-optional).
- `build.rs` — new `build_with_overrides`; threaded `override_ids` through
  `build_with_cuda_requirement`; `materialize_initializers` now keeps the
  override's declared symbolic shape (not the initializer dims), forces an
  **owned** buffer (never the borrowed mmap alias), and stashes default
  shape/bytes; `build_name_indexes` registers overrides in `input_index` but not
  `required_inputs`; `size_buffers_excluding` sizes overrides per run instead of
  skipping them as initializers; `compile_all` marks overrides non-constant.
- `bindings.rs` — `bind_symbols` seeds each *unfed* override's declared symbol(s)
  from its default shape; the fed path already takes the `Dim::Symbolic` branch
  in `bind_input_shape` (confirmed — no `Dim::Static(0)` rejection because the
  declared shape stays symbolic).
- `run.rs` — `prepare_run_buffers` reinstates default bytes for every unfed
  override after binding the fed ones.
- `dispatch.rs` — override inputs are excluded from the per-dispatch
  `constant_inputs` set so a kernel never packs/transposes stale override bytes.
- `capture.rs` — `// CUDA phase (P5)` marker where a device-bound override must
  join the persistent-binding set and capture signature (no CUDA code).

**Why (byte-identity guarantee):** Pure structural detection ("a graph input
that is also an initializer") is dangerous: pre-IR-4 ONNX lists *every*
initializer as a graph input, so a structural rule would silently reclassify all
initializers and break existing models. I verified our loader already prevents
this overlap for loaded models — `graph_builder.rs:120-126` skips any graph-input
`ValueInfo` whose name is also an initializer, so `graph.inputs ∩
graph.initializers` is **always empty** for models produced by our loader. The
overlap only exists for graphs constructed programmatically (the P1 tests, and
the future P2 injection pass), and even then a value becomes an override *only*
if its id is in the explicit `override_ids` set. Therefore, for any graph with no
registered overrides the set is empty, every gated branch is inert, and the
executor builds and runs byte-for-byte as before — asserted by the
`optional_override_empty_set_is_byte_identical_to_baseline` test (identical
required inputs, empty descriptor map, identical output) and by the full
pre-existing session suite staying green (95 lib + all integration tests).

**Tests (CPU, `executor/tests.rs`):** `optional_override_unfed_falls_back_to_default`,
`optional_override_host_feed_binds_and_resizes`,
`optional_override_restores_default_after_feed`,
`optional_override_rank_change_rebinds_and_resizes`,
`optional_override_empty_set_is_byte_identical_to_baseline` — a synthetic
`MatMul → MatMul → Add` graph with two override inputs (default rank r=0 ⇒
empty-inner-dim delta ⇒ `Add` no-op). All 5 pass.

**Design-doc note:** No §B.3 assumption turned out wrong at the code level — the
cited line ranges had shifted but the described mechanisms matched. The one
sharpened point: the byte-identity risk the doc flagged is *already* neutralized
by the loader's input/initializer separation, which made the explicit opt-in set
a belt-and-suspenders choice rather than the sole safeguard.
