---
name: build-static-cache-model
description: Build a loadable static-cache (TensorScatter) ONNX model for onnx-genai. Use when continuous batching does not engage, when a model fails to load with "declares no model.io.static_cache", when a model loads but never stops generating, or when you need a real model for benchmarking or demoing continuous batching. Prefer scripts/build_qwen.sh, which now performs this whole recipe.
---

# Building a loadable static-cache model

## Why this exists

Continuous batching in `onnx-genai` engages **only for static-cache models**. The gate is in `crates/onnx-genai-server/src/driver.rs:407-421` — `continuous_batch_manager(max_batch).is_ok()` succeeds only when the model exposes a static KV cache. Dynamic-cache models silently fall back to the per-request path.

So if you are benchmarking or demonstrating batching, you need a static-cache model — and the obvious ways to produce one currently give you a model that will not load.

## Symptom → diagnosis

**Symptom A — the model fails to load:**

```
Invalid argument: graph exposes a TensorScatter static-cache scatter ABI but
declares no `model.io.static_cache`; its integer scatter control ports
(write_indices / kv_sequence_length) are shape-indistinguishable and cannot be
bound by port name.
```

**Symptom B — the model loads, but batching never engages.** The server logs:

```
continuous batch driver disabled; using per-request engine path
```

Batch-related metrics stay flat. **Always check the model before debugging the batching code** — this log line is the fastest diagnostic in the system.

When batching *is* working you will see:

```
INFO onnx_genai_server::driver: continuous batch driver enabled max_batch=4
```

## Do not follow a recipe when a script exists

`scripts/build_qwen.sh` now performs everything below, including the two steps
this document used to leave to the reader. Use it:

```bash
cd "$(git rev-parse --show-toplevel)"
STATIC_CACHE=1 MAX_SEQ_LEN=4096 OUT_DIR=/tmp/qwen-scatter scripts/build_qwen.sh
scripts/verify_model.sh /tmp/qwen-scatter
```

The rest of this document explains what those two commands do and why, so you
can diagnose a failure or reproduce it against a different model. **It is
reference material, not a checklist to type out.**

> **NEVER build into a directory that already contains a model, and never into
> `models/qwen2.5-0.5b-scatter-v2`.** An earlier version of this file told you
> to do exactly that. Two reasons, and the second is worse than the first:
> `resolve_model_path` (`crates/onnx-genai-ort/src/loader.rs:405-410`) hard-errors
> when it finds multiple `.onnx` files, and — far more dangerous — **leftovers
> from a previous build SATISFY a completeness check**, so a partial build looks
> like a complete one. `build_qwen.sh` now refuses a non-empty `OUT_DIR` unless
> you pass `FORCE=1`. Build to scratch, verify, then move it into place.

## Root cause

Three independent problems. The first two stop the model loading; **the third
lets it load, batch, answer, and never terminate.**

1. **The runtime target must be `--runtime onnx-genai`.** `--runtime ort-genai`
   emits only `genai_config.json`; the runtime needs `inference_metadata.yaml`.
2. **Mobius omits the `io:` block** for static-cache builds, so
   `model.io.static_cache` has to be added after the export.
3. **The `onnx-genai` runtime target writes ONLY `tokenizer.json`** — no
   `tokenizer_config.json` (`__main__.py:322` vs `:339`; the two targets are
   mutually exclusive). Without it, `load_eos_token_ids`
   (`crates/onnx-genai-ort/src/tokenizer.rs:103`) falls back to `<|endoftext|>`,
   **which exists in Qwen's vocabulary** — so nothing errors and the model
   generates until it hits the token budget on every single request.

> **Problem 3 is the one to fear, because every symptom of it looks like
> success.** It is also why `models/qwen2.5-0.5b-scatter-v2` worked: its
> tokenizer files are dated months before its graph — leftovers from an earlier
> build into the same directory. **That model was correct by accident of
> directory reuse, and a fresh clone reproduces neither the accident nor the
> result.**

The scatter control ports (`write_indices`, `nonpad_kv_seqlen`) are both integer
tensors of indistinguishable shape, so the loader cannot infer which is which
from the graph alone. That is why the declaration is mandatory rather than
optional — and why the loader keys on **the presence of those input ports**
rather than on TensorScatter nodes (`crates/onnx-genai-ort/src/decode/io.rs:194-198`).

## What the script does

### 1. Export with the correct runtime target

```bash
cd "$(git rev-parse --show-toplevel)"
HF_HOME=models/.hf_cache TMPDIR=models/.scratch PYTHONPATH=/path/to/mobius/src python -m mobius build --model Qwen/Qwen2.5-0.5B-Instruct   /tmp/qwen-scatter   --dtype f32 --ep default --static-cache --max-seq-len 4096   --runtime onnx-genai
```

`HF_HOME` and `TMPDIR` are redirected into `models/` because the download and
scratch space are multi-gigabyte and you usually do not want them on the default
volume. **The output directory is scratch, not `models/`.**

### 2. Derive the `model.io.static_cache` block

**Do not hand-author this and do not copy the fixture.** Run the generator,
which reads the block out of the graph you actually built:

```bash
python scripts/lib/write_static_cache_metadata.py /tmp/qwen-scatter
```

Use the **same Python that has Mobius installed** — this reads the graph with
`onnx`, and a bare system Python fails with `the 'onnx' package is required`.

Hand-authoring has a failure mode that is invisible on small models: the four
per-layer lists **pair positionally**, so the layer ports must be sorted
**numerically**. Sorted as strings, `key_cache.10` lands before `key_cache.2`
and **every buffer past layer 9 is silently mis-paired** — a model that loads
and produces plausible-looking garbage. Under ten layers the two orderings are
identical, which is why a small test cannot catch it.

Pass `--check` to print the derived block instead of writing it (read-only).

### 3. Copy the tokenizer companions

```bash
python scripts/lib/write_tokenizer_assets.py \
  Qwen/Qwen2.5-0.5B-Instruct /tmp/qwen-scatter
```

Takes **two** arguments — the source model id (or a local directory) and the
built model directory. Resolves `tokenizer_config.json` (and friends) through
the Hugging Face cache,
never overwrites what the exporter produced, and **fails loudly** if the file
cannot be obtained — because a missing one produces problem 3 above, and a build
that silently omits it is indistinguishable from a good one.

Cached files are reused, but resolution still contacts the Hub, so **this step
needs network access on a machine that has not downloaded the model before.**
Pass a local directory instead of a model id to work fully offline.

### 4. Verify — and do not skip this

```bash
scripts/verify_model.sh /tmp/qwen-scatter
```

**This is the terminal step of the recipe.** The model is not built until this
passes; exit 0 is the only evidence that counts.

It asserts three things, and the second and third exist because of real defects:

- the model **loads**;
- `continuous batch driver enabled` appears in the log. **Both outcomes log at
  `INFO`** (`driver.rs:417` and `:420`), so it greps for the *enabled* line
  **and separately fails on the *disabled* line** — the absence of an error
  proves nothing;
- the completion returns **`finish_reason=stop`**. A model with the tokenizer
  defect returns `finish_reason=length`, and this is the only cheap check that
  catches it.

If the port is busy it **refuses to run** rather than reporting on whatever else
is listening; pass `--port` to move it.

Reference numbers for Qwen2.5-0.5B f32 on CPU: ~10-50 s cold model load; a
single 24-token completion ~2.2 s; four concurrent 32-token completions ~8.7 s
total. **Timings are not a pass criterion** — they vary ~10% with background
load alone.

### 5. Only now promote it — to a NEW directory, never onto an existing one

```bash
# The destination MUST NOT EXIST. Bump the version rather than replacing.
test ! -e models/qwen2.5-0.5b-scatter-v3 || { echo "destination exists; bump the version"; exit 1; }
mv /tmp/qwen-scatter models/qwen2.5-0.5b-scatter-v3
scripts/verify_model.sh models/qwen2.5-0.5b-scatter-v3   # RE-VERIFY AT THE FINAL PATH
```

> **☠️ AN EARLIER VERSION OF THIS STEP SAID `mv /tmp/qwen-scatter
> models/qwen2.5-0.5b-scatter-v2`, AND ITS FAILURE MODE IS SILENT.** `models/`
> is gitignored, so `-v2` is the only copy of the only known-good static model
> in the project and @fc8b5d97's performance baseline is measured against it.
> But it would not have been *destroyed*, which is the part worth understanding:
>
> **`mv src dst` where `dst` is an existing directory moves `src` INSIDE it**,
> producing `…-v2/qwen-scatter/`. `resolve_model_path`
> (`crates/onnx-genai-ort/src/loader.rs:398-400`) scans with
> **`std::fs::read_dir` — non-recursive — and filters on `path.is_file()`**, so
> the nested copy is invisible to it. It does *not* trip the multiple-`.onnx`
> hard error documented above. The directory still resolves, and it resolves to
> the **original** `model.onnx`.
>
> **So the promotion silently does nothing, and everything downstream passes —
> against the model you believed you had just replaced.** `verify_model.sh`
> would certify it, because the model it loaded genuinely is good. The tool
> built to prevent false confidence becomes the thing producing it.
>
> **This is why step 5 re-runs verification at the FINAL path.** Verifying an
> artifact at the path where it was *built* proves nothing about the path where
> it will be *used*, and every step between the two is unverified.

Leave `-v2` in place. Nothing needs to be deleted; a superseded model dir costs
disk and removes a rollback.

## Gotchas

- **`--model` takes a directory, not a config file.** The CLI coerces a file path to its parent via `resolve_model_dir` (`crates/onnx-genai-cli/src/lib.rs:674`); the **server does not**. Passing `.../genai_config.json` to the server fails with `model directory does not exist`.
- **Always `export CARGO_TARGET_DIR`** to a shared target directory. A cold build of this 39-crate workspace is enormous; a warm build of the server is under a minute. Cargo takes an exclusive lock, so concurrent builds serialize.
- **Tiny test fixtures have a ~16-token context.** Use `max_tokens: 4` against `tests/fixtures/tiny-llm*` or requests are rejected. Use a real model for anything resembling a benchmark.
- **Static-cache and paged KV are mutually exclusive.** A static-cache model gets continuous batching but *no* paged-KV page activity, no prefix caching, and no preemption (`crates/onnx-genai-engine/src/batched.rs:757` hardcodes `PreemptionPolicy::Disabled`). A prefix-cache hit count of zero on this path is correct, not a bug.
