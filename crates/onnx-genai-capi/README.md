# onnx-genai-capi

A C ABI over `onnx-genai-engine`, focused on the **pluggable sampler** seam: C /
C++ / Python (ctypes/cffi) code can supply its own token sampler and drive
generation with it.

The Rust generation loop still runs the full logit-processor chain configured on
the request — temperature, top-k / top-p, min-p, repetition / frequency /
presence penalties, and constraints — exactly as onnxruntime-genai layers those
filters. Your sampler only replaces the **terminal token selection** (the pick
that would otherwise be greedy argmax or categorical sampling).

## Surface

See [`include/onnx_genai.h`](include/onnx_genai.h) for the full, documented API.

| Function | Purpose |
| --- | --- |
| `oge_sampler_new` / `oge_sampler_free` | build/destroy a foreign sampler from an `OgeSamplerVTable` |
| `oge_engine_load` / `oge_engine_free` | load/free a model directory |
| `oge_engine_generate` | generate with the engine's default sampler |
| `oge_engine_generate_with_sampler` | generate with your sampler (consumes it) |
| `oge_string_free` | free any `char*` this library returns |
| `oge_last_error` | thread-local message for the last failure |

An `OgeSamplerVTable` is a `user_data` pointer, a required `sample` callback, an
optional `name`, and an optional `free` destructor for `user_data`:

```c
uint32_t my_sample(void *user_data, const float *logits, size_t logits_len,
                   const uint32_t *generated, size_t generated_len, size_t step);
```

`logits` are the post-processor scores for the step (length == vocabulary size;
filtered-out tokens are `-inf`); return the chosen token id.

## Build

```sh
cargo build -p onnx-genai-capi --release          # CPU
cargo build -p onnx-genai-capi --release --features cuda   # + CUDA EP / on-device argmax
```

This produces a `cdylib` and `staticlib` (plus a Rust `lib`) in `target/`. See
[`examples/custom_sampler.c`](examples/custom_sampler.c) for a complete C program
that implements an argmax sampler on the C side and runs generation with it.

## Safety

- Fallible functions return `NULL` on failure; read `oge_last_error()`.
- Handles are freed exactly once by their `*_free` (all NULL-tolerant).
- No panic crosses the FFI boundary (every body is `catch_unwind`-guarded).
- The `ForeignSampler` adapter (Rust side) implements `onnx_genai_engine::Sampler`,
  so it is unit-tested through real `extern "C"` callbacks without a model.
