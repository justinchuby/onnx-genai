# Build features

This repository has two entirely different CUDA execution providers. Which one a
binary contains is decided at build time, by a Cargo feature, and the two produce
indistinguishable command lines and wildly different performance. Getting it
wrong reads as a 22x regression in your own code rather than as a wrong build
(see Trap 7 in `.github/skills/cuda-perf-measurement/SKILL.md`).

## The two CUDA paths

| Feature | Execution provider | Needs a CUDA toolkit to build? | Ships in |
| --- | --- | --- | --- |
| `ort-cuda` | ONNX Runtime's built-in `CUDAExecutionProvider` | No. GPU ONNX Runtime arrives at runtime from the `onnxruntime-gpu` wheel. | The Python wheels |
| `native-cuda` | This repo's hand-written EP, `onnx-runtime-ep-cuda` | Yes | The standalone binaries |

`native-cuda` is a **strict superset** of `ort-cuda`: it enables everything
`ort-cuda` does and adds our kernels plus the `native-backend` needed to reach
them. There is no configuration in which you want both spelled out.

They are deliberately **not merged into one `cuda` feature**. The CUDA wheels are
built from the same crates with `--features ort-cuda` precisely so that no CUDA
code is bundled and no CUDA toolkit is required on the wheel builders
(`.github/workflows/wheels.yml`, `.github/workflows/publish.yml`). Merging would
force every wheel build to compile our kernels.

## Checking what you actually built

Feature names are a claim; the binary is the evidence. The two differ by ~13 MB
and by every kernel symbol:

```console
$ strings -a target/release/onnx-genai | grep -c matmul_nbits_gemv
129     # native-cuda
0       # ort-cuda
```

`cargo tree -p onnx-genai-cli -i onnx-runtime-ep-cuda` printing "nothing to
print" is the other tell.

## Which crates offer a choice

A crate that offers a choice must name the choice. A crate with only one CUDA
path keeps the plain `cuda` name, because there is nothing there to confuse it
with.

| Crate | CUDA features |
| --- | --- |
| `onnx-genai-cli` | `ort-cuda`, `native-cuda` |
| `onnx-genai-server` | `ort-cuda`, `native-cuda` |
| `onnx-genai-engine` | `ort-cuda`, `native-cuda` |
| `onnx-genai-bench` | `ort-cuda`, `native-cuda` |
| `onnx-genai-capi` | `ort-cuda` only — the C ABI has no native path |
| `onnx-genai`, `onnx-genai-python` | `cuda` — ONNX Runtime, no alternative |
| `onnx-genai-ort` | `cuda` — it *is* the ONNX Runtime binding |
| `onnx-runtime-session`, `onnx-runtime-ep-cuda`, `onnx-runtime-python` | `cuda` — native only, no alternative |

## `native-cuda` implies `native-backend`

Enabling our CUDA EP without `native-backend` compiles the whole EP and then
cannot reach it, because the native session that dispatches to it is absent. The
build looks like it should be fast and behaves exactly like `ort-cuda`. Every
`native-cuda` feature therefore enables `native-backend` itself rather than
leaving it to the caller.

`native-backend` on its own (without `native-cuda`) remains valid and means the
native CPU backend.

## CUDA API version

Orthogonal to the above, and unambiguous. Exactly one of `cuda-12060`,
`cuda-12080`, `cuda-12090`, `cuda-13000` selects the driver API binding set;
`cuda-13000` is the default. Use `--no-default-features` when overriding. These
never build-depend on a toolkit — the driver is `dlopen`ed at runtime.

## Common invocations

```bash
# Measure our kernels.
cargo build --release -p onnx-genai-cli --features native-cuda --bin onnx-genai

# The wheel-compatible path.
cargo build --release -p onnx-genai-cli --no-default-features \
  --features cuda-13000,ort-cuda,onnx-genai-server/metrics --bin onnx-genai

# A/B the two providers in one bench binary.
cargo build --release -p onnx-genai-bench --features native-cuda,ort-cuda \
  --bin profile_native
```
