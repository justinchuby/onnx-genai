# CPU kernel performance and numeric quality

The pure-Rust CPU EP has a reusable Criterion harness and fixed numeric golden
tests in
[`crates/onnx-runtime-ep-cpu/benches/`](../crates/onnx-runtime-ep-cpu/benches/README.md).
It currently covers Add, ReduceMean, Gather, and MatMul at small, medium, and
large inference-oriented shapes, including f16/bf16 where supported.

The standing acceptance bar for a kernel rewrite is:

- no numeric-regression failure;
- no slowdown versus ONNX Runtime's CPU EP for matching operation, shape, and
  dtype; and
- no loss of the Rust EP's broader dtype coverage.

The next optimization phase should port the highest-leverage MLAS primitive
first: GEMM/MatMul (then quantized MatMul). MatMul dominates transformer
prefill/projection work, already has representative benchmark coverage, and
provides a backend seam that can be changed without altering the EP contract.
Every port should add relevant shapes/dtypes to both the Criterion and ORT
baseline matrices before replacing the implementation.

For the medium f32 MatMul shape (`32×512×512`), the current warm-cache,
allocation-outside-timing comparison is thread-matched and self-reported by
both harnesses. The Rust EP now defaults to the built-in **`SimdX86`** backend
(MLAS-style packed SIMD f32 SGEMM: `6×16` AVX2/FMA register microkernel, panel
packing, K/N cache blocking, column-strip Rayon parallelism), runtime-selected
via `is_x86_feature_detected!("avx2"/"fma")` with a safe fallback to the
`Generic` blocked GEMM on non-AVX2 hosts:

| Workers | Rust (Generic, old) | Rust (SimdX86, new) | ORT 1.27 CPU EP | New Rust / ORT |
|---:|---:|---:|---:|---:|
| 1 | 2.801 ms | 285 µs | 131 µs | 2.2× |
| 8 | 502 µs | ~155 µs | 30.6 µs | ~5× |

That is a **9.8× single-thread** and **3.2× eight-thread** speedup over the
previous pure-Rust GEMM, closing the ORT gap from ~16–21× to ~2–5×. The larger
`32×1024×1024` f32 shape (no ORT baseline recorded) runs 1.27 ms → 327 µs from 1
to 8 workers. f16/bf16 MatMul also gets faster (same SIMD GEMM after widening);
their remaining cost is the pre-existing widen-to-f32 conversion, not the GEMM.

Remaining gap versus ORT: (1) no AVX-512 microkernel yet (AVX2/FMA only), (2) a
single `6×16` kernel rather than MLAS's shape-specialized kernel family with
software prefetch, and (3) eight-thread scaling is bounded on the small medium
shape (M=32, ~8.4 MFLOP) by per-strip B repacking and Rayon task overhead; MLAS
shares a single packed B panel and partitions work more finely. Larger shapes
scale better (~3.9× at 8 threads on the large shape).

Rust runs inside dedicated 1- and 8-worker Rayon pools. ORT pins the matching
intra-op count and uses one inter-op thread for its one-node model. Thus the
honest current gap is approximately 2–5×, not an unqualified default-thread
comparison. Add, ReduceMean, and Gather are single-threaded internally and are
reported as such. The next port target remains quantized MatMul.

## Multi-threaded vendored MLAS (`NXRT_CPU_GEMM_BACKEND=mlas`)

The opt-in `mlas` feature (`CpuBackend::Mlas`, x86-64) now runs **multi-threaded**.
MLAS's high-level `MlasGemmBatch` does its own cache-aware M/N tile partitioning
and dispatches the tiles through `MlasTrySimpleParallel` /
`MlasGetMaximumThreadCount`. In the standalone `BUILD_MLAS_NO_ONNXRUNTIME` build
those two primitives were serial with a hard thread cap of 1; `mlas-sys` now
installs a **Rayon-backed parallel-for backend** (`vendor/shim.cpp` hooks +
`mlas_set_threading` from `src/lib.rs`) so MLAS keeps its native partitioning
while executing the tiles across the current Rayon pool — the same pool
`SimdX86`/`Generic` use, so there is no oversubscription. For `32×512×512` at 8
threads MLAS chooses `ThreadCountN=8, ThreadCountM=1` (its native N-partition).

### Isolated GEMM parity (warm, buffers reused — the `mlas-sys` perf probe)

This is the apples-to-apples comparison to the recorded ORT baseline (ORT also
reuses/prepacks buffers). `32×512×512`, repack-B-per-call:

| Workers | ORT 1.27 CPU EP | vendored MLAS (this slice) | MLAS / ORT |
|---:|---:|---:|---:|
| 1 | 131 µs | ~123 µs | 0.94× |
| 8 | 30.6 µs | ~32 µs | 1.05× |

**Parity reached**: the vendored MLAS SGEMM kernel is at/below ORT single-thread
and within noise of ORT at 8 threads (~500 GFLOP/s, ~3.8× scaling from 1→8).
Reproduce with `cargo test -p mlas-sys --release -- --ignored --nocapture
perf_sgemm_multithread`.

### End-to-end `MatMul` kernel (Criterion harness, f32, thread-matched)

Through the full `MatMul` kernel (each call allocates and zeroes a fresh output
buffer — a cost shared by *every* backend and not yet optimized), warm-cache
Criterion medians on this Sapphire Rapids host:

| Shape | Backend | 1 thread | 8 threads |
|---|---|---:|---:|
| 1×256×256 | Generic | 48.9 µs | 55.5 µs |
| 1×256×256 | SimdX86 | 31.2 µs | 48.7 µs |
| 1×256×256 | **MLAS** | **22.0 µs** | **38.6 µs** |
| 32×512×512 | Generic | 2.797 ms | 492 µs |
| 32×512×512 | SimdX86 | 300 µs | 162 µs |
| 32×512×512 | **MLAS** | **178 µs** | **147 µs** |
| 32×1024×1024 | Generic | 11.93 ms | 1.922 ms |
| 32×1024×1024 | SimdX86 | 1.239 ms | 363 µs |
| 32×1024×1024 | **MLAS** | **808 µs** | **306 µs** |

MLAS is the fastest backend at every shape and thread count. The absolute 8-thread
end-to-end numbers do **not** reach the isolated ~32 µs because the per-call
output allocation dominates once the GEMM itself is this fast (measured: for
`32×512×512` at 8 threads the fresh-`Vec` allocation alone costs ~50–90 µs). That
allocation overhead is backend-agnostic and is the next lever (see
`TODO(mlas prepack)` in `matmul.rs` — cache `PackedB` and reuse output buffers);
it is out of scope for this GEMM-threading slice. The tiny `1×256×256` shape does
not benefit from 8 threads (compute < scheduling overhead) for any backend.

`CpuBackend::Mlas` stays opt-in (`NXRT_CPU_GEMM_BACKEND=mlas`) and off the default
build; auto-detect still selects `SimdX86`. Flipping the default once buffer reuse
lands is a one-line change in `CpuBackend::auto_detect`.

