// Shim: expose plain C entry points over MLAS's C++ SGEMM API so Rust can call
// it via FFI without bindgen.
//
// Threading (multi-thread MLAS): MLAS's high-level GEMM (`MlasGemmBatch`)
// computes its own cache-aware M/N thread partitioning and dispatches the tiles
// through `MlasTrySimpleParallel` / `MlasGetMaximumThreadCount`. In
// `BUILD_MLAS_NO_ONNXRUNTIME` (standalone) mode those two primitives normally
// degrade to a serial loop with a hard thread cap of 1. Rather than fight MLAS
// by re-partitioning at the Rust level, we let MLAS keep its own partitioning
// and give it a *pluggable parallel-for backend*: the vendored standalone
// primitives call the `MlasStandalone*` hooks below, which forward the
// parallel-for onto a real thread pool that Rust drives with Rayon (the same
// global pool the rest of ep-cpu uses, so there is no oversubscription). When
// no backend is registered (e.g. the mlas-sys unit tests, or the ep-cpu default
// build) the hooks run serially — identical to the original spike behaviour.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/mlas/inc/mlas.h"

#include <cstddef>
#include <functional>

// ---- Pluggable parallel-for backend (driven from Rust/Rayon) ----------------

extern "C" {

// One unit of MLAS work: run partition `tid`. `task_ctx` is opaque to Rust and
// points back to the C++ trampoline state.
typedef void (*mlas_task_fn)(void* task_ctx, std::ptrdiff_t tid);

// Run `task(task_ctx, tid)` for every tid in [0, iterations) across the backing
// pool. `rust_ctx` is the opaque pointer registered with `mlas_set_threading`.
typedef void (*mlas_parallel_for_fn)(
    void* rust_ctx,
    std::ptrdiff_t iterations,
    mlas_task_fn task,
    void* task_ctx);

// Report the degree of parallelism MLAS may use for its partitioning.
typedef int (*mlas_max_threads_fn)(void* rust_ctx);

}  // extern "C"

namespace {
mlas_parallel_for_fn g_parallel_for = nullptr;
mlas_max_threads_fn g_max_threads = nullptr;
void* g_rust_ctx = nullptr;
}  // namespace

// Register (or clear, with all-null args) the Rust-backed threading backend.
extern "C" void mlas_set_threading(
    mlas_parallel_for_fn parallel_for,
    mlas_max_threads_fn max_threads,
    void* rust_ctx)
{
    g_parallel_for = parallel_for;
    g_max_threads = max_threads;
    g_rust_ctx = rust_ctx;
}

// Hook called by the vendored standalone `MlasGetMaximumThreadCount`.
extern "C" int MlasStandaloneMaxThreads()
{
    if (g_max_threads != nullptr) {
        int n = g_max_threads(g_rust_ctx);
        return n > 0 ? n : 1;
    }
    return 1;
}

// Trampoline that lets the C parallel-for callback invoke a C++
// `std::function<void(ptrdiff_t)>` (the closure MLAS passes to
// `MlasTrySimpleParallel`) without exposing C++ types across FFI.
namespace {
void mlas_std_function_trampoline(void* task_ctx, std::ptrdiff_t tid)
{
    (*static_cast<const std::function<void(std::ptrdiff_t)>*>(task_ctx))(tid);
}
}  // namespace

// Hook called by the vendored standalone `MlasTrySimpleParallel`. Forwards the
// parallel-for onto the registered backend, or runs serially if none is set.
extern "C" void MlasStandaloneParallelFor(std::ptrdiff_t iterations, void* work)
{
    const auto& fn = *static_cast<const std::function<void(std::ptrdiff_t)>*>(work);
    if (g_parallel_for != nullptr && iterations > 1) {
        g_parallel_for(
            g_rust_ctx,
            iterations,
            &mlas_std_function_trampoline,
            const_cast<void*>(static_cast<const void*>(&fn)));
    } else {
        for (std::ptrdiff_t tid = 0; tid < iterations; ++tid) {
            fn(tid);
        }
    }
}

extern "C" void mlas_sgemm(
    int transA,   // 0 = no-transpose, 1 = transpose
    int transB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc)
{
    MLAS_SGEMM_DATA_PARAMS data;
    data.A = A;
    data.lda = lda;
    data.B = B;
    data.ldb = ldb;
    data.C = C;
    data.ldc = ldc;
    data.alpha = alpha;
    data.beta = beta;
    data.BIsPacked = false;

    MlasGemmBatch(
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        M, N, K,
        &data, 1,
        /*ThreadPool=*/nullptr,
        /*BackendKernelSelectorConfig=*/nullptr);
}

// ---- Pre-packed B variant (mirrors how ORT pre-packs constant weights) ----

extern "C" size_t mlas_sgemm_pack_b_size(int transA, int transB, size_t N, size_t K)
{
    return MlasGemmPackBSize(
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        N, K, nullptr);
}

extern "C" void mlas_sgemm_pack_b(
    int transA, int transB, size_t N, size_t K,
    const float* B, size_t ldb, void* packed_b)
{
    MlasGemmPackB(
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        N, K, B, ldb, packed_b, nullptr);
}

extern "C" void mlas_sgemm_packed(
    int transA,
    int transB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const void* packed_b,
    float beta,
    float* C,
    size_t ldc)
{
    MLAS_SGEMM_DATA_PARAMS data;
    data.A = A;
    data.lda = lda;
    data.B = reinterpret_cast<const float*>(packed_b);
    data.ldb = 0;
    data.C = C;
    data.ldc = ldc;
    data.alpha = alpha;
    data.beta = beta;
    data.BIsPacked = true;

    MlasGemmBatch(
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        M, N, K,
        &data, 1,
        /*ThreadPool=*/nullptr,
        /*BackendKernelSelectorConfig=*/nullptr);
}
