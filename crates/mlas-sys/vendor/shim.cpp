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
#include "core/mlas/inc/mlas_qnbit.h"

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

// ---- Blocked n-bit quantized GEMM (SQNBitGemm) ------------------------------
//
// Plain-C wrappers over MLAS's templated `MlasQNBitGemmBatch<float>` and its
// pack/query helpers so Rust can drive the int4/int8 blockwise-quantized
// MatMulNBits decode path without binding the C++ template/struct directly.
//
// `comp_type` is the raw `MLAS_QNBIT_GEMM_COMPUTE_TYPE` value:
//   0 = SQNBIT_CompFp32 (fp32 activation, fp32 accumulate)
//   3 = SQNBIT_CompInt8 (int8 activation, int32 accumulate) -- accuracy_level=4.
//
// Threading: like the SGEMM shim, MLAS's own N/M tile partitioning is routed
// through the registered Rust/Rayon parallel-for backend (`MlasStandalone*`
// hooks above). `MlasQNBitGemmBatch` only takes its parallel branch when
// `ThreadPool != nullptr`, so pass a non-null sentinel (the pointer is never
// dereferenced in the standalone build -- `MlasGetMaximumThreadCount` and
// `MlasTrySimpleParallel` both ignore it) to enable multi-threading.

extern "C" int mlas_qnbit_gemm_available(size_t bits, size_t blk_len, int comp_type)
{
    return MlasIsQNBitGemmAvailable(
               bits, blk_len, static_cast<MLAS_QNBIT_GEMM_COMPUTE_TYPE>(comp_type))
               ? 1
               : 0;
}

extern "C" size_t mlas_qnbit_gemm_pack_b_size(
    size_t n, size_t k, size_t bits, size_t blk_len, int has_zp, int comp_type)
{
    return MlasQNBitGemmPackQuantBDataSize(
        n, k, bits, blk_len, has_zp != 0,
        static_cast<MLAS_QNBIT_GEMM_COMPUTE_TYPE>(comp_type),
        /*BackendKernelSelectorConfig=*/nullptr);
}

extern "C" void mlas_qnbit_gemm_pack_b(
    size_t n,
    size_t k,
    size_t bits,
    size_t blk_len,
    int comp_type,
    const void* quant_b_data,
    void* packed_b,
    const void* quant_b_scale,
    int has_zp,
    const void* quant_b_zero_point)
{
    MlasQNBitGemmPackQuantBData(
        n, k, bits, blk_len,
        static_cast<MLAS_QNBIT_GEMM_COMPUTE_TYPE>(comp_type),
        quant_b_data,
        packed_b,
        quant_b_scale,
        has_zp != 0,
        quant_b_zero_point,
        /*ThreadPool=*/nullptr,
        /*BackendKernelSelectorConfig=*/nullptr);
}

extern "C" size_t mlas_qnbit_gemm_workspace_size(
    size_t m, size_t n, size_t k, size_t bits, size_t blk_len, int has_zp, int comp_type)
{
    return MlasQNBitGemmBatchWorkspaceSize(
        m, n, k, /*BatchN=*/1, bits, blk_len, has_zp != 0,
        static_cast<MLAS_QNBIT_GEMM_COMPUTE_TYPE>(comp_type),
        /*BackendKernelSelectorConfig=*/nullptr);
}

extern "C" void mlas_qnbit_gemm(
    size_t m,
    size_t n,
    size_t k,
    size_t bits,
    size_t blk_len,
    int comp_type,
    const float* a,
    size_t lda,
    const void* packed_b,
    const float* quant_b_scale,
    int has_zp,
    const void* quant_b_zero_point,
    const float* bias,
    float* c,
    size_t ldc,
    void* workspace,
    int multithread)
{
    MLAS_QNBIT_GEMM_DATA_PARAMS<float> params;
    params.A = a;
    params.lda = lda;
    params.Bias = bias;
    params.C = c;
    params.ldc = ldc;

    const auto ct = static_cast<MLAS_QNBIT_GEMM_COMPUTE_TYPE>(comp_type);
    if (ct == SQNBIT_CompInt8) {
        // The int8-compute path derives PackedQuantBData / QuantBScale /
        // QuantBBlkSum from the combined workspace produced by
        // MlasQNBitGemmPackQuantBData (which baked scale + zero point into the
        // block sums), so only the workspace pointer is needed here.
        params.QuantBDataWorkspace = packed_b;
    } else {
        // The fp32-compute path repacks only the quantized nibbles; scales and
        // (optional) zero points are consumed at compute time in their original
        // ONNX layout.
        params.PackedQuantBData = static_cast<const std::byte*>(packed_b);
        params.QuantBScale = quant_b_scale;
        params.QuantBZeroPoint = has_zp != 0 ? quant_b_zero_point : nullptr;
    }

    MLAS_THREADPOOL* thread_pool =
        multithread != 0 ? reinterpret_cast<MLAS_THREADPOOL*>(1) : nullptr;

    MlasQNBitGemmBatch<float>(
        m, n, k, /*BatchN=*/1, bits, blk_len, ct, &params, workspace, thread_pool,
        /*BackendKernelSelectorConfig=*/nullptr);
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
