// FEASIBILITY SPIKE shim: expose a single plain C entry point over MLAS's
// C++ SGEMM API so Rust can call it via FFI without bindgen. Single-threaded
// (NULL threadpool) — the spike only measures the kernel, not MLAS threading.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/mlas/inc/mlas.h"

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
