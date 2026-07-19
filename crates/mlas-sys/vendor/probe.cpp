// FEASIBILITY SPIKE probe: report which f32 GEMM microkernel MLAS's runtime
// dispatch selected on this host, to prove the AVX-512 kernel (not the AVX2/
// FMA3 fallback) is active. Uses MLAS internals (mlasi.h), so it is compiled
// in the same group that defines BUILD_MLAS_NO_ONNXRUNTIME.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlasi.h"

extern "C" int mlas_float_kernel_id()
{
#if defined(MLAS_TARGET_AMD64_IX86)
    auto* k = GetMlasPlatform().GemmFloatKernel;
    if (k == MlasGemmFloatKernelAvx512F) {
        return 512;
    }
    if (k == MlasGemmFloatKernelFma3) {
        return 3;
    }
    if (k == MlasGemmFloatKernelAvx) {
        return 1;
    }
    return -1;
#else
    return 0;
#endif
}
