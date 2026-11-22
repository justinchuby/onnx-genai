/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    eltwise.cpp

Abstract:

    This module implements routines to compute element-wise operations on two vectors.

    nxrt-mlas-add-retirement: only multiplication for the SiLU reference
    remains here. Element-wise Add runs in the native CPU implementation.

--*/

#include "mlasi.h"

template <>
void
MLASCALL
MlasEltwiseMul<float>(
    const float* left,
    const float* right,
    float* output,
    size_t N
) {
    while (N > 0) {
        if (N >= 4) {
            MLAS_FLOAT32X4 LeftVec = MlasLoadFloat32x4(left);
            MLAS_FLOAT32X4 RightVec = MlasLoadFloat32x4(right);

            MLAS_FLOAT32X4 ResultVec = MlasMultiplyFloat32x4(LeftVec, RightVec);

            MlasStoreFloat32x4(output, ResultVec);

            left += 4;
            right += 4;
            output += 4;
            N -= 4;
        } else {
            *output = *left * *right;

            left += 1;
            right += 1;
            output += 1;
            N -= 1;
        }
    }
}
