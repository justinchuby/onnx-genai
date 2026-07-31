// Minimal KleidiAI QNBit micro-kernel registry for the standalone MLAS build.
//
// ONNX Runtime's full kai_ukernel_interface.cpp registers SME/SME2 GEMM,
// convolution, and QGEMM micro-kernels as well. mlas-sys only needs the
// NEON/DotProd/I8MM qsi4 QNBit entries, so keep this translation unit small and
// avoid pulling the unrelated KleidiAI assembly families into the Cargo build.

#include "mlasi.h"

#include "kai_ukernel_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.h"

#define KAI_WRAP_QNBIT_UKERNEL(STEM)                         \
    {                                                        \
        "kai_run_" #STEM,                                   \
        {kai_get_m_step_##STEM,                              \
         kai_get_n_step_##STEM,                              \
         kai_get_mr_##STEM,                                  \
         kai_get_nr_##STEM,                                  \
         kai_get_kr_##STEM,                                  \
         kai_get_sr_##STEM,                                  \
         kai_get_lhs_packed_offset_##STEM,                   \
         kai_get_rhs_packed_offset_##STEM,                   \
         kai_get_dst_offset_##STEM,                          \
         kai_get_dst_size_##STEM,                            \
         kai_run_##STEM}                                     \
    }

namespace {

const KaiQnbitGemmKernel kai_qnbit_gemv_dotprod =
    KAI_WRAP_QNBIT_UKERNEL(matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod);

const KaiQnbitGemmKernel kai_qnbit_gemv_dotprod_n8 =
    KAI_WRAP_QNBIT_UKERNEL(matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod);

const KaiQnbitGemmKernel kai_qnbit_gemm_dotprod =
    KAI_WRAP_QNBIT_UKERNEL(matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod);

const KaiQnbitGemmKernel kai_qnbit_gemm_i8mm =
    KAI_WRAP_QNBIT_UKERNEL(matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm);

}  // namespace

const KaiQnbitGemmKernel& GetKleidiAIGemmUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_I8MM()) {
        return kai_qnbit_gemm_i8mm;
    }
    return kai_qnbit_gemm_dotprod;
}

const KaiQnbitGemmKernel& GetKleidiAIGemvUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_I8MM()) {
        return kai_qnbit_gemv_dotprod_n8;
    }
    return kai_qnbit_gemv_dotprod;
}

