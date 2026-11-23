// DequantizePacked-29. GGML-derived decoding: see the MIT notice in
// onnx-runtime-quantization. The shared block decoder prelude precedes this file.

__device__ int plane(const unsigned char* b, int j, int bits, int group) {
    const int fields = 8 / bits;
    return (b[j / (group * fields) * group + j % group] >>
            (bits * (j / group % fields))) & ((1 << bits) - 1);
}

__device__ float half_at(const unsigned char* b, int offset) {
    return fp16_to_fp32(load_u16_le(b + offset));
}

// These case labels are ONNX profile IDs (GGML tensor IDs), not the private
// BlockQuantizedMatMul kernel IDs. Reused profiles are routed by the factory.
__device__ float decode_standalone(const unsigned char* b, int kind, int j) {
    switch (kind) {
    case 2: case 3: case 6: case 7: {
        const bool five = kind == 6 || kind == 7;
        const bool affine = kind == 3 || kind == 7;
        const int h = affine ? 4 : 2;
        const int q = plane(b + h + (five ? 4 : 0), j, 4, 16)
                      + (five ? 16 * plane(b + h, j, 1, 1) : 0);
        const float d = half_at(b, 0);
        return affine ? __fadd_rn(__fmul_rn(d, q), half_at(b, 2))
                      : d * (q - (five ? 16 : 8));
    }
    case 9:
        return half_at(b, 0) * (signed char)b[4 + j];
    case 12: {
        int a, m;
        q4_k_scale_min(b + 4, j / 32, &a, &m);
        return __fsub_rn(__fmul_rn(half_at(b, 0) * a, plane(b + 16, j, 4, 32)),
                         __fmul_rn(half_at(b, 2), m));
    }
    case 15:
        return __uint_as_float(load_u32_le(b)) * (signed char)b[4 + j];
    case 34: {
        const int start = j < 160 ? 0 : (j < 240 ? 32 : 48);
        const int width = j < 160 ? 32 : (j < 240 ? 16 : 4);
        const int offset = j - (j < 160 ? 0 : (j < 240 ? 160 : 240));
        const int powers[] = {1, 3, 9, 27, 81};
        const int wrapped = (b[start + offset % width] * powers[offset / width]) & 255;
        return half_at(b, 52) * (wrapped * 3 / 256 - 1);
    }
    case 35:
        return half_at(b, 64) * (plane(b, j, 2, 32) - 1);
    case 39: {
        // ONNX treats exponent byte 255 as a finite half-scale, unlike
        // the private matmul E8M0 helper.
        const unsigned int e = b[0];
        const unsigned int bits = e < 2 ? 0x00200000u << e : (e - 1) << 23;
        return __uint_as_float(bits) * e2m1_doubled[plane(b + 1, j, 4, 16)];
    }
    case 40: {
        const unsigned int e = b[j / 16], m = e & 7, exp = (e >> 3) & 15;
        const float d = e == 0 || e == 127 ? 0.0f
            : (exp == 0 ? m * (1.0f / 1024.0f)
                        : __uint_as_float(((exp + 119) << 23) | (m << 20)));
        return d * e2m1_doubled[plane(b + 4, j, 4, 8)];
    }
    case 41:
        return plane(b + 2, j, 1, 1) ? half_at(b, 0) : -half_at(b, 0);
    case 42:
        return half_at(b, 0) * (plane(b + 2, j, 2, 1) - 1);
    }
    return __uint_as_float(0x7fc00000u);  // unreachable for validated formats
}

__device__ unsigned short to_bfloat16(float value) {
    const unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
    return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}

extern "C" __global__ void dequantize_packed(
    const unsigned char* x, unsigned char* y, unsigned long long count,
    int width, int bytes, int kind, int shared_kind, int dtype)
{
    unsigned long long j = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (; j < count; j += stride) {
        const unsigned char* block = x + j / width * bytes;
        const int lane = j % width;
        const float value = shared_kind < 0 ? decode_standalone(block, kind, lane)
                                           : decode_block_weight<true>(block, shared_kind, lane);
        switch (dtype) {
        case 1: reinterpret_cast<float*>(y)[j] = value; break;
        case 11: reinterpret_cast<double*>(y)[j] = value; break;
        case 10: {
            unsigned short f16;
            asm("cvt.rn.f16.f32 %0, %1;" : "=h"(f16) : "f"(value));
            reinterpret_cast<unsigned short*>(y)[j] = f16;
            break;
        }
        case 16: reinterpret_cast<unsigned short*>(y)[j] = to_bfloat16(value); break;
        }
    }
}
