//! Tiled affine block-dequant GEMM for grouped QMoE expert rows.

use std::sync::OnceLock;

pub(super) const ENTRY: &str = "qmoe_grouped_linear_f32";

const PREFIX: &str = r#"
__device__ __forceinline__ float qmoe_grouped_decode_weight(
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const int expert,
    const int output,
    const int depth,
    const int out_features,
    const int packed_in,
    const int blocks,
    const int zero_point_bytes,
    const int bits,
    const int block_size)
{
    const int pack_size = 8 / bits;
    const unsigned long long expert_row =
        (unsigned long long)expert * out_features + output;
    const unsigned char byte =
        packed[expert_row * packed_in + depth / pack_size];
    const int mask = bits == 8 ? 255 : ((1 << bits) - 1);
    const int quantized = (byte >> ((depth % pack_size) * bits)) & mask;
    const int block = depth / block_size;
    int zero_point = 1 << (bits - 1);
    if (zero_points) {
        const unsigned char packed_zero =
            zero_points[expert_row * zero_point_bytes + block / pack_size];
        zero_point =
            (packed_zero >> ((block % pack_size) * bits)) & mask;
    }
    return ((float)quantized - (float)zero_point)
        * scales[expert_row * blocks + block];
}

__device__ __forceinline__ float qmoe_grouped_block_sum(float value, int row)
{
    extern __shared__ float warp_sums[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
        warp_sums[row * warps + warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < warps ? warp_sums[row * warps + lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
    }
    return value;
}
"#;

const KERNEL: &str = r#"
extern "C" __global__ void qmoe_grouped_linear_f32(
    const float* grouped_input,
    const unsigned long long* grouped_routes,
    const unsigned long long* counts,
    const unsigned long long* offsets,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const unsigned long long routes,
    const unsigned long long tasks,
    const unsigned long long gemm_min_tokens,
    const int experts,
    const int out_features,
    const int in_features,
    const int packed_in,
    const int blocks,
    const int zero_point_bytes,
    const int bits,
    const int block_size)
{
    for (unsigned long long task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int expert = (int)(task / (unsigned long long)out_features);
        const int output_feature = (int)(task % (unsigned long long)out_features);
        if (expert < 0 || expert >= experts) {
            continue;
        }
        const unsigned long long count = counts[expert];
        const unsigned long long expert_offset = offsets[expert];
        if (count < gemm_min_tokens
            || expert_offset > routes
            || count > routes - expert_offset) {
            continue;
        }

        for (unsigned long long row_base = 0; row_base < count;
             row_base += QMOE_TILE_M) {
            float values[QMOE_TILE_M] = {0.0f};
            for (int depth = (int)threadIdx.x; depth < in_features;
                 depth += (int)blockDim.x) {
                const float weight = qmoe_grouped_decode_weight(
                    packed, scales, zero_points, expert, output_feature, depth,
                    out_features, packed_in, blocks, zero_point_bytes, bits,
                    block_size);
#pragma unroll
                for (int row = 0; row < QMOE_TILE_M; ++row) {
                    const unsigned long long local_row =
                        row_base + (unsigned long long)row;
                    if (local_row < count) {
                        const unsigned long long grouped_row =
                            expert_offset + local_row;
                        values[row] += grouped_input[
                            grouped_row * (unsigned long long)in_features
                                + (unsigned long long)depth] * weight;
                    }
                }
            }

#pragma unroll
            for (int row = 0; row < QMOE_TILE_M; ++row) {
                const float value = qmoe_grouped_block_sum(values[row], row);
                const unsigned long long local_row =
                    row_base + (unsigned long long)row;
                if (threadIdx.x == 0 && local_row < count) {
                    const unsigned long long grouped_row =
                        expert_offset + local_row;
                    const unsigned long long route = grouped_routes[grouped_row];
                    if (route < routes) {
                        const unsigned long long bias_index =
                            (unsigned long long)expert * out_features
                                + (unsigned long long)output_feature;
                        output[
                            route * (unsigned long long)out_features
                                + (unsigned long long)output_feature] =
                            value + (bias ? bias[bias_index] : 0.0f);
                    }
                }
                __syncthreads();
            }
        }
    }
}
"#;

pub(super) fn tile_for(
    compute_capability: (u32, u32),
    threads: u32,
    max_shared_memory: u32,
) -> u32 {
    let preferred = if compute_capability.0 >= 8 {
        8
    } else if compute_capability.0 >= 7 {
        4
    } else {
        2
    };
    [8, 4, 2, 1]
        .into_iter()
        .find(|&tile| {
            tile <= preferred
                && threads
                    .checked_mul(tile)
                    .and_then(|value| value.checked_mul(4))
                    .is_some_and(|bytes| bytes <= max_shared_memory)
        })
        .unwrap_or(1)
}

pub(super) fn module_source(tile: u32) -> (&'static str, &'static str) {
    fn build(tile: u32) -> String {
        format!("{PREFIX}\n#define QMOE_TILE_M {tile}\n{KERNEL}")
    }

    static TILE_1: OnceLock<String> = OnceLock::new();
    static TILE_2: OnceLock<String> = OnceLock::new();
    static TILE_4: OnceLock<String> = OnceLock::new();
    static TILE_8: OnceLock<String> = OnceLock::new();
    match tile {
        1 => ("qmoe_grouped_gemm_tile1", TILE_1.get_or_init(|| build(1))),
        2 => ("qmoe_grouped_gemm_tile2", TILE_2.get_or_init(|| build(2))),
        4 => ("qmoe_grouped_gemm_tile4", TILE_4.get_or_init(|| build(4))),
        8 => ("qmoe_grouped_gemm_tile8", TILE_8.get_or_init(|| build(8))),
        _ => unreachable!("QMoE GEMM tile must be one of 1, 2, 4, or 8"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_selection_is_sm_general_and_falls_back_for_shared_memory() {
        assert_eq!(tile_for((9, 0), 256, 48 * 1024), 8);
        assert_eq!(tile_for((8, 6), 256, 48 * 1024), 8);
        assert_eq!(tile_for((7, 5), 256, 48 * 1024), 4);
        assert_eq!(tile_for((7, 0), 256, 3 * 1024), 2);
        assert_eq!(tile_for((7, 0), 256, 1024), 1);
    }
}
