//! GPU-side expert grouping and activation gather for QMoE prefill.

pub(super) const MODULE: &str = "qmoe_expert_grouping_v1";
pub(super) const INIT_ENTRY: &str = "qmoe_group_init";
pub(super) const COUNT_ENTRY: &str = "qmoe_group_count";
pub(super) const PREFIX_ENTRY: &str = "qmoe_group_prefix";
pub(super) const ASSIGN_ENTRY: &str = "qmoe_group_assign";
pub(super) const GATHER_F32_ENTRY: &str = "qmoe_gather_f32";
pub(super) const GATHER_F16_ENTRY: &str = "qmoe_gather_f16";
pub(super) const GATHER_BF16_ENTRY: &str = "qmoe_gather_bf16";

pub(super) const CUDA_SRC: &str = r#"
#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#define QMOE_HAS_HALF 1
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

extern "C" __global__ void qmoe_group_init(
    unsigned long long* counts,
    unsigned long long* offsets,
    unsigned long long* cursors,
    unsigned long long* grouped_routes,
    const unsigned long long routes,
    const int experts)
{
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    const unsigned long long expert_entries = (unsigned long long)experts + 1ull;
    const unsigned long long total =
        routes > expert_entries ? routes : expert_entries;
    for (unsigned long long index = first; index < total; index += stride) {
        if (index < (unsigned long long)experts) {
            counts[index] = 0ull;
            cursors[index] = 0ull;
        }
        if (index < expert_entries) {
            offsets[index] = 0ull;
        }
        if (index < routes) {
            grouped_routes[index] = routes;
        }
    }
}

extern "C" __global__ void qmoe_group_count(
    const int* selected_experts,
    unsigned long long* counts,
    const unsigned long long routes,
    const int experts)
{
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long route = first; route < routes; route += stride) {
        const int expert = selected_experts[route];
        if (expert >= 0 && expert < experts) {
            atomicAdd(counts + expert, 1ull);
        }
    }
}

extern "C" __global__ void qmoe_group_prefix(
    const unsigned long long* counts,
    unsigned long long* offsets,
    const unsigned long long routes,
    const int experts)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    unsigned long long running = 0ull;
    offsets[0] = 0ull;
    for (int expert = 0; expert < experts; ++expert) {
        const unsigned long long count = counts[expert];
        running = count <= routes - running ? running + count : routes;
        offsets[(unsigned long long)expert + 1ull] = running;
    }
}

extern "C" __global__ void qmoe_group_assign(
    const int* selected_experts,
    const unsigned long long* offsets,
    unsigned long long* cursors,
    unsigned long long* grouped_routes,
    const unsigned long long routes,
    const int experts)
{
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long route = first; route < routes; route += stride) {
        const int expert = selected_experts[route];
        if (expert < 0 || expert >= experts) {
            continue;
        }
        const unsigned long long local = atomicAdd(cursors + expert, 1ull);
        const unsigned long long offset = offsets[expert];
        if (offset <= routes && local < routes - offset) {
            grouped_routes[offset + local] = route;
        }
    }
}

template <typename Input>
__device__ __forceinline__ float qmoe_group_load(
    const Input* input, unsigned long long index);

template <>
__device__ __forceinline__ float qmoe_group_load<float>(
    const float* input, unsigned long long index)
{
    return input[index];
}

#ifdef QMOE_HAS_HALF
template <>
__device__ __forceinline__ float qmoe_group_load<__half>(
    const __half* input, unsigned long long index)
{
    return __half2float(input[index]);
}

template <>
__device__ __forceinline__ float qmoe_group_load<__nv_bfloat16>(
    const __nv_bfloat16* input, unsigned long long index)
{
    return __bfloat162float(input[index]);
}
#endif

template <typename Input>
__device__ void qmoe_gather_impl(
    const Input* input,
    const unsigned long long* grouped_routes,
    float* grouped_input,
    const unsigned long long routes,
    const unsigned long long input_rows,
    const int input_rows_are_routes,
    const int top_k,
    const int features)
{
    const unsigned long long total = routes * (unsigned long long)features;
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long index = first; index < total; index += stride) {
        const unsigned long long grouped_row = index / (unsigned long long)features;
        const int feature = (int)(index % (unsigned long long)features);
        const unsigned long long route = grouped_routes[grouped_row];
        const unsigned long long input_row = input_rows_are_routes
            ? route
            : route / (unsigned long long)top_k;
        grouped_input[index] = route < routes && input_row < input_rows
            ? qmoe_group_load(
                input,
                input_row * (unsigned long long)features
                    + (unsigned long long)feature)
            : 0.0f;
    }
}

extern "C" __global__ void qmoe_gather_f32(
    const float* input,
    const unsigned long long* grouped_routes,
    float* grouped_input,
    const unsigned long long routes,
    const unsigned long long input_rows,
    const int input_rows_are_routes,
    const int top_k,
    const int features)
{
    qmoe_gather_impl(
        input, grouped_routes, grouped_input, routes, input_rows,
        input_rows_are_routes, top_k, features);
}

#ifdef QMOE_HAS_HALF
extern "C" __global__ void qmoe_gather_f16(
    const __half* input,
    const unsigned long long* grouped_routes,
    float* grouped_input,
    const unsigned long long routes,
    const unsigned long long input_rows,
    const int input_rows_are_routes,
    const int top_k,
    const int features)
{
    qmoe_gather_impl(
        input, grouped_routes, grouped_input, routes, input_rows,
        input_rows_are_routes, top_k, features);
}

extern "C" __global__ void qmoe_gather_bf16(
    const __nv_bfloat16* input,
    const unsigned long long* grouped_routes,
    float* grouped_input,
    const unsigned long long routes,
    const unsigned long long input_rows,
    const int input_rows_are_routes,
    const int top_k,
    const int features)
{
    qmoe_gather_impl(
        input, grouped_routes, grouped_input, routes, input_rows,
        input_rows_are_routes, top_k, features);
}
#endif
"#;
