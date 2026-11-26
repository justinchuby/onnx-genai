extern "C" __global__ void select_experts(const B* probabilities, B* ids, B* weights,
    U tokens, U experts, U k) {
    GRID(token, tokens) {
        for (U slot = 0; slot < k; ++slot) {
            long long best = -1;
            float value = 0.0f;
            for (U expert = 0; expert < experts; ++expert) {
                bool used = false;
                for (U prior = 0; prior < slot; ++prior)
                    used |= read<long long>(ids, token * k + prior) == (long long)expert;
                if (used) continue;
                const float p = read<float>(probabilities, token * experts + expert);
                if (best < 0 || before(p, value, (long long)expert, best, 1)) {
                    best = (long long)expert;
                    value = p;
                }
            }
            write<long long>(ids, token * k + slot, best);
            write<float>(weights, token * k + slot, value);
        }
    }
}

__device__ void validate_scales(const B* scales, U count, unsigned* status) {
    GRID(i, count) {
        const float s = read<float>(scales, i);
        if (!(s > 0.0f) || !isfinite(s)) atomicOr(status, 2u);
    }
}

extern "C" __global__ void project_activate(const B* x, const B* codes, const B* scales,
    const B* zeros, const B* ids, B* activated, unsigned* status, U scale_count,
    U routes, U k, U hidden, U inter, U block, int zero) {
    validate_scales(scales, scale_count, status);
    GRID(j, routes * inter) {
        const U route = j / inter, col = j % inter;
        const U expert = (U)read<long long>(ids, route);
        const U gate_row = expert * (2 * inter) + col;
        const float gate = affine_grouped_dot(x, codes, scales, zeros, gate_row, route / k, hidden, block, zero);
        const float up = affine_grouped_dot(x, codes, scales, zeros, gate_row + inter, route / k, hidden, block, zero);
        const float silu = __fdiv_rn(gate, __fadd_rn(1.0f, expf(-gate)));
        write<float>(activated, j, __fmul_rn(silu, up));
    }
}

extern "C" __global__ void project_combine(const B* activated, const B* codes, const B* scales,
    const B* zeros, const B* ids, const B* weights, B* output, unsigned* status,
    U scale_count, U tokens, U k, U hidden, U inter, U block, int zero) {
    validate_scales(scales, scale_count, status);
    const unsigned slot = threadIdx.x % warpSize;
    const U warp_count = blockDim.x / warpSize;
    for (U j = (U)blockIdx.x * warp_count + threadIdx.x / warpSize; j < tokens * hidden;
         j += (U)gridDim.x * warp_count) {
        const U token = j / hidden, col = j % hidden;
        float sum = 0.0f;
        if (slot < k) {
            const U route = token * k + slot;
            const U row = (U)read<long long>(ids, route) * hidden + col;
            const float down = affine_grouped_dot(activated, codes, scales, zeros, row, route, inter, block, zero);
            // Deliberately multiply even a zero probability: the graph has no mask.
            sum = __fadd_rn(sum, __fmul_rn(down, read<float>(weights, route)));
        }
        // The standard block ReduceSum combines descending power-of-two offsets.
        // Above one warp all other lanes are zero for this admitted K.
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, offset));
        if (slot == 0) write<float>(output, j, sum);
    }
}
