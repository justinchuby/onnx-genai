__device__ int affine_nibble(const B* codes, U row, U col, U width) {
    return (codes[row * ((width + 1) / 2) + col / 2] >> ((col % 2) * 4)) & 15;
}

__device__ float affine_grouped_dot(const B* x, const B* w, const B* scales,
    const B* zeros, U row, U token, U k, U block_size, int zero) {
    const U blocks = k / block_size;
    float sum = 0.0f;
    for (U j = 0; j < k; ++j) {
        const U p = j / block_size;
        const int z = zeros ? affine_nibble(zeros, row, p, blocks) : zero;
        const float decoded = __fmul_rn(affine_nibble(w, row, j, k) - z,
                                         read<float>(scales, row * blocks + p));
        sum += read<float>(x, token * k + j) * decoded;
    }
    return sum;
}
