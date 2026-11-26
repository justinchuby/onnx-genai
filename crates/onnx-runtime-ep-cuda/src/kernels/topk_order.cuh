__device__ bool before(float a, float b, long long ia, long long ib, int largest) {
    int ka = __float_as_int(a);
    int kb = __float_as_int(b);
    ka ^= (int)(((unsigned int)(ka >> 31)) >> 1);
    kb ^= (int)(((unsigned int)(kb >> 31)) >> 1);
    if (ka == kb) return ia < ib;
    return largest ? ka > kb : ka < kb;
}
