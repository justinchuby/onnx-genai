__device__ U mask(int bits) { return (1ull << bits) - 1; }

template<typename Word, bool unpack>
__device__ void packing(const B* input, B* output, const B* count_input,
                        unsigned* status, U lanes, U count, U words, U inner,
                        U radix, U group, U bits, U limit) {
    constexpr U width = sizeof(Word) * 8;
    if (unpack && read<long long>(count_input, 0) != (long long)count) {
        if (blockIdx.x == 0 && threadIdx.x == 0) *status = 2;
        return;
    }
    GRID(i, lanes) {
        const U span = unpack ? count : words;
        const U j = (i / inner) % span;
        const U outer = i / inner / span;
        const U lane = i % inner;
        if (unpack) {
            U bit = (j / group) * bits;
            U index = 0;
            for (U used = 0; used < bits;) {
                const U off = bit % width;
                const U take = min(bits - used, width - off);
                const U word = read<Word>(input, (outer * words + bit / width) * inner + lane);
                index |= ((word >> off) & mask(take)) << used;
                used += take;
                bit += take;
            }
            if (index >= limit) {
                atomicOr(status, 1u);
                continue;
            }
            for (U k = j % group; k != 0; --k) index /= radix;
            output[i] = index % radix;
        } else {
            const U word_bit = j * width;
            const U groups = count / group + (count % group != 0);
            U value = 0;
            for (U g = word_bit / bits; g < groups && g * bits < word_bit + width; ++g) {
                const U start = g * group;
                U index = 0;
                for (U k = min(group, count - start); k != 0; --k) {
                    const U digit = input[(outer * count + start + k - 1) * inner + lane];
                    if (digit >= radix) atomicOr(status, 1u);
                    index = index * radix + digit % radix;
                }
                const U bit = g * bits;
                value |= bit < word_bit ? index >> (word_bit - bit) : index << (bit - word_bit);
            }
            write<Word>(output, i, value);
        }
    }
}

#define PACK_ENTRY(WORD, NAME, UNPACK) \
extern "C" __global__ void NAME(const B* x, B* y, const B* count_input, unsigned* status, \
                  U lanes, U count, U words, U inner, U radix, U group, U bits, U limit) { \
    packing<WORD, UNPACK>(x,y,count_input,status,lanes,count,words,inner,radix,group,bits,limit); \
}
PACK_ENTRY(B, pack_u8, false)
PACK_ENTRY(unsigned, pack_u32, false)
PACK_ENTRY(B, unpack_u8, true)
PACK_ENTRY(unsigned, unpack_u32, true)
