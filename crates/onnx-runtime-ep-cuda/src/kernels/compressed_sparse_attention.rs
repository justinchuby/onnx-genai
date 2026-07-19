//! `pkg.nxrt::CompressedSparseAttention` v1: correctness-first, **host-staged**
//! CUDA execution of the DeepSeek-V4-Flash / GLM-5.2 compressed sparse-attention
//! (CSA) operator.
//!
//! The fully-implemented CPU kernel in
//! `crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs` is the
//! authoritative numerical oracle for this op. Re-deriving its ~4.6k lines of
//! frozen-contract math (learned FP8/FP4 compression, the ratio-4 index-key
//! stream, sparse sink-softmax, and the stateful compressed KV cache/carry
//! lifecycle) on the device would be error-prone and is explicitly a later,
//! separately-tracked phase. This kernel therefore guarantees bit-parity by
//! **delegating to the CPU kernel itself**:
//!
//! 1. every device input tensor is copied host-side (D2H),
//! 2. the CPU `CompressedSparseAttention` kernel — built by the CPU factory from
//!    the same node, so it carries the identical attribute configuration — runs
//!    over host-resident views, producing every output (`Y`, the present
//!    compressed KV cache, the present compression carry, and, for ratio-4, the
//!    present index key / index carry / selected indices),
//! 3. each host output is uploaded back to its device buffer (H2D).
//!
//! ## Statefulness
//!
//! CSA is stateful, but the state is threaded through the graph as ordinary
//! `past_* → present_*` input/output tensors (the standard ONNX KV-cache
//! pattern), not held inside the kernel. A `prefill → decode → decode` sequence
//! feeds each step's `present_*` outputs back in as the next step's `past_*`
//! inputs. Because this kernel reuses the CPU kernel verbatim, the entire
//! compressed-cache / carry / index-carry lifecycle is reproduced exactly, and
//! the host-resident staging keeps state correct across steps (device-resident
//! state is the Phase-B optimization).
//!
//! ## `cuda_graph_compatible` = false
//!
//! Like the correctness-first `standard_attention` / `sparse_kv_gather`
//! kernels, execution round-trips through host memory and synchronizes the
//! stream on every D2H/H2D copy, neither of which is legal during CUDA-graph
//! capture.
//!
//! ## Claim-time gating
//!
//! [`unsupported_reason`] rejects, at claim time, any ratio / cache-layout /
//! sink-mode / dtype / arity combination the CPU oracle does not accept (by
//! dry-running the CPU factory, which validates the full frozen-v1 attribute set,
//! plus explicit checks on the dtype-fixed inputs). This upholds the doc §4.8
//! contract: "`supports_op` must reject unsupported ratio/layout/dtype/shape
//! combinations instead of claiming the node and falling back inside the kernel."
//!
// TODO(csa-cuda phase B): replace this host-staged path with a device-resident
// fused CSA kernel (device-resident compressed cache/carry, fused
// selection/score/sink-softmax/value-reduction, CUDA-graph capture, no host
// round trip). See docs/DEEPSEEK_CSA_MTP_RUNTIME.md §4.8.

use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ep_cpu::kernels::compressed_sparse_attention::CompressedSparseAttentionFactory as CpuCsaFactory;
use onnx_runtime_ir::{DataType, DeviceId, Dim, Node, Shape, as_static_shape};

use crate::error::{driver_err, not_implemented};
use crate::kernels::block_quant;
use crate::kernels::csa_device_state::{CsaBufferLayout, CsaDeviceBufferManager};
use crate::runtime::{CudaRuntime, cuptr};

const OP: &str = "CompressedSparseAttention";

/// Device stage-7 (sparse sink-softmax attention) + stage-6 (candidate read)
/// for **ratio-128** (B1). One CUDA block computes one `(batch, query, head)`
/// output row, reproducing the CPU oracle's `ratio128_attention` numerics
/// **bit-for-bit**:
///
/// * score/value reductions accumulate in pure f32 in **ascending candidate
///   index order** (dense window first, then compressed records), matching the
///   oracle's `dot` / `accumulate_value`. `__fadd_rn` / `__fmul_rn` keep every
///   multiply-add un-fused so the device sum equals the CPU's non-FMA order.
/// * the softmax is a faithful **two-pass** max → denominator → value reduction
///   (NOT an online rescale), because a running rescale would reorder the f32
///   sum and diverge from the oracle.
/// * the learned `head_sink` is added to the denominator **after** the running
///   max, as a logit-only mass (no value contribution).
/// * `exp` is evaluated as `(float)exp((double)x)` — the same double-precision
///   evaluation the GQA reference kernel uses to match glibc's correctly-rounded
///   `expf` bit-for-bit.
///
/// An invalid/skipped candidate (the fused `-1` index: dense position past the
/// causal limit, before the `current_kv` window, or a not-yet-completed
/// compressed record) is represented by a `-inf` score and excluded from both
/// the denominator and the value reduction, exactly like the oracle.
const ATTENTION_MODULE: &str = "csa_ratio128_attention";
const ATTENTION_ENTRY: &str = "csa_ratio128_sink_attention";
const RATIO4_ATTENTION_MODULE: &str = "csa_ratio4_attention";
const RATIO4_ATTENTION_ENTRY: &str = "csa_ratio4_sink_attention";
const ATTENTION_BLOCK: u32 = 256;

/// Stage-1 ratio-128 compressor.  Deliberately one thread owns a batch row: the
/// oracle specifies an order-dependent f32 reduction for each dimension, so
/// parallel reductions are not interchangeable here.  Decode uses S=1 and the
/// short prefill is amortised by the later sparse-attention work.
const COMPRESSION_MODULE: &str = "csa_ratio128_compression";
const COMPRESSION_ENTRY: &str = "csa_ratio128_compress";
const INDEX_COMPRESSION_MODULE: &str = "csa_ratio4_index_compression";
const INDEX_COMPRESSION_ENTRY: &str = "csa_ratio4_index_compress";
/// B4 stage-3/4/5 ratio-4 index scoring + deterministic top-k selection.
const INDEX_SELECT_MODULE: &str = "csa_ratio4_index_select";
const INDEX_SELECT_ENTRY: &str = "csa_ratio4_index_select";
const COMPRESSION_SOURCE: &str = r#"
__device__ __forceinline__ unsigned short csa_bf16_bits(float x) {
    unsigned int bits = __float_as_uint(x);
    return (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float csa_bf16(float x) {
    return __uint_as_float((unsigned int)csa_bf16_bits(x) << 16);
}
extern "C" __global__ void csa_ratio128_compress(
    const float* kv, const float* gate, const float* ape, const float* norm,
    const float* past_carry, const unsigned char* past_cache,
    float* carry, unsigned char* cache,
    int batch, int sequence, int dim, int past_records, int cache_records, int cache_fp8,
    long long start)
{
    const int b = blockIdx.x;
    if (b >= batch || threadIdx.x != 0) return;
    const int carry_stride = 2 * 128 * dim;
    const int cache_width = cache_fp8 ? 583 : dim * 4;
    // The graph outputs are the next state.  Copy only the old records/carry;
    // newly completed records are written below.
    for (int i = 0; i < carry_stride; ++i) carry[b * carry_stride + i] = past_carry[b * carry_stride + i];
    for (int i = 0; i < past_records * cache_width; ++i)
        cache[b * cache_records * cache_width + i] = past_cache[b * past_records * cache_width + i];
    if (start == 0) {
        // A completed ratio-128 block consumes every slot.  Clear the complete
        // block after finalizing it (the next token writes only its own slot).
        for (int reset_slot = 0; reset_slot < 128; ++reset_slot)
            for (int d = 0; d < dim; ++d) {
                carry[b * carry_stride + (2 * reset_slot) * dim + d] = 0.0f;
                carry[b * carry_stride + (2 * reset_slot + 1) * dim + d] = __int_as_float(0xff800000);
            }
    }
    int emitted = 0;
    for (int s = 0; s < sequence; ++s) {
        const long long pos = start + s;
        const int slot = (int)(pos & 127);
        for (int d = 0; d < dim; ++d) {
            carry[b * carry_stride + (2 * slot) * dim + d] = kv[((b * sequence + s) * dim) + d];
            carry[b * carry_stride + (2 * slot + 1) * dim + d] =
                __fadd_rn(gate[((b * sequence + s) * dim) + d], ape[slot * dim + d]);
        }
        if (((pos + 1) & 127) != 0) continue;
        float record[512];
        for (int d = 0; d < dim; ++d) {
            float maximum = __int_as_float(0xff800000);
            for (int j = 0; j < 128; ++j)
                maximum = fmaxf(maximum, carry[b * carry_stride + (2 * j + 1) * dim + d]);
            float denominator = 0.0f, numerator = 0.0f;
            for (int j = 0; j < 128; ++j) {
                float score = carry[b * carry_stride + (2 * j + 1) * dim + d];
                if (score == __int_as_float(0xff800000)) continue;
                float weight = (float)exp((double)__fsub_rn(score, maximum));
                denominator = __fadd_rn(denominator, weight);
                numerator = __fadd_rn(numerator, __fmul_rn(weight, carry[b * carry_stride + (2 * j) * dim + d]));
            }
            record[d] = csa_bf16(__fdiv_rn(numerator, denominator));
        }
        float square_sum = 0.0f;
        for (int d = 0; d < dim; ++d)
            square_sum = __fadd_rn(square_sum, __fmul_rn(record[d], record[d]));
        float inverse_rms = __frcp_rn(__fsqrt_rn(__fadd_rn(__fdiv_rn(square_sum, (float)dim), 1.0e-6f)));
        for (int d = 0; d < dim; ++d)
            record[d] = csa_bf16(__fmul_rn(__fmul_rn(record[d], inverse_rms), norm[d]));
        // The compressed RoPE tail is BF16-rounded after each component, just
        // as the frozen CPU finalize path does.
        for (int pair = 0; pair < 32; ++pair) {
            float ramp = fminf(1.0f, fmaxf(0.0f, ((float)pair - 15.0f) / 10.0f));
            float base = powf(160000.0f, -((float)(2 * pair)) / 64.0f);
            float frequency = __fadd_rn(__fmul_rn(base, 1.0f - ramp), __fmul_rn(base / 16.0f, ramp));
            float sn, cs; sincosf((float)(pos - 127) * frequency, &sn, &cs);
            int d = 448 + 2 * pair; float re = record[d], im = record[d + 1];
            record[d] = csa_bf16(__fsub_rn(__fmul_rn(re, cs), __fmul_rn(im, sn)));
            record[d + 1] = csa_bf16(__fadd_rn(__fmul_rn(re, sn), __fmul_rn(im, cs)));
        }
        const int out = past_records + emitted++;
        if (cache_fp8) {
            unsigned char* dst = cache + (b * cache_records + out) * 583;
            for (int block = 0; block < 7; ++block)
                quantize_fp8_e4m3_block(record + block * 64, dst + block * 65, dst + block * 65 + 1);
            for (int d = 0; d < 64; ++d) {
                unsigned short bits = csa_bf16_bits(record[448 + d]);
                dst[455 + 2 * d] = (unsigned char)bits;
                dst[455 + 2 * d + 1] = (unsigned char)(bits >> 8);
            }
        } else {
            float* dst = (float*)cache + (b * cache_records + out) * dim;
            for (int d = 0; d < dim; ++d) dst[d] = record[d];
        }
        for (int reset_slot = 0; reset_slot < 128; ++reset_slot)
            for (int d = 0; d < dim; ++d) {
                carry[b * carry_stride + (2 * reset_slot) * dim + d] = 0.0f;
                carry[b * carry_stride + (2 * reset_slot + 1) * dim + d] = __int_as_float(0xff800000);
            }
    }
}
"#;
const INDEX_COMPRESSION_SOURCE: &str = r#"
__device__ __forceinline__ unsigned short csa_index_bf16_bits(float x) {
    unsigned int bits = __float_as_uint(x);
    return (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float csa_index_bf16(float x) {
    return __uint_as_float((unsigned int)csa_index_bf16_bits(x) << 16);
}
extern "C" __global__ void csa_ratio4_index_compress(
    const float* kv, const float* gate, const float* ape, const float* norm,
    const unsigned char* past_key, const float* past_carry,
    unsigned char* key, float* carry,
    int batch, int sequence, int dim, int rope_dim, int past_records, int key_records,
    long long start)
{
    const int b = blockIdx.x;
    if (b >= batch || threadIdx.x != 0) return;
    const int source_width = 2 * dim;
    const int carry_stride = 8 * 2 * source_width;
    for (int i = 0; i < carry_stride; ++i) carry[b * carry_stride + i] = past_carry[b * carry_stride + i];
    for (int i = 0; i < past_records * 68; ++i)
        key[(b * key_records) * 68 + i] = past_key[(b * past_records) * 68 + i];
    const float NEG = __int_as_float(0xff800000);
    if (start == 0) {
        for (int slot = 0; slot < 8; ++slot)
            for (int state = 0; state < 2; ++state)
                for (int d = 0; d < source_width; ++d)
                    carry[b * carry_stride + ((slot * 2 + state) * source_width + d)] =
                        state == 0 ? 0.0f : NEG;
    }
    int emitted = 0;
    for (int s = 0; s < sequence; ++s) {
        const long long pos = start + s;
        const int phase = (int)(pos & 3);
        const int slot = 4 + phase;
        for (int d = 0; d < source_width; ++d) {
            carry[b * carry_stride + ((slot * 2) * source_width + d)] =
                kv[((b * sequence + s) * source_width) + d];
            carry[b * carry_stride + ((slot * 2 + 1) * source_width + d)] =
                __fadd_rn(gate[((b * sequence + s) * source_width) + d], ape[phase * source_width + d]);
        }
        if (((pos + 1) & 3) != 0) continue;

        float record[128];
        for (int d = 0; d < dim; ++d) {
            float maximum = NEG;
            for (int candidate = 0; candidate < 8; ++candidate) {
                const int source_dim = candidate < 4 ? d : dim + d;
                maximum = fmaxf(maximum,
                    carry[b * carry_stride + ((candidate * 2 + 1) * source_width + source_dim)]);
            }
            float numerator = 0.0f, denominator = 0.0f;
            for (int candidate = 0; candidate < 8; ++candidate) {
                const int source_dim = candidate < 4 ? d : dim + d;
                const float score =
                    carry[b * carry_stride + ((candidate * 2 + 1) * source_width + source_dim)];
                if (score == NEG) continue;
                const float weight = (float)exp((double)__fsub_rn(score, maximum));
                numerator = __fadd_rn(numerator, __fmul_rn(weight,
                    carry[b * carry_stride + ((candidate * 2) * source_width + source_dim)]));
                denominator = __fadd_rn(denominator, weight);
            }
            record[d] = csa_index_bf16(__fdiv_rn(numerator, denominator));
        }
        float square_sum = 0.0f;
        for (int d = 0; d < dim; ++d)
            square_sum = __fadd_rn(square_sum, __fmul_rn(record[d], record[d]));
        const float inverse_rms =
            __frcp_rn(__fsqrt_rn(__fadd_rn(__fdiv_rn(square_sum, (float)dim), 1.0e-6f)));
        for (int d = 0; d < dim; ++d)
            record[d] = csa_index_bf16(__fmul_rn(__fmul_rn(record[d], inverse_rms), norm[d]));
        for (int pair = 0; pair < rope_dim / 2; ++pair) {
            const float ramp = fminf(1.0f, fmaxf(0.0f, ((float)pair - 15.0f) / 10.0f));
            const float base = powf(160000.0f, -((float)(2 * pair)) / (float)rope_dim);
            const float frequency =
                __fadd_rn(__fmul_rn(base, 1.0f - ramp), __fmul_rn(base / 16.0f, ramp));
            float sn, cs; sincosf((float)(pos - 3) * frequency, &sn, &cs);
            const int d = dim - rope_dim + 2 * pair;
            const float re = record[d], im = record[d + 1];
            record[d] = csa_index_bf16(__fsub_rn(__fmul_rn(re, cs), __fmul_rn(im, sn)));
            record[d + 1] = csa_index_bf16(__fadd_rn(__fmul_rn(re, sn), __fmul_rn(im, cs)));
        }
        for (int span = 1; span < dim; span *= 2)
            for (int base = 0; base < dim; base += 2 * span)
                for (int offset = 0; offset < span; ++offset) {
                    const float left = record[base + offset];
                    const float right = record[base + offset + span];
                    record[base + offset] = __fadd_rn(left, right);
                    record[base + offset + span] = __fsub_rn(left, right);
                }
        const float hadamard_scale = __frcp_rn(__fsqrt_rn((float)dim));
        for (int d = 0; d < dim; ++d) record[d] = csa_index_bf16(__fmul_rn(record[d], hadamard_scale));
        unsigned char* dst = key + (b * key_records + past_records + emitted++) * 68;
        for (int block = 0; block < 4; ++block)
            quantize_fp4_e2m1_block(record + 32 * block, dst + 17 * block, dst + 17 * block + 1);

        for (int previous = 0; previous < 4; ++previous)
            for (int state = 0; state < 2; ++state)
                for (int d = 0; d < source_width; ++d) {
                    const int from = ((4 + previous) * 2 + state) * source_width + d;
                    const int to = (previous * 2 + state) * source_width + d;
                    carry[b * carry_stride + to] = carry[b * carry_stride + from];
                    carry[b * carry_stride + from] = state == 0 ? 0.0f : NEG;
                }
    }
}
"#;
/// B4 stages 3–5 for ratio-4: index-query finalize (RoPE → Hadamard `1/√ID` →
/// FP4 E2M1 round-trip), `dot → relu → weighted-head-sum` scoring with causal +
/// valid-length masking, and a deterministic top-k selection reproducing the CPU
/// oracle's `select_ratio4_topk` **bit-for-bit**:
///
/// * one CUDA block owns one `(batch, query)` row; a single thread runs the
///   order-dependent reductions so the f32 accumulation order equals the CPU
///   oracle (`dot` sums ascending in the dimension index; the head sum accrues
///   ascending in the head index). `__fadd_rn`/`__fmul_rn` keep every
///   multiply-add un-fused.
/// * the index-query finalize reuses the frozen index-record math (the same
///   `csa_index_bf16`, compressed-RoPE, Hadamard `1/√ID`, and FP4 E2M1
///   round-trip primitives proven bit-identical in B3), just keyed on the query
///   position and without the RMSNorm/pooling stages.
/// * the candidate keys are dequantized on the fly from the freshly written
///   `present_index_key` FP4 buffer — identical values, in the identical
///   ascending-dimension order, to the oracle's `all_index_logical`.
/// * selection reproduces PyTorch `topk(..., largest=True, sorted=True)` exactly
///   as the oracle freezes it: a strict total order of `(-score, +record)` using
///   Rust's `f32::total_cmp` on the bit-identical scores, with `-1` padding for
///   the unfilled tail when fewer than `topk_width` causal records exist. The
///   oracle rejects exact `==` score ties, so the ascending-record tiebreak is a
///   determinism guard that never triggers for an accepted input.
const INDEX_SELECT_SOURCE: &str = r#"
__device__ __forceinline__ unsigned short csa_sel_bf16_bits(float x) {
    unsigned int bits = __float_as_uint(x);
    return (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float csa_sel_bf16(float x) {
    return __uint_as_float((unsigned int)csa_sel_bf16_bits(x) << 16);
}
// Rust `f32::total_cmp`: a strict total order over the raw bit patterns.
__device__ __forceinline__ int csa_total_cmp(float a, float b) {
    int ia = __float_as_int(a);
    int ib = __float_as_int(b);
    ia ^= (int)(((unsigned int)(ia >> 31)) >> 1);
    ib ^= (int)(((unsigned int)(ib >> 31)) >> 1);
    return ia < ib ? -1 : (ia > ib ? 1 : 0);
}
// Record `a` ranks strictly before record `b` under `sorted=True, largest=True`:
// higher score first (descending `total_cmp`), then lower original record index.
__device__ __forceinline__ bool csa_rank_before(float sa, int ia, float sb, int ib) {
    const int c = csa_total_cmp(sa, sb);
    if (c > 0) return true;
    if (c < 0) return false;
    return ia < ib;
}
extern "C" __global__ void csa_ratio4_index_select(
    const float* index_query,        // [batch, sequence, index_heads, index_dim]
    const float* index_weight,       // [batch, sequence, index_heads]
    const unsigned char* index_key,  // [batch, records, 68] packed FP4 E2M1
    float* transformed,              // [batch * sequence * index_heads * index_dim] scratch
    float* scores,                   // [batch, sequence, records] scratch
    int* selected,                   // [batch, sequence, topk_width] scratch
    int batch, int sequence, int index_heads, int index_dim, int rope_dim,
    int records, long long start, int topk_width)
{
    const int bs = blockIdx.x;
    if (bs >= batch * sequence || threadIdx.x != 0) return;
    const int s = bs % sequence;
    const int b = bs / sequence;

    const long long position = start + (long long)s;
    long long valid_ll = (position + 1) / 4;
    int limit = records;
    if (valid_ll < (long long)limit) limit = (int)valid_ll;
    if (limit < 0) limit = 0;

    // weight_scale = (1 / sqrt(index_dim)) / sqrt(index_heads), matching the
    // oracle's two-step division exactly.
    float weight_scale = __fdiv_rn(1.0f, __fsqrt_rn((float)index_dim));
    weight_scale = __fdiv_rn(weight_scale, __fsqrt_rn((float)index_heads));

    // Stage 3: finalize every index-query head for this (batch, query) row.
    const int query_stride = index_heads * index_dim;
    for (int head = 0; head < index_heads; ++head) {
        float* q = transformed + ((long long)bs * query_stride) + (long long)head * index_dim;
        const float* src =
            index_query + ((((long long)(b * sequence + s) * index_heads + head) * index_dim));
        for (int d = 0; d < index_dim; ++d) q[d] = csa_sel_bf16(src[d]);
        // Compressed RoPE on the last `rope_dim` components, keyed on `position`.
        for (int pair = 0; pair < rope_dim / 2; ++pair) {
            const float ramp = fminf(1.0f, fmaxf(0.0f, ((float)pair - 15.0f) / 10.0f));
            const float base = powf(160000.0f, -((float)(2 * pair)) / (float)rope_dim);
            const float frequency =
                __fadd_rn(__fmul_rn(base, 1.0f - ramp), __fmul_rn(base / 16.0f, ramp));
            float sn, cs; sincosf((float)position * frequency, &sn, &cs);
            const int d = index_dim - rope_dim + 2 * pair;
            const float re = q[d], im = q[d + 1];
            q[d]     = csa_sel_bf16(__fsub_rn(__fmul_rn(re, cs), __fmul_rn(im, sn)));
            q[d + 1] = csa_sel_bf16(__fadd_rn(__fmul_rn(re, sn), __fmul_rn(im, cs)));
        }
        // Hadamard transform (plain f32 add/sub), then bf16 `1/√ID` scaling.
        for (int span = 1; span < index_dim; span *= 2)
            for (int base2 = 0; base2 < index_dim; base2 += 2 * span)
                for (int offset = 0; offset < span; ++offset) {
                    const float left = q[base2 + offset];
                    const float right = q[base2 + offset + span];
                    q[base2 + offset] = __fadd_rn(left, right);
                    q[base2 + offset + span] = __fsub_rn(left, right);
                }
        const float hadamard_scale = __frcp_rn(__fsqrt_rn((float)index_dim));
        for (int d = 0; d < index_dim; ++d) q[d] = csa_sel_bf16(__fmul_rn(q[d], hadamard_scale));
        // FP4 E2M1 round-trip: quantize each block, then dequantize back in place.
        for (int block = 0; block < index_dim / 32; ++block) {
            unsigned char scale_byte; unsigned char packed[16];
            quantize_fp4_e2m1_block(q + 32 * block, &scale_byte, packed);
            const float block_scale = e8m0_scale(scale_byte);
            for (int pair = 0; pair < 16; ++pair) {
                const unsigned char byte = packed[pair];
                q[32 * block + 2 * pair]     = decode_e2m1(byte) * block_scale;
                q[32 * block + 2 * pair + 1] = decode_e2m1(byte >> 4) * block_scale;
            }
        }
    }

    // Stage 4: score each causal candidate record.
    const int weight_row = (b * sequence + s) * index_heads;
    float* row_scores = scores + ((long long)(b * sequence + s) * records);
    for (int record = 0; record < limit; ++record) {
        const unsigned char* key = index_key + ((long long)b * records + record) * 68;
        float score = 0.0f;
        for (int head = 0; head < index_heads; ++head) {
            const float* q = transformed + ((long long)bs * query_stride) + (long long)head * index_dim;
            float acc = 0.0f;
            for (int block = 0; block < index_dim / 32; ++block) {
                const float block_scale = e8m0_scale(key[17 * block]);
                for (int pair = 0; pair < 16; ++pair) {
                    const unsigned char byte = key[17 * block + 1 + pair];
                    const int d = 32 * block + 2 * pair;
                    acc = __fadd_rn(acc, __fmul_rn(q[d], decode_e2m1(byte) * block_scale));
                    acc = __fadd_rn(acc, __fmul_rn(q[d + 1], decode_e2m1(byte >> 4) * block_scale));
                }
            }
            const float relu = fmaxf(acc, 0.0f);
            const float weight = index_weight[weight_row + head];
            score = __fadd_rn(score, __fmul_rn(__fmul_rn(relu, weight), weight_scale));
        }
        row_scores[record] = score;
    }

    // Stage 5: deterministic top-k selection over the frozen total order.
    int* row_selected = selected + ((long long)(b * sequence + s) * topk_width);
    for (int slot = 0; slot < topk_width; ++slot) row_selected[slot] = -1;
    int prev = -1; float prev_score = 0.0f;
    for (int slot = 0; slot < topk_width; ++slot) {
        int best = -1; float best_score = 0.0f;
        for (int record = 0; record < limit; ++record) {
            const float sr = row_scores[record];
            // Skip any record that ranks at or before the previously chosen one.
            if (prev != -1 && !csa_rank_before(prev_score, prev, sr, record)) continue;
            if (best == -1 || csa_rank_before(sr, record, best_score, best)) {
                best = record; best_score = sr;
            }
        }
        if (best == -1) break;  // fewer causal records than topk_width: leave -1.
        row_selected[slot] = best;
        prev = best; prev_score = best_score;
    }
}
"#;
const ATTENTION_SOURCE: &str = r#"
extern "C" __global__ void csa_ratio128_sink_attention(
    const float* query,        // [batch, sequence, heads, dim]
    const float* current_kv,   // [batch, current_kv_len, dim]
    const float* compressed,   // [batch, compressed_records, dim]
    const float* sink,         // [heads]
    float* output,             // [batch, sequence, heads, dim]
    float* scores,             // [rows * candidate_count] scratch
    int batch, int sequence, int heads, int dim,
    int current_kv_len,
    long long current_kv_base,
    long long query_start,
    int compressed_records,
    int dense_candidates,
    int candidate_count,
    float scale)
{
    const int row = blockIdx.x;
    const int rows = batch * sequence * heads;
    if (row >= rows) return;
    const int h = row % heads;
    int tmp = row / heads;
    const int s = tmp % sequence;
    const int b = tmp / sequence;

    const float NEG = __int_as_float(0xff800000);
    const long long position = query_start + (long long)s;
    long long window = position + 1 - 128;
    if (window < 0) window = 0;
    const long long dense_start = current_kv_base > window ? current_kv_base : window;
    const long long valid_compressed = (position + 1) / 128;
    int comp_limit = compressed_records < (int)valid_compressed
        ? compressed_records : (int)valid_compressed;

    float* row_scores = scores + (long long)row * candidate_count;
    const long long q_base = ((long long)(b * sequence + s) * heads + h) * dim;

    __shared__ float s_max;
    __shared__ float s_denom;
    __shared__ int s_valid;

    if (threadIdx.x == 0) {
        for (int c = 0; c < candidate_count; ++c) row_scores[c] = NEG;
        float maximum = NEG;
        // Dense window candidates, ascending absolute position.
        for (int c = 0; c < dense_candidates; ++c) {
            const long long absolute = dense_start + (long long)c;
            if (absolute > position) continue;
            const long long relative = absolute - current_kv_base;
            if (relative >= (long long)current_kv_len) continue;
            const float* kv = current_kv + ((long long)b * current_kv_len + relative) * dim;
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d)
                acc = __fadd_rn(acc, __fmul_rn(query[q_base + d], kv[d]));
            const float score = __fmul_rn(acc, scale);
            row_scores[c] = score;
            maximum = fmaxf(maximum, score);
        }
        // Completed compressed records, ascending.
        for (int rec = 0; rec < comp_limit; ++rec) {
            const float* kv = compressed + ((long long)b * compressed_records + rec) * dim;
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d)
                acc = __fadd_rn(acc, __fmul_rn(query[q_base + d], kv[d]));
            const float score = __fmul_rn(acc, scale);
            row_scores[dense_candidates + rec] = score;
            maximum = fmaxf(maximum, score);
        }
        if (maximum == NEG) {
            s_valid = 0;
        } else {
            float denom = 0.0f;
            for (int c = 0; c < candidate_count; ++c) {
                if (row_scores[c] != NEG)
                    denom = __fadd_rn(denom, (float)exp((double)(row_scores[c] - maximum)));
            }
            // Sink is a logit-only denominator mass, added after the max.
            denom = __fadd_rn(denom, (float)exp((double)(sink[h] - maximum)));
            s_denom = denom;
            s_valid = 1;
        }
        s_max = maximum;
    }
    __syncthreads();

    if (s_valid == 0) {
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            output[q_base + d] = 0.0f;
        return;
    }

    const float maximum = s_max;
    const float denom = s_denom;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float result = 0.0f;
        for (int c = 0; c < candidate_count; ++c) {
            const float score = row_scores[c];
            if (score == NEG) continue;
            const float prob = (float)exp((double)(score - maximum)) / denom;
            float val;
            if (c < dense_candidates) {
                const long long absolute = dense_start + (long long)c;
                const long long relative = absolute - current_kv_base;
                val = current_kv[(((long long)b * current_kv_len + relative) * dim) + d];
            } else {
                const int rec = c - dense_candidates;
                val = compressed[(((long long)b * compressed_records + rec) * dim) + d];
            }
            result = __fadd_rn(result, __fmul_rn(prob, val));
        }
        output[q_base + d] = result;
    }
}
"#;
const RATIO4_ATTENTION_SOURCE: &str = r#"
extern "C" __global__ void csa_ratio4_sink_attention(
    const float* query, const float* current_kv, const unsigned char* compressed,
    const int* selected, const float* sink, const float* bias, float* output, float* scores,
    int batch, int sequence, int heads, int dim, int current_kv_len,
    long long current_kv_base, long long query_start, int compressed_records,
    int index_heads, int topk_width, int dense_candidates, int candidate_count,
    int bias_present, int bias_b, int bias_h, int bias_s, int bias_k, float scale)
{
    const int row = blockIdx.x;
    const int rows = batch * sequence * heads;
    if (row >= rows) return;
    const int h = row % heads;
    const int bs = row / heads;
    const int s = bs % sequence;
    const int b = bs / sequence;
    const float NEG = __int_as_float(0xff800000);
    const long long position = query_start + (long long)s;
    long long window = position + 1 - 128;
    if (window < 0) window = 0;
    const long long dense_start = current_kv_base > window ? current_kv_base : window;
    float* row_scores = scores + (long long)row * candidate_count;
    const long long q_base = ((long long)(b * sequence + s) * heads + h) * dim;
    const int selected_base = ((b * index_heads) * sequence + s) * topk_width;
    __shared__ float s_max;
    __shared__ float s_denom;
    __shared__ int s_valid;
    if (threadIdx.x == 0) {
        for (int c = 0; c < candidate_count; ++c) row_scores[c] = NEG;
        float maximum = NEG;
        for (int c = 0; c < dense_candidates; ++c) {
            const long long absolute = dense_start + (long long)c;
            if (absolute > position) continue;
            const long long relative = absolute - current_kv_base;
            if (relative < 0 || relative >= (long long)current_kv_len) continue;
            const float* kv = current_kv + ((long long)b * current_kv_len + relative) * dim;
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d)
                acc = __fadd_rn(acc, __fmul_rn(query[q_base + d], kv[d]));
            float score = __fmul_rn(acc, scale);
            if (bias_present) {
                const int bb = bias_b == 1 ? 0 : b;
                const int bh = bias_h == 1 ? 0 : h;
                const int bq = bias_s == 1 ? 0 : s;
                const int bk = bias_k == 1 ? 0 : c;
                score = __fadd_rn(score, bias[(((bb * bias_h + bh) * bias_s + bq) * bias_k + bk)]);
            }
            row_scores[c] = score; maximum = fmaxf(maximum, score);
        }
        for (int slot = 0; slot < topk_width; ++slot) {
            const int record = selected[selected_base + slot];
            if (record < 0 || record >= compressed_records) continue;
            const unsigned char* packed = compressed + ((long long)b * compressed_records + record) * 583;
            float acc = 0.0f;
            for (int block = 0; block < 7; ++block) {
                const float block_scale = e8m0_scale(packed[65 * block]);
                for (int d = 0; d < 64; ++d)
                    acc = __fadd_rn(acc, __fmul_rn(query[q_base + 64 * block + d],
                        decode_e4m3fn(packed[65 * block + 1 + d]) * block_scale));
            }
            for (int d = 448; d < dim; ++d) {
                const int tail = d - 448;
                const unsigned short bits = (unsigned short)packed[455 + 2 * tail]
                    | ((unsigned short)packed[455 + 2 * tail + 1] << 8);
                acc = __fadd_rn(acc, __fmul_rn(query[q_base + d],
                    __uint_as_float((unsigned int)bits << 16)));
            }
            const int c = dense_candidates + slot;
            float score = __fmul_rn(acc, scale);
            if (bias_present) {
                const int bb = bias_b == 1 ? 0 : b;
                const int bh = bias_h == 1 ? 0 : h;
                const int bq = bias_s == 1 ? 0 : s;
                const int bk = bias_k == 1 ? 0 : c;
                score = __fadd_rn(score, bias[(((bb * bias_h + bh) * bias_s + bq) * bias_k + bk)]);
            }
            row_scores[c] = score; maximum = fmaxf(maximum, score);
        }
        if (maximum == NEG) s_valid = 0;
        else {
            float denom = 0.0f;
            for (int c = 0; c < candidate_count; ++c)
                if (row_scores[c] != NEG)
                    denom = __fadd_rn(denom, (float)exp((double)(row_scores[c] - maximum)));
            denom = __fadd_rn(denom, (float)exp((double)(sink[h] - maximum)));
            s_max = maximum; s_denom = denom; s_valid = 1;
        }
    }
    __syncthreads();
    if (!s_valid) {
        for (int d = threadIdx.x; d < dim; d += blockDim.x) output[q_base + d] = 0.0f;
        return;
    }
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float result = 0.0f;
        for (int c = 0; c < candidate_count; ++c) {
            const float score = row_scores[c];
            if (score == NEG) continue;
            const float probability = (float)exp((double)(score - s_max)) / s_denom;
            float value;
            if (c < dense_candidates) {
                const long long relative = dense_start + (long long)c - current_kv_base;
                value = current_kv[((long long)b * current_kv_len + relative) * dim + d];
            } else {
                const int record = selected[selected_base + c - dense_candidates];
                const unsigned char* packed = compressed + ((long long)b * compressed_records + record) * 583;
                if (d < 448) {
                    const int block = d / 64, in_block = d % 64;
                    value = decode_e4m3fn(packed[65 * block + 1 + in_block])
                        * e8m0_scale(packed[65 * block]);
                } else {
                    const int tail = d - 448;
                    const unsigned short bits = (unsigned short)packed[455 + 2 * tail]
                        | ((unsigned short)packed[455 + 2 * tail + 1] << 8);
                    value = __uint_as_float((unsigned int)bits << 16);
                }
            }
            result = __fadd_rn(result, __fmul_rn(probability, value));
        }
        output[q_base + d] = result;
    }
}
"#;

/// Factory for the host-staged CUDA CSA kernel. It builds the CPU CSA kernel
/// from the same node (reusing the CPU oracle's attribute validation and compute
/// core) and wraps it so execution stages tensors through the host.
pub struct CompressedSparseAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for CompressedSparseAttentionFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        // Delegate construction to the CPU factory: it validates the full frozen
        // v1 attribute set (ratio, cache_format, sink_mode, index dims, arity,
        // required input names) and produces the stateful oracle kernel whose
        // compute we reuse verbatim.
        let inner = CpuCsaFactory.create(node, input_shapes)?;
        let ratio = usize::try_from(
            node.attr("compression_ratio")
                .and_then(|attribute| attribute.as_int())
                .expect("CPU factory accepted compression_ratio"),
        )
        .expect("CPU factory accepted positive compression_ratio");
        // This is runner initialization, never capability claiming: reserve every
        // fixed-address stream now so an OOM fails before any execution.
        let layout = CsaBufferLayout::from_runner(node, input_shapes, ratio)?;
        let device_state = CsaDeviceBufferManager::reserve(self.runtime.clone(), layout)?;

        // B1: flip stage-6 (candidate read) + stage-7 (sparse sink-softmax
        // attention) to Device for ratio-128 with the f32 record cache, where
        // the host-staged compression already produces the dequantized candidate
        // records (`present_compressed_kv` == the f32 logical records). FP8
        // ratio-128 FP8 records remain host-staged this slice; ratio-4 B5
        // dequantizes packed candidate records directly on device, including
        // optional attention bias. Compression/writeback and B4's index readback
        // keep `cuda_graph_compatible()` false.
        let cache_format = node
            .attr("cache_format")
            .and_then(|attribute| attribute.as_str())
            .unwrap_or("f32")
            .to_string();
        let has_attention_bias = node.inputs.get(19).is_some_and(Option::is_some);
        let device_compression = ratio == 128 && !has_attention_bias;
        let device_index_compression = ratio == 4;
        let device_index_scoring = ratio == 4;
        let device_attention = ratio == 128 && cache_format == "f32" && !has_attention_bias;
        // B5 fused ratio-4 selection→attention runs on device, but ONLY when the
        // node emits `selected_indices` (output 5). For a 5-output ratio-4 node
        // that omits the optional `selected_indices`, there is no device-selected
        // record stream to dereference, so `Y` must stay on the host oracle. This
        // flag keys strictly on ratio; the output-count gate lives at dispatch.
        let device_ratio4_attention = ratio == 4;
        let configured_scale = node
            .attr("scale")
            .and_then(|attribute| attribute.as_float())
            .unwrap_or(0.0);
        let mut dispatch = CsaStageDispatch::default();
        if device_compression {
            dispatch.set(CsaPipelineStage::CompressionUpdate, CsaStageMode::Device);
        }
        if device_index_compression {
            dispatch.set(CsaPipelineStage::IndexKeyUpdate, CsaStageMode::Device);
        }
        if device_index_scoring {
            dispatch.set(CsaPipelineStage::IndexQueryFinalize, CsaStageMode::Device);
            dispatch.set(CsaPipelineStage::IndexScoring, CsaStageMode::Device);
            dispatch.set(CsaPipelineStage::Selection, CsaStageMode::Device);
        }
        if device_attention || device_ratio4_attention {
            dispatch.set(CsaPipelineStage::CandidateAssembly, CsaStageMode::Device);
            dispatch.set(
                CsaPipelineStage::SparseSinkSoftmaxAttention,
                CsaStageMode::Device,
            );
        }

        Ok(Box::new(CompressedSparseAttentionKernel {
            runtime: self.runtime.clone(),
            inner,
            device_state,
            dispatch,
            device_compression,
            device_index_compression,
            device_index_scoring,
            device_attention,
            device_ratio4_attention,
            qk_rope_head_dim: usize::try_from(
                node.attr("qk_rope_head_dim")
                    .and_then(|attribute| attribute.as_int())
                    .expect("CPU factory accepted qk_rope_head_dim"),
            )
            .expect("CPU factory accepted positive qk_rope_head_dim"),
            index_num_heads: usize::try_from(
                node.attr("index_num_heads")
                    .and_then(|attribute| attribute.as_int())
                    .unwrap_or(0),
            )
            .unwrap_or(0),
            index_head_dim: usize::try_from(
                node.attr("index_head_dim")
                    .and_then(|attribute| attribute.as_int())
                    .unwrap_or(0),
            )
            .unwrap_or(0),
            configured_scale,
            golden_capture: CsaGoldenCapture::from_environment(),
        }))
    }
}

/// Host-staged CUDA CSA kernel: wraps the CPU oracle kernel and moves data
/// device↔host around each `execute`.
struct CompressedSparseAttentionKernel {
    runtime: Arc<CudaRuntime>,
    inner: Box<dyn Kernel>,
    // Kept alive for stable device addresses; B0 still uses graph-threaded state.
    device_state: CsaDeviceBufferManager,
    dispatch: CsaStageDispatch,
    /// B2: ratio-128 stage-1 is a device kernel for both f32 and hybrid-FP8
    /// caches.  Ratio-4 remains entirely host-staged.
    device_compression: bool,
    /// B3: ratio-4 stage-2 index-key compression is independently finalized on
    /// the device; scoring and attention deliberately remain host-staged.
    device_index_compression: bool,
    /// B4/B5: ratio-4 selection writes device `selected_indices`; fused
    /// candidate assembly and sparse attention consume it directly.
    device_index_scoring: bool,
    /// B1: ratio-128 f32-cache path runs stage-7 attention on device.
    device_attention: bool,
    /// B5: ratio-4 fused selection→attention runs stage-6/7 on device, but only
    /// when the node emits `selected_indices` (6 outputs). A 5-output ratio-4
    /// node keeps `Y` from the host oracle — the device ratio-128 attention
    /// kernel must never run for a ratio-4 node.
    device_ratio4_attention: bool,
    qk_rope_head_dim: usize,
    /// Ratio-4 index-stream geometry (`index_num_heads`, `index_head_dim`);
    /// zero for ratio-128 where the index stream is absent.
    index_num_heads: usize,
    index_head_dim: usize,
    /// Raw `scale` attribute (0.0 → `1/sqrt(dim)`), resolved at launch.
    configured_scale: f32,
    golden_capture: CsaGoldenCapture,
}

impl std::fmt::Debug for CompressedSparseAttentionKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompressedSparseAttentionKernel").finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsaStageMode {
    Host,
    Device,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsaPipelineStage {
    CompressionUpdate,
    IndexKeyUpdate,
    IndexQueryFinalize,
    IndexScoring,
    Selection,
    CandidateAssembly,
    SparseSinkSoftmaxAttention,
    Writeback,
}

impl CsaPipelineStage {
    const ALL: [Self; 8] = [
        Self::CompressionUpdate,
        Self::IndexKeyUpdate,
        Self::IndexQueryFinalize,
        Self::IndexScoring,
        Self::Selection,
        Self::CandidateAssembly,
        Self::SparseSinkSoftmaxAttention,
        Self::Writeback,
    ];
}

#[derive(Clone, Debug)]
pub struct CsaStageDispatch {
    modes: [CsaStageMode; 8],
}
impl Default for CsaStageDispatch {
    fn default() -> Self {
        Self {
            modes: [CsaStageMode::Host; 8],
        }
    }
}
impl CsaStageDispatch {
    pub fn mode(&self, stage: CsaPipelineStage) -> CsaStageMode {
        self.modes[stage as usize]
    }

    /// B0 still delegates Device modes to the oracle; later phases replace only
    /// the selected stage's branch.
    pub fn set(&mut self, stage: CsaPipelineStage, mode: CsaStageMode) {
        self.modes[stage as usize] = mode;
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct CsaGoldenBoundary {
    stage: CsaPipelineStage,
    mode: CsaStageMode,
    payload: Vec<u8>,
}
#[derive(Debug)]
struct CsaGoldenCapture {
    enabled: bool,
    boundaries: Mutex<Vec<CsaGoldenBoundary>>,
}
impl CsaGoldenCapture {
    fn from_environment() -> Self {
        Self {
            enabled: std::env::var_os("NXRT_CSA_GOLDEN_CAPTURE").is_some(),
            boundaries: Mutex::new(Vec::new()),
        }
    }
    fn record(&self, stage: CsaPipelineStage, mode: CsaStageMode, inputs: &[TensorView]) {
        if self.enabled {
            let mut payload = Vec::new();
            for input in inputs.iter().filter(|input| !input.is_absent()) {
                // SAFETY: B0 calls this only for the live host-staged views, and
                // copies exactly their contiguous byte extent for a future diff.
                unsafe {
                    payload.extend_from_slice(std::slice::from_raw_parts(
                        input.data_ptr::<u8>(),
                        input.byte_size(),
                    ));
                }
            }
            self.boundaries
                .lock()
                .expect("CSA golden capture mutex poisoned")
                .push(CsaGoldenBoundary {
                    stage,
                    mode,
                    payload,
                });
        }
    }
}

impl CompressedSparseAttentionKernel {
    fn run_host_staged_pipeline(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
    ) -> Result<()> {
        let _stable_capacity = self.device_state.layout.max_seq_len;
        for stage in CsaPipelineStage::ALL {
            self.golden_capture
                .record(stage, self.dispatch.mode(stage), inputs);
        }
        self.inner.execute(inputs, outputs)
    }

    /// B1 device stage-6/7 for ratio-128 (f32 cache). Launches the CUDA
    /// sink-softmax attention kernel over the device query / `current_kv` /
    /// candidate-record buffers, writing `Y` (output 0) directly and matching
    /// the CPU oracle's `ratio128_attention` numerics bit-for-bit.
    fn run_device_attention(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        staged: &[Vec<u8>],
    ) -> Result<()> {
        // Reproduce the oracle's derived attention geometry from the staged
        // inputs and inferred output shapes (see the CPU
        // `execute_stateful_ratio128` / `ratio128_attention`).
        let query_shape = inputs[0].shape;
        let batch = query_shape[0];
        let sequence = query_shape[1];
        let heads = query_shape[2];
        let dim = query_shape[3];
        let current_kv_len = inputs[1].shape[1];
        let compressed_records = outputs[1].shape[1];

        let total_bytes = &staged[9];
        if total_bytes.len() != 8 {
            return Err(not_implemented(format!(
                "{OP}: device attention expects a scalar total_sequence_length"
            )));
        }

        let total = i64::from_ne_bytes(total_bytes[..8].try_into().expect("8 bytes")) as usize;
        let start = total.checked_sub(sequence).ok_or_else(|| {
            not_implemented(format!("{OP}: total < sequence in device attention"))
        })?;
        let current_kv_base = total.checked_sub(current_kv_len).ok_or_else(|| {
            not_implemented(format!(
                "{OP}: current_kv longer than total in device attention"
            ))
        })?;
        let dense_candidates = if start == 0 {
            current_kv_len.min(128)
        } else {
            128
        };
        let candidate_count = dense_candidates + compressed_records;
        let rows = batch * sequence * heads;
        if rows == 0 || candidate_count == 0 {
            return Ok(());
        }

        let scale = if self.configured_scale == 0.0 {
            1.0f32 / (dim as f32).sqrt()
        } else {
            self.configured_scale
        };

        let query_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let current_kv_ptr = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        // Compressed candidate records are the freshly uploaded f32
        // `present_compressed_kv` (output 1). When there are no completed
        // records the pointer is unused by the kernel.
        let compressed_ptr = cuptr(outputs[1].data_ptr_mut::<u8>() as *const c_void);
        let sink_ptr = cuptr(inputs[10].data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let score_bytes = rows
            .checked_mul(candidate_count)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| not_implemented(format!("{OP}: score scratch size overflow")))?;
        let scratch = self.runtime.alloc_raw(score_bytes.max(1))?;

        let launch = || -> Result<()> {
            let func =
                self.runtime
                    .nvrtc_function(ATTENTION_MODULE, ATTENTION_SOURCE, ATTENTION_ENTRY)?;
            let batch_i = batch as i32;
            let sequence_i = sequence as i32;
            let heads_i = heads as i32;
            let dim_i = dim as i32;
            let current_kv_len_i = current_kv_len as i32;
            let current_kv_base_i = current_kv_base as i64;
            let query_start_i = start as i64;
            let compressed_records_i = compressed_records as i32;
            let dense_candidates_i = dense_candidates as i32;
            let candidate_count_i = candidate_count as i32;
            let grid = u32::try_from(rows)
                .map_err(|_| not_implemented(format!("{OP}: attention row count exceeds u32")))?;

            let stream = self.runtime.stream();
            let mut builder = stream.launch_builder(&func);
            builder
                .arg(&query_ptr)
                .arg(&current_kv_ptr)
                .arg(&compressed_ptr)
                .arg(&sink_ptr)
                .arg(&output_ptr)
                .arg(&scratch)
                .arg(&batch_i)
                .arg(&sequence_i)
                .arg(&heads_i)
                .arg(&dim_i)
                .arg(&current_kv_len_i)
                .arg(&current_kv_base_i)
                .arg(&query_start_i)
                .arg(&compressed_records_i)
                .arg(&dense_candidates_i)
                .arg(&candidate_count_i)
                .arg(&scale);
            // SAFETY: argument types and order match `csa_ratio128_sink_attention`;
            // all pointers refer to live contiguous device allocations sized by
            // the shapes above, and the scratch covers `rows * candidate_count`.
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (ATTENTION_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| driver_err("launch csa_ratio128_sink_attention", error))?;
            self.runtime.synchronize()
        };

        let result = launch();
        // SAFETY: `scratch` came from this runtime's `alloc_raw` and is freed once.
        let free = unsafe { self.runtime.free_raw(scratch) };
        result.and(free)
    }

    /// B5 stages 6–7 for ratio-4. The selected record IDs remain in the
    /// device `selected_indices` output populated by B4; this kernel dereferences
    /// those IDs into the packed FP8/BF16 cache directly rather than assembling a
    /// dense selected-KV tensor.
    fn run_device_ratio4_attention(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        staged: &[Vec<u8>],
    ) -> Result<()> {
        let [batch, sequence, heads, dim] = inputs[0].shape.try_into().map_err(|_| {
            not_implemented(format!(
                "{OP}: ratio-4 device attention requires rank-4 query"
            ))
        })?;
        let current_kv_len = inputs[1].shape[1];
        let compressed_records = outputs[1].shape[1];
        let topk_width = outputs[5].shape[3];
        let total =
            i64::from_ne_bytes(staged[9].as_slice().try_into().map_err(|_| {
                not_implemented(format!("{OP}: expected scalar total_sequence_length"))
            })?) as usize;
        let start = total.checked_sub(sequence).ok_or_else(|| {
            not_implemented(format!("{OP}: total < sequence in ratio-4 attention"))
        })?;
        let current_kv_base = total.checked_sub(current_kv_len).ok_or_else(|| {
            not_implemented(format!(
                "{OP}: current_kv longer than total in ratio-4 attention"
            ))
        })?;
        let dense_candidates = if start == 0 {
            current_kv_len.min(128)
        } else {
            128
        };
        let candidate_count = dense_candidates
            .checked_add(topk_width)
            .ok_or_else(|| not_implemented(format!("{OP}: ratio-4 candidate count overflow")))?;
        let rows = batch
            .checked_mul(sequence)
            .and_then(|n| n.checked_mul(heads))
            .ok_or_else(|| not_implemented(format!("{OP}: ratio-4 attention rows overflow")))?;
        if rows == 0 || candidate_count == 0 {
            return Ok(());
        }
        let mut bias_shape = [1i32; 4];
        let bias_present = inputs.get(19).is_some_and(|input| !input.is_absent());
        if bias_present {
            let shape = inputs[19].shape;
            bias_shape[4 - shape.len()..].copy_from_slice(
                &shape
                    .iter()
                    .copied()
                    .map(|n| {
                        i32::try_from(n)
                            .map_err(|_| not_implemented("CSA bias dimension exceeds i32"))
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        let scale = if self.configured_scale == 0.0 {
            1.0f32 / (dim as f32).sqrt()
        } else {
            self.configured_scale
        };
        let scratch = self.runtime.alloc_raw(
            rows.checked_mul(candidate_count)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| not_implemented(format!("{OP}: ratio-4 score scratch overflow")))?
                .max(1),
        )?;
        let mut launch = || -> Result<()> {
            let source = format!("{}\n{}", block_quant::source(), RATIO4_ATTENTION_SOURCE);
            let func = self.runtime.nvrtc_function(
                RATIO4_ATTENTION_MODULE,
                &source,
                RATIO4_ATTENTION_ENTRY,
            )?;
            let query = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
            let current_kv = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
            let compressed = cuptr(outputs[1].data_ptr_mut::<u8>() as *const c_void);
            let selected = cuptr(outputs[5].data_ptr_mut::<u8>() as *const c_void);
            let sink = cuptr(inputs[10].data_ptr::<u8>() as *const c_void);
            let bias = if bias_present {
                cuptr(inputs[19].data_ptr::<u8>() as *const c_void)
            } else {
                cuptr(std::ptr::null::<c_void>())
            };
            let output = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
            let ints = [
                batch,
                sequence,
                heads,
                dim,
                current_kv_len,
                compressed_records,
                self.index_num_heads,
                topk_width,
                dense_candidates,
                candidate_count,
            ]
            .map(|n| {
                i32::try_from(n).map_err(|_| not_implemented("CSA ratio-4 geometry exceeds i32"))
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
            let current_kv_base = current_kv_base as i64;
            let start = start as i64;
            let bias_present_i = i32::from(bias_present);
            let grid = u32::try_from(rows)
                .map_err(|_| not_implemented(format!("{OP}: ratio-4 attention rows exceed u32")))?;
            let mut builder = self.runtime.stream().launch_builder(&func);
            builder
                .arg(&query)
                .arg(&current_kv)
                .arg(&compressed)
                .arg(&selected)
                .arg(&sink)
                .arg(&bias)
                .arg(&output)
                .arg(&scratch)
                .arg(&ints[0])
                .arg(&ints[1])
                .arg(&ints[2])
                .arg(&ints[3])
                .arg(&ints[4])
                .arg(&current_kv_base)
                .arg(&start)
                .arg(&ints[5])
                .arg(&ints[6])
                .arg(&ints[7])
                .arg(&ints[8])
                .arg(&ints[9])
                .arg(&bias_present_i)
                .arg(&bias_shape[0])
                .arg(&bias_shape[1])
                .arg(&bias_shape[2])
                .arg(&bias_shape[3])
                .arg(&scale);
            // SAFETY: every tensor is contiguous and the argument order matches
            // `csa_ratio4_sink_attention`; B4 has already written selected IDs.
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (ATTENTION_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| driver_err("launch csa_ratio4_sink_attention", error))?;
            self.runtime.synchronize()
        };
        let result = launch();
        // SAFETY: scratch was allocated by this runtime and has not escaped.
        let free = unsafe { self.runtime.free_raw(scratch) };
        result.and(free)
    }
    /// B2 stage-1.  The cache/carry graph outputs are the authoritative
    /// externally-visible state, so this kernel first copies exactly the past
    /// prefix, then mutates only the new carry slots and records.  This avoids a
    /// whole-cache rewrite and preserves the cache address chosen by the runner.
    fn run_device_compression(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        staged: &[Vec<u8>],
    ) -> Result<()> {
        let shape = inputs[0].shape;
        let (batch, sequence, dim) = (shape[0], shape[1], shape[3]);
        let past_records = inputs[6].shape[1];
        let cache_records = outputs[1].shape[1];
        if batch == 0 || sequence == 0 {
            return Ok(());
        }
        let start_total = staged[9].as_slice();
        if start_total.len() != 8 {
            return Err(not_implemented(format!(
                "{OP}: device compression expects scalar total_sequence_length"
            )));
        }
        let total = i64::from_ne_bytes(start_total.try_into().expect("8 bytes"));
        let start = total.checked_sub(sequence as i64).ok_or_else(|| {
            not_implemented(format!("{OP}: total < sequence in device compression"))
        })?;
        let cache_fp8 = i32::from(outputs[1].dtype == DataType::Uint8);
        let source = format!("{}\n{}", block_quant::source(), COMPRESSION_SOURCE);
        let func = self
            .runtime
            .nvrtc_function(COMPRESSION_MODULE, &source, COMPRESSION_ENTRY)?;
        let kv = cuptr(inputs[2].data_ptr::<u8>() as *const c_void);
        let gate = cuptr(inputs[3].data_ptr::<u8>() as *const c_void);
        let ape = cuptr(inputs[4].data_ptr::<u8>() as *const c_void);
        let norm = cuptr(inputs[5].data_ptr::<u8>() as *const c_void);
        let past_carry = cuptr(inputs[7].data_ptr::<u8>() as *const c_void);
        let past_cache = cuptr(inputs[6].data_ptr::<u8>() as *const c_void);
        let carry = cuptr(outputs[2].data_ptr_mut::<u8>() as *const c_void);
        let cache = cuptr(outputs[1].data_ptr_mut::<u8>() as *const c_void);
        let mut builder = self.runtime.stream().launch_builder(&func);
        let batch_i = i32::try_from(batch).map_err(|_| not_implemented("CSA batch exceeds i32"))?;
        let sequence_i =
            i32::try_from(sequence).map_err(|_| not_implemented("CSA sequence exceeds i32"))?;
        let dim_i = i32::try_from(dim).map_err(|_| not_implemented("CSA dimension exceeds i32"))?;
        let past_i =
            i32::try_from(past_records).map_err(|_| not_implemented("CSA records exceed i32"))?;
        let records_i =
            i32::try_from(cache_records).map_err(|_| not_implemented("CSA records exceed i32"))?;
        builder
            .arg(&kv)
            .arg(&gate)
            .arg(&ape)
            .arg(&norm)
            .arg(&past_carry)
            .arg(&past_cache)
            .arg(&carry)
            .arg(&cache)
            .arg(&batch_i)
            .arg(&sequence_i)
            .arg(&dim_i)
            .arg(&past_i)
            .arg(&records_i)
            .arg(&cache_fp8)
            .arg(&start);
        // SAFETY: all arguments are contiguous device buffers whose shape was
        // validated by the CPU factory; one serial thread owns each batch row.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (batch as u32, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch csa_ratio128_compress", error))?;
        self.runtime.synchronize()
    }

    fn run_device_index_compression(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        staged: &[Vec<u8>],
    ) -> Result<()> {
        let (batch, sequence) = (inputs[0].shape[0], inputs[0].shape[1]);
        if batch == 0 || sequence == 0 {
            return Ok(());
        }
        let total_bytes = &staged[9];
        if total_bytes.len() != 8 {
            return Err(not_implemented(format!(
                "{OP}: device index compression expects scalar total_sequence_length"
            )));
        }
        let total = i64::from_ne_bytes(total_bytes.as_slice().try_into().expect("8 bytes"));
        let start = total.checked_sub(sequence as i64).ok_or_else(|| {
            not_implemented(format!(
                "{OP}: total < sequence in device index compression"
            ))
        })?;
        let dim = inputs[16].shape[0];
        let past_records = inputs[17].shape[1];
        let key_records = outputs[3].shape[1];
        let source = format!("{}\n{}", block_quant::source(), INDEX_COMPRESSION_SOURCE);
        let func = self.runtime.nvrtc_function(
            INDEX_COMPRESSION_MODULE,
            &source,
            INDEX_COMPRESSION_ENTRY,
        )?;
        let kv = cuptr(inputs[13].data_ptr::<u8>() as *const c_void);
        let gate = cuptr(inputs[14].data_ptr::<u8>() as *const c_void);
        let ape = cuptr(inputs[15].data_ptr::<u8>() as *const c_void);
        let norm = cuptr(inputs[16].data_ptr::<u8>() as *const c_void);
        let past_key = cuptr(inputs[17].data_ptr::<u8>() as *const c_void);
        let past_carry = cuptr(inputs[18].data_ptr::<u8>() as *const c_void);
        let key = cuptr(outputs[3].data_ptr_mut::<u8>() as *const c_void);
        let carry = cuptr(outputs[4].data_ptr_mut::<u8>() as *const c_void);
        let batch_i = i32::try_from(batch).map_err(|_| not_implemented("CSA batch exceeds i32"))?;
        let sequence_i =
            i32::try_from(sequence).map_err(|_| not_implemented("CSA sequence exceeds i32"))?;
        let dim_i =
            i32::try_from(dim).map_err(|_| not_implemented("CSA index dimension exceeds i32"))?;
        let rope_i = i32::try_from(self.qk_rope_head_dim)
            .map_err(|_| not_implemented("CSA RoPE dimension exceeds i32"))?;
        let past_i =
            i32::try_from(past_records).map_err(|_| not_implemented("CSA records exceed i32"))?;
        let records_i =
            i32::try_from(key_records).map_err(|_| not_implemented("CSA records exceed i32"))?;
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&kv)
            .arg(&gate)
            .arg(&ape)
            .arg(&norm)
            .arg(&past_key)
            .arg(&past_carry)
            .arg(&key)
            .arg(&carry)
            .arg(&batch_i)
            .arg(&sequence_i)
            .arg(&dim_i)
            .arg(&rope_i)
            .arg(&past_i)
            .arg(&records_i)
            .arg(&start);
        // SAFETY: one serial thread owns each batch row, preserving every
        // order-dependent oracle reduction and carry transition.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (batch as u32, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch csa_ratio4_index_compress", error))?;
        self.runtime.synchronize()
    }

    /// B4 stages 3–5 for ratio-4: index-query finalize + `dot→relu→
    /// weighted-head-sum` scoring + deterministic top-k selection.  Reads the
    /// freshly written device `present_index_key` (output 3), computes the
    /// selection device-resident, then reads back the shared per-`(batch,query)`
    /// index set (permitted by decision D5 — indices only) and writes the
    /// per-index-head replicated `selected_indices` (output 5).
    fn run_device_index_scoring(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
    ) -> Result<()> {
        let (batch, sequence) = (inputs[0].shape[0], inputs[0].shape[1]);
        let index_heads = self.index_num_heads;
        let index_dim = self.index_head_dim;
        let records = outputs[3].shape[1];
        // selected_indices is [batch, index_heads, sequence, topk_width].
        let topk_width = outputs[5].shape[3];
        if batch == 0 || sequence == 0 || index_heads == 0 || topk_width == 0 {
            return Ok(());
        }
        let start = {
            let total_bytes = inputs[9];
            // `total_sequence_length` is a scalar Int64 device tensor.
            let mut bytes = [0u8; 8];
            // SAFETY: input 9 is a live device scalar of 8 bytes (Int64).
            unsafe {
                self.runtime.dtoh(
                    &mut bytes,
                    cuptr(total_bytes.data_ptr::<u8>() as *const c_void),
                )?;
            }
            let total = i64::from_ne_bytes(bytes);
            usize::try_from(total)
                .ok()
                .and_then(|total| total.checked_sub(sequence))
                .ok_or_else(|| {
                    not_implemented(format!("{OP}: total < sequence in device index scoring"))
                })?
        };

        let rows = batch * sequence;
        let transformed_len = rows
            .checked_mul(index_heads)
            .and_then(|value| value.checked_mul(index_dim))
            .ok_or_else(|| not_implemented(format!("{OP}: index scoring workspace overflow")))?;
        let scores_len = rows
            .checked_mul(records.max(1))
            .ok_or_else(|| not_implemented(format!("{OP}: index score buffer overflow")))?;
        let selected_len = rows
            .checked_mul(topk_width)
            .ok_or_else(|| not_implemented(format!("{OP}: index selection buffer overflow")))?;

        let transformed = self
            .runtime
            .alloc_raw(transformed_len.checked_mul(4).unwrap_or(usize::MAX).max(1))?;
        let scores = self
            .runtime
            .alloc_raw(scores_len.checked_mul(4).unwrap_or(usize::MAX).max(1))?;
        let selected = self
            .runtime
            .alloc_raw(selected_len.checked_mul(4).unwrap_or(usize::MAX).max(1))?;

        let index_key_ptr = cuptr(outputs[3].data_ptr_mut::<u8>() as *const c_void);
        let run = || -> Result<Vec<i32>> {
            let source = format!("{}\n{}", block_quant::source(), INDEX_SELECT_SOURCE);
            let func =
                self.runtime
                    .nvrtc_function(INDEX_SELECT_MODULE, &source, INDEX_SELECT_ENTRY)?;
            let index_query = cuptr(inputs[11].data_ptr::<u8>() as *const c_void);
            let index_weight = cuptr(inputs[12].data_ptr::<u8>() as *const c_void);
            let index_key = index_key_ptr;
            let batch_i =
                i32::try_from(batch).map_err(|_| not_implemented("CSA batch exceeds i32"))?;
            let sequence_i =
                i32::try_from(sequence).map_err(|_| not_implemented("CSA sequence exceeds i32"))?;
            let heads_i = i32::try_from(index_heads)
                .map_err(|_| not_implemented("CSA index heads exceed i32"))?;
            let dim_i = i32::try_from(index_dim)
                .map_err(|_| not_implemented("CSA index dimension exceeds i32"))?;
            let rope_i = i32::try_from(self.qk_rope_head_dim)
                .map_err(|_| not_implemented("CSA RoPE dimension exceeds i32"))?;
            let records_i =
                i32::try_from(records).map_err(|_| not_implemented("CSA records exceed i32"))?;
            let start_i = start as i64;
            let topk_i =
                i32::try_from(topk_width).map_err(|_| not_implemented("CSA topk exceeds i32"))?;
            let grid = u32::try_from(rows)
                .map_err(|_| not_implemented(format!("{OP}: index scoring rows exceed u32")))?;

            let mut builder = self.runtime.stream().launch_builder(&func);
            builder
                .arg(&index_query)
                .arg(&index_weight)
                .arg(&index_key)
                .arg(&transformed)
                .arg(&scores)
                .arg(&selected)
                .arg(&batch_i)
                .arg(&sequence_i)
                .arg(&heads_i)
                .arg(&dim_i)
                .arg(&rope_i)
                .arg(&records_i)
                .arg(&start_i)
                .arg(&topk_i);
            // SAFETY: argument order/types match `csa_ratio4_index_select`; every
            // pointer is a live contiguous allocation sized by the shapes above,
            // and one block per `(batch, query)` row owns its scratch slices.
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| driver_err("launch csa_ratio4_index_select", error))?;
            self.runtime.synchronize()?;

            let mut host = vec![0u8; selected_len * 4];
            // SAFETY: `selected` covers exactly `selected_len` i32 values.
            unsafe {
                self.runtime.dtoh(&mut host, selected)?;
            }
            Ok(host
                .chunks_exact(4)
                .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("4 bytes")))
                .collect())
        };

        let selection = run();
        // SAFETY: each scratch came from this runtime's `alloc_raw`, freed once.
        let free = unsafe { self.runtime.free_raw(transformed) };
        let free = free.and(unsafe { self.runtime.free_raw(scores) });
        let free = free.and(unsafe { self.runtime.free_raw(selected) });
        let shared = selection.and_then(|shared| free.map(|()| shared))?;

        // Replicate the shared `[batch, sequence, topk_width]` selection across
        // every index head into `[batch, index_heads, sequence, topk_width]`,
        // exactly like the oracle's `write_shared_selected_i32`.
        let row_width = sequence * topk_width;
        let mut replicated = vec![-1i32; batch * index_heads * row_width];
        for b in 0..batch {
            let source = &shared[b * row_width..(b + 1) * row_width];
            for head in 0..index_heads {
                let destination = (b * index_heads + head) * row_width;
                replicated[destination..destination + row_width].copy_from_slice(source);
            }
        }
        let bytes: Vec<u8> = replicated
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        if !bytes.is_empty() {
            // SAFETY: output 5 is a live device allocation whose dense size equals
            // `batch * index_heads * sequence * topk_width` i32 values.
            unsafe {
                self.runtime.htod(
                    &bytes,
                    cuptr(outputs[5].data_ptr_mut::<u8>() as *const c_void),
                )?;
            }
        }
        self.runtime.synchronize()
    }
}

impl Kernel for CompressedSparseAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // Stage every present input host-side. Contiguity is required because the
        // host copy is a dense byte blit; the CPU oracle then reads it densely.
        let mut staged: Vec<Vec<u8>> = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.iter().enumerate() {
            if input.is_absent() {
                staged.push(Vec::new());
                continue;
            }
            if !input.is_contiguous() {
                return Err(not_implemented(format!(
                    "{OP}: non-contiguous input {index} on CUDA (host-staged path requires contiguous inputs)"
                )));
            }
            let bytes = input.byte_size();
            let mut host = vec![0u8; bytes];
            if bytes > 0 {
                // SAFETY: `input` is a live contiguous device tensor and `host`
                // is exactly its dense storage size.
                unsafe {
                    self.runtime
                        .dtoh(&mut host, cuptr(input.data_ptr::<u8>() as *const c_void))?;
                }
            }
            staged.push(host);
        }

        // Build host-resident input views over the staged buffers, reusing each
        // input's (contiguous) shape/strides. `DevicePtr` is a raw pointer, so
        // these views borrow nothing from `staged` at the type level — `staged`
        // is kept alive until after `execute`.
        let host_inputs: Vec<TensorView> = inputs
            .iter()
            .zip(&staged)
            .map(|(input, buf)| {
                if input.is_absent() {
                    TensorView::absent(input.dtype)
                } else {
                    TensorView::new(
                        onnx_runtime_ep_api::DevicePtr(buf.as_ptr() as *const c_void),
                        input.dtype,
                        input.shape,
                        input.strides,
                        DeviceId::cpu(),
                    )
                }
            })
            .collect();

        // Snapshot output metadata and allocate matching host buffers. The
        // session has already shape-inferred and allocated the device outputs, so
        // their shapes are authoritative for the oracle's own shape checks.
        for (index, output) in outputs.iter().enumerate() {
            if !output.is_contiguous() {
                return Err(not_implemented(format!(
                    "{OP}: non-contiguous output {index} on CUDA (host-staged path requires contiguous outputs)"
                )));
            }
        }
        let out_dtypes: Vec<DataType> = outputs.iter().map(|o| o.dtype).collect();
        let out_shapes: Vec<Vec<usize>> = outputs.iter().map(|o| o.shape.to_vec()).collect();
        let out_strides: Vec<Vec<i64>> = outputs.iter().map(|o| o.strides.to_vec()).collect();
        let mut out_bufs: Vec<Vec<u8>> = outputs.iter().map(|o| vec![0u8; o.byte_size()]).collect();

        let mut host_outputs: Vec<TensorMut> = out_bufs
            .iter_mut()
            .enumerate()
            .map(|(index, buf)| {
                TensorMut::new(
                    onnx_runtime_ep_api::DevicePtrMut(buf.as_mut_ptr() as *mut c_void),
                    out_dtypes[index],
                    &out_shapes[index],
                    &out_strides[index],
                    DeviceId::cpu(),
                )
            })
            .collect();

        // The B0 dispatch seam deliberately routes every stage through this one
        // host oracle invocation. Later phases replace individual `Host` arms;
        // changing a mode today cannot alter numerical behavior.
        self.run_host_staged_pipeline(&host_inputs, &mut host_outputs)?;

        // Release the borrow of `out_bufs` before uploading the results.
        drop(host_outputs);
        drop(host_inputs);

        for (index, output) in outputs.iter_mut().enumerate() {
            let bytes = &out_bufs[index];
            if !bytes.is_empty() {
                // SAFETY: `output` is a live device allocation whose dense size
                // equals `bytes.len()` (built from `output.byte_size()`).
                unsafe {
                    self.runtime
                        .htod(bytes, cuptr(output.data_ptr_mut::<u8>() as *const c_void))?;
                }
            }
        }

        // B2 device stage-1: overwrite the host-oracle cache/carry with the
        // independently computed device result.  The host invocation remains
        // the compatibility path for all other stages and output shapes.
        if self.device_compression
            && self.dispatch.mode(CsaPipelineStage::CompressionUpdate) == CsaStageMode::Device
            // The f32 cache remains B1's strict-Y reference path.  Its device
            // attention consumes the exact f32 oracle record, while B2 owns the
            // hybrid FP8 record format (including its BF16 RoPE tail).
            && outputs[1].dtype == DataType::Uint8
        {
            self.run_device_compression(inputs, outputs, &staged)?;
        }

        if self.device_index_compression
            && self.dispatch.mode(CsaPipelineStage::IndexKeyUpdate) == CsaStageMode::Device
        {
            self.run_device_index_compression(inputs, outputs, &staged)?;
        }

        // B4 device stages 3–5: recompute `selected_indices` (output 5) on device
        // from the freshly written device `present_index_key` (output 3).
        if self.device_index_scoring
            && outputs.len() == 6
            && self.dispatch.mode(CsaPipelineStage::Selection) == CsaStageMode::Device
        {
            self.run_device_index_scoring(inputs, outputs)?;
        }

        // B1 device stage-7: ratio-128 f32-cache consumes the f32 candidate
        // records and recomputes `Y` on device. This path is never taken for
        // ratio-4 (its cache is the packed FP8/BF16 583-byte record, not f32).
        if self.device_attention
            && self
                .dispatch
                .mode(CsaPipelineStage::SparseSinkSoftmaxAttention)
                == CsaStageMode::Device
        {
            self.run_device_attention(inputs, outputs, &staged)?;
        }

        // B5 device stage-6/7: ratio-4 fused selection→attention. Gated on the
        // ratio-4 flag AND `selected_indices` presence (6 outputs) — the device
        // kernel dereferences the B4 device-selected record IDs held in output 5.
        // A 5-output ratio-4 node omits `selected_indices`, so there is nothing to
        // dereference: `Y` stays the host-staged oracle result already uploaded
        // above, and we must NOT fall through to `run_device_attention` (the
        // ratio-128 kernel), which would read the 583-byte packed record as
        // `f32×512` out of bounds and clobber the correct `Y`.
        if self.device_ratio4_attention
            && outputs.len() == 6
            && self
                .dispatch
                .mode(CsaPipelineStage::SparseSinkSoftmaxAttention)
                == CsaStageMode::Device
        {
            self.run_device_ratio4_attention(inputs, outputs, &staged)?;
        }

        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        // The host-staging blit is dense; strided inputs are rejected in execute.
        false
    }

    fn cuda_graph_compatible(&self) -> bool {
        // Host round-trip (D2H inputs, H2D outputs) plus per-copy stream syncs
        // are illegal during CUDA-graph capture. Device-resident capture is a
        // Phase-B goal (docs/DEEPSEEK_CSA_MTP_RUNTIME.md §4.8).
        false
    }
}

/// Claim-time denial for `pkg.nxrt::CompressedSparseAttention`. Rejects any
/// ratio / cache-layout / sink-mode / arity combination the CPU oracle does not
/// accept (via a dry-run of the CPU factory), plus explicit dtype gating on the
/// dtype-fixed inputs, so unsupported combinations never reach `execute`
/// (docs/DEEPSEEK_CSA_MTP_RUNTIME.md §4.8).
pub(crate) fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    // Attribute/arity gating: the CPU factory validates the full frozen-v1
    // attribute set and required-input names; any rejection there is a config we
    // cannot correctly execute host-staged either.
    let concrete_shapes = shapes
        .iter()
        .map(|shape| as_static_shape(shape))
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    if let Err(error) = CpuCsaFactory.create(node, &concrete_shapes) {
        return Some(Cow::Owned(format!("{OP}: {error}")));
    }

    if shapes.len() != node.inputs.len() || input_dtypes.len() != node.inputs.len() {
        return Some(Cow::Owned(format!(
            "{OP}: claim metadata must cover all {} positional inputs (got {} shapes and {} dtypes)",
            node.inputs.len(),
            shapes.len(),
            input_dtypes.len()
        )));
    }

    let ratio = usize::try_from(
        node.attr("compression_ratio")
            .and_then(|attribute| attribute.as_int())
            .expect("CPU factory accepted compression_ratio"),
    )
    .expect("CPU factory accepted positive compression_ratio");
    let cache_format = node
        .attr("cache_format")
        .and_then(|attribute| attribute.as_str())
        .unwrap_or("f32");

    // Claim-time sizing is metadata-only: it validates fixed static bounds but
    // does not query free memory or reserve device storage.
    if let Err(error) = CsaBufferLayout::from_claim(node, shapes, ratio) {
        return Some(Cow::Owned(format!("{OP}: {error}")));
    }

    let result = match ratio {
        4 => validate_ratio4_claim(node, shapes, input_dtypes, cache_format),
        128 => validate_ratio128_claim(node, shapes, input_dtypes, cache_format),
        _ => unreachable!("CPU factory rejected unsupported compression ratio"),
    }
    .and_then(|()| validate_attention_bias_claim(node, shapes, input_dtypes));

    result
        .err()
        .map(|reason| Cow::Owned(format!("{OP}: {reason}")))
}

fn validate_ratio4_claim(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
    cache_format: &str,
) -> std::result::Result<(), String> {
    if node.inputs.len() < 19 || node.inputs[11..19].iter().any(Option::is_none) {
        return Err("ratio-4 requires all eight index inputs (11..=18)".into());
    }
    if !(5..=6).contains(&node.outputs.len()) {
        return Err(format!(
            "ratio-4 requires 5 or 6 outputs, got {}",
            node.outputs.len()
        ));
    }
    if node
        .attr("index_head_dim")
        .and_then(|attribute| attribute.as_int())
        != Some(128)
    {
        return Err("ratio-4 requires index_head_dim=128".into());
    }
    if cache_format != "fp8_e4m3_block64" {
        return Err(format!(
            "ratio-4 requires cache_format='fp8_e4m3_block64', got '{cache_format}'"
        ));
    }
    require_fixed_contract(node, 4)?;

    for &(index, expected, name) in &[
        (0, DataType::Float32, "query"),
        (1, DataType::Float32, "current_kv"),
        (2, DataType::Float32, "compressor_kv"),
        (3, DataType::Float32, "compressor_gate"),
        (4, DataType::Float32, "compressor_ape"),
        (5, DataType::Float32, "compressor_norm"),
        (6, DataType::Uint8, "past_compressed_kv"),
        (7, DataType::Float32, "past_compression_carry"),
        (8, DataType::Int32, "seqlens_k"),
        (9, DataType::Int64, "total_sequence_length"),
        (10, DataType::Float32, "head_sink"),
        (11, DataType::Float32, "index_query"),
        (12, DataType::Float32, "index_weight"),
        (13, DataType::Float32, "index_compressor_kv"),
        (14, DataType::Float32, "index_compressor_gate"),
        (15, DataType::Float32, "index_compressor_ape"),
        (16, DataType::Float32, "index_compressor_norm"),
        (17, DataType::Uint8, "past_index_key"),
        (18, DataType::Float32, "past_index_carry"),
    ] {
        require_dtype(input_dtypes, index, expected, name)?;
    }
    let heads = required_attr(node, "num_heads")?;
    let index_heads = required_attr(node, "index_num_heads")?;
    for (index, name, contract) in [
        (0, "query", vec![Any, NonZero, Fixed(heads), Fixed(512)]),
        (1, "current_kv", vec![Same(0, 0), Any, Fixed(512)]),
        (
            2,
            "compressor_kv",
            vec![Same(0, 0), Same(0, 1), Fixed(1024)],
        ),
        (
            3,
            "compressor_gate",
            vec![Same(0, 0), Same(0, 1), Fixed(1024)],
        ),
        (4, "compressor_ape", vec![Fixed(4), Fixed(1024)]),
        (5, "compressor_norm", vec![Fixed(512)]),
        (6, "past_compressed_kv", vec![Same(0, 0), Any, Fixed(583)]),
        (
            7,
            "past_compression_carry",
            vec![Same(0, 0), Fixed(8), Fixed(2), Fixed(1024)],
        ),
        (8, "seqlens_k", vec![Same(0, 0)]),
        (9, "total_sequence_length", vec![]),
        (10, "head_sink", vec![Fixed(heads)]),
        (
            11,
            "index_query",
            vec![Same(0, 0), Same(0, 1), Fixed(index_heads), Fixed(128)],
        ),
        (
            12,
            "index_weight",
            vec![Same(0, 0), Same(0, 1), Fixed(index_heads)],
        ),
        (
            13,
            "index_compressor_kv",
            vec![Same(0, 0), Same(0, 1), Fixed(256)],
        ),
        (
            14,
            "index_compressor_gate",
            vec![Same(0, 0), Same(0, 1), Fixed(256)],
        ),
        (15, "index_compressor_ape", vec![Fixed(4), Fixed(256)]),
        (16, "index_compressor_norm", vec![Fixed(128)]),
        (17, "past_index_key", vec![Same(0, 0), Any, Fixed(68)]),
        (
            18,
            "past_index_carry",
            vec![Same(0, 0), Fixed(8), Fixed(2), Fixed(256)],
        ),
    ] {
        require_shape(shapes, index, name, &contract)?;
    }
    Ok(())
}

fn validate_ratio128_claim(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
    cache_format: &str,
) -> std::result::Result<(), String> {
    for index in 11..19.min(node.inputs.len()) {
        if node.inputs[index].is_some() {
            return Err(format!(
                "ratio-4-only input {index} is unsupported for ratio-128"
            ));
        }
    }
    if node.outputs.len() != 3 {
        return Err(format!(
            "ratio-128 requires exactly 3 outputs, got {}",
            node.outputs.len()
        ));
    }
    if cache_format == "fp4_e2m1_block32" {
        return Err(
            "ratio-128 attention-compressor state uses f32 or hybrid FP8/BF16 records, not FP4"
                .into(),
        );
    }
    require_fixed_contract(node, 128)?;

    let cache_dtype = if cache_format == "f32" {
        DataType::Float32
    } else {
        DataType::Uint8
    };
    for &(index, expected, name) in &[
        (0, DataType::Float32, "query"),
        (1, DataType::Float32, "current_kv"),
        (2, DataType::Float32, "compressor_kv"),
        (3, DataType::Float32, "compressor_gate"),
        (4, DataType::Float32, "compressor_ape"),
        (5, DataType::Float32, "compressor_norm"),
        (6, cache_dtype, "past_compressed_kv"),
        (7, DataType::Float32, "past_compression_carry"),
        (8, DataType::Int32, "seqlens_k"),
        (9, DataType::Int64, "total_sequence_length"),
        (10, DataType::Float32, "head_sink"),
    ] {
        require_dtype(input_dtypes, index, expected, name)?;
    }

    let heads = required_attr(node, "num_heads")?;
    let stored_width = if cache_format == "f32" { 512 } else { 583 };
    for (index, name, contract) in [
        (0, "query", vec![Any, NonZero, Fixed(heads), Fixed(512)]),
        (1, "current_kv", vec![Same(0, 0), Any, Fixed(512)]),
        (2, "compressor_kv", vec![Same(0, 0), Same(0, 1), Fixed(512)]),
        (
            3,
            "compressor_gate",
            vec![Same(0, 0), Same(0, 1), Fixed(512)],
        ),
        (4, "compressor_ape", vec![Fixed(128), Fixed(512)]),
        (5, "compressor_norm", vec![Fixed(512)]),
        (
            6,
            "past_compressed_kv",
            vec![Same(0, 0), Any, Fixed(stored_width)],
        ),
        (
            7,
            "past_compression_carry",
            vec![Same(0, 0), Fixed(128), Fixed(2), Fixed(512)],
        ),
        (8, "seqlens_k", vec![Same(0, 0)]),
        (9, "total_sequence_length", vec![]),
        (10, "head_sink", vec![Fixed(heads)]),
    ] {
        require_shape(shapes, index, name, &contract)?;
    }
    Ok(())
}

fn validate_attention_bias_claim(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> std::result::Result<(), String> {
    if !node.inputs.get(19).is_some_and(Option::is_some)
        || input_dtypes.get(19) == Some(&DataType::Undefined)
    {
        return Ok(());
    }

    require_dtype(input_dtypes, 19, DataType::Float32, "attention_bias")?;
    let bias_shape = &shapes[19];
    if bias_shape.len() > 4 {
        return Err(format!(
            "input 19 ('attention_bias') rank {} unsupported; expected rank <= 4",
            bias_shape.len()
        ));
    }

    if let Some(static_shape) = as_static_shape(bias_shape) {
        let elements = static_shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension));
        if elements
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(format!(
                "input 19 ('attention_bias') byte count overflow or exceeds isize::MAX for shape {static_shape:?}"
            ));
        }
    }

    let heads = required_attr(node, "num_heads")?;
    let target = [
        shapes[0][0].as_static(),
        Some(heads),
        shapes[0][1].as_static(),
        None,
    ];
    let offset = 4 - bias_shape.len();
    for (axis, dimension) in bias_shape.iter().enumerate() {
        let Some(got) = dimension.as_static() else {
            continue;
        };
        let target_axis = offset + axis;
        if got != 1 && target[target_axis].is_some_and(|expected| got != expected) {
            return Err(format!(
                "input 19 ('attention_bias') shape {bias_shape:?} is not broadcastable to attention scores [{:?}, {heads}, {:?}, ?]",
                shapes[0][0], shapes[0][1]
            ));
        }
    }
    Ok(())
}

fn require_fixed_contract(node: &Node, ratio: usize) -> std::result::Result<(), String> {
    if required_attr(node, "head_dim")? != 512 {
        return Err(format!("ratio-{ratio} requires head_dim=512"));
    }
    let rope_dim = match node.attr("qk_rope_head_dim") {
        Some(attribute) => attribute
            .as_int()
            .ok_or_else(|| "qk_rope_head_dim must be an integer".to_string())?,
        None => 0,
    };
    if rope_dim != 64 {
        return Err(format!("ratio-{ratio} requires qk_rope_head_dim=64"));
    }
    Ok(())
}

fn required_attr(node: &Node, name: &str) -> std::result::Result<usize, String> {
    node.attr(name)
        .and_then(|attribute| attribute.as_int())
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("missing or invalid integer attribute '{name}'"))
}

fn require_dtype(
    input_dtypes: &[DataType],
    index: usize,
    expected: DataType,
    name: &str,
) -> std::result::Result<(), String> {
    let got = input_dtypes[index];
    if got != expected {
        return Err(format!(
            "input {index} ('{name}') dtype {got:?} unsupported; expected {expected:?}"
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ShapeAxis {
    Any,
    NonZero,
    Fixed(usize),
    Same(usize, usize),
}
use ShapeAxis::{Any, Fixed, NonZero, Same};

fn require_shape(
    shapes: &[Shape],
    index: usize,
    name: &str,
    contract: &[ShapeAxis],
) -> std::result::Result<(), String> {
    let shape = &shapes[index];
    if shape.len() != contract.len() {
        return Err(format!(
            "input {index} ('{name}') rank {} unsupported; expected {}",
            shape.len(),
            contract.len()
        ));
    }
    for (axis, requirement) in contract.iter().enumerate() {
        let mismatch = match requirement {
            Any => None,
            NonZero if shape[axis] == Dim::Static(0) => Some("must be nonzero".into()),
            NonZero => None,
            Fixed(expected) => shape[axis]
                .as_static()
                .filter(|got| got != expected)
                .map(|got| format!("is {got}; expected {expected}")),
            Same(other_input, other_axis) => {
                match (
                    shape[axis].as_static(),
                    shapes[*other_input][*other_axis].as_static(),
                ) {
                    (Some(got), Some(expected)) if got != expected => {
                        Some(format!("is {got}; expected {expected}"))
                    }
                    _ => None,
                }
            }
        };
        if let Some(mismatch) = mismatch {
            return Err(format!("input {index} ('{name}') axis {axis} {mismatch}"));
        }
    }
    Ok(())
}
