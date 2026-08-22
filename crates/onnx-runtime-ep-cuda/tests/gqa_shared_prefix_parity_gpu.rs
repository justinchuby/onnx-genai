//! First production consumer of the pinned shared-prefix primitive (#777).
//!
//! PR #803 landed `create_shared_prefix` / `commit_shared_prefix` on the
//! production `CudaVmmAllocator`, but nothing outside its own GPU tests called
//! them — the exact "contract with no live consumer" shape that stranded #721
//! stage 3. This binary wires the primitive to a **real consumer**: the fused
//! fp16 seq-major (BSNH) group-query-attention decode kernel (#782/#792), the
//! same kernel the `gqa_seqmajor_parity_gpu` oracle proves bit-identical to
//! head-major. It drives that kernel over KV caches that are **physically
//! shared** through the allocator-agnostic `SharedMapping` seam. A protection
//! fault from an invalid kernel store is fail-stop, like any CUDA illegal
//! address; correct decode kernels only read the shared prefix.
//!
//! 1. **Byte-identical output.** Two sequences whose read-only prefix KV is one
//!    physically shared set of granules produce **byte-identical** decode output
//!    to two independent sequences that each own a private copy of that prefix —
//!    for both a K cache and a V cache, exactly the `layers × 2` seq-major
//!    contiguous ranges the layout gives (`docs/memory/MEMORY_ARCHITECTURE.md`, "KV
//!    layout and residency"). An independent CPU oracle guards against the two
//!    GPU paths being symmetrically wrong.
//! 2. **Request-local fallback.** A third request starts with an incompatible
//!    value target whose prefix granule is already privately committed. The
//!    production K/V pair transaction rolls back the key mapping, reports that
//!    private fallback is safe, and the request continues with fully private KV
//!    while its peers keep sharing.
//! 3. **Admission is private bytes only.** Admitting the second sharer needs
//!    **zero** incremental owned bytes for the shared prefix
//!    (`incremental_owned_bytes_for_shared_prefix == 0`), and the governor's
//!    owned axis rises by only that sequence's private tail — the arithmetic
//!    (#745) that turns prefix sharing into concurrency (#750).
//! 4. **Physical bytes are charged once.** Two sequences sharing one prefix
//!    commit strictly fewer physical granules than two independent sequences, by
//!    exactly `(sharers − 1) × prefix_granules` for each of the K and V caches.
//!
//! # Harness discipline (#797, #804)
//!
//! Every device operation — the VMM allocator, the prefix fill, the kernel, and
//! all readbacks — runs on the **one** `CudaExecutionProvider` context and its
//! single stream (the VMM allocator is built on `runtime().cuda_context()`), so
//! a readback can never race a fill on a mutually-unsynchronized stream. The
//! physical-handle-pool option is set **before** any allocator is constructed,
//! and this binary is one test in its own process; run it single-threaded
//! (`--test-threads=1`).

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::uninlined_format_args,
    clippy::manual_is_multiple_of
)]

use std::ffi::c_void;

use half::f16;
use onnx_runtime_cuda_memory::vmm_allocator::{
    CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, CudaVmmAllocator,
};
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::{CudaExecutionProvider, GroupQueryAttentionKernel};
use onnx_runtime_ir::{DataType, compute_contiguous_strides};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, KvFragmentation, LeaseLedger, LedgerGovernor,
    MemoryGovernor, MemoryRole, ModelKvGeometry, SharedDevicePrefix, SharedPrefixCommitTarget,
    Tier, commit_shared_prefix_pair, evaluate_prefix_shareability,
};

const BATCH: usize = 1;
const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const GROUP: usize = QUERY_HEADS / KV_HEADS;
const ELEM: usize = std::mem::size_of::<f16>();
const HOLDER: HolderId = HolderId::new(777);

/// One f16 KV token of a single binding occupies `KV_HEADS × HEAD_DIM × ELEM`
/// bytes, identical for both layouts (`docs/memory/MEMORY_ARCHITECTURE.md`).
const BYTES_PER_TOKEN: usize = KV_HEADS * HEAD_DIM * ELEM;

fn typed_bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: test data is plain-old-data with no padding.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn require_cuda() -> CudaExecutionProvider {
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => panic!(
            "CUDA test requires CUDA device/runtime; CPU-only runs must leave this test ignored: {error}"
        ),
        Err(_) => panic!(
            "CUDA test requires CUDA runtime libraries; CPU-only runs must leave this test ignored"
        ),
    }
}

fn set_pool_env() {
    // SAFETY: this integration test owns its process and sets the pool option
    // before constructing any allocator (the #804 freeze-on-first-read trap does
    // not apply — the allocator reads this at construction). A non-zero bound
    // installs the production physical-handle pool the shared-prefix primitive
    // requires.
    unsafe {
        std::env::set_var(
            CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
            (64usize << 20).to_string(),
        );
    }
}

/// Fill `[dst, dst+src.len())` on the shared EP stream (blocking copy).
fn upload_to(ep: &CudaExecutionProvider, dst: u64, src: &[u8]) {
    if src.is_empty() {
        return;
    }
    // SAFETY: `dst` addresses at least `src.len()` committed device bytes.
    unsafe { ep.runtime().htod(src, dst).expect("htod") }
}

/// Read `bytes` from device address `src` (synchronizes the EP stream first).
fn read_from(ep: &CudaExecutionProvider, src: u64, bytes: usize) -> Vec<u8> {
    let mut host = vec![0u8; bytes];
    // SAFETY: `src` addresses at least `bytes` committed device bytes.
    unsafe { ep.runtime().dtoh(&mut host, src).expect("dtoh") }
    host
}

/// Seq-major (BSNH) flat index of token `t`, kv-head `h`, dim `d` within a
/// binding of `capacity` tokens: `((t × kv_heads) + h) × head_dim + d`.
fn bsnh_index(t: usize, h: usize, d: usize) -> usize {
    (t * KV_HEADS + h) * HEAD_DIM + d
}

/// Lay `logical[h][t][d]` into a seq-major buffer of `capacity` tokens.
fn seed_seqmajor(logical: &[Vec<Vec<f16>>], capacity: usize) -> Vec<f16> {
    let mut buffer = vec![f16::ZERO; capacity * KV_HEADS * HEAD_DIM];
    for h in 0..KV_HEADS {
        for t in 0..logical[h].len() {
            for d in 0..HEAD_DIM {
                buffer[bsnh_index(t, h, d)] = logical[h][t][d];
            }
        }
    }
    buffer
}

/// Softmax attention over positions `0..=past` per query head; GQA maps query
/// head `qh` to kv head `qh / GROUP`. `key`/`value` are indexed `[kvh][pos][d]`
/// over `past + 1` positions (prefix plus the one appended token).
fn cpu_reference(
    query: &[f16],
    key: &[Vec<Vec<f16>>],
    value: &[Vec<Vec<f16>>],
    scale: f32,
) -> Vec<f32> {
    let valid = key[0].len();
    let mut out = vec![0.0_f32; QUERY_HEADS * HEAD_DIM];
    for qh in 0..QUERY_HEADS {
        let kvh = qh / GROUP;
        let mut scores = vec![0.0_f32; valid];
        let mut max_score = f32::NEG_INFINITY;
        for p in 0..valid {
            let mut dot = 0.0_f32;
            for d in 0..HEAD_DIM {
                dot += query[qh * HEAD_DIM + d].to_f32() * key[kvh][p][d].to_f32();
            }
            scores[p] = dot * scale;
            max_score = max_score.max(scores[p]);
        }
        let mut denom = 0.0_f32;
        for p in 0..valid {
            scores[p] = (scores[p] - max_score).exp();
            denom += scores[p];
        }
        for d in 0..HEAD_DIM {
            let mut acc = 0.0_f32;
            for p in 0..valid {
                acc += scores[p] / denom * value[kvh][p][d].to_f32();
            }
            out[qh * HEAD_DIM + d] = acc;
        }
    }
    out
}

fn fp16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn granularity_of(ep: &CudaExecutionProvider) -> usize {
    use cudarc::driver::sys as cu;
    let device = ep.device_id().index as i32;
    let mut prop: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    let mut granularity = 0usize;
    let result = unsafe {
        cu::cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
    };
    assert_eq!(
        result,
        cu::CUresult::CUDA_SUCCESS,
        "granularity: {result:?}"
    );
    assert_ne!(granularity, 0, "CUDA reported zero VMM granularity");
    granularity
}

/// Inputs to one decode step, uploaded once and reused across every cache the
/// kernel is run over (independent or shared) for one sequence.
struct DecodeInputs {
    query: onnx_runtime_ep_api::DeviceBuffer,
    current_key: onnx_runtime_ep_api::DeviceBuffer,
    current_value: onnx_runtime_ep_api::DeviceBuffer,
    seqlens: onnx_runtime_ep_api::DeviceBuffer,
    total: onnx_runtime_ep_api::DeviceBuffer,
}

type DecodeStep = (
    DecodeInputs,
    Vec<f16>,
    Vec<Vec<Vec<f16>>>,
    Vec<Vec<Vec<f16>>>,
);

/// Build one decode step's non-cache inputs for a sequence whose appended token
/// is derived from `seed`. Returns the uploaded device inputs plus the logical
/// K/V (prefix positions plus the appended token) for the CPU oracle.
fn build_inputs(
    ep: &CudaExecutionProvider,
    past_key: &[Vec<Vec<f16>>],
    past_value: &[Vec<Vec<f16>>],
    past_len: usize,
    seed: u32,
) -> DecodeStep {
    let query: Vec<f16> = (0..QUERY_HEADS * HEAD_DIM)
        .map(|i| f16::from_f32((((i as u32 * 17 + seed * 3) % 97) as f32 - 48.0) / 256.0))
        .collect();

    // The appended token (position past_len) is this sequence's private K/V.
    let mut current_key = vec![f16::ZERO; KV_HEADS * HEAD_DIM];
    let mut current_value = vec![f16::ZERO; KV_HEADS * HEAD_DIM];
    let mut full_key = past_key.to_vec();
    let mut full_value = past_value.to_vec();
    for h in 0..KV_HEADS {
        let mut ck = vec![f16::ZERO; HEAD_DIM];
        let mut cv = vec![f16::ZERO; HEAD_DIM];
        for d in 0..HEAD_DIM {
            let kb = ((h as u32 * 37 + d as u32 * 5 + seed * 11) % 103) as f32;
            let vb = ((h as u32 * 23 + d as u32 * 9 + seed * 7) % 109) as f32;
            let kv = f16::from_f32((kb - 51.0) / 256.0);
            let vv = f16::from_f32((vb - 54.0) / 128.0);
            current_key[h * HEAD_DIM + d] = kv;
            current_value[h * HEAD_DIM + d] = vv;
            ck[d] = kv;
            cv[d] = vv;
        }
        full_key[h].push(ck);
        full_value[h].push(cv);
    }
    debug_assert_eq!(full_key[0].len(), past_len + 1);

    let seqlens = [past_len as i32];
    let total = [(past_len + 1) as i32];
    let inputs = DecodeInputs {
        query: upload(ep, typed_bytes(&query)),
        current_key: upload(ep, typed_bytes(&current_key)),
        current_value: upload(ep, typed_bytes(&current_value)),
        seqlens: upload(ep, typed_bytes(&seqlens)),
        total: upload(ep, typed_bytes(&total)),
    };
    (inputs, query, full_key, full_value)
}

fn upload(ep: &CudaExecutionProvider, bytes: &[u8]) -> onnx_runtime_ep_api::DeviceBuffer {
    let mut buffer = ep
        .allocate(bytes.len().max(1), 256)
        .expect("allocate input");
    upload_to(ep, buffer.as_mut_ptr() as u64, bytes);
    buffer
}

/// Run one fused seq-major decode step over the given K/V cache device pointers
/// and return the fp16 output bytes. The cache pointers may be independent
/// `cudaMalloc` buffers or shared VMM reservations — the kernel neither knows
/// nor cares.
fn run_decode(
    ep: &CudaExecutionProvider,
    inputs: &DecodeInputs,
    key_cache: *mut c_void,
    value_cache: *mut c_void,
    capacity: usize,
) -> Vec<u8> {
    let runtime = ep.runtime();
    let device = ep.device_id();
    let scale = 1.0_f32;

    let kernel = GroupQueryAttentionKernel::new(
        runtime.clone(),
        QUERY_HEADS,
        KV_HEADS,
        Some(scale),
        false,
        false,
        -1,
        0.0,
    )
    .unwrap()
    .with_kv_layout(1);

    let query_shape = [BATCH, 1, QUERY_HEADS * HEAD_DIM];
    let current_shape = [BATCH, 1, KV_HEADS * HEAD_DIM];
    let cache_shape = [BATCH, KV_HEADS, capacity, HEAD_DIM];
    let seqlens_shape = [BATCH];
    let scalar_shape: [usize; 0] = [];
    let output_shape = query_shape;

    let query_strides = compute_contiguous_strides(&query_shape);
    let current_strides = compute_contiguous_strides(&current_shape);
    let cache_strides = compute_contiguous_strides(&cache_shape);
    let seqlens_strides = compute_contiguous_strides(&seqlens_shape);
    let scalar_strides = compute_contiguous_strides(&scalar_shape);
    let output_strides = compute_contiguous_strides(&output_shape);

    let mut output_buffer = ep
        .allocate(QUERY_HEADS * HEAD_DIM * ELEM, 256)
        .expect("output");

    let inputs_arr = [
        TensorView::new(
            DevicePtr(inputs.query.as_ptr()),
            DataType::Float16,
            &query_shape,
            &query_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(inputs.current_key.as_ptr()),
            DataType::Float16,
            &current_shape,
            &current_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(inputs.current_value.as_ptr()),
            DataType::Float16,
            &current_shape,
            &current_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(key_cache as *const c_void),
            DataType::Float16,
            &cache_shape,
            &cache_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(value_cache as *const c_void),
            DataType::Float16,
            &cache_shape,
            &cache_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(inputs.seqlens.as_ptr()),
            DataType::Int32,
            &seqlens_shape,
            &seqlens_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(inputs.total.as_ptr()),
            DataType::Int32,
            &scalar_shape,
            &scalar_strides,
            device,
        ),
    ];
    let mut outputs_arr = [
        TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            DataType::Float16,
            &output_shape,
            &output_strides,
            device,
        ),
        TensorMut::new(
            DevicePtrMut(key_cache),
            DataType::Float16,
            &cache_shape,
            &cache_strides,
            device,
        ),
        TensorMut::new(
            DevicePtrMut(value_cache),
            DataType::Float16,
            &cache_shape,
            &cache_strides,
            device,
        ),
    ];
    kernel.execute(&inputs_arr, &mut outputs_arr).unwrap();

    let out = read_from(
        ep,
        output_buffer.as_ptr() as u64,
        QUERY_HEADS * HEAD_DIM * ELEM,
    );
    ep.deallocate(output_buffer).unwrap();
    out
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn shared_peers_survive_request_local_private_fallback() {
    set_pool_env();
    let ep = require_cuda();
    let context = ep.runtime().cuda_context();
    let granule = granularity_of(&ep);
    assert_eq!(
        granule % BYTES_PER_TOKEN,
        0,
        "test geometry assumes the granule is a whole number of KV tokens"
    );

    // One granule of prefix per binding; a two-granule cache so both decode
    // tokens land in the writable private tail (granule 1), never the read-only
    // shared prefix (granule 0).
    let prefix_tokens = granule / BYTES_PER_TOKEN;
    let capacity = 2 * prefix_tokens;
    let past_len = prefix_tokens;
    let alloc_bytes = capacity * BYTES_PER_TOKEN;
    assert_eq!(alloc_bytes, 2 * granule);

    // Drive the share decision through the layout-general arithmetic predicate
    // (#777), not a hard-coded "seq-major is shareable" assumption. This binding
    // is one seq-major (BSNH) layer's KV: each of K and V is one contiguous
    // fragment of `KV_HEADS x HEAD_DIM` bytes per token, so a one-granule prefix
    // reaches exactly one shareable granule per fragment and two multi-map ops
    // (K and V) — which is precisely what this test then performs. If the
    // arithmetic ever says this configuration is not shareable, refuse loudly
    // here rather than mis-map.
    let share = evaluate_prefix_shareability(
        KvFragmentation::seq_major_bsnh(ModelKvGeometry {
            layers: 1,
            kv_heads: KV_HEADS as u64,
            head_dim: HEAD_DIM as u64,
            dtype_bytes: ELEM as u64,
        }),
        prefix_tokens as u64,
        granule as u64,
    );
    assert!(
        share.shareable,
        "arithmetic predicate must admit this prefix before we share it: {:?}",
        share.refusal_reason()
    );
    assert_eq!(share.shareable_granules_per_fragment, 1);
    assert_eq!(share.multi_map_ops, 2, "one shared map for each of K and V");

    // Deterministic shared prefix content: positions 0..past_len, identical for
    // every sequence (a pinned system prompt).
    let mut prefix_key = vec![Vec::new(); KV_HEADS];
    let mut prefix_value = vec![Vec::new(); KV_HEADS];
    for h in 0..KV_HEADS {
        for t in 0..past_len {
            let mut k = vec![f16::ZERO; HEAD_DIM];
            let mut v = vec![f16::ZERO; HEAD_DIM];
            for d in 0..HEAD_DIM {
                let kb = ((h * 131 + (t % 101) * 13 + d * 7) % 101) as f32;
                let vb = ((h * 29 + (t % 97) * 19 + d * 3) % 113) as f32;
                k[d] = f16::from_f32((kb - 50.0) / 256.0);
                v[d] = f16::from_f32((vb - 56.0) / 128.0);
            }
            prefix_key[h].push(k);
            prefix_value[h].push(v);
        }
    }
    let prefix_key_bytes = typed_bytes(&seed_seqmajor(&prefix_key, prefix_tokens)).to_vec();
    let prefix_value_bytes = typed_bytes(&seed_seqmajor(&prefix_value, prefix_tokens)).to_vec();
    assert_eq!(prefix_key_bytes.len(), granule);

    // ------------------------------------------------------------------ //
    // Shared: one pinned prefix (K and V), charged once, mapped read-only //
    // into both sequences through the DeviceAllocator seam (#777).        //
    // ------------------------------------------------------------------ //
    let shared_governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let shared_alloc = CudaVmmAllocator::new(
        context.clone(),
        DeviceKey::device(0),
        ep.device_id().index as i32,
        512 << 20,
        &shared_governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("shared VMM allocator");
    let shared_stats = shared_alloc
        .physical_pool_stats()
        .expect("production allocator installs the physical pool");
    let shared: &dyn DeviceAllocator = &shared_alloc;
    let shared_backing = shared
        .as_virtual_backing()
        .expect("shared VMM exposes VirtualBacking");
    let shared_mapping = shared
        .as_shared_mapping()
        .expect("pooled VMM exposes SharedMapping independently");

    // Create the two pinned prefixes (K, V) through the production capability
    // seam. Their first stream writes are deliberately deferred until after the
    // request-local fallback decision.
    let key_prefix: Box<dyn SharedDevicePrefix> = shared_mapping
        .create_shared_prefix(granule)
        .expect("pin key prefix");
    let value_prefix: Box<dyn SharedDevicePrefix> = shared_mapping
        .create_shared_prefix(granule)
        .expect("pin value prefix");
    assert_eq!(key_prefix.committed_physical_bytes(), granule as u64);
    assert_eq!(value_prefix.committed_physical_bytes(), granule as u64);

    // Two prefixes, charged exactly once on the owned axis.
    assert_eq!(
        shared_governor.used(Tier::Device),
        (2 * granule) as u64,
        "the two shared prefixes (K and V) are charged exactly once each"
    );

    let mut actual_out = vec![vec![Vec::new(); 2]; 3];
    let mut shared_ptrs = Vec::new();
    let mut second_sharer_admission = None;
    for request in 0..2usize {
        let owned_before = shared_governor.used(Tier::Device);

        // Each sequence commits only its PRIVATE tail granule (granule 1); the
        // prefix region (granule 0) is left uncommitted for the shared map.
        let tail = granule..alloc_bytes;
        let key_ptr = shared_backing
            .allocate_committed(alloc_bytes, granule, std::slice::from_ref(&tail))
            .expect("shared-seq key reservation");
        let value_ptr = shared_backing
            .allocate_committed(alloc_bytes, granule, std::slice::from_ref(&tail))
            .expect("shared-seq value reservation");

        // The shared prefix costs zero incremental owned bytes to admit.
        assert_eq!(
            shared_mapping
                .incremental_owned_bytes_for_shared_prefix(key_prefix.as_ref())
                .expect("key prefix belongs to this mapping capability"),
            0,
            "admitting sharer {request} needs zero incremental owned bytes for the key prefix"
        );
        assert_eq!(
            shared_mapping
                .incremental_owned_bytes_for_shared_prefix(value_prefix.as_ref())
                .expect("value prefix belongs to this mapping capability"),
            0
        );

        let owned_after_private = shared_governor.used(Tier::Device);

        // Map K and V as one admission transaction. A caller never receives a
        // half-converted cache if the second map fails.
        let [kc, vc] = commit_shared_prefix_pair(
            shared,
            SharedPrefixCommitTarget {
                prefix: key_prefix.as_ref(),
                ptr: key_ptr,
                allocation_bytes: alloc_bytes,
                align: granule,
                byte_offset: 0,
            },
            SharedPrefixCommitTarget {
                prefix: value_prefix.as_ref(),
                ptr: value_ptr,
                allocation_bytes: alloc_bytes,
                align: granule,
                byte_offset: 0,
            },
        )
        .expect("map K/V prefix pair into sequence");
        assert_eq!(
            kc.additional_owned_bytes, 0,
            "the shared map charges no owned bytes"
        );
        assert_eq!(vc.additional_owned_bytes, 0);
        assert_eq!(kc.granules, 1);
        assert_eq!(vc.granules, 1);

        let owned_after_commit = shared_governor.used(Tier::Device);
        assert_eq!(
            owned_after_commit, owned_after_private,
            "mapping the shared prefix must not move the owned axis"
        );

        // This sequence's admission cost = its private tails only (K + V).
        let admission = owned_after_private - owned_before;
        assert_eq!(
            admission,
            (2 * granule) as u64,
            "sharer {request} pays only its two private tail granules"
        );
        if request == 1 {
            second_sharer_admission = Some(admission);
        }

        shared_ptrs.push(Some((key_ptr, value_ptr, alloc_bytes)));
    }

    // Request 2 has an empty key prefix target but a precommitted value prefix.
    // The first map therefore succeeds and the second fails, forcing the
    // production pair transaction to prove it can restore all-private topology.
    let tail = granule..alloc_bytes;
    let full = 0..alloc_bytes;
    let fallback_candidate_key = shared_backing
        .allocate_committed(alloc_bytes, granule, std::slice::from_ref(&tail))
        .expect("fallback candidate key");
    let fallback_candidate_value = shared_backing
        .allocate_committed(alloc_bytes, granule, std::slice::from_ref(&full))
        .expect("fallback candidate value");
    let before_failed_commit = shared_governor.used(Tier::Device);
    let commit_error = commit_shared_prefix_pair(
        shared,
        SharedPrefixCommitTarget {
            prefix: key_prefix.as_ref(),
            ptr: fallback_candidate_key,
            allocation_bytes: alloc_bytes,
            align: granule,
            byte_offset: 0,
        },
        SharedPrefixCommitTarget {
            prefix: value_prefix.as_ref(),
            ptr: fallback_candidate_value,
            allocation_bytes: alloc_bytes,
            align: granule,
            byte_offset: 0,
        },
    )
    .expect_err("precommitted V must roll back the successful K shared map");
    assert!(
        commit_error.private_fallback_is_safe(),
        "a request may fall back only after the pair transaction removed K"
    );
    assert!(
        commit_error.to_string().contains("already committed"),
        "fallback must report the request-local target conflict explicitly: {commit_error}"
    );
    assert_eq!(
        shared_backing.allocation_committed_bytes(fallback_candidate_key, alloc_bytes, granule),
        granule,
        "failed V commit must roll K back to its private tail only"
    );
    assert_eq!(
        shared_backing.allocation_committed_bytes(fallback_candidate_value, alloc_bytes, granule),
        alloc_bytes,
        "the pre-existing private V allocation remains intact"
    );
    assert_eq!(
        shared_governor.used(Tier::Device),
        before_failed_commit,
        "failed shared commit must not change physical-byte accounting"
    );

    // Complete K privately only after the pair transaction explicitly says
    // fallback is safe. V was already fully private; no peer mapping is touched.
    shared_backing
        .commit_allocation_range(fallback_candidate_key, alloc_bytes, granule, 0, granule)
        .expect("commit private K prefix after shared-pair refusal");
    assert_eq!(
        shared_governor.used(Tier::Device),
        (2 * granule + 2 * 2 * granule + 2 * 2 * granule) as u64,
        "fallback target must remain charged as fully private K/V"
    );

    let fallback_key = fallback_candidate_key;
    let fallback_value = fallback_candidate_value;

    // The host-only admission/commit decision is now complete. From this point
    // onward the harness may enqueue stream copies and real GQA decode work.
    upload_to(&ep, key_prefix.device_ptr(), &prefix_key_bytes);
    upload_to(&ep, value_prefix.device_ptr(), &prefix_value_bytes);
    let zero = vec![0u8; granule];
    for (key_ptr, value_ptr, _) in shared_ptrs.iter().flatten() {
        upload_to(&ep, key_ptr.as_ptr() as u64 + granule as u64, &zero);
        upload_to(&ep, value_ptr.as_ptr() as u64 + granule as u64, &zero);
    }

    // Three genuinely different requests, each decoded for two tokens. Build
    // complete logical histories for the independent CPU oracle and upload the
    // request-specific kernel inputs.
    let mut request_steps = Vec::new();
    for request in 0..3u32 {
        let mut key_history = prefix_key.clone();
        let mut value_history = prefix_value.clone();
        let mut steps = Vec::new();
        for step in 0..2u32 {
            let seed = 1 + request * 17 + step * 101;
            let built = build_inputs(
                &ep,
                &key_history,
                &value_history,
                past_len + step as usize,
                seed,
            );
            key_history = built.2.clone();
            value_history = built.3.clone();
            steps.push(built);
        }
        request_steps.push(steps);
    }

    // Independent baseline: each request owns a full private KV cache.
    let indep_governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let indep_alloc = CudaVmmAllocator::new(
        context.clone(),
        DeviceKey::device(0),
        ep.device_id().index as i32,
        512 << 20,
        &indep_governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("independent VMM allocator");
    let indep: &dyn DeviceAllocator = &indep_alloc;
    let indep_backing = indep
        .as_virtual_backing()
        .expect("VMM exposes VirtualBacking through the selected allocator");

    let mut indep_ptrs = Vec::new();
    for _request in 0..3usize {
        let key_ptr = indep_backing
            .allocate_committed(alloc_bytes, granule, std::slice::from_ref(&full))
            .expect("independent key cache");
        let value_ptr = indep_backing
            .allocate_committed(alloc_bytes, granule, std::slice::from_ref(&full))
            .expect("independent value cache");
        upload_to(&ep, key_ptr.as_ptr() as u64, &prefix_key_bytes);
        upload_to(&ep, value_ptr.as_ptr() as u64, &prefix_value_bytes);
        upload_to(&ep, key_ptr.as_ptr() as u64 + granule as u64, &zero);
        upload_to(&ep, value_ptr.as_ptr() as u64 + granule as u64, &zero);
        indep_ptrs.push((key_ptr, value_ptr));
    }
    let independent_committed = indep_governor.used(Tier::Device);
    assert_eq!(
        independent_committed,
        (3 * 2 * 2 * granule) as u64,
        "three independent requests commit a full private prefix + tail for K and V"
    );

    upload_to(&ep, fallback_key.as_ptr() as u64, &prefix_key_bytes);
    upload_to(&ep, fallback_value.as_ptr() as u64, &prefix_value_bytes);
    upload_to(&ep, fallback_key.as_ptr() as u64 + granule as u64, &zero);
    upload_to(&ep, fallback_value.as_ptr() as u64 + granule as u64, &zero);
    shared_ptrs.push(Some((fallback_key, fallback_value, alloc_bytes)));

    let shared_prefix_key_before = read_from(&ep, key_prefix.device_ptr(), granule);
    let shared_prefix_value_before = read_from(&ep, value_prefix.device_ptr(), granule);

    // Only after the fallback transaction is settled do any real GQA decode
    // enqueues occur. First produce fully private GPU references.
    let mut decode_enqueues = 0usize;
    let mut independent_out = Vec::new();
    for request in 0..3usize {
        let (key_ptr, value_ptr) = indep_ptrs[request];
        let mut outputs = Vec::new();
        for step in 0..2usize {
            outputs.push(run_decode(
                &ep,
                &request_steps[request][step].0,
                key_ptr.as_ptr().cast(),
                value_ptr.as_ptr().cast(),
                capacity,
            ));
            decode_enqueues += 1;
        }
        independent_out.push(outputs);
    }

    // Confirm the CPU oracle: every independent GPU decode step matches an
    // independent CPU reference so the shared path cannot be symmetrically wrong.
    for request in 0..3usize {
        for step in 0..2usize {
            let (_, query, full_key, full_value) = &request_steps[request][step];
            let got = fp16_values(&independent_out[request][step]);
            let expected = cpu_reference(query, full_key, full_value, 1.0);
            let max_err = got
                .iter()
                .zip(&expected)
                .map(|(g, e)| (g - e).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_err < 3e-3,
                "independent seq-major GPU decode diverged from CPU oracle \
                 (request {request}, step {step}): max_abs={max_err:e}"
            );
        }
    }

    // Interleave the two shared peers and the private fallback across two
    // decode rounds. Peer 0 completes and exits first; peer 1 and the fallback
    // then continue, proving that retiring one shared mapping does not disturb
    // either surviving request.
    let before_exit = [(0usize, 0usize), (2, 0), (1, 0), (0, 1)];
    for (request, step) in before_exit {
        let (key_ptr, value_ptr, _) = shared_ptrs[request].expect("live request");
        actual_out[request][step] = run_decode(
            &ep,
            &request_steps[request][step].0,
            key_ptr.as_ptr().cast(),
            value_ptr.as_ptr().cast(),
            capacity,
        );
        decode_enqueues += 1;
    }

    let mapped_before_exit = shared_stats.snapshot().mapped_bytes;
    let (departed_key, departed_value, departed_bytes) =
        shared_ptrs[0].take().expect("peer 0 exits once");
    unsafe {
        shared_alloc.deallocate(departed_key, departed_bytes, granule);
        shared_alloc.deallocate(departed_value, departed_bytes, granule);
    }
    assert_eq!(
        shared_stats.snapshot().mapped_bytes,
        mapped_before_exit - (4 * granule) as u64,
        "peer exit must retire its K/V shared maps and private tails"
    );

    for (request, step) in [(2usize, 1usize), (1, 1)] {
        let (key_ptr, value_ptr, _) = shared_ptrs[request].expect("surviving request");
        actual_out[request][step] = run_decode(
            &ep,
            &request_steps[request][step].0,
            key_ptr.as_ptr().cast(),
            value_ptr.as_ptr().cast(),
            capacity,
        );
        decode_enqueues += 1;
    }
    assert_eq!(decode_enqueues, 12);

    let shared_committed = shared_governor.used(Tier::Device);
    // 2 prefixes (charged once) + 2 shared peers' private K/V tails + one
    // fallback request's fully private K/V.
    assert_eq!(
        shared_committed,
        (2 * granule + 2 * 2 * granule + 2 * 2 * granule) as u64,
        "fallback accounting must include shared prefixes, peer tails, and fully private KV"
    );

    // ------------------------------------------------------------------ //
    // Published correctness and isolation results.                        //
    // ------------------------------------------------------------------ //

    // Every interleaved shared/fallback output is byte-identical to the same
    // request running alone over fully private KV.
    for step in 0..2usize {
        assert_ne!(
            independent_out[0][step], independent_out[1][step],
            "shared peers need distinct data so cross-request contamination is observable"
        );
        assert_ne!(
            independent_out[0][step], independent_out[2][step],
            "peer and fallback outputs need distinct data"
        );
        assert_ne!(
            independent_out[1][step], independent_out[2][step],
            "peer and fallback outputs need distinct data"
        );
    }
    for request in 0..3usize {
        for step in 0..2usize {
            assert_eq!(
                actual_out[request][step], independent_out[request][step],
                "request {request} step {step} must match its independent private decode"
            );
        }
    }

    // The physical K/V prefix owner windows remain byte-for-byte unchanged
    // after both peers and the fallback request have progressed.
    assert_eq!(
        read_from(&ep, key_prefix.device_ptr(), granule),
        shared_prefix_key_before,
        "shared physical K prefix must remain immutable"
    );
    assert_eq!(
        read_from(&ep, value_prefix.device_ptr(), granule),
        shared_prefix_value_before,
        "shared physical V prefix must remain immutable"
    );

    // Physical bytes: only the two admitted peers share, removing one prefix
    // copy for each of K and V. The fallback remains truthfully fully private.
    let prefix_copies_removed = independent_committed - shared_committed;
    assert_eq!(
        prefix_copies_removed,
        (2 * granule) as u64,
        "two shared peers plus one private fallback remove one K and one V prefix copy"
    );

    eprintln!(
        "gqa_shared_prefix_fallback: prefix_tokens={} capacity={} granule={}B\n  \
         committed physical bytes: independent={}B ({} granules), shared={}B ({} granules)\n  \
         removed by two-peer sharing: {}B ({} granules)\n  \
         second-sharer admission: {}B (private tails only), prefix incremental owned = 0\n  \
         fallback: failed V share rolled K back before enqueue, then both stayed private\n  \
         output: 3 requests x 2 interleaved steps byte-identical to independent + CPU",
        prefix_tokens,
        capacity,
        granule,
        independent_committed,
        independent_committed / granule as u64,
        shared_committed,
        shared_committed / granule as u64,
        prefix_copies_removed,
        prefix_copies_removed / granule as u64,
        second_sharer_admission.unwrap(),
    );

    // Teardown, no assertions inside any Drop (#777 platform rule): explicitly
    // free live reservations, then prove both ledgers return to zero.
    for (key_ptr, value_ptr) in indep_ptrs {
        // SAFETY: live allocations from `indep_alloc`, no CUDA work in flight.
        unsafe {
            indep_alloc.deallocate(key_ptr, alloc_bytes, granule);
            indep_alloc.deallocate(value_ptr, alloc_bytes, granule);
        }
    }
    for (key_ptr, value_ptr, bytes) in shared_ptrs.into_iter().flatten() {
        // SAFETY: live allocations from `shared_alloc`, no CUDA work in flight.
        unsafe {
            shared_alloc.deallocate(key_ptr, bytes, granule);
            shared_alloc.deallocate(value_ptr, bytes, granule);
        }
    }
    drop(key_prefix);
    drop(value_prefix);
    drop(shared_alloc);
    drop(indep_alloc);
    assert_eq!(
        shared_governor.used(Tier::Device),
        0,
        "shared/fallback cleanup must return all physical-byte accounting"
    );
    assert_eq!(
        indep_governor.used(Tier::Device),
        0,
        "independent-reference cleanup must return all physical-byte accounting"
    );

    for steps in request_steps {
        for (inputs, ..) in steps {
            ep.deallocate(inputs.query).unwrap();
            ep.deallocate(inputs.current_key).unwrap();
            ep.deallocate(inputs.current_value).unwrap();
            ep.deallocate(inputs.seqlens).unwrap();
            ep.deallocate(inputs.total).unwrap();
        }
    }
}
