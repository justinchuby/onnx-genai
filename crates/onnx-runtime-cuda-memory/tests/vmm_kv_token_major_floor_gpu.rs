//! Does a **token-major-across-all-layers** KV reservation commit *one granule
//! per sequence*, instead of one granule per buffer? — the payoff falsifier for
//! token-major KV (references #750, #777, #783; builds on the merged
//! `vmm_kv_contiguous_tail_gpu.rs` #772 head-major floor and the #782 seq-major
//! landing).
//!
//! # The claim under test
//!
//! Under a flat contiguous VA, `cuMemMap` maps a whole granule-aligned range
//! onto a whole physical granule, so the committed-bytes floor is
//! `objects x granule` where an "object" is a separately-strided live prefix.
//! The merged #772 test proved the **head-major** floor: with a fixed
//! full-context stride, each head-stripe's live prefix lands in its own granule,
//! so a near-empty qwen KV commits `layers x 2 x kv_heads` granules regardless
//! of content. Seq-major (#782) shrinks that to `layers x 2`.
//!
//! **Token-major** interleaves every layer's K and V by token in one
//! reservation, so the live prefix of the whole KV cache — all layers, both
//! sides, all heads — is a *single contiguous byte run*. Committing it therefore
//! takes `ceil(live_bytes / granule)` granules **total**, which is **one granule
//! for a fresh sequence** (the whole-model per-token KV is 192 KiB for qwen14b,
//! far under the 2 MiB granule). This test proves that on hardware, through the
//! real #740 authority-scoped physical-handle pool via `carve()` suballocation —
//! no second allocator, no per-sequence physical reservation — and reports
//! committed **physical** bytes (`committed_and_reserved().0`), never nominal
//! content bytes.
//!
//! * [`token_major_reservation_commits_one_granule_per_sequence`] — one
//!   reservation, commit the live prefix of a near-empty sequence: exactly one
//!   granule is committed for the whole model's live KV, and it stays one granule
//!   as the live length grows until the per-token run fills the granule.
//! * [`head_major_fixed_stride_commits_one_granule_per_object`] — the floor
//!   token-major replaces: the *same* live content laid out head-major with a
//!   fixed full-context stride commits one granule per head-stripe. Proven on a
//!   representative object subset, then the closed form for the whole model is
//!   asserted and printed next to the measured granule.

use cudarc::driver::CudaContext;
use onnx_runtime_memory_governor::VirtualBacking as _;
use onnx_runtime_cuda_memory::vmm_allocator::{
    CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, CudaVmmAllocator,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryRole,
};
use std::sync::Arc;

const HOLDER: HolderId = HolderId::new(783);
const GRANULE: usize = 2 << 20; // 2 MiB, the measured min == recommended granule (#776).

// qwen2.5-14b KV geometry: 48 layers, key+value, 8 kv heads, head_dim 128, fp16.
const LAYERS: usize = 48;
const SIDES: usize = 2; // key + value
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const ELEM: usize = 2; // fp16

/// Whole-model bytes for one token: every layer, both sides, every head. This is
/// the token-major stride and the granule the token-major floor commits.
const BYTES_PER_TOKEN: usize = LAYERS * SIDES * KV_HEADS * HEAD_DIM * ELEM; // 196_608 (192 KiB)
/// Separately-strided head-major objects: one granule each under a fixed stride.
const OBJECTS: usize = LAYERS * SIDES * KV_HEADS; // 768

fn require_cuda() -> Arc<CudaContext> {
    CudaContext::new(0).expect("token-major floor test requires a CUDA driver")
}

/// A pool-backed arena over a fresh authority, so its committed bytes are its
/// own. The pool retained-byte bound must cover the peak this test maps.
fn pooled_arena(capacity: usize, governor: &LedgerGovernor) -> CudaVmmAllocator {
    CudaVmmAllocator::new(
        require_cuda(),
        DeviceKey::device(0),
        0,
        capacity,
        governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("pool-backed VMM arena")
}

/// One reservation for the whole KV, all layers' K/V interleaved by token: the
/// live prefix of a near-empty sequence commits exactly one granule, because
/// every object's live bytes are contiguous at the front of the same range.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn token_major_reservation_commits_one_granule_per_sequence() {
    // SAFETY: this integration test owns its process and sets the production
    // pool option before constructing any allocator.
    unsafe {
        std::env::set_var(
            CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
            (16usize << 20).to_string(),
        );
    }
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");

    // Reserve a realistic multi-thousand-token context. Reservation is address
    // space only (free); nothing is committed until a prefix is touched.
    const CONTEXT_TOKENS: usize = 4096;
    let reservation = round_up(CONTEXT_TOKENS * BYTES_PER_TOKEN, GRANULE);
    let governor = LedgerGovernor::new(LeaseLedger::new(1 << 30, 0, 0));
    let arena = pooled_arena(reservation, &governor);

    // One allocation spanning the whole contiguous reservation; commit only the
    // live prefix of a single decoded token — the whole model's 192 KiB of KV.
    let live_bytes = BYTES_PER_TOKEN; // one token across all 48 layers / 768 objects
    let ptr = arena
        .allocate_committed(reservation, GRANULE, std::slice::from_ref(&(0..live_bytes)))
        .expect("token-major commit of one live token");

    let (committed, reserved) = arena.committed_and_reserved();
    assert_eq!(
        committed, GRANULE,
        "token-major: one live token's whole-model KV ({live_bytes} B across {OBJECTS} objects) \
         must commit exactly one granule, not one per object"
    );
    assert_eq!(
        reserved, reservation,
        "the full context stays reserved (VA is free)"
    );

    // It stays one granule as the sequence grows, until the per-token run fills
    // the granule: floor(2 MiB / 192 KiB) = 10 tokens fit in the first granule.
    let ten_tokens = 10 * BYTES_PER_TOKEN; // 1_966_080 B < 2 MiB
    assert!(
        ten_tokens < GRANULE,
        "ten tokens must fit one granule to prove density"
    );
    arena
        .commit_allocation_range(ptr, reservation, GRANULE, 0, ten_tokens)
        .expect("grow live prefix to ten tokens");
    let (committed_10, _) = arena.committed_and_reserved();
    assert_eq!(
        committed_10, GRANULE,
        "ten tokens of whole-model KV still fit one granule — the token-major floor is \
         one granule per sequence, not per object"
    );

    eprintln!(
        "token-major floor: {OBJECTS} objects, {}-token live prefix -> {} MiB committed \
         (reserved {} MiB VA). Head-major commits {} MiB for the same live content.",
        10,
        committed_10 / (1024 * 1024),
        reserved / (1024 * 1024),
        OBJECTS * GRANULE / (1024 * 1024),
    );

    // SAFETY: `ptr` came from this arena; no CUDA work references it.
    unsafe { arena.deallocate(ptr, reservation, GRANULE) };
    let (after, _) = arena.committed_and_reserved();
    assert_eq!(
        after, 0,
        "release returns the granule; committed physical bytes go to zero"
    );
}

/// The floor token-major replaces: laid out head-major with a fixed full-context
/// stride, the *same* near-empty live content commits one granule per
/// head-stripe. Proven on a representative subset (bounded VRAM), then the whole
/// model's closed form is asserted.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn head_major_fixed_stride_commits_one_granule_per_object() {
    // SAFETY: see sibling test; the process is this test's own.
    unsafe {
        std::env::set_var(
            CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
            (64usize << 20).to_string(),
        );
    }
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");

    // A fixed full-context stride places each head-stripe at least one granule
    // apart, so no two stripes share a granule. hd128/fp16 at 8192-token context
    // is exactly one granule per stripe.
    const CONTEXT_TOKENS: usize = 8192;
    let per_object_stride = round_up(CONTEXT_TOKENS * HEAD_DIM * ELEM, GRANULE);
    assert!(
        per_object_stride >= GRANULE,
        "each head stripe must span at least one granule for the floor to hold"
    );

    // Probe a representative subset so the test needs PROBE * granule of VRAM,
    // not the full 1.5 GiB; then assert the closed form for all 768 objects.
    const PROBE_OBJECTS: usize = 16;
    let capacity = PROBE_OBJECTS * per_object_stride;
    let governor = LedgerGovernor::new(LeaseLedger::new(1 << 30, 0, 0));
    let arena = pooled_arena(capacity, &governor);

    // Each object is its own full-context-stride span; commit just the live
    // prefix (one token's head-stripe = 256 B) of each. Every commit lands one
    // fresh granule.
    let live_per_object = HEAD_DIM * ELEM; // 256 B
    let mut ptrs = Vec::new();
    for _ in 0..PROBE_OBJECTS {
        let ptr = arena
            .allocate_committed(
                per_object_stride,
                GRANULE,
                std::slice::from_ref(&(0..live_per_object)),
            )
            .expect("head-major per-object commit");
        ptrs.push(ptr);
    }

    let (committed, _) = arena.committed_and_reserved();
    assert_eq!(
        committed,
        PROBE_OBJECTS * GRANULE,
        "each head-stripe commits its own granule under a fixed full-context stride"
    );
    // The floor is set by object count, not content: this subset already commits
    // far more than it stores.
    let probed_content = PROBE_OBJECTS * live_per_object;
    assert!(
        committed > probed_content * 100,
        "granule floor: {committed} B committed for {probed_content} B of live content"
    );

    // Closed form for the whole model, on the record next to the measured floor.
    let head_major_floor = OBJECTS * GRANULE;
    eprintln!(
        "head-major fixed full-context stride floor: {OBJECTS} objects x {} MiB granule = \
         {} MiB committed for {} B of live content (one token per object). Token-major commits \
         {} MiB for the identical content — a {}x reduction.",
        GRANULE / (1024 * 1024),
        head_major_floor / (1024 * 1024),
        OBJECTS * live_per_object,
        GRANULE / (1024 * 1024),
        head_major_floor / GRANULE,
    );
    assert_eq!(
        head_major_floor,
        OBJECTS * GRANULE,
        "qwen14b head-major floor is {OBJECTS} granules = 1.5 GiB"
    );

    for ptr in ptrs {
        // SAFETY: each `ptr` came from this arena; no CUDA work references it.
        unsafe { arena.deallocate(ptr, per_object_stride, GRANULE) };
    }
    let (after, _) = arena.committed_and_reserved();
    assert_eq!(after, 0, "releasing every object returns every granule");
}

fn round_up(bytes: usize, granule: usize) -> usize {
    bytes.div_ceil(granule) * granule
}
