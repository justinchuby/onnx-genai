//! Does the KV-cache **layout** decide committed physical bytes at equal
//! content? — the residency headline for the binding-views work (#794 / #787).
//!
//! PR #794 converted flash prefill and decode to the shared stride descriptor
//! and then measured that head-major and seq-major commit **identical** physical
//! bytes, because the native CUDA binding committed a single flat bucket range
//! `0..(capacity × kv_heads × head_dim × elem)` regardless of layout: a flat
//! range from offset 0 maps the same contiguous granules whatever the axis order
//! means. The fix is to make the committed byte ranges follow the layout. This
//! test proves, at the CUDA driver level with no engine and no model, that on a
//! **fixed full-context stride** (the stable-VA regime that removes
//! growth-triggered re-capture) the layout alone moves committed bytes by the
//! `kv_heads` factor:
//!
//! * [`layout_decides_committed_bytes_at_the_near_empty_floor`] — for the
//!   qwen14b and qwen2.5-0.5b KV geometries, physically commit the granules each
//!   layout's live prefix touches for one live token per binding and read the
//!   committed bytes back. Head-major scatters one fragment per head stripe and
//!   pays `kv_heads` granules per binding; seq-major is one dense run and pays
//!   one. The ratio is `kv_heads×` (8× and 2×), matching the documented floors.
//! * [`seq_major_fixed_stride_grows_under_captured_replay_without_recapture`] —
//!   reserve a seq-major binding's full-context VA, commit the dense live
//!   prefix, capture a graph over it, then grow the live length by committing
//!   *tail* granules at the same VA and stride. The originally captured graph
//!   replays unchanged afterwards: a stable stride removes the growth-triggered
//!   re-capture that bucket growth forces (#778).
//!
//! Together these are the measured basis for "layout controls residency": the
//! reads are free (#787), the cost is on the binding layer, and moving the
//! committed geometry to follow the descriptor is what converts a correct
//! kernel-indexing change into an actual residency win.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;
// `test_support` is gated on `feature = "gpu-tests"` in the library (cfg(test)
// does not propagate to integration test crates). The import is conditional so
// this binary still compiles in the base (cuda-only) configuration where the
// test functions are #[ignore]d and their bodies are cfg'd out.
#[cfg(feature = "gpu-tests")]
use onnx_runtime_cuda_memory::test_support::TestStream;

const SENTINEL: u8 = 0x5a;

fn require_cuda() -> Arc<CudaContext> {
    match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "CUDA VMM test requires a CUDA driver; CPU-only runs must leave this test ignored: {error}"
        ),
    }
}

fn check(call: &'static str, result: cu::CUresult) {
    assert_eq!(result, cu::CUresult::CUDA_SUCCESS, "{call}: {result:?}");
}

fn allocation_prop(device_ordinal: i32) -> cu::CUmemAllocationProp {
    let mut prop: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_ordinal;
    prop
}

fn granularity(device_ordinal: i32) -> usize {
    let prop = allocation_prop(device_ordinal);
    let mut granularity = 0usize;
    let result = unsafe {
        cu::cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
    };
    check("cuMemGetAllocationGranularity", result);
    assert_ne!(granularity, 0, "CUDA reported zero VMM granularity");
    granularity
}

fn reserve(size: usize) -> cu::CUdeviceptr {
    let mut base = 0;
    let result = unsafe { cu::cuMemAddressReserve(&mut base, size, 0, 0, 0) };
    check("cuMemAddressReserve", result);
    base
}

fn free_reservation(base: cu::CUdeviceptr, size: usize) {
    if base != 0 {
        let _ = unsafe { cu::cuMemAddressFree(base, size) };
    }
}

fn create_handle(device_ordinal: i32, size: usize) -> cu::CUmemGenericAllocationHandle {
    let prop = allocation_prop(device_ordinal);
    let mut handle = 0;
    let result = unsafe { cu::cuMemCreate(&mut handle, size, &prop, 0) };
    check("cuMemCreate", result);
    handle
}

fn release_handle(handle: cu::CUmemGenericAllocationHandle) {
    let _ = unsafe { cu::cuMemRelease(handle) };
}

fn set_access(device_ordinal: i32, address: cu::CUdeviceptr, size: usize) {
    let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
    access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_ordinal;
    access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    let result = unsafe { cu::cuMemSetAccess(address, size, &access, 1) };
    check("cuMemSetAccess", result);
}

/// Map one physical granule at `address` and return its handle for teardown.
fn commit_granule(
    device_ordinal: i32,
    address: cu::CUdeviceptr,
    granule: usize,
) -> cu::CUmemGenericAllocationHandle {
    let handle = create_handle(device_ordinal, granule);
    let result = unsafe { cu::cuMemMap(address, granule, 0, handle, 0) };
    check("cuMemMap", result);
    set_access(device_ordinal, address, granule);
    handle
}

fn write_host(address: cu::CUdeviceptr, value: u8, len: usize) {
    let bytes = vec![value; len];
    let result = unsafe { cu::cuMemcpyHtoD_v2(address, bytes.as_ptr().cast(), bytes.len()) };
    check("cuMemcpyHtoD_v2", result);
}

fn read_host(address: cu::CUdeviceptr, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    let result = unsafe { cu::cuMemcpyDtoH_v2(bytes.as_mut_ptr().cast(), address, bytes.len()) };
    check("cuMemcpyDtoH_v2", result);
    bytes
}

/// A one-line summary of how a readback deviates from an all-`want` buffer: the
/// match count and the first mismatching offset/value. Distinguishes "the write
/// never became visible" (all bytes are the pre-write value) from "a partial or
/// torn write" (a prefix matches, then diverges) so a failure names its cause.
#[cfg(feature = "gpu-tests")]
fn mismatch_report(seen: &[u8], want: u8) -> String {
    let matched = seen.iter().filter(|&&b| b == want).count();
    let first = seen.iter().position(|&b| b != want);
    match first {
        None => format!(
            "{}/{} bytes == 0x{want:02x} (full match)",
            seen.len(),
            seen.len()
        ),
        Some(off) => format!(
            "{}/{} bytes == 0x{want:02x}; first mismatch at offset {off} = 0x{:02x}",
            matched,
            seen.len(),
            seen[off]
        ),
    }
}

/// One binding's per-token KV geometry (a single `(layer, side)` key or value
/// buffer).
#[derive(Clone, Copy)]
struct Geometry {
    kv_heads: usize,
    head_dim: usize,
    elem_bytes: usize,
    /// Full-context capacity (the fixed grow-axis stride).
    capacity: usize,
    /// Layer count × 2 (key + value): the number of independent bindings the
    /// whole model reserves. Used for the closed-form model-wide floor.
    bindings: usize,
}

impl Geometry {
    fn bytes_per_token(self) -> usize {
        self.kv_heads * self.head_dim * self.elem_bytes
    }
    fn bytes_per_token_per_head(self) -> usize {
        self.head_dim * self.elem_bytes
    }
    fn head_stride(self) -> usize {
        self.capacity * self.bytes_per_token_per_head()
    }
    /// Full binding byte size at the fixed full-context stride.
    fn full_binding_bytes(self) -> usize {
        self.capacity * self.bytes_per_token()
    }
}

/// The distinct granule indices a seq-major dense prefix of `valid_len` tokens
/// touches within one binding.
fn seq_major_granules(geo: Geometry, valid_len: usize, granule: usize) -> Vec<usize> {
    if valid_len == 0 {
        return Vec::new();
    }
    let bytes = valid_len * geo.bytes_per_token();
    let last = (bytes - 1) / granule;
    (0..=last).collect()
}

/// The distinct granule indices a head-major scattered prefix of `valid_len`
/// tokens touches within one binding (one fragment per head stripe).
fn head_major_granules(geo: Geometry, valid_len: usize, granule: usize) -> Vec<usize> {
    if valid_len == 0 {
        return Vec::new();
    }
    let live_width = valid_len * geo.bytes_per_token_per_head();
    let mut granules = std::collections::BTreeSet::new();
    for head in 0..geo.kv_heads {
        let start = head * geo.head_stride();
        let end = start + live_width;
        let first = start / granule;
        let last = (end - 1) / granule;
        for g in first..=last {
            granules.insert(g);
        }
    }
    granules.into_iter().collect()
}

/// Physically commit exactly the given granule indices inside a freshly reserved
/// full-context VA for one binding, write a sentinel into each, read it back
/// (proving the granule is really resident), and return the committed bytes.
/// Frees everything before returning.
fn commit_and_measure(
    device: i32,
    granule: usize,
    full_binding_bytes: usize,
    granule_indices: &[usize],
) -> usize {
    let reserved = full_binding_bytes.div_ceil(granule) * granule;
    let base = reserve(reserved);
    let mut handles = Vec::new();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        for &g in granule_indices {
            let address = base + (g * granule) as u64;
            handles.push(commit_granule(device, address, granule));
            write_host(address, SENTINEL, 64);
            let seen = read_host(address, 64);
            assert!(
                seen.iter().all(|&b| b == SENTINEL),
                "committed granule {g} must be physically resident"
            );
        }
        granule_indices.len() * granule
    }));
    for (i, handle) in handles.into_iter().enumerate() {
        let address = base + (granule_indices[i] * granule) as u64;
        let _ = unsafe { cu::cuMemUnmap(address, granule) };
        release_handle(handle);
    }
    free_reservation(base, reserved);
    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// A captured graph that memsets `len` bytes at `address` — the decode hot path
/// in miniature.
#[cfg(feature = "gpu-tests")]
struct CapturedMemset {
    graph: cu::CUgraph,
    exec: cu::CUgraphExec,
    len: usize,
}

#[cfg(feature = "gpu-tests")]
impl CapturedMemset {
    fn capture(stream: cu::CUstream, address: cu::CUdeviceptr, value: u8, len: usize) -> Self {
        let mut graph = std::ptr::null_mut();
        let mut exec = std::ptr::null_mut();
        let begin = unsafe {
            cu::cuStreamBeginCapture_v2(
                stream,
                cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        };
        check("cuStreamBeginCapture_v2", begin);
        let record = unsafe { cu::cuMemsetD8Async(address, value, len, stream) };
        let end = unsafe { cu::cuStreamEndCapture(stream, &mut graph) };
        check("cuMemsetD8Async during capture", record);
        check("cuStreamEndCapture", end);
        assert!(!graph.is_null(), "cuStreamEndCapture returned a null graph");
        let mut node_count = 0usize;
        check("cuGraphGetNodes", unsafe {
            cu::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut node_count)
        });
        assert_eq!(
            node_count, 1,
            "capture must record exactly the memset node, got {node_count} nodes"
        );
        let inst = unsafe { cu::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        check("cuGraphInstantiateWithFlags", inst);
        assert!(!exec.is_null(), "null graph exec");
        Self { graph, exec, len }
    }

    fn replay(&self, stream: cu::CUstream) {
        check("cuGraphLaunch", unsafe {
            cu::cuGraphLaunch(self.exec, stream)
        });
        check("cuStreamSynchronize after cuGraphLaunch", unsafe {
            cu::cuStreamSynchronize(stream)
        });
    }
}

#[cfg(feature = "gpu-tests")]
impl Drop for CapturedMemset {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            let _ = unsafe { cu::cuGraphExecDestroy(self.exec) };
        }
        if !self.graph.is_null() {
            let _ = unsafe { cu::cuGraphDestroy(self.graph) };
        }
        let _ = self.len;
    }
}

// qwen14b: 48 layers, key+value => 96 bindings; 8 kv heads; head_dim 128; f16;
// context 32768.
const QWEN14B: Geometry = Geometry {
    kv_heads: 8,
    head_dim: 128,
    elem_bytes: 2,
    capacity: 32_768,
    bindings: 96,
};
// qwen2.5-0.5b: 24 layers, key+value => 48 bindings; 2 kv heads; head_dim 64;
// f16; context 32768.
const QWEN05B: Geometry = Geometry {
    kv_heads: 2,
    head_dim: 64,
    elem_bytes: 2,
    capacity: 32_768,
    bindings: 48,
};

/// The headline: on a fixed full-context stride, layout alone sets committed
/// physical bytes. Physically prove one binding of each model under both
/// layouts at the near-empty floor (one live token), then report the closed-form
/// model-wide floors that #794 measured as identical and this design separates.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
#[allow(clippy::identity_op)] // `bindings * 1 * granule` mirrors the `* kv_heads *` shape above
fn layout_decides_committed_bytes_at_the_near_empty_floor() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);
    assert_eq!(
        granule,
        2 * 1024 * 1024,
        "test numbers assume a 2 MiB granule"
    );

    for (name, geo) in [("qwen2.5-0.5b", QWEN05B), ("qwen14b", QWEN14B)] {
        let valid = 1; // one live token: the near-empty floor

        let head_granules = head_major_granules(geo, valid, granule);
        let seq_granules = seq_major_granules(geo, valid, granule);

        // Physically commit and read back both layouts for one binding.
        let head_bytes =
            commit_and_measure(device, granule, geo.full_binding_bytes(), &head_granules);
        let seq_bytes =
            commit_and_measure(device, granule, geo.full_binding_bytes(), &seq_granules);

        // Head-major pays one granule per head; seq-major pays one, for one
        // token of identical content. The ratio is kv_heads.
        assert_eq!(
            head_granules.len(),
            geo.kv_heads,
            "{name}: head-major must open one granule per head stripe"
        );
        assert_eq!(
            seq_granules.len(),
            1,
            "{name}: seq-major must open exactly one dense granule"
        );
        assert_eq!(
            head_bytes / seq_bytes,
            geo.kv_heads,
            "{name}: committed-bytes ratio must equal kv_heads"
        );

        let model_head = geo.bindings * head_granules.len() * granule;
        let model_seq = geo.bindings * seq_granules.len() * granule;
        eprintln!(
            "{name} near-empty floor (fixed full-context stride, 1 live token):\n  \
             per-binding committed  head-major={} MiB  seq-major={} MiB  (ratio {}×)\n  \
             model-wide floor       head-major={} MiB ({} granules)  seq-major={} MiB ({} granules)",
            head_bytes / (1024 * 1024),
            seq_bytes / (1024 * 1024),
            head_bytes / seq_bytes,
            model_head / (1024 * 1024),
            geo.bindings * head_granules.len(),
            model_seq / (1024 * 1024),
            geo.bindings * seq_granules.len(),
        );
    }

    // The documented model-wide floors (closed form), on the record next to the
    // measured per-binding ratios above.
    assert_eq!(
        QWEN14B.bindings * QWEN14B.kv_heads * granule,
        1536 * 1024 * 1024
    );
    assert_eq!(QWEN14B.bindings * 1 * granule, 192 * 1024 * 1024);
    assert_eq!(
        QWEN05B.bindings * QWEN05B.kv_heads * granule,
        192 * 1024 * 1024
    );
    assert_eq!(QWEN05B.bindings * 1 * granule, 96 * 1024 * 1024);
}

/// Seq-major's second win: because the stride is fixed at full context, growing
/// the live length only maps *tail* granules at the same VA — it does not change
/// the stride, so a captured graph is not invalidated. Bucket growth, which
/// re-strides the packed buffer, forces a re-capture on every crossing (#778).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn seq_major_fixed_stride_grows_under_captured_replay_without_recapture() {
    // Body uses TestStream which is only available with gpu-tests. When the
    // feature is absent the test is #[ignore]d so this empty body is fine.
    #[cfg(feature = "gpu-tests")]
    {
        let context = require_cuda();
        context.bind_to_thread().expect("bind CUDA context");
        let device = 0;
        let granule = granularity(device);
        let geo = QWEN14B;

        // Reserve a modest full-context VA (8 granules) for one seq-major binding
        // and commit the dense prefix for a short live length.
        const RESERVED_GRANULES: usize = 8;
        let reserved = granule * RESERVED_GRANULES;
        let base = reserve(reserved);
        // Single-stream discipline (#797): every device operation below — the
        // baseline zero-fill, the captured memset, its replays, the readbacks, and
        // the tail write — flows through this one non-blocking stream, so a single
        // `sync` is a total order. Mixing in a default-stream copy here would
        // reintroduce the cold-context race this test was flaky on.
        let test_stream = TestStream::with_context(context.clone());
        let stream = test_stream.raw();

        // valid_len chosen so the dense prefix is under one granule (near-empty).
        let short_valid = 100usize;
        let short_bytes = short_valid * geo.bytes_per_token(); // 100 * 2048 = 200 KiB

        let mut handles = Vec::new();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            // Commit the first granule (holds the short dense prefix) and capture a
            // graph that writes the live prefix — the decode step in miniature.
            handles.push(commit_granule(device, base, granule));
            test_stream.fill(base, 0, short_bytes);
            let captured = CapturedMemset::capture(stream, base, SENTINEL, short_bytes);
            captured.replay(stream);
            let seen = test_stream.read(base, short_bytes);
            assert!(
                seen.iter().all(|&b| b == SENTINEL),
                "captured replay must fill the committed dense prefix: {}",
                mismatch_report(&seen, SENTINEL)
            );

            // Grow the live length past the first granule by committing tail
            // granules at the SAME VA and stride — no re-stride, no data move.
            let long_valid = 4000usize; // 4000 * 2048 = ~7.8 MiB → spans 4 granules
            let long_bytes = long_valid * geo.bytes_per_token();
            let last_granule = (long_bytes - 1) / granule;
            for g in 1..=last_granule {
                handles.push(commit_granule(device, base + (g * granule) as u64, granule));
            }

            // The ORIGINAL captured graph still replays correctly after growth: a
            // stable stride did not invalidate it. This is the property bucket
            // growth cannot offer.
            captured.replay(stream);
            let seen_after = test_stream.read(base, short_bytes);
            assert!(
                seen_after.iter().all(|&b| b == SENTINEL),
                "the pre-growth captured graph must replay unchanged after tail growth: {}",
                mismatch_report(&seen_after, SENTINEL)
            );

            // And the newly committed tail is usable by a fresh write, proving the
            // growth actually extended residency at the same VA.
            test_stream.fill(base + granule as u64, SENTINEL, 64);
            let tail = test_stream.read(base + granule as u64, 64);
            assert!(
                tail.iter().all(|&b| b == SENTINEL),
                "grown tail granule must be resident at the stable VA"
            );

            last_granule
        }));

        for (i, handle) in handles.into_iter().enumerate() {
            let _ = unsafe { cu::cuMemUnmap(base + (i * granule) as u64, granule) };
            release_handle(handle);
        }
        free_reservation(base, reserved);
        drop(test_stream);
        let last_granule = match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        };
        eprintln!(
            "seq-major fixed-stride growth: committed granule 0 then grew to granule {last_granule} \
             at a stable VA; the pre-growth captured graph replayed unchanged (0 re-captures)."
        );
    }
}
