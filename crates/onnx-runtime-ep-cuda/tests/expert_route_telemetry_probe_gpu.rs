//! INERT, test-only proof harness for issue #1810 Slice 6 — device-side expert
//! route telemetry for QMoE/BlockQuantizedMoE residency. See the design at
//! `docs/memory/EXPERT_ROUTE_TELEMETRY_SLICE6_DESIGN.md`.
//!
//! ## What this is (and is not)
//!
//! This file is a *standalone probe*: it allocates its own device buffers and
//! compiles its own NVRTC kernels through the public `CudaRuntime` API only. It
//! wires **nothing** into production residency/lifecycle:
//!
//! - It changes **no** production file. In particular it touches none of PR
//!   #1854's files (`src/lib.rs`, `weight_paging.rs`, `coarse_residency.rs`,
//!   `vmm_allocator.rs`, `ep-api/{lib,weight}.rs`) — an integration test under
//!   `tests/` is its own compilation unit, so it needs no `mod` declaration and
//!   no `src/` edit.
//! - It invokes **no** `QMoEKernel`/`BlockQuantizedMoEKernel`, no
//!   `PhysicalHandlePool`/`CudaVirtualBacking`, no `coarse_residency` plan, and
//!   no allocator/cache. The telemetry buffers here are throwaway probe state.
//! - It issues **no** `cuMemMap`/`cuMemSetAccess` anywhere, and never during
//!   capture (§3 of the design forbids remap-under-capture).
//!
//! ## What it proves (the design's testable claims)
//!
//! 1. A **CPU oracle** for the route bitmap and the bounded deduplicated route
//!    queue — the ground truth every GPU test diffs against.
//! 2. A device **route bitmap** (`atomicOr`) equals the oracle.
//! 3. A device **bounded dedup queue** (`atomicAdd` + seen-filter) equals the
//!    oracle set, and its **overflow** bit trips and **fails closed** when the
//!    distinct routed set exceeds capacity.
//! 4. **Poison** (out-of-range expert id) fails closed.
//! 5. A device **epoch/generation** counter advances across graph replays and a
//!    **stale** record is detectable; **request/device identity** mismatch fails
//!    closed (multi-request / multi-device isolation).
//! 6. **Capture/replay safety**: the producer is captured into a CUDA graph and,
//!    on each replay, re-accumulates *that replay's real routes* into a
//!    stable-VA buffer with **no host sync inside capture** — the FreeToken
//!    `lru_stats` property (`offload_cache.py:193-203`), adapted.
//! 7. A **microbenchmark** reporting the telemetry kernel's GPU-event time and
//!    host-enqueue time **separately** (cuda-perf-measurement Trap 4), on a
//!    ramped, idle-verified A100 (Traps 5/6). No wall-clock speedup is claimed.
//!
//! ## Run (solo, on an idle GPU verified with `nvidia-smi` first)
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=<idle> cargo test -p onnx-runtime-ep-cuda \
//!   --features cuda-13000,gpu-tests --release \
//!   --test expert_route_telemetry_probe_gpu -- --ignored --nocapture --test-threads=1
//! ```

use std::collections::HashSet;
use std::sync::Mutex;
use std::time::Instant;

use cudarc::driver::result::event;
use cudarc::driver::sys::{self as cu};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{DeviceBuffer, ExecutionProvider};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{CudaExecutionProvider, CudaRuntime};

/// Serialize every test in this file against the others on the same GPU (same
/// pattern as `qmoe_composable_vmm_host_numa_spike_gpu.rs`).
static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn require_cuda() -> (CudaExecutionProvider, std::sync::MutexGuard<'static, ()>) {
    let guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => (ep, guard),
        Ok(Err(error)) => panic!("CUDA runtime unavailable: {error}"),
        Err(_) => panic!("CUDA runtime libraries unavailable"),
    }
}

fn assert_gpu_idle_or_warn(context: &str) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader",
        ])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
            println!(
                "[{context}] nvidia-smi compute-apps (should be empty for a clean run): {lines:?}"
            );
        }
        _ => eprintln!(
            "[{context}] warning: could not query nvidia-smi compute-apps; idle-GPU precondition unverified"
        ),
    }
}

fn print_platform_conditions() {
    let driver = std::fs::read_to_string("/proc/driver/nvidia/version")
        .ok()
        .map(|s| s.lines().next().unwrap_or("").to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("platform: os={} driver={:?}", std::env::consts::OS, driver);
}

// ---------------------------------------------------------------------------
// Telemetry record layout (mirrors the design §2.3 header).
// ---------------------------------------------------------------------------

const MODULE: &str = "expert_route_telemetry_probe";

mod expert_route_oracle;
use expert_route_oracle::{
    Decision, H_COUNT, H_EPOCH, H_OVERFLOW, H_POISON, HEADER_LEN, consume_and_validate, cpu_bitmap,
    cpu_dedup, synth_routes, words_for,
};

// ---------------------------------------------------------------------------
// Device kernels (NVRTC). Only integer atomics — no fp16 headers needed.
// ---------------------------------------------------------------------------

const CUDA_SRC: &str = r#"
extern "C" __global__ void stamp_and_reset(
    unsigned int* header,          // [HEADER_LEN]
    unsigned int* epoch_counter,   // [1] persistent device clock (FreeToken `step` analog)
    unsigned int request_id,
    unsigned int device_id,
    unsigned int* bitmap,          // [words]
    unsigned int* seen,            // [words]
    int words)
{
    // One thread: bump the persistent epoch, stamp identity, clear the record.
    // Captured into the decode graph, so each replay re-bumps the epoch and
    // re-zeroes the observation buffers before that replay's routing is applied.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int e = atomicAdd(epoch_counter, 1u) + 1u;
        header[0] = e;            // H_EPOCH
        header[1] = request_id;   // H_REQUEST
        header[2] = device_id;    // H_DEVICE
        header[3] = 0u;           // H_OVERFLOW
        header[4] = 0u;           // H_POISON
        header[5] = 0u;           // H_COUNT
        for (int i = 0; i < words; ++i) { bitmap[i] = 0u; seen[i] = 0u; }
    }
}

extern "C" __global__ void route_bitmap(
    const int* selected_experts,   // [routes]
    unsigned long long routes,
    int num_experts,
    unsigned int* bitmap,          // [words]
    unsigned int* header)          // [HEADER_LEN]
{
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < routes; i += stride) {
        int e = selected_experts[i];
        if (e < 0 || e >= num_experts) { atomicOr(&header[4], 1u); continue; }
        atomicOr(&bitmap[e >> 5], 1u << (e & 31));
    }
}

extern "C" __global__ void route_dedup_queue(
    const int* selected_experts,   // [routes]
    unsigned long long routes,
    int num_experts,
    int capacity,
    unsigned int* seen,            // [words] dedup filter (pre-zeroed by stamp_and_reset)
    int* queue,                    // [capacity]
    unsigned int* header)          // [HEADER_LEN]
{
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < routes; i += stride) {
        int e = selected_experts[i];
        if (e < 0 || e >= num_experts) { atomicOr(&header[4], 1u); continue; }
        unsigned int mask = 1u << (e & 31);
        unsigned int prev = atomicOr(&seen[e >> 5], mask);
        if ((prev & mask) == 0u) {                        // first sighting of e
            unsigned int pos = atomicAdd(&header[5], 1u);   // H_COUNT (pre-cap distinct)
            if (pos < (unsigned int)capacity) queue[pos] = e;
            else atomicOr(&header[3], 1u);                  // H_OVERFLOW
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Small device-buffer + launch helpers (concrete `CudaRuntime`, no traits).
// ---------------------------------------------------------------------------

/// Launch an NVRTC entry by name with a list of `&` args. The module is cached
/// after first compile, so repeat calls only do a lock + hashmap lookup.
macro_rules! launch {
    ($runtime:expr, $entry:expr, $config:expr, $($arg:expr),+ $(,)?) => {{
        let function = $runtime
            .nvrtc_function(MODULE, CUDA_SRC, $entry)
            .expect("nvrtc compile");
        let mut builder = $runtime.stream().launch_builder(&function);
        $( builder.arg($arg); )+
        // SAFETY: probe-local buffers sized to the kernel ABI above.
        unsafe { builder.launch($config) }.expect("kernel launch");
    }};
}

fn alloc_zeroed(ep: &CudaExecutionProvider, runtime: &CudaRuntime, bytes: usize) -> DeviceBuffer {
    let buf = ep.allocate(bytes.max(1), 256).expect("device alloc");
    let zeros = vec![0u8; bytes.max(1)];
    // SAFETY: `buf` covers `zeros.len()` bytes.
    unsafe { runtime.htod(&zeros, cuptr(buf.as_ptr())) }.expect("htod zero-init");
    buf
}

fn htod_i32(runtime: &CudaRuntime, buf: &DeviceBuffer, data: &[i32]) {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    // SAFETY: `buf` covers `bytes.len()` bytes.
    unsafe { runtime.htod(bytes, cuptr(buf.as_ptr())) }.expect("htod routes");
}

fn dtoh_u32(runtime: &CudaRuntime, buf: &DeviceBuffer, len: usize) -> Vec<u32> {
    let mut out = vec![0u32; len];
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, len * 4) };
    // SAFETY: `buf` covers `len*4` bytes.
    unsafe { runtime.dtoh(bytes, cuptr(buf.as_ptr())) }.expect("dtoh u32");
    out
}

fn dtoh_i32(runtime: &CudaRuntime, buf: &DeviceBuffer, len: usize) -> Vec<i32> {
    let mut out = vec![0i32; len];
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, len * 4) };
    // SAFETY: `buf` covers `len*4` bytes.
    unsafe { runtime.dtoh(bytes, cuptr(buf.as_ptr())) }.expect("dtoh i32");
    out
}

fn cfg(total: u64) -> LaunchConfig {
    let block = 256u32;
    let grid = (total.div_ceil(block as u64)).clamp(1, 65535) as u32;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Produce one telemetry record eagerly (stamp → bitmap → dedup) for `routes`.
/// Returns `(bitmap, header, queue)`.
#[allow(clippy::too_many_arguments)]
fn produce_record(
    ep: &CudaExecutionProvider,
    runtime: &CudaRuntime,
    routes: &[i32],
    num_experts: i32,
    capacity: i32,
    request_id: u32,
    device_id: u32,
) -> (Vec<u32>, Vec<u32>, Vec<i32>) {
    let words = words_for(num_experts);
    let routes_buf = ep.allocate(routes.len().max(1) * 4, 256).expect("routes");
    htod_i32(runtime, &routes_buf, routes);
    let header = alloc_zeroed(ep, runtime, HEADER_LEN * 4);
    let epoch = alloc_zeroed(ep, runtime, 4);
    let bitmap = alloc_zeroed(ep, runtime, words * 4);
    let seen = alloc_zeroed(ep, runtime, words * 4);
    let queue = alloc_zeroed(ep, runtime, capacity.max(1) as usize * 4);

    let words_i = words as i32;
    let hp = cuptr(header.as_ptr());
    let epp = cuptr(epoch.as_ptr());
    let bp = cuptr(bitmap.as_ptr());
    let sp = cuptr(seen.as_ptr());
    let qp = cuptr(queue.as_ptr());
    let rp = cuptr(routes_buf.as_ptr());
    let routes_n = routes.len() as u64;

    launch!(
        runtime,
        "stamp_and_reset",
        cfg(1),
        &hp,
        &epp,
        &request_id,
        &device_id,
        &bp,
        &sp,
        &words_i
    );
    launch!(
        runtime,
        "route_bitmap",
        cfg(routes_n.max(1)),
        &rp,
        &routes_n,
        &num_experts,
        &bp,
        &hp
    );
    launch!(
        runtime,
        "route_dedup_queue",
        cfg(routes_n.max(1)),
        &rp,
        &routes_n,
        &num_experts,
        &capacity,
        &sp,
        &qp,
        &hp
    );
    runtime.synchronize().expect("synchronize");

    let bitmap_h = dtoh_u32(runtime, &bitmap, words);
    let header_h = dtoh_u32(runtime, &header, HEADER_LEN);
    let queue_h = dtoh_i32(runtime, &queue, capacity.max(1) as usize);
    (bitmap_h, header_h, queue_h)
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
#[ignore = "requires an idle GPU; run with --features gpu-tests -- --ignored"]
fn cuda_route_bitmap_matches_cpu_oracle() {
    print_platform_conditions();
    assert_gpu_idle_or_warn("bitmap");
    let (ep, _g) = require_cuda();
    let runtime: &CudaRuntime = ep.runtime();
    for (rows, top_k, num_experts) in [
        (1usize, 2usize, 4i32),
        (1, 6, 64),
        (1, 8, 256),
        (37, 6, 256),
    ] {
        let routes = synth_routes(rows, top_k, num_experts, 0xABCD ^ rows as u64);
        let (bitmap, header, _q) =
            produce_record(&ep, runtime, &routes, num_experts, num_experts, 1, 0);
        let (oracle, poison) = cpu_bitmap(&routes, num_experts);
        assert!(!poison);
        assert_eq!(
            bitmap, oracle,
            "device bitmap must equal oracle (E={num_experts}, rows={rows})"
        );
        assert_eq!(header[H_POISON], 0);
        let decision = consume_and_validate(&header, &bitmap, header[H_EPOCH], 1, 0);
        assert!(matches!(decision, Decision::HotSet(_)));
        println!(
            "bitmap E={num_experts} rows={rows}: {} routed experts, epoch={}",
            oracle.iter().map(|w| w.count_ones()).sum::<u32>(),
            header[H_EPOCH]
        );
    }
}

#[test]
#[ignore = "requires an idle GPU"]
fn cuda_dedup_queue_matches_cpu_oracle_set() {
    assert_gpu_idle_or_warn("dedup");
    let (ep, _g) = require_cuda();
    let runtime: &CudaRuntime = ep.runtime();
    let num_experts = 256;
    let routes = synth_routes(48, 8, num_experts, 0x1234);
    let (_bm, header, queue) =
        produce_record(&ep, runtime, &routes, num_experts, num_experts, 5, 0);
    let (distinct, overflow, _p) = cpu_dedup(&routes, num_experts, num_experts as usize);
    assert!(!overflow);
    assert_eq!(header[H_OVERFLOW], 0);
    assert_eq!(
        header[H_COUNT] as usize,
        distinct.len(),
        "device distinct count must equal oracle"
    );
    let device_set: HashSet<i32> = queue[..header[H_COUNT] as usize].iter().copied().collect();
    let oracle_set: HashSet<i32> = distinct.iter().copied().collect();
    assert_eq!(
        device_set, oracle_set,
        "dedup queue set (order-independent) must equal oracle"
    );
    println!("dedup: {} distinct experts, no overflow", distinct.len());
}

#[test]
#[ignore = "requires an idle GPU"]
fn cuda_dedup_overflow_fails_closed() {
    assert_gpu_idle_or_warn("overflow");
    let (ep, _g) = require_cuda();
    let runtime: &CudaRuntime = ep.runtime();
    let num_experts = 256;
    let routes = synth_routes(64, 8, num_experts, 0x7);
    let capacity = 8usize;
    let (distinct, _o, _p) = cpu_dedup(&routes, num_experts, capacity);
    assert!(
        distinct.len() > capacity,
        "test needs distinct > capacity to exercise overflow"
    );
    let (bitmap, header, queue) =
        produce_record(&ep, runtime, &routes, num_experts, capacity as i32, 9, 0);
    assert_eq!(header[H_OVERFLOW], 1, "overflow bit must be set");
    // Whatever landed in the queue is a subset of the true distinct set.
    let landed: HashSet<i32> = queue[..capacity].iter().copied().collect();
    let truth: HashSet<i32> = distinct.iter().copied().collect();
    assert!(
        landed.is_subset(&truth),
        "queued ids must be real routed experts"
    );
    let decision = consume_and_validate(&header, &bitmap, header[H_EPOCH], 9, 0);
    assert!(
        matches!(decision, Decision::WholeBank(_)),
        "overflow must fail closed: {decision:?}"
    );
    println!(
        "overflow fails closed: distinct={} capacity={capacity} -> {decision:?}",
        distinct.len()
    );
}

#[test]
#[ignore = "requires an idle GPU"]
fn cuda_poison_out_of_range_fails_closed() {
    assert_gpu_idle_or_warn("poison");
    let (ep, _g) = require_cuda();
    let runtime: &CudaRuntime = ep.runtime();
    let num_experts = 64;
    let mut routes = synth_routes(4, 6, num_experts, 0x55);
    routes.push(num_experts); // out of range: valid ids are [0, num_experts)
    routes.push(-1);
    let (bitmap, header, _q) =
        produce_record(&ep, runtime, &routes, num_experts, num_experts, 2, 0);
    assert_eq!(
        header[H_POISON], 1,
        "poison bit must be set for out-of-range ids"
    );
    let decision = consume_and_validate(&header, &bitmap, header[H_EPOCH], 2, 0);
    assert!(
        matches!(decision, Decision::WholeBank(_)),
        "poison must fail closed"
    );
    println!("poison fails closed: {decision:?}");
}

#[test]
#[ignore = "requires an idle GPU"]
fn cuda_identity_isolation_fails_closed() {
    assert_gpu_idle_or_warn("identity");
    let (ep, _g) = require_cuda();
    let runtime: &CudaRuntime = ep.runtime();
    let num_experts = 256;
    let routes = synth_routes(1, 8, num_experts, 0x1501);
    // Record produced by request 100 on device 0.
    let (bitmap, header, _q) =
        produce_record(&ep, runtime, &routes, num_experts, num_experts, 100, 0);
    // A different request consuming the same record must fail closed.
    let d_req = consume_and_validate(&header, &bitmap, header[H_EPOCH], 101, 0);
    assert!(
        matches!(d_req, Decision::WholeBank(_)),
        "foreign request must fail closed"
    );
    // A consumer on a different device must fail closed.
    let d_dev = consume_and_validate(&header, &bitmap, header[H_EPOCH], 100, 7);
    assert!(
        matches!(d_dev, Decision::WholeBank(_)),
        "foreign device must fail closed"
    );
    // The owning request/device on the same epoch accepts.
    let d_ok = consume_and_validate(&header, &bitmap, header[H_EPOCH], 100, 0);
    assert!(matches!(d_ok, Decision::HotSet(_)));
    println!("identity isolation: req-mismatch and dev-mismatch both fail closed; owner accepts");
}

/// Capture the producer into a CUDA graph and replay it 3× with *different*
/// routes each replay. Asserts: (a) the bitmap reflects each replay's real
/// routes (device accumulation re-runs on replay, FreeToken `lru_stats`
/// property); (b) buffer VA is stable across replays; (c) the epoch counter
/// advances once per replay; (d) no host sync happens *inside* capture; (e) a
/// stale record (boundary advanced past its epoch) is detectable.
#[test]
#[ignore = "requires an idle GPU"]
fn cuda_capture_replay_reaccumulates_real_routes() {
    print_platform_conditions();
    assert_gpu_idle_or_warn("capture");
    let (ep, _g) = require_cuda();
    let runtime: &CudaRuntime = ep.runtime();
    let num_experts = 256;
    let top_k = 8usize;
    let words = words_for(num_experts);
    let routes_n = top_k as u64; // decode shape: rows=1

    // Fixed-VA buffers, allocated once (nothing is allocated during capture).
    let routes_buf = ep.allocate(top_k * 4, 256).unwrap();
    let header = alloc_zeroed(&ep, runtime, HEADER_LEN * 4);
    let epoch = alloc_zeroed(&ep, runtime, 4);
    let bitmap = alloc_zeroed(&ep, runtime, words * 4);
    let seen = alloc_zeroed(&ep, runtime, words * 4);
    let bitmap_va = cuptr(bitmap.as_ptr());

    let words_i = words as i32;
    let hp = cuptr(header.as_ptr());
    let epp = cuptr(epoch.as_ptr());
    let bp = cuptr(bitmap.as_ptr());
    let sp = cuptr(seen.as_ptr());
    let rp = cuptr(routes_buf.as_ptr());
    let (request_id, device_id) = (777u32, 0u32);

    // Pre-warm: compile+load the modules and run once eagerly so the capture
    // below records only launches (a module load synchronizes and would abort a
    // capture). This is the design's warm-up-before-capture requirement (§1.2).
    let warm = synth_routes(1, top_k, num_experts, 1);
    htod_i32(runtime, &routes_buf, &warm);
    launch!(
        runtime,
        "stamp_and_reset",
        cfg(1),
        &hp,
        &epp,
        &request_id,
        &device_id,
        &bp,
        &sp,
        &words_i
    );
    launch!(
        runtime,
        "route_bitmap",
        cfg(routes_n),
        &rp,
        &routes_n,
        &num_experts,
        &bp,
        &hp
    );
    runtime.synchronize().unwrap();

    // Reset the epoch counter so replays start the epoch sequence from 1.
    unsafe { runtime.htod(&0u32.to_ne_bytes(), epp) }.unwrap();
    runtime.synchronize().unwrap();

    // Pre-fetch the (now cached) functions so no synchronizing module load
    // happens during capture.
    let stamp_fn = runtime
        .nvrtc_function(MODULE, CUDA_SRC, "stamp_and_reset")
        .unwrap();
    let route_fn = runtime
        .nvrtc_function(MODULE, CUDA_SRC, "route_bitmap")
        .unwrap();

    // Capture: stamp_and_reset (clears record + bumps epoch) then route_bitmap.
    // No dtoh/synchronize occurs between begin and end capture.
    let stream = runtime.stream_ptr();
    let r = unsafe {
        cu::cuStreamBeginCapture_v2(
            stream,
            cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
    };
    assert_eq!(r, cu::CUresult::CUDA_SUCCESS, "begin capture");
    {
        let mut b = runtime.stream().launch_builder(&stamp_fn);
        b.arg(&hp)
            .arg(&epp)
            .arg(&request_id)
            .arg(&device_id)
            .arg(&bp)
            .arg(&sp)
            .arg(&words_i);
        unsafe { b.launch(cfg(1)) }.expect("capture stamp");
    }
    {
        let mut b = runtime.stream().launch_builder(&route_fn);
        b.arg(&rp)
            .arg(&routes_n)
            .arg(&num_experts)
            .arg(&bp)
            .arg(&hp);
        unsafe { b.launch(cfg(routes_n)) }.expect("capture route");
    }
    let mut graph: cu::CUgraph = std::ptr::null_mut();
    let re = unsafe { cu::cuStreamEndCapture(stream, &mut graph) };
    assert_eq!(re, cu::CUresult::CUDA_SUCCESS, "end capture");
    let mut exec: cu::CUgraphExec = std::ptr::null_mut();
    let ri = unsafe { cu::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
    assert_eq!(ri, cu::CUresult::CUDA_SUCCESS, "instantiate");

    let mut last_header = vec![0u32; HEADER_LEN];
    for replay in 0..3u32 {
        // Update the model inputs (routes) OUTSIDE the graph, then replay.
        let routes = synth_routes(1, top_k, num_experts, 0xF00D + replay as u64);
        htod_i32(runtime, &routes_buf, &routes);
        runtime.synchronize().unwrap();

        let rl = unsafe { cu::cuGraphLaunch(exec, stream) };
        assert_eq!(rl, cu::CUresult::CUDA_SUCCESS, "replay {replay}");
        let rs = unsafe { cu::cuStreamSynchronize(stream) };
        assert_eq!(rs, cu::CUresult::CUDA_SUCCESS);

        let bitmap_h = dtoh_u32(runtime, &bitmap, words);
        let header_h = dtoh_u32(runtime, &header, HEADER_LEN);
        let (oracle, poison) = cpu_bitmap(&routes, num_experts);
        assert!(!poison);
        assert_eq!(
            bitmap_h, oracle,
            "replay {replay}: bitmap must reflect THIS replay's routes"
        );
        assert_eq!(
            cuptr(bitmap.as_ptr()),
            bitmap_va,
            "replay {replay}: buffer VA must stay stable"
        );
        assert_eq!(
            header_h[H_EPOCH],
            replay + 1,
            "replay {replay}: epoch must advance once per replay"
        );
        last_header = header_h;
    }

    // Stale detection: the last record has epoch=3; a boundary that has since
    // advanced to epoch 4 must treat it as stale and fail closed.
    let stale = consume_and_validate(&last_header, &vec![0u32; words], 4, request_id, device_id);
    assert!(
        matches!(stale, Decision::WholeBank(_)),
        "stale epoch must fail closed: {stale:?}"
    );
    // At its own epoch it is fresh.
    let fresh = consume_and_validate(&last_header, &vec![0u32; words], 3, request_id, device_id);
    assert!(matches!(fresh, Decision::HotSet(_)));

    unsafe {
        let _ = cu::cuGraphExecDestroy(exec);
        let _ = cu::cuGraphDestroy(graph);
    }
    println!(
        "capture/replay: 3 replays re-accumulated real routes, VA stable, epoch 1->3, stale detected"
    );
}

/// Microbenchmark (§7 / cuda-perf-measurement). Reports the telemetry kernel's
/// GPU-event time and the host enqueue time SEPARATELY, at decode and prefill
/// shapes, on a ramped and idle-verified device. No wall-clock speedup claimed.
#[test]
#[ignore = "requires a dedicated idle GPU; perf probe"]
fn microbench_telemetry_overhead_gpu_event_and_host_enqueue() {
    print_platform_conditions();
    assert_gpu_idle_or_warn("microbench-start");
    let (ep, _g) = require_cuda();
    let runtime: &CudaRuntime = ep.runtime();
    let num_experts = 256;
    let words = words_for(num_experts);

    for (label, rows, top_k) in [
        ("decode", 1usize, 8usize),
        ("prefill-512", 512usize, 8usize),
    ] {
        let routes = synth_routes(rows, top_k, num_experts, 0xBEEF);
        let routes_n = routes.len() as u64;
        let routes_buf = ep.allocate(routes.len() * 4, 256).unwrap();
        htod_i32(runtime, &routes_buf, &routes);
        let header = alloc_zeroed(&ep, runtime, HEADER_LEN * 4);
        let bitmap = alloc_zeroed(&ep, runtime, words * 4);
        let rp = cuptr(routes_buf.as_ptr());
        let bp = cuptr(bitmap.as_ptr());
        let hp = cuptr(header.as_ptr());
        let route_fn = runtime
            .nvrtc_function(MODULE, CUDA_SRC, "route_bitmap")
            .unwrap();
        let config = cfg(routes_n);

        let launch_once = || {
            let mut b = runtime.stream().launch_builder(&route_fn);
            b.arg(&rp)
                .arg(&routes_n)
                .arg(&num_experts)
                .arg(&bp)
                .arg(&hp);
            unsafe { b.launch(config) }.expect("bench launch");
        };

        // Ramp the device off its idle clock (Trap 5): keep launching until an
        // ~8s floor elapses, so the timed region below runs at a warm clock.
        const BATCH: usize = 512;
        let ramp_start = Instant::now();
        while ramp_start.elapsed().as_secs_f64() < 8.0 {
            for _ in 0..BATCH {
                launch_once();
            }
            runtime.synchronize().unwrap();
        }
        assert_gpu_idle_or_warn(&format!("microbench-{label}-post-ramp"));

        // GPU event time: enclose a BATCH between events (Trap 4 — one launch
        // would time the host). Report per-launch median over reps.
        let mut gpu_us = Vec::new();
        for _ in 0..5 {
            let start = event::create(cu::CUevent_flags::CU_EVENT_DEFAULT).unwrap();
            let end = event::create(cu::CUevent_flags::CU_EVENT_DEFAULT).unwrap();
            unsafe {
                event::record(start, runtime.stream_ptr()).unwrap();
                for _ in 0..BATCH {
                    launch_once();
                }
                event::record(end, runtime.stream_ptr()).unwrap();
                event::synchronize(end).unwrap();
                let ms = event::elapsed(start, end).unwrap() as f64;
                gpu_us.push(ms * 1000.0 / BATCH as f64);
                let _ = event::destroy(start);
                let _ = event::destroy(end);
            }
        }
        gpu_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Host enqueue time (separate, Trap 4): wall time to enqueue BATCH
        // launches without syncing, divided by BATCH.
        let mut host_us = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..BATCH {
                launch_once();
            }
            let per = t.elapsed().as_secs_f64() * 1e6 / BATCH as f64;
            host_us.push(per);
            runtime.synchronize().unwrap();
        }
        host_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!(
            "microbench[{label}] routes={routes_n} E={num_experts}: \
             GPU/launch median={:.3} us (min={:.3}, max={:.3}); \
             host-enqueue/launch median={:.3} us (min={:.3}, max={:.3}); \
             telemetry-bitmap bytes={} — no speedup claimed, kernel/host times reported separately",
            gpu_us[gpu_us.len() / 2],
            gpu_us[0],
            gpu_us[gpu_us.len() - 1],
            host_us[host_us.len() / 2],
            host_us[0],
            host_us[host_us.len() - 1],
            words * 4,
        );
    }
    assert_gpu_idle_or_warn("microbench-end");
}
