#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::uninlined_format_args,
    clippy::type_complexity,
    clippy::manual_is_multiple_of
)]
//! A100 measurement for the `BlockQuantizedMoE` prefill ahead-of-need prefetch
//! (issue #82 cycle 7, `weight_paging.rs::prefetch_block_quantized_moe` /
//! `promote_pending_prefetch`).
//!
//! Uses the exact "repeat-launch + CUDA-event/single-sync" measurement
//! primitive this repo already established for `qmoe_gpu.rs`'s
//! `qmoe_expert_gemv_bandwidth_probe` (PR #1777): the promotion's
//! `resolve_prefetch_fence` **is** that single-sync host-blocking event wait
//! (see `runtime.rs::resolve_prefetch_fence`), so its
//! `CudaResidencyStats::prefetch_promote_wait_ns` field is a direct,
//! already-instrumented measurement of "how much of the transfer leaked past
//! the intervening compute" -- not a second ad-hoc timer.
//!
//! # Shapes (measurement-discipline: no invented dimensions)
//!
//! Same two real model configs already cited in `qmoe_gpu.rs` (duplicated
//! here, not imported -- each `tests/*.rs` file is a separate compilation
//! unit and `qmoe_gpu.rs`'s constants are private to it):
//!   - `hidden_size=2048`, `moe_intermediate_size=1408`, `n_routed_experts=64`
//!     -- `huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/config.json`,
//!     fetched 2026-08-22.
//!   - `hidden_size=6144`, `moe_intermediate_size=2048`, `n_routed_experts=256`
//!     -- `huggingface.co/zai-org/GLM-5.2/raw/main/config.json`, fetched
//!     2026-08-22.
//!
//! The per-layer packed byte volume is computed from this repo's own mxfp4
//! `BlockQuantizedMoE` packing scheme (`QK=32` elements/block, `BLOCK_BYTES=17`
//! bytes/block: 16 packed 4-bit values + 1 scale byte), matching
//! `block_quantized_moe_gpu.rs::pack_projection`/`Config::fc1_size` exactly
//! (fused gate+up `fc1`, `fc2` down-projection; this repo's existing
//! quantization-scheme convention for these two models, not a `config.json`
//! field either upstream config names -- same caveat `qmoe_gpu.rs::
//! moe_bench_case` already states for QMoE's bits/block_size). This is the
//! REAL full-model per-layer byte volume (294 MiB for DeepSeek-V2-Lite, 4.78
//! GiB for GLM-5.2) -- not a toy/reduced size; only the *number of layers*
//! simulated (2) and the weight *content* (a fast-filled repeating byte
//! pattern, not real quantized values) are reduced for tractability, exactly
//! as `qmoe_expert_gemv_bandwidth_probe` reduces the correctness path's expert
//! count but keeps the bandwidth path's expert count real.
//!
//! # What this does NOT claim
//!
//! No end-to-end tok/s number. This measures the paging/prefetch mechanism in
//! isolation: transfer bytes, page-in accounting, and the residual
//! (non-overlapped) transfer time under one fixed compute-proxy kernel.
//!
//! # Truthful no-win: both real models decline via the pool-capacity guard
//!
//! Building this probe found a real soundness bug in the prefetch path (now
//! fixed, same commit): it bypassed the shared `PinnedStagingPool` entirely,
//! recreating issue #837's per-page-in `cuMemHostAlloc`/`cuMemFreeHost` cost.
//! The fix adds `PinnedStagingPool::can_retain_concurrent`, which
//! `prefetch_block_quantized_moe` now consults up front: the executor's real
//! `prefetch_lazy_weights_after(pi)` (issue the *next* boundary's prefetch)
//! before `exec_plan_node(pi)` (promote *this* boundary's own already-issued
//! prefetch) ordering means steady state genuinely needs **two**
//! concurrently-live pinned buffers of the boundary's byte size. Under the
//! pool's default 512 MiB retention cap, `2 * layer_bytes` exceeds that cap
//! for **both** cited real models (DeepSeek-V2-Lite: 2 * 281 MiB = 562 MiB;
//! GLM-5.2: a single 4.78 GiB layer already exceeds it alone) -- so this probe
//! reports, honestly, that the ON arm **declines** for both real-model rows
//! (`prefetch_declined_pool_capacity` fires, `on_us` collapses to the same
//! synchronous per-demand page-in `resident_mapped` already does in the OFF
//! arm) rather than asserting an overlap win that cannot occur under today's
//! pool sizing. This is a truthful no-win, not a benchmark bug, and it is the
//! primary finding of this probe for the two cited models.
//!
//! A third, clearly-separated **synthetic-capacity-sufficient** row (128 MiB
//! per layer -- comfortably under half the pool's 512 MiB cap, NOT tied to
//! any real model shape) proves the double-buffer overlap mechanism itself
//! works correctly in principle once its prerequisite (adequate pool
//! capacity) is met. Keeping it out of the DeepSeek/GLM rows prevents it from
//! being mistaken for a claim about either cited model.
//!
//! # A second finding: the wall-clock cost of a cold page-in is materialize-
//! dominated, not DMA-bound -- and the pool-capacity decline makes every
//! real-model page-in pay it
//!
//! `weight_paging.rs` already instruments the on-demand page-in path with
//! three separate global counters this probe reads via
//! `global_offload_stats()` (`materialize_ns`/`htod_ns`/`vram_alloc_ns` --
//! not new instrumentation): the CPU-side `copy_from_slice` from the mmap
//! source into the pinned staging buffer, the measured host->device DMA
//! itself, and the `cuMemAlloc` for the destination VRAM page. Reading them
//! for the isolated single-page-in measurement below shows the *real* H2D DMA
//! consistently lands at ~25 GB/s (~79% of this box's confirmed PCIe Gen4 x16
//! link -- `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`
//! reports `4, 16` -- so [`PCIE4_X16_PEAK_GBPS`] is the correct reference, not
//! the HBM peak an earlier revision of this probe mistakenly divided by,
//! which produced a nonsensical "0.04% of peak" reading for a healthy
//! transfer). The DMA is not the bottleneck.
//!
//! The dominant wall-clock cost is `materialize_ns`: for a FIRST-TIME
//! `staging_pool.acquire(len)` of hundreds of MiB to several GiB, writing into
//! the freshly `cuMemHostAlloc`'d pinned buffer forces the OS/driver to
//! populate and lock those physical pages right then, and this repeats on
//! *every single page-in* for DeepSeek-V2-Lite and GLM-5.2 specifically
//! *because* the pool-capacity guard (previous section) means their staging
//! buffer is never retained for reuse -- the pool always drops it on release,
//! so every page-in for these two shapes is a cold one. This is the direct,
//! previously-invisible cost of the pool-capacity gap: it is not merely "no
//! overlap benefit", it is "every real-model page-in -- prefetch or ordinary
//! on-demand -- pays a full fresh pin every time", strengthening (not
//! weakening) the case for the next-gate recommendation in the final report
//! to enlarge the pool's retention bounds for boundary-scale buffers.
//!
//! # Why every case here uses `layers: 2`, not more (a rejected extension)
//!
//! An earlier revision of this probe tried extending the synthetic row to 8
//! layers to observe amortized steady-state buffer reuse instead of only the
//! mechanism's cold-start transient. That revision failed its own assertion:
//! from the third layer onward, every ON-arm prefetch attempt was declined
//! with `declined_busy`, not issued.
//!
//! The root cause is a real property of the single-pending-slot design
//! applied to *this probe's* loop shape, not a regression from this cycle's
//! change. The production call order in
//! `executor/dispatch.rs::run_plan_eager` is, for each plan node `pi` in turn:
//! `prefetch_lazy_weights_after(pi)` (issues a prefetch for node `pi +
//! lookahead`'s lazy weight) immediately followed by `exec_plan_node(pi)`
//! (which is what promotes node `pi`'s *own* prefetch, issued back at turn
//! `pi - lookahead`). With `lookahead == 1` (the default), this only avoids a
//! same-turn self-conflict in production because a real compiled plan
//! interleaves many non-lazy-weight nodes (attention, norm, etc.) between one
//! `BlockQuantizedMoe` boundary node and the next, so the previous boundary's
//! prefetch has already been promoted -- freeing the single pending slot --
//! long before the next boundary's issue attempt runs.
//!
//! A tight loop that chains `issue(layer + 1)` directly followed by
//! `promote(layer)` with *nothing* in between, for `layers >= 3`, does not
//! reproduce that spacing: at the start of iteration `layer`, slot occupancy
//! still holds `layer`'s own not-yet-promoted prefetch (issued last
//! iteration), so the new issue for `layer + 1` finds the slot busy and
//! declines every time. That is an artifact of this probe's zero-intervening-
//! work loop shape, not evidence about the pool-capacity fix under test here
//! -- so rather than manufacture a steady-state number from an unfaithful
//! loop, every case in this probe is deliberately kept at `layers: 2` (the
//! shape that cannot hit this self-conflict, since only one prefetch is ever
//! issued) and this limitation is reported instead.
//!
//! Faithfully modeling the real steady-state timing would require knowing the
//! actual plan-node distance between one `BlockQuantizedMoe` boundary and the
//! next in a loaded DeepSeek-V2-Lite / GLM-5.2 execution plan -- a property of
//! a compiled model this synthetic probe cannot source without one. That
//! distance is the next research input this line of work needs.

use std::time::{Duration, Instant};

use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, ExternalMmapRegion, LazyWeight, MmapRegionSource,
    ResidentWeight, TensorMut, TensorView, WeightHandleError,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{DataType, DeviceId, Node, NodeId, compute_contiguous_strides};

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

/// This box's confirmed PCIe Gen4 x16 host link (`nvidia-smi
/// --query-gpu=pcie.link.gen.current,pcie.link.width.current` reports `4,
/// 16`), giving the standard 16 GT/s x16 unidirectional theoretical peak.
/// This -- not the A100's ~2 TB/s HBM2e peak `qmoe_gpu.rs`/`matmul_nbits.rs`
/// cite for *device-internal* compute-bandwidth probes -- is the correct
/// reference for a host->device transfer: an H2D copy is PCIe-bound, never
/// HBM-bound. See the module doc's second finding.
const PCIE4_X16_PEAK_GBPS: f64 = 31.5;

/// This repo's mxfp4 `BlockQuantizedMoE` block-quantization packing, matching
/// `block_quantized_moe_gpu.rs`'s `QK`/`BLOCK_BYTES` exactly.
const QK: usize = 32;
const BLOCK_BYTES: usize = 17;

#[derive(Clone, Copy, Debug)]
struct MoeModelShape {
    name: &'static str,
    hidden: usize,
    inter: usize,
    experts: usize,
}

/// See module doc for the citation. Matches `qmoe_gpu.rs::DEEPSEEK_V2_LITE_MOE`.
const DEEPSEEK_V2_LITE_MOE: MoeModelShape = MoeModelShape {
    name: "deepseek-v2-lite",
    hidden: 2048,
    inter: 1408,
    experts: 64,
};

/// See module doc for the citation. Matches `qmoe_gpu.rs::GLM_5_2_MOE`.
const GLM_5_2_MOE: MoeModelShape = MoeModelShape {
    name: "glm-5.2",
    hidden: 6144,
    inter: 2048,
    experts: 256,
};

/// Bytes for one layer's fused `fc1` (gate+up, `swiglu_fusion`) + `fc2` (down)
/// packed `BlockQuantizedMoE` weight bank -- the real full-model per-layer
/// paging unit for `shape`, not a reduced/toy size. See module doc.
fn bqmoe_layer_bytes(shape: MoeModelShape) -> u64 {
    let fc1_out = shape.inter * 2;
    let fc1_blocks = shape.hidden.div_ceil(QK);
    let fc1_bytes = shape.experts * fc1_out * fc1_blocks * BLOCK_BYTES;
    let fc2_blocks = shape.inter.div_ceil(QK);
    let fc2_bytes = shape.experts * shape.hidden * fc2_blocks * BLOCK_BYTES;
    (fc1_bytes + fc2_bytes) as u64
}

/// A host buffer standing in for an ONNX external-data mmap, holding
/// `layers` distinct fast-filled (not really quantized -- content is
/// irrelevant to a paging/bandwidth measurement) regions of `layer_bytes`
/// each, back-to-back after a padding prefix (proves offset handling).
struct LayeredMmap {
    mapping_id: usize,
    bytes: Vec<u8>,
}

impl MmapRegionSource for LayeredMmap {
    fn region_bytes(&self, region: &ExternalMmapRegion) -> Result<&[u8], WeightHandleError> {
        if region.mapping_id != self.mapping_id {
            return Err(WeightHandleError::DeviceBinding(format!(
                "unknown mapping {}",
                region.mapping_id
            )));
        }
        let end = region
            .offset
            .checked_add(region.len)
            .ok_or_else(|| WeightHandleError::DeviceBinding("region overflow".into()))?;
        self.bytes
            .get(region.offset..end)
            .ok_or_else(|| WeightHandleError::DeviceBinding("region out of bounds".into()))
    }
}

/// Build `layers` `BlockQuantizedMoE`-boundary lazy weights of `layer_bytes`
/// each, plus the host mmap backing them. Each layer's bytes are a distinct
/// repeating pattern so a byte-identity check after paging is meaningful (not
/// all-zero, which would pass trivially even with a short/misordered copy as
/// long as it stayed within the zero-filled region). `layer_bytes` is either
/// the real full-model per-layer byte volume (from [`bqmoe_layer_bytes`]) or
/// the module doc's synthetic-capacity-sufficient size -- never invented per
/// call site.
fn layered_weights(layer_bytes: u64, layers: usize) -> (LayeredMmap, Vec<LazyWeight>) {
    let layer_bytes = layer_bytes as usize;
    let mapping_id = 42;
    let prefix = 4096usize;
    let mut backing = vec![0xABu8; prefix];
    let mut lazies = Vec::with_capacity(layers);
    for layer in 0..layers {
        let offset = backing.len();
        let pattern = 0x10u8.wrapping_add(layer as u8);
        backing.resize(offset + layer_bytes, pattern);
        let region = ExternalMmapRegion {
            mapping_id,
            offset,
            len: layer_bytes,
        };
        let dtype_shape = vec![layer_bytes];
        let lazy =
            LazyWeight::block_quantized_moe(DataType::Uint8, dtype_shape.clone(), vec![region], {
                move || {
                    ResidentWeight::new(
                        DataType::Uint8,
                        dtype_shape.clone(),
                        vec![pattern; layer_bytes],
                    )
                    .map(onnx_runtime_ep_api::ResidentWeightMaterialization::reused)
                }
            })
            .unwrap();
        lazies.push(lazy);
    }
    (
        LayeredMmap {
            mapping_id,
            bytes: backing,
        },
        lazies,
    )
}

/// A fixed-size dense fp32 matmul standing in for "one layer's other real
/// compute" -- calibrated once (its own measured duration is reported, not
/// assumed) rather than tuned per model shape to manufacture an apparent
/// overlap win. Built once outside the timed region; only `execute` runs
/// inside it.
struct ComputeProxy {
    a: onnx_runtime_ep_api::DeviceBuffer,
    b: onnx_runtime_ep_api::DeviceBuffer,
    c: onnx_runtime_ep_api::DeviceBuffer,
    kernel: Box<dyn onnx_runtime_ep_api::Kernel>,
    m: usize,
    k: usize,
    n: usize,
}

impl ComputeProxy {
    fn new(ep: &CudaExecutionProvider) -> Self {
        let (m, k, n) = (2048usize, 4096usize, 4096usize);
        let a = ep.allocate(m * k * 4, 256).unwrap();
        let b = ep.allocate(k * n * 4, 256).unwrap();
        let c = ep.allocate(m * n * 4, 256).unwrap();
        // Content does not matter for a compute-proxy timing kernel; zeroed
        // device memory is a valid dense fp32 operand.
        let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
        let kernel = ep.get_kernel(&node, &[vec![m, k], vec![k, n]], 17).unwrap();
        Self {
            a,
            b,
            c,
            kernel,
            m,
            k,
            n,
        }
    }

    fn run(&mut self, ep: &CudaExecutionProvider) {
        let dev: DeviceId = ep.device_id();
        let a_shape = [self.m, self.k];
        let b_shape = [self.k, self.n];
        let out_shape = [self.m, self.n];
        let a_strides = compute_contiguous_strides(&a_shape);
        let b_strides = compute_contiguous_strides(&b_shape);
        let out_strides = compute_contiguous_strides(&out_shape);
        let a_view = TensorView::new(
            DevicePtr(self.a.as_ptr()),
            DataType::Float32,
            &a_shape,
            &a_strides,
            dev,
        );
        let b_view = TensorView::new(
            DevicePtr(self.b.as_ptr()),
            DataType::Float32,
            &b_shape,
            &b_strides,
            dev,
        );
        let out_view = TensorMut::new(
            DevicePtrMut(self.c.as_mut_ptr()),
            DataType::Float32,
            &out_shape,
            &out_strides,
            dev,
        );
        self.kernel
            .execute(&[a_view, b_view], &mut [out_view])
            .unwrap();
    }

    fn teardown(self, ep: &CudaExecutionProvider) {
        ep.deallocate(self.a).unwrap();
        ep.deallocate(self.b).unwrap();
        ep.deallocate(self.c).unwrap();
    }
}

/// Median of `Duration`s (odd count expected; falls back to the lower-middle
/// element on an even count).
fn median_duration(mut samples: Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

/// One row of the benchmark matrix. `expect_pool_capacity_decline` records
/// this probe's *a priori* expectation under today's
/// `PinnedStagingPool::DEFAULT_MAX_BYTES` (512 MiB) so a future change to
/// either the pool's bounds or a cited model's shape is caught as a test
/// failure here (an assertion that flips is a signal to update this
/// expectation and the module doc, not silently drift). See the module doc's
/// "Truthful no-win" section for the full argument.
struct BenchCase {
    name: &'static str,
    layer_bytes: u64,
    expect_pool_capacity_decline: bool,
    /// Number of layers to simulate for this case. Real-model rows use the
    /// module doc's fixed `2` (enough to exercise exactly one look-ahead
    /// prefetch attempt without provisioning excessive host/VRAM budget for
    /// GLM-5.2's multi-GiB layers). The synthetic row uses more (see its call
    /// site) to distinguish a one-time cold-provisioning cost (this
    /// mechanism's first cycle always needs 2 concurrently-live pinned
    /// buffers -- module doc's second finding) from the steady-state
    /// overlap the mechanism is meant to provide once both buffers exist.
    layers: usize,
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn bqmoe_prefill_double_buffer_overlap_probe() {
    let ep = require_cuda();
    let runtime = ep.runtime().clone();
    let runs = 3usize;

    // Calibrate the compute proxy's own duration once, in isolation (no
    // concurrent transfer), via repeated launches + a single trailing sync --
    // this repo's established cheap-probe pattern.
    let mut proxy = ComputeProxy::new(&ep);
    for _ in 0..3 {
        proxy.run(&ep); // warm up: first launches pay JIT/cache costs.
    }
    runtime.synchronize().unwrap();
    let calib_reps = 20u32;
    let calib_start = Instant::now();
    for _ in 0..calib_reps {
        proxy.run(&ep);
    }
    runtime.synchronize().unwrap();
    let compute_us = calib_start.elapsed().as_secs_f64() * 1e6 / f64::from(calib_reps);
    println!(
        "compute proxy calibration: {}x{}x{} fp32 matmul, {:.1} us/launch (median-free single-block average over {} reps)",
        proxy.m, proxy.k, proxy.n, compute_us, calib_reps
    );

    println!(
        "{:<38} {:>10} {:>10} {:>10} {:>12} {:>10} {:>10} {:>12} {:>12}",
        "case",
        "layer_MiB",
        "off_us",
        "on_us",
        "promote_wait_us",
        "htod_GBps",
        "%pcie_peak",
        "materialize_us",
        "vram_alloc_us"
    );

    // Real-model rows are expected, TODAY (`PinnedStagingPool::DEFAULT_MAX_BYTES`
    // = 512 MiB), to decline via the pool-capacity guard -- see module doc. The
    // synthetic row is sized to comfortably satisfy `can_retain_concurrent(len,
    // 2)` and proves the double-buffer mechanism itself works when its
    // prerequisite is met.
    let cases = [
        BenchCase {
            name: DEEPSEEK_V2_LITE_MOE.name,
            layer_bytes: bqmoe_layer_bytes(DEEPSEEK_V2_LITE_MOE),
            expect_pool_capacity_decline: true,
            layers: 2,
        },
        BenchCase {
            name: GLM_5_2_MOE.name,
            layer_bytes: bqmoe_layer_bytes(GLM_5_2_MOE),
            expect_pool_capacity_decline: true,
            layers: 2,
        },
        BenchCase {
            name: "synthetic-capacity-sufficient-128MiB",
            // 128 MiB: comfortably under half the pool's default 512 MiB
            // retention cap, so 2 concurrent buffers of this size fit. NOT a
            // real model shape -- proves the mechanism, not a DeepSeek/GLM claim.
            // `layers=2` deliberately, not more -- see the module doc's third
            // finding ("why not more layers") for why a naive >2-layer
            // extension of this tight loop is NOT a faithful steady-state
            // probe and was rejected rather than used to manufacture a number.
            layer_bytes: 128 * 1024 * 1024,
            expect_pool_capacity_decline: false,
            layers: 2,
        },
    ];

    for case in cases {
        let layer_bytes = case.layer_bytes;
        let layers = case.layers;
        let (host, lazies) = layered_weights(layer_bytes, layers);
        let budget = layer_bytes * layers as u64;

        // --- Pure transfer bandwidth, isolated from any concurrent compute:
        // one page-in with nothing else running on the compute stream. ---
        let bw_residency = ep.weight_residency(budget);
        onnx_runtime_ep_cuda::reset_global_offload_stats();
        let bw_start = Instant::now();
        let page = bw_residency
            .resident_mapped(0, &lazies[0], &host)
            .expect("isolated bandwidth page-in must succeed");
        let bw_wall_us = bw_start.elapsed().as_secs_f64() * 1e6;
        // Break the cold page-in's wall time down using this repo's existing
        // per-page-in instrumentation (`weight_paging.rs`'s
        // `materialize_ns`/`htod_ns`/`vram_alloc_ns`, not new counters here) --
        // see the module doc's second finding. `htod_gbps`/`htod_pct_peak` is
        // the real DMA-only bandwidth against the correct PCIe reference;
        // `bw_wall_us` also includes the one-time pinned-buffer materialize
        // (CPU copy + first-touch pin) and VRAM alloc costs, which dominate
        // wall time for a shape too large for the pool to retain.
        let diag = onnx_runtime_ep_cuda::global_offload_stats();
        let htod_gbps = (diag.htod_bytes as f64 / 1e9) / (diag.htod_ns as f64 / 1e9).max(1e-12);
        let htod_pct_peak = 100.0 * htod_gbps / PCIE4_X16_PEAK_GBPS;
        let materialize_us = diag.materialize_ns as f64 / 1e3;
        let vram_alloc_us = diag.vram_alloc_ns as f64 / 1e3;
        drop(page);
        drop(bw_residency);

        // --- Correctness gate: dtoh the paged bytes back and compare against
        // the canonical source. A fast wrong copy must not pass. ---
        let verify_residency = ep.weight_residency(budget);
        let page0 = verify_residency
            .resident_mapped(0, &lazies[0], &host)
            .unwrap();
        let mut readback = vec![0u8; layer_bytes as usize];
        // SAFETY: `readback` is sized to the page's exact byte length.
        unsafe {
            runtime
                .dtoh(&mut readback, cuptr(page0.device_ptr()))
                .unwrap()
        };
        let canonical = host
            .region_bytes(&ExternalMmapRegion {
                mapping_id: host.mapping_id,
                offset: 4096, // layer 0's region starts right after the padding prefix.
                len: layer_bytes as usize,
            })
            .unwrap();
        assert_eq!(
            readback, canonical,
            "{}: paged layer 0 bytes must be byte-identical to the source",
            case.name
        );
        drop(page0);
        drop(verify_residency);

        // --- OFF arm: synchronous page-in, no prefetch. ---
        let mut off_samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let residency = ep.weight_residency(budget);
            let start = Instant::now();
            for layer in 0..layers {
                let page = residency
                    .resident_mapped(layer as u64, &lazies[layer], &host)
                    .expect("OFF arm page-in must succeed");
                drop(page);
                proxy.run(&ep);
            }
            runtime.synchronize().unwrap();
            off_samples.push(start.elapsed());
        }
        let off_us = median_duration(off_samples).as_secs_f64() * 1e6;

        // --- ON arm: issue a look-ahead prefetch for layer+1 before using
        // layer, mirroring the executor's real `prefetch_lazy_weights_after(pi)`
        // /`exec_plan_node(pi)` ordering exactly (see `weight_offload_gpu.rs`'s
        // `prefetch_pipeline_alternates_wins_under_single_slot_and_repeats_correctly`). ---
        let mut on_samples = Vec::with_capacity(runs);
        let mut last_stats = None;
        // Captured only on the very last layer of the very last run: for the
        // capacity-sufficient case this is the one page in this arm that was
        // actually promoted from a look-ahead prefetch (issued the previous
        // iteration) rather than paged in on demand, so verifying it -- not
        // just layer 0's on-demand page-in in the correctness gate above --
        // proves the prefetch/promote path itself, not only the fallback it
        // declines into for the two real-model rows.
        let mut on_arm_last_layer_readback: Option<Vec<u8>> = None;
        for run_idx in 0..runs {
            let residency = ep.weight_residency(budget);
            let start = Instant::now();
            for layer in 0..layers {
                if layer + 1 < layers {
                    residency
                        .prefetch_block_quantized_moe((layer + 1) as u64, &lazies[layer + 1], &host)
                        .expect("ON arm prefetch must not error");
                }
                let page = residency
                    .resident_mapped(layer as u64, &lazies[layer], &host)
                    .expect("ON arm page-in/promotion must succeed");
                if run_idx + 1 == runs && layer + 1 == layers {
                    let mut readback = vec![0u8; layer_bytes as usize];
                    // SAFETY: `readback` is sized to the page's exact byte
                    // length; `resident_mapped`'s promotion path already
                    // host-synchronized this page's transfer before
                    // returning it, so this copy reads its final bytes.
                    unsafe {
                        runtime
                            .dtoh(&mut readback, cuptr(page.device_ptr()))
                            .unwrap()
                    };
                    on_arm_last_layer_readback = Some(readback);
                }
                drop(page);
                proxy.run(&ep);
            }
            runtime.synchronize().unwrap();
            on_samples.push(start.elapsed());
            last_stats = Some(residency.stats());
        }
        let on_us = median_duration(on_samples).as_secs_f64() * 1e6;
        let stats = last_stats.expect("runs >= 1");
        let last_layer = layers - 1;
        let expected_last_layer = vec![0x10u8.wrapping_add(last_layer as u8); layer_bytes as usize];
        assert_eq!(
            on_arm_last_layer_readback.expect("ON arm must run at least one iteration"),
            expected_last_layer,
            "{}: the ON arm's final-layer page must be byte-identical to its \
             source, whether it arrived via a promoted look-ahead prefetch or \
             (for the two real-model rows, which decline every prefetch) the \
             on-demand fallback",
            case.name
        );

        // --- No-fallback gate, branched on this case's a priori expectation
        // (see `BenchCase` doc): a real model whose `2 * layer_bytes` exceeds
        // the pinned pool's retention cap must decline every time via the
        // pool-capacity guard (a truthful no-win, not a test failure); the
        // synthetic capacity-sufficient case must actually fire the fast
        // path with zero declines of any kind. Either branch failing its
        // expectation is a real signal (pool bounds, model shape, or the
        // guard itself changed) and must not be silently accepted. ---
        if case.expect_pool_capacity_decline {
            assert_eq!(
                stats.prefetch_declined_pool_capacity,
                (layers - 1) as u64,
                "{}: expected every look-ahead prefetch to decline via the \
                 pool-capacity guard (2 * layer_bytes exceeds the pinned \
                 pool's default retention cap for this real model shape) -- \
                 if this now fails, the pool bounds or shape changed and the \
                 module doc's 'truthful no-win' claim must be re-verified",
                case.name
            );
            assert_eq!(
                stats.prefetch_issued, 0,
                "{}: a declined prefetch must never be counted as issued",
                case.name
            );
            assert_eq!(
                stats.prefetch_promoted, 0,
                "{}: nothing was issued, so nothing can have been promoted",
                case.name
            );
        } else {
            assert_eq!(
                stats.prefetch_declined_pool_capacity, 0,
                "{}: the capacity-sufficient synthetic case must never hit the pool-capacity guard",
                case.name
            );
            assert_eq!(
                stats.prefetch_issued,
                (layers - 1) as u64,
                "{}: the ON arm must actually issue every look-ahead prefetch",
                case.name
            );
            assert_eq!(
                stats.prefetch_promoted,
                (layers - 1) as u64,
                "{}: the ON arm must actually promote every issued prefetch",
                case.name
            );
            assert_eq!(
                (
                    stats.prefetch_declined_budget,
                    stats.prefetch_declined_busy,
                    stats.prefetch_declined_unsupported,
                    stats.prefetch_declined_resident
                ),
                (0, 0, 0, 0),
                "{}: unexpected decline in the two-layer ON arm (budget, busy, unsupported, resident)",
                case.name
            );
        }

        let promote_wait_us = stats.prefetch_promote_wait_ns as f64 / 1e3;
        let outcome = if case.expect_pool_capacity_decline {
            "declined_pool_capacity"
        } else {
            "overlap_measured"
        };

        println!(
            "{:<38} {:>10.1} {:>10.1} {:>10.1} {:>12.1} {:>10.1} {:>10.2}% {:>12.1} {:>12.1}  outcome={}",
            case.name,
            layer_bytes as f64 / (1024.0 * 1024.0),
            off_us,
            on_us,
            promote_wait_us,
            htod_gbps,
            htod_pct_peak,
            materialize_us,
            vram_alloc_us,
            outcome,
        );
        // Stable machine-readable line (issue #82 cycle 7 baseline):
        // {model_shape, layers, layer_bytes, off_us, on_us, promote_wait_us,
        //  htod_GBps, pct_of_theoretical_pcie_bw, cold_materialize_us,
        //  cold_vram_alloc_us, cold_first_page_in_us, outcome}. `on_us <
        // off_us` means the look-ahead prefetch hid part of the transfer
        // behind the compute-proxy kernel; `promote_wait_us` is the exact
        // residual (0 = fully hidden, ~= transfer time = not hidden at all).
        // `htod_GBps`/`pct_of_theoretical_pcie_bw` isolate the real DMA-only
        // bandwidth (module doc's second finding: the DMA itself is
        // healthy); `cold_materialize_us`/`cold_vram_alloc_us` are the
        // one-time pinned-buffer-fill and VRAM-alloc costs, and
        // `cold_first_page_in_us` is their sum plus DMA (the isolated
        // single-page-in's total wall time) -- a page-in this size pays this
        // EVERY time when (as here, for the two real-model rows)
        // `outcome=declined_pool_capacity` means the pool never retains the
        // staging buffer for reuse. `outcome=declined_pool_capacity` is a
        // truthful no-win (the fast path never fired; `on_us` is expected ~=
        // `off_us`), reported as such rather than papered over -- see the
        // module doc.
        println!(
            "BQMOE_PREFETCH_OVERLAP model_shape={} layers={} layer_bytes={} compute_proxy_us={:.1} \
             off_us={:.1} on_us={:.1} promote_wait_us={:.1} htod_GBps={:.1} \
             pct_of_theoretical_pcie_bw={:.2} cold_materialize_us={:.1} cold_vram_alloc_us={:.1} \
             cold_first_page_in_us={:.1} prefetch_declined_pool_capacity={} outcome={} correctness=pass",
            case.name,
            layers,
            layer_bytes,
            compute_us,
            off_us,
            on_us,
            promote_wait_us,
            htod_gbps,
            htod_pct_peak,
            materialize_us,
            vram_alloc_us,
            bw_wall_us,
            stats.prefetch_declined_pool_capacity,
            outcome,
        );
    }

    proxy.teardown(&ep);
}
