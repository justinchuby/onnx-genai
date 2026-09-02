#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::unusual_byte_groupings,
    clippy::doc_lazy_continuation,
    clippy::uninlined_format_args,
    clippy::cloned_ref_to_slice_refs,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::manual_repeat_n,
    clippy::manual_is_multiple_of,
    clippy::err_expect,
    clippy::clone_on_copy
)]
//! On-GPU test for live GPU weight offload (WEIGHT_OFFLOAD Phase 3b).
//!
//! Proves the cardinal offload invariant: a weight served through the live GPU
//! paging path — VRAM page allocated, canonical bytes copied host→device via the
//! [`CudaWeightPager`] binder, then bound as a kernel input — produces output
//! *byte-identical* to the same weight uploaded the ordinary resident way.
//! Offload is an optimization, never an output change.
//!
//! Gated on a real device: CPU-only CI reports this as ignored unless
//! `gpu-tests` is enabled; feature-enabled runs fail loudly without CUDA.
//!
//! Run pinned to a free GPU, e.g.:
//!   CUDA_VISIBLE_DEVICES=0 taskset -c 1 \
//!     cargo test -p onnx-runtime-ep-cuda --test weight_offload_gpu

use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, ExternalMmapRegion, LazyDeviceWeightBinder,
    LazyWeight, LazyWeightBoundary, MmapRegionSource, ResidentWeight, TensorMut, TensorView,
    WeightHandleError,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{DataType, DeviceId, Node, NodeId, compute_contiguous_strides};

/// Reinterpret an `&[f32]` as its little-endian host bytes.
fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: `f32` is `Copy` with no padding; same lifetime, 4x length.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A host buffer standing in for an ONNX external-data mmap. Region bytes are
/// resolved by absolute offset, exactly as the executor's weight store will.
struct HostMmap {
    mapping_id: usize,
    bytes: Vec<u8>,
}

impl MmapRegionSource for HostMmap {
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

/// Row-major reference GEMM: `C[M,N] = A[M,K] · B[K,N]`.
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Run `A · B` on the GPU where the raw device pointer for `B` is supplied by
/// the caller (either a resident upload or a live-paged VRAM weight). Returns
/// the raw output bytes so equality is checked bit-for-bit.
fn run_matmul_with_b_ptr(
    ep: &CudaExecutionProvider,
    a: &[f32],
    b_ptr: *const std::ffi::c_void,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<u8> {
    let dev: DeviceId = ep.device_id();
    let rt = ep.runtime();

    let a_buf = ep.allocate(std::mem::size_of_val(a), 256).unwrap();
    let mut c_buf = ep.allocate(m * n * 4, 256).unwrap();
    // SAFETY: `a_buf` is sized for `a`'s bytes.
    unsafe { rt.htod(f32_bytes(a), cuptr(a_buf.as_ptr())).unwrap() }

    let a_shape = [m, k];
    let b_shape = [k, n];
    let out_shape = [m, n];
    let a_strides = compute_contiguous_strides(&a_shape);
    let b_strides = compute_contiguous_strides(&b_shape);
    let out_strides = compute_contiguous_strides(&out_shape);

    let a_view = TensorView::new(
        DevicePtr(a_buf.as_ptr()),
        DataType::Float32,
        &a_shape,
        &a_strides,
        dev,
    );
    let b_view = TensorView::new(
        DevicePtr(b_ptr),
        DataType::Float32,
        &b_shape,
        &b_strides,
        dev,
    );
    let out_view = TensorMut::new(
        DevicePtrMut(c_buf.as_mut_ptr()),
        DataType::Float32,
        &out_shape,
        &out_strides,
        dev,
    );

    let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
    let kernel = ep
        .get_kernel(&node, &[a_shape.to_vec(), b_shape.to_vec()], 17)
        .unwrap();
    kernel.execute(&[a_view, b_view], &mut [out_view]).unwrap();

    let mut out_bytes = vec![0u8; m * n * 4];
    // SAFETY: `c_buf` holds `m*n` f32 = m*n*4 bytes.
    unsafe { rt.dtoh(&mut out_bytes, cuptr(c_buf.as_ptr())).unwrap() }

    ep.deallocate(a_buf).unwrap();
    ep.deallocate(c_buf).unwrap();
    out_bytes
}

/// Build a `pkg.nxrt::BlockQuantizedMoE`-boundary lazy weight for `b` placed at
/// `offset` inside a fresh host mmap, plus that mmap.
fn lazy_weight_for(b: &[f32], k: usize, n: usize, offset: usize) -> (LazyWeight, HostMmap) {
    lazy_weight_with_boundary(LazyWeightBoundary::BlockQuantizedMoe, b, k, n, offset)
}

/// Same as [`lazy_weight_for`] but for an arbitrary offload `boundary` — used
/// by the prefetch scope-containment test to prove a dense `MatMul`-boundary
/// weight (which never goes through the executor's ahead-of-need prefetch
/// path) is correctly declined by `prefetch_block_quantized_moe`.
fn lazy_weight_with_boundary(
    boundary: LazyWeightBoundary,
    b: &[f32],
    k: usize,
    n: usize,
    offset: usize,
) -> (LazyWeight, HostMmap) {
    let mapping_id = 42;
    let b_bytes = f32_bytes(b).to_vec();
    let len = b_bytes.len();
    let mut backing = vec![0xABu8; offset]; // padding proves offset handling
    backing.extend_from_slice(&b_bytes);
    let host = HostMmap {
        mapping_id,
        bytes: backing,
    };
    let region = ExternalMmapRegion {
        mapping_id,
        offset,
        len,
    };
    let shape = vec![k, n];
    let resident_bytes = b_bytes.clone();
    let lazy = LazyWeight::new(boundary, DataType::Float32, shape.clone(), vec![region], {
        let shape = shape.clone();
        move || {
            ResidentWeight::new(DataType::Float32, shape.clone(), resident_bytes.clone())
                .map(onnx_runtime_ep_api::ResidentWeightMaterialization::reused)
        }
    })
    .unwrap();
    (lazy, host)
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn offloaded_weight_is_byte_identical_to_resident() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (m, k, n) = (3usize, 4usize, 2usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 + 0.125).collect();

    // Resident reference: upload B the ordinary way and run.
    let rt = ep.runtime();
    let b_resident = ep.allocate(std::mem::size_of_val(&b[..]), 256).unwrap();
    // SAFETY: buffer sized for B's bytes.
    unsafe { rt.htod(f32_bytes(&b), cuptr(b_resident.as_ptr())).unwrap() }
    let resident_out = run_matmul_with_b_ptr(&ep, &a, b_resident.as_ptr(), m, k, n);
    ep.deallocate(b_resident).unwrap();

    // Offload path: page B into VRAM through the live binder, run from there.
    let (lazy, host) = lazy_weight_for(&b, k, n, 512);
    let page = ep
        .weight_pager(&host)
        .bind_block_quantized_moe(&lazy)
        .expect("live GPU weight binding must succeed");
    assert_eq!(page.len(), std::mem::size_of_val(&b[..]));
    assert_eq!(page.dtype(), DataType::Float32);
    assert_eq!(page.shape(), &[k, n]);
    let offloaded_out = run_matmul_with_b_ptr(&ep, &a, page.device_ptr(), m, k, n);

    // Cardinal invariant: byte-for-byte identical output.
    assert_eq!(
        offloaded_out, resident_out,
        "offloaded weight output must be byte-identical to the resident path"
    );

    // Sanity: both agree with an independent CPU reference.
    let reference = cpu_matmul(&a, &b, m, k, n);
    let gpu: Vec<f32> = offloaded_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(
        gpu.iter()
            .zip(&reference)
            .all(|(x, y)| (x - y).abs() <= 1e-4),
        "gpu {gpu:?} vs reference {reference:?}"
    );
}

/// Mutation guard: a binding that copies the wrong region bytes must NOT match
/// the resident output. Locks in the test's sensitivity so a broken H2D copy
/// (wrong offset/length) cannot pass silently.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn offloaded_weight_wrong_region_diverges() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (m, k, n) = (3usize, 4usize, 2usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 + 0.125).collect();

    let rt = ep.runtime();
    let b_resident = ep.allocate(std::mem::size_of_val(&b[..]), 256).unwrap();
    // SAFETY: buffer sized for B's bytes.
    unsafe { rt.htod(f32_bytes(&b), cuptr(b_resident.as_ptr())).unwrap() }
    let resident_out = run_matmul_with_b_ptr(&ep, &a, b_resident.as_ptr(), m, k, n);
    ep.deallocate(b_resident).unwrap();

    // Point the region at corrupted bytes: same length, different offset window.
    let (mut lazy, mut host) = lazy_weight_for(&b, k, n, 512);
    // Corrupt the backing so the paged bytes differ from canonical B.
    for byte in host.bytes.iter_mut().skip(512) {
        *byte ^= 0xFF;
    }
    // Keep the region pointing at the (now corrupted) window.
    lazy.regions[0].offset = 512;
    let page = ep
        .weight_pager(&host)
        .bind_block_quantized_moe(&lazy)
        .unwrap();
    let corrupted_out = run_matmul_with_b_ptr(&ep, &a, page.device_ptr(), m, k, n);

    assert_ne!(
        corrupted_out, resident_out,
        "corrupted paged weight must not match the resident output — test would be blind"
    );
}

/// Build one host mmap holding every weight in `bs` back-to-back after a padding
/// prefix, plus a lazy weight per entry pointing at its region. All share one
/// mapping so a single `CudaWeightResidency` can page every one of them.
fn combined_weights(bs: &[Vec<f32>], k: usize, n: usize) -> (HostMmap, Vec<LazyWeight>) {
    let mapping_id = 42;
    let prefix = 256usize; // padding proves offset handling
    let mut backing = vec![0xABu8; prefix];
    let mut lazies = Vec::with_capacity(bs.len());
    for b in bs {
        let offset = backing.len();
        let b_bytes = f32_bytes(b).to_vec();
        let len = b_bytes.len();
        backing.extend_from_slice(&b_bytes);
        let region = ExternalMmapRegion {
            mapping_id,
            offset,
            len,
        };
        let shape = vec![k, n];
        let resident_bytes = b_bytes.clone();
        let lazy =
            LazyWeight::block_quantized_moe(DataType::Float32, shape.clone(), vec![region], {
                let shape = shape.clone();
                move || {
                    ResidentWeight::new(DataType::Float32, shape.clone(), resident_bytes.clone())
                        .map(onnx_runtime_ep_api::ResidentWeightMaterialization::reused)
                }
            })
            .unwrap();
        lazies.push(lazy);
    }
    (
        HostMmap {
            mapping_id,
            bytes: backing,
        },
        lazies,
    )
}

/// Live residency: paging under a one-page VRAM budget must page-in on a miss,
/// reuse on a hit, evict LRU on pressure, and stay byte-identical to resident.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn residency_pages_in_reuses_and_evicts() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (m, k, n) = (3usize, 4usize, 2usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b0: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 + 0.125).collect();
    let b1: Vec<f32> = (0..k * n).map(|i| (i as f32) * -0.5 + 3.0).collect();
    let page_bytes = (k * n * 4) as u64;

    // Resident references for both weights.
    let rt = ep.runtime();
    let resident_out = |b: &[f32]| {
        let buf = ep.allocate(std::mem::size_of_val(b), 256).unwrap();
        // SAFETY: buffer sized for b's bytes.
        unsafe { rt.htod(f32_bytes(b), cuptr(buf.as_ptr())).unwrap() }
        let out = run_matmul_with_b_ptr(&ep, &a, buf.as_ptr(), m, k, n);
        ep.deallocate(buf).unwrap();
        out
    };
    let resident_b0 = resident_out(&b0);
    let resident_b1 = resident_out(&b1);

    // One-page budget: holding one weight fills the cache; a second evicts it.
    let (host, lazies) = combined_weights(&[b0.clone(), b1.clone()], k, n);
    let residency = ep.weight_residency(page_bytes);

    // Miss on weight 0: one page-in, exactly one page resident.
    let page0 = residency.resident(0, &lazies[0], &host).unwrap();
    let out0 = run_matmul_with_b_ptr(&ep, &a, page0.device_ptr(), m, k, n);
    assert_eq!(out0, resident_b0, "paged weight 0 must match resident");
    let s = residency.stats();
    assert_eq!(s.page_ins, 1);
    assert_eq!(s.hits, 0);
    assert_eq!(s.pages_resident, 1);
    assert_eq!(s.resident_bytes, page_bytes);

    // Hit on weight 0 while still held: reuse the *same* device pointer, no copy.
    let page0_again = residency.resident(0, &lazies[0], &host).unwrap();
    assert_eq!(page0_again.device_ptr(), page0.device_ptr());
    let s = residency.stats();
    assert_eq!(s.page_ins, 1, "cache hit must not page in");
    assert_eq!(s.hits, 1);

    // Release both handles so the cache is the page's sole owner and can evict.
    drop(page0);
    drop(page0_again);

    // Miss on weight 1 under the one-page budget: weight 0 is evicted.
    let page1 = residency.resident(1, &lazies[1], &host).unwrap();
    let out1 = run_matmul_with_b_ptr(&ep, &a, page1.device_ptr(), m, k, n);
    assert_eq!(out1, resident_b1, "paged weight 1 must match resident");
    let s = residency.stats();
    assert_eq!(s.page_ins, 2);
    assert_eq!(s.evictions, 1, "one-page budget must evict the LRU page");
    assert_eq!(s.pages_resident, 1);
    assert_eq!(s.resident_bytes, page_bytes);
    assert_eq!(s.peak_resident_bytes, page_bytes);
}

/// Use-safety: a page still referenced by a live handle is never evicted, even
/// under budget pressure — the cache runs transiently over budget instead.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn residency_never_evicts_a_referenced_page() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (k, n) = (4usize, 2usize);
    let b0: Vec<f32> = (0..k * n).map(|i| (i as f32) + 1.0).collect();
    let b1: Vec<f32> = (0..k * n).map(|i| (i as f32) * 2.0).collect();
    let page_bytes = (k * n * 4) as u64;

    let (host, lazies) = combined_weights(&[b0, b1], k, n);
    let residency = ep.weight_residency(page_bytes);

    // Hold weight 0's handle across the admission of weight 1.
    let _held = residency.resident(0, &lazies[0], &host).unwrap();
    let _page1 = residency.resident(1, &lazies[1], &host).unwrap();

    let s = residency.stats();
    assert_eq!(s.evictions, 0, "a referenced page must not be evicted");
    assert_eq!(s.pages_resident, 2);
    assert_eq!(
        s.resident_bytes,
        2 * page_bytes,
        "both pages stay resident, transiently over budget"
    );
}

/// Live-dispatch path: `resident_materialized` (the entry point the CUDA EP's
/// `page_lazy_weight` calls) must materialize a lazy weight's canonical bytes,
/// stream them host→device, evict under a tiny budget, and stay byte-identical
/// to the resident upload. Also proves the process-global counters advance so an
/// opaque end-to-end run can observe that paging really happened.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn residency_materialized_pages_evicts_and_matches_resident() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (m, k, n) = (3usize, 4usize, 2usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b0: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 + 0.125).collect();
    let b1: Vec<f32> = (0..k * n).map(|i| (i as f32) * -0.5 + 3.0).collect();
    let page_bytes = (k * n * 4) as u64;

    let rt = ep.runtime();
    let resident_out = |b: &[f32]| {
        let buf = ep.allocate(std::mem::size_of_val(b), 256).unwrap();
        // SAFETY: buffer sized for b's bytes.
        unsafe { rt.htod(f32_bytes(b), cuptr(buf.as_ptr())).unwrap() }
        let out = run_matmul_with_b_ptr(&ep, &a, buf.as_ptr(), m, k, n);
        ep.deallocate(buf).unwrap();
        out
    };
    let resident_b0 = resident_out(&b0);
    let resident_b1 = resident_out(&b1);

    let (_host, lazies) = combined_weights(&[b0.clone(), b1.clone()], k, n);
    let residency = ep.weight_residency(page_bytes);
    onnx_runtime_ep_cuda::reset_global_offload_stats();

    // Miss on weight 0 via the materialized (no MmapRegionSource) path.
    let page0 = residency.resident_materialized(0, &lazies[0]).unwrap();
    let out0 = run_matmul_with_b_ptr(&ep, &a, page0.device_ptr(), m, k, n);
    assert_eq!(
        out0, resident_b0,
        "materialized weight 0 must match resident"
    );
    assert_eq!(page0.len() as u64, page_bytes);
    drop(page0);

    // Miss on weight 1 under the one-page budget evicts weight 0.
    let page1 = residency.resident_materialized(1, &lazies[1]).unwrap();
    let out1 = run_matmul_with_b_ptr(&ep, &a, page1.device_ptr(), m, k, n);
    assert_eq!(
        out1, resident_b1,
        "materialized weight 1 must match resident"
    );

    let stats = residency.stats();
    assert_eq!(stats.page_ins, 2, "each miss must page in");
    assert_eq!(
        stats.evictions, 1,
        "one-page budget must evict the LRU page"
    );
    assert_eq!(stats.pages_resident, 1);

    let global = onnx_runtime_ep_cuda::global_offload_stats();
    assert!(
        global.page_ins >= 2 && global.evictions >= 1,
        "process-global counters must observe paging: {global:?}"
    );
}

// ---------------------------------------------------------------------------
// BlockQuantizedMoE prefill ahead-of-need prefetch (issue #82 cycle 7).
//
// `residency.stats()` (a per-`CudaWeightResidency`-instance snapshot) is used
// for every exact-count assertion below, exactly like the pre-existing tests
// above use it for `page_ins`/`hits`/`evictions`. The process-global
// `GLOBAL_PREFETCH_*` counters this feature also updates are shared across
// every test in this binary (which `cargo test` runs in parallel by default),
// so they are only ever safe to assert with `>=`; the per-instance mirror in
// `CudaResidencyStats` is scoped to one residency and is what makes exact
// assertions here trustworthy under parallel test execution.
// ---------------------------------------------------------------------------

/// Basic correctness: issuing a prefetch enqueues the H2D copy but must not
/// yet be visible as resident; promoting it (via the ordinary `resident_mapped`
/// entry point a real dispatch uses) must resolve the fence, admit exactly
/// like an on-demand page-in, and produce output byte-identical to the
/// resident-upload reference.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn prefetch_then_promote_matches_resident_and_is_byte_identical() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (m, k, n) = (3usize, 4usize, 2usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 + 0.125).collect();
    let page_bytes = (k * n * 4) as u64;

    let rt = ep.runtime();
    let b_resident = ep.allocate(std::mem::size_of_val(&b[..]), 256).unwrap();
    // SAFETY: buffer sized for b's bytes.
    unsafe { rt.htod(f32_bytes(&b), cuptr(b_resident.as_ptr())).unwrap() }
    let resident_out = run_matmul_with_b_ptr(&ep, &a, b_resident.as_ptr(), m, k, n);
    ep.deallocate(b_resident).unwrap();

    let (lazy, host) = lazy_weight_for(&b, k, n, 512);
    let residency = ep.weight_residency(page_bytes);
    let key = 7u64;

    let issued = residency
        .prefetch_block_quantized_moe(key, &lazy, &host)
        .expect("prefetch must not error");
    assert!(issued, "an empty single slot must accept the prefetch");
    let s = residency.stats();
    assert_eq!(s.prefetch_issued, 1);
    assert_eq!(s.prefetch_issued_bytes, page_bytes);
    assert_eq!(
        s.pages_resident, 0,
        "an issued-but-not-yet-promoted prefetch must not appear resident"
    );
    assert_eq!(s.page_ins, 0, "issuing must not itself count as a page-in");

    // The real dispatch entry point: promotes the pending prefetch instead of
    // re-copying.
    let page = residency
        .resident_mapped(key, &lazy, &host)
        .expect("promotion must succeed");
    assert_eq!(page.len(), page_bytes as usize);
    assert_eq!(page.dtype(), DataType::Float32);
    assert_eq!(page.shape(), &[k, n]);

    let s = residency.stats();
    assert_eq!(s.prefetch_promoted, 1);
    assert_eq!(
        s.page_ins, 1,
        "promotion must go through the same admit() accounting as an on-demand page-in"
    );
    assert_eq!(s.hits, 0);
    assert_eq!(s.pages_resident, 1);
    assert_eq!(s.resident_bytes, page_bytes);

    let offloaded_out = run_matmul_with_b_ptr(&ep, &a, page.device_ptr(), m, k, n);
    assert_eq!(
        offloaded_out, resident_out,
        "prefetched-then-promoted weight output must be byte-identical to the resident path"
    );
}

/// Budget gate: a prefetch that would need to evict a resident page to fit
/// must decline rather than ever evicting to make room for a look-ahead guess.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn prefetch_declines_when_a_resident_page_would_have_to_be_evicted() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (k, n) = (4usize, 2usize);
    let b0: Vec<f32> = (0..k * n).map(|i| (i as f32) + 1.0).collect();
    let b1: Vec<f32> = (0..k * n).map(|i| (i as f32) * 2.0).collect();
    let page_bytes = (k * n * 4) as u64;

    let (host, lazies) = combined_weights(&[b0, b1], k, n);
    // Budget for exactly one page: admitting weight 0 leaves zero headroom.
    let residency = ep.weight_residency(page_bytes);
    let _page0 = residency
        .resident_mapped(0, &lazies[0], &host)
        .expect("first admission must fit the budget exactly");
    assert_eq!(residency.stats().pages_resident, 1);

    let issued = residency
        .prefetch_block_quantized_moe(1, &lazies[1], &host)
        .expect("decline must not error");
    assert!(
        !issued,
        "a prefetch that would require evicting a resident page must decline"
    );
    let s = residency.stats();
    assert_eq!(s.prefetch_declined_budget, 1);
    assert_eq!(s.prefetch_issued, 0);
    assert_eq!(
        s.pages_resident, 1,
        "the declined prefetch must not have touched the resident set"
    );
}

/// Single-slot busy gate, isolated from the budget gate: a second candidate
/// cannot preempt or queue behind a first still-pending prefetch, and the slot
/// frees up the instant the first is promoted.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn prefetch_single_slot_declines_a_second_key_until_the_first_is_promoted() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (k, n) = (4usize, 2usize);
    let b0: Vec<f32> = (0..k * n).map(|i| (i as f32) + 1.0).collect();
    let b1: Vec<f32> = (0..k * n).map(|i| (i as f32) * 2.0).collect();
    let page_bytes = (k * n * 4) as u64;
    // Budget for both pages: this test isolates the *slot* invariant from the
    // budget gate covered separately above.
    let (host, lazies) = combined_weights(&[b0, b1], k, n);
    let residency = ep.weight_residency(2 * page_bytes);

    assert!(
        residency
            .prefetch_block_quantized_moe(0, &lazies[0], &host)
            .unwrap(),
        "an empty slot must accept the first prefetch"
    );
    assert!(
        !residency
            .prefetch_block_quantized_moe(1, &lazies[1], &host)
            .unwrap(),
        "a second key must not preempt or queue behind the first"
    );
    let s = residency.stats();
    assert_eq!(s.prefetch_issued, 1);
    assert_eq!(s.prefetch_declined_busy, 1);

    // Promote the first: the slot is now free.
    let _page0 = residency.resident_mapped(0, &lazies[0], &host).unwrap();
    assert_eq!(residency.stats().prefetch_promoted, 1);

    assert!(
        residency
            .prefetch_block_quantized_moe(1, &lazies[1], &host)
            .unwrap(),
        "the slot must accept a new prefetch once the previous one is promoted"
    );
    let s = residency.stats();
    assert_eq!(s.prefetch_issued, 2);
    assert_eq!(s.prefetch_declined_busy, 1, "must not grow further");
}

/// Already-resident gate: prefetching a key that is already in the resident
/// set is declined (there is nothing to prefetch), not silently accepted or
/// treated as a slot conflict.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn prefetch_declines_when_already_resident() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (k, n) = (4usize, 2usize);
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) + 1.0).collect();
    let page_bytes = (k * n * 4) as u64;
    let (lazy, host) = lazy_weight_for(&b, k, n, 512);
    let residency = ep.weight_residency(page_bytes);

    let _page = residency
        .resident_mapped(0, &lazy, &host)
        .expect("ordinary on-demand admission");
    assert_eq!(residency.stats().pages_resident, 1);

    let issued = residency
        .prefetch_block_quantized_moe(0, &lazy, &host)
        .expect("decline must not error");
    assert!(!issued, "prefetching an already-resident key is a no-op");
    let s = residency.stats();
    assert_eq!(s.prefetch_declined_resident, 1);
    assert_eq!(s.prefetch_issued, 0);
}

/// Scope containment: the prefetch path is `BlockQuantizedMoE`-only. A dense
/// `MatMul`-boundary weight (or any other boundary) must always decline, and
/// must decline *silently* — no counter fires, because there was never a
/// legitimate candidate to begin with (unlike the other decline reasons, which
/// all fire a counter).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn prefetch_declines_for_non_block_quantized_moe_boundary() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (k, n) = (4usize, 2usize);
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) + 1.0).collect();
    let page_bytes = (k * n * 4) as u64;
    let (lazy, host) = lazy_weight_with_boundary(LazyWeightBoundary::MatMul, &b, k, n, 512);
    let residency = ep.weight_residency(page_bytes);

    let issued = residency
        .prefetch_block_quantized_moe(0, &lazy, &host)
        .expect("boundary rejection must not error");
    assert!(!issued, "a MatMul-boundary weight is out of scope");
    let s = residency.stats();
    assert_eq!(s.prefetch_issued, 0);
    assert_eq!(s.prefetch_declined_budget, 0);
    assert_eq!(s.prefetch_declined_busy, 0);
    assert_eq!(s.prefetch_declined_unsupported, 0);
    assert_eq!(s.prefetch_declined_resident, 0);
}

/// The zero-copy hybrid (#864) and this prefetch increment are independent,
/// out-of-scope-for-each-other features (#82's directive: no interaction with
/// residency mechanisms this cycle does not own). When the hybrid is active,
/// prefetch must decline through the same "unsupported" reason VMM stable-VA
/// admission uses, not silently do the wrong thing.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn prefetch_declines_when_zero_copy_hybrid_is_active() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (k, n) = (4usize, 2usize);
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) + 1.0).collect();
    let page_bytes = (k * n * 4) as u64;
    let (lazy, host) = lazy_weight_for(&b, k, n, 512);
    let residency = ep.weight_residency(page_bytes).with_zero_copy_hybrid(true);

    let issued = residency
        .prefetch_block_quantized_moe(0, &lazy, &host)
        .expect("decline must not error");
    assert!(!issued);
    let s = residency.stats();
    assert_eq!(s.prefetch_declined_unsupported, 1);
    assert_eq!(s.prefetch_issued, 0);
}

/// Copy-failure rollback: a source read failure inside prefetch issuance
/// (region bytes cannot be resolved) must surface as an `Err`, must not touch
/// any success counter, and must not leave the single slot stuck — the very
/// next prefetch for a different, valid key must succeed normally.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn prefetch_source_error_rolls_back_cleanly() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (m, k, n) = (3usize, 4usize, 2usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b_bad: Vec<f32> = (0..k * n).map(|i| (i as f32) + 1.0).collect();
    let b_good: Vec<f32> = (0..k * n).map(|i| (i as f32) * 3.0 - 1.0).collect();
    let page_bytes = (k * n * 4) as u64;

    let (mut lazy_bad, host_bad) = lazy_weight_for(&b_bad, k, n, 512);
    // Point the region at a mapping the host mmap does not recognize, forcing
    // `fill_staging_from_regions` to fail inside prefetch issuance.
    lazy_bad.regions[0].mapping_id = 999;

    let residency = ep.weight_residency(2 * page_bytes);
    let result = residency.prefetch_block_quantized_moe(0, &lazy_bad, &host_bad);
    assert!(
        result.is_err(),
        "an unresolvable region must surface as an error, not a silent Ok(false)"
    );
    let s = residency.stats();
    assert_eq!(
        s.prefetch_issued, 0,
        "a failed fill must never reach the enqueue/publish step"
    );
    assert_eq!(s.pages_resident, 0);

    // The slot must not be stuck: a subsequent, valid prefetch for a different
    // key must succeed and promote correctly.
    let (lazy_good, host_good) = lazy_weight_for(&b_good, k, n, 512);
    assert!(
        residency
            .prefetch_block_quantized_moe(1, &lazy_good, &host_good)
            .expect("a valid prefetch after a failed one must not error"),
        "the slot must not be stuck busy after a failed prefetch"
    );
    let page = residency
        .resident_mapped(1, &lazy_good, &host_good)
        .expect("promotion of the valid prefetch must succeed");
    let out = run_matmul_with_b_ptr(&ep, &a, page.device_ptr(), m, k, n);
    let reference = cpu_matmul(&a, &b_good, m, k, n);
    let gpu: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(
        gpu.iter()
            .zip(&reference)
            .all(|(x, y)| (x - y).abs() <= 1e-4),
        "gpu {gpu:?} vs reference {reference:?}"
    );
}

/// The single-slot, lookahead-1 design's central, honestly-documented
/// limitation: under the executor's real per-node call order —
/// `prefetch_lazy_weights_after(pi)` (which issues a prefetch for node
/// `pi + 1`'s weight) runs *before* `exec_plan_node(pi)` (which needs node
/// `pi`'s own weight, promoting *its* prefetch if one is pending) — at most
/// every other layer transition can benefit from prefetch. Node 0's own
/// weight is never itself prefetched (nothing runs before it); node 1's
/// prefetch (issued during node 0's turn, into an empty slot) wins; node 2's
/// attempt finds node 1's own promotion has not happened yet and is declined
/// busy; node 3 then wins because node 1's promotion freed the slot; and so
/// on. This is traced exactly here, including a second and third pass over
/// the same 5-layer pipeline (as a decode loop would repeat it), where every
/// key is already resident and the whole pipeline degrades to ordinary cache
/// hits plus `declined_resident` on the now-redundant look-ahead prefetch
/// attempts.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn prefetch_pipeline_alternates_wins_under_single_slot_and_repeats_correctly() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (m, k, n) = (3usize, 4usize, 2usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let layers: usize = 5;
    let bs: Vec<Vec<f32>> = (0..layers)
        .map(|layer| {
            (0..k * n)
                .map(|i| (i as f32) * 0.1 + layer as f32)
                .collect()
        })
        .collect();
    let page_bytes = (k * n * 4) as u64;

    // Independent CPU-free GPU reference for every layer, computed once
    // up-front via a direct resident upload (never touches the residency
    // under test).
    let rt = ep.runtime();
    let resident_outs: Vec<Vec<u8>> = bs
        .iter()
        .map(|b| {
            let buf = ep.allocate(std::mem::size_of_val(&b[..]), 256).unwrap();
            // SAFETY: buffer sized for b's bytes.
            unsafe { rt.htod(f32_bytes(b), cuptr(buf.as_ptr())).unwrap() }
            let out = run_matmul_with_b_ptr(&ep, &a, buf.as_ptr(), m, k, n);
            ep.deallocate(buf).unwrap();
            out
        })
        .collect();

    let (host, lazies) = combined_weights(&bs, k, n);
    // Budget for all five layers resident at once: this test's point is the
    // prefetch/promote/decline bookkeeping, not eviction, which the other
    // tests above already cover in isolation.
    let residency = ep.weight_residency(layers as u64 * page_bytes);

    // Expected cumulative per-instance counters after each of 3 passes, hand-
    // traced from the single-slot + lookahead-1 interleaving documented above.
    struct Expected {
        prefetch_issued: u64,
        prefetch_declined_busy: u64,
        prefetch_declined_resident: u64,
        prefetch_promoted: u64,
        page_ins: u64,
        hits: u64,
    }
    let expected_after_pass = [
        Expected {
            prefetch_issued: 2,
            prefetch_declined_busy: 2,
            prefetch_declined_resident: 0,
            prefetch_promoted: 2,
            page_ins: 5,
            hits: 0,
        },
        Expected {
            prefetch_issued: 2,
            prefetch_declined_busy: 2,
            prefetch_declined_resident: 4,
            prefetch_promoted: 2,
            page_ins: 5,
            hits: 5,
        },
        Expected {
            prefetch_issued: 2,
            prefetch_declined_busy: 2,
            prefetch_declined_resident: 8,
            prefetch_promoted: 2,
            page_ins: 5,
            hits: 10,
        },
    ];

    for (pass, expect) in expected_after_pass.iter().enumerate() {
        for pi in 0..layers {
            if pi + 1 < layers {
                // Models `prefetch_lazy_weights_after(pi)`: look one node ahead.
                residency
                    .prefetch_block_quantized_moe((pi + 1) as u64, &lazies[pi + 1], &host)
                    .unwrap_or_else(|e| panic!("pass {pass} node {pi} prefetch(+1) error: {e}"));
            }
            // Models `exec_plan_node(pi)`: needs node pi's own weight now.
            let page = residency
                .resident_mapped(pi as u64, &lazies[pi], &host)
                .unwrap_or_else(|e| panic!("pass {pass} node {pi} resident_mapped error: {e}"));
            let out = run_matmul_with_b_ptr(&ep, &a, page.device_ptr(), m, k, n);
            assert_eq!(
                out, resident_outs[pi],
                "pass {pass} layer {pi} output diverged from the resident reference"
            );
        }

        let s = residency.stats();
        assert_eq!(
            s.prefetch_issued, expect.prefetch_issued,
            "pass {pass}: prefetch_issued"
        );
        assert_eq!(
            s.prefetch_declined_busy, expect.prefetch_declined_busy,
            "pass {pass}: prefetch_declined_busy"
        );
        assert_eq!(
            s.prefetch_declined_resident, expect.prefetch_declined_resident,
            "pass {pass}: prefetch_declined_resident"
        );
        assert_eq!(
            s.prefetch_promoted, expect.prefetch_promoted,
            "pass {pass}: prefetch_promoted"
        );
        assert_eq!(s.page_ins, expect.page_ins, "pass {pass}: page_ins");
        assert_eq!(s.hits, expect.hits, "pass {pass}: hits");
        assert_eq!(
            s.pages_resident, layers as u64,
            "pass {pass}: pages_resident"
        );
        assert_eq!(
            s.evictions, 0,
            "pass {pass}: a big-enough budget must never evict"
        );
    }
}

/// Regression guard for issue #82's pool-capacity soundness gate
/// (`PinnedStagingPool::can_retain_concurrent`): a look-ahead prefetch whose
/// byte size the *shared* pinned-staging pool cannot retain two
/// concurrently-live copies of must decline up front via
/// `prefetch_declined_pool_capacity` -- not silently accept and pay a fresh
/// `cuMemHostAlloc`/`cuMemFreeHost` pair every steady-state cycle, which would
/// reintroduce issue #837's exact regression for this specific path. The
/// oversized region's length only needs to be *declared*, never backed by
/// real data, because the guard must fire strictly before `source` is ever
/// read (proven here by a materializer/host buffer that would panic or be too
/// short if touched).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn prefetch_declines_when_the_pinned_pool_cannot_retain_two_concurrent_buffers() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    // Bigger than half the pinned pool's default 512 MiB retention cap, so
    // `can_retain_concurrent(len, 2)` is false regardless of the pool's
    // buffer-count bound. Order-of-magnitude matches DeepSeek-V2-Lite's real
    // ~294 MiB BlockQuantizedMoE per-layer bank (see the issue #82
    // bqmoe-prefetch-overlap benchmark), rounded for the assertion.
    let oversized_bytes: usize = 300 * 1024 * 1024;
    let mapping_id = 99;
    let host = HostMmap {
        mapping_id,
        bytes: vec![0u8; 16], // deliberately far short of `oversized_bytes`
    };
    let region = ExternalMmapRegion {
        mapping_id,
        offset: 0,
        len: oversized_bytes,
    };
    let lazy = LazyWeight::block_quantized_moe(
        DataType::Uint8,
        vec![oversized_bytes],
        vec![region],
        || {
            panic!(
                "resident materializer must never run: the pool-capacity guard must decline before eager admission"
            )
        },
    )
    .unwrap();

    // Budget large enough that the decline is attributable to the pool guard
    // alone, not the (already-covered) budget gate.
    let residency = ep.weight_residency(oversized_bytes as u64);
    let issued = residency
        .prefetch_block_quantized_moe(1, &lazy, &host)
        .expect("decline must not error");
    assert!(
        !issued,
        "a boundary the shared pinned pool cannot retain two concurrent copies of must decline"
    );
    let s = residency.stats();
    assert_eq!(s.prefetch_declined_pool_capacity, 1);
    assert_eq!(s.prefetch_issued, 0);
    assert_eq!(
        s.pages_resident, 0,
        "the declined prefetch must not have touched the resident set"
    );
    assert_eq!(
        s.pinned_pool_alloc_calls, 0,
        "the pool-capacity guard must decline before ever touching the pinned pool"
    );
}

/// Regression guard for the pool-routing fix itself (issue #82): once a
/// prefetch's staging buffer is promoted, it must return to the *shared*
/// [`PinnedStagingPool`] for reuse -- not be dropped and freed, which would
/// silently reintroduce issue #837's per-page-in `cuMemHostAlloc` cost for
/// this path (the bug this cycle found and fixed: the prefetch path used to
/// call `alloc_pinned` directly and drop the buffer on promotion, bypassing
/// the pool entirely). Two same-size prefetch+promote cycles, back to back,
/// must therefore show exactly one real pinned allocation and one reuse.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn prefetch_promote_returns_the_staging_buffer_to_the_shared_pinned_pool_for_reuse() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let (k, n) = (4usize, 2usize);
    let b0: Vec<f32> = (0..k * n).map(|i| (i as f32) + 1.0).collect();
    let b1: Vec<f32> = (0..k * n).map(|i| (i as f32) * 3.0 - 2.0).collect();
    let page_bytes = (k * n * 4) as u64;

    let (host, lazies) = combined_weights(&[b0, b1], k, n);
    // Budget for both pages resident at once, so the second cycle's admission
    // needs no eviction of the first -- isolates pool-reuse behavior from the
    // (already-covered) eviction/budget gates.
    let residency = ep.weight_residency(page_bytes * 2);

    // Cycle 1: key 0. The first-ever prefetch of this size must page-lock
    // exactly one buffer.
    assert!(
        residency
            .prefetch_block_quantized_moe(0, &lazies[0], &host)
            .expect("cycle 1 prefetch must not error"),
        "an empty single slot must accept the prefetch"
    );
    residency
        .resident_mapped(0, &lazies[0], &host)
        .expect("cycle 1 promotion must succeed");
    let s1 = residency.stats();
    assert_eq!(s1.pinned_pool_alloc_calls, 1);
    assert_eq!(s1.pinned_pool_reuses, 0);

    // Cycle 2: key 1, same size -- must be served by the buffer cycle 1's
    // promotion retired to the pool, not a fresh `cuMemHostAlloc`.
    assert!(
        residency
            .prefetch_block_quantized_moe(1, &lazies[1], &host)
            .expect("cycle 2 prefetch must not error"),
        "the freed single slot must accept the second prefetch"
    );
    residency
        .resident_mapped(1, &lazies[1], &host)
        .expect("cycle 2 promotion must succeed");
    let s2 = residency.stats();
    assert_eq!(
        s2.pinned_pool_alloc_calls, 1,
        "a same-size second cycle must not page-lock a second buffer"
    );
    assert_eq!(
        s2.pinned_pool_reuses, 1,
        "the second cycle must be served by the buffer the first cycle's promotion retired"
    );
}
