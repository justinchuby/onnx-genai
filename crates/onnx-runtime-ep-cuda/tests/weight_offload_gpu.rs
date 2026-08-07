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
    LazyWeight, MmapRegionSource, ResidentWeight, TensorMut, TensorView, WeightHandleError,
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
    let lazy = LazyWeight::block_quantized_moe(DataType::Float32, shape.clone(), vec![region], {
        let shape = shape.clone();
        move || ResidentWeight::new(DataType::Float32, shape.clone(), resident_bytes.clone())
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
