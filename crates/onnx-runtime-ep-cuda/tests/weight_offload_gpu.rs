//! On-GPU test for live GPU weight offload (WEIGHT_OFFLOAD Phase 3b).
//!
//! Proves the cardinal offload invariant: a weight served through the live GPU
//! paging path — VRAM page allocated, canonical bytes copied host→device via the
//! [`CudaWeightPager`] binder, then bound as a kernel input — produces output
//! *byte-identical* to the same weight uploaded the ordinary resident way.
//! Offload is an optimization, never an output change.
//!
//! Gated on a real device: prints `skip` and returns when no CUDA GPU is
//! present, so the crate still tests cleanly on non-GPU machines.
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

#[test]
fn offloaded_weight_is_byte_identical_to_resident() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            return;
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
#[test]
fn offloaded_weight_wrong_region_diverges() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            return;
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
