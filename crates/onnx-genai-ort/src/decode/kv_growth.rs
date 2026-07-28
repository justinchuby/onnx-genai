use super::*;
pub(super) use onnx_genai_kv::kv_capacity_bucket;

/// Where a shared-KV buffer lives, selecting the prefix-copy transfer used when
/// growing it. Non-CUDA device EPs are rejected earlier (see
/// [`DecodeSession::grow_device`]) so only these two paths reach the copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GrowDevice {
    /// Host-resident buffer: rearrange the prefix directly in host memory.
    Host,
    /// CUDA device-resident buffer: copy the prefix device-to-device via
    /// `cudart` (`cudaMemcpy` / `cudaMemset`).
    Cuda,
}

/// Allocate a `new_shape` shared-KV buffer and copy the valid sequence prefix
/// `[0, valid_len)` of `old` into it, zeroing positions `>= valid_len`.
///
/// `device` selects the transfer path. Host buffers rearrange the prefix
/// directly in host memory. CUDA device buffers are NOT host-addressable, so
/// their bytes are relocated with a direct device-to-device `cudaMemcpy` on the
/// raw KV device pointers — ORT's `CopyTensors` cannot be used because the
/// built-in CUDA EP registers no env-level data-transfer (it fails with "Data
/// transfer implementation ... not found (code 9)").
pub(super) fn grow_kv_value(
    old: &Value,
    new_shape: &[i64],
    seq_axis: usize,
    valid_len: usize,
    device: GrowDevice,
    device_allocator: Option<&crate::Allocator>,
) -> Result<Value> {
    match device {
        GrowDevice::Host => grow_kv_value_host(old, new_shape, seq_axis, valid_len),
        GrowDevice::Cuda => {
            let allocator = device_allocator.ok_or_else(|| {
                OrtError::InvalidArgument("CUDA KV growth requires the device allocator".into())
            })?;
            grow_kv_value_cuda(old, new_shape, seq_axis, valid_len, allocator)
        }
    }
}

/// Grow a host-resident KV buffer: read the old contents, rearrange the prefix
/// with [`copy_seq_prefix`], and materialize a fresh host tensor. The
/// dtype-specific arms differ only in the element type used for the round-trip
/// (16-bit float dtypes are moved as their raw bit patterns).
fn grow_kv_value_host(
    old: &Value,
    new_shape: &[i64],
    seq_axis: usize,
    valid_len: usize,
) -> Result<Value> {
    match old.dtype() {
        DataType::Float32 => {
            let src = old.to_vec_f32()?;
            let grown = copy_seq_prefix(&src, old.shape(), new_shape, seq_axis, valid_len);
            Value::from_vec_f32(grown, new_shape)
        }
        DataType::Float16 => {
            let src = old.to_vec_f16_bits()?;
            let grown = copy_seq_prefix(&src, old.shape(), new_shape, seq_axis, valid_len);
            Value::from_vec_f16_bits(grown, new_shape)
        }
        DataType::BFloat16 => {
            let src = old.to_vec_bf16_bits()?;
            let grown = copy_seq_prefix(&src, old.shape(), new_shape, seq_axis, valid_len);
            Value::from_vec_bf16_bits(grown, new_shape)
        }
        dtype => Err(OrtError::InvalidArgument(format!(
            "cannot grow shared KV buffer with dtype {dtype:?}"
        ))),
    }
}

/// Grow a CUDA device-resident KV buffer with a direct device-to-device copy.
///
/// Allocates the larger buffer on the same device allocator, zeroes it so the
/// tail past `valid_len` is defined (those positions are masked out, but keeping
/// them zero avoids relying on uninitialized device memory), then copies each
/// per-block valid prefix run from the old device pointer to the new one via
/// `cudaMemcpy(cudaMemcpyDeviceToDevice)`. The strided byte layout matches
/// [`copy_seq_prefix`]; see [`plan_kv_prefix_copy`] for the offset arithmetic.
fn grow_kv_value_cuda(
    old: &Value,
    new_shape: &[i64],
    seq_axis: usize,
    valid_len: usize,
    device_allocator: &crate::Allocator,
) -> Result<Value> {
    #[cfg(feature = "cuda")]
    {
        let dtype = old.dtype();
        if !matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(OrtError::InvalidArgument(format!(
                "cannot grow shared KV buffer with dtype {dtype:?}"
            )));
        }
        let dst = Value::empty_in(new_shape, dtype, device_allocator)?;
        let plan =
            plan_kv_prefix_copy(old.shape(), new_shape, seq_axis, valid_len, dtype.size_of());

        let src_addr = old.data_ptr_addr()?;
        let dst_addr = dst.data_ptr_addr()?;
        // Pin the thread's current CUDA device to the KV buffers' device for the
        // whole grow sequence. The raw cudart calls below act on the *current*
        // device, but the KV lives on the EP's configured device (which may be
        // non-zero via ONNX_GENAI_CUDA_DEVICE); without this the barriers could
        // target the wrong device. Restored on drop.
        let _device_guard =
            crate::cuda_rt::DeviceGuard::set(device_allocator.memory_info.device_id)?;
        // The prior decode step's KV writes run on the ORT CUDA EP's stream and
        // may still be in flight; block until they land before we read the old
        // buffer, so the copied prefix is complete rather than partially written.
        crate::cuda_rt::device_synchronize()?;
        // Define the whole destination first, then overwrite the valid prefix
        // blocks — cheaper to reason about than zeroing only the gaps.
        crate::cuda_rt::memset_zero(dst_addr, plan.total_bytes)?;
        for seg in &plan.segments {
            crate::cuda_rt::memcpy_device_to_device(
                dst_addr + seg.dst_offset,
                src_addr + seg.src_offset,
                seg.len,
            )?;
        }
        // Our memset/memcpy run on the default stream; block until they finish
        // so the next ORT Run (on the EP's stream) reads a fully populated
        // buffer rather than racing our copy.
        crate::cuda_rt::device_synchronize()?;
        Ok(dst)
    }
    #[cfg(not(feature = "cuda"))]
    {
        // Unreachable in practice: a CUDA session cannot exist without the
        // `cuda` feature (the EP append fails), so `GrowDevice::Cuda` is never
        // produced here. Kept compiling to satisfy the type checker.
        let _ = (old, new_shape, seq_axis, valid_len, device_allocator);
        Err(OrtError::InvalidArgument(
            "CUDA KV growth requires building onnx-genai-ort with --features cuda".into(),
        ))
    }
}

/// Rearrange the valid sequence prefix from an `src_shape` KV tensor into a
/// fresh `dst_shape` buffer (positions past `valid_len` default to zero).
///
/// The sequence axis is generally not the outermost dimension (KV tensors are
/// `[batch, kv_heads, seq, head_dim]`, `seq_axis = rank-2`), so a prefix along
/// it is strided: each of the `blocks = prod(dims before seq_axis)` leading
/// blocks holds a contiguous `valid_len * inner` run that must be relocated from
/// the old per-block stride (`src_cap * inner`) to the new one (`dst_cap *
/// inner`).
fn copy_seq_prefix<T: Copy + Default>(
    src: &[T],
    src_shape: &[i64],
    dst_shape: &[i64],
    seq_axis: usize,
    valid_len: usize,
) -> Vec<T> {
    let inner: usize = dst_shape[seq_axis + 1..]
        .iter()
        .map(|&d| d as usize)
        .product();
    let blocks: usize = dst_shape[..seq_axis].iter().map(|&d| d as usize).product();
    let src_cap = src_shape[seq_axis] as usize;
    let dst_cap = dst_shape[seq_axis] as usize;
    let mut out = vec![T::default(); blocks * dst_cap * inner];
    let run = valid_len * inner;
    for block in 0..blocks {
        let src_off = block * src_cap * inner;
        let dst_off = block * dst_cap * inner;
        out[dst_off..dst_off + run].copy_from_slice(&src[src_off..src_off + run]);
    }
    out
}

/// A single contiguous byte run to copy from the old KV buffer into the new one
/// while growing it, expressed as device-buffer byte offsets.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KvCopySegment {
    /// Byte offset of the run within the source buffer.
    src_offset: usize,
    /// Byte offset of the run within the destination buffer.
    dst_offset: usize,
    /// Length of the run in bytes.
    len: usize,
}

/// The full plan for relocating a KV prefix into a larger buffer: the total
/// destination size (so the whole buffer can be zeroed first) plus one copy
/// segment per strided block.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct KvPrefixCopyPlan {
    /// Total size of the destination buffer in bytes.
    total_bytes: usize,
    /// One contiguous run per `blocks` (batch * kv_heads for a rank-4 tensor).
    segments: Vec<KvCopySegment>,
}

/// Compute the strided byte-offset plan for relocating the valid sequence
/// prefix `[0, valid_len)` of a KV tensor into a larger buffer.
///
/// Mirrors [`copy_seq_prefix`]'s block layout in raw bytes: the sequence axis is
/// generally not outermost (KV tensors are `[batch, kv_heads, seq, head_dim]`,
/// `seq_axis = rank-2`), so each of the `blocks = prod(dims before seq_axis)`
/// leading blocks holds a contiguous `valid_len * inner` element run that moves
/// from the old per-block stride (`src_cap * inner`) to the new one
/// (`dst_cap * inner`). Multiplying by `elem_size` yields the device byte
/// offsets a `cudaMemcpy` needs. Factored out (and pure) so the offset
/// arithmetic is unit-testable without a GPU.
#[cfg(any(feature = "cuda", test))]
fn plan_kv_prefix_copy(
    src_shape: &[i64],
    dst_shape: &[i64],
    seq_axis: usize,
    valid_len: usize,
    elem_size: usize,
) -> KvPrefixCopyPlan {
    let inner: usize = dst_shape[seq_axis + 1..]
        .iter()
        .map(|&d| d as usize)
        .product();
    let blocks: usize = dst_shape[..seq_axis].iter().map(|&d| d as usize).product();
    let src_cap = src_shape[seq_axis] as usize;
    let dst_cap = dst_shape[seq_axis] as usize;
    let src_stride = src_cap * inner * elem_size;
    let dst_stride = dst_cap * inner * elem_size;
    let run = valid_len * inner * elem_size;
    let total_bytes = blocks * dst_stride;
    let segments = if run == 0 {
        Vec::new()
    } else {
        (0..blocks)
            .map(|block| KvCopySegment {
                src_offset: block * src_stride,
                dst_offset: block * dst_stride,
                len: run,
            })
            .collect()
    };
    KvPrefixCopyPlan {
        total_bytes,
        segments,
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_capacity_bucket_rounds_to_power_of_two_min_256() {
        let hard = 32768;
        assert_eq!(kv_capacity_bucket(0, hard), 256);
        assert_eq!(kv_capacity_bucket(1, hard), 256);
        assert_eq!(kv_capacity_bucket(128, hard), 256);
        assert_eq!(kv_capacity_bucket(256, hard), 256);
        assert_eq!(kv_capacity_bucket(257, hard), 512);
        assert_eq!(kv_capacity_bucket(5000, hard), 8192);
    }

    #[test]
    fn kv_capacity_bucket_caps_at_hard_max() {
        // A power-of-two round-up past hard_max clamps to hard_max. The caller
        // guarantees `len <= hard_max`, so the bucket still holds `len`.
        assert_eq!(kv_capacity_bucket(20000, 32768), 32768);
        assert_eq!(kv_capacity_bucket(32768, 32768), 32768);
        // Hard cap below MIN_BUCKET still clamps to the cap.
        assert_eq!(kv_capacity_bucket(100, 128), 128);
    }

    #[test]
    fn kv_capacity_bucket_is_monotonic_and_within_bounds() {
        let hard = 32768;
        let mut prev = 0;
        for len in [0usize, 1, 200, 256, 257, 1000, 4096, 8000, 16000, 32768] {
            let bucket = kv_capacity_bucket(len, hard);
            assert!(bucket >= len, "bucket {bucket} smaller than len {len}");
            assert!(bucket <= hard, "bucket {bucket} exceeds hard_max {hard}");
            assert!(bucket >= prev, "bucket {bucket} not monotonic at len {len}");
            prev = bucket;
        }
    }

    #[test]
    fn kv_capacity_bucket_never_exceeds_hard_max() {
        // The caller rejects this case with an actionable error, but the bucket
        // helper itself must never produce an over-limit allocation.
        assert_eq!(kv_capacity_bucket(40000, 32768), 32768);
    }

    #[test]
    fn copy_seq_prefix_preserves_strided_prefix_and_zeros_tail() {
        // Rank-4 [B=1, H=2, S, D=2] KV tensor, seq_axis = 2 (the common GQA
        // layout). Old cap 3 -> new cap 5, valid_len 2: each head's first
        // `valid_len*D` elements relocate to the new per-head stride; the rest
        // must be zero.
        let src_shape = [1i64, 2, 3, 2];
        let dst_shape = [1i64, 2, 5, 2];
        let src: Vec<f32> = vec![
            // head 0: pos0, pos1, pos2(dropped)
            0.0, 1.0, 10.0, 11.0, 20.0, 21.0, //
            // head 1: pos0, pos1, pos2(dropped)
            100.0, 101.0, 110.0, 111.0, 120.0, 121.0,
        ];
        let out = copy_seq_prefix(&src, &src_shape, &dst_shape, 2, 2);
        assert_eq!(out.len(), 2 * 5 * 2);
        // head 0 block occupies out[0..10] (dst stride = new_cap*D = 5*2).
        assert_eq!(&out[0..4], &[0.0, 1.0, 10.0, 11.0]);
        assert!(out[4..10].iter().all(|&v| v == 0.0));
        // head 1 block occupies out[10..20].
        assert_eq!(&out[10..14], &[100.0, 101.0, 110.0, 111.0]);
        assert!(out[14..20].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn copy_seq_prefix_contiguous_rank3_batch1() {
        // Rank-3 [B=1, S, D=2], seq_axis = 1: with batch 1 the prefix is
        // contiguous, so a single leading run is copied.
        let src_shape = [1i64, 3, 2];
        let dst_shape = [1i64, 8, 2];
        let src: Vec<u16> = vec![1, 2, 3, 4, 5, 6];
        let out = copy_seq_prefix(&src, &src_shape, &dst_shape, 1, 2);
        assert_eq!(out.len(), 16);
        assert_eq!(&out[0..4], &[1, 2, 3, 4]);
        assert!(out[4..].iter().all(|&v| v == 0));
    }

    #[test]
    fn grow_kv_value_cpu_preserves_prefix_byte_for_byte() {
        // End-to-end CPU grow path (on_device = false): build a host KV buffer,
        // grow it, and verify the valid prefix survives exactly while the new
        // tail is zeroed. The `null` env is never dereferenced on the CPU path.
        let old_shape = [1i64, 2, 4, 3]; // [B, H, S=4, D=3], seq_axis = 2
        let numel = 2 * 4 * 3;
        let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
        let old = Value::from_vec_f32(data.clone(), &old_shape).expect("old kv");

        let new_shape = [1i64, 2, 8, 3];
        let valid_len = 3usize;
        let grown = grow_kv_value(&old, &new_shape, 2, valid_len, GrowDevice::Host, None)
            .expect("grown kv");
        assert_eq!(grown.shape(), &new_shape);

        let grown_data = grown.to_vec_f32().expect("grown data");
        let expected = copy_seq_prefix(&data, &old_shape, &new_shape, 2, valid_len);
        assert_eq!(grown_data, expected);
        // Spot check head 1, position 2 (all dims): old flat [18..21] must land
        // at new head-1 stride offset 8*3 + 2*3 = 30.
        assert_eq!(&grown_data[30..33], &data[18..21]);
    }

    #[test]
    fn grow_kv_value_cpu_preserves_fp16_bits() {
        // The fp16 arm round-trips raw 16-bit patterns unchanged.
        let old_shape = [1i64, 1, 4, 2];
        let bits: Vec<u16> = vec![
            0x3c00, 0xc000, 0x4000, 0x4200, 0x4400, 0x4500, 0x0000, 0x0000,
        ];
        let old = Value::from_vec_f16_bits(bits.clone(), &old_shape).expect("old fp16 kv");

        let new_shape = [1i64, 1, 8, 2];
        let grown =
            grow_kv_value(&old, &new_shape, 2, 2, GrowDevice::Host, None).expect("grown fp16 kv");
        let grown_bits = grown.to_vec_f16_bits().expect("grown fp16 bits");
        // valid_len 2 => first 2 positions (4 elems) preserved, rest zero.
        assert_eq!(&grown_bits[0..4], &bits[0..4]);
        assert!(grown_bits[4..].iter().all(|&v| v == 0));
    }

    #[test]
    fn plan_kv_prefix_copy_matches_strided_byte_layout() {
        // Rank-4 [B=1, H=2, S, D=3] fp16 (elem_size 2), seq_axis = 2.
        // Old cap 4 -> new cap 8, valid_len 3: one contiguous run per head, at
        // the per-head strides, sized in bytes.
        let src_shape = [1i64, 2, 4, 3];
        let dst_shape = [1i64, 2, 8, 3];
        let elem_size = 2;
        let plan = plan_kv_prefix_copy(&src_shape, &dst_shape, 2, 3, elem_size);

        let inner = 3; // head_dim
        let run = 3 * inner * elem_size; // valid_len * inner * elem_size = 18
        let src_stride = 4 * inner * elem_size; // old_cap * inner * elem_size = 24
        let dst_stride = 8 * inner * elem_size; // new_cap * inner * elem_size = 48
        assert_eq!(plan.total_bytes, 2 * dst_stride); // blocks * dst_stride = 96
        assert_eq!(
            plan.segments,
            vec![
                KvCopySegment {
                    src_offset: 0,
                    dst_offset: 0,
                    len: run
                },
                KvCopySegment {
                    src_offset: src_stride,
                    dst_offset: dst_stride,
                    len: run,
                },
            ]
        );
    }

    #[test]
    fn plan_kv_prefix_copy_bytes_agree_with_copy_seq_prefix() {
        // The byte plan must relocate exactly the elements copy_seq_prefix
        // moves. Apply the plan to a flat f32 byte buffer and compare against
        // the element-level host rearrangement.
        let src_shape = [1i64, 2, 3, 2];
        let dst_shape = [1i64, 2, 5, 2];
        let elem_size = std::mem::size_of::<f32>();
        let src: Vec<f32> = (0..(2 * 3 * 2)).map(|i| i as f32).collect();
        let expected = copy_seq_prefix(&src, &src_shape, &dst_shape, 2, 2);

        let plan = plan_kv_prefix_copy(&src_shape, &dst_shape, 2, 2, elem_size);
        assert_eq!(plan.total_bytes, expected.len() * elem_size);
        let src_bytes: Vec<u8> = src.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut dst_bytes = vec![0u8; plan.total_bytes];
        for seg in &plan.segments {
            dst_bytes[seg.dst_offset..seg.dst_offset + seg.len]
                .copy_from_slice(&src_bytes[seg.src_offset..seg.src_offset + seg.len]);
        }
        let expected_bytes: Vec<u8> = expected.iter().flat_map(|v| v.to_ne_bytes()).collect();
        assert_eq!(dst_bytes, expected_bytes);
    }

    #[test]
    fn plan_kv_prefix_copy_empty_prefix_has_no_segments() {
        // valid_len 0 (fresh prefill before any token): nothing to copy, but the
        // whole destination is still sized so it can be zeroed.
        let src_shape = [1i64, 2, 4, 3];
        let dst_shape = [1i64, 2, 8, 3];
        let plan = plan_kv_prefix_copy(&src_shape, &dst_shape, 2, 0, 2);
        assert!(plan.segments.is_empty());
        assert_eq!(plan.total_bytes, 2 * 8 * 3 * 2);
    }
}
