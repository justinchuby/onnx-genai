//! Validated logical packed bytes. No backing owner, cache or admission policy.
use super::decoded_projection::ProjectionScratch;
use crate::strided::{elem_offset, next_index, validate_host_storage};
use onnx_runtime_ep_api::{EpError, Result, TensorView};
use smallvec::SmallVec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PackedByteDigest {
    pub(super) bytes: usize,
    pub(super) hash: u64,
}

/// Source backing stays immutable and live through the borrowed tensor view.
/// Construction proves representable addresses; allocation bounds are supplied
/// by the owning EP, just as for other CPU tensor readers.
#[derive(Debug)]
pub(super) struct PackedByteSource<'a> {
    #[cfg(test)]
    op: &'static str,
    origin: *const u8,
    shape: &'a [usize],
    strides: &'a [i64],
    len: usize,
    dense: bool,
}

impl<'a> PackedByteSource<'a> {
    pub(super) fn full(op: &'static str, view: &'a TensorView<'_>) -> Result<Self> {
        let len = validate_host_storage::<u8>(op, "packed bytes", view)?;
        Ok(Self {
            #[cfg(test)]
            op,
            // Empty geometry must not perform origin/offset pointer arithmetic.
            origin: if len == 0 {
                view.data.0.cast()
            } else {
                view.data_ptr::<u8>()
            },
            shape: view.shape,
            strides: view.strides,
            len,
            dense: onnx_runtime_ir::is_contiguous(view.shape, view.strides),
        })
    }

    pub(super) fn expert(
        op: &'static str,
        view: &'a TensorView<'_>,
        expert: usize,
    ) -> Result<Self> {
        validate_host_storage::<u8>(op, "packed expert bank", view)?;
        if view.shape.len() != 4 || view.strides.len() != 4 {
            return Err(EpError::KernelFailed(format!(
                "{op}: packed expert bank must have rank four, got shape {:?} and strides {:?}; bind [experts, outputs, blocks, block_bytes]",
                view.shape, view.strides
            )));
        }
        if expert >= view.shape[0] {
            return Err(EpError::KernelFailed(format!(
                "{op}: packed expert index {expert} is outside {} experts; select an existing expert",
                view.shape[0]
            )));
        }
        let shape = &view.shape[1..];
        let len = onnx_runtime_ir::checked_numel(shape).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "{op}: packed expert shape {shape:?} overflows byte indexing; reduce projection dimensions"
            ))
        })?;
        let origin = if len == 0 {
            view.data.0.cast()
        } else {
            let offset = (expert as i128)
                .checked_mul(i128::from(view.strides[0]))
                .and_then(|n| isize::try_from(n).ok())
                .ok_or_else(|| EpError::KernelFailed(format!(
                    "{op}: packed expert {expert} with stride {} has an unaddressable origin; bind representable expert strides",
                    view.strides[0]
                )))?;
            // SAFETY: the full Host span proof includes every expert's origin.
            // The owning EP supplies complete allocation bounds and lifetime.
            unsafe { view.data_ptr::<u8>().offset(offset) }
        };
        Ok(Self {
            #[cfg(test)]
            op,
            origin,
            shape,
            strides: &view.strides[1..],
            len,
            dense: Self::expert_layout_is_dense(view),
        })
    }

    /// Pure descriptor classification: outer expert padding/reversal does not
    /// affect whether one selected expert can be borrowed as contiguous bytes.
    pub(super) fn expert_layout_is_dense(view: &TensorView<'_>) -> bool {
        view.shape.len() == 4
            && view.strides.len() == 4
            && onnx_runtime_ir::is_contiguous(&view.shape[1..], &view.strides[1..])
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Descriptor-only sizing: no payload slice or read before admission.
    pub(super) fn copy_bytes(&self) -> usize {
        if self.dense { 0 } else { self.len }
    }

    pub(super) fn borrowed(&self) -> Option<&'a [u8]> {
        if self.len == 0 {
            return Some(&[]);
        }
        if !self.dense {
            return None;
        }
        // SAFETY: construction proves this exact source span; no whole-bank
        // slice is fabricated for padded/reversed expert-major storage.
        Some(unsafe { std::slice::from_raw_parts(self.origin, self.len) })
    }

    fn walk(&self, mut consume: impl FnMut(usize, u8)) {
        if self.len == 0 {
            return;
        }
        if let Some(bytes) = self.borrowed() {
            for (index, &byte) in bytes.iter().enumerate() {
                consume(index, byte);
            }
        } else {
            let mut coordinates = SmallVec::<[usize; 4]>::new();
            coordinates.resize(self.shape.len(), 0);
            for index in 0..self.len {
                let offset = elem_offset(self.strides, &coordinates);
                // SAFETY: the checked full span bounds every logical coordinate;
                // u8 requires no extra alignment and is valid at each byte.
                let byte = unsafe { self.origin.offset(offset).read() };
                consume(index, byte);
                next_index(self.shape, &mut coordinates);
            }
        }
    }

    pub(super) fn digest(&self) -> Result<PackedByteDigest> {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        self.walk(|_, byte| {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        });
        Ok(PackedByteDigest {
            bytes: self.len,
            hash,
        })
    }

    pub(super) fn read<'b>(&'b self, lane: &'b mut ProjectionScratch<'_>) -> Result<&'b [u8]> {
        if let Some(bytes) = self.borrowed() {
            return Ok(bytes);
        }
        // Capacity rejection and the f32-proof reset precede every source read.
        let target = lane.byte_storage(self.len)?;
        self.walk(|index, byte| {
            target[index].write(byte);
        });
        // SAFETY: the infallible validated walker wrote every exact-prefix byte.
        // This short shared borrow prevents lane mutation until consumption ends.
        Ok(unsafe { std::slice::from_raw_parts(target.as_ptr().cast::<u8>(), self.len) })
    }

    /// Test-only materialization oracle; execution always uses an admitted lane.
    #[cfg(test)]
    pub(super) fn owned(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(self.len).map_err(|source| {
            EpError::Memory(onnx_runtime_memory_api::MemoryError::AllocationFailed {
                tier: "host", requested: self.len as u64,
                reason: format!("{}: cannot materialize {} packed bytes: {source}; reduce projection dimensions or concurrent work", self.op, self.len),
            })
        })?;
        self.walk(|_, byte| bytes.push(byte));
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::DevicePtr;
    use onnx_runtime_ir::{DataType, DeviceId};

    fn legacy_hash(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    #[test]
    fn dense_full_and_padded_or_reversed_experts_borrow_exact_origins() {
        let bytes = [0u8, 1, 2, 3, 99, 99, 99, 99, 4, 5, 6, 7];
        let full_view = TensorView::new(
            DevicePtr(bytes.as_ptr().cast()),
            DataType::Uint8,
            &[12],
            &[1],
            DeviceId::cpu(),
        );
        let full = PackedByteSource::full("packed test", &full_view).unwrap();
        assert_eq!(full.borrowed().unwrap(), bytes);
        assert_eq!(full.digest().unwrap().hash, legacy_hash(&bytes));
        for reverse in [false, true] {
            let strides = [if reverse { -8 } else { 8 }, 4, 2, 1];
            let view = TensorView::new(
                DevicePtr(bytes.as_ptr().cast()),
                DataType::Uint8,
                &[2, 1, 2, 2],
                &strides,
                DeviceId::cpu(),
            )
            .with_byte_offset(if reverse { 8 } else { 0 });
            assert!(PackedByteSource::expert_layout_is_dense(&view));
            for expert in 0..2 {
                let source = PackedByteSource::expert("packed test", &view, expert).unwrap();
                let physical = if reverse { 1 - expert } else { expert } * 8;
                let expected = &bytes[physical..physical + 4];
                let borrowed = source.borrowed().unwrap();
                assert_eq!(borrowed.as_ptr(), expected.as_ptr());
                assert_eq!(borrowed, expected);
                assert_eq!(
                    source.digest().unwrap(),
                    PackedByteDigest {
                        bytes: 4,
                        hash: legacy_hash(expected)
                    }
                );
            }
        }
    }

    #[test]
    fn strided_logical_hash_copy_and_owned_payload_agree() {
        let bytes = [0u8, 1, 99, 99, 4, 5];
        let view = TensorView::new(
            DevicePtr(bytes.as_ptr().cast()),
            DataType::Uint8,
            &[2, 2],
            &[4, -1],
            DeviceId::cpu(),
        )
        .with_byte_offset(1);
        let source = PackedByteSource::full("packed test", &view).unwrap();
        let expected = [1u8, 0, 5, 4];
        assert!(source.borrowed().is_none());
        assert_eq!(source.len(), expected.len());
        assert_eq!(
            source.digest().unwrap(),
            PackedByteDigest {
                bytes: 4,
                hash: legacy_hash(&expected)
            }
        );
        assert_eq!(source.owned().unwrap(), expected);
        let mut storage = [f32::NAN; 2];
        let mut lane = ProjectionScratch::test_binding(&mut storage);
        assert_eq!(source.read(&mut lane).unwrap(), expected);
        assert_eq!(source.read(&mut lane).unwrap(), expected);
        let expert_view = TensorView::new(
            view.data,
            DataType::Uint8,
            &[1, 2, 1, 2],
            &[8, 4, 4, -1],
            DeviceId::cpu(),
        )
        .with_byte_offset(1);
        let expert = PackedByteSource::expert("packed test", &expert_view, 0).unwrap();
        assert!(!PackedByteSource::expert_layout_is_dense(&expert_view));
        assert_eq!(expert.digest().unwrap(), source.digest().unwrap());
        assert_eq!(expert.read(&mut lane).unwrap(), expected);

        // The walker spills rather than imposing an inline-rank API limit.
        let higher_rank = TensorView::new(
            view.data,
            DataType::Uint8,
            &[2, 1, 1, 1, 1, 2],
            &[4, 0, 0, 0, 0, -1],
            DeviceId::cpu(),
        )
        .with_byte_offset(1);
        let source = PackedByteSource::full("packed test", &higher_rank).unwrap();
        assert_eq!(source.owned().unwrap(), expected);
        assert_eq!(source.digest().unwrap().hash, legacy_hash(&expected));
    }

    #[test]
    fn short_lane_refuses_before_dereferencing_source() {
        let view = TensorView::new(
            DevicePtr(std::ptr::dangling::<u128>().cast()),
            DataType::Uint8,
            &[2, 2],
            &[4, 2],
            DeviceId::cpu(),
        );
        let source = PackedByteSource::full("packed test", &view).unwrap();
        let mut storage = [];
        let mut lane = ProjectionScratch::test_binding(&mut storage);
        assert!(source.read(&mut lane).is_err());
    }

    #[test]
    fn invalid_host_dtype_span_and_expert_geometry_refuse_without_reads() {
        for kind in 0..4 {
            let mut view = TensorView::new(
                DevicePtr(std::ptr::dangling::<u128>().cast()),
                DataType::Uint8,
                &[4],
                &[1],
                DeviceId::cpu(),
            );
            match kind {
                0 => view.device = DeviceId::cuda(0),
                1 => view.dtype = DataType::Float32,
                2 => view.data = DevicePtr((usize::MAX - 1) as *const _),
                _ => view.byte_offset = usize::MAX,
            }
            let error = PackedByteSource::full("packed test", &view).unwrap_err();
            assert!(error.to_string().contains("packed test"), "{error}");
        }
        let bytes = [0u8; 4];
        let rank = TensorView::new(
            DevicePtr(bytes.as_ptr().cast()),
            DataType::Uint8,
            &[4],
            &[1],
            DeviceId::cpu(),
        );
        assert!(!PackedByteSource::expert_layout_is_dense(&rank));
        assert!(PackedByteSource::expert("packed test", &rank, 0).is_err());
        let view = TensorView::new(
            DevicePtr(bytes.as_ptr().cast()),
            DataType::Uint8,
            &[1, 1, 2, 2],
            &[4, 4, 2, 1],
            DeviceId::cpu(),
        );
        assert!(PackedByteSource::expert("packed test", &view, 1).is_err());
        let overflow = TensorView::new(
            DevicePtr(bytes.as_ptr().cast()),
            DataType::Uint8,
            &[usize::MAX, 2],
            &[1, 1],
            DeviceId::cpu(),
        );
        assert!(PackedByteSource::full("packed test", &overflow).is_err());
    }

    #[test]
    fn empty_full_and_expert_sources_need_no_lane_or_pointer_offset() {
        let view = TensorView::new(
            DevicePtr(std::ptr::dangling::<u128>().cast()),
            DataType::Uint8,
            &[usize::MAX, 2, 0],
            &[0, 0, 0],
            DeviceId::cpu(),
        );
        let source = PackedByteSource::full("packed test", &view).unwrap();
        assert_eq!(source.borrowed(), Some([].as_slice()));
        assert_eq!(
            source.digest().unwrap(),
            PackedByteDigest {
                bytes: 0,
                hash: legacy_hash(&[])
            }
        );
        assert!(source.owned().unwrap().is_empty());
        let expert = TensorView::new(
            view.data,
            DataType::Uint8,
            &[2, 1, 0, 34],
            &[i64::MIN, 0, 34, 1],
            DeviceId::cpu(),
        );
        let source = PackedByteSource::expert("packed test", &expert, 1).unwrap();
        let mut storage = [];
        let mut lane = ProjectionScratch::test_binding(&mut storage);
        assert!(source.read(&mut lane).unwrap().is_empty());
    }
}
