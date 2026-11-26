use super::*;
use onnx_runtime_ep_api::DevicePtr;
use onnx_runtime_ir::DeviceId;

struct Backing {
    words: Vec<u128>,
    shape: [usize; 3],
    strides: [i64; 3],
    dtype: DataType,
    base: usize,
    offset: usize,
}
impl Backing {
    fn new(dtype: DataType, shape: [usize; 3], bytes: &[u8], mode: usize) -> Self {
        let [e, n, c] = shape;
        let width = dtype.byte_size();
        let dense = mode == 0 || mode == 4;
        let step = if dense { 1 } else { 2 };
        let row = if dense { c } else { 2 * c + 1 };
        let expert = if dense { n * row } else { n * row + 3 };
        let reversed = matches!(mode, 2 | 3);
        let origin = if reversed && c > 0 { (c - 1) * step } else { 0 };
        let base = usize::from(mode == 3 || mode == 4);
        let strides = [
            expert as i64,
            row as i64,
            if reversed {
                -(step as i64)
            } else {
                step as i64
            },
        ];
        let span = (e.saturating_sub(1) * expert
            + n.saturating_sub(1) * row
            + c.saturating_sub(1) * step
            + 1)
            * width
            + base;
        let mut words = vec![0xa5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5u128; span.div_ceil(16).max(1)];
        // SAFETY: byte writes preserve valid u128 storage and remain within the
        // allocated span. No tensor view or typed value reference exists yet.
        let storage = unsafe {
            std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), words.len() * 16)
        };
        for i in 0..e {
            for j in 0..n {
                for k in 0..c {
                    let logical = ((i * n + j) * c + k) * width;
                    let offset = (origin as i64
                        + i as i64 * strides[0]
                        + j as i64 * strides[1]
                        + k as i64 * strides[2]) as usize
                        * width
                        + base;
                    storage[offset..offset + width]
                        .copy_from_slice(&bytes[logical..logical + width]);
                }
            }
        }
        Self {
            words,
            shape,
            strides,
            dtype,
            base,
            offset: origin * width,
        }
    }
    fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(unsafe { self.words.as_ptr().cast::<u8>().add(self.base).cast() }),
            self.dtype,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
        .with_byte_offset(self.offset)
    }
}
fn scale_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

#[test]
fn borrowed_dense_padded_reversed_and_packed_banks_match_scalar_weight_bits() {
    for (depth, block) in [(3usize, 1usize), (6, 3), (18, 6), (32, 16)] {
        let (experts, width) = (2, 3);
        let blocks = depth / block;
        let codes = (0..experts * width * depth.div_ceil(2))
            .map(|i| 0x10u8.wrapping_add((i * 29) as u8))
            .collect::<Vec<_>>();
        let scales = (0..experts * width * blocks)
            .map(|i| (1 + (i % 7)) as f32 * 0.125)
            .collect::<Vec<_>>();
        let zeros = (0..experts * width * blocks.div_ceil(2))
            .map(|i| 0x21u8.wrapping_add((i * 37) as u8))
            .collect::<Vec<_>>();
        for mode in 0..5 {
            for zero_points in [false, true] {
                let c = Backing::new(
                    DataType::Uint8,
                    [experts, width, depth.div_ceil(2)],
                    &codes,
                    mode,
                );
                let s = Backing::new(
                    DataType::Float32,
                    [experts, width, blocks],
                    &scale_bytes(&scales),
                    mode,
                );
                let z = Backing::new(
                    DataType::Uint8,
                    [experts, width, blocks.div_ceil(2)],
                    &zeros,
                    mode,
                );
                let (cv, sv, zv) = (c.view(), s.view(), z.view());
                let bank = AffineBank::prepare(
                    &cv,
                    &sv,
                    zero_points.then_some(&zv),
                    [experts, width, depth],
                    block,
                    7,
                )
                .and_then(|bank| bank.validate())
                .unwrap();
                for special in [false, true] {
                    let input = (0..depth)
                        .map(|i| {
                            if special {
                                [
                                    f32::from_bits(0x7fc12345),
                                    f32::INFINITY,
                                    -0.,
                                    f32::NEG_INFINITY,
                                    1.,
                                ][i % 5]
                            } else {
                                (i as f32 % 7. - 3.) * 0.25
                            }
                        })
                        .collect::<Vec<_>>();
                    for row in 0..experts * width {
                        let mut expected = 0f32;
                        for (column, &x) in input.iter().enumerate() {
                            let p = column / block;
                            let q = (codes[row * depth.div_ceil(2) + column / 2]
                                >> ((column % 2) * 4))
                                & 15;
                            let zero = if zero_points {
                                (zeros[row * blocks.div_ceil(2) + p / 2] >> ((p % 2) * 4)) & 15
                            } else {
                                7
                            };
                            let weight = f32::from(i16::from(q) - i16::from(zero))
                                * scales[row * blocks + p];
                            expected += x * weight;
                        }
                        let actual = bank.dot(row, &input);
                        let context = format!(
                            "depth={depth} block={block} mode={mode} zeros={zero_points} row={row} special={special}"
                        );
                        if expected.is_nan() {
                            // This is an arithmetic result, not a raw copy. IEEE-754
                            // and ONNX do not prescribe which NaN payload survives
                            // when the fold encounters multiple NaN-producing terms.
                            assert!(actual.is_nan(), "{context}: got {actual:?}");
                        } else {
                            assert_eq!(actual.to_bits(), expected.to_bits(), "{context}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn every_bank_is_host_validated_before_any_scale_read() {
    let codes = Backing::new(DataType::Uint8, [1, 2, 2], &[0x12; 4], 0);
    let scales = Backing::new(DataType::Float32, [1, 2, 2], &scale_bytes(&[1.; 4]), 0);
    let zeros = Backing::new(DataType::Uint8, [1, 2, 1], &[0x34; 2], 0);
    for slot in 0..3 {
        let mut views = [codes.view(), scales.view(), zeros.view()];
        // No constructor may read scale bytes until the last bank's device is
        // checked. If that guard regresses, this pointer cannot be dereferenced.
        views[1].data = DevicePtr(4usize as *const _);
        views[slot].device = DeviceId::cuda(0);
        views[slot].data = DevicePtr(4usize as *const _);
        let error =
            match AffineBank::prepare(&views[0], &views[1], Some(&views[2]), [1, 2, 4], 2, 7)
                .and_then(|bank| bank.validate())
            {
                Ok(_) => panic!("non-Host bank must fail"),
                Err(e) => e,
            };
        let message = error.to_string();
        assert!(message.contains("Host-accessible"), "{message}");
        assert!(
            message.contains(["codes", "scales", "zero points"][slot]),
            "{message}"
        );
    }
}

#[test]
fn malformed_scales_shapes_and_address_spans_fail_with_context() {
    let codes = Backing::new(DataType::Uint8, [1, 2, 2], &[0x12; 4], 0);
    let zeros = Backing::new(DataType::Uint8, [1, 2, 1], &[0x34; 2], 0);
    for bad in [0., -0., -1., f32::NAN, f32::INFINITY] {
        let scales = Backing::new(
            DataType::Float32,
            [1, 2, 2],
            &scale_bytes(&[1., 1., bad, 1.]),
            0,
        );
        let (cv, sv, zv) = (codes.view(), scales.view(), zeros.view());
        let error = match AffineBank::prepare(&cv, &sv, Some(&zv), [1, 2, 4], 2, 7)
            .and_then(|bank| bank.validate())
        {
            Ok(_) => panic!("bad scale must fail"),
            Err(e) => e,
        };
        let message = error.to_string();
        assert!(
            message.contains("row 1, block 0") && message.contains("positive and finite"),
            "{message}"
        );
    }
    let scales = Backing::new(DataType::Float32, [1, 2, 2], &scale_bytes(&[1.; 4]), 0);
    for case in 0..5 {
        let mut views = [codes.view(), scales.view(), zeros.view()];
        match case {
            0 => views[0].strides = &[4, i64::MAX, 1],
            1 => views[1].strides = &[4, 2, i64::MIN],
            2 => views[2].data = DevicePtr((usize::MAX - 1) as *const _),
            3 => views[1].byte_offset = usize::MAX - 3,
            _ => views[1].shape = &[1, 1, 4],
        }
        let error =
            match AffineBank::prepare(&views[0], &views[1], Some(&views[2]), [1, 2, 4], 2, 7)
                .and_then(|bank| bank.validate())
            {
                Ok(_) => panic!("case {case} must fail"),
                Err(e) => e,
            };
        let message = error.to_string();
        assert!(
            message.contains("span") || message.contains("do not match"),
            "case={case}: {message}"
        );
    }
}

#[test]
fn block_parameters_preserve_scalar_rounding_across_odd_boundaries_and_mixed_storage() {
    for (depth, block) in [(1usize, 1usize), (6, 3), (65, 5), (128, 32)] {
        let (experts, width) = (2, 3);
        let blocks = depth / block;
        let codes = (0..experts * width * depth.div_ceil(2))
            .map(|i| (i.wrapping_mul(73) ^ (i / 3)) as u8)
            .collect::<Vec<_>>();
        let zeros = (0..experts * width * blocks.div_ceil(2))
            .map(|i| (i.wrapping_mul(29) ^ 0xb3) as u8)
            .collect::<Vec<_>>();
        for extreme in [false, true] {
            let scales = (0..experts * width * blocks)
                .map(|i| {
                    if extreme {
                        [f32::from_bits(1), f32::MIN_POSITIVE, f32::MAX, 0.1][i % 4]
                    } else {
                        [0.1, 1. / 3., 1.0000001, 0.7][i % 4]
                    }
                })
                .collect::<Vec<_>>();
            let input = (0..depth)
                .map(|i| [-0., 0., 0.1, -1. / 7., 12345.67, f32::MIN_POSITIVE][i % 6])
                .collect::<Vec<_>>();
            // Codes, scales and zeros can independently be dense, padded,
            // reversed or unaligned; the shared addressing mode must handle all.
            for modes in [[0, 0, 0], [0, 2, 4], [1, 0, 3], [4, 4, 0], [2, 1, 2]] {
                let c = Backing::new(
                    DataType::Uint8,
                    [experts, width, depth.div_ceil(2)],
                    &codes,
                    modes[0],
                );
                let s = Backing::new(
                    DataType::Float32,
                    [experts, width, blocks],
                    &scale_bytes(&scales),
                    modes[1],
                );
                let z = Backing::new(
                    DataType::Uint8,
                    [experts, width, blocks.div_ceil(2)],
                    &zeros,
                    modes[2],
                );
                let (cv, sv, zv) = (c.view(), s.view(), z.view());
                for has_zero in [false, true] {
                    let bank = AffineBank::prepare(
                        &cv,
                        &sv,
                        has_zero.then_some(&zv),
                        [experts, width, depth],
                        block,
                        7,
                    )
                    .unwrap()
                    .validate()
                    .unwrap();
                    for row in 0..experts * width {
                        // Frozen column-wise oracle: deliberately derive the
                        // parameter index independently for each scalar term.
                        let mut expected = 0.0f32;
                        for (col, &x) in input.iter().enumerate() {
                            let p = col / block;
                            let q =
                                (codes[row * depth.div_ceil(2) + col / 2] >> ((col % 2) * 4)) & 15;
                            let zero = if has_zero {
                                (zeros[row * blocks.div_ceil(2) + p / 2] >> ((p % 2) * 4)) & 15
                            } else {
                                7
                            };
                            let weight = f32::from(i16::from(q) - i16::from(zero))
                                * scales[row * blocks + p];
                            expected += x * weight;
                        }
                        let actual = bank.dot(row, &input);
                        let context = format!(
                            "depth={depth}, block={block}, modes={modes:?}, zero={has_zero}, extreme={extreme}, row={row}"
                        );
                        if expected.is_nan() {
                            assert!(actual.is_nan(), "{context}");
                        } else {
                            assert_eq!(actual.to_bits(), expected.to_bits(), "{context}");
                        }
                    }
                }
            }
        }
    }
}
