//! Native DequantizePacked conformance, with no CPU fallback for device runs.
//!
//! The deterministic corpus recipe and GGML oracle SHA-256 digests are adapted
//! from ONNX tests/python/dequantize_packed_test.py at
//! 622adf94d4e720142950aa8178f7357d52e63db0.
//! Copyright (c) ONNX Project Contributors. SPDX-License-Identifier: Apache-2.0

mod common;

use common::{Tensor, input, require_cuda, run_cpu, run_cuda};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType};
use sha2::{Digest, Sha256};

// Format suffix, values per block, packed bytes per block, half-scale offsets,
// and digest of the 64 little-endian f32 blocks from the pinned GGML oracle.
type OracleCase = (&'static str, usize, usize, &'static [usize], &'static str);
#[rustfmt::skip]
const CASES: &[OracleCase] = &[
    ("q4_0", 32, 18, &[0], "09f0f0b2d729e51f2f646e693cc3c6e18b51d795f4b6cbe73097e9be7d87b6e0"),
    ("q4_1", 32, 20, &[0,2], "1b44f52d3d6eb9725387950ae0bd486124a8fd1940edca23eab00bd311132662"),
    ("q5_0", 32, 22, &[0], "31398b6c3cac68a02d7507c3b18ca0793edd16b3110210a0249f88c56e5abc31"),
    ("q5_1", 32, 24, &[0,2], "e8328a8f385ad8b9e37ef5b0b97e666bb0cd06ff61be6b1efb971d2098fab9e5"),
    ("q8_0", 32, 34, &[0], "2b4f748be8ea71681a7dcf29556f393a9d20d931e0f34eec5116db312ab1c6bf"),
    ("q8_1", 32, 36, &[0], "12c88d0aa1a44a5a6dde0d51fad26d7e021f3ac6af8e86b5f24be6fcee211664"),
    ("q2_k", 256, 84, &[80,82], "66915016ab9716dbb720328e90de144109f7ffc1b94d1a071946a3ef559a9143"),
    ("q3_k", 256, 110, &[108], "9e556592671f4e83889fd79ee42509cd4d93353727950ecc2712f989b1e989fe"),
    ("q4_k", 256, 144, &[0,2], "a7a2145b24199ff10b9198c38cac8bc558bf1cdc205f3edaa8ac00ec3cae01ac"),
    ("q5_k", 256, 176, &[0,2], "075f87db3496c7a5742fe2e7225bcecd08dc2b5ec16bdfee9fb80562bc55bedf"),
    ("q6_k", 256, 210, &[208], "8ee3fb3409c3accaf8db9fdb1aeed9caaf0a195d1cc4c93ebf848b61e8047ca9"),
    ("q8_k", 256, 292, &[], "96100a02d884e76c76fe40589cc81e6a8564670002541cc73c96b4dfd0df8006"),
    ("iq2_xxs", 256, 66, &[0], "681174e4bd8c22e1000d5f7f30460173742b7daab8f2277e364c3f9891c332ac"),
    ("iq2_xs", 256, 74, &[0], "0f6f01226d6e1f01a24ba36d11b4854af31e0da8149c2a039d29b231ba8e27fa"),
    ("iq3_xxs", 256, 98, &[0], "5d823365863f15bae4877cd80f09f6472c2c06ab0bec8b329eac9c15ad3a7dab"),
    ("iq1_s", 256, 50, &[0], "a56cb261436667ddb8fe7d80e845ee4f7234b9fc4faf4b9df653868554264611"),
    ("iq4_nl", 32, 18, &[0], "07db142dcc9a02c0e22aa1137f673e791f7eaf8e6ef925cebb7d2be661b7e9c5"),
    ("iq3_s", 256, 110, &[0], "1f5ae32b37d4272f52cda0a718982e287368c5b8970f9c3faee7db518c10ba00"),
    ("iq2_s", 256, 82, &[0], "b547a636de8076d309194c7a88234dae1ee8e5452d7f24a573254139447ee544"),
    ("iq4_xs", 256, 136, &[0], "691fb3a9f7266598a48b64d783702c012127a9a983c847dacdecf7bef2856100"),
    ("iq1_m", 256, 56, &[], "96889e22b60e45700add104bc47d5280cf608af7f9104a5fd6212de7abefdc3e"),
    ("tq1_0", 256, 54, &[52], "a191edf81615eb82787f56545027c6680c287510293e35ec4f602ddb84f30e59"),
    ("tq2_0", 256, 66, &[64], "7c9ca5cf4a7877d79ee1112a48c216a10a2b52f751c66d5a4f6ce036c236b732"),
    ("mxfp4", 32, 17, &[], "a259fc1ca83ab1225afd5daeb12da0ca2f1115ef182b48117aedfd254f37da82"),
    ("nvfp4", 64, 36, &[], "50f826780120a76023dac805e84bed3688aff1a165cf8a2499ced857cbc8949e"),
    ("q1_0", 128, 18, &[0], "9fb4de79ccb7e86f0803f11b83e6c1d8d54d3d20c1b44a65573074f87c55d0e5"),
    ("q2_0", 64, 18, &[0], "057c75b6666c8cc1d2de74644fdf630872e0baf042b50024312feb3facd3419b"),
];

fn corpus(name: &str, bytes: usize, offsets: &[usize]) -> Vec<u8> {
    let scales = [
        0.3125f64,
        -0.625,
        0.0,
        1.125,
        0.1,
        -3.25,
        2f64.powi(-24),
        65504.0,
    ];
    let bits = |i| half::f16::from_f64(scales[i]).to_bits();
    let mut packed = Vec::new();
    for i in 0..64 {
        let mut block: Vec<u8> = (0..bytes)
            .map(|j| (j * 73 + i * 29 + j / 7 * 17 + i / 3 * 11) as u8)
            .collect();
        for (k, &offset) in offsets.iter().enumerate() {
            block[offset..offset + 2].copy_from_slice(&bits((i + k) % 8).to_le_bytes());
        }
        match name {
            "q8_k" => block[..4].copy_from_slice(&((scales[i % 8] / 3.0) as f32).to_le_bytes()),
            "iq1_m" => {
                for k in 0..4 {
                    let offset = 48 + 2 * k;
                    block[offset + 1] =
                        (block[offset + 1] & 15) | (((bits(i % 8) >> (4 * k)) & 15) as u8) << 4;
                }
            }
            "mxfp4" => block[0] = (i * 17) as u8,
            _ => {}
        }
        packed.extend(block);
    }
    packed
}

fn run(
    cuda: Option<&CudaExecutionProvider>,
    name: &str,
    packed: &[u8],
    shape: &[usize],
    dtype: Option<DataType>,
) -> Vec<u8> {
    let inputs: [Tensor; 1] = [input(DataType::Uint8, &[packed.len()], packed)];
    let outputs = [(dtype.unwrap_or(DataType::Float32), shape.to_vec())];
    let mut attrs = vec![
        (
            "format",
            Attribute::String(format!("gguf:{name}").into_bytes()),
        ),
        (
            "shape",
            Attribute::Ints(shape.iter().map(|&n| n as i64).collect()),
        ),
    ];
    if let Some(dtype) = dtype {
        attrs.push(("output_dtype", Attribute::Int(i64::from(dtype.to_onnx()))));
    }
    let result = match cuda {
        Some(ep) => run_cuda(ep, "DequantizePacked", "", 29, &inputs, &outputs, &attrs),
        None => run_cpu("DequantizePacked", "", 29, &inputs, &outputs, &attrs),
    }
    .pop()
    .unwrap();
    if cfg!(target_endian = "big") && cuda.is_none() {
        let mut result = result;
        for lane in result.chunks_exact_mut(outputs[0].0.byte_size()) {
            lane.reverse();
        }
        return result;
    }
    result
}

fn check_corpus(cuda: Option<&CudaExecutionProvider>) {
    for &(name, width, bytes, offsets, expected) in CASES {
        let packed = corpus(name, bytes, offsets);
        let actual = run(cuda, name, &packed, &[4, 8, 2 * width], None);
        assert_eq!(format!("{:x}", Sha256::digest(actual)), expected, "{name}");
    }
}

fn check_edges(cuda: Option<&CudaExecutionProvider>) {
    assert!(run(cuda, "q2_k", &[], &[2, 0, 256], None).is_empty());
    // Q4_1 lane order and ties-to-even when narrowing: 1 + 2^-11 is
    // halfway between f16 values. 1 + 8*2^-11 is halfway in bf16.
    let mut packed = vec![0x00, 0x10, 0x00, 0x3c];
    packed.extend([0x81; 16]);
    let expected16 = [0x3c00u16; 16].into_iter().chain([0x3c04; 16]);
    let expected_bf16 = [0x3f80u16; 32];
    for (dt, expected) in [
        (
            DataType::Float16,
            expected16.flat_map(u16::to_le_bytes).collect::<Vec<_>>(),
        ),
        (
            DataType::BFloat16,
            expected_bf16
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        ),
        (
            DataType::Float64,
            [1.00048828125f64; 16]
                .into_iter()
                .chain([1.00390625; 16])
                .flat_map(f64::to_le_bytes)
                .collect(),
        ),
    ] {
        assert_eq!(
            run(cuda, "q4_1", &packed, &[32], Some(dt)),
            expected,
            "{dt:?}"
        );
    }

    // Special MXFP4 exponent: 255 is finite before multiplying by E2M1.
    // E8M0 exponent 0 must preserve its gradual-underflow subnormal.
    for (exponent, bits) in [(0, 0x0020_0000u32), (255, 0x7f00_0000)] {
        let mut packed = vec![exponent];
        packed.extend([0x91; 16]);
        let expected = [bits; 16]
            .into_iter()
            .chain([bits | 0x8000_0000; 16])
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(run(cuda, "mxfp4", &packed, &[32], None), expected);
    }

    // Q8_K uses a binary32 scale directly, ignores the auxiliary sums,
    // and multiplies before any dtype widening. Check signed zero,
    // subnormal multiplication, overflow, and arithmetic NaNs.
    for (scale, expected) in [
        (-0.0f32, [-0.0, 0.0, -0.0]),
        (
            f32::from_bits(1),
            [f32::from_bits(1), f32::from_bits(0x8000_0002), 0.0],
        ),
        (f32::MAX, [f32::MAX, f32::NEG_INFINITY, 0.0]),
        (f32::INFINITY, [f32::INFINITY, f32::NEG_INFINITY, f32::NAN]),
        (f32::NAN, [f32::NAN; 3]),
    ] {
        let mut packed = vec![0; 292];
        packed[..4].copy_from_slice(&scale.to_le_bytes());
        packed[4..7].copy_from_slice(&[1, 254, 0]);
        packed[260..].fill(0xff);
        let out = run(cuda, "q8_k", &packed, &[256], Some(DataType::Float64));
        for (lane, expected) in out.chunks_exact(8).zip(expected) {
            let value = f64::from_le_bytes(lane.try_into().unwrap());
            if expected.is_nan() {
                assert!(
                    value.is_nan(),
                    "expected NaN at scale {scale:?}, got {value:?}"
                );
            } else {
                assert_eq!(
                    value.to_bits(),
                    f64::from(expected).to_bits(),
                    "scale {scale:?}"
                );
            }
        }
    }

    // The full scale byte participates in NVFP4's special case: 0x7f is
    // zero, but 0xff is the largest magnitude. The sign bit otherwise
    // does not change subnormal scales.
    let mut packed = vec![0x7f, 0xff, 0x01, 0x81];
    packed.extend([0x91; 32]);
    let expected = [0.0f32, 240.0, 1.0 / 1024.0, 1.0 / 1024.0]
        .into_iter()
        .flat_map(|d| [d; 8].into_iter().chain([-d; 8]))
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(run(cuda, "nvfp4", &packed, &[64], None), expected);
}

#[test]
fn cpu_dequantize_packed_conformance() {
    check_corpus(None);
    check_edges(None);
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
fn cuda_dequantize_packed_conformance() {
    let ep = require_cuda();
    check_corpus(Some(&ep));
    check_edges(Some(&ep));
}
