use crate::common::{self, input, raw};
use onnx_runtime_ir::{Attribute, DataType, RUNTIME_DOMAIN};

pub(crate) type Case = (
    &'static str,
    &'static str,
    u64,
    Vec<common::Tensor>,
    DataType,
    Vec<usize>,
    Vec<(&'static str, Attribute)>,
    Vec<u8>,
);

pub(crate) fn cases() -> [Case; 5] {
    use DataType::{Float32, Int32, Int64, Uint8};
    let mut packed_q8 = vec![0u8, 0x3c]; // FP16 scale = 1.
    packed_q8.extend([1u8; 32]);
    [
        (
            "Pack",
            "",
            29,
            vec![input(Uint8, &[4], &[0u8, 1, 2, 3])],
            Uint8,
            vec![1],
            vec![("bits", Attribute::Int(2))],
            vec![0xe4u8],
        ),
        (
            "Unpack",
            "",
            29,
            vec![input(Uint8, &[1], &[0xe4u8]), input(Int64, &[], &[4i64])],
            Uint8,
            vec![4],
            vec![("bits", Attribute::Int(2))],
            vec![0u8, 1, 2, 3],
        ),
        (
            "GroupedMatMul",
            "",
            29,
            vec![
                input(Float32, &[1, 2], &[1f32, 2.]),
                input(Float32, &[1, 2, 1], &[3f32, 4.]),
                input(Int32, &[1, 1], &[0i32]),
            ],
            Float32,
            vec![1, 1, 1],
            vec![],
            raw(&[11f32]),
        ),
        (
            "AffineQuantizedGroupedMatMul",
            RUNTIME_DOMAIN,
            1,
            vec![
                input(Float32, &[1, 2], &[1f32, 2.]),
                input(Uint8, &[1, 1, 1], &[0x43u8]),
                input(Int32, &[1, 1], &[0i32]),
                input(Float32, &[1, 1, 1], &[1f32]),
            ],
            Float32,
            vec![1, 1, 1],
            vec![
                ("depth", Attribute::Int(2)),
                ("block_size", Attribute::Int(2)),
                ("zero", Attribute::Int(0)),
            ],
            raw(&[11f32]),
        ),
        (
            "PackedGroupedMatMul",
            RUNTIME_DOMAIN,
            1,
            vec![
                input(Float32, &[1, 32], &[1f32; 32]),
                input(Uint8, &[34], &packed_q8),
                input(Int32, &[1, 1], &[0i32]),
            ],
            Float32,
            vec![1, 1, 1],
            vec![
                (
                    "format",
                    Attribute::String(
                        onnx_runtime_ir::dequantize_packed::Format::Q8_0
                            .name()
                            .as_bytes()
                            .to_vec(),
                    ),
                ),
                ("shape", Attribute::Ints(vec![1, 1, 32])),
            ],
            raw(&[32f32]),
        ),
    ]
}
