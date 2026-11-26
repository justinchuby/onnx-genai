use super::*;
use crate::kernels::testutil::{Owned, PreparedTestKernel, Repacked};
use onnx_runtime_ep_api::DevicePtrMut;
use onnx_runtime_ir::{Attribute, NodeId};

fn node(format: &str) -> Node {
    let mut node = Node::new(NodeId(0), OP, vec![], vec![]);
    for (name, value) in [
        ("K", Attribute::Int(32)),
        ("N", Attribute::Int(2)),
        ("format", Attribute::String(format.as_bytes().to_vec())),
        ("activation_arithmetic", Attribute::String(b"q8_0".to_vec())),
        ("block_layout_version", Attribute::Int(2)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    node
}

#[test]
fn q8_factory_claim_and_create_enforce_the_same_numerical_contract() {
    let factory = BlockQuantizedMatMulFactory::new(Default::default(), 1 << 20);
    for format in ["q8_0", "gguf:q4_0"] {
        let mut n = node(format);
        assert!(factory.unsupported_reason(&n, 1, &[], &[]).is_none());
        assert!(factory.create(&n, &[]).is_ok());
        n.attributes
            .insert("block_layout_version".into(), Attribute::Int(1));
        assert!(
            factory
                .unsupported_reason(&n, 1, &[], &[])
                .unwrap()
                .contains("integer 2")
        );
        assert!(factory.create(&n, &[]).is_err());
    }
    let n = node("mxfp4");
    assert!(
        factory
            .unsupported_reason(&n, 1, &[], &[])
            .unwrap()
            .contains("float32")
    );
    assert!(factory.create(&n, &[]).is_err());
}

#[test]
fn explicit_q8_changes_the_real_kernel_result_for_all_storage_and_stride_forms() {
    let mut x = vec![0.; 32];
    x[0] = 1.;
    x[1] = 2f32.powi(-10);
    let mut w = vec![0u8; 68];
    w[..2].copy_from_slice(&half::f16::ONE.to_le_bytes());
    w[3] = 1;
    w[34..36].copy_from_slice(&half::f16::from_f32(2f32.powi(-11)).to_le_bytes());
    w[36] = 1;
    let b = Owned::u8(&[2, 1, 34], &w);
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        let (a, bias) = match dtype {
            DataType::Float32 => (Owned::f32(&[1, 32], &x), Owned::f32(&[2], &[0.; 2])),
            DataType::Float16 => (Owned::f16(&[1, 32], &x), Owned::f16(&[2], &[0.; 2])),
            DataType::BFloat16 => (Owned::bf16(&[1, 32], &x), Owned::bf16(&[2], &[0.; 2])),
            _ => unreachable!(),
        };
        for mode in 0..6 {
            let a = Repacked::new(&a, mode);
            let b = Repacked::new(&b, mode);
            let bias = Repacked::new(&bias, mode);
            for arithmetic in [ActivationArithmetic::Float32, ActivationArithmetic::Q8_0] {
                let kernel = BlockQuantizedMatMulKernel {
                    arithmetic,
                    k: 32,
                    n: 2,
                    format: BlockFormat::Q8_0,
                    packed_b_constant: true,
                    weight_identity: DenseWeightIdentity::default(),
                    prepared_projection: std::cell::Cell::new(None),
                    weight_cache: WeightCache::new(Default::default(), 1 << 20),
                };
                let mut y = match dtype {
                    DataType::Float32 => Owned::zeros_f32(&[1, 2]),
                    DataType::Float16 => Owned::f16(&[1, 2], &[0.; 2]),
                    DataType::BFloat16 => Owned::bf16(&[1, 2], &[0.; 2]),
                    _ => unreachable!(),
                };
                for _ in 0..2 {
                    kernel
                        .execute_prepared(&[a.view(), b.view(), bias.view()], &mut [y.view_mut()])
                        .unwrap();
                    let values = match dtype {
                        DataType::Float32 => y.to_f32(),
                        DataType::Float16 => y.to_f16_as_f32(),
                        DataType::BFloat16 => y.to_bf16_as_f32(),
                        _ => unreachable!(),
                    };
                    match arithmetic {
                        ActivationArithmetic::Float32 => {
                            assert_eq!(values, [2f32.powi(-10), 2f32.powi(-11)])
                        }
                        ActivationArithmetic::Q8_0 => {
                            assert_eq!(values[0], 0.);
                            assert!(values[1] > 0.);
                            assert_eq!(kernel.weight_cache.activity(), (0, 0));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn q8_rejects_missing_short_or_misaligned_workspace_before_payload_and_can_retry() {
    let kernel = BlockQuantizedMatMulKernel {
        arithmetic: ActivationArithmetic::Q8_0,
        k: 32,
        n: 2,
        format: BlockFormat::Q8_0,
        packed_b_constant: false,
        weight_identity: DenseWeightIdentity::default(),
        prepared_projection: std::cell::Cell::new(None),
        weight_cache: WeightCache::new(Default::default(), 1 << 20),
    };
    let a = Owned::f32(&[1, 32], &[1.; 32]);
    let b = Owned::u8(&[2, 1, 34], &packed_for_format(BlockFormat::Q8_0, 1, 2, 32));
    let mut y = Owned::f32(&[1, 2], &[37.; 2]);
    let mut backing = [91u128; 8];
    // Valid owning buffers but impossible byte origins: any materialization
    // would fail the source bounds guard. Scratch must win before that read.
    let mut bad_a = a.view();
    bad_a.byte_offset = usize::MAX;
    let views = [bad_a, b.view()];
    let metadata = [
        TensorMetadata::new(DataType::Float32, &[1, 32], true),
        TensorMetadata::new(DataType::Uint8, &[2, 1, 34], true),
    ];
    for workspace in [
        None,
        Some(WorkspaceView::new(
            DevicePtrMut(backing.as_mut_ptr().cast()),
            32,
        )),
        Some(WorkspaceView::new(
            DevicePtrMut(unsafe { backing.as_mut_ptr().cast::<u8>().add(1).cast() }),
            64,
        )),
    ] {
        kernel
            .workspace_requirement_for_execution(&views, &[y.view_mut()], &metadata, &[], None)
            .unwrap()
            .require_ready()
            .unwrap();
        let error = kernel
            .execute_with_workspace(&views, &mut [y.view_mut()], workspace)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("BlockQuantizedMatMul execution workspace binding"),
            "{error}"
        );
        assert_eq!(y.to_f32(), [37.; 2]);
        assert_eq!(backing, [91; 8]);
    }
    kernel
        .execute_prepared(&[a.view(), b.view()], &mut [y.view_mut()])
        .unwrap();
    assert!(y.to_f32().iter().all(|x| x.is_finite()));
}
