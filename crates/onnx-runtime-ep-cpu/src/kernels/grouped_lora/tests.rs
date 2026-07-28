//! Unit tests for the `GroupedLoraDelta` group-by-adapter dense kernel.

use std::sync::Arc;

use onnx_runtime_ep_api::{
    AdapterId, Kernel, KernelFactory, LoraFactorInput, LoraModuleId, LoraPoolRegistry,
    LoraWeightPool,
};
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId};

use super::super::testutil::Owned;
use super::GroupedLoraDeltaFactory;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Reference `delta[m,n] = scale * (x[m,k] @ a_t[k,r]) @ b_t[r,n]` in f64.
fn reference_delta(
    m: usize,
    k: usize,
    r: usize,
    n: usize,
    scale: f32,
    x: &[f32],
    a_t: &[f32],
    b_t: &[f32],
) -> Vec<f32> {
    let mut mid = vec![0.0f64; m * r];
    for i in 0..m {
        for j in 0..r {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += x[i * k + p] as f64 * a_t[p * r + j] as f64;
            }
            mid[i * r + j] = acc;
        }
    }
    let mut delta = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..r {
                acc += mid[i * r + p] * b_t[p * n + j] as f64;
            }
            delta[i * n + j] = (acc * scale as f64) as f32;
        }
    }
    delta
}

fn make_kernel(
    pool: Arc<LoraWeightPool>,
    k: usize,
    n: usize,
    module_id: u32,
    max_rank: usize,
) -> Box<dyn Kernel> {
    let pool_id = LoraPoolRegistry::global().register(pool);
    let mut node = Node::new(NodeId(0), "GroupedLoraDelta", vec![None, None], vec![]);
    node.attributes.insert("K".into(), Attribute::Int(k as i64));
    node.attributes.insert("N".into(), Attribute::Int(n as i64));
    node.attributes
        .insert("module_id".into(), Attribute::Int(module_id as i64));
    node.attributes
        .insert("max_rank".into(), Attribute::Int(max_rank as i64));
    node.attributes
        .insert("pool_id".into(), Attribute::Int(pool_id.0 as i64));
    GroupedLoraDeltaFactory.create(&node, &[]).unwrap()
}

fn admit_f32(
    pool: &mut LoraWeightPool,
    adapter: u64,
    module: u32,
    k: usize,
    r: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    scale: f32,
) {
    let a_bytes = f32_bytes(a);
    let b_bytes = f32_bytes(b);
    pool.admit(
        AdapterId(adapter),
        LoraModuleId(module),
        LoraFactorInput {
            dtype: DataType::Float32,
            rows: k,
            cols: r,
            bytes: &a_bytes,
        },
        LoraFactorInput {
            dtype: DataType::Float32,
            rows: r,
            cols: n,
            bytes: &b_bytes,
        },
        scale,
    )
    .unwrap();
}

#[test]
fn single_adapter_dense_matches_reference() {
    let (k, n, r) = (4, 3, 2);
    let scale = 0.5f32;
    let a: Vec<f32> = (0..k * r).map(|i| i as f32 * 0.1 - 0.3).collect();
    let b: Vec<f32> = (0..r * n).map(|i| i as f32 * -0.05 + 0.2).collect();

    let mut pool = LoraWeightPool::with_capacity_bytes(1 << 20);
    admit_f32(&mut pool, 0, 0, k, r, n, &a, &b, scale);
    let kernel = make_kernel(Arc::new(pool), k, n, 0, 8);

    let m = 3;
    let x: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.25 - 1.0).collect();
    let x_owned = Owned::f32(&[m, k], &x);
    let segments = Owned::i32(&[m], &[0, 0, 0]);
    let mut out = Owned::zeros_f32(&[m, n]);

    kernel
        .execute(&[x_owned.view(), segments.view()], &mut [out.view_mut()])
        .unwrap();

    let expected = reference_delta(m, k, r, n, scale, &x, &a, &b);
    for (g, e) in out.to_f32().iter().zip(&expected) {
        assert!(
            (g - e).abs() < 1e-5,
            "got {:?} expected {expected:?}",
            out.to_f32()
        );
    }
}

#[test]
fn null_adapter_rows_are_zero_delta() {
    let (k, n, r) = (4, 3, 2);
    let a: Vec<f32> = (0..k * r).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..r * n).map(|i| i as f32 * 0.1).collect();
    let mut pool = LoraWeightPool::with_capacity_bytes(1 << 20);
    admit_f32(&mut pool, 0, 0, k, r, n, &a, &b, 1.0);
    let kernel = make_kernel(Arc::new(pool), k, n, 0, 8);

    let m = 2;
    let x: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
    let x_owned = Owned::f32(&[m, k], &x);
    let segments = Owned::i32(&[m], &[-1, -1]);
    let mut out = Owned::zeros_f32(&[m, n]);
    kernel
        .execute(&[x_owned.view(), segments.view()], &mut [out.view_mut()])
        .unwrap();
    assert!(out.to_f32().iter().all(|&v| v == 0.0));
}

#[test]
fn mixed_adapters_group_by_adapter_matches_per_row_reference() {
    let (k, n, r) = (4, 3, 2);
    let a0: Vec<f32> = (0..k * r).map(|i| i as f32 * 0.1 - 0.2).collect();
    let b0: Vec<f32> = (0..r * n).map(|i| i as f32 * 0.05).collect();
    let a1: Vec<f32> = (0..k * r).map(|i| i as f32 * -0.07 + 0.1).collect();
    let b1: Vec<f32> = (0..r * n).map(|i| i as f32 * -0.02 + 0.3).collect();

    let mut pool = LoraWeightPool::with_capacity_bytes(1 << 20);
    admit_f32(&mut pool, 0, 0, k, r, n, &a0, &b0, 0.5);
    admit_f32(&mut pool, 1, 0, k, r, n, &a1, &b1, 1.5);
    let kernel = make_kernel(Arc::new(pool), k, n, 0, 8);

    let m = 4;
    let x: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.3 - 1.0).collect();
    let x_owned = Owned::f32(&[m, k], &x);
    let segments = Owned::i32(&[m], &[0, 1, -1, 0]);
    let mut out = Owned::zeros_f32(&[m, n]);
    kernel
        .execute(&[x_owned.view(), segments.view()], &mut [out.view_mut()])
        .unwrap();

    let got = out.to_f32();
    for (row, seg) in [0i32, 1, -1, 0].iter().enumerate() {
        let xr = &x[row * k..(row + 1) * k];
        let expected = match seg {
            0 => reference_delta(1, k, r, n, 0.5, xr, &a0, &b0),
            1 => reference_delta(1, k, r, n, 1.5, xr, &a1, &b1),
            _ => vec![0.0; n],
        };
        for (c, e) in expected.iter().enumerate() {
            let g = got[row * n + c];
            assert!((g - e).abs() < 1e-5, "row {row} col {c}: got {g} expected {e}");
        }
    }
}

#[test]
fn missing_page_fails_loud() {
    let (k, n) = (4, 3);
    let pool = LoraWeightPool::with_capacity_bytes(1 << 20);
    let kernel = make_kernel(Arc::new(pool), k, n, 0, 8);
    let x_owned = Owned::f32(&[1, k], &[1.0, 2.0, 3.0, 4.0]);
    let segments = Owned::i32(&[1], &[7]);
    let mut out = Owned::zeros_f32(&[1, n]);
    let err = kernel
        .execute(&[x_owned.view(), segments.view()], &mut [out.view_mut()])
        .unwrap_err();
    assert!(format!("{err}").contains("no resident page"), "{err}");
}

#[test]
fn fp16_factors_use_fp32_accumulators() {
    let (k, n, r) = (256, 2, 1);
    let a: Vec<f32> = (0..k * r).map(|_| 0.05).collect();
    let b: Vec<f32> = (0..r * n).map(|_| 0.5).collect();
    let a_h: Vec<u8> = a
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let b_h: Vec<u8> = b
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let mut pool = LoraWeightPool::with_capacity_bytes(1 << 20);
    pool.admit(
        AdapterId(0),
        LoraModuleId(0),
        LoraFactorInput {
            dtype: DataType::Float16,
            rows: k,
            cols: r,
            bytes: &a_h,
        },
        LoraFactorInput {
            dtype: DataType::Float16,
            rows: r,
            cols: n,
            bytes: &b_h,
        },
        1.0,
    )
    .unwrap();
    let kernel = make_kernel(Arc::new(pool), k, n, 0, 8);

    let x: Vec<f32> = (0..k).map(|_| 1.0).collect();
    let x_owned = Owned::f32(&[1, k], &x);
    let segments = Owned::i32(&[1], &[0]);
    let mut out = Owned::zeros_f32(&[1, n]);
    kernel
        .execute(&[x_owned.view(), segments.view()], &mut [out.view_mut()])
        .unwrap();
    let a_ref: Vec<f32> = a.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
    let b_ref: Vec<f32> = b.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
    let expected = reference_delta(1, k, r, n, 1.0, &x, &a_ref, &b_ref);
    for (g, e) in out.to_f32().iter().zip(&expected) {
        assert!(
            (g - e).abs() < 1e-3,
            "got {:?} expected {expected:?}",
            out.to_f32()
        );
    }
    assert!((out.to_f32()[0] - 6.4).abs() < 0.1);
}
