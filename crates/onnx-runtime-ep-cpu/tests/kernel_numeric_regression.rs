#[path = "../benches/common/mod.rs"]
mod common;

use common::{FloatDType, Tensor, assert_close, make_kernel};
use onnx_runtime_ir::Attribute;

fn add_golden(dtype: FloatDType, tolerance: f32) {
    let a = Tensor::floats(dtype, &[2, 3], &[1.0, -2.0, 3.5, 4.0, 0.5, -6.0]);
    let b = Tensor::floats(dtype, &[3], &[0.5, 2.0, -1.5]);
    let mut output = Tensor::zeros(dtype, &[2, 3]);
    make_kernel("Add", [], &[vec![2, 3], vec![3]], 13)
        .execute(&[a.view(), b.view()], &mut [output.view_mut()])
        .unwrap();
    assert_close(
        &output.to_f32(),
        &[1.5, 0.0, 2.0, 4.5, 2.5, -7.5],
        tolerance,
    );
}

fn gather_golden(dtype: FloatDType, tolerance: f32) {
    let data = Tensor::floats(
        dtype,
        &[4, 3],
        &[
            0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0,
        ],
    );
    let indices = Tensor::i64(&[3], &[2, 0, -1]);
    let mut output = Tensor::zeros(dtype, &[3, 3]);
    make_kernel(
        "Gather",
        [("axis", Attribute::Int(0))],
        &[vec![4, 3], vec![3]],
        13,
    )
    .execute(&[data.view(), indices.view()], &mut [output.view_mut()])
    .unwrap();
    assert_close(
        &output.to_f32(),
        &[20.0, 21.0, 22.0, 0.0, 1.0, 2.0, 30.0, 31.0, 32.0],
        tolerance,
    );
}

fn matmul_golden(dtype: FloatDType, tolerance: f32) {
    let a = Tensor::floats(dtype, &[2, 3], &[1.0, 2.0, 3.0, -1.0, 0.5, 2.0]);
    let b = Tensor::floats(dtype, &[3, 2], &[2.0, -1.0, 0.0, 3.0, 4.0, 0.5]);
    let mut output = Tensor::zeros(dtype, &[2, 2]);
    make_kernel("MatMul", [], &[vec![2, 3], vec![3, 2]], 13)
        .execute(&[a.view(), b.view()], &mut [output.view_mut()])
        .unwrap();
    assert_close(&output.to_f32(), &[14.0, 6.5, 6.0, 3.5], tolerance);
}

macro_rules! dtype_regression {
    ($name:ident, $runner:ident, $dtype:expr, $tolerance:expr) => {
        #[test]
        fn $name() {
            $runner($dtype, $tolerance);
        }
    };
}

dtype_regression!(add_f32_golden, add_golden, FloatDType::F32, 0.0);
dtype_regression!(add_f16_golden, add_golden, FloatDType::F16, 0.0);
dtype_regression!(add_bf16_golden, add_golden, FloatDType::Bf16, 0.0);
dtype_regression!(gather_f32_golden, gather_golden, FloatDType::F32, 0.0);
dtype_regression!(gather_f16_golden, gather_golden, FloatDType::F16, 0.0);
dtype_regression!(gather_bf16_golden, gather_golden, FloatDType::Bf16, 0.0);
dtype_regression!(matmul_f32_golden, matmul_golden, FloatDType::F32, 1e-6);
dtype_regression!(matmul_f16_golden, matmul_golden, FloatDType::F16, 1e-3);
dtype_regression!(matmul_bf16_golden, matmul_golden, FloatDType::Bf16, 2e-2);

#[test]
fn reduce_mean_f32_golden() {
    let input = Tensor::floats(
        FloatDType::F32,
        &[2, 2, 3],
        &[1.0, 2.0, 3.0, 5.0, 7.0, 9.0, -1.0, 0.0, 1.0, 4.0, 6.0, 8.0],
    );
    let mut output = Tensor::zeros(FloatDType::F32, &[2, 1, 3]);
    make_kernel(
        "ReduceMean",
        [
            ("axes", Attribute::Ints(vec![1])),
            ("keepdims", Attribute::Int(1)),
        ],
        &[vec![2, 2, 3]],
        13,
    )
    .execute(&[input.view()], &mut [output.view_mut()])
    .unwrap();
    assert_close(&output.to_f32(), &[3.0, 4.5, 6.0, 1.5, 3.0, 4.5], 0.0);
}
