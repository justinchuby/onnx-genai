use onnx_runtime_ir::{DataType, EinsumInput, EinsumPlan};

#[test]
fn typed_einsum_plan_public_api_preserves_exact_dtype() {
    let shape = [2usize, 3];

    for dtype in [DataType::Float16, DataType::Float32] {
        let inputs = [EinsumInput::new(dtype, &shape)];
        let plan = EinsumPlan::build("ij->ji", &inputs).unwrap();
        assert_eq!(plan.dtype(), dtype);
    }
}
