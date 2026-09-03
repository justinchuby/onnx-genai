use onnx_runtime_ir::{DataType, EinsumAxis, EinsumInput, EinsumPlan};

#[test]
fn typed_einsum_plan_public_api_preserves_exact_dtype() {
    let shape = [2usize, 3];

    for dtype in [DataType::Float16, DataType::Float32] {
        let inputs = [EinsumInput::new(dtype, &shape)];
        let plan = EinsumPlan::build("ij->ji", &inputs).unwrap();
        assert_eq!(plan.dtype(), dtype);
    }
}

#[test]
fn typed_einsum_plan_accepts_case_sensitive_ascii_labels() {
    let left = [2usize, 3];
    let right = [3usize, 4];
    let inputs = [
        EinsumInput::new(DataType::Float32, &left),
        EinsumInput::new(DataType::Float32, &right),
    ];
    let plan = EinsumPlan::build("Za,aB", &inputs).unwrap();
    let labels: String = plan
        .output_axes()
        .iter()
        .map(|axis| match axis {
            EinsumAxis::Label(label) => label.as_char(),
            EinsumAxis::Ellipsis(_) => panic!("equation has no ellipsis"),
        })
        .collect();

    assert_eq!(labels, "BZ");
    assert_eq!(
        plan.output_shape()
            .iter()
            .map(|dim| dim.as_static())
            .collect::<Vec<_>>(),
        [Some(4), Some(2)]
    );
}
