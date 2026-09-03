use onnx_runtime_ir::{
    DataType, EinsumAxis, EinsumClassification, EinsumContractionTreeStep, EinsumCostBound,
    EinsumInput, EinsumPlan, EinsumShapePlan,
};

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

#[test]
fn contraction_tree_public_api_is_source_compatible_and_runtime_resolvable() {
    let vector = [2usize];
    let matrix = [2usize, 8];
    let right = [8usize];
    let inputs = [
        EinsumInput::new(DataType::Float16, &vector),
        EinsumInput::new(DataType::Float16, &matrix),
        EinsumInput::new(DataType::Float16, &right),
    ];
    let plan = EinsumPlan::build("i,ij,j->", &inputs).unwrap();
    let tree = match plan.classification() {
        EinsumClassification::ContractionTree(tree) => tree,
        _ => panic!("expected a contraction tree"),
    };
    let selected = tree.preferred_candidate().unwrap().supported().unwrap();
    assert!(
        selected
            .steps()
            .iter()
            .any(|step| matches!(step, EinsumContractionTreeStep::BinaryContraction(_)))
    );
    assert!(matches!(selected.cost().flops(), EinsumCostBound::Exact(_)));
    assert!(selected.cost().intermediate_bytes(2).is_some());

    let resolved = plan
        .resolve_concrete_contraction_tree(&[&vector, &matrix, &right])
        .unwrap()
        .unwrap();
    assert!(resolved.preferred_candidate().unwrap().cost().is_some());

    let shape_plan =
        EinsumShapePlan::build("i,ij,j->", &[&vector[..], &matrix[..], &right[..]]).unwrap();
    assert!(
        shape_plan
            .resolve_concrete_contraction_tree(&[&vector, &matrix, &right], 2)
            .unwrap()
            .unwrap()
            .preferred_candidate()
            .is_some()
    );
}
