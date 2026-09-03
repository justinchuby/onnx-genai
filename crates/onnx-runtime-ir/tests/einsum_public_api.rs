use onnx_runtime_ir::{
    DataType, EinsumAxis, EinsumContractionTreeStep, EinsumCostBound, EinsumInput, EinsumPlan,
    EinsumPlannerQuality, EinsumPlanningClassification, EinsumSchema, EinsumShapePlan,
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
fn schema_aware_build_and_generic_fallback_are_public() {
    let shape = [2usize];
    let inputs = [EinsumInput::new(DataType::BFloat16, &shape)];
    assert!(EinsumPlan::build("i->i", &inputs).is_err());
    let plan = EinsumPlan::build_for_schema("i->i", &inputs, EinsumSchema::V28).unwrap();
    assert_eq!(plan.schema(), EinsumSchema::V28);
    assert_eq!(
        plan.precision_policy().intermediate_dtype(),
        DataType::Float32
    );
    assert_eq!(
        plan.generic_native().index_program().operands()[0].physical_axis_to_iteration_axis(),
        &[0]
    );

    assert!(EinsumShapePlan::build_for_opset("i->i", &[&shape], 11).is_err());
    assert_eq!(
        EinsumShapePlan::build_for_opset("i->i", &[&shape], 28)
            .unwrap()
            .schema(),
        EinsumSchema::V28
    );
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
fn contraction_tree_planning_api_is_runtime_resolvable() {
    let vector = [2usize];
    let matrix = [2usize, 8];
    let right = [8usize];
    let inputs = [
        EinsumInput::new(DataType::Float16, &vector),
        EinsumInput::new(DataType::Float16, &matrix),
        EinsumInput::new(DataType::Float16, &right),
    ];
    let plan = EinsumPlan::build("i,ij,j->", &inputs).unwrap();
    let tree = match plan.planning_classification() {
        EinsumPlanningClassification::ContractionTree(tree) => tree,
        _ => panic!("expected a contraction tree"),
    };
    assert_eq!(tree.quality(), EinsumPlannerQuality::ExactSubsetDp);
    let selected = tree.preferred_candidate().unwrap().supported().unwrap();
    assert!(
        selected
            .steps()
            .iter()
            .any(|step| matches!(step, EinsumContractionTreeStep::BinaryContraction(_)))
    );
    assert!(matches!(selected.cost().flops(), EinsumCostBound::Exact(_)));
    assert!(selected.cost().intermediate_bytes(4).is_some());

    let resolved = plan
        .resolve_concrete_contraction_tree(&[&vector, &matrix, &right])
        .unwrap()
        .unwrap();
    assert!(resolved.preferred_candidate().unwrap().cost().is_some());

    let shape_plan =
        EinsumShapePlan::build("i,ij,j->", &[&vector[..], &matrix[..], &right[..]]).unwrap();
    assert!(
        shape_plan
            .resolve_concrete_contraction_tree(&[&vector, &matrix, &right], 4)
            .unwrap()
            .unwrap()
            .preferred_candidate()
            .is_some()
    );
}
