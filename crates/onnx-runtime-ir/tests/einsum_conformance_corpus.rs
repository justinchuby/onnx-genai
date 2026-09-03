use onnx_runtime_einsum_conformance::{
    ConformanceDType, DeclaredDType, MalformedKind, PlannerQuality, default_corpus, malformed_cases,
};
use onnx_runtime_ir::{DataType, EinsumInput, EinsumPlan, EinsumPlannerQuality};

fn dtype(dtype: ConformanceDType) -> DataType {
    match dtype {
        ConformanceDType::Uint8 => DataType::Uint8,
        ConformanceDType::Uint16 => DataType::Uint16,
        ConformanceDType::Uint32 => DataType::Uint32,
        ConformanceDType::Uint64 => DataType::Uint64,
        ConformanceDType::Int8 => DataType::Int8,
        ConformanceDType::Int16 => DataType::Int16,
        ConformanceDType::Int32 => DataType::Int32,
        ConformanceDType::Int64 => DataType::Int64,
        ConformanceDType::Float16 => DataType::Float16,
        ConformanceDType::Float32 => DataType::Float32,
        ConformanceDType::Float64 => DataType::Float64,
        ConformanceDType::BFloat16 => DataType::BFloat16,
    }
}

fn declared_dtype(dtype: DeclaredDType) -> DataType {
    match dtype {
        DeclaredDType::Numeric(dtype) => self::dtype(dtype),
        DeclaredDType::Bool => DataType::Bool,
        DeclaredDType::String => DataType::String,
        DeclaredDType::Complex64 => DataType::Complex64,
        DeclaredDType::Complex128 => DataType::Complex128,
    }
}

#[test]
fn canonical_planner_accepts_every_independent_legal_record() {
    let cases = default_corpus();
    assert!(!cases.is_empty());
    let mut observed_exact_dp = false;
    let mut observed_heuristic = false;
    let mut recorded_exact_dp = false;
    let mut recorded_heuristic = false;
    for case in cases {
        let inputs = case
            .input_shapes
            .iter()
            .map(|shape| EinsumInput::new(dtype(case.dtype), shape))
            .collect::<Vec<_>>();
        let plan = EinsumPlan::build_for_opset(&case.equation, &inputs, case.opset)
            .unwrap_or_else(|error| panic!("{}: {error}", case.id));
        let output_shape = plan
            .output_shape()
            .iter()
            .map(|dimension| dimension.as_static().unwrap())
            .collect::<Vec<_>>();
        let expected =
            onnx_runtime_einsum_conformance::infer_output_shape(&case.equation, &case.input_shapes)
                .unwrap();
        assert_eq!(output_shape, expected, "{}", case.id);
        if let Some(tree) = plan.semantic_plan().contraction_tree() {
            observed_exact_dp |= tree.quality() == EinsumPlannerQuality::ExactSubsetDp;
            observed_heuristic |= tree.quality() == EinsumPlannerQuality::DeterministicGreedy;
        }
        for quality in case
            .route_probes
            .iter()
            .filter_map(|probe| probe.planner_quality)
        {
            recorded_exact_dp |= quality == PlannerQuality::ExactSubsetDp;
            recorded_heuristic |= quality == PlannerQuality::DeterministicHeuristic;
            let observed = plan
                .semantic_plan()
                .contraction_tree()
                .unwrap_or_else(|| panic!("{}: optimized route has no planner tree", case.id))
                .quality();
            let expected = match quality {
                PlannerQuality::ExactSubsetDp => EinsumPlannerQuality::ExactSubsetDp,
                PlannerQuality::DeterministicHeuristic => EinsumPlannerQuality::DeterministicGreedy,
            };
            assert_eq!(observed, expected, "{}", case.id);
        }
    }
    assert!(observed_exact_dp, "corpus did not exercise exact subset DP");
    assert!(
        observed_heuristic,
        "corpus did not exercise deterministic heuristic planning"
    );
    assert!(recorded_exact_dp && recorded_heuristic);
}

#[test]
fn canonical_planner_rejects_every_applicable_independent_malformed_record() {
    let cases = malformed_cases();
    let applicable = cases
        .into_iter()
        .filter(|case| case.kind != MalformedKind::OutputCount)
        .collect::<Vec<_>>();
    assert!(!applicable.is_empty());
    for case in applicable {
        let inputs = case
            .input_shapes
            .iter()
            .zip(&case.input_dtypes)
            .map(|(shape, &dtype)| EinsumInput::new(declared_dtype(dtype), shape))
            .collect::<Vec<_>>();
        assert!(
            EinsumPlan::build_for_opset(&case.equation, &inputs, case.opset).is_err(),
            "{} unexpectedly planned",
            case.id
        );
    }
}
