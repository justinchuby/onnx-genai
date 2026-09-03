#![deny(warnings)]

use onnx_runtime_ir::{
    DataType, EinsumClassification, EinsumInput, EinsumPlan, EinsumPlanErrorKind,
    EinsumPlanningClassification, EinsumUnsupportedReason,
};

fn exhaustive_legacy_classification(value: &EinsumClassification) -> &'static str {
    match value {
        EinsumClassification::ViewOnlyPermutation(_) => "view",
        EinsumClassification::DiagonalView(_) => "diagonal",
        EinsumClassification::ReductionOrElementwise(_) => "generic",
        EinsumClassification::Gemm(_) => "gemm",
        EinsumClassification::Unsupported(_) => "unsupported",
    }
}

fn exhaustive_legacy_error_kind(value: &EinsumPlanErrorKind) -> &'static str {
    match value {
        EinsumPlanErrorKind::NoInputs => "no inputs",
        EinsumPlanErrorKind::InputCount {
            equation_terms: _,
            inputs: _,
        } => "input count",
        EinsumPlanErrorKind::MultipleOutputArrows => "arrows",
        EinsumPlanErrorKind::InvalidCharacter {
            side: _,
            offset: _,
            found: _,
        } => "character",
        EinsumPlanErrorKind::MultipleEllipses { side: _ } => "ellipses",
        EinsumPlanErrorKind::MissingInputDtype { input: _ } => "dtype",
        EinsumPlanErrorKind::MissingInputShape { input: _ } => "shape",
        EinsumPlanErrorKind::UnsupportedInputDtype { input: _, dtype: _ } => "unsupported dtype",
        EinsumPlanErrorKind::InputDtypeMismatch {
            input: _,
            expected: _,
            actual: _,
        } => "dtype mismatch",
        EinsumPlanErrorKind::InputRankMismatch {
            input: _,
            rank: _,
            named_labels: _,
            has_ellipsis: _,
        } => "rank",
        EinsumPlanErrorKind::EllipsisRankMismatch {
            first_input: _,
            first_rank: _,
            input: _,
            rank: _,
        } => "ellipsis rank",
        EinsumPlanErrorKind::ResolvedInputCountMismatch {
            expected: _,
            found: _,
        } => "resolved count",
        EinsumPlanErrorKind::ResolvedInputRankMismatch {
            input: _,
            expected: _,
            found: _,
        } => "resolved rank",
        EinsumPlanErrorKind::ResolvedInputDimensionMismatch {
            input: _,
            axis: _,
            expected: _,
            found: _,
        } => "resolved dimension",
        EinsumPlanErrorKind::LabelMultiplicityOverflow { label: _ } => "label count",
        EinsumPlanErrorKind::DuplicateOutputLabel { label: _ } => "duplicate output",
        EinsumPlanErrorKind::OutputLabelMissingFromInputs { label: _ } => "missing output",
        EinsumPlanErrorKind::LabelDimensionMismatch {
            label: _,
            first: _,
            first_size: _,
            second: _,
            second_size: _,
        } => "label dimension",
        EinsumPlanErrorKind::EllipsisDimensionMismatch {
            axis: _,
            first: _,
            first_size: _,
            second: _,
            second_size: _,
        } => "ellipsis dimension",
        EinsumPlanErrorKind::GeometryOverflow { target: _ } => "geometry",
    }
}

#[test]
fn legacy_imports_variants_traits_and_exhaustive_matches_compile() {
    let shapes = [&[2usize, 3][..], &[3, 4][..], &[4, 5][..]];
    let inputs = shapes
        .iter()
        .map(|shape| EinsumInput::new(DataType::Float32, shape))
        .collect::<Vec<_>>();
    let plan = EinsumPlan::build("ij,jk,kl->il", &inputs).unwrap();

    assert_eq!(
        exhaustive_legacy_classification(plan.classification()),
        "generic"
    );
    assert!(matches!(
        plan.planning_classification(),
        EinsumPlanningClassification::ContractionTree(_)
    ));

    let axes = plan.reduction_axes().to_vec();
    let reasons = [
        EinsumUnsupportedReason::NaryContraction {
            input_count: 3,
            axes: axes.clone(),
        },
        EinsumUnsupportedReason::MixedContractionAndOperandReduction {
            contract_axes: axes.clone(),
            local_reduction_axes: axes.clone(),
        },
        EinsumUnsupportedReason::ReducedEllipsis { axes },
    ];
    for reason in reasons {
        assert_eq!(reason.clone(), reason);
        assert!(!format!("{reason:?}").is_empty());
        assert!(!reason.to_string().is_empty());
        let compatibility_value = EinsumClassification::Unsupported(reason);
        assert_eq!(
            exhaustive_legacy_classification(&compatibility_value),
            "unsupported"
        );
    }

    let invalid =
        EinsumPlan::build("i->j", &[EinsumInput::new(DataType::Float32, &[2])]).unwrap_err();
    assert_eq!(
        exhaustive_legacy_error_kind(invalid.kind()),
        "missing output"
    );
}
