use std::collections::{BTreeMap, BTreeSet};

use onnx_runtime_ir::{
    DataType, EinsumAxis, EinsumBinaryLowering, EinsumClassification, EinsumConcretePlanError,
    EinsumContractionCost, EinsumContractionTreeStep, EinsumCostBound, EinsumExecutionSelection,
    EinsumInput, EinsumIntegerOverflowSemantics, EinsumOpsetPlanError, EinsumPlan,
    EinsumPlanErrorKind, EinsumPlannerBudget, EinsumPlannerFallbackReason, EinsumPlannerQuality,
    EinsumPrecisionPolicy, EinsumSchema, EinsumShapePlan, EinsumTemporaryStoragePolicy,
};

type LegalCase<'a> = (&'a str, Vec<&'a [usize]>, Vec<usize>);

fn plan(schema: EinsumSchema, equation: &str, dtype: DataType, shapes: &[&[usize]]) -> EinsumPlan {
    let inputs = shapes
        .iter()
        .map(|shape| EinsumInput::new(dtype, shape))
        .collect::<Vec<_>>();
    EinsumPlan::build_for_schema(equation, &inputs, schema).unwrap()
}

fn tree(plan: &EinsumPlan) -> &onnx_runtime_ir::EinsumContractionTreePlan {
    plan.semantic_plan()
        .contraction_tree()
        .expect("multi-operand equation has a semantic tree")
}

fn label(axis: EinsumAxis) -> char {
    match axis {
        EinsumAxis::Label(label) => label.as_char(),
        EinsumAxis::Ellipsis(_) => '.',
    }
}

#[test]
fn schema_resolution_and_authoritative_numeric_type_matrix() {
    assert!(EinsumSchema::resolve(11).is_err());
    assert_eq!(EinsumSchema::resolve(12).unwrap(), EinsumSchema::V12);
    assert_eq!(EinsumSchema::resolve(27).unwrap(), EinsumSchema::V12);
    assert_eq!(EinsumSchema::resolve(28).unwrap(), EinsumSchema::V28);
    assert_eq!(EinsumSchema::resolve(10_000).unwrap(), EinsumSchema::V28);

    let numeric_v12 = [
        DataType::Uint8,
        DataType::Uint16,
        DataType::Uint32,
        DataType::Uint64,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::Float16,
        DataType::Float32,
        DataType::Float64,
    ];
    let scalar: [usize; 0] = [];
    for dtype in numeric_v12 {
        let input = [EinsumInput::new(dtype, &scalar)];
        assert!(EinsumPlan::build_for_opset("->", &input, 12).is_ok());
        assert!(EinsumPlan::build_for_opset("->", &input, 27).is_ok());
        assert!(EinsumPlan::build_for_opset("->", &input, 28).is_ok());
    }

    let bf16 = [EinsumInput::new(DataType::BFloat16, &scalar)];
    for opset in [12, 27] {
        let error = EinsumPlan::build_for_opset("->", &bf16, opset).unwrap_err();
        let plan_error = error.plan_error().unwrap();
        assert!(matches!(
            plan_error.kind(),
            EinsumPlanErrorKind::UnsupportedInputDtype {
                dtype: DataType::BFloat16,
                ..
            }
        ));
        assert_eq!(plan_error.schema(), Some(EinsumSchema::V12));
    }
    assert!(EinsumPlan::build_for_opset("->", &bf16, 28).is_ok());
    assert!(matches!(
        EinsumPlan::build_for_opset("->", &bf16, 11).unwrap_err(),
        EinsumOpsetPlanError::UnsupportedOpset {
            imported_opset: 11,
            ..
        }
    ));

    for rejected in [
        DataType::Undefined,
        DataType::String,
        DataType::Bool,
        DataType::Complex64,
        DataType::Complex128,
        DataType::Float8E4M3FN,
        DataType::Int4,
    ] {
        let input = [EinsumInput::new(rejected, &scalar)];
        assert!(EinsumPlan::build_for_opset("->", &input, 28).is_err());
    }
}

#[test]
fn precision_policy_is_explicit_and_backend_neutral() {
    let scalar: [usize; 0] = [];
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let schema = if dtype == DataType::BFloat16 {
            EinsumSchema::V28
        } else {
            EinsumSchema::V12
        };
        let plan = plan(schema, "->", dtype, &[&scalar]);
        let policy = plan.precision_policy();
        assert_eq!(
            EinsumPrecisionPolicy::for_schema(schema, dtype),
            Some(policy)
        );
        assert_eq!(policy.input_output_dtype(), dtype);
        assert_eq!(policy.accumulator_dtype(), DataType::Float32);
        assert_eq!(policy.intermediate_dtype(), DataType::Float32);
        assert!(policy.narrow_once_at_output());
        assert_eq!(policy.integer_overflow(), None);
    }
    assert_eq!(
        EinsumPrecisionPolicy::for_schema(EinsumSchema::V12, DataType::BFloat16),
        None
    );
    for dtype in [DataType::Float32, DataType::Float64] {
        let policy = plan(EinsumSchema::V12, "->", dtype, &[&scalar]).precision_policy();
        assert_eq!(policy.accumulator_dtype(), dtype);
        assert_eq!(policy.intermediate_dtype(), dtype);
        assert!(!policy.narrow_once_at_output());
    }
    for dtype in [
        DataType::Uint8,
        DataType::Uint16,
        DataType::Uint32,
        DataType::Uint64,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
    ] {
        let policy = plan(EinsumSchema::V12, "->", dtype, &[&scalar]).precision_policy();
        assert_eq!(policy.accumulator_dtype(), dtype);
        assert_eq!(policy.intermediate_dtype(), dtype);
        assert_eq!(
            policy.integer_overflow(),
            Some(EinsumIntegerOverflowSemantics::WrappingModuloPowerOfTwo)
        );
    }
}

#[test]
fn every_required_legal_equation_has_generic_native_semantics() {
    let cases: Vec<LegalCase<'_>> = vec![
        ("i,i,i->", vec![&[2], &[2], &[2]], vec![]),
        ("i,j,k->ijk", vec![&[2], &[3], &[4]], vec![2, 3, 4]),
        ("...i,...i->", vec![&[1, 3], &[5, 3]], vec![]),
        (",->", vec![&[], &[]], vec![]),
        ("Za,aB->BZ", vec![&[2, 3], &[3, 4]], vec![4, 2]),
        ("ii->i", vec![&[3, 3]], vec![3]),
        ("...i,...i->...i", vec![&[1, 3], &[5, 3]], vec![5, 3]),
        ("i,i->", vec![&[0], &[0]], vec![]),
    ];
    for (equation, shapes, expected) in cases {
        let plan = plan(EinsumSchema::V12, equation, DataType::Float32, &shapes);
        assert_eq!(
            plan.output_shape()
                .iter()
                .map(|dimension| dimension.as_static().unwrap())
                .collect::<Vec<_>>(),
            expected,
            "{equation}"
        );
        assert_eq!(
            plan.generic_native().index_program().operands().len(),
            shapes.len(),
            "{equation}"
        );
        assert!(
            !matches!(plan.classification(), EinsumClassification::Unsupported(_)),
            "{equation}"
        );
    }
}

#[test]
fn arbitrary_four_eight_and_sixteen_operand_equations_are_bounded() {
    for arity in [4usize, 5, 8, 16] {
        let equation = format!(
            "{}->",
            std::iter::repeat_n("i", arity)
                .collect::<Vec<_>>()
                .join(",")
        );
        let shapes = vec![vec![1usize]; arity];
        let refs = shapes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let plan = plan(EinsumSchema::V12, &equation, DataType::Float32, &refs);
        let tree = tree(&plan);
        assert_eq!(tree.arity(), arity);
        assert!(tree.preferred_candidate().is_some());
        assert!(
            tree.usage().candidates()
                <= tree
                    .usage()
                    .budget()
                    .max_candidates
                    .max(tree.usage().budget().max_heuristic_candidates)
        );
        assert_eq!(
            tree.quality(),
            if arity <= 5 {
                EinsumPlannerQuality::ExactSubsetDp
            } else {
                EinsumPlannerQuality::DeterministicGreedy
            }
        );
    }
}

#[test]
fn invalid_syntax_output_ellipsis_dtype_and_shapes_are_rejected() {
    let vector = [2usize];
    let matrix = [2usize, 3];
    let square = [2usize, 2];
    let one = [EinsumInput::new(DataType::Float32, &vector)];
    for equation in ["i\t->i", "i\u{00a0}->i", "λ->λ", "i$->i", "i->i->i"] {
        assert!(EinsumPlan::build(equation, &one).is_err(), "{equation:?}");
    }
    for equation in ["i->ii", "i->j", "......i->i", "i->......i"] {
        assert!(EinsumPlan::build(equation, &one).is_err(), "{equation}");
    }

    let mixed = [
        EinsumInput::new(DataType::Float32, &vector),
        EinsumInput::new(DataType::Float16, &vector),
    ];
    assert!(matches!(
        EinsumPlan::build("i,i->i", &mixed).unwrap_err().kind(),
        EinsumPlanErrorKind::InputDtypeMismatch { .. }
    ));

    let diagonal = [EinsumInput::new(DataType::Float32, &matrix)];
    assert!(matches!(
        EinsumPlan::build("ii->i", &diagonal).unwrap_err().kind(),
        EinsumPlanErrorKind::LabelDimensionMismatch { .. }
    ));
    let nonbroadcast = [
        EinsumInput::new(DataType::Float32, &vector),
        EinsumInput::new(DataType::Float32, &[1usize]),
    ];
    assert!(matches!(
        EinsumPlan::build("i,i->i", &nonbroadcast)
            .unwrap_err()
            .kind(),
        EinsumPlanErrorKind::LabelDimensionMismatch { .. }
    ));
    let rank = [EinsumInput::new(DataType::Float32, &square)];
    assert!(matches!(
        EinsumPlan::build("i->i", &rank).unwrap_err().kind(),
        EinsumPlanErrorKind::InputRankMismatch { .. }
    ));
}

#[test]
fn reductions_wait_until_every_axis_occurrence_is_inside_the_subtree() {
    let shapes = [&[2usize][..], &[2usize][..], &[2usize][..]];
    let plan = plan(EinsumSchema::V12, "i,i,i->", DataType::Float32, &shapes);
    let occurrences = plan
        .logical_axes()
        .iter()
        .map(|axis| {
            (
                axis.axis(),
                axis.occurrences()
                    .iter()
                    .map(|occurrence| occurrence.input())
                    .collect::<BTreeSet<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let output = plan.output_axes().iter().copied().collect::<BTreeSet<_>>();
    for candidate in tree(&plan).candidates() {
        let Some(candidate) = candidate.supported() else {
            continue;
        };
        for step in candidate.steps() {
            let EinsumContractionTreeStep::BinaryContraction(binary) = step else {
                continue;
            };
            let left = binary
                .left_leaf_inputs()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            let right = binary
                .right_leaf_inputs()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            let subtree = left.union(&right).copied().collect::<BTreeSet<_>>();
            for axis in binary.contract_axes() {
                assert!(!output.contains(axis));
                assert!(occurrences[axis].is_subset(&subtree));
                assert!(
                    !occurrences[axis].is_subset(&left) && !occurrences[axis].is_subset(&right),
                    "{axis} was reducible below this merge"
                );
            }
        }
    }
    let selected = tree(&plan)
        .preferred_candidate()
        .unwrap()
        .supported()
        .unwrap();
    let binary_steps = selected
        .steps()
        .iter()
        .filter_map(|step| match step {
            EinsumContractionTreeStep::BinaryContraction(binary) => Some(binary.as_ref()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        binary_steps[0].lowering(),
        EinsumBinaryLowering::GenericNative
    );
    assert_eq!(
        binary_steps.last().unwrap().lowering(),
        EinsumBinaryLowering::GemmCompatible
    );
    let root = selected
        .steps()
        .iter()
        .rev()
        .find_map(|step| match step {
            EinsumContractionTreeStep::BinaryContraction(binary) => Some(binary),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        root.contract_axes()
            .iter()
            .copied()
            .map(label)
            .collect::<String>(),
        "i"
    );
}

fn brute_tree_ids(leaves: &[usize]) -> BTreeSet<String> {
    if let [leaf] = leaves {
        return BTreeSet::from([leaf.to_string()]);
    }
    let mut result = BTreeSet::new();
    let full = (1usize << leaves.len()) - 1;
    for left_mask in 1..full {
        let left = leaves
            .iter()
            .enumerate()
            .filter_map(|(index, leaf)| ((left_mask >> index) & 1 == 1).then_some(*leaf))
            .collect::<Vec<_>>();
        let right = leaves
            .iter()
            .enumerate()
            .filter_map(|(index, leaf)| ((left_mask >> index) & 1 == 0).then_some(*leaf))
            .collect::<Vec<_>>();
        for left_id in brute_tree_ids(&left) {
            for right_id in brute_tree_ids(&right) {
                result.insert(format!("({left_id},{right_id})"));
            }
        }
    }
    result
}

fn compare_cost_bound(left: EinsumCostBound, right: EinsumCostBound) -> std::cmp::Ordering {
    match (left, right) {
        (EinsumCostBound::Exact(left), EinsumCostBound::Exact(right)) => left.cmp(&right),
        (EinsumCostBound::Exact(_), EinsumCostBound::UnknownUpperBound) => std::cmp::Ordering::Less,
        (EinsumCostBound::UnknownUpperBound, EinsumCostBound::Exact(_)) => {
            std::cmp::Ordering::Greater
        }
        _ => std::cmp::Ordering::Equal,
    }
}

fn compare_cost(left: &EinsumContractionCost, right: &EinsumContractionCost) -> std::cmp::Ordering {
    [
        (left.flops(), right.flops()),
        (left.unary_or_product_work(), right.unary_or_product_work()),
        (left.intermediate_elements(), right.intermediate_elements()),
        (
            left.peak_live_temporary_elements(),
            right.peak_live_temporary_elements(),
        ),
        (
            left.total_intermediate_traffic_elements(),
            right.total_intermediate_traffic_elements(),
        ),
        (
            left.layout_or_packing_traffic_elements(),
            right.layout_or_packing_traffic_elements(),
        ),
        (
            left.broadcast_amplification_elements(),
            right.broadcast_amplification_elements(),
        ),
    ]
    .into_iter()
    .map(|(left, right)| compare_cost_bound(left, right))
    .find(|ordering| !ordering.is_eq())
    .unwrap_or_else(|| left.slot_count().cmp(&right.slot_count()))
}

#[test]
fn exact_subset_dp_matches_independent_brute_force_tree_enumerator() {
    for arity in 2..=4 {
        let equation = format!(
            "{}->",
            std::iter::repeat_n("i", arity)
                .collect::<Vec<_>>()
                .join(",")
        );
        let owned = vec![vec![2usize]; arity];
        let shapes = owned.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let plan = plan(EinsumSchema::V12, &equation, DataType::Float32, &shapes);
        let actual = tree(&plan)
            .candidates()
            .iter()
            .map(|candidate| candidate.id().as_str().to_string())
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, brute_tree_ids(&(0..arity).collect::<Vec<_>>()));
        let tree = tree(&plan);
        assert_eq!(tree.quality(), EinsumPlannerQuality::ExactSubsetDp);
        assert!(tree.usage().max_depth() < arity);
        assert_eq!(
            tree.usage().candidate_id_bytes(),
            tree.candidates()
                .iter()
                .map(|candidate| candidate.id().as_str().len())
                .sum::<usize>()
        );
        assert!(tree.usage().work() >= tree.usage().metadata_units());
        assert!(tree.usage().candidates() <= tree.usage().budget().max_candidates);
        assert!(
            tree.usage().metadata_units()
                <= tree.usage().budget().exact_metadata_units_limit().unwrap()
        );
        if arity == 4 {
            assert_eq!(
                tree.preferred_candidate().unwrap().id().as_str(),
                actual.first().unwrap(),
                "symmetric equal-cost trees use the stable lexicographic tie-break"
            );
        }
    }
}

#[test]
fn exact_dp_preference_matches_independent_generated_brute_force_comparator() {
    let labels = ['a', 'b', 'c', 'd'];
    let extents = BTreeMap::from([('a', 2usize), ('b', 3), ('c', 2), ('d', 3)]);
    let mut seed = 0x2349_u64;
    for _ in 0..24 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let arity = 2 + seed as usize % 3;
        let mut terms = Vec::new();
        let mut shapes = Vec::new();
        for input in 0..arity {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let first = labels[(seed as usize + input) % labels.len()];
            let second = labels[((seed >> 11) as usize + input + 1) % labels.len()];
            let term = if first == second {
                vec![first]
            } else {
                vec![first, second]
            };
            shapes.push(term.iter().map(|label| extents[label]).collect::<Vec<_>>());
            terms.push(term);
        }
        let equation = format!(
            "{}->",
            terms
                .iter()
                .map(|term| term.iter().collect::<String>())
                .collect::<Vec<_>>()
                .join(",")
        );
        let refs = shapes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let plan = plan(EinsumSchema::V12, &equation, DataType::Float32, &refs);
        let tree = tree(&plan);
        let expected = tree
            .candidates()
            .iter()
            .filter_map(|candidate| candidate.supported().map(|plan| (candidate, plan.cost())))
            .min_by(|(left_candidate, left), (right_candidate, right)| {
                compare_cost(left, right)
                    .then_with(|| left_candidate.id().cmp(right_candidate.id()))
            })
            .unwrap()
            .0
            .id();
        assert_eq!(
            tree.preferred_candidate().unwrap().id(),
            expected,
            "{equation}"
        );
    }
}

#[test]
fn large_planning_is_deterministic_budgeted_and_stably_tied() {
    let arity = 16;
    let equation = format!(
        "{}->",
        std::iter::repeat_n("i", arity)
            .collect::<Vec<_>>()
            .join(",")
    );
    let owned = vec![vec![1usize]; arity];
    let shapes = owned.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let first = plan(EinsumSchema::V12, &equation, DataType::Float32, &shapes);
    let second = plan(EinsumSchema::V12, &equation, DataType::Float32, &shapes);
    let first_tree = tree(&first);
    let second_tree = tree(&second);
    assert_eq!(
        first_tree.quality(),
        EinsumPlannerQuality::DeterministicGreedy
    );
    assert_eq!(first_tree.usage(), second_tree.usage());
    assert_eq!(
        first_tree.preferred_candidate().unwrap().id(),
        second_tree.preferred_candidate().unwrap().id()
    );
    assert!(
        first_tree.usage().candidates() <= first_tree.usage().budget().max_heuristic_candidates
    );

    let forced = EinsumPlannerBudget {
        exact_operand_limit: 16,
        max_states: 2,
        max_candidates: 2,
        max_exact_axes: 64,
        max_heuristic_candidates: 8,
    };
    let inputs = shapes
        .iter()
        .map(|shape| EinsumInput::new(DataType::Float32, shape))
        .collect::<Vec<_>>();
    let forced =
        EinsumPlan::build_for_schema_with_budget(&equation, &inputs, EinsumSchema::V12, forced)
            .unwrap();
    assert_eq!(
        tree(&forced).quality(),
        EinsumPlannerQuality::GenericNativeFallback
    );
    assert_eq!(
        tree(&forced).fallback_reason(),
        Some(EinsumPlannerFallbackReason::WorkOrMetadataBudgetExceeded)
    );
    assert!(tree(&forced).candidates().is_empty());
    assert_eq!(tree(&forced).usage().work(), 0);
    assert_eq!(tree(&forced).usage().metadata_units(), 0);
}

#[test]
fn tied_unit_extents_build_a_balanced_bounded_greedy_candidate() {
    let arity = 20;
    let equation = format!(
        "{}->",
        std::iter::repeat_n("i", arity)
            .collect::<Vec<_>>()
            .join(",")
    );
    let owned = vec![vec![1usize]; arity];
    let shapes = owned.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let first = plan(EinsumSchema::V12, &equation, DataType::Float32, &shapes);
    let second = plan(EinsumSchema::V12, &equation, DataType::Float32, &shapes);
    let first_tree = tree(&first);
    let second_tree = tree(&second);

    assert_eq!(
        first_tree.quality(),
        EinsumPlannerQuality::DeterministicGreedy
    );
    assert_eq!(first_tree.usage(), second_tree.usage());
    assert!(first_tree.usage().max_depth() <= 5);
    assert!(first_tree.usage().max_depth() <= EinsumPlannerBudget::MAX_CONTRACTION_TREE_DEPTH);
    assert!(first_tree.usage().work() <= first_tree.usage().budget().max_heuristic_candidates);
    assert!(first_tree.usage().metadata_units() <= first_tree.usage().work());
    let selected = first_tree.preferred_candidate().unwrap();
    assert_eq!(
        selected.id(),
        second_tree.preferred_candidate().unwrap().id()
    );
    assert_eq!(
        first_tree.usage().candidate_id_bytes(),
        selected.id().as_str().len()
    );
    let selected = selected.supported().unwrap();
    assert_eq!(
        selected
            .steps()
            .iter()
            .filter(|step| matches!(step, EinsumContractionTreeStep::BinaryContraction(_)))
            .count(),
        arity - 1
    );
    assert!(
        selected
            .temporaries()
            .iter()
            .map(|temporary| temporary.leaf_inputs().len())
            .sum::<usize>()
            <= arity * 6
    );
}

#[test]
fn hundreds_and_thousands_of_operands_use_zero_tree_metadata_fallback() {
    let one = [1usize];
    for arity in [256usize, 1024, 2048] {
        let equation = format!(
            "{}->",
            std::iter::repeat_n("i", arity)
                .collect::<Vec<_>>()
                .join(",")
        );
        let inputs = vec![EinsumInput::new(DataType::Float32, &one); arity];
        let plan = EinsumPlan::build_for_schema(&equation, &inputs, EinsumSchema::V12).unwrap();
        let tree = tree(&plan);

        assert_eq!(tree.quality(), EinsumPlannerQuality::GenericNativeFallback);
        assert_eq!(
            tree.fallback_reason(),
            Some(EinsumPlannerFallbackReason::WorkOrMetadataBudgetExceeded)
        );
        assert!(tree.candidates().is_empty());
        assert!(tree.preferred_candidate().is_none());
        assert_eq!(tree.usage().states(), 0);
        assert_eq!(tree.usage().candidates(), 0);
        assert_eq!(tree.usage().work(), 0);
        assert_eq!(tree.usage().metadata_units(), 0);
        assert_eq!(tree.usage().max_depth(), 0);
        assert_eq!(tree.usage().candidate_id_bytes(), 0);
        assert_eq!(
            plan.generic_native().index_program().operands().len(),
            arity
        );

        let shapes = vec![&one[..]; arity];
        assert_eq!(
            plan.select_concrete_execution(&shapes, u128::MAX).unwrap(),
            EinsumExecutionSelection::GenericNative
        );
    }
}

#[test]
fn costs_are_checked_u128_zero_annihilates_and_memory_can_fall_back() {
    let maximum = [usize::MAX];
    let wide = plan(
        EinsumSchema::V12,
        "i,i->",
        DataType::Float32,
        &[&maximum, &maximum],
    );
    let flops = tree(&wide)
        .preferred_candidate()
        .unwrap()
        .supported()
        .unwrap()
        .cost()
        .flops()
        .exact()
        .unwrap();
    assert!(flops > u64::MAX as u128);

    let zero_max = [0usize, usize::MAX];
    let zero = plan(
        EinsumSchema::V12,
        "ij,ij->",
        DataType::Float32,
        &[&zero_max, &zero_max],
    );
    assert_eq!(
        tree(&zero)
            .preferred_candidate()
            .unwrap()
            .supported()
            .unwrap()
            .cost()
            .flops(),
        EinsumCostBound::Exact(0)
    );

    let chain = plan(
        EinsumSchema::V12,
        "ij,jk,kl->il",
        DataType::Float16,
        &[&[2, 3], &[3, 4], &[4, 5]],
    );
    let concrete = chain
        .resolve_concrete_contraction_tree(&[&[2, 3], &[3, 4], &[4, 5]])
        .unwrap()
        .unwrap();
    assert!(concrete.preferred_candidate().is_some());
    assert!(
        concrete
            .preferred_candidate_with_memory_ceiling(0)
            .is_none()
    );
    assert_eq!(
        chain
            .select_concrete_execution(&[&[2, 3], &[3, 4], &[4, 5]], 0)
            .unwrap(),
        EinsumExecutionSelection::GenericNative
    );
    assert!(!chain.generic_native().index_program().operands().is_empty());

    let shape_plan =
        EinsumShapePlan::build("ij,jk,kl->il", &[&[2, 3][..], &[3, 4][..], &[4, 5][..]]).unwrap();
    assert!(matches!(
        shape_plan.resolve_concrete_contraction_tree(&[&[2, 3], &[3, 4], &[4, 5]], 0),
        Err(EinsumConcretePlanError::InvalidElementSize {
            element_size: 0,
            ..
        })
    ));
}

#[test]
fn liveness_layout_and_accumulator_storage_are_published() {
    let plan = plan(
        EinsumSchema::V12,
        "ij,jk,kl,lm->im",
        DataType::Float16,
        &[&[2, 3], &[3, 4], &[4, 5], &[5, 6]],
    );
    let candidate = tree(&plan)
        .preferred_candidate()
        .unwrap()
        .supported()
        .unwrap();
    for temporary in candidate.temporaries() {
        assert!(temporary.birth_step() <= temporary.last_use_step());
        assert_eq!(
            temporary.axes().len(),
            temporary.global_iteration_axis_indices().len()
        );
        assert_eq!(
            temporary.storage_policy(),
            EinsumTemporaryStoragePolicy::Accumulator
        );
        assert_eq!(
            plan.precision_policy().intermediate_dtype(),
            DataType::Float32
        );
    }
    assert!(matches!(
        candidate.cost().peak_live_temporary_elements(),
        EinsumCostBound::Exact(_)
    ));
}

fn row_major_offset(indices: &[usize], shape: &[usize]) -> usize {
    indices
        .iter()
        .zip(shape)
        .fold(0usize, |offset, (&index, &extent)| offset * extent + index)
}

fn for_each_index(shape: &[usize], mut visit: impl FnMut(&[usize])) {
    if shape.contains(&0) {
        return;
    }
    let mut index = vec![0usize; shape.len()];
    loop {
        visit(&index);
        let Some(axis) = (0..shape.len())
            .rev()
            .find(|&axis| index[axis] + 1 < shape[axis])
        else {
            break;
        };
        index[axis] += 1;
        for later in index.iter_mut().take(shape.len()).skip(axis + 1) {
            *later = 0;
        }
    }
}

fn independent_reference(
    operand_labels: &[Vec<char>],
    output_labels: &[char],
    extents: &BTreeMap<char, usize>,
    operands: &[Vec<f64>],
) -> Vec<f64> {
    let all_labels = operand_labels
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    let reductions = all_labels
        .iter()
        .copied()
        .filter(|label| !output_labels.contains(label))
        .collect::<Vec<_>>();
    let mut iteration_labels = output_labels.to_vec();
    iteration_labels.extend(&reductions);
    let iteration_shape = iteration_labels
        .iter()
        .map(|label| extents[label])
        .collect::<Vec<_>>();
    let output_shape = output_labels
        .iter()
        .map(|label| extents[label])
        .collect::<Vec<_>>();
    let mut output = vec![0.0; output_shape.iter().product()];
    for_each_index(&iteration_shape, |index| {
        let by_label = iteration_labels
            .iter()
            .copied()
            .zip(index.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let product = operand_labels
            .iter()
            .zip(operands)
            .map(|(labels, data)| {
                let operand_index = labels
                    .iter()
                    .map(|label| by_label[label])
                    .collect::<Vec<_>>();
                let shape = labels
                    .iter()
                    .map(|label| extents[label])
                    .collect::<Vec<_>>();
                data[row_major_offset(&operand_index, &shape)]
            })
            .product::<f64>();
        let output_offset = row_major_offset(&index[..output_labels.len()], &output_shape);
        output[output_offset] += product;
    });
    output
}

fn evaluate_index_program(plan: &EinsumPlan, operands: &[Vec<f64>]) -> Vec<f64> {
    let program = plan.generic_native().index_program();
    let dimensions = plan
        .logical_axes()
        .iter()
        .map(|axis| (axis.axis(), axis.dimension().as_static().unwrap()))
        .collect::<BTreeMap<_, _>>();
    let iteration_shape = program
        .iteration_axes()
        .iter()
        .map(|axis| dimensions[axis])
        .collect::<Vec<_>>();
    let output_shape = &iteration_shape[..program.output_rank()];
    let mut output = vec![0.0; output_shape.iter().product()];
    for_each_index(&iteration_shape, |index| {
        let product = program
            .operands()
            .iter()
            .zip(operands)
            .map(|(operand, data)| {
                let shape = plan.operands()[operand.input()]
                    .shape()
                    .iter()
                    .map(|dimension| dimension.as_static().unwrap())
                    .collect::<Vec<_>>();
                let operand_index = operand
                    .physical_axis_to_iteration_axis()
                    .iter()
                    .zip(operand.physical_axis_broadcasts_when_one())
                    .zip(&shape)
                    .map(|((&axis, &broadcasts), &extent)| {
                        if broadcasts && extent == 1 {
                            0
                        } else {
                            index[axis]
                        }
                    })
                    .collect::<Vec<_>>();
                data[row_major_offset(&operand_index, &shape)]
            })
            .product::<f64>();
        let output_offset = row_major_offset(&index[..program.output_rank()], output_shape);
        output[output_offset] += product;
    });
    output
}

#[test]
fn generated_generic_index_program_matches_independent_reference_model() {
    let mut seed = 0x5eed_u64;
    for case in 0..32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let arity = 1 + (seed as usize % 4);
        let labels = ['a', 'b', 'c', 'd'];
        let extents = BTreeMap::from([('a', 2), ('b', 2), ('c', 3), ('d', 2)]);
        let mut operand_labels = Vec::new();
        for input in 0..arity {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let first = labels[(seed as usize + input) % labels.len()];
            let second = labels[((seed >> 8) as usize + input + 1) % labels.len()];
            operand_labels.push(if case % 7 == 0 && input == 0 {
                vec![first, first]
            } else if first == second {
                vec![first]
            } else {
                vec![first, second]
            });
        }
        let all = operand_labels
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let output_labels = all
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, label)| (index % 2 == case % 2).then_some(label))
            .collect::<Vec<_>>();
        let equation = format!(
            "{}->{}",
            operand_labels
                .iter()
                .map(|labels| labels.iter().collect::<String>())
                .collect::<Vec<_>>()
                .join(","),
            output_labels.iter().collect::<String>()
        );
        let shapes = operand_labels
            .iter()
            .map(|labels| {
                labels
                    .iter()
                    .map(|label| extents[label])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let shape_refs = shapes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let plan = plan(EinsumSchema::V12, &equation, DataType::Float64, &shape_refs);
        let operands = shapes
            .iter()
            .enumerate()
            .map(|(input, shape)| {
                (0..shape.iter().product())
                    .map(|index| 1.0 + ((input + index) % 5) as f64)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            evaluate_index_program(&plan, &operands),
            independent_reference(&operand_labels, &output_labels, &extents, &operands),
            "{equation}"
        );
    }
}

#[test]
fn generic_index_program_encodes_ellipsis_broadcast_indexing() {
    let plan = plan(
        EinsumSchema::V12,
        "...i,...i->...",
        DataType::Float64,
        &[&[1, 2], &[3, 2]],
    );
    assert_eq!(
        plan.generic_native().index_program().operands()[0].physical_axis_broadcasts_when_one(),
        &[true, false]
    );
    assert_eq!(
        evaluate_index_program(&plan, &[vec![2.0, 3.0], vec![1.0, 4.0, 5.0, 2.0, 3.0, 6.0]]),
        vec![14.0, 16.0, 24.0]
    );
}
