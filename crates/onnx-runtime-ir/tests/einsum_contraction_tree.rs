use onnx_runtime_ir::{
    DataType, Dim, EinsumAxis, EinsumClassification, EinsumContractionTreeCandidate,
    EinsumContractionTreeStep, EinsumCostBound, EinsumInput, EinsumPlan, EinsumUnsupportedReason,
    SymbolId,
};

fn plan(equation: &str, shapes: &[&[usize]]) -> EinsumPlan {
    let inputs = shapes
        .iter()
        .map(|shape| EinsumInput::new(DataType::Float32, shape))
        .collect::<Vec<_>>();
    EinsumPlan::build(equation, &inputs).unwrap()
}

fn tree(plan: &EinsumPlan) -> &onnx_runtime_ir::EinsumContractionTreePlan {
    match plan.classification() {
        EinsumClassification::ContractionTree(tree) => tree,
        other => panic!("expected contraction tree, found {other:?}"),
    }
}

fn candidate<'a>(
    tree: &'a onnx_runtime_ir::EinsumContractionTreePlan,
    id: &str,
) -> &'a EinsumContractionTreeCandidate {
    tree.candidates()
        .iter()
        .find(|candidate| candidate.id().as_str() == id)
        .unwrap_or_else(|| panic!("missing candidate {id}"))
}

fn unordered_pair(mut pair: [usize; 2]) -> [usize; 2] {
    pair.sort_unstable();
    pair
}

fn first_binary(
    candidate: &EinsumContractionTreeCandidate,
) -> &onnx_runtime_ir::EinsumBinaryContractionPlan {
    candidate
        .supported()
        .unwrap()
        .steps()
        .iter()
        .find_map(|step| match step {
            EinsumContractionTreeStep::BinaryContraction(binary) => Some(binary),
            EinsumContractionTreeStep::UnaryReduction(_) => None,
            _ => None,
        })
        .unwrap()
}

#[test]
fn scalar_vector_matrix_vector_enumerates_and_eliminates_at_lowest_nodes() {
    let plan = plan("i,ij,j->", &[&[3], &[3, 5], &[5]]);
    let tree = tree(&plan);
    assert_eq!(tree.arity(), 3);
    assert_eq!(tree.leaf_values().len(), 3);
    assert_eq!(tree.candidates().len(), 12);

    let candidate = candidate(tree, "((0,1),2)").supported().unwrap();
    let binaries = candidate
        .steps()
        .iter()
        .filter_map(|step| match step {
            EinsumContractionTreeStep::BinaryContraction(binary) => Some(binary),
            EinsumContractionTreeStep::UnaryReduction(_) => None,
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(binaries.len(), 2);
    assert_eq!(
        binaries[0]
            .contract_axes()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["label `i`"]
    );
    assert_eq!(
        binaries[1]
            .contract_axes()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["label `j`"]
    );
    assert!(plan.output_axes().is_empty());
}

#[test]
fn x_transpose_a_y_order_flips_from_independent_operation_counts() {
    for (i, j) in (1usize..=8).flat_map(|i| (1usize..=8).map(move |j| (i, j))) {
        let plan = plan("i,ij,j->", &[&[i], &[i, j], &[j]]);
        let selected = tree(&plan).preferred_candidate().unwrap();

        // Independent scalar-operation model:
        // (x^T A)y = J(2I-1) + (2J-1)
        // x^T(Ay) = I(2J-1) + (2I-1)
        let left_first = j * (2 * i - 1) + (2 * j - 1);
        let right_first = i * (2 * j - 1) + (2 * i - 1);
        if left_first != right_first {
            let expected_pair = if left_first < right_first {
                [0, 1]
            } else {
                [1, 2]
            };
            assert_eq!(unordered_pair(selected.first_pair()), expected_pair);
        }
    }
}

#[test]
fn batched_and_ellipsis_trilinear_plans_record_batch_and_virtual_axes() {
    let batched = plan("bi,bij,bj->b", &[&[7, 3], &[7, 3, 5], &[7, 5]]);
    let first = first_binary(candidate(tree(&batched), "((0,1),2)"));
    assert_eq!(first.batch_axes().len(), 1);
    assert!(first.left_virtual_singleton_axes().is_empty());
    assert!(first.right_virtual_singleton_axes().is_empty());

    let ellipsis = plan("i,...ij,...j->...", &[&[3], &[2, 7, 3, 5], &[2, 7, 5]]);
    let first = first_binary(candidate(tree(&ellipsis), "((0,1),2)"));
    assert_eq!(first.batch_axes().len(), 2);
    assert_eq!(first.left_virtual_singleton_axes().len(), 2);
    assert!(first.right_virtual_singleton_axes().is_empty());
    assert_eq!(
        first.geometry().batch_shape(),
        &[
            onnx_runtime_ir::EinsumDimension::Static(2),
            onnx_runtime_ir::EinsumDimension::Static(7),
        ]
    );
    let structural = candidate(tree(&ellipsis), "((0,1),2)").supported().unwrap();
    assert_eq!(
        structural.cost().broadcast_amplification_elements(),
        EinsumCostBound::Exact(39)
    );
    let concrete = ellipsis
        .resolve_concrete_contraction_tree(&[&[3], &[2, 7, 3, 5], &[2, 7, 5]])
        .unwrap()
        .unwrap();
    assert_eq!(
        concrete
            .candidates()
            .iter()
            .find(|candidate| candidate.id().as_str() == "((0,1),2)")
            .unwrap()
            .cost()
            .unwrap()
            .broadcast_amplification_elements(),
        39
    );

    let singleton = plan("...i,...ij,...j->...", &[&[1, 3], &[7, 3, 5], &[7, 5]]);
    assert_eq!(
        candidate(tree(&singleton), "((0,1),2)")
            .supported()
            .unwrap()
            .cost()
            .broadcast_amplification_elements(),
        EinsumCostBound::Exact(18)
    );
}

#[test]
fn chain_order_flips_for_opposite_asymmetry() {
    for (shape, expected_pair) in [
        ((2usize, 100usize, 3usize, 2usize), [0, 1]),
        ((100usize, 2usize, 3usize, 100usize), [1, 2]),
    ] {
        let (a, b, c, d) = shape;
        let plan = plan("ab,bc,cd->ad", &[&[a, b], &[b, c], &[c, d]]);
        let selected = tree(&plan).preferred_candidate().unwrap();

        // Independent direct-loop operation counts for the two conventional
        // chain orders. This intentionally does not call the production
        // comparator or reconstruct its metric tuple.
        let left_first = a * c * (2 * b - 1) + a * d * (2 * c - 1);
        let right_first = b * d * (2 * c - 1) + a * d * (2 * b - 1);
        let independent_choice = if left_first < right_first {
            [0, 1]
        } else {
            [1, 2]
        };
        assert_eq!(independent_choice, expected_pair);
        assert_eq!(unordered_pair(selected.first_pair()), expected_pair);
    }
}

#[test]
fn tensor_vector_vector_tree_preserves_only_requested_k() {
    let plan = plan("ijk,i,j->k", &[&[3, 5, 7], &[3], &[5]]);
    let selected = tree(&plan).preferred_candidate().unwrap();
    // Forming i⊗j once (15 multiplies) and contracting both axes for each k
    // costs less here than separately materializing jk or ik.
    assert_eq!(unordered_pair(selected.first_pair()), [1, 2]);
    assert_eq!(plan.output_shape()[0].as_static(), Some(7));
}

#[test]
fn diagonal_local_reduction_mixed_case_and_implicit_output_are_preserved() {
    let diagonal = plan("iik,ij,j->k", &[&[3, 3, 7], &[3, 5], &[5]]);
    assert_eq!(diagonal.operands()[0].diagonal_axis_indices(), &[0]);
    assert!(matches!(
        diagonal.classification(),
        EinsumClassification::ContractionTree(_)
    ));

    let local = plan("xAi,Aij,j->", &[&[2, 3, 5], &[3, 5, 7], &[7]]);
    let tree = tree(&local);
    let candidate = candidate(tree, "((0,1),2)").supported().unwrap();
    let unary = candidate
        .steps()
        .iter()
        .find_map(|step| match step {
            EinsumContractionTreeStep::UnaryReduction(unary) => Some(unary),
            EinsumContractionTreeStep::BinaryContraction(_) => None,
            _ => None,
        })
        .unwrap();
    assert_eq!(
        unary
            .reduction_axes()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["label `x`"]
    );
    assert_eq!(
        local
            .output_axes()
            .iter()
            .map(|axis| match axis {
                EinsumAxis::Label(label) => label.as_char(),
                EinsumAxis::Ellipsis(_) => '.',
            })
            .collect::<String>(),
        ""
    );

    let implicit = plan("Ai,ij,j", &[&[3, 5], &[5, 7], &[7]]);
    assert_eq!(
        implicit
            .output_axes()
            .iter()
            .map(|axis| match axis {
                EinsumAxis::Label(label) => label.as_char(),
                EinsumAxis::Ellipsis(_) => '.',
            })
            .collect::<String>(),
        "A"
    );
}

#[test]
fn three_input_four_input_and_three_way_reductions_have_distinct_reasons() {
    let three = plan("ab,bc,cd->ad", &[&[2, 3], &[3, 4], &[4, 5]]);
    assert!(matches!(
        three.classification(),
        EinsumClassification::ContractionTree(_)
    ));

    let four = plan("ab,bc,cd,de->ae", &[&[2, 3], &[3, 4], &[4, 5], &[5, 6]]);
    assert!(matches!(
        four.classification(),
        EinsumClassification::Unsupported(
            EinsumUnsupportedReason::InputCountExceedsContractionTreeLimit {
                input_count: 4,
                maximum: 3,
            }
        )
    ));

    let three_way = plan("i,i,i->", &[&[5], &[5], &[5]]);
    assert!(matches!(
        three_way.classification(),
        EinsumClassification::Unsupported(
            EinsumUnsupportedReason::ReducedAxisSpansTooManyOperands {
                maximum: 2,
                axes,
            }
        ) if axes.len() == 1
    ));

    let reduced_ellipsis = plan("...i,...ij,j->", &[&[2, 3], &[2, 3, 5], &[5]]);
    assert!(matches!(
        reduced_ellipsis.classification(),
        EinsumClassification::Unsupported(EinsumUnsupportedReason::ReducedEllipsis {
            axes,
        }) if axes == &[EinsumAxis::Ellipsis(0)]
    ));
}

#[test]
fn zero_and_symbolic_costs_are_exact_or_unbounded_without_fabrication() {
    let zero = plan("i,ij,j->", &[&[0], &[0, 5], &[5]]);
    let selected = tree(&zero)
        .preferred_candidate()
        .unwrap()
        .supported()
        .unwrap();
    assert_eq!(selected.cost().flops(), EinsumCostBound::Exact(0));

    let i = Dim::Symbolic(SymbolId(1));
    let j = Dim::Symbolic(SymbolId(2));
    let left = [i];
    let matrix = [i, j];
    let right = [j];
    let symbolic_inputs = [
        EinsumInput::new(DataType::Float16, &left),
        EinsumInput::new(DataType::Float16, &matrix),
        EinsumInput::new(DataType::Float16, &right),
    ];
    let symbolic = EinsumPlan::build("i,ij,j->", &symbolic_inputs).unwrap();
    let symbolic_tree = tree(&symbolic);
    assert!(symbolic_tree.requires_concrete_rescore());
    assert_eq!(
        symbolic_tree
            .preferred_candidate()
            .unwrap()
            .supported()
            .unwrap()
            .cost()
            .flops(),
        EinsumCostBound::UnknownUpperBound
    );

    let concrete = symbolic
        .resolve_concrete_contraction_tree(&[&[2], &[2, 64], &[64]])
        .unwrap()
        .unwrap();
    assert_eq!(
        unordered_pair(
            concrete
                .preferred_candidate()
                .unwrap()
                .id_pair(symbolic_tree)
        ),
        [1, 2]
    );
    assert_eq!(
        concrete
            .preferred_candidate()
            .unwrap()
            .cost()
            .unwrap()
            .intermediate_bytes()
            % 2,
        0
    );
}

trait ConcreteCandidateExt {
    fn id_pair(&self, tree: &onnx_runtime_ir::EinsumContractionTreePlan) -> [usize; 2];
}

impl ConcreteCandidateExt for onnx_runtime_ir::EinsumConcreteContractionTreeCandidate {
    fn id_pair(&self, tree: &onnx_runtime_ir::EinsumContractionTreePlan) -> [usize; 2] {
        tree.candidates()
            .iter()
            .find(|candidate| candidate.id() == self.id())
            .unwrap()
            .first_pair()
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn static_cost_overflow_declines_candidates_without_wrapping() {
    let i = usize::MAX / 2 + 1;
    let plan = plan("i,ij,j->", &[&[i], &[i, 1], &[1]]);
    let tree = tree(&plan);
    assert!(tree.preferred_candidate().is_none());
    assert!(
        tree.candidates()
            .iter()
            .all(|candidate| candidate.unsupported_reason().is_some())
    );
}

#[test]
fn candidate_order_ties_and_peak_liveness_are_stable() {
    let first = plan("i,ij,j->", &[&[4], &[4, 4], &[4]]);
    let second = plan("i,ij,j->", &[&[4], &[4, 4], &[4]]);
    let first_tree = tree(&first);
    let second_tree = tree(&second);
    let ids = first_tree
        .candidates()
        .iter()
        .map(|candidate| candidate.id().as_str())
        .collect::<Vec<_>>();
    assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(
        ids,
        second_tree
            .candidates()
            .iter()
            .map(|candidate| candidate.id().as_str())
            .collect::<Vec<_>>()
    );
    let preferred = first_tree.preferred_candidate().unwrap();
    let tied_minimum_id = first_tree
        .candidates()
        .iter()
        .filter(|candidate| {
            candidate
                .supported()
                .zip(preferred.supported())
                .is_some_and(|(left, right)| left.cost() == right.cost())
        })
        .map(|candidate| candidate.id())
        .min()
        .unwrap();
    assert_eq!(preferred.id(), tied_minimum_id);

    let liveness = plan("xi,yij,zj->", &[&[2, 3], &[5, 3, 7], &[11, 7]]);
    let candidate = candidate(tree(&liveness), "((0,1),2)").supported().unwrap();
    assert_eq!(candidate.cost().slot_count(), 3);
    assert_eq!(
        candidate.cost().peak_live_temporary_elements(),
        EinsumCostBound::Exact(31)
    );
    assert_eq!(
        candidate.cost().intermediate_elements(),
        EinsumCostBound::Exact(38)
    );
}

#[test]
fn final_output_permutation_and_concrete_geometry_are_public() {
    let plan = plan("ab,bc,cd->da", &[&[2, 3], &[3, 5], &[5, 7]]);
    let tree = tree(&plan);
    let candidate = candidate(tree, "((0,1),2)").supported().unwrap();
    assert_eq!(candidate.final_output_permutation(), &[1, 0]);

    let concrete = plan
        .resolve_concrete_contraction_tree(&[&[2, 3], &[3, 5], &[5, 7]])
        .unwrap()
        .unwrap();
    let resolved = concrete
        .candidates()
        .iter()
        .find(|candidate| candidate.id().as_str() == "((0,1),2)")
        .unwrap();
    assert_eq!(resolved.binary_geometries().len(), 2);
    assert_eq!(resolved.binary_geometries()[0].m(), 2);
    assert_eq!(resolved.binary_geometries()[0].k(), 3);
    assert_eq!(resolved.binary_geometries()[0].n(), 5);
}
