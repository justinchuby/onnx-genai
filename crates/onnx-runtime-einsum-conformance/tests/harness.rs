use std::collections::BTreeSet;

use onnx_runtime_einsum_conformance::{
    BackendKind, BackendObservation, CanonicalTensor, CaptureExpectation, CaseRecord,
    ComparisonMode, ConformanceDType, CorpusSnapshot, DEFAULT_GENERATOR, ForcedRoute,
    GeneratorConfig, PlannerQuality, RouteProbe, SchemaAuthority, ValueProfile, ValueSpec,
    WorkspaceClass, compare, corpus_digest, default_corpus, evaluate, generated_cases,
    malformed_cases, materialize_inputs, named_cases, verify_observation,
};

fn case(
    equation: &str,
    dtype: ConformanceDType,
    shapes: &[&[usize]],
    profile: ValueProfile,
) -> CaseRecord {
    CaseRecord {
        id: "manual".into(),
        equation: equation.into(),
        opset: if dtype == ConformanceDType::BFloat16 {
            28
        } else {
            12
        },
        dtype,
        input_shapes: shapes.iter().map(|shape| shape.to_vec()).collect(),
        values: ValueSpec { seed: 1, profile },
        limits: DEFAULT_GENERATOR.limits,
        route_probes: vec![],
    }
}

#[test]
fn pinned_schema_authority_proves_einsum_28_bf16_boundary() {
    SchemaAuthority::verify().unwrap();
    assert_eq!(SchemaAuthority::since_version(12).unwrap(), 12);
    assert_eq!(SchemaAuthority::since_version(27).unwrap(), 12);
    assert_eq!(SchemaAuthority::since_version(28).unwrap(), 28);
    for dtype in ConformanceDType::V12_TYPES {
        assert!(SchemaAuthority::supports(12, dtype).unwrap());
        assert!(SchemaAuthority::supports(28, dtype).unwrap());
    }
    assert!(!SchemaAuthority::supports(12, ConformanceDType::BFloat16).unwrap());
    assert!(!SchemaAuthority::supports(27, ConformanceDType::BFloat16).unwrap());
    assert!(SchemaAuthority::supports(28, ConformanceDType::BFloat16).unwrap());
}

#[test]
fn named_and_seeded_corpus_cover_required_semantics_with_bounded_resources() {
    let cases = default_corpus();
    let ids = cases
        .iter()
        .map(|case| case.id.as_str())
        .collect::<BTreeSet<_>>();
    for required in [
        "bilinear-dot-f32",
        "bilinear-dot-f16",
        "bilinear-dot-bf16-specials",
        "bilinear-batched",
        "bilinear-fixed-ellipsis",
        "chain-three-left-asymmetric",
        "chain-three-right-asymmetric",
        "trilinear-full-reduction",
        "trilinear-keep-k",
        "shared-three-way-reduction",
        "three-local-reductions",
        "reduced-fixed-ellipsis",
        "chain-4-way",
        "chain-8-way",
        "chain-16-way",
        "case-sensitive-all-52-labels",
        "scalar-times-vector",
        "outer-product",
        "hadamard-product",
        "zero-extent-reduction",
        "one-extent-broadcast",
        "integer-wrapping-i8",
        "integer-matmul-i32",
    ] {
        assert!(ids.contains(required), "missing required case {required}");
    }
    let arities = cases
        .iter()
        .map(|case| case.input_shapes.len())
        .collect::<BTreeSet<_>>();
    for arity in [1usize, 2, 3, 4, 8, 16] {
        assert!(arities.contains(&arity), "missing arity {arity}");
    }
    assert!(cases.iter().any(|case| !case.equation.contains("->")));
    assert!(cases.iter().any(|case| case.equation.contains("...")));
    assert!(cases.iter().any(|case| case.equation.contains(",,")));
    assert!(
        cases
            .iter()
            .any(|case| case.input_shapes.iter().flatten().any(|&dim| dim == 0))
    );
    for route in [
        ForcedRoute::GenericNative,
        ForcedRoute::OptimizedDp,
        ForcedRoute::OptimizedHeuristic,
        ForcedRoute::MatMul,
        ForcedRoute::CudaCublas,
    ] {
        assert!(
            cases
                .iter()
                .flat_map(|case| &case.route_probes)
                .any(|probe| probe.route == route),
            "missing forced-route record {route:?}"
        );
    }
    for case in &cases {
        case.validate()
            .unwrap_or_else(|error| panic!("{}: {error}", case.id));
        assert!(
            case.route_probes.iter().any(|probe| {
                probe.backend == BackendKind::Cpu && probe.route == ForcedRoute::GenericNative
            }),
            "{} lacks CPU GenericNative handoff",
            case.id
        );
        assert!(
            case.route_probes.iter().any(|probe| {
                probe.backend == BackendKind::Cuda && probe.route == ForcedRoute::GenericNative
            }),
            "{} lacks CUDA GenericNative handoff",
            case.id
        );
        let inputs = materialize_inputs(case).unwrap();
        let result = evaluate(case, &inputs).unwrap();
        assert_eq!(result.output().dtype(), case.dtype, "{}", case.id);
    }
}

#[test]
fn generator_is_seed_deterministic_and_seed_sensitive() {
    let first = generated_cases(DEFAULT_GENERATOR).unwrap();
    let second = generated_cases(DEFAULT_GENERATOR).unwrap();
    assert_eq!(first, second);
    let changed = generated_cases(GeneratorConfig {
        seed: DEFAULT_GENERATOR.seed ^ 1,
        ..DEFAULT_GENERATOR
    })
    .unwrap();
    assert_ne!(corpus_digest(&first), corpus_digest(&changed));
    let error = generated_cases(GeneratorConfig {
        max_operands: 0,
        ..DEFAULT_GENERATOR
    })
    .unwrap_err();
    assert!(error.to_string().contains("1..=32"));
}

#[test]
fn checked_in_snapshot_locks_generated_case_identity_and_counts() {
    let snapshot: CorpusSnapshot =
        serde_json::from_str(include_str!("../fixtures/corpus-v1.json")).unwrap();
    assert_eq!(snapshot.generator, DEFAULT_GENERATOR);
    let cases = default_corpus();
    let malformed_count = malformed_cases().len();
    let route_probe_count: usize = cases.iter().map(|case| case.route_probes.len()).sum();
    let digest = corpus_digest(&cases);
    eprintln!(
        "snapshot: cases={}, malformed={}, routes={}, sha256={digest}",
        cases.len(),
        malformed_count,
        route_probe_count
    );
    assert_eq!(cases.len(), snapshot.expected_case_count);
    assert_eq!(malformed_count, snapshot.expected_malformed_count);
    assert_eq!(route_probe_count, snapshot.expected_route_probe_count);
    assert_eq!(digest, snapshot.generated_sha256);
}

#[test]
fn malformed_corpus_is_nonvacuous_and_covers_every_required_category() {
    let cases = malformed_cases();
    assert!(cases.len() >= 20);
    let categories = cases.iter().map(|case| case.kind).collect::<BTreeSet<_>>();
    assert_eq!(categories.len(), 11);
    for case in cases {
        let error = case.validate_failure().expect_err(&case.id);
        assert!(
            error.to_string().contains(&case.expected_fragment),
            "{}: {error}",
            case.id
        );
    }
}

#[test]
fn direct_oracle_uses_fixed_factor_reduction_order_and_integer_modulo() {
    let dot = case(
        "i,i->",
        ConformanceDType::Float32,
        &[&[2], &[2]],
        ValueProfile::Finite,
    );
    let dot_inputs = [
        CanonicalTensor::new(
            ConformanceDType::Float32,
            vec![2],
            vec![u64::from(1.0f32.to_bits()), u64::from(2.0f32.to_bits())],
        )
        .unwrap(),
        CanonicalTensor::new(
            ConformanceDType::Float32,
            vec![2],
            vec![u64::from(3.0f32.to_bits()), u64::from(4.0f32.to_bits())],
        )
        .unwrap(),
    ];
    let result = evaluate(&dot, &dot_inputs).unwrap();
    assert_eq!(result.output().to_f64_values(), [11.0]);

    let wrapping = case(
        "i,i->i",
        ConformanceDType::Int8,
        &[&[2], &[2]],
        ValueProfile::IntegerEdges,
    );
    let integer_inputs = [
        CanonicalTensor::new(ConformanceDType::Int8, vec![2], vec![0x7f, 0x80]).unwrap(),
        CanonicalTensor::new(ConformanceDType::Int8, vec![2], vec![2, 2]).unwrap(),
    ];
    let result = evaluate(&wrapping, &integer_inputs).unwrap();
    assert_eq!(result.output().raw_bits(), &[0xfe, 0x00]);
}

#[test]
fn direct_oracle_executes_every_schema_numeric_dtype() {
    let dtypes = ConformanceDType::V12_TYPES
        .into_iter()
        .chain([ConformanceDType::BFloat16]);
    for dtype in dtypes {
        let case = case("->", dtype, &[&[]], ValueProfile::Finite);
        let one = match dtype {
            ConformanceDType::Float16 => u64::from(half::f16::ONE.to_bits()),
            ConformanceDType::BFloat16 => u64::from(half::bf16::ONE.to_bits()),
            ConformanceDType::Float32 => u64::from(1.0f32.to_bits()),
            ConformanceDType::Float64 => 1.0f64.to_bits(),
            _ => 1,
        };
        let input = CanonicalTensor::new(dtype, vec![], vec![one]).unwrap();
        let result = evaluate(&case, &[input]).unwrap();
        assert_eq!(result.output().raw_bits(), &[one], "{dtype:?}");
    }
}

#[test]
fn bf16_special_values_promote_to_f32_and_narrow_once_at_output() {
    let identity = named_cases()
        .into_iter()
        .find(|case| case.id == "bf16-special-values-identity")
        .unwrap();
    let inputs = materialize_inputs(&identity).unwrap();
    let result = evaluate(&identity, &inputs).unwrap();
    assert_eq!(result.output().raw_bits(), inputs[0].raw_bits());

    let product = case(
        "i,i->",
        ConformanceDType::BFloat16,
        &[&[2], &[2]],
        ValueProfile::Finite,
    );
    let left = CanonicalTensor::new(
        ConformanceDType::BFloat16,
        vec![2],
        vec![
            u64::from(half::bf16::from_f32(1.5).to_bits()),
            u64::from(half::bf16::from_f32(2.25).to_bits()),
        ],
    )
    .unwrap();
    let right = CanonicalTensor::new(
        ConformanceDType::BFloat16,
        vec![2],
        vec![
            u64::from(half::bf16::from_f32(3.0).to_bits()),
            u64::from(half::bf16::from_f32(-0.5).to_bits()),
        ],
    )
    .unwrap();
    let result = evaluate(&product, &[left, right]).unwrap();
    let expected = half::bf16::from_f32(1.5f32 * 3.0 + 2.25 * -0.5).to_bits();
    assert_eq!(result.output().raw_bits(), &[u64::from(expected)]);
}

#[test]
fn route_contract_rejects_unforced_route_workspace_and_capture_drift() {
    let case = named_cases()
        .into_iter()
        .find(|case| case.id == "bilinear-dot-f32")
        .unwrap();
    let inputs = materialize_inputs(&case).unwrap();
    let expected = evaluate(&case, &inputs).unwrap();
    let probe = RouteProbe {
        backend: BackendKind::Cuda,
        route: ForcedRoute::OptimizedDp,
        planner_quality: Some(PlannerQuality::ExactSubsetDp),
        comparison: ComparisonMode::CanonicalBits,
        workspace: WorkspaceClass::Gpu64MiB,
        capture: CaptureExpectation::MustCapture,
    };
    let good = BackendObservation {
        backend: BackendKind::Cuda,
        route: ForcedRoute::OptimizedDp,
        planner_quality: Some(PlannerQuality::ExactSubsetDp),
        workspace_bytes: 1024,
        captures: 1,
        replays: 1,
        capture_fallbacks: 0,
        output: expected.output().clone(),
    };
    verify_observation(&case, &expected, &probe, &good).unwrap();

    let wrong_route = BackendObservation {
        route: ForcedRoute::GenericNative,
        ..good.clone()
    };
    assert!(verify_observation(&case, &expected, &probe, &wrong_route).is_err());
    let wrong_backend = BackendObservation {
        backend: BackendKind::Cpu,
        ..good.clone()
    };
    assert!(verify_observation(&case, &expected, &probe, &wrong_backend).is_err());
    let wrong_quality = BackendObservation {
        planner_quality: None,
        ..good.clone()
    };
    assert!(verify_observation(&case, &expected, &probe, &wrong_quality).is_err());
    let excessive_workspace = BackendObservation {
        workspace_bytes: probe.workspace.max_bytes(),
        ..good.clone()
    };
    assert!(verify_observation(&case, &expected, &probe, &excessive_workspace).is_err());
    let uncaptured = BackendObservation {
        captures: 0,
        replays: 0,
        ..good
    };
    assert!(verify_observation(&case, &expected, &probe, &uncaptured).is_err());
}

#[test]
fn condition_aware_f16_f32_f64_rules_are_distinct_from_exact_bits() {
    for dtype in [
        ConformanceDType::Float16,
        ConformanceDType::Float32,
        ConformanceDType::Float64,
    ] {
        let case = case("->", dtype, &[&[]], ValueProfile::Finite);
        let one = match dtype {
            ConformanceDType::Float16 => u64::from(half::f16::ONE.to_bits()),
            ConformanceDType::Float32 => u64::from(1.0f32.to_bits()),
            ConformanceDType::Float64 => 1.0f64.to_bits(),
            _ => unreachable!(),
        };
        let input = CanonicalTensor::new(dtype, vec![], vec![one]).unwrap();
        let expected = evaluate(&case, &[input]).unwrap();
        let nearby = CanonicalTensor::new(dtype, vec![], vec![one + 1]).unwrap();
        assert!(
            compare(&case, &expected, &nearby, ComparisonMode::CanonicalBits).is_err(),
            "{dtype:?}"
        );
        compare(&case, &expected, &nearby, ComparisonMode::ConditionAware)
            .unwrap_or_else(|error| panic!("{dtype:?}: {error}"));
    }
}

#[test]
fn special_comparison_checks_nan_infinity_and_signed_zero_separately() {
    let case = named_cases()
        .into_iter()
        .find(|case| case.id == "bf16-special-values-identity")
        .unwrap();
    let inputs = materialize_inputs(&case).unwrap();
    let expected = evaluate(&case, &inputs).unwrap();

    let mut signed_zero = expected.output().raw_bits().to_vec();
    signed_zero[1] = 0x0000;
    let signed_zero = CanonicalTensor::new(case.dtype, vec![9], signed_zero).unwrap();
    assert!(
        compare(
            &case,
            &expected,
            &signed_zero,
            ComparisonMode::ConditionAware
        )
        .is_err()
    );

    let mut infinity = expected.output().raw_bits().to_vec();
    infinity[4] = 0xff80;
    let infinity = CanonicalTensor::new(case.dtype, vec![9], infinity).unwrap();
    assert!(compare(&case, &expected, &infinity, ComparisonMode::ConditionAware).is_err());

    let mut nan = expected.output().raw_bits().to_vec();
    nan[6] = 0x7fff;
    let nan = CanonicalTensor::new(case.dtype, vec![9], nan).unwrap();
    compare(&case, &expected, &nan, ComparisonMode::ConditionAware).unwrap();
    assert!(compare(&case, &expected, &nan, ComparisonMode::CanonicalBits).is_err());
}
