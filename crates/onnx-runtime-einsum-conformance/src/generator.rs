use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    BackendKind, CPU_WORKING_SET_BYTES, CaptureExpectation, CaseLimits, CaseRecord, ComparisonMode,
    ConformanceDType, DeclaredDType, ForcedRoute, GPU_CASE_BYTES, MalformedCase, MalformedKind,
    PlannerQuality, RouteProbe, UNIT_TENSOR_BYTES, ValueProfile, ValueSpec, WorkspaceClass,
};

const LABELS: &[u8; 52] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
/// Maximum retained/transient generator metadata admitted in one call.
///
/// This is an implementation resource guard, not an ONNX operand-arity limit.
pub const GENERATOR_METADATA_BYTES: usize = 16 * 1024 * 1024;

/// Seeded bounded generator configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Stable generator seed.
    pub seed: u64,
    /// Number of random legal records after the named corpus.
    pub random_cases: usize,
    /// Minimum sampled operand arity. ONNX requires at least one input.
    pub min_operands: usize,
    /// Maximum sampled operand arity. ONNX itself imposes no maximum.
    pub max_operands: usize,
    /// Maximum physical rank of one generated operand.
    pub max_rank: usize,
    /// Maximum generated dimension extent.
    pub max_dimension: usize,
    /// Maximum element count of any generated input or output tensor.
    pub max_tensor_elements: usize,
    /// Maximum aggregate input/output element count of one generated case.
    pub max_total_elements: usize,
    /// Resource ceilings.
    pub limits: CaseLimits,
}

/// Repository-default corpus generator.
pub const DEFAULT_GENERATOR: GeneratorConfig = GeneratorConfig {
    seed: 0x0E15_2A28_5EED_C0DE,
    random_cases: 128,
    min_operands: 1,
    max_operands: 16,
    max_rank: 8,
    max_dimension: 3,
    max_tensor_elements: 2_000_000,
    max_total_elements: 4_000_000,
    limits: CaseLimits {
        unit_tensor_bytes: UNIT_TENSOR_BYTES,
        cpu_working_set_bytes: CPU_WORKING_SET_BYTES,
        gpu_case_bytes: GPU_CASE_BYTES,
        oracle_work_items: 2_000_000,
    },
};

/// Named targets plus the seeded property corpus.
pub fn default_corpus() -> Vec<CaseRecord> {
    let mut cases = named_cases();
    cases.extend(
        generated_cases(DEFAULT_GENERATOR)
            .expect("repository-default Einsum generator configuration is valid"),
    );
    cases
}

/// Required bilinear, trilinear, arity, diagonal, scalar, ellipsis, and
/// extent-edge records.
pub fn named_cases() -> Vec<CaseRecord> {
    let mut cases = Vec::new();
    let mut push = |id: &str,
                    equation: &str,
                    shapes: &[&[usize]],
                    dtype: ConformanceDType,
                    opset: u64,
                    profile: ValueProfile,
                    matmul: bool| {
        let arity = shapes.len();
        let mut route_probes = generic_route_probes();
        if optimized_named_case(id) {
            route_probes.extend(optimized_route_probes(arity));
        }
        if matmul {
            route_probes.extend(matmul_route_probes(dtype));
        }
        cases.push(CaseRecord {
            id: id.into(),
            equation: equation.into(),
            opset,
            dtype,
            input_shapes: shapes.iter().map(|shape| shape.to_vec()).collect(),
            values: ValueSpec {
                seed: stable_id_seed(id),
                profile,
            },
            limits: DEFAULT_GENERATOR.limits,
            route_probes,
        });
    };

    for (suffix, dtype, opset, profile) in [
        ("f32", ConformanceDType::Float32, 12, ValueProfile::Finite),
        ("f16", ConformanceDType::Float16, 12, ValueProfile::Finite),
        (
            "bf16-specials",
            ConformanceDType::BFloat16,
            28,
            ValueProfile::BinarySpecials,
        ),
    ] {
        push(
            &format!("bilinear-dot-{suffix}"),
            "i,ij,j->",
            &[&[3], &[3, 4], &[4]],
            dtype,
            opset,
            profile,
            false,
        );
    }
    push(
        "bilinear-batched",
        "bi,bij,bj->b",
        &[&[2, 3], &[2, 3, 4], &[2, 4]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "bilinear-fixed-ellipsis",
        "i,...ij,...j->...",
        &[&[3], &[2, 3, 4], &[2, 4]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "chain-three-left-asymmetric",
        "ab,bc,cd->ad",
        &[&[2, 7], &[7, 3], &[3, 11]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "chain-three-right-asymmetric",
        "ab,bc,cd->ad",
        &[&[11, 3], &[3, 7], &[7, 2]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "trilinear-full-reduction",
        "ijk,i,j,k->",
        &[&[2, 3, 4], &[2], &[3], &[4]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "trilinear-keep-k",
        "ijk,i,j->k",
        &[&[2, 3, 4], &[2], &[3]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "shared-three-way-reduction",
        "i,i,i->",
        &[&[5], &[5], &[5]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "three-local-reductions",
        "abcd,a->",
        &[&[2, 3, 2, 4], &[2]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "reduced-fixed-ellipsis",
        "...i,...i->",
        &[&[2, 1, 3], &[2, 4, 3]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "retained-fixed-ellipsis",
        "...i,...i->...i",
        &[&[0, 1, 3], &[0, 4, 3]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "implicit-fixed-ellipsis",
        "ij,...jk",
        &[&[2, 3], &[4, 5, 3, 6]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "diagonal-extraction",
        "ii->i",
        &[&[5, 5]],
        ConformanceDType::Float16,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "bf16-special-values-identity",
        "i->i",
        &[&[9]],
        ConformanceDType::BFloat16,
        28,
        ValueProfile::BinarySpecials,
        false,
    );
    push(
        "bf16-finite-identity",
        "i->i",
        &[&[5]],
        ConformanceDType::BFloat16,
        28,
        ValueProfile::Finite,
        false,
    );
    push(
        "triple-diagonal-reduction",
        "iii->",
        &[&[3, 3, 3]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "scalar-times-vector",
        ",i->i",
        &[&[], &[7]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "two-scalars",
        ",->",
        &[&[], &[]],
        ConformanceDType::Float64,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "outer-product",
        "i,j->ij",
        &[&[3], &[4]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "hadamard-product",
        "ij,ij->ij",
        &[&[2, 4], &[2, 4]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "zero-extent-reduction",
        "i,i->",
        &[&[0], &[0]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "one-extent-broadcast",
        "...i,...i->...i",
        &[&[1, 3], &[5, 3]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    push(
        "matmul-explicit",
        "ik,kj->ij",
        &[&[3, 5], &[5, 4]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        true,
    );
    push(
        "integer-wrapping-i8",
        "i,i->i",
        &[&[8], &[8]],
        ConformanceDType::Int8,
        12,
        ValueProfile::IntegerEdges,
        false,
    );
    push(
        "integer-matmul-i32",
        "ik,kj->ij",
        &[&[2, 3], &[3, 2]],
        ConformanceDType::Int32,
        12,
        ValueProfile::IntegerEdges,
        true,
    );
    let all_labels = String::from_utf8(LABELS.to_vec()).expect("ASCII labels");
    push(
        "case-sensitive-all-52-labels",
        &format!("{all_labels}->{all_labels}"),
        &[&[1; 52]],
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    let high_arity = 64;
    let high_arity_equation = format!("{}->", ",".repeat(high_arity - 1));
    let high_arity_shapes = vec![Vec::new(); high_arity];
    let high_arity_refs = high_arity_shapes
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    push(
        "scalar-product-64-operands",
        &high_arity_equation,
        &high_arity_refs,
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );
    let max_depth_arity = 128;
    let max_depth_equation = format!("{}->", ",".repeat(max_depth_arity - 1));
    let max_depth_shapes = vec![Vec::new(); max_depth_arity];
    let max_depth_refs = max_depth_shapes
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    push(
        "scalar-product-128-operands",
        &max_depth_equation,
        &max_depth_refs,
        ConformanceDType::Float32,
        12,
        ValueProfile::Finite,
        false,
    );

    for arity in [4usize, 8, 16] {
        let (equation, shapes) = chain(arity);
        let refs = shapes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        push(
            &format!("chain-{arity}-way"),
            &equation,
            &refs,
            ConformanceDType::Float32,
            12,
            ValueProfile::Finite,
            false,
        );
    }

    for case in &cases {
        case.validate()
            .unwrap_or_else(|error| panic!("named case {} is invalid: {error}", case.id));
    }
    cases
}

/// Generate bounded legal expressions from a deterministic seed.
pub fn generated_cases(config: GeneratorConfig) -> Result<Vec<CaseRecord>, GeneratorError> {
    if config.min_operands == 0 {
        return Err(GeneratorError::MinOperands);
    }
    if config.min_operands > config.max_operands {
        return Err(GeneratorError::OperandRange {
            min: config.min_operands,
            max: config.max_operands,
        });
    }
    let attempt_limit = config
        .random_cases
        .checked_mul(500)
        .map(|attempts| attempts.max(500))
        .ok_or(GeneratorError::CaseCountOverflow(config.random_cases))?;
    let metadata_limit = config
        .limits
        .cpu_working_set_bytes
        .min(GENERATOR_METADATA_BYTES);
    let minimum_corpus_metadata = config
        .random_cases
        .checked_mul(std::mem::size_of::<CaseRecord>())
        .ok_or(GeneratorError::CaseCountOverflow(config.random_cases))?;
    if minimum_corpus_metadata > metadata_limit {
        return Err(GeneratorError::CorpusMetadata {
            requested: config.random_cases,
            bytes: minimum_corpus_metadata,
            limit: metadata_limit,
        });
    }
    if config.random_cases == 0 {
        return Ok(Vec::new());
    }
    let per_operand_metadata = std::mem::size_of::<String>()
        + 2 * std::mem::size_of::<Vec<usize>>()
        + 6 * std::mem::size_of::<usize>()
        + 8;
    let metadata_base = std::mem::size_of::<CaseRecord>() + 2;
    let practical_max_operands =
        metadata_limit.saturating_sub(metadata_base) / per_operand_metadata;
    let sampled_max_operands = config.max_operands.min(practical_max_operands);
    if config.min_operands > sampled_max_operands {
        return Err(GeneratorError::Unsatisfiable {
            field: "min_operands",
            value: config.min_operands,
            reason: "the checked metadata budget cannot hold the equation and operand metadata",
        });
    }
    let mut rng = SplitMix64::new(config.seed);
    let required_arities = [1usize, 2, 3, 4, 8, 16];
    let mut cases = Vec::new();
    let mut retained_metadata = 0usize;
    let mut attempts = 0usize;
    while cases.len() < config.random_cases {
        attempts = attempts
            .checked_add(1)
            .ok_or(GeneratorError::CaseCountOverflow(config.random_cases))?;
        if attempts > attempt_limit {
            return Err(GeneratorError::Exhausted {
                generated: cases.len(),
                requested: config.random_cases,
            });
        }
        let index = cases.len();
        let arity = required_arities
            .get(index)
            .copied()
            .map(|arity| arity.clamp(config.min_operands, sampled_max_operands))
            .unwrap_or_else(|| rng.range_inclusive(config.min_operands, sampled_max_operands));
        let dtype = match index % 29 {
            3 => ConformanceDType::Float16,
            7 => ConformanceDType::BFloat16,
            11 => ConformanceDType::Float64,
            17 => ConformanceDType::Uint8,
            23 => ConformanceDType::Int16,
            _ => ConformanceDType::Float32,
        };
        let opset = if dtype == ConformanceDType::BFloat16 {
            28
        } else {
            12
        };
        let profile = if dtype.integer_bits().is_some() {
            ValueProfile::IntegerEdges
        } else if index % 31 == 13 {
            ValueProfile::BinarySpecials
        } else {
            ValueProfile::Finite
        };
        let candidate = random_case(&mut rng, index, arity, dtype, opset, profile, config);
        if generated_case_is_within_config(&candidate, config) {
            let candidate_metadata = generated_case_metadata_bytes(&candidate)
                .ok_or(GeneratorError::CaseCountOverflow(config.random_cases))?;
            let next_metadata = retained_metadata
                .checked_add(candidate_metadata)
                .ok_or(GeneratorError::CaseCountOverflow(config.random_cases))?;
            if next_metadata > metadata_limit {
                return Err(GeneratorError::CorpusMetadata {
                    requested: config.random_cases,
                    bytes: next_metadata,
                    limit: metadata_limit,
                });
            }
            retained_metadata = next_metadata;
            cases.push(candidate);
        }
    }
    Ok(cases)
}

/// Stable malformed syntax, arity, rank, dimension, and dtype corpus.
pub fn malformed_cases() -> Vec<MalformedCase> {
    let f32 = DeclaredDType::Numeric(ConformanceDType::Float32);
    let bf16 = DeclaredDType::Numeric(ConformanceDType::BFloat16);
    let mut cases = vec![
        malformed(
            "syntax-dollar",
            "i$->i",
            12,
            &[f32],
            &[&[2]],
            1,
            MalformedKind::Syntax,
            "invalid character",
        ),
        malformed(
            "syntax-multiple-arrows",
            "i->i->i",
            12,
            &[f32],
            &[&[2]],
            1,
            MalformedKind::Syntax,
            "more than one",
        ),
        malformed(
            "syntax-broken-arrow",
            "i-i",
            12,
            &[f32],
            &[&[2]],
            1,
            MalformedKind::Syntax,
            "invalid character",
        ),
        malformed(
            "output-duplicate",
            "i->ii",
            12,
            &[f32],
            &[&[2]],
            1,
            MalformedKind::Output,
            "appears more than once",
        ),
        malformed(
            "output-unknown",
            "i->j",
            12,
            &[f32],
            &[&[2]],
            1,
            MalformedKind::Output,
            "does not appear",
        ),
        malformed(
            "ellipsis-input-multiple",
            "...i...->i",
            12,
            &[f32],
            &[&[2, 3]],
            1,
            MalformedKind::Ellipsis,
            "more than one ellipsis",
        ),
        malformed(
            "ellipsis-output-multiple",
            "...i->......i",
            12,
            &[f32],
            &[&[2, 3]],
            1,
            MalformedKind::Ellipsis,
            "more than one ellipsis",
        ),
        malformed(
            "ellipsis-fixed-rank",
            "...i,...i->...i",
            12,
            &[f32, f32],
            &[&[2, 3], &[4, 5, 3]],
            1,
            MalformedKind::Ellipsis,
            "expansion rank",
        ),
        malformed(
            "syntax-tab",
            "i\t->i",
            12,
            &[f32],
            &[&[2]],
            1,
            MalformedKind::Syntax,
            "invalid character",
        ),
        malformed(
            "syntax-unicode",
            "λ->λ",
            12,
            &[f32],
            &[&[2]],
            1,
            MalformedKind::Syntax,
            "invalid character",
        ),
        malformed(
            "zero-inputs",
            "i->i",
            12,
            &[],
            &[],
            1,
            MalformedKind::NoInputs,
            "at least one input",
        ),
        malformed(
            "zero-outputs",
            "i->i",
            12,
            &[f32],
            &[&[2]],
            0,
            MalformedKind::OutputCount,
            "exactly 1 output",
        ),
        malformed(
            "two-outputs",
            "i->i",
            12,
            &[f32],
            &[&[2]],
            2,
            MalformedKind::OutputCount,
            "exactly 1 output",
        ),
        malformed(
            "rank-mismatch",
            "i->i",
            12,
            &[f32],
            &[&[2, 3]],
            1,
            MalformedKind::Rank,
            "rank 2",
        ),
        malformed(
            "diagonal-dimension",
            "ii->i",
            12,
            &[f32],
            &[&[2, 3]],
            1,
            MalformedKind::DiagonalDimension,
            "diagonal label",
        ),
        malformed(
            "named-dimension",
            "i,i->i",
            12,
            &[f32, f32],
            &[&[2], &[3]],
            1,
            MalformedKind::Dimension,
            "equal dimensions",
        ),
        malformed(
            "nonbroadcast-ellipsis",
            "...i,...i->...i",
            12,
            &[f32, f32],
            &[&[2, 3], &[5, 3]],
            1,
            MalformedKind::Dimension,
            "cannot broadcast",
        ),
        malformed(
            "mixed-dtypes",
            "i,i->i",
            12,
            &[f32, DeclaredDType::Numeric(ConformanceDType::Float16)],
            &[&[2], &[2]],
            1,
            MalformedKind::MixedDType,
            "does not match",
        ),
        malformed(
            "nonnumeric-bool",
            "i->i",
            12,
            &[DeclaredDType::Bool],
            &[&[2]],
            1,
            MalformedKind::Nonnumeric,
            "numeric tensors",
        ),
        malformed(
            "nonnumeric-string",
            "i->i",
            12,
            &[DeclaredDType::String],
            &[&[2]],
            1,
            MalformedKind::Nonnumeric,
            "numeric tensors",
        ),
        malformed(
            "nonnumeric-complex64",
            "i->i",
            12,
            &[DeclaredDType::Complex64],
            &[&[2]],
            1,
            MalformedKind::Nonnumeric,
            "numeric tensors",
        ),
        malformed(
            "nonnumeric-complex128",
            "i->i",
            12,
            &[DeclaredDType::Complex128],
            &[&[2]],
            1,
            MalformedKind::Nonnumeric,
            "numeric tensors",
        ),
        malformed(
            "bf16-opset12",
            "i->i",
            12,
            &[bf16],
            &[&[2]],
            1,
            MalformedKind::Bf16Before28,
            "not admitted by Einsum-12",
        ),
        malformed(
            "bf16-opset27",
            "i->i",
            27,
            &[bf16],
            &[&[2]],
            1,
            MalformedKind::Bf16Before28,
            "not admitted by Einsum-12",
        ),
    ];
    for case in &cases {
        let error = case
            .validate_failure()
            .expect_err("malformed record must be rejected");
        assert!(
            error.to_string().contains(&case.expected_fragment),
            "{}: expected {:?} in {error}",
            case.id,
            case.expected_fragment
        );
    }
    cases.shrink_to_fit();
    cases
}

fn random_case(
    rng: &mut SplitMix64,
    index: usize,
    arity: usize,
    dtype: ConformanceDType,
    opset: u64,
    profile: ValueProfile,
    config: GeneratorConfig,
) -> CaseRecord {
    let max_dimension = practical_dimension_limit(config, dtype);
    let label_count = rng.range_inclusive(1, 8.min(LABELS.len()));
    let mut labels = LABELS.to_vec();
    for i in (1..labels.len()).rev() {
        let swap = rng.range_inclusive(0, i);
        labels.swap(i, swap);
    }
    labels.truncate(label_count);
    let label_extents = labels
        .iter()
        .map(|&label| (label, random_extent(rng, max_dimension)))
        .collect::<BTreeMap<_, _>>();
    let use_ellipsis = config.max_rank > 0 && rng.next().is_multiple_of(2);
    let ellipsis_rank = if use_ellipsis {
        rng.range_inclusive(0, config.max_rank.min(2))
    } else {
        0
    };
    let ellipsis_extents = (0..ellipsis_rank)
        .map(|_| random_extent(rng, max_dimension))
        .collect::<Vec<_>>();
    let mut terms = Vec::with_capacity(arity);
    let mut shapes = Vec::with_capacity(arity);
    let mut occurrences = BTreeMap::<u8, usize>::new();
    let mut any_ellipsis = false;
    for input in 0..arity {
        let scalar = config.max_rank == 0 || (arity > 1 && rng.next().is_multiple_of(11));
        let has_ellipsis = use_ellipsis && !scalar && (input == 0 || rng.next().is_multiple_of(2));
        any_ellipsis |= has_ellipsis;
        let max_named = config
            .max_rank
            .saturating_sub(if has_ellipsis { ellipsis_rank } else { 0 })
            .min(4);
        let named = if scalar {
            0
        } else {
            rng.range_inclusive(0, max_named)
        };
        let mut named_labels = Vec::with_capacity(named);
        for axis in 0..named {
            let label = if axis > 0 && rng.next().is_multiple_of(7) {
                named_labels[axis - 1]
            } else {
                labels[rng.range_inclusive(0, labels.len() - 1)]
            };
            named_labels.push(label);
            *occurrences.entry(label).or_default() += 1;
        }
        let ellipsis_position = if has_ellipsis {
            rng.range_inclusive(0, named_labels.len())
        } else {
            named_labels.len()
        };
        let mut term = String::new();
        let mut shape = Vec::new();
        for position in 0..=named_labels.len() {
            if has_ellipsis && position == ellipsis_position {
                term.push_str("...");
                for &extent in &ellipsis_extents {
                    let operand_extent = if rng.next().is_multiple_of(3) {
                        1
                    } else {
                        extent
                    };
                    shape.push(operand_extent);
                }
            }
            if let Some(&label) = named_labels.get(position) {
                term.push(label as char);
                shape.push(label_extents[&label]);
            }
        }

        terms.push(term);
        shapes.push(shape);
    }
    let explicit = rng.next().is_multiple_of(2);
    let equation = if explicit {
        let mut output = String::new();
        if any_ellipsis && rng.next().is_multiple_of(2) {
            output.push_str("...");
        }
        let used = occurrences.keys().copied().collect::<Vec<_>>();
        for label in used {
            if rng.next().is_multiple_of(3) {
                output.push(label as char);
            }
        }
        format!("{}->{output}", terms.join(","))
    } else {
        terms.join(",")
    };
    CaseRecord {
        id: format!("generated-{index:03}"),
        equation,
        opset,
        dtype,
        input_shapes: shapes,
        values: ValueSpec {
            seed: rng.next(),
            profile,
        },
        limits: config.limits,
        route_probes: generic_route_probes(),
    }
}

fn generated_case_is_within_config(case: &CaseRecord, config: GeneratorConfig) -> bool {
    if !(config.min_operands..=config.max_operands).contains(&case.input_shapes.len()) {
        return false;
    }
    let mut total_elements = 0usize;
    for shape in &case.input_shapes {
        if shape.len() > config.max_rank
            || shape
                .iter()
                .any(|&dimension| dimension > config.max_dimension)
        {
            return false;
        }
        let Some(elements) = checked_product(shape.iter().copied()) else {
            return false;
        };
        if elements > config.max_tensor_elements {
            return false;
        }
        let Some(updated) = total_elements.checked_add(elements) else {
            return false;
        };
        total_elements = updated;
    }
    let Ok(analysis) = crate::analyze_equation(&case.equation, &case.input_shapes) else {
        return false;
    };
    let Some(output_elements) = checked_product(analysis.output_shape().iter().copied()) else {
        return false;
    };
    if output_elements > config.max_tensor_elements {
        return false;
    }
    let Some(total_elements) = total_elements.checked_add(output_elements) else {
        return false;
    };
    total_elements <= config.max_total_elements && case.validate().is_ok()
}

fn generated_case_metadata_bytes(case: &CaseRecord) -> Option<usize> {
    let shape_bytes = case.input_shapes.iter().try_fold(
        case.input_shapes
            .len()
            .checked_mul(std::mem::size_of::<Vec<usize>>())?,
        |bytes, shape| {
            shape
                .len()
                .checked_mul(std::mem::size_of::<usize>())
                .and_then(|shape_bytes| bytes.checked_add(shape_bytes))
        },
    )?;
    let route_bytes = case
        .route_probes
        .len()
        .checked_mul(std::mem::size_of::<RouteProbe>())?;
    std::mem::size_of::<CaseRecord>()
        .checked_add(case.id.len())?
        .checked_add(case.equation.len())?
        .checked_add(shape_bytes)?
        .checked_add(route_bytes)
}

fn practical_dimension_limit(config: GeneratorConfig, dtype: ConformanceDType) -> usize {
    config
        .max_dimension
        .min(config.max_tensor_elements)
        .min(config.max_total_elements)
        .min(config.limits.unit_tensor_bytes / dtype.byte_size())
        .min(config.limits.gpu_case_bytes / dtype.byte_size())
        .min(config.limits.cpu_working_set_bytes / std::mem::size_of::<u64>())
}

fn generic_route_probes() -> Vec<RouteProbe> {
    vec![
        RouteProbe {
            backend: BackendKind::Cpu,
            route: ForcedRoute::GenericNative,
            planner_quality: None,
            comparison: ComparisonMode::ConditionAware,
            workspace: WorkspaceClass::Cpu32MiB,
            capture: CaptureExpectation::NotApplicable,
        },
        RouteProbe {
            backend: BackendKind::Cuda,
            route: ForcedRoute::GenericNative,
            planner_quality: None,
            comparison: ComparisonMode::ConditionAware,
            workspace: WorkspaceClass::Gpu64MiB,
            capture: CaptureExpectation::MustCapture,
        },
    ]
}

fn optimized_route_probes(arity: usize) -> Vec<RouteProbe> {
    let (route, quality) = if arity <= 5 {
        (ForcedRoute::OptimizedDp, PlannerQuality::ExactSubsetDp)
    } else {
        (
            ForcedRoute::OptimizedHeuristic,
            PlannerQuality::DeterministicHeuristic,
        )
    };
    let mut probes = Vec::new();
    for backend in [BackendKind::Cpu, BackendKind::Cuda] {
        probes.push(RouteProbe {
            backend,
            route,
            planner_quality: Some(quality),
            comparison: ComparisonMode::ConditionAware,
            workspace: if backend == BackendKind::Cpu {
                WorkspaceClass::Cpu32MiB
            } else {
                WorkspaceClass::Gpu64MiB
            },
            capture: if backend == BackendKind::Cpu {
                CaptureExpectation::NotApplicable
            } else {
                CaptureExpectation::MustCapture
            },
        });
    }
    probes
}

fn optimized_named_case(id: &str) -> bool {
    id.starts_with("bilinear-dot-")
        || matches!(
            id,
            "bilinear-batched"
                | "bilinear-fixed-ellipsis"
                | "chain-three-left-asymmetric"
                | "chain-three-right-asymmetric"
                | "trilinear-full-reduction"
                | "trilinear-keep-k"
                | "shared-three-way-reduction"
                | "chain-4-way"
                | "chain-8-way"
        )
}

fn matmul_route_probes(dtype: ConformanceDType) -> Vec<RouteProbe> {
    if !matches!(dtype, ConformanceDType::Float16 | ConformanceDType::Float32) {
        return Vec::new();
    }
    vec![
        RouteProbe {
            backend: BackendKind::Cpu,
            route: ForcedRoute::MatMul,
            planner_quality: None,
            comparison: ComparisonMode::ConditionAware,
            workspace: WorkspaceClass::Cpu32MiB,
            capture: CaptureExpectation::NotApplicable,
        },
        RouteProbe {
            backend: BackendKind::Cuda,
            route: ForcedRoute::CudaCublas,
            planner_quality: None,
            comparison: ComparisonMode::ConditionAware,
            workspace: WorkspaceClass::Gpu64MiB,
            capture: CaptureExpectation::MustCapture,
        },
    ]
}

fn chain(arity: usize) -> (String, Vec<Vec<usize>>) {
    assert!(arity < LABELS.len());
    let terms = (0..arity)
        .map(|index| format!("{}{}", LABELS[index] as char, LABELS[index + 1] as char))
        .collect::<Vec<_>>();
    let equation = format!(
        "{}->{}{}",
        terms.join(","),
        LABELS[0] as char,
        LABELS[arity] as char
    );
    let shapes = (0..arity).map(|_| vec![1, 1]).collect();
    (equation, shapes)
}

#[allow(clippy::too_many_arguments)]
fn malformed(
    id: &str,
    equation: &str,
    opset: u64,
    input_dtypes: &[DeclaredDType],
    input_shapes: &[&[usize]],
    output_count: usize,
    kind: MalformedKind,
    expected_fragment: &str,
) -> MalformedCase {
    MalformedCase {
        id: id.into(),
        equation: equation.into(),
        opset,
        input_dtypes: input_dtypes.to_vec(),
        input_shapes: input_shapes.iter().map(|shape| shape.to_vec()).collect(),
        output_count,
        kind,
        expected_fragment: expected_fragment.into(),
    }
}

fn random_extent(rng: &mut SplitMix64, max_dimension: usize) -> usize {
    match rng.next() % 10 {
        0 => 0,
        1..=3 => max_dimension.min(1),
        4..=7 => max_dimension.min(2),
        _ if max_dimension <= 3 => max_dimension,
        _ => rng.range_inclusive(3, max_dimension),
    }
}

fn checked_product(values: impl IntoIterator<Item = usize>) -> Option<usize> {
    values
        .into_iter()
        .try_fold(1usize, |product, value| product.checked_mul(value))
}

fn stable_id_seed(id: &str) -> u64 {
    id.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}

struct SplitMix64 {
    state: u64,
}

/// Invalid or unsatisfiable seeded generator configuration.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GeneratorError {
    /// ONNX Einsum requires at least one input.
    #[error("Einsum generator min_operands must be at least 1")]
    MinOperands,
    /// Sampled operand bounds are reversed.
    #[error(
        "Einsum generator min_operands {min} exceeds max_operands {max}; ONNX imposes no maximum, but the configured sampling interval must be nonempty"
    )]
    OperandRange {
        /// Inclusive lower bound.
        min: usize,
        /// Inclusive upper bound.
        max: usize,
    },
    /// Requested case arithmetic overflowed before allocation.
    #[error(
        "Einsum generator random_cases {0} overflows the checked retry or metadata budget calculation"
    )]
    CaseCountOverflow(usize),
    /// The retained corpus cannot fit its configured generation budget.
    #[error(
        "Einsum generator needs at least {bytes} bytes of retained metadata for {requested} random cases, exceeding the {limit}-byte checked metadata budget"
    )]
    CorpusMetadata {
        /// Requested random records.
        requested: usize,
        /// Minimum retained bytes.
        bytes: usize,
        /// Configured budget.
        limit: usize,
    },
    /// A requested lower bound cannot fit the configured resource ceilings.
    #[error("Einsum generator {field}={value} is unsatisfiable: {reason}")]
    Unsatisfiable {
        /// Rejected configuration field.
        field: &'static str,
        /// Rejected value.
        value: usize,
        /// Actionable reason.
        reason: &'static str,
    },
    /// Resource ceilings prevented enough legal cases from being generated.
    #[error(
        "Einsum generator produced {generated} of {requested} requested cases before exhausting bounded retries; raise the element/work ceilings or reduce the requested corpus"
    )]
    Exhausted {
        /// Successfully generated records.
        generated: usize,
        /// Requested records.
        requested: usize,
    },
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn range_inclusive(&mut self, low: usize, high: usize) -> usize {
        debug_assert!(low <= high);
        let span = (high as u128) - (low as u128) + 1;
        let offset = (u128::from(self.next()) % span) as usize;
        low + offset
    }
}
