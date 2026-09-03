//! Independent ONNX `Einsum` conformance corpus, parser, and numeric oracle.
//!
//! This crate deliberately has no dependency on `onnx-runtime-ir` or an
//! execution provider. Production plans, contraction trees, costs, and index
//! programs therefore cannot leak into the expected tensors. CPU and CUDA
//! tests consume the same [`CaseRecord`] and [`Evaluation`] values through the
//! route-neutral [`RouteProbe`] and [`BackendAdapter`] interfaces.

mod adapter;
mod case;
mod equation;
mod generator;
mod oracle;
mod route;
mod schema;

pub use adapter::{
    AdapterProbe, AdapterStatus, PythonEngine, PythonReferenceAdapter, ReferenceAdapterError,
};
pub use case::{
    CPU_WORKING_SET_BYTES, CaseLimits, CaseRecord, CaseValidationError, ConformanceDType,
    CorpusSnapshot, DeclaredDType, GPU_CASE_BYTES, MalformedCase, MalformedKind, UNIT_TENSOR_BYTES,
    ValueProfile, ValueSpec, corpus_digest, validate_case_signature,
};
pub use equation::{
    EquationAnalysis, EquationError, OperandLayout, analyze_equation, infer_output_shape,
};
pub use generator::{
    DEFAULT_GENERATOR, GENERATOR_METADATA_BYTES, GeneratorConfig, GeneratorError, default_corpus,
    generated_cases, malformed_cases, named_cases,
};
pub use oracle::{
    CanonicalTensor, ComparisonFailure, ComparisonMode, ComparisonReport, Evaluation, OracleError,
    compare, evaluate, materialize_inputs,
};
pub use route::{
    BackendAdapter, BackendKind, BackendObservation, CaptureExpectation, ExecutionRequest,
    ForcedRoute, PlannerQuality, RouteAssertionError, RouteProbe, WorkspaceClass,
    verify_observation,
};
pub use schema::{SchemaAuthority, SchemaAuthorityError};
