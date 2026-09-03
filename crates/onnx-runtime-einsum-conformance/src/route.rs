use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CPU_WORKING_SET_BYTES, CanonicalTensor, CaseRecord, ComparisonFailure, ComparisonMode,
    ComparisonReport, Evaluation, GPU_CASE_BYTES, compare,
};

/// Backend family targeted by a future forced-route test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// Native CPU execution provider.
    Cpu,
    /// Native CUDA execution provider.
    Cuda,
}

/// Route a backend test must actually force.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForcedRoute {
    /// Universal direct index program.
    GenericNative,
    /// Exact subset-DP contraction tree.
    OptimizedDp,
    /// Bounded deterministic heuristic contraction tree.
    OptimizedHeuristic,
    /// Native matrix multiplication lowering.
    MatMul,
    /// CUDA cuBLAS/cuBLASLt lowering.
    CudaCublas,
}

/// Planner quality that must be observed, not inferred from route intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlannerQuality {
    /// Exact subset dynamic programming.
    ExactSubsetDp,
    /// Deterministic bounded heuristic.
    DeterministicHeuristic,
}

/// Capture assertion for a forced CUDA route.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureExpectation {
    /// CPU or otherwise capture-independent.
    NotApplicable,
    /// The route must remain CUDA-graph capturable.
    MustCapture,
}

/// Named workspace ceiling carried into backend assertions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceClass {
    /// CPU aggregate working set is capped at 32 MiB.
    Cpu32MiB,
    /// GPU case footprint/workspace is capped at 64 MiB.
    Gpu64MiB,
}

impl WorkspaceClass {
    /// Maximum bytes admitted by this class.
    pub const fn max_bytes(self) -> usize {
        match self {
            Self::Cpu32MiB => CPU_WORKING_SET_BYTES,
            Self::Gpu64MiB => GPU_CASE_BYTES,
        }
    }
}

/// One future backend assertion attached to a corpus case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteProbe {
    /// Backend under test.
    pub backend: BackendKind,
    /// Route the adapter must force and report.
    pub route: ForcedRoute,
    /// Planner-quality assertion, when applicable.
    pub planner_quality: Option<PlannerQuality>,
    /// Numeric comparison mode.
    pub comparison: ComparisonMode,
    /// Workspace/case ceiling.
    pub workspace: WorkspaceClass,
    /// CUDA graph capture requirement.
    pub capture: CaptureExpectation,
}

/// Inputs to a real backend adapter.
pub struct ExecutionRequest<'a> {
    /// Corpus record.
    pub case: &'a CaseRecord,
    /// Deterministic materialized inputs.
    pub inputs: &'a [CanonicalTensor],
    /// Forced route assertion.
    pub probe: &'a RouteProbe,
}

/// Measurements returned by a real CPU or CUDA adapter.
///
/// No production adapter is implemented in this crate. Later backend work must
/// report the route actually taken and the measured workspace/capture facts.
#[derive(Clone, Debug)]
pub struct BackendObservation {
    /// Backend that produced this observation.
    pub backend: BackendKind,
    /// Route actually taken.
    pub route: ForcedRoute,
    /// Planner quality actually selected.
    pub planner_quality: Option<PlannerQuality>,
    /// Measured peak workspace bytes attributable to the case.
    pub workspace_bytes: usize,
    /// Graph captures attributed to the case.
    pub captures: usize,
    /// Graph replays attributed to the case.
    pub replays: usize,
    /// Capture fallbacks attributed to the case.
    pub capture_fallbacks: usize,
    /// Backend output.
    pub output: CanonicalTensor,
}

/// Interface implemented by real backend tests.
pub trait BackendAdapter {
    /// Adapter-specific execution error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Execute one case while genuinely forcing `request.probe.route`.
    fn execute(&self, request: ExecutionRequest<'_>) -> Result<BackendObservation, Self::Error>;
}

/// Verify route identity, planner quality, workspace, capture, and numerics.
pub fn verify_observation(
    case: &CaseRecord,
    expected: &Evaluation,
    probe: &RouteProbe,
    observed: &BackendObservation,
) -> Result<ComparisonReport, RouteAssertionError> {
    if observed.backend != probe.backend {
        return Err(RouteAssertionError::Backend {
            expected: probe.backend,
            observed: observed.backend,
        });
    }
    if observed.route != probe.route {
        return Err(RouteAssertionError::Route {
            expected: probe.route,
            observed: observed.route,
        });
    }
    if observed.planner_quality != probe.planner_quality {
        return Err(RouteAssertionError::PlannerQuality {
            expected: probe.planner_quality,
            observed: observed.planner_quality,
        });
    }
    let workspace_limit = probe.workspace.max_bytes();
    let total_bytes = case
        .native_io_bytes()
        .map_err(RouteAssertionError::Case)?
        .checked_add(observed.workspace_bytes)
        .ok_or(RouteAssertionError::WorkspaceOverflow)?;
    if total_bytes > workspace_limit {
        return Err(RouteAssertionError::Workspace {
            observed: total_bytes,
            limit: workspace_limit,
        });
    }
    if probe.capture == CaptureExpectation::MustCapture
        && (observed.captures == 0 || observed.replays == 0 || observed.capture_fallbacks != 0)
    {
        return Err(RouteAssertionError::Capture {
            captures: observed.captures,
            replays: observed.replays,
            fallbacks: observed.capture_fallbacks,
        });
    }
    compare(case, expected, &observed.output, probe.comparison).map_err(Into::into)
}

/// Failure of a real backend observation to satisfy its corpus contract.
#[derive(Debug, Error)]
pub enum RouteAssertionError {
    /// Adapter reported the wrong backend.
    #[error("Einsum backend mismatch: expected {expected:?}, observed {observed:?}")]
    Backend {
        /// Required backend.
        expected: BackendKind,
        /// Observed backend.
        observed: BackendKind,
    },
    /// Backend did not take the forced route.
    #[error("forced Einsum route {expected:?} was not taken; backend reported {observed:?}")]
    Route {
        /// Required route.
        expected: ForcedRoute,
        /// Observed route.
        observed: ForcedRoute,
    },
    /// Planner quality drifted.
    #[error(
        "Einsum planner quality mismatch: expected {expected:?}, backend reported {observed:?}"
    )]
    PlannerQuality {
        /// Required quality.
        expected: Option<PlannerQuality>,
        /// Observed quality.
        observed: Option<PlannerQuality>,
    },
    /// Workspace exceeded the case contract.
    #[error(
        "Einsum backend I/O plus workspace used {observed} bytes, exceeding the {limit}-byte cap"
    )]
    Workspace {
        /// Observed peak.
        observed: usize,
        /// Allowed peak.
        limit: usize,
    },
    /// Required CUDA capture failed.
    #[error(
        "Einsum CUDA route requires captures>0, replays>0, fallbacks=0; observed captures={captures}, replays={replays}, fallbacks={fallbacks}"
    )]
    Capture {
        /// Observed captures.
        captures: usize,
        /// Observed replays.
        replays: usize,
        /// Observed fallbacks.
        fallbacks: usize,
    },
    /// Numeric mismatch.
    #[error(transparent)]
    Comparison(#[from] ComparisonFailure),
    /// Case accounting could not be resolved.
    #[error(transparent)]
    Case(#[from] crate::CaseValidationError),
    /// I/O plus workspace byte accounting overflowed.
    #[error("Einsum backend I/O plus workspace byte accounting overflowed")]
    WorkspaceOverflow,
}
