//! Validate metadata against runtime capabilities.

use std::collections::{BTreeMap, BTreeSet};

use crate::schema::{InferenceMetadata, PipelineSpec, PipelineStrategy, PipelineStrategyKind};

/// Capabilities this runtime supports.
pub struct RuntimeCapabilities {
    pub supported: Vec<String>,
}

impl Default for RuntimeCapabilities {
    fn default() -> Self {
        Self {
            supported: vec![
                "kv_cache".to_string(),
                "grouped_query_attention".to_string(),
                "multi_head_attention".to_string(),
                "prefix_cache".to_string(),
                "continuous_batching".to_string(),
            ],
        }
    }
}

/// Validate that all required capabilities are supported.
pub fn validate(
    metadata: &InferenceMetadata,
    runtime: &RuntimeCapabilities,
) -> Result<(), Vec<String>> {
    let unsupported: Vec<String> = metadata
        .required_capabilities
        .iter()
        .filter(|cap| !runtime.supported.contains(cap))
        .cloned()
        .collect();

    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(unsupported)
    }
}

/// All structural problems found in a pipeline specification.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid pipeline spec: {errors:?}")]
pub struct PipelineValidationError {
    pub errors: Vec<String>,
}

/// Validate the pipeline DAG and component references.
pub fn validate_pipeline_spec(spec: &PipelineSpec) -> Result<(), PipelineValidationError> {
    let mut errors = Vec::new();

    if spec.models.is_empty() {
        errors.push("pipeline.models must contain at least one component".to_string());
    }

    for (name, component) in &spec.models {
        if name.trim().is_empty() {
            errors.push("pipeline model names must not be empty".to_string());
        }
        if name.contains('.') {
            errors.push(format!("pipeline model name must not contain '.': {name}"));
        }
        if component.filename.trim().is_empty() {
            errors.push(format!("pipeline model {name} must declare a filename"));
        }
        if component.role.trim().is_empty() {
            errors.push(format!("pipeline model {name} must declare a type"));
        }
    }

    let mut adjacency: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for name in spec.models.keys() {
        adjacency.entry(name.as_str()).or_default();
    }

    for edge in &spec.dataflow {
        match parse_endpoint(&edge.from) {
            Some((component, port)) => {
                if !spec.models.contains_key(component) {
                    errors.push(format!(
                        "dataflow edge source references unknown component: {}",
                        edge.from
                    ));
                }
                if port.is_empty() {
                    errors.push(format!(
                        "dataflow edge source has an empty port: {}",
                        edge.from
                    ));
                }
            }
            None => errors.push(format!(
                "dataflow edge source must be component.port: {}",
                edge.from
            )),
        }

        match parse_endpoint(&edge.to) {
            Some((component, port)) => {
                if !spec.models.contains_key(component) {
                    errors.push(format!(
                        "dataflow edge destination references unknown component: {}",
                        edge.to
                    ));
                }
                if port.is_empty() {
                    errors.push(format!(
                        "dataflow edge destination has an empty port: {}",
                        edge.to
                    ));
                }
            }
            None => errors.push(format!(
                "dataflow edge destination must be component.port: {}",
                edge.to
            )),
        }

        if let (Some((from, _)), Some((to, _))) =
            (parse_endpoint(&edge.from), parse_endpoint(&edge.to))
            && spec.models.contains_key(from)
            && spec.models.contains_key(to)
            // A self-edge (`A.x -> A.y`) is a loop-carried temporal dependency
            // (e.g. a diffusion denoiser fed its previous step's output), not a
            // same-step DAG cycle; it is excluded from the acyclic check.
            && from != to
        {
            adjacency.entry(from).or_default().insert(to);
        }
    }

    for phase_component in spec.phases.keys() {
        if !spec.models.contains_key(phase_component) {
            errors.push(format!(
                "phase references unknown component: {phase_component}"
            ));
        }
    }

    validate_strategy(&spec.strategy, &spec.models, "strategy", &mut errors);
    validate_acyclic(&adjacency, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(PipelineValidationError { errors })
    }
}

fn parse_endpoint(endpoint: &str) -> Option<(&str, &str)> {
    let (component, port) = endpoint.split_once('.')?;
    if component.is_empty() {
        return None;
    }
    Some((component, port))
}

fn validate_strategy(
    strategy: &PipelineStrategy,
    models: &BTreeMap<String, crate::schema::PipelineComponentSpec>,
    path: &str,
    errors: &mut Vec<String>,
) {
    match strategy.kind {
        PipelineStrategyKind::Autoregressive => {
            require_strategy_model(strategy.decoder.as_deref(), "decoder", path, models, errors);
        }
        PipelineStrategyKind::SinglePass => {
            require_strategy_model(strategy.model.as_deref(), "model", path, models, errors);
        }
        PipelineStrategyKind::Iterative => {
            require_strategy_model(
                strategy.denoiser.as_deref(),
                "denoiser",
                path,
                models,
                errors,
            );
        }
        PipelineStrategyKind::Composite => {
            if strategy.stages.is_empty() {
                errors.push(format!("{path}.stages must contain at least one stage"));
            }
            for stage in &strategy.stages {
                if stage.name.trim().is_empty() {
                    errors.push(format!("{path}.stages contains a stage with an empty name"));
                }
                validate_strategy(
                    &stage.strategy,
                    models,
                    &format!("{path}.stages[{}]", stage.name),
                    errors,
                );
            }
        }
        PipelineStrategyKind::Other(_) => {}
    }
}

fn require_strategy_model(
    value: Option<&str>,
    field: &str,
    path: &str,
    models: &BTreeMap<String, crate::schema::PipelineComponentSpec>,
    errors: &mut Vec<String>,
) {
    match value {
        Some(name) if models.contains_key(name) => {}
        Some(name) => errors.push(format!(
            "{path}.{field} references unknown component: {name}"
        )),
        None => errors.push(format!("{path}.{field} is required")),
    }
}

fn validate_acyclic(adjacency: &BTreeMap<&str, BTreeSet<&str>>, errors: &mut Vec<String>) {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mark {
        Visiting,
        Done,
    }

    fn visit<'a>(
        node: &'a str,
        adjacency: &BTreeMap<&'a str, BTreeSet<&'a str>>,
        marks: &mut BTreeMap<&'a str, Mark>,
        stack: &mut Vec<&'a str>,
        errors: &mut Vec<String>,
    ) {
        match marks.get(node) {
            Some(Mark::Done) => return,
            Some(Mark::Visiting) => {
                stack.push(node);
                errors.push(format!(
                    "pipeline dataflow contains a cycle: {}",
                    stack.join(" -> ")
                ));
                stack.pop();
                return;
            }
            None => {}
        }

        marks.insert(node, Mark::Visiting);
        stack.push(node);
        if let Some(next_nodes) = adjacency.get(node) {
            for next in next_nodes {
                visit(next, adjacency, marks, stack, errors);
            }
        }
        stack.pop();
        marks.insert(node, Mark::Done);
    }

    let mut marks = BTreeMap::new();
    let mut stack = Vec::new();
    for node in adjacency.keys() {
        visit(node, adjacency, &mut marks, &mut stack, errors);
    }
}

/// Error type for metadata operations.
#[derive(Debug, thiserror::Error)]
pub enum MetadataError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Unsupported capabilities: {0:?}")]
    Unsupported(Vec<String>),
}

// Re-export at crate level
pub use MetadataError as Error;
