//! Canonical workflow execution-capability admission.

use onnx_genai_metadata::{
    ComponentImplementation, InferenceMetadata, RuntimeInputRole, SemanticInputRole,
    SpeculativeAcceptedPath, SpeculativeContract, SpeculativeProposalExecution, WorkflowStep,
    capabilities, derived_capabilities,
};

use crate::engine::PackageCapabilityError;

type InvocationBindings<'a> = (
    &'a std::collections::BTreeMap<String, String>,
    &'a std::collections::BTreeMap<String, String>,
);

/// The one typed answer to whether this runtime may execute a loaded workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WorkflowExecutionAdmission {
    Admitted,
    CandidateTreeUnavailable {
        version: String,
        reason: String,
    },
    DFlashUnavailable {
        version: String,
        capability: &'static str,
    },
}

impl WorkflowExecutionAdmission {
    pub(crate) fn from_metadata(metadata: &InferenceMetadata) -> Self {
        let admission = Self::from_speculative(
            metadata.speculative.as_ref(),
            metadata
                .pipeline
                .as_ref()
                .map(|pipeline| &pipeline.workflow),
        );
        if matches!(admission, Self::DFlashUnavailable { .. }) {
            debug_assert!(
                derived_capabilities(metadata).contains(capabilities::DFLASH_FLAT_BLOCK),
                "a validated DFlash declaration must derive its execution capability"
            );
        }
        admission
    }

    pub(crate) fn from_speculative(
        speculative: Option<&SpeculativeContract>,
        workflow: Option<&onnx_genai_metadata::WorkflowSpec>,
    ) -> Self {
        let Some(contract) = speculative else {
            return Self::Admitted;
        };
        match &contract.proposal_execution {
            SpeculativeProposalExecution::CandidateTree { .. } => {
                let reason = candidate_tree_unavailable_reason(contract, workflow);
                match reason {
                    Some(reason) => Self::CandidateTreeUnavailable {
                        version: contract.version.clone(),
                        reason,
                    },
                    None => Self::Admitted,
                }
            }
            SpeculativeProposalExecution::DflashFlatBlock { version, .. } => {
                Self::DFlashUnavailable {
                    version: version.clone(),
                    capability: capabilities::DFLASH_FLAT_BLOCK,
                }
            }
            _ => Self::Admitted,
        }
    }

    pub(crate) fn require_supported(&self) -> Result<(), PackageCapabilityError> {
        match self {
            Self::Admitted => Ok(()),
            Self::CandidateTreeUnavailable { version, reason } => {
                Err(PackageCapabilityError::CandidateTreeExecutionUnavailable {
                    version: version.clone(),
                    reason: reason.clone(),
                })
            }
            Self::DFlashUnavailable {
                version,
                capability,
            } => Err(PackageCapabilityError::DFlashExecutionUnavailable {
                version: version.clone(),
                capability: (*capability).to_string(),
            }),
        }
    }
}

fn candidate_tree_unavailable_reason(
    contract: &SpeculativeContract,
    workflow: Option<&onnx_genai_metadata::WorkflowSpec>,
) -> Option<String> {
    if contract.version != "1" {
        return Some(format!(
            "contract version '{}' is unsupported; this runtime implements exactly version '1'",
            contract.version
        ));
    }
    if !matches!(
        contract.verification.accepted_path,
        SpeculativeAcceptedPath::Runtime { .. }
    ) {
        return Some(
            "component-owned accepted paths are unsupported; version-1 execution requires the \
                     declared runtime accepted-prefix binding"
                .to_string(),
        );
    }
    let Some(workflow) = workflow else {
        return Some("the package has no pipeline.workflow execution graph".to_string());
    };
    if !workflow.effects.is_empty() {
        return Some(
            "effectful candidate-tree regions are unsupported until every external effect joins \
                     the accepted-path transaction"
                .to_string(),
        );
    }
    if !workflow
        .outputs
        .contains_key(onnx_genai_metadata::decoder_workflow::TOKENS_OUTPUT)
    {
        return Some(format!(
            "the workflow does not declare the '{}' output required for commit-only token \
                     publication",
            onnx_genai_metadata::decoder_workflow::TOKENS_OUTPUT
        ));
    }
    if let Some(reason) = candidate_tree_bypassed_step(
        &workflow.steps,
        "pipeline.workflow.steps",
        &contract.proposer,
        &contract.target,
    ) {
        return Some(reason);
    }
    if let Some(output) = workflow
        .outputs
        .keys()
        .find(|output| output.as_str() != onnx_genai_metadata::decoder_workflow::TOKENS_OUTPUT)
    {
        return Some(format!(
            "declared output '{output}' has no candidate-tree output-publication participant; \
             the current driver stages only the declared token stream, so admitting it would \
             skip its S4 family/site"
        ));
    }
    for component in [&contract.proposer, &contract.target] {
        let Some(declaration) = workflow.components.get(component) else {
            return Some(format!("component '{component}' is undeclared"));
        };
        if !matches!(
            declaration.implementation,
            ComponentImplementation::Onnx { .. }
        ) {
            return Some(format!(
                "component '{component}' is not an ONNX artifact; binding/adapter candidate-tree \
                         components have no production execution authority"
            ));
        }
        if invocation_count(&workflow.steps, component) != 1 {
            return Some(format!(
                "component '{component}' must have exactly one workflow invocation whose typed \
                         bindings define the candidate-tree ABI"
            ));
        }
    }
    for (role, ports, component) in [
        (
            "context_tokens",
            &contract.port_bindings,
            contract.proposer.as_str(),
        ),
        (
            "context_tokens",
            &contract.target_port_bindings,
            contract.target.as_str(),
        ),
        (
            "candidate_tokens",
            &contract.target_port_bindings,
            contract.target.as_str(),
        ),
        (
            "ancestor_mask",
            &contract.target_port_bindings,
            contract.target.as_str(),
        ),
        (
            "position_ids",
            &contract.target_port_bindings,
            contract.target.as_str(),
        ),
        (
            "accepted_tokens",
            &contract.target_port_bindings,
            contract.target.as_str(),
        ),
    ] {
        let Some(port) = ports.get(role) else {
            return Some(format!(
                "candidate-tree role '{role}' is absent for component '{component}'"
            ));
        };
        let Some(declaration) = workflow.components.get(component) else {
            continue;
        };
        if !declaration.ports.inputs.contains_key(port) {
            return Some(format!(
                "candidate-tree role '{role}' selects absent input port '{port}' on component \
                         '{component}'"
            ));
        }
    }
    for (component, ports) in [
        (&contract.proposer, &contract.port_bindings),
        (&contract.target, &contract.target_port_bindings),
    ] {
        let port = ports
            .get("context_tokens")
            .expect("checked candidate-tree context role above");
        let Some((inputs, _)) = component_invocation(&workflow.steps, component) else {
            continue;
        };
        let Some(binding) = inputs.get(port) else {
            return Some(format!(
                "candidate-tree context port '{component}::{port}' has no workflow binding"
            ));
        };
        if !matches!(
            workflow.inputs.get(binding).map(|input| &input.role),
            Some(SemanticInputRole::Runtime { role, .. })
                if *role == RuntimeInputRole::PromptTokens
        ) {
            return Some(format!(
                "candidate-tree context port '{component}::{port}' must bind the workflow's \
                         runtime prompt_tokens input, not '{binding}'"
            ));
        }
    }
    let Some((proposer_inputs, proposer_outputs)) =
        component_invocation(&workflow.steps, &contract.proposer)
    else {
        return Some("the proposer invocation disappeared during admission".to_string());
    };
    let Some((target_inputs, target_outputs)) =
        component_invocation(&workflow.steps, &contract.target)
    else {
        return Some("the target invocation disappeared during admission".to_string());
    };
    for (role, port) in &contract.port_bindings {
        if !proposer_inputs.contains_key(port) {
            return Some(format!(
                "candidate-tree proposer role '{role}' selects input '{port}', but its workflow \
                 invocation does not bind that port"
            ));
        }
    }
    for (role, port) in &contract.target_port_bindings {
        if !target_inputs.contains_key(port) {
            return Some(format!(
                "candidate-tree target role '{role}' selects input '{port}', but its workflow \
                 invocation does not bind that port"
            ));
        }
    }
    let SpeculativeProposalExecution::CandidateTree {
        candidate_tokens,
        topology,
    } = &contract.proposal_execution
    else {
        unreachable!("candidate-tree admission is called only for candidate-tree contracts");
    };
    let topology_output = match topology {
        onnx_genai_metadata::CandidateTreeTopology::ParentIndices { output }
        | onnx_genai_metadata::CandidateTreeTopology::AncestorMask { output } => output,
    };
    for output in [candidate_tokens, topology_output] {
        if !proposer_outputs.contains_key(output) {
            return Some(format!(
                "candidate-tree proposer output '{output}' is not bound by its workflow invocation"
            ));
        }
    }
    if !target_outputs.contains_key(&contract.verification.target_output.output) {
        return Some(format!(
            "candidate-tree target verifier output '{}' is not bound by its workflow invocation",
            contract.verification.target_output.output
        ));
    }
    if let Some(probabilities) = &contract.verification.probabilities {
        if !proposer_outputs.contains_key(&probabilities.proposal.output) {
            return Some(format!(
                "candidate-tree proposal probability output '{}' is not bound by its workflow \
                 invocation",
                probabilities.proposal.output
            ));
        }
        if !target_outputs.contains_key(&probabilities.target.output) {
            return Some(format!(
                "candidate-tree target probability output '{}' is not bound by its workflow \
                 invocation",
                probabilities.target.output
            ));
        }
    }
    None
}

/// Find authored steps the specialized tree drive would otherwise skip.
///
/// A sequence is only structural grouping, so recursively inspecting it admits
/// the same two-component template wherever it is grouped. A loop, branch,
/// unrelated invocation, or emit has ordering/effect/output semantics the
/// driver does not interpret; accepting one would make the metadata advisory.
fn candidate_tree_bypassed_step(
    steps: &[WorkflowStep],
    path: &str,
    proposer: &str,
    target: &str,
) -> Option<String> {
    for (index, step) in steps.iter().enumerate() {
        let step_path = format!("{path}[{index}]");
        match step {
            WorkflowStep::Invoke { component, .. }
                if component == proposer || component == target => {}
            WorkflowStep::Invoke { component, .. } => {
                return Some(format!(
                    "candidate-tree driver cannot execute unrelated component '{component}' at \
                     {step_path}; it would be skipped between proposer/target invocations"
                ));
            }
            WorkflowStep::Sequence { steps } => {
                if let Some(reason) = candidate_tree_bypassed_step(
                    steps,
                    &format!("{step_path}.steps"),
                    proposer,
                    target,
                ) {
                    return Some(reason);
                }
            }
            WorkflowStep::Loop { setup, .. } => {
                let detail = if setup.is_empty() {
                    "its condition/body ordering"
                } else {
                    "its non-empty setup, condition/body ordering"
                };
                return Some(format!(
                    "candidate-tree driver cannot faithfully execute loop at {step_path}: \
                     {detail} would be skipped; use a flat proposer/target invocation template \
                     until the generic workflow transaction hosts the candidate-tree seam"
                ));
            }
            WorkflowStep::Branch { .. } => {
                return Some(format!(
                    "candidate-tree driver cannot faithfully execute branch at {step_path}; its \
                     predicate/join would be skipped before accepted-prefix commit"
                ));
            }
            WorkflowStep::Emit { output, .. } => {
                return Some(format!(
                    "candidate-tree driver cannot publish emit at {step_path} into output \
                     '{output}'; its S4 family/site would be skipped"
                ));
            }
        }
    }
    None
}

fn invocation_count(steps: &[WorkflowStep], component: &str) -> usize {
    steps
        .iter()
        .map(|step| match step {
            WorkflowStep::Invoke {
                component: invoked, ..
            } => usize::from(invoked == component),
            WorkflowStep::Sequence { steps } => invocation_count(steps, component),
            WorkflowStep::Loop { setup, steps, .. } => {
                invocation_count(setup, component) + invocation_count(steps, component)
            }
            WorkflowStep::Branch { cases, default, .. } => {
                cases
                    .values()
                    .map(|step| invocation_count(std::slice::from_ref(step), component))
                    .sum::<usize>()
                    + default
                        .as_deref()
                        .map(|step| invocation_count(std::slice::from_ref(step), component))
                        .unwrap_or_default()
            }
            WorkflowStep::Emit { .. } => 0,
        })
        .sum()
}

fn component_invocation<'a>(
    steps: &'a [WorkflowStep],
    component: &str,
) -> Option<InvocationBindings<'a>> {
    for step in steps {
        match step {
            WorkflowStep::Invoke {
                component: invoked,
                inputs,
                outputs,
            } if invoked == component => return Some((inputs, outputs)),
            WorkflowStep::Sequence { steps } => {
                if let Some(found) = component_invocation(steps, component) {
                    return Some(found);
                }
            }
            WorkflowStep::Loop { setup, steps, .. } => {
                if let Some(found) = component_invocation(setup, component)
                    .or_else(|| component_invocation(steps, component))
                {
                    return Some(found);
                }
            }
            WorkflowStep::Branch { cases, default, .. } => {
                for case in cases.values() {
                    if let Some(found) = component_invocation(std::slice::from_ref(case), component)
                    {
                        return Some(found);
                    }
                }
                if let Some(default) = default
                    && let Some(found) =
                        component_invocation(std::slice::from_ref(default), component)
                {
                    return Some(found);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_and_canonical_chained_mtp_remain_admitted() {
        assert_eq!(
            WorkflowExecutionAdmission::from_metadata(&InferenceMetadata::default()),
            WorkflowExecutionAdmission::Admitted
        );
        let mtp = onnx_genai_metadata::parse_metadata(
                include_str!(
                    "../../../../examples/inference_metadata/catalogue/22-qwen3-chained-speculative-decoding.yaml"
                ),
                Some("yaml"),
            )
            .expect("canonical chained MTP fixture parses");
        assert_eq!(
            WorkflowExecutionAdmission::from_metadata(&mtp),
            WorkflowExecutionAdmission::Admitted
        );
    }

    #[test]
    fn exact_dflash_contract_resolves_to_one_capability_refusal() {
        let dflash = onnx_genai_metadata::parse_metadata(
            include_str!("../../tests/fixtures/dflash-admission/inference_metadata.yaml"),
            Some("yaml"),
        )
        .expect("DFlash fixture parses");
        for version in ["1", "2"] {
            let mut versioned = dflash.clone();
            let SpeculativeProposalExecution::DflashFlatBlock {
                version: declared, ..
            } = &mut versioned
                .speculative
                .as_mut()
                .expect("fixture declares speculation")
                .proposal_execution
            else {
                panic!("fixture declares DFlash")
            };
            *declared = version.to_string();
            assert_eq!(
                WorkflowExecutionAdmission::from_metadata(&versioned),
                WorkflowExecutionAdmission::DFlashUnavailable {
                    version: version.to_string(),
                    capability: capabilities::DFLASH_FLAT_BLOCK,
                }
            );
        }
    }

    #[test]
    fn canonical_candidate_tree_fixture_is_admitted() {
        let tree = onnx_genai_metadata::parse_metadata(
            include_str!("../../tests/fixtures/unsupported-candidate-tree/inference_metadata.yaml"),
            Some("yaml"),
        )
        .expect("candidate tree parses");
        assert!(matches!(
            WorkflowExecutionAdmission::from_metadata(&tree),
            WorkflowExecutionAdmission::CandidateTreeUnavailable { reason, .. }
                if reason.contains("tokens")
        ));
    }
}
