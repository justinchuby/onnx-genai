//! Canonical workflow execution-capability admission.

use onnx_genai_metadata::{
    InferenceMetadata, SpeculativeContract, SpeculativeProposalExecution, capabilities,
    derived_capabilities,
};

use crate::engine::PackageCapabilityError;

/// The one typed answer to whether this runtime may execute a loaded workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WorkflowExecutionAdmission {
    Admitted,
    CandidateTreeUnavailable {
        version: String,
    },
    DFlashUnavailable {
        version: String,
        capability: &'static str,
    },
}

impl WorkflowExecutionAdmission {
    pub(crate) fn from_metadata(metadata: &InferenceMetadata) -> Self {
        let admission = Self::from_speculative(metadata.speculative.as_ref());
        if matches!(admission, Self::DFlashUnavailable { .. }) {
            debug_assert!(
                derived_capabilities(metadata).contains(capabilities::DFLASH_FLAT_BLOCK),
                "a validated DFlash declaration must derive its execution capability"
            );
        }
        admission
    }

    pub(crate) fn from_speculative(speculative: Option<&SpeculativeContract>) -> Self {
        let Some(contract) = speculative else {
            return Self::Admitted;
        };
        match &contract.proposal_execution {
            SpeculativeProposalExecution::CandidateTree { .. } => Self::CandidateTreeUnavailable {
                version: contract.version.clone(),
            },
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
            Self::CandidateTreeUnavailable { version } => {
                Err(PackageCapabilityError::CandidateTreeExecutionUnavailable {
                    version: version.clone(),
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
}
