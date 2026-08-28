//! Canonical workflow execution-capability admission.

use onnx_genai_metadata::{
    DFlashStructure, InferenceMetadata, SpeculativeContract, SpeculativeProposalExecution,
    capabilities, derived_capabilities,
};

use crate::engine::{EngineDecodeBackend, PackageCapabilityError};

/// The one typed answer to whether this runtime may execute a loaded workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WorkflowExecutionAdmission {
    Admitted,
    DFlashUnavailable {
        version: String,
        capability: &'static str,
    },
}

impl WorkflowExecutionAdmission {
    pub(crate) fn from_metadata(
        metadata: &InferenceMetadata,
        backend: EngineDecodeBackend,
    ) -> Self {
        let admission = Self::from_speculative(metadata.speculative.as_ref(), backend);
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
        backend: EngineDecodeBackend,
    ) -> Self {
        let Some(SpeculativeProposalExecution::DflashFlatBlock {
            version, structure, ..
        }) = speculative.map(|contract| &contract.proposal_execution)
        else {
            return Self::Admitted;
        };
        if version == "1"
            && matches!(structure.as_ref(), DFlashStructure::Base)
            && matches!(
                backend,
                EngineDecodeBackend::Auto | EngineDecodeBackend::Ort
            )
        {
            return Self::Admitted;
        }
        Self::DFlashUnavailable {
            version: version.clone(),
            capability: capabilities::DFLASH_FLAT_BLOCK,
        }
    }

    pub(crate) fn require_supported(&self) -> Result<(), PackageCapabilityError> {
        match self {
            Self::Admitted => Ok(()),
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
            WorkflowExecutionAdmission::from_metadata(
                &InferenceMetadata::default(),
                EngineDecodeBackend::Ort,
            ),
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
            WorkflowExecutionAdmission::from_metadata(&mtp, EngineDecodeBackend::Ort),
            WorkflowExecutionAdmission::Admitted
        );
    }

    #[test]
    fn only_the_implemented_dflash_v1_ort_pair_is_admitted() {
        let dflash = onnx_genai_metadata::parse_metadata(
            include_str!("../../tests/fixtures/dflash-admission/inference_metadata.yaml"),
            Some("yaml"),
        )
        .expect("DFlash fixture parses");
        assert_eq!(
            WorkflowExecutionAdmission::from_metadata(&dflash, EngineDecodeBackend::Ort),
            WorkflowExecutionAdmission::Admitted
        );
        assert_eq!(
            WorkflowExecutionAdmission::from_metadata(&dflash, EngineDecodeBackend::Native),
            WorkflowExecutionAdmission::DFlashUnavailable {
                version: "1".to_string(),
                capability: capabilities::DFLASH_FLAT_BLOCK,
            }
        );

        let mut versioned = dflash;
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
        *declared = "2".to_string();
        assert_eq!(
            WorkflowExecutionAdmission::from_metadata(&versioned, EngineDecodeBackend::Ort),
            WorkflowExecutionAdmission::DFlashUnavailable {
                version: "2".to_string(),
                capability: capabilities::DFLASH_FLAT_BLOCK,
            }
        );
    }
}
