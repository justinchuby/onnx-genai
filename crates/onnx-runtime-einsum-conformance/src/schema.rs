use std::borrow::Cow;

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::ConformanceDType;

const AUTHORITY: &str = include_str!("../fixtures/onnx-einsum-schema-authority.txt");
const AUTHORITY_COMMIT: &str = "5732eb5de3e6b353e1a5aa49fe5d577f81bb58e0";
const AUTHORITY_SHA256: &str = "3cf77daa0d23c4e96ef350b6a82abe0be717ea85cb207b29bfe07b2f65ada113";

/// Vendored authority for the ONNX `Einsum-12` and `Einsum-28` type contracts.
///
/// The fixture is a pinned excerpt from ONNX source, not the schema exposed by
/// whichever Python wheel happens to be installed on a developer machine.
pub struct SchemaAuthority;

impl SchemaAuthority {
    /// Pinned upstream ONNX commit.
    pub const fn source_commit() -> &'static str {
        AUTHORITY_COMMIT
    }

    /// SHA-256 of the complete vendored authority fixture in canonical LF form.
    pub const fn fixture_sha256() -> &'static str {
        AUTHORITY_SHA256
    }

    /// Verify that the fixture still proves the expected schema boundary.
    pub fn verify() -> Result<(), SchemaAuthorityError> {
        verify_authority(AUTHORITY)
    }

    /// ONNX `since_version` selected by an imported opset.
    pub fn since_version(opset: u64) -> Result<u64, SchemaAuthorityError> {
        Self::verify()?;
        match opset {
            0..=11 => Err(SchemaAuthorityError::UnsupportedOpset(opset)),
            12..=27 => Ok(12),
            28.. => Ok(28),
        }
    }

    /// Whether the pinned schema admits `dtype` at `opset`.
    pub fn supports(opset: u64, dtype: ConformanceDType) -> Result<bool, SchemaAuthorityError> {
        let since = Self::since_version(opset)?;
        Ok(dtype != ConformanceDType::BFloat16 || since == 28)
    }
}

fn verify_authority(authority: &str) -> Result<(), SchemaAuthorityError> {
    let authority = canonical_line_endings(authority);
    let actual = format!("{:x}", Sha256::digest(authority.as_bytes()));
    if actual != AUTHORITY_SHA256 {
        return Err(SchemaAuthorityError::Digest {
            expected: AUTHORITY_SHA256,
            actual,
        });
    }
    for required in [
        "commit=5732eb5de3e6b353e1a5aa49fe5d577f81bb58e0",
        "Einsum,\n    12,\n    OpSchema().FillUsing(defs::math::utils::EinsumOpGenerator(OpSchema::all_numeric_types()))",
        "Einsum,\n    28,\n    OpSchema().FillUsing(defs::math::utils::EinsumOpGenerator(OpSchema::all_numeric_types_ir4()))",
        "TensorProto::BFLOAT16",
    ] {
        if !authority.contains(required) {
            return Err(SchemaAuthorityError::MissingEvidence(required));
        }
    }
    let (v28_types, v12_types) = authority
        .split_once("const std::vector<std::string>& OpSchema::all_numeric_types()")
        .ok_or(SchemaAuthorityError::MissingEvidence(
            "separate all_numeric_types_ir4 and all_numeric_types definitions",
        ))?;
    if !v28_types.contains("TensorProto::BFLOAT16") {
        return Err(SchemaAuthorityError::MissingEvidence(
            "BFLOAT16 in all_numeric_types_ir4",
        ));
    }
    if v12_types.contains("TensorProto::BFLOAT16") {
        return Err(SchemaAuthorityError::UnexpectedEvidence(
            "BFLOAT16 in legacy all_numeric_types",
        ));
    }
    Ok(())
}

fn canonical_line_endings(authority: &str) -> Cow<'_, str> {
    if authority.contains("\r\n") {
        Cow::Owned(authority.replace("\r\n", "\n"))
    } else {
        Cow::Borrowed(authority)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_crlf_checkout_preserves_schema_authority() {
        let crlf = AUTHORITY.replace('\n', "\r\n");
        verify_authority(&crlf).unwrap();
        assert_eq!(
            canonical_line_endings(AUTHORITY),
            canonical_line_endings(&crlf)
        );
    }

    #[test]
    fn noncanonical_carriage_return_still_changes_authority() {
        let changed = AUTHORITY.replacen("commit=", "commit=\r", 1);
        assert!(matches!(
            verify_authority(&changed),
            Err(SchemaAuthorityError::Digest { .. })
        ));
    }
}

/// Failure to establish the pinned ONNX schema authority.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SchemaAuthorityError {
    /// The fixture bytes changed without an intentional authority refresh.
    #[error(
        "vendored ONNX Einsum schema authority changed: expected SHA-256 {expected}, found {actual}; regenerate it from the pinned ONNX source and review the schema delta"
    )]
    Digest {
        /// Expected digest.
        expected: &'static str,
        /// Actual digest.
        actual: String,
    },
    /// Required source evidence disappeared.
    #[error("vendored ONNX Einsum schema authority no longer contains {0}")]
    MissingEvidence(&'static str),
    /// Evidence contradicts the expected boundary.
    #[error("vendored ONNX Einsum schema authority unexpectedly contains {0}")]
    UnexpectedEvidence(&'static str),
    /// Imported opset predates `Einsum`.
    #[error("ai.onnx opset {0} predates Einsum-12")]
    UnsupportedOpset(u64),
}
