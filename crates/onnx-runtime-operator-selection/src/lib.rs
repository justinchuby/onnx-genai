//! Shared catalog for model-driven native execution-provider builds.

/// A stable Cargo capability group.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct OperatorGroup {
    pub feature: &'static str,
    pub fully_gated: bool,
}

/// One operator family known to the minimal-build tooling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OperatorCatalogEntry {
    pub domain: &'static str,
    pub op_type: &'static str,
    pub since_version: u64,
    pub group: OperatorGroup,
}

pub const OPS_CORE: OperatorGroup = OperatorGroup {
    feature: "ops-core",
    fully_gated: false,
};
pub const OPS_REDUCTION: OperatorGroup = OperatorGroup {
    feature: "ops-reduction",
    fully_gated: false,
};
pub const OPS_TRANSFORMER: OperatorGroup = OperatorGroup {
    feature: "ops-transformer",
    fully_gated: false,
};
pub const OPS_CNN: OperatorGroup = OperatorGroup {
    feature: "ops-cnn",
    fully_gated: true,
};
pub const OPS_QUANTIZED: OperatorGroup = OperatorGroup {
    feature: "ops-quantized",
    fully_gated: false,
};

/// Representative CPU catalog. `fully_gated` identifies groups whose modules and
/// registry factories are compile-time excluded when the feature is absent.
pub const CPU_OPERATOR_CATALOG: &[OperatorCatalogEntry] = &[
    entry("", "Add", 1, OPS_CORE),
    entry("", "Cast", 1, OPS_CORE),
    entry("", "Constant", 1, OPS_CORE),
    entry("", "Gather", 1, OPS_CORE),
    entry("", "Identity", 1, OPS_CORE),
    entry("", "MatMul", 1, OPS_CORE),
    entry("", "Reshape", 1, OPS_CORE),
    entry("", "Shape", 1, OPS_CORE),
    entry("", "Transpose", 1, OPS_CORE),
    entry("", "ReduceMean", 1, OPS_REDUCTION),
    entry("", "Softmax", 1, OPS_REDUCTION),
    entry("", "Attention", 23, OPS_TRANSFORMER),
    entry("com.microsoft", "Attention", 1, OPS_TRANSFORMER),
    entry("com.microsoft", "GroupQueryAttention", 1, OPS_TRANSFORMER),
    entry("", "Gemm", 1, OPS_TRANSFORMER),
    entry("", "LayerNormalization", 1, OPS_TRANSFORMER),
    entry("com.microsoft", "MatMulNBits", 1, OPS_QUANTIZED),
    entry("", "QLinearMatMul", 10, OPS_QUANTIZED),
    entry("", "AffineGrid", 20, OPS_CNN),
    entry("", "AveragePool", 1, OPS_CNN),
    entry("", "BatchNormalization", 7, OPS_CNN),
    entry("", "CenterCropPad", 18, OPS_CNN),
    entry("", "Col2Im", 18, OPS_CNN),
    entry("", "Conv", 1, OPS_CNN),
    entry("", "ConvTranspose", 1, OPS_CNN),
    entry("", "GlobalAveragePool", 1, OPS_CNN),
    entry("", "GlobalMaxPool", 1, OPS_CNN),
    entry("", "GlobalLpPool", 2, OPS_CNN),
    entry("", "GroupNormalization", 18, OPS_CNN),
    entry("", "GridSample", 16, OPS_CNN),
    entry("", "InstanceNormalization", 6, OPS_CNN),
    entry("", "LpPool", 18, OPS_CNN),
    entry("", "MaxPool", 1, OPS_CNN),
    entry("", "PRelu", 16, OPS_CNN),
    entry("", "Resize", 10, OPS_CNN),
    entry("", "SpaceToDepth", 13, OPS_CNN),
];

const fn entry(
    domain: &'static str,
    op_type: &'static str,
    since_version: u64,
    group: OperatorGroup,
) -> OperatorCatalogEntry {
    OperatorCatalogEntry {
        domain,
        op_type,
        since_version,
        group,
    }
}

/// Normalize the standard ONNX domain to its manifest spelling.
pub fn normalize_domain(domain: &str) -> &str {
    if domain.is_empty() || domain == "ai.onnx" {
        "ai.onnx"
    } else {
        domain
    }
}

/// Find the catalog entry compatible with an operator requirement.
pub fn find_cpu_operator(
    domain: &str,
    op_type: &str,
    opset: u64,
) -> Option<&'static OperatorCatalogEntry> {
    let domain = normalize_domain(domain);
    CPU_OPERATOR_CATALOG
        .iter()
        .filter(|entry| normalize_domain(entry.domain) == domain)
        .filter(|entry| entry.op_type == op_type && entry.since_version <= opset)
        .max_by_key(|entry| entry.since_version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_domain_aliases_resolve_identically() {
        assert_eq!(
            find_cpu_operator("", "Conv", 21),
            find_cpu_operator("ai.onnx", "Conv", 21)
        );
    }

    #[test]
    fn rejects_opsets_before_the_registered_version() {
        assert!(find_cpu_operator("ai.onnx", "Attention", 22).is_none());
        assert!(find_cpu_operator("ai.onnx", "Attention", 23).is_some());
    }
}
