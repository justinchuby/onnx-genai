//! Activation arithmetic is independent of compressed weight storage.
use crate::Node;

pub const ATTRIBUTE: &str = "activation_arithmetic";

/// Arithmetic requested by a native packed projection, not a provider hint.
/// A provider must decline a mode it cannot implement instead of ignoring it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActivationArithmetic {
    #[default]
    Float32,
    /// Per-32 activation amax/127, nearest integer (ties away from zero),
    /// f16 stored scale, and scaled i32 block dots. This can change outputs.
    Q8_0,
}
impl ActivationArithmetic {
    pub fn from_node(node: &Node) -> Result<Self, String> {
        let arithmetic = match node.attr(ATTRIBUTE) {
            None => Self::Float32,
            Some(value) => match value.as_str() {
                Some("float32") => Self::Float32,
                Some("q8_0") => Self::Q8_0,
                _ => {
                    return Err(format!(
                        "{}: {ATTRIBUTE} must be the UTF-8 string 'float32' or 'q8_0', got {value:?}; select supported arithmetic independently of weight format",
                        node.op_type
                    ));
                }
            },
        };
        let required = arithmetic.layout_version();
        match node.attr("block_layout_version") {
            None if arithmetic == Self::Float32 => {}
            Some(crate::Attribute::Int(version)) if *version == required => {}
            value => {
                return Err(format!(
                    "{}: block_layout_version must be integer {required} for activation_arithmetic='{}', got {value:?}; export that explicit ABI tag (q8_0 requires version 2 so older runtimes refuse instead of ignoring the new arithmetic)",
                    node.op_type,
                    arithmetic.name()
                ));
            }
        }
        Ok(arithmetic)
    }

    /// Each active arithmetic has one ABI tag. The new mode must carry its tag
    /// explicitly, because version-1 factories may ignore unknown attributes.
    pub fn layout_version(self) -> i64 {
        match self {
            Self::Float32 => 1,
            Self::Q8_0 => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Q8_0 => "q8_0",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Attribute, NodeId};
    #[test]
    fn default_and_explicit_modes_are_distinct_from_weight_format() {
        let mut node = Node::new(NodeId(0), "BlockQuantizedMatMul", vec![], vec![]);
        node.attributes
            .insert("format".into(), Attribute::String(b"q8_0".to_vec()));
        assert_eq!(
            ActivationArithmetic::from_node(&node).unwrap(),
            ActivationArithmetic::Float32
        );
        for (name, expected) in [
            ("float32", ActivationArithmetic::Float32),
            ("q8_0", ActivationArithmetic::Q8_0),
        ] {
            node.attributes.insert(
                ATTRIBUTE.into(),
                Attribute::String(name.as_bytes().to_vec()),
            );
            node.attributes.insert(
                "block_layout_version".into(),
                Attribute::Int(expected.layout_version()),
            );
            assert_eq!(ActivationArithmetic::from_node(&node).unwrap(), expected);
            assert_eq!(expected.name(), name);
        }
    }
    #[test]
    fn invalid_modes_include_operation_argument_and_remedy() {
        for value in [
            Attribute::Int(1),
            Attribute::String(vec![255]),
            Attribute::String(b"auto".to_vec()),
        ] {
            let mut node = Node::new(NodeId(0), "BlockQuantizedMoE", vec![], vec![]);
            node.attributes.insert(ATTRIBUTE.into(), value);
            let error = ActivationArithmetic::from_node(&node).unwrap_err();
            assert!(
                error.contains("BlockQuantizedMoE")
                    && error.contains(ATTRIBUTE)
                    && error.contains("float32")
                    && error.contains("q8_0"),
                "{error}"
            );
        }
    }
    #[test]
    fn quantized_arithmetic_requires_the_explicit_non_downgradable_abi_tag() {
        let mut node = Node::new(NodeId(0), "BlockQuantizedMatMul", vec![], vec![]);
        node.attributes
            .insert(ATTRIBUTE.into(), Attribute::String(b"q8_0".to_vec()));
        for value in [
            None,
            Some(Attribute::Int(1)),
            Some(Attribute::Int(3)),
            Some(Attribute::String(b"2".to_vec())),
        ] {
            node.attributes.remove("block_layout_version");
            if let Some(value) = value {
                node.attributes.insert("block_layout_version".into(), value);
            }
            let error = ActivationArithmetic::from_node(&node).unwrap_err();
            assert!(
                error.contains("block_layout_version")
                    && error.contains("integer 2")
                    && error.contains("q8_0"),
                "{error}"
            );
        }
        node.attributes
            .insert("block_layout_version".into(), Attribute::Int(2));
        assert_eq!(
            ActivationArithmetic::from_node(&node).unwrap(),
            ActivationArithmetic::Q8_0
        );
        node.attributes
            .insert(ATTRIBUTE.into(), Attribute::String(b"float32".to_vec()));
        assert!(
            ActivationArithmetic::from_node(&node)
                .unwrap_err()
                .contains("integer 1")
        );
    }
}
