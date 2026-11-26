//! Shared ONNX DFT mode and real/complex geometry.
//! See <https://onnx.ai/onnx/operators/onnx__DFT.html> (schemas 17 and 20).

use crate::Node;

/// The transform's semantic mode, independent of backend or FFT algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DftMode {
    Forward,
    Inverse,
    RealForward,
    RealInverse,
}

impl DftMode {
    pub fn from_flags(inverse: bool, onesided: bool) -> Self {
        match (inverse, onesided) {
            (false, false) => Self::Forward,
            (true, false) => Self::Inverse,
            (false, true) => Self::RealForward,
            (true, true) => Self::RealInverse,
        }
    }

    pub fn parse(node: &Node) -> Result<Self, String> {
        let flag = |name| match node.attr(name) {
            None => Ok(false),
            Some(value) => match value.as_int() {
                Some(0) => Ok(false),
                Some(1) => Ok(true),
                _ => Err(format!(
                    "DFT: attribute '{name}' must be integer 0 or 1, got {value:?}"
                )),
            },
        };
        Ok(Self::from_flags(flag("inverse")?, flag("onesided")?))
    }

    pub fn inverse(self) -> bool {
        matches!(self, Self::Inverse | Self::RealInverse)
    }

    pub fn validate_components(self, components: usize) -> Result<(), String> {
        let valid = match self {
            Self::RealForward => components == 1,
            Self::RealInverse => components == 2,
            _ => matches!(components, 1 | 2),
        };
        if valid {
            Ok(())
        } else {
            let expected = match self {
                Self::RealForward => {
                    "onesided=1 with inverse=0 is valid only for real input (last dimension 1)"
                }
                Self::RealInverse => {
                    "onesided=1 with inverse=1 requires a complex half-spectrum (last dimension 2)"
                }
                _ => "full transforms require last dimension 1 (real) or 2 (complex)",
            };
            Err(format!(
                "DFT: mode {self:?} cannot consume last dimension {components}; {expected}"
            ))
        }
    }

    pub fn output_components(self) -> usize {
        if self == Self::RealInverse { 1 } else { 2 }
    }

    pub fn default_length(self, input_length: usize) -> Result<usize, String> {
        let length = if self == Self::RealInverse {
            input_length.checked_sub(1).and_then(|n| n.checked_mul(2))
                .ok_or_else(|| format!("DFT: one-sided input length {input_length} cannot produce an addressable inverse length; provide a positive dft_length"))?
        } else {
            input_length
        };
        Self::validate_length(length)?;
        Ok(length)
    }

    pub fn validate_length(length: usize) -> Result<(), String> {
        if length == 0 {
            Err("DFT: transform length must be positive; provide a positive dft_length or a nonempty signal axis (at least two spectrum bins for an implicit real inverse)".into())
        } else {
            Ok(())
        }
    }

    pub fn output_length(self, length: usize) -> usize {
        if self == Self::RealForward {
            length / 2 + 1
        } else {
            length
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attributes_reject_invalid_flags_with_the_argument_name() {
        for name in ["inverse", "onesided"] {
            for value in [
                crate::Attribute::Int(2),
                crate::Attribute::Int(-1),
                crate::Attribute::String("true".into()),
            ] {
                let mut node = Node::new(crate::NodeId(0), "DFT", vec![], vec![]);
                node.attributes.insert(name.into(), value);
                let error = DftMode::parse(&node).unwrap_err();
                assert!(
                    error.contains(name) && error.contains("integer 0 or 1"),
                    "{error}"
                );
            }
        }
    }

    #[test]
    fn modes_bind_input_kind_output_width_and_default_length() {
        for (inverse, onesided, expected, components, output, length) in [
            (false, false, DftMode::Forward, 1, 2, 4),
            (true, false, DftMode::Inverse, 2, 2, 4),
            (false, true, DftMode::RealForward, 1, 2, 4),
            (true, true, DftMode::RealInverse, 2, 1, 6),
        ] {
            let mode = DftMode::from_flags(inverse, onesided);
            assert_eq!(mode, expected);
            mode.validate_components(components).unwrap();
            assert_eq!(mode.output_components(), output);
            assert_eq!(mode.default_length(4).unwrap(), length);
            assert_eq!(
                mode.output_length(5),
                if mode == DftMode::RealForward { 3 } else { 5 }
            );
        }
        assert!(DftMode::RealForward.validate_components(2).is_err());
        assert!(DftMode::RealInverse.validate_components(1).is_err());
        for n in [0, 1, usize::MAX] {
            assert!(DftMode::RealInverse.default_length(n).is_err());
        }
    }
}
