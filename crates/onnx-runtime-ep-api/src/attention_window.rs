use onnx_runtime_ir::Node;

use crate::{EpError, Result};

/// Inclusive, independent bounds relative to the absolute query position.
#[derive(Clone, Copy, Debug, Default)]
pub struct AttentionWindow {
    pub left: Option<i64>,
    pub right: Option<i64>,
}

impl AttentionWindow {
    pub fn from_node(node: &Node, opset: u64) -> Result<Self> {
        let bound = |name| match node.attr(name) {
            None => Ok(None),
            Some(_) if opset < 25 => Err(EpError::KernelFailed(format!(
                "Attention: {name} requires opset 25 or later; remove it from this opset-{opset} node"
            ))),
            Some(attr) => match attr.as_int() {
                Some(-1) => Ok(None),
                Some(value) if value >= 0 => Ok(Some(value)),
                _ => Err(EpError::KernelFailed(format!(
                    "Attention: invalid {name}={attr:?}; set it to the integer -1 (unbounded) or a nonnegative integer"
                ))),
            },
        };
        Ok(Self {
            left: bound("left_window_size")?,
            right: bound("right_window_size")?,
        })
    }

    pub fn is_active(self) -> bool {
        self.left.is_some() || self.right.is_some()
    }

    /// Half-open intersection with the valid key interval; an empty row has
    /// equal endpoints. A negative query position is meaningful for external KV.
    pub fn key_range(self, position: i64, valid_keys: usize, causal: bool) -> (usize, usize) {
        let clip = |index: i64| (index.max(0) as u64).min(valid_keys as u64) as usize;
        let start = self
            .left
            .map_or(0, |left| clip(position.saturating_sub(left)));
        let right = if causal { Some(0) } else { self.right };
        let end = right.map_or(valid_keys, |right| {
            clip(position.saturating_add(right).saturating_add(1))
        });
        (start.min(end), end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, NodeId};

    #[test]
    fn attention25_validates_window_attrs_at_schema_boundary() {
        let mut node = Node::new(NodeId(0), "Attention", vec![], vec![]);
        assert!(!AttentionWindow::from_node(&node, 24).unwrap().is_active());
        node.attributes
            .insert("left_window_size".into(), Attribute::Int(-1));
        assert!(
            AttentionWindow::from_node(&node, 24)
                .unwrap_err()
                .to_string()
                .contains("requires opset 25")
        );
        assert!(!AttentionWindow::from_node(&node, 25).unwrap().is_active());
        for invalid in [Attribute::Int(-2), Attribute::Float(1.0)] {
            node.attributes.insert("right_window_size".into(), invalid);
            assert!(
                AttentionWindow::from_node(&node, 25)
                    .unwrap_err()
                    .to_string()
                    .contains("invalid right_window_size")
            );
        }
    }
}
