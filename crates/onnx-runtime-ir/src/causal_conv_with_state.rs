//! Shared dilation and history geometry for tensor-state CausalConvWithState.

use crate::Node;

#[derive(Clone, Copy, Debug)]
pub struct Dilation(i64);

impl Dilation {
    pub fn parse(node: &Node, opset: u64) -> Result<Self, String> {
        let Some(attr) = node.attr("dilation") else {
            return Ok(Self(1));
        };
        if !node.is_default_domain() || opset < 29 {
            return Err(format!(
                "dilation requires the ai.onnx CausalConvWithState schema at opset 29 or later; \
                 remove the attribute or use that schema (got domain {:?}, opset {opset})",
                node.domain
            ));
        }
        match attr.as_int() {
            Some(value) if value > 0 => Ok(Self(value)),
            _ => Err(format!("dilation must be an integer >= 1, got {attr:?}")),
        }
    }

    pub fn value(self) -> i64 {
        self.0
    }

    pub fn history(self, kernel: i64) -> Result<i64, String> {
        if kernel < 1 {
            return Err(format!("weight kernel width K must be >= 1, got {kernel}"));
        }
        (kernel - 1)
            .checked_mul(self.0)
            .filter(|&history| history < i64::MAX)
            .ok_or_else(|| {
                format!(
                    "effective kernel size (K - 1) * dilation + 1 exceeds int64 \
                     (K={kernel}, dilation={}); reduce the kernel width or dilation",
                    self.0
                )
            })
    }

    pub fn history_usize(self, kernel: usize) -> Result<usize, String> {
        let kernel = i64::try_from(kernel)
            .map_err(|_| format!("weight kernel width K={kernel} exceeds int64"))?;
        let history = self.history(kernel)?;
        usize::try_from(history).map_err(|_| {
            format!("history width {history} does not fit the host index size; reduce dilation")
        })
    }
}
