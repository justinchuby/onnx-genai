//! Pure fixed-capacity CSA state layout facts.
//!
//! Graph-threaded state uses native session owners; computational scratch uses
//! caller workspace. These facts never allocate memory or own a quota lease.

use onnx_runtime_ep_api::{EpError, Result};
use onnx_runtime_ir::{Node, Shape};

const ATTN_WIDTH: usize = 583;
const INDEX_WIDTH: usize = 68;
const DENSE_WIDTH: usize = 583;
const MAX_SEQUENCE_LEN: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CsaBufferLayout {
    pub batch: usize,
    pub max_seq_len: usize,
    pub window: usize,
    pub attention_r4_bytes: usize,
    pub attention_r4_carry_bytes: usize,
    pub attention_r128_bytes: usize,
    pub attention_r128_carry_bytes: usize,
    pub index_r4_bytes: usize,
    pub index_r4_carry_bytes: usize,
    pub dense_ring_bytes: usize,
}

impl CsaBufferLayout {
    pub(crate) fn from_claim(node: &Node, shapes: &[Shape], ratio: usize) -> Result<Option<Self>> {
        let Some(batch) = shapes
            .first()
            .and_then(|shape| shape.first())
            .and_then(|dim| dim.as_static())
        else {
            return Ok(None);
        };
        let sequence = shapes
            .first()
            .and_then(|shape| shape.get(1))
            .and_then(|dim| dim.as_static());
        let past_records = shapes
            .get(6)
            .and_then(|shape| shape.get(1))
            .and_then(|dim| dim.as_static());
        Self::from_values(node, batch, sequence, past_records, ratio).map(Some)
    }

    pub(crate) fn from_runner(
        node: &Node,
        input_shapes: &[Vec<usize>],
        ratio: usize,
    ) -> Result<Self> {
        let query = input_shapes
            .first()
            .ok_or_else(|| error("missing query shape"))?;
        let batch = *query
            .first()
            .ok_or_else(|| error("query batch axis is missing"))?;
        let sequence = *query
            .get(1)
            .ok_or_else(|| error("query sequence axis is missing"))?;
        let past_records = input_shapes.get(6).and_then(|shape| shape.get(1)).copied();
        Self::from_values(node, batch, Some(sequence), past_records, ratio)
    }

    fn from_values(
        node: &Node,
        batch: usize,
        sequence: Option<usize>,
        past_records: Option<usize>,
        ratio: usize,
    ) -> Result<Self> {
        let metadata_max =
            static_attr(node, "max_seq_len").or_else(|| static_attr(node, "max_sequence_length"));
        let inferred = match (past_records, sequence) {
            (Some(records), Some(sequence)) => records
                .checked_mul(ratio)
                .and_then(|v| v.checked_add(sequence)),
            _ => None,
        };
        let max_seq_len = metadata_max.or(inferred).ok_or_else(|| {
            error("max_seq_len metadata or static cache/query capacity is required")
        })?;
        if batch == 0 || max_seq_len == 0 || max_seq_len > MAX_SEQUENCE_LEN {
            return Err(error(format!(
                "batch={batch} and max_seq_len={max_seq_len} must be within supported fixed bounds"
            )));
        }
        let window = static_attr(node, "sliding_window")
            .or_else(|| static_attr(node, "window_size"))
            .unwrap_or(max_seq_len);
        if window == 0 || window > max_seq_len {
            return Err(error(format!(
                "dense window {window} must be in 1..={max_seq_len}"
            )));
        }
        let records4 = ceil_div(max_seq_len, 4)?;
        let records128 = ceil_div(max_seq_len, 128)?;
        Ok(Self {
            batch,
            max_seq_len,
            window,
            attention_r4_bytes: bytes(&[batch, records4, ATTN_WIDTH], 1)?,
            attention_r4_carry_bytes: bytes(&[batch, 8, 2, 1024], 4)?,
            attention_r128_bytes: bytes(&[batch, records128, ATTN_WIDTH], 1)?,
            attention_r128_carry_bytes: bytes(&[batch, 128, 2, 512], 4)?,
            index_r4_bytes: bytes(&[batch, records4, INDEX_WIDTH], 1)?,
            index_r4_carry_bytes: bytes(&[batch, 8, 2, 256], 4)?,
            dense_ring_bytes: bytes(&[batch, window, DENSE_WIDTH], 1)?,
        })
    }
}

fn static_attr(node: &Node, name: &str) -> Option<usize> {
    node.attr(name)
        .and_then(|attribute| attribute.as_int())
        .and_then(|value| usize::try_from(value).ok())
}
fn ceil_div(value: usize, divisor: usize) -> Result<usize> {
    value
        .checked_add(divisor - 1)
        .map(|v| v / divisor)
        .ok_or_else(|| error("CSA buffer capacity overflow"))
}
fn bytes(shape: &[usize], element_bytes: usize) -> Result<usize> {
    shape
        .iter()
        .try_fold(element_bytes, |n, &d| n.checked_mul(d))
        .ok_or_else(|| error("CSA buffer byte size overflow"))
}
fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!(
        "CompressedSparseAttention fixed-capacity state: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, Graph, Node, NodeId, static_shape};

    #[test]
    fn layout_uses_static_metadata_without_allocating() {
        let mut graph = Graph::new();
        let query = graph.create_named_value(
            "q",
            onnx_runtime_ir::DataType::Float32,
            static_shape([2, 1, 1, 512]),
        );
        let cache = graph.create_named_value(
            "cache",
            onnx_runtime_ir::DataType::Uint8,
            static_shape([2, 0, 583]),
        );
        let mut node = Node::new(
            NodeId(0),
            "CompressedSparseAttention",
            vec![Some(query), Some(cache)],
            vec![],
        );
        node.domain = "pkg.nxrt._private".into();
        node.attributes
            .insert("max_seq_len".into(), Attribute::Int(1024));
        node.attributes
            .insert("sliding_window".into(), Attribute::Int(128));
        let layout = CsaBufferLayout::from_claim(
            &node,
            &[static_shape([2, 1, 1, 512]), static_shape([2, 0, 583])],
            4,
        )
        .unwrap()
        .unwrap();
        assert_eq!(layout.attention_r4_bytes, 2 * 256 * 583);
        assert_eq!(layout.index_r4_bytes, 2 * 256 * 68);
        assert_eq!(layout.dense_ring_bytes, 2 * 128 * 583);
    }
}
