//! Fixed-capacity, device-resident CSA state allocation.
//!
//! The Phase-B cache contract reserves stable addresses during kernel/runner
//! construction. Claim-time callers use [`CsaBufferLayout::from_claim`] only; it
//! performs the same checked static sizing without touching CUDA memory.

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{EpError, Result};
use onnx_runtime_ir::{Node, Shape};

use crate::runtime::CudaRuntime;

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
            .get(0)
            .and_then(|shape| shape.first())
            .and_then(|dim| dim.as_static())
        else {
            return Ok(None);
        };
        let sequence = shapes
            .get(0)
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

/// Stable-address buffers reserved once for a CSA runner. They are intentionally
/// not read by B0: graph-threaded `past_* → present_*` remains authoritative.
pub(crate) struct CsaDeviceBufferManager {
    runtime: Arc<CudaRuntime>,
    pub(crate) layout: CsaBufferLayout,
    buffers: Vec<CUdeviceptr>,
    /// B6 pooled scratch (index transform / scores / selection / attention
    /// scores). Reserved once at runner init with stable addresses so the
    /// device-only capture path never allocates per call.
    workspaces: Vec<CUdeviceptr>,
}

impl CsaDeviceBufferManager {
    pub(crate) fn reserve(
        runtime: Arc<CudaRuntime>,
        layout: CsaBufferLayout,
        workspace_bytes: &[usize],
    ) -> Result<Self> {
        let sizes = [
            layout.attention_r4_bytes,
            layout.attention_r4_carry_bytes,
            layout.attention_r128_bytes,
            layout.attention_r128_carry_bytes,
            layout.index_r4_bytes,
            layout.index_r4_carry_bytes,
            layout.dense_ring_bytes,
        ];
        let mut buffers = Vec::with_capacity(sizes.len());
        let mut workspaces = Vec::with_capacity(workspace_bytes.len());
        let rollback = |buffers: &mut Vec<CUdeviceptr>, workspaces: &mut Vec<CUdeviceptr>| {
            for ptr in workspaces.drain(..).rev() {
                // SAFETY: each pointer was allocated by this runtime and has not escaped.
                let _ = unsafe { runtime.free_raw(ptr) };
            }
            for ptr in buffers.drain(..).rev() {
                // SAFETY: each pointer was allocated by this runtime and has not escaped.
                let _ = unsafe { runtime.free_raw(ptr) };
            }
        };
        for size in sizes {
            match runtime.alloc_raw(size) {
                Ok(ptr) => buffers.push(ptr),
                Err(error) => {
                    rollback(&mut buffers, &mut workspaces);
                    return Err(error);
                }
            }
        }
        for &size in workspace_bytes {
            match runtime.alloc_raw(size.max(1)) {
                Ok(ptr) => workspaces.push(ptr),
                Err(error) => {
                    rollback(&mut buffers, &mut workspaces);
                    return Err(error);
                }
            }
        }
        Ok(Self {
            runtime,
            layout,
            buffers,
            workspaces,
        })
    }

    /// Stable address of pooled workspace `index` (reserved in `reserve`).
    pub(crate) fn workspace(&self, index: usize) -> CUdeviceptr {
        self.workspaces[index]
    }
}

impl Drop for CsaDeviceBufferManager {
    fn drop(&mut self) {
        for ptr in self.workspaces.drain(..).rev() {
            // SAFETY: this manager exclusively owns every pointer it reserved.
            let _ = unsafe { self.runtime.free_raw(ptr) };
        }
        for ptr in self.buffers.drain(..).rev() {
            // SAFETY: this manager exclusively owns every pointer it reserved.
            let _ = unsafe { self.runtime.free_raw(ptr) };
        }
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
        node.domain = "pkg.nxrt".into();
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
