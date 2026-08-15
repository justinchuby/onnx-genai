#![cfg_attr(
    all(not(feature = "native-backend"), not(test)),
    expect(
        dead_code,
        reason = "exact native KV sizing is consumed only by the native backend"
    )
)]

use anyhow::{Context, bail};
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum KvDimension {
    Fixed(u64),
    PerSequenceBatch,
    Context,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvStorageType {
    Float32,
    Float16,
    BFloat16,
}

impl KvStorageType {
    fn bytes(self, elements: u64) -> anyhow::Result<u64> {
        let width = match self {
            Self::Float32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
        };
        elements
            .checked_mul(width)
            .context("KV tensor storage byte size overflowed")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KvTensorSpec {
    pub(crate) name: String,
    pub(crate) dtype: KvStorageType,
    pub(crate) shape: Vec<KvDimension>,
}

pub(crate) fn kv_cache_bytes_for_tensors(
    tensors: &[KvTensorSpec],
    context: u64,
) -> anyhow::Result<u64> {
    let mut total = 0_u64;
    for tensor in tensors {
        let context_axes = tensor
            .shape
            .iter()
            .filter(|dim| matches!(dim, KvDimension::Context))
            .count();
        if context_axes > 1 {
            bail!(
                "cannot size KV tensor '{}': {:?} has {context_axes} symbolic non-batch axes, so \
                 which one grows with context is ambiguous and a reservation would be a guess",
                tensor.name,
                tensor.shape
            );
        }

        let elements = tensor.shape.iter().try_fold(1_u64, |elements, dim| {
            let extent = match dim {
                KvDimension::Fixed(value) => *value,
                KvDimension::PerSequenceBatch => 1,
                KvDimension::Context => context,
            };
            elements
                .checked_mul(extent)
                .with_context(|| format!("KV tensor '{}' element count overflowed", tensor.name))
        })?;
        let bytes = tensor.dtype.bytes(elements)?;
        total = total
            .checked_add(bytes)
            .with_context(|| format!("total KV byte size overflowed at '{}'", tensor.name))?;
    }
    Ok(total)
}
