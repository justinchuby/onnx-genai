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

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(name: &str, head_size: u64) -> KvTensorSpec {
        // [batch, num_kv_heads, context, head_size] — the conventional decoder
        // KV shape, with the head size varying per tensor.
        KvTensorSpec {
            name: name.to_owned(),
            dtype: KvStorageType::Float16,
            shape: vec![
                KvDimension::PerSequenceBatch,
                KvDimension::Fixed(16),
                KvDimension::Context,
                KvDimension::Fixed(head_size),
            ],
        }
    }

    /// MLA (DeepSeek-V2/V3) carries the decoupled-RoPE dim on the key only, so
    /// key and value head sizes differ: 192 = qk_nope(128) + qk_rope(64) against
    /// a value of 128. Sizing must sum the two contributions independently — an
    /// `num_kv_heads * head_size * 2` shortcut is wrong by 64 elements per head
    /// per token here, and a single scalar `decoder.head_size` cannot express it
    /// at all (#1012).
    #[test]
    fn asymmetric_key_and_value_head_sizes_are_sized_independently() {
        let tensors = [kv("past.0.key", 192), kv("past.0.value", 128)];
        let bytes = kv_cache_bytes_for_tensors(&tensors, 1).expect("asymmetric KV sizes");
        // 16 heads * (192 + 128) * 2 bytes = 10,240 per token.
        assert_eq!(bytes, 16 * (192 + 128) * 2);

        // And it must scale linearly with context, not quadratically or per-axis.
        let at_seven = kv_cache_bytes_for_tensors(&tensors, 7).expect("asymmetric KV sizes");
        assert_eq!(at_seven, bytes * 7);
    }

    /// The symmetric case must be unchanged by the above: it is the same sum
    /// with equal terms, not a separate code path.
    #[test]
    fn symmetric_head_sizes_still_sum_to_twice_one_side() {
        let tensors = [kv("past.0.key", 128), kv("past.0.value", 128)];
        let bytes = kv_cache_bytes_for_tensors(&tensors, 1).expect("symmetric KV sizes");
        assert_eq!(bytes, 16 * 128 * 2 * 2);
    }

    /// Two context axes on one tensor mean the runtime cannot tell which grows,
    /// and a reservation would be a guess. It must refuse rather than pick one.
    #[test]
    fn ambiguous_context_axes_are_refused_not_guessed() {
        let spec = KvTensorSpec {
            name: "past.0.key".to_owned(),
            dtype: KvStorageType::Float16,
            shape: vec![KvDimension::Context, KvDimension::Context],
        };
        assert!(kv_cache_bytes_for_tensors(&[spec], 4).is_err());
    }
}
