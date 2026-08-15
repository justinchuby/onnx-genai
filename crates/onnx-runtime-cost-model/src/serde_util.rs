//! Serde helpers.

/// Serialize a `BTreeMap` with a **non-string** key as a JSON sequence of
/// `[key, value]` pairs.
///
/// JSON object keys must be strings, but the cost model keys its maps by
/// structured [`crate::DeviceKey`] / [`crate::TransferKey`] values. Rather than
/// flatten those to a lossy string, the maps are stored as ordered sequences of
/// entries, which round-trips exactly and stays human-readable.
pub mod map_as_seq {
    use std::collections::BTreeMap;

    use serde::de::Deserialize;
    use serde::ser::Serialize;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S, K, V>(map: &BTreeMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        K: Serialize + Ord,
        V: Serialize,
    {
        let entries: Vec<(&K, &V)> = map.iter().collect();
        entries.serialize(serializer)
    }

    pub fn deserialize<'de, D, K, V>(deserializer: D) -> Result<BTreeMap<K, V>, D::Error>
    where
        D: Deserializer<'de>,
        K: Deserialize<'de> + Ord,
        V: Deserialize<'de>,
    {
        let entries: Vec<(K, V)> = Vec::deserialize(deserializer)?;
        Ok(entries.into_iter().collect())
    }
}
