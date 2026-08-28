//! Swappable persistent backing stores for cold KV payloads.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::connector::{KvCacheKey, KvLayerPayload, KvPayload, KvPayloadDtype};

static NEXT_SCRATCH_ID: AtomicU64 = AtomicU64::new(0);
const MAGIC: &[u8; 8] = b"OGKV0001";

/// Storage for payloads whose pages have left host memory.
///
/// Implementations must return the exact bytes passed to [`Self::write`]. The
/// connector serializes access through its interior mutex, so a store need not
/// add its own synchronization.
pub trait KvBackingStore: Send {
    /// Persist a payload under `key`, replacing a previous value for that key.
    fn write(&mut self, key: &KvCacheKey, payload: &KvPayload) -> Result<(), String>;
    /// Load the payload previously persisted under `key`.
    fn read(&mut self, key: &KvCacheKey) -> Result<KvPayload, String>;
    /// Delete a persisted payload, if present.
    fn remove(&mut self, key: &KvCacheKey);
    /// Number of successful writes, useful for observability and tests.
    fn spill_count(&self) -> u64;
}

/// Test-friendly backing store which keeps cold payloads in memory.
#[derive(Default)]
pub struct InMemoryKvBackingStore {
    payloads: HashMap<KvCacheKey, KvPayload>,
    spills: u64,
}

impl KvBackingStore for InMemoryKvBackingStore {
    fn write(&mut self, key: &KvCacheKey, payload: &KvPayload) -> Result<(), String> {
        self.payloads.insert(key.clone(), payload.clone());
        self.spills += 1;
        Ok(())
    }

    fn read(&mut self, key: &KvCacheKey) -> Result<KvPayload, String> {
        self.payloads
            .get(key)
            .cloned()
            .ok_or_else(|| "cold payload not found".to_owned())
    }

    fn remove(&mut self, key: &KvCacheKey) {
        self.payloads.remove(key);
    }

    fn spill_count(&self) -> u64 {
        self.spills
    }
}

/// Disk-backed scratch store. Each instance owns a unique child directory and
/// removes it on drop, leaving a caller-provided parent directory untouched.
pub struct DiskKvBackingStore {
    scratch_dir: PathBuf,
    files: HashMap<KvCacheKey, PathBuf>,
    next_file: u64,
    spills: u64,
}

impl DiskKvBackingStore {
    /// Create a store below `parent`, creating the parent if needed.
    pub fn new(parent: impl AsRef<Path>) -> Result<Self, String> {
        let parent = parent.as_ref();
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        let unique = NEXT_SCRATCH_ID.fetch_add(1, Ordering::Relaxed);
        let scratch_dir = parent.join(format!(
            "onnx-genai-kv-{}-{}-{unique}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|error| error.to_string())?
                .as_nanos()
        ));
        fs::create_dir(&scratch_dir).map_err(|error| error.to_string())?;
        Ok(Self {
            scratch_dir,
            files: HashMap::new(),
            next_file: 0,
            spills: 0,
        })
    }

    /// Owned scratch directory, exposed for diagnostics and cleanup tests.
    pub fn scratch_dir(&self) -> &Path {
        &self.scratch_dir
    }

    fn path_for(&mut self, key: &KvCacheKey) -> PathBuf {
        if let Some(path) = self.files.get(key) {
            return path.clone();
        }
        let path = self.scratch_dir.join(format!("{}.kv", self.next_file));
        self.next_file += 1;
        self.files.insert(key.clone(), path.clone());
        path
    }
}

impl Drop for DiskKvBackingStore {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.scratch_dir);
    }
}

impl KvBackingStore for DiskKvBackingStore {
    fn write(&mut self, key: &KvCacheKey, payload: &KvPayload) -> Result<(), String> {
        if !payload.is_well_formed() {
            return Err("refusing to spill malformed KV payload".to_owned());
        }
        let path = self.path_for(key);
        let mut file = File::create(path).map_err(|error| error.to_string())?;
        file.write_all(MAGIC).map_err(|error| error.to_string())?;
        for value in [
            payload.num_tokens,
            payload.num_layers,
            payload.num_kv_heads,
            payload.head_dim,
        ] {
            file.write_all(&(value as u64).to_le_bytes())
                .map_err(|error| error.to_string())?;
        }
        file.write_all(&[match payload.dtype {
            KvPayloadDtype::F32 => 0,
        }])
        .map_err(|error| error.to_string())?;
        for layer in &payload.layers {
            for values in [&layer.key, &layer.value] {
                for value in values {
                    file.write_all(&value.to_bits().to_le_bytes())
                        .map_err(|error| error.to_string())?;
                }
            }
        }
        file.sync_all().map_err(|error| error.to_string())?;
        self.spills += 1;
        Ok(())
    }

    fn read(&mut self, key: &KvCacheKey) -> Result<KvPayload, String> {
        let path = self
            .files
            .get(key)
            .ok_or_else(|| "cold payload not found".to_owned())?;
        let mut file = File::open(path).map_err(|error| error.to_string())?;
        let mut magic = [0; 8];
        file.read_exact(&mut magic)
            .map_err(|error| error.to_string())?;
        if &magic != MAGIC {
            return Err("invalid KV spill file header".to_owned());
        }
        let mut read_usize = || -> Result<usize, String> {
            let mut bytes = [0; 8];
            file.read_exact(&mut bytes)
                .map_err(|error| error.to_string())?;
            usize::try_from(u64::from_le_bytes(bytes)).map_err(|error| error.to_string())
        };
        let num_tokens = read_usize()?;
        let num_layers = read_usize()?;
        let num_kv_heads = read_usize()?;
        let head_dim = read_usize()?;
        let mut dtype = [0; 1];
        file.read_exact(&mut dtype)
            .map_err(|error| error.to_string())?;
        if dtype != [0] {
            return Err("unsupported KV spill dtype".to_owned());
        }
        let values_per_tensor = num_kv_heads
            .checked_mul(num_tokens)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| "KV spill dimensions overflow".to_owned())?;
        let mut read_values = || -> Result<Vec<f32>, String> {
            let mut values = Vec::with_capacity(values_per_tensor);
            for _ in 0..values_per_tensor {
                let mut bytes = [0; 4];
                file.read_exact(&mut bytes)
                    .map_err(|error| error.to_string())?;
                values.push(f32::from_bits(u32::from_le_bytes(bytes)));
            }
            Ok(values)
        };
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(KvLayerPayload {
                key: read_values()?,
                value: read_values()?,
            });
        }
        let mut extra = [0; 1];
        if file.read(&mut extra).map_err(|error| error.to_string())? != 0 {
            return Err("KV spill file has trailing bytes".to_owned());
        }
        Ok(KvPayload {
            num_tokens,
            num_layers,
            num_kv_heads,
            head_dim,
            dtype: KvPayloadDtype::F32,
            layers,
        })
    }

    fn remove(&mut self, key: &KvCacheKey) {
        if let Some(path) = self.files.remove(key) {
            let _ = fs::remove_file(path);
        }
    }

    fn spill_count(&self) -> u64 {
        self.spills
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(chunk_hash: u64) -> KvCacheKey {
        KvCacheKey {
            isolation: crate::KvCacheIsolation::Session(1),
            model_id: "backing-store-test".to_owned(),
            layer_range: 0..2,
            chunk_hash,
            chunk_index: 0,
            num_tokens: 2,
        }
    }

    fn payload() -> KvPayload {
        KvPayload {
            num_tokens: 2,
            num_layers: 2,
            num_kv_heads: 1,
            head_dim: 2,
            dtype: KvPayloadDtype::F32,
            layers: vec![
                KvLayerPayload {
                    key: vec![1.0, -0.0, f32::from_bits(0x7fc0_0001), 4.0],
                    value: vec![5.0, 6.0, 7.0, 8.0],
                },
                KvLayerPayload {
                    key: vec![9.0, 10.0, 11.0, 12.0],
                    value: vec![13.0, 14.0, 15.0, 16.0],
                },
            ],
        }
    }

    fn unique_parent(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "onnx-genai-kv-{name}-{}-{}",
            std::process::id(),
            NEXT_SCRATCH_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn disk_store_round_trips_every_f32_bit_and_cleans_scratch_directory() {
        let parent = unique_parent("round-trip");
        let key = key(1);
        let expected = payload();
        let scratch_dir;
        {
            let mut store = DiskKvBackingStore::new(&parent).unwrap();
            scratch_dir = store.scratch_dir().to_owned();
            store.write(&key, &expected).unwrap();
            assert!(scratch_dir.join("0.kv").is_file());
            assert_eq!(store.spill_count(), 1);
            let actual = store.read(&key).unwrap();
            for (actual_layer, expected_layer) in actual.layers.iter().zip(&expected.layers) {
                assert_eq!(
                    actual_layer
                        .key
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected_layer
                        .key
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    actual_layer
                        .value
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected_layer
                        .value
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
        assert!(!scratch_dir.exists(), "drop must remove KV scratch files");
        fs::remove_dir_all(parent).unwrap();
    }
}
