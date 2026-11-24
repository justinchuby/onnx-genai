//! Versioned, bounded host-state codec. Parsing never mutates an executor.
use super::*;
use onnx_runtime_memory_api::Tier;
use onnx_runtime_memory_governor::{MemoryGovernor, MemoryLease};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"NXRTST01";
const HEADER: u64 = 56;
const DIGEST: u64 = 32;

/// Admission bounds for one state file, independent of allocator headroom.
#[derive(Debug, Clone, Copy)]
pub struct NativeStateFileLimits {
    pub max_bytes: u64,
    pub max_manifest_bytes: u64,
    pub max_tokens: usize,
}
impl Default for NativeStateFileLimits {
    fn default() -> Self {
        Self {
            max_bytes: 4 * 1024 * 1024 * 1024,
            max_manifest_bytes: 16 * 1024 * 1024,
            max_tokens: 1024 * 1024,
        }
    }
}
impl NativeStateFileLimits {
    fn validate(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.max_bytes > HEADER + DIGEST && self.max_manifest_bytes > 0 && self.max_tokens > 0,
            "invalid Native state file limits {self:?}; supply positive token/manifest bounds and a byte limit larger than the file header"
        );
        anyhow::ensure!(
            cfg!(target_endian = "little"),
            "Native state file v1 requires a little-endian host; use a supported host or implement explicit tensor byte conversion"
        );
        Ok(())
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    tokens: Vec<TokenId>,
    len: usize,
    past: Vec<TensorSpec>,
    cpu: Option<Vec<CpuSpec>>,
    hidden: Option<usize>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorSpec {
    name: String,
    dtype: u8,
    shape: Vec<usize>,
    bytes: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CpuSpec {
    pair: (String, String),
    dtype: u8,
    shape: Vec<usize>,
}

fn reserve(memory: &dyn MemoryGovernor, bytes: u64) -> anyhow::Result<MemoryLease> {
    memory.reserve(Tier::Host, bytes, crate::engine::Holder::NativeSessionSnapshot.role(),
        crate::engine::Holder::NativeSessionSnapshot.id())
        .context("reserve bounded Native state file buffers; increase the host memory budget or use a smaller state file")
}
fn buffer(bytes: u64) -> anyhow::Result<Vec<u8>> {
    let count = usize::try_from(bytes)
        .context("Native state buffer exceeds host address space; restore a smaller file")?;
    let mut result = Vec::new();
    result.try_reserve_exact(count).context(
        "allocate admitted Native state buffer; free host memory or restore a smaller file",
    )?;
    result.resize(count, 0);
    Ok(result)
}
fn add(a: u64, b: u64) -> anyhow::Result<u64> {
    a.checked_add(b)
        .context("Native state file size overflow; use a smaller cache")
}
fn times(a: u64, b: u64) -> anyhow::Result<u64> {
    a.checked_mul(b)
        .context("Native state file size overflow; use a smaller cache")
}
fn dtype(raw: u8) -> anyhow::Result<DataType> {
    DataType::from_onnx(i32::from(raw))
        .filter(|d| d.bit_size() > 0)
        .with_context(|| {
            format!(
                "invalid Native state tensor dtype {raw}; restore a file written by this runtime"
            )
        })
}
fn bytes(kind: DataType, shape: &[usize]) -> anyhow::Result<u64> {
    onnx_runtime_ir::checked_expected_bytes(kind, shape)
        .map(u64::try_from)
        .transpose()?
        .context("Native state tensor dimensions overflow; restore an intact state file")
}

impl NativePastSnapshot {
    pub(crate) fn write_file(
        &self,
        tokens: &[TokenId],
        identity: [u8; 32],
        writer: &mut impl Write,
        limits: NativeStateFileLimits,
        memory: &dyn MemoryGovernor,
    ) -> anyhow::Result<u64> {
        limits.validate()?;
        anyhow::ensure!(
            tokens.len() <= limits.max_tokens && self.len <= tokens.len(),
            "Native state has {} logical and {} materialized tokens, exceeding file token limit {}; increase the limit or save a shorter state",
            tokens.len(),
            self.len,
            limits.max_tokens
        );
        // Reserve before cloning logical history or constructing JSON. Six
        // bytes per UTF-8 byte bounds JSON escaping; numeric fields need <=20.
        let mut estimate = add(256, times(tokens.len() as u64, 11)?)?;
        for name in self.past.keys() {
            estimate = add(estimate, add(512, times(name.len() as u64, 6)?)?)?;
        }
        for tensor in self.past.values() {
            estimate = add(estimate, times(tensor.shape.len() as u64, 22)?)?;
        }
        if let Some(cpu) = &self.cpu_kv {
            for entry in &cpu.entries {
                estimate = add(
                    estimate,
                    add(
                        512,
                        times((entry.pair.0.len() + entry.pair.1.len()) as u64, 6)?,
                    )?,
                )?;
            }
        }
        let _metadata = reserve(memory, times(estimate, 2)?)?;
        for (name, tensor) in &self.past {
            anyhow::ensure!(
                tensor.layout.is_contiguous(&tensor.shape),
                "Native state tensor '{name}' is not contiguous; implement an explicit layout-preserving codec before saving this executor"
            );
        }
        let mut names = self.past.keys().collect::<Vec<_>>();
        names.sort();
        let manifest = Manifest {
            tokens: tokens.to_vec(),
            len: self.len,
            past: names
                .iter()
                .map(|name| {
                    let t = &self.past[*name];
                    TensorSpec {
                        name: (*name).clone(),
                        dtype: t.dtype as u8,
                        shape: t.shape.clone(),
                        bytes: t.as_bytes().len() as u64,
                    }
                })
                .collect(),
            cpu: self.cpu_kv.as_ref().map(|cpu| {
                cpu.entries
                    .iter()
                    .map(|entry| {
                        let mut shape = entry.shape.clone();
                        shape[2] = cpu.logical_len;
                        CpuSpec {
                            pair: entry.pair.clone(),
                            dtype: entry.dtype as u8,
                            shape,
                        }
                    })
                    .collect()
            }),
            hidden: self.last_hidden.as_ref().map(Vec::len),
        };
        anyhow::ensure!(
            manifest.payload_bytes()? == self.bytes,
            "Native state snapshot payload accounting disagrees with its tensor layouts; reset this session and report the decoder state contract"
        );
        let json = serde_json::to_vec(&manifest).context("encode Native state file manifest")?;
        anyhow::ensure!(
            json.len() as u64 <= estimate && json.len() as u64 <= limits.max_manifest_bytes,
            "Native state manifest requires {} bytes, exceeding its admitted estimate {estimate} or configured bound {}; increase the manifest limit or shorten the state",
            json.len(),
            limits.max_manifest_bytes
        );
        let size = add(add(HEADER + DIGEST, json.len() as u64)?, self.bytes)?;
        anyhow::ensure!(
            size <= limits.max_bytes,
            "Native state file requires {size} bytes, exceeding max_bytes {}; increase the file bound or save a shorter state",
            limits.max_bytes
        );
        let mut hash = Sha256::new();
        let mut put = |data: &[u8]| -> anyhow::Result<()> {
            writer
                .write_all(data)
                .context("write Native state file; check storage space and permissions")?;
            hash.update(data);
            Ok(())
        };
        put(MAGIC)?;
        put(&identity)?;
        put(&(json.len() as u64).to_le_bytes())?;
        put(&self.bytes.to_le_bytes())?;
        put(&json)?;
        for name in names {
            put(self.past[name].as_bytes())?;
        }
        if let Some(cpu) = &self.cpu_kv {
            for entry in &cpu.entries {
                for rows in &entry.rows {
                    put(rows)?;
                }
            }
        }
        if let Some(hidden) = &self.last_hidden {
            for value in hidden {
                put(&value.to_le_bytes())?;
            }
        }
        writer
            .write_all(&hash.finalize())
            .context("write Native state file checksum; retry with sufficient storage")?;
        Ok(size)
    }
}

pub(crate) struct DecodedState {
    pub tokens: Vec<TokenId>,
    pub state: NativePastSnapshot,
    pub leases: Vec<MemoryLease>,
    pub encoded_bytes: u64,
}
impl NativeDecodeSession {
    pub(crate) fn read_state_file(
        &self,
        reader: &mut impl Read,
        limits: NativeStateFileLimits,
        memory: &dyn MemoryGovernor,
    ) -> anyhow::Result<DecodedState> {
        limits.validate()?;
        anyhow::ensure!(
            self.supports_past_snapshots(),
            "Native state files require host-owned state; select the original host-state executor"
        );
        let identity = self.state_identity.context("Native state files are disabled for this model; reload with EngineConfig.native_state_files=Required")?;
        let mut header = [0_u8; HEADER as usize];
        reader
            .read_exact(&mut header)
            .context("read Native state file header; restore a complete NXRT state file")?;
        anyhow::ensure!(
            &header[..8] == MAGIC,
            "unsupported Native state file magic/version; use a v1 NXRT state file (llama.cpp binary caches are not interchangeable)"
        );
        anyhow::ensure!(
            header[8..40] == identity,
            "Native state file model identity differs from the loaded model, metadata, tokenizer or precision; load its original model package"
        );
        let manifest_len = u64::from_le_bytes(header[40..48].try_into()?);
        let payload_len = u64::from_le_bytes(header[48..56].try_into()?);
        let total = add(add(HEADER + DIGEST, manifest_len)?, payload_len)?;
        anyhow::ensure!(
            manifest_len > 0
                && manifest_len <= limits.max_manifest_bytes
                && total <= limits.max_bytes,
            "Native state file declares {manifest_len} manifest bytes and {payload_len} payload bytes ({total} total), exceeding limits {limits:?}; restore a smaller file or increase explicit bounds"
        );
        // JSON vectors, strings and layout records are conservatively covered
        // before parsing. Payload admission covers input bytes plus Tensor's
        // aligned copy; no executor mutation occurs while decoding.
        let metadata_lease = reserve(memory, times(manifest_len, 16)?)?;
        let mut json = buffer(manifest_len)?;
        reader.read_exact(&mut json).context(
            "read complete Native state manifest; check read access and restore an intact file",
        )?;
        let manifest: Manifest = serde_json::from_slice(&json).context(
            "parse Native state manifest; restore an intact file with the supported schema",
        )?;
        self.validate_file_manifest(&manifest, limits)?;
        let calculated = manifest.payload_bytes()?;
        anyhow::ensure!(
            payload_len == calculated,
            "Native state payload length {payload_len} disagrees with tensor layouts ({calculated}); restore an intact state file"
        );
        let row_metadata = manifest.cpu.as_ref().map_or(Ok(0), |entries| {
            entries.iter().try_fold(0, |sum, entry| {
                let (blocks, _) = entry.rows()?;
                add(
                    sum,
                    times(blocks as u64, std::mem::size_of::<Vec<u8>>() as u64)?,
                )
            })
        })?;
        let lease = reserve(
            memory,
            add(
                add(
                    times(payload_len, 2)?,
                    times(manifest.tokens.len() as u64, 4)?,
                )?,
                row_metadata,
            )?,
        )?;
        let mut hash = Sha256::new();
        hash.update(header);
        hash.update(&json);
        let mut take = |count: u64| -> anyhow::Result<Vec<u8>> {
            let mut data = buffer(count)?;
            reader
                .read_exact(&mut data)
                .context("read complete Native state tensor payload; check read access and restore an intact file")?;
            hash.update(&data);
            Ok(data)
        };
        let mut past = HashMap::new();
        for spec in manifest.past {
            let raw = take(spec.bytes)?;
            let tensor =
                Tensor::from_raw(dtype(spec.dtype)?, spec.shape, &raw).with_context(|| {
                    format!("allocate imported Native state tensor '{}'", spec.name)
                })?;
            past.insert(spec.name, tensor);
        }
        let cpu_kv = manifest
            .cpu
            .map(|entries| -> anyhow::Result<_> {
                let mut decoded = Vec::with_capacity(entries.len());
                for spec in entries {
                    let (blocks, row_size) = spec.rows()?;
                    let mut rows = Vec::with_capacity(blocks);
                    for _ in 0..blocks {
                        rows.push(take(row_size)?);
                    }
                    decoded.push(cpu::CpuKvSnapshotEntry {
                        pair: spec.pair,
                        shape: spec.shape,
                        dtype: dtype(spec.dtype)?,
                        rows,
                    });
                }
                Ok(cpu::CpuKvSnapshot {
                    logical_len: manifest.len,
                    entries: decoded,
                })
            })
            .transpose()?;
        let last_hidden = manifest
            .hidden
            .map(|len| -> anyhow::Result<_> {
                Ok(take(times(len as u64, 4)?)?
                    .chunks_exact(4)
                    .map(|bytes| {
                        f32::from_le_bytes(bytes.try_into().expect("four-byte hidden chunk"))
                    })
                    .collect())
            })
            .transpose()?;
        let mut checksum = [0; 32];
        reader.read_exact(&mut checksum).context(
            "read complete Native state checksum; check read access and restore an intact file",
        )?;
        anyhow::ensure!(
            hash.finalize().as_slice() == checksum,
            "Native state checksum mismatch; the file is corrupt, restore a complete original copy"
        );
        let mut extra = [0];
        anyhow::ensure!(
            reader
                .read(&mut extra)
                .context("check Native state file boundary; ensure the file remains readable")?
                == 0,
            "Native state file contains trailing data; restore exactly one complete state file"
        );
        Ok(DecodedState {
            tokens: manifest.tokens,
            state: NativePastSnapshot {
                past,
                len: manifest.len,
                bytes: payload_len,
                last_hidden,
                cpu_kv,
            },
            leases: vec![lease, metadata_lease],
            encoded_bytes: total,
        })
    }

    fn validate_file_manifest(
        &self,
        manifest: &Manifest,
        limits: NativeStateFileLimits,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            manifest.len <= manifest.tokens.len() && manifest.tokens.len() <= limits.max_tokens,
            "Native state logical/materialized token counts {}/{} exceed max_tokens {} or disagree; restore an intact file",
            manifest.tokens.len(),
            manifest.len,
            limits.max_tokens
        );
        let expected_cpu = self.cpu_kv.as_ref();
        anyhow::ensure!(
            manifest.cpu.is_some() == expected_cpu.is_some(),
            "Native state CPU KV storage mode differs from this executor; load with the original state storage configuration"
        );
        let mut expected = self
            .present_to_past
            .values()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        if let Some(cpu) = expected_cpu {
            for (pair, _, _) in cpu.state_layout() {
                expected.remove(pair.1.as_str());
            }
        }
        let empty = manifest.len == 0
            && manifest.tokens.is_empty()
            && manifest.past.is_empty()
            && manifest.hidden.is_none();
        anyhow::ensure!(
            empty || expected.len() == manifest.past.len(),
            "Native state tensor count differs from the model; restore the original state file"
        );
        for spec in &manifest.past {
            anyhow::ensure!(
                expected.remove(spec.name.as_str()),
                "Native state tensor '{}' is unexpected or duplicated; restore an intact file",
                spec.name
            );
            let kind = dtype(spec.dtype)?;
            anyhow::ensure!(
                bytes(kind, &spec.shape)? == spec.bytes,
                "Native state tensor '{}' byte count differs from its dtype/shape; restore an intact file",
                spec.name
            );
            let meta = self
                .session
                .inputs()
                .iter()
                .find(|m| m.name == spec.name)
                .with_context(|| {
                    format!(
                        "Native state port '{}' is absent from the loaded model",
                        spec.name
                    )
                })?;
            anyhow::ensure!(
                meta.dtype == kind && meta.shape.len() == spec.shape.len(),
                "Native state tensor '{}' dtype/rank differs from declared model port; load its original model",
                spec.name
            );
            let facts = csa::RecordTensorFacts::contiguous(kind, &spec.shape);
            if let Some(record) = self.compressed_state.record_for_past(&spec.name) {
                csa::validate_record_transition(
                    record,
                    facts,
                    facts,
                    manifest.len,
                    manifest.len,
                    1,
                )?;
            } else if let Some(carry) = self
                .compressed_state
                .carries()
                .find(|s| s.input == spec.name)
            {
                csa::validate_carry_transition(carry, facts, facts, 1)?;
            } else {
                let axis =
                    meta.shape.len().checked_sub(2).context(
                        "Native state tensor rank is below 2; unsupported cache geometry",
                    )?;
                for (i, dim) in meta.shape.iter().enumerate() {
                    let wanted = match dim {
                        Dim::Static(n) => *n,
                        Dim::Symbolic(_) if i == 0 => 1,
                        Dim::Symbolic(_) if i == axis => manifest.len,
                        _ => bail!(
                            "Native state port '{}' axis {i} is symbolic outside batch/sequence; export an explicit state geometry",
                            spec.name
                        ),
                    };
                    anyhow::ensure!(
                        spec.shape[i] == wanted,
                        "Native state tensor '{}' axis {i} is {}, expected {wanted}; restore an intact file",
                        spec.name,
                        spec.shape[i]
                    );
                }
            }
        }
        if let (Some(specs), Some(cpu)) = (&manifest.cpu, expected_cpu) {
            let layouts = cpu.state_layout().collect::<Vec<_>>();
            anyhow::ensure!(
                layouts.len() == specs.len(),
                "Native state CPU KV binding count differs from the model; restore the original file"
            );
            for (spec, (pair, shape, kind)) in specs.iter().zip(layouts) {
                anyhow::ensure!(
                    spec.pair == *pair
                        && dtype(spec.dtype)? == kind
                        && spec.shape.len() == 4
                        && shape.len() == 4
                        && spec.shape[2] == manifest.len
                        && spec
                            .shape
                            .iter()
                            .zip(shape)
                            .enumerate()
                            .all(|(i, (a, b))| i == 2 || a == b),
                    "Native state CPU KV binding {:?} has mismatched ports/dtype/geometry/cursor; load its original model",
                    spec.pair
                );
                spec.rows()?;
            }
        }
        match (&self.hidden_output, manifest.hidden) {
            (None, None) => {}
            (Some(name), Some(width)) => {
                let meta = self
                    .session
                    .outputs()
                    .iter()
                    .find(|m| m.name == *name)
                    .context("loaded Native hidden port is missing; reload the model")?;
                anyhow::ensure!(
                    width > 0
                        && meta.shape.last().is_none_or(|d| match d {
                            Dim::Static(n) => *n == width,
                            _ => true,
                        }),
                    "Native state hidden width {width} differs from output '{name}'; load its original model"
                );
            }
            (_, None) if manifest.len == 0 => {}
            _ => bail!(
                "Native state hidden conditioning differs from the model; restore an intact state file"
            ),
        }
        Ok(())
    }
}
impl CpuSpec {
    fn rows(&self) -> anyhow::Result<(usize, u64)> {
        anyhow::ensure!(
            self.shape.len() == 4,
            "Native CPU KV file layout must have rank 4; restore an intact file"
        );
        let blocks = self.shape[0]
            .checked_mul(self.shape[1])
            .context("Native CPU KV block count overflow")?;
        let row = bytes(dtype(self.dtype)?, &[self.shape[2], self.shape[3]])?;
        Ok((blocks, row))
    }
}
impl Manifest {
    fn payload_bytes(&self) -> anyhow::Result<u64> {
        let mut result = 0;
        for t in &self.past {
            result = add(result, t.bytes)?;
        }
        if let Some(cpu) = &self.cpu {
            for spec in cpu {
                let (blocks, rows) = spec.rows()?;
                result = add(result, times(blocks as u64, rows)?)?;
            }
        }
        if let Some(hidden) = self.hidden {
            result = add(result, times(hidden as u64, 4)?)?;
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "state_file_tests.rs"]
mod tests;
