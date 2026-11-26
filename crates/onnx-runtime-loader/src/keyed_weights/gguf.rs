use super::{Entries, Entry, GgufMetadataValue, byte_len};
use onnx_runtime_ir::DataType;
use std::collections::{BTreeMap, HashSet};

struct Reader<'a> {
    rest: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let (head, rest) = self
            .rest
            .split_at_checked(n)
            .ok_or("truncated GGUF header")?;
        self.rest = rest;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn size(&mut self) -> Result<usize, String> {
        usize::try_from(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            .map_err(|_| "GGUF size exceeds the host address space".into())
    }

    fn string(&mut self) -> Result<&'a str, String> {
        let len = self.size()?;
        std::str::from_utf8(self.take(len)?).map_err(|_| "invalid UTF-8 in GGUF string".into())
    }

    fn metadata(&mut self, kind: u32, allow_array: bool) -> Result<(), String> {
        if let Some(width) = scalar_width(kind) {
            self.take(width)?;
        } else {
            match kind {
                8 => {
                    self.string()?;
                }
                9 if allow_array => {
                    let element = self.u32()?;
                    let count = self.size()?;
                    if let Some(width) = scalar_width(element) {
                        self.take(
                            count
                                .checked_mul(width)
                                .ok_or("GGUF metadata array size overflows")?,
                        )?;
                    } else if element == 8 {
                        for _ in 0..count {
                            self.metadata(element, false)?;
                        }
                    } else {
                        return Err(format!("invalid GGUF array element type {element}"));
                    }
                }
                _ => return Err(format!("invalid GGUF metadata type {kind}")),
            }
        }
        Ok(())
    }
}

fn scalar_width(kind: u32) -> Option<usize> {
    Some(match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

pub(super) fn parse(data: &[u8]) -> Result<Entries, String> {
    parse_with_metadata(data).map(|(entries, _)| entries)
}
pub(super) fn parse_with_metadata(
    data: &[u8],
) -> Result<(Entries, BTreeMap<&str, GgufMetadataValue<'_>>), String> {
    let mut adapter_metadata = BTreeMap::new();
    let mut reader = Reader { rest: data };
    match reader.take(4)? {
        b"GGUF" => {}
        b"FUGG" => {
            return Err("big-endian GGUF is unsupported; convert to little-endian first".into());
        }
        _ => return Err("invalid GGUF magic".into()),
    }
    let version = reader.u32()?;
    if matches!(version.swap_bytes(), 2 | 3) {
        return Err("big-endian GGUF is unsupported; convert to little-endian first".into());
    }
    if !matches!(version, 2 | 3) {
        return Err(format!(
            "unsupported GGUF version {version}; use little-endian v2 or v3"
        ));
    }
    let tensors = reader.size()?;
    let metadata = reader.size()?;
    let mut alignment = 32usize;
    let mut quantization_version = None;
    let mut keys = HashSet::new();
    for _ in 0..metadata {
        let key = reader.string()?;
        if key.is_empty() || !keys.insert(key) {
            return Err(format!("empty or repeated GGUF metadata key {key:?}"));
        }
        let kind = reader.u32()?;
        if key == "general.alignment" {
            if kind != 4 {
                return Err("GGUF general.alignment must be UINT32".into());
            }
            alignment = reader.u32()? as usize;
            if !alignment.is_power_of_two() {
                return Err(format!(
                    "GGUF general.alignment {alignment} is not a power of two"
                ));
            }
        } else if key == "general.quantization_version" {
            if kind != 4 {
                return Err("GGUF general.quantization_version must be UINT32".into());
            }
            quantization_version = Some(reader.u32()?);
        } else if key == "general.type" || key.starts_with("adapter.") {
            let value = match kind {
                8 => GgufMetadataValue::String(
                    reader
                        .string()
                        .map_err(|error| format!("GGUF metadata {key:?}: {error}"))?,
                ),
                6 => GgufMetadataValue::Float32(f32::from_le_bytes(
                    reader
                        .take(4)
                        .map_err(|error| format!("GGUF metadata {key:?}: {error}"))?
                        .try_into()
                        .unwrap(),
                )),
                _ => {
                    reader
                        .metadata(kind, true)
                        .map_err(|error| format!("GGUF metadata {key:?}: {error}"))?;
                    GgufMetadataValue::Other { kind }
                }
            };
            adapter_metadata.insert(key, value);
        } else {
            reader.metadata(kind, true)?;
        }
    }
    let mut entries = Entries::new();
    for _ in 0..tensors {
        let key = reader.string()?;
        let rank = reader.u32()?;
        if !(1..=4).contains(&rank) {
            return Err(format!("GGUF tensor {key:?} has invalid rank {rank}"));
        }
        let mut dims = (0..rank)
            .map(|_| reader.size())
            .collect::<Result<Vec<_>, _>>()?;
        dims.reverse(); // GGML's first dimension is the contiguous axis.
        let encoding = reader.u32()?;
        let offset = reader.size()?;
        let (dtype, packed_format, len) = if let Some((format, block_elements, block_bytes)) =
            packed_layout(encoding)
        {
            if quantization_version != Some(2) {
                return Err(format!(
                    "GGUF tensor {key:?} uses {format} but general.quantization_version is {quantization_version:?}; supply a version-2 quantized GGUF"
                ));
            }
            let (&row, outer) = dims.split_last().unwrap();
            if row % block_elements != 0 {
                return Err(format!(
                    "GGUF tensor {key:?}: contiguous dimension {row} is not divisible by the {format} block size {block_elements}"
                ));
            }
            let len = if dims.contains(&0) {
                0
            } else {
                outer
                    .iter()
                    .try_fold(block_bytes, |bytes, &dim| bytes.checked_mul(dim))
                    .and_then(|bytes| bytes.checked_mul(row / block_elements))
                    .ok_or_else(|| format!("GGUF tensor {key:?}: packed tensor size overflows"))?
            };
            dims = vec![len];
            (DataType::Uint8, Some(format), len)
        } else {
            let dtype = match encoding {
                0 => DataType::Float32,
                1 => DataType::Float16,
                24 => DataType::Int8,
                25 => DataType::Int16,
                26 => DataType::Int32,
                27 => DataType::Int64,
                28 => DataType::Float64,
                30 => DataType::BFloat16,
                _ => {
                    return Err(format!(
                        "GGUF tensor {key:?} uses unsupported GGML type {encoding}"
                    ));
                }
            };
            let len = byte_len(dtype, &dims).map_err(|e| format!("GGUF tensor {key:?}: {e}"))?;
            (dtype, None, len)
        };
        insert(
            &mut entries,
            key,
            Entry {
                dtype,
                packed_format,
                dims,
                offset,
                len,
            },
        )?;
    }
    if entries.is_empty() {
        return Ok((entries, adapter_metadata));
    }
    let header = data.len() - reader.rest.len();
    let base = header
        .checked_next_multiple_of(alignment)
        .ok_or("GGUF data alignment overflows")?;
    let mut ordered: Vec<_> = entries.iter_mut().collect();
    ordered.sort_unstable_by_key(|(_, entry)| entry.offset);
    let mut end = base;
    for (key, entry) in ordered {
        if entry.offset % alignment != 0 {
            return Err(format!(
                "GGUF tensor {key:?} offset {} is not aligned to {alignment}",
                entry.offset
            ));
        }
        entry.offset = base
            .checked_add(entry.offset)
            .ok_or("GGUF tensor offset overflows")?;
        if (entry.len != 0 && entry.offset < end)
            || entry
                .offset
                .checked_add(entry.len)
                .is_none_or(|e| e > data.len())
        {
            return Err(format!(
                "GGUF tensor {key:?} overlaps or extends beyond the file"
            ));
        }
        if entry.len != 0 {
            end = entry.offset + entry.len;
        }
    }

    // GGML type IDs and the ONNX DequantizePacked opset-29 (elements, bytes)
    // contract. The importer takes the exact span and never decodes these blocks.
    fn packed_layout(kind: u32) -> Option<(&'static str, usize, usize)> {
        Some(match kind {
            2 => ("gguf:q4_0", 32, 18),
            3 => ("gguf:q4_1", 32, 20),
            6 => ("gguf:q5_0", 32, 22),
            7 => ("gguf:q5_1", 32, 24),
            8 => ("gguf:q8_0", 32, 34),
            9 => ("gguf:q8_1", 32, 36),
            10 => ("gguf:q2_k", 256, 84),
            11 => ("gguf:q3_k", 256, 110),
            12 => ("gguf:q4_k", 256, 144),
            13 => ("gguf:q5_k", 256, 176),
            14 => ("gguf:q6_k", 256, 210),
            15 => ("gguf:q8_k", 256, 292),
            16 => ("gguf:iq2_xxs", 256, 66),
            17 => ("gguf:iq2_xs", 256, 74),
            18 => ("gguf:iq3_xxs", 256, 98),
            19 => ("gguf:iq1_s", 256, 50),
            20 => ("gguf:iq4_nl", 32, 18),
            21 => ("gguf:iq3_s", 256, 110),
            22 => ("gguf:iq2_s", 256, 82),
            23 => ("gguf:iq4_xs", 256, 136),
            29 => ("gguf:iq1_m", 256, 56),
            34 => ("gguf:tq1_0", 256, 54),
            35 => ("gguf:tq2_0", 256, 66),
            39 => ("gguf:mxfp4", 32, 17),
            40 => ("gguf:nvfp4", 64, 36),
            41 => ("gguf:q1_0", 128, 18),
            42 => ("gguf:q2_0", 64, 18),
            _ => return None,
        })
    }
    Ok((entries, adapter_metadata))
}

fn insert(entries: &mut Entries, key: &str, entry: Entry) -> Result<(), String> {
    if key.is_empty() || entries.insert(key.to_owned(), entry).is_some() {
        return Err(format!("empty or repeated GGUF tensor name {key:?}"));
    }
    Ok(())
}
