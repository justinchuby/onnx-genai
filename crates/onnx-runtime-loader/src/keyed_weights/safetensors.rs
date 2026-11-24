use super::{Entries, Entry, byte_len};
use onnx_runtime_ir::DataType;
use serde::de::{Error, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::collections::{BTreeMap, HashSet};

#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct Header(BTreeMap<String, Tensor>);

impl<'de> Deserialize<'de> for Header {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = Header;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a SafeTensors header object with unique tensor names")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Header, A::Error> {
                let mut keys = HashSet::new();
                let mut tensors = BTreeMap::new();
                while let Some(key) = map.next_key::<String>()? {
                    if key.is_empty() || !keys.insert(key.clone()) {
                        return Err(A::Error::custom(format!(
                            "empty or repeated tensor name {key:?}"
                        )));
                    }
                    if key == "__metadata__" {
                        let _: BTreeMap<String, String> = map.next_value()?;
                    } else {
                        tensors.insert(key, map.next_value()?);
                    }
                }
                Ok(Header(tensors))
            }
        }
        de.deserialize_map(HeaderVisitor)
    }
}

pub(super) fn parse(data: &[u8]) -> Result<Entries, String> {
    let (prefix, rest) = data
        .split_at_checked(8)
        .ok_or("truncated SafeTensors length prefix")?;
    let len = usize::try_from(u64::from_le_bytes(prefix.try_into().unwrap()))
        .map_err(|_| "SafeTensors header length exceeds host address space")?;
    if len > 100_000_000 {
        return Err("SafeTensors header exceeds the 100 MB format limit".into());
    }
    let (header, payload) = rest
        .split_at_checked(len)
        .ok_or("truncated SafeTensors header")?;
    if header.first() != Some(&b'{') {
        return Err("SafeTensors header must start with '{'".into());
    }
    let Header(tensors) =
        serde_json::from_slice(header).map_err(|e| format!("invalid SafeTensors header: {e}"))?;
    let mut tensors: Vec<_> = tensors.into_iter().collect();
    tensors.sort_unstable_by_key(|(_, tensor)| tensor.data_offsets);
    let mut entries = Entries::new();
    let mut end = 0;
    for (key, tensor) in tensors {
        let [start, stop] = tensor.data_offsets;
        if start != end || stop < start || stop > payload.len() {
            return Err(format!(
                "invalid SafeTensors byte range [{start}, {stop}) for key {key:?}; ranges must partition the data buffer"
            ));
        }
        let dtype = match tensor.dtype.as_str() {
            "BOOL" => DataType::Bool,
            "F4" => DataType::Float4E2M1,
            // Keyed bindings use the ONNX bit order. SafeTensors transports
            // opaque producer bytes and does not describe tiled layouts.
            "F6_E2M3" => DataType::Float6E2M3,
            "F6_E3M2" => DataType::Float6E3M2,
            "I8" => DataType::Int8,
            "U8" => DataType::Uint8,
            "I16" => DataType::Int16,
            "U16" => DataType::Uint16,
            "I32" => DataType::Int32,
            "U32" => DataType::Uint32,
            "I64" => DataType::Int64,
            "U64" => DataType::Uint64,
            "F16" => DataType::Float16,
            "BF16" => DataType::BFloat16,
            "F32" => DataType::Float32,
            "F64" => DataType::Float64,
            "C64" => DataType::Complex64,
            "F8_E4M3" => DataType::Float8E4M3FN,
            "F8_E4M3FNUZ" => DataType::Float8E4M3FNUZ,
            "F8_E5M2" => DataType::Float8E5M2,
            "F8_E5M2FNUZ" => DataType::Float8E5M2FNUZ,
            "F8_E8M0" => DataType::Float8E8M0,
            kind => {
                return Err(format!(
                    "unsupported SafeTensors dtype {kind:?} for key {key:?}"
                ));
            }
        };
        // SafeTensors requires whole bytes in each tensor, not in each row.
        // Check the product modulo eight without overflowing on large shapes.
        let bits_mod_byte = tensor
            .shape
            .iter()
            .fold(dtype.bit_size() % 8, |bits, dim| bits * (dim % 8) % 8);
        if bits_mod_byte != 0 {
            return Err(format!(
                "SafeTensors key {key:?}: {} shape {:?} does not fill whole bytes; export a logical element count divisible by {}",
                tensor.dtype,
                tensor.shape,
                if dtype.is_float6() { 4 } else { 2 }
            ));
        }
        let size =
            byte_len(dtype, &tensor.shape).map_err(|e| format!("SafeTensors key {key:?}: {e}"))?;
        if stop - start != size {
            return Err(format!(
                "SafeTensors key {key:?} declares {} payload bytes but {} {:?} requires {size}",
                stop - start,
                tensor.dtype,
                tensor.shape
            ));
        }
        entries.insert(
            key,
            Entry {
                dtype,
                packed_format: None,
                dims: tensor.shape,
                offset: 8 + len + start,
                len: size,
            },
        );
        end = stop;
    }
    if end != payload.len() {
        return Err("SafeTensors has unindexed trailing payload bytes".into());
    }
    Ok(entries)
}
