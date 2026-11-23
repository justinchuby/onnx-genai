use onnx_runtime_loader::proto::onnx;
use onnx_runtime_loader::{KeyedWeightFile, load_model_bytes_with_keyed_weights};
use prost::Message;
use std::path::{Path, PathBuf};

fn initializer(
    name: &str,
    data_type: i32,
    dims: &[i64],
    fields: &[(&str, &str)],
) -> onnx::TensorProto {
    onnx::TensorProto {
        name: name.into(),
        data_type,
        dims: dims.to_vec(),
        data_location: onnx::tensor_proto::DataLocation::External as i32,
        external_data: fields
            .iter()
            .map(|(key, value)| onnx::StringStringEntryProto {
                key: (*key).into(),
                value: (*value).into(),
            })
            .collect(),
        ..Default::default()
    }
}

fn model(initializer: Vec<onnx::TensorProto>) -> Vec<u8> {
    onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 21,
        }],
        graph: Some(onnx::GraphProto {
            name: "weights".into(),
            initializer,
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec()
}

fn directory(test: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(test);
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn mapped_bytes<'a>(
    store: &'a onnx_runtime_loader::WeightStore,
    weight: &'a onnx_runtime_ir::WeightRef,
) -> &'a [u8] {
    let (id, offset, len) = store
        .external_mmap_provenance(weight)
        .expect("must be mmap-backed");
    let file = store.mmap_full_bytes(id).unwrap();
    let bytes = store.bytes(weight).unwrap();
    assert!(std::ptr::eq(bytes, &file[offset..offset + len]));
    bytes
}

fn safetensors(dir: &Path, file: &str, header: &str, payload: &[u8]) -> KeyedWeightFile {
    let mut json = header.as_bytes().to_vec();
    json.resize(json.len().next_multiple_of(8), b' ');
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend(json);
    bytes.extend(payload);
    let path = dir.join(file);
    std::fs::write(&path, bytes).unwrap();
    KeyedWeightFile::safetensors(path)
}

fn gguf(dir: &Path, file: &str, packed_row: u64) -> (PathBuf, Vec<u8>) {
    gguf_fixture(dir, file, packed_row, 2)
}

fn gguf_fixture(dir: &Path, file: &str, packed_row: u64, rows: u64) -> (PathBuf, Vec<u8>) {
    fn string(out: &mut Vec<u8>, value: &str) {
        out.extend((value.len() as u64).to_le_bytes());
        out.extend(value.as_bytes());
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(2u64.to_le_bytes());
    bytes.extend(1u64.to_le_bytes());
    string(&mut bytes, "general.quantization_version");
    bytes.extend(4u32.to_le_bytes());
    bytes.extend(2u32.to_le_bytes());
    let dense_offset = if rows == 0 { 0 } else { 64 };
    for (name, dims, kind, offset) in [
        ("packed", [packed_row, rows], 2u32, 0u64), // two 18-byte Q4_0 blocks when nonempty
        ("matrix", [3, 2], 0, dense_offset), // F32; 28 bytes of alignment padding when nonempty
    ] {
        string(&mut bytes, name);
        bytes.extend(2u32.to_le_bytes());
        for dim in dims {
            bytes.extend(dim.to_le_bytes());
        }
        bytes.extend(kind.to_le_bytes());
        bytes.extend(offset.to_le_bytes());
    }
    let base = bytes.len().next_multiple_of(32);
    bytes.resize(base, 0);
    if rows != 0 {
        bytes.extend(0u8..36);
        bytes.resize(base + 64, 0);
    }
    bytes.extend((1..=6).flat_map(|x| (x as f32).to_le_bytes()));
    let path = dir.join(file);
    std::fs::write(&path, &bytes).unwrap();
    (path, bytes)
}

#[test]
fn binds_empty_packed_gguf_sharing_an_offset_with_dense_data() {
    let dir = directory("keyed_gguf_empty");
    let (path, _) = gguf_fixture(&dir, "empty.gguf", 32, 0);
    let (graph, store) = load_model_bytes_with_keyed_weights(
        &model(vec![
            initializer("packed", 2, &[0], &[]),
            initializer("matrix", 1, &[2, 3], &[]),
        ]),
        &dir,
        &[KeyedWeightFile::gguf(path)],
    )
    .unwrap();
    let mut lengths = graph
        .initializers
        .values()
        .map(|w| mapped_bytes(&store, w).len())
        .collect::<Vec<_>>();
    lengths.sort_unstable();
    assert_eq!(lengths, [0, 24]);
}

#[test]
fn binds_packed_and_dense_gguf_with_exact_geometry() {
    let dir = directory("keyed_gguf");
    let (path, _) = gguf(&dir, "weights.gguf", 32);
    let files = [KeyedWeightFile::gguf(path).rename("matrix", "renamed")];
    let (graph, store) = load_model_bytes_with_keyed_weights(
        &model(vec![
            initializer("packed", 2, &[36], &[]),
            initializer("renamed", 1, &[2, 3], &[("location", ""), ("offset", "")]),
        ]),
        &dir,
        &files,
    )
    .unwrap();
    let mut weights = graph
        .initializers
        .values()
        .map(|w| mapped_bytes(&store, w))
        .collect::<Vec<_>>();
    weights.sort_by_key(|data| data.len());
    assert_eq!(
        weights[0],
        (1..=6)
            .flat_map(|x| (x as f32).to_le_bytes())
            .collect::<Vec<_>>()
    );
    assert_eq!(weights[1], (0u8..36).collect::<Vec<_>>());
    for (dtype, dims, message) in [
        (2, &[64][..], "Uint8 [64]"),
        (3, &[36][..], "Int8 [36]"),
        (2, &[2, 18][..], "Uint8 [2, 18]"),
    ] {
        let error = load_model_bytes_with_keyed_weights(
            &model(vec![initializer("packed", dtype, dims, &[])]),
            &dir,
            &files,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("gguf:q4_0") && error.contains(message),
            "{error}"
        );
    }
}

#[test]
fn rejects_malformed_gguf_geometry_and_endianness() {
    let dir = directory("keyed_gguf_malformed");
    let (bad_row, _) = gguf(&dir, "bad_row.gguf", 16);
    let (truncated, mut bytes) = gguf(&dir, "truncated.gguf", 32);
    bytes.pop();
    std::fs::write(&truncated, &bytes).unwrap();
    let big_endian = dir.join("big-endian.gguf");
    bytes[4..8].copy_from_slice(&3u32.to_be_bytes());
    std::fs::write(&big_endian, &bytes).unwrap();
    let bytes = model(vec![initializer("packed", 2, &[36], &[])]);
    for (path, detail) in [
        (bad_row, "block size 32"),
        (truncated, "extends beyond"),
        (big_endian, "big-endian"),
    ] {
        let error =
            load_model_bytes_with_keyed_weights(&bytes, &dir, &[KeyedWeightFile::gguf(path)])
                .unwrap_err()
                .to_string();
        assert!(error.contains(detail), "{error}");
    }
}

#[test]
fn binds_safetensors_and_ordinary_data_with_explicit_precedence() {
    let dir = directory("keyed_safetensors");
    let same = safetensors(
        &dir,
        "implicit.safetensors",
        r#"{"weight":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
        &[9, 8],
    );
    let renamed = safetensors(
        &dir,
        "explicit.safetensors",
        r#"{"other":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
        &[1, 2],
    )
    .rename("other", "weight");
    std::fs::write(dir.join("ordinary.bin"), [0, 3]).unwrap();
    let (graph, store) = load_model_bytes_with_keyed_weights(
        &model(vec![
            initializer("weight", 2, &[2], &[("location", ""), ("offset", "")]),
            initializer(
                "ordinary",
                2,
                &[1],
                &[("location", "ordinary.bin"), ("offset", "1")],
            ),
        ]),
        &dir,
        &[same, renamed],
    )
    .unwrap();
    let mut payloads = graph
        .initializers
        .values()
        .map(|w| mapped_bytes(&store, w))
        .collect::<Vec<_>>();
    payloads.sort_by_key(|data| data.len());
    assert_eq!(payloads, [&[3][..], &[1, 2]]);
}

#[test]
fn diagnoses_mapping_and_metadata_errors() {
    let dir = directory("keyed_mapping_errors");
    let header = r#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
    let a = safetensors(&dir, "a.safetensors", header, &[2]);
    let b = safetensors(&dir, "b.safetensors", header, &[3]);
    let duplicate = safetensors(
        &dir,
        "duplicate.safetensors",
        r#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[2],
    );
    let bad_range = safetensors(
        &dir,
        "bad_range.safetensors",
        r#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
        &[0, 1],
    );
    for (files, name, fields, detail) in [
        (
            vec![a.clone(), b.clone()],
            "weight",
            vec![],
            "matches more than one",
        ),
        (vec![a.clone()], "missing", vec![], "has no keyed weight"),
        (
            vec![a.clone().rename("unknown", "weight")],
            "weight",
            vec![],
            "absent or repeated",
        ),
        (
            vec![
                a.clone().rename("weight", "name"),
                b.clone().rename("weight", "name"),
            ],
            "name",
            vec![],
            "mapped more than once",
        ),
        (
            vec![a.clone(), b.rename("weight", "typo")],
            "weight",
            vec![],
            "is not an external initializer",
        ),
        (
            vec![a.clone()],
            "weight",
            vec![("offset", "bogus")],
            "unsupported external_data offset",
        ),
        (
            vec![a.clone()],
            "weight",
            vec![("length", "2")],
            "differs from container length",
        ),
        (
            vec![a],
            "weight",
            vec![("location", ""), ("location", "")],
            "repeats",
        ),
        (vec![duplicate], "weight", vec![], "repeated tensor name"),
        (
            vec![bad_range],
            "weight",
            vec![],
            "invalid SafeTensors byte range",
        ),
        (vec![], "weight", vec![], "missing 'location'"),
    ] {
        let error = load_model_bytes_with_keyed_weights(
            &model(vec![initializer(name, 2, &[1], &fields)]),
            &dir,
            &files,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains(detail), "{error}");
    }
}
