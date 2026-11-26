// Independent little-endian GGUF v3 fixture writer. No runtime-loader calls.
pub type Fact = (String, u32, Vec<u8>);
pub type Tensor = (String, u32, Vec<u64>, Vec<u8>);
pub fn string(value: &str) -> Vec<u8> {
    let mut bytes=(value.len() as u64).to_le_bytes().to_vec();bytes.extend(value.as_bytes());bytes
}
pub fn container(facts:&[Fact],tensors:&[Tensor])->Vec<u8> {
    let mut bytes=b"GGUF".to_vec();bytes.extend(3u32.to_le_bytes());
    bytes.extend((tensors.len() as u64).to_le_bytes());bytes.extend((facts.len() as u64).to_le_bytes());
    for (key,kind,value) in facts {bytes.extend(string(key));bytes.extend(kind.to_le_bytes());bytes.extend(value);}
    let mut payload=Vec::new();
    for (name,kind,shape,data) in tensors {
        payload.resize(payload.len().next_multiple_of(32),0);
        bytes.extend(string(name));bytes.extend((shape.len() as u32).to_le_bytes());
        // GGML stores the contiguous dimension first; payload is row-major.
        for dim in shape.iter().rev() {bytes.extend(dim.to_le_bytes());}
        bytes.extend(kind.to_le_bytes());bytes.extend((payload.len() as u64).to_le_bytes());payload.extend(data);
    }
    bytes.resize(bytes.len().next_multiple_of(32),0);bytes.extend(payload);bytes
}
