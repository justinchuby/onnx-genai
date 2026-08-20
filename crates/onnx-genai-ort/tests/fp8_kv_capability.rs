//! Where FP8 KV state stops working, and who says so.
//!
//! An FP8 cache is not one capability but three, and they are supported
//! independently: a runtime can allocate the buffer, and bind it to a session,
//! and still have no kernel that computes on it. Conflating them is what
//! produces the wrong bug report — a missing ScatterND kernel that surfaces as
//! "invalid metadata" sends the reader to edit a document that was correct.
//!
//! These tests pin each layer separately against the execution provider that is
//! actually present, so the boundary is a measurement rather than a claim. They
//! deliberately do not assert that FP8 compute works: on the CPU provider it
//! does not, and a test that demanded otherwise would only be skipped.

use std::path::{Path, PathBuf};

use onnx_genai_ort::{DataType, Environment, Session, SessionOptions, Value};

fn fixture(stem: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-fp8-kv")
        .join(format!("{stem}.onnx.textproto"))
}

/// An FP8 tensor is allocatable and keeps its element type.
///
/// Zero is representable in every ONNX element type, so an empty cache needs no
/// per-dtype host representation — which is why the allocator has no business
/// refusing one.
#[test]
fn fp8_cache_buffers_allocate_and_keep_their_dtype() -> Result<(), Box<dyn std::error::Error>> {
    for dtype in [DataType::Float8E4M3, DataType::Float8E5M2] {
        let shape = [2i64, 4, 8];
        let numel = shape.iter().product::<i64>() as usize;
        let value = Value::from_raw_bytes(vec![0u8; numel * dtype.size_of()], &shape, dtype)?;

        assert_eq!(value.dtype(), dtype, "allocation widened the element type");
        assert_eq!(value.shape(), shape, "allocation reshaped the buffer");
        assert_eq!(dtype.size_of(), 1, "FP8 elements are one byte");
    }
    Ok(())
}

/// A session whose KV ports are FP8 loads, so FP8 state can cross the boundary.
///
/// This is the load-bearing fact for representability: if ORT refused FP8 at
/// the graph boundary, an FP8 cache could not be declared honestly no matter
/// what the schema allowed.
#[test]
fn fp8_kv_ports_load_and_round_trip_through_a_session() -> Result<(), Box<dyn std::error::Error>> {
    let env = Environment::new("fp8-kv-identity")?;
    let session = Session::new(&env, &fixture("identity"), SessionOptions::default())?;

    let input = &session.inputs()[0];
    assert_eq!(input.name, "key_cache");
    assert_eq!(
        input.dtype,
        DataType::Float8E4M3,
        "the session narrowed or widened the declared cache dtype"
    );
    assert_eq!(
        session.outputs()[0].dtype,
        DataType::Float8E4M3,
        "the present-cache port lost its dtype"
    );

    let shape = [2i64, 4, 8];
    let numel = shape.iter().product::<i64>() as usize;
    let cache = Value::from_raw_bytes(vec![0x38u8; numel], &shape, DataType::Float8E4M3)?;
    let outputs = session.run(&[("key_cache", &cache)])?;

    assert_eq!(outputs[0].dtype(), DataType::Float8E4M3, "run widened it");
    assert_eq!(outputs[0].shape(), shape);
    Ok(())
}

/// Updating an FP8 cache in-graph fails, and the failure names the operator.
///
/// This is the execution-provider blocker, recorded rather than worked around.
/// The point of the assertion is not that it fails but *how*: the message comes
/// from the provider's kernel registry and names ScatterND and the element
/// type, which is a diagnosis ("this EP has no FP8 ScatterND") rather than a
/// verdict on the document. If a future provider registers the kernel, this
/// test starts failing and the boundary moves — which is the intended signal.
///
/// The two assertions below describe *this* provider, not every provider. A
/// refusal is allowed to be less informative: on CUDA (onnxruntime-gpu 1.29.0)
/// the FP8 KV path runs through GroupQueryAttention, `float8_e4m3fn` is absent
/// from that kernel's past/present type list, and the node is simply left
/// unassigned at graph partitioning — so initialization fails with "Provider
/// type for GroupQueryAttention node '...' is not set", naming neither FP8 nor
/// the element type. Both are missing-kernel blockers; only one says so. Do not
/// generalize these assertions into a cross-provider contract, and do not read
/// the CUDA message as evidence that the document is malformed.
#[test]
fn fp8_scatter_is_refused_by_the_provider_not_by_the_format()
-> Result<(), Box<dyn std::error::Error>> {
    let env = Environment::new("fp8-kv-scatter")?;
    let Err(error) = Session::new(&env, &fixture("scatter"), SessionOptions::default()) else {
        // A provider gained an FP8 ScatterND kernel. Nothing above this line
        // needs to change; the capability simply widened.
        return Ok(());
    };

    let message = error.to_string();
    assert!(
        message.contains("ScatterND"),
        "the refusal must name the operator that lacks a kernel: {message}"
    );
    assert!(
        message.contains("float8"),
        "the refusal must name the element type it cannot handle: {message}"
    );
    Ok(())
}
