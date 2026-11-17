//! Canonical ONNX schema messages shared by the IR and loader.
//!
//! The loader reexports these exact generated types and descriptors, so
//! consumers share one protobuf schema without depending on the loader.

#![allow(clippy::all, missing_docs, non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
