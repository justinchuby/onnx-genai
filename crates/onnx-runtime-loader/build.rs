//! Generates the ONNX prost types from the vendored `proto/onnx.proto3`.
//!
//! We compile the schema with [`protox`] (a pure-Rust protobuf compiler) so the
//! build does not depend on a system `protoc` binary, then hand the resulting
//! `FileDescriptorSet` to `prost-build` for Rust code generation. The generated
//! file lands in `$OUT_DIR/onnx.rs` (named after the proto `package onnx;`) and
//! is `include!`d by `src/proto.rs`.

use std::path::PathBuf;

fn main() {
    let proto = "proto/onnx.proto3";
    println!("cargo:rerun-if-changed={proto}");
    println!("cargo:rerun-if-changed=build.rs");

    let file_descriptors = protox::compile([proto], ["proto"])
        .expect("failed to compile proto/onnx.proto3 with protox");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));

    prost_build::Config::new()
        .out_dir(&out_dir)
        .compile_fds(file_descriptors)
        .expect("prost-build failed to generate ONNX types");
}
