//! Generate C API bindings from the repository-owned, pinned ORT headers.
//! Native execution does not need an ORT installation or downloaded binary.
use std::env;
use std::path::{Path, PathBuf};

const ORT_VERSION: &str = "1.32.0";
const ORT_API_VERSION: &str = "32";

fn main() {
    println!("cargo:rerun-if-changed=third_party/onnxruntime/include");
    println!("cargo:rustc-env=NXRT_ORT_VERSION={ORT_VERSION}");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    let include_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("Cargo supplies CARGO_MANIFEST_DIR"),
    )
    .join("third_party/onnxruntime/include");
    for name in [
        "onnxruntime_c_api.h",
        "onnxruntime_ep_c_api.h",
        "onnxruntime_error_code.h",
    ] {
        assert!(
            include_dir.join(name).is_file(),
            "bundled ORT header {} is missing; restore the checked-in third_party/onnxruntime headers",
            include_dir.join(name).display()
        );
    }
    let header_path = include_dir.join("onnxruntime_c_api.h");
    assert!(
        header_matches_api_version(&header_path),
        "bundled ORT header {} must declare API {ORT_API_VERSION} for ORT {ORT_VERSION}; restore the pinned headers",
        header_path.display()
    );

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        .allowlist_function("Ort.*")
        .allowlist_type("Ort.*")
        .allowlist_var("ORT_.*")
        .blocklist_function("OrtGetApiBase")
        .derive_debug(true)
        .derive_default(true)
        .prepend_enum_name(false)
        .generate()
        .unwrap_or_else(|error| {
            panic!(
                "generate ORT API {ORT_API_VERSION} bindings from {}: {error}; install libclang and set LIBCLANG_PATH to its directory, then restore the pinned headers if needed",
                header_path.display()
            )
        });

    let bindings_path = out_dir.join("bindings.rs");
    bindings
        .write_to_file(&bindings_path)
        .unwrap_or_else(|error| {
            panic!(
                "write generated ORT bindings to {}: {error}; free disk space and ensure Cargo's output directory is writable",
                bindings_path.display()
            )
        });
}

fn header_matches_api_version(header_path: &Path) -> bool {
    std::fs::read_to_string(header_path).is_ok_and(|header| {
        header.lines().any(|line| {
            line.split_whitespace()
                .eq(["#define", "ORT_API_VERSION", ORT_API_VERSION])
        })
    })
}
