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
        .expect("Failed to generate ORT bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Failed to write bindings");
}

fn header_matches_api_version(header_path: &Path) -> bool {
    std::fs::read_to_string(header_path).is_ok_and(|header| {
        header.lines().any(|line| {
            line.split_whitespace()
                .eq(["#define", "ORT_API_VERSION", ORT_API_VERSION])
        })
    })
}
