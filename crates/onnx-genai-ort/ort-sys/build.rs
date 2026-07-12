//! Build script for onnx-genai-ort-sys.
//!
//! Locates ONNX Runtime and generates Rust bindings from its C API headers.
//!
//! ORT is found via (in priority order):
//! 1. ORT_LIB_DIR env var — path to directory containing libonnxruntime.so/dylib/dll
//! 2. ORT_ROOT env var — path to ORT installation root (lib/ and include/ subdirs)
//! 3. Automatic download of prebuilt ORT from GitHub releases

use std::env;
use std::path::{Path, PathBuf};

const ORT_VERSION: &str = "1.27.0";
const ORT_RELEASE_BASE: &str = "https://github.com/microsoft/onnxruntime/releases/download";

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Find ORT installation
    let ort_root = find_ort_root();
    let lib_dir = ort_root.join("lib");
    let include_dir = ort_root.join("include");

    // Verify header exists
    let header_path = include_dir.join("onnxruntime_c_api.h");
    if !header_path.exists() {
        panic!(
            "Cannot find onnxruntime_c_api.h at {}. \
             Set ORT_ROOT to your ONNX Runtime installation directory, \
             or ORT_LIB_DIR to the directory containing the shared library.",
            header_path.display()
        );
    }

    // Link
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=onnxruntime");

    // For downstream crates
    println!("cargo:ort_lib_dir={}", lib_dir.display());
    println!("cargo:ort_include_dir={}", include_dir.display());

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        .allowlist_function("Ort.*")
        .allowlist_type("Ort.*")
        .allowlist_var("ORT_.*")
        .derive_debug(true)
        .derive_default(true)
        .prepend_enum_name(false)
        .generate()
        .expect("Failed to generate ORT bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Failed to write bindings");
}

fn find_ort_root() -> PathBuf {
    // 1. ORT_LIB_DIR — just the lib directory, infer root
    if let Ok(lib_dir) = env::var("ORT_LIB_DIR") {
        let lib_path = PathBuf::from(&lib_dir);
        // Assume root is parent of lib/
        if let Some(root) = lib_path.parent() {
            if root.join("include").join("onnxruntime_c_api.h").exists() {
                return root.to_path_buf();
            }
        }
        // Maybe lib_dir IS the root (flat layout)
        if lib_path.join("onnxruntime_c_api.h").exists() {
            return lib_path;
        }
        // Use as-is, trust the user
        println!("cargo:rustc-link-search=native={}", lib_dir);
        return lib_path.parent().unwrap_or(&lib_path).to_path_buf();
    }

    // 2. ORT_ROOT — explicit installation root
    if let Ok(root) = env::var("ORT_ROOT") {
        let root_path = PathBuf::from(&root);
        if root_path.join("include").join("onnxruntime_c_api.h").exists() {
            return root_path;
        }
        panic!(
            "ORT_ROOT={} does not contain include/onnxruntime_c_api.h",
            root
        );
    }

    // 3. Try pkg-config
    if let Ok(output) = std::process::Command::new("pkg-config")
        .args(["--variable=libdir", "onnxruntime"])
        .output()
    {
        if output.status.success() {
            let lib_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let root = PathBuf::from(&lib_dir).parent().unwrap().to_path_buf();
            if root.join("include").join("onnxruntime_c_api.h").exists() {
                return root;
            }
        }
    }

    // 4. Auto-download prebuilt
    let download_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("ort-prebuilt");
    if download_dir.join("include").join("onnxruntime_c_api.h").exists() {
        return download_dir;
    }

    download_prebuilt(&download_dir);
    download_dir
}

fn download_prebuilt(target_dir: &Path) {
    let (os, ext) = if cfg!(target_os = "linux") {
        ("linux-x64", "tgz")
    } else if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            ("osx-arm64", "tgz")
        } else {
            ("osx-x86_64", "tgz")
        }
    } else if cfg!(target_os = "windows") {
        ("win-x64", "zip")
    } else {
        panic!("Unsupported platform for automatic ORT download");
    };

    let filename = format!("onnxruntime-{}-{}.{}", os, ORT_VERSION, ext);
    let url = format!("{}/v{}/{}", ORT_RELEASE_BASE, ORT_VERSION, filename);

    eprintln!("Downloading ONNX Runtime {} from {}", ORT_VERSION, url);

    // Use curl for download (available on all CI platforms)
    let download_path = target_dir.parent().unwrap().join(&filename);
    let status = std::process::Command::new("curl")
        .args(["-L", "-o"])
        .arg(&download_path)
        .arg(&url)
        .status()
        .expect("Failed to run curl. Install curl or set ORT_ROOT manually.");

    if !status.success() {
        panic!("Failed to download ORT from {}", url);
    }

    // Extract
    std::fs::create_dir_all(target_dir).unwrap();

    if ext == "tgz" {
        let status = std::process::Command::new("tar")
            .args(["xzf"])
            .arg(&download_path)
            .arg("-C")
            .arg(target_dir.parent().unwrap())
            .status()
            .expect("Failed to run tar");
        if !status.success() {
            panic!("Failed to extract ORT archive");
        }
        // Rename extracted directory
        let extracted = target_dir.parent().unwrap().join(format!("onnxruntime-{}-{}", os, ORT_VERSION));
        if extracted.exists() {
            std::fs::rename(&extracted, target_dir).unwrap();
        }
    } else {
        // zip on Windows
        let status = std::process::Command::new("unzip")
            .arg(&download_path)
            .arg("-d")
            .arg(target_dir.parent().unwrap())
            .status()
            .expect("Failed to run unzip");
        if !status.success() {
            panic!("Failed to extract ORT archive");
        }
        let extracted = target_dir.parent().unwrap().join(format!("onnxruntime-{}-{}", os, ORT_VERSION));
        if extracted.exists() {
            std::fs::rename(&extracted, target_dir).unwrap();
        }
    }

    // Cleanup
    let _ = std::fs::remove_file(&download_path);
}
