//! Build script for onnx-genai-ort-sys.
//!
//! Locates ONNX Runtime and generates Rust bindings from its C API headers.
//!
//! ORT is found via (in priority order):
//! 1. ORT_LIB_DIR env var — path to directory containing libonnxruntime.so/dylib/dll
//! 2. ORT_ROOT env var — path to ORT installation root (lib/ and include/ subdirs)
//! 3. Pinned native runtime from PyPI, with matching bundled official C headers

use std::env;
use std::path::{Path, PathBuf};

mod build_support;

// This pin matches the bundled Microsoft C headers and the pinned PyPI wheels.
const ORT_VERSION: &str = "1.32.0";
const ORT_API_VERSION: &str = "32";

fn main() {
    println!("cargo:rerun-if-env-changed=ONNX_GENAI_METAL_EP_LIB");
    println!("cargo:rerun-if-env-changed=ORT_ROOT");
    println!("cargo:rerun-if-env-changed=ORT_LIB_DIR");
    println!("cargo:rerun-if-changed=third_party/onnxruntime/include");
    println!("cargo:rustc-env=NXRT_ORT_VERSION={ORT_VERSION}");
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

    if !header_matches_api_version(&header_path) {
        panic!(
            "ONNX Runtime header {} does not declare API {ORT_API_VERSION}, required for ORT {ORT_VERSION}; point ORT_ROOT to the matching install, or unset ORT_ROOT/ORT_LIB_DIR to use the pinned PyPI runtime and bundled headers",
            header_path.display()
        );
    }

    // Make the build-time ORT location available to dependents and, on ELF/Mach-O
    // targets, as a loader search path. The runtime itself is opened explicitly
    // from Rust so users can select the exact shared library before the first ORT
    // API call; do not emit `rustc-link-lib=onnxruntime` here.
    ensure_major_version_runtime_link(&lib_dir);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    if target_os() == "macos" || target_os() == "linux" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }

    // For downstream crates
    println!("cargo:ort_lib_dir={}", lib_dir.display());
    println!("cargo:ort_include_dir={}", include_dir.display());

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

fn find_ort_root() -> PathBuf {
    // A plugin EP must use the same ORT dylib as the host process. When the
    // Metal plugin is selected at build/run time, prefer the ORT installation
    // recorded in the plugin's Mach-O dependencies.
    if target_os() == "macos"
        && target_arch() == "aarch64"
        && let Some(root) = metal_plugin_ort_root()
    {
        return root;
    }

    // 1. ORT_LIB_DIR — just the lib directory, infer root
    if let Ok(lib_dir) = env::var("ORT_LIB_DIR") {
        let lib_path = PathBuf::from(&lib_dir);
        // Assume root is parent of lib/
        if let Some(root) = lib_path.parent()
            && root.join("include").join("onnxruntime_c_api.h").exists()
        {
            return root.to_path_buf();
        }
        // Maybe lib_dir IS the root (flat layout)
        if lib_path.join("onnxruntime_c_api.h").exists() {
            return lib_path;
        }
        // Use as-is, trust the user
        println!("cargo:rustc-link-search=native={lib_dir}");
        return lib_path.parent().unwrap_or(&lib_path).to_path_buf();
    }

    // 2. ORT_ROOT — explicit installation root
    if let Ok(root) = env::var("ORT_ROOT") {
        let root_path = PathBuf::from(&root);
        if root_path
            .join("include")
            .join("onnxruntime_c_api.h")
            .exists()
        {
            return root_path;
        }
        panic!("ORT_ROOT={root} does not contain include/onnxruntime_c_api.h");
    }

    // 3. Try pkg-config
    if let Ok(output) = std::process::Command::new("pkg-config")
        .args(["--variable=libdir", "onnxruntime"])
        .output()
        && output.status.success()
    {
        let lib_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let root = PathBuf::from(&lib_dir).parent().unwrap().to_path_buf();
        if root.join("include").join("onnxruntime_c_api.h").exists() {
            return root;
        }
    }

    // 4. Auto-download prebuilt
    let download_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("ort-prebuilt");
    if cached_prebuilt_matches_version(&download_dir) {
        return download_dir;
    }
    if download_dir.exists() {
        eprintln!(
            "Removing stale ONNX Runtime prebuilt cache at {} (expected {})",
            download_dir.display(),
            ORT_VERSION
        );
        std::fs::remove_dir_all(&download_dir).unwrap_or_else(|err| {
            panic!(
                "Failed to remove stale ORT prebuilt cache at {}: {err}",
                download_dir.display()
            )
        });
    }

    download_prebuilt(&download_dir);
    download_dir
}

fn metal_plugin_ort_root() -> Option<PathBuf> {
    let plugin = PathBuf::from(env::var_os("ONNX_GENAI_METAL_EP_LIB")?);
    if !plugin.is_file() {
        return None;
    }
    let output = std::process::Command::new("otool")
        .arg("-L")
        .arg(&plugin)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let runtime = stdout
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().next())
        .map(PathBuf::from)
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("libonnxruntime") && name.ends_with(".dylib"))
                && path.is_absolute()
                && path.is_file()
        })?;
    let root = runtime.parent()?.parent()?.to_path_buf();
    if !root.join("include").join("onnxruntime_c_api.h").is_file() {
        return None;
    }
    println!(
        "cargo:warning=using ONNX Runtime from Metal plugin dependency: {}",
        runtime.display()
    );
    Some(root)
}

fn download_prebuilt(target_dir: &Path) {
    let artifact = build_support::artifact_for(&target_os(), &target_arch())
        .unwrap_or_else(|message| panic!("{message}"));
    let parent = target_dir.parent().expect("ORT cache has a parent");
    std::fs::create_dir_all(parent)
        .unwrap_or_else(|e| panic!("cannot create ORT cache {}: {e}", parent.display()));
    let download_path = parent.join(artifact.filename);
    eprintln!(
        "Downloading ONNX Runtime {ORT_VERSION} native runtime from {}",
        artifact.url
    );
    let mut last_failure = String::new();
    for attempt in 1..=4 {
        let result = std::process::Command::new("curl")
            .args(["-sSL", "--connect-timeout", "30", "--max-time", "300", "-w", "%{http_code}", "-o"])
            .arg(&download_path).arg(artifact.url).output()
            .unwrap_or_else(|e| panic!("cannot run curl to download {}: {e}; install curl or set ORT_ROOT to a local ORT {ORT_VERSION} install", artifact.url));
        let status = String::from_utf8_lossy(&result.stdout).trim().to_owned();
        if result.status.success() && status == "200" {
            last_failure.clear();
            break;
        }
        last_failure = format!(
            "curl exit {:?}, HTTP {status}: {}",
            result.status.code(),
            String::from_utf8_lossy(&result.stderr).trim()
        );
        if attempt < 4 {
            let pause = 3u64 * (1 << (attempt - 1));
            eprintln!(
                "ORT PyPI download attempt {attempt}/4 failed: {last_failure}; retrying in {pause}s"
            );
            std::thread::sleep(std::time::Duration::from_secs(pause));
        }
    }
    if !last_failure.is_empty() {
        panic!(
            "cannot download ONNX Runtime {} from {}: {last_failure}; the upstream service may be unavailable. Retry later or set ORT_ROOT to a local install",
            artifact.filename, artifact.url
        );
    }
    build_support::verify_checksum(&download_path, &artifact)
        .unwrap_or_else(|message| panic!("{message}"));
    let required = expected_versioned_runtime_path(Path::new(""));
    build_support::extract_native_libraries(
        &download_path,
        &target_dir.join("lib"),
        required.to_str().expect("static library name"),
    )
    .unwrap_or_else(|message| panic!("{message}"));
    let include = target_dir.join("include");
    std::fs::create_dir_all(&include).unwrap_or_else(|e| {
        panic!(
            "cannot create ORT include directory {}: {e}",
            include.display()
        )
    });
    let bundled =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo manifest directory"))
            .join("third_party/onnxruntime/include");
    for name in [
        "onnxruntime_c_api.h",
        "onnxruntime_ep_c_api.h",
        "onnxruntime_error_code.h",
    ] {
        std::fs::copy(bundled.join(name), include.join(name)).unwrap_or_else(|e| {
            panic!(
                "cannot install bundled ORT header {name} into {}: {e}",
                include.display()
            )
        });
    }
    std::fs::write(target_dir.join("VERSION_NUMBER"), ORT_VERSION).unwrap_or_else(|e| {
        panic!(
            "cannot record ORT {ORT_VERSION} cache version in {}: {e}",
            target_dir.display()
        )
    });
    let _ = std::fs::remove_file(download_path);
}

fn cached_prebuilt_matches_version(download_dir: &Path) -> bool {
    let header_path = download_dir.join("include").join("onnxruntime_c_api.h");
    header_matches_api_version(&header_path)
        && download_dir
            .join("include/onnxruntime_ep_c_api.h")
            .is_file()
        && download_dir
            .join("include/onnxruntime_error_code.h")
            .is_file()
        && std::fs::read_to_string(download_dir.join("VERSION_NUMBER"))
            .is_ok_and(|version| version.trim() == ORT_VERSION)
        && expected_versioned_runtime_path(&download_dir.join("lib")).is_file()
}

fn header_matches_api_version(header_path: &Path) -> bool {
    let Ok(header) = std::fs::read_to_string(header_path) else {
        return false;
    };

    header.lines().any(|line| {
        let fields: Vec<_> = line.split_whitespace().collect();
        fields.as_slice() == ["#define", "ORT_API_VERSION", ORT_API_VERSION]
    })
}

fn ensure_major_version_runtime_link(lib_dir: &Path) {
    let Some((major_name, versioned_name)) = runtime_library_names() else {
        return;
    };

    let major_path = lib_dir.join(major_name);
    let versioned_path = lib_dir.join(&versioned_name);
    if !versioned_path.exists() {
        panic!(
            "Cannot find expected ONNX Runtime {} shared library at {}",
            ORT_VERSION,
            versioned_path.display()
        );
    }

    #[cfg(unix)]
    {
        let unversioned = if target_os() == "macos" {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        };
        build_support::ensure_runtime_links(lib_dir, &versioned_name, &[major_name, unversioned])
            .unwrap_or_else(|message| panic!("{message}"));
    }

    configure_macos_install_names(lib_dir, &major_path, &versioned_path);
}

fn configure_macos_install_names(lib_dir: &Path, major_path: &Path, versioned_path: &Path) {
    if target_os() != "macos" {
        return;
    }

    set_macos_install_name(major_path, versioned_path);

    let unversioned_path = lib_dir.join("libonnxruntime.dylib");
    if unversioned_path.exists() {
        set_macos_install_name(major_path, &unversioned_path);
    }
}

fn set_macos_install_name(major_path: &Path, library_path: &Path) {
    let desired_id = major_path.to_string_lossy();
    let current_id = std::process::Command::new("otool")
        .arg("-D")
        .arg(library_path)
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .and_then(|stdout| stdout.lines().nth(1).map(str::to_string));

    if current_id.as_deref() != Some(desired_id.as_ref()) {
        let status = std::process::Command::new("install_name_tool")
            .arg("-id")
            .arg(desired_id.as_ref())
            .arg(library_path)
            .status()
            .expect("Failed to run install_name_tool");

        if !status.success() {
            panic!(
                "Failed to set ONNX Runtime install name for {}",
                library_path.display()
            );
        }
    }

    let status = std::process::Command::new("codesign")
        .args(["--force", "--sign", "-"])
        .arg(library_path)
        .status()
        .expect("Failed to run codesign after modifying ONNX Runtime install name");

    if !status.success() {
        panic!(
            "Failed to ad-hoc codesign modified ONNX Runtime library {}",
            library_path.display()
        );
    }
}

fn expected_versioned_runtime_path(lib_dir: &Path) -> PathBuf {
    if target_os() == "macos" {
        lib_dir.join(format!("libonnxruntime.{ORT_VERSION}.dylib"))
    } else if target_os() == "linux" {
        lib_dir.join(format!("libonnxruntime.so.{ORT_VERSION}"))
    } else if target_os() == "windows" {
        lib_dir.join("onnxruntime.dll")
    } else {
        lib_dir.join(format!("libonnxruntime.{ORT_VERSION}"))
    }
}

fn runtime_library_names() -> Option<(&'static str, String)> {
    if target_os() == "macos" {
        Some((
            "libonnxruntime.1.dylib",
            format!("libonnxruntime.{ORT_VERSION}.dylib"),
        ))
    } else if target_os() == "linux" {
        Some((
            "libonnxruntime.so.1",
            format!("libonnxruntime.so.{ORT_VERSION}"),
        ))
    } else {
        None
    }
}

fn target_os() -> String {
    env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| env::consts::OS.to_string())
}

fn target_arch() -> String {
    env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| env::consts::ARCH.to_string())
}
