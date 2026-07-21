//! Build script for onnx-genai-ort-sys.
//!
//! Locates ONNX Runtime and generates Rust bindings from its C API headers.
//!
//! ORT is found via (in priority order):
//! 1. ORT_LIB_DIR env var — path to directory containing libonnxruntime.so/dylib/dll
//! 2. ORT_ROOT env var — path to ORT installation root (lib/ and include/ subdirs)
//! 3. Automatic download of prebuilt ORT from GitHub releases

use std::env;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

// Keep this aligned with the ORT headers used by bindgen. ORT 1.27.x exposes
// ORT_API_VERSION 27, so the downloaded runtime must also be 1.27.x.
const ORT_VERSION: &str = "1.27.0";
const ORT_API_VERSION: &str = "27";
const ORT_RELEASE_BASE: &str = "https://github.com/microsoft/onnxruntime/releases/download";

const ORT_ARCHIVE_CHECKSUMS: &[(&str, &str)] = &[
    (
        "onnxruntime-linux-x64-1.27.0.tgz",
        "547e40a48f1fe73e3f812d7c88a948612c23f896b91e4e2ee1e232d7b468246f",
    ),
    (
        "onnxruntime-osx-arm64-1.27.0.tgz",
        "545e81c58152353acb0d1e8bd6ce4b62f830c0961f5b3acfedc790ffd76e477a",
    ),
    (
        "onnxruntime-win-x64-1.27.0.zip",
        "c5c81710938e68079ff1a192b04897faabe4b43830d48f39f27ecd4e16138bfc",
    ),
    (
        "onnxruntime-win-arm64-1.27.0.zip",
        "a32f2650575b3c20df462e337519fd1cc4105356130d11dba9771c6f374d952f",
    ),
];

fn main() {
    println!("cargo:rerun-if-env-changed=ONNX_GENAI_METAL_EP_LIB");
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
    ensure_major_version_runtime_link(&lib_dir);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=onnxruntime");
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
        println!("cargo:rustc-link-search=native={}", lib_dir);
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
        panic!(
            "ORT_ROOT={} does not contain include/onnxruntime_c_api.h",
            root
        );
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
    let (os, ext) = prebuilt_target();

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

    verify_archive_checksum(&download_path, &filename);

    // Extract
    let parent_dir = target_dir.parent().unwrap();
    std::fs::create_dir_all(parent_dir).unwrap();

    if ext == "tgz" {
        let status = std::process::Command::new("tar")
            .args(["xzf"])
            .arg(&download_path)
            .arg("-C")
            .arg(parent_dir)
            .status()
            .expect("Failed to run tar");
        if !status.success() {
            panic!("Failed to extract ORT archive");
        }
        // Rename extracted directory
        let extracted = parent_dir.join(format!("onnxruntime-{}-{}", os, ORT_VERSION));
        if extracted.exists() {
            if target_dir.exists() {
                std::fs::remove_dir_all(target_dir).unwrap();
            }
            std::fs::rename(&extracted, target_dir).unwrap();
        }
    } else {
        // .zip (Windows). Extract with the pure-Rust `zip` crate instead of
        // shelling out to an external `unzip` binary, which is not present on a
        // clean Windows host and would fail the native build before compilation
        // (docs/CROSS_PLATFORM.md, "ORT Windows bootstrap"). This path is
        // identical on Linux/macOS/Windows and needs no external tool.
        extract_zip(&download_path, parent_dir);
        let extracted = parent_dir.join(format!("onnxruntime-{}-{}", os, ORT_VERSION));
        if extracted.exists() {
            if target_dir.exists() {
                std::fs::remove_dir_all(target_dir).unwrap();
            }
            std::fs::rename(&extracted, target_dir).unwrap();
        }
    }

    // Cleanup
    let _ = std::fs::remove_file(&download_path);
}

/// Extract a `.zip` archive into `dest_dir` using the pure-Rust `zip` crate.
///
/// Portable across Linux/macOS/Windows with no external `unzip` binary. The
/// output tree mirrors what `unzip -d dest_dir archive` produced: every archive
/// entry is written verbatim under `dest_dir`, preserving the archive's internal
/// directory layout (e.g. the top-level `onnxruntime-<os>-<version>/` folder the
/// rest of the build later renames into place). Unix permission bits are
/// preserved when the archive records them so extracted shared libraries keep
/// their mode; this is a no-op on Windows.
fn extract_zip(archive_path: &Path, dest_dir: &Path) {
    let file = std::fs::File::open(archive_path).unwrap_or_else(|err| {
        panic!(
            "Failed to open downloaded ORT archive {} for extraction: {err}. \
             Delete it and re-run to re-download, or set ORT_ROOT to a local \
             ONNX Runtime installation to skip the download entirely.",
            archive_path.display()
        )
    });
    let mut zip = zip::ZipArchive::new(file).unwrap_or_else(|err| {
        panic!(
            "Failed to read ORT zip archive {}: {err}. The download may be \
             truncated or corrupt; delete it and re-run, or set ORT_ROOT to a \
             local ONNX Runtime installation.",
            archive_path.display()
        )
    });

    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).unwrap_or_else(|err| {
            panic!(
                "Failed to read entry #{i} of ORT zip {}: {err}. The archive may \
                 be corrupt; delete it and re-run, or set ORT_ROOT to a local \
                 ONNX Runtime installation.",
                archive_path.display()
            )
        });

        // `enclosed_name` rejects absolute paths and `..` traversal, guarding
        // against zip-slip writes outside `dest_dir`.
        let Some(rel_path) = entry.enclosed_name() else {
            panic!(
                "ORT zip {} contains an unsafe entry path {:?}; refusing to \
                 extract outside {}.",
                archive_path.display(),
                entry.name(),
                dest_dir.display()
            )
        };
        let out_path = dest_dir.join(rel_path);

        if entry.is_dir() {
            std::fs::create_dir_all(&out_path).unwrap_or_else(|err| {
                panic!(
                    "Failed to create directory {} while extracting {}: {err}",
                    out_path.display(),
                    archive_path.display()
                )
            });
        } else {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent).unwrap_or_else(|err| {
                    panic!(
                        "Failed to create directory {} while extracting {}: {err}",
                        parent.display(),
                        archive_path.display()
                    )
                });
            }
            let mut out_file = std::fs::File::create(&out_path).unwrap_or_else(|err| {
                panic!(
                    "Failed to create file {} while extracting {}: {err}",
                    out_path.display(),
                    archive_path.display()
                )
            });
            std::io::copy(&mut entry, &mut out_file).unwrap_or_else(|err| {
                panic!(
                    "Failed to write extracted file {} from {}: {err}",
                    out_path.display(),
                    archive_path.display()
                )
            });

            // Preserve recorded unix permissions (e.g. executable bit on shared
            // objects). No-op on Windows, where zip entries carry no unix mode.
            #[cfg(unix)]
            if let Some(mode) = entry.unix_mode() {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&out_path, std::fs::Permissions::from_mode(mode));
            }
        }
    }
}

fn prebuilt_target() -> (&'static str, &'static str) {
    if target_os() == "linux" {
        ("linux-x64", "tgz")
    } else if target_os() == "macos" {
        if target_arch() == "aarch64" {
            ("osx-arm64", "tgz")
        } else {
            ("osx-x86_64", "tgz")
        }
    } else if target_os() == "windows" {
        if target_arch() == "aarch64" {
            ("win-arm64", "zip")
        } else {
            ("win-x64", "zip")
        }
    } else {
        panic!("Unsupported platform for automatic ORT download");
    }
}

fn verify_archive_checksum(download_path: &Path, filename: &str) {
    let Some(expected) = expected_archive_checksum(filename) else {
        let message = format!(
            "No SHA-256 checksum is pinned for ORT archive {filename}. \
             Add its official digest to ORT_ARCHIVE_CHECKSUMS in ort-sys/build.rs; \
             this build cannot verify the downloaded native runtime."
        );
        println!("cargo:warning={message}");
        eprintln!("WARNING: {message}");
        return;
    };

    let actual = sha256_file_hex(download_path).unwrap_or_else(|err| {
        panic!(
            "Failed to compute SHA-256 for downloaded ORT archive {}: {err}",
            download_path.display()
        )
    });

    if actual != expected {
        panic!(
            "SHA-256 mismatch for downloaded ORT archive {filename}: \
             expected {expected}, got {actual}. Refusing to extract native runtime."
        );
    }
}

fn expected_archive_checksum(filename: &str) -> Option<&'static str> {
    ORT_ARCHIVE_CHECKSUMS
        .iter()
        .find_map(|(known_filename, checksum)| (*known_filename == filename).then_some(*checksum))
}

fn sha256_file_hex(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn cached_prebuilt_matches_version(download_dir: &Path) -> bool {
    let header_path = download_dir.join("include").join("onnxruntime_c_api.h");
    header_matches_api_version(&header_path)
        && expected_versioned_runtime_path(&download_dir.join("lib")).exists()
}

fn header_matches_api_version(header_path: &Path) -> bool {
    let Ok(header) = std::fs::read_to_string(header_path) else {
        return false;
    };

    header.contains(&format!("#define ORT_API_VERSION {ORT_API_VERSION}"))
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

    if major_path.exists() {
        configure_macos_install_names(lib_dir, &major_path, &versioned_path);
        return;
    }

    #[cfg(unix)]
    std::os::unix::fs::symlink(&versioned_name, &major_path).unwrap_or_else(|err| {
        panic!(
            "Failed to create ONNX Runtime major-version symlink {} -> {}: {err}",
            major_path.display(),
            versioned_name
        )
    });

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
