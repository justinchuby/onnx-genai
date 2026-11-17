//! Build script for onnx-genai-ort-sys.
//!
//! Locates ONNX Runtime and generates Rust bindings from its C API headers.
//!
//! ORT is found via (in priority order):
//! 1. ORT_LIB_DIR env var — path to directory containing libonnxruntime.so/dylib/dll
//! 2. ORT_ROOT env var — path to ORT installation root (lib/ and include/ subdirs)
//! 3. pkg-config
//! 4. Automatic download of Microsoft's NuGet SDK (matching headers and runtime)

use std::env;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

mod build_support;

// Keep this aligned with the ORT headers used by bindgen. ORT 1.29.x exposes
// ORT_API_VERSION 29, so the downloaded runtime must also be 1.29.x.
const ORT_VERSION: &str = "1.29.0";
const ORT_API_VERSION: &str = "29";
const ORT_SDK_BASE: &str = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime";
// Verified against NuGet's x-ms-meta-sha512 metadata for the official package.
const ORT_SDK_SHA256: &str = "4557021fec4da3924e2e12ff44b1cc50b176de0d2c5c7daa89ab11c208b3a454";

fn main() {
    println!("cargo:rerun-if-changed=build_support.rs");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ORT_ROOT");
    println!("cargo:rerun-if-env-changed=ORT_LIB_DIR");
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
    let target = build_support::sdk_target(
        &target_os(),
        &target_arch(),
        &env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default(),
    )
    .unwrap_or_else(|err| panic!("{err}"));
    let filename = format!("microsoft.ml.onnxruntime.{ORT_VERSION}.nupkg");
    let url = format!("{ORT_SDK_BASE}/{ORT_VERSION}/{filename}");

    eprintln!("Downloading ONNX Runtime {ORT_VERSION} from {url}");

    // Use curl for download (available on all CI platforms). curl exits 0 on an
    // HTTP 404 unless told otherwise, so capture the HTTP status via -w and
    // check it explicitly. Without this a 404 "Not Found" body is saved as the
    // "archive" and only surfaces later as an opaque tar/zip "unrecognized
    // format" failure that names neither the URL nor the status (RULES.md #1).
    //
    // We wrap curl in a Rust-level retry loop instead of relying on curl's
    // `--retry-all-errors`: that flag only exists in curl >=7.71.0, but the
    // manylinux_2_28 (AlmaLinux 8) build container ships curl 7.61.1, which
    // rejects it outright (exit 2). curl's plain `--retry` does NOT retry the
    // transient failures we actually hit on CI (exit 52 empty-reply, exit 56
    // connection-died) on ANY curl version, so this portable loop covers them.
    // `--connect-timeout` / `--max-time` are ancient and safe on curl 7.61.
    let download_path = target_dir.parent().unwrap().join(&filename);
    let max_attempts = 4;
    let mut http_code = String::new();
    let mut output = None;
    for attempt in 1..=max_attempts {
        let result = std::process::Command::new("curl")
            .args([
                "-sSL",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                // Bound each attempt so a hung connection can't consume the job.
                "--connect-timeout",
                "30",
                "--max-time",
                "300",
                "-w",
                "%{http_code}",
                "-o",
            ])
            .arg(&download_path)
            .arg(&url)
            .output()
            .expect("Failed to run curl. Install curl or set ORT_ROOT manually.");

        http_code = String::from_utf8_lossy(&result.stdout).trim().to_string();
        let ok = result.status.success() && http_code == "200";
        if ok || attempt == max_attempts {
            output = Some(result);
            break;
        }

        // Transient failure (e.g. curl exit 52 empty-reply / 56 connection-died,
        // or a 5xx) that is NOT the final attempt: warn and back off, then retry.
        let stderr = String::from_utf8_lossy(&result.stderr);
        let stderr_tail: String = stderr.trim().chars().take(500).collect();
        let backoff = 3u64 * (1 << (attempt - 1)); // 3s, 6s, 12s
        eprintln!(
            "curl download of {url} failed (attempt {attempt}/{max_attempts}, \
             curl exit {:?}, HTTP status {http_code}); retrying in {backoff}s. stderr: {stderr_tail}",
            result.status.code(),
        );
        std::thread::sleep(std::time::Duration::from_secs(backoff));
    }
    let output = output.expect("curl retry loop produced no output");

    if !output.status.success() {
        panic!(
            "curl failed to download ONNX Runtime from {url} \
             (curl exit {:?}, HTTP status {http_code}). stderr: {}. \
             Set ORT_ROOT to a local ONNX Runtime {ORT_VERSION} install to skip the download.",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim(),
        );
    }

    if http_code != "200" {
        panic!(
            "Failed to download ONNX Runtime SDK: {url} returned HTTP {http_code} \
             (expected 200). {}. Check access to api.nuget.org or set ORT_ROOT \
             to a local ONNX Runtime {ORT_VERSION} SDK with API {ORT_API_VERSION} headers.",
            describe_download(&download_path),
        );
    }

    let actual = sha256_file_hex(&download_path)
        .unwrap_or_else(|err| panic!("Cannot hash SDK {}: {err}", download_path.display()));
    if actual != ORT_SDK_SHA256 {
        panic!(
            "SHA-256 mismatch for ONNX Runtime SDK from {url}: expected \
             {ORT_SDK_SHA256}, got {actual}. Refusing to extract native runtime. \
             Delete {} and retry, or set ORT_ROOT to a trusted local SDK.",
            download_path.display()
        );
    }

    // Guard against a non-archive body (e.g. an HTML/text error page or a
    // redirect returned with HTTP 200) reaching tar/zip as an "unrecognized
    // format". This turns a cryptic extractor error into an actionable one that
    // names the URL, byte size, and first bytes actually downloaded.
    verify_archive_magic(&download_path, &url);
    build_support::extract_sdk(
        &download_path,
        target_dir,
        &target,
        ORT_VERSION,
        ORT_API_VERSION,
    )
    .unwrap_or_else(|err| {
        panic!(
            "Cannot install ONNX Runtime SDK {} from {url} into {}: {err}. \
             Delete the partial installation and retry, or set ORT_ROOT to a \
             local ONNX Runtime {ORT_VERSION} SDK with API {ORT_API_VERSION} headers.",
            download_path.display(),
            target_dir.display()
        )
    });

    // Cleanup
    let _ = std::fs::remove_file(&download_path);
}

/// Read up to `max` leading bytes of `path`, returning what was read (possibly
/// fewer bytes, or empty if the file cannot be opened).
fn read_first_bytes(path: &Path, max: usize) -> Vec<u8> {
    let mut buf = vec![0_u8; max];
    match std::fs::File::open(path).and_then(|mut f| f.read(&mut buf)) {
        Ok(n) => {
            buf.truncate(n);
            buf
        }
        Err(_) => Vec::new(),
    }
}

/// Human-readable description of a just-downloaded file: its size and a lossy
/// preview of its first bytes. Used to make download/extract failures name what
/// actually came back (a 404 body, an HTML error page, a truncated file).
fn describe_download(path: &Path) -> String {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let head = read_first_bytes(path, 64);
    let preview = String::from_utf8_lossy(&head);
    format!(
        "Saved file is {size} bytes; first bytes: {:?}",
        preview.trim()
    )
}

/// Fail fast if the downloaded file does not start with the magic bytes of the
/// expected archive format, before handing it to tar/zip. Catches error pages
/// or redirects that were served with HTTP 200.
fn verify_archive_magic(path: &Path, url: &str) {
    let head = read_first_bytes(path, 8);
    if !head.starts_with(b"PK") {
        panic!(
            "Downloaded file from {url} is not a valid zip archive. {}. The download most \
             likely returned an error page or a redirect instead of the ONNX Runtime binary. \
             Confirm access to the official api.nuget.org SDK package and \
             re-run, or set ORT_ROOT to a local ONNX Runtime {ORT_VERSION} install.",
            describe_download(path),
        );
    }
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

    header
        .lines()
        .any(|line| line.trim() == format!("#define ORT_API_VERSION {ORT_API_VERSION}"))
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
    for name in [
        major_name,
        if target_os() == "macos" {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        },
    ] {
        let path = lib_dir.join(name);
        if !path.exists() {
            std::os::unix::fs::symlink(&versioned_name, &path).unwrap_or_else(|err| {
                panic!(
                    "Failed to create ONNX Runtime symlink {} -> {}: {err}. \
                     Ensure the SDK lib directory is writable.",
                    path.display(),
                    versioned_name
                )
            });
        }
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
