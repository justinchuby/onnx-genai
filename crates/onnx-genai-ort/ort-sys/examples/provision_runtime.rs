//! Explicit opt-in reference-runtime provisioning, never called by a build script.
#[path = "../build_support.rs"]
mod build_support;

use std::path::{Path, PathBuf};

fn main() {
    if let Err(message) = provision() {
        eprintln!("ORT reference provisioning: {message}");
        std::process::exit(1);
    }
}

fn provision() -> Result<(), String> {
    let mut arguments = std::env::args_os().skip(1);
    let destination = PathBuf::from(arguments.next().ok_or(
        "provide a new destination directory: cargo run -p onnx-genai-ort-sys --example provision_runtime -- /path/to/reference-ort",
    )?);
    if arguments.next().is_some() {
        return Err("expected exactly one destination directory".into());
    }
    let artifact = build_support::artifact_for(std::env::consts::OS, std::env::consts::ARCH)?;
    std::fs::create_dir(&destination).map_err(|error| format!(
        "create new reference directory {}: {error}; choose an unused path with an existing parent", destination.display()
    ))?;
    let download = destination.join(artifact.filename);
    let status = std::process::Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--retry",
            "3",
            "--connect-timeout",
            "30",
            "--max-time",
            "300",
            "--output",
        ])
        .arg(&download)
        .arg(artifact.url)
        .status()
        .map_err(|error| {
            format!(
                "run curl for {}: {error}; install curl and retry using a new destination",
                artifact.url
            )
        })?;
    if !status.success() {
        return Err(format!(
            "curl exited {status} downloading {}; inspect network access and retry using a new destination",
            artifact.url
        ));
    }
    build_support::verify_checksum(&download, &artifact)?;
    let runtime = if cfg!(target_os = "macos") {
        format!("libonnxruntime.{}.dylib", onnx_genai_ort_sys::ORT_VERSION)
    } else if cfg!(target_os = "windows") {
        "onnxruntime.dll".into()
    } else {
        format!("libonnxruntime.so.{}", onnx_genai_ort_sys::ORT_VERSION)
    };
    let libraries = destination.join("lib");
    build_support::extract_native_libraries(&download, &libraries, &runtime)?;
    #[cfg(unix)]
    {
        let aliases = if cfg!(target_os = "macos") {
            ["libonnxruntime.1.dylib", "libonnxruntime.dylib"]
        } else {
            ["libonnxruntime.so.1", "libonnxruntime.so"]
        };
        build_support::ensure_runtime_links(&libraries, &runtime, &aliases)?;
    }
    std::fs::remove_file(&download)
        .map_err(|error| format!("remove verified download {}: {error}", download.display()))?;
    println!(
        "Reference runtime installed at {}; select it with ONNX_GENAI_ORT_LIB_DIR={}",
        libraries.display(),
        absolute(&libraries)?.display()
    );
    Ok(())
}

fn absolute(path: &Path) -> Result<PathBuf, String> {
    path.canonicalize()
        .map_err(|error| format!("resolve installed runtime {}: {error}", path.display()))
}
