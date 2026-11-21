//! Pinned PyPI native artifacts. Only native libraries are extracted: the
//! Python extension is neither imported nor linked by the Rust runtime.

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy)]
pub struct Artifact {
    pub filename: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
}

pub fn artifact_for(os: &str, arch: &str) -> Result<Artifact, String> {
    Ok(match (os, arch) {
        ("linux", "x86_64") => Artifact {
            filename: "onnxruntime-1.32.0-cp311-cp311-manylinux_2_28_x86_64.whl",
            url: "https://files.pythonhosted.org/packages/1d/46/e6f356c9539d2eed44126de3421a81e217da2ad91b22dc11b379cb1e384f/onnxruntime-1.32.0-cp311-cp311-manylinux_2_28_x86_64.whl",
            sha256: "5a34fc4dfeb5a7b4a74c95dcba1e773d722abda720b474edf1b0c419dc6d163d",
        },
        ("linux", "aarch64") => Artifact {
            filename: "onnxruntime-1.32.0-cp311-cp311-manylinux_2_28_aarch64.whl",
            url: "https://files.pythonhosted.org/packages/b7/30/103820b32e9dc1d1fcfc5b519fd2de5bd699d25a25895caf5d726350d1b5/onnxruntime-1.32.0-cp311-cp311-manylinux_2_28_aarch64.whl",
            sha256: "2303372a85299b6757b0e11e1cbf13e410ddf7e10c37f4b37e99d1c90e7daf7c",
        },
        ("macos", "aarch64") => Artifact {
            filename: "onnxruntime-1.32.0-cp311-cp311-macosx_14_0_arm64.whl",
            url: "https://files.pythonhosted.org/packages/b7/5d/50be370b540a6cb9ea173503f5418e8315dfe5757400ad7720542d348b94/onnxruntime-1.32.0-cp311-cp311-macosx_14_0_arm64.whl",
            sha256: "366b9e53d07812cc5c92822cecf7563b76a8a0f32ba3bb8dd232a206ac3bcb6c",
        },
        ("windows", "x86_64") => Artifact {
            filename: "onnxruntime-1.32.0-cp311-cp311-win_amd64.whl",
            url: "https://files.pythonhosted.org/packages/07/ff/c797464da311c580ec477077a5c65d7841599a46c35392b5978c0a5ae9a4/onnxruntime-1.32.0-cp311-cp311-win_amd64.whl",
            sha256: "d7e0c45cc5f456ad7805e6dd256a7662736cda9c2fd17becae194eb3c2f6df72",
        },
        ("windows", "aarch64") => Artifact {
            filename: "onnxruntime-1.32.0-cp311-cp311-win_arm64.whl",
            url: "https://files.pythonhosted.org/packages/bd/ff/e14181a5315745007f06612d963c6f739b371452b089b5b8109c2379bd32/onnxruntime-1.32.0-cp311-cp311-win_arm64.whl",
            sha256: "23361cdf98afac352d10aa7d9e0a10ceaeb02d38430fb693c6eff370196db62d",
        },
        _ => {
            return Err(format!(
                "no pinned ONNX Runtime PyPI wheel for target {os}/{arch}; supported targets are Linux x86_64/aarch64, macOS aarch64 (14+), and Windows x86_64/aarch64. Set ORT_ROOT to a matching local ONNX Runtime install to build this target"
            ));
        }
    })
}

pub fn verify_checksum(path: &Path, artifact: &Artifact) -> Result<(), String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("cannot read ONNX Runtime wheel {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| format!("cannot hash ONNX Runtime wheel {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != artifact.sha256 {
        return Err(format!(
            "SHA-256 mismatch for ONNX Runtime wheel {}: expected {}, got {actual}; refusing extraction. Remove {} and retry the pinned PyPI download",
            artifact.filename,
            artifact.sha256,
            path.display()
        ));
    }
    Ok(())
}

pub fn extract_native_libraries(
    wheel: &Path,
    lib_dir: &Path,
    required: &str,
) -> Result<(), String> {
    let file = std::fs::File::open(wheel)
        .map_err(|e| format!("cannot open ONNX Runtime wheel {}: {e}", wheel.display()))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| {
        format!(
            "cannot decode ONNX Runtime wheel {} as ZIP: {e}; remove it and retry",
            wheel.display()
        )
    })?;
    // Validate every entry before writing anything; reject path traversal even
    // in an entry that would otherwise be ignored.
    let mut libraries = Vec::new();
    for index in 0..archive.len() {
        let entry = archive
            .by_index(index)
            .map_err(|e| format!("cannot inspect entry {index} in {}: {e}", wheel.display()))?;
        let path = entry.enclosed_name().ok_or_else(|| {
            format!(
                "unsafe path {:?} in ONNX Runtime wheel {}; refusing extraction",
                entry.name(),
                wheel.display()
            )
        })?;
        if entry.is_dir() || path.parent() != Some(Path::new("onnxruntime/capi")) {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if (name.starts_with("libonnxruntime")
            && (name.ends_with(".so") || name.contains(".so.") || name.ends_with(".dylib")))
            || (name.starts_with("onnxruntime") && name.ends_with(".dll"))
        {
            libraries.push((index, name.to_owned()));
        }
    }
    if !libraries.iter().any(|(_, name)| name == required) {
        return Err(format!(
            "ONNX Runtime wheel {} does not contain onnxruntime/capi/{required}; choose the pinned artifact for the target or set ORT_ROOT",
            wheel.display()
        ));
    }
    std::fs::create_dir_all(lib_dir).map_err(|e| {
        format!(
            "cannot create ONNX Runtime library directory {}: {e}",
            lib_dir.display()
        )
    })?;
    for (index, name) in libraries {
        let path = lib_dir.join(name);
        let mut entry = archive
            .by_index(index)
            .map_err(|e| format!("cannot read entry {index} in {}: {e}", wheel.display()))?;
        let mut output = std::fs::File::create(&path)
            .map_err(|e| format!("cannot create ONNX Runtime library {}: {e}", path.display()))?;
        std::io::copy(&mut entry, &mut output).map_err(|e| {
            format!(
                "cannot extract ONNX Runtime library {}: {e}",
                path.display()
            )
        })?;
    }
    Ok(())
}
