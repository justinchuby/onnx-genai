#[path = "../build_support.rs"]
mod build_support;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use build_support::{Artifact, artifact_for, extract_native_libraries, verify_checksum};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "nxrt-ort-bootstrap-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn wheel(&self, entries: &[(&str, &[u8])]) -> PathBuf {
        let path = self.0.join("fixture.whl");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        for (name, bytes) in entries {
            zip.start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(bytes).unwrap();
        }
        zip.finish().unwrap();
        path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn native_artifacts_cover_only_declared_target_pairs() {
    for (os, arch, tag) in [
        ("linux", "x86_64", "manylinux_2_28_x86_64"),
        ("linux", "aarch64", "manylinux_2_28_aarch64"),
        ("macos", "aarch64", "macosx_14_0_arm64"),
        ("windows", "x86_64", "win_amd64"),
        ("windows", "aarch64", "win_arm64"),
    ] {
        let artifact = artifact_for(os, arch).unwrap();
        assert!(artifact.filename.contains(tag));
        assert!(
            artifact
                .filename
                .starts_with(&format!("onnxruntime-{}-", onnx_genai_ort_sys::ORT_VERSION))
        );
        assert!(
            artifact
                .url
                .starts_with("https://files.pythonhosted.org/packages/")
        );
        assert!(artifact.url.ends_with(artifact.filename));
        assert_eq!(artifact.sha256.len(), 64);
        assert!(artifact.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
    }
    for (os, arch) in [
        ("macos", "x86_64"),
        ("linux", "riscv64"),
        ("windows", "x86"),
        ("freebsd", "x86_64"),
    ] {
        let err = artifact_for(os, arch).unwrap_err();
        assert!(err.contains(&format!("{os}/{arch}")));
        assert!(err.contains("ORT_ROOT"));
    }
}

#[test]
fn checksum_refuses_modified_runtime_before_extraction() {
    let fixture = Fixture::new();
    let path = fixture.0.join("runtime.whl");
    std::fs::write(&path, b"abc").unwrap();
    let artifact = Artifact {
        filename: "runtime.whl",
        url: "https://files.pythonhosted.org/runtime.whl",
        sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    };
    verify_checksum(&path, &artifact).unwrap();
    std::fs::write(&path, b"changed").unwrap();
    let error = verify_checksum(&path, &artifact).unwrap_err();
    assert!(error.contains("runtime.whl"));
    assert!(error.contains("expected ba7816"));
    assert!(error.contains("refusing extraction"));
    assert!(error.contains("retry"));
    std::fs::remove_file(&path).unwrap();
    let error = verify_checksum(&path, &artifact).unwrap_err();
    assert!(error.contains(&path.display().to_string()));
}

#[test]
fn wheel_extracts_native_runtime_without_python_extension() {
    for required in [
        "libonnxruntime.1.32.0.dylib",
        "libonnxruntime.so.1.32.0",
        "onnxruntime.dll",
    ] {
        let fixture = Fixture::new();
        let name = format!("onnxruntime/capi/{required}");
        let wheel = fixture.wheel(&[
            (&name, b"native runtime"),
            (
                "onnxruntime/capi/libonnxruntime_providers_shared.so",
                b"provider",
            ),
            (
                "onnxruntime/capi/onnxruntime_pybind11_state.so",
                b"python extension",
            ),
            ("onnxruntime/__init__.py", b"must not execute"),
        ]);
        let lib = fixture.0.join("lib");
        extract_native_libraries(&wheel, &lib, required).unwrap();
        assert_eq!(
            std::fs::read(lib.join(required)).unwrap(),
            b"native runtime"
        );
        assert_eq!(
            std::fs::read(lib.join("libonnxruntime_providers_shared.so")).unwrap(),
            b"provider"
        );
        assert_eq!(std::fs::read_dir(&lib).unwrap().count(), 2);
    }
}

#[test]
fn missing_required_runtime_does_not_publish_partial_extraction() {
    let fixture = Fixture::new();
    let wheel = fixture.wheel(&[(
        "onnxruntime/capi/libonnxruntime_providers_shared.so",
        b"provider",
    )]);
    let lib = fixture.0.join("lib");
    let error = extract_native_libraries(&wheel, &lib, "onnxruntime.dll").unwrap_err();
    assert!(error.contains("onnxruntime/capi/onnxruntime.dll"));
    assert!(error.contains("ORT_ROOT"));
    assert!(!lib.exists());
}

#[test]
fn unsafe_archive_path_is_rejected_before_any_write() {
    let fixture = Fixture::new();
    let wheel = fixture.wheel(&[
        ("onnxruntime/capi/onnxruntime.dll", b"runtime"),
        ("../escaped", b"unsafe"),
    ]);
    let lib = fixture.0.join("lib");
    let error = extract_native_libraries(&wheel, &lib, "onnxruntime.dll").unwrap_err();
    assert!(error.contains("unsafe path"));
    assert!(error.contains("../escaped"));
    assert!(!lib.exists());
}

#[test]
fn corrupt_zip_names_the_download_and_remedy() {
    let fixture = Fixture::new();
    let path = fixture.0.join("bad.whl");
    std::fs::write(&path, b"<html>upstream unavailable</html>").unwrap();
    let error =
        extract_native_libraries(&path, &fixture.0.join("lib"), "onnxruntime.dll").unwrap_err();
    assert!(error.contains("bad.whl"));
    assert!(error.contains("ZIP"));
    assert!(error.contains("retry"));
}

#[test]
fn bundled_headers_match_the_runtime_api() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("third_party/onnxruntime/include");
    let header = std::fs::read_to_string(root.join("onnxruntime_c_api.h")).unwrap();
    assert!(header.lines().any(|line| line.trim()
        == format!(
            "#define ORT_API_VERSION {}",
            onnx_genai_ort_sys::ORT_API_VERSION
        )));
    assert_eq!(onnx_genai_ort_sys::ORT_API_VERSION, 32);
    assert!(root.join("onnxruntime_ep_c_api.h").is_file());
    assert!(root.join("onnxruntime_error_code.h").is_file());
}
