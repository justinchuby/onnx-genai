use std::io::Write;
use std::path::Path;

use zip::write::SimpleFileOptions;

#[path = "../build_support.rs"]
mod build_support;

fn write_sdk(path: &Path, entries: &[(&str, &[u8])]) {
    let mut writer = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    for (name, contents) in entries {
        writer
            .start_file(*name, SimpleFileOptions::default())
            .unwrap();
        writer.write_all(contents).unwrap();
    }
    writer.finish().unwrap();
}

fn scratch() -> tempfile::TempDir {
    tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap()
}

#[test]
fn installs_only_matching_native_sdk_with_versioned_runtime_names() {
    let temp = scratch();
    let archive = temp.path().join("sdk.nupkg");
    write_sdk(
        &archive,
        &[
            (
                "build/native/include/onnxruntime_c_api.h",
                b"#define ORT_API_VERSION 29\n",
            ),
            ("build/native/include/onnxruntime_ep_c_api.h", b"EP API"),
            ("runtimes/linux-x64/native/libonnxruntime.so", b"linux-x64"),
            (
                "runtimes/linux-arm64/native/libonnxruntime.so",
                b"linux-arm64",
            ),
            (
                "runtimes/osx-arm64/native/libonnxruntime.dylib",
                b"osx-arm64",
            ),
            ("runtimes/win-x64/native/onnxruntime.dll", b"win-x64"),
            ("runtimes/win-x64/native/onnxruntime.lib", b"import library"),
            ("runtimes/win-arm64/native/onnxruntime.dll", b"win-arm64"),
            ("runtimes/ios/native/onnxruntime.xcframework.zip", b"unused"),
            ("lib/net8.0/OnnxRuntime.dll", b"managed, not native"),
        ],
    );
    for (os, arch, env, rid, filename) in [
        (
            "linux",
            "x86_64",
            "gnu",
            "linux-x64",
            "libonnxruntime.so.1.29.0",
        ),
        (
            "linux",
            "aarch64",
            "gnu",
            "linux-arm64",
            "libonnxruntime.so.1.29.0",
        ),
        (
            "macos",
            "aarch64",
            "",
            "osx-arm64",
            "libonnxruntime.1.29.0.dylib",
        ),
        ("windows", "x86_64", "msvc", "win-x64", "onnxruntime.dll"),
        ("windows", "aarch64", "msvc", "win-arm64", "onnxruntime.dll"),
    ] {
        let target = build_support::sdk_target(os, arch, env).unwrap();
        let dest = temp.path().join(rid);
        build_support::extract_sdk(&archive, &dest, &target, "1.29.0", "29").unwrap();
        assert_eq!(
            std::fs::read(dest.join("lib").join(filename)).unwrap(),
            rid.as_bytes()
        );
        assert_eq!(
            std::fs::read(dest.join("include/onnxruntime_ep_c_api.h")).unwrap(),
            b"EP API"
        );
        assert!(!dest.join("runtimes").exists());
        assert!(!dest.join("lib/net8.0").exists());
        if rid == "win-x64" {
            assert_eq!(
                std::fs::read(dest.join("lib/onnxruntime.lib")).unwrap(),
                b"import library"
            );
        }
    }
}

#[test]
fn rejects_targets_without_a_matching_published_binary_instead_of_guessing() {
    for (os, arch, env) in [
        ("linux", "riscv64", "gnu"),
        ("linux", "x86_64", "musl"),
        ("macos", "x86_64", ""),
        ("windows", "x86_64", "gnu"),
    ] {
        let err = build_support::sdk_target(os, arch, env)
            .err()
            .expect("unsupported target must fail");
        assert!(err.contains(&format!("arch={arch}")));
        assert!(err.contains(&format!("env={env}")));
        assert!(err.contains("Set ORT_ROOT"));
    }
}

#[test]
fn rejects_wrong_api_headers_and_missing_target_runtime() {
    let temp = scratch();
    let archive = temp.path().join("sdk.nupkg");
    let target = build_support::sdk_target("linux", "aarch64", "gnu").unwrap();
    for api in ["28", "290"] {
        let header = format!("#define ORT_API_VERSION {api}\n");
        write_sdk(
            &archive,
            &[
                (
                    "build/native/include/onnxruntime_c_api.h",
                    header.as_bytes(),
                ),
                ("runtimes/linux-arm64/native/libonnxruntime.so", b"runtime"),
            ],
        );
        let err =
            build_support::extract_sdk(&archive, &temp.path().join(api), &target, "1.29.0", "29")
                .unwrap_err();
        assert!(err.contains("ORT_API_VERSION 29"), "{err}");
        assert!(
            err.contains("refusing to mix headers and runtime ABI"),
            "{err}"
        );
    }
    write_sdk(
        &archive,
        &[(
            "build/native/include/onnxruntime_c_api.h",
            b"#define ORT_API_VERSION 29\n",
        )],
    );
    let err = build_support::extract_sdk(
        &archive,
        &temp.path().join("missing"),
        &target,
        "1.29.0",
        "29",
    )
    .unwrap_err();
    assert!(err.contains("missing runtime"), "{err}");
    assert!(err.contains("linux-arm64"), "{err}");
}

#[test]
fn rejects_paths_that_escape_the_selected_sdk_subtree() {
    let temp = scratch();
    let archive = temp.path().join("sdk.nupkg");
    let target = build_support::sdk_target("windows", "aarch64", "msvc").unwrap();
    for (index, name) in [
        "build/native/include/../../escape",
        "runtimes/win-arm64/native/../../escape",
        "runtimes/win-arm64/native/C:\\escape",
    ]
    .iter()
    .enumerate()
    {
        write_sdk(&archive, &[(name, b"unsafe")]);
        let err = build_support::extract_sdk(
            &archive,
            &temp.path().join(index.to_string()),
            &target,
            "1.29.0",
            "29",
        )
        .unwrap_err();
        assert!(err.contains("Unsafe SDK entry"), "{err}");
        assert!(!temp.path().join("escape").exists());
    }
}
