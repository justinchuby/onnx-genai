use std::path::{Component, Path};

pub struct SdkTarget {
    pub rid: &'static str,
    library: &'static str,
}

impl SdkTarget {
    fn versioned_library(&self, version: &str) -> String {
        if self.library.ends_with(".so") {
            format!("{}.{version}", self.library)
        } else if let Some(stem) = self.library.strip_suffix(".dylib") {
            format!("{stem}.{version}.dylib")
        } else {
            self.library.to_string()
        }
    }
}

pub fn sdk_target(os: &str, arch: &str, env: &str) -> Result<SdkTarget, String> {
    let (rid, library) = match (os, arch, env) {
        ("linux", "x86_64", "gnu") => ("linux-x64", "libonnxruntime.so"),
        ("linux", "aarch64", "gnu") => ("linux-arm64", "libonnxruntime.so"),
        ("macos", "aarch64", "") => ("osx-arm64", "libonnxruntime.dylib"),
        ("windows", "x86_64", "msvc") => ("win-x64", "onnxruntime.dll"),
        ("windows", "aarch64", "msvc") => ("win-arm64", "onnxruntime.dll"),
        _ => {
            return Err(format!(
                "The pinned ONNX Runtime NuGet SDK has no runtime for \
                 os={os}, arch={arch}, env={env}. Set ORT_ROOT to a compatible \
                 local SDK with include/ and lib/ directories."
            ));
        }
    };
    Ok(SdkTarget { rid, library })
}

pub fn extract_sdk(
    archive_path: &Path,
    dest: &Path,
    target: &SdkTarget,
    version: &str,
    api_version: &str,
) -> Result<(), String> {
    let file = std::fs::File::open(archive_path).map_err(|err| err.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|err| err.to_string())?;
    let native_prefix = format!("runtimes/{}/native/", target.rid);
    let versioned_library = target.versioned_library(version);

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|err| err.to_string())?;
        let entry_name = entry.name().map_err(|err| err.to_string())?.into_owned();
        let (directory, relative) =
            if let Some(relative) = entry_name.strip_prefix("build/native/include/") {
                ("include", relative)
            } else if let Some(relative) = entry_name.strip_prefix(&native_prefix) {
                ("lib", relative)
            } else {
                continue;
            };
        if relative.is_empty() || entry.is_dir() {
            continue;
        }
        if relative.contains('\\')
            || !Path::new(relative)
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
            || entry.is_symlink()
        {
            return Err(format!(
                "Unsafe SDK entry {:?}; refusing to extract outside {} or follow symlinks.",
                entry_name,
                dest.display()
            ));
        }
        let name = if directory == "lib" && relative == target.library {
            versioned_library.as_str()
        } else {
            relative
        };
        let path = dest.join(directory).join(name);
        std::fs::create_dir_all(path.parent().expect("SDK entry has a parent"))
            .map_err(|err| format!("Cannot create parent of {}: {err}", path.display()))?;
        let mut output = std::fs::File::create(&path)
            .map_err(|err| format!("Cannot create {}: {err}", path.display()))?;
        std::io::copy(&mut entry, &mut output)
            .map_err(|err| format!("Cannot extract {}: {err}", path.display()))?;
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode))
                .map_err(|err| format!("Cannot set permissions on {}: {err}", path.display()))?;
        }
    }

    let header = dest.join("include/onnxruntime_c_api.h");
    let expected_api = format!("#define ORT_API_VERSION {api_version}");
    let contents = std::fs::read_to_string(&header)
        .map_err(|err| format!("Cannot read SDK header {}: {err}", header.display()))?;
    if !contents.lines().any(|line| line.trim() == expected_api) {
        return Err(format!(
            "SDK header {} does not declare ORT_API_VERSION {api_version}; \
             refusing to mix headers and runtime ABI.",
            header.display()
        ));
    }
    let library = dest.join("lib").join(versioned_library);
    if !library.is_file() {
        return Err(format!(
            "SDK is missing runtime {} for {}.",
            library.display(),
            target.rid
        ));
    }
    Ok(())
}
