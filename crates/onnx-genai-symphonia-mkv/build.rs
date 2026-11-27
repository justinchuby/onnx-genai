// Copyright (c) 2026 Justin Chu.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Stage corresponding covered source for Maturin's dependency OUT_DIR includes.
//! No source generation, network, external commands, or optional runtime loading.
use std::{env, fs, io, path::Path};

fn copy_file(source: &Path, target: &Path) -> io::Result<()> {
    println!("cargo:rerun-if-changed={}", source.display());
    let kind = fs::symlink_metadata(source).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "inspect MPL source {}: {error}; restore the canonical reader file",
                source.display()
            ),
        )
    })?;
    if !kind.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "MPL source {} must be a regular file; restore the canonical source",
                source.display()
            ),
        ));
    }
    fs::copy(source, target).map(|_| ()).map_err(|error| {
        io::Error::new(error.kind(), format!("copy MPL corresponding source {} to {}: {error}; restore the canonical reader source and ensure OUT_DIR is writable", source.display(), target.display()))
    })
}

fn copy_tree(source: &Path, target: &Path) -> io::Result<()> {
    println!("cargo:rerun-if-changed={}", source.display());
    fs::create_dir_all(target).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "create MPL source directory {}: {error}; ensure OUT_DIR is writable",
                target.display()
            ),
        )
    })?;
    let mut entries = fs::read_dir(source).and_then(|entries| entries.collect::<Result<Vec<_>, _>>()).map_err(|error| io::Error::new(error.kind(), format!("read canonical MPL source directory {}: {error}; restore the complete reader package", source.display())))?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if entry.file_name() == "__pycache__"
            || path.extension().is_some_and(|extension| extension == "pyc")
        {
            continue;
        }
        let destination = target.join(entry.file_name());
        let kind = entry.file_type().map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "inspect canonical MPL source {}: {error}; restore the reader file",
                    path.display()
                ),
            )
        })?;
        if kind.is_dir() {
            copy_tree(&path, &destination)?;
        } else if kind.is_file() {
            copy_file(&path, &destination)?;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "MPL source path {} is not a regular file/directory; restore the canonical reader source",
                    path.display()
                ),
            ));
        }
    }
    Ok(())
}

fn main() {
    let source = env::var_os("CARGO_MANIFEST_DIR").expect("Cargo supplies CARGO_MANIFEST_DIR");
    let out = env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR");
    let version = env::var("CARGO_PKG_VERSION").expect("Cargo supplies the package version");
    let destination = Path::new(&out).join(format!("onnx-genai-symphonia-mkv-{version}"));
    let result = (|| -> io::Result<()> {
        // Remove stale generated entries after a canonical source file is deleted.
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        fs::create_dir_all(&destination)?;
        let source = Path::new(&source);
        for name in [
            "Cargo.toml",
            "build.rs",
            "LICENSE",
            "NOTICE.md",
            "README.md",
            "CHANGES.md",
            "UPSTREAM.json",
            "verify_upstream.py",
            "verify_distribution.py",
        ] {
            copy_file(&source.join(name), &destination.join(name))?;
        }
        // Cargo registry archives carry both the normalized build manifest and
        // the original authored manifest; preserve each when present.
        if source.join("Cargo.toml.orig").is_file() {
            copy_file(
                &source.join("Cargo.toml.orig"),
                &destination.join("Cargo.toml.orig"),
            )?;
        }
        copy_tree(&source.join("src"), &destination.join("src"))?;
        copy_tree(&source.join("upstream"), &destination.join("upstream"))?;
        copy_tree(&source.join("tests"), &destination.join("tests"))?;
        fs::write(
            destination.join("SOURCE_VERSION"),
            format!("onnx-genai-symphonia-mkv {version}\n"),
        )?;
        Ok(())
    })();
    if let Err(error) = result {
        panic!(
            "stage MPL corresponding source under {}: {error}; restore the canonical reader package and ensure OUT_DIR is writable",
            destination.display()
        );
    }
}
