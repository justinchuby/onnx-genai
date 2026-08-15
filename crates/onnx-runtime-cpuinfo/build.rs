use std::env;
use std::path::PathBuf;

/// Locate `ninja.exe` on `PATH`. Inside a Visual Studio developer environment
/// Ninja is on `PATH` (it is bundled under `CommonExtensions\Microsoft\CMake`),
/// so this succeeds exactly when a Ninja-driven build is viable.
fn find_ninja() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "ninja.exe" } else { "ninja" };
    let paths = env::var_os("PATH")?;
    env::split_paths(&paths)
        .map(|dir| dir.join(exe))
        .find(|candidate| candidate.is_file())
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpuinfo = manifest.join("vendor/cpuinfo");
    let header = manifest.join("vendor/cpuinfo/include/cpuinfo.h");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let wrappers = out_path.join("cpuinfo_static_wrappers");

    println!("cargo:rerun-if-changed=vendor/cpuinfo/CMakeLists.txt");
    println!("cargo:rerun-if-changed=vendor/cpuinfo/include/cpuinfo.h");

    // Build cpuinfo via cmake
    let mut config = cmake::Config::new(&cpuinfo);
    config
        .define("CPUINFO_BUILD_TOOLS", "OFF")
        .define("CPUINFO_BUILD_UNIT_TESTS", "OFF")
        .define("CPUINFO_BUILD_MOCK_TESTS", "OFF")
        .define("CPUINFO_BUILD_BENCHMARKS", "OFF")
        .define("CPUINFO_BUILD_PKG_CONFIG", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_INSTALL_LIBDIR", "lib");

    // On Windows/MSVC the `cmake` crate derives the Visual Studio generator from
    // the active toolchain's `VisualStudioVersion`. A stale or mismatched value
    // (e.g. a leftover VS preview env) makes it request a generator string the
    // installed CMake can't create ("Could not create named generator ..."). When
    // the caller hasn't explicitly pinned `CMAKE_GENERATOR`, prefer Ninja: it ships
    // with Visual Studio and is independent of the VS generator version scheme.
    let target_windows_msvc = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    let ninja = if target_windows_msvc && env::var_os("CMAKE_GENERATOR").is_none() {
        find_ninja()
    } else {
        None
    };
    if let Some(ninja) = ninja {
        config.generator("Ninja");
        config.define("CMAKE_MAKE_PROGRAM", ninja.to_string_lossy().as_ref());
    }

    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=cpuinfo");

    // Generate Rust bindings
    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy().into_owned())
        .rust_target(
            bindgen::RustTarget::stable(85, 0)
                .unwrap_or_else(|_| panic!("bindgen must support Rust 1.85")),
        )
        .rust_edition(bindgen::RustEdition::Edition2024)
        .wrap_static_fns(true)
        .wrap_static_fns_path(&wrappers)
        .allowlist_function("cpuinfo_.*")
        .allowlist_type("cpuinfo_.*")
        .allowlist_var("cpuinfo_.*")
        .generate()
        .expect("Unable to generate cpuinfo bindings");

    bindings
        .write_to_file(out_path.join("cpuinfo_bindings.rs"))
        .expect("Couldn't write bindings");

    cc::Build::new()
        .file(wrappers.with_extension("c"))
        .include(cpuinfo.join("include"))
        .compile("cpuinfo_static_wrappers");
}
