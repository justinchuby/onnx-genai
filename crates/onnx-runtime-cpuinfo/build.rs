use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpuinfo = manifest.join("vendor/cpuinfo");
    let header = manifest.join("vendor/cpuinfo/include/cpuinfo.h");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let wrappers = out_path.join("cpuinfo_static_wrappers");

    println!("cargo:rerun-if-changed=vendor/cpuinfo/CMakeLists.txt");
    println!("cargo:rerun-if-changed=vendor/cpuinfo/include/cpuinfo.h");

    // Build cpuinfo via cmake
    let dst = cmake::Config::new(&cpuinfo)
        .define("CPUINFO_BUILD_TOOLS", "OFF")
        .define("CPUINFO_BUILD_UNIT_TESTS", "OFF")
        .define("CPUINFO_BUILD_MOCK_TESTS", "OFF")
        .define("CPUINFO_BUILD_BENCHMARKS", "OFF")
        .define("CPUINFO_BUILD_PKG_CONFIG", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        .build();

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
