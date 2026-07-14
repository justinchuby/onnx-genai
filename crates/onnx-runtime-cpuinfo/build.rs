use std::env;
use std::path::PathBuf;

fn main() {
    // Build cpuinfo via cmake
    let dst = cmake::Config::new("../../third_party/cpuinfo")
        .define("CPUINFO_BUILD_TOOLS", "OFF")
        .define("CPUINFO_BUILD_UNIT_TESTS", "OFF")
        .define("CPUINFO_BUILD_MOCK_TESTS", "OFF")
        .define("CPUINFO_BUILD_BENCHMARKS", "OFF")
        .define("CPUINFO_BUILD_PKG_CONFIG", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=cpuinfo");
    println!("cargo:rustc-link-lib=static=clog");

    // Generate Rust bindings
    let bindings = bindgen::Builder::default()
        .header("../../third_party/cpuinfo/include/cpuinfo.h")
        .allowlist_function("cpuinfo_.*")
        .allowlist_type("cpuinfo_.*")
        .allowlist_var("cpuinfo_.*")
        .generate()
        .expect("Unable to generate cpuinfo bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("cpuinfo_bindings.rs"))
        .expect("Couldn't write bindings");
}
