//! Build script for `onnx-runtime-ep-cpu`.
//!
//! For the **default** build this is a no-op: the crate is pure Rust and offline
//! (`docs/ORT2.md` §25.2 "Generic" backend). Only when the non-default `onednn`
//! feature is enabled do we cmake-build oneDNN from the `third_party/onednn`
//! submodule as a static CPU-only library and bindgen its C API, mirroring the
//! `onnx-runtime-cpuinfo` recipe.

fn main() {
    // Re-run only when the feature toggles or oneDNN inputs change.
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "onednn")]
    onednn::build();
}

#[cfg(feature = "onednn")]
mod onednn {
    use std::env;
    use std::path::{Path, PathBuf};

    pub fn build() {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        // third_party/onednn lives at the workspace root: crates/<crate>/../../
        let src = manifest_dir
            .join("../../third_party/onednn")
            .canonicalize()
            .unwrap_or_else(|_| manifest_dir.join("../../third_party/onednn"));

        let header = src.join("include/oneapi/dnnl/dnnl.h");
        if !src.join("CMakeLists.txt").exists() {
            panic!(
                "oneDNN source not found at {}. The `onednn` feature requires the \
                 third_party/onednn git submodule. Fetch it with:\n\
                 \n    git submodule update --init --depth 1 third_party/onednn\n\n\
                 (it is pinned to v3.9.2). Or build without --features onednn to use \
                 the default pure-Rust CPU backend.",
                src.display()
            );
        }

        println!("cargo:rerun-if-changed={}", header.display());
        println!(
            "cargo:rerun-if-changed={}",
            src.join("CMakeLists.txt").display()
        );

        // Build oneDNN as a STATIC, CPU-only library in Release. OpenMP CPU
        // runtime by default; set ONEDNN_CPU_RUNTIME=SEQ to fall back to the
        // sequential runtime if OpenMP linking is troublesome on the host.
        let cpu_runtime = env::var("ONEDNN_CPU_RUNTIME").unwrap_or_else(|_| "OMP".to_string());

        let dst = cmake::Config::new(&src)
            .profile("Release")
            .define("DNNL_LIBRARY_TYPE", "STATIC")
            .define("DNNL_BUILD_TESTS", "OFF")
            .define("DNNL_BUILD_EXAMPLES", "OFF")
            .define("DNNL_CPU_RUNTIME", &cpu_runtime)
            .define("DNNL_GPU_RUNTIME", "NONE")
            .define("BUILD_SHARED_LIBS", "OFF")
            .build();

        // oneDNN installs its static lib under lib/ or lib64/ depending on host.
        for sub in ["lib", "lib64"] {
            let dir = dst.join(sub);
            if dir.exists() {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
        }
        println!("cargo:rustc-link-lib=static=dnnl");

        // Link the C++ standard library oneDNN needs, and OpenMP when selected.
        link_cxx_stdlib();
        if cpu_runtime.eq_ignore_ascii_case("OMP") {
            link_openmp();
        }

        // The GNU C++/OpenMP runtimes (libstdc++/libgomp) do not exist under the
        // MSVC toolchain. When targeting MSVC we emit no explicit runtime libs
        // (the linker supplies them); make the human requirement explicit so a
        // Windows build of the non-default `onednn` feature is auditable.
        if target_env() == "msvc" {
            println!(
                "cargo:warning=onednn feature on MSVC: the C++ runtime (msvcprt) \
                 and OpenMP (vcomp, via cl.exe /openmp) are linked automatically, \
                 so no stdc++/gomp is emitted. Supply a oneDNN static lib built \
                 with MSVC (matching DNNL_CPU_RUNTIME). If your prebuilt oneDNN \
                 uses Intel OpenMP, set ONEDNN_OMP_LIB (e.g. libiomp5md) to link it."
            );
        }

        generate_bindings(&header, &dst);
    }

    fn generate_bindings(header: &Path, install_dir: &Path) {
        // dnnl.h includes generated headers (dnnl_config.h, dnnl_version.h) from
        // the install include dir; make both source and installed includes
        // visible to libclang.
        let installed_include = install_dir.join("include");
        let source_include = header
            .parent() // .../oneapi/dnnl
            .and_then(|p| p.parent()) // .../oneapi
            .and_then(|p| p.parent()) // .../include
            .map(Path::to_path_buf);

        let mut builder = bindgen::Builder::default()
            .header(header.to_string_lossy())
            .clang_arg(format!("-I{}", installed_include.display()))
            .allowlist_function("dnnl_sgemm")
            .allowlist_type("dnnl_status_t")
            .allowlist_type("dnnl_dim_t")
            .allowlist_var("dnnl_success")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

        if let Some(inc) = source_include {
            builder = builder.clang_arg(format!("-I{}", inc.display()));
        }

        let bindings = builder.generate().expect(
            "Unable to generate oneDNN bindings from dnnl.h. If bindgen cannot find \
             standard headers (e.g. stdbool.h), set LIBCLANG_PATH and \
             BINDGEN_EXTRA_CLANG_ARGS as documented in the crate README.",
        );

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("onednn_bindings.rs"))
            .expect("Couldn't write oneDNN bindings");
    }

    fn link_cxx_stdlib() {
        // oneDNN is C++, so its static objects need the target toolchain's C++
        // runtime linked. The correct library name is toolchain-specific, so
        // gate strictly by the *target* env/os (build.rs runs on the host, so we
        // must read CARGO_CFG_* rather than use host-based `cfg!`):
        //   - macOS (clang):    libc++     -> link "c++"
        //   - Linux/other GNU:  libstdc++  -> link "stdc++"   (unchanged)
        //   - Windows MSVC:     msvcprt/vcruntime is linked automatically by the
        //                       MSVC linker; emitting "stdc++"/"c++" would break
        //                       the link, so we emit nothing.
        match (target_os().as_str(), target_env().as_str()) {
            ("macos", _) => println!("cargo:rustc-link-lib=c++"),
            (_, "msvc") => { /* MSVC C++ runtime linked automatically */ }
            _ => println!("cargo:rustc-link-lib=stdc++"),
        }
    }

    fn link_openmp() {
        // Link the OpenMP runtime matching the target toolchain (oneDNN's OMP CPU
        // runtime needs it). An explicit ONEDNN_OMP_LIB override always wins, on
        // every target, for hosts that ship the runtime under a different name.
        if let Ok(omp) = env::var("ONEDNN_OMP_LIB") {
            println!("cargo:rustc-link-lib={omp}");
            return;
        }

        // Default names are toolchain-specific:
        //   - GNU (Linux):   libgomp -> "gomp"  (unchanged from prior behavior)
        //   - macOS clang:   libomp  -> "omp"   (LLVM/Homebrew libomp)
        //   - Windows MSVC:  vcomp (MSVC OpenMP) is linked automatically when
        //                    oneDNN is compiled with cl.exe /openmp, so emit
        //                    nothing. Set ONEDNN_OMP_LIB (e.g. libiomp5md) if a
        //                    prebuilt oneDNN needs Intel OpenMP instead.
        match (target_os().as_str(), target_env().as_str()) {
            (_, "msvc") => { /* vcomp linked automatically via /openmp */ }
            ("macos", _) => println!("cargo:rustc-link-lib=omp"),
            _ => println!("cargo:rustc-link-lib=gomp"),
        }
    }

    fn target_os() -> String {
        env::var("CARGO_CFG_TARGET_OS").unwrap_or_default()
    }

    fn target_env() -> String {
        env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default()
    }
}
