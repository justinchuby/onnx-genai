// FEASIBILITY SPIKE build script.
//
// Compiles a vendored subset of ONNX Runtime's MLAS for x86-64 Linux directly
// with the `cc` crate (no cmake), grouped by instruction-set extension exactly
// as `cmake/onnxruntime_mlas.cmake` does for the `X86_64` branch. Runtime CPU
// dispatch (in platform.cpp) then picks the best kernel for the host, so on a
// Sapphire Rapids box the AVX-512 SGEMM kernel is selected automatically.
//
// The whole platform kernel set is compiled (not just the SGEMM `.S` files)
// because platform.cpp's dispatch-table constructor references symbols from
// every kernel translation unit; omitting one yields an undefined reference.
// High-level entry points that are unnecessary for SGEMM *and* drag in
// external ORT/GSL/SafeInt headers (cast.cpp, convolve.cpp, q4*.cpp) are
// excluded. MLAS's own `BUILD_MLAS_NO_ONNXRUNTIME` mode supplies a standalone
// CPUID/threading shim so no ORT runtime headers are needed.
//
// A few vendored TUs rely on system headers (<cstring>, <unistd.h>, etc.)
// being pulled in transitively by the full ORT include graph. Since we compile
// them in isolation, C++ groups force-include those headers (never applied to
// `.S` files, where it would break the assembler) to keep the vendored source
// pristine.

use std::path::PathBuf;

struct Paths {
    root: PathBuf,
    lib: PathBuf,
    asm: PathBuf,
    arm64_asm: PathBuf,
    includes: Vec<PathBuf>,
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let vendor = root.join("vendor/mlas/onnxruntime");
    let lib = vendor.join("core/mlas/lib");
    let asm = lib.join("x86_64");
    let arm64_asm = lib.join("arm64");
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    println!("cargo:rerun-if-changed=vendor/shim.cpp");
    println!("cargo:rerun-if-changed=vendor/mlas");

    let includes = vec![
        vendor.clone(),
        lib.clone(),
        asm.clone(),
        vendor.join("core/mlas/inc"),
        root.join("vendor/compat"),
    ];
    let p = Paths {
        root,
        lib,
        asm,
        arm64_asm,
        includes,
    };

    // --- C++ groups (drivers + intrinsic kernels) ---

    // Generic drivers / dispatch table. shim.cpp lives here too.
    let mut generic: Vec<PathBuf> = [
        "platform.cpp",
        "threading.cpp",
        "sgemm.cpp",
        "dgemm.cpp",
        "qgemm.cpp",
        "qgemm_kernel_default.cpp",
        "qnbitgemm.cpp",
        "qlutgemm.cpp",
        "qkv_quant.cpp",
        "rotary_embedding.cpp",
        // Scalar fallback kernels referenced by platform.cpp's dispatch table.
        "compute.cpp",
        "activate.cpp",
        "erf.cpp",
        "gelu.cpp",
        "silu.cpp",
        "tanh.cpp",
        "logistic.cpp",
        "qladd.cpp",
        "quantize.cpp",
        "dequantize.cpp",
        "convolve.cpp",
        "snchwc.cpp",
        "reorder.cpp",
        "pooling.cpp",
        "sconv_nchw_depthwise_multiplier_1.cpp",
        "sconv_nchw_depthwise_multiplier_greater_than_1.cpp",
        "convsym.cpp",
        "eltwise.cpp",
        "qdwconv.cpp",
    ]
    .iter()
    .map(|f| p.lib.join(f))
    .collect();
    if target_arch == "aarch64" {
        generic.extend(
            [
                "activate_fp16.cpp",
                "pooling_fp16.cpp",
                "halfgemm.cpp",
                "qdwconv_kernelsize.cpp",
                "halfgemm_kernel_neon.cpp",
                "qgemm_kernel_neon.cpp",
                "qgemm_kernel_udot.cpp",
                "qgemm_kernel_sdot.cpp",
                "qnbitgemm_kernel_neon.cpp",
                "sqnbitgemm_kernel_neon_fp32.cpp",
                "sqnbitgemm_kernel_neon_int8.cpp",
                "sqnbitgemm_kernel_avx512_2bit.cpp",
                "sqnbitgemm_kernel_neon_int8_2bit.cpp",
                "cast_kernel_neon.cpp",
                "hqnbitgemm_kernel_neon_fp16.cpp",
                "hqnbitgemm_kernel_neon_fp16_8bit.cpp",
                "rotary_embedding_kernel_neon.cpp",
                "rotary_embedding_kernel_neon_fp16.cpp",
                "qkv_quant_kernel_neon.cpp",
                "hgemm_kernel_neon.cpp",
                "halfgemm_kernel_neon_fp16.cpp",
                "softmax_kernel_neon.cpp",
                "softmax_kernel_neon_fp16.cpp",
                "eltwise_kernel_neon.cpp",
                "eltwise_kernel_neon_fp16.cpp",
                "sqnbitgemm_kernel_neon_int8_i8mm.cpp",
            ]
            .iter()
            .map(|f| p.lib.join(f)),
        );
    } else {
        generic.push(p.lib.join("qgemm_kernel_avx2.cpp"));
    }
    generic.push(p.root.join("vendor/shim.cpp"));
    generic.push(p.root.join("vendor/probe.cpp"));
    p.compile_cpp("mlas_generic", &[], &generic);

    if target_arch == "aarch64" {
        if target_env == "msvc" {
            p.compile_msvc_arm64_asm(
                "mlas_arm64_asm",
                &[
                    "ConvSymS8KernelDot.asm",
                    "ConvSymS8KernelDotLd64.asm",
                    "ConvSymU8KernelDot.asm",
                    "ConvSymS8KernelNeon.asm",
                    "ConvSymU8KernelNeon.asm",
                    "DepthwiseQConvSymS8KernelNeon.asm",
                    "DepthwiseQConvSymU8KernelNeon.asm",
                    "DepthwiseQConvKernelSize9Neon.asm",
                    "HalfGemmKernelNeon.asm",
                    "QgemmU8X8KernelNeon.asm",
                    "QgemmS8S8KernelNeon.asm",
                    "QgemmU8X8KernelUdot.asm",
                    "QgemmS8S8KernelSdot.asm",
                    "SgemmKernelNeon.asm",
                    "SgemvKernelNeon.asm",
                    "SymQgemmS8KernelNeon.asm",
                    "SymQgemmS8KernelSdot.asm",
                    "SymQgemmS8KernelSdotLd64.asm",
                ],
            );
        }
        if target_env != "msvc" {
            println!("cargo:rustc-link-lib=stdc++");
        }
        return;
    }

    p.compile_cpp(
        "mlas_sse2_cpp",
        &["-msse2"],
        &[p.lib.join("qgemm_kernel_sse.cpp")],
    );

    p.compile_cpp(
        "mlas_avx_cpp",
        &["-mavx"],
        &[p.lib.join("intrinsics/avx/min_max_elements.cpp")],
    );

    p.compile_cpp(
        "mlas_avx2_cpp",
        &["-mavx2", "-mfma", "-mf16c", "-mavxvnni"],
        &[
            p.lib.join("intrinsics/avx2/qladd_avx2.cpp"),
            p.lib.join("intrinsics/avx2/qdwconv_avx2.cpp"),
            p.lib.join("intrinsics/avx2/saturation_check_avx2.cpp"),
            p.lib.join("sqnbitgemm_kernel_avx2.cpp"),
            p.lib.join("sqnbitgemm_lut_kernel_avx2.cpp"),
            p.lib.join("rotary_embedding_kernel_avx2.cpp"),
            p.lib.join("qkv_quant_kernel_avx2.cpp"),
        ],
    );

    p.compile_cpp(
        "mlas_avx512f_cpp",
        &["-mavx512f"],
        &[
            p.lib.join("intrinsics/avx512/gelu_avx512f.cpp"),
            p.lib.join("intrinsics/avx512/silu_avx512f.cpp"),
            p.lib.join("intrinsics/avx512/quantize_avx512f.cpp"),
            p.lib.join(
                "intrinsics/avx512/sconv_nchw_depthwise_multiplier_greater_than_1_avx512f.cpp",
            ),
        ],
    );

    p.compile_cpp(
        "mlas_avx512core_cpp",
        &[
            "-mfma",
            "-mavx512vnni",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
        ],
        &[
            p.lib.join("sqnbitgemm_kernel_avx512.cpp"),
            p.lib.join("sqnbitgemm_kernel_avx512_2bit.cpp"),
        ],
    );

    p.compile_cpp(
        "mlas_avx512vnni_cpp",
        &[
            "-mfma",
            "-mavx512vnni",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512f",
        ],
        &[
            p.lib.join("sqnbitgemm_kernel_avx512vnni.cpp"),
            p.lib.join("qkv_quant_kernel_avx512vnni.cpp"),
            // Provides MlasFpQ4GemmDispatchAvx512 / MlasQ8Q4GemmDispatchAvx512vnni,
            // referenced by platform.cpp's (non-minimal) AVX-512 dispatch block.
            p.lib.join("q4gemm_avx512.cpp"),
        ],
    );

    p.compile_cpp(
        "mlas_amx_cpp",
        &[
            "-mavx2",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512f",
        ],
        &[p.lib.join("qgemm_kernel_amx.cpp")],
    );

    // --- assembly groups (.S, GAS/Linux) ---

    p.compile_asm(
        "mlas_sse2_asm",
        &["-msse2"],
        &[
            "DgemmKernelSse2.S",
            "SgemmKernelSse2.S",
            "SgemmTransposePackB16x4Sse2.S",
            "SconvKernelSse2.S",
            "SpoolKernelSse2.S",
            "cvtfp16a.S",
        ],
    );

    p.compile_asm(
        "mlas_avx_asm",
        &["-mavx"],
        &[
            "DgemmKernelAvx.S",
            "SgemmKernelAvx.S",
            "SgemmKernelM1Avx.S",
            "SgemmKernelM1TransposeBAvx.S",
            "SgemmTransposePackB16x4Avx.S",
            "SconvKernelAvx.S",
            "SpoolKernelAvx.S",
            "SoftmaxKernelAvx.S",
        ],
    );

    p.compile_asm(
        "mlas_avx2_asm",
        &["-mavx2", "-mfma", "-mf16c", "-mavxvnni"],
        &[
            "QgemmU8S8KernelAvx2.S",
            "QgemvU8S8KernelAvx2.S",
            "QgemmU8U8KernelAvx2.S",
            "QgemvU8S8KernelAvxVnni.S",
            "QgemmU8X8KernelAvx2.S",
            "ConvSymKernelAvx2.S",
            "DgemmKernelFma3.S",
            "SgemmKernelFma3.S",
            "SconvKernelFma3.S",
            "TransKernelFma3.S",
            "LogisticKernelFma3.S",
            "TanhKernelFma3.S",
            "ErfKernelFma3.S",
            "cvtfp16Avx.S",
        ],
    );

    p.compile_asm(
        "mlas_avx512f_asm",
        &["-mavx512f"],
        &[
            "DgemmKernelAvx512F.S",
            "SgemmKernelAvx512F.S",
            "SconvKernelAvx512F.S",
            "SoftmaxKernelAvx512F.S",
            "SpoolKernelAvx512F.S",
            "TransKernelAvx512F.S",
        ],
    );

    p.compile_asm(
        "mlas_avx512core_asm",
        &[
            "-mfma",
            "-mavx512vnni",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
        ],
        &[
            "QgemvU8S8KernelAvx512Core.S",
            "QgemvU8S8KernelAvx512Vnni.S",
            "QgemmU8X8KernelAvx512Core.S",
            "ConvSymKernelAvx512Core.S",
        ],
    );

    p.compile_asm(
        "mlas_amx_asm",
        &[
            "-mavx2",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512f",
        ],
        &["QgemmU8S8KernelAmxCommon.S", "QgemmU8S8KernelAmx.S"],
    );

    if target_env != "msvc" {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

impl Paths {
    fn base(&self) -> cc::Build {
        let mut b = cc::Build::new();
        b.cpp(true);
        b.std("c++17");
        b.define("BUILD_MLAS_NO_ONNXRUNTIME", None);
        b.define("_USE_MATH_DEFINES", None);
        // Full (non-minimal) build: keeps platform.cpp's AVX-512 kernel
        // selection block enabled (it is gated behind !ORT_MINIMAL_BUILD),
        // which is exactly the SGEMM parity we are validating.
        b.define("NDEBUG", None);
        b.opt_level(3);
        b.warnings(false);
        for inc in &self.includes {
            b.include(inc);
        }
        b
    }

    fn compile_cpp(&self, name: &str, flags: &[&str], files: &[PathBuf]) {
        let mut b = self.base();
        // Headers ORT normally supplies transitively across its include graph.
        if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
            for h in ["cstring", "cstdlib", "cstdint"] {
                b.flag(&format!("/FI{h}"));
            }
        } else {
            for h in ["cstring", "cstdlib", "cstdint", "unistd.h"] {
                b.flag("-include").flag(h);
            }
        }
        for f in flags {
            b.flag(f);
        }
        for f in files {
            assert!(f.exists(), "missing vendored source: {}", f.display());
            b.file(f);
        }
        b.compile(name);
    }

    fn compile_asm(&self, name: &str, flags: &[&str], files: &[&str]) {
        let mut b = self.base();
        for f in flags {
            b.flag(f);
        }
        for f in files {
            let path = if f.ends_with(".asm") {
                self.arm64_asm.join(f)
            } else {
                self.asm.join(f)
            };
            assert!(path.exists(), "missing vendored asm: {}", path.display());
            b.file(path);
        }
        b.compile(name);
    }

    fn compile_msvc_arm64_asm(&self, name: &str, files: &[&str]) {
        use std::process::Command;

        let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
        let compiler = self.base().get_compiler();
        let cl = compiler.path();
        let armasm64 = cl
            .parent()
            .expect("MSVC cl.exe has a parent directory")
            .join("armasm64.exe");

        let mut objects = Vec::new();
        for f in files {
            let src = self.arm64_asm.join(f);
            assert!(src.exists(), "missing vendored asm: {}", src.display());
            let stem = src
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("asm file stem");
            let preprocessed = out_dir.join(format!("{stem}.i"));
            let obj = out_dir.join(format!("{stem}.obj"));

            let mut cl_cmd = compiler.to_command();
            cl_cmd.arg("/nologo").arg("/P").arg(&src);
            cl_cmd.arg(format!("/Fi{}", preprocessed.display()));
            cl_cmd.arg("/DBUILD_MLAS_NO_ONNXRUNTIME");
            cl_cmd.arg("/D_USE_MATH_DEFINES");
            cl_cmd.arg("/DNDEBUG");
            for inc in &self.includes {
                cl_cmd.arg(format!("/I{}", inc.display()));
            }
            let status = cl_cmd.status().expect("run cl.exe to preprocess ARM64 asm");
            assert!(
                status.success(),
                "cl.exe failed to preprocess {}",
                src.display()
            );

            let mut asm_cmd = Command::new(&armasm64);
            for (key, value) in compiler.env() {
                asm_cmd.env(key, value);
            }
            asm_cmd.arg("-nologo").arg(&preprocessed).arg(&obj);
            let status = asm_cmd.status().expect("run armasm64.exe");
            assert!(
                status.success(),
                "armasm64.exe failed for {}",
                src.display()
            );

            objects.push(obj);
        }

        let mut b = self.base();
        for obj in objects {
            b.object(obj);
        }
        b.compile(name);
    }
}
