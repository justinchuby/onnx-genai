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
    amd64_asm: PathBuf,
    arm64_asm: PathBuf,
    aarch64_asm: PathBuf,
    kai: PathBuf,
    includes: Vec<PathBuf>,
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let vendor = root.join("vendor/mlas/onnxruntime");
    let lib = vendor.join("core/mlas/lib");
    let asm = lib.join("x86_64");
    let amd64_asm = lib.join("amd64");
    let arm64_asm = lib.join("arm64");
    let aarch64_asm = lib.join("aarch64");
    let kai = root.join("vendor/mlas/kleidiai");
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    let is_apple_target = target_vendor == "apple" || matches!(target_os.as_str(), "macos" | "ios");

    println!("cargo:rerun-if-changed=vendor/shim.cpp");
    println!("cargo:rerun-if-changed=vendor/mlas");

    // The vendored sources cover x86-64 and aarch64 and nothing else. Anywhere
    // else the code below would push `-msse2` and an AVX2 translation unit at a
    // compiler that has never heard of either, and the build would die a few
    // hundred lines later inside someone else's C++.
    //
    // This matters more than it used to: `mlas` is a *default* feature of
    // `onnx-runtime-ep-cpu`, so an unsupported host now meets this on a plain
    // `cargo build` rather than only when it asked for MLAS. Failing loudly with
    // the opt-out in the message is the point -- quietly compiling a pure-Rust
    // EP instead would hand the user a library an order of magnitude slower on
    // quantized matmul without telling them.
    if target_arch != "x86_64" && target_arch != "aarch64" {
        panic!(
            "mlas-sys has no vendored kernels for target architecture '{target_arch}' \
             (supported: x86_64, aarch64).\n\
             MLAS is on by default. Build without it:\n  \
             cargo build -p onnx-runtime-ep-cpu --no-default-features --features full\n  \
             cargo build -p onnx-runtime-ep-cpu-plugin --no-default-features\n\
             For the Python wheel, set NXRT_EP_CPU_NO_MLAS=1."
        );
    }

    let includes = vec![
        vendor.clone(),
        lib.clone(),
        asm.clone(),
        vendor.join("core/mlas/inc"),
        kai.clone(),
        root.join("vendor/compat"),
    ];
    let p = Paths {
        root,
        lib,
        asm,
        amd64_asm,
        arm64_asm,
        aarch64_asm,
        kai,
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
    // fp16 NEON kernels need armv8.2-a+fp16, which the generic group does not
    // ask for. Collected here and compiled as their own group below.
    let mut arm64_fp16_sources: Vec<PathBuf> = Vec::new();
    if target_arch == "aarch64" {
        let arm64_sources = vec![
            "halfgemm.cpp",
            "qdwconv_kernelsize.cpp",
            "halfgemm_kernel_neon.cpp",
            "qgemm_kernel_neon.cpp",
            "qgemm_kernel_udot.cpp",
            "qgemm_kernel_sdot.cpp",
            "sqnbitgemm_kernel_neon_fp32.cpp",
            "sqnbitgemm_kernel_avx512_2bit.cpp",
            "cast_kernel_neon.cpp",
            "rotary_embedding_kernel_neon.cpp",
            "qkv_quant_kernel_neon.cpp",
            "hgemm_kernel_neon.cpp",
            "softmax_kernel_neon.cpp",
            "eltwise_kernel_neon.cpp",
        ];
        // Upstream MLAS intentionally leaves fp16 NEON helpers disabled on
        // Apple ARM64, so these TUs reference undeclared MlasLoad/StoreFloat16x8.
        if !is_apple_target {
            arm64_fp16_sources = [
                "activate_fp16.cpp",
                "pooling_fp16.cpp",
                "hqnbitgemm_kernel_neon_fp16.cpp",
                "hqnbitgemm_kernel_neon_fp16_8bit.cpp",
                "rotary_embedding_kernel_neon_fp16.cpp",
                "halfgemm_kernel_neon_fp16.cpp",
                "softmax_kernel_neon_fp16.cpp",
                "eltwise_kernel_neon_fp16.cpp",
                // `platform.cpp` installs these into MLAS_PLATFORM
                // unconditionally under MLAS_F16VEC_INTRINSICS_SUPPORTED, so
                // omitting them breaks the link, not just the fp16 kernels.
                "erf_neon_fp16.cpp",
                "gelu_neon_fp16.cpp",
            ]
            .iter()
            .map(|f| p.lib.join(f))
            .collect();
        }
        generic.extend(arm64_sources.iter().map(|f| p.lib.join(f)));
    } else {
        generic.push(p.lib.join("qgemm_kernel_avx2.cpp"));
    }
    generic.push(p.root.join("vendor/shim.cpp"));
    generic.push(p.root.join("vendor/probe.cpp"));
    p.compile_cpp("mlas_generic", &[], &generic);

    if target_arch == "aarch64" {
        // `vaddq_f16` and friends are `always_inline` and refuse to inline into
        // a translation unit compiled without the fp16 target feature, so these
        // sources fail with "target specific option mismatch" rather than
        // degrading. MSVC's ARM64 compiler exposes the fp16 intrinsics
        // unconditionally and has no equivalent switch, so it keeps the generic
        // flags. This is what makes `aarch64-unknown-linux-gnu` build at all:
        // the cross-compile gate in `scripts/check_cross_compile.sh` reaches
        // this crate now that `mlas` is a default feature.
        let fp16_flags: &[&str] = if target_env == "msvc" {
            &[]
        } else {
            &["-march=armv8.2-a+fp16"]
        };
        if !arm64_fp16_sources.is_empty() {
            p.compile_cpp("mlas_arm64_fp16_cpp", fp16_flags, &arm64_fp16_sources);
        }

        let dot_flags: &[&str] = if target_env == "msvc" {
            &[]
        } else {
            &["-march=armv8.2-a+dotprod"]
        };
        let i8mm_flags: &[&str] = if target_env == "msvc" {
            &[]
        } else {
            &["-march=armv8.2-a+i8mm"]
        };

        p.compile_cpp_with_defines(
            "mlas_arm64_qnbit_kleidiai_cpp",
            &[("USE_KLEIDIAI", None)],
            dot_flags,
            &[
                p.lib.join("qnbitgemm_kernel_neon.cpp"),
                p.lib.join("sqnbitgemm_kernel_neon_int8.cpp"),
                p.root.join("vendor/kai_qnbit_interface.cpp"),
            ],
        );

        p.compile_cpp(
            "mlas_arm64_qnbit_dotprod_cpp",
            dot_flags,
            &[p.lib.join("sqnbitgemm_kernel_neon_int8_2bit.cpp")],
        );

        let mut i8mm_sources = vec![p.lib.join("sqnbitgemm_kernel_neon_int8_i8mm.cpp")];
        // `platform.cpp:780` wires the Ummla/Smmla integer GEMM dispatches in
        // under `#if defined(__linux__)`, so those two TUs are link-required
        // there and dead everywhere else. Matching the vendor's own guard keeps
        // Apple and MSVC ARM64 from compiling code their platform never calls.
        if target_os == "linux" {
            i8mm_sources.push(p.lib.join("qgemm_kernel_ummla.cpp"));
            i8mm_sources.push(p.lib.join("qgemm_kernel_smmla.cpp"));
        }
        p.compile_cpp("mlas_arm64_qnbit_i8mm_cpp", i8mm_flags, &i8mm_sources);

        p.compile_kleidiai_qnbit_c();

        if target_env == "msvc" {
            p.compile_msvc_kleidiai_asm(
                "kleidiai_qnbit_asm",
                &[
                    "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod_asm.S",
                    "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod_asm.S",
                    "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod_asm.S",
                    "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm_asm.S",
                ],
            );
        } else {
            p.compile_kleidiai_asm(
                "kleidiai_qnbit_asm",
                &[
                    "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod_asm.S",
                    "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod_asm.S",
                    "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod_asm.S",
                    "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm_asm.S",
                ],
            );
        }

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
        } else {
            // The GAS counterparts of the MASM list above. Without these the
            // ARM64 link fails on `MlasConvSym*Kernel*`, `MlasSgemmKernel*`
            // and friends: `convsym.cpp` builds dispatch tables of function
            // pointers to them, so the C++ compiles and only the *link* breaks.
            // `scripts/check_cross_compile.sh` runs `cargo clippy`, which never
            // links, which is why this survived until MLAS became a default
            // feature and something actually tried to produce an ARM64 binary.
            //
            // Split by the assembler extension each file needs. Every group was
            // verified to assemble with `aarch64-linux-gnu-gcc`; the four
            // failing without a flag fail loudly ("selected processor does not
            // support ...") rather than silently emitting nothing.
            p.compile_aarch64_asm(
                "mlas_aarch64_asm",
                &[],
                &[
                    "ConvSymS8KernelDot.S",
                    "ConvSymS8KernelDotLd64.S",
                    "ConvSymS8KernelNeon.S",
                    "ConvSymU8KernelDot.S",
                    "ConvSymU8KernelNeon.S",
                    "DepthwiseQConvKernelSize9Neon.S",
                    "DepthwiseQConvSymS8KernelNeon.S",
                    "DepthwiseQConvSymU8KernelNeon.S",
                    "QgemmS8S8KernelNeon.S",
                    "QgemmS8S8KernelSdot.S",
                    "QgemmU8X8KernelNeon.S",
                    "QgemmU8X8KernelUdot.S",
                    "SconvDepthwiseKernelNeon.S",
                    "SconvKernelNeon.S",
                    "SconvKernelNeonBf16.S",
                    "SconvNchwcKernelNeon.S",
                    "SconvPointwiseKernelNeon.S",
                    "SconvPointwiseKernelNeonBf16.S",
                    "SgemmKernelNeon.S",
                    "SgemvKernelNeon.S",
                    "SymQgemmS8KernelNeon.S",
                    "SymQgemmS8KernelSdot.S",
                    "SymQgemmS8KernelSdotLd64.S",
                ],
            );
            p.compile_aarch64_asm(
                "mlas_aarch64_asm_fp16",
                &["-march=armv8.2-a+fp16"],
                &["HalfGemmKernelNeon.S"],
            );
            // The i8mm and bf16 microkernels pair with the `__linux__`-only
            // dispatches above. Verified on `aarch64-unknown-linux-gnu`; not
            // enabled for Apple, where no runner exists here to prove the
            // Mach-O assembler accepts them and where nothing references them.
            if target_os == "linux" {
                p.compile_aarch64_asm(
                    "mlas_aarch64_asm_i8mm",
                    &["-march=armv8.2-a+i8mm"],
                    &["QgemmS8S8KernelSmmla.S", "QgemmU8X8KernelUmmla.S"],
                );
                p.compile_aarch64_asm(
                    "mlas_aarch64_asm_bf16",
                    &["-march=armv8.2-a+bf16"],
                    &["SbgemmKernelNeon.S", "SconvDepthwiseKernelNeonBf16.S"],
                );
            }
        }
        emit_cpp_stdlib_link(&target_env, is_apple_target);
        return;
    }

    p.compile_cpp(
        "mlas_sse2_cpp",
        &["-msse2"],
        &[p.lib.join("qgemm_kernel_sse.cpp")],
    );

    p.compile_cpp(
        "mlas_sse41_cpp",
        &["-msse4.1"],
        &[p.lib.join("qgemm_kernel_sse41.cpp")],
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

    emit_cpp_stdlib_link(&target_env, is_apple_target);
}

fn emit_cpp_stdlib_link(target_env: &str, is_apple_target: bool) {
    if target_env == "msvc" {
        return;
    }
    if is_apple_target {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
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
        Self::add_forced_includes(&mut b);
        self.add_flags(&mut b, flags);
        for f in files {
            assert!(f.exists(), "missing vendored source: {}", f.display());
            b.file(f);
        }
        b.compile(name);
    }

    fn compile_cpp_with_defines(
        &self,
        name: &str,
        defines: &[(&str, Option<&str>)],
        flags: &[&str],
        files: &[PathBuf],
    ) {
        let mut b = self.base();
        Self::add_forced_includes(&mut b);
        for (key, value) in defines {
            b.define(key, *value);
        }
        self.add_flags(&mut b, flags);
        for f in files {
            assert!(f.exists(), "missing vendored source: {}", f.display());
            b.file(f);
        }
        b.compile(name);
    }

    fn add_forced_includes(b: &mut cc::Build) {
        // Headers ORT normally supplies transitively across its include graph.
        if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
            for h in ["cstring", "cstdlib", "cstdint"] {
                b.flag(format!("/FI{h}"));
            }
        } else {
            for h in ["cstring", "cstdlib", "cstdint", "unistd.h"] {
                b.flag("-include").flag(h);
            }
        }
    }

    fn add_flags(&self, b: &mut cc::Build, flags: &[&str]) {
        if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
            let mut arch_level = 0u8;
            for f in flags {
                match *f {
                    "-msse2" | "-msse4.1" => {}
                    "-mavx" => arch_level = arch_level.max(1),
                    "-mavx2" | "-mfma" | "-mf16c" | "-mavxvnni" => arch_level = arch_level.max(2),
                    "-mavx512f" | "-mavx512vnni" | "-mavx512bw" | "-mavx512dq" | "-mavx512vl" => {
                        arch_level = arch_level.max(3)
                    }
                    other => {
                        b.flag(other);
                    }
                }
            }
            match arch_level {
                1 => {
                    b.flag("/arch:AVX");
                }
                2 => {
                    b.flag("/arch:AVX2");
                }
                3 => {
                    b.flag("/arch:AVX512");
                }
                _ => {}
            }
        } else {
            for f in flags {
                b.flag(f);
            }
        }
    }

    fn compile_kleidiai_qnbit_c(&self) {
        let mut b = cc::Build::new();
        b.include(&self.kai);
        b.opt_level(3);
        b.warnings(false);
        for f in [
            "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c",
            "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.c",
            "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.c",
            "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c",
            "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod.c",
            "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.c",
        ] {
            let path = self.kai.join(f);
            assert!(path.exists(), "missing KleidiAI source: {}", path.display());
            b.file(path);
        }
        b.compile("kleidiai_qnbit_c");
    }

    fn compile_kleidiai_asm(&self, name: &str, files: &[&str]) {
        let mut b = cc::Build::new();
        b.include(&self.kai);
        b.opt_level(3);
        b.warnings(false);
        for f in files {
            let path = self.kai.join(f);
            assert!(path.exists(), "missing KleidiAI asm: {}", path.display());
            b.file(path);
        }
        b.compile(name);
    }

    fn compile_asm(&self, name: &str, flags: &[&str], files: &[&str]) {
        if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
            && std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64")
        {
            self.compile_msvc_amd64_asm(name, files);
            return;
        }

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

    /// Assemble GAS `.S` sources from `lib/aarch64` for non-MSVC ARM64 targets.
    ///
    /// Separate from [`Self::compile_asm`] because that one resolves `.S` under
    /// `lib/x86_64` and `.asm` under `lib/arm64`, neither of which is where
    /// these live.
    fn compile_aarch64_asm(&self, name: &str, flags: &[&str], files: &[&str]) {
        let mut b = self.base();
        b.include(&self.aarch64_asm);
        for f in flags {
            b.flag(f);
        }
        for f in files {
            let path = self.aarch64_asm.join(f);
            assert!(
                path.exists(),
                "missing vendored aarch64 asm: {}",
                path.display()
            );
            b.file(path);
        }
        b.compile(name);
    }

    fn compile_msvc_amd64_asm(&self, name: &str, files: &[&str]) {
        use std::collections::BTreeSet;
        use std::process::Command;

        let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
        let compiler = self.base().get_compiler();
        let cl = compiler.path();
        let ml64 = cl
            .parent()
            .expect("MSVC cl.exe has a parent directory")
            .join("ml64.exe");

        let mut mapped = BTreeSet::new();
        for f in files {
            let stem = f.trim_end_matches(".S").trim_end_matches(".asm");
            let asm = match stem {
                // Windows x64 MLAS keeps the SGEMM transpose pack routines in
                // the aggregate MASM source rather than a one-file-per-kernel
                // translation unit like the ELF/GAS tree.
                "SgemmTransposePackB16x4Sse2" => Some("sgemma.asm".to_string()),
                "SgemmTransposePackB16x4Avx" => Some("sgemma.asm".to_string()),
                "SgemmKernelM1TransposeBAvx" => Some("SgemmKernelM1Avx.asm".to_string()),
                // The MASM AMX file includes the common code directly.
                "QgemmU8S8KernelAmxCommon" => None,
                other => Some(format!("{other}.asm")),
            };
            if let Some(asm) = asm {
                mapped.insert(asm);
            }
        }

        let mut objects = Vec::new();
        for f in mapped {
            let src = self.amd64_asm.join(&f);
            assert!(
                src.exists(),
                "missing vendored amd64 asm: {}",
                src.display()
            );
            let stem = src
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("asm file stem");
            let obj = out_dir.join(format!("{name}_{stem}.obj"));

            let mut asm_cmd = Command::new(&ml64);
            for (key, value) in compiler.env() {
                asm_cmd.env(key, value);
            }
            asm_cmd.arg("/nologo").arg("/c");
            asm_cmd.arg(format!("/Fo{}", obj.display()));
            asm_cmd.arg(format!("/I{}", self.amd64_asm.display()));
            asm_cmd.arg("/DBUILD_MLAS_NO_ONNXRUNTIME");
            asm_cmd.arg("/D_USE_MATH_DEFINES");
            asm_cmd.arg("/DNDEBUG");
            asm_cmd.arg(&src);
            let status = asm_cmd.status().expect("run ml64.exe");
            assert!(status.success(), "ml64.exe failed for {}", src.display());

            objects.push(obj);
        }

        let mut b = self.base();
        for obj in objects {
            b.object(obj);
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

    fn compile_msvc_kleidiai_asm(&self, name: &str, files: &[&str]) {
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
            let src = self.kai.join(f);
            assert!(src.exists(), "missing KleidiAI asm: {}", src.display());
            let stem = src
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("asm file stem");
            let preprocessed = out_dir.join(format!("{stem}.kai.asm"));
            let obj = out_dir.join(format!("{stem}.kai.obj"));

            let status = Command::new("clang")
                .arg("-E")
                .arg("-P")
                .arg("-x")
                .arg("assembler-with-cpp")
                .arg("-D_MSC_VER=1930")
                .arg("-D_M_ARM64=1")
                .arg(&src)
                .arg("-o")
                .arg(&preprocessed)
                .status()
                .expect("run clang to preprocess KleidiAI ARM64 asm");
            assert!(
                status.success(),
                "clang failed to preprocess {}",
                src.display()
            );

            let mut asm_cmd = Command::new(&armasm64);
            for (key, value) in compiler.env() {
                asm_cmd.env(key, value);
            }
            asm_cmd.arg("-nologo").arg(&preprocessed).arg(&obj);
            let status = asm_cmd.status().expect("run armasm64.exe for KleidiAI");
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
