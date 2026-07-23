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
    includes: Vec<PathBuf>,
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let vendor = root.join("vendor/mlas/onnxruntime");
    let lib = vendor.join("core/mlas/lib");
    let asm = lib.join("x86_64");

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
        "qgemm_kernel_avx2.cpp",
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
    generic.push(p.root.join("vendor/shim.cpp"));
    generic.push(p.root.join("vendor/probe.cpp"));
    p.compile_cpp("mlas_generic", &[], &generic);

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

    println!("cargo:rustc-link-lib=stdc++");
}

impl Paths {
    fn base(&self) -> cc::Build {
        let mut b = cc::Build::new();
        b.cpp(true);
        b.std("c++17");
        b.define("BUILD_MLAS_NO_ONNXRUNTIME", None);
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
        for h in ["cstring", "cstdlib", "cstdint", "unistd.h"] {
            b.flag("-include").flag(h);
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
            let path = self.asm.join(f);
            assert!(path.exists(), "missing vendored asm: {}", path.display());
            b.file(path);
        }
        b.compile(name);
    }
}
