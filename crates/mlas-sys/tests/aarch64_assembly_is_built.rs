//! Every aarch64 kernel MSVC assembles is also assembled by everyone else.
//!
//! The vendored tree ships the same ARM64 kernels twice: `lib/arm64/*.asm` in
//! ARMASM syntax for MSVC, and `lib/aarch64/*.S` in GAS syntax for GNU as and
//! Apple's assembler. `build.rs` has to name both, and for a long time it named
//! only the first.
//!
//! The consequence was invisible here and expensive elsewhere. The dispatch
//! tables — `MlasConvSymS8DispatchNeon` and friends — are arrays of function
//! pointers in C++ that compiles perfectly well without the kernels; the
//! missing definitions only surface when something links a shared object, i.e.
//! in whichever *downstream* crate produces a cdylib, with a nineteen-symbol
//! `Undefined symbols for architecture arm64` wall that names no file in this
//! crate.
//!
//! Assembly cannot be cross-checked by compiling here (this test runs on x86-64
//! in CI), so this reads `build.rs` structurally instead: for every ARMASM file
//! in the MSVC group there must be a `.S` of the same stem in the non-MSVC
//! group, or an explicit exclusion below. That is the *shape* of the defect —
//! one dialect wired up, the other forgotten — rather than one instance of it.

use std::path::{Path, PathBuf};

/// Files whose GAS form is deliberately not assembled.
///
/// Each needs a per-file `-march` for an ISA extension, and nothing compiled
/// on this path references it. They are listed rather than silently dropped so
/// that adding a dispatcher that needs one is a decision and not a surprise.
const EXCLUDED: &[(&str, &str)] = &[(
    "HalfGemmKernelNeon",
    "fp16 arithmetic; needs -march=armv8.2-a+fp16, spelled differently by GNU \
     as and Apple's assembler, and nothing here references MlasHalfGemmKernelNeon",
)];

#[test]
fn every_msvc_arm64_kernel_has_a_gas_counterpart() {
    let build = read_build_rs();
    let msvc = group(&build, "compile_msvc_arm64_asm", ".asm");
    let gas = group(&build, "compile_aarch64_asm", ".S");

    assert!(
        msvc.len() > 10,
        "read only {} ARMASM files from build.rs, so the comparison below would \
         be vacuous",
        msvc.len()
    );
    assert!(
        gas.len() > 10,
        "read only {} GAS files from build.rs; the aarch64 group is how every \
         non-MSVC target gets these kernels",
        gas.len()
    );

    let missing: Vec<&String> = msvc
        .iter()
        .filter(|stem| !gas.contains(*stem))
        .filter(|stem| !EXCLUDED.iter().any(|(name, _)| *name == stem.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "these kernels are assembled for MSVC but not for GNU/Apple aarch64: \
         {missing:?}. Every one of them is an undefined symbol at link time on \
         macOS arm64 and Linux aarch64. Add the `.S` to `compile_aarch64_asm`, \
         or record why not in EXCLUDED."
    );

    let unexplained: Vec<&&str> = EXCLUDED
        .iter()
        .map(|(name, _)| name)
        .filter(|name| gas.contains(&(*name).to_string()))
        .collect();
    assert!(
        unexplained.is_empty(),
        "{unexplained:?} are assembled now but still listed as excluded; drop \
         them from EXCLUDED so the list keeps describing reality"
    );
}

/// Both dialects of every kernel named in `build.rs` are actually vendored.
///
/// A typo in either list is otherwise a link error on a platform this
/// workspace's CI reaches only through a cross-build.
#[test]
fn every_named_assembly_file_exists() {
    let build = read_build_rs();
    let lib =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/mlas/onnxruntime/core/mlas/lib");

    let msvc = group(&build, "compile_msvc_arm64_asm", ".asm");
    let gas = group(&build, "compile_aarch64_asm", ".S");
    assert!(
        !msvc.is_empty() && !gas.is_empty(),
        "read {} ARMASM and {} GAS files; empty lists would make the loops \
         below assert nothing",
        msvc.len(),
        gas.len()
    );

    for stem in msvc {
        let path = lib.join("arm64").join(format!("{stem}.asm"));
        assert!(
            path.exists(),
            "build.rs names a missing file: {}",
            path.display()
        );
    }
    for stem in gas {
        let path = lib.join("aarch64").join(format!("{stem}.S"));
        assert!(
            path.exists(),
            "build.rs names a missing file: {}",
            path.display()
        );
    }
}

fn read_build_rs() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("build.rs");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// File stems in the `&[...]` literal passed at the *call site* of `call`.
///
/// Anchored on `p.{call}(` rather than the bare name so the function's own
/// definition — whose body quotes other paths — is not read as its argument
/// list, and bounded at the first `]` so the list cannot run on into the next
/// statement.
fn group(build: &str, call: &str, extension: &str) -> Vec<String> {
    let after = build
        .split(&format!("p.{call}("))
        .nth(1)
        .unwrap_or_else(|| panic!("build.rs calls p.{call}(...)"));
    let list = after
        .split_once("&[")
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(list, _)| list)
        .unwrap_or_else(|| panic!("p.{call}(...) passes a slice literal"));
    list.lines()
        .map(|line| line.trim().trim_end_matches(','))
        .filter_map(|line| line.strip_prefix('"')?.strip_suffix('"'))
        .filter_map(|name| name.strip_suffix(extension))
        .map(str::to_string)
        .collect()
}
