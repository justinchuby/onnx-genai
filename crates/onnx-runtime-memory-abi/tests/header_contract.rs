//! The public C header is part of the contract, so it is tested like one.
//!
//! Two things can silently drift apart here: the sizes recorded in the header
//! and the Rust definitions, and the header and a real C compiler's opinion of
//! it. Both are checked.
//!
//! Neither test is allowed to pass by doing nothing. The layout test fails if
//! it does not find every annotation it expects, and the compile test fails
//! loudly when no C compiler can be found rather than quietly skipping.

use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::process::Command;

use onnx_runtime_memory_abi::*;

fn header_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("include/nxmem_memory_abi.h")
}

fn example_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/minimal_plugin.c")
}

/// Every `NXMEM_LAYOUT: <name> size=<n>` annotation in the header.
fn annotated_sizes() -> Vec<(String, usize)> {
    let header = std::fs::read_to_string(header_path()).expect("the public header is readable");
    header
        .lines()
        .filter_map(|line| {
            let rest = line.split_once("NXMEM_LAYOUT:")?.1.trim();
            let (name, size) = rest.split_once("size=")?;
            let size = size
                .trim()
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("malformed layout annotation: {line}"));
            Some((name.trim().to_string(), size))
        })
        .collect()
}

/// The Rust size of every type the header annotates.
///
/// Adding a `#[repr(C)]` type to the header without adding it here fails the
/// test below, which is the point: the two lists must move together.
fn rust_sizes() -> Vec<(&'static str, usize)> {
    vec![
        ("NxmemStatus", size_of::<NxmemStatus>()),
        ("NxmemDeviceId", size_of::<NxmemDeviceId>()),
        ("NxmemAllocation", size_of::<NxmemAllocation>()),
        ("NxmemByteRange", size_of::<NxmemByteRange>()),
        ("NxmemAllocRequest", size_of::<NxmemAllocRequest>()),
        ("NxmemAllocResult", size_of::<NxmemAllocResult>()),
        ("NxmemRangeRequest", size_of::<NxmemRangeRequest>()),
        ("NxmemReleaseOutcome", size_of::<NxmemReleaseOutcome>()),
        (
            "NxmemReleaseCompletion",
            size_of::<NxmemReleaseCompletion>(),
        ),
        ("NxmemReclaimRequest", size_of::<NxmemReclaimRequest>()),
        ("NxmemUnloadReport", size_of::<NxmemUnloadReport>()),
        (
            "NxmemSharedPrefixHandle",
            size_of::<NxmemSharedPrefixHandle>(),
        ),
        (
            "NxmemSharedPrefixCommitRequest",
            size_of::<NxmemSharedPrefixCommitRequest>(),
        ),
        (
            "NxmemSharedPrefixCommitInfo",
            size_of::<NxmemSharedPrefixCommitInfo>(),
        ),
        ("NxmemHostCallbacks", size_of::<NxmemHostCallbacks>()),
        ("NxmemOpenRequest", size_of::<NxmemOpenRequest>()),
        (
            "NxmemVirtualBackingVtable",
            size_of::<NxmemVirtualBackingVtable>(),
        ),
        (
            "NxmemSharedMappingVtable",
            size_of::<NxmemSharedMappingVtable>(),
        ),
        ("NxmemAllocatorVtable", size_of::<NxmemAllocatorVtable>()),
        (
            "NxmemAllocatorFactoryVtable",
            size_of::<NxmemAllocatorFactoryVtable>(),
        ),
        ("NxmemVersionRange", size_of::<NxmemVersionRange>()),
        ("NxmemNegotiateRequest", size_of::<NxmemNegotiateRequest>()),
        (
            "NxmemNegotiateResponse",
            size_of::<NxmemNegotiateResponse>(),
        ),
    ]
}

/// The header's recorded sizes match the Rust definitions.
///
/// Sizes are only equal like this on a 64-bit target, where the header's
/// pointer-sized members and Rust's agree; on anything else the annotations
/// describe a different machine and comparing them would be meaningless.
#[test]
#[cfg(target_pointer_width = "64")]
fn header_layout_matches_the_rust_definitions() {
    let annotated = annotated_sizes();
    assert!(
        annotated.len() >= 20,
        "the header lost its layout annotations: found only {}",
        annotated.len()
    );

    for (name, rust_size) in rust_sizes() {
        let (_, header_size) = annotated
            .iter()
            .find(|(annotated_name, _)| annotated_name == name)
            .unwrap_or_else(|| panic!("the header has no NXMEM_LAYOUT annotation for {name}"));
        assert_eq!(
            *header_size, rust_size,
            "{name}: the header says {header_size} bytes, Rust says {rust_size}"
        );
    }

    // And nothing is annotated that this test does not know about, so the
    // header cannot grow a type that silently goes unchecked.
    for (name, _) in &annotated {
        assert!(
            rust_sizes().iter().any(|(known, _)| known == name),
            "the header annotates {name}, which this test does not check"
        );
    }
}

/// The first C compiler on this machine, if any.
fn find_cc() -> Option<PathBuf> {
    if let Ok(cc) = std::env::var("CC")
        && !cc.is_empty()
    {
        return Some(PathBuf::from(cc));
    }
    for candidate in ["cc", "clang", "gcc"] {
        if let Ok(output) = Command::new(candidate).arg("--version").output()
            && output.status.success()
        {
            return Some(PathBuf::from(candidate));
        }
    }
    None
}

/// A real C compiler accepts the header and the example plugin.
///
/// This is what backs the "one minimal plugin example that documents the
/// contract without requiring workspace linking" claim: the example is
/// compiled with nothing on the include path but the header itself.
#[test]
#[cfg(unix)]
fn nxmem_c_example_compiles() {
    let cc = find_cc().expect(
        "no C compiler found (tried $CC, cc, clang, gcc); \
         the public C header cannot be checked without one",
    );
    let out = std::env::temp_dir().join(format!("nxmem_minimal_plugin_{}.o", std::process::id()));
    let output = Command::new(&cc)
        .arg("-c")
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg("-fPIC")
        .arg("-I")
        .arg(header_path().parent().expect("include directory"))
        .arg(example_path())
        .arg("-o")
        .arg(&out)
        .output()
        .expect("the C compiler runs");

    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let _ = std::fs::remove_file(&out);
    assert!(
        output.status.success(),
        "the minimal C plugin must compile against the public header alone:\n{stderr}"
    );
}

/// A C compiler agrees with Rust about every annotated struct size.
///
/// The layout test above compares Rust against a *comment*. This one compares
/// Rust against the compiler that will actually build a plugin, which is the
/// opinion that decides whether a real plugin works.
#[test]
#[cfg(all(unix, target_pointer_width = "64"))]
fn a_c_compiler_agrees_with_the_rust_layouts() {
    let cc = find_cc().expect(
        "no C compiler found (tried $CC, cc, clang, gcc); \
         the public C header cannot be checked without one",
    );

    let mut source = String::from("#include \"nxmem_memory_abi.h\"\n");
    for (name, size) in rust_sizes() {
        source.push_str(&format!(
            "_Static_assert(sizeof({name}) == {size}, \"{name} disagrees with Rust\");\n"
        ));
    }
    source.push_str("int nxmem_layout_probe(void) { return 0; }\n");

    let dir = std::env::temp_dir().join(format!("nxmem_layout_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp directory");
    let probe = dir.join("probe.c");
    std::fs::write(&probe, source).expect("probe source is writable");

    let output = Command::new(&cc)
        .arg("-c")
        .arg("-std=c11")
        .arg("-I")
        .arg(header_path().parent().expect("include directory"))
        .arg(&probe)
        .arg("-o")
        .arg(dir.join("probe.o"))
        .output()
        .expect("the C compiler runs");

    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        output.status.success(),
        "a C compiler disagrees with the Rust struct layouts:\n{stderr}"
    );
}
